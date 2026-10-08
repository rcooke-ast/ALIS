# Dashboard Stage 4 log

The Data and Regions tabs (steps 1–3): the spectrum view, the systems and datasets,
the transitions and line IDs, snips, regions and the continuum. The plan is in
`claude_prompts/dashboard_stage4.md`.

### 2026-10-07 (Prompt 1: reading, and queries Q4.13–Q4.16)

**Reading.** The code plan, the dashboard prompts (D1–D51, the QF queries), the Stage
0–3 documents and logs, this stage's document with RJC's responses to Q4.1–Q4.12, and
the code Stage 4 builds on: `alis/dashboard/` (`project.py`, `modes.py`, `edit.py`,
`validate.py`, the module list of the rest, `qt/tabs.py`), `alis/load.py` (how a data
line's fitted and loaded pixels and its buffer are found; `find_shared_pixels`), the
continuum functions (`legendre`, `chebyshev`, `polynomial`, `constant`, `spline`),
`alis/prepfit/specplot.py` (its solar abundances), and the shared continua of
`DH_orders/Q1243p307`.

**Facts that shaped the queries.**
- ALIS divides by the error in χ² (`model_eval.myfunct`), so a pixel with a NaN or an
  error ≤ 0 can never be fitted; this is not outlier masking (Q4.11).
- A data line with `fitrange=columns` and no fitted pixel stops ALIS's `load_data`
  (`np.min` of an empty array), so a snip with no region yet makes the model
  unrunnable until a region is drawn (Q4.2(a)). The validator's full check will need
  to catch this before ALIS does, rather than report an unexpected error.
- Only `legendre` takes `min=`/`max=`; `chebyshev` takes its range from the pixels,
  and `polynomial` uses raw wavelengths. So only a Legendre can be shared (Q4.8).
  Legendre stops at order 10, Chebyshev at 9.
- ALIS's `spline` function is a cubic spline through `locations=` (wavelengths) with
  the flux at each location a parameter (at least four), which fits a continuum held
  through knots in the fit itself (Q4.13(c)).
- The default atomic table has about 830 transitions.
- In the bundle, a bad pixel of the source becomes a 0 in the snip's fit mask, like
  any pixel outside the regions, so it is not known from the snip alone (Q4.15).

**The responses.** Most are clear and are recorded as "Notes on the responses" in the
document: SNIP's ±300 km/s is a starting extent with free handles (Q4.2); specids are
added automatically, said in the Undo entry (Q4.3); line IDs with an "all lines"
switch and a count of those left out (Q4.4); no forest, quasar-redshift or telluric
flags (Q4.5); ±15 km/s around the model's lines (Q4.6); sharing only between file rows
for one transition, Legendre only (Q4.8); a confirmation for every merge (Q4.9); no
automatic masking of outliers, the bad-pixel mask respected (Q4.11); no model drawn on
the Regions tab (Q4.12).

**New queries,** each with a lean:
- **Q4.13:** continuum knots: through them in the starting model only, or in the fit
  too (links on the Legendre, or ALIS's `spline`).
- **Q4.14:** the two meanings of "structure" behind Q4.10: a spectrum's column roles
  (RJC's) and an imported fit's inferred systems and rows (Claude's), with what each
  allows.
- **Q4.15:** where a snip's bad pixels are kept, so that they are respected without
  the source.
- **Q4.16:** whether the first guess of a new snip also leaves out the transitions of
  the systems' line IDs, since the model has no absorption lines yet.

**`deferred_work.md`:** a new §7 (the dashboard) with the telluric flags (Q4.5(b))
and the best-fit model on the Regions tab (Q4.12).

**Baseline.** `pytest -m unit`: 1730 passed, 120 skipped, 0 failed (89 s);
`pytest -m gui`: 65 passed (25 s). The same as at the close of Stage 3.

**Found:** `alis/dashboard/qt/` (Stage 3's windows) and `doc/dashboard/skeleton/` are
not tracked by git, although the tests of the windows are committed.

### 2026-10-07 (Prompt 2: the responses to Q4.13–Q4.16, and Q4.17)

**The responses.**
- **Q4.13:** (a). Knots only set the starting coefficients (a constrained
  least-squares fit through them); they are never part of the fit.
- **Q4.14:** both, as proposed: a row's column roles can be changed while it has no
  snips; an imported fit's systems are edited (z, name, primary, add, remove), its rows
  only renamed, then confirmed. Datasets are never merged.
- **Q4.15:** neither (a) nor (b): the bad-pixel mask is stored in the snip file itself,
  and a snip without one has all its pixels good.
- **Q4.16:** yes; the first guess also leaves out ±15 km/s around the systems' line
  IDs, and the knots adjust it by hand.

**Checked for Q4.15.** `load_ascii` reads each column with `np.loadtxt(usecols=...)`,
by the numbers in `columns=`, so a fifth column is ignored unless named, and
`load_data` stops on a role not in its `colallow` list. Labelling the column is
therefore a choice between a new ALIS role in `columns=` (a change to ALIS) and a
comment line in the file (no change to ALIS). Asked as **Q4.17**, with the lean of a new
role, `badpix`, that ALIS honours.

### 2026-10-07 (Prompt 3: Q4.17 answered; the Design and Tasks updated; Tasks 4.1–4.4)

**Q4.17:** (a), the `badpix` column as a role ALIS honours. The Design section and the
tasks were brought in line with Q4.1–Q4.17 (no forest or telluric flags; no automatic
masking of outliers; the `badpix` column; knots for the starting continuum only;
sharing one transition across rows; merges confirmed; rows renamed, never merged).

**4.1 Transitions and line IDs (`alis/dashboard/lines.py`).**
- `Atomic`: every transition of a chemical element in the atomic table (725 lines),
  isotopes folded into their element (`main_ion`: `("O", "I")` → `16O_I`; D I is not
  a main ion), with Lyman names (`Lyβ 1025.7`, `Ly7 926.2`).
- `Coverage`: one file row's pixels (its source, or its snips) and the runs of good
  pixels (finite, positive error, not bad), split where a step exceeds 5 median steps.
  `place` gives "gap" (no good pixels) or "edge" (closer than a snip's half-width to
  an end of its run).
- `transitions`/`ion_transitions`: one ion's transitions on the data and any with a
  snip, ranked by fλ, lines closer than the resolution in one row (`O I 971.7 ×3`).
  `elements_on_data` offers only elements and stages with a transition on the data.
- `line_ids`: the system's ions (model and line IDs) in full, strong lines by
  log10(fλ) + log ε − 12 ≥ −2 (Asplund 2009, the table `prepfit` uses; a preference),
  or every line; with the count left unlabelled (Q4.4). Kinds: hydrogen, metal, other.
- `identify`/`candidate_menus`: the transitions with z ≥ 0 for a click, the first
  choice the strongest whose Lyα is within the data; menus Element → Stage →
  Transition, H first.
- `snip_transitions`: each snip's transition, from SNIP's record in `ui/project.json`,
  or from its file name (`_H_I_923.2_`; a whole-number name such as `_O_I_988_` means
  the strongest line of that Å), or the system line nearest its middle.
- `coverage_grid`: ■ snip, □ covered by the source, · not covered.
- Tests (`tests/test_dashboard_lines.py`, 18): J1358p6522's H I list (Lyα "gap" and
  snipped, Lyβ–Ly8 snipped, Ly9 and Ly10 not, no forest flag, the whole series to its
  limit); O I 971.7 ×3; gap and edge on synthetic spectra; bad pixels and errors ≤ 0;
  a click on Lyβ offers z = 3.06726; Q1243p307's grid, 16 transitions × 3 files (16,
  16, 12 snips); line IDs counting what they leave out.

**4.2 Snips (`alis/dashboard/snips.py`) and ALIS's `badpix` column.**
- **ALIS:** `ColumnMap`, `ColumnPosition` and `DataOpt` gain `badpix`; `load_data`
  accepts it in `columns=`, reads it with a new `load.load_badpix` (ascii, FITS, memory
  or an array; the readers' signatures are unchanged), never fits a bad pixel whatever
  the fit range says, stops with a message when every fitted pixel is bad, and records
  the mask; `save_asciifits`/`save_fitsfits` write it back. Tests in
  `tests/test_load_files.py` (7 new). No model uses the role, so the harness is
  unaffected.
- **Snips:** `snip_step` (SNIP: every row whose source has pixels there, ±300 km/s,
  four columns plus `badpix` when the source has a mask, no region, the row's
  resolution and shift or, for its first snip, its FWHM fixed and a free shift unless
  it is the reference, the first-guess continuum, the specid added to absorption lines
  with a transition in it and said so in the step's name, the row's zero level, and
  the records in `ui/project.json`); `clear_plan` (CLEAR, which also removes a section
  it leaves empty, so SNIP then CLEAR gives back the text); regions added, moved,
  removed and masked by hand, never fitting an unfittable pixel; `edges_step` (inwards
  keeps lines; outwards keeps the old lines byte for byte and adds the source's at full
  precision; a Legendre's `min=`/`max=` follow with its coefficients re-expressed);
  `copy_regions_step` (D18, moved by the rows' shifts); `buffers`; `shared_pixels`,
  `keep_step`, `merge_plan` (a confirmation listing what changes, Q4.9).
- `project.py`: `SnipData.bad` and `fittable`; `set_regions` never fits an unfittable
  pixel; a `constant` can be a snip's continuum (after any polynomial). `validate.py`:
  a snip with no region says "draw one on the Regions tab".
- Tests (`tests/test_dashboard_snips.py`, 17).

**4.3 The continuum (`alis/dashboard/continuum.py`).**
- Every curve and fit basis is ALIS's own function (`call_CPU`, with one coefficient
  at a time for a basis), on the snip's wavelengths moved by its row's `vshift`. Range:
  `min=`/`max=`, else ALIS's own sub-pixel range from its loaders (cached), else the
  snip's extent. Dashboard continua are written with power-of-ten `scale=` factors and,
  for a Legendre, `min=`/`max=`, in one fixed layout (so + then − gives back the text).
- The continuum pixels are chosen once, by a fit at the highest order allowed, clipped
  2.5σ below and 3σ above, leaving out ±15 km/s around the model's lines and the
  systems' line IDs (Q4.16). Every order is fitted to them; the BIC chooses (up to 5).
  A first try clipped each order separately: an order too low for the curve clipped
  away most of the pixels (62 of 624), and the comparison collapsed.
- Knots (Q4.13): equality constraints on the least-squares fit; a knot beyond n + 1
  raises the order. Sharing (Q4.8): one Legendre for one transition in several rows.
- Tests (`tests/test_dashboard_continuum.py`, 27): the curve equals ALIS's own
  evaluation to 1e-12 for every emission line of the 16 `examples/` fit models; the
  first guess recovers an order-2 Legendre under three absorption lines (to 3 in
  1000, noise 5) and the BIC picks 2; + then − gives back the text; a refit passes
  through four knots; function changes; a hidden continuum stays hidden; sharing and
  unsharing read in ALIS.

**4.4 Datasets and systems (`alis/dashboard/datasets.py`).**
- The reference (its shift removed, the old reference's freed or added), FWHM and
  shift values and badges (free, fixed, tied, Q0.8), the zero level on and off (also
  before the first snip, which SNIP then gives it), a new file as a row (its source in
  the bundle), renaming, column roles while a row has no snips (Q4.14); systems: z
  (which moves nothing), add, rename, make primary. `Step` gains `sources`, so that a
  new file row is undone exactly; `project.structure()` keeps what only
  `ui/project.json` says of a row.
- Tests (`tests/test_dashboard_datasets.py`, 11): each is one step that ALIS reads,
  undone and redone byte for byte (text, files, project data, sources).

**Batches:** `unit` 1810 passed, 0 failed (83 s); `gui` 65 passed.

**4.5 The spectrum view (`alis/dashboard/qt/plots.py`).**
- `SpectrumView` (a pyqtgraph `PlotWidget`): the data as steps centred on the pixels,
  built by hand (pyqtgraph 0.14 cannot clip or downsample a curve in its own step
  mode) and broken at gaps; the error thin and grey, not drawn where it is 0 or less
  (J1358p6522's file holds −6×10³¹ there); bad pixels as crosses; the continuum
  dashed; the zero line; Flux | Log; the normalised view. Its home flux range is
  robust (0 to 1.25 × the 99.7th percentile), so a wild pixel does not stretch it.
- Overlays: fit regions (movable), regions of another snip, pixels fitted twice
  (hatched), the snip's edges (orange handles), knots (draggable), line IDs (colour
  and line style by kind, three rows, no overlaps, "+ n lines not labelled"), a
  marker; and pieces (a row's snips, each scaled to its peak, D35).
- `SpectrumBox`: the wheel zooms the wavelength axis only; modes pan, zoom box, draw,
  mask and click. `VelocityAxis`: signed velocity ticks above (D43). `ZoomBar`:
  ⌂ ← → ✥ ⬚ + −, its own history, for one view or several linked ones.
- Tests (`tests/test_dashboard_qt_plots.py`, 11, `gui`): steps and gaps; + and −
  along the wavelength axis only; back, forward and home; the zoom box; the wheel;
  J1358p6522's 40,402 pixels drawn in under 0.5 s off-screen (0.2 s here); signed
  velocity ticks; log and normalised views; overlays and their signals; no number in
  the tooltips.

**4.6 The Data tab (`alis/dashboard/qt/data.py`).**
- Systems (the primary z typed; Identify a feature with its panel; the other systems
  with Rename…, Make primary, Set z…, Remove…; the generic absorbers; Add system…;
  Confirm for an imported fit), Coverage, Blinding, the Datasets table (reference,
  contents, FWHM and shift with badges and typed values, zero level, checksum and
  Relink…, Add file…, Make reference, Rename…, Column roles…, Remove…), the Spectrum
  (the source, or the snips scaled; line IDs of the primary system; the zoom bar),
  and cross-highlighting both ways. What the cells say is decided in `datasets.py`
  (`contents`, `checksum`, `system_lines`, `generic_summary`), without Qt.
- New dialogs (`qt/dialogs.py`): `PlanDialog` (what a removal or merge changes, line by
  line, through the gate, S27, Q4.9) and `AskDialog` (a name, a redshift, a value).
- The window: each tab is given the window (`Tab.attach`); `Tab` moved to
  `qt/widgets.py` (so the tabs' own modules import it without a cycle); the imported
  fit's banner clears once the structure is kept.
- Found while building it, and fixed:
  - **an imported fit's rows merged on a tie:** rows are guessed from shared
    resolution labels, so tying KIRKMAN's FWHM to PROCHASKA's made them one row.
    `datasets.keep_structure` writes the guessed structure into `ui/project.json`
    with the first edit of a row or system (as Confirm does); asked as Q4.18(c);
  - **Identify a feature's first choice** was Lyα for a click on J1358p6522's Lyβ
    trough; it is now the transition nearest a known system (Lyβ, z = 3.06726), as
    the mockup shows, and otherwise the strongest whose Lyα is on the data;
  - **a narrow Systems pane** (the `.mod` panel open) clipped the z and Identify
    row: it stacks them;
  - `examples/blind`'s snip holds a flux of −9.6×10⁹: the robust range covers the
    snips drawn as pieces too.
- New preferences (`preferences.py`): the snip's half-width, spectra before
  scrolling, the overlap from which merging is offered first, the continuum's highest
  first-guess order, its clipping and the width left out around known lines, and the
  line IDs' strength.
- Tests (`tests/test_dashboard_qt_data.py`, 13, `gui`): Identify a feature sets the
  primary z on a new project from `OI_SiII.dat` (and undo takes it back); typing z and
  adding a system; the first choice near a known system; the table of a new project;
  Add file, the reference and the zero level; Remove… shows its plan first; column
  roles fixed once a row has snips; on Q1243p307 a tie reaches the model and one undo
  removes it; Confirm clears the "!" and the banner; on `examples/blind` (given
  distinctive hidden values) no hidden value in the tab; no design reference;
  line IDs and their count; cross-highlighting both ways.

**4.7 The review page.** `doc/dashboard/stage4/take_screenshots.py` and
`build_review.py` (in the series of Stage 3's page, with the mockups' stylesheet):
the new project from J1358p6522's spectrum with Identify a feature open beside
`data_new`; Q1243p307 beside `data_datasets`; and, with no mockup, Q1243p307 after a
tie with the `.mod` panel open, J1358p6522 imported, `examples/blind`, Add file… and
Remove…. Published as a new private page,
https://claude.ai/artifact/T4v5mJk179rDAqCjU1GJNe, with Q4.18 (what to decide).
The stage pauses here for RJC's review (Q4.1).

**Not yet done (Tasks 4.8–4.13):** the Regions tab (its transition list, spectra,
tools, Continuum and Snip boxes), the tab-scoped actions of the registry (the Regions
tab's keys and Identify a feature), the end-to-end test, and the stage's close
(documents, coverage, the `gui-component` skill).

**Batches:** `unit` 1810 passed, 0 failed; `gui` 89 passed; `fast` 111 passed, 0
failed (13 min 32 s), run now because `load_data` changed (the `badpix` column). black,
isort and ruff are clean on every file changed.

### 2026-10-07 (Prompt 4: the Data tab's review applied; Tasks 4.8–4.13)

**RJC's review of the Data tab (Q4.18)**, applied with no further review, as asked:
- the preview of a removal or a merge (`dialogs.PlanDialog`) sizes each column to its
  contents and scrolls sideways;
- an imported fit's structure, written at its first edit so that a tie never merges
  two rows, is marked `guessed` in `ui/project.json` (`project.GUESSED`,
  `working_structure`); the Data tab's "!" and the banner stay until Confirm, which
  runs the full check again (`panel.check_now`); the marker's tooltip says what to do;
- the solar abundances in `atomic.ecsv` are `deferred_work.md` §7.3; the dashboard
  uses Asplund et al. (2009), as `prepfit` does.

**The registry's tab scope.** `actions.Action.scope`, the groups `DATA_TAB` and
`REGIONS_TAB` (`TAB_GROUPS`), and `actions.scoped(tab)`. The window builds a tab's
actions on the tab (`window._build_scoped`) with `WidgetWithChildrenShortcut`, so a
single key acts only while that tab has the focus and never while typing in the
`.mod` panel; each is bound to the tab's `do_<what>` and placed by the tab's
`bind_action` (`widgets.Tab`). 15 tab actions: Identify a feature
(`data.identify`), and the Regions tab's keys ↑ ↓ [ ] − + and Delete (Remove region),
SNIP, CLEAR, Draw region, Mask pixels, Add knot, Normalised and Auto first guess.
`tests/test_dashboard_actions.py` and `test_dashboard_qt_actions.py` check the
groups, the scope, and that every tab action is placed on its tab.

**4.8–4.11 The Regions tab (`qt/regions.py`, `RegionsTab`).**
- Transitions: System (the primary first), Element with ‹ › and the ion stage, the
  ion's transitions strongest first with λobs (masked when the system's z is hidden),
  an fλ bar and flags, ● for a snip. Without a source, only the snipped transitions,
  elements and stages are offered and SNIP is off. The Key, one item per line, drawn
  as the spectrum draws it.
- The spectra: one per dataset (`snips.strips`), stacked, scrolling beyond
  `regions.spectra`, the velocity axis named once above and the wavelength once below
  (`SpectrumView.set_labels`), linked in wavelength; a click selects a dataset (and
  its lines in the `.mod` panel). The snip's pixels, its source around it (grey), the
  continuum (ALIS's own curve), the fit regions, the other snips' regions, the pixels
  fitted twice (hatched), the edges, the knots and the line IDs. The zoom is kept
  across steps while the transition stays the same. The tab redraws only while it is
  shown (`_stale`, `_tab_shown`), and takes the transition the Data tab's coverage
  grid sets in `session.view["regions"]`.
- Tools: Normalised; Draw region (on a transition with no snip it SNIPs first; the
  region is copied to the datasets with none, one step); Mask pixels; Add knot (a
  click adds one, a click on one removes it, a drag moves it; in Normalised, in its
  frame); SNIP (enabled while a covering dataset lacks the snip); CLEAR (previewed);
  the edges (outwards re-cut from the source); a region's edges in Draw mode;
  right-click Remove this region / Copy to the datasets with none; Delete.
- Continuum: function, order − n + (and − + = keys), Auto first guess, the table of
  orders n ± 1 (χ², Δχ², ΔBIC, against the same continuum pixels), knots and Clear
  knots, Share with <row> / Unshare (Legendre, one transition), and a note for a
  hidden continuum.
- Snip: extent and pixels, fitted pixels and regions, the buffers in Å against
  ALIS's need (✓ or !); pixels fitted twice with Keep and Merge…, the one offered
  first leading (`regions.merge_overlap`), and Apply the fix to all N datasets
  (`same_pairs`, `keep_all_step`, `merge_all_plan`; the merge previewed); with several
  datasets, the table of fitted and shared pixels and the regions' origin, and Copy
  <row> regions to all.
- `snips.set_mask_step` records "copied from" on an imported snip too (its record is
  made then), so the table says so for Q1243p307.
- Markers: the Regions tab's "!" for a buffer narrower than the resolution needs
  (`markers._narrow_buffers`, as the validator warns); Q1243p307's Ly6k is one.
- Found while taking the screenshots and fixed: the tools elided with the `.mod`
  panel open (now a row of their own); the Continuum row cramped; the System menu
  repeated z; stage buttons offered with nothing to list on an imported fit; pyqtgraph
  scaled the flux axis to "(x0.001)" (`enableAutoSIPrefix(False)`); strips too short
  for three datasets (compact tables, axis titles once).
- Tests (`tests/test_dashboard_qt_regions.py`, 24, `gui`): J1358p6522's H I list
  from its spectrum, and SNIP's ●; the keys only with the tab's focus, none while
  typing in the `.mod` panel; without a source, only snipped transitions and no SNIP;
  the key and the tools; no design reference; a drawn region in the mask column and
  undo restoring the file, Mask pixels splitting it; Draw region SNIPping first, one
  step; SNIP then CLEAR; Normalised; the zoom kept; two datasets with the region
  copied on drawing; Q1243p307's regions copied from HIRES (moved by KIRKMAN's shift)
  and undone; J1358p6522's edge re-cut from the source; right-click and Delete; + then
  − giving back the text; a knot added, moved and removed, one step each; a hidden
  continuum never shown, before and after a refit; sharing and unsharing; the
  function, Auto first guess and a region's edges dragged; J1358p6522's Ly7 with its
  36 shared pixels, keep, and the "!" gone once all overlaps are kept; Q1243p307's
  merge offered first and made in three datasets; the Snip box of a new snip;
  cross-highlighting both ways; the coverage grid opening a transition.

**4.12 End to end** (`tests/test_dashboard_qt_end_to_end.py`, `fast`): a new project
from `OI_SiII.dat` (z = 0, its fourth column ignored), SNIP O I 1302 and Si II 1304,
the region 1301–1305 Å drawn on each, merged, Auto first guess, the two voigt lines
added by `edit.add_ion` (b tied, T fixed), saved and fitted as `run_alis
project.model` does: O I 13.984 ± 0.015 and Si II 13.044 ± 0.049, against the
reference's 13.985 ± 0.016 and 13.041 ± 0.050 (χ² 358.4 against 357.9). 4 s.

**4.13 Closing.**
- The review page: `take_screenshots.py` takes the Regions tab too (`OUTDIR
  regions` for those alone), and `build_review_regions.py` builds the page: J1358p6522
  imported beside `regions`, Q1243p307 beside `regions_datasets`, and a new project's
  Lyβ, Normalised, the merge's preview, CLEAR's preview, a hidden continuum, and the
  Data tab as it is now. Published as a new private page,
  https://claude.ai/artifact/HPnK8zEaFdACVwcGSKx2aT, with Q4.19 (what to decide).
- `doc/ALIS_workflow.md`: "Preparing a fit: the Data and Regions tabs" (§4.3), and
  the `badpix` column in the data keywords. `CHANGELOG.md`, `tests/README.md`, and
  the `gui-component` skill (tab-scoped actions, refreshing only when shown, and the
  spectrum view).
- Coverage of the new modules over their tests: 89% (see the Status).
- The stage document: Tasks 4.8–4.12 done, Q4.19, the Status and "What Stage 5
  receives". The stage table of `dashboard_stage0.md` is unchanged: nothing moved.

**Batches:** `unit` 1811 passed, 1 failed
(`test_atomic_mass.py::test_every_value_in_the_xml_survived_the_conversion`, from an
uncommitted edit to `alis/data/atomic.ecsv` that Claude did not make: four Balmer
rows' `SolarAbundance` changed from nan to 0.0; RJC has since reverted it); `gui` 115
passed; `fast` 112 passed, 0 failed (13 min 36 s). black, isort and ruff are clean on
every dashboard file and test changed.

### 2026-10-07 (Prompt 5: RJC's review of the Regions tab, draft 2)

Prompt 5 was written in `dashboard_stage3.md`'s Prompts (the document open in the
IDE); its content is Stage 4's review of the Regions tab, so it is done here. RJC also
reverted the edit to `alis/data/atomic.ecsv` that failed `test_atomic_mass.py` in
Prompt 4. The comments on draft 1 (7 threads) are recorded under Q4.19.

- **Ion ▾** with ‹ › (below Element) replaces the stage buttons (`ion_box`,
  `_step_ion`).
- **The tools split** by what they act on: SNIP, CLEAR and "Fit regions: Add region,
  Exclude pixels" above the spectra; Auto first guess, Add knot, Clear knots and
  Normalised in the Continuum box (`TOOL_BUTTONS`, `CONTINUUM_BUTTONS`,
  `_order_continuum_tools`). The registry's texts: "Add region" and "Exclude pixels"
  (was "Draw region" and "Mask pixels"), with tips saying what each is for.
- **One set of regions per transition** (`snips.linked_snips`, `linked_step`,
  `linked_name`): a region added, moved or removed on one dataset is added, moved or
  removed on every dataset, moved by the difference of the shifts, in one step;
  moving an edge adds or takes out only the pixels between the old edge and the new,
  so pixels excluded in one dataset stay excluded; removing a region removes every
  region it overlaps in each dataset ("drop"). Exclude pixels acts on one dataset.
  "Copy HIRES regions to all" is "Use HIRES's regions in every dataset" (an imported
  fit's datasets have regions of their own). `linked_step` keeps a dataset's
  "copied from" when an edit only adds to regions it had.
- **The continuum's first guess** (`continuum.continuum_pixels`): three passes of
  rising order from a straight line (`FIRST_ORDER`, `PASSES`), each started from the
  pixels the last kept (`fit(..., start=)`) and judging every usable pixel again,
  rejecting those 2.5σ below or 3σ above its curve; every order is then fitted to the
  pixels left, and the BIC chooses. Strictly cumulative passes failed the unit test of
  a known order-2 continuum by 45σ (a straight line throws away the curved ends), and
  one re-judgement per pass by 5 (≈1σ); converging each pass passes it. J1358p6522's
  Ly7 table: χ² 494.0, 492.2, 476.6 at orders 3–5 (was 2515.8, 2045.0, 540.3). The
  end-to-end fit still agrees with its reference.
- SNIP's half-width stays ±300 km/s (Q4.19(b)).
- **Q4.20** (a stacked spectrum of a transition, RJC's question): proposed for a later
  stage, in `deferred_work.md` §7.4; the thread on the page stays open for RJC.
- Tests: `test_dashboard_continuum.py` (the passes keep a curved continuum's ends
  and leave out the lines); `test_dashboard_snips.py` (regions the same in every
  dataset, an exclusion kept, a region dropped in both, undone byte for byte);
  `test_dashboard_qt_regions.py` (Q1243p307: HIRES's regions used in every dataset,
  a region removed on HIRES and one added on PROCHASKA reaching all three, undone;
  the function, Auto first guess and an edge moved); the ion chosen through its menu
  in the tests and the screenshot script.
- The review page, draft 2 (same address,
  https://claude.ai/artifact/HPnK8zEaFdACVwcGSKx2aT): "What changed since draft 1",
  and To decide (anything else; Exclude pixels on one dataset; Q4.20). Each thread
  answered; six resolved, the one asking about the stacked spectrum left open.
- Documents: the stage document's Regions tab design (Ion, the tools, linked regions,
  Exclude pixels, the first guess), Q4.19's responses and what was done, Q4.20, Task
  4.13 and the Status; `doc/ALIS_workflow.md`, `CHANGELOG.md`, `tests/README.md`.

**Batches:** `unit` 1814 passed, 0 failed; `gui` 115 passed; `fast` 112 passed, 0
failed (13 min 37 s). black, isort and ruff are clean on every file changed.

### 2026-10-08 (Prompt 5 of this document: draft 2's comments; the stage closed; Stage 5's document)

RJC's comments on draft 2 (recorded as Q4.20's response and Q4.21), applied with no
further review, as asked:
- **Two region tools, as in `prepfit`:** *Add regions to all* (`regions.draw`) and
  *Tweak dataset regions* (`regions.tweak`, replacing Exclude pixels). In both a drag
  adds a fit region and a right-drag leaves pixels out (`plots.SpectrumBox`: in draw
  mode a right drag gives a mask span); the first acts on every dataset of the
  transition, the second on the selected dataset alone (`snips.linked_step(...,
  linked=False)`). Moving a region, Delete, and the right-click menu follow the tool
  on; the menu offers "Remove this region from every dataset" / "… from this dataset
  only", and, while tweaking, "Exclude this pixel". The step that leaves pixels out
  is named "Exclude n pixels of …".
- **The stacked spectrum** (Q4.20, "Agreed"): placed with Orders mode and PypeIt
  spec1d loading, after v1, in D32 and "After v1" of `ALIS_v2_dashboard_prompts.md`,
  the stage table and "Also later" of `dashboard_stage0.md`, and `deferred_work.md`
  §7.4.
- Tests: `test_dashboard_qt_plots.py` (a right drag in draw mode masks, through the
  drag handler); `test_dashboard_qt_regions.py` (Add regions to all and Tweak dataset
  regions on two datasets, the menu while tweaking; the tool renamed in the others);
  `test_dashboard_snips.py` (the step's new name).
- The review page shows the final state (same address), with "What changed since
  draft 2"; the two new threads and the stacked-spectrum thread answered and resolved.

**The stage closed:** Task 4.13 done; the Status says so; Stage 4's decisions are
D52–D59 in `ALIS_v2_dashboard_prompts.md` (the badpix column, SNIP and CLEAR, the
Regions tab's controls, one set of regions per transition, the continuum's first
guess, merges, an imported fit's structure, line IDs); "What Stage 5 receives" gains
`linked_snips` and how to evaluate the model through `alis.model_eval`. The stage
table of `dashboard_stage0.md` is unchanged for Stages 4–6.

**Stage 5's document**, `claude_prompts/dashboard_stage5.md` (the Components tab):
the design (the ion navigator, the panels per transition and dataset with the
normalised model and residuals, adding components with an AOD first guess, dragging
with a live preview evaluated by ALIS, the cards, links and interlopers, removal,
imported models' notices, cross-highlighting), Tasks 5.1–5.11 (two new modules,
`profile.py` and `components.py`; a review at 5.7; the end-to-end fit through the tab
at 5.10), and Queries Q5.1–Q5.11, each with a lean.

**Batches at the close:** `unit` 1814 passed, 0 failed; `gui` 117 passed; `fast` 112
passed, 0 failed (13 min 25 s). black, isort and ruff are clean on every file changed.
