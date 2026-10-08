# Prompt file for ALIS software dashboard creation -- STAGE 4

> **The Data and Regions tabs (steps 1–3).** The first two tabs of the skeleton are
> filled in, and the dashboard draws its first plots (pyqtgraph). At the end of the
> stage, a user can take a new project from its spectra to snips with fit regions and
> continua, without typing in the `.mod` panel:
> - **Data (step 1):** see each file and its spectrum; set and refine the redshifts,
>   typed or with "Identify a feature" (S24, D16); add and remove systems (D15);
>   edit the datasets: reference, FWHM, shift, zero level, and ties between files
>   (S25, D14, D35); add and remove files (S3); check and confirm an imported fit's
>   structure (Q2.4, Q3.16(c)).
> - **Regions (steps 2–3):** choose transitions from a list ranked by strength (S2),
>   with line identifications (S5); make snips with SNIP and remove them with CLEAR
>   (Q0.6); draw fit regions, mask pixels and move the snip's edges (S6, S7); set
>   each continuum from an automatic first guess, by its order and by dragging (S8,
>   D17), shared between snips where wanted (S10); see the normalised view (S9); see
>   and fix pixels fitted twice (S26, D19); copy regions between datasets (D18).
>
> Everything the tabs show comes from the project model, through the blinding gate,
> and every action is one step of the history, as in Stage 3. The logic needs no Qt
> and goes in `alis/dashboard/` (three new modules: `lines.py`, `snips.py` and
> `continuum.py`), tested with pytest alone; the windows go in `alis/dashboard/qt/`.
> ALIS outside `alis/dashboard/` does not change, unless a query agrees otherwise.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`. "Q0.*n*" to "Q3.*n*" refer to
> `dashboard_stage0.md` to `dashboard_stage3.md`. The plan for all stages is in
> `dashboard_stage0.md`.

## Design

*Written by Claude on 2026-10-06, from the design documents, the mockups of the Data
and Regions tabs (`doc/dashboard/mockups/data_new.html`, `data_datasets.html`,
`regions.html`, `regions_datasets.html`) and what Stages 0–3 taught. The open
choices are the Queries below; each gives Claude's lean, and this section follows the
leans. Updated in Prompt 3 to follow RJC's responses to Q4.1–Q4.17: SNIP's extent is
a starting value (Q4.2); no forest or telluric flags (Q4.5); every line can be
labelled (Q4.4); no automatic masking of outliers, and the bad-pixel mask carried in
the snip as a `badpix` column that ALIS honours (Q4.11, Q4.15, Q4.17); knots set the
starting continuum (Q4.7, Q4.13); a continuum is shared only by one transition in
several rows (Q4.8); a merge is always confirmed (Q4.9); an imported fit's systems
are edited and its rows renamed, never merged, and a row's column roles can be
changed while it has no snips (Q4.10, Q4.14).*

### What a user can do at the end of this stage

1. **New project** (Stage 3): the spectra, their FWHM and column roles, and the
   primary redshift.
2. **Data:** the spectrum is shown. The redshift is typed, or found by clicking a
   feature and choosing its transition; other systems are added the same way. The
   datasets table sets the reference file, each file's FWHM, shift and zero level,
   and ties between files.
3. **Regions:** for each system, element and ion, the transitions that fall on the
   data are listed, strongest first. Clicking one shows it in every dataset. SNIP
   cuts its snips; fit regions are drawn, pixels masked, edges moved; the continuum
   starts from an automatic guess, and its order is changed with − and +. Pixels
   fitted twice are shown, with the fixes.
4. **Components** (Stage 5) and **Fit** (Stage 6) follow. Until then, the absorption
   lines are typed in the `.mod` panel, and `run_alis project.model` fits.

### Layers

```
alis/dashboard/qt/   plots.py    the spectrum view: pyqtgraph, its zoom toolbar and
                                 its overlays, shared by Data, Regions (and Fit later)
                     data.py     the Data tab's panels
                     regions.py  the Regions tab's panels
alis/dashboard/      lines.py    transitions, line IDs, flags and coverage
                     snips.py    cutting, extending and merging snips; masks; buffers
                     continuum.py  the continuum: evaluated as ALIS does, first
                                 guess, order, handles, sharing
```

The names may change as the code is written; the log records the names as built.

- **No logic in the widgets** (Stage 3). What a list contains, what a step changes,
  what a flag says: all decided in `alis/dashboard/` and tested there.
- **Every action is one step** of the project's `History`, made by an `edit.py`,
  `project.py`, `remove.py` or new-module function. A drag is one step
  (`history.group`). A widget never changes the text, a snip or `ui/project.json`
  itself.
- **Every value goes through the `Gate`** (F8). The plots show data, regions and
  continua, which reveal no hidden value (D24 allows blinded profiles to be drawn);
  numbers in boxes, tables and tooltips go through the gate.
- **Curves and pixels come from ALIS.** The continuum drawn is computed by ALIS's
  own function class (`legendre`, `chebyshev`, `polynomial`, `constant`), on the
  pixels ALIS loads for that data line (`min=`/`max=` as ALIS reads them). The buffer
  a snip needs comes from its resolution function's `getminmax`, as ALIS (and
  `validate._buffer`) computes it. What the dashboard draws is what ALIS fits.
- **Source spectra are read once** per session (`modes.QSOAbsLineMode.load`, with
  the row's column roles), kept by checksum, and never written into the model: a
  snip is what ALIS reads.
- **One change to ALIS itself** (Q4.17): a data line's `columns=` may name a
  `badpix` column (1 for a bad pixel), and `load_data` never fits a bad pixel,
  whatever its fit range says. A snip cut from a source with a bad-pixel mask carries
  it as a fifth column; other snips keep `prepfit`'s four columns (D4), and a snip
  without the column has every pixel good (Q4.15).

### The spectrum view (`qt/plots.py`; D38, D43)

One widget for every spectrum in the dashboard, built on pyqtgraph:
- **The toolbar** (D38, Stage 0's draft 6): ⌂ home, ← back, → forward, ✥ pan, ⬚ zoom
  box, and + / − (zoom along the wavelength axis only). The wheel zooms the wavelength
  axis. Each view keeps its own history of zooms for ← and →.
- **Drawing:** the data as steps, centred on the pixels; the error, thin and grey;
  the continuum, dashed; the zero line, dashed grey. Large spectra (J1358p6522's
  source has 40,402 pixels) use pyqtgraph's downsampling and clip to the view.
- **Overlays** (an API the tabs use): fit regions (green), regions fitted by another
  snip (light blue), pixels fitted twice (hatched), masked and bad pixels, the snip's
  edges (orange handles), the continuum's knots, line IDs (ticks and labels, coloured
  by kind as in the key, with "+ n lines not labelled" and a switch for all lines,
  Q4.4), and a marker for a clicked point.
- **Look** (D43): colours from `qt/style.py` (Okabe–Ito), every state also told by a
  pattern or an icon; axes in Å, a velocity axis signed (+100, −100, 0); judged at
  1440×900.

### The Data tab (`qt/data.py`; D35, S24, S25, D14–D16, F13, S3)

Left: **Systems**, **Coverage** (when there are several files) and **Blinding**.
Right: the **Datasets** table and the **Spectrum**, as in the mockups and Stage 3's
frame.

**Systems** (D15, D16, S24 as revised in D35):
- **The primary z,** typed; Enter makes it one step. Changing it re-centres the
  views and moves no snip and no component (D16). There is no redshift history
  (D35): undo goes back.
- **⌖ Identify a feature…** arms the spectrum view. A click at λ opens a small panel
  beside the point with three menus, Element → Ion → Transition. Each transition is
  listed with the z it implies, z = λ/λ₀ − 1, and only those giving z ≥ 0 are
  offered. The first choice is the strongest transition that gives a z within the
  data's Lyα reach. Then **Set as primary z**, or **Add as a new system**.
- **Other systems:** one row each, with its z and line IDs ("z ≈ 2.40 H I 1215.7,
  H I 1025.7"); a right-click offers Rename, Make primary, Set z… and Remove…
  (S27, `remove.plan_system`, with the list of `.mod` lines it changes).
- **Generic absorbers** (`1Ly_a`, `1H_IB`): one row, with the number of
  components.
- **Add system…** asks for z and a name.

**Blinding:** "Blind the whole fit" (Stage 3's action: one step, not undone past,
switched off only by Unblind…), the list of what is blinded (ions and labels, no
values), and the Blind lines… and Unblind… buttons of Stage 3.

**Coverage** (several files; Q0.5): one row per transition of the primary system
that has a snip or a line ID, one column per file: ■ has a snip, □ covered by its
source with no snip yet, · not covered. Double-clicking a cell opens that transition
on the Regions tab.

**Datasets** (one row per file; S25, D14, D35, Q0.5, Q0.8):

| Ref. | Dataset (file) | Contents | FWHM (km/s) | Shift (km/s) | Zero level | Checksum |
|---|---|---|---|---|---|---|
| ◉ | its name | "wave, flux, error · 40,402 px · 3650–6305 Å", or "23 snips · no source spectrum" | value and badge | "0 · reference", or value and badge | Off / On | ✓ unchanged, moved (Relink…), changed, embedded |

- **Badges** (the same widget as Stage 5's component cards): ● free, ■ fixed,
  ⛓ tied to another row. Clicking one offers Free, Fixed and "Tie to <row>"
  (`edit.tie_rows`, Q0.8). A value is typed in its cell (`edit.set_fwhm`,
  `edit.set_shift`). By default (D14): the FWHM fixed; the reference's shift fixed at
  0, every other row's free.
- **The reference** (◉): choosing another row is one step. Its shift becomes fixed
  at 0, and the old reference's shift becomes free.
- **Zero level** On adds one `constant` for the row (`edit.add_zero_level`, D14);
  Off removes it, listing what changes if it was tied.
- **Add file…** opens Stage 3's "Add a spectrum (ascii)" dialog (file, FWHM,
  column roles, S3). The new row has its source, a free shift with its own label
  (`edit.new_row_labels`) and no snips; its snips are cut on the Regions tab.
- **Remove…** (right-click; S27, `remove.plan_dataset`) lists the `.mod` lines it
  changes: the row's data lines, their continua, and its specids in other lines.
- **Relink…** is Stage 3's dialog (F13).
- Clicking a row shows its spectrum below.

**Spectrum:** the selected row's source spectrum; without a source, its snips, each
scaled to its own peak (D35). Above it: Flux | Log, "Transitions of z = …" (the
primary system's line IDs, S5), and the toolbar.

**A row's column roles** (Q4.14). "Column roles…" in a row's right-click menu
reopens Stage 3's dialog for its source. The roles can be changed while the row has
no snips; once it has some, the dialog shows them without letting them change, and
says to CLEAR its snips first.

**An imported fit's structure** (Q2.4, Q4.14). The systems and rows were inferred;
the Data tab says so in each pane, and the banner points to it. The user edits the
systems as above, and renames rows; rows are never merged, and snips are not moved
between rows (a row follows the model's own resolution label). The first edit writes
the guessed structure, marked guessed, so that no edit merges rows; the "!" and the
banner stay until **Confirm** (`project.confirm_structure()`), which keeps the
structure, clears them, and checks the model again (Q4.18).

### The Regions tab (`qt/regions.py`; D36, S2, S5–S10, S26, D17–D19)

Left: **Transitions** and the **Key**. Centre: one spectrum per dataset, with the
tools above; below, the **Continuum** and **Snip** boxes for the selected dataset.

**Transitions** (S2, D36; `lines.py`):
- **System ▾** (the primary by default), **Element ▾** with ‹ ›, and **Ion ▾** (the
  ion stage, I, II, …) with ‹ › (Q4.19). Only elements and stages with a transition
  on the data are offered.
- **The list:** that ion's transitions, ranked by fλ, each with its rest and
  observed wavelengths and its flags: **gap** (no good pixels near it) and **edge**
  (near an end of the data); there are no forest or telluric flags (Q4.5; telluric
  bands are deferred). Every transition of the ion that falls on the data is listed,
  whatever its strength (Q4.4). Isotopes are not
  listed separately (D I is in H I's snips); transitions closer together than the
  FWHM share one row ("O I 971.7 ×2"). Only transitions that fall on the data are
  listed, and any that already have a snip (● in the list).
- **Keys** (single keys act only while the Regions tab has the focus, so typing in
  the `.mod` panel never triggers them): ↑ ↓ the next transition, [ ] the next
  element, − + the continuum's order, Delete the fit region last clicked.
- Without a source spectrum (an imported fit, D12), only transitions with a snip are
  listed, and SNIP is disabled ("needs the source spectrum").

**Key:** one item per line, as in the mockups: fitted pixels, fitted by another
snip, fitted twice, snip edges, continuum, and the three kinds of line ID.

**The spectra** (D36): one per dataset that covers the transition, stacked, never
overlaid, scrolling when there are more than three (a preference). The title names
the transition and the dataset ("Ly7 926.2 · J1358p6522_fluxcal.dat", or "O I 1302.2
· 3 datasets"). Each spectrum is centred on the transition at the system's z, ± the
snip's half-width; the bottom axis is wavelength, the top axis velocity (Q4.12).
Clicking a spectrum selects its dataset. Above them, what acts on the spectra
(Q4.19): **SNIP**, **CLEAR**, the fit regions' **Add regions to all** and **Tweak
dataset regions** (Q4.21), and the toolbar. What acts on the continuum is in the
Continuum box.

**SNIP and CLEAR** (Q0.6; `snips.py`):
- **SNIP** cuts the transition from each dataset whose source covers it (Q4.2). Each
  snip starts as the source's pixels within ± the half-width (300 km/s, a
  preference); its edges are then the user's to move (Q4.2). It is written in
  `prepfit`'s four-column format at full precision (Q2.7), with a fifth `badpix`
  column when the source has a bad-pixel mask (Q4.17). Pixels with a NaN, an error
  ≤ 0, or a bad-pixel mask of 1 are never fitted (S7). Its data line follows Stage 2's
  template (`fitrange=columns`, `loadrange=all`, the row's resolution and shift); it
  has no fit region until one is drawn (Q4.2), so until then its line carries an
  error ("draw a fit region") and Regions shows "!"; its continuum is the automatic
  first guess (S8); its specid is added to every absorption line whose transition
  falls in it, and the step's name says so (Q4.3). One step.
- **CLEAR** removes the transition's regions and snip in the selected dataset
  (`remove.plan_snip`, previewed as in S27). One step.
- Adding a region on a transition with no snip makes it one, as SNIP (Q4.2).

**Regions and masks** (S6, S7, D18):
- **Two tools, as in `prepfit`** (Q4.21): in both, a drag across a spectrum adds a
  fit region (`project.set_regions`, which keeps the mask column character for
  character, Q2.7, and converts `fitrange=[lo,hi]` when needed, Q2.10), and a
  right-drag leaves pixels out (their mask set to 0), splitting a region where needed.
  **Add regions to all** does it in every dataset of the transition; **Tweak dataset
  regions** in the selected dataset alone (a cosmic ray, a bad pixel). A region's
  edges can be dragged while either is on; Delete or the right-click menu removes a
  region (from every dataset, or from this one alone).
- **No automatic masking of outliers** (Q4.11): the user leaves pixels out by hand,
  through the regions and the right-drag. Outliers against the model wait for Stage 6
  (S29). A bad pixel (`badpix` 1) is never fitted, and no region or edit can fit it.
- **Several datasets** (D18, Q4.19): one set of fit regions per transition. With Add
  regions to all, a region added, moved or removed on one dataset is added, moved or
  removed on every dataset of the transition, moved by the difference of their shifts,
  in one step (`snips.linked_step`); pixels left out of one dataset while tweaking stay
  out there only. An
  imported fit's datasets may have regions of their own: "Use HIRES's regions in
  every dataset" makes them the same ("copied from HIRES").

**The snip's edges** (S6): orange handles. Moving one inwards keeps the lines of the
pixels that stay (`project.set_snip_edges`, Q2.7). Moving one outwards re-cuts the
snip from the source, keeping the old lines byte for byte and writing the new pixels
at full precision (Q4.9); without a source it is refused with a message.

**Continuum** (S8, D17, S9, S10; `continuum.py`):
- **The function** ▾: Legendre by default (D17); Chebyshev, Polynomial and Constant
  are offered too. Others are typed in the `.mod` panel.
- **Its tools** (Q4.19): Auto first guess, Add knot, Clear knots and Normalised are
  in the Continuum box, apart from what acts on the spectra.
- **The order,** − n +, also on the − and + keys: each change refits the coefficients
  to the continuum pixels, one step. Legendre stops at order 10, as ALIS does.
- **Auto first guess** (Q4.6, Q4.16, Q4.19): the snip's pixels, leaving out bad and
  masked pixels and ±15 km/s (a preference) around the transitions of the model's
  absorption lines and of the systems' line IDs that fall in the snip (which include
  the transition snipped), are clipped in three passes of rising order: a straight
  line, then one order higher, then one more, each started from the pixels the last
  kept and judging every pixel again, rejecting those far below the curve (2.5σ,
  absorption) or above it (3σ). Every order is fitted to the pixels left; its order
  is the one with the lowest BIC, up to 5
  (a preference).
- **The table:** for the orders n − 1, n and n + 1, the χ² of the continuum pixels,
  Δχ² and ΔBIC against the current order (the mockup).
- **Knots** (Q4.7, Q4.13): points (λ, flux) that the starting continuum must pass
  through, added, moved and removed on the spectrum; the user's adjustment of the
  first guess. Every refit (the order, Auto first guess) is a least-squares fit to
  the continuum pixels held through the knots. The knots set only the starting
  coefficients: ALIS fits the continuum freely, so the best fit may leave them. They
  are kept in `ui/project.json`. A Legendre of order n passes through at most n + 1
  knots: a knot beyond that raises the order (up to 10), and − stops at the number of
  knots less one. One knot added, moved or removed is one step.
- **Shared continua** (S10, Q4.8): a continuum is shared only by snips of one
  transition in different file rows: one `legendre` line with `specid=A,B` and one
  `min=`/`max=`, the union of the snips' extents, kept up to date when an edge moves.
  Only a Legendre can be shared (only it takes `min=`/`max=`). It is drawn on each of
  them, and the box says so. "Share with <row>" joins them; "Unshare" gives each its
  own again, refitted.
- A hidden continuum (`blind=True`) is drawn and can be edited; its coefficients are
  never shown, and stay hidden (Stage 3).

**Snip** (S6, S26, D19):
- **Extent** λ₁–λ₂ Å and its pixels; **fitted** pixels in n regions; **buffer** blue
  and red, against what the resolution needs (✓ or !, from `getminmax`).
- **Pixels fitted twice** (D19, S26; `load.find_shared_pixels`): "36 fitted pixels
  are also fitted by other snips (18 with Ly8 923.2, 18 with Ly6 930.7). Each pixel
  should enter χ² once." Two fixes, each one step:
  - **Keep them in this snip only:** the pixels leave the other snips' fits;
  - **Merge the snips…:** one snip from both (the union of their pixels and regions,
    one continuum refitted, the model lines' specids merged). It always opens a
    confirmation, like Remove…, that lists the lines it changes and names the specid
    kept and the one replaced, with Merge and Cancel (Q4.9). It is offered first when
    the overlap is large (a quarter of either snip's fitted pixels, a preference;
    D19). Overlapping snips are merged from their own files (Q4.9(b)).
  - **Apply the fix to all datasets** when the same overlap is in each.
  An imported fit that overlaps may still be fitted (D19).
- **Several datasets:** a table of each dataset's fitted and shared pixels, and where
  its regions came from ("drawn here", "copied from HIRES").

### Cross-highlighting (S16)

- **From the tabs:** a selected row, system, snip or continuum highlights its lines in
  the `.mod` panel (`project.lines_of`, `panel.select_items`).
- **From the `.mod` panel:** the cursor on a data line selects its row on the Data
  tab and its snip on the Regions tab; on a continuum line, its snip and the
  Continuum box; on an absorption line, its system (`project.items_at`).

### Markers (F12, Q3.6)

Stage 3's rules stand. One is added to the Regions tab's "!": a snip whose buffer is
narrower than its resolution needs (F5's check, already in the validator).

### Preferences and actions

- **New preferences** (`preferences.py`, each a `Spec`): the snip's half-width (300
  km/s); the number of spectra shown before scrolling (3); the continuum's highest
  order in the first guess (5), its clipping (2.5σ below, 3σ above) and the width
  left out around known lines (±15 km/s, Q4.6); the overlap from which merging is
  offered first (25%); the strength above which a line is labelled (Q4.4).
- **New actions** (`actions.py`): the Regions tab's keys above, Identify a feature,
  SNIP, CLEAR, Add region, Exclude pixels, Add knot, Flux | Normalised, and Auto first
  guess. The registry gains a scope, so that a single-key shortcut acts only on its
  tab (Stage 3's shortcuts are all window-wide).

### Not in this stage

- The absorption model drawn on the spectra, and the components (Stage 5: S11–S15,
  S30; Q4.12).
- Downward outliers, which need the model (S29, Stage 6).
- Attaching a source spectrum to an imported fit (S23), spec1d files and Orders mode
  (later).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage4_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests;
> run the `fast` batch before the stage closes.

**4.1 — Transitions and line IDs (`lines.py`, no Qt; S2, S5, D36).** [DONE 2026-10-07]
- An ion's transitions on the data: rest and observed wavelengths, fλ, the ranking,
  isotopes folded into their element, near-coincident lines in one row, and the
  flags (gap and edge; Q4.5).
- Line IDs for every system in a wavelength range, with the number left unlabelled
  and a switch for all lines (Q4.4); the transitions offered by Identify a feature,
  with the z each implies; the coverage grid.
- **Check** (unit tests): J1358p6522's H I list and its flags match the mockup
  (Lyβ to Ly10, Lyα "gap"); Q1243p307's coverage grid (16 transitions × 3 files)
  matches the mockup; synthetic spectra for gap and edge; a click on the Lyβ of
  J1358p6522 offers z = 3.06726.

**4.2 — Snips (`snips.py`, no Qt; S6, S7, D18, D19, Q2.7, Q2.10).** [DONE 2026-10-07]
- The `badpix` column in ALIS (Q4.17): `load_data` accepts it in `columns=`, and
  never fits a pixel whose `badpix` is 1 (`load_ascii` and `load_fits`); a unit test,
  and the regression harness unchanged.
- Cutting a snip from a source (columns, the bad-pixel mask as a `badpix` column,
  NaN and non-positive errors), its file and specid names, the SNIP step (data line,
  file, continuum, specids in the absorption lines, `ui/project.json`), CLEAR, moving
  the edges (inwards from its own lines, outwards from the source), masking pixels by
  hand (never a bad pixel), copying regions between datasets, the buffer, and the two
  fixes for shared pixels (keep in one snip; merge, with its confirmation).
- **Check** (unit tests): after every step that leaves each snip a region, ALIS reads
  the model and its data (`validate.check`, with no fit); one undo restores the text,
  the files and `ui/project.json` byte for byte; lines of a snip that stay are
  unchanged; a bad pixel is never fitted, by the dashboard or by ALIS; after "keep" or
  "merge", `load.find_shared_pixels` finds nothing; on the `metal_line_abs` spectrum,
  the O I 1302 and Si II 1304 snips overlap and merge into one.

**4.3 — The continuum (`continuum.py`, no Qt; S8, S10, D17).** [DONE 2026-10-07]
- Evaluating a continuum line with ALIS's own function on the pixels ALIS loads; the
  first guess; the order table; changing the order; the knots (Q4.13); sharing and
  unsharing (one transition, several rows, Legendre only; Q4.8).
- **Check** (unit tests): the curve equals ALIS's own evaluation, to 1e-12, for every
  emission line of the `examples/` models; on synthetic data the first guess recovers
  a known polynomial under absorption lines, and the BIC picks its order; a refit
  passes through every knot and fits the other pixels; a shared continuum reads in
  ALIS with its `min=`/`max=`.

**4.4 — Datasets and systems (no Qt; S25, D14–D16, Q0.8, Q2.4).** [DONE 2026-10-07]
- The edits behind the Data tab that do not yet exist: changing the reference, the
  zero level on and off, adding a file as a row, renaming a row, a row's column roles
  (while it has no snips), and the systems (z, add, rename, make primary, remove);
  `confirm_structure`. Rows are never merged (Q4.14).
- **Check** (unit tests): each is one step that ALIS reads, and one undo restores
  it; the reference's shift is fixed at 0 and the old one's freed; a row with snips
  keeps its column roles; removing a system lists its lines first.

**4.5 — The spectrum view (`qt/plots.py`; D38, D43).** [DONE 2026-10-07]
- The pyqtgraph widget, its toolbar and history of zooms, the overlays, Flux | Log
  and Flux | Normalised, colours from `qt/style.py`.
- **Check** (`gui` tests): home, ← →, the zoom box, and + − along the wavelength axis
  only; J1358p6522's 40,402 pixels draw in under half a second off-screen; nothing
  in the view's tooltips holds a hidden value.

**4.6 — The Data tab (`qt/data.py`; D35).** [DONE 2026-10-07]
- Systems with Identify a feature, Blinding, Coverage, the Datasets table with its
  badges, the spectrum, an imported fit's structure and Confirm, and
  cross-highlighting.
- **Check** (`gui` tests): on a new project from `OI_SiII.dat`, Identify a feature
  sets the primary z; on Q1243p307, a tie between two rows reaches the model and one
  undo removes it; Confirm clears the Data tab's "!" and the banner; on
  `examples/blind`, no hidden value in the tab; screenshots at 1440×900 beside the
  mockups.

**4.7 — Review of the Data tab (Q4.1).** [DONE 2026-10-07; RJC's comments applied, Q4.18]
- Publish the Data tab's screenshots beside its mockups, on the page used in Stage
  3 (https://claude.ai/artifact/A6ohUEnAe91TEDsKBLwuBt) or a new one, for RJC's
  comments, and apply them before the Regions tab is built on the same view.

**4.8 — The transition list and the key (`qt/regions.py`; S2, D36).** [DONE 2026-10-07]
- System, Element ‹ ›, the stage buttons, the ranked list with its flags and ●, the
  key, the keys, and the tab-scoped shortcuts.
- **Check** (`gui` tests): J1358p6522's H I list as in the mockup; ↑ ↓ and [ ] while
  the tab has the focus, and no shortcut taken while typing in the `.mod` panel;
  without a source, only snipped transitions are listed and SNIP is disabled.

**4.9 — The spectra and their tools (S6, S7, S9, D18; Q0.6).** [DONE 2026-10-07]
- One spectrum per dataset, Draw region, Mask pixels, SNIP, CLEAR, the edge
  handles, Flux | Normalised, the line IDs, and copying regions.
- **Check** (`gui` tests): a drawn region reaches the snip's mask column and one undo
  restores the file; SNIP then CLEAR leaves the project as it was; on Q1243p307,
  a region drawn on HIRES is copied to PROCHASKA and KIRKMAN; an edge dragged
  outwards on J1358p6522 re-cuts the snip from its source.

**4.10 — The Continuum box (S8, S9, S10, D17).** [DONE 2026-10-07]
- The function, the order with − + and the keys, Auto first guess, the table of
  orders, the knots, and sharing.
- **Check** (`gui` tests): + then − gives back the same text; adding, moving or
  removing a knot is one step; a continuum hidden with Blind lines… is drawn, and its
  coefficients are never shown.

**4.11 — The Snip box (S6, S26, D19).** [DONE 2026-10-07]
- The extent, the fitted pixels, the buffer, the pixels fitted twice and the two
  fixes (for all datasets at once), and the table of datasets.
- **Check** (`gui` tests): J1358p6522's Ly7 shows the mockup's 36 shared pixels (18
  with Ly8, 18 with Ly6); "keep" clears them and the Regions tab's "!"; on
  Q1243p307, merging O I 1302 and Si II 1304 is offered first and applies to the
  three datasets.

**4.12 — From a spectrum to a fit (end to end).** [DONE 2026-10-07]
- A `fast` test: New project from `examples/metal_line_abs/data/OI_SiII.dat` (z = 0,
  its fourth column ignored, as the example's model reads it), driven through the
  Data and Regions tabs (SNIP O I 1302 and Si II 1304, merge, the fit region
  1301–1305 Å, the first-guess continuum), with the two absorption lines added by
  `edit.add_ion` (their panel is Stage 5's). `run_alis project.model` then fits, and
  the column densities agree with `fit_spectra.mod.out.reference` within their errors.

**4.13 — Close the stage.** [DONE 2026-10-08: drafts 1 and 2 reviewed (Q4.19–Q4.21), RJC's comments applied]
- Publish the screenshots of both tabs beside the mockups for RJC's review, as in
  Stage 3, and apply the review.
- `doc/ALIS_workflow.md`: a short section on preparing a fit with the Data and
  Regions tabs. Also update `CHANGELOG.md`, `tests/README.md` and the `gui-component`
  skill (how to use the spectrum view).
- Run `test-coverage` on the new modules, and the `unit`, `gui` and `fast` batches.
- Record in this document what Stage 5 receives, and update the stage table in
  `dashboard_stage0.md` if anything moved.

## Status (2026-10-08)

*Written by Claude at the end of Prompt 4, and brought up to date with each prompt
since. The details are in `claude_prompts/logs/dashboard_stage4_log.md`.*

**Stage 4 is closed** (Prompt 5 of this document). Tasks 4.1–4.13 are done. The
Data tab was reviewed halfway (Q4.18, https://claude.ai/artifact/T4v5mJk179rDAqCjU1GJNe)
and the Regions tab at the close, in two drafts (Q4.19–Q4.21,
https://claude.ai/artifact/HPnK8zEaFdACVwcGSKx2aT, now showing the stage's final
state); every comment is applied. The stage's decisions are D52–D59 of
`ALIS_v2_dashboard_prompts.md`; the stacked spectrum (Q4.20) is placed with Orders
mode, after v1. Stage 5's document is `dashboard_stage5.md`.
- **Tests.** 9 new test files: 4 without Qt (`unit`), 4 of the windows (`gui`), and
  the end-to-end fit (`fast`); 7 `badpix` tests in `test_load_files.py`. Coverage of
  the new modules over their tests: 89% (`lines` 97%, `continuum` 90%, `snips` 89%,
  `qt/regions` 89%, `qt/plots` 87%, `qt/data` 85%, `datasets` 83%).
- **The end-to-end fit** (4.12): a new project from `OI_SiII.dat`, taken through the
  two tabs, fits to O I 13.984 ± 0.015 and Si II 13.044 ± 0.049 (the reference:
  13.985 ± 0.016 and 13.041 ± 0.050), χ² 358.4 (357.9).
- **The batches** (Prompt 4): `unit` 1811 passed, 1 failed (below); `gui` 115
  passed; `fast` 112 passed, 0 failed (13 min 36 s). After draft 2 (Prompt 5):
  `unit` 1814 passed, 0 failed; `gui` 115 passed; `fast` 112 passed, 0 failed. At the
  close: `unit` 1814 passed, 0 failed; `gui` 117 passed; `fast` 112 passed, 0 failed
  (13 min 25 s).
- **ALIS outside `alis/dashboard/`:** the `badpix` column (`config.py`, `load.py`,
  `save.py`; Q4.17). ALIS's fitting code is otherwise unchanged.
- **Not from this stage:** in Prompt 4,
  `test_atomic_mass.py::test_every_value_in_the_xml_survived_the_conversion` failed
  on an uncommitted edit to `alis/data/atomic.ecsv` (not Claude's), which RJC has
  since reverted; it passes again.

### What Stage 5 receives

The Components tab of Stage 5 is built in the frame of `qt/tabs.py`
(`ComponentsTab`), as the `gui-component` skill describes, on what Stage 4 adds:
- **The spectrum view** (`qt/plots.py`): `SpectrumView` (data as steps, context,
  continuum, Normalised, regions, edges, knots, line IDs, a velocity axis,
  `set_labels` for stacked views) and `ZoomBar` (linked views, history); the skill
  documents both. Stage 5 adds the model drawn on the spectra (Q4.12).
- **Transitions and line IDs** (`lines.py`): `atomic(registry)` (lines by ion, the
  main isotope of each element and stage), `ion_transitions`, `line_ids`,
  `snip_transitions(project)` (the transition of every snip, recorded or guessed),
  `snipped_by_ion`, `system_ions`.
- **Snips** (`snips.py`): `snips_of(project, system, ion, rest)`, `regions_of`,
  `fitted`, `row_shift`, `strips`, `linked_snips` (a transition's snip in every
  dataset); SNIP adds a new snip's specid to every absorption line with a transition
  inside it (Q4.3), so a component added later takes its specids from `snips_of` and
  `snip_transitions`.
- **The continuum** (`continuum.py`): `evaluate(project, snip)` (ALIS's own curve),
  `current`, `knots_of`; the Components tab's live preview can divide by it.
- **Datasets and systems** (`datasets.py`): the systems (add, set z, rename, make
  primary, remove) and `keep_structure` (an imported fit's structure written, marked
  `guessed` until Confirm, at its first edit).
- **The registry's tab scope** (`actions.py`): a tab's own actions take
  `scope="Components"` and a `COMPONENTS_TAB` group (to add to `TAB_GROUPS`); the
  window builds them on the tab (`window._build_scoped`, `Tab.bind_action`), so a
  single key acts only while the tab has the focus.
- **A tab that refreshes only when shown** (`RegionsTab.refresh`, `_tab_shown`), and
  `session.view["regions"]` as the way another tab opens a transition there (the Data
  tab's coverage grid does).
- **Preferences** (`preferences.py`): `regions.*`, `continuum.*` and `lines.strength`
  join Stage 3's.
- **Drawing a model** is new in Stage 5: the continuum is evaluated by ALIS's own
  function classes (`continuum.evaluate`, through `_load_with_alis`); the absorption
  model can be evaluated the same way, through `alis.model_eval` (`model_func`, with
  `getemab` for the emission and absorption parts) on the state ALIS builds for the
  project.

## Skills to use for this stage

- `gui-component`: each panel of the two tabs, wired to the project, the history,
  the validator and the gate; and the spectrum view, which it then documents.
- `gui-dev`: driving the window off-screen and taking the screenshots at 1440×900.
- `atomic-data`: reading the transitions (ion, wavelength, f-value) from ALIS's
  atomic data.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task; the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of `lines.py`, `snips.py` and
  `continuum.py`.
- `check-fit`: the end-to-end fit of Task 4.12 against its reference.
- Claude Code's `artifact-design` skill: the review pages (4.7, 4.13).

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`:
  - D7, D12–D19, D24, D35, D36, D38, D43–D50;
  - QF.5, QF.6, QF.7, QF.9, QF.21, QF.23, QF.24 and QF.28;
  - F5, F8, F12, F13, S2, S3, S5–S10, S16, S24–S27.
- `claude_prompts/dashboard_stage0.md`: Q0.5 (one row per file), Q0.6 (SNIP and
  CLEAR), Q0.8 (ties between files), and the reviews of the Data and Regions drafts
  (Task 0.10).
- `claude_prompts/dashboard_stage1.md`: the bundle's sources, and the shared-pixel
  check (`load.find_shared_pixels`).
- `claude_prompts/dashboard_stage2.md`: Q2.3 (labels), Q2.4 (an imported fit's
  structure), Q2.7 and Q2.10 (editing snips and their fit ranges).
- `claude_prompts/dashboard_stage3.md`: "What Stage 4 receives", Q3.16–Q3.19 (RJC's
  review of the skeleton: New project, the column roles, the bad-pixel mask).
- `doc/ALIS_workflow.md`, section 0 (preparing fitting regions, and what `prepfit`
  does).
- **The code:**
  - `alis/dashboard/` (the project model, and Stage 3's modules);
  - `alis/dashboard/qt/tabs.py` (the frames of `DataTab` and `RegionsTab`);
  - `alis/load.py` (`find_shared_pixels`; how a data line's pixels are loaded, and
    the buffer around the fit);
  - `alis/functions/legendre.py`, `chebyshev.py`, `polynomial.py`, `constant.py` and
    `vfwhm.py` (the continua, and `getminmax`);
  - `alis/prepfit/specplot.py` (its transitions, and the solar abundances behind the
    expected strengths).
- **The mockups:** `doc/dashboard/mockups/data_new.html`, `data_datasets.html`,
  `regions.html` and `regions_datasets.html` (built by `build_mockups.py`), published
  at https://claude.ai/artifact/J43w9ERNESDo9hez9o918B.
- **The data:** `examples/metal_line_abs/` (a source spectrum and its model),
  `context/fitting_examples/VMP_DLA/J1358p6522/` (a source spectrum and its snips),
  and `context/fitting_examples/DH/Q1243p307` (three datasets, no source spectra).

## Queries

*Raised by Claude on 2026-10-06, while writing this document. Each gives Claude's lean,
and the Design section and the tasks follow the leans.*

**Q4.1 — A review halfway through the stage.** Both tabs draw on the same spectrum
view, and Stage 3's review took three drafts. I propose to publish the Data tab's
screenshots once it is built (Task 4.7), so that the view and the look are agreed
before the Regions tab builds on them, and to review both tabs again at the close.

My lean: yes, two reviews.

**Response:** I agree

**Q4.2 — What SNIP cuts.** Q0.6's response gives SNIP and CLEAR buttons. To decide:
- **(a)** A new snip spans ±300 km/s around the transition (a preference), and has no
  fit regions until some are drawn (the Regions tab shows "!" meanwhile). The
  alternative is a default region of ±60 km/s, as Stage 2's template uses.
- **(b)** SNIP cuts the transition from every dataset whose source covers it, and
  CLEAR (in one dataset's spectrum) removes it from that dataset only. D14 asks that
  the user choose which datasets are fitted for each transition; this does it by
  removal.
- **(c)** Drawing a region on a transition that is not yet a snip makes it one, as if
  SNIP had been pressed.

My lean: (a) no default region; (b) every covering dataset; (c) yes.

**Response:** I agree, but want to clarify: (a) no default region. The user should be able to move the handles as they wish to define the snip region. There can be a starting value for each snip, but ultimately, the user decides where to snip the data; (b) every covering dataset; (c) yes.

**Q4.3 — Keeping specids in step.** ALIS applies an absorption line only to the data
whose specid it lists. When a snip is cut, I propose to add its specid to every
absorption line (of any system) with a transition inside the snip, so that the model
stays complete for the new data; CLEAR already removes it (S27). A blend of two ions
in one snip is then handled without the user listing specids.

My lean: yes, automatically, and the step's description in Undo says so.

**Response:** Yes, automatically, and the step's description in Undo should clearly indicate that specids have been updated for all relevant absorption lines.

**Q4.4 — Which lines are labelled.** The atomic data hold thousands of transitions, so
"Transitions of z" (Data) and the line IDs (S5, Regions) need a rule. I propose:
- every transition of the ions in the model or in a system's line IDs;
- plus the strong lines: those whose expected strength, from the solar abundance (the
  table `prepfit` uses) and fλ, is above a threshold (a preference);
- and a switch for all lines in the view.

My lean: as proposed.

**Response:** This is a good decision overall, but we should not hide any lines that would then prevent the user from considering them for fitting. The user should have the option to view all lines if they wish, even if they are weak. To make this clear, the user should be able to see how many lines are transitions are currently being hidden from the view. For example, "+ 1024 lines not listed".

**Q4.5 — The forest and telluric flags.** S2 flags transitions in the Lyα forest and
in telluric bands:
- **(a)** The forest is blueward of the quasar's Lyα emission, which needs the
  quasar's redshift; the project does not have it. The mockups used the primary
  system's own Lyα, which is always inside the forest. I propose that, plus an
  optional quasar redshift in the Systems pane, which, when given, sets the edge.
- **(b)** ALIS has no list of telluric bands. I propose a small built-in list of the
  strongest bands (the O₂ A, B and γ bands, and the H₂O bands near 7200, 8200 and
  9300 Å), in the observed frame.

My lean: (a) and (b) as proposed.

**Response:** I don't think (a). The goal of these transitions is to allow for single blended lines. This most commonly happens in the forest, but not always. It's just a label for an unidentified blend, really. I also don't think (b) is needed at this stage. We can revist it at a later time, and feel free to add this to the `deferred work.md` file for consideration later.

**Q4.6 — The continuum's first guess (S8).** I propose a least-squares fit of the
continuum function to the snip's pixels, repeated while clipping pixels more than
2.5σ below or 3σ above the curve. Masked pixels are left out, and so are pixels
within ±50 km/s of the transitions of absorption lines already in the model. The
order with the lowest BIC, from 0 to 5, is chosen. The table then shows the orders
either side of it.

My lean: as proposed; the numbers are preferences.

**Response:** That's a good idea, but only mask within ±15 km/s of the transitions of absorption lines already in the model. The user should have the option to adjust this range if they wish, as some lines may be broader or narrower than expected. Additionally, we should provide a way for the user to manually adjust the first guess if they feel it is not accurate enough.

**Q4.7 — Dragging the continuum.** A polynomial of order n passes through n + 1
points. I propose n + 1 handles on the curve, at fixed places across the snip; dragging
one moves the curve through it while the others stay put. The alternative is to drag
the curve anywhere, which moves its nearest handle.

My lean: handles, shown while the Continuum box is open.

**Response:** 'Dragging' the continuum is not the best approach here. Instead, we could allow the user to add fitting knots that the continuum MUST pass through. This would give the user some more control, particularly over broad absorption features, where there is very little continuum. Such continuum knots should be able to be added, moved, or removed by the user.

**Q4.8 — Shared continua (S10).** The Orders-mode models of
`context/fitting_examples/DH_orders/Q1243p307` share one polynomial between snips
(`specid=A,B`, some with `min=`/`max=`). I propose that the Regions tab shows a shared
continuum, keeps it working when the snips change, and offers "Share with the next
snip" and "Unshare". Orders mode needs the same later (D30).

My lean: all three in this stage.

**Response:** Yes, all three in this stage, as it will prepare us for when we use the orders mode later. Just note, the sharing option only applies to different input datasets that cover the same transition. We should avoid sharing a continuum between two different transitions, as this would not be physically meaningful. The only option to share a continuum should be between two snips that are of the same transition, but from different datasets, and they MUST have the same `min=` and `max=` options set.

**Q4.9 — Growing and merging snips.** Q2.7's response allows a snip to be written
again when its edges change. I propose:
- **(a)** moving an edge outwards keeps the snip's old lines byte for byte and adds
  the new pixels from the source, at full precision;
- **(b)** merging two overlapping snips keeps both files' lines, as one file sorted
  by wavelength, so that no source is needed; non-overlapping snips are merged from
  the source;
- **(c)** the merged snip keeps the first snip's specid, and the second's specid is
  replaced by it in every model line.

My lean: (a), (b) and (c) as proposed.

**Response:** I agree with (a), (b), and (c) as proposed. However, we should also provide a warning to the user when merging snips that have different specids, as this could lead to confusion about which data is being used for the fit. The user should be able to confirm or cancel the merge operation.

**Q4.10 — Editing an imported fit's structure (Q3.16(c)).** Your response asks that
the user can edit what was inferred. I propose these edits in this stage:
- **systems:** z, name, which is primary, add, remove;
- **datasets:** rename a row, merge two rows, and move snips from one row to another;
- **Confirm**, which keeps the result.

Moving components between systems belongs to the Components tab (Stage 5).

My lean: as proposed.

**Response:** I think we might be talking about something different here, so let's try to resolve the confusion. My response was referring to the ability to edit the columns read in by the ascii file. For example, if the user has a file with 5 columns, but the first column is not wavelength, they should be able to edit the column roles to specify which column is wavelength, flux, error, etc. This is different from editing the structure of the fit itself (systems and datasets). So, I propose that we allow the user to edit the column roles for loading ascii datasets. We will deal with loading PypeIt spec1d fits files at a later stage. Also, related to your query, we should never allow datasets to be merged. If your query relates to something different, let's discuss very clearly what you are proposing, and how this is different to my understanding, based on my response here, and my response to Q3.16(c). Thanks!

**Q4.11 — Masking outliers before the model exists (S7).** Without a model, an
absorption line looks like a downward outlier. I propose that Mask spikes masks only
pixels above the continuum (cosmic rays, hot pixels). Downward outliers wait for the
fit (S29, Stage 6).

My lean: as proposed.

**Response:** No masking of outliers should be implemented during this stage. Sometimes, an input dataset will contain a bad pixel mask where bad pixels have already been flagged. In this case, we should respect the input bad pixel mask and not allow the user to unmask these pixels. In a future stage, we can implement a feature that will allow pixels to be removed from the fit region based on detecting outliers relative to the model fit, however, this should not be implemented in this stage. The user should be able to manually mask out any pixels they feel are bad (via their choice of fit regions), but we should not implement any automatic masking of outliers at this stage.

**Q4.12 — What the spectra on the Regions tab show.**
- **(a)** No absorption model is drawn in this stage: the components, and the live
  preview that draws them (S13), are Stage 5's. An imported fit's model is shown on
  the Fit tab (Stage 6).
- **(b)** The bottom axis is wavelength (Å), and the top axis velocity relative to the
  transition at the system's z, signed (D43).

My lean: (a) and (b) as proposed.

**Response:** Both are great ideas! Note that at this stage, the regions panel is not expected to overlay a best-fitting model. However, we can consider adding a feature in a future stage that will allow the user to toggle the display of a best-fitting model over the data, if they wish and a best-fitting model exists. For now, we will focus on allowing the user to define fit regions and mask pixels without any model overlay.

*Raised by Claude on 2026-10-07 (Prompt 1), after reading the responses above. How I
read each response comes first, then four new queries, each with a lean. The Design
section and the tasks still read as they were written, and will be brought in line
once these are answered.*

**Notes on the responses.**
- **Q4.2.** SNIP cuts ±300 km/s (a preference) only as the starting extent. Both edges
  are handles that the user moves freely: inwards keeps the lines that stay, outwards
  re-cuts from the source (Q4.9(a)). No fit region is drawn for the user. ALIS stops
  on a data line that fits no pixels, so until the first region is drawn the line
  carries an error in the `.mod` panel ("draw a fit region"), and Regions shows "!".
- **Q4.3.** Automatic. The step's name in Edit → Undo says so, for example "SNIP O I
  1302.2 (its specid added to 2 absorption lines)".
- **Q4.4.** The line IDs show every transition of the ions in the model and in the
  systems' line IDs, plus the strong lines. A switch shows every line, and a note
  gives the number left out ("+ 23 lines not labelled"). The Transitions list of the
  Regions tab hides nothing by strength: it lists every transition of the chosen ion
  that falls on the data.
- **Q4.5.** No forest flag, no quasar redshift and no telluric flag: the flags are gap
  and edge. I read your note as: an unidentified blend, in the forest or not, is
  fitted with a generic absorber (`1Ly_a`, `1H_IB`). A list of telluric bands is now
  in `deferred_work.md` (§7.1).
- **Q4.6.** ±15 km/s, a preference the user can change. The manual adjustment of the
  first guess is the knots of Q4.7 (Q4.13 below).
- **Q4.8.** "Share with <the next snip>" (two transitions of one dataset) is dropped.
  A continuum is shared only by snips of one transition in different file rows: one
  `legendre` line with `specid=A,B` and one `min=`/`max=`, set to the union of the
  snips' extents and kept up to date when an edge moves. Only a Legendre can be
  shared, since only it takes `min=`/`max=` in ALIS.
- **Q4.9.** Every merge joins two snips with different specids, so the warning always
  applies: "Merge the snips…" opens a confirmation, like Remove…, that lists the lines
  it changes and names the specid kept and the one replaced (and in how many lines),
  with Merge and Cancel.
- **Q4.11.** Mask spikes is removed, and nothing masks outliers automatically. Mask
  pixels (by hand) stays. A pixel of the bad-pixel mask is never fitted: a region drawn
  over it leaves it out. Pixels that ALIS cannot fit at all (a NaN, or an error of 0 or
  less, since ALIS divides by the error) are left out in the same way; they are not
  outliers. Where the bad pixels are kept is Q4.15.
- **Q4.12.** As proposed. A switch to draw the best-fit model on the Regions tab is
  now in `deferred_work.md` (§7.2).

**Q4.13 — Continuum knots (your Q4.6 and Q4.7 responses).** A knot is a point (λ,
flux) that the continuum must pass through; the user adds, moves and removes knots
on the spectrum. ALIS has no notion of a knot, so one choice decides the rest:
- **(a) Through the knots in the starting model only.** The dashboard fits the
  coefficients to the continuum pixels with the curve held through the knots (a
  constrained least-squares fit) and writes them into the `.mod`. ALIS then fits the
  continuum freely, so the best fit may move away from the knots. Nothing new is
  written in the `.mod`; the knots are kept in `ui/project.json`.
- **(b) Through the knots in the fit too, with the Legendre.** For each knot, one
  coefficient becomes an expression of the others in the `link` block, so that the
  curve passes through the knot whatever the fit does with the free coefficients. No
  change to ALIS, but the expressions are long, and are rewritten whenever the order,
  a knot or the snip's range changes.
- **(c) Through the knots in the fit too, with ALIS's `spline`.** The continuum
  becomes `spline f1 f2 … locations=λ1,λ2,…`: a cubic spline through the knots, with
  the flux at each knot a parameter, free or fixed. It always passes through its
  knots; the knots replace the order, and at least four are needed. It suits broad
  absorption with little continuum (a DLA's wings).

In (a) and (b), a Legendre of order n can be held through at most n + 1 knots. I
propose that a knot beyond that raises the order by one (up to 10), and that − cannot
go below the number of knots less one.

My lean: (a), with the knots kept in the project and honoured by every refit (the
order's − and +, Auto first guess). If you want the fit itself to keep to the knots,
(c) as one more function in the Continuum box's menu, for the snips that need it.

**Response:** I agree with (a). We should not include the knots in the fit itself. The knots are exclusively used to help set the initial starting parameters.

**Q4.14 — Two meanings of "structure" (your Q4.10 response).** We used the word for
two different things. Both are needed, and they are separate:
- **What you meant: a spectrum's column roles.** Which column of a text file holds
  the wavelength, flux, error, continuum or bad-pixel mask, and which are ignored.
  Stage 3 built this ("Add a spectrum (ascii)", from New project), and the Data tab's
  "Add file…" uses the same dialog. I propose one addition: "Column roles…" in a
  row's right-click menu reopens the dialog for that file. Its snips were cut with
  the roles it had, so the roles can be changed while the row has no snips; once it
  has some, the dialog shows the roles without letting them change, and says to
  CLEAR its snips first.
- **What I meant: an imported fit's systems and rows.** When a plain `.mod` is opened
  (`alis fit.mod`, F4), nothing in it says which lines form a system or a dataset, so
  the dashboard guesses (Q2.4): a system is a group of absorbers within 500 km/s of
  each other; a row is the data lines that share a resolution label (or else a file
  name stem); a component is the lines that share a redshift label. The banner and the
  Data tab's "!" ask the user to check the guess, and Q4.10 asked what the user may
  change before pressing Confirm. With your answer (datasets are never merged), I
  propose:
  - **systems:** change the z and the name, make primary, add and remove (the same
    actions as in a new project);
  - **rows:** rename only. A row follows the model's own resolution label, so moving
    a snip to another row would change the model rather than correct the guess; it is
    dropped, as is merging;
  - **Confirm** keeps the result in `ui/project.json`, and clears the banner and the
    "!".

My lean: both, as proposed.

**Response:** I agree with both, and thank you for clarifying!

**Q4.15 — Keeping the bad-pixel mask without the source (your Q4.11 response).** A
snip file keeps only the fit mask (the snip format does not change, D4), in which a
bad pixel is a 0, like any pixel outside the regions. So the bad pixels must be known
from elsewhere whenever regions are drawn:
- **(a)** from the source, read again each session: nothing new is stored, but when
  the source is missing (moved, or the project opened on another machine), a region
  drawn over a bad pixel would fit it;
- **(b)** from `ui/project.json`, where each snip's bad pixels are recorded (as
  wavelength runs) when it is cut: they are respected with or without the source.

In both, pixels with a NaN or an error ≤ 0 are found from the snip itself.

My lean: (b).

**Response:** Alternatively, we can store a bad pixel mask in the snip file itself, so that it is always carried with the snip. This is the easiest way to ensure the bad pixel mask is always available, and accurately flags the pixels that are bad. If a snip is loaded without the bad pixel mask labelled, then we should assume that all pixels are good.

**Q4.16 — The first guess of a new snip (your Q4.6 response).** Until the Components
tab (Stage 5) adds absorption lines, the model has none, so "±15 km/s of the model's
lines" leaves out nothing, and the transition just snipped pulls the guess down; the
uneven clipping (2.5σ below, 3σ above) only partly prevents that for a strong line.
Should the first guess also leave out ±15 km/s (the same preference) around every
transition of the systems' line IDs that falls in the snip, which includes the
transition snipped?

My lean: yes.

**Response:** Yes, this is acceptable, and still allows the user to make adjustments to the first guess if they feel it is not accurate enough. The user should be able to manually adjust the first guess if they feel it is not accurate enough.

*Raised by Claude on 2026-10-07 (Prompt 2), after reading the responses to
Q4.13–Q4.16. Those to Q4.13, Q4.14 and Q4.16 need nothing further: the knots only set
the starting coefficients (the manual adjustment of the first guess), the two kinds of
"structure" are handled as proposed, and the first guess also leaves out the systems'
line IDs. The response to Q4.15 raises one choice.*

**Q4.17 — How the snip's bad-pixel column is labelled (your Q4.15 response).** A snip
with a bad-pixel mask gets a fifth column (1 for a bad pixel), and a snip without one
has all its pixels good. ALIS reads a snip's columns one by one, by the numbers given
in the data line's `columns=`, so an extra column is harmless; but it stops on a role
it does not know. Two ways to label the column:
- **(a) In `columns=`, as a new role that ALIS understands:**
  `columns=[wave:0,flux:1,error:2,fitrange:3,badpix:4]`. This changes ALIS itself, in
  `load_data` and the two readers (`load_ascii`, `load_fits`): a pixel whose `badpix` is
  1 is never fitted, whatever its fit range says. A command-line run then respects the
  mask too, even after the fit-range column is edited by hand. No existing model uses
  the role, so the regression harness is unaffected.
- **(b) In a comment line at the top of the snip file** (`# wave flux error fitrange
  badpix`), which ALIS skips. ALIS does not change, and does not know which pixels are
  bad: only the dashboard keeps them out of the fit range.

In both, the dashboard also writes a 0 in the fit-range column of every bad pixel, so
the snip fits the same pixels with a version of ALIS that ignores the mask. The fifth
column is written only when the source spectrum has a bad-pixel mask; other snips keep
`prepfit`'s four columns (D4).

My lean: (a), so that the mask is labelled where ALIS reads every other column's role,
and every run of ALIS respects it.

**Response:** I agree with (a).

*Raised by Claude on 2026-10-07 (Prompt 3), with the review of the Data tab (Task
4.7).*

**Q4.18 — The Data tab, draft 1.** Its screenshots are published beside the mockups,
as a private page you can comment on: https://claude.ai/artifact/T4v5mJk179rDAqCjU1GJNe
(a new page, so that Stage 3's review page stays as it was). To decide:
- **(a)** Does the Data tab do what the mockups and Q4.1–Q4.17 asked? Anything to
  change before the Regions tab is built on the same spectrum view (Tasks 4.8–4.11)?
- **(b) Line IDs.** The strong lines are those with log10(fλ) + log ε − 12 ≥ −2 (the
  solar abundances `prepfit` ranks lines with; a preference). On J1358p6522 that
  labels about 30 lines and leaves 235 unlabelled ("+ 235 lines not labelled"; All
  lines labels them). More, fewer, or as it is?
- **(c) An imported fit's structure.** Rows are guessed from shared resolution labels,
  so a tie between two guessed rows would otherwise merge them. The first edit of a
  system or a dataset therefore keeps the structure as it then is, as Confirm does,
  and the banner and the Data tab's "!" go at that edit. Keep this, or keep the "!"
  until Confirm is pressed (the structure still being kept at the first edit)?

My lean: (a) your comments on the page; (b) as it is; (c) as built.

**Response:** I have provided comments on the page to respond to these queries.

*RJC's comments on the page (2026-10-07), recorded by Claude:*
- **(a)** "Overall, this looks good. I just have a few minor comments as feedback." On
  the preview of Remove…: "This screen probably needs a horizontal scrollbar, as
  well."
- **(b)** "The solar abundances are not currently included in the `atomic.ecsv` file
  correctly, and will need to be included. Currently, they are all just 0.0, so
  provided the solar abundance column of this datafile is interpreted as log epsilon,
  this should not change what elements are being displayed. This is an item to
  implement for future work."
- **(c)** "If something requires action, it should remain a "!" on that tab. It may
  not be intuitive to a user that they need to action something. Is there a help menu
  that appears if the cursor hovers over the "!" symbol? Ideally, pressing confirm
  should recheck everything and test if "!" is still the most appropriate symbol to
  show."

*Done by Claude on 2026-10-07 (Prompt 4); RJC asked for no further review of the Data
tab.*
- **(a)** The preview of a removal or merge (`PlanDialog`) gives each column the width
  of its contents, with a horizontal scroll bar.
- **(b)** The threshold stays at −2. The dashboard reads no abundance from
  `atomic.ecsv`: it uses Asplund et al. (2009), as `prepfit` does. Filling the
  column (as log ε) and reading it there is in `deferred_work.md` (§7.3).
- **(c)** The first edit of an imported fit still writes its guessed structure (so a
  tie never merges two rows), but marks it `guessed` in `ui/project.json`; the Data
  tab keeps its "!" and the banner stays until Confirm. The marker's tooltip (shown
  on hovering over the tab) says what to do: "check them on the Data tab, then press
  Confirm there." Confirm runs the full check of the model again, and every tab's
  marker is recomputed.

**Q4.19 — The Regions tab, draft 1 (Task 4.13).** Its screenshots are published
beside the mockups, with the screens that have no mockup and the Data tab as it is
now: https://claude.ai/artifact/HPnK8zEaFdACVwcGSKx2aT. To decide:
- **(a)** Does the Regions tab do what the mockups and Q4.1–Q4.18 asked? Anything to
  change?
- **(b) SNIP's starting width.** SNIP cuts ±300 km/s (a preference). For a transition
  in a wide absorption (J1358p6522's Lyβ, a DLA's Lyα) the snip then holds no
  continuum, and the first guess follows the absorption until the edges are dragged
  out (the review page shows Lyβ after its edges were moved to ±1500 km/s). Keep
  this, or start H I's strongest lines wider?
- **(c) The table of orders.** Every order is fitted to one set of continuum pixels,
  chosen by a clipped fit at the highest order of the first guess (5) (Q4.6). On a
  crowded snip that set favours the higher orders (J1358p6522's Ly7: χ² 2045 at
  order 4, 540 at order 5). Keep, or choose the pixels at the current order?
- **(d)** Drawing a region on one dataset copies it to the datasets with no regions
  yet (D18) in the same step; on a transition with no snip, Draw region SNIPs first
  (Q4.2(c)). As expected?

Also added while building, for your review: Delete removes the fit region last
clicked (the design's "Delete or the right-click menu removes it"), as a Regions tab
action; "=" raises the order as + does; SNIP is enabled only while a dataset whose
source covers the transition has no snip of it; a Regions "!" for a buffer narrower
than the resolution needs (as the validator warns).

My lean: (a) your comments on the page; (b) as it is, with the preference; (c) as it
is; (d) as built.

*Raised by Claude on 2026-10-07 (Prompt 4), with the review of Task 4.13.*

*RJC's comments on the page (2026-10-07; Prompt 5, written in `dashboard_stage3.md`),
recorded by Claude:*
- **(a)** "What is the difference between "Mask pixels" and "Draw region"? I think we
  should either remove "Mask pixels", or make it more clear what it is for." "Let's
  add "Ion" here as a label (similar to "Element") above, and have the ion (e.g. I,
  II, III, IV, etc.) as a drop down menu with < and > boxes to decrease and increase
  the ion stage with a button click?" "If some of these are continuum operations and
  others are panel operations, can we separate these and make it clear that there are
  panel operations (such as zooming, panning, snipping, clearing, drawing regions) and
  all continuum operations (add knot, normalised, etc.) in the dedicated Continuum
  panel".
- **(b)** "Keep the default. A user can adjust this as needed, and possibly add a knot
  near the middle of the wide absorption profile."
- **(c)** "The user can adjust the continuum and has the control to do that. I would
  recommend that we should perform a more robust first fit, where a low order
  polynomial is fit to the data first, reject pixels below the continuum defined with
  the low order polynomial, and then proceed to refit with a slightly higher order
  polynomial, and reject again pixels below, and then one more time increase the
  order of the polynomial by 1, and reject the pixels below. Based on these masked
  pixels, we can recalculate the continuum, and determine the optimal first guess of
  the order and BIC."
- **(d)** "SNIP cuts out the same transition from every dataset where there is
  wavelength overlap. Then, when applying regions to one file, the regions are
  identically applied to all files." With a question: Q4.20 below.
- Also: "I made the edits in the atomic.ecsv file, and I have now reverted this."

*Done by Claude on 2026-10-07 (Prompt 5); draft 2 of the page, at the same address.*
- **(a)** "Mask pixels" is now **Exclude pixels**, and "Draw region" **Add region**;
  they sit together under "Fit regions:", their tips say what each is for, and
  Exclude pixels acts on one dataset (a cosmic ray, a bad pixel) where Add region acts
  on all. An **Ion ▾** menu with ‹ › replaces the stage buttons. The tools are split:
  SNIP, CLEAR, Add region, Exclude pixels and the zoom bar above the spectra; Auto
  first guess, Add knot, Clear knots and Normalised in the Continuum box.
- **(b)** Kept at ±300 km/s.
- **(c)** The first guess's pixels are chosen in three passes of rising order (a
  straight line, then one order higher, then one more), rejecting the pixels far below
  (2.5σ) or above (3σ) each curve; every order is then fitted to the pixels left and
  the BIC chooses (`continuum.continuum_pixels`, `FIRST_ORDER`, `PASSES`). Each pass
  starts from the pixels the last kept but judges every pixel again: strictly
  cumulative, a straight line threw away the true continuum at the ends of a curved
  snip, and no later pass took it back (the unit test of a known order-2 continuum
  failed by 45σ). J1358p6522's Ly7 now reads χ² 494.0, 492.2 and 476.6 at orders 3,
  4 and 5 (2515.8, 2045.0 and 540.3 before).
- **(d)** One set of fit regions per transition: a region added, moved or removed on
  one dataset is added, moved or removed on every dataset, in one step
  (`snips.linked_step`, moved by the difference of the shifts); removing a region
  removes every region it overlaps in each dataset; pixels excluded in one dataset stay
  excluded there. "Copy HIRES regions to all" is now "Use HIRES's regions in every
  dataset", for an imported fit whose datasets have regions of their own.

**Q4.20 — A stacked spectrum of a transition (your Q4.19(d) comment).** You asked
whether to implement now, or in a later stage, an option to combine optimally the data
of one transition from all datasets, and to draw the regions and continuum on that
stacked spectrum, applied to every snip identically. My recommendation is a later
stage (after v1, or with the Plot tab), recorded in `deferred_work.md` §7.4:
- the datasets must be resampled onto one grid and weighted by their errors, though
  their resolutions differ (HIRES and the others in Q1243p307), and the stack is then
  not what ALIS fits;
- regions map back simply (they already do, through the shifts), but a continuum
  drawn on the stack has to be carried back to each dataset's own flux scale and
  shape (Q1243p307's HIRES is in counts of ≈1100, PROCHASKA ≈175), which is a
  continuum per dataset after all;
- Stage 4's linked regions already give "identical regions in every dataset", and
  Share with <row> gives one continuum shape for a transition in several datasets.

My lean: a later stage. Should it be part of v1 (say, Stage 5 or 6), or after?

*Raised by Claude on 2026-10-07 (Prompt 5).*

**Response** (RJC, on the review page, 2026-10-08): "Agreed, please make changes to the
overall plan to ensure that this is incorporated in a future stage (the best place to
do this is when we implement the Orders mode, and loading individual PypeIt spec1d
files)."

*Done by Claude on 2026-10-08 (Prompt 5 of this document):* D32 of
`ALIS_v2_dashboard_prompts.md` and its "After v1" list, the stage table and the
"Also later" line of `dashboard_stage0.md`, and `deferred_work.md` §7.4 place the
stacked spectrum with Orders mode and PypeIt spec1d input, after v1, offered in QSO Abs
Line mode too.

**Q4.21 — The Regions tab, draft 2 (RJC's comments, 2026-10-08).** Recorded by Claude:
- "Exclude is working as intended (it would be better if there was the functionality
  as in prepfit, that a left click and drag adds regions, while a right click and drag
  excludes regions). Let's have two buttons: Add regions to all (this works as the
  current "Add regions" does). It applies added regions to all datasets concurrently.
  Tweak dataset regions (this works like prepfit, and allows users to add/exclude
  regions from the currently active dataset)."
- The stacked spectrum: Q4.20's response above.
- Prompt 5 of this document: no further review of the Regions tab is needed.

*Done by Claude on 2026-10-08 (Prompt 5):* **Add regions to all** (`regions.draw`) and
**Tweak dataset regions** (`regions.tweak`, which replaces Exclude pixels). In both, as
in `prepfit`, a drag adds a fit region and a right-drag leaves pixels out
(`plots.SpectrumBox`: a right drag in draw mode gives a mask span); the first acts on
every dataset of the transition, the second on the selected dataset alone. Moving a
region's edges, Delete, and the right-click menu follow the tool on: the menu offers
"Remove this region from every dataset" and "… from this dataset only", and, while
tweaking, "Exclude this pixel". A step that leaves pixels out is named "Exclude n
pixels of …".

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-3; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`, `dashboard_stage3.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`,  `dashboard_stage3_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. If everything is clear, then please notify me when you are ready to start implementing Stage 4, and I will provide you with a list of the tasks to implement.

2. Please read the responses to your queries in the Queries section of this document, and provide any further queries you may have. If everything is clear, then please notify me when you are ready to start implementing Stage 4, and I will provide you with a list of the tasks to implement.

3. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order, with a pause at Task 4.7 as you recommended to provide design feedback on the html file. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

4. I have annotated some feedback directly on the html file. If you have any further queries about the implementation of the Data tab, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please incorporate my feedback about the data tab into the next draft (Given the feedback is so minor, there is no need for another review at this stage). Unless you feel there is something significant that turns up that I should provide further feedback on, I am happy for you to then proceed to execute the remaining tasks of this design document. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

5. I have annotated some feedback directly on the html file. If you have any further queries about the implementation of the Regions tab, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please incorporate my feedback about the Regions tab into the next draft (Given the feedback is so minor, there is no need for another review at this stage). Unless you feel there is something significant that turns up that I should provide further feedback on, I am happy for you to then proceed to execute the remaining tasks of this design document. If there are no more tasks to develop on this stage, please close this stage and generate a design document for stage 5. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).