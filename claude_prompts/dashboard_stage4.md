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
leans.*

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

### The spectrum view (`qt/plots.py`; D38, D43)

One widget for every spectrum in the dashboard, built on pyqtgraph:
- **The toolbar** (D38, Stage 0's draft 6): ⌂ home, ← back, → forward, ✥ pan, ⬚ zoom
  box, and + / − (zoom along the wavelength axis only). The wheel zooms the wavelength
  axis. Each view keeps its own history of zooms for ← and →.
- **Drawing:** the data as steps, centred on the pixels; the error, thin and grey;
  the continuum, dashed; the zero line, dashed grey. Large spectra (J1358p6522's
  source has 40,402 pixels) use pyqtgraph's downsampling and clip to the view.
- **Overlays** (an API the tabs use): fit regions (green), regions fitted by another
  snip (light blue), pixels fitted twice (hatched), masked pixels, the snip's edges
  (orange handles), line IDs (ticks and labels, coloured by kind as in the key), and
  a marker for a clicked point.
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

**An imported fit's structure** (Q2.4, Q3.16(c)). The systems and rows were
inferred; the Data tab says so in each pane, and the banner points to it. The user
edits them (Q4.10): systems as above; rows renamed, merged, or their snips moved to
another row (each moved data line takes the target row's FWHM and shift
parameters). **Confirm** (`project.confirm_structure()`) keeps the structure, clears
the banner and the Data tab's "!" (Stage 3's marker rule).

### The Regions tab (`qt/regions.py`; D36, S2, S5–S10, S26, D17–D19)

Left: **Transitions** and the **Key**. Centre: one spectrum per dataset, with the
tools above; below, the **Continuum** and **Snip** boxes for the selected dataset.

**Transitions** (S2, D36; `lines.py`):
- **System ▾** (the primary by default), **Element ▾** with ‹ ›, and the ion stage
  buttons I–IV. Only elements and stages with a transition on the data are offered.
- **The list:** that ion's transitions, ranked by fλ, each with its rest and
  observed wavelengths and its flags: **gap** (no good pixels near it), **edge**
  (near an end of the data), **forest** (Q4.5), **telluric** (Q4.5). Isotopes are not
  listed separately (D I is in H I's snips); transitions closer together than the
  FWHM share one row ("O I 971.7 ×2"). Only transitions that fall on the data are
  listed, and any that already have a snip (● in the list).
- **Keys** (single keys act only while the Regions tab has the focus, so typing in
  the `.mod` panel never triggers them): ↑ ↓ the next transition, [ ] the next
  element, − + the continuum's order.
- Without a source spectrum (an imported fit, D12), only transitions with a snip are
  listed, and SNIP is disabled ("needs the source spectrum").

**Key:** one item per line, as in the mockups: fitted pixels, fitted by another
snip, fitted twice, snip edges, continuum, and the three kinds of line ID.

**The spectra** (D36): one per dataset that covers the transition, stacked, never
overlaid, scrolling when there are more than three (a preference). The title names
the transition and the dataset ("Ly7 926.2 · J1358p6522_fluxcal.dat", or "O I 1302.2
· 3 datasets"). Each spectrum is centred on the transition at the system's z, ± the
snip's half-width; the bottom axis is wavelength, the top axis velocity (Q4.12).
Clicking a spectrum selects its dataset. Above them: Flux | Normalised (S9: divided
by the continuum, as ALIS's plots show it), **Draw region**, **Mask pixels**,
**SNIP**, **CLEAR**, and the toolbar.

**SNIP and CLEAR** (Q0.6; `snips.py`):
- **SNIP** cuts the transition from each dataset whose source covers it (Q4.2). Each
  snip is the source's pixels within ± the half-width (300 km/s, a preference),
  written in `prepfit`'s four-column format at full precision (Q2.7). Pixels with a
  NaN, an error ≤ 0, or a bad-pixel mask of 1 are never fitted (S7). Its data line
  follows Stage 2's template (`fitrange=columns`, `loadrange=all`, the row's
  resolution and shift); its continuum is the automatic first guess (S8); its specid
  is added to every absorption line whose transition falls in it (Q4.3). One step.
- **CLEAR** removes the transition's regions and snip in the selected dataset
  (`remove.plan_snip`, previewed as in S27). One step.
- Drawing a region on a transition with no snip makes it one, as SNIP (Q4.2).

**Regions and masks** (S6, S7, D18):
- **Draw region:** a drag across a spectrum adds a fit region (`project.set_regions`,
  which keeps the mask column character for character, Q2.7, and converts
  `fitrange=[lo,hi]` when needed, Q2.10). A region's edges can be dragged; Delete or
  the right-click menu removes it.
- **Mask pixels:** a click or a drag leaves those pixels out of the fit (their mask
  set to 0), splitting a region where needed.
- **Mask spikes** (S7): pixels more than 5σ above the continuum (a preference) inside
  the fit regions are masked, in one step, listed first. Pixels below the continuum
  wait for the model (S29, Stage 6; Q4.11).
- **Several datasets** (D18): regions drawn on one dataset are copied to the same
  transition in the others that have no regions of their own yet ("copied from
  HIRES"), then edited per dataset. "Copy HIRES regions to all" does it again on
  request.

**The snip's edges** (S6): orange handles. Moving one inwards keeps the lines of the
pixels that stay (`project.set_snip_edges`, Q2.7). Moving one outwards re-cuts the
snip from the source, keeping the old lines byte for byte and writing the new pixels
at full precision (Q4.9); without a source it is refused with a message.

**Continuum** (S8, D17, S9, S10; `continuum.py`):
- **The function** ▾: Legendre by default (D17); Chebyshev, Polynomial and Constant
  are offered too. Others are typed in the `.mod` panel.
- **The order,** − n +, also on the − and + keys: each change refits the coefficients
  to the continuum pixels, one step. Legendre stops at order 10, as ALIS does.
- **Auto first guess** (Q4.6): a σ-clipped least-squares fit to the snip's pixels,
  clipping more below the curve than above (absorption), leaving out masked pixels
  and the cores of known absorption lines. Its order is the one with the lowest BIC,
  up to 5 (a preference).
- **The table:** for the orders n − 1, n and n + 1, the χ² of the continuum pixels,
  Δχ² and ΔBIC against the current order (the mockup).
- **Dragging** (Q4.7): handles on the curve, one per coefficient, at fixed places
  (the Chebyshev nodes of the snip); dragging a handle moves the curve through it,
  the other handles staying where they are. One drag is one step.
- **Shared continua** (S10, Q4.8): a continuum that spans several snips (`specid=A,B`
  with `min=`/`max=`) is drawn across all of them, and the box says so. "Share with
  <the next snip>" joins two neighbouring snips of one dataset into one continuum;
  "Unshare" gives each its own again, refitted.
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
    one continuum refitted, the model lines' specids merged), previewed first. It is
    offered first when the overlap is large (a quarter of either snip's fitted pixels,
    a preference; D19). Overlapping snips are merged from their own files when there
    is no source (Q4.9).
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
  order in the first guess (5) and its clipping (2.5σ below, 3σ above); the spike
  threshold (5σ); the overlap from which merging is offered first (25%); which line
  IDs are shown (Q4.4).
- **New actions** (`actions.py`): the Regions tab's keys above, Identify a feature,
  SNIP, CLEAR, Draw region, Mask pixels, Mask spikes, Flux | Normalised, and Auto first
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

**4.1 — Transitions and line IDs (`lines.py`, no Qt; S2, S5, D36).**
- An ion's transitions on the data: rest and observed wavelengths, fλ, the ranking,
  isotopes folded into their element, near-coincident lines in one row, and the
  flags (gap, edge, forest, telluric).
- Line IDs for every system in a wavelength range (Q4.4); the transitions offered by
  Identify a feature, with the z each implies; the coverage grid.
- **Check** (unit tests): J1358p6522's H I list and its flags match the mockup
  (Lyβ to Ly10, "forest", Lyα "gap"); Q1243p307's coverage grid (16 transitions × 3
  files) matches the mockup; synthetic spectra for gap and edge; a click on the Lyβ
  of J1358p6522 offers z = 3.06726.

**4.2 — Snips (`snips.py`, no Qt; S6, S7, D18, D19, Q2.7, Q2.10).**
- Cutting a snip from a source (columns, bad pixels, NaN and non-positive errors),
  its file and specid names, the SNIP step (data line, file, continuum, specids in
  the absorption lines, `ui/project.json`), CLEAR, extending the edges from the
  source, masking pixels and spikes, copying regions between datasets, the buffer,
  and the two fixes for shared pixels (keep in one snip, merge).
- **Check** (unit tests): after every step, ALIS reads the model and its data
  (`validate.check`, with no fit); one undo restores the text, the files and
  `ui/project.json` byte for byte; lines of a snip that stay are unchanged; after
  "keep" or "merge", `load.find_shared_pixels` finds nothing; on the
  `metal_line_abs` spectrum, the O I 1302 and Si II 1304 snips overlap and merge
  into one.

**4.3 — The continuum (`continuum.py`, no Qt; S8, S10, D17).**
- Evaluating a continuum line with ALIS's own function on the pixels ALIS loads; the
  first guess; the order table; changing the order; the handles and the drag;
  sharing and unsharing.
- **Check** (unit tests): the curve equals ALIS's own evaluation, to 1e-12, for every
  emission line of the `examples/` models; on synthetic data the first guess recovers
  a known polynomial under absorption lines, and the BIC picks its order; dragging a
  handle moves the curve through it and keeps the others; a shared continuum reads in
  ALIS with its `min=`/`max=`.

**4.4 — Datasets and systems (no Qt; S25, D14–D16, Q0.8, Q2.4).**
- The edits behind the Data tab that do not yet exist: changing the reference, the
  zero level on and off, adding a file as a row, moving snips between rows, merging
  rows, and the systems (z, add, rename, make primary, remove); `confirm_structure`.
- **Check** (unit tests): each is one step that ALIS reads, and one undo restores
  it; the reference's shift is fixed at 0 and the old one's freed; a moved snip takes
  its new row's FWHM and shift; removing a system lists its lines first.

**4.5 — The spectrum view (`qt/plots.py`; D38, D43).**
- The pyqtgraph widget, its toolbar and history of zooms, the overlays, Flux | Log
  and Flux | Normalised, colours from `qt/style.py`.
- **Check** (`gui` tests): home, ← →, the zoom box, and + − along the wavelength axis
  only; J1358p6522's 40,402 pixels draw in under half a second off-screen; nothing
  in the view's tooltips holds a hidden value.

**4.6 — The Data tab (`qt/data.py`; D35).**
- Systems with Identify a feature, Blinding, Coverage, the Datasets table with its
  badges, the spectrum, an imported fit's structure and Confirm, and
  cross-highlighting.
- **Check** (`gui` tests): on a new project from `OI_SiII.dat`, Identify a feature
  sets the primary z; on Q1243p307, a tie between two rows reaches the model and one
  undo removes it; Confirm clears the Data tab's "!" and the banner; on
  `examples/blind`, no hidden value in the tab; screenshots at 1440×900 beside the
  mockups.

**4.7 — Review of the Data tab (Q4.1).**
- Publish the Data tab's screenshots beside its mockups, on the page used in Stage
  3 (https://claude.ai/artifact/A6ohUEnAe91TEDsKBLwuBt) or a new one, for RJC's
  comments, and apply them before the Regions tab is built on the same view.

**4.8 — The transition list and the key (`qt/regions.py`; S2, D36).**
- System, Element ‹ ›, the stage buttons, the ranked list with its flags and ●, the
  key, the keys, and the tab-scoped shortcuts.
- **Check** (`gui` tests): J1358p6522's H I list as in the mockup; ↑ ↓ and [ ] while
  the tab has the focus, and no shortcut taken while typing in the `.mod` panel;
  without a source, only snipped transitions are listed and SNIP is disabled.

**4.9 — The spectra and their tools (S6, S7, S9, D18; Q0.6).**
- One spectrum per dataset, Draw region, Mask pixels, Mask spikes, SNIP, CLEAR, the
  edge handles, Flux | Normalised, the line IDs, and copying regions.
- **Check** (`gui` tests): a drawn region reaches the snip's mask column and one undo
  restores the file; SNIP then CLEAR leaves the project as it was; on Q1243p307,
  a region drawn on HIRES is copied to PROCHASKA and KIRKMAN; an edge dragged
  outwards on J1358p6522 re-cuts the snip from its source.

**4.10 — The Continuum box (S8, S9, S10, D17).**
- The function, the order with − + and the keys, Auto first guess, the table of
  orders, the handles, and sharing.
- **Check** (`gui` tests): + then − gives back the same text; a drag is one step; a
  continuum hidden with Blind lines… is drawn, and its coefficients are never shown.

**4.11 — The Snip box (S6, S26, D19).**
- The extent, the fitted pixels, the buffer, the pixels fitted twice and the two
  fixes (for all datasets at once), and the table of datasets.
- **Check** (`gui` tests): J1358p6522's Ly7 shows the mockup's 36 shared pixels (18
  with Ly8, 18 with Ly6); "keep" clears them and the Regions tab's "!"; on
  Q1243p307, merging O I 1302 and Si II 1304 is offered first and applies to the
  three datasets.

**4.12 — From a spectrum to a fit (end to end).**
- A `fast` test: New project from `examples/metal_line_abs/data/OI_SiII.dat` (z = 0,
  its fourth column ignored, as the example's model reads it), driven through the
  Data and Regions tabs (SNIP O I 1302 and Si II 1304, merge, the fit region
  1301–1305 Å, the first-guess continuum), with the two absorption lines added by
  `edit.add_ion` (their panel is Stage 5's). `run_alis project.model` then fits, and
  the column densities agree with `fit_spectra.mod.out.reference` within their errors.

**4.13 — Close the stage.**
- Publish the screenshots of both tabs beside the mockups for RJC's review, as in
  Stage 3, and apply the review.
- `doc/ALIS_workflow.md`: a short section on preparing a fit with the Data and
  Regions tabs. Also update `CHANGELOG.md`, `tests/README.md` and the `gui-component`
  skill (how to use the spectrum view).
- Run `test-coverage` on the new modules, and the `unit`, `gui` and `fast` batches.
- Record in this document what Stage 5 receives, and update the stage table in
  `dashboard_stage0.md` if anything moved.

## Status

*To be written by Claude when the stage closes.*

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

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-3; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`, `dashboard_stage3.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`,  `dashboard_stage3_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order until Task 4.7, where we will pause for a review (see my response to Q4.1). If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).