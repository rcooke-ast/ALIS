# Dashboard Stage 9 log

The Figure tab (until now the Plot tab): a publication figure from a fit (D28, D42,
S22). The plan is in `claude_prompts/dashboard_stage9.md`.

### 2026-11-03 (Prompt 1: reading, the responses, and queries Q9.7–Q9.11)

**Read.**
- This stage's document, and RJC's responses to Q9.1 and Q9.3–Q9.6.
- Of `ALIS_v2_dashboard_prompts.md`: step (8), F8, S22, D24, D28, D34, D42–D44, D49,
  D77, D86, D88 and D89–D101, QF.10, QF.14 and QF.27.
- Of `dashboard_stage0.md`: the stages, "After v1", 0.8, the reviews of the Plot
  mockup, Q0.7 and Q0.10. Also the mockup's own text (`doc/dashboard/mockups/
  plot.html`).
- Of `dashboard_stage8.md` and its log: point 2 of the feedback, Q8.4, Q8.5, Tasks
  8.13 and 8.14, and the close, including the Ubuntu `gui` job's time.
- The pattern of earlier Prompt 1s (the logs of Stages 7 and 8), and every mention of
  the Plot tab or `plotscript` in the documents and logs of Stages 1–7 and in
  `deferred_work.md` (§1.1 and §5: the DH layouts never run on real data).
- The code the stage builds on: `alis/plotscript.py` and its call in `main.py`, the
  `plotscript` settings (`config.py`, `dashboard/settings.py`), `export.py` (`Choice`,
  `plan`), `history.py` (steps already carry `ui/project.json` and files), and
  `qt/tabs.py` (`PlotTab`).

**Facts that shaped the queries.**
- `DH_Lyseries.py` is the layout that Q9.4's response describes. It has three
  velocity panels; then one panel across all three columns (a `GridSpec(3, 1)` cell),
  the Lyman limit in rest wavelength over 913.15–915.2 Å, with a tick at each of 18
  Lyman lines rather than at each component; then three more velocity panels. Its
  axis titles are on chosen panels. It is the only reference script that draws
  against wavelength.
- The other 20 scripts write each axis title once for the whole figure, 14 on an
  invisible `add_subplot(111)` and 6 on a hidden subplot or with `fig.text`. They
  number the x axis on the bottom panels only (`hidex`), and the three metals
  scripts leave the middle column's y axis unnumbered (`hidey`).
- The scripts compute velocity from the uncorrected wavelength. Only the ticks would
  be off by a snip's shift, so Q9.8 (iii) proposes Export's corrected wavelength.
- The Plot mockup's tray moves a transition: "Drop on a cell to place a transition
  there (the one it replaces goes back here)". Q9.6's response allows the same
  transition twice, so Q9.10 proposes a copy drag.
- LaTeX (`latex`, `dvipng`) is installed on this Mac, so the no-LaTeX path of Task
  9.4 must be tested with `latex` hidden from the path.
- No code has changed since Stage 8 closed (`git diff HEAD`: only this stage's
  document). The baseline is therefore the close's count: `unit` 2130 passed and 120
  skipped, `gui` 229 passed, `fast` 149 passed (pytest 9.2.0).

**The responses.**
- Q9.1 (a), with the tab renamed Figure. Q9.3: no automatic redraw, only a Draw
  preview button. Q9.4: panels spanning columns within a row, an x axis for each panel
  (Velocity, Observed wavelength or Rest wavelength), and axis labels shown or hidden
  for each panel. Q9.5 (a). Q9.6 (a), with three additions: Export… writes everything
  a figure needs (its snips, the style, the script and a PDF); a project holds several
  figures (Create new figure, Delete); and a transition shown twice is allowed, but
  warned.
- **Q9.2 has no response.** Its lean, (a), stands. Task 9.4 waits on it.

**Written** in the document's Queries section, after Q9.6:
- a note that Q9.2 is still open;
- Claude's readings of the responses to Q9.1, Q9.3, Q9.4, Q9.5 and Q9.6, for RJC to
  correct;
- Q9.7 (a span per cell, or columns per row), Q9.8 (a panel's x axis: its range, rest
  wavelength, the snip's shift, the ticks), Q9.9 (shared and per-panel titles, and
  tick numbers), Q9.10 (what counts as the same transition, and how the warning is
  shown) and Q9.11 (exporting a figure, and the tab's button), each with a lean;
- a note at the top of the document.

The design and tasks are revised once Q9.2 and Q9.7–Q9.11 are answered. No code was
changed.

### 2026-11-03 (Prompt 2: the responses to Q9.2 and Q9.7–Q9.11, the design and tasks revised, Q9.12)

**The responses.**
- **Q9.2:** (b), the dashboard gets its own emitter, and `plotscript.py` stays for
  standalone use. RJC was open to reasons for (a); I gave none. A publication figure
  is made only in the dashboard, so one emitter shared with `run_alis` would gain
  nothing. D28's update of the `context/plotting/` references (QF.27) still applies,
  in comments only: `plotscript.py` lines 9 and 193, and `config.py` line 183 (the
  only three left).
- **Q9.7:** (a), a span per cell. **Q9.9:** (a), shared titles plus each panel's own
  titles and tick numbers. **Q9.10:** as proposed.
- **Q9.8:** (i)–(iv) as written, and three additions: an options button on each panel,
  holding all its options; Apply to all panels for the ranges, the tick numbers and
  the titles; and an invitation to propose further options from the reference
  scripts. RJC has added `context/misc/plotting_routines.py`, the module those
  scripts import.
- **Q9.11:** as proposed, with the figure's name in the file names, spaces as
  underscores (`<project>_Figure_1.py`).
- RJC did not correct any of the readings of Q9.1, Q9.3–Q9.6, so they stand.

**Read.** `context/misc/plotting_routines.py`: `plot_pm`, `plot_rmzerosign`,
`replot_ticks` and `turnoff`. Then, from the 21 reference scripts, what each sets per
panel or per figure, counting uncommented lines only:
- `plot_pm` in all 21; `tick_right` in 16; a second ion's ticks in 16 (D I in the H I
  panels; in `metals.py`, Si II 989.87 in N III 989.79's panel);
- panel titles in 12 (the datasets' names, and the blends' lines); component numbers
  above the ticks (`zabs_txt`) in 6; a Legendre re-normalisation (`coeff`, `scale`)
  in 3;
- per-panel y ticks and ticked components in `metals.py` (`tck`); the residual helpers
  `plot_devs` (strip) and `plot_residuals` (inside the panel, in the helium scripts).

Also checked: `export._safe` already turns spaces into underscores; `pyproject.toml`
packages `alis/data/*`, so the style file needs no packaging change; the preferences
folder is `~/.alis/` (`preferences.py`), so saved presets go in `~/.alis/figures/`.

**Revised** in `dashboard_stage9.md`:
- the header, and "What exists" (Q9.2's outcome; `plotting_routines`; `DH_Lyseries.py`
  now built by hand);
- the Design, rewritten from every response: the tab; the figures; the cells (spans,
  x axes, labels, the options button and Apply to all panels); arranging and
  duplicates; the presets; the script (`figscript.py`); the preview; Export; blinding;
- the Tasks, 9.1–9.8, each with its check. Blinding is now part of 9.1 (the tab greyed)
  and 9.7 (Export), and Export of figures is a task of its own (9.7). The table of
  elements now has a row for each response.

**Raised:** Q9.12, the further options. These are a panel's label corner, title, y
tick positions, right-hand numbers, and ticked components and ions; and a figure's
signed velocities, component numbers and spacing. Some things would be script
variables only (font sizes, line widths, dashes, tick heights, the residuals' offset
and scale). The re-normalisation is not carried over. Lean: as listed.

Once Q9.12 is answered, its options go into Tasks 9.2, 9.4 and 9.6, and the stage
waits for RJC's approval to implement. No code was changed.

### 2026-11-03 (Prompt 3: Q9.12's response, and the document made ready)

- **Q9.12's response:** the options as proposed. RJC will review them, and may add,
  change or remove some, once the stage is built and the scripts can be tried. With
  that, nothing is unclear, so the tasks run in order from 9.1.
- **Folded into the document:** Q9.12's options, in the Design (the cells' options,
  the figure's signed velocities and component numbers, and the script-only
  variables) and in the checks of Tasks 9.2, 9.4 and 9.6.
- **Two changes found while reading the code:**
  - The figures are kept in `ui/project.json`, under `figures`, rather than in a
    member of their own. Every kind of step the history makes already carries
    `ui/project.json` (`Step.ui`); a new member would need new code in `Project`,
    `Step` and `to_bundle` for no gain. Nothing else hashes `ui/project.json`, so
    editing a figure marks no other tab out of date.
  - Q9.8 (iii) corrects each panel's wavelengths for its snip's shift, so the script
    also reads Export's Corrected wavelength. That column joins the locked columns of
    Q9.11.

### 2026-11-03 (Prompt 3: Task 9.1, the Figure tab enabled once there is a best fit)

- **The rename (Q9.1).**
  - `actions.TABS` names the fifth tab Figure, and `view.plot` becomes
    `view.figure` (Ctrl+5, "Show the Figure tab"); it is no longer marked as coming
    later.
  - `qt/tabs.py`'s `PlotTab` gives way to `FigureTab` in a new `qt/figure.py`.
  - `markers.plot_marker` becomes `figure_marker`, and the docstrings of
    `modes.py` and `window.py` follow.
- **When it can be used** (`figure.unavailable`, new `alis/dashboard/figure.py`).
  The tab is greyed with the reason as its tooltip and its marker's tip, and is
  otherwise enabled:
  - "A figure for publication needs a best fit: run the fit first." This covers no
    run, a run that kept no `.mod.out` or no best-fit values, and the fit in
    progress;
  - "Unblind… first: a figure shows the hidden lines' profiles." This covers
    anything hidden (Export's own test, `export.blinded`).

  The tab draws the run chosen in the Fit tab's run history, never the fit in
  progress: `FitTab.committed_view` is new. Choosing a run there refreshes the
  Figure tab (`MainWindow.refresh_figure`), and a greyed tab that was shown gives
  way to the Fit tab. The tab names the run it draws.
- **Tests.**
  - New: `test_dashboard_qt_figure.py`, with 8 tests: a new project; a `.model`
    with a stored run; an imported fit with its `.mod.out` (metal_line_abs copied,
    its reference outputs as `.mod.out`); a run with no best fit; the run chosen in
    the history; `examples/blind` greyed, then enabled after Unblind…; one hidden
    line; no Plot named.
  - Updated: the Plot expectations of `test_dashboard_markers.py`,
    `test_dashboard_qt_window.py` (where the tab is enabled now depends on the run)
    and `test_dashboard_qt_markers.py`.
- **Batches** (pytest 9.2.0, in a scratch venv as CI runs them): `unit` 2130
  passed, 120 skipped; `gui` 237 passed (229 at Stage 8's close, plus the 8 new).

### 2026-11-03 (Prompt 3: Task 9.2, the figures, their cells and their steps)

- **`figure.py`** (no Qt) now holds the figures, in `ui/project.json` under
  `figures`. Every change is a `Project.set_ui` step, so undo is exact:
  - `new_figure` (with the layout of another figure, every cell emptied; or a
    preset's layout; or the defaults), `rename` (letters, digits, spaces, hyphens,
    underscores and dots; unique, ignoring case), `delete`;
  - `set_grid` (a smaller grid drops the cells that no longer fit and cuts a span at
    the new edge), `set_span` (it takes the cells it covers and gives back empty
    ones; it cannot pass the row's end);
  - `place` (from Not shown), `move` (a swap carrying label, title and ticks, or a
    copy), `remove` (a drag off the grid);
  - `set_option` (the figure's), and `set_cell` (a cell's, or with `every` Apply to
    all panels). A velocity range applied to all becomes the figure's and clears the
    velocity panels' own ranges. A wavelength range cannot be applied to all. A
    changed x axis turns on the panel's own x title (Q9.9);
  - `save_preset`, `load_preset` and `saved_presets`, in `~/.alis/figures/<name>.json`.
    Each cell's transition is kept by ion, wavelength and whether its system is the
    primary one, and is filled in another project from the first entry that matches.
- **What a cell can show** (`transitions`): for each snip with its pixels, each
  system's ions with a transition inside it (isotopes folded into their element). The
  transition used is the one the snip was cut for when it is of that ion, and
  otherwise the strongest inside the snip (as `components.ion_snips`, D61).
- **Duplicates and missing snips.** `same` matches the same transition of the same
  system from the same snip (Q9.10). `duplicates` and `elsewhere` name the panels
  ("Figure 1, row 2, column 3"). `missing` gives a note for a cell whose snip is
  gone; the stored cell is left as it is, so undoing the removal shows the panel
  again.
- **Numbering by place** (`numbers`): x numbers on a panel with no panel below it
  (or one with another axis or range); y numbers on the first and last panels of a
  row, the last numbered on its right; an empty cell counts as no panel.
- **The marker:** ○ while greyed or with no figure showing a panel; ! for a
  transition shown twice, naming the panels; ✓ otherwise, naming the figures.
- **Tests:** `test_dashboard_figure.py`, 15 tests, as Task 9.2's check lists them.
  The blind check compares the figures' numbers as whole words with the hidden
  values: "0.0" is a hidden value of `examples/blind`, and a substring match found it
  inside "0.05".

### 2026-11-03 (Prompt 3: Task 9.3, the presets)

- **`figpresets.py`** (no Qt): `layout(project, name, system, view, home)` builds
  metals, DH, blends and helium as the design says, or loads a saved preset.
  `apply` is the step that fills a figure, keeping its name and its velocity zero
  unless the preset chooses one (blends: the interloper's strongest component).
  `guess` picks the preset that suits a project (DH with H I and D I; helium with
  He I; metals otherwise), and `first_figure` creates Figure 1 from it. Sizes follow
  the reference scripts: 7.0 in wide, 1.5 in a row for metals, 9/7 in for DH, 3.6 in
  wide for blends, 2.5 in a row for helium.
- **On RJC's projects:**
  - J1358p6522 (one dataset) gives Lyα–Lyδ down the left column and Lyε–Ly8 down
    the right, 7.0 × 5.14 in;
  - Q1243p307 (three datasets) gives HIRES and PROCHASKA side by side, Lyα to Ly8,
    with the columns titled "HIRES data" and "PROCHASKA data".
- **Tests:** `test_dashboard_figpresets.py`, 9 tests (J1358p6522 and Q1243p307 are
  skipped where absent). The synthetic helium fit has three snips (10833, 3889 and
  3188 Å), and the synthetic blend has Si II of a second system inside a Lyα snip.
  The primary system is the one with the most ions, so the blend's H I system has
  D I too.

### 2026-11-03 (Prompt 3: Task 9.4, the script and the shipped style)

- **`figscript.py`** (no Qt), the dashboard's emitter (Q9.2 (b)):
  - `describe` builds the script's data: the velocity zero (the figure's component,
    or Export's default, D100) and each component's best-fit redshift (Export's
    `_reference_z`). For each panel it records the snip's Export file, its
    transition and redshift (an interloper's own strongest component), axis, ranges,
    words (plain and LaTeX), numbers and ticks. A velocity panel ticks every ion of
    its system (ratio ions such as D I included) with a transition in range, at each
    component's redshift. A wavelength panel ticks each transition of its own ion
    (Q9.8 (iv)).
  - `script` writes the text: a docstring with the commands; every colour, width,
    size, range and the details of Q9.12 as variables at the top; `COMPONENTS`,
    `Z_ZERO` and `PANELS` as data; then fixed drawing code that reads the files by
    their header names, normalises, draws data as steps, the model, the dashed
    levels, the residual strip or the residuals inside, the ticks, the labels and
    the titles. The helper `signed` plays the part of `plot_pm`; LaTeX is used when
    `latex` and `dvipng` are both found. The PDF is saved with no creation date.
  - `files` gives everything the script needs: the script, the snips' files it
    reads (Export's `tables` with the eight columns, Corrected wavelength included),
    and the style. `run` runs a script with Agg and reports a failure by its last
    line (D86).
- **The colours** were checked with the `dataviz` skill's validator, on white and
  over every pair, since ticks of different ions and the model can sit side by
  side. Vermillion (the model), bluish green and blue pass every check. Adding
  reddish purple drops the worst deuteranopia ΔE to 7.6, which the skill allows only
  with a second encoding that ticks lack. So two ions get colours, and a third
  ion's ticks are grey. Data, levels and residuals are drawn in neutral inks. The
  published scheme is the reference scripts' (r, g, b, dodgerblue, forestgreen).
- **Seen while checking:** labels in the lower corners sat in the residual strip,
  so they now sit just above the zero level (data y = 0.07, the reference scripts'
  `ytxt`). A panel followed by an empty cell was left unnumbered, so empty cells now
  count as no panel (see Task 9.2).
- **`alis/data/alis_publication.mplstyle`** is made from `context/misc/matplotlibrc`
  without the `webagg`, backend, `figure.autolayout` and `max_open_warning` entries,
  and with DejaVu Serif added at the end of `font.serif`. `pyproject.toml`'s
  `alis = ["data/*"]` already packages it.
- **ALIS outside the dashboard, comments only** (D28, QF.27): `plotscript.py`'s two
  `context/plotting/` references and `config.py`'s one now name
  `context/plotting_examples/`. `plotscript.py`'s docstring no longer says the GUI
  will drive its emitter. `test_plotscript.py`: 10 passed, unchanged.
- **Tests:** `test_dashboard_figscript.py`, 14 tests. The script of each preset runs
  under `python -I` with `latex` hidden from the path and imports only os, shutil,
  sys, numpy and matplotlib. LaTeX is tested where installed (here it is). The
  tests also cover a colour edited at the top, a figure read back from matplotlib
  (spans, ranges, numbers, titles; `DH_Lyseries.py`'s layout with a tick at each
  Lyman line in 913.15–915.2 Å; Q9.12's options), the PDF identical when drawn
  twice, the style packaged, and the LaTeX labels.
- **Timing here:** metal_line_abs's figure draws in 3.5 s to PDF with LaTeX and in
  1.4 s to PNG.

### 2026-11-03 (Prompt 3: Task 9.5, the preview)

- **`qt/figure.py`, the Preview pane.** Draw preview writes the figure's files
  (`figscript.files`) into the tab's own temporary folder. It runs the script there
  in a `QProcess`, with `MPLBACKEND=Agg`, into `preview.png` at the pane's width
  (`figscript.preview_dpi`, times the screen's pixel ratio), and shows the PNG. While
  the script runs, the button reads Stop, and Stop kills the process and keeps the
  last preview. Nothing redraws by itself (Q9.3). After a change the note reads "The
  figure has changed since this preview was drawn.", judged by comparing the figure
  and the run with those of the last drawing. A failure shows its message in the
  pane: the last line of the script's errors, or the reason `files` refused (a style
  file that cannot be read, no panel).
- **Tests** (`test_dashboard_qt_figure.py`, 5 more):
  - the preview equals, as a `QImage`, the PNG that the same script writes when run
    elsewhere at the preview's resolution;
  - a change marks the preview out of date with no process started, and a new
    drawing clears the mark;
  - Stop;
  - a style file deleted after it was chosen;
  - a script made to fail (the drawing patched to raise) shows "ValueError: Too
    small", and no window text holds "Traceback".

### 2026-11-03 (Prompt 3: Task 9.6, the tab)

- **The panes** (`qt/figure.py`, `FigureTab`):
  - **Layout:** the figure menu with Rename…, then Create new figure and Delete
    figure… (it asks first, and Undo takes the deletion back); the first figure
    comes from `figpresets.first_figure`. Below them: the preset with Apply
    (blends with a menu of the other systems) and Save as preset…; grid, size,
    spacing, velocity range, zero (Default, or a component); residuals, colours,
    style (ALIS publication, or Other file…, chosen through the window); shared x
    and y titles; Signed velocities; Number the ticks; Draw preview and Export
    figure….
  - **Panels:** the picture of the grid. Each cell shows its line, then its dataset
    with any span or wavelength axis, with ⚙ for its options and ! (with a tooltip)
    for a duplicate or a missing snip. Below is the Not shown tray, each entry
    naming the other figures that show it, and a note on the drags.
- **Drags (D42, Q9.10).** A cell is a drag source and a drop target; the tray takes
  a cell (off the grid) and gives entries. Ctrl or ⌥ held at the drop copies, and a
  transition shown elsewhere is said in the status bar.
- **The options dialog** (`OptionsDialog`). Every option of a cell, each change a
  step at once, and Apply to all panels beside the options marked ✱ (and beside the
  x range for a velocity panel). A refusal is shown in the dialog.
- **Every change** goes through `MainWindow._run`, and a `FigureError` is told as a
  message (D86).
- **Seen in the screenshots at 1440×900, and fixed.** With the `.mod` panel open the
  Layout pane is about 260 px wide, so labels and buttons were clipped and cells
  wrapped into tall columns. Now the figure buttons take two rows, the grid,
  size and spacing fields are compact with shorter labels, and the cells are fixed
  at 46 px, two lines elided, the text 10% smaller (the app's font is set in
  pixels). At three columns a cell still elides "O I 1302.2" to "O I 130…" while
  the `.mod` panel is open; the tooltip holds the whole name. Left for RJC's review.
- **Tests** (`test_dashboard_qt_figure.py`, 6 more):
  - figures created (the first from its preset), renamed through the dialog (a
    taken name refused), deleted, and brought back by Undo; the menu;
  - eight controls, eight steps, all undone;
  - the drags, including real `QDropEvent`s (a swap, and a copy with Ctrl held,
    warned in the status bar, the ! mark naming the other panel, the marker
    attention);
  - the options dialog (axis, ranges, title, ticks, corner, two Apply to all
    panels, a refused range);
  - a preset applied, saved, listed, and DH refused with its reason;
  - the panes in the mockup's order, with no design reference.

### 2026-11-03 (Prompt 3: Task 9.7, exporting figures)

- **`export.Choice.figures`** lists the figures ticked. **`figscript.plan`** gives
  each one's script, the snips it draws (left to Export when Export writes the snips
  itself), and the style once. A figure that cannot be written is refused with its
  name: no best fit, anything hidden, no panel.
- **The dialog** (`dialogs.ExportDialog`) lists the figures, the shown one ticked.
  While one is ticked, the columns its script reads are ticked and cannot be
  unticked (`locked`). With no best fit, or anything hidden, the figures are greyed
  with the reason. Export figure… opens it with only that figure ticked and nothing
  else.
- **The window** (`MainWindow.export`, which `do_file_export` calls) merges the plans
  and asks before writing over a file, the PDFs included. It then runs each script
  in the folder for its PDF; a script that fails leaves its files, and the message
  names it with its last line.
- **Tests:** `test_dashboard_figscript.py::test_exporting_two_figures` (names with
  underscores, the style and snip once, the snips left to Export).
  `test_dashboard_qt_figure.py`, 3 more: Export figure… writes the four files (the
  PDF a real one) with the columns locked, and asks before writing over; a failing
  script leaves its files; `examples/blind` with a run and a figure made by hand has
  the figure greyed in the dialog with the blinding reason, and `plan` refuses it.

### 2026-11-03 (Prompt 3: Task 9.8, the close: draft 1 of the review)

- **The review page**, draft 1: https://claude.ai/artifact/A3mGPjERKw8C2s52ecULv3.
  It is built by `doc/dashboard/stage9/take_screenshots.py OUT [SCREEN ...]` and
  `build_review.py SHOTS OUT.html`, and published from the session scratchpad's
  `stage9_review.html`. The page shows, at 1440×900 with the `.mod` panel closed as
  the mockup has it:
  - J1358p6522's DH figure beside the Plot mockup;
  - Q1243p307 (one dataset per column), tet02OriA (helium), metal_line_abs
    (metals) and a synthetic blend. The runs are the reference outputs, and every
    preview is drawn by the figure's script with LaTeX;
  - a second figure showing a transition the first shows;
  - `DH_Lyseries.py`'s layout, built by hand;
  - a panel's options;
  - Export figure…'s dialog, with the preview beside the exported PDF (rendered with
    `sips`);
  - `examples/blind`: the tab greyed, and the figure greyed in Export.
  - Three questions for RJC: narrow cells while the `.mod` panel is open; two ion
    colours, then grey; labels over a saturated trough.
- **Found while taking the screenshots, and fixed.** Cells of an earlier project
  stayed drawn under the new ones: `deleteLater` had not run, and a widget taken out
  of a layout still sits on its parent. They are now hidden and unparented first.
  Two tray entries with the same name (two snips holding O I 929.5) now name their
  snips. A size set in the screenshot script came back as 4.5 in; that was the
  script's own fault, since it made three whole-`ui` steps before applying any
  (noted in the `gui-component` skill).
- **The documents:**
  - `doc/ALIS_workflow.md` §4.3 has a Figure tab entry, and its tab list and
    "Version 1" note no longer say Plot;
  - `CHANGELOG.md` has the dashboard's Stage 9 and the shipped style;
  - `tests/README.md` has a Stage 9 row;
  - the `gui-component` skill has "Stage 9 additions";
  - D102–D109 are recorded in `ALIS_v2_dashboard_prompts.md`;
  - `dashboard_stage0.md` records Stage 9 built, with its review pending, and S22's
    preview off the "Later" list;
  - each task is marked done in this stage's document.
- **CI:** `.github/workflows/ci.yml`'s `gui` job now has 90 minutes rather than 60.
  At Stage 8's close, Ubuntu was 86% through 229 tests at 42 minutes, and the local
  `gui` batch has grown from 484 s to 550 s with this stage's window tests, which
  puts Ubuntu near 56 minutes. The YAML parses; RJC's push is the check.
- **Coverage of what changed** (the stage's own tests): `figure.py` 94%,
  `figpresets.py` 91%, `figscript.py` 94%, `qt/figure.py` 92%. Added:
  `test_another_system_s_panel_and_a_removed_snip`. An interloper's panel takes its
  own system's strongest component, and a panel whose snip was removed is left out
  with a note. Left untested: the cells' mouse-driven drag start (`QDrag.exec`
  blocks; the drop side is tested with real `QDropEvent`s), and the refusals of
  out-of-range sizes and spacings.
- **Pre-commit** passed on every changed file.
- **Batches** (pytest 9.2.0, in the scratch venv, as CI runs them), against the
  start of the stage (Stage 8's close):
  - `unit` 2170 passed, 120 skipped (2130 and 120);
  - `gui` 251 passed (229), in 9 min 36 s here (8 min 4 s after Task 9.1);
  - `fast` 149 passed (149), the regression harness included, in 17 min 40 s. No
    test of this stage is marked `fast`; ALIS outside the dashboard changed in
    comments, a data file and CI's time limit.
- **What remains:** RJC's review of the page, and the application of its comments
  (with the decisions they add after D109); RJC's push, to check CI on both
  platforms with the `gui` job's new limit; and RJC's trial of the scripts, for
  Q9.12's options. The stage closes then.

### 2026-11-03 (after Prompt 3: RJC's review of draft 1)

- **RJC's comments**, three threads on the page:
  - Question 1: when the Figure tab is selected, the `.mod` editor should be
    hidden, as in the screenshots;
  - Question 2: two ion colours, then grey, agreed;
  - Question 3: the labels' place is fine, since the corner can be changed for
    one panel or for all.
- **The change** (`qt/window.py`): `_tab_changed` hides the docked panel when the
  Figure tab is selected, and brings it back when another tab is. A panel the user
  opens meanwhile stays open, and a panel in its own window is left alone.
  `_remember_panel` does not record the tab's hiding, so the panel's saved state
  stays the user's. Test: `test_the_figure_tab_hides_the_mod_panel`; it and
  `test_dashboard_qt_window.py` and `test_dashboard_qt_modpanel.py` pass (49).
- **The page:** version 2 marks the three questions as settled or agreed, in their
  places so that the threads stay anchored. The three threads were answered and
  resolved. The decision is D110, refining D50.
- **Batch** after the change (pytest 9.2.0): `gui` 252 passed (251 plus the new
  test), in 9 min 4 s. `unit` is unaffected: 2170 passed, 120 skipped. RJC's fourth
  thread (the Q9.12 options, to be tried and commented on) was resolved with its
  auto-reply, and needed no change.

### 2026-11-03 (RJC: CI passes)

- RJC reports that every CI job passes on the push of Stage 9, so the `gui` job runs
  within its new 90-minute limit on both platforms. RJC will try the Figure tab and
  write feedback in the next prompt; the stage closes once that is answered.

### 2026-11-06 (RJC's question: merging a 3 × 2 grid's bottom three panels)

- **The question:** can the bottom three panels of a 3 × 2 grid be merged into one?
- **The answer, and an omission found.** The logic has had spans since Task 9.2:
  `figure.set_span`, tested, drawn by the picture of the grid and by the script.
  But no control in the window set them. The options dialog lacked Span, although
  Task 9.6's check ("the options popup sets each option") and the review page both
  implied it. The `DH_Lyseries.py` screen was made by calling `set_span` directly.
- **The fix** (`qt/figure.py`): Span is the first field of a panel's options, a spin
  box from 1 to the columns left in the panel's row ("1 of 3 columns"). Each change
  is one step (`FigureTab.set_span`); a refusal is said in the dialog. Test:
  `test_a_panel_spans_its_row_from_its_options`. RJC's case is checked: the bottom
  row of a 3 × 2 grid made one cell across three columns, the panel it covered sent
  back to Not shown, and a middle cell offered at most 2. The tab's tests pass (24).
- **The page:** version 3 has the options screenshot retaken with Span, and the
  section's text names it.

### 2026-11-06 (Prompt 4: RJC's feedback from using the tab)

- **1. A drop that raised "not enough values to unpack"** (a three-column
  rest-wavelength panel dragged onto a velocity panel).
  - *Cause:* the drop made its step at once, inside the drag. The step refreshed
    the window, and the refresh rebuilt every cell, deleting the cell the drag began
    on. That cell owned the drag (`QDrag(self)`) and so its data, while `drag.exec`
    was still running in it. The drop event then read empty data (on macOS it can be
    delivered again after the first). The autosave timer's refresh, landing during a
    drag, could do the same. The native drag cannot be run off-screen, so it was not
    reproduced here.
  - *Fix* (`qt/figure.py`): the drag belongs to the tab (`CellWidget.drag`) and is
    deleted after it ends. A drop reads its data at once (`_cell_of` and `_entry_of`
    return None for data that cannot be read) and makes its step with
    `QTimer.singleShot(0, …)`, once the drag has ended; the tray's drop does the
    same. Unreadable data are ignored, with "That drop could not be read: drag the
    panel again." in the status bar. `move` looks the cell up inside the guarded
    action.
  - *And what the drag does:* a moved panel now takes its own options with it
    (`figure.CARRIED`: transition, label, title, ticked ions and components, x axis,
    x and flux ranges, flux ticks, own x title, label corner). The span and the
    numbers, which follow the place, stay with the cell. So RJC's rest-wavelength
    panel stays one where it is dropped, rather than becoming a ±125 km/s velocity
    panel in the cell it lands in.
- **2. An x title cut off the bottom.**
  - *Cause:* the margins allowed for the numbers and the shared titles only. A
    panel's own x title in the bottom row (as when the panels' axes differ and there
    is no shared x title) fell below the figure.
  - *Fix:* the emitter works the margins out (`figscript.auto_margins`): 0.3 in more
    for an own x title in the bottom row, an own y title at either side, or a shared
    title, and 0.25 in for titles above the first row. Layout gains Margins:
    Automatic, or four values in inches (left, right, bottom, top), each change one
    step. The automatic values are shown (`figscript.margins_of`), and unticking
    Automatic keeps them as the figure's own. The script holds `MARGINS` and
    `TITLE_PAD` (the panels' own titles' padding, in points) at its top. A figure
    saved before `margins` reads with it as None (`figure._complete`).
- **3 and 4. A rest-wavelength panel ticked the velocity zero's component alone,
  and numbered none.** Both came from Q9.8 (iv) as written: a wavelength panel
  ticked each transition of its own ion at the zero's redshift, with no component,
  so Number the ticks had nothing to number. Now every panel, on any axis, ticks
  every component of every ion of its system with a transition in range, thinned
  by the panel's Ions and Components ticked options, and the ticks are numbered.
- **Decision:** D111 (refining D106 and D107). `doc/ALIS_workflow.md` §4.3 and
  `CHANGELOG.md` follow.
- **Tests:**
  - `test_dashboard_figure.py`: a dragged rest-wavelength panel spanning a row
    keeps its axis and ranges while its span stays (and a copy takes them); the
    margins option and its refusal; a figure saved without `margins`.
  - `test_dashboard_figscript.py`: a rest-wavelength panel ticks H I and D I at
    both components, at their places, numbered, and thinned to one component by its
    options. The margins keep a bottom panel's own x title inside the figure (read
    back from matplotlib), and set by hand they are the figure's.
  - `test_dashboard_qt_figure.py`: drops are made after the drag; the drag belongs
    to the tab and its data survive the cells being rebuilt; RJC's drop swaps the
    panels; unreadable data are ignored with a word; the Margins control.
  - The stage's tests pass: 17, 9, 18 and 26.
- **Batches** (pytest 9.2.0): `unit` 2174 passed, 120 skipped (2170 before Prompt 4,
  plus 4); `gui` 255 passed (252, plus 3 new, among them the span test of
  2026-11-06), in 8 min 56 s. Pre-commit passed. Not yet pushed: RJC's next push is
  CI's check.

### 2026-11-08 (Prompt 5: the flux numbers' side)

- **RJC's point:** "Flux numbers' side" (Automatic, Left, Right) should also offer
  Both (for a panel across all the columns) and None (for a panel in the middle).
- **The change.** The flux numbers' two options, Flux numbers (Automatic, Shown,
  Hidden) and their side (Automatic, Left, Right), become one: Flux numbers' side,
  Automatic, Left, Right, Both or None. None does what Hidden did, and Left, Right
  or Both what Shown did, so two controls would have allowed contradictions such as
  Hidden with Both.
  - `figure.py`: a cell's `yside` (None, or one of `YSIDES`) replaces `ynumbers`
    and `yright`, and is among the options Apply to all panels sets. `numbers()`
    returns `(x, side)`; Automatic is left on a row's first drawn panel, right on its
    last and none between, as before. `_complete` reads a figure saved with the two
    options as the side they meant (shown False → none; yright → right or left).
  - `figscript.py`: each panel's `yside` numbers the left, the right, both or
    neither side (`labelleft` and `labelright`). The margins leave room on the right
    for Right or Both.
  - `qt/figure.py`: the options dialog has one Flux numbers' side menu, with its
    Apply to all panels.
- **Decision:** D112 (refining D106).
- **Tests:**
  - `test_dashboard_figure.py`: the automatic sides, and Both and None set and
    refused; a figure saved with the two options;
  - `test_dashboard_figscript.py`: none, right and both read back from the drawn
    figure, with room on the right;
  - `test_dashboard_qt_figure.py`: the menu's five choices, the old two gone, one
    Apply to all panels.
  - The stage's tests pass (18, 19, 27). Pre-commit passed.
- **Batches** (pytest 9.2.0): `unit` 2176 passed, 120 skipped (2174, plus 2); `gui`
  256 passed (255, plus 1), in 9 min 1 s. Not yet pushed.

### 2026-11-08 (Prompt 6: Stage 9 closed, and Stage 10's document)

- **Closed** at RJC's request, after RJC committed every file. Every element of D28,
  D42 and S22, every response to Q9.1–Q9.12, the review's comments (D110) and RJC's
  feedback in Prompts 4 and 5 (D111, D112) have their changes logged as done, with
  their checks passing. The decisions are D102–D112.
- **Final batches** (pytest 9.2.0), against the start of the stage (Stage 8's
  close): `unit` 2176 passed, 120 skipped (2130 and 120); `gui` 256 passed (229);
  `fast` 149 passed (149). CI passed on both platforms for the stage's push (RJC,
  2026-11-03); the pushes since are RJC's.
- **Written:** `claude_prompts/dashboard_stage10.md`, modelled on Stage 8's document
  (itself modelled on Stage 7's), as Prompt 6 asks:
  - the Feedback section, empty, for RJC to fill in, with how each point is written
    (its kind, its tab, where it can be reproduced, D99's rule for users' files);
  - three items carried over from Stage 9: the DH preset's start member, the
    spacing below a middle row's own x title, and the untested paths;
  - how the feedback becomes tasks, and the closing task 10.N with its checks
    against Stage 9's close;
  - Q10.1–Q10.3 (the carried items; RJC's own feedback on the Figure tab in the
    list; a closed list, and what follows the stage), each with a lean.
- **Updated:** `dashboard_stage0.md` (Stage 9 done; Stage 10's document; "After v1"),
  and `deferred_work.md` (the Plot tab done; the GUI entry can be closed).
  `dashboard_stage9.md` records the close.
- Awaiting RJC's review and approval of Stage 10's document. No code was changed.
