# Prompt file for ALIS software dashboard creation -- STAGE 9

> **The Figure tab** (the Plot tab until Q9.1). A publication figure from a fit: a
> grid of panels, presets for metals, D/H, blends and helium, a script written for the
> user to edit, and a preview drawn by running that script (D28, D42, S22). Point 2 of Stage 8's
> Feedback asked for the tab to be enabled once a fit has been made, and for its
> design to be written as this document (Q8.4, Q8.5). Feedback that arrived after
> Stage 8's list closed goes to Stage 10.
>
> The rules of Stages 3–8 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only where
> a query agrees.
>
> "D*n*", "S*n*" and "QF.*n*" refer to `claude_prompts/ALIS_v2_dashboard_prompts.md`;
> "Q0.*n*" to "Q8.*n*" to `dashboard_stage0.md` to `dashboard_stage8.md`.

*Written by Claude on 2026-11-03 (Stage 8, Task 8.13). No code was changed. The
queries at the end are for RJC; the tasks are revised once they are answered.*

*2026-11-03 (Prompt 1): RJC has answered Q9.1 and Q9.3–Q9.6. After Q9.6 come Claude's
readings of those answers and new queries, Q9.7–Q9.11. The design and the tasks will
be revised once these and Q9.2 are answered.*

*2026-11-03 (Prompt 2): RJC has answered Q9.2 and Q9.7–Q9.11, and the Design and
Tasks are revised from all the responses. Q9.12 (the further options that RJC's
response to Q9.8 asked for) is the one query still open.*

*2026-11-03 (Prompt 3): Q9.12 answered; Tasks 9.1–9.7 done, and Task 9.8's review
published (draft 1): https://claude.ai/artifact/A3mGPjERKw8C2s52ecULv3. The stage
closes once RJC has reviewed the page and CI has passed.*

## What exists

- **The decisions.**
  - **D28:** a general grid of panels, with presets for metals, D/H, blends and helium
    (QF.14, QF.27).
  - **D42:** the layout (preset, grid, figure size), with transitions dragged onto a
    picture of the grid: between cells to swap them, off the grid back to "Not shown";
    residuals shown and colour-blind-safe colours by default; a script that sets every
    colour as a variable at its top and is edited outside the dashboard (no script
    box); the publication style shipped with ALIS (from `context/misc/matplotlibrc`,
    for the script alone) or another style file; LaTeX text where LaTeX is installed,
    matplotlib's own text otherwise (Q0.7, Q0.10).
  - **S22:** the preview is the output: it is drawn by running the script that is
    written, so it is exactly what the script draws. Layouts can be saved as presets.
  - **D24** (blinding), **D43** (the look) and **D44** (no design references in the
    window) apply as on every tab.
- **The mockup** (Stage 0, `doc/dashboard/mockups/plot.html`, draft 4). At the left,
  Layout: Preset (metals · DH · blends · helium), Grid (columns × rows), Figure size
  (in), Velocity range, Residuals, Colours, Style (ALIS publication, Other file…), and
  Write plotting script…. In the middle, the preview. At the right, Panels · drag to
  arrange: the cells with their transitions, and "Not shown".
- **The emitter.** `alis/plotscript.py` writes a standalone Python 3 script from an
  ALIS run (`write_plotscript(slf, params)`), with the `metals` layout (three columns,
  the transitions of an ion adjacent) and `DH` (two columns: one dataset down both, or
  one dataset per column). It reads ALIS's `_fit.dat` files, inlines the few helpers
  the reference scripts take from `plotting_routines` (not distributed), and still
  refers to `context/plotting/` (QF.27: to be updated when this step begins). By
  Q9.2's response, it stays as it is for `run_alis`, and the dashboard gets its own
  emitter.
- **`plotting_routines`** (`context/misc/plotting_routines.py`, added by RJC for
  Q9.8): the module that all 21 reference scripts import, which users do not have.
  The scripts use three of its helpers: `plot_pm` (a + on positive tick numbers),
  `replot_ticks` and `turnoff` (hiding the invisible axis that carries the shared
  titles).
- **The reference scripts** in `context/plotting_examples/` (21 scripts, 6,284 lines),
  matched to presets in the table below. What they share: a velocity axis from a
  component's redshift; data as steps in black and the model in red; dashed lines at
  the continuum (1) and the zero level (0); a tick above the continuum for each
  component, coloured by ion (H I red, D I green); a label in each panel; the axis
  labels written once, on an invisible axis over the whole figure; serif text with
  LaTeX.
- **The style.** `context/misc/matplotlibrc`: serif Computer Modern, bold, 14 pt,
  `text.usetex: True`, with backend and web-server entries that a style file does not
  take. Q0.7's response: it is for the script alone, never the dashboard.
- **What Stage 7 built that this one uses.** Export (D88) writes each snip's file with
  the columns chosen, among them the best-fit model, the continuum, the zero level,
  the fit regions, the velocity from a chosen component and the corrected wavelength,
  through the gate. The script can read those files rather than ALIS's own.
- **The tab now.** `qt/tabs.py`, `PlotTab`: shown, disabled (D42, D77).

### The reference scripts and their presets

| Script | Layout | Preset | What the preset takes from it |
|---|---|---|---|
| `metals.py` | 3×3, 7.0×4.5 in, ±125 km/s | metals | the grid, the per-panel flux range, ticks chosen per panel |
| `metals_J0903.py` | 3×3, 7.0×4.5 in | metals | the same, another absorber |
| `metals_v2.py` | 3×3, 7.0×4.65 in, a residual strip | metals | the residual strip below the data (`plot_devs`) |
| `DH_Lya-Ly7.py` | 7×2, 7.0×9 in, ±190 km/s, one dataset per column | DH | one dataset per column, each titled; H I and D I ticks |
| `DH_Lya-Ly7_J1332.py` | 7×2 | DH | the same |
| `DH_Lya-Ly7_J1358.py` | 7×2 | DH | the same; the style of the Stage 0 preview (0.8) |
| `DH_Lya-Ly7_v2.py` | 7×2 | DH | the same, with residuals |
| `DH_Ly8-Ly14.py` | 7×2, Ly8–Ly14 | DH | the higher members, with a start member chosen |
| `DH_Ly8-Ly14_J1332.py` | 7×2 | DH | the same |
| `DH_Ly8-Ly14_J1358.py` | 7×2 | DH | the same |
| `DH_Ly8-Ly14_v2.py` | 7×2 | DH | the same, with residuals |
| `DH_Lya-Ly14.py` | 7×2, one dataset down both columns | DH | the series continuing from the left column to the right |
| `DH_Lyseries.py` | mixed: 3×3 and 3×1 grids on one page | none | built by hand: a panel spanning three columns, in rest wavelength (Q9.4, Q9.7, Q9.8) |
| `blends_1.py` … `blends_4.py` | 7×1, 3.6×9 in, an asymmetric range | blends | one column of an interloper system's lines, titled with the line it blends, a residual strip |
| `model_fits.py` | 2×2, 7.0×5 in | helium | residuals inside each panel, offset below the data; the axis named from the strongest line |
| `model_fits_HD319718.py` | 2×2 | helium | the same |
| `model_fits_Her36.py` | 2×2 | helium | the same |
| `model_fits_tet02OriA.py` | 2×2, 7.0×7 in | helium | the same, taller |

## Design

*Revised on 2026-11-03 (Prompt 2), from RJC's responses to Q9.1–Q9.11. The options
that depend on Q9.12 are marked as such.*

**The tab (point 2; Q9.1, Q9.5).** The Plot tab is renamed Figure, both in the window
and in the code (`FigureTab`, `qt/figure.py`). It is enabled once the run the Fit tab
shows (the run chosen in its history, the latest by default) has a best fit. That
holds whatever opened the project: a `.model` with a stored run, an imported fit with
its `.mod.out`, or a new project that has run. Every figure draws that run, as Fit ·
Results does. Otherwise the tab is greyed, and its tooltip gives the reason:
- with no best fit: "A figure for publication needs a best fit: run the fit first.";
- while anything is hidden (the whole analysis or one line): "Unblind… first: a
  figure shows the hidden lines' profiles." Figures made before the blinding stay in
  the project, unchanged, and come back on unblinding.

**The figures (Q9.6; no Qt, a new `alis/dashboard/figure.py`).** A project holds any
number of figures, kept in its bundle in `ui/project.json` (under `figures`, so the
steps the history already makes carry them). Each change is one step of the history,
and undo is exact. A menu at the top of Layout chooses the figure shown,
and three buttons sit beside it:
- **Create new figure** names the new figure Figure 2, Figure 3, and so on. It copies
  the shown figure's layout (grid, spans, axes, labels, size, spacing, ranges,
  residuals, colours, style), with every cell empty and every transition in Not shown.
- **Rename…** takes a name made of letters, digits, spaces, hyphens, underscores and
  dots, unique in the project. The exported files are named after it (Q9.11).
- **Delete figure…** asks first, and is one step that Undo takes back.

Each figure holds:
- the grid (columns × rows), the figure size (in), the spacing between panels, the
  velocity range (km/s), and the velocity zero. The zero is a component, by default
  the one Export measures velocities from (D100);
- the residuals (none, a strip below the data, or inside each panel as in the helium
  figures); the colours (colour-blind safe by default, or the published red and
  green); the style (the shipped one, or a file chosen); the shared axis titles
  (Q9.9); signed velocities (+100, on by default) and the components' numbers above
  their ticks (off by default; Q9.12);
- its cells (below), and the preset it came from.

A layout can be saved as a preset of the user's own (S22). It is kept in
`~/.alis/figures/`, so every project can use it. A saved preset keeps the layout and
each cell's transition, by ion and wavelength. Loaded in another project, it fills the
cells whose transitions that project has, and leaves the others empty.

**The cells (Q9.4, Q9.7–Q9.9).**
- **What a cell shows:** one transition of a system, in one snip (held by the snip's
  identity), with its label. A cell may be empty. A cell whose snip is removed is
  emptied, with a note.
- **Spans (Q9.7 (a)).** A cell spans from 1 to all of the grid's columns within its
  row, and a panel is never more than one row high. The cells a span covers leave the
  row, and their transitions go back to Not shown. A grid made smaller does the same
  with the cells that no longer fit.
- **The x axis (Q9.8).** Each panel has one of three x axes:
  - Velocity, the default. A velocity panel takes the figure's velocity range unless
    it has its own.
  - Observed wavelength, or Rest wavelength (the observed wavelength over 1 + z of the
    velocity zero). A wavelength panel has its own range in Å; by default, the
    figure's velocity range converted at its transition.

  Wavelengths are corrected for the snip's shift, as Export's Corrected wavelength is.
  A velocity panel ticks every component. A wavelength panel ticks each transition of
  its ion within its range, at the velocity zero's redshift.
- **Labels (Q9.9 (a)).** The figure has shared x and y titles, on by default. Each
  panel has its own x and y titles, off by default, and its tick numbers on or off for
  each axis; by default, each panel is numbered as its preset's reference scripts
  number it. A panel whose x axis differs from the figure's turns its own x title on.
- **The options button (Q9.8's response).** Each cell of the Panels picture has an
  options button (⚙). It opens all that panel's options in one place: its x axis and x
  range, its y range, its titles and tick numbers, and Q9.12's options: its label's
  text and corner, a title above it, its y tick positions, its y numbers on the right,
  and which components and ions are ticked (by default every component of every ion
  of the system with a transition in its range, each ion in its own colour).
  Beside each option that can be shared, Apply to all panels sets it on every panel of
  the figure, as one step. That covers the y range, the titles, the tick numbers,
  the x axis, the label's corner, the y tick positions and the y numbers' side. A velocity range is applied to every velocity panel: it becomes the
  figure's velocity range, and the panels' own ranges are cleared. A wavelength range
  cannot be applied to all, since each wavelength panel covers its own lines.

**Arranging (D42, Q9.10).** Transitions are dragged onto the picture of the grid in
the Panels pane:
- a drop on a cell places the transition there, and the one it replaces goes back to
  Not shown;
- a drag between cells swaps them;
- a drag off the grid sends a transition back to Not shown;
- a drag with the copy key held (Ctrl, or ⌥ on macOS) copies instead of moving.

Each figure's Not shown tray lists the transitions it does not show, each marked with
where else it is shown ("· Figure 1"). The same transition of the same system from
the same snip may be shown twice, in one figure or across figures, but with a warning:
a message in the status bar when it is placed, and a mark on each such cell. The
mark's tooltip names the other panels, and it stays for as long as both are shown.
The same line from another dataset's snip does not count as the same; the DH preset
shows it once per dataset.

**The presets (D28).** Each fills the shown figure from the project as its reference
scripts do (the table above), and sets their labels and numbering:
- **metals:** three columns, as many rows as needed, with the transitions of each ion
  adjacent, ordered by ion and wavelength; ±125 km/s.
- **DH:** two columns. With one dataset, the Lyman series runs down the left column and
  on down the right. With two, there is one dataset per column and one member per row,
  each column titled. With more, it takes the first two, and the user drags in others
  (`plotscript.py`'s rule). H I and D I ticks are in their own colours.
- **blends:** one column of a chosen system's lines (an interloper), each titled with
  the line of the primary system it blends, and a residual strip.
- **helium:** two columns of He I lines, with residuals inside each panel; the velocity
  axis is taken from the strongest line of the triplet.

`DH_Lyseries.py` has no preset. With a span and a rest-wavelength panel, the user
builds its layout by hand, and can save it as a preset of their own.

**The script (Q9.2 (b); no Qt, a new `alis/dashboard/figscript.py`).** The dashboard
has its own emitter. `alis/plotscript.py` stays as it is, for `run_alis`'s
`plotscript` settings, except that its references to `context/plotting/` become
`context/plotting_examples/`, as do `config.py`'s (comments only; D28, QF.27). The
script:
- is self-contained Python 3, needing only numpy and matplotlib. The helpers the
  reference scripts import from `plotting_routines` (`context/misc/plotting_routines.py`,
  which is not distributed) are written into it: the signed tick numbers of `plot_pm`,
  `replot_ticks` and `turnoff`;
- sets every colour, line width, font size and range as a named variable at its top
  (D42), with the figure's details that are not in the window (Q9.12): the dashes of
  the continuum and zero lines, the ticks' heights, the residuals' offset and scale,
  and the spacing of the x ticks;
- reads the snips' files that Export writes beside it (D88), finding their columns by
  the names in their headers, so it runs anywhere, without ALIS or the project;
- uses the shipped style, `alis/data/alis_publication.mplstyle` (made from
  `context/misc/matplotlibrc` without its backend and server entries; packaged by
  `alis/data/*`), written beside it, or else the style file chosen;
- turns `text.usetex` on when `latex` is found on the path and off otherwise, with the
  labels written to read well either way (Q0.10);
- writes its PDF with no creation date, so the same figure always gives the same file.

**The preview (S22, Q9.3).** Draw preview runs the script in a separate process, with
the Agg backend, writing a PNG at the preview pane's size; the tab shows that PNG.
While the script runs, the button reads Stop. Nothing redraws by itself. After a
change, the last preview stays, marked "The figure has changed since this preview was
drawn." Before the first draw, the pane says what the button does. A script that fails
shows its last line in place of the preview, never a traceback (D86).

**Export (Q9.6, Q9.11).** File → Export… gains The figures, which lists each figure by
name, with the shown figure ticked by default. For each figure ticked, it writes:
- `<project>_<figure>.py` and `<project>_<figure>.pdf`, with the figure's spaces as
  underscores (`J1358p6522_Figure_1.py`). The PDF is drawn by running the script in the
  folder (S22);
- the snips' files the figure draws, under Export's names, whether or not the snips
  are ticked. They hold the columns the script reads (Wavelength, Flux, Error, Fit
  range, Zero level, Continuum, Best-fit model, and the Corrected wavelength that
  Q9.8 (iii) needs), which cannot be unticked while a figure is ticked;
- the style file, written once for all the figures.

If a script fails, the files it needs stay written, no PDF is drawn, and the message is
the script's last line (D86). The tab's Export figure… opens the same dialog, with only
its own figure ticked. The figures is greyed, with the reason, while anything is hidden
or while the shown run has no best fit.

**Blinding (D24, Q9.5 (a)).** While anything is hidden, the tab and The figures are
greyed (above), so neither a figure nor its files can show a hidden line. A figure's
description holds no parameter value: its velocity zero is a component, and its ranges
are the user's. The figures in `ui/project.json` therefore have nothing to hide.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage9_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests
> (also under CI's pytest, memory `ci-test-versions`); run the `fast` batch before
> the stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness.

*Revised on 2026-11-03 (Prompt 2), with the design. Q9.12's options are in Tasks 9.2,
9.4 and 9.6 (Prompt 3); RJC will review them once the stage is built.*

**9.1 — The Figure tab, enabled once there is a best fit (point 2 of Stage 8; Q9.1,
Q9.5).**
- The Plot tab is renamed Figure: the tab, its tooltip, the shortcut sheet, and
  `FigureTab` in the code. It is enabled when the run the Fit tab shows has a best
  fit, and follows that run. Otherwise it is greyed, with the reason as its tooltip:
  no best fit, or anything hidden.
- **Check.** `test_dashboard_qt_figure.py`:
  - a new project is greyed, with the reason;
  - a `.model` with a stored run (metal_line_abs) is enabled, and so is an imported fit
    with its `.mod.out`;
  - `examples/blind` with a run is greyed with the blinding reason, and is enabled after
    Unblind…; a single hidden line also greys the tab;
  - the run chosen in Fit's history is the one the tab names;
  - the window's text no longer names a Plot tab.

*Done on 2026-11-03 (Prompt 3): `qt/figure.py`, `figure.unavailable`, `FitTab.committed_view`; 8 window tests. See the log.*

**9.2 — The figures, their cells and their steps (D28, D42, S22; Q9.6–Q9.10;
`figure.py`).**
- What this task builds:
  - several figures in `ui/project.json`, created (layout copied, cells empty),
    renamed and deleted;
  - each figure's grid, spans, size, spacing, ranges, x axes, titles and numbers,
    residuals, colours and style;
  - the cells' options (with Q9.12's label corner, title, y tick positions, y
    numbers on the right, and ticked components and ions) and Apply to all panels;
  - the figure's signed velocities and component numbers (Q9.12);
  - placing, swapping, removing and copying transitions;
  - the duplicates, and where else each transition is shown;
  - a saved preset, in `~/.alis/figures/`.
- **Check.** `test_dashboard_figure.py`:
  - each change is one step, and undo is exact;
  - Create new figure copies the layout with every cell empty; Delete, then Undo,
    brings the figure back;
  - a name that is taken, or would not make a file name, is refused;
  - a span takes its row's cells and sends their transitions to Not shown, and cannot
    pass the row's end; a grid made smaller does the same;
  - a drop, a swap and a drag off behave as D42 says, and a copy duplicates;
  - the same transition from the same snip, shown twice in one figure or across two,
    is listed with the other panels' names; Lyα from two datasets' snips is not;
  - Apply to all panels is one step. It sets a velocity range on the velocity panels
    alone, and the y range, titles and numbers on every panel;
  - a wavelength panel's default range is the velocity range at its transition;
  - a preset saved, then loaded in another project, fills the cells it can;
  - a cell whose snip is removed is emptied, with a note;
  - each of Q9.12's options is a step, and those marked ✱ apply to all panels;
  - the figures in `examples/blind`'s `ui/project.json` hold no hidden value.

*Done on 2026-11-03 (Prompt 3): `alis/dashboard/figure.py`; 15 tests in `test_dashboard_figure.py`. The figures are kept in `ui/project.json` (Prompt 3's log). See the log.*

**9.3 — The presets from the examples (D28; QF.27; `figpresets.py`).**
- metals, DH, blends and helium, filled from the project as the table says, with
  their reference scripts' labels and numbering.
- **Check.** On RJC's projects (skipped when absent) and on synthetic ones:
  - J1358p6522 under DH gives the Lyman series down two columns;
  - Q1243p307, with two datasets, gives one per column, each titled;
  - metal_line_abs under metals gives O I and Si II adjacent;
  - a synthetic helium fit under helium gives a 2×2 grid with residuals in the panels;
  - a blends figure takes the system chosen, and titles each line with the line it
    blends.

*Done on 2026-11-03 (Prompt 3): `alis/dashboard/figpresets.py`; 9 tests, on J1358p6522 and Q1243p307 too. See the log.*

**9.4 — The script and the shipped style (D42, S22; Q9.2 (b); `figscript.py`,
`alis/data/`).**
- The dashboard's emitter writes the script from a figure, reading the snips' files
  that Export writes. `alis_publication.mplstyle` is shipped, and LaTeX is used where
  found. The references of `plotscript.py` and `config.py` to `context/plotting/` are
  updated, and nothing else in those files changes.
- **Check.** `test_dashboard_figscript.py`:
  - the script written for each preset runs in a clean folder under `python -I`, with
    no ALIS to import, and writes a figure;
  - with `latex` hidden from the path, it still runs, using matplotlib's text;
  - every colour is a variable at the top, and changing one changes the figure;
  - the figure the script draws matches the description: the spans' widths, and each
    panel's x axis, range, titles and numbers, read back from the figure;
  - `DH_Lyseries.py`'s layout is drawn: a span of three, holding the Lyman limit in
    rest wavelength over 913.15–915.2 Å, with a tick at each line;
  - Q9.12's options are drawn: a title, a label's corner, y ticks at chosen positions
    and numbered on the right, a second ion's ticks in its own colour, signed
    velocities, and the components' numbers above their ticks;
  - the PDF's bytes are the same on a second run;
  - the style file is in the wheel's `alis/data/` and is read by `matplotlib.style.use`;
  - `test_plotscript.py` passes unchanged.

*Done on 2026-11-03 (Prompt 3): `alis/dashboard/figscript.py`, `alis/data/alis_publication.mplstyle`, the comments of `plotscript.py` and `config.py`; 16 tests. Colours checked with the `dataviz` skill's validator: two ion colours, then grey. See the log.*

**9.5 — The preview (S22, Q9.3).**
- Draw preview runs the script in a separate process and shows its PNG. It also
  covers Stop, the out-of-date mark, and a failure shown as a message.
- **Check.**
  - The preview's PNG equals, pixel for pixel, the figure the written script draws at
    the same size.
  - A change marks the preview out of date and does not redraw it.
  - Stop ends the process and keeps the last preview.
  - A broken style file gives its message in place of the preview.
  - No traceback reaches the window.

*Done on 2026-11-03 (Prompt 3): the Preview pane, run in a `QProcess`; 5 window tests. See the log.*

**9.6 — The tab (D42, D43; `qt/figure.py`).**
- **Layout, at the left:** the figure menu, with Create new figure, Rename… and Delete
  figure…; the preset, grid, figure size, spacing, velocity range, residuals, colours,
  style and shared titles; Draw preview; and Export figure….
- **The preview,** in the middle.
- **Panels, at the right:** the picture of the grid with its spans, each cell's options
  button, the duplicates' marks, and the Not shown tray.
- **Check.** `test_dashboard_qt_figure.py`:
  - each control is one step;
  - the drags work onto a cell, between cells, off the grid and with the copy key;
  - the options popup sets each option, Q9.12's among them, and Apply to all panels;
  - a duplicate gives its message and its mark;
  - the figure menu switches figures;
  - the screenshots at 1440×900 match the mockup's arrangement;
  - the tab has no design reference (D44).

*Done on 2026-11-03 (Prompt 3): the tab, its options dialog and its drags; 6 window tests. See the log.*

**9.7 — Exporting figures (Q9.6, Q9.11; `export.py`).**
- The figures in File → Export…: the script, its PDF, its snips' files (their columns
  locked) and the style, as the design says; and the tab's Export figure….
- **Check.** `test_dashboard_export.py` and the window's Export tests:
  - two figures exported write `<project>_Figure_1.py` and `.pdf`, then
    `<project>_Figure_2.py` and `.pdf`, with the style once and the snips each draws;
  - with only a figure ticked, only its snips are written, and their columns cannot be
    unticked;
  - the PDF equals the one the script writes when run again;
  - a failing script leaves its files, writes no PDF, and gives its last line;
  - writing over a file asks first;
  - with `examples/blind` and a run, The figures is greyed with the reason, `plan`
    refuses a figure, and no file written holds a hidden value.

*Done on 2026-11-03 (Prompt 3): `export.Choice.figures`, `figscript.plan`, the Export dialog's figures with their columns locked; 1 unit test and 3 window tests. See the log.*

**9.8 — Close the stage.**
- **The review.** The tab's screenshots beside the mockup, on a new page with the
  earlier pages' stylesheet (built by `doc/dashboard/stage9/`), with RJC's comments
  applied. The page shows:
  - each preset on RJC's projects;
  - two figures, and a duplicate's warning;
  - a span, and the wavelength axes;
  - the options button;
  - the preview and the exported PDF side by side;
  - a blinded project.
- **The documents.** `doc/ALIS_workflow.md` §4.3 (the Figure tab), `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill. The decisions go in
  `ALIS_v2_dashboard_prompts.md` (D102 onwards, from Q9.1–Q9.12, refining D34 and
  D42). `dashboard_stage0.md` records the Figure tab done, and S22's preview comes off
  the "Later" list.
- **Check.**
  - Every element in the table below is in a task logged as done.
  - The `unit`, `gui` and `fast` batches pass, with their counts set against Stage 8's
    close (2130 passed with 120 skipped, 229, and 149).
  - CI passes on both platforms. Ubuntu's `gui` job stays within its 60 minutes: at
    Stage 8's close it was 86% through at 42 minutes. If the new window tests bring it
    near the limit, the limit is raised or the window tests are split.
  - `test-coverage` runs on what changed.

*Draft 1 of the review on 2026-11-03 (Prompt 3):
https://claude.ai/artifact/A3mGPjERKw8C2s52ecULv3. The documents and D102–D109 are
written; the `gui` job's CI limit is 90 minutes. Batches under CI's pytest 9.2.0:
`unit` 2170 passed, 120 skipped (2130 and 120 at the start of the stage); `gui` 251
passed (229); `fast` 149 passed (149), with the regression harness. Coverage of the
new modules 91–94%. Waiting on your comments on the page, your push (CI), and your
trial of the scripts (Q9.12).*

### Where each element goes

| Element | Task |
|---|---|
| D28: a general grid of panels | 9.2 |
| D28: presets for metals, DH, blends and helium | 9.3 |
| D28: the references to `context/plotting/` updated | 9.4 |
| D42: preset, grid and figure size | 9.2, 9.6 |
| D42: transitions dragged onto the grid, swapped, dragged off to Not shown | 9.2, 9.6 |
| D42: residuals shown and colour-blind-safe colours by default | 9.2, 9.4 |
| D42: every colour a variable at the script's top | 9.4 |
| D42: edited outside the dashboard, no script box | 9.4, 9.6 |
| D42: the shipped publication style, or another style file | 9.4, 9.6 |
| D42: LaTeX where installed, matplotlib's text otherwise | 9.4 |
| S22: the preview drawn by running the script | 9.5 |
| S22: layouts saved as presets | 9.2 |
| Point 2 of Stage 8: enabled once a fit has been made | 9.1 |
| Q9.1: the tab called Figure, greyed without a best fit | 9.1 |
| Q9.3: Draw preview, no automatic redraw | 9.5, 9.6 |
| Q9.5: greyed while anything is hidden | 9.1, 9.7 |
| Q9.6: several figures; Create new figure, Delete figure… | 9.2, 9.6 |
| Q9.7: a panel spanning columns | 9.2, 9.4, 9.6 |
| Q9.8: a panel's x axis, its options button, Apply to all panels | 9.2, 9.4, 9.6 |
| Q9.9: shared titles, and each panel's titles and numbers | 9.2, 9.4 |
| Q9.10: the same transition twice, warned | 9.2, 9.6 |
| Q9.11: a figure exported with its script, PDF, snips and style | 9.7 |
| Q9.12: the further options | 9.2, 9.4, 9.6 |

## Skills to use for this stage

- `gui-component`: the tab and its panes. `gui-dev`: driving the window off-screen,
  and the screenshots at 1440×900.
- `run-tests`, `gen-tests`, `test-coverage`: after each task, and at the close.
- `dataviz`: the colours and marks of the figure and its presets.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: D24, D28, D42–D44, S22, QF.14, QF.27.
- `claude_prompts/dashboard_stage0.md`: 0.8 (the Plot tab's mockup), Q0.7 and Q0.10
  (colours, the style, LaTeX), and "After v1".
- `doc/dashboard/mockups/plot.html`: the mockup.
- `alis/plotscript.py` and its tests; `context/plotting_examples/` (the table above);
  `context/misc/matplotlibrc` and `context/misc/plotting_routines.py`.
- `alis/dashboard/export.py` (D88): the snips' files the script reads.

## Queries

*Raised by Claude with this document (2026-11-03). Each gives Claude's lean.*

**Q9.1 — What the tab draws before a run.** Point 2 asks for the tab once a fit has
been made. Two ways:
- **(a) A run's best fit only.** The tab is greyed until the project has a run.
- **(b) The starting model too.** The tab is always enabled once there is a snip, and
  draws the run shown or, before any run, the starting model, labelled so.

My lean: (a). A publication figure is of a fit, and (a) is what point 2 asks for.

**Response:** Yes, we should only enable the Plot tab if a best-fit result exists (option (a)). If not, the tab should be greyed out with a tooltip explaining that a best-fit result is required to generate a figure for publication. Related to this, I think the "Plot" tab should be relabeled to "Figure".

**Q9.2 — Where the emitter lives.** `alis/plotscript.py` writes a script from an ALIS
run, from ALIS's internals. The Plot tab needs one from a project and its run. Two ways:
- **(a) One emitter in `plotscript.py`,** fed either by an ALIS run or by the
  dashboard (through the snips' files that Export writes), so that both write the same
  script. ALIS outside the dashboard changes.
- **(b) A dashboard emitter** in `alis/dashboard/`, and `plotscript.py` left as it is.

My lean: (a). One script, one set of presets, and `plotscript.py`'s references are to
be updated anyway (QF.27).

**Response:** Given that the plotting script is intended to be generated from within the dashboard, and the publication figure will only be generated in the dashboard (never from a standalone run of ALIS), I think it makes sense to have the emitter live in `alis/dashboard/`. However, if you have good reasons to make it a more general-purpose script outside the dashboard, I am open to hearing your thoughts. My opinion is that including it in the dashboard will allow the dashboard to have more control over the script generation process, the user defaults, and make it easier to maintain and update in the future. The `plotscript.py` can still exist for standalone use, but the dashboard should have its own emitter that is tailored to its needs.

**Q9.3 — The cost of an exact preview.** Running the script for each preview takes a
second or two, more with LaTeX. Two ways:
- **(a) Always the script** (S22 as written), run after the user pauses, with the
  previous preview kept until the new one is drawn.
- **(b) A quick drawing while arranging, the script on demand:** the tab draws the
  panels itself as the user drags, and runs the script when asked ("Draw the preview").

My lean: (a). S22 makes the preview the figure; a pause before redrawing keeps the
dragging smooth.

**Response:** No automatic updates are needed. We just need the user to click a "Draw Preview" button after they have finished arranging the panels. This will keep the interface responsive while still allowing the user to see the final figure.

**Q9.4 — `DH_Lyseries.py`'s mixed grids.** It puts a 3×3 grid and a 3×1 grid on one
page, cells of different sizes. Two ways:
- **(a) No preset for it,** and the general grid keeps cells of one size. The written
  script can be edited to mix grids.
- **(b) Cells that span** several rows or columns, as a property of a cell.

My lean: (a) for this stage, with (b) listed in "After v1".

**Response:** Rows will never be combined, but some columns panels in a single row could be combined. It would be useful to have the ability for a single row to span a user-specified number of columns. It would also be useful for some panels to allow different options of what to present on the x-axis. The three most likely choices would be "Velocity", "Observed Wavelength" and "Rest Wavelength". The default would be "Velocity", but the user could choose to change it for a given panel. This would allow the user to present the data in the most appropriate way for their audience. Users should also be allowed to decide if the x-axis and y-axis labels are shown or not shown for each panel. This will allow the user to create a cleaner figure with less clutter.

**Q9.5 — A figure of a blinded project (D24).** The model of a hidden line, and its
ticks, tell its values. Three ways:
- **(a) Refused** while anything is blinded: the tab says why; a publication figure
  follows unblinding.
- **(b) The data alone** for the panels a hidden line reaches: no model, no ticks, and a
  note on the tab and in the script.
- **(c) Everything,** with a warning, as Export writes hidden lines hidden but warns.

My lean: (a). It is the simplest rule that cannot leak, and figures for a paper are
made after unblinding.

**Response:** I agree with option (a).

**Q9.6 — Is the figure part of the project?** Two ways:
- **(a) Yes:** kept in the bundle (`ui/plot.json`), each change a step that Undo takes
  back, and saved with the project.
- **(b) A view:** kept in `ui/view.json`, like the zoom and the open groups, outside the
  history.

My lean: (a). A figure is work the user would not want to lose or redo; D42's drags are
edits.

**Response:** I agree with option (a). Note that the "Export..." dialog should allow an option to export everything needed to generate the figure (this will include the relevant snip files, the style file, the python script, and a pdf of the current version of the figure). This will allow the user to make edits to the figure outside the dashboard. Furthermore, there should be an option in the dashboard for the user to create multiple figures from the same project, each with its own layout and set of panels. There should be a button that says "Create New Figure" that will create a new figure, starting with the same layout as the current figure, but with empty panels. The user should also be able to remove/delete a figure. If a panel of one figure shows a transition that also appears on any other panel (either on this figure, or any other figure in the project), the user should be warned that this transition appears on another figure panel (note that the user is *allowed* to include the same transition in different panels, but they must be warned).

*Raised by Claude on 2026-11-03 (Prompt 1), from the responses to Q9.1–Q9.6. Each
gives Claude's lean.*

**Q9.2 is still open.** It has no response yet, and its lean stands: (a), one emitter
in `plotscript.py`. Task 9.4 waits on it; Tasks 9.1–9.3 do not. *Answered since
then: (b), a dashboard emitter (Q9.2's response).*

**My readings of the other responses.** None needs a query; please correct any that
is wrong.
- **Q9.1.** The tab is called Figure everywhere in the window (its name, its tooltip,
  the shortcut sheet), and in the code (`FigureTab`, `qt/figure.py`). It is greyed
  until the run the Fit tab shows has a best fit, with the reason as its tooltip: "A
  figure for publication needs a best fit: run the fit first." Every figure draws
  the run the Fit tab shows.
- **Q9.3.** Nothing redraws by itself. Draw preview runs the script, so the preview
  is still exactly the figure (S22). After a change, the last preview stays, marked
  "The figure has changed since this preview was drawn." Before the first, the
  middle pane says what the button does. While the script runs, the button reads
  Stop.
- **Q9.4.** `DH_Lyseries.py` still gets no preset. Spans and axes (Q9.7, Q9.8) let
  the user build its layout by hand and save it as a preset of their own (S22).
- **Q9.5.** While anything is hidden (the whole analysis or one line), the tab is
  greyed with the reason ("Unblind… first: a figure shows the hidden lines'
  profiles."), and Export's figure option is greyed too. Figures made before the
  blinding stay in the project, unchanged, and come back on unblinding.
- **Q9.6.** A project holds any number of figures, kept in `ui/figures.json`, and each
  change is one step. A menu at the top of Layout chooses the figure, beside Create
  new figure, Rename… and Delete figure…. The buttons follow the window's sentence
  case. A new figure is named Figure 2, Figure 3, and so on. It copies the shown
  figure's layout (grid, spans, axes, labels, size, ranges, residuals, colours,
  style), with every cell empty and every transition in Not shown. Delete asks
  first, and is one step that Undo takes back. A preset fills only the figure shown.

**Q9.7 — A panel that spans columns (Q9.4's response).** I read the response as
follows: within a row, one panel may take the width of several columns, but a panel
is never more than one row high. `DH_Lyseries.py` does this. It has three velocity
panels, then one panel across all three columns (the Lyman limit, in rest
wavelength, 913.15–915.2 Å), then three more velocity panels. There are two ways to
hold this:
- **(a) A span per cell.** The grid keeps its columns × rows, and each cell has a
  span from 1 to the number of columns. The cells a span covers leave the row, and
  their transitions go back to Not shown. A row of three columns can then be
  1 + 1 + 1, 2 + 1, 1 + 2 or 3.
- **(b) Columns per row.** Each row has its own number of equal panels (3, then 1,
  then 3). A three-column figure can then have a row of two equal panels, but not
  2 + 1.

My lean: (a). It is what the response describes, and it draws `DH_Lyseries.py`. A
panel that spans columns keeps the figure's x range unless it is given its own
(Q9.8 (i)).

**Response:** I agree with option (a).

**Q9.8 — A panel's x axis (Q9.4's response).** Each panel chooses Velocity (the
default), Observed wavelength or Rest wavelength. Four details:
- **(i) Its range.** A velocity panel takes the figure's velocity range. A wavelength
  panel has its own range, in Å. By default, that range is the figure's velocity
  range converted at the panel's transition, so changing the axis keeps the same
  pixels. Any panel can be given a range of its own, as `DH_Lyseries.py`'s Lyman limit
  is.
- **(ii) Rest wavelength** is the observed wavelength divided by 1 + z. Here z is the
  redshift of the component that is the figure's velocity zero, just as
  `DH_Lyseries.py` divides by its first component's 1 + z.
- **(iii) A snip's shift.** The wavelengths are corrected for the snip's shift (as
  Export's Corrected wavelength and Corrected velocity are), so the components' ticks
  fall on the lines in every dataset. A snip with no shift is unchanged.
- **(iv) The ticks.** A velocity panel ticks every component, as every reference
  script does. A wavelength panel may hold many lines. It ticks each transition of its
  ion within its range, at the redshift of the velocity zero, as `DH_Lyseries.py`'s
  Lyman limit does (18 ticks, rather than 18 for each component).

My lean: as written in (i)–(iv).

**Response:** I agree with (i)-(iv) as written. It might be beneficial to allow users to click a button on the panel to toggle all options (e.g. x-axis limits, y-axis limits, whether to show the x-axis and y-axis labels, the x-axis type (i.e. velocity, Observed Wavelength, Rest Wavelength), if the velocity range is updated for Velocity mode, there should be a button that says "Apply range to all panels" which will set the same velocity range for all panels. etc.). This will make it easier for users to customize the figure without having to go through multiple menus. Based on the reference scripts, if you think additional options would be beneficial to customize, please let me know and I can provide suggestions. Also note that most users won't have the plotting_routines module that I've written, so I've added this script as context during development (it is located here: `context/misc/plotting_routines.py`). Several options could include the "Apply to all panels" option, such as the tick labels, the y-axis range, the x-axis range, and the axis labels. This will allow users to quickly apply changes to all panels without having to manually change each one.

**Q9.9 — A panel's labels (Q9.4's response).** Twenty of the 21 reference scripts
write each axis title once for the whole figure, either on an invisible axis or with
`fig.text`. They number the x axis on the bottom panels only (`hidex`), and the
metals scripts leave the middle column's y axis unnumbered (`hidey`). `DH_Lyseries.py` alone
writes titles on chosen panels. "Labels" could mean the axis titles, the tick numbers,
or both. Two ways:
- **(a) Separate switches.** The figure has shared titles (one for x, one for y),
  which are on by default as in the 20 scripts. Each panel has its own x and y titles,
  off by default, and its own tick numbers on or off for each axis. By default, each
  preset numbers its panels as its reference scripts do. A panel whose x axis differs
  from the figure's (Q9.8) turns its own x title on.
- **(b) One switch for each axis of each panel,** covering both its title and its
  numbers, with no shared titles. The single centred title of the 20 scripts is then
  lost.

My lean: (a). It draws both kinds of reference figure.

**Response:** I agree with option (a). Please also see my response to Q9.8 regarding the ability to toggle options for each panel.

**Q9.10 — The same transition twice (Q9.6's response).** Two questions:
- **What counts as the same.** I would count the same transition of the same system,
  from the same snip: the same pixels drawn about the same line. Two cases would not
  be warned. With two datasets, the DH preset shows Lyα once from each dataset's snip,
  by design. A blends panel shows another system's line, in a snip that a metals
  figure may also show.
- **How it is shown.** Each figure's Not shown tray lists the transitions that figure
  does not show. Each entry is marked with where else it is shown ("· Figure 1").
  Dropping one that is shown elsewhere places it and warns in two ways: a message in
  the status bar, and a mark on the cell whose tooltip names the other panels. The
  mark stays for as long as both are shown. Within one figure, a drag moves a
  transition (D42). Dragging with the copy key held (Ctrl, or ⌥ on macOS) copies it,
  and warns the same way.

My lean: as written for both.

**Response:** I agree with the proposed approach.

**Q9.11 — Exporting a figure (Q9.6's response).** File → Export… gains The figures,
listing each figure of the project by name, with the figure shown ticked by default.
For each figure ticked, it writes:
- `<project>_<figure>.py`, the script;
- `<project>_<figure>.pdf`, drawn by running that script in the folder (S22), so the
  PDF is exactly what the script draws;
- the snips' files the figure draws, under Export's names, whether or not the snips
  are ticked. They hold the columns the script reads (Wavelength, Flux, Error, Fit
  range, Zero level, Continuum, Best-fit model), which cannot be unticked while a
  figure is ticked;
- the style file, once: `alis_publication.mplstyle`, or the file chosen.

The tab's Write plotting script… becomes Export figure…, which opens the same dialog
with only that figure ticked. If a script fails, the files it needed stay written, no
PDF is drawn, and the message is the script's last line (D86).

My lean: as written. With one dialog there is one set of file names and one blinding
rule.

**Response:** I agree with the proposed approach - excellent idea to unify the export! One clarification, let's use the names `<project>_<figure>.py` where `<figure>` is the name of the figure (e.g. `Figure 1`, `Figure 2`, etc.) rather than a number. This will make it easier for users to identify that this corresponds to a figure file, and the corresponding name of the script and PDF. Note, there should be no spaces in the file name, so we can replace spaces with underscores (e.g. `Figure_1`, `Figure_2`, etc.).

*Raised by Claude on 2026-11-03 (Prompt 2), from the responses to Q9.2 and
Q9.7–Q9.11. The Design and Tasks now follow every response. Q9.2's response is taken
as written: I see no reason to share an emitter with `run_alis`, since a publication
figure is made only in the dashboard. Q9.11's file names are in the Design's Export.*

**Q9.12 — Further options, from the reference scripts (Q9.8's response).** You asked
which other options the reference scripts suggest. I went through the 21 scripts and
`plotting_routines.py` for what they set panel by panel, or for a whole figure. My
proposals are below.

*A panel's options*, in its options popup. Those marked ✱ have Apply to all panels:
- **Its label's text and corner** (✱ for the corner). The label is "Si II 1260" by
  default, at the lower left (`xtxt`, `ytxt`). It is written in LaTeX when LaTeX is
  used.
- **A title above it.** 12 scripts give panels titles: the datasets' names ("HIRES
  data", "UVES data") and the blends ("D I Ly7 blend at z = …"). It is empty by
  default, except where the preset sets one.
- **Its y tick positions** ✱. `metals.py` numbers a shallow line at 0.6, 0.8 and 1.0,
  rather than at 0, 0.5 and 1. By default they are chosen from the y range.
- **Its y numbers on the right** ✱. 16 scripts number the right-hand column on its
  right (`tick_right`), and so do the presets by default.
- **Which components, and which ions, are ticked.** `metals.py` chooses the ticked
  components panel by panel (`tck`). 16 scripts tick a second ion in its own colour:
  D I in the H I panels, or Si II 989.87 in N III 989.79's panel. By default, every
  component is ticked, for every ion of the system with a transition in the panel's
  range, each ion in its own colour.

*A figure's options*, in Layout:
- **Signed velocities** (+100), on by default. All 21 scripts sign them with `plot_pm`,
  as D43 does in the window.
- **Component numbers above the ticks**, off by default. 6 scripts number them
  (`zabs_txt`).
- **The spacing between panels.** `metals.py` packs its panels closely (0.05 across,
  0.1 down). `DH_Lyseries.py`, whose panels have their own titles, uses 0.2 and 0.35.
  Each preset sets its own spacing.

*Variables at the script's top only*, not in the window: the font sizes (12, 11 and
8 pt), the line widths, the dashes of the continuum and zero lines (`cldash`,
`zldash`), the ticks' heights (`tmin`, `tmax`), the residuals' offset and scale
(`offs`, `wons`), and the spacing of the major and minor x ticks.

*Not carried over:* the Legendre re-normalisation in 3 scripts (`coeff`, `scale`). The
snips' files hold the best-fit continuum, and the script divides by it. None of the
scripts uses `plot_rmzerosign`.

My lean: as listed. Every panel option is in its one popup, as your response asks,
and each can still be changed in the written script.

**Response:** I agree with the proposed options. Please note that I will review these and suggest possible additions/changes/removals after I have a chance to test the scripts when everything in Stage 9 is implemented.

## Prompts

1. Please read this document, and the work of Stages 0–8 (their documents and logs). I have responded to your first set of queries. Please review these responses, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, we will start to implement the new features covered by this design document.

2. I have responded to your queries. Please review these responses, and if anything is unclear, ask further questions in the Queries section, and I will respond; when everything is clear, I will give the approval to implement the new features covered by this design document.

3. I have responded to your new queries. Please review these responses, and if anything is unclear, ask further questions in the Queries section, and I will respond; when everything is clear, please implement the new features covered by this design document in the order of the tasks, and with the checks, until Stage 9 is complete. If it is beneficial to stop during the middle of this stage for a review, please ask for the review and I will provide feedback, as in the earlier stages of this development. There is no need to stop between tasks of this stage, unless you think it is useful. Please keep the design document open while you work, and refer to it for the details of each task. If you find any issues or have suggestions for improvements, please note them in the Queries section, and I will respond.
