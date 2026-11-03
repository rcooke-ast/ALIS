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
