# Dashboard Stage 8 log

### 2026-11-01 (Stage 7's Prompt 10: the document)

- `claude_prompts/dashboard_stage8.md` written at the close of Stage 7, as its
  Prompt 10 asks: the Feedback section for RJC to fill in, four items carried over
  from Stage 7, how the feedback becomes tasks, the closing task (8.N), and queries
  Q8.1–Q8.5. Awaiting RJC's review and approval.

### 2026-11-01 (RJC's report: shifts and zero levels missing from Fit · Results)

- **Report.** After a run of RJC's J193957m100241 project, Fit · Results showed two
  of its three shifts and none of its four zero levels.
- **Cause.** Not the fit, and not `View.table()`: the table had every row (three
  shifts, four zero levels at −0.0139 to −0.0359, as Task 7.7 meant). The Instrument
  group was collapsed by default (Stage 6's design), and a collapsed group shows only
  its flagged rows: Shift 1 and Shift 3 were "unconstrained", so they alone showed,
  under "▸ Instrument · 2 flagged rows shown". Reproduced off-screen in the window.
- **Fix.**
  - `results.py`: the Instrument group is open by default. Since Task 7.13 it holds
    each snip's own shift and zero level, which are fitted values. A group the user
    closes stays closed, as before (`ui/view.json`).
  - `qt/fitresults.py`: a closed group's title says how many rows it hides
    ("▸ Instrument · 2 flagged of 8 rows shown").
  - `project.py`: `specids_name` names what a set of specids covers (each dataset, or
    its snips by label), factored out of `zero_level_name`, which now names a zero
    level over two datasets by both. `results.py` names shifts and FWHMs with it, as
    zero levels already were: "Shift · J193957m100241, Lyb" rather than "Shift 1".
- **Tests** (each failed before the fix): `test_dashboard_results.py`
  (J193957m100241's Instrument rows and names, skipped when absent; `specids_name`
  on a synthetic two-dataset project; J1358p6522's Instrument group open) and
  `test_dashboard_qt_results.py` (J193957m100241 in the window, open and closed;
  J1358p6522's titles).
- `test_dashboard_runs_process.py::test_rjc_j1939_zero_levels_leave_0` compared the
  new χ² with the project's latest run. RJC's project now holds a run made after Task
  7.7 (χ² 3263.67), which a new run equals, so the test failed whatever the code. It
  now compares with the zero-pinned run's χ² (run 2, 3659.37), as Task 7.7 meant.
- **Batches.** `unit`: 2090 passed, 119 skipped. `gui`: 212 passed.
  `test_dashboard_runs_process.py`: 13 passed.

### 2026-11-02 (Prompt 1: the design, the tasks and Q8.6–Q8.11)

- **Read.** The Feedback list (points 1–6), the four items carried over (Q8.2 (a)),
  the responses to Q8.1–Q8.5, D1–D88, `dashboard_stage7.md`, `deferred_work.md` §7,
  and the code that each point touches.
- **Reproduced or traced** (details in the document's Design):
  - **Point 6,** on a scratch copy of RJC's J101155p294141. Off-screen, a drag of a2
    begun 5 km/s from its tick stops at 51,062 K, and one begun 10 km/s away stops
    at 34,985 K: b can narrow only by the distance at which the drag began (Q5.23).
    `components.drag` itself is right (b = 1 km/s gives T = 61 K). Pure thermal, on
    a synthetic model of the project's shape, frees a T label shared with other
    lines: `0.0ZEROT` becomes `6060.7zerot` on a1 and on the forest line too, as in
    RJC's project.
  - **Point 5:** a `QLineEdit` filled with its text keeps its cursor at the end.
  - **Point 3:** `mask_regions` spans pixel centres, while `step_curve` draws to
    pixel edges.
  - **Point 4,** from the CI log only: a Bus error in pyqtgraph's
    `PlotCurveItem.paint`, in the teardown of
    `test_new_restarts_replace_the_kept_ones_after_a_question`, with PySide6 6.12.0
    on CI; 6.11.2 here, where the `gui` batch passes. Not yet reproduced.
  - **Point 1:** ALIS has nine resolution functions, and `bundle.unsupported`
    refuses `lsffile`.
  - **Also found:** the pre-flight check refuses RJC's project as saved (a3's
    `0.0BX1` lies below its `lim param BX1 [0.01,None]`), and its low-b note is
    given for fixed b_turb.
- **Written.** Design ("What was found", "What stays as it is"), Tasks 8.1–8.14,
  and Queries Q8.6–Q8.11, each with a lean. No code was changed. Awaiting RJC's
  responses.

### 2026-11-02 (Prompt 2: RJC's responses to Q8.6–Q8.11)

- **The responses.** RJC took every lean: (b) for Q8.6, with the per-snip override
  confirmed, and (a) for Q8.7–Q8.11. Q8.8's response adds that a user who chooses
  `lsf` must be told to install linetools.
- **Checked.** ALIS fits a continuum alone (metal_line_abs's Legendre over
  1301.0–1301.6 Å, run in the scratchpad: χ² 47.2 → 41.7), with an empty
  `absorption` header and without one. Task 8.10 therefore needs no change to ALIS.
- **Revised.** Task 8.8: `lsf` is marked "needs linetools" rather than greyed;
  choosing it asks first, with `pip install linetools`; and the Snip box, the
  pre-flight check and the panels repeat the note while linetools is missing. The
  Add a spectrum dialog keeps its FWHM in km/s. Task 8.10: its "ALIS first" step is
  replaced by the result above. Design, task and query notes say the responses are
  in. Every other task stands as written.
- No new queries. The tasks are ready to execute, 8.1 first. No code was changed.

### 2026-11-02 (Prompt 3: Task 8.1, paused on Q8.12)

- **Task 8.1, reproduction tried.** On this Mac (arm64), in scratchpad environments,
  the `gui` tests up to and including `test_dashboard_qt_fit_tools.py` were run in
  CI's order, with the 23 tests CI skipped (they need `context/`) deselected:
  - PySide6 6.12.0 over the conda env: 86 passed (174.6 s). The whole set with
    nothing deselected: 109 passed;
  - with numpy 2.5.4 as well: 86 passed (164.7 s);
  - a PyPI-only environment made with uv (Python 3.13.18, numpy 2.5.4, PySide6
    6.12.0, pyqtgraph 0.14.0, pytest 9.2.0, pytest-qt 4.5.0): 86 passed (199.4 s);
  - `test_dashboard_qt_fit_tools.py` alone, with 6.12.0: 7 passed.
- **Ruled out.**
  - pyqtgraph's `arrayToQPath` (it writes into Qt's buffers): 300 random paths, some
    with gaps, were checked element by element against their input under 6.11.2 and
    6.12.0; none was wrong.
  - Threads: the only worker (the `.mod` panel's `_FullCheck`) runs ALIS's loaders
    and touches no Qt object.
  - The restarts chart draws markers only (`pen=None`), so the crashing
    `PlotCurveItem` is another of the window's curves.
  - A web search found no report of the crash for PySide6 6.12.
- **Timing.** The crash came in pytest-qt's first `processEvents` of the teardown,
  before the fixtures' own teardown, so the window was alive. PySide6 6.12.0 was
  released on 8 October; numpy 2.5.4 on 10 October.
- **Q8.12** raised, with a lean of one diagnostic push (6.12.0 and 6.11.2 side by
  side on macOS). Development is paused until RJC answers, as Prompt 3 asks. No code
  was changed. The scratch environments stay in the scratchpad for the next step;
  the link uv made in `~/.local/bin` was removed.

### 2026-11-03 (Prompt 3: Task 8.1, done after Q8.12's response)

- **Q8.12's response.** RJC's next push passed the macOS `gui` job with no change, so
  the crash is set aside and watched for. The same push failed the `lint` job
  (`context/misc/GitHub_lint.txt`: `fatal: bad object 0000…0000`, exit 128), and the
  response asks for it to be fixed before the stage closes.
- **The cause.** The first push of a new branch has an all-zeros
  `github.event.before`. `git rev-parse --verify` accepts any well-formed 40-digit
  name without looking it up (checked here: exit 0 for all zeros), so the step's
  guard let it through and `git diff` stopped.
- **The fix** (`.github/workflows/ci.yml`, the lint step): the guard is
  `git rev-parse --verify --quiet "$BASE^{commit}"`, which looks the commit up (exit
  1 for all zeros). A new branch skips lint, as the step meant; later pushes lint
  their changed files as before.
- **Checked.** The step's shell, run on four bases: all zeros and empty both skip;
  `cd82193` (this branch's first commit) finds no changed Python files; `58afc94`
  (Stage 7's closing commit) lists the changed files. The workflow parses as YAML. The
  `unit`, `gui` and `examples` jobs are untouched; RJC's next push is the check on CI.

### 2026-11-03 (Prompt 3: Task 8.2, a card's value shows its start)

- **The fix.** `qt/widgets.py` gains `ValueField`, a `QLineEdit` whose cursor goes
  to the start when it is filled (`setText`) and when it loses the focus. Its
  tooltip is the whole value as the field holds it (so the mask while it is hidden),
  above the text given to `setToolTip`, which existing call sites keep using for
  their help. It replaces the plain field in the four places the task names: the
  cards' values (`qt/components.py`, `_value_row`), the Data tab's primary z, the
  Snip box's own shift and the Run box's settings.
- **Tests.** `test_a_long_value_shows_its_start` (`test_dashboard_qt_components.py`):
  a card's z of 0.00012345678901234 on a 40-pixel field shows its start, after it is
  filled and after a new value is typed and the field left (one step); the Data tab's
  z of 2.5007852870610208 too. The Snip box's shift, the Run box's settings and a
  masked card value's tooltip are checked in their existing tests. With the plain
  `QLineEdit` put back (a one-off pytest plugin in the scratchpad), the card and
  shift checks fail, under PySide6 as the tests run it.
- **Batches.** `unit` 2091 passed, 119 skipped; `gui` 213 passed. pre-commit passed.

### 2026-11-03 (Prompt 3: Task 8.3, a temperature mode changes its component alone)

- **The fix** (`components.py`). `temperature_plan` returns a `TemperaturePlan`
  (the change, the lines it relabels, and notes); `temperature` makes its step, as
  before. `_own_labels` finds a T or b_turb label of the component's main line that
  a line outside the component carries, live or commented out, and replaces it on
  the component's lines with a new label of the scheme (`edit.new_component`, the
  case kept). The modes then work on the new label. A label carried only within the
  component (D20) stays.
- **Three things the task did not spell out**, each kept to the spirit of Q8.10 (a),
  that the relabel changes nothing but the sharing:
  - the component's own commented-out lines (the ions it is excluded from) take the
    new label too, so an ion included again joins the component and not the old
    label (`test_an_excluded_ion_keeps_the_component_s_new_label`);
  - a `lim param` of the old label is copied to the new one, so the component keeps
    its limit (RJC's project has `lim param zerot [1.0,None]`; a2's `ta3` gets the
    same line);
  - a label that a link uses is refused with a message ("change the link first"),
    since moving the component off it would change what the link computes.
- **The window.** The card's temperature menu shows the plan (`PlanDialog`) when a
  label stays with other lines: the line that changes, and "The T of a1 and of 1 line
  of the Lyα forest keep ZEROT, fixed at 0 K." (values through the gate). A refusal is
  a message, not a traceback.
- **A Stage 7 test changed.** `test_a_b_turb_tied_across_components_gets_one_limit`
  expected pure thermal on a1 to move a2's b_turb, tied to `ba1`, with it. Q8.10's
  response chose (a) over (b) "ties are followed", so the test is now
  `…_is_split_by_pure_thermal`: a1 takes `BA3`, a2 keeps `ba1`, and only `BA3` has
  the limit.
- **Tests.** In `test_dashboard_components.py`, on the J101155p294141 shape (two H I
  and a forest line on `0.0ZEROT`): each of the four modes on a2 gives a2 a new T
  label (`ta1`/`TA1`), a1's and the forest's lines byte for byte unchanged, no limit
  on ZEROT, ALIS's check clean, one undo restoring every byte; a shared free b_turb
  (`bsh`, with its limit); a T shared by H I and D I of one component staying one
  label; the excluded ion; the link refusal; and RJC's project (skipped when absent),
  with and without Untie first: a2 takes `ta3`, a1 and the three forest lines stay as
  written, and undo restores every byte. Pure thermal on a2 of the synthetic model
  writes 6060.7 K, the value the Design found on a1 and the forest. In
  `test_dashboard_qt_components.py`: the card's Pure thermal shows the plan, Cancel
  changes nothing, Set is one step.
- **Batches** (Task 8.3). `unit` 2101 passed, 119 skipped; `gui` 214 passed.
  pre-commit passed.

### 2026-11-03 (Prompt 3: Task 8.4, a drag narrows a line down to its bound)

- **The fix** (`qt/components.py`, `_box_dragged`). The drag records the side of the
  tick it begins on, and b changes by that side times the pointer's movement:
  toward the tick narrows, away widens, at 1 km/s per km/s, and moving on past the
  tick keeps narrowing until `components.drag` clamps b_turb or T at its bound. A
  key pressed during the drag (N, B) rebases as before. The readout adds ", at its
  limit" to a value held at its bound (`components.bounds`), and stays Δχ² alone for
  a hidden value.
- **Tests** (`test_dashboard_qt_components.py`):
  - `test_a_drag_narrows_past_the_tick_down_to_the_bound`: a synthetic pure-thermal
    H I (b_turb `0.0BX1`, T `70000.0ta1`, with pure thermal's limits). Begun 5 km/s
    from the tick and ended 40 km/s past it: "T 1 K, at its limit", and T = 1 K in
    one step. Ended at the tick: b narrower by 5 km/s. Away on either side: wider by
    4 km/s. From the left, 8 km/s toward and past the tick: narrower by 8 km/s.
  - `test_a_turbulent_drag_past_the_tick_stops_at_voigt_s_own`: O I's b_turb stops
    at 0.5 km/s, "at its limit".
  - `test_rjc_j1011_a_drag_of_a2_reaches_5000_k` (skipped when absent): begun 5 km/s
    from a2's tick, on the side away from a3, and ended 20 km/s past it, T is within
    5 % of 5,000 K.
  - With the old rule put back for one run, all three fail (RJC's a2 went to
    145,672 K, since the old rule widened past the tick). The existing drag tests
    pass unchanged.

### 2026-11-03 (Prompt 3: Task 8.5, pure thermal gives back the limit it replaced)

- **The change** (`components.py`). `_floor`, on an existing `lim param` line, writes
  the new limit and appends `   # pure thermal; was <old>` (`THERMAL_WAS`), building
  the line from its own text so nothing else moves. `_unfloor` reads the comment
  (`_was`) and gives the line back exactly; a respaced comment is still read. A line
  pure thermal wrote has no comment and is taken out, as before, and so is one whose
  comment was edited away (Q8.11). Leaving the mode, b_turb's lowest value is the
  restored limit's lower bound (0.01 km/s, say) rather than voigt.py's 0.5 km/s.
- **The pre-flight note.** `_low_b` skips a fixed b_turb.
- **Tests.** In `test_dashboard_components.py`: with `lim param bx1 [0.01,None]` and
  `lim param ta1 [0.0,100000.0]`, pure thermal writes both lines with their comments,
  and Free T, a 0 K preset and 8000 K fixed each give both back byte for byte (T's
  label upper case when the mode fixes it), with ALIS's check clean and undo exact; a
  line pure thermal wrote is taken out; one whose comment was edited away is taken
  out. Task 8.3's shared-b_turb test now expects the comment on `BA1`'s line. In
  `test_dashboard_preflight.py`: a b_turb of 0.3 km/s is noted when free (`da`) and
  not when fixed (`DA`).
- **Batches** (Tasks 8.4 and 8.5 together, the 8.3 `gui` run having just finished):
  `unit` 2108 passed, 119 skipped; `gui` below. pre-commit passed.
  `gui` 217 passed.

### 2026-11-03 (Prompt 3: Task 8.6, fit regions shaded to the pixels' edges)

- **The helper** (`qt/plots.py`, beside `step_curve`). `pixel_edges` gives each
  pixel's step edges, as `step_curve` draws them (which now uses it):
  half way to the neighbours, and beside a gap or at an end as far as the pixel's
  other edge. `pixel_spans(x, spans)` turns spans (by the pixels' centres, as
  `mask_regions` and `regions_of` give them, or any range of the axis) into spans
  from the first pixel's left edge to the last pixel's right edge, of the pixels
  whose centres lie in each; a run across a gap gives a span on each side.
- **Where it is used.** In the views themselves, so that every caller is covered:
  `SpectrumView.set_regions` (the Regions tab's fit, other and shared pixels; the
  Components tab's panels; Inspect and Compare) snaps to the pixels it draws, and
  `ResidualView.set_unfitted` to the residuals it draws. `mask_regions`, the fit
  range and `FV.unfitted_spans` are unchanged.
- **The drag** (`qt/regions.py`, `move_region`). The dragged edges choose the pixels
  whose centres lie between them, and the edits are made from those centres; moved
  within the same pixels, no step is made and the shading goes back to its edges.
- **Tests.** `test_shaded_spans_end_at_the_step_edges` (`test_dashboard_qt_plots.py`,
  a uniform grid and a non-uniform one with a gap): the edges equal `step_curve`'s,
  for runs at the start and the end, of one pixel and across the gap; a range holds
  the pixels whose centres lie in it. `test_fit_regions_are_shaded_to_the_pixels_edges`
  (Regions tab): the shading ends at the step edges to 10⁻⁹ Å; the pixel holding the
  drawn edge, its centre outside, is neither fitted nor shaded; an edge dragged past
  a pixel's centre fits it and settles on its edge, and short of the next centre
  does not; a move within the same pixels makes no step. The Components panel
  (velocity), and the Fit panel and its residual strip after a run of
  metal_line_abs (`test_a_run_from_the_tab`), end at the step edges. With
  `pixel_spans` made to return the spans as given (a one-off plugin), the three
  window tests fail. The existing tests pass unchanged.
- **Batches.** `unit` 2108 passed, 119 skipped; `gui` below. pre-commit passed.
  `gui` 221 passed.

### 2026-11-03 (Prompt 3: Task 8.7, an LSF file a project can run)

- **In ALIS** (outside the dashboard, as Q8.7 agreed):
  - `functions/lsffile.py`: `set_files(files)` gives it a bundle's files held in
    memory (by path from the model); `loadlsf` reads the kernel from them, parsed
    once per run, and otherwise from disk as before. `fwhm()` gives the loaded
    kernel's half-maximum width (what `getminmax` estimated inline). `getminmax`
    loads the kernel its own data line names, rather than keeping the first one
    loaded, since each snip may now name its own file.
  - `load.load_data` hands its in-memory mapping to `lsffile` (None from disk), so
    `ClassMain`, `alis()`, the validator and the window's drawing all read the
    project's kernel.
  - `bundle.py`: `lsf_files(datlines)`; `unsupported(..., files=None)` refuses
    `lsffile` only when its file is not among `files` ("names a file the bundle
    does not hold"), and refuses it as before without them; `execute` passes the
    bundle's files; `pack` packs an LSF file beside the model (relative, existing)
    with the snips. `run_alis --extract` writes it back with the tree, unchanged.
- **In the dashboard.** The pre-flight check and the validator pass the project's
  files to `unsupported`. `components.fwhm_kms` gives an `lsffile` kernel's FWHM
  (`lsf_fwhm`: the kernel `index:` names, else the first), for the first guess;
  the buffer check gives `lsffile` the project's files before `getminmax`. Export:
  "Everything run_alis needs" writes the file with the tree; the snips' export
  (`plan`) writes each LSF file beside the snips (`lsf_beside`, `_2` on a clash) and
  the models' data lines name it there (`point_at`'s `lsf`).
- **Found.** `lsffile` stops under `run bintype A` ("bintype is set to 'A'…"), so
  the tests fit `examples/lsf_file`'s `fit_spectra.mod` through `test_lsf.dat` with
  ALIS's default km/s bins, as `generate_spectra.mod` has them. And `getminmax`
  turns the kernel's width into a velocity at a fixed 1309 Å, so at longer
  wavelengths it asks for more buffer than it needs. Neither is changed (ALIS
  outside the dashboard, beyond Q8.7); both go to RJC at the close.
- **Tests.** `test_bundle.py`: `unsupported` with and without the file;
  `test_a_bundle_fits_with_the_lsf_file_it_holds` (packed, run from an empty folder
  and extracted, it matches the plain run beside the file); a bundle without its
  file stops with "does not hold"; two kernels (FWHM 0.05 and 0.12 Å): `index:1`
  gives the same best fit in two runs, and not `index:0`'s. `test_dashboard_export.py`:
  the bundle's run, the "everything" export run with `run_alis`, and the snips'
  export (its starting model, the LSF file beside it) run with `run_alis` agree
  within the harness's tolerance; the kernel's FWHM is 0.05 Å at 1302 Å within 1 %,
  the buffers are given, the drawing is finite over the fitted pixels, and neither
  check refuses; the name of `../lsf/k.dat` is rewritten to the file written.
- **Batches.** `unit` 2113 passed, 119 skipped; the regression harness
  (`pytest tests/test_regression.py -m fast`, its `test_bundle` mode included) 96
  passed (713 s); the two LSF fits of `test_bundle.py` (`fast`, `examples`) 2 passed;
  `gui` 221 passed. pre-commit passed.

### 2026-11-03 (Prompt 3: notes for Task 8.8, for the next thread)

Task 8.8 is begun in a new thread (CLAUDE.md: past half the context). What was read
for it, so that it need not be read again:
- **The pattern to follow is Task 7.13's own shift.** `project.Project._own_levels`
  (≈ line 1350) reads `row.own_shift` / `row.shared_shift` by grouping the shared
  snips by an identity (the label, or a fixed value, or the line), and honours
  `project.own(specid)` (`ui/project.json` `snips[specid]["own"]`). Resolution
  wants the same: `row.own_resolution`, `row.shared_resolution`, and `Row.fwhm`
  (≈ line 500, now the first snip's) and `datasets._row_param` / `_row_lines`
  (≈ lines 121–133) taking the shared snips only, as the shift does.
- **The steps.** `datasets.snip_shift_step` (≈ line 445: a new label `vsnN` from
  `edit.used_labels`, `E.set_keyword(pm, item, "shift", word)`, back with the
  others' word or `E.remove_keyword`, `_own_ui` and `keep_structure`) is the model
  for `snip_resolution_step`. `datasets.set_value_step` / `E.set_fwhm` set the
  dataset's FWHM today.
- **The Snip box** (`qt/regions.py`, ≈ lines 555–670): `own_box`, `shift_choice`
  (As the dataset / Its own), `shift_value` (now a `W.ValueField`), `shift_badge`,
  `_fill_own`, `_own_chosen`, `_shift_typed`.
- **From Task 8.7, ready for use.** An LSF file goes into the bundle at
  `files/lsf/<name>` (the data line naming `../lsf/<name>` from `files/model/`), as
  a step's `files` entry; `B.lsf_files`, `C.lsf_fwhm(project, call)` (the kernel's
  FWHM in Å, `index:` or the first), `X.lsf_beside` for the export, and
  `lsffile.set_files` / `fwhm()` in ALIS. A file of several kernels must be written
  with `index:` (the first by default). `lsffile` stops under `run bintype A`.
- **Widths.** `components.fwhm_kms` (first guess) already reads each snip's own
  data line, `lsffile` included; `snips.buffers` likewise. The line IDs' joining
  of close lines is still to be checked (`lines.py`).

### 2026-11-03 (Prompt 4: the `gui` jobs on CI, both platforms)

- **The failure.** RJC's push of Tasks 8.1–8.7 failed the macOS and Ubuntu `gui`
  jobs on one test, `test_a_panel_s_fit_regions_end_at_the_pixels_edges` (Task 8.6),
  with `TypeError: pytest.approx() does not support nested data structures`
  (`context/misc/GitHub_gui_*_py313.txt`). CI installs pytest 9.2.0; this machine has
  9.0.2, which accepts a list of pairs. Not the Bus error of Q8.12.
- **Reproduced** in a scratch venv (`--system-site-packages`, pytest 9.2.0, the conda
  env's pytest-remotedata plugin left out): the test fails as on CI.
- **The fix.** The test compares the spans' edges as one flat list. Under 9.2.0 the
  whole `gui` batch found one more of the same kind, in
  `test_the_residuals_of_pixels_not_fitted` (`test_dashboard_qt_fit.py`), which CI
  skips (it needs `context/`); fixed the same way. No other test of the repository
  passes a list of pairs to `approx`.
- **Batch** under pytest 9.2.0, before Task 8.8: `gui` 220 passed and that one
  failed; after the fix, see Task 8.8's batches.

### 2026-11-03 (Prompt 4: Task 8.8, a snip's own resolution)

- **Found first: each function needs its own bins.** ALIS stops when a data line's
  bins (`run bintype`, or the line's own `bintype=`, `load.py`) do not suit its
  resolution: `Afwhm` needs Å bins (`afwhm.py`), `apod` Hz (`apod.py`), and the
  others km/s, `lsffile` and `lsf` included. So Task 8.7's finding that `lsffile`
  stops under `run bintype A` is this rule, and `examples/lsf_file` (`run bintype A`,
  `Afwhm`) is consistent. The dashboard writes the bins a function needs on its data
  line, `bintype=A` or `bintype=Hz` (or `bintype=km/s` in a model whose
  `run bintype` is A), and none where they are the model's own
  (`resolution.bins_patches`). SNIP gives a new snip its dataset's `bintype=` with its
  resolution.
- **The functions** (`alis/dashboard/resolution.py`, new, no Qt): the nine of ALIS,
  each with its kind (a value, a file, keywords, several parameters), unit and bins;
  the word each writes; the text the Data tab's cell and the Snip box show ("0.042 Å",
  "σ 3 km/s", "file test_lsf.dat", "apod kind:uniform", "voigtconv (in the .mod
  panel)", values through the gate); `check_keywords` (each keyword one the function
  takes, apod's kinds, and what lsf needs for COS and STIS); `read_kernels` (an LSF
  file read and checked as `lsffile` reads it: a column of offsets that increases,
  then kernels, each FWHM); `convert` (a starting value from one function to
  another); the linetools note and mark.
- **The project** (`project.py`). A row's resolution is read as Task 7.13 reads its
  shift: `Row.own_resolution`, `Row.resolution_snips`, `Row.resolution` (the call),
  and `Row.fwhm` now the dataset's (that of the snips without one of their own).
  Snips share a resolution by its label, or by its words when its value is fixed or
  not written; a snip set to Its own in the Snip box is recorded in
  `ui/project.json` (`own: ["resolution"]`).
- **The steps** (`datasets.py`): `resolution_step` (a dataset's function, on the lines
  of its snips that take the dataset's; the label kept unless another dataset shares
  it), `snip_resolution_step` (Its own: the dataset's function and value with a label
  `fwsnN` of its own, fixed as the dataset's; back: the dataset's word and bins, byte
  for byte), `snip_function_step`, and `lsf_member` (`files/lsf/<name>`, the member
  already there kept when its bytes are the same, `_2` on a clash). A file of several
  kernels is written with `index:`. Typing a value is refused for a function with none;
  Reset FWHM and its ↺ are for `vfwhm` alone (the default is in km/s).
- **The window.** The Snip box gains Resolution (As the dataset, with the dataset's
  resolution shown, or Its own): the function menu (several parameters greyed, `lsf`
  marked "needs linetools" where it is missing); a value with its unit and badge; the
  keywords' field; Choose a file…, the kernel's FWHM in Å and km/s, and Column for a
  file of several; and a note, with a link that selects the data line in the `.mod`
  panel for a function of several parameters. The Data tab's FWHM column is
  Resolution; the badge's menu has Free, Fixed, Tie and Reset for a value, and a
  Function submenu; a value is asked for in its unit, from the present width.
- **lsf without linetools** (Q8.8's response): choosing it asks first, with the
  install line ("Choose it anyway" or Cancel); the Snip box shows the note; the
  pre-flight check refuses Run (`preflight._linetools`); the panels show the note in
  place of the model (`profile.Profile` raises it as a `ProfileError`); and a project
  opened with `lsf` shows it in the window's notice banner.
- **Also found.** `text.replace_token` keeps the next word's column only while a word
  fits its gap. After a word longer than the gap (an `lsffile` path, say), going back
  leaves the next column moved: the words are those before, the spaces not. The
  `.mod` panel's Align columns tidies it. Direct round trips are byte for byte. The
  line IDs' joining of close lines uses a fixed 7 km/s (`lines.GROUP_KMS`) and never
  read a row's FWHM, so nothing there changes.
- **Tests.**
  - `test_dashboard_datasets.py` (8 new): a snip's own resolution in each function
    (its word on that line alone, with its bins, ALIS's check clean, undo exact; the
    LSF file in the project; back gives the words before); a direct round trip byte
    for byte; `lsf` from ALIS's defaults (`linetools`); a dataset's function reaching
    its snips and SNIP, with `bintype=A`; SNIP taking the dataset's and not a snip's
    own; a file of two kernels (`index:1`), a second file of the same name (`_2`), and
    four files that cannot be read; `voigtconv` on one snip read as its own and kept,
    the functions of several parameters refused; the keywords' checks and `convert`.
  - `test_dashboard_qt_regions.py`: the Snip box (Its own, each kind of function, the
    greyed menu, value, badge, keywords refused and kept, a file refused, the Data
    tab's count, back, every undo exact); `lsf` without linetools (the mark, the
    question and Cancel, the note, the pre-flight refusal, the panels' note);
    `examples/lsf_hst` loaded (`linetools`): `lsf` and its keywords read as the
    dataset's, the banner and the box's note; and **`examples/lsf_file` fitted from
    the window** (`fast`): the snip given `lsffile` with `test_lsf.dat` in the Snip
    box (its line takes `bintype=km/s`, the model's `run bintype` being A), run: log N
    of O I and the Si II/O I ratio within 3σ of `fit_spectra.mod.out.reference`
    (13.948 ± 0.026, −0.982 ± 0.065); then given a free `Afwhm`, run again: the FWHM
    of the kernel the data were made with, 0.05 Å, within 3σ.
  - `test_dashboard_qt_data.py`: the Resolution column, each function's cell, the
    value asked for in its unit, Free, Reset for `vfwhm` alone, SNIP giving `Afwhm`
    and `bintype=A`, `voigtconv` not offered. `test_dashboard_qt_blinding.py`: a
    hidden resolution masked in the cell and in the Snip box's value and tooltip.
- **Batches** (Task 8.8 and the CI fix), run under CI's pytest 9.2.0 in the scratch
  environment: `unit` 2120 passed, 120 skipped (one test of this task first failed
  there: its new snips had no fit region; fixed, and the file passes under 9.0.2 and
  9.2.0); `gui` 227 passed. The `examples/lsf_file` window test is in the `fast`
  batch (13 s). pre-commit passed on every changed file.

### 2026-11-03 (Prompt 4: notes for Tasks 8.9–8.14, for the next thread)

Stopped after Task 8.8, past half the context (CLAUDE.md). Tasks 8.9–8.14 are to
be done in order in a new thread. For it:
- **CI.** RJC's next push is the check of the `gui` fix on both platforms. Run the
  window tests under CI's pytest too (memory `ci-test-versions`: a venv with
  `--system-site-packages` and `pytest==9.2.0`).
- **Task 8.9** (a blind run's `.mod.out`): `runs.Run.parsed` / `out_values` /
  `out_errors` (`runs.py` ≈ lines 240–380) read the `.mod.out` with the dashboard's
  parser; `examples/blind`'s reference holds `------ BLIND MODEL ------` in place of
  the Si II line.
- **Task 8.14's review page** must also show Task 8.8's Snip box (each kind of
  function, the linetools note) and the Data tab's Resolution column, and the
  `bintype=` a function writes. The documents should say that a resolution function
  writes its bins.

### 2026-11-03 (Prompt 4, continued: Task 8.9, a blind run's `.mod.out`)

- RJC asked to continue in this thread, checking the context between tasks (an alert
  past 75% after a task, or 90% during one), so the notes above for the next thread
  are not needed yet.
- **The fix** (`runs.py`). `readable(text)`: each `------ BLIND MODEL ------` line
  of a `.mod.out` is read as the hidden line it replaces, taken in order from the
  copy of the input model at the end of the `.mod.out` (its model lines with
  `blind=True`, as `save.py` writes them). When the input does not give one hidden
  line for each, the text is read as it was. `out_values` leaves out the values on a
  line put back (they are the input's, not a best fit); a parameter tied to another
  line (`da`, `TA`) takes that line's. `out_errors` reads the hidden line's errors
  from the errors block, which ALIS writes in full. `Run.parsed` and `Run.texts` read
  the same text; `Run.blind_model_lines()` counts the lines put back.
- **Results and Final** (`results.py`, `qt/tabs.py`). `View.best_fit_hidden`: the
  best fit is known but for hidden lines a plain run wrote as BLIND MODEL. Such a run
  is shown in Fit · Results (the sub-tabs' view keeps it), its hidden rows masked
  (`RunGate.best_fit` masks a hidden value that is not known, rather than leaving the
  cell empty), with their errors. Inspect's Final draws nothing for it, and its note
  says why (`BLIND_MODEL_WHY`: "… was written by ALIS with 1 blinded line as BLIND
  MODEL, so its best fit is not known here and its model cannot be drawn.").
- **Tests.** `test_dashboard_runs.py::test_a_blind_run_s_mod_out_is_read`: the
  reference's Si II line put back; ALIS reads it; the continuum's values as written;
  O I's, offset by ALIS, equal metal_line_abs's plain fit (to the eight digits ALIS
  writes of an offset value); Si II's log N and z not known, its b and T those of
  O I (tied), its errors read; a `.mod.out` without the hidden line, or without
  BLIND MODEL, read as it was. `test_dashboard_qt_blinding.py::
  test_a_plain_blind_run_shows_its_results`: `examples/blind` imported with that
  `.mod.out` (Si II given values no other text holds): Fit · Results shows the run,
  no note; the continuum's c0 is 1.00006; every O I and Si II row masked, with its
  error; Final draws no curve and says why; no hidden value, nor O I's best fit, in
  the window's text.
- **Batches** (pytest 9.2.0): `unit` 2121 passed, 120 skipped; `gui` 228 passed.
  pre-commit passed.

### 2026-11-03 (Prompt 4, continued: Task 8.10, a continuum alone can be fitted)

- **The change** (`preflight._nothing_to_fit`, refining D85 as Q8.3 (b) asks). Run is
  refused for: no snip, as before; no snip with a fitted pixel (`NO_FITTED`, code
  `no_fitted`; the validator also refuses each snip with no fit region, but this
  says it for the project); and a model with no absorption line only when none of
  the continuum's parameters (the emission section) is free: "Nothing is free to
  fit: add a component, or free the continuum." (`NO_COMPONENTS`, its code kept).
  The Fit tab's marker leaves `no_fitted` to the earlier tabs, as it does the other
  two (`NOT_YET`).
- **Results, Inspect and Compare** needed no change: a run of a continuum alone is
  shown with the continuum's rows, the statistics, the marks and its Final model.
  Compare lists the absorbers, the FWHM and the statistics (D41), never the
  continuum's coefficients, so for such a run it shows χ²ν, the degrees of freedom,
  the iterations and the FWHM.
- **Tests.** `test_dashboard_preflight.py`: metal_line_abs without its absorption
  lines goes ahead (its Legendre free); its continuum fixed, it is refused with the
  message; a new project's snip with no fit region is refused with `NO_FITTED`
  (`test_no_component_to_fit` became the first of these). `test_dashboard_qt_fit.py`:
  `test_run_waits_for_a_component` became `test_run_waits_for_something_free_to_fit`
  (Run enabled with the auto continuum; each coefficient fixed, refused with the
  message; a component added, enabled; undo, refused);
  `test_run_is_greyed_while_there_is_nothing_to_fit` passes unchanged under the new
  rule; and `test_a_continuum_alone_is_fitted_from_the_window` (`fast`):
  metal_line_abs with no component, run from the window, completes; Results lists
  c0–c2 with values and errors and χ²ν; Final draws the model with its marks; a
  second run, compared, shows χ²ν and the FWHM.
- **Batches** (pytest 9.2.0): `unit` 2122 passed, 120 skipped; `gui` 229 passed.
  pre-commit passed.

### 2026-11-03 (Prompt 4, continued: Task 8.11, Export's corrected wavelength)

- **Where ALIS takes the reference.** `model_eval.model_func` calls the shift
  function on `wavespx[sp][ll:lu]`, the sub-pixel grid of the pixels ALIS loads for
  the snip, and `vshiftscale` and `polyshift` scale about its middle element
  (`shift.py`, `x[x.size//2]`), which their `call_CPU` cannot take from outside.
- **The change.** `profile.Profile.shift_reference(index)` gives that element, from
  the profile's own ALIS state (`_wavespx`, `_posnspx`), so it is the drawn model's
  (D60). `export.scale_shift(name, wave, numbers, reference)` computes the two
  functions as `shift.py` does about a given reference, in one place, and
  `_shift_wave` uses it for them when the reference is known (the run's profile for
  a best fit, the project's otherwise); without one it falls back to the file's
  middle, as before. `vshift` and `Ashift`, which have no reference, are unchanged.
- **Test** (`test_dashboard_export.py::
  test_a_scaled_shift_is_taken_out_about_alis_s_reference`): `write_shifted_fit`'s
  Lyα snip with `vshiftscale(2.0,1.0005)` and Lyβ with `polyshift(2.0,1.0005,2e-4)`,
  each loaded in part (a fit range from −20 to +120 km/s, with its buffer). The
  reference is the middle of the grid ALIS shifts, and lies more than 0.05 Å from the
  file's middle pixel. On that grid, `scale_shift` equals ALIS's own `call_CPU` to
  10⁻¹⁰ Å; the exported column equals the same formula on every pixel of the file;
  and it differs from the old result (the file's middle) by more than 10⁻⁶ Å. Stage
  7's test of `vshift`, `Ashift` and `polyshift` on whole files passes unchanged.
- **Batches** (pytest 9.2.0): `unit` 2123 passed, 120 skipped; `gui` 229 passed.
  pre-commit passed.

### 2026-11-03 (Prompt 4, continued: Task 8.12, the paths Stage 7 left untested)

- **Coverage before** (`pytest --cov`, the `unit` and `fast` tests of
  `test_dashboard_components.py` and `test_dashboard_export.py`): `components.py`
  88% (152 lines missed, among them 2295, 2319 and 2322: `label_owner` finding no
  component and `_shared_ratios`' two skips); `export.py` 90% (48 missed, among them
  188, 190 and 206: a run in progress and a run with no `.mod.out`).
- **A fault found.** A snip whose file cannot be read was skipped by `tables()`
  without a word, and ALIS then could not evaluate the model, so Export stopped with
  "ALIS cannot evaluate the model as it is (AlisStop)": a message, but not its cause.
  Export now stops first, naming the snip: "The file of snip 0 cannot be read, so ALIS
  cannot evaluate the model: relink or correct its file, or remove the snip, then
  export again." (`export.UNREAD`; the window tells an `ExportError` as a message, as
  before).
- **Tests.** `test_dashboard_export.py`: a fit in progress (`run_reason`,
  `best_fit_reason`, `tables` refusing the best fit's columns and writing those that
  need no run from the starting values), a run with no `.mod.out`, no run; a snip
  whose file is unreadable stops `tables` and `plan` with the message.
  `test_dashboard_components.py`: `label_owner` of a live line's label and of one no
  line carries is None; `_shared_ratios` skips a link line excluded with a component
  and a ratio that gives another ion, and lists the one that gives the ion.
- **Coverage after:** `components.py` 88% (149 missed; the three lines covered);
  `export.py` 91% (44 missed; 188, 190, 206 covered). Of the paths the task names,
  none is left.
- **Batches** (pytest 9.2.0): `unit` 2127 passed, 120 skipped; `gui` 229 passed.
  pre-commit passed.

### 2026-11-03 (Prompt 4, continued: Task 8.13, the Plot tab's design document)

- **Read.** D24, D28, D42–D44, S22, QF.14, QF.27; `dashboard_stage0.md` (0.8, Q0.7,
  Q0.10, "After v1"); the mockup `doc/dashboard/mockups/plot.html`;
  `alis/plotscript.py`; the 21 scripts of `context/plotting_examples/` (their grids,
  sizes, ranges, ticks, labels and residuals); `context/misc/matplotlibrc`.
- **Written.** `claude_prompts/dashboard_stage9.md`: what exists; a table matching
  every reference script to a preset (metals 3, DH 9, blends 4, helium 4), and
  `DH_Lyseries.py`'s mixed grids to none, with the reason (Q9.4); the design (the tab
  enabled once there is a run, the figure as project data, the presets, a script that
  reads the snips' files Export writes, the shipped style and LaTeX, the preview run
  as the script, blinding); eight draft tasks (9.1–9.8), each with its check; a table
  placing every element of D28, D42 and S22 in a task; and Q9.1–Q9.6, each with a
  lean. No code was changed.

### 2026-11-03 (Prompt 4, continued: Task 8.14, the close, draft 1 of the review)

- **The review page**, draft 1: https://claude.ai/artifact/Qkycw3e1TQR6oD8HGP5TY9
  (`doc/dashboard/stage8/take_screenshots.py OUT [SCREEN ...]`, `build_review.py
  SHOTS OUT.html`, published from the session scratchpad's `stage8_review.html`).
  Ten sections and six smaller figures, every screen taken from real ALIS runs in the
  window: the Snip box's Resolution with each kind of function and lsf without
  linetools, `examples/lsf_file` fitted through its LSF file, the Data tab's
  Resolution column, fit regions to the pixels' edges, a long value, a temperature
  mode's plan on a shared label, a drag held at T's bound, the limit pure thermal
  gives back, a plain blind run's results and Final, a continuum fitted alone, and the
  Instrument group. RJC's comments are applied before the stage closes.
- **Found while taking the screenshots, and fixed:**
  - the Snip box's Resolution controls, on the row of its choice, squeezed the box's
    other menus ("As the da…") and showed apod's keywords in a sliver. They are now on
    a line of their own beneath, the dataset's resolution beside the choice, and the
    keywords' field is a `ValueField` (its start shown) of a usable width;
  - `project.specids_name` named a snip by its data line's `label=`, which SNIP sets to
    the dataset's name for every snip, so a snip's own shift read "Shift · B, B". A
    label that does not tell the snip from the dataset's others now gives way to its
    specid ("Shift · B, 2"); RJC's J193957m100241 keeps "…, Lyb"
    (`test_what_a_set_of_specids_is_named` extended).
- **The documents:** `doc/ALIS_workflow.md` §4.3 ("Since Stage 8"), `CHANGELOG.md`
  (ALIS's `lsffile` in a bundle, and the dashboard's Stage 8), `tests/README.md` (a
  Stage 8 row), the `gui-component` skill ("Stage 8 additions").
- **The decisions:** D89–D99 in `ALIS_v2_dashboard_prompts.md`, refining D14, D21,
  D63, D81, D83, D85 and Q1.5's refusal of `lsffile`; those from the review are added
  once it is answered. `dashboard_stage0.md`: Stage 8 built (its review pending),
  Stage 9 the Plot tab, Stage 10 the later feedback, and the "After v1" list.
- **Q8.13** raised: `lsffile`'s buffer taken at a fixed 1309 Å, and `apod`'s read as
  Hz (ALIS outside the dashboard; lean (a), fix `lsffile`'s at least).
- **Coverage of what changed** (unit tests): `resolution.py` 77% from the datasets
  tests, 80% from the new display test alone, the rest (the linetools paths, the
  Snip box's file and keywords) covered by the window tests; `preflight.py` 94%;
  `datasets.py` 80%; `export.py` 91%. Added: `test_what_a_resolution_shows_and_the_
  kernels_a_project_holds` (each function's cell text, the project's kernels,
  `word`'s refusal, `convert`).
- **Batches.** `fast` (pytest 9.2.0; the regression harness with its `test_bundle`
  mode, and the dashboard's fits): 149 passed (1041 s), against Stage 7's close of
  143. `unit` and `gui` below, against the start of the stage (2090 unit passed, 119
  skipped; 212 `gui`).
  `unit` 2128 passed, 120 skipped; `gui` 229 passed (both under pytest 9.2.0, after
  the close's fixes). pre-commit passed on every changed file.
- **What remains:** RJC's push (the check of the `gui` fix on CI, both platforms),
  RJC's review of the page and its application, and the answer to Q8.13. The stage
  closes then.
