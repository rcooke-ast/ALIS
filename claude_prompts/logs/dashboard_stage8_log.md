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
