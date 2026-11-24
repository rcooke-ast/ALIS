# Dashboard Stage 13 log

Feedback from users, after Stage 12's list closed (Q12.2 (a)). The plan is in
`claude_prompts/dashboard_stage13.md`.

### 2026-11-24 (Prompt 1: the design, Tasks 13.1–13.9, and Q13.3–Q13.5)

**Read.**
- This stage's document: Feedback points 1–3 (point 3 in two parts), the two items
  carried over from Stage 12, and RJC's responses to Q13.1 (item 1 joins; item 2 is
  ignored) and Q13.2 (a Stage 14 document next; the dashboard likely pauses between
  Stages 13 and 14 for the documentation).
- `dashboard_stage12.md` (its design, tasks and queries) and its log's Prompt 1 and
  close, for the form and the counts at its close; `deferred_work.md` §7.10 and
  §7.11.
- RJC's files: `context/feedback01/alis_crash_report_01.txt` and `.model`;
  `context/feedback01/MWC137_02/` (`alis2.mod`, its data, and the project saved after
  the run, `data/mwc137.model`).
- The code each point touches: `runner.py` (`Commands`, `main`) and `runs.py`
  (`Process`); `results.py` (`View.why_not`, `statistics`, `statistics_rows`),
  `runs.out_values`, `qt/tabs.py` (`view_of`), `qt/window.py` (`RUN_ENDED`, the end
  of a run, `choose`, `do_file_save_with_spectra`); `alis/functions/lsfspline.py`
  (its loader), `alis/functions/base.py` (`parout`), `alis/save.py` (the data line's
  resolution); `qt/dialogs.py` (`ShortcutsDialog`), `actions.py` (`MOUSE_NOTE`),
  `qt/components.py` (`DragKeys`, Esc), `qt/regions.py` (=), `qt/knots.py` (Delete);
  `project._build_rows` and `loadfit`'s unique names; `doc/ALIS_workflow.md` §4.3's
  keys and gestures.

**Reproduced** with scratch scripts in the session's scratchpad, on copies of RJC's
files (no code changed).
- *Point 2.* The report's process has a Python parent, two threads and no Qt: the
  fitting process. Thread 0 aborts in `_enter_buffered_busy` during `Py_Finalize`;
  thread 1 is blocked in a buffered `readline` on a pipe (the runner's `Commands`).
  `examples/metal_line_abs` run through `runs.Process`: exit −6 (SIGABRT) with the
  standard input held open after `finished`, and −6 again with it closed as the
  messages ended (the process was already shutting down). `Process.detach` took
  1.35 s on `metal_line_abs` and 1.61 s on `alis_crash_report_01.model` (one
  iteration), waiting for the reader thread's lock.
- *Point 3 (i).* The saved run: status completed, `status_code` 5, "The maximum
  number of iterations has been reached", 100 iterations, χ² 2914465.7 → 24919.0,
  `.mod.out` and covariance stored. `results.View.latest`: 0 values of 204 free,
  `why_not` "The latest run has no best-fit values.". Parsing the `.mod.out`: "locations
  must be set for - lsfspline 1.6229079va,…" (specids 6 and 7 written without
  `locations`). With `;locations:…` put back on those two lines, `out_values` gives
  300 values. In `lsfspline`'s loader, keywords are not marked as given (`lsf` and
  `apod` mark theirs), so `parout` skips `locations`; `save.py` joins `parout`'s
  output with commas, and `lsfspline` splits its keywords on `;`. Results' Fit
  statistics already has "Stopped because" (plain). The datasets: specids 0, 4, 7, 10,
  12 and 13 in emsmwcs137; 5 and 14 differ from `emsmwcs137.dat` (flux ratio spread
  1.4×10⁻⁴ and 1.1×10⁻⁵), 3, 8 and 9 have errors 0.38 times `EM_MWC__137.dat`'s, and 6
  differs in the core, so each is unassigned with its own data (Task 12.12), as
  designed; EM_MWC__137 holds no snip.
- *Point 3 (ii).* `do_file_save_with_spectra` calls `choose("save_copy", …)`;
  `choose` knows `"open"`, `"save"`, `"folder"` and `"open_many"`, and opens the Open
  dialog for any other. Every other call passes a known kind.
- *Point 1.* The sheet lists `actions.ACTIONS` and `MOUSE_NOTE`. Keys outside the
  menus, not listed: N and B (`DragKeys`), Esc (Components), = (Regions), Delete
  and Backspace (knots dialog), Delete (recent projects).
- *Item 1.* Inferred rows take their data lines' `label` with no check for a taken
  name; Load snips… adds " 2" (`loadfit`). On MWC137_02 with its spectra, specid 7
  is in emsmwcs137, so no two names meet there.

**Written.**
- `dashboard_stage13.md`:
  - the Design: how the feedback becomes tasks; "13.1–13.7", what was found for each
    point, and what stays as it is;
  - Tasks 13.1–13.9, each with what it builds or fixes and its check: the fitting
    process's exit (bug), `lsfspline`'s `locations` in the `.mod.out` with runs
    stored before and `examples/lsf_spline` as a regression case (bug), a run that
    cannot be shown, the warning at the maximum number of iterations, Save a copy
    with spectra…'s dialog (bug), each dataset its own name (bug), every key and
    gesture in one list, the review, and the close (against Stage 12's counts:
    2308/119 unit (9.2.0: 2307/120), 330 `gui`, 149 `fast`; ALIS outside
    `alis/dashboard/` against 7343048);
  - Q13.3 (ALIS writes `locations`: lean (a), ALIS fixed, old runs read, and the
    regression case), Q13.4 (what warns and where: lean (b), status 5 and a failed
    convergence check, in Fit statistics, the status bar and the run history), Q13.5
    (the list of keys: lean (a), a second table in the F10 sheet, kept in
    `actions.py`);
  - notes under the responses to Q13.1 and Q13.2, the header's and the Queries'.

The work waits for RJC's responses to Q13.3–Q13.5.

### 2026-11-24 (Prompt 2: RJC's responses, Feedback point 4, Q13.6, the list checked)

**Read.** RJC's responses: Q13.3 (a), Q13.4 (the lean, (b) in the three places),
Q13.5 (a) and (c) (a button on each tab that opens that tab's rows of the list);
Feedback point 4 (`lsffile` with a first column of Δv in km/s, a change, not a bug).

**Found** for point 4 (no code changed): `alis/functions/lsffile.py` reads Δλ in Å
(`loadlsf`, `call_CPU` interpolating at each sub-pixel's offset in Å, `getminmax` and
`fwhm()` in Å); its loader marks its keywords as given, so a new one reaches the
`.mod.out`. `vfwhm.call_CPU` samples its Gaussian at (x/x[df] − 1), the velocity offset
from the centre pixel over c. In the dashboard: `resolution.parse_kernels` and
`Kernels.fwhm` in Å, the Snip box's note (`qt/regions.py`), `components.fwhm_kms` and
`lsf_fwhm`, `datasets._lsf_keywords`; the buffer check (`snips.buffers`) calls ALIS's
`getminmax`. `examples/lsf_file` (Å) is a regression case by discovery; no test of
`lsffile` itself exists.

**Written** in `dashboard_stage13.md`:
- Task 13.7, an LSF file in km/s (point 4), with the lean of Q13.6: `lsffile`'s keyword
  `unit:kms`, and the Snip box's and Data tab's drop-down; its check, a new
  `tests/test_lsffile.py` (a km/s Gaussian kernel against `vfwhm`, the Å case as now,
  the buffer, the `.mod.out` read back) and the dashboard's tests.
- Task 13.8 (the keys, 13.7 at Prompt 1) with Q13.5 (c): each tab's button and its
  check. The review became 13.9 and the close 13.10; their lists take 13.7 and point 4,
  Q13.6, `lsffile.py` in the diff of ALIS, and Q8.7 and Q8.13 among the decisions
  refined.
- Tasks 13.2 and 13.4 marked as RJC's responses; notes under the responses to
  Q13.3–Q13.5; point 4 in the Design (what was found, what stays); the header's,
  the Tasks' and the Queries' notes.
- Q13.6 (how a model says its LSF file is in km/s): lean (a), a keyword of `lsffile`,
  with the dashboard's drop-down.

**Checked.** Tasks 13.1–13.10 are numbered in order, each with what it builds or fixes
and its check; the last closes the stage with the checks that it is complete. Every
cross-reference to a task names its present number (the Q13.2 note's close corrected
to 13.10). The work waits for RJC's response to Q13.6.

### 2026-11-24 (Prompt 3: RJC's response to Q13.6, the list checked)

**Read.** RJC's response to Q13.6: (a), a keyword of `lsffile` (`unit:kms`, Å without
it), with the drop-down beside Choose a file… (First column: Δλ (Å) | Δv (km/s)). No
new Feedback point.

**Written** in `dashboard_stage13.md`: Task 13.7 marked as RJC's response (it was
written with the lean); a note under the response; the header's, the Design's ("asked
in Q13.6 and agreed"; what stays), the Tasks' and the Queries' notes. No further query
was needed.

**Checked.** Tasks 13.1–13.10 are numbered in order. Each has what it builds or fixes
(the bug and the fix, or what) and its check; 13.10 closes the stage, with the
documents, what follows (Stage 14's document, the pause for the documentation) and the
checks that the stage is complete: every point in its task (1 in 13.8, 2 in 13.1,
3 (i) in 13.2–13.4, 3 (ii) in 13.5, 4 in 13.7, item 1 in 13.6; item 2 dropped),
Q13.1–Q13.6 answered, the review of 13.9 applied, the batches against Stage 12's
counts, CI, coverage, D44 and pre-commit, and ALIS outside `alis/dashboard/` changed
only in `lsfspline.py`, `lsffile.py` and, if needed, `save.py`. Every query is
answered, and the stage is ready to be built from Task 13.1.

## Prompt 4: the tasks (2026-11-24)

*Batches* run on a snapshot of the tree after each task (so that work goes on
meanwhile), with pytest 9.0.2 locally and in a scratch venv with CI's pytest 9.2.0
(script `/private/tmp/claude-501/-Users-rcooke-Software-ALIS/56e69144-efc2-48da-8030-1650741340b9/scratchpad/batches.sh`;
RJC's `context/` linked into the snapshot). Before the first task: unit 2308 passed,
119 skipped, as at Stage 12's close.

### Task 13.1 — The fitting process ends without an abort (point 2)

**Made into tests first** (each failing on the code before, checked by putting the
old `runner.py` and `runs.py` back for the run):
- `tests/test_dashboard_runner.py` (new, `unit`): `runner.Input` reads the header, a
  3 MB project (more than a pipe holds) and then the commands, a `stop` written with
  the project's tail included; a last line with no newline; the command thread reads
  `stop`; the runner on a project it cannot read, its standard input held open, ends
  with 0 (before: `assert -6 == 0`).
- `tests/test_dashboard_runs_process.py` (`fast`): a run of `metal_line_abs`, its
  standard input held open after `finished`, ends with 0 (before −6); a `stop`
  written straight after the project stops the fit at its first iteration
  (interrupted; before, the stop worked but the exit was −6); Keep it running's
  `detach` returns within 0.1 s while a fit iterates (before 1.30 s); a run from a
  session ends with 0 (added to `test_a_run_gives_what_run_alis_gives`).

**Built.**
- `runner.py`: `Input`, the standard input read from descriptor 0 with `os.read`
  (the header, the project, then the commands, from one buffer, so nothing that came
  with the project is lost); `main` reads through it, and `Commands` iterates it.
  `sys.__stdin__` is never read, so Python's shutdown closes it with nothing to wait
  for; `sys.stdin` still goes to `/dev/null` (Q7.17).
- `runs.py` (`Process`): the messages are read from a copy of the pipe's descriptor
  with `os.read`, never through `popen.stdout`'s buffered reader (`_read`, `_take`),
  so that `detach` closes it at once; the process's standard input is closed once
  `finished` arrives (`_close_stdin`).

**Checked.** The new tests and the process tests above: 6 passed (30 s). The runner on
an unreadable project now ends in 0.3 s (1.3 s before, a second of it Python waiting
for the lock). Scratch scripts: `metal_line_abs` through `runs.Process` with the
input held open ends with 0 (was −6); `detach` takes 0.00 s on `metal_line_abs` and
on RJC's `alis_crash_report_01.model` (1.35 s and 1.61 s before); RJC's
`alis_crash_report_01.model` run to its end through the dashboard's process
(completed, 10 iterations, χ² 1563.45, 4 s) ends with 0. pre-commit passes on the
four files.

**Batches** after 13.1 (snapshot `t13_1`): unit 2312 passed, 119 skipped (9.2.0:
2311, 120); gui 330 passed (9.2.0: 330).

### Task 13.2 — A run's `.mod.out` that ALIS reads back (point 3 (i)) — paused at Q13.7

**Made into tests first.**
- `tests/test_save_helpers.py` (`unit`; the writer's tests, where a `.mod.out` is
  written from a synthetic spectrum in milliseconds, rather than `test_lsfspline.py`
  as the task names): each resolution that takes keywords is written and read back
  with the same keywords and values, and written again the same (`lsfspline`, with
  one keyword and with two, `lsffile` with and without `index:`, `apod`); and
  `lsfspline`'s line ends `;locations:…)`. Before the fix, the two `lsfspline` cases
  failed (`SystemExit` reading the line back; the token without `locations`); the
  others passed: `lsf` (already tested), `lsffile` and `apod` write their keywords.
  `multivfwhm` takes none.
- `tests/test_dashboard_runs.py` (`unit`): metal_line_abs's `.mod.out` with an
  `lsfspline` resolution written as before the fix (no `locations`, which the copy of
  the input model at its end keeps) parses with no error and gives every free
  parameter's value (before: "locations must be set …"); the line gets its
  `locations`, a second pass changes nothing; a `.mod.out` with nothing left out is
  read as it is.

**Built.**
- `alis/functions/lsfspline.py`: the loader marks a keyword given
  (`self._keywd['input'][…] = 1`, as `lsf` and `apod` do), so `parout` writes it; the
  class's `_kwsep = ';'`, the separator before each keyword on a data line.
- `alis/save.py`: the `.mod.out`'s data line puts each keyword after the function's
  `_kwsep` (`,` for every other function, as before).
- `runs.readable`: puts back an `lsfspline` line's keywords left out of a `.mod.out`
  written before, from the input model's line for the same file and specid (the copy
  at the end of the `.mod.out`, or the input given) — `_restore_keywords`,
  `_data_resolutions`.
- `CHANGELOG.md`: under Fixed, `lsfspline`'s `locations` in the `.mod.out`, and Task
  13.1's fix.

**Checked.**
- `test_save_helpers.py` and `test_lsfspline.py`: 60 passed; `test_dashboard_runs.py`:
  18 passed. pre-commit passes (black reformatted `test_dashboard_runs.py`).
- RJC's MWC137_02 project (scratch `why_final.py`): the run's `.mod.out` reads, 300
  values, none of its 204 free parameters missing, `has_best_fit`, marks for its 12
  snips; its statistics: completed, 100 iterations, "The maximum number of iterations
  has been reached". Its Final model (the dashboard's evaluation, scratch
  `final_vs_fit.py`) equals each snip's `_fit.dat` to 0.000 of the error, at all 12.
- `examples/lsf_spline`: references written from a fit made as the harness makes it
  (`alisrun.run_alis` on a staged copy: 8.5 s, 342 iterations, χ² 459.61 → 450.12,
  `.mod.out` with its `locations`), as `model/fit_spectra.mod.out.reference` and
  `data/reference_fits/OI_SiII_fit.dat`; `tests/manifest.py` lists the case (fast).
  `test_minimisation` and `test_bundle` pass; `test_fixed_param` fails: the model at
  15 of 460 pixels, worst 2.33 times the error.
- The cause, found with the sub-pixels pinned in the evaluation (`run nsubmin` =
  `nsubmax` = n): ALIS sets each snip's sub-pixels at load from the starting values;
  the fit (smoothing 4 km/s at the start) used 6, the evaluation (0.12 km/s, the best
  fit) 21. Pinned at 6 the evaluation agrees exactly; at 5 (18 pixels, 2.1σ), 7 (5
  pixels, 0.32σ), 8 to 21 (8 to 18 pixels, 1.0σ to 2.35σ) it does not. The
  dashboard's run model is the run's `input.mod` (`results._entry`), so its
  sub-pixels are the fit's.

**Paused** at Q13.7 (how `examples/lsf_spline` joins the harness: ALIS records the
sub-pixels, the case without mode (b), or the example pins them; lean (b)). The
references are in place, so `test_fixed_param[examples/lsf_spline…]` fails until
Q13.7 is answered.

## RJC's response to Q13.7, and the tasks continued (2026-11-24)

**Read.** Q13.7's response: (b); the sub-pixels a `.mod.out` is evaluated on go to
`deferred_work.md` §7.10, the key item of Stage 14, which comes next, before the
documentation; at the close, a draft `dashboard_stage14.md` around §7.10 with the
related deferred items for RJC to prioritise. No further query.

### Task 13.2, finished

- `tests/test_regression.py`: `examples/lsf_spline` in `FIXEDPARAM_EXCLUDE`, with its
  reason ("sub-pixels set from the starting smoothing (6 in the fit, 21 at the best
  fit); Q13.7, deferred_work 7.10") and a comment beside tophat's; the case runs
  `test_minimisation` and `test_bundle` (2 passed, 25 s).
- `deferred_work.md`: §7.10 retitled (ALIS's sub-pixels: the default `nsubmin` of
  10, and the sub-pixels a `.mod.out` is evaluated on), marked the key item of Stage
  14, with what was found and its "To finish"; §7.11 marked done (kept as written);
  low priority 24 says so.
- `dashboard_stage13.md`: a note under Q13.7's response; Task 13.2's regression case
  and its check; Task 13.10's documents, what follows (Stage 14's draft around
  §7.10, then the documentation) and its diff of ALIS (`save.py` now named).

**Batches** after 13.2: on snapshot `t13_2` (unit, gui under both pytests, and the
regression harness's `fast` cases); counts below when they finish.

### Task 13.3 — A run that cannot be shown says why, and shows its statistics (point 3 (i))

**Made into tests first.**
- `test_dashboard_results.py` (`unit`): a metal run whose `.mod.out` has an
  `lsfspline` with no `locations` and no input to take them from (`_broken_mod_out`):
  `why_not` starts "The outputs of the latest run do not read:" and names "locations
  must be set" (before, "The latest run has no best-fit values."), and its statistics
  are the run's (χ², 1000 iterations, ALIS's reason).
- `test_why_a_run_does_not_read_is_masked_by_the_gate`: on the blinded metal project,
  for every hidden value written into the broken line, the reason quotes it and the
  gate masks it. This found a bug of the gate: `_LINE_REF`, which keeps the line
  numbers a message gives ("line 47"), matched inside "lsfspline 8000" and kept the
  number unmasked (`8000` whole; `13.0` as `13▒▒▒▒`). Made into a test in
  `test_dashboard_blinding.py` (before: "lsfspline 3▒▒▒▒"), and fixed: `\blines?`.
- `test_dashboard_qt_results.py` (`gui`): Results shows the reason and the run's
  statistics (Iterations 1000, χ² with its start, "Stopped because"; before, none),
  titled "Fit statistics · the latest run"; the status bar at the end of such a run
  reads "The fit has finished, but its best fit cannot be shown. The outputs of …"
  (before, "…; the Fit tab shows its best fit.").

**Built.**
- `runs.out_problem`: why a `.mod.out` does not read (the parser's first error, its
  whitespace joined), or "".
- `results.View.outputs_problem` (read once), used by `why_not` before "has no
  best-fit values".
- `qt/fitresults.py`: `_statistics_view`, the run's statistics from its record when
  its best fit cannot be shown (not while a fit runs).
- `qt/window.py`: `CANNOT_SHOW`, the end of a completed or stopped run whose best fit
  cannot be shown, with why (`fit_tab.view_of`, through the gate).
- `blinding.py`: `_LINE_REF` matches "line"/"lines" as a word.

### Task 13.4 — A fit stopped at its maximum number of iterations warns (point 3 (i), Q13.4 (b))

**Made into tests first.**
- `test_dashboard_results.py` (`unit`): status 5 flags `at_maxiter`; the rows add
  "What to do" ("The fit stopped at chisq maxiter (1000) before it converged:
  Continue from best fit…, then Run, carries on from here; or raise chisq
  maxiter."); `statistics_warnings` names "Stopped because" and "What to do";
  `warning_text`; the history row's `warning` and its tooltip's "⚠ …"; statuses 1–4
  flag nothing; a run with no status (imported, `run_alis`'s words in its
  `.mod.out`) is flagged from the words; a failed convergence check flags
  "Convergence" and "What to do", a passed one nothing.
- `test_dashboard_qt_results.py` (`gui`): the warned rows in red (the palette's
  vermillion, `SEVERITY_COLOURS["error"]`: the window has one light theme), the
  others not; the history's first cell ends "⚠" in red, its tooltip the reason; the
  status bar at the end: "The fit has finished; the Fit tab shows its best fit. ⚠ It
  stopped at the maximum number of iterations (chisq maxiter 1000), before it
  converged. The convergence check found 2 of 9 parameters not converged. Fit
  statistics, on Results, says what to do."; no design reference (D44); a converged
  run shows no warning.

**Built.** `results.py`: `Statistics.at_maxiter`, `maxiter` (the run's model's
`chisq maxiter`), `not_converged`; `MAXITER_STATUS`, `MAXITER_REASON`, the advice,
`at_maxiter`, `not_converged`, `warning_text`, `statistics_warnings`; the "What to
do" row; `HistoryRow.warning` (`_history_warning`, from the record, without the
maxiter, which would need the run's model parsed). `qt/fitresults.py`: the warned
rows in red and bold, `statistics_warned`; the history's ⚠. `qt/window.py`: the
warning at the end of a run.

**Checked.** `test_dashboard_results.py`, `test_dashboard_blinding.py`: 59 passed;
`test_dashboard_qt_results.py`, `test_dashboard_qt_fit.py`, `test_dashboard_runs.py`
(not `fast`): passed. pre-commit passes on every file of 13.3 and 13.4.

### Task 13.5 — Save a copy with spectra… asks where to save (point 3 (ii))

**Made into tests first.**
- `test_dashboard_qt_foundation.py` (`unit`, no Qt): every `choose(kind, …)` call in
  `alis/dashboard/qt/` (12, read from the source) passes a kind the chooser knows
  (`open`, `save`, `folder`, `open_many`); before, `('window.py', 1762, 'save_copy')`.
- `test_dashboard_qt_relink.py` (`gui`): the window's own `choose`, with
  `QFileDialog.getSaveFileName` and `getOpenFileName` replaced: File → Save a copy
  with spectra… asks the Save dialog, "Save a copy with spectra", starting at
  "…_with_spectra.model", and the copy holds every spectrum (before, the Open
  dialog); `choose("save_copy", …)` raises. The existing test answers "save" now
  (it answered the unknown "save_copy", so it never met the dialog).

**Built.** `qt/window.py`: the action asks for `"save"`; `CHOOSER_KINDS`, and
`choose` refuses any other kind (`ValueError`) rather than opening the Open dialog.

**Checked.** `test_dashboard_qt_relink.py`, `test_dashboard_qt_foundation.py`: 17
passed. pre-commit passes.

### Task 13.6 — Each dataset its own name (item 1 carried over)

**Made into tests first.**
- `test_dashboard_project.py` (`unit`): `_one_label`, `write_twin_snips`'s fit with
  each data line `label=K7701` and its own resolution label (`va`, `vab`, as MWC137's
  specids 6 and 7): two rows "K7701" and "K7701 2" (before, "K7701" twice); a third
  "K7701 3"; rows read from `ui/project.json` with one name keep it (Q12.17's
  response).
- `test_dashboard_qt_data.py` (`gui`): imported in the window, the Data tab's cells
  and the Regions tab's strips read "K7701" and "K7701 2".

**Built.** `project._build_rows`: an inferred row takes a name no row before it has,
as Load snips… does ("… 2", "… 3").

**Checked.** `test_dashboard_project.py`: 26 passed; the `gui` test passed.
pre-commit passes. Stage 12's MWC137 `alis2.mod` imported without its spectra
(scratch `mwc_names.py`): emsmwcs137 (0, 4, 5, 10, 12, 13, 14), EM_MWC__137 (3, 8,
9), "39KI7701" (6) and "39KI7701 2" (7).

**Batches** after 13.2 (snapshot `t13_2`): unit 2327 passed, 119 skipped (9.2.0:
2326, 120); gui 330 (9.2.0: still running as 13.6 was logged); the harness below.

### Task 13.7 — An LSF file in km/s (point 4, Q13.6 (a))

**Made into tests first.**
- `tests/test_lsffile.py` (new, `unit`): a Gaussian kernel in km/s (FWHM 3 and 7)
  gives `vfwhm`'s model, to 1e-4 of the continuum on sub-pixels uniform in log
  wavelength, and to 1e-3 on sub-pixels linear in wavelength (measured 1.3e-4 and
  1.8e-4: there each takes its velocity offsets from another sub-pixel, `vfwhm` near
  the start of the array and `lsffile` at its centre); before, the kernel read as Å
  (0.71 off). The same file with no unit, or `unit:A`, gives the model it gives now,
  and a kernel of the same width in km/s the same at the line. A km/s kernel's buffer
  is `vfwhm`'s at 30σ (to 1%: `fwhm()` reads the width from the samples). An unknown
  unit is refused.
- `test_save_helpers.py`: `lsffile(name:…,unit:kms)` written into the `.mod.out` and
  read back (passed before ALIS knew the keyword only as a failure to load; now kept).
- `test_dashboard_datasets.py` (`unit`): a snip's own and a dataset's `lsffile` in
  km/s (`unit:kms` written, ALIS reads it, one undo takes it back); `kernel_unit`;
  `components.fwhm_kms` takes the width as km/s; `kernel_text`'s FWHM in both units;
  the same file in Å as before; an unknown unit refused; the Snip box's buffer check
  takes 30σ in km/s (before, 30σ as Å).
- `test_dashboard_qt_regions.py` (`gui`): the Snip box's drop-down beside Choose a
  file…, at Å for a new file; the note "kms_lsf.dat · FWHM 6.27 Å (1441 km/s)" shows
  the wrong unit; km/s chosen, one step, `unit:kms`, the note "… 6.27 km/s (0.027…
  Å)"; a new file keeps the unit; undo takes both back; no design reference (D44).
- `test_dashboard_qt_data.py` (`gui`): on the Data tab the file is followed by the LSF
  file dialog: its note changes with the unit, Set writes `unit:kms`, Cancel nothing.
  The existing test of every function passes `unit="A"` (the dialog asks otherwise).

**Built.**
- `alis/functions/lsffile.py`: the keyword `unit` (`A`, `kms`; reset for each line,
  since the loader keeps keywords on the instance), the offsets c·(λ/λc − 1) in
  `call_CPU` for km/s, `getminmax`'s width taken to Å, an unknown unit refused.
- `resolution.py`: `LSF_UNITS`, `kernel_unit`, `kernel_widths`, `kernel_text` (a
  width of 100 or more whole: "1441 km/s"); `parse_kernels`'s messages name either
  unit. `datasets.py`: `unit` on `resolution_step` and `snip_function_step`
  (`_lsf_keywords` writes `unit:kms`; another unit refused). `components.py`:
  `fwhm_kms` by the line's unit.
- `qt/regions.py`: `res_lsf_unit` ("First column: Δλ (Å)", "Δv (km/s)"), its step
  (`_res_lsf_unit_chosen`); a new file or another kernel keeps the line's unit.
  `qt/dialogs.py`: `LsfFileDialog` (the unit, the kernel of a file of several, the
  FWHM in both units). `qt/data.py`: `set_function(…, unit=None)` asks with it.
- `CHANGELOG.md`: under Added.

**Checked.** The files 13.7 touches, not `fast`: 211 passed (datasets, components,
export, lsffile, save_helpers, snips) and 70 (`qt_regions`, `qt_data`).
`test_function_interface.py` with `test_save_helpers.py`: 290 passed. pre-commit
passes (black reformatted four files).

**Batches** after 13.2 (snapshot `t13_2`): unit 2327 passed, 119 skipped (9.2.0:
2326, 120); gui 330 (9.2.0: 330); the harness below.

**The harness after 13.2** (on the snapshot): 75 passed, 23 failed. Every failure was a
`context/` case, all with one error: `ValueError: '…/ALIS/context/…covar' is not in
the subpath of '…/snap_t13_2/context/…'`, the snapshot's link to `context/` making
the cases' paths resolve outside their staged copies. Every `examples/` case passed,
`lsf_spline`'s fit and bundle included. The `context/` cases were run again from the
repo (below); the batch script keeps the link for the `unit` and `gui` batches, and
the harness is run from the repo from now on. From the repo, the 23 `context/` cases of the `fast`
batch passed (6 min): the harness after 13.2 passes whole (75 + 23).

### Task 13.8 — Every key and gesture in one list, and each tab's own (point 1; Q13.5 (a) and (c))

**Tests** (the list was written just before its tests; on the code before, they fail
on `A.VIEW_KEYS`, which did not exist):
- `test_dashboard_actions.py` (`unit`): N and B held during a drag, for the
  Components tab; every key the window's code handles outside the menus, read from
  its source (`Qt.Key.Key_…`, `QKeySequence("…")`, `StandardKey.…`, and the drag
  keys; Undo and Redo, which the `.mod` panel hands to the menus, are the menus'),
  has its row: N, B, Escape, =, Delete, Backspace; each tab's rows are its own and,
  for the four tabs that draw spectra, every spectrum's, in the sheet's order.
- `test_dashboard_qt_window.py` (`gui`): F10's sheet has two pages, Menus (as before,
  every action) and "Keys and mouse in the views" (every row of the list), opened at
  the first row of the tab shown (Components); each tab's Keys… button (the tabs'
  corner) lists that tab's rows, titled "Keys and mouse · Regions tab"; the Figure
  tab, greyed without a run, checked with its dialog directly; Show all opens the
  sheet's second page at the tab's rows; no design reference (D44).

**Built.**
- `actions.py`: `ViewKey`, `VIEW_KEYS` (36 rows: every spectrum, each tab, Set
  LSFSpline knots…, the `.mod` panel, the recent projects; each row's text checked
  against the code and `doc/ALIS_workflow.md` §4.3, its menus named as the code
  builds them), `DRAG_KEYS`, `view_keys`, `view_key_codes`.
- `qt/components.py`: `DragKeys.KEYS` built from `A.DRAG_KEYS`.
- `qt/dialogs.py`: the sheet's two pages (`KEYS_PAGE`, `key_rows`, `show_place`);
  `TabKeysDialog` with Show all. `MOUSE_NOTE` stays under the menus' table (a Stage 12
  test reads it there), and its keys are rows of the list.
- `qt/window.py`: the corner button ("Keys…", "Keys and mouse on this tab"; first "⌨ Keys", whose glyph the off-screen font draws as a dash) on every
  tab, `shown_tab`, `show_tab_keys`, `show_all_keys`; F10 opens at the tab shown.

**Checked.** `test_dashboard_qt_window.py`, `test_dashboard_qt_actions.py`,
`test_dashboard_actions.py` (not `fast`): 37 passed; the drag tests of
`test_dashboard_qt_components.py` pass. pre-commit passes (black reformatted
`actions.py`).

**Batches** after 13.8 (snapshot `t13_8`, covering 13.3–13.8): below when they finish.

Batches after 13.8 (snapshot `t13_8`, holding 13.3–13.8): unit 2353 passed, 119
skipped (9.2.0: 2352, 120); gui 340 passed (9.2.0: 340).

### Task 13.9 — The review (draft 1)

**Built.**
- `doc/dashboard/stage13/take_screenshots.py`: off-screen at 1440×900, synthetic
  scenes only, on Stage 12's helpers: examples/lsf_spline's run, its `.mod.out`
  written as before (no `locations`), on Inspect and Results (13.2); a metal run
  whose `.mod.out` does not read, on Results with the status bar (13.3); a metal run
  at status 5 with a failed convergence check, Results and its statistics pane
  (13.4); `_one_label`'s fit on the Data and Regions tabs (13.6); `write_ratio_fit`'s
  Snip box with an LSF file in Å then km/s, and the Data tab's LSF file dialog
  (13.7); the sheet's second page, the Components tab's list, and the window with
  the corner button (13.8). 13 PNGs, none failing.
- `doc/dashboard/stage13/build_review.py`: six sections with what to check, and "Also
  done" for 13.1 and 13.5 with their tests named; draft 1; one question (whether Fit
  statistics should grow to show every row when it warns; lean yes).
- Published: https://claude.ai/artifact/5S24J3u5J6RgHW8HdiTUtF (memory
  `stage-review-pages` updated).

**Found while taking the screens, and fixed (in 13.7's code).**
- The Snip box's note read "kms_lsf.dat cannot be read" on an imported fit: for a
  `.mod` kept beside its data (model member `files/ratio.mod`), `datasets.lsf_member`
  put the LSF file "beside the model's folder", `lsf/kms_lsf.dat`, outside the
  project's files, where `data_for_run` and ALIS missed it ("LSF file not found:
  ../lsf/kms_lsf.dat", from Stage 8). Made into a test first
  (`test_an_lsf_file_of_a_model_at_the_root_of_the_project`), and fixed: such a
  model's LSF file goes in `files/lsf/`, named `lsf/kms_lsf.dat`.
  `test_dashboard_datasets.py`: 34 passed.
- The corner button's "⌨ Keys" drew its glyph as a dash off-screen; it reads
  "Keys…" (the dashboard's "…" for a button that opens a dialog).

pre-commit passes on both builder scripts. The work waits for RJC's comments on the
page, which the close (13.10) applies first.

### RJC's comment on the review (2026-11-24): Task 13.10, Q13.8

**Read.** On the review's question (Fit statistics too short for "What to do"): in
place of scrolling it, a red button "Warnings and options to resolve..." between the
Fit statistics and Correlations panes, opening a dialog that lists the fit's
warnings, each with a suggestion to resolve it (blank where there is none). The page's
auto-reply asked whether the button shows only with warnings, and whether the red rows
stay in Fit statistics.

**Found.** ALIS's warnings during a run reach the dashboard as the process's
`warning` messages and are kept in memory only (`runs.Run.warnings`, used by the
status bar at the end of the run); `run.json` has none, so a run opened later has lost
them.

**Written.** `dashboard_stage13.md`: Task 13.10 (the button and the dialog, written
with the leans of Q13.8, built after RJC's response), the close renumbered 13.11 (its
checks: Q13.1–Q13.8, the review with 13.10); Q13.8 (i) when the button shows, (ii)
what stays in Fit statistics, (iii) what the dialog lists, with ALIS's warnings kept
in `run.json`; leans (a), (a), (a). A reply in the page's thread names Q13.8 and its
third question; the thread stays open for RJC's answer.

The work waits for RJC's response to Q13.8.

### RJC's second comment on the review (2026-11-24): numbers in parentheses

**Read.** On "K7701 2": a dataset named with a number takes it in parentheses,
"K7701 (2)", so that it is not read as K77012.

**Built.** `project._build_rows` and both of `loadfit`'s unique names (Load snips…,
and a fit's unassigned rows) write "{name} ({n})". Tests changed with it:
`test_dashboard_project.py` ("K7701 (2)", "K7701 (3)"), `test_dashboard_qt_data.py`
(the cells exactly "K7701", "K7701 (2)"); no test held Load snips…'s old form.
`test_dashboard_project.py`, `test_dashboard_loadfit.py`, `test_dashboard_qt_data.py`,
`test_dashboard_qt_loadfit.py` (not `fast`): 111 passed after the one assertion was
rewritten. pre-commit passes. Stage 12's MWC137: "39KI7701", "39KI7701 (2)".
Task 13.6's text says so.

**Review.** Draft 2 (the naming screens retaken, the section's text, the lede);
the thread answered and resolved. The warnings dialog's thread stays open for Q13.8.

### Task 13.10 — Warnings and options to resolve… (RJC's comment; Q13.8 (a), (a), (a))

**Read.** RJC's response to Q13.8 (the leans), in the stage document and the page's
thread.

**Tests** (each failing on the code before):
- `test_dashboard_results.py` (`unit`): `fit_warnings` of a run at status 5 (the
  warning with chisq maxiter's value, and `MAXITER_ADVICE`), of a failed check
  (`NOT_CONVERGED_ADVICE`), and of ALIS's warnings from `run.json` (no suggestion);
  none for a run that converged; "What to do" no longer among Fit statistics' rows,
  and `statistics_warnings` the two rows alone (13.4's tests changed so); a hidden
  value in ALIS's warning masked by the gate.
- `test_dashboard_runs_process.py` (`fast`): a run with `run warn_subpix 1` keeps
  ALIS's sub-pixel warning in `run.json`, each once (before, `run.json` had none).
- `test_dashboard_qt_results.py` (`gui`): the button red, shown, between Fit
  statistics and Correlations in their column; its dialog's three rows (chisq maxiter,
  the check, "ALIS warned: …" with no suggestion); hidden for a run that converged;
  the status bar ends "Warnings and options to resolve…, on Results, says what to
  do."; no design reference (D44).

**Built.**
- `runner.py`: `_Warnings.messages` (each warning once, up to 50, as sent: masked);
  `run.json`'s `warnings`.
- `results.py`: `Statistics.alis_warnings` (through the gate), `FitWarning`,
  `fit_warnings`, `WARNINGS_BUTTON`; the advice shortened to the suggestion alone;
  the "What to do" row removed from `statistics_rows`.
- `qt/fitresults.py`: the button (red, bold, hidden without a warning) between the two
  panes, `fit_warnings`, `show_warnings`. `qt/dialogs.py`: `WarningsDialog` (Warning,
  Suggestion). `qt/window.py`: the end of a run names the button.

**Checked.** `test_dashboard_results.py`, `test_dashboard_qt_results.py`,
`test_dashboard_runner.py` (not `fast`): 67 passed; the `fast` test passed (8 s).
pre-commit passes.

**Review.** Draft 3: the section on fits that may not have converged takes 13.10 (the
window with the button, and the dialog, retaken); the thread answered and resolved.
Both of RJC's threads are resolved.

### Task 13.11 — Close the stage (2026-11-24)

**The documents.**
- `doc/ALIS_workflow.md` §4.3: "Since Stage 13", six items.
- `CHANGELOG.md`: under Added, dashboard Stage 13 (and `lsffile`'s `unit:kms`, written
  with 13.7); under Fixed, the names, the Save dialog, the LSF file's folder, the
  gate, beside 13.1's and 13.2's.
- `tests/README.md`: the Stage 13 row.
- `.claude/skills/gui-component/SKILL.md`: "Stage 13 additions" (the keys list, the
  chooser's kinds, pipes read from their descriptor, a run's warnings, numbers in
  parentheses, the gate's "line", off-screen glyphs).
- `ALIS_v2_dashboard_prompts.md`: D139–D146 (the process's exit; a `.mod.out` that
  reads back; a run that cannot be shown; the warnings; the Save dialog; each
  dataset its own name; an LSF file in km/s; every key and gesture), each naming the
  decisions it refines (F6, Q7.17, S19, S31, Q12.7, Q2.4, D89, Q8.7, Q8.13, F10).
- `dashboard_stage0.md`: Stage 13 done (D139–D146); Stage 14 next, ALIS's
  sub-pixels, then the documentation.
- `deferred_work.md`: §7.10 and §7.11 were brought up to date in Task 13.2 (Q13.7);
  nothing else is left undone but the item below.
- Memory: `stage-review-pages` (the Stage 13 page), `batch-snapshots` (new).

**Found and not done.** `lsffile`'s loader keeps `index` on its instance between data
lines, so a line that names none takes the previous line's (Task 13.7 reset `unit`
only): Stage 14's "Carried over from Stage 13", item 1, with Q14.2.

**What follows.** `claude_prompts/dashboard_stage14.md`, drafted: the key item
(`deferred_work.md` §7.10: (a) the sub-pixels a `.mod.out` is evaluated on, (b) the
default `nsubmin` of 10), four related deferred items for RJC to prioritise (§2.4
brokenpowerlaw, §2.1 wavelength-dependent resolution, §7.7 apod, §5 the context fits
in CI), a Feedback section, the item carried over, and Q14.1–Q14.3 with leans.

**Checks so far.**
- Every point and the item carried over has its task, logged above as done: point 1
  in 13.8, point 2 in 13.1, point 3 (i) in 13.2–13.4 and 13.10, point 3 (ii) in 13.5,
  point 4 in 13.7, item 1 in 13.6; item 2 dropped (Q13.1).
- Q13.1–Q13.8 are answered in the Queries section; no other query was raised.
- The review (13.9) is applied: drafts 1–3, both of RJC's threads answered and
  resolved.
- ALIS outside `alis/dashboard/`: `git diff --stat 7343048 -- alis ':!alis/dashboard'`
  lists `alis/functions/lsffile.py`, `alis/functions/lsfspline.py` and
  `alis/save.py`, as Tasks 13.2 and 13.7 say.
- pre-commit passes on the stage's 38 changed or new Python files (none of RJC's
  untracked files among them).
- Coverage (`unit` tests of the changed modules): `resolution.py` 93%, `results.py`
  90%, `lsffile.py` 74%, `runs.py` 69%, `lsfspline.py` 48%, `runner.py` 33%. The
  runner's new paths (`Input`, `_Warnings.messages`, `run.json`'s warnings) run in
  the fitting process, which the `fast` tests start and coverage does not follow; no
  test was added for coverage alone.
- The window's text has no design reference (D44): the `gui` tests' checks.
- Batches, in the repository (pytest 9.0.2 and CI's 9.2.0), against Stage 12's close
  (2308 unit passed, 119 skipped (9.2.0: 2307, 120); 330 gui; 149 fast): unit 2356
  passed, 119 skipped (9.2.0: 2355, 120); gui 340 (9.2.0: 340); fast 155 (9.2.0: 155),
  the regression harness whole with its `test_bundle` mode and the 23 `context/` cases,
  `examples/lsf_spline` in mode (a) and out of mode (b) with its reason (Q13.7). No
  failures and no new warnings.

Stage 13 closed on 2026-11-24. CI waits for RJC's push. Stage 14's draft
(`claude_prompts/dashboard_stage14.md`) waits for RJC's answers to Q14.1–Q14.3.
