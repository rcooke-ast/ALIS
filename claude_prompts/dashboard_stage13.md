# Prompt file for ALIS software dashboard creation -- STAGE 13

> **Feedback from users, after Stage 12's list closed.** Stage 12 took a closed list
> of users' feedback (eight points on RJC's MWC137 project) and the item Stage 11
> carried over that joined it, with `lsfspline`'s smoothing made a FWHM in ALIS
> (Q12.2 (a)). Stage 13 takes the feedback that arrives after that list closed. RJC
> collects each point in the Feedback section below. Claude turns the list into a
> design, queries and tasks, each with its check, and builds them in order, with a
> review before the stage closes.
>
> The rules of Stages 3–12 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees. A question that comes up while a task is built pauses the
> work: it is asked in the Queries section, and its answer is logged there (Stage 11,
> Prompt 2).
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D138 at the close of Stage 12).
> "Q0.*n*" to "Q12.*n*" refer to `dashboard_stage0.md` to `dashboard_stage12.md`.
> What follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-17 (Stage 12's Task 12.13). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is. The items carried over are what Stage 12 found and did not do. Prompt 1
(2026-11-24) wrote the design and Tasks 13.1–13.9 from Feedback points 1–3 and the
item carried over that joins (Q13.1: item 1; item 2 dropped), with Q13.3–Q13.5. No
code was changed. Prompt 2 (2026-11-24) applied RJC's responses to Q13.3–Q13.5 (Task
13.8, the keys, takes Q13.5 (c) as well: a button on each tab), and added Feedback
point 4 as Task 13.7 (`lsffile` in km/s), with Q13.6; the keys became 13.8, the
review 13.9 and the close 13.10. No code was changed. Prompt 3 (2026-11-24)
applied RJC's response to Q13.6 ((a)) to Task 13.7, and checked the list; no further
query was needed, and no code was changed. Prompt 4 (2026-11-24) built Task 13.1,
and Task 13.2 up to Q13.7; after RJC's response, Tasks 13.2–13.8 and the review's
draft 1 (Task 13.9) were built the same day. RJC's comment on the review added Task
13.10 (Warnings and options to resolve…), with Q13.8, and the close became 13.11;
after RJC's response, 13.10 was built and the review's draft 3 published. Task 13.11
closed the stage on 2026-11-24 (D139–D146), and drafted `dashboard_stage14.md`.*

## Feedback

*Collected by RJC from users of the dashboard: one point per item, numbered, added as
it arrives. For each point:*
- *what the user did and saw, and what they expected instead;*
- *its kind: **(bug)**, **(change)** or **(new)**;*
- *the tab or dialog it concerns;*
- *where it can be reproduced: a project, or the model and data, and the steps. A
  user's files are referenced from here, and each test that needs them is skipped
  where they are absent (D99);*
- *who raised it, only if they agree to be named.*

*A point already on the list is not added twice: a note goes under the first, saying
how many users raised it. Claude reads this list and does not add to it.*

1. Add a help list that show's all of the keys that can be used. A bit like the Help menu keyboard shortcuts, but there's no 'b' and 'n' for b-value and column density.
2. When a fit has finished, some users have experienced a "Python has quit unexpectedly" error. I have attached the report as context. Please see the file `context/feedback01/alis_crash_report_01.txt` for the full report. This is a bug that needs to be fixed. Note that the GUI did not crash. I have also included the .model file that was used that caused the crash. Please see the file `context/feedback01/alis_crash_report_01.model` for the model file.
3. I have included another context file in `context/feedback01/MWC137_02/` that reveals several issues:
   (i) Running `alis alis2.mod` on the command line asks to load the datasets. I load the datasets (dataset 1 is `emsmwcs137.dat` with vFWHM resolution 2.14 km/s, change the fourth column of the data to `Ignore`, and dataset 2 is `EM_MWC__137.dat` with vFWHM resolution 2.72 km/s, change the fourth column of data to `Ignore`). The unassigned snips are then loaded into a single unassigned spectrum. If I now go to the `Fit` tab and select `Run`, the fit runs and finishes, but there is no `Final` model to display in the panels. The run finishes at the maximum number of chi-squared steps. Is there a reason the fits are not able to be displayed? There is also no option to inspect the Results. I was also not presented with a warning that the fit ended because the maximum number of chi-squared steps was reached. I think the user should be warned that the fit has ended because the maximum number of chi-squared steps was reached, perhaps in the "Fit Statistics" panel (possibly in red, so that this is seen as a warning).
   (ii) When I select "File" ==> "Save a copy with spectra...", a dialog appears where I can *open* a `.model` file, not *save* a `.model` file with spectra. This is a bug that needs to be fixed. The dialog should allow the user to save a copy of the model with spectra, not open a model file.
4. Currently, lsffile only allows the user to input a line-spread function (LSF) that has Delta Angstroms as the first column, and the profile as the second column. It would be good if there was a way to support LSFs that have the first column as Delta Velocity (in km/s) and the second column as the profile. This is a change request/new feature, and not a bug.

## Carried over from Stage 12

*Found by Claude during Stage 12 and not done there. RJC decides whether each joins
this stage (Q13.1).*

1. **(bug)** Two datasets of a fit opened without its spectrum can share a name.
   RJC's `alis2.mod` (`context/feedback01/MWC137/`), opened without its spectra,
   gives two datasets named "39KI7701": snips 6 and 7 are grouped apart by their
   resolution labels (`va`, `vab`), and each group is named by the `label=39KI7701`
   that both data lines carry (`project._build_rows`, Q2.4). The Data tab, the Regions tab's strips and every menu that names a dataset
   then show the same name twice. Load snips… already makes a taken name unique
   ("… 2"); an imported fit's rows could do the same. Found in Task 12.12's check of
   MWC137 (scratch `mwc_open.py`, logged).
2. **(change)** An unassigned dataset emptied before Task 12.12 (a project saved
   before, its dataset with neither snips nor data) advises Load snips…, but Load
   snips… of the same file puts the snip in a new dataset ("other (unassigned) 2")
   and leaves the emptied one (Q12.15). Q12.17's response keeps projects saved before
   as they are, and new `.model` files can be started; the emptied dataset could
   instead take the snip back.

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D138 and the stage documents. A point that asks for what a decision ruled out
  becomes a query that quotes the decision, not a task.
- **A bug becomes a test first.** It is reproduced from the user's files where there
  are some, then made into a synthetic test that fails before the fix and passes
  after it. The users' files are checked with scratch scripts, logged (D99).
- **Outside the dashboard.** Point 3 (i) needs ALIS to change (the `.mod.out` it
  writes for `lsfspline`), asked in Q13.3 and agreed; point 4 is a change to ALIS's
  `lsffile`, asked in Q13.6 and agreed. No other point needs ALIS to change.
- **Grouping and order.** Points are grouped by the part of the window they touch,
  and a bug comes before the work that builds on it.
- **Shown before it closes.** Every change a user sees is shown on a review page,
  built by `doc/dashboard/stage13/` with the earlier pages' stylesheet, and RJC's
  comments on it are applied before the stage closes.

### 13.1–13.8 — The Feedback list and the item carried over

*Written by Claude on 2026-11-24 (Prompt 1), from Feedback points 1–3, the item
carried over that joins (Q13.1: item 1), and the code at the start of the stage
(d5e6d40); point 4 at Prompt 2, the same day. Each point's design is in its task
below.*

**What was found.** Each point was reproduced off-screen with scratch scripts, on
copies of RJC's files in the session's scratchpad, before its task was written. No
code was changed.
- **Point 2** (Python quits after a fit).
  - The report is the fitting process's, not the window's: a Python whose parent is a
    Python, with two threads and no Qt loaded. Thread 0 aborts while the interpreter
    shuts down (`Py_Finalize`, closing a text stream as a module is cleared, in
    `_enter_buffered_busy`); thread 1 waits on a pipe through the same buffered
    reader. That is the runner's `Commands` thread (`runner.py`), a daemon that reads
    its standard input for `stop`.
  - At shutdown, Python closes `sys.__stdin__`, waits a second for the lock that the
    waiting thread holds, and aborts ("could not acquire lock … at interpreter
    shutdown, possibly due to daemon threads"). The dashboard keeps that pipe open,
    so every run ends this way: `examples/metal_line_abs` run through `runs.Process`
    ends with −6 (SIGABRT). Closing the pipe at `finished` is not enough: by then the
    process may be shutting down already (it was, in the check), and a daemon thread
    cannot give the lock back once shutdown has begun.
  - The result is stored before `finished` is sent, so nothing is lost; the window
    does not crash, and the process's standard error goes nowhere, so its message is
    never seen. macOS reports each abort, which the user sees as "Python quit
    unexpectedly". The tests of the process never looked at how it exits.
  - The dashboard's side has the same shape: `Process.detach` (closing the window
    with Keep it running) closes the pipe that its reader thread waits on, and waits
    for that thread's lock until the next message arrives: 1.6 s on RJC's
    `alis_crash_report_01.model`, one iteration of its fit, and as long as one
    iteration of a slow fit.
- **Point 3 (i)** (no Final model, no Results, no warning). On RJC's project after
  the run (`MWC137_02/data/mwc137.model`):
  - The run is stored, complete: 100 iterations, χ² 2914466 → 24919, and ALIS's reason
    "The maximum number of iterations has been reached" (its status 5), with its
    `.mod.out`, fits and covariance.
  - Its `.mod.out` does not read. ALIS writes the two `lsfspline` resolutions
    (specids 6 and 7) without their `locations`, and reading them back fails:
    "locations must be set for lsfspline …" (`deferred_work.md` §7.11). So the run
    gives none of its 204 free parameters' best-fit values, and Inspect's Final and
    Results both show only "The latest run has no best-fit values."; Results shows no
    statistics either. With the `locations` put back, the same `.mod.out` gives every
    value (300 in all).
  - Why ALIS drops them: `lsfspline`'s loader marks its values as given but not its
    keyword (`lsf` and `apod` mark theirs), so `parout` leaves `locations` out; and
    `save.py` joins what `parout` gives with commas, while `lsfspline` reads its
    keywords after a `;`.
  - The reason is shown already, as Results' Fit statistics row "Stopped because",
    in plain text; the status bar at the end said "The fit has finished; the Fit tab
    shows its best fit.", though it showed none.
  - The datasets are as designed (Task 12.12): snips 0, 4, 7, 10, 12 and 13 are pixels
    of `emsmwcs137.dat`; 5 and 14 differ from it by up to 10⁻⁴ of their flux (cut, it
    seems, from an earlier version of the spectrum), and 3, 8 and 9 have edited
    errors (0.38 times the source's), so each is kept as data of its own, unassigned;
    specid 6 differs in K I 7701's core (Stage 12, point 1). EM_MWC__137 then holds no
    snip.
- **Point 3 (ii)** (Save a copy with spectra… opens a file). The action asks the
  window's `choose` for the kind `"save_copy"`, which it does not know, and `choose`
  takes any kind but `"save"`, `"folder"` and `"open_many"` as `"open"`. Its test
  answers `choose` by kind, so it never met the dialog. No other call passes a kind
  that `choose` does not know.
- **Point 1** (a list of the keys). The sheet (F10, `ShortcutsDialog`) lists the
  menus' actions, from `actions.ACTIONS`, with a note on the wheel, ⇧, ⌂, ↔ and ↕
  (`MOUSE_NOTE`). It lists none of the keys handled outside the menus: N and B held
  during a drag (`DragKeys`), Esc on the Components tab (+ Add component waits no
  more, else the selection is cleared), = on the Regions tab (raises the order, as
  +), and Delete in the knots dialog and in the welcome screen's recent projects; nor
  the mouse's gestures, which `doc/ALIS_workflow.md` §4.3 describes (each tab's
  drags, the right drags, double-clicks and right-click menus).
- **Item 1** (two datasets of one name; Q13.1's response). `project._build_rows`
  names each inferred row by its data lines' `label`; Load snips… makes a taken name
  unique ("… 2", `loadfit`), and the inferred rows do not. On MWC137_02 with its
  spectra, specid 7 is now in emsmwcs137, so the two names do not meet there; the
  Stage 12 MWC137 project opened without its spectra still shows them.
- **Point 4** (an LSF file in km/s; added at Prompt 2).
  - ALIS's `lsffile` (`alis/functions/lsffile.py`) reads a text file: Δλ in Å, then
    one kernel per column (`name:`, `index:`). `call_CPU` interpolates the kernel at
    each sub-pixel's offset in Å from the centre one, and `getminmax` takes its FWHM
    in Å for the buffer (Q8.13). Nothing reads a velocity.
  - `vfwhm` already samples its Gaussian at each sub-pixel's velocity offset from the
    centre, c·(λ/λc − 1), and a kernel in km/s can be sampled in the same way.
  - In the dashboard, `resolution.parse_kernels` checks a chosen file as Å and gives
    each kernel's FWHM in Å; the Snip box's note reads "FWHM … Å (… km/s)";
    `components.fwhm_kms` converts at the snip's wavelength. The Snip box's buffer
    check calls ALIS's own `getminmax`, so it follows ALIS.
  - `examples/lsf_file` (a regression case, in Å) and `test_bundle.py` use `lsffile`;
    nothing tests `lsffile` itself.

**What stays as it is.**
- The fitting process's protocol (the header, the project, `stop`), and Stop, Cancel
  and Keep it running as the user meets them.
- How snips are matched to a spectrum (D19, Q1.8, Task 12.12): MWC137_02's
  unassigned snips are as designed.
- `chisq maxiter`, ALIS's default and the model's: a run stopped at it is stored and
  shown as any other, and warns (Task 13.4).
- ALIS outside `alis/dashboard/` changes only as Q13.3 agrees (`lsfspline`'s
  `.mod.out`) and Q13.6 agrees (`lsffile`'s unit). An `lsffile` line written before
  reads its file in Å, as now.
- The menus' shortcuts, and F10 opening the sheet.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage13_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (13.1, 13.2, …), and per item carried over
from Stage 12 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*The last task is the review (a page built by `doc/dashboard/stage13/`), and the one
after it closes the stage, as Stage 12's Tasks 12.11 and 12.13 did.*

*Tasks 13.1–13.9 were written by Claude on 2026-11-24 (Prompt 1), with the leans of
Q13.3–Q13.5. At Prompt 2 (the same day), RJC's responses agreed with 13.2 (Q13.3 (a))
and 13.4 (Q13.4 (b)), and added a button on each tab to the keys (Q13.5 (a) and (c));
Feedback point 4 became Task 13.7, written with the lean of Q13.6 (agreed at Prompt
3), before the keys so
that their list covers what the stage adds. The order puts a bug before the work
that builds on it, and keeps work on one part of the window together:*
- *the fitting process: point 2's abort (13.1);*
- *a run's results, point 3 (i): ALIS's `.mod.out`, then a run that cannot be shown,
  then the warning (13.2–13.4);*
- *the File menu and the Data tab: point 3 (ii)'s dialog, item 1's names (13.5,
  13.6);*
- *the resolution: point 4's LSF file in km/s (13.7);*
- *Help: point 1's keys, the sheet and each tab's button (13.8);*
- *the review (13.9), and the close (13.10).*

*RJC's comment on the review (2026-11-24) added Task 13.10, a dialog of the fit's
warnings, written with the leans of Q13.8 and built after RJC's response; the close
became 13.11.*

**13.1 — The fitting process ends without an abort (point 2, bug; `runner.py`,
`runs.py`; F6, Q6.14, Q7.17).**
- **The bug.** The runner's `Commands` thread waits on the standard input through
  Python's buffered reader, holding its lock. At the end of every run, Python's
  shutdown closes that reader, cannot take the lock, and aborts (SIGABRT), which
  macOS reports as "Python quit unexpectedly". The result is stored before, so
  nothing is lost. On the dashboard's side, `Process.detach` (Keep it running) waits
  for its reader thread's lock until the process's next message.
- **The fix.**
  - **The fitting process** reads its standard input's descriptor directly
    (`os.read`): the header, the project and then the commands, so that no buffer of
    Python's is shared with the thread, and a `stop` sent early is not left in one.
    `sys.__stdin__` is then never read, and closes at shutdown with nothing to wait
    for. `sys.stdin` still goes nowhere (Q7.17: a forked worker closes it).
  - **The dashboard** closes the process's standard input once `finished` arrives,
    and reads the messages from the descriptor in the same way, so that `detach`
    returns at once.
- **Check.** Made into tests first:
  - a `unit` test: a Python process that starts the runner's command reader on a pipe
    and ends while the pipe is held open exits with 0 (before, −6);
  - in `test_dashboard_runs_process.py` (`fast`, as its other tests): a run of
    `examples/metal_line_abs`, its standard input held open after `finished`, ends
    with 0 (before, −6); `stop` stops a run at its iteration, also when sent while
    the project is still being handed over; `detach` returns within 0.1 s while a fit
    iterates (before, at the next iteration's message);
  - the existing tests of runs, Stop, Cancel, Keep it running and random restarts
    pass unchanged;
  - RJC's `alis_crash_report_01.model`, run through the dashboard's process, ends with
    0 (scratch script, logged).

**13.2 — A run's `.mod.out` that ALIS reads back: `lsfspline`'s `locations` (point 3
(i), bug; `alis/functions/lsfspline.py`, `alis/save.py`, `runs.py`; D89; Q13.3;
`deferred_work.md` §7.11).** *Q13.3 (a), RJC's response.*
- **The bug.** ALIS writes an `lsfspline` resolution into the `.mod.out` without its
  `locations`, so the run's model does not read: the dashboard gets none of its
  best-fit values (no Final, no Results), and `run_alis` cannot start from it.
- **The fix (ALIS).** `lsfspline`'s loader marks `locations` as given, as `lsf` and
  `apod` mark their keywords, and the data line is written
  `lsfspline(values…;locations:…)`, which `lsfspline` reads. `CHANGELOG.md` says so.
- **Runs stored before.** The dashboard reads a `.mod.out` whose `lsfspline` lacks its
  `locations` by taking them from that run's own input model (`input.mod`, the data
  line of the same specid), so that RJC's MWC137_02 run shows its best fit without
  being run again. Nothing is written into the run.
- **`examples/lsf_spline`** joins the regression harness (§7.11's "To finish"): its
  references from a fit, and the harness on it, CI's `examples` batch included.
  §7.11 is then done. *Q13.7 (b), RJC's response: its evaluation at the best fit
  (mode (b)) meets ALIS's sub-pixels, set at load from the starting values, so it is
  left out (`FIXEDPARAM_EXCLUDE`), and recording the sub-pixels waits in §7.10, the
  key item of Stage 14.*
- **Check.** Made into tests first:
  - in `tests/test_lsfspline.py` (`unit`): the `.mod.out` that ALIS writes for a model
    with an `lsfspline` resolution has its `locations`, and reads back with the same
    kernel (before, "locations must be set"); each other resolution function that
    takes keywords (`lsf`, `lsffile`, `apod`, `multivfwhm`) reads back the same way;
  - in `test_dashboard_runs.py`: `out_values` of a `.mod.out` without `locations`,
    with its run's input model, gives every free parameter's value (before, none);
  - RJC's MWC137_02 project gives its run's 204 free values, its Final model and its
    Results (scratch script, logged);
  - the regression harness, its `test_bundle` mode included, with
    `examples/lsf_spline` as a case (its fit and its bundle; Q13.7 (b)) and every
    other reference unchanged;
    `test_function_interface.py` passes.

**13.3 — A run that cannot be shown says why, and shows its statistics (point 3 (i);
`results.py`, `qt/tabs.py`, `qt/fitresults.py`, `qt/window.py`; S19, S31).**
- **What.**
  - **Why.** A run whose outputs do not read says so, with ALIS's reason through the
    gate, on Inspect and on Results: "The outputs of the latest run do not read:
    locations must be set for lsfspline …", in place of "The latest run has no
    best-fit values.".
  - **Results** shows such a run's Fit statistics (χ², iterations, why it stopped,
    from `run.json`), with the reason in place of the table.
  - **The end of the run.** The status bar says the fit has finished but that its best
    fit cannot be shown, and why, in place of "the Fit tab shows its best fit".
- **Check.** New tests, on a synthetic run whose `.mod.out` is broken on one line:
  - in `test_dashboard_results.py`: `why_not` names the problem (before, "no best-fit
    values"), and `statistics` gives the run's;
  - in `test_dashboard_qt_results.py`: Results shows the statistics and the
    reason; in `test_dashboard_qt_window.py`, the status bar's message at the end of
    such a run names it;
  - a hidden value in ALIS's reason is masked (the gate).

**13.4 — A fit stopped at its maximum number of iterations warns (point 3 (i);
`results.py`, `qt/fitresults.py`, `qt/window.py`; S19, S31; Q13.4).** *Q13.4 (b), in
the three places, RJC's response.*
- **What.**
  - **Which runs.** ALIS's status 5, "The maximum number of iterations has been
    reached" (from `run.json`'s `status_code`; for a run of `run_alis`, from its
    `.mod.out`'s Convergence Reason); and a convergence check (`run convergence`)
    that found free parameters not converged.
  - **Fit statistics.** The row "Stopped because" is in red, with what to do: "The
    fit stopped after `chisq maxiter` (100) iterations, before it converged:
    Continue from best fit…, then Run, carries on from here; or raise `chisq
    maxiter`." The row "Convergence" is in red when the check failed.
  - **The end of the run.** The status bar's message says so.
  - **The run history** marks the run ⚠, its tooltip the reason.
  - The red is the theme's warning colour, in light and dark.
- **Check.** New tests:
  - in `test_dashboard_results.py`: `statistics` flags status 5 and a check not
    converged (before, nothing flags them), and not statuses 1–4; a run of `run_alis`
    is flagged from its `.mod.out` alone;
  - in `test_dashboard_qt_results.py` and `test_dashboard_qt_window.py`: the rows
    in red with the advice, the status bar's message, and the history's ⚠ with its
    tooltip;
  - the window's text has no design reference (D44).

**13.5 — Save a copy with spectra… asks where to save (point 3 (ii), bug;
`qt/window.py`; Q12.7).**
- **The bug.** The action asks `choose` for `"save_copy"`, a kind it does not know,
  and the Open dialog comes up.
- **The fix.** The action asks with the Save dialog, starting from
  "…_with_spectra.model"; `choose` refuses a kind it does not know, rather than
  taking it as `"open"`.
- **Check.** Made into tests first, in `test_dashboard_qt_relink.py`, with
  `QFileDialog`'s functions replaced:
  - File → Save a copy with spectra… calls `getSaveFileName` with the suggested name
    (before, `getOpenFileName`);
  - every action that asks for a file (Open, Save as…, Load snips…, Export…, Relink…,
    Save a copy with spectra…) passes a kind that `choose` knows;
  - the existing test of Save a copy with spectra… passes.

**13.6 — Each dataset its own name (item 1 carried over, bug; `project.py`; Q2.4).**
- **The bug.** A fit opened without its spectrum can give two datasets one name (two
  groups of data lines whose `label` is the same), shown twice on the Data tab, the
  Regions tab's strips and every menu that names a dataset.
- **The fix.** An inferred row takes a name no other row has, as Load snips… does:
  "39KI7701", then "39KI7701 (2)", its number in parentheses so that it is not read
  as part of the name (RJC's comment on the review; Load snips… likewise). Rows read from a project's `ui/project.json` keep
  their names (Q12.17's response); Rename… on the Data tab tells them apart.
- **Check.** Made into tests first:
  - in `test_dashboard_project.py`: a synthetic fit whose two groups of data lines
    share a `label` gives two rows of different names (before, the same), and a third
    "… 3"; a project saved with two rows of one name opens as it is;
  - in `test_dashboard_qt_data.py`: the Data tab and the Regions tab's strips name
    each dataset once;
  - Stage 12's MWC137 opened without its spectra (scratch script, logged).

**13.7 — An LSF file in km/s (point 4, new; `alis/functions/lsffile.py`,
`resolution.py`, `datasets.py`, `components.py`, `qt/regions.py`, `qt/data.py`; D89,
Q8.7, Q8.13; Q13.6).** *Q13.6 (a), RJC's response.*
- **What (ALIS).** `lsffile` takes a keyword `unit`: `A`, the default (the first
  column is Δλ in Å, as now), or `kms` (Δv in km/s).
  - With `kms`, the kernel is sampled at each sub-pixel's velocity offset from the
    centre one, c·(λ/λc − 1), as `vfwhm` makes its Gaussian, and normalised as now.
  - The buffer (`getminmax`) takes the kernel's σ in km/s to Å at each end of the fit
    range; `fwhm()` gives the width in the file's unit.
  - The keyword is marked as given, so that the `.mod.out` keeps it (as Task 13.2
    does for `lsfspline`). An unknown unit is refused, with its reason.
  - A model without it reads as before. `CHANGELOG.md`, the function's docstring and
    `doc/ALIS_workflow.md`'s description of `lsffile` say so.
- **What (the dashboard).**
  - Where an LSF file is chosen (the Snip box, and the Data tab's resolution), a
    drop-down beside Choose a file…: **First column: Δλ (Å) | Δv (km/s)**, starting
    at Å, or at the line's own unit. Choosing the other unit writes `unit:kms` or
    takes it away, as one step.
  - The note gives the kernel's FWHM in the file's unit, and in the other at the
    snip's wavelength ("k.dat · FWHM 6.3 km/s (0.12 Å)"), so that a wrong unit shows
    at once. The button's tooltip names both units.
  - `resolution.parse_kernels`, `components.fwhm_kms` and `lsf_fwhm` take the unit;
    the buffer check follows ALIS's `getminmax`; Export… keeps the keyword.
- **Check.** Made into tests first:
  - a new `tests/test_lsffile.py` (`unit`): a Gaussian kernel in km/s of FWHM 5 km/s
    gives the model that `vfwhm(5.0)` gives, to 10⁻⁴ of the continuum, on grids
    linear in wavelength and in log wavelength (before, the keyword is refused); the
    same kernel in Å, at a line's wavelength, gives the model it gives now; a km/s
    kernel's buffer is `vfwhm`'s of the same FWHM; a `.mod.out` with `unit:kms` reads
    back with it; an unknown unit is refused with its reason;
  - `examples/lsf_file`, the regression harness and its `test_bundle` mode pass
    unchanged; `test_function_interface.py` passes;
  - in `test_dashboard_datasets.py`: a km/s file's FWHM in both units, the step that
    writes `unit:kms` (ALIS reads the model, and one undo gives back the text), and
    the buffer check;
  - in `test_dashboard_qt_regions.py` and `test_dashboard_qt_data.py`: the drop-down,
    its step, and the note's FWHM in both units;
  - the window's text has no design reference (D44).

**13.8 — Every key and gesture in one list, and each tab's own (point 1; `actions.py`,
`qt/dialogs.py`, `qt/components.py`, `qt/regions.py`, `qt/knots.py`, `qt/tabs.py`,
`qt/window.py`; F10; Q13.5).** *Q13.5 (a) and (c), RJC's response. (13.7 at Prompt 1.)*
- **What.**
  - **The sheet** (Help → Keyboard shortcuts, F10) keeps its table of the menus, and
    gains a second, **Keys and mouse in the views**: each key held or pressed, and
    each gesture of the mouse, with where it works (every spectrum, a tab, the knots
    dialog, the `.mod` panel, the welcome screen) and what it does. Among them: N held
    during a drag on the Components tab (log N alone), B (b alone), Esc, =, Delete,
    the wheel and ⇧ with it, each tab's drags and right drags, the double-clicks and
    the right-click menus. `MOUSE_NOTE`'s text becomes rows of it.
  - **One list** in `actions.py` holds these rows, and the code takes its keys from it
    (`DragKeys`'s N and B, the Regions tab's =, the knots dialog's Delete), so that a
    key is not added without its row.
  - The sheet opens at the rows of the tab shown. `doc/ALIS_workflow.md` §4.3 names it.
  - **Each tab's button** (Q13.5 (c)): a small button in the top-right corner of the
    Data, Regions, Components, Fit and Figure tabs, its tooltip "Keys and mouse on
    this tab", opens that tab's rows of the same list: its own, and those of every
    spectrum where it shows one. **Show all** in it opens the sheet at those rows.
- **Check.**
  - in `test_dashboard_actions.py`: the list has N and B for the Components tab
    (before, nothing), and every key that the code handles outside the menus
    (`DragKeys.KEYS`, the tabs' `QShortcut`s, the dialogs' key handlers) has its row;
    each tab's rows are the list's rows for that tab and for every spectrum (for the
    tabs that show one), in the sheet's order;
  - in `test_dashboard_qt_window.py`: the sheet shows both tables, and opens at the
    rows of the tab shown; each tab has its button, which shows that tab's rows
    alone, and Show all opens the sheet;
  - the existing tests of N and B during a drag pass unchanged;
  - the window's text has no design reference (D44).

**13.9 — The review.** *(13.8 at Prompt 1.)*
- **What.** A review page built by `doc/dashboard/stage13/`, with the earlier pages'
  stylesheet. Screenshots at 1440×900 of:
  - Inspect and Results of a run with an `lsfspline` resolution, its best fit drawn
    (13.2);
  - Results of a run whose outputs do not read: its statistics and why (13.3);
  - Fit statistics of a run stopped at its maximum number of iterations, in red, with
    the run history's ⚠ and the status bar's message (13.4);
  - the Data tab with two datasets of one `label`, named apart (13.6);
  - the Snip box with an LSF file in km/s: the drop-down and the note (13.7);
  - the sheet with its keys and gestures, and a tab's button with that tab's list
    (13.8).

  Where a still cannot show it (the process's exit, 13.1; the system's Save dialog,
  13.5), the page names the test. It is published as a private page, and RJC's
  comments are applied.
- **Check.** The page builds from screenshots of the `gui` tests' scenes (synthetic
  projects, `tests/dashboard_fits.py`). Every task of 13.1–13.8 that a user sees has
  its picture, or its test named. RJC's comments are applied and logged.

**13.10 — Warnings and options to resolve… (RJC's comment on the review; point 3
(i); `results.py`, `runner.py`, `qt/fitresults.py`, `qt/dialogs.py`; S19; Q13.4,
Q13.8).** *Added after the review's draft 1, written with the leans of Q13.8, and
built after RJC's response (Q13.8 (a), (a), (a)).*
- **What** (RJC's comment: in place of scrolling the Fit statistics pane).
  - **The button**, in red, "Warnings and options to resolve…", between the Fit
    statistics and Correlations panes on Results, shown only while the shown run has
    a warning (Q13.8 (i)).
  - **The dialog** lists each warning of the run with a suggestion to resolve it,
    left blank where there is none: stopped at `chisq maxiter` ("Continue from best
    fit…, then Run, carries on from here; or raise chisq maxiter"); a convergence
    check that failed ("Continue from best fit…, then Run"); and ALIS's own warnings
    during the run (no suggestion) (Q13.8 (iii)).
  - **ALIS's warnings are kept with the run**, in `run.json` (they were held in
    memory only, for the status bar), so a run opened later lists them too.
  - **Fit statistics** keeps "Stopped because" and "Convergence" in red; the row
    "What to do" moves into the dialog, so the pane needs no scrolling (Q13.8 (ii)).
- **Check.** New tests:
  - in `test_dashboard_results.py`: the warnings of a run (status 5, a failed check,
    ALIS's warnings from `run.json`), each with its suggestion or none; none for a
    run that converged; Fit statistics' rows without "What to do";
  - in `test_dashboard_runs_process.py`: a run that ALIS warns about keeps the
    warning in `run.json`;
  - in `test_dashboard_qt_results.py`: the button red and shown only with a
    warning, between the two panes; the dialog's rows; the gate masks a hidden value
    in ALIS's warning; no design reference (D44);
  - the review page shows the button and the dialog.

**13.11 — Close the stage.** *(13.9 at Prompt 1, 13.10 until RJC's comment added the
task above.)*
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill (its "Stage 13 additions") up to
  date; `CHANGELOG.md` says that a fit's process no longer aborts at its end, that a
  `.mod.out` keeps `lsfspline`'s `locations`, and that `lsffile` reads a kernel in
  km/s with `unit:kms`. Record the decisions in `ALIS_v2_dashboard_prompts.md` (D139
  onwards, from Q13.1–Q13.6 and the review), saying which earlier decisions they
  refine: F6 and Q7.17, F10, S19 and S31, D89, Q2.4, Q8.7 and Q8.13, and Q12.7.
  Update `dashboard_stage0.md`: Stage 13 done, the "After v1" list, and what follows:
  Stage 14 on ALIS's sub-pixels, then the documentation (Q13.2's and Q13.7's
  responses). Update `deferred_work.md`: §7.11 done (Q13.3 (a)) and §7.10 with the
  sub-pixels a `.mod.out` is evaluated on (Q13.7 (b), both done during Task 13.2);
  §1.4 (the GPU), §7.8 and §7.9 stay, and anything this stage leaves undone is
  added.
- **Found and not done.** What was found during the stage and not built goes to Stage
  14's "Carried over from Stage 13", with a query, as Stage 12's close did.
- **What follows (Q13.2's and Q13.7's responses).** Draft
  `claude_prompts/dashboard_stage14.md`, as Stage 12's close wrote this document. Its
  key item is `deferred_work.md` §7.10, ALIS's sub-pixels: the default `nsubmin` of
  10, and each snip's sub-pixels recorded in the `.mod.out`, with every reference
  that moves regenerated and `examples/lsf_spline`'s mode (b) back in the gate. Its
  draft design adds the items of `deferred_work.md` related to it, for RJC to review
  and prioritise, and it keeps a Feedback section for what arrives after this list
  closes. Stage 14 comes next, before the documentation.
- **Check.**
  - Every point of the Feedback list and the item carried over has its task logged
    as done with its check passing: point 1 in 13.8, point 2 in 13.1, point 3 (i) in
    13.2–13.4, point 3 (ii) in 13.5, point 4 in 13.7, and item 1 in 13.6; item 2 is
    dropped (Q13.1's response).
  - Q13.1–Q13.8, and any query raised while the tasks were built, are answered in the
    Queries section.
  - The review of 13.9 is applied, with Task 13.10 (RJC's comment).
  - The `unit`, `gui` and `fast` batches pass, also under CI's pytest. Their counts go
    in the log, against Stage 12's close: 2308 unit tests passed with 119 skipped
    (2307 and 120 under pytest 9.2.0), 330 `gui` tests, and 149 `fast` tests, the
    regression harness and its `test_bundle` mode included.
  - CI passes on both platforms, the `gui` job within its 90 minutes.
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44), and pre-commit passes on every
    changed file.
  - ALIS outside `alis/dashboard/` changes only as Tasks 13.2 and 13.7 say: `git diff
    --stat 7343048 -- alis ':!alis/dashboard'` lists `alis/functions/lsfspline.py`,
    `alis/functions/lsffile.py` and `alis/save.py`, against Stage 12's close; the
    regression harness passes, its references unchanged but `examples/lsf_spline`'s,
    which are new (its fit and its bundle; Q13.7 (b)). This stage's tests are synthetic and need none of RJC's files; RJC's files
    are checked with scratch scripts, logged.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs, the Figure
  tab, and the lessons of Stages 10–12 (their "additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D138.
- `claude_prompts/dashboard_stage12.md` and its log: how users' feedback was handled
  last time, from the list to the close, the work pausing for RJC's queries, and
  the review's comments applied (drafts 1–7).
- `claude_prompts/deferred_work.md`: what was deferred (§1.4, the GPU kept on the
  device, a series of its own; §7, from the dashboard's stages, §7.10 and §7.11
  from Stage 12).
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 12's builder is `doc/dashboard/stage12/`; its scenes come from
  `tests/dashboard_synthetic.py` and `tests/dashboard_fits.py`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude: Q13.1 and Q13.2 with this document (2026-11-17), answered by RJC
and applied at Prompt 1 (2026-11-24); Q13.3–Q13.5 at Prompt 1, answered and applied
at Prompt 2 (2026-11-24); Q13.6 at Prompt 2, for Feedback point 4, answered and
applied at Prompt 3 (2026-11-24); Q13.7 while Task 13.2 was built (Prompt 4); Q13.8
from RJC's comment on the review. Each gives Claude's
lean. A question that comes up while a task is built pauses the work; it is asked
here, and its answer is logged here (Stage 11, Prompt 2).*

**Q13.1 — The items carried over from Stage 12.** For each, two ways:
- **(a) It joins this stage**, after the feedback's tasks.
- **(b) It waits** for a user to ask for it, in `deferred_work.md` §7.

My lean: (a) for item 1, a small fix that tells two datasets apart wherever they
are named; (b) for item 2, which only a project saved before Task 12.12 can meet,
and which Q12.17's response leaves as it is.

**Response:** I think item 1 should join this stage. We can ignore item 2.

*Applied (Prompt 1, 2026-11-24): item 1 joins the stage as Task 13.6, after the
feedback's tasks. Item 2 is dropped, and is not added to `deferred_work.md`.*

**Q13.2 — What follows this stage.** Two ways:
- **(a) Another stage of feedback** (Stage 14) for what arrives after this list
  closes, as now.
- **(b) Another series first**: the GPU kept on the device (`deferred_work.md` §1.4),
  the documentation (`documentation_stage0.md`), or an item marked "Later" in
  `dashboard_stage0.md`.

My lean: (a), unless RJC names one of (b).

**Response:** We should create a stage 14 document that builds on this feedback. I will likely pause the dashboard development between stages 13 and 14 to focus on the documentation.

*Applied (Prompt 1): the close (Task 13.11 since RJC's comment on the review) writes `dashboard_stage14.md` for the
feedback that arrives after this list closes, and records in `dashboard_stage0.md`
that the dashboard pauses for the documentation (`documentation_stage0.md`) before
Stage 14.*

**Q13.3 — The `.mod.out` of a fit with `lsfspline` (point 3 (i), Task 13.2).** ALIS
writes an `lsfspline` resolution into the `.mod.out` without its `locations`, so
neither the dashboard nor `run_alis` can read that `.mod.out` back
(`deferred_work.md` §7.11). This is why RJC's MWC137_02 run has no Final model and no
Results. The fix is in ALIS, outside the dashboard. Two ways:
- **(a) ALIS writes them.** The `.mod.out` keeps `;locations:…` on the data line
  (`alis/functions/lsfspline.py`, and `alis/save.py` if its join needs it), and
  `examples/lsf_spline` joins the regression harness (§7.11's "To finish"). The
  dashboard also reads a run stored before the fix, taking the `locations` from that
  run's input model, so that RJC's MWC137_02 run shows its best fit without being
  run again.
- **(b) The dashboard alone** takes the `locations` from each run's input model.
  ALIS stays as now, so its `.mod.out` still cannot be run again from the command
  line, and §7.11 waits.

My lean: (a). The `.mod.out` is where a next fit starts; mended in ALIS, it is mended
for every reader.

**Response:** I agree with option (a).

*Applied (Prompt 2, 2026-11-24): Task 13.2 as written: ALIS's `.mod.out` keeps
`lsfspline`'s `locations`, runs stored before are read with their input model's, and
`examples/lsf_spline` joins the regression harness.*

**Q13.4 — The warning when a fit stops at its maximum number of iterations (point 3
(i), Task 13.4).** Results' Fit statistics already reads "Stopped because: The maximum
number of iterations has been reached", in plain text. Which runs warn, in red with
what to do ("Continue from best fit…, then Run; or raise `chisq maxiter`"):
- **(a) ALIS's status 5 alone** (`chisq maxiter` reached).
- **(b) Also a convergence check** (`run convergence`) that found free parameters
  not converged, whose row "Convergence" is in plain text now too.

And where, besides Fit statistics: the status bar's message at the end of the run,
and a ⚠ on the run in the run history, its tooltip the reason.

My lean: (b), in all three places. Both say that the best fit may not be the best, and
the status bar and the history are seen without opening Results.

**Response:** Thanks for confirming the Fit Statistics already includes this information. I agree with your lean. Please include this in the stage 13 development.

*Applied (Prompt 2): Task 13.4 as written: status 5 and a convergence check that
failed warn, in red in Fit statistics with what to do, in the status bar's message at
the end of the run, and as ⚠ in the run history.*

**Q13.5 — The list of keys (point 1, Task 13.8; 13.7 when asked).** Three ways:
- **(a) One sheet.** Help → Keyboard shortcuts (F10) keeps its table of the menus, and
  gains a second, "Keys and mouse in the views": every key and gesture that works
  outside the menus, where it works and what it does (N and B held during a drag,
  Esc, =, Delete, ⇧ with the wheel, the drags, right drags, double-clicks and
  right-click menus). It opens at the rows of the tab shown.
- **(b) A sheet of its own**, Help → Keys and mouse…, with that second table; F10 as
  now.
- **(c) As (a), and each tab's own keys** in a "?" beside its zoom bar, its tooltip
  listing them.

My lean: (a). There is one place to look, which the Help menu already names. The list
is kept in `actions.py`, and the code takes its keys from it, so that the two cannot
part.

**Response:** I agree with option (a). It sounds like option (c) could also be a quick shortcut that would also be beneficial. A small button on each Tab that opens a list of the keys and gestures for that Tab would be useful. It basically filters down the complete list that is available in the Help menu. I would like to see that added to the list of tasks for this stage.

*Applied (Prompt 2): Task 13.8 (13.7 at Prompt 1) takes (a) and (c). The sheet gains
its second table, and each tab (Data, Regions, Components, Fit, Figure) a small button
in its top-right corner that opens that tab's rows of the same list (its own, and
every spectrum's where it shows one), with Show all for the sheet. Point 4's task
comes before it, so that the list covers what the stage adds.*

**Q13.6 — An LSF file whose first column is in km/s (point 4, Task 13.7).** ALIS's
`lsffile` reads its file's first column as Δλ in Å: `call_CPU` interpolates the kernel
at each sub-pixel's offset in Å, and `getminmax` takes its FWHM in Å. A kernel in km/s
would be sampled at each sub-pixel's velocity offset from the centre instead,
c·(λ/λc − 1), as `vfwhm` makes its Gaussian. This changes ALIS outside the dashboard.
How the model says which unit the file uses, three ways:
- **(a) A keyword of `lsffile`:** `lsffile(name:k.dat,unit:kms)`. Without it, Å, so
  every model written before reads as before. One function, and one entry in the
  resolution menus.
- **(b) A function of its own,** `vlsffile`, its first column in km/s, as `vfwhm`
  stands beside `Afwhm`.
- **(c) A line at the top of the file** (`# unit: km/s`), the model unchanged.

In the dashboard, either way: where an LSF file is chosen (the Snip box, the Data
tab), a drop-down beside Choose a file…, First column: Δλ (Å) | Δv (km/s), starting
at Å. The note gives the kernel's FWHM in the file's unit and in the other at the
snip's wavelength, so that a wrong unit shows at once ("FWHM 6.3 Å (330 km/s)").

My lean: (a), with the drop-down. The unit stays on the model's line, where a reader
of the model sees it and the `.mod.out` keeps it; (b) doubles a function for one word
of difference, and (c) hides the unit in a file that a reader of the model never
opens.

**Response:** I agree with option (a). The drop-down beside the Choose a file… button is a good idea. It will make it clear to the user which unit is being used, and allow them to change it if needed. Please include this in the stage 13 development.

*Applied (Prompt 3, 2026-11-24): Task 13.7 as written: `lsffile` takes `unit:kms`
(Å without it, as before), and the Snip box and the Data tab have the drop-down,
First column: Δλ (Å) | Δv (km/s), beside Choose a file…, with the kernel's FWHM in
both units.*

**Q13.7 — `examples/lsf_spline` evaluated at its best fit (Task 13.2; raised while
it was built, Prompt 4).** With `locations` written, `examples/lsf_spline` joins the
harness: its fit (mode (a)) and its bundle pass, against references written from a
fit (8.5 s, 342 iterations, χ² 459.6 → 450.1). Its evaluation at the best fit (mode
(b)) fails: the model differs at 15 of 460 pixels, by up to 2.3 times the error.
- **Why.** ALIS sets each snip's sub-pixels when it loads a model, from the starting
  values (`load.py`, each function's `set_vars`). The fit started at a smoothing of
  4 km/s and used 6 sub-pixels; the `.mod.out` starts at its best fit, 0.12 km/s, and
  takes ALIS's most, 21. With the sub-pixels pinned at 6, the evaluation agrees with
  the fit exactly; at 5, or 7 to 21, it does not.
- **The dashboard is not affected.** It draws a run's best fit with the run's own
  input model, so its sub-pixels are the fit's: on RJC's MWC137_02 run, Final equals
  each snip's `_fit.dat` to 0.000 of the error, at all 12 snips, the two `lsfspline`
  snips included.
- **Who meets it.** Only a `.mod.out` run again, or evaluated, from its best fit (with
  `run_alis`, or Continue from best fit then Run), for a function whose width moves
  far in a fit: `lsfspline`'s smoothing most of all.

Three ways:
- **(a) ALIS records each snip's sub-pixels in the `.mod.out`**, as a data-line
  keyword (as Stage 5.4's `bufferpix` records the pixels loaded), and uses them when
  given; mode (b) then passes. This changes ALIS's loader and writer, for every model.
- **(b) The case joins with modes (a) and the bundle**; mode (b) is left out for it,
  with its reason, as `tophat`'s is (`FIXEDPARAM_EXCLUDE`). Recording the sub-pixels
  waits in `deferred_work.md` §7, beside §7.10 (`nsubmin`), for a stage on ALIS's
  sub-pixels.
- **(c) The example pins its sub-pixels** (`run nsubmin 6`, `run nsubmax 6`), so that
  both runs use one grid. This hides what (a) would fix.

My lean: (b). The dashboard already shows what the fit made; (a) changes how ALIS
works out every model, and its references with it, a series of its own as §7.10 is.

**Response:** I agree with option (b). Please add this to deferred work in §7.10. We should make this the key item of Stage 14, and we will do this next (before the documentation). Once this stage is complete, please draft a stage 14 document that addresses §7.10 in `deferred_work.md`. If there are other items in deferred work that are related to this development and you think should be included in Stage 14, please add them to the draft design document. We will then review and prioritize them all for Stage 14.

*Applied (2026-11-24, after Prompt 4): Task 13.2 as (b). `examples/lsf_spline` is a
case of the harness with its fit and its bundle (both pass); its evaluation at the
best fit is left out (`FIXEDPARAM_EXCLUDE`, with its reason). `deferred_work.md`
§7.10 takes the sub-pixels a `.mod.out` is evaluated on, beside the default
`nsubmin`, as dashboard Stage 14's key item, next; §7.11 is marked done. Task 13.11 (13.10 then)
drafts `dashboard_stage14.md` around §7.10, with the related items of
`deferred_work.md` in its draft design for RJC to prioritise, and records that Stage
14 comes before the documentation.*

**Q13.8 — Warnings and options to resolve… (Task 13.10; RJC's comment on the review,
2026-11-24).** RJC asked for a red button between the Fit statistics and Correlations
panes that opens a dialog listing the fit's warnings, each with a suggestion to
resolve it (blank where there is none), in place of scrolling Fit statistics. Three
choices:
- **(i) When the button shows.** (a) Only while the shown run has a warning; (b)
  always, greyed when there is none.
- **(ii) Fit statistics.** (a) "Stopped because" and "Convergence" stay red there,
  and "What to do" moves into the dialog; (b) Fit statistics is plain again, and only
  the button warns.
- **(iii) What the dialog lists.** (a) The two warnings of Task 13.4 (stopped at
  `chisq maxiter`, a convergence check that failed), with their suggestions, and
  ALIS's own warnings during the run (no suggestion), which are then kept in the
  run's `run.json` (they are held in memory only now, so a run opened later has lost
  them); (b) the two of Task 13.4 alone. Results' table flags (at limit,
  unconstrained) stay in the table either way.

My lean: (i) (a), so that the red always means something; (ii) (a), the facts stay
where the statistics are and the advice goes to the dialog; (iii) (a), every warning
the run gave, in one place.

**Response:** I agree with your leans. Please include this in the stage 13 development.

*Applied (2026-11-24): Task 13.10 as written, (i) (a), (ii) (a), (iii) (a); built the
same day, and shown in the review's draft 3. RJC agreed in the page's thread too.*

## Prompts
1. Please read the list of changes and bug fixes, and the work of Stages 0–12 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Feedback`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (13.1, 13.2, etc.). The final Task (currently labelled 13.N) should be to close the stage, and should include the checks that will verify that the stage is complete.
2. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. I have also added a new feedback (point 4) to be included in the development of this stage. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (13.1, 13.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
3. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (13.1, 13.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
4. Please execute the tasks in order. There's no need to pause between tasks, unless you have queries that require my thoughts and responses. Please check your context limit between tasks. If you reach 75% context, please pause and ask for my guidance on the next steps. If you have queries during development, please pause the development and log the queries and responses in the Queries section.