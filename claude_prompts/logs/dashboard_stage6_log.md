# Dashboard Stage 6 log

The Fit tab (step 6), which completes v1: running a fit in the background with its
progress and Stop, the settings and the pre-flight check, Inspect, Results, Compare,
the convergence check, random restarts and outliers, and runs made outside the
dashboard. The plan is in `claude_prompts/dashboard_stage6.md`.

### 2026-10-18 (Prompt 1: reading, and queries Q6.14–Q6.19)

**Reading.** The code plan; the dashboard prompts (F1–F14, S1–S31, QF.1–QF.36,
D1–D68); the Stage 0–5 documents and logs (Stages 1 and 5 in full, the others for
the Fit tab, runs, sessions and markers); this stage's document with RJC's responses
to Q6.1–Q6.13; and the code the tab builds on: `alis/bundle.py` (`run`, `_run_info`,
`_set_latest`, `unsupported`, `read`), `alis/scripts/run_alis.py`, `alis/config.py`
(`ArgFlag`), `alis/minimise.py` (`signal_handler`, `defiter`, the iteration call),
`alis/main.py` (the end of a fit, the interrupted errors, the simulations),
`alis/report.py`, `alis/convergence.py` (`assess_restarts`), `alis/simulate.py`
(`perturb`), `alis/dashboard/session.py` (`runs_state`, `check_disk`, `take_runs`,
`save`), `markers.py` (the Fit marker), `project.model_changed_since_run`,
`qt/tabs.py` (the Fit frames), and the Fit actions in `actions.py` and
`qt/window.py`.

**Facts that shaped the notes and queries.**
- `bundle.run(path, …)` reads the bundle from a file and stores the run back into the
  same file. Running a project handed over in memory (RJC's Q6.2 response) needs it
  split into the in-memory run and the storing. `read` already opens a file-like
  object (only its error messages assume a path).
- The minimiser calls its `iterfunct` (`defiter` by default, which prints
  "ITERATION n CHI-SQUARED = …" at `out verbose 1`, and the model at 2) after every
  iteration. `ClassMain` passes no `iterfunct`, so nothing can receive the parameter
  values at each iteration (RJC's Q6.3 response) without a small hook.
- `signal_handler` (SIGQUIT) sets a flag that ends the fit at its iteration with
  status −20; `main.py` then sets the errors to the values.
- `simulate.perturb` draws its starts about `slf._modpass['p0']` (the starting
  values), not about `bparams` (passed, unused), clamped to ±`sim maxperturb` (10%) of
  each value, on the same data; it writes each restart's `.mod.out` to `sims/` and one
  table, `<model>.perturb_<a>-<b>` (`np.savetxt`: the starting values, then each
  restart's best fit). Unlike `sim_random`, it does not call
  `convergence.assess_restarts`.
- `report.py` flags a snip when (χ²ν − 1)/√(2/N) exceeds `out reportsig`, so only
  above, or when |runs z| does; its statistics are private (`_runs_z`, `_regions`)
  and need an ALIS instance (Q6.6(b)).
- J1358p6522's Lyα forest b_turb 0.292413 (`lim voigt bturb [0.2,None]`) is above its
  bound, so with RJC's rule for "at limit" (Q6.12) it reads "unconstrained".
- Ctrl+Shift+R is free in the action registry (for Save and Run).

**Baseline.** `pytest -m unit`: 1881 passed, 121 skipped, 0 failed (90 s);
`pytest -m gui`: 146 passed (219 s). The same as at the close of Stage 5.

**The responses.** Recorded as "Notes on the responses" in the document: reviews at
6.5 and the close (Q6.1); no project file written for a run, with Run and Save and
Run (Q6.2, Q6.14); the fit watched while it runs (Q6.3, Q6.15); every setting kept
available, and the command line running the project as written (Q6.4, Q6.16);
Initial and Final as proposed (Q6.5); Q6.6's two changes, and two more (Q6.19); the
run history (Q6.7) and Compare (Q6.9) as proposed; Continue from best fit as
proposed, with per-card and per-continuum copies (Q6.8, Q6.17); the restarts' values
kept and exported (Q6.10, Q6.18); outliers at 5σ, a preference, reviewed before
Apply (Q6.11); "at limit" only on a bound (Q6.12); Stage 7's template at Task 6.11,
and Q6.13(b) and (c) as the lean unless RJC says otherwise.

**New queries,** each with a lean:
- **Q6.14:** the project handed to the fitting process through a pipe, never as a
  file; the result stored in the project's file at once (`runs/latest/` only) or kept
  in memory until Save; Save and Run (Ctrl+Shift+R);
- **Q6.15:** the fit in progress drawn on Inspect after each iteration, with a notice;
  Stop and Cancel as proposed; no Pause;
- **Q6.16:** the dashboard's own run settings on the command line only; All settings…
  offering every setting, the computer's and the "command line only" ones included,
  with the pre-flight check refusing Run while one of the latter is set;
- **Q6.17:** "Use best fit" on each component card and in the Continuum box, copying
  even into lines changed since the run, matched by the redshift label or the snip's
  specid, function and order;
- **Q6.18:** the restarts' values kept with their run and exported in `sim perturb`'s
  own form, hidden parameters left out; where the starts are drawn (the best fit, as
  Q6.10 says, or the starting values, as `simulate.perturb` does);
- **Q6.19:** `bundle.run` split into the in-memory run and the storing, and an
  optional per-iteration callback in `ClassMain`.

### 2026-10-18 (Prompt 2: the responses to Q6.14–Q6.19; queries Q6.20–Q6.22)

**The responses.** Q6.14, Q6.15 and Q6.19 as proposed. Q6.16 as proposed, with
generating data recorded as future work. Q6.17 (a), (c) and (d) as proposed. RJC added
two things to it: for the continuum, a raised order takes the run's coefficients with
zeros above them, and a lowered order drops coefficients that are zero; and the FWHM
can be returned to its default. Q6.18 (a)–(c) as proposed; (d), where the starts are
drawn, deferred.

**Facts checked for the new queries.**
- `main.py` reaches `simulate.perturb` by two routes. After the fit in the same run,
  `_modpass['p0']` still holds the starting values: the minimiser works on a copy
  (`self.params = xall.copy()`) and writes into `xall` only to clamp a start outside
  its limits. With `sim beginfrom <fit>`, `p0` and the covariance are read from that
  fit's `.mod.out` and `.covar`, so its starts are drawn about the best fit. A draw
  about a value of zero is not clamped by `sim maxperturb`.
- A Data tab row's starting FWHM (the "Add a spectrum" dialog's) is kept in
  `ui/project.json` (`rows[].fwhm`; `datasets.add_file_step`, `modes` for a new
  project), and is what a new snip of the row starts with (`snips.py`). An imported
  fit's rows have none (`Project.structure`).
- The dashboard's continua are ALIS's Legendre, Chebyshev, polynomial and constant
  (`continuum.FUNCTIONS`), all polynomials, so a run's continuum can be carried into a
  line of another function, range or `scale=` exactly when the order allows it.

**Written.**
- `deferred_work.md`: §7.5 (generating data, RJC's Q6.16 response) and §7.6
  (simulated realisations and random restarts: the two routes and their centres, the
  clamp, the bundle's refusal; RJC's Q6.18(d) response), with the header of §7 noting
  them.
- `dashboard_stage6.md`: notes on the responses, and three queries, each with a lean:
  - **Q6.20:** the continuum carried over as a curve (exact for a raised order, and
    for a changed range or function); what "zero" means for a dropped coefficient —
    lean: within its error;
  - **Q6.21:** Reset FWHM in each FWHM cell's menu (and ↺), to the FWHM given when the
    file was added, or recorded at import; the value only;
  - **Q6.22:** Random restarts… built in this stage, drawing as `sim perturb` draws
    after `sim beginfrom` the shown run, or left until §7.6 is decided — lean: built.
- The opening summary, the Design section and the Tasks brought in line with every
  response, and with the leans of Q6.20–Q6.22:
  - **running:** Run, and Save and Run; the fitting process (`runner.py`, new)
    handed the project through a pipe; where the result goes; the fit in progress;
    Stop, Cancel, no Pause; closing during a run;
  - **settings:** the dashboard's own run settings on the command line only; All
    settings… with every setting, "command line only" ones refused by the pre-flight
    check;
  - **results:** "at limit" only on a bound (J1358p6522's 0.29 ± 60.53 now
    "unconstrained"); what a committed run keeps; Use best fit; Reset FWHM;
  - **restarts:** what is kept, the window, Export restarts…, and their blinding;
  - **outliers:** 5σ, with accepting and rejecting before Apply;
  - **tasks:** Q6.19's changes to ALIS in Task 6.1, Q6.6(a) in Task 6.2 and Q6.6(b)
    in Task 6.3, so that the numbering (the review at 6.5, the close at 6.11) is
    unchanged; Use best fit and Reset FWHM in Task 6.6; the close (6.11) with
    `prepfit`'s note and the Stage 7 template. The regression harness, with its
    `test_bundle` mode, runs after any task that changes ALIS outside
    `alis/dashboard/`.

### 2026-10-18 (Task 6.1: runs and the fitting process)

RJC confirmed the start of development, up to and including Task 6.2, and answered
Q6.20–Q6.22 meanwhile (below, under Task 6.2).

**ALIS (Q6.19).**
- `alis/bundle.py`: `run` is split into `execute(bundle, argflag, overrides, command,
  iterhook, started)`, which runs a bundle held in memory and returns a `RunResult`
  (`run.json`'s fields, the outputs and which of them are hidden, the model as run,
  the snips it fitted, the error or the exit), and `store(target, result)`, which
  stores it in a bundle file (under its lock, as before) or in a `Bundle` in memory.
  `run` does both, as before, and re-raises how ALIS stopped. `read` also takes a
  bundle's bytes ("the project handed over" in its messages), and `to_bytes` gives
  them, never written to a file; `_write_unlocked` shares the zip writer.
- **The snips each run fitted** (Q6.7(a)): `_set_latest` keeps them under
  `runs/files/<sha256>`, once each, and lists them in `run.json` (`inputs`, with
  `model`, the model's member). `prune_run_files` drops those no run has any longer,
  by `run_inputs` (every `runs/*/run.json`). A packed plain run's outputs record the
  packed snips. This holds for `run_alis project.model` too, so that a command-line
  run can be committed with its snips as they were.
- `alis/main.py`: `ClassMain(..., iterhook=None)` passes it to the three fits it makes
  (the fit, the iterated fit, the convergence check). `alis/minimise.py`: `alfit(...,
  iterhook=None)` wraps `iterfunct` (`_hooked`): after ALIS's own printout, the hook
  is called with the iteration, χ², a copy of the parameters, the degrees of freedom
  and whether this is the convergence check. **A true return value stops the fit as
  `SIGQUIT` does** (`self.handler = False`, status −20): this is how the dashboard's
  Stop reaches the fit, through the pipe rather than a signal (an addition to Q6.19(b),
  which said the hook is called; it is also what makes Stop work where `SIGQUIT`
  does not).

**`alis/dashboard/runner.py`** (new; `python -m alis.dashboard.runner`): reads one
JSON line (the project's file or none, the settings, the command, the size, the
protocol's version) and the project's bytes from its standard input; sends JSON lines
(`started`, `iteration` with χ², dof and the values, `warning`, `storing`, `runs`,
`finished`) on a duplicate of its standard output, after pointing file descriptors 1
and 2 at `/dev/null`, so that ALIS's printout goes nowhere and a dashboard that has
closed cannot break the fit with a broken pipe; reads `stop` on its standard input.
It stores the result in the project's file; with no file, or a file it cannot write,
it sends the `runs/` members back. `run.json` gains `origin: dashboard`, `progress`
(χ² of each iteration, ending with the best fit's) and the run's `ui/project.json`
among its inputs. ALIS's verbosity is given as `run_alis -v` gives it, because
`--set 'out verbose N'` is ignored by `set_params` (found by the settings test).

**`alis/dashboard/runs.py`** (new):
- `Process` (the pipe, a reader thread and a queue, Stop, Cancel unless the process
  is storing, `detach` for Keep it running; `start_new_session=True`, so that it
  survives the dashboard and the terminal's signals, and so that Cancel ends its
  whole process group, ALIS's pool of workers included: killing the process alone
  left four workers running) and `Run` (hands the session's
  bundle over without its `runs/`, follows the messages, `wait`, `take` into the
  session).
- `RunRecord` for the latest and committed runs: `run.json`'s fields, `origin`
  (`dashboard`, `cmd`, `imported`), outputs with hidden lines restored in memory,
  `values` (the `.mod.out` read by `ParsedModel`: the same parameter numbers as the
  input model), `errors` (the `.mod.out`'s error block matched word for word to the
  model block's lines, and the FWHMs and shifts from the covariance or the
  `.mod.out`'s lists; none for an interrupted run), `covariance`, `inputs`, `ui`.
  Reading the errors with ALIS's loaders was rejected: they clamp an error below a
  parameter's lower limit (b_turb's 0.5 km/s).
- `commit`, `describe`, `remove` (numbers from `runs/index.json`, never reused; the
  latest run records `committed_as`), `committed_state`, `take_latest`,
  `take_committed`, `apply_runs`, and `restore_step` (the model, the snips and the
  run's `ui/project.json` as one step).

**`alis/dashboard/session.py`:** `take_runs(force=False)` takes the latest run and
last error only, keeping the session's committed runs; `force` takes them even when
the model on disk also changed. `runs_changed()` counts a change to the run history,
which makes the session dirty and the recovery copy be written again; Save writes the
committed runs and the latest run, unless a newer one was stored meanwhile; a
recovery copy brings its committed runs back; unblinding merges the session's runs
before unblinding them.

**Checks.** `test_dashboard_runs.py` (13, `unit`; runs made from reference outputs):
bytes read back; a stored run's snips kept once, the old ones dropped; an error leaves
the latest run; metal_line_abs's values, errors and covariance, and J1358p6522's 100+
errors agreeing with its covariance; no errors for an interrupted run; commit,
describe, remove, numbers never reused; the history saved with Save, a newer run kept,
taken in, recovered; a run kept in memory for a project with no file, in the recovery
copy and in Save as; restoring a run one step, undone by one undo.
`test_dashboard_runs_process.py` (8, `fast`, 38 s): metal_line_abs through the
process gives `run_alis`'s best-fit block, χ² and iterations, the χ² of every
iteration, and no file outside the project; an unsaved edit is fitted and the model
on disk unchanged; Stop keeps an interrupted run with its values and no errors;
Cancel keeps nothing; no file: memory, recovery copy, Save as; an error recorded and
masked; Cancel with two CPUs leaves no process of its group; examples/blind with and
without global blind: no hidden value in the
messages, the progress, `run.json` or the plain members of the file. The `unit` batch:
1894 passed, 121 skipped. The regression harness's bundle mode (`-k test_bundle`): 27
passed.

### 2026-10-18 (Task 6.2: settings and the pre-flight check)

**ALIS (Q6.6(a)).** `alis/config.py`: every setting's field is `_s(default, help)`
(a `field` with a one-line `help` in its metadata), 83 in all; `settings_help()`
gives them by section. Four are described as "Not used" (`run last_update`, `run
logn`, `out reletter`, `iterate data`). `run_alis --list-settings` prints each under
its default. No default or type changed (checked).

**`alis/dashboard/settings.py`** (new): `catalogue()` (every setting with its kind:
tick box, whole number, number, word, word from a list, anything; its choices, its
description and its owner: the model, the computer, the dashboard, the command line
only); `LEFT_OUT` (`run modname`, which ALIS sets, and the four unused ones);
`check_value` (ALIS's own `set_params`, and the lists); `model_settings`, `changed`
(the model's settings that ALIS reads as other than the default); `set_step` and
`clear_step` (`edit.set_setting`; every line of the setting removed); `run_settings`
(`out verbose 1`, `plot fits False`, `out report True`, `out covar <model>.covar`
unless the model names one, and the computer's settings unless the model sets them);
`computer_used`, `command_line_only`.

**Preferences:** `fit.backend` (a word from a list: a new `str` kind of `Spec`, with
`choices`, shown as a menu in the Preferences dialog), `fit.ncpus`, `fit.ngpus`,
`fit.low_b` (1 km/s).

**`alis/dashboard/preflight.py`** (new): `check(project, prefs, problems)` gives a
`Report` of `Finding`s (refuse or note, the message through the gate, the line or
snip, a code) and its summary ("Pre-flight check: no problems; 2 notes"). Refused: a
model that does not read, ALIS's errors (`validate.check`), a "command line only"
setting or a data line that reads a file (read from the text, so given even while the
model does not read), a free starting value outside its limits (unless `run limpar`),
`run ncpus 0` from the model or the preference (ALIS's pool cannot have no worker),
no free parameter, more free parameters than fitted pixels. Noted: a free parameter
that no snip with fitted pixels reaches (ALIS's influence table, through two new
public methods of `Profile`: `fitted_pixels` and `influence`), a starting value on a
limit, a b_turb below `fit.low_b`, a snip with no continuum (neither a continuum line
nor a continuum column), pixels fitted twice (`markers.fitted_twice`, now shared with
the Regions marker), the imported fit's notices, an ion in no snip, regions changed
since the components were last edited. A note that would tell something of a hidden
value (on its limit, below 1 km/s) is not made.

**Checks.** `test_dashboard_settings.py` (8) and `test_dashboard_preflight.py` (16),
`unit`: every setting offered or left out with a reason, with its description;
owners, kinds and the backend's choices equal to `alis.gpu.BACKENDS`; set and cleared
byte for byte; a change undone byte for byte; ALIS's refusals; the settings changed; what a
run is given, and the model's computer setting winning; command-line-only settings
written; each refusal and note on metal_line_abs made to have it; J1358p6522 as
imported: its two notices as notes, no refusal; no hidden value in a message.
`test_config.py` (2) and `test_cli.py`: every description, the defaults unchanged,
`--list-settings` printing them. The `fast` test that a run is given the dashboard's
settings (in `run.json` only, the model text unchanged, the covariance it asked for).

**RJC's responses to Q6.20–Q6.22** (written during Tasks 6.1–6.2): Q6.20, the
continuum's current function and order are fitted to the run's best-fit continuum,
whatever the order, and the user sees how far the copy departs from it; Q6.21 and
Q6.22 as proposed. The Design ("Use best fit") and Task 6.3's check follow them.

**Batches after Tasks 6.1–6.2.** `pytest -m unit`: 1920 passed, 121 skipped (78 s);
`pytest -m gui`: 146 passed (208 s); `pytest -m fast`: 120 passed (14 min), with the
regression harness and its bundle mode, and the new process tests. Cancel's change to
end the process group was made while the `fast` batch ran, and its test was then run
on its own (passed).

### 2026-10-18 (Prompt 3: Task 6.3, results)

**Reading.** A new conversation: this stage's document (with RJC's responses to
Q6.1–Q6.22) and log, the mockups of the Fit tab (`fit_inspect.html`,
`fit_results.html`), Q0.4 of `dashboard_stage0.md` and "What Stage 6 receives" of
`dashboard_stage5.md`, and the code the task builds on: `alis/report.py`,
`model_eval.model_func` (what `_modfinal` and `_contfinal` hold), `save.py` (the
`_fit.dat` columns), the functions' `parout` (the `blindrange` offsets),
`dashboard/profile.py`, `runs.py`, `model.py`, `project.py` (steps, notices,
`mark_hidden`), `components.py`, `continuum.py`, `snips.py` (fit masks), `edit.py`,
`blinding.py` and `preferences.py`. No new query blocks the task; one is raised
(Q6.23, below).

**Facts that shaped the code.**
- **The run's model, not the current one.** A run's values are numbered as the model
  it fitted, which may since have gained or lost lines, so the results read the
  run's own model as a project of its own, on the snips as they are now (Q6.5).
  Drawn this way, J1358p6522's best fit agrees with its `_fit.dat` to 0.0007σ, and
  its snips' χ²ν and runs z equal the `.report`'s.
- **ALIS offsets a `blindrange` line's best fit.** `parout` adds a seeded uniform
  offset to every value it writes of such a line: `examples/blind`'s O I is written
  as log N −616947.11. Drawn or continued from as written, the model would be
  nonsense, and shown without the offset, it would reveal what ALIS hides.
- **`_fit.dat`'s continuum column** is ALIS's best-fit continuum line (`_contfinal`)
  over the fit window, which checks the best-fit continuum.

**ALIS (Q6.6(b)).** `alis/report.py`: `runs_z` (was `_runs_z`), `chi2_deviation` and
`region_stats` are public functions of a region's residuals; `build_report` uses
them and writes the same report (`test_fit_report.py`: the old tests on the public
name, and one that `region_stats` gives the report's numbers).

**Smaller changes in the dashboard.**
- `continuum.py`: `evaluate`, `coefficients` and `model_wave` take `values` (by
  ALIS's number): a continuum, and its shift, at a run's best fit; `scales` gives a
  line's `scale=` factors.
- `runs.py`: `RunRecord.values` takes ALIS's `blindrange` offsets away
  (`blind_offset`, `blind_offsets`; a place written without one is preferred);
  `RunRecord.texts` gives the `.mod.out`'s words (their precision); `Run.bundle`
  keeps the project as handed to the process, the model of the fit in progress.
- `blinding.py`: `Gate.best_fit` also masks a parameter of a `blindrange` line
  (`offset_index`, `best_fit_hidden`); its starting values are shown as before.
- `preferences.py`: `fit.outlier_sigma` (5σ, Q6.11), `fit.loose_logn` (1 dex),
  `fit.loose_kms` (10 km/s) and `fit.loose_fraction` (1× the value, Q6.12).

**`alis/dashboard/results.py`** (new):
- `View(project, record=…)` or `View(project, run=…)` (the fit in progress): the
  run's model (`model_of`, cached by its content and the snips' files), its values,
  errors (none in progress or interrupted), covariance and words; `match_lines`
  (lines matched by what each is, whatever its values, `difflib` on their shapes)
  and `RunGate` (a value masked when hidden in the run's model or the current one,
  by line or label, under either's global blind, or offset by ALIS).
- `curves()`: each project snip's best fit (`Profile.evaluate` of the run's model,
  links applied), with its best-fit continuum; None for a snip the run did not fit
  (`unfitted`). `marks()` and `quality(curve, nsig)`: `report.region_stats` on the
  fitted pixels; ✓, ! beyond 2σ, ✕ beyond `out reportsig` (χ²ν only above).
- `table(prefs)`: `Group`s (each system, the primary first, its ions and
  components; the Lyα forest, collapsed; the continua; the instrument; others) of
  `Row`s, one per ALIS parameter where it first appears (fixed defaults left out):
  value and error through the gate, state (`●■⛓ƒ`) and its label or link, the
  velocity of a z in its system, the current model's lines (S16), and flags:
  "blinded", "at limit (b)" (`at_limit`: on a bound to half a unit of the
  `.mod.out`'s last digit, Q6.12), "unconstrained" (`unconstrained`; for a hidden
  value, only the tests that do not involve it), "not tied to H I" and "z not tied
  to a1" (D20, D22). `copy_text` (Copy as table), `correlation_groups` and
  `correlations(key)` (D51; hidden parameters included, by name), `statistics()`
  (from `run.json`, or the `.mod.out`'s header for an imported fit).
- `continue_plan(project, view)` (S20, Q6.8): a `Plan` (the step, the lines changed
  and what each receives, the lines skipped and why). Per parameter, as RJC's Q6.8
  response puts it, rather than per line as the Design first said (now updated): a
  parameter receives its best fit when every place of it is on a matched line with
  the same value written; fixed and linked ones are left. Values are written as the
  `.mod.out` writes them.
- `component_step` and `component_reason` (Q6.17): a component matched by its
  redshift label (or, with none, by its line), its z, b_turb, T and each ion's
  log N copied even into changed lines; an ion the run did not have keeps its values.
- `continuum_copy` and `continuum_reason` (Q6.17, Q6.20): the line's function, order,
  range and `scale=` fitted to the run's best-fit continuum on the good pixels of the
  snips that share it, weighted by the errors (`continuum._solve`), with terms below
  10⁻¹² of the curve written as zero; how far it departs over the fitted pixels
  (largest and rms, in σ); the run's continuum and the copy, for the dashed curve.
- `outliers(curves, project, threshold, prefs)` and `outliers_step(project, chosen)`
  (Q6.11): fitted pixels beyond 5σ, lone ones and pairs ticked, three or more in a
  row not; the chosen ones left out of the fit mask (`snips.set_mask_step`), all
  snips as one step ("Leave out 3 outliers").
- Blinding: steps' descriptions and plans name no value; under global blind every
  line that receives a value is hidden, and so is a `blindrange` line.

**Checks.** `test_dashboard_results.py` (27, `unit`, 3 s): lines matched whatever
their values; the best fit drawn equals ALIS's `_fit.dat` on metal_line_abs and on
J1358p6522's twelve snips (< 0.01σ; the continuum too, for J1358p6522); the fit in
progress drawn from its values, with no errors; a snip the run did not fit; the
marks equal the `.report`'s (nine snips ✕ by the runs test alone, the mockup's
amber marks); J1358p6522's table as in the mockup, every parameter once; the Lyα
forest's b_turb rows 1.09 ± 10.78 and 0.29 ± 60.5 "unconstrained", nothing "at
limit"; a value on its bound "at limit (6)"; the thresholds as preferences;
correlations (symmetric, unit diagonal) and statistics; Copy as table; Continue from
best fit one step that ALIS reads and one undo restores, skipping a parameter changed
since the run and a line added since, from an interrupted run, and under global
blind hiding what it fills; Use best fit on a component into changed lines, and its
reasons; Use best fit on a continuum with its order unchanged, raised (zeros above),
its function or range changed (exact, 10⁻⁸), and lowered (the nearest line, its
departure); outliers at 5σ and Apply as one step that undo reverses; offsets taken
away before drawing; no hidden or offset value in the table, its copy, the
statistics, a plan or the `.mod` panel after Continue. `test_dashboard_results_fit.py`
(1, `fast`, 6 s): metal_line_abs made to stop on a bound (`lim voigt bturb
[6.0,None]`) and fitted through the dashboard's process: b_turb 6.00 "at limit
(6)", the best fit drawn equals the run's own `_fit.dat`, the marks its `.report`,
and Continue from best fit writes the bound. Also `test_dashboard_blinding.py` (the
gate masks an offset best fit, not its start) and `test_dashboard_runs.py` (the
offset taken away equals ALIS's, from `examples/blind`'s reference).

**Query raised:** Q6.23, "unconstrained" on continuum coefficients (15 of
J1358p6522's 54 under the agreed rule) — lean: not flagged, only "at limit". The
rule stays as agreed until RJC answers; Task 6.6 applies the answer.

**Batches after Task 6.3.** `pytest -m unit`: 1950 passed, 121 skipped (89 s);
`pytest -m gui`: 146 passed (218 s). Since `report.py` changed, the regression
harness's `fast` cases with its bundle mode (27 `test_bundle` cases), with
`test_fit_report.py`, the process tests and `test_dashboard_results_fit.py`: 106
passed (14 min).

### 2026-10-19 (Prompt 4: Task 6.4, Inspect)

**Reading.** A new conversation. Prompt 4 is the last line of the Prompts section,
which has no newline after it. Read for it: the code plan, the dashboard prompts
(D1–D68), this stage's document with RJC's responses to Q6.1–Q6.22, this log, the
Inspect mockup (`page_fit_inspect` in `build_mockups.py`), the `gui-component` skill,
and the code the sub-tab builds on: `qt/tabs.py`, `qt/window.py`, `qt/plots.py`,
`qt/components.py`, `qt/widgets.py`, `qt/dialogs.py`, `runs.py`, `results.py`,
`settings.py`, `preflight.py`, `markers.py`, `session.py`, `components.py` (the
panels and ticks), `actions.py` and `preferences.py`. Q6.23 has no response yet. Its
answer is applied in Task 6.6, so it does not block this task, and no new query
arose.

**Facts that shaped the code.**
- **"All snips" cannot link the panels' x axes**, as the Components tab does: each
  snip covers its own velocity range (J1358p6522's Ly8 snip spans −3000 to +300
  km/s, O I 1302 ±300). One ion's panels, and the Lyα forest's, do share an axis.
- **A plain run's `.mod.out` writes "BLIND MODEL"** in place of a `blind=True`
  line. A bundle's run writes the real values and keeps them hidden (`save.py`,
  Stage 1.6). Final therefore needs the bundle's form, as the Task 6.3 tests store
  it (`_blind_out`). `examples/blind`'s plain reference gives no best fit.
- **ALIS's defaults are `run blind True` and `out verbose 2`**, so a model's `run
  blind False` is listed among "Settings you changed". That is correct, and the
  line is shown read-only there (only Model → Unblind… changes blinding).
- **The pre-flight check costs 6–250 ms** (J1358p6522, Q1243p307: the first call
  builds the profile). It is therefore made when the Fit tab is shown and at Run,
  never on every refresh. The state key it is kept under costs 1 ms on Q1243p307
  (3.5 MB of snips).

**`alis/dashboard/fitview.py`** (new; no Qt): what Inspect shows.
- `choices(project)`: the View menu (D38). "All snips (12)"; under "By ion", each
  ion of each system that a snip holds, as the navigator groups them ("H I with D I
  · 8 snips", prefixed by the system when there are several), then "Lyα forest · 10
  snips"; under "One snip", each snip as "Ly8 923.2 · H1" (with its dataset when
  several snips share a specid; its wavelength range when its transition is not
  known). `find` and `step` (‹ › go round the menu).
- `panels(project, choice, axis)`: a `Panel` per snip. In velocity, the axis is in
  the model's frame, about the transition the snip was cut for (`lines.
  snip_transitions`, with the ion's other transitions inside the snip for the
  echoes). An ion's panels use the Components tab's panels (the ion's strongest
  transition), and the forest's use `components.forest_panels`. In wavelength, the
  pixels as they are. Each panel gives `x`, `regions`, `home` and `half_range`.
- `ticks(project, panel, view=None)`: the Components tab's ticks, locked. Initial
  ticks are at the starting redshifts. Final ticks are at the run's best fit:
  `components.panel_ticks`, `forest_ticks` and `component_z` take a `values`
  argument (every parameter by ALIS's number).
- `chips(project, view)`: each snip's `Chip` with the Final marks
  (`results.quality`), or why it has none ("no run yet", "not fitted by this run",
  why the run has no best fit). `mark_texts` ("χ²ν 0.86", "runs −9.6", signed as
  the axes are, D43) and `worst`.
- `progress(project, job)`: χ² against iteration. While a run goes, the run is
  drawn against the latest run (grey). Otherwise, the latest run is drawn against
  the newest committed run before it. A command-line run gives only its end (S31),
  and an imported fit none, and the note says so. `status_text` ("Running ·
  iteration 9 · χ² 3796.2 · 0:41", "Stopping the fit at the end of iteration 9…",
  "Storing the result…", "Convergence check · …"), `marker_text` and `notice_text`.
  The in-progress run carries no number: runs are numbered when committed (Q6.7).

**`alis/dashboard/qt/fit.py`** (new):
- `FitRun` (the window's): starts a `runs.Run`, polls it every 100 ms
  (`progressed`), and handles Stop, then Cancel when pressed again. At the end it
  takes the run into the project (`ended`). `wait` (closing with Stop, the window
  still drawn) and `detach` (Keep it running).
- `InspectTab` (D38, D39), in the mockup's columns. The left column is at least
  250 px wide, so that the Run box reads beside the `.mod` panel.
  - **The Run box:** Backend, CPUs and GPUs are this computer's preferences, saved
    when changed and not steps. A setting the model writes is greyed, with "The
    model's own run ncpus is used." The settings changed are each edited in place
    or cleared with ✕, one step each (`settings.set_step`, `clear_step`).
    All settings… The pre-flight summary is coloured with its icon (✓, !, ✕), and
    its findings are a click away. A finding with a line puts the `.mod` panel's
    cursor there; one with a snip shows the snip. Run, Save and Run, and Stop are
    the window's actions.
  - **Progress:** the chart (`ProgressChart`), Linear | Log (the new preference
    `fit.progress_axis`), and a note naming the run drawn and the grey one.
  - **View:** the View menu with headings; ‹ › (keys [ and ], the tab's own
    actions `fit.previous_view` and `fit.next_view`, scope "Fit"); Columns (the
    Components tab's setting in the view state, greyed for one snip); Velocity |
    Wavelength; Initial | Final; the zoom bar. Final is greyed, with the reason
    as its tooltip, while there is no best fit. Below them: the fit in progress's
    notice, the shown model ("Final · the latest run"), and the panels.
  - **Each panel** (`InspectPanel`): its title and marks; the data over the shown
    model's own continuum (the best-fit continuum for Final); the model; the fit
    regions; the residuals; and the ticks, labelled unless more than six crowd
    them. One snip fills the view, and several keep 210 px each and scroll. An
    ion's or the forest's panels share x and start at the Components tab's range
    (±200 km/s). Elsewhere each panel keeps its own range: `ZoomBar.attach(...,
    link=False)` keeps every view's range for ← → and zooms each about its own
    middle.
  - **All snips:** a `Chip` per snip in a wrapping `widgets.FlowLayout`, each mark
    with its tint and icon (`style.MARK_COLOURS`); a click shows the snip, and the
    shown chip is outlined.
  - **The fit in progress** (Q6.15): Final draws `results.View(project, run=job)`
    under its notice. Once its first iteration's values arrive, it is redrawn in
    place, keeping the zoom, no sooner than 300 ms (or twice the last drawing's
    time) after the last drawing. The chips follow it.
  - **The view state** `fit` (`view`, `axis`, `model`) is saved with the project.
- `qt/tabs.py`: `FitTab` hands its window, refreshes, keys and buttons to Inspect.
  `InspectTab` moved from the frames to `qt/fit.py`.

**The window** (`qt/window.py`):
- Run (Ctrl+R) and Save and Run (Ctrl+Shift+R): the pre-flight check runs first,
  with ALIS's own check (`preflight.check`, no cached problems). A refusal is told,
  with each problem, and the findings are shown. Otherwise the run starts with
  `settings.run_settings`.
- Stop, which becomes Cancel once pressed, is in the Fit menu, on the Run box and
  in the status bar beside the run's progress.
- When the run ends, the status bar says how ("The fit was stopped: where it got to
  is kept…", "…cancelled: nothing was kept", with ALIS's first warning through the
  gate). An error is told through the gate, and its line is shown in the `.mod`
  panel (`validate.locate`, the public form of `_locate`).
- The disk watcher leaves the dashboard's own run alone: it is taken in when it
  ends, and not reported as a run "outside the dashboard".
- Closing the project during a run asks: Stop the run (waits for it to store),
  Cancel the run, Keep it running (only when the run has a file to store into), or
  Don't close.
- `refresh_markers` is now called after the tabs refresh.
- `markers.fit_marker(project, running, refused)`: ↻ "Running: iteration 9.";
  ! "The pre-flight check refuses Run: …", taken from Inspect's report when that
  report is of the model as it is now.
- `actions.py`: `fit.run`, `fit.save_run` and `fit.stop` are built (Commit run…
  comes with Results); `FIT_TAB` holds the tab's keys.
- `dialogs.AllSettingsDialog` (F9, Q6.16(b)): every setting by section. Each
  shows its value in the model (a tick box, a menu of ALIS's words, or a line
  whose placeholder is the default), ↺ (back to the default, its line removed),
  the default, and what it does. The notes name this computer's settings (what the
  runs use), the dashboard's (what its runs use; `run_alis` uses the model's) and
  the command-line-only ones (Run refused, and how to run them). `run blind` is
  read-only. A filter, and `changes()`. OK checks each value as ALIS reads it, and
  the tab makes every change one step ("Change 3 settings").

**For the review (Task 6.5).**
- Each panel's corner shows the marks of the model it draws, Initial's too. The
  chips always show Final's, as the design says. Showing a starting model's
  χ²ν seemed more useful than Final's marks beside an Initial curve, and is
  easily changed.
- Final's ticks are drawn at the best-fit redshifts. This includes hidden
  components and global blind, as the Components tab draws hidden starting
  values: they carry no value, and the profiles are drawn anyway (D24).
- Not placed yet: Continue from best fit (Task 6.6), and Convergence check…,
  Random restarts… and Review clipped pixels… (Task 6.8).

**Checks.**
- `test_dashboard_fitview.py` (14, `unit`):
  - J1358p6522's View menu as in the mockup, and ‹ › round it.
  - Each snip about its transition, in the model's frame, or in wavelength. An ion,
    the forest and one snip. A snip with no transition is drawn in wavelength.
  - Initial ticks at the start, Final ticks at the best fit, in wavelength too.
  - J1358p6522's chips equal its marks, nine ✕ by the runs test alone. Chips with
    no run, and a snip the run did not fit.
  - The progress against the latest and committed runs, and a command-line run.
  - The status bar's words, the marker running or refused, and an error's line.
  - Blinded lines: no hidden value in a title, label, mark or note.
- `test_dashboard_qt_fit.py` (17, `gui`; 7 also `fast`, they fit):
  - J1358p6522's chips and panels as in the mockup, the View menu, and no design
    reference.
  - A chip shows its snip; ‹ › and their keys; Columns greyed for one snip and
    shared with the Components tab; ±200 km/s for an ion.
  - Velocity | Wavelength. Initial and Final draw `Profile`'s starting model and
    the run's best fit.
  - Final greyed with no run; Linear | Log.
  - The Run box: preferences, not steps; a setting edited and cleared as steps
    that undo; the model's own `run ncpus`.
  - All settings…: filter, three changes as one step that one undo reverses.
  - `sim perturb` refuses Run (told, with Export plain files), the marker !, and a
    finding's click.
  - Blinded lines and global blind: no hidden best-fit or starting value shown.
  - A run of metal_line_abs from the tab, with an unsaved edit, going ahead with
    its notes. The status bar, Stop shown, the marker ↻. The fit in progress drawn
    under its notice. Then its best fit, the marker ✓, the progress drawn from
    `run.json`, the file holding the run, the model on disk unchanged, and the
    edit still unsaved.
  - Save and Run saves first; Stop then Cancel keeps nothing.
  - Stop keeps an interrupted run with no errors.
  - Closing during a run: Stop the run, Keep it running (stored, and shown when
    reopened), or Don't close.
  - An ALIS error: told, marker !, `last_error.json`.
- `test_dashboard_actions.py`: the Fit menu as built.

**Batches after Task 6.4.** `pytest -m unit`: 1964 passed, 121 skipped (89 s);
`pytest -m gui`: 163 passed (279 s); the dashboard's `fast` tests (the process,
results-fit and new window tests that fit): 20 passed (86 s). Nothing outside
`alis/dashboard/` changed, so the regression harness was not run.

### 2026-10-19 (Prompt 5: Task 6.5, the review of Inspect and running)

**Reading.** This stage's document. RJC answered Q6.23: continuum coefficients are
flagged neither "unconstrained" nor "at limit", and a continuum is flagged when its
best fit is negative anywhere. Notes on that response are in the document. The
Design ("Results") and Task 6.6's check now follow it, and it is applied in 6.6.

**The screenshots** (`doc/dashboard/stage6/take_screenshots.py OUTDIR`). Every run on
the page is a real fit, made from the window, off-screen at 1440×900:
- J1358p6522, imported, its D I blinded, saved, and fitted once from the dashboard:
  χ² 3790.5707, as its reference. Then O I's starting log N was raised 0.3 dex and
  the project run again; Ly8 923.2 was caught at the eighth iteration of that run,
  beside the `fit_inspect` mockup.
- After the first run: all snips; H I with D I beside the `.mod` panel; O I 1302 in
  wavelength with the pre-flight check's seven notes.
- metal_line_abs fitted from the window (χ² 357.8742, as its reference): Initial
  and Final.
- Run refused by `sim perturb 10`, and its message; All settings… (filtered to
  chisq); the question when the project closes during a run.

**The page** (`doc/dashboard/stage6/build_review.py SHOTS OUT.html`, in the series'
style, from the mockups' stylesheet): published as draft 1 at
https://claude.ai/artifact/49jLw53SHEZY7Wevj6HqWz. It asks four questions:
1. Each panel's corner shows the marks of the model it draws. Keep, or Final's?
2. Final's ticks are at the best-fit redshifts, hidden components and global
   blind included. Keep?
3. Each snip of "All snips" is on its own axis; one ion's panels share one at
   ±200 km/s. Keep?
4. "Settings you changed" shows `run blind False` and `chisq miniter 10`, since
   ALIS's defaults are True and 0. Keep?

**Found and fixed while taking them.**
- **The CPUs preference broke runs.** metal_line_abs, which does not set `run
  ncpus`, stopped with "Number of processes must be at least 1". The dashboard
  gave its runs the preference as written (`run ncpus -1`, all bar one), and ALIS
  counts −1 so only when it reads a model, not on its command line: `run_alis
  fit_spectra.mod --set "run ncpus -1"` fails the same way. The window tests had
  set one CPU, and the process tests added `run ncpus 1` after the dashboard's
  settings, so neither saw it.
  - `settings.cpus(value)` counts CPUs with ALIS's own `load.cpucheck`, and
    `run_settings` gives the number itself.
  - The pre-flight check now refuses any count that leaves no CPU (−20 on a
    14-CPU computer), not only 0.
  - ALIS's side is Q6.24 (lean: count again in `reapply_cli_overrides`).
  - Tests: `test_dashboard_settings.py` and `test_dashboard_preflight.py`, and a
    `fast` test that fits with the default preference
    (`test_dashboard_runs_process.py`).
- **The status bar.** The message "The fit runs in the background…" hid the run's
  own progress there (a Qt status message hides the bar's widgets), and so did the
  last run's end message during the next run. Run now clears the bar's message and
  writes none. At the end, ALIS's first warning is added only for a run that was
  kept, on one line.
- **The Run box.**
  - Its contents scroll above Run, Save and Run and Stop, which stay put; the
    findings overlapped the buttons before.
  - The progress pane is at most 290 px high, and the left column at least 270 px.
  - The findings list wraps its lines (its vertical scroll bar is always there:
    wrapped items overlapped when it appeared).
- **The fit in progress.** "The fit has not finished its first iteration yet" no
  longer stays above the panels once the fit in progress is drawn.

**Batches after Task 6.5.** `pytest -m unit`: 1964 passed, 121 skipped (95 s);
`pytest -m gui`: 163 passed (284 s); the dashboard's `fast` tests: 21 passed (81 s),
the new default-CPUs fit included. Nothing outside `alis/dashboard/` changed.

**RJC's comment on the page** ("Does the runs value account for masked pixels? It
should be based only on the pixels inside the green fitted regions."). Checked:
- **Yes.** The marks count only the pixels inside the fit regions that ALIS fits
  (`_wavefit`, with a positive error), as ALIS's `.report` does. Pixels left out of
  the fit and bad pixels are not counted. J1358p6522: Ly8's 1091 pixels are all
  inside its 15 regions, and none outside; the same for the others.
- **The gaps.** Within a snip the regions' pixels are one sequence, so a run
  continues across the gap between two regions. Broken at the gaps, the runs z of
  snips cut into many regions is less extreme (Lyβ −6.05 → −4.74, Lyα −5.22 →
  −4.04), though no mark of J1358p6522 changes. Changing it means changing ALIS's
  report as well: **Q6.25** (lean: break them at the gaps, in both).
- The page now says which pixels the marks count (version 2), and the thread was
  answered and resolved.

**RJC's second comment** (draft 1: "a way to indicate that these pixels are
masked… a vertical light grey band in the residuals panel… or a way to turn off
the residuals for the non-fitted pixels"). Both are done, as the page's reply
said:
- `results.counted(curve)`: the pixels the marks count (`residuals` uses it).
- `fitview.unfitted_spans(x, counted)`: the stretches of a panel's axis whose
  pixels are not counted, each from halfway to its neighbours.
- `ResidualView.set_unfitted`: light grey bands there (`COLOURS["unfitted"]`).
- Inspect's **Residuals All | Fitted**: Fitted hides the residuals of the pixels
  not counted and keeps the bands; All (the default) draws every residual. The
  choice is kept in the view state (`residuals`).
- Tests: the spans hold exactly the pixels not counted (Ly8: 1091 counted), and the
  bands and the toggle on the window.
- The page is now draft 2 (a screen of Ly8 with Fitted), republished at the same
  URL, and the thread resolved.

**RJC's answer to the page's question 2** (the ticks of hidden components): keep
them, with no value shown. A tick shows only its component's name and a hover
highlight (no tooltip); `test_no_hidden_value_on_the_sub_tab` already searches the
ticks' labels, the marks and every text of the window, under hidden lines and
global blind. No code changed. The Design's Blinding section records it, and the
page's question shows the answer.

**RJC's answer to the page's question 3**: keep each snip of "All snips" on its own
axis, and one ion's panels (and the Lyα forest's) on one shared axis from ±200
km/s. No code changed; the Design's Inspect section and the page say so.

**Batches after the review's changes.** `pytest -m unit`: 1965 passed, 121 skipped
(89 s); `pytest -m gui`: 164 passed (281 s); the dashboard's `fast` tests: 21 passed
(81 s).

**RJC's answer to the page's question 4**: make ALIS's own defaults `run blind
False` (users should choose to blind) and `chisq miniter 10`. That changes ALIS
outside the dashboard (`alis/config.py`), and the change was not allowed from a
comment on the page, so it was not made: `config.py` is as it was. It is
**Q6.26** in the document, for RJC to confirm, with what it would change:
- `run blind`: no model the harness fits relies on the default.
- `chisq miniter`: helium34's Her36 and tet02OriA stopped before their 10th
  iteration (xtol) and would run on; their `test_minimisation` cases would check
  them.
- Also `test_config.py`, the CHANGELOG, `ALIS_workflow.md` and the dashboard's
  tests that assume the old defaults.

The page's thread has a correction to its automatic reply, and stays open until
then.

**RJC's answer to the page's question 1**: the marks follow the model the panels
draw, in the panels and the chips alike. Done:
- `fitview.chips(project, view, initial=False)`: with `initial`, each snip's marks
  are those of the model as it stands (`Profile.evaluate`, `results.quality`);
  Inspect passes which model is drawn.
- Tests: Initial's chips equal the starting model's marks, also with no run; on the
  window, the chips and the panel follow Initial | Final.
- The Design's "Fit-quality marks" now says so, with the residual strip's bands and
  Residuals All | Fitted.
- The page's question 1 shows the answer, its Initial screen was retaken, it was
  republished, and the thread resolved.

**Batches after the review's answers.** `pytest -m unit`: 1966 passed, 121 skipped
(86 s); `pytest -m gui`: 164 passed (291 s); the dashboard's `fast` tests: 21 passed
(84 s).

## Prompt 6 — RJC's responses to Q6.24, Q6.25 and Q6.26 (2026-10-19)

**Q6.25** (keep the runs joined across the gaps): no change. Checked that ALIS's
report counts only the pixels it fits that have a positive error
(`report._regions`: `_wavefit`, `flue > 0`), as the dashboard does
(`results.counted`). Recorded in the document's notes; joining is sound under the
test's own hypothesis.

**Q6.24** (count `run ncpus` from the command line too; a count leaving no CPU
uses one, with a warning) and **Q6.26** (`run blind False`, `chisq miniter 5`):
not made. The edits to `alis/load.py` and `alis/config.py` were refused by this
session's permission check, so ALIS is as it was, and so are the tests, the
CHANGELOG and `ALIS_workflow.md` that follow from them. They wait for RJC.

**Q6.27** raised: whether new projects write `chisq miniter 5` once that is ALIS's
default (lean: yes).

### Q6.24, Q6.26 and Q6.27, after RJC's go-ahead in chat

**Q6.24 — CPUs on ALIS's command line** (`alis/load.py`):
- `reapply_cli_overrides` counts `run ncpus` with `cpucheck`, as a model's own is
  counted, so `--set "run ncpus -1"` is all bar one. It counts quietly
  (`cpucheck(..., quiet=True)`): `optarg` has already counted the same value and
  said so. ALIS's logger ignores `verbose=`, so the quiet count was the only way not
  to print the warning twice, found by tracing a real run.
- `cpucheck`: a count that leaves no CPU (0, or -40 on this computer's 14) fits
  with one, with a warning; "all bar one" on a one-CPU computer is one.
- `run_alis fit_spectra.mod -p 0 --set "run ncpus -1"`, which stopped with a
  traceback before, fits on 13 CPUs, χ² 357.8742 as the reference; `-40` fits
  on one CPU, with one warning.
- Tests: `test_load_units.py` (0, -8, -20 → 1 with the warning; one-CPU computer;
  `reapply_cli_overrides` counts, quietly), `test_cli.py` (optarg → model line →
  `check_argflag` → re-applied: 7 of 8, whether or not the model sets `run ncpus`).
- Dashboard: runs are given the CPUs preference as it is (`run ncpus -1`), and
  `settings.cpus` (Task 6.5's workaround) is gone; `settings.leaves_no_cpu` says
  whether a count leaves none. The pre-flight check's `cpus` finding is a note now,
  not a refusal ("ALIS will fit with 1 CPU: the CPUs preference (-14) leaves none
  of this computer's 14…"). `preflight.py`'s docstring and the Design list it with
  the notes.

**Q6.26 — ALIS's defaults** (`alis/config.py`): `run blind False`, `chisq miniter
5`.
- `test_config.py` pins both. CHANGELOG: a Changed section (both defaults) and a
  Fixed entry (the command line's CPUs). `ALIS_workflow.md` §2.1's table (`blind`
  False; `ncpus` with `--set` and a count that leaves none), and the user guide's
  table (`doc/tex_files/description.tex`: blind False, miniter 5; the PDF is not
  rebuilt).
- `chisq miniter` holds back only the `ftol` and `xtol` stops (`minimise.py`,
  statuses 1–3), not `gtol` or `atol`; the CHANGELOG says so.
- The dashboard follows: a model with no `run blind` line is not under global
  blind, so deleting `run blind True` is refused as switching it to False is
  (`test_dashboard_guard.py`), and the unblinding test of a model relying on the
  old default became one of a model without the line. `blinding.unblind`'s branch
  for a model that relies on the default is kept (it reads the default), with its
  comment corrected.
- The Run box no longer lists the metal example's `run blind False`; its
  `chisq miniter 10` is listed (`test_the_run_box_settings`, which now checks the
  read-only `run blind` row with the model under global blind).

**Q6.27 (a)** — a new project writes `chisq miniter 5`
(`preferences.NEW_PROJECT_SETTINGS`).

The review page's question 4 shows the answer (draft 3).

**Checks.** The harness cases of the six models that leave `chisq miniter` to the
default (`test_minimisation`, `test_fixed_param`, summed_coldens's `test_bundle`,
Her36's `test_bundle_matches_plain`): 14 passed (292 s). The review page's
screenshots were retaken (real fits from the window) and the page republished as
draft 3; its question 4 thread answered and resolved. `pytest -m unit`: 1973
passed, 121 skipped (85 s; the 16 warnings are `test_save_helpers.py`'s log10 of
a zero, in code this work did not touch); `pytest -m gui`: 164 passed (289 s);
`pytest -m fast`: 129 passed (875 s).
