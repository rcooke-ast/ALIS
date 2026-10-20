# Prompt file for ALIS software dashboard creation -- STAGE 6

> **The Fit tab (step 6). This stage completes v1.** The fourth tab of the skeleton is
> filled in, with its three sub-tabs, Inspect, Results and Compare (D34). At the end
> of the stage, a user can take a project from a spectrum to a fit that has been run,
> inspected, kept and compared, without leaving the dashboard:
> - **Running (F6, F9, S17):** the settings of the fit, generated from ALIS's own
>   list; a pre-flight check; Run, and Save and Run, which fit in a separate process
>   handed the project in memory, never as a file; the χ² and the model of each
>   iteration drawn as it goes; Stop and Cancel.
> - **Inspect (D38, D39, S18):** every snip, one ion or one snip, with the starting
>   model, the best fit or the fit in progress, the residuals, and the fit-quality
>   marks of each snip.
> - **Results (D40, D51, S19, S20, F7, D26):** each parameter's value and error, its
>   tie state and flags; the correlations; the fit statistics; Continue from best
>   fit, and Use best fit for one component or one continuum; the run history, to
>   which the user commits a run with a description.
> - **Compare (D41):** any two runs, their quantities matched across different
>   parametrisations, the change of each in σ, and both models on the same data.
> - **Convergence and outliers (S21, S29, D27):** ALIS's convergence check; random
>   restarts, their values kept and exported; and pixels that the best fit leaves far
>   out, reviewed before they are left out of the fit.
> - **Runs made outside the dashboard (S31):** a fit made by `run_alis
>   project.model` is shown as the latest run, ready to inspect or commit.
>
> Everything the tab shows comes from the project model, through the blinding gate,
> and every change to the model is one step of the history, as in Stages 3–5. The
> logic needs no Qt and goes in `alis/dashboard/` (new modules: `runs.py`,
> `runner.py`, `settings.py`, `preflight.py`, `results.py` and `compare.py`), tested
> with pytest alone; the windows go in `alis/dashboard/qt/`. ALIS outside
> `alis/dashboard/` changes only where a query agrees (Q6.6, Q6.19).
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`. "Q0.*n*" to "Q5.*n*" refer to
> `dashboard_stage0.md` to `dashboard_stage5.md`. The plan for all stages is in
> `dashboard_stage0.md`.

## Design

*Written by Claude on 2026-10-15, from the design documents (D24–D27, D34, D38–D41,
D48, D49, D51, D60; QF.11, QF.20, QF.26(c), QF.32), the mockups of the Fit tab
(`doc/dashboard/mockups/fit_inspect.html`, `fit_results.html`, `fit_compare.html`),
RJC's review of them (Task 0.10 and Q0.4, Q0.9, Q0.11 of `dashboard_stage0.md`) and
what Stages 1–5 taught. Revised on 2026-10-18 to follow RJC's responses to
Q6.1–Q6.22. Each part names the queries that shaped it.*

### What a user can do at the end of this stage (v1)

1. **Prepare** (Stages 3–5): the spectra, the systems, the snips with their fit
   regions and continua, and the components, on the Data, Regions and Components
   tabs, with the `.mod` panel beside them.
2. **Run:** press Run (Ctrl+R) on the Fit tab, or Save and Run (Ctrl+Shift+R) to save
   the project first. A pre-flight check lists what would stop the fit, and what is
   worth a look first. The fit runs in the background while the user goes on
   working, with its χ² and its model redrawn after each iteration. Stop ends it early
   and keeps where it got to; Cancel ends it and keeps nothing.
3. **Analyse:** inspect each snip with the best fit and its marks; read the results
   and the correlations; continue from the best fit, or take the best fit of one
   component or one continuum; commit the run, with a description, to the run
   history; compare any two runs; check the convergence, with random restarts whose
   values can be exported; review the pixels the fit leaves far out.
4. **Plot** (step 8) is after v1. Until then, ALIS's own `plotscript` settings write a
   plotting script, as they do for `run_alis`.

### Layers

```
alis/dashboard/qt/   fit.py         the Fit tab: Inspect, Results and Compare, which
                                    replace the frames of tabs.py; the fitting
                                    process (QProcess), its progress, the fit in
                                    progress, Stop and Cancel
                     plots.py       the spectrum view gains a second model (Compare)
                                    and marked pixels (outliers)
                     dialogs.py     All settings…, Commit run…, Random restarts…
                     components.py  Use best fit on each component card (Q6.17)
                     regions.py     the Model switch gains the best fit (Q5.8); Use
                                    best fit in the Continuum box (Q6.17, Q6.20)
                     data.py        Reset FWHM (Q6.21)
                     window.py      Run, Save and Run, Stop and Commit; the run in the
                                    status bar; closing with a run going
alis/dashboard/      runs.py        a run of the project: the bytes handed to the
                                    process, the dashboard's own settings, the
                                    progress, Stop and Cancel; the latest run and the
                                    committed runs (commit, describe, remove,
                                    restore); what a run holds (values, errors, the
                                    covariance, its statistics, its restarts)
                     runner.py      the fitting process (`python -m
                                    alis.dashboard.runner`): reads the project from
                                    its standard input, runs it with ALIS in memory,
                                    sends its progress, stores its result
                     settings.py    the settings, from ArgFlag (F9): the model's own,
                                    the computer's, the dashboard's for its runs, and
                                    those for the command line only
                     preflight.py   the pre-flight check (S17)
                     fitview.py     what Inspect shows: the View menu, the panels
                                    with their axes and ticks, the chips, the
                                    progress and the status bar's words (Task 6.4)
                     results.py     the results table and its flags, the
                                    correlations, the fit-quality marks, the best fit
                                    of each snip (through profile.py), Continue from
                                    best fit, Use best fit, the outliers
                     compare.py     two runs' quantities, matched across
                                    parametrisations, their changes in σ; the random
                                    restarts' draws, assessment and export
alis/                bundle.py      run split into running in memory and storing
                                    (Q6.19(a))
                     main.py        an optional per-iteration callback, passed to
                                    the minimiser (Q6.19(b))
                     config.py      a description of each setting (Q6.6(a))
                     report.py      its statistics made public (Q6.6(b))
```

The names may change as the code is written; the log records the names as built.
As built: Results, Compare and the tools' window are `qt/fitresults.py`,
`qt/fitcompare.py` and `qt/fittools.py`, beside `qt/fit.py` (Inspect and the run
going); the random restarts' logic is `restarts.py`.

- **No logic in the widgets**, every change to the model one step of the `History`,
  every value through the `Gate` (Stages 3–5). Running, committing and describing a run
  change no model text, and are not steps (Q6.7).
- **A run is ALIS's own run of the project.** The dashboard's process runs the project
  as it stands, handed to it in memory and never written to a file, exactly as
  `run_alis project.model` runs it (D9; Q6.2, Q6.14). Its result is the bundle's
  `runs/latest/`, as `run_alis project.model` leaves it.
- **The best fit drawn is ALIS's** (D60): `profile.py` evaluates the run's `.mod.out`
  values on the snips, as the Components tab evaluates the starting values (Q6.5), and
  the values of the fit in progress in the same way (Q6.15).
- **Blinding (D24, D49):** best-fit profiles may be drawn; values never shown. What
  each part of the tab shows of a blinded parameter is listed under "Blinding" below.

### Running a fit (F6, F9, S17, D38; Q6.2–Q6.4, Q6.14–Q6.16, Q6.19)

- **Run, and Save and Run** (Q6.2, Q6.14(c)): Run (Ctrl+R) fits the project as it
  stands, unsaved changes included, and is never a Save: the model on disk changes
  only with Save (D48). Save and Run (Ctrl+Shift+R) saves the project, then makes the
  same run, so that the model on disk is the one fitted, and a later `run_alis
  project.model` fits the same model.
- **The fitting process** (Q6.14(a), Q6.19): the fit runs in a separate process
  (`runner.py`), so that the window stays responsive and Stop works. The dashboard
  hands it the project through a pipe: the bytes that Save would write, never written
  to a file. The process runs ALIS on them in memory, as `run_alis project.model`
  does: the data, the atomic table, and the hidden lines restored in memory only
  (`bundle.run`'s first half, Q6.19(a)). It sends back its progress and its end.
- **Where the result goes** (Q6.14(b)): into the project's file at once, under the
  bundle's lock, as `run_alis project.model` stores a run (`bundle.run`'s second
  half): its `runs/latest/` and nothing else, so that a finished fit is never lost.
  The open project takes it in as it takes a command-line run (`Session.take_runs`).
  A project with no file yet (a new project, or an imported fit before Save as) keeps
  the result in memory and in its recovery copy, and its next Save writes it.
- **What it is given** (Q6.16(a)): the dashboard's own choices for its runs, as
  command-line settings (`bundle.run`'s overrides), never written in the model: this
  computer's backend, CPUs and GPUs (unless the model sets them, Q6.4(a)), the outputs
  the tab reads (the covariance and the report), `out verbose 1` and no plot windows
  (`plot fits False`). `run.json` records them, as it records any command-line
  setting.
- **One run at a time.** The model can be edited while it runs: the run fits the model
  as it was when Run was pressed, and the Fit marker then says the model has changed
  since.
- **Progress, and the fit in progress** (Q6.15(a), Q6.19(b)): after each iteration,
  ALIS's per-iteration callback gives the process the iteration, the χ² and the
  parameter values, which it sends through the pipe; they are never printed.
  - The χ² goes on the progress chart (linear or log, D38) and in the status bar ("Run
    5 · iteration 9 · χ² 3796.2 · 0:41"), with the previous run's χ² against
    iteration in grey (kept in its `run.json`).
  - The values are drawn as the fit in progress on Inspect (Final) and shown on
    Results, under a notice: "The fit in progress: the model is redrawn after each
    iteration (iteration 9, χ² 3796.2)". Drawing is throttled, so that a fast fit
    skips iterations rather than falling behind (J1358p6522's 12 snips draw in 30 ms,
    Q1243p307's 51 in 230 ms).
  - Nothing else of ALIS's output is shown. The messages the tab shows (a warning, or
    the error that stopped the run) pass the bundle's mask of hidden lines
    (`bundle._MaskHidden`).
- **Stop and Cancel** (Q6.3, Q6.15(b)): Stop ends the fit at its current iteration,
  as ALIS's own interruption does (the minimiser's handler of `SIGQUIT`). ALIS writes
  its outputs, and the run is kept as "interrupted", with its values, to inspect and
  to continue from (Continue from best fit, then Run). ALIS sets an interrupted fit's
  errors to its values, so they are never shown. Pressing Stop again (now "Cancel")
  ends the process at once; nothing is kept, and the latest run stays as it was. There
  is no Pause (Q6.15(c)).
- **An error** (ALIS stopped with one): the Fit marker's "!" with the message, as
  Stage 3 shows it (`runs/last_error.json`), and the line it names highlighted in the
  `.mod` panel when it names one.
- **Closing the dashboard during a run** asks: Stop the run, Cancel it, or Keep it
  running. A run kept running still stores its result in the project's file, and the
  Fit tab shows it when the project is next opened, as a command-line run (S31). A
  project with no file offers only Stop and Cancel. Re-attaching to a run with its
  live progress is after v1 (F6).

### Settings (F9, D38; Q6.4, Q6.6, Q6.16)

- **The Run box** shows the computer's settings (Backend, CPUs, GPUs), then "Settings
  you changed": the model's `par` lines whose values differ from ALIS's defaults, each
  editable in place (one step) and each with ✕, which returns it to the default (its
  line removed).
- **The computer's settings** (Q6.4(a)) are preferences of this computer, given to
  each run on its command line, so that a model fitted on a laptop runs unchanged on a
  GPU machine. A model that sets them keeps its lines, which win; the Run box says
  that the model's own value is used.
- **The dashboard's own settings for its runs** (Q6.16(a); "What it is given" above)
  are never written in the model. The model keeps what the user wrote, so `run_alis`
  on the project, or on its exported plain files on another machine (a headless one
  included), runs it exactly as written, plot windows included (Q6.4).
- **All settings…** (Q6.16(b)) lists every setting of `alis.config.ArgFlag`, by
  section, with its default, its type (a tick box, a number, a word, or a menu where
  ALIS takes only some words, such as `run backend`) and a one-line description
  (Q6.6(a)), with a filter. A change is a step that writes or changes the setting's
  `par` line.
  - The computer's settings are there too.
  - Those the dashboard sets for its own runs show, beside the model's value (which
    `run_alis` uses), what the dashboard's runs use.
  - Those a bundle cannot run (`sim random`, `sim perturb`, `sim beginfrom`, `iterate
    model`, `generate data`; `bundle.unsupported`) are written like any other, and
    marked "command line only". While one is set, the pre-flight check refuses Run
    (below). Generating data from the dashboard is future work (`deferred_work.md`
    §7.5).

### The pre-flight check (S17; Q6.4, Q6.16)

Run checks the project first (`preflight.py`). Each finding names its line in the
`.mod` panel, or its snip, and a click shows it.
- **Run is refused** when ALIS would stop, or a bundle cannot run the model:
  - the model does not read, or ALIS's loaders refuse it (`validate.full`);
  - a "command line only" setting, or a data line that reads a file from disk
    (`bundle.unsupported`), with the way to run it: File → Export plain files…, then
    `run_alis` on them (Q6.16(b));
  - a starting value outside its limits;
  - no free parameter, or more free parameters than fitted pixels.
- **Notes**, after which Run goes ahead (the Run box shows "Pre-flight check: no
  problems; 2 notes", and the notes are a click away):
  - a free parameter that reaches no fitted pixel, by ALIS's own influence table
    (`load.load_par_influence`), so that the data cannot constrain it;
  - an ion in no snip (Stage 5's marker);
  - a starting value at one of its limits;
  - a b_turb starting below 1 km/s (a preference; `doc/ALIS_workflow.md` §4.1.2);
  - a snip with no continuum;
  - CPUs that leave none of this computer's (`run ncpus 0`, or −20 on a computer
    with 8), so that ALIS fits with one (Q6.24);
  - the notices of an imported fit kept as imported (D20, D22), pixels fitted twice
    (D19), and regions changed since the components were last edited (Stage 5's
    marker).

### Inspect (D38, D39, S18; Q6.5, Q6.15)

- **Left:** the Run box (above); Run, Save and Run, and Stop; Convergence check… and
  Random restarts… (S21); under them the progress chart.
- **The view:** a View menu (all snips, each on its own axis; one ion, grouped as
  the Components tab's navigator, the Lyα forest included, its panels sharing one
  axis that starts at the Components tab's range; one snip) with ‹ ›; Columns, the Components
  tab's setting (D38), greyed when one snip is shown; Velocity | Wavelength (velocity
  relative to the transition a snip was cut for, as the Lyα forest's panels; a snip
  with no transition in the project, in wavelength); the zoom toolbar (D38).
- **Initial | Final** (Q6.5): Initial is the model as it stands now, the one the next
  run starts from (as the Components tab draws it); Final is the shown run's best fit,
  or, while a run goes, the fit in progress, under its notice (Q6.15). Each panel
  shows the data divided by the shown model's own continuum (the best-fit continuum
  for Final), the model, the fit regions, the residuals underneath, and the component
  ticks, locked (no drags on this tab).
- **The shown run** is the latest run, or the run chosen in the run history; its name
  is above the panels ("Final · run 3").
- **Fit-quality marks** (S18, D39): each panel's corner, and a chip per snip in the
  "All snips" strip, show χ²ν and the runs test of the model the panels draw
  (Initial's or Final's; RJC's review of Task 6.5), each with its own mark: ✓ within
  2σ, ! 2–3σ, ✕ beyond 3σ (`out reportsig`). They are computed as `alis/report.py`
  computes them (χ²ν against its expected spread, which is flagged only above; the
  runs-test z, flagged either way), over the pixels ALIS fits inside the fit regions
  (Q6.6(b)), and follow the fit in progress while a run goes. The residual strip
  shades the pixels they do not count, and Residuals All | Fitted shows or hides
  those pixels' residuals (RJC's review). A click on a chip shows that snip.
- **Continue from best fit** and **Review clipped pixels…** sit above the panels
  (below).
- The best fit of a snip that the run did not have (cut since) is not drawn, and its
  chip says so.

### Results (S19, S20, D40, D51, F7, D26; Q6.7, Q6.8, Q6.12, Q6.17, Q6.20, Q6.21)

- **The results table** of the shown run ("Results · run 3"), or, while a run goes,
  the values in progress under its notice, with no errors (ALIS computes them at the
  end; Q6.15(a)). It is grouped as in the mockup:
  - each system, the primary first, with its ions and components: log N, z (with v
    relative to the system), b and T;
  - the Lyα forest, collapsed, with its flagged rows shown;
  - the continua and the instrument (FWHM, shifts, zero levels), collapsed.

  Each row has Value, Error, State (● free, ■ fixed, ⛓ tied, ƒ linked, as on the
  cards) and Flags: "at limit", "unconstrained", "blinded", and the notices kept as
  imported ("not tied to H I"). A row selected highlights its lines in the `.mod`
  panel (S16). "Copy as table" copies the rows as tab-separated text, through the
  gate.
- **"At limit" and "unconstrained"** (Q6.12) mark a value worth a closer look:
  - **at limit:** the value equals one of its bounds (a `lim` line, or voigt.py's own
    limits) to the precision of the run's values, as the minimiser leaves a parameter
    that presses on its bound;
  - **unconstrained:** the error larger than the value for b and T, larger than 1 dex
    for log N, larger than 10 km/s for a redshift (as velocity), and larger than the
    value for the other parameters, each threshold a preference. A parameter at a
    limit is flagged "at limit" only;
  - **a continuum** (Q6.23): its coefficients are flagged neither way, since a
    polynomial's higher terms are often consistent with zero. Instead, the
    continuum is flagged "negative" when its best fit is below zero at a pixel of
    its snip from the first fitted pixel to the last (Q6.28). Zero is allowed, and
    the coefficients themselves may be negative.

  J1358p6522's Lyα forest b_turb of 0.29 ± 60.53 (its bound 0.2) therefore reads
  "unconstrained", not "at limit" as the mockup shows it.
- **Fit statistics:** χ² (and its starting value), degrees of freedom, χ²ν,
  iterations, the run time, why the fit stopped (`utils.getreason`), the convergence
  check's result when it ran, and where and when the run took place (host, command,
  finished; S31).
- **Correlations** (D51, under the statistics): the correlation matrix of the free
  parameters of a group chosen in a menu (a system, an ion, the continua, everything),
  from the run's covariance, in a colour-blind-safe diverging scale (D43), with the
  pair and its coefficient on hover. A correlation reveals no value, so blinded
  parameters are included, by name.
- **The run history** (F7, D26, D51, Q6.7, beside the table): the latest run first,
  with Commit…, which asks for the user's own description (D40); then the committed
  runs, newest first: number, description (edited by a double-click), χ²ν, when,
  where ("cmd" for a run made by `run_alis`, S31). Selecting a run shows it on Inspect
  and Results; a committed run can be removed, after a confirmation.
  - A committed run keeps its `run.json` (with the description), its input model, its
    outputs, its restarts (Q6.18(a)) and its snips as they were, so that its model can
    be restored. Each snip's content is stored once, under `runs/files/<sha256>`, and
    shared by every run that has it.
  - Committing, describing and removing are saved with Save, like the model, but are
    not steps of the undo history. Runs are numbered in the order they are committed,
    and a number is never reused.
- **Continue from best fit** (S20, Q6.8): the shown run's best-fit value of every
  parameter is copied into the model, as one step, after a preview of the lines it
  changes. Only parameters unchanged since the run receive values (as RJC's Q6.8
  response puts it; Task 6.3): each place of the parameter on a line that is the same
  line as in the run's model, whatever its values, with the same value written. The
  preview lists the others, which keep theirs. Linked parameters are left to ALIS,
  and fixed ones keep their values. A value copied into a hidden line stays hidden,
  and under global blind every line that receives a value is hidden
  (`Project.mark_hidden`, S20). An interrupted run can be continued from, so that a
  stopped fit can be resumed.
- **Use best fit** (Q6.17, Q6.20) copies the best fit of one component, or of one
  continuum, as one step:
  - **a component card** (Components tab): the shown run's best-fit z, b_turb, T and
    each ion's log N of that component, into its lines. It is asked for one item, so
    it copies even into lines changed since the run, provided the run has the
    component (matched by its redshift label); an ion the run did not have keeps its
    value;
  - **the Continuum box** (Regions tab): the snip's continuum, matched by the snip's
    specid. The line's current function, order, range and `scale=` factors are fitted
    to the run's best-fit continuum on the snip's pixels, weighted by the data's
    errors, whatever the order (Q6.20). For an order raised or unchanged, the fit
    gives the run's continuum exactly (zeros above the run's coefficients); for an
    order lowered, the nearest curve of that order. The Continuum box then says how
    far the copy departs from the run's continuum, in units of the data's errors
    (largest and rms over the fitted pixels), and the panel draws the run's
    continuum dashed beside it, so that the user can change the order and copy
    again;
  - the button is disabled, with a tooltip, when the shown run has no best fit for the
    item. Blinding is as for Continue from best fit: the card shows the mask, and the
    step's name holds no value.
- **The Data tab's FWHM, shift and zero level** are left to Continue from best fit
  (Q6.17(d)). Each row's FWHM can be reset to its default, the FWHM given when its
  file was added, or that it had when it was imported, as one step ("Reset FWHM to 7
  km/s" in the cell's menu, and ↺ in the cell while it differs; Q6.21). Only the value
  is reset; free or fixed is left as it is.

### Compare (D41; Q6.9)

- **Choose** any two runs (the committed runs and the latest), Swap, and "Restore run
  N's model…", which replaces the model and its snips with the run's, as one step,
  after a preview of what changes. By default the latest run is compared with the
  newest committed run, or, when the latest run is that one, with the committed run
  before it (RJC's review).
- **The quantities** (Q6.9), matched across different parametrisations:
  - each ion's total log N in each system, from the components' column densities,
    with its error from the run's covariance;
  - each isotope's ratio to its element (log D/H), whether the run fits the two
    column densities or the ratio itself (`ion=2H_I/1H_I`, or a `variable` linked to
    it), with its error from the covariance;
  - the components, matched by system, ion and velocity (the nearest within 10 km/s,
    a preference): v, b and T, and how many components each ion has;
  - the instrument (FWHM), χ²ν, degrees of freedom and iterations.

  Each row shows both values, Δ (later − earlier) and |Δ|/σ, with σ the larger of the
  two runs' errors, and "—" when a run has no covariance.
- **Blinded quantities** show only |Δ|/σ, with no values and no sign (Q0.9, Q0.11;
  `Gate.sigma_change`); a quantity derived from a hidden value is hidden.
- **The view:** both best fits on the same data (the first solid, the second dashed),
  their difference in units of the error underneath, with Inspect's View menu, ‹ ›,
  Columns and toolbar (D41).

### Convergence and random restarts (S21; Q6.10, Q6.18, Q6.22)

- **Convergence check…** runs the fit with ALIS's own check (`run convergence True`
  on its command line, with `run convcriteria` shown and editable); Fit statistics
  then says "Converged", or "Not converged for 3 of 41 parameters".
- **Random restarts…** (Q6.10, Q6.22): after a run with a covariance, N fits (10, a
  preference) start from values drawn as `sim perturb` draws them after `sim
  beginfrom` the shown run: about its best fit, from its covariance, each draw clamped
  to ±`sim maxperturb` of its value and then to its limits, with fixed and linked
  parameters unchanged. They run one after another in the fitting process, with
  "restart 3 of 10" in the status bar, and Stop ends them. They are assessed as
  `convergence.assess_restarts` assesses them (`sim convergetest` and `sim
  convergesig`): for each free parameter, the largest |Δ|/σ across the restarts,
  flagged beyond the tolerance, and one line of summary. A restart that reaches a
  lower χ² is offered to continue from. How ALIS's own `sim perturb` draws, and whether
  a bundle can run it, is deferred (`deferred_work.md` §7.6, Q6.18(d)); the dashboard
  follows what is decided.
- **What is kept** (Q6.18(a)): each restart's starting values, best-fit values and
  errors, its χ², degrees of freedom, iterations and status, with the run the restarts
  started from, and committed with it. One set is kept with a run: new restarts
  replace it, after a warning and a question (RJC's review). They form a table of one row per restart and
  one column per parameter, named by its ion, component and parameter ("O I a1 log
  N"), with its label.
- **The restarts' window, and Export restarts…** (Q6.18(b)): the window shows the
  assessment table, and χ² against a parameter chosen in a menu. Export restarts…
  writes the table as text, in the form `sim perturb` writes its own
  (`<model>.perturb_0-10`: one row per fit, the first row the starting values, here
  the shown run's best fit), with a header naming the columns and χ² as the last
  column, so that a script that reads ALIS's file reads this one too.
- **Blinding** (Q6.18(c)): a hidden parameter's values are neither exported nor
  plotted. Its columns are left out, and the header says so; its |Δ|/σ appears in the
  assessment, unsigned. Unblinding brings its values back.

### Outliers (S29, D27; Q6.11)

- **Review clipped pixels…** marks, on Inspect's panels, the fitted pixels whose
  residual from the shown best fit is beyond 5σ (a preference). Lone pixels and pairs
  are ticked; three or more in a row are shown unticked, since they may be absorption
  the model misses (QF.26(c)).
- **The user accepts or rejects them:** each panel has Accept all and Reject all, and a
  click on a pixel ticks or unticks it. Nothing leaves the fit until Apply, which
  leaves the ticked pixels out of the fit, as Tweak dataset regions' right drag does
  (the snip's fit column; Stage 4), as one step ("Leave out 6 outliers") that undo
  reverses, and offers Run.

### Command-line runs and the bundle on disk (S31; Q3.8, Q6.14)

- A run made by `run_alis project.model` while the project is open is taken in
  without asking (Q3.8(a), `Session.take_runs`), and one made while it was closed, or
  stored by a run kept running after the dashboard closed, is there when it opens. The
  run history lists it as the latest run, "cmd" for a command-line run, with where and
  when it ran, ready to inspect and commit.
- The χ² against iteration of a command-line run is not known (the command line does
  not record it), so the progress chart shows only its end.

### The Regions tab's Model switch (Q5.8; `deferred_work.md` §7.2)

- The switch gains a choice when a run exists: the starting model, or the latest
  run's best fit, drawn as on Inspect.

### Blinding (D24, D49, QF.20)

- **Global blind** is `run blind True`, which a model writes to be blind: ALIS's
  default is False (Q6.26), so a model without the line is not blind, and deleting
  the line is refused as switching it off is.
- **Values:** best-fit values, and those of the fit in progress, through
  `Gate.best_fit` (masked for a hidden parameter and under global blind); starting
  values as on the other tabs. A parameter hidden in the current model is hidden in
  every run's results too (Task 6.3).
- **Offsets** (Task 6.3): ALIS writes the best fit of a line with a `blindrange` with
  its offset added (`parout`). The dashboard takes the offset away to draw the best
  fit and to copy it, so those best-fit values are masked, and a line of them that
  receives values is hidden.
- **Errors** are shown (QF.20(b)), except an interrupted run's, which ALIS sets to
  the values.
- **χ²**, its progress, the fit-quality marks, the correlations and the outliers are
  shown: none reveals a value.
- **Compare and restarts:** a hidden quantity shows only |Δ|/σ, unsigned. Export
  restarts… and the restarts' chart leave hidden parameters out.
- **Profiles** of blinded components are drawn (D24), in the fit in progress too.
  So are their ticks, Final's at the best-fit redshifts, under global blind too,
  each with its component's name and never a value (RJC's review of Task 6.5).
- **Copy as table, tooltips and messages** pass the gate; the process's output is not
  shown, and its messages pass the bundle's mask.
- **Continue from best fit and Use best fit** keep a hidden line hidden, and under
  global blind hide every line they fill (S20).

### Markers, preferences and actions

- **The Fit marker** (F12; Stage 3's rules): ○ no run yet; ✓ the latest run, with its
  χ² and when; ↻ the model has changed since the latest run, or a run is going
  ("Running: iteration 9"); ! the last run stopped with an error, or the pre-flight
  check refuses Run.
- **New preferences** (`preferences.py`, `fit.*`): this computer's backend, CPUs and
  GPUs; the progress chart's axis (linear); the threshold of outliers (5σ); the
  number of random restarts (10); the starting b_turb below which the check notes it
  (1 km/s); the velocity within which Compare matches components (10 km/s); the
  thresholds of "unconstrained" (Q6.12).
- **Actions** (`actions.py`, scope "Fit"): Run (Ctrl+R) and Commit run… (both built
  now, no longer "coming"), Save and Run (Ctrl+Shift+R), Stop (Cancel when pressed
  again), Continue from best fit, Convergence check…, Random restarts…, Export
  restarts…, Review clipped pixels…, and ‹ › for the view (the tab's own keys). Use
  best fit (the cards and the Continuum box) and Reset FWHM (the Data tab) are buttons
  and menu items of their own tabs.

### Not in this stage

- The Plot tab and the plotting presets (step 8, D28, D42), after v1.
- Re-attaching to a run with its live progress after the dashboard was closed (F6),
  command search (F10), S15's link editor (Q5.9), reproducing an imported fit (S28),
  attaching a source spectrum to an imported model (S23): after v1.
- Pausing a run (Q6.15(c)).
- Generating data from the dashboard (`deferred_work.md` §7.5); ALIS's own `sim
  random` and `sim perturb` in a bundle, and the centre of their draws (§7.6).
- Orders mode, PypeIt spec1d and the stacked spectrum (D29–D32, Q4.20): after v1.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage6_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests;
> run the `fast` batch before the stage closes. A task that changes ALIS outside
> `alis/dashboard/` also runs the regression harness, its `test_bundle` mode included.

**6.1 — Runs and the fitting process (`runs.py`, `runner.py`; `bundle.py`, `main.py`;
F6, F7, D26, S31; Q6.2, Q6.3, Q6.7, Q6.14, Q6.15, Q6.19).**
- ALIS (Q6.19): `bundle.run` split into running a bundle held in memory (its outputs
  and `run.json`) and storing them in the project's file, `run_alis project.model`
  doing both as now; `ClassMain`'s optional per-iteration callback, passed to the
  minimiser, `None` by default.
- The fitting process: the project handed through a pipe, as the bytes Save would
  write; the dashboard's own settings; the progress (the iteration, χ² and values)
  sent back; Stop and Cancel; the result stored in the project's file under its lock,
  or kept in memory and in the recovery copy for a project with no file. The latest
  run and the committed runs (commit with a description, describe, remove, restore a
  run's model; each snip stored once, under `runs/files/<sha256>`); what a run holds
  (its values, errors and covariance, mapped to the model's parameters; its
  `run.json`).
- **Check** (unit tests): the regression harness unchanged, and `run_alis
  project.model` as before; a run of `examples/metal_line_abs` through the process
  gives the `.mod.out` that `run_alis` gives, and the χ² and values of every
  iteration, and writes no file but the project's own; a result stored while the
  project has unsaved changes leaves the model on disk unchanged; a project with no
  file keeps the result in memory and in its recovery copy, and Save writes it; Stop
  leaves an "interrupted" run with its values, and Cancel leaves the latest run as it
  was; a committed run survives Save and reopening, and restoring it gives its model
  and snips byte for byte, as one step; on `examples/blind` and under global blind,
  nothing hidden appears in the progress the tab shows, the messages or `run.json`
  (the gate's `strings` test).

**6.2 — Settings and the pre-flight check (`settings.py`, `preflight.py`;
`config.py`; F9, S17; Q6.4, Q6.6(a), Q6.16).**
- ALIS (Q6.6(a)): a one-line description of each setting, as metadata of its field in
  `alis/config.py`, printed by `run_alis --list-settings`.
- The settings from `ArgFlag`, with their types and descriptions: the model's own, the
  computer's, the dashboard's for its runs, and those for the command line only;
  setting and clearing one as a step. The pre-flight check, with what refuses Run and
  what is a note.
- **Check** (unit tests): every setting of `ArgFlag` is offered and has a description
  (`run_alis --list-settings` prints it); a setting set and cleared restores the text
  byte for byte; a run leaves the model's text unchanged, and the dashboard's own
  settings appear only in `run.json`; a computer's setting written in the model wins
  over the preference; a "command line only" setting refuses Run, and says how to run
  it; each finding of the check, refusing and noting, on a model made to have it;
  J1358p6522 as imported gives its two notices as notes and no refusal.

**6.3 — Results (`results.py`; `report.py`; S18–S20, D39, D40; Q6.5, Q6.6(b), Q6.8,
Q6.12, Q6.17, Q6.20).**
- ALIS (Q6.6(b)): `report.py`'s statistics (the runs-test z, the χ²ν deviation) made
  public functions of a snip's residuals, unchanged in what they compute.
- The best fit of each snip (`profile.py` with the run's values, or the values in
  progress); the fit-quality marks; the results table with its groups, states and
  flags; the correlations; the statistics; Continue from best fit; Use best fit for a
  component and for a continuum; the outliers.
- **Check** (unit tests): the best fit drawn agrees with ALIS's `_fit.dat` model, on
  metal_line_abs and J1358p6522, to the precision of the values the `.mod.out` prints;
  the marks equal the `.report`'s χ²ν and runs z (J1358p6522's reference: nine snips ✕
  by the runs test alone); "at limit" on a model made to stop on a bound, and
  J1358p6522's Lyα forest b_turb rows "unconstrained"; Continue from best fit is one
  step that ALIS reads and one undo restores, skips a line changed since the run, and
  hides what it fills under global blind; Use best fit copies one component into lines
  changed since the run; one continuum with its order unchanged or raised, or its
  range or function changed, gives the run's continuum to the precision of its values,
  and with its order lowered gives the nearest curve, with how far it departs from
  the run's.

**6.4 — Inspect (`qt/fit.py`; D38, D39, F6; Q6.14, Q6.15).**
- The Run box (the computer's settings, the settings changed, All settings…, the
  pre-flight check); Run, Save and Run, Stop and Cancel; the progress chart; the View
  menu with ‹ ›, Columns, Velocity | Wavelength, Initial | Final; the panels and the
  snip chips with their marks; the fit in progress with its notice; the status bar;
  closing during a run; the Fit marker.
- **Check** (`gui` tests): a run of metal_line_abs from the tab draws its progress and
  the fit in progress under its notice, then its best fit, and the marker becomes ✓;
  after Run the project's file holds the run and the model on disk is unchanged,
  while Save and Run saves first; Stop keeps an interrupted run, and Cancel keeps
  nothing; Run is refused with a pre-flight problem and goes ahead with notes;
  J1358p6522's chips as in the mockup; Initial and Final draw the starting model and
  the best fit.

**6.5 — Review of Inspect and running (Q6.1).**
- Publish the sub-tab's screenshots beside its mockup, on a new page, for RJC's
  comments, and apply them before Results and Compare are built.

**6.6 — Results, and Use best fit (`qt/fit.py`, `qt/components.py`,
`qt/regions.py`, `qt/data.py`; S19, S20, D40, D51, F7; Q6.17, Q6.21).**
- The results table (the values in progress while a run goes), Copy as table, the
  statistics, the correlations with their menu, the run history (Commit…, the
  description, removing a run), Continue from best fit; Use best fit on the component
  cards and in the Continuum box; Reset FWHM on the Data tab.
- **Check** (`gui` tests): J1358p6522's imported outputs shown as in the mockup, with
  Q6.12's "at limit", and no continuum coefficient flagged (Q6.23); a continuum made
  negative at a pixel flagged "negative"; Commit… asks for a description, which a double-click edits; a
  committed run survives Save; Use best fit on a card and in the Continuum box, each
  one step; Reset FWHM after Continue from best fit, as one step, on a row added from
  a file and on an imported one; on `examples/blind`, no hidden value in the table,
  its tooltips, the cards or what Copy as table copies.

**6.7 — Compare (`compare.py`, `qt/fit.py`; D41; Q6.9).**
- The two runs, Swap, Restore; the matched quantities with Δ and |Δ|/σ; the two models
  on the same data, with their difference.
- **Check** (unit and `gui` tests): metal_line_abs fitted with T fixed at 8000 K and
  with T free, matched component by component; fitted with O I's column density in
  linear form (`logN=False`, as `fit_spectra_linear.mod` writes it, made in the test),
  giving the same log N(O I) row; J1358p6522 fitted with D I's own column density and
  with D I as `2H_I/1H_I` (made in the test) gives one log D/H row; a blinded D I
  shows only |Δ|/σ; Restore is one step, and one undo restores the model and its
  snips.

**6.8 — Convergence, restarts and outliers (S21, S29, D27; Q6.10, Q6.11, Q6.18,
Q6.22).**
- Convergence check…; Random restarts…, their assessment, what is kept, their window
  and its chart, and Export restarts…; Review clipped pixels… and Apply.
- **Check** (`gui` and `fast` tests): metal_line_abs's restarts all agree; the starts
  are drawn about the best fit, within ±`sim maxperturb` and the limits; their table
  is kept with the run, committed with it, and survives Save; Export restarts… writes
  `sim perturb`'s form, which `numpy.loadtxt` reads, with its header and χ² column,
  and leaves a hidden parameter's columns out; an outlier added to its data is marked
  at 5σ and left out by Apply, as one step that undo reverses; three outliers in a row
  are shown unticked; ALIS's convergence check's result shown.

**6.9 — Markers, actions, preferences, command-line runs (F12, S31; Q5.8, Q6.14).**
- The Fit marker's states; the actions and preferences; a run made by `run_alis
  project.model`, open and closed, and a run kept running after the dashboard closed;
  the Regions tab's best fit.
- **Check** (`gui` tests): `run_alis project.model` while the project is open shows a
  "cmd" run without asking; a run kept running when the dashboard closes is shown when
  the project is next opened; the marker in each state; the Regions tab's switch draws
  the best fit.

**6.10 — From a spectrum to a fit, in the dashboard (end to end).**
- The `fast` test of Task 5.10, with the last step on the Fit tab instead of
  `run_alis`: Run, the best fit agreeing with `fit_spectra.mod.out.reference` within
  the errors; Commit…; the temperature changed to free on the Components tab and Run
  again; Compare of the two runs; Continue from best fit, and Use best fit on one
  card.

**6.11 — Close the stage and v1 (Q6.13).**
- Publish the screenshots of the tab beside the mockups for RJC's review, and apply
  the review.
- `doc/ALIS_workflow.md`: a section on the Fit tab, and §4.3 brought up to date for
  v1. Also update `CHANGELOG.md`, `tests/README.md`, the `gui-component` skill, and
  `deferred_work.md` (§7.2 done).
- Deprecate `prepfit` as D4 plans: it keeps working, and prints a note that the
  dashboard (`alis`) now covers its features (Q6.13(b)).
- Run `test-coverage` on the new modules, and the `unit`, `gui` and `fast` batches.
- Record the stage's decisions in `ALIS_v2_dashboard_prompts.md`, mark v1 complete in
  the stage table of `dashboard_stage0.md`, and list what follows v1: Stage 7, then the
  Plot tab and the items marked "Later" (Q6.13(a)). The release itself is RJC's
  (Q6.13(c)).
- Write `dashboard_stage7.md` as a template (Q6.13): the usual sections, with a list
  for RJC's changes and bug fixes, each of which becomes a task with its check.

## Skills to use for this stage

- `gui-component`: the Fit tab's sub-tabs, wired to the project, the history, the
  validator and the gate; it documents the spectrum view, the model and the ticks.
- `gui-dev`: driving the window off-screen, a run included, and taking the screenshots
  at 1440×900.
- `check-fit`: reading `.mod.out` files, the fit-quality numbers, and the end-to-end
  fit against its reference (6.3, 6.10).
- `convergence-check`: the random restarts and their assessment (6.8).
- `run-tests`: the `unit`, dashboard and `gui` tests after each task; the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of the new modules.
- Claude Code's `artifact-design` skill: the review pages (6.5, 6.11).

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`:
  - D8, D9, D24–D27, D34, D38–D41, D43, D44, D48, D49, D51, D60;
  - QF.11, QF.20, QF.26(c) and QF.32;
  - F6, F7, F9, F12, S17–S21, S29 and S31.
- `claude_prompts/dashboard_stage0.md`: Q0.4 (the runs test), Q0.9 and Q0.11
  (blinded changes), and the reviews of the Fit drafts (Task 0.10).
- `claude_prompts/dashboard_stage1.md`: the bundle's `runs/` members (Q1.7) and what
  `run.json` records (Task 1.7).
- `claude_prompts/dashboard_stage3.md`: Q3.8 (a bundle changed on disk), the Fit
  marker, and the frames of the Fit tab (`qt/tabs.py`).
- `claude_prompts/dashboard_stage5.md`: "What Stage 6 receives".
- `claude_prompts/deferred_work.md`: §7.2 (the best fit on the Regions tab), §7.5
  (generating data) and §7.6 (simulated realisations and random restarts).
- `doc/ALIS_workflow.md`: §2.1 (the settings), §4.1.2 (starting b), §4.2 (bundles),
  §5 (the outputs) and §6 (convergence).
- **The code:**
  - `alis/bundle.py` (`run`, `_run_info`, `_set_latest`, `unsupported`, `update`,
    `read`), `alis/outputs.py` (`MemoryOutputs`) and `alis/scripts/run_alis.py`, for
    a run and its split (Q6.19(a));
  - `alis/main.py` (`ClassMain`, the end of a fit, the interrupted errors) and
    `alis/minimise.py` (`iterfunct`, `defiter`, the `SIGQUIT` handler, status −20),
    for the progress, the per-iteration callback (Q6.19(b)) and Stop;
  - `alis/config.py` (`ArgFlag`), for the settings;
  - `alis/report.py`, `alis/convergence.py` (`assess_restarts`), `alis/simulate.py`
    (`perturb`) and `alis/save.py` (`save_model`, `save_covar`), for what a run
    writes, and the restarts;
  - `alis/dashboard/` (`profile.py`, `components.py`, `continuum.py`, `datasets.py`
    with a row's starting FWHM, `markers.py`, `session.py` with `take_runs`, `save`
    and the recovery copies, `blinding.Gate.best_fit` and `sigma_change`,
    `project.mark_hidden`, `validate.py`) and `alis/dashboard/qt/` (`tabs.py`,
    `plots.py`, `components.py`, `regions.py`, `data.py`, `window.py`).
- **The mockups:** `doc/dashboard/mockups/fit_inspect.html`, `fit_results.html` and
  `fit_compare.html` (built by `build_mockups.py`), published at
  https://claude.ai/artifact/J43w9ERNESDo9hez9o918B.
- **The data:** `examples/metal_line_abs/` (a quick fit, with turbulent, thermal and
  linear models and their references; the end-to-end fit),
  `context/fitting_examples/VMP_DLA/J1358p6522/` (an imported fit with its `.mod.out`,
  covariance and report; its D I, and the `2H_I/1H_I` parametrisation of
  `context/fitting_examples/DH/J1358p6522/`), `context/fitting_examples/DH/Q1243p307`
  (three datasets, with reference outputs) and `examples/blind` (a hidden line).

## Queries

*Raised by Claude on 2026-10-15, while writing this document. Each gives Claude's
lean, and the Design section and the tasks follow the leans.*

**Q6.1 — Reviews.** As in Stages 4 and 5, I propose to publish the tab's screenshots
for your comments once running and the Inspect sub-tab are built (Task 6.5), before
Results and Compare, and again at the close.

My lean: yes, a review at Task 6.5, and another at the close.

**Response:** I agree with your lean. Publishing the tab's screenshots for review at Task 6.5 will allow for early feedback and adjustments before proceeding to Results and Compare. A second review at the close will ensure that all changes and implementations are satisfactory and meet the desired standards.

**Q6.2 — Where the dashboard's runs run, and where their results go.**
- **(a)** Run writes the project as it stands (unsaved changes included) to a
  temporary bundle and runs `run_alis` on it in a separate process; the result is
  stored in the project's file at once (its runs only, as `run_alis project.model`
  stores them), so that a finished fit is never lost, while the model on disk
  changes only with Save; or
- **(b)** Run saves the project first, then runs `run_alis project.model` on the
  file itself.

(b) is simpler, but makes Run a Save, which D48 keeps as the only way the model is
written. With (a), closing the dashboard during a run can offer to keep the run
going: its result is stored in the project's file, and shown when the project is
next opened (S31), which covers the common case of F6's re-attaching, without the
live progress.

My lean: (a), with the offer to keep a run going when the dashboard closes.

**Response:** Neither of the proposed approaches is ideal, because it requires a `.model` file to be
written to disk. A better approach would be to pass the information directly to ALIS by calling a wrapper without writing the project to disk, and instead pass the necessary data directly to the process. This would avoid making Run a Save operation and allow for more flexibility in handling unsaved changes. There should be an option to "Save and Run", or an option to just "Run" without saving.

**Q6.3 — Stop.** ALIS already stops a fit cleanly on `SIGQUIT` (`Ctrl+\` in a
terminal): the minimiser ends its iteration, ALIS writes its outputs, and the run is
marked "interrupted", with its errors set to its values. I propose:
- **Stop** sends that signal, and keeps the run as "interrupted": its values can be
  inspected and continued from, and its errors are not shown (they are the values,
  so they would show a blinded one);
- pressing it again, **Cancel**, ends the process at once, and keeps nothing.

My lean: as proposed.

**Response:** The original goal of this functionality was to allow users to stop a fit cleanly and inspect the results (i.e. a plot of the current model on top of the data) before deciding whether to continue or cancel. If there was a way to allow users to inspect the results (e.g. plot the best-fitting model on the data and display it on the dashboard) during the fit, that would be even better. If this is implemented, there would need to be a warning or information message on the dashboard to alert the user that the model currently shown is being updated at each iteration to show the current status of the fit.

**Q6.4 — Settings (F9).**
- **(a)** The settings that describe the computer (`run backend`, `ncpus`, `ngpus`)
  are preferences of this computer, given to each run on its command line, so that a
  model fitted on a laptop runs unchanged on a GPU machine; every other setting is the
  model's, a `par` line, set in the Run box or in All settings… as one step.
- **(b)** The dashboard asks ALIS, on the command line, for what the tab reads (the
  covariance and the report) and for `out verbose 1`, and never for plot windows; none
  of this is written in the model.
- **(c)** Settings a bundle cannot run are not offered.

My lean: (a), (b) and (c).

**Response:** These are all reasonable approaches, provided that all current functionality is maintained if the user requests. Although many people will use the dashboard for their fits, we need to make sure that the command line functionality is still available for those who want to run ALIS on the command line, or run multiple fits on a headless supercomputer.

**Q6.5 — Initial and Final.** Step (6) asks for the initial or the final model on the
data. I propose that Initial is the model as it stands now (the one the next run
starts from, as the Components tab draws it), and Final the shown run's best fit,
drawn by `profile.py` from the run's `.mod.out` on the snips as they are now. The
alternative is that Initial is the shown run's own starting model, which differs from
the current one once the model is edited after the run.

My lean: as proposed, since the toggle is then the way to see what an edit changed
since the last fit.

**Response:** Agreed.

**Q6.6 — Changes to ALIS outside `alis/dashboard/`.** Two small ones:
- **(a)** `ArgFlag`'s settings have defaults and types but no descriptions, so All
  settings… would have to keep its own. I propose a one-line description of each
  setting as metadata of its field in `alis/config.py`, printed by `run_alis
  --list-settings` as well, with a unit test that every setting has one.
- **(b)** `alis/report.py`'s statistics (the runs-test z and the χ²ν deviation) are
  private functions of an ALIS instance. I propose to make them public functions of a
  snip's residuals, unchanged in what they compute, so that the tab's marks are
  computed exactly as ALIS's report computes them, for every run, an imported one or
  a run without `out report` included.

Both are checked by the refactor's regression harness and change no fit.

My lean: (a) and (b).

**Response:** Agreed.

**Q6.7 — The run history.** D26 keeps a run only when it is committed. I propose:
- **(a)** a committed run keeps its `run.json` (with the description), its input
  model, its outputs and its snips as they were, so that its model can be restored;
  each snip's content is stored once, under `runs/files/<sha256>`, and shared by every
  run that has it, so that ten committed runs of one project do not hold ten copies of
  its snips;
- **(b)** committing, describing and removing a committed run are saved with Save,
  like the model, but are not steps of the undo history, since they do not change the
  model; removing one asks first;
- **(c)** the runs are numbered in the order they are committed, and a number is never
  reused.

My lean: (a), (b) and (c).

**Response:** Agreed.

**Q6.8 — Continue from best fit (S20).** The model may have changed since the run. I
propose to copy the best-fit values only into lines unchanged since the run (matched
to the run's input model), after a preview that lists the lines changed and the
lines skipped, as one step; to leave linked parameters to ALIS; and to allow it after
an interrupted run, so that a stopped fit can be resumed.

My lean: as proposed.

**Response:** I agree that we should allow a stopped fit to resume. It would be helpful to have a functionality to copy the best-fit parameter values into the model, while skipping any parameters that have been changed since the run. This would allow users to continue fitting from the best-fit values without losing any changes they have made to the model. It may also be useful to copy individual component values (maybe a button on the component cards that is enabled whena best-fit value is available) so that users can selectively copy values from the best-fit model to their current model. This would also be helpful if we could do this for the definition of the initial starting values of the continuum as well, so that users can copy just the best-fit continuum values into their current model instead of manually adjusting the continuum in the regions tab once a best-fit model is available.

**Q6.9 — Matching quantities in Compare (D41).** I propose:
- **(a)** the quantities: each ion's total log N in each system; each isotope's ratio
  to its element (log D/H), from the two column densities or from a ratio line or
  linked `variable`; each component's v, b and T, matched by system, ion and the
  nearest velocity within 10 km/s (a preference); the FWHM, χ²ν, degrees of freedom
  and iterations;
- **(b)** errors of totals and ratios from each run's covariance, and "—" when a run
  has none;
- **(c)** |Δ|/σ with σ the larger of the two runs' errors. The two runs fit the same
  data, so their errors are not independent, and adding them in quadrature would
  understate a change.

My lean: (a), (b) and (c).

**Response:** Agreed.

**Q6.10 — Random restarts (S21).** A bundle refuses `sim perturb` (it writes its
fits to a `sims/` directory on disk). I propose that the dashboard draws the starts
itself, as `simulate.perturb` draws them (from the covariance, about the best fit,
within the limits), runs each as an ordinary run, one after another, and assesses
them with `convergence.assess_restarts`'s tests, showing only |Δ|/σ for each free
parameter. Ten restarts by default (a preference). The alternative, making `sim
perturb` run inside a bundle, changes ALIS outside the dashboard for a feature that
the dashboard can drive with what exists.

My lean: as proposed, with no change to ALIS.

**Response:** It would be good for the user to have access to the `sim perturb` values. The user may wish to plot their own diagnostics (e.g. chi-squared vs a parameter of interest) to see which fit produces the least chi-squared, or to see if the fits are converging to a single solution. It would be good to have a way to export the values of the random restarts, so that the user can plot them and analyze them further. Is there a way to store an array of parameter values for each random restart, so that the user can plot them and analyze them further? This would be useful for users who want to see how the random restarts are behaving and whether they are converging to a single solution.

**Q6.11 — Outliers (S29, D27).** I propose a threshold of 4σ (a preference): with
4,000 fitted pixels, 3σ would mark about eleven by chance alone. Lone pixels and
pairs are ticked; three or more in a row are shown but unticked, since they may be
absorption the model misses. Accepted pixels are left out of the fit as Tweak dataset
regions leaves pixels out (Stage 4), as one step.

My lean: as proposed.

**Response:** I propose 5 sigma, but this should be a preference that the user can set. The user should also have the option to accept/reject the proposed outlier rejection.

**Q6.12 — "At limit" and "unconstrained".** I propose:
- **at limit:** the value closer to a bound (a `lim` line, or voigt.py's own limits)
  than its error, or equal to it when the error is zero;
- **unconstrained:** the error larger than the value for b and T, larger than 1 dex
  for log N, larger than 10 km/s for a redshift (as velocity), and larger than the
  value for the other parameters; each threshold a preference. A parameter at a
  limit is flagged "at limit" only.

These give the mockup's Lyα forest rows from J1358p6522's reference (b_turb = 1.09 ±
10.78, unconstrained; 0.29 ± 60.53, at its limit of 0.2).

My lean: as proposed.

**Response:** I agree with this, although this is just a labelling that the dashboard indicates as a value that might need to be inspected with detail. The proposed message sounds good to me, although I'd prefer "at limit" to be restricted to cases where the value is equal to the limit. The proposed description of "unconstrained" is acceptable.

**Q6.13 — v1.** This stage completes v1 (D33). At its close, I propose:
- **(a)** to mark v1 complete in the stage table of `dashboard_stage0.md`, and to list
  what follows (the Plot tab first, then the items marked "Later"), for you to order;
- **(b)** to deprecate `prepfit` as D4 plans: it keeps working, and prints a note
  that the dashboard (`alis`) now covers its features;
- **(c)** to leave the release itself (the version number, the tag, publishing) to
  you.

My lean: (a), (b) and (c).

**Response:** I intend to generate a seventh stage of this development, which is based on user feedback from new real world problems. Stage 7 will be a a relatively simple stage document that needs to allow me to provide a list of changes and bug fixes that are needed to be implemented in the dashboard. At the end of Stage 6, we will generate the Stage 7 design document together (Claude will construct the template, and I will provide the list of changes and bug fixes as a bullet point list).

*Raised by Claude on 2026-10-18 (Prompt 1), after reading the responses above. How I
read each response comes first, then six new queries, each with a lean. The Design
section and the tasks still read as they were written, and will be brought in line
once these are answered.*

**Notes on the responses.**
- **Q6.1.** A review at Task 6.5 (running and Inspect), and another at the close.
- **Q6.2.** No project file is written for a run, and there are two actions, Run and
  Save and Run. How the project reaches the process, and where the result goes, is
  Q6.14.
- **Q6.3.** You would like to see the fit while it runs, not only after stopping it:
  Q6.15.
- **Q6.4.** Every current function stays available, and the command line runs the
  project as it is written, on a laptop or a headless machine. How that changes (b)
  and (c) is Q6.16.
- **Q6.5.** As proposed: Initial is the model as it stands now; Final is the shown
  run's best fit.
- **Q6.6.** (a) and (b). Your responses to Q6.2 and Q6.3 need two more small changes
  to ALIS: Q6.19.
- **Q6.7.** As proposed.
- **Q6.8.** Continue from best fit as proposed, and an interrupted run can be
  continued from. Copying the best fit of one component, or of one continuum, is
  Q6.17.
- **Q6.9.** As proposed.
- **Q6.10.** Yes, the restarts' values can be kept and exported: Q6.18.
- **Q6.11.** 5σ, a preference. Accepting and rejecting are as designed: each panel's
  Accept all and Reject all, a click on one pixel to tick or untick it, and nothing
  leaves the fit until Apply, which is one step that undo reverses.
- **Q6.12.** "At limit" only when the value equals a bound. The minimiser stops a
  parameter exactly on its bound, so the test is equality to the precision of the
  run's values. J1358p6522's b_turb of 0.29 ± 60.53 (its bound 0.2) then reads
  "unconstrained", not "at limit" as the mockup shows it. "Unconstrained" is as
  proposed.
- **Q6.13.** At Task 6.11, I write `dashboard_stage7.md` as a template: the usual
  sections, with a list for your changes and bug fixes, each of which becomes a task
  with its check. You then fill in the list. (b) and (c) had no response, so I will
  follow the lean (`prepfit` keeps working and prints a note that `alis` covers its
  features; the release is yours) unless you say otherwise.

**Q6.14 — Running without writing the project to disk (your Q6.2 response).** I
propose:
- **(a) How the project reaches ALIS.** The fit still runs in a separate process, so
  that the window stays responsive and Stop works. The dashboard hands it the project
  as it stands, unsaved changes included, through a pipe: the bytes that Save would
  write, but never written to a file. The process runs ALIS on them as `run_alis
  project.model` does, all in memory (the data, the atomic table, and the hidden lines
  restored in memory only). It sends back its progress and, at the end, its outputs.
  This needs `bundle.run` split in two (Q6.19(a)).
- **(b) Where the result goes.** The options:
  - **(i)** into the project's file at once, under its lock: its `runs/latest/` only,
    as `run_alis project.model` stores a run. The model on disk changes only with
    Save, so Run is not a Save. A finished fit is never lost, and a run can be kept
    going when the dashboard is closed (Q6.2's offer); its result is then shown when
    the project is next opened;
  - **(ii)** into the project in memory, and its recovery copy, written to the file
    with the next Save. Closing the dashboard during a run then offers only Stop or
    Cancel.
- **(c) Save and Run** (Ctrl+Shift+R, beside Run's Ctrl+R) saves the project, then
  makes the same run, so that the model on disk is the one fitted, and a later `run_alis
  project.model` fits the same model.

My lean: (a); (b)(i), with (ii) for an imported fit that has no file until Save as;
(c) as described.

**Response:** Agreed.

**Q6.15 — Watching the fit as it runs (your Q6.3 response).** I propose:
- **(a) The fit in progress.** After each iteration, the process sends its parameter
  values through the pipe (they are never printed). While the run goes, Inspect's
  Final draws ALIS's model for those values (`profile.py`), with a notice above the
  panels: "The fit in progress: the model is redrawn after each iteration (iteration
  9, χ² 3796.2)". Results shows the values in progress in the same way, through the
  gate, with no errors (ALIS computes them only at the end). Blinded values are hidden
  as for a finished run, and their profiles are drawn (D24). Drawing is throttled
  (J1358p6522's 12 snips take 30 ms, Q1243p307's 51 take 230 ms), so a fast fit skips
  iterations rather than falling behind.
- **(b) Stop and Cancel** as in Q6.3. Stop ends the fit at its current iteration, and
  keeps it as "interrupted", to inspect and to continue from (Continue from best fit,
  then Run). Cancel ends the process at once and keeps nothing.
- **(c) A Pause** that freezes the process and later resumes the same fit is possible
  on macOS and Linux. I do not propose it: with (a) the fit can be inspected without
  stopping it, and Stop then Continue from best fit resumes from where it stopped.

Getting the values out at each iteration needs a small hook in ALIS's minimiser
(Q6.19(b)).

My lean: (a) and (b), with no Pause.

**Response:** Agreed.

**Q6.16 — Keeping every setting available (your Q6.4 response).** I read "all current
functionality is maintained if the user requests" as two rules:
- **(a)** The dashboard's choices for its own runs are given on the run's command line
  only, never written in the model: `out verbose 1`, no plot windows, the covariance
  and the report, and this computer's backend, CPUs and GPUs. The model keeps what the
  user wrote. `run_alis` on the project, or on its exported plain files on another
  machine, runs it exactly as written, plot windows included.
- **(b)** All settings… offers every setting of `ArgFlag`:
  - the computer's settings too. Written into the model when the user wants them
    there, they then win over this computer's preferences (the Run box says so);
  - the settings a bundle cannot run (`sim random`, `sim perturb`, `sim beginfrom`,
    `iterate model`, `generate data`). They are written like any other setting and
    marked "command line only". While one is set, the pre-flight check refuses Run,
    and says to export plain files (File → Export plain files…) and run `run_alis`
    on them. This replaces Q6.4(c)'s "not offered".

My lean: (a) and (b).

**Response:** Agreed. Please make a note that generating data is to be implemented as future work.

**Q6.17 — The best fit of one component, or one continuum (your Q6.8 response).** I
propose:
- **(a) Where.** Each component card gets "Use best fit". It copies the shown run's
  best-fit z, b_turb, T and each ion's log N of that component into its lines, as one
  step. The Regions tab's Continuum box gets the same button for the snip's continuum
  (its coefficients). The shown run is the latest, or the one chosen in the run
  history, as on Inspect and Results.
- **(b) Lines changed since the run.** Continue from best fit skips lines changed since
  the run (Q6.8), since it copies everything at once. "Use best fit" is asked for one
  item, so it copies even into lines changed since the run, provided the run has that
  item. A component is matched by its redshift label, and a continuum by its snip's
  specid and its function and order. A continuum whose order has changed since the run
  cannot take the run's coefficients, so the button is disabled, and its tooltip says
  why.
- **(c) Blinding**, as for Continue from best fit. A hidden line stays hidden, and
  under global blind every line that receives a value is hidden. The card shows the
  mask, and the step's name holds no value.
- **(d)** The Data tab's FWHM, shift and zero level are left to Continue from best fit.

My lean: (a)–(d) as proposed.

**Response:** (a) Agreed. (b) Agreed, but for the continuum, if the order is increased, the coefficients of the lower order terms can be copied, and the higher order terms can be set to zero. If reducing the order, and the final coefficient is zero, then the highest order coefficient can be removed and the rest of the terms still allowed. (c) Agreed. (d) Agreed. But, we need to allow an option to "return to default" for the FWHM.

**Q6.18 — The random restarts' values (your Q6.10 response).** I propose:
- **(a) What is kept.** Each restart's starting values, best-fit values and errors, its
  χ², degrees of freedom, iterations and status are kept with the run the restarts
  started from, and committed with it. They form a table with one row per restart and
  one column per parameter, named by its ion, component and parameter (for example
  "O I a1 log N"), with its label.
- **(b) Export and a first look.** "Export restarts…" writes the table as text, in the
  form `sim perturb` writes its own (`<model>.perturb_0-10`, one row per fit, the first
  row the starting values), with a header naming the columns and a χ² column, so that
  a script that reads ALIS's file reads this one too. The restarts' window shows the
  assessment table, and χ² against a parameter chosen in a menu.
- **(c) Blinding.** A hidden parameter's values are neither exported nor plotted: its
  columns are left out, and the header says so. Its |Δ|/σ appears in the assessment,
  unsigned. Unblinding brings its values back.
- **(d) Where the starts are drawn.** While checking how `sim perturb` draws its starts
  (`simulate.perturb`), I found that:
  - it draws them about the model's *starting* values (`p0`), not about the best fit
    (`bparams` is passed to it, but not used);
  - it clamps each draw to ±10% of the value (`sim maxperturb`).

  Q6.10 says "about the best fit", as `sim newstart` and §6 of `ALIS_workflow.md`
  describe it.

My lean: (a)–(c) as proposed. (d): the dashboard draws about the best fit, with
`sim maxperturb`'s clamp, and `simulate.perturb` is left as it is in this stage. If
its centre is a fault, I would record it in `deferred_work.md` for you to decide.

**Response:** (a)-(c) Agreed. (d) How we approach the simulated realisations requires more thought. Please add this to deferred work for now.

**Q6.19 — The changes to ALIS that Q6.14 and Q6.15 need.** Besides Q6.6's two:
- **(a)** `bundle.run` is split in two: running a bundle held in memory, which gives its
  outputs and `run.json`; and storing them in the project's file. `run_alis
  project.model` does both, as now. The dashboard's process runs the project it is
  handed through the pipe with the first, and stores the result with the second
  (Q6.14(b)).
- **(b)** `ClassMain` gains an optional per-iteration callback, which it passes to the
  minimiser. It is called after ALIS's own printout of each iteration, with the
  iteration, χ² and the parameter values. It is `None` by default, so command-line
  runs do not change.

Neither changes a fit; the regression harness and its `test_bundle` mode check them.
The alternative is to patch ALIS from inside the dashboard's process. That keeps ALIS
unchanged, but it would break silently whenever ALIS's internals change.

My lean: (a) and (b).

**Response:** Agreed.

*Raised by Claude on 2026-10-18 (Prompt 2), after reading the responses to
Q6.14–Q6.19. The Design section and the tasks now follow every response, and the
leans of the three queries below.*

**Notes on the responses.**
- **Q6.14, Q6.15, Q6.19.** As proposed.
- **Q6.16.** As proposed. Generating data is recorded as future work in
  `deferred_work.md` (§7.5).
- **Q6.17.** (a) and (c) as proposed. For (b), the continuum's order rule leaves one
  question: Q6.20. For (d), the FWHM's "return to default": Q6.21.
- **Q6.18.** (a)–(c) as proposed. (d) is recorded in `deferred_work.md` (§7.6), with
  one more fact I found (below). What this stage's random restarts do meanwhile is
  Q6.22.

**Q6.20 — "Use best fit" for a continuum whose order has changed (your Q6.17(b)
response).** A continuum is a polynomial (Legendre, Chebyshev, polynomial or
constant), so I propose:
- **(a)** to carry over the run's best-fit continuum as a curve. It is written in the
  line's current function, range (`min=`/`max=`) and `scale=` factors, and this is
  exact whenever the line's order is at least the curve's degree. A raised order
  then gets the run's coefficients with zeros above them, as you say. A continuum
  whose range or function has changed since the run (a snip re-cut, a Legendre made a
  Chebyshev) also takes the run's continuum exactly.
- **(b)** For a lowered order, your rule copies when the run's coefficients above the
  new order are zero. A fitted coefficient is almost never exactly zero, so what
  should "zero" mean?
  - **(i)** exactly zero, to the precision of the run's values, as when the run held
    it fixed at zero;
  - **(ii)** zero within its error (|c| ≤ σ, from the run's covariance; exactly zero
    for a run with no errors, such as an interrupted one);
  - **(iii)** any value. The copy is always allowed, and the preview says how many σ
    each dropped coefficient is from zero.

  For (i) and (ii), the button is otherwise disabled, and its tooltip says which
  coefficient is how many σ from zero (unsigned, so that it reveals no value).

My lean: (a), and (b)(ii). A coefficient within its error of zero is the usual reason
for lowering the order, and dropping it moves the starting continuum by no more than
about the fit's own uncertainty.

**Response:** Perhaps a general rule that works is to fit the new continuum parameters (regardless of whether the order has increased, decreased, or stayed the same) to the best-fit continuum from the previous run. This would allow for a smooth transition between different orders and ensure that the new continuum is consistent with the previous best-fit. The user should be able to see how well the new continuum fits the previous best-fit and make adjustments as necessary.

**Q6.21 — "Return to default" for the FWHM (your Q6.17(d) response).** I propose
"Reset FWHM to 7 km/s" in the context menu of each FWHM cell on the Data tab, shown
also as ↺ in the cell while the FWHM differs from its default. It is one step:
- **the default** is the FWHM given when the file was added (the "Add a spectrum"
  dialog's, which the project already keeps with the row in `ui/project.json`). A row
  of an imported fit has none, so the FWHM it had when imported is recorded with it at
  import, and is its default;
- only the value is reset. Free or fixed is left as it is (the cell's badge sets it),
  and rows tied to it share it, as they share any value of the FWHM.

My lean: as proposed.

**Response:** Agreed.

**Q6.22 — The dashboard's random restarts in this stage (your Q6.18(d) response).**
While writing §7.6 of `deferred_work.md`, I found that ALIS reaches `sim perturb` by
two routes, which draw about different centres:
- `sim perturb 10` in the model of a fit draws about that run's *starting* values,
  with the best fit's covariance;
- `sim perturb 10` with `sim beginfrom <fit>` draws about that fit's *best fit* (its
  `.mod.out`), with its `.covar`. This is what `ALIS_workflow.md` §6 and Q6.10
  describe.

What should Random restarts… do in this stage?
- **(a)** Build it, drawing as `sim perturb` draws after `sim beginfrom` the shown
  run: about its best fit, from its covariance, each draw clamped to ±`sim
  maxperturb` of its value and then to its limits, with Q6.18(a)–(c). ALIS is not
  changed, and the dashboard follows whatever §7.6 later decides; or
- **(b)** leave Random restarts… and Export restarts… until §7.6 is decided, after v1.
  ALIS's own convergence check (`run convergence`) stays in this stage.

My lean: (a). It is ALIS's existing behaviour for a restart from a finished fit, and
keeps what Q6.18(a)–(c) agreed.

**Response:** I agree with option (a).

*Raised by Claude on 2026-10-18 (Prompt 3), while building Task 6.3. Development
goes on meanwhile, with the rule as agreed; the answer is applied in Task 6.6.*

**Q6.23 — "Unconstrained" continuum coefficients (Q6.12).** With the rule agreed in
Q6.12 ("the error larger than the value for the other parameters"), 15 of
J1358p6522's 54 continuum coefficients are flagged "unconstrained", such as H7's
c2, 2.02 ± 7.29 (`scale=0.01`). A Legendre's higher terms are often consistent with
zero, which is no fault of the fit: it is the usual reason to try a lower order,
which Use best fit in the Continuum box (Q6.20) and the Regions tab's table of
orders already help with. The flags then hide the rows that matter among the
continua's collapsed group, whose flagged rows are shown.
- **(a)** leave continuum coefficients (and zero levels) unflagged by
  "unconstrained", flagging them only "at limit"; or
- **(b)** keep the rule for them, as now.

My lean: (a).

**Response:** Continuum coefficients should not be flagged as "unconstrained" or "at limit". The key warning for a continuum is if any pixels of the continuum are negative. The final model continuum can have zero values ("at limit") or positive values, but never negative values. Just to be clear, the coefficients can be negative, but the actual final continuum model generated from the coefficients cannot be negative anywhere.

*Raised by Claude on 2026-10-19 (Prompt 5), while taking the screenshots of Task
6.5.*

**Notes on the response.**
- **Q6.23.** A continuum's coefficients are flagged neither "unconstrained" nor "at
  limit". Instead, the continuum is flagged "negative" when its best fit is below
  zero at any pixel of its snip; its coefficients may be negative. This is applied
  in Task 6.6, and the Design ("Results") and that task's check now say so. I read
  "continuum coefficients" as the continuum lines' parameters (the polynomial and
  its `scale=` factors). The zero levels (the Data tab's `constant` of a dataset)
  keep the general rule. Say if they should go with the continua.

**Q6.24 — `run ncpus -1` given on ALIS's command line.** ALIS counts a negative
number of CPUs ("all bar one" for −1) when it reads a model (`load.cpucheck`), but
not when the setting is given on its command line (`load.reapply_cli_overrides`
sets it as written). So `run_alis fit_spectra.mod --set "run ncpus -1"` stops with
a traceback ("Number of processes must be at least 1"). I found this because the
dashboard's runs were given this computer's CPUs preference as written (−1 by
default), so a model that does not set `run ncpus` could not be fitted from the
dashboard. The dashboard now gives the number itself (`settings.cpus`, with ALIS's
own count), and refuses Run when the count leaves no CPU. That is fixed and tested
(Task 6.5's log). For ALIS itself:
- **(a)** count the CPUs again after the command-line settings are applied, in
  `reapply_cli_overrides`, with a unit test and the regression harness; or
- **(b)** leave ALIS as it is, and record it in `deferred_work.md`.

My lean: (a). It is a one-line change to ALIS outside the dashboard, so I have not
made it without your agreement.

**Response:** Agreed, but we want to avoid users having to know what their CPU count is. The benefit of quoting a negative number is that users can use all CPUs except for a certain number. We just need to make sure that the negative number isn't greater than the number of CPUs available! Is there a way to continue supporting negative numbers for CPU counts? If so, we should support this, to retain ALIS' previous functionality. If not, we need to make it clear to users what their CPU count is. I prefer the negative numbers approach.

*Raised by Claude on 2026-10-19, answering RJC's comment on the review page of Task
6.5 ("Does the runs value account for masked pixels?").*

**Q6.25 — The runs test across the gaps between fit regions.** The marks count only
the pixels inside the green fit regions that ALIS fits, as ALIS's `.report` does
(`report._regions`, and `results.residuals` for the dashboard). Pixels left out of
the fit and bad pixels are not counted: Ly8's 1091 pixels are all inside its 15
regions. Within a snip, though, both take those pixels as one sequence, so a run of
same-signed residuals can continue across the gap between two regions, although
the pixels at either side of a gap are not neighbours. Counting each region's runs
on its own (a new run at each region) gives, for J1358p6522's best fit:

| Snip | Regions | Joined (now) | Broken at the gaps |
|------|---------|--------------|--------------------|
| H1 (Ly8) | 15 | −9.60 | −9.05 |
| H4 (Lyε) | 10 | −2.79 | −2.09 |
| H6 (Lyγ) | 14 | −4.92 | −4.08 |
| H7 (Lyβ) | 23 | −6.05 | −4.74 |
| H8 (Lyα) | 40 | −5.22 | −4.04 |
| O1–O4 | 1–2 | unchanged | unchanged |

No snip's mark changes here (✕ stays ✕, H4's ! stays !), but snips cut into many
regions read more correlated than they are.
- **(a)** break the runs at the gaps, in ALIS's report and in the dashboard alike
  (the expected number of runs and its spread taken over the regions together), so
  that the marks still equal the `.report`. This changes ALIS outside the
  dashboard, with the regression harness; or
- **(b)** keep the runs joined, as ALIS's report has them.

My lean: (a), since pixels on either side of a gap are not neighbours.

**Response:** I don't think there's any need to calculate the runs test for each separate region. Since pixels either side of a region gap are likely uncorrelated, it should be OK to stitch each region together and calculate a single runs value for the residuals in a given snip. We just need to make sure that the data excluded in the fit are excluded from the runs test, and it sounds like ALIS is already doing this correctly. If you think there is something more correct that we should be doing, please let me know, otherwise, the above approach seems reasonable.

*Raised by Claude on 2026-10-19, from RJC's comment on question 4 of the review
page ("Let's make the default to be `run blind False` (users should actively choose
to blind their data). A minimum number of chisq iterations of 10 is better, so let's
change the default there, too.").*

**Q6.26 — ALIS's defaults: `run blind False` and `chisq miniter 10`.** These are
ALIS's own defaults (`alis/config.py`, `ArgFlag`), outside the dashboard. The
comment came through the review page, and the change was not allowed from there, so
I have not made it. Please confirm here. What it would change:
- **`run blind`** (True → False). Every model the regression harness fits writes
  `run blind` itself, so none of them changes. A model that does not write it
  would no longer be blinded by default: ALIS would then write its `.mod.out`,
  `_fit.dat` and covariance, and print its values. The dashboard reads the
  default too (a model without the line would no longer be under global blind).
- **`chisq miniter`** (0 → 10). Six harness models leave it to the default. Four
  stopped after 10 or more iterations and are unchanged. Two would now iterate
  past where they stopped: helium34's Her36 (stopped at 9) and tet02OriA (at 7),
  both because their parameters had stopped changing (`xtol`), so their best fits
  should move little. Their `test_minimisation` cases, and the bundle mode, would
  check that against the references, which stay as they are.
- **Also:** `test_config.py` (which checks that no default changed), the
  CHANGELOG, `ALIS_workflow.md` §2.1 where it gives defaults, and the dashboard's
  tests that assume the old defaults. After the change, `run blind False` and
  `chisq miniter 10` no longer appear under "Settings you changed" for a model that
  writes them.

My lean: as you asked, with the harness's affected cases and the `fast` batch run
before it is kept.

**Response:** I agree that we should change the defaults to `run blind False`, and perhaps a better option is to set `chisq miniter 5`. My choice of 10 was arbitrary at best, so let's change this to 5 as the default, and it won't affect the harness. This will ensure that users are not surprised by their data being blinded by default, and that the minimization process has a reasonable number of iterations to converge. Please go ahead and make these changes in ALIS. I also agree with your final point. Let's make the necessary changes to the harness, the tests, and the documentation to reflect these new defaults.

**Notes on the responses (Prompt 6).**
- **Q6.24.** Negative counts stay. ALIS will count `run ncpus` given on its command
  line as it counts the model's own (−1 is every CPU bar one), in
  `reapply_cli_overrides`. A count that leaves no CPU (−20 on a computer with 8, or
  0) will fit with one CPU, with a warning, rather than stop. So nobody has to know
  their computer's count. The dashboard's pre-flight check notes it, where it
  refused Run, and the dashboard gives its runs the CPUs preference as it is
  (−1), since ALIS now counts it. Done (Prompt 6's log).
- **Q6.25.** Kept as it is, (b): one runs test per snip, over its fitted pixels
  joined across the gaps. Neither ALIS nor the dashboard changes. ALIS's report
  counts only the pixels it fits (those inside the fit regions, without bad pixels)
  that have a positive error (`report._regions`), and the dashboard counts the same
  ones (`results.counted`). Joining is sound: the test asks whether the residuals
  are independent, and if they are, the two pixels at a gap are one more
  independent pair, like any other. There is nothing more correct to do.
- **Q6.26.** `run blind False` and `chisq miniter 5`. The six harness models that
  leave `chisq miniter` to the default stopped at their 7th iteration or later, so
  none of them changes. A model without a `run blind` line is no longer under
  global blind, so deleting `run blind True` is refused, as switching it to False
  is. Done (Prompt 6's log); see Q6.27 for new projects.

*Raised by Claude on 2026-10-19 (Prompt 6), from RJC's response to Q6.26.*

**Q6.27 — New projects' `chisq miniter` (Q2.2).** A new project's model writes
`chisq miniter 10` (with `run blind False`, as you chose in Q2.2). Once ALIS's
default is 5:
- **(a)** new projects write `chisq miniter 5`, ALIS's default, so that no new
  project shows it under "Settings you changed" (what review question 4 wanted);
  or
- **(b)** they keep `chisq miniter 10`, shown there as a change from ALIS's default.

My lean: (a).

**Response:** Agreed, please implement (a).

**Note on the response.** A new project writes `chisq miniter 5`
(`preferences.NEW_PROJECT_SETTINGS`), and `run blind False` as before; neither
differs from ALIS's defaults now, so "Settings you changed" lists neither for a new
project.

*Raised by Claude on 2026-10-19 (Prompt 7), while building Task 6.6. Development
goes on meanwhile, with the rule as agreed.*

**Q6.28 — Which pixels "negative" looks at (Q6.23).** As agreed, a continuum is
flagged "negative" when its best fit is below zero at any pixel of its snip.
J1358p6522's Lyα snip (H8, 4750–5150 Å) is flagged: its Legendre continuum is below
zero at 563 of its 9,696 pixels, all at 5126–5150 Å, beyond its last fit region (its
fitted pixels end at 5056 Å). There the continuum is extrapolated and constrains
nothing in the fit, although Inspect draws the data divided by it, which then
diverges. The choices:
- **(a)** every pixel of the snip, as now (H8 is flagged);
- **(b)** only the fitted pixels (H8 is not);
- **(c)** from the snip's first fitted pixel to its last, gaps between regions
  included, where the continuum is interpolated rather than extrapolated (H8 is
  not).

My lean: (c). A continuum that dips below zero between two fit regions (in a
damped line's trough, say) is worth a look, while one that runs away beyond the
last region says only that the snip is cut wider than it is fitted.

*Raised by Claude on 2026-10-20, from RJC's answers on the closing review page
(https://claude.ai/artifact/Ea9QrHxSR1NThEEd4UDUmE).*

**Notes on the review's answers.**
- **Q6.28:** (c), "from the first fitted pixel to the last fitted pixel in a given
  snip". Done: J1358p6522's Lyα continuum is no longer flagged.
- **Random restarts:** one set is kept with a run; when one is kept, Run restarts
  warns that the new ones replace it, and asks first. Done.
- **Compare's default pair:** the latest run against the newest committed run, or,
  when the latest run is that committed run, against the committed run before it.
  This is the pair the sub-tab chose already; the rule is now written out and
  tested.

**Response:** Agreed, please implement (c).

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-5; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`, `dashboard_stage3.md`, `dashboard_stage4.md`, `dashboard_stage5.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`,  `dashboard_stage3_log.md`, `dashboard_stage4_log.md`, `dashboard_stage5_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. If everything is clear, then please notify me when you are ready to start implementing Stage 6, so that I can confirm before beginning.

2. Please read the responses to your queries in the Queries section of this document, and provide any further queries you may have. If everything is clear, then please notify me when you are ready to start implementing Stage 6.

3. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-5; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`, `dashboard_stage3.md`, `dashboard_stage4.md`, `dashboard_stage5.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`,  `dashboard_stage3_log.md`, `dashboard_stage4_log.md`, `dashboard_stage5_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. If everything is clear, then please notify me when you are ready to start implementing Stage 6, so that I can confirm before beginning. Please read the responses to your queries in the Queries section of this document, and provide any further queries you may have. If everything is clear, then please continue with the development, starting with Task 6.3.

4. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. If everything is clear, then please notify me when you are ready to start implementing Stage 6, so that I can confirm before beginning. Please read the responses to your queries in the Queries section of this document, and provide any further queries you may have. If everything is clear, then please continue with the development, starting with Task 6.4.

5. Please read this document, and then execute Task 6.5. I will then review the results and provide feedback. If you have any questions or need clarification, please ask in the Queries section of this document.

6. Please read this document, and then continue the development with my responses to your Q6.24, Q6.25, and Q6.26. If you have any questions or need clarification, please ask in the Queries section of this document if my responses to the above queries require further clarification. Once these are implemented, please let me know, and I will commit the changes before moving to Task 6.6. 

7. I have committed the changes, and everything looks good so far. Please complete the remaining development of Stage 6, starting with Task 6.6, and executing all tasks until they are complete. If you have any questions or need clarification, please please the development and ask for clarification in the Queries section of this document.

8. I have committed all files to GitHub. If there are no more tasks to develop on this stage, please close this stage and generate a design document for stage 7. The design document for stage 7 should be similar to the design documents for all other stages of development, but it should provide an additional section called `Feedback`. This feedback section will contain an enumerated list of all points I receive as feedback from users. The feedback section will be a collection of all feedback received from users, and it will be used to inform the development of future stages. Please ensure that the design document for stage 7 is clear, concise, and well-organized, and that it provides a comprehensive overview of the planned development for that stage. Once the design document for stage 7 is complete, please submit it for review and approval.
