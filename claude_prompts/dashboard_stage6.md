# Prompt file for ALIS software dashboard creation -- STAGE 6

> **The Fit tab (step 6). This stage completes v1.** The fourth tab of the skeleton is
> filled in, with its three sub-tabs, Inspect, Results and Compare (D34). At the end
> of the stage, a user can take a project from a spectrum to a fit that has been run,
> inspected, kept and compared, without leaving the dashboard:
> - **Running (F6, F9, S17):** the settings of the fit, generated from ALIS's own
>   list; a pre-flight check; Run, which fits in a separate process with the χ² of
>   each iteration drawn as it goes; Stop.
> - **Inspect (D38, D39, S18):** every snip, one ion or one snip, with the starting
>   model or the best fit, the residuals, and the fit-quality marks of each snip.
> - **Results (D40, D51, S19, S20, F7, D26):** each parameter's value and error, its
>   tie state and flags; the correlations; the fit statistics; Continue from best
>   fit; the run history, to which the user commits a run with a description.
> - **Compare (D41):** any two runs, their quantities matched across different
>   parametrisations, the change of each in σ, and both models on the same data.
> - **Convergence and outliers (S21, S29, D27):** ALIS's convergence check, random
>   restarts, and pixels that the best fit leaves far out, reviewed before they are
>   left out of the fit.
> - **Runs made outside the dashboard (S31):** a fit made by `run_alis
>   project.model` is shown as the latest run, ready to inspect or commit.
>
> Everything the tab shows comes from the project model, through the blinding gate,
> and every change to the model is one step of the history, as in Stages 3–5. The
> logic needs no Qt and goes in `alis/dashboard/` (new modules: `runs.py`,
> `settings.py`, `preflight.py`, `results.py` and `compare.py`), tested with pytest
> alone; the windows go in `alis/dashboard/qt/`. ALIS outside `alis/dashboard/`
> changes only where a query agrees (Q6.6).
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
what Stages 1–5 taught. The design follows Claude's lean on each query; each part
names the queries that would change it.*

### What a user can do at the end of this stage (v1)

1. **Prepare** (Stages 3–5): the spectra, the systems, the snips with their fit
   regions and continua, and the components, on the Data, Regions and Components
   tabs, with the `.mod` panel beside them.
2. **Run:** press Run (Ctrl+R) on the Fit tab. A pre-flight check lists what would
   stop the fit, and what is worth a look first. The fit runs in the background,
   with its χ² drawn after each iteration, while the user goes on working; Stop ends
   it early and keeps where it got to.
3. **Analyse:** inspect each snip with the best fit and its marks; read the results
   and the correlations; continue from the best fit; commit the run, with a
   description, to the run history; compare any two runs; check the convergence;
   review the pixels the fit leaves far out.
4. **Plot** (step 8) is after v1. Until then, ALIS's own `plotscript` settings write a
   plotting script, as they do for `run_alis`.

### Layers

```
alis/dashboard/qt/   fit.py         the Fit tab: Inspect, Results and Compare, which
                                    replace the frames of tabs.py; the run's process
                                    (QProcess), its progress and Stop
                     plots.py       the spectrum view gains a second model (Compare)
                                    and marked pixels (outliers)
                     dialogs.py     All settings…, Commit run…, Random restarts…
                     regions.py     the Model switch gains the best fit (Q5.8)
                     window.py      Run, Stop and Commit; the run in the status bar;
                                    closing with a run going
alis/dashboard/      runs.py        a run of the project: the run bundle, the command,
                                    its progress, Stop; the latest run and the
                                    committed runs (commit, describe, remove,
                                    restore); what a run holds (values, errors, the
                                    covariance, its statistics); random restarts
                     settings.py    the settings, from ArgFlag (F9): the model's own
                                    and the computer's; setting and clearing them
                     preflight.py   the pre-flight check (S17)
                     results.py     the results table and its flags, the
                                    correlations, the fit-quality marks, the best fit
                                    of each snip (through profile.py), Continue from
                                    best fit, the outliers
                     compare.py     two runs' quantities, matched across
                                    parametrisations, their changes in σ; the
                                    assessment of random restarts
```

The names may change as the code is written; the log records the names as built.

- **No logic in the widgets**, every change to the model one step of the `History`,
  every value through the `Gate` (Stages 3–5). Running, committing and describing a run
  change no model text, and are not steps (Q6.7).
- **A run is ALIS's own run of the bundle.** The dashboard runs `run_alis` on a copy of
  the project as it stands, in a separate process, exactly as a user would on the
  command line (D9); its result is the bundle's `runs/latest/`, as `run_alis
  project.model` leaves it (Q6.2).
- **The best fit drawn is ALIS's** (D60): `profile.py` evaluates the run's `.mod.out`
  values on the snips, as the Components tab evaluates the starting values (Q6.5).
- **Blinding (D24, D49):** best-fit profiles may be drawn; values never shown. What
  each part of the tab shows of a blinded parameter is listed under "Blinding" below.

### Running a fit (F6, F9, S17, D38; Q6.2–Q6.4)

- **Where it runs** (Q6.2): Run writes the project as it stands, unsaved changes
  included, to a run bundle in `~/.alis/runs/` (or `$ALIS_HOME`), and starts
  `run_alis` on it in a separate process. When the fit ends, the process stores the
  result in the project's own file at once (its `runs/` members only, under the
  bundle's lock, as `run_alis project.model` does), so a finished fit is never lost;
  the model on disk changes only with Save (D48). A project that has no file yet
  keeps the result in memory, and in its recovery copy, until it is saved. The run
  bundle is then deleted.
- **What it is given** on its command line (`--set`), never written in the model:
  the computer's settings (backend, CPUs, GPUs; Q6.4), the outputs the tab reads (the
  covariance and the report), `out verbose 1` (so that the process prints the χ² of
  each iteration, without the parameters), and no plot windows (`plot fits False`).
  `run.json` records them, as it records any command-line setting.
- **One run at a time.** The model can be edited while it runs: the run fits the model
  as it was when Run was pressed, and the Fit marker then says the model has changed
  since.
- **Progress:** the χ² of each iteration, read from the process's output, on the
  progress chart (linear or log, D38) and in the status bar ("Run 5 · iteration 9 · χ²
  3796.2 · 0:41"); the previous run's χ² against iteration in grey (kept in its
  `run.json`). Nothing else of the process's output is shown (ALIS prints the
  best-fit model at the end of an unblinded run), and the messages the tab shows, a
  warning or the error that stopped the run, pass the bundle's mask of hidden lines
  (`bundle._MaskHidden`).
- **Stop** (Q6.3): ALIS's own interruption (the minimiser's handler of `SIGQUIT`)
  ends the fit at its current iteration; ALIS writes its outputs, and the run is
  kept as "interrupted", with its values, which Continue from best fit can start
  from. ALIS sets an interrupted fit's errors to its values, so they are never shown.
  Pressing Stop again (now "Cancel") ends the process at once; nothing is kept, and
  the latest run stays as it was.
- **An error** (ALIS stopped with one): the Fit marker's "!" with the message, as
  Stage 3 shows it (`runs/last_error.json`), and the line it names highlighted in the
  `.mod` panel when it names one.
- **Closing the dashboard during a run** asks: Stop the run, or Keep it running. A run
  kept running still stores its result in the project's file, and the Fit tab shows it
  when the project is next opened, as a command-line run (S31). Re-attaching to a run
  with its live progress is after v1 (F6).

### Settings (F9, D38; Q6.4, Q6.6)

- **The Run box** shows the computer's settings (Backend, CPUs, GPUs), then "Settings
  you changed": the model's `par` lines whose values differ from ALIS's defaults, each
  editable in place (one step) and each with ✕, which returns it to the default (its
  line removed).
- **All settings…** lists every setting of `alis.config.ArgFlag`, by section, with
  its default, its type (a tick box, a number, a word, or a menu where ALIS takes
  only some words, such as `run backend`) and a one-line description (Q6.6), with a
  filter. A change is a step that writes or changes the setting's `par` line.
  Settings a bundle cannot run (`sim random`, `sim perturb`, `sim beginfrom`,
  `iterate model`, `generate data`; `bundle.unsupported`) are not offered, and those
  the dashboard sets for its own runs (`out verbose`, the plot windows) are shown
  greyed, with what the dashboard uses.
- **The computer's settings** (Q6.4(a)) are preferences of this computer, not lines
  of the model, so that a model fitted on a laptop runs unchanged on a GPU machine.
  A model that already sets them keeps its lines, and the Run box says that the
  model's own value is used.

### The pre-flight check (S17; Q6.4)

Run checks the project first (`preflight.py`). Each finding names its line in the
`.mod` panel, or its snip, and a click shows it.
- **Run is refused** when ALIS would stop: the model does not read, or ALIS's loaders
  refuse it (`validate.full`); a setting a bundle cannot run; a starting value outside
  its limits; no free parameter, or more free parameters than fitted pixels.
- **Notes**, after which Run goes ahead (the Run box shows "Pre-flight check: no
  problems; 2 notes", and the notes are a click away):
  - a free parameter that reaches no fitted pixel, by ALIS's own influence table
    (`load.load_par_influence`), so that the data cannot constrain it;
  - an ion in no snip (Stage 5's marker);
  - a starting value at one of its limits;
  - a b_turb starting below 1 km/s (a preference; `doc/ALIS_workflow.md` §4.1.2);
  - a snip with no continuum;
  - the notices of an imported fit kept as imported (D20, D22), pixels fitted twice
    (D19), and regions changed since the components were last edited (Stage 5's
    marker).

### Inspect (D38, D39, S18; Q6.5)

- **Left:** the Run box (above), Run and Stop, Convergence check… and Random
  restarts… (S21); under it the progress chart.
- **The view:** a View menu (all snips; one ion, grouped as the Components tab's
  navigator, the Lyα forest included; one snip) with ‹ ›; Columns, the Components
  tab's setting (D38), greyed when one snip is shown; Velocity | Wavelength (velocity
  relative to the transition a snip was cut for, as the Lyα forest's panels; a snip
  with no transition in the project, in wavelength); the zoom toolbar (D38).
- **Initial | Final** (Q6.5): Initial is the model as it stands now, the one the next
  run starts from (as the Components tab draws it); Final is the shown run's best fit.
  Each panel shows the data divided by the shown model's own continuum (the best-fit
  continuum for Final), the model, the fit regions, the residuals underneath, and the
  component ticks, locked (no drags on this tab).
- **The shown run** is the latest run, or the run chosen in the run history; its name
  is above the panels ("Final · run 3").
- **Fit-quality marks** (S18, D39): each panel's corner, and a chip per snip in the
  "All snips" strip, show χ²ν and the runs test of the Final model, each with its own
  mark: ✓ within 2σ, ! 2–3σ, ✕ beyond 3σ (`out reportsig`). They are computed as
  `alis/report.py` computes them (χ²ν against its expected spread, which is flagged
  only above; the runs-test z, flagged either way), from the best fit of each snip
  (Q6.6). A click on a chip shows that snip.
- **Continue from best fit** and **Review clipped pixels…** sit above the panels
  (below).
- The best fit of a snip that the run did not have (cut since) is not drawn, and its
  chip says so.

### Results (S19, S20, D40, D51, F7, D26; Q6.7, Q6.8, Q6.12)

- **The results table** of the shown run ("Results · run 3"), grouped as in the
  mockup:
  - each system, the primary first, with its ions and components: log N, z (with v
    relative to the system), b and T;
  - the Lyα forest, collapsed, with its flagged rows shown;
  - the continua and the instrument (FWHM, shifts, zero levels), collapsed.

  Each row has Value, Error, State (● free, ■ fixed, ⛓ tied, ƒ linked, as on the
  cards) and Flags: "at limit", "unconstrained" (Q6.12), "blinded", and the notices
  kept as imported ("not tied to H I"). A row selected highlights its lines in the
  `.mod` panel (S16). "Copy as table" copies the rows as tab-separated text, through
  the gate.
- **Fit statistics:** χ² (and its starting value), degrees of freedom, χ²ν,
  iterations, the run time, why the fit stopped (`utils.getreason`), the convergence
  check's result when it ran, and where and when the run took place (host, command,
  finished; S31).
- **Correlations** (D51, under the statistics): the correlation matrix of the free
  parameters of a group chosen in a menu (a system, an ion, the continua, everything),
  from the run's covariance, in a colour-blind-safe diverging scale (D43), with the
  pair and its coefficient on hover. A correlation reveals no value, so blinded
  parameters are included, by name.
- **The run history** (F7, D26, D51, beside the table): the latest run first, with
  Commit…, which asks for the user's own description (D40); then the committed runs,
  newest first: number, description (edited by a double-click), χ²ν, when, where
  ("cmd" for a run made by `run_alis`, S31). Selecting a run shows it on Inspect and
  Results; a committed run can be removed, after a confirmation. What a committed run
  keeps is Q6.7.
- **Continue from best fit** (S20, Q6.8): the shown run's best-fit value of every
  parameter is copied into the model, as one step, after a preview of the lines it
  changes. Only lines unchanged since the run receive values; the preview lists the
  others, which keep theirs. Linked parameters are left to ALIS. A value copied into a
  hidden line stays hidden, and under global blind every line that receives a value
  is hidden (`Project.mark_hidden`, S20). Interrupted runs can be continued from.

### Compare (D41; Q6.9)

- **Choose** any two runs (the committed runs and the latest), Swap, and "Restore run
  N's model…", which replaces the model and its snips with the run's, as one step,
  after a preview of what changes.
- **The quantities** (Q6.9), matched across different parametrisations:
  - each ion's total log N in each system, from the components' column densities,
    with its error from the run's covariance;
  - each isotope's ratio to its element (log D/H), whether the run fits the two
    column densities or the ratio itself (`ion=2H_I/1H_I`, or a `variable` linked to
    it), with its error from the covariance;
  - the components, matched by system, ion and velocity (the nearest within 10 km/s,
    a preference): v, b and T, and how many components each ion has;
  - the instrument (FWHM), χ²ν, degrees of freedom and iterations.

  Each row shows both values, Δ (later − earlier) and |Δ|/σ.
- **Blinded quantities** show only |Δ|/σ, with no values and no sign (Q0.9, Q0.11;
  `Gate.sigma_change`); a quantity derived from a hidden value is hidden.
- **The view:** both best fits on the same data (the first solid, the second dashed),
  their difference in units of the error underneath, with Inspect's View menu, ‹ ›,
  Columns and toolbar (D41).

### Convergence (S21; Q6.10)

- **Convergence check…** runs the fit with ALIS's own check (`run convergence True`
  on its command line, with `run convcriteria` shown and editable); Fit statistics
  then says "Converged", or "Not converged for 3 of 41 parameters".
- **Random restarts…** (Q6.10): after a run with a covariance, N fits (10, a
  preference) start from values drawn from that covariance about the best fit, as
  `sim perturb` draws them, within the limits, with fixed and linked parameters
  unchanged. They run one after another, each a run of its own, with "restart 3 of
  10" in the status bar, and Stop ends them. They are then assessed as
  `convergence.assess_restarts` assesses them (`sim convergetest` and `sim
  convergesig`): for each free parameter, the largest |Δ|/σ across the restarts,
  flagged beyond the tolerance, and one line of summary. A restart that reaches a
  lower χ² is offered to continue from. The restarts are kept with the run they
  started from, and committed with it.

### Outliers (S29, D27; Q6.11)

- **Review clipped pixels…** marks, on Inspect's panels, the fitted pixels whose
  residual from the shown best fit is beyond 4σ (a preference). Lone pixels and pairs
  are ticked; three or more in a row are shown unticked, since they may be
  absorption the model misses (QF.26(c)).
- Each panel has Accept all and Reject all, and a click on a pixel ticks or unticks
  it. Apply leaves the ticked pixels out of the fit, as Tweak dataset regions' right
  drag does (the snip's fit column; Stage 4), as one step ("Leave out 6 outliers"), and
  offers Run.

### Command-line runs and the bundle on disk (S31; Q3.8)

- A run made by `run_alis project.model` while the project is open is taken in
  without asking (Q3.8(a), `Session.take_runs`), and one made while it was closed is
  there when it opens: the run history lists it as the latest run, "cmd", with where
  and when it ran, ready to inspect and commit.
- Its χ² against iteration is not known (the command line does not record it), so
  the progress chart shows only its end.

### The Regions tab's Model switch (Q5.8; `deferred_work.md` §7.2)

- The switch gains a choice when a run exists: the starting model, or the latest
  run's best fit, drawn as on Inspect.

### Blinding (D24, D49, QF.20)

- **Values:** best-fit values through `Gate.best_fit` (masked for a hidden parameter
  and under global blind); starting values as on the other tabs.
- **Errors** are shown (QF.20(b)), except an interrupted run's, which ALIS sets to
  the values.
- **χ²**, its progress, the fit-quality marks, the correlations and the outliers are
  shown: none reveals a value.
- **Compare and restarts:** a hidden quantity shows only |Δ|/σ, unsigned.
- **Profiles** of blinded components are drawn (D24).
- **Copy as table, tooltips and messages** pass the gate; the process's output is not
  shown, and its messages pass the bundle's mask.
- **Continue from best fit** under global blind hides every line it fills (S20).

### Markers, preferences and actions

- **The Fit marker** (F12; Stage 3's rules): ○ no run yet; ✓ the latest run, with its
  χ² and when; ↻ the model has changed since the latest run, or a run is going
  ("Running: iteration 9"); ! the last run stopped with an error, or the pre-flight
  check refuses Run.
- **New preferences** (`preferences.py`, `fit.*`): this computer's backend, CPUs and
  GPUs; the progress chart's axis (linear); the threshold of outliers (4σ); the
  number of random restarts (10); the starting b_turb below which the check notes it
  (1 km/s); the velocity within which Compare matches components (10 km/s); the
  thresholds of "unconstrained" (Q6.12).
- **Actions** (`actions.py`, scope "Fit"): Run (Ctrl+R) and Commit run… (both built
  now, no longer "coming"), Stop, Continue from best fit, Convergence check…, Random
  restarts…, Review clipped pixels…, and ‹ › for the view (the tab's own keys).

### Not in this stage

- The Plot tab and the plotting presets (step 8, D28, D42), after v1.
- Re-attaching to a run with its live progress after the dashboard was closed (F6),
  command search (F10), S15's link editor (Q5.9), reproducing an imported fit (S28),
  attaching a source spectrum to an imported model (S23): after v1.
- Orders mode, PypeIt spec1d and the stacked spectrum (D29–D32, Q4.20): after v1.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage6_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests;
> run the `fast` batch before the stage closes.

**6.1 — Runs (`runs.py`, no Qt; F6, F7, D26, S31; Q6.2, Q6.3, Q6.7).**
- The run bundle of a project as it stands; the command and its settings; the
  progress read from the process's output; Stop and Cancel; storing the result in the
  project's file; the latest run and the committed runs (commit with a description,
  describe, remove, restore a run's model); what a run holds (its values, errors and
  covariance, mapped to the model's parameters; its `run.json`).
- **Check** (unit tests): a run of `examples/metal_line_abs` through `runs.py` gives the
  `.mod.out` that `run_alis` gives, and the χ² of every iteration; Stop leaves an
  "interrupted" run with its values, and Cancel leaves the latest run as it was; a
  result stored while the project has unsaved changes leaves the model on disk
  unchanged; a committed run survives Save and reopening, and restoring it gives its
  model and snips byte for byte, as one step; on `examples/blind` and under global
  blind, nothing hidden appears in the progress, the messages or `run.json` (the
  gate's `strings` test).

**6.2 — Settings and the pre-flight check (`settings.py`, `preflight.py`; F9, S17;
Q6.4, Q6.6).**
- The settings from `ArgFlag`, with their types and descriptions; the model's own and
  the computer's; setting and clearing one as a step; those a bundle cannot run, and
  those the dashboard sets. The pre-flight check, with what refuses Run and what is a
  note.
- **Check** (unit tests): every setting of `ArgFlag` is offered or deliberately left
  out, and has a description (`run_alis --list-settings` prints it); a setting set
  and cleared restores the text byte for byte; each finding of the check, refusing
  and noting, on a model made to have it; J1358p6522 as imported gives its two
  notices as notes and no refusal.

**6.3 — Results (`results.py`; S18–S20, D39, D40; Q6.5, Q6.8, Q6.12).**
- The best fit of each snip (`profile.py` with the run's values); the fit-quality
  marks; the results table with its groups, states and flags; the correlations; the
  statistics; Continue from best fit; the outliers.
- **Check** (unit tests): the best fit drawn agrees with ALIS's `_fit.dat` model, on
  metal_line_abs and J1358p6522, to the precision of the values the `.mod.out`
  prints; the marks equal the `.report`'s χ²ν and runs z
  (J1358p6522's reference: nine snips ✕ by the runs test alone); "at limit" and
  "unconstrained" on J1358p6522's Lyα forest as the mockup shows them; Continue from
  best fit is one step that ALIS reads and one undo restores, skips a line changed
  since the run, and hides what it fills under global blind.

**6.4 — Inspect (`qt/fit.py`; D38, D39, F6).**
- The Run box (the computer's settings, the settings changed, All settings…, the
  pre-flight check), Run and Stop, the progress chart; the View menu with ‹ ›,
  Columns, Velocity | Wavelength, Initial | Final; the panels and the snip chips with
  their marks; the status bar; closing during a run; the Fit marker.
- **Check** (`gui` tests): a run of metal_line_abs from the tab draws its progress and
  then its best fit, and the marker becomes ✓; Stop keeps an interrupted run; Run is
  refused with a pre-flight problem and goes ahead with notes; J1358p6522's chips as
  in the mockup; Initial and Final draw the starting model and the best fit.

**6.5 — Review of Inspect and running (Q6.1).**
- Publish the sub-tab's screenshots beside its mockup, on a new page, for RJC's
  comments, and apply them before Results and Compare are built.

**6.6 — Results (`qt/fit.py`; S19, S20, D40, D51, F7).**
- The results table, Copy as table, the statistics, the correlations with their menu,
  the run history (Commit…, the description, removing a run), Continue from best fit.
- **Check** (`gui` tests): J1358p6522's imported outputs shown as in the mockup;
  Commit… asks for a description, which a double-click edits; a committed run
  survives Save; on `examples/blind`, no hidden value in the table, its tooltips or
  what Copy as table copies.

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

**6.8 — Convergence, restarts and outliers (S21, S29, D27; Q6.10, Q6.11).**
- Convergence check…; Random restarts… and their assessment; Review clipped
  pixels… and Apply.
- **Check** (`gui` and `fast` tests): metal_line_abs's restarts all agree; an outlier
  added to its data is marked and left out by Apply, as one step; three outliers in a
  row are shown unticked; ALIS's convergence check's result shown.

**6.9 — Markers, actions, preferences, command-line runs (F12, S31; Q5.8).**
- The Fit marker's states; the actions and preferences; a run made by `run_alis
  project.model`, open and closed; the Regions tab's best fit.
- **Check** (`gui` tests): `run_alis project.model` while the project is open shows a
  "cmd" run without asking; the marker in each state; the Regions tab's switch draws
  the best fit.

**6.10 — From a spectrum to a fit, in the dashboard (end to end).**
- The `fast` test of Task 5.10, with the last step on the Fit tab instead of
  `run_alis`: Run, the best fit agreeing with `fit_spectra.mod.out.reference` within
  the errors; Commit…; the temperature changed to free on the Components tab and Run
  again; Compare of the two runs; Continue from best fit.

**6.11 — Close the stage and v1.**
- Publish the screenshots of the tab beside the mockups for RJC's review, and apply
  the review.
- `doc/ALIS_workflow.md`: a section on the Fit tab, and §4.3 brought up to date for
  v1. Also update `CHANGELOG.md`, `tests/README.md` and the `gui-component` skill.
- Run `test-coverage` on the new modules, and the `unit`, `gui` and `fast` batches.
- Record the stage's decisions in `ALIS_v2_dashboard_prompts.md`, mark v1 complete in
  the stage table of `dashboard_stage0.md`, and list what follows v1 (Q6.13).

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
- `claude_prompts/deferred_work.md` §7.2 (the best fit on the Regions tab).
- `doc/ALIS_workflow.md`: §2.1 (the settings), §4.1.2 (starting b), §4.2 (bundles),
  §5 (the outputs) and §6 (convergence).
- **The code:**
  - `alis/bundle.py` (`run`, `_run_info`, `_set_latest`, `unsupported`, `update`)
    and `alis/scripts/run_alis.py`, for a run;
  - `alis/config.py` (`ArgFlag`), for the settings;
  - `alis/minimise.py` (`defiter`, the `SIGQUIT` handler, status −20), for the
    progress and Stop;
  - `alis/report.py`, `alis/convergence.py`, `alis/simulate.py` (`perturb`) and
    `alis/save.py` (`save_model`, `save_covar`), for what a run writes;
  - `alis/dashboard/` (`profile.py`, `components.py`, `markers.py`, `session.py`
    with `take_runs`, `blinding.Gate.best_fit` and `sigma_change`,
    `project.mark_hidden`, `validate.py`) and `alis/dashboard/qt/` (`tabs.py`,
    `plots.py`, `window.py`).
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

**Q6.3 — Stop.** ALIS already stops a fit cleanly on `SIGQUIT` (`Ctrl+\` in a
terminal): the minimiser ends its iteration, ALIS writes its outputs, and the run is
marked "interrupted", with its errors set to its values. I propose:
- **Stop** sends that signal, and keeps the run as "interrupted": its values can be
  inspected and continued from, and its errors are not shown (they are the values,
  so they would show a blinded one);
- pressing it again, **Cancel**, ends the process at once, and keeps nothing.

My lean: as proposed.

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

**Q6.5 — Initial and Final.** Step (6) asks for the initial or the final model on the
data. I propose that Initial is the model as it stands now (the one the next run
starts from, as the Components tab draws it), and Final the shown run's best fit,
drawn by `profile.py` from the run's `.mod.out` on the snips as they are now. The
alternative is that Initial is the shown run's own starting model, which differs from
the current one once the model is edited after the run.

My lean: as proposed, since the toggle is then the way to see what an edit changed
since the last fit.

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

**Q6.8 — Continue from best fit (S20).** The model may have changed since the run. I
propose to copy the best-fit values only into lines unchanged since the run (matched
to the run's input model), after a preview that lists the lines changed and the
lines skipped, as one step; to leave linked parameters to ALIS; and to allow it after
an interrupted run, so that a stopped fit can be resumed.

My lean: as proposed.

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

**Q6.10 — Random restarts (S21).** A bundle refuses `sim perturb` (it writes its
fits to a `sims/` directory on disk). I propose that the dashboard draws the starts
itself, as `simulate.perturb` draws them (from the covariance, about the best fit,
within the limits), runs each as an ordinary run, one after another, and assesses
them with `convergence.assess_restarts`'s tests, showing only |Δ|/σ for each free
parameter. Ten restarts by default (a preference). The alternative, making `sim
perturb` run inside a bundle, changes ALIS outside the dashboard for a feature that
the dashboard can drive with what exists.

My lean: as proposed, with no change to ALIS.

**Q6.11 — Outliers (S29, D27).** I propose a threshold of 4σ (a preference): with
4,000 fitted pixels, 3σ would mark about eleven by chance alone. Lone pixels and
pairs are ticked; three or more in a row are shown but unticked, since they may be
absorption the model misses. Accepted pixels are left out of the fit as Tweak dataset
regions leaves pixels out (Stage 4), as one step.

My lean: as proposed.

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

**Q6.13 — v1.** This stage completes v1 (D33). At its close, I propose:
- **(a)** to mark v1 complete in the stage table of `dashboard_stage0.md`, and to list
  what follows (the Plot tab first, then the items marked "Later"), for you to order;
- **(b)** to deprecate `prepfit` as D4 plans: it keeps working, and prints a note
  that the dashboard (`alis`) now covers its features;
- **(c)** to leave the release itself (the version number, the tag, publishing) to
  you.

My lean: (a), (b) and (c).

## Prompts

> RJC will be responsible for writing this section.
