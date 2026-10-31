# Dashboard Stage 7 log

Changes and bug fixes from real use of version 1 of the dashboard: RJC's list, each
item a task with its check. The plan is in `claude_prompts/dashboard_stage7.md`.

### 2026-10-30 (Prompt 1: reading, the design of Task 7.1, and queries Q7.1–Q7.5)

**Reading.**
- This stage's document: item 1 of the list, and Task 7.1.
- The decisions D1–D77 of `ALIS_v2_dashboard_prompts.md`, with S23 and QF.5.
- The "After v1" list and Q0.5 of `dashboard_stage0.md`, and items 8 and 15 of
  `deferred_work.md`.
- The pattern of Stage 6's document and log.
- The code the task builds on:
  - `alis/dashboard/project.py`: `read_snip`, `SnipData`, `Snip.regions`,
    `import_model`, `_build_rows`, `_build_systems`, `structure`,
    `working_structure`, `inferred`;
  - `alis/dashboard/snips.py`: `snip_step`, `_names`, `_row_words`, `clear_plan`;
  - `alis/dashboard/lines.py` (`snip_transitions`, `_guess_transition`) and
    `alis/dashboard/sources.py`;
  - `modes.empty_project`, `datasets.default_fwhm`, `remove.plan_snip`;
  - in ALIS itself, `bundle.pack`, `load.load_ascii` and `prepfit/specplot.py`
    (`write_data`, `props`);
  - in the window, `qt/window.py` (`import_path`), and the Regions and File actions
    of `actions.py` and `qt/regions.py`.

**Facts that shaped the design.**
- **Snips from the example.** Both of
  `context/fitting_examples/Temperature/J013301m400628/data/*_reg.dat` are unbroken
  runs of `J013301m400628.dat`'s pixels (17463–17843 and 37052–37434), equal to the
  last bit in wave, flux and error. Their fourth column is a fit range of 0s and 1s,
  with 69 and 110 pixels fitted.
- **What `prepfit` writes.** Its snips are `<stem>_<mass><El>_<stage>_<λ:.1f>_reg.dat`,
  holding wave, flux × continuum, error × continuum, and the fit range. The source's
  fifth column is the continuum when there is one. The dashboard's source spectrum
  multiplies in a continuum column the same way (D13), so the snips can be compared
  to it directly.
- **The example's `.mod`.** Its data lines say `columns=[wave,flux,error,fitrange,
  continuum]`, but the files have four columns. `load_ascii` warns and uses a
  continuum of 1, and `project.read_snip` ignores a column that is not there.
- **Snips need a system.** A new project has no system when no redshift is given
  (`modes.empty_project`). The Regions tab lists snips only under a system's
  transitions, so a snip whose transition has no system is not shown (Q7.4).
- **The bundle's layout matches the example.** `bundle.pack` stores a `.mod`'s snips
  where its relative paths name them. A new project's model is
  `files/model/<name>.mod` and SNIP writes its snips to `files/data/`, so
  `../data/<snip>` from the model needs no rewriting (Q7.1(a)).
- **Reset FWHM's default.** For a file row it is the FWHM given when the file was
  added (`datasets.default_fwhm`). A loaded `.mod`'s FWHM that differs from it shows
  ↺ (Q7.3).
- **Test data.** Tests read `context/fitting_examples/…` and skip when it is absent
  (J1358p6522), so the J013301m400628 check can do the same.

**Baseline.** `pytest -m unit`: 2002 passed, 119 skipped (101 s; the 16 warnings are
`test_save_helpers.py`'s, as before); `pytest -m gui`: 190 passed (365 s). At the
close of Stage 6 the log recorded 2000 passed with 121 skipped, and 188 passed. No
test fails.

**The design and the queries.** The Design section of the stage document now
describes Task 7.1:
- (a) Load snips… on the Regions tab;
- (b) Load a .mod… in the File menu and the `.mod` panel;
- a new module, `alis/dashboard/loadfit.py`, with no Qt.

Q7.1–Q7.5 ask about five decisions, and the design follows their leans:
- Q7.1: whether a `.mod` brings its own snips;
- Q7.2: what counts as an overlap, and how a snip is replaced;
- Q7.3: a loaded `.mod`'s settings, blinding and FWHM, and its earlier run;
- Q7.4: snips with no system to place them in;
- Q7.5: S23.

Task 7.1 waits for RJC's responses. No code was changed.

### 2026-10-30 (RJC's responses to Q7.1–Q7.5, and Task 7.1)

**The responses.** RJC agreed with the lean of each of Q7.1–Q7.5: a `.mod` brings its
own snips; overlaps are replaced after one dialog; a loaded `.mod`'s settings, `run
blind`, FWHM and earlier run are handled as proposed; snips that cannot be placed are
refused; S23 comes off the "After v1" list at the close. No further query was needed.

**Task 7.1 — Load snips… and Load a .mod….**
- `alis/dashboard/loadfit.py` (new, no Qt):
  - `read_snip_file` reads a table of 3 to 5 columns (a fourth of 0s and 1s is the
    fit range, a fifth the bad-pixel mask) and refuses anything else, with the
    reason;
  - `compare` and `match_row` match a snip to the file row it was cut from: every
    wavelength on a source pixel (to 5% of its width), and the flux and error equal
    there (to 10⁻³ of the error). Otherwise the reason is given for the nearest row:
    outside its range, not on its pixels, or the flux or error differs at *n* pixels;
  - `place` finds the transition the file name names, at the system whose redshift
    puts it inside the snip, or refuses (Q7.4);
  - `load_snips_plan` gives the problems (all or nothing), the snips each file
    replaces (any shared pixel of the same spectrum; two chosen files that share
    pixels are refused, Q7.2), and one step. The step clears each replaced snip as
    CLEAR does, then adds the new ones as SNIP does. Each file is stored byte for
    byte under its own name, with a column of 0s added when it has no fit range,
    and the source's bad pixels when it has a mask (D52);
  - `load_mod_plan` reads the `.mod` as `bundle.pack` packs it, without its outputs,
    into a project of its own, and matches its snips. Its step replaces the text
    (the `.mod`'s as written, with `run blind` following the project and a path that
    would leave the project's files rewritten to `../data/`), the snips, and
    `ui/project.json`. The rows get the `.mod`'s specids; the transitions are
    recorded where they can be placed; knots, kept notices and the old snips'
    records go; `imported_fwhm` holds the `.mod`'s FWHM. Systems inferred from its
    components are marked guessed (D58). The preview lists the model's lines that go,
    with notes on the snips loaded and removed, and on an FWHM that differs from the
    one given for the file (Q7.3).
- `alis/dashboard/snips.py`: SNIP's naming (`snip_folder`, `free_member`,
  `new_specid`) and its four makers (`NewSnip`, `add_snip_makers`: the files and
  data lines with their records, the continua, the absorption lines' specids, the
  zero levels) are factored out of `snip_step`, which now uses them; SNIP's behaviour
  is unchanged (its tests pass as before).
- `alis/dashboard/lines.py`: `named_line` and `placing_system`, factored out of
  `_guess_transition`, which keeps its behaviour.
- The window:
  - `actions.py` adds `file.load_mod` (File menu, after Import a fit) and
    `regions.load_snips`;
  - `qt/window.py` adds `do_file_load_mod` (the file dialog, the message when it
    cannot be loaded, the `PlanDialog` preview, one step, a status message), the
    `open_many` file choice, and Load a .mod… in the `.mod` panel's right-click menu
    (`qt/modpanel.py`'s `set_panel_actions` gains `menu_only`). As a button in the
    panel's header it widened the panel's least width from under 460 px to 529 px,
    which `test_the_panel_s_width_is_the_user_s` caught;
  - `qt/regions.py` adds `do_load_snips` (the files, one warning listing every
    problem, the replacement dialog, one step, then the first loaded transition
    shown), enabled while the project has a source spectrum.
- **A change from the design.** Beside SNIP and CLEAR, the button cut short "Add
  regions to all" and "Tweak dataset regions" at 1440 px with the `.mod` panel open
  (seen in an off-screen screenshot), so it sits under the transition list instead.
  The Design section says so, and it is shown in the review at the close.

**Checks.**
- `tests/test_dashboard_loadfit.py` (27 unit tests, new; 95% of `loadfit.py`):
  - reading, matching and placing, with every refusal;
  - `prepfit` snips loaded with their fit ranges, ALIS reading the result, and one
    undo giving back every byte;
  - a three-column snip loaded with no fit region;
  - a source's bad-pixel mask carried;
  - one failing file stopping the load;
  - two chosen files sharing pixels refused, and an overlapping snip replaced
    (undo exact);
  - SNIP still cutting after a load;
  - a `.mod` brought in as written with its snips replacing the project's, undo
    exact;
  - `run blind` following the project both ways, with the guard on;
  - the `.mod`'s FWHM used, and the given one kept as Reset FWHM's default;
  - components far from every system forming a guessed system;
  - a `.mod` with a snip from another spectrum, one that cannot be read, or one
    whose model does not read, refused;
  - a data path outside the project rewritten.
- **The check of Task 7.1** (`test_the_j013301m400628_snips_and_mod_load`, skipped
  when the example is absent). A project made from `J013301m400628.dat` at z = 2.7934
  loads the two `prepfit` snips as Lyβ 1025.7 and Lyα 1215.7 of system a. Their
  pixels are the spectrum's (wave, flux and error equal to the last bit), and their
  fit ranges are their own: 69 and 110 pixels fitted, in 3 regions and 1. Loading
  `J013301m400628.mod` then gives its text exactly, with specids 0 and 1, the FWHM of
  4.181 km/s, and system a's three components, each with H I and D I. ALIS reads the
  model each time.
- The window (`test_dashboard_qt_regions.py`):
  - Load snips… loads nothing, with one message, when a file fails; it loads two
    snips and shows O I 1302.2 with its regions; it asks before replacing, and
    Cancel changes nothing; two undos restore the project;
  - Load a .mod… is offered in the File menu and the `.mod` panel. Cancel in its
    preview changes nothing; Load puts the `.mod` in the editor, its component in
    the Components tab and its snips in the Regions tab;
  - without a source spectrum, Load snips… is disabled;
  - no design reference appears in the window's text.
- `test_dashboard_actions.py`: the File menu's list gains Load a .mod….
- Off-screen screenshots of the example, driven through the window, show the
  Regions tab (Lyβ with its 69 fitted pixels in 3 regions and its continuum) and the
  Components tab (a1–a3, H I with D I computed by the link).
- **The batches**, after the last change:
  - `pytest -m unit`: 2029 passed, 119 skipped (87 s), up from 2002 at the baseline
    by the 27 new tests;
  - `pytest -m gui`: 192 passed (371 s), up from 190 by the two new window tests.

  ALIS outside `alis/dashboard/` is unchanged, so the regression harness was not
  run.

### 2026-10-31 (Prompt 2: RJC's list reviewed, Tasks 7.2–7.15, queries Q7.6–Q7.13)

**Reading.**
- Items 2–14 of the list, and the prompt.
- The decisions D13–D24, D35–D38 and D52–D77.
- The "After v1" list of `dashboard_stage0.md`.
- `context/fitting_examples/Temperature/` (`context.md`, and the J013301m400628 and
  J101155p294141 models).
- The code behind each item:
  - `components.py` (`exclusion`, `include`, `is_excluded`, `drag`, `draggable`,
    `bounds`, `temperature`);
  - `project.py` (`Component`, `_components`, `_attach_ratios`, `_attach_excluded`,
    `_untied_notices`, `tie_component`, `Row`);
  - `qt/components.py` (the card, `_toggle_ion`);
  - `qt/dialogs.py` (`SpectrumDialog`);
  - `qt/window.py` (the start page, the toolbar, `_fit_actions`, the two `.mod`
    actions);
  - `preferences.Recent`;
  - `qt/fit.py` (the View pane, `_draw_panel`) and `qt/plots.py`'s
    `set_normalised`;
  - `datasets.py` (shift and zero level), `snips.row_shift`, `edit.set_limits` and
    `add_zero_level`;
  - in ALIS, voigt.py's limits, and the shift and zero level in `load.py` and
    `model_eval.py`.

**Facts that shaped the tasks.** Each bug was reproduced first, with scripts in the
session's scratchpad.
- **Items 5 and 6.** The test model was J101155p294141's absorption lines, with
  J013301m400628's snips (its own are not in `context/`). Each component is one H I
  line with an unlabelled redshift, and D I comes from the ratio line.
  - The card lists D I under "Excluded", since `comp.lines_of("2H_I")` is empty.
  - `exclusion` raises the error RJC quoted for both ions. The project lists no
    notice but "inferred", because `_untied_notices` needs two lines.
  - `_toggle_ion` calls `exclusion` outside `_act`, hence the traceback.
- **Item 2** (J013301m400628). Excluding H I from a1 comments out its line and, with
  a note, the link `nda(nha,FIXDH) = nha + FIXDH`. D I's log N is then free, and
  excluding D I is `last`, which offers only removal.
- **Item 12.** b_turb = 0 fixed, with T free, makes ALIS stop: "A parameter that = 0.0
  is not within the specified limits" (voigt.py's lower limit is 0.5 km/s). With
  `lim param BFIX [0.0,None]` the fit runs, with b_turb kept at 0.
- **Item 11.** `temperature_mode` is "thermal" only for b_turb fixed at exactly 0, so
  RJC's `0.5BFIX` reads as "free". `drag` then moves b_turb, which `_check_drag`
  allows because it refuses only linked parameters.
- **Item 14.** A synthetic saturated Lyα (log N 15.5, b 20 km/s, 800 pixels, σ = 0.01)
  with a zero level of 0.05, fitted by `run_alis` from `constant 0.0` in the
  `zerolevel` section, gave 0.0536 ± 0.0029. ALIS's own zero level works, so the
  fault, if any, is in the dashboard's path or in how it was used (Q7.11).
- **Item 8.** Off-screen, after J013301m400628.dat is chosen, the dialog grows from
  233 to 349 px. The table (y 95–265) and the hint (y 241–286) overlap, as RJC saw.
- **Items 3, 7, 9, 10 and 13.**
  - The two `.mod` items differ only by whether the open project has a source
    spectrum.
  - `Recent` has `remove` but no `clear`, and the start page has no buttons for
    either.
  - The shift and the zero level are per row (D14): `snips.row_shift`, and one
    `constant` per row, with ALIS allowing one zero level per specid.
  - `SpectrumView.set_normalised(None)` already draws data as they are.
  - `fit.run` is a shared action, so the toolbar can carry it.

**The document.**
- The Design section gains "7.2–7.14": what was found, and what stays as it is.
- Tasks 7.2–7.14 are written in the order: exclusions, b_turb and T, the zero level,
  the smaller fixes and changes, the new features, the documentation's design. Each
  names its item, with its check.
- 7.N became 7.15, with the checks that close the stage.
- Q7.6–Q7.13 ask about the decisions, and the tasks follow their leans:
  - Q7.6: a component excluded from every ion;
  - Q7.7: what a card shows for a ratio;
  - Q7.8: the label written by an exclusion;
  - Q7.9: one item to open a `.mod`;
  - Q7.10: pure thermal's limit, what counts as pure thermal, and what a drag moves;
  - Q7.11: RJC's steps for the zero level;
  - Q7.12: a snip's own shift and zero level;
  - Q7.13: where the documentation's design document goes.
- `build-docs` was added to the skills, and the Temperature examples to the context.

No code was changed, so no batch was run. Task 7.2 waits for RJC's responses.

### 2026-10-31 (Prompt 3: RJC's responses to Q7.6–Q7.13, the tasks revised, Q7.14–Q7.15)

**Reading.**
- RJC's responses to Q7.6–Q7.13. All accept the leans, and add:
  - Q7.6: an excluded component's labels must not be used by another component;
  - Q7.8: the label written must be unique;
  - Q7.9: with no source spectrum, ask first whether to load it, and recommend it;
  - Q7.10: check that ALIS still runs, with no side effects;
  - Q7.11: the zero level was typed in the `.mod` panel; the project is
    `context/fitting_examples/Temperature/J193957m100241/model/J193957m100241.model`
    (the response's path leaves out `fitting_examples/`);
  - Q7.12: a loaded `.mod`'s snips must allow their own shift and zero level.
- RJC's project: its model, its runs and `runs/last_error.json`, and its
  `ui/project.json`.
- The code: `preflight._limits`; `edit.used_labels`, `label_uses`, `rename_label`,
  `fix`, `untie` and `set_limits`; `edit.zero_level_line` and `Project`'s reading of a
  row's zero level; `results.at_limit` and the results' rows; `constant.py`'s and
  voigt.py's limits; `model_eval`'s zero level; `load.load_par_influence`.

**Facts that changed the tasks.** Scripts in the session's scratchpad.
- **Item 14.** The last run of RJC's project left its four zero levels at exactly 0,
  with zero variance, though every fit region has dark pixels. `run_alis` outside the
  dashboard gives the same, so the cause is in ALIS. Cut down to Lyα alone, it
  persists with `fitrange=all`, with no continuum column, and from a start of 0.01,
  which ends at 0. A legendre continuum in the synthetic test still fits.
  - The cause: `constant`'s value has `_limited = [1, 0]`, `_limits = [0.0, 0.0]`, a
    lower limit of 0 that a zero level inherits. The troughs lie below zero.
  - With `lim param zla [None,None]`, Lyα's zero level is −0.01357737 ± 0.00088546,
    and χ² falls from 918.8 to 695.5.
  - The results table did flag each zero level "at limit (0)", shown in the collapsed
    Instrument group. The pre-flight check's note says "value on line 16 starts at one
    of its limits."
- **Per-snip zero levels.** `Project` reads RJC's row as having no zero level, since
  `zero_level_line` wants a `constant` covering all four snips.
  `zero_level_step(…, True)` is accepted, and adds a fifth `constant`. ALIS stops on
  that ("You can only specify the zero-level once for each specid",
  `model_eval.py`).
- **Item 12.** Prompt 2's log said the pre-flight check refuses pure thermal; it does
  not. The dashboard's own pure-thermal step writes `0.000BA1`. The pre-flight check
  finds no refusal, since `_limits` skips fixed parameters, and `run_alis` stops ("A
  parameter that = 0.0 is not within the specified limits").
  - With `lim param BA1 [0.0,None]`, the fit runs: T = 22,851 K, with b_turb at 0.
  - With T also at 0, ALIS stops: "Voigt Total Doppler parameter is zero" (voigt.py,
    T's lower limit 0 K). This is RJC's saved last error.
  - The pre-flight check and the validator accept `lim param <label> [0.0,None]` and
    `lim voigt bturb [0.0,None]`, on RJC's model and on synthetic ones. What rejected
    RJC's limit is not certain; the saved error is the likeliest.
- **Labels.** `used_labels` already includes commented-out lines' labels, so new
  components and Q7.8's label are unique. The four checks of a typed label use
  `ParsedModel.label_uses`, which sees only live lines. `edit.label_uses`, used by a
  rename, does not touch commented-out lines either.

**The document.**
- The Design's "What was found" corrects item 12 and replaces item 14, and adds the
  per-snip zero levels and the labels of excluded lines. "What stays as it is" allows
  ALIS's zero-level limit to change (Q7.14).
- The tasks are revised:
  - 7.2 writes a new label;
  - 7.4 keeps an excluded component's labels its own;
  - 7.5 adds T's floor (Q7.15) and the pre-flight check's refusal of a fixed value
    outside its limits, with a run through the Fit tab as its side-effect check;
  - 7.6 adds RJC's project;
  - 7.7 is rewritten: the zero level's lower limit, in ALIS (Q7.14), and the
    dashboard's names and flags;
  - 7.9 asks first, recommending the spectrum;
  - 7.13 reads a snip's own shift and zero level from any model;
  - 7.15 records Q7.1–Q7.15, and runs the regression harness for 7.7.
- Q7.14 (where the zero level's limit is lifted) and Q7.15 (pure thermal and a total b
  of 0) are new, with leans. The Context names RJC's project.

No code was changed, so no batch was run. Tasks 7.2–7.4 can start now. Tasks 7.5 and
7.7 follow the leans of Q7.15 and Q7.14 unless RJC responds otherwise.

### 2026-10-31 (Prompt 4: RJC's responses to Q7.14–Q7.15; the tasks are ready)

**Reading.** RJC agreed to both leans: (a) for Q7.14, so ALIS lets a zero level go
below 0, and (a) for Q7.15, so pure thermal gives T a floor of 1 K. Nothing else in the
document had changed. Q7.15's response, written on the same line as the lean, was put
on a **Response:** line of its own, with RJC's words unchanged.

**What was checked before saying the tasks are ready.**
- **Where Task 7.7 changes ALIS.** `load.py` gives a line its function's limits twice:
  `call_function_load` when the line is read, and then the pass that applies
  `lim <function> <parameter>` commands. That pass calls `adjust_lim` for every line
  whose limits differ from its function's, so a limit dropped only at load would be
  put back. The task now names both places. Prompt 3's run with
  `lim param zla [None,None]` showed that a `lim param` on a zero level's label
  applies after both.
- **Which functions.** Many functions have a lower limit of their own: `linear`'s
  intercept, `powerlaw`'s amplitude, `gaussian`'s amplitude and width, and others.
  Only `constant` changes, as Q7.14 agreed. A `linear` zero level's intercept keeps its
  limit of 0; that is left for RJC to raise if a sloped zero level is wanted.

**The document.**
- The notes on the Design, the Tasks and Q7.14–Q7.15 now record the responses.
- Task 7.5: an existing `lim param` of T's label has its lower bound raised to 1 K.
- Task 7.7: the place in `load.py`, that only `constant` changes, and a check that a
  `lim param` on a zero level still holds.
- No other task changed. Tasks 7.2–7.15 are ready to execute in order.

No code was changed, so no batch was run.
