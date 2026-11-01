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

### 2026-10-31 (Prompt 5: Tasks 7.2–7.15)

RJC asked for the tasks to be executed in order, pausing only for queries.

**Task 7.2 — Excluding from a component with no redshift label (item 6).**
- **The tests first.** `tests/dashboard_synthetic.py` (new) writes a fit in
  J101155p294141's shape: H I components with unlabelled redshifts, D I from one ratio
  line, on a flat Lyα snip that ALIS reads. Four tests in
  `test_dashboard_components.py` failed with RJC's error before the change:
  - excluding D I (the ratio) from a1 ties a1 and comments out the ratio line, with
    a2 named as sharing it; Include, and the undos, are exact;
  - excluding H I, a1's last line, asks as for any last ion;
  - a component of two untied lines (H I and D I) is tied and excluded in one step;
  - with `za1` on a commented-out line, the label written is `za2`.

  The old `test_exclude_needs_a_redshift_label`, which asserted the error, was
  replaced by them. A window test (`test_dashboard_qt_components.py`) clicks Exclude
  from H I on the two-line component: the preview is shown, then one step, and no
  message.
- **The change.**
  - `project.component_label` (new) gives the label `tie_component` ties to, and
    `tie_component` uses it. A new label comes from `edit.new_component`, which skips
    labels on commented-out lines.
  - `components.exclusion` no longer refuses a component without a shared redshift
    label. `_tied_first` makes the tie, and the lines are commented out of the tied
    text, in one change. The preview lists the lines that gain the label ("edit")
    with those commented out.
  - `qt/components._toggle_ion` shows a `ComponentError` from Exclude or Include as a
    message, as the Regions tab shows its refusals.
- **Found for Task 7.4.** A card has one Exclude button, for the ion shown. An isotope
  under it (D I under H I) has no button of its own, so the window gives no way to
  exclude D I alone. Task 7.4 adds one.
- **Checks.** `test_dashboard_components.py`: 34 passed;
  `test_dashboard_qt_components.py`: 30 passed.

**Task 7.3 — Ions given by a ratio, on the cards (item 5).**
- **The tests first.** In `test_dashboard_components.py`:
  - on the synthetic J101155p294141 fit, `card_ions` gives a1's D I as a ratio, the
    ratio is shared with a2, and a ratio is not dragged;
  - with D I excluded from a1, it stays among the system's ions and reads excluded
    in both components. Included from a2, whose exclusion it was not, the ratio
    line comes back.

  In `test_dashboard_qt_components.py`: a1's card has a "↳ D I · log D I/H I" row
  with the ratio's value, a Fixed badge and "shared with a2", and no "Excluded" line.
  Typing a value is one step, seen on a2's card too. Excluded, the card reads
  "Excluded: D I", and a click includes it. A blinded ratio line shows "▒▒▒▒". That
  test is in the components window tests, beside the others on this fit, rather
  than `test_dashboard_qt_blinding.py`.
- **The change.**
  - `components.card_ions` (new): each ion of the system as a card shows it, with
    `ratio_line` and `ratio_sharers`.
  - `_ratio_ions` also counts ratio lines that an exclusion commented out, so D I
    stays on the cards.
  - `include` brings back a ratio line commented out by another component's
    exclusion when this component shares it (`_shared_ratios`). Without that, a
    click on a2's "Excluded: D I" would have added a new D I line to a2.
  - `include` restores each noted line with its own note, rather than one built
    from the component's label.
  - The card builds a ratio row from `card_ions`.

**Task 7.4 — Excluding a component from every ion (item 2).**
- **The tests first** (`test_dashboard_components.py`).
  - `dashboard_synthetic.write_linked_fit` (new) writes J013301m400628's shape: D I's
    log N computed from H I's by a link.
  - Excluding H I, then D I, from a1 gives two steps that ALIS reads. The first
    preview says D I's log N becomes free; the second says a1 is then excluded from
    every ion. a1 keeps its id and redshift, with no live line.
  - Include, in either order, restores the text byte for byte, and each undo too.
  - While D I is excluded, including H I leaves the link out, with its note moved
    to D I's exclusion.
  - A component commented out by hand reads as excluded from every ion. Include
    brings it back, and ✕ removes it.
  - The labels stay a1's (Q7.6's response): they are reserved, a new component
    skips them, and a rename, untie or new label is refused with `ExcludedLabel`.
    `label_owner` names a1. Typed on a live line in the `.mod` panel, the label gets
    a warning, and Include is refused.
  - A rename of a1's redshift label while H I is excluded renames the excluded line
    too, and Include brings it back to a1.
  - J013301m400628 itself (skipped when absent): a1 excluded from both ions, ids
    kept, and both included again with the text restored.

  Two older tests asserted D64's removal of the last ion. They now check that ✕
  deletes the commented-out lines, and that the last ion leaves the component
  excluded from every ion. In the window: the last ion's preview, the greyed card
  with "Include in O I" and ✕, and item 2's flow on the linked fit.
- **The change.**
  - `project.py`:
    - a commented-out line's redshift is read (`Commented.z`);
    - `_excluded_components` makes a component for commented-out voigt lines whose
      redshift label no live line carries;
    - `Component.main` is None for such a component, and `everywhere_excluded`
      says so;
    - `temperature_mode`, `_z_hidden` and `lines_of` handle it.
  - `components.py`:
    - `exclude` no longer refuses the last ion;
    - `Exclusion` notes what a link commented out leaves free, and that the last
      ion leaves the component excluded from every ion;
    - `include` keeps out a line that names a label carried only by a line still
      excluded, moving its note (`_kept_out`);
    - for a component excluded from every ion, `include` refuses while a live line
      carries one of its free labels (`_check_untaken`). Fixed labels, such as the
      forest's `ZEROT`, are shared on purpose, so they are left out;
    - `add_ion`, `temperature` and `_agree` skip or refuse such a component.
  - `edit.py`:
    - `line_labels`, `live_labels` and `reserved_labels` (new), with
      `commented_labels` built on them;
    - `check_new_label` refuses a label only commented-out lines carry
      (`ExcludedLabel`), for `rename_label`, `fix`, `untie` and `set_limits`;
    - `rename_label` also renames the label on commented-out lines, their notes
      included.
  - `validate.quick` warns when a free label of a component excluded from every ion
    is on a live line (`_excluded_ties`). The `.mod` panel's quick checks see only
    the parsed model, so the warning names the component by its redshift label.
  - `blinding.Gate` masks such a component's velocity when one of its lines is
    hidden. `remove.plan_system` also deletes its components' commented-out lines.
    The blind-lines dialog skips a component with no line.
  - `qt/components.py`:
    - a card has a toggle for each ion of its navigator row (H I and D I), in the
      row above the body, so that D I can be excluded while the card is greyed for
      H I. This gives the window a way to exclude D I alone, which it lacked;
    - a component excluded from every ion has a greyed card, with an Include for
      each of its ions and ✕;
    - excluding the last ion shows the preview, not the removal question;
    - a label refused is a message, naming the excluded component;
    - the badge menu's Tie to leaves out such a component.
- **Checks.** `test_dashboard_components.py`: 45 passed;
  `test_dashboard_qt_components.py`: 33 passed. `pytest -m unit`: 2043 passed, 119
  skipped (from 2029 after Task 7.1).

The gui batch after Task 7.4: 196 passed (363 s), from 192 after Task 7.1.

**Task 7.5 — Pure thermal that ALIS fits (item 12).**
- **The tests.**
  - In `test_dashboard_components.py`:
    - pure thermal writes `lim param BA1 [0.0,None]` and `lim param ta1 [1.0,None]`,
      and passes the pre-flight check;
    - Free T, preset 0 and fixed 8000 K each take both limits out, with b_turb back
      at 0.5 km/s or more, and no refusal;
    - a T limit the user wrote above 1 K stays, through pure thermal and out;
    - a b_turb tied across two components gets one limit;
    - an unlabelled component is tied first, with `ba1` and `ta1` of one number;
    - `0.5BFIX`, and `0.01BX1` with RJC's `lim param BX1 [0.01,None]`, read as pure
      thermal.
  - `test_dashboard_preflight.py`: a b_turb fixed at 0 with no limit is refused
    (`fixed_outside_limits`), and passes with the limit.
  - `test_dashboard_qt_fit.py`, the side effects (Q7.10's response), as the Fit tab
    runs a fit. `dashboard_synthetic.write_thermal_fit` (new) makes Lyα and Lyβ of
    H I and D I with `scipy.special.wofz`, at T = 15000 K with no turbulence, beside
    a turbulent H I component.
    - Pure thermal runs to completion. T is within 3σ of 15000 K, with b_turb at 0
      and no flag on either.
    - Continue from best fit keeps both limits, and the turbulent component keeps
      voigt's 0.5 km/s.
    - Started at 1 K, on its floor, the fit also completes. With `run_alis` alone,
      the same fit gives T = 15001 ± 40 K.
- **The change.**
  - `components.temperature`, in pure thermal, ties untied b_turb and T first. The
    scheme's labels are taken once, so that they share a number. It then writes the
    two floors last (`_floor`), since a command line moves the lines after it;
    `_here` finds a parameter through the chain's line map.
  - Out of pure thermal, the limits it wrote go if unchanged (`_unfloor`). b_turb
    then keeps voigt's own lower limit (`_own_low`), not pure thermal's 0.
  - `Component.temperature_mode` reads a b_turb fixed at 0.5 km/s or less, with T
    free, as pure thermal.
  - `project.tie_component` takes a label.
  - `preflight`: `_fixed_outside` refuses a fixed value outside its limits, unless
    `run limpar` is on, since `minimise.py` moves fixed values too. A hidden value's
    message gives no value or side. `_zero_b` refuses a line whose b_turb and T
    both start at 0, ALIS's "Voigt Total Doppler parameter is zero". A b_turb tied
    into a component whose T is 0 led to that while the tests were written.
- **Not done:** an existing `lim param` of b_turb has its lower bound set to 0.0. On
  leaving pure thermal it is then taken out, as written by pure thermal, so a user's
  lower bound there is not restored; one undo restores it.

**Task 7.6 — A drag moves the free one of b_turb and T (item 11).**
- **The tests.** In `test_dashboard_components.py`, for b_turb fixed at 0 or 0.5
  with T free, T fixed, both free, and both fixed: the parameter `drag` changes, and
  the total b at the end is the b dragged to within 10⁻⁶ km/s. With both fixed,
  `drag` refuses a drag across, gives the reason, and still moves log N. In the
  window: a drag on a `0.5BFIX` line writes T as one step, with T in the readout.
  On RJC's J193957m100241 project (skipped when absent), a drag on the `0.01BX1`
  component writes T and holds b_turb.
- **The change.** `components.drag_target` (new) gives the parameter a drag across
  moves: b_turb if free, else T if free, else nothing, with the reason. `drag` and
  `draggable` use it. With b_turb held, T comes from b² − b_turb², not b² alone.
  The window says the reason in the status bar when a drag across can move nothing.

**Task 7.7 — A zero level below 0 (item 14).**
- **In ALIS** (`alis/load.py`). A `constant` read in the `zerolevel` section has its
  lower limit taken out of `modpass['mlim']`. The pass that applies `lim` commands
  then leaves it out too, unless a `lim constant value` command set constant's
  limits; `funclim` records such commands. A `lim param` on the zero level's label
  still holds, as it applies after both passes. RJC's Q1243p307 and J0814p5029 had
  worked around the trap with `lim constant value [None,None]`, which still
  applies.
- **The tests.**
  - `test_load_model.py` (unit): the zero level has no lower limit and an emission
    `constant` keeps its 0; `lim param zl [0.0,None]` and `lim constant value
    [-0.5,None]` still bound it.
  - `test_zerolevel_fit.py` (new, `fast`): saturated troughs with zero levels of
    −0.03 and +0.05, fitted by `run_alis` from `constant 0.0`, give each within 3σ.
    `dashboard_synthetic.write_trough` writes them, as `model_eval` applies a zero
    level.
  - `test_dashboard_runs_process.py` (`fast`): the −0.03 trough, with the Data
    tab's Zero level On and with the line typed. The results row is
    "Zero level · sat", with no flag and within 3σ. The model drawn for Final goes
    below zero, and Continue from best fit writes the value.
  - RJC's J193957m100241 (skipped when absent): its four zero levels leave 0, with
    no flag, and χ² falls below its last run's 3659.
  - `test_dashboard_preflight.py` and `test_dashboard_results.py`: a zero level is
    named "The zero level of <dataset> (line N)", and a flag on a function's own
    limit reads "at limit (0.5, voigt's own)".
- **The run.** RJC's model cut to Lyα, under `run_alis`: the zero level is now
  −0.01357737 ± 0.00088546 and χ² 695.5, as with the limit lifted by hand.
- **The dashboard.** Limits come from ALIS's own parameter info (`model.py`), so the
  dashboard follows with no change. `Project.zero_level_name` (new) names a zero
  level by its dataset, and by its snips' labels when it covers only some of them.
  The pre-flight check's `_name` and the results' rows use it, and the results'
  `own_limit` (new) names a function's own limit in the "at limit" flag.

**Task 7.8 — The Add a spectrum dialog (item 8).**
- **The change.** Once a file is read (or refused), `SpectrumDialog._fit_contents`
  sizes the dialog by its layout's height for its width. The wrapped labels' heights
  depend on the width, which is what `adjustSize` misjudged. A
  `SetMinimumSize` constraint was tried first: it fixed the overlap but overrode the
  640 px minimum width, so it was dropped. A guard covers the file field's
  `editingFinished`, which fires as the dialog is deleted.
- **The test.** `test_dashboard_qt_window.py`: with J013301m400628.dat (skipped when
  absent) and synthetic files of 3 and 5 columns, the table and the hint do not
  overlap, and both are inside a dialog 640 px wide or more. An unreadable file
  hides the table, and the dialog shrinks. Off-screen with J013301m400628.dat, the
  dialog is now 640 × 431: the table spans y 95–287 and the hint starts at 293.

**Task 7.9 — One way to open a .mod (item 3).**
- **The change.**
  - The File menu has one item, Open a fit (.mod)… (`file.import`, Ctrl+I).
  - With a project that has a source spectrum, it previews loading the fit into
    it. `PlanDialog`'s new `other` button offers Open as a new project….
  - Otherwise it asks first, the recommended way first and as the default (Q7.9's
    response):
    - Open with its spectrum (recommended)… opens the New project dialog in the
      folder of the `.mod`'s snips, with its FWHM (`loadfit.mod_hints`, new). The
      project is made with `Session.new(save=False)`, and written only once the fit
      is loaded, so that Cancel leaves no file;
    - Open without a spectrum imports it as before.
  - A refusal says why, and offers a new project, or another spectrum, or none.
  - The start page's button (the same action) and `alis fit.mod` (`launch`) ask the
    same question.
  - Load a fit (.mod) into this project… stays in the `.mod` panel's right-click
    menu. It is renamed `panel.load_mod`, as the panel's actions are named.
  - `do_file_new` shares `_new_session` with it.
- **The tests.**
  - `test_dashboard_actions.py`: the File menu's one item, and the panel's action.
  - `test_dashboard_qt_regions.py`:
    - Task 7.1's test now uses Open a fit: its preview shows Open as a new
      project…, and the panel's item still loads;
    - with no project, the question: Cancel opens nothing; without a spectrum, as
      Import made it; with its spectrum, the dialog in the snips' folder at the
      `.mod`'s FWHM, then the preview, and the file written. Cancel at the preview
      leaves no file;
    - with a project of another spectrum, the reason; Open as a new project… asks
      the question, and so does `alis fit.mod`.

**Task 7.10 — Clearing the recent projects (item 7).**
- **The change.**
  - `preferences.Recent.clear()` (new).
  - The start page's Recent projects pane has Remove from list (also Delete, and
    the right-click menu) and Clear list…, which asks once ("The projects
    themselves are not changed"). Both are disabled when there is nothing to act on.
  - File ▸ Open recent ends with Clear recent projects….
  - Only `recent.json` changes.
- **The tests.** `test_dashboard_preferences.py`: `remove` and `clear` write the
  list, and no other file in the folder changes. `test_dashboard_qt_window.py`:
  - the button, Delete and the menu item work, and Cancel keeps the list;
  - the empty list reads "No recent projects.", with both buttons disabled;
  - no project's file is touched.

**Task 7.11 — Run on the toolbar (item 13).**
- **The change.** The Fit tab's `fit.run` action itself goes on the toolbar after
  Save and its autosave time, before the pill, marked `toolbar=True`. It is enabled
  as the Fit tab's Run is, and keeps its tooltip and Ctrl+R. Its icon text was not
  changed, since the Fit tab's button shows the same action.
- **The tests.** `test_dashboard_actions.py`: the toolbar's actions. In
  `test_dashboard_qt_actions.py`:
  - the toolbar's first five actions are Undo, Redo, Open, Save, Run;
  - (`fast`) Run on the toolbar starts a fit from the Data tab, is disabled while
    it runs, and enabled after.

**Task 7.12 — Normalised or with the continuum, in Inspect (item 10).**
- **The change.**
  - `fitview.NORMALISED`, `CONTINUUM` and `scale` (new).
  - Inspect's View pane has a Normalised | Continuum segment between Model and
    Residuals, kept in the view state (`scale` in `VIEW_KEYS`). Continuum draws the
    data as they are and ALIS's model with its continuum, with the continuum dashed
    (`set_continuum`), and the y range follows the data. The residuals and the
    marks do not change.
  - A checkable `fit.normalised` action (Fit tab, no key) is listed in the shortcut
    sheet, and follows the segment. `FitTab.bind_action` gives it to Inspect rather
    than making it a ‹/› button.
- **The tests.** `test_dashboard_fitview.py`: `scale`. `test_dashboard_qt_fit.py`
  (`fast`), on metal_line_abs after a run, for Initial and Final:
  - with Continuum, the view holds the flux and ALIS's model undivided, the model
    drawn equals it, and the continuum is drawn;
  - Normalised draws the model divided by it;
  - the marks and the residuals are the same in both;
  - the choice is kept when the tab is left and shown again, and the action
    switches back.

**The regression harness, for Task 7.7.**
- `pytest tests/test_regression.py -m fast` (its `test_bundle` mode included): 96
  passed (707 s).
- The medium cases with a zero level, J0814p5029 (DH), J1358p0349,
  J1358p6522_original and Q1243p307: 6 passed (887 s).
- The slow DH_orders Q1243p307 was not run: it sets
  `lim constant value [None,None]`, which applies as before.
- Every other reference zero level is above 0 or fixed, so no reference fit changed.

**Task 7.13 — A snip's own shift and zero level (items 9 and 14).**
- **What is read** (`project.Project._own_levels`, new). A snip set to its own in
  the Snip box is recorded in `ui/project.json` (`snips[specid]["own"]`). Otherwise
  the model is read:
  - **Zero levels.** The dataset's is the `constant` that covers all its snips not
    recorded as having their own; every other `constant` touching the row is its
    snips' own (`Row.zero`, `Row.own_zero`).
  - **Shifts.** Snips share one by its label; an unlabelled fixed one by its value
    (the reference's `vshift(0.0)`, and ALIS's own `Ashift` 0 on a line with none);
    an unlabelled free one never. The dataset's is that of the most snips not
    recorded, and a different one is a snip's own (`Row.shift`, `Row.own_shift`,
    `Row.shared_snips`).

  RJC's J193957m100241 reads with each of its four snips' own zero level.
- **The steps** (`datasets.py`).
  - `snip_shift_step` writes `vshift(0.0vsn<n>)` on the snip's line, or gives it
    back the dataset's word (none for the reference).
  - `snip_zero_step` takes the specid out of the dataset's `constant` and adds one
    of its own, or the reverse. The `zerolevel` header goes with its last line.
  - The dataset's FWHM and shift steps, badge and ties use the dataset's snips
    alone.
  - Zero level On gives the row's `constant` only the snips with none of their own,
    and is refused when none is left (`EVERY_SNIP_ZERO`).
  - `own_count` gives the Data tab's counts.
- **What follows.**
  - SNIP gives a new snip the dataset's shift (`_row_words`) and zero level
    (`zero_level_of`), not a snip's own.
  - Regions copied or tweaked between datasets move by each snip's own shift
    (`snips.snip_shift`, new). `row_shift` gives the dataset's.
  - CLEAR takes a snip's own zero level (its specid is dropped from every line) and
    its shift (with its data line).
- **The window.**
  - The Snip box has Shift and Zero level, each As the dataset or Its own. Its own
    shift is typed (one step) and badged Free or Fixed. Its own zero level shows its
    value, and the snips it is shared with.
  - The Data tab's shift and zero-level cells add "· 1 snip its own" or
    "· 4 snips their own"; Zero level's tooltip and click say when every snip has
    its own.
- **The tests.**
  - `test_dashboard_datasets.py`:
    - own shift and zero level: ALIS reads each step, with the specid in one
      `constant` only, and the dataset's value leaves the snip's own as it is;
    - back to As the dataset gives the text and `ui/project.json` byte for byte,
      and every step undoes exactly;
    - a reference snip's own shift;
    - regions copied move by a snip's own 30 km/s;
    - SNIP of O I 1302 for a second system gets the dataset's shift and zero level,
      and CLEAR takes a snip's own;
    - J193957m100241 (skipped when absent): four own zero levels, and On refused;
    - a project's own dataset shift and zero level read as before.
  - `test_dashboard_loadfit.py`: a `.mod` with `vshift` and a `constant` on one of
    a row's two data lines, loaded with Load a .mod…, reads them as that snip's own.
  - `test_dashboard_runs_process.py` (`fast`): two synthetic datasets
    (`dashboard_synthetic.write_shifted_fit`, new), B's Lyβ snip shifted by 3 km/s.
    Given its own shift, the fit gives it back within 3σ.
  - `test_dashboard_qt_regions.py`: the Snip box's two choices, typing, the Data
    tab's counts, and back to the text as it was.
- **Checks.** `pytest -m unit`: 2072 passed, 119 skipped.

**Task 7.14 — The documentation's design document (item 4).**
- **Written:** `claude_prompts/documentation_stage0.md` (Q7.13), in the pattern of
  `dashboard_stage0.md`:
  - what exists: each chapter of `doc/tex_files/` and what has changed since v1.0,
    `to_be_completed.txt`, `doc/ALIS_workflow.md`, `doc/installing.rst`, the
    docstrings, the examples, and that there is no Sphinx configuration yet;
  - the readers: a new user, a user of v1, a modeller, a developer;
  - a 17-chapter outline, with each chapter's sources and what is new;
  - the reference lists of functions and settings, to be made from the code and
    checked in CI;
  - the tools (Sphinx with MyST, autodoc for the developers' chapter, pandoc for
    the v1.0 text) and how the build is checked (`-W`, linkcheck, the lists, the
    `.mod` blocks parsed);
  - the screenshots, made by one script from the review pages' scripts;
  - five Documentation Stages, each with its check;
  - queries DQ.1–DQ.7, each with a lean.
- **Check** (a script in the session's scratchpad):
  - the 14 chapter files `alis_v1p0.tex` inputs, commented out or not, are all in
    the document;
  - so are the 32 functions `build_funcarray` registers, the 30 modules of
    `alis/functions/`, and the 83 settings of `config.py` in 8 sections.
- No code was changed.

**Task 7.15 — Close the stage (draft 1 of the review).**
- **The review page.**
  - `doc/dashboard/stage7/take_screenshots.py` and `build_review.py` (new). The
    screenshots are off-screen at 1440×900, of synthetic fits in the shapes of RJC's
    Temperature examples and a real fit of metal_line_abs. The page uses the earlier
    pages' stylesheet, with one figure per item and what to check.
  - Published privately, draft 1: https://claude.ai/artifact/HJQxjgvzAs6AtHd7TnxWKg
    (recorded in Claude's memory with the other review pages). RJC's comments are to
    be applied when they come.
- **The documents.**
  - `doc/ALIS_workflow.md` §4.3: "Since version 1", and `alis fit.mod`.
  - `CHANGELOG.md`: Added (the dashboard's Stage 7) and Fixed (the zero level, and
    the dashboard's bugs).
  - `tests/README.md`: a row for Stage 7's tests and `dashboard_synthetic.py`.
  - The `gui-component` skill: reserved labels, refusals as messages, components
    with no live line, a row's own shifts and zero levels, `PlanDialog`'s `other`.
- **The decisions.** D78–D87 in `ALIS_v2_dashboard_prompts.md`, from Q7.1–Q7.15,
  each naming the decision it refines (D14, D21, D35, D64, D72).
  `dashboard_stage0.md`'s "After v1": Stage 7 done; S23 off (Q7.5, D78); the
  documentation added as its own stages (D87).
- **Four gui tests changed with the stage**, each checking the new behaviour:
  - the action registry's placement of `panel.load_mod` and `fit.normalised`;
  - the Open recent menu's last entry;
  - `alis fit.mod` asking first (`test_a_mod_is_imported`).
- **test-coverage** on the changed modules (components, datasets, edit, fitview,
  loadfit, preflight, project, validate): 90%.
  - The new code's untested lines were the refusals. Tests were added for a second
    Its own (shift and zero level), and for a drag across a linked b_turb with T
    fixed.
  - What stays untested of the new code: `label_owner` finding no component, and
    `_shared_ratios`' skipped lines.
- **Style.** ruff passes on every changed file. The changed files were black-clean
  before the stage and are again. `alis/load.py` was left out: it has never been
  formatted with black.
- **The batches.**
  - `pytest -m unit`: 2073 passed, 119 skipped, from 2002 at Prompt 1's baseline and
    2029 after Task 7.1.
  - `pytest -m gui`: 206 passed (457 s), from 190 at the baseline and 192 after Task
    7.1.
- **D44.** The window's strings are checked for design references by the gui tests
  (`design_references`), the new ones included; none.
  - `pytest -m fast` (the regression harness's fast batch, its `test_bundle` mode,
    and the dashboard's fits): 143 passed (962 s).
- **What remains:** RJC's review of the page, and applying it. The stage closes
  then.

### 2026-11-01 (RJC's review of the page)

- **The View pane's drop-downs** (RJC's comment on Inspect with the continuum: the
  row of toggles under View was crowded).
  - `qt/fit.py`: the four segments become drop-downs (`_choice_box`), in the same
    order: Velocity / Wavelength, Initial / Final, Normalised / Continuum, and All
    Residuals / Fit Residuals. The Residuals label is gone. Each item keeps its
    tooltip, and the box's tooltip is its choice's.
  - Final, with no run to show, is a greyed item whose tooltip gives the reason, as
    the greyed button did.
  - `InspectTab.view_boxes` (keyed axis, model, scale, residuals) replaces
    `view_buttons`; `select(key, value)` and `selected(key)` are new. The Progress
    pane's Linear | Log stays a segment.
  - D84 records it.
- **The tests.** `test_dashboard_qt_fit.py` uses `select`/`selected`;
  `test_the_view_choices_are_drop_downs` (new) checks the four boxes' items, that
  no Residuals label is left, and that a choice is shown, kept and drawn. The
  Stage 6 and 7 screenshot scripts use `select`.
  - `test_dashboard_qt_fit.py`: 21 passed, the fast ones included.
- **The page.** Its Inspect figure was retaken and the page republished at the same
  URL (version 2).
- **The cards' lists of ions** (RJC's comment on A component excluded from every
  ion: a button per ion would crowd a card of many ions).
  - `qt/components.py`: the Exclude from / Include in buttons (`_toggles`) and the
    "Excluded:" line (`_excluded_line`) are gone. Under the parameter rows, each
    behind a faint rule (`QFrame#cardRule`, in `style.py`), come "Included ions
    (click to exclude):" and "Excluded ions (click to include):", each followed by
    its ions as links, in the navigator's order, with the ion shown in bold
    (`_ion_sections`, `_ion_list`). An empty list reads "none".
  - Included ions are those with a line of their own or given by a ratio; Excluded
    ions are those commented out or never in the component. A click goes through
    `_toggle_ion`, so the previews and refusals are as before.
  - The lists are outside the card's body, so they stay live when the card is
    greyed for the ion shown. A component excluded from every ion lists all the
    system's ions under Excluded ions.
  - `card.included` and `card.excluded` replace `card.toggles`, `card.toggle` and
    the old `card.excluded`. D79 records it.
- **The tests.** `test_dashboard_qt_components.py`: the card tests click the lists
  (`click_ion`); `test_the_lists_exclude_and_include_an_ion` replaces the Excluded
  line's test and excludes Si II from the card while O I is shown. 35 passed. The
  Stage 5 screenshot script clicks the list.
- **The page** was rebuilt with the cards retaken and republished (version 3).
- **The batches.** `pytest -m gui`: 207 passed after the drop-downs (405 s), and
  207 again after the cards' lists (412 s).
- **Run greyed when nothing can run** (RJC's comment: Run should be greyed when a
  model cannot run, as with no snips, no regions or no components).
  - `preflight.py`: two refusals, first among the check's own, `no_snips`
    ("There is no snip to fit yet: cut one on the Regions tab.") and
    `no_components` ("There is no component to fit yet: add one on the Components
    tab.", when the absorption section has no live line). A snip with no fit
    region was refused already (ALIS's loader, through the validator). A
    continuum alone is no longer fitted from the window.
  - `qt/window.py`: `run_refusal()` gives the first refusal of the Inspect
    sub-tab's check (kept until the project, the preferences or the validator's
    problems change), through the gate. `_fit_actions` greys Run, Save and Run and
    Convergence check… with it as their tooltip, and gives back each action's own
    tooltip (kept as its `tip` property, with the shortcut) when a run can go.
    `refresh` calls `_fit_actions` after the tabs, so that Inspect has the session.
    The Fit marker now shows a refusal before Inspect is first shown, since the
    check is made with every refresh.
  - Cost: the check takes 0.25 s on Q1243p307 after a change (the profile
    rebuilt), and 0.05 s when only the validator's problems changed.
- **The tests.** `test_dashboard_preflight.py`: `test_no_component_to_fit`,
  `test_a_new_project_has_no_snip_to_fit`. `test_dashboard_qt_fit.py`:
  `test_run_is_greyed_while_there_is_nothing_to_fit` (no snip, then a snip with no
  fit region) and `test_run_waits_for_a_component` (greyed on the toolbar, the Fit
  marker, enabled with its own tooltip after a component, greyed again on Undo).
- **Run greyed: the batches and two fixes.** `pytest -m unit`: 2075 passed, 119
  skipped. `pytest -m gui`: 207 passed, 2 failed, both from the change:
  - `test_run_is_refused_with_a_problem` clicked Run to see the refusal; it now
    checks that Run is greyed with the reason, and that `start_run` still refuses.
  - `test_an_empty_project_opens`: a new project's Fit marker became "attention".
    The Fit marker now leaves `no_snips` and `no_components` (`PC.NOT_YET`) to the
    Regions and Components markers, and with nothing to fit yet the check no longer
    adds "No parameter is free" (`_counts` is skipped).
  - The page was republished (version 4) with `toolbar_greyed.png`.
- **Exporting only what is needed** (RJC's comment on the D I ratio figure: an
  Export dialog with checkboxes, a column picker for the snips, one ASCII file per
  snip in a chosen folder). Paused: Q7.16 asks (a)–(g) in the stage document, the
  thread is left open, and the work waits for the response.
- **Q7.16 answered in part, and built** (RJC on the thread: (a) yes, (b) the name
  `ProjectName_Ion_TransitionRestWavelength_zabsBLAH_fit.dat`). The reply proposed
  refinements of (b) and that (c)–(g) follow the leans unless RJC says otherwise;
  the export was built so.
  - `alis/dashboard/export.py` (new, no Qt): `COLUMNS` (eleven; seven ticked by
    default), `file_names` (the project name without `.model`, spaces and slashes
    as underscores; the ion with no space or mass prefix; the rest wavelength to 2
    places; `zabs` and z to 4 places, left out while the system's z is hidden;
    `_2`, `_3`), `tables` (each snip's columns: Fit range from `SN.fitted`, as 1/0;
    the continuum and zero level the shown run's best fit, else the starting
    model's; a continuum or zero level whose value is hidden left out, and said in
    the header), `point_at` (a model's data lines name the exported files, with
    `columns=` and `fitrange=columns`; `run datadirc ./`), `plan`, `write`,
    `best_fit_reason`, `run_reason`.
  - `bundle.atomic_to_write` (from `extract`): the packed atomic table goes beside
    exported models when it differs from the installed one.
  - `qt/dialogs.py`: `ExportDialog`: four boxes, the column list (ticked or not,
    dragged or ↑ ↓, the run's columns greyed without one), the folder, Export
    greyed with the reason. `qt/window.py`: `do_file_export` asks about a blinded
    project only for the starting model or everything; one overwrite question
    (`_overwrite`).
  - "Export plain files…" becomes "Export…" in the menu, the toolbar, the
    command-line-only advice (`settings.EXPORT_ADVICE`) and the docs.
  - Checked by hand: both exported models of metal_line_abs run with `run_alis` in
    the export folder.
- **The tests.** `test_dashboard_export.py` (new, 9): names, a second snip's `_2`,
  the columns in their order, no best fit without a run, a zero level, the models'
  data lines, the best-fitting model refused while blinded, a hidden continuum
  left out, and (`fast`) `run_alis` fitting an exported model.
  `test_dashboard_qt_blinding.py`: the export tests answer the dialog;
  `test_the_export_dialog`, `test_export_writes_the_snips_and_both_models`. The
  action names, the pre-flight message and the refusal test follow the rename.
- **The page**: the Export dialog after the metal_line_abs run, its own section.
- **The documents**: D88; CHANGELOG; `ALIS_workflow.md` §4.3 (Export…, Run greyed);
  `tests/README.md`.
- **The batches after the review's changes.** `pytest -m unit`: 2084 passed, 119
  skipped (2073 at Prompt 5's close). `pytest -m gui`: 211 passed (206 at Prompt
  5's close). The page was republished (version 5) with the Export dialog.
- **What remains:** RJC's word on Q7.16 (c)–(g) (the export follows the leans; its
  thread is left open), and any further comments on the page. The stage closes then.

### 2026-11-01 (Prompt 6: RJC's response to Q7.16 (c)–(g))

- **The response.** The proposals are agreed. Blind parameters and blind models
  must stay hidden in the exported files (written as `BLIND`, say); no mass number
  on an ion in a file name.
- **Hidden values as BLIND.**
  - `blinding.py`: `MaskedView` and `Gate.view` take the mask's word (`▒▒▒▒` by
    default).
  - `export.py`: `masked_start` (the gate's view with `BLIND`); `masked_best` (the
    best-fit values the run's gate masks, every `blindseed`/`blindrange`, and the
    `.mod.out`'s copy of the input model masked as the panel masks it, matched
    with the blind keywords left out, `_copy_key`; a line hidden since the run is
    masked whole); `blinded`; `everything` (`bundle.extract` with the masked model,
    hidden comment lines as "BLIND MODEL", and the `.mod.out` replaced by
    `masked_best`, since the bundle's writes ALIS's offsets with their seed). The
    best-fitting model is no longer refused while blinded (`HIDDEN_FIT` gone); a
    note says the exported models run only once the project is unblinded.
  - `bundle.extract` takes `model_text` and `blind_comments` (defaults unchanged,
    so `run_alis --extract` still writes D10's plain text).
  - `qt/window.py`: the blinded question before an export is gone.
- **Names.** The ion's mass number is taken off for every isotope (`13C IV` is
  `CIV`); a hidden redshift is `zabsBLIND` (and `z = BLIND` in the header).
- **Found on the way.** A plain blind run's `.mod.out` (examples/blind's
  reference) holds "------ BLIND MODEL ------", which the dashboard's parser
  refuses, so such a run gives no best-fit values. Runs from the dashboard keep the
  real line, hidden in the bundle, so they are not affected; not changed.
- **The tests.** `test_dashboard_export.py`: `test_hidden_values_are_written_as_blind`
  (both models, the input copy, the seed, everything; each masked best-fit value
  as ALIS wrote it is gone), `test_no_mass_number_and_a_hidden_redshift`; 10
  passed. `test_dashboard_qt_blinding.py`: `test_export_keeps_hidden_values_hidden`
  replaces the warning's test; 8 passed.
- **The documents.** D88, `ALIS_workflow.md` §4.3, CHANGELOG, Q7.16's note, the
  review page's text.
- **The batches.** `pytest -m unit`: 2085 passed, 119 skipped. `pytest -m gui`: 211
  passed (435 s). The page was republished (version 6) with the export's new text.
- **Asked of RJC:** whether `run_alis --extract` should also write hidden values as
  `BLIND`; whether a blinded project's snips keep their Best-fit model column (and
  leave a hidden continuum or zero level out, rather than writing `BLIND` in a
  numeric file).
- **A comment missed, then applied.** RJC commented on the Export figure (10:44): a
  zero level asked for but not in the model is written as zeros, the default. The
  auto-reply said "Done" and republished the page's text, but the code was
  unchanged; this session's watch had dropped, and its publish was refused over
  that version. Applied: `_columns` writes 0 for a snip with no zero level in the
  model (a hidden one is still left out); the dialog's note, D88, `ALIS_workflow.md`
  and the page say so. `test_a_zero_level_is_written_and_0_where_there_is_none`;
  the window's export test expects the column. The page was rebuilt on top of
  that version and republished (version 7); the reply corrects the "Done", and the
  export's two threads are resolved.
- **Prompt 6** (added to the stage document): `run_alis --extract` should do what
  Everything run_alis needs does; leaving a hidden continuum or zero level out is
  agreed; the dashboard keeps its grey masks, and only exported files say BLIND.
  Executed below.

### 2026-11-01 (Prompt 6: run_alis --extract, and two comments)

- **`run_alis --extract`** does what Everything run_alis needs does.
  - `export.extract(bundle, project, name, folder, overwrite)` (new; `everything`
    calls it): `bundle.extract` with the masked model and hidden comment lines as
    "BLIND MODEL", and the `.mod.out` replaced by `masked_best`, only when anything
    is hidden; a project with nothing hidden is written byte for byte, as before.
  - `alis/scripts/run_alis.py`: `_bundle_command`'s extract reads the bundle, makes
    its `Project` and calls `export.extract` (no Qt).
  - The other two points of Prompt 6 held already: a hidden continuum or zero
    level is left out of its snip's file, and the dashboard shows its grey masks,
    with `BLIND` only in exported files.
- **The two comments** on the Export figure (11:13):
  - "Should we have (.dat) here?": Snips reads "Snips, one ASCII file each (.dat)".
  - "This should be .mod.out": Best-fitting model reads "(.mod.out)", and the file
    is now `<project>_bestfit.mod.out`.
- **The tests.** `test_run_alis_extract_writes_hidden_values_as_blind` (the blind
  example's Si II as BLIND, no seed; metal_line_abs byte for byte); the names of
  the best-fitting model in the export tests. `test_dashboard_export.py`,
  `test_cli.py`, `test_bundle.py`: 123 passed, 28 skipped;
  `test_dashboard_qt_blinding.py`: 8 passed. `pytest -m gui` before the change:
  211 passed.
- **The documents.** D88; CHANGELOG (Changed: `run_alis --extract`);
  `ALIS_workflow.md` §4.2.- **The batches after Prompt 6.** `pytest -m unit`: 2086 passed, 119 skipped.
  `pytest -m gui`: 211 passed (432 s). The page was republished (version 8); no
  comment thread is open.
- **What remains:** RJC's word that the review is complete; the stage then closes.

### 2026-11-01 (Prompt 7: the normalised columns)

- **The request.** Three normalised columns, not two: flux, error and best-fit
  model, each with the zero level removed and divided by the continuum less the
  zero level (`norm_flux = (flux − zero)/(continuum − zero)`, `norm_error =
  error/(continuum − zero)`, `norm_model = (model − zero)/(continuum − zero)`);
  said clearly; `norm_continuum` not offered.
- **The change.** `export.py`: `Column` has a `tip`; `nmodel` "Normalised best-fit
  model" (`norm_model`, needs a run, `NEEDS_RUN`); `_columns` takes the zero level
  first, then the continuum, and makes the three from them as asked, leaving all
  three out (with a header note) while the zero level or the continuum is hidden; a
  snip's header says how the normalised columns are made when one is written
  (`NORMALISED`). The dialog's items carry the formulas as tooltips, and its note
  says the normalised columns take the zero level away.
- **The tests.** `test_the_normalised_columns` (a trough with a zero level of
  −0.02: flux and error by the formula; no `norm_model` without a run; with a run,
  metal_line_abs's normalised model); the hidden continuum's notes.
  `test_dashboard_export.py`: 12 passed; `test_dashboard_qt_blinding.py`: 8 passed.
- **The documents.** D88, `ALIS_workflow.md` §4.3, the page's text.- **The dialog's height.** With a twelfth column the list overlapped the note
  under it (the dialog was sized before the note wrapped); the dialog now takes
  its layout's height for its width.
- **The batches.** `pytest -m unit`: 2087 passed, 119 skipped. `pytest -m gui`: 211
  passed. The page was republished (version 9).

### 2026-11-01 (Prompt 8: the velocity and the corrected wavelength)

- **The request.** A Velocity column, relative to a component the user chooses; a
  Corrected wavelength column (the wavelength with the snip's `shift=` taken out),
  just before Normalised flux; `vshift` now, the other shift models if easy.
- **The change.** `export.py`:
  - `Corrected wavelength` (`corr_wave`): `_shift_wave` calls the data line's own
    ALIS shift function (`Ashift`, `vshift`, `vshiftscale`, `polyshift`: all have
    `call_CPU(x, p)`, and the parser keeps every shift parameter with role
    "shift"), with the shown run's best fit or the starting values; left out (with
    the velocity) while a shift parameter is hidden. `vshiftscale` and `polyshift`
    scale about the middle pixel of the array given, here the whole snip file,
    which can differ slightly from ALIS's when it loads part of it.
  - `Velocity` (`velocity`): `P.velocity(λ_corr/λ_rest − 1, z_ref)`, for the
    transition the snip was cut for; `z_ref` the chosen component's best fit with a
    run (`view.gate.masked` hides it), else its starting value
    (`C.component_z`). `reference_components` lists the components whose
    redshift is shown; the first is the default. A header line names the
    component, its z and the transition; another, the shift taken out.
  - `Choice.velocity_of`; `tables(..., velocity_of=...)`.
- **The dialog.** "Velocity relative to", a drop-down of the components, enabled
  while Velocity is ticked; Velocity greyed when no component's redshift can be
  shown. The window passes `reference_components`.
- **The tests.** `test_the_velocity_and_the_corrected_wavelength` (the shifted
  synthetic fit with polyshift, Ashift and vshift on three snips: each corrected
  wavelength by its formula, the velocity, 0 km/s at the lines); the dialog test
  ticks Velocity. `test_dashboard_export.py`: 13 passed;
  `test_dashboard_qt_blinding.py`: 8 passed.
- **The documents.** D88, `ALIS_workflow.md` §4.3, the page's text.- **The batches.** `pytest -m unit`: 2088 passed, 119 skipped. `pytest -m gui`: 211
  passed. The page was republished (version 10).

### 2026-11-01 (Prompt 9: Velocity and Corrected velocity)

- **The request.** Velocity under Wavelength, on the wavelength; a Corrected
  velocity under Corrected wavelength, as Velocity but on the corrected wavelength.
- **The change.** `export.py`: `velocity` moved under `wave` and measured on the
  wavelength; `cvelocity` "Corrected velocity" (`corr_velocity`) under `cwave`, on
  the corrected wavelength (`VELOCITIES` maps each to its wavelength). One header
  line, "Velocities: …", names the component, its z, the transition and the
  velocities written. A hidden shift leaves out the corrected wavelength and
  velocity only. The dialog's component drop-down is enabled while either velocity
  is ticked, and both are greyed when no component's redshift can be shown.
- **The tests.** The velocity test checks both against the wavelength and the
  corrected wavelength; the dialog test checks the order and the drop-down for
  Corrected velocity, and moves Fit range by its row. `test_dashboard_export.py`:
  13 passed; `test_dashboard_qt_blinding.py`: 8 passed.
- **The documents.** D88, `ALIS_workflow.md` §4.3, the page's text.- **The batches.** `pytest -m unit`: 2088 passed, 119 skipped. `pytest -m gui`: 211
  passed. The page was republished (version 11).

