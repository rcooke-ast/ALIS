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
