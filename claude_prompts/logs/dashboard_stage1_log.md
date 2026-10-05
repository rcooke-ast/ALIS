# Dashboard Stage 1 log

Changes to ALIS itself: project bundles, in-memory data, the shared-pixel warning, one
writer for every output, and the removal of onefits. The plan is in
`claude_prompts/dashboard_stage1.md`.

### 2026-10-05 (Prompt 1: queries checked, two follow-ups, Tasks 1.1–1.9)

**Queries.** RJC answered Q1.1–Q1.9 and agreed with every lean except Q1.9: onefits
is removed outright, so `out onefits` is now an ordinary unrecognised setting, with
no special message. Two follow-ups were asked in the session and recorded in the
document:
- **Q1.10:** the atomic table the model uses is always packed, and a bundle run
  always uses the packed copy;
- **Q1.11:** snips named on commented-out data lines are packed when they exist.

The Design section and Tasks 1.1, 1.3, 1.5 and 1.7 were updated to match. In
response to RJC's caution on Q1.8, Task 1.3 gained two guards against chance
matches.

**Baseline, before any change.**
- `pytest -m unit`: 3 failures, all `test_writer_round_trip` on the helium34
  `*_FINAL_MODEL.mod` files (the free-parameter count changes on re-reading).
- The `fast` regression batch: 11 failures, all `context/` cases (10
  `test_fixed_param` and the tet02OriA minimisation).

The same failures appear on a clean export of HEAD, so they predate this stage. The
`context/` folder is not tracked by git. Every `examples/` case passes.

**1.1 Remove onefits.**
- Deleted `save_onefits` and `load_onefits` (with its interactive menu), the
  `out onefits` setting, the `.fits` branch of `load_input`, the
  `alisfits == "onefits"` branch of `load_fits`, and `_isonefits` (in `main.py`,
  `load.py` and two test fixtures).
- Removed the setting from `doc/tex_files/description.tex`.
- `grep -ri onefits alis tests` now finds nothing.
- A binary model file (such as a FITS file) now stops with "not a text file"
  instead of a traceback.
- Two small bugs in `load_input`'s end checks were fixed: a model given as text
  had no `loadname` for its error messages, and a missing `model end` was tested
  with the data counter.
- Test: `test_config.py::test_out_onefits_is_an_unrecognised_setting`.

**1.2 Several data lines from memory.**
- `load_data(data=...)` also accepts a mapping from each data file's path (as
  resolved: `datadirc` plus the name) to its bytes.
- `load_ascii` and `load_fits` read the bytes through `io` streams with `open`'s
  defaults, so the parser sees the same text as from disk.
- New helpers: `load.data_file_paths` and `load.memory_key` (tries the name as
  given, then normalised).
- Test: `tests/test_load_memory.py`. Every `examples/` model, J1358p6522 and
  Q1243p307 load identically, bit for bit, from disk and from memory.

**1.3 Pixels fitted twice.**
- `load.find_shared_pixels` matches wavelengths (1 part in 10⁹) and flux/error
  ratios (1 part in 10⁶). It ignores zero flux and errors of zero or less, and
  counts only runs of at least 3 consecutive pixels.
- `load.warn_shared_pixels` warns once per pair, listing 10 pairs and then a count.
- `load_data` calls both.
- Found in the real data:
  - **J1358p6522:** H I 923/926 (18), 926/930 (18), 930/937 (23), and also H I 972
    with O I 976 (13);
  - **Q1243p307:** O I 1302 with Si II 1304 (97, in each of the three datasets),
    plus H I 923/926 (6 each) and H I 949.7 with the z = 2.44 H I 972.5 (17).
- No shipped example fits a pixel twice.
- Test: `tests/test_shared_pixels.py`.

**1.4 One writer for every output.**
- New `alis/outputs.py`:
  - `DiskOutputs`, the default, writes exactly as before, prompts included;
  - `MemoryOutputs` keeps files by normalised path, never prompts, and records
    hidden files and lines.
- Routed through it:
  - the snip writers, `save_modelfits` (including its renaming), `save_model`,
    `save_covar` and its PNG;
  - the report, the plotting script and the PDF;
  - the convergence files, and `convY`/`convN`;
  - the `out wavecorr` file. `FitState` carries the writer, so a worker process
    writes to its own copy, which is discarded; the best-fit file is written by
    the final evaluation in the main process.
- `ClassMain` gained `writer=` and `atomic=`.
- At first I planned to refuse `out wavecorr` in bundles. All nine helium34
  models use it, so it is routed instead, as the document first planned.
- A PDF is now written only when there are figures, as `PdfPages` already behaved
  on disk.
- `save_covar` now overwrites an existing FITS covariance file when told to.
  Before, astropy refused.
- Test: `tests/test_outputs.py`. One fit of `metal_line_abs` with each writer gives
  the same files, with the same bytes apart from dates.

**1.5 `alis/bundle.py`.**
- Implements pack, read, write (atomic, under a lock file), update, extract,
  unblind, and the source functions (add, check, relink, embed, unembed). The
  layout is as in the Design section.
- `load.atomic_table_path` was split out of `load_atomic`, which also takes the
  packed table (`table=(name, bytes)`).
- Test: `tests/test_bundle.py`.
  - 47 models (from `examples/` and `context/`) pack and extract byte for byte.
  - 38 are rightly refused:
    - 23 generate data;
    - 5 `Temperature/` models name data files that are not in the folder;
    - J1558m0031_zerolevel does not load.

**1.6 Hidden values.**
- Lines with `blind=True` become `<hidden:n>`; the real lines are kept in
  `hidden.bin` (zlib-compressed JSON).
- `load_input` stops at a placeholder.
- During a bundle run:
  - a filter on the `alis` logger shows a hidden line's placeholder in any message
    that quotes it, and masks every number in that message;
  - the in-memory writer hides every output line that repeats a hidden line, such
    as the copy of the input model at the end of a `.mod.out`.
- The nine `parout` functions that print "BLIND MODEL" now ask
  `base.hides_blind_line`. Inside `base.revealing_blind_lines()` they print the
  real line, which a bundle stores hidden.
- `check_argflag(keep_blind_outputs=True)` keeps the `.mod.out`, `_fit.dat` files
  and covariance matrix of a blind bundle run. Under `run blind True` the whole
  `.mod.out` and the plotting script are hidden.
- `unblind()` restores everything and logs the time and a note in the manifest.
- **A blinding bug found and fixed.** `apply_cli_settings` skipped any flag or
  `--set` that repeated the default. So `--set "run blind True"` was undone by a
  model's `run blind False`: a run the user asked to be blind was not. Every
  explicit setting is now recorded. The blind rules (`load.apply_blind_rules`,
  split out of `check_argflag`) are applied again after the command line.
- Tests in `test_bundle.py`, `test_cli.py` and `test_outputs.py`:
  - the Si II line of `examples/blind` appears in no member of the zip;
  - a bundle run prints no hidden value;
  - a run made to fail on a limit in the hidden line records only
    "A parameter that = ▒▒▒ … `<hidden:1>`";
  - `run blind True` hides the whole `.mod.out`;
  - `unblind()` gives back the original text.

**1.7 `run_alis project.model`.**
- `bundle.run`:
  - checks what a bundle cannot run (Q1.5, Q1.6);
  - runs the fit with its data, atomic table and outputs in memory;
  - then replaces `runs/latest/` (with `run.json` and `input.mod`) in the bundle
    as it is on disk at the end.
- A run that stops with an error writes `runs/last_error.json` and leaves the
  latest run unchanged.
- **Harness:** the new mode `test_regression.py::test_bundle` packs, runs and
  extracts each `fast` example. All 25 pass, and the runs leave nothing on disk
  but the bundle.
- **Real fits:** the plan named VMP_DLA/J1358p6522, but on this machine it fails
  as a plain run too. RJC's untracked `alis/data/atomic_rjc.xml` has no `Ly`
  entries, and the model uses `1Ly_a`. J0814p5029 and Q1243p307 fail the same
  way, and Her36 disagrees with its reference in plain mode. So
  `test_bundle_matches_plain` runs two real fits both ways and compares them with
  each other: VMP_DLA/J0903p2628, and helium34/Her36, which writes
  `out wavecorr` files.

**1.8 `--extract` and `--pack`.**
- Both are added to `run_alis`, with a second, optional positional argument:
  - for `--extract`, the directory to write to (by default named after the
    bundle);
  - for `--pack`, the bundle to write (by default the model's name).
- Neither writes over a file without `-w`.
- `--extract` writes:
  - the hidden starting values, after a warning (D10);
  - the hidden best-fit lines as "BLIND MODEL";
  - no `.mod.out` under global blind;
  - the packed atomic table beside the model, when it differs from the installed
    one.
- Tests in `test_cli.py`, and in `test_bundle.py`: pack, extract, then a plain run
  of what was extracted, which matches the reference.

**1.9 Closing.**
- Documentation:
  - `CHANGELOG.md` (Unreleased);
  - `doc/ALIS_workflow.md` §4.2 (bundles) and §4.3 (pixels fitted twice), now
    version 0.5;
  - `tests/README.md`.
- Coverage: `alis/bundle.py` 92%, `alis/outputs.py` 96%.

### 2026-10-05 (Prompt 2: the Stage 2 design document)

**Final batches of Stage 1.** `unit` and `fast` together gave 940 passed and 14
failed. The 14 failures are exactly the 3 + 11 recorded in the baseline above, all
in `context/`.

**`claude_prompts/dashboard_stage2.md` written:** the project model, with no Qt.
- **Design:** four layers (model text, parsed model, project, all on the bundle);
  the table mapping each dashboard concept to the `.mod`; the text-sync contract;
  and the services (validator, blinding gate, history, removal, opening a fit,
  modes).
- **Tasks 2.1–2.11:**
  - the package and a check that it imports no Qt;
  - the model text, round-trip tested on every model;
  - the parsed model, cross-checked against ALIS's own parser;
  - targeted edits;
  - the project;
  - the validator;
  - the blinding gate;
  - history;
  - removal;
  - modes;
  - closing, including the rewrite of the `gui-dev` and `gui-component` skills.
- **Queries Q2.1–Q2.9**, each with a lean:
  - the modules;
  - `ui/project.json`;
  - the label scheme;
  - inferring datasets, systems and components;
  - catching ALIS's errors without changing ALIS;
  - spacing after an edit;
  - editing only a snip's mask column;
  - not saving the undo history;
  - building the S16 and F12 logic now.

**The medium batch (finished after Prompt 2's reply).** 3 passed and 13 failed. The
three passes include both `test_bundle_matches_plain` cases. The 13 failures are
all plain-mode `context/` cases:
- Most stop with "Element 1Ly not found": the local `atomic_rjc.xml` has no `Ly`
  entries.
- The rest disagree with their references. Four of them (J0903p2628, J1358p0349,
  J1558m0031, Q0913p072) fail the same way on a clean export of HEAD, so they
  predate Stage 1.
- No bundle test failed.

### 2026-10-05 (`atomic_rjc.xml` retired; the gate rerun)

**`run atomic atomic_rjc.xml` removed** (RJC: the file is out of date; `.ecsv` only
from now on, `atomic.ecsv` the default).
- Deleted the active setting line from 23 files: 11 `.mod`, 8 `.mod.out.reference`
  (which the fixed-parameter tests run as models) and 4 stale `.mod.out`.
- Kept:
  - commented copies in the outputs' "copy of the input model" sections;
  - five archived `*.mod.out.orig` files, which no test reads;
  - the mentions in the `context.md` notes.
- The originals were backed up to the session scratchpad, because `context/` is
  not tracked by git.

**The gate (`fast or medium`):** 93 passed, 20 failed (24 failed before).
- The four VMP_DLA/J0814p5029 and VMP_DLA/J1358p6522 cases (minimisation and
  fixed-param) now pass.
- 7 of the failing minimisations are tagged `machine_dependent`, which the gate
  skips off the reference machine (`--skip-machine-dependent`): HS0105p1619,
  J1358p0349, J1558m0031, Q0913p072, Q1243p307, J0035m0918, J0903p2628.
- The unit batch: 854 passed, 3 failed (the helium34 round trips).

**Why the rest fail: the `context/` references on disk are stale.**
- The refactor's Stage 5 log records that RJC regenerated all 42 references on
  2026-08-03, after the Stage 5.6 atomic-mass fix and the Stage 5.4 writer fixes,
  and that the full harness then passed (613 passed, 0 failed).
- The `context/` references in this working copy are dated 14–21 July 2026, so
  they predate that regeneration. The `examples/` ones, which are tracked, are
  dated August and pass.
- The two atomic tables agree for the lines these fits use: H I, D I, O I, C II
  and N II have the same f-values and damping constants apart from one or two
  lines each. So the atomic data do not explain the differences.
- The helium34 round trips fail on old references that carry `damping=0.0000000`
  on every line. That writer bug was fixed in Stage 5.4
  (`load.call_function_load`); the current writer is correct. (In the session I
  first took this for a live bug, then corrected it.)
- The remaining fixed-parameter and minimisation differences are consistent with
  the atomic-mass fix, which changed thermal broadening.

**What would make the gate pass:** restore the regenerated `context/` references
from wherever the 2026-08-03 set is kept, or run `regen_harness.sh` again. That
script is not in this repository, and regeneration has always been RJC's to do.

### 2026-10-05 (The gate after RJC regenerated the references)

RJC regenerated the harness references (commit `27a42e6`, "regen harness").
- **Unit batch:** 857 passed, 0 failed.
- **`fast or medium` batch,** with the machine-dependent fits included: 114 passed,
  1 failed.
- **The failure:** `test_fit_report.py::test_report_chi2_self_consistent`. It copies
  the whole `examples/metal_line_abs` folder and expected one `.report`. The
  regeneration had left three reports there, plus a PDF and a PNG, all untracked.
- **Fixed in two ways:**
  - the 36 untracked run outputs left across `examples/` (25 `.report`, 8 `.pdf`,
    1 `.png`, 1 `.mod.out`, 1 `_fit.dat`) were moved to the session scratchpad;
  - the test now removes earlier outputs from its copy and reads
    `fit_spectra.mod.report` by name. It passes both on the clean folder and with
    an old report put back.
