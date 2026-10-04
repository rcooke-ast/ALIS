# Prompt file for ALIS software dashboard creation -- STAGE 1

> **Changes to ALIS itself.** The dashboard keeps each project in one file, a zip
> bundle called `project.model` (D8). This stage teaches ALIS to run such a bundle
> without the dashboard, and makes the other changes the dashboard needs in ALIS:
> - `run_alis project.model` reads the bundle into memory, runs the fit and writes the
>   outputs back into the bundle (D9);
> - `run_alis --extract` writes a bundle out as plain files (D9, D10);
> - `load_data` reads several data lines from memory (D9);
> - `run_alis` warns when a pixel is fitted twice (D19);
> - onefits is removed (D11).
>
> There is no dashboard code and no Qt in this stage, and no new dependency: the
> bundle uses only Python's standard library (`zipfile`, `json`, `hashlib`, `zlib`).
> Plain `.mod` files run exactly as before. The refactor's regression harness must
> stay green after every task, and a new harness mode checks that a bundle gives the
> same results as the plain files it was made from.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`; "Q0.*n*" to `dashboard_stage0.md`,
> which also holds the plan for all stages.

## Design

*Written by Claude on 2026-10-04. The choices still open are the Queries below; each
gives Claude's lean, and this section follows the leans.*

### The bundle

A bundle is an ordinary zip file. Unzipping it gives files that a user recognises
(QF.32(a)):

```
J1358p6522.model                     a zip file
├── manifest.json                    format version, ALIS version, dates, the
│                                    model's path, and a SHA-256 for each member
├── files/                           the fit as plain files, laid out as on disk
│   ├── model/J1358p6522.mod         the model, with hidden lines as placeholders
│   └── data/J1358p6522_H1.dat …     its snips
├── hidden.bin                       the hidden lines, compressed (not readable)
├── sources.json                     source spectra: path, SHA-256, size, dataset
├── sources/…                        source spectra embedded for archiving (optional)
├── ui/…                             the dashboard's own state (ALIS keeps it as is)
└── runs/
    ├── latest/                      the most recent run
    │   ├── run.json                 when and where it ran, the command, status, χ²
    │   ├── input.mod                the model it ran
    │   └── files/…                  its outputs, at the paths ALIS writes them to
    └── 0001/ …                      runs committed to the history (Stage 6, D26)
```

- **The `files/` tree.** The model and its snips are stored at the same relative paths
  as on disk. The root of the tree is the lowest directory that holds the model and
  all of its data, so `run datadirc ../data/` and data lines such as
  `../../data/J013301m400628_B/…` (both used by the context models) need no change to
  the text. The model text is stored exactly as written (D7).
- **Reading a data line.** ALIS resolves the file name as it does now (`datadirc` plus
  the name on the line, relative to the model) and takes that file's bytes from the
  bundle instead of the disk.
- **Outputs.** A run writes `.mod.out`, `_fit.dat`, the report, the PDF, the
  covariance matrix and the plotting script to the paths it would use on disk, but
  into `runs/latest/files/` in memory. Extracting the bundle therefore gives exactly
  the directory a plain run would have left.
- **Source spectra** (Q0.5). A dataset may have several source files. Each has its
  path, SHA-256 and size, and may also be embedded for archiving (D8). ALIS does not
  need them to fit (the snips are the data), but the bundle library checks them
  (missing, changed or present), relinks a moved file when its checksum matches (F13)
  and embeds or removes a copy. The dashboard uses these in Stage 3.
- **Writing.** A bundle is never half-written. It is written to a temporary file
  beside it and then renamed over it, under a lock file. Members that ALIS does not
  use (`ui/`, committed runs) are copied unchanged (F1).

### Hidden values

Hiding prevents accidental viewing, not a determined user (QF.19(d)).
- **What is hidden.** In the starting model, every line with `blind=True` (D24). The
  library can hide any line, because the dashboard will also need to hide best-fit
  values copied into the model (S20) when global blind is on (QF.20(a)).
- **How.** A hidden line is replaced in the stored text by a placeholder line,
  `<hidden:3>`, with its indentation kept. The real lines are kept in `hidden.bin`
  (compressed JSON). A plain `run_alis` that meets a placeholder stops with a message
  saying that the model came from a bundle, rather than skipping the line.
- **Restoring.** `run_alis project.model` puts the real lines back in memory only. No
  message, warning or printout shows a hidden line: messages that quote a model line
  show the placeholder instead.
- **Best-fit values.** ALIS writes no `.mod.out`, `_fit.dat` or covariance matrix for
  a blind run today (`alis/load.py:403-420`). For a bundle, the dashboard needs them
  (QF.10(c), QF.20(c)), so a bundle run writes them, and hides what reveals a value:
  - with `run blind True`, the whole `.mod.out` and the plotting script;
  - with `blind=True` on some lines, those lines of the `.mod.out` (with their real
    best-fit values, where a plain run writes "BLIND MODEL"), and the plotting script;
  - `_fit.dat` files, the report, the PDF and the covariance matrix are not hidden:
    profiles may be drawn, and errors may be shown (D24).
- **The terminal.** A bundle run prints what a plain run with the same settings
  prints.

### Running a bundle

`run_alis project.model`:
1. reads the bundle and checks the manifest's checksums;
2. restores the hidden lines in memory and passes the text to `load_input(textstr=…)`;
3. passes the snips' bytes to `load_data`, keyed by their paths (Task 1.2);
4. runs the fit as usual, with every output written to memory (Task 1.4);
5. merges the outputs into the bundle as `runs/latest/`, with `run.json` recording
   when and where the run took place, the command, the ALIS version, the status, χ²,
   the degrees of freedom and the number of iterations, and the SHA-256 of the model
   it ran. The dashboard shows this run on opening (S31), and can tell when the model
   has changed since (F12).

Nothing is unpacked to disk (D9).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage1_log.md`.
> After every task, run the `unit` and `fast` batches of the test suite; run the
> `medium` batch before the stage closes.

**1.1 — Remove onefits (D11, QF.35).** Remove it first, because it sits in the code
that the next tasks change.
- Remove:
  - the `out onefits` setting (`alis/config.py:164`);
  - `save_onefits` (`alis/save.py:135`) and its uses in `save_modelfits`;
  - `load_onefits` and its menu (`alis/load.py:1783`);
  - the `.fits` branch of `load_input` (`alis/load.py:463`);
  - the `alisfits == "onefits"` branch of `load_fits`;
  - `_isonefits` in `main.py`, `load.py` and the two test fixtures that set it;
  - the `onefits` lines in `check_argflag` and `ClassMain.main`.
- Keep `out fits`, which writes one FITS file per snip.
- A `.fits` model file stops with a message that onefits was removed and points to the
  bundle. `out onefits True` in a model does the same; `out onefits False` gives a
  warning and is ignored (Q1.9).
- Correct the onefits paragraph of `doc/tex_files/description.tex`.
- **Check:** `grep -ri onefits alis tests` finds only the new message; the test suite
  passes.

**1.2 — Load several data lines from memory (D9).** Today `load_data` takes one
in-memory array, and only when the model has a single data line (`load.py:734`).
- `data=` also accepts a mapping from each data file's path, as `load_data` resolves
  it (`datadirc` plus the name on the line), to that file's bytes.
- The bytes are read by the same code as a file on disk (`load_ascii` and `load_fits`,
  through `io.BytesIO`), so the arrays are identical bit for bit.
- A data line whose file is not in the mapping stops with a message naming the file.
- The single-array form of `data=` keeps working as now.
- `ClassMain` and `alis.main.alis()` pass the mapping through unchanged.
- **Check:** a new unit test loads every `examples/` model, and the J1358p6522 and
  Q1243p307 models, from disk and from memory, and compares every loaded array with
  `np.array_equal`.

**1.3 — Warn about pixels fitted twice (D19, QF.21, S26).**
- A pure function, `find_shared_pixels`, takes the fitted pixels of every snip and
  returns each pair of snips that share pixels, with the shared indices. Two pixels
  are the same when their wavelengths and their flux/error ratios agree (Q1.8). The
  ratio does not change when a snip has been multiplied by its continuum, but it
  differs between exposures on a common wavelength grid, so those are not flagged
  (QF.21(a)).
- `load_data` calls it once the data are loaded. It warns, once per pair, with the
  number of pixels, their wavelength range and the two file names, and suggests
  merging the snips or keeping the pixels in one of them. The fit still runs
  (QF.21(c)).
- The function is the logic behind the hatching and the one-click fixes of S26, which
  the dashboard builds in Stage 4.
- **Check:** unit tests show that:
  - J1358p6522's H I 923/926, 926/930 and 930/937 snips are flagged, with 18–23
    shared pixels each (found in Stage 0);
  - Q1243p307's O I 1302 and Si II 1304 snips are flagged in each of its three
    datasets (97 pixels each);
  - a file loaded twice is flagged;
  - two synthetic exposures on one wavelength grid are not flagged;
  - the shipped `examples/` that do not overlap give no warning.

  Fit results do not change.

**1.4 — Send every output through one writer (D9).** ALIS writes its outputs from
many places, each with its own `open`, `savetxt`, `writeto` or `savefig`, and several
ask "Overwrite? (y/n)" at the terminal. A bundle run must collect them in memory.
- Add a small writer object, held by the fit as `slf._outputs`, with two forms:
  - **on disk:** the default; it writes exactly as today, prompts included;
  - **in memory:** it keeps each file's bytes by path, never prompts, and replaces a
    file written twice.
- Route through it:
  - `save_asciifits`, `save_fitsfits` and `save_modelfits` (including the renaming of
    snips that are used twice);
  - `save_model` and `save_covar` (including the correlation-matrix PNG);
  - `report.write_report`, `plotscript.write_plotscript` and `plot.plot_pdf`;
  - the convergence files (`convergence.py`, and the `convY`/`convN` files written in
    `main.py:404-430`);
  - the `out wavecorr` file (`model_eval.py:584`).
- The writer also records which outputs are hidden (Design, "Hidden values"). It does
  not write them out in plain form.
- `simulate.py` is not routed, because bundles do not support simulations (Q1.6).
- **Check:** the regression harness is unchanged. A unit test runs
  `examples/metal_line_abs` with each form of writer and finds the same files, with
  the same bytes apart from dates and run times.

**1.5 — The bundle format: `alis/bundle.py` (D8, QF.19, QF.32, Q0.5).** One module,
with no dashboard code, which both `run_alis` and the dashboard use.
- **Pack:** make a bundle from a plain `.mod` (Q1.4):
  - find its data files as `load_data` does;
  - choose the root of the `files/` tree;
  - copy the model and snips in;
  - hide the `blind=True` lines (Task 1.6);
  - optionally add source spectra, by path or embedded;
  - write the manifest.

  Absolute data paths stop with a message (Q1.1).
- **Open:** read the manifest, check every member's SHA-256, and refuse a newer format
  version with a message to update ALIS. Give the model text, the data bytes by path,
  the sources, the runs and any other members.
- **Write:** atomically, under a lock file, keeping unknown members (Design, "The
  bundle").
- **Extract:** write the `files/` tree and the latest run's outputs to a directory
  (Task 1.8).
- **Sources:**
  - check every source, and report each as present, missing or changed;
  - relink a moved source, accepted only if its SHA-256 matches (F13);
  - embed a source, or remove the embedded copy.
- **Check:** unit tests show that:
  - packing then extracting every model in `examples/` and `context/fitting_examples/`
    gives back byte-identical files at the same relative paths;
  - a damaged member, or a newer format version, is refused;
  - `ui/` and committed runs survive a rewrite;
  - an interrupted write leaves the old bundle intact;
  - a moved source can be relinked, and a changed one is reported.

**1.6 — Hidden values (D24, QF.19, QF.20).**
- Hide and restore lines (Design, "Hidden values"). `hidden.bin` is compressed JSON
  mapping each placeholder to its line.
- The plain loader (`load_input`) stops at a `<hidden:n>` placeholder with a message
  saying that the model belongs to a bundle.
- While a bundle runs, a filter on the `alis` logger replaces any hidden line in a
  message with its placeholder. Printouts of values already follow the blinding
  rules (`print_model(blind=True)` and "BLIND MODEL").
- When the bundle is blinded, `check_argflag` no longer turns off `out model`,
  `out fits` and `out covar`. The outputs that reveal values are hidden instead (the
  `.mod.out` in whole or in part, and the plotting script), and the terminal output is
  unchanged.
- `unblind()` in the library replaces every placeholder with its real line, empties
  `hidden.bin`, and records the time in the manifest's unblinding log (D24). The
  dashboard asks the user to confirm before calling it (Stage 2). There is no
  command-line flag for it in this stage (Q1.3).
- **Check:** unit tests show that:
  - after packing `examples/blind`, its Si II line appears in no member of the zip,
    once each member is decompressed;
  - a bundle run of that model, and a run made to fail on a limit in a hidden line,
    print no hidden value;
  - the `.mod.out` in the bundle holds the real line, hidden;
  - with `run blind True`, the whole `.mod.out` is hidden, and the `_fit.dat` files
    and the covariance matrix are stored in plain form;
  - `unblind()` gives back the original text and is logged.

**1.7 — `run_alis project.model` (D9, S31).**
- `run_alis` recognises a bundle by its `.model` extension and confirms it is a zip
  file. It then runs it as in the Design ("Running a bundle"). Command-line settings
  (`-p 0`, `--set …`) apply as they do to a `.mod`.
- The supported runs are a fit, with or without the convergence check, `sim repeat`,
  and plotting without fitting (`-j`). A simulation, `iterate model` or
  `generate data` stops with a message suggesting that the bundle be extracted first
  (Q1.6). So does a model that reads an auxiliary file the bundle does not hold
  (Q1.5).
- **When the run ends** (Q1.7):
  - it re-opens the bundle as it is then on disk and replaces only `runs/latest/` and
    the manifest, so edits saved by the dashboard during the run are kept;
  - an interrupted fit (Ctrl-C) still writes its outputs, as now;
  - a run that stops with an error leaves `runs/latest/` as it was and records the
    error in `runs/last_error.json`.
- `run.json`:
  - start and end times, host, ALIS version, command line and the settings changed
    on it;
  - SHA-256 of the input model and of the atomic table used;
  - status, initial and final χ², degrees of freedom, number of iterations, the
    reason for convergence, and the list of outputs with their SHA-256.
- **Check:** a new regression-harness mode, `test_bundle`, for the `examples` cases of
  the `fast` batch, and for `VMP_DLA/J1358p6522` in the `medium` batch:
  - stages the case;
  - packs it, runs the bundle with `-p 0`, and extracts it;
  - compares the outputs with the reference, using the tolerances of mode (a).

  The bundle must give the same results as the plain run. Blind cases are compared as
  mode (a) compares them now.

**1.8 — Extract and pack from the command line (D9, D10, Q1.4).**
- `run_alis --extract project.model [DIR]`:
  - writes the `files/` tree and the latest run's outputs to `DIR`, by default a new
    directory named after the bundle, beside it;
  - refuses to write over existing files unless `-w` is given;
  - writes hidden starting values in plain text, after a warning (D10), because the
    extracted model must run;
  - writes hidden best-fit values only as a plain blind run would (Q1.3). With
    `run blind True` there is no `.mod.out`, and lines with `blind=True` read
    "BLIND MODEL".
- `run_alis --pack fit.mod [project.model]` makes a bundle from a plain fit (Task
  1.5). It also adds the fit's existing outputs as `runs/latest/` when they are
  present.
- Update `run_alis --help` and the positional argument's text ("model file (.mod) or
  project bundle (.model)").
- **Check:** `tests/test_cli.py` covers both commands. `--pack` then `--extract` of
  `examples/metal_line_abs`, run as a plain model, matches the reference.

**1.9 — Close the stage.**
- `CHANGELOG.md`: bundles, `--extract`, `--pack`, the shared-pixel warning, and the
  removal of onefits.
- `doc/ALIS_workflow.md`: a short section on project bundles and the shared-pixel
  warning. The full user documentation waits for the deferred documentation task
  (refactor 6.3).
- `tests/README.md`: the `test_bundle` mode and the new unit tests.
- Run the `unit`, `fast` and `medium` batches, and `test-coverage` on
  `alis/bundle.py` and the new `load`/`save` code.
- Record in this document what Stage 2 receives from `alis/bundle.py`, so that the
  project model can be built on it.
- Update the stage table in `dashboard_stage0.md` if anything moved between stages.

## Skills to use for this stage

- `run-tests`: the `unit`, `fast` and `medium` batches after each task.
- `gen-tests`: unit tests for `alis/bundle.py`, `find_shared_pixels` and the writer.
- `test-coverage`: on the new code, at the end of the stage.
- `run-example` and `check-fit`: comparing a bundle run with a plain run.
- `gui-dev` and `gui-component` are not used. They describe `prepfit`'s GUI and are
  rewritten for PySide6 and pyqtgraph before Stage 3.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`:
  - D7–D11, D19 and D24;
  - QF.4, QF.5, QF.10, QF.19–QF.21, QF.32 and QF.35;
  - F1, F3, F13, S26 and S31.
- `claude_prompts/dashboard_stage0.md`: the stage plan, and Q0.5 (several source files
  per dataset).
- `claude_prompts/ALIS_v2_code_plan.md`: "The dashboard (planned separately)".
- **The code:**
  - `alis/load.py`: `check_argflag` (388), `load_input` (448), `load_data` (725),
    `load_userdata` (1184), `load_fits` (1297) and `load_onefits` (1783);
  - `alis/save.py`: `file_exists` (13), the snip writers (30, 78), `save_onefits`
    (135), `save_modelfits` (249), `save_model` (510) and `save_covar` (711);
  - `alis/main.py`: `ClassMain` (68), and the outputs written after a fit
    (`main.py:438-482`);
  - `alis/report.py:192`, `alis/plotscript.py:423`, `alis/plot.py:677`,
    `alis/convergence.py`;
  - `alis/scripts/run_alis.py` and `alis/config.py`.
- **The tests:** `tests/README.md` (the harness and its batches), `tests/alisrun.py`,
  `tests/test_regression.py`, `tests/test_load_files.py`, `tests/test_save_helpers.py`
  and `tests/test_cli.py`.
- **The models:**
  - `examples/blind`: a line with `blind=True`;
  - `VMP_DLA/J1358p6522` and `DH/Q1243p307`: shared pixels;
  - `examples/lsf_file`: the only model with a live reference to an auxiliary file;
  - the context models with `run datadirc ../data/`, or with data two levels up.

## Queries

*Raised by Claude on 2026-10-04, while writing this document. Each gives Claude's
lean, and the Design section and the tasks follow the leans.*

**Q1.1 — The bundle's layout.** The Design section proposes this layout:
- the model and its snips under `files/`, at their relative paths on disk, so the text
  never changes and extracting gives back the original directory;
- the outputs of the latest run under `runs/latest/`, laid out the same way;
- the committed runs under `runs/0001/` and so on, written by the dashboard in
  Stage 6;
- the manifest, the sources and the hidden lines as separate members.

Questions:
- **(a)** One model per bundle?
- **(b)** Absolute paths on data lines (none of the context models uses them): should
  packing stop and ask for relative paths, or store such files under
  `files/absolute/…`?

My lean: the layout above; (a) one model; (b) stop.

**Response:** Yes, one model per bundle, but there could be multiple committed runs. Absolute paths should be rejected when packing, and the user should be prompted to use relative paths instead.

**Q1.2 — How hidden values are stored.** There are two ways to hide a value in the
stored model:
- **(i) Whole lines.** A hidden line is replaced by `<hidden:3>`.
- **(ii) Single values.** Each hidden number is replaced by a placeholder, and the rest
  of the line stays readable, for example `voigt ion=2H_I <h3>  <h4>a  …  blind=True`.

(i) is simpler and cannot leak part of a line. Under (ii), someone who unzips the
bundle sees more of the model. The dashboard is not affected either way: it holds the
real lines in memory and masks values on screen (F8, D24). `hidden.bin` would be
compressed JSON. It is not readable, but any code can decode it, as QF.19(d) allows.

My lean: (i), with the outputs hidden as in the Design section.

**Response:** Yes, whole lines should be hidden, as it is simpler and more secure. The outputs should be hidden according to the rules in the Design section.

**Q1.3 — Extracting a blinded bundle.**
- **Starting values.** D10 says that a plain export writes the hidden starting values
  in plain text, with a warning, because the exported model must run.
- **Best-fit values.** I propose that `--extract` never writes hidden best-fit values.
  It writes the outputs that a plain blind run would leave (no `.mod.out` under
  `run blind True`, and "BLIND MODEL" lines for `blind=True`). Unblinding stays a
  separate, confirmed and logged action (D24).
- **Unblinding.** I propose to offer it only from the dashboard for now, through the
  library function `unblind()`.

Should `run_alis` also have an `--unblind` flag, which asks for confirmation and logs
the unblinding?

My lean: as proposed, with no `--unblind` flag yet. It is easy to add if command-line
users ask for it.

**Response:** I agree with the lean. `run_alis` should not have an `--unblind` flag at this stage. Unblinding should only be available through the dashboard for now, with confirmation and logging.

**Q1.4 — The commands.**
- **(a)** One `--extract` serves both D9 (outputs as plain files) and D10 (the plain
  model and its snips). It writes everything to a new directory beside the bundle,
  named after it, and refuses to overwrite files without `-w`.
- **(b)** A new `run_alis --pack fit.mod`, which makes a bundle from a plain fit. The
  tests need the function anyway. With the flag, a command-line user can make a bundle
  without the dashboard, and the dashboard can import an existing fit through the same
  code (F4, Stage 4).

My lean: yes to both.

**Response:** I agree to both.

**Q1.5 — Files a model reads other than its snips.** A model can also name:
- an LSF file (`resolution=lsffile(name:…)`);
- a systematics file or module (`systematics=`, `systmodule=`);
- a starting fit (`sim beginfrom`);
- an iteration module (`iterate model`);
- a custom atomic table (`run atomic`).

The only live use among the models is `examples/lsf_file/model/generate_spectra.mod`;
the context models have these lines commented out. The exception is J1358p6522's
`run atomic atomic_rjc.xml`, which is looked up beside the model first, then in
`alis/data/`. If neither exists, ALIS falls back to the default table with only a
warning, so a bundle run on another machine could silently use different atomic
data.

My lean: in this stage, a bundle holds the snips, plus a custom atomic table that is
found beside the model. A bundle whose model needs any other file stops with a message
naming the keyword. `run.json` records the SHA-256 of the atomic table used, so a
different table shows up. Other files can be packed later if they are needed.

**Response:** Yes, the most important thing to include in the packing is the atomic data file, and this should be included in the bundle.

**Q1.6 — Which runs a bundle supports.**
- **Supported:** a fit (with or without the convergence check), `sim repeat`, and
  plotting without fitting (`-j`).
- **Stopped with a message to extract first:** simulations (`sim random`,
  `sim perturb`), `iterate model` and `generate data`. Each writes many files of its
  own, or reads modules from disk.

My lean: as above.

**Response:** I agree that the supported runs should be a fit, `sim repeat`, and plotting without fitting. The other commands should stop with a message suggesting to extract first.

**Q1.7 — A bundle edited during a run, and a run that fails.** A long command-line run
may still be going when the user saves the bundle from the dashboard (S31).

I propose that:
- the run holds its outputs in memory;
- at the end, it merges them into the bundle as it then is on disk, under a lock file,
  replacing only `runs/latest/` and the manifest;
- `run.json` records the SHA-256 of the model the run used, so the dashboard can show
  that a result belongs to an older model (F12);
- a run that stops with an error leaves `runs/latest/` unchanged and writes
  `runs/last_error.json`, so a failed rerun does not wipe the last good result.

My lean: as proposed.

**Response:** Good plan. The run should hold outputs in memory and merge them into the bundle at the end, replacing only `runs/latest/` and the manifest. The SHA-256 of the model used should be recorded in `run.json`, and a failed run should leave `runs/latest/` unchanged and write `runs/last_error.json`.

**Q1.8 — Shared pixels: the test and the message.**
- **The test.** Two pixels are the same when:
  - their wavelengths agree to 1 part in 10⁹; and
  - their flux/error ratios agree to 1 part in 10⁶.

  Pixels with an error of zero or less are ignored.
- **The message.** There is one warning per pair of snips. After 10 pairs, a single
  line gives the number of further pairs, so that models with hundreds of data lines
  (the DH_orders model has 351) stay readable.
- **No setting.** There is no setting to switch the warning off, because the fit still
  runs.

My lean: as above.

**Response:** I agree with the proposed approach for handling shared pixels. Just be careful to note that two independent data files on the same wavelength grid should not be flagged, as they are not considered shared pixels. It's only shared pixels if they are extracted from the same parent data.

**Q1.9 — `out onefits` in old models.** No model in `examples/` or
`context/fitting_examples/` sets it. My lean:
- `out onefits True` stops with a message pointing to the bundle;
- `out onefits False` gives a warning and is ignored, because it asks for nothing that
  has been lost.

**Response:** onefits will be entirely removed, since it was just an experimental feature. We will not support this in any way, and all instances of it in the code should be removed. Any model that has `out onefits True` should stop with an error saying that it is an unrecognised keyword.

## Prompts

1. Read this doc, check my responses to the queries, and ask more queries if needed. If there are no further queries, please execute the tasks in order, logging each in `ALIS/claude_prompts/logs/dashboard_stage1_log.md`.