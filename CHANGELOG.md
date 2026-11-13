# Changelog

All notable changes to ALIS are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dashboard Stage 11 (a fit of several spectra): Open a fit (.mod)… and Load snips…
  show where each snip goes, in a dialog that assigns them; a snip that no spectrum
  holds is cut again from the spectrum it is given, loaded with no source
  ("unassigned", in one dataset per `.mod` dataset or each in its own), or left out,
  never refused. Each spectrum of a new project takes its `.mod` dataset's FWHM.
  Renumber labels… on the Components tab; labels from the component's scheme
  (`za3`, `na3si2`); one choice of line IDs for the Data and Regions tabs.
- Project bundles (`alis/bundle.py`): a whole fit in one zip file,
  `project.model`, holding the model, its snips, the atomic table it uses, the
  outputs of its latest run and, for the ALIS dashboard, its own state. Lines
  with `blind=True` are stored hidden, and so are the outputs that reveal
  blinded values.
- `run_alis project.model` runs a bundle in memory, with nothing unpacked to
  disk, and stores the outputs back in it with a record of the run
  (`runs/latest/run.json`).
- `run_alis --pack fit.mod [project.model]` makes a bundle from a plain fit;
  `run_alis --extract project.model [DIR]` writes one out as plain files.
- A warning when the same pixel of the same data is fitted by more than one
  snip (`load.find_shared_pixels`). The fit still runs.
- `load_data` reads several data files from memory (`data=` takes a mapping
  from each file's path to its bytes).
- `alis/outputs.py`: every output file is written through one writer, which
  can keep the files in memory.
- `alis/dashboard/`: the project model of the ALIS dashboard, in plain Python with
  no Qt (dashboard Stage 2). It reads a model as ALIS reads it, with the position
  of every word (`text.py`, `model.py`, the meaning of each parameter coming from
  ALIS's own loaders); makes targeted edits that keep the rest of the text byte for
  byte (`edit.py`); builds the datasets, snips, regions, systems, components,
  isotopes and continua of a project, and opens an existing fit with notices for
  untied components and isotopes (`project.py`); validates the model as it is typed
  and with ALIS's loaders (`validate.py`); masks blinded values wherever they would
  be shown, and unblinds only when confirmed (`blinding.py`); keeps one undo/redo
  history (`history.py`); plans removals with their dependents (`remove.py`); and
  makes new QSO Abs Line mode projects from a spectrum (`modes.py`).
- An optional `gui` extra (PySide6, qtpy, pyqtgraph) for the dashboard's windows;
  `hypothesis` joins the `dev` extra, and a `gui-test` extra holds `pytest-qt`.
- The ALIS dashboard's window (dashboard Stage 3), opened with the new `alis`
  command: `alis` shows a start page (New project, Open, Import a fit, recent
  projects), `alis project.model` opens a project, `alis fit.mod` imports an existing
  fit. Without the `gui` extra it says how to install it. The window has the five
  tabs of the agreed layout, each with a marker (complete, out of date, needs
  attention, not started), menus, a toolbar and a shortcut sheet built from one list
  of actions, and the `.mod` panel: the model with blinded values masked, typed in
  with undo and redo, read again after a pause and checked, with problems marked on
  their lines. Projects are saved only by Save; autosave keeps a recovery copy every
  minute (`~/.alis/recovery/`, or `$ALIS_HOME`), offered when the project is next
  opened. Runs made by `run_alis project.model` while a project is open are kept,
  and a project changed on disk by another program is noticed. Moved spectra can be
  relinked (accepted only when their checksum matches). Blinding: the whole analysis
  or chosen lines can be blinded, which undo does not pass; an edit that would show
  a blinded value is refused; unblinding asks for a note and a confirmation, and is
  recorded in the project. Preferences are kept in `~/.alis/dashboard.json`, over
  the shipped defaults. New project lists its spectra in a table, each added in its
  own dialog with its FWHM and the role of each column of a text spectrum
  (wavelength, flux, error, continuum, a bad-pixel mask, or ignored; guessed first,
  a fourth column of 0s and 1s as ignored);
  the project's mode, "QSO Abs Line", is chosen there. The `.mod` panel's width is set
  by dragging its edge, it can be moved to a window of its own and back, and its
  "Align columns" lines the model's values up in columns (spaces only).
- The dashboard's project model (`alis/dashboard/`) gains `preferences.py`,
  `session.py`, `markers.py`, `sources.py`, `actions.py`, `livetext.py` and
  `align.py`, all without Qt; the windows are in `alis/dashboard/qt/`, the only part of ALIS that
  imports Qt (through `qtpy`).
- A `gui` marker for the window tests (`pytest -m gui`, off-screen), and a CI job
  that runs them.
- A `badpix` column in a data file (`columns=[...,badpix:4]`): a pixel whose
  `badpix` is 1 is never fitted, by `load_data` or in the `_fit.dat` files.
- The dashboard's Data and Regions tabs (dashboard Stage 4). Data: the systems,
  with Identify a feature (click a line, choose its transition); the datasets table
  (the reference, FWHM and shift free, fixed or tied, the zero level, the source's
  checksum), Add file…, renaming and removing with a preview; the coverage of each
  transition; an imported fit's structure and Confirm. Regions: the transitions of
  an ion strongest first, with keys (↑ ↓, [ ], − +) that act only on the tab; one
  spectrum per dataset with a velocity axis; SNIP and CLEAR; Add regions to all (one
  set of regions for a transition, the same in every dataset) and Tweak dataset
  regions (one dataset), each adding with a drag and leaving pixels out with a
  right-drag, as in prepfit; the snip's edges (moved outwards, re-cut from the
  source); Flux |
  Normalised; line IDs (strong lines, with the count of the others); the continuum's
  function and order, Auto first guess (three clipping passes of rising order, then
  the order by BIC), its table of orders, knots, and sharing between datasets; and the
  pixels fitted twice with their two fixes (keep, or merge after a preview), for
  every dataset at once. New modules `lines.py`, `snips.py`, `continuum.py` and
  `datasets.py` (no Qt), and `qt/plots.py`, `qt/data.py` and `qt/regions.py`.
- The dashboard's Components tab (dashboard Stage 5). The ion navigator (each
  system's ions, isotopes with their element, one Lyα forest group; Add ion… and Add
  system…); one panel per snip that holds the ion, on a velocity axis at its
  strongest transition, with the data and ALIS's own model over the continuum, the
  residuals, and a tick per component; + Add component and a click, which adds a
  component with a line for every ion of its system and each log N guessed from the
  apparent optical depth; a click selecting the nearest component, and drags
  that move it (z above the continuum; log N and b below it, N or B held keeping
  one), bounded by the model's limits, with the model evaluated as the drag goes,
  and a right drag panning; a card
  per component (free, fixed and tied values, limits, the temperature's modes,
  typed values, Exclude and Include, which comment a line out and back, Remove and
  Blind…); Add interloper… (a line of the Lyα forest, or a component of another
  system); links read and shown, and not dragged; the notices of an imported fit
  with their fixes; cross-highlighting with the `.mod` panel both ways; and a Model
  switch on the Regions tab. New modules `profile.py` (each snip's model as ALIS
  computes it, equal to `model_eval.model_func`) and `components.py` (no Qt), and
  `qt/components.py`.
- The dashboard's Fit tab (dashboard Stage 6), which completes its version 1. Run
  (Ctrl+R) fits the project as it stands, in a separate process that is handed the
  project and writes no file but the project's own; Save and Run saves first; Stop
  keeps where the fit got to, and Cancel keeps nothing; the χ² of each iteration and
  the fit in progress are drawn as it goes. Inspect: the Run box (this computer's
  backend, CPUs and GPUs as preferences, the settings the model changes, All
  settings…, a pre-flight check), the progress chart, and each snip's data, model
  and residuals with χ²ν and the runs test as ALIS's report computes them. Results:
  every parameter's value, error, state and flags (at limit, unconstrained,
  blinded, a negative continuum), the fit statistics, the correlations, and the run
  history (Commit… with a description; runs kept in the project with their models
  and snips). Continue from best fit, and Use best fit for one component or one
  continuum; Reset FWHM on the Data tab. Compare: two runs' quantities matched
  across parametrisations (total log N, log D/H, components by velocity), with Δ
  and |Δ|/σ, and both best fits on the same data. Convergence check… (ALIS's own),
  Random restarts… (drawn as `sim perturb` draws after `sim beginfrom`, assessed as
  ALIS assesses them, kept with the run, and exported in `sim perturb`'s form), and
  Review clipped pixels…. A run made by `run_alis project.model` is shown as a
  "cmd" run; the Regions tab's Model switch can draw the best fit. New modules
  `runs.py`, `runner.py`, `settings.py`, `preflight.py`, `results.py`,
  `fitview.py`, `compare.py` and `restarts.py` (no Qt), and `qt/fit.py`,
  `qt/fitresults.py`, `qt/fitcompare.py` and `qt/fittools.py`.
- `bundle.execute` runs a bundle held in memory (or handed over as bytes, which
  `bundle.read` takes) and returns its result, and `bundle.store` stores a result
  in a bundle file or in memory; `bundle.run`, which `run_alis project.model`
  uses, does both, as before. A run keeps the snips it fitted (once each, under
  `runs/files/`) so that a run can be kept with them.
- `ClassMain(iterhook=...)`: a function called after each iteration of the fit
  with the iteration, χ², the parameters and the degrees of freedom; a true return
  value stops the fit at that iteration, as Ctrl+\ does. None by default.
- Every setting of `alis.config` has a one-line description
  (`config.settings_help`), and `run_alis --list-settings` prints each with its
  default.
- `report.runs_z`, `report.chi2_deviation` and `report.region_stats`: the
  report's statistics as public functions of a region's residuals, unchanged in
  what they compute.
- The dashboard's Stage 7, from RJC's list of changes and bug fixes:
  - an earlier fit's `prepfit` snips and `.mod` are loaded, through one File-menu
    item, Open a fit (.mod)…, which offers to open the fit with its source spectrum
    (recommended);
  - a component can be excluded from every ion and included again, and D I excluded
    on its own;
  - an ion given by a ratio has its row on the cards;
  - pure thermal writes the limits ALIS needs, and a drag moves T when b_turb is
    fixed;
  - a snip can have its own shift and zero level (the Regions tab's Snip box);
  - Inspect draws the data and the model with their continuum;
  - recent projects can be removed from the start page's list, and Run is on the
    toolbar;
  - from RJC's review of the stage: Inspect's view choices are drop-downs; each
    component card lists its included and excluded ions, a click excluding or
    including one; Run is greyed, with the reason, while the model cannot run (no
    snip, no fit region, no component); and File → Export… writes only what is
    ticked (the snips as ASCII files with the columns chosen, the starting model,
    the best-fitting model, or everything `run_alis` needs), hidden values written
    as `BLIND`.
- `run_alis` runs a bundle whose `lsffile` resolution names an LSF file the
  bundle holds (`bundle.unsupported` refuses it only when the file is missing;
  `lsffile` reads it from memory, `functions/lsffile.set_files`); `--pack` packs the
  file with the snips, and `--extract` writes it back.
- The dashboard's Stage 8, from users' feedback:
  - a snip's own resolution, and a dataset's, with any of ALIS's functions of one
    parameter or one file (`vfwhm`, `Afwhm`, `vsigma`, `lsffile`, `lsf`, `apod`),
    the bins each needs written as `bintype=` on its data line; the functions of
    several parameters are set in the `.mod` panel; `lsf` says that it needs
    linetools before it is used;
  - an LSF file kept in the project, so that it runs anywhere;
  - a continuum can be fitted alone, while one of its parameters is free;
  - a temperature mode changes its own component only, pure thermal gives back the
    limit it replaced, and a drag narrows a line down to its bound;
  - fit regions shaded to the pixels' edges, long values showing their start;
  - a plain ALIS blind run's `.mod.out` is read, its hidden lines masked;
  - Export's corrected wavelength scales `vshiftscale` and `polyshift` about the
    wavelength ALIS uses, and a snip whose file cannot be read stops Export with its
    name.
  - Export ticks every column by default, and measures the velocities from the
    primary system's component with the highest column density.
- The dashboard's Stage 9, the Figure tab (it was the Plot tab, shown greyed since
  Stage 3): publication figures from the best fit of the run the Fit tab shows,
  greyed while anything is hidden. A project holds any number of figures, kept in
  its `ui/project.json`, each change a step of the history:
  - a grid of panels, filled by a preset (metals, DH, blends, helium) or by
    dragging transitions onto it; a panel may span the columns of its row, draw
    against velocity or observed or rest wavelength, and has its own ranges,
    numbers, titles, label and ticks; layouts saved as presets of one's own;
  - a transition shown in two panels is marked, naming the other panels;
  - the margins are worked out from the panels' numbers and titles, or set;
  - the preview is drawn by running the figure's own script (Draw preview);
  - Export figure… (and File → Export…) writes the figure's standalone script
    (numpy and matplotlib alone, every colour a variable at its top), the snips'
    files it reads, the style, and the PDF the script draws.
- `alis/data/alis_publication.mplstyle`: the publication style of the dashboard's
  plotting scripts, from the matplotlibrc of RJC's published figures.

### Deprecated
- `prepfit` (`alis/prepfit/specplot.py`): the dashboard (`alis`) now does what it
  does. It keeps working, and says so once when it starts.

### Fixed
- Dashboard Stage 11: a loaded `.mod` now decides the reference dataset; an imported
  fit's second snip of a transition in one dataset is shown on the Regions tab; every
  button keeps the "…" of its action; a commented-out voigt line that names its
  redshift (`redshift=…`) no longer has its b_turb read as its redshift, which made
  absorption systems that are not real.
- `lsffile` asks for the kernel's width in Å either side of the fit range, at any
  wavelength; it used to take the width as a velocity at a fixed 1309 Å, so its
  buffer was too wide at longer wavelengths and too narrow at shorter.
- `run_alis --extract`'s note about hidden starting values no longer ends with a
  reference to a design document.
- A command-line setting that repeats the default (for example
  `--set "run blind True"`) is no longer overridden by the model file, and a
  blind run asked for on the command line follows the same rules as one asked
  for in the model file.
- `save_covar` overwrites an existing FITS covariance file when told to,
  rather than failing.
- A parameter's value is read as the word without its tie label
  (`functions.base.tie_value`). It used to be read with `rstrip(label)`, which
  strips the label's characters, so a label holding the value's last digit took
  that digit too: `11.2878481n1a` was read as `11.287848`.
- `Afwhm` adds its width, in Ångströms, to the fitted range when it decides how
  much data to load, instead of scaling the range by it as if it were a fraction
  (which asked for ±21% of the wavelength at 0.1 Å). Fewer unfitted pixels are
  loaded; a Legendre continuum of a model with `Afwhm` is then expressed over that
  narrower range, so its coefficients differ while the fit does not.
- A zero level can go below zero: a `constant` in the `zerolevel` section no longer
  takes `constant`'s lower limit of 0, which is a continuum's. A trough below zero
  pinned the zero level at 0, with an error of 0. An emission `constant` keeps the
  limit; a `lim param` or `lim constant value` command still applies.
- The dashboard: Exclude on a component whose lines share no redshift label gives it
  one, instead of a traceback; the Add a spectrum dialog no longer lays the column
  table over its hint; pure thermal no longer writes a b_turb that ALIS refuses.
- The dashboard on Linux: a fit run from the dashboard (Run, random restarts, the
  convergence check) never got past its first iteration. The fitting process read
  Stop from its standard input on a thread, and each worker that ALIS's Pool forked
  for the Jacobian waited forever for that input's lock. macOS starts its workers
  with `spawn`, and was not affected.
- The dashboard's Fit · Results: the Instrument group (FWHMs, shifts, zero levels)
  is open by default, and a closed group says how many rows it hides; shifts and
  FWHMs are named by their dataset and snip, as zero levels are.
- The dashboard's model text: deleting a hidden value (`▒▒▒▒`) erased it without
  asking, leaving, for example, `blindrange=` with nothing after it. Deleting or
  retyping a hidden value alone now asks for the new value, as cutting into one
  does; deleting or retyping its whole line is still made as typed.
- `run ncpus` given on the command line (`--set "run ncpus -1"`) is counted as in
  a model file, so -1 is all CPUs bar one; it used to be set as written, and the
  fit stopped ("Number of processes must be at least 1"). A count that leaves no
  CPU (`run ncpus 0`, or -20 on a computer with 8) fits with one, with a warning,
  rather than stopping.

### Changed
- `run_alis --extract` writes a blinded project's hidden values as `BLIND`
  (`blindseed`, `blindrange` and the `.mod.out`'s copy of the input model too),
  as the dashboard's Export… does, rather than its hidden starting values in plain
  text. A project with nothing hidden is extracted byte for byte, as before.
- `run blind` is False by default: a model that does not set it is no longer
  blinded. A blind analysis is asked for with `run blind True` (or `blind=True`
  on a line).
- `chisq miniter` is 5 by default (it was 0): a fit takes at least 5 iterations
  before it may stop because χ² or its parameters have stopped changing (`ftol`,
  `xtol`).
- The pre-commit hook's black is 26.5.1 (it was 24.10.0), the version the `dev`
  extra installs, and every file outside the formatting exclude lists is now
  black- and isort-clean (formatting only).
- `plotscript`'s panel labels round a rest wavelength down, as the field names a
  transition (C II 1334 for 1334.5323, H I 1025 for Lyβ), where they rounded to the
  nearest whole number; the dashboard's figures do the same (dashboard Stage 10).

### Removed
- onefits (`out onefits`, `run_alis file.fits`): an experimental single-file
  format that no longer worked. `out onefits` is now an unrecognised setting.

## [2.0.0.dev0] - 2026-07-22

Start of the ALIS v2 development line. Stage 1 is behaviour-preserving
modernisation only: the Stage 0 regression suite stays green throughout, and
no fitting results change.

### Added
- PEP 517/518/621 `pyproject.toml` packaging (setuptools backend), replacing
  the legacy `setup.py`. Optional extras: `gpu` (Stage 4), `dev`, `docs`.
- `run_alis` console entry point (`alis.scripts.run_alis:console_entry`).
- Pre-commit configuration (ruff, isort, black) and a GitHub Actions CI
  workflow: the example regression batch (`pytest -m examples`) on Ubuntu +
  macOS / Python 3.13, plus changed-file linting.
- Single source of truth for the version (`alis.__version__`), from which
  `pyproject.toml` derives the version dynamically.
- This changelog.

### Changed
- Minimum supported Python raised to 3.13.
- Code style line length standardised to 88 (black default).

### Removed
- Python 2 compatibility cruft: `from __future__` imports, `raw_input`
  fallbacks, and IPython `embed()` debug hooks, along with the `bin/run_alis`
  launcher and the legacy `setup.py`.

[Unreleased]: https://github.com/rcooke-ast/ALIS/compare/v2.0.0.dev0...HEAD
[2.0.0.dev0]: https://github.com/rcooke-ast/ALIS/releases/tag/v2.0.0.dev0
