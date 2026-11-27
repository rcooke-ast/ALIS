# Changelog

All notable changes to ALIS are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The documentation's build, for readthedocs (`alis-fitting.readthedocs.io`):
  the site's sources in `doc/source/`, written in Markdown (MyST) with the furo
  theme; `.readthedocs.yaml`, which builds it as HTML and fails on a warning; and
  `doc/source/_ext/`, the site's own Sphinx extension, with a highlighter for model
  files (code blocks fenced as `alis`) and one for ALIS's messages
  (`alis-message`). Every page of the site has its address, which it keeps;
  `tests/test_docs.py` checks that each exists and lists its sources, and that no
  page names what the site leaves out (Documentation Stage 1).
- `CITATION.cff`: how to cite ALIS (Cooke et al. 2014, ApJ, 781, 31), which GitHub
  offers as "Cite this repository"; the documentation's About ALIS page gives the
  same citation, with the licence and acknowledgements (Documentation Stage 1).
- A `.mod.out` records the sub-pixels per pixel its fit used, `nsubpixel=N` on each
  data line (as `bufferpix` records its pixels), and ALIS uses a data line's
  `nsubpixel=N` as given. ALIS works the sub-pixels out from the starting values,
  and a `.mod.out` starts from the best fit, which can ask for others: run again,
  `examples/lsf_spline`'s `.mod.out` gave a model that differed from its fit by up
  to 2.3 times the error; it now gives the fit's model exactly. ALIS warns, once at
  load, when a recorded `nsubpixel` leaves a profile with fewer than its snip's
  `nsubsigma` sub-pixels across its standard deviation. With
  `nsubpixel_renew`, none is written. The GUI reads it.
- ALIS warns where a profile is left with too few sub-pixels, fewer than its
  snip's `nsubsigma` across its standard deviation, and of nothing else about the
  sub-pixels: at the end of a fit, of each snip whose best fit asks for more than
  the fit used (its sub-pixels are worked out from the starting values, and a
  profile that narrows during the fit can be left short; not with
  `nsubpixel_renew`); and at load, of snips that `nsubpixel_max` caps below what
  their lines ask for (it capped them in silence), saying what to raise it to.
- GUI Stage 14: a data line's `nsubpixel=N` (from a `.mod.out` opened with
  Open a fit (.mod)…, or typed) is kept as written by every edit of the line, by
  Export… and by the bundle, and the GUI never adds one (Continue from best
  fit copies values alone); Set LSFSpline knots… draws the kernel on it; the
  pre-flight check notes a snip whose line records too few, with a fix on a
  right-click, Work them out again, that removes the keyword (one step); and
  Warnings and options to resolve… lists ALIS's warnings of too few sub-pixels,
  each with what resolves it.
- `lsffile` reads an LSF file whose first column is Δv in km/s, with `unit:kms`
  (`lsffile(name:k.dat,unit:kms)`): its kernel is sampled at each sub-pixel's
  velocity offset from the centre one, as `vfwhm`'s Gaussian is, and its buffer is
  its width taken to Å at each end of the fit range. Without the keyword, or with
  `unit:A`, the first column is Δλ in Å, as before. The GUI sets it with a
  drop-down beside Choose a file… in the Snip box, and when a file is chosen on the
  Data tab, showing the kernel's FWHM in both units.
- GUI Stage 13 (users' feedback): Warnings and options to resolve… on the
  Fit tab's Results, listing a run's warnings (stopped at chisq maxiter, a failed
  convergence check, ALIS's own warnings, now kept in `run.json`) with a suggestion
  for each, and those warnings in red in Fit statistics, ⚠ in the run history and in
  the status bar; a second page of Help → Keyboard shortcuts listing every key and
  gesture outside the menus, and Keys… beside the tabs for the tab shown; an LSF
  file in km/s (below); a run whose outputs do not read says why, and shows its
  statistics.
- GUI Stage 12 (users' feedback): every view keeps its scroll and zoom through
  each step, and a project opens where it was left (the window's size and position
  in the preferences); ⇧ Shift with the wheel zooms the flux axis, and ↔ and ↕ beside
  ⌂ take one axis home; Ignore warning for two snips that share pixels but are
  different data; a drag on a column density that a link computes moves the link's
  one free input; a component excluded from every ion can take an ion it never had;
  Keep a copy in the project, and File → Save a copy with spectra…, so that a project
  opens elsewhere with no relinking; Set LSFSpline knots…, with Delta Function, for
  `lsfspline` in the Snip box and the Data tab. A snip loaded unassigned, and each
  dataset of a fit opened without its spectrum, keeps its pixels as the dataset's
  data, so that CLEAR leaves them and SNIP cuts from them.
- GUI Stage 11 (a fit of several spectra): Open a fit (.mod)… and Load snips…
  show where each snip goes, in a dialog that assigns them; a snip that no spectrum
  holds is cut again from the spectrum it is given, loaded with no source
  ("unassigned", in one dataset per `.mod` dataset or each in its own), or left out,
  never refused. Each spectrum of a new project takes its `.mod` dataset's FWHM.
  Renumber labels… on the Components tab; labels from the component's scheme
  (`za3`, `na3si2`); one choice of line IDs for the Data and Regions tabs.
- Project bundles (`alis/bundle.py`): a whole fit in one zip file,
  `project.model`, holding the model, its snips, the atomic table it uses, the
  outputs of its latest run and, for the ALIS GUI, its own state. Lines
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
- `alis/dashboard/`: the project model of the ALIS GUI, in plain Python with
  no Qt (GUI Stage 2). It reads a model as ALIS reads it, with the position
  of every word (`text.py`, `model.py`, the meaning of each parameter coming from
  ALIS's own loaders); makes targeted edits that keep the rest of the text byte for
  byte (`edit.py`); builds the datasets, snips, regions, systems, components,
  isotopes and continua of a project, and opens an existing fit with notices for
  untied components and isotopes (`project.py`); validates the model as it is typed
  and with ALIS's loaders (`validate.py`); masks blinded values wherever they would
  be shown, and unblinds only when confirmed (`blinding.py`); keeps one undo/redo
  history (`history.py`); plans removals with their dependents (`remove.py`); and
  makes new QSO Abs Line mode projects from a spectrum (`modes.py`).
- An optional `gui` extra (PySide6, qtpy, pyqtgraph) for the GUI's windows;
  `hypothesis` joins the `dev` extra, and a `gui-test` extra holds `pytest-qt`.
- The ALIS GUI's window (GUI Stage 3), opened with the new `alis`
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
- The GUI's project model (`alis/dashboard/`) gains `preferences.py`,
  `session.py`, `markers.py`, `sources.py`, `actions.py`, `livetext.py` and
  `align.py`, all without Qt; the windows are in `alis/dashboard/qt/`, the only part of ALIS that
  imports Qt (through `qtpy`).
- A `gui` marker for the window tests (`pytest -m gui`, off-screen), and a CI job
  that runs them.
- A `badpix` column in a data file (`columns=[...,badpix:4]`): a pixel whose
  `badpix` is 1 is never fitted, by `load_data` or in the `_fit.dat` files.
- The GUI's Data and Regions tabs (GUI Stage 4). Data: the systems,
  with Identify a feature (click a line, choose its transition); the datasets table
  (the reference, FWHM and shift free, fixed or tied, the zero level, the source's
  checksum), Add file…, renaming and removing with a preview; the coverage of each
  transition; an imported fit's structure and Confirm. Regions: the transitions of
  an ion strongest first, with keys (↑ ↓, [ ], − +) that act only on the tab; one
  spectrum per dataset with a velocity axis; SNIP and CLEAR; Add regions to all (one
  set of regions for a transition, the same in every dataset) and Tweak dataset
  regions (one dataset), each adding with a drag and leaving pixels out with a
  right-drag; the snip's edges (moved outwards, re-cut from the
  source); Flux |
  Normalised; line IDs (strong lines, with the count of the others); the continuum's
  function and order, Auto first guess (three clipping passes of rising order, then
  the order by BIC), its table of orders, knots, and sharing between datasets; and the
  pixels fitted twice with their two fixes (keep, or merge after a preview), for
  every dataset at once. New modules `lines.py`, `snips.py`, `continuum.py` and
  `datasets.py` (no Qt), and `qt/plots.py`, `qt/data.py` and `qt/regions.py`.
- The GUI's Components tab (GUI Stage 5). The ion navigator (each
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
- The GUI's Fit tab (GUI Stage 6), which completes its version 1. Run
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
- The GUI's Stage 7, from Ryan Cooke's list of changes and bug fixes:
  - an earlier fit's snips and `.mod` are loaded, through one File-menu
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
  - from Ryan Cooke's review of the stage: Inspect's view choices are drop-downs; each
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
- The GUI's Stage 8, from users' feedback:
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
- The GUI's Stage 9, the Figure tab (it was the Plot tab, shown greyed since
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
- `alis/data/alis_publication.mplstyle`: the publication style of the GUI's
  plotting scripts, from the matplotlibrc of Ryan Cooke's published figures.

### Fixed
- Each data line's `lsffile`, `apod` and `lsf` resolution keeps its own keywords.
  ALIS loads every line of a function on one instance, and these three kept a
  line's keywords there for the next: a line that named no `index:` took the line
  before's (it now draws a kernel at random, as it does alone); a line that named no
  file (`lsffile`, `lsf`) or no `kind:` (`apod`) loaded with the line before's
  (it is now refused: "… must be set"); a keyword a line did not name (`lsf`'s
  `grating`, `slit`, `cen_wave`) took the line before's value, not the default; and
  a `blindrange` given on two lines gathered both. A model whose lines each name
  their keywords loads as before.
- An `lsfspline` resolution keeps its `locations` in the `.mod.out`
  (`lsfspline(values…;locations:…)`): they were left out, so the output model could
  not be read back ("locations must be set"), by `run_alis` or by the GUI,
  which then showed no best fit and no Results for the run. The GUI reads a
  run stored before the fix by taking the `locations` from the copy of the input
  model at the end of its `.mod.out`.
- GUI Stage 13: a dataset worked out from a fit takes a name no other has,
  its number in parentheses ("K7701 (2)"), as Load snips… now does too; File → Save a
  copy with spectra… opens the Save dialog (it opened the Open dialog); an LSF file
  chosen for a fit whose `.mod` is kept beside its data goes inside the project
  (it went outside it, where neither the GUI nor ALIS found it); a hidden value
  quoted after a word ending in "line" ("lsfspline 8000") is masked like any other.
- GUI Stage 13: the fitting process no longer aborts as it ends ("Python quit
  unexpectedly", SIGABRT once its result was stored): its thread for Stop waited on
  the standard input through Python's buffered reader, whose lock Python's shutdown
  could not take. Closing the window with Keep it running no longer waits for the
  fit's next iteration.
- GUI Stage 12: each rubidium isotope has its own name (87Rb I was shown as
  Rb I); Save writes the spectra kept in a project.
- `lsfspline`'s kernel is centred on its line: the model was one sub-pixel to the red
  of it (0.2 km/s for 2 km/s pixels on 10 sub-pixels), which a fit absorbed in a free
  shift or redshift, and which biased them when they were fixed.
- GUI Stage 11: a loaded `.mod` now decides the reference dataset; an imported
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
- The GUI: Exclude on a component whose lines share no redshift label gives it
  one, instead of a traceback; the Add a spectrum dialog no longer lays the column
  table over its hint; pure thermal no longer writes a b_turb that ALIS refuses.
- The GUI on Linux: a fit run from the GUI (Run, random restarts, the
  convergence check) never got past its first iteration. The fitting process read
  Stop from its standard input on a thread, and each worker that ALIS's Pool forked
  for the Jacobian waited forever for that input's lock. macOS starts its workers
  with `spawn`, and was not affected.
- The GUI's Fit · Results: the Instrument group (FWHMs, shifts, zero levels)
  is open by default, and a closed group says how many rows it hides; shifts and
  FWHMs are named by their dataset and snip, as zero levels are.
- The GUI's model text: deleting a hidden value (`▒▒▒▒`) erased it without
  asking, leaving, for example, `blindrange=` with nothing after it. Deleting or
  retyping a hidden value alone now asks for the new value, as cutting into one
  does; deleting or retyping its whole line is still made as typed.
- `run ncpus` given on the command line (`--set "run ncpus -1"`) is counted as in
  a model file, so -1 is all CPUs bar one; it used to be set as written, and the
  fit stopped ("Number of processes must be at least 1"). A count that leaves no
  CPU (`run ncpus 0`, or -20 on a computer with 8) fits with one, with a warning,
  rather than stopping.

### Changed
- The `docs` extra installs `sphinx`, `furo` and `myst-parser`, in place of
  `sphinx-rtd-theme` (Documentation Stage 1).
- `pyproject.toml` gives the licence as setuptools now reads it, an SPDX expression
  (`license = "BSD-3-Clause"`, with `license-files = ["LICENSE"]`), in place of a
  table and the licence classifier, which setuptools deprecates and will stop
  building in 2027. Building ALIS needs setuptools 77 or later (Documentation
  Stage 1).
- Every text of the GUI (`alis`) that a user reads calls it the GUI: its window's
  title ("ALIS GUI — *project*"), About ALIS, its messages and tooltips, `alis
  --help`, the headers of the files that Export… and the figure scripts write, and
  the command a run records (`alis (GUI)`). The `.mod` panel, in its own window,
  goes "Back to the main window". A project's earlier runs keep what they
  recorded, and are listed as from the GUI. ALIS's warning on an atomic table in
  the old `.xml` format asks for new transitions through an issue on GitHub. A
  test reads every text of ALIS for the old words (Documentation Stage 1).
- Every model is worked out on at least 10 sub-pixels per pixel (`run
  nsubpixel_min`, 5 before) unless it sets its own: 5 left broad profiles on too few
  sub-pixels to integrate them well. A fit of broad lines can take up to twice as
  long; a `.mod.out` written since it records its sub-pixels (`nsubpixel=N`)
  keeps its own. The regression references that moved were refitted, checked and
  regenerated (`tests/README.md`).
- Every name of the sub-pixels follows one convention: a name that counts sub-pixels
  per pixel begins `nsubpixel`, and one that counts them across a profile's standard
  deviation begins `nsubsigma`; `_min` and `_max` bound them. `run nsubpix` is now
  `run nsubsigma` (and a data line's `nsubpix=N` is `nsubsigma=N`), `run nsubmin`
  is `run nsubpixel_min`, `run nsubmax` is `run nsubpixel_max`, and
  `run renew_subpix` is `run nsubpixel_renew`; what each does, and its default, is
  unchanged. The old names are not read: each is refused with a message that names
  its new one, in a model, through `--set` and in the GUI, whose `.mod` panel
  renames it with a right-click on its problem (one step). ALIS's log gives each snip's `nsubpixel`. A
  model function's `set_vars` takes `sampling=` (the bin type, and `nsubsigma`
  times the width of a pixel) where it took `nexbin=`.
- `lsfspline`'s first value, its smoothing, is now the FWHM in km/s of the Gaussian
  that smooths the knots (it was σ√2: a Gaussian of σ = smoothing/√2, a FWHM 1.665
  times the value), and the sub-pixels it asks for follow `vfwhm`'s. A model written
  before gets a kernel smoothed 0.60 times as wide; multiply its smoothing by 1.665
  to keep the kernel it had. Nothing converts old models.
- `run_alis --extract` writes a blinded project's hidden values as `BLIND`
  (`blindseed`, `blindrange` and the `.mod.out`'s copy of the input model too),
  as the GUI's Export… does, rather than its hidden starting values in plain
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
  nearest whole number; the GUI's figures do the same (GUI Stage 10).

### Removed
- `prepfit` (`alis/prepfit/`) and its example (`examples/prepfit/`), with
  `alis/data/atomic.dat`, the atomic table only it read: the GUI (`alis`) does what
  it did. Snips that it saved still load, with their fit ranges, through Load
  snips… (Documentation Stage 1).
- `run warn_subpix` and its warning of a snip on more sub-pixels per pixel than it
  (100): too many sub-pixels are not a problem, and ALIS warns only where a profile
  has too few. A model that sets it is refused, with a message saying so.
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
