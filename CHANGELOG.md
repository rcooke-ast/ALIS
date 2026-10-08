# Changelog

All notable changes to ALIS are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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
