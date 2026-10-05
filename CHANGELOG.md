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

### Fixed
- A command-line setting that repeats the default (for example
  `--set "run blind True"`) is no longer overridden by the model file, and a
  blind run asked for on the command line follows the same rules as one asked
  for in the model file.
- `save_covar` overwrites an existing FITS covariance file when told to,
  rather than failing.

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
