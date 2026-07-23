# Changelog

All notable changes to ALIS are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
