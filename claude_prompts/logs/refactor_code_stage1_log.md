# Stage 1 refactoring log

Low-risk modernisation (Python 2 cruft removal, pyproject packaging,
pre-commit/CI, semantic versioning). Every subtask must leave the Stage 0
regression suite green. No fitting behaviour should change.

## 2026-07-21 (Prompt 1: read doc, review, raise queries)

Read `refactor_code_stage1.md` including the answered queries Q1.1–Q1.3
(runtime deps = numpy/scipy/astropy/matplotlib, drop IPython + remove
`embed()`; CI on Python 3.13+ / macOS + Linux, GPU excluded; start version
`2.0.0.dev0`). The doc is internally consistent — no stale content.

Mapped the concrete Stage 1 surface:
- **1.1** `from __future__` in 6 `alis/` modules (`alis.py`, `alfunc_base.py`,
  `alload.py`, `alsims.py`, `scripts/run_alis.py`, `alsave.py`, `almsgs.py`) +
  `setup.py`. `try: input = raw_input` in `alload.py`, `alconv.py`, `alsims.py`,
  `alsave.py` (the `alis.py:28` one is already commented out). No six/future/past
  shims. `IPython import embed` in `alis.py`, `alload.py`, `alfunc_splineabs.py`,
  `alfunc_lsfspline.py`; two live `embed()` call sites — `alload.py:1468`
  (`if debug:` block) and `alload.py:1670` (`except:`).
- **1.2** `setup.py` is old-style distutils (`version='0.1.dev0'`,
  `requires=['Python (>2.7.0)']`); entry point is `bin/run_alis` calling
  `parser()` then `main(args)` (main takes parsed args → console entry needs a
  zero-arg wrapper).
- **1.4** no `alis.__version__` today; version lives only in `setup.py`.

Raised two implementation queries (now recorded as Q1.4/Q1.5 in the doc), both
resolved by RJC:
- **Q1.4** `except: embed()` at `alload.py:1670` — remove the `try`/`except`
  entirely so the real exception propagates (call `set_vars` directly). The
  `if debug: embed(); assert False` at `alload.py:1468` is dropped with its block.
- **Q1.5** `run_alis` entry point — replace `bin/run_alis` with a
  `[project.scripts]` console entry `run_alis = alis.scripts.run_alis:console_entry`
  (zero-arg wrapper) and delete `bin/run_alis`.

No code or reference files changed in this prompt. Ready to commence 1.1 on the
next prompt.

## 2026-07-21 (Prompt 2: execute Task 1.1 — remove Python 2 cruft)

Removed all Python 2 compatibility cruft from `alis/`:
- **`from __future__` imports** deleted from all 7 modules: `alis.py`,
  `alfunc_base.py` (also dropped `unicode_literals`), `alload.py`, `alsims.py`,
  `scripts/run_alis.py`, `alsave.py` (also `unicode_literals`), `almsgs.py`.
- **`try: input = raw_input / except NameError` stubs** removed from `alload.py`,
  `alconv.py`, `alsims.py`, `alsave.py`; the already-commented stub in `alis.py`
  was deleted too.
- **IPython removed:** `from IPython import embed` deleted from `alis.py`,
  `alload.py`, `alfunc_splineabs.py`, `alfunc_lsfspline.py`. The two live
  `embed()` call sites in `alload.py` handled per Q1.4: the `if debug: embed();
  assert False` debug block dropped entirely, and the `except: embed()` around
  `set_vars(...)` replaced by a direct call (no wrapper) so the real exception
  propagates. A stray `# embed()` comment was also removed.
- Confirmed no `xrange`/`iteritems`/`has_key`/`six`/`basestring`/`__metaclass__`
  or other 2/3 shims remain.

Verified: all edited modules import cleanly
(`python -c "import alis.alis, ..."` → ALL IMPORTS OK).

Ran the fast regression suite (`pytest -m "not slow and not medium" -q`):
**57 passed, 19 deselected in 8:13** (exit 0). Stage 0 safety net stays green.

Only `alis/` source touched; no reference/golden files changed.

## 2026-07-21 (Prompt 3: execute Task 1.2 — modern packaging)

Replaced the legacy distutils packaging with a PEP 517/518/621
`pyproject.toml` (setuptools backend):
- `requires-python = ">=3.13"`; version `2.0.0.dev0` (Q1.3).
- Runtime deps: numpy, scipy, astropy, matplotlib (Q1.1).
- Optional extras: `gpu` (cupy — minimal placeholder, finalised in Stage 4),
  `dev` (pytest, ruff, black, isort, pre-commit), `docs` (sphinx,
  sphinx-rtd-theme).
- Package data: `alis = ["data/*"]` (atomic.dat/.xml/_README, molecule.dat,
  phionxsec.dat, settings.alis, converter script).
- Console entry point (Q1.5): `run_alis = alis.scripts.run_alis:console_entry`.
- Deleted `setup.py`, `bin/run_alis`, and the now-empty `bin/` dir.

Entry point: added `console_entry()` (zero-arg wrapper: `parser()` then
`main()`) to `alis/scripts/run_alis.py`, plus an `if __name__ == "__main__"`
guard that sets `__spec__ = None` (mirrors the old bin/run_alis launcher; a
multiprocessing 'spawn' re-import workaround) and calls `console_entry()`.

Harness update (test infra, not `alis/`): `tests/alisrun.py` invoked alis via
the deleted `bin/run_alis`; repointed `RUN_ALIS` to
`alis/scripts/run_alis.py` (run directly as a script → its `__main__` guard →
`console_entry`, the same path as the installed console script). Updated the
two stale `bin/run_alis` mentions in that module.

Verified: pyproject parses via `setuptools.config.pyprojecttoml.read_configuration`
(name/version/scripts correct); `console_entry` imports; `run_alis.py --help`
works; fast suite **57 passed, 19 deselected in 8:18** (exit 0).

No reference/golden files changed.

## 2026-07-22 (Prompt 4: execute Task 1.3 — pre-commit + CI)

Clarified CI scope with RJC (recorded as Q1.6): CI runs **only the `examples/`
cases**, not the `context/fitting_examples/` refactor-only fits; line length is
**88** (black default); linting lands **config-first** (no repo-wide reformat
yet; CI lints changed files only).

Marker plumbing (test infra):
- `pytest.ini`: registered two source markers — `examples` (shipped example
  fits; the CI batch) and `context` (refactor-only reference fits).
- `tests/test_regression.py`: `_params` now applies a second marker per case
  derived from its top-level dir (`case.name.split("/", 1)[0]`), alongside the
  batch marker. Verified: `-m examples` → 48 tests (no context leakage),
  `-m context` → 28, total 76.

Config added:
- `.pre-commit-config.yaml`: ruff (v0.6.9, `--fix`), isort (5.13.2), black
  (24.10.0). Runs on changed files by pre-commit's default.
- `pyproject.toml`: `[tool.black]` / `[tool.isort]` (profile=black) /
  `[tool.ruff]`, all line-length 88, target py313.
- `.github/workflows/ci.yml`: `test` job (matrix ubuntu + macos, py3.13,
  `pip install -e .[dev]`, `pytest -m examples -v`) and `lint` job (pre-commit
  on the push/PR's changed `*.py` files only, skipping cleanly when there is no
  base ref to diff). GPU stack excluded (Q1.2).
- `CLAUDE.md`: line-length note updated 79 → 88; the stale "preserve
  `from __future__`" bullet replaced with a "target 3.13+, shims removed in 1.1"
  note.

Verified: both YAML files parse; pyproject still reads (deps + gpu/dev/docs
extras); representative marker-selected run `-m "examples and fast" -k
"CNabs or powerlaw or generate"` → 6 passed, 70 deselected (55 s). The additive
marker change does not alter fitting behaviour.

No `alis/` source or reference/golden files changed in this task.

## 2026-07-22 (Prompt 5: execute Task 1.4 — versioning + CHANGELOG)

Established a single source of truth for the version and started the changelog.

- **`alis/__init__.py`** (was empty): added a module docstring and
  `__version__ = "2.0.0.dev0"` (Q1.3) — the canonical version string.
- **`pyproject.toml`**: replaced the static `version = "2.0.0.dev0"` with
  `dynamic = ["version"]` and added
  `[tool.setuptools.dynamic] version = { attr = "alis.__version__" }`, so the
  build derives the version from the module. Only place the number lives now.
- **`CHANGELOG.md`** (new): Keep a Changelog 1.1.0 / SemVer format, with an
  `[Unreleased]` section and a `2.0.0.dev0` entry summarising the Stage 1
  changes (Added / Changed / Removed).

Verified: `import alis; alis.__version__` → `2.0.0.dev0`; setuptools resolves
the dynamic attr (`read_configuration(..., expand=True)` → version
`2.0.0.dev0`); representative run `-m "examples and fast" -k
"CNabs or powerlaw or generate"` → 6 passed, 70 deselected (1:01). Adding a
version literal + docstring to `__init__.py` is import-only and does not affect
fitting.

Note: the stale `alis.egg-info/` (from the old setup.py) still reports
`1.0.dev0`; a `pip install -e .` refreshes it to `2.0.0.dev0`. Not required for
the push or CI.

Stage 1 (Tasks 1.1–1.4) complete. No reference/golden files changed.
