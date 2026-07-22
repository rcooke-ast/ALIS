# Prompt file for ALIS software refactoring -- STAGE 1

> **Low-risk modernisation.** Mechanical, behaviour-preserving housekeeping:
> remove Python 2 cruft, modern packaging, linting/CI, and versioning. None of
> this should change fitting results — the Stage 0 regression suite must stay
> green after every subtask. Stage 1 is independent of Stage 2+ and can proceed
> in parallel with Stage 0 once the harness exists.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage1_log.md`.

**1.1 — Remove Python 2 compatibility cruft.**
- Delete `from __future__ import absolute_import, division, print_function` and
  any `try: input = raw_input` / `raw_input` stubs across all modules in `alis/`.
- Remove other 2/3 shims if present. Run the fast regression suite afterwards.

**1.2 — Modern packaging (`pyproject.toml`).**
- Replace `setup.py` with a PEP 517/518/621 `pyproject.toml`; set
  `requires-python = ">=3.13"`.
- Declare runtime dependencies and optional extras: `[project.optional-dependencies]`
  `gpu` (GPU stack, Stage 4), `dev` (pytest, ruff, black, isort, pre-commit),
  `docs` (sphinx, theme). Keep the `run_alis` console entry point.

**1.3 — Pre-commit and CI.**
- Add `.pre-commit-config.yaml` (ruff, black, isort).
- Add a GitHub Actions workflow that runs the fast regression suite
  (`pytest -m "not slow"`) plus linting on push / PR.

**1.4 — Semantic versioning and CHANGELOG.**
- Establish a single source of truth for the version (e.g. `pyproject.toml` +
  `alis.__version__`) and start `CHANGELOG.md` (Keep a Changelog style).

## Skills to use for this stage

- `run-tests` — confirm the Stage 0 suite stays green after each change.
- `update-config` — packaging / config edits.

## Context

- Plan Q6 (Python 3.13+), Q12 (dev/optional deps acceptable; no new *runtime*
  deps without RJC sign-off).
- Existing `setup.py` and imports across `alis/` (note `IPython` is imported for
  `embed()` — see Q1.1).

## Queries

**Q1.1 — Runtime dependency list.** Please confirm the definitive runtime
dependency set (numpy, scipy, astropy, matplotlib, …). `IPython` is currently
imported in `alis.py` for `embed()` — should it be dropped from runtime (the
`embed()` calls removed) or kept as a dev-only aid?

**Response:** The definitive runtime dependencies are: `numpy`, `scipy`, `astropy`,
and `matplotlib`. The `IPython` dependency should be dropped from the runtime, and the
`embed()` calls can be removed.

**Q1.2 — CI matrix.** Which Python versions and OSes should CI cover (3.13 only?
macOS + Linux?), and should GPU tests be excluded from CI (run only where a GPU
is available)?

**Response:** CI should cover Python 3.13+, and it should run on both macOS and Linux.
GPU tests should be excluded from the main CI workflow and only run in environments
where a GPU is available.

**Q1.3 — Starting version.** What version string should v2 development start
from (e.g. `2.0.0.dev0`)?

**Response:** The starting version for v2 development should be `2.0.0.dev0`.

**Q1.4 — `except: embed()` at `alload.py:1670` (raised during Prompt 1).**
Removing the `embed()` here is not behaviour-neutral: it is a bare `except`
around `set_vars(...)` that currently swallows the real exception and drops into
a debugger. Deleting only `embed()` would leave an empty `except:` that silently
continues with `params`/`nbn` undefined. How should the error path behave?

**Response:** Remove the `try`/`except` entirely — call `set_vars(...)` directly
so the original exception propagates naturally with its real traceback. (The
debug-only `if debug: embed(); assert False` at `alload.py:1468` is dropped with
the whole `if debug:` block.)

**Q1.5 — `run_alis` entry point under pyproject (raised during Prompt 1).** The
current entry point is a `bin/run_alis` script that calls `parser()` then
`main(args)`. Keep it, or replace with a `[project.scripts]` console entry?

**Response:** Replace with a console entry point:
`run_alis = "alis.scripts.run_alis:console_entry"` (a small zero-arg wrapper that
parses then calls `main`), and delete `bin/run_alis`.

**Q1.6 — CI test scope and lint rollout (raised during Prompt 4).** The task
text says CI runs `pytest -m "not slow"`, but the markers designate `medium` as
a nightly batch, and `context/fitting_examples/` cases are refactor-only. What
should CI actually run, and how should linting land on a codebase that was never
auto-formatted?

**Response:**
- **CI runs only the `examples/` cases** (`pytest -m examples`) — the shipped
  example fits. `context/fitting_examples/` cases are for refactor development
  only and are excluded from CI (dedicated unit tests will come with ALIS v2).
  A new `examples` marker (and a `context` marker) is applied per case.
- **Line length 88** (black default); `CLAUDE.md` updated to match.
- **Linting is config-first:** add pre-commit + CI lint config now but do not
  reformat the legacy codebase; the CI lint job runs the hooks on the files
  changed by each push/PR only, so legacy files do not fail CI. A repo-wide
  reformat is deferred to a dedicated task.

## Prompts

1. Please read this doc, including my responses to your queries, check if any updates need to be made to this document before commencing, and ask further queries if needed.

2. Please read this doc, and execute Task 1.1

3. Please read this doc, and execute Task 1.2

4. Please read this doc, and execute Task 1.3

[comment]: # (During execution, I responded with the following text: The CI tests should only run the tests in the `examples/` folder. The files in `context/fitting_examples/` is only for the refactoring developing. Eventually we will make unit tests with the new       
  version of ALIS as well. So, is it possible to run all of the tests in the `examples/` folder, instead? )

5. Please read this doc, and execute Task 1.4

