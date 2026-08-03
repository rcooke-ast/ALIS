# Prompt file for ALIS software refactoring -- STAGE 6

> **Usability, GUI, and documentation.** Modernise the command-line interface,
> grow the `prepfit` GUI into a single prepare/run/inspect tool, and move the
> documentation to Sphinx + ReadTheDocs (code as the source of truth) with a
> tutorial and a contribution guide. Depends on Stage 2; independent of
> Stages 3/4/5. User-facing behaviour changes here must not alter fit results
> (Stage 0 gate).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage6_log.md`.

**6.1 — CLI modernisation.**
- Replace the custom argument parser with `argparse` (or `typer`) for a
  self-documenting `run_alis --help`, tab completion, and consistent option
  naming. Preserve the existing invocation (`run_alis <model>.mod`) and options.

**6.2 — GUI.**
- Extend the existing `prepfit` GUI (matplotlib, `Qt5Agg` backend) into a single
  interface that prepares a model, runs the fit, and inspects the result,
  iterating within one tool.

**6.6 — Interactive plotting-script generation (from Stage 5.3, RJC 2026-08-01).**
- Let the user generate the Stage 5.3 standalone plotting script *interactively*
  from the GUI once a fit has completed, rather than only via a setting in the
  model file — pick the mode (`metals`, ...), adjust the panel selection and
  layout, then write the script out.
- Depends on 5.3 having landed the script emitter and its modes; this task is
  the GUI surface on top of it. Further details to be discussed in Stage 6.

**6.3 — Documentation.**
- Move from LaTeX/PDF to Sphinx + ReadTheDocs, using the *code* as the source of
  truth (the LaTeX in `doc/tex_files/` is reference only). Include a full
  tutorial (walk a user through preparing, running, and analysing a fit), the
  expanded example suite, and API autodoc. Add `CONTRIBUTING.md`.

**6.5 — Repo-wide formatting + clear the legacy ruff findings.**
> Deferred here deliberately from Stage 4.1, which introduced *incremental*
> lint adoption instead: the linters are enforced on every file except an
> explicit exclusion list in `pyproject.toml` (black `force-exclude`, ruff
> `extend-exclude`, isort `extend_skip`). That list is a shrinking to-do list --
> this task empties it.
- Scope measured 2026-07-30: **46 excluded files** (39 legacy `alis/` modules;
  the rest is vendored reference code under `context/voigt_gpu/` and standalone
  example scripts, which stay excluded permanently). Under ruff's default rule
  set the `alis/` modules carry **~1518 findings**, of which only ~21 are
  auto-fixable: 1217 E701 (`if x: return y` one-liners, which black rewrites),
  66 E722 bare excepts, 70 F841 unused variables, 39 F821 undefined names.
  *Re-measured 2026-08-03 (after Stage 4):* E701 **1215**, E722 66, F841 70,
  F821 **37**. The two lost F821s are the `voigt.py` `SourceModule` pair the
  Stage 4 GPU port removed — see the last bullet below. Reproduce with
  `ruff check --isolated --select F821 alis/`: **`--isolated` is required**,
  because every file listed here is in `extend-exclude`, so a plain
  `ruff check alis/` reports "All checks passed" and tells you nothing.
- **Do black + isort first, ruff by hand.** Black is AST-preserving and so
  cannot change behaviour; `ruff --fix` is *not* safe to apply blindly here --
  it auto-removes "unused" imports (F401), which breaks side-effect imports,
  and E711 (`== None` -> `is None`) is a genuine semantic change for numpy
  arrays, where `arr == None` returns an array and `arr is None` a bool. ALIS
  has **5 live** `== None` / `!= None` comparisons (`main.py:483,489`,
  `minimise.py:1422`, `model_eval.py:653`, `load.py:22`), plus one commented out
  at `minimise.py:1833`. (Verified 2026-08-03; the doc previously said 6 live,
  which counted the commented one.)
- Each file must stay green under the Stage 0 gate as it is un-excluded; do it
  in small batches, not one sweep, so a regression stays attributable.
- **F821 undefined names are real latent bugs, not style** (found 2026-07-30,
  worth fixing regardless of the reformat):
  - `szflx` is undefined in the wavelength-dependent-resolution branch of
    **four convolution functions** -- `afwhm.py:63,66`, `vfwhm.py:63,66`,
    `voigtconv.py:70,73`, `vsigma.py:63,66`. That branch raises `NameError` if
    reached; the shipped examples only use scalar resolution, so no test
    covers it.
  - `alis/plot.py` **700-735: 23** references to an undefined `slf` (re-counted
    2026-08-03; previously recorded as ~698-725/~20).
  - `alis/prepfit/specplot.py:391`: `self` used at module scope.
  - `alis/convergence.py:188` `nput`; `alis/functions/lsfspline.py:221`
    `sidlist`; `alis/functions/chebyshev.py:73` `sys` (missing import).
  - **`alis/load.py:611` calls `imp.load_source(...)` and `imp` is neither
    imported nor importable** — the module was deleted from the standard library
    in Python 3.12, and ALIS targets 3.13. Found 2026-08-03; not previously
    recorded. This is the worst of the F821s because the call sits inside
    `try: ... except: msgs.error("Could not import module {0:s}")`, so the
    `NameError` is swallowed and reported as a missing user module: the
    `systmodule=` (user systematics module) feature is **dead**, and anyone using
    it is told their own file is at fault. No test covers it because every
    `systmodule=` line in the repo is commented out (three DH context models).
    Replace with `importlib.util.spec_from_file_location`, and narrow the bare
    `except` so a real import error is still distinguishable.
  - `SourceModule` is the dead PyCUDA scaffolding. The Stage 4 GPU port removed
    it from `voigt.py` (which now has no PyCUDA remnant at all), so what is left
    is **`constant.py:38` and `linear.py:39`** — plus a commented-out
    `#from pycuda.compiler import SourceModule` in `lineemission.py`,
    `lsfspline.py`, `phionxs.py`, `splineabs.py`, `main.py` and `model_eval.py`.
    Clear the remaining two here.

**6.4 — Unit tests for this stage's stable surface (do last).**
- Following the cross-cutting unit-test policy
  (`claude_prompts/refactor_code_unit_tests.md`), add `unit`-marked tests for the
  Stage 6 stable surface: the CLI argument parsing / command dispatch (6.1) and
  any pure GUI-backing logic (6.2) that can be exercised without a display. Keep
  them fast and isolated (no full fits); the existing `unit` CI job picks them up
  automatically. This closes out the incremental unit-test coverage across all
  stages.

## Skills to use for this stage

- `gui-dev`, `gui-component` — exercise / scaffold the GUI.
- `build-docs` — build the Sphinx docs and report warnings / broken references.

## Context

- `alis/prepfit/` (current GUI, matplotlib `Qt5Agg`), `alis/scripts/run_alis.py`
  (current CLI entry), `doc/tex_files/` (out-of-date LaTeX, reference only),
  `doc/ALIS_workflow.md` (the up-to-date workflow reference).
- Plan Q8 (GUI toolkit — decide here).

## Queries

**Q6.1 — CLI library.** `argparse` (stdlib, no dep) or `typer` (nicer UX, new
dep)? And must the exact current CLI flags be preserved for backward
compatibility, or may they be renamed/cleaned up (with aliases)?

**Q6.2 — GUI toolkit.** Stay with matplotlib/Qt for the extended GUI, or move to
another toolkit (e.g. a web-based interface)? (Deferred from plan Q8.)

**Q6.3 — Docs hosting.** Is there a ReadTheDocs project/account to target, and
should the API reference be generated via `sphinx-autodoc` from the (now
type-annotated) code?

## Prompts

> RJC will be responsible for writing this section.
