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

**6.3 — Documentation.**
- Move from LaTeX/PDF to Sphinx + ReadTheDocs, using the *code* as the source of
  truth (the LaTeX in `doc/tex_files/` is reference only). Include a full
  tutorial (walk a user through preparing, running, and analysing a fit), the
  expanded example suite, and API autodoc. Add `CONTRIBUTING.md`.

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
