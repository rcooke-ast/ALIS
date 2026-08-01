# Prompt file for ALIS software refactoring -- STAGE 5

> **Data and I/O modernisation.** Add YAML/TOML model-file support *alongside*
> the existing text format (never removing text support), modernise the atomic
> data file so it needs no manual `nrows` maintenance, and add an option to emit
> standalone matplotlib plotting scripts. Depends on Stage 2; independent of
> Stages 3/4. All existing `.mod` files must still parse and produce identical
> results (Stage 0 gate).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage5_log.md`.

**5.1 — YAML/TOML model files.**
- Add YAML/TOML parsing (and writing) that maps to the same internal model
  representation as the text `.mod` format. Text support remains the default and
  must be untouched. TOML reading is stdlib (`tomllib`, 3.13); TOML writing / YAML
  need small libs (see Q5.2).

**5.2 — Atomic data modernisation.**
- Replace/augment `atomic.xml` with a human-readable, whitespace-aligned
  plain-text table (ECSV is the leading candidate since `astropy` is already a
  dependency): self-describing header, obvious rows, no manual `nrows`. Provide a
  converter from the current `atomic.xml`, and validation (units, duplicates).

**5.3 — Plotting-script output.**
- Add an option to emit a standalone matplotlib script that reproduces the
  publication-quality figure for a fit, so users can tweak the plot outside ALIS.

**5.4 — Output-writer round-trip faithfulness (deferred from Stage 0).**
> A saved `.mod.out` is documented to be a valid `.mod` input, but the Stage 0
> fixed-parameter gate found several cases where re-reading it does not
> reproduce the fit. RJC asked (Q0.13 pt 4, Q0.14) that these be fixed here.
> Each fix must keep the Stage 0 suite green, and — where the fix makes the
> plain `.mod.out.reference` a faithful re-input again — the corresponding
> hand-fixed `<name>.mod.out.reference_adjusted` in `tests/` (currently
> `examples/lsf_hst`, `examples/spline/…_splineContAbs`) should be removed so
> the gate uses the plain reference.
- **`lsf` keyword echo.** The writer emits `resolution=lsf(name=STIS,grating=…)`
  but the reader requires colons (`lsf(name:STIS,grating:…)`); re-reading a saved
  `.mod.out` crashes. Emit the reader's syntax. (`examples/lsf_hst`.)
- **`splineabs` `locations=` echo.** The writer emits an empty `locations=`
  keyword whose element count no longer matches the parameters; re-reading
  crashes. Emit a valid (or omitted) `locations=`. (`examples/…_splineContAbs`.)
- **Explicit default `damping=0.0000000`.** The writer prints the voigt default
  damping as a *suffixless* keyword, which on re-read becomes a **free**
  parameter (implicit default is fixed) and changes the DOF (helium34: DOF 618
  vs 621). The Stage 0 harness strips it as a workaround; fix the writer so the
  echo round-trips the free/fixed status, then drop the workaround
  (`alisrun.make_fixedparam_mod`).
- **Best-accepted vs rejected step.** Investigate whether ALIS can write the
  parameters of a final *rejected* trial step rather than the best accepted
  point when convergence triggers on `atol` (suspected cause of
  `examples/brokenpowerlaw`'s reference evaluating to χ²=374.98 vs the recorded
  338.15; Q0.12). This is a fitting-engine concern (Stage 3) surfacing in the
  writer — coordinate with Stage 3.
- **Covariance-PNG filename uses `str.rstrip` as a suffix strip.**
  `save.save_covar` derives the correlation-matrix image name with
  `filename.rstrip(fnspl[-1]) + 'png'`, where `fnspl = filename.split('.')`.
  `str.rstrip` removes any trailing characters *in that set*, not the suffix.
  It happens to be correct for every current covariance filename (the `.`
  terminates the strip), so this is not an active bug — but it breaks for a
  covariance name with no extension (`out covar mycovar` writes to `png`).
  Replace with `removesuffix`/`os.path.splitext` while tidying the writer.
  (Found during Stage 4 prep, 2026-07-30; deliberately deferred here.)
- **Fitted-vs-starting resolution in the echo.** The writer records the *fitted*
  resolution in the data line (e.g. `vfwhm(0.075va)`); because ALIS sizes the
  pixel-load buffer from the resolution at load time, re-reading loads a
  different pixel set. RJC worked around the affected examples by fixing their
  `vfwhm`; consider whether the loader should size the buffer independently of
  the (fittable) resolution so a saved model always reloads the same pixels.

**5.5 — Unit tests for this stage's stable surface (do last).**
- Following the cross-cutting unit-test policy
  (`claude_prompts/refactor_code_unit_tests.md`), add `unit`-marked tests for the
  Stage 5 stable surface once the I/O is reshaped: this is where the deferred
  file-format loaders (`load_fits`/`load_ascii`/`load_data`/model parsing) get
  their unit tests, using small synthetic fixtures, plus the new YAML/TOML model
  reader/writer (5.1), the atomic-data loader/converter (5.2), and the
  output-writer round-trip helpers (5.4). Keep them fast and isolated (no full
  fits); the existing `unit` CI job picks them up automatically.

## Skills to use for this stage

- `atomic-data` — add/validate/convert atomic entries.
- `run-tests` — Stage 0 gate (parsing parity between text and YAML/TOML).

## Context

- `alload.py` (`.mod` parser), `alis/data/atomic.xml` and its VOTable format,
  `alplot.py` (current plotting).
- `doc/ALIS_workflow.md` §"Atomic Data File" and §"Output Files".
- Plan Q7/Q13 (ECSV candidate; whitespace-separated, easy to add/validate).

## Queries

**Q5.1 — Atomic format confirmation.** Confirm ECSV for the atomic data (per Q13),
and whether the old `atomic.xml` should be retained as a supported input during a
deprecation period or replaced outright (with a one-off converter).



**Q5.2 — YAML/TOML dependencies.** YAML support needs `PyYAML` at runtime for
users who choose YAML, and TOML *writing* needs a small writer lib (e.g.
`tomli-w`) — both would be new runtime/optional deps (plan Q12). Do you want:
(a) TOML-only (stdlib read, optional writer), (b) YAML too (add `PyYAML` as an
optional extra), or (c) both as optional extras?

**Q5.3 — Schema.** Should the YAML/TOML schema be a faithful 1:1 mapping of the
text `.mod` structure (settings / data / model / link blocks), or a cleaner
redesigned schema (with a converter between them)?

## Prompts

1. 
