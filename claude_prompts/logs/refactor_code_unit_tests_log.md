# Log -- Unit-test task (cross-cutting, Stages 1-3 stable surface)

Plan: `claude_prompts/refactor_code_unit_tests.md`.

## Prompt 1 -- generate the tests (COMPLETE)

RJC approved the plan (Q-U1: standalone doc is fine; Q-U2: list is fine for now,
and add a final unit-test task to the Stage 4-6 docs). Implemented per the plan.

**Marker + tooling**
- Registered the `unit` marker in `pytest.ini` (fast, isolated; runs on every
  push; `--strict-markers` is on so registration is required).
- Added `pytest-cov` to the `[dev]` extra in `pyproject.toml`.

**New unit-test files (all `unit`-marked, no fits/subprocess/golden files):**
- `tests/test_utils.py` -- `numtorn`/`rntonum` round-trips + `subone` toggle +
  range errors; `numtoelem` representative/special-case (Z 36/42/54)/error;
  `getreason` known + defensive-unknown.
- `tests/test_config.py` -- `_DictLike` get/set/contains/keys + KeyError;
  documented defaults incl. Stage 3 changes (`cache=True`, report/convergence
  defaults); nested `ArgFlag` access; per-instance independence.
- `tests/test_minimise_helpers.py` -- `_cache_key_spsn` (both key shapes),
  `_param_spsn_map` (inversion + shared-param + 4 fallbacks), `_slice_emab`
  (no-cache / whole-cache / subset-select / union / modelem-modelab dropped),
  `enorm` == sqrt(dot).
- `tests/test_load_units.py` -- `cpucheck` (monkeypatched core count: all/None/
  explicit/negative/over-request), `get_binsize` (A/km-s/maxonly/unknown-type),
  `getis`/`load_tied` (tied-expr global->free remap), `load_atomic`
  (schema + HI Lya present).
- `tests/test_logger.py` -- `set_verbosity` gating at verbose 0/1/2 (via a
  capturing handler), invalid-verbosity no-op, `error` reports + SystemExit.

**Extended existing files (marked the pure tests `unit`, added edge cases):**
- `tests/test_convergence.py` -- module `pytestmark = unit`; added all-fixed,
  single-restart (=> None), and `convergesig` boundary-scaling cases.
- `tests/test_fit_report.py` -- `@unit` on `_runs_z`; added `_runs_z` degenerate
  (all-same-sign / <2 nonzero / empty => 0) and two `_regions` tests
  (isin-restriction to fitted pixels; zero-error-pixel exclusion). The
  fit-driven integration test stays `fast`+`examples` (NOT `unit`).

**CI**
- New `unit` job in `.github/workflows/ci.yml` (ubuntu-latest + macos-latest,
  py3.13) running `pytest -m unit --cov=alis --cov-report=term-missing`
  (report only, no gate). Existing `test`/`lint` jobs unchanged.
- `tests/README.md`: added a "Unit tests" section (`pytest -m unit`, coverage).

**Stage 4-6 docs** (per Q-U2): added a final "unit tests for this stage's
stable surface (do last)" task -- 4.6, 5.5, 6.4 -- each pointing back to the
unit-test policy doc. (Stage 5's 5.5 is where the deferred file-format I/O
loaders get their unit tests.)

**Verification**
- `pytest -m unit`: **55 passed, 83 deselected in ~1.7s**.
- Coverage report runs (config.py 100%; large modules partial as expected --
  no gate). Full collection: 138 tests, no `--strict-markers` errors.
- One planned target found a latent source quirk, not added as a test:
  `numtoelem`'s range guard (`cnt < 1`) fires before the `subone` offset, so
  `numtoelem(0, subone=True)` exits instead of returning "H" (unlike `numtorn`,
  whose guard accounts for `subone`). Flagged to RJC; no code change made
  (this task is tests-only). `load_par_influence` was dropped from the unit
  scope -- it needs the full alfunc/model stack (integration-level, already
  exercised by `test_cache_equivalence`), not a pure unit.
- Regression harness untouched by construction (only additive test files +
  marker/dep/CI/docs); `pytest -m examples` re-run to confirm.
