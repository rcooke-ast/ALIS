# ALIS regression test harness (Stage 0)

This directory holds the automated regression harness that gates every
refactoring stage: it runs the current ALIS on the committed example fits and
compares the output against the golden "reference" files, within agreed
tolerances. It never modifies any file in the repository — every run is staged
in a temporary copy — and it never overwrites a reference.

## Layout

- `manifest.py` — discovers the test cases (any `.mod` with a sibling
  `.mod.out.reference`, under `examples/` and `context/fitting_examples/`) and
  records, per case, the data files and their `reference_fits/` goldens, the
  covariance reference, blind/random status, runtime, and batch. Run it
  directly for a summary table: `python tests/manifest.py`.
- `alisrun.py` — stages a disposable copy of an example and runs `run_alis`
  (headless, `-p 0`); builds the fixed-parameter input.
- `compare.py` — parses `.mod.out` and applies the tolerance comparisons.
- `test_regression.py` — the pytest suite (see the two test modes below).
- `conftest.py` / `../pytest.ini` — import path and batch-marker config.

## Test modes

Each non-blind fit case has two tests; blind cases run mode (a) only.

- **`test_minimisation` (mode a)** — a full `run_alis` re-fit, comparing the
  `.mod.out` best-fit parameters (within 10% of each 1σ error), 1σ errors,
  χ² (1%), the `_fit.dat` model column (1e-4 relative), and the covariance
  matrix where a golden copy exists (1% relative with a
  `sqrt(C_ii·C_jj)` floor).
- **`test_fixed_param` (mode b)** — re-runs the `.mod.out.reference` with
  `chisq miniter 0` / `maxiter 0` (a zero-iteration evaluation at the best-fit
  point) and compares χ² (0.1%), DOF (exact) and the model column
  (`|new − ref| / max(reference_model) < 2e-3` per pixel). Skipped for blind
  cases.

The `generate` example is a special case: it runs `generate_spectra.mod` and
compares the produced data file to its golden copy.

## Running the batches

Tests are marked `fast`, `medium`, or `slow` by per-test wall-time:

```bash
pytest -m fast              # every commit: fast fits + all fixed-param evals
                            #   (except the ~4 min DH_orders eval) (~10 min)
pytest -m "fast or medium"  # nightly: adds single-object real-world fits
pytest --run-slow           # everything, including the slowest (>= 10 min
                            #   minimisations, e.g. DH/J0814p5029)
```

The `slow` batch is gated behind `--run-slow` (see `conftest.py`): a plain
`pytest` runs `fast` + `medium` and *skips* `slow`. Use `--run-slow` (on its
own, or with `-m slow` for only the slow tests) to include it. Batch
membership is by each case's reference runtime, so a regenerated reference can
move a case between batches (e.g. J1419p0829 / J1358p6522_original became slow
after regeneration).

Useful flags: `-v` (per-test names), `--durations=15` (slowest tests),
`-k <expr>` (select by name), `-x` (stop on first failure). A failing test
leaves its staged working copy under pytest's `tmp_path` for inspection;
passing tests clean theirs up.

## Before changing `alis/` (Stage 1+)

Run the full harness a few independent times and confirm it is green and
deterministic on the current code first — it is the safety net for every
subsequent change.
