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
  `.mod.out` best-fit parameters (within 10% of each 1σ error), 1σ errors
  (10%), χ² (1%), the `_fit.dat` model column (`|new − ref| < 0.01 × error`
  per pixel, the error-based check of Q0.22/Q0.23), and the covariance matrix
  where a golden copy exists (1% relative with a `sqrt(C_ii·C_jj)` floor).
- **`test_fixed_param` (mode b)** — re-runs the `.mod.out.reference` with
  `chisq miniter 0` / `maxiter 0` (a zero-iteration evaluation at the best-fit
  point) and compares χ² (0.1%), DOF (exact) and the model column
  (`|new − ref| < 0.15 × error` — looser than mode (a) because the reference's
  parameters are only printed to 8 digits, which moves saturated cores).
  Skipped for blind cases.

Covariance goldens exist for 17 cases: all 16 under `context/fitting_examples/`
plus `examples/metal_line_abs/fit_spectra`, which carries one so the CI
`examples` batch exercises the covariance writer. Only mode (a) compares them —
mode (b) strips `out covar` when building its fixed-parameter input.

The `generate` example is a special case: it runs `generate_spectra.mod` and
compares the produced data file to its golden copy.

## Unit tests

Separate from the regression harness above, the `unit`-marked tests exercise the
*stable surface* of individual functions in isolation — no fits, no subprocess,
no golden files — so they run in seconds and localise failures to a single
function. They cover the pure logic added/refactored through Stages 1–3
(`utils` conversions, `config` dataclasses, the Stage 3.4 cache/Jacobian helpers
in `minimise`, `convergence`, the `report` residual math, and the stable
non-I/O parts of `load` and `logger`).

```bash
pytest -m unit                                   # whole unit batch (seconds)
pytest -m unit --cov=alis --cov-report=term-missing   # with a coverage report
```

The `unit` batch runs on every push via the `unit` CI job (Ubuntu + macOS,
py3.13), which also prints the coverage report (no threshold gate — modules are
still evolving). File-format I/O loaders and GPU code are intentionally excluded
here; their unit tests are added in the stage that reshapes them (I/O in
Stage 5, GPU in Stage 4). See `claude_prompts/refactor_code_unit_tests.md`.

Two of these files are checked in a second way, because a test over an interface
can pass while asserting nothing: `test_function_interface.py` (Stage 4.4) and
`test_shared_arrays.py` (Stage 4.5) were each run against deliberately broken
versions of the code they cover, and every invariant was confirmed to fail on
the mistake it names. Keep that property when adding to them.

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

### The GPU batch

Tests marked `gpu` need a working CUDA device. They are the other way round
from `slow`: they are fast, so they run automatically wherever a device is
present and *skip* where one is not (CI, in particular), rather than hiding
behind a flag on a machine that could run them. To exercise them deliberately:

```bash
pytest --run-gpu -m gpu     # all GPU tests
```

`--run-gpu` turns a missing GPU into a `UsageError` instead of a skip, so a run
meant to test the GPU cannot pass silently on a broken CUDA install.

Everything *else* is pinned to the CPU: `alisrun.force_cpu_backend` rewrites each
staged `.mod` with `run backend cpu` before it runs, so a regression case can
never wander onto the GPU (or into an `auto` timing probe, whose choice can vary
run to run) whatever the model file asks for.

Useful flags: `-v` (per-test names), `--durations=15` (slowest tests),
`-k <expr>` (select by name), `-x` (stop on first failure). A failing test
leaves its staged working copy under pytest's `tmp_path` for inspection;
passing tests clean theirs up.

## Before changing `alis/` (Stage 1+)

Run the full harness a few independent times and confirm it is green and
deterministic on the current code first — it is the safety net for every
subsequent change.
