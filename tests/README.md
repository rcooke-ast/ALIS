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
- **`test_bundle` (mode a, as a project bundle)** — dashboard Stage 1.7. Each
  shipped example of the `fast` batch is packed into a `.model` bundle, run with
  `run_alis case.model` from an empty directory (the run must leave nothing on
  disk but the bundle), extracted, and compared with the references exactly as
  mode (a) compares a plain run.
- **`test_bundle_matches_plain`** — two real context fits (`VMP_DLA/J0903p2628`,
  and a helium34 fit that writes `out wavecorr` files) are run both as plain
  files and as a bundle, and the two are compared with mode (a)'s tolerances.
  They are compared with each other, not with the references, so the test checks
  the bundle alone.

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
non-I/O parts of `load` and `logger`), plus the Stage 4–5 additions below.

```bash
pytest -m unit                                   # whole unit batch (seconds)
pytest -m unit --cov=alis --cov-report=term-missing   # with a coverage report
```

The `unit` batch runs on every push via the `unit` CI job (Ubuntu + macOS,
py3.13), which also prints the coverage report (no threshold gate — modules are
still evolving). See `claude_prompts/refactor_code_unit_tests.md`, which
deferred the file-format loaders and GPU code to the stages that reshape them.
Both deferrals are now discharged — GPU in Stage 4, I/O in Stage 5:

| file | what it covers |
|---|---|
| `test_load_files.py` | `load_ascii` / `load_fits` / `load_userdata` / `load_data`, on synthetic fixtures written into `tmp_path` |
| `test_load_model.py` | the `.mod` model-block parser: named vs positional parameters, tie labels, `fix` and `lim` |
| `test_save_helpers.py` | the writer: `print_model`, the data line it rebuilds, `save_covar`, and a save-then-reload round trip |
| `test_writer_round_trip.py` | the other half of the round trip — re-reading all 40 committed `.mod.out.reference` files |
| `test_atomic_mass.py`, `test_plotscript.py` | Stage 5.2 and 5.3 |
| `test_load_memory.py` | dashboard Stage 1.2: real models' data loaded from memory are identical, bit for bit, to the data loaded from disk |
| `test_shared_pixels.py` | dashboard Stage 1.3: `find_shared_pixels` on synthetic snips, and on the real overlaps of J1358p6522 and Q1243p307 |
| `test_outputs.py` | dashboard Stage 1.4: the output writer, in memory and on disk (one test runs a fit, and is marked `fast` rather than `unit`) |
| `test_bundle.py` | dashboard Stage 1.5–1.8: packing every model and extracting it byte for byte, the manifest checks, atomic writes and the lock, source spectra, hidden lines (the bundle-run tests are marked `fast`) |
| `test_dashboard_*.py` | dashboard Stage 2, the project model in `alis/dashboard/`: no Qt imported; every model read back byte for byte and split as `load_input` splits it; the parsed model agreeing with ALIS's own loaders on 92 models; targeted edits; the project's systems, components and datasets; the validator; the blinding gate (no hidden value in any string shown); undo/redo and removal; QSO Abs Line mode (the two tests that run a fit are marked `fast`). The property tests use `hypothesis` |
| `test_dashboard_{preferences,session,launch,livetext,actions,markers,sources,guard,align}.py` | dashboard Stage 3, without Qt: preferences over shipped defaults; sessions (save byte for byte, autosave and recovery, runs written meanwhile kept, conflicts on disk, Save as); the `alis` launcher and new projects; typing in the `.mod` panel kept in step with the masks; the registry of actions; the tab markers; relinking spectra; the blinding guard of the history; aligning a model's columns (every model, word for word, and as ALIS reads it). `test_dashboard_session_run.py` runs `run_alis project.model` while a project is open, and is marked `fast` |
| `test_dashboard_qt_*.py` | dashboard Stage 3, the windows, marked `gui` and run off-screen with pytest-qt (`pip install -e ".[gui,gui-test,dev]"`, then `pytest -m gui`); skipped where the gui extra or pytest-qt is missing. `test_dashboard_qt_foundation.py` (marked `unit`) needs no Qt: only `alis/dashboard/qt/` imports Qt, through `qtpy`; `alis --help`; the install message without Qt. `test_dashboard_qt_review.py` covers the changes of RJC's review (the mode in the status bar, Fit · Results, Align columns on the `.mod` panel, the panel's own window and width, New project's table of spectra and the dialog for each spectrum, its column roles with Ignore, and the bad-pixel mask). `dashboard_qt_helpers.py` makes a window whose questions and file choices the test answers |
| `test_dashboard_{lines,snips,continuum,datasets}.py` | dashboard Stage 4, without Qt: transitions and line IDs (J1358p6522's H I list, Q1243p307's coverage grid, gap and edge flags, Identify a feature); snips (SNIP and CLEAR, regions and masks, edges in and out, copying regions, bad pixels never fitted, keep and merge for pixels fitted twice), each step read by ALIS and undone byte for byte; the continuum (ALIS's own curve to 1e-12, the first guess and BIC, knots, sharing); the datasets and systems edits. The `badpix` column is in `test_load_files.py` |
| `test_dashboard_qt_{plots,data,regions}.py` | dashboard Stage 4, the spectrum view, the Data tab and the Regions tab, marked `gui`: zooming and its history, J1358p6522's 40,402 pixels drawn fast; Identify a feature, ties, Confirm, no hidden value; the transition list and its keys (only with the tab's focus), adding regions to all datasets and tweaking one, SNIP then CLEAR, one set of regions across Q1243p307's three datasets, edges re-cut from the source, the order − + and knots each one step, a hidden continuum never shown, J1358p6522's 36 shared pixels kept, Q1243p307's O I and Si II merged in three datasets, cross-highlighting |
| `test_dashboard_qt_end_to_end.py` | dashboard Stage 4.12, marked `fast`: a new project from `OI_SiII.dat` taken through the Data and Regions tabs to a fit whose column densities agree with the example's reference |

`tests/conftest.py` provides two fixtures these share: `logmsgs`, which collects
what `msgs` emits (neither `capsys` nor `capfd` sees it — the shared 'alis'
logger binds its stderr handler at import), and `atomic_data`, which loads
`alis/data/atomic.ecsv` once per session.

Several of these files are checked in a second way, because a test over an
interface can pass while asserting nothing: `test_function_interface.py`
(Stage 4.4), `test_shared_arrays.py` (Stage 4.5) and the three Stage 5.5 files
above were each run against deliberately broken versions of the code they cover,
and every invariant was confirmed to fail on the mistake it names. Keep that
property when adding to them.

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

### The window batch

Tests marked `gui` drive the dashboard's windows (dashboard Stage 3) with pytest-qt,
off-screen (`QT_QPA_PLATFORM=offscreen`, set by `conftest.py`, which also gives
pytest-qt the binding the dashboard uses, PySide6 by default). They need the `gui`
extra and pytest-qt, which is kept in its own extra because it stops pytest from
starting at all when no Qt binding is installed:

```bash
pip install -e ".[gui,gui-test,dev]"
pytest -m gui                       # every window test (~20 s)
```

Where the gui extra or pytest-qt is missing they are skipped, so the `unit` batch
and CI's `unit` and `examples` jobs stay free of Qt; CI's `gui` job runs them.

### The GPU batch

Tests marked `gpu` need a working CUDA device. They are the other way round
from `slow`: they are fast, so they run automatically wherever a device is
present and *skip* where one is not (CI, in particular), rather than hiding
behind a flag on a machine that could run them. To exercise them deliberately:

```bash
pytest --run-gpu -m gpu             # all GPU tests (~7 min)
pytest --run-gpu -m "gpu and unit"  # just the fast ones (~10 s)
```

`--run-gpu` turns a missing GPU into a `UsageError` instead of a skip, so a run
meant to test the GPU cannot pass silently on a broken CUDA install.

The batch has two layers, which catch different things (Stage 4.6):

| | what it checks | sensitivity |
|---|---|---|
| `test_voigt_gpu.py`, `test_gpu_dispatch.py` (`gpu and unit`) | the kernel against `call_CPU`, and the batched dispatch against the per-row loop | 1e-12 absolute (Q4.3) |
| `test_gpu_regression.py` | each shipped example fit that uses `voigt` (19 of 25) re-run on the GPU backend at `ngpus 1` and `4`, against the **CPU** references | ~1e-3 relative |

The second layer is deliberately blunt: it uses the ordinary Stage 0 tolerances
(1% on χ², model columns within 1% of the error bar), so it catches a broken
dispatch, a wrong reduction or a mis-bound device, not a slightly wrong profile.
Both numbers above were measured by perturbing the kernel until each layer
noticed. Neither replaces the other.

Two things stop the regression layer quietly testing nothing: it forces
`run gputhresh 0` (at the shipped default *no* example is big enough to
dispatch), and it asserts on the kernel-launch count ALIS prints at the end of
the fit.

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
