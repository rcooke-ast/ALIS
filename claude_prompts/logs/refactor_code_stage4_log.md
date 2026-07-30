# Log -- Stage 4 (GPU support and modularity)

Plan: `claude_prompts/refactor_code_stage4.md`.
Bitwise A/B baselines from the committed post-3.5 code (in `/tmp/s40base/`):
metal_line_abs full fit and DH_orders 1-iteration (volatile header lines
stripped).

## Task 4.0 -- Return-not-mutate contract + `_pinfl` freeze (IN PROGRESS)

Completes the deferred Stage 3.5.3. Two bitwise-gated steps:

**Step 1 -- Freeze `_pinfl` (per-fit constant, never recomputed during the fit).**
- `model_eval.model_func`: removed the base-call per-iteration recompute
  (`state._pinfl = load_par_influence(...)`, old line 120). `_pinfl` is now only
  ever set by the callers before evaluation.
- `main.py`: the two sim `beginfrom` branches now set
  `self._pinfl = load.load_par_influence(self, modpass['p0'])` explicitly before
  evaluating/snapshotting the loaded model (they previously relied on the
  now-removed recompute). The standard fit (main.py:178) and iterate branch
  (main.py:289) already set `_pinfl` before their fits.
- Rationale: on the fit path the old recompute landed on the throwaway
  `copy.copy(state)` and was discarded (Stage 3.5 Finding F1, empirically
  confirmed), so removing it is bitwise-neutral -- and it is the prerequisite for
  Step 2 (without the copy, that write would otherwise mutate the shared table
  every iteration).

**Step 1 verified:** metal_line_abs + DH_orders 1-iteration **BITWISE-IDENTICAL**.

**Step 2 -- Drop the per-call `copy.copy(state)` in `_minimiser_eval`.**
- `model_eval._minimiser_eval` now passes `state` straight through (no
  `copy.copy`). Safe because: (i) `_pinfl` is frozen (Step 1), so the base call no
  longer mutates the shared influence table; (ii) the remaining state writes
  (`_modfinal`/`_contfinal`/`_zerofinal`) are per-call scratch rebuilt from
  scratch each call and never read back on the fit path (main-process base call
  mutates the FitState harmlessly; workers hold a private unpickled state); (iii)
  the residual column is already returned, not mutated. This is the return-not-
  mutate seam the GPU kernel path needs; any remaining incidental scratch writes
  in the derivative branch will disappear when that path becomes a device kernel
  (4.2/4.3).
- Verified: metal_line_abs (full multi-iteration fit) + DH_orders 1-iteration
  **BITWISE-IDENTICAL** vs baseline.
- Unit test: `tests/test_model_eval.py::test_minimiser_eval_does_not_copy_state`
  (monkeypatches `myfunct`, asserts the identical `state` object passes through)
  -- the pure-surface check of the 4.0 contract; numerical equivalence is covered
  by the Stage 0 bitwise gate. (Deferred 3.5.6 eval-contract test.)

## Task 4.0 -- COMPLETE, Stage 0 gate GREEN

Full suite `pytest --run-slow -k "not (DH and J0814p5029)"`: **142 passed, 1
deselected in 30:56**. Unit batch `pytest -m unit`: **60 passed**. All changes
bitwise-identical to the pre-4.0 baseline (metal_line_abs full fit + DH_orders
1-iteration verified per step; the full multi-iteration suite confirms). No
reference/golden files changed. Task 4.0 also completes the deferred Stage 3.5.3
(return-not-mutate) and freezes `_pinfl` per RJC's set-once invariant. The
minimiser derivative path is now a return-based evaluation with no per-call
`copy.copy` -- the seam the GPU kernel path (4.2/4.3) will build on.

## Environment / housekeeping (new machine, pre-4.1)

Work moved to a second machine (Linux, py313, 4x RTX 2080 Ti, numba 0.66 with
working CUDA). Items resolved before starting 4.1, each gated by the suite:
- **macOS transfer junk.** 1723 AppleDouble `._*` files came across with
  `context/fitting_examples/`; 16 were `._*.mod.out.reference`, which the
  manifest's `rglob` treated as real cases -> `UnicodeDecodeError` at collection
  (the whole suite could not even collect). Removed by RJC.
- **matplotlib 3.9 removal.** `save.save_covar` used `matplotlib.cm.get_cmap`,
  removed in 3.9 (this box has 3.11). Replaced with the registry form
  `matplotlib.colormaps['jet'].resampled(10)`. Cosmetic only -- it colours the
  correlation-matrix png; the compared `.covar` text is written earlier.
- **`np.core.defchararray`** (private, warns on numpy 2.x) -> `np.char.add` at
  `load.py` x2 and `prepfit/specplot.py`. Verified identical values *and*
  dtypes over all 815 rows of `atomic.xml` before switching.
- **Dependency floors** set from the looser of RJC's two machines:
  numpy>=2.4.3, scipy>=1.17.1, astropy>=7.2.0, matplotlib>=3.10.8. Unbounded
  versions are how the `get_cmap` removal reached us silently.
- **CI was failing on GitHub (examples job, both OSes, both runs).** Root cause
  was *not* any of the above: `test_cache_equivalence.py` put
  `@pytest.mark.examples` on the whole test function, so its sixth case
  (`context/fitting_examples/helium34/Her36`) inherited the marker -- but
  `context/` is not in the repository, so a clean checkout hit
  `FileNotFoundError` in `copytree`. Fixed by deriving the source marker per
  case from the top-level directory, exactly as `test_regression.py` does.
  `-m examples` 55 -> 54, `-m context` 28 -> 29, totals unchanged; Her36 still
  runs locally. Verified by reproducing CI locally in a tree built from
  `git ls-files` only: 1 failed/54 passed before, **54 passed** after.
- RJC added a covariance golden to `examples/metal_line_abs/fit_spectra` so the
  CI `examples` batch exercises the covariance writer (17 covar refs in total).
  Docs updated for the new counts (`refactor_code_stage0.md`,
  `tests/README.md`, incl. two stale tolerance descriptions).

## Task 4.1 -- Formalise the CPU/GPU model interface

**The contract** (documented in the `Base` class docstring, `functions/base.py`,
so the source is the reference for `new-alfunc`/`port-to-gpu`):
- Every function implements `call_CPU`; `call_GPU` is optional.
- A function opts in by **both** overriding `call_GPU` **and** setting the class
  attribute `_gpu_supported = True`; `Base.supports_gpu()` reports it.
- `call_GPU` must match `call_CPU` to **1e-12 absolute** (Q4.3) in **float64**
  (Q4.6); it takes **device arrays** and a *batch* of components and returns a
  device array -- it launches, it does not transfer. Transfers, batching and the
  small-array CPU fallback belong to the 4.3 dispatcher.

**Why an explicit flag rather than override-detection.** Four shipped functions
(`polynomial`, `spline`, `chebyshev`, `legendre`) carried a *verbatim copy* of
the inherited stub, so "is `call_GPU` overridden?" would have reported them as
GPU-capable. Those four copies were deleted (they added nothing over the
inherited method and nothing calls `call_GPU` anywhere yet).

**Fallback fixed.** The stub called `self.call_CPU(...)` **without returning**,
so `call_GPU` silently produced `None`. It now returns the result, and its
signature mirrors `call_CPU` exactly (`ae`, `mkey`, `ncpus`) so the two are
interchangeable for the dispatcher and an unported function still works under
the GPU backend.

**Lazy numba boundary.** New `alis/gpu.py` -- `is_available()`, `device_count()`,
`unavailable_reason()`, `select_device(rank)`, `reset_probe_cache()`. Every
`numba` import happens *inside* those functions, never at module scope, so a
CPU-only install never imports numba and needs no CUDA toolchain. The probe is
cached per process and degrades to "no GPU" with a clear reason for all three
failure modes (numba absent, no device, broken CUDA) rather than raising.
`pyproject.toml`: `gpu` extra `cupy` -> `numba` (Q4.11), noting the CUDA toolkit
is the user's responsibility.

**Tests** (`tests/test_gpu_interface.py`, 11 `unit` tests, no GPU needed): the
opt-in flag and that overriding alone does not imply support; the fallback
returns `call_CPU`'s result; signature parity; the probe under absent-numba /
broken-CUDA / N-device fakes; `select_device` range handling; and -- in a clean
subprocess -- that importing ALIS does not pull in numba. The CPU-vs-GPU
numerical-equivalence tests are `gpu`-marked and belong with the kernel (4.2/4.6).

No fitting behaviour is touched: nothing in the fit path calls `call_GPU`, and
`alis/gpu.py` is not yet imported by any runtime module.
