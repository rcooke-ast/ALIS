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

## Lint rollout (between 4.1 and 4.2) -- incremental adoption, not a big-bang

RJC asked for a recommendation on lint before 4.2. Measured the ground truth
first: the `alis/` package carries **~1518 ruff findings** under the default
rule set (1217 E701 `if x: return y` one-liners, 66 E722 bare excepts, 70 F841
unused variables, 39 F821 undefined names), of which only ~21 are auto-fixable.
A repo-wide rollout is therefore a multi-session task in its own right, not a
pre-4.2 step -- and `ruff --fix` is *not* semantically safe to apply blindly
(it auto-removes F401 "unused" imports, and E711 `== None` -> `is None` changes
behaviour for numpy arrays; ALIS has 6 live `== None`/`!= None` comparisons).

**Adopted instead: enforce everywhere except an explicit exclusion list.**
- `pyproject.toml`: black `force-exclude` (verbose regex), ruff
  `extend-exclude` + `force-exclude = true`, isort `extend_skip` -- 46 files
  (39 legacy `alis/` modules; the rest vendored `context/voigt_gpu/` code and
  example scripts). Both `force-exclude` settings and isort's `--filter-files`
  are *required*, because pre-commit passes explicit filenames and the plain
  exclude settings do not filter those. Verified each one individually.
- `.pre-commit-config.yaml`: added `--filter-files` to the isort hook.
- **Brought 14 files up to standard** rather than excluding them (they were
  black-only failures, ruff-clean): `config.py`, `logger.py`, `report.py`,
  `functions/user.py`, `data/convert_datFormat_to_xmlFormat.py`, and 9
  `tests/` modules. Black is AST-preserving, so this cannot change behaviour.
  Also removed one genuinely unused import (`dataclasses.fields` in
  `config.py`, F401).
- `context/voigt_gpu/numba_test.py` is excluded because **it is not Python** --
  it is byte-identical to `faddeeva.cc` (C++ saved with a `.py` extension), and
  black cannot parse it.

Result: black / ruff (pinned v0.6.9 rule set) / isort are **all clean across
every non-excluded file**, so the CI `lint` job goes green and is meaningful for
all new code, while the legacy reformat is sequenced as **Stage 6.5** (added to
`refactor_code_stage6.md`, with the F821 findings recorded as real latent bugs
-- notably `szflx`, undefined in the variable-resolution branch of four
convolution functions).

Note on the measurement: an early pass used `black --check -q`, whose `-q` flag
suppresses the "would reformat" line, so 14 files were mis-classified as clean.
Corrected by splitting black-only failures from ruff findings and re-measuring.

## Task 4.2 -- GPU Voigt profile [COMPLETE]

`Voigt.call_GPU` is implemented and gated. Agreement with `call_CPU` is
**<= 2.1e-14 absolute** across every regime tested, against a 1e-12 budget
(Q4.3), in float64 throughout (Q4.6).

**What landed**
- `alis/functions/voigt_gpu.py` (new) -- float64 numba port of the Faddeeva
  function (`_erfcx_y100`, `_erfcx`, `_sinc`, `_sinh_taylor`, `_sqr`,
  `_faddeeva_real`) plus the batched `_voigt_kernel` and a host entry point
  `evaluate()`.
- `alis/data/erfcx_coeffs.dat` (new) -- the 100x7 Chebyshev table, 5600 bytes.
  No `pyproject.toml` change was needed: the existing `package-data` glob
  `alis = ["data/*"]` already ships it.
- `alis/functions/voigt.py` -- `_gpu_supported = True`, a `call_GPU` that
  imports the kernel module lazily, and the dead PyCUDA `GPU_kernal`
  scaffolding (plus its commented `pycuda` imports) deleted. That clears the
  `SourceModule` F821 for this file; `constant.py` and `linear.py` still carry
  theirs and are noted in Stage 6.5.
- `alis/functions/lineemission.py` -- `_gpu_supported = False` (see below).
- `pytest.ini` / `tests/conftest.py` -- the `gpu` marker and `--run-gpu`.
- `tests/test_voigt_gpu.py` (new, 18 tests), `tests/test_gpu_interface.py`
  (2 tests updated/added).

**Parameter layout.** `Voigt.set_vars` hands `call_CPU`/`call_GPU` a 6-column
array, one row per transition per component: column density, redshift, *total*
Doppler b (thermal+turbulent already combined by `parin`), rest wavelength
(q-shifted when `DELTAa/a` is fitted), oscillator strength, Gamma. A
multi-line component is therefore already a batch, which is the shape the
kernel wants. The three keywords that change the arithmetic (`freq`, `logN`,
`ColDensScale`) live in `mkey` and are **per row**, so they are encoded into a
small `(nrow, 3)` device array rather than hoisted -- a model may legitimately
mix logN and linear column densities in one call.

**Two bugs in the `context/voigt_gpu/` reference, both of which had to be
fixed to meet the tolerance.** Anyone reading that example should know:
1. `simple_test.py` hardcodes `relerr = 1.0E-7`, but pairs it with the `a`,
   `c`, `a2` constants and the precomputed `expa2n2` table that the C original
   uses *only* on its `relerr <= DBL_EPSILON` path (`numba_test.py:690, 828`).
   The mismatch truncates the series early while claiming full-precision
   constants. scipy calls `Faddeeva_w` with `relerr = 0`, which the C promotes
   to `DBL_EPSILON`. **This one cost ~8 digits**: worst-case relative error
   against `wofz` was 2.3e-9 before and 3.3e-15 after, and it took the
   40-component end-to-end comparison from 2.5e-10 (a FAIL) to 2.1e-14.
   It is invisible in the example itself, which only asserts `decimal=5`.
2. `erfcx_y100` casts the interval index to `types.float32`. Harmless in
   practice (integers < 2^24 are exact) but a float64 violation; not
   reproduced.
The example also uses `v = wv*ww*((1/ww)-(1/wv))/bl` where `call_CPU` uses the
better-conditioned `v = wv*((wv/ww)-1)/bl`. The CPU form is the one ported.

**A latent bug this task exposed.** `LineEmission` subclasses `Voigt` but
replaces `call_CPU` with a different model. `_gpu_supported` is a class
attribute, so setting it on `Voigt` silently opted `LineEmission` in too --
the dispatcher would have run the *Voigt* kernel for an emission line and
returned a wrong profile with no error. Fixed by setting
`_gpu_supported = False` on `LineEmission`, and guarded generally by
`test_gpu_support_is_not_inherited_past_a_new_call_cpu`, which asserts that
any function claiming GPU support defines `call_CPU` and `call_GPU` in the
*same* class. Task 4.1 chose an explicit flag over override-detection because
inheritance made "is `call_GPU` overridden?" unreliable; this is the same
hazard seen from the other side, and the invariant now covers both.

**Design notes**
- **Placement.** `@cuda.jit` runs at *import* time, so a module-scope kernel in
  `voigt.py` would import numba on every CPU-only run. The kernel therefore
  lives in a sibling module imported from inside `call_GPU()`. The Task 4.1
  lazy boundary still holds -- `test_importing_alis_does_not_import_numba`
  passes unchanged.
- **Reduction in-thread.** One thread per pixel loops over all rows and
  accumulates the sum (`ae='em'`) or product directly, so the `(nrow, nwave)`
  intermediate `call_CPU` builds is never materialised, and the accumulation
  order matches numpy's axis-0 reduction.
- **Per-row scalars are recomputed per thread** rather than precomputed on the
  host. That is a few flops against the hundreds the Faddeeva needs, and it is
  what lets `pin` stay on the device untouched as the contract requires.
- **Constant memory (Q4.11) is well-chosen, not merely adequate.** It
  broadcasts at full speed only when a warp reads one address, and both
  lookups here are warp-uniform: `erfcx` is indexed by the damping parameter,
  a property of the transition and so identical across threads, and `expa2n2`
  by a counter that starts at 1 everywhere.
- **Keyword cache.** The encoded `(nrow, 3)` array is cached on its contents,
  since keywords come from the model file and never change during a fit while
  `call_GPU` is invoked thousands of times. `reset_key_cache()` exists for
  anything that changes CUDA context.

**Test gating.** `gpu`-marked tests run automatically wherever a device is
present and skip where one is not -- deliberately *not* the `--run-slow`
opt-in pattern, because these tests are fast (3.8 s) and hiding them behind a
flag on a machine that can run them would mean they rarely ran. `--run-gpu`
inverts the failure mode: it makes a missing GPU a `UsageError` rather than a
skip, so a run meant to exercise the GPU cannot pass silently on a broken CUDA
install. Verified both ways with `CUDA_VISIBLE_DEVICES=""`: 18 skipped / 12
passed without the flag, hard error with it.

**Validation**
- Faddeeva vs `scipy.special.wofz` over 100k points spanning the upper half
  plane (|Re z| from 1e-8 to 1e4, Im z from 0 to ~30, plus Voigt-shaped and
  zero-damping slabs): max absolute 2.2e-16, max relative 3.3e-15, median
  relative **0.0** (bit-identical for most points). The relative bound matters
  more than the absolute one -- `Re w` is <= 1 in the upper half plane, but the
  far-wing values are multiplied by column densities of ~1e7 to give optical
  depths of order unity, so an absolute-only check would miss a truncated
  series.
- End-to-end flux, worst case per regime: damped logN=20, weak logN=12.5,
  saturated, zero damping, off-grid z=0.5, b=1.5, b=200, 40-component product
  and sum, linear column density, frequency axis, mixed keywords across rows.
  Worst overall **2.1e-14** (the 40-component sum, where 40 terms each ~1e-16
  accumulate).
- `-m unit --run-gpu`: 90 passed. Stage 0 fast gate: see below.

**Kernel-only timing** (arrays already resident; RTX 2080 Ti vs one CPU core):

| nwave | nrow | CPU | GPU | speed-up |
|---|---|---|---|---|
| 1 000 | 1 | 0.115 ms | 0.128 ms | **0.9x** |
| 10 000 | 1 | 0.800 ms | 0.152 ms | 5.2x |
| 10 000 | 20 | 16.0 ms | 0.640 ms | 25x |
| 100 000 | 20 | 158 ms | 3.79 ms | 42x |

The 0.9x row is the useful one for **Task 4.3**: below roughly 1e4
pixel-components the launch overhead eats the gain, so the dispatcher's
small-array CPU fallback needs a threshold around there, measured rather than
guessed.

**Not done here** (belongs to 4.3+): host<->device transfer, batching of
same-type components, multi-GPU dispatch, and the `ngpus` argument. `call_GPU`
returns a device array and does not transfer, per the contract.

**Stage 0 gate.** `-m fast --run-gpu`: **63 passed, 110 deselected in 4:56**
(0 failures). `-m unit --run-gpu`: 90 passed. The CPU path is unchanged --
`call_CPU` was not touched, and the only edits to `voigt.py` were the opt-in
flag, the new `call_GPU`, and deleting the dead PyCUDA method.
