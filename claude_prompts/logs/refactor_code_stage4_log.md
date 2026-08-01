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

## CI fixes and the cross-machine QA policy (post-4.2)

Three fixes after the first GitHub run of the Stage 4 branch.

**1. Headless matplotlib -- a real bug, not just a CI annoyance.**
`alis/plot.py:3` called `matplotlib.use("TkAgg")` unconditionally at module
scope, so *importing* ALIS raised `ImportError: Cannot load backend 'TkAgg'
which requires the 'tk' interactive framework, as 'headless' is currently
running` anywhere without a display. That is not only CI: it breaks batch fits
over SSH and on cluster nodes. Note `matplotlib.use()` overrides `MPLBACKEND`,
so setting the environment variable in the test would have masked it rather
than fixed it. Now honours a non-empty `MPLBACKEND` and falls back to `Agg`
when Tk is unavailable, so only on-screen plotting is lost and the fit still
runs. `test_importing_alis_does_not_import_numba` is deliberately left
importing ALIS with no backend forced, so it keeps guarding this.

**2. linetools is optional; the harness now treats it that way.**
`alis/functions/lsf.py` caught the `ImportError` with a warning but left
`ltLSF` unbound, so using the `lsf` function without linetools failed with a
bare `NameError: name 'ltLSF' is not defined` from inside the model
evaluation. `ltLSF` is now bound to `None` and both call sites go through
`_require_linetools()`, which reports what is missing and how to install it.
The harness gained a `linetools` marker, applied automatically to any case
whose `.mod` uses `lsf(` (matching that literal distinguishes it from the pure
ALIS `lsfspline(` and `lsffile(`), and auto-skipped when linetools is absent --
the same "skip where impossible" principle as the `gpu` marker. Only
`examples/lsf_hst` is affected: 2 tests, and the `examples` batch stays at 54.

**3. Cross-machine minimisation divergence -- `--skip-machine-dependent`.**
The nine mode (a) `context/` failures are now tagged `machine_dependent` and
skipped *only* when `--skip-machine-dependent` is passed, so the reference
machine and CI stay strict. This is an opt-out for working on a second
machine, not a relaxation.

**On whether identical cross-OS results are achievable: no, and chasing them
is the wrong target -- but the sensitivity is reducible.**
- Bit-identical floating point across operating systems is not deliverable by
  any package choice. `exp`/`log`/`pow` are only accurate to ~1 ulp and differ
  between glibc, macOS and Windows libm; BLAS differs (OpenBLAS / Accelerate /
  MKL) and *multi-threaded* BLAS reductions are non-deterministic even on one
  machine; compilers contract `a*b+c` into FMA differently; and SIMD width
  changes numpy's pairwise-summation blocking. Arbitrary-precision arithmetic
  (mpmath) would fix it and is far too slow to fit with.
- The quantity actually diverging is the **covariance**, not the fit. All nine
  fail `compare_covar`; only one also misses the 1% chi-squared band; all nine
  pass the 0.1-sigma parameter check. The covariance is `(J^T J)^-1` with `J`
  finite-differenced at `sqrt(eps)`, so `J` carries ~1e-8 relative noise, which
  the inverse amplifies by the condition number -- large for these blended,
  near-degenerate real-world fits. Percent-level covariance spread across
  platforms is the expected behaviour of that construction.
- **The one change that would genuinely help is analytic derivatives for the
  Voigt.** `dw/dz = -2 z w(z) + 2i/sqrt(pi)`, so the derivatives with respect
  to column density, redshift and Doppler parameter are closed-form in terms
  of the `w(z)` already being evaluated. That replaces a `sqrt(eps)`-accurate
  Jacobian with a ~1e-15-accurate one, removing the dominant amplifier, and it
  is *faster* (no extra model evaluation per free parameter per iteration). It
  does not guarantee bit-identity, but it would shrink the cross-machine
  covariance spread by orders of magnitude. Worth considering as a Stage 3/4
  follow-on; it interacts well with the GPU port, since the kernel already has
  `w(z)` in hand.
- Pinning `OPENBLAS_NUM_THREADS=1` is worth doing as a guard, but it will not
  explain the current spread: J0903p2628 gave chi-squared 5111.503747 on two
  independent runs here (current code and the extracted 182bc20), bit for bit,
  so this machine is already deterministic run-to-run and threading noise is
  not a live contributor. The divergence is genuinely between machines.

## Task 4.3 -- groundwork and design (superseded; kept for the reasoning)

Surveyed but deliberately not begun: 4.3 rewrites the hot path
(`model_eval.model_func`) that every regression test depends on, and there was
not enough context budget left to finish it safely. Starting an invasive
change to that function and stopping half way is worse than not starting.
Findings so far, so the next session does not re-derive them:

**Shape of the problem.** `model_func` (`model_eval.py:106-440`) is a triple
nested loop -- spectra `sp` x snips `sn` x model components `i` -- calling
`call_CPU` at six sites and accumulating into host arrays `modelem` /
`modelab` / `mzero` / `mcont`, then convolving and downsampling. Making the
intermediates device-resident, as the task requires, means threading a device
path through all of that while keeping the CPU path bitwise-identical.

**The decomposition, in dependency order:**
1. *GPU worker pool* (`minimise.py`): `fdjac2` already holds a persistent
   `mpPool(processes=self.ncpus)` (line ~1410) with chunked columns and
   per-chunk `_slice_emab` subset-pickling. The GPU backend sizes it to
   `ngpus`, needs `multiprocessing.get_context("spawn")` (CUDA contexts do not
   survive `fork`, Q4.8), and calls `gpu.select_device(rank)` in the child.
   Note `alis/gpu.py:select_device` was written in 4.1 for exactly this.
   Worker rank is not currently available -- chunks are dispatched by
   `starmap`-style args, so rank must be threaded through or derived from
   `multiprocessing.current_process()`.
2. *Backend resolution* -- overlaps 4.3a; `RunConfig` has `ncpus` and `ngpus`
   (`config.py:63-64`) but no `backend` field yet.
3. *Device path through `model_func`* -- the bulk of the work and the whole of
   the risk.
4. *Size threshold + CPU fallback.*

Item 1 alone is not worth landing early: it touches the regression-critical
Jacobian path while having no observable effect until item 3 exists.

**A finding that affects the plan (and Q4.9's test strategy).** The 4.2
benchmark put the CPU/GPU crossover at roughly 1e4 pixel-components (1000x1
was 0.9x, i.e. a loss; 10000x20 was 25x). The shipped examples carry 388-2762
data pixels per file, which after sub-pixellation is order 1e4 sub-pixels for
a handful of Voigt rows -- *straddling* that threshold. So the Q4.9 plan of
validating by running existing examples with `run ngpus 1` and `ngpus 4`
against the same CPU references risks being **vacuous**: the size threshold
would route many of those snips to the CPU path and the GPU kernel would never
execute. Two consequences to settle before building 4.3:
- the threshold needs to be overridable (a setting, or a test-only hook), so
  the `gpu`-marked regression runs can force the device path; and
- the 4.6 GPU regression tests should assert that the GPU path was *actually
  taken*, not merely that the answer matched -- otherwise they pass by
  falling back to the CPU and prove nothing.

**Already in place from 4.1/4.2 that 4.3 builds on:** `alis/gpu.py`
(`is_available`, `device_count`, `select_device`, `unavailable_reason`), the
`Base.supports_gpu()` / `call_GPU` contract, `Voigt.call_GPU` returning a
device array without transferring, and the `gpu` pytest marker with
`--run-gpu`.

## Task 4.3 -- Multiprocessed CPU/GPU dispatch [COMPLETE]

The backend is now an either-or choice per fit, the GPU backend distributes the
Jacobian's derivative columns over one worker per device, and the model
evaluation batches whole component groups into single kernel launches on a
device-resident wave grid. The CPU path is **bitwise-identical**.

**What landed**
- `alis/gpu_dispatch.py` (new) -- the dispatch layer: routing (`should_dispatch`),
  the device wavelength-buffer cache (`wave_device` / `note_grid` /
  `begin_iteration`), and the batched launch plus its continuum split (`batch`).
  No numba import at module scope, as in `alis/gpu.py`.
- `alis/gpu.py` -- `resolve_backend()` (the either-or decision, clamping and CPU
  fallback), `current_device()`, and the Q4.10 idle-GPU notice.
- `alis/model_eval.py` -- a 22-line GPU branch in `model_func`, taken *instead of*
  the per-row CPU loop. The CPU loop itself is untouched.
- `alis/minimise.py` -- `ngpus`/`gputhresh` on `alfit`, backend resolution before
  the first evaluation, `_make_gpu_pool` / `_gpu_worker_init` (spawn, one device
  per worker), the `prepare_iteration` device hook, and `_report_dispatch`.
- `alis/config.py`, `alis/data/settings.alis` -- the `run gputhresh` setting.
- `alis/main.py`, `alis/simulate.py` -- `ngpus`/`gputhresh` threaded to all six
  `alfit` call sites.
- `doc/ALIS_workflow.md` -- a "Fitting on GPUs" section; `tests/README.md` -- the
  `gpu` batch and `pytest --run-gpu -m gpu`.
- `tests/test_gpu_dispatch.py` (new, 30 tests: 25 hardware-free, 5 `gpu`-marked).

**Backend selection (the part of 4.3a that 4.3 needs).** `run ngpus N` (N > 0) is
the opt-in, per Q4.10; `RunConfig.backend` and the timed `auto` probe stay with
4.3a. `resolve_backend` clamps N to the devices present, and degrades to the CPU
with a warning when the GPU is unusable -- so a `.mod` written on a GPU box still
runs everywhere. When GPUs are present but idle it prints the Q4.10 notice once
per process, gated on `importlib.util.find_spec("numba")` so a CPU-only install
neither imports numba nor sees the message.

**The GPU worker pool.** `fdjac2`'s persistent Pool is sized to `ngpus` instead of
`ncpus`, created with `multiprocessing.get_context("spawn")` (a CUDA context does
not survive `fork`, Q4.8), and given an initializer that binds one device per
worker. The rank comes from a shared `Value` counter, not from
`current_process()` -- Pool worker numbering is an implementation detail, the
counter is exact. This is the one place where an initializer is right: the CPU
Pool avoids one because on `spawn` it serialises the heavy re-import at Pool
creation (Stage 3.4), but a CUDA context must be created once per worker and
before any launch. A worker that cannot bind a device warns and falls back to the
CPU rather than raising -- an exception in a Pool initializer makes the Pool
respawn the worker indefinitely.

**Dispatch shape.** Where the CPU loop evaluates one profile row at a time (Stage
3.1 made the cache per-row), the dispatcher sends a whole `(sp, sn, ea, md)`
group -- every transition of every component of one model type in one snip -- in
a single launch, and the kernel reduces it in the same row order, so the result
is not merely close to the CPU's incremental `+=` / `*=` but rounds the same way.
Two details:
- **The continuum split.** The CPU loop decides *per row* whether to accumulate
  into `mcont`, which a batched reduction loses. A mixed group therefore gets a
  second launch over just its continuum rows; an all-continuum group reuses the
  first result, and a group with none skips it.
- **The profile cache is bypassed on the device path** (it is still used, exactly
  as before, by everything routed to the CPU). Caching is a per-row optimisation
  and batching is a per-group one; keeping both would mean caching device arrays,
  which cannot be pickled to the workers. The cost is nil in practice: within an
  influenced snip only the group holding the perturbed component would ever have
  to be relaunched anyway, and the other groups in these models are `legendre` /
  `constant` continua, which have no GPU implementation and so still take the
  cached CPU path.

**Device residency.** The (shifted) sub-pixel wave grid is uploaded once and kept
on the device, keyed by `(snip, shift model, shift parameters)` plus the identity
of the sub-pixel grid itself. `renew_subpix` returns a fresh grid list from
`load_subpixels`, so a changed identity is exactly the staleness signal -- no
array comparison, no `id()` reuse hazard (the *current* grid is held by a strong
reference, and a change clears the cache outright). `prepare_iteration` releases
the buffers at the iteration boundary, which is what makes the upload
once-per-iteration rather than once-per-fit, and bounds parent-side device memory
when the shift model is free. Within an iteration the base call and every
line-search evaluation share one upload; a worker keeps its buffers for the whole
chunk.

**The size threshold.** `run gputhresh` (default 10000) is the minimum
sub-pixels x rows for a group to be launched; below it, the group falls through
to the CPU loop. The default comes from the Stage 4.2 kernel benchmark (1000x1
ran at 0.9x the CPU, 10000x1 at 5.2x). It is a real setting rather than a
test-only hook, which answers the groundwork's worry that the Q4.9 GPU
regression runs could pass vacuously: `run gputhresh 0` forces the device path,
and `_report_dispatch` prints the launch counts, so a run can be shown to have
used the GPU rather than merely asked for it.

**Bitwise gate (CPU path).** metal_line_abs (full fit + covariance) and
DH_orders (351 spectra, 1 iteration) are **BITWISE-IDENTICAL** to the pre-4.3
baseline, all 1063 collected output files. Independently confirmed by accident:
the first DH_orders "GPU" run silently stayed on the CPU (see the harness note
below) and reproduced the baseline exactly.

**GPU vs CPU numerics.** metal_line_abs on 1 GPU and on 4 GPUs are
**bitwise-identical to each other** -- distributing the columns over devices does
not change the answer -- and agree with the CPU fit to 2e-9 relative on the
covariance and ~1e-9 on the fitted parameters (the redshift differs by 1e-17
absolute against a 6.4e-7 error bar). That is far inside the Stage 0 tolerances
and consistent with the 1e-12 profile budget of Q4.3: the covariance amplifies
model differences by the condition number, exactly as recorded in the
cross-machine section above. DH_orders at 1 iteration agrees to 3e-36 relative.

**Performance on DH_orders (351 spectra, 1-iteration fit, RTX 2080 Ti x4 vs 20
cores).** Times are the fit's own "Running Time" from `.mod.out`, so data
loading and file writing are excluded.

| Backend | workers | where the Voigt ran | time |
|---|---:|---|---:|
| CPU Pool | 12 (the model's `run ncpus`) | CPU | **327 s** |
| CPU Pool | 4 | CPU | 419 s |
| GPU Pool | 4 | CPU (`gputhresh 1e9`) | 432 s |
| GPU Pool | 4 | GPU (`gputhresh 10000`, default) | **357 s** |
| GPU Pool | 4 | GPU (`gputhresh 0`, forced) | 356 s |

Run-to-run spread is about 3%: the default-threshold GPU configuration was
measured three times at 357 / 348 / 345 s.

Reading these:
- **At matched worker count the GPU backend is 1.17x faster end-to-end**
  (419 -> 357 s). Against the model's own 12-core setting it is 1.09x *slower*,
  because the box has 12 usable cores and only 4 GPUs.
- **The default threshold is already right for this model** (357 s vs 356 s
  forced): DH_orders groups are mostly above 1e4 pixel-components, so almost
  nothing is being left on the CPU by the default.
- **This benchmark understates the derivative speed-up.** Going from 4 to 12 CPU
  workers only buys 1.28x (419 -> 327 s), so most of a 1-iteration run is *not*
  in the parallel Jacobian -- it is one-off setup plus the end-of-fit covariance
  on a 75164 x N Jacobian, which a 75-iteration production fit amortises away.
  An attempt to difference one iteration out (`maxiter` 1 vs 2) failed to
  separate cleanly: both runs terminated after 2 iterations, so 419/357 s are the
  numbers that can be stood behind.

**Why the gain is ~1.2x and not the ~50x Q4.8 assumed.** It is entirely
predicted by the Stage 4.2 kernel table, and the assumption -- not the
implementation -- is what was wrong. A DH_orders snip is ~310 pixels x nsubpix 5
= ~1550 sub-pixels; batching a group of ~7-20 profiles reaches ~1e4-3e4
pixel-components, which 4.2 measured at 5x, not 50x. The 50x regime starts at
~1e6 pixel-components, i.e. only if **all 351 spectra go into one launch**. That
is the "/spectra" half of Q4.7's "batches same-type components/spectra", and it
is the one thing in 4.3 that was not built -- see below.

Two candidate optimisations were considered and rejected on the arithmetic
rather than on effort:
- *Asynchronous launches / CUDA streams* (removing the blocking `copy_to_host`
  per group). A derivative column touches a handful of snips, so a Jacobian
  issues order 1e3-1e4 round trips at ~30 us -> ~0.1-0.3 s against a 357 s run.
  Not the bottleneck.
- *Reusing a device output buffer* (avoiding one `cuda.device_array` per launch).
  Same order of magnitude, and it would need an `out=` parameter on the 4.1
  `call_GPU` contract, breaking the signature-parity test for a fraction of a
  percent.
The time is genuine kernel compute at a size the GPU is not yet good at. Only a
bigger launch changes that.

**Not done, deliberately: batching across spectra.** The dispatcher batches
components *within* a snip, not across snips. Doing the latter needs a
*segmented* kernel -- one launch over the concatenated wave grids of many snips,
with each profile row applying only to its own segment -- because a single
`call_GPU(wave, pin)` evaluates every row over every wavelength by definition.
That changes the Task 4.1 per-component contract (agreed with RJC on 2026-07-29)
and turns `model_func`'s single pass into collect-then-scatter. It is the right
next move for performance and the wrong thing to bolt onto the end of a task
that already rewrites the hot path; it should be a decision, not a side effect.
Recorded as the first candidate for 4.5, where the buffer-lifecycle work lives.

**Not done, for a concrete reason: full device residency.** The stage doc asks
for the shifted wave to be derived on-device and for emission/absorption
intermediates to stay there, "downloading only the final convolved model". Both
require GPU implementations that do not exist: the shift functions
(`vshift`/`Ashift`/...) and, above all, the convolution functions
(`vfwhm`/`lsf`/...), which are numpy FFTs. Until those are ported the model must
come back to the host to be convolved. What this costs is small and now
measured-by-construction: the alternative would save ~10 kB of transfer per snip
per evaluation (12 kB downloaded per group instead of ~2.5 kB after
convolution), i.e. it is worth doing *when the convolution itself moves to the
GPU*, not before. The wave-grid half of the requirement -- the part that is
per-iteration and read-only -- **is** implemented, and the shifted grid is cached
on the device keyed by the shift parameters, which avoids the re-upload without
needing a GPU shift kernel.

**A harness trap worth recording.** The first DH_orders GPU run showed no
speed-up and outputs bitwise-identical to the CPU baseline -- because the
benchmark *prepended* `run ngpus 4` to a `.mod` that already contained
`run ngpus 0` further down, and settings are applied in file order, so the later
line won. Any future GPU benchmark must **replace** the existing setting, not
prepend. (The accident was useful: it is an independent confirmation that the
CPU path is untouched.)

**Tests** (`tests/test_gpu_dispatch.py`, 30). Hardware-free (25): backend
resolution in all five states (unset / requested / over-provisioned / no device /
broken CUDA / unreadable value), the once-per-process idle-GPU notice, the
dispatcher lifecycle and threshold arithmetic, wave-buffer caching and its three
invalidation paths, and -- with the launch replaced by the numpy reduction it has
to reproduce -- the row reduction, the em/ab distinction and all three continuum
cases. `gpu`-marked (5): the batched launch against the per-row CPU loop for both
`ae` values and for a mixed-continuum group (all within 1e-12 of Q4.3), device
buffer reuse, and the multi-GPU Pool actually binding one distinct device per
worker under `spawn`.

**Gate.** `pytest -m unit --run-gpu`: **120 passed** (90 before this task).
Stage 0 fast batch `pytest -m fast --run-gpu`: **63 passed, 140 deselected in
5:24**, unchanged from 4.2. Lint (pinned pre-commit: ruff v0.6.9 / isort 5.13.2 /
black 24.10.0) clean on every changed file. No reference or golden file changed.

**One cost worth knowing about.** The Q4.10 idle-GPU notice has to probe to know
how many GPUs are present, and the probe imports numba (~0.3 s) and initialises
the CUDA driver (~0.5 s). So on a machine that *has* the `gpu` extra installed,
every ALIS run -- including CPU-only ones -- now pays ~0.8 s once at start-up,
which adds ~2 minutes across the ~150-run regression suite. It is nothing against
a real fit and it is what makes the feature RJC asked for possible; a CPU-only
install pays nothing, because `find_spec` finds no numba and the probe is never
reached. Cheap Linux-only alternatives (stat-ing `/dev/nvidia*`) were rejected as
platform-specific guesswork. Verified that importing numba and calling `cuInit`
in the parent does not upset the *fork*-started CPU Pool (the children make no
CUDA calls).

## Post-4.3 -- duplicate-setting warning (the general fix for the harness trap)

RJC asked whether the `.mod` files carrying `run ngpus 0` should be edited so
the 4.3 benchmark trap cannot recur. They should not, and the reasons are worth
recording:
- **It would change nothing that ships.** All 18 `.mod` files with a `run ngpus`
  line are under `context/`, which is not in the repository; `examples/` has
  none, and all 18 set it to `0` (which is also the `settings.alis` default, so
  the lines are redundant rather than wrong).
- **It would not stop the trap recurring**, which fires on any repeated key
  (`run ncpus`, `chisq maxiter`, `out covar`), not on `ngpus` specifically.
- It *would* have been harmless: `compare.parse_mod_out` only reads the header
  chi-squared/DOF and the `model read` ... `model end` block, so the settings
  block is never compared and no reference would have needed regenerating. The
  edit is simply low-value, not risky.

**What was done instead.** `load.set_params` now warns when a setting appears
more than once *within one call* -- i.e. within one file -- naming the setting
and both values. Repeats *across* calls are untouched: that is the intended
override mechanism (`settings.alis`, then the model file). The message goes to
stderr, so no `.mod.out` or golden file is affected. Verified end to end on
metal_line_abs: silent on the shipped file, and on a copy with `run ngpus 4`
prepended and `run ngpus 0` appended it reports exactly the situation that
fooled the benchmark.

**A second, latent instance of the same bug, found while checking.**
`tests/test_cache_equivalence.py` *prepended* `run cache False` / `run cache
True` to force its A/B. None of its six cases sets `run cache` today, so the
test works -- but a case that ever did would have silently run both halves with
the same setting and passed vacuously, which is precisely the failure mode the
cache A/B exists to prevent. Now strips then inserts, matching
`alisrun.make_fixedparam_mod`, which was already correct (it strips
`chisq miniter`/`maxiter`, `out covar` and `run convergence` before re-inserting
them). Those two are the only places in the harness that inject settings.

**Tests.** Six `unit` tests in `tests/test_load_units.py`: last-one-wins is
unchanged; the warning fires once with both values and the `setstr` context; it
stays silent for single settings, for repeats in comments, and across separate
calls; and the shipped `settings.alis` is checked to have no repeats, since one
there would warn on every run.

**Gate.** `pytest -m unit`: **126 passed** (120 after 4.3). Stage 0 fast batch:
**63 passed, 146 deselected in 5:23**. Lint clean.

## Task 4.3a -- Backend selection (`run backend = auto | cpu | gpu`) [COMPLETE]

**What landed**
- `alis/config.py`, `alis/data/settings.alis` -- the `run backend` setting
  (default `auto`); `alis/load.py` validates it at start-up, so a typo is an
  error rather than a fit that silently ran on the wrong backend.
- `alis/gpu.py` -- `resolve_backend(backend, ngpus)` now returns one of
  `("cpu", 0)`, `("gpu", n)` or `("probe", n)`.
- `alis/gpu_dispatch.py` -- `warm_up()`: creates the CUDA context and compiles
  and launches the kernel once.
- `alis/minimise.py` -- `_probe_backends` (the timed decision), `_run_jacobian`
  (the Jacobian split out of `fdjac2` so both backends run the identical
  computation), `_make_cpu_pool`, `_gpu_wins`, `_await_siblings` /
  `_cpu_worker_init` / `_pool_is_ready` (the warm-up barrier), and `backend` on
  `alfit` plus its six call sites in `main.py` / `simulate.py`.
- `tests/alisrun.py` -- `force_cpu_backend`, applied to every staged `.mod`.
- `doc/ALIS_workflow.md`, `tests/README.md`.
- `tests/test_gpu_dispatch.py` -- 43 tests (was 30).

**Where the probe happens, and why there.** `auto` cannot answer without a
Jacobian to time, so `resolve_backend` returns `"probe"` and `alfit` settles it
at the **first `fdjac2` call** -- which is already at `p0`, exactly the sample
the stage doc asks for. No restructuring of the LM loop was needed: `fdjac2`
already holds `fvec`, the step vector and the influence-sliced payload. The
losing backend's Jacobian is discarded and the whole fit continues on the
winner, so no fit ever mixes CPU- and GPU-computed derivative columns.

The one seam: the `p0` *base* evaluation has already happened (it is what
produced `fvec`) and is therefore always CPU. Re-running it on the winner would
desynchronise `fvec` from the Jacobian the caller is about to use, so it is left
alone; the cost is a ~1e-12 shift in the first iteration's residuals, against
`ftol` 1e-10 and a typical `atol` of 0.01.

**Warming, and the mistake it took to get right.** The stage doc is right that a
cold probe would mis-pick: measured on an RTX 2080 Ti, a first launch costs
0.58 s of CUDA context plus **0.94 s** of `@cuda.jit` compile, against 0.25 ms
steady-state. Both pools are therefore fully started before either is timed.

The first attempt warmed via `pool.map` of one warm task per worker, which is
wrong twice over: a `Barrier` **cannot be sent through `map`** at all
(`RuntimeError: Condition objects should only be shared between processes
through inheritance`), and without one a fast worker can take every warm task
while a straggler is still importing -- whose start-up then lands inside the
timed Jacobian. The working shape puts the barrier in the Pool *initializer*
(inherited via `initargs`, which is the sanctioned route) and relies on a
property that makes a single trivial task sufficient afterwards: a worker cannot
accept its first task until its initializer returns, and the initializer returns
only once every worker has reached the barrier. `_pool_is_ready` is that task.
The CPU Pool gains an initializer *only* on the probe path, so the Stage 3.4
lazy-start behaviour is untouched for normal fits.

**The measurement that justifies the whole feature.** One Jacobian of DH_orders
(351 spectra), timed by the probe itself on a 12-core / 4-GPU box:

| Backend | one Jacobian at p0 |
|---|---:|
| CPU, 12 workers | **109.9 s** |
| GPU, 4 devices | 176.7 s |

so `auto` correctly keeps that fit on the **CPU**. This is a much cleaner
measurement than the 4.3 full-run differencing, and it sharpens that entry: per
*worker* the GPU is 12 x 109.9 / (4 x 176.7) = **1.86x** faster -- there are
simply three times fewer of them. On the small `metal_line_abs` example the
probe reads 0.02 s vs 0.04 s and also picks the CPU.

**So `auto`'s main job on this hardware is to stop people using the GPU**, which
is the opposite of the assumption behind Q4.8 but is exactly what a
reproducibility-conscious default should do. Probe overhead is one discarded
Jacobian (110-177 s here), ~1% of a 75-iteration DH_orders fit -- the stage
doc's "negligible for long fits" holds.

**`backend cpu` is a genuine fast path, not just a label.** It returns before
any probe, so it never imports numba or initialises CUDA -- which is why the
Stage 0 harness now sets it (`alisrun.force_cpu_backend`, strip-then-insert on
the staged `.mod`, per the lesson recorded above). Measured side benefit: the
fast batch went from 5:23 to **4:54**, because ~54 subprocess runs stopped paying
the ~0.8 s idle-GPU probe flagged as a cost in the 4.3 entry. That cost is now
confined to `backend auto` on a machine that has the `gpu` extra.

A CLI `--backend` flag was considered and rejected: `load.optarg` runs *before*
the model file is parsed, so a flag would be silently overridden by any `run
backend` line -- a worse trap than the one this session just fixed. The setting
and the harness helper are the single mechanism.

**Resolution table** (all covered by tests):

| `run backend` | `run ngpus` | GPU present | result |
|---|---|---|---|
| `cpu` | anything | either | CPU, no probe (says so if `ngpus` was set) |
| `gpu` | unset/0 | yes | GPU on **every** device |
| `gpu` | N | yes | GPU on min(N, ndev) |
| `auto` | unset/0 | yes | CPU + the Q4.10 idle-GPU notice |
| `auto` | N | yes | **probe**, then commit |
| `gpu`/`auto` | any | no | warn, fall back to CPU |

**Bitwise gate.** metal_line_abs (full fit + covariance) and DH_orders (1
iteration) are **BITWISE-IDENTICAL** to the pre-4.3 baseline -- all 1063 files --
so neither the `backend` plumbing nor the `fdjac2` split moved the CPU numbers.
Also removed a 204 MB dead allocation spotted during that split: `fdjac2`
pre-allocated an `m x n` Jacobian that `_run_jacobian` immediately replaces.

**Gate.** `pytest -m unit --run-gpu`: **139 passed** (126 before). Stage 0 fast
batch `pytest -m fast --run-gpu`: **63 passed, 159 deselected in 4:54**. Lint
clean. All four backend modes smoke-tested end to end on metal_line_abs.

## Task 4.4 -- New-function ergonomics [COMPLETE]

The brief is "make it straightforward to add a new model function with both CPU
and GPU paths plus its own unit tests". Three things were in the way, and the
skills that were supposed to guide the work described a code base that has not
existed since Stage 2.

**1. The GPU warm-up hard-coded the Voigt.** 4.3a's `warm_up()` imported
`voigt_gpu` by name, so a second ported function would not have been warmed and
its ~0.9 s JIT would have landed inside the `run backend auto` timing probe --
the exact failure the warm-up exists to prevent. Replaced with a per-function
hook (flagged as a to-do in the 4.3a entry):
- `Base.gpu_warmup_args()` returns `(x, p, kwargs)` for one tiny throwaway
  launch, or `None`; `Voigt` returns a single Lyman-alpha profile on 64
  sub-pixels.
- `gpu_dispatch.gpu_capable_functions()` walks `Base`'s subclass tree -- *not*
  the `base.call()` registry, which also loads the user's function module and
  prints while doing it, neither of which belongs in a Pool initializer. Every
  shipped function is imported at the bottom of `functions/base.py`, so the tree
  is complete.
- `warm_up()` now warms all of them and warns by name if one claims GPU support
  without providing the hook.
Porting a second function therefore needs no edit to `gpu_dispatch.py`. With
several kernels this will compile some the model does not use; the initializer
has no model to consult, and a wasted compile is far cheaper than one inside the
timed Jacobian.

**2. Nothing checked that a new function was well-formed.** Adding one means
filling in a dozen parallel class attributes and registering it in *two* places,
and every way of getting it wrong surfaces far from the mistake: a keyword
silently ignored, an `IndexError` inside `parout`, or -- the nastiest --
`'NoneType' object is not subscriptable` from inside `set_vars` because the
function was never added to `sendatomic`. New `tests/test_function_interface.py`
(255 tests, ~1.8 s) runs every invariant over every registered function:

| Invariant | What it catches |
|---|---|
| registry key == `_idstr` | error messages naming something absent from the `.mod` |
| `_parid`/`_defpar`/`_fixpar`/`_limited`/`_limits`/`_svfmt` same length | `IndexError` far from the typo |
| `len(_parid) >= _pnumr` | miscounted parameters (`>=`, because `lsfspline` is genuinely variable-length) |
| `_limited`/`_limits` entries are pairs | malformed bounds |
| `_keywd`/`_keych`/`_keyfm` describe the same keywords | a keyword never validated, or `KeyError` when writing the model |
| `_keywd['input']` covers every parameter and keyword | parameters silently dropped from `.mod.out` |
| `_prekw` names real keywords | `msgs.bug("prekw variable ... bad argument")` |
| every module defining a `Base` subclass is registered | "unknown function" from the model file |
| everything that reads `self._atomic` receives it | the `NoneType` failure above |
| GPU-capable functions provide `gpu_warmup_args()` | a cold kernel inside the `auto` probe |

All 32 shipped functions pass every one. The atomic-data check works by passing
a sentinel through `base.call(getinst=True, atomic=...)` and asserting that
every function whose source *reads* `self._atomic` received it -- the regex
excludes the `self._atomic = atomic` assignment every `__init__` carries, so it
detects reads rather than the attribute's existence. **Each invariant was
verified to bite** by constructing deliberately broken functions (short
`_svfmt`, deleted `_keyfm` entry, missing `input` entry, bogus `_prekw`, wrong
`_idstr`, GPU flag with no warm-up hook, an atomic reader): all eight caught.

**3. The skills were stale.** `new-alfunc` still targeted
`alis/alfunc_<name>.py` and `alfunc_base.Base`, which Stage 2 removed, and told
the author to make `call_GPU` raise `NotImplementedError` -- which would break
the 4.3 dispatcher, whose whole design is that it can call `call_CPU` or
`call_GPU` uniformly (the inherited stub falls back to the CPU). Rewritten
around `alis/functions/`, the real interface table, the two registration steps,
the interface gate above, and what a test file should actually cover. Also
updated:
- `port-to-gpu`: the real `call_GPU` signature (it had `(self, x, p, ae='em')`,
  missing `mkey`/`ncpus`); the **three** opt-in requirements rather than one
  (`_gpu_supported`, same-class `call_CPU`/`call_GPU`, `gpu_warmup_args`); why
  the kernel goes in a sibling `<name>_gpu.py` (`@cuda.jit` compiles at import,
  so a module-scope kernel pulls numba into every CPU-only run); and a warning
  that a freshly ported function will report *zero* launches until
  `run gputhresh` is lowered -- otherwise the author reasonably concludes the
  port is broken.
- `port-to-gpu` also had two claims that Task 4.3 measured and disproved:
  "keep intermediates on the device ... only the final convolved model comes
  back" and the implication that batching spans spectra. Corrected to what the
  dispatcher actually does, with the reason (no GPU convolution or shift
  functions yet) and the pointer to 4.5.
- `gen-tests`: `alfunc_voigt`/`alload` -> the real paths; the `unit` and `gpu`
  markers; and "GPU tests that require **CuPy**" -> `numba.cuda` (Q4.11 chose
  numba; CuPy was the Stage 1 placeholder).
- `test-coverage`, `profile-fit`: stale `alfunc_*` / `alconv` / `alcsmin` paths
  corrected in place (one line each; no rewrite).

**Deliberately not done.** No template or scaffold *file* was added. A checked-in
template that no test exercises rots exactly the way these skills did; the
interface test file is the executable specification, and `gaussian.py` (minimal)
and `voigt.py` (full, with GPU) are the worked examples the skill points at.

**Gate.** `pytest -m unit --run-gpu`: **394 passed, 31 skipped** (139 before;
the skips are the GPU warm-up check on the 31 functions with no GPU path).
Stage 0 fast batch `pytest -m fast --run-gpu`: **63 passed, 445 deselected in
4:54**. metal_line_abs and DH_orders **BITWISE-IDENTICAL**
-- `Base` and `Voigt` gained a method and nothing else. Forced-GPU and
`backend auto` runs re-checked end to end after the warm-up refactor. Lint clean.

## Task 4.5 -- Shared-memory read-only arrays [COMPLETE]

Carried in from Task 3.4 Phase 3. The brief is to back the read-only arrays
passed to `_worker_chunk` with `multiprocessing.shared_memory`, keep the
Jacobian bitwise-identical, and benchmark memory and wall on DH_orders and a
compact fit.

**What the payload actually is.** The stage doc's figure was ~1.1 GB per
evaluation, from the Phase 2 measurement. Measured again now, on DH_orders
(413 free parameters, 12 chunks):

| Per chunk | | Per Jacobian |
|---|---|---|
| constant fit state (`functkw`) | 102.1 MB | 1.23 GB |
| profile-cache slice | 157 MB (chunk 0), 429 MB mean | 5.15 GB |
| `fvec` / `xall` / job list | 0.6 MB | 7 MB |
| | | **6.38 GB** |

So the doc's 1.1 GB was the *constant* part, and it is the smaller half. Inside
the 100.4 MB `FitState`, three lists account for 86 MB: `_wavespx`, `_contspx`
and `_zerospx`, the sub-pixel grids. The cache slices are individually smaller
than the whole cache but overlap heavily across chunks, which is why sending
them separately costs 4x what the cache itself is worth.

**Design.** New `alis/shared_arrays.py`. Each publisher owns **one** segment,
not one per array -- DH_orders would otherwise need ~4000 of them. The segment
holds a header, a pickled *skeleton* (the original structure with each large
array replaced by an offset/shape/dtype triple), then the array bytes at
64-byte alignment. What travels through `pool.map` is a `Handle`: tag, segment
name, generation, and optionally a tuple of dict keys -- ~100 bytes in place of
hundreds of megabytes. A worker attaches once and rebuilds each array as a
**view**, so the arrays exist once in RAM however many workers there are.

Two publishers, because the payloads have different lifetimes: `functkw`
(constant for the fit, but republished each Jacobian, see below) and
`compcache` (rebuilt by every base evaluation). Both are released in
`_close_pool`, after the Pool, with a `weakref` registry and an `atexit` sweep
behind them.

**Decisions worth recording.**

- *Views are read-only.* Not tidiness: 12 workers share one buffer, so a write
  is a data race across processes. `writeable = False` converts it into an
  immediate `ValueError` in the worker that did it. That the full fits run
  clean is therefore also the evidence that nothing on the derivative path
  writes to this data.
- *Republished per Jacobian, not once per fit.* `FitState._modfinal` /
  `_contfinal` / `_zerofinal` are rebound by the parent's base evaluation each
  iteration. `model_func` overwrites them before reading, so publishing once
  would probably be safe -- but "probably" is not a basis for a silent staleness
  bug, and re-copying 100 MB costs ~30 ms against a 110 s Jacobian.
- *The cache selection is preserved exactly.* Phase 2 sends each chunk only the
  entries its parameters influence; the shared path sends the same **keys**, so
  the dict a worker sees has the same entries in the same order. `_slice_emab`
  was refactored onto the new `_chunk_cache_keys` so the two paths cannot drift
  apart -- and the test for it checks the selection rule directly rather than
  the two paths against each other, which after that refactor would have been a
  tautology (it was, and the mutation check caught it).
- *The hydrated object is cached per generation.* `model_func` keys the Stage
  4.3 device wave cache on the **identity** of the sub-pixel grid list, so a
  worker that rebuilt the state per task would invalidate the GPU's resident
  buffers on every task.
- *The generation is stored in the segment as well as in the handle.* Reading
  through a stale handle then raises instead of returning whatever the segment
  holds now. This was found by a test, not by reasoning: the first version
  silently hydrated one payload's handle into another's contents.
- *`track=False` when attaching in a worker* (Python 3.13). The parent owns the
  segment; a worker that registered it with its own resource tracker would
  unlink it on exit and report it as leaked.
- *Best-effort, with a switch.* An `OSError` from `SharedMemory` -- a
  container's 64 MB `/dev/shm` is the realistic case -- warns once and falls
  back to the pickle path. `run shmem False` (new `RunConfig.shmem`, default
  True) forces that path outright.
- Only C-contiguous, non-object, non-structured arrays >= 4 kB are shared;
  anything else would have to be repacked, which is what pickle already does
  well. Dataclasses are shallow-copied and written field by field rather than
  rebuilt through `dataclasses.replace`, which re-runs `__init__` and would
  drop any field the constructor does not take.

**Carried-in items, and why two of them are still deferred.**

- *(a) Batch across spectra.* Unchanged from the 4.3 assessment: it needs a
  segmented kernel and a collect-then-scatter pass in `model_func`, which
  changes the 4.1 per-component `call_GPU` contract. The stage doc says
  explicitly that this is RJC's decision rather than a silent refactor, so it
  stays open.
- *(b) Full device residency.* Still blocked on GPU ports of the shift and
  convolution functions, and still worth only ~10 kB per snip per evaluation
  until then.
- *(c) The Stage 3.5.5 conditional `renew_subpix` recompute.* **Declined**, and
  the reason is concrete rather than an estimate of effort: no fit anywhere in
  the repository sets `run renew_subpix True` -- every occurrence in `examples/`
  and `context/fitting_examples/` is `False`. A change to the derivative's
  sub-pixel path therefore could not be gated bitwise against anything, and an
  ungated numerics change to the Jacobian is precisely what the Stage 0
  discipline exists to prevent. It also turns out not to interact with the
  buffer lifecycle at all (a `renew_subpix` grid is built inside the worker and
  was never shared), so it can be done later at no extra cost.

**Tests.** New `tests/test_shared_arrays.py`, 31 `unit` tests: round-trip
fidelity (dtype, shape, nesting, container types, dataclass fields), what is
deliberately *not* shared, that the results are views rather than copies and are
read-only, segment reuse and growth, generation invalidation, the subset
selection, lifecycle and cleanup, and one test that publishes in the parent and
hydrates in a real `spawn`ed child. As in 4.4, **each was verified to bite**:
13 deliberate breakages of the module (views copied, views left writeable, no
generation check, hydration not cached, outgrown segment left mapped,
`replace()` instead of field assignment, strided arrays repacked, segment not
unlinked, no fallback, two ways of getting the key selection wrong, an unknown
key skipped instead of raising, worker ignoring handles) -- **13/13 caught**.
Three assertions had to be strengthened when a run showed them missing their
mutation, and the third was the interesting one:
- one was simply too weak -- it checked that a dict entry had been replaced,
  not that the outgrown segment had actually been unmapped;
- one had become a tautology when the two selection paths were refactored onto
  shared code, so it agreed with any mistake they made together;
- the generation check looked untestable because the cached arrays are *views
  into the segment*: republishing the same shapes writes new bytes at the same
  offsets, so a stale skeleton still reads correct data. It only bites when the
  layout changes, and only if the segment does not also grow -- growing gets a
  new name, and re-attaching drops the caches for a different reason. The test
  now republishes a smaller payload into the same segment.

**Measured.** DH_orders (351 spectra, 413 free parameters, `ncpus 12`, one
iteration), three paired runs of `run shmem` False vs True on the same binary.
"PSS" is proportional set size summed over the process tree -- the tree's true
physical footprint, which charges a shared page once rather than once per
sharer.

| | `shmem False` | `shmem True` |
|---|---|---|
| peak PSS (whole tree) | 13.08 / 13.20 / 13.39 GB | **7.68 / 7.67 / 7.67 GB** |
| peak RSS (sum over tree) | 25.93-26.11 GB | 26.16 GB |
| `/dev/shm` in use | 0 | 1.30 GB |
| wall | 381.8 / 378.9 / 373.2 s | 368.7 / 365.9 / 371.0 s |
| left behind after the fit | -- | nothing |

**The footprint is the result: 13.2 GB -> 7.7 GB, a 42% cut**, and the ON
figures repeat to 0.01 GB. RSS is *unchanged*, which is the expected companion
result rather than a contradiction: RSS charges a shared page to every process
that touches it, so 12 workers reading one 530 MB region look the same as 12
workers each holding their own. PSS is the metric that distinguishes them.

Wall time improved ~2.5% (means 378.0 s -> 368.5 s, with the three ON runs all
below the three OFF runs). Real but secondary, and consistent with the stage
doc's own prediction: serialisation was only ~0.3-0.4% of the Jacobian's CPU,
so most of that 9.5 s is the allocation and page-fault churn of building and
freeing 6.4 GB of buffers per Jacobian, not the pickling itself.

On the **compact fit** (metal_line_abs, 8 free parameters) there is no
difference to measure, which is the right answer: wall 3.4 s either way, PSS
0.23 -> 0.24 GB. Its whole payload is a fraction of a megabyte, of which only
the sub-pixel grids clear the 4 kB threshold. The point of checking was that
the machinery costs nothing when it has nothing to do.

**Gate.**
- metal_line_abs (full fit + covariance) and DH_orders (one iteration):
  **BITWISE-IDENTICAL** to the pre-4.3 baseline, all 1065 files.
- `run shmem False` produces byte-identical output to `run shmem True`, so the
  fallback path is not a second set of numbers.
- Forced-GPU smoke test (`backend gpu`, `ngpus 2`, `gputhresh 0`) unchanged:
  "9 kernel launches over 18 profiles, 9 component groups on the GPU and 9 on
  the CPU". This matters because the GPU Pool uses `spawn` -- a different start
  method, and the one macOS uses for the CPU Pool too. Attaching with
  `track=False` is what keeps a spawned worker from unlinking the parent's
  segment on exit.
- `run cache False` (no profile cache to publish) also byte-identical.
- `pytest -m unit --run-gpu`: **425 passed, 31 skipped** (394 before).
- Stage 0 fast batch `pytest -m fast --run-gpu`: **63 passed, 476
  deselected in 4:46**.
- Lint (ruff / isort / black) clean.

## Task 4.6 -- Unit tests for this stage's stable surface [COMPLETE]

Two halves: `unit` tests for the Stage 4 surface now that it has settled, and
the Q4.9 GPU regression tests that re-run the example fits on the GPU backend
against the CPU references.

**Half 1: what was actually missing.** Most of the surface was already covered
by the tests written alongside 4.1/4.3/4.3a/4.5, so this began with a coverage
run rather than a guess: `alis/gpu.py` 86%, `gpu_dispatch.py` 93%,
`shared_arrays.py` 95%. (`voigt_gpu.py` reads 20%, but that is an artefact --
almost all of it is inside `@cuda.jit` device functions, which are compiled and
so invisible to a Python line counter, even on a GPU run.) The real gaps:

- **The GPU Voigt had no coverage at all without a device.** Every test in
  `test_voigt_gpu.py` was `gpu`-marked, so on CI nothing in it ran. Its
  module-level mark is now `unit`, with `gpu` on the tests that genuinely need
  a device, and nine host-side tests were added for `_encode_keywords` -- the
  one part of the file that is ordinary Python. They run with `numba` installed
  and no device, by substituting a stand-in for `cuda` whose `to_device` is the
  identity. This is the "mocked" half of Q4.2.
- **The keyword defaults are written three times** -- in `Voigt._keywd`, and
  *twice* inside `_encode_keywords` (a literal for "no mkey at all", a per-key
  fallback for "mkey given, this key absent"). A divergence gives a model that
  omits the keyword one value on the CPU and another on the GPU. Now asserted
  against `Voigt._keywd` for both branches.
- **Two unbounded-growth guards were untested**: the device wave-buffer cache
  (`_WAVE_CACHE_MAX`) and the keyword cache (`_KEY_CACHE_MAX`). Each holds a
  device allocation per distinct key, and a fit with a free shift parameter
  makes a new wave key on every derivative evaluation -- so without the bound
  the failure is an opaque out-of-memory error hours into a long fit.
- **The 4.4 warm-up warning was untested.** The 4.4 log claims `warm_up()`
  "warns by name if one claims GPU support without providing the hook"; nothing
  checked it, and that warning is the only thing that makes a forgotten
  `gpu_warmup_args()` diagnosable rather than just slow.
- `gpu.current_device()` and `shared_arrays._release_all()` (the atexit
  backstop) had no test.

**Half 2: the GPU regression batch.** New `tests/test_gpu_regression.py`: every
shipped example fit that contains a `voigt` -- 19 of the 25 -- re-run with
`run backend gpu` at `run ngpus 1` and `run ngpus 4`, compared against the same
golden references the CPU produced, through the same `compare_mod_out` /
`compare_fit_dat` the CPU suite uses. **40 tests, all passing, 6:31.**

The six examples without a `voigt` cannot launch a kernel, so re-running all of
them would have doubled the batch time to re-verify the CPU path. One of them
(the cheapest) is covered instead by a test asserting the *opposite*: a GPU-
backend fit of a model with no GPU-capable function produces the right answer
with zero launches -- which exercises the spawned pool, the per-worker CUDA
context and the 4.5 shared-memory payload travelling through `spawn` rather
than `fork`.

Two things stop this batch quietly testing nothing:

- **`run gputhresh 0` is forced.** At the shipped default of 10000
  pixel-components *no* example is large enough to dispatch, so a GPU run of
  them is a CPU run wearing a different hat. There is a test asserting that too,
  so if the default ever changes the reason for forcing it gets re-read.
- **Every test asserts on the launch count** ALIS prints at the end of the fit.
  This caught its own first bug immediately: the count was being read from
  `proc.stdout`, but ALIS's logger writes to stderr, so the first run failed
  with "the dispatcher was never enabled -- this ran on the CPU". Without that
  assertion the whole batch would have passed on CPU-computed numbers.

`alisrun.force_cpu_backend` was generalised to `force_settings(mod, **run)`,
keeping the strip-then-insert shape that the Stage 4.3 harness trap made
necessary (settings apply in file order, so a prepended override loses to a
later line).

**What the two layers actually catch (measured, not assumed).** The stage doc
notes that 1e-12 << the Stage 0 tolerances, which is the argument that the GPU
cannot cause *spurious* failures. It is worth being explicit that the converse
also holds: the regression layer is not a 1e-12 gate. Perturbing the kernel's
output by a relative factor and asking which layer notices:

| kernel error | `test_voigt_gpu.py` (1e-12) | `test_gpu_regression.py` |
|---|---|---|
| 1e-6 | fails | **passes** |
| 1e-4 | fails | passes |
| 1e-3 | fails | fails |
| 1e-2 | fails | fails |

So the regression layer's sensitivity is ~1e-3 relative -- unsurprising, since
its tolerances are 1% on chi-squared and model columns within 1% of the error
bar. It is there to catch a broken dispatch, a wrong reduction or a mis-bound
device; the 1e-12 profile accuracy is the unit layer's job. Six orders of
magnitude apart, and neither replaces the other. Recorded in `tests/README.md`
so the next reader does not assume the fits are checking the kernel's accuracy.

**Each new test was verified to bite**, as in 4.4 and 4.5: 11 deliberate
breakages (kernel off by 1e-6 and by 1e-3, dispatcher sending nothing to the
device, keyword defaults diverging, an un-defaulted absent keyword, a cache
confusing configurations, both unbounded-growth guards removed, an unchanged
grid re-uploaded, the missing warm-up hook passing silently, `current_device`
not reporting absence) -- **11/11 caught**. Three needed the test strengthened
first:
- the 1e-6 kernel error was originally aimed at the regression layer, which is
  the measurement in the table above rather than a test weakness;
- the keyword-defaults test covered only the `mkey is None` literal, leaving
  the per-key fallback free to drift -- it is now parametrised over both;
- the `current_device` fake had no `current_context`, so removing the
  `device_count()` guard was unobservable; the fake now names a device, and the
  guard is the only thing preventing a false claim that device 0 exists.

**One cross-test bug found by running the whole batch rather than the file.**
The fake "forgot its warm-up hook" function was written as a `base.Base`
subclass, which put it permanently into the subclass tree that
`gpu_capable_functions()` walks -- so `test_function_interface.py`'s "the walk
matches the registry" check failed whenever both files ran in one session. It
is now a plain class; `warm_up()` needs only three members, and nothing that
fakes a model function should be discoverable as one.

**Gate.**
- `pytest -m unit --run-gpu`: **440 passed, 31 skipped** (425 before).
- `pytest -m unit` (no device): the 25 `gpu and unit` tests skip, and the nine
  new host-side Voigt tests still run -- which is the coverage the GPU Voigt
  had none of before.
- `pytest --run-gpu -m gpu`: **65 tests, all passing** -- 25 fast unit-level
  (8.8 s) and the 40 regression fits (6:31). The whole batch was run green end
  to end at 6:59; the regression 40 have now passed on three independent runs,
  which matters because they are real minimisations and a fit landing near a
  tolerance boundary would show up as flakiness.
- No `/dev/shm` segment left behind by any of it (the Stage 4.5 payload travels
  through `spawn` in every GPU run here).
- Stage 0 fast batch `pytest -m fast --run-gpu`: **63 passed in 4:44**.
- Lint (ruff / isort / black) clean.

## Pre-4.7 analysis -- two proposed performance changes (RJC, 2026-08-01)

RJC asked, before closing the stage, whether two changes would be faster on
DH_orders: **(A)** dropping the parameter-influence table and applying one flat
list of Voigt profiles to every snip at once, and **(B)** fixing the
sub-pixellation at the start of the fit so the wavelength arrays reach the GPU
only once. Both were measured rather than argued.

**Method.** One Jacobian of DH_orders (351 spectra, 413 free parameters), timed
in-process around `fdjac2` and stopped after it, so no run-to-run fit-path
variation enters. `load_par_influence` already takes a `setall` flag, so "remove
all influence" is measurable without writing a fused kernel; a `freevoigt`
variant replaces `Voigt.call_CPU` with a constant, which prices the *limit* of
an infinitely fast GPU -- no kernel, however good, can be cheaper than free.

| configuration | workers | Jacobian |
|---|---|---|
| **current** | CPU-12 | **100.25 s** |
| current, Voigt arithmetic free | CPU-12 | 74.68 s |
| nothing influences anything | CPU-12 | 20.46 s |
| influence removed (`setall`) | CPU-12 | 159.15 s |
| influence removed **and** Voigt free | CPU-12 | 138.72 s |
| `renew_subpix True` | CPU-12 | 164.15 s |
| **current** | GPU-4 | **169.6 / 171.1 s** |
| influence removed **and** Voigt free | GPU-4 | 379.47 s |

**(A) Dropping influence is slower, and cannot be rescued by fusing the
launches.** 100.25 -> 159.15 s (1.59x) on its own. The bound is the decisive
figure: with the Voigt costing *nothing*, the influence-free design still takes
138.72 s at 12 workers (1.38x worse than today's real 100.25 s) and 379.47 s at
4 workers (2.24x worse than today's real 169.6 s). The reason is the split of
the Jacobian:

- **25.6 s (25.5%)** -- the Voigt profile arithmetic, the only part a kernel
  touches (100.25 - 74.68);
- **20.5 s (20.4%)** -- machinery that runs whatever the influence table says
  (measured directly by emptying the table);
- **54.2 s (54.1%)** -- influence-*dependent* host marshalling.

Each parameter influences a mean of 39.9 of the 351 snips (median 14, p90 120,
max 311) -- 11.4% -- so removing the table multiplies the host marshalling,
which is three quarters of the cost, by **8.8x**. Accelerating a quarter cannot
pay for that.

**Correction to the Task 4.3/4.5 entries.** Those say "DH_orders groups reach
only ~1e4-3e4 pixel-components today". That is wrong by an order of magnitude.
Measured by spying on the real `Dispatcher.should_dispatch` calls of one base
evaluation: 351 voigt groups, 7985 rows, **118,694,903 pixel-components**, mean
**338,162** per group, median 125,664, max 4,047,519 -- and **301 of the 351
already clear the shipped `gputhresh` of 10000**. The groups are not small, so
the premise behind carrying "batch across spectra" forward was mistaken: fusing
351 launches into 1 saves launch overhead (351 x ~5-10 us, a few ms) against a
base evaluation, not a factor. This makes the 4.5 carry-in (a) low-value at
DH_orders scale, and it should not be prioritised on the strength of the old
figure.

**(B) The sub-pixellation is already fixed, and the grids already upload once.**
`RunConfig.renew_subpix` defaults to **False** and DH_orders does not set it, so
`load_subpixels` runs once before the fit and `model_func` reuses
`state._wavespx` unchanged. Forcing `True` costs 164.15 s against 100.25 s, so
the default is already worth **1.64x** -- there is no further gain to take,
only that loss to avoid. The GPU half of the premise is also already satisfied
by the 4.3 device wave cache: the first evaluation uploads all 351 grids
(28.8 MB total) and every later one reuses them -- measured **351 uploads / 0
reuses** on the first evaluation and **0 uploads / 187 reuses** across
subsequent derivative columns.

**What the measurements do point at.** RJC's underlying instinct -- that the
per-snip Python work dominates -- is right; it is the proposed *mechanism* that
backfires. Profiling four derivative columns (cProfile inflates Python-level
calls relative to numpy, so this is for composition, not for shares):

| | calls in 4 columns | note |
|---|---|---|
| `config._DictLike.__getitem__` | 1,724,030 | the dataclass-as-dict shim |
| `builtins.getattr` | 1,824,189 | |
| `voigt.set_vars` | 5,326 | parameter marshalling |
| `numpy.fft._raw_fft` | 765 | the convolution, on the host |
| `voigt.model` | 72 | the actual profile arithmetic |

Two specific wastes, both re-deriving quantities that are constant for the whole
fit, on every one of the 413 derivative columns:

1. The setup loop (`model_eval.py` ~154-240) evaluates the shift model
   (`set_vars` + `call_CPU`) for **all 351 snips** before any `ddpid` test; the
   influence check at line 209 skips *components*, not the snip.
2. The convolution loop is influence-gated (line 398), but its skip path still
   runs `np.where` + `np.isin` over each snip's wavelengths to advance
   `stf`/`enf` -- counts that depend only on `x`, `_posnfit` and `_wavefit`, all
   fixed for the fit. That is ~145,000 `np.isin` calls per Jacobian.

Together these are the 20.5 s influence-independent floor, and the marshalling
above is most of the 54.2 s. Proposed as tasks 4.7 and 4.8 in the stage doc,
for RJC to accept or decline; no task was written for (A) or (B), since the
measurements do not support either.

## Task 4.7 -- Precompute the fit-constant model structure [COMPLETE]

The setup loop of `model_func` re-derived, on every model evaluation, which
components apply to each snip and how they group -- 351 snips x ~245 components
of `emab`/`specid` tests, plus a model-type list searched and grown with
`np.where`/`np.append`. None of it depends on the parameters.

**Sized first.** Timing the three phases of `model_func` over 12 DH_orders
derivative columns:

| phase | before |
|---|---|
| setup loop | 2.534 s (58.1%) |
| — of which: shift `set_vars` + `call_CPU` | 0.052 s (1.2%) |
| — of which: component `set_vars` | 0.669 s (15.3%) |
| — of which: **structure derivation** | **1.814 s (41.6%)** |
| evaluation loop | 0.834 s (19.1%) |
| convolution + assembly | 0.995 s (22.8%) |

**The change.** New `SnipPlan` / `build_model_plan` / `model_plan` in
`model_eval.py`. Once per fit, each (sp, sn) gets its zero-level components, its
ordered `(i, mtyp, ea, mid)` entries and its per-group model-type slots; the
loop then walks that list instead of rediscovering it. Cached on a new
`FitState._mplan`, keyed on the snip layout so the plotting, simulation and
`iterate` paths cannot be served a stale plan (they pass different positions or
build a fresh state). `getattr`/`setattr` because `state` is a `ClassMain` on
some paths, not always a `FitState`.

Two smaller things came with it:
- The per-component `np.append` accumulation became one `np.concatenate` per
  slot. The old form recopied the whole block for every row added -- quadratic
  in a group's row count, which reaches 134 on DH_orders. Same pieces in the
  same order, so the same bytes, copied once.
- The influence test moved to a single `influenced` decision per snip rather
  than being retested for each of the ~245 components.

**Result (one Jacobian, DH_orders, CPU-12):**

| | before | after |
|---|---|---|
| Jacobian | 100.25 / 100.28 s | **85.95 / 86.82 s** |
| | | **1.16x** |

The mechanism worked as designed -- re-measuring the phases afterwards, the
structure derivation fell from **1.814 s to 0.202 s** (89% removed) and the
setup loop from 2.534 s to 0.930 s, taking the in-process wall for 12 columns
from 4.517 s to 2.936 s (**1.54x**).

**Be careful with the 41.6% figure: the Jacobian gained 14%, not 42%.** The
in-process measurement isolates `model_func` on an idle machine; the Jacobian
is 12 workers contending for 20 cores with the memory traffic of a 351-spectrum
model, and there the per-column cost is several times higher. So the phase
percentages size the *work removed* correctly but overstate the *wall* gain,
and the honest number for the change is the end-to-end 1.16x. The gap between
1.54x in-process and 1.16x end-to-end says the remaining Jacobian time is
mostly outside `model_func`; that is where any further work should look, and it
is worth re-checking before starting 4.8.

**Also measured, and it revises 4.8.** The shift `set_vars` + `call_CPU` that
4.8(a) proposes to skip for un-influenced snips is only **1.2%** of a derivative
column -- not the large share the "influence-independent floor" of 20.5 s
suggested. 4.8(a) is therefore not worth the ordering hazard it carries;
4.8(b), the `np.isin` bookkeeping in the convolution loop's skip path, is
untouched by this measurement and remains the part worth doing. The stage doc
entry for 4.8 has been updated to say so.

**Tests.** New `tests/test_model_plan.py`, 17 `unit` tests pinning the
resolution rules directly on small synthetic models: what the `emab`/`specid`
filters exclude, that the zero level is separated (it is read before the
influence test, so an un-influenced snip still sets it) and belongs to the first
snip only, the grouping rules (a change of `emab` opens a group, `va` does not,
the same model type twice shares one slot), the caching and its invalidation,
that the plan pickles (it travels to every worker through the 4.5 shared
segment), and that concatenating the pieces equals appending them one by one.
The equivalence with the old per-call derivation is held by the Stage 0 bitwise
gate, which is the right instrument for it. The model-validity error
("must specify emission before absorption") moved into the builder with the loop
it came from -- it now fires once, at the first evaluation, and there is a test
that it still fires.

**Gate.** metal_line_abs (full fit + covariance) and DH_orders (one iteration)
**BITWISE-IDENTICAL** to the pre-4.3 baseline, all 1065 files.
`pytest -m unit`: **457 passed, 31 skipped** (440 before). Lint clean.

## Task 4.8 -- Load balance and per-snip constants [COMPLETE]

The task's own last bullet said to re-measure before starting, because after
4.7 the in-process `model_func` wall had improved 1.54x but the Jacobian only
1.16x -- so most of the remaining time was somewhere else. That measurement
found something an order of magnitude larger than either item the task was
scoped for.

**Where the Jacobian's time was going.** Timing every one of the 406 columns
serially:

| | |
|---|---|
| total serial work | 276.7 s |
| perfectly balanced over 12 workers | 23.1 s |
| column cost | mean 0.681 s, median 0.285 s, **max 8.682 s** |
| **slowest contiguous chunk** | **84.1 s** (fastest 4.4 s) |
| measured Jacobian | 86 s |

The Jacobian wall *was* the slowest chunk. Per-chunk totals under the Stage 3.4
contiguous split: `[12.2, 9.1, 18.4, 8.1, 4.4, 4.7, 61.9, 84.1, 23.2, 14.4,
7.7, 28.6]` s -- a 19x spread. Column cost is roughly a fixed 0.23 s plus the
model evaluation of every snip the parameter influences (0 to 311 snips), and
parameters for the same object sit together in the model file, so contiguous
blocks concentrate the expensive columns.

**The fix is to deal the columns round-robin.** Compared on the measured costs:

| assignment | slowest chunk | vs balanced |
|---|---|---|
| contiguous (Stage 3.4) | 82.7 s | 3.61x |
| **round-robin** | **28.2 s** | **1.23x** |
| greedy on the true costs | 22.9 s | 1.00x |
| greedy on an influence-table cost model | 27.9 s | 1.22x |

A cost model built from `_pinfl` was tried and did **not** beat dealing: it
correlates with the true cost at only r ~ 0.7, and greedy on the raw predictor
is actively worse (3.2x) because it heaps all the zero-weight columns -- which
still cost 0.23 s each -- into one bin. Dealing needs no model, no measurement
and no tuning, so it is the whole fix.

Reordering is numerically inert: each column is computed independently and
written to its own column of `fjac`. Stage 3.4 chose contiguous blocks so a
chunk's parameters would touch overlapping sp/sn and need a smaller cache
slice; **Stage 4.5 removed that reason** by moving the cache into shared
memory, where a chunk names its entries instead of carrying them. Measured on
the fallback path too (`run shmem False`): 49.1 s dealt, against a contiguous
floor of >=84 s that is compute-bound whatever the transport, so dealing wins
there as well.

**Item (b), the originally scoped one, was also done.** The convolution loop
skips a snip the derivative does not influence, but still ran a `np.where` plus
a `np.isin` over the snip's wavelengths to work out how far to step in the
packed model vector -- a count fixed for the whole fit. New `build_fit_windows`
/ `fit_windows` computes it once (cached on `FitState._nfitpix`, keyed on the
identity of `x` as well as the snip layout, because `ClassMain.model_func` lets
a caller pass a different wavelength array). That removes ~145,000 `np.isin`
calls per Jacobian. **Measured at 4.2% of a derivative column** -- real, but a
fifth of what the original "20.5 s influence-independent floor" estimate
implied, since that floor also contained the structure work 4.7 removed.

**Item (a) was dropped**, as the revised task said it should be: the shift
`set_vars` + `call_CPU` it proposed to skip is 1.2% of a column, which does not
justify its ordering hazard (`wvrng` from the shift feeds `set_vars` for the
components).

**Result (one Jacobian, DH_orders, CPU-12):**

| | Jacobian |
|---|---|
| before 4.7 | 100.25 / 100.28 s |
| after 4.7 | 85.95 / 86.82 s |
| after 4.8 dealing | 34.61 / 35.40 s |
| **after 4.8 dealing + fit windows** | **31.81 / 32.91 s** |

**3.09x against the pre-4.7 baseline**, 2.66x from 4.8 alone. The remaining
32 s sits against a perfectly balanced floor of 23.1 s for the same work, so
what is left is ~1.2x of residual imbalance plus pool overhead -- close enough
to the floor that further gains have to come from making the columns cheaper,
not from scheduling them better.

**Tests.** `tests/test_model_plan.py` grew 6 tests for the fitted-pixel counts
(what the fit range and the fitted-pixel mask each exclude, per-snip
separation, caching, and rebuilding when the wavelength array changes);
`tests/test_minimise_helpers.py` grew 5 for the dealing (that it covers every
column exactly once -- a dropped column leaves a zero in the Jacobian without
failing any numerical gate -- that it is round-robin rather than contiguous,
that it balances a cost that runs with the column index, that it never emits an
empty chunk, and the fewer-columns-than-workers case). **16/16 mutations
caught** across 4.7 and 4.8, including reverting the dealing to contiguous
blocks and dropping the tail of the columns.

**Gate.** metal_line_abs (full fit + covariance) and DH_orders (one iteration)
**BITWISE-IDENTICAL** to the pre-4.3 baseline, all 1065 files.
`pytest -m unit`: **468 passed, 31 skipped** (457 before). Lint clean.
