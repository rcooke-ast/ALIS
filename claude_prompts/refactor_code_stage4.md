# Prompt file for ALIS software refactoring -- STAGE 4

> **GPU support and modularity.** Formalise a clean CPU/GPU model-function
> interface on the refactored base class, implement a GPU Voigt profile, and add
> multiprocessed GPU dispatch with a CPU fallback. GPU is an optional install
> extra — CPU-only users must need no CUDA toolchain (plan Q4). Depends on
> Stage 2 (clean core), Stage 3.4 (persistent Pool + chunked Jacobian) and
> Stage 3.5 (the `prepare_iteration` seam); the GPU path must reproduce the CPU
> results within an agreed numerical tolerance, checked against the Stage 0
> references.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage4_log.md`.

**4.0 — Return-not-mutate evaluation contract (carried from Stage 3.5, do first).
[COMPLETE -- bitwise-verified; Stage 0 gate green (142 passed, 1 deselected);
see `logs/refactor_code_stage4_log.md`]**
- Deferred here from Stage 3.5 (its Task 3.5.3). Complete the derivative
  evaluation contract before the GPU kernel work: make the derivative path
  (`model_eval.model_func` with `ddpid` set, reached via
  `minimise._worker_funcderiv` -> `_minimiser_eval` -> `myfunct`) a pure function
  of `(the per-iteration invariants in self._emab, perturbed params)` that
  *returns* its residual column instead of writing model scratch into the shared
  `state` (`_modfinal`/`_contfinal`/`_zerofinal`), so the per-call
  `copy.copy(state)` in `_minimiser_eval` can be dropped. This is the exact shape
  a GPU kernel needs (a kernel cannot mutate shared Python state); co-design it
  with 4.3 (dispatch) and 4.5 (shared memory). Stage 3.5.2's
  `prepare_iteration()` / `self._emab` already isolate the per-iteration
  invariant (the component cache), and `_pinfl` is a per-*fit* constant
  (Stage 3.5 Finding F1), so only the per-call model-scratch writes remain to
  eliminate -- a localized change. Must stay **bitwise-identical** under the
  Stage 0 gate; do it in small, individually gated steps, and add its `unit`
  tests here (deferred from Stage 3.5's 3.5.6).
- **Interaction to handle (from Finding F1):** dropping `copy.copy(state)` means
  the serial *base* call would now write to the shared `state` directly. Today the
  per-iteration `_pinfl` recompute (`model_eval.py:120`) is harmless only because
  it lands on the throwaway copy; without the copy it would mutate the shared
  influence table every iteration -- changing the numerics and violating the
  "influence fixed for the whole fit" invariant (RJC, Stage 3.5). So 4.0 must
  first **freeze `_pinfl`**: compute it once before the fit and stop recomputing
  it during the fit (guarding/removing line 120 on the fit path), which is exactly
  the set-once-never-update-during-fit behaviour already agreed, and is what makes
  the copy removal bitwise-safe. (The sim `beginfrom` path still needs its
  pre-fit influence set explicitly -- see the Stage 3.5 log.)

**4.1 — Formalise the CPU/GPU model interface.
[COMPLETE -- Stage 0 fast gate green (63 passed); unit batch 60 -> 71;
see `logs/refactor_code_stage4_log.md`]**
- In the base class `Base` (`alis/functions/base.py`) -- which already carries a
  `call_CPU` template and a `call_GPU` stub (currently just delegating to
  `call_CPU`) -- formalise a clear `call_CPU` / `call_GPU` contract with a
  numerical-equivalence requirement and a graceful "no GPU → CPU" fallback.
  Document how a function opts into GPU support. (Model functions live in
  `alis/functions/<name>.py`, e.g. `voigt.py`.
- Design principle (resolved 2026-07-29): keep the kernels **in** the per-function
  `call_GPU` (modular; good for `new-alfunc`/`port-to-gpu`), designed to take
  already-on-device arrays and a *batch* of profiles (it launches, it does not
  transfer). The kernel's module *location* is runtime-irrelevant -- a compiled
  `@cuda.jit` kernel runs identically wherever it is defined, and Python dispatch
  is ~ns against ms launches/transfers -- so hard-coding kernels in
  `alfit`/`model_eval` would **not** be faster. Speed comes from the dispatch
  layer (transfer frequency, batching, on-device residency; see 4.3), which is
  orthogonal to where the kernel lives.
- Requirement: `numba`/CUDA must be imported **lazily** -- only when the GPU
  backend is actually selected -- so CPU-only installs (without the `gpu` extra)
  never import them and need no CUDA toolchain (plan intro; Q4.11). A missing
  `numba`/GPU at GPU-selection time falls back to CPU with a clear message.

**4.2 — GPU Voigt profile.** [COMPLETE]
- Implement `call_GPU` for the Voigt profile (`alis/functions/voigt.py`, which
  has a `GPU_kernal` stub, commented `pycuda` imports, and the CPU `model`)
  using the `context/voigt_gpu/` example (Faddeeva function), validated close to
  the CPU version.
- numba feature note (Q4.11): the port needs **neither dynamic parallelism nor
  texture memory** (both unsupported by numba). The dispatch is host-driven (no
  device-side kernel launches), and the Faddeeva is an analytic evaluation whose
  small read-only coefficient tables (`erfcx_coeffs.dat` / the `hA..hD` arrays)
  belong in numba **constant memory** (`cuda.const.array_like`), not texture --
  there is no interpolation / 2D-locality use case. The `context/voigt_gpu/`
  example is itself pure numba and already hits ~1e-15, which is the existence
  proof that these features are not required.

**4.3 — Multiprocessed CPU/GPU dispatch (either-or, not hybrid).**
- The parallel backend is an **either-or**, chosen per fit: CPU (the Stage 3.4
  persistent Pool over `ncpus`) **or** GPU, never both computing derivative
  columns at once (resolved 2026-07-29; see Q4.8). Rationale: a GPU model-eval is
  ~50x a CPU one at DH_orders scale, so GPU-only already beats CPU-only ~15x even
  across only ~4 GPUs, while a hybrid would add only single-digit % for a large
  complexity / mixed-validation-gate / determinism cost.
- **GPU backend:** reuse the persistent-Pool + chunked-Jacobian machinery, but
  size the pool to `ngpus` with each worker bound to a distinct GPU
  (`cuda.select_device(rank)`), and distribute the Jacobian's derivative columns
  across the GPU workers. CUDA contexts do not survive `fork` -> `spawn` start
  method (Q4.8). Clean fallback to CPU when no GPU is present.
- **Dispatch shape (resolved, Q4.7):** the per-function interface stays
  per-component (`call_GPU` in `alis/functions/<name>.py`, operating on device
  arrays), but the dispatcher **batches** same-type components/spectra into one
  launch, with an array-size threshold below which it uses the CPU path (small
  snips lose to launch+transfer). Upload the per-iteration read-only data --
  above all the sub-pixel **wave grid** -- to the device **once per iteration** in
  `prepare_iteration()` (it can change per iteration via `renew_subpix`, so
  once-per-iteration, not once-per-fit), keep it device-resident, derive the
  shifted wave on-device, and keep emission/absorption intermediates on-device,
  downloading only the final convolved model.

**4.3a — Backend selection (`run backend = auto | cpu | gpu`).**
- New `RunConfig.backend` setting (default `auto`). `cpu`/`gpu` force a backend
  -- use these for **reproducibility**, since `auto`'s choice can vary run-to-run
  when timings are close, which would flip CPU-bitwise vs GPU-1e-12 numerics.
- `auto`: if no GPU is present -> CPU (no probe). Otherwise **warm both backends
  first** (force the numba `@cuda.jit` compile + CUDA context + a dummy launch;
  spin up the CPU Pool), then time one **Jacobian** evaluation at `p0` on each and
  commit the **entire** fit to the faster backend (so the whole fit stays under a
  single numerical gate), continuing from that backend's result. Warming first is
  essential -- a cold GPU probe would mostly time JIT/context and mis-pick CPU. A
  single sample suffices (the CPU<->GPU gap is large); probe overhead ~= one
  discarded backend Jacobian (negligible for long fits).
- Interplay: `backend cpu` -> CPU Pool (`ncpus`), GPUs ignored. `backend gpu` ->
  GPU (all detected GPUs if `ngpus` unset, else `ngpus`). `backend auto` -> probe
  only when `ngpus` (>0) is explicitly requested (Q4.10); otherwise CPU.
- **GPU-available notice (Q4.10):** when GPUs are detected but not being used
  (`ngpus` unset / `backend` resolved to CPU), log an INFO line stating how many
  GPUs are present and that setting `run ngpus N` (with `backend auto`/`gpu`)
  enables them -- so users on a GPU box discover the option without it being
  forced on them.
- The Stage 0 harness forces `backend cpu` so CI/regression stays deterministic
  and CPU-only (Q4.2); GPU coverage is the on-demand `gpu`-marked tests (Q4.9).

**4.4 — New-function ergonomics.**
- Make it straightforward to add a new model function (`alis/functions/<name>.py`)
  with both CPU and GPU paths plus its own unit tests (reuse/extend the
  `new-alfunc` and `port-to-gpu` skills).

**4.5 — Shared-memory read-only arrays (carried forward from Task 3.4 Phase 3).**
- Origin: the deferred "Phase 3" of Task 3.4. Phases 1-2 (persistent Pool +
  chunked derivatives + subset-pickling) already made the profile cache a
  universal win (`RunConfig.cache` now defaults True; DH_orders cache-on 0.54x
  cache-off) and cut the per-iteration cache pickle to a small per-chunk slice.
  Phase 3 -- placing the *read-only* worker inputs (the `FitState` data arrays,
  and on GPU the model/cache buffers) in `multiprocessing.shared_memory` for
  zero-copy access -- was intentionally deferred to here because its value is
  memory/GPU-side, not CPU time.
- Measurement that justifies the deferral (DH_orders, cache on, Phase 2,
  `fdjac2` profiled): per Jacobian eval the `pool.map` wall is ~49 s but the
  payload *serialisation CPU* is only ~0.25-0.40 s (**~0.3-0.4%** of the wall) --
  the Jacobian is ~95%+ actual `model_func` compute (GIL-bound Python loops),
  which shared memory does **not** speed up. So Phase 3 buys little CPU time on
  its own. However each eval still ships ~1.1 GB (the constant `FitState` arrays
  are re-sent per worker/task, not deduplicated across tasks), so the real
  Phase 3 wins are (a) **per-worker memory footprint** (one mapped buffer vs
  `ncpus` copies) and (b) a prerequisite for the GPU path, where the arrays must
  live in a shared/device buffer anyway and where -- once compute moves off the
  CPU -- that 1.1 GB transfer becomes a relatively larger share of the total.
- Scope: back the read-only arrays passed to `_worker_chunk`
  (`alis/minimise.py`) with shared memory (attach/detach per lazily-started
  worker; robust buffer lifecycle + `resource_tracker` cleanup on macOS
  `spawn`), keeping the Jacobian **bitwise-identical** (Stage 0 gate) and serving
  the GPU dispatch of 4.3. Benchmark memory and wall on DH_orders (many-worker)
  and a compact fit.
- Optional carry-in from Stage 3.5.5 (deferred): the bitwise-safe `renew_subpix`
  *conditional* recompute (reuse the base sub-pixel grid for derivative
  perturbations of parameters that do not feed the subpix/resolution model). It
  fits naturally with the per-iteration buffer management here; do it only if it
  stays simple and does not complicate the buffer lifecycle.

**4.6 — Unit tests for this stage's stable surface (do last).**
- Following the cross-cutting unit-test policy
  (`claude_prompts/refactor_code_unit_tests.md`), add `unit`-marked tests for the
  *stable* code introduced in Stage 4 once its interfaces settle: the CPU/GPU
  dispatch/selection logic and fallback (4.1), pure helpers of the GPU Voigt
  (4.2), and the shared-memory buffer lifecycle (4.5) — with a CPU-vs-GPU
  numerical-equivalence test where GPU hardware is available (else skipped/mocked
  per Q4.2). Keep them fast and isolated (no full fits); the existing `unit` CI
  job picks them up automatically.
- **GPU regression tests (Q4.9):** add a `gpu` pytest marker and re-run the
  existing example fits with `run backend gpu` at `run ngpus 1` and `run ngpus 4`,
  comparing against the **same CPU references** (1e-12 << Stage 0 tolerances).
  These are hardware-dependent, so gate them behind a `--run-gpu` opt-in (mirror
  the existing `--run-slow` gate in `tests/conftest.py`) so they are deselected by
  default and CI stays CPU-only. **On-demand local command (RJC asked):**
  `pytest --run-gpu -m gpu` (all GPU tests); document it in `tests/README.md`.

## Skills to use for this stage

- `port-to-gpu` — port a `call_CPU` to `call_GPU`, verifying numerical equivalence.
- `gpu-benchmark` — CPU vs GPU throughput for a function / full fit.
- `new-alfunc` — scaffold new functions with CPU+GPU paths.
- `run-tests` — Stage 0 gate (CPU path).

## Context

- `context/voigt_gpu/` (Faddeeva implementation: `faddeeva.py/.cc/.hh`,
  `numba_test.py`, `erfcx_coeffs.dat`, `simple_test.py`), `alis/functions/voigt.py`
  (existing `GPU_kernal` stub, commented `pycuda` imports, and CPU `voigtking`).
  Commented-out PyCUDA/GPU scaffolding also survives in `alis/main.py`,
  `alis/model_eval.py`, and several `alis/functions/*.py`.
- The `DH_orders` model (351 spectra) is the prime GPU beneficiary.
- Stage 3.5 seam: `alfit.prepare_iteration()` prepares the per-iteration
  invariant (`self._emab`); the derivative path is `minimise._worker_funcderiv`
  / `_worker_chunk`; `_pinfl` is a per-fit constant (Finding F1).
- GPU stack: **`numba.cuda`** (Q4.1), float64 only (Q4.3/Q4.6), as an optional
  install extra. The `gpu` extra in `pyproject.toml` becomes `["numba"]` (was the
  placeholder `cupy`; Q4.11). GPU users must **also install a CUDA toolkit**
  matching their driver (numba.cuda needs it; it is not pip-installed by the
  extra) -- document this in the install/GPU docs. numba's lack of dynamic
  parallelism and texture memory does not affect us (see Task 4.2).

## Queries

**Q4.1 — GPU stack.** Decide the target now: CuPy, `numba.cuda`, or PyCUDA? The
old code used PyCUDA; the `voigt_gpu` example uses numba. Recommendation: pick one
that installs cleanly as an optional extra and matches the Faddeeva example.

**Response:** We will use `numba.cuda`, since this is what the `voigt_gpu` code uses.

**Q4.2 — GPU testing in CI.** Is GPU hardware available for CI, or should GPU
tests run only locally / on demand (with CI covering the CPU path + a mocked GPU
interface)?

**Response:** GPU hardware is only available locally, so RJC will run those tests
on demand. For now, let's make sure CPU only tests are run by default. We will need
to write our own GPU tests and examples (that cover usage of a single GPU and multiple
simultaneous GPUs).

**Q4.3 — CPU↔GPU equivalence tolerance.** What relative tolerance is acceptable
between the GPU and CPU Voigt (single vs double precision on GPU)?

**Response:** The current `simple_test.py` has an accuracy that is better than 1e-15.
This level of accuracy is not currently required, but we should aim for an absolute
tolerance of 1e-12 for the difference between the GPU and CPU implementations of the
Voigt profile.

**Q4.4 — Shared memory scope (Task 4.5).** Should shared memory back only the
`FitState` data arrays (CPU-side, memory-footprint win), or be designed together
with the GPU device buffers of 4.2/4.3 as one buffer-lifecycle layer? The Task
3.4 profiling shows little standalone CPU-time upside, so co-designing it with
the GPU path (where the arrays must be shared/device-resident anyway) likely
gives the best return. Recommendation: fold it into the 4.2/4.3 GPU work rather
than build a CPU-only shared-memory layer first.

**Response:** We will fold the shared memory implementation into the GPU work, as
this will give us the best return on investment. The shared memory will back both
the `FitState` data arrays and the GPU device buffers, allowing for a more efficient
and streamlined implementation.

**Q4.5 — Development environment on this machine.** ALIS is not currently
installed in any environment here, and `pytest` is absent from all of them, so
the Stage 0 gate cannot run yet. The default `python` is
`~/anaconda3/envs/py311` (3.11.9), which is below the `requires-python = ">=3.13"`
floor. `~/anaconda3/envs/py313` (3.13.14) already has numpy/scipy/astropy plus
**numba 0.66 with working CUDA** (a trivial `@cuda.jit` kernel runs; `cuda.gpus`
reports **4x NVIDIA RTX 2080 Ti**). Recommendation: make `py313` the target
environment, `pip install -e ".[dev]"` there (plus `numba`), and certify with
`pytest -m unit` and the fast batch *before* any Stage 4 code is written.
Confirm?

**Response:** `py313` is the correct version to use. I have installed `pytest`
and also installed ALIS in editable mode with the dev extras.

**Q4.6 — GPU precision.** The Q4.3 tolerance (1e-12 absolute) requires float64
throughout the GPU Voigt. Note that on the RTX 2080 Ti (Turing, compute 7.5)
FP64 runs at 1/32 of the FP32 rate, so per-kernel speedups on these cards will be
modest; the real win for `DH_orders` comes from batching 351 spectra, not raw
double-precision throughput. Recommendation: implement float64 only and do *not*
add a float32 fast path (it could not meet the 1e-12 gate and would fork the
numerical behaviour). Agreed?

**Measurements (Claude, 2026-07-28; RTX 2080 Ti, py313 + numba 0.66).** RJC asked
what float32 would actually buy in speed and cost in accuracy, so both were
measured rather than estimated.

*Speed 1 — the real float64 Faddeeva kernel from `context/voigt_gpu/simple_test.py`
vs the current `scipy.wofz` CPU path (per call, mean of 10-20 reps):*

| Sub-pixels | CPU `wofz` | GPU fp64 | speedup |
|-----------:|-----------:|---------:|--------:|
| 1,000 | 0.105 ms | 0.057 ms | 1.8x |
| 10,000 | 0.877 ms | 0.056 ms | 15.6x |
| 100,000 | 8.20 ms | 0.231 ms | 35.4x |
| 1,000,000 | 101 ms | 1.85 ms | 54.7x |
| 10,000,000 | 1132 ms | 15.6 ms | 72.7x |

*Speed 2 — synthetic kernel with the Faddeeva inner-loop op mix (exp + FMA +
divide in a branchy loop), compiled float32 vs float64:* fp64 9.116 ms ->
fp32 0.274 ms = **33.3x**, matching Turing's theoretical 32:1 FP32:FP64 ratio.
The real kernel is compute-bound too (at 10M pixels it moves 160 MB in 15.6 ms
= ~10 GB/s, ~1.7% of the card's 616 GB/s peak), so a genuine fp32 Voigt would
plausibly land in the 10-30x range over fp64.

**Correction to the query text above:** "per-kernel speedups on these cards will
be modest" is wrong. Even in float64 the GPU beats the CPU by 15-73x for arrays
greater than 10^4 sub-pixels. The FP64 penalty is real (fp32 would be another ~33x), but
float64 already captures a large win.

*Accuracy — float32 emulated at three separate points in ALIS's exact expression
(`v = wv*((wv/ww)-1)/bl`), over logN 13-20.5, b 1-20 km/s, z 0 and 2.5, on a
0.5 km/s sub-pixel grid. Worst-case **absolute** flux error:*

| What is float32 | Worst error | Verdict |
|---|---:|---|
| Wavelength array + argument arithmetic | 3.0e-2 | fatal |
| Argument arithmetic only (wave stays f64) | 3.0e-2 | fatal |
| Only w(z), argument well-conditioned in f64 | 2.1e-8 | realistic floor |

The limiting factor is **not** the Faddeeva function but the argument:
`wv/ww ~ 1` and then 1 is subtracted, which is catastrophic cancellation and
burns essentially all ~7 significant float32 digits. Storing the wavelength grid
in float32 is independently fatal (1.2e-4 A quantisation at Lya ~ 0.03 km/s of
grid jitter, a percent of a narrow line's width). The cancellation *is* fixable
by computing the velocity offset in double and passing a well-conditioned Dv/b to
the kernel -- which lands on the third row, **~2e-8 absolute**, the honest best
case for float32 and still 4 orders short of the Q4.3 target of 1e-12.

**Recommendation (unchanged, but for a sharper reason): float64 only.** 2e-8 is
physically negligible -- five orders below the noise even at S/N ~ 1000 -- and
would pass the Stage 0 gates comfortably. The real risk is the **Jacobian**:
`minimise.fdjac2` builds two-sided finite differences, so a 2e-8 model noise
floor becomes a percent-level error in derivative columns wherever the parameter
step changes the model by ~1e-6, degrading the covariance and convergence. That
is exactly the failure mode Stage 3 guarded against bitwise (the caching bug that
passed `.mod.out` but drifted `.covar`). Given float64 already delivers 15-73x,
float32 is a second-order optimisation bought with a numerically fragile second
code path. If wanted later, the sane framing is an opt-in `run gpuprec single`
for exploratory/survey work, added *after* the fp64 path lands, with the argument
restructured to avoid cancellation and gated on its own covariance validation --
out of Stage 4 scope.

**Bearing on Q4.7:** at 1,000 sub-pixels the GPU is only 1.8x faster (launch and
transfer dominate). `DH_orders` snips are ~310 pixels x nsubpix 5 ~ 1,550
sub-pixels, so a naive per-component `call_GPU` sits squarely in that
low-payoff regime; batched across all 351 spectra it reaches ~5x10^5 sub-pixels
and the ~50x regime. Batching is therefore a more consequential design choice for
Stage 4 than the precision question.

**Response:** Let's ensure float64 is used throughout the GPU Voigt implementation,
as this will meet the requirements.

**Q4.7 — GPU dispatch granularity (Tasks 4.1/4.3).** Two options: (a) a
per-component `call_GPU` that is a clean drop-in for `call_CPU` in
`model_eval.model_func` — simple, but pays a kernel launch plus host->device and
device->host transfer per component, which will be *slower* than the CPU for the
small snips in most examples; or (b) batching all Voigt components/spectra into
one kernel launch, which is where the `DH_orders` win actually lives.
Recommendation: define the *interface* per-component in 4.1 (so `port-to-gpu`
stays simple for new functions), but have the 4.3 dispatch batch across
components/spectra, with an array-size threshold below which it falls back to the
CPU path.

**Response (resolved in discussion, 2026-07-29):** Per-component interface, batch
in the dispatcher. Keep `call_GPU` per-component in `alis/functions/<name>.py`
(operating on device arrays) for clean new-function ergonomics; the 4.3
dispatcher batches same-type components/spectra into one launch with a size-
threshold CPU fallback. The kernel's *location* is runtime-irrelevant, so
hard-coding kernels in `alfit`/`model_eval` would not be faster -- speed comes
from transfer frequency (upload the wave grid once per iteration), batching, and
on-device intermediates, all owned by the dispatcher. Folded into Tasks 4.1/4.3.

**Q4.8 — GPU workers vs the existing process Pool (Tasks 4.3/4.5).**
`minimise.fdjac2` now holds a *persistent* `ncpus` Pool for the chunked Jacobian
(Task 3.4 Phase 1). How should `run ngpus` compose with it: (a) each existing
Pool worker binds to a GPU round-robin (`cuda.select_device(rank % ngpus)`), or
(b) a separate GPU worker pool sized by `ngpus` alongside the CPU Pool? Note
CUDA contexts do not survive `fork`, so this decision also pins the
multiprocessing start method (`spawn` required on Linux if contexts are created
before forking). Recommendation: (a) — it reuses the Phase 1/2 machinery
(persistent pool, chunking, subset-pickling) and keeps one worker model.

**Response (resolved in discussion, 2026-07-29):** Either-or, **not** hybrid (see
Task 4.3): the backend is CPU or GPU per fit, never both computing columns at
once. The GPU backend reuses the persistent Pool machinery but sizes it to
`ngpus` with one GPU bound per worker (`cuda.select_device(rank)`, `spawn`). A
per-fit `auto`/`cpu`/`gpu` selector (new Task 4.3a) picks CPU vs GPU by a warmed
Jacobian probe at `p0`, then commits the whole fit to the winner.

**Q4.9 — GPU test/example references (Task 4.2/4.6, follows Q4.2).** Rather than
generating a separate set of GPU golden files, I propose running *existing*
examples with `run ngpus 1` and `run ngpus 4` and comparing against the **same
CPU references**: a 1e-12 profile difference is far inside the Stage 0 tolerances
(params 10% of 1σ, chi-squared 0.1–1%, `_fit.dat` error-based). These would carry
a new `gpu` marker, deselected by default so CI stays CPU-only (Q4.2), and run on
demand locally. Does that satisfy the "single GPU and multiple simultaneous GPUs"
coverage you asked for, or do you want dedicated GPU examples with their own
reference files?

**Response:** I agree with your proposal. We will run existing examples with
`run ngpus 1` and `run ngpus 4` and compare against the same CPU references.
This will provide sufficient coverage for both single GPU and multiple
simultaneous GPUs without the need for dedicated GPU examples with their
own reference files. It would be good if there is a command that I can
use to run the GPU tests on demand locally.

**Q4.10 — `auto` semantics when `ngpus` is unset.** `backend`
defaults to `auto`, but `ngpus` defaults to `None` (GPU off). On a GPU machine
with `ngpus` unset, should `auto` (a) stay CPU-only -- probe GPU only when the
user explicitly requests `ngpus` (>0); or (b) auto-detect the GPUs and probe
anyway (using all of them)? (a) keeps CPU the default unless the user opts in (and
keeps CI/tests CPU-only without special-casing); (b) is more "automatic" but
silently uses the GPU for every fit on a GPU box. Recommendation: (a).

**Response:** I agree with your recommendation. We will stay CPU-only unless
the user explicitly requests `ngpus` (>0). This keeps CPU the default unless
the user opts in. However, if might be good to have a warning message that
informs the user that there are GPUs available (and their number) and that
they can use them by setting `ngpus` to a value greater than 0.

**Q4.11 — `gpu` optional-dependency extra (new, 2026-07-29).** `pyproject.toml`
currently declares `gpu = ["cupy"]` -- a Stage 1 placeholder. Since the stack is
`numba.cuda` (Q4.1), the extra should become `gpu = ["numba"]` (numba is already
present in the `py313` env, Q4.5). Confirm the switch to `numba` (and that CUDA
itself is provided by the user's driver/toolkit, not pip)? Recommendation: yes.

**Response:** Yes, we will switch the `gpu` optional-dependency extra to `numba`.
I think we should also make sure that the user installs the cudatoolkit. Related to
this, numba does not currently support dynamic parallelism and texture memory. Do
you foresee this as a problem for our implementation?

## Prompts

1. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing (note that some filenames mentioned are out of date, e.g. `alfunc_BLAH`, and need to be updated). Ask further queries if needed. Can we delay this stage until later?

2. Please read this doc, including my responses to your queries, and ask any further queries you have before commencing. Once you have clarified any remaining questions, please let me know, and I will give you the go-ahead to start work on this stage.

3. Please read this doc, including my responses to your queries, and execute Task 4.0.

4. Please read this doc, including my responses to your queries, and execute Task 4.1.

5. Please read this doc, including my responses to your queries, and execute Task 4.2.

6. Please read this doc, including my responses to your queries, and execute Task 4.3.
