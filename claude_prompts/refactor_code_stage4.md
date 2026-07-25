# Prompt file for ALIS software refactoring -- STAGE 4

> **GPU support and modularity.** Formalise a clean CPU/GPU model-function
> interface on the refactored base class, implement a GPU Voigt profile, and add
> multiprocessed GPU dispatch with a CPU fallback. GPU is an optional install
> extra — CPU-only users must need no CUDA toolchain (plan Q4). Depends on
> Stage 2; the GPU path must reproduce the CPU results within an agreed
> numerical tolerance, checked against the Stage 0 references.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage4_log.md`.

**4.1 — Formalise the CPU/GPU model interface.**
- In the (now clean) `alfunc_base`, define a clear `call_CPU` / `call_GPU`
  contract with a numerical-equivalence requirement and a graceful "no GPU →
  CPU" fallback. Document how a function opts into GPU support.

**4.2 — GPU Voigt profile.**
- Implement `call_GPU` for the Voigt profile using the `context/voigt_gpu/`
  example (Faddeeva function), validated close to the CPU version.

**4.3 — Multiprocessed GPU dispatch.**
- Add GPU dispatch in the model-evaluation loop (multiple GPUs via `run ngpus`),
  with a clean fallback to CPU when no GPU is present.

**4.4 — New-function ergonomics.**
- Make it straightforward to add a new `alfunc_*` with both CPU and GPU paths
  plus its own unit tests (reuse/extend the `new-alfunc` and `port-to-gpu`
  skills).

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

**4.6 — Unit tests for this stage's stable surface (do last).**
- Following the cross-cutting unit-test policy
  (`claude_prompts/refactor_code_unit_tests.md`), add `unit`-marked tests for the
  *stable* code introduced in Stage 4 once its interfaces settle: the CPU/GPU
  dispatch/selection logic and fallback (4.1), pure helpers of the GPU Voigt
  (4.2), and the shared-memory buffer lifecycle (4.5) — with a CPU-vs-GPU
  numerical-equivalence test where GPU hardware is available (else skipped/mocked
  per Q4.2). Keep them fast and isolated (no full fits); the existing `unit` CI
  job picks them up automatically.

## Skills to use for this stage

- `port-to-gpu` — port a `call_CPU` to `call_GPU`, verifying numerical equivalence.
- `gpu-benchmark` — CPU vs GPU throughput for a function / full fit.
- `new-alfunc` — scaffold new functions with CPU+GPU paths.
- `run-tests` — Stage 0 gate (CPU path).

## Context

- `context/voigt_gpu/` (Faddeeva implementation: `faddeeva.py/.cc/.hh`,
  `numba_test.py`, `erfcx_coeffs.dat`), `alfunc_voigt.py` (existing `GPU_kernal`
  stub and CPU `voigtking`), the commented PyCUDA scaffolding in `alis.py`.
- The `DH_orders` model (351 spectra) is the prime GPU beneficiary.
- Plan Q4 (GPU optional extra; stack TBD).

## Queries

**Q4.1 — GPU stack.** Decide the target now: CuPy, `numba.cuda`, or PyCUDA? The
old code used PyCUDA; the `voigt_gpu` example uses numba. Recommendation: pick one
that installs cleanly as an optional extra and matches the Faddeeva example.

**Q4.2 — GPU testing in CI.** Is GPU hardware available for CI, or should GPU
tests run only locally / on demand (with CI covering the CPU path + a mocked GPU
interface)?

**Q4.3 — CPU↔GPU equivalence tolerance.** What relative tolerance is acceptable
between the GPU and CPU Voigt (single vs double precision on GPU)?

**Q4.4 — Shared memory scope (Task 4.5).** Should shared memory back only the
`FitState` data arrays (CPU-side, memory-footprint win), or be designed together
with the GPU device buffers of 4.2/4.3 as one buffer-lifecycle layer? The Task
3.4 profiling shows little standalone CPU-time upside, so co-designing it with
the GPU path (where the arrays must be shared/device-resident anyway) likely
gives the best return. Recommendation: fold it into the 4.2/4.3 GPU work rather
than build a CPU-only shared-memory layer first.

## Prompts

1. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing (note that some filenames mentioned are out of date). Ask further queries if needed. Can we delay this stage until later?
