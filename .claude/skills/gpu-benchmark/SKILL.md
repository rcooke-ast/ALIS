---
name: gpu-benchmark
description: Benchmark CPU vs GPU throughput for a given ALIS model function or full fit using numba.cuda, and report the speedup factor.
---

Measure and compare the execution time of an ALIS model function or full fit on
CPU vs GPU. The GPU stack is **`numba.cuda`**, float64 throughout (Stage 4,
Q4.1/Q4.6). Model functions live in `alis/functions/<name>.py`.

## Steps

1. Identify what to benchmark: a single model function's `call_CPU`/`call_GPU`,
   or a full ALIS fit.

2. For a **single model function**:

   ```python
   import numpy as np, timeit
   from numba import cuda
   from alis.functions import base

   func = base.call(getinst=True)['<name>']       # or import the class directly
   x = np.linspace(1200.0, 1300.0, 100_000)       # float64
   p = np.array([[...]])                           # representative params

   d_x, d_p = cuda.to_device(x), cuda.to_device(p)

   # Warm up BOTH paths: the first GPU call pays numba JIT + CUDA context
   # creation (seconds), which would otherwise dominate the measurement.
   func.call_CPU(x, p, ae='ab', mkey=mkey)
   func.call_GPU(d_x, d_p, ae='ab'); cuda.synchronize()

   n = 200
   t_cpu = timeit.timeit(lambda: func.call_CPU(x, p, ae='ab', mkey=mkey),
                         number=n) / n
   t_gpu = timeit.timeit(
       lambda: (func.call_GPU(d_x, d_p, ae='ab'), cuda.synchronize()),
       number=n) / n
   print(f"CPU {t_cpu*1e3:.3f} ms | GPU {t_gpu*1e3:.3f} ms "
         f"| {t_cpu/t_gpu:.1f}x")
   ```

3. **Sweep the problem size.** This is the point of the exercise: report the
   crossover, not a single number. Use decades of sub-pixel count (1e3 … 1e7)
   and, separately, the number of batched components/spectra. Established
   float64 baseline on 4× RTX 2080 Ti vs `scipy.wofz`:

   | sub-pixels | CPU `wofz` | GPU fp64 | speedup |
   |-----------:|-----------:|---------:|--------:|
   | 1e3 | 0.105 ms | 0.057 ms | 1.8x |
   | 1e4 | 0.877 ms | 0.056 ms | 15.6x |
   | 1e5 | 8.20 ms | 0.231 ms | 35.4x |
   | 1e6 | 101 ms | 1.85 ms | 54.7x |
   | 1e7 | 1132 ms | 15.6 ms | 72.7x |

   A new measurement far below this line means the kernel is transfer-bound or
   under-occupied, not that the GPU is slow.

4. For a **full fit**, time the same `.mod` under `run backend cpu` vs
   `run backend gpu` (with `run ngpus N`), on a model large enough to matter —
   `context/fitting_examples/DH_orders/` (351 spectra) is the prime beneficiary;
   `helium34/Her36` is the compact-model counterpoint. Report wall time per
   Jacobian evaluation as well as total fit time, and confirm the two backends
   land on the same chi-squared.

5. Report:
   - Function / fit name, GPU model, CPU model, `ncpus`/`ngpus` used
   - CPU time, GPU time **including `cuda.synchronize()`**, speedup
   - Whether the kernel is compute- or bandwidth-bound (compare achieved GB/s
     against the card's peak — 616 GB/s on a 2080 Ti; the Faddeeva kernel runs
     at ~1.7% of peak, i.e. firmly compute-bound)
   - The array size at which the GPU overtakes the CPU — this feeds the
     dispatcher's size threshold for CPU fallback

## Notes

- **Always `cuda.synchronize()` before stopping the timer.** Kernel launches are
  asynchronous; omitting it measures launch latency, not compute.
- **Warm up both paths.** The first `@cuda.jit` call compiles and creates a CUDA
  context — seconds, not milliseconds. This is also why `run backend auto` warms
  both backends before its timing probe (Task 4.3a): a cold probe mis-picks CPU.
- Benchmark transfers **separately** from compute. `call_GPU` takes device
  arrays by contract, so a like-for-like kernel comparison excludes transfer;
  quote the host↔device cost as its own line so the dispatcher's
  once-per-iteration upload strategy can be judged on the evidence.
- FP64 on Turing (compute 7.5) runs at 1/32 the FP32 rate. That penalty is
  accepted deliberately — float32 cannot meet the 1e-12 equivalence gate — so do
  not report FP64 as a defect.
- If no GPU is present, say so and state which hardware the benchmark needs.
