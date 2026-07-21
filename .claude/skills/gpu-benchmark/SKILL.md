---
name: gpu-benchmark
description: Benchmark CPU vs GPU throughput for a given ALIS model function or full fit, and report the speedup factor.
---

Measure and compare the execution time of an ALIS model function or full fit on CPU vs GPU.

## Steps

1. Identify what to benchmark from the user's request: a single `alfunc` method or a full ALIS fit.

2. For a **single model function**, run:
   ```python
   import numpy as np, cupy as cp, timeit
   from alis import alfunc_<name>
   
   func = alfunc_<name>.call()
   x_np = np.linspace(1200.0, 1300.0, 100_000)
   p_np = np.array([[<representative_params>]])
   x_cp, p_cp = cp.asarray(x_np), cp.asarray(p_np)
   
   # Warm up (important for GPU JIT compilation)
   func.call_CPU(x_np, p_np)
   func.call_GPU(x_cp, p_cp); cp.cuda.Stream.null.synchronize()
   
   n = 200
   t_cpu = timeit.timeit(lambda: func.call_CPU(x_np, p_np), number=n) / n
   t_gpu = timeit.timeit(
       lambda: (func.call_GPU(x_cp, p_cp), cp.cuda.Stream.null.synchronize()),
       number=n
   ) / n
   print(f"CPU: {t_cpu*1e3:.2f} ms | GPU: {t_gpu*1e3:.2f} ms | Speedup: {t_cpu/t_gpu:.1f}x")
   ```

3. For a **full fit**, time `alis.main()` with GPU enabled vs disabled (once this option exists in the new version).

4. Vary the problem size (wavelength array length, number of velocity components) and report how speedup scales.

5. Report:
   - Model function / fit name and hardware used (GPU model, CPU model)
   - CPU time, GPU time (including device synchronisation), speedup factor
   - Whether speedup is memory-bandwidth-limited or compute-limited
   - Recommended minimum array size at which GPU becomes beneficial over CPU

## Notes

- Always call `cp.cuda.Stream.null.synchronize()` before stopping the GPU timer — omitting this measures only kernel-launch latency, not actual compute time.
- Include host→device and device→host data transfer time in GPU benchmarks to give a realistic comparison for the full pipeline.
- Warm up both CPU and GPU before timing to avoid cold-start effects (numba JIT compilation, CUDA context initialisation).
- If no GPU is available, report this and note which hardware the benchmark should be re-run on.
