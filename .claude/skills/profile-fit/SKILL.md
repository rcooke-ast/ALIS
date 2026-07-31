---
name: profile-fit
description: Profile an ALIS fit using cProfile, identify the top CPU bottlenecks by cumulative time, and suggest optimisation targets for GPU porting.
---

Profile an ALIS fit to identify where CPU time is spent, then prioritise which functions to optimise or port to GPU.

## Steps

1. Identify the model file to profile. If the user does not specify one, suggest a small example such as `examples/lls/model/fit_spectra.mod`.

2. Write and run a profiling script:
   ```python
   import cProfile, pstats, io
   from alis import alis
   
   pr = cProfile.Profile()
   pr.enable()
   alis.main(modelfile='<path/to/fit.mod>')
   pr.disable()
   
   s = io.StringIO()
   ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
   ps.print_stats(30)
   print(s.getvalue())
   ```

3. Parse the profiler output and report:
   - Top 15 functions by **cumulative time**, with file path and line number
   - Top 5 functions by **self time** (most expensive individual calls)
   - Which ALIS modules account for the most time (e.g. `functions/voigt`, the convolution functions, `minimise`)

4. Identify GPU porting candidates — functions that are:
   - Called inside the minimiser loop (executed thousands of times per fit)
   - Dominated by numpy array operations (naturally parallelisable)
   - Embarassingly parallel across spectral pixels

5. Suggest concrete next steps: which `call_CPU` methods to port first, and whether Voigt profile evaluation (`functions/voigt`) or convolution (`functions/vfwhm`, `functions/lsf`, ...) is the dominant cost.

## Notes

- For line-level profiling of a specific bottleneck, use `line_profiler`:
  ```
  kernprof -l -v script.py
  ```
  Decorate the target function with `@profile` temporarily.
- Always profile with a realistic model (multiple components, multiple spectral regions) to get representative timings.
- Do not modify any source files during profiling.
- Run from the repo root with the virtual environment active.
