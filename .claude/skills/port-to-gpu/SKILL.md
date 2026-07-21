---
name: port-to-gpu
description: Port an existing call_CPU model function to call_GPU using CuPy or numba CUDA, following the alfunc_base.Base interface, and verify numerical equivalence.
---

Port the `call_CPU` method of an ALIS model function to `call_GPU`, then verify the GPU result matches the CPU result to numerical precision.

## Interface (from `alis/alfunc_base.py`)

```python
def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
    # x  : 1D numpy array of wavelengths
    # p  : 2D numpy array of parameters, shape (nparams, ncomponents)
    # ae : 'em' (emission — sum components) or 'ab' (absorption — product)
    # returns: 1D numpy array, same length as x

def call_GPU(self, x, p, ae='em'):
    # Same semantics as call_CPU but runs on GPU
    # returns: CuPy array or numpy array after device→host transfer
```

## Steps

1. Identify which model function to port. Read its `call_CPU` implementation in full.

2. Assess portability:
   - **Pure numpy operations** → straightforward CuPy substitution (replace `np` with `cp`)
   - **Special functions** (e.g. Voigt/Faddeeva) → use `cupy.scipy.special` or write a numba CUDA kernel
   - **Python loops over components** → vectorise or replace with a numba `@cuda.jit` kernel

3. Choose the GPU approach:
   - **CuPy** (preferred): replace `import numpy as np` with `import cupy as cp` in the GPU method
   - **numba CUDA** (for custom kernels): use `@cuda.jit` for performance-critical inner loops that cannot be expressed as array operations

4. Implement `call_GPU` in the model function file, immediately after `call_CPU`.

5. Write and run a numerical equivalence test:
   ```python
   import numpy as np, cupy as cp
   import numpy.testing as npt
   
   x_np = np.linspace(1200.0, 1300.0, 10000)
   p_np = np.array([[<representative_param_values>]])
   
   cpu_out = func.call_CPU(x_np, p_np)
   gpu_out = func.call_GPU(cp.asarray(x_np), cp.asarray(p_np))
   npt.assert_allclose(cpu_out, cp.asnumpy(gpu_out), rtol=1e-5,
                       err_msg="GPU output does not match CPU output")
   print("Equivalence test passed")
   ```

6. Report: which GPU approach was used, any numerical precision differences observed, and whether any operations required a custom CUDA kernel.

## Notes

- Do not modify `call_CPU` — the GPU port must be a separate method.
- If a GPU is not available locally, implement the code and note it could not be run; mark the equivalence test with `@pytest.mark.skipif(not cupy_available, reason="no GPU")`.
- CuPy is preferred over numba CUDA for operations that map directly to numpy; reserve numba CUDA for special functions or fused kernels.
