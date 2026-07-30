---
name: port-to-gpu
description: Port an existing call_CPU model function to call_GPU using numba.cuda, following the functions/base.Base interface, and verify numerical equivalence against the CPU version.
---

Port the `call_CPU` method of an ALIS model function to `call_GPU`, then verify
the GPU result matches the CPU result to the agreed tolerance.

Model functions live in `alis/functions/<name>.py` (e.g. `voigt.py`); the base
class is `Base` in `alis/functions/base.py`. The historical `alfunc_<name>.py`
layout was removed in Stage 2 — do not look for it.

## Decisions already made (Stage 4 — do not re-litigate)

- **Stack: `numba.cuda`.** Not CuPy, not PyCUDA (Q4.1). The reference
  implementation is `context/voigt_gpu/simple_test.py`, a pure-numba Faddeeva.
- **float64 throughout** (Q4.3/Q4.6). No float32 fast path: the argument
  `v = wv*((wv/ww)-1)/bl` suffers catastrophic cancellation, so float32 bottoms
  out at ~2e-8 absolute — four orders short of the 1e-12 gate, and it would
  degrade the finite-difference Jacobian and hence the covariance matrix.
- **Per-component interface, batched dispatch** (Q4.7). `call_GPU` stays
  per-component and **operates on arrays that are already on the device** — it
  launches a kernel, it does not transfer. The dispatcher (Task 4.3) batches
  same-type components/spectra into one launch and owns transfers; below a
  size threshold it falls back to the CPU path.
- **Lazy import.** `numba` / CUDA must be imported *inside* the GPU code path,
  never at module import time, so CPU-only installs (no `gpu` extra, no CUDA
  toolkit) never touch them. A missing `numba`/GPU at selection time falls back
  to CPU with a clear message.
- **No dynamic parallelism, no texture memory** (numba supports neither, and
  neither is needed). Small read-only coefficient tables belong in numba
  **constant memory** (`cuda.const.array_like`).

## Interface

```python
# alis/functions/base.py
def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
    # x    : 1D numpy array of (sub-pixel) wavelengths
    # p    : 2D numpy array of parameters, shape (ncomponents, nparams)
    # ae   : 'em' (emission -> sum components) | 'ab' (absorption -> product)
    #        | 'zl' (zero level) | 'cv' (convolution)
    # mkey : list of per-component keyword dicts (one per row of p)
    # returns: 1D numpy array, same length as x

def call_GPU(self, x, p, ae='em'):
    # Same semantics, device arrays in and out.
```

Check the current signature in `alis/functions/base.py` before you start —
Task 4.1 formalises this contract and may have refined it.

## Steps

1. Read the target function's `call_CPU` in full. Note exactly which parameters
   and `mkey` keywords it consumes (e.g. `voigt` reads `freq`, `logN`,
   `ColDensScale`) — the kernel must reproduce every branch.

2. Assess portability:
   - **Elementwise numpy** → direct `@cuda.jit` kernel over the wavelength grid.
   - **Special functions** (Voigt/Faddeeva, `scipy.special.wofz`) → port the
     device functions from `context/voigt_gpu/simple_test.py`
     (`faddeeva_real`, `faddeeva_re`, `erfcx_y100`, `sinh_taylor`, …), with
     `erfcx_coeffs.dat` and the `expa2n2` table in constant memory.
   - **Python loops over components** → make the component index a grid
     dimension so one launch covers the whole batch.

3. Implement `call_GPU` immediately after `call_CPU` in the same module, with
   the `numba` import inside the method (or inside a lazily-initialised helper).
   Do **not** modify `call_CPU` — the port is additive.

4. Verify numerical equivalence (`gpu`-marked test, see below):

   ```python
   import numpy as np
   from numba import cuda

   x = np.linspace(1200.0, 1300.0, 100_000)          # float64
   p = np.array([[...]])                              # representative params

   cpu = func.call_CPU(x, p, ae='ab', mkey=mkey)
   d_x = cuda.to_device(x)
   gpu = func.call_GPU(d_x, cuda.to_device(p), ae='ab').copy_to_host()

   assert np.max(np.abs(gpu - cpu)) < 1e-12
   ```

   The gate is **absolute 1e-12** (Q4.3). `context/voigt_gpu/` reaches ~1e-15,
   so a result near 1e-8 means an argument-conditioning or precision bug, not an
   acceptable tolerance.

5. Place the test behind the `gpu` marker so CPU-only CI skips it:

   ```python
   pytestmark = pytest.mark.gpu     # deselected unless --run-gpu is given
   ```

   Run it with `pytest --run-gpu -m gpu`.

6. Report: which device functions were needed, where the coefficient tables
   live, the measured worst-case `|GPU - CPU|`, and the array size above which
   the GPU beats the CPU (feeds the dispatcher's size threshold).

## Notes

- This machine has 4× RTX 2080 Ti (Turing, compute 7.5). FP64 runs at 1/32 of
  the FP32 rate, yet the float64 Faddeeva kernel still beats `scipy.wofz` by
  15–73× for ≥1e4 sub-pixels. Below ~1e3 sub-pixels the margin collapses to
  ~1.8× (launch + transfer dominated) — hence the batched dispatch.
- Keep intermediates **on the device**. Downloading per component would erase
  the win; only the final convolved model comes back to the host.
- The sub-pixel wavelength grid is uploaded once per iteration by
  `alfit.prepare_iteration()` (it can change when `run renew_subpix True`), not
  once per fit and not per component.
- If no GPU is available, implement the code, state clearly that it could not be
  executed, and leave the equivalence test `gpu`-marked for a machine that has
  one.
