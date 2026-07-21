---
name: new-alfunc
description: Scaffold a new ALIS model-function module (alfunc_<name>.py) from the alfunc_base.Base interface, filling in all required methods and registering the function.
---

Create a new model-function file `alis/alfunc_<name>.py` for the ALIS fitting engine. Every model function must subclass `alfunc_base.Base` and implement the full interface.

## Required interface (from `alis/alfunc_base.py`)

Every model function must implement these methods:

| Method | Purpose |
|--------|---------|
| `__init__` | Set `_idstr`, `_pnumr`, `_keywd`, `_keych`, `_keyfm`, `_parid`, `_defpar`, `_fixpar`, `_limited`, `_limits`, `_svfmt`, `_prekw` |
| `call_CPU` | Evaluate the model on the CPU given wavelength array `x` and parameter array `p` |
| `call_GPU` | GPU equivalent (stub raising `NotImplementedError` is acceptable initially) |
| `load` | Parse a parameter string from the model file into the `mp` data structure |
| `parin` | Set parameter values and info for the minimiser |
| `parout` | Format best-fit parameters for output files |
| `set_pinfo` | Set minimiser parameter info (bounds, fixed flags, tied parameters) |
| `set_vars` | Evaluate the model for a given parameter set during fitting |
| `tick_info` | Return parameter tick labels used in diagnostic plots |

## Steps

1. Ask the user for:
   - The function name (used in `_idstr` and the filename)
   - The number of parameters (`_pnumr`)
   - Parameter names, default values, whether each is fixed by default, and lower/upper limits
   - Whether this is an emission, absorption, or convolution model

2. Read `alis/alfunc_base.py` fully for the interface, then read `alis/alfunc_gaussian.py` as a minimal reference implementation and `alis/alfunc_voigt.py` for a more complete example.

3. Create `alis/alfunc_<name>.py`, implementing all required methods. Match the code style of existing `alfunc_*.py` files exactly (same import order, same attribute naming, same `__init__` structure).

4. Register the new function in `alis/alfunc_base.py` in the `call()` function at the bottom of the file, where other model functions are imported and returned.

5. Write a minimal smoke test in `tests/test_<name>.py`:
   - Instantiate the class
   - Call `call_CPU` with a simple linspace wavelength array and check output shape
   - Check `ae='em'` and `ae='ab'` both return the correct array shape

## Notes

- `call_GPU` can initially raise `NotImplementedError("GPU not yet implemented for <name>")`.
- Do not remove or modify existing model functions.
- The `_keywd['input']` dict must be constructed in `__init__` exactly as shown in `alfunc_base.Base.__init__` — do not skip this step.
