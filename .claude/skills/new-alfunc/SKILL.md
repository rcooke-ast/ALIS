---
name: new-alfunc
description: Scaffold a new ALIS model function (alis/functions/<name>.py) from the functions.base.Base interface, register it, and add its unit tests -- with a GPU path if it is worth one.
---

Create a new model function for the ALIS fitting engine. Every model function
subclasses `Base` in `alis/functions/base.py` and lives in
`alis/functions/<name>.py`.

**The pre-Stage-2 layout (`alis/alfunc_<name>.py`, `alfunc_base`) no longer
exists — do not look for it.**

## Read these first

- `alis/functions/base.py` — the interface, and the CPU/GPU contract in the
  `Base` docstring.
- `alis/functions/gaussian.py` — the minimal reference implementation.
- `alis/functions/voigt.py` — the full one: keywords, atomic data, a GPU path.
- `tests/test_function_interface.py` — the invariants your function must
  satisfy. Run it as soon as the file exists; it is faster than a fit and it
  names what is wrong.

## Ask the user

1. The function name (becomes `_idstr`, the filename, and the registry key —
   all three must match).
2. What it models, and whether it is emission, absorption, zero-level or
   convolution.
3. Its parameters: name, default, whether fixed by default, and lower/upper
   limits.
4. Any keywords beyond `specid` / `continuum` / `blind`.
5. Whether it needs atomic data.

## The interface

| Method | Purpose |
|--------|---------|
| `__init__` | Set `_idstr`, `_pnumr`, `_keywd`, `_keych`, `_keyfm`, `_parid`, `_defpar`, `_fixpar`, `_limited`, `_limits`, `_svfmt`, `_prekw`, then build `_keywd['input']` |
| `call_CPU` | Evaluate the model: `(x, p, ae='em', mkey=None, ncpus=1)` -> 1D array the length of `x` |
| `load` | Parse one model-file line into the `mp` structure |
| `parin` | Convert an input parameter to the value `call_CPU` uses |
| `parout` | Format best-fit parameters for `.mod.out` |
| `set_pinfo` | Limits, fixed flags and tied expressions for the minimiser |
| `set_vars` | Build the `(nrows, nparams)` array `call_CPU` receives |
| `tick_info` | Wavelengths/labels for plot tick marks (return `[], []` if none) |

`call_GPU` is **optional** — see below. Do not override it to raise
`NotImplementedError`: the inherited version falls back to `call_CPU`, and the
Task 4.3 dispatcher relies on being able to call either uniformly.

## Steps

1. Write `alis/functions/<name>.py`, matching the style of the neighbouring
   modules (import order, attribute naming, `if getinst: return` at the end of
   `__init__`).

2. Build the input map exactly as `Base.__init__` does — it is what `parout`
   consults to decide which parameters the user actually supplied, and a
   parameter missing from it is silently dropped from the written model:

   ```python
   tempinput = self._parid + list(self._keych.keys())
   self._keywd['input'] = dict(zip(tempinput, [0] * np.size(tempinput)))
   ```

3. **Register it in `call()` at the bottom of `alis/functions/base.py`** — add
   an entry to the `fd` dict. This is the step people forget; until it is done
   the model file just reports an unknown function.

4. **If it reads atomic data, add its name to `sendatomic`** in the same
   function. Otherwise `self._atomic` stays `None` and the fit dies with
   `'NoneType' object is not subscriptable` from inside `set_vars`.

5. Run the interface gate — it covers every invariant in steps 1-4:

   ```bash
   pytest tests/test_function_interface.py -q
   ```

6. Add `tests/test_<name>.py`, marked `pytest.mark.unit`:
   - `call_CPU` returns the right shape for one row and for several;
   - `ae='em'` sums the rows and `ae='ab'` multiplies them;
   - the model's own physics: a value you can compute by hand, the behaviour at
     a limit (zero amplitude, infinite width), and each keyword that changes the
     arithmetic;
   - `load` accepts a representative model-file line and `parout` round-trips it.

7. Run `pytest -m unit` and the fast gate (`pytest -m fast`) before finishing.

## Adding a GPU path

Only worth it for a function whose cost scales with the sub-pixel grid — the
Stage 4.2 measurements put the crossover at ~1e4 pixel-components, and the
dispatcher will not send anything smaller than `run gputhresh` to the device
anyway. Use the `port-to-gpu` skill; the three things that make it *work end to
end* rather than merely exist are:

- `_gpu_supported = True` as a **class** attribute (overriding `call_GPU` alone
  does not opt in — several functions used to carry a verbatim copy of the stub);
- `call_CPU` and `call_GPU` defined in the **same class** (a subclass that
  replaces `call_CPU` inherits the parent's flag and would run the parent's
  kernel — this is enforced by a test);
- a `gpu_warmup_args()` hook, so `run backend auto` can compile the kernel
  before it times anything.

## Notes

- Do not modify existing model functions.
- `_pnumr` is the number of parameters, and every per-parameter list must have
  that many entries — except for a genuinely variable-length function (see
  `lsfspline`), where `_pnumr` is the minimum.
- `_keywd`, `_keych` and `_keyfm` must describe the *same* keywords.
- User-supplied functions are a separate mechanism (`alis/functions/user.py`);
  a function that belongs in ALIS goes in `alis/functions/`.
