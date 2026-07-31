---
name: gen-tests
description: Generate comprehensive pytest unit tests for a specified ALIS module or function, covering normal operation, boundary conditions, and edge cases.
---

Write pytest unit tests for an ALIS module or function.

## Steps

1. Identify what to test from the user's request (e.g. "write tests for the
   Voigt function" -> `alis/functions/voigt.py`, or "test `set_params` in
   `alis/load.py`"). Model functions live in `alis/functions/<name>.py`; the
   pre-Stage-2 `alfunc_*.py` / `alload` / `alcsmin` names no longer exist.

2. Read the target module in full to understand:
   - All public functions and methods and their signatures
   - Input types, expected output types, and invariants
   - Any documented edge cases, parameter bounds, or constraints
   - Exceptions that can be raised and under what conditions

3. Read existing tests in `tests/` for style reference and to avoid duplicating coverage.

4. Generate tests in `tests/test_<module>.py`. For each function or method, write at least:
   - A **happy-path** test with representative inputs
   - A **boundary** test (empty array, single element, parameter at its minimum or maximum limit)
   - An **error** test where applicable (confirm that bad input raises the expected exception)

5. For model functions (`alis/functions/<name>.py`), always include:
   - Instantiation: `func = base.call(getinst=True)['<name>']`
   - The shared interface invariants are already covered for *every* function by
     `tests/test_function_interface.py` -- run it rather than duplicating it
   - Output shape: `assert func.call_CPU(x, p).shape == x.shape`
   - Both emission and absorption modes: `ae='em'` and `ae='ab'`
   - Numerical regression: compare `call_CPU` output against a hardcoded expected value for a fixed input, using `numpy.testing.assert_allclose`

6. Use `numpy.testing.assert_allclose` for all floating-point comparisons. Never use `==` for floats.

7. Give each test a one-line docstring explaining what it tests.

## Notes

- Tests must pass with `pytest tests/` from the repo root in the project virtual environment.
- Mark every test `pytest.mark.unit` -- that is the batch CI runs on each push
  (`pytest -m unit`).
- Tests needing a CUDA device are marked `pytest.mark.gpu`. They run wherever a
  device is present and skip where one is not; `--run-gpu` turns a missing GPU
  into an error instead of a skip. The GPU stack is `numba.cuda`, not CuPy.
- Do not remove or modify existing tests.
- Match the import style and fixture patterns of existing test files in `tests/`.
