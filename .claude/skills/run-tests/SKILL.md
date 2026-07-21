---
name: run-tests
description: Run the ALIS pytest suite, optionally filtering by module or test name; reports pass/fail counts and tracebacks for any failures.
---

Run the ALIS test suite using pytest. If the user specifies a module, file, or test name to filter on, pass it as a `-k` or path argument to pytest.

## Steps

1. Identify the test scope from the user's request:
   - No filter → run all tests: `pytest tests/ -v`
   - Module filter (e.g. "voigt") → `pytest tests/ -v -k voigt`
   - Specific file → `pytest tests/test_voigt.py -v`

2. Activate the project virtual environment if needed (`.venv/` in the repo root at `/Users/rcooke/Software/ALIS`).

3. Run pytest from the repo root.

4. Report:
   - Total passed / failed / skipped counts
   - Full tracebacks for every failure, including the relevant source lines
   - Any warnings that may indicate future breakage

5. If all tests pass, confirm and briefly summarise what was tested.

6. If tests fail, identify the root cause from the traceback and suggest a fix. Do not modify tests to make them pass — fix the underlying code.

## Notes

- Test files live in `tests/` and are named `test_*.py`.
- Do not run git commands or modify any files without explicit instruction.
