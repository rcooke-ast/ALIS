---
name: test-coverage
description: Run pytest with coverage reporting, identify untested code paths in a given module, and suggest the highest-value tests to add next.
---

Measure test coverage for the ALIS codebase or a specific module, then recommend where to add tests for the most impact.

## Steps

1. Identify the scope from the user's request: all modules, or a specific file (e.g. `alis/alfunc_voigt.py`).

2. Install `pytest-cov` if not already present:
   ```
   pip install pytest-cov
   ```

3. Run pytest with coverage:
   ```
   pytest tests/ --cov=alis --cov-report=term-missing --cov-report=html -v
   ```
   For a specific module:
   ```
   pytest tests/ --cov=alis/alfunc_voigt --cov-report=term-missing -v
   ```

4. Parse the `term-missing` output and report:
   - Overall coverage percentage for the scope
   - Per-file coverage percentage
   - Line numbers of uncovered code for each file

5. For each uncovered region, assess its importance:
   - **High value**: code in the minimiser loop (executed on every fit iteration), model evaluation paths
   - **Medium value**: error-handling branches, parameter-limit checks, edge-case conditionals
   - **Low value**: dead code, Python 2 compatibility stubs (`try: input = raw_input`), commented-out code

6. Suggest the 3–5 highest-value tests to add, with a brief description of each. Prefer tests that exercise load-bearing code over those that inflate coverage with trivial assertions.

## Notes

- Requires `pytest-cov` (`pip install pytest-cov`).
- The HTML report is written to `htmlcov/`; open `htmlcov/index.html` for a line-by-line view.
- A high coverage percentage is not the goal — correctness is. Prioritise tests that would catch real regressions.
- Do not add trivial tests (e.g. asserting `__init__` sets an attribute) purely to raise the coverage number.
