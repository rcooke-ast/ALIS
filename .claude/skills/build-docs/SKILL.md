---
name: build-docs
description: Build the ALIS Sphinx documentation locally (ReadTheDocs target) and report any warnings or broken cross-references.
---

Build the ALIS documentation using Sphinx and report warnings, errors, and broken cross-references.

## Steps

1. Check whether a Sphinx configuration file (`conf.py`) exists under `doc/` or `docs/`:
   ```
   find /Users/rcooke/Software/ALIS -name "conf.py" -not -path "*/.venv/*"
   ```
   If no `conf.py` is found, note that the Sphinx build has not been set up yet and offer to scaffold a minimal `doc/conf.py` with ReadTheDocs theme.

2. If Sphinx is set up, install dependencies if needed:
   ```
   pip install sphinx sphinx-rtd-theme numpydoc
   ```

3. Run the build with warnings as errors:
   ```
   sphinx-build -W -b html doc/ doc/_build/html
   ```

4. Parse the build output for:
   - Warnings: undefined references, missing docstrings, bad cross-links (`:func:`, `:class:`, `:mod:`)
   - Errors: missing source files, import failures in autodoc
   - Final summary line (number of warnings/errors, success/failure)

5. Report:
   - Total warning and error count
   - Each issue with filename and line number
   - Whether the build succeeded and where to find the output (`doc/_build/html/index.html`)

6. If autodoc raises `ImportError`, check that the package is installed in editable mode:
   ```
   pip install -e /Users/rcooke/Software/ALIS
   ```

## Notes

- The LaTeX files in `doc/tex_files/` are historical reference only — do not use them as the source of truth. The code itself is authoritative for documentation content.
- When writing or suggesting docstring content, read the current source code directly.
- Do not modify any documentation source files without explicit instruction.
