---
name: build-docs
description: Build the ALIS Sphinx documentation locally (ReadTheDocs target) and report any warnings or broken cross-references.
---

Build ALIS's documentation site (`doc/source/`: Sphinx, with the pages in Markdown
through MyST and the furo theme) as readthedocs and CI build it, run its tests, and
report every warning. Linkcheck runs only when asked.

## Steps

1. **An environment with the `docs` extra** (`sphinx`, `furo`, `myst-parser`).
   Check the current one:
   ```
   python -c "import sphinx, furo, myst_parser"
   ```
   If that fails, use a scratch venv in the session's scratchpad rather than
   changing the user's environment:
   ```
   python3 -m venv <scratchpad>/docs-venv
   <scratchpad>/docs-venv/bin/python -m pip install -e "/Users/rcooke/Software/ALIS[docs]"
   ```
   `conf.py` imports ALIS from the repository itself (only `alis/__init__.py`, for
   the version), so the docs packages alone are enough for the build. The
   editable install matches readthedocs, and is needed once the API pages (the
   Developer guide's) import ALIS's modules.

2. **Build, failing on any warning** (as CI's `docs` job and readthedocs do):
   ```
   sphinx-build -W --keep-going -b html doc/source doc/_build/html
   ```
   `doc/_build/` is ignored by git. Output elsewhere (the scratchpad) is fine too.

3. **Run the documentation's tests** (marked `unit`; they need no Sphinx):
   ```
   python -m pytest tests/test_docs.py -q
   ```
   They check that every page of `PAGES` exists and lists its sources, that no page
   is missing from `PAGES`, that no file of the site names what it leaves out (a
   removed tool, the v1.0 manual, `context/`: `LEFT_OUT`), or says "dashboard" or
   "RJC", that About ALIS and `CITATION.cff` cite the same paper, and that the
   `alis` lexer reads every example model.

4. **Linkcheck, on request only** (other sites fail at random, so it is not in CI):
   ```
   sphinx-build -b linkcheck doc/source doc/_build/linkcheck
   ```
   Report a broken link to another site; do not "fix" it blindly. A link to the
   site's own readthedocs address fails until the version it names is built.

5. **Report:** each warning or error with its file and line, the tests' result,
   linkcheck's broken links if it ran, and where the HTML is
   (`doc/_build/html/index.html`).

## Notes

- How the pages are written (one question per page, the GUI and Ryan Cooke by
  name, the code as the authority, one source for each fact) is in
  `claude_prompts/documentation_stage0.md`, "How the pages are written". Each
  page's sources are in its `% Sources:` comment.
- A new page needs its address in `tests/test_docs.py`'s `PAGES` and a place in a
  toctree; addresses never change once published.
- The LaTeX files in `doc/tex_files/` are v1.0's manual, kept for reference only.
  The code is the authority for what a page says.
- This skill builds and reports. It does not change pages unless asked.
