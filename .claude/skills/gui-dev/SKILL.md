---
name: gui-dev
description: Launch the ALIS dashboard (PySide6 through qtpy, with pyqtgraph), exercise a specific interaction, headless or on screen, and report errors, blinding leaks or visual regressions.
---

Launch and drive the ALIS dashboard. Its windows live in `alis/dashboard/qt/` (Stage 3
onwards); everything they show comes from the plain-Python project model in
`alis/dashboard/` (Stage 2): `project.Project`, `history.History`, `edit`,
`validate`, `blinding.Gate`, `remove` and `modes`. The toolkit is Qt 6 through
PySide6, written against `qtpy`, with pyqtgraph for the interactive panels (D2).

## Steps

1. Identify what to test: one interaction (drag a component, draw a region, type in
   the `.mod` panel, undo/redo), a newly added panel, or a smoke test of every tab.

2. Check the `gui` extra is installed:
   ```
   python -c "import qtpy, pyqtgraph; from qtpy import QtWidgets; print(qtpy.API_NAME, pyqtgraph.__version__)"
   ```
   If it fails, report it and suggest `pip install -e ".[gui]"`. `QT_API=pyside6`
   (the default binding) or `QT_API=pyqt6` chooses the binding.

3. Choose a project to open, from the two fits the mockups were drawn from
   (`doc/dashboard/mockups/`):
   - `context/fitting_examples/VMP_DLA/J1358p6522/model/J1358p6522.mod` (one
     dataset, isotopes, notices for D20/D22);
   - `context/fitting_examples/DH/Q1243p307/model/Q1243p307_converge_newstart76.mod`
     (three datasets, several systems);
   - `examples/blind/model/fit_spectra.mod` for anything touching blinding;
   - or a new project from a spectrum (`modes.get("voigt").new_project(...)`).
   Work on a copy (`run_alis --pack` into the scratchpad), never on the files in
   `context/` or `examples/`.

4. Launch:
   ```
   run_alisgui project.model        # a bundle
   run_alisgui fit.mod              # imports a plain fit (F4)
   ```
   Headless (CI, or no display): set `QT_QPA_PLATFORM=offscreen` and drive the
   window from a script or with pytest-qt's `qtbot` (`qtbot.mouseClick`,
   `qtbot.keyClicks`, `qtbot.waitUntil`). Take screenshots with `widget.grab().save(...)`
   and look at them (at 1440x900, the size the design is judged at, D43).

5. Exercise the interaction, then check:
   - the `.mod` text changed only where it should (compare `project.text` before and
     after, or the plan of a removal);
   - one undo gives back the text, files and `ui/project.json` exactly;
   - no blinded value appears anywhere on screen: run `blinding.strings(project)` and
     search it, and look at the screenshots for unmasked values (`▒▒▒▒` expected);
   - the terminal: an unexpected error prints its traceback there, and the window
     shows only "The ALIS dashboard has encountered an unexpected error ...";
   - no text in the window names a design document (D44: no D/F/S/QF/Q numbers).

6. Report: whether the window opened without Qt warnings, what the interaction did,
   the screenshots, any traceback with its source line, and any leak or regression.
   If something failed, find the source line in `alis/dashboard/` and suggest a fix.

## Notes

- `prepfit` (`alis/prepfit/specplot.py`, matplotlib with Qt5Agg) is the old region
  selector, kept in v2.0 and deprecated once the dashboard covers it (D4). Run it only
  when asked: `cd examples/prepfit && python select_fitting_regions.py`.
- ALIS's fits run in worker processes that are spawned, so scripts that start a fit
  must be files with an `if __name__ == "__main__":` guard, never `python -` from
  standard input (the workers cannot import `<stdin>` and the pool hangs).
- Do not modify GUI source files without explicit instruction.
