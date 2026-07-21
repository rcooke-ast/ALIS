---
name: gui-dev
description: Launch the ALIS prepfit GUI, exercise a specific interaction, and report any errors or visual regressions.
---

Launch and exercise the ALIS prepfit GUI (`alis/prepfit/specplot.py`), which uses matplotlib with the Qt5Agg backend for interactive spectral region selection and fit inspection.

## Steps

1. Identify what to test: a specific GUI interaction, a newly added widget, or a general smoke test.

2. Verify that the Qt5 backend is available:
   ```
   python -c "import matplotlib; matplotlib.use('Qt5Agg'); import matplotlib.pyplot as plt; print('Qt5Agg OK')"
   ```
   If this fails, report the missing dependency (`PyQt5` or `PySide2`) and stop.

3. Launch the GUI. For the region-selection workflow, use the prepfit example:
   ```
   cd /Users/rcooke/Software/ALIS/examples/prepfit
   python select_fitting_regions.py
   ```
   For the main `specplot` widget, construct a minimal launch script that loads a spectrum from one of the `examples/` directories.

4. Exercise the specific interaction the user requested (e.g. region selection, zoom, key binding `s` to save regions).

5. Report:
   - Whether the GUI launched without errors or Qt warnings
   - Whether the interaction worked as expected
   - Any tracebacks or visual anomalies observed
   - A description of what was displayed

6. If errors occurred, identify the source line in `alis/prepfit/specplot.py` and suggest a fix.

## Notes

- The GUI requires a display. If running headlessly, note this and suggest using a virtual framebuffer (`Xvfb`) or skipping GUI tests.
- Documented key bindings in `SelectRegions`: `s` saves the regions file. Any new bindings added during development must be listed in the class docstring.
- Do not modify GUI source files without explicit instruction.
