---
name: gui-component
description: Scaffold a new GUI widget or panel for the ALIS prepfit/fitting GUI, following the design patterns in alis/prepfit/specplot.py.
---

Create a new GUI component for the ALIS prepfit/fitting interface. The existing GUI uses matplotlib with the Qt5Agg backend; all widgets follow the class-based event pattern established in `alis/prepfit/specplot.py`.

## Existing GUI patterns

From `alis/prepfit/specplot.py`:
- Widgets are classes (e.g. `SelectRegions`) that receive `canvas`, `ax`, and spectrum data in `__init__`.
- Key events: `canvas.mpl_connect('key_press_event', self.onkeypress)`
- Mouse events: `canvas.mpl_connect('button_press_event', self.onclick)`
- State is stored on `self`; the canvas is redrawn with `self.canvas.draw()`.
- Key bindings are documented in the class docstring.

## Steps

1. Ask the user for:
   - Name and purpose of the new component
   - Data it needs (spectrum arrays, fit parameters, atomic data, etc.)
   - Interactions it should support (key bindings, mouse clicks, text entry)
   - Where it fits in the GUI flow (standalone window, panel in the main window, overlay)

2. Read `alis/prepfit/specplot.py` fully to understand the existing class structure and style before writing any code.

3. Create the new component as a class:
   - If self-contained and small: add it to `alis/prepfit/specplot.py`
   - If substantial: create `alis/prepfit/<component_name>.py`

4. Implement:
   - `__init__`: set up axes, connect events, initialise state
   - Event handlers: `onkeypress`, `onclick`, or custom handlers as needed
   - Helper methods for drawing or updating the display
   - A `show()` or `__call__` method to launch the component

5. Document all key bindings in the class docstring using the same format as `SelectRegions`.

6. Add a minimal usage example to `examples/prepfit/` showing how to invoke the new component.

## Notes

- Follow the code style of `specplot.py` exactly (PEP 8, class-based, matplotlib idioms).
- Do not call `plt.show()` at module level; let the caller control the event loop.
- Do not modify existing classes or key bindings without explicit instruction.
