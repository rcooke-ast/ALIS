---
name: gui-component
description: Scaffold a new panel or widget for the ALIS dashboard (alis/dashboard/qt/, PySide6 through qtpy, pyqtgraph), wired to the project model, the history, the validator and the blinding gate.
---

Create a new panel or widget for the ALIS dashboard. The windows live in
`alis/dashboard/qt/`; the logic they show lives in the plain-Python project model of
`alis/dashboard/` (Stage 2), which has no Qt and is tested with pytest alone.

## The rules every component follows

- **Qt only in `qt/`.** Import Qt through `from qtpy import QtCore, QtGui, QtWidgets`
  and plot with `pyqtgraph`. Nothing outside `alis/dashboard/qt/` may import Qt or
  pyqtgraph; `tests/test_dashboard_no_qt.py` fails if it does.
- **The text is authoritative (D7).** A component never edits the `.mod` text, a snip
  or `ui/project.json` itself. An action calls an `edit.py` function (or a `project`
  or `remove` method) to get a change or a step, and gives it to the project's
  `History` (`history.edit(change)`, `history.do(step)`), so it can be undone (F2).
  A drag is one step: wrap its updates in `with history.group("Drag ..."):`.
- **Labels.** New parameters take labels from `edit.new_component` and
  `edit.new_row_labels` (Q2.3). When an edit raises `edit.LabelNeeded`, ask the user
  for a label; offer `error.suggestion` when it is not None (column densities have
  none).
- **Every value goes through the blinding gate (F8).** Show values with
  `blinding.Gate(project)`: `value`, `word`, `best_fit`, `error`, `velocity`,
  `keyword`, `message`/`problem` for any text that may quote a value, and
  `gate.view()` for the `.mod` panel (typing over a mask goes through
  `MaskedView.to_real`). Never format a parameter's value directly.
- **Problems on their lines (F5).** Use `validate.quick(project.parsed,
  project.data_for_run())` after each edit and `validate.check(project)` after a
  pause; show each problem on its line, through the gate.
- **While the text does not read** (`project.paused`), panel edits are disabled
  (`project.require_reading()` raises); the `.mod` panel still works (QF.3). The
  window disables each tab and shows a banner meanwhile.
- **Blinding in the history** (Stage 3): a step that would show a hidden value raises
  `blinding.RevealError` from `history.do`/`edit` (the step is undone); a step that
  hides something cannot be undone past. Let the window report the error: run the
  handler through `window._run(name, handler)`, which also reads what was typed in
  the `.mod` panel first and refreshes the window (markers, panel, status bar).
- **Questions and files** go through `window.ask(kind, ...)`, `window.choose(...)`
  and `window.run_dialog(dialog)`, never a bare `QMessageBox` or `QFileDialog`, so
  that tests can answer them.
- **Cross-highlighting (S16):** `project.lines_of(item)` and `project.items_at(line)`.
- **Look (D43, D44):** neutral and light, Fusion-like; colours only from the
  Okabe–Ito palette, and every coloured state also has an icon; no text names a
  design document. Lines of code are at most 88 characters.

## Steps

1. Ask the user for: the component's name and purpose; the tab it belongs to (Data,
   Regions, Components, Fit, Plot, or the `.mod` panel, D34); the project concepts it
   shows (rows, snips, systems, components, regions, notices); and its interactions
   (clicks, drags, keys).

2. Read the agreed layout: the mockup of that tab in `doc/dashboard/mockups/` (built
   by `build_mockups.py`, published as the commentable page) and the decisions
   D34–D44 in `claude_prompts/ALIS_v2_dashboard_prompts.md`.

3. Put any logic that does not need Qt into `alis/dashboard/` (a new function in the
   module it belongs to), with pytest tests. Keep the Qt class thin.

4. Replace the placeholder `Pane` of its tab in `alis/dashboard/qt/tabs.py` (each
   tab's frame already has the mockup's panes, columns and proportions), and create
   the component in `alis/dashboard/qt/<name>.py` as a `QWidget` subclass:
   - `__init__(self, project, history, parent=None)`: build the widgets, connect the
     signals;
   - `refresh()`: redraw from the project (the tab's `refresh(session)` is called by
     the window after every step, undo and redo); colours from `qt/style.py`;
   - handlers that make steps through the history, as above;
   - every keyboard action added to the registry in `alis/dashboard/actions.py`
     (one entry: name, menu, shortcut, tip; the window binds `do_<name>`), which
     builds the menus, the toolbar and the shortcut sheet (F10), and reachable with
     the mouse too. `tests/test_dashboard_actions.py` fails on a shortcut used twice.
     A tab's own actions (Stage 4) take `scope="<Tab>"` and the tab's group
     (`actions.DATA_TAB`, `REGIONS_TAB`): the window creates them on the tab with
     `WidgetWithChildrenShortcut`, so a single key acts only while the tab has the
     focus (never while typing in the `.mod` panel), binds `do_<what>` on the tab,
     and calls the tab's `bind_action(name, action)`, which places a button (or a
     `KeyHolder` for a key) in `tab.tab_buttons`.
   - A tab that is expensive to draw refreshes only while it is shown: keep
     `self._stale` in `refresh` when `self.main.tabs.currentWidget() is not self`,
     and redraw on `tabs.currentChanged` (see `qt/regions.py`).

5. Document the component's actions and keys in its class docstring, with
   "Generated by RJC and Claude." and inputs/outputs in every method's docstring.

6. Test it: the logic with plain pytest (`-m unit`), and the widget with pytest-qt
   (`qtbot`), marked `gui` (`tests/conftest.py` runs it off-screen, and skips it
   where the gui extra or pytest-qt is missing), using
   `tests/dashboard_qt_helpers.make_window`. Check that one undo restores the
   project, that `blinding.strings(project)` and the `.mod` panel hold no hidden
   value after the interaction, and that `design_references(ui_strings(window))` is
   empty.

## The spectrum view (`qt/plots.py`, Stage 4)

Every spectrum the dashboard draws is a `plots.SpectrumView` (a pyqtgraph
`PlotWidget`), with a `plots.ZoomBar` attached to one or more views:

```python
view = PL.SpectrumView()
view.set_data(wave, flux, error, bad)   # steps centred on the pixels, broken at gaps
view.set_context(wave, flux)            # grey steps around a snip (its source)
view.set_continuum(wave, curve)         # dashed; set_normalised(curve) divides by it
view.set_regions("fit", spans, movable=True)   # also "other", "shared" (hatched)
view.set_edges(low, high)               # the snip's orange handles
view.set_knots([(x, y), ...])           # draggable squares
view.set_line_ids([(wave, label, kind), ...], hidden)   # kind: hydrogen/metal/other
view.set_velocity_centre(wave)          # the velocity axis above
view.set_labels(top=False, bottom=False)  # stacked views name their axes once
view.set_home((x0, x1)); view.home()    # ⌂; the flux axis is robust to wild pixels
view.set_mode(PL.DRAW)                  # PAN, ZOOM, DRAW, MASK or CLICK
zoom = PL.ZoomBar(); zoom.attach(view, *others)   # x axes linked to the first
```

Its signals carry wavelengths: `clicked(x, y)`, `spanned(mode, low, high)` (a drag in
DRAW or MASK mode, or a click in MASK), `edgeMoved(side, x, done)`, `regionMoved(n,
low, high)`, `knotMoved(n, x, y)` and `contextRequested(x, y, screen_pos)`. The view
shows no parameter value, so it needs no gate; the wheel zooms the wavelength axis
only. pyqtgraph's `stepMode` cannot clip or downsample, so steps are built with
`plots.step_curve`; keep the view's `clipToView` and downsampling for large spectra.

## Models, ticks and drags (Stage 5)

The model drawn is always ALIS's own: `profile.of(project)` (cached on the project's
state) gives a `Profile` whose `evaluate(values=None, snips=None)` returns a `Curve`
per snip (`model`, `continuum`, `normalised`, `residuals`, `chi2`), equal to
`model_eval.model_func`. During a drag, give it values instead of changing the
text: `profile.values({alis_index: value})` applies the links as ALIS does, and
`evaluate(values, snips=[...])` evaluates only the snips shown.

```python
view.set_model(x, model)                # the model, divided by the view's continuum
resid = PL.ResidualView(); resid.set_residuals(x, values)   # ±1σ lines, ±4σ clip
view.set_ticks([{"x": v, "label": "a1", "kind": "own", "key": key,
                 "selected": False}, ...])   # kinds: TICK_STYLE; not dragged
view.set_edits(True)   # a left drag gives dragged(x0, y0, x, y, state); right pans
```

Signals: `tickClicked(key)` and `dragged(x0, y0, x, y, state)` (``"start"``,
``"move"``, ``"finish"``). A drag is values while it moves (`components.drag`
bounds them; `components.draggable` says what no link computes) and one step when
it ends (`components.drag_step`); do not refresh the tab while a drag is held.
Keys held during a drag (N, B) need an application event filter
(`qt/components.DragKeys`): Qt reports only modifiers' states. Values
a link computes are shown as ALIS computes them (`components.linked_values`,
`components.shown`), through the gate. Cross-highlighting from a tab whose keys
need its focus: `window.panel.select_items(items, move=False)` highlights and
scrolls without moving the `.mod` cursor; `window.panel.cursor_line()` and
`components.at_line(project, line)` say what the cursor's line describes.

## Notes

- `alis/prepfit/specplot.py` is the old matplotlib GUI; do not add to it (D4).
- Do not change existing components or key bindings without explicit instruction.
