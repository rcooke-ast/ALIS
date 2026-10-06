# Dashboard Stage 3 log

The Qt skeleton of the chosen layout: the window, the five tabs and their markers, the
`.mod` panel, sessions (open, save, autosave), the launcher, preferences, relinking
spectra, and the blinding and export dialogs. The plan is in
`claude_prompts/dashboard_stage3.md`.

### 2026-10-06 (Prompt 1: queries, and Task 3.1)

**Reading.** The code plan, the dashboard prompts (D1–D44, the QF queries), the Stage
0–2 documents and logs, and the code Stage 3 builds on: `alis/dashboard/` (every
module), `alis/bundle.py` (reading, writing under the lock, `update`, sources,
`extract`, `unblind`, `run` and `run.json`), `alis/scripts/run_alis.py`, the
`gui-dev` and `gui-component` skills, the CI workflow, and the mockups' builder
(`window`, `mod_panel`, the panes of every tab).

**Queries.** RJC's responses to Q3.1–Q3.11 were read. Two changed the design: a
spectrum is not opened from the command line (Q3.7), and chosen lines can be blinded
as well as the whole fit (Q3.9). Four new queries were asked in the session, and
recorded in the document with RJC's answers:
- **Q3.12:** one New project dialog (name and folder, written at once; mode; one or
  more spectra; an optional primary z; global blind). A spectrum on the command line
  is refused with a message. As proposed.
- **Q3.13:** Model → Blind lines… (a dialog of lines by system, component and ion,
  with no values) and "Blind this line" in the `.mod` panel's menu; undo stops at
  that step. As proposed.
- **Q3.14:** an edit that would show a hidden value is refused, pointing to
  Unblind…; deleting a hidden line whole is allowed; undo stops at any step that
  blinds something. As proposed.
- **Q3.15:** RJC renamed the launcher: `alis` opens the dashboard
  (`alis project.model`), and `run_alis` fits. Claude runs the editable install, and
  removes "(D10)" from `bundle.extract`'s note.

The note on Q3.2: autosave writes the recovery copy every 60 s while there are changes
not yet kept (RJC asked for "every minute or so"). The Design section and the tasks were
updated to follow all of these.

**Baseline.** `pytest -m unit`: 1428 passed, 120 skipped, 0 failed (78 s).

**Environment.** PySide6 6.11.2, pyqtgraph 0.14.0, qtpy 2.4.3 and pytest-qt 4.5.0 are
installed (RJC installed PySide6 and pyqtgraph). PyQt6 6.10 is also installed; qtpy
takes PySide6 because `QT_API` says so. Off-screen rendering (`QT_QPA_PLATFORM=
offscreen`) draws text, including ✓ ↻ ! ○ and ▒, and `widget.grab()` saves it.

**3.1 The Qt foundation.**
- `alis/dashboard/qt/__init__.py`: the package docstring says it is the only part of
  ALIS that imports Qt, always through `qtpy`, and sets `QT_API=pyside6` unless the
  user has set it.
- `alis/scripts/dashboard.py`: the `alis` command (`[project.scripts]` in
  `pyproject.toml`), with `--help`. It checks the gui extra first (qtpy, a binding,
  pyqtgraph): without it, it prints `pip install "alis[gui]"` and returns 1. Then it
  asks `session.launch_target` what the path is (Task 3.4), and starts the
  application (`qt/app.py`).
- **pytest-qt is not in `dev`, but in a new extra, `gui-test`.** pytest-qt stops
  pytest from starting at all when no Qt binding is installed ("pytest-qt requires
  either PySide6, PyQt5 or PyQt6 installed", raised in its `pytest_configure`), so in
  `dev` it would break a plain `.[dev]` install and the CI `unit` and `examples` jobs,
  which have no Qt. The window tests are run with `pip install -e ".[gui,gui-test,dev]"`.
- `pytest.ini`: a `gui` marker. `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`
  and `QT_API=pyside6` (unless set), and gives pytest-qt the same binding through
  `PYTEST_QT_API`, before pytest-qt chooses one. The `gui` tests are skipped where the
  gui extra or pytest-qt is missing.
- CI: a `gui` job (Ubuntu and macOS) installs `.[gui,gui-test,dev]`, with the system
  libraries Qt needs on Ubuntu, and runs `pytest -m gui` off-screen. The `unit` job
  stays free of Qt.
- The editable install was run again (`pip install -e ".[gui,gui-test,dev]"`;
  nothing new was downloaded), so `alis` is installed. RJC's shell has an old alias,
  `alis` → `python /Users/rcooke/Software/ALIS/src/alis.py`, which hides it in an
  interactive shell until it is removed.
- Tests (`tests/test_dashboard_qt_foundation.py`, 10, `unit`): only `qt/` imports
  qtpy, a binding or pyqtgraph (an AST scan of every module of `alis/`); `qt/` imports
  no binding directly; `QT_API` defaults to `pyside6` unless set; `alis --help`, by
  the module and by the installed command; and with qtpy, the bindings and pyqtgraph
  hidden from a fresh interpreter, `alis` and `alis project.model` print the install
  message and exit with 1, without a traceback. `test_dashboard_no_qt.py` still
  passes (14 pass together).

**3.2 Preferences (`preferences.py`).**
- `home()` is `$ALIS_HOME` or `~/.alis` (Q3.3), holding `dashboard.json`,
  `recent.json` and `recovery/`.
- `SPECS`: each preference's key (`section.name`), default, type, range, title, unit
  and help (for the dialog): the Components tab's columns (3) and velocity range
  (±200 km/s, D37); the autosave interval (60 s, RJC's Q3.2); the typing pause (1 s),
  the reading delay (0.3 s) and the full check's delay (2 s) (Q3.5); a new dataset's
  FWHM (7 km/s) and a new model's settings (`run ncpus -1` and the rest of the list
  Stage 2's template used, RJC's Q2.2); and the length of the recent list (10).
- `Preferences.load()` overrides the defaults with the user's file. A value the
  preference cannot take (out of range, not a whole number, not a number, a setting
  ALIS does not know or whose value ALIS's own `set_params` refuses) falls back to the
  default; an unknown key is ignored; a file that is not JSON gives the defaults. Each
  gives a message, kept in `messages` and sent as an ALIS warning. `set` checks as
  `load` does; `save` writes only what differs from the defaults, atomically.
- `Recent`: newest first, capped, kept in `recent.json` apart from the preferences
  (the dashboard writes it, the user does not); a damaged file is an empty list.
- Tests (`tests/test_dashboard_preferences.py`, 18, `unit`, in a temporary
  `$ALIS_HOME`): the folder; every default; the defaults are values their preferences
  take; overrides; eight kinds of bad value, each falling back with a message naming
  the key; unknown keys; an unreadable file; set, save (only the changes) and load
  again; the recent list (capped, reordered, removed, kept apart, damaged).
- The `unit` batch: 1456 passed, 0 failed.

**3.3 Sessions (`session.py`).**
- `Session(project, path, prefs, folder, origin)` holds the project, its `History`
  (with the typing pause of the preferences), the path (None for an unsaved import),
  the view state (`ui/view.json`, over the defaults: the Data tab, the `.mod` panel
  open at 460 px), and what the bundle on disk held when it was read or written.
- **Unsaved changes** are found by a fingerprint of what Save writes (the text, the
  files, `ui/project.json`, the extra hidden lines, the sources), so undoing back to
  the saved state is clean again. An import is unsaved until Save as. The view is not
  part of it: a clean close writes a changed view into the bundle by itself (only that
  member, under the lock).
- **The bundle on disk** (`check_disk`): `same`; `runs` when only `runs/` changed (a
  `run_alis project.model` finished, Q3.8(a)), which `take_runs` brings into the
  project so the Fit tab can show it; `changed` when the model, its files, the project
  data or the sources changed (Q3.8(b)); `missing`; `unsaved`. A cheap check of the
  file's time and size comes first.
- **Save** to the project's own file goes through `bundle.update` (re-read under the
  lock, replace only what the project owns: the files tree, the model's hidden lines,
  `ui/project.json`, `ui/view.json`, the sources), so the runs written meanwhile are
  kept (Q1.7). A project changed on disk raises `Conflict` unless `overwrite` is given;
  the window asks overwrite, reload or save as (Q3.8). Save as, or a deleted file,
  writes a new bundle. Saving adds the project to the recent list.
- **Autosave** writes a recovery copy (a bundle, with a small JSON note: path, name,
  origin, time) to `recovery/`, named after the project's path, or after the session
  for an unsaved import. It writes only when there are changes not yet kept, and
  removes the copy when there are none. `find_recovery(path)` offers a copy newer than
  the bundle (an older one is removed); `Session.open(path, recover=note)` opens the
  copy's project over the disk's runs, with unsaved changes. `orphans()` lists the
  copies of unsaved imports for the start page, and `from_recovery` reopens one.
  Saving, and a clean close, remove the copy.
- Tests (`tests/test_dashboard_session.py`, 15, `unit`, on a bundle of
  `examples/metal_line_abs` in a temporary folder): saving and reopening gives back the
  text, the files, `ui/project.json` and `ui/view.json` byte for byte (after a value
  edit, a region edit that changes a snip's file, a confirmed structure and a view
  change), and saving again unchanged gives the same members; undo back to the saved
  state is clean; after an edit a recovery copy exists, the bundle is unchanged,
  reopening "after a crash" offers the copy and gives the edited text, and a clean
  close removes it; an out-of-date copy is not offered; saving removes the copy; Save
  keeps a run written meanwhile (only `runs/` changed, so no conflict); the runs are
  taken in; a conflicting save from another session is detected, then overwritten, or
  reloaded (history cleared); a deleted bundle; an unsaved import saved with Save as
  (and added to the recent list); an unsaved import recovered from the start page;
  Save as moves the recovery copy and keeps the runs; close keeps the view; the typing
  pause comes from the preferences.
- `tests/test_dashboard_session_run.py` (`fast`, `examples`): with an edit not yet
  saved, `run_alis project.model -p 0` fits the saved bundle in another process;
  `check_disk` says `runs`; Save keeps the run and writes the edit; and the reopened
  project's model has changed since that run (8 s).

**Found while building the `.mod` panel (Stage 2's gate), and fixed.** The blinding
gate masked only what the parsed model knew. Two leaks followed:
- **a text that does not read** has no parameters, so `Gate.view()` showed the values
  of a `blind=True` line in plain text (`ion=28Si_II 13.0 0.0 1.0da 8000TA` in
  `examples/blind`, once `model end` was mistyped). The `.mod` panel works while the
  text does not read (QF.3), so this would have been on screen;
- **a commented-out line with `blind=True`** is hidden in the bundle
  (`bundle.blind_line_numbers` finds it), but its values were shown, since a comment
  has no parameters.

The gate now also masks every value of every hidden line from its words
(`blinding.line_value_spans`), whatever the state of the text, and `blindrange` and
`blindseed` on every line; while the text does not read, it masks every value that
carries a label hidden when the text last read (`Project.last_read`, kept by
`_rebuild`). Overlapping spans are joined. Tests in `test_dashboard_blinding.py` (3
new; 12 pass): a commented-out hidden line, values on a hidden line and the blind
range while the text does not read, and a hidden label used on another line and in a
`lim param`.

**3.4 The launcher and new projects.**
- `session.launch_target(path)`: nothing gives the start page; a `.model` (a zip) that
  project; a `.mod` or `.mod.out` an import; anything else is refused with "… is not a
  project (.model) or a fit (.mod). To start a project from a spectrum, run alis and
  choose New project." (RJC, Q3.7). A missing file, a folder and a `.model` that is not
  a zip are refused too. `alis` prints the message and exits with status 2.
- `modes.VoigtMode.empty_project(sources, name, z, blind, settings, fwhm, columns,
  registry, bundle_path)`: a model of the settings and empty data and model blocks
  (`run blind True` when asked); the default atomic table; each spectrum read by D13's
  rule (a continuum column multiplied in, with a note), recorded by path, path
  relative to the bundle, and checksum, and made one file row of `ui/project.json`
  (the first the reference, each with the FWHM of the preferences); the primary system
  when a redshift is given. `VoigtMode.SETTINGS` is now the preferences' default list.
- `Session.new(path, sources, z, blind, mode, prefs, …)` makes it and writes it at
  once (Q3.12); it refuses an existing file unless told, and a name not ending in
  `.model`.
- Two small changes to Stage 2's `project.py`, for a new project: a file row of
  `ui/project.json` with a source but no snips yet is kept (it was dropped), with no
  zero level; and `inferred` now means that `ui/project.json` does not list the systems
  or rows, rather than that the lists are empty (a new project with no redshift has no
  systems, and is not inferred).
- The New project dialog (`qt/dialogs.py`): name and folder (showing the file it will
  write), mode (Orders listed, disabled), the spectra (each with a summary: columns,
  pixels, wavelength range, any note), the primary z (optional), and global blind. It
  says what is missing before Create is enabled. The window asks before replacing an
  existing file.
- `qt/app.py`: `make_application` (Fusion and the look), `open_window`, and `run`,
  which the `alis` command calls.
- Tests: `tests/test_dashboard_launch.py` (10, `unit`): the dispatch of each kind of
  path, the command's status 2 for a spectrum, a new project from two spectra written
  at once and read back (rows, reference, sources and their relative paths, columns,
  FWHM, the system, no notices, the settings, empty blocks, only the "no data lines
  yet" warning, sources present), the preferences' settings and FWHM and global
  blind, a continuum column's note, refusals, and a round trip through the bundle.
  `tests/test_dashboard_qt_window.py` (`gui`): the window opened for nothing, a
  `.model`, an imported J1358p6522 and Q1243p307 (unsaved, with the "imported"
  banner), and a new project made through the New project dialog (on the Data tab,
  with the continuum note told), and the dialog's own checks.

**3.5 The window.**
- `qt/style.py`: Fusion, a light palette, the system's sans-serif at 13 px and a fixed
  font at 12 px for the `.mod` panel (the first installed of a list, so that Qt does
  not search for a missing family), the Okabe–Ito colours, one style sheet, and the
  icons: a round marker per state (✓ blue, ↻ orange, ! vermillion, ○ hollow grey) and
  a problem icon per severity (✕, !).
- `qt/widgets.py`: a titled `Pane` (a placeholder line "Not built yet: …" when it has
  no body), a `Banner` with buttons, the blinding `Pill`, the `Strip` of the collapsed
  panel (its label written upwards), and rows and columns with the mockups' widths as
  stretch factors.
- `qt/tabs.py`: the five tabs as frames of the agreed layout, with the mockups' panes
  and proportions: Data (Systems, Coverage when there are several files, Blinding |
  Datasets, Spectrum), Regions (Transitions, Key | Transition, then Continuum and
  Snip), Components (Ions | Panels | Components), Fit with Inspect (Run, Progress |
  View, All snips), Results (Results | Correlations | Fit statistics, Run history) and
  Compare (Compare | Parameters, View), and Plot (Layout | Preview | Panels), shown but
  disabled with the tooltip "Not yet available".
- `qt/window.py`: the main window. The menus, the toolbar (Undo, Redo | Open, Save,
  the autosave time | Mode, the pill | Export, Shortcuts) and the shortcut sheet are
  built from `actions.py` (Task 3.7); the tabs with their markers (3.8); three banners
  (the text does not read; the file changed on disk; an imported fit's notice); the
  `.mod` panel in a splitter, collapsing to the strip; the status bar (file, "No fit
  running", the model's state, the last autosave); the start page (New project, Open,
  Import a fit, the recent projects, unsaved work to recover, and any message about
  the preferences file). Every question and file choice goes through `ask`, `choose`
  and `run_dialog`, which the tests answer. Every action reads what was typed first,
  and an unexpected error is reported with its traceback on the terminal (Q2.5).
  Timers: autosave, and a look at the bundle on disk every 2 s.
- **Screenshots** (`doc/dashboard/skeleton/take_screenshots.py`, which writes PNGs to
  a folder given; the images are not committed): the start page; J1358p6522 imported
  with its D I line blinded, every tab and Fit sub-tab, the panel open and closed, and
  the Unblind dialog; Q1243p307, every tab; `examples/blind`; an empty project, every
  tab; the New project dialog. **Compared by eye with the mockups:** the menus, the
  toolbar (with "Blinded: D I" dark, as the mockups' pill), the five tabs with icon
  markers, the `.mod` panel with line numbers, coloured lines, masks, marks beside the
  lines with problems and the list below, the status bar, and each tab's panes in the
  mockups' columns (Coverage appears for Q1243p307's three datasets) all match. The
  differences: the panes hold placeholders (as planned); the panel is open on every
  tab by default (the mockups close it on some pages); the Plot tab is disabled;
  the window's title bar is not drawn by an off-screen grab.
- Fixed while looking at them: label colouring caught digits inside file names
  (`J1358p6522`) and ion names (`ion=1H_I`), now only at the start of a word; the
  panel's footer kept the previous project's line; ALIS's notes that a snip has no
  continuum, zero-level or systematics column (24 of J1358p6522's 29 "problems") are
  no longer listed as problems (`validate.IGNORED`); the outputs packed with an
  imported fit, which have no record of the model that made them, no longer mark the
  Fit tab out of date; the New project dialog's help lines wrap.
- Tests (`tests/test_dashboard_qt_window.py`, `gui`, 13 pass): the window opens for
  J1358p6522, Q1243p307, `examples/blind` and an empty project, each with the five tabs,
  the Fit sub-tabs, the menus, the panel and the status bar; the empty project's
  markers; collapsing the panel and keeping the view in the bundle; and **no design
  reference** in any widget's text, tooltip, status tip, action, menu, tab or list
  item, on every tab of the three projects, nor in any of the six dialogs (a pattern
  for D/F/S numbers, QF.n and Qn.m; the F1 key of the shortcut sheet is a key, and
  left out).

**3.6 The `.mod` panel.**
- **Typing, without Qt** (`alis/dashboard/livetext.py`): reading the model at every
  key costs 0.03–0.4 s (Q3.5), so the panel keeps the shown (masked) text and the real
  text in step itself between readings. `diff(old, new, cursor)` finds the one span an
  edit replaced (the cursor decides where repeated characters make it ambiguous);
  `LiveText.edit` turns it into a patch of the real text and moves the masks with the
  text around them. An edit that cuts into a mask returns a `MaskEdit` instead, and
  the panel asks for the value in a small box (QF.20(c)); one that replaces whole masks
  (deleting a hidden line, or retyping it) is made as it is, since it shows nothing; a
  key just before or after a mask does not touch it, so a hidden value's label can be
  edited. After the pause, `pending()` gives every patch as one change, which goes to
  `History.type` (grouped by pause, F2); the project reads the text again; the panel
  is redrawn from the gate, keeping the cursor's line and column. Qt's editor shows
  `\r\n` as one newline, so a CRLF model maps each line ending as a segment too (none
  of the repository's models is CRLF, but `text.py` reads them).
- **The widget** (`qt/modpanel.py`): the editor (its own undo off, and its Undo/Redo
  keys left to the window's actions), a gutter with line numbers and a ✕ or ! beside
  each line with a problem, a highlighter by kind of line (comments, block markers,
  settings, data lines, functions, `fix`/`lim`, labels, masks), the current line and
  the selected item's lines highlighted, the list of problems (click to go to the
  line), and a footer saying what the cursor's line describes (no values). The quick
  check runs after each reading; the full check after 2 s, in a `QThreadPool` worker,
  and its result is dropped when its generation is not the panel's (typing, or
  another reading, moves it on). Typing is read when it pauses (0.3 s), and before
  any action of the window, so Undo, Save and autosave always see it. A step made
  elsewhere (a panel of Stages 4–6, undo, redo) redraws the panel (`sync`, when the
  text or the hidden lines differ from what it drew). "Blind this line" is in its
  context menu. "◂ Hide" collapses it to the strip.
- **Found:** a step made by a panel did not redraw the `.mod` panel, and Undo was
  disabled until the first typing was read. Both fixed (`ModPanel.sync`, called by the
  window's refresh; a `typing` signal enables Undo at the first key).
- Tests:
  - `tests/test_dashboard_livetext.py` (9, `unit`): `diff`, with the cursor deciding;
    typing outside a mask patches the right place of the real text; a key beside a
    mask does not touch it; an edit inside a mask asks for the value, which replaces
    the whole value and stays hidden once read; deleting or retyping a whole hidden
    line; CRLF line endings; offsets; and a hypothesis property test that any typing
    between the masks keeps every hidden value exactly, the shown text always being
    the real text with its hidden values masked.
  - `tests/test_dashboard_qt_modpanel.py` (11, `gui`): typing changes the project's
    text exactly as typed and one undo restores it (and redo); the editor's own undo is
    off; with `examples/blind` open (its hidden Si II line given 13.4729), the
    panel's document never holds the hidden value, after typing, undo, redo and a
    check, nor does the list of problems, nor the clipboard after copying all; typing
    over a mask asks for the value, stores it, keeps it hidden, and one undo restores
    the old one; a mistyped function name puts an error mark on its line and in the
    list, pauses the tabs with the banner, and typing it back clears both; a full
    check made stale by more typing is dropped, and shown when left alone; the cursor
    keeps its line and column when the panel is redrawn; typing `False` over
    `blind=True` is refused (the text and the masks unchanged, the user told to use
    Unblind…); deleting a hidden line is allowed and undone; cross-highlighting (the
    cursor's line gives its component; a component selected in a tab highlights its
    lines); "Blind this line" blinds the line, which undo does not pass.

**3.7 Actions, undo and redo, and the shortcut sheet.**
- `alis/dashboard/actions.py` (no Qt): one registry of 32 actions (name, menu entry,
  menu, shortcut, tooltip, submenu, toolbar, "later", section). The window makes one
  `QAction` per entry, bound to `do_<name>`, and builds the menus (with separators
  between sections, the Mode submenu, and Open recent after Open), the toolbar and the
  shortcut sheet from it. File: New project, Open, Open recent, Import a fit, Save,
  Save as, Export plain files, Relink spectra, Preferences, Close project, Quit. Edit:
  Undo and Redo (with what they undo), Cut, Copy, Paste, Select all. View: the five
  tabs (Ctrl+1–5) and the `.mod` panel (Ctrl+Shift+M). Model: Check the model now
  (Ctrl+K), Blind the analysis, Blind lines, Unblind, and Mode (Voigt; Orders
  disabled). Fit: Run (Ctrl+R) and Commit run, shown but disabled until Stage 6. Help:
  Keyboard shortcuts (F1), About ALIS. Preferences, Quit and About take their macOS
  menu roles.
- Undo and redo read what was typed first. Undo that would pass a step that hid
  something stops, with a message in the status bar.
- Tests: `tests/test_dashboard_actions.py` (5, `unit`): every action has a menu, a
  text and a tip; no two share a shortcut; the sheet lists every action with its
  shortcut; the menus and the toolbar of the design. `tests/test_dashboard_qt_actions.py`
  (6, `gui`): every QAction is in its menu, and in the shortcut sheet with its shortcut
  as this platform shows it (also in its tooltip); no two QActions share a key
  sequence; the toolbar; undo and redo across typing in the panel, a step made by a
  panel (an edit through the history, which also redraws the panel), and more typing,
  undone to the start and redone to the end; Undo reads typing not yet read; the View
  actions show the tabs.
- **Found:** ALIS's default is `run blind True` (`config.py`), so commenting out a
  model's `run blind False` blinds the analysis. The guard rightly makes that a step
  undo does not pass. But Stage 2's `blinding.unblind` only changed an existing
  `run blind` line, so a model without one would have stayed blind after unblinding;
  it now writes `run blind False` when no line sets it.

**3.8 Tab markers.**
- `alis/dashboard/markers.py` (no Qt): `markers(project, path)` gives each tab a
  `Marker(state, tip)` by the table of the Design section. Data: ○ no file rows; ! a
  source missing or changed (from `sources.py`, Task 3.9), or the structure inferred
  and not yet confirmed; ✓ "n file rows, m snips". Regions: ○ no snips; ! a snip with no
  fitted pixels, or pixels fitted twice (`load.find_shared_pixels` on each snip's
  fitted pixels, naming up to three pairs); ✓. Components: ○ no components or
  absorbers; ! an untied or isotope notice; ↻ the regions changed after the components
  were last edited; ✓. Fit: ○ no run; ! the last run stopped with an error
  (`runs/last_error.json` newer than the latest run); ↻ the model changed since the
  latest run; ✓ with the status, χ², time and host. Plot: ○ "Not yet available".
- The outputs packed with an imported fit have no record of the model that made them;
  they give ✓ ("found beside the model when it was imported"), not ↻.
- **The fingerprints** (`ui/project.json`, `marks`): one of the components (the lines
  of the absorption section) and one of the regions (each snip's file, fitted range
  and columns). `markers.Tracker`, which the session gives the history, records them
  as part of any step that edits the components, so undo restores them; a region edit
  on a project with no marks yet records the regions as they were before it.
- The window draws each marker as the tab's icon, with its tip (through the gate) as
  the tab's tooltip.
- Tests: `tests/test_dashboard_markers.py` (10, `unit`): J1358p6522 (Data ! inferred;
  Regions ! "Pixels are fitted twice: H1 and H2…"; Components ! its notices; Fit ✓
  imported outputs; Plot ○), Q1243p307, a new project (only Data ✓), Data ! for a
  missing and for a changed source and ✓ again for an embedded copy, Regions ! for a
  snip with no fitted pixels and for pixels fitted twice, Components ↻ after a region
  edit (and ✓ after its undo, marks gone after the component edit's undo, ↻ for a
  region edit before any marks, ✓ after a component edit, kept through save and
  reopen), Fit ↻ after an edit once a run exists (✓ after its undo), Fit ! for a
  later error (not an earlier one), and every marker's icon and tip.
  `tests/test_dashboard_qt_markers.py` (3, `gui`): the tabs show the icon of each state
  and its tooltip, which follow an edit and its undo; J1358p6522's Regions tab is
  marked for its shared pixels; every state has its own icon.

**3.9 Relinking moved spectra.**
- `alis/dashboard/sources.py` (no Qt): `states(bundle, path)` reports each source as
  present, missing, changed, or **embedded** (not on disk, but a copy is in the bundle:
  never missing; `bundle.check_sources` reported such a source as missing, so the
  dashboard does not use it). A source is looked for beside the bundle first, at its
  path relative to it, then at its absolute path, so a project moved with its spectra
  still finds them. Checksums are cached by path, time and size. `relink` accepts a
  file only if its checksum matches; a changed file is reported, never accepted.
  `describe` gives one line per state.
- On opening, a project with a missing or changed source asks "Relink them now?";
  File → Relink spectra… opens the same dialog: one row per source missing or changed,
  each with "Locate…". A relinked source makes the project unsaved (Save keeps its new
  place); the Data tab's marker follows.
- Tests: `tests/test_dashboard_sources.py` (6, `unit`): present; a moved source
  relinked when its checksum matches (saved and reopened at its new place); a different
  file refused, and a source changed in place reported and refused; an embedded source
  never missing; a project moved with its spectra finds them beside it; checksums read
  again when a file changes. `tests/test_dashboard_qt_relink.py` (2, `gui`): opening a
  project whose spectrum has moved offers to relink; the dialog refuses a different
  file with the reason and accepts the moved one; the Data marker becomes ✓; Save keeps
  it; File → Relink with nothing to relink says so.
- Fixed: the dialog's "every source is present" line overwrote the message about the
  relink just made.

**3.10 Blinding and export.**
- **The guard** (`blinding.Guard`, given to the history by the session): before a step,
  the real-text spans of the hidden values and of every shown value, and the global
  blind; after it, each span followed through the change. A hidden value still in the
  text and no longer masked, or global blind switched off, is a **reveal**: the history
  undoes the step and raises `RevealError` ("This change would show values that are
  hidden… choose Model → Unblind…"). A shown value now masked, or global blind switched
  on, **hides** something: the history's `floor` moves above the step, so undo stops
  there (`can_undo`, `undo_description`), and typing is not joined to it. A value
  deleted or typed over is not shown, so deleting a hidden line, or typing a new value
  over it, is allowed. It works whether the step was typed or made by a panel.
- **The pill** (`blinding.pill_text`): "Not blinded"; "Blinded: Si II" (the ions, or
  the labels or functions, of the hidden lines); "Blind analysis"; or both, joined.
- **Model → Blind the analysis…** asks (it cannot be undone), then writes
  `run blind True` with a new `edit.set_setting` (replaces the setting's value, or adds
  a line after the other settings in their layout). **Blind lines…** lists the model's
  lines by system, component and ion (no values), with tick boxes, and blinds those
  ticked as one step; "Blind this line" in the `.mod` panel asks first.
- **Unblind…** (`UnblindDialog`): it says how many hidden values on how many lines it
  will show (and the best-fit values under global blind), that it cannot be undone and
  clears the history, and that the project is saved; Unblind is enabled only with a
  note and the tick box. `Session.unblind(note)` unblinds in memory
  (`blinding.unblind`, which clears the history) and, for a saved project, at once in
  the bundle on disk (`bundle.unblind`, logged with the note, the runs' hidden lines
  put back) with the project saved; it refuses while the file has changed on disk.
- **Export plain files…** warns first when anything is blinded (the hidden starting
  values are written in plain text, hidden best-fit values are not), asks before
  overwriting, and says how many files it wrote, with `bundle.extract`'s notes.
  `bundle.extract`'s note no longer ends "(D10)" (Q3.15).
- Tests: `tests/test_dashboard_guard.py` (13, `unit`): global blind, set from a panel
  or typed, is a step undo does not pass, while later steps are undone down to it;
  blinding a line, from a panel or typed, likewise; typing `blind=False`, another
  value, or removing `blind=True` is refused, the text unchanged; `run blind False` is
  refused, while deleting the `run blind True` line is allowed (ALIS's default keeps
  the analysis blind); typing over a hidden value and deleting a hidden line are
  allowed, and undone with the value still hidden; unhiding an extra hidden line is
  refused; the pill; unblinding confirmed, logged in the bundle with its note, final,
  history cleared, the run's hidden line restored on disk; unblinding a model that
  relies on ALIS's default writes `run blind False`. `tests/test_dashboard_qt_blinding.py`
  (6, `gui`): the pill; Blind the analysis (cancelled, then done, then undo does not
  pass, then "blinded already"); Blind lines (the dialog lists lines with no values);
  the Unblind dialog refuses without both the note and the tick, then unblinds, logs,
  clears the history and shows the values; Export warns when blinded (cancelled:
  nothing written; then written, with the hidden starting value in plain text), not
  when unblinded, and asks before overwriting.
- **Found:** the Blind lines dialog's choice was asked again by the window; the dialog
  already says that blinding cannot be undone, so only "Blind this line" asks.

**Window tests of the session flows** (`tests/test_dashboard_qt_session.py`, 7, `gui`):
autosave and its label, and the recovery copy offered on opening ("Recover" gives the
edit back, unsaved); "Open the saved project" discards the copy; closing with unsaved
changes (Cancel keeps the project; Don't save closes it and removes the copy; Save
saves it); a conflict on Save (Cancel writes nothing; Reload gives the disk's version;
Overwrite writes this one); Save as for an import; a run finished outside the
dashboard (taken in, with a status message and the Fit marker) and a model changed by
another program (a banner with Reload); and the Preferences dialog (a bad setting
refused with its reason; the timers, the history's pause and `dashboard.json`
follow). `tests/test_dashboard_qt_app.py` (6, `gui`): the `alis` command's dispatch
in-process (status 2 for a missing file); the application starts on the Fusion style
and returns 0 when its window closes; `open_window` opens a project; the start page
lists the recent projects (a missing one marked) and unsaved work, which can be
recovered, and the Open recent menu; an unexpected error in an action is reported
with its traceback on the terminal; clicking a problem goes to its line.
- **Fixed:** an unexpected error in an action was reported with the validator's
  wording ("…while checking the model"); the window has its own message. The Fusion
  style is recorded on the application (`alis_style`), because a style sheet wraps it.

**3.11 Closing the stage.**
- **The review page**, published as a private page for RJC's comments:
  https://claude.ai/artifact/A6ohUEnAe91TEDsKBLwuBt. `doc/dashboard/skeleton/
  take_screenshots.py` takes the screenshots (the `.mod` panel as each mockup shows
  it: open only on J1358p6522's Regions tab), and `build_review.py` builds the page:
  each tab's screenshot beside its mockup's window (taken as it is from
  `doc/dashboard/mockups/*.html`, with the Stage 0 page's stylesheet, so the two pages
  read as a series), "What to check" and "Differences from the mockup" for each, the
  screens with no mockup (start page, New project, a new project, Unblind, the panel
  beside the Components tab), a layout switch (side by side, or one above the other),
  and what RJC is asked to decide (Q3.16). The page is 2 MB (the PNGs embedded).
- Fixed while making the screenshots: the collapsed strip's label was drawn past its
  24 px width.
- `doc/ALIS_workflow.md` (version 0.6): §4.3, the dashboard (`alis`, new projects,
  saving and autosave, the `.mod` panel, preferences, the markers); pixels fitted
  twice is now §4.4.
- `CHANGELOG.md`: the dashboard's window and the `alis` command, the new modules, the
  `gui-test` extra, the `gui` marker and CI job; the extract note. `tests/README.md`:
  the new test files and the window batch. The `gui-dev` and `gui-component` skills:
  the `alis` command, `ALIS_HOME`, the test helpers, the screenshot and review
  scripts, the guard of the history, `window._run`, `ask`/`choose`/`run_dialog`, the
  registry of actions, and the tabs' frames to fill.
- `dashboard_stage0.md`'s stage table, and F14 and D3 in
  `ALIS_v2_dashboard_prompts.md`: the command is `alis` (Q3.15). Nothing moved
  between stages.
- **Coverage** (the `test-coverage` skill, over the dashboard tests, `unit` and
  `gui`): 93% at first. The uncovered lines that mattered were tested rather than the
  number chased: the application's start and close (`qt/app.py`, 0%, only ever run in
  a subprocess), the command's dispatch in-process, the start page's lists and Open
  recent, an unexpected error in an action, and clicking a problem. Now 94% of 6166
  statements: `actions` 100%, `livetext` 99%, `blinding` 98%, `markers` 98%,
  `sources` 98%, `history` 97%, `preferences` 96%, `session` 95%; `qt/tabs` 100%,
  `qt/style` 99%, `qt/app` 96%, `qt/dialogs` 94%, `qt/modpanel` 92%, `qt/widgets` 92%,
  `qt/window` 89%; `scripts/dashboard` 81% (the rest runs in subprocesses). What is
  left is mostly the message boxes and file dialogs that the tests answer in their
  place, and error branches.
- The stage document: tasks marked done, the status, what Stage 4 receives, and
  Q3.16 (the review).
- **The batches at the close:** `unit` 1526 passed, 0 failed (63 s); `gui` 54 passed
  (about 25 s); `fast` 111 passed, 0 failed (13 min 36 s), including the new
  `test_dashboard_session_run.py`.
- **ALIS outside `alis/dashboard/`:** the `alis` command and its entry point, the
  `gui-test` extra, the `gui` marker, `tests/conftest.py`, the CI `gui` job, and the
  extract note. ALIS's fitting code is unchanged.

**Stage 3 is complete.** Q3.16 (the review of the skeleton) is open for RJC.

### 2026-10-06 (Prompt 2: draft 2 of the skeleton, after RJC's comments)

**The comments.** RJC left nine comments on the review page (recorded in the
document under Q3.16): the window is "overall looking excellent", with minor tweaks;
keep the `.mod` panel open by default; the mode should be chosen at New project and
not offered on the toolbar, and be called "QSO Abs Line" everywhere; line the `.mod`
values up in columns; let the user set the panel's width, and move it to a window of
its own and back; New project should ask which columns hold wave, flux and error (and
optional ones such as the continuum) for a text file, guessing first, with PypeIt
spec1d FITS files to come; swap Correlations and Run history in Fit · Results. Two
choices were asked in the session (Q3.17): alignment as an action, with new text
written aligned (RJC's choice); and the mode renamed in the code too, keeping the
`voigt` function, its `Voigt` class and its `_idstr` (RJC's note).

**Changes.**
- **The mode:** `modes.VoigtMode` → `QSOAbsLineMode` (`name = "qso_abs_line"`, `title
  = "QSO Abs Line"`); `modes.get()` reads the old name `"voigt"` too, through
  `ALIASES`, so projects made during development still open. `ui/project.json`,
  `Session.new` and the dialog use the new name. The toolbar's Mode button and the
  Model menu's Mode submenu are removed; the status bar shows "Mode: QSO Abs Line".
  The design documents say "QSO Abs Line mode" wherever they said "Voigt mode"
  (`ALIS_v2_dashboard_prompts.md`, `dashboard_stage0.md`, `dashboard_stage2.md`, this
  document, `CHANGELOG.md`, `tests/README.md`, the `gui-dev` skill); the `voigt`
  function and the Voigt profile are untouched. The logs and the Stage 0 mockups are
  left as they were, as a record. New decisions D45–D51 in
  `ALIS_v2_dashboard_prompts.md` collect what Stage 3 decided.
- **Alignment** (`alis/dashboard/align.py`, no Qt): groups of lines (the settings, the
  data lines, consecutive lines of one function in one section, the `fix`/`lim`
  commands), the k-th value of each starting in one column, then the keywords after
  the values together, each keyword in its own column when the lines give them in one
  order. Only spaces change; trailing comments are kept; comment lines inside a group
  are left alone. Model → Align columns (Ctrl+L) makes it one undoable step; an
  imported fit's banner offers it; both templates of `modes.py` write their models
  aligned. On a hidden line, the columns after a mask do not line up in the panel: a
  mask keeps one length, since its length could hint at the value (its sign).
- **The `.mod` panel in a dock** (`window.dock`): its width is set by dragging its
  edge; "⧉ Own window" (or View → The `.mod` panel in its own window, Ctrl+Shift+W)
  makes it a window of its own, with the window's actions added to it so its
  shortcuts work there; "↩ Back to the dashboard", or dragging it back, re-attaches
  it. Shown, width, own window and its place are kept in `ui/view.json`. The tabs'
  minimum widths were lowered, and the panel's state label allowed to shrink, so the
  panel can be made both wider and narrower.
- **New project's column roles:** `modes.file_kind` (text, FITS, or unknown),
  `modes.ROLES` (wavelength, flux, error, continuum, mask), `modes.check_roles` and
  `modes.preview` (no Qt); `qt/dialogs.ColumnRoles` (a menu above each column of the
  first rows). The roles start from D13's rule, can be changed, and are checked
  (wavelength, flux and error needed, one column each); a FITS file is listed as read
  by a later mode. `Session.new(columns=...)` passes them to `empty_project`.
- **Fit · Results:** Results | Run history | Fit statistics above Correlations.
- The review page (`build_review.py`): "Draft 2: what changed", new screenshots (the
  model after Align columns, the panel in its own window, New project with its
  columns, J1358p6522 as imported), and a new "To decide" list; published as version
  2 of the same page. Each of RJC's nine comment threads got a reply saying what
  changed, and was resolved.

**Tests.** `tests/test_dashboard_align.py` (203, `unit`): every model in `examples/`
and `context/` aligned keeps its words line by line and its line endings, and aligning
twice changes nothing; every `examples/` model reads the same to ALIS after aligning
(settings, data lines, every parameter's value, label and state); J1358p6522's
column densities, redshifts, b and T each in one column, its Legendre `specid=` in one
column, its settings' values in one column; keywords by name, comments kept; the
groups; Align is one undoable step that keeps hidden values hidden.
`tests/test_dashboard_qt_review.py` (8, `gui`): no Mode on the toolbar or in the menus,
"Mode: QSO Abs Line" in the status bar; the Results layout; Align columns (the banner's
button, the undo text, "aligned already"); the panel in its own window, typing there,
back, and remembered by the project; the panel's width; New project's guessed roles,
changed, checked; a FITS file refused for now; a new project keeps the roles chosen.
Updated: the mode's tests (and its old name), the aligned settings of a new project.

**Batches.** `unit` 1729 passed, 0 failed (79 s); `gui` 62 passed (23 s); the
dashboard's `fast` tests (the two fits of new projects, and `run_alis` while a project
is open) passed. black, isort and ruff are clean on the changed files.

**Q3.18** asks for RJC's second round of comments.

### 2026-10-06 (Prompt 3: draft 3 of the skeleton, after RJC's second round)

**The comments.** RJC answered Q3.18 on the review page (recorded in the document
under Q3.18): draft 2 does what was asked, but there is more to change before Stage 4;
no zero level or systematics in New project; Stage 3 not ready to close. The changes
asked for: the column roles (an Ignore role, every other role on one column only,
wavelength, flux and error required, and the mask a bad-pixel mask, not a fit range);
New project laid out again (Project Name, Project Folder, no Columns section, the
spectra in a table, "Add spectra (ascii)", "Add spectra (spec1d)", and Remove enabled
only with a spectrum selected; each spectrum in a dialog of its own with its file, FWHM
and columns); and Align columns only on the `.mod` panel, beside "Back to the
dashboard", not on the imported fit's banner. No question needed asking before the
draft.

**Changes.**
- **Column roles** (`modes.py`, no Qt): `check_roles` takes a list (None for Ignore)
  or a dict, and says what to fix ("Flux is given to more than one column; choose the
  error column.", "Choose the flux, error columns."); `roles_of` turns the choices
  into roles. A mask is read as a bool (`!= 0`), and `_cut` leaves the masked pixels
  out of `fit` even inside a fit region. `empty_project` takes a FWHM per spectrum.
  `ROLE_TITLES` names the mask "Mask"; `IGNORE = "Ignore"`.
- **New project** (`qt/dialogs.py`): `NewProjectDialog` has "Project Name", "Project
  Folder", the mode, and a table of the spectra (File, Wavelength range, Pixels, FWHM
  (km/s), Contains), with "Add spectra (ascii)…", "Add spectra (spec1d)…" (disabled,
  its tooltip saying a later mode reads spec1d files) and Remove (enabled only with a
  row selected); a double-click reopens a spectrum's dialog. `SpectrumDialog` ("Add a
  spectrum (ascii)") asks for the file, its FWHM and its columns (`ColumnRoles`, now
  with Ignore, and a `changed` signal), shows what is wrong, and keeps Add disabled
  until the roles are right; a FITS or unknown file is refused with a note.
  `spectrum_facts` gives a table row's range, pixels and contents. `values()` gives
  `columns` and `fwhms` per path; `Session.new(fwhm=...)` passes them on.
- **Align columns** (`actions.py`, `qt/window.py`, `qt/modpanel.py`): the action is
  `panel.align`, in a new group ".mod panel" (`actions.PANEL`; `GROUPS` is the menus
  and the panel, and the shortcut sheet lists both); it is out of the Model menu and
  off the banner (which offers only Hide). `ModPanel.set_panel_actions` puts its
  button in the panel's header just before the window button ("⧉ Own window" / "↩ Back
  to the dashboard"), and in the panel's right-click menu; Ctrl+L is kept.
- **The panel's header** (`qt/widgets.py`): `ElidedLabel`, for the panel's state, is
  cut with "…" (its tooltip gives the whole text) when the panel is narrow; the
  project's name keeps its width. Without it, the two labels ran into each other once
  the panel had been narrowed to its new minimum.
- The review page (`build_review.py`): "Draft 3: what changed", the New project and
  "Add a spectrum (ascii)" screenshots (`take_screenshots.py` adds two spectra, one
  with its own FWHM, and the spectrum dialog), the captions of Align columns, and a
  new "To decide" list (Q3.19). A note left on the page between the drafts (the
  column roles, "for draft 3") is folded into the draft 3 list. Published as version 4
  of the same page; each of RJC's six open threads got a reply and was resolved.
- The documents: Q3.18's responses and Q3.19 in this stage's document, its Design
  (New project, the `.mod` panel, the menus), D46 and D50 in
  `ALIS_v2_dashboard_prompts.md`, and `CHANGELOG.md`.

**Tests.** `test_dashboard_qt_review.py` (10, `gui`): Align columns on the panel only
(not on the banner, not in a menu, beside the window button, one undoable step,
"aligned already", and in its own window); the panel narrowed without its labels
running into each other; New project's labels, table, buttons and Remove; the
spectrum dialog (file, FWHM, the roles guessed, Ignore, a role given twice, the
required three, changed by a double-click); a spectrum with two columns refused; a
bad-pixel mask read as a bool; a new project keeps its roles and its FWHM.
`test_dashboard_actions.py`: the panel's group, and Align columns out of the menus.
Updated: the placed actions now count the panel's buttons; the D44 check of the
dialogs' text includes the spectrum dialog.

**Batches.** `unit` 1730 passed, 0 failed (64 s); `gui` 64 passed (24 s); the
dashboard's `fast` tests, 3 passed. black, isort and ruff are clean.

**Q3.19** asks whether draft 3 does what was asked, how to guess a 0/1 fourth column
(lean: keep guessing Mask), and whether Stage 3 may close.

### 2026-10-06 (Prompt 4: the last fixes, the stage closed, and Stage 4's document)

**The comments.** RJC left two on the page of draft 3 (recorded under Q3.19), and
Prompt 4 closes the stage once they are applied:
- "Guess Ignore. There should not be a fitrange loaded for these spectra. Snips have
  a fitrange, not the full spectrum." (Q3.19(b));
- "There appears to be two hide buttons here." (the Data tab of Q1243p307).

**Changes.**
- **The guess** (`modes.column_roles`): a fourth column of 0s and 1s is no longer
  given a role, so it is ignored; any other fourth column is still a continuum. A
  bad-pixel mask is chosen in the spectrum's dialog. D13 in
  `ALIS_v2_dashboard_prompts.md` records the refinement, and `CHANGELOG.md` the
  guess.
- **The second Hide** was a bug of `widgets.Banner.show_message`: the old buttons
  were taken out of the layout and only marked with `deleteLater`, so until the
  event loop ran, the previous project's Hide stayed drawn where it had been, over
  the text. The screenshots are taken one after another, without the event loop in
  between, and caught it; a user could have seen it for a moment. The old buttons are
  now hidden and detached at once. (Holding the widget before `setParent(None)`
  matters: the layout item no longer returns it afterwards.)
- The review page: "Draft 3, final", with "After draft 3: the last fixes" in place of
  "To decide", the screenshots retaken (one Hide on the banner), and the spectrum
  dialog's caption. Published as version 5; both threads got a reply and were
  resolved.
- The stage document: Q3.19's responses, the status (Stage 3 closed), and the points
  for Stage 4.

**Tests.** `test_dashboard_modes.py`: a fourth column of 0s and 1s gets no role.
`test_dashboard_qt_review.py`: the bad-pixel mask is now chosen in the dialog (the
guess is Ignore), and a new test, `test_the_banner_shows_only_its_own_buttons`, shows
two messages with no event loop between them and finds only the second's button.

**Batches.** `unit` 1730 passed, 0 failed (67 s); `gui` 65 passed (23 s); the
dashboard's `fast` tests, 3 passed. black, isort and ruff are clean.

**Stage 4's document** (`claude_prompts/dashboard_stage4.md`), written from
`ALIS_v2_code_plan.md`, `ALIS_v2_dashboard_prompts.md`, the stage table of
`dashboard_stage0.md`, the mockups of the Data and Regions tabs, and Stages 1–3. It
covers steps 1–3 of the workflow (S2, S3, S5–S10, S16, S24–S27, D12–D19, D35, D36):
- the design: one spectrum view on pyqtgraph for every tab; three modules without Qt
  (`lines.py`, `snips.py`, `continuum.py`); curves and buffers computed by ALIS's
  own functions, so the dashboard draws what ALIS fits; the Data tab (systems and
  Identify a feature, the datasets table with its badges and ties, coverage,
  blinding, confirming an imported fit's structure); the Regions tab (the ranked
  transition list with its flags, SNIP and CLEAR, regions, masks, the snip's edges,
  the continuum with its first guess, order, handles and sharing, and the pixels
  fitted twice with their two fixes);
- 13 tasks, among them a review of the Data tab halfway through (4.7) and an end to
  end test (4.12): a project built from `metal_line_abs`'s spectrum through the two
  tabs, fitted by `run_alis`, against the example's reference;
- 12 queries (Q4.1–Q4.12), each with a lean.

Facts checked while writing it: ALIS's Legendre takes its range from the loaded
pixels unless `min=`/`max=` are given, and stops at order 10; the buffer ALIS wants
comes from the resolution function's `getminmax` (already used by
`validate._buffer`); `load.find_shared_pixels` finds the mockup's 36 shared pixels of
J1358p6522's Ly7 (18 with Ly8, 18 with Ly6); only the Orders-mode models of
`DH_orders/Q1243p307` share a continuum between snips; pyqtgraph 0.14 is installed
with the `gui` extra.
