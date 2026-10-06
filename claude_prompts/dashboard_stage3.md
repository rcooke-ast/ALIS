# Prompt file for ALIS software dashboard creation -- STAGE 3

> **The Qt skeleton of the chosen layout.** The dashboard gets its window:
> - the five tabs of D34, with their status markers (F12);
> - the collapsible `.mod` panel, the menus, the toolbar and the status bar.
>
> It can:
> - open, save and autosave a project (F1);
> - import an existing fit (F4) and relink moved spectra (F13);
> - list its shortcuts (F10);
> - be launched with `alis` (F14; the command was named `run_alisgui` until Q3.15).
>
> The user's preferences file (Q2.2) comes in this stage too. The tabs are frames of
> the agreed layout, which Stages 4–6 fill in. The `.mod` panel is the first panel
> that works.
>
> Everything the window shows comes from Stage 2's project model, through the
> blinding gate, and every action is a step of the history. The windows live in
> `alis/dashboard/qt/`, the only part of ALIS that imports Qt (through `qtpy`) or
> pyqtgraph. Logic that needs no Qt goes beside the Stage 2 modules and is tested with
> pytest alone. ALIS outside `alis/dashboard/` does not change, unless a query agrees
> otherwise.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`. "Q0.*n*", "Q1.*n*" and "Q2.*n*" refer
> to `dashboard_stage0.md`, `dashboard_stage1.md` and `dashboard_stage2.md`. The plan
> for all stages is in `dashboard_stage0.md`.

## Design

*Written by Claude on 2026-10-06, from the design documents, the mockups and what
Stages 0–2 taught. The open choices are the Queries below; each gives Claude's lean, and
this section follows the leans. Updated in Prompt 1 to follow RJC's responses to
Q3.1–Q3.15: the launcher is `alis` (Q3.15), a new project is made in one dialog, not
from a spectrum on the command line (Q3.7, Q3.12), chosen lines can be blinded
(Q3.9, Q3.13), no edit shows a hidden value (Q3.14), and autosave runs every minute
(Q3.2). Updated in Prompt 2 (draft 2) to follow RJC's review of the skeleton (Q3.16,
Q3.17): the mode is "QSO Abs Line", chosen in New project and not on the toolbar;
New project asks for the role of each column of a text spectrum; the `.mod` panel can
be a window of its own; Model → Align columns; and the run history and the
correlations swap places in Fit · Results.*

### Two layers

```
alis/dashboard/qt/   the windows: Qt through qtpy, and pyqtgraph from Stage 4
  │                  thin: they draw the project, and turn the user's actions into steps
alis/dashboard/      the project model of Stage 2, and the modules of this stage that
                     need no Qt: preferences.py, session.py, markers.py (and, as
                     built, sources.py, actions.py and livetext.py)
```

- **No logic in the widgets.** What can be decided without Qt is decided in
  `alis/dashboard/` and tested there: what a tab marker says, what the launcher opens,
  when a recovery copy is offered, what a preference falls back to.
- **Every action is a step** of the project's `History` (F2): an edit made through
  `edit.py`, `project.py` or `remove.py`, or typing in the `.mod` panel. A widget never
  changes the text, a snip or `ui/project.json` itself.
- **Every value goes through the `Gate`** (F8). This covers values, words, best-fit
  values and messages, and the `.mod` panel shows `Gate.view()`.
- **The text is authoritative** (D7). While it does not read, the tabs show a banner,
  and their actions are disabled; the `.mod` panel always works (QF.3).

### The window (D34, D43)

```
┌ ALIS dashboard — J1358p6522.model ──────────────────────────────────────────────┐
│ File   Edit   View   Model   Fit   Help                                         │
│ ↶ Undo ↷ Redo │ Open  Save  autosaved 12:41 │ Blinded: D I │    Export plain files…  ? │
│ ✓ Data │ ✓ Regions │ ↻ Components │ ! Fit │ ○ Plot                               │
│ ┌──────────────── the tab ────────────────────────┐ ┌──── .mod panel ───────────┐ │
│ │                                                 │ │ J1358p6522.mod ⧉ ◂ Hide   │ │
│ │                                                 │ │  1 run  ncpus  -1         │ │
│ │                                                 │ │ 57   voigt ion=2H_I ▒▒▒▒ …│ │
│ │                                                 │ │ ! 3 problems              │ │
│ └─────────────────────────────────────────────────┘ └───────────────────────────┘ │
│ J1358p6522.model │ No fit running │ The model reads    Mode: QSO Abs Line │ 12:41 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- **Menus:**
  - **File:** New project, Open…, Open recent, Import a fit (.mod)…, Save, Save as…,
    Export plain files…, Relink spectra…, Preferences…, Quit.
  - **Edit:** Undo and Redo (with what they undo), and Cut, Copy and Paste in the
    `.mod` panel.
  - **View:** the five tabs (Ctrl+1 to Ctrl+5), Show the `.mod` panel, and the
    `.mod` panel in its own window.
  - **Model:** Check the model now, Blind the analysis, Blind lines…, Unblind…. The
    mode is not here: it is chosen in New project, and not changed afterwards (RJC's
    review). Align columns is on the `.mod` panel.
  - **Fit:** Run and Commit run, shown but disabled until Stage 6.
  - **Help:** Keyboard shortcuts, About ALIS.
- **Toolbar:** as in the mockups, without the mode (RJC's review). It holds Undo and
  Redo, Open and Save with the time of the last autosave, the blinding pill, Export,
  and the shortcut sheet.
- **Tabs:** Data, Regions, Components, Fit (with its sub-tabs Inspect, Results and
  Compare), and Plot. Each tab has a marker with an icon and a colour: ✓ complete,
  ↻ out of date, ! needs attention, ○ not started.
- **Status bar:** the project's file, the fit ("No fit running" until Stage 6), the
  state of the model ("The model reads", "3 problems", or "The model does not read:
  the panels are paused"), the mode ("Mode: QSO Abs Line"), and the last autosave.
- **Look (D43, Q3.10):** Qt's Fusion style with a light palette. The colours come from
  the Okabe–Ito palette, all in one module (`qt/style.py`), and every coloured state
  also has an icon. The window is judged at 1440×900. No text in it names a design
  document (D44).

### The tabs in this stage (Q3.4)

Each tab is a widget with the panes of its agreed layout (D35–D42), each pane titled,
holding a short line saying it is not built yet. Stages 4–6 replace the placeholders,
following the `gui-component` skill. The Fit tab has its three sub-tabs; in Results,
the run history is beside the results and the correlations are under the fit
statistics (RJC's review swapped them, from the mockup). The Plot tab is shown, but
disabled, with the tooltip "Not yet available": it is built after v1 (D42).

### The `.mod` panel (D5, D7, F5, F8, S16; Q3.5)

- **What it shows.** `Gate.view()`, so hidden values are masked. Each line has its
  number, and is coloured by its kind (`text.py`: comments, block markers, settings,
  data lines, functions, labels). The current line is highlighted, and so are the
  lines of the item selected in a tab (`project.lines_of`).
- **Typing.** The widget's own undo is switched off. Each edit is translated by
  `MaskedView.to_real` into a change of the real text and given to `History.type`, so
  typing is grouped by pause (F2). After a short pause the project reads the text
  again, and the panel is redrawn from the gate, keeping the cursor's line and column.
- **Problems (F5).** `validate.quick` runs after each reading. `validate.full` runs
  after a longer pause, in a worker thread (`alis_quietly` collects only its own
  thread's messages), and its result is dropped if the text has changed meanwhile.
  Problems are shown as marks beside their lines and in a list under the editor,
  through the gate. An unexpected error shows the message of Q2.5; its traceback is
  already on the terminal.
- **Cross-highlighting (S16).** Moving the cursor to a line tells the tabs what it
  describes (`project.items_at`); Stages 4–6 select it.
- **Collapsing.** "◂ Hide" collapses the panel to a strip at the right ("▸ .mod
  editor"). Whether it is open, and its width, are kept in `ui/view.json`.
- **Its width and its own window** (RJC's review). The panel is a dock at the right of
  the window: its width is set by dragging its edge. "⧉ Own window" (or View → The
  `.mod` panel in its own window) moves it to a window of its own, which still works
  with the dashboard (the same panel, the same project, the window's shortcuts);
  "↩ Back to the dashboard", or dragging it back, re-attaches it. Its own window, and
  where it is, are kept in `ui/view.json`.
- **Aligned columns** (RJC's review, Q3.17(a)). "Align columns", a button of the panel
  beside its window button, in its right-click menu and on Ctrl+L, and only there
  (draft 3), lines the model's values up in columns (`align.py`): in each group of
  lines (the settings, the data lines, consecutive lines of one function in one
  section, the `fix`/`lim` commands) the k-th value starts in one column, and the
  keywords after the values start together, each in its own column. Only spaces
  change, so ALIS reads the same model; it is one undoable step. A new project's model
  is written aligned, lines the dashboard adds take their neighbours' columns, and an
  imported fit is left as written until the user aligns it. On a hidden line the
  columns after a mask do not line up in the panel: a mask has one length whatever it
  hides, since its length would hint at the value (its sign, for example).

### Sessions: open, save, autosave (F1; Q3.2, Q3.8)

`session.py` (no Qt) has a `Session`, which holds:
- a `Project` and its `History`;
- the bundle's path (None until it is first saved) and whether there are unsaved
  changes;
- the view state, `ui/view.json`;
- the fingerprint of the bundle on disk when it was opened or last saved.

What it does:
- **Save** writes the bundle with `Project.save`, which is atomic, works under the lock
  and keeps the runs that `run_alis project.model` wrote meanwhile (Q1.7). It also
  writes `ui/view.json`. If the model, the files or the project data changed on disk
  since the session read them, Save asks first: overwrite, reload, or save as another
  file.
- **Autosave** writes a recovery copy, not the bundle, every 60 seconds while there
  are changes it has not yet kept (RJC, Q3.2). The interval is a preference. A clean
  close removes the copy. Opening a project whose recovery copy is newer than the
  bundle offers it. A command-line run
  (`run_alis project.model`) therefore only ever sees what the user saved.
- **The view state** (`ui/view.json`): the tab, the `.mod` panel and the sizes of the
  panes, and what later stages add (the selected system, snip or ion, the zoom). It is
  not part of the history.
- **A bundle changed on disk while open** (S31): the session notices. When it has no
  unsaved changes, it offers to reload; otherwise it warns, and Save asks as above.
- The undo history is not saved (Q2.8).

### The launcher (F14, F4; Q3.7, Q3.12, Q3.15)

`alis [path]` opens the dashboard (`run_alis` still runs fits):
- **nothing given:** a start page, with New project…, Open…, Import a fit… and the
  recent projects;
- **a `.model`:** that project;
- **a `.mod`:** the fit imported (F4) by `bundle.pack`. It is unsaved until Save, which
  offers `<name>.model` beside it. The notice that the structure was inferred (Q2.4)
  is shown in a banner;
- **anything else** stops with a message: "To start a project from a spectrum, run
  alis and choose New project." (exit status 2, as for a wrong argument; RJC, Q3.7).

**New project…** (Q3.12) collects in one dialog:
- the project's name, and the folder where its `.model` is saved. The bundle is
  written at once, so a project is never untitled;
- the mode (QSO Abs Line; Orders is listed as coming later). It is chosen here and
  not changed afterwards (RJC's review);
- the spectra (RJC's review, draft 3): a table, one row per spectrum, with its file,
  wavelength range, number of pixels, FWHM and what it contains. "Add spectra
  (ascii)…" opens a dialog of its own for a text spectrum: where the file is, its FWHM
  (km/s, the preferences' by default), and the role of each column, shown above its
  first rows: Wavelength, Flux, Error, Continuum, Mask or Ignore, guessed by D13's
  rule and changed by the user (S3). Every role but Ignore goes to one column only;
  Wavelength, Flux and Error are needed, and until they are given the spectrum is not
  added. A column set to Ignore is not loaded. The mask is a bad-pixel mask: a pixel
  whose mask is 1 is left out of the fit, even inside a fit region. "Add spectra
  (spec1d)…" is for PypeIt spec1d files, read by a mode that comes later (disabled
  until then). "Remove" is enabled when a spectrum is selected; double-clicking a
  spectrum changes it. Zero-level and systematics columns are not offered (RJC);
- the primary system's redshift, optional;
- whether to blind the whole fit.

The project's model is empty (settings, and empty data and model blocks), made by
`modes.QSOAbsLineMode.empty_project`.

If the `gui` extra is not installed, `alis` says `pip install "alis[gui]"` and stops
with status 1.

An empty model reads, and the validator says "The model has no data lines yet". Its
full check used to crash inside ALIS's loaders on such a model and report an
unexpected error; this was found and fixed while writing this document (Stage 2 log).

### Preferences (Q2.2; Q3.3)

`preferences.py` holds the shipped defaults, which a user file overrides. A bad
value falls back to the default, and an unknown key is ignored, each with a message.
They are edited in a Preferences dialog. The first preferences are:
- the Components tab's default number of columns (D37), and its default velocity
  range, ±200 km/s (D37);
- the autosave interval, the typing pause, and the delays of the reading and the full
  check;
- the settings of a new project's model, such as `run ncpus` (RJC, Q2.2), and its
  default FWHM;
- the length of the recent-projects list.

The recent projects are kept apart from the preferences, because the dashboard writes
them and the user does not.

### Tab markers (F12, D34; Q3.6)

`markers.py` gives each tab a state and a tooltip:

| Tab | ○ not started | ! needs attention | ↻ out of date | ✓ complete |
|---|---|---|---|---|
| Data | no file rows | a source missing or changed; the structure inferred and not yet confirmed | — | otherwise |
| Regions | no snips | a snip with no fitted pixels; pixels fitted twice | — | otherwise |
| Components | no components | a notice (untied ions, isotopes) | the regions changed after the components were last edited | otherwise |
| Fit | no run | the last run stopped with an error | the model changed since the latest run (Q2.9) | otherwise |
| Plot | always (after v1) | | | |

"The regions changed after the components" is found from two fingerprints that the
project keeps in `ui/project.json`: one of the regions, taken whenever the components
are edited.

### Relinking spectra (F13)

On opening, and from File → Relink spectra…, `bundle.check_sources` reports each source
as present, missing or changed. A dialog lists those missing or changed. "Locate…"
accepts a file only if its checksum matches (`bundle.relink_source`). A changed file is
reported, never accepted. A source embedded in the bundle is never missing. The Data
tab (Stage 4) shows the same states in its rows.

### Blinding and export (D24, D10, Q1.3, Q2.11; Q3.9)

- **The pill** in the toolbar says what is blinded: the ions or labels of the hidden
  lines, "Blind analysis" under `run blind True`, or "Not blinded".
- **Model → Blind the analysis** (global blind, which may be switched on part-way,
  QF.20(e)) writes `run blind True`. Undo stops at that step: undoing it after a
  blinded fit would show that fit's values. Only unblinding switches it off.
- **Model → Blind lines…** (Q3.13) lists the model's lines by system, component and
  ion, with no values, and writes `blind=True` on the lines ticked. The `.mod` panel's
  context menu has "Blind this line"; Stage 5 adds a switch to each component card.
  Undo stops at that step too.
- **No edit shows a hidden value** (Q3.14). An edit, typed or from a panel, that would
  make a hidden value visible (`blind=False` over `blind=True`, `run blind False`) is
  refused, with a message pointing to Model → Unblind…. Deleting a hidden line whole is
  allowed. Undo stops at any step that blinds something, however it was made.
- **Model → Unblind…** opens a dialog. It says that unblinding cannot be undone and
  how many values it will show, and asks for a note and an explicit confirmation.
  Only then does it call `blinding.unblind(confirmed=True, note=..., history=...)`.
- **File → Export plain files…** writes the model, its snips and the latest run's
  outputs to a folder (`bundle.extract`). When anything is blinded, it first warns
  that the hidden starting values will be written in plain text (D10).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage3_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the new `gui` tests;
> run the `fast` batch before the stage closes.

**3.1 — The Qt foundation (D2, Q3.1, Q3.15).** [DONE 2026-10-06]
- Create `alis/dashboard/qt/`, with a docstring saying that it is the only part of ALIS
  that imports Qt. All Qt goes through `qtpy`. `QT_API` is set to `pyside6` before
  `qtpy` is imported, unless the user has set it.
- Add the `alis` command to `pyproject.toml`, in `alis/scripts/dashboard.py`, with
  `--help`. Without the `gui` extra it prints how to install it and stops with
  status 1.
- Add the test set-up: pytest-qt in the `dev` extra, a `gui` marker in `pytest.ini`,
  and tests run off-screen (`QT_QPA_PLATFORM=offscreen`), skipped where Qt is missing.
  Add a CI job that installs `.[gui,dev]` and runs `-m gui`.
- **Check:**
  - `tests/test_dashboard_no_qt.py` still passes;
  - a test finds no direct import of PySide6, PyQt6 or pyqtgraph outside `qtpy`
    calls in `alis/dashboard/qt/`;
  - `alis --help` works;
  - with Qt hidden from the interpreter, `alis` prints the install message and exits
    with 1.

**3.2 — Preferences (`preferences.py`, Q2.2, Q3.3).** [DONE 2026-10-06]
- The shipped defaults, and the user file that overrides them, in the folder of
  Q3.3. A bad value falls back with a message, and an unknown key is ignored with a
  message.
- The recent projects, kept apart and capped.
- **Check:** unit tests of the defaults, the overrides, the fallbacks and the recent
  list, run in a temporary folder.

**3.3 — Sessions (`session.py`, F1, Q3.2, Q3.8).** [DONE 2026-10-06]
- `Session`: open, save, save as, autosave to a recovery copy, recovery on opening,
  the view state, and the detection of a bundle changed on disk.
- **Check** (unit tests, no Qt):
  - saving and reopening gives back the text, the files, `ui/project.json` and
    `ui/view.json`, byte for byte;
  - after edits, a recovery copy exists; reopening (as after a crash) offers it, and
    a clean close removes it;
  - Save keeps a run that `run_alis project.model` wrote meanwhile;
  - a conflicting change on disk is detected;
  - an unsaved import is saved with Save as.

**3.4 — The launcher and new projects (F14, F4, Q3.7, Q3.12).** [DONE 2026-10-06]
- The dispatch of `alis [path]`, and the New project dialog.
  `modes.QSOAbsLineMode.empty_project(sources)` (named `VoigtMode` until Q3.17)
  makes the empty project.
- **Check:**
  - unit tests of the dispatch for nothing, a `.model`, a `.mod`, and a spectrum
    (refused);
  - unit tests of an empty project from one and from two spectra, written at once;
  - a `gui` test opens the window for each, with J1358p6522 and Q1243p307 as the
    imported fits.

**3.5 — The window (D34, D43, D44, Q3.4, Q3.10).** [DONE 2026-10-06]
- The main window: the menus, the toolbar, the five tabs with their frames and the
  Fit sub-tabs, the status bar, and `qt/style.py`.
- **Check:**
  - the window opens off-screen for J1358p6522, Q1243p307, `examples/blind` and an
    empty project;
  - screenshots of each tab at 1440×900, compared by eye with the mockups and
    recorded in the log;
  - a test walks every widget's text, tooltip and menu entry, and finds no design
    reference (D44).

**3.6 — The `.mod` panel (D7, F5, F8, S16, Q3.5).** [DONE 2026-10-06]
- As in the Design section: the masked view, typing through `History.type`, reading
  after a pause, the quick and full checks, marks and the list of problems,
  cross-highlighting, collapsing, and the banner while the text does not read.
- **Check** (`gui` tests with `qtbot`):
  - typing in the panel changes the project's text exactly as typed, and one undo
    restores it;
  - with `examples/blind` open, the panel's document never holds a hidden value;
  - typing over a mask stores the value, still hidden;
  - problems appear on their lines after the pause;
  - a full check made stale by more typing is dropped;
  - the cursor keeps its place when the panel is redrawn.

**3.7 — Actions, undo and redo, and the shortcut sheet (F2, F10).** [DONE 2026-10-06]
- One registry of actions (name, shortcut, menu, handler). The menus, the toolbar and
  the shortcut sheet are built from it, and a later command search can use it.
- **Check:**
  - every action is in a menu and in the shortcut sheet, with its shortcut;
  - no two actions share a shortcut;
  - undo and redo work across typing in the panel and a step made by a panel.

**3.8 — Tab markers (`markers.py`, F12, Q3.6).** [DONE 2026-10-06]
- The rules of the Design section, and their display (icon, colour, tooltip).
- **Check:**
  - unit tests of each rule on J1358p6522, Q1243p307 and small models. For example:
    Fit ↻ after an edit once a run exists; Components ↻ after a region edit; Data !
    when a source is missing;
  - a `gui` test that the tabs show them.

**3.9 — Relinking moved spectra (F13).** [DONE 2026-10-06]
- The check on opening, and the Relink dialog.
- **Check:**
  - a moved source is relinked when its checksum matches;
  - a changed file is refused;
  - an embedded source is never reported missing.

**3.10 — Blinding and export (D24, D10, Q3.9, Q3.13, Q3.14).** [DONE 2026-10-06]
- The pill, Blind the analysis, Blind lines…, the Unblind dialog, the refusal of edits
  that would show a hidden value, and Export plain files with its warning.
- **Check:**
  - unblinding needs the confirmation, is logged in the bundle and clears the history;
  - export warns when anything is blinded;
  - switching global blind on, and blinding a line, are each one step, which undo does
    not pass, however they were made;
  - typing `blind=False` over `blind=True`, or `run blind False`, is refused, and
    deleting a hidden line is not;
  - the pill says what is blinded.

**3.11 — Close the stage.** [DONE 2026-10-06]
- Publish screenshots of the window beside the mockups, as a private page that RJC can
  comment on (Q3.11).
- `doc/ALIS_workflow.md`: a short section on launching the dashboard (`alis`). Also
  update `CHANGELOG.md` and `tests/README.md`.
- Run `test-coverage` on the new modules of `alis/dashboard/`, the `unit`, `gui` and
  `fast` batches.
- Record in this document what Stage 4 receives, and update the stage table in
  `dashboard_stage0.md` if anything moved.

## Status (2026-10-06)

*Written by Claude at the end of Prompt 1, and brought up to date with each prompt
since. The details are in `claude_prompts/logs/dashboard_stage3_log.md`.*

**Stage 3 is closed** (Prompt 4). Tasks 3.1–3.11 are done. The screenshots are
published beside the mockups for RJC's review
(https://claude.ai/artifact/A6ohUEnAe91TEDsKBLwuBt); Q3.16 asked for it, draft 2
(Prompt 2) applies RJC's comments on it, draft 3 (Prompt 3) the second round (Q3.18),
and Prompt 4 the last two (Q3.19). Stage 4's document is `dashboard_stage4.md`.
- **Tests.** 18 new test files and a helper module: 10 without Qt (`unit`, one
  `fast`), and 8 of the windows (`gui`, 54 tests, about 25 s, off-screen with
  pytest-qt). Coverage of `alis/dashboard/` over the dashboard tests: 94%.
- **The batches at the close:** `unit` 1526 passed, 0 failed; `gui` 54 passed; `fast`
  111 passed, 0 failed (13 min 36 s). After draft 2: `unit` 1729 passed, 0 failed;
  `gui` 62 passed; the dashboard's `fast` tests passed. After draft 3: `unit` 1730
  passed, 0 failed; `gui` 64 passed; the dashboard's `fast` tests passed.
- **ALIS outside `alis/dashboard/`:** the `alis` command (`alis/scripts/dashboard.py`,
  `pyproject.toml`), a `gui-test` extra, the `gui` marker, `tests/conftest.py` (Qt
  off-screen, and the `gui` tests skipped without the extra), the CI `gui` job, and
  the "(D10)" removed from `bundle.extract`'s note (Q3.15). ALIS's fitting code is
  unchanged.
- **Found and fixed along the way (in Stage 2's modules):**
  - two leaks of the blinding gate: a hidden line's values were shown while the text
    did not read, and on a commented-out hidden line; the gate now masks every value
    of a hidden line from its words, and the labels hidden when the text last read;
  - unblinding a model with no `run blind` line left it blind (ALIS's default is
    `run blind True`); it now writes `run blind False`;
  - a file row of a new project, with no snips yet, was dropped; and `inferred` now
    means that `ui/project.json` does not list the systems or rows;
  - ALIS's per-file notes about optional columns (continuum, zero level,
    systematics) are no longer listed as problems.

### What Stage 4 receives

The Data and Regions tabs of Stage 4 are built in the frames of `qt/tabs.py`
(`DataTab`, `RegionsTab`), replacing their placeholder panes, as the `gui-component`
skill describes:
- **The window** (`qt/window.py`, `MainWindow`): `session` (the open
  `session.Session`), `refresh()` after every step (it redraws the tabs through each
  tab's `refresh(session)`, the markers, the pill, the banners, the status bar, and the
  `.mod` panel when the text changed), `_run(name, handler)` for every action (reads
  typing first, reports a refused edit or an unexpected error), `ask`, `choose` and
  `run_dialog` for every question, file and dialog, and the `.mod` panel's
  `itemsAtCursor` signal, passed to the current tab's `show_items(items)` (S16).
  `panel.select_items(items)` highlights the lines of what a tab selects.
- **Sessions** (`session.py`): `Session.open`, `import_model`, `new` (an empty project
  from spectra), `save` (`Conflict`), `autosave`, `check_disk`, `take_runs`, `reload`,
  `unblind`, `view` and `set_view` (keep the selected file row, transition or zoom
  in `ui/view.json`), and `history`, which carries the blinding guard and the markers'
  tracker: a step that would show a hidden value raises `blinding.RevealError`, and
  a step that hides something cannot be undone past.
- **A new project** has file rows with sources but no snips: `project.rows` (each with
  `source`, its `columns` as chosen in New project (guessed by D13's rule, changed by
  the user), and its starting FWHM in `ui/project.json`), and the primary system when
  a redshift was given. Its model is written aligned. Stage 4 cuts the first snips
  (`edit.add_data_line`, `modes.QSOAbsLineMode`), and gives the Data tab's "Add
  file…" the same column roles (`qt/dialogs.ColumnRoles`, `modes.preview`,
  `modes.check_roles`, `modes.file_kind`).
- **Alignment** (`align.py`): `align(pm)` gives the change that lines the model up;
  lines the panels add take their neighbours' columns (`text.layout_like`).
- **The `.mod` panel** is in a dock (`window.dock`): `set_panel_open`,
  `set_panel_window` (its own window), and its state in `ui/view.json`.
- **The markers** (`markers.py`) already judge the Data and Regions tabs: sources,
  the inferred structure (`project.confirm_structure()` clears it), snips with no
  fitted pixels and pixels fitted twice.
- **Sources** (`sources.py`): `states`, `problems`, `relink`, `describe`, for the
  Data tab's rows (present, missing, changed, embedded).
- **Preferences** (`preferences.py`): add a `Spec` for any new preference (the
  Preferences dialog is built from `SPECS`).
- **Actions** (`actions.py`): add an entry for any new keyboard action; the menus,
  toolbar and shortcut sheet follow.
- **The look** (`qt/style.py`): the Okabe–Ito colours, the marker and problem icons,
  the fonts; pyqtgraph's plots should take their colours from it.
- **Tests:** `tests/dashboard_qt_helpers.py` (`make_window`, `wait_idle`,
  `ui_strings`, `design_references`); `doc/dashboard/skeleton/take_screenshots.py`
  and `build_review.py` for the next review.

Points for Stage 4:
- pyqtgraph is not yet used: the first plot (the Data tab's spectrum) is Stage 4's.
- RJC's review of the skeleton (Q3.16–Q3.19) changed New project, the column roles
  (Ignore; a bad-pixel mask; a column of 0s and 1s guessed as Ignore), the mode's
  name and place, and the `.mod` panel (a dock, its own window, Align columns). The
  frames of the Data and Regions tabs are as built.
- The `.mod` panel opens on every tab by default (Q3.16).

## Skills to use for this stage

- `gui-dev`: launching the window off-screen, driving it, and taking screenshots
  at 1440×900 (rewritten for the dashboard in Stage 2).
- `gui-component`: each tab's frame and the `.mod` panel, wired to the project, the
  history, the validator and the gate.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task; the `fast` batch
  at the end.
- `gen-tests` and `test-coverage`: the tests of `preferences.py`, `session.py` and
  `markers.py`.
- Claude Code's `artifact-design` skill: the page of screenshots (3.11).

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`:
  - D2, D5–D8, D10, D12, D24 and D34–D44;
  - QF.3, QF.10, QF.19, QF.20 and QF.32;
  - F1–F5, F8, F10–F14, S16 and S31.
- `claude_prompts/dashboard_stage0.md`: the plan and the v1 table; Q0.2 (the window's
  size and look).
- `claude_prompts/dashboard_stage1.md`: the bundle and `run.json`; Q1.3 (extracting a
  blinded bundle) and Q1.7 (a bundle edited during a run).
- `claude_prompts/dashboard_stage2.md`: "What Stage 3 receives"; Q2.2 (preferences),
  Q2.5 (unexpected errors), Q2.8 (history) and Q2.11 (unblinding).
- **The code:**
  - `alis/dashboard/` (the project model);
  - `alis/bundle.py`: `read`, `write`, `update`, `check_sources`, `relink_source`,
    `extract`, `unblind`;
  - `alis/scripts/run_alis.py`: the pattern for the entry point and its parser.
- **The mockups:** `doc/dashboard/mockups/build_mockups.py` (`window`, `mod_panel`,
  `fit_status`) and the published page,
  https://claude.ai/artifact/J43w9ERNESDo9hez9o918B.

## Queries

*Raised by Claude on 2026-10-06, while writing this document. Each gives Claude's lean,
and the Design section and the tasks follow the leans.*

**Q3.1 — The Qt binding, and testing the windows.** D2 chooses PySide6, written
against `qtpy` so that PyQt6 also works. This environment has PyQt6 6.10, qtpy 2.4 and
pytest-qt 4.5, but neither PySide6 nor pyqtgraph. I propose:
- the `gui` extra (PySide6, pyqtgraph) is installed in the development environment,
  so the windows are developed and tested on PySide6, the binding users get;
- `QT_API` is set to `pyside6` unless the user has chosen otherwise;
- the windows are tested with pytest-qt (added to the `dev` extra), off-screen, under
  a `gui` marker, and skipped where Qt is missing;
- a CI job installs `.[gui,dev]` and runs `-m gui` on Ubuntu and macOS. The `unit` job
  stays free of Qt.

Installing packages changes your environment, so I would like your go-ahead: either
you run `pip install -e ".[gui,dev]"`, or I do.

My lean: as proposed, with Claude installing the extra.

**Response:** Yes, I agree with this proposal. I have just installed pyside6 and pyqtgraph in my environment. The windows should be tested with pytest-qt off-screen, under a `gui` marker, and skipped where Qt is missing.

**Q3.2 — Autosave.** F1 says nothing is lost. There are two ways to autosave:
- **(a)** write the bundle itself, a short time after each change, as many modern
  applications do; Save is then rarely needed;
- **(b)** write a recovery copy elsewhere, and write the bundle only on Save. A
  newer recovery copy is offered when the project is next opened.

(b) keeps the bundle in a state the user chose: `run_alis project.model`, run from a
terminal while the dashboard is open (S31), then fits what was saved, never a
half-made edit. (a) loses nothing even without a recovery step, but a run can start
from an edit in progress.

My lean: (b), with a recovery copy 30 seconds after the last change, and the "autosaved"
time in the toolbar.

**Response:** I agree with your lean on this. Perhaps we can have a save option, and an autosave option that saves the state every minute or so. This will be particularly important during the beginning stages, so that the user does not lose any work while any bugs are ironed out.

**Q3.3 — Where the dashboard keeps its own files.** Your Q2.2 response asks for a
user file of preferences over shipped defaults. The dashboard also needs a list of
recent projects and somewhere for recovery copies. I propose one folder, `~/.alis/`
(or `$ALIS_HOME` when it is set). It would hold `dashboard.json` (the preferences, in
JSON, which needs only the standard library), `recent.json`, and `recovery/`. One
folder is easy to find on every platform, and easy to point at a temporary folder in
tests.

My lean: as proposed.

**Response:** I agree with your lean. Let's use `~/.alis/` for the dashboard files, including `dashboard.json`, `recent.json`, and `recovery/`. This will make it easy to manage and locate these files across different platforms.

**Q3.4 — What the tabs hold in this stage.** The tabs are frames of the agreed layout,
each pane titled and saying that it is not built yet. Stages 4–6 fill them, so the
window can be reviewed now (Q3.11) without waiting for them. The Fit tab has its
sub-tabs. The Plot tab is shown but disabled, with the tooltip "Not yet available".

My lean: as described.

**Response:**  I agree with your lean.

**Q3.5 — The `.mod` panel's timings.** Reading the model again takes 0.03 s for
J1358p6522 and up to 0.4 s for DH_orders, so it cannot happen at every key. I propose:
- the text is read again 0.3 s after typing stops;
- the full check runs 2 s after typing stops, in a worker thread;
- typing is grouped into one undo step until it pauses for 1 s, as now.

All three are preferences.

My lean: as proposed.

**Response:** I agree with this.

**Q3.6 — The rules of the tab markers.** The table in the Design section is my reading
of F12 and D34. Two choices in it:
- **(a)** "Components ↻" needs the project to remember the regions as they were
  when the components were last edited, as a fingerprint in `ui/project.json`;
- **(b)** the Plot tab is always "○" until it is built.

Would you like other rules, for example "!" on the Fit tab when the pre-flight check
(S17, Stage 6) finds something?

My lean: the table as it is; Stage 6 adds the pre-flight rule.

**Response:** The table described in the Design section is acceptable.

**Q3.7 — Opening a spectrum.** F14 says that `run_alisgui spectrum.dat` starts a new
project with that spectrum as its first dataset. Until Stage 4, nothing can cut snips
from it, so the project has the spectrum as its first file row and an empty model
(settings, and empty data and model blocks), on the Data tab. The column-role dialog
(S3) opens in Stage 4.

My lean: as described.

**Response:** Perhaps we should not allow opening a spectrum from the command line. A better approach is to open the GUI, and then decide to "Open existing project" or "Start a new project". This will allow the user to specify more information about the project in one place, rather than having to open a spectrum and then set up the project afterwards.

**Q3.8 — A bundle changed on disk while open.** S31 says the dashboard offers to reload
rather than overwrite. Two cases:
- **(a)** only the runs changed (a `run_alis project.model` finished): Save keeps them
  without asking, as `bundle.update` already does, and Stage 6 shows the new run;
- **(b)** the model, the files or the project data changed (another dashboard, or an
  edit of an extracted copy packed back): the dashboard offers to reload when it has
  no unsaved changes; otherwise Save asks whether to overwrite, reload, or save as a
  new file.

My lean: as described.

**Response:** I agree with your lean.

**Q3.9 — Switching blinding on and off.** QF.20(e) allows global blind to be switched
on part-way, and Q2.11 makes unblinding final. I propose:
- "Blind the analysis" writes `run blind True`, and undo stops at that step: undoing
  it after a blinded fit would show the values of that fit. In the dashboard global
  blind hides best-fit values only (QF.20(a));
- the Unblind dialog asks for a note and an explicit confirmation (a tick box and the
  button), and says how many hidden values it will show.

My lean: as proposed.

**Response:** I agree with your lean. The "Blind the analysis" option should write `run blind True`, and undo should stop at that step. The Unblind dialog should ask for a note and an explicit confirmation, and indicate how many hidden values will be revealed. There should also be an option to partially blind the analysis (using the blind=True option on models).

**Q3.10 — A dark theme.** Q0.2 left a dark theme to this stage. D43 chose a neutral,
light look, and every colour is in one module. I propose a light theme only in v1; a
dark one can be added later, by a second palette.

My lean: light only.

**Response:** I agree with your lean. A light theme only for v1 is acceptable, and a dark theme can be added later if needed.

**Q3.11 — Reviewing the skeleton.** As in Stage 0, I would publish screenshots of each
tab at 1440×900 beside the matching mockups, as a private page you can comment on.
Your comments would then shape Stages 4–6 before the panels are built.

My lean: yes, at the end of the stage.

**Response:** Yes, this makes sense. Please prepare screenshots and request approval. If the screenshots closely match the mockups, there will be relatively few iterations at this point.

*Raised by Claude on 2026-10-06 (Prompt 1), after reading the responses above. Claude's
note on Q3.2 comes first, then four new queries, asked in the session.*

**Note on Q3.2.** Save writes the bundle, and only Save does. Autosave writes the
recovery copy every 60 seconds while there are changes it has not yet kept (the
interval is a preference), and the toolbar shows the time of the last one. A clean
close removes the copy; opening a project whose copy is newer offers it.

**Q3.12 — The New project dialog (your Q3.7 response).** `run_alisgui` with no
argument opens the start page, with New project…, Open…, Import a fit… and the recent
projects. New project… opens one dialog that collects, in one place:
- the project's name, and the folder where its `.model` is saved. The bundle is
  written at once, so a project is never untitled and its recovery copy always has
  a home;
- the mode (QSO Abs Line; Orders is listed as coming later);
- the spectra: one or more files, each becoming one row of the Data tab, with its
  columns read by D13's rule and summarised (columns, pixels, wavelength range).
  The column-role dialog (S3) joins this step in Stage 4;
- the primary system's redshift, optional (it can be typed or found later in the
  Data tab);
- whether to blind the whole fit (`run blind True`).

`run_alisgui spectrum.dat` then stops with a message: "To start a project from a
spectrum, run run_alisgui and choose New project." (exit status 2, as for a wrong
argument).

My lean: as described.

**Response (RJC, in the session):** As proposed.

**Q3.13 — Blinding chosen lines (your Q3.9 response).** ALIS blinds whole lines
(`blind=True`). I propose:
- Model → "Blind lines…" opens a dialog listing the model's lines by system,
  component and ion (no values), with a tick box for each; the ticked lines get
  `blind=True`, which hides their starting values and their best-fit values;
- the `.mod` panel's context menu has "Blind this line";
- in Stage 5, each component card gets the same switch;
- as for global blind, undo stops at the step that blinds a line.

My lean: as described.

**Response (RJC, in the session):** As proposed.

**Q3.14 — Edits that would reveal hidden values.** Typing `blind=False` over
`blind=True`, or `run blind False`, would show hidden values without the Unblind
dialog. I propose:
- an edit, typed or from a panel, that would show a hidden value is refused, with a
  message pointing to Model → Unblind…. Deleting a hidden line whole is allowed,
  since nothing is shown;
- undo stops at any step that blinds something, however it was made (typed, or from
  the Model menu).

My lean: as described.

**Response (RJC, in the session):** As proposed.

**Q3.15 — Two small things outside `alis/dashboard/`.**
- **(a)** The `run_alisgui` command exists only once the package is installed again
  (`pip install -e ".[gui,dev]"`; nothing new is downloaded, since PySide6, pyqtgraph
  and pytest-qt are already here). Until then, the tests call
  `python -m alis.scripts.run_alisgui`.
- **(b)** `bundle.extract` gives a note that ends "(D10)", which `run_alis --extract`
  prints, and which the dashboard's Export would show. D44 keeps design references
  out of what users see. I would remove the "(D10)".

My lean: (a) Claude runs the install; (b) remove it.

**Response (RJC, in the session):** "Perhaps we should avoid `run_alisgui`. Let's just
have a new `alis` command on the command line that opens the gui. Then,
`run_alis project.model` runs a fit on the command line, while `alis project.model`
opens project.model in the dashboard." Asked again in the session: Claude runs the
install, and removes the "(D10)".

*Claude's note.* The launcher is therefore `alis [path]` (F14 renamed), in
`alis/scripts/dashboard.py`. The shell here has an old alias, `alis` →
`python /Users/rcooke/Software/ALIS/src/alis.py` (a file that no longer exists), which
hides the new command in an interactive shell until RJC removes it.

*Raised by Claude on 2026-10-06, at the end of Prompt 1.*

**Q3.16 — Reviewing the skeleton (your Q3.11 response).** The screenshots are
published beside the mockups, as a private page you can comment on:
https://claude.ai/artifact/A6ohUEnAe91TEDsKBLwuBt. To decide:
- **(a)** Does the window match the mockups closely enough to start filling the panes
  in Stages 4–6?
- **(b)** The `.mod` panel opens on every tab by default, and each project remembers
  whether it is open and its width. The mockups show it closed on most pages. Keep it
  open by default?
- **(c)** The banner about an imported fit's inferred structure stays until it is
  hidden; in Stage 4 it would also offer to confirm the structure. Is that the right
  place?
- **(d)** Anything in the menus, toolbar, status bar or start page to change?

My lean: (a) yes; (b) open by default; (c) yes; (d) your comments on the page.

**Response (RJC, as comments on the page, 2026-10-06):**
- **(a)** "This is overall looking excellent, but I have made several comments that
  indicate what minor tweaks are needed before we can proceed to stages 4-6."
- **(b)** "Yes, keep it open by default. It's useful for users to know that it exists,
  and is editable."
- **(c)** "I think the dashboard should guess the structure (and use this to set the
  default options) but then the user should be able to edit the structure themselves,
  if they wish. Note that an ascii file is just one filetype that will be supported,
  so we will need different options depending on what the filetype is. The two
  options currently foreseen are: (1) ascii files, like the examples already provided
  as context; and (2) PypeIt spec1d fits files. No context is currently provided on
  these files, but this will be provided at a later stage."
- **(d)** and the comments on the screenshots:
  - **Mode** (the toolbar): "I don't think Mode should necessarily be an option here.
    It is something that would be set at the New Project stage, and not changed
    thereafter. Also, can we please change "Mode=Voigt" to "Mode = QSO Abs Line"
    everywhere in the dashboard, and in the design documents?"
  - **The `.mod` text** (Regions): "It would be visually better if all parameters line
    up. For example, all column densities start from the same character value
    relative to the left (i.e. 0th character on each line). Similarly, all of the
    redshift variables start from the same character value from the left, and so
    forth. It would also be better if other parameters follow a similar convention of
    lining up in the .mod file, to make it a little easier to read."
  - **The `.mod` panel**: "Is it possible for the User to decide the width of the .mod
    text editor? Is it possible to make an option for it to be separated from the main
    dashboard window (i.e. a second window that the user can interact with, that still
    talks to the dashboard). If it's possible to separate the .mod editor window from
    the main dashboard window, it should also be possible to reattach a separate
    window to the main dashboard window."
  - **New project**: "Note that the filetype of "Spectra" could be an ascii file (as in
    the example shown) or a PypeIt spec1d file (no context example has been provided
    about this yet, it will come in a future stage, and is essential for a different
    "Mode"). If the file to be opened is an ascii file, I think this window pane should
    also request which column the wave, flux, and error are located. It would also be
    good to allow for other optional columns, such as the continuum."
  - **Fit · Results**: "Please swap the two subpanels labelled "Correlations" and "Run
    history"."

*Raised by Claude on 2026-10-06 (Prompt 2), before draft 2; RJC answered in the
session.*

**Q3.17 — Two choices for draft 2.**
- **(a) Lining up the `.mod` columns.** The dashboard keeps the text as written (D7),
  so aligning changes its spacing (spaces only; ALIS reads it the same). Options: an
  action, Model → Align columns, as one undoable step, with a new project's model and
  the lines the dashboard adds written aligned, and an imported fit left as written
  until the user aligns it; or always automatic; or aligned on screen only. Lean: the
  action.
- **(b) Renaming the mode.** Only what users and the documents read, or the code's
  names too (`VoigtMode` → `QSOAbsLineMode`, stored as `"qso_abs_line"`, with
  `"voigt"` still read)? Lean: everywhere.

**Response (RJC, in the session):** (a) the action, as the lean. (b) As the lean, "but
please note that there is a function called voigt.py with a class called Voigt, and a
parameter called self._idstr='voigt'. None of these should be changed. The function
and everything associated with this function should remain as voigt. It is only the
Mode name that should change."

*Raised by Claude on 2026-10-06, with draft 2 of the screenshots (Prompt 2), on the
same page: https://claude.ai/artifact/A6ohUEnAe91TEDsKBLwuBt.*

**Q3.18 — Draft 2 of the skeleton.** Draft 2 applies every comment on draft 1: the
mode "QSO Abs Line", chosen in New project and shown in the status bar; New project
asks for the role of each column of a text spectrum; Model → Align columns; the
`.mod` panel's width, and its own window; and the run history and the correlations
swapped in Fit · Results. To decide:
- **(a)** Do the changes do what you asked?
- **(b)** New project offers these roles for a column of a text spectrum: wavelength,
  flux, error, continuum, mask (0 or 1), or not used. ALIS can also read a zero level
  and systematics from a data file. Should New project offer them too?
- **(c)** If nothing else needs changing, may Stage 3 close, and Stage 4's document be
  written (Prompt 3)?

My lean: (a) yes; (b) not now: they belong to the snips rather than to the source
spectrum, and can be added with the Data tab in Stage 4 if wanted; (c) yes.

**Response (RJC, as comments on the page, 2026-10-06):**
- **(a)** "They do, thanks! However, I still have some further feedback to implement
  before we can move to stage 4."
- **(b)** "At this point, no, it should not offer to load a zero level or
  systematics."
- **(c)** "Not ready yet. I have requested another draft first."
- **Column roles:** "...we should add another option called "Ignore", this means that
  the column of data will be ignored and not loaded. This is useful when there are
  many columns in the input file (some of which are irrelevant, or duplicates). We
  should also check that each of the options are only specified once (e.g. Wavelength
  is only specified once, Flux is only specified once, etc.). The only one that can be
  specified more than once is "Ignore". We should also make sure that at least
  Wavelength, Flux, and Error are all specified, otherwise, ask the user to check the
  column designation and refuse to load the file until this requirement is satisfied.
  Note that the Mask column is not a Fit Range. It represents a Bad Pixel Mask, so
  that if it is a value of 1 or True, then this indicates the corresponding pixel
  should be excluded from the fitting procedure even if the fitrange encompasses the
  pixel."
- **New project:** "This screen still needs a lot of work. ... Please remove the Columns
  section from this screen. Please change "Name" to "Project Name". Please change
  "Folder" to "Project Folder". Instead of "Add Spectra..." and "Remove" buttons,
  please use the following buttons instead: "Add spectra (ascii)", "Add spectra
  (spec1d)", "Remove" (this option should only be enabled when a spectrum is selected
  in the information box to its left). When someone clicks the "Add spectra (blah)"
  button, a new screen opens up requesting the information about this spectrum: The
  location of the file on disk, the FWHM, the column information (for ascii). I think
  the box containing the `Spectra` information should be columns of information. The
  information should contain: Filename, Wavelength range, Number of pixels, FWHM, what
  data it contains (wavelength, flux, error, mask, etc.)"
- **Align columns:** "Should there be a button next to "Back to the dashboard" that
  says "Align columns". This would be a better place to put the column alignment
  feature." and "I think the Align columns button should only be on the .mod editor.
  Let's remove it from here [the banner]."

*Done by Claude on 2026-10-06 (Prompt 3), as draft 3, published on the same page;
the details are in the log.* Every comment is applied: the roles Wavelength, Flux,
Error, Continuum, Mask and Ignore, each but Ignore on one column only, the first three
required, and Mask a bad-pixel mask; New project as RJC laid it out, with a dialog of
its own for each spectrum (file, FWHM, columns); and Align columns only on the `.mod`
panel, beside its window button (and in its right-click menu), with its Ctrl+L. One
reading to confirm is Q3.19.

**Q3.19 — Draft 3 of the skeleton.** To decide:
- **(a)** Does draft 3 do what you asked?
- **(b)** A fourth column of 0s and 1s is guessed to be the mask, now a bad-pixel mask
  (1 leaves the pixel out). A `prepfit` snip has a fourth column of 0s and 1s too, but
  there 1 means "fit this pixel". A snip opened as a spectrum would then have its
  fitted pixels guessed as bad. The guess can be changed (to Ignore) in the spectrum's
  dialog. Is that acceptable, or should a 0/1 fourth column be guessed as Ignore?
- **(c)** If nothing else needs changing, may Stage 3 close, and Stage 4's document be
  written (Prompt 4)?

My lean: (a) yes; (b) keep guessing Mask, since the source spectra are rarely snips,
and the dialog shows the guess; (c) yes.

**Response (RJC, on the page and in Prompt 4, 2026-10-06):**
- **(a)** Yes, with "some very minor feedback" left as comments on the page: "There
  appears to be two hide buttons here." (the Data tab of Q1243p307).
- **(b)** "Guess Ignore. There should not be a fitrange loaded for these spectra.
  Snips have a fitrange, not the full spectrum."
- **(c)** Yes: Prompt 4 asks for the minor changes, then Stage 4's design document.

*Done by Claude on 2026-10-06 (Prompt 4); the details are in the log.* A fourth
column of 0s and 1s is guessed as Ignore (`modes.column_roles`), and D13 says so. The
second Hide was the previous project's: the banner's old buttons were only marked for
deletion, and drawn where they were until the event loop ran; they are now removed at
once. The page shows both ("Draft 3, final"). Stage 3 is closed.

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-2; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

2. Please see the comments I have provided on the html of the dashboard that is currently designed. Please make the appropriate changes based on these comments and share a draft 2 of the updated dashboard design. I will then provide another round of comments and we will iterate until the design is finalized. If you have any further queries before starting draft 2 of the design, please let me know. Otherwise, proceed to generate draft 2 of the updated dashboard design based on my feedback.

3. Please see the comments I have provided on the html of the dashboard that is currently designed. Please make the appropriate changes based on these comments and share a draft 3 of the updated dashboard design. I will then provide another round of comments and we will iterate until the design is finalized. If you have any further queries before starting draft 3 of the design, please let me know. Otherwise, proceed to generate draft 3 of the updated dashboard design based on my feedback.

4. There are some very minor feedback to take care of. I have left comments on the html of the dashboard. Please implement these minor changes first. Then, based on the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-3, please generate the design document for Stage 4.
