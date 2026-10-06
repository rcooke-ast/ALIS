# Prompt file for ALIS software dashboard creation -- STAGE 3

> **The Qt skeleton of the chosen layout.** The dashboard gets its window:
> - the five tabs of D34, with their status markers (F12);
> - the collapsible `.mod` panel, the menus, the toolbar and the status bar.
>
> It can:
> - open, save and autosave a project (F1);
> - import an existing fit (F4) and relink moved spectra (F13);
> - list its shortcuts (F10);
> - be launched with `run_alisgui` (F14).
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
this section follows the leans.*

### Two layers

```
alis/dashboard/qt/   the windows: Qt through qtpy, and pyqtgraph from Stage 4
  │                  thin: they draw the project, and turn the user's actions into steps
alis/dashboard/      the project model of Stage 2, and three modules of this stage that
                     need no Qt: preferences.py, session.py, markers.py
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
│ ↶ Undo ↷ Redo │ Open… Save  autosaved 12:41 │ Mode: Voigt ▾  Blinded: N(D I) │ … ? │
│ ✓ Data │ ✓ Regions │ ↻ Components │ ! Fit │ ○ Plot                               │
│ ┌──────────────── the tab ────────────────────────┐ ┌──── .mod panel ───────────┐ │
│ │                                                 │ │ J1358p6522.mod      ◂ Hide│ │
│ │                                                 │ │  1 run  ncpus  -1         │ │
│ │                                                 │ │ 57   voigt ion=2H_I ▒▒▒▒ …│ │
│ │                                                 │ │ ! 3 problems              │ │
│ └─────────────────────────────────────────────────┘ └───────────────────────────┘ │
│ J1358p6522.model │ No fit running │ The model reads                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- **Menus:**
  - **File:** New project, Open…, Open recent, Import a fit (.mod)…, Save, Save as…,
    Export plain files…, Relink spectra…, Preferences…, Quit.
  - **Edit:** Undo and Redo (with what they undo), and Cut, Copy and Paste in the
    `.mod` panel.
  - **View:** the five tabs (Ctrl+1 to Ctrl+5), and Show the `.mod` panel.
  - **Model:** Check the model now, Blind the analysis, Unblind…, and Mode.
  - **Fit:** Run and Commit run, shown but disabled until Stage 6.
  - **Help:** Keyboard shortcuts, About ALIS.
- **Toolbar:** as in the mockups. It holds Undo and Redo, Open and Save with the time
  of the last autosave, the mode, the blinding pill, Export, and the shortcut sheet.
- **Tabs:** Data, Regions, Components, Fit (with its sub-tabs Inspect, Results and
  Compare), and Plot. Each tab has a marker with an icon and a colour: ✓ complete,
  ↻ out of date, ! needs attention, ○ not started.
- **Status bar:** the project's file, the fit ("No fit running" until Stage 6), the
  state of the model ("The model reads", "3 problems", or "The model does not read:
  the panels are paused"), and the last autosave.
- **Look (D43, Q3.10):** Qt's Fusion style with a light palette. The colours come from
  the Okabe–Ito palette, all in one module (`qt/style.py`), and every coloured state
  also has an icon. The window is judged at 1440×900. No text in it names a design
  document (D44).

### The tabs in this stage (Q3.4)

Each tab is a widget with the panes of its agreed layout (D35–D42), each pane titled,
holding a short line saying it is not built yet. Stages 4–6 replace the placeholders,
following the `gui-component` skill. The Fit tab has its three sub-tabs. The Plot tab
is shown, but disabled, with the tooltip "Not yet available": it is built after v1
(D42).

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
- **Autosave** writes a recovery copy, not the bundle, a short time after each change.
  The interval is a preference. A clean close removes the copy. Opening a project
  whose recovery copy is newer than the bundle offers it. A command-line run
  (`run_alis project.model`) therefore only ever sees what the user saved.
- **The view state** (`ui/view.json`): the tab, the `.mod` panel and the sizes of the
  panes, and what later stages add (the selected system, snip or ion, the zoom). It is
  not part of the history.
- **A bundle changed on disk while open** (S31): the session notices. When it has no
  unsaved changes, it offers to reload; otherwise it warns, and Save asks as above.
- The undo history is not saved (Q2.8).

### The launcher (F14, F4; Q3.7)

`run_alisgui [path]` opens:
- **nothing given:** a start page on the Data tab, with New project, Open…, Import a
  fit… and the recent projects;
- **a `.model`:** that project;
- **a `.mod`:** the fit imported (F4) by `bundle.pack`. It is unsaved until Save, which
  offers `<name>.model` beside it. The notice that the structure was inferred (Q2.4)
  is shown in a banner;
- **anything else:** a spectrum, which becomes a new project with that spectrum as its
  first file row. The project's model is empty (settings, and empty data and model
  blocks), and its column roles follow D13; the column-role dialog comes in Stage 4.

If the `gui` extra is not installed, `run_alisgui` says
`pip install "alis[gui]"` and stops with status 1.

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

**3.1 — The Qt foundation (D2, Q3.1).**
- Create `alis/dashboard/qt/`, with a docstring saying that it is the only part of ALIS
  that imports Qt. All Qt goes through `qtpy`. `QT_API` is set to `pyside6` before
  `qtpy` is imported, unless the user has set it.
- Add `run_alisgui` to `pyproject.toml`, in `alis/scripts/run_alisgui.py`, with
  `--help`. Without the `gui` extra it prints how to install it and stops with
  status 1.
- Add the test set-up: pytest-qt in the `dev` extra, a `gui` marker in `pytest.ini`,
  and tests run off-screen (`QT_QPA_PLATFORM=offscreen`), skipped where Qt is missing.
  Add a CI job that installs `.[gui,dev]` and runs `-m gui`.
- **Check:**
  - `tests/test_dashboard_no_qt.py` still passes;
  - a test finds no direct import of PySide6, PyQt6 or pyqtgraph outside `qtpy`
    calls in `alis/dashboard/qt/`;
  - `run_alisgui --help` works;
  - with Qt hidden from the interpreter, `run_alisgui` prints the install message and
    exits with 1.

**3.2 — Preferences (`preferences.py`, Q2.2, Q3.3).**
- The shipped defaults, and the user file that overrides them, in the folder of
  Q3.3. A bad value falls back with a message, and an unknown key is ignored with a
  message.
- The recent projects, kept apart and capped.
- **Check:** unit tests of the defaults, the overrides, the fallbacks and the recent
  list, run in a temporary folder.

**3.3 — Sessions (`session.py`, F1, Q3.2, Q3.8).**
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

**3.4 — The launcher and new projects (F14, F4, Q3.7).**
- The dispatch of `run_alisgui [path]`. `modes.VoigtMode.empty_project(sources)` makes
  the empty project.
- **Check:**
  - unit tests of the dispatch for nothing, a `.model`, a `.mod` and a spectrum;
  - a `gui` test opens the window for each, with J1358p6522 and Q1243p307 as the
    imported fits.

**3.5 — The window (D34, D43, D44, Q3.4, Q3.10).**
- The main window: the menus, the toolbar, the five tabs with their frames and the
  Fit sub-tabs, the status bar, and `qt/style.py`.
- **Check:**
  - the window opens off-screen for J1358p6522, Q1243p307, `examples/blind` and an
    empty project;
  - screenshots of each tab at 1440×900, compared by eye with the mockups and
    recorded in the log;
  - a test walks every widget's text, tooltip and menu entry, and finds no design
    reference (D44).

**3.6 — The `.mod` panel (D7, F5, F8, S16, Q3.5).**
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

**3.7 — Actions, undo and redo, and the shortcut sheet (F2, F10).**
- One registry of actions (name, shortcut, menu, handler). The menus, the toolbar and
  the shortcut sheet are built from it, and a later command search can use it.
- **Check:**
  - every action is in a menu and in the shortcut sheet, with its shortcut;
  - no two actions share a shortcut;
  - undo and redo work across typing in the panel and a step made by a panel.

**3.8 — Tab markers (`markers.py`, F12, Q3.6).**
- The rules of the Design section, and their display (icon, colour, tooltip).
- **Check:**
  - unit tests of each rule on J1358p6522, Q1243p307 and small models. For example:
    Fit ↻ after an edit once a run exists; Components ↻ after a region edit; Data !
    when a source is missing;
  - a `gui` test that the tabs show them.

**3.9 — Relinking moved spectra (F13).**
- The check on opening, and the Relink dialog.
- **Check:**
  - a moved source is relinked when its checksum matches;
  - a changed file is refused;
  - an embedded source is never reported missing.

**3.10 — Blinding and export (D24, D10, Q3.9).**
- The pill, Blind the analysis, the Unblind dialog, and Export plain files with its
  warning.
- **Check:**
  - unblinding needs the confirmation, is logged in the bundle and clears the history;
  - export warns when anything is blinded;
  - switching global blind on is one step, which undo does not pass;
  - the pill says what is blinded.

**3.11 — Close the stage.**
- Publish screenshots of the window beside the mockups, as a private page that RJC can
  comment on (Q3.11).
- `doc/ALIS_workflow.md`: a short section on launching the dashboard. Also update
  `CHANGELOG.md` and `tests/README.md`.
- Run `test-coverage` on the new modules of `alis/dashboard/`, the `unit`, `gui` and
  `fast` batches.
- Record in this document what Stage 4 receives, and update the stage table in
  `dashboard_stage0.md` if anything moved.

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

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-2; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

2. Based on the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-3, please generate the design document for Stage 4.
