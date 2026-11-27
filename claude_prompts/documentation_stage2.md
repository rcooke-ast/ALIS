# Prompt file for ALIS documentation -- STAGE 2

> **Getting started and the GUI.** After this stage, a new user can install ALIS,
> make a first fit in the GUI, and find every tab, menu item and dialog described:
> - Getting started: Introduction, Installing, and Your first fit;
> - The GUI: its seven pages, from The GUI at a glance to Earlier fits and several
>   spectra;
> - `doc/source/screenshots.py`, the one script that makes every image those pages
>   use;
> - Help → ALIS documentation, which opens the page of the tab in view (DQ.12).
>
> Its one change to ALIS is the Help link: a menu item, a module without Qt that
> maps each tab to its page, and their tests. The rest of the stage is pages and
> their tests.
>
> The rules of `documentation_stage0.md` hold: "How the pages are written" for every
> page, and "How each stage runs" for the stage. Those include the GUI and Ryan
> Cooke by name. After every task:
> - the site builds with no warning;
> - `tests/test_docs.py` passes;
> - pre-commit passes on the task's Python files.
>
> A task that changes ALIS also runs the batches it touches. A question that comes up
> while a task is built pauses the work: it is asked in the Queries section, and its
> answer is logged there.
>
> "DQ.*n*" refers to `documentation_stage0.md`'s queries, "DQ1.*n*" to
> `documentation_stage1.md`'s, and "DQ2.*n*" to this document's. "WF" is
> `doc/ALIS_workflow.md`. "D*n*" refers to `claude_prompts/ALIS_v2_dashboard_prompts.md`.

*Written by Claude on 2026-11-27, at the close of Documentation Stage 1 (Task 1.9, its
Prompt 4), from the plan in `documentation_stage0.md`, Stage 1's log, and the
repository at 4f05fca. The design gives each file and what it holds, and the tasks
give the order and each task's check. DQ2.1–DQ2.3 ask what is Ryan Cooke's to
choose, each with Claude's lean. No code was changed.*

*Three changes to the plan's Stage 2, each for a reason found while drafting:*
- *The Help link comes first (Task 2.1), not eighth. It is the stage's one change to
  ALIS, so its batches run once, on code that the later tasks do not change. The
  pages can then describe it as it is.*
- *`screenshots.py` comes before the pages (Task 2.2), because the Introduction's
  figure is one of its images.*
- *The completeness test (Task 2.9) comes after the last page, since it fails until
  every page is written. The test of the Help link's anchors comes with the Fit page
  (Task 2.6), whose headings they name.*

*Changed by RJC before the stage started (2026-11-27): readthedocs builds `alis_v2`
as one hidden version, and this stage as a preview of its pull request into
`alis_v2` (DQ.8's note). "RJC first", the review (2.10) and the close (2.11)
follow.*

## RJC first

readthedocs builds `alis_v2` as one hidden version, and each stage as a preview of
its pull request into `alis_v2` (DQ.8, as changed after Stage 1's close).
*(Done: Stage 1 is merged into `alis_v2` at 7590f56, `alis_v2`'s hidden version
built it, and pull-request previews are on.)*

Before Task 2.1:
1. Make the branch `alis_v2_doc_stage2` from `alis_v2`. Its first commit holds this
   document and the changes made with it, which are not yet committed.
2. Push the branch, and open a pull request from it into `alis_v2`. readthedocs then
   builds a preview of it on every push, at the address that the pull request's
   check links, usually
   `https://alis-fitting--<number>.org.readthedocs.build/en/<number>/`.

`alis_v2_doc_stage1`'s hidden version can be deactivated. Only Stage 1's review page
links to it, and that review is closed.

## Design

### How the stage is built

- **The order.**
  - The Help link comes first (2.1), since it is the one change to ALIS. Then
    `screenshots.py` (2.2), which the pages' images come from.
  - Next come the pages, in the order a new user meets them: Introduction and
    Installing (2.3); the GUI's pages (2.4–2.7); and Your first fit (2.8), which
    links to all of them.
  - The completeness test (2.9) comes once every GUI page exists.
  - The review (2.10) and the close (2.11) end the stage.
- **The code is the authority.** Each GUI page is written from the code and from
  the window, seen with `gui-dev`. WF §4.3 and the decisions of
  `ALIS_v2_dashboard_prompts.md` are used to find what a tab does. Where WF and the
  code differ, the page follows the code, and the difference is logged.
- **One source.**
  - A page that names a menu item or a dialog uses the window's own words. The
    completeness test (2.9) checks that every one of them is named.
  - Your first fit's numbers come from the run that `screenshots.py` makes, and a
    test checks that the page quotes them (2.8).
  - The Help link's pages are checked against `PAGES`.
- **A page written in this stage loses its `% Written in:` line.** Its `% Sources:`
  stays, brought up to date with what the page was written from.
- **No page is written ahead of its stage.** The GUI's pages link to the Reference
  (Settings, Model functions) and to The model file, which exist from Stage 1 as
  skeletons. Those links work now, and the pages behind them come in Stages 3 and 4.

### The pages

Ten pages, each at the address Stage 1 fixed, with the title and sentence of purpose
that RJC reviewed in Stage 1. Each section below gives what the page holds, in the
order the page gives it, and its images (from `screenshots.py`).

**How a GUI page is laid out:**
- the sentence of purpose;
- the tab's tasks, each a `##` section that shows how to do one thing ("Cut a snip",
  "Add a component"), in the window's words;
- a last section, **More on this tab**, which describes the tab's other actions and
  dialogs, one short entry each;
- a link to the Reference where a setting or a function needs more than a line.

A key is written as the Keyboard shortcuts sheet shows it on Linux and Windows
(`Ctrl+S`). The GUI at a glance says once that macOS shows Ctrl as ⌘.

**`user/start/introduction.md` — Introduction.**
- What ALIS fits: a model of the emission (the continuum), the absorption (each ion's
  Voigt profiles, from its atomic data) and the instrument's resolution, matched to
  one or more spectra by minimising χ². One figure: the first fit's (Your first fit),
  drawn by the Figure tab.
- The two ways to use ALIS:
  - the GUI (`alis`), which prepares, runs and inspects a fit;
  - `run_alis`, which runs a model file or a project from the command line.

  A project (`.model`) is one file that both read.
- Where to go next: Installing, Your first fit, The model file. Citing ALIS is on
  About ALIS.
- *Image:* `introduction-figure.png`.

**`user/start/installing.md` — Installing.**
- Python 3.13 or newer.
- The install, as DQ2.1 decides.
- The extras, as a table:
  - `gui` for the GUI (PySide6, qtpy and pyqtgraph);
  - `gpu` for GPUs (numba). The CUDA toolkit comes from the computer's own driver,
    not from the extra.

  `linetools`, for the `lsf` function, is installed on its own.
- The developers' extras (`dev`, `gui-test`, `docs`) are named once, with a link to
  the Developer guide.
- Checking the install: `alis` opens the start page, and `run_alis` fits the example
  that Your first fit uses.
- Updating, as DQ2.1 decides.
- Written from `pyproject.toml`. `doc/installing.rst`, which describes v1, is removed.
  The v1.0 manual's `install.tex` links to that file on `master`, and is left as it
  is (DQ.11).

**`user/start/first-fit.md` — Your first fit.** `examples/metal_line_abs`, a spectrum
of O I 1302 and Si II 1304, from its spectrum to its result:
- **In the GUI:**
  1. New project…, then Add a spectrum (ascii) with `OI_SiII.dat`, and its
     resolution;
  2. the Data tab: the redshift (0);
  3. the Regions tab: a snip for each line, and the continuum;
  4. the Components tab: a component for each ion;
  5. the Fit tab: Run, then Results (Fit statistics, and each ion's log N with its
     error);
  6. the Figure tab: a figure;
  7. Save.

  Each step links to its tab's page.
- **From the command line:** as DQ2.3 decides.
- Its numbers come from `screenshots.py`'s run of these steps, which writes them to
  `first_fit.json`. A test checks that the page quotes each of them (2.8).
- *Images:* `first-fit-new-project.png`, `first-fit-regions.png`,
  `first-fit-components.png`, `first-fit-results.png`.

**`user/gui/overview.md` — The GUI at a glance.**
- Opening the GUI:
  - `alis` (the start page: New project…, Open…, and the recent projects);
  - `alis project.model`;
  - `alis fit.mod`, which imports a fit, with a link to Earlier fits.
- **Projects and saving:**
  - what a project holds (the model, its snips, its atomic table and its runs, in one
    file);
  - Save, Save as…, and Save a copy with spectra…;
  - autosave and recovery after a crash;
  - Close project.
- **The window:**
  - the tabs, and their markers;
  - the `.mod` panel: Show the .mod panel, The .mod panel in its own window, Align
    columns, and Load a fit (.mod) into this project…;
  - the status bar.
- **Undo** and Redo: one history for the tabs and the panel. Also Check the model
  now.
- **Blinding:** Blind the analysis…, Blind lines…, and Unblind…, with their dialogs.
- **Preferences…**, and where they are kept.
- **Keys:** Keyboard shortcuts (F1), and each tab's Keys… button with its dialog.
- **Help:** ALIS documentation, and About ALIS.
- *Images:* `start-page.png`, `window.png` (the Data tab with the `.mod` panel open).

**`user/gui/data.md` — Data.**
- New project…: its dialog (the name, the folder and the mode).
- Adding spectra: Add a spectrum (ascii), and the Column roles dialog.
- The redshift and the systems: typing it, Identify a feature…, a second system, and
  removing one.
- The datasets table: the reference, FWHM and resolution (with `lsffile`'s
  dialog), shift, and zero level.
- Coverage: which transitions each spectrum covers.
- Spectra kept with a project: Relink spectra…, and Keep a copy in the project.
- *Images:* `new-project.png`, `data-tab.png`, `column-roles.png`.

**`user/gui/regions.md` — Regions.**
- Choosing a transition: the system, the element (‹ › or `[` `]`) and the ion;
  Previous transition and Next transition.
- SNIP and CLEAR.
- Fit regions: Adjust fit regions: all spectra, Adjust fit regions: one spectrum, and
  Remove region.
- The continuum:
  - Lower order and Higher order;
  - Auto first guess;
  - Normalised and Model;
  - knots, with Add knot;
  - a continuum shared between snips.
- A snip's own resolution, shift and zero level, with Set LSFSpline knots….
- Pixels fitted twice, and the Snip box's two fixes.
- Line IDs.
- *Images:* `regions-tab.png`, `lsfspline-knots.png`.

**`user/gui/components.md` — Components.**
- Ions and systems: Previous ion and Next ion, Add ion…, and Add system….
- Adding and dragging components: + Add component, a drag, and the N and B keys.
- The cards:
  - each parameter's state (free, fixed, tied);
  - the temperature modes;
  - limits;
  - Exclude and Include;
  - Blind….
- Blends: Add interloper…, and the Lyα forest.
- Links: the `.mod` panel's `link` block, and what a card shows of a link.
- Renumber labels….
- *Images:* `components-tab.png`.

**`user/gui/fit.md` — Fit.** Its three sub-tabs have the headings **Inspect**,
**Results** and **Compare**, which the Help link opens (`fit.html#results`).
- Run, Save and Run, and Stop. The pre-flight check, each of its findings with what
  to do.
- **Inspect:** the Run box, the views (Previous view, Next view, Normalised), and the
  residuals.
- **Results:**
  - Fit statistics;
  - Warnings and options to resolve…, each warning with what to do;
  - the correlations;
  - the run history, Commit run…, and restoring an earlier run's model.

  The pre-flight check and Warnings and options to resolve… are the GUI's own
  suggestions, which DQ.15's response relies on. Troubleshooting (Stage 3) links to
  them here.
- **Compare:** any two runs, each ion's total log N and each isotope's ratio.
- The tools:
  - Continue from best fit…;
  - Convergence check…;
  - Random restarts… and Export restarts…;
  - Review clipped pixels…;
  - Export… and All settings….
- *Images:* `fit-inspect.png`, `fit-results.png`, `fit-compare.png`,
  `warnings-dialog.png`, `random-restarts.png`, `export-dialog.png`.

**`user/gui/figure.md` — Figure.**
- Figures, presets and the layout.
- Panels and their options, with the Panel options dialog.
- The preview.
- Exporting a figure, with its plotting script.
- *Images:* `figure-tab.png`, `panel-options.png`.

**`user/gui/earlier-fits.md` — Earlier fits and several spectra.**
- Open a fit (.mod)…: what the GUI reads of a `.mod` and its `.mod.out`, and the
  questions it asks.
- Load snips…:
  - a snip's four columns (wavelength, flux, error and fit range), and its file
    name, which gives its ion and rest wavelength (`J0133_1H_I_1215.7_reg.dat`), as
    the GUI's texts say after Stage 1;
  - where the snips go, with its dialog.
- A project of several spectra: the datasets, which snip came from which spectrum,
  and Relink spectra….
- Load a fit (.mod) into this project… (the `.mod` panel).
- *Images:* `snips-assign.png`, `two-spectra.png`.

That is 23 images. The review pages' full windows at 1440×900 take 130–190 kB as Qt
writes them, and their dialogs 20–75 kB, so the folder holds about 3 MB. The task
that adds an image logs its size, and the close logs the folder's.

### `doc/source/screenshots.py`

One script makes every image the pages use (DQ.7's response), committed with them:

```
python doc/source/screenshots.py [NAME ...]
```

It writes into `doc/source/_static/screenshots/`. With names, it remakes those
images alone.

- **`SHOTS`**, a tuple at the top of the module, names every file it writes: the
  images, and `first_fit.json`. It is the one list, which the tests read with `ast`.
- **How it draws.**
  - The window is off-screen at 1440×900, the size the GUI is designed for (D43).
  - `ALIS_HOME` is a temporary folder, so that the user's preferences and recent
    projects neither change the images nor are changed.
  - A dialog, a menu or a message box is drawn on its own, at its own size.
  - The helpers are gathered from the review pages' scripts in
    `doc/dashboard/stage11–14/` (`done`, `shoot_dialog`, `shoot_box`, `shoot_menu`,
    and the timer that grabs a menu while it is open). Those scripts are left as
    they are.
- **What it draws from:** the shipped example `examples/metal_line_abs`, and the
  tests' synthetic fits (`tests/dashboard_fits.py`, `tests/dashboard_synthetic.py`),
  never `context/` (DQ.7). Random draws have fixed seeds.
- **Your first fit's run.** It makes the fit of Your first fit, step by step, as the
  page describes it. Its images come from that run, and so does the Introduction's
  figure. It writes the fit's numbers to `first_fit.json`.
- **The images are PNGs as Qt writes them.** Pillow's `optimize=True` saved 3% on
  two of the review pages' images, which is not worth a step.
- **It is a file of the site,** so `test_docs.py` reads it: no "RJC", no
  "dashboard" except in a code path (`alis.dashboard.qt`), and no `context/`. Its
  docstrings end "Generated by Ryan Cooke and Claude."
- **The window's title bar is drawn by the operating system,** so an off-screen grab
  cannot show it. No page needs it.

**`conf.py`:** `html_static_path = ["_static"]`.

**On a page,** an image is a MyST `{image}` with its `:alt:` text, at
`/_static/screenshots/<name>.png`.

**The test** (`tests/test_docs.py`, Task 2.2):
- every image a page uses is in `_static/screenshots/`, exists, and is named in
  `SHOTS`;
- every file in that folder is named in `SHOTS`;
- no name is in `SHOTS` twice.

### Help → ALIS documentation (DQ.12)

**`alis/dashboard/helpdocs.py`** (new, no Qt):
- `SITE = "https://alis-fitting.readthedocs.io/en/"`;
- `PAGES`, the map from what is in view to its page: the start page to
  `user/gui/overview`; each tab (Data, Regions, Components, Fit, Figure) to its
  page;
- `SECTIONS`, the Fit tab's sub-tabs to their anchors on its page: Inspect to
  `inspect`, Results to `results`, Compare to `compare`;
- `version_slug(version)`, the version of the site, as DQ2.2 decides;
- `url(tab=None, subtab=None, version=alis.__version__)`, the address to open.

**`actions.py`:** the action `help.documentation`, "ALIS documentation", first in the
Help menu, with the tooltip "Open the documentation's page for what is in view, in
your web browser". It has no key: F1 stays Keyboard shortcuts. The actions become
69.

**`qt/window.py`:**
- `do_help_documentation` opens `helpdocs.url(...)` for the tab in view, and for the
  Fit tab its sub-tab;
- the opening goes through a window method, `open_url`, which the tests replace, as
  they replace `window.ask`;
- if the browser cannot be opened, the window says so and gives the address, so that
  it can be copied.

**The tests:**
- `tests/test_dashboard_helpdocs.py` (`unit`): the address of each tab and sub-tab;
  the start page's; a development version's and a release's.
- `tests/test_docs.py`:
  - every page of `helpdocs.PAGES` is in `PAGES` (Task 2.1);
  - every anchor of `SECTIONS` is a heading of the Fit page, by MyST's rule for
    anchors (Task 2.6).
- A `gui` test: the action, triggered on each tab and each of the Fit tab's
  sub-tabs, opens the address of its page. It also checks what the window says when
  the browser cannot be opened.
- `test_gui_by_name.py` reads the new texts as it reads the others.

**`CHANGELOG.md`:** under Added, Help → ALIS documentation.

`dashboard_stage15.md`'s item 17 (Help → Atomic data) adds its own entry to
`helpdocs.PAGES` when it is built.

### The completeness test (Task 2.9)

In `tests/test_docs.py`, read with `ast` and Python alone (no Qt), so that it runs in
the `unit` batch:
- **Every action** of `actions.ACTIONS` is named, by its text, on a page under
  `user/gui/` or on Your first fit. Cut, Copy, Paste and Select all are exempt: they
  are the standard ones of every program.
- **Every dialog:**
  - each class of `alis/dashboard/qt/` that derives from `QDialog` (directly, or
    through another dialog class there) is in the test's `DIALOGS`, as is the
    Column roles dialog built in `qt/data.py`;
  - `DIALOGS` gives the words of each one's title that a page uses ("New project",
    "Where the snips of", "Panel options"), and each is named on a GUI page;
  - where the title is a literal in the code, the test checks that it begins with
    those words, so that a renamed dialog fails;
  - the general dialogs whose titles come from their callers (`AskDialog`,
    `PlanDialog`, `LsfFileDialog`, `NameDialog`) are listed with the reason they are
    exempt. Their uses are described where a task meets them.

  The window's message boxes (`window.ask`, `window.tell`) are questions in passing,
  not dialogs, and are not counted.
- **Check:** a scratch copy with an action added to `ACTIONS`, a dialog class added,
  or a dialog's title changed fails the test.

### What stays as it is

- WF, until Stage 3 makes it a pointer. Its line `pip install "alis[gui]"` (§4.3)
  goes with it then (DQ2.1).
- The review pages' scripts in `doc/dashboard/`. `screenshots.py` copies their
  helpers and does not import them.
- ALIS's messages (DQ.15). Two of them say `pip install "alis[gui]"`
  (`scripts/dashboard.py`) and `pip install 'alis[gpu]'` (`gpu.py`). With ALIS
  installed, pip reads those extras from the installed ALIS and fetches nothing
  named `alis` from PyPI (checked with `pip install --dry-run "alis[gui]"`). They are
  right, unless DQ2.1 chooses (c).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/documentation_stage2_log.md`.
> After every task:
> - the site builds with no warning (`build-docs`, in a scratch venv with the `docs`
>   extra);
> - `tests/test_docs.py` passes;
> - pre-commit passes on the task's Python files (never on untracked `context/`).
>
> Task 2.1 also runs the `unit` and `gui` batches, also under CI's versions (memory
> `ci-test-versions`, on a scratch snapshot). No fit changes in this stage, so the
> regression harness is not run.
>
> Each page task (2.3–2.8) also reads its pages against "How the pages are written":
> one question per page; tasks before detail; only what a user meets; the window's
> own words.

**2.1 — Help → ALIS documentation (`alis/dashboard/helpdocs.py`, `actions.py`,
`qt/window.py`, `tests/`; DQ.12, DQ2.2).**
- The module, the action, the handler and the tests of "Help → ALIS documentation"
  above. The anchors' test waits for the Fit page (2.6).
- `CHANGELOG.md`: under Added.
- **Check:**
  - with `gui-dev`, headless: the Help menu lists ALIS documentation, Keyboard
    shortcuts and About ALIS; on each tab and sub-tab, the action asks to open the
    address of its page (`open_url` replaced, so that no browser opens);
  - the address of a development version, and of a release, as DQ2.2 decides;
  - `test_gui_by_name.py` passes;
  - the `unit` and `gui` batches pass, also under CI's versions, with the new tests
    added and no other change in their counts.

**2.2 — `screenshots.py` (`doc/source/screenshots.py`,
`doc/source/_static/screenshots/`, `doc/source/conf.py`, `tests/test_docs.py`;
DQ.7).**
- The script of "`doc/source/screenshots.py`" above, with its helpers and the run of
  Your first fit. In this task it makes `first_fit.json` and the Introduction's
  figure. Each page task adds its own images.
- `conf.py`'s `html_static_path`; the images' test.
- **Check:**
  - the script runs from a fresh clone's tree, off-screen, and writes every file of
    `SHOTS`. Run again, it writes the same images: same size, and the same pixels
    apart from what a clock or a run's time shows;
  - nothing it reads is under `context/`;
  - the test fails on a scratch copy with a page whose image is missing, and with
    an image in the folder that `SHOTS` does not name.

**2.3 — Introduction and Installing (`doc/source/user/start/introduction.md`,
`installing.md`; `doc/installing.rst` removed; DQ2.1).**
- The two pages of "The pages" above.
- **Check:**
  - every command on Installing is run in a fresh scratch venv: the install, `alis
    --help`, and the example's fit;
  - the extras' table matches `pyproject.toml`;
  - nothing else links to `doc/installing.rst` (`git grep`), apart from the v1.0
    manual (`install.tex`).

**2.4 — The GUI at a glance, and Data (`doc/source/user/gui/overview.md`,
`data.md`).**
- The two pages, and their images.
- **Check:** each action and dialog the two pages name, seen in the window with
  `gui-dev`, has the name the page gives it. Each image shows what its paragraph says.

**2.5 — Regions and Components (`regions.md`, `components.md`).**
- The two pages, and their images.
- **Check:** as 2.4.

**2.6 — Fit and Figure (`fit.md`, `figure.md`).**
- The two pages, and their images.
- Every finding of the pre-flight check (`alis/dashboard/preflight.py`), and every
  warning that Warnings and options to resolve… offers a fix for
  (`alis/dashboard/results.py`), is described with what to do.
- `test_docs.py`: the Help link's anchors are headings of the Fit page.
- **Check:**
  - as 2.4;
  - each finding and warning on the page is quoted as the window shows it, made
    with the synthetic fits that the GUI's tests use for it;
  - the anchors' test fails on a scratch copy with the Results heading renamed.

**2.7 — Earlier fits and several spectra (`earlier-fits.md`).**
- The page, and its images.
- **Check:** as 2.4. A snip's columns and file name are as `loadfit.py` and
  `snips.py` read and write them.

**2.8 — Your first fit (`doc/source/user/start/first-fit.md`, `tests/test_docs.py`;
DQ2.3).**
- The page of "The pages" above, its images, and the test that each number of
  `first_fit.json` is quoted on it, as the page rounds it.
- **Check:**
  - the steps, followed in the window with `gui-dev` from the shipped spectrum, give
    the page's numbers;
  - the command-line half, run as the page says, gives them too (DQ2.3 (a)), or the
    example's own numbers (DQ2.3 (b));
  - the test fails on a scratch copy with one number changed.

**2.9 — The completeness test (`tests/test_docs.py`).**
- The test of "The completeness test" above.
- **Check:** it passes; each scratch fault of its design fails it.

**2.10 — The review.** A claude.ai page, built with the `artifact-design` skill. It
gives:
- the pull request's preview, with a link to each of the ten pages on it;
- what to look for on each: its tasks, its images, and whether a new user could
  follow it;
- the Help link, shown on each tab;
- the stage's open questions, if any.

RJC's comments arrive as threads on it, and each is answered or applied before the
close.

**2.11 — The close.**
- The stage's checks:
  - the site builds with no warning, here and on the pull request's preview;
  - the ten pages have no `% Written in:` line, and the other 26 keep theirs;
  - every tab and sub-tab has its page and its Help link, and the images are made by
    `screenshots.py` alone, from the shipped example and synthetic data;
  - the completeness test passes;
  - Your first fit's numbers are `first_fit.json`'s;
  - no text a user sees says "dashboard" or "RJC";
  - the `unit` and `gui` batches pass.
- CI passes on every job after RJC's last push to the branch.
- The log has the stage's counts (pages written, images and their size, tests,
  batches).
- Documentation Stage 3's document (`documentation_stage3.md`), drafted from the
  plan and this stage's log, with its queries.

RJC then merges the pull request into `alis_v2`. Stage 3's first prompt checks that
`alis_v2`'s hidden version built the merge, and that CI passed on it.

## Skills to use for this stage

- `build-docs`: each task's build.
- `gui-dev`: the window, seen for every page; the Help link (2.1); the images
  (2.2–2.8).
- `run-example`: Your first fit's command-line half, and Installing's check.
- `run-tests`: `tests/test_docs.py` after each task, and the `unit` and `gui`
  batches in 2.1 and 2.11.
- `gen-tests`: `tests/test_dashboard_helpdocs.py` and the completeness test.
- `test-coverage`: `helpdocs.py` and the Help link's handler.
- Claude Code's `artifact-design` skill: the review page (2.10).

## Context

- `claude_prompts/documentation_stage0.md`: the outline of the User guide (pages
  1–10), "How the pages are written", "The dashboard's screenshots", and DQ.7, DQ.9,
  DQ.12 and DQ.15.
- `claude_prompts/documentation_stage1.md` and its log: the addresses, the pages'
  sentences of purpose, `tests/test_docs.py`, and the GUI by name.
- `doc/ALIS_workflow.md` §4.1 (the worked example), §4.2 (bundles), §4.3 (the GUI)
  and §4.4 (pixels fitted twice).
- `alis/dashboard/actions.py` (every menu item and tab action), `alis/dashboard/qt/`
  (the tabs and the dialogs), `preflight.py`, `results.py`, `loadfit.py` and
  `snips.py`.
- `claude_prompts/ALIS_v2_dashboard_prompts.md` (the decisions, D1–D153), and the GUI's
  stage documents and logs, to find what a tab does and why.
- `doc/dashboard/stage11–14/take_screenshots.py` (the helpers),
  `tests/dashboard_fits.py`, `tests/dashboard_synthetic.py` and
  `tests/dashboard_qt_helpers.py`.
- `examples/metal_line_abs/` (Your first fit), `pyproject.toml` (Installing),
  `alis/scripts/dashboard.py` (the `alis` command).

## Queries

*Raised by Claude with this draft (2026-11-27, Documentation Stage 1's Task 1.9).
Each gives Claude's lean.*

**DQ2.1 — How a user installs ALIS.** Installing must give a command that works.
Three facts shape it:
- **The name `alis` on PyPI is another project's:** "ALIS API Python Client",
  version 0.0.5, from `alisista/py-alis`. `pip install alis` would install that
  package, not this one. ALIS itself is not on PyPI.
- **The examples are not in the package.** `pyproject.toml` ships `alis/` and
  `alis/data/` alone. Your first fit needs the example's spectrum, and Installing's
  check needs the example.
- **Until the merge, `master` holds v1.** The site becomes public at the merge, so a
  command that installs from the repository's default branch is right from then on.

The options:
- **(a)** Clone the repository, then `pip install ".[gui]"` in it, adding `gpu` if
  wanted. The clone brings the examples. Updating is `git pull`, then the same `pip
  install`. Whether ALIS goes on PyPI, and under which name, is decided before the
  release (Stage 5).
- **(b)** `pip install "alis[gui] @ git+https://github.com/rcooke-ast/ALIS"`, with
  no clone. Your first fit then links to the example's spectrum on GitHub, to
  download it. Updating is the same command with `--upgrade`.
- **(c)** Publish ALIS on PyPI now, under another name (for example `alis-fitting`,
  as on readthedocs), while `import alis` stays the same. That is a change to
  `pyproject.toml`'s `name` and to the two messages that name the extras, and a
  release step for RJC. Installing then says `pip install "alis-fitting[gui]"`.

My lean: (a). It installs exactly what the site describes, with the examples that
Your first fit and the Examples page use, and it needs no change to ALIS. PyPI's name
can be settled at the release, when there is a version to publish. If RJC chooses
(c) then, Installing gains one line.

**DQ2.2 — Which version's pages the Help link opens.** ALIS's version is
`2.0.0.dev0` now. readthedocs builds `latest` from `master`, and a version for each
release tag once it is activated (DQ.6).
- **(a)** The installed version's own pages:
  - a development version (`.dev` in its number) opens `latest`;
  - a release opens the version of its tag (`/en/v2.0.0/`).

  Each release is then tagged `v` and its number (`v2.0.0`). An automation rule on
  readthedocs can activate each new tag, so that no step is forgotten.
- **(b)** A release opens `stable` (the newest release), and a development version
  opens `latest`. No tag form is needed, but a user of an older release may read
  pages that describe a newer GUI.
- **(c)** Every version opens `latest`.

With any of the three, until `alis_v2` is merged into `master`, `latest` builds nothing
(`master` has no configuration). The link of a development version then reaches a
missing page. The `gui` test checks the address, not the page.

My lean: (a). The page then describes the GUI the user has, and the pages' addresses
never change from one version to the next (Stage 1). If (a), please confirm the
tags' form, `v2.0.0`. The only tag now is `Release`.

**DQ2.3 — Your first fit from the command line.** The plan has the page's second
half run the example's own model, `run_alis fit_spectra.mod`. That model is written
by hand: one fit range over both lines, the b-values tied by a label, and its own
starting values. The fit made in the GUI in the page's first half is a different
model, with a snip for each line and the GUI's defaults, so the two halves give
different numbers.
- **(a)** The second half runs the project that the first half saved:
  `run_alis first_fit.model`. It is the same fit, so it gives the same numbers. It
  shows that a project made in the GUI runs anywhere, such as on a cluster.
  `run_alis --extract` then shows its model file, with a link to The model file. The
  example's `fit_spectra.mod` is described on the Examples page (Stage 5).
- **(b)** As planned: `run_alis fit_spectra.mod`, with its own numbers (from the
  example's reference output), and a sentence on why they differ from the GUI's.

My lean: (a). One fit, start to finish, is easier to follow for a new user, and it
needs only the spectrum. With (b), the reader has two fits and two sets of numbers
on their first page.

## Prompts

> RJC will be responsible for writing this section.
