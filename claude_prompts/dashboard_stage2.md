# Prompt file for ALIS software dashboard creation -- STAGE 2

> **The project model, with no Qt.** Everything the dashboard knows about a project,
> held in plain Python and tested with pytest, before any window exists:
> - the text-sync layer, which maps the parts of a project (datasets, snips,
>   systems, components, continua) to their `.mod` lines and makes targeted edits
>   (D7), tested by opening every model and writing it back unchanged (QF.5);
> - opening an existing fit (F4, D12), including the notices for imported models that
>   break D20 or D22 (Q0.3);
> - the validator (F5), the blinding gate (F8), undo/redo (F2), removal with
>   dependents (S27, D23), and the mode interface with Voigt mode only (F11).
>
> The code goes in a new package, `alis/dashboard/`, built on `alis/bundle.py` (Stage
> 1). Nothing in it imports Qt or pyqtgraph, and a test checks that. ALIS outside
> `alis/dashboard/` does not change, unless a query agrees otherwise. The regression
> harness must stay green.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`; "Q0.*n*" and "Q1.*n*" to
> `dashboard_stage0.md` and `dashboard_stage1.md`. The plan for all stages is in
> `dashboard_stage0.md`.

## Design

*Written by Claude on 2026-10-05, from the design documents and what Stages 0 and 1
taught. The open choices are the Queries below; each gives Claude's lean, and this
section follows the leans.*

### Four layers

```
Project            the dashboard's concepts: datasets, snips, systems, components,
  │                continua, sources; built from the layers below plus ui/project.json
Parsed model       a read-only structure rebuilt after every edit: settings, data
  │                lines, model lines, fix/lim commands, links, and a table of labels
Model text         the .mod as lines and tokens, each with its position; the only
  │                thing ever edited (D7)
Bundle             alis/bundle.py (Stage 1): the model text, the snips, hidden
                   lines, sources, runs and ui/ members
```

- **The text is authoritative** (D7, QF.3). Every action, whether typed or done in a
  panel, becomes a patch to lines of the text. The parsed model and the project are
  rebuilt from the text after each patch, so they can never disagree with it.
- **Positions.** Every token knows its line and columns, so a panel item can find
  its `.mod` lines (S16, cross-highlighting) and an error can be shown on its line
  (F5).
- **A parse error pauses panel edits** until the text parses again (QF.3). Typing in
  the `.mod` panel always works.
- **Hidden lines.** The project works on the restored text in memory. Whenever it
  writes the model back to the bundle, it hides the lines again: every line with
  `blind=True`, and, under `run blind True`, any line holding best-fit values copied
  in (S20, QF.20(a)). The blinding gate decides what reaches the screen.

### How the dashboard's concepts map to the `.mod`

| Concept | In the model text | Elsewhere |
|---|---|---|
| File row (D35) | the data lines whose snips came from the file; FWHM is `resolution=vfwhm(x<label>)`, shift is `shift=vshift(x<label>)` | the source file, its checksum and column roles in `sources.json` and `ui/project.json` |
| Tied rows (D35) | the same label on the rows' FWHM, shift or zero level | |
| Reference row (D14) | its shift fixed at 0 (or no shift) | flag in `ui/project.json` |
| Zero level (D14) | a `constant` in the `zerolevel` block, with the file's specids | |
| Snip | one data line, with its own `specid` | the snip file in the bundle, whose fit-mask column holds the regions |
| Region (S6, D36) | `fitrange=[lo,hi]` on the data line, when the line has one region (Q2.10) | the snip's mask column (Q2.7) |
| Continuum (D17) | a `legendre` line per snip in the `emission` block | |
| System (D15) | — | name, z, primary flag and line IDs in `ui/project.json` |
| Component (D20) | the `voigt` lines of one system that share z, b and T labels, one line per ion | |
| Isotopes (D22) | separate `voigt` lines sharing the element's z, b and T labels | |
| Temperature mode (D21) | the case and value of the T and b labels | |
| Generic absorber (D15) | a `voigt` line with `1Ly_a` or `1H_IB`, in no system | |
| Blinded parameter (D24) | `blind=True` on the line | hidden in the bundle (Stage 1) |
| Limits and links (S15) | `lim`/`fix` lines, and the `link` block | |

### The text-sync contract

- **Reading never changes the text.** Opening a model and saving it unchanged gives
  the same bytes, for every model in `examples/` and `context/fitting_examples/`, and
  every `.mod.out.reference` (QF.5).
- **An edit changes only what it must.** Changing a value replaces that token alone,
  keeping its tie label and the line's other spacing (Q2.6). A new line copies the
  layout of the nearest line of the same kind.
- **Labels are never renamed** unless the user renames them. New parameters get
  labels from one scheme (Q2.3). Column densities get none: when one has to be
  fixed, tied or limited, the user names its label (RJC, Q2.3).
- **Comments, commented-out lines and long comments** (`#-->` … `<--#`) are kept.
  They are never parsed as model content, but an edit never breaks them.

### The services

- **Validator (F5).** It works at two levels:
  - **Every keystroke:** a fast check from the parsed model alone:
    - block structure;
    - unknown functions and ions;
    - a `1e4`-style label;
    - a `specid` with no data line, or with no continuum;
    - tied labels given different starting values (`ALIS_workflow.md` §2.5);
    - a fitrange with no pixels;
    - a buffer narrower than the convolution needs.
  - **After a pause:** ALIS's own loaders, run on a copy with the data in memory
    (Stage 1.2). A handler on the `alis` logger collects their warnings and
    errors and keeps them off the terminal, and the exit of `msgs.error` is
    caught, as `bundle.run` does (Q2.5). An unexpected exception prints its
    traceback to the terminal, and the dashboard says that it has met an
    unexpected error and that the terminal has the details (RJC, Q2.5).

  Each problem has a line, a severity (error or warning) and a message. None of the
  messages refers to the design documents (D44).
- **Blinding gate (F8, D24, QF.20).** One module that every value bound for the
  screen passes through. That covers panels, tables, tooltips, the `.mod` panel,
  messages and plot labels.
  - **Masks:** a hidden value shows as `▒▒▒▒` with its tie label (`▒▒▒▒da`).
  - **Shown:** errors and changes in χ².
  - **Masked:** `blindrange` offsets.
  - **Typing over a mask** stores the typed value hidden.
  - **Unblinding** calls `bundle.unblind` after the user confirms (Stage 3), and is
    logged. It also writes `blind=False` and `run blind False` into the text, so
    nothing is masked again. It cannot be undone, and it clears the undo history
    (Q2.11).
- **History (F2).** One undo/redo history for panels and the `.mod` panel. Each step
  is a command that changes the text, a snip's mask, or `ui/project.json`, and can be
  undone exactly. Typing is grouped into one step per pause, and a drag is one step.
  The history is not saved in the bundle (Q2.8).
- **Removal (S27, D23, QF.28).** Removing a component, ion, snip, system or dataset
  first produces a plan, which the user sees as a list of the `.mod` lines that will
  change, and then applies it as one undoable step. The plan covers:
  - the lines to delete;
  - specids to drop from other lines;
  - `lim`/`fix param` lines and links that use a removed label;
  - ties left with one member.
- **Opening an existing fit (F4, D12).** `bundle.pack` (Stage 1) makes the bundle. The
  project then infers its datasets, systems and components (Q2.4), and lists, with a
  one-click fix, each imported component that breaks D20 or D22. The model may still
  be fitted as imported (Q0.3). An imported model also gets a general notice asking
  the user to check that its components were read correctly (RJC, Q2.4). Without
  source spectra, regions are limited to each snip's extent (D12).
- **Regions (Q2.7, Q2.10).** A region edit changes only the mask column of the
  affected lines of a snip. A change of the snip's edges writes a new snip file over
  the old one (RJC, Q2.7). A data line with `fitrange=[lo,hi]` keeps that form while
  it has one region; a second region switches it to `fitrange=columns` and rewrites
  its snip with a mask column.
- **Modes (F11).** A mode supplies five things:
  - a data loader;
  - a builder for the display spectrum;
  - a mapper from regions to fitted data;
  - a model template;
  - plot presets.

  Voigt mode is the only one built now; Orders mode comes after v1.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage2_log.md`.
> After every task, run the `unit` batch and the new dashboard tests; run the `fast`
> batch before the stage closes.

**2.1 — The package (D3, Q2.1). [DONE 2026-10-05]**
- Create `alis/dashboard/`, with the modules of Q2.1, each with a docstring saying
  what it holds.
- Declare the `gui` extra in `pyproject.toml` (PySide6, `qtpy`, pyqtgraph; D2). This
  stage does not use it.
- **Check:** a test imports every module of `alis/dashboard/` in a subprocess and
  fails if `qtpy`, `PySide6`, `PyQt6` or `pyqtgraph` is in `sys.modules` afterwards.

**2.2 — The model text (`text.py`). [DONE 2026-10-05]**
- Split a model into lines and tokens, as `load.load_input` splits it:
  - the settings, data, model and link blocks;
  - comments and long comments;
  - `fix`/`lim` lines;
  - values with their labels.
- Every token records its line and columns.
- Patches replace a token, a line or a range of lines, and are reversible.
- **Check:** a round-trip test over every `.mod` and `.mod.out.reference` in
  `examples/` and `context/fitting_examples/`, which must give the same bytes; and
  property tests showing that a patch changes only its own span.

**2.3 — The parsed model (`model.py`). [DONE 2026-10-05]**
- From the text, build the following, each knowing its line and token positions:
  - settings, data lines (file, specid, fitrange, resolution, shift, keywords);
  - model lines by block (emission, absorption, zerolevel, variable), with function,
    ion, parameter values, labels and keywords;
  - `fix`/`lim` commands, and links;
  - a table of labels: each label's occurrences, and whether it is free, fixed or
    tied.
- **Check:** for every model, the parsed model agrees with ALIS's own parser (with
  the data loaded from memory) on:
  - the number of parameters;
  - which parameters are tied together;
  - the number of free parameters;
  - each parameter's line (`modpass['line']`).

**2.4 — Targeted edits (`edit.py`). [DONE 2026-10-05]** Operations used by the panels:
- set a value;
- set a parameter free, fixed or tied (S11), by writing the label's case and suffix;
- set and clear limits (S15);
- add and remove keywords (`blind=True`, `specid=…`);
- add a line of a kind (a component's ion, a continuum, a zero level, a data line);
- set a dataset's FWHM, shift and zero level, and tie rows (D35).

Each returns a patch, and new labels follow Q2.3.
- **Check:** unit tests for each operation, on real models. Each must leave every
  other byte of the text unchanged, and give a model that ALIS's parser reads with
  the intended structure.

**2.5 — The project (`project.py`). [DONE 2026-10-05]**
- Build the dashboard's concepts from the parsed model, the bundle and
  `ui/project.json` (Q2.2): files and datasets, snips, systems, components (D20),
  isotopes (D22), temperature modes (D21), continua and generic absorbers.
- Opening an existing fit infers what `ui/project.json` does not yet say (Q2.4), and
  lists the D20/D22 notices with their fixes (Q0.3).
- Regions: read and edit a snip's fit mask (Q2.7).
- **Check:** tests on J1358p6522 and Q1243p307 (the Stage 0 mockup fits). They must
  give the systems, components, isotopes and the three Q1243p307 datasets that the
  mockups show. Also tests on a model that breaks D20 and one that breaks D22.

**2.6 — The validator (`validate.py`, F5). [DONE 2026-10-05]**
- Fast checks from the parsed model, and the full check with ALIS's loaders (Q2.5),
  as in the Design section.
- **Check:** a test for each example in F5's list, each placing its problem on the
  right line. Every model in `examples/` gives no errors.

**2.7 — The blinding gate (`blinding.py`, F8, D24, QF.20). [DONE 2026-10-05]**
- Masks, errors and χ² changes shown, typing over a mask, `blindrange` masked, and
  unblinding through `bundle.unblind` with a log entry.
- **Check:**
  - a test renders every value a panel can show, for blinded versions of
    `examples/blind` and of the D/H fit `DH/J1358p6522_original` (with
    `blind=True` on `dhrand`), and finds no hidden value in any string;
  - a test confirms that a hidden line is hidden again when the model is written
    back.

**2.8 — History (`history.py`, F2). [DONE 2026-10-05]**
- Commands that change the text, the snips' masks and `ui/project.json`, with undo
  and redo, typing grouped by pause, and drags as one step.
- **Check:** property tests. Any sequence of operations, undone in full, gives back
  the original bytes, and redone gives the same result again.

**2.9 — Removal (`remove.py`, S27, D23). [DONE 2026-10-05]**
- Plans for removing a component, ion, snip, system and dataset, as in the Design
  section. Each plan lists the lines it changes, and is applied as one step.
- **Check:** tests on `DH/J1358p6522/model/J1358p6522_original.mod`, whose D/H is
  linked through the variable `dhrand` (removing D I there must remove or rewrite
  its links), and on Q1243p307 (removing a dataset removes its snips, zero level and
  shift). After each removal ALIS's parser reads
  the model without error, and undo gives back the original.

**2.10 — Modes (`modes.py`, F11). [DONE 2026-10-05]**
- The mode interface, and Voigt mode: one fitted dataset per display spectrum,
  regions mapped straight to the snip's mask, and a model template for a new project.
- **Check:** a new Voigt-mode project, built from `J1358p6522_fluxcal.dat` with one
  system and one transition, gives a model that ALIS runs.

**2.11 — Close the stage. [DONE 2026-10-05]**
- Rewrite the `gui-dev` and `gui-component` skills for PySide6 and pyqtgraph, as
  carried forward from Stage 0 for use in Stage 3.
- Run `test-coverage` on `alis/dashboard/`.
- Record in this document what Stage 3 receives.
- Update `CHANGELOG.md`, and the stage table in `dashboard_stage0.md` if anything
  moved.

## Status (2026-10-05)

*Written by Claude at the end of Prompt 1. The details are in
`claude_prompts/logs/dashboard_stage2_log.md`.*

Tasks 2.1–2.11 are done.
- **Tests.** Ten new test files, `tests/test_dashboard_*.py`: 563 pass and 35 are
  skipped here (context models that ALIS does not read on its own). Coverage of
  `alis/dashboard/` is 94%. All but two of the tests are in the `unit` batch; the two
  that run a fit are in `fast`.
- **The batches at the close:** `unit` 1418 passed, 0 failed; `fast` 110 passed,
  0 failed (13 min 41 s).
- **ALIS outside `alis/dashboard/` is unchanged.** The other changes are
  `pyproject.toml` (the `gui` extra, and `hypothesis` in `dev`), `.gitignore`
  (`.hypothesis/`), the two skills, `CHANGELOG.md`, and the stage table of
  `dashboard_stage0.md` (S16 and F12 now have their logic in Stage 2; the user's
  preferences file is in Stage 3).
- **Found along the way:**
  - two small faults in ALIS (Q2.12), worked around in the dashboard;
  - an inferred system could take its redshift from a hidden component, which every
    other component's velocity would then reveal; it now takes it from a component
    that is not hidden;
  - ALIS's fit workers are spawned, so a script that starts a fit must be a file with
    a main guard (not `python -`).

### What Stage 3 receives

The windows of Stage 3 are built on `alis/dashboard/`, which imports no Qt:
- **Projects.** `Project.open(path)`; `Project.import_model(fit.mod)` (F4: packs it
  with `bundle.pack` and infers its structure); `modes.get("voigt").new_project(source,
  z, [(ion, rest)], ...)` (a new project from a spectrum); `project.save(path)`
  (atomic, under the lock, keeping the runs: what autosave calls, F1);
  `project.to_bundle()`.
- **State and history.** `project.text`, `.files`, `.ui` (`ui/project.json`),
  `.parsed` and `.paused`. Every change is a `Step`. `History(project)` has `edit`,
  `do`, `type` (typing grouped by pause), `group` (a drag), `undo`, `redo`, their
  descriptions for the menus, and `clear`.
- **What the tabs show.** `project.rows` (the Data tab's rows, with FWHM, shift, zero
  level, reference and source); `project.snips` (pixels, extent, regions, continuum);
  `project.systems` and `.components` (ions, isotopes, temperature mode, velocity);
  `.absorbers` (the Lyα forest) and `.others`; `.notices`, each with a one-click fix
  (`project.fix_notice`); `project.structure()` and `confirm_structure()` for the
  user to confirm what was inferred.
- **Edits.** `edit.py` for values, free/fixed/tied, limits, keywords, new lines and
  the dataset rows; `project.set_regions` and `set_snip_edges`; `remove.plan_*`, shown
  with `plan.preview(gate)` and made with `history.do(plan.step(project))`. An edit
  that raises `edit.LabelNeeded` asks the user for a label.
- **The `.mod` panel.** `blinding.Gate(project).view()` gives the masked text, and
  `MaskedView.to_real` turns typing into an edit of the real text;
  `project.lines_of(item)` and `project.items_at(line)` cross-highlight (S16);
  `validate.quick(project.parsed, project.data_for_run())` at each keystroke, and
  `validate.check(project)` after a pause, give problems on their lines; for the
  mockup fits each takes less than 0.3 s.
- **Blinding.** Every value shown goes through the `Gate`; `blinding.unblind(project,
  confirmed=True, note=..., history=...)` follows the confirmation dialog;
  `blinding.strings(project)` lists what the panels show, for tests.
- **Tab markers (F12).** `project.model_changed_since_run()`, `project.notices` and
  `project.paused`.
- **Errors.** ALIS's messages are collected by `model.alis_quietly()`, never printed;
  `validate.unexpected(error)` writes a traceback to the terminal and gives the
  message to show (Q2.5).

Points for Stage 3:
- Reading the model again costs 0.03 s (J1358p6522) to 0.4 s (DH_orders), so the
  `.mod` panel should read it after a short pause in typing, not at every key.
- Fits run in another process (F6); ALIS's own worker processes are spawned.
- The user's preferences file (Q2.2) and the view state (`ui/view.json`) are Stage 3's.
- Q2.12 (two small fixes to ALIS) is open.

## Skills to use for this stage

- `run-tests`: the `unit` batch and the dashboard tests after each task; the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of each module.
- `atomic-data`: ions and transitions for the validator and the project (unknown ions,
  isotopes).
- `check-fit`: confirming that edited models still fit as expected.
- `gui-dev` and `gui-component` are rewritten in Task 2.11, not used.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`:
  - D7, D12, D14–D24 and D34–D41;
  - QF.3, QF.5, QF.8, QF.20, QF.25, QF.28 and QF.33;
  - F2, F4, F5, F8, F11, S11, S15, S16 and S27.
- `claude_prompts/dashboard_stage1.md`: "What Stage 2 receives", and Q1.2 (hidden
  lines).
- `claude_prompts/dashboard_stage0.md`: Q0.3 (imported models that break D20 or D22),
  Q0.5 (several files per dataset).
- `doc/ALIS_workflow.md` §2: the model file's syntax, labels, `fix`/`lim`, links.
- **The code:**
  - `alis/bundle.py`;
  - `alis/load.py`: `load_input`, `load_data`, `load_model` (and its
    `modpass['line']`), `load_links`;
  - `alis/functions/base.py`: `check_tie_label`;
  - `alis/save.py`: `print_model`, for how values are formatted.
- **The tests:** `tests/test_writer_round_trip.py` (every reference model, re-read)
  and `tests/test_bundle.py`.
- **The mockups:** `doc/dashboard/mockups/build_mockups.py`. Its `load_j1358`,
  `load_q1243` and `load_dh_run` already read the two fits into the structures the
  panels need.

## Queries

*Raised by Claude on 2026-10-05, while writing this document. Each gives Claude's
lean, and the Design section and the tasks follow the leans.*

**Q2.1 — The package's modules.** I propose these modules in `alis/dashboard/`:
- `text.py`, `model.py` and `edit.py` (the text-sync layer);
- `project.py`;
- `validate.py`, `blinding.py`, `history.py` and `remove.py`;
- `modes.py`.

Stage 3 adds the Qt modules beside them, in a `qt/` subpackage. Only that
subpackage imports Qt.

My lean: as proposed.

**Response:** I agree with the proposed modules in `alis/dashboard/`. The separation of concerns is clear, and it will facilitate testing and maintenance. The addition of a `qt/` subpackage in Stage 3 for the GUI components is a good approach to keep the core logic independent of the user interface.

**Q2.2 — Where the dashboard's own project data are kept.** Some of the project is
not in the model text:
- systems: their names, redshifts, which is primary, and their line IDs;
- which files form a dataset, and which is the reference;
- each source's column roles.

The bundle already keeps `ui/` unchanged for the dashboard. I propose
`ui/project.json` for these project data. The view state (tab, zoom, column count)
would go in `ui/view.json` in Stage 3, so that the project data can be versioned and
tested apart from the view.

My lean: as proposed.

**Response:** It is not clear to me what data this refers to. If anything is related to the user input (e.g. absorber names, redshifts, the primary system, etc.), then it makes sense for this to be stored in a file created by the user. For example, when the user opens the dashboard, it would be a blank space where they can "Create New Project" or "Open Existing Project". If they create a new project, then the dashboard would create a new file to store the project data (this would then be included in the bundle). If they open an existing project, then the dashboard would read a bundle file to load the project data. This way, each user can manage their own projects independently of the dashboard's internal logic. If I have misunderstood the intent, please clarify what data is being referred to and how it is expected to be used by the code. If there are code defaults/settings that can be edited by the user for their preferences (e.g. how many columns to display by default, or the number of CPUs they wish to use for fitting), then those should be stored in a separate configuration file (e.g. `config.json`) that is not part of the project data. In this case, there would be a set of default settings, and then a user-specified settings file that overrides the defaults. This would allow users to customize their experience without affecting the core functionality of the dashboard.

**Q2.3 — Labels for new parameters.** ALIS ties parameters by labels (§2.3), and new
components, continua and datasets need new labels. I propose short, systematic
labels that tell the user what they belong to:
- **components:** `z`, `b` and `t` plus the system's letter and the component's
  number, for example `za1`, `ba1`, `ta1`;
- **datasets:** `fwhm` and `vs` plus the row's number;
- **column densities:** no label until one is needed (to fix, tie or limit one),
  then `n` plus the ion and the component, for example `nhi1`.

Labels never begin with `e` followed only by digits, and existing labels are never
renamed.

My lean: as proposed. You may prefer other names; this is the place to say.

**Response:** I agree with everything proposed, however, there may be some issues with the proposed column density labels. For example, if the user wants to fit Si II and S III, this would both show up as `nsiii1` according to the proposed ion+component labelling. It is uncommon that column densities would be tied, so I would propose that at this stage, we do not assign any labels to the column densities, and allow the user to define their own column density labels.

**Q2.4 — Inferring structure from an imported model.** An existing `.mod` says
nothing about systems or datasets. I propose:
- **Datasets** are the groups of data lines that share a resolution label. Lines
  with no shared label but the same file-name stem before the transition are one
  dataset per stem.
- **Systems:** the absorber redshifts are grouped where they lie within ±500 km/s of
  each other. The group with the most ions is the primary. `1Ly_a` and `1H_IB` are
  generic absorbers.
- **Components** are the groups of `voigt` lines in a system that share a z label.
  Untied lines with the same z (within 0.1 km/s) are offered as one component, with
  a D20 notice.

Everything inferred is shown for the user to confirm or change (Stage 4) before it
is saved in `ui/project.json`.

My lean: as proposed, tested on all context models in Task 2.5.

**Response:** This sounds fine, but perhaps we should also provide a warning to the user that some components may not have been correctly loaded, and that it is best to check the components have been loaded correctly. It's more important that the experience is smooth for users that use the dashboard for their entire workflow. Allowing the dashboard to load a `.mod` file will not be a common use case. 

**Q2.5 — The full check uses ALIS's loaders as they are.** ALIS stops on an error
with `msgs.error`, which exits. The validator can catch that exit and keep the
message, as `bundle.run` does in Stage 1, and find the line from the text the
message quotes.

The alternative is to change `msgs.error` to raise a typed exception that carries
the line. That is cleaner, but it changes ALIS itself, and many errors do not know
their line.

My lean: catch the exit, with no change to ALIS. Messages that quote no line are
shown against the whole model.

**Response:** ALIS now uses a new logging system (see `logger.py`). Is there a way that we can utilise this new logging system to catch errors and warnings, and print the traceback to the user (either in the terminal they run ALIS from and a message to screen saying that the ALIS dashboard has encountered an unexpected error, please refer to the terminal window for further details about the error)?

**Q2.6 — Spacing when a value changes length.** Many models align their columns.
When `13.0` becomes `13.04821`, I propose to keep the next token's column if the gap
allows it, and otherwise to keep one space. New lines copy the layout of their
neighbour.

My lean: as proposed.

**Response:** I agree with this approach.

**Q2.7 — Editing regions changes the snip file.** Regions live in the snip's
fit-mask column (D4: the snip format does not change). Rewriting a snip with
`np.savetxt` would reformat every number, so the bytes, and possibly the last digit
of the data, would change.

I propose that a region edit changes only the mask column of each affected line,
character for character, and leaves the wavelength, flux and error text untouched.
A new snip, cut from a source spectrum in Stage 4, is written in `prepfit`'s
four-column format at full precision.

My lean: as proposed.

**Response:** If I understand correctly, you are concerned about the machine precision of printing the data. The only issue is that the size of the snip could be changed, as well. Your proposal to only adjust the mask column of the snip file is a good approach, provided that either of the snip edges are not changed. If the user wants to change the edges of the snip, then we should allow them to do so, but this would require a new snip file to be created (writing over the old snip file).

**Q2.8 — Saving the undo history.** I propose that the history lives only while the
project is open, and is not saved in the bundle. Autosave (F1, Stage 3) keeps the
work, and a saved history would grow without limit.

My lean: not saved.

**Response:** I agree, the undo history should not be saved in the bundle. The autosave feature will ensure that the user's work is preserved, and saving the history would indeed lead to unnecessary growth of the bundle size.

**Q2.9 — What else belongs in Stage 2.** Two small pieces of logic need no Qt:
- the item-to-line map behind cross-highlighting (S16), which falls out of the token
  positions;
- the "model changed since the last run" test behind the Fit tab's marker (F12),
  which compares the model with `run.json`'s `model_sha256` (Stage 1).

I propose to build both here, and their display in Stage 3.

My lean: yes.

**Response:** I agree with the proposal to build the item-to-line map and the "model changed since the last run" test in Stage 2. These features are essential for the functionality of the dashboard and can be implemented without relying on Qt. By building them in Stage 2, we can ensure that they are well-tested and integrated into the core logic of the dashboard before moving on to the GUI components in Stage 3.

*Raised by Claude on 2026-10-05 (Prompt 1), after reading the responses above. Claude's
answers to RJC's notes come first, then three new queries, which RJC answered in the
session. The Design section follows all of them.*

**Notes on the responses.**
- **Q2.2.** The project data are what you describe: what the user enters in the
  dashboard that the `.mod` cannot hold. These are the systems (name, z, the primary
  flag, line IDs), which data lines came from which file row, the reference row, and
  each source's column roles. The dashboard writes them to `ui/project.json` when the
  project is created, and that file lives inside the bundle. Your user preferences
  (shipped defaults, overridden by a user file) are a separate file outside the
  bundle. RJC chose to build that file in Stage 3, where its first users are; Stage 2's
  model template takes its defaults as arguments.
- **Q2.3.** Column densities get no label. When one has to be fixed, tied or limited,
  the dashboard asks the user to name its label; it never makes one up. The scheme
  still names z, b and T (`za1`, `ba1`, `ta1`) and the datasets' FWHM and shift.
- **Q2.4.** An imported model gets a general notice asking the user to check that its
  components were read correctly, besides the D20/D22 notices.
- **Q2.5.** Yes. `msgs` is a `logging.Logger` since refactor Stage 2.5, so the
  validator attaches its own handler to the `alis` logger while ALIS's loaders run. The
  handler collects every warning and error, with no change to ALIS, and keeps them off
  the terminal (otherwise each pause would print ALIS's start-up messages). The exit of
  `msgs.error` is caught, and its message is placed on the line it quotes. Any other
  exception is a bug: its full traceback is written to the terminal through the same
  logger, and the dashboard says that it has met an unexpected error and that the
  terminal has the details.
- **Q2.7.** Agreed: a region edit changes the mask column alone. When the user moves a
  snip's edge, the snip file is written again. Moving an edge inwards cuts the snip
  from its own lines, keeping their text. Moving it outwards needs the source spectrum
  (Stage 4, or S23 for an imported fit).

**Q2.10 — Regions of data lines with no mask column.** 66 of the 108 `fitrange` settings
in `examples/` and `context/` use `fitrange=[lo,hi]`, not a mask column: every shipped
example, and many context lines. How should a region edit on such a line work? Lean:
one region edits the `fitrange=[lo,hi]` token in place. A second region switches the
line to `fitrange=columns` and rewrites the snip with a mask column, in `prepfit`'s
four-column format at full precision, as Q2.7's response allows when the edges change.

**Response (RJC, in the session):** As the lean: keep the form, and convert when needed.

**Q2.11 — What unblinding does.** `bundle.unblind` restores the hidden lines and logs
the time. But the text still says `blind=True` (and perhaps `run blind True`), so the
values would be masked again at once. Unblinding should therefore also write
`blind=False` and `run blind False`. Can it be undone? Lean: no. It is logged in the
bundle and clears the undo history, because what has been seen cannot be unseen.

**Response (RJC, in the session):** As the lean: it cannot be undone.

*Raised by Claude on 2026-10-05, at the end of Prompt 1. It does not block anything:
Stage 2 works around both points without changing ALIS.*

**Q2.12 — Two small faults in ALIS, found while building the parsed model and the
validator.**
- **(a) Reading a value strips the label's characters.** Every function's
  `check_tied_param` reads a value as `float(word.rstrip(label))`, and `rstrip`
  removes *characters*, not the label. When the label holds a digit that the value
  ends with, the digit goes too: `11.2878481n1a` is read as `11.287848`. Among all
  the models this happens twice, both in `examples/CNabs/model/fit_spectra.mod.out.reference`,
  so re-reading that output loses a digit of two column densities (one of them
  linked). The dashboard keeps ALIS's reading, and its validator warns about such a
  label.
- **(b) `Afwhm.getminmax` treats its width as a fraction.** It returns
  `fitrange × (1 ± Nsig·σ)` with σ in Ångström, so a FWHM of 0.1 Å asks for ±40% of
  the wavelength. ALIS then only loads more data than it needs (in
  `examples/lsf_file`), but the same numbers would make the `bufferpix` warning
  wrong. The validator uses the additive width, `fitrange ± Nsig·σ`, for `Afwhm`.

Both fixes change ALIS outside `alis/dashboard/`: (a) by reading the value as the
word without its label (`word[:len(word) - len(label)]`), in one helper shared by the
~25 copies of `check_tied_param`; (b) by making the range additive. The regression
harness should stay green: (a) changes two values of one reference by 10⁻⁷, and (b)
changes only how many unfitted pixels are loaded. Should they be made, and when?

My lean: make both, each with a unit test, as a small task of their own before
Stage 3 (or at its start), run against the `unit`, `fast` and `medium` batches.

**Response:** I agree with both of the proposed fixes to ALIS. Let's carry them out now before beginning the stage 3 development.

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-1; see their design documents (`dashboard_stage0.md` and `dashboard_stage1.md`) and the logs (`dashboard_stage0_log.md` and `dashboard_stage1_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

2. Good job with the stage 2 development. I have responded to Q2.12, and I agree with your proposed fixes to ALIS. Please implement these fixes now before proceeding to stage 3 development.

