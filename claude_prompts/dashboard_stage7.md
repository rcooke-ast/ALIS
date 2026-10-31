# Prompt file for ALIS software dashboard creation -- STAGE 7

> **Changes and bug fixes from real use.** Version 1 of the dashboard (Stages 1–6) is
> complete. This stage takes RJC's list of what needs to change after fitting real
> problems with it: each item of the list becomes a task, with its check, built in
> the order of the list unless a query says otherwise.
>
> The rules of Stages 3–6 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> model is one step of the history. ALIS outside `alis/dashboard/` changes only where
> a query agrees.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`. "Q0.*n*" to "Q6.*n*" refer to
> `dashboard_stage0.md` to `dashboard_stage6.md`. What follows v1 is listed in
> `dashboard_stage0.md` ("After v1").

## Changes and bug fixes

*RJC's list. One item per point: what happens now, and what should happen instead.
A bug is easiest to fix with the project (or the model and data) it happened with,
and the steps that led to it. Mark an item **(bug)**, **(change)** or **(new)**.*

1. **(new)** Allow the dashboard to load a `.mod` file and multiple snips from an analysis previously made with `ALIS`.
2. **(bug)** When an ion (e.g. `D I`) is tied to another (e.g. `H I`), and then if `H I` is excluded from the fit, the `D I` component is not excluded from the fit. This is the expected behaviour, but there is no way to exclude the `D I` line from the fit, as well. The only way at the moment is to comment out the component. An example of this behaviour is provided in `context/fitting_examples/Temperature/J013301m400628/`.
3. **(change)** The dashboard has two options "import a fit (.mod)..." and "Load a .mod...". I think it's a bit confusing having both of these options. Is there a distinction, and can/should we combine them into one option?
4. **(new)** We should consider adding some documentation that can be included in readthedocs, that describes the dashboard, ALIS, and how to use them. This should include all available functionality of ALIS, and should build on the existing documentation that is already available in the `doc/tex_files/` folder. This should be a separate task, and not part of this stage. Please make a task (to be completed as the final part of this stage) to write the design document to prepare for the documentation.
5. **(bug)** For the example (`context/fitting_examples/Temperature/J101155p294141/`), the component cards list `D I` components as "Excluded" when a ratio (`voigt   ion=2H_I/1H_I ...`) is specified.
6. **(bug)** For the component cards, I sometimes receive the error:
```
alis.dashboard.components.ComponentError: The lines of component a1 do not share a redshift label: tie them first (the notice above the panels).
Traceback (most recent call last):
  File "/Users/rcooke/Software/ALIS/alis/dashboard/qt/components.py", line 1152, in <lambda>
    lambda _=False, c=comp, i=ion, e=excluded: self._toggle_ion(c, i, e)
                                               ~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/Users/rcooke/Software/ALIS/alis/dashboard/qt/components.py", line 1551, in _toggle_ion
    plan = C.exclusion(project, comp, ion)
  File "/Users/rcooke/Software/ALIS/alis/dashboard/components.py", line 1515, in exclusion
    raise ComponentError(
    ...<2 lines>...
    )
```
when clicking the "Exclude" button. I _think_ this happens when there is no letter suffix on the redshift number (e.g. `z=0.123` instead of `z=0.123a`).
7. **(change)** When opening alis on the command line without any other input, the landing page contains a list of recent projects. It would be good to have the ability to clear all recent projects from this list, and also to clear individual projects from the list.
8. **(bug)** On the "Add a spectrum (ascii)" screen. Before selecting a file, everything looks fine. After selecting a file, the column information appears and the text near the bottom that says "Wavelength, flux and error are needed. Continuum..." is blended with the column information.
9. **(new)** Each snip can have a shift parameter. It would be good to include this functionality in the dashboard on the "Regions" tab.
10. **(new)** One the `Fit` tab, there is an `Inspect` subtab with a `View` panel. The view panel always displays the data and model with the continuum normalised. It would be good if there was a way to toggle between showing the continuum normalised or the actual model continuum.
11. **(bug)** On the components tab, the user can drag the mouse so that the logN and bturb values can be adjusted interactively. However, if bturb is a fixed component (i.e. a thermal fit), then dragging the component values in the "Panels" view should adjust the temperature and not the turbulent Doppler parameter. However, if the temperature is fixed, then dragging the component values in the "Panels" view should adjust the turbulent Doppler parameter and not the temperature. If this is a turbulent+thermal component (i.e. both are free), then dragging the component values in the "Panels" view should adjust only the turbulent Doppler parameter. At the moment, all dragging only affects the turbulent Doppler parameter.
12. **(change)** When a component's temperature is set to "pure thermal", the dashboard currently sets the bturb value to 0.0, but the limits don't currently allow this. The limits should automatically be adjusted to allow the bturb value to be set to 0.0 when the component is set to "pure thermal".

## Design

*Written by Claude from the list above, once it is filled in: for each item, what
changes, in which modules, and what stays as it is. Items that need RJC's decision
become queries below, each with a lean.*

### 7.1 — Loading snips and a `.mod` from an earlier analysis

*Written by Claude on 2026-10-30 (Prompt 1), from item 1 of the list, Task 7.1, the
example in `context/fitting_examples/Temperature/J013301m400628/`, and the code of
Stages 2–6. It follows the leans of Q7.1–Q7.5, which RJC accepted, and was brought up
to date with what Task 7.1 built: where the button is, the tolerances of the match,
and the tests.*

**What exists.**
- **Import a fit** (F4, D12, D45, `Project.import_model`): a new project made from a
  `.mod` and its snips, with no source spectrum, so regions stop at each snip's edges
  and SNIP cannot cut new transitions. Attaching a spectrum afterwards (S23) is on
  the "After v1" list.
- **SNIP** (D53, `snips.snip_step`) cuts a snip from a file row's source spectrum. It
  writes `prepfit`'s four columns at full precision, adds a data line with the row's
  resolution and shift, writes the continuum's first guess (D56), and adds the specid
  to every absorption line whose transition falls inside the snip. Its transition
  goes in `ui/project.json` (`snips`: system, ion, rest, row). **CLEAR**
  (`snips.clear_plan`) removes a snip after a preview.
- **The example.** `J013301m400628.dat` has 3 columns and 44 199 pixels. Its two
  snips have `prepfit`'s 4 columns (wave, flux, error, and a fit range of 0s and 1s;
  383 and 381 pixels, of which 110 and 69 are fitted). Each snip is an unbroken run
  of the spectrum's pixels, equal to the last bit in wave, flux and error.
- **The example's `.mod`.** It names the snips by relative path
  (`../data/…_reg.dat`) with specids 0 and 1, `resolution=vfwhm(4.181VFIX)`, and
  `columns=[wave,flux,error,fitrange,continuum]`. The files have no continuum column,
  so ALIS warns and uses 1. The model has three H I and three D I components, with the
  D I column densities linked to H I.

Task 7.1 gets to S23's result from the other end. The user starts a new project from
the spectrum, then brings the old snips and model in. The project then behaves as if
it had started from that spectrum.

**(a) Load snips… (Regions tab).** A button under the transition list opens a file
dialog that can choose several files. (Beside SNIP and CLEAR, as first planned, it
cut short the fit regions' two tools at 1440 px with the `.mod` panel open; the
window has no Regions menu.) It is enabled while the project has a source spectrum.
For each file:
1. **Read it.** A text table of 3 to 5 columns: wave, flux, error; a fourth column of
   0s and 1s is the fit range (`prepfit`'s); a fifth of 0s and 1s is a bad-pixel mask,
   as the dashboard writes its own snips (D52). Any other table is refused.
2. **Match it to a file row.** Every pixel's wavelength must be a pixel of the row's
   source spectrum, as the mode reads it (with any continuum multiplied in, D13, as
   `prepfit` writes it), to 5% of the pixel's width. The flux and error must equal
   the source's there, to 10⁻³ of the error, so that a snip written with fewer digits
   still matches. The first row that matches is the snip's dataset. With no match, the
   file is refused, with the reason for the nearest row: outside its range, not on
   its pixels, or the flux or error differs at *n* of *N* pixels.
3. **Find its transition.** It comes from the file name, as `prepfit` names it
   (`<stem>_1H_I_1215.7_reg.dat`, the rule that `lines._guess_transition` already
   uses for an imported fit). It belongs to the system whose redshift puts it inside
   the snip, nearest the middle. A file whose name names no transition, or whose
   transition no system puts inside it, is refused (Q7.4).
4. **Check for overlaps.** A snip that shares pixels with a snip of the same row
   already in the project replaces it. Before anything changes, one dialog lists
   every snip that would be replaced, with Replace and Cancel. Two chosen files
   sharing pixels of one row are refused (Q7.2).
5. **All or nothing.** If any file fails, nothing is loaded, and one message lists
   each file and why.

The load is one step ("Load 2 snips: H I 1215.7, H I 1025.7"):
- each replaced snip is removed as CLEAR removes it;
- each file is stored byte for byte, under its own name beside the model's other
  snips. A fit range of 0s is added when it has none ("no fit region until one is
  drawn", as for SNIP), and a bad-pixel column when its source has a mask and the
  file does not (D52);
- its data line, continuum first guess, the absorption lines' specids, and its
  record in `ui/project.json` are as SNIP writes them. The specid is SNIP's
  (`HI_1215`).

The Regions tab then shows the first loaded transition.

**(b) Load a .mod… (File menu, and the `.mod` panel's right-click menu).**
1. **Read it.** The `.mod` is read as `bundle.pack` packs it, without the earlier
   run's outputs: its text, and the snips its data lines name relative to it. Anything
   a bundle cannot hold is refused with `bundle.unsupported`'s reasons.
2. **Check its snips.** Each data line's snip is checked against the file rows as in
   (a), steps 2 and 5, with its columns taken from the data line's `columns=` rather
   than guessed. Its fit region is whatever the data line says: a column, a range, or
   all.
3. **Take its text.** The model text becomes the `.mod`'s as written, comments and
   spacing kept (D7), with two exceptions (Q7.3):
   - a data line's file path is rewritten to `../data/<name>` when its path from the
     project's model would leave the project's file tree;
   - `run blind` follows the project's blinding.
4. **Replace the snips.** The project's snips are replaced by the `.mod`'s, each
   stored where its data line names it, and the records of the earlier snips are
   removed (Q7.1).
5. **Assign the rows.** Each data line's specid goes to the row its snip matched.
   The FWHM is the `.mod`'s, and the step says so when it differs from the one given
   for the file (Q7.3). Each snip's transition is recorded as in (a) step 3 when it
   can be placed, and otherwise guessed, as for an imported fit.
6. **Build the systems.** The project's systems are kept. Components within 500 km/s
   of a system join it. The others form systems inferred as an imported fit's are
   (Q2.4), marked guessed, with the Data tab's "!" until Confirm (D58). The
   Components tab shows its notices for whatever breaks D20 or D22 (D67).
7. **Preview, then one step.** A preview lists the snips removed and loaded, and the
   lines that change. The load is one step, and can be undone.

**What stays as it is.** Import a fit (the start page, `alis fit.mod`), SNIP, CLEAR
and merging (D57) are unchanged. So are ALIS outside `alis/dashboard/`, `prepfit`,
and the snip format (D4).

**Modules.**
- `alis/dashboard/loadfit.py` (new, no Qt): reading and matching a snip file, the
  plan of Load snips (its problems, the snips it replaces, and its step), and the
  plan of Load a .mod….
- `alis/dashboard/snips.py`: SNIP's writing of a data line and its record is factored
  out, so that SNIP and Load snips share it.
- `alis/dashboard/actions.py`: `regions.load_snips` and `file.load_mod`.
- `alis/dashboard/qt/regions.py` (the button), `qt/window.py` (the File menu, the file
  dialogs and the messages), and `qt/modpanel.py` (an action in its right-click menu
  only, so that its header keeps its width). Both the replacements and the preview
  are shown in `qt/dialogs.py`'s `PlanDialog`, which is unchanged.
- `alis/dashboard/lines.py`: the transition a file name names (`named_line`) and the
  system that places it (`placing_system`), factored out of the guess for an
  imported fit's snips.
- **Tests.** Unit tests in `tests/test_dashboard_loadfit.py`, on synthetic spectra and
  snips made in `tmp_path`, as `test_dashboard_snips.py` makes them, with the failing
  cases (refused files, overlaps, all or nothing, undo). The J013301m400628 check of
  Task 7.1 runs from `context/fitting_examples/`, and is skipped when the files are
  absent, as the J1358p6522 tests are. The window tests of both actions are in
  `test_dashboard_qt_regions.py`.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage7_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests; run
> the `fast` batch before the stage closes. A task that changes ALIS outside
> `alis/dashboard/` also runs the regression harness, its `test_bundle` mode included.

*One task per item of the list (7.1, 7.2, …), each with:*
- *what it builds or fixes;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

**7.1 — Load snips from files previously saved with `prepfit`.**
- This new feature will allow users to load information from `ALIS` fits that were analysed with the previously released version of `ALIS`. The key elements of loading old/past `ALIS` fits include:
  1. Load a full spectrum
  2. Load one or more snips that were previously saved with the `prepfit` tool
  3. Load the contents of a .mod file into the .mod editor
- On the snips tab, we should allow users to load snips that have previously been saved with the `prepfit` tool. This will allow users to easily access and work with their previously saved snips.
- We will need to load the information and check that the snips match the expected wavelength, flux, and error spectrum, and that there is a fitrange mask for each snip. If any of these checks fail, we will need to raise an error and inform the user. We will also need to check if an overlapping snip is already loaded, and if so, we need to warn the user that by loading this snip, the existing snip will be replaced. The user should be able to choose whether to proceed with the replacement or cancel the load.
- We should allow the user to either load one snip at a time, or load multiple snips simultaneously. The user should be able to select multiple snips from the file dialog and load them all at once.
- The process should be as follows:
  1. User selects a (full) spectrum to load and enters the information like FWHM etc. in the same way that they would if they are starting the analysis of a new spectrum.
  2. The user can load one or multiple snips simultaneously on the snips tab.
  3. The dashboard checks that the snip's wavelength, flux, and error spectrum match the expected values from the full spectrum.
  4. The system checks that there is a fitrange mask for each snip. If it does, the fitrange is loaded, if not, the snip is still loaded with no fitrange.
  5. If all checks pass, the snips are loaded into the snips tab; otherwise, an error message is displayed.
  6. The user can load a .mod file from disk and the dashboard will set up the components cards and populate the .mod editor with the contents of the `.mod` file.
- **Check:** Load a previously saved snip and verify that the information is correct and that the fitrange mask is present.
- **Check:** I have added some additional context to check that this is working as expected. Please see the files in `context/fitting_examples/Temperature/J013301m400628/` for examples of previously saved snips that could be loaded into the dashboard. The files include a full spectrum (`J013301m400628.dat`), two snips (`*_reg.dat`), and a .mod file. The snips were created with `prepfit`. Please load these files into the dashboard and verify that the information is correct and that the fitrange mask is present for each snip.

**7.N — Close the stage.**
- Publish the screenshots of what changed for RJC's review, and apply the review.
- `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md`, `tests/README.md` and the
  `gui-component` skill brought up to date.
- Run the `unit`, `gui` and `fast` batches, and `test-coverage` on what changed.
- Record the stage's decisions in `ALIS_v2_dashboard_prompts.md`, and update the
  "After v1" list of `dashboard_stage0.md`.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added; it documents the project
  model, the history, the gate, the spectrum view and the Fit tab's runs.
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task; the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where an item touches a fit's results.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D77.
- `claude_prompts/dashboard_stage6.md` and its log: the Fit tab, and how v1 was
  closed; the earlier stage documents for the tabs they built.
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).

## Queries

*Raised by Claude after reading the list. Each gives Claude's lean.*

*Q7.1–Q7.5 were raised on 2026-10-30 (Prompt 1), about Task 7.1. The Design section
follows the leans.*

**Q7.1 — What Load a .mod… does with the snips already in the project.** A `.mod`
names its own snips. Two ways to bring it in:
- **(a) The `.mod` brings its snips.** The model text becomes the `.mod`'s as
  written. The snips its data lines name are loaded from beside it, checked against
  the spectrum as Load snips checks them, and replace the project's snips, including
  any loaded earlier with Load snips. A snip loaded earlier and not changed since is
  the same file, so nothing is lost; a fit region redrawn since is replaced by the
  `.mod`'s snip's.
- **(b) The `.mod` brings only its model.** Its settings, model and link sections
  replace the project's. Each of its data lines is matched to a snip already loaded
  (same dataset, same pixels), whose specid then replaces the `.mod`'s in every line.
  A data line with no loaded snip is refused.

(a) puts the `.mod` in the editor as it is written (step 6 of Task 7.1), and needs
one action rather than two. Load snips is then for starting a new model from old
snips, and Load a .mod… for carrying on with an old fit. (b) keeps regions redrawn in
the dashboard, but rewrites the `.mod`'s data lines and specids.

My lean: (a).

**Response:** I agree with option (a). It seems more straightforward to have the `.mod` bring its own snips and replace any existing ones, ensuring consistency with the model. This approach also simplifies the workflow for users who want to continue working with an old fit.

**Q7.2 — Replacing a snip that overlaps.**
- **What overlaps:** a loaded snip that shares at least one pixel of the same
  spectrum with a snip already in the project. D57's merging is not offered here,
  since the task asks for a replacement.
- **How it is replaced:** the old snip is removed as CLEAR removes it (its data line,
  its continuum, its specid in the absorption lines, and its regions). The new one is
  added as a new snip, with its own continuum first guess and specid.
- **Asking:** with several files, one dialog lists every replacement, with Replace
  and Cancel, rather than one question per file.
- **Within one load:** two chosen files that share pixels of one spectrum are
  refused.

My lean: as described.

**Response:** I agree with the proposed approach for handling overlapping snips. It is clear and user-friendly to have a single dialog listing all replacements, allowing the user to make an informed decision. Refusing two chosen files that share pixels of one spectrum within one load is also a good safeguard against potential conflicts.

**Q7.3 — The settings, the blinding and the FWHM of a loaded `.mod`.**
- **Settings:** taken as written in the `.mod` (here `chisq miniter 10`), except
  `run blind`, which follows the project. A project made with global blind stays
  blinded. The `.mod`'s `blind=True` lines are hidden, as they are when a fit is
  imported.
- **FWHM:** the `.mod`'s (4.181 km/s), even when the New project dialog was given
  another. The step says so, and Reset FWHM (↺) returns to the dialog's value.
- **The earlier run:** its outputs (the `.mod.out` beside the `.mod`) are not brought
  in, so the Fit tab starts with no run. A `.mod.out` can itself be loaded, and its
  best-fit values then become the starting values.

My lean: as described.

**Response:** I agree with the proposed handling of settings, blinding, and FWHM for a loaded `.mod`. It makes sense to take the settings as written in the `.mod`, while ensuring that the `run blind` setting follows the project's configuration. Using the `.mod`'s FWHM and allowing Reset FWHM to return to the dialog's value provides flexibility for users. Not bringing in the earlier run's outputs is also a good decision, as it allows users to start fresh with their analysis.

**Q7.4 — Snips that cannot be placed.** The Regions tab shows a snip under a system's
transition. A snip's transition comes from its file name, as `prepfit` names it, at
the system whose redshift puts the transition inside the snip. With no redshift set
yet (the New project dialog leaves it optional), or for a line of a system not yet
added, the snip has nowhere to go. I propose:
- such a file is refused, and nothing is loaded. The message says to set the
  redshift on the Data tab, or add the system, first. A file whose name names no
  transition is refused the same way;
- a `.mod`'s snips are never refused for this. Its components give the systems, and
  each snip's transition is guessed as an imported fit's is.

My lean: as proposed.

**Response:** I agree with the proposed handling of snips that cannot be placed. Refusing such files and providing a clear message to the user about setting the redshift or adding the system first is a good approach. Allowing a `.mod`'s snips to be loaded without refusal, as their components provide the necessary context, is also a reasonable decision.

**Q7.5 — S23 (attaching a source spectrum to an imported fit).** After this task, an
old fit is brought in with its spectrum through a new project: New project, then Load
a .mod…. S23 would do the same from the other end, starting from an imported fit.

My lean: S23 comes off the "After v1" list of `dashboard_stage0.md` at the close of
this stage, as covered by Task 7.1.

**Response:** I agree with the proposal to remove S23 from the "After v1" list of `dashboard_stage0.md` at the close of this stage, as Task 7.1 effectively covers the functionality of bringing in an old fit with its spectrum through a new project.

## Prompts

1. Please read this document, with my current list of changes and bug fixes, and the work of Stages 0–6 (their documents and logs). I have written Task 1 based on a feature that would be good to include in the ALIS dashboard. Let's implement this first, and then I will draft a more complete list of bugs, improvements, and additions to the dashboard. If anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please execute Task 7.1.
