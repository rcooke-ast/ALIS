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
13. **(change)** Just below the main menu, there are buttons labelled "Undo", "Redo", "Open", and "Save". It would be good if there was another button to the right os "Save" that says "Run" and has the same function as the "Run" button on the "Fit" tab ("Inspect" subtab).
14. **(bug)** Possible bug. I added in a `constant` zerolevel model to a saturated absorption line. The parameters didn't change though. This is a possible bug and it would be good to check if this is working in the dashboard or not. If we add a shift option to each SNIP, then we could also add a zerolevel option to each SNIP. 

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

### 7.2–7.14 — The other items of the list

*Written by Claude on 2026-10-31 (Prompt 2), from items 2–14 of the list, the examples
in `context/fitting_examples/Temperature/`, and the code of Stages 2–7. Revised the
same day (Prompt 3) after RJC's responses to Q7.6–Q7.13, and from RJC's J193957m100241
project (`context/fitting_examples/Temperature/J193957m100241/model/J193957m100241.model`,
Q7.11). Each item's design is in its task below, and follows RJC's responses to
Q7.6–Q7.15 (Q7.14 and Q7.15 answered in Prompt 4).*

**What was found.** Each bug was reproduced before its task was written.
- **Items 5 and 6.** The test model was J101155p294141's, with J013301m400628's snips,
  since its own snips are not in `context/`. Each component there is one H I line with
  an unlabelled redshift, and D I comes from the ratio line `ion=2H_I/1H_I`.
  - The card lists D I under "Excluded". It looks only for the component's own lines
    of an ion, not its ratio lines (`qt/components.py`, `_card`).
  - Exclude from H I and Exclude from D I both raise the error RJC quoted:
    `components.exclusion` needs a redshift label. Its message points to a notice
    that is made only for a component of two lines or more
    (`project._untied_notices`).
  - The error shows as a traceback because `_toggle_ion` calls `exclusion` outside the
    window's error handling.
- **Item 2** (J013301m400628). Excluding H I from a1 comments out its line. It also
  comments out the link that computes D I from it (`nda(nha,FIXDH) = …`, with a
  note), so D I's log N becomes free. D I is then a1's last ion, and excluding it
  offers only to remove a1 (D64).
- **Item 11.** The drag moves T only in the pure thermal mode, and that mode is
  recognised only when b_turb is fixed at exactly 0 (`Component.temperature_mode`).
  RJC's fits write a fixed b_turb as `0.5BFIX`. Such a b_turb is moved by the drag,
  since `_check_drag` refuses only a linked parameter.
- **Item 12.** voigt.py keeps b_turb at 0.5 km/s or more, but pure thermal fixes it at
  0 (`0.000BA1`). The pre-flight check lets that through, since it checks only free
  values against their limits, and ALIS then stops ("A parameter that = 0.0 is not
  within the specified limits"). Tried on a synthetic Lyα, after the dashboard's own
  pure-thermal step:
  - with `lim param BA1 [0.0,None]` added, ALIS fits it (T = 22,851 K, b_turb stays
    0);
  - with T also at 0, ALIS stops with "Voigt Total Doppler parameter is zero". T's
    lower limit is 0 K, so a T that reaches it during a fit stops ALIS the same way.
    This is the last error saved in RJC's J193957m100241 project
    (`runs/last_error.json`), and may be the rejection Q7.10's response describes
    (Q7.15);
  - the pre-flight check and the validator accept both `lim param <label> [0.0,None]`
    and `lim voigt bturb [0.0,None]`.
- **Item 14** (RJC's J193957m100241 project, Q7.11). Its zero levels are four
  `constant 0.0` lines typed in the `zerolevel` section, one per snip. The run left all
  four at exactly 0, with an error of 0, and `run_alis` outside the dashboard does the
  same. The cause is in ALIS: `constant`'s value has a lower limit of 0
  (`alis/functions/constant.py`), set for a continuum and inherited by a zero level.
  The troughs of these snips lie slightly below zero, so the fit pins each zero level
  on that limit (Q7.14).
  - With `lim param zla [None,None]`, Lyα's zero level fits to −0.0136 ± 0.0009, and
    χ² falls from 919 to 695 (Lyα alone).
  - The results table flagged each one "at limit (0)", but nothing says that the limit
    is `constant`'s own. The pre-flight check's note names the value only as "value on
    line 16".
  - Prompt 2's synthetic test had a zero level of +0.05 (fitted to 0.054 ± 0.003), so
    it never met the limit.
- **Per-snip zero levels** (Q7.12's response). The Data tab reads the same project as
  having no zero level. A row's zero level is a `constant` that covers all its snips
  (`edit.zero_level_line`), and none of the four does. Switching it On adds a fifth
  `constant` over the same snips, and ALIS then stops ("You can only specify the
  zero-level once for each specid").
- **Labels of excluded lines** (Q7.6's response). New components and the scheme's
  labels already skip labels written on commented-out lines (`edit.used_labels`). A
  label the user types does not: `rename_label`, `fix`, `untie` and `set_limits` check
  only the live lines (`ParsedModel.label_uses`). A rename also leaves the old label on
  the excluded lines, which then lose their component.
- **Item 8.** After J013301m400628.dat is chosen, the dialog grows from 233 to 349 px.
  That is less than its contents need, so the column table (y 95–265) runs under the
  hint below it (y 241–286).

**What stays as it is.** These stay unchanged:
- ALIS outside `alis/dashboard/`, except a zero level's lower limit (Q7.14, Task
  7.7);
- the bundle and the snip format (D4);
- the decisions D1–D77, except where a query refines one: D64 (Q7.6), D45 and F4
  (Q7.9), D21 (Q7.10, Q7.15), D14 and D35 (Q7.12).

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

*Tasks 7.2–7.15 were written by Claude on 2026-10-31 (Prompt 2), and revised the same
day (Prompt 3) after RJC's responses to Q7.6–Q7.13. RJC's responses to Q7.14 and
Q7.15 (Prompt 4) agreed with Tasks 7.7 and 7.5 as written; only 7.5 and 7.7 were made
more precise. Each task names its item of the list. The order
puts a bug before the work that builds on it, and keeps work on one part of the window
together:*
- *the exclusions on the cards (7.2–7.4);*
- *b_turb and T (7.5–7.6);*
- *the zero level (7.7);*
- *the smaller fixes and changes (7.8–7.11);*
- *the new features (7.12–7.13);*
- *the documentation's design last (7.14), as item 4 asks.*

**7.2 — Excluding from a component with no redshift label (item 6, bug;
`components.py`, `qt/components.py`; D64; Q7.8).**
- **The bug.** Exclude on a component whose lines share no redshift label raises a
  `ComponentError`. Its message points to a notice that a component of one line never
  has, and the window shows it as a traceback.
- **The fix (Q7.8).** The exclusion's step first gives the component a redshift label,
  as the notice's fix does (`project.tie_component`), so that the commented-out line
  still names its component. The preview lists the lines that gain the label. A
  component of several untied lines is tied in the same step, its lines taking the
  main line's z.
- **The label is new** (Q7.8's response). No line of the model uses it, commented-out
  lines included (`edit.used_labels`, as for a new component). It therefore cannot tie
  the component to another, now or when an excluded line is included again.
- Every refusal of Exclude or Include is shown as a message, never as a traceback.
- **Check.** Made into tests first, on a synthetic project written as J101155p294141's
  model is: one unlabelled H I line per component, D I from the ratio line.
  - In `test_dashboard_components.py`: Exclude from D I is one step that ALIS reads.
    Its preview lists the label written on a1's H I line, and the ratio line that a2
    shares. Include brings back exactly what was excluded, and one undo restores every
    byte. Exclude from H I, a1's last line, asks as for any last ion (Task 7.4 changes
    what follows). A component of two untied lines is tied and excluded in one step.
  - With `za1` already written on a commented-out line, the exclusion writes `za2`.
  - In `test_dashboard_qt_components.py`: Exclude on such a component shows the
    preview, not an error.

**7.3 — Ions given by a ratio, on the cards (item 5, bug; `components.py`,
`qt/components.py`; D22, D64; Q7.7).**
- **The bug.** A card lists an ion that the component has only through a ratio line
  (D I from `ion=2H_I/1H_I`) under "Excluded".
- **The fix (Q7.7).** The ion gets its own row under the ion it is a ratio to
  ("↳ D I"):
  - the row shows the ratio's value (log D I/H I), typed and badged like any value;
  - a note names the other components that share the ratio line;
  - a ratio is not dragged.

  "Excluded" lists only the ions the component has neither a line nor a ratio of,
  and ratios commented out by an exclusion of them (`is_excluded`).
- A helper in `components.py` gives each ion of the system as the card shows it: its
  own line, a ratio, or excluded. It is tested without Qt.
- **Check.** Made into tests first, on 7.2's project.
  - The helper's test, in `test_dashboard_components.py`.
  - In `test_dashboard_qt_components.py`: a1's card has a D I row with the ratio's
    value, and no "Excluded" line. Typing a new value is one step, seen on a2's card
    too. With D I excluded, the card shows "Excluded: D I", and a click includes it.
  - In `test_dashboard_qt_blinding.py`: with the ratio line blinded, its value is
    masked.

**7.4 — Excluding a component from every ion (item 2, bug; `project.py`,
`components.py`, `remove.py`, `qt/components.py`; D20, D64; Q7.6).**
- **Now.** Excluding a component's last ion only asks to remove the component (D64).
  Say a link computes D I from H I. After H I is excluded, D I's log N is free, and
  excluding D I too would remove the component. The only way to keep the component
  but leave it out of the fit is to comment out its lines by hand.
- **The change (Q7.6).** Excluding the last ion keeps the component, excluded from
  every ion. All its lines are commented out, and:
  - its card is greyed, with Include for each of its ions;
  - it keeps its place among the system's components, so that no id changes;
  - its velocity is masked when one of its lines is blinded.

  The project reads such a component from commented-out voigt lines that share a
  redshift label no live line carries. A component commented out by hand is
  therefore shown the same way. ✕ removes it, and deletes its commented lines.
- **The preview of an exclusion** names the values a link computed from the excluded
  line, which become free ("D I's log N in a1 becomes free").
- **Including an ion** brings back the lines excluded with it, except a line that
  names a label carried only by a line still excluded (here the link, while D I is
  still excluded). That line stays out, and its note moves to that line's
  exclusion. ALIS can then read the model after every step.
- **Its labels stay its own** (Q7.6's response), so that including it again cannot tie
  it to another component:
  - new components and the scheme's labels already skip the labels of excluded lines;
  - a label the user types (`rename_label`, `fix`, `untie`, `set_limits`) is refused
    when an excluded line carries it. The message names the component ("za1 belongs to
    a1, which is excluded from the fit");
  - a card offers no excluded component as a target to tie to;
  - a live line given such a label in the `.mod` panel gets a warning from the
    validator. Include is refused while that lasts, and its message names the line;
  - a rename also renames the label on the excluded lines.
- **Check.** Made into tests first, on J013301m400628's model
  (`test_dashboard_components.py`, with synthetic snips, and with the example itself,
  skipped when absent).
  - Exclude H I, then D I, from a1: two steps, and ALIS reads the model after each.
    a1 is still listed, excluded from both, and a2 and a3 keep their ids.
  - Include H I, then D I, and also in the other order: ALIS reads every step, and
    the model's text returns to what it was before the exclusions. Each undo restores
    every byte.
  - A model with a component commented out by hand shows it excluded, and Include
    brings it back.
  - With a1 excluded from both ions, a new component, a typed label and a rename never
    take a1's labels. Typing `za1` on a live line in the `.mod` panel gives the warning,
    and Include is refused until the label is gone.
  - Renaming a1's redshift label while H I alone is excluded also renames it on the
    excluded line. Include then brings that line back to a1.
  - In `test_dashboard_qt_components.py`: the greyed card, its Include buttons, and ✕.

**7.5 — Pure thermal that ALIS fits (item 12, change; `components.py`, `edit.py`,
`preflight.py`; D21; Q7.10, Q7.15).**
- **Now.** Pure thermal fixes b_turb at 0, below voigt.py's limit of 0.5 km/s. The
  pre-flight check lets it through, since it checks only free values against their
  limits, and ALIS stops. And with b_turb at 0, a T that reaches its lower limit of
  0 K gives a total b of 0, which stops ALIS too (see Design).
- **The change (Q7.10, Q7.15).** In the same step, pure thermal also writes:
  - `lim param <label> [0.0,None]` for the component's b_turb (`edit.set_limits`,
    with the uppercase label that fixes it). An existing `lim param` of that label has
    its lower bound set to 0.0;
  - `lim param <label> [1.0,None]` for its T (Q7.15), unless T already has a lower
    limit of 1 K or more. An existing `lim param` of T's label has its lower bound
    raised to 1.0.

  A b_turb or T with no label is given a new one of the scheme, as in Task 7.2.
  Leaving pure thermal takes out the commands it wrote, when they are still as
  written. A limit the user wrote stays, such as RJC's `lim param BX1 [0.01,None]`.
- A b_turb fixed at 0.5 km/s or less, with T free (RJC's `0.5BFIX` and `0.01BX1`), is
  shown as pure thermal. It is left as written until the user changes the mode.
- **The pre-flight check** also refuses a fixed value outside its limits, since ALIS
  stops on it ("b_turb on line 24 is fixed at 0, below its limit of 0.5 km/s").
- **Check.** Made into tests first.
  - In `test_dashboard_components.py`:
    - pure thermal writes both commands;
    - Free T, then a fixed preset, take them back out, and a limit the user wrote
      stays;
    - one undo of each restores every byte;
    - a b_turb tied across two components gets one `lim` command;
    - `0.5BFIX` and `0.01BX1`, with T free, read as pure thermal.
  - In `test_dashboard_preflight.py`: a b_turb fixed at 0 with no limit is refused,
    and passes with pure thermal's commands.
  - **Side effects** (Q7.10's response), as the Fit tab runs a fit (a `gui` test in
    `test_dashboard_qt_fit.py`). The project is a synthetic Lyα and Lyβ of H I and D I,
    made with a known T and no turbulence.
    - Pure thermal runs to completion. The best fit gives T within 3σ, with b_turb at
      0 and no flag on either.
    - Continue from best fit keeps both commands.
    - The same project with T starting at 1 K also completes.
    - A turbulent component beside it keeps b_turb's limit of 0.5 km/s.

**7.6 — A drag moves the free one of b_turb and T (item 11, bug; `components.py`,
`qt/components.py`; D21, D63; Q7.10).**
- **The bug.** A horizontal drag below the continuum moves T only when b_turb is
  fixed at exactly 0. In every other case it moves b_turb, even a fixed one.
- **The fix (Q7.10).** The drag moves the free parameter:
  - T when b_turb is fixed: b_turb is held, and T comes from
    b² = b_turb² + 2kT/m;
  - b_turb when T is fixed, or when both are free (T held, D21);
  - nothing when both are fixed, or when the free one is computed by a link. The
    status bar says why; log N still moves.

  One helper gives the parameter a horizontal drag moves. `drag` and `draggable` use
  it, so the cursor, the tooltip and the step agree.
- **Check.** Made into tests first.
  - In `test_dashboard_components.py`, for each case (b_turb fixed at 0, or at 0.5,
    with T free; T fixed; both free; both fixed): the parameter `drag` changes. Where
    one moves, the total b of the profile drawn at the end of the drag equals the b
    dragged to, within 10⁻⁶ km/s. A fixed parameter is never changed.
  - In `test_dashboard_qt_components.py`: a drag on a `0.5BFIX` component writes T,
    not b_turb, as one step. So does a drag on one of the `0.01BX1` components of RJC's
    J193957m100241 project (skipped when absent).

**7.7 — A zero level below 0 (item 14, bug; in ALIS, `load.py`; in the dashboard,
`model.py`, `preflight.py`, `results.py`; D14; Q7.11, Q7.14).**
- **The bug.** A zero level cannot go below 0. `constant`'s value has a lower limit of
  0, set for a continuum, and a `constant` in the `zerolevel` section inherits it. In
  RJC's J193957m100241 the troughs lie below zero, so the fit pins all four zero levels
  at 0, with an error of 0 (see Design). The results flag them "at limit (0)", but no
  limit is written in the model, so nothing says where it comes from.
- **The fix (Q7.14).** In ALIS, a `constant` in the `zerolevel` section has no lower
  limit. An emission `constant` keeps its limit of 0.
  - The place: `alis/load.py`. A line takes its function's limits twice there: when
    it is loaded (`call_function_load`), and in the pass that follows, which applies
    `lim <function> <parameter>` commands and resets each line's limits to its
    function's (`adjust_lim`). A `constant` of the `zerolevel` section takes no lower
    limit from `constant.py` in either. A `lim param` on its label still holds.
  - Only `constant` changes, as Q7.14 agreed. Other functions keep their own limits
    wherever they are used.
  - The dashboard reads limits the same way (`model.py`), so its pre-flight check, its
    drag bounds and its results follow.
- **The dashboard's words.**
  - The pre-flight check and the results name a zero level as the Data tab does ("the
    zero level of J193957m100241, Lyα"), not as "value on line 16".
  - An "at limit" flag says when the limit is the function's own, not a `lim` line
    ("at limit (0.5, voigt's own)").
- **Check.** Made into tests first.
  - In ALIS (a unit test in `tests/`): synthetic saturated troughs with zero levels of
    −0.03 and +0.05, each fitted from `constant 0.0` in the `zerolevel` section, give
    the true value within 3σ. An emission `constant` still cannot go below 0, and a
    zero level with `lim param <label> [0.0,None]` stays at 0 or above.
  - In the dashboard (`test_dashboard_runs.py`): the same troughs, fitted with the Data
    tab's Zero level switched On, and with the line typed in the `.mod` panel. The
    results table lists each zero level with no flag, Continue from best fit writes it,
    and Inspect's Final draws the model with it.
  - RJC's J193957m100241 project, run as it is (skipped when absent): its four zero
    levels leave 0, and χ² falls below the 3659 of its last run.
  - In `test_dashboard_preflight.py` and `test_dashboard_results.py`: the new names
    and flags.
  - The regression harness, with its `test_bundle` mode, since ALIS outside the
    dashboard changes. A fit whose zero level was above 0 gives the same result; any
    fit that changes is listed in the log with the reason.

**7.8 — The Add a spectrum dialog (item 8, bug; `qt/dialogs.py`).**
- **The bug.** Once a file is chosen, the column table appears, but the dialog does
  not grow enough. The table runs under the hint below it ("Wavelength, flux and error
  are needed…").
- **The fix.** The layout makes room for the table when it is shown, with the hint
  below it, and the dialog grows to fit.
- **Check.** Made into a `gui` test first (`test_dashboard_qt_window.py`), with
  J013301m400628.dat and synthetic files of 3 and 5 columns. After a file is chosen,
  the table's and the hint's rectangles do not overlap, and both are inside the
  dialog. The same holds after a second file with more columns, and after changing to
  a file that cannot be read. A screenshot goes in the review.

**7.9 — One way to open a .mod (item 3, change; `actions.py`, `qt/window.py`,
`qt/modpanel.py`; D45, F4; Q7.9).**
- **Now.** The File menu has two items for a `.mod`:
  - Import a fit (.mod)… opens a `.mod` and its snips as a new project, with no
    source spectrum;
  - Load a .mod… brings a `.mod` into the open project, with its snips checked
    against the project's spectra (Task 7.1).
- **The change (Q7.9).** One File-menu item, Open a fit (.mod)… (Ctrl+I):
  - with a project that has a source spectrum, it previews loading the fit into the
    project, as Load does now. The preview also offers Open as a new project…;
  - with no project open, or one with no source spectrum (an imported fit), it first
    asks how to open the fit, and says which way is recommended:
    - **Open with its spectrum (recommended).** The New project dialog opens in the
      folder of the `.mod`'s snips, with the `.mod`'s FWHM. Once the project is made,
      the fit is loaded into it as Load does, with its preview;
    - **Open without a spectrum.** The fit opens as a new project, as Import does now;
    - **Cancel.**
  - Open as a new project… asks the same question. A new project closes the open one
    as Open… does, asking to save its changes;
  - when the snips do not match the spectrum, the message says why. It offers to
    choose another spectrum, or to open the fit without one.

  The start page's button and `alis fit.mod` ask the same question. The `.mod` panel's
  right-click item still loads into the open project.
- **Check.** Made into tests first.
  - In `test_dashboard_actions.py`: the File menu has one item for a `.mod`.
  - In `test_dashboard_qt_window.py`, with J013301m400628's files:
    - with no project, the question comes first, with the recommended way as its
      default;
    - with its spectrum, the project is made and the fit loaded, as in Task 7.1's
      check. Without, the project is the one Import makes now;
    - with J013301m400628's project open, the preview loads the fit, or opens it as a
      new project;
    - with a project of another spectrum, the message says why;
    - Cancel, at any point, changes nothing;
    - the start page's button and `alis fit.mod` ask the same question.
  - Task 7.1's tests pass, renamed where they name the menu items.

**7.10 — Clearing the recent projects (item 7, change; `preferences.py`,
`qt/window.py`).**
- **The change.** The start page's Recent projects gains two buttons:
  - Remove from list, for the selected project (also Delete and the right-click
    menu);
  - Clear list, which asks once first ("The projects themselves are not changed").

  File ▸ Open recent ends with Clear recent projects. Only `recent.json` changes, and
  no project's file is touched. `Recent.clear()` is added beside `Recent.remove()`.
- **Check.** Made into tests first.
  - In `test_dashboard_preferences.py`: `clear` and `remove` write the list, and no
    other file changes.
  - In `test_dashboard_qt_window.py`: the buttons, Delete and the menu item work.
    Cancel in the question keeps the list. The empty list reads "No recent
    projects.", and its buttons are disabled.

**7.11 — Run on the toolbar (item 13, change; `qt/window.py`).**
- **The change.** The Fit tab's Run action itself (`fit.run`) goes on the toolbar,
  right of Save and its autosave time, before the blinding pill. It is enabled
  whenever the Fit tab's Run is, so it is disabled while a run goes (Stop is in the
  status bar). It keeps its tooltip and Ctrl+R.
- **Check.** In `test_dashboard_qt_actions.py`:
  - the toolbar's actions come in order: Undo, Redo, Open, Save, Run, …;
  - Run starts a run from any tab, as the Fit tab's Run does (a `gui` test with
    metal_line_abs, as in Stage 6);
  - it is disabled while the run goes, and enabled after.

**7.12 — Normalised or with the continuum, in Inspect (item 10, new; `fitview.py`,
`qt/fit.py`, `actions.py`; D72).**
- **Now.** Inspect's panels always show the data and the model divided by the shown
  model's continuum.
- **The change.** A Normalised | Continuum switch in the View pane, beside Model and
  Residuals. Continuum shows the data as they are, the model with its continuum, and
  the continuum itself (dashed, as the Regions tab draws it). The continuum is
  Initial's starting continuum or Final's best fit, following Model.
  - The residuals and the marks do not change.
  - The choice is kept as the View pane's other choices are.
  - It is a checkable action (`fit.normalised`), so it is listed in the shortcut
    sheet.
- **Check.** Made into tests first.
  - In `test_dashboard_fitview.py`: what the switch needs without Qt.
  - In `test_dashboard_qt_fit.py`, on metal_line_abs, for Initial and Final: with
    Continuum, the data drawn equal the snip's flux, and the model equals ALIS's
    model before division, to the last bit. Normalised draws what it draws now. The
    residual strip and the marks are the same in both. The choice is kept when the
    tab is left and shown again.

**7.13 — A snip's own shift and zero level (items 9 and 14, new; `datasets.py`,
`snips.py`, `edit.py`, `qt/regions.py`, `qt/data.py`; D14, D35, D55; Q7.12).**
- **Now.** The shift and the zero level belong to a dataset (D14, D35), and are set on
  the Data tab:
  - one `vshift` for all of a row's snips (none for the reference);
  - one `constant` for all of them.
- **The change (Q7.12).** The Regions tab's Snip box, for the selected dataset's snip,
  gains Shift and Zero level. Each is As the dataset (the default) or Its own, and
  each change is one step:
  - its own shift is a `vshift` on that data line, with a label of its own. It starts
    free at 0 km/s, and is then typed and badged as on the Data tab;
  - its own zero level is a `constant` with that snip's specid alone, its specid taken
    out of the dataset's;
  - back to As the dataset restores the dataset's word and specid.
- **What follows from it.**
  - Regions copied between datasets (D55) move by each snip's own shift; until now
    they followed `snips.row_shift`.
  - SNIP gives a new snip the dataset's shift and zero level.
  - CLEAR, and removing the snip, remove its own shift and zero level.
  - The Data tab's shift and zero-level cells say how many snips have their own.
- **What is read** (Q7.12's response). A snip's own shift and zero level are read from
  the model, whatever wrote it: Load a .mod… or Open a fit (Tasks 7.1 and 7.9), a
  project of Stage 6, or the `.mod` panel.
  - A `vshift` on one data line that the row's other lines do not share is that snip's
    own.
  - A `constant` that covers some of a row's snips, but not all, is their own. When it
    covers more than one, the Snip box says which snips share it.
  - When every snip of a row has its own zero level (RJC's J193957m100241), the Data
    tab's cell reads "Off · 4 snips their own". On gives the row's `constant` only the
    snips that have none of their own. With none left, On is disabled, and its tooltip
    says why. ALIS never sees two zero levels for one snip.
- **Check.** Made into tests first (`test_dashboard_datasets.py`,
  `test_dashboard_snips.py`, `test_dashboard_qt_regions.py`):
  - RJC's J193957m100241 project (skipped when absent) reads with each of its four
    snips' own zero level, and the Data tab's On is disabled;
  - a synthetic `.mod` with `shift=vshift(…)` on one of a row's two data lines, loaded
    with Load a .mod…, reads as that snip's own shift;
  - a project of Stage 6, with a dataset's zero level and shift, reads as before;
  - a snip given its own shift and zero level gives a model that ALIS reads, with the
    shift on that data line alone and the specid in one `constant` only;
  - back to As the dataset gives the text before, byte for byte;
  - a region drawn on one dataset lands on the other, moved by that snip's own shift;
  - a synthetic two-dataset project, with one snip shifted by 3 km/s, fits the shift
    back within 3σ;
  - SNIP and CLEAR behave as above, and every undo is exact;
  - in the window: the Snip box's two choices, and the Data tab's count.

**7.14 — The documentation's design document (item 4, new; Q7.13).**
- **The aim.** Item 4 asks for documentation on readthedocs that describes ALIS and the
  dashboard, and how to use them. It should cover all of ALIS's functionality, and
  build on the LaTeX manual of v1.0 in `doc/tex_files/`. The documentation itself is
  later work. This task writes its design document, as `dashboard_stage0.md` was
  written for the dashboard.
- **The document** (Q7.13: `claude_prompts/documentation_stage0.md`) covers:
  - **what exists:** the chapters of `doc/tex_files/` and what has changed since
    v1.0, its `to_be_completed.txt`, `doc/ALIS_workflow.md`, `doc/installing.rst`,
    the docstrings and the examples;
  - **the readers,** and what each needs;
  - **the site's outline,** chapter by chapter, with the source of each and what must
    be new (v2: the bundle, the dashboard, the GPU path, the settings);
  - **the reference lists,** the model functions and the settings, made from the code
    so that they stay complete;
  - **the tools,** and how the build is checked (`build-docs`, warnings, links);
  - **the dashboard's screenshots,** made by the scripts that make the review pages;
  - **the stages** of the work, each with its check, and queries with leans for RJC.
- **Check.**
  - The document covers every chapter of `doc/tex_files/alis_v1p0.tex`.
  - It covers every model function in `alis/functions/` and every setting in
    `alis/config.py`. A short script compares them with the document's lists, and its
    output goes in the log.
  - No code changes.

**7.15 — Close the stage.**
- **The review.** Publish the screenshots of what changed for RJC's review, on a new
  page with the earlier pages' stylesheet, and apply the review. The page shows:
  - Task 7.1's Load snips… and Load a .mod…;
  - the cards' exclusions and ratio rows, and the temperature modes;
  - the Add a spectrum dialog, Open a fit (.mod)… and its question, the start page's
    recent list and the toolbar;
  - Inspect with the continuum;
  - the Snip box's shift and zero level.
- **The documents.** `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md` (with ALIS's change
  to the zero level, Task 7.7), `tests/README.md` and the `gui-component` skill brought
  up to date.
- **The decisions.** Record them in `ALIS_v2_dashboard_prompts.md` (D78 onwards, from
  Q7.1–Q7.15 and the review), saying which earlier decisions they refine (D14, D21,
  D35, D45, D64, and F4). Update the "After v1" list of `dashboard_stage0.md`: S23
  comes off (Q7.5), and the documentation goes on as its own stages, planned by 7.14's
  document.
- **Check.**
  - Every item of the list has its task logged as done, with its check passing.
  - The `unit`, `gui` and `fast` batches pass. Their counts go in the log, against the
    baseline of Prompt 1 (2002 unit tests passed with 119 skipped; 190 `gui` tests
    passed).
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44).
  - Task 7.7 changes ALIS outside `alis/dashboard/` (Q7.14), so the regression harness
    runs again at the close, with its `test_bundle` mode.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added; it documents the project
  model, the history, the gate, the spectrum view and the Fit tab's runs.
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task; the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where an item touches a fit's results.
- `build-docs`: what the documentation's build has now, for Task 7.14.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D77.
- `claude_prompts/dashboard_stage6.md` and its log: the Fit tab, and how v1 was
  closed; the earlier stage documents for the tabs they built.
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
- `context/fitting_examples/Temperature/`: RJC's examples for items 1, 2, 5 and 6
  (`context.md` describes them). `J193957m100241/model/J193957m100241.model` is RJC's
  project for items 11, 12 and 14 (Q7.11, Q7.12).

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

*Q7.6–Q7.13 were raised on 2026-10-31 (Prompt 2), about items 2–14. Tasks 7.2–7.14
follow the leans.*

**Q7.6 — Excluding a component from every ion (item 2, Task 7.4).** Excluding an ion
comments out the component's line of it (D64). A component is read from its live
lines, so excluding the last ion it has asks only to remove the component. Once H I is
excluded from a1, excluding D I as well therefore removes a1. Two ways to keep it:
- **(a) A component can be excluded from every ion.**
  - All its lines are commented out, and it stays on the Components tab: a greyed
    card with Include for each ion, in its place among the system's components, so
    that no id changes.
  - It is read from commented-out lines that share a redshift label no live line
    carries. A component commented out by hand is therefore shown the same way.
  - ✕ still removes it.
- **(b) Excluding H I takes D I with it** when a link computes D I's log N from H I's,
  as a ratio line goes with the ion it gives. Excluding the last ion still asks to
  remove the component.

(a) gives a way to do what you now do by hand, whatever ties the ions. (b) changes
what excluding H I does, which the item calls the expected behaviour. It also still
leaves no way to keep a component out of the fit without removing it. With (a), the
preview of excluding H I also says what becomes free ("D I's log N in a1 is computed
from H I's by a link: it becomes free").

My lean: (a).

**Response:** I agree with option (a). However, we need to make sure that if a component is excluded (commented out) for all ions, any tie labels for that component should not be used for another component (either existing, or newly added). This check ensures that if this component is reintroduced to the fit, it does not inadvertently affect other components that may have been added in the meantime.

**Q7.7 — What a card shows for an ion given by a ratio (item 5, Task 7.3).** In
J101155p294141, D I is `voigt ion=2H_I/1H_I -4.5974FIXDH`. That one line gives
log(D I/H I) for every H I component that shares its snips, so each card should show
D I as present. Two ways:
- **(a) The ratio itself:** a row "↳ D I · log D I/H I −4.5974 ■", typed and badged
  like any value, with a note naming the other components that share it ("shared
  with a2"). Typing it changes it for all of them, as for a tied value.
- **(b) D I's log N as computed:** log N(H I) plus the ratio, shown read-only like a
  value a link computes (D65), with the ratio named in its tooltip.

(a) shows the parameter the fit actually has (here fixed), which is what a D/H
measurement turns on, and where to change it. (b) shows a column density that can be
compared with the card's other ions.

My lean: (a).

**Response:** I agree with option (a). Displaying the ratio itself provides clarity on the actual parameter being used in the fit, and allows users to easily identify and modify it if necessary. The note indicating which components share the ratio is also helpful for understanding the relationships between different ions.

**Q7.8 — Excluding from a component with no redshift label (item 6, Task 7.2).** A
commented-out line is tied to its component by the redshift label they share (Q5.15).
A component with no label (redshift `2.50360` rather than `2.50360za1`) cannot be
excluded from an ion today. The message asks the user to tie it from a notice, but a
component of one line never has that notice. Two ways:
- **(a) The exclusion writes the label.** The step gives the component's lines a
  redshift label, as the notice's fix does, then comments out the ion's line. The
  preview lists the lines that gain the label. A component of several untied lines is
  tied in the same step.
- **(b) Ask first.** The message offers "Tie and exclude", which makes the same change
  once the user agrees.

With (a), excluding takes one click, as any other exclusion does, and its preview
already shows what changes. On a component of one line, a lowercase label changes no
value, and it is undone with the exclusion.

My lean: (a).

**Response:** I agree with option (a). Automatically writing the redshift label when excluding a component with no label streamlines the process and reduces the number of steps for the user. The preview listing the lines that gain the label provides transparency and allows users to see what changes will be made before confirming the action. We just need to make sure that the redshift label is unique and does not conflict with existing labels in the project.

**Q7.9 — One way to open a .mod (item 3, Task 7.9).** There is a distinction:
- Import a fit (.mod)… makes a new project from a `.mod` and its snips, with no source
  spectrum;
- Load a .mod… brings a `.mod` into the open project, checking its snips against the
  project's spectra (Task 7.1).

Which of the two applies follows from what is open, so one item can do both:
- **(a) One item, Open a fit (.mod)….**
  - With no project open, or a project with no source spectrum, it opens the fit as a
    new project.
  - With a project that has a source spectrum, it previews loading the fit into that
    project. The preview also offers Open as a new project.
  - When the snips do not match the project's spectra, the message says why, and
    offers the same.

  The start page's button and `alis fit.mod` open a new project, as now.
- **(b) Keep both, named for what they do:** New project from a fit (.mod)…, and Load
  a fit (.mod) into this project…, the second enabled only with a source spectrum.

My lean: (a).

**Response:** I agree with option (a). If no project is open, or a project is open with no source spectrum, we should prompt the user and ask if they would like to load the complete source spectrum before loading the .mod file. We should also make it clear that this is the recommended approach.

**Q7.10 — Pure thermal, and what a drag moves (items 11 and 12, Tasks 7.5 and
7.6).** D21's pure thermal is b_turb fixed at 0 with T free. Three points:
- **The limit.** voigt.py keeps b_turb at 0.5 km/s or more, so pure thermal's
  b_turb = 0 is refused by the pre-flight check, and would stop ALIS. I propose that
  pure thermal also writes `lim param <label> [0.0,None]` for the component's b_turb,
  in the same step, and that leaving pure thermal takes it out again (when nothing
  else uses the label). This is the narrowest change. The alternative,
  `lim voigt bturb [0.0,None]` written once in the model, would lower the limit for
  every line.
- **What counts as pure thermal.** Your Temperature fits write it as b_turb `0.5BFIX`
  with T free. I propose that a b_turb fixed at 0.5 km/s or less, with T free, is
  shown as pure thermal too, and left as written.
- **What a horizontal drag moves** (item 11):
  - T when b_turb is fixed;
  - b_turb when T is fixed, or when both are free (D21);
  - nothing when both are fixed. log N still moves vertically, and the status bar
    gives the reason.

My lean: as proposed.

**Response:** I agree with the proposed handling of pure thermal and the behaviour of the horizontal drag. I tried manually changing the b_turb limit to 0.0 in the model, but it was rejected by the dashboard. Writing `lim param <label> [0.0,None]` for the component's b_turb when pure thermal is selected seems like a reasonable solution to allow for pure thermal fits without affecting other components. However, please check that the dashboard still allows ALIS to run with this change, and that it does not introduce any unintended side effects.

**Q7.11 — The zero level that did not change (item 14, Task 7.7).** ALIS fits a zero
level: on a synthetic saturated Lyα with a zero level of 0.05, `constant 0.0` in the
`zerolevel` section gives 0.054 ± 0.003. To find where your case went wrong, it would
help to know:
- how the zero level was added: with the Data tab's Zero level (On), or typed in the
  `.mod` panel, and in which section;
- whether the line's trough was inside the fit regions. Only fitted pixels near zero
  flux set a zero level; without them it stays where it starts;
- where you saw that it did not change: in the `.mod` panel (which changes only with
  Continue from best fit, D74), in the results table, or in the model drawn;
- the project or `.mod`, if you still have it.

My lean: Task 7.7 tests the dashboard's two ways of adding a zero level, from Run to
Continue from best fit, and fixes whatever fails. With your answer, your case becomes
one of its tests.

**Response:** I added the zero level by typing it directly into the `.mod` panel, in the `zerolevel` section. I have added the `.model` file that I was using as context: `context/Temperature/J193957m100241/J193957m100241.model`.

**Q7.12 — A snip's own shift and zero level (items 9 and 14, Task 7.13).** Today both
belong to a dataset (D14), and are set on the Data tab: one `shift` for all of a
row's snips (none for the reference), and one `constant` for all of them. I propose:
- **Where:** the Regions tab's Snip box, for the selected dataset's snip, gains Shift
  and Zero level. Each is "As the dataset" (the default) or "Its own".
- **Its own shift:** a `vshift` with its own label on that data line, free at 0 km/s,
  then typed and badged as on the Data tab. A snip of the reference row can have one
  too.
- **Its own zero level:** a `constant` with that snip's specid alone, taken out of the
  dataset's, since ALIS allows one zero level per specid. A snip can have one whether
  or not the dataset's zero level is on.
- **What follows:**
  - regions copied between datasets (D55) move by each snip's own shift;
  - SNIP gives a new snip the dataset's shift and zero level;
  - CLEAR removes a snip's own;
  - the Data tab's cells say how many snips have their own ("+1 snip").
- **Back to As the dataset:** restores the dataset's word and specid, and removes the
  snip's own.

My lean: as proposed.

**Response:** When loading a `.mod` into a project, please ensure that the snips in the `.mod` file are correctly loaded into the `Data` tab, so that users can incorporate a shift and zero level for each snip. It's possible that the current version of loading a `.mod` file may not incorporate this correctly.  You can check this with the context file: `context/Temperature/J193957m100241/J193957m100241.model`. Given the above, I agree with the proposed changes above.

**Q7.13 — The documentation's design document (item 4, Task 7.14).** I propose that it
is written as `claude_prompts/documentation_stage0.md`, in the pattern of
`dashboard_stage0.md`. It would cover what exists, the readers, the site's outline
with the source of each chapter, the tools and how the build is checked, the stages
of the work, and its own queries. The documentation itself then follows in stages of
its own after this one, and is added to the "After v1" list at the close. The tools
(Sphinx and its format, the theme) are among that document's queries.

My lean: as proposed.

**Response:** I agree with this plan.

*Q7.14–Q7.15 were raised on 2026-10-31 (Prompt 3), from RJC's J193957m100241 project.
RJC agreed to both leans (Prompt 4), which Tasks 7.7 and 7.5 follow.*

**Q7.14 — Letting a zero level go below 0 (item 14, Task 7.7).** Your four zero levels
stayed at 0 because `constant`'s value has a lower limit of 0
(`alis/functions/constant.py`). That limit is meant for a continuum, but a zero level
inherits it. Your troughs lie slightly below zero, so the fit pins each zero level on
that limit. With the limit lifted, Lyα's zero level fits to −0.0136 ± 0.0009, and χ²
falls from 919 to 695 (see Design). A zero level corrects an offset of either sign.
Two ways:
- **(a) In ALIS.** A `constant` in the `zerolevel` section has no lower limit, and an
  emission `constant` keeps its limit of 0.
  - This changes ALIS outside the dashboard, where a model line takes its function's
    limits, so the regression harness is run.
  - An old `.mod` whose zero level sat at 0 would now fit differently, and correctly.
- **(b) In the dashboard only.** The dashboard writes each zero level with a label and
  `lim param <label> [None,None]`. A zero level typed in the `.mod` panel without one
  gets a notice, with that fix.
  - ALIS is unchanged.
  - But every zero level needs an extra line, and a `.mod` run outside the dashboard
    keeps the trap.

My lean: (a).

**Response:** I agree with option (a). Allowing a zero level to go below 0 in ALIS makes sense, as it can correct for offsets of either sign. This change will ensure that fits are more accurate and reflective of the actual data.

**Q7.15 — Pure thermal, and a total b of 0 (item 12, Task 7.5).** I checked pure
thermal for side effects, as Q7.10's response asks. With b_turb at 0, the total b is
the thermal b alone, so it is 0 when T is. T's lower limit is 0 K, so a fit that drives
T down to it stops ALIS ("Voigt Total Doppler parameter is zero"). That is the last
error saved in your J193957m100241 project. It may be what stopped you when you tried
a b_turb limit of 0. Two ways:
- **(a) A floor for T.** Pure thermal also writes `lim param <T label> [1.0,None]`,
  and takes it out with the b_turb limit when the mode changes.
  - At 1 K, H I's thermal b is 0.13 km/s, so the floor never shapes a fit.
  - A T that ends on it is flagged "at limit" in the results.
- **(b) A floor for b_turb instead.** Pure thermal fixes b_turb at 0.01 km/s, with
  `lim param <label> [0.01,None]`, as your J193957m100241 does.
  - b_turb then adds 10⁻⁴ km²/s² to b², and T needs no limit.
  - But b_turb is not 0, as item 12 asks.

My lean: (a).

**Response:** I agree with option (a).

*Q7.16 was raised on 2026-11-01, from RJC's review of the Task 7.15 page (a comment on
the D I ratio figure). The export is paused until it is answered; the other review
comments are applied (see the log).*

**Q7.16 — Exporting only what is needed (review).** File ▸ Export plain files… now
writes everything `run_alis` needs (the model, every snip, the latest run's outputs).
Your comment asks for a dialog with checkboxes instead: snips, the starting model,
the best-fitting model. Ticking snips opens a column picker, defaulting to Wavelength,
Flux, Error, Fit range, Zero level (when the snip has one), Continuum and Best-fit
model. Each snip goes to its own ASCII file, in a folder you choose. Some details
need deciding before I build it:
- **(a) Fit range.** Written as 1 for a fitted pixel and 0 otherwise. This is how
  ALIS reads a `fitrange` column, so an exported snip can be fitted again as it is.
  My lean: yes.
- **(b) File names.** Each file is named after its snip's label (`OI_1302.dat`),
  with a `#` header line naming the columns. A file that exists already is
  overwritten only after a question. My lean: yes.
- **(c) Which best fit.** The best-fit model, continuum and zero level come from the
  run the Fit tab shows: the latest, or the one chosen in the run history, as Use
  best fit does. Without a run, the best-fit options are greyed, with the reason.
  The starting model, and an extra Starting model column, are always available.
  My lean: yes.
- **(d) The two models.** The starting model is the `.mod` as it stands. The
  best-fitting model is ALIS's `.mod.out` of that run. When snips are exported too,
  both models' data lines point at the exported files, so `run_alis` runs either one
  in the chosen folder. My lean: yes.
- **(e) Blinding.** As now, the starting model writes its hidden starting values in
  plain text, after a warning. The best-fitting model would write hidden best-fit
  values, so it is greyed while any value is hidden. The snips' Best-fit model column
  stays, as the panels already draw that curve. My lean: yes.
- **(f) The other columns.** The picker can also offer, unticked by default:
  Starting model, Normalised flux and error (divided by the continuum), and
  Residual ((flux − model)/error). Each column can be ticked and moved up or down.
  My lean: offer them.
- **(g) Everything, as now.** A fourth checkbox, "Everything run_alis needs (model,
  snips, run outputs)", keeps today's export. The menu item becomes Export…, opening
  the dialog with your three boxes ticked. My lean: yes.

**Response** (on the review page's thread, 2026-11-01): "Yes, the Fit Range should be
0 and 1. Yes, the files should be named after their snip labels. Something like:
`ProjectName_Ion_TransitionRestWavelength_zabsBLAH_fit.dat`, where "BLAH" should be a
4 decimal place redshift value of the system this transition corresponds to. If you
can suggest something that might be more suitable, please do!"

*Refinements of (b) proposed in reply, which the export follows unless RJC says
otherwise: the ion without spaces or ALIS's mass prefix (HI, CIV, FeII); the rest
wavelength to 2 decimals, so that close lines (Si II 1260.42, Fe II 1260.53) do not
share a name; `_2`, `_3` added when two snips still would; a snip holding several
transitions named after the one it was cut for, the others in the file's header; the
project name's spaces and slashes as underscores. For example
`J101155p294141_HI_1215.67_zabs2.5020_fit.dat`. (c)–(g) follow the leans unless RJC
says otherwise.*

**Response:** I agree with the proposed suggestions. Please note that blind parameters and blind models must stay hidden in the exported files. For example, blind parameters could be replaced with the word BLIND. Also, just to be clear, note that the output snip naming convention should not include any mass number on the ion. Your example is correct (`J101155p294141_HI_1215.67_zabs2.5020_fit.dat`), but we do not want to  use `1HI` in the file names (for example, we should avoid `J101155p294141_1HI_1215.67_zabs2.5020_fit.dat`).

*Applied (2026-11-01): hidden values are written as `BLIND` in both models and in
"everything" (with `blindseed`, `blindrange` and the `.mod.out`'s copy of the input
model), so the best-fitting model is no longer greyed for a blinded project and the
plain-text warning is gone; no ion in a file name has a mass number (`13C IV` is
`CIV`); a hidden redshift is `zabsBLIND`. See D88 and the log.*

## Prompts

1. Please read this document, with my current list of changes and bug fixes, and the work of Stages 0–6 (their documents and logs). I have written Task 1 based on a feature that would be good to include in the ALIS dashboard. Let's implement this first, and then I will draft a more complete list of bugs, improvements, and additions to the dashboard. If anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please execute Task 7.1.

2. Please read the list of changes and bug fixes, and the work of Stages 0–6 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Changes and bug fixes`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. I have already written Task 7.1. Please write all of the remaining tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be written in the same style as Task 7.1, and should be numbered sequentially (7.2, 7.3, etc.). The final Task (currently labelled 7.N) should be to close the stage, and should include the checks that will verify that the stage is complete.

3. Please read the list of changes and bug fixes, and the work of Stages 0–6 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Changes and bug fixes`). I have responded to your first round of queries. Please review all of this information, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. I have already written Task 7.1. Please write all the remaining tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be written in the same style as Task 7.1, and should be numbered sequentially (7.2, 7.3, etc.). The final Task (currently labelled 7.N) should be to close the stage, and should include the checks that will verify that the stage is complete.

4. I have responded to your next round of queries. Please review all of this information, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please update the ordered list of tasks, if needed. I have already written Task 7.1. Please write all the remaining tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be written in the same style as Task 7.1, and should be numbered sequentially (7.2, 7.3, etc.). Once everything is clear, please let me know when you're ready to execute the tasks.

5. Please execute the tasks in order, starting with Task 7.2-7.15. There's no need to pause between tasks, unless you have queries that require my thoughts and responses. If you have queries during development, please pause the development and log the queries and responses in the Queries section.

6. You mentioned what should happen with the command line `run_alis --extract`, This should do the same thing as the dashboard option "Everything run_alis needs". I agree with omitting the zero level and continuum if the user has blinded those values. Please note that the dashboard should still show greyed out characters for the blinded entries, as before, it's only the output .mod files that should show BLIND for blinded parameters.
