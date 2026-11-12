# Prompt file for ALIS software dashboard creation -- STAGE 11

> **Feedback from users, after Stage 10's list closed.** Stage 10 took a closed list
> of users' feedback and RJC's own on the Figure tab (Q10.3 (a)). Stage 11 takes the
> feedback that arrives after that list closed. RJC collects each point in the
> Feedback section below. Claude turns the list into a design, queries and tasks,
> each with its check, and builds them in order, with a review before the stage
> closes.
>
> The rules of Stages 3–10 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D119 at the close of Stage 10).
> "Q0.*n*" to "Q10.*n*" refer to `dashboard_stage0.md` to `dashboard_stage10.md`.
> What follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-08 (Stage 10's Task 10.17). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is. Items 2 and 3 carried over, and Q11.1 with them, were added at Stage 10's close
(2026-11-09), from what RJC's Prompts 6 and 7 found. Prompt 1 (2026-11-10) added
Q11.3, where Feedback point 1 (the GPU) is explored, and Q11.4, on the response to
Q11.1. Prompt 2 (2026-11-12) wrote the design and the first tasks (Feedback point 2
and the items carried over), with Q11.5 and Q11.6; point 1 waits in
`deferred_work.md` (§1.4). Prompt 3 (the same day) revised them after the responses,
with the close, and raised Q11.7–Q11.10. Prompt 4 added the fix of a bug found at
Prompt 3 (Task 11.2, the tasks after it renumbered) and the user's choice of how
unassigned snips are loaded, and raised Q11.11.*

## Feedback

*Collected by RJC from users of the dashboard: one point per item, numbered, added as
it arrives. For each point:*
- *what the user did and saw, and what they expected instead;*
- *its kind: **(bug)**, **(change)** or **(new)**;*
- *the tab or dialog it concerns;*
- *where it can be reproduced: a project, or the model and data, and the steps. A
  user's files are referenced from here, and each test that needs them is skipped
  where they are absent (D99);*
- *who raised it, only if they agree to be named.*

*A point already on the list is not added twice: a note goes under the first, saying
how many users raised it. Claude reads this list and does not add to it.*
1. **(new)** Would it be better to make the GPU handling different? If the subpixellation does not change, it may be faster if all large arrays are stored on the GPU, and the only thing that is passed to the GPU on each iteration is the parameters (a relatively small array). Then, the only thing returned from the GPU is the chi^2 value (a single number). This would be a big change, but it may be worth considering. To explore this, we should create a separate design document (and GitHub branch to allow detailed testing and to explore this idea without breaking the current code) and make small steps to see if this can be implemented effectively. Since the dominant bottleneck with the GPU is transferring the large arrays to/from the GPU, we should explore ways that we only transfer large arrays to the GPU once, and then only transfer small arrays (the parameters) to the GPU on each iteration, and receive back a single number. Please consider this, and if it is worthwhile exploring, write a design document that we can explore independently of the other changes/fixes that are to be implemented during this stage.

2. **(change)** When users open from a .mod file, they are recommended to load the complete spectrum. How should we handle the case where the user has a .mod file that requires more than one spectrum to be loaded? The current functionality does not work for this case, and it is not clear how to handle it. We should consider how to handle this case, and if we need to make changes to the code to support this functionality.

## Carried over from Stage 10

*Found by Claude during Stage 10 and not done there. RJC decides whether each joins
this stage (Q11.1).*

1. **(change)** The line IDs' choice (This transition, This ion, the model's ions
   and strong lines, Every line) is the Regions tab's alone. The Data tab keeps its
   own switch for every line (D59), and the Components tab labels as before; a user
   may want one choice for every tab.
2. **(bug)** A button made from an action drops the action's trailing "…" unless the
   window sets its text (Prompt 7 set Random restarts…'s). The Components tab's Add
   ion, Add system and Add interloper buttons open a dialog or a menu first, yet show
   no "…". Each is one line in `qt/window.py` or `qt/components.py`.
3. **(change)** A label asked for before fixing, tying or limiting an unlabelled
   value is suggested from the ``x`` scheme (``bx1``, `edit.suggest_label`), for
   every component; it could take the component's own scheme (its system's letter,
   or ``ly`` for a line of the Lyα forest, D119).

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D119 and the stage documents. A point that asks for what a decision ruled out
  becomes a query that quotes the decision, not a task.
- **A bug becomes a test first.** It is reproduced from the user's project where there
  is one, or from a synthetic project otherwise (`tests/dashboard_synthetic.py`, and
  the helpers of `test_dashboard_figpresets.py` and `test_dashboard_figscript.py` for
  the Figure tab). The test fails before the fix and passes after it.
- **Outside the dashboard.** A point about ALIS itself, or about the documentation
  (planned in `documentation_stage0.md`), becomes a query: the dashboard's stages
  change ALIS only where a query agrees.
- **Users who disagree.** Two points that ask for opposite things become one query,
  with both described.
- **Grouping and order.** Points are grouped by the part of the window they touch
  (Data, Regions, Components, Fit, Figure, the `.mod` panel, Export), and a bug comes
  before the work that builds on it.
- **Shown before it closes.** Every change a user sees is shown on a review page,
  built by `doc/dashboard/stage11/` with the earlier pages' stylesheet, and RJC's
  comments on it are applied before the stage closes.

### 11.1–11.11 — Feedback point 2 and the items carried over

*Written by Claude on 2026-11-12 (Prompt 2), from Feedback point 2, the three items
carried over from Stage 10 (Q11.1: items 1 and 2 join; item 3 as Q11.4's response
asks), and the code at the start of the stage. Feedback point 1, the GPU, waits in
`deferred_work.md` §1.4 (Prompt 2; Q11.3 (b)). Each point's design is in its task
below.*

**What was found.** Each was reproduced off-screen, with scratch scripts and a scratch
`gui` test, before its task was written. No code was changed.
- **Point 2** (Open a fit (.mod)… of a fit whose snips come from several spectra).
  The fit used: the metal example's spectrum and a copy with its flux 1% higher, one
  snip cut from each, and a `.mod` with one resolution label per spectrum.
  - `loadfit.load_mod_plan` loads it into a project of both spectra: each snip is
    matched to the spectrum it was cut from, whichever row that is. Through the
    window, Open with its spectrum (recommended)… loads it too, when both spectra are
    added in the New project dialog, which takes any number of spectra (D46).
  - What fails is the way there. The question asks for "the full spectrum its snips
    were cut from", and the New project dialog says nothing about the fit. With one
    of the two spectra added, the load is refused: each snip of the other is listed
    ("its flux differs from specA's at 120 pixels of its 120"), and the message ends
    "add it on the Data tab". No project exists yet, so there is no Data tab to add
    it on. The choices are Choose another spectrum…, which opens an empty dialog
    again, Open without a spectrum, and Cancel.
  - Every spectrum is given the dialog's FWHM, which is the `.mod`'s first `vfwhm`
    (`loadfit.mod_hints`). The preview then says "The .mod's FWHM for specB is
    6.0 km/s; 4.0 km/s was given", and Reset FWHM would return specB to 4.0 km/s, a
    value the user never gave it.
  - **A bug, found on the way.** The reference stays the first spectrum added (D14),
    whatever the `.mod` says. With a `.mod` that frees the first spectrum's shift and
    leaves the second unshifted, the Data tab marks the first as the reference, with
    a free shift, and the second not, with none. A project inferred from a `.mod`
    alone already takes the `.mod`'s reference (`Project._infer_reference`: the first
    row whose shift is absent or fixed at 0), but the rows of a new project are
    already marked, so the rule is never applied.
  - A dataset of a `.mod` is not always one spectrum: two exposures of one instrument
    share a resolution label but are two files, so two rows (D35). The design counts
    the snips each spectrum holds, not one spectrum per dataset.
  - RJC's J0035m0918 `.mod`s (`context/fitting_examples/VMP_DLA/J0035m0918/model/`)
    do not load against the three spectra of the Stage 10 project
    (`context/feedback01/`, sources in RJC's research folder): their snips are on
    other pixels ("90 of its 120 pixels are not on J0035m0918_ESPRESSO's pixels"),
    so they were cut from other reductions. The point names no files, so its tests
    are synthetic.
- **Item 2.** A button made from an action shows Qt's icon text, which drops a
  trailing "…". On a project of the metal example, with every tab shown, nine
  buttons drop it: Add ion…, Add system…, Add interloper…, ⌖ Identify a feature…,
  Open…, Commit run…, Continue from best fit…, Convergence check… and Review clipped
  pixels…. Two keep it, Export… and Random restarts…, whose text the window sets by
  hand (`qt/window.py`).
- **Item 1.** Line IDs are drawn on two tabs. The Data tab's spectrum has the switches
  "Transitions of z" and "All lines", for the primary system only. The Regions tab's
  spectra have D114's four choices, kept in `ui/view.json`. The Components tab draws
  its components' ticks, and the Fit tab's Inspect draws no line IDs.
- **Item 3 and Q11.4's response** (`LS#`), checked against the code and the tests:
  - The labels the dashboard makes already follow it for z, b and T. + Add component,
    the fixes of D20 and D22, and a temperature mode's own labels (D93) write `za1`,
    `ba1` and `ta1`: the system's letter (`ly` for a line of the forest, D119), then
    the first number that none of the three uses (`edit.new_component`).
  - Three places do not follow it. The label suggested before fixing, tying or
    limiting an unlabelled value uses `x` (`edit.suggest_label`; `zx1` in
    `test_dashboard_edit.py`). A component of no system is given its redshift label
    with `x` too (`components._tied_first`). A column density gets no suggestion
    (Q2.3).
  - "#" is not always the component's number. A component's number is its place by
    redshift in its system (`a1`, `a2`, …), worked out at every read, while labels are
    never renamed (Q2.3). A component added between two others takes the number
    between them and renumbers those after it, but its labels take the first free
    number. A temperature mode's new label also takes the first number that none of
    z, b and T uses, which can differ from the number of the component's other labels.
  - `n` and the component alone (`na2`) cannot label a column density: each ion of a
    component has its own, and two values that share a label are tied by ALIS.
    Q11.5.

**Revised (Prompt 3, 2026-11-12)** after RJC's responses to Q11.5 and Q11.6.
- *Point 2 (Q11.6's response).* The user adds every spectrum they have, and a dialog
  then assigns the snips: the datasets listed at the top, the chosen dataset's snips
  on the left, and the snips of no dataset on the right, moved by drag and drop, then
  Confirm. The dashboard's best guess fills it, from the snips that each spectrum
  holds. A snip left on the right is loaded without a source. This replaces Create
  waiting for every dataset. RJC's responses to Q11.8 and Q11.9: the dialog opens
  every time a fit is loaded with spectra, and is used in every way a fit is
  loaded. No snip is refused: one that no spectrum holds is cut again from the
  spectrum it is given, loaded unassigned, or left out.
- *Item 3 (Q11.5's response).* A column density's label adds its ion (Q11.5 (a)). A
  label keeps the number it was made with, so a component's number in its labels
  is the order it was given them, not its place by redshift. A new action renames a
  system's labels to follow their components' order by redshift, after a warning.
  Undo takes it back, and the cards keep their order by redshift (Q11.7 (a)).
- *"Only looking in the first dataset"* (Q11.8's response). No code looks for a
  snip's spectrum in the first dataset alone: `loadfit.match_row` tries every row.
  But a snip that no spectrum holds is refused with the reason of the row nearest
  to matching, the first when they tie. With RJC's J0035m0918 files and all three
  spectra, the O I 1039 snip (3470 Å) is refused as "outside the range of
  J0035m0918_ESPRESSO", though J0035m0918.dat covers 3128–5966 Å: the snip is
  within its range but not on its pixels, and the message does not say so. Tasks
  11.3 and 11.4 give a reason for every spectrum (Q11.9).
- *No snip refused* (Q11.9's response). A snip that no spectrum holds is assigned to
  a spectrum and cut again from it, loaded unassigned (cut from no spectrum), or
  left out with the lines only it uses. RJC's responses to Q11.10: one unassigned
  dataset for each dataset of the `.mod` where the dashboard can hold the snips
  together (Prompt 4 makes it the user's choice, below), a Load tick for each snip, a
  snip cut again over its own range after a warning, and Load snips… through the
  same dialog.
- *Overlapping snips in one dataset* (Q11.10's response). The dashboard holds two
  snips whose ranges overlap in one dataset: the model and ALIS take them, and the
  check for pixels fitted twice (D19, `snips.shared_pixels`) compares every pair of
  snips by their pixels' wavelength, flux and error, whatever their dataset. But the
  Regions tab shows one snip of a transition per dataset (`snips.snips_of` keeps
  the first it meets), so a second snip of the same transition in that dataset
  would be out of sight there. An imported fit already meets this: Open without a
  spectrum groups two such snips (two exposures sharing a resolution label) into one
  dataset, and the Regions tab shows only the first.
- *Prompt 4.* The bug is fixed in this stage: the Regions tab shows one strip per
  snip (11.2). The user chooses how unassigned snips are loaded, one unassigned
  dataset or one for each snip, and is warned when snips that would share a dataset
  overlap in wavelength, with each snip its own dataset advised (11.3–11.5). This
  replaces the rule, written at Prompt 3, that put a second snip of a transition in
  another unassigned dataset.

**What stays as it is.**
- A fit whose snips all come from one dataset is asked about as now (D78); the
  assignment dialog then shows where each snip goes, as for any fit (Q11.8 (b)).
- Labels are renamed only by the user: as now, or by the new action (Q11.5's
  response). The datasets' labels (`fwhm#`, `vs#`) are as now.
- A component's name on its card (`a1`, `a2`, …), the panels' ticks and the figures'
  numbers follow redshift, as now (Q11.7 (a)).
- The Components tab's ticks and the Fit tab's views draw no line IDs, as now.
- ALIS outside `alis/dashboard/` does not change.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage11_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (11.1, 11.2, …), and per item carried over
from Stage 10 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*Tasks 11.1–11.11 were written by Claude on 2026-11-12: first at Prompt 2, then
revised at Prompt 3 after RJC's responses to Q11.5 and Q11.6. Those responses
replace Create waiting for every dataset with a dialog that assigns the snips
(11.3–11.5), keep a label's number as it was made, and add an action that renumbers
a system's labels (11.8, 11.9). Prompt 3, run again after the responses to Q11.7
((a) for both) and Q11.8 ((b), (a), (a)), revised 11.3–11.5: the dialog opens every
time a fit is loaded with spectra, and every spectrum's reason is given. Run a third
time after the response to Q11.9, it revised them again: no snip is refused, and
each is held, cut again, unassigned or left out (11.3). Run a fourth time after the
response to Q11.10 ((a) for all four), it added the warning before a snip is cut
again (11.4). Prompt 4 (the same day) added Task 11.2, the fix of the bug found at
Prompt 3 (an imported fit's second snip of a transition in one dataset, not shown
on the Regions tab), and renumbered the tasks after it. It replaced the rule that
kept two snips of one transition in separate unassigned datasets with the user's
choice of how unassigned snips are loaded, and a warning when snips that would
share a dataset overlap (11.3–11.5). Prompt 5 found every task following the
responses, Q11.11's ((a) for both) among them. The Feedback list closes with these
tasks (Q11.2 (a)), and 11.11 closes the stage. The order puts a bug before the work
that builds on it, and keeps work on one part of the window together:*
- *point 2's bug (11.1);*
- *the Regions tab's bug, since point 2's unassigned snips build on it (11.2);*
- *point 2: its logic, the dialog that assigns the snips, and the ways a fit is
  opened and loaded (11.3–11.5);*
- *every button's "…" (11.6);*
- *the Data and Regions tabs' line IDs (11.7);*
- *the Components tab's labels, then their renumbering (11.8, 11.9);*
- *the review (11.10), and the close (11.11).*

**11.1 — A loaded `.mod` decides the reference (point 2, bug; `loadfit.py`; D14,
D78).**
- **The bug.** A `.mod` loaded into a project of several spectra leaves the project's
  reference where it was, on the first spectrum added. This happens even where the
  `.mod` frees that row's shift and leaves another's at 0.
- **The fix.** The step of `load_mod_plan` makes the reference the first row whose
  loaded snips have no shift or a shift fixed at 0 (`Project._infer_reference`'s
  rule, D14). Where no row has one, the reference stays where it was, and the
  preview says so ("No dataset of old.mod has its shift fixed at 0, so specA stays
  the reference."). A project of one spectrum is unchanged. Undo gives back the old
  flags.
- **Check.** Made into a test first, in `test_dashboard_loadfit.py`, on two synthetic
  spectra:
  - a `.mod` that frees the first spectrum's shift and leaves the second unshifted
    makes the second the reference (before the fix, the first stays), and the
    first's shift reads free;
  - ALIS reads the model, and one undo gives back the text, the files and
    `ui/project.json`;
  - a `.mod` that shifts both keeps the first as the reference, with the note;
  - the existing tests of one spectrum pass unchanged.

**11.2 — Every snip of a transition on the Regions tab (found in Stage 11, bug;
Prompt 4; `snips.py`, `qt/regions.py`; Q2.4, D36, D55, D115).**
- **The bug.** The Regions tab shows one snip of a transition per dataset:
  `snips.snips_of` keeps the first snip it meets in each row, and `snips.strips`
  gives one strip per row. An imported fit (Open without a spectrum) groups two
  snips of one transition into one dataset when they share a resolution label (two
  exposures of one instrument, Q2.4). The second is then shown nowhere on the tab,
  so its regions, continuum and edges cannot be changed there. Unassigned snips
  loaded into one dataset (11.3) would meet the same.
- **The fix.**
  - `snips_of` gives every snip of the transition in each row, in the model's order.
    Its callers take each: the linked regions, the edges' grips, and the offers to
    merge in the other datasets.
  - The Regions tab shows one strip per snip. A dataset with two snips of the
    transition has two strips, each titled with the dataset and the snip's file
    ("vb · J0035_C_II_1334_2_reg.dat"). A dataset whose source covers the transition
    but has no snip of it keeps its one strip.
  - A strip is selected by its snip, not by its dataset. The Continuum and Snip
    boxes, and Adjust fit regions' one spectrum (D115), act on the selected snip.
  - A region added on one strip is added on every snip of the transition, in every
    dataset, moved by each snip's shift (D55). An edge's grip moves that edge of
    every snip (D115).
- **Check.** Made into tests first, on an imported fit with two snips of O I 1302 cut
  from two copies of the metal example's spectrum (different noise, one resolution
  label):
  - in `test_dashboard_snips.py`, `snips_of` gives both snips in their one row
    (before the fix, the first only);
  - in `test_dashboard_qt_regions.py`, the Regions tab shows two strips for that
    dataset, each titled by its snip (before the fix, one). Selecting the second
    shows its continuum and its Snip box. A region drawn on the first reaches the
    second, moved by its shift. Adjust fit regions' one spectrum changes the selected
    snip alone, and the grip moves both;
  - the existing tests of the Regions tab, each with one snip of a transition per
    dataset, pass unchanged.

**11.3 — Where each snip goes: held, cut again, unassigned or left out (point 2;
`loadfit.py`, `snips.py`, `remove.py`; Q2.4, D12, D23, D52, D97, D113; Q11.9,
Q11.10).**
- **What.** No Qt; used by the dialog of 11.4 and the ways of loading of 11.5. No
  snip is refused (Q11.9's response).
  - `fit_datasets(modfile)`: the `.mod`'s snips, grouped as an imported fit groups
    them (`Project._build_rows`, Q2.4: by a shared resolution label, else by the
    file name's stem). Each group has its name, its snips, its FWHM in km/s (its
    `vfwhm`, or None) and whether its shift is free.
  - `snip_holders(fit, spectra)`: for each snip, the rows whose spectrum holds it
    (`compare`: every pixel on the spectrum's, with its flux and error), and for
    every other row, why it does not. Each spectrum's wavelengths are sorted once,
    not once per snip as `positions` sorts them now (J0035m0918's sources have
    77,000–111,000 pixels each).
  - `best_guess(fit, holders)`: each snip assigned to the row that holds it. One
    held by several rows goes to the row that holds most of the other snips of its
    `.mod` dataset (the first, if equal), and is marked as held by the others too.
    One held by none is unassigned.
  - `load_mod_plan(project, modfile, spectra, assignment=None)`: the assignment
    gives each snip one of four fates. Without an assignment, the plan takes the
    best guess.
    - **Held:** assigned to a row whose spectrum holds it. Loaded as now.
    - **Cut again** (Q11.9's response, (1)): assigned to a row whose spectrum does
      not hold it. Its pixels are cut from that spectrum over the snip's own
      wavelength range (Q11.10 (iii)). Its fit range is carried over by
      wavelength: a new pixel is fitted when its step lies within a fitted run of
      the old snip's (D97). A bad-pixel mask column comes with it when the source
      has one (D52). The data line names the new file, named as SNIP names its
      files, and the model's values stay as the `.mod` has them. The user is warned
      first (11.4; Q11.10's response). A spectrum with fewer than 10 usable pixels
      over that range cannot take the snip (D113's rule), and says why.
    - **Unassigned** (Q11.9's response, (3); Prompt 4): loaded as it is, cut from
      no spectrum, in a dataset without a source marked "unassigned". The user
      chooses how (11.4): one unassigned dataset (Q11.11 (i)), or one dataset for
      each snip, named after its file. `unassigned_overlaps` finds the snips whose
      wavelength ranges overlap among those that would share a dataset, for the
      warning of 11.5. Pixels fitted twice are warned as in any dataset (D19). D12's
      limits apply: regions within the snip, and no new snips in that dataset.
    - **Left out** (Q11.9's response, (2)): not loaded. Its data line goes, with
      the lines that only it uses, as CLEAR removes a snip (`remove.plan_snip`),
      and its specid leaves the lines it shares with other snips.
  - **Every spectrum's reason.** Now `match_row` gives the reason of the row nearest
    to matching alone, the first when they tie, so J0035m0918's O I 1039 snip is
    said to be "outside the range of J0035m0918_ESPRESSO" alone, though another of
    its spectra covers it (see the Design). The reasons now name every spectrum
    tried, in the dialog and in Load snips… (Q11.10 (iv)).
  - The preview names each snip cut again (and from which spectrum), unassigned or
    left out, with the lines that change.
- **Check.** New `unit` tests in `test_dashboard_loadfit.py`, on synthetic spectra:
  - the datasets of a `.mod` of two spectra are found, with their snips, FWHMs and
    shifts, and a dataset of two files (one resolution label) is found as one;
  - two spectra on the same wavelengths with different fluxes each hold their own
    snips and none of the other's;
  - a snip held by two identical spectra is guessed by its dataset, and marked;
  - a snip that none of three spectra holds has a reason for each of the three
    (before: one, the first's);
  - a snip cut again from the other spectrum has that spectrum's pixels over its own
    range, its fit range carried over by wavelength, and its data line naming the
    new file; a spectrum with too few usable pixels there cannot take it;
  - unassigned snips load without a source, in one unassigned dataset or in one
    dataset for each snip, as chosen, and SNIP passes over them;
  - the overlaps among unassigned snips that would share a dataset are found: two
    whose wavelength ranges overlap, of one transition or of two, and none for two
    that only meet at an edge;
  - a snip left out takes its data line and its own continuum with it, and leaves
    the other snips' lines;
  - for each fate, ALIS reads the model, and one undo gives back the text, the files
    and `ui/project.json`;
  - the existing tests pass, the refusals' texts updated where they named one
    spectrum.

**11.4 — The dialog that assigns the snips (point 2, Q11.6's and Q11.9's responses;
`qt/dialogs.py`; Q11.8, Q11.10).**
- **What.** The layout of Q11.6's response:
  - **At the top**, the list of the project's datasets (its spectra), one chosen.
  - **On the left**, the snips assigned to the chosen dataset. **On the right**, the
    unassigned snips ("Not from these spectra").
  - **Each snip** shows its file name and its dataset in the `.mod` (its resolution
    label). A snip held by more than one spectrum is marked ("also in specB"). A
    snip that will be cut again is marked ("cut again from specB"). A snip on the
    right gives, as its tooltip, why each spectrum does not hold it, one line per
    spectrum, so that every dataset is seen to have been tried.
  - **Each snip on the right** has a tick, Load (Q11.10 (ii)): ticked, it is loaded
    unassigned; unticked, it is left out. Ticked by default.
  - **How the unassigned snips are loaded** (Prompt 4), a choice under the
    right-hand list: One unassigned dataset (Q11.11 (i)), or Each snip its own
    dataset. One unassigned dataset is chosen to start with (Q11.11 (ii)). The
    choice is greyed while no snip on the right is ticked.
  - **Moving snips**: drag and drop either way, several at once, or → and ← between
    the lists, or a double-click.
  - **Cutting again, after a warning** (Q11.9's response, (1); Q11.10's response). A
    snip dropped on a dataset whose spectrum does not hold it (its wavelengths,
    flux or error differ) asks first: "specB_28Si_II_1304.4_reg.dat will be cut
    again from specA so that it matches it: its pixels become specA's over its own
    range, and its fit range is carried over. To keep it as it is, leave it
    unassigned." Cut again marks the snip so; Cancel leaves it where it was.
    Several snips dropped together are named in one warning. Only a spectrum that
    cannot take a snip (too few usable pixels over its range) refuses the drop, and
    the dialog's line says why.
  - **Confirm** returns the assignment, and the preview follows (11.5). Cancel loads
    nothing.

  The dialog holds no logic: it shows 11.3's best guess and returns what the user
  made of it.
- **Check.** `gui` tests in a new `test_dashboard_qt_loadfit.py`:
  - the best guess is shown, each snip under its spectrum, and a snip of no
    spectrum on the right, ticked;
  - a snip moved to the right and back (with the buttons, and through the lists'
    drop handler) returns in the assignment;
  - a snip dropped on a spectrum that does not hold it asks first: Cancel leaves it
    where it was, and Cut again marks it as cut again from that spectrum; one
    dropped where there are too few usable pixels is refused, with the reason;
  - unticking a snip on the right leaves it out of the assignment;
  - the choice of how the unassigned snips are loaded returns with the assignment,
    and is greyed with none ticked;
  - a snip held by two spectra is marked, and a snip on the right has a reason for
    every spectrum;
  - Confirm returns the assignment, and Cancel returns none;
  - the dialog's text has no design reference (D44).

**11.5 — Opening and loading a fit of several spectra (point 2; `qt/window.py`,
`qt/regions.py`, `qt/dialogs.py`; D46, D78; Q11.8, Q11.10).**
- **What.**
  - **The question** says what the fit needs when its snips come from more than one
    dataset: "Open old.mod with the full spectra its snips were cut from? Its 12
    snips come from 3 datasets: add every spectrum you have in the next dialog. This
    is the recommended way: …". A fit of one dataset is asked about as now.
  - **The New project dialog**, when Open a fit opens it, says how many datasets the
    fit's snips come from. Each spectrum added takes the FWHM of the `.mod` dataset
    whose snips it holds, in place of the dialog's default, so that the preview has
    no FWHM note and Reset FWHM returns to the `.mod`'s value.
  - **The assignment dialog (11.4)** opens every time a fit is loaded with spectra
    (Q11.8 (b)): after Create and before the preview, even when one spectrum holds
    every snip, so that the user sees where each snip goes. It opens the same way
    when a fit is loaded into a project that exists: Open a fit into a project of
    its spectra, and Load a fit into this project (Q11.8 (a)). Open without a
    spectrum has no spectra, so it opens no dialog.
  - **The overlap warning** (Prompt 4). At Confirm, when One unassigned dataset is
    chosen and two of the snips it would hold overlap in wavelength, the dashboard
    warns, naming each pair ("J0035_C_II_1334_1_reg.dat and
    J0035_C_II_1334_2_reg.dat overlap from 4453.2 to 4461.9 Å"), and advises that
    each unassigned snip be its own dataset. The user chooses Each snip its own
    dataset (advised) or One unassigned dataset. Either loads, and the Regions tab
    shows every snip in both cases (11.2). With no overlap, there is no warning.
  - **Load snips…** (snips without a model, Task 7.1) opens the same dialog in place
    of refusing every file when one fails (Q11.10 (iv)). A snip cut again or
    unassigned is added as Load snips… adds its snips; one left out is not loaded.
  - **A refusal after Create** (a snip that is not a table of numbers) offers Back to
    the New project dialog, with its name, folder, spectra and redshift kept. This
    replaces Choose another spectrum…, which opened an empty dialog.
- **Check.** `gui` tests in `test_dashboard_qt_loadfit.py`, and the existing tests of
  Open a fit and Load snips… in `test_dashboard_qt_regions.py`:
  - the question counts the datasets of a fit of three;
  - with both of a fit's two spectra added, each spectrum's FWHM is its dataset's;
    the assignment dialog shows each snip under its spectrum, and after Confirm the
    fit loads with no FWHM note and the reference of 11.1;
  - with one of the two added, the other's snip is on the right: ticked, it loads
    unassigned; unticked, it is left out with its lines; dragged onto the first
    spectrum, it is cut again from it;
  - with One unassigned dataset chosen and two unassigned snips that overlap, the
    warning names the pair: Each snip its own dataset (advised) gives two
    datasets, and One unassigned dataset gives one, the Regions tab showing both
    snips either way; with no overlap, no warning; with Each snip its own dataset
    chosen, no warning;
  - Load a fit into this project, and Load snips…, open the dialog in a project of
    two spectra;
  - Back keeps what was entered;
  - with one spectrum that holds every snip, the dialog opens with every snip under
    it;
  - the existing tests of Open a fit, Load a fit and Load snips… answer the dialog
    with Confirm, and otherwise pass unchanged (Cancel, Open without a spectrum); the
    test that one failing file stops Load snips… becomes the test that it goes to
    the right-hand list.

**11.6 — Every button keeps its "…" (item 2, bug; `qt/window.py`).**
- **The bug.** Nine buttons made from actions drop the "…" that their action's text
  ends with (see the Design).
- **The fix.** One rule where the window makes its actions: an action whose text ends
  with "…" keeps it in its icon text, so that every button and toolbar item made
  from it shows it. The two lines that set Export… and Random restarts… by hand go;
  "?  Shortcuts" keeps its own text. The menus are unchanged.
- **Check.** Made into a test first, in `test_dashboard_qt_window.py`: on a project
  with every tab shown, every tool button whose action's text ends with "…" shows
  that text (before the fix, nine do not). The `gui` batch shows that no other text
  changed.

**11.7 — One choice of line IDs for the Data and Regions tabs (item 1; `qt/data.py`,
`qt/regions.py`; D59, D114).**
- **What.**
  - **One setting.** The line IDs' choice (This transition, This ion, The model's ions
    and strong lines, Every line) becomes one view setting of the project,
    `line_ids` in `ui/view.json`, read by both tabs. Choosing on either tab changes
    both.
  - **The Regions tab's menu** is as now.
  - **The Data tab's menu.** The "All lines" switch becomes the same menu, after
    "Transitions of z", which still shows or hides the line IDs.
  - **What the Data tab labels.** This transition and This ion label the transition
    and the ion that the Regions tab shows, of its system; with none shown yet, the
    model's choice is drawn. The model's ions and strong lines, and Every line, label
    the primary system, as the switch did.
  - **Projects saved before.** A project saved with the Regions tab's choice reads it
    as the shared one. One saved with none starts at the model's ions and strong
    lines (D59), as both tabs start now (the switch started unticked).
- **Check.** `gui` tests in `test_dashboard_qt_data.py` and
  `test_dashboard_qt_regions.py`:
  - Every line chosen on the Data tab is the Regions tab's choice, and the reverse;
  - This transition on the Data tab labels the Regions tab's transition alone, and
    counts the rest;
  - the choice is kept in `ui/view.json`, and read back after Save and Open;
  - a project saved before keeps the Regions tab's choice;
  - the Data tab's existing tests of line IDs are updated for the menu.

**11.8 — Labels from the component's scheme (item 3, Q11.4's and Q11.5's responses;
`edit.py`, `components.py`, `project.py`, `qt/components.py`; Q2.3, D93, D119).**
- **What.** Every label the dashboard makes or suggests for a component is `L` + `S`
  + `#`.
  - **The parts.** `L` is `z`, `b`, `t`, or `n` for a column density. `S` is the
    system's letter, or `ly` for a line of the Lyα forest. `#` is the component's
    number in its labels.
  - **The number** (Q11.5's response). A component's labels share one number: the
    number its labels of the scheme already carry (`za3` gives 3). A component with
    none takes the first number that no label of the scheme uses in its system, so
    the numbers follow the order in which components are given labels. A label keeps
    its number until the user renames it, or until 11.9 renumbers the system.
  - **A column density** (Q11.5 (a)) adds its ion: its element in lower case and its
    stage as a number (`na2si2` for Si II, `na2s3` for S III, `na2d1` for D I);
    another rarer isotope adds its mass number (`na2c2_13`). A line of the forest
    has one ion, so it is `nly3`.
  - **Where.** The suggestion asked for before fixing, tying or limiting an
    unlabelled value (`suggest_label`, from the component that holds the value,
    through the card's question); the fixes of D20 and D22; a temperature mode's own
    labels (D93); and the redshift label of a component of no system
    (`components._tied_first`, `x` now).
  - **As now:** upper case for a fixed value (`NA2SI2`), and a label already in use is
    never suggested.
- **Check.** New `unit` tests in `test_dashboard_edit.py` and
  `test_dashboard_components.py`:
  - an unlabelled z, b, T and N of a component with no labels, in a system whose
    labels use 1 and 2, are offered `za3`, `ba3`, `ta3` and `na3si2`;
  - a component whose z label is `za3` is offered `ba3`;
  - forest line ly3, with no labels, is offered `bly…` and `nly…` with the forest's
    first free number;
  - Si II and S III of one component are offered `na3si2` and `na3s3`, D I `na3d1`;
  - a temperature mode's new label shares the component's number;
  - `test_a_redshift_gets_a_label_from_the_scheme` expects the component's label in
    place of `zx1`, as item 3 asks;
  - a `gui` test shows the suggestion in the card's question.

**11.9 — Renumber a system's labels by redshift (Q11.5's response; `components.py`,
`qt/components.py`; Q2.3, Q7.6; Q11.7).**
- **What.**
  - **Renumber labels…** on the Components tab, under the navigator beside Add ion…
    and Add system…, acts on the system selected there, or on the Lyα forest.
  - **What it renames.** Each component's labels of the scheme take the component's
    number by redshift, which its card shows: for component a2, `za3` becomes `za2`,
    with its `ba3`, `ta3` and `na3si2`. It is one change, so labels that swap
    numbers need no temporary name. Every use follows, as a rename does
    (`edit.rename_label`): the lines, those commented out included (Q7.6), the `fix`
    and `lim` commands, and the links.
  - **What it leaves**, and the preview names: labels the user named, labels shared
    with another component, and labels with another system's letter.
  - **The warning** comes first, with the lines that change: it changes the labels of
    every component of the system, and labels written outside the dashboard (notes,
    scripts) will no longer match. Undo takes it back, as any step (Q11.7 (a)).
  - **Greyed** when the system's labels are in order already, its tooltip saying so.
    The action shows no hidden value: labels are not hidden.
- **Check.** New `unit` tests in `test_dashboard_components.py`:
  - three components whose labels read `za1`, `za3`, `za2` in order of redshift
    become `za1`, `za2`, `za3`, with their b, T and N labels;
  - a `lim` command, a link and a line commented out follow;
  - a label the user named and a label shared with another component stay, and are
    named in the plan;
  - ALIS reads the model, and one undo gives back the text;
  - a system in order already has nothing to renumber;
  - a `gui` test: the action, its warning and preview, and its greyed state.

**11.10 — The review.**
- **What.** A review page built by `doc/dashboard/stage11/`, with the earlier pages'
  stylesheet. Screenshots at 1440×900 of:
  - the Regions tab with two snips of one transition in one dataset (11.2);
  - the Open a fit question for a fit of several datasets, the New project dialog,
    the dialog that assigns the snips (with a snip of no spectrum, and the choice of
    how unassigned snips are loaded), and the overlap warning;
  - the buttons with their "…";
  - the Data tab's menu of line IDs;
  - a card's question for a label, and Renumber labels…'s warning.

  It is published as a private page, and RJC's comments are applied.
- **Check.** The page builds from the screenshots of the `gui` tests' scenes. Every
  task of 11.1–11.9 that a user sees has its picture, or its test named where a still
  image cannot show it. RJC's comments are applied and logged.

**11.11 — Close the stage.**
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill (its "Stage 11 additions") up to
  date. Record the decisions in `ALIS_v2_dashboard_prompts.md` (D120 onwards, from
  Q11.1–Q11.11 and the review), saying which earlier decisions they refine: D12,
  D14, D36, D46, D55, D59, D78, D93, D114, D115 and D119, Q2.3, and Q7.4's rule. Update
  `dashboard_stage0.md`: Stage 11 done, and the "After v1" list. Update
  `deferred_work.md`: §1.4 (the GPU) stays, and anything this stage leaves undone is
  added.
- **Found and not done.** What was found during the stage and not built goes to
  Stage 12's "Carried over from Stage 11", with a query, as Stage 10's close did.
  (The imported fit's hidden snip, found at Prompt 3, is fixed in this stage: 11.2.)
- **What follows (Q11.2 (a)).** Write `claude_prompts/dashboard_stage12.md` for the
  feedback that arrives after this list closed, as Stage 10's close wrote this
  document, unless RJC names another stage to come first (the GPU's series,
  `deferred_work.md` §1.4; the documentation, `documentation_stage0.md`; or an item
  marked "Later" in `dashboard_stage0.md`).
- **Check.**
  - Every point of the Feedback list and every item carried over has its task logged
    as done with its check passing, or its query answered: point 1 in
    `deferred_work.md` §1.4 (Prompt 2), point 2 in 11.1 and 11.3–11.5, the bug
    found at Prompt 3 in 11.2, items 1–3 in 11.6–11.9.
  - Q11.1–Q11.11, and any query raised while the tasks were built, are answered in
    the Queries section.
  - The review of 11.10 is applied.
  - The `unit`, `gui` and `fast` batches pass, also under CI's pytest. Their counts go
    in the log, against Stage 10's close: 2211 unit tests passed with 120 skipped,
    279 `gui` tests, and 149 `fast` tests, the regression harness and its
    `test_bundle` mode included (all under pytest 9.2.0).
  - CI passes on both platforms, the `gui` job within its 90 minutes.
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44), and pre-commit passes on every
    changed file.
  - ALIS outside `alis/dashboard/` is unchanged (`git diff` against Stage 10's close,
    00d5a6f). This stage's tests are synthetic and need none of RJC's files.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs and the Figure
  tab (its "Stage 9 additions"), and the lessons of Stage 10 ("Stage 10 additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D119.
- `claude_prompts/dashboard_stage10.md` and its log: how users' feedback was handled
  last time, from the list to the close (and `dashboard_stage8.md` before it).
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 10's builder is `doc/dashboard/stage10/`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude: Q11.1 and Q11.2 with this document (2026-11-08), Q11.3 and Q11.4 at
Prompt 1 (2026-11-10), Q11.5 and Q11.6 at Prompt 2, Q11.7–Q11.10 at Prompt 3, Q11.11
at Prompt 4 (2026-11-12). Each gives Claude's lean. A question that comes up while a
task is built pauses the work; it is asked here, and its answer is logged here (RJC,
Prompt 2).*

**Q11.1 — The items carried over from Stage 10.** For each, two ways:
- **(a) It joins this stage**, after the feedback's tasks.
- **(b) It waits** for a user to ask for it, in `deferred_work.md` §7.

My lean: (b) for item 1, since no user has asked for it and each tab's labels suit
what it shows; (a) for items 2 and 3, small fixes that make the window consistent.

**Response:** Let's implement options 1 and 2. I don't agree with option 3, because a line for Si I and S II might get the same prefix sii, and that would be confusing and introduce unseen bugs that are not flagged by the tests. Instead, let's keep the current naming scheme. `nx#` for column density, `bx#` for turbulent broadening, `tx#` for temperature, and `zx#` for redshift, where in all cases, the `#` is the number/ID of the component in the system. This is consistent (I think!) with the current naming scheme and avoids confusion.

*Applied (Prompt 1): items 1 and 2 join the stage, after the feedback's tasks (item 1
against my lean). Item 3 waits on Q11.4: the scheme the response describes differs
from the code in two places.*

**Q11.2 — When the stage closes, and what follows it.** As Q10.3: two ways:
- **(a) The same as Stage 10.** The list closes when you judge it long enough.
  Feedback that arrives later goes to Stage 12's document, written at this stage's
  close, unless you name another stage to come first: one of the items marked
  "Later" in `dashboard_stage0.md`, the documentation (`documentation_stage0.md`),
  or the deferred work.
- **(b) An open stage.** Stage 11 builds points in batches as they arrive, with a
  review for each batch.

My lean: (a), as Stages 8 and 10 had.

**Response:** I agree with option (a).

**Q11.3 — Where Feedback point 1 is explored: this stage, or a series of its own
(Prompt 1).** Point 1 asks that the large arrays stay on the GPU for the whole fit, so
that each evaluation sends only the parameters and returns only χ². Refactor Stage 4
built the GPU path and asked much the same at its Prompt 11, so I read the point
against its measurements on DH_orders (`refactor_code_stage4.md`, Tasks 4.3–4.9, and
the "Pre-4.7" and "Pre-4.9" analyses of its log). The GPU modules have not changed
since, and the rest of the evaluation by about 80 lines, so the figures should still
hold.

1. *Part of it is in place.* The sub-pixels are worked out once unless
   `run renew_subpix True` (the default is False, and no fit in the repository sets
   it). Each snip's sub-pixel grid is kept on the device
   (`gpu_dispatch.Dispatcher.wave_device`): 28.8 MB for DH_orders, uploaded once per
   iteration and reused by every evaluation within it.
2. *The transfers are small; the round trips and the host's work are not.* An
   evaluation sends each group's parameters (kB) and brings back one sub-pixel array
   per snip (tens of kB). What costs is that this happens 351 times per evaluation,
   each ending in a synchronising `copy_to_host` (the CUDA driver is ~13% of a
   derivative column, ~1.27 ms per group), and that the host does the rest in between:
   the convolution (`vfwhm`, 20% of a GPU Jacobian), the continuum (`legendre`, 9.3%)
   and the loop over snips. A GPU-4 Jacobian of DH_orders takes 61.9 s, against about
   32–37 s on CPU-12, and `run backend auto` picks the CPU on your machine.
3. *So it is worth exploring, though not for the transfers.* An evaluation kept whole
   on the device (every function of the model, the convolution, the residuals) removes
   the round trips and the host's work together. The arithmetic itself is small: the
   Voigt kernel takes 15.6 ms for 10 million sub-pixels on one RTX 2080 Ti, and a base
   evaluation of DH_orders is 119 million pixel-components, about 0.2 s of kernel time.
   Whether the rest comes down to that scale has to be measured. Stage 4 met three
   speed-ups whose premise did not survive measurement (the ~50× of Q4.8, and
   "Pre-4.7" (A) and (B)), so the work should begin with a prototype and a figure it
   must beat.
4. *χ² alone cannot be all that comes back.* The minimiser is MPFIT's
   Levenberg–Marquardt: each evaluation needs every pixel's residual, and each
   iteration the Jacobian, whose QR factorisation gives the step and the covariance.
   For DH_orders that is 75,164 residuals (0.6 MB) and a 75,164 × 413 Jacobian (248 MB
   dense): tens of ms over PCIe, against a 32 s Jacobian. Returning χ² alone needs a
   minimiser without a Jacobian (slower to converge, and no covariance); returning
   JᵀJ and Jᵀr changes the numerics from QR to the normal equations.
5. *It is larger than a stage, and outside the dashboard.* It changes ALIS's core
   (`minimise.py`, `model_eval.py`, `gpu_dispatch.py`, the functions); every function a
   fit uses needs a device version; the convolution needs an FFT, which numba.cuda
   does not have (Q4.1 chose numba; CuPy, or a direct convolution at 21× the
   arithmetic, is a decision of its own); and its checks are the regression harness,
   the CPU path's bitwise gate and the `gpu` batch on your machine
   (`pytest --run-gpu -m gpu`), not the `gui` tests and a review page.

Two ways:
- **(a) In this stage**, as tasks after the carried items, under this document's
  rules.
- **(b) A series of its own.** `claude_prompts/gpu_stage0.md` is its design document:
  the measurements, the prototype and the figure it must beat, and the queries of
  points 4 and 5, with the hardware to measure on (the RTX 2080 Ti runs float64 at
  1/32 of float32, Q4.6; a data-centre card at 1/2). Stages 1, 2, … build what it
  decides, logged in `claude_prompts/logs/gpu_stage<n>_log.md`, on branches you make,
  such as `alis_v2_gpu_stage0` from Stage 10's close (00d5a6f). The CPU path stays
  bitwise-identical at every step, and today's GPU backend stays until the new one
  beats it. This stage keeps the dashboard's feedback (Q11.2 (a)); point 1 stays in
  its list as your record, and the Design says where it went.

My lean: (b). The work may end in "no", and a stage cannot close well on that: waiting
would hold the dashboard's fixes back, and closing first would leave it half done. It
has its own tests, skills (`port-to-gpu`, `gpu-benchmark`, `profile-fit`) and history,
none of which the dashboard's stages use. It is refactor Stage 4's Task 4.9 (still
"awaiting RJC") grown to the whole evaluation, so that task would close as taken over
by the series. The two meet only at the Fit tab's Run box, whose Backend choices are
`alis.gpu.BACKENDS` (dashboard Stage 6): a new backend would appear there on its own,
and the series would check that it does.

**Response:** Agreed, let's explore this in a separate series of design documents (starting with a new design document labelled stage 0). This will allow us to explore the GPU functionality without breaking the current code and to measure the performance improvements effectively. A task of this stage is to write that design document, and we will proceed with the development of the dashboard's feedback in this stage.

*Applied (Prompt 2, 2026-11-12): Prompt 2 asks that the GPU wait as deferred work. It
is in `deferred_work.md` (§1.4, and priority 13), with this query's findings and what
`gpu_stage0.md` will hold, so it is not one of Tasks 11.1–11.11. `gpu_stage0.md` is
written when you start the series. If you want it written before this stage closes,
it can be the stage's last task.*

**Q11.4 — The labels suggested for unlabelled values (carried item 3, the response to
Q11.1).** The response keeps the current scheme and declines item 3, since a line of
Si I and one of S II might get the same prefix `sii`. Two parts of it differ from the
code, so I ask before closing the item:
- *Item 3 does not use the ion.* It offers the component's own system letter: `ba3`
  for component 3 of system a, `bly2` for the forest's second line (D119), like the
  labels the dashboard gives a new component (Q2.3). The Si I / S II clash is Q2.3's,
  for column densities named by their ion (`nsiii1`), and it is why a column density
  has no label until the user names one.
- *The current scheme* (`edit.suggest_label`) offers `zx#`, `bx#` and `tx#`, where `#`
  is the first number none of the three uses yet anywhere in the model, not the
  component's number in its system; for a column density it offers nothing, and the
  user types a label (Q2.3). With a literal `x`, the running number is what keeps the
  suggestions apart: by the component's number, component 2 of system a and
  component 2 of system b would both be offered `bx2`.

Three ways:
- **(a) As now.** Item 3 closes with no task.
- **(b) As now, and `nx#` for a column density**, with the same running number: a
  suggestion the user can type over, which cannot clash as `nsiii1` could. A small
  task, which revises Q2.3.
- **(c) Item 3 as written**, by the system's letter and the component's number;
  column densities as now.

My lean: (b), the scheme your response lists, with the running number; `nx#` removes
the reason Q2.3 gave for leaving column densities unlabelled. (a) if you meant the
scheme exactly as it is.

**Response:** I like the idea of using `LS#`, where `L` represents the variable type (`n` for column density, `b` for turbulent broadening, `t` for temperature, and `z` for redshift), and `S` represents the system (`a` for primary, `b` for secondary, `c` for teritary, etc. and `ly` for interloper). Finally, `#` represents the component number in that system. I think this is consistent with the current naming scheme and avoids confusion. Please check that this is consistent with the current code and tests, and if not, let's implement it.

*Checked (Prompt 2): see the Design (item 3). The labels the dashboard makes follow
`LS#` for z, b and T. The suggestion before fixing, tying or limiting uses `x`, and a
column density gets none. Task 11.8 implements `LS#` for every label made or
suggested for a component. Two parts need your decision: Q11.5.*

**Q11.5 — A column density's label, and the component's number (Q11.4's response,
Task 11.8).** Two parts of `LS#` need a decision before 11.8.
- *A column density.* Each ion of a component has its own column density, so `na2`
  can belong to only one of them. If Si II's in component a2 is `na2`, and C II's is
  offered `na2` too, ALIS ties the two. Two ways:
  - **(a) `na2` and the ion:** its element in lower case and its stage as a number:
    `na2si2` for Si II, `na2s3` for S III, `na2d1` for D I. Another rarer isotope adds
    its mass number (`na2c2_13` for ¹³C II). Si II and S III no longer meet, as they
    did in Q2.3's `nsiii1`. A line of the forest has one ion, so it is `nly3`.
  - **(b) No suggestion**, as now (Q2.3): the user names it.
- *The number.* Labels are never renamed (Q2.3), but a component's number follows its
  redshift, and changes as components are added or removed. A component added
  between a1 and a2 becomes a2, and the old a2 becomes a3 while keeping its `za2`.
  So "#" can be the component's number only when its first label is made. Two ways:
  - **(a) A component's labels share one number:** the number its labels already
    carry, else its own number when free, else the first free one (as 11.8 was
    written).
  - **(b) The dashboard renames** a system's labels of the scheme after each component
    is added or removed, so that "#" is always the component's number. Every line,
    `fix` and `lim` command and link that uses them changes with them, and the labels
    in the user's notes no longer match.

My lean: (a) for both. Each keeps the label unique, and (b) for the number reverses
Q2.3 for a change that is cosmetic.

**Response:** I agree with (a) for the first part. For the second part, I like the idea of the number representing the component in order of redshift, but this is more likely to complicate the code. Is it possible for a component to always keep its number based on the order it is added, but we can introduce a button that allows the user to relabel the component numbers based on redshift order? There should be a warning that this cannot be undone, and that it will change the labels of all components in the system.

*Applied (Prompt 3): (a) for a column density (Task 11.8). A label keeps the number
it was made with, so a component without labels takes the first free number (11.8),
and Renumber labels… brings a system's labels into its components' order by redshift
(11.9). Two parts of the button need your decision: Q11.7.*

**Q11.6 — Open a fit when one of its spectra is missing (point 2, Tasks
11.3–11.5).** A user may have only some of a fit's spectra, say because an old
reduction was lost. Two ways:
- **(a) Every dataset needs its spectrum**, as Q7.4's rule needs every snip to match.
  Create waits until each spectrum is added. Without one of them, the fit opens with
  Open without a spectrum: every dataset without its spectrum.
- **(b) The missing ones open without.** Create is allowed: the datasets with a
  spectrum are checked and keep it, and the others come in as an imported fit's do,
  with regions limited to each snip and no new snips (D12). Such a dataset could not
  be given its spectrum later: Relink… takes only the same file, moved (its
  checksum), so that would be new work (S23, `deferred_work.md` priority 8).

My lean: (a) for this stage. It keeps one rule for a fit's snips, and (b) is worth
building only together with a way to give a dataset its spectrum later.

**Response:** I propose an alternative option. The reason for this alternative is because a user may have two separate datasets that have the exact same wavelength values, and there may be some confusion about which snip belongs to which dataset. So, I propose that the user has the chance to load all available datasets, and the dashboard will assign a best guess of which snip belongs to which dataset. This could be a pop up dialog that has the following layout: (1) a list box on the top that allows users to select the dataset; (2) a container list on the left that lists all of the snips that were automatically identified as belonging to this dataset; (3) a container list on the right that lists all snips that could not be automatically identified as belonging to *any* dataset. The user can then drag and drop snips from the right container to the left container to assign them to the selected dataset. If snips are misidentified with this dataset, the user can drag and drop a snip to the right container, select a different dataset from the list box at the top, and then drag the snip from the right container to the left container. Once all snips have been assigned, the user can click a "Confirm" button to finalize the assignments. This way, the user has control over which snips belong to which dataset, and it avoids any confusion that may arise from automatic assignments. It also allows the user to specifically assign snips to the "unidentified snips" container (on the right) if they do not belong to any of the loaded datasets. This approach gives the user more control and flexibility in managing their datasets and snips.

*Applied (Prompt 3): Tasks 11.3–11.5 now build this dialog, with the dashboard's best
guess from the snips each spectrum holds (11.3). A snip left on the right comes in
without a source, in a dataset grouped as an imported fit's (Q2.4), with D12's limits.
Three parts of it need your decision: Q11.8.*

**Q11.7 — Renumber labels… (Q11.5's response, Task 11.9).** Two points:
- *Undo.* Every change to the project is one step of the history, which Undo takes
  back. The one exception is blinding, which Undo does not pass (D49). Two ways:
  - **(a) Undo takes it back**, as any step. The warning lists every line that
    changes, and says that labels written outside the dashboard (notes, scripts)
    will no longer match.
  - **(b) Undo does not pass it**, as blinding, and the warning says that it cannot
    be undone.
- *What a card shows.* A component's name on its card (`a1`, `a2`, …), the panels'
  ticks and the figures' numbers follow redshift, while a label keeps its number
  until the system is renumbered. So card a2's labels may read `za3` until then. Two
  ways:
  - **(a) As now:** the cards follow redshift, and Renumber labels… brings the labels
    into line with them.
  - **(b) The cards follow the labels' numbers**, the order in which components were
    given labels. A card's name, its panels' ticks, the Figure tab's numbers and
    Export's choice of component would then all change to match.

My lean: (a) for both. Nothing is lost by letting Undo take a renumbering back, and
a mistaken one can then be put right; the risk the warning names (labels used outside
the dashboard) is the same either way. The cards' order by redshift is what every
tab shows now, and the button is there to bring the labels into it.

**Response:** I agree with (a) for both points.

*Applied (Prompt 3, run again): Task 11.9 was written with these leans, and stands.*

**Q11.8 — The dialog that assigns the snips (Q11.6's response, Tasks 11.3–11.5).**
Three points:
- *When it opens.* Two ways:
  - **(a) When there is a choice:** the project has two or more spectra, or a snip is
    held by no spectrum or by more than one. With one spectrum that holds every
    snip, the preview comes at once, as now.
  - **(b) Every time** a fit is loaded with spectra.
- *A snip that the chosen dataset's spectrum does not hold*, because its pixels are
  not that spectrum's: it was cut from another reduction, or from a copy with
  different values. Two ways:
  - **(a) The drop is refused**, with the reason: a snip joins a dataset only if it
    was cut from its spectrum. It stays on the right, and comes in without a source.
  - **(b) The drop is allowed.** The snip keeps its own pixels in that dataset,
    marked as not cut from its spectrum, and is treated as a snip without one: its
    edges move only inwards, and it is never cut again or merged from the spectrum,
    since either would replace its pixels with the spectrum's without saying so.
- *Where it is used.* Two ways:
  - **(a) In every way a fit is loaded with spectra:** Open a fit as a new project
    with its spectra, Open a fit into a project of its spectra, and Load a fit into
    this project (the `.mod` panel's menu).
  - **(b) Only when opening a fit as a new project.**

My lean: (a) for all three. For the second point, (b) puts two reductions' pixels in
one dataset, sharing one FWHM and shift, and needs a guard on every action that cuts
from the source. Two spectra on the same wavelengths are still told apart, since a
snip must have a spectrum's flux and error too, and a snip that both hold is marked
in the dialog for the user to choose.

**Response:** I propose (b) for the first part, and I agree with (a) for the second and third parts. The reason I think (b) is better for the first part is that there may only be one dataset loaded, but there could be snips that don't fit with the dataset. This gi nves the user a chance to assign the snips to the correct dataset, and it avoids confusion about which snip belongs to which dataset. Furthermore, there may be snips thatcome from two datasets, but only one dataset is loaded. One user reported there being an issue with the dashboard only looking in the first dataset for snips, and this introduced some issues. If a user assigns a snip to the "unidentified snips" container, they can still be loaded and used in their analysis.

*Applied (Prompt 3, run again): Task 11.5 opens the dialog every time a fit is loaded
with spectra, even when one spectrum holds every snip (b). A drop on a spectrum that
does not hold the snip is refused, and the dialog is used in every way a fit is
loaded (a), as 11.3–11.5 were written. The user's report is Q11.9. (The refused drop
gave way to Q11.9's response: such a snip is cut again from the spectrum it is
dropped on.)*

**Q11.9 — "Only looking in the first dataset for snips" (Q11.8's response).** I found
no code that looks for a snip's spectrum in the first dataset alone:
`loadfit.match_row`, used by Open a fit, Load a fit and Load snips…, tries every row.
What the user may have met is the refusal. A snip that no spectrum holds is given the
reason of the row nearest to matching, the first when they tie. With RJC's
J0035m0918 `.mod` and all three of its spectra, the O I 1039 snip (3470 Å) is refused
as "outside the range of J0035m0918_ESPRESSO". J0035m0918.dat covers 3128–5966 Å, so
the snip is within its range but not on its pixels, and the message names neither
it nor UVES2010: it reads as if only the first spectrum was tried. Two ways:
- **(a) This is the report.** Tasks 11.3 and 11.4 give a reason for every spectrum
  tried: in the refusals, in Load snips…, and on each snip of the dialog's right-hand
  list.
- **(b) The report is something else.** Please give the steps, or the user's files
  (referenced from the Feedback list, D99), and it becomes a task of its own, a bug
  made into a test first.

My lean: (a), unless you know the report to be something else; 11.3 and 11.4 are
written with it.

**Response:** We should not refuse snips. There may be a case where all snips fit a dataset except 1. We should offer the user the chance to either: (1) assign that snip to a dataset (and recut the data if the data do not overlap); (2) ignore that snip, but load all correctly assigned snips; (3) load all snips, but mark any unassigned snips as "unassigned" and do not cut from any dataset (this means each unassigned snip becomes its own dataset). This way, the user has control over how to handle snips that do not fit with the loaded datasets.

*Applied (Prompt 3, run a third time): Tasks 11.3–11.5 now refuse no snip. A snip is
held by its spectrum, cut again from the spectrum it is given (1), left out with the
lines only it uses (2), or loaded unassigned, cut from no spectrum (3). The reasons
for every spectrum stay, so that the user sees why a snip is on the right. Four
details of the three choices: Q11.10.*

**Q11.10 — The three choices for a snip that no spectrum holds (Q11.9's response,
Tasks 11.3–11.5).** Four points:
- *(i) The datasets of unassigned snips.* The response says each unassigned snip
  becomes its own dataset. Two ways:
  - **(a) One unassigned dataset for each dataset of the `.mod`**, as an imported
    fit's snips are grouped (Q2.4: by their resolution label). Snips that share a
    FWHM and a shift in the `.mod` stay together on the Data tab.
  - **(b) One dataset for each unassigned snip**, as the response says. A `.mod` cut
    from other reductions (RJC's J0035m0918 `.mod`, 34 snips) gives 34 rows on the
    Data tab, tied where they share a label.
- *(ii) Loading or leaving out.* Two ways:
  - **(a) For each snip:** each snip on the right has a tick, Load, ticked by
    default; unticked, the snip is left out.
  - **(b) Once for all:** at Confirm, one choice for every snip on the right.
- *(iii) The range a snip is cut again over.* Two ways:
  - **(a) The snip's own wavelength range**, its fit range carried over by
    wavelength, its model values as the `.mod` has them.
  - **(b) SNIP's window** (±300 km/s about its transition), as a new snip, with its
    fit range carried over.
- *(iv) Load snips…* (snips without a model, D78) refuses every file when one does
  not match. Two ways:
  - **(a) The same dialog**, so that a snip there is not refused either.
  - **(b) As now** (Q7.4), with a reason for every spectrum.

My lean: (a) for all four. (i): the `.mod` already says which snips share a FWHM and
a shift, and one dataset per snip would scatter them. (ii): it is the one choice the
response describes for a single snip. (iii): the snip's regions, continuum and model
were set over its own range. (iv): it is the same question asked by another way in.

**Response:** For part (i), I agree with (a) unless there is wavelength overlap between two unassigned snips. If the dashboard can handle two snips covering an overlapping wavelength range within the same unassigned dataset, then this is the preferred option. For part (ii), I agree with (a). For part (iii), I agree with (a). Also note that if a user assigns a snip to a dataset but the wavelengths, fluxes, or errors do not overlap then we should allow warn the user that the snip will be recut from the dataset so that it matches the dataset. If they do not wish for this to happen, they should leave it as an unassigned snip. For part (iv), I agree with (a).

*Applied (Prompt 3, run a fourth time): (a) for all four. For (i), the dashboard holds
two snips that overlap at different transitions in one dataset, but shows one snip
of a transition per dataset on the Regions tab, so a second snip of the same
transition starts another unassigned dataset (Task 11.3; see the Design). The warning
before a snip is cut again is in Task 11.4. (Prompt 4 replaced the rule for a second
snip of a transition with the user's choice and a warning, and fixed the Regions tab
so that it shows every snip: Tasks 11.2–11.5.)*

**Q11.11 — How unassigned snips are loaded (Prompt 4, Tasks 11.3–11.5).** Prompt 4
asks the user to choose "a single 'unassigned' dataset" or each snip its own dataset,
with a warning when snips overlap. Two points:
- *(i) "A single unassigned dataset"*, when the unassigned snips come from several of
  the fit's datasets (snips of va and of vb, say). Two ways:
  - **(a) One unassigned dataset for each of the fit's datasets**, as Q11.10 (i)
    agreed: the snips that share a FWHM and a shift in the `.mod` stay together, and
    the overlap check runs within each.
  - **(b) One unassigned dataset for all of them.** Snips with different resolution
    labels then share a dataset, and each keeps its own resolution (D89), so the
    Data tab's FWHM for that dataset means nothing.
- *(ii) Where the choice starts.* Two ways:
  - **(a) One unassigned dataset**, with the warning when its snips overlap.
  - **(b) Each snip its own dataset**, the advised choice where snips overlap.

My lean: (a) for both. (i): the fit's own datasets are what it says about its
snips. (ii): most fits' snips do not overlap, so most users keep fewer datasets and
see no warning; where snips overlap, the warning advises the other choice.

**Response:** I agree with (a) for both points, but I'll test the drafted design and propose changes later, if needed.

*Applied (Prompt 5): Tasks 11.3–11.5 were written with these leans, and stand. Changes
you propose after testing the design are asked here as they come, and the work
pauses for them (Prompt 2).*


## Prompts

1. Please read the one suggested improvement in the `Feedback` section, and the work of Stages 0–10 (their documents and logs). Please review this suggestion, and my responses to your queries. If anything is unclear, ask new questions in the Queries section, and I will respond; The first decision to decide is if we should solely explore the GPU functionality as part of this design document (stage 11) or if we should separate this into a separate series of design documents (starting with a new design document labelled stage 0). Please review this possibility. Once we make a decision about including this during the current stage or a new series of stages, we will proceed with the development.
2. Please read the Feedback section, and the work of Stages 0–10 (their documents and logs). Please review the items carried over from Stage 10, and my responses to your queries. Let's avoid the GPU tests and speedups at this point, and save it for deferred work. If anything is unclear, ask new questions in the Queries section, and I will respond. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section). For now, please prepare the first set of tasks, including the items carried over from Stage 10, and the new item (Item 2) listed in the Feedback section.
3. I have responded to your next set of queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (11.1, 11.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
4. Let's fix the existing bug you mentioned in this stage. Regarding the condition on unassigned datasets, we should ask the user how they wish to treat the unassigned snips: A single "unassigned" dataset, or each snip becomes its own dataset. If the first option is selected, then we should check if there's any wavelength overlap. If there is wavelength overlap, we should warn the user that there is wavelength overlap, and we advise they should treat the unassigned snips as independent datasets. The user can then choose if they wish to treat them as independent unassigned datasets (as advised) or to include all snips into a single "unassigned" dataset. Please make the appropriate changes to the stage 11 design document to account for these two points, and if you have any further queries, please ask them in the queries section.
5. I have responded to your next set of queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (11.1, 11.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
6. Please execute the tasks in order. There's no need to pause between tasks, unless you have queries that require my thoughts and responses. Please check your context limit between tasks. If you reach 75% context, please pause and ask for my guidance on the next steps. If you have queries during development, please pause the development and log the queries and responses in the Queries section.
7. I have added some context in `context/feedback01/MWC137/`. There is a .mod file that was imported from the `alis2.mod` file and snips in the `data/` directory. I have saved the `.model` file. As you can see, the import has included several secondary absorption line systems. These are not real. They are commented out lines in the `.mod` file, and the redshift values actually correspond to the turbulent Doppler parameters of several different commented out absorption lines. Could you please investigate this bug before we close off this stage of development? Please let me know if you have any questions or need further clarification.