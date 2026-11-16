# Prompt file for ALIS software dashboard creation -- STAGE 12

> **Feedback from users, after Stage 11's list closed.** Stage 11 took a closed list
> of users' feedback (a fit of several spectra, and the GPU, which it deferred) and
> the items Stage 10 carried over (Q11.2 (a)). Stage 12 takes the feedback that
> arrives after that list closed. RJC collects each point in the Feedback section
> below. Claude turns the list into a design, queries and tasks, each with its
> check, and builds them in order, with a review before the stage closes.
>
> The rules of Stages 3–11 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees. A question that comes up while a task is built pauses the
> work: it is asked in the Queries section, and its answer is logged there (Stage 11,
> Prompt 2).
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D128 at the close of Stage 11).
> "Q0.*n*" to "Q11.*n*" refer to `dashboard_stage0.md` to `dashboard_stage11.md`.
> What follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-13 (Stage 11's Task 11.11). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is. The items carried over are what Stage 11 found and did not do.*

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

1. **(bug)** In some cases, the wavelengths of two datasets are identical, and the dashboard things that the wavelength regions overlap, but they are actually different datasets. For example, the context file `context/feedback01/MWC137/mwc137.model` provides an example where two datasets (specid=6 and specid=7) have overlapping data, but they are actually different datasets. The dashboard should be able to distinguish between these two datasets and not treat them as overlapping. This can be reproduced by loading the context file and checking the datasets with specid=6 and specid=7.

2. **(bug)** In the Components tab, it is not possible to click the excluded component a2 to make it included. Instead, it informs the user that they should add it to an ion first. I believe there should be the functionality to simply click the ion in the Excluded list to add this component to an ion. Alternatively, there needs to be a more helpful message to the user about how they can add this component to an ion. This can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model` and going to the Components tab, then trying to click on an ion in the excluded component a2.

3. **(change)** On the Components tab, if I scroll down to the bottom of the components list, and include an ion that is currently excluded, the page refreshes and the scroll changes so that the components list navigates to the top of the components list. This means that the user needs to scroll down to the bottom of the components list again to add another component. This is not ideal, and can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model`, going to the components tab, scrolling down to the components near the bottom of the list, including an ion into a component, and watch the menu scroll to the top of the components list. Overall, it would be good if none of the windows/scrollbars/menus/zooms reset to the default values. If the user has scrolled down, or zoomed in, or changed the window size, it would be good if those values were remembered and not reset to the default values. We must avoid changing things to the default value unless it is absolutely necessary. Furthermore, there is always an option for the user to go to the "Home" view by clicking on the "Home" button that appears on some plots. Otherwise, the user should have complete control over the zoom and scroll values of all windows.

4. **(new)** It would be good if there were a way to zoom into all spectra in both directions. Currently, a scroll only allows the wavelength to zoom in about the mouse cursor. Maybe we should make it so that the user can hold down a key and while the key is down (e.g. 'y') then scrolling would zoom in the y direction about the mouse cursor. Similarly, it would be good if there was a button that allows the user to reset the default x separately from the default y. This could be something like the Home button that appears on all plotting windows, but instead have an x and y button that resets the x and y axes separately.

5. **(change)** In the Component Tab, dragging below the continuum for K I 7701 (in the `emsmwcs137` dataset) for component a7 does not work. The b-value is allowed to change, but the column density does not change. This can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model`, going to the Components tab, and dragging below the continuum for K I 7701 for component a7. This is probably because the column density is linked to another variable. Is there a way that we can allow the column density to change so that the linked variable can be updated, or, a way that we can change the linked variable so that the a7 column density for K I could be modified? Alternatively, if the column density is linked to a another variable and we cannot update this easily, then perhaps when the user attempts to drag the column density, the text that would have displayed the new column density value could instead make it clear that the column density is linked. This would make it clear to the user why they cannot change the column density.

6. **(bug)** When loading the context file `context/feedback01/MWC137/mwc137.model` the user is presented with a message: "Some source spectra of this project are missing or have changed. Relink them now?" This suggests that the files are not included in the `.model` file. If a user wishes to share their `.model` file with another user, they would also need to share the source spectra files. It would be good if the `.model` file could include the source spectra files, so that a user can share their `.model` file with another user without needing to share the source spectra files separately. This can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model` and observing the message about missing or changed source spectra. I have also provided another context file `context/feedback01/MWC137/mwc137_relinked.model` where I have relinked the source spectra to the files that are on this machine, so the message does not appear.

7. On the Regions Tab, if the user selects `lsfspline` as their line spread function, we should allow a button to appear beside the list box. This button should have a label `Set LSFSpline knots...` that when clicked, opens a separate dialog to build the starting parameters of the lsfspline with draggable/insertable knots. The key starting values are a smoothing parameter, and three knots. The first knot should be fixed at a value of 0.0 and be located at the left edge (-10 km/s). The second knot should be fixed at a value of 1.0 at 0.0 km/s. Finally, the third knot should be located at the right edge (+10 km/s) with a fixed value of 0.0. There should be an option to add knots, delete knots, or move/drag knots around in both x and y directions. The y-values must be between 0 and 1. Knots cannot be allowed to have the same x value, but they can have the same yvalue. The x-values can be anything between the leftmost knot and the rightmost know. The edge knots should be moveable in the x direction only. The plot should clearly display the knots, together with a faint/thin black line that connects the knots (ordered from the leftmost knot to the right most knot). There should also be displayed the smoothed version of the linearly interpolated spline (a normal weight red line) that looks like the actual LSF kernel used by ALIS to convolve the data. Finally, it would be useful if the FWHM of the LSF was indicated on the plot, together with the value, so that the user has a rough idea of the full width at half maximum of the smoothed curve used for the LSF.

8. **(bug)** On the Components Tab, Rb I appears twice in the included/excluded list of ions. Is this because 87RbI is commented out? For further context, please see the following file that will show an example of this behaviour: `context/feedback01/MWC137/mwc137.model`

## Carried over from Stage 11

*Found by Claude during Stage 11 and not done there. RJC decides whether each joins
this stage (Q12.1).*

1. **(change)** Load snips… stops the whole load when a file cannot be read or its
   name names no transition (D122, Q7.4): such a file has no transition to be placed
   at. The dialog that assigns snips could take it instead, on the right, if the
   transition were guessed from the system's ions or chosen by the user. Asked on
   Stage 11's review page; no answer was given before the stage closed.
2. **(bug)** An unassigned dataset whose last snip is cleared (CLEAR) stays on the
   Data tab with no snips and no source ("0 snips · unassigned: no source
   spectrum"), since Load snips… needs such a row to exist before its first snip is
   added. Remove dataset takes it away. It could go with its last snip.
3. **(change)** Open a fit's question is RJC's wording for every fit (D123), with
   "spectra"; a fit whose snips all share one dataset was asked in the singular
   before. Kept unless RJC wants the singular back for those (asked at the close,
   2026-11-12; the stage closed on RJC's word).

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage12_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (12.1, 12.2, …), and per item carried over
from Stage 11 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*The last task is the review (a page built by `doc/dashboard/stage12/`), and the one
after it closes the stage, as Stage 11's Tasks 11.10 and 11.11 did.*

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs, the Figure
  tab, and the lessons of Stages 10 and 11 (their "additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D128.
- `claude_prompts/dashboard_stage11.md` and its log: how users' feedback was handled
  last time, from the list to the close, the design pausing for RJC's queries.
- `claude_prompts/deferred_work.md`: what was deferred (§1.4, the GPU kept on the
  device, a series of its own; §7, from the dashboard's stages).
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 11's builder is `doc/dashboard/stage11/`; its synthetic fits come from
  `tests/dashboard_fits.py`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude. Each gives Claude's lean.*

**Q12.1 — The items carried over from Stage 11.** For each, two ways:
- **(a) It joins this stage**, after the feedback's tasks.
- **(b) It waits** for a user to ask for it, in `deferred_work.md` §7.

My lean: (b) for item 1 (a file that names no transition has nowhere to go without a
guess the user would have to check) and item 3 (RJC's wording fits every fit); (a)
for item 2, a small fix that leaves no empty dataset behind.

**Response:** I agree with your lean. For item 2, we need to make sure a dataset (even if it is unassigned and contains no SNIPs) is only removed explicitly by the user. A warning message could instead be displayed to the user so they know they can remove the dataset, as it currently contains no SNIPs and is unassigned. If they wish, they could alternatively cut a SNIP from this unassigned dataset.

**Q12.2 — What follows this stage.** Two ways:
- **(a) Another stage of feedback** (Stage 13) for what arrives after this list
  closes, as now.
- **(b) Another series first**: the GPU kept on the device (`deferred_work.md` §1.4),
  the documentation (`documentation_stage0.md`), or an item marked "Later" in
  `dashboard_stage0.md`.

My lean: (a), unless RJC names one of (b).

**Response:** I agree with option (a), for now.

## Prompts

1. Please read the list of changes and bug fixes, and the work of Stages 0–11 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Feedback`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (12.1, 12.2, etc.). The final Task (currently labelled 12.N) should be to close the stage, and should include the checks that will verify that the stage is complete.