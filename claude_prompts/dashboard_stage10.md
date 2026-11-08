# Prompt file for ALIS software dashboard creation -- STAGE 10

> **Feedback from users, after Stage 8's list closed.** Stage 8 took a closed list of
> users' feedback, and Stage 9 built the Figure tab. Stage 10 takes the feedback
> that has arrived since Stage 8's list closed (Q8.4, D99), and any on the Figure tab.
> RJC collects each point in the Feedback section below. Claude turns the list into a
> design, queries and tasks, each with its check, and builds them in order, with a
> review before the stage closes.
>
> The rules of Stages 3–9 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D112 at the close of Stage 9).
> "Q0.*n*" to "Q9.*n*" refer to `dashboard_stage0.md` to `dashboard_stage9.md`. What
> follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-08 (Stage 9's Prompt 6). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is.*

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

1. **(change)** Regions Tab: When a user drags the snip handles, it would be good to have two handles. One handle drags the edges for just this snip, and the other handle drags the edges for all snips simultaneously. This would make it easier to adjust multiple snips at once.
2. **(change)** Regions Tab: It appears like the default amount of the spectrum available to snip is too small and cannot be adjusted. However, dragging the handles into a region that doesn't show data makes new data appear. It would be better if there was a way to show more data as the user zooms out. Then more spectrum becomes available to snip when they release the drag. If you think something else might work better, please suggest it. We don't want to continuously update the amount of spectrum available during a drag, as it might slow down the user interface. We also don't want to show all pixels of the entire spectrum by default, as this could also slow down the user interface. We want to show a reasonable amount of spectrum by default, and then allow the user to zoom out to see more spectrum if they want to.
3. **(change)** Regions Tab: There are two options at the top of the Regions window: "Add regions to all" and "Tweak dataset regions" with a label that says "Fit regions:". I think it would be more intuitive to have a label that says: "Adjust fit regions:" with a drop down list that has three options: "all spectra", "one spectrum", "off". The "all spectra" option would allow the user to adjust the regions for all spectra simultaneously. The "one spectrum" option would allow the user to adjust the regions for just one spectrum. the "off" option would turn off the ability to adjust regions. This would make it more clear what the user is doing when they adjust regions.
4. **(change)** Regiosn Tab: Each panel displays the number of lines not labelled. It would be good to have a button that toggles between showing all lines and only a limited number of lines. For example, there could be an option that shows the main line only that is selected for this ion, or all lines from the selected ion that appear in this window's wavelength range, or all lines from all ions covered by the shown wavelength range. This would make it easier to see the main lines and not be overwhelmed by too many lines.
5. **(change)** Regions Tab: We no longer need the button that says "Load snips". This button can be removed, since there is a way to load the snips from a `.mod` file instead. Something that would be beneficial, instead, is a list box that contains *all* available snips for this System (every ion and transition that has been SNIPped). This allows the user to see what transitions have already been snipped, and to select them for display. When a snip is selected from this drop down list box, the corresponding element/ion should be updated to match, near the top of the Transitions box. Note that this list box should only contain snips that are available for this System only, and not for other Systems. If the user updates the System from the System box, the list of available snips should be updated to match the new System. This would make it easier for the user to see what snips are available for this System, and to select them for display and edit. Every time a new SNIP is created, it should be added to this list box. If a snip is deleted, it should be removed from this list box. This would make it easier for the user to manage all of their snips in one place.
6. **(change)** Regions Tab: When the cursor is near the edge of a region, there is a faint red/orange line that appears that allows the user to drag the edge of the region. It would be good if this line was a lot more visible, about as thick as the orange "Snip edges" lines. The line should only be shown when the cursor is near the edge of a region, and not when the cursor is far away from the edge. This would make it easier for the user to see where they can drag the edges of a region.
7. **(bug)** Regions Tab: When one of the H I lines is selected, the isotopes of H I and not being shown. It would be good to show the tick marks of an isotope.
8. **(bug)** Regions Tab: I encountered a bug when pressing the SNIP button:
```
The ALIS dashboard has encountered an unexpected error.
             Traceback (most recent call last):
               File "/Users/rcooke/Software/ALIS/alis/dashboard/qt/window.py", line 700, in _run
                 handler()
                 ~~~~~~~^^
               File "/Users/rcooke/Software/ALIS/alis/dashboard/qt/regions.py", line 2228, in do_snip
                 step = SN.snip_step(
                     self.session.project,
                 ...<5 lines>...
                     **self._options(),
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/snips.py", line 671, in snip_step
                 step = compose(
                     project, "", add_snip_makers(news, max_order, line_kms, clip, counted)
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/snips.py", line 101, in compose
                 step = maker(project)
               File "/Users/rcooke/Software/ALIS/alis/dashboard/snips.py", line 528, in make
                 best = CO.guess(
                     proj, snip, "legendre", max_order, clip=clip, line_kms=line_kms
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/continuum.py", line 810, in guess
                 use = continuum_pixels(
                     project, snip, function, knots, xrange, clip, line_kms, max_order
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/continuum.py", line 661, in continuum_pixels
                 keep = fit(
                        ~~~^
                     project.registry,
                     ^^^^^^^^^^^^^^^^^
                 ...<9 lines>...
                     start=keep,
                     ^^^^^^^^^^^
                 ).keep
                 ^
               File "/Users/rcooke/Software/ALIS/alis/dashboard/continuum.py", line 520, in fit
                 raise ContinuumError("There are too few good pixels to fit the continuum.")
             alis.dashboard.continuum.ContinuumError: There are too few good pixels to fit the continuum.
```
This appears to be a problem with the continuum fitting in a region where no data, or bad data exist. If it is just one dataset where the problem is, we could alert the user to there being an issue with dataset `<dataset name>` and that they should check the data. We should still allow them to proceed using any other datasets that do not have concerns. This error occurred when I was trying to press the SNIP button for the Al II 1670 transition after loading three datasets. The corresponding data are provided as context in the following folder: `context/feedback01/`. I suspect the error is with the `UVES2010` data. I've included the `J0035m0918.model` file in that directory. This also happens with the Si II 1526 line.
9. **(change)** Regions Tab: Currently the transitions are sorted by strength. It would be good to have a way to sort the transitions by wavelength, or by strength. This would make it easier for the user to find the transition they are looking for. The default should be ordered by wavelength.
10. **(change)** Regions Tab: When the user selects the "Zoom box" and zooms into a region of data they are interested in, and then selects the "Tweak dataset regions" option, the y-axis range resets to the full range of the data. It would be good if the y-axis range didn't reset automatically. After all, there is a "Home" button that allows the user to reset the ranges themselves, if they wish. We should not automatically change how the user has positions the y-axis and x-axis ranges.
11. **(change)** Components Tab: When the temperature is set to "T free", there is no way to set the temperature value. Like all other variables, there should be a way to set the temperature value, when it is either free or fixed. This would make it easier for the user to set the temperature value, without having to adjust the `.mod` file directly.
12. **(bug)** Plot Tab: Each panel shows a transition's label, with the element, ion stage, and wavelength. The current version of the Plot tab *rounds* the wavelength to the nearest integer. The standard in the field is to round down, so C II λ1334.5323 becomes C II λ1334 (but it is currently incorrectly shown as C II λ1335). 
13. **(bug)** Plot Tab: If I change the y-axis scale of a panel, the residuals and the label are no longer shown (they are probably drawn, but are not visible due to the different y-axis range shown).

## Carried over from Stage 9

*Found by Claude during Stage 9 and not done there. RJC decides whether each joins
this stage (Q10.1).*

1. **(change)** The DH preset always starts at Lyα. A figure of the higher members,
   as `DH_Ly8-Ly14*.py` draw them, needs its cells filled by dragging; the preset
   could instead start from a member the user chooses.
2. **(change)** A panel's own x title in a middle row (a wavelength panel among
   velocity ones, as `DH_Lyseries.py` has) can meet the panel below it. The margins
   are worked out from the panels (D111), but the spacing between rows is not; the
   user widens it by hand.
3. **(tests)** The paths Stage 9 left untested: a drag begun with the mouse on a cell
   (the drop side is tested with real drop events, but `QDrag.exec` cannot run
   off-screen); and the refusals of a figure size or spacing out of range.

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D112 and the stage documents. A point that asks for what a decision ruled out
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
  built by `doc/dashboard/stage10/` with the earlier pages' stylesheet, and RJC's
  comments on it are applied before the stage closes.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage10_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (10.1, 10.2, …), and per item carried over
from Stage 9 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

**10.N — Close the stage.**
- **The review.** Publish the screenshots of what changed for RJC's review, on a new
  page with the earlier pages' stylesheet (built by `doc/dashboard/stage10/`), and
  apply the review.
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill up to date. Record the decisions in
  `ALIS_v2_dashboard_prompts.md` (D113 onwards, from Q10.* and the review), saying
  which earlier decisions they refine. Update `dashboard_stage0.md`: Stage 10 done,
  and the "After v1" list.
- **Check.**
  - Every point of the list, and every item carried over that joined, has its task
    logged as done with its check passing, or its query answered.
  - The `unit`, `gui` and `fast` batches pass. Their counts go in the log, against
    Stage 9's close: 2176 unit tests passed with 120 skipped, 256 `gui` tests, and
    149 `fast` tests (all under pytest 9.2.0).
  - CI passes on both platforms, the `gui` job within its 90 minutes (Stage 9 raised
    the limit; Ubuntu's job was near 56 minutes by estimate).
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44).

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs and the Figure
  tab (its "Stage 9 additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D112.
- `claude_prompts/dashboard_stage8.md` and its log: how users' feedback was handled
  last time, from the list to the close.
- `claude_prompts/dashboard_stage9.md` and its log: the Figure tab, and RJC's feedback
  on it (Prompts 4 and 5).
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 9's builder is `doc/dashboard/stage9/`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude with this document (2026-11-08). Each gives Claude's lean.*

**Q10.1 — The items carried over from Stage 9.** Three ways:
- **(a) All three join this stage**, after the feedback's tasks.
- **(b) Only the tests (3)** join. The two changes wait for a user to ask for them,
  in `deferred_work.md` §7.
- **(c) None join.** All three go to `deferred_work.md` §7.

My lean: (a). They are small, and items 1 and 2 are what a user drawing D/H figures
meets first.

**Response:** I agree with (a).

**Q10.2 — Your feedback on the Figure tab.** You said that you would try Q9.12's
options in the scripts and send what to add, change or remove. Two ways:
- **(a) In this list**, as points marked "(Figure tab)", with the users' points.
- **(b) In a prompt of their own**, before the list is read.

My lean: (a). One list gives one design, one order and one review.

**Response:** I have included several items in this design document. Let's limit the changes to this list for now. If I receive further feedback, I can add these specifically in a future development stage. 

**Q10.3 — When the stage closes, and what follows it.** Stage 8's list was closed
before its design was written (Q8.4 (a)), and the Plot tab came next (Q8.5 (a)).
Two ways for Stage 10:
- **(a) The same.** The list closes when you judge it long enough. Feedback that
  arrives later goes to Stage 11's document, written at this stage's close, unless
  you name another stage to come first: one of the items marked "Later" (re-attaching
  to a run, command search, the `link` editor, reproducing an imported fit, Orders
  mode), the documentation (`documentation_stage0.md`), or the deferred work.
- **(b) An open stage.** Stage 10 builds points in batches as they arrive, with a
  review for each batch.

My lean: (a). A closed list gives one order, one review and a close, as Stage 8 had.

**Response:** I agree with option (a).

## Prompts

1. Please read the list of changes and bug fixes, and the work of Stages 0–9 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Feedback`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all of the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (10.1, 10.2, etc.). The final Task (currently labelled 10.N) should be to close the stage, and should include the checks that will verify that the stage is complete.
