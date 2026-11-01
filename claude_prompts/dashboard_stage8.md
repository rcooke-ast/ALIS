# Prompt file for ALIS software dashboard creation -- STAGE 8

> **Feedback from users.** Stage 7 made the changes RJC found by fitting real
> problems with version 1. Stage 8 does the same for the feedback of other users:
> RJC collects each point in the Feedback section below; Claude turns the list into
> a design, queries and tasks, each with its check, and builds them in order, with a
> review before the stage closes.
>
> The rules of Stages 3–7 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> model is one step of the history. ALIS outside `alis/dashboard/` changes only where
> a query agrees.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D88 at the close of Stage 7).
> "Q0.*n*" to "Q7.*n*" refer to `dashboard_stage0.md` to `dashboard_stage7.md`. What
> follows v1 is listed in `dashboard_stage0.md` ("After v1").

## Feedback

*Collected by RJC from users of the dashboard: one point per item, numbered, added as
it arrives. For each point:*
- *what the user did and saw, and what they expected instead;*
- *its kind: **(bug)**, **(change)** or **(new)**;*
- *where it can be reproduced: a project, or the model and data, and the steps (see
  Q8.1 for where the files go);*
- *who raised it, only if they agree to be named.*

*A point already on the list is not added twice: a note goes under the first, saying
how many users raised it. Claude reads this list and does not add to it.*

1. 

## Carried over from Stage 7

*Found by Claude during Stage 7 and not fixed there. RJC decides whether each joins
this stage (Q8.2).*

1. **(bug)** A `.mod.out` written by a plain ALIS blind run holds the line
   `------ BLIND MODEL ------`. The dashboard's parser refuses it, so such a run
   (examples/blind's reference output, say) shows no best-fit values. Runs made from
   the dashboard keep the real line, hidden in the bundle, and are not affected.
2. **(change)** Pure thermal (Task 7.5): a b_turb limit set before pure thermal was
   chosen is replaced by `[0.0,None]`. When the mode is left, the line is taken out
   rather than given back (Undo gives it back).
3. **(change)** Export's Corrected wavelength (Prompt 8): `vshiftscale` and
   `polyshift` scale about the middle pixel of the whole snip file, which can differ
   slightly from ALIS's when it loads only part of a snip.
4. **(tests)** The paths Stage 7 left untested: `label_owner` finding no component;
   `_shared_ratios`' skipped lines; and, in `export.py` (90% covered), a run still in
   progress, a run that wrote no `.mod.out`, and a snip whose file cannot be read.

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D88 and the stage documents. A point that asks for what a decision ruled out
  becomes a query that quotes the decision, not a task.
- **A bug becomes a test first.** It is reproduced from the user's project where there
  is one, or from a synthetic project in `tests/dashboard_synthetic.py` otherwise. The
  test fails before the fix and passes after it.
- **Outside the dashboard.** A point about ALIS itself, or about the documentation
  (planned in `documentation_stage0.md`), becomes a query: the dashboard's stages
  change ALIS only where a query agrees.
- **Users who disagree.** Two points that ask for opposite things become one query,
  with both described.
- **Grouping and order.** Points are grouped by the part of the window they touch,
  and a bug comes before the work that builds on it, as in Stage 7.
- **Shown before it closes.** Every change a user sees is shown on a review page (as
  Stage 7's, built by `doc/dashboard/stage8/`), and RJC's comments on it are applied
  before the stage closes.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage8_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests; run
> the `fast` batch before the stage closes. A task that changes ALIS outside
> `alis/dashboard/` also runs the regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (8.1, 8.2, …), and per item carried over
from Stage 7 that joins the stage, written by Claude once the list is filled in. Each
has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

**8.N — Close the stage.**
- **The review.** Publish the screenshots of what changed for RJC's review, on a new
  page with the earlier pages' stylesheet, and apply the review.
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill up to date.
- **The decisions.** Record them in `ALIS_v2_dashboard_prompts.md` (D89 onwards, from
  the Q8 queries and the review), saying which earlier decisions they refine. Update
  `dashboard_stage0.md`: Stage 8 done, and the "After v1" list.
- **Check.**
  - Every point of the list has its task logged as done with its check passing, or
    its query answered.
  - The `unit`, `gui` and `fast` batches pass. Their counts go in the log, against
    Stage 7's close: 2088 unit tests passed with 119 skipped, 211 `gui` tests passed,
    and the `fast` batch of Prompt 10.
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44).
  - If ALIS changed outside `alis/dashboard/`, the regression harness runs again at
    the close, with its `test_bundle` mode.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view and the Fit tab's runs.
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D88.
- `claude_prompts/dashboard_stage7.md` and its log: how the changes from real use
  were handled, and the review and its comments (the Export dialog, D88).
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 7's builder is `doc/dashboard/stage7/`.
- `tests/dashboard_synthetic.py`: the synthetic fits a bug's test can be built on.
- The users' files, where the Feedback list names them (Q8.1).

## Queries

*Raised by Claude with this document. Each gives Claude's lean.*

*Q8.1–Q8.5 were raised on 2026-11-01 (Stage 7's Prompt 10).*

**Q8.1 — Where a user's files go.** A point is easiest to fix with the user's project,
or its model and data. Two ways:
- **(a) In `context/feedback/`.** One folder per point, named by its number
  (`context/feedback/3/`), not committed, as `context/fitting_examples/` is not. Users'
  data stay out of the repository, and each test that needs them is skipped where
  they are absent.
- **(b) Described only.** The point says what was done, and Claude builds a synthetic
  project for each test.

My lean: (a), with (b) for a point that comes with no files.

**Q8.2 — The items carried over from Stage 7.** Three ways:
- **(a) All four join this stage**, after the feedback's tasks.
- **(b) Only the bug (1)** joins. The other three go to `deferred_work.md` §7.
- **(c) None join.** All four go to `deferred_work.md` §7.

My lean: (a). They are small, and items 1 and 2 are known faults.

**Q8.3 — A continuum alone.** Run is greyed while the model has no component (your
review of Stage 7, D85). So a user who fits a continuum alone, to check it before
adding components, cannot run from the window. Two ways:
- **(a) Keep it**, as you asked in the review.
- **(b) Allow a continuum alone.** Run is still greyed with no snip or no fit region,
  but not for a model with no component when a continuum parameter is free.

My lean: (a), unless a user's feedback asks for (b).

**Q8.4 — When the stage starts and closes.** Feedback arrives over time. Two ways:
- **(a) A closed list.** Stage 8 starts once you judge the list long enough. It
  designs, builds and closes as Stage 7 did. Feedback that arrives later goes to
  Stage 9's document, written at the close, as this one was.
- **(b) An open stage.** Stage 8 stays open and builds points in batches as they
  arrive, with a review for each batch.

My lean: (a). A closed list gives one order, one review and a close.

**Q8.5 — The rest of "After v1".** The Plot tab, the items marked "Later", the
documentation and the deferred work are listed in `dashboard_stage0.md` ("After v1").
Two ways:
- **(a) Stage 8 is the feedback alone.** The Plot tab and the rest get stages of their
  own, in the order you give the list.
- **(b) Stage 8 also starts the Plot tab.**

My lean: (a). A stage with one source keeps its review readable.

## Prompts

> RJC will be responsible for writing this section.
