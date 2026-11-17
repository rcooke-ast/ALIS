# Prompt file for ALIS software dashboard creation -- STAGE 13

> **Feedback from users, after Stage 12's list closed.** Stage 12 took a closed list
> of users' feedback (eight points on RJC's MWC137 project) and the item Stage 11
> carried over that joined it, with `lsfspline`'s smoothing made a FWHM in ALIS
> (Q12.2 (a)). Stage 13 takes the feedback that arrives after that list closed. RJC
> collects each point in the Feedback section below. Claude turns the list into a
> design, queries and tasks, each with its check, and builds them in order, with a
> review before the stage closes.
>
> The rules of Stages 3–12 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees. A question that comes up while a task is built pauses the
> work: it is asked in the Queries section, and its answer is logged there (Stage 11,
> Prompt 2).
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D138 at the close of Stage 12).
> "Q0.*n*" to "Q12.*n*" refer to `dashboard_stage0.md` to `dashboard_stage12.md`.
> What follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-17 (Stage 12's Task 12.13). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is. The items carried over are what Stage 12 found and did not do.*

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

## Carried over from Stage 12

*Found by Claude during Stage 12 and not done there. RJC decides whether each joins
this stage (Q13.1).*

1. **(bug)** Two datasets of a fit opened without its spectrum can share a name.
   RJC's `alis2.mod` (`context/feedback01/MWC137/`), opened without its spectra,
   gives two datasets named "39KI7701": snips 6 and 7 are grouped apart by their
   resolution labels (`va`, `vab`), and each group is named by the `label=39KI7701`
   that both data lines carry (`project._build_rows`, Q2.4). The Data tab, the Regions tab's strips and every menu that names a dataset
   then show the same name twice. Load snips… already makes a taken name unique
   ("… 2"); an imported fit's rows could do the same. Found in Task 12.12's check of
   MWC137 (scratch `mwc_open.py`, logged).
2. **(change)** An unassigned dataset emptied before Task 12.12 (a project saved
   before, its dataset with neither snips nor data) advises Load snips…, but Load
   snips… of the same file puts the snip in a new dataset ("other (unassigned) 2")
   and leaves the emptied one (Q12.15). Q12.17's response keeps projects saved before
   as they are, and new `.model` files can be started; the emptied dataset could
   instead take the snip back.

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage13_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (13.1, 13.2, …), and per item carried over
from Stage 12 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*The last task is the review (a page built by `doc/dashboard/stage13/`), and the one
after it closes the stage, as Stage 12's Tasks 12.11 and 12.13 did.*

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs, the Figure
  tab, and the lessons of Stages 10–12 (their "additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D138.
- `claude_prompts/dashboard_stage12.md` and its log: how users' feedback was handled
  last time, from the list to the close, the work pausing for RJC's queries, and
  the review's comments applied (drafts 1–7).
- `claude_prompts/deferred_work.md`: what was deferred (§1.4, the GPU kept on the
  device, a series of its own; §7, from the dashboard's stages, §7.10 and §7.11
  from Stage 12).
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 12's builder is `doc/dashboard/stage12/`; its scenes come from
  `tests/dashboard_synthetic.py` and `tests/dashboard_fits.py`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude. Each gives Claude's lean.*

**Q13.1 — The items carried over from Stage 12.** For each, two ways:
- **(a) It joins this stage**, after the feedback's tasks.
- **(b) It waits** for a user to ask for it, in `deferred_work.md` §7.

My lean: (a) for item 1, a small fix that tells two datasets apart wherever they
are named; (b) for item 2, which only a project saved before Task 12.12 can meet,
and which Q12.17's response leaves as it is.

**Response:**

**Q13.2 — What follows this stage.** Two ways:
- **(a) Another stage of feedback** (Stage 14) for what arrives after this list
  closes, as now.
- **(b) Another series first**: the GPU kept on the device (`deferred_work.md` §1.4),
  the documentation (`documentation_stage0.md`), or an item marked "Later" in
  `dashboard_stage0.md`.

My lean: (a), unless RJC names one of (b).

**Response:**

## Prompts
