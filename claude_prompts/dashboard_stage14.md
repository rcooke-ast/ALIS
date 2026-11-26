# Prompt file for ALIS software dashboard creation -- STAGE 14

> **ALIS's sub-pixels, then the feedback that arrives.** Stage 13 took the users'
> feedback after Stage 12's list (four points and one item carried over). Its Q13.7
> found that ALIS sets each snip's sub-pixels when it loads a model, from the
> starting values, so a `.mod.out` run again from its best fit can use other
> sub-pixels than its fit did; RJC made that, with the default `nsubmin` of 10
> deferred in Stage 12, the key item of this stage, which comes before the
> documentation (Q13.2's and Q13.7's responses). This draft gathers the key item
> and the deferred items related to it, for RJC to review and prioritise, and keeps
> a Feedback section for what users report meanwhile.
>
> The rules of Stages 3–13 hold for the dashboard: the logic needs no Qt and goes in
> `alis/dashboard/`, tested with pytest alone; the windows go in
> `alis/dashboard/qt/`; everything shown comes from the project model through the
> blinding gate, and every change to the project is one step of the history. This
> stage changes ALIS itself (its loader, its writer, its default settings and the
> harness's references), and each change is agreed in a query first. A question that
> comes up while a task is built pauses the work: it is asked in the Queries section,
> and its answer is logged there.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D146 at the close of Stage 13).
> "Q0.*n*" to "Q13.*n*" refer to `dashboard_stage0.md` to `dashboard_stage13.md`.

*Drafted by Claude on 2026-11-24 (Stage 13's Task 13.11). No code was changed. The
key item and the related items are proposals for RJC to review and prioritise; the
design and the tasks are written once RJC has chosen. Prompt 1 (2026-11-24) applied
RJC's responses to Q14.1–Q14.3 (no related item joins; the item carried over joins;
the sub-pixels recorded first, then `nsubmin` 10), wrote the design and Tasks
14.1–14.7 with Q14.4 and Q14.5, and drafted `dashboard_stage15.md`, the stage after
the documentation. Prompt 2 (2026-11-24) applied RJC's responses to Q14.4 and Q14.5
((a) in each point), added Task 14.2 (a clearer name for `nsubpix`, with Q14.6) and
Task 14.4 (the other ways a profile may be under-sampled, as warnings, with Q14.7),
so that the tasks became 14.1–14.9, and noted the CI run RJC reported in the close.
Prompt 3 (2026-11-24) applied RJC's responses to Q14.6 (every name of the sub-pixels
by one convention, `nsubpixel…` per pixel and `nsubsigma…` per standard deviation,
with no old name read) and Q14.7 (warn only where the sub-pixels should be
increased) to Tasks 14.2–14.9, with Q14.8 and Q14.9 on what they left open. Prompt 3,
run again (2026-11-25), applied RJC's responses to Q14.8 and Q14.9 (each lean; the
archive of RJC's files directly in `context/`) and checked the list against the code.
No code was changed.*

## The key item: ALIS's sub-pixels (`deferred_work.md` §7.10)

**(a) The sub-pixels a `.mod.out` is evaluated on** (Stage 13, Task 13.2, Q13.7).
- ALIS works out each snip's sub-pixels when it loads a model (`load.py`, each
  function's `set_vars` and its `nexbin`), from the starting values: the narrowest
  profile asks for the most, between `run nsubmin` and `run nsubmax`.
- A `.mod.out` holds the best fit as its starting values, so a run of it, or the
  harness's evaluation at the best fit (mode (b)), can use other sub-pixels than the
  fit did. On `examples/lsf_spline` the fit started at a smoothing of 4 km/s on 6
  sub-pixels; its `.mod.out` starts at 0.12 km/s and takes 21, and the model differs
  at 15 of 460 pixels, by up to 2.3 times the error. Pinned at 6, it agrees exactly.
- The dashboard is not affected: it draws a run with the run's own input model.
- **To finish:** ALIS records each snip's sub-pixels in the `.mod.out`, as a
  data-line keyword (as Stage 5.4's `bufferpix` records the pixels loaded), and uses
  them when given; `examples/lsf_spline`'s mode (b) comes back into the gate
  (`FIXEDPARAM_EXCLUDE`).

**(b) ALIS's default `nsubmin` of 10** (Stage 12, Q12.9 and Q12.13 (b)).
- Raising the fewest sub-pixels per pixel from 5 to 10 failed 27 of the `fast`
  batch's 149 tests in Stage 12: 15 fits (9 shipped examples, each as a bundle too,
  and 6 context fits), 2 evaluations at the best fit, and `examples/generate`'s data.
  Some fits moved (J0903p2628: a parameter 143.6 → 128.2, 3.5σ; HS0105p1619: χ²
  2086.3 → 2137.0; `splineabs` linear: a 1σ error 113.2 → 6.2).
- **To finish:** each case that moves refitted and checked by eye, its references
  regenerated with their χ² before and after, `doc/ALIS_workflow.md`'s table,
  `doc/tex_files/description.tex` and `CHANGELOG.md`.

The two meet: references written after (a) carry their sub-pixels, so (a) first,
then (b), regenerates each reference once (Q14.3).

## Related deferred items, for RJC to prioritise

Each is in `deferred_work.md`, with what it would take. A lean on whether it joins
this stage is in Q14.1. *None joins (Q14.1's response): each waits in
`deferred_work.md`, and is offered to Stage 15 (`dashboard_stage15.md`).
`brokenpowerlaw`'s references are regenerated with the others in Task 14.7 if
`nsubmin` 10 moves them, and it stays out of the evaluation at the best fit.*

1. **§2.4 `examples/brokenpowerlaw`**: its `.mod.out.reference` evaluates to χ²
   374.98 against a recorded 338.15, and it is out of the fixed-parameter gate. The
   suspected cause is the writer emitting a rejected trial step rather than the best
   accepted point. Related: it is the other case where a `.mod.out` does not
   reproduce its fit, and its fix moves references as (b) does.
2. **§2.1 Wavelength-dependent resolution**: a Gaussian LSF whose width varies with
   wavelength, per pixel, removed in Stage 6.5 as untested. Related: it convolves on
   the sub-pixels, and its test would be written against them. Larger than the
   rest; RJC's real target is a non-Gaussian LSF that varies with wavelength.
3. **§7.7 `apod`'s buffer and example data in frequency**: `apod`'s buffer reads its
   value as a FWHM in Hz, and nothing tests data in frequency. Related: the buffer
   and the sub-pixels are both worked out from the resolution at load.
4. **§5 The `context/` fits in CI**: the 23 context cases of the `fast` batch run
   only on RJC's machine. Related: (b) regenerates several of their references, and
   CI would then watch them; their data are RJC's own, so this needs RJC's say.

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

## Carried over from Stage 13

*Found by Claude during Stage 13 and not done there. RJC decides whether each joins
this stage (Q14.2).*

1. **(bug)** `lsffile`'s loader keeps its keywords on the function's instance
   between data lines (`alis/functions/lsffile.py`, `load`): a data line that names
   no `index:` takes the previous line's, where it should take the default (a kernel
   drawn at random from a file of several). Stage 13 reset `unit` for each line
   (Task 13.7) and left `index` as it was. The dashboard always writes `index:` for a
   file of several kernels, so only a model written by hand meets it. Found while
   building Task 13.7. *It joins (Q14.2's response): Task 14.1, with `apod` and
   `lsf`, whose loaders do the same (Q14.5).*

## Design

*Written by Claude once RJC has chosen the items (Q14.1–Q14.3): for each, what
changes, in which modules, and what stays as it is.*

### How the stage is built

- **Agreed first.** Each change to ALIS outside `alis/dashboard/` is agreed in a
  query before it is built: the key item in Q13.7 and Q14.3, the item carried over in
  Q14.2; the keyword's name and rules in Q14.4, `apod` and `lsf` in Q14.5, the names
  of the sub-pixels in Q14.6, Q14.8 and Q14.9, and the warnings of profiles with too
  few sub-pixels in Q14.7 and Q14.9.
- **A bug becomes a test first**, which fails before the fix and passes after it.
- **References.** No reference changes until its case's new fit has been checked.
  A change to the references lists each case that moves, with its χ² before and
  after, the largest change of a parameter in σ, and its runtime, and is checked by
  eye on the review page. The blind cases are reported by χ² and the size of each
  change in σ, without values, signs or figures (D24, Q0.9, Q0.11).
- **The order** (Q14.3 (a)): the loader's fix first (14.1), so that the harness after
  it is the stage's baseline, and every name of the sub-pixels by one convention (14.2),
  so that what follows is written with them; then the sub-pixels recorded (14.3, ALIS),
  the warnings of profiles with too few sub-pixels (14.4) and the dashboard's side
  (14.5); then `nsubpixel_min` 10, measured on every batch (14.6); then each reference
  that moves regenerated once, with its sub-pixels (14.7); the review (14.8) and the
  close (14.9).
- **Feedback.** A point RJC adds to the Feedback section while the stage is built
  becomes a task after 14.7, once its queries are answered; the review and the
  close are renumbered.

### 14.1–14.7 — The key item and the item carried over

*Written by Claude on 2026-11-24 (Prompt 1), from the responses to Q14.1–Q14.3 and
the code at the start of the stage (f9f4539); Prompt 2 added the names and the
warnings, from the responses to Q14.4, and Prompt 3 widened them, from the responses
to Q14.6 and Q14.7. What was found is written with the names of the code at the
start of the stage. Each item's design is in its task below.*

**What was found.** By reading the code, and one scratch script (logged). No code
was changed.
- **The sub-pixels a `.mod.out` is evaluated on** (§7.10, (a) above).
  - `load.load_subpixels(slf, p0)` gives each snip (a specid's data line) its
    sub-pixels per pixel, `nexbins[sp][sn]`: `nsubmin`, raised to the most that any
    line of the model (`set_vars`'s `nbn`, from the values it is given) or the
    resolution asks for, and capped at `nsubmax`. `main.py` calls it once at load,
    with the starting values (`self._nexbins`); `model_eval` uses that grid at every
    evaluation, unless `renew_subpix` works it out again each time.
  - `save.save_model` rewrites each data line of the `.mod.out` with the fitted
    resolution and shift, and adds `bufferpix=[l,r]` (Stage 5.4) where the line gave
    no `loadrange` or `bufferpix`, so that the `.mod.out` loads the pixels its fit
    loaded. Nothing records `nexbins`.
  - A data line's `nsubpix=N` is not the sub-pixels: it is the oversampling per
    standard deviation (`run nsubpix`, 5) from which each line works out what it asks
    for. The new keyword needs a name of its own (Q14.4 (i)), and `nsubpix` a
    clearer one (Q14.4 (i)'s response, Q14.6). 34 of RJC's `context/` models set
    `run nsubpix` (5, 10, 100, 150 or 1000); no shipped example sets it, and no model
    gives it on a data line.
  - The dashboard's parser refuses a data-line keyword it does not know
    (`model.DATA_KEYWORDS`: "Unknown keyword on a data line"), and the Fit tab reads
    each run's `.mod.out` with it (`runs.out_values`). A `.mod.out` carrying the
    keyword would show no best fit until the dashboard knows it, so Task 14.3 adds it
    with ALIS's change.
  - The dashboard works out sub-pixels through ALIS's own `load_subpixels`
    (`profile.py`, `continuum.py`, `validate.py`), which will follow the keyword with
    no change; `lsfknots.subpixels`, which draws the knots dialog's kernel, repeats
    ALIS's sum and will not.
  - Continue from best fit copies the best-fit values alone (`results.continue_plan`),
    never a data line.
  - Where a line asks for more sub-pixels than `nsubmax`, `load_subpixels` caps it
    there and says nothing; and once a fit has its grid, nothing compares it with what
    the best fit asks for (`lsf_spline`: 6 used, 21 asked for at its best fit). Its
    only sub-pixel warning is `warn_subpix`'s, when a snip uses more than 100, which
    is about the cost (Q14.4 (ii)'s response, Q14.7).
- **The default `nsubmin` of 10** (§7.10, (b) above).
  - `config.py` gives 5; `doc/ALIS_workflow.md`'s table "5/21",
    `doc/tex_files/description.tex` 5. The dashboard reads the model's setting, or
    ALIS's default (`lsfknots.spix` and `subpixels`), so it follows a new default.
  - Stage 12 measured 10 on the `fast` batch alone (27 of its 149 tests failed). The
    `medium` batch (five context fits of 2 to 9 minutes, two of them blind) and the
    `slow` batch (DH_orders, a 21-minute fit) were not run, so the list of the cases
    that move is not yet complete.
  - The harness has 43 cases (`tests/manifest.py`): 37 `fast`, 5 `medium`, 1 `slow`;
    5 blind (their fits alone, never evaluated at the best fit), 17 with a covariance
    golden. The `context/` cases' references reproduce on RJC's machine, where the
    batches run (the nine of `MINIMISATION_MACHINE_DEPENDENT` pass there; Linux CI
    skips them).
  - `brokenpowerlaw`, `tophat` and `lsf_spline` are left out of the evaluation at the
    best fit (`FIXEDPARAM_EXCLUDE`); only `lsf_spline`'s reason is this stage's.
- **The item carried over** (`lsffile`'s `index`). A scratch script
  (`keywd_carry.py`) loads one `lsffile` instance on three lines: `index:2`, then a
  line with none, whose `index` is 2 (it should be −1, a kernel drawn at random); a
  third line naming no file loads with the previous line's (it should stop: "name
  must be set"), since which keywords were given is kept on the instance too.
  - `apod` does the same with `kind` (the same script: a second line with no `kind:`
    takes the first's `uniform`, where a line alone stops); `lsf` loads `name`,
    `grating`, `life_position`, `cen_wave` and `slit` the same way (by reading its
    code). In all three, a `blindrange` given on two lines of one function would
    gather both lines' values, a list appended to on the instance and shared with the
    earlier line's copy (by reading).
  - `base.Base` and `lsfspline` start each line from a copy of their defaults.
  - No model in `examples/` or `context/` meets it: its three `lsf` lines and one
    `lsffile` line each name their file.

**What stays as it is.**
- How ALIS works out the sub-pixels of a snip whose line records none (by the new
  names of 14.2): `nsubpixel_min`, what the lines and the resolution ask for at the
  values given (`nsubsigma` · pixel / σ), `nsubpixel_max`, and `nsubpixel_renew`. A
  `.mod.out` written before this stage reads as before, unless it names a setting by
  an old name (14.2: refused, with the new name).
- `bufferpix`, and the other data-line keywords of the `.mod.out`.
- What `nsubsigma` (5), `nsubpixel_max` (21) and `nsubpixel_renew` do, and their
  defaults; only their names change.
- The dashboard draws a run with the run's own input model (Q13.7), and Continue
  from best fit copies values alone (Q14.4 (iii)).
- The four related items of Q14.1 wait (`brokenpowerlaw` stays out of the
  evaluation at the best fit, §2.4); they are offered to Stage 15.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage14_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included, from the repository (its
> `context/` cases need their own paths).

*One task per chosen item, each with what it builds or fixes and its check (the test
that fails before the change and passes after it), written by Claude once RJC has
chosen. A change to the references lists each case that moves, with its χ² before
and after, and is checked by eye. The last task but one is the review (a page built
by `doc/dashboard/stage14/`), and the last closes the stage.*

*Tasks 14.1–14.7 were written by Claude on 2026-11-24 (Prompt 1), from the responses
to Q14.1–Q14.3, with the leans of Q14.4 and Q14.5. At Prompt 2 (the same day), RJC's
responses agreed with both ((a) in each point). Q14.4 (i)'s response added a clearer
name for `nsubpix` (Task 14.2, written with the lean of Q14.6), and (ii)'s the other
ways a profile may be under-sampled, as warnings with what resolves them (Task 14.4,
written with the lean of Q14.7). The tasks became 14.1–14.9. At Prompt 3 (the same day),
the response to Q14.6 widened Task 14.2 to every name of the sub-pixels, by one
convention (`nsubpixel…` counts per pixel, `nsubsigma…` per standard deviation), with no
old name read; the response to Q14.7 kept 14.4's warnings to too few sub-pixels, and
removed the warning of too many. Q14.8 and Q14.9 asked what the responses left open,
and Tasks 14.2–14.9 were written with their leans. At Prompt 3, run again
(2026-11-25), RJC's responses agreed with each lean, and asked that the archive of his
files be directly in `context/` (Task 14.2); checked against the code, 14.2's modules
lose `scripts/run_alis.py` (`--set` is read in `load.py`) and the dashboard's
`settings.py` (it takes ALIS's list), and its dashboard gains a problem for an old
name, with a fix. The order (Q14.3 (a)):*
- *the loader and the names: the item carried over (14.1), and every name of the
  sub-pixels by one convention (14.2), so that what follows is written with them;*
- *the key item's (a), the sub-pixels recorded: ALIS (14.3), its warnings of profiles
  with too few sub-pixels (14.4), then the dashboard (14.5);*
- *its (b), `nsubpixel_min` 10: what it moves, measured and checked (14.6), then the
  references regenerated once, with their sub-pixels (14.7);*
- *the review (14.8), and the close (14.9).*

**14.1 — Each data line's own resolution keywords: `lsffile`, `apod` and `lsf` (item
1 carried over, bug; `alis/functions/lsffile.py`, `apod.py`, `lsf.py`; D145; Q14.2,
Q14.5).** *Q14.2 (a) and Q14.5 (a), RJC's responses.*
- **The bug.** The loaders of `lsffile`, `apod` and `lsf` keep their keywords, and
  which of them were given, on the function's instance from one data line to the
  next. A line that names no `index:` takes the previous line's (it should take −1,
  a kernel drawn at random from a file of several); a line that names no file
  (`lsffile`, `lsf`) or no `kind:` (`apod`) takes the previous line's, where ALIS
  should stop with "… must be set"; a `blindrange` given on two lines gathers both.
  Stage 13 reset `lsffile`'s `unit` alone.
- **The fix.** Each line starts from a copy of the function's defaults, and of which
  keywords it requires, as `base.Base` and `lsfspline` do; `lsffile`'s reset of
  `unit` becomes part of it. A model whose lines each name their keywords loads as
  now. `CHANGELOG.md` (Fixed).
- **Check.** Made into tests first:
  - in `tests/test_lsffile.py` (`unit`): a file of three kernels on two lines,
    `index:2` then none: the second line's `index` is −1 (before, 2); a line naming
    no file after one that does stops with "name must be set" (before, it loaded the
    first line's file); `unit` per line, as Stage 13's test has it;
  - in a new `tests/test_resolution_keywords.py` (`unit`): the same for `apod`
    (`kind`) and `lsf` (`name`; marked with `optional_deps` if its loader needs
    linetools); a `blindrange` on two lines of one function stays each line's own,
    for all three (before, the first line's holds both);
  - every model of `examples/` and `context/` loads as before (the parser tests over
    every model; `test_function_interface.py`); `test_save_helpers.py`'s round trips
    pass; the regression harness, its `test_bundle` mode included, with every
    reference unchanged (`examples/lsf_file` and `lsf_hst` among them). Its counts
    are the stage's baseline.

**14.2 — Every name of the sub-pixels, by one convention (Q14.6's and Q14.7's
responses; `alis/config.py`, `load.py`, `main.py`, `model_eval.py`, `minimise.py`,
`gpu_dispatch.py`, `alis/functions/`; the dashboard's `model.py`, `lsfknots.py`,
`profile.py`, `continuum.py`, `validate.py`, `qt/knots.py`; F9; Q14.6–Q14.9).**
*Q14.6's and Q14.7's responses, and RJC's to Q14.8 and Q14.9 (each lean, the archive
directly in `context/`).*
- **The convention** (Q14.6's response): a name that counts sub-pixels per pixel
  begins `nsubpixel`, and one that counts them across a profile's standard deviation
  begins `nsubsigma`; `_min` and `_max` bound them.
- **The names** (Q14.8 (i)):
  - `run nsubpix` → `run nsubsigma` (5): the sub-pixels across one standard
    deviation of the narrowest profile, from which each line works out the
    sub-pixels per pixel it asks for (`nsubsigma` · pixel / σ). A data line's
    `nsubpix=N` → `nsubsigma=N`, for its file alone.
  - `run nsubmin` → `run nsubpixel_min` (5; 10 from Task 14.6), `run nsubmax` →
    `run nsubpixel_max` (21): the fewest and the most sub-pixels per pixel.
  - `run renew_subpix` → `run nsubpixel_renew`: each snip's `nsubpixel` worked out
    again at every evaluation.
  - The data-line keyword of Task 14.3 is `nsubpixel=N`, the snip's sub-pixels per
    pixel (Q14.4 (i)'s `subpixels=N`, by the convention).
  - `run nsubsigma_min` (new, Q14.9): the fewest sub-pixels across a profile's
    standard deviation before ALIS warns (Tasks 14.3 and 14.4); 5 by default. A model
    whose `nsubsigma` is below it is refused, naming both. There is no
    `nsubsigma_max` (Q14.9).
  - `run warn_subpix` and its warning (a snip on more than 100 sub-pixels per pixel)
    are removed: too many sub-pixels are not a problem, and ALIS warns only where
    they should be increased (Q14.7's response).
  - ALIS's messages use the names: the log's "sub-pixellation scale: 6 for spectrum
    …, segment 0" gives "nsubpixel 6" for the snip, and "Cannot calculate voigt
    subpixellation -- width = 0.0" (in six functions) says that `nsubpixel` cannot be
    worked out for a width of 0.
- **No old name is read** (Q14.6's response, Q14.8 (ii)). A setting, a data-line
  keyword or a `--set` that gives an old name is refused, with a message naming its
  new one, from a table of each old name and its new one in `config.py`, kept for
  that message alone (`load.py` reads the settings, the data lines and `--set`).
  `run_alis --list-settings` lists the new names alone.
- **Every model that names an old one, rewritten** (Q14.8 (iii)): in the repository,
  every model, test and page (`examples/` has none), but the records of earlier
  stages (the mockups, Stage 12's review builder, `claude_prompts/`), which describe
  what was then. RJC's files that do — 66 models and references under `context/`, and
  4 project bundles under `context/feedback01/` — are rewritten by a script (their
  settings lines and data-line keywords alone), after their originals are archived,
  with their paths, in `context/stage14_old_names.tar.gz` (directly in `context/`,
  Q14.8 (iii)'s response), each change logged. A bundle is extracted and packed again
  with `alis/bundle.py`, so that its checksums stay right, with every member that
  names one rewritten: its model, and each stored run's input model and `.mod.out`,
  so that its runs still show their best fits. The harness's references change in
  their settings lines alone.
- **The internal names** (Q14.8 (iv)): each snip's sub-pixels per pixel (`nexbins`,
  `_nexbins`) → `nsubpixel`; `datopt['nsubpix']`, which holds `nsubsigma` times the
  pixel's width → `datopt['nsubsigma_width']`; the `nexbin=` argument of every
  function's `set_vars` (that width, with the bin type) → `sampling=`, in the 26
  modules of `alis/functions/` that name it (`base.py` among them) and
  `doc/tex_files/newfunctions.tex`.
- **The dashboard.** All settings… offers the new names, with their descriptions from
  `config.py`, as it takes ALIS's list (`settings.py` unchanged); `DATA_KEYWORDS`
  takes `nsubsigma` and `nsubpixel`; `lsfknots.py`, `profile.py`, `continuum.py` and
  `qt/knots.py` use the new names, and so does any text of the window (the knots
  dialog's spix, one sub-pixel at `nsubpixel_min`). A project's model that gives an
  old name, on a settings line or a data line (a `.mod` written before this stage,
  opened with Open a fit (.mod)…), has a problem in the validator naming its new one
  (from `config.py`'s table), with a fix that renames it, one step, so that Run is
  not left to ALIS's refusal. Task 13.10's test of a run's warnings
  (`test_dashboard_runs_process.py`), which made ALIS warn with `run warn_subpix 1`,
  takes another of ALIS's warnings.
- `CHANGELOG.md` (Changed: each old name and its new one, and that the old are
  refused; Removed: `warn_subpix`); `doc/ALIS_workflow.md`'s tables of settings and
  data-line keywords, the convention in a sentence, and wherever it names one;
  `doc/tex_files/description.tex` and `newfunctions.tex`.
- **Check.** Made into tests first:
  - in `test_config.py` (`unit`): the settings `nsubsigma` (5), `nsubsigma_min` (5),
    `nsubpixel_min` (5), `nsubpixel_max` (21) and `nsubpixel_renew` (False) (before,
    none), and none of `nsubpix`, `nsubmin`, `nsubmax`, `warn_subpix` and
    `renew_subpix`;
  - in `test_load_model.py`, `test_load_files.py` and `test_cli.py` (`unit`): each old
    name, as a setting, as a data-line keyword and through `--set`, is refused with a
    message naming its new one (before, read); the new names give the sub-pixels the
    old gave (each snip's `nsubpixel` on a synthetic spectrum, as before);
    `nsubsigma` below `nsubsigma_min` is refused; `--list-settings` lists the new
    names alone;
  - `grep -rE 'nsubpix\b|nsubmin|nsubmax|warn_subpix|renew_subpix|nexbin'` over
    `alis/`, `tests/`, `doc/` and `examples/` finds the old names only where they are
    meant: `config.py`'s table, the tests of the refusal and of the validator's
    problem, `CHANGELOG.md`'s entry of the rename, and the records of earlier stages
    (the mockups of `doc/dashboard/mockups/` and Stage 12's review builder), which
    describe what was then and stay as written; none in `context/` after the
    script, but in the archive;
  - in `test_dashboard_settings.py`: each new name offered with its description, set
    and cleared byte for byte, one step each; in `test_dashboard_validate.py`, a data
    line with `nsubsigma=` or `nsubpixel=` has no problem; one with `nsubpix=` has
    one that names `nsubsigma`, and a settings line `run nsubmin 10` one that names
    `nsubpixel_min` (before, none); each fix is one step, whose model ALIS reads, and
    one undo gives back the text byte for byte;
  - `test_function_interface.py` (every function's `set_vars` with `sampling=`); every
    model of `examples/` and `context/` reads with the same values and sub-pixels as
    before; the regression harness, its `test_bundle` mode included, every reference
    unchanged but in its settings lines; RJC's four bundles open in the dashboard and
    read (scratch script, logged); pre-commit on every changed file.

**14.3 — ALIS records each snip's sub-pixels in the `.mod.out`, and uses them when
given (the key item (a), `deferred_work.md` §7.10; `alis/load.py`, `alis/save.py`,
`alis/config.py`, `alis/dashboard/model.py`; Stage 5.4; D140; Q13.7; Q14.3 (a);
Q14.4 (i), (ii); Q14.6).** *Q14.4 (i) (a) and (ii) (a), RJC's responses, with the
keyword named by Q14.6's response.*
- **What (ALIS).**
  - **The keyword.** A data line may carry `nsubpixel=N`, N a whole number of at
    least 1: the snip's sub-pixels per pixel, the number ALIS's log gives for it
    (14.2). `load_data` reads it, refuses any other value (naming the line), and
    keeps it with the line's options (`datopt`).
  - **Used when given.** `load_subpixels` gives the snip N in place of what it works
    out, whatever `nsubpixel_min`, `nsubpixel_max` and `nsubpixel_renew` say, as
    `bufferpix` takes the place of the resolution's rule. It still works out what the
    lines ask for; where N leaves one of them with fewer than `nsubsigma_min`
    sub-pixels across its standard deviation (Q14.9), ALIS warns once at load, naming
    each such snip, N and what its lines ask for, and saying that removing
    `nsubpixel=` from its data line works them out again. The log's line for each
    snip says when its `nsubpixel` comes from its data line.
  - **Written.** `save_model` adds `nsubpixel=N` to each data line of the `.mod.out`,
    beside `bufferpix`: the sub-pixels its fit used. A line that gave its own keeps
    it as written. With `nsubpixel_renew`, a fit's sub-pixels change at each
    evaluation, and those of its best fit are what a `.mod.out` works out again, so
    none is added.
  - `CHANGELOG.md` (Added); `doc/ALIS_workflow.md`'s table of data-line keywords
    gains `nsubpixel=N`, beside `nsubsigma=N` (14.2).
- **What (the dashboard).** `nsubpixel` is in `model.DATA_KEYWORDS` (14.2), so that a
  run's `.mod.out` that carries it reads (its Final model and Results), and a model
  that carries it opens with no problem. The rest of the dashboard's side is 14.5.
- **Check.** Made into tests first:
  - in `tests/test_save_helpers.py` (`unit`, a synthetic spectrum): a model whose line
    asks for 8 sub-pixels at its starting values is written with best-fit values that
    would ask for 21: the `.mod.out`'s data line carries `nsubpixel=8`, and read
    back, `load_subpixels` gives 8 (before, no keyword, and 21); a line given
    `nsubpixel=3` keeps it and is evaluated on 3; with `nsubpixel_renew`, no keyword
    is added;
  - in `tests/test_load_files.py` (`unit`): `nsubpixel=0`, `-2`, `6.5` and `x` are
    refused, each naming its line; a given N is used as given beside `run
    nsubpixel_min 10` and `run nsubpixel_max 4`; where N leaves a line below
    `nsubsigma_min`, one warning names the snip, N and what is asked (`logmsgs`), and
    none where it does not;
  - a new `tests/test_subpixels_fit.py` (`fast`): `examples/lsf_spline` fitted on a
    staged copy, then its own `.mod.out` evaluated at zero iterations: the model
    equals the fit's `_fit.dat` at every pixel (before, 15 of 460 pixels differ, by
    up to 2.3 times the error, Q13.7); `examples/metal_line_abs` the same, a case
    whose sub-pixels do not move;
  - in `test_dashboard_runs.py`: a `.mod.out` whose data lines carry `nsubpixel=`
    gives every free parameter's value;
  - the regression harness, its `test_bundle` mode included: every reference
    unchanged (none carries the keyword yet, so each is evaluated as before);
    `test_writer_round_trip.py`, `test_function_interface.py` and the dashboard's
    fits (`fast`) pass.

**14.4 — A profile with too few sub-pixels warns (Q14.4 (ii)'s and Q14.7's responses;
`alis/load.py`, `alis/main.py`; Q14.9).** *Q14.7's response: ALIS warns only where
the sub-pixels should be increased; the level `nsubsigma_min`, Q14.9 (a).*
- **What.** Beside 14.3's warning (a line that records too few sub-pixels), ALIS
  warns of the two other ways a profile can be left with fewer than `nsubsigma_min`
  sub-pixels across its standard deviation (Q14.9), and of nothing else about the
  sub-pixels (Q14.7's response; `warn_subpix`'s warning of too many went in 14.2):
  - **A best fit that asks for more sub-pixels than its fit used.** A fit's
    sub-pixels are set from its starting values. At the end of the fit, ALIS works out
    what each snip's lines ask for at the best fit (as `load_subpixels` does at load,
    leaving the grid as it is) and warns once, naming each snip where the fit's
    `nsubpixel` leaves a line below `nsubsigma_min`, with the fit's `nsubpixel` and
    what is asked (`examples/lsf_spline`: 6, and 21 asked for), and saying that a fit
    started from this best fit with its sub-pixels worked out again (`nsubpixel=`
    removed from the `.mod.out`'s data lines) uses them. Not with `nsubpixel_renew`,
    whose grid follows the fit.
  - **Lines that ask for more than `nsubpixel_max`.** At load, where `nsubpixel_max`
    caps a snip below what its lines ask for at `nsubsigma_min` (ALIS caps them in
    silence today), ALIS warns once, naming each such snip with what is asked and
    `nsubpixel_max`, and saying that raising `run nsubpixel_max` gives them.
  - Both go through `msgs.warn`, so that a run from the dashboard keeps them in
    `run.json` (Task 13.10). `CHANGELOG.md` (Added).
- **Check.** Made into tests first, in a new `tests/test_subpixel_warnings.py`
  (`unit`, synthetic spectra and `logmsgs`):
  - a model whose line, at its best-fit values, is left below `nsubsigma_min` on its
    fit's grid: one warning naming the snip and both numbers (before, none); none
    where it is not, none with `nsubpixel_renew`, and none where `nsubsigma_min` is
    lowered below what the grid gives;
  - lines capped by `nsubpixel_max` below `nsubsigma_min`: one warning naming the
    snip, what is asked and `nsubpixel_max` (before, none); none where the cap leaves
    them at `nsubsigma_min` or more;
  - no warning when a snip is on many sub-pixels (`run nsubpixel_min 200`; before,
    `warn_subpix`'s);
  - `examples/lsf_spline` fitted (the `fast` test of 14.3) warns of its best fit's
    21 against its fit's 6;
  - the harness's cases on which each warning fires, found with a scratch script over
    their models and references, are listed in the log. A warning that would fire on
    most of them is brought to RJC as a query before it is kept (the work pauses).
    The harness passes, its references unchanged.

**14.5 — The dashboard and a data line's recorded sub-pixels (the key item (a);
`lsfknots.py`, `preflight.py`, `results.py`, `qt/knots.py`, `qt/fit.py`; S17, S20,
D142; Q13.8; Q14.4 (iii), Q14.7).** *Q14.4 (iii) (a) and Q14.7's responses.*
- **What.**
  - **Kept as written.** An `nsubpixel=` on a project's data line (from a `.mod.out`
    opened with Open a fit (.mod)…, or typed in the `.mod` panel) is kept by every
    edit of the line, by Export… and by the bundle, as `bufferpix` is. The dashboard
    never adds it: Continue from best fit copies the best-fit values alone, as now,
    so that no pin stays in a model the user goes on editing.
  - **The knots dialog** (`lsfknots.subpixels`) draws the kernel on a snip's recorded
    `nsubpixel` where its line gives one, as ALIS evaluates it.
  - **The pre-flight check** notes each snip whose line records too few sub-pixels
    (14.3's case: "O I 1302.17 is worked out on 6 sub-pixels per pixel, from
    nsubpixel=6 on its data line; its lines ask for at least 21."), with a fix, Work
    them out again, that removes the keyword, one step. It is a note: Run stays
    enabled.
  - **Warnings and options to resolve…** (Task 13.10) lists each of ALIS's sub-pixel
    warnings of the run (14.3, 14.4), each with what resolves it (Q14.4 (ii)'s
    response):
    - a line that records too few: "Work them out again, on the pre-flight check,
      removes nsubpixel= from these data lines.";
    - a best fit that asks for more than its fit used: "Continue from best fit…, then
      Run: the next fit works its sub-pixels out from this best fit.";
    - lines capped by `nsubpixel_max`: "Raise nsubpixel_max (All settings…) to N.".
- **Check.** New tests:
  - in `test_dashboard_lsfknots.py`: the kernel of a snip whose line gives
    `nsubpixel=6` is drawn on 6, and without it on what ALIS works out (before, on
    what ALIS works out in both);
  - in `test_dashboard_preflight.py`: the note for a line that records too few, none
    for one that records enough; its fix is one step, which ALIS reads, and one undo
    gives back the text byte for byte;
  - in `test_dashboard_results.py`: each of the three warnings, kept in `run.json`,
    listed with its suggestion (before, "ALIS warned: …" with none); Continue from
    best fit of a run whose `.mod.out` records sub-pixels changes values alone (the
    data lines byte for byte); the gate masks a hidden value in a warning's text;
  - in `test_dashboard_runs_process.py` (`fast`): a run of `examples/lsf_spline`
    through the dashboard's process keeps the best fit's sub-pixel warning in
    `run.json`;
  - in `test_dashboard_datasets.py`, `test_dashboard_export.py` and `test_bundle.py`:
    the keyword kept through a change of the line's resolution, a snip cut again,
    Export… and a bundle;
  - in the `gui` tests of the Fit tab: the note and its fix in the window's
    pre-flight check, and the dialog's rows with their suggestions; no design
    reference (D44).

**14.6 — ALIS's default `nsubpixel_min` of 10, and what it moves (the key item (b),
`deferred_work.md` §7.10; `alis/config.py`; Q12.9, Q12.11, Q12.13; Q14.3 (a)).**
- **What.**
  - **The default.** `nsubpixel_min` (`nsubmin` until 14.2) 10 in `config.py`;
    `doc/ALIS_workflow.md`'s table, `doc/tex_files/description.tex`; `CHANGELOG.md`
    (Changed: every model is worked out on at least 10 sub-pixels per pixel unless it
    sets `run nsubpixel_min`; a fit of broad lines takes up to twice as long; a
    `.mod.out` written since 14.3 keeps its own sub-pixels).
  - **The dashboard** follows the default with no change of code. spix (one sub-pixel
    at `nsubpixel_min`) halves for a model that sets none, so Delta Function's knots
    are twice as close and the knots' rules follow (Q12.11's response). Knots set
    before keep their places and still pass the rules, since spix only shrinks.
  - **What moves, measured.** The whole harness is run with the new default, all
    three batches (`fast`, `medium` and `slow`, about an hour on RJC's machine), its
    `test_bundle` mode included, from the repository, before any reference changes.
    For each case that fails: χ² before and after, the largest change of a parameter
    in units of its 1σ error (and which), the errors that change most, the largest
    change of the model column in units of the error, and the runtime before and
    after (a case may change batch). How often 14.4's warnings fire with the new
    default goes in the log beside 14.4's count.
  - **Each case that moves, checked** (`check-fit`, `convergence-check`): its old best
    fit evaluated with `nsubpixel_min` 10 (as mode (b) evaluates it), against its new
    fit. Where the new fit's χ² is not below that evaluation's, or a parameter moves
    by more than 1σ, the case is refitted from several starts (`convergence-check`),
    to tell one minimum that moved from two minima. A case whose new fit lands in a
    worse minimum than its old best fit gives is put to RJC as a query, with both
    fits, before its reference is replaced (the work pauses).
  - Every result goes in the log, and a table for the review (14.8). No reference
    changes in this task.
- **Check.**
  - in `test_config.py` (`unit`): the default is 10 (before, 5); `run_alis
    --list-settings` gives 10;
  - the unit and dashboard tests that assumed 5 change with it, each named in the log
    (`test_lsfspline.py`'s sub-pixels asked for, `test_dashboard_lsfknots.py`'s spix
    and Delta Function, and any other); every other `unit`, dashboard and `gui` test
    passes unchanged, the dashboard's fits (`fast`) within their tolerances;
  - the harness's failures are exactly the cases listed in the log, each with its
    figures: Stage 12's 27 failures of the `fast` batch, `examples/lsf_spline`'s,
    and any case of the `medium` and `slow` batches, each named with its reason.

**14.7 — The references, regenerated once, with their sub-pixels (the key item;
`examples/*/`, `context/fitting_examples/*/`, `tests/test_regression.py`,
`tests/manifest.py`; Q14.3 (a)).**
- **What.**
  - **Each case of 14.6's list**, as RJC's responses to any query of 14.6 have it: its
    `.mod.out.reference` from a fit made as the harness makes it (`alisrun.run_alis`
    on a staged copy, the case's own `.mod` unchanged), its `reference_fits/`
    goldens, its covariance golden where it has one, and its runtime in the
    manifest. `examples/generate`'s golden data are generated again.
  - **With their sub-pixels.** Each regenerated `.mod.out.reference` carries
    `nsubpixel=` on every data line (14.3), so that its evaluation at the best fit
    uses its fit's sub-pixels.
  - **`examples/lsf_spline`** leaves `FIXEDPARAM_EXCLUDE`, and its evaluation at the
    best fit is back in the gate (§7.10's "To finish"). `brokenpowerlaw` and `tophat`
    stay out of it, with their reasons (§2.4 waits, Q14.1).
  - A case that changes batch moves with its runtime; `FIXEDPARAM_BATCH_OVERRIDE` and
    `MINIMISATION_MACHINE_DEPENDENT` keep their cases. The cases not on the list keep
    their references as they are.
  - The log lists every regenerated case with its χ² before and after.
- **Check.**
  - the whole harness passes on all three batches, its `test_bundle` mode included,
    from the repository: each case of 14.6's list against its new references, every
    other unchanged, `examples/lsf_spline` evaluated at its best fit (before, left
    out);
  - in `test_writer_round_trip.py` (`unit`): every regenerated `.mod.out.reference`
    carries `nsubpixel=` on each data line, and every reference reads back;
  - `git diff --stat` lists the references of 14.6's cases alone (with
    `examples/generate`'s golden), and the harness's two lists;
  - the blind cases' new references keep their blinded lines as a blind run writes
    them, and no log or page shows a value of them;
  - CI's `examples` job passes on both platforms within its time (the shipped
    examples are slower with `nsubpixel_min` 10). `test_gpu_regression.py` compares
    the GPU with these references, and runs only where there is a CUDA device: RJC
    runs `pytest --run-gpu -m gpu` there.

**14.8 — The review.**
- **What.** A review page built by `doc/dashboard/stage14/`, with the earlier pages'
  stylesheet:
  - **the names** (14.2): the convention, each old name and its new one, and All
    settings… with them (a screenshot);
  - **the sub-pixels a `.mod.out` is evaluated on:** `examples/lsf_spline`'s model
    from its `.mod.out` against its fit, per pixel in units of the error, before and
    after 14.3; a data line of a `.mod.out` with its `nsubpixel=`;
  - **the dashboard's side** (screenshots at 1440×900, synthetic scenes): the
    pre-flight note and its fix, and Warnings and options to resolve… with ALIS's
    sub-pixel warnings and their suggestions (14.3–14.5); Set LSFSpline knots…'s
    Delta Function with spix at `nsubpixel_min` 10 (14.6);
  - **`nsubpixel_min` 10:** the table of every case that moved (χ² before and after,
    the largest change in σ, the runtime before and after), and, for each unblinded
    case that moved by more than 1σ or whose χ² changed by more than 1%, its fit
    before and after, drawn (the data, both models, the residuals): the check by eye.
    The blind cases are in the table alone;
  - **also done:** 14.1, with its tests named.

  It is published as a private page, and RJC's comments are applied.
- **Check.** The page builds from what 14.6 and 14.7 kept of the harness's outputs,
  and from the `gui` tests' scenes (synthetic projects). Every case of 14.6's list is
  in its table; every task a user sees has its picture, or its test named. RJC's
  comments are applied and logged.

**14.9 — Close the stage.**
- **The documents.** `CHANGELOG.md` (14.1–14.4 and 14.6); `doc/ALIS_workflow.md` (the
  names and their convention, the data-line keywords, the default, and §4.3's "Since
  Stage 14"); `doc/tex_files/description.tex` and `newfunctions.tex`; `tests/README.md`
  (the Stage 14 row, and the references regenerated with `nsubpixel_min` 10); the
  `gui-component` skill's "Stage 14 additions", if the dashboard's side taught anything.
  Record the decisions in `ALIS_v2_dashboard_prompts.md` (D147 onwards, from Q14.1–Q14.9
  and the review), among them the naming convention, which the documentation and later
  code follow (Q14.6's response), saying which earlier decisions they refine: F9, Q12.9,
  Q12.11 and Q12.13, Q13.7, Q13.8, D140, D142 and D145, S17 and S20. Update
  `deferred_work.md`: §7.10 done (priority 24 so), the four items of Q14.1 (§2.1, §2.4,
  §5, §7.7) offered to Stage 15, and anything this stage leaves undone added. Update
  `dashboard_stage0.md`: Stage 14 done; the documentation next (Documentation Stages
  1–5, `documentation_stage0.md`, whose queries DQ.1–DQ.7 wait for RJC); then Stage 15.
  Bring `documentation_stage0.md` up to date: its list of settings (`nsubsigma`,
  `nsubsigma_min`, `nsubpixel_min` 10, `nsubpixel_max`, `nsubpixel_renew`; no
  `warn_subpix`), the data line's `nsubpixel`, and the convention, so that the
  documentation starts from this stage's ALIS.
- **Found and not done.** What was found during the stage and not built goes to Stage
  15's "Carried over from Stage 14", with a query, as Stage 13's close did.
- **What follows.** `dashboard_stage15.md`, drafted at Prompt 1 for the stage after
  the documentation, brought up to date: the items carried over, any Feedback point
  that arrived and was not taken, the state of the deferred items it offers, and this
  close's counts.
- **CI** (RJC, Prompt 2). When Stage 13's branch was pushed, CI's `gui` job on Ubuntu
  stopped with a segmentation fault in pytest-qt's processing of events, as
  `test_dashboard_qt_regions.py::test_the_view_stays_where_the_user_put_it[1]` was set
  up, its `[3]` having passed (`context/misc/GitHub_gui_Ubuntu-latest_py313.txt`);
  on `alis_v2`, every job passed but `examples`. Both are looked at once this
  stage's branch is pushed: if either fails again, its cause is found and fixed (or
  put to RJC as a query) before the stage closes.
- **Check.**
  - Every task is logged as done with its check passing: the item carried over in
    14.1 (Q14.2, Q14.5); the names in 14.2 (Q14.4 (i), Q14.6, Q14.8, Q14.9); the key
    item in 14.3–14.7 (§7.10 (a) and (b), Q14.3 (a), Q14.4, Q14.7); the four related
    items not taken (Q14.1).
  - Q14.1–Q14.9, and any query raised while the tasks were built, are answered in the
    Queries section.
  - The review of 14.8 is applied.
  - The `unit`, `gui` and `fast` batches pass, also under CI's pytest. Their counts go
    in the log, against Stage 13's close: 2356 unit tests passed with 119 skipped
    (2355 and 120 under pytest 9.2.0), 340 `gui` tests, and 155 `fast` tests, the
    regression harness and its `test_bundle` mode included. The `medium` and `slow`
    batches pass too, with their counts.
  - CI passes on both platforms, every job: the `gui` job within its 90 minutes and
    with no crash, and the `examples` job.
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44), and pre-commit passes on every
    changed file.
  - No old name of the sub-pixels is left (14.2's `grep`), and ALIS outside
    `alis/dashboard/` changes only as Tasks 14.1–14.4 and 14.6 say: `git diff --stat
    f9f4539 -- alis ':!alis/dashboard'` lists `alis/config.py`, `load.py`, `main.py`,
    `model_eval.py`, `minimise.py`, `gpu_dispatch.py`, `save.py` and
    `alis/functions/` (14.1's three loaders, and every function's `sampling=`),
    against Stage 13's close; the references changed are
    those of 14.6's list, each logged with its χ² before and after, and every
    reference's settings lines (14.2).
  - This stage's tests are synthetic or use ALIS's shipped examples, and need none of
    RJC's files; RJC's context fits are refitted and checked by the harness, from the
    repository, on RJC's machine.

## Skills to use for this stage

- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch, the harness included.
- `check-fit` and `convergence-check`: each fit whose reference moves.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `gui-component` and `gui-dev`: any change the dashboard shows.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/deferred_work.md` §7.10, §2.1, §2.4, §5 and §7.7.
- `claude_prompts/dashboard_stage12.md` (Q12.9–Q12.13, Task 12.9) and its log: the
  `fast` batch with `nsubmin` 10.
- `claude_prompts/dashboard_stage13.md` (Q13.7, Task 13.2) and its log: the
  sub-pixels of `examples/lsf_spline`, pinned 5 to 21.
- `tests/README.md` (the harness's modes and batches), `tests/test_regression.py`
  (`FIXEDPARAM_EXCLUDE`), `tests/alisrun.py`.
- `alis/load.py` (the sub-pixels, `bufferpix`), `alis/save.py` (the `.mod.out`'s data
  lines), `alis/config.py` (`nsubmin`, `nsubmax`, `nsubpix`).

## Queries

*Raised by Claude: Q14.1–Q14.3 with this draft (2026-11-24), answered by RJC and applied
at Prompt 1 (2026-11-24); Q14.4 and Q14.5 at Prompt 1, answered and applied at Prompt 2
(2026-11-24); Q14.6 and Q14.7 at Prompt 2, from the responses to Q14.4, answered and
applied at Prompt 3 (2026-11-24); Q14.8 and Q14.9 at Prompt 3, from the responses to
Q14.6 and Q14.7, answered and applied at Prompt 3, run again (2026-11-25); Q14.10 at
Prompt 4 (2026-11-25), during Task 14.6. Each gives Claude's lean. A question that comes up while a task is
built pauses the work; it is asked here, and its answer is logged here.*

**Q14.1 — Which related items join this stage.** For each of the four above: (a) it
joins, after the key item; (b) it waits in `deferred_work.md`.

My lean: (a) for 1 (`brokenpowerlaw`), since it is the other `.mod.out` that does not
reproduce its fit and its fix moves references as the key item does; (b) for 2, a
series of its own; (b) for 3, until there are data in frequency; and 4 as RJC
prefers, since the data are his.

**Response:** None of these will join this stage.

*Applied (Prompt 1, 2026-11-24): none joins. Each waits in `deferred_work.md`, and
the four are offered to Stage 15 (`dashboard_stage15.md`, Q15.2).
`brokenpowerlaw`'s references are regenerated in Task 14.7 if `nsubmin` 10 moves
them; it stays out of the evaluation at the best fit (§2.4).*

**Q14.2 — The item carried over from Stage 13** (`lsffile`'s `index` between data
lines). (a) It joins, a small fix in ALIS's loader with its test; (b) it waits in
`deferred_work.md`.

My lean: (a): it is in the loader this stage changes.

**Response:** It will join this stage.

*Applied (Prompt 1): Task 14.1, first, so that the harness after it is the stage's
baseline. Reading its code found the same fault in `apod` and `lsf` (Q14.5); the
task is written with the lean of Q14.5.*

**Q14.3 — The order of the key item.** (a) Record the sub-pixels in the `.mod.out`
first, then raise `nsubmin` to 10, so that each reference is regenerated once and
carries its sub-pixels; (b) the other way round; (c) only one of the two.

My lean: (a).

**Response:** I agree with option (a), so let's implement it like that if that is your preference.

*Applied (Prompt 1; numbers since Prompt 2): the sub-pixels are recorded first (Task
14.3 in ALIS, 14.5 in the dashboard), then `nsubmin` 10 is measured on every batch
with no reference changed (14.6), then each reference that moves is regenerated
once, carrying its sub-pixels (14.7). Stage 12 measured the `fast` batch alone; 14.6
runs the `medium` and `slow` batches too, so that no case is missed. How the keyword
is named and used is Q14.4.*

**Q14.4 — The sub-pixels a data line records (Tasks 14.3 and 14.5, 14.2 and 14.3 at
Prompt 1; Prompt 1).** ALIS works out each snip's sub-pixels when it loads a model,
from the starting values. Task 14.3 records them in the `.mod.out`, as a data-line
keyword, and uses them when given. Three points:
- *(i) The keyword's name.* A data line already takes `nsubpix=N`, which is not the
  sub-pixels: it is the oversampling per standard deviation (`run nsubpix`, 5) from
  which each line works out what it asks for. The new keyword gives the snip's
  sub-pixels per pixel, the number ALIS's log calls its "sub-pixellation scale".
  (a) `subpixels=6`; (b) `subpix=6`; (c) `nexbin=6`, ALIS's own name for it inside
  `load.py`.
- *(ii) A line that records fewer sub-pixels than its lines now ask for.* The keyword
  must win, or `examples/lsf_spline`'s `.mod.out` does not reproduce its fit, which is
  the key item. But a `.mod.out` is also where the next fit starts: a narrow line
  added to it, or a width that shrank during the fit (`lsf_spline`: 6 recorded, 21
  asked for at its best fit), is worked out on the recorded sub-pixels until the
  keyword is removed.
  - **(a) ALIS warns** once at load, naming each such snip with both numbers, and
    saying that removing `subpixels=` from its data line works them out again.
  - **(b) A line in ALIS's log alone** (information), with no warning.
  - **(c) The larger of the two is used.** The next fit is then never under-sampled,
    but `lsf_spline`'s `.mod.out` no longer reproduces its fit.
- *(iii) The dashboard.*
  - **(a)** A `subpixels=` on a project's data line (from a `.mod.out` opened with Open
    a fit (.mod)…, or typed) is kept as written. The pre-flight check notes a snip
    whose line records fewer than its lines ask for, with a one-step fix that removes
    the keyword (Run stays enabled); ALIS's warning of (ii) is listed by Warnings and
    options to resolve… with that suggestion. Continue from best fit copies the
    best-fit values alone, as now, so the dashboard never adds the keyword.
  - **(b)** As (a), and Continue from best fit also writes each snip's recorded
    sub-pixels from the run's `.mod.out` onto the project's data lines, so that the
    continued run starts on its fit's sub-pixels, as `run_alis` on the `.mod.out`
    does.
  - **(c)** As (a), without the pre-flight note.

My lean: (i) (a), the word of ALIS's log, and far from `nsubpix`; (ii) (a): in both
cases the warning is true (`lsf_spline`'s fit did end on sub-pixels coarser than its
best fit asks for, and its next fit should know); (iii) (a): the user sees the
recorded sub-pixels before Run, and one added by Continue from best fit would stay
through every later edit of the model, where the user would not see it.

**Response:** For point (i), I agree with option (a), although this is also an opportunity to change the other name as well to something more descriptive so that it is clear that these are different parameters and what each parameter does. Please offer some suggestions that you think would be most reasonable, given the functionality of the code. For point (ii), I agree with option (a). If there is a chance that some model profiles may not be properly sampled at subpixel scales, we can provide this in the list of "Warnings and options to resolve..." dialog. For point (iii), I agree with option (a).

*Applied (Prompt 2, 2026-11-24): Task 14.3 (14.2 at Prompt 1) takes (i) (a),
`subpixels=N`, and (ii) (a), ALIS's warning; Task 14.5 (14.3 then) takes (iii) (a),
and lists the warning in Warnings and options to resolve… with what resolves it.
For `nsubpix`'s new name, Q14.6 offers three names (Task 14.2, written with its
lean). For the other ways a profile may not be properly sampled on its sub-pixels,
Q14.7 names the two ALIS can see (Task 14.4, written with its lean).*

**Q14.5 — `apod` and `lsf` keep their keywords between data lines too (Task 14.1;
found while reading the code for Q14.2, Prompt 1).** `lsffile`'s fault is in two
other resolution functions written the same way: `apod` (a scratch script: a second
line naming no `kind:` takes the first line's `uniform`, where a line alone stops
with "kind must be set") and `lsf` (`name`, `grating`, `life_position`, `cen_wave` and
`slit`, by reading its code). In all three, a `blindrange` given on two lines of one
function would gather both lines' values. `base.Base` and `lsfspline` start each
line from a copy of their defaults. No model in `examples/` or `context/` meets it.
- **(a) Task 14.1 fixes all three**, the same few lines in each, with their tests.
- **(b) `lsffile` alone**; `apod` and `lsf` are noted in `deferred_work.md`.

My lean: (a): one fault, in the same lines of three functions, with one test each.

**Response:** I agree, let's do option (a).

*Applied (Prompt 2): Task 14.1 fixes `lsffile`, `apod` and `lsf`, as written.*

**Q14.6 — A clearer name for `nsubpix` (Task 14.2; Q14.4 (i)'s response, Prompt 2).**
`nsubpix` (`run nsubpix`, 5, and a data line's `nsubpix=N` for one file) is the
number of sub-pixels across one standard deviation of a profile. Each line asks for
`nsubpix` · pixel / σ sub-pixels per pixel (`vfwhm`, σ = FWHM / 2√(2 ln 2); `voigt`
and `splineabs`, σ = b / √2), and a snip takes the most that any line asks for,
between `nsubmin` and `nsubmax`. `subpixels=N` (Q14.4 (i)) gives the result, per
pixel. 34 of RJC's `context/` models set `run nsubpix`; no shipped example sets it,
and no model gives it on a data line. Two points:
- *(i) The new name* (for the setting and the data-line keyword alike):
  - **(a) `subpix_per_sigma`:** says what it counts and over what, beside
    `subpixels` (per pixel), with the `subpix` of `warn_subpix` and `renew_subpix`;
  - **(b) `nsubsigma`:** short, in the family of `nsubmin` and `nsubmax`, but as
    terse as the name it replaces;
  - **(c) `sigma_sampling`:** reads well, but does not say that it counts
    sub-pixels.

  `nsubmin` and `nsubmax` keep their names: they already count sub-pixels per pixel,
  as `subpixels=` does, and 12 of RJC's models set each.
- *(ii) The old name.*
  - **(a) Still read**, as the new one, with a line in ALIS's log, in a model's
    settings, on a data line and in `--set`; a model giving both is refused. RJC's 34
    models run unchanged, and the documentation names the old name once, as old.
  - **(b) Read with a warning** that it is old: every run of those 34 models warns,
    and the dashboard lists it in Warnings and options to resolve….
  - **(c) Refused**, with a message naming the new one: those 34 models must be
    edited before they run.

My lean: (i) (a), the name that says what the number is; (ii) (a): the meaning and
the default do not change, so nothing is gained by making old models warn or stop.

**Response:** For point (i), I propose a change everywhere (without backwards compatibility) to `nsubpixel` when it subsamples the pixel, and `nsubsigma` when it subsamples the standard deviation. For point (ii), I agree with option (a). Then `nsubmin` would actually become `nsubpixel_min` and `nsubmax` would become `nsubpixel_max`. This would make it clear that these parameters are related to the sub-pixel sampling of the pixel, while `nsubsigma` would be related to the sub-pixel sampling of the standard deviation. We could also introduce parameters `nsubsigma_min` and `nsubsigma_max` so there are similar min/max parameters for the standard deviation sampling. This would make the naming consistent and clear. If there are any inconsistencies anywhere in the code related to this naming convention, let's fix it now, and we will carry this naming convention forward into the documentation and future code. Please note that we do not need to focus on backwards compatibility for this change, as we are making a clear distinction between the two types of sub-pixel sampling.

*Applied (Prompt 3, 2026-11-24): Task 14.2 is widened to every name of the
sub-pixels, by the response's convention, with no old name read; Task 14.3's keyword
is `nsubpixel=N`. What the response leaves open is Q14.8 (the names of `renew_subpix`
and `warn_subpix`; its (ii) (a) beside "no backwards compatibility"; RJC's files
that use the old names; the names inside the code) and Q14.9 (`nsubsigma_min` and
`nsubsigma_max`). Tasks 14.2–14.9 are written with their leans.*

**Q14.7 — Profiles that may be under-sampled (Task 14.4; Q14.4 (ii)'s response,
Prompt 2).** Q14.4 (ii)'s response asks that a chance of profiles not properly
sampled on their sub-pixels be listed in Warnings and options to resolve…. Task
14.3's warning (a line that records fewer sub-pixels than its lines ask for) is
listed there (Task 14.5). ALIS can see two other ways, and says nothing of either
today:
- *A best fit that asks for more sub-pixels than its fit used.* A fit's sub-pixels
  are set from its starting values, and a width that shrinks during the fit leaves
  its profile on fewer than it asks for at the end (`examples/lsf_spline`: 6 used,
  21 asked for at its best fit). Seen at the end of the fit; the suggestion,
  Continue from best fit…, then Run (the next fit works its sub-pixels out from this
  best fit).
- *Lines that ask for more than `nsubmax`.* `load_subpixels` caps them at
  `nsubmax` (21). Seen at load; the suggestion, raise `nsubmax`.

Two ways:
- **(a) ALIS warns of both** (Task 14.4), and the dialog lists each with its
  suggestion (Task 14.5). Task 14.4 counts the harness's cases on which each would
  fire, and a warning that would fire on most of them comes back to RJC before it is
  kept.
- **(b) Task 14.3's warning alone**, as Prompt 1 wrote it.

My lean: (a). Both are the case the response names, the first is exactly
`lsf_spline`'s, and a warning listed with what resolves it costs the user one look.

**Response:** Generally speaking, it is more important to warn when the model profile is not sampled enough. Having too many subpixels is not considered a problem, but having too few subpixels can lead to inaccurate results. Therefore, I think the best approach is to only warn when the subpixels should be increased.

*Applied (Prompt 3): Task 14.4 keeps its two warnings, both of too few sub-pixels,
with `nsubsigma_min` as their level (the lean of Q14.9). `warn_subpix`, ALIS's
warning of too many (a snip on more than 100 per pixel), is removed in Task 14.2
(Q14.8 (i)).*

**Q14.8 — The names, the old ones, and RJC's files (Task 14.2; Q14.6's response,
Prompt 3).** Four points:
- *(i) The names.* By the response's convention (`nsubpixel…` per pixel, `nsubsigma…`
  across a profile's standard deviation):

  | Now | Becomes | What it counts |
  |---|---|---|
  | `run nsubpix` (5); data line `nsubpix=N` | `run nsubsigma`; `nsubsigma=N` | sub-pixels across one σ of the narrowest profile: the target from which each line asks |
  | `run nsubmin` (5) | `run nsubpixel_min` | the fewest sub-pixels per pixel |
  | `run nsubmax` (21) | `run nsubpixel_max` | the most sub-pixels per pixel |
  | (Q14.4's `subpixels=N`) | data line `nsubpixel=N` | a snip's sub-pixels per pixel, recorded in the `.mod.out` |
  | `run renew_subpix` | `run nsubpixel_renew` | (a switch) each snip's `nsubpixel` worked out again at every evaluation |
  | `run warn_subpix` (100) | removed | it warned of a snip on too many sub-pixels |

  The response did not name the last two. For `renew_subpix`: (a) `nsubpixel_renew`,
  so that every setting of the sub-pixels per pixel sorts together; (b)
  `renew_nsubpixel`. For `warn_subpix`: (a) removed, since too many sub-pixels are
  not a problem (Q14.7's response); (b) `nsubpixel_warn`, kept.
- *(ii) The old names.* The response chose (a) of Q14.6 (ii), the old names still
  read, and also asked for no backwards compatibility; I read the second as what you
  want. (a) No old name is read: each is refused with a message that names its new
  one (a small table, kept for that message alone); (b) refused as any unknown
  setting is ("--set does not recognise the setting …"), with no table; (c) still
  read, as Q14.6 (ii) (a) offered.
- *(iii) RJC's files.* With (a) or (b), 66 of RJC's models and references under
  `context/` (34 models set `run nsubpix`, 12 `run nsubmin`, 12 `run nsubmax`; 13
  references repeat them) and 4 project bundles under `context/feedback01/` (MWC137's:
  `run nsubpix 150`, `run nsubmin 100` and `run nsubmax 200` in their models and
  stored runs) stop reading until they are rewritten. They are not in git, so nothing
  would restore them. (a) Claude rewrites them with a script (the settings lines and
  data-line keywords alone; a bundle extracted and packed again with
  `alis/bundle.py`), after archiving the originals in
  `context/misc/stage14_old_names.tar.gz`, and logs each change; (b) Claude writes
  the script, and RJC runs it; (c) they are left as they are.
- *(iv) The names inside the code* (the response: "any inconsistencies anywhere in
  the code"). (a) The names a user meets: the settings, the keywords, ALIS's
  messages, the documentation and the dashboard; (b) also the internal names of these
  numbers: `nexbins` (each snip's sub-pixels per pixel) → `nsubpixel`;
  `datopt['nsubpix']` (`nsubsigma` times the pixel's width) →
  `datopt['nsubsigma_width']`; and the `nexbin=` argument of every function's
  `set_vars` (that width, with the bin type) → `sampling=`, in 26 modules of
  `alis/functions/` (`base.py` among them) and `doc/tex_files/newfunctions.tex`,
  about 220 lines in all, mechanical, and checked by `test_function_interface.py`
  and the harness.

My lean: (i) as the table, (a) for both; (ii) (a): the message costs one table and
saves a user a search; (iii) (a), with the archive; (iv) (b), so that the
documentation's chapter on writing a function describes `sampling=`, not a name that
no longer means anything.

**Response:** I agree with the lean of (i) as the table, (a) for both; For point (ii) I agree with (a); For point (iii), I agree with option (a), to archive the data for now. Please make sure the .tar.gz file is located in the `context/` folder so that I know where to find them if needed; and for point (iv), I agree with option (b).

*Applied (Prompt 3, run again, 2026-11-25): Task 14.2 as written with the leans: the
names of the table, with `renew_subpix` → `nsubpixel_renew` and `warn_subpix`
removed; each old name refused with a message naming its new one (and, in the
dashboard, a problem with a fix that renames it); RJC's files rewritten by Claude's
script after their originals are archived, with their paths, in
`context/stage14_old_names.tar.gz`, directly in `context/` (not `context/misc/`);
and the internal names renamed too.*

**Q14.9 — `nsubsigma_min` and `nsubsigma_max` (Tasks 14.2–14.4; Q14.6's response,
Prompt 3).** The response suggests them, beside `nsubpixel_min` and `nsubpixel_max`.
`nsubsigma` is not worked out, as `nsubpixel` is: it is the target from which each
line asks for its sub-pixels per pixel (`nsubsigma` · pixel / σ). A snip takes the
most that any line asks for, between `nsubpixel_min` and `nsubpixel_max`, so its
narrowest profile gets `nsubsigma` across its σ, and broader ones more. A minimum and
a maximum of it can therefore mean:
- **`nsubsigma_min`:** the fewest sub-pixels across a profile's σ that ALIS accepts
  without a warning. It gives Tasks 14.3's and 14.4's warnings their level (Q14.7's
  response): a snip warns where one of its lines is left below it, on a recorded
  `nsubpixel`, at a best fit narrower than its start, or under `nsubpixel_max`. With
  a default of 5, `nsubsigma`'s, ALIS warns whenever a profile falls short of the
  target; a user who accepts less lowers it, and one who wants more raises
  `nsubsigma` while keeping 5 as the level. A model whose `nsubsigma` is below its
  `nsubsigma_min` is refused, naming both.
- **`nsubsigma_max`:** I found no role. The grid is set by the narrowest profile, so
  the broader profiles of a snip get more across their σ, and too many sub-pixels are
  not a problem (Q14.7's response).

Three ways: (a) `nsubsigma_min` as above, 5 by default, and no `nsubsigma_max`; (b)
neither, the warnings' level being `nsubsigma` itself; (c) both, with a role for
`nsubsigma_max` that RJC names.

My lean: (a). The target and the level of the warning are two settings, so a user can
ask for 10 across σ and accept 5.

**Response:** I agree with option (a).

*Applied (Prompt 3, run again, 2026-11-25): Tasks 14.2–14.4 as written: `run
nsubsigma_min`, 5 by default, the level of every warning of too few sub-pixels, with a
model whose `nsubsigma` is below it refused; no `nsubsigma_max`.*

**Q14.10 — Three fits that land in a worse minimum with `nsubpixel_min` 10 (Task
14.6, Prompt 4; Task 14.7 waits for the response).** The harness was run with the new
default on all three batches (the log has every figure): 35 of its 105 tests fail,
22 fits among them (18 unblinded, 4 blind; DH_orders' fit, run with `--run-slow`,
among them), every one of them checked. For 14 of the 18 unblinded fits the minimum
moved and stayed one minimum: each new fit's χ² is below its old best fit evaluated
with 10. For three it is not. For each, a fit with
10 started from its old best fit stays there, and the fit the harness makes from the
case's own `.mod` lands in a worse minimum:

| Case | Reference χ² (5) | Old best fit at 10 | New fit from its `.mod` | From the old best fit, at 10 | Largest change of the new fit |
|---|---|---|---|---|---|
| DH/HS0105p1619 | 2086.26 | 2086.24 | 2136.99 (+50.7) | 2086.24 (0.04σ) | 3.4σ |
| VMP_DLA/J0903p2628 | 5152.08 | 5151.99 | 5173.55 (+21.6) | 5151.99 (0.05σ) | 700σ |
| DH/J1419p0829 | 7648.99 | 7649.15 | 7653.81 (+4.7) | 7648.38 (0.72σ) | 44σ |
| DH/Q1243p307 newstart76 | 12116.06 | 12116.20 | 12116.31 (+0.1) | 12116.19 (0.04σ) | 0.07σ |

(Q1243p307's is the same minimum: its new fit stops 0.1 short of it, within its
tolerance, and it is regenerated with the others. HS0105p1619, J0903p2628 and
Q1243p307 are the cases of Q0.17, whose fits were sensitive to their start then too.)
The blind fits are reported by χ² and the size of their largest change alone, since
their old best fits cannot be evaluated: J0814p5029 −0.05% (5.8σ), J1358p0349 −0.60%
(16.6σ), J1558m0031 −0.06% (17.3σ), each lower than its reference, and Q0913p072
+0.004% (0.16σ).

For the three, what becomes of their references (Task 14.7):
- (a) Each starts from its old best fit: its `.mod`'s starting values replaced by
  those of its `.mod.out.reference` (RJC's files archived first, in `context/`), so
  that the harness's fit lands in the better minimum, and its reference is
  regenerated from that fit (χ² 2086.24, 5151.99, 7648.38).
- (b) The `.mod` files as they are, and each reference regenerated from the fit its
  `.mod` gives with 10 (the worse minimum), as Q0.17 regenerated the fits that moved
  then. `tests/README.md` and the review page say which minimum it is.
- (c) The three keep `nsubpixel_min` 5 (`run nsubpixel_min 5` in their `.mod` and
  reference), so that their fits and references stay as they are, and they do not
  test the new default.

My lean: (a). A reference 50 χ² from the better minimum (HS0105p1619) would put a
worse fit on the review page than the one RJC published, and the harness would then
guard it. Started from its old best fit, each case still tests a fit with 10 (the
parameters move by up to 0.72σ), and its reference is the better minimum. The
default of 10 itself is not in question: these are 3 of 42 cases, all known to be
sensitive to their start.

**Response:** Starting the fits from their old best fits does test the values of the model created with a set of parameters. So, it is fine to start with that now. Might it also be a good idea to start from the best fit parameters and perturb those values by the covariance matrix, to see if that allows the code to find the same minimum as before? Overall, this disagreement seems to show that these models depend on the subpixellation, which I wasn't expecting given that we have only doubled the number of subpixels. I propose that we attempt to perturb the model parameters from the best-fit values by the covariance matrix, and check to see if the parameters converge to their best-fit values. If that doesn't work, let's resort to option (a), and just use the previous best-fit parameters as the new starting parameters. Related to this decision, I think this highlights an even bigger issue that we will need to address. What techniques can we use to ensure that we are not converging to a local minimum? Is there a way that we can test the sensitivity of the chi-squared minimisation to the subpixellation factor. These questions should be addressed as a separate stage of development independent of the dashboard development. Please log this as deferred work, and make suggestions for techniques that we should attempt to implement in the future.

*Applied (Prompt 5, 2026-11-25). For each of the three, five starts were drawn about
its old best fit from its covariance, as Random restarts… draws them
(`restarts.draw_starts`: `sim perturb` after `sim beginfrom`, clamped to ±`sim
maxperturb`), and fitted with `nsubpixel_min` 10. A start "returns" when its fit's
χ² is within 0.1 of the fit from the old best fit and every parameter is within 1σ of
the old best fit; where all five return, the first start becomes the case's starting
values, and otherwise its old best fit's own values do (option (a)), as the response
asks:*
- *VMP_DLA/J0903p2628: all five return (χ² 5151.99–5152.02, within 0.14σ): its `.mod`
  starts from the first draw.*
- *DH/HS0105p1619: four return (2086.24–2086.27, within 0.08σ); the fifth stops at
  2087.86, 1.0σ from it: its `.mod` starts from its old best fit.*
- *DH/J1419p0829: the five end in different minima (7646.04, 7647.87, 7648.48,
  7650.29, 7650.30; parameters up to 14.6σ from the old best fit), two of them lower
  than the fit from its old best fit (7648.38): its `.mod` starts from its old best
  fit. Its lower minima are evidence for the deferred item, and are reported to RJC.*

*Fitted as the harness fits them, the three new `.mod` files reach 2086.24, 7648.38
and 5151.99 with 10. RJC's originals are archived in
`context/stage14_starts_before_14_6.tar.gz`; only starting values changed (the model
block, and each data line's resolution, a free parameter). The deferred work is
`deferred_work.md` §1.5 (priority 25), with the techniques suggested.*

**Q14.11 — ALIS's warnings on one line, or with their line breaks (Task 14.8's
review; Tasks 14.4 and 14.5).** Warnings and options to resolve… shows each of ALIS's
warnings on one line, its whitespace collapsed (Stage 13's dialog, D142), so a
sub-pixel warning's three parts run on as one sentence: "… spectrum 0, segment 0:
nsubpixel=3 on its data line; its lines ask for 8 Remove nsubpixel= from a data line
to work its sub-pixels out again." (the review page's Warnings section, its second
screenshot).
- (a) Keep ALIS's line breaks in the dialog: the header, each snip, and the advice
  on lines of their own (the row grows with them).
- (b) One line per warning, its parts joined by "; " so that they read apart.
- (c) As it is.

*Lean (a): the snips are a list, and a warning that names several (the cap names
each snip and what it asks for) is read most easily as one. The dialog's rows
already wrap; the status bar and the run history's tooltip keep one line.*

**Response:** I think the first point is that there are some fullstops missing at the ends of some sentences. Removing the line breaks is fine, because it compacts the text, however, we need to include the fullstops in this case. If we remove or retain the line breaks, the warning will be better with fullstops. I think the dialog will be fine with the line breaks removed (as is currently shown).

*Applied (Prompt 6, 2026-11-25), with RJC's comment on the review page to the same
effect: the dialog keeps one line per warning (option (c)), and each part of ALIS's
three sub-pixel warnings is now a sentence (`alis/load.py`): the header ends with a
full stop, not " -", and each snip's line begins with a capital and ends with a full
stop ("Spectrum 0, segment 0: nsubpixel=3 on its data line; its lines ask for 8.").
On one line: "… across its standard deviation. Spectrum 0, segment 0: …; its lines
ask for 8. Remove nsubpixel= from a data line …". No full stop is added at a line
break by the dashboard, so none lands mid-sentence; the warnings' other lines already
ended with one. Test: `test_each_part_of_a_warning_is_a_sentence` (three cases).*

**Q14.12 — Is `nsubsigma_min` needed? (RJC's comment on the review page; Q14.9;
Tasks 14.2–14.4).** The comment: `nsubsigma` is set and never changes, and each
snip's `nsubpixel` is worked out from it, so is the `nsubsigma_min` check needed?

What the code does. `nsubsigma` is the target: at load, from the starting values,
each snip takes the most sub-pixels per pixel that any of its lines asks for
(`nsubsigma` · pixel / σ), raised to `nsubpixel_min` and capped at `nsubpixel_max`.
`nsubsigma` never changes, but what a profile actually gets across its σ can fall
below it, in three ways, and these are where ALIS warns (Q14.7's response), at the
level `nsubsigma_min`:
1. `nsubpixel_max` caps a snip below what its lines ask for (at load).
2. A data line records `nsubpixel=N` (a `.mod.out`'s, Task 14.3, or one written by
   hand), fewer than its lines ask for (at load, and the pre-flight check's note).
3. A profile narrows during the fit: the sub-pixels were set from the starting values
   and are kept (unless `nsubpixel_renew`), so the best fit can ask for more (at the
   end of the fit).

Besides, a model whose `nsubsigma` (or a data line's `nsubsigma=`) is below
`nsubsigma_min` is refused. So the warnings are needed; `nsubsigma_min` only sets
their level. At the defaults (both 5) it is the same as warning below `nsubsigma`.
They differ only when one is changed: with `nsubsigma` 10 and `nsubsigma_min` 5, ALIS
aims for 10 across σ and warns below 5 (without `nsubsigma_min`, below 10); with
`nsubsigma` 3, ALIS refuses the model (without it, it fits on 3 and warns below 3).

- (a) Keep `nsubsigma_min` as Q14.9 (a) has it.
- (b) Remove it: the warnings' level is the snip's `nsubsigma` (its data line's,
  where it gives one), and no model is refused for it; `run nsubsigma_min` is refused
  as an old name would be, saying it was removed. It changes `config.py`, `load.py`
  (the three warnings, and the refusal), the pre-flight note (`preflight.py`), their
  tests, `CHANGELOG.md`, `doc/ALIS_workflow.md` and the review page; no reference and
  no fit changes (warnings do not change a fit), and `nsubpixel_min` 10 keeps its
  floor.

*Lean (b). The case for (a), aiming higher than one accepts, is a refinement nobody
has asked for, and it adds a setting whose default is another's; the refusal guards
only a user who lowers `nsubsigma` on purpose. With (b), one setting says how many
sub-pixels a profile should have across its σ, and ALIS warns wherever a profile has
fewer.*

**Response:** I agree with the lean (option b).

*Applied (Prompt 7, 2026-11-25): `run nsubsigma_min` is removed (`alis/config.py`),
and so is the refusal of a model whose `nsubsigma` was below it (`load.py`). The three
sub-pixel warnings, and the pre-flight check's note (`preflight.py`), take each snip's
own `nsubsigma` (the run's, or its data line's `nsubsigma=`) as their level: the
header names it ("nsubsigma (5)"), or "its snip's nsubsigma" where the snips named
differ. `subpixels_asked` loses its `level` argument. `run nsubsigma_min` in a model is
refused as a removed name ("'run nsubsigma_min' was removed: ALIS warns where a
profile has fewer sub-pixels than nsubsigma across its σ. Remove the line."), and the
.mod panel's right-click removes the line. No model or reference named it; no fit
changes. Tests: the removed name among the old ones, `test_a_lower_nsubsigma_asks_for_fewer`
(before, refused), `test_a_data_line_s_own_nsubsigma_is_its_warning_s_level`, and the
cases that lowered `nsubsigma_min` rewritten. `CHANGELOG.md` and
`doc/ALIS_workflow.md` follow; the review page is draft 3.*

**Q14.13 — `brokenpowerlaw` and `tophat` pass the evaluation at the best fit (Task
14.9; Q14.1, `deferred_work.md` §2.4).** Both are left out of the harness's mode (b)
(`FIXEDPARAM_EXCLUDE`): `brokenpowerlaw` for "reference point (χ² 374.98 vs recorded
338.15); Q0.12", whose suspected cause is that ALIS writes the minimiser's last,
rejected, trial step when a fit stops on `atol` (§2.4, priority 3, and Stage 15's
proposed key item); `tophat` for "sharp edge (~14× error at edge pixel); Q0.15". Found
while closing the stage, with the harness's own functions (`make_fixedparam_mod`,
`compare_mod_out`, `compare_fit_dat`):

| Reference | Evaluated on | χ² recorded | χ² evaluated | Mode (b) |
|---|---|---|---|---|
| `brokenpowerlaw`, before 14.7 (no `nsubpixel=`) | `nsubpixel_min` 5 | 3503.0708 | 3555.2921 | fails (χ², model) |
| `brokenpowerlaw`, before 14.7 | `nsubpixel_min` 10 | 3503.0708 | 3501.9938 | fails (model) |
| `brokenpowerlaw`, regenerated (14.7) | its `nsubpixel=10` | 3499.947053 | 3499.947056 | **passes** |
| `tophat`, before 14.7 | `nsubpixel_min` 5 | 393.6293 | 393.2619 | fails (model) |
| `tophat`, before 14.7 | `nsubpixel_min` 10 | 393.6293 | 394.1329 | fails (χ²) |
| `tophat`, regenerated (14.7) | its `nsubpixel=10` | 394.132678 | 394.132678 | **passes** |

So a reference evaluated on its fit's sub-pixels gives its recorded χ² to 1 part in
10⁹, and its model: for `brokenpowerlaw`, the parameters written are those of the χ²
written, which a rejected trial step would not give. Its old mismatch is the
sub-pixels (§7.10), not the minimiser; `tophat`'s edge, likewise.

- (a) Both back in mode (b)'s gate now (`FIXEDPARAM_EXCLUDE` emptied, two `fast`
  tests more); §2.4 closed as explained by §7.10, the suspected cause not needed;
  Stage 15's candidate 1 dropped, and Q15.1's lean moving to the simulations
  (candidate 2).
- (b) Both back in the gate now, but §2.4 kept open for Stage 15 to confirm on the
  minimiser's stopping paths that no fit reports a rejected step.
- (c) As they are, for Stage 15.

*Lean (b). The evidence closes these two cases, so the gate should have them; but
one example is not every fit that stops on `atol`, and the check §2.4 proposes (every
case's written parameters give its written χ², which mode (b) now does for every case
it runs) is cheap to keep as Stage 15's first task, smaller than its key item.*

**Response:** I agree with lean (b). If there are problems in the future with `brokenpowerlaw` and `tophat`, we can exclude them. They are not very important for QSO absorption line analyses that are  typically undertaken with ALIS.

*Applied (Prompt 8, 2026-11-25): `FIXEDPARAM_EXCLUDE` is empty (`tests/test_regression.py`,
its comment giving each case's history), and `test_fixed_param` runs on
`brokenpowerlaw` and `tophat` (`fast`, both passing). `deferred_work.md` §2.4 kept open
for Stage 15 to confirm the minimiser's stopping paths at `atol`, with the evidence and
RJC's note that either case can be excluded again; priority 3 and Stage 15's candidate
1 and Q15.1's lean follow (the simulations now the lean, with §2.4 as a small first
task). `tests/README.md` and D149 say that no case is excluded.*

**Q14.14 — CI's `examples` job: `examples/lsf_spline` under numpy 2.5.5 (Task 14.9,
CI; Q0.16).** On the stage's push (506939b, run 45703269345) every job passed but
`examples`, which failed on Ubuntu and macOS, as it has since Stage 13's push
(f9f4539; Stage 12's, 7343048, passed). The gui job's segfault of Stage 13 did not
recur. CI's log needs admin rights to download, so the failure was reproduced here:
CI installs the newest numpy, 2.5.5 (released 2026-11-15), and the references were
made with 2.4.3. On a clean copy of 506939b with numpy 2.5.5, CI's `pytest -m
examples` gives 2 failed, 115 passed: `test_minimisation` and `test_bundle` of
`examples/lsf_spline`, each with one comparison, "convolution errors line 1, token 1:
1-sigma error 6127.9272242 vs 0.0"; with numpy 2.4.3, all pass.

The token is `lsfspline`'s smoothing FWHM, which the data do not constrain: the
example's data were made with 4.0 km/s, but the fit trades the smoothing against the
knots' heights, and any smoothing narrower than a sub-pixel (0.14 km/s) leaves the
model as it is. With numpy 2.4.3 the fit takes it to its bound (2.9×10⁻⁷ km/s), where
it leaves the covariance and its error is 0; with 2.5.5 it stops at 0.082 km/s, with
an error of 6128. Every other value, error, χ² and the model agree. The harness asks
a zero reference error to stay zero (Q0.16: it confirms the degeneracy), so the
example fails wherever rounding moves it off the bound.

- **(a) Refine Q0.16's rule:** a zero reference error is matched by a zero error, or
  by one at least as large as the parameter's value (undetermined either way); a
  small error, which would mean the parameter became constrained, still fails.
  Prototyped on the clean copy: `lsf_spline` passes under both numpy versions. The
  example and its references stay as they are; unit tests of the comparator's rule
  are added (none exist).
- **(b) Make the example well posed:** its smoothing fixed at the 4.0 km/s its data
  were made with, its reference regenerated. It then no longer shows a best fit that
  asks for more sub-pixels than its start, which `test_subpixels_fit.py` and the
  review use it for, so that test needs another example.
- **(c) Skip `lsf_spline`'s refit and bundle on CI**, on both platforms (as
  `MINIMISATION_MACHINE_DEPENDENT` does on Linux); its evaluation at the best fit
  still runs.

*Lean (a). The failure is a parameter the data cannot determine, reported two ways;
the rule can say so without loosening anything that can be measured, and it covers
the next example with a parameter at a bound. The Ubuntu job, which skips the
machine-dependent refits, ran 3 min 39 s against macOS's 8 min 55 s; its log, if RJC
saves it to `context/misc/` as before, would confirm that it is the same two tests.*

**Response:** OK, let's go with option (a) for now. This test does seem to have issues though, since the model cannot be reproduced with the data. It suggests that the test data should be generated with higher S/N.

*Applied (2026-11-26):* (a). `tests/compare.py` matches a zero 1σ error with a zero
error or with one at least as large as the parameter's value, either way round (so
that a reference made where rounding stops short of the bound passes where it does
not); a small error still fails, and so does a zero error whose value is blind. Its
rule is unit-tested in `tests/test_compare.py` (9 tests, 4 of which fail on the
comparator as it was). D153 records it. The example's data at a higher S/N are
`deferred_work.md` §7.13 (priority 27), offered to Stage 15 as its item 16, with
Q15.8. CI's `examples` batch under numpy 2.5.5 and 2.4.3 is in the log.

## Prompts
1. Please read my responses to your queries, and the work of Stages 0–13 (their documents and logs). If anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (14.1, 14.2, etc.). The final Task (currently labelled 14.N) should be to close the stage, and should include the checks that will verify that the stage is complete. Note that we will implement the documentation after this stage is complete, but please write a stage 15 design documentation, which will be implemented after the documentation is up to date.
2. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (14.1, 14.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion. I also wanted to note that the CI tests failed when I pushed the branch to GitHub. I have pasted in the output in the following context file: `context/misc/GitHub_gui_Ubuntu-latest_py313.txt`. I also note that the CI tests passed when I pushed the `alis_v2` branch (except for the examples tests), so it might be OK. Let's check the CI tests at the end of this stage, and if they fail, we can investigate the cause.
3. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (14.1, 14.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
4. Please execute the tasks in order. There's no need to pause between tasks, unless you have queries that require my thoughts and responses. Please check your context limit between tasks. If you reach 75% context, please pause and ask for my guidance on the next steps. If you have queries during development, please pause the development and log the queries and responses in the Queries section.