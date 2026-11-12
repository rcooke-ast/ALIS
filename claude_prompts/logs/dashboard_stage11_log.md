# Dashboard Stage 11 log

Feedback from users, after Stage 10's list closed (Q10.3 (a)). The plan is in
`claude_prompts/dashboard_stage11.md`.

### 2026-11-10 (Prompt 1: Feedback point 1, the GPU, and queries Q11.3–Q11.4)

**Read.**
- This stage's document: Feedback point 1 (**(new)**: keep the large arrays on the
  GPU for the whole fit, send only the parameters, return only χ², explored on its
  own branch and design document), the three items carried over from Stage 10, and
  RJC's responses to Q11.1 (items 1 and 2 join; item 3 declined, with a label scheme
  described) and Q11.2 ((a), a closed list).
- For point 1, the GPU's history: `refactor_code_stage4.md` in full (Tasks 4.0–4.9,
  Q4.1–Q4.11, RJC's Prompts 11 and 14, which asked much the same) and its log's
  "Pre-4.7" and "Pre-4.9" analyses; the GPU mentions of every other stage document
  and log (refactor Stages 1, 3, 3.5, 5, 6; dashboard Stages 5–7; `deferred_work.md`;
  `documentation_stage0.md`; `context.md`).
- For the stage's conventions: the end of Stage 10's log, its document's outline and
  queries, and the stage templates (`dashboard_stageX.md`, `refactor_code_stageX.md`).
- The code: `alis/gpu_dispatch.py` in full, the minimiser's outline
  (`minimise.py`: MPFIT's Levenberg–Marquardt, QR of the m × n Jacobian at
  `qrfac(fjac, pivot=1)`), the `run` settings in `config.py`, and, for item 3,
  `edit.suggest_label`, `edit.new_component` and Q2.3 of `dashboard_stage2.md`.
- `git diff` since refactor Stage 4: the GPU modules (`gpu.py`, `gpu_dispatch.py`,
  `shared_arrays.py`, `voigt_gpu.py`) are unchanged; `minimise.py`, `model_eval.py`,
  `voigt.py`, `vfwhm.py` and `legendre.py` differ by 84 lines. Stage 4's measurements
  therefore still describe the GPU path, though the series would measure again.

**Found** (from Stage 4's measurements on DH_orders; nothing was run).
- Part of point 1 is in place: the sub-pixels are fixed unless `run renew_subpix
  True` (default False), and each snip's sub-pixel grid stays on the device, uploaded
  once per iteration (28.8 MB) and reused within it.
- The bytes moved per evaluation are small (parameters up, one sub-pixel array per
  snip down). What costs is the 351 synchronising round trips per evaluation (CUDA
  driver ~13% of a column, ~1.27 ms per group) and the host's work between them
  (`vfwhm` 20% and `legendre` 9.3% of a GPU Jacobian, and the loop over snips).
  GPU-4 61.9 s against CPU-12 about 32–37 s per Jacobian.
- Point 1's design, the whole evaluation kept on the device, removes both, so it is
  worth exploring; the transfers are not the reason. The Voigt arithmetic of a base
  evaluation is about 0.2 s of kernel time on one RTX 2080 Ti (119 million
  pixel-components at 15.6 ms per 10 million).
- χ² alone cannot be all that returns: the Levenberg–Marquardt minimiser needs the
  residuals (75,164 for DH_orders, 0.6 MB) per evaluation and the Jacobian (75,164 ×
  413, 248 MB dense) per iteration, tens of ms over PCIe. Returning less means another
  minimiser or the normal equations.
- It is larger than a stage and outside `alis/dashboard/`: a device version of every
  function a fit uses, an FFT for the convolution (numba.cuda has none; Q4.1), and
  checks of its own (the regression harness, the CPU bitwise gate, the `gpu` batch).
- Item 3: the response describes `nx#`, `bx#`, `tx#`, `zx#` with `#` the component's
  number in its system, as the current scheme. The code offers `zx#`, `bx#`, `tx#`
  with the first number unused in the model, and nothing for a column density (Q2.3).
  Item 3 proposed the system's letter (`ba3`), not the ion; the Si I / S II clash RJC
  names is Q2.3's, for column densities.

**Written** in `dashboard_stage11.md`:
- Q11.3, where point 1 is explored: (a) in this stage, or (b) a series of its own,
  `gpu_stage0.md` as its design document (measurements, a prototype and the figure it
  must beat, the queries on what returns, the FFT and the hardware), on branches RJC
  makes. Lean (b): the work may end in "no", it has its own tests, skills and history,
  and it is refactor Stage 4's Task 4.9 (still awaiting RJC) grown to the whole
  evaluation.
- Q11.4, the suggested labels: (a) as now; (b) as now, plus `nx#` for a column
  density with the same running number; (c) item 3 as written. Lean (b).
- A note under Q11.1's response (items 1 and 2 join; item 3 waits on Q11.4), and the
  header's and the Queries' notes.

No code was changed. Waiting on RJC's responses to Q11.3 and Q11.4. Under Q11.3 (b),
the next step is `gpu_stage0.md`; this stage's Design and tasks are written when its
Feedback list closes (Q11.2 (a)).

### 2026-11-12 (Prompt 2: the first set's design and Tasks 11.1–11.7, Q11.5–Q11.6)

**Read.**
- This stage's document afresh: Feedback point 2, new since Prompt 1 (**(change)**:
  opening a `.mod` whose snips need more than one spectrum), and RJC's responses
  to Q11.3 ((b), a series of its own, its design document a task of this stage) and
  Q11.4 (`LS#`: `n`, `b`, `t` or `z`, then the system's letter or `ly`, then the
  component's number, "check that this is consistent with the current code and
  tests, and if not, let's implement it"). Prompt 2 adds that the GPU waits as
  deferred work, and that a question during a task pauses it and is asked and
  answered in the Queries section.
- `ALIS_v2_dashboard_prompts.md`'s decisions D1–D119 in full; Q2.3 of
  `dashboard_stage2.md` (column densities left unlabelled); `deferred_work.md`.
- The code each point touches: `loadfit.py` (`mod_hints`, `match_row`, `compare`,
  `positions`, `load_mod_plan`, `ModPlan.message`), `qt/window.py` (`open_fit`,
  `open_fit_new`, `open_fit_with_spectrum`, `_fit_plan`, the toolbar's icon texts),
  `qt/dialogs.py` (`NewProjectDialog`), `project.py` (`_build_rows`,
  `_infer_reference`, `_build_systems`, `component_label`, `tie_component`,
  `tie_isotope`), `datasets.py` (`add_file_step`, `set_reference_step`), `edit.py`
  (`check_label`, `new_component`, `suggest_label`), `components.py`
  (`set_temperature`'s scheme, `_tied_first`, `interlopers`), `qt/components.py`
  (`bind_action`, `_with_label`), `qt/data.py` and `qt/regions.py` (`_draw_ids`),
  `lines.py` (`line_ids`, `LABELLED`), and the tests of Open a fit
  (`test_dashboard_loadfit.py`, `test_dashboard_qt_regions.py`).

**Reproduced** (scratch scripts and scratch `gui` tests in the session's scratchpad;
no code changed).
- *Point 2.* A synthetic fit of two spectra (the metal example and a copy with its
  flux 1% higher, one snip from each). `load_mod_plan` loads it into a project of
  both. Through the window, Open with its spectrum (recommended)… loads it when
  both are added in the New project dialog; with one added, the refusal lists the
  other's snip and says "add it on the Data tab", where no project exists yet. Each
  spectrum gets the `.mod`'s first `vfwhm` (4.0 km/s for the second, whose own is
  6.0), so the preview notes the difference and Reset FWHM would return to 4.0.
- *Point 2, a bug.* With a `.mod` that frees the first spectrum's shift and leaves
  the second's out, the first stays the reference with a free shift: a new
  project's rows are already marked, so `_infer_reference` is never applied.
- *Point 2, RJC's files.* The three J0035m0918 `.mod`s do not load against the
  Stage 10 project's three spectra: their snips are on other pixels, cut from other
  reductions (`generate_other_models.mod` cannot be packed: `generate data`).
- *Item 2.* Every tool button of the window on a project with every tab shown: nine
  drop their action's "…" (Add ion, Add system, Add interloper, Identify a feature,
  Open, Commit run, Continue from best fit, Convergence check, Review clipped
  pixels); Export… and Random restarts…, set by hand, keep it.

**Found in the code.**
- *Item 1.* Line IDs are drawn on the Data tab ("Transitions of z", "All lines",
  primary system only) and the Regions tab (D114's four choices, `ui/view.json`);
  the Components tab draws ticks, the Fit tab no line IDs.
- *Item 3 (Q11.4).* + Add component, D20's and D22's fixes and D93's labels write
  `za1`, `ba1`, `ta1` (the system's letter, `ly` for the forest; the first number
  none of the three uses). `suggest_label` uses `x`; `components._tied_first` uses
  `x` for a component of no system; a column density has no suggestion (Q2.3). A
  component's number is its place by redshift, recomputed at every read, while
  labels are never renamed (Q2.3), so "#" matches only when the label is made. `na2`
  cannot label a column density, since each ion of a component has its own.

**Written.**
- `dashboard_stage11.md`:
  - the Design's "11.1–11.7 — The first set": what was found, and what stays;
  - Tasks 11.1–11.7, each with its check: the reference from a loaded `.mod` (bug),
    a `.mod`'s snips against several spectra (`fit_datasets`, `match_spectrum`, the
    refusal grouped by dataset), Open a fit with several spectra (the question, the
    dialog's list of datasets, each spectrum's FWHM, Create waiting, Back), every
    button's "…" (bug), one choice of line IDs for the Data and Regions tabs, labels
    from the component's scheme, and the review page;
  - Q11.5 (a column density's label: lean `na2si2`; the number: lean one number per
    component, no renaming) and Q11.6 (a missing spectrum: lean every dataset needs
    its spectrum);
  - notes under the responses to Q11.3 (the GPU in `deferred_work.md`; its design
    document written when the series starts) and Q11.4 (what the check found); the
    header's and the Queries' notes, the latter with Prompt 2's rule for questions
    during a task.
- `deferred_work.md`: §1.4, the GPU evaluation kept on the device (what, why
  deferred, where, what Q11.3 found, what `gpu_stage0.md` will hold), and priority 13
  (medium); the low priorities renumbered 14–22.

No code was changed. Waiting on RJC's responses to Q11.5 and Q11.6; Tasks 11.3 and
11.6 are written with the leans and are revised where a response departs from them.

### 2026-11-12 (Prompt 3: RJC's responses to Q11.5 and Q11.6; Tasks 11.1–11.10)

- **The responses.**
  - Q11.5: (a) for a column density (`na2si2`). For the number, a label keeps the
    number it was made with (the order components are given labels), and a button
    relabels a system's components by redshift, after a warning that it cannot be
    undone and changes every component's labels.
  - Q11.6: neither way. The user adds every spectrum they have; a dialog assigns the
    snips, with the datasets listed at the top, the chosen one's snips on the left
    and the snips of no dataset on the right, moved by drag and drop, then Confirm.
    The reason: two datasets may share their wavelengths.
- **Checked against the code.** A label rename already reaches every use
  (`edit.rename_label`: the lines, those commented out, the commands and the links),
  so renumbering is one change of several renames. A row with snips and no source is
  already read (`Project._build_rows`), and snips left out of the rows' specids are
  grouped as an imported fit's, so a snip of no spectrum needs no new kind of row.
  SNIP passes over rows without a spectrum (`snips.covering`). Matching needs a
  snip's flux and error as well as its wavelengths, so two spectra on the same
  wavelengths are told apart unless their values are the same too.
- **Revised.** The Design ("Revised (Prompt 3)"; what stays as it is), and the
  Tasks, now 11.1–11.10, numbered in order, each with what it builds or fixes and
  its check:
  - 11.1 the reference from a loaded `.mod` (bug);
  - 11.2 which spectrum holds each snip, the best guess, and snips of no spectrum
    loaded without a source;
  - 11.3 the dialog that assigns the snips (Q11.6's layout);
  - 11.4 opening and loading a fit of several spectra (the question, the New project
    dialog's FWHM per spectrum, the dialog in every way of loading, Back);
  - 11.5 every button's "…" (bug); 11.6 one choice of line IDs;
  - 11.7 labels from the component's scheme (a label's number as it was made; a
    column density with its ion);
  - 11.8 Renumber labels… by redshift;
  - 11.9 the review; 11.10 the close (the documents, D120 onwards, Stage 12's
    document, and the checks against Stage 10's close: 2211/120 unit, 279 `gui`,
    149 `fast`).
- **Raised** Q11.7 (Renumber labels…: Undo takes it back, lean (a), against the
  response's "cannot be undone"; the cards keep their order by redshift, lean (a))
  and Q11.8 (the assignment dialog: when it opens, lean when there is a choice; a
  drop on a spectrum that does not hold the snip, lean refused; where it is used,
  lean every way a fit is loaded). Notes under the responses to Q11.5 and Q11.6
  say what was applied; the stale task numbers in Q11.3–Q11.6's notes were brought
  up to date.

No code was changed. Waiting on RJC's responses to Q11.7 and Q11.8; Tasks 11.2–11.4
and 11.8 are written with the leans, and are revised where a response departs.

### 2026-11-12 (Prompt 3, run again: RJC's responses to Q11.7 and Q11.8)

- **The responses.** Q11.7: (a) for both, as leant (Undo takes a renumbering back;
  the cards keep their order by redshift). Q11.8: (b) for when the dialog opens,
  every time a fit is loaded with spectra, so that snips that fit no dataset, or come
  from a dataset not loaded, are seen and assigned; (a) for the other two (a drop on
  a spectrum that does not hold the snip is refused; the dialog in every way a fit is
  loaded). The response adds a user's report: "the dashboard only looking in the
  first dataset for snips".
- **Checked against the code.** No code looks for a snip's spectrum in the first row
  alone: `loadfit.match_row` tries every row, for Open a fit, Load a fit and Load
  snips…. Its refusal gives only the reason of the row nearest to matching, the
  first when they tie. Rerun on RJC's J0035m0918 `.mod` with its three spectra: the
  O I 1039 snip (3470 Å) is refused as "outside the range of J0035m0918_ESPRESSO"
  (3783–7883 Å), though J0035m0918.dat covers 3128–5966 Å; neither it nor UVES2010
  (3760–9466 Å) is named. The other `rows[0]` uses in `alis/dashboard/` are a
  default view (the Components tab's first dataset, D61) and the first reference.
- **Revised.** The Design (the revised notes, what stays, the report); Tasks 11.2
  (a reason for every spectrum tried, in a `.mod`'s refusal and Load snips…'s), 11.3
  (each snip on the right gives every spectrum's reason), 11.4 (the dialog every time
  a fit is loaded with spectra; the existing tests answer it), 11.10 (Q11.1–Q11.9;
  Q7.4's rule among those refined). Notes under the responses to Q11.7 and Q11.8.
- **Raised** Q11.9: whether the user's report is the refusal that names one spectrum
  (lean (a); 11.2 and 11.3 are written with it), or something else, to be given as
  steps or files and made a task of its own.
- **The task list** is 11.1–11.10, numbered in order; each says what it builds or
  fixes, with its check, and 11.10 closes the stage with the checks that the stage is
  complete. Tasks 11.1–11.8 now follow every response; only Q11.9 is open.

No code was changed. Waiting on RJC's response to Q11.9; the tasks are then ready to
execute, 11.1 first.

### 2026-11-12 (Prompt 3, run a third time: RJC's response to Q11.9)

- **The response.** No snip should be refused. A snip that fits no dataset can be
  (1) assigned to a dataset, its data cut again where the pixels differ; (2) left
  out, the snips that are assigned still loaded; or (3) loaded and marked
  "unassigned", cut from no dataset, each becoming its own dataset. This replaces
  Q11.8's refused drop.
- **Checked against the code.** What the three choices need is there:
  `remove.plan_snip` takes a snip out with the lines only it uses (as CLEAR does),
  `snips.regions_of` gives a snip's fit range by wavelength, SNIP's cut gives a
  spectrum's pixels over a range, and a row with snips and no source is already read.
- **Revised.** The Design (point 2's notes; "No snip refused"); Tasks 11.2 (each
  snip held, cut again, unassigned or left out; every spectrum's reason; the
  preview names each fate), 11.3 (the Load tick on each unassigned snip; a drop on
  a spectrum that does not hold the snip cuts it again, refused only where the
  spectrum has too few usable pixels, D113), 11.4 (the three choices end to end; Load
  snips… opens the same dialog); 11.10 (Q11.1–Q11.10; D12 among those refined).
  Notes under the responses to Q11.8 and Q11.9.
- **Raised** Q11.10: unassigned snips as one dataset per dataset of the `.mod`
  (lean (a)) or one per snip (the response; RJC's J0035m0918 `.mod` would give 34
  rows); a Load tick per snip (lean) or one choice for all; cut again over the snip's
  own range (lean) or SNIP's window; Load snips… through the same dialog (lean) or
  as now.
- **The task list** is 11.1–11.10, numbered in order, each with what it builds or
  fixes and its check, 11.10 closing the stage. Tasks 11.2–11.4 are written with
  Q11.10's leans; every other task follows the responses.

No code was changed. Waiting on RJC's response to Q11.10.

### 2026-11-12 (Prompt 3, run a fourth time: RJC's response to Q11.10; the tasks ready)

- **The response.** (a) for all four parts. Part (i) on a condition: one unassigned
  dataset for each dataset of the `.mod`, if the dashboard can hold two unassigned
  snips that overlap in wavelength in one dataset. Added: a snip given to a dataset
  whose wavelengths, flux or error it does not share is cut again only after a
  warning; to keep it as it is, the user leaves it unassigned.
- **Checked against the code (part (i)'s condition).** The model and ALIS take two
  overlapping snips in one dataset, and the check for pixels fitted twice
  (`snips.shared_pixels`, D19) compares every pair of snips by pixel, whatever their
  dataset. But the Regions tab shows one snip of a transition per dataset:
  `snips.snips_of` keeps the first it meets (`setdefault`), and `snips.strips` gives
  one strip per row. So a second snip of the same transition in one dataset would be
  out of sight there. Found on the way: an imported fit (Open without a spectrum)
  already groups two such snips, two exposures that share a resolution label, into
  one inferred row, and the Regions tab shows only the first.
- **Revised.** Task 11.2 (two unassigned snips of one transition go to separate
  unassigned datasets; overlap at different transitions stays together; the warning
  before cutting again), 11.3 (the warning, with Cut again and Cancel, one for
  several snips dropped together), 11.10 (what was found and not done goes to Stage
  12's "Carried over from Stage 11", the imported fit's hidden snip among it); the
  Design (the responses, and "Overlapping snips in one dataset"); a note under the
  response.
- **The task list** is 11.1–11.10, numbered in order; each has what it builds or
  fixes and its check, and 11.10 closes the stage with the checks that it is
  complete. No query is open (Q11.1–Q11.10 answered), and every task follows the
  responses.

No code was changed. The tasks are ready to execute, 11.1 first.

### 2026-11-12 (Prompt 4: the Regions tab's bug in this stage; how unassigned snips are loaded)

- **Prompt 4.** Fix in this stage the bug found at Prompt 3 (an imported fit's second
  snip of a transition in one dataset, not shown on the Regions tab). Ask the user
  how unassigned snips are loaded: one "unassigned" dataset, or each snip its own.
  With one dataset, check for wavelength overlap; where snips overlap, warn and
  advise each snip its own dataset, and let the user choose.
- **Checked against the code.** The one-snip-per-dataset assumption is in
  `snips.snips_of` (`setdefault`), used by `snips.strips`, `linked_snips` (the
  linked regions), the edges' grips and the merge offers in other datasets, and in
  `qt/regions.py`, whose strips, selection and boxes are keyed by row id (about 40
  uses). A user who keeps overlapping unassigned snips in one dataset meets the same
  limit, so the fix is the Regions tab showing one strip per snip, not a split of
  the imported fit's rows.
- **Revised.**
  - New Task 11.2, the bug made into tests first: `snips_of` gives every snip of a
    transition in a row; one strip per snip, titled by its file; selection, the
    boxes and Adjust fit regions' one spectrum by snip; linked regions and grips
    reach every snip. The tasks after it are renumbered 11.3–11.11, and every task
    reference in the document with them.
  - 11.3: unassigned snips in one unassigned dataset or one for each snip, as the
    user chooses, with `unassigned_overlaps` (replacing Prompt 3's rule that put a
    second snip of a transition in another dataset); 11.4: the choice under the
    right-hand list; 11.5: the overlap warning at Confirm, Each snip its own dataset
    advised; 11.10: the review shows both; 11.11: the bug no longer carried to Stage
    12, and D36, D55 and D115 among the decisions refined.
  - The Design (Prompt 4's note), the header's note, a note under Q11.10's response.
- **Raised** Q11.11: whether "a single unassigned dataset" means one for each of the
  fit's datasets (lean, as Q11.10 (i)) or one for all; and where the choice starts
  (lean one unassigned dataset, with the warning where snips overlap).
- **The task list** is 11.1–11.11, numbered in order, each with what it builds or
  fixes and its check; 11.11 closes the stage. Tasks 11.3–11.5 are written with
  Q11.11's leans.

No code was changed. Waiting on RJC's response to Q11.11.

### 2026-11-12 (Prompt 5: RJC's response to Q11.11; the tasks ready)

- **The response.** (a) for both points of Q11.11: one unassigned dataset for each
  of the fit's datasets, and One unassigned dataset chosen to start with. RJC will
  test the design as built and propose changes if needed.
- **Applied.** Tasks 11.3–11.5 were written with these leans, and stand. A note
  under the response says that changes RJC proposes after testing are asked in the
  Queries section as they come, pausing the work (Prompt 2). The tasks' header note
  says the responses are all in.
- **The task list checked.** 11.1–11.11, numbered in order; each says what it builds
  or fixes and has its check; 11.11 closes the stage, with the checks that the stage
  is complete. Q11.1–Q11.11 are answered; no query is open.

No code was changed. The tasks are ready to execute, 11.1 first.
