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

### 2026-11-12 (Prompt 6: the tasks, in order)

**Task 11.1 — A loaded `.mod` decides the reference (point 2, bug).** Done.
- *Tests first* (`test_dashboard_loadfit.py`, on the metal example's spectrum and
  `noisy_copy`, a copy with other noise in its flux):
  - `test_a_mod_of_two_spectra_decides_the_reference`: a `.mod` that frees src's
    shift (`vshift(0.0vsa)`) and leaves other's unwritten. Before the fix src stayed
    the reference, with its free shift (checked with a scratch script, and the test
    failed at the missing note). Now other is the reference, src's shift reads free,
    the preview says "other becomes the reference: old.mod leaves its shift at 0.",
    ALIS reads the model, and one undo gives back the text, the files and
    `ui/project.json`.
  - `test_a_mod_that_shifts_every_spectrum_keeps_the_reference`: both shifted, src
    stays, with "No dataset of old.mod has its shift fixed at 0, so src stays the
    reference."; a `.mod` that leaves src's shift at 0 changes nothing and says
    nothing.
- *The fix* (`loadfit.py`): `_loaded_reference` takes the first row whose loaded
  snips have no shift or one fixed at 0 (`Project._infer_reference`'s rule, D14),
  and `load_mod_plan` sets the rows' `reference` flags in `ui/project.json` to it.
  A project of one row, a `.mod` that loads no snips, or one whose every dataset is
  shifted leave the flags as they were. `_reference_notes` writes the notes.
- *Shared test helpers*: `tests/dashboard_fits.py` (new) holds `noisy_copy`,
  `prepfit_snip` (moved from `test_dashboard_loadfit.py`), `write_mod` and
  `two_exposures`, for this stage's tests of fits of several spectra.

**Task 11.2 — Every snip of a transition on the Regions tab (bug).** Done.
- *Tests first*, on `two_exposures`: an imported fit of two O I 1302 snips cut from
  two noisy copies of the spectrum, one resolution label (one dataset, "vb"), the
  second with its own shift of +3 km/s.
  - `test_two_snips_of_one_transition_in_one_dataset` (`test_dashboard_snips.py`)
    failed at `snips_of`, which gave the first snip alone. Now: both snips in their
    row, in the model's order; `strips` gives one each; `linked_snips` gives both,
    and a linked step's name counts datasets ("in 1 dataset"); a region added on
    the first reaches the second, moved by its shift, within a pixel; ALIS reads the
    model and undo restores it; the edges' grip moves both high edges, each in its
    own frame.
  - `test_two_snips_of_one_transition_in_one_dataset`
    (`test_dashboard_qt_regions.py`): two strips titled "vb · exp1_16O_I_1302.2_
    reg.dat" and "vb · exp2_…", the plot titled "O I 1302.2 · vb · 2 snips"; a click
    on the second selects it (framed, bold), and the Continuum and Snip boxes are
    its own, with its own shift (3); the snips table has a row for each; a region
    drawn on the first with all spectra reaches the second, moved by its shift; one
    drawn on the second with one spectrum changes the second alone; the grip moves
    both edges, and one snip's own edge moves it alone.
- *The fix.*
  - `snips.py`: `snips_of` gives a list of snips for each row. `linked_snips`,
    `same_pairs`, `strips` (one entry per snip) and `edges_all_step` (each snip to
    the velocity in its own frame, by `snip_shift`) take each. `dataset_count`
    counts the datasets in a linked step's name and in the edges' step.
  - `qt/regions.py`: a strip's `key` is (row id, specid); the tab selects by
    `selected` (the row, as before) and `selected_snip`, kept in the view state as
    "snip". `select_row(row_id, specid=None)`, `_snip_of_row`, `draw`, `exclude`,
    `move_edges` and `move_all_edges` take the strip's snip. A dataset with two
    strips titles each "dataset · file", as do the Continuum and Snip boxes and the
    snips table; the plot's title counts snips and datasets; Share with… names the
    snip; a snip's velocity axis is in its own frame; another snip of the same
    transition is not drawn as "fitted by another snip". A strip keeps its zoom
    across SNIP, as before.
  - The one existing test that read `snips_of`'s result directly
    (`test_two_datasets_and_regions_copied_on_drawing`) takes the row's first snip;
    the other tests of the Regions tab pass unchanged (121 with the snips, export
    and shared-pixel tests).

**Task 11.3 — Where each snip goes: held, cut again, unassigned or left out (point
2).** Done.
- *Tests* (`test_dashboard_loadfit.py`, synthetic: the example's spectrum, copies with
  other noise, and `dashboard_fits.write_mod`):
  - `test_the_datasets_of_a_mod_of_two_spectra`: src's two snips (one label) and
    other's one, with FWHMs 7 and 5 km/s and other's shift free; `two_exposures`'
    two files of one label are one dataset.
  - `test_two_spectra_hold_their_own_snips`: same wavelengths, other fluxes: each
    holds its own and none of the other's, with the reasons.
  - `test_a_snip_held_by_two_spectra_is_guessed_by_its_dataset`: O I held by src
    and by "half" (src with other noise about Si II only) goes to half, which holds
    its dataset's Si II too, and is marked as held by src.
  - `test_a_snip_that_no_spectrum_holds_has_a_reason_for_each`: three reasons,
    joined by semicolons, in `match_row`'s refusal and in `Holders.why` (before:
    the nearest row's alone).
  - `test_a_snip_is_cut_again_from_the_spectrum_it_is_given`: Si II given to src
    takes src's pixels over its own range, its fit range, a file named as SNIP names
    its own (`src_Si_II_1304.4_reg.dat`), the data line pointing at it, and the
    model block as written; `test_a_fit_range_is_carried_over_by_wavelength` on a
    spectrum half a pixel along: a pixel is fitted when its whole step lies in an
    old fitted run (one fewer than before); `test_a_spectrum_with_too_few_usable_
    pixels_cannot_take_a_snip`: "broken has 0 pixels that can be fitted over its
    range (…); a snip needs at least 10."
  - `test_unassigned_snips_load_without_a_source[one|each]`: "other (unassigned)"
    with both, or one dataset per snip named after its file; no source, byte for
    byte, and SNIP (`covering`) passes over them.
  - `test_unassigned_snips_that_would_share_a_dataset_and_overlap`: two O I that
    overlap and a wide Si II that overlaps both are found (three pairs, with the
    overlap's range); two that meet at one pixel are not; leaving one out, or each
    its own dataset, changes the pairs.
  - `test_a_snip_left_out_takes_its_own_lines_with_it`: its data line and continuum
    go, the voigt lines keep specid 0, and the preview names the lines of old.mod.
  - Each fate: ALIS reads the model, and one undo restores every byte.
  - `test_a_mod_whose_snip_is_not_from_the_spectra_is_refused` became
    `…_loads_it_unassigned`: by the best guess, the snip is loaded in "src
    (unassigned)" with the reason in the plan.
- *The code* (`loadfit.py`): `sorting` (each spectrum sorted once; `positions` and
  `compare` take it), `reasons_text`, `FitDataset`, `Fit`, `read_fit`,
  `fit_datasets`, `Holders` and `snip_holders` (held, reasons, and the rows that
  cannot take a snip by D113's 10 pixels, counted by a running sum), `Assignment`
  (rows, load ticks, one or each) with `fate`, `best_guess`, `unassigned_groups`,
  `unassigned_overlaps`, `cut_again` (D97's step edges), and `load_mod_plan(project,
  modfile, spectra, assignment=None, fit=None, holders=None)` with the four fates
  and their notes. `remove.plan_snips` removes several snips at once.
  `Row.unassigned` is read from `ui/project.json`, and the Data tab's Contents says
  "unassigned: no source spectrum". `ModPlan.message` ends "Nothing was loaded."

**Task 11.4 — The dialog that assigns the snips (point 2).** Done.
- *The dialog* (`qt/dialogs.py`): `AssignDialog(fit, holders, guess, ask=)`, with
  `SnipList` (dragged snips carry their specids; a drop is a copy, the dialog fills
  both lists again) and `DatasetList` (the datasets at the top; snips dropped on one
  go to it). Left: the chosen dataset's snips, "file · its dataset", marked "also in
  …" or "cut again from …". Right: "Not from these spectra", each with Load, and
  a tooltip of every spectrum's reason (or "… holds it."). Under it One unassigned
  dataset (chosen) and Each snip its own dataset, greyed while none is ticked. → and
  ←, double-clicks, and drops move snips; a snip given to a spectrum that does not
  hold it asks first (one warning for several), and one that cannot be taken is
  refused on the dialog's line. Confirm returns `assignment()`.
- *Tests* (`test_dashboard_qt_loadfit.py`, new; four spectra: src, other, copy and
  broken, and a fit of O I from src, Si II from other, O I from a spectrum the
  project lacks): the best guess shown (the O I marked "also in copy, broken",
  broken differing above 1303 Å only); moved right and back by the buttons, by
  `dropEvent` with real `QDropEvent`s, and by double-clicks; the warning's text,
  Cancel, and Cut again for two at once; broken refusing Si II with its reason;
  a drop on a dataset of the top list; unticking (left out) and the choice greyed;
  Confirm and Cancel; no design reference.

**Task 11.5 — Opening and loading a fit of several spectra (point 2).** Done.
- *The window* (`qt/window.py`): the question counts the datasets ("Its 3 snips
  come from 3 datasets: add every spectrum you have in the next dialog"); the New
  project dialog (`fit=`) says how many datasets and which, and each spectrum added
  takes its dataset's FWHM (`loadfit.dataset_fwhm`) in place of the default;
  `assign_snips` opens the dialog every time a fit is loaded with spectra (Open with
  its spectrum, Open a fit into a project, Load a fit into this project), then warns
  when unassigned snips that would share a dataset overlap, naming each pair, with
  Each snip its own dataset advised (the default); a refusal after Create (a snip
  that is not a table of numbers) offers Back, which opens the New project dialog
  again with its name, folder, spectra, FWHMs and redshift (`restore`). Choose
  another spectrum… is gone.
- *Load snips…* (`qt/regions.py`, `loadfit.read_snip_files`, `FileFit`,
  `load_snips_plan(..., assignment=, files=, holders=)`): the same dialog; a file
  cut again or unassigned is added as Load snips… adds its snips (an unassigned
  dataset is added first, kept while it has no snips yet), one left out is not.
  *Decided while building:* a file that cannot be read, or whose name names no
  transition, still stops the whole load before the dialog (Q7.4), as a fit's
  snip that is not a table of numbers is refused after Create: such a file has no
  place to go. The check's "one failing file … goes to the right-hand list" is met
  for a file that matches no spectrum; the existing test's failing file names no
  transition, so it stays a refusal. RJC may wish otherwise; it is a small change.
- *Tests* (`test_dashboard_qt_loadfit.py`): the question of a fit of three; both
  spectra added (FWHMs 7 and 5, each snip under its spectrum, no FWHM note, other the
  reference); one added (the other's snip on the right, loaded unassigned; unticked,
  left out with its continuum; dragged onto src, cut again after the warning);
  overlapping unassigned snips warned, Each giving three datasets and One two, the
  Regions tab showing all three snips either way; no warning with Each chosen or
  without overlaps; Load a fit and Load snips… opening the dialog in a project of
  two spectra (a file of no spectrum loading in "far (unassigned)"); Back keeping
  what was entered; one spectrum holding every snip. In `test_dashboard_qt_
  regions.py`, the tests of Open a fit and Load snips… answer the dialog with
  Confirm (`confirm()`); `test_open_a_fit_of_another_spectrum_says_why` now finds
  each snip on the dialog's right with its reason, where it had been refused.

**Task 11.6 — Every button keeps its "…" (item 2, bug).** Done.
- *Test first*: `test_every_button_keeps_its_ellipsis` (`test_dashboard_qt_window.py`)
  on the metal example, every tab shown. Before the fix ten buttons dropped the
  "…", the nine actions the design names (Open…, ⌖ Identify a feature…, Add ion…,
  Add system…, Add interloper…, Commit run…, Continue from best fit…, Convergence
  check… and Review clipped pixels…), one of them on two buttons. Now none does;
  ?  Shortcuts keeps its own text, and the menus are as they were.
- *The fix* (`qt/window.py`): `keep_ellipsis(action)` sets the icon text of an action
  whose text ends with "…" to its text, where its icon text is still Qt's own (so
  ?  Shortcuts is kept). It runs as each window action is made, and for a tab's
  action after the tab has bound it (the Data tab gives Identify its "⌖"). The two
  lines that set Export… and Random restarts… by hand are gone.

**Task 11.7 — One choice of line IDs for the Data and Regions tabs (item 1).** Done.
- *Tests first* (`test_dashboard_qt_data.py`): `test_one_choice_of_line_ids_for_
  both_tabs`: the Data tab's menu (the four entries, after Transitions of z, greyed
  with it off); This transition with no transition shown yet draws the model's
  choice; Every line chosen on the Data tab is the Regions tab's, and the reverse;
  with Si II 1304 shown on the Regions tab, This transition labels it alone and
  counts the rest, and This ion labels Si II's lines; no step; kept and read back
  after Save and Open. `test_a_project_saved_before_keeps_the_regions_choice`: a
  `ui/view.json` written as Stage 10 wrote it (the choice under "regions") reads as
  the shared choice, on both tabs, and stays after the Regions tab redraws.
  `test_the_spectrum_shows_line_ids_and_counts_the_rest` uses the menu in place of
  All lines; the Regions tab's `test_the_line_ids_menu` passes unchanged.
- *The change*: `lines.LINE_IDS` and `labelled_choice(view)`; `session.view_of`
  reads a Regions-tab choice as the shared one; the Regions tab's `SETTINGS` lose
  "line_ids", its menu reads and writes the shared setting; the Data tab's "All
  lines" switch becomes the same menu, `_draw_ids` labelling the Regions tab's
  transition or ion of its system for This transition and This ion, the primary
  system otherwise; `MainWindow.line_ids_changed` tells the other tab
  (`sync_line_ids`).

**Task 11.8 — Labels from the component's scheme (item 3, Q11.4 and Q11.5).** Done.
- *The scheme* (`edit.py`): `SCHEME_PREFIX` adds `n` for a column density;
  `scheme_number(label, letter)` reads a label's number (`za3`, `NA3SI2`, `nly3`;
  not `zaa1` for the letter a); `free_number(pm, letter)` is the first number no
  label of the scheme uses in the system, live or commented out; `new_component`
  takes it (so a column density's label counts too). `suggest_label` keeps the
  datasets' suggestions and gives none for a component's value, which the component
  suggests.
- *The component's labels* (`project.py`): `ion_word` (`si2`, `s3`, `d1`, `h1`,
  `o1s`, `c2_13`; nothing for the forest), `component_number` (its redshift label's
  number, else its other labels', else the system's first free number) and
  `scheme_label(pm, comp, name, letter, ion=)`: L + S + #, with the ion for N; a
  label a line outside the component uses is never given (the first free number
  then). D20's and D22's fixes (`component_label`, `tie_isotope`) use it.
  `components.py`: a temperature mode's own labels (D93) use it, `_tied_first`
  gives a component of no system the forest's letter in place of `x`, and
  `holding` and `suggest_label(project, par)` give the card its question's label
  (`qt/components.py`, `_with_label`).
- *Tests*: `test_the_scheme_s_numbers` and `test_a_redshift_gets_a_label_from_the_
  scheme` (`test_dashboard_edit.py`: J1358's H I component a1 is offered `za1` and
  `na1h1`, in place of `zx1`; `edit` alone offers none). In
  `test_dashboard_components.py`: `test_a_component_s_values_are_offered_its_scheme_
  s_labels` (a3, in a system whose labels use 1 and 2: `za3`, `ba3`, `ta3`, `na3h1`,
  `na3si2`, `na3s3`, `na3d1`; forest line ly3: `bly3`, `nly3`), `test_a_component_
  s_labels_share_its_number` (z `za3` gives `ba3` and `na3si2`; a1's `za5` gives
  `na5h1`; a3 sharing a1's `ba1` is offered `za3`, `za1` being a1's), `test_a_
  temperature_mode_s_label_shares_the_component_s_number` (pure thermal on a
  component of `za3` writes `ta3`). `test_the_card_asks_for_the_component_s_label`
  (`test_dashboard_qt_components.py`): Fixed on a3's z asks with `za3`, on its Si II
  N with `na3si2`, and writes them fixed.
- *Tests changed by the rule*: `test_an_excluded_ion_keeps_the_component_s_new_label`
  (the exclusion ties a2 to `za1`, so its own T label is `ta1`, not `ta2`) and
  `test_rjc_j1011_pure_thermal_on_a2_leaves_the_others` (a2 carries `BA1`, so `ta1`,
  not `ta3`): the new label shares the component's number, as Q11.5's response
  asks.

**Task 11.9 — Renumber a system's labels by redshift (Q11.5's response).** Done.
- *The plan* (`components.py`): `renumber_plan(project, system=None)` (None: the Lyα
  forest) gives each component's labels of the scheme, on its lines and on the
  lines commented out of it, the number its id shows (a2: `za3` → `za2`, with its
  b, T and N). Labels the user named, labels shared with another component, and
  labels of another system's scheme are left, and named in the plan's notes, after
  the warning (the labels of every component change; labels written outside the
  dashboard will no longer match; Undo takes it back). Its lines say what each line
  renames. `renumbered` says whether there is nothing to do. `edit.rename_labels`
  renames several labels in one change (the lines, those commented out, `fix` and
  `lim` commands, links), so that swaps need no name between.
- *The action*: `components.renumber`, "Renumber labels…", a button under the
  navigator beside Add ion… and Add system…; it acts on the system shown, or the
  forest; the warning is a preview (`PlanDialog`, "Renumber") with the lines that
  change; it is greyed, its tooltip saying so, when the labels are in order.
- *Tests* (`test_dashboard_components.py`): `test_renumber_labels_by_redshift` (a1,
  a2, a3 reading 1, 3, 2 become 1, 2, 3 with their b and N; `ta3`, shared by a2 and
  a3, and `bsh`, the user's, stay and are named; the `lim` command, the link and a
  commented-out line of a2 follow; ALIS reads it; in order afterwards; one undo),
  `test_the_renumber_plan_lists_the_lines_that_change`. In
  `test_dashboard_qt_components.py`: `test_renumber_labels_after_a_warning` (the
  button, the warning's text and lines, Cancel, one step, greyed with its tooltip,
  undo, no design reference).

*Batches* (on a snapshot of the tree, so that work went on meanwhile; locally with
pytest 9.0.2 and in a scratch venv with CI's pytest 9.2.0):
- after 11.1: unit 2214 passed, 119 skipped; gui 279 passed; under 9.2.0, unit 2213
  passed, 120 skipped, and gui 279 passed.
- after 11.4: unit 2227 passed, 119 skipped (9.2.0: 2226, 120); gui 286 passed and 1
  failed, under both: `test_open_a_fit_of_another_spectrum_says_why`, which expected
  the refusal that 11.3 removed. The window and that test were changed in 11.5, as
  planned.
- after 11.7: unit 2227 passed, 119 skipped (9.2.0: 2226, 120); gui 299 passed (and
  299 under 9.2.0).
- after 11.9: unit 2233 passed, 119 skipped (9.2.0: 2232, 120); gui 301 passed (and
  301 under 9.2.0). No failures. (Taken before the two review changes, which their
  own tests cover; the close runs every batch again.)

**Task 11.10 — The review.** Draft 1 published, waiting on RJC's comments.
- `doc/dashboard/stage11/take_screenshots.py` (all synthetic, from
  `tests/dashboard_fits.py`; dialogs drawn on their own, questions and warnings as
  the window's message box draws them) and `build_review.py` (the earlier pages'
  stylesheet). Eleven screens: the question, the New project dialog, the dialog that
  assigns the snips, the warning before cutting again, the overlap warning, the
  preview (specB becomes the reference; specC unassigned), the Regions tab's two
  snips of O I in one dataset, the Components tab's buttons, the Data tab's line IDs
  (This transition, Si II 1304), a card's label question (`na3si2`) and Renumber
  labels…'s warning. Tasks with nothing new to see are listed under "Also done".
- Found while taking them: the snips table's first column elides "vb · …"; its
  cells now carry the full name as a tooltip.
- Published as a private page: https://claude.ai/artifact/Etked222zQpXMBmeJQoqBi
  (version 1). It asks whether a Load snips… file that names no transition should go
  to the dialog's right rather than stop the load (see 11.5).

- *RJC's comments on draft 1.* "These buttons cannot be seen. I think it would be
  better to have them vertically stacked, rather than horizontally stacked." (the
  Components tab's buttons under the navigator, which read "Ad…n…", "A……",
  "Re…s…" side by side in the narrow pane). Applied: `qt/components.py` stacks them
  (`nav_buttons` is a `QVBoxLayout`), each expanding to the pane's width;
  `test_renumber_labels_after_a_warning` checks that the three are stacked, equally
  wide and not cut short. The screenshot was retaken and the page republished as
  version 2; the thread is answered and resolved.
- *RJC's comment on draft 2* (the overlap warning): "Can we change this message to:
  It is advised that each snip has its own dataset. However, you can choose to keep
  these in a single dataset if you prefer." Applied in `qt/window.py`
  (`assign_snips`): the warning names each overlapping pair, then gives RJC's two
  sentences (the sentence about the Regions tab is gone).
  `test_overlapping_unassigned_snips_are_warned` checks the new ending. The
  screenshot was retaken and the page republished as version 3; the thread is
  answered and resolved.
- *RJC's comment on draft 3* (the warning before a snip is cut again): "The two
  options here are not clear enough. The buttons should say "Cut again" and
  "Unassign snip"". Applied in `qt/dialogs.py` (`AssignDialog._agree` and `move`):
  the buttons are Cut again and Unassign snip (Unassign snips for several dropped
  together); Unassign snip puts the snips on the right, ticked (loaded as they are),
  where Cancel had left them where they were. `test_a_snip_given_to_another_
  spectrum_is_cut_again_after_a_warning` checks both buttons and the snip on the
  right. Screenshot retaken, page republished as version 4, thread answered and
  resolved.
- *RJC's comment on draft 4* (the assignment dialog's message): "Please change this
  information message to read: …" (two paragraphs: check each snip's dataset, move
  a snip through the unassigned snips, drag or use the arrows or double-click; the
  unassigned snips as their own datasets or one "unassigned" dataset, and untick to
  exclude). Applied in `qt/dialogs.py` (`AssignDialog`'s message), with the arrows
  as → and ←, and "datset" read as "dataset". `test_the_best_guess_is_shown` checks
  the message. Screenshot retaken, page republished as version 5, thread answered
  and resolved.
- *RJC's comment on draft 5* (Open a fit's question): "How does the project know
  that the three snips come from 3 datasets? At this stage, we can't know that
  information." with the question to write instead. Applied in `qt/window.py`
  (`open_fit_new`): RJC's wording, with the `.mod`'s name, for every fit (the
  question no longer differs by the `.mod`'s number of datasets).
  `test_the_question_counts_the_datasets` became `test_the_question_asks_for_the_
  full_spectra`, checking the text. The New project dialog made the same claim, so
  it now says only what the `.mod` names ("old.mod names 2 snips in 2 datasets (src,
  other). Add every spectrum the snips were cut from: …"), and
  `test_a_fit_of_two_spectra_opens_with_both` checks it. Asked in the thread: whether
  a fit of one dataset should keep its old question, and whether the button "Open
  with its spectrum (recommended)…" should say "spectra". Screenshots retaken, page
  republished as version 6, thread answered and resolved.
- *RJC's comments on draft 6* (three threads): the right-hand list to be titled
  "Unassigned snips" (was "Not from these spectra"), and the choice to read "Each
  snip is its own dataset", in the dialog and in the overlap warning (one text,
  `dialogs.EACH_TEXT`). Applied in `qt/dialogs.py`; `test_the_best_guess_is_shown`
  and `test_overlapping_unassigned_snips_are_warned` check them. Screenshots
  retaken, page republished as version 7, the three threads answered and resolved.
- *RJC's comment on draft 7*: Open a fit's first button to read "Open with full
  spectra (recommended)…" (was "Open with its spectrum (recommended)…"). Applied in
  `qt/window.py` (`WITH_SPECTRUM`); `test_the_question_asks_for_the_full_spectra`
  checks it. Screenshot retaken, page republished as version 8, thread answered and
  resolved. This answers the second of the two questions asked on draft 5.
