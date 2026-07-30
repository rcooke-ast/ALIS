# ALIS v2 Code Plan Log

---

### 2026-07-13 (Task 1: Draft refactor plan)

Reviewed the context needed to draft the ALIS v2 refactoring plan:

- `claude_prompts/context.md` (the `Context` section), which lists all the
  intended changes: circular-import removal, single model-function
  instantiation, removing `ClassMain`/`self` passing, dataclasses in place of
  nested dicts, `logging` instead of `almsgs`, Python 2 cleanup, type
  annotations, profile caching, convergence robustness, fit diagnostics, GPU
  support, modularity, unit tests, GUI, packaging (`pyproject.toml`), CI,
  pre-commit, semantic versioning, YAML/TOML model files, atomic-data
  modernisation, plotting-script output, CLI modernisation, and docs.
- `doc/ALIS_workflow.md` (already authored in the prior work stream).
- The current source in `alis/` — surveyed all module sizes and read the crux
  files.

**Key architectural findings (current v1):**

- `alis.py::myfunct_wrap` builds a brand-new `ClassMain` on *every* χ² iteration
  and copies state in via `instance.__dict__.update(fdict)` — this is the
  circular-import / `self`-passing problem at the centre of the refactor.
- `ClassMain.model_func` is a single very large method that carries all state on
  `self` and rebuilds per-iteration parameter arrays; the model-function classes
  live in a `_funcarray` triple `[functions, funccall, funcinst]` that is
  re-instantiated redundantly (including inside the `sim` repeat loop).
- `alfunc_base.Base` already defines the interface every model function
  implements: `call_CPU`, a `call_GPU` stub, `set_vars`, `load`, `parin`,
  `parout`, `set_pinfo`, `tick_info`. `alfunc_voigt.Voigt` already has a
  `GPU_kernal` method stub.
- The largest modules are `alload.py` (1945), `alcsmin.py` (1616, an `mpfit`
  wrapper), `alshift.py` (1087), and `alis.py` (1012).
- `context/voigt_gpu/` contains a Faddeeva-function Voigt implementation
  (numba/`faddeeva`) that is the reference for the GPU port.

**Work done:**

- Wrote Draft 1 of the plan into the `Code Plan` section of
  `claude_prompts/ALIS_v2_code_plan.md`. Structured it as four guiding
  principles (feature parity first, test before touching, incremental steps,
  refactor→optimise→extend) followed by a phased, ordered task list:
  - Phase 0: regression-test safety net from the existing examples
  - Phase 1: low-risk modernisation (Py2 cleanup, `pyproject.toml`, CI,
    pre-commit, semver/CHANGELOG)
  - Phase 2: core architecture (dataclasses, single instantiation, remove
    `ClassMain`, fix multiprocessing state, `logging`)
  - Phase 3: fitting engine (caching, convergence robustness, diagnostics)
  - Phase 4: GPU + modularity
  - Phase 5: I/O (YAML/TOML, atomic data, plotting scripts)
  - Phase 6: usability (CLI, GUI, Sphinx/RTD docs, CONTRIBUTING)
  - Plus a dependency summary.
- Raised 9 queries for RJC (regression tolerance and seeding, landing strategy,
  `.mod`/`.mod.out` compatibility, GPU stack, structured-config library,
  minimum Python version, atomic-data format, GUI toolkit, and phase priority).

No skills were used: this is a planning/writing task, none of the ALIS-specific
skills apply, and RJC's review of the draft is itself the intended feedback step.

Per the instructions, this is a draft only — awaiting RJC's responses to the
queries before preparing Draft 2.

---

### 2026-07-13 (Task 1 review: RJC responses + reordering)

Reviewed RJC's responses to the 9 Draft-1 queries and the ordering changes RJC
made to the plan. Key decisions captured from the responses:

- **Regression tolerance:** numerical, not bitwise — best-fit parameters within
  1% of their 1σ error, χ² within 0.1% relative; fixed seed for randomised
  examples.
- **Landing strategy:** incremental. RJC does all git ops; each subtask on its
  own branch, merged into `alis_v2_stageX`, then all stages merged into
  `alis_v2`.
- **`.mod.out`:** must remain usable as input to an independent ALIS run and
  include a commented copy of the original input; not usable when values are
  blinded.
- **GPU:** optional install extra; start from `context/voigt_gpu/`; stack chosen
  later.
- **Config library:** no new dependencies without prior discussion.
- **Python floor:** 3.13+ (enables stdlib `tomllib`, modern type hints).
- **Atomic data:** human-readable, easily-edited/validated plain-text; format
  chosen later.
- **GUI:** `prepfit` uses matplotlib with the `Qt5Agg` backend; toolkit decided
  later.
- **Priority:** concentrate effort on Phase 2 (architecture cleanup).

RJC also reordered two pairs of subtasks: Phase 2 swapped logging (now 2.4) and
multiprocessing state passing (now 2.5); Phase 3 swapped fit diagnostics (now
3.2) and convergence robustness (now 3.3). I reviewed both and agree, with one
flagged coupling: 2.3 (remove `ClassMain`) and 2.5 (multiprocessing state
passing) are the same underlying mechanism (`myfunct_wrap` rebuilds `ClassMain`
from `fdict` for worker-process pickling), so they should be treated as a coupled
unit, and the logging design (2.4) must be multiprocessing-aware from the start.

RJC also added a workflow note: once the plan is finalised, Claude will generate
per-phase prompt files by copying the provided template `refactor_code_stageX.md`
(which I read — it has Tasks / Skills / Context / Queries / Prompts sections and
directs per-phase logs to `claude_prompts/logs/refactor_code_stageX_log.md`).

Added four follow-up queries (Q10–Q13) and a reordering-review note to the
`Queries` section of `ALIS_v2_code_plan.md`:

- Q10: "phase" vs "stage" terminology and whether the generated file should be
  `refactor_code_stage0.md` or `refactor_code_phase0.md` (recommend standardising
  on "stage").
- Q11: whether the regression tolerance is measured on full-refit endpoints or on
  fixed-parameter model/χ² output (recommend the latter as the primary gate).
- Q12: whether "no new dependencies" is runtime-only, so dev tooling (pytest,
  ruff, sphinx) and optional GPU/YAML/TOML-writer libs remain acceptable.
- Q13: whether astropy ECSV is an acceptable atomic-data candidate for Phase 5.2.

No phase files were generated yet — that step waits until the plan is finalised.
No skills were used: this remains a planning/review task.

---

### 2026-07-13 (Task 1 review round 2: promote to Draft 2 + reference-file convention)

Read RJC's responses to the Draft-2 follow-up queries (Q10–Q13) and to the
reordering note, and read Prompt 3 introducing the reference-file testing
convention. Key decisions:

- **Q10:** standardise on "stage" everywhere; no "phase" in the plan. RJC also
  reverted Stage 2 to the original order (multiprocessing state passing 2.4 →
  logging 2.5).
- **Q11:** the regression tolerance is measured primarily on a *fixed* parameter
  set (model + χ² within 0.1% relative); the full-refit endpoint is a secondary
  check (best-fit parameters within 10% of 1σ, χ² within 1%).
- **Q12:** "no new dependencies" is runtime-only; dev/build tooling and optional
  extras are acceptable. New runtime deps still need RJC sign-off.
- **Q13:** ECSV is an acceptable atomic-data candidate; prefer extra whitespace
  so columns are clearly separated.
- **Prompt 3 (reference files):** RJC will generate golden
  `.mod.out.reference` files (beside each `.mod`, in `model/`) and
  `_fit.dat.reference` files (in a `reference_fits/` folder inside `data/`).
  These must never be overwritten. A `.mod` with no `.mod.out.reference` beside
  it is excluded from the test suite.

**Edits made to `ALIS_v2_code_plan.md` (now Draft 2):**

- Renamed every "Phase" → "Stage" in the Code Plan section (headings, gating
  note, dependency summary) and in my two reordering-review bullets; updated the
  Status block to describe Draft 2.
- Rewrote the task-list preamble to describe copying `refactor_code_stageX.md`
  → `refactor_code_stage<N>.md` with per-stage logs in
  `claude_prompts/logs/refactor_code_stage<N>_log.md`.
- Stage 0: replaced the generic "golden files" text with the concrete
  reference-file convention (`.mod.out.reference` beside the `.mod`;
  `_fit.dat.reference` in `data/reference_fits/`; `.mod` skipped if no
  reference), added the fixed-seed note, and replaced the tolerance placeholder
  with the concrete primary/secondary gates from Q11.
- Stage 1.2: pinned Python 3.13.
- Stage 5.2: pointed the atomic-data work at a human-readable,
  whitespace-aligned plain-text table (ECSV candidate).
- Dependency summary: added the runtime-vs-dev/optional dependency policy.
- Added four follow-up queries (Q14–Q17) about the reference-file specifics: the
  `_fit.dat.reference` location, the (garbled) `_fit.dat` suite-inclusion rule,
  exact naming/pairing, and whether to retitle the template's "PHASE X" heading
  to "STAGE X".

Left intact as historical record: RJC's response text (which uses lowercase
"phase") and the Q10 discussion itself (which is intrinsically about the
phase/stage naming). Still awaiting RJC's answers to Q14–Q17 before generating
the per-stage `refactor_code_stage<N>.md` files.

---

### 2026-07-13 (Task 1 review round 3: inspect the `metal_line_abs` reference example)

Read RJC's answers to Q14–Q17 (fit references are named `_fit.dat`, not
`_fit.dat.reference`, and live in `data/reference_fits/`; pairing is by
model-name for `.mod.out` and by data-name for `_fit.dat`; the template's "PHASE"
heading was fixed by RJC) and inspected the example RJC generated in
`examples/metal_line_abs/` (Prompt 4).

**Findings — the layout is sensible and matches the convention:**

- `model/` has three fit models (`fit_spectra.mod`, `fit_spectra_linear.mod`,
  `fit_spectra_thermal.mod`), each with a matching `<name>.mod.out.reference`, so
  all three are in the suite. The two `generate_*.mod` files have no reference and
  are correctly excluded (they generate data, not fits).
- `data/reference_fits/` holds `OI_SiII_fit.dat` and `OI_SiII_thermal_fit.dat`,
  paired by data-file name.
- Verified `fit_spectra.mod.out` is currently byte-identical to its `.reference`
  (a no-op refactor passes trivially).

**Two issues raised as Q18 and Q19:**

- **Q18 (collision):** `fit_spectra.mod` and `fit_spectra_linear.mod` both read
  `OI_SiII.dat`, so both write `data/OI_SiII_fit.dat`, but there is only one
  golden `reference_fits/OI_SiII_fit.dat`. The two runs overwrite each other's
  output and the reference is ambiguous. Proposed keying fit references by model
  (subfolder or `<modelname>__<dataname>_fit.dat`) or designating a single
  `_fit.dat` owner per data file. The `.mod.out` check and the primary
  fixed-parameter gate are unaffected.
- **Q19 (volatile header):** the `.mod.out` header has per-run lines (`Generated
  by ALIS on <date> at <time>`, `Running Time (hrs)`, `Num. of Iterations`), so
  the harness must parse and compare numeric fields (best-fit params, χ², DOF)
  within tolerance rather than byte-diffing against the reference. Confirmed this
  is necessary by inspecting the reference header.

Told RJC the layout is sensible to replicate for further examples, subject to
resolving Q18 (the shared-data-file case) so the `_fit.dat` pairing scales. Still
have not generated the per-stage files — awaiting finalisation.

---

### 2026-07-13 (Task 1 review round 4: audit all example reference files + propose unit-test task)

Read RJC's responses to Q18/Q19 (fix: one `.dat` per model so `_fit.dat` names
are unique; harness parses numeric fields rather than byte-diffing) and executed
Prompt 5: audit every reference example under `examples/` and propose a task to
add one unit example per model function.

**Audit result — all models suitable.** Cross-checked each fit `.mod`'s data
block against `data/reference_fits/` for all 11 tested examples (CNabs,
emission_line_ratio, generate, lls, lsf_file, lsf_hst, lsf_multigauss,
metal_line_abs, spline, splineabs, summed_coldens):

- Q18 fix confirmed everywhere — each fit model now reads a uniquely-named data
  file (e.g. `metal_line_abs`/`splineabs` gained `OI_SiII_linear.dat` →
  `OI_SiII_linear_fit.dat`), so no `_fit.dat` collisions remain.
- Every `generate_*.mod` correctly lacks a `.mod.out.reference` (excluded from
  the fit suite).
- The new `generate` example is correct: data-block name `OI_SiII.dat` matches
  the golden `data/reference_fits/OI_SiII.dat` (no `_fit` suffix).

**Coverage analysis of `alfunc_*` functions.** Extracted the registered id string
from each `alfunc_*.py` and grepped all example `.mod` files (case-insensitive) to
map coverage:

- Covered (14): constant, gaussian, legendre, linear, phionxs, spline, splineabs,
  variable, voigt, vfwhm, afwhm, lsf, lsffile, multivfwhm.
- Gaps (11) — no example: apod, brokenpowerlaw, chebyshev, lineemission,
  polynomial, powerlaw, random, thar, tophat, voigtconv, vsigma.
- `lsfspline` only appears in the excluded `lsf_spline` example; `base`/`user`
  are the abstract base and user template (no example needed).

**Actions:**

- Added **Task 2** to the Tasks section: generate minimal, fast unit examples for
  the 11 uncovered functions (full generate→fit pattern with references), listing
  covered/uncovered/special-case functions. Marked "do not create yet — RJC to
  review".
- Added a review note plus three queries (Q20–Q22) to the Queries section: how
  randomness is seeded so `generate`/`random` references are reproducible (Q20);
  whether to add a tested `lsfspline` unit example (Q21); and confirmation of the
  unit-example pattern and minimal host models for the convolution/emission/
  special functions (Q22).

Did not create any unit examples (per instruction). Still awaiting finalisation
before generating the per-stage `refactor_code_stage<N>.md` files.

---

### 2026-07-13 (Task 2: build unit examples — 5 created, 6 functions found broken)

Executed Task 2 (create one unit example per uncovered `alfunc_*`). Used
`generate peaksnr -1`? No — for *fit* examples the data must be noisy with
non-zero errors (peaksnr -1 zeroes flux/error, which only suits the dedicated
`generate` test), so I mirrored the template peaksnr (20 for the metal_line_abs
base, 100 for emission_line_ratio). Data is committed once, so the fit is
deterministic.

**Created and verified (5) — references saved** (Template 1: new function as the
emission continuum in both generate and fit, atop the O I / Si II voigt
absorption):

| Example | best-fit reduced χ² |
|---------|--------------------|
| `powerlaw` | 1.085 |
| `chebyshev` | 0.951 |
| `polynomial` | 1.181 |
| `brokenpowerlaw` | 0.971 (only `coefficient` free; blueindex/redindex/location/strength fixed to avoid the flat-continuum degeneracy) |
| `tophat` | 1.091 (offset/height free; centroid/width fixed) |

Each has `generate_spectra.mod`, `fit_spectra.mod`,
`fit_spectra.mod.out.reference`, and `data/reference_fits/OI_SiII_<name>_fit.dat`.

**Could NOT produce references — these functions are broken in the *current*
ALIS** (a significant finding; exactly the kind of breakage a unit suite should
guard against). I removed the partial example directories to keep `examples/`
clean; each is trivially recreatable once the function is fixed:

- `vsigma` (Template 2, `resolution=vsigma(3.0VS)`): `alfunc_vsigma.py:52`
  `np.zeros(2*df+1)` with `df` a float → `TypeError`.
- `voigtconv` (`resolution=voigtconv(3.0SG,0.0DP)`): `VoigtConv.call_CPU()` has no
  `mkey` kwarg but `alis.py:380` passes `mkey=...` → `TypeError`.
- `apod` (`resolution=apod(1.0SC,kind=uniform)`): requires `kind=` to be set, then
  `alfunc_apod.py:191` `param[i]=isspl[i]` → `IndexError` in `load()`.
- `thar` (Template 3, `thar 10.0 6563.0 30.0`): drops into an IPython `embed()`
  during "Preparing model parameters" and hangs.
- `lineemission` (Template 3): `ion=` must be `<el>_<stage>_<wavelength>` (e.g.
  `1H_I_6564.623`); after fixing that, `alfunc_lineemission.py:73` (`voigtking`)
  raises `TypeError: only 0-dimensional arrays can be converted to Python
  scalars`.
- `random` (Template 1): structurally a variable-type function — `call_CPU`
  returns `None`, requires a `command=` keyword, and produces random values by
  design; it cannot be a drop-in emission component for `legendre`.

Net: 14 (previously covered) + 5 (new) = 19 model functions now have a tested
example; 6 remain untestable until fixed/clarified. Recorded specifics so RJC can
regenerate the 6 once addressed. Raised this as a follow-up note in the plan's
Queries section.

---

### 2026-07-14 (Task 2 verification: check RJC's new references for the fixed functions)

RJC fixed `vsigma`, `voigtconv`, `random`, `lineemission` and generated their
examples/references (`apod` and `thar` deliberately left unsupported). Prompt 8
asked me to check the new references are correctly formatted and named.

**Audit — all correct.** Each of the 9 unit examples (my 5 + RJC's 4) has a fit
`fit_spectra.mod` with a matching `fit_spectra.mod.out.reference`, a
reference-less `generate_spectra.mod` (excluded from the suite), and a
`data/reference_fits/<dataname>_fit.dat` that pairs by data-file name with the
fit output. `apod`/`thar` are correctly absent. The `.mod.out.reference` files are
well-formed (best-fit model block + commented copy of the input, per Q3). All
`_fit.dat` references are 4-column.

**Reproduction check — re-ran the 4 fixed fits against their references:**

| function | rerun reduced χ² | reference | `_fit.dat` |
|---|---|---|---|
| vsigma | 1.064831 | 1.064831 | byte-identical |
| voigtconv | 1.048276 | 1.048276 | byte-identical |
| lineemission | 1.023691 | 1.023691 | byte-identical |
| random | 1.124713 | 1.124713 | max\|Δ\|=1.1e-5 (rel 1.7e-5), within tolerance |

`random` reproduces the same best-fit χ² and parameters despite its random
starting value (as RJC expected); its `_fit.dat` model column varies at the
~1e-5 level run-to-run (RNG), which is far inside the 0.1% gate but means the
harness must compare `random`'s `_fit.dat` with the numeric tolerance, not a
byte-diff (consistent with the Q19 design).

Minor cosmetic note: RJC's new examples name the data file `OI_SiII.dat` (generic)
whereas mine use `OI_SiII_<name>.dat`; both are fine because each example is an
isolated directory, so no `_fit.dat` collisions occur.

Coverage now: 14 previously-covered + 5 (mine) + 4 (RJC's fixes) = 23 model
functions with a tested example. Only `apod` and `thar` remain untested
(unsupported for now).

---

### 2026-07-17 (Prompt 9: confirm readiness, fold in test guidance, draft Stage 0 file)

Confirmed the test suite is ready and generated the Stage 0 prompt file.

**Readiness scan:**
- `examples/`: every fit example has matching `.mod.out.reference` and
  `reference_fits/*_fit.dat` (metal_line_abs 3/3, spline/splineabs/
  emission_line_ratio 2/2, all unit examples 1/1); no `.covar` files here;
  `generate`, `prepfit`, `lsf_spline` handled per convention.
- `context/fitting_examples/`: 16 `.mod.out.reference`, 17 `reference_fits/`
  dirs, and **10 `.covar.reference`** files (the DH / DH_orders / helium34 fits)
  — this is where `.covar` testing applies.
- A top-level `tests/` dir exists (empty of tests) to hold the harness.

**Incorporated RJC's Prompt-9 guidance** into the plan's Stage 0.2: the harness
runs only the fit models (data pre-exists, so `generate_spectra.mod` is not run
at test time) and compares `.mod.out`, `_fit.dat`, and `.covar` (where a
`.covar.reference` exists).

**Generated `claude_prompts/refactor_code_stage0.md`** from the template, with a
detailed Stage 0 breakdown (0.1 inventory/manifest; 0.2 pytest harness with
numeric `.mod.out`/`_fit.dat`/`.covar` comparison and volatile-header handling;
0.3 fast/slow split + determinism; 0.4 primary/secondary tolerances; 0.5
CI-readiness), plus Skills, Context, and four Stage-0 queries (Q0.1
fixed-parameter evaluation mechanism; Q0.2 `.covar` emission; Q0.3 harness
location/framework; Q0.4 how to test the `generate` example).

**Note on Task 3:** the Tasks section defines only Tasks 1 and 2 — "Task 3" is
not written. Interpreted "proceed with the next stage of the refactor" as the
plan's finalization step (generate the per-stage prompt files) and produced the
Stage 0 file as the concrete next step. Asked RJC to confirm and to say whether
to generate the remaining stage files (1–6) now or stage-by-stage.

---

### 2026-07-17 (Task 3: incorporate real-world references + generate all stage files)

RJC defined Task 3 (use the `context/fitting_examples/` reference files in the
test set; handle multi-`../data*/` data dirs; test `.mod.out.reference`,
`reference_fits/*_fit.dat`, and `.covar.reference` where present; update the
Staged Task List) and asked me to re-check `refactor_code_stage0.md` and then
write all stage files 1–6.

**Audited `context/fitting_examples/`:** 16 fits with `.mod.out.reference`, 13
with `.covar.reference`, 17 `reference_fits/` dirs. Discovery must be by
reference presence (varied `.mod` names: `*_FINAL_MODEL.mod`,
`*_converge_newstart76.mod`, etc.). Data lives in multiple `../data*/` dirs —
`DH/J1358p6522` uses `data/` + `data_hrdx/`; `DH_orders` uses `datafit_orders/` —
each with its own `reference_fits/`. `DH/Q1243p307/…_converge_newstart76.mod` is a
Monte-Carlo `newstart` (randomised) case.

**Updated the plan's Stage 0.2** to spell out: discovery by `.mod.out.reference`
presence, fit-only execution, `.covar` comparison where a reference exists, and
multi-`data*/` directory resolution. **Refined `refactor_code_stage0.md`** (0.1)
to resolve each input file's `_fit.dat` reference relative to that file's own
directory and to flag the randomised cases.

**Wrote `refactor_code_stage1.md` … stage6.md`** from the template, each with an
ordered subtask breakdown (from the Code Plan), relevant skills, context, and
stage-specific queries:
- Stage 1 (low-risk modernisation): Q1.1 runtime deps / IPython; Q1.2 CI matrix;
  Q1.3 starting version.
- Stage 2 (core architecture — priority): Q2.1 module layout; Q2.2 logging style;
  Q2.3 incremental dataclass migration.
- Stage 3 (fitting engine): Q3.1 caching correctness bar; Q3.2 diagnostics output;
  Q3.3 convergence approach.
- Stage 4 (GPU + modularity): Q4.1 GPU stack; Q4.2 GPU in CI; Q4.3 CPU↔GPU
  tolerance.
- Stage 5 (data/IO): Q5.1 ECSV confirmation; Q5.2 YAML/TOML deps; Q5.3 schema.
- Stage 6 (usability/GUI/docs): Q6.1 CLI library; Q6.2 GUI toolkit; Q6.3 docs
  hosting.

All 7 stage files (0–6) now exist alongside the `refactor_code_stageX.md`
template; each has Tasks + Queries and no "phase" wording. RJC will write each
file's `Prompts` section and review before that stage is executed. No `alis/`
code was changed.
---

### 2026-07-28 (Prompt 10: minimiser <-> model-eval call-chain performance analysis)

RJC asked for a holistic analysis (not implementation) of the
`minimise.alfit -> fcn -> myfunct -> model_func` path before starting Stage 4,
noting that `fcn` never varies and that per-derivative setup (e.g. renew_subpix)
is redundant.

Mapped the *current* (post-Stage-2/3) structure and reconciled it with the
prompt's older ClassMain-era wording:
- `fcn` is ALWAYS `model_eval._minimiser_eval` (module-level) at all 6 `alfit(...)`
  call sites (main.py x3, simulate.py x3). `alfit.call()` just does
  `fcn(x, **functkw)` with `functkw={x,y,err,state}` -> generic MPFIT indirection
  that is never exercised with a different function. Confirms RJC: `fcn` can be
  built into `alfit`.
- `myfunct`/`model_func` are NO LONGER in ClassMain -- Stage 2.3/2.4 already moved
  them to standalone `model_eval.py` functions taking an explicit picklable
  `FitState`. `main.py` keeps thin `self.myfunct/model_func` wrappers used only by
  non-minimiser paths (initial chi2, plot, sim setup).
- `base.call()` re-instantiation is ALREADY fixed (Stage 2.2 `build_funcarray`
  loads each function class once into `[names, classes, instances]`).
- `model_func_ddp` is DEAD (body starts `msgs.bug("Shifts not implemented...")`,
  only reachable via an unused main.py wrapper) -- safe to delete.
- renew_subpix: `model_func` recomputes `load.load_subpixels` at the TOP of every
  call when `run renew_subpix True`. Default is False, BUT it IS enabled in the
  real-world helium34 fits (Her36, HD319718) -> during the Jacobian this reruns
  2n times/iteration. Genuine redundancy, not hypothetical.
- Influence table `_pinfl`: rebuilt via `load_par_influence` on every base
  `model_func` call (model_eval.py:120), i.e. once/iteration, in addition to the
  two setup builds in main.py. Only 1x/iter (derivative calls skip it), so lower
  priority, but a candidate to hoist if it is value-independent.
- The per-derivative `copy.copy(state)` in `_minimiser_eval` (2n/iter) exists
  because `model_func` writes back into `state._modfinal/_contfinal/_zerofinal/
  _pinfl`; a cleaner return-value contract would remove the need to copy.

Proposed approach (for RJC review; nothing implemented): (A) absorb `fcn` into
`alfit` so the minimiser owns `FitState`; (B) make the per-iteration invariants
explicit -- a `prepare_iteration(p)` computed ONCE per accepted step (subpix grid,
`_pinfl`, base component cache) that the 2n derivative evals consume, removing the
renew_subpix 2n redundancy; (C) treat that `prepare()`/`eval_derivative(j)` split
as the CPU/GPU seam for Stage 4 (Task 4.5 shared memory uploads the per-iteration
invariants once); (D) delete dead `model_func_ddp`. Recorded queries on scope
(new stage doc vs Stage 4 task 4.0), the bitwise Stage-0 requirement, and whether
to also bank CPU perf now (renew_subpix hoist) vs a pure structural cleanup.

### 2026-07-28 (Prompt 10 follow-up: created Stage 3.5 doc + influence investigation)

RJC answered the four queries: (1) fold into a dedicated pre-Stage-4 doc; (2)
bitwise-identical under Stage 0; (3) pure structural cleanup enabling the GPU
port, banking CPU wins only where they don't complicate GPU; (4) investigate the
influence-table value-dependence.

Wrote `claude_prompts/refactor_code_stage3p5_minimiser.md` (Tasks 3.5.1-3.5.6:
absorb `fcn` into `alfit`; explicit `prepare_iteration`; return-not-mutate
`eval_derivative` contract; delete dead `model_func_ddp`; optional bitwise-safe
`renew_subpix` conditional recompute; unit tests). Nothing implemented -- RJC
writes the Prompts section and reviews first.

Investigation results recorded as Findings F1/F2 in that doc:
- **F1:** `_pinfl` is value-dependent. `functions/voigt.py:set_vars` returns an
  empty influence list when a component's *redshifted* lines
  (`Wavelength*(1+pt[1])`) fall outside the region window (`wvrng`, itself set by
  the shift params). So influence tracks the component redshift + shift and
  cannot be hoisted to once-per-fit; it is already computed once-per-iteration
  (base call) and reused by all derivatives, which is the correct granularity and
  what keeps Stage 3.1/3.4 caching bitwise-exact. `base.py:set_vars` ignores
  `wvrng`, so the value-dependence is specific to line functions (voigt/gaussian/
  lineemission/splineabs).
- **F2:** `renew_subpix` per-derivative recompute is NOT a safe blanket hoist:
  `load_subpixels` sizes sub-pixels from fitted line widths + instrumental FWHM,
  so perturbing those genuinely changes the grid (a real derivative dependence).
  Bitwise-safe win is a *conditional* recompute keyed on whether the perturbed
  parameter feeds the subpix model (Task 3.5.5), not "compute once per
  iteration."

### 2026-07-29 (Stage 3.5: defer Task 3.5.3 to Stage 4)

RJC asked whether Task 3.5.3 (return-not-mutate eval contract) can be deferred to
the GPU stage without affecting the other Stage 3.5 tasks. Confirmed yes, and it
is the better sequencing:
- The other tasks (3.5.1 absorb `fcn`, 3.5.2 `prepare_iteration`, 3.5.4 dead-code,
  3.5.5 renew_subpix, 3.5.6 tests) do not consume the return contract. The
  residual is *already* returned (`myfunct` -> `[status,(y-modf)/err]`); the
  per-call `copy.copy(state)` only isolates scratch mutations
  (`_modfinal/_contfinal/_zerofinal/_pinfl`), so removing it is orthogonal to the
  other tasks.
- 3.5.3 is the highest bitwise-risk task, its necessity is GPU-specific (a kernel
  can't mutate shared Python state), and its CPU benefit is negligible (~2n
  shallow ref-rebinds/iter vs a ~95%-compute Jacobian; read-only arrays are
  shared not duplicated, so it doesn't block Task 4.5 shared memory).
Edits: in `refactor_code_stage3p5_minimiser.md` marked 3.5.3 DEFERRED (kept the
slot/number, added rationale), added a "define a clean `eval_derivative(j)`
boundary" instruction to 3.5.2 so the deferred contract is a localized Stage 4
change, dropped the eval-contract item from 3.5.6, and filled Q3.5.6. In
`refactor_code_stage4.md` added Task 4.0 (return-not-mutate contract, do first,
with its unit tests). Nothing implemented.

### 2026-07-29 (Pre-Stage-4: reconcile stage4 doc with the post-3.5 codebase)

After Stage 3.5 completed, updated `refactor_code_stage4.md` so it matches the
current code and the 3.5 outcomes: fixed stale filenames (`alfunc_base` -> class
`Base` in `alis/functions/base.py`, which already has `call_CPU`+`call_GPU` stub;
`alfunc_voigt.py` -> `alis/functions/voigt.py`; `alfunc_*` -> `alis/functions/
<name>.py`; the commented PyCUDA scaffolding is in `main.py`/`model_eval.py`/
several `functions/*.py`, not `alis.py`); corrected Task 4.0 to reference the real
seam (`prepare_iteration()`/`self._emab` + `_worker_funcderiv`, not a literal
`eval_derivative(j)` method) and the real mutated scratch
(`_modfinal/_contfinal/_zerofinal`; `_pinfl` is per-fit, F1); updated the
depends-on line (Stage 2 + 3.4 + 3.5); added a Context note for the 3.5 seam; and
noted the deferred 3.5.5 renew_subpix conditional recompute as an optional
carry-in to Task 4.5. Confirmed the `port-to-gpu`/`gpu-benchmark`/`new-alfunc`/
`run-tests` skills exist. Q4.1-Q4.6 answered; Q4.7/Q4.8/Q4.9 still open (flagged
to RJC before starting Stage 4).

### 2026-07-29 (Pre-Stage-4: CPU/GPU backend design decisions)

Design discussion with RJC settled the Stage 4 parallelism model; folded into
`refactor_code_stage4.md`:
- **Either-or, not hybrid** (Q4.8 resolved): the parallel backend is CPU *or* GPU
  per fit, never both computing derivative columns at once. A GPU model-eval is
  ~50x a CPU one at DH_orders scale, so GPU-only beats CPU-only ~15x even across
  ~4 GPUs; a hybrid adds only single-digit % for large complexity / mixed-gate /
  determinism cost. GPU backend reuses the persistent-Pool machinery sized to
  `ngpus`, one GPU bound per worker (`cuda.select_device(rank)`, `spawn`).
- **Dispatch shape** (Q4.7 resolved): keep `call_GPU` per-component in
  `functions/<name>.py` (device arrays, batched); the dispatcher batches same-type
  components/spectra with a size-threshold CPU fallback; upload the sub-pixel wave
  grid + read-only data once per iteration via `prepare_iteration()`, keep
  device-resident, intermediates on-device. Kernel *location* is runtime-
  irrelevant -> hard-coding kernels in alfit/model_eval would not be faster.
- **Backend selection** (new Task 4.3a): `run backend = auto|cpu|gpu` (default
  auto). `auto` warms both backends then times a Jacobian at p0 on each and
  commits the whole fit to the winner (one numerical gate); `cpu`/`gpu` force a
  backend for reproducibility; Stage 0 harness forces cpu. Warming first is
  essential or a cold GPU probe times JIT/context and mis-picks CPU.
- Open queries remaining before Stage 4: **Q4.9** (reuse CPU refs + `gpu` marker
  vs dedicated GPU references) and **Q4.10** (auto semantics when `ngpus` unset;
  recommend CPU-only unless ngpus explicitly requested).

### 2026-07-29 (Prompt 2: fold Q4.9/Q4.10 responses; final pre-Stage-4 queries)

RJC answered Q4.9 (reuse CPU references for GPU runs via a `gpu` marker; wants an
on-demand local command) and Q4.10 (CPU-only unless `ngpus` explicitly set; wants
a "GPUs available" info message). Folded both into `refactor_code_stage4.md`:
- 4.6: `gpu` marker + `--run-gpu` opt-in (mirroring `--run-slow`), on-demand
  command `pytest --run-gpu -m gpu`, documented in tests/README.md.
- 4.3a: `backend`/`ncpus`/`ngpus` interplay + a GPU-available INFO notice when
  GPUs are present but unused.
- 4.0: added the Finding-F1 interaction -- dropping `copy.copy(state)` requires
  freezing `_pinfl` (compute once pre-fit, stop recomputing during the fit) so the
  base call doesn't mutate the shared influence table; this is the agreed
  set-once behaviour and is what makes the copy removal bitwise-safe.
- 4.1: `numba`/CUDA must be imported lazily (GPU-selection only) so CPU-only
  installs never import them.
- Context: GPU stack recorded as numba.cuda / float64.
One new query raised: **Q4.11** -- switch the `pyproject.toml` `gpu` extra from the
placeholder `cupy` to `numba` (recommend yes). Awaiting RJC's go-ahead to start
Stage 4 (per Prompt 2).

### 2026-07-29 (Prompt 2 re-run: Q4.11 resolved; numba limitations assessed)

RJC confirmed Q4.11 (switch `gpu` extra cupy -> numba) and asked whether numba's
lack of dynamic parallelism and texture memory is a problem. Assessment: **no**.
- Dynamic parallelism (device-side kernel launches) is not needed -- dispatch is
  host-driven; the Voigt/Faddeeva is flat data-parallel over (sub-pixels x
  profiles) with host-side batching.
- Texture memory is not needed -- the small read-only Faddeeva coefficient tables
  belong in numba **constant memory** (`cuda.const.array_like`); there is no
  interpolation / 2D-locality use case. Existence proof: `context/voigt_gpu/` is
  pure numba and already reaches ~1e-15.
- Later GPU convolution (full-model on device) should be a direct-sum kernel
  (also pure numba), not cuFFT (numba doesn't wrap it) -- not a blocker, noted.
Folded into the doc: Task 4.2 (constant-memory, no texture/dyn-parallelism note)
and the Context GPU-stack note (cudatoolkit must be user-installed; gpu extra ->
numba). No remaining open queries; awaiting RJC's go-ahead to start Task 4.0.
