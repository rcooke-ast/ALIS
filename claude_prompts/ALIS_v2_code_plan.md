# Plan for the major code refactoring of ALIS 

## Goals

- Review the `ALIS/claude_prompts/context.md` file (particularly the `Context` Section) to have a greater understanding of the goals of the new version of ALIS.
- Review the `ALIS/doc/ALIS_workflow.md` file to have a greater understanding of the workflow of ALIS.
- Review all existing code in `ALIS/alis/` to understand the current version of ALIS.
- Review all `context.md` files in subdirectories of `ALIS/context/fitting_examples/` to understand real-world working examples of ALIS.
- Prepare a plan for the major code refactoring of ALIS, including an ordered list of tasks to be completed.  The plan should be written in the `Refactor Plan` section of `ALIS/claude_prompts/ALIS_v2_code_plan.md`.

## Tasks

1. Given the goals above, prepare a plan for the major code refactoring of ALIS, including an
ordered list of tasks to be completed.  The plan should be written in the `Code Plan` section
of `ALIS/claude_prompts/ALIS_v2_code_plan.md`. Before writing the plan, review the `Claude` section
of this document, paying particular attention to the `Context` section, and use that information
to inform your plan.  The plan should be written in a clear and concise manner, and should be easy
to follow. At this stage, just write a draft plan, and ask for feedback from RJC.  The plan should
be written in the `Code Plan` section of this document, and all queries you have should be written
in the `Queries` section of this document. We will make a second draft of the plan during the next
stage, after review by RJC.

2. **Generate minimal "unit" test examples, one per model function.** For every
`alfunc_<modelname>.py` currently supported by ALIS that is not yet exercised by
an existing *tested* example, create a small, fast example under `ALIS/examples/`
following the established convention (a `generate_spectra.mod` to synthesise the
data and a `fit_spectra.mod` to fit it, plus the golden `.mod.out.reference` and
`data/reference_fits/<dataname>_fit.dat` files). Aim for at least one test per
supported model function, keeping each example quick to run.

   - **Currently uncovered (6) — need new examples (see Template 1 below):** `brokenpowerlaw`,
     `chebyshev`, `polynomial`, `powerlaw`, `random`, `tophat`.
   - **Currently uncovered (3) — need new examples (see Template 2 below):**
     `apod`, `voigtconv`, `vsigma`.
   - **Currently uncovered (2) — need new examples (see Template 3 below):**
     `lineemission`, `thar`.
   - **Already covered by tested examples:** `constant`, `gaussian`, `legendre`,
     `linear`, `phionxs`, `spline`, `splineabs`, `variable`, `voigt`, `vfwhm`,
     `afwhm`, `lsf`, `lsffile`, `multivfwhm`.
   - **Special cases:** `lsfspline` is used only by the `lsf_spline` example,
     which is currently excluded from testing (see Q21). `alfunc_base` (abstract
     base) and `alfunc_user` (user-defined-function template) do not need a
     standalone example.

>**Template 1:** Please use the `fit_spectra.mod` and `generate_spectra.mod` files in the
`examples/metal_line_abs/` example as a template for the new files, where the
new model to be tested will be used as the emission component (instead of `legendre`).

>**Template 2:** Please use the `fit_spectra.mod` and `generate_spectra.mod` files in the
`examples/metal_line_abs/` example as a template for the new files, where the
new model to be tested will be used as the `vfwhm` model in the resolution argument of the
data block.

>**Template 3:** Please use the `fit_spectra.mod` and `generate_spectra.mod` files in the
`examples/emission_line_ratio/` example as a template for the new files, where the
new model to be tested will be used as the emission component (instead of `gaussian`).
To make things simpler, we don't need to replace both `gaussian` functions, just the `wave=6563.0` one.

3. I have generated several reference files for the real world examples in `context/fitting_examples/`.
When building a test set, please also use these reference files to test the working version of ALIS.
Note that some data files are located in multiple directories, usually prefixed with `../data*/` relative
to the locations of the `.mod` file. Finally, note that there are several reference files that should
be tested if they exist:
  - the `.mod.out.reference` file.
  - the `data/reference_fits/<dataname>_fit.dat` best-fit data and model files
  - the `.covar.reference` file (if it exists) in the `model/` directory of the example. This is the output covariance matrix.
Please check that everything is in place. You may need to update the `Staged Task List` section below.

## Claude

### Skills

Consider using the skills in `ALIS/.claude/skills/` for any task that would benefit.

### Context

When developing new code, use the context provided in `ALIS/context/`. This includes:

- The most important context is provided by example model fits themselves and their outputs. These are provided in `ALIS/context/fitting_examples/`. There are multiple `context.md` files in many of the subfolders of `fitting_examples/` that describe the context of the example fits. Read those to obtain a greater understanding of the context of the example fits. The new version of ALIS should successfully run on these examples and produce similar outputs.
- Information in any of the latex files provided in the `ALIS/context/publications_using_alis/` folder.
- An example of several scripts and files that demonstrate a Voigt profile computation on a GPU in the `ALIS/context/voigt_gpu/` folder.
- An example of latex documentation provided by an independent Voigt profile fitting package is provided in `ALIS/context/VPFIT_doc/`.

You should also include the following information for context:

- Review the `ALIS/claude_prompts/context.md` file (particularly the `Context` Section) to have a greater understanding of the goals of the new version of ALIS.
- Review the `ALIS/doc/ALIS_workflow.md` file to have a greater understanding of the workflow of ALIS.
- Review all existing code in `ALIS/alis/` to understand the current version of ALIS.


## Coding

Here are guidelines for coding: 

- Use Python where possible
- The main files used by the code are in `ALIS/alis/`
- Add inline comments to explain the effort
- Reuse existing code when possible
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods
- Limit lines of code to 80 characters per line
- Add "Generated by RJC and Claude" in the doc string of all methods you create

## Code Plan

### Guiding principles

These principles apply to every task below and are the reason for the ordering:

1. **Feature parity first, always.** The entire current functionality must be
   preserved. Every example in `context/fitting_examples/` and `examples/` must
   continue to produce the same result (within an agreed tolerance) at every
   step of the refactor.
2. **Test before you touch.** No architectural change lands before a regression
   test exists that would catch a behavioural change. The existing example fits
   are our ground truth and become "golden" regression tests.
3. **Incremental, reversible steps.** Prefer many small, independently reviewable
   changes over a single "big-bang" rewrite. Keep the code runnable at every
   commit.
4. **Refactor, then optimise, then extend.** Clean the architecture first (so the
   code is testable and modular), then add performance features (caching, GPU),
   then add new user-facing capabilities (YAML/TOML, GUI, docs).

### Staged task list

Once this draft is finalised, Claude will generate one prompt file per stage by
copying the provided template `refactor_code_stageX.md` to
`refactor_code_stage<N>.md` (where `<N>` is the stage number) in
`ALIS/claude_prompts/`, and filling in a detailed breakdown of that stage's
subtasks. Per-stage work is logged in
`claude_prompts/logs/refactor_code_stage<N>_log.md`.

#### Stage 0 — Safety net and regression harness (do this first)

0.1 Inventory every example in `examples/` and `context/fitting_examples/`.
    Record for each: runtime, whether it involves randomness (`sim`, `random`),
    and its key output numbers (best-fit parameters, χ², DOF).
0.2 Build a regression-test harness driven by RJC's **reference files** — the
    golden outputs generated from the *current* code, which must never be
    overwritten or changed:
    - `<model>.mod.out.reference` sits in the same directory as its `.mod` file
      (the example's `model/` folder). A `.mod` file is part of the test suite
      **only if** its `.mod.out.reference` exists alongside it; otherwise it is
      skipped.
    - `<data>_fit.dat` golden fit files live in a `reference_fits/` folder
      inside the data directory. Note a fit may read data from several
      directories (referenced as `../data*/` — e.g. `data/` and `data_hrdx/` for
      `DH/J1358p6522`, or `datafit_orders/` for `DH_orders`); each such data
      directory has its own `reference_fits/`.
    - `<model>.covar.reference` (present for 13 of the 16 `context/fitting_examples/`
      fits) is the golden covariance matrix, sitting next to the `.mod`.
    Test cases are discovered by the presence of a `.mod.out.reference` (the fit
    `.mod` filenames vary — `*_FINAL_MODEL.mod`, `*_converge_newstart76.mod`,
    etc. — so do not rely on a name pattern). The harness runs only the fit
    models (`run_alis <fit>.mod`) — the input data already exists, so
    `generate_spectra.mod` is *not* run at test time — and compares the resulting
    `.mod.out`, `_fit.dat`, and (if a `.covar.reference` exists) `.covar` against
    the references.
0.3 Split the suite into **fast** (seconds, run on every change) and **slow**
    (minutes+, e.g. `DH_orders`, Monte-Carlo runs; run nightly / on demand).
    Randomised examples (`sim`, `random`) use a fixed seed for determinism.
0.4 Apply the agreed tolerances:
    - **Primary gate (every subtask):** the model spectrum and χ² evaluated at a
      *fixed* parameter set must reproduce the reference (χ² within 0.1%
      relative). This is the sensitive, stable check for refactor regressions.
    - **Secondary check (end-to-end):** after a full re-run of the minimiser, the
      best-fit parameters must fall within 10% of their 1σ error and χ² within
      1%.

#### Stage 1 — Low-risk modernisation (mechanical, safe)

1.1 Remove Python 2 compatibility cruft: `from __future__ import ...`,
    `try: input = raw_input`, and related stubs across all modules.
1.2 Add `pyproject.toml` (PEP 517/518/621), replacing `setup.py`; pin the minimum
    Python version to **3.13** (enabling stdlib `tomllib` and modern type-hint
    syntax).
1.3 Add `.pre-commit-config.yaml` (ruff / black / isort) and a CI workflow
    (GitHub Actions) that runs the fast regression suite + linting on every push.
1.4 Establish semantic versioning and start a `CHANGELOG.md`.

#### Stage 2 — Core architecture refactor (the heart of the work)

2.1 **Config as dataclasses.** Replace the nested string-keyed dictionaries
    (`argflag`, `modpass`, `fdict`) with typed `dataclass`es, introduced at the
    loading boundary in `alload.py`. Add type annotations as each structure is
    converted.
2.2 **Model-function registry with single instantiation.** Load each `alfunc_*`
    class exactly once at start-up into a registry, and reuse the instances for
    the whole fit (removing the repeated `alfunc_base.call(...)` re-instantiation
    seen in `alis.py`, including inside the `sim` loop).
2.3 **Remove `ClassMain` / stop passing `self`/`slf`.** Extract the model
    evaluation (`model_func`, `myfunct`) into standalone functions/objects that
    take explicit, typed state (e.g. a `Model` and a `FitState`) instead of a
    monolithic instance. This directly eliminates the `myfunct_wrap` circular
    import (a fresh `ClassMain` is currently built every χ² iteration via
    `instance.__dict__.update(fdict)`).
2.4 **Fix multiprocessing state passing.** Replace the `fdict` __dict__ copy with
    an explicit, picklable state object passed to worker processes.
2.5 **Replace `almsgs` with `logging`.** Swap `almsgs.msgs()` for Python's
    standard `logging` (levels, handlers, formatters), preserving current
    message content so example output/logs remain recognisable.

*Each subtask in Stage 2 is gated by the Stage 0 regression suite: output must
match the reference files within the agreed tolerances.*

#### Stage 3 — Fitting engine improvements

3.1 **Profile caching / memoization.** Cache model components whose parameters
    are unchanged between iterations (fixed and tied components) and recompute
    only when inputs change.
3.2 **Fit diagnostics / residual analysis.** Add a post-fit "fit quality report":
    per-region reduced χ², flagging of poorly-fit intervals, and residual
    summaries.
3.3 **Convergence robustness.** Formalise and extend the multi-start / random
    restart checks (currently in `alsims.py` and the external Monte-Carlo
    workflow) so that fits with hundreds of parameters can be shown not to
    "remember" their starting values.

#### Stage 4 — GPU support and modularity

4.1 **Formalise the CPU/GPU model interface** in the (now clean) `alfunc_base`:
    a clear `call_CPU` / `call_GPU` contract with a numerical-equivalence
    requirement.
4.2 **Implement a GPU Voigt profile** using the `context/voigt_gpu/` example
    (Faddeeva function via numba/CuPy — see Query 4), validated bitwise-close
    against the CPU version.
4.3 **Multiprocessed GPU dispatch** in the model-evaluation loop, with a clean
    fallback to CPU when no GPU is available.
4.4 **Modularity / new-function ergonomics.** Make it straightforward to add a
    new `alfunc_*` with both CPU and GPU paths plus its own unit tests.

#### Stage 5 — Data and I/O modernisation

5.1 **YAML/TOML model files.** Add YAML/TOML parsing/writing *alongside* the
    current text format (never removing text support).
5.2 **Atomic data modernisation.** Replace/augment `atomic.xml` with a
    human-readable, whitespace-aligned plain-text table (ECSV is a leading
    candidate, since `astropy` is already a dependency) that needs no manual
    `nrows` maintenance and makes it obvious which rows are present.
5.3 **Plotting-script output.** Add an option to emit standalone matplotlib
    scripts that reproduce publication-quality figures.

#### Stage 6 — Usability, GUI, and documentation

6.1 **CLI modernisation.** Replace the custom argument parser with `argparse`
    (or `typer`) for self-documenting `--help`, tab completion, consistent options.
6.2 **GUI.** Extend the existing `prepfit` GUI into a single interface that
    prepares, runs, and inspects a fit iteratively (see Query 8).
6.3 **Documentation.** Move from LaTeX/PDF to Sphinx + ReadTheDocs, using the
    *code* as source of truth; include a full tutorial and the expanded example
    suite. Add `CONTRIBUTING.md`.

### Dependency summary

- Stage 0 gates everything (it is the safety net).
- Stage 1 is independent and can proceed in parallel with Stage 0.
- Stage 2 depends on Stage 0.
- Stage 3 and Stage 4 depend on Stage 2 (they need the clean, testable core).
- Stage 5 and Stage 6 depend on Stage 2 but are otherwise independent of 3/4.

**Dependency policy (per Queries 5 and 12):** no new *runtime* dependencies
without prior discussion with RJC. Dev/build tooling (`pytest`, `ruff`, `black`,
`pre-commit`, Sphinx) and optional extras (GPU: `numba`/CuPy; YAML: `PyYAML`;
TOML writing: a small writer lib) are acceptable, as they are not imposed on
ordinary CPU users.

## Queries

The following points would materially change the plan; RJC's responses will
shape Draft 2.

1. **Regression tolerance.** For "unchanged output", is bitwise-identical
   required, or a numerical tolerance (e.g. best-fit parameters within some
   fraction of their 1σ error, χ² within a small relative tolerance)? What
   tolerance is acceptable? For examples that use randomness (`sim`, `random`),
   how should they be made deterministic for testing (fixed seed)?

**Response:** For "unchanged output", a numerical tolerance is acceptable.
The best-fit parameters should be within 1% of their 1σ error, and χ² should
be within 0.1% relative tolerance. For examples that use randomness, a fixed
seed should be used to make them deterministic for testing.

2. **Landing strategy.** Should the refactor land incrementally on `alis_v2`
   through many small PRs (strangler-fig, old and new code coexist behind a
   switch), or is a parallel rewrite preferred? My recommendation is the
   incremental approach so the code stays runnable and tested throughout.

**Response:** Yes, the incremental approach is preferred. RJC will perform all git
operations. Once each stage and subtask within each stage is completed, RJC will
create a new branch to work on the next subtask. Once all subtasks within a stage
are completed, RJC will merge the branch into the `alis_v2_stageX` branch
(where X is the stage number). Once all stages are completed, RJC will merge
the `alis_v2_stageX` branches into the `alis_v2` branch.

3. **`.mod` / `.mod.out` compatibility.** I assume the new version must read all
   existing `.mod` files unchanged. For `.mod.out`, is *semantic* equivalence
   enough, or must the output remain byte-for-byte compatible with the current
   format?

**Response:** The `.mod.out` files should be logically equivalent to the current version.
Each should be possible to use the `.mod.out` file as input to an independent run of ALIS.
The `.mod.out` files should also contain a commented out copy of the original input file.
The `.mod.out` cannot be used as input in cases where there are blinded values. We can
discuss the best approach during the relevant phase of the process.

4. **GPU stack.** The old commented-out code used PyCUDA; the `voigt_gpu`
   example uses numba. Which do you want to target — CuPy, `numba.cuda`, or
   PyCUDA? And should GPU be an optional install extra so CPU-only users need no
   CUDA toolchain?

**Response:** GPU should be an optional install extra so CPU-only users need no CUDA toolchain.
Please consider using the `context/voigt_gpu/` example as a starting point for the GPU implementation.
We can discuss the best GPU stack during that phase of the process.

5. **Structured-config library.** For replacing the nested dicts, do you have a
   preference among stdlib `dataclasses`, `attrs`, or `pydantic`? My
   recommendation is stdlib `dataclasses` (no new hard dependency), adding
   optional validation only at the input boundary.

**Response:** Unless there is significantly better performance of other advantages,
please do not add additional dependencies. If additional dependencies are required,
please discuss with RJC, explaining the reasons for requiring a new dependency,
before adding them.

6. **Minimum Python version.** What is the floor? This affects type-hint syntax,
   `tomllib` availability (3.11+), and other features.

**Response:** Please only include support for python v3.13 and higher.

7. **Atomic data format.** Do you have a preferred replacement for `atomic.xml`
   (e.g. a plain FITS table, VOTable kept but auto-`nrows`, HDF5, or an
   `astropy`-readable ECSV)? The key requirement is removing the manual `nrows`
   maintenance step.

**Response:** Let's discuss these options during the relevant phase of the process.
The key requirement is removing the manual `nrows` maintenance step, but to allow
some simplicity in adding new atomic data when needed. This ideally should be a
human-readable text file that can be opened in a text editor, where it is obvious
what rows are included in the file. The format should be easy to read and understand,
and should be easy to add new atomic data when needed. The format should also be
easy to parse and read into the code. The format should also be easy to write out
from the code. The format should also be easy to validate and check for errors.

8. **GUI toolkit.** What toolkit does the current `prepfit` GUI use (Qt / Tk /
   matplotlib)? Should the new GUI stay in that toolkit, or is a move (e.g. a
   web-based interface) acceptable?

**Response:** `prepfit` currently uses matplotlib with the `Qt5Agg` backend.
At a later stage in the process, we will discuss the best toolkit to use for the GUI.

9. **Priority.** Among the phases, which delivers the most value first for you —
   the architecture cleanup (Phase 2), GPU speed-ups (Phase 4, implied by the
   `DH_orders` example), or the usability/GUI/docs work (Phase 6)? This will not
   change the *dependency* ordering, but it tells me where to concentrate effort.

**Response:** At this stage, spend the most effort on Phase 2 (architecture cleanup).
We can then later decide how much effort to spend on the subsequent stages.

---

### Follow-up queries and notes (Draft 2 review)

Having read RJC's responses and the reordering, here are follow-up points. Once
these are resolved I will fold all resolved answers into a clean Draft 2 (e.g.
replacing the "see Query N" pointers with the agreed tolerance, Python 3.13+,
etc.).

**Review of the reordering — one concern, otherwise agreed.**

- *Stage 2 (logging ↔ multiprocessing swapped):* Agreed. One thing to flag:
  step 2.3 (remove `ClassMain`) and step 2.5 (fix multiprocessing state passing)
  are the *same underlying mechanism* — `myfunct_wrap` rebuilds a `ClassMain`
  from `fdict` precisely to get state into worker processes for pickling. So 2.3
  is not truly complete until 2.5 is done; I suggest treating 2.3 + 2.5 as a
  coupled unit. Doing logging (2.4) between them is harmless, but it means the
  logging design must be multiprocessing-aware from the start (log records must
  survive being emitted inside worker processes).
- *Stage 3 (diagnostics ↔ convergence swapped):* Agreed, no concern — the two
  are independent, and having the diagnostics tooling (3.2) in place first is
  actually useful when validating the convergence work (3.3).

**Response:** Agreed. I have reordered to the original order, so that the logging
work is done after the multiprocessing work.

**Q10 — "Phase" vs "stage" terminology and generated filenames.** The plan uses
"Phase N", but the responses and template use "stage": the template is
`refactor_code_stageX.md`, branches are `alis_v2_stageX`, and its logs go to
`claude_prompts/logs/refactor_code_stageX_log.md` — yet the note in the Phased
task list says to generate `refactor_code_phaseX.md`. Are "phase" and "stage" the
same thing? When I copy the template, should the copy be named
`refactor_code_stage0.md` (matching the template/branch/log convention) or
`refactor_code_phase0.md`? My recommendation is to standardise on **"stage"**
everywhere, so the per-phase prompt file, branch, and log names all agree.

**Response:** Yes, please standardise on "stage" everywhere. There should be no
mention of "phase" anywhere in the code plan.

**Q11 — What the regression tolerance is measured on.** The "within 1% of the 1σ
error" parameter tolerance and 0.1% χ² tolerance: should these be checked on
(a) the final best-fit parameters after a full re-run of the minimiser, or
(b) the model spectrum and χ² evaluated at a *fixed* parameter set? Concern: a
full re-fit has its own stopping criteria (`chisq atol`, `ftol`), so two runs of
*identical* code can land more than 0.01σ apart, which would make a 1%-of-1σ gate
flaky. Evaluating the model and χ² at a fixed parameter set is far more sensitive
and stable for catching refactor regressions. Proposal: gate each refactor step
primarily on fixed-parameter model/χ² equivalence, and apply the (looser)
parameter/χ² tolerances to the full-fit endpoints.

**Response:** The regression tolerance should absolutely be accepted at a fixed parameter set.
It would also be ideal that the final best-fit parameters after a full re-run of the minimiser
are within 10% of the 1σ error and 1% χ² tolerance.

**Q12 — Scope of "no new dependencies".** Does this apply to *runtime*
dependencies only? The plan needs dev/build tooling (`pytest` for Phase 0;
`ruff`/`black`/`pre-commit` for 1.3; Sphinx for 6.3) and optional extras
(`numba`/CuPy behind the GPU extra). Are these acceptable as **dev-only /
optional-extra** dependencies that ordinary CPU users never install? Separately:
YAML model-file support (5.1) needs `PyYAML` at runtime for users who choose YAML;
TOML *reading* is stdlib (`tomllib`, fine on 3.13) but TOML *writing* needs a
small lib (e.g. `tomli-w`). Unless you say otherwise, I will treat runtime
dependencies as strict "no new deps without discussion" and treat dev/optional
extras as acceptable.

**Response:** The proposed dev/build dependencies are acceptable as
dev-only / optional-extra dependencies that ordinary CPU users never install.
If any new runtime dependencies are required, please discuss with RJC, explaining
the reasons for requiring a new dependency, before adding them. New runtime
dependencies that optimise and simplify aspects of the code, are allowable,
but please check with RJC first.

**Q13 — Atomic-data candidate (for Phase 5.2, not blocking).** Since `astropy` is
already a dependency and you want a human-readable, easily-edited, easily-validated
plain-text table with no manual row count, is astropy's **ECSV** format (a
self-describing YAML header over plain-text rows) an acceptable candidate to
evaluate when we reach that phase? Just confirming a direction, not asking to
decide now.

**Response:** ECSV is an acceptable candidate to evaluate when we reach that phase. We can discuss
the best approach during the relevant phase of the process. If possible, a format that contains
extra whitespace on each row so that individual columns are clearly separated and easy to read
is preferred.

---

### Follow-up queries (Draft 2 → reference files)

These concern the reference-file convention described in Prompt 3, so the Stage 0
harness reads and pairs the golden files correctly.

**Q14 — `_fit.dat.reference` location.** Prompt 3 says these "appear in the
`examples/*/data/` directories" but also that "these will be located in a
`reference_fits/` folder ... in the `data/` folder". I have taken the
authoritative location to be **`<example>/data/reference_fits/`**. Please confirm.

**Response:** That is correct. I noticed a mistake in my previous response. The
`_fit.dat.reference` files will actually be called `_fit.dat` and are located in the
`<example>/data/reference_fits/` folder.

**Q15 — the `_fit.dat` test-suite rule.** For `.mod.out` the rule is clear: no
`.mod.out.reference` next to a `.mod` ⇒ that `.mod` is skipped. The corresponding
sentence for `_fit.dat` ("If a reference file does not exist in the same directory
as a `_fit.dat` file, then this ... should also be used as part of the test
suite") reads as garbled and seems to contradict the `.mod.out` rule. Also, if the
`_fit.dat.reference` lives in `data/reference_fits/`, it is *not* in the same
directory as the `_fit.dat`. What is the intended rule for when a `_fit.dat`
comparison is included, and how should a `_fit.dat` be paired with its
`_fit.dat.reference`?

**Response:** You are correct. My response was garbled. The intended rule is that 
if a `_fit.dat` file exists in the `data/reference_fits/` folder of an example,
then it should be used as part of the test suite. The `_fit.dat` file should be paired
with its corresponding `_fit.dat` file in the `data/` folder.

**Q16 — naming and pairing.** Please confirm the exact reference filenames so the
harness can pair them unambiguously: `<modelname>.mod.out.reference` alongside
`<modelname>.mod`, and `<dataname>_fit.dat.reference` inside
`data/reference_fits/`. For examples containing several `.mod` and/or data files,
is pairing done purely by matching the base filename?

**Response:** Yes, the exact reference filenames are as follows:
- After running the input file `<modelname>.mod` through ALIS, it will produce an output file `<modelname>.mod.out` that should be compared against the reference file `<modelname>.mod.out.reference`.
- After running the input file `<modelname>.mod` through ALIS, for each input data file called `<dataname>.dat` ALIS will produce an output file `<dataname>_fit.dat` that should be compared against the reference file `<dataname>_fit.dat` located in the `data/reference_fits/` folder.

**Q17 — template title (minor).** The template `refactor_code_stageX.md` is titled
"PHASE X" internally. When I generate the per-stage copies, shall I rename that
heading to "STAGE X" for consistency with the agreed terminology?

**Response:** I have just made these changes to the template file, and it should not show any mention of "phase" anywhere.

---

### Follow-up queries (Draft 2 → reviewing the `metal_line_abs` example)

I inspected `examples/metal_line_abs/`. The layout matches the agreed convention
and looks sensible:

- `model/`: `fit_spectra.mod`, `fit_spectra_linear.mod`, `fit_spectra_thermal.mod`
  each have a matching `<name>.mod.out.reference`, so all three are in the suite.
  The two `generate_*.mod` files have **no** reference, so they are correctly
  excluded (they generate data rather than fit).
- `data/reference_fits/` holds the golden `OI_SiII_fit.dat` and
  `OI_SiII_thermal_fit.dat`, paired by data-file name as per Q16.

Two points to resolve before generating more examples:

**Q18 — Two models that share one data file collide on the fit reference.** Both
`fit_spectra.mod` and `fit_spectra_linear.mod` read `OI_SiII.dat`, so both write
`data/OI_SiII_fit.dat`, but there is only one golden
`reference_fits/OI_SiII_fit.dat`. Consequences: (a) running the suite, the two
fits overwrite each other's `OI_SiII_fit.dat`; (b) the single reference is
ambiguous as to which model owns it. The `.mod.out` check is unaffected (distinct
filenames), and the *primary* fixed-parameter gate is fine (both give the same
spectrum at identical parameters), but the `_fit.dat` pairing needs a rule. Options:
key fit references by model (e.g. `reference_fits/<modelname>/<dataname>_fit.dat`,
or `reference_fits/<modelname>__<dataname>_fit.dat`); or designate one model per
data file as the `_fit.dat` owner and validate the others by `.mod.out` only. How
would you like to handle this?

**Response:** Well-spotted! The best approach is to have one `.dat` file per model. This will avoid any collisions and make it clear which model owns the fit reference. The `_fit.dat` files should be named according to the model that produced them, so that each model has its own unique `_fit.dat` file. This will also make it easier to validate the fits for each model independently. I will make these changes now while generating all possible examples.

**Q19 — Volatile header lines in `.mod.out`.** The `.mod.out` header contains lines
that change on every run — `Generated by ALIS on <date> at <time>`,
`Running Time (hrs)`, and `Num. of Iterations` (the last may also shift slightly
after a refactor). So the harness cannot byte-diff `.mod.out` against
`.mod.out.reference`; it must parse the numeric fields (best-fit parameters, χ²,
DOF) and compare them within the agreed tolerances, ignoring the volatile
comment lines. I will implement it that way unless you prefer otherwise — just
confirming the intent.

**Response:** Yes, that is correct. The harness should parse the numeric fields and compare them within the agreed tolerances, ignoring the volatile comment lines. Parsing the information is required, because the tolerances are based on the numeric values in combination with the 1σ error.

---

### Follow-up queries (Draft 2 → reviewing all `examples/` reference files)

I checked every example under `examples/` (except `prepfit` and `lsf_spline`).
**All models are suitable:** the Q18 fix is applied throughout — each fit model
reads a uniquely-named data file, so every produced `<dataname>_fit.dat` has a
matching, non-colliding reference in `data/reference_fits/` (e.g. `metal_line_abs`
and `splineabs` now have a separate `OI_SiII_linear.dat`). Each `generate_*.mod`
correctly has no `.mod.out.reference` and is excluded from the fit suite. The new
`generate` example is correct: its data-block filename `OI_SiII.dat` matches the
golden `data/reference_fits/OI_SiII.dat`, with no `_fit` suffix.

I also mapped every supported `alfunc_*` against the examples and proposed a new
Task 2 (above) to fill the 11 gaps. Three queries on that:

**Q20 — Seeding randomness.** The `generate` reference (and a future `random`
unit example) depend on noise, so the golden files are only reproducible with a
fixed RNG seed. How is the seed fixed — a setting inside the `.mod` file, a global
default in `settings.alis`, or something the harness controls? I want the unit
examples for `generate`/`random` to be deterministic against their references.

**Response:** Good point. The `generate peaksnr -1` command ensures that the spectra are generated noise free. The `random` function is more difficult to guard against, however, I suspect the minimiser should converge to the same solution, if the random window is not too large.

**Q21 — `lsfspline` coverage.** The only `lsfspline` example is `lsf_spline`,
which you have excluded from testing for now. Should Task 2 add a small *tested*
`lsfspline` unit example (so the function is covered), or will `lsf_spline` gain
reference files later? Same question applies in principle to `prepfit` (though
that is a GUI tool, so probably out of scope for the fit suite).

**Response:** The `prepfit` test is out of scope for the fit suite. The `lsfspline` function takes a long time to compute, and is not stable at present. We should ignore testing this function for now, and I will test it separately once the refactor is complete.

**Q22 — Unit-example pattern.** Please confirm the unit examples should follow the
full generate→fit pattern (a `generate_spectra.mod` + a `fit_spectra.mod`, with
both `.mod.out.reference` and a `data/reference_fits/<dataname>_fit.dat`), kept
deliberately tiny for speed. Some functions are convolution/instrumental
(`apod`, `voigtconv`, `vsigma`) and will need a host line (e.g. a `voigt` or
`gaussian`) to convolve; others are emission (`powerlaw`, `brokenpowerlaw`,
`polynomial`, `chebyshev`) or special (`random`, `thar`, `tophat`, `lineemission`).
I will choose a minimal valid model for each unless you would like specific setups.

**Response:** I have provided instructions in Task 2 for the templates to use for each function.

---

### Task 2 outcome (unit examples) — 5 created, 6 functions found broken

Built the unit examples per Task 2. **Five now exist with references**
(`examples/{powerlaw,chebyshev,polynomial,brokenpowerlaw,tophat}/`), each
converging to reduced χ² ≈ 0.95–1.18.

**Six functions cannot produce a reference because they are broken/unsuitable in
the *current* ALIS** (I removed the partial dirs; all are recreatable once fixed):

- `vsigma` — `alfunc_vsigma.py:52` `np.zeros(2*df+1)`, `df` is a float → TypeError.
- `voigtconv` — `call_CPU()` lacks the `mkey` kwarg that `alis.py:380` passes.
- `apod` — requires `kind=`, then `load()` `IndexError` at `alfunc_apod.py:191`.
- `thar` — drops into an IPython `embed()` during model prep and hangs.
- `lineemission` — needs `ion=<el>_<stage>_<wavelength>`; then `voigtking`
  (`alfunc_lineemission.py:73`) raises "only 0-dimensional arrays can be
  converted to Python scalars".
- `random` — variable-type (`call_CPU` returns `None`, needs `command=`); not an
  emission drop-in for `legendre`.

**Q23 — how to proceed with the 6 broken functions?** These are exactly what the
test suite should protect, but no golden reference can be made from code that
crashes. Options: (a) leave them out of the suite for now and add examples as each
is repaired during the refactor (fixing them then becomes a tracked Stage 2/other
subtask); or (b) treat "does not crash" as the initial test and add numerical
references once fixed. My recommendation is (a). Either way, these six are
flagged as pre-existing bugs to fix.

**Response:** I have resolved several bugs in these functions in ALIS, and the following functions
are now working correctly: `vsigma`, `voigtconv`, `random`, `lineemission`. I have generated the examples
and created all of the reference files for these functions. The `apod` and `thar` functions are still broken,
and so the following functions can be ignored for the following reasons:
- `apod` — this function requires frequency units for the input.
- `thar` — this function is not currently supported.

## Prompts

1. Read this doc, and execute Task 1.

2. Read this doc. I have responded to the queries. Please ask more queries if needed. I have also made slight changes to the order of some tasks that I think are slightly more important or should be executed in a different order. Please review the changes and let me know if you have any questions or concerns.

3. Read this doc. I have responded to the queries. Please ask more queries if needed. I will generate a series of `.mod.out` files based on the current version of ALIS. I will call these files the "reference files", and they will be given extension names `.mod.out.reference`. The reference files will appear in the `ALIS/examples/*/model/` directories and in the subdirectories of the `ALIS/context/fitting_examples/` directory. These golden files should not be overwritten/changed. They represent the optimal values that should be reproduced after refactoring the code at each stage and subtask. If a reference file does not exist in the same directory as a `.mod` file, then this means that this `.mod` file should not be used as part of the test suite. I will also generate a series of `_fit.dat` files based on the current version of ALIS. I will call these files the "reference files", and they will be given extension names `_fit.dat`. The reference files will appear in the `ALIS/examples/*/data/` directories and in the subdirectories of the `ALIS/context/fitting_examples/` directory. These golden files should not be overwritten/changed. They represent the optimal values that should be reproduced after refactoring the code at each stage and subtask. If a reference file does not exist in the same directory as a `_fit.dat` file, then this means that this `_fit.dat` files that should also be used as part of the test suite. These will be located in a `reference_fits/` folder, which is located in the `data/` folder corresponding to the example. 

4. Read this doc. I have responded to the queries. Please ask more queries if needed. I have generated one example in the `ALIS/examples/metal_line_abs/` directory. Please see this for reference, and if this is sensible, I will generate additional examples.

5. Read this doc. I have responded to the queries. Please ask more queries if needed. I have now generated reference files for all examples in the `ALIS/examples/` directory (except the `prepfit` and `lsf_spline` examples, which do not need to be tested at this time). I have also included a new example that tests the `generate` functionality (Note that because this is not fitted, it does not have an `_fit` suffix. The filename listed in the data block of `generate_spectra.mod` is the same as the filename provided in the `data/reference_fits/` directory). Please check that all models are suitable. Is it possible for Claude to generate additional examples in the `ALIS/examples/` directory before finalising the test-suite? These "unit" examples are quick to run, and we should aim to have one test for each `alfunc_<modelname>.py` function currently supported by ALIS. If so, please add a new task in the `Tasks` Section above for Claude to generate these examples. Do not create these examples yet - RJC will review the task before proceeding.

6. Read this doc, and particular my responses to the queries. Please ask more queries if needed. If everything is clear, please proceed to execute Task 2. We will not 

7. I received an `API Error: Unable to connect to API (ConnectionRefused)` during the last prompt. Please retry the last prompt, or resume from the point before you were interrupted.

8. Read this doc, and particular my responses to the queries. Please ask more queries if needed. I have responded to your most recent query, and generated new references in the `examples` directory. Please check they are correctly formatted and named.

9. Please check if you have any more queries, and that you have all the information you need to proceed with the next stage of the refactor. Please ask more queries if something is unclear. Please note that all data should already exist in the `examples/<testname>/data/` folders, therefore there is no need to run `generate_spectra.mod` for the tests each time. We should only run `run_alis fit_<example_name>.mod` and test the `.mod.out` file, the `.covar` files (if they exist) and the output `_fit.dat` files. If everything is in place for the `examples/`, please proceed to execute Task 3.

[//]: # (COMMENT:
During execution of Prompt 9, Claude returned the following question:
I've generated refactor_code_stage0.md. How would you like to proceed with "Task 3" / the next stage?
MY RESPONSE: Before proceeding, please run Task 3, and assess if the refactor_code_stage0.md needs to be
rewritten/checked. If there are no further queries on the content in ALIS_v2_code_plan.md, then please
proceed to write all stage files 1-6 with any queries relevant to those stages in the          
corresponding file.
)

10. Currently, Stages 1-3 of the refactor have been completed, and we're about to complete Stage 4.
However, before proceeding with Stage 4, I have noticed a possible issue that might need to be
resolved that may improve the performance of the code (both CPU, and particularly the GPU).
In `minimise.alfit()`, the `call()` function is calling a generic function (`fcn`) that is
external to the `minimise()` class. In principle, we could build in the function inside the
class (since it never changes), and it would inherit the properties of the class without
needing to pass them through the function. Related to this, the functions that eventually
get called (`myfunct`, and subsequently `model_func` and `model_func_ddp`) are locked in
the main ClassMain class. I think we should take a step back and consider the best approach
to refactor this part of the code to improve performance. For example, in `model_eval`, one
of the first things that is done is a call to renew_subpix. This could in principle be done
once per iteration inside the minimise class, rather than being carried out each time the
two-sided derivative is calculated. Please take a holistic view of this part of the code,
and consider the best approach to refactor it to improve performance, now with the appreciation
that the call to alfit does not require the `fcn` variable. How would this change the way the
code is structured, and what would be the best approach to refactor this part of the code to
improve performance?

[//]: # (COMMENT:
During execution of Prompt 10, Claude returned the following questions:
  1. Where should this live? A dedicated pre-Stage-4 prompt doc (e.g. refactor_code_stage3p5_minimiser.md), or folded into Stage 4 as task 4.0 (the interface seam that 4.1–4.5 build on)? My lean: a small
  dedicated doc, since it's a Stage-3-style engine refactor that must land bitwise before GPU work.                                                                                                        
  2. Bitwise requirement. I assume this must stay bitwise-identical under the Stage 0 gate (as Stage 3.4 did) — confirm?                                                                                   
  3. Appetite. Pure structural cleanup that enables the GPU port (minimal risk, defer perf to Stage 4), or also bank the CPU wins now (renew_subpix hoist, drop the per-call copy)?                        
  4. _pinfl hoisting. Do you want me to investigate whether the influence table is truly value-independent (hoistable to once-per-fit)? It changes only if shift/velocity params move a region's coverage —
  worth confirming before relying on it.   
MY RESPONSE:
My responses to your queries are:
  1. Please fold this into a dedicated pre-Stage-4 prompt doc (e.g. refactor_code_stage3p5_minimiser.md), since it's a Stage-3-style engine refactor that must land bitwise before GPU work. The vision of this new document should be to make this an essential preparatory step for Stage 4, and to ensure that the code is structured in a way that will allow for the GPU port to be implemented with minimal risk. The document should also include any relevant queries and responses, as well as any additional information that may be useful for the refactor.
  2. Yes, the bitwise requirement must stay bitwise-identical under the Stage 0 gate (as Stage 3.4 did).
  3. Please focus on pure structural cleanup that enables the GPU port (minimal risk, defer perf to Stage 4), but if there are ways that improve CPU performance without complicating GPU implementation, then we should bank the CPU wins, as well.
  4. Yes, please investigate whether the influence table is truly value-independent (hoistable to once-per-fit, or once-per-iteration).
)
## Logging

Create a file called `ALIS/logs/ALIS_v2_code_plan_logs.md` to record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

Simple description of the work and what you learned

### <Date> (Short summary of the work)

Simple description of the work and what you learned

...
