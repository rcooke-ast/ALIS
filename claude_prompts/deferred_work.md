# ALIS v2 refactor — deferred work

Everything the Stages 0–6 refactor knowingly left unfinished, in one place, so
that picking it up later does not mean re-reading eight logs to work out what
was intended. Written 2026-08-05 at RJC's request (Stage 6, Q6.17/Q6.23).

Each entry says **what** is missing, **why** it was deferred, **where** the
original intent is recorded, and **what finishing it involves**. Where a
decision has already been taken, the query that took it is named — the queries
carry the reasoning, and the reasoning is usually the part that is expensive to
reconstruct.

**Conventions.** `Q<n.m>` is a query in `claude_prompts/refactor_code_stage<n>.md`;
the logs are `claude_prompts/logs/refactor_code_stage<n>_log.md`.

---

## Priorities

*Added 2026-10-20, after dashboard Stage 6 (v1 complete), at RJC's request. Drawn from
the sections below, the dashboard's "Later" items (`dashboard_stage0.md`) and what
each dashboard stage left out. The priorities are Claude's judgement, weighing
correctness first, then what users need next, then effort; RJC orders the work.
"D*n*", "F*n*" and "S*n*" are in `ALIS_v2_dashboard_prompts.md`.*

### High priority
1. **The Plot tab** (dashboard step 8; D28, D42, S22): **done** as the Figure tab,
   dashboard Stage 9 (2026-11-08, D102–D112), with an emitter of its own
   (`alis/dashboard/figscript.py`), whose presets were tried on RJC's D/H projects.
   `alis/plotscript.py`'s two D/H layouts, which serve `run_alis`, have still never
   run on real data (§1.1, §5).
2. **How ALIS draws `sim perturb` and `sim random`** (§7.6). This is a correctness
   question in ALIS itself. `simulate.perturb` ignores the best fit it is given and
   draws about the starting values, so the two ways of reaching it draw about
   different centres. The decision also settles whether a project file can run
   simulations, and the dashboard's Random restarts would follow it.
3. **Diagnose `examples/brokenpowerlaw`'s χ² mismatch** (§2.4: 374.98 against
   338.15). The suspected cause is that ALIS reports a rejected trial step when the
   fit stops on `atol`. If so, it would affect any model that stops on `atol`, not
   just this example. *Dashboard Stage 14 (Q14.13): the mismatch was the
   sub-pixels; `brokenpowerlaw` and `tophat` are back in the gate. Left: confirming
   the minimiser's stopping paths (Stage 15's candidate 1).*
4. **Sphinx/ReadTheDocs documentation and `CONTRIBUTING.md`** (§1.2). This was
   deferred until v2.0.0 is ready to release, which is now the next milestone.
   `doc/tex_files/` is out of date.

25. **Local minima, and how sensitive a fit is to its sub-pixels** (§1.5; added at
    dashboard Stage 14, Q14.10's response): RJC's "even bigger issue". With
    `nsubpixel_min` 10, three of the harness's fits reach a worse minimum from their
    own starting values. A series of its own, independent of the dashboard.

### Medium priority
5. **Wavelength-dependent resolution** (§2.1). Restore the per-pixel Gaussian
   convolution, with a test against a hand-computed result. A non-Gaussian line
   spread function that varies with wavelength, the real target, is a much larger
   project.
6. **Orders mode, PypeIt spec1d input and the stacked spectrum** (D29–D32, §7.4),
   with the tidy-up of `load_fits`'s format detection (§3.1) that FITS input needs.
7. **Re-attaching to a run** after the dashboard was closed, with its live progress
   (F6).
8. **Attaching a source spectrum to an imported fit** (S23). Without one, an
   imported fit cannot SNIP new transitions.
9. **Generating data from the dashboard** (§7.5). This needs `generate data` to
   write through a project file's outputs.
10. **Telluric flags** in the transition list (§7.1). A small change, useful for
    ground-based data.
11. **Two quick ALIS fixes:**
    - the first data line must carry `columns=`, otherwise ALIS fails with an
      unhelpful "Error reading in file" (§3.2);
    - Ctrl+C makes ALIS exit with status 0, so a script cannot tell a stopped fit
      from a finished one (§3.5).
12. **The 63 bare `except:` statements**, several in the file loaders (§4.2). Each
    can hide a real failure, and narrowing them needs tests of the failure paths
    first.

13. **A GPU evaluation kept on the device** (§1.4): the whole model evaluation on
    the GPU for the whole fit, explored as a series of its own that measures before
    it builds (dashboard Stage 11, Feedback point 1 and Q11.3). It speeds up the
    largest fits only, such as DH_orders; smaller fits are faster on the CPU.
26. **Three faults of the resolution functions' keywords** (§7.12; found in
    dashboard Stage 14, Task 14.1): `apod` cannot be evaluated at all, `lsf`'s buffer
    is that of COS G130M at 1309 Å whatever the line names, and a model loaded again
    starts each line from the last line's keywords. Offered to dashboard Stage 15
    (its items 12–14, Q15.4–Q15.6).

### Low priority
14. The link editor and its templates (S15). Links can already be typed in the
    `.mod` panel.
15. Command search (F10).
16. Reproducing an imported fit (S28).
17. Solar abundances in `atomic.ecsv`, replacing the two hard-coded copies (§7.3).
    This changes no behaviour.
18. Deciding whether to keep `systmodule=`, which nobody uses (§2.3).
19. Whether `msgs.bug` should stop the fit (§3.4).
20. The black/isort reformat and the remaining lint rules, then the black badge on
    the README (§4.1, §4.2, §6).
21. Testing gaps: the plotting code, GPU coverage, and the `context/` fits in CI
    (§5).
22. The metal example's junk last pixel (`examples/metal_line_abs/data/OI_SiII.dat`,
    flux −10¹⁰, outside the fit range), found in dashboard Stage 6.
23. Load snips… for a file that cannot be read or names no transition (§7.8), and
    Open a fit's question in the singular for a fit of one dataset (§7.9), both
    waiting for a user to ask (dashboard Stage 12, Q12.1 (b)).
24. ALIS's default `nsubmin` of 10 (§7.10), and `lsfspline` written into a
    `.mod.out` without its `locations`, with `examples/lsf_spline` as a regression
    case (§7.11): dashboard Stage 12, Q12.13 (b) and Q12.10's response. *§7.11 done
    in dashboard Stage 13; §7.10, with the sub-pixels a `.mod.out` is evaluated on,
    is the key item of dashboard Stage 14, next (Q13.7's response).* ***Done:** §7.10
    in dashboard Stage 14 (2026-11-25, D148–D152).*
27. **`examples/lsf_spline`'s data at a higher S/N** (§7.13; dashboard Stage 14,
    RJC's response to Q14.14): its fit cannot determine the smoothing it was made
    with, which failed CI's `examples` job under numpy 2.5.5 until the harness took
    an undetermined parameter as it comes (D153). Offered to dashboard Stage 15 (its
    item 16, Q15.8).

### Not planned, or entries to close
- **YAML/TOML model files** (§1.3): dropped by RJC (Q5.2/Q5.3).
- **Pausing a run**: declined by RJC (Q6.15(c) of `dashboard_stage6.md`).
- **The onefits menu** (§2.5): onefits has been removed and its code is gone, so
  the entry can be closed.
- **The GUI** (§1.1): built as the dashboard, the Plot tab (as the Figure tab)
  included since dashboard Stage 9 (2026-11-08); the entry can be closed.
- **`1e4`** (§3.3): a permanent limit of the model-file format, which now reports an
  error.

---

## 1. Large features deferred whole

### 1.1 The GUI (Stage 6.2, and 6.6 with it)
- **What:** grow `alis/prepfit/` (matplotlib, `Qt5Agg`) into one tool that
  prepares a model, runs the fit and inspects the result, iterating without
  leaving it. And, on top of that, **6.6**: generate the Stage 5.3 plotting
  script *interactively* once a fit has completed — pick the mode, adjust the
  panel selection and layout, write the script out.
- **Why deferred:** RJC, Q6.2 and Q6.4 — moved to a separate design document.
  The toolkit choice (stay with matplotlib/Qt, or move to something web-based)
  is open and was itself deferred from plan Q8.
- **Where:** `refactor_code_stage6.md` §6.2, §6.6, Q6.2, Q6.4.
- **To finish:** the emitter half of 6.6 is **done** — `alis/plotscript.py`, ten
  `plotscript` settings, ten unit tests in `tests/test_plotscript.py`. Only the
  GUI surface is missing. `alis/prepfit/specplot.py:390` `set_regions` was kept
  and repaired specifically for this (Q6.8/Q6.10).
- **Note:** two of the three `plotscript` `DH` layouts have never been exercised
  on real data — no model in the harness has exactly one or two datasets
  covering a Lyman series (stage 5 log, Task 5.3).

### 1.2 Sphinx / ReadTheDocs documentation (Stage 6.3)
- **What:** move from LaTeX/PDF to Sphinx + ReadTheDocs with the code as the
  source of truth, a full tutorial, the example suite and API autodoc; plus
  `CONTRIBUTING.md`.
- **Why deferred:** RJC, Q6.3 — until v2.0.0 is ready for deployment.
- **Where:** `refactor_code_stage6.md` §6.3, Q6.3.
- **To finish:** there is no `docs/` tree, no `.readthedocs.yaml` and no
  `CONTRIBUTING.md` yet. `doc/ALIS_workflow.md` is the current reference and is
  up to date as of Stage 6.7; `doc/tex_files/` is out of date and reference only.
  `run_alis --list-settings` (Stage 6.1, Q6.24) covers settings discovery in the
  meantime.

### 1.3 YAML/TOML model files (Stage 5.1)
- **What:** a second model-file format alongside `.mod`.
- **Why dropped:** RJC, Q5.2/Q5.3, with a worked comparison in the Stage 5
  appendix. YAML leaves the `1.0da` micro-syntax hand-parsed anyway, expands
  `DH_orders` from 1,252 to ~4,000 lines, and adds a second reader/writer pair
  to keep in agreement with the first.
- **Revisit if:** models need to be generated or consumed by other tools. The
  cheaper answer then is a documented `.mod` <-> dict API in `alis/load.py`,
  which the Stage 5.5 parser tests already pin down, rather than a second
  first-class format.

### 1.4 A GPU evaluation kept on the device (dashboard Stage 11, Feedback point 1)
- **What:** keep the whole model evaluation on the GPU for the whole fit. The
  fit's constant arrays are uploaded once, each evaluation sends only the
  parameters, and as little as possible comes back. RJC's Feedback point 1 of
  `dashboard_stage11.md` (2026-11-10), to be explored as a series of its own:
  `claude_prompts/gpu_stage0.md` first, then `gpu_stage1.md`, …, on its own branches
  (Q11.3 (b)).
- **Still deferred** at Stage 11's close (2026-11-13): it stays here, a series of its
  own, until RJC starts it (Q12.2 in `dashboard_stage12.md`).
- **Why deferred:** RJC, Prompt 2 of `dashboard_stage11.md` (2026-11-12): the GPU's
  tests and speed-ups wait, and the stage takes the dashboard's feedback.
- **Where:** Q11.3 of `dashboard_stage11.md` (the review, against refactor Stage 4's
  measurements); `refactor_code_stage4.md`, Tasks 4.3–4.9 (4.9, "Cut the GPU's
  per-group round trips", is still proposed) and the "Pre-4.7" and "Pre-4.9"
  analyses of its log; `alis/gpu_dispatch.py`, `alis/minimise.py`,
  `alis/model_eval.py`.
- **What was found** (Q11.3, from Stage 4's figures on DH_orders):
  - the sub-pixel grids already stay on the device: 28.8 MB, uploaded once per
    iteration;
  - the transfers are small. What costs is the 351 synchronising round trips per
    evaluation, and the host's work between them (`vfwhm` 20% and `legendre` 9.3% of
    a GPU Jacobian, and the loop over snips). A Jacobian takes 61.9 s on 4 GPUs
    against about 32–37 s on 12 CPUs;
  - the Voigt arithmetic of one base evaluation is about 0.2 s on one RTX 2080 Ti;
  - the Levenberg–Marquardt minimiser needs the residuals (0.6 MB) at each
    evaluation and the Jacobian (248 MB dense) at each iteration, not χ² alone.
- **To finish:** `gpu_stage0.md`, the series' design document:
  - measure again, then build a prototype of the evaluation kept on the device, with
    the figure it must beat;
  - its queries: what comes back (the residuals and the Jacobian, or JᵀJ and Jᵀr, or
    χ² with another minimiser); the convolution's FFT, which numba.cuda lacks (CuPy,
    or a direct convolution at 21× the arithmetic); the hardware to measure on
    (float64 at 1/32 of float32 on the RTX 2080 Ti, 1/2 on a data-centre card); and
    which functions to port first (those DH_orders uses).

  The CPU path stays bitwise-identical throughout. Refactor Task 4.9 closes as taken
  over by the series.

### 1.5 Local minima, and how sensitive the minimisation is to the sub-pixels (dashboard Stage 14, Q14.10)
- **What:** ways to tell when a fit has settled in a local minimum rather than the
  best one, and to measure how far a fit's minimum depends on its sub-pixels. A
  series of its own, independent of the dashboard (RJC, Q14.10's response,
  2026-11-25): `claude_prompts/minimisation_stage0.md` first, on its own branches.
- **What was found** (Task 14.6 of `dashboard_stage14.md`, and its log): with
  `nsubpixel_min` 10 instead of 5, the harness's fits of DH/HS0105p1619,
  VMP_DLA/J0903p2628 and DH/J1419p0829 reach, from their own starting values, a
  minimum with χ² higher by 50.7, 21.6 and 4.7 than their old best fit evaluated at
  10. Fitted at 10 from the old best fit, or from starts drawn about it from its
  covariance, they return to it: both are minima, and which one the
  Levenberg–Marquardt minimiser reaches from a given start changed with the
  sub-pixels. HS0105p1619 and J0903p2628 were already sensitive to their start at
  refactor Q0.17. Doubling the sub-pixels also moved the model column of
  J0903p2628's best fit by up to 23 times the error: at 5, its narrowest profiles
  were not integrated to within the noise.
- **Why deferred:** a question of the minimiser and of the model's numerics, larger
  than the dashboard's stage; Stage 14 started the three cases from their old best
  fits (or starts about them) so that their references are the better minimum.
- **Techniques to try** (Claude's suggestions):
  - *Restarts as a routine check.* Starts drawn about the best fit from its
    covariance exist (`sim perturb` after `sim beginfrom`; the dashboard's Random
    restarts…). Run them by default after a fit, or in a nightly batch for the cases
    known to be sensitive, and report the lowest χ² found, flagging a restart that
    beats the fit (the dashboard already offers to continue from one). §7.6 (where
    `sim perturb` draws) comes first.
  - *Wider starts.* The covariance only explores the basin the fit is in. Sample the
    parameters most likely to be multimodal (the column densities, b and redshifts of
    blended components) within their limits: a Sobol or Latin-hypercube design, each
    start fitted with the local minimiser; or basin hopping (perturb, fit, accept by
    a Metropolis rule), as `scipy.optimize.basinhopping` does, with ALIS's minimiser
    as the local step.
  - *Sampling the posterior.* An ensemble MCMC (emcee) or nested sampling (dynesty,
    UltraNest) on ALIS's χ², started about the best fit. Separate modes show as
    separate clusters of samples, and nested sampling also gives the evidence. It is
    costly for the largest models (DH_orders' 300-odd parameters): try it on
    sub-models first, and with the GPU series (§1.4).
  - *Coarse to fine in the sub-pixels.* Fit at few sub-pixels, then refit at more from
    each solution, so the minimum is followed as the grid refines rather than
    jumped. `nsubpixel_renew` already lets the grid follow the fit.
  - *The sensitivity to the sub-pixels, measured.* For each harness case, evaluate
    and refit its best fit at `nsubpixel_min` 5, 7, 10, 15, 21 and 41, and plot χ²
    and each parameter against it. A well-sampled fit is flat beyond some N; report
    the N past which no parameter moves by more than 0.1σ. It could become a check
    (`run_alis --check-subpixels`, or a report in the dashboard) and a test of the
    harness. Beside it: the error of integrating a profile of width σ on N
    sub-pixels against a very fine integration, in units of a pixel's error, which
    gives the sub-pixels the data need (the `nsubsigma` rule, checked).
  - *The local minimiser itself.* Compare against another (`scipy.optimize.
    least_squares`'s trust-region reflective, with bounds); check the stopping rules
    (`atol` against `ftol`, §2.4's rejected step); analytic derivatives where they
    are cheap; a final Newton step to polish the minimum.
  - *In the harness.* The start-sensitive cases (the three above, and Q1243p307
    newstart76) gain a nightly test that N starts about the reference return to it.
- **Where:** `dashboard_stage14.md` Q14.10 and its response; its log, Task 14.6
  (every case that moved, with its figures; scratch scripts named there);
  `refactor_code_stage0.md` Q0.17; `alis/minimise.py`, `alis/simulate.py`,
  `alis/dashboard/restarts.py`.
- **To finish:** `minimisation_stage0.md`, the series' design document: measure the
  sensitivity first (the sweep in N above, on every harness case), then choose and
  prototype the techniques, each against the cases where a better minimum is known.

---

## 2. Features that do not work

### 2.1 Wavelength-dependent resolution

*Not taken in dashboard Stage 14 (Q14.1); offered to Stage 15 (its candidate 3).*
- **What:** convolving each pixel with its own sigma, read from the
  `resolution` data column, for data whose line spread function is Gaussian
  everywhere but varies with wavelength.
- **State:** **removed and now reports itself.** The branch existed in four
  convolution functions (`afwhm`, `vfwhm`, `voigtconv`, `vsigma`) but referred to
  an undefined name and raised `NameError` on entry, so it had never run and its
  numerics had never been checked. Stage 6.5 replaced it with an explicit
  "a wavelength-dependent resolution is not supported" error.
- **Why:** RJC, Q6.16 option (c) — better to support a single value honestly
  than to enable untested numerics.
- **To finish:** reinstate the per-pixel convolution *with a test* that checks it
  against a hand-computed convolution. Beyond that, RJC notes the real target is
  a wavelength-dependent **non-Gaussian** LSF, which is a much larger problem.
- **Where:** Q6.16; the old code is in git history before Stage 6.5.

### 2.2 SuperMongo output (`out sm`) — removed
- **State:** deleted in Stage 6.5. `prep_arrs` (`alis/plot.py`) and
  `save_smfiles` (`alis/save.py`) were dead — their only callers were commented
  out — and the live `out sm True` branch did nothing but raise "not implemented
  yet" *after* the fit had finished. ~200 lines and 22 of the 36 F821s.
- **Recorded here so it is not re-added by accident.** RJC, Q6.6/Q6.8/Q6.9: no
  user has it set, delete outright, no deprecation warning. The need it served
  is better served by Stage 5.3's standalone matplotlib scripts.

### 2.3 `systmodule=` — revived, but on probation
- **State:** the user systematics-module hook called `imp.load_source`, and
  `imp` was removed from the standard library in Python 3.12 while ALIS targets
  3.13 — so it raised `NameError`, a bare `except` swallowed it, and the user was
  told *their own file* could not be imported. Stage 6.5 rewrote it with
  `importlib.util.spec_from_file_location` and narrowed the `except`.
- **Open question:** RJC, Q6.17 — "revisit in the future to make sure it is still
  worthwhile supporting". Every `systmodule=` line in the repository is commented
  out (two DH context models), so nobody is using it.

### 2.4 `examples/brokenpowerlaw` — excluded from the fixed-parameter gate

*Back in the gate at dashboard Stage 14's close (Q14.13 (b), 2026-11-25), with
`examples/tophat`; kept open for Stage 15 to confirm (its candidate 1). With their
references regenerated (Task 14.7), each recording the sub-pixels its fit used
(`nsubpixel=`), both pass the evaluation at the best fit: `brokenpowerlaw` χ²
3499.947056 against 3499.947053 recorded, `tophat` 394.132678 against 394.132678, both
model columns within the tolerance. Their references before, which recorded no
sub-pixels, fail it (`brokenpowerlaw` 3555.29 against 3503.07 with 5 sub-pixels per
pixel). So the mismatch below was the sub-pixels (§7.10), and the parameters
`brokenpowerlaw`'s `.mod.out` writes are those of the χ² it writes, which the
suspected cause would not give. **To finish:** read the minimiser's stopping paths at
`atol` and confirm that none reports a rejected trial step; mode (b), which now runs
on every unblinded case, checks each case's written parameters against its written
χ². If either case fails again, it can be excluded (RJC, Q14.13: they matter little
to QSO absorption-line work). Kept below as it was written.*
- **What:** its reference evaluates to chi-squared 374.98 against a recorded
  338.15, so it is in `FIXEDPARAM_EXCLUDE`.
- **Suspected cause:** the writer may emit a final *rejected* trial step rather
  than the best accepted point when convergence triggers on `atol`.
- **Why deferred:** RJC, Q5.4 option (c) — the model is unlikely ever to be
  needed. Changing which parameter vector the minimiser reports is a numerics
  change that would move every reference and deserves its own gated task.
- **Where:** Stage 0 Q0.12, `refactor_code_stage5.md` §5.4, Q5.4.

### 2.5 The onefits interactive menu
- **State:** `load.load_onefits` presents a five-option menu on stdin. It is
  untested, and `load_input(textstr=...)` — which option 2 uses — returns lines
  *without* their newlines while the file path keeps them, so `save_model` would
  concatenate the settings into `run ngpus 0run backend cpu`. Only reachable
  through the menu, and only for plotting, so it is not live.
- **Where:** stage 5 log, Task 5.5.

---

## 3. Known defects and rough edges, recorded not fixed

### 3.1 `load_fits` oddities
- The continuum default was zeros where `load_ascii` returns ones, which made
  the model be multiplied by zero for any FITS spectrum with no continuum
  column. **Fixed** by RJC 2026-08-04, together with the companion slip in the
  zero-level branch that assigned to `contin` instead of `zeroin`.
- Still rough: the format detection is three nested `try`/bare-`except` blocks,
  and a `.fits` file that is none of the three recognised shapes fails with a
  message about `run+datatype`. No shipped example loads FITS data, so none of
  this is exercised by the harness. Covered at unit level by
  `tests/test_load_files.py`.

### 3.2 The first data line of a block must carry `columns=`
- `colspl` is assigned only inside the `columns` branch of `load_data`'s loop, so
  omitting it on the *first* line raises `NameError`, which the bare `except`
  turns into "Error reading in file". Later lines are fine — they inherit the
  previous line's `colspl`, which is why 131 of the 181 data lines in
  `examples/` can omit it.
- **Where:** stage 5 log, Task 5.5.

### 3.3 `1e4` is not a number to ALIS
- The parameter micro-syntax packs value and tie label into one word, and only a
  *signed* exponent is recognised — so `1e4` reads as the value 1.0 tied to a
  label `e4`, and two lines both written `1e4` silently share one free
  parameter. Stage 5.7 made this an **error** rather than a silent
  reinterpretation, with the rule "reject `e`/`E` followed only by digits".
- Recorded here because the underlying ambiguity remains: the format cannot
  represent an unsigned exponent, and never will without changing the syntax.
- **Where:** `refactor_code_stage5.md` §5.7.

### 3.4 `msgs.bug` reports and continues
- `msgs.error` exits 1 since Stage 6.1 (Q6.11), and the `sys.exit()` calls that
  follow a `msgs.bug` were changed to exit 1 with it. But `msgs.bug` *itself*
  does not exit: a reported internal bug prints and the fit carries on. Whether
  that is right is untested either way.

### 3.5 Ctrl+C exits 0
- `logger.signal_handler` calls a bare `sys.exit()`, so an interrupted fit
  reports success to a shell. Left deliberately in Stage 6.1 — the user asked to
  stop — but a caller cannot distinguish "finished" from "abandoned".

---

## 4. Lint and formatting backlog (Stage 6.5, option b)

Stage 6.5 took Q6.22 option (b): the **linter** now guards every `alis/` module
(they were removed from ruff's `extend-exclude`), but the **reformat** was
deliberately separated. What remains:

### 4.1 The black/isort reformat
- **Scope, measured 2026-08-05:** 39 files, −8,533 / +18,282 lines, **43% of
  their 19,867 lines**. It is dominated by one pattern: ALIS is written with
  **1,201** `if x: y` one-liners, and black splits each into two.
- **Why deferred:** RJC, Q6.22 — the diff is not reviewable by eye, and the
  value (consistent layout) is much smaller than the value of the linting, which
  has now been obtained without it.
- **To finish:** the lists are `[tool.black] force-exclude` and
  `[tool.isort] extend_skip` in `pyproject.toml`. RJC's stated preference
  (Q6.14) is **batch-by-batch commits plus a mechanical AST-equivalence check** —
  parse each file before and after and assert the trees match, which is complete
  evidence rather than sampled. Add the black badge to `README.md` only once
  this is done, or it advertises something untrue.

### 4.2 Rules ignored until then
`[tool.ruff.lint] ignore` in `pyproject.toml`, with counts as of 2026-08-05:

| rule | count | why it is ignored |
|---|---|---|
| E701/E703/E401 | 1,205 | formatting only; exactly what §4.1 fixes |
| F841 | 70 | unused locals; each needs a look, some are unpacking |
| E712 | 51 | `== True` in legacy code |
| E402 | 31 | imports below code |
| E741 | 15 | `l` as a variable name |
| F401 | 15 | unused imports — **not** to be swept blind, some are side-effect imports |
| E722 | 63 | bare `except:`, several load-bearing (`load_fits` uses one to detect the file format, `load_ascii` to decide a column is absent). Narrowing changes which errors are swallowed, in code no test covers on the failure path (Q6.15) |

Everything else in E4/E7/E9/F **is** enforced repo-wide, which is the part that
catches real defects — F821 undefined names, E9 syntax errors, and the rest of F.

### 4.3 The rule set is pinned
`select = ["E4", "E7", "E9", "F"]` is now explicit in `pyproject.toml`. Ruff's
*default* changed between the v0.6.9 the pre-commit hook installs and current
releases (0.16 adds UP/C4/SIM/PLR and reports ~1,200 further findings), so
leaving it implicit made the gate depend on which ruff happened to be installed.

---

## 5. Testing gaps

*The `context/` fits in CI were not taken in dashboard Stage 14 (Q14.1); offered to
Stage 15 (its candidate 5).*

- **`plot.py` and `prepfit/specplot.py` are not reached by the Stage 0 gate.**
  The harness exercises fitting, not plotting. `plot dims 0` is now covered by
  the fast batch, but the drawing code is not.
- **The `plotscript` `DH` layouts for one and two datasets** are unit-tested
  against RJC's Q5.9/Q5.11 descriptions but have never been run on real data
  (stage 5 log, Task 5.3).
- **`systmodule=`** has one unit test (Stage 6.5) and no end-to-end coverage;
  no model in the repository uses it.
- **GPU coverage** is 40 tests and skips structurally where there is no CUDA
  device; `context/` fits are not run in CI at all.

---

## 6. README badges

Added in Stage 6.7: GitHub, licence (BSD 3-Clause, confirmed against `LICENSE`
and `pyproject.toml`), Python version, and the CI status badge — which is the
useful one, since it is live and goes red when the `unit` / `examples` / `lint`
jobs fail. The **black** badge is deliberately *not* there yet; it goes up when
§4.1 is done.

---

## 7. The dashboard: items deferred from its stages

Added 2026-10-07 (dashboard Stage 4). Here `Q4.n` is a query in
`claude_prompts/dashboard_stage4.md`. §7.5 and §7.6 were added on 2026-10-18 (dashboard
Stage 6), where `Q6.n` is a query in `claude_prompts/dashboard_stage6.md`. §7.8 and
§7.9 were added on 2026-11-16 (dashboard Stage 12, Q12.1 (b)), §7.10 and §7.11 on
2026-11-17 (Task 12.9, Q12.13 (b) and Q12.10's response).

### 7.1 Telluric flags in the transition list
- **What:** flag a transition whose observed wavelength falls in a telluric band,
  from a small built-in list of the strongest bands in the observed frame: the O₂
  A, B and γ bands, and the H₂O bands near 7200, 8200 and 9300 Å. S2 asked for the
  flag; ALIS has no such list.
- **Why deferred:** RJC, Q4.5(b): not needed in Stage 4, to be revisited later.
- **Where:** S2 in `ALIS_v2_dashboard_prompts.md`; Q4.5 in `dashboard_stage4.md`.
- **To finish:** the band list (with its source) as data in `alis/data/`; the flag
  computed in `alis/dashboard/lines.py` beside "gap" and "edge"; shown in the
  Regions tab's transition list.

### 7.2 The best-fit model on the Regions tab
- **Done (dashboard Stage 6.9, 2026-10-19):** once a run exists, a menu beside the
  Regions tab's Model switch draws the starting model or the Fit tab's shown run's
  best fit (`qt/regions.py`, `model_from`). Kept here for the record.
- **What:** a switch that draws a best-fit model (when one exists) over the data on
  the Regions tab, while the regions are being drawn.
- **Why deferred:** RJC, Q4.12: the Regions tab draws no model in Stage 4; the model
  is drawn on the Components tab (Stage 5) and the Fit tab (Stage 6).
- **Where:** Q4.12 in `dashboard_stage4.md`.
- **To finish:** the model of the latest run, per snip, through the blinding gate
  (D24 allows profiles to be drawn), as an overlay of the shared spectrum view
  (`qt/plots.py`), off by default.

### 7.3 Solar abundances in the atomic table
- **What:** fill the `SolarAbundance` column of `alis/data/atomic.ecsv`, which holds
  0.0 (or NaN) for every line, with each element's solar abundance as log ε (H = 12),
  and read it from there: the dashboard's line IDs (`alis/dashboard/lines.py`, the
  `SOLAR` table) and `prepfit` (`alis/prepfit/specplot.py`, `solar()`) each keep
  their own copy of Asplund et al. (2009) today.
- **Why deferred:** RJC, on the review page of the Data tab (Q4.18(b)): an item for
  future work. Read as log ε, the column would not change which lines are labelled.
- **Where:** Q4.18 in `dashboard_stage4.md`; the "Line IDs" design of Stage 4.
- **To finish:** the values in the ECSV (with their source), `check_atomic_table`
  checking them, and `lines.SOLAR` and `specplot.solar()` replaced by the column.

### 7.4 A stacked spectrum of a transition
- **What:** combine the data of one transition from every dataset (resampled onto one
  grid, weighted by their errors) into one spectrum on the Regions tab, on which the
  regions and the continuum are drawn and then applied to every snip of it.
- **Why deferred:** RJC asked (Q4.19(d)) whether it belongs in Stage 4 or later;
  Claude proposed later (Q4.20): the datasets differ in resolution and flux scale
  (Q1243p307's HIRES ≈1100 counts, PROCHASKA ≈175), so a continuum drawn on the stack
  has to be carried back to each dataset's own scale, and the stack is not what ALIS
  fits. Stage 4 already keeps one set of regions per transition (`snips.linked_step`)
  and can share one continuum between datasets (S10).
- **Where:** Q4.19 and Q4.20 in `dashboard_stage4.md`.
- **When:** with Orders mode and PypeIt spec1d loading, after v1 (RJC, Q4.20); the
  overall plan says so (D32 in `ALIS_v2_dashboard_prompts.md`, and the stage table of
  `dashboard_stage0.md`).
- **To finish:** the stack (`snips.py`, no Qt: resampling, weights, the mapping of
  regions and continuum back to each snip), its strip on the Regions tab, and tests on
  Q1243p307's three datasets.

### 7.5 Generating data from a model (`generate data`)
- **What:** synthetic spectra made from a model, as ALIS's `generate data True` makes
  them: the model evaluated on a grid of `generate pixelsize`, with noise set by
  `generate peaksnr` and `generate skyfrac`, written as data files. From the
  dashboard, for example to test whether a feature would be detected, or to plan
  observations.
- **Why deferred:** a bundle cannot run it, since it writes data files
  (`bundle.unsupported`). RJC, Q6.16 of `dashboard_stage6.md`: to be implemented as
  future work. In Stage 6, All settings… offers it, marked "command line only", and
  the pre-flight check refuses Run while it is set, and says to export plain files
  (File → Export plain files…) and run `run_alis` on them.
- **Where:** Q6.4 and Q6.16 in `dashboard_stage6.md`; `alis/main.py` (the `generate
  data` branch), `alis/load.py` (the grid from `generate pixelsize`), `alis/save.py`
  (the noise from `generate peaksnr` and `generate skyfrac`).
- **To finish:** the generated data written through a bundle's outputs instead of to
  files beside the model, so that a bundle can run it; in the dashboard, a dialog for
  the grid, the S/N and the sky fraction, the generated spectrum added to the project
  as a new dataset or exported; tests against `run_alis` with `generate data True`.

### 7.6 Simulated realisations and random restarts (`sim random`, `sim perturb`)
- **What:** decide how ALIS draws its simulated realisations and restarts, and whether
  a bundle can run them. While planning the dashboard's random restarts (Stage 6), Claude
  found that `simulate.perturb` draws each restart's start about `slf._modpass['p0']`.
  Its `bparams` argument is passed but not used. ALIS reaches `perturb` by two routes,
  which therefore draw about different centres:
  - `sim perturb N` in the model of a fit (`alis/main.py`, after the fit): `p0` is the
    run's *starting* values, and the covariance is the best fit's;
  - `sim perturb N` with `sim beginfrom <fit>`: `p0` and the covariance are read from
    that fit's `.mod.out` and `.covar`, so the starts are drawn about its *best fit*.
    This is what `sim newstart`, §6 of `doc/ALIS_workflow.md` and Q6.10 describe.

  Each draw is clamped to ±`sim maxperturb` (10%) of its value (a value of zero is not
  clamped), then to its limits. Unlike `sim_random`, `perturb` does not call
  `convergence.assess_restarts`. A bundle refuses `sim random`, `sim perturb` and `sim
  beginfrom`, since they write to `sims/` or read a fit from disk.
- **Why deferred:** RJC, Q6.18(d) of `dashboard_stage6.md`: the approach to the
  simulated realisations needs more thought.
- **Where:** Q6.10, Q6.18 and Q6.22 in `dashboard_stage6.md`; `alis/simulate.py`
  (`perturb`, `sim_random`), `alis/main.py`, `alis/convergence.py`.
- **To finish:** decide the centre of each route's draws, and whether the 10% clamp
  stays. Then decide whether `sim random` and `sim perturb` write through a bundle's
  outputs, so that a bundle (and the dashboard) can run them. The dashboard's Random
  restarts… (Q6.22) then follow the decision.
- **Meanwhile (Stage 6.8):** the dashboard draws its own starts as the second route
  does (about the shown run's best fit, from its covariance, the 10% clamp and the
  limits: `alis/dashboard/restarts.py`, `draw_starts`), fits each as an ordinary run,
  and assesses them with `convergence.assess_restarts`. Only `draw_starts` changes
  when the decision is made.

### 7.7 `apod`: example data in frequency, and its buffer

*Not taken in dashboard Stage 14 (Q14.1); offered to Stage 15 (its candidate 4). Stage
14 found that `apod` cannot be evaluated by ALIS at all (§7.12).*

- **What:** ALIS's `apod` resolution function (an apodisation of the kind named, for
  spectra in frequency) has no example in `examples/`, and nothing in the test suite
  uses data in frequency. Its `getminmax` reads its value as a FWHM in Hz, so on a
  spectrum in Å it asks for an extraction buffer of about 130 Å (metal_line_abs's
  Si II snip), and the dashboard's Snip box marks that snip's buffer too narrow.
  `apod` also needs `bintype=Hz` (the dashboard writes it on the snip's data line,
  D92).
- **Why deferred:** RJC, Q8.13 of `dashboard_stage8.md`: there are no frequency data
  among the examples, so `apod` waits; `lsffile`'s buffer, which had a similar fault
  (a velocity taken at a fixed 1309 Å), was fixed in Stage 8.
- **Where:** `alis/functions/apod.py` (`getminmax`, `set_vars`'s bins);
  `alis/dashboard/resolution.py` (`apod` offered with `kind:uniform`).
- **To finish:** add an example fitted with `apod` on data in frequency (made with
  `generate data`, §7.5, or from a Fourier-transform spectrometer), with its reference
  outputs in the regression harness. Then check whether `apod`'s buffer is right for
  such data, and correct `getminmax` if it is not.

### 7.8 Load snips…: a file that cannot be read, or names no transition

- **What:** Load snips… (snips without a model) still stops the whole load when one
  file cannot be read as a table of numbers, or when its name names no transition
  (D122, Q7.4): such a file has no transition to be placed at. The dialog that assigns
  snips could take it instead, on its right, if the transition were guessed from the
  system's ions or chosen by the user.
- **Why deferred:** RJC, Q12.1 (b) of `dashboard_stage12.md`: it waits for a user to
  ask for it. A guessed transition is one the user would have to check, and a file
  of `prepfit`'s names its transition.
- **Where:** Stage 11's "Carried over" item 1 (`dashboard_stage12.md`), D122;
  `alis/dashboard/loadfit.py` (the files' transitions), `alis/dashboard/qt/window.py`
  (Load snips…).
- **To finish:** a file whose name names no transition goes to the dialog's
  right-hand list with a transition to choose (a menu of the system's ions'
  transitions over its wavelengths); one that cannot be read is named and left out,
  and the others load.

### 7.9 Open a fit's question for a fit of one dataset

- **What:** Open a fit asks every fit, in RJC's words, for "the full spectra" its
  snips were cut from (D123), since their number cannot be known before they are
  given. A fit whose snips all share one dataset was asked in the singular before
  Stage 11.
- **Why deferred:** RJC, Q12.1 (b) of `dashboard_stage12.md`: RJC's wording fits
  every fit, and it waits for a user to ask for the singular back.
- **Where:** Stage 11's "Carried over" item 3 (`dashboard_stage12.md`), D123;
  `alis/dashboard/qt/window.py` (`open_fit`).
- **To finish:** the question counts the `.mod`'s datasets (`loadfit.fit_datasets`)
  and words itself for one or for several.

### 7.10 ALIS's sub-pixels: the default `nsubmin` of 10, and the sub-pixels a `.mod.out` is evaluated on

*Done in dashboard Stage 14 (2026-11-25, D148–D152): the default is 10
(`nsubpixel_min`, every name of the sub-pixels by one convention); a `.mod.out`
records the sub-pixels its fit used (`nsubpixel=N`) and is evaluated on them;
`examples/lsf_spline`'s evaluation at the best fit is back in the gate; the 22 fits
whose references moved were refitted, checked and regenerated. Three of them reached a
worse minimum from their own start, which is §1.5. Kept below as it was written.*

*The key item of dashboard Stage 14, which comes next (RJC, Q13.7's response of
`dashboard_stage13.md`, 2026-11-24).*

- **What:** raising ALIS's default fewest sub-pixels per pixel from 5 to 10, so that
  every model is worked out on at least 10 sub-pixels (RJC, Q12.9's response).
- **Why deferred:** RJC, Q12.13 (b) of `dashboard_stage12.md`. With the default at
  10, the regression harness's `fast` batch failed 27 of its 149 tests: 15 fits
  (9 shipped examples, each as a bundle too, and 6 context fits), 2 evaluations at
  the reference's best fit, and `examples/generate`'s data. Some fits changed their
  result (J0903p2628: a parameter 143.6 → 128.2, 3.5σ; HS0105p1619: χ² 2086.3 →
  2137.0; `splineabs` linear: a 1σ error 113.2 → 6.2). The `lsfspline` delta
  function does not need it: a model sets `run nsubmin 10` where it wants finer
  sub-pixels, and the dashboard's spix follows the model's `nsubmin`.
- **Where:** `alis/config.py` (`nsubmin`), `doc/ALIS_workflow.md`'s table,
  `doc/tex_files/description.tex`; the log of dashboard Stage 12 (Task 12.9).
- **To finish:** a stage of its own: the 16 cases refitted, each new fit checked by
  eye, their references regenerated with their χ² before and after, and
  `CHANGELOG.md`.
- **And the sub-pixels a `.mod.out` is evaluated on** (dashboard Stage 13, Task 13.2,
  Q13.7 (b)). ALIS sets each snip's sub-pixels when it loads a model, from the
  starting values (`load.py`, each function's `set_vars`), so a `.mod.out` run again
  or evaluated from its best fit can use other sub-pixels than the fit did, where a
  width moved far in the fit. `examples/lsf_spline`: the fit started at a smoothing of
  4 km/s on 6 sub-pixels; its `.mod.out` starts at 0.12 km/s and takes ALIS's most,
  21, and its model differs at 15 of 460 pixels, by up to 2.3 times the error (pinned
  at 6, it agrees exactly). The dashboard is not affected: it draws a run with the
  run's own input model. The case's evaluation at the best fit (the harness's mode
  (b)) is left out until this is done (`FIXEDPARAM_EXCLUDE` in
  `tests/test_regression.py`).
  - **To finish:** ALIS records each snip's sub-pixels in the `.mod.out`, as a
    data-line keyword (as Stage 5.4's `bufferpix` records the pixels loaded), and uses
    them when given; `examples/lsf_spline`'s mode (b) back in the gate; with the
    default `nsubmin` above, every reference that moves regenerated.

### 7.11 `lsfspline` in a `.mod.out`, and `examples/lsf_spline` as a regression case

*Done in dashboard Stage 13 (Task 13.2, Q13.3 (a), 2026-11-24): `lsfspline` keeps its
`locations` in the `.mod.out`, the dashboard reads a run stored before, and
`examples/lsf_spline` is a case of the harness (its fit and its bundle); its
evaluation at the best fit waits in §7.10 (Q13.7 (b)). Kept below as it was
written.*

- **What:** ALIS writes an `lsfspline` resolution into the `.mod.out` without its
  `locations` keyword (`SplineLSF.parout` writes the values only), so the output
  model cannot be read back: "locations must be set". The regression harness's
  evaluation at the reference's best fit (mode (b)) rereads it, so
  `examples/lsf_spline` cannot be a case until then.
- **Why deferred:** RJC, Q12.10's response of `dashboard_stage12.md`: if the
  harness has problems with the case, it waits, and the unit tests stand
  (`tests/test_lsfspline.py`). Its data were generated again in Task 12.9, with
  the smoothing as a FWHM and the kernel centred; its fit takes 9 s (0.002 h), and
  modes (a) and the bundle passed with references written from it.
- **Where:** `alis/functions/lsfspline.py` (`parout`); `examples/lsf_spline/`.
- **To finish:** `parout` writes `;locations:…` (and the other keywords given),
  checked by reading a `.mod.out` back; then the case's references
  (`model/fit_spectra.mod.out.reference`, `data/reference_fits/OI_SiII_fit.dat`)
  from a fit, and the harness on it, CI's `examples` batch included.

### 7.12 Three faults of the resolution functions' keywords

*Found in dashboard Stage 14 (Task 14.1, 2026-11-25), while each data line's `lsffile`,
`apod` and `lsf` keywords were made its own; offered to Stage 15 (its items 12–14,
with Q15.4–Q15.6).*

- **`apod` cannot be evaluated by ALIS.** `model_eval` calls every resolution's
  `call_CPU` with `mkey=`, and `apod.call_CPU(self, x, y, p, ncpus=1)` takes none
  ("TypeError: APOD.call_CPU() got an unexpected keyword argument 'mkey'"). Nothing
  tests a fit with `apod` (§7.7).
- **`lsf`'s buffer.** `lsf.getminmax`, which sizes a data line's buffer, reads the
  instance's keywords, not the line's, and `load_data` calls it before any `lsf` line
  is loaded: every `lsf` buffer is worked out for COS G130M at 1309 Å
  (`examples/lsf_hst` is STIS E140H). Its evaluation is right. Fixing it moves
  `examples/lsf_hst`'s loaded pixels, so its reference.
- **A model loaded again** on the same instances (`sim repeat`, `iterate model`;
  `main.py`) starts each line from the keywords of the last line evaluated
  (`_keywd = mkey[i]`), in any function's loader, `base.Base`'s included. A fix gives
  every load fresh defaults in `load.call_function_load`.
- **Where:** `alis/functions/apod.py`, `alis/functions/lsf.py`, `alis/load.py`
  (`load_data`, `call_function_load`), `alis/main.py`.

### 7.13 `examples/lsf_spline`'s data at a higher S/N

*Found in dashboard Stage 14 (Q14.14, 2026-11-26), when CI's `examples` job failed
from Stage 13 on; offered to Stage 15 (its item 16, Q15.8).*

- **What:** the example's data were made with an `lsfspline` smoothing of 4.0 km/s,
  but its fit trades the smoothing against the knots' heights, and a smoothing
  narrower than a sub-pixel (0.14 km/s) leaves the model as it is: the fit cannot
  reproduce the model its data were made with. Under numpy 2.4.3 the smoothing goes
  to its bound (2.9×10⁻⁷ km/s, error 0); under 2.5.5 it stops at 0.082 km/s (error
  6128). The harness passes both (D153), but the example shows a degenerate fit.
- **Why deferred:** RJC chose to refine the harness's rule for now (Q14.14 (a)), and
  noted that the test data should be generated with a higher S/N.
- **Where:** `examples/lsf_spline/model/generate_spectra.mod`, its data and
  references; `tests/test_subpixels_fit.py` and dashboard Stage 14's review use the
  example for a best fit that asks for more sub-pixels than its start, which new data
  may not give (another example would then serve).
- **To finish:** data made at a higher S/N, and perhaps knots the smoothing is not
  traded against, until a fit from the example's start returns the smoothing it was
  made with, with a finite error; then the references regenerated, and the harness
  run under the newest numpy as CI installs it.
