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
1. **The Plot tab** (dashboard step 8; D28, D42, S22). It is next in the agreed
   plan, and the part that writes the plotting script already exists
   (`alis/plotscript.py`). Two of that script's D/H layouts (for one and two
   datasets) have never run on real data (§1.1, §5).
2. **How ALIS draws `sim perturb` and `sim random`** (§7.6). This is a correctness
   question in ALIS itself. `simulate.perturb` ignores the best fit it is given and
   draws about the starting values, so the two ways of reaching it draw about
   different centres. The decision also settles whether a project file can run
   simulations, and the dashboard's Random restarts would follow it.
3. **Diagnose `examples/brokenpowerlaw`'s χ² mismatch** (§2.4: 374.98 against
   338.15). The suspected cause is that ALIS reports a rejected trial step when the
   fit stops on `atol`. If so, it would affect any model that stops on `atol`, not
   just this example.
4. **Sphinx/ReadTheDocs documentation and `CONTRIBUTING.md`** (§1.2). This was
   deferred until v2.0.0 is ready to release, which is now the next milestone.
   `doc/tex_files/` is out of date.

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

### Low priority
13. The link editor and its templates (S15). Links can already be typed in the
    `.mod` panel.
14. Command search (F10).
15. Reproducing an imported fit (S28).
16. Solar abundances in `atomic.ecsv`, replacing the two hard-coded copies (§7.3).
    This changes no behaviour.
17. Deciding whether to keep `systmodule=`, which nobody uses (§2.3).
18. Whether `msgs.bug` should stop the fit (§3.4).
19. The black/isort reformat and the remaining lint rules, then the black badge on
    the README (§4.1, §4.2, §6).
20. Testing gaps: the plotting code, GPU coverage, and the `context/` fits in CI
    (§5).
21. The metal example's junk last pixel (`examples/metal_line_abs/data/OI_SiII.dat`,
    flux −10¹⁰, outside the fit range), found in dashboard Stage 6.

### Not planned, or entries to close
- **YAML/TOML model files** (§1.3): dropped by RJC (Q5.2/Q5.3).
- **Pausing a run**: declined by RJC (Q6.15(c) of `dashboard_stage6.md`).
- **The onefits menu** (§2.5): onefits has been removed and its code is gone, so
  the entry can be closed.
- **The GUI** (§1.1): built as the dashboard. Only the Plot tab remains, so the entry
  needs updating.
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

---

## 2. Features that do not work

### 2.1 Wavelength-dependent resolution
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
Stage 6), where `Q6.n` is a query in `claude_prompts/dashboard_stage6.md`.

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
