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
