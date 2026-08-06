# Prompt file for ALIS software refactoring -- STAGE 6

> **Usability, GUI, and documentation.** Modernise the command-line interface,
> grow the `prepfit` GUI into a single prepare/run/inspect tool, and move the
> documentation to Sphinx + ReadTheDocs (code as the source of truth) with a
> tutorial and a contribution guide. Depends on Stage 2; independent of
> Stages 3/4/5. User-facing behaviour changes here must not alter fit results
> (Stage 0 gate).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage6_log.md`.
>
> **After RJC's answers to Q6.2–Q6.8, the stage is 6.1 -> 6.5 -> 6.7 -> 6.4.**
> 6.2 (GUI) and 6.3 (documentation) are deferred to separate documents, and 6.6
> goes with 6.2 (Q6.4). **6.7 is new** — the settings/workflow documentation
> catch-up RJC asked for in Q6.5 — and runs after 6.1 and 6.5 so that it
> documents the final state (6.1 renames flags, 6.5 retires `out sm`). The
> numbers are kept as they were rather than renumbered, so references from the
> other stage docs and the logs stay valid.

**6.1 — CLI modernisation. [DONE 2026-08-05/06]**
- Replace the custom argument parser with `argparse` (or `typer`) for a
  self-documenting `run_alis --help`, tab completion, and consistent option
  naming. Preserve the existing invocation (`run_alis <model>.mod`); **flags may
  be renamed and cleaned up, and no aliases are needed for backward
  compatibility** (Q6.1, revised 2026-08-04).
> **Premise partly overtaken by events — measured 2026-08-04.** `run_alis.py`
> **already uses `argparse`**: `parser()` builds a real `ArgumentParser` with the
> positional `alis_modfile` and 14 options, and `[project.scripts]` installs
> `run_alis`. What is still hand-rolled is the *other half* of the path, so the
> task is now "finish the job", not "replace the parser". See the audit below
> for the six specific items.
>
> **Item 1 is DONE (2026-08-04, Prompt 2).** The model filename now comes from
> `args.alis_modfile` instead of `sys.argv[-1]`, in both branches of
> `run_alis.main`. Verified end to end: `run_alis fit_spectra.mod -f -w -p 0`
> now fits (chi-squared 357.874162, matching the reference) where before it
> reported "The filename does not exist" — and **exited 0 while doing so**,
> because `msgs.error` calls a bare `sys.exit()`, so the failure was invisible
> to any caller checking the return code. `tests/test_cli.py` (15 unit tests) is
> new; restoring the old line fails 4 of them. The comment in
> `tests/alisrun.py` that told readers to keep the `.mod` file last is updated.
>
> **Item 7 is DONE (2026-08-04, Q6.11): `msgs.error` now exits 1.** It called a
> bare `sys.exit()`, i.e. status 0, so every error ALIS reported looked like
> success to a shell, a Makefile or CI — which is exactly why item 1 survived a
> subprocess-based harness that asserts on `returncode`. Changed in
> `alis/logger.py`, together with the five `sys.exit()` calls that follow a
> `msgs.bug` or an error `print` (`functions/base.py` x2, `functions/legendre.py`,
> `functions/chebyshev.py`, `prepfit/specplot.py:491`), on the same principle.
> **Deliberately left at 0**, because they are successful or user-requested
> exits, not failures: `logger.signal_handler` (Ctrl+C), `load.py:1745` and
> `load.py:1811` (the onefits menu after printing/extracting the model
> successfully), and `specplot.py:287` (the GUI's own quit). `load.py:377` is
> unreachable — it follows a `msgs.error` — and is left for the 6.5 sweep.
> Pinned by `tests/test_logger.py` (the status is asserted, not just the
> SystemExit) and by an end-to-end `tests/test_cli.py` test that runs the real
> entry point on a missing model file in a subprocess.
>
> Items 2-6 remain.
>
> **Q6.7 (answered): option (b).** Add flags for the Stage 4 knobs *and* a
> general `--set "run backend cpu"` escape hatch accepting any `settings.alis`
> line, so the CLI cannot fall behind the config again. `--set` should route
> through `load.set_params`, which already parses exactly that grammar and
> already warns when a setting is given twice — and it should be applied *after*
> the model file's own `par` lines, so the command line wins.

**6.2 — GUI. [DROPPED FROM STAGE 6 — RJC, Q6.2: deferred to a separate design
document.]**
- Extend the existing `prepfit` GUI (matplotlib, `Qt5Agg` backend) into a single
  interface that prepares a model, runs the fit, and inspects the result,
  iterating within one tool.

**6.6 — Interactive plotting-script generation (from Stage 5.3, RJC 2026-08-01).
[DEFERRED WITH 6.2 to the GUI design document — RJC, Q6.4.]**
- Let the user generate the Stage 5.3 standalone plotting script *interactively*
  from the GUI once a fit has completed, rather than only via a setting in the
  model file — pick the mode (`metals`, ...), adjust the panel selection and
  layout, then write the script out.
- Depends on 5.3 having landed the script emitter and its modes; this task is
  the GUI surface on top of it. Further details to be discussed in Stage 6.
- 5.3 **has** landed (`alis/plotscript.py`, ten `plotscript` settings, ten unit
  tests), so the emitter this task sits on top of is ready. Only the GUI half is
  missing, which is why it moves with 6.2.

**6.3 — Documentation. [DROPPED FROM STAGE 6 — RJC, Q6.3: deferred until v2 is
ready for deployment.]**
- Move from LaTeX/PDF to Sphinx + ReadTheDocs, using the *code* as the source of
  truth (the LaTeX in `doc/tex_files/` is reference only). Include a full
  tutorial (walk a user through preparing, running, and analysing a fit), the
  expanded example suite, and API autodoc. Add `CONTRIBUTING.md`.
> **One consequence needs a decision now (Q6.5).** With 6.3 deferred,
> `doc/ALIS_workflow.md` is the only user documentation, and it does not
> describe the settings Stages 3–5 added; nor does the shipped
> `alis/data/settings.alis`, which is the file users read to discover settings.
> Measured 2026-08-04, both below.

**6.5 — Repo-wide formatting + clear the legacy ruff findings.
[DONE 2026-08-06 as Q6.22 option (b): lint now, reformat deferred.]**
> Deferred here deliberately from Stage 4.1, which introduced *incremental*
> lint adoption instead: the linters are enforced on every file except an
> explicit exclusion list in `pyproject.toml` (black `force-exclude`, ruff
> `extend-exclude`, isort `extend_skip`). That list is a shrinking to-do list --
> this task empties it.
- Scope measured 2026-07-30: **46 excluded files** (39 legacy `alis/` modules;
  the rest is vendored reference code under `context/voigt_gpu/` and standalone
  example scripts, which stay excluded permanently). Under ruff's default rule
  set the `alis/` modules carry **~1518 findings**, of which only ~21 are
  auto-fixable: 1217 E701 (`if x: return y` one-liners, which black rewrites),
  66 E722 bare excepts, 70 F841 unused variables, 39 F821 undefined names.
  *Re-measured 2026-08-03 (after Stage 4):* E701 **1215**, E722 66, F841 70,
  F821 **37**. The two lost F821s are the `voigt.py` `SourceModule` pair the
  Stage 4 GPU port removed — see the last bullet below. Reproduce with
  `ruff check --isolated --select F821 alis/`: **`--isolated` is required**,
  because every file listed here is in `extend-exclude`, so a plain
  `ruff check alis/` reports "All checks passed" and tells you nothing.
  *Re-measured 2026-08-04 (after Stage 5):* the exclusion list is **still 46 and
  still correct** — 39 `alis/` + 7 permanent, none of them deleted, and Stage 5's
  two new modules (`alis/plotscript.py`, `alis/data/convert_xmlFormat_to_ecsvFormat.py`)
  were written clean and are *not* on it. Per-rule counts: E701 **1214**,
  E722 66, F841 70, F821 37.
  **Do not treat "~1518 findings" as a target — it is not reproducible.** Ruff's
  *default* rule set changed between the pinned v0.6.9 and a current ruff (0.16
  here): today `ruff check --isolated alis/` reports **1410**, and that total
  *excludes* E701, which is no longer selected by default. Only the per-rule
  counts above are a stable measure; pin the rule list explicitly when tracking
  progress.
- **Do black + isort first, ruff by hand.** Black is AST-preserving and so
  cannot change behaviour; `ruff --fix` is *not* safe to apply blindly here --
  it auto-removes "unused" imports (F401), which breaks side-effect imports,
  and E711 (`== None` -> `is None`) is a genuine semantic change for numpy
  arrays, where `arr == None` returns an array and `arr is None` a bool.
  **E711 re-measured 2026-08-04: 7 live, not 5** — `main.py:487,489` -> now
  **`main.py:487,493`**, `minimise.py:1422`, `model_eval.py:653`, `load.py:22`,
  **plus `load.py:1443,1445`**, which the earlier count missed. Those two are
  inside `load_model`'s `adjust_lim` guard
  (`... == (0 if modpass['mlim'][cntr][j][0]==None else 1)`) and are old code, so
  they were always there. One more is commented out at `minimise.py:1833`.
- Each file must stay green under the Stage 0 gate as it is un-excluded; do it
  in small batches, not one sweep, so a regression stays attributable.
- **Delete the SuperMongo output first (Q6.6/Q6.8, RJC: remove all references to
  `sm`, and `prep_arrs` with it).** It is the cheapest 22 of the 37 F821s, and
  it removes ~200 lines rather than reformatting them. Blast radius measured
  2026-08-04 — the whole of it:

  | file | what goes |
  |---|---|
  | `alis/plot.py:690-739` | `prep_arrs` (50 lines; all 22 `slf` F821s) |
  | `alis/save.py:731-865` | `save_smfiles` (135 lines; end of file) |
  | `alis/main.py:432,447-451` | the commented `prep_arrs` call, and the live `out sm` branch that only raises "not implemented yet" |
  | `alis/load.py:161-165` | the "you must set fits for SuperMongo" check in `optarg` |
  | `alis/load.py:315-317` | the blind-analysis branch (note `if argflag['out']['sm'] or argflag['out']['sm']` — the condition is duplicated) |
  | `alis/config.py:162` | `OutConfig.sm` |
  | `alis/data/settings.alis:48` | `out sm False` |

  No `.mod` file in the repository sets `out sm`, no test references it, and
  `doc/ALIS_workflow.md` does not mention it — so nothing else moves. **Delete
  it outright, with no deprecation warning** (RJC, Q6.9: no user has it set).
- **`alis/prepfit/specplot.py` — DONE (RJC, 2026-08-04).** `set_regions` kept
  for the GUI (Q6.8) and given its missing `self` (Q6.10), so the file is now
  F821-clean and the `alis/` total is **36**, not 37. It still needs black/isort
  before it can leave the exclusion list.
- **F821 undefined names are real latent bugs, not style** (found 2026-07-30,
  worth fixing regardless of the reformat). *All 37 re-located 2026-08-04; the
  line numbers below are current, and three of the doc's were stale.*
  - `szflx` is undefined in the wavelength-dependent-resolution branch of
    **four convolution functions** -- `afwhm.py:63,66`, `vfwhm.py:63,66`,
    `voigtconv.py:70,73`, `vsigma.py:63,66`. That branch raises `NameError` if
    reached; the shipped examples only use scalar resolution, so no test
    covers it. (8 of the 37; unchanged.)
  - `alis/plot.py` **700-735: 22** references to an undefined `slf` (the doc said
    23; recounted exactly 2026-08-04 as 37 minus the 15 elsewhere).
    **Newly established: this is dead code, so the fix is a decision, not a
    repair.** All 22 are inside `prep_arrs(snip_ions, snip_detl, posnfit,
    verbose=2)`, which does not take `slf` — the function raises `NameError` on
    its first line. Its only call site is **commented out** (`main.py:432`), and
    the `save.save_smfiles` it feeds is likewise called only from a commented-out
    line (`main.py:451`). The surviving live branch is `main.py:448-451`, which
    on `out sm True` calls `msgs.error("Sorry, supermongo generated files are not
    implemented yet")` — i.e. it aborts *after* the fit has finished. See Q6.6.
  - `alis/prepfit/specplot.py:391`: `def set_regions(arr)` assigns to `self` but
    does not take it — a method declared without `self`. **No callers anywhere**
    in `alis/` or `examples/`.
  - `alis/convergence.py:188` `nput(` — a plain typo for `input(`, and the next
    line calls `input(` correctly. This one is **live**: it is the "file exists,
    overwrite?" prompt in `save_convtest`, so it fires whenever `out convtest`
    names a file that already exists.
  - `alis/functions/lsfspline.py:222` `sidlist` (the doc said 221; Stage 5.7
    inserted a line above it) — undefined in the `specid` keyword branch.
  - `alis/functions/chebyshev.py:73` `sys` (missing import), reachable for a
    Chebyshev polynomial of order >= 10.
  - **`alis/load.py:727` calls `imp.load_source(...)` and `imp` is neither
    imported nor importable** (the doc said 611; `load.py` has grown through
    Stages 4-5) — the module was deleted from the standard library in Python
    3.12, and ALIS targets 3.13. Confirmed absent again 2026-08-04. This is the
    worst of the F821s because the call sits inside
    `try: ... except: msgs.error("Could not import module {0:s}")`, so the
    `NameError` is swallowed and reported as a missing user module: the
    `systmodule=` (user systematics module) feature is **dead**, and anyone using
    it is told their own file is at fault. No test covers it because every
    `systmodule=` line in the repo is commented out — **two** context models, not
    three: `DH/J1558m0031/model/J1558m0031_FINAL_MODEL.mod` (3 lines) and
    `DH/Q0913p072/model/Q0913p072.mod` (1 line).
    Replace with `importlib.util.spec_from_file_location`, and narrow the bare
    `except` so a real import error is still distinguishable.
  - `SourceModule` is the dead PyCUDA scaffolding. The Stage 4 GPU port removed
    it from `voigt.py` (which now has no PyCUDA remnant at all), so what is left
    is **`constant.py:38` and `linear.py:39`** — plus a commented-out
    `#from pycuda.compiler import SourceModule` in `lineemission.py`,
    `lsfspline.py`, `phionxs.py`, `splineabs.py`, `main.py` and `model_eval.py`.
    Clear the remaining two here.

**6.4 — Unit tests for this stage's stable surface (do last). [DONE 2026-08-06]**
- Following the cross-cutting unit-test policy
  (`claude_prompts/refactor_code_unit_tests.md`), add `unit`-marked tests for the
  Stage 6 stable surface: the CLI argument parsing / command dispatch (6.1) and
  any pure GUI-backing logic (6.2) that can be exercised without a display. Keep
  them fast and isolated (no full fits); the existing `unit` CI job picks them up
  automatically. This closes out the incremental unit-test coverage across all
  stages.
> **Scope narrows with 6.2 (2026-08-04):** with the GUI deferred there is no
> GUI-backing logic to cover, so 6.4 is the CLI surface — `parser()` option
> parsing, `optarg`'s Namespace -> `argflag` mapping (every flag, including the
> ones 6.1 adds), and the `modname` bug above. Nothing in 6.5 needs new tests: it
> must not change behaviour, and the Stage 0 gate is what proves that.
> **Started early (2026-08-04):** `tests/test_cli.py` exists, 15 tests, written
> with the `modname` fix so the fix did not land unpinned. It covers the parser,
> the Namespace -> `argflag` mapping for every current flag, and that an absent
> flag does not overwrite a value set in `settings.alis` or the model file. 6.4
> extends it to whatever 6.1 adds (`--set`, the Stage 4 knobs) and closes out.
> Baseline: `pytest -m unit` = **716 passed, 31 skipped** (701 at the end of
> Stage 5, +15).

**6.7 — Bring `settings.alis` and `ALIS_workflow.md` up to date.
[NEW — RJC, Q6.5 option (a). DONE 2026-08-06 — `settings.alis` was deleted
rather than updated (Q6.21), and `run_alis --list-settings` replaces it
(Q6.24).]**
- Stages 3–5 added user-facing settings that are documented nowhere. A setting
  nobody can find is not really shipped, and with 6.3 deferred there is no
  Sphinx build coming to fix it.
> **Superseded in part by Q6.21 (2026-08-05): `settings.alis` is to be deleted,
> not brought up to date.** The bullet below is kept for the record — the counts
> are still the evidence that the settings are undiscoverable — but the work
> becomes "make the dataclass authoritative, delete the file", and 6.7 reduces to
> `ALIS_workflow.md` plus whatever replaces the file as the place a user can see
> what they may set (Q6.24).
- **`alis/data/settings.alis`** lists 55 of the 86 settings in `ArgFlag`
  (measured 2026-08-04). Add the user-facing ones that are missing: the ten
  `plotscript` keys (5.3), `out report` / `out reportsig` (3.2),
  `sim convergetest` / `sim convergesig` (3.3), `run cache` (3.1),
  `chisq atol`, `plot only`, `plot xaxis`, `sim repeat` / `random` / `startid` /
  `perturb` / `systematics`, `run capvalue`, `run convergence`, `run datadirc`,
  `out reletter`, `out wavecorr`. Leave the genuinely internal ones out
  (`prognm`, `modname`, `last_update`) — and say so in a comment, so the next
  reader does not have to re-derive it.
- **`doc/ALIS_workflow.md`** does not mention `run gputhresh` (4.3), the
  `bufferpix` data-line keyword (5.4), the `plotscript` section (5.3),
  `out report` / `reportsig` (3.2) or `sim convergetest` / `convergesig` (3.3),
  and it still refers to `atomic.xml` six times although `atomic.ecsv` has been
  the default since 5.2. §2.3 was updated in 5.7 for the signed-exponent rule, so
  that one is current.
- Run **after** 6.1 and 6.5, so the flag names and the retired `out sm` are
  final before they are written down.
- Verification: a test asserting every non-internal `ArgFlag` field appears in
  `settings.alis` — cheap, and it keeps the file from drifting again. That is
  the one piece of 6.7 that belongs in 6.4's batch.

## Status (2026-08-06)

Stage 6 is implemented. `claude_prompts/logs/refactor_code_stage6_log.md` has
the detail; `claude_prompts/deferred_work.md` has everything deliberately left
undone, across the whole refactor.

| task | state |
|---|---|
| 6.1 CLI | **done** — modname bug, exit status, precedence, `.mod.out` override block, `--set`, `--list-settings`, flag tidy, `optarg` refactor |
| 6.1b defaults | **done** — `alis/data/settings.alis` deleted; `ArgFlag` is the only source (Q6.21) |
| 6.5 lint | **done** as Q6.22 option (b) — all 39 modules un-excluded from ruff, 36 F821 + 7 E711 + 2 further real bugs fixed; the reformat is deferred |
| 6.7 docs | **done** — `ALIS_workflow.md`, README badges, `deferred_work.md` |
| 6.4 tests | **done** — `pytest -m unit` 717 -> **736** |
| 6.2 GUI, 6.6 plot-script GUI | deferred to the GUI design document (Q6.2, Q6.4) |
| 6.3 Sphinx docs | deferred until v2.0.0 (Q6.3) |

**Gate.** `pytest -m unit` 736 passed / 31 skipped; `pytest -m "unit or fast"`
791 passed / 31 skipped / 0 failed; `ruff check alis/ tests/` clean; and the
full **`pytest --run-slow` = 859 passed, 31 skipped, 0 failed (2:21:30)**
against the finished tree. The baseline before the stage was 840/31/0 and the
unit batch grew by 19, so the whole difference is the new tests -- no
regression case moved.

**No golden reference was regenerated, and none needed to be** —
`compare_mod_out` ignores the settings block, so the new `#[cli]` lines do not
move the comparison. Q6.26 step 5 budgeted 2.5 h for a regeneration that turns
out to be optional (it would make the references *representative* of what the
writer emits, nothing more).

**One reading taken rather than asked about.** Q6.25's answer proposed "a
commented out list of commands", while Q6.12 and Q6.20 both say the settings
must be **live** so that `run_alis model.mod.out` reproduces the run. Those are
reconcilable only one way, and that is what is implemented: the persisting
settings are written live, and alongside them, commented, go the value each
replaced and the flags that deliberately did not persist. Say if you meant the
overrides themselves to be commented out — it is a two-line change, but it would
mean a re-run no longer reproduces the original.

## Skills to use for this stage

- `run-tests` — the Stage 0 gate, run as each file leaves the 6.5 exclusion list.
- ~~`gui-dev`, `gui-component`~~ — for 6.2/6.6, which move to the GUI design doc.
- ~~`build-docs`~~ — for 6.3, deferred; there is no Sphinx tree to build yet
  (no `docs/`, no `.readthedocs.yaml`, checked 2026-08-04).

## Context

- `alis/scripts/run_alis.py` (CLI entry; already argparse — see 6.1) and
  `alis/load.py`'s `optarg` / `usage` (the half that is still hand-rolled).
- `pyproject.toml` — the three exclusion lists 6.5 empties (`[tool.black]`
  `force-exclude`, `[tool.ruff]` `extend-exclude`, `[tool.isort]` `extend_skip`);
  all three currently hold the same 46 paths.
- `doc/ALIS_workflow.md` (the up-to-date workflow reference),
  `doc/tex_files/` (out-of-date LaTeX, reference only),
  `alis/data/settings.alis` (the shipped settings listing users read).
- `alis/prepfit/` (current GUI, matplotlib `Qt5Agg`) — for the deferred 6.2.
- Plan Q8 (GUI toolkit) — moves to the GUI design doc with 6.2.

## Audit before commencing (2026-08-04)

Everything this document names was checked against the code as it stands at the
end of Stage 5. Corrections are folded into the tasks above; this section records
what was measured and what is new.

**Paths.** All still exist: `alis/prepfit/`, `alis/scripts/run_alis.py`,
`doc/tex_files/`, `doc/ALIS_workflow.md`, `context/voigt_gpu/`,
`claude_prompts/refactor_code_unit_tests.md`, and every file named in 6.5.
`claude_prompts/logs/refactor_code_stage6_log.md` does not exist yet — expected;
it is created when work starts.

**Stage 5 did not disturb 6.5's scope.** It added two `alis/` modules, both
written clean and neither excluded, and removed none. The three exclusion lists
are byte-identical to one another (46 paths each).

### What 6.1 actually has to do

`argparse` is already in place, so the task is the six items below, not a parser
swap. Measured 2026-08-04.

1. **Live bug: the model filename is taken from `sys.argv[-1]`, not from the
   parser.** `run_alis.main` does `argflag['run']['modname'] = sys.argv[-1]`
   while `args.alis_modfile` holds the correct value. Verified:
   `run_alis fit_spectra.mod -w` leaves `modname='-w'`. Any invocation that puts
   a flag after the model file is broken; the shipped examples and the harness
   all put flags first, which is why nothing has caught it.
2. **`load.optarg` is the hand-rolled half.** It takes the argparse `Namespace`
   as its `argv` argument and copies 14 fields onto `argflag` one `if` at a time,
   then loads `settings.alis` and runs two checks. It also still carries the
   **Python 2** `getopt` block as a dead string literal (`load.py:100-121`,
   including `except getopt.GetoptError, err:`).
3. **`optarg` locates `data/settings.alis` by string-splitting a path on `'/'`**
   — exactly the anti-pattern Stage 5.2 replaced in `load_atomic` with
   `load.atomic_datadir()`. Reuse that.
4. **`argflag['run']['prognm']` is now dead.** `optarg` sets it to `__file__`
   (i.e. `alis/load.py`); Stage 5.2 removed its only real consumer, and the one
   remaining reference passes it to `base.call(prgname=...)`, where the parameter
   **is never used**. `logger.alisheader` also takes a program name and prints
   `python %s [options] model.mod`.
5. **`--help` is untidy.** `load.usage()` is now a stub — every option line in it
   is commented out — so all it returns is `msgs.alisheader('ALIS')`, which
   embeds raw ANSI colour escapes into `--help` (they appear literally when the
   output is piped) and prints `Usage : python ALIS [options] model.mod`
   immediately under argparse's own correct `usage: run_alis ...` line.
6. **Stages 4 and 5 added settings with no CLI flag.** `-g/--gpu` is a
   `store_true`, so it sets `run ngpus` to the **boolean `True`** where the
   config declares `Optional[int]`; there is no `--ngpus N`, and nothing for
   `run backend`, `run gputhresh`, `run shmem`, or any of the ten `plotscript`
   settings. See Q6.7.

### Documentation drift (relevant because 6.3 is deferred)

`doc/ALIS_workflow.md` is now the only user documentation. It does not mention
`run gputhresh` (4.3), `bufferpix` (5.4), the `plotscript` section at all (5.3),
`out report` / `out reportsig` (3.2), or `sim convergetest` / `sim convergesig`
(3.3); and it still refers to `atomic.xml` six times although `atomic.ecsv` has
been the default since 5.2.

`alis/data/settings.alis` lists **55 of the 86 settings** in `ArgFlag`. Of the 31
missing, some are internal (`prognm`, `modname`, `last_update`), but the
user-facing ones include **all ten `plotscript` keys**, `out report`,
`out reportsig`, `sim convergetest`, `sim convergesig`, `run cache`, and
`chisq atol`. Nothing breaks — the defaults live in the dataclass — but this is
the file users read to find out what they can set. See Q6.5.

There is no `CONTRIBUTING.md`, no `docs/` tree and no `.readthedocs.yaml`; those
were all 6.3.

### Audit addendum (2026-08-04, second pass — after the Q6.4–Q6.8 answers)

Re-checked with RJC's answers in hand. Everything the document names is still
correct, and three things changed:

- **The `modname` bug is fixed** (6.1 item 1; see that task). `tests/test_cli.py`
  is new. `alis/scripts/run_alis.py` and `tests/alisrun.py` are the only files
  touched.
- **`out sm`'s blast radius is now measured exactly** — seven sites, ~200 lines,
  no model file and no test affected. Folded into 6.5 as a table.
- **`set_regions` cannot be kept unchanged.** RJC asked for it to stay; as
  written it is a method without `self` and raises `NameError` on call, so
  "keep it" and "clear the F821" are the same one-word edit. Q6.10.

**Measured for the Prompt 3 queries (2026-08-04).** Four numbers that decide how
6.5 and 6.1 are done, recorded here so the queries below do not have to repeat
them:

| measurement | value |
|---|---|
| black's reformat of the 39 excluded `alis/` files | **-8,533 / +18,282 lines**, i.e. 43% of their 19,867 lines rewritten |
| bare `except:` (E722) | 65, of which **40 are in `alis/load.py`** |
| settings precedence | `settings.alis` -> CLI flags -> **model file**, so the model file wins; `-p 0` is overridden by `plot dims` in **44 of 48** example models |
| `load.optarg` callers | 5 — `run_alis` twice *with* `argv`, and `main.py:514,518,532` / `prepfit/specplot.py:427` **without** it, purely to load defaults |

That last one matters for 6.1 item 2: `optarg` is doing two unrelated jobs
("load the default settings" and "apply the command line"), and four of its five
callers only want the first. Splitting them is the refactor; the four callers
must keep working.

Two further things worth knowing before 6.5 starts, both found while measuring:

- **`msgs.error` exits with status 0.** It calls a bare `sys.exit()`, so every
  user-input error ALIS reports looks like success to a shell, a Makefile or CI.
  This is why the `modname` bug could sit in a subprocess-based harness
  undetected: `tests/alisrun.py` asserts on `returncode`, and the returncode was
  0. Not a Stage 6 task as written, but it is a one-line change in
  `alis/logger.py` and it makes every other failure visible. See Q6.11.
- **`alis/load.py:315`** reads `if argflag['out']['sm'] or argflag['out']['sm']`
  — the same condition twice. It disappears with `out sm`, but it suggests the
  blind-analysis branch was written by copy-paste and is worth a read-through
  while 6.5 is in there.

## Queries

**Q6.1 — CLI library.** `argparse` (stdlib, no dep) or `typer` (nicer UX, new
dep)? And must the exact current CLI flags be preserved for backward
compatibility, or may they be renamed/cleaned up (with aliases)?

**Response:** We will use `argparse`. It is OK to rename/clean up the CLI flags. Aliases do not need to be provided for backward compatibility.

**Q6.2 — GUI toolkit.** Stay with matplotlib/Qt for the extended GUI, or move to
another toolkit (e.g. a web-based interface)? (Deferred from plan Q8.)

**Response:** We will drop this step from Stage 6 and defer it to a separate design document.

**Q6.3 — Docs hosting.** Is there a ReadTheDocs project/account to target, and
should the API reference be generated via `sphinx-autodoc` from the (now
type-annotated) code?

** Response:** We will drop this step from Stage 6 and defer it to a separate development item later, once version 2 of ALIS is ready for deployment.

**Q6.4 — Does 6.6 go with the GUI?** 6.6 is "generate the Stage 5.3 plotting
script *interactively from the GUI*", so it is a GUI surface and Q6.2 has just
deferred the GUI. The emitter it sits on is finished and tested (5.3), so nothing
is lost by moving it. I propose 6.6 moves to the same design document as 6.2,
leaving Stage 6 as 6.1 -> 6.5 -> 6.4. Confirm, or is there a non-GUI form of 6.6
you want now (e.g. `run_alis --plotscript metals <model>.mod.out`, which would
regenerate the script from a finished fit without re-fitting)?

** Response:** Yes, 6.6 will be deferred to the GUI design document along with 6.2.

**Q6.5 — Where does the settings documentation go, now that 6.3 is deferred?**
Stages 3–5 added user-facing settings that appear in neither
`doc/ALIS_workflow.md` nor `alis/data/settings.alis` (measured above:
`plotscript` ×10, `out report`/`reportsig`, `sim convergetest`/`convergesig`,
`run cache`, `run gputhresh`, `chisq atol`, and the `bufferpix` data-line
keyword; the workflow doc also still says `atomic.xml`). Options: (a) add a small
task to Stage 6 that brings `settings.alis` and `ALIS_workflow.md` up to date —
cheap, no new infrastructure, and it makes the features discoverable now;
(b) leave it all for the deferred Sphinx work. I lean (a): a setting nobody can
find is not really shipped, and it is an hour's work rather than a docs
migration.

**Response:** We will go with option (a) and add a small task to Stage 6 that brings `settings.alis` and `ALIS_workflow.md` up to date.

**Q6.6 — SuperMongo output: delete or repair?** The 22 `slf` F821s in
`alis/plot.py` are all in `prep_arrs`, which is dead — its only caller is
commented out, as is the `save.save_smfiles` it feeds — while the live
`out sm True` branch raises "Sorry, supermongo generated files are not
implemented yet" *after* the fit completes. So `out sm` is a setting that cannot
succeed. (a) Delete `prep_arrs`, `save_smfiles` and the `out sm` setting, which
clears 22 of the 37 F821s outright; (b) repair `prep_arrs` (it already receives
`snip_ions`/`snip_detl`/`posnfit` as arguments, so the fix looks mechanical) and
wire the branch up; (c) leave as is and just silence the linter. I lean (a) —
SuperMongo is long dead as a plotting tool and 5.3 now emits matplotlib scripts,
which is the same need served better.

**Response:** We will go with option (a) and delete `prep_arrs`, `save_smfiles` and the `out sm` setting, which clears 22 of the 37 F821s outright.

**Q6.7 — How much CLI surface should 6.1 add?** `-g/--gpu` is a `store_true`
that sets `run ngpus` to `True` rather than a count, and Stages 4–5 added
`run backend`, `run ngpus`, `run gputhresh`, `run shmem` and the ten `plotscript`
settings with no flags at all. Do you want (a) `--gpu` fixed to take an optional
count plus flags for `--backend`, `--gputhresh` and `--shmem` (the Stage 4
performance knobs, which are the ones you would want to vary between runs
without editing the model file); (b) that plus a general
`--set "run backend cpu"` escape hatch that accepts any `settings.alis` line, so
the CLI never falls behind the config again; or (c) leave the flag set as it is
and only fix the `ngpus` type? I lean (b) — it is about ten lines, and it is the
reason the CLI drifted in the first place.

**Response:** We will go with option (b) and add a general `--set "run backend cpu"` escape hatch that accepts any `settings.alis` line, so the CLI never falls behind the config again.

**Q6.8 — May 6.5 change `alis/plot.py`'s and `specplot.py`'s dead code, given
neither is covered by the Stage 0 gate?** The regression harness exercises
fitting, not plotting: `prep_arrs`, `set_regions` and the `out sm` path are not
run by any test. So for those specific edits the gate proves nothing, and I would
be relying on reading alone. Do you want me to (a) proceed anyway for code that
is probably unreachable (no callers), (b) add unit tests first for whatever is
reachable in `plot.py` before touching it, or (c) leave both files on the
exclusion list and record them as a known remainder?

**Response:** Please remove all references to `sm`. `prep_arrs` can also be removed.
Please do not remove `set_regions`, as this is something in prepfit that will eventually
be used by the GUI.

**Q6.9 — Retiring `out sm`: hard error or a deprecation warning?** `set_params`
rejects an unknown setting with "Settings contains bad line (arg 2)" and exits,
so once `OutConfig.sm` is deleted, any user `.mod` file containing `out sm False`
stops working — including files that only ever set it to False and never wanted
SuperMongo at all. Nothing in this repository is affected (no `.mod` sets it),
but your own working models might. Options: (a) delete it outright, as you said,
and let such a file fail loudly; (b) delete the field but keep a small
"retired settings" list in `set_params` that warns "`out sm` was removed in v2;
delete this line" and continues. (b) is about six lines and gives the same end
state, so I lean (b) — but it is your call, and (a) is what you actually asked
for.

**Response:** No users use `out sm`, so we can just delete that without a warning.

**Q6.10 — `set_regions`: add the missing `self`?** You asked to keep it for the
GUI. As written (`def set_regions(arr)`, `alis/prepfit/specplot.py:390-392`) it
is a method of `class props` with no `self` parameter, so any call raises
`NameError` — it cannot be used by the GUI in this form, and it is the last F821
in `specplot.py`. I propose `def set_regions(self, arr):`, which keeps the
function, makes it callable, and lets the file leave the exclusion list. Confirm?

**Response:** Done by RJC (2026-08-04) — `set_regions(self, arr)`. `specplot.py`
is now F821-clean and the `alis/` total is 37 -> **36**.

**Q6.11 — Should `msgs.error` exit non-zero?** `alis/logger.py:128` calls a bare
`sys.exit()`, which is status **0**, so every user-input error ALIS reports looks
like success to a shell, a Makefile, or CI. Found while verifying the `modname`
fix: with the bug in place, `run_alis fit.mod -f -w -p 0` printed
"[ERROR] :: The filename does not exist" and still exited 0. Changing it to
`sys.exit(1)` is one line and makes every error visible to a caller. It is
outside 6.1 as written, so I have not done it. Do you want it (a) now, as part of
6.1, (b) as its own small task in Stage 6, or (c) left alone because something
you run depends on the current status? I lean (a) — the regression harness checks
`returncode`, so this is the difference between the harness catching a bad
invocation and silently passing it.

**Response:** The harness should catch a bad invocation. Silently passing
something that's broken is worse. **Done 2026-08-04** — see 6.1 item 7.

*Queries below raised 2026-08-04 in response to Prompt 3, after measuring the
remaining work. Each is one where two readings would produce materially
different code; the measurements behind them are in the addendum above.*

**Q6.12 — Should the command line beat the model file? (6.1, and it changes
existing behaviour.)** Settings are applied in three passes —
`settings.alis` defaults, then the CLI flags in `optarg`, then the model file's
own `par` lines in `load_input` — so **today the model file wins over the command
line**. Measured, not inferred: `run_alis -p 0 fit_spectra.mod` ends up with
`plot dims = '2x2'`, because the model file says so. 44 of the 48 example models
set `plot dims`, and all 48 set `out fits True`, so this is the normal case, not
a corner. It also means the regression harness's `-p 0` — whose comment says it
is there to stop a blocking figure hanging a headless run — **does nothing**;
what actually makes those runs headless is `MPLBACKEND=Agg`.

This has to be settled before `--set` is written, because Q6.7's answer implies
the command line wins, and having `--set` win while `-p` loses would be two rules
in one interface. Options:
- **(a) The command line wins**, for `--set` and for the existing flags alike.
  Most people's expectation, and it makes `--set` genuinely useful. It *is* a
  behaviour change for every example (`-p 0` would start taking effect), though a
  benign one — plotting does not affect any compared output, and I would run the
  fast batch to confirm.
- **(b) The model file keeps winning**, and `--set` follows the same rule. No
  behaviour change, but `--set` then cannot override anything a model file
  mentions, which removes most of the reason for it.
- **(c) `--set` wins, plain flags keep losing.** Preserves today's behaviour
  exactly and still gives you an override, at the cost of two precedence rules.

I lean **(a)**, with the note that it is a real change and should be called out
in the log. Which do you want?

**Response:** I agree with option (a). Is it also possible to make it so that the
`.mod.out` file will store the command line values that were used to override the
model file values? This would be useful for reproducibility, so that if a user runs
a model file with `-p 0` and it overrides the model file's `plot dims`, the `.mod.out`
file will store that the user ran it with `-p 0` so that they can reproduce the same
run later. This should not just be a comment, but the actual setting used should be
stored in the `.mod.out` file, therefore executing `run_alis modelname.mod.out` will
automatically use any command line arguments provided during the `run_alis modelname.mod ...`
step?

**Q6.13 — What gate does each 6.5 batch get?** "Each file must stay green under
the Stage 0 gate as it is un-excluded" is right, but the full harness is 2.5
hours and there are 39 files. Running it per batch is not feasible. I propose:
`pytest -m "unit or fast"` (~7.5 min) after every batch, and one full
`pytest --run-slow` at the end of 6.5. Related: the full harness has not been run
since the Stage 5.7 exponent check and the Q6.11 exit-status change landed —
should I run it once now to establish a clean baseline before 6.5 starts, or is
the green fast batch enough to begin?

**Response:** Yes, let's run it once now to make sure we are still green on the
full set. Then the `unit or fast` can be run after every batch, and one full
`pytest --run-slow` at the end of 6.5.

**Done 2026-08-05: `pytest --run-slow` = 840 passed, 31 skipped, 0 failed
(2:22:31).** The `slow`, `gpu` and `machine_dependent` batches are included, and
all 31 skips are the structural "no GPU implementation" ones. So the tree is
green on the full set *after* Stage 5.5, the 5.7 exponent check, the `load_fits`
continuum fix and the Q6.11 exit-status change — this is the baseline 6.5 starts
from.

**Q6.14 — How do you want to review the reformat?** Black rewrites **43% of the
19,867 lines** in the 39 excluded files: −8,533 / +18,282, net +9,749 (measured
by running `black --diff` on copies). That is not reviewable by eye, and you have
verified previous stages from the git diffs. Options: (a) one commit, trusting
that black is AST-preserving and that the gate is green; (b) black committed
separately, batch by batch, so each diff is attributable to a few files; (c) as
(b), plus a mechanical check I write that parses each file before and after and
asserts the ASTs are identical modulo formatting — stronger evidence than reading
9,749 lines, and about 20 lines of code. I lean **(b) + (c)**.

**Response:** Before doing this, can you please explain why such a reformat is necessary,
or is this something that has already been implemented? If it's something that has already
been implemented and I have previously approved the code changes, then let's go with
option (b) + (c). Otherwise, please explain why the reformat is required, what is the impact
(i.e. optimisation, comments, docstrings), and how much code
will change. Related to this, should we add badges
on the README.md file? The black badge is something like:
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
and we could add the GitHub one as well:
.. |github| image:: https://img.shields.io/badge/GitHub-PypeIt-brightgreen
   :target: https://github.com/rcooke-ast/ALIS
and the license, plus any others that you think would be useful.

**Answer (2026-08-05) — no, none of it is implemented yet, and "necessary" is
too strong.** Taking the three parts of the question in turn.

*What black actually changes.* Layout only. It re-wraps long lines, splits
`if x: return y` into two lines, normalises quote characters and string
prefixes, and strips trailing whitespace. It does **not** touch the text of a
comment or the content of a docstring, does not reorder anything, and does not
optimise: it parses the file, discards the formatting, and prints the same
syntax tree back. That is why it cannot change behaviour, and why option (c)
above — comparing the parse trees before and after — is a complete check rather
than a sampling one.

*How much changes.* -8,533 / +18,282 lines across the 39 files, 43% of their
19,867 lines. It is dominated by one thing: ALIS is written with **1,214**
`if x: y` one-liners, and each becomes two lines. That single pattern accounts
for most of the net +9,749.

*Why do it at all.* Not for the formatting. The reason is that a file is
currently either on **all three** exclusion lists or none, so `ruff` does not run
on these 39 files either — which is why 36 undefined names (`F821`), 7 `== None`
comparisons and 65 bare `except:` sit there uncaught, including the
`imp.load_source` call that silently killed the `systmodule=` feature. The lint
is the point; the reformat is the toll currently charged for it. **They can be
decoupled** — see Q6.22, which puts that option to you rather than assuming.

*Badges.* Yes, worth adding. Two notes on the snippets: the GitHub one is
**reStructuredText** (`.. |github| image::`) and `README.md` is Markdown, so it
would render as literal text; and it says **PypeIt**, so it is a copy from
another project. In Markdown, and pointing at ALIS, I would add:

```markdown
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub](https://img.shields.io/badge/GitHub-ALIS-brightgreen)](https://github.com/rcooke-ast/ALIS)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/rcooke-ast/ALIS/actions/workflows/ci.yml/badge.svg)](https://github.com/rcooke-ast/ALIS/actions/workflows/ci.yml)
```

The CI one is live and self-updating, so it is the most useful of the five — it
goes red when the `unit`/`examples`/`lint` jobs fail. The black badge should only
go up once 6.5 has actually run, or it advertises something untrue. `LICENSE` and `pyproject.toml` both say **BSD 3-Clause**, so that badge is
correct as written. These belong with 6.7 (documentation), not 6.5.

**Q6.15 — E722: how far do we go with 65 bare `except:`?** Narrowing one is a
genuine behaviour change — it changes which exceptions get swallowed — and in
several places the bare except *is* the mechanism: `load_fits` uses it to detect
which of three file formats it has, and `load_ascii` uses it to decide a column
is absent. **40 of the 65 are in `alis/load.py`.** E722 is in the enforced
E4/E7/E9/F set, so a file cannot leave the exclusion list while it has one.
Options: (a) narrow all 65 by hand, with the gate as evidence; (b) narrow only
where the intended exception is unambiguous, and put `# noqa: E722` with a
one-line reason on the rest; (c) drop E722 from the enforced set. I lean **(b)** —
it converts an unexamined blanket catch into a documented decision, without
guessing at exception types in code that has no test covering the failure path.

**Response:** OK, let's go with option (b).

**Q6.16 — `szflx`: enable the branch, or leave it fenced off?** The intent is
unambiguous — `ysize = y.size` is defined eight lines above and the loop is
`for i in range(szflx)`, so `szflx` is `ysize`, and the fix is one word in eight
places. But this is the **wavelength-dependent resolution** branch of four
convolution functions, and it has never executed in its life: fixing the name
turns a guaranteed `NameError` into untested numerics. Options: (a) fix it and
add a unit test that drives an array-valued resolution and checks the result
against a hand-computed convolution — the only option that leaves it actually
usable; (b) fix the name only, and record that the branch is unverified; (c)
replace the branch with an explicit `msgs.error("wavelength-dependent resolution
is not implemented")`, which is honest about the state of it. I lean **(a)** if
you can confirm what the branch is meant to do (convolve each pixel with its own
sigma, read from the `resolution` data column?), otherwise **(c)**.

**Response:** The intention is to convolve each pixel with its own sigma, read from
the `resolution` data column. This assumes that the input data has a Gaussian line
spread function everywhere, but it varies with wavelength. For now, let's go with
option (c), because I think it is better to only support a single value for the time
being. Eventually, we will need to implement a wavelength dependent non-Gaussian
line spread function, but that is a much more complicated problem and will require
a lot of work to implement. So for now, let's just support a single value for the
line spread function.

**Q6.17 — `systmodule=` / the `imp` call: revive or retire?** The user
systematics-module feature has been dead since Python 3.12 removed `imp`, and the
`NameError` is swallowed by a bare `except` and reported as "Could not import
module <theirs>" — so a user is told their own file is at fault. Nobody can be
using it. Reviving it is ~10 lines with `importlib.util.spec_from_file_location`,
plus a small test with a toy module. Retiring it removes a feature that is
documented in the data-line keywords. Your instinct on `out sm` was to delete;
this one differs in that it was intended to work and only broke with a Python
upgrade, so I lean **revive + test**. Which?

**Response:** For now, let's revive + test, but we will need to revisit this in the
future to make sure it is still worthwhile supporting. On this note, could you please
generate a TODO list of all the features that are currently not working or are dead
code, or were not implemented in this refactor, so that we can keep track of them
and decide what to do with them in the future? This includes things like the GUI, and
the wavelength dependent non-Gaussian line spread function, other badges we could
eventually include in the GitHub README.md file, etc. and other things that
have been delayed to the future as part of this refactor.

**Q6.18 — 6.7: live settings lines, or commented examples?** Adding the 31
missing settings to `alis/data/settings.alis` as live lines writes every default
down in a second place, alongside the `ArgFlag` dataclass — so the two can drift,
and the file is read on every run. Options: (a) live lines, matching the file's
current style, **plus a test asserting `settings.alis` and `ArgFlag` agree on
every value** (which also stops future drift, and is the 6.4 test I flagged);
(b) add the new ones commented out, as documentation only, so there is nothing to
drift. I lean **(a)** — the file's purpose is to be the readable listing of what
you can set, and a commented setting reads like one that does not work.

**Response:** I agree with option (a). However, it would be best if there were not
two different places where the default values are stored. I would like to see a single
source of truth for the default values, and have `settings.alis` generated from that
source of truth. This will prevent any future drift between the two. Alternatively, if
`setting.alis` is not used for any purpose other than documentation, then it should be
removed entirely. Eventually, the documentation will be generated from the code itself,
so having a separate file for settings is not necessary.

**Q6.19 — How much flag rationalisation do you want?** Q6.1 now allows renames
with no aliases. Concretely, and with `--set` available for everything else:
`-g/--gpu` must change (it is a `store_true`, so `run ngpus` becomes the boolean
`True` where an `Optional[int]` is declared) — I would make it `--ngpus N`.
Beyond that: `-a/--repeat` and `-r/--random` are easy to confuse, `-m/--model` is
vague, and `-j/--justplot` duplicates `plot only`. Do you want (a) only the
`--gpu` fix, everything else untouched; (b) a small rationalisation I propose in
the log for your approval before applying; or (c) trim the flag list to the
handful used daily (`-f -w -p -v -q -o`) and route the rest through `--set`?
I lean **(b)**. Two small things I will fold in unless you object: the `--help`
banner still says **"ALIS : Absorption LIne Software v1.0"** while
`alis.__version__` is `2.0.0.dev0`, and it injects raw ANSI colour escapes into
`--help` (they appear literally when the output is piped) — I would take the
version from `__version__` and colour only when stdout is a TTY. And
`run prognm` is now dead (Stage 5.2 removed its last real consumer; the only
remaining reference passes it to `base.call(prgname=...)`, which ignores it) —
I would delete the field, as with `out sm`.

**Response:** I agree with option (b), and I also agree with the two small things you propose to fold in.

*Queries below raised 2026-08-05, from the answers to Q6.12–Q6.19.*

**Q6.20 — Which command-line settings may be written into the `.mod.out`?**
(From your Q6.12 answer.) Recording the overrides so `run_alis model.mod.out`
reproduces the run is the right instinct, but some flags describe the *model* and
some describe *this one invocation*, and persisting the second kind changes what a
re-run does:

| flag | setting | persist? | what happens if it is persisted |
|---|---|---|---|
| `-p/--plot`, `-x/--xaxis`, `-l/--labels` | `plot dims`/`xaxis`/`labels` | yes | harmless; describes how to draw it |
| `-f/--fits`, `-m/--model` | `out fits`/`model` | yes | re-run writes the same outputs, which is the point |
| `-j/--justplot` | `plot only` | **no** | the re-run **never fits** — it plots and stops. This alone would break the harness's mode (b), which re-runs the `.mod.out` |
| `-o/--outname` | `out modelname` | **no** | the re-run writes to the *original* output name and clobbers it |
| `-w/--writeover` | `out overwrite` | **no** | the re-run silently overwrites without asking |
| `-a/--repeat`, `-r/--random`, `-s/--startid` | `sim repeat`/`random`/`startid` | **no** | the re-run repeats the whole simulation set |
| `-c/--cpus`, `--ngpus` | `run ncpus`/`ngpus` | **no** | machine-specific; the machine you re-run on may have no GPU |

I propose persisting only the first two rows plus any `--set` whose section/key is
not in the "no" list, written as ordinary live settings under a comment header
(`# --- applied from the command line ---`) so they take effect on re-read but are
still attributable. Does that split match what you had in mind, or would you
rather **only `--set` overrides** were persisted and plain flags never were?

**Response:** This split is indeed what I had in mind. So, please implement only the
first two rows plus any `--set` whose section/key is not in the "no" list, written as
ordinary live settings under a comment header (`# --- applied from the command line ---`)
so they take effect on re-read but are still attributable. Note, there is also a
(commented out) copy of the original .mod file written to the .mod.out file, so that
is also a record of what was run.

**One consequence to weigh either way: this moves every golden reference.** The
harness runs `-f -w -p 0`, so with the split above every `.mod.out.reference`
would gain `out fits True` and `plot dims 0` lines. `compare_mod_out` does not
compare the settings block, so the *comparison* would still pass — but the 41
committed reference files would all differ from what the code now emits, so they
would need regenerating (a 2.5-hour run, as in Stage 5.4). Restricting the feature
to `--set` only would avoid that entirely, since the harness passes no `--set`.

**Response:** That's OK, we can regenerate the harness reference files.
The important thing is that the `.mod.out` file will store the command line values
that were used to override the model file values.

**Q6.21 — `settings.alis` or the dataclass: which is authoritative?** (From your
Q6.18 answer.) You are right that there should be one source of truth, and the
drift you were worried about has **already happened** — measured 2026-08-05, the
shipped file overrides the dataclass on four settings, two of which change fits:

| setting | `ArgFlag` dataclass | `settings.alis` | effect |
|---|---|---|---|
| `chisq fstep` | 1.0 | **20.0** | finite-difference step factor — changes the Jacobian, so it changes every fit |
| `chisq maxiter` | 20000 | **2000** | iteration cap |
| `run ngpus` | `None` | 0 | equivalent in practice |
| `generate skyfrac` | 0.0 | **0.1** | changes generated spectra |

(The five `''` vs `'""'` differences are not real — `main.py:120` and
`check_argflag` handle both.) The file wins today, because `load_settings` reads
it over the dataclass, so **the dataclass defaults are documentation that lies**.
Generating `settings.alis` *from* the dataclass, as you suggested, would therefore
silently change `fstep` from 20.0 to 1.0 and move every reference. The two safe
orders are:
- **(a)** make the dataclass match the shipped file for those four, then generate
  `settings.alis` from the dataclass — behaviour-neutral, and you get the single
  source of truth you asked for;
- **(b)** delete `settings.alis` entirely and keep only the dataclass — also a
  single source of truth, and simpler, but it *does* change those four defaults
  unless (a) is done first.
I recommend **(a) then optionally (b)**: fix the disagreement first, in its own
commit with the gate green, so that if anything does move it is attributable.
Which do you want — and is `fstep = 20.0` deliberate?

**Response:** First, let's make the dataclass match the settings.alis file for
those four settings. Second, run the gate to make sure everything is still green.
Remove the `settings.alis` file entirely and keep only the dataclass. This will
ensure that there is a single source of truth for the default values. Then,
delete the `settings.alis` file, since there is no need for it. Next, run the
harness again to make sure everything still works. Finally, yes, `fstep = 20.0`
is deliberate, and it is the value that should be kept.

**Q6.22 — Should the lint and the reformat be decoupled?** (Following the answer
to Q6.14 above.) The 36 F821s, 7 E711s and 65 E722s are worth fixing on their own
merits; the 9,749-line black diff is the toll for them only because a file is
either on all three exclusion lists or none. It does not have to be:
- **(a) As written** — black + isort + ruff together, per batch: empties the list,
  costs the large diff, and new code in those files is auto-formatted thereafter.
- **(b) Lint now, format later** — remove the 39 files from ruff's
  `extend-exclude` only, leaving black's and isort's lists alone. CI then guards
  them against F821/E711/E722/E9 immediately, the diff is just the bug fixes
  (tens of lines, all reviewable), and the reformat becomes a separate decision
  you can take when it suits.
- **(c) Format only the files you are already editing**, letting the list shrink
  as a side effect of other work.
I lean **(b)**: it buys the whole safety benefit for a diff you can actually read,
and it does not foreclose (a). If you want (a), my Q6.14 answer stands —
batch-by-batch commits plus the AST-equivalence check.

**Response:** OK, let's do (b) at this time, but we should add (a) as a
deferred task to the deferred-work list, so that we can do it in the future.

**Q6.23 — Where should the deferred-work list live, and how wide is it?** (From
your Q6.17 answer.) I read the scope as everything knowingly left unfinished by
this refactor: the GUI (6.2) and interactive plot-script generation (6.6), Sphinx
docs and `CONTRIBUTING.md` (6.3), the wavelength-dependent LSF (Q6.16, and the
non-Gaussian version behind it), `systmodule=` pending a decision on whether it
earns its keep, SuperMongo (removed, recorded so it is not re-added by accident),
`examples/brokenpowerlaw`'s best-accepted-vs-rejected-step exclusion (Stage 0
Q0.12 / 5.4), `load_fits`'s remaining oddities, the `1e4` micro-syntax trap, the
onefits interactive menu, and the badges above. Two questions: (a) should it be
one **user-facing** file (`doc/KNOWN_LIMITATIONS.md`, listing what does not work
and what to use instead) or one **developer** file
(`claude_prompts/deferred_work.md`, listing why each was deferred and what
finishing it involves)? I would write the developer one, since most entries are
not things a user can act on. And (b) should it also carry the items each stage
log records as deferred, so it is the single index — which means reading all eight
logs, an hour or so of work, but it is the only way it ends up complete.

**Response:** It should be just one developer file. It should provide enough details
and references to the logs and `claude_prompts/` files so that when we continue this
work at a later date, we have sufficient references to the originally intended work.

*Queries below raised 2026-08-05, from the answers to Q6.20–Q6.23.*

**Q6.24 — With `settings.alis` deleted, where does a user see what they can
set?** Your two answers pull slightly against each other and I would rather ask
than pick. Q6.5 created 6.7 precisely because the Stage 3–5 settings are
undiscoverable ("a setting nobody can find is not really shipped"); Q6.21 now
deletes the file that was going to be the listing. After the deletion the only
record of the 86 settings is the `ArgFlag` dataclass in the source, and the
Sphinx docs that would have rendered it are deferred (6.3). Options:
- **(a) `run_alis --list-settings`** — prints every section, key, default and the
  field's comment, straight from the dataclass. About 15 lines, cannot drift, and
  it works from an installed wheel with no source checkout.
- **(b) A generated settings table in `doc/ALIS_workflow.md`**, written by a small
  script, with a unit test asserting the table matches the dataclass so it cannot
  go stale.
- **(c) Both.**
- **(d) Nothing until the Sphinx docs land.**
I lean **(c)**: (a) is what you want when you are at the terminal mid-fit, (b) is
what you want when deciding whether to install. Either way this is what 6.7
becomes, since the file it was going to update will no longer exist.

**Response:** This should be option (a). The Sphinx docs will be provided before
the release of v2.0.0, so that will provide the user with a more detailed description
of the settings and how to use them. The `run_alis --list-settings` will provide a quick
reference for the user to see what they can set, and it will be generated from the
dataclass, so it will always be up to date. Furthermore, this utility will be useful,
even when the Sphinx docs are available.

**Q6.25 — When a `.mod.out` is itself re-fitted, should the recorded override
block be replaced or appended?** `save_model` writes `slf._parlines` verbatim,
and `_parlines` is *everything* the reader found in the settings block — so once
a `.mod.out` carries a `# --- applied from the command line ---` block, re-fitting
that file and saving again would write those lines out from `_parlines` **and**
append the new invocation's overrides. The result accumulates, and the next read
triggers `set_params`'s "is set more than once" warning (added in Stage 4.3 for
exactly this class of confusion). Options:
- **(a)** The writer strips any previous override block out of `_parlines` before
  appending the current one, so the block always describes the run that produced
  *this* file. My recommendation — it keeps the file's meaning exact.
- **(b)** Append unconditionally and rely on last-one-wins. Semantically correct,
  but the file grows and warns on every re-read.
- **(c)** Merge key by key, keeping one line per setting.
(a) and (c) end up the same in practice; (a) is simpler to explain. Confirm (a)?

**Response:** Another option is to append `# --- applied from the command line ---`
below the settings block, and then append a commented out list of commands that the
user overrode on the command line, so that the user can see what they overrode and
what the default values are. This would be useful for reproducibility, so that if a
user runs a model file with `-p 0` and it overrides the model file's `plot dims`, the
`.mod.out` file will store that the user ran it with `-p 0` so that they can reproduce
the same run later.

**Q6.26 — How many full-harness runs do you want, and in what order?** The
remaining work contains three behaviour-neutral changes and one that moves every
reference, and taken literally the answers ask for a 2.5-hour run after most of
them. A tighter ordering gets the same evidence from **two** runs plus one
regeneration:

| step | work | risk | gate |
|---|---|---|---|
| 1 | Q6.21 step 1 — dataclass adopts `fstep 20.0`, `maxiter 2000`, `ngpus 0`, `skyfrac 0.1` | none by construction (the file already wins) | `unit or fast` |
| 2 | Q6.21 steps 3–4 — delete `settings.alis`, `optarg` uses the dataclass | none if step 1 is right | `unit or fast` |
| 3 | 6.1 items 2–6 and 6.5 (option b: lint only) | none intended | `unit or fast` per batch |
| 4 | **one full `--run-slow`** — proves steps 1–3 together | — | full harness (2.5 h) |
| 5 | Q6.20 — persist CLI overrides into `.mod.out` | **moves all 41 references** | regenerate + verify each diff is settings-only |
| 6 | **one full `--run-slow`** on the regenerated references | — | full harness (2.5 h) |

That is two full runs rather than four, and it keeps the one change that moves
golden files isolated at the end, where a diff that contains anything but added
settings lines is immediately suspicious. Does that ordering suit, or do you want
the harness run after step 1 and step 2 individually as your Q6.21 answer reads?

**A note on step 5 either way.** Because the harness passes `-f -w -p 0` and the
split persists `-f` and `-p`, every regenerated reference should differ from its
predecessor by **exactly two added lines** (`out fits True`, `plot dims 0`) and
nothing else. I will check that file by file with the harness's own parser, as in
Stage 5.4, rather than assuming it.

**Response:** This ordering is fine, and everything suggested here sounds good to me.


## Prompts

1. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing (please check all filenames mentioned in this document reflect all updates to the code so far, and update as needed). Ask further queries if needed.

2. Please fix the `argflag['run']['modname'] = sys.argv[-1]` bug you identified. Then, read this doc, including my responses to your queries (note I updated my response to Q6.1), and check if any updates need to be made to this document before commencing (please check all filenames mentioned in this document reflect all updates to the code so far, and update as needed). Ask further queries if needed.

3. Please read this doc, and my responses to your queries. Please ask additional queries if anything is unclear about the implementation of Stage 6, or if you need further clarification on any of the tasks.

4. Please read this doc, and my responses to your queries. Please ask additional queries if anything is unclear about the implementation of Stage 6, or if you need further clarification on any of the tasks. Then, please implement Stage 6, and update this document with any changes made to the code or filenames. Also, if (and only if) there are no more queries, please write the list of prompts in this section that will order the tasks to be done in Stage 6, and the order in which they should be done.

---

### The Stage 6 task order (written 2026-08-06, per Prompt 4)

There were no outstanding queries, so this is the order the work was done in and
the order to re-run it in if any of it has to be revisited. Steps 1-8 are
**complete**; 9 is running; 10 is optional and explained below.

Each step is written as a prompt, so it can be handed back verbatim.

1. **Make the dataclass authoritative.** "In `alis/config.py`, adopt the four
   values the shipped `settings.alis` was overriding — `chisq fstep 20.0`,
   `chisq maxiter 2000`, `run ngpus 0`, `generate skyfrac 0.1`. This is
   behaviour-neutral, because the file already won. Then run `pytest -m unit`."
   *(Q6.21 step 1.)*

2. **Delete `settings.alis`.** "Delete `alis/data/settings.alis` and make
   `load.load_settings` return the dataclass. Keep accepting a path so a user's
   own settings file still works. Fix the test that read the shipped file. Run
   `pytest -m "unit or fast"`." *(Q6.21 steps 3-4.)*

3. **Fix the CLI's remaining items 2-6.** "Refactor `load.optarg`: drop the
   dead Python-2 `getopt` block and the path-splitting that located
   `settings.alis`, and replace the fourteen `if`s with a table. Delete the dead
   `run prognm`. Tidy `--help`: version from `alis.__version__`, colour only on
   a TTY." *(Q6.19.)*

4. **Make the command line win, and add `--set` / `--list-settings`.** "Re-apply
   the command line after `load_input`, so an explicit flag beats the model
   file. Add `--set 'section key value'`, repeatable, routed through
   `set_params`, and `--list-settings`, generated from `ArgFlag`. Change
   `-g/--gpu` to `-g/--ngpus N` with a **required** count. Expect `-p 0` to
   start taking effect for the first time — make sure `plot dims 0` actually
   suppresses plotting." *(Q6.7, Q6.12, Q6.19, Q6.24.)*

5. **Record the command line in the `.mod.out`.** "Write the persisting
   overrides as live settings under `# --- applied from the command line ---`,
   with the non-persisting ones as comments. Mark each live line with `#[cli]`
   as a trailing comment — *not* with delimiter comments, which do not survive
   `load_input`. Carry a previous block forward and replace it key by key rather
   than appending. Check three generations of `.mod.out` in a row."
   *(Q6.12, Q6.20, Q6.25.)*

6. **Delete SuperMongo.** "Remove every reference to `out sm`: `prep_arrs`,
   `save_smfiles`, the `main.py` branch, the two `load.py` checks,
   `OutConfig.sm`. No deprecation warning." *(Q6.6, Q6.8, Q6.9.)*

7. **Clear the F821s and E711s, then un-exclude from ruff.** "Fix the remaining
   undefined names: `imp.load_source` -> `importlib` (and narrow the `except`);
   the `szflx` branch -> an explicit "not supported" error; `nput` -> `input`;
   the missing `import sys`; `sidlist`; the two dead `SourceModule`s. Fix the 7
   `== None` comparisons after checking none is a numpy array. Then remove the
   `alis/` paths from ruff's `extend-exclude`, pin
   `select = ["E4","E7","E9","F"]`, and ignore the formatting-shaped rules and
   E722 with the reasons recorded. `ruff check alis/ tests/` must be clean."
   *(Q6.15, Q6.16, Q6.17, Q6.22 option b.)*

8. **Documentation and tests.** "Update `doc/ALIS_workflow.md` for
   `--list-settings` / `--set` / `atomic.ecsv` / `bufferpix` and the precedence
   rule. Add the GitHub, licence, Python and CI badges to `README.md` — but not
   the black badge, which would advertise a reformat that has not happened.
   Write `claude_prompts/deferred_work.md`. Add unit tests for everything steps
   3-7 introduced." *(Q6.5, Q6.14, Q6.23.)*

9. **Prove it.** "Run `pytest --run-slow` and report." *(Done 2026-08-06:
   859 passed, 31 skipped, 0 failed in 2:21:30.)*

10. **Optional: regenerate the references.** Q6.26 step 5 budgeted 2.5 hours for
    this on the assumption that recording settings in the `.mod.out` would move
    every reference. **It does not** — `compare_mod_out` ignores the settings
    block, and the fast batch is green without it. Regenerating would only make
    the committed references *representative* of what the writer now emits.
    Worth doing at the next regeneration for another reason; not worth 2.5 hours
    on its own.

### After Stage 6

Two documents carry what is left:

- `claude_prompts/deferred_work.md` — everything deferred, removed or
  known-broken across Stages 0-6, with the reasoning and where to pick it up.
- A **GUI design document**, not yet written, for 6.2 and 6.6.

The other two large items are the **Sphinx documentation** (6.3, before the
v2.0.0 release) and the **black/isort reformat** (§4.1 of `deferred_work.md`).

