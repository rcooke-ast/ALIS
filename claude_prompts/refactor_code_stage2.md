# Prompt file for ALIS software refactoring -- STAGE 2

> **Core architecture refactor — the heart of the work.**
> Remove the `ClassMain` / circular-import pattern, instantiate each model
> function once, replace nested string-keyed dictionaries with typed dataclasses,
> fix multiprocessing state passing, and swap the custom messaging for the stdlib
> `logging` module. Every subtask is gated by the Stage 0 regression suite: the
> `.mod.out`, `_fit.dat`, and `.covar` outputs must still match the references
> within the agreed tolerances. Use stdlib `dataclasses` only — no new runtime
> dependencies without RJC sign-off (plan Q5/Q12).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage2_log.md`.
> Prefer many small, independently reviewable commits (plan principle 3).

**2.0 — File reorganization (behaviour-preserving).**
- Standalone up-front step, done before the semantic refactors so the moves are
  separated from logic changes (Q2.4). Pure renames + import rewrites, gated by
  the Stage 0 suite after each batch:
  - the 28 `alfunc_*` modules and `alshift.py` (also function classes) move to
    `alis/functions/` with the prefix dropped (`alfunc_voigt.py` →
    `functions/voigt.py`, `alfunc_base.py` → `functions/base.py`,
    `alshift.py` → `functions/shift.py`);
  - top-level modules drop the `al` prefix per the Q2.5 map
    (`alload`→`load`, `alsave`→`save`, `alplot`→`plot`, `alutils`→`utils`,
    `alconv`→`convergence`, `alcsmin`→`minimise`, `alsims`→`simulate`,
    `alis.py`→`main.py`);
  - `almsgs.py` is left in place until Task 2.5 (superseded by `logger.py`);
  - `config.py` is created in Task 2.1 (new dataclasses, not a rename).
- Executed by filesystem move + import rewrite; RJC commits (git detects the
  renames) — Q2.6.

**2.1 — Config as dataclasses.**
- Replace the nested dicts (`argflag`, `modpass`, `fdict`, …) with typed
  `dataclass`es introduced at the loading boundary in `alload.py`. Add type
  annotations as each structure is converted. Migrate incrementally (adapters are
  fine) so the suite stays green throughout.

**2.2 — Model-function registry with single instantiation.**
- Load each `alfunc_*` class exactly once at start-up into a registry and reuse
  the instances for the whole fit. Remove the repeated `alfunc_base.call(...)`
  re-instantiation in `alis.py`, including inside the `sim` repeat loop.

**2.3 — Remove `ClassMain` / stop passing `self`/`slf`.**
- Extract the model evaluation (`model_func`, `myfunct`) into standalone
  functions/objects that take explicit, typed state (e.g. a `Model` and a
  `FitState`) rather than a monolithic instance. This eliminates
  `myfunct_wrap`, which currently rebuilds a `ClassMain` every χ² iteration via
  `instance.__dict__.update(fdict)`.

**2.4 — Fix multiprocessing state passing.**
- Replace the `fdict` `__dict__` copy with an explicit, picklable state object
  passed to worker processes. Coupled with 2.3 — treat as one unit.

**2.5 — Replace `almsgs` with `logging`.**
- Swap `almsgs.msgs()` for Python's `logging` (levels, handlers, formatters),
  designed to be multiprocessing-aware (log records must survive emission inside
  worker processes). Preserve message content so example output remains
  recognisable.

## Skills to use for this stage

- `run-tests` — the Stage 0 gate after every change.
- `profile-fit` — check the single-instantiation / registry work does not regress
  per-iteration performance.

## Context

- `alis.py` (`ClassMain`, `myfunct_wrap`, `model_func`, `myfunct`), `alload.py`
  (loaders, `modpass`, `argflag`), `alcsmin.py` (minimiser + multiprocessing),
  `alfunc_base.py` (the `call`/`getinst` registry and the model-function
  interface).
- Plan §7 "Current ALIS Code Structure" and §"Known Issues" in `doc/ALIS_workflow.md`.
- Plan Q5 (stdlib dataclasses, no new deps).

## Queries

**Q2.1 — New-core module layout.** Preferred structure for the extracted core —
e.g. new modules `alis/model.py`, `alis/fitstate.py`, `alis/registry.py`,
`alis/config.py` — or keep the refactored code within the existing files to
minimise churn?

**Response:** All functions (i.e. files beginning with `alfunc_BLAH.py`) should be
moved to a new folder called `alis/functions/`. The prefix `al` on all files should be removed,
and sensible filenames should be used (e.g. `alload.py` could become `load.py`, or similar).
All of the configuration dataclasses should be moved to a new file called `alis/config.py`.
This structure will help in organizing the code better and make it easier to maintain.

**Q2.2 — Logging style.** Should user-facing console output keep the current
`[INFO] ::` / `[WARNING] ::` look (via a custom formatter) or adopt a standard
`logging` format? What default level (INFO?) and should there be a `--verbose` /
`--quiet` CLI control (dovetails with Stage 6.1)?

**Response:** I have provided a context example of a logger in `context/misc/logger.py`.
If you need further details, please refer to that file. This is mostly a standard
logging format, but it has been customized to fit the needs of the project. The
default logging level should be set to INFO, and there should be CLI controls for
`--verbose` and `--quiet` to allow users to adjust the logging level as needed.

**Q2.3 — Dataclass boundary.** Is an incremental migration acceptable (convert
one dict at a time, with the dataclass exposing dict-like access during the
transition), given each commit must keep the Stage 0 suite green?

**Response:** Yes, an incremental migration is acceptable. Each commit should convert
one dict at a time, and the dataclass can expose dict-like access during the transition
to ensure that the Stage 0 suite remains green.

**Q2.4 — Reorg sequencing (raised during Prompt 1).** Do the file
reorganization up-front as a standalone step, or interleave it into 2.1–2.5?

**Response:** Up-front, as a standalone behaviour-preserving step (new Task 2.0):
pure moves + import rewrites in small commits, suite green after each, before
the dataclass / registry / logging work.

**Q2.5 — Module rename map (raised during Prompt 1).** Approved mapping:

| Current | New |
|---|---|
| `alfunc_*.py` (27) | `alis/functions/*.py` (prefix dropped) |
| `alfunc_base.py` | `alis/functions/base.py` |
| `alshift.py` | `alis/functions/shift.py` (function classes) |
| `alload.py` | `load.py` |
| `alsave.py` | `save.py` |
| `alplot.py` | `plot.py` |
| `alutils.py` | `utils.py` |
| `alconv.py` | `convergence.py` |
| `alcsmin.py` | `minimise.py` |
| `alsims.py` | `simulate.py` |
| `alis.py` | `main.py` |
| `almsgs.py` | → `logger.py` at Task 2.5 |
| *(new)* | `config.py` (Task 2.1) |

**Q2.6 — Rename mechanics + behaviour notes (raised during Prompt 1).** How are
renames executed given Claude cannot run git, and what behaviour must be
preserved through the later tasks?

**Response:**
- Claude moves files via the filesystem and rewrites imports; RJC commits (git
  auto-detects the renames, preserving history).
- **`msgs.error()` calls `sys.exit()`** — it is control flow, not just a log
  line. The Task 2.5 logging swap must preserve early-exit on validation errors
  (a plain `logger.error()` would let execution continue); `run_alis.py`'s
  `except SystemExit: pass` relies on this.
- The Task 2.5 logger adapts `context/misc/logger.py` (rename `pypeit`→`alis`,
  **drop its `from IPython import embed`** — removed in Stage 1), keeps INFO
  default + `--verbose`/`--quiet`, and is **extended to be multiprocessing-aware**
  (the example is console/file only; log records must survive worker processes).

## Prompts

1. Please read this doc, including my responses to your queries, check if any updates need to be made to this document before commencing, and ask further queries if needed.

2. Please read this doc, and execute Task 2.0

3. Please read this doc, and execute Task 2.1

4. Please perform the next incremental migration step for Task 2.1, converting one nested dict to a dataclass, ensuring the Stage 0 suite remains green.

5. Please perform the next incremental migration step (start with `atmdata`, and once `atmdata` is green for the fast test suite, proceed to `wfe`) for Task 2.1, converting one nested dict to a dataclass, ensuring the Stage 0 suite remains green.

6. Please perform the next incremental migration step (start with `ucind` in `load_datafile`, and once that update is green for the fast test suite, proceed to `lnkpass` in `load_links`) for Task 2.1, converting one nested dict to a dataclass, ensuring the Stage 0 suite remains green.
