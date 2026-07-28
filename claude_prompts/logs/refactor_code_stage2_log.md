# Stage 2 refactoring log

Core architecture refactor: remove `ClassMain` / circular imports, single
model-function instantiation, typed dataclasses for config, explicit picklable
state for multiprocessing, and stdlib `logging`. Every subtask is gated by the
Stage 0 regression suite (`.mod.out` / `_fit.dat` / `.covar` within tolerance).
No fitting results may change.

## 2026-07-22 (Prompt 1: read doc, review, raise queries)

Read `refactor_code_stage2.md` and the Q2.1–Q2.3 responses. Investigated the
refactor surface before raising queries:

- **Reorg scope (Q2.1):** 28 `alfunc_*` modules → `alis/functions/`;
  `alshift.py` is also function classes (`Ashift`/`vshift`/`vshiftscale`/
  `polyshift`, all `alfunc_base.Base`) so it moves to `functions/shift.py`. The
  registry in `alfunc_base.call()` is a hardcoded keyword→class dict from
  top-of-file imports (plus a `sendatomic` list) — one place to rewrite. Import
  graph dominated by `from alis import almsgs` (37×) / `alfunc_base` (26×)
  across ~38 files. `tests/` does not import `alis` modules (subprocess only),
  so renames don't touch the harness. User functions are hand-registered in
  `alfunc_user.py`.
- **Behaviour-critical (for later tasks):** `msgs.error()` calls `sys.exit()`
  (control flow, not just a message) — the 2.5 logging swap must preserve
  early-exit-on-validation-error. The provided `context/misc/logger.py` is a
  good console/file logger to adapt but imports IPython (removed in Stage 1) and
  is not multiprocessing-aware (2.5 needs MP-safe records).

Raised three queries (recorded as Q2.4–Q2.6), all resolved:
- **Q2.4:** file reorg is an up-front standalone step → new **Task 2.0**.
- **Q2.5:** module rename map approved as proposed (functions/*, load, save,
  plot, utils, convergence, minimise, simulate, main; almsgs→logger at 2.5;
  new config.py at 2.1).
- **Q2.6:** Claude filesystem-moves + rewrites imports, RJC commits (git detects
  renames); msgs.error `sys.exit` semantics and MP-aware logger noted for 2.5.

Doc updated: added Task 2.0 and Q2.4–Q2.6. No code or reference files changed in
this prompt. Ready to commence Task 2.0 on the next prompt.

## 2026-07-22 (Prompt 2: execute Task 2.0 — file reorganization)

Behaviour-preserving moves + import rewrites per the Q2.5 map. New layout:
- `alis/functions/` (new subpackage, `__init__.py` added): the 28 `alfunc_*`
  modules with the prefix dropped, plus `alshift.py` → `functions/shift.py`
  (also function classes). `alfunc_base.py` → `functions/base.py`.
- Top-level renames: `alload`→`load`, `alsave`→`save`, `alplot`→`plot`,
  `alutils`→`utils`, `alconv`→`convergence`, `alcsmin`→`minimise`,
  `alsims`→`simulate`, `alis.py`→`main.py`. `almsgs.py` unchanged (→ 2.5).

Mechanics: a reviewed migration script (`/tmp/alis_reorg.py`) moved the files and
rewrote imports with word-boundary regex (so `alfunc_lsf` vs `alfunc_lsffile`
etc. are safe): Phase A renamed the unambiguous module tokens, Phase B pointed
the moved function modules at `alis.functions.*`, Phase C mapped the `alis`
submodule → `alis.main`. `scripts/run_alis.py` was hand-edited (it aliases the
`alis` submodule as `alis`): now `from alis import load` and
`from alis import main as alismain`. 35 files rewritten.

Fix uncovered by the suite: `functions/phionxs.py` located `phionxsec.dat`
relative to `__file__` (`.../data/phionxsec.dat`), which broke once the module
moved a level deeper. Changed it to
`os.path.join(os.path.dirname(alis.__file__), "data", "phionxsec.dat")` — robust
to module location. (Other data-file loaders — `load.py`, `prepfit/specplot.py`
— did not move relative to their data dir, so were unaffected.)

Verified: 17 representative modules import cleanly; the registry builds (32
keys); `run_alis.py --help` works; the failing `lls` pair passes after the
phionxs fix. Full fast suite re-run to certify green: **57 passed, 19 deselected in 8:11**
(exit 0).

No reference/golden files changed. `almsgs` deliberately left for Task 2.5.

## 2026-07-22 (Prompt 3: execute Task 2.1 — config as dataclasses, increment 1: argflag)

First increment of the incremental dataclass migration (Q2.3): converted the
loading-boundary config `argflag`. The other nested dicts (`modpass`, `fdict`,
`datopt`, …) remain and are later increments.

Investigation: `argflag` is a fixed 7-section dict-of-dicts (run/chisq/plot/out/
sim/generate/iterate) built only in `load.load_settings.initialise()`. Surveyed
every dict operation used on it across the 8 consuming files: only subscripted
get/set, top-level `argflag.keys()` (set_params L172), section `.keys()`
(set_params L173), type-introspection on values (`type(...) is int/bool/...`),
and `copy.deepcopy`. No `.items()/.values()`, no iteration, no `**`/`dict()`
coercion, no pickling.

Implementation:
- New `alis/config.py`: seven typed section dataclasses (`RunConfig`, …) and an
  `ArgFlag` dataclass, field names/defaults/types mirroring `initialise()`
  exactly. A `_DictLike` mixin provides `__getitem__`/`__setitem__`/`keys()`/
  `__contains__` so `argflag['section']['key']` still works unchanged (Q2.3
  transition adapter); attribute access (`cfg.run.ncpus`) also works.
- `load.py`: `initialise()` now returns `ArgFlag()` (dict construction removed);
  added `from alis.config import ArgFlag`.

Verified with a targeted drop-in test: section names, dict + attribute access,
`in` membership, type-introspection (int/bool/float/None preserved), subscript
set, `deepcopy` independence, `KeyError` parity for unknown keys, and a
`set_params` round-trip (`run ncpus 3`, `chisq ftol 1.0E-8`, `run blind False`,
`out verbose 1`) all pass. Representative subset green (8 passed). Full fast
suite: **57 passed, 19 deselected in 8:19** (exit 0).

No reference/golden files changed.

## 2026-07-22 (Prompt 4: Task 2.1 increment 2 — modpass)

Second dataclass increment: converted `modpass` (the parsed model structure).

Investigation: `modpass` is a dict of 12 parallel lists (mtyp, mpar, mtie, mlim,
mlnk, mfix, tpar, mkey, psto, p0, emab, line), built only in
`load.load_model`. Surveyed usage across all 24 consuming files (load 130×,
main 94×, plot, simulate, minimise, and 18 function modules): access is purely
`modpass['key']` get (lists mutated in place via append/index), two subscript
assignments both to existing keys (`modpass['line']`, `_modpass['p0']`), and one
`modpass == None` guard in `minimise.py`. No `.keys()/.items()/.values()`, no
iteration, no membership, no copy/len/deepcopy, no new-key additions.

Implementation:
- `config.py`: new `ModelPass` dataclass — 12 `list` fields
  (`field(default_factory=list)`, so per-instance lists are independent),
  reusing the `_DictLike` mixin.
- `load.py`: `load_model` now builds `modpass = ModelPass()` instead of the dict
  literal; import extended to `from alis.config import ArgFlag, ModelPass`.

Verified with a targeted drop-in test: key order, independent default lists,
in-place append via subscript, subscript set, attribute access, membership, and
`== None` parity (False for an instance — matches the old dict, so the
`minimise.py:1058` guard is unchanged). Representative subset green (10 passed).
Full fast suite: **57 passed, 19 deselected in 8:15** (exit 0).

No reference/golden files changed.

## 2026-07-22 (Prompt 4 repeat: Task 2.1 increment 3 — datopt)

Third dataclass increment: converted `datopt` (per-snip data options).

Investigation: `datopt` is a dict of 11 parallel lists (specid, fitrange,
loadrange, plotone, nsubpix, bintype, columns, systematics, systmodule, label,
yrange), built only in `load.load_data` and assigned to `slf._datopt`. Used in
load (33×), save (20×), plot (10×), simulate (6×). Access is purely
`datopt['key']` get with in-place `.append(...)`; no subscript assignments, no
`.keys()/.items()/.values()`, no membership, iteration, copy or deepcopy.

Implementation:
- `config.py`: new `DataOpt` dataclass — 11 `list` fields
  (`field(default_factory=list)`), reusing `_DictLike`.
- `load.py`: `load_data` now builds `datopt = DataOpt()`; import extended to
  `from alis.config import ArgFlag, DataOpt, ModelPass`.

Verified: drop-in test (key order, independent default lists, append, attribute
access, membership). Representative subset green (8 passed). Full fast suite:
**57 passed, 19 deselected in 8:27** (exit 0).

No reference/golden files changed.

## 2026-07-22 (Prompt 5: Task 2.1 increments 4 & 5 — atmdata then wfe)

### Increment 4 — atmdata (self._atomic)

`atmdata` is built empty in `load.load_atomic` then populated with 8 parallel
numpy arrays (Ion, Wavelength, fvalue, Gamma, Qvalue, Kvalue, Element,
AtomicMass); passed as `atomic=` to every model function and stored as
`self._atomic`. Surveyed all ~30 consuming files: access is purely
`_atomic['key']` get (then numpy indexing / `x in _atomic['Ion']` array
membership); the only subscript assignments are the 8 fill sites plus one
`_atomic['Ion'] =` reassignment — all existing keys. No dict `.keys()/.items()`,
no dict membership/iteration/copy.

- `config.py`: new `AtomicData` dataclass (8 `Optional[Any]` fields defaulting
  to None, filled after construction), reusing `_DictLike`.
- `load.py`: `load_atomic` builds `atmdata = AtomicData()`; import extended.

Verified: drop-in test (empty-then-fill, array-value membership, attribute
access). Full fast suite: **57 passed, 19 deselected in 8:09** (exit 0).

### Increment 5 — wfe (column indices, datopt['columns'])

`wfe` is a 9-key column-index dict (wave/flux/error/continuum/zerolevel/
systematics/fitrange/loadrange/resolution; -1 = absent), built per data line in
`load.load_data` and stored in `datopt['columns'][sp][sn]`. More entangled than
the earlier structures:
- it is **mutated with variable keys** (`wfe[clspid[0].strip()] = ...`,
  load.py:700-701) — but the allowed column names (`colallow`) are exactly the 9
  fields, so the strict `_DictLike.__setitem__` accepts every valid assignment;
- `list(wfe.keys())` is used (load.py:888/947) and, via the nested copy in
  `datopt['columns']`, `save.py` calls `.keys()` then iterates + gets by key.
  `_DictLike.keys()` returns fields in literal order, so save output ordering is
  unchanged.

- `config.py`: new `ColumnMap` dataclass (9 `int` fields, literal order),
  reusing `_DictLike`.
- `load.py`: `load_data` builds `wfe = ColumnMap()`; import extended to
  `ArgFlag, AtomicData, ColumnMap, DataOpt, ModelPass`.

Verified: drop-in test (key order, variable-key assignment, the usecols-building
iteration `usecols += (wfe[wfek[j]],)`, nested save-style access). Subset green
(12 passed incl. lsf_file's `columns`/`resolution` override). Full fast suite:
**57 passed, 19 deselected in 8:21** (exit 0).

No reference/golden files changed.

## 2026-07-22 (Prompt 6: Task 2.1 increments 6 & 7 — ucind then lnkpass)

### Increment 6 — ucind (load_userdata)

(The prompt named `load_datafile`; the only real `ucind` lives in
`load.load_userdata` — `load_datafile` merely dispatches. Same column-loading
area.) `ucind` maps each *present* column name to its row in the compacted
`datain` array, built by incrementing `uccnt` over the columns kept in
`usecols`. Same 9 column names as `ColumnMap`, but a dynamic subset and
different values (compacted positions, not file indices). Access is guarded
gets (`if wfe['x'] != -1: datain[ucind['x']]`); wave/flux/error read unguarded
but always present. No dict `.keys()/.items()`, membership, iteration or copy.

- `config.py`: new `ColumnPosition` dataclass (9 `int` fields default `-1` =
  not loaded), reusing `_DictLike`.
- `load.py`: `load_userdata` builds `ucind = ColumnPosition()`; import extended.

Verified: drop-in test reproducing the build loop (present cols → sequential
positions, absent cols keep `-1`). Full fast suite: **57 passed, 19 deselected
in 8:11** (exit 0).

### Increment 7 — lnkpass (load_links → slf._links)

`lnkpass` is a 3-key dict (opA / opB / exp, parallel lists) built only in
`load.load_links`, returned and stored as `slf._links`, and consumed by every
function's `set_pinfo` (param `lnk`). Usage across ~13 function modules + base +
load is purely `lnk['opA'|'opB'|'exp']` list access — `len`, indexing, nested
indexing, and `in slf._links['opA']` (list membership on the value). No
dict-level ops, no post-construction subscript assignment.

- `config.py`: new `LinkPass` dataclass (3 `list` fields), reusing `_DictLike`.
- `load.py`: `load_links` builds `lnkpass = LinkPass(opA=linka, opB=linkb,
  exp=linke)`; import extended to include `LinkPass`.

Verified: drop-in test (construct-with-lists, len/index/nested-index,
list-membership, independent default lists). Subset green (14 passed incl.
tied/linked-parameter examples). Full fast suite: **57 passed, 19 deselected in
8:09** (exit 0).

No reference/golden files changed.

## 2026-07-22 (Prompt 7: Task 2.1 audit + Task 2.2 — registry / single instantiation)

### Task 2.1 audit (recorded as Q2.7 in the stage doc)

Audited remaining string-keyed structural dicts. Conclusion: no further
standalone `_DictLike` increments warranted for 2.1. Deferred structures:
`fdict` + mpfit `functkw`/`fa` → Tasks 2.3/2.4; `parinfo`/`parbase`
(per-parameter mpfit info) → a dedicated increment alongside the minimiser;
per-function `_keywd`/`_keych`/`_keyfm` (heterogeneous per function) → a later
function-interface stage. Proceeded to Task 2.2.

### Task 2.2 — model-function registry with single instantiation

`_funcarray = [names, classes, instances]` was rebuilt via three `base.call`
invocations at every entry point, and **re-instantiated every `sim repeat`
iteration** (`main.py` __init__, the sim loop, and `initialise()`).

Analysis before changing it:
- `_funcarray[1][k]` is exactly `type(_funcarray[2][k])` and `_funcarray[0]` is
  `list(_funcarray[2].keys())` — verified `base.call()[k] is type(getinst()[k])`
  for every keyword — so the whole registry derives from one `getinst` call.
- Function instances hold **no per-fit state**: no `self._X =` outside
  `__init__` in `base.py`/`voigt.py`, and externally only `_keywd` (reset before
  every use) and `_verbose` (set once from argflag) are assigned. So reusing
  instances across `sim repeat` is behaviourally identical to fresh ones.

Implementation (`main.py`):
- New `build_funcarray(argflag, atomic)` helper: one `base.call(getinst=True)`,
  sets `_verbose` on each instance, derives names + classes. Returns the
  `[names, classes, instances]` list.
- `ClassMain.__init__`, the `sim repeat` loop, and `initialise()` all use it;
  the loop no longer re-instantiates (reuses the start-up registry).

Coverage note: **no example exercises `sim repeat`/`sim random`**, so this path
is not in the Stage 0 suite. Verified instead by (a) a structural-equivalence
test (new build == old three-call build: identical names, `is`-identical class
objects, correctly-typed instances); (b) full fast suite **57 passed, 19
deselected in 8:13** (exit 0, normal path unaffected); and (c) a manual
sim-repeat smoke test — powerlaw example with `sim repeat 2` added produced
`REPEAT000` and `REPEAT001` identical apart from the timestamp and cumulative
running-time (i.e. identical model + parameters + chi-squared), proving the
reused registry is not corrupted across realisations.

Recommend adding a `sim repeat` example to the Stage 0 harness so this path is
regression-covered in future (flagged to RJC).

No reference/golden files changed.

## 2026-07-22 (Prompt 8: full Stage 0 regression checkpoint)

Ran the complete Stage 0 suite to certify all Stage 2 work so far (2.0 reorg,
2.1 seven dataclass increments, 2.2 registry), excluding only the ~1 h
`DH/J0814p5029` minimisation per RJC's instruction:

    pytest --run-slow -k "not (DH and J0814p5029)" -v --durations=15

Deselected exactly 1 test (the DH/J0814p5029 minimisation); the two
`VMP_DLA/J0814p5029` tests (a different, faster object) were kept in.

Result: **75 passed, 1 deselected in 4701.38s (1:18:21)** (exit 0). Every
minimisation (mode a), every fixed-parameter eval (mode b) and the generate
special case are green under the error-based `_fit.dat` check, chi-squared,
parameter and covariance tolerances. Slowest: J1358p6522_original 723 s,
J1419p0829 653 s, DH_orders 488 s. No failures — nothing to investigate.

The full safety net confirms the Stage 2.0–2.2 changes are behaviour-preserving.
No `alis/` code or reference files changed in this prompt.

## 2026-07-22 (Prompt 9: Task 2.3 + 2.4 — extract eval path, remove myfunct_wrap / __dict__)

Scope + placement resolved with RJC (Q2.8): targeted eval-path extraction into a
new `alis/model.py`; `ClassMain` stays as the (slimmer) orchestrator.

### Increment 1 — extract the eval trio to `alis/model.py`

Moved `model_func` / `model_func_ddp` / `myfunct` out of `ClassMain` into
standalone functions taking an explicit `state` (mechanical `self`->`state`).
`ClassMain` keeps thin delegating methods, so every call site + the (then still
present) `myfunct_wrap` worked unchanged with `state` = the instance. New
`alis/model.py` (imports copy/np/msgs/load/save; no cycle). Full fast suite:
**57 passed** (8:18).

### Increment 2 — FitState + remove myfunct_wrap and the __dict__ copy

Analysis that shaped the design:
- The minimiser calls `fcn(p, **functkw)`, so `myfunct` was reordered to
  `p`-first with `state` as a kwarg (mirroring the old `myfunct_wrap`).
- `load_par_influence` is value-dependent (it evaluates the shift model to get
  wavelength ranges), and the old `__dict__` reconstitution has a real, golden-
  baked asymmetry: the base call (`ddpid=None`) recomputes `_pinfl` for the
  current `p` on a throwaway, while derivative workers use the fa-build (`p0`)
  `_pinfl` snapshot. A naively shared `FitState` would leak the base recompute
  to the workers and change results.
- Fix: `_minimiser_eval` does `copy.copy(state)` per call — the exact mechanical
  equivalent of `ClassMain(getinst) + instance.__dict__.update(fdict)` (new
  object, shared value refs, local rebinds). This guarantees byte-identical
  behaviour, so the fast suite is a sufficient gate (a missing field ->
  AttributeError, caught; mechanical equivalence -> numerical identity).

Implementation:
- `alis/model.py`: new `FitState` dataclass (30 `Any` fields = exactly the
  attributes the eval path -> `load_par_influence`/`load_subpixels` read/write,
  enumerated by grep) + `FitState.from_orchestrator` (shared refs, like
  `fdict = self.__dict__`) + `_minimiser_eval` (per-call `copy.copy`). `myfunct`
  signature reordered.
- `main.py`: deleted `myfunct_wrap`; `myfunct` delegator now
  `model_eval.myfunct(p, ..., state=self)`; fit driver builds
  `fa = {'x','y','err','state': FitState.from_orchestrator(self)}`; the three
  `alfit(myfunct_wrap, ...)` -> `alfit(model_eval._minimiser_eval, ...)`; the
  onefits `fa['fdict'] = self.__dict__` -> `fa['state'] = FitState...`.
- `simulate.py`: same swaps in all three sim functions (3 FitState builds, the
  direct output=2/3 calls and `alfit` now use `model_eval._minimiser_eval`).
- Name-collision fix: the module `model` clashes with the ubiquitous local
  variable `model` (the model spectrum); imported as `model_eval` in
  `main.py`/`simulate.py`.

Verification: subset (both fit modes) 15 passed; **full fast suite 57 passed**
(9:17); and a manual `sim random 2` run of the (Stage-0-uncovered) `simulate.py`
path completed clean (exit 0, wrote `sims/*.rand1`). Extra medium-batch gate for
shift-using context fits (`pytest -m medium`): **16 passed, 60 deselected in
45:11** (exit 0) — the value-dependent `_pinfl`/shift path is behaviour-
preserved, confirming the `copy.copy` isolation is exact.

No reference/golden files changed.

## 2026-07-22 (Prompt 10: rename model.py -> model_eval.py)

Per RJC, resolved the module/local-variable name collision by renaming the file
rather than aliasing the import. `alis/model.py` -> `alis/model_eval.py`
(filesystem move; RJC commits, git detects the rename). Updated the 4 import
sites (`main.py` x1, `simulate.py` x3) from
`from alis import model as model_eval` to `from alis import model_eval`; the
`model_eval.<name>` call sites are unchanged. Confirmed no other importer of the
module. Imports resolve; full fast suite **57 passed, 19 deselected in 8:15**
(exit 0). No reference/golden files changed.

## 2026-07-23 (Prompt 11: Task 2.5 — almsgs -> logging)

Approach + format resolved with RJC (Q2.9): a `logging.Logger` subclass keeping
the `msgs` API, with ALIS's coloured `[LEVEL] ::` prefixes.

Surface: `msgs` had ~900 call sites (323 `error()`, 277 inline `newline()`, plus
info/warn/bug/work/simulate/test/input) across 38 files, all writing to stderr
(never into the compared `.mod.out`/`_fit.dat`), so console-format changes can't
affect the Stage 0 suite.

Implementation:
- New **`alis/logger.py`**: `AlisLogger(logging.Logger)` exposing the full msgs
  API — `info`/`warn`/`bug`/`work`/`simulate`/`test` (custom levels
  PROGRESS/TEST/SIMULATE/BUG), `error` (emits then `sys.exit()`, Q2.6),
  `newline`/`indent`/`input` (unchanged continuation/prompt strings),
  `alisheader`, `signal_handler`. `AlisFormatter` reproduces the historical
  coloured prefixes exactly. `msgs()` factory returns the shared, per-process
  configured logger; `set_verbosity(0/1/2)` -> WARNING/INFO/DEBUG.
- **MP-aware:** `msgs()` attaches a stderr handler in *every* process; spawned
  Pool workers re-import and reconfigure, so records emitted inside workers
  survive (Q2.6) rather than being dropped by a handler-less logger.
- Bulk swap of all 38 instantiation sites: `from alis import almsgs` ->
  `from alis import logger`; `msgs = almsgs.msgs()` -> `msgs = logger.msgs()`.
  Deleted `alis/almsgs.py` (the Q2.5 rename target). The ~900 `msgs.X(...)`
  call sites are unchanged.
- CLI (Q2.2): added `-q/--quiet` to `run_alis`; `load.optarg` maps it to
  verbosity 0; `logger.set_verbosity` is applied from `argflag['out']['verbose']`
  in `optarg` and `ClassMain.__init__` (`-v` still sets the level).

Verified: isolated logger test (all bands render with ALIS prefixes; `error()`
prints `[ERROR]` then exits; `newline`/`indent`/`input` correct); `run_alis
--help` shows `-q/--quiet`; a headless fit shows 29 `[INFO]` lines vs 2 with
`--quiet`; a `sim random 1` run of the (Stage-0-uncovered) `simulate.py` path is
clean (exit 0, `[INFO]` prefixes render). Full fast suite: **57 passed, 19
deselected in 8:19** (exit 0). Stage-complete full-suite run: `pytest --run-slow -k "not (DH and
J0814p5029)"` -> **75 passed, 1 deselected in 4540.14s (1:15:40)** (exit 0).

No reference/golden files changed.

---

## Stage 2 complete (2026-07-23)

All subtasks done and gated by the full Stage 0 suite (75/75, excluding the ~1 h
DH/J0814p5029 per RJC):

- **2.0** functions/ reorg + module renames
- **2.1** seven config dataclasses (`ArgFlag`, `ModelPass`, `DataOpt`,
  `AtomicData`, `ColumnMap`, `ColumnPosition`, `LinkPass`) in `config.py`
- **2.2** single model-function registry, reused across the fit
- **2.3/2.4** eval path extracted to `model_eval.py`; `myfunct_wrap` + the
  `self.__dict__` copy replaced by a typed, picklable `FitState`
- **2.5** `almsgs` -> stdlib `logging` (`logger.py`), MP-aware, msgs API kept

No fitting results changed anywhere. `ClassMain` remains as the (slimmer)
loader/orchestrator (full dismantling deferred, per Q2.8).
