# Stage 3 refactoring log

Fitting-engine improvements: profile caching/memoisation (bitwise-identical),
post-fit diagnostics/residual report, and convergence robustness. Every change
is gated by the Stage 0 regression suite; caching must not change the numbers.

## 2026-07-23 (Prompt 1: read doc, review, raise queries)

Read `refactor_code_stage3.md` and the Q3.1–Q3.3 responses. Investigated the
Task 3.1 caching landscape before raising queries:

- The minimiser (`minimise.funcderiv`) uses **two-sided** finite-difference
  derivatives. The base call runs `getemab=True`, so `myfunct`/`model_func`
  return the cached emission/absorption component contributions (`emab`), which
  are passed into each derivative call.
- `model_func` already skips components not in `_pinfl[0][sp][sn]` during a
  derivative (`ddpid` set) — numerically safe only because, for two-sided
  derivatives, un-influenced regions cancel between `fp` and `fm`.
- There is a WIP `model_func_ddp(state, x, p, pp, pos, ddpid, emab, ...)` — the
  prior caching attempt RJC noted "didn't quite work".

Raised two queries (recorded as Q3.4/Q3.5), both resolved:
- **Q3.4:** build Task 3.1 on the existing `emab`/`getemab` machinery — recombine
  cached per-component em (additive) / ab (multiplicative) arrays, recomputing
  only `_pinfl`-influenced components (no division → bitwise-safe).
- **Q3.5:** gate caching behind a `run cache` toggle (default = current uncached
  behaviour) with an A/B self-check asserting bitwise-identical model arrays;
  flip the default once proven.

Deferred to their own prompts: Task 3.2 specifics (region definition, poorly-fit
threshold, `.report` layout, always-on vs opt-in) and Task 3.3 specifics (the
`X·σ` agreement tolerance, the convergence-test menu).

Doc updated with Q3.4/Q3.5. No code or reference files changed in this prompt.
Ready to commence Task 3.1.

## 2026-07-23 (Prompt 2: reconcile stale file references with the Stage 2 renames)

Scanned the Stage 3 doc for pre-Stage-2 module names and cross-checked each
against the code. Updated the two genuinely stale references:
- Task 3.3: `alsims.py` -> `simulate.py`.
- Context: `alcsmin.py` (minimiser) -> `minimise.py`; `alsims.py` (Monte-Carlo)
  -> `simulate.py`.

Verified everything else is already correct / valid:
- `minimise.py`, `simulate.py`, `model_eval.py` all exist; the Monte-Carlo /
  `newstart` machinery (`sim_random`, `sim_systematics`, `perturb`,
  `make_directory`, `newstart`) is in `simulate.py`; `model_func_ddp` is in
  `model_eval.py`, so Q3.1's `alis/model_eval.model_func_ddp()` reference is
  accurate.
- `doc/ALIS_workflow.md` exists and has §6 "Monte Carlo Convergence Testing".
- `context.md` (line 41) resolves to the sibling `claude_prompts/context.md`
  (the Stage 3 doc lives in `claude_prompts/`), so the reference is valid.
- Prompt 2's own text (which names `alcsmin.py`/`alsims.py` deliberately) left
  unchanged.

No discrepancies beyond the renames — the technical claims all hold under the
current filenames, so no clarification was needed. No code or reference files
changed.

## 2026-07-23 (Prompt 3: execute Task 3.1 — profile caching, bitwise)

Design (why the WIP `model_func_ddp` "didn't quite work"): it updated the
*aggregated* per-specid `modelem`/`modelab` in place — emission via
`+= (new-old)` (float addition isn't associative) and absorption via
`/= old; *= new` (division; unstable in saturated cores) — so it was not
bitwise-identical.

Bitwise-safe approach implemented instead (Q3.4): cache each model component's
evaluated array `mout`, keyed by position `(sp, sn, ea, md, mm)`, on the base
call; on a derivative, **reuse the cached array for any component whose
parameters are bitwise-equal** (identical params -> identical `call_CPU`, so the
reuse is exact) and recompute only the changed component — combining with the
*same* `+=`/`*=` sequence as the uncached path. Key properties:
- No division, no aggregate deltas -> bitwise-identical by construction.
- Compares the *effective* params (post-tying), so tied/linked components
  recompute exactly when they should; `variable`/`random` are already skipped.
- Component-agnostic: caches `mout` regardless of function type; shifts /
  convolution / zero-level are untouched (always recomputed) so they stay exact.
- The cache rides in `emab` (`[modelem, modelab, compcache]`), which already
  flows base -> workers via `funcderiv`, so it is pickled to Pool workers and
  read-only there.

Implementation:
- `config.py`: `RunConfig.cache` toggle.
- `model_eval.model_func`: `compcache` param; base call (ddpid None, cache on)
  builds the cache, derivative reuses it; `getemab` returns the cache in `emab`.
- `model_eval.myfunct`: derivative branch extracts `emab[2]` and passes it in.
- `tests/test_cache_equivalence.py`: A/B self-check (Q3.5) — runs 5 examples
  (many absorbers, tied params, emission, phionxs, Chebyshev continuum) with
  `run cache False` vs `True` and asserts the `.mod.out` is byte-identical.

Verified (A/B): metal_line_abs / CNabs / emission_line_ratio / lls / chebyshev
all bitwise-identical (params + chi-squared to the last digit). Speed-up on the
many-Voigt helium34/Her36 fit: **84.0s -> 19.2s (x4.37)**, bitwise-identical.
Fast suite (default off at the time) 62 passed.

Rollout (Q3.5): flipped `RunConfig.cache` default to **True** and ran the full
regression suite under cache-on.

Bug caught by the full suite (and fixed): the first cache-on full run had 4
`covar` failures — helium34 (x3) and VMP_DLA/J0035 — while params/chi-squared
passed. Cause: those fits vary a shift/velocity parameter, which changes the
per-sp+sn *shifted wavelength grid* (`wave`); a component's `mout` depends on
`wave` as well as its own params, but the cache only compared the component
params, so it wrongly reused arrays computed at the old `wave`. Params still
converged identically (so `.mod.out` matched -- which is why the first, `.mod.out`-
only A/B check missed it), but the final Jacobian -> covariance drifted.

Fix: also cache the `wave` per (sp, sn) and require it to be bitwise-equal
before reusing any component there (a shift-changing perturbation invalidates
the whole sp+sn). Verified helium34/Her36 now bitwise-identical on **both**
`.mod.out` and `.covar` (cache off vs on).

The A/B test was strengthened accordingly: it now also compares `.covar`
byte-for-byte, and gained a medium-marked shift-varying case
(helium34/Her36) that exercises the wavelength-grid invalidation. Fast A/B (5
cases) and medium A/B (helium34, incl. covar) both green.

Re-ran the full suite under cache-on: **81 passed, 1 deselected in 2:58:52**
(exit 0) — every case (incl. the 4 previously-failing covar checks) is now
bitwise-correct under caching.

Performance finding (decisive for the default): caching is **model-dependent**.
Durations under cache-on vs the Stage-2.5 uncached baseline:
- helium34/Her36 (compact, many Voigt): **x4.37 faster** (84s -> 19s).
- DH_orders minimisation: **~5x slower** (487s -> 2547s); the whole suite went
  1:15 -> 2:59.

Cause: the component cache rides in `emab`, which `minimise.funcderiv` pickles
to every Pool worker for every derivative. For the large DH fits (hundreds of
spectra x many components) that serialization dwarfs the compute it saves; for
compact models the cache is small and it is a clear win.

Decision: **keep `RunConfig.cache` default = False (opt-in)**, NOT flipped on
(contrary to the provisional Q3.5 plan), because defaulting it on would slow
exactly the "many-parameter, slow" DH/DH_orders fits this stage targets. Users
enable `run cache True` per-model where it helps (compact many-component fits).
(Correction: `run ncpus 1` does *not* avoid the pickling — `funcderiv` always
uses a `Pool` [`mpPool(processes=self.ncpus)`; the serial path is commented
out], so even one worker still receives the pickled `functkw`/`emab` per
derivative.) Flagged to RJC:
the opt-in cache is correct (bitwise, full-suite-proven) and a big win for
compact models; making it a universal win needs the worker cache to avoid
per-derivative pickling (e.g. shared memory / a worker-side cache) -- a
follow-up optimisation.

No reference/golden files changed. Task 3.1 delivered: bitwise-correct profile
cache (`run cache` toggle, default off) + A/B self-checks (`.mod.out` + `.covar`,
fast + medium).

## 2026-07-24 (Prompt 4: caching-optimisation recommendation)

Advisory prompt. Established that the per-derivative worker pickling is of the
*arguments* to `funcderiv` (`functkw` with `FitState`, `emab` with `compcache`),
pickled by value; functions are pickled by reference, so RJC's idea of moving
the cached functions to `model_eval_cached.py` would not reduce the payload.
Recommended optimisations if pursued: (a) subset-pickling (send only each
derivative's influenced-sp/sn cache), (b) shared memory, (c) serial path for
`ncpus=1`. Recorded as Q3.6.

**RJC decision: proceed to Task 3.2.** The subset-pickling optimisation (a) is
logged as a scoped follow-up (make caching a universal win + default it on,
with DH_orders benchmarking); the cache remains the verified opt-in for now.
No code or reference files changed in this prompt.

## 2026-07-24 (Prompt 5: execute Task 3.2 — fit-quality / residual report)

Settled the deferred specifics (both RJC-approved): a **region = a fitted snip**
(contiguous fitrange of one specid); poorly-fit flag = **principled reduced-chi2
deviation** ((redchi2-1)/sqrt(2/Npix) > `reportsig`) OR |runs-test z| >
`reportsig`; report is **on by default** with an `out report` toggle.

Implementation:
- New `alis/report.py`: `build_report` (per-region Npix, chi2, reduced chi2,
  residual mean/scatter, worst outlier, #>Nsigma outliers, Wald-Wolfowitz runs
  z, flag) + `write_report` (prints to console and writes `<model>.report`;
  defensive -- never raises, so a report bug can't abort a fit).
- `config.OutConfig`: `report=True`, `reportsig=3.0` toggles.
- `main.main`: calls `report.write_report(self)` after the fit (best-fit model
  already generated), gated by `out report`; standard-fit path only (not
  generate/justplot).
- Tests `tests/test_fit_report.py`: runs-test z unit test + a fast integration
  test asserting the per-region chi2 sums to the fit total.

Bug found + fixed during development: the per-region residuals initially summed
to ~3423 vs the fit's 594 on helium34/Her36 (He00 inflated). Cause: the report
included in-range pixels ALIS did not fit; fixed by applying the same
`isin(..., _wavefit)` mask as `model_func`/`save_modelfits`. Now the per-region
chi2 sums **exactly** to `m.fnorm` (verified 594.360 == 594.360), and the report
usefully flags real structure (helium34 region 2 flagged on the runs test).

Verified: report renders for single- and multi-region fits; flag logic sane
(tophat's one-pixel edge not flagged; helium34 correlated region flagged);
report tests pass; fast suite **62 passed, 20 deselected** (report on by default
runs on every fit without breaking anything). The `.report` file is extra output
and does not affect the Stage 0 comparisons.

No reference/golden files changed.

## 2026-07-24 (Prompt 6: execute Task 3.3 — convergence robustness)

Settled specifics (both RJC-approved): test menu = **maxdev (default) + scatter
(option)** via `sim convergetest`; agreement tolerance = **3.0 sigma** via
`sim convergesig`.

Gap addressed: the existing `sim newstart` multi-start already runs randomised
restarts (`simulate.sim_random` accumulates each restart's best-fit in
`outrand`) but never *assessed* agreement — that was left to an external script.
Task 3.3 formalises the assessment.

Implementation:
- `config.SimConfig`: `convergetest="maxdev"`, `convergesig=3.0`.
- `convergence.assess_restarts(slf, restarts, refparams, perror)`: for each free
  parameter (perror>0) computes either **maxdev** (max |restart-bestfit|/sigma)
  or **scatter** (restart std/sigma); converged iff all free params <= X sigma.
  Prints a report and writes `<modelname>.converge`; returns
  `(converged, n_failed, n_free)`. Defensive (never raises -> can't abort the
  sims). The report states that `sim random` restarts each fit a fresh noise
  realisation, so the spread includes noise as well as start-dependence.
- `simulate.sim_random`: calls `assess_restarts(slf, outrand[1:], bparams,
  perror)` after the restart loop.
- Tests `tests/test_convergence.py`: maxdev converged; maxdev flags a stuck
  restart; scatter is robust to a single outlier that maxdev flags; fixed
  params excluded.

Verified: unit tests (3) pass; a real `sim random 5` + `newstart` run of the
(Stage-0-uncovered) sim path is clean (exit 0) and writes a `.converge` report
(powerlaw restarts CONVERGED, all 6 free params within 3 sigma). Fast suite
result appended below (the assessment only runs in the sim path, so the normal
fit path is untouched).

Note (recorded for RJC): `sim newstart` confounds start-dependence with the
noise realisation each restart fits, so `scatter` (restart spread vs fit error)
is arguably the more meaningful test for this mechanism; `maxdev` is the literal
Q3.3 criterion and best interpreted for the pure-start case. A pure
start-independence mode (same data, many starts) could be a future enhancement.

Fast suite: **67 passed, 20 deselected** (exit 0) — the normal fit path is
untouched by the sim-only assessment.

No reference/golden files changed.

---

## Stage 3 complete (2026-07-24)

- **3.1 Profile caching** — bitwise-correct component cache (`run cache` toggle,
  **default off**); A/B self-checks (`.mod.out` + `.covar`). 4.37x on compact
  models; slower on large parallel fits (worker-pickling) -> kept opt-in;
  subset-pickling optimisation logged as a follow-up (Q3.6).
- **3.2 Fit diagnostics** — per-region residual report (reduced chi2, scatter,
  outliers, runs test, poorly-fit flags) to console + `<model>.report`
  (`out report`/`out reportsig`); per-region chi2 sums exactly to the fit total.
- **3.3 Convergence robustness** — multi-start assessment (`maxdev`/`scatter`,
  `sim convergesig`) over the `sim newstart` restarts -> `<model>.converge`.

New tests: `test_cache_equivalence.py`, `test_fit_report.py`,
`test_convergence.py`. No fitting results changed; no reference/golden files
changed anywhere in the stage. Open follow-up: the caching subset-pickling
optimisation (make it a universal win + default on).

## 2026-07-24 (Prompt 7: add Task 3.4 — eliminate per-derivative pickling)

Stages 4/5/6 deferred by RJC (they only depend on Stage 2, so ordering is free).
Priority: the per-derivative worker pickling as a general CPU-performance win.

No code implemented (per instruction). Investigated `minimise.fdjac2`/`funcderiv`
and confirmed two inefficiencies: (i) a fresh `Pool` is spawned every iteration;
(ii) the *constant* fit state (`functkw`'s `FitState` + `emab`'s `compcache`) is
re-pickled per derivative (n x per iteration). The `FitState` transfer is a cost
even uncached -> fixing it is a general CPU win, not just a caching one.

Reconfirmed that RJC's `model_eval_cached.py` relocation would not help (payload
is data-by-value; functions pickle by reference). Added **Task 3.4** to the doc
plus **Q3.7** with the full analysis and recommended sequencing: (1) persistent
Pool + chunked derivatives (low-risk, biggest win, bitwise-neutral since each
Jacobian column is an independent two-sided derivative), (2) send constant data
once via a Pool initializer, (3) subset-pickling / shared memory only if
benchmarks warrant. Flip the caching default on only once (1)+(2) make it a net
win on DH_orders; benchmark DH_orders + helium34.

Open query recorded for RJC: Phase-1 (persistent Pool + chunking + send-once)
first and measure, vs going straight for shared memory; and the acceptable
complexity/risk in the `minimise.py` parallel core.

No code or reference files changed.

## 2026-07-24 (Prompt 8: implement Task 3.4 Phase 1)

RJC confirmed Phase 1 (persistent Pool + chunked derivatives + send-once);
shared memory deferred to the GPU work.

Implementation (`minimise.py`):
- Module-level Jacobian workers (`_worker_tie`/`_worker_call`/`_worker_funcderiv`
  /`_worker_chunk`) that are exact replicas of `alfit.tie`/`call`/`funcderiv`
  (only `nfev` dropped -- it was never propagated from workers). Module-level is
  required because `self` now holds the un-picklable persistent Pool.
- `fdjac2`: creates the Pool once per fit (`self._pool`, reused across
  iterations) and computes the `n` Jacobian columns in `~ncpus` chunked
  `pool.map` tasks instead of one `apply_async` per column. Each column is an
  independent two-sided derivative, so chunking/reordering is bitwise-neutral.
- Pool lifecycle: `self._pool=None`, `_close_pool()` (idempotent) called at the
  normal end and the fdjac2-None early return; `__del__` + the Pool's own
  finalizer are the safety net.

Send-once decision (measured, not assumed): a Pool `initializer` was **abandoned**
-- on macOS `spawn` an initializer blocks Pool creation until every worker
finishes its heavy re-import, so pool creation cost **17.86s** and the fast suite
regressed to 26 min. Without the initializer, pool creation is **0.08s** (workers
start lazily) and the constant state travels per chunk (~ncpus x/iteration, still
far below the old n x). This keeps the persistent-Pool + chunking wins without
the spawn-wait penalty.

Bitwise verification (RJC's ask): ncpus=1 (serial single chunk) vs default
(chunked, 13 workers) is **bitwise-identical** on helium34/Her36 (`.mod.out` +
`.covar`) and on DH_orders (`.mod.out`, 3 iterations) -- proving the chunked
parallel Jacobian equals the serial/old path (the workers are exact replicas).

Fast suite: **67 passed, 20 deselected in 9:31** -- regression gone (slightly
faster than the ~10 min pre-3.4 baseline).

Full suite: **86 passed, 1 deselected in 46:45** (exit 0) -- Stage 0 green with
Phase 1. Big general (uncached) speed-up vs the Prompt-8 pre-3.4 baseline (which
was 1:15:40 for *fewer* tests): DH/J1358p0349 445->156s (2.9x), DH/J1419p0829
653->255s (2.6x), DH/J1358p6522 723->328s (2.2x), DH_orders 487->334s (1.5x).
The persistent Pool (spawn once, not per iteration) + chunking sped up every
many-parameter fit, uncached -- the general CPU win intended, not just caching.

Caching benefit under Phase 1 (DH_orders, full fit, same chi-squared
92636.619932 both ways): cache OFF **327s** vs cache ON **366s** = **1.12x**.
Phase 1 collapsed the caching penalty from **5.2x** (Prompt 8: 2547 vs 487s) to
**1.12x** -- the chunking cut the per-iteration compcache pickle from n x to
~ncpus x. Caching is now *almost* a universal win but still ~12% slower on the
huge DH_orders cache, so:
- **Default kept OFF** (per RJC's "default on once it's a universal win"). For
  compact models caching remains a large win (4.37x on helium34); for DH_orders
  it is now a small (12%) loss rather than a 5x one.
- To close the final gap -> **Phase 2 (subset-pickling)**: send only each
  chunk's influenced-sp/sn cache slice (`model_func` already skips the rest),
  which should tip DH_orders to cache <= uncached and let the default flip on.
  Logged as the remaining follow-up.

Phase 1 delivered: bitwise-neutral persistent-Pool + chunked Jacobian; 1.5-2.9x
faster many-parameter fits (uncached); caching penalty 5.2x -> 1.12x. Stage 0
green (86 passed). No reference/golden files changed.

---

## Prompt 9 -- Task 3.4 Phase 2: subset-pickling (COMPLETE)

Implemented per-chunk cache slicing in `alis/minimise.py`. New helpers after
`_worker_chunk`:
- `_param_spsn_map(functkw, compcache)` -- inverts `state._pinfl[0]` into
  `{param_index -> {(sp,sn), ...}}` (which regions each free param influences).
  Returns None if `_pinfl` is unavailable (falls back to sending the full cache).
- `_cache_key_spsn(k)` -- maps a compcache key to its `(sp,sn)` (handles both the
  `('wave',sp,sn)` and `(sp,sn,ea,md,mm)` key shapes).
- `_slice_emab(compcache, param_spsn, chunk)` -- for a chunk's set of params,
  unions their influenced `(sp,sn)` and returns `[None, None, sub]` where `sub`
  is only the cache entries for those regions. Always drops `modelem`/`modelab`
  (the derivative path never reads `emab[0]`/`emab[1]`).

`fdjac2` now builds contiguous chunks via `numpy.array_split`, computes
`param_spsn` once, and sends each worker only its sliced `emab`. So the
per-iteration pickled cache volume drops from ~ncpus x full-cache to the small
per-chunk slice.

Bitwise verification (RJC's ask):
- uncached metal_line smoke: chi-squared 358.112321 unchanged (dropping
  modelem/modelab from the payload is safe).
- Fast cache A/B (5 examples, `.mod.out` + `.covar`): **5 passed**.
- Medium cache (helium34/Her36, shift-varying multi-region): **1 passed**.
- DH_orders, 1-iteration dev check: cached ncpus=1 vs cached default
  **BITWISE-IDENTICAL**; cached vs uncached **BITWISE-IDENTICAL**
  (chi-squared 92636.619932).
- Full suite `pytest --run-slow -k "not (DH and J0814p5029)"`:
  **86 passed, 1 deselected in 44:56** (exit 0) -- Stage 0 green.

Caching benefit under Phase 2 (DH_orders, full fit, same chi-squared
92636.619932 both ways): cache OFF **462s** vs cache ON **250s** = **0.54x**.
Phase 2 completes the arc: caching penalty went **5.2x (Prompt 8) -> 1.12x
(Phase 1) -> 0.54x (Phase 2)**. Caching is now a **universal win** -- ~46%
faster on the huge DH_orders cache and still a large win on compact models
(4.37x on helium34) -- and bitwise-identical everywhere.

**Default flipped ON**: `RunConfig.cache` `False -> True` in `alis/config.py`
(the criterion RJC set -- "default on once it's a universal win" -- is met). The
cache A/B test injects `run cache`/`run cache False` explicitly, so it is
independent of the default; and because caching is bitwise-identical, all golden
references still match with the default on.

Task 3.4 delivered: bitwise-neutral persistent-Pool + chunked Jacobian +
subset-pickled per-chunk cache; caching is now default-on and a universal speed
win; 1.5-2.9x faster many-parameter fits even uncached. Stage 0 green
(86 passed). No reference/golden files changed.

---

## Task 3.4 Phase 3 (shared memory) -- measured, deferred to Stage 4

RJC asked whether Phase 3 (option 4: shared memory for the read-only worker
arrays) is worth pursuing. Decision: **implement it in Stage 4, not now** --
carried forward as new task **4.5** in `refactor_code_stage4.md` (with Q4.4).

Measurement (DH_orders, cache on = new default, Phase 2, `fdjac2` profiled with
temporary env-gated instrumentation, since removed): per Jacobian eval the
`pool.map` wall is ~49 s, but payload *serialisation CPU* is only ~0.25-0.40 s
(**~0.3-0.4%** of wall). The Jacobian is ~95%+ real `model_func` compute
(GIL-bound Python loops), which shared memory does not accelerate -- so Phase 3
gives negligible standalone CPU-time gain. Each eval still ships ~1.1 GB (the
constant `FitState` arrays re-sent per worker/task), so Phase 3's real wins are
per-worker **memory footprint** and being a **prerequisite for the GPU path**
(arrays must be shared/device-resident, and once compute leaves the CPU the
transfer becomes a larger relative share). Hence it belongs with Stage 4's GPU
buffer work. Instrumentation reverted; `alis/minimise.py` unchanged from the
committed Phase 2 state.
