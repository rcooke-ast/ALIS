# Dashboard Stage 14 log

ALIS's sub-pixels (`deferred_work.md` §7.10), the key item (Q13.7's response), with
the item Stage 13 carried over. The plan is in `claude_prompts/dashboard_stage14.md`.

### 2026-11-24 (Prompt 1: the design, Tasks 14.1–14.7, Q14.4 and Q14.5, and Stage 15's draft)

**Read.**
- This stage's document and RJC's responses: Q14.1, none of the four related items
  joins; Q14.2, the item carried over (`lsffile`'s `index`) joins; Q14.3 (a), the
  sub-pixels recorded first, then `nsubmin` 10. Prompt 1 also asks for a design
  document for Stage 15, to be built after the documentation is up to date.
- Stage 13's document and log in full (the form of the tasks, Q13.7, Task 13.2, the
  counts at its close); Stage 12's Q12.9–Q12.13 and its log's Prompts 3–6 and Task
  12.9 (`nsubmin` 10 on the `fast` batch: 27 of 149 failed); Stage 0's document (the
  plan, the stage table, After v1); `deferred_work.md` in full;
  `documentation_stage0.md` in full. Stages 1–11's documents and logs were not read
  again in full: their work is the dashboard's, which this stage does not change
  beyond what Stage 13's and 12's documents record, and the decisions D1–D146 they
  led to are named where they bear on a task.
- The code: `load.load_data` (the data-line keywords, `nsubpix`, `bufferpix`),
  `load.load_subpixels`, `main.py` and `model_eval.py` (where the sub-pixels are
  worked out and used, `renew_subpix`), `save.save_model` (the data lines of the
  `.mod.out`, `bufferpix`), `config.py` (`nsubpix`, `nsubmin`, `nsubmax`,
  `warn_subpix`, `renew_subpix`); `alis/functions/lsffile.py`, `apod.py`, `lsf.py`,
  `lsfspline.py` and `base.py` (how each loads its keywords); the dashboard's
  `model.DATA_KEYWORDS` and its check, `lsfknots.py` (`spix`, `subpixels`),
  `results.continue_plan`, `runs.readable` and `out_values`, `preflight.py`;
  `tests/README.md`, `tests/test_regression.py` (`FIXEDPARAM_EXCLUDE`,
  `MINIMISATION_MACHINE_DEPENDENT`), `tests/alisrun.py` (`make_fixedparam_mod`),
  `tests/conftest.py` and CI's `--skip-machine-dependent`; `doc/ALIS_workflow.md`'s
  tables and `doc/tex_files/description.tex`.

**Found** (no code changed).
- *The sub-pixels.* `load_subpixels` gives each snip `nsubmin`, raised to what its
  lines and its resolution ask for at the values given, capped at `nsubmax`; `main.py`
  calls it once at load with the starting values, and `model_eval` reuses the grid
  unless `renew_subpix`. `save_model` adds `bufferpix` to each data line of the
  `.mod.out`, and nothing records the sub-pixels. A data line's `nsubpix=N` is the
  oversampling per σ, not the sub-pixels, so the new keyword needs a name of its own.
- *The dashboard.* Its parser refuses an unknown data-line keyword ("Unknown keyword
  on a data line"), and the Fit tab reads each run's `.mod.out` with it: the keyword
  must join `DATA_KEYWORDS` with ALIS's change, or every new run would show no best
  fit. `profile.py`, `continuum.py` and `validate.py` call ALIS's `load_subpixels`
  and will follow the keyword; `lsfknots.subpixels` repeats ALIS's sum and will not.
  Continue from best fit copies values alone.
- *The harness* (`python tests/manifest.py`): 43 cases, 37 `fast`, 5 `medium` (2 to 9
  minutes), 1 `slow` (DH_orders, 21 minutes); 5 blind; 17 with a covariance golden.
  Stage 12 measured `nsubmin` 10 on the `fast` batch alone, so the `medium` and
  `slow` cases have not been measured. The `context/` cases reproduce on RJC's
  machine: Stage 13's batches ran without `--skip-machine-dependent`, and all passed.
- *The item carried over*, with a scratch script
  (`/private/tmp/claude-501/-Users-rcooke-Software-ALIS/72dc95e9-748f-42db-8ab2-8de8a21b93c5/scratchpad/s14/keywd_carry.py`):
  one `lsffile` instance loading `name:k3.dat,index:2`, then `name:k3.dat`, gives the
  second line `index` 2 (should be −1); a third line, `index:1` with no file, loads
  with `k3.dat` (should stop: "name must be set"). `apod`: `1.0,kind:uniform`, then
  `1.0`, gives the second line `kind` `uniform`, where `1.0` alone stops with "kind
  must be set". `lsf` loads its keywords the same way (by reading: it needs
  linetools to run). In all three a `blindrange` list is appended to on the instance
  and shared with earlier lines' copies (by reading). `base.Base` and `lsfspline`
  copy their defaults for each line. No model in `examples/` or `context/` relies on
  it (three `lsf` lines and one `lsffile` line, each naming its file).

**Written.**
- `dashboard_stage14.md`:
  - the header's note; notes on the related items (none joins) and on the item
    carried over (it joins, with `apod` and `lsf`);
  - the Design: how the stage is built (agreed first, a bug a test first, references
    checked before they change and blind cases by χ² and |Δ|/σ alone, the order,
    feedback that arrives), what was found, and what stays as it is;
  - Tasks 14.1–14.7, each with what it builds or fixes and its check: each data
    line's own resolution keywords (bug; `lsffile`, `apod`, `lsf`); ALIS records each
    snip's sub-pixels in the `.mod.out` as `subpixels=N` and uses them when given,
    with the dashboard's parser; the dashboard and a recorded line (kept as written,
    the knots dialog, a pre-flight note with its fix, the run's warning, Continue
    from best fit unchanged); `nsubmin` 10 measured on all three batches, each case
    that moves checked, no reference changed; the references regenerated once with
    their sub-pixels, `examples/lsf_spline` back in the evaluation at the best fit;
    the review; the close (against Stage 13's counts: 2356/119 unit (9.2.0:
    2355/120), 340 `gui`, 155 `fast`; ALIS outside `alis/dashboard/` against
    f9f4539), which brings `dashboard_stage15.md` up to date;
  - notes under the responses to Q14.1–Q14.3; Q14.4 (the keyword's name: lean
    `subpixels`; a line recording fewer than its lines ask for: lean, ALIS warns; the
    dashboard: lean, kept as written, a pre-flight note, Continue from best fit
    unchanged) and Q14.5 (`apod` and `lsf` with `lsffile`: lean, fix all three).
- `dashboard_stage15.md` (new), the stage after the documentation, a draft for RJC
  to prioritise: eleven candidates from `deferred_work.md` (the two correctness
  questions in ALIS, `brokenpowerlaw`'s χ² and the simulations' centre; the three
  related items Stage 14 did not take; the dashboard's "Later" items; small fixes in
  ALIS), each with what it would take; a Feedback section; Carried over from Stage
  14 (filled at its close); the new rule that each task keeps the documentation's
  pages up to date; Q15.1–Q15.3 with leans.

**Checked.** Tasks 14.1–14.7 are numbered in order; each has what it builds or fixes
and its check; 14.7 closes the stage, with the checks that it is complete. Every
cross-reference names its present number. No code was changed. The work waits for
RJC's responses to Q14.4 and Q14.5; Tasks 14.1–14.3 are written with their leans.

### 2026-11-24 (Prompt 2: RJC's responses to Q14.4 and Q14.5, Q14.6 and Q14.7, the list checked)

**Read.** RJC's responses: Q14.4 (i) (a), `subpixels=N`, and a clearer name for
`nsubpix` too, with suggestions; (ii) (a), ALIS warns, and a chance of profiles not
properly sampled on their sub-pixels listed in Warnings and options to resolve…;
(iii) (a). Q14.5 (a): `lsffile`, `apod` and `lsf` fixed together. Prompt 2's note on
CI: Stage 13's branch failed CI, with the `gui` job's output in
`context/misc/GitHub_gui_Ubuntu-latest_py313.txt`; `alis_v2` passed but for
`examples`; CI to be checked at the stage's end.

**Found** (no code changed).
- *`nsubpix`.* Each line asks for `nsubpix` · pixel / σ sub-pixels per pixel (`vfwhm`
  and `lsf*` with σ = FWHM / 2√(2 ln 2); `voigt` and `splineabs` with σ = b / √2, read
  in their `set_vars`), so it is the sub-pixels across one standard deviation. 34 of
  RJC's `context/` models set `run nsubpix` (5 ×19, 10 ×3, 100 ×5, 150 ×3, 1000 ×6
  lines); none of `examples/`, and no data line anywhere; 12 set `run nsubmin`, 12
  `run nsubmax`. Settings are read in `load.set_params` and `--set` in
  `load.py`; `--list-settings` is `scripts/run_alis.py`'s; the dashboard's settings
  are keyed by their section and name (`settings.model_settings`), and `lsfknots`
  reads `nsubpix` by name.
- *Under-sampled profiles.* `load_subpixels` caps a line's ask at `nsubmax` without a
  word, and nothing compares a fit's grid with what its best fit asks for; ALIS's
  only sub-pixel warning is `warn_subpix`'s (more than 100, a cost).
- *CI.* The `gui` job on Ubuntu (Python 3.13.18) stopped with a segmentation fault
  (exit 139) in pytest-qt's `_process_events`, during the set-up of
  `test_dashboard_qt_regions.py::test_the_view_stays_where_the_user_put_it[1]`, after
  `[3]` passed. Not looked into further now, as RJC asks: the close checks CI.

**Written** in `dashboard_stage14.md`:
- Task 14.2 (new), a clearer name for `nsubpix`, with the lean of Q14.6:
  `subpix_per_sigma`, the old name still read; the dashboard's settings and keywords;
  its checks.
- Task 14.4 (new), a profile that may be under-sampled warns, with the lean of Q14.7:
  a best fit that asks for more sub-pixels than its fit used, and lines capped by
  `nsubmax`; its checks, with the harness's cases each would fire on counted and
  brought back to RJC if a warning would fire on most.
- The tasks renumbered 14.1–14.9: the sub-pixels recorded 14.3, the dashboard 14.5
  (with every sub-pixel warning listed with what resolves it), `nsubmin` 10 14.6, the
  references 14.7, the review 14.8 (All settings… and the warnings added), the close
  14.9 (Q14.1–Q14.7; F9 and Q13.8 among the decisions refined; the diff of ALIS with
  `main.py` and, if needed, `scripts/run_alis.py`; CI's crash and `alis_v2`'s
  `examples` job checked once the branch is pushed).
- The header's, the Design's (agreed first, the order, what was found, what stays)
  and the Tasks' notes; notes under the responses to Q14.4 and Q14.5; Q14.6 (the new
  name: lean `subpix_per_sigma`; the old name: lean, still read) and Q14.7 (the two
  other ways: lean, ALIS warns of both, listed with their suggestions).
- `dashboard_stage15.md`: Stage 14's task numbers.

**Checked.** Tasks 14.1–14.9 are numbered in order, each with what it builds or fixes
and its check; 14.9 closes the stage, with the checks that it is complete. Every
cross-reference names its present number. No code was changed. The work waits for
RJC's responses to Q14.6 and Q14.7; Tasks 14.2, 14.4 and 14.5 are written with their
leans.

### 2026-11-24 (Prompt 3: RJC's responses to Q14.6 and Q14.7, Q14.8 and Q14.9, the list checked)

**Read.** RJC's responses:
- Q14.6: a change everywhere, without backwards compatibility, to `nsubpixel` where a
  name counts sub-pixels per pixel and `nsubsigma` where it counts them across the
  standard deviation; `nsubmin` → `nsubpixel_min`, `nsubmax` → `nsubpixel_max`;
  perhaps `nsubsigma_min` and `nsubsigma_max` too; any inconsistency in the code
  fixed now, and the convention carried into the documentation and later code. Its
  point (ii) answered (a), the old names still read, which the rest of the response
  contradicts.
- Q14.7: warn only where the sub-pixels should be increased; too many is not a
  problem.

**Found** (no code changed).
- *Where the names are.* Tracked: `alis/config.py`, `load.py`, `model_eval.py`,
  `minimise.py`, `gpu_dispatch.py`, the dashboard's `model.py`, `lsfknots.py` and
  `qt/knots.py`; tests (`test_dashboard_lsfknots.py`, `test_dashboard_qt_knots.py`,
  `test_dashboard_runs_process.py` (`run warn_subpix 1`, Task 13.10's warning),
  `test_dashboard_settings.py`, `test_gpu_dispatch.py`, `test_lsfspline.py`); `doc/`
  (`ALIS_workflow.md`, `tex_files/description.tex`, `newfunctions.tex`, two review
  builders and a mockup); no model of `examples/`. Untracked: 66 of RJC's models and
  references under `context/`, and 4 project bundles under `context/feedback01/`
  (MWC137's: `run nsubpix 150`, `run nsubmin 100`, `run nsubmax 200`, in each
  bundle's model and stored runs). No skill names them.
- *Inside the code.* `nexbin`/`nexbins` in about 220 lines: `load.py` (22),
  `model_eval.py` (9), `main.py`, 26 modules of `alis/functions/` (each `set_vars`
  takes `nexbin=`, `[bintype, nsubpix × the pixel's width]`), `newfunctions.tex`, and
  the dashboard's `profile.py`, `continuum.py` and `lsfknots.py`. `voigt` asks for
  √2 · nsubpix · pixel / b sub-pixels per pixel, σ = b / √2, as Q14.6 said. ALIS's
  log says "sub-pixellation scale", and six functions "Cannot calculate …
  subpixellation". `alis/functions/` is on black's and isort's exclude lists, and
  ruff checks it.

**Written** in `dashboard_stage14.md`:
- Task 14.2, every name of the sub-pixels by one convention, with the leans of Q14.8
  and Q14.9: `nsubsigma`, `nsubsigma_min` (new), `nsubpixel_min`, `nsubpixel_max`,
  `nsubpixel_renew`, the data line's `nsubsigma=` and `nsubpixel=`; `warn_subpix`
  removed (Q14.7's response); ALIS's messages; no old name read, each refused with
  its new name; every model rewritten, RJC's archived first and his bundles (with
  their stored runs) packed again with `alis/bundle.py`; the internal names
  (`nsubpixel`, `datopt['nsubsigma_width']`, `set_vars`'s `sampling=`); the
  dashboard; its checks, with a `grep` for every old name.
- Tasks 14.3–14.9 in the new names: the keyword `nsubpixel=N`; the warnings of 14.3
  and 14.4 at the level `nsubsigma_min`, and none for too many (14.4's check); the
  dialog's suggestions (`nsubpixel_max`); `nsubpixel_min` 10 (14.6); the review shows
  the names (14.8); the close records the convention as a decision for the
  documentation and later code, brings `documentation_stage0.md`'s settings up to
  date, and its diff of ALIS takes the files of the rename (14.9).
- The header's, the Design's (agreed first, the order, what stays: an old name in a
  `.mod.out` is refused) and the Tasks' notes; notes under the responses to Q14.6
  and Q14.7; Q14.8 (the table of names, with `renew_subpix` → `nsubpixel_renew` and
  `warn_subpix` removed; the old names refused with a message naming the new;
  RJC's files rewritten after an archive; the internal names too) and Q14.9
  (`nsubsigma_min` as the warnings' level, 5; no `nsubsigma_max`).

**Checked.** Tasks 14.1–14.9 are numbered in order, each with what it builds or fixes
and its check; 14.9 closes the stage, with the checks that it is complete. Every
cross-reference names its present number, and the tasks use the new names
throughout (the old appear only where a task renames them). No code was changed.
The work waits for RJC's responses to Q14.8 and Q14.9; Tasks 14.2–14.5 are written
with their leans.

### 2026-11-25 (Prompt 3, run again: RJC's responses to Q14.8 and Q14.9, the list checked)

**Read.** RJC's responses: Q14.8, the lean in each point ((i) the table, with
`renew_subpix` → `nsubpixel_renew` and `warn_subpix` removed; (ii) (a), each old name
refused with a message naming its new one; (iii) (a), RJC's files rewritten by
Claude's script after an archive, which is to be directly in `context/` so that RJC
knows where to find it; (iv) (b), the internal names too); Q14.9 (a), `nsubsigma_min`
(5) as the warnings' level, and no `nsubsigma_max`. Nothing in them is unclear, so no
query is raised.

**Found** (read-only `grep` over `alis/`, `tests/`, `doc/`, `examples/`, and reading):
- The old names are in `alis/config.py`, `load.py`, `main.py`, `model_eval.py`,
  `minimise.py`, `gpu_dispatch.py`, 26 modules of `alis/functions/`, the dashboard's
  `model.py`, `lsfknots.py`, `profile.py`, `continuum.py` and `qt/knots.py`, seven
  test files, `doc/ALIS_workflow.md`, `description.tex` and `newfunctions.tex`; and in
  the records of earlier stages: `doc/dashboard/mockups/` (`build_mockups.py`,
  `fit_inspect.html`, and RJC's untracked drafts 1, 2, 4 and 5) and
  `doc/dashboard/stage12/build_review.py`. `CHANGELOG.md` names none.
- `run_alis` is `alis/scripts/run_alis.py`, not `scripts/run_alis.py`, and names
  none: `--set` is read in `load.py` (its "--set does not recognise the setting"
  message), and `--list-settings` prints `config.py`'s list.
- The dashboard's `settings.py` takes ALIS's list (`ArgFlag`) and names none; its
  validator does not check a settings line's name, so a project's model with an old
  name would show no problem until ALIS refused its run.
- Every test file the tasks name exists; Task 13.10's `run warn_subpix 1` is in
  `test_dashboard_runs_process.py`.

**Written** in `dashboard_stage14.md`:
- Task 14.2: the archive at `context/stage14_old_names.tar.gz`, with the originals'
  paths; the table of old names in `config.py`, read by `load.py`; its modules
  without `scripts/run_alis.py` and the dashboard's `settings.py`; the dashboard's
  validator gives a problem for an old name on a settings line or a data line,
  naming its new one, with a fix that renames it in one step (and its test); the
  records of earlier stages stay as written, and its `grep` check says where the old
  names are meant to remain (`config.py`'s table, the tests of the refusal,
  `CHANGELOG.md`'s entry, those records, and the archive).
- Task 14.4's note (the level by Q14.9 (a)); Task 14.9's diff of ALIS without
  `scripts/run_alis.py`.
- The header's, the Tasks' and the Queries' notes; notes under the responses to
  Q14.8 and Q14.9.

**Checked.** Tasks 14.1–14.9 are numbered in order, each with what it builds or fixes
and its check; 14.9 closes the stage, with the checks that it is complete. The
module lists and test files match the code at f9f4539. No code was changed. Every
query is answered; the tasks are ready to be built from 14.1.

## Prompt 4: the tasks (2026-11-25)

*Batches* run on a snapshot of the tree after each task, with pytest 9.0.2 locally
and in a scratch venv with CI's pytest 9.2.0 (script
`/private/tmp/claude-501/-Users-rcooke-Software-ALIS/72dc95e9-748f-42db-8ab2-8de8a21b93c5/scratchpad/batches.sh`,
Stage 13's with this session's paths; RJC's `context/` linked into the snapshot).
The `fast` batch, with the regression harness, runs from the repository (memory
`batch-snapshots`). Before the first task: Stage 13's close (f9f4539), unit 2356
passed with 119 skipped (2355 and 120 under 9.2.0), `gui` 340, `fast` 155.

### Task 14.1 — Each data line's own resolution keywords (`lsffile`, `apod`, `lsf`)

**Tests first.** In `tests/test_lsffile.py` (4) and a new
`tests/test_resolution_keywords.py` (9), each loading the lines as ALIS does
(`load.call_function_load`, one instance for every line). Before the fix 11 of the
13 failed: `lsffile`'s second line took `index` 2 (now −1), a line with no file
loaded the first line's (now "name must be set"); `apod`'s line with no `kind`
took `hanning` (now "kind must be set"); `lsf`'s line with no `name` took `STIS`,
and a line naming only `name:COS` took the first line's `E140H` and `0.2x0.09` (now
the defaults `G130M`, `None`, `1309`); a `blindrange` on two lines gave all three
lines one list, `['0.5', '0.7']`; and loading changed the instance's own defaults.
The two that passed before (each line's `kind`, `unit` per line) guard what stays.

**Fix.** In each loader, the keywords are read into `cpy_keywd = self._keywd.copy()`
and `cpy_keych = self._keych.copy()`, as `base.Base` does; a list keyword is given a
new list (`cpy_keywd[k] + [value]`), never appended to in place; `mkey` takes the
copy. `lsffile`'s reset of `unit` (Stage 13) is now part of it. No other line of
the three files changed. `CHANGELOG.md` (Fixed).

**Found, and not done** (for the close: Stage 15's "Carried over", each with a
query):
- `apod` cannot be evaluated by ALIS at all: `model_eval` calls every resolution's
  `call_CPU` with `mkey=`, and `apod.call_CPU(self, x, y, p, ncpus=1)` takes none
  (`TypeError: APOD.call_CPU() got an unexpected keyword argument 'mkey'`, scratch
  check). Nothing tests a fit with `apod` (`deferred_work.md` §7.7, "example data in
  frequency"), which is where its fix would get its test.
- `lsf.getminmax`, which sizes a data line's buffer, reads the instance's keywords,
  not the line's, and `load_data` calls it before any `lsf` line is loaded: every
  `lsf` buffer is worked out for COS G130M at 1309 Å, whatever the line names
  (`examples/lsf_hst` is STIS E140H). Its evaluation is right: `model_eval` and
  `load_subpixels` give the instance each line's keywords first. Fixing the buffer
  moves `lsf_hst`'s loaded pixels, so its reference.
- When ALIS loads the model again on the same instances (`sim repeat`,
  `iterate model`, `main.py` 161 and 312), each instance holds the keywords of the
  last line evaluated (`_keywd = mkey[i]`), so any function's loader, `base.Base`'s
  included, starts a line from them: an optional keyword a line does not name takes
  that line's value. It needs one of those modes and two lines of a function that
  differ; a fix would give every load fresh defaults in `load.call_function_load`.

**Check.** The 13 tests pass. `test_function_interface.py`, `test_save_helpers.py`,
`test_writer_round_trip.py` and `test_lsfspline.py`: 399 passed, 31 skipped.
Batches (snapshot `t14_1`): unit 2369 passed, 119 skipped (9.2.0: 2368, 120), Stage
13's counts and the 13 new tests; `gui` 340 passed (9.2.0: below); `fast` 155
passed from the repository, the harness and its `test_bundle` mode included, every
reference unchanged (`examples/lsf_file` and `lsf_hst` among them): the stage's
baseline. Pre-commit passes on the five files (black wrapped one line of the new
test).

**Note on 14.1's `gui` batch under pytest 9.2.0.** It overlapped Task 14.2's rewrite
of RJC's `context/` files (the snapshot links `context/`): 23 failures, every one a
`context/` fit whose model had just been rewritten in names the 14.1 code does not
read (`Settings contains bad line (arg 2): run nsubsigma 5`, 50 times). The 9.0.2
run had finished before the rewrite, 340 passed. 14.2's batches cover both pytests.

### Task 14.2 — Every name of the sub-pixels, by one convention

Built in a scratch copy of the tree while 14.1's `fast` batch held the repository,
then copied in (53 files).

**ALIS.**
- `config.py`: `nsubsigma` (5), `nsubsigma_min` (5, new), `nsubpixel_min` (5),
  `nsubpixel_max` (21), `nsubpixel_renew` (False); `warn_subpix` removed;
  `RENAMED_SETTINGS`, `RENAMED_DATA_KEYWORDS` and the messages
  (`renamed_setting`, `renamed_data_keyword`): "'run nsubmin' is now called
  'run nsubpixel_min': rename it."; "'run warn_subpix' was removed: ALIS now warns
  only where a profile has too few sub-pixels. Remove the line."; `DataOpt`'s
  `nsubpix` → `nsubsigma_width`.
- `load.py`: an old name is refused with its message in `set_params` (a model's
  settings), in `apply_cli_settings` (`--set`) and in `load_data` (`nsubpix=`); the
  data-line keyword is `nsubsigma=`; `nsubsigma` below `nsubsigma_min` is refused
  ("nsubsigma is 3 for the data file …, below nsubsigma_min (5). Raise nsubsigma,
  or lower 'run nsubsigma_min' …", Q14.9); `load_subpixels`: `nsubpixel`, the
  warning of too many removed, and the log's line "nsubpixel 6 for spectrum 0,
  segment 0".
- `main.py`, `model_eval.py`, `minimise.py`, `gpu_dispatch.py`: `_nexbins` →
  `_nsubpixel`, `renew_subpix` → `nsubpixel_renew`.
- `alis/functions/` (26 modules, script `s14/rename_functions.py`): `nexbin=` →
  `sampling=` (219 places), and "Cannot calculate voigt subpixellation -- width =
  0.0" → "Cannot work out nsubpixel for voigt: a profile's width is 0.0" (11
  places in 9 modules); `base.set_vars`'s docstring describes `sampling`.
- `alis/scripts/run_alis.py` needed no change: `--set` is read in `load.py`, and
  `--list-settings` prints `config.py`'s list.

**The dashboard.** `model.py`: `DATA_KEYWORDS` takes `nsubsigma` (`nsubpixel` comes
with ALIS's reading of it, in 14.3); the parser's problem for an old name is
ALIS's message. `settings.py`: `old_names(text)` and `rename_step(project, line)`,
which renames the word (or removes a removed setting's line) on the text, so that
it works while the model does not read. `qt/modpanel.py`: a right-click on such a
problem offers "Rename to 'run nsubpixel_min'" (or "Remove the line"), its tooltip
says so, and `renameRequested` makes the step through the window
(`qt/window.py`, `rename_old_name`). `lsfknots.py`, `profile.py`, `continuum.py`,
`qt/knots.py`: the new names. `validate.py` needed no change: its full check runs
ALIS's loaders, whose refusal names the new name.

**Tests.** New `tests/test_subpixel_names.py` (23, `unit`): the settings; each old
name refused with its new name as a setting and through `--set`; `nsubpix=`
refused naming `nsubsigma=`; the new names give each snip the sub-pixels the old
gave (9, 17, 17, 12, 7, 33: the same six cases run with the old names on the 14.1
code gave the same numbers, scratch `s14/try/nsub.py`); `nsubsigma` below
`nsubsigma_min` refused, as a setting and on a data line; `--list-settings` lists
the new names alone. Against the 14.1 code 22 of the 23 fail. In
`test_dashboard_settings.py` (+12): each new name offered with its description, set
and cleared byte for byte; each old setting a problem naming its new one, its
fix one step (the model then reads, ALIS's full check passes on the renamed data
line) and one revert giving the text back; `warn_subpix`'s fix removes its line. In
`test_dashboard_qt_modpanel.py` (+1, `gui`): typed by hand, `run nsubmin 10` stops
the model reading, its problem and tooltip name the fix, and the rename is one
step that one undo reverses. Tests renamed: `test_dashboard_lsfknots.py`,
`test_dashboard_profile.py`, `test_dashboard_qt_knots.py`,
`test_dashboard_settings.py`, `test_gpu_dispatch.py`, `test_lsfspline.py`; Task
13.10's test of a run's warnings (`test_dashboard_runs_process.py`) makes ALIS warn
with `run ncpus 100000` ("You don't have 100000 CPUs!", kept in `run.json`).

**Documents.** `CHANGELOG.md` (Changed: each name, the refusal, `nsubsigma_min`,
`sampling=`; Removed: `warn_subpix`); `doc/ALIS_workflow.md` (the example settings,
the table, the convention in a paragraph, the data-line keyword);
`doc/tex_files/description.tex` (the settings rows, both mentions of the data-line
keyword) and `newfunctions.tex` (`sampling`).

**RJC's files** (Q14.8 (iii); script `s14/rename_context.py`, dry run first). The
66 text files (34 `.mod`, 17 `.mod.out`, 15 `.mod.out.reference`) and 4 bundles
(`MWC137/mwc137.model`, `mwc137_relinked.model`, `testing.model`,
`MWC137_02/data/mwc137.model`: each model, and each stored run's `input.mod` and
`.mod.out`) were archived with their paths in `context/stage14_old_names.tar.gz` (70
entries), then rewritten: 228 changes (settings lines, commented or not; 18
`warn_subpix` lines removed), each listed in `context/stage14_old_names_changes.txt`
beside the archive. A bundle is written back with `alis.bundle.update` (checksums
recomputed); a stored run whose model was current keeps its checksum current (none
was: all three runs were already marked "The model has changed since the latest
run", so no marker changes). Left as written, and listed there: RJC's notes
(`context.md`, 7) and backups (`*.mod.out.orig`, 9), which record what was then.
Afterwards no model, output, reference or bundle in `context/` names an old name,
every bundle reads (its checksums checked), and the four open in the dashboard and
read, their runs' best fits with them (240, 240 and 300 values; `testing.model` has
no run; scratch `s14/open_bundles.py`).

**Check.** `grep` for every old name (also `renew\_subpix`, `warn\_subpix` as
the LaTeX writes them) over `alis/`, `tests/`, `doc/`, `examples/` and
`CHANGELOG.md` finds them only where meant: `config.py`'s table, the dashboard's
handling of them (`settings.py`'s docstring), the tests of the refusal and of the
fix, `CHANGELOG.md`'s entry, the guide's paragraph on the convention, and the
records of earlier stages (`doc/dashboard/mockups/`, `doc/dashboard/stage12/`).
Batches (snapshot `t14_2`): unit 2402 passed, 1 failed, 119 skipped (9.2.0: 2401,
1, 120): `test_config.py::test_every_setting_has_a_one_line_description`, the new
description of `nsubsigma` being 109 characters (at most 80). Shortened ("Sub-pixels
across the narrowest profile's σ: each snip's nsubpixel follows.", 75), and
`test_config.py`, `test_subpixel_names.py` and `test_dashboard_settings.py` pass (54).
`gui` 341 passed (Stage 13's 340 and the new one; 9.2.0: with Task 14.3's batches).
`fast` 155 passed from the repository, the harness and its `test_bundle` mode
included, the `context/` references read in their new settings lines. Pre-commit
passes on every changed file (black reformatted three).

Batches after 14.2 under pytest 9.2.0: `gui` 341 passed.

### Task 14.3 — ALIS records each snip's sub-pixels in the `.mod.out`, and uses them

Built in the scratch copy while 14.2's batches held the repository, then copied in
(12 files and one new test).

**ALIS.**
- `config.py`: `DataOpt` keeps each snip's `nsubsigma` and its data line's
  `nsubpixel` (None when not given).
- `load.py`: `load_data` reads `nsubpixel=N` and refuses anything but a whole number
  of at least 1 ("nsubpixel must be a whole number of at least 1, not '6.5', on the
  data line - …"). `load_subpixels` is split in three: `subpixels_asked(slf, parin,
  level)` (what each snip's lines ask for so that the narrowest gets `level`
  across its σ, unbounded; by default the snip's `nsubsigma`), `bounded_subpixels`
  (`nsubpixel_min`, raised to what is asked, capped at `nsubpixel_max` — the same
  rule as before, written once — or the data line's `nsubpixel` as given), and
  `subpixel_warnings`, called at load (`warn=True` from `main.py`, never at each
  evaluation): one warning naming each snip whose recorded `nsubpixel` is below
  what its lines ask for at `nsubsigma_min`, "Remove nsubpixel= from a data line
  to work its sub-pixels out again." The log's line says ", from its data line".
- `save.py`: `save_model` writes `nsubpixel=N` on each data line, beside
  `bufferpix`: the fit's sub-pixels; a line that gave its own keeps it as written;
  none with `nsubpixel_renew`.
- The dashboard: `nsubpixel` in `DATA_KEYWORDS`.

**Tests.** In `test_save_helpers.py` (+3): a model that takes 8 at its start
(10 after Task 14.6) written with best-fit values whose b is 1 km/s carries
`nsubpixel=8` and reads back with 8, where without it the best fit asks for 21; a
line's own `nsubpixel=3` kept and used; none with `nsubpixel_renew`. In
`test_load_files.py` (+5): 0, −2, 6.5 and x refused naming the line; the value kept
in `datopt`. In `test_subpixel_names.py` (+5): a given N used whatever
`nsubpixel_min` (10) or `nsubpixel_max` (4) say; one warning, naming the snip, N,
what is asked (9) and `nsubsigma_min`, where 3 is recorded; none where 9 is, where
`nsubsigma_min` is 1, or where nothing is recorded. In `test_dashboard_runs.py`
(+1): a `.mod.out` whose data line carries `nsubpixel=8` reads with no problem and
gives every best-fit value. A new `tests/test_subpixels_fit.py` (`fast`, 2):
`examples/lsf_spline` and `metal_line_abs` fitted on a staged copy, then their own
`.mod.out` evaluated with no iteration, compared with the fit's `_fit.dat`.
Measured with a scratch script (`s14/before_delta.py`): on the 14.2 code
`lsf_spline`'s `.mod.out` gave a model that differed from its fit by up to 2.33σ
(22 of 460 pixels by more than 0.1σ), `metal_line_abs`'s by 0.005σ; with 14.3 both
record their sub-pixels (`nsubpixel=6`, `nsubpixel=8`) and agree exactly (0.000σ).
On the 14.2 code the fast test fails (no `nsubpixel` recorded), as do the three
writer tests.

**Documents.** `CHANGELOG.md` (Added); `doc/ALIS_workflow.md`'s data-line keywords
(`nsubpixel=N`).

**Check.** Pre-commit on the 13 files: black reformatted three tests, and ruff's
E731 (a `lambda` assigned in `subpixels_asked`) became a small `def`. Batches
(snapshot `t14_3`): unit 2417 passed, 119 skipped (9.2.0: 2416, 120); `gui` 341
(9.2.0: with Task 14.4's batches); `fast` 157 passed from the repository, the
harness and its `test_bundle` mode included with every reference unchanged (none
carries the keyword, and the harness compares parameters, not data lines), and the
two new fits. The `medium` and `slow` batches run after it, as the baseline of Task
14.6 (below).

**The harness from a snapshot.** From Task 14.4 the batches, the harness included,
run on a snapshot holding a real copy of `context/` (508 MB), not a link to it
(script `batches_real.sh`): the link was what put the harness's context cases
outside their staged copies (memory `batch-snapshots`). The repository is then free
while they run. 14.3's `medium` and `slow` baseline was moved to such a snapshot
(`snap_t14_3b`) after its first case.

### Task 14.4 — A profile with too few sub-pixels warns

Built in the scratch copy, then copied in (8 files, one new test).

**ALIS** (`load.py`, `main.py`).
- `subpixel_warnings` (at load) adds, beside 14.3's warning, one of the snips that
  `nsubpixel_max` caps below what their lines ask for at `nsubsigma_min`:
  "nsubpixel_max (6) leaves 1 snip with too few sub-pixels to give each profile
  nsubsigma_min (5) across its standard deviation - spectrum 0, segment 0: its
  lines ask for 9 - Raise 'run nsubpixel_max' to 9 to give them." With
  `nsubsigma` ≥ `nsubsigma_min` (refused otherwise), a snip falls short only where
  the cap holds it.
- `best_fit_subpixel_warnings(slf, params)`, called by `main.py` once the fit has its
  best fit: one warning of each snip whose fit's `nsubpixel` is below what its lines
  ask for at the best fit, "… spectrum 0, segment 0: the fit used 6; its best fit
  asks for 23 - A fit started from this best fit, with its sub-pixels worked out
  again (nsubpixel= removed from the .mod.out's data lines), uses them." Not with
  `nsubpixel_renew`.
- Both through `msgs.warn`, so a dashboard run keeps them in `run.json`.

**Tests.** A new `tests/test_subpixel_warnings.py` (`unit`, 10, synthetic): a best fit
whose b went from 10 to 2 km/s warns, naming the snip, 9 and 41; none where the best
fit asks for what was used or fewer, none with `nsubpixel_renew`, none where
`nsubsigma_min` is lowered to 1; `nsubpixel_max` 6 warns, naming 9 and what to
raise it to; none at 9, none where `nsubsigma` and `nsubsigma_min` are 3, none by
default; no warning for 200 sub-pixels per pixel. On the 14.3 code 7 of the 10 fail
(the 3 that pass expect no warning). In `test_subpixels_fit.py`: the fit of
`examples/lsf_spline` warns "the fit used 6; its best fit asks for …", and
`metal_line_abs`'s does not.

**Where the warnings fire** (scratch `s14/count_warnings.py`, over the harness's 42
fit cases: each `.mod` loaded for the cap, and each `.mod.out.reference` for what
its best fit asks against the start's `nsubpixel`; blind cases by count alone):
- `nsubpixel_max`'s cap: 7 of 42 cases (helium34's three, which set `nsubpixel_min
  100` and `nsubpixel_max` 100 or 101: 1–2 snips each; VMP_DLA J1358p6522, 1 of 12;
  DH Q1243p307, 4 of 51; DH_orders, 271 of 351; DH J0814p5029, blind, 8 of 26).
- A best fit that asks for more than its fit used: 16 of 37 unblinded cases
  (`lsf_spline`, `lsf_multigauss`, `splineabs` and `splineabs` linear,
  `spline/splineContAbs`; context: HS0105p1619 1 of 24 snips, J1358p6522_original 8
  of 29, J1419p0829 2 of 17, Q1243p307 4 of 51, DH_orders 203 of 351, J0035m0918 1 of
  34, J0903p2628 1 of 14, VMP J1358p6522 1 of 12, helium34's three).
- Neither fires on most cases, so neither is put to RJC (the task's rule); both are
  kept. The count with `nsubpixel_min` 10 is in Task 14.6.

**Documents.** `CHANGELOG.md` (Added); `doc/ALIS_workflow.md`'s `nsubsigma_min` row.

**Check.** Pre-commit passes on the 8 files.

### Task 14.5 — The dashboard and a data line's recorded sub-pixels

Built in the scratch copy, then copied in (5 modules, 9 test files).

**What.**
- **Kept as written.** Nothing needed: every edit of a data line changes the words it
  is about and keeps the rest, as it keeps `bufferpix`. The new tests show
  `nsubpixel=6` kept through the dataset's resolution, a snip's own resolution and
  function, and the snip cut again (`edges_step`); a new snip is not given one;
  Export… writes the project's on the start model and the run's on the best fit;
  the bundle stores and extracts it as written. Continue from best fit copies
  values alone, as before (`results.continue_plan` never writes a data line).
- **The knots dialog** (`lsfknots.recorded_subpixels`, `subpixels`): a snip whose
  line records `nsubpixel=N` is drawn on N, whatever is asked.
- **The pre-flight check** (`preflight.py`): a note for each snip whose recorded
  `nsubpixel` is below what its lines ask for at `nsubsigma_min` (ALIS's own
  `load.subpixels_asked`, on the profile's state): "The snip 0 is worked out on 3
  sub-pixels per pixel, from nsubpixel=3 on its data line; its lines ask for at
  least 8." Run stays enabled. A `Finding` may carry a `fix` (its label);
  `fix_step(project, finding)` makes it: here "Work them out again", which removes
  the keyword, one step. In the Fit tab (`qt/fit.py`) a right-click on such a
  finding offers it, and its tooltip says so, as the `.mod` panel offers a rename
  (14.2).
- **Warnings and options to resolve…** (`results.alis_suggestion`): ALIS's three
  warnings of too few sub-pixels each get what resolves it: "Work them out again, on
  the pre-flight check, removes nsubpixel= from these data lines."; "Continue from
  best fit…, then Run: the next fit works its sub-pixels out from this best fit.";
  "Raise nsubpixel_max (All settings…) to 34." (the number from ALIS's warning).

**Tests.** `test_dashboard_lsfknots.py` (the recorded value drawn, before 21);
`test_dashboard_preflight.py` (+2: the note, its fix one step that ALIS reads, one
revert giving the text back; none for 8 recorded, where 8 are asked);
`test_dashboard_results.py` (+2: the three warnings with their suggestions; Continue
from a run whose `.mod.out` records `nsubpixel=8` leaves the data lines byte for
byte; Task 13.10's test of a warning with no suggestion now uses another of ALIS's
warnings); `test_dashboard_runs_process.py` (+1, `fast`: `examples/lsf_spline` run
through the dashboard's process keeps the best fit's warning, "the fit used 6", in
`run.json`, and its `.mod.out` records `nsubpixel=6`); `test_dashboard_datasets.py`,
`test_dashboard_export.py`, `test_bundle.py` (+1 each: kept); `gui`:
`test_dashboard_qt_fit.py` (+1: the note, Run enabled, the tooltip, the fix one
step that one undo reverses) and `test_dashboard_qt_results.py` (Task 13.10's dialog
test shows the best-fit warning with its suggestion).

**Documents.** `CHANGELOG.md` (Added, Dashboard Stage 14).

**Check.** Pre-commit passes on the 14 files. Batches below.

Batches after 14.3 under pytest 9.2.0: `gui` 341 passed. **14.3's baseline** (the
harness with `nsubpixel_min` 5, snapshot `snap_t14_3b`, a real copy of `context/`):
`medium` and `slow` 6 passed, and DH_orders' fit (`slow`, which needs `--run-slow`)
1 passed in 22 min. With `fast`'s 157, every case reproduces its reference with the
old default.

### Task 14.6 — ALIS's default `nsubpixel_min` of 10, and what it moves (paused: Q14.10)

**The default.** `config.py` 10 (with a comment); `doc/ALIS_workflow.md`'s table
(10/21), `description.tex`, `CHANGELOG.md` (Changed). The tests that assumed 5
changed with it: `test_dashboard_lsfknots.py` (spix, the largest pixel over 10, in
`test_the_delta_function` and `test_a_dataset_s_spix_is_its_snips_largest`);
`test_save_helpers.py` (the writer's test now starts on 10); my own
`test_subpixel_names.py` and `test_subpixel_warnings.py` set `run nsubpixel_min 5`
in their helpers, since they test the names and the warnings, not the default; and
`test_config.py` gains `test_the_fewest_sub_pixels_per_pixel_is_10` (and
`--list-settings` gives 10). `test_lsfspline.py` needed no change (it calls
`set_vars` directly). With these, `unit` passes on the scratch copy (2434 before the
default; the batches of this task follow Q14.10).

**What moves, measured** (snapshot `snap_t14_6m`: the 14.5 tree with the default 10
and a real copy of `context/`; the harness's whole three batches, `test_bundle`
included, `--basetemp` keeping each failing case's staged copy): 34 failed, 70
passed, 1 skipped (DH_orders' fit, which needs `--run-slow`: run alone, below). The
failures: 21 fits (mode (a)), 2 evaluations at the best fit (`lls`, `powerlaw`), the
generated data of `examples/generate`, and the bundle runs of the 10 shipped fits
that moved (each as its plain fit). Stage 12 measured the `fast` batch alone (27 of
149); the `medium` batch adds DH J0814p5029, J1358p0349, J1358p6522_original,
J1419p0829 and Q1243p307 newstart76. Every case that moved (scratch `s14/moved.py`:
χ² against the reference, the largest change of a parameter in units of the
reference 1σ, the largest change of the model column in units of the error, the
running time from the `.mod.out`; the blind cases by χ² and size in σ alone):
- DH/HS0105p1619 [a] chi2 2086.2566 -> 2136.9908 (+2.432%) | largest 3.43σ (voigt line, parameter 1) | model 2.339σ | runtime 17.5s -> 27.0s
- DH/J0814p5029 [a] chi2 4966.6291 -> 4964.0821 (-0.051%) | largest |Δ| 5.79σ | runtime 528.6s -> 688.7s
- DH/J1358p0349 [a] chi2 6876.8181 -> 6835.5647 (-0.600%) | largest |Δ| 16.62σ | runtime 128.5s -> 194.6s
- DH/J1358p6522 [a] chi2 6178.4660 -> 6178.4052 (-0.001%) | largest 0.03σ (vfwhm line, parameter 0) | model 0.005σ | runtime 135.0s -> 178.4s
- DH/J1419p0829 [a] chi2 7648.9893 -> 7653.8135 (+0.063%) | largest 43.91σ (voigt line, parameter 3) | model 0.437σ | runtime 135.8s -> 79.2s
- DH/J1558m0031 [a] chi2 4543.6866 -> 4540.8821 (-0.062%) | largest |Δ| 17.31σ | runtime 48.6s -> 97.5s
- DH/Q0913p072 [a] chi2 5496.8942 -> 5497.1032 (+0.004%) | largest |Δ| 0.16σ | runtime 29.4s -> 51.0s
- DH/Q1243p307 [a] chi2 12116.0624 -> 12116.3073 (+0.002%) | largest 0.07σ (voigt line, parameter 1) | model 0.030σ | runtime 100.5s -> 97.6s
- VMP_DLA/J0035m0918 [a] chi2 1963.3660 -> 1963.3683 (+0.000%) | largest 0.00σ (vshift line, parameter 0) | model 0.002σ | runtime 12.4s -> 11.5s
- VMP_DLA/J0903p2628 [a] chi2 5152.0825 -> 5173.5541 (+0.417%) | largest 699.54σ (voigt line, parameter 2) | model 23.158σ | runtime 21.2s -> 26.2s
- VMP_DLA/J1358p6522 [a] chi2 3790.5707 -> 3790.5388 (-0.001%) | largest 0.02σ (vfwhm line, parameter 0) | model 0.002σ | runtime 15.4s -> 16.6s
- brokenpowerlaw [a] chi2 3503.0708 -> 3499.9471 (-0.089%) | largest 1.12σ (voigt line, parameter 1) | model 0.047σ | runtime 3.8s -> 4.3s
- lls [a] chi2 4354.5574 -> 4353.8760 (-0.016%) | largest 0.02σ (legendre line, parameter 2) | model 0.336σ | runtime 5.2s -> 5.9s
- lls [b] chi2 4354.5574 -> 4353.8774 (-0.016%) | model 0.337σ
- lsf_hst [a] chi2 18590.9798 -> 18590.9799 (+0.000%) | largest 0.12σ (voigt line, parameter 2) | model 0.000σ | runtime 11.7s -> 19.5s
- lsf_spline [a] chi2 450.1170 -> 450.1389 (+0.005%) | largest 0.02σ (lsfspline line, parameter 4) | model 0.005σ | runtime 6.7s -> 12.3s
- powerlaw [a] chi2 2857.6378 -> 2847.5178 (-0.354%) | largest 2.78σ (voigt line, parameter 3) | model 0.159σ | runtime 4.9s -> 5.8s
- powerlaw [b] chi2 2857.6378 -> 2857.8718 (+0.008%) | model 1.543σ
- spline [a] chi2 369.6434 -> 369.4580 (-0.050%) | largest 0.57σ (splineabs line, parameter 9) | model 0.211σ | runtime 4.6s -> 5.4s
- splineabs [a] chi2 354.9322 -> 354.9325 (+0.000%) | largest 0.02σ (splineabs line, parameter 5) | model 0.012σ | runtime 3.1s -> 3.0s
- splineabs [a] chi2 354.9337 -> 354.9332 (-0.000%) | largest 0.03σ (splineabs line, parameter 1) | model 0.027σ | runtime 3.0s -> 2.9s
- tophat [a] chi2 393.6293 -> 394.1327 (+0.128%) | largest 0.01σ (tophat line, parameter 0) | model 0.127σ | runtime 3.9s -> 4.1s
- voigtconv [a] chi2 377.2713 -> 377.2799 (+0.002%) | largest 0.00σ (voigt line, parameter 3) | model 0.011σ | runtime 4.1s -> 4.5s

(No case changed its DOF. The runtimes were measured with four runs sharing the
machine, so they are high; the regenerated references of Task 14.7 give the clean
ones. `examples/generate`'s data: generated with 10 sub-pixels, its golden data
differ.)

**Each case checked.** The old best fit of each unblinded fit evaluated with 10, as
the harness's mode (b) does (scratch `s14/oldbest10.py`), against the new fit: 13 of
the 17 new fits are below it, so their minimum moved and stayed one minimum (the
largest changes: `powerlaw` 2.8σ, its new χ² 10.4 below; `brokenpowerlaw` 1.1σ). Four
are not: HS0105p1619 (new 2136.99, old best at 10 2086.24), J0903p2628 (5173.55
against 5151.99), J1419p0829 (7653.81 against 7649.15) and Q1243p307 newstart76
(12116.31 against 12116.20). Refitted with 10 from their old best fits (scratch
`s14/fromold.py`): 2086.24, 5151.99, 7648.38 and 12116.19, each parameter within
0.04σ, 0.05σ, 0.72σ and 0.04σ of the old best fit. So for the first three there are
two minima, and the fit from the case's own start lands in the worse one; for
Q1243p307 it is one minimum, its new fit stopping 0.1 short. The blind fits'
old best fits cannot be evaluated (their values are hidden); their new χ² are lower
than their references' for three, and 0.004% higher for Q0913p072 (0.16σ). This is
put to RJC as **Q14.10** (the work pauses): what the three's references become.

**14.4's warnings with the default 10** (`s14/count_warnings.py` on `snap_t14_6m`):
the cap at `nsubpixel_max` still on 7 cases; a best fit asking for more than its fit
used on 14 of 37 (16 with 5: DH J1358p6522_original and J1419p0829 no longer).

**Batches of 14.4 and 14.5** run meanwhile on their snapshots; their counts follow.

**Batches of 14.4** (snapshot `snap_t14_4`, a real copy of `context/`): unit 2427
passed, 119 skipped (9.2.0: 2426, 120); `gui` 341 (9.2.0: 341); `fast` 157 passed,
the harness and its `test_bundle` mode included, every reference unchanged.

**DH_orders with the default 10** (`slow`, `--run-slow`, 25 min with the machine
shared): its fit fails its comparison: χ² 92636.48 → 92636.34, the largest change of
a parameter 0.27σ, but 31 comparisons (errors of one component up to 3.1 times the
reference's, and the model column at some pixels; the largest change 0.064σ).
Its old best fit evaluated with 10: 92636.51, above the new fit, so its minimum moved
and stayed one minimum. With it, 35 of the harness's 105 tests fail with 10, 22
fits among them, and 14 of the 18 unblinded fits are below their old best fit at 10
(Q14.10 updated).

**Batches of 14.5** (snapshot `snap_t14_5`, the tree before 14.6's default): unit
2434 passed, 119 skipped (9.2.0: 2433, 120); `gui` 342 (9.2.0: 342), the Fit tab's
new test among them. Its new `fast` test (`lsf_spline` run through the dashboard's
process) passed on the scratch copy; the `fast` batch runs again before the close.

**Paused** at Task 14.6 for RJC's response to Q14.10. The repository holds Tasks
14.1–14.5 and 14.6's default of 10 with the tests that assumed 5 (their batches,
and the `fast` batch's expected failures until Task 14.7, follow the response).

## Prompt 5: RJC's response to Q14.10, and the rest of the stage (2026-11-25)

**Read.** Q14.10's response: start from the old best fit is fine, but first perturb
the best-fit values by the covariance and check that the fits return to the best
fit; if not, use the old best-fit values as the new starting values (option (a)).
And log, as deferred work for a series of its own, how to tell a local minimum from
the best one and how sensitive the minimisation is to the sub-pixels, with
techniques to try. Nothing in it needs a further query: the number of draws, what
"returns" means, and which start is kept were settled as defaults and are written
in the Queries section's note.

### Task 14.6 (continued) — the three cases' starts

**Perturbed starts** (scratch `s14/perturb.py`: each case's `.mod` imported as a
project, its reference and covariance stored as its run, five starts drawn with
`restarts.draw_starts`, seed 1414, each written into the `.mod` with
`restarts.restart_bundle` and fitted with 10 on a staged copy):
- VMP_DLA/J0903p2628: 5151.9937, 5151.9919, 5152.0206, 5151.9945, 5151.9905; within
  0.07, 0.07, 0.14, 0.07, 0.06σ of the old best fit. All return.
- DH/HS0105p1619: 2086.2412, 2086.2385, 2086.2698, 2087.8628, 2086.2390; within
  0.04, 0.04, 0.08, 1.00, 0.04σ. Four return; the fourth start stops 1.6 short.
- DH/J1419p0829: 7650.2899, 7648.4772, 7650.2965, 7647.8650, 7646.0382; up to 3.31,
  0.71, 14.61, 2.56, 3.71σ. Different minima, two below the fit from the old best
  fit (7648.3790): the χ² surface has several minima close together.

**The new starts** (scratch `s14/new_starts.py`, `fit_new_starts.py`): J0903p2628's
`.mod` from the first draw; HS0105p1619's and J1419p0829's from their old best fit's
values (`restart_bundle` with the run's own best fit). Only starting values change:
the model block, and each data line's resolution (a free parameter). Fitted as the
harness fits them, with 10: 2086.2384 (0.04σ from the old best fit), 7648.3790
(0.72σ) and 5151.9937 (0.07σ). RJC's three originals were archived in
`context/stage14_starts_before_14_6.tar.gz`, then replaced.

**Deferred.** `deferred_work.md` §1.5, "Local minima, and how sensitive the
minimisation is to the sub-pixels" (priority 25): what was found (with J1419p0829's
lower minima), why it is deferred, and the techniques suggested: restarts as a
routine check, wider starts (Sobol or Latin-hypercube, basin hopping), sampling the
posterior (emcee, nested sampling), coarse to fine in the sub-pixels, the
sensitivity to the sub-pixels measured as a sweep in `nsubpixel_min` on every
harness case, the local minimiser compared and its stopping rules checked, and a
nightly harness test for the start-sensitive cases. A series of its own,
`minimisation_stage0.md`, independent of the dashboard.

**14.6's batches** (snapshot `snap_t14_6`: the default 10, the three new starts):
unit 2435 passed, 119 skipped (9.2.0: 2434, 120); `gui` 342 (9.2.0: 342; no window
test assumed 5); `fast` 31 failed, 127 passed. 29 of the failures are the harness's
cases of the measurement's list, exactly (the measurement's other five are `medium`
cases); the other two expected `lsf_spline`'s fit to start on 6 sub-pixels
(`test_subpixels_fit.py`, `test_dashboard_runs_process.py`): it starts on 10 now,
and its best fit still asks for more, so both now assert 10, and pass. No reference
changed in this task.

**Check of 14.6.** The default is 10 (`test_config.py`); the tests that assumed 5
are named above; every other test passes unchanged; the harness's failures are the
cases listed, each with its figures and each checked (the three start-sensitive
cases by Q14.10 and its response). Task 14.6 is done.

### Task 14.7 — The references, regenerated once, with their sub-pixels

**Before.** RJC's `context/` goldens of the 12 context cases (630 files: each
`.mod.out.reference`, `.covar.reference` and `reference_fits/` file) archived in
`context/stage14_references_before_14_7.tar.gz`; the shipped examples' are in git.

**Regenerated** (scratch `s14/regen.py`: each of the 23 cases of 14.6's list fitted
as the harness fits it, from the repository's ALIS with `nsubpixel_min` 10, on a
staged copy, one after another with nothing else running; its `.mod.out`,
covariance and `_fit.dat` files, or generated data, copied onto its goldens; no
other case touched). Each `.mod.out.reference` carries `nsubpixel=` on every data
line. χ² (DOF unchanged), runtime:
- DH/HS0105p1619/model/HS0105p1619: 26 goldens | chi2 2086.2384 | runtime 9.4s | nsubpixel= on 24 data lines
- DH/J0814p5029/model/J0814p5029: 28 goldens | chi2 4964.0821 | runtime 724.0s | nsubpixel= on 26 data lines
- DH/J1358p0349/model/J1358p0349: 15 goldens | chi2 6835.5647 | runtime 176.4s | nsubpixel= on 13 data lines
- DH/J1358p6522/model/J1358p6522_original: 31 goldens | chi2 6178.4052 | runtime 150.2s | nsubpixel= on 29 data lines
- DH/J1419p0829/model/J1419p0829: 19 goldens | chi2 7648.3790 | runtime 11.4s | nsubpixel= on 17 data lines
- DH/J1558m0031/model/J1558m0031_FINAL_MODEL: 24 goldens | chi2 4540.8821 | runtime 83.5s | nsubpixel= on 22 data lines
- DH/Q0913p072/model/Q0913p072: 19 goldens | chi2 5497.1032 | runtime 47.9s | nsubpixel= on 17 data lines
- DH/Q1243p307/model/Q1243p307_converge_newstart76: 53 goldens | chi2 12116.3073 | runtime 74.4s | nsubpixel= on 51 data lines
- DH_orders/Q1243p307/model_orders/Q1243p307_orders: 353 goldens | chi2 92636.3370 | runtime 1441.6s | nsubpixel= on 351 data lines
- VMP_DLA/J0035m0918/model/J0035m0918_all_data: 36 goldens | chi2 1963.3683 | runtime 10.7s | nsubpixel= on 34 data lines
- VMP_DLA/J0903p2628/model/J0903p2628: 12 goldens | chi2 5151.9937 | runtime 8.9s | nsubpixel= on 14 data lines
- VMP_DLA/J1358p6522/model/J1358p6522: 14 goldens | chi2 3790.5388 | runtime 16.3s | nsubpixel= on 12 data lines
- examples/brokenpowerlaw/model/fit_spectra: 2 goldens | chi2 3499.9471 | runtime 3.8s | nsubpixel= on 1 data lines
- examples/lls/model/fit_spectra: 2 goldens | chi2 4353.8760 | runtime 5.4s | nsubpixel= on 1 data lines
- examples/lsf_hst/model/fit_spectra: 2 goldens | chi2 18590.9799 | runtime 18.2s | nsubpixel= on 1 data lines
- examples/lsf_spline/model/fit_spectra: 2 goldens | chi2 450.1389 | runtime 6.8s | nsubpixel= on 1 data lines
- examples/powerlaw/model/fit_spectra: 2 goldens | chi2 2847.5178 | runtime 5.2s | nsubpixel= on 1 data lines
- examples/spline/model/fit_spectra_splineContAbs: 2 goldens | chi2 369.4580 | runtime 4.3s | nsubpixel= on 1 data lines
- examples/splineabs/model/fit_spectra: 2 goldens | chi2 354.9325 | runtime 2.6s | nsubpixel= on 1 data lines
- examples/splineabs/model/fit_spectra_linear: 2 goldens | chi2 354.9332 | runtime 2.6s | nsubpixel= on 1 data lines
- examples/tophat/model/fit_spectra: 2 goldens | chi2 394.1327 | runtime 3.8s | nsubpixel= on 1 data lines
- examples/voigtconv/model/fit_spectra: 2 goldens | chi2 377.2799 | runtime 4.3s | nsubpixel= on 1 data lines
- examples/generate/model/generate_spectra: 1 goldens

The three re-started cases reach the better minimum (Q14.10): HS0105p1619 2086.24,
J1419p0829 7648.38, J0903p2628 5151.99. Batches follow the runtimes (`manifest.py`):
J1419p0829 `medium` → `fast` (11.4 s from its new start, 135.8 s before),
J1558m0031 `fast` → `medium` (48.6 → 83.5 s), J0814p5029 `medium` → `slow` (528.6 →
724.0 s); `FIXEDPARAM_BATCH_OVERRIDE` and `MINIMISATION_MACHINE_DEPENDENT` keep
their cases. `examples/lsf_spline` leaves `FIXEDPARAM_EXCLUDE` (`test_regression.py`);
`brokenpowerlaw` and `tophat` stay, with their reasons.

**Tests.** In `test_writer_round_trip.py` (`unit`, +22):
`test_a_regenerated_reference_records_its_sub_pixels`, each regenerated reference's
data lines carrying one `nsubpixel=` of at least 10 (the context ones skipped where
absent); every reference reads back (103 passed). The blind references hide what
their old ones hid (BLIND lines 6, 4, 1, 1, as before; compared by count, no value
read). `git diff --stat -- examples tests/test_regression.py`: the 10 shipped fits'
references and goldens, `examples/generate`'s data, and `test_regression.py` (22
files).

**Check.** The `fast`, `medium` and `slow` batches against the new references, from
the repository (`-m "fast or medium or slow" --run-slow`, pytest 9.0.2): **167
passed**, 2945 deselected, in 3799.8 s (1:03:19). `examples/lsf_spline` is among them
in every mode: `test_minimisation`, `test_fixed_param` (mode (b), back from
`FIXEDPARAM_EXCLUDE`), `test_bundle`, and `test_subpixels_fit`'s
`test_a_mod_out_gives_its_fit_s_model`. `pre-commit run --files
tests/test_writer_round_trip.py tests/test_regression.py`: ruff, isort, black passed.

### Task 14.8 — The review

**Built** by `doc/dashboard/stage14/` (Stage 12's and 13's helpers, the mockups'
stylesheet):
- `draw_fits.py OLD SHOTS`: the table of every regenerated case (`moved.json`: χ² of
  the old reference against the new, the largest change of a parameter in units of
  its old 1σ and which, the largest change of the model column, the runtime and
  batch, and the start of the three re-started cases; a blind case by χ², the size of
  its change and its runtime alone). The fit of each unblinded case that moved by
  more than 1σ or 1% in χ², drawn: `powerlaw` (2.78σ) and `brokenpowerlaw` (1.12σ),
  with the data, both models, the residuals of each, and the change of the model
  (at most 0.16σ and 0.05σ). `examples/lsf_spline`'s reference evaluated at its best
  fit with and without its `nsubpixel=`: without it, 21 sub-pixels, up to 0.61σ (8
  of 460 pixels above 0.1σ); with `nsubpixel=10`, at most 0.0004σ. OLD is
  `context/stage14_references_before_14_7.tar.gz` and `git archive f9f4539
  examples`. A `_fit.dat`'s model is its last column (five columns for the context
  cases): a first draft read the fourth, caught by its zeros.
- `take_screenshots.py OUT`: the old name's problem and its rename, All settings…
  filtered on "nsub" (the five settings), the pre-flight note (and the Run box close
  up) and its fix, Warnings and options to resolve… with the three sub-pixel
  warnings, and the Delta Function (spix 0.141 km/s, FWHM 3.00 km/s). A right-click's
  menu is built from the same action text and drawn on its own.
- `build_review.py SHOTS OUT.html`: six sections and the 14.1 box (its nine tests
  named).

**Published** (draft 1, private): https://claude.ai/artifact/3qx9212EdtFb5KUr6jv9wc,
from the session scratchpad's `stage14_review.html`. Its one question, the dialog's
one-line warnings, is Q14.11 in the Queries section. The blind cases show no value;
the page is built from the harness's references and the `gui` tests' scenes.
Pre-commit passes on the three scripts.

**Paused** for RJC's comments on the page and Q14.11 (Task 14.8's check: the comments
applied and logged), before Task 14.9.

## Prompt 6: RJC's comments on the review page, and Q14.11 (2026-11-25)

**Read.** Two comment threads on draft 1, and Q14.11's response. (i) On the Warnings
section's question: one line per warning is fine, but sentences need their full
stops (the response says the same). (ii) On the names table's `run nsubsigma_min`:
is it needed, since `nsubsigma` is set and never changes? The second asks for a
decision that is RJC's: it is **Q14.12** (options (a) keep, (b) remove; lean (b)),
answered in its thread and left open.

**Full stops (Q14.11, comment (i)).** Test first: `test_each_part_of_a_warning_is_a_sentence`
(`test_subpixel_warnings.py`, three cases: recorded, capped, best fit), each part of
the warning ending with a full stop and the snip's line beginning "Spectrum 0,
segment 0: "; with the tests that quote the warnings (`test_subpixel_warnings.py`,
`test_subpixel_names.py`) and the dashboard's copies of them
(`test_dashboard_results.py`'s `SUBPIXEL_WARNINGS`, with a check of the one-line text,
and `test_dashboard_qt_results.py`'s `BEST_FIT_WARNING`), 6 failed before the fix.
Fix (`alis/load.py`): `_snip_line` writes a snip's line as a sentence; the three
headers end with "." in place of " -". The dashboard is unchanged (it matches the
warnings by their first words and the cap's number).

**Found: J1358p6522's tests, after Task 14.7.** Running the dashboard's results tests
found three that read RJC's VMP_DLA/J1358p6522 reference (the `j1358` fixture,
skipped where `context/` is absent, as on CI) and failed against its regenerated
reference: χ² 3790.570655 → 3790.538798 (initial 4002.709893 → 4002.792235), χ²ν
0.914272 → 0.914264, a forest b's error 10.78 → 10.71, run time 15 s → 16 s. Their
expected values are updated (`test_dashboard_results.py`, three tests;
`test_dashboard_qt_results.py`, one). The `unit` and `gui` batches had not run since
14.7 (the harness had); they run now on a snapshot with a real `context/`
(`snap_t14_8`), to find any other.

**The page, draft 2** (same URL): the warnings' full stops noted, the Warnings
screenshot retaken (`take_screenshots.py OUT warnings`), the answered question kept
in place for its thread ("Answered, Q14.11."), Q14.12 named in the lede. Thread (i)
replied to and resolved; thread (ii) replied to, open for Q14.12.

**Batches** (snapshot `snap_t14_8`: the full stops, and J1358p6522's first four
values): unit 2459 passed, 119 skipped, 1 failed (9.2.0: 2458, 120, 1); `gui` 341
passed, 1 failed (9.2.0: the same). The two failures were J1358p6522's again: the
run's time (0.004288 → 0.004521 hrs) in `test_correlations_and_statistics`, and a
forest b's error in `test_j1358s_results_as_in_the_mockup` (± 60.55 → ± 60.27);
both updated. The four files run again (`test_dashboard_results.py`,
`test_dashboard_qt_results.py`, `test_subpixel_warnings.py`, `test_subpixel_names.py`):
106 passed. No other test reads a regenerated reference. Pre-commit passes on the
changed files.

**Paused** for RJC's response to Q14.12 (and any further comment on draft 2), before
Task 14.9.

## Prompt 7: RJC's response to Q14.12 (2026-11-25)

**Read.** Option (b): `nsubsigma_min` is removed, and ALIS warns wherever a profile
has fewer sub-pixels across its σ than its snip's `nsubsigma`. No further query.

**Tests first** (11 failed before the change): `test_subpixel_names.py` (the settings
without `nsubsigma_min`; `run nsubsigma_min` refused as removed, in a model and
through `--set`; `--list-settings` without it; `test_a_lower_nsubsigma_asks_for_fewer`,
a run or data-line `nsubsigma` of 3 giving 5 sub-pixels with no warning, where it was
refused; `test_a_data_line_s_own_nsubsigma_is_its_warning_s_level`, `nsubsigma=10`
with `nsubpixel=9` warning "nsubsigma (10)" and "ask for 17."; a recorded
`nsubpixel=3` with `nsubsigma=1` not warning), `test_subpixel_warnings.py` (the
level named "nsubsigma (5)"; the no-warning cases that lowered `nsubsigma_min` now
`nsubpixel_min 41` at b = 2 km/s, and the cap at 6 with `nsubsigma 3`),
`test_dashboard_settings.py` (its names), and the dashboard's copies of the warnings.

**Change.** `alis/config.py`: the setting removed; `REMOVED_SETTINGS` gives each
removed name's reason (`warn_subpix`'s message unchanged). `alis/load.py`: the
refusal removed; `subpixels_asked` without `level`; `_nsubsigma_text` names the
level in each warning's header. `alis/dashboard/preflight.py`: the note's level.
`CHANGELOG.md`, `doc/ALIS_workflow.md` (`nsubsigma`'s row says where ALIS warns).
No model, reference or document outside the stage's own names it.

**Check.** The changed and dashboard files' tests (`test_subpixel_names`,
`test_subpixel_warnings`, `test_dashboard_settings`, `test_dashboard_results`,
`test_dashboard_preflight`, `test_config`, `test_subpixels_fit`,
`test_dashboard_qt_results`, `test_dashboard_qt_fit`, `test_dashboard_qt_modpanel`):
203 passed. Pre-commit passes on every changed file.

**The page, draft 3** (same URL): the names table marks `run nsubsigma_min`
removed (the row kept, as RJC's thread is anchored to it), the notes follow, All
settings… (four settings) and Warnings retaken. RJC's thread replied to and
resolved. Both of RJC's comments are applied: Task 14.8's check.

**Batches** for the close run on `snap_t14_9` (unit and `gui` under both pytests,
then `fast`, `medium` and `slow`).

### Task 14.9 — Close the stage

**The documents.** `CHANGELOG.md` already held 14.1–14.4 and 14.6 (checked;
`nsubsigma_min` taken out at Q14.12). `doc/ALIS_workflow.md`: §4.3's "Since Stage 14"
(a `.mod.out`'s sub-pixels, too few, the names, the knots); §2.1's table and the
convention were written at 14.2. `doc/tex_files/description.tex`: `nsubsigma_min`'s row
removed (a grep for `nsubsigma_min` had missed TeX's `nsubsigma\_min`), `nsubsigma`'s
saying where ALIS warns. `tests/README.md`: the Stage 14 row, the references recording
their sub-pixels and regenerated with 10, and the three batch moves. The
`gui-component` skill's "Stage 14 additions" (a fix on a right-click, removed names,
a data-line keyword kept, warnings as sentences, menus off-screen). Decisions
D147–D152 in `ALIS_v2_dashboard_prompts.md`, the convention among them (D148),
refining F9, Q12.9, Q12.11, Q12.13, Q13.7, Q13.8, D137, D140, D142, D145, S17 and S20.
`deferred_work.md`: §7.10 done (priority 24), §2.1, §2.4, §5 and §7.7 offered to Stage
15, and §7.12 (priority 26) for the three faults Task 14.1 found.
`dashboard_stage0.md`: Stage 14 done; the documentation next, then Stage 15.
`documentation_stage0.md`: the run settings as they are, the convention, the data
line's `nsubsigma=` and `nsubpixel=`, and the warnings.

**Found and not done** → `dashboard_stage15.md`'s "Carried over from Stage 14", each
with a query: `apod` cannot be evaluated (Q15.4), `lsf`'s buffer (Q15.5), a model
loaded again (Q15.6), and local minima and the sub-pixels (§1.5; Q15.7, the
series' order against Stage 15). The draft's header, its D range (D1–D152) and its
candidates' state brought up to date.

**Found while closing: Q14.13.** `examples/brokenpowerlaw` and `examples/tophat`, left
out of mode (b) since Stage 0 (`FIXEDPARAM_EXCLUDE`), pass it with their regenerated
references, evaluated with the harness's own functions: `brokenpowerlaw` χ²
3499.947056 against 3499.947053 recorded, `tophat` 394.132678 against 394.132678,
both model columns within mode (b)'s tolerance. Their references before 14.7, which
record no `nsubpixel=`, fail it with 5 or with 10 (`brokenpowerlaw`: 3555.29 against
3503.07 with 5). So `brokenpowerlaw`'s mismatch (§2.4, priority 3, Stage 15's proposed
key item) is the sub-pixels of §7.10, not a rejected trial step. Put to RJC as Q14.13:
both back in the gate, and what becomes of §2.4 (lean: back in the gate, §2.4 kept for
Stage 15 to confirm on the minimiser's stopping paths).

**Checks.**
- 14.2's `grep` for the old names: in the repository, only where meant
  (`config.py`'s tables, `settings.py`'s docstring, the tests of the refusal, the
  CHANGELOG, `doc/ALIS_workflow.md`'s sentence naming them, the mockups and Stage 12's
  builder, and Stage 14's screenshot script, which types one); in `context/`, the 7
  notes and 9 backups 14.2 left as records.
- `git diff --stat f9f4539 -- alis ':!alis/dashboard'`: `config.py`, `load.py`,
  `main.py`, `model_eval.py`, `minimise.py`, `gpu_dispatch.py`, `save.py` and 26
  modules of `alis/functions/` (33 files), as the task lists.
- `test-coverage` on what changed (the stage's unit and dashboard tests, 391 passed,
  measured against the lines changed since f9f4539): `config.py`, `save.py` and
  `results.py` fully covered. Untested: in `load.py`, a data line's `nsubsigma=0` and
  `_nsubsigma_text`'s mixed case, and the grid's lines that only fits reach (run by
  the `fast` batch in a subprocess, which coverage does not see; likewise `main.py`'s
  three); trivial guards in `preflight.py`, `settings.py`, `lsfknots.py`; the loaders'
  boolean and integer branches (`apod`, `lsf`, `lsffile`), and `apod`'s bin-type
  errors (§7.7). Added: `test_a_data_line_s_nsubsigma_of_0_is_refused`,
  `test_a_warning_names_the_level_of_its_snips` (two cases), and
  `test_a_boolean_and_an_integer_keyword_stay_each_line_s_own` (three loaders: a
  line's `blind:True,blindseed:7` its own, the next line's the defaults, which before
  Task 14.1 would have taken the first's). 57 passed.
- The window's text: no design reference in any string the stage added to
  `alis/dashboard/` (the `gui` tests' `design_references` checks run in the batch).
- Pre-commit on every Python file the stage changed or added (75): passed.
- CI: waits for RJC's push of the branch (the `gui` job's segfault of Stage 13 and the
  `examples` job, with `nsubpixel_min` 10, are checked then).

## Prompt 8: RJC's response to Q14.13, and CI (2026-11-25)

**Read.** Q14.13 (b): both cases back in the gate, §2.4 kept open for Stage 15; either
can be excluded again if it gives trouble. RJC committed and pushed the stage (506939b,
"stage 14 - completed"); CI is checked before the close.

**Q14.13 applied.** `tests/test_regression.py`: `FIXEDPARAM_EXCLUDE` is empty, its
comment giving each case's history; `test_fixed_param` on `brokenpowerlaw` and
`tophat`: 2 passed (`fast`, two tests more). `deferred_work.md` §2.4 (kept open, with
the evidence and what is left) and priority 3; `dashboard_stage15.md`'s candidate 1
(now a confirmation of the minimiser's stopping paths) and Q15.1's lean (the
simulations, with §2.4 a small first task); `tests/README.md`; D149. Pre-commit
passes. These changes come after 506939b, for RJC's next commit.

**CI** (run 45703269345, on 506939b): lint passed; unit, gui and examples on Ubuntu
and macOS running.

**CI's result** (run 45703269345 on 506939b): lint passed; unit passed on Ubuntu and
macOS; `gui` passed on both (Ubuntu 18 min 12 s; Stage 13's segfault did not recur);
`examples` failed on both, as it has since Stage 13's push (f9f4539 failed it on both
platforms; 7343048, Stage 12, passed). The job's log needs admin rights to download
(the API answers 403), so the failure was reproduced: a clean copy of 506939b (`git
archive`, no `context/`) under CI's command. With numpy 2.4.3 (this machine): 115
passed (2 errors of this machine's setup, pytest-qt not loaded). With numpy 2.5.5, the
newest, which CI installs (a scratch venv): 2 failed, 115 passed:
`examples/lsf_spline`'s `test_minimisation` and `test_bundle`, each "convolution
errors line 1, token 1: 1-sigma error 6127.9272242 vs 0.0". The smoothing FWHM of its
`lsfspline` is undetermined (the fit trades it against the knots, and anything
narrower than a sub-pixel leaves the model as it is): at its bound with error 0 under
2.4.3, at 0.082 km/s with error 6128 under 2.5.5; everything else agrees. The
harness's Q0.16 asks a zero reference error to stay zero. A refined rule (a zero
reference error also matched by an error at least the parameter's value) was
prototyped on the clean copy and passes under both. Put to RJC as **Q14.14**; the
close waits for it.

**The close's batches** (snapshot `snap_t14_9`: the tree after Q14.12, before the
three coverage tests and Q14.13; a real copy of `context/`): unit 2460 passed, 119
skipped (9.2.0: 2459, 120), against Stage 13's close of 2356 and 119 (2355, 120);
`gui` 342 passed under both (Stage 13: 340); `fast` 159 passed (Stage 13: 155), the
regression harness and its `test_bundle` mode included; `medium` and `slow` 8
passed (`--run-slow`). Since the snapshot: the three coverage tests (57 passed with
their files) and Q14.13's two `test_fixed_param` cases (2 passed) were run on
their own. The close waits for Q14.14 (CI's `examples` job).

## Prompt 9: RJC's response to Q14.14 (2026-11-26)

**Q14.14 applied** (RJC's response: (a) for now; the test data should be made at a
higher S/N). `tests/compare.py`'s `_compare_error_line` matches a zero 1σ error with a
zero error or with one at least as large as the parameter's value, read from the
value line in the same file (`_undetermined`; `_compare_section` hands each error line
its two value lines through `_function_values`, a blind value as None). The rule holds
either way round, so a reference made where rounding stops short of the bound passes
where it reaches it; a small error, or a zero error whose value is blind, still
fails. `tests/test_compare.py` (`unit`, 9 tests) checks it on lines shaped like
`lsf_spline`'s convolution section, and a blind line's error slot; 4 of the 9 fail on
the comparator as committed (506939b). Recorded as D153; `tests/README.md` updated.
The data at a higher S/N are `deferred_work.md` §7.13 (priority 27), offered to Stage
15 as item 16 with Q15.8 (D range D1–D153). pre-commit passed on the changed files.

CI's `examples` batch (`pytest -m examples`, from the repo): numpy 2.5.5, 120 passed,
42 skipped; numpy 2.4.3, 120 passed, 42 skipped (Q14.13's three cases in mode (b)
included; before, 2 failed and 115 passed under 2.5.5). `lsf_spline`'s, `brokenpowerlaw`'s and
`tophat`'s `test_minimisation`, `test_fixed_param` and `test_bundle` listed as passed
under 2.5.5. `unit` on a snapshot of this tree (`snap_q14_14`): 2475 passed, 119
skipped (9.2.0: 2474, 120). The close waits for CI on RJC's push.

## Prompt 10: the stage document's Prompt 5, CI's Ubuntu `examples` job (2026-11-26)

**CI on 0744c8a** (runs 45795870010 on `alis_v2_db_stage14` and 45795926188 on
`alis_v2`): lint, unit and gui passed on Ubuntu and macOS, and `examples` on macOS
(Q14.14's rule: `lsf_spline` passes); `examples` failed on Ubuntu. RJC's copy of its
log (`context/misc/GitHub_examples_Ubuntu-latest_py313.txt`): 2 failed, 110 passed,
`examples/splineabs/model/fit_spectra_linear`'s `test_minimisation` and `test_bundle`,
each with four 1σ errors of O I's `splineabs` outside 10% (column 0.892 vs 1.345,
bturb 2.876 vs 6.218, knot 3 1.615 vs 1.946, knot 6 0.178 vs 0.316) and the model
column at 9 of 368 pixels (worst 0.074 of a pixel's error). χ², every value and its
evaluation at the best fit (mode (b)) passed; `lsf_spline`'s three tests passed on
Linux too. Its reference was regenerated at Task 14.7, so the Ubuntu failure at
506939b was most likely this case as well as `lsf_spline` (Q14.14 assumed the two
`lsf_spline` tests alone).

**Cause:** the degeneracy Q7.21 diagnosed for its sibling `splineabs/fit_spectra`.
The regenerated reference (`nsubpixel_min` 10) left O I's knot 1 at its bound,
0.0 ± 0.107 (the old reference had it at 9.65×10⁻⁵ ± 0.76, and its errors moved
113.2 → 6.2 at the regeneration). Here the refit reproduces the reference exactly.
Refitting from starts perturbed by 10⁻⁶ (values strictly inside their limits): χ²
from 354.9318 to 354.9357 across 24 starts, with bturb's error 0, 2.9, 6.3, 24 or
112 and the column's 1.2–1.9, as different knots end at the bound. Fixing knot 1 at
0 gives bturb's error 2.92, as on Linux, though not Linux's other errors exactly;
Linux lands on another of these points. (Starts just inside a bound, a knot at
0.999999 for 1, stop the minimiser after 2 iterations at the start's χ², the
stopping path of `deferred_work.md` §2.4 and §1.5.) No Linux machine was at hand to
run it there.

**Fix** (Q7.21 (a), RJC's remedy for the same degeneracy): `fit_spectra_linear` joins
`MINIMISATION_MACHINE_DEPENDENT` in `tests/test_regression.py`, with its diagnosis,
so the Ubuntu `examples` job (`--skip-machine-dependent`) skips its refit and bundle;
its evaluation at the best fit, and every test on macOS, still run. Checked here:
`-k splineabs` with the flag, 2 passed (mode (b)), 4 skipped; without it (numpy
2.5.5), 6 passed. Ubuntu's command under numpy 2.5.5: 116 passed, 46 skipped
(strict: 120 passed, 42 skipped). Making the example well posed (Q7.21 (b)) is
`deferred_work.md` §7.14 (priority 28), offered to Stage 15 with `lsf_spline`'s data
as item 16, Q15.8 now asking for both. `tests/README.md` updated; pre-commit passed.
The close waits for CI on RJC's push.

## Prompt 11: the close (2026-11-26)

**CI** (run 45805049616, on 5f776f7, `alis_v2_db_stage14`): every job passed — lint;
unit on Ubuntu (2 min 15 s) and macOS (3 min 33 s); `examples` on Ubuntu (4 min 43 s,
the two `splineabs` refits and bundles skipped) and macOS (10 min 57 s, every case);
`gui` on Ubuntu (36 min) and macOS (30 min), within their 90 minutes, and Stage 13's
segfault did not recur. Both CI items of Task 14.9 are met: the `gui` job, and the
`examples` job (Q14.14's rule, D153, and `splineabs/fit_spectra_linear` among the
machine-dependent refits).

**Task 14.9's checks**, all met: every task logged as done with its check; Q14.1–Q14.14
answered and applied; the review applied (both threads resolved); the batches' counts
against Stage 13's close (unit 2475 passed, 119 skipped on this tree, 2474 and 120
under pytest 9.2.0; `gui` 342; `fast` 159; `medium` and `slow` 8; `examples` 120
passed, 42 skipped, under numpy 2.4.3 and 2.5.5); CI on every job; `test-coverage`
and the tests it suggested; no design reference in the window's text; pre-commit on
every changed file; no old name of the sub-pixels left; ALIS outside
`alis/dashboard/` changed only as the tasks said.

**What follows.** `dashboard_stage0.md`: Stage 14 done (D147–D153); next, the
documentation, `documentation_stage0.md`, brought up to date for this close with its
queries (DQ.1–DQ.7, and DQ.8 onwards, new); then Stage 15 (`dashboard_stage15.md`,
Q15.1–Q15.8). `dashboard_stage15.md` is untracked still, for RJC to add.

**Stage 14 is closed.**
