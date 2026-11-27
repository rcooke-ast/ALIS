# ALIS regression test harness (Stage 0)

This directory holds the automated regression harness that gates every
refactoring stage: it runs the current ALIS on the committed example fits and
compares the output against the golden "reference" files, within agreed
tolerances. It never modifies any file in the repository — every run is staged
in a temporary copy — and it never overwrites a reference.

## Layout

- `manifest.py` — discovers the test cases (any `.mod` with a sibling
  `.mod.out.reference`, under `examples/` and `context/fitting_examples/`) and
  records, per case, the data files and their `reference_fits/` goldens, the
  covariance reference, blind/random status, runtime, and batch. Run it
  directly for a summary table: `python tests/manifest.py`.
- `alisrun.py` — stages a disposable copy of an example and runs `run_alis`
  (headless, `-p 0`); builds the fixed-parameter input.
- `compare.py` — parses `.mod.out` and applies the tolerance comparisons; its
  rule for a zero 1σ error is unit-tested in `test_compare.py` (Q14.14).
- `optional_deps.py` — marks a model that uses the `lsf` function (which needs the
  optional linetools) with the `linetools` marker, which `conftest.py` skips where
  linetools is not installed (as on CI) and runs where it is. Use `marked(paths)`
  for a test that takes every model of a folder.
- `test_regression.py` — the pytest suite (see the two test modes below).
- `conftest.py` / `../pytest.ini` — import path and batch-marker config.

## Test modes

Each non-blind fit case has two tests; blind cases run mode (a) only.

- **`test_minimisation` (mode a)** — a full `run_alis` re-fit, comparing the
  `.mod.out` best-fit parameters (within 10% of each 1σ error), 1σ errors
  (10%; a zero error stays zero, or becomes one at least as large as its
  value, either way round, as a parameter the data cannot determine does —
  Q0.16, Q14.14), χ² (1%), the `_fit.dat` model column (`|new − ref| < 0.01 × error`
  per pixel, the error-based check of Q0.22/Q0.23), and the covariance matrix
  where a golden copy exists (1% relative with a `sqrt(C_ii·C_jj)` floor).
- **`test_fixed_param` (mode b)** — re-runs the `.mod.out.reference` with
  `chisq miniter 0` / `maxiter 0` (a zero-iteration evaluation at the best-fit
  point) and compares χ² (0.1%), DOF (exact) and the model column
  (`|new − ref| < 0.15 × error` — looser than mode (a) because the reference's
  parameters are only printed to 8 digits, which moves saturated cores).
  Skipped for blind cases.
- **`test_bundle` (mode a, as a project bundle)** — dashboard Stage 1.7. Each
  shipped example of the `fast` batch is packed into a `.model` bundle, run with
  `run_alis case.model` from an empty directory (the run must leave nothing on
  disk but the bundle), extracted, and compared with the references exactly as
  mode (a) compares a plain run.
- **`test_bundle_matches_plain`** — two real context fits (`VMP_DLA/J0903p2628`,
  and a helium34 fit that writes `out wavecorr` files) are run both as plain
  files and as a bundle, and the two are compared with mode (a)'s tolerances.
  They are compared with each other, not with the references, so the test checks
  the bundle alone.

Covariance goldens exist for 17 cases: all 16 under `context/fitting_examples/`
plus `examples/metal_line_abs/fit_spectra`, which carries one so the CI
`examples` batch exercises the covariance writer. Only mode (a) compares them —
mode (b) strips `out covar` when building its fixed-parameter input.

The `generate` example is a special case: it runs `generate_spectra.mod` and
compares the produced data file to its golden copy.

Since dashboard Stage 14, every fit's `.mod.out.reference` records the sub-pixels
per pixel its fit used, `nsubpixel=N` on each data line, so that mode (b) evaluates
the best fit on them (ALIS works the sub-pixels out from the starting values, and a
best fit can ask for others). `examples/lsf_spline`, left out of mode (b) before for
that reason, is evaluated so, and so are `examples/brokenpowerlaw` and
`examples/tophat`, left out since Stage 0 (Q0.12, Q0.15), whose mismatches were the
same (Q14.13): no case is excluded from mode (b) now (`FIXEDPARAM_EXCLUDE` is empty). The default `nsubpixel_min` became 10 (from 5), and the
references that moved were refitted, checked against their old best fit evaluated
with 10, and regenerated as the harness fits them (Task 14.7: 22 fits and
`examples/generate`'s data; the old references are archived in
`context/stage14_references_before_14_7.tar.gz` and in git). Three of RJC's fits
landed in a worse minimum from their own start, so their `.mod` files start from
their old best fit (DH/HS0105p1619, DH/J1419p0829) or from a start drawn about it
(VMP_DLA/J0903p2628); how sensitive a fit is to its start and to the sub-pixels is
`claude_prompts/deferred_work.md` §1.5.

## Unit tests

Separate from the regression harness above, the `unit`-marked tests exercise the
*stable surface* of individual functions in isolation — no fits, no subprocess,
no golden files — so they run in seconds and localise failures to a single
function. They cover the pure logic added/refactored through Stages 1–3
(`utils` conversions, `config` dataclasses, the Stage 3.4 cache/Jacobian helpers
in `minimise`, `convergence`, the `report` residual math, and the stable
non-I/O parts of `load` and `logger`), plus the Stage 4–5 additions below.

```bash
pytest -m unit                                   # whole unit batch (seconds)
pytest -m unit --cov=alis --cov-report=term-missing   # with a coverage report
```

The `unit` batch runs on every push via the `unit` CI job (Ubuntu + macOS,
py3.13), which also prints the coverage report (no threshold gate — modules are
still evolving). See `claude_prompts/refactor_code_unit_tests.md`, which
deferred the file-format loaders and GPU code to the stages that reshape them.
Both deferrals are now discharged — GPU in Stage 4, I/O in Stage 5:

| file | what it covers |
|---|---|
| `test_load_files.py` | `load_ascii` / `load_fits` / `load_userdata` / `load_data`, on synthetic fixtures written into `tmp_path` |
| `test_load_model.py` | the `.mod` model-block parser: named vs positional parameters, tie labels, `fix` and `lim` |
| `test_save_helpers.py` | the writer: `print_model`, the data line it rebuilds, `save_covar`, and a save-then-reload round trip |
| `test_writer_round_trip.py` | the other half of the round trip — re-reading all 40 committed `.mod.out.reference` files |
| `test_atomic_mass.py`, `test_plotscript.py` | Stage 5.2 and 5.3 |
| `test_load_memory.py` | dashboard Stage 1.2: real models' data loaded from memory are identical, bit for bit, to the data loaded from disk |
| `test_shared_pixels.py` | dashboard Stage 1.3: `find_shared_pixels` on synthetic snips, and on the real overlaps of J1358p6522 and Q1243p307 |
| `test_outputs.py` | dashboard Stage 1.4: the output writer, in memory and on disk (one test runs a fit, and is marked `fast` rather than `unit`) |
| `test_bundle.py` | dashboard Stage 1.5–1.8: packing every model and extracting it byte for byte, the manifest checks, atomic writes and the lock, source spectra, hidden lines (the bundle-run tests are marked `fast`) |
| `test_dashboard_*.py` | dashboard Stage 2, the project model in `alis/dashboard/`: no Qt imported; every model read back byte for byte and split as `load_input` splits it; the parsed model agreeing with ALIS's own loaders on 92 models; targeted edits; the project's systems, components and datasets; the validator; the blinding gate (no hidden value in any string shown); undo/redo and removal; QSO Abs Line mode (the two tests that run a fit are marked `fast`). The property tests use `hypothesis` |
| `test_dashboard_{preferences,session,launch,livetext,actions,markers,sources,guard,align}.py` | dashboard Stage 3, without Qt: preferences over shipped defaults; sessions (save byte for byte, autosave and recovery, runs written meanwhile kept, conflicts on disk, Save as); the `alis` launcher and new projects; typing in the `.mod` panel kept in step with the masks; the registry of actions; the tab markers; relinking spectra; the blinding guard of the history; aligning a model's columns (every model, word for word, and as ALIS reads it). `test_dashboard_session_run.py` runs `run_alis project.model` while a project is open, and is marked `fast` |
| `test_dashboard_qt_*.py` | dashboard Stage 3, the windows, marked `gui` and run off-screen with pytest-qt (`pip install -e ".[gui,gui-test,dev]"`, then `pytest -m gui`); skipped where the gui extra or pytest-qt is missing. `test_dashboard_qt_foundation.py` (marked `unit`) needs no Qt: only `alis/dashboard/qt/` imports Qt, through `qtpy`; `alis --help`; the install message without Qt. `test_dashboard_qt_review.py` covers the changes of RJC's review (the mode in the status bar, Fit · Results, Align columns on the `.mod` panel, the panel's own window and width, New project's table of spectra and the dialog for each spectrum, its column roles with Ignore, and the bad-pixel mask). `dashboard_qt_helpers.py` makes a window whose questions and file choices the test answers |
| `test_dashboard_{lines,snips,continuum,datasets}.py` | dashboard Stage 4, without Qt: transitions and line IDs (J1358p6522's H I list, Q1243p307's coverage grid, gap and edge flags, Identify a feature); snips (SNIP and CLEAR, regions and masks, edges in and out, copying regions, bad pixels never fitted, keep and merge for pixels fitted twice), each step read by ALIS and undone byte for byte; the continuum (ALIS's own curve to 1e-12, the first guess and BIC, knots, sharing); the datasets and systems edits. The `badpix` column is in `test_load_files.py` |
| `test_dashboard_qt_{plots,data,regions}.py` | dashboard Stage 4, the spectrum view, the Data tab and the Regions tab, marked `gui`: zooming and its history, J1358p6522's 40,402 pixels drawn fast; Identify a feature, ties, Confirm, no hidden value; the transition list and its keys (only with the tab's focus), adding regions to all datasets and tweaking one, SNIP then CLEAR, one set of regions across Q1243p307's three datasets, edges re-cut from the source, the order − + and knots each one step, a hidden continuum never shown, J1358p6522's 36 shared pixels kept, Q1243p307's O I and Si II merged in three datasets, cross-highlighting |
| `test_dashboard_{profile,components}.py` | dashboard Stage 5, without Qt: each snip's model equal to ALIS's `model_func` (1e-10) on every example, J1358p6522 and Q1243p307, evaluated for some snips only, with links, blinded lines and hidden values; a new component's lines and labels, the first guess of log N (within 0.1 dex of `metal_line_abs`'s best fit; the lowest detection, saturation, upper limits, no data), drags and their bounds, the temperature modes, Exclude and Include byte for byte (dependent lines, the last ion), interlopers and Add interloper's choices, one panel per snip, links shown and not dragged, notices and their fixes, what a `.mod` line describes, the "!" of an ion in no snip |
| `test_dashboard_qt_components.py` | dashboard Stage 5, the Components tab, marked `gui`: J1358p6522's and Q1243p307's navigators, panels (two columns, never more than panels; the model drawn equal to `profile.py`'s; datasets; the Lyman-limit snip one panel), one click adds one component, a click selects the nearest, drags above (z) and below (log N and b) the continuum, N and B holding one, a right drag panning (one step, bounds, blinded: only Δχ²), every card badge, menu and button one step, the Excluded line, no hidden value on a card; an interloper from a panel's menu, a linked parameter, J1358p6522's notices fixed in one click; cross-highlighting both ways; the "!" and its tooltip |
| `test_dashboard_qt_end_to_end.py` | dashboard Stages 4.12, 5.10 and 6.10, marked `fast`: a new project from `OI_SiII.dat` taken through the Data, Regions and Components tabs (one click on O I 1302, then 8000 K with Fixed at…) to a fit whose column densities agree with the example's reference; then the same with its last steps on the Fit tab: Run, Commit…, T free and Run again, Compare, Continue from best fit, Use best fit on the card |
| `test_dashboard_{runs,settings,preflight,results,fitview,compare,restarts}.py` | dashboard Stage 6, without Qt: runs (the bytes handed over, a stored run's snips kept once, the latest and committed runs, commit, describe, remove, restore one step, values and errors read from a `.mod.out` and its covariance, the convergence check's result); every setting offered with its description, set and cleared byte for byte; each finding of the pre-flight check; the results of a run (the best fit equal to ALIS's `_fit.dat`, the marks equal to its `.report`, the table and its flags, Continue from best fit and Use best fit, a line blinded since the run kept hidden); Inspect's view menu, panels, ticks, chips and progress; Compare (T fixed against T free, log N in linear form, log D/H fitted as two column densities or as a ratio, a blinded D I as |Δ|/σ only); random restarts (drawn as `sim perturb` draws them, kept with the run, assessed as ALIS assesses them, exported in `sim perturb`'s form). The fits through the dashboard's process are in `test_dashboard_runs_process.py`, `test_dashboard_results_fit.py` and `test_dashboard_compare_fit.py`, marked `fast` |
| Stage 7 (`dashboard_stage7.md`) | RJC's list of changes and bug fixes, tested in the files above: Load snips and `.mod`s (`test_dashboard_loadfit.py`); excluding a component from every ion, ratios on the cards, pure thermal's limits and the drag (`test_dashboard_components.py`, `test_dashboard_qt_components.py`); the pre-flight check's fixed values outside their limits and a total b of 0 (`test_dashboard_preflight.py`); a snip's own shift and zero level (`test_dashboard_datasets.py`, `test_dashboard_qt_regions.py`); Open a fit, the recent projects, the Add a spectrum dialog, Run on the toolbar and Inspect with the continuum (`test_dashboard_qt_*`). `test_zerolevel_fit.py` (marked `fast`) fits zero levels of either sign with `run_alis`, and `test_load_model.py` checks a zero level's limits. From the review: Export… (`test_dashboard_export.py`, which runs an exported model with `run_alis` under `fast`; `test_dashboard_qt_blinding.py`), Run greyed (`test_dashboard_preflight.py`, `test_dashboard_qt_fit.py`), the View drop-downs and the cards' lists of ions. `dashboard_synthetic.py` writes the synthetic fits these use: J101155p294141's and J013301m400628's shapes, a thermal H I and D I, a trough with a zero level, and two shifted datasets |
| Stage 8 (`dashboard_stage8.md`) | users' feedback, tested in the files above: a snip's and a dataset's resolution in each function, its bins, an LSF file in the project, lsf without linetools, and `examples/lsf_file` fitted from the window through its LSF file and with a free Afwhm (`test_dashboard_datasets.py`, `test_dashboard_qt_regions.py` (the fit marked `fast`), `test_dashboard_qt_data.py`, `test_dashboard_qt_blinding.py`); an LSF file in a bundle and in an export (`test_bundle.py`, `test_dashboard_export.py`); a temperature mode on a shared label, the limit pure thermal gives back, the drag down to its bound and a long value (`test_dashboard_components.py`, `test_dashboard_qt_components.py`); fit regions shaded to the pixels' edges (`test_dashboard_qt_plots.py`, `test_dashboard_qt_regions.py`, `test_dashboard_qt_fit.py`); a plain blind run's `.mod.out` (`test_dashboard_runs.py`, `test_dashboard_qt_blinding.py`); a continuum fitted alone (`test_dashboard_preflight.py`, `test_dashboard_qt_fit.py`, the fit marked `fast`); Export's scaled shifts and its stated faults (`test_dashboard_export.py`). Window tests compare flat lists with `approx`: CI's pytest 9.2 refuses a list of pairs |
| Stage 9 (`dashboard_stage9.md`) | the Figure tab: its figures, cells, spans, drags, duplicates, options and saved presets, every change one step (`test_dashboard_figure.py`); the presets on synthetic fits and on J1358p6522 and Q1243p307, skipped when absent (`test_dashboard_figpresets.py`); the written script of each preset run under `python -I` with LaTeX hidden (and with it, where installed), a colour edited at its top, the figure read back from matplotlib, `DH_Lyseries.py`'s layout, Q9.12's options, the PDF drawn twice the same, the shipped style, and two figures planned for export (`test_dashboard_figscript.py`; these run their scripts in processes of their own, about 15 s); the tab greyed or enabled, the preview pixel for pixel the script's figure, Stop, a failing style or script, figures created, renamed and deleted, each control a step, the drags (real drop events), a panel's options, presets, Export figure… with its PDF, and a blinded project (`test_dashboard_qt_figure.py`). The Plot expectations of `test_dashboard_markers.py`, `test_dashboard_qt_window.py` and `test_dashboard_qt_markers.py` follow the rename |
| Stage 10 (`dashboard_stage10.md`) | users' feedback and the items Stage 9 carried over, tested in the files above: SNIP over a dataset with no usable pixels, and a first guess that cannot be fitted (`test_dashboard_snips.py`, `test_dashboard_qt_regions.py`, with RJC's J0035m0918 skipped when its sources are absent); the isotopes' line IDs, the four choices of the line IDs and the list by wavelength or strength (`test_dashboard_lines.py`); the view kept across every action, the system's snips, Load snips… in the File menu, Adjust fit regions, the whole source drawn for the view and the top grip of a snip's edge (`test_dashboard_qt_regions.py`); a region's edge near the cursor, steps broken in the flux at a gap, and the two grips (`test_dashboard_qt_plots.py`); every snip's edge to one velocity (`test_dashboard_snips.py`); T's value on a card, on shared and tied labels (`test_dashboard_components.py`, `test_dashboard_qt_components.py`); the Lyα forest's lines as components of no system, their cards, ticks, drags and Use best fit, and another system adjusted as the primary is (RJC's Prompt 6: `test_dashboard_components.py`, `test_dashboard_results.py`, `test_dashboard_qt_components.py`); labels rounded down (`test_dashboard_figscript.py`, `test_plotscript.py`); the residuals, a lower label and the ticks placed in a panel, a panel's flux from its lowest value (its highest worked out), the spacing worked out, and the DH preset from a chosen member (`test_dashboard_figscript.py`, `test_dashboard_figure.py`, `test_dashboard_figpresets.py`); a drag begun on a cell (with `QDrag.exec` replaced) and the Layout pane's refusals (`test_dashboard_qt_figure.py`) |
| Stage 11 (`dashboard_stage11.md`) | a fit of several spectra and the items Stage 10 carried over, all on synthetic fits (`dashboard_fits.py`: the metal example's spectrum, copies with other noise, snips cut from them, `.mod`s naming them): a loaded `.mod` deciding the reference, the datasets of a `.mod`, which spectra hold each snip and why the others do not, the best guess, a snip cut again (its fit range carried over by wavelength), loaded unassigned or left out, the overlaps of unassigned snips, and Load snips…'s fates (`test_dashboard_loadfit.py`); every snip of a transition in a dataset (`test_dashboard_snips.py`, `test_dashboard_qt_regions.py`); the dialog that assigns snips, by buttons, real drops and double-clicks, its warnings, and every way a fit is opened or loaded with it (`test_dashboard_qt_loadfit.py`); every button's "…" (`test_dashboard_qt_window.py`); one choice of line IDs (`test_dashboard_qt_data.py`); labels from the component's scheme and Renumber labels… (`test_dashboard_edit.py`, `test_dashboard_components.py`, `test_dashboard_qt_components.py`); a commented-out line that names its redshift (`test_dashboard_components.py`) |
| Stage 12 (`dashboard_stage12.md`) | users' feedback on RJC's MWC137 project, all on synthetic fits (`dashboard_synthetic.py`: `write_rubidium_fit`, `write_twin_snips`, `write_linked_fit`; `dashboard_fits.py`): each isotope's name and the table of main isotopes against the atomic data (`test_dashboard_project.py`, `test_dashboard_components.py`, `test_dashboard_qt_components.py`); an ion a component never had, and a drag that moves a link's one free input (`test_dashboard_components.py`, `test_dashboard_qt_components.py`); Ignore warning, Warn again and the snips table's "(ignored)" (`test_dashboard_snips.py`, `test_dashboard_markers.py`, `test_dashboard_qt_regions.py`); every view's place through a step and between sessions (`test_dashboard_qt_components.py`, `test_dashboard_qt_regions.py`, `test_dashboard_qt_fit.py`, `test_dashboard_qt_session.py`, `test_dashboard_session.py`, `test_dashboard_qt_window.py`); ⇧ Shift with the wheel, ↔ and ↕ (`test_dashboard_qt_plots.py`, `test_dashboard_qt_window.py`); spectra kept in the project and Save a copy with spectra… (`test_dashboard_session.py`, `test_dashboard_qt_relink.py`); every dataset keeping its data (its snips' pixels joined, CLEAR and SNIP again, commenting out, bad pixels, a fit opened without its spectrum, a project saved before opened as it is: `test_dashboard_loadfit.py`, `test_dashboard_session.py`, `test_dashboard_qt_data.py`); `lsfspline`'s smoothing as a FWHM and its kernel centred, for three to nine knots on odd and even pixels (`test_lsfspline.py`, `unit`); the knots, the delta function, the rules, the kernel and the step (`test_dashboard_lsfknots.py`); Set LSFSpline knots… (`test_dashboard_qt_knots.py`) |
| Stage 13 (`dashboard_stage13.md`) | users' feedback after Stage 12's list, on synthetic fits: the fitting process's input read from its descriptor and its exit with 0 while its input is held open, and Keep it running letting go at once (`test_dashboard_runner.py`, `unit`; `test_dashboard_runs_process.py`, `fast`); `lsfspline`'s `locations`, `lsffile`'s `unit:kms` and the other resolution keywords written into a `.mod.out` and read back (`test_save_helpers.py`), and a run stored before read with its input model's (`test_dashboard_runs.py`); `lsffile` in km/s against `vfwhm`, its buffer and an unknown unit (`test_lsffile.py`, `unit`), and in the dashboard (`test_dashboard_datasets.py`, `test_dashboard_qt_regions.py`, `test_dashboard_qt_data.py`); a run whose outputs do not read, the warnings at chisq maxiter and of a failed convergence check, ALIS's warnings kept in `run.json`, and Warnings and options to resolve… (`test_dashboard_results.py`, `test_dashboard_qt_results.py`, `test_dashboard_runs_process.py`); the gate on a number after a word ending in "line" (`test_dashboard_blinding.py`); every file dialog's kind and Save a copy with spectra…'s Save dialog (`test_dashboard_qt_foundation.py`, `test_dashboard_qt_relink.py`); datasets of one label named apart, "K7701 (2)" (`test_dashboard_project.py`, `test_dashboard_qt_data.py`); the keys and gestures, each key the code handles with its row, the sheet's second page and each tab's Keys… (`test_dashboard_actions.py`, `test_dashboard_qt_window.py`). `examples/lsf_spline` joins the harness (its fit and its bundle; its evaluation at the best fit waits in `deferred_work.md` §7.10) |
| Stage 14 (`dashboard_stage14.md`) | ALIS's sub-pixels and their names: each data line's own `lsffile`, `apod` and `lsf` keywords, and their defaults kept (`test_lsffile.py`, `test_resolution_keywords.py`, `unit`); every name by one convention, the old ones and `nsubsigma_min` refused with their message, in a model, through `--set` and as a data-line keyword, and the new names giving the sub-pixels the old gave (`test_subpixel_names.py`, `test_config.py`); a `.mod.out`'s `nsubpixel=` written and used as given, and `examples/lsf_spline`'s and `metal_line_abs`'s `.mod.out` giving their fit's model (`test_save_helpers.py`, `test_load_files.py`; `test_subpixels_fit.py`, `fast`); the warnings of too few sub-pixels, at the snip's `nsubsigma`, each part a sentence (`test_subpixel_warnings.py`); every regenerated reference recording its sub-pixels (`test_writer_round_trip.py`); in the dashboard, the data-line keywords kept by every edit, Export… and the bundle (`test_dashboard_runs.py`, `test_dashboard_datasets.py`, `test_dashboard_export.py`, `test_bundle.py`), an old name's rename on a right-click (`test_dashboard_settings.py`, `test_dashboard_qt_modpanel.py`), the pre-flight note and Work them out again (`test_dashboard_preflight.py`, `test_dashboard_qt_fit.py`), the warnings' suggestions (`test_dashboard_results.py`, `test_dashboard_qt_results.py`), the knots on a recorded `nsubpixel` (`test_dashboard_lsfknots.py`), and a run through the dashboard's process recording 10 (`test_dashboard_runs_process.py`, `fast`). `examples/lsf_spline`'s, `brokenpowerlaw`'s and `tophat`'s evaluations at the best fit join the harness (`FIXEDPARAM_EXCLUDE` empty). The harness's rule for a zero 1σ error, which `lsf_spline`'s smoothing broke under numpy 2.5.5 on CI (`test_compare.py`, `unit`; Q14.14). `splineabs/fit_spectra_linear`'s refit and bundle skipped on Linux CI, as `fit_spectra`'s are (`MINIMISATION_MACHINE_DEPENDENT`; Prompt 5) |
| `test_dashboard_qt_{fit,results,compare,fit_tools,fit_cmd}.py` | dashboard Stage 6, the Fit tab, marked `gui` (those that fit also `fast`): Inspect (J1358p6522's chips and panels as in the mockup, the Run box, All settings…, Run refused, a run from the tab with Stop and Cancel, closing during a run); Results (as in the mockup, commit, describe, Save, remove, Continue, Use best fit on a card and in the Continuum box, Reset FWHM, no hidden value); Compare (two parametrisations of D I, Swap, Restore); Review clipped pixels, the restarts' window and Export restarts…, real restarts and the convergence check; the Fit marker in each state, a `run_alis` run taken in, the Regions tab's best fit. |

`tests/conftest.py` provides two fixtures these share: `logmsgs`, which collects
what `msgs` emits (neither `capsys` nor `capfd` sees it — the shared 'alis'
logger binds its stderr handler at import), and `atomic_data`, which loads
`alis/data/atomic.ecsv` once per session.

Several of these files are checked in a second way, because a test over an
interface can pass while asserting nothing: `test_function_interface.py`
(Stage 4.4), `test_shared_arrays.py` (Stage 4.5) and the three Stage 5.5 files
above were each run against deliberately broken versions of the code they cover,
and every invariant was confirmed to fail on the mistake it names. Keep that
property when adding to them.

## Running the batches

Tests are marked `fast`, `medium`, or `slow` by per-test wall-time:

```bash
pytest -m fast              # every commit: fast fits + all fixed-param evals
                            #   (except the ~4 min DH_orders eval) (~10 min)
pytest -m "fast or medium"  # nightly: adds single-object real-world fits
pytest --run-slow           # everything, including the slowest (>= 10 min
                            #   minimisations, e.g. DH/J0814p5029)
```

The `slow` batch is gated behind `--run-slow` (see `conftest.py`): a plain
`pytest` runs `fast` + `medium` and *skips* `slow`. Use `--run-slow` (on its
own, or with `-m slow` for only the slow tests) to include it. Batch
membership is by each case's reference runtime, so a regenerated reference can
move a case between batches (e.g. J1419p0829 / J1358p6522_original became slow
after regeneration; at dashboard Stage 14, J1419p0829 medium → fast, J1558m0031
fast → medium, and DH/J0814p5029 medium → slow).

### The window batch

Tests marked `gui` drive the dashboard's windows (dashboard Stage 3) with pytest-qt,
off-screen (`QT_QPA_PLATFORM=offscreen`, set by `conftest.py`, which also gives
pytest-qt the binding the dashboard uses, PySide6 by default). They need the `gui`
extra and pytest-qt, which is kept in its own extra because it stops pytest from
starting at all when no Qt binding is installed:

```bash
pip install -e ".[gui,gui-test,dev]"
pytest -m gui                       # every window test (~20 s)
```

Where the gui extra or pytest-qt is missing they are skipped, so the `unit` batch
and CI's `unit` and `examples` jobs stay free of Qt; CI's `gui` job runs them.

### The GPU batch

Tests marked `gpu` need a working CUDA device. They are the other way round
from `slow`: they are fast, so they run automatically wherever a device is
present and *skip* where one is not (CI, in particular), rather than hiding
behind a flag on a machine that could run them. To exercise them deliberately:

```bash
pytest --run-gpu -m gpu             # all GPU tests (~7 min)
pytest --run-gpu -m "gpu and unit"  # just the fast ones (~10 s)
```

`--run-gpu` turns a missing GPU into a `UsageError` instead of a skip, so a run
meant to test the GPU cannot pass silently on a broken CUDA install.

The batch has two layers, which catch different things (Stage 4.6):

| | what it checks | sensitivity |
|---|---|---|
| `test_voigt_gpu.py`, `test_gpu_dispatch.py` (`gpu and unit`) | the kernel against `call_CPU`, and the batched dispatch against the per-row loop | 1e-12 absolute (Q4.3) |
| `test_gpu_regression.py` | each shipped example fit that uses `voigt` (19 of 25) re-run on the GPU backend at `ngpus 1` and `4`, against the **CPU** references | ~1e-3 relative |

The second layer is deliberately blunt: it uses the ordinary Stage 0 tolerances
(1% on χ², model columns within 1% of the error bar), so it catches a broken
dispatch, a wrong reduction or a mis-bound device, not a slightly wrong profile.
Both numbers above were measured by perturbing the kernel until each layer
noticed. Neither replaces the other.

Two things stop the regression layer quietly testing nothing: it forces
`run gputhresh 0` (at the shipped default *no* example is big enough to
dispatch), and it asserts on the kernel-launch count ALIS prints at the end of
the fit.

Everything *else* is pinned to the CPU: `alisrun.force_cpu_backend` rewrites each
staged `.mod` with `run backend cpu` before it runs, so a regression case can
never wander onto the GPU (or into an `auto` timing probe, whose choice can vary
run to run) whatever the model file asks for.

Useful flags: `-v` (per-test names), `--durations=15` (slowest tests),
`-k <expr>` (select by name), `-x` (stop on first failure). A failing test
leaves its staged working copy under pytest's `tmp_path` for inspection;
passing tests clean theirs up.

## Before changing `alis/` (Stage 1+)

Run the full harness a few independent times and confirm it is green and
deterministic on the current code first — it is the safety net for every
subsequent change.
