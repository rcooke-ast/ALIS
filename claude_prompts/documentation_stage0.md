# Prompt file for ALIS documentation -- STAGE 0

> **The design of ALIS's documentation.** ALIS v2 has a new command line, project
> bundles, a GPU path, a settings system and a dashboard, none of which the v1.0
> manual describes. This document plans a documentation site on readthedocs that
> describes ALIS and the dashboard, and how to use them, covering all of ALIS's
> functionality. It is written from the workflow guide and the code, with the LaTeX
> manual of v1.0 in `doc/tex_files/` as a guide only (DQ.3).
>
> Stage 0 writes this plan and changes no code. The documentation itself is written
> in the stages it plans (Documentation Stages 1–5 below), each with its own
> document, once RJC has answered the queries at the end.
>
> Written by Claude on 2026-10-31, as Task 7.14 of `dashboard_stage7.md` (item 4 of
> its list; Q7.13). "D*n*" and "S*n*" refer to `ALIS_v2_dashboard_prompts.md`
> (D1–D153 at the close of dashboard Stage 14).

*Brought up to date by Claude on 2026-11-26, at the close of dashboard Stage 14, so
that the documentation can start from the ALIS that stage left. Dashboard Stages 8–14 came
between the plan and now. They added the Figure tab (Stage 9), fits of several
spectra (Stage 11), `lsfspline`'s smoothing as a FWHM and its `.mod.out`, `lsffile`'s
`unit:kms` (Stages 12–13), Warnings and options to resolve… (Stage 13), and ALIS's
sub-pixels (Stage 14). Stage 14's changes were its names by one convention, a `.mod.out`
that records them, the warnings of too few, and `nsubpixel_min` 10. "What exists", the outline, the reference lists
(checked against the code), the tools and the stages are revised for these. DQ.1–DQ.7 are as they were raised, with notes where
something has changed, and DQ.8–DQ.14 are new. All of them wait for RJC. Once they
are answered, Documentation Stage 1's document is written (`documentation_stage1.md`),
as each dashboard stage's was.*

*Prompt 1 (2026-11-26) applied RJC's responses to DQ.1–DQ.14:*
- *The site is written in Markdown (MyST) with the furo theme.*
- *Its pages are written from the workflow guide and the code. The v1.0 manual is a
  guide only, and no page mentions it.*
- *The user's and the developer's documentation are separate parts, and each is
  kept succinct.*
- *The dashboard is the one way to prepare data, and `prepfit` is not mentioned.*
- *Errors and warnings are numbered (reverted by Prompt 2, DQ.15).*

*Prompt 1 also checked "What exists" against the repository, with two corrections:
`doc/installing.rst` describes v1, and the workflow guide gives F10 for Keyboard
shortcuts where the code has F1. It rewrote the outline in two parts, added the
rules the pages are written by ("How the pages are written"), and replaced the
table of stages with a plan of six stages, each with its tasks and checks.
DQ.15–DQ.17 are new: where the codes of the messages live, what happens to
`prepfit`'s code, and how ALIS is cited. No code was changed. The log is
`claude_prompts/logs/documentation_stage0_log.md`.*

*Prompt 2 (2026-11-26) applied RJC's responses to DQ.15–DQ.17:*
- *The errors and warnings have no codes (DQ.15). ALIS prints its messages as it
  does now, and the dashboard's suggestions are a user's first help. Troubleshooting
  quotes the messages a user meets most, with what to do about each. Stage 5 of the
  six (the codes) is gone, and Troubleshooting moves to Stage 3, so there are five
  stages.*
- *`prepfit` is removed from ALIS in Documentation Stage 1 (DQ.16): the module, its
  test, the example, their lint entries, the README's section, and every text,
  comment and note that names it. Prompt 1 missed `tests/test_prepfit_deprecated.py`,
  and the comments and docstrings in six of the dashboard's modules; Stage 1's task
  now lists them all.*
- *About ALIS and `CITATION.cff` cite Cooke et al. (2014), ApJ, 781, 31, with GitHub's
  issues as the contact (DQ.17).*

*The stages' numbers in the earlier "Applied" notes follow the five stages. DQ.18 and
DQ.19 are new: whether a page lists every message, now that they have no codes, and
`alis/data/atomic.dat`, which only `prepfit` reads. No code was changed.*

*Prompt 2, run again (2026-11-26), applied RJC's responses to DQ.18 and DQ.19:*
- *No page lists every message (DQ.18 (a)).*
- *`alis/data/atomic.dat` is removed with `prepfit` in Stage 1 (DQ.19 (a)).*
- *RJC asked for a Help → Atomic data dialog. It is now item 17 of
  `dashboard_stage15.md`, and the Atomic data page gains it in that stage.*

*No query is open, and the plan is ready for Documentation Stage 1's document. No
code was changed.*

*Prompt 3 (2026-11-26) wrote Documentation Stage 1's document,
`claude_prompts/documentation_stage1.md`, from this plan. Its queries DQ1.1–DQ1.3
ask about the pages' addresses, the sidebar, and About ALIS's details.*

*Stage 1's Prompt 1 (2026-11-26) brought this plan into step with RJC's responses to
DQ1.1–DQ1.3:*
- *a caption for each section of the User guide;*
- *the GUI and Ryan Cooke by name, as a rule of "How the pages are written";*
- *"The GUI" and "The GUI at a glance" in the outline;*
- *the rule joins the stage template at Stage 5.*

## What exists

### The v1.0 manual (`doc/tex_files/`)

`alis_v1p0.tex` (with its PDF, `alis_v1p0.pdf`, and `doc/alis_userguide.pdf`) builds
the v1.0 manual from one file per chapter. It stays in the repository for reference
only. It is not edited again, no page of the site mentions it, and no PDF is made
for v2 (DQ.3, DQ.11). Each chapter is used as a checklist of topics: what it holds,
and what has changed since v1.0:

| File | Chapter | Holds | Since v1.0 |
|---|---|---|---|
| `introduction.tex` | Introduction | What ALIS is; the licence | Unchanged in substance; v2's additions to name. Its citation is a placeholder (DQ.17), and its licence text is not the BSD 3-Clause of `LICENSE` |
| `acknowledgements.tex` | (commented out) | — | Written from RJC's text (DQ.5) |
| `install.tex` | Installing and Updating ALIS | Requirements; updating | Python 3.13+, `pip install` with extras (`gui`, `gpu`, `gui-test`, `dev`, `docs`). `doc/installing.rst` is v1's too |
| `description.tex` | How ALIS works | `settings.alis`, `atomic.xml`, the `.mod` file | Settings are now `config.py`'s dataclasses, set in the `.mod` or with `--set`; atomic data in `atomic.ecsv`; bundles (`.model`). Its settings table was kept in step through dashboard Stage 14 (the sub-pixels' rows) |
| `fitting.tex` | Your first fit (commented out; a stub) | "hello", "there" | Replaced by the worked example of `doc/ALIS_workflow.md` §4.1 |
| `plotting.tex` | The ALIS plotting environment | Plot settings; the interface | `plot` and `plotscript` settings; `-p 0`; the plot script written beside a fit; the dashboard's Figure tab writes its own (`figscript.py`) |
| `examples.tex` | Example `.mod` files and fitting | First fit; a metal-poor DLA; α variation; isotope ratios; a quasar spectrum | `examples/` now has 22 examples (26 fits, each run by the regression harness); `examples/prepfit/` is removed in Stage 1 (DQ.16) |
| `generatedata.tex` | Generate Fake Data | The `generate` settings | Unchanged in substance |
| `montecarlo.tex` | Monte Carlo Simulations | `random`; `systematics` | `sim` settings; Random restarts… and convergence checks from the dashboard |
| `functions.tex` | Built-in functions | Afwhm, base, brokenpowerlaw, chebyshev, constant, gaussian, legendre, linear, polynomial, powerlaw, random, … | Several functions since (see the reference pages below); GPU kernels for voigt; each data line's own resolution keywords (Stage 14) |
| `newfunctions.tex` | Writing your own function | The Base function; arbitrary and polynomial functions | `functions/base.Base`, `user.py`, `set_vars`' `sampling=` (Stage 14), and the `new-alfunc` skill's recipe |
| `troubleshoot.tex` | Troubleshooting | Lists of errors and warnings (to be completed); FAQ | Messages now go through `msgs`; the dashboard's pre-flight check and Warnings and options to resolve… explain many. No codes (DQ.15); Troubleshooting quotes the common messages |
| `comments.tex` | Comments/Questions/Additions? | Contact | GitHub issues (DQ.17) |
| `contact.tex`, `bibliography.tex` | Contact; references | — | References kept where a page cites them |

`to_be_completed.txt` lists v1.0's unfinished parts: the grey boxes' width, a journal
reference (Section 1), Table 1, the atomic data note (Section 3.2), "Your first fit"
(Section 4), the examples (Section 5), the proton-to-electron mass ratio in voigt
(Section 8), the user function file (Section 9), and the lists of errors and warnings
(Sections 10.1, 10.2). Each topic is taken up by the stage that writes its page. The
lists of errors and warnings are not made (DQ.15, DQ.18): Troubleshooting quotes the
messages a user meets most.

### Since v1.0

- **`doc/ALIS_workflow.md`** ("WF", 1,452 lines). It covers:
  - preparing fitting regions (`prepfit` and snips, §0), and data (§1);
  - the model file section by section (§2): settings, data, fixing, tying and
    limiting, emission, absorption, zero level, variables and links;
  - synthetic data (§3);
  - running ALIS, project bundles, the dashboard as a user meets it (§4.3, 437
    lines, kept up to date through Stage 14), and pixels fitted twice (§4.4);
  - the output files (§5) and convergence testing (§6);
  - two appendices: voigt's parameters, and `atomic.ecsv`.

  WF is the most complete and current description, and the main source for the user
  guide. Its faults, which the new pages fix:
  - §0 describes `prepfit`, which the site does not mention (DQ.9) and Stage 1
    removes (DQ.16).
  - §0.6 ("Notes for the dashboard design") is for the dashboard's designers.
  - §4.3 is ordered by the stage that added each feature ("Since Stage 8 …"), not by
    the task a user is doing.
  - §4.3 gives F10 for Keyboard shortcuts, where the code has F1.
  - §6's subsections are numbered 5.1 and 5.2, and there is no §8.
  - §7 ("Current ALIS Code Structure", its data structures "current v1") is out of
    date.
  - §10 calls `atomic.ecsv` a VOTable file.
  - It cites `context/fitting_examples/`, which the site does not use (DQ.7).
- **`doc/installing.rst`** (92 lines): v1's installation (Python 2.7 or 3.3,
  `python setup.py develop`), not v2's. It is replaced by the Installing page (DQ.1).
- **The docstrings**, in numpydoc form throughout `alis/` and `alis/dashboard/`.
- **`examples/`**: 22 examples (CNabs to vsigma), each a `.mod` with its data and
  references, 26 fits in all; `generate` makes data. `context/fitting_examples/` holds
  real fits; they are not shipped and not used by the site (DQ.7).
- **`prepfit`**, until Stage 1 removes it (DQ.16):
  - the module, `alis/prepfit/specplot.py`, and its test,
    `tests/test_prepfit_deprecated.py`;
  - the example, `examples/prepfit/`;
  - `alis/data/atomic.dat`, which nothing else reads (DQ.19);
  - its mentions in `pyproject.toml`, the README, WF, two skills, `deferred_work.md`
    and the CHANGELOG;
  - in the dashboard:
    - four texts a user sees;
    - comments and docstrings in six modules;
    - the tests' helper `prepfit_snip`.
- **The command line**: `run_alis` (with `--set`, `--extract`, `--pack`,
  `--list-settings`, and the options of `alis/scripts/run_alis.py`) and `alis`, the
  dashboard (`alis project.model`).
- **The dashboard**: its menus and tab actions (68 in `alis/dashboard/actions.py`),
  its dialogs (23 classes in `alis/dashboard/qt/dialogs.py`, `figure.py`,
  `fittools.py` and `knots.py`, and the Column roles dialog of `data.py`), and its
  documents:
  - `claude_prompts/ALIS_v2_dashboard_prompts.md` (the decisions);
  - the stage documents (0–14) and their logs;
  - the review pages' screenshot scripts (`doc/dashboard/stage4–14/`,
    `take_screenshots.py` and the others), and the mockups.
- **`CHANGELOG.md`** and `tests/README.md` (the regression harness).
- **How to cite ALIS.** The papers in `context/publications_using_alis/` cite Cooke et
  al. 2014, ApJ, 781, 31, with the repository's URL. The site cites it too, until RJC
  prepares a citation of v2 (DQ.17).
- **No Sphinx configuration yet**: there is no `conf.py` or `.readthedocs.yaml`.
  - `pyproject.toml` has a `docs` extra (`sphinx`, `sphinx-rtd-theme`).
  - `MANIFEST.in` names a `docs/` tree, a leftover of the astropy template.
  - The `build-docs` skill builds `doc/` with the readthedocs theme, and says the tex
    files are history and the code is the authority.
  - Here, Sphinx 8.2.3 is installed; `myst-parser` and `furo` are not.

  All of these are brought into step in Documentation Stage 1.

## The readers

1. **A new user** who has a spectrum and wants a first fit: install, a first fit in
   the dashboard, and the dashboard's tabs. Needs no prior ALIS.
2. **A user of ALIS v1** with `.mod` files and snips: what changed, how to run an old
   fit from the command line or open it in the dashboard (Open a fit (.mod)…, Load
   snips…), and the new settings. An old setting's name is refused with its new name
   (D148), so the page on upgrading lists every renamed and removed setting.
3. **A modeller** writing `.mod` files by hand: the model file's full syntax, every
   function and parameter, every setting, ties, limits and links, and what the
   common messages mean.
4. **A developer** adding a model function or a GPU kernel, or changing ALIS: the
   Base interface, the registry, the tests and the regression harness.

The User guide serves readers 1–3. The Developer guide serves reader 4, and a user
never needs to open it (DQ.14).

## How the pages are written

From RJC's response to DQ.14: the documentation must be succinct for users and easy
to navigate. Every stage writes by these rules, and each review checks them.

- **One question per page.** A page is for one reader and answers one question ("how
  do I cut snips?", "what does `nsubsigma` do?"). Its first sentence says what it is
  for. A guide page can be read in a few minutes; detail beyond that goes to the
  Reference, which the guide links to.
- **Tasks before detail.** A guide page shows how to do the thing first, then explains
  the options. A reference page is tables, made from the code wherever the code holds
  the facts.
- **Users and developers apart.** Internals (module names, data structures, the
  harness, the blinding gate's code) go in the Developer guide alone. A user page
  links to it at most once, where a user might want to extend ALIS.
- **Only what a user meets.** No design history (D/Q numbers, "since Stage 12"), no
  stage notes, no mention of `prepfit` (DQ.9) or the v1.0 manual (DQ.11), and nothing
  from `context/` (DQ.7). The CHANGELOG keeps the history, in the Developer guide.
- **The code is the authority.** Where WF and the code differ, the page follows the
  code, and the difference is noted in the stage's log.
- **One source.** Each fact has one page. Others link to it rather than repeat it,
  and the reference tables are made from the code in every build, so that they
  cannot drift (DQ.10).
- **Pictures where text cannot do it.** A screenshot shows a tab or a dialog whose
  layout the text refers to. It is never decoration (DQ.7).
- **ALIS's own words.** Menu items, buttons and dialog titles are written as the
  window shows them, so that a user can find them. British spelling, as in ALIS.
- **The GUI, and Ryan Cooke, by name** (`documentation_stage1.md`, DQ1.2's and
  DQ1.3's responses).
  - A page says "the GUI", never "dashboard", except within a code path such as
    `alis/dashboard/`.
  - It names Ryan Cooke, never "RJC".
  - The same holds for every text ALIS shows or ships. Between RJC and Claude, and in
    the code, "dashboard" and "GUI" are the same thing.

## The site's outline

The site has a front page and two parts, the **User guide** and the **Developer
guide**. furo's sidebar has a caption for each section of the User guide (Getting
started, The GUI, Fitting with ALIS, Reference, About), then the Developer guide's,
last (`documentation_stage1.md`, DQ1.2). The front page says in three sentences what
ALIS is, and offers three ways in: Your first fit, The GUI at a glance, and The model
file. One line below them points developers to their part.

The pages' addresses are fixed in Documentation Stage 1 (`user/…`, `developer/…`;
for example `user/gui/fit.html`), so that links, and the GUI's Help menu (DQ.12),
never move. `documentation_stage1.md` gives each address (DQ1.1). "WF" is
`doc/ALIS_workflow.md`.

### User guide (readers 1–3)

*Getting started*
1. **Introduction**: what ALIS fits and how (a χ² fit of a model of emission,
   absorption and the instrument), with one figure of a fit of
   `examples/metal_line_abs`; the two ways to use ALIS (the dashboard and
   `run_alis`); and a pointer to citing ALIS. Sources: WF Overview; tex
   `introduction` as a guide.
2. **Installing**: Python 3.13+; `pip install` with the extras a user needs (`gui`
   for the dashboard, `gpu` with a CUDA toolkit, `linetools` for `lsf`); checking the
   install with an example. Written from `pyproject.toml`; it replaces
   `doc/installing.rst`.
3. **Your first fit**: `examples/metal_line_abs` from its spectrum to its result in
   the dashboard, then the same fit with `run_alis`. Sources: WF §4.1, and the
   dashboard.

*The GUI*

Sources for pages 4–10: WF §4.3, the code, and the decisions of
`ALIS_v2_dashboard_prompts.md`, ordered by task.

4. **The GUI at a glance**: opening it; projects and saving (bundles, autosave,
   recovery); the window (the tabs and their markers, the `.mod` panel, the status
   bar); Undo; blinding (Blind the analysis…, Blind lines…, Unblind…); Preferences…;
   keys (Keyboard shortcuts, Keys…).
5. **Data**: a new project; adding spectra and their column roles; the redshift and
   the systems (Identify a feature…); the datasets table (the reference, FWHM and
   resolution, shift, zero level); coverage; relinking spectra, and keeping them in
   the project.
6. **Regions**: choosing a transition; SNIP and CLEAR; fit regions; the continuum
   (order, Auto first guess, knots, sharing); a snip's own resolution, shift and zero
   level (with Set LSFSpline knots…); pixels fitted twice; line IDs.
7. **Components**: ions and systems; adding and dragging components; the cards
   (states, temperature modes, limits, Exclude and Include, Blind…); blends and the
   Lyα forest; links; Renumber labels….
8. **Fit**: Run, Stop and the pre-flight check. Then the sub-tabs:
   - Inspect;
   - Results (Fit statistics, Warnings and options to resolve…, the correlations,
     the run history and Commit run…);
   - Compare.

   Then the tools: Continue from best fit…, Convergence check…, Random restarts…,
   Review clipped pixels…, and Export….
9. **Figure**: figures, presets and the layout; panels and their options; the
   preview; Export figure….
10. **Earlier fits and several spectra**: Open a fit (.mod)…; Load snips…; where the
    snips go; a project of several spectra; what the dashboard reads of a `.mod` and
    its `.mod.out`.

*Fitting with ALIS*

11. **The model file**: its blocks; settings; data lines and their keywords (columns,
    fit and load ranges, resolution, shift, `bufferpix`, and the sub-pixels'
    `nsubsigma=` and `nsubpixel=`, by the convention of their names, D148); emission,
    absorption and zero level; fixing, tying and limiting; variables and links.
    Sources: WF §1–2, and the code.
12. **Running from the command line**: `run_alis` and `--set`; project bundles
    (`--pack`, `--extract`); the outputs (the `.mod.out`, which records each data
    line's `bufferpix` and `nsubpixel`; `_fit.dat`; the covariance; the report);
    plots and plot scripts (`plot`, `plotscript`); GPUs and memory; the warnings of
    too few sub-pixels (D150). Sources: WF §2.1, §4 and §5.
13. **Synthetic data and simulations**: `generate`; the convergence check; `sim`
    (random realisations, perturbed starts, `beginfrom`, systematics); and how these
    relate to the Fit tab's tools. Sources: WF §3 and §6, and `simulate.py`; tex
    `generatedata` and `montecarlo` as guides.
14. **Examples**: each shipped example, with what it shows, its model, and how to run
    it (DQ.7).
15. **Troubleshooting**: the questions users ask (a fit in a local minimum, a χ² that
    is not a number, a fit stopped at `maxiter`, too few sub-pixels, no GPU found, a
    small `/dev/shm`), each answered with what to do. Then the messages a user meets
    most, each quoted as ALIS prints it (so that the site's search finds it), with
    what it means and what to do. There are no codes (DQ.15). The dashboard's own
    findings are on the pages of the tabs that show them, which this page links to.
    Sources: WF §4.1.2 and §4.4, and the code.

*Reference*

16. **Model functions**: five pages by kind (absorption; emission and continuum;
    instrumental resolution; shifts; parameters), each function's table made from
    the code.
17. **Settings**: every setting, by section, made from `alis/config.py`.
18. **Command line**: every option of `run_alis` and `alis`, made from their parsers.
19. **Atomic data**: the atomic table, ions' names, the pseudo-ions, and using
    another table (`run atomic`). Sources: WF §10 (corrected), and the
    `atomic-data` skill.
20. **Upgrading from ALIS v1**: what a v1 user meets. This covers the renamed and
    removed settings (made from the code); the defaults and behaviours that changed
    (`settings.alis` removed, `atomic.ecsv`, `lsfspline`'s smoothing, `loadall`,
    labels like `1e4`, exit status 1, Python 3.13); and the dashboard in place of
    hand-cut snips. Sources: `CHANGELOG.md`, `config.RENAMED_SETTINGS` and
    `REMOVED_SETTINGS`.

*About*

21. **About ALIS**: citing ALIS (Cooke et al. 2014, ApJ, 781, 31, with the
    repository's URL; DQ.17); the licence (BSD 3-Clause, from `LICENSE`); the
    acknowledgements, in RJC's words (DQ.5); contact, through GitHub's issues
    (DQ.17).

### Developer guide (reader 4)

1. **How ALIS is organised**: the package's modules, and the path of a fit through
   them, written from the code (WF §7 is out of date).
2. **Adding a model function**: `functions/base.Base`, the registry
   (`main.build_funcarray`), `user.py`, `set_vars`' `sampling=`, and the tests;
   checked against the `new-alfunc` skill's recipe.
3. **GPU kernels**: `voigt_gpu.py` as the model, the dispatch and its threshold, and
   testing against the CPU (the `port-to-gpu` skill).
4. **The GUI's code**: the logic in `alis/dashboard/`, the windows in
   `alis/dashboard/qt/`, the project model, the history, the blinding gate, the
   validator and the pre-flight check, and the modes.
5. **Tests and the regression harness**: the batches and markers, the harness's
   modes and tolerances, the machine-dependent refits, and CI (from
   `tests/README.md`).
6. **Writing the documentation**: building the site, the pages made from the code,
   the screenshots, and the rule every stage follows (DQ.10).
7. **Contributing**: `CONTRIBUTING.md` (`deferred_work.md` §1.2), covering branches,
   lint with pre-commit, the changelog, and a new or reworded message that a user
   meets often, which updates its entry on Troubleshooting.
8. **API**: the modules of DQ.14 (a), from their docstrings: `functions/base.py`,
   `user.py`, `config.py`, `main.py`'s registry, `load.py`'s public functions,
   `bundle.py`, and `alis/dashboard/project.py` and `history.py`.
9. **Changelog**: `CHANGELOG.md`, included as it is.

### Not on the site

- `prepfit`, which Stage 1 removes from ALIS (DQ.9, DQ.16). The Changelog's entry
  for its removal is the one place it is named.
- Codes for the errors and warnings (DQ.15), and a list of every message (DQ.18).
- The v1.0 manual (DQ.11).
- The `context/` fits (DQ.7).
- WF §0, §0.6 and §7.
- The dashboard's design documents and stage notes.

## The reference pages

The reference pages are made from the code in every build, by a small Sphinx
extension of ALIS's own (`doc/source/_ext/alisdoc.py`, Documentation Stage 4).
Directives write the tables, and the prose around them is written by hand. Tests in
`tests/test_docs.py` fail when the code has something the pages do not (DQ.13). The
lists below are this document's own: written on 2026-10-31 and checked against the
code on 2026-11-26.

### Model functions

Every function the registry builds (`alis.main.build_funcarray`), by module of
`alis/functions/` (the modules unchanged since 2026-10-31):

| Module | Functions | Kind |
|---|---|---|
| `afwhm.py` | Afwhm | resolution |
| `apod.py` | apod | resolution |
| `brokenpowerlaw.py` | brokenpowerlaw | emission |
| `chebyshev.py` | chebyshev | emission |
| `constant.py` | constant | emission, zero level |
| `gaussian.py` | gaussian, gaussianconst, gaussianconsthelium | emission or absorption |
| `legendre.py` | legendre | emission (continuum) |
| `linear.py` | linear | emission |
| `lineemission.py` | lineemission | emission |
| `lsf.py` | lsf | resolution |
| `lsffile.py` | lsffile | resolution |
| `lsfspline.py` | lsfspline | resolution |
| `multivfwhm.py` | multivfwhm | resolution |
| `phionxs.py` | phionxs | absorption |
| `polynomial.py` | polynomial | emission |
| `powerlaw.py` | powerlaw | emission |
| `random.py` | random | parameter |
| `shift.py` | Ashift, vshift, vshiftscale, polyshift | shift |
| `spline.py` | spline | emission |
| `splineabs.py` | splineabs | absorption |
| `thar.py` | thar | emission (arc lines) |
| `tophat.py` | tophat | absorption |
| `variable.py` | variable | parameter |
| `vfwhm.py` | vfwhm | resolution |
| `voigt.py` | voigt | absorption |
| `voigtconv.py` | voigtconv | resolution |
| `vsigma.py` | vsigma | resolution |

These are not functions of the registry, and appear in the Developer guide:
`base.py` (the Base interface), `user.py` (the template for a user's function) and
`voigt_gpu.py` (voigt's GPU kernels).

The functions are grouped by kind on five pages (absorption; emission and continuum;
instrumental resolution; shifts; parameters), not one page each, so that a user
compares the functions of a kind in one place. Each function's section gives:
- a description, written by hand (tex `functions` as a guide);
- its table, made from the code: its parameters (`_parid`), their defaults
  (`_defpar`), whether each is fixed by default (`_fixpar`), their limits
  (`_limited`, `_limits`), and its keywords (`_keywd`);
- an example line, which a test parses.

Three faults found in dashboard Stage 14 touch these pages (`apod` cannot be
evaluated, `lsf`'s buffer, a model loaded again). They are offered to dashboard Stage
15 (its items 12–14). The pages describe the functions as they should work, each with
a note saying what does not work yet, which Stage 15 removes (DQ.10).

### Settings

Every setting of `alis/config.py`, by section:

- **run**: atomic, backend, bintype, blind, cache, capvalue, convcriteria,
  convergence, convnostop, datadirc, datatype, gputhresh, last_update, limpar, logn,
  modname, ncpus, ngpus, nsubpixel_max, nsubpixel_min (10 since dashboard Stage 14),
  nsubpixel_renew, nsubsigma, shmem.
  - The sub-pixels' names follow one convention (dashboard Stage 14, D148). A name
    that counts sub-pixels per pixel begins `nsubpixel`; one that counts them across
    a profile's standard deviation begins `nsubsigma`; `_min` and `_max` bound them.
  - The old names (`nsubpix`, `nsubmin`, `nsubmax`, `renew_subpix`) and the removed
    `warn_subpix` and `nsubsigma_min` are refused. The Settings page gives the new
    names alone; Upgrading from ALIS v1 lists the old ones.
  - A data line takes `nsubsigma=N` (its own target) and `nsubpixel=N` (its
    sub-pixels per pixel, used as given). ALIS writes `nsubpixel=N` on each data line
    of a `.mod.out` (D149), and warns where a profile has fewer sub-pixels than its
    snip's `nsubsigma` across its standard deviation (D150).
- **chisq**: atol, fstep, ftol, gtol, maxiter, miniter, xtol.
- **plot**: dims, fitregions, fits, labels, only, pages, residuals, ticklabels,
  ticks, xaxis.
- **plotscript**: figsize, filename, fontsize, format, labels, numcol, overwrite,
  residuals, velrange, ylim.
- **out**: convtest, covar, fits, model, modelname, overwrite, plots, reletter,
  report, reportsig, verbose, wavecorr.
- **sim**: beginfrom, convergesig, convergetest, dirname, edgecut, maxperturb,
  newstart, perturb, random, repeat, startid, systematics, systmodule.
- **generate**: data, overwrite, peaksnr, pixelsize, skyfrac.
- **iterate**: data, model.

Each setting's entry gives:
- its type, default and what it does, from `config.py`'s field metadata, so that they
  cannot drift;
- whether it describes the model or the run (the dashboard's split, Q6.2);
- the command-line option that sets it, if any.

Upgrading from ALIS v1 lists the renamed and removed settings from
`config.RENAMED_SETTINGS` and `REMOVED_SETTINGS`.

### Errors and warnings

ALIS has 335 `msgs.error` calls and 122 `msgs.warn` calls in 40 modules
(2026-11-26; one of the warnings is `prepfit`'s, which Stage 1 removes). They have no
codes (DQ.15):
- ALIS prints them as it does now;
- the dashboard keeps reading ALIS's messages by their text (the `.mod` panel's
  problems, `model.py`; the run's warnings, `results.py`);
- the dashboard's suggestions (the pre-flight check, the `.mod` panel's problems,
  Warnings and options to resolve…) are a user's first help.

The site describes the messages in one place, Troubleshooting (Stage 3):
- **Which messages.** Those a user meets most:
  - the model file's syntax, labels, limits and links;
  - data files and their columns;
  - the sub-pixels' warnings;
  - a fit that stops at `maxiter`, or whose χ² is not a number;
  - GPUs and shared memory.

  Stage 3 chooses them from the code, and from the questions WF answers.
- **Each entry** gives:
  - the message as ALIS prints it, with the parts filled in when it is raised shown
    as …;
  - what it means and what to do;
  - a link to the page that says more.
- **Kept in step.**
  - Each message is quoted in a fenced `alis-message` block (a plain-text lexer,
    registered in `conf.py` beside `alis`).
  - Many messages are built from pieces, for example `"Keyword '" + name + "' is
    unknown for -"`. A test in `tests/test_docs.py` therefore checks that each fixed
    part of a quote (the text between its …) is in the string literals of one
    `msgs.error` or `msgs.warn` call. A reworded message then fails the test until
    its entry is updated.
  - A stage that adds a message a user will often meet adds its entry.

## The tools

- **Sphinx** (8.2 here) with **MyST-Parser**: every page in Markdown (DQ.1).
- **The theme:** `furo` (DQ.2). The `docs` extra becomes `sphinx`, `furo` and
  `myst-parser`.
- **Pages made from the code:** the extension `doc/source/_ext/alisdoc.py`, with a
  directive each for a function's table, a section's settings, the renamed settings,
  and the command line's options.
- **Model files on a page** are fenced as `alis`, for a small Pygments lexer
  registered in `conf.py`. It highlights them, and lets a test find and parse them.
  The messages Troubleshooting quotes are fenced as `alis-message`, for the same
  reason.
- **The API pages:** `sphinx.ext.autodoc` with `sphinx.ext.napoleon` (which reads the
  numpydoc form), for the Developer guide alone and the modules of DQ.14 (a). The
  User guide does not depend on docstrings.
- **No conversion from LaTeX** (DQ.3): the pages are written, not converted, and
  `pandoc` is not used. Equations are MyST math, drawn by MathJax.
- **readthedocs:** the project `ALIS-fitting`, at `alis-fitting.readthedocs.io`
  (DQ.6). `.readthedocs.yaml` builds on Ubuntu with Python 3.13 (`pip install
  .[docs]`, `fail_on_warning: true`), in HTML alone: no PDF or ePub (DQ.3). Each
  documentation stage's branch is built as a hidden version (DQ.8).
- **The checks** (DQ.13 (a)):
  - a `docs` job in CI, on Ubuntu, on every push:
    `sphinx-build -W --keep-going -b html doc/source doc/_build/html`;
  - `tests/test_docs.py`, in the `unit` batch (so also on every push). It checks
    that nothing the code has is missing from the pages (functions, settings, the
    command line's options, the dashboard's actions), that every `alis` block parses,
    that every message Troubleshooting quotes is one ALIS prints, and that every
    screenshot a page uses exists;
  - linkcheck (`sphinx-build -b linkcheck`), in the `build-docs` skill and at each
    release, since other sites' links fail at random.

## The dashboard's screenshots

- **One script.** `doc/source/screenshots.py` (Documentation Stage 2) gathers the
  review pages' helpers (`doc/dashboard/stage11–14/`). It drives the window
  off-screen at 1440×900 and writes every screenshot the pages use into
  `doc/source/_static/screenshots/`.
- **Committed.** The screenshots are committed, and the build only reads them (DQ.7).
- **Shipped examples and synthetic data only** (`tests/dashboard_fits.py`), never
  `context/` (DQ.7). The review pages of Stages 9 and 10 used RJC's projects; the
  script does not.
- **Which pages.** The dashboard's pages and Your first fit have screenshots. The
  Examples page draws a figure for a few examples only (DQ.7).
- **When.** The screenshots are made again before each release, and by any later
  stage that changes what a tab shows.

## The plan of the work

### How each stage runs

- **Its documents.** Each stage has its own document
  (`claude_prompts/documentation_stageN.md`), written when the stage before it
  closes, with its design, tasks, checks and queries. Its log is
  `claude_prompts/logs/documentation_stageN_log.md`.
- **Its branch.** Each stage has a branch (`alis_v2_doc_stageN`, made by RJC), which
  readthedocs builds as a hidden version (DQ.8).
- **After every task:**
  - the site builds with no warning (`build-docs`);
  - `tests/test_docs.py` passes;
  - pre-commit passes on the task's files.

  A task that changes ALIS also runs the batches it touches (`unit`, `gui`, and the
  regression harness by Stage 14's rules), also under CI's versions (memory
  `ci-test-versions`).
- **The review is the last task but one.** It is a claude.ai page that links each page
  the stage built on the hidden version, says what to look for on each, and asks the
  stage's open questions. RJC's comments arrive as threads on it.
- **The last task closes the stage:** its checks, CI on every job, and the next
  stage's document.
- **What a documentation stage changes outside `doc/`:**
  - the build (`pyproject.toml`'s `docs` extra, `.readthedocs.yaml`, `MANIFEST.in`,
    `.gitignore`);
  - its tests and CI;
  - the `build-docs` skill;
  - the README, `CONTRIBUTING.md` and `CITATION.cff`;
  - ALIS itself only where an answered query says: `prepfit`'s removal, with
    `atomic.dat` (DQ.16, DQ.19), and the Help menu (DQ.12). ALIS's messages are not
    changed (DQ.15).

### The stages

| Stage | Builds | Changes to ALIS |
|---|---|---|
| Documentation 1 | The skeleton: the build, every page of the outline with its sources, About ALIS, the checks in CI, readthedocs | `prepfit` removed (DQ.16) |
| Documentation 2 | Getting started and the dashboard: Introduction, Installing, the seven dashboard pages, Your first fit, `screenshots.py` | Help → ALIS documentation (DQ.12) |
| Documentation 3 | Fitting with ALIS: the model file, the command line, synthetic data and simulations, atomic data, Troubleshooting; WF becomes a pointer | Three comments cite the site (DQ.10) |
| Documentation 4 | The reference made from the code: model functions, settings, command line, Upgrading from ALIS v1 | None |
| Documentation 5 | Examples, the Developer guide, the README and `CONTRIBUTING.md`; the release | None |

**Why this order:**
- The dashboard is how most users meet ALIS, and the one way to prepare data (DQ.9),
  so its pages come first after the skeleton.
- Troubleshooting comes with Fitting with ALIS. Its answers come from WF (§4.1.2's
  local minima, §4.4's pixels fitted twice), and WF hands over to the site at the end
  of Stage 3.
- The reference pages come after the guides that link to them. Their addresses exist
  from Stage 1, so the links work before the pages are filled.
- The Developer guide comes last, because it describes the build, tests and
  extension that the other stages make.

Without the codes (DQ.15), and with no page of every message (DQ.18), the plan has
five stages, as D87, `deferred_work.md` §1.2 and `dashboard_stage15.md` already say.

### Documentation Stage 1 — The skeleton

Every later stage writes into pages that already have their place and address, on a
build that is checked from the first push.

**RJC first** (DQ.6, DQ.8):
- Create the readthedocs project `ALIS-fitting`, linked to the GitHub repository.
- Make the branch `alis_v2_doc_stage1`, and activate it on readthedocs as a hidden
  version.
- Until `alis_v2` is merged, `master` has no configuration, so `latest` builds
  nothing. It can be deactivated until the merge.

**Tasks:**
1. **The build:**
   - `doc/source/conf.py`, with MyST (colon fences, definition lists, math), furo,
     autodoc and napoleon, the `alis` and `alis-message` lexers, and the version from
     `alis.__version__`;
   - the `docs` extra: `sphinx`, `furo` and `myst-parser`;
   - `.readthedocs.yaml`, building HTML alone;
   - `doc/_build/` in `.gitignore`, and `MANIFEST.in`'s `docs` lines pointed at
     `doc/source`.
2. **The pages.**
   - Every page of the outline has its title, its sentence of purpose, and a comment
     that lists its sources.
   - The front page offers its three ways in.
   - The sidebar has the User guide's and the Developer guide's captions.
   - About ALIS is written in full:
     - the acknowledgements in RJC's words (DQ.5);
     - the licence;
     - citing ALIS: Cooke et al. (2014), ApJ, 781, 31, with the repository's URL
       (DQ.17);
     - contact through GitHub's issues.
   - `CITATION.cff` goes at the repository's root. It describes the software (title,
     authors, licence, repository, version), with the 2014 paper as its preferred
     citation, so that GitHub offers "Cite this repository".
   - When RJC prepares a citation of v2, About ALIS and `CITATION.cff` are the two
     places it changes.
3. **The checks:**
   - the CI `docs` job (Ubuntu, every push, `-W`);
   - `tests/test_docs.py` (unit marker), with its first test: every page of the
     outline exists, at its fixed address.
4. **The `build-docs` skill** pointed at `doc/source`, with furo, MyST, `-W`, and
   linkcheck on request.
5. **`prepfit` removed** (DQ.16 (a)):
   - **Removed:**
     - `alis/prepfit/` (`specplot.py` and `__init__.py`);
     - its test, `tests/test_prepfit_deprecated.py`, and its row in
       `tests/README.md`;
     - `examples/prepfit/` (its README and two scripts);
     - their entries in `pyproject.toml`'s black, isort and ruff lists;
     - `alis/data/atomic.dat`, which only `prepfit` reads (DQ.19). The first lines
       of `atomic_README` then name two files: `atomic.ecsv`, which ALIS reads, and
       `atomic.xml`, the old format.
   - **The texts a user sees, reworded:**
     - two tooltips in `actions.py`: Load snips… and Adjust fit regions: one
       spectrum;
     - the tooltip of the Adjust fit regions ▾ menu (`qt/regions.py`);
     - Load snips…'s message for a file whose name names no transition
       (`loadfit.NO_NAME`).

     A snip is described by its columns (wavelength, flux, error, then a fit range
     of 1s and 0s) and its file name (`<stem>_<ion>_<wavelength>_reg.dat`). The
     tests of these texts follow.
   - **The code's comments and docstrings, reworded:**
     - `lines.py`, whose ranking of lines is described for itself;
     - `loadfit.py`, `modes.py` and `snips.py`, where "`prepfit`'s four-column
       format" becomes the snip's four columns;
     - `qt/plots.py` and `qt/regions.py`.
   - **The tests:**
     - the helper `prepfit_snip` (`tests/dashboard_fits.py`) is renamed;
     - so are the two tests named for `prepfit` in `test_dashboard_loadfit.py`;
     - the docstrings that name it are reworded.
   - **The documents:**
     - the README's "Setup a fit" section becomes a line pointing to the dashboard
       (Stage 5 rewrites the README);
     - WF's §0 becomes a paragraph pointing to the Data and Regions tabs, keeping
       the snip's format, and WF's other mentions are reworded (WF becomes a pointer
       in Stage 3);
     - the notes in the `gui-dev` and `gui-component` skills;
     - the entries in `deferred_work.md` that name `alis/prepfit/`:
       - §1.1's note on `set_regions`;
       - §5's testing gap, which keeps `plot.py`;
       - §7.3's second copy of the solar abundances;
       - §7.8's reason.
   - **The CHANGELOG:** under [Unreleased], the Deprecated entry becomes a Removed
     one, and the two mentions in Added are reworded.
6. **The design documents in step.** `deferred_work.md` §1.2 says that the plan's
   queries are answered. D87 and `dashboard_stage15.md` already name five stages.
7. **The review:** the skeleton on its hidden version, its navigation and the
   outline, About ALIS, and the texts reworded in the dashboard.
8. **The close.**

**Checks:**
- The local build and readthedocs' hidden version build with no warning.
- The CI `docs` job passes.
- Every page of the outline exists at its address, with its sources listed.
- No file under `doc/source/` names `prepfit`, the v1.0 manual or `context/`.
- `git grep -i prepfit` finds only:
  - the CHANGELOG's Removed entry;
  - the history: `claude_prompts/`, `logs/`, and the review pages' scripts in
    `doc/dashboard/`.
- No file outside the history names `atomic.dat`.
- The `unit` and `gui` batches pass, also under CI's versions. No fit changes, so
  the regression harness is not needed.
- `CITATION.cff` is valid (`cffconvert --validate`, in a scratch venv).

### Documentation Stage 2 — Getting started and the dashboard

A new user can install ALIS, make a first fit in the dashboard, and find every tab,
menu item and dialog described.

**Tasks:**
1. **Introduction and Installing.**
   - The Introduction's figure comes from `examples/metal_line_abs`.
   - Installing is rewritten from `pyproject.toml`: the extras, CUDA, `linetools`, and
     the check with an example.
   - `doc/installing.rst` is removed.
2. **`screenshots.py`:** the script, the folder of screenshots, and a test that every
   image a page uses exists and is one the script names.
3. **The GUI at a glance**, and **Data**.
4. **Regions** and **Components**.
5. **Fit**, with its three sub-tabs and its tools, and **Figure**. The pre-flight
   check's findings and Warnings and options to resolve… are described here, each
   with what to do. These are the dashboard's suggestions that DQ.15's response
   relies on.
6. **Earlier fits and several spectra.** Load snips… describes a snip by its columns
   and its file name, as the dashboard's texts do after Stage 1.
7. **Your first fit.** `examples/metal_line_abs` in the dashboard (New project…, the
   Data, Regions, Components and Fit tabs, then a figure), then from the command line
   (`run_alis fit_spectra.mod`). Its numbers come from a fresh run.
8. **Help → ALIS documentation** (DQ.12):
   - The action opens the page of the tab in view (and the sub-tab of Fit), on the
     documentation of the installed version (`latest` for a development version).
   - The map from tabs to pages lives in `alis/dashboard/` and is tested there; the
     action has a `gui` test.
   - Keyboard shortcuts keeps F1.
9. **The completeness test:** every action of `actions.ACTIONS`, and every dialog's
   title, is named on a dashboard page. Cut, Copy, Paste and the other standard edit
   actions are exempt.
10. **The review.**
11. **The close.**

**Checks:**
- The build has no warning.
- Every tab and sub-tab has its page, and its screenshots are made by the script
  alone, from shipped examples and synthetic data.
- The completeness test passes.
- The Help link's tests pass, with the `gui` batch.
- Your first fit's numbers match a fresh run.

### Documentation Stage 3 — Fitting with ALIS

A modeller can write, run and check a `.mod` from the pages alone, and WF hands over
to the site (DQ.10).

**Tasks:**
1. **The model file** (WF §1–2, and the code): every block and every data-line
   keyword.
2. **Running from the command line** (WF §2.1, §4, §5): options, `--set`, bundles,
   outputs, plots and plot scripts, GPUs and memory, and the warnings of too few
   sub-pixels.
3. **Synthetic data and simulations** (WF §3, §6 with its numbering fixed, and
   `simulate.py`). How `sim perturb` draws its starts is Stage 15's candidate 2: the
   page describes what ALIS does now, with a note.
4. **Atomic data** (WF §10 corrected; checked with the `atomic-data` skill).
5. **Troubleshooting** (as "The reference pages" describes; DQ.15):
   - the questions: local minima and a small starting b (WF §4.1.2); pixels fitted
     twice (WF §4.4); a χ² that is not a number; a fit stopped at `maxiter`; too few
     sub-pixels; no GPU found; a small `/dev/shm`;
   - the messages a user meets most, chosen from the code, each in an
     `alis-message` block with what it means and what to do;
   - links to the dashboard's findings on the Fit page (Stage 2).
6. **The tests:**
   - every `alis` block on any page is read by ALIS's loader. Fragments (a data
     line, a few settings) are read inside a minimal model;
   - every `alis-message` block's text is in a `msgs.error` or `msgs.warn` call.
7. **One source** (DQ.10):
   - WF becomes a short page that points to the site;
   - the comments in `alis/dashboard/validate.py` (two) and `preflight.py` (one) cite
     the site's pages.
8. **The review.**
9. **The close.**

**Checks:**
- The log has a table with every WF section and tex chapter, and its page, or why it
  was dropped (WF §0, which Stage 1 cut; §0.6; §7).
- Every `alis` block parses.
- Every message Troubleshooting quotes is one ALIS prints. A message reworded in a
  scratch copy fails the test.
- `to_be_completed.txt`'s items for these pages are done (the atomic data's note, the
  first fit). Its lists of errors and warnings are not made (DQ.15, DQ.18).
- No page names a `context/` fit.
- The build has no warning.

### Documentation Stage 4 — The reference, made from the code

Every function, setting and option is on a page that the code writes. A function or
setting added without its page fails CI.

**Tasks:**
1. **`alisdoc.py`:** the directives for a function's table (`alis-function`), a
   section's settings (`alis-settings`), the renamed and removed settings
   (`alis-renamed`), and the command line (`alis-cli`, from `run_alis`'s and `alis`'s
   parsers).
2. **Model functions**, on five pages: each function's description, its table, and
   an example line. The notes on Stage 15's items 12–14 go here (DQ.10).
3. **Settings:** the tables, and a note by hand on each setting that needs more than
   its line.
4. **Command line.**
5. **Upgrading from ALIS v1.**
6. **The completeness tests:**
   - every function of `build_funcarray` has its directive on a page;
   - every section of `ArgFlag` has its directive on the Settings page;
   - the Command line page has the directive of each parser;
   - every function's example line parses.
7. **The review.**
8. **The close.**

**Checks:**
- A function added to the registry in a scratch test, with no page, fails
  `test_docs.py`.
- The tables match `--list-settings` and `build_funcarray`.
- The build has no warning.

### Documentation Stage 5 — Examples, the Developer guide, and the release

The site is complete, its links are checked, and the rule that keeps it in step is
written where every later stage reads it.

**Tasks:**
1. **Examples:**
   - a table of the shipped examples, giving what each shows and the functions it
     uses;
   - each example's purpose, its model (included from `examples/`, so it cannot
     drift), and how to run it;
   - a figure for a few examples only (DQ.7).
2. **The Developer guide, part 1:** How ALIS is organised; Adding a model function
   (checked against the `new-alfunc` skill); GPU kernels.
3. **The Developer guide, part 2:** The GUI's code (which says once that the GUI's
   code is `alis/dashboard/`); Tests and the regression harness; Writing the
   documentation.
4. **`CONTRIBUTING.md`** at the repository's root, included in the site. **API**
   (DQ.14 (a)), with no warning from those modules' docstrings. A hook in
   `alisdoc.py` drops each docstring's "Generated by …" line, so that no API page
   shows "RJC" (`documentation_stage1.md`, DQ1.4 (a)); the code keeps its comments
   and docstrings as they are. **Changelog**, with the CHANGELOG's
   [Unreleased] section rewritten for the release: the GUI by name, and no design
   stages or query numbers.
5. **The README:** short, with the site's link, the install line, and citing ALIS
   (as About ALIS does).
6. **The rules for later stages** (DQ.10): the template (`dashboard_stageX.md`) and
   `dashboard_stage15.md` gain them.
   - The task just before the review updates the site's pages that the stage's
     changes touch, and the site builds with no warning after every task (Q15.3).
   - Every text a user sees says "GUI" and "Ryan Cooke", never "dashboard" or "RJC"
     (`documentation_stage1.md`, DQ1.2 and DQ1.3).
7. **Before the release:** linkcheck, the screenshots made again, and a final pass of
   the whole site against "How the pages are written".
8. **The review** of the whole site, on its hidden version.
9. **The close.**

**Checks:**
- Every shipped example has its entry, and its included model is the harness's own.
- Linkcheck passes.
- The API pages build with no warning, and none shows "RJC".
- RJC's review is applied.
- The rule is in the template.

### After the documentation

- **Dashboard Stage 15** comes next, under DQ.10's rule (its Q15.3 (a)).
  - Its items 12–14 remove the pages' notes on the three faults.
  - Its item 17, Help → Atomic data (RJC, DQ.19's response), adds the dialog to the
    Atomic data page and to The GUI at a glance.
- **Every later stage** keeps the site in step: the task just before its review
  updates the pages its changes touch (DQ.10's response).
- **At the merge** of `alis_v2` into `master`, RJC sets readthedocs to build `master`
  as `latest`, each release tag as its own version, and `stable` as the newest
  release (DQ.6). The hidden versions of the stages are then deactivated.

## Skills to use

- `build-docs`: each task's build, and linkcheck at the release.
- `gui-dev`: the dashboard's screenshots, and the Help menu's link.
- `run-example`: Your first fit and the examples' figures.
- `run-tests` and `gen-tests`: `tests/test_docs.py`, and the batches of the stages
  that change ALIS.
- `new-alfunc`: checking Adding a model function against its recipe.
- `atomic-data`: the Atomic data page.
- `test-coverage`: the Help menu's test.
- Claude Code's `artifact-design` skill: each stage's review page.

## Context

- `doc/ALIS_workflow.md` (the main source), `CHANGELOG.md`, `tests/README.md`,
  `pyproject.toml`.
- `doc/tex_files/` and `to_be_completed.txt`, as a checklist only (DQ.3).
- `alis/config.py` (the settings and their metadata, `RENAMED_SETTINGS`,
  `REMOVED_SETTINGS`), `alis/functions/`, `alis/main.py` (`build_funcarray`),
  `alis/scripts/` (the parsers), `alis/logger.py` (`msgs`, for Troubleshooting's
  test).
- `alis/dashboard/actions.py` (every menu item and tab action) and
  `alis/dashboard/qt/dialogs.py` (the dialogs).
- `claude_prompts/ALIS_v2_dashboard_prompts.md`, the dashboard's stage documents and
  their logs.
- `doc/dashboard/` (the review pages' screenshot scripts), and
  `tests/dashboard_fits.py` (synthetic fits).
- `examples/` (each example's `.mod`, data and references).
- For `prepfit`'s removal (Stage 1): the files listed in its task, and
  `claude_prompts/deferred_work.md`.

## Queries

*DQ.1–DQ.7 raised by Claude on 2026-10-31 (Task 7.14 of `dashboard_stage7.md`);
DQ.8–DQ.14 on 2026-11-26, at the close of dashboard Stage 14, with notes added to the
first seven where something has changed. Each gives Claude's lean. Prompt 1
(2026-11-26) applied RJC's responses (noted under each) and raised DQ.15–DQ.17.
Prompt 2 (2026-11-26) applied the responses to DQ.15–DQ.17 and raised DQ.18 and
DQ.19. Prompt 2, run again, applied their responses. No query is open.*

**DQ.1 — The source format.** Markdown (through MyST) or reStructuredText for the
pages. Markdown is what `doc/ALIS_workflow.md` is written in and is easier to edit;
reStructuredText is Sphinx's own, with richer cross-references.

My lean: Markdown (MyST), with `doc/installing.rst` kept as it is.

**Response:** I agree with your lean. Markdown is best. It's also fine if `doc/installing.rst` is converted to Markdown, but it's not necessary.

*Applied (Prompt 1): every page is Markdown. `doc/installing.rst` describes v1
(Python 2.7, `setup.py`), so the Installing page replaces it in Stage 2 instead of
converting it.*

**DQ.2 — The theme.** `sphinx-rtd-theme` (readthedocs' familiar look) or `furo` (a
cleaner, more modern one, with a dark mode).

My lean: `furo`.

*Note (2026-11-26): the `docs` extra already names `sphinx-rtd-theme`, from the
refactor's Task 2.5; that was not a choice of theme, and the extra follows this
answer.*

**Response:** Furo sounds like a better choice for a modern look and dark mode. Let's choose `furo`.

*Applied (Prompt 1): the `docs` extra becomes `sphinx`, `furo` and `myst-parser` in
Stage 1.*

**DQ.3 — The v1.0 manual.** Convert its chapters with `pandoc` and edit them, or
rewrite each chapter from WF and the code, using the tex chapters as a checklist.
Much of tex `description` and `plotting` is out of date.

My lean: rewrite from WF, keeping tex text where it is still right (functions,
generatedata, montecarlo, newfunctions), converted with `pandoc`.

*Note (2026-11-26): the `build-docs` skill already says the tex files are history and
the code is the authority, which this lean follows.*

**Response:** The original documentation should be used as a guide only (feel free to include whatever information is available), and the new documentation should be based on the workflow and code. No tex files or pdf will be made available for v2 and beyond. The tex files will be kept in the repository for reference only.

*Applied (Prompt 1): the pages are written, not converted (no `pandoc`), with the
tex chapters as a checklist of topics. readthedocs builds HTML alone, with no PDF or
ePub.*

**DQ.4 — The lists of errors and warnings.** Made from the code (the `msgs.error` and
`msgs.warn` calls, grouped by module, each with a hand-written explanation), or
written by hand for the common ones only.

My lean: made from the code, so that none is missed, with explanations written for
the commonest first.

*Note (2026-11-26): there are 335 errors and 122 warnings in 40 modules. Made from the
code, each gets its place at once, and an explanation is written for the commonest (those
a new user meets: the model file's syntax, limits, data files, the sub-pixels'
warnings); the rest show their message and module until written.*

**Response:** Perhaps we should number the errors and warnings for easier reference. For example, we could use a format like "Error Code: E.1.001" for errors in Module 1, error 001. Similarly, "Warning code: W.1.001" can be used for warnings in Module 1, error 001. This would make it easier to refer to specific messages in the documentation and in discussions.

*Applied (Prompt 1): the codes take RJC's form (`E.3.012`, `W.3.012`), as described
in "The reference pages". Where they live, so that the code a user sees is the one the
pages give and never changes, is DQ.15.*

*Reverted (Prompt 2): the messages have no codes (DQ.15's response).*

**DQ.5 — The acknowledgements.** v1.0 has the chapter commented out.

My lean: a short page, with the funding and the people you name.

**Response:** It should acknowledge the following funding sources:
Throughout the development of ALIS, RJC acknowledges support from the following funding sources:
 - Peterhouse, Cambridge
 - University of California Santa Cruz Morrison Fellowship
 - NASA Hubble Fellowship
 - Royal Society (RS) University Research Fellowship (URF)
 - Royal Society (RS) Faraday Discovery Fellowship (FDF)
Version 2 of ALIS was developed with the support of Anthropic Claude. Many others have contributed to suggesting improvements and testing the software, and their contributions are gratefully appreciated.

*Applied (Prompt 1): About ALIS carries this text, written in Stage 1.*

**DQ.6 — Where the site lives.** readthedocs under `alis` (if free) or another name,
built from `master`, with a version per release tag.

My lean: as described, once the v2 branch is merged into `master`.

*Note (2026-11-26): the name's availability is not checked here; creating the
readthedocs project, and linking it to the GitHub repository, are RJC's (DQ.8 asks
what it builds before the merge).*

**Response:** I agree with your lean. The project name on ReadTheDocs is called `ALIS-fitting`.

*Applied (Prompt 1): `alis-fitting.readthedocs.io`. RJC's steps are at the start of
Stage 1, and the merge's under "After the documentation".*

**DQ.7 — The screenshots in the repository.** Commit them under `doc/source/_static/`,
or make them in the readthedocs build (which needs Qt and a display there).

My lean: commit them, made by `screenshots.py` before each release; the build only
reads them.

*Note (2026-11-26): the examples' figures (chapter 11, and the Introduction's) raise
the same question; drawing them in the build would run 26 fits there. The lean covers
both: committed, each made by its script.*

**Response:** There;s no need to add screenshots for all the examples, and no examples should be made based on the context files. I agree that the screenshots should be committed and made by `screenshots.py` before each release.

*Applied (Prompt 1): the screenshots are committed. `screenshots.py` uses shipped
examples and synthetic data alone, the Examples page draws a figure for a few
examples, and no page names or draws a `context/` fit.*

**DQ.8 — What readthedocs builds while the site is written.** The stages take the
site from a skeleton to the release over several branches, before `alis_v2` is merged
into `master` (DQ.6).
- **(a)** readthedocs builds each documentation stage's branch as a hidden version
  (it has a URL but is not listed), so that each review is of the real site, and the
  public versions start at the merge.
- **(b)** Built here only, with each review a page made from the local build, and
  readthedocs set up at Documentation Stage 5.
- **(c)** Public from Documentation Stage 1, marked as in progress.

My lean: (a). The build on readthedocs is the one users will get, and Stage 1's check
needs it. It needs the project to exist from Stage 1 (RJC's, DQ.6).

**Response:** I agree with your lean. The build on ReadTheDocs should be the one users will get, and it is important to have the project set up from Stage 1.

*Applied (Prompt 1): RJC's setup opens Stage 1. Each stage's review page links its
pages on that stage's hidden version.*

**DQ.9 — `prepfit` and preparing data.** The scripts that select fitting regions and
write snips live in `examples/prepfit/`, outside the package, and are copied and
edited by each user (its README). The dashboard does the same job with Add a
spectrum, the Regions tab and Load snips….
- **(a)** Preparing data leads with the dashboard, and describes `prepfit` as the
  example it is, for users who prefer scripts or have snips from v1.
- **(b)** As (a), and `prepfit` becomes part of the package (a console command), which
  is a change to ALIS for a dashboard stage, not a documentation one.
- **(c)** Preparing data describes `prepfit` alone (as WF §0 does), and the dashboard's
  way is in chapter 4.

My lean: (a). Users who prefer scripts or have v1 snips keep a documented path, and
packaging it can wait for someone to ask.

**Response:** There should be no mention of prepfit anymore. The dashboard is now the only supported way to prepare data. The documentation should focus on the dashboard's functionality for preparing data.

*Applied (Prompt 1): the chapter "Preparing data" is gone. Its subject is the Data and
Regions pages, and Load snips… is described for snips from earlier fits, by their
columns. ALIS still ships `prepfit` (`alis/prepfit/specplot.py`, which this query
missed, as well as `examples/prepfit/`), and the dashboard's text names it in four
places: DQ.16.*

**DQ.10 — One source, from Documentation Stage 2.** Until now, each dashboard stage's
close updated WF, the tex files and the CHANGELOG. Once the site's chapters hold that
text, two copies would drift.
- **(a)** The site is the one source. WF becomes a short page that points to it, the tex
  files stay as v1.0's manual and are no longer edited (DQ.11), and every later stage's
  close (dashboard Stage 15 first) updates the site's pages that its changes touch, as
  part of its checks.
- **(b)** WF is kept as a one-page guide beside the site, and both are updated.
- **(c)** WF is the source, and the site includes it whole.

My lean: (a). It is how the reference pages already work (made from the code), and it
gives each stage one place to update. Three comments in `alis/dashboard/`
(`validate.py` twice, `preflight.py` once) cite WF's sections, and are changed to cite
the site's pages. A fault that waits for a fix (the
three of Stage 14's items 12–14) is described as it should work, with a note saying
what does not work yet, and Stage 15's close removes the notes.

**Response:** I agree with your lean. I will also note that the tex files will be kept in the repository for reference only, but they will not be updated or edited anymore. The site will serve as the primary source of all documentation. We will ensure that the penultimate task of each stage of development is always to update the site's pages that are affected by the changes made in that stage.

*Applied (Prompt 1): WF becomes a pointer in Stage 3, once its text is on the site.
Stage 5 writes the rule into the stage template and `dashboard_stage15.md`. A stage's
last two tasks are its review and its close, so "penultimate" is read as the task
just before the review, so that RJC reviews the updated pages with the rest of the
stage. The CHANGELOG is kept, and included in the Developer guide.*

**DQ.11 — The v1.0 manual's files.** `doc/tex_files/` (its sources and
`alis_v1p0.pdf`) and `doc/alis_userguide.pdf`, once the site covers them.
- **(a)** Kept as v1.0's manual, unedited from then on, and linked from the site's page
  on changes.
- **(b)** Removed from the repository (git keeps them), with the site's page on changes
  linking to v1.0's tag.

My lean: (a). The PDF is what users of v1 know, and keeping it costs nothing.

**Response:** There should be no mention of the v1.0 manual in the new documentation. The v1.0 manual will be kept in the repository for reference only, but it will not be updated or edited anymore. The site will serve as the primary source of all documentation for ALIS v2 and beyond.

*Applied (Prompt 1): no page mentions the manual, and Stage 1's check looks for it.
Upgrading from ALIS v1 describes ALIS v1's settings and behaviour, not its manual.*

**DQ.12 — The dashboard's link to the site.** Its Help menu has Keyboard shortcuts
(F1) and About ALIS, and no link to any documentation.
- **(a)** Help → ALIS documentation, which opens the site's page for the tab in
  view, in Documentation Stage 4 (a small change to the dashboard, with its test).
- **(b)** One link to the site's front page.
- **(c)** No link: the README and the site are enough.

My lean: (a). A user who is stuck on a tab reaches the page that explains it in one
step; the pages' addresses are fixed in Stage 4, with the dashboard chapter.

**Response:** I agree with your lean. The Help menu should have a link to the ALIS documentation, which opens the site's page for the tab in view. This will provide users with quick access to relevant documentation while using the dashboard.

*Applied (Prompt 1): built in Stage 2, with the dashboard's pages, on the
documentation of the installed version. The addresses are fixed in Stage 1.*

**DQ.13 — The documentation in CI.** The four checks of "The tools", and when each
runs.
- **(a)** A `docs` job on Ubuntu on every push: the build with no warning, the
  reference lists against the code, and the `.mod` blocks parsed. Linkcheck runs in
  the `build-docs` skill and at each release, since other sites' links fail at
  random.
- **(b)** All four on every push.
- **(c)** None in CI: readthedocs' build is the check.

My lean: (a). A setting or function added without its page fails the push that adds
it, and an external site being down does not.

**Response:** I agree with your lean.

*Applied (Prompt 1): the `docs` job runs the build. The tests against the code and
the blocks' test are in `tests/test_docs.py`, which the `unit` job runs on every
push.*

**DQ.14 — The API pages.** The developer chapter's reference to the code.
- **(a)** The modules a developer extends or calls: `functions/base.py`, `user.py`,
  `config.py`, `main.py`'s registry, `load.py`'s public functions, `bundle.py`, and the
  dashboard's model (`alis/dashboard/project.py` and `history.py`), each from its
  docstrings.
- **(b)** The whole package, `alis/` and `alis/dashboard/` (including `qt/`), made
  with `autosummary`.
- **(c)** None: the chapter describes the layout in prose, and points to the code.

My lean: (a). Those docstrings are the ones a developer reads, and keeping the build
free of warnings over the whole package (b) would be a stage of docstring work on its
own, which can follow.

**Response:** I agree with you lean. Please note that it would be helpful not to mix developer and user documentation. There should be a separate section for anything that developers/contributors need to know. We do not want to overwhelm users with developer information. Let's spend effort to really focus on one specific goal: We must ensure that the documentation is succinct to users, and that it is easy to navigate.

*Applied (Prompt 1): the site is two parts, the User guide and the Developer guide.
"How the pages are written" sets the rules for succinct pages. The API pages of (a)
are in the Developer guide (Stage 5).*

*DQ.15–DQ.17 raised by Claude on 2026-11-26 (Prompt 1).*

**DQ.15 — Where the codes of the errors and warnings live (DQ.4's response).** A code
helps a user only if it is the code they see in the terminal, and only if it never
changes. Two questions.
- **(i) Where the code is kept.**
  - **(a)** In ALIS.
    - Each `msgs.error` and `msgs.warn` call carries its code, and ALIS prints it
      after the level: `[ERROR]   :: E.3.012  Error reading in file …`.
    - The module numbers are in one table (`alis/messages.py`), assigned once and
      never reused.
    - The Errors and warnings page is made from the calls.
    - A test fails on a call with no code, a code used twice, or a code in the
      wrong module.
    - This changes ALIS's output in 40 modules. It is built in Documentation Stage 5,
      with the regression harness and the dashboard's tests run. The dashboard reads
      ALIS's messages: it would show the code, and recognise the sub-pixels'
      warnings by their code rather than their text.
  - **(b)** In the documentation alone.
    - The pages number the messages, and ALIS prints them as now.
    - A user finds a message on the page by its text.
    - A number moves whenever a message is added before it, unless someone keeps a
      table by hand that matches each text to its code.
  - **(c)** As (a), but built in dashboard Stage 15 (a change to ALIS), with the
    page written then.
- **(ii) Which messages get codes.**
  - **(a)** ALIS's errors and warnings alone. The dashboard's own findings (the
    pre-flight check, the `.mod` panel's problems) are described on the pages of the
    tabs that show them.
  - **(b)** The dashboard's findings too, with a letter of their own.

My lean: (i)(a) and (ii)(a). A code printed with its message takes a user straight
to its entry, and a table in the code keeps the code stable. The dashboard's findings
are already sentences with their fixes beside them, on the tab that shows them. If
(i)(b), Stage 5 joins Stage 4, and the plan has five stages.

**Response:** I'm now not convinced that we need to have all of the codes. Let's revert this idea of having error and warning codes. There is sufficient suggestions in the dashboard to help users that encounter errors and warnings.

*Applied (Prompt 2):*
- *No codes. ALIS's messages are not changed, and the dashboard keeps reading them by
  their text.*
- *The stage of the codes is gone, and Troubleshooting moves to Stage 3, with the
  pages whose questions it answers. The plan has five stages.*
- *The Errors and warnings page is gone from the outline. Troubleshooting quotes the
  messages a user meets most, each with what to do, and a test keeps each quote in
  step with the code.*
- *The dashboard's suggestions are described on the Fit page (Stage 2).*
- *Whether a page should still list every message, without codes, is DQ.18.*

**DQ.16 — `prepfit`'s code (DQ.9's response).** The site will not mention `prepfit`,
but ALIS still ships it, in these places:
- `alis/prepfit/specplot.py`: deprecated since dashboard Stage 6, untested, and a
  matplotlib window on the Qt5Agg backend;
- `examples/prepfit/`: two scripts and a README;
- their entries in `pyproject.toml`'s lint lists;
- the README's "Setup a fit" section;
- four places in the dashboard's text: Load snips…'s tooltip, the tooltip of
  Adjust fit regions: one spectrum, and the Regions tab's hint (all three "as in
  prepfit"), and Load snips…'s message for a file whose name names no transition,
  which says how `prepfit` names its snips.

The options:
- **(a)** Removed in Documentation Stage 1: the module, the example, their lint
  entries and the README's section. The dashboard's four texts are reworded without
  the name (a snip is described by its columns and its file name). A CHANGELOG entry
  goes under Removed. Nothing imports `alis.prepfit` (checked), and the tests'
  helper `prepfit_snip` is renamed, since it is internal.
- **(b)** As (a), but in dashboard Stage 15.
- **(c)** Kept, and not mentioned.

My lean: (a). Then nothing ALIS ships or shows contradicts the site from its first
page. The removal is self-contained: one module that nothing imports, an example
folder, and four strings.

**Response:** I agree with your lean. We should remove `prepfit` in Documentation Stage 1, including the module, the example, their lint entries, and the README's section. The dashboard's four texts should be reworded without mentioning `prepfit`. This will ensure that nothing ALIS ships or shows contradicts the site from its first page.

*Applied (Prompt 2): Stage 1's task 5 lists every change. Checking for this prompt
found more than the query listed:*
- *`prepfit` has a test, `tests/test_prepfit_deprecated.py`, which imports it in a
  process of its own. "Untested" and "nothing imports it" above were wrong. The test
  is removed with the module.*
- *Comments and docstrings name it in six of the dashboard's modules (`lines.py`,
  `loadfit.py`, `modes.py`, `snips.py`, `qt/plots.py`, `qt/regions.py`). Most
  describe a snip as "`prepfit`'s four-column format", and they are reworded.*
- *It is also named in WF, the `gui-dev` and `gui-component` skills, four entries of
  `deferred_work.md`, and the CHANGELOG. These are reworded too, and the CHANGELOG's
  Deprecated entry becomes a Removed one.*
- *`alis/data/atomic.dat` is read by `prepfit` alone: DQ.19.*

*Stage 1's check: `git grep -i prepfit` finds only the CHANGELOG's Removed entry and
the history (`claude_prompts/`, `logs/`, `doc/dashboard/`).*

**DQ.17 — Citing ALIS, and contact.** The v1.0 manual's citation is a placeholder
("Cooke R. (2013), JOURNAL, REFERENCE"). The papers in
`context/publications_using_alis/` cite Cooke et al. 2014, ApJ, 781, 31, with the
repository's URL.
- **(a)** About ALIS asks users to cite Cooke et al. (2014, ApJ, 781, 31) and to give
  the repository's URL. A `CITATION.cff` at the repository's root says the same, so
  that GitHub offers "Cite this repository". The contact is GitHub's issues alone.
- **(b)** As (a), plus a DOI for each release (from Zenodo, set up by RJC), cited
  beside the 2014 paper.
- **(c)** Another reference, or your email as a contact as well.

My lean: (a), with (b) whenever you want v2 to have a citable version of its own. The
2014 paper is what the papers already cite, and GitHub's issues keep questions and
their answers where other users find them.

**Response:** I agree with option (a), and I will work towards preparing a proper citation in the future. For now, please continue to mention the paper: Cooke et al. (2014), ApJ, 781, 31

*Applied (Prompt 2): in Stage 1, About ALIS asks users to cite Cooke et al. (2014),
ApJ, 781, 31, with the repository's URL. `CITATION.cff` gives the same paper as its
preferred citation. The contact is GitHub's issues. When the new citation is ready,
these two places change, and the README (Stage 5) follows About ALIS.*

*DQ.18 and DQ.19 raised by Claude on 2026-11-26 (Prompt 2).*

**DQ.18 — A page of every message, without codes (DQ.15's response).** The plan
had an Errors and warnings page that listed every message by its code. Without the
codes, the page could go, or stay as a plain list.
- **(a)** No such page.
  - Troubleshooting quotes the messages a user meets most (the model file's syntax,
    labels, limits and links, data files, the sub-pixels, convergence, GPUs and
    memory), each with what it means and what to do.
  - A test checks that each quote is still a message ALIS prints.
  - The rest are left to the message's own words, and to the dashboard's
    suggestions.
- **(b)** A Messages page in the Reference, made from the code.
  - It lists all 456 messages (once `prepfit`'s warning is gone), grouped by
    subject, each with its text, and an explanation where one is written.
  - A test checks that every message is on it.
  - It is built in Stage 4, with the reference's extension.

My lean: (a). Most messages already say what is wrong ("Keyword '…' is unknown for
…"). A list of 456, most of them with no explanation, is long to read, and it
would go against the succinct pages of DQ.14's response. Users of the dashboard have
its suggestions. A user of `run_alis` who searches the site for a message finds it
on Troubleshooting if it is a common one.

**Response:** I agree with (a). This is no longer necessary.

*Applied (Prompt 2, run again): the outline, "The reference pages" and Stage 4 have
no Messages page. Troubleshooting (Stage 3) is the one place the site describes
messages, and its test keeps each quote in step with the code.*

**DQ.19 — `alis/data/atomic.dat`.** It is an atomic table in an old format (15 kB,
shipped in the package through `data/*`). Only `prepfit` reads it (`specplot.py`).
`atomic_README` says "atomic.dat <--- This is used for prepfit", and the converter
`convert_datFormat_to_xmlFormat.py` does not read it (despite its name, it holds its
lines in the script). ALIS reads `atomic.ecsv`, and still accepts `atomic.xml`.
- **(a)** Removed with `prepfit` in Stage 1. `atomic_README`'s first lines then name
  two files: `atomic.ecsv`, which ALIS reads, and `atomic.xml`, the old format. git
  keeps the file, should anyone want it.
- **(b)** Kept, and `atomic_README` says it is an old table that ALIS does not read.

My lean: (a). Once `prepfit` is gone, nothing reads it, and a table that ALIS ships
but never reads would raise questions on the Atomic data page.

**Response:** I agree with your lean. On a related note, please add a note to the stage 15 dashboard design document to include a Help menu item called `Atomic data` that opens a dialog listing the atomic data for every transition in ALIS. It can be filtered by element, ion, and wavelength range. We do not need to discuss it further here, but please ensure that it is included in the design document for stage 15.

*Applied (Prompt 2, run again):*
- *Stage 1's task 5 removes `atomic.dat` and rewrites `atomic_README`'s first lines.
  A new check is that no file outside the history names `atomic.dat`.*
- *Help → Atomic data is item 17 of `dashboard_stage15.md`, in a new section,
  "Asked for by RJC". It is in the stage, not a candidate to choose. The item gives
  what it would take:*
  - *the logic without Qt, reading the project's own atomic table;*
  - *the dialog, read-only, with its filters by element, ion and wavelength range;*
  - *the action and its tests;*
  - *the pages it changes.*
- *"After the documentation" notes it.*

## Prompts
1. Please read my responses to your queries, the ALIS code base, and the dashboard development (their design documents and logs). If anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please update this document to reflect any changes to the design document based on my responses. When you have a clear perspective on the documentation, please write a plan for the documentation development, including the stages, tasks, and checks, and any other relevant information. We will develop one stage at a time, so please make a plan for how many stages are required, and what each stage will accomplish.
2. Please read my responses to your queries. If anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please update this document to reflect any changes to the design document based on my responses. When you have a clear perspective on the documentation, please update the plan for the documentation development, including the stages, tasks, and checks, and any other relevant information. We will develop one stage at a time, so please make a plan for how many stages are required, and what each stage will accomplish. Note that we will no longer be using error and warning codes, and we will remove `prepfit` in Documentation Stage 1. Please ensure that the documentation reflects these changes and that the plan for the documentation development is updated accordingly, including each stage design documents.
3. Please write the stage 1 design document, and ask any questions, if anything is unclear.