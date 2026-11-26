# Prompt file for ALIS documentation -- STAGE 0

> **The design of ALIS's documentation.** ALIS v2 has a new command line, project
> bundles, a GPU path, a settings system and a dashboard, none of which the v1.0
> manual describes. This document plans a documentation site on readthedocs that
> describes ALIS and the dashboard, and how to use them, covering all of ALIS's
> functionality and building on the LaTeX manual of v1.0 in `doc/tex_files/`.
>
> Stage 0 writes this plan and changes no code. The documentation itself is written
> in the stages it plans (Documentation Stages 1–5 below), each with its own
> document, after RJC has answered the queries at the end.
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

## What exists

### The v1.0 manual (`doc/tex_files/`)

`alis_v1p0.tex` (with its PDF, `alis_v1p0.pdf`, and `doc/alis_userguide.pdf`) builds
the v1.0 manual from one file per chapter. Each chapter, what it holds, and what has
changed since v1.0:

| File | Chapter | Holds | Since v1.0 |
|---|---|---|---|
| `introduction.tex` | Introduction | What ALIS is; the licence | Unchanged in substance; v2's additions to name |
| `acknowledgements.tex` | (commented out) | — | To write, or leave out (DQ.5) |
| `install.tex` | Installing and Updating ALIS | Requirements; updating | Python 3.13+, `pip install` with extras (`gui`, `gui-test`, `dev`, `docs`); `doc/installing.rst` is newer |
| `description.tex` | How ALIS works | `settings.alis`, `atomic.xml`, the `.mod` file | Settings are now `config.py`'s dataclasses, set in the `.mod` or with `--set`; atomic data in `atomic.ecsv`; bundles (`.model`). Its settings table was kept in step through dashboard Stage 14 (the sub-pixels' rows) |
| `fitting.tex` | Your first fit (commented out; a stub) | "hello", "there" | Replaced by the worked example of `doc/ALIS_workflow.md` §4.1 |
| `plotting.tex` | The ALIS plotting environment | Plot settings; the interface | `plot` and `plotscript` settings; `-p 0`; the plot script written beside a fit; the dashboard's Figure tab writes its own (`figscript.py`) |
| `examples.tex` | Example `.mod` files and fitting | First fit; a metal-poor DLA; α variation; isotope ratios; a quasar spectrum | `examples/` now has 23 folders and 26 fits, each run by the regression harness |
| `generatedata.tex` | Generate Fake Data | The `generate` settings | Unchanged in substance |
| `montecarlo.tex` | Monte Carlo Simulations | `random`; `systematics` | `sim` settings; Random restarts… and convergence checks from the dashboard |
| `functions.tex` | Built-in functions | Afwhm, base, brokenpowerlaw, chebyshev, constant, gaussian, legendre, linear, polynomial, powerlaw, random, … | Several functions since (see the reference lists below); GPU kernels for voigt; each data line's own resolution keywords (Stage 14) |
| `newfunctions.tex` | Writing your own function | The Base function; arbitrary and polynomial functions | `functions/base.Base`, `user.py`, `set_vars`' `sampling=` (Stage 14), and the `new-alfunc` skill's recipe |
| `troubleshoot.tex` | Troubleshooting | Lists of errors and warnings (to be completed); FAQ | Messages now go through `msgs`; the dashboard's pre-flight check and Warnings and options to resolve… explain many |
| `comments.tex` | Comments/Questions/Additions? | Contact | GitHub issues |
| `contact.tex`, `bibliography.tex` | Contact; references | — | Kept |

`to_be_completed.txt` lists v1.0's unfinished parts: the grey boxes' width, a journal
reference (Section 1), Table 1, the atomic data note (Section 3.2), "Your first fit"
(Section 4), the examples (Section 5), the proton-to-electron mass ratio in voigt
(Section 8), the user function file (Section 9), and the lists of errors and warnings
(Sections 10.1, 10.2). Each is taken up in the stage that writes its chapter.

### Since v1.0

- **`doc/ALIS_workflow.md`** ("WF", 1,452 lines): preparing fitting regions (`prepfit`
  and snips), data, the model file section by section (settings, data, fixing, tying
  and limiting, emission, absorption, zero level, variables, links), synthetic data,
  running ALIS, project bundles, the dashboard as a user meets it (§4.3, 437 lines,
  kept up to date through Stage 14), pixels fitted twice (§4.4), the output files,
  convergence testing, and two appendices (voigt's parameters; `atomic.ecsv`). It is
  the most complete and current description, and the main source for the new
  chapters. Its faults, for the rewrite: §6's subsections are numbered 5.1 and 5.2,
  there is no §8, §0.6 ("Notes for the dashboard design") is for the dashboard's
  designers, and §7 ("Current ALIS Code Structure", its data structures "current v1")
  is out of date.
- **`doc/installing.rst`** (92 lines): installing v2 (reStructuredText already).
- **The docstrings**, in numpydoc form throughout `alis/` and `alis/dashboard/`.
- **`examples/`**: 23 folders and 26 fits with references (CNabs to vsigma), each a `.mod` with
  its data; `generate` makes data; `prepfit` holds the scripts that select fitting
  regions and write snips (`select_fitting_regions.py`, `myobjects.py`, and a README),
  which are not part of the package (DQ.9). `context/fitting_examples/` holds real
  fits, not shipped.
- **The command line**: `run_alis` (with `--set`, `--extract`, `--pack`,
  `--list-settings`) and `alis`, the dashboard (`alis project.model`).
- **The dashboard's documents**: `claude_prompts/ALIS_v2_dashboard_prompts.md` (the
  decisions), the stage documents (0–14), the review pages' screenshot scripts
  (`doc/dashboard/stage4–14/`, `take_screenshots.py` and the others) and the mockups.
- **`CHANGELOG.md`** and `tests/README.md` (the regression harness).
- **No Sphinx configuration yet**: there is no `conf.py` or `.readthedocs.yaml`.
  `pyproject.toml` has a `docs` extra (`sphinx`, `sphinx-rtd-theme`), and the
  `build-docs` skill builds `doc/` and says the tex files are history and the code
  is the authority. Both are updated in Documentation Stage 1 (the theme by DQ.2, the
  skill to `doc/source/`).

## The readers

1. **A new user** who has a spectrum and wants a first fit: install, a tour of the
   dashboard, a worked example from spectrum to results. Needs no prior ALIS.
2. **A user of ALIS v1** with `.mod` files and snips: what changed, how to run an old
   fit from the command line or open it in the dashboard (Open a fit (.mod)…), and the
   new settings. An old setting's name is refused with its new name (D148), so the
   page on changes lists every renamed and removed setting.
3. **A modeller** writing `.mod` files by hand: the model file's full syntax, every
   function and parameter, every setting, ties, limits and links, and messages.
4. **A developer** adding a model function or a GPU kernel, or changing ALIS: the
   Base interface, the registry, the tests and the regression harness.

## The site's outline

Each chapter, its sources, and what must be written new. "WF" is
`doc/ALIS_workflow.md`; "tex" is the v1.0 chapter of that name.

1. **Introduction** — tex `introduction`; what v2 adds. *New:* a one-page "what ALIS
   does", with a figure from an example fit.
2. **Installing** — `doc/installing.rst`, tex `install`. *New:* the extras, GPUs
   (CUDA through numba), and checking the install with an example.
3. **Your first fit** — WF §4.1 (metal_line_abs). Replaces tex `fitting`. *New:* the
   same fit done twice, from the command line and from the dashboard.
4. **The dashboard** — WF §4.3 and `ALIS_v2_dashboard_prompts.md`.
   - *New, one page per tab:* Data, Regions, Components, Fit (its Inspect, Results and
     Compare sub-tabs) and Figure, with the `.mod` panel, blinding, the history,
     preferences and keys.
   - The menus (File, Edit, View, Model, Fit, Help) and the dialogs (23), among them
     Add a spectrum, All settings…, Export…, Set LSFSpline knots…, Random restarts…,
     Continue from best fit, and Warnings and options to resolve….
   - A project of several spectra (Stage 11), and opening an earlier fit (Open a fit
     (.mod)…, Load snips…).
   - Each page gets screenshots made by script (below).
5. **Preparing data** — WF §0–1 (`prepfit`, snips, fitting regions), tex
   `description`; the dashboard's Add a spectrum and Regions tab (DQ.9).
6. **The model file** — WF §2 and tex `description`.
   - Each block: settings, data, the model's emission, absorption and zero level,
     variables and links.
   - Fixing, tying and limiting.
   - The data line's keywords: the resolution and each line's own keywords (`lsffile`,
     `lsfspline`, `apod`, `lsf`), `bufferpix`, and the sub-pixels (`nsubsigma=`,
     `nsubpixel=`), with the convention of their names (D148).
   - *New:* bundles, and `run datadirc`.
7. **Running ALIS** — WF §4, the `run_alis` command line (`--set`, `--extract`,
   `--pack`, `--list-settings`), outputs (`.mod.out`, which records each data line's
   `bufferpix` and `nsubpixel`; `_fit.dat`; the covariance; the report), the warnings
   of too few sub-pixels (D150), GPUs and memory (WF §2.1).
8. **Plotting** — tex `plotting`; `plot` and `plotscript` settings; the plot script;
   the Figure tab's scripts, with a pointer to chapter 4.
9. **Synthetic data** — tex `generatedata`, WF §3.
10. **Simulations and convergence** — tex `montecarlo`, WF §6; the `sim` settings;
    Random restarts… and the convergence checks of the Fit tab.
11. **Examples** — tex `examples` and `examples/`: one page per example, its purpose,
    its `.mod`, and its fit drawn.
12. **Reference: model functions** — tex `functions`, made from the code (below).
13. **Reference: settings** — made from `alis/config.py` (below).
14. **Writing your own function** — tex `newfunctions`, `functions/base.py`,
    `functions/user.py`; adding a GPU kernel (`voigt_gpu.py` as the model).
15. **Troubleshooting** — tex `troubleshoot`; the lists of errors and warnings, made
    from the `msgs.error` and `msgs.warn` calls (DQ.4); the pre-flight check's
    findings and Warnings and options to resolve…; an FAQ.
16. **For developers** — the tests (`tests/README.md`), the regression harness (its
    modes, tolerances, and the machine-dependent refits), the dashboard's code layout
    (the logic in `alis/dashboard/`, the windows in `alis/dashboard/qt/`, the
    blinding gate and the history), the API (DQ.14), `CONTRIBUTING.md`
    (`deferred_work.md` §1.2), and how the documentation is built.
17. **Changes, contact and references** — `CHANGELOG.md`, the move from v1 (renamed
    and removed settings), tex `comments`, `contact`, `bibliography`, and the
    acknowledgements (DQ.5).

## The reference lists

Both lists are made from the code by a script (`doc/source/make_reference.py`,
Documentation Stage 3), so that they stay complete. The pages it writes are checked
in CI against the code: a function or a setting the code has and the pages do not
fails the build. The lists below are this document's own, written on 2026-10-31 and
checked against the code on 2026-11-26.

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

Not functions of the registry: `base.py` (the Base interface, chapter 14),
`user.py` (the template for a user's function, chapter 14) and `voigt_gpu.py`
(voigt's GPU kernels, chapters 7 and 14).

Each function's page gives its parameters (`_parid`), their defaults (`_defpar`),
whether they are fixed by default (`_fixpar`), their limits (`_limited`,
`_limits`), its keywords (`_keywd`), and an example line. The parameters come from
the code; the descriptions are written by hand, from tex `functions` where it has
them. Three faults found in dashboard Stage 14 touch these pages (`apod` cannot be
evaluated, `lsf`'s buffer, a model loaded again), and are offered to dashboard Stage
15 (its items 12–14): the pages describe the functions as they should work, and say
which is waiting for a fix until it is made (DQ.10).

### Settings

Every setting of `alis/config.py`, by section:

- **run** — atomic, backend, bintype, blind, cache, capvalue, convcriteria,
  convergence, convnostop, datadirc, datatype, gputhresh, last_update, limpar, logn,
  modname, ncpus, ngpus, nsubpixel_max, nsubpixel_min (10 since dashboard Stage 14),
  nsubpixel_renew, nsubsigma, shmem. The names of the sub-pixels follow one convention
  (dashboard Stage 14, D148): a name that counts sub-pixels per pixel begins
  `nsubpixel`, one that counts them across a profile's standard deviation begins
  `nsubsigma`, and `_min` and `_max` bound them; the old names (`nsubpix`, `nsubmin`,
  `nsubmax`, `renew_subpix`) and the removed `warn_subpix` and `nsubsigma_min` are
  refused, and the pages give the new names alone. A data line takes `nsubsigma=N`
  (its own target) and `nsubpixel=N` (its sub-pixels per pixel, used as given; ALIS
  writes it on each data line of a `.mod.out`, D149), and ALIS warns where a profile
  has fewer than its snip's `nsubsigma` across its standard deviation (D150).
- **chisq** — atol, fstep, ftol, gtol, maxiter, miniter, xtol.
- **plot** — dims, fitregions, fits, labels, only, pages, residuals, ticklabels,
  ticks, xaxis.
- **plotscript** — figsize, filename, fontsize, format, labels, numcol, overwrite,
  residuals, velrange, ylim.
- **out** — convtest, covar, fits, model, modelname, overwrite, plots, reletter,
  report, reportsig, verbose, wavecorr.
- **sim** — beginfrom, convergesig, convergetest, dirname, edgecut, maxperturb,
  newstart, perturb, random, repeat, startid, systematics, systmodule.
- **generate** — data, overwrite, peaksnr, pixelsize, skyfrac.
- **iterate** — data, model.

Each setting's entry gives its type, default, what it does, whether it describes
the model or the run (the dashboard's split, Q6.2), and the command-line flag that
sets it, if any. The type, default and help text come from `config.py`'s field
metadata, so they cannot drift. The renamed and removed settings come from
`config.RENAMED_SETTINGS` and `REMOVED_SETTINGS`, for the page on changes.

### Errors and warnings

ALIS has 335 `msgs.error` calls and 122 `msgs.warn` calls in 40 modules (2026-11-26).
How their list is made is DQ.4.

## The tools

- **Sphinx**, with sources in Markdown through MyST (DQ.1). `doc/ALIS_workflow.md` and
  the new chapters are Markdown already. `doc/installing.rst` stays
  reStructuredText, which Sphinx reads beside it.
- **The theme:** `sphinx-rtd-theme` or `furo` (DQ.2).
- **The API pages:** `sphinx.ext.autodoc` with `sphinx.ext.napoleon` (Sphinx's own,
  which reads the numpydoc form) for the developer chapter alone, its scope by
  DQ.14. The user chapters do not depend on docstrings.
- **The v1.0 chapters:** converted from LaTeX by `pandoc`, then edited by hand
  (DQ.3). Equations stay as LaTeX math (MathJax).
- **readthedocs:** `.readthedocs.yaml`, `doc/source/conf.py`, and the `docs` extra in
  `pyproject.toml` (with `myst-parser` and the theme). The project on readthedocs is
  RJC's to create (DQ.6, DQ.8).
- **How the build is checked:**
  - `sphinx-build -W -b html doc/source doc/_build/html`: no warning;
  - `sphinx-build -b linkcheck`: no broken link;
  - the reference lists against the code (above);
  - each code block of a `.mod` that the pages show is parsed by ALIS in a test.

  The `build-docs` skill runs the first two. Which run in CI, and when, is DQ.13.

## The dashboard's screenshots

The review pages' scripts already drive the window off-screen at 1440×900 and save
each tab, on real fits and on synthetic ones (`doc/dashboard/stage4–14/`,
`take_screenshots.py` and `take_close_screenshots.py`). Documentation Stage 4 gathers
them into one script, `doc/source/screenshots.py`, which writes every screenshot the
pages use, from the shipped examples and synthetic data alone. The screenshots are
made again before each release, so that they show the dashboard as it is. Where they
are kept is DQ.7.

## The stages of the work

Each stage gets its own document (`documentation_stageN.md`), written when the stage
before it is done, as the dashboard's stages were, with its tasks, their checks and a
log (`claude_prompts/logs/documentation_stageN_log.md`). Each ends with RJC's review
of the pages it built, on a page made from the built site. They change no behaviour of
ALIS: what they change outside `doc/` is the build, its tests and CI, the `docs` extra,
the `build-docs` skill, the README's link to the site, and the dashboard's Help menu
if DQ.12 says so.

| Stage | Builds | Check |
|---|---|---|
| Documentation 1 | The skeleton: `conf.py`, `.readthedocs.yaml`, the `docs` extra, the outline's 17 pages (titles and sources), the `build-docs` skill pointed at it, and the build in CI (DQ.13) | `build-docs` builds with no warning; readthedocs builds the branch (DQ.8) |
| Documentation 2 | The user chapters from the existing text: Introduction, Installing, Preparing data, The model file, Running ALIS, Plotting, Synthetic data, Simulations (from WF and the tex chapters) | Every tex chapter and WF section has its place; every `.mod` block shown parses; `to_be_completed.txt`'s items for these chapters are done |
| Documentation 3 | The references: `make_reference.py`, the function and settings pages, the lists of errors and warnings, the page on changes from v1 | The pages list every function, setting and message the code has (a test), and the build fails when one is missing |
| Documentation 4 | The dashboard chapter and Your first fit, with `screenshots.py` (and the Help menu's link, DQ.12) | Every tab, sub-tab and dialog has its page and screenshot; the screenshots are made by the script alone |
| Documentation 5 | Examples, Writing your own function, Troubleshooting, For developers (with `CONTRIBUTING.md` and the API, DQ.14); the README's link; the review and the release | Every example has its page; linkcheck passes; RJC's review applied; readthedocs serves the site (DQ.6) |

After Documentation Stage 5, dashboard Stage 15 (`dashboard_stage15.md`) changes ALIS
where its items say (`apod`, `lsf`'s buffer, a model loaded again, and the examples
of its item 16), and its close updates the pages they touch (DQ.10).

## Skills to use

- `build-docs`: each stage's build.
- `run-example`: the examples' pages and their figures.
- `gui-dev`: the dashboard's screenshots.
- `new-alfunc`: checking the Writing your own function chapter against its recipe.
- `atomic-data`: the atomic data's section.

## Context

- `doc/tex_files/` (the v1.0 manual) and `to_be_completed.txt`.
- `doc/ALIS_workflow.md`, `doc/installing.rst`, `CHANGELOG.md`, `tests/README.md`.
- `alis/config.py` (the settings and their metadata, `RENAMED_SETTINGS`,
  `REMOVED_SETTINGS`) and `alis/functions/`.
- `claude_prompts/ALIS_v2_dashboard_prompts.md` and the dashboard's stage documents.
- `doc/dashboard/` (the screenshot scripts of the review pages).
- `examples/` (each example's `.mod`, data and reference).

## Queries

*DQ.1–DQ.7 raised by Claude on 2026-10-31 (Task 7.14 of `dashboard_stage7.md`);
DQ.8–DQ.14 on 2026-11-26, at the close of dashboard Stage 14, with notes added to the
first seven where something has changed. Each gives Claude's lean.*

**DQ.1 — The source format.** Markdown (through MyST) or reStructuredText for the
pages. Markdown is what `doc/ALIS_workflow.md` is written in and is easier to edit;
reStructuredText is Sphinx's own, with richer cross-references.

My lean: Markdown (MyST), with `doc/installing.rst` kept as it is.

**Response:** I agree with your lean. Markdown is best. It's also fine if `doc/installing.rst` is converted to Markdown, but it's not necessary.

**DQ.2 — The theme.** `sphinx-rtd-theme` (readthedocs' familiar look) or `furo` (a
cleaner, more modern one, with a dark mode).

My lean: `furo`.

*Note (2026-11-26): the `docs` extra already names `sphinx-rtd-theme`, from the
refactor's Task 2.5; that was not a choice of theme, and the extra follows this
answer.*

**Response:** Furo sounds like a better choice for a modern look and dark mode. Let's choose `furo`.

**DQ.3 — The v1.0 manual.** Convert its chapters with `pandoc` and edit them, or
rewrite each chapter from WF and the code, using the tex chapters as a checklist.
Much of tex `description` and `plotting` is out of date.

My lean: rewrite from WF, keeping tex text where it is still right (functions,
generatedata, montecarlo, newfunctions), converted with `pandoc`.

*Note (2026-11-26): the `build-docs` skill already says the tex files are history and
the code is the authority, which this lean follows.*

**Response:** The original documentation should be used as a guide only (feel free to include whatever information is available), and the new documentation should be based on the workflow and code. No tex files or pdf will be made available for v2 and beyond. The tex files will be kept in the repository for reference only.

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

**DQ.6 — Where the site lives.** readthedocs under `alis` (if free) or another name,
built from `master`, with a version per release tag.

My lean: as described, once the v2 branch is merged into `master`.

*Note (2026-11-26): the name's availability is not checked here; creating the
readthedocs project, and linking it to the GitHub repository, are RJC's (DQ.8 asks
what it builds before the merge).*

**Response:** I agree with your lean. The project name on ReadTheDocs is called `ALIS-fitting`.

**DQ.7 — The screenshots in the repository.** Commit them under `doc/source/_static/`,
or make them in the readthedocs build (which needs Qt and a display there).

My lean: commit them, made by `screenshots.py` before each release; the build only
reads them.

*Note (2026-11-26): the examples' figures (chapter 11, and the Introduction's) raise
the same question; drawing them in the build would run 26 fits there. The lean covers
both: committed, each made by its script.*

**Response:** There;s no need to add screenshots for all the examples, and no examples should be made based on the context files. I agree that the screenshots should be committed and made by `screenshots.py` before each release.

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

**DQ.11 — The v1.0 manual's files.** `doc/tex_files/` (its sources and
`alis_v1p0.pdf`) and `doc/alis_userguide.pdf`, once the site covers them.
- **(a)** Kept as v1.0's manual, unedited from then on, and linked from the site's page
  on changes.
- **(b)** Removed from the repository (git keeps them), with the site's page on changes
  linking to v1.0's tag.

My lean: (a). The PDF is what users of v1 know, and keeping it costs nothing.

**Response:** There should be no mention of the v1.0 manual in the new documentation. The v1.0 manual will be kept in the repository for reference only, but it will not be updated or edited anymore. The site will serve as the primary source of all documentation for ALIS v2 and beyond.

**DQ.12 — The dashboard's link to the site.** Its Help menu has Keyboard shortcuts
(F1) and About ALIS, and no link to any documentation.
- **(a)** Help → ALIS documentation, which opens the site's page for the tab in
  view, in Documentation Stage 4 (a small change to the dashboard, with its test).
- **(b)** One link to the site's front page.
- **(c)** No link: the README and the site are enough.

My lean: (a). A user who is stuck on a tab reaches the page that explains it in one
step; the pages' addresses are fixed in Stage 4, with the dashboard chapter.

**Response:** I agree with your lean. The Help menu should have a link to the ALIS documentation, which opens the site's page for the tab in view. This will provide users with quick access to relevant documentation while using the dashboard.

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

## Prompts
1. Please read my responses to your queries, the ALIS code base, and the dashboard development (their design documents and logs). If anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please update this document to reflect any changes to the design document based on my responses. When you have a clear perspective on the documentation, please write a plan for the documentation development, including the stages, tasks, and checks, and any other relevant information. We will develop one stage at a time, so please make a plan for how many stages are required, and what each stage will accomplish.
