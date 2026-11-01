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
> its list; Q7.13). "D*n*" and "S*n*" refer to `ALIS_v2_dashboard_prompts.md`.

## What exists

### The v1.0 manual (`doc/tex_files/`)

`alis_v1p0.tex` (with its PDF, `alis_v1p0.pdf`, and `doc/alis_userguide.pdf`) builds
the v1.0 manual from one file per chapter. Each chapter, what it holds, and what has
changed since v1.0:

| File | Chapter | Holds | Since v1.0 |
|---|---|---|---|
| `introduction.tex` | Introduction | What ALIS is; the licence | Unchanged in substance; v2's additions to name |
| `acknowledgements.tex` | (commented out) | — | To write, or leave out (DQ.5) |
| `install.tex` | Installing and Updating ALIS | Requirements; updating | Python 3.13+, `pip install` with extras (`gui`, `gui-test`, `dev`); `doc/installing.rst` is newer |
| `description.tex` | How ALIS works | `settings.alis`, `atomic.xml`, the `.mod` file | Settings are now `config.py`'s dataclasses, set in the `.mod` or with `--set`; atomic data in `atomic.ecsv`; bundles (`.model`) |
| `fitting.tex` | Your first fit (commented out; a stub) | "hello", "there" | Replaced by the worked example of `doc/ALIS_workflow.md` §4.1 |
| `plotting.tex` | The ALIS plotting environment | Plot settings; the interface | `plot` and `plotscript` settings; `-p 0`; the plot script written beside a fit |
| `examples.tex` | Example `.mod` files and fitting | First fit; a metal-poor DLA; α variation; isotope ratios; a quasar spectrum | `examples/` now has 26 examples, each run by the regression harness |
| `generatedata.tex` | Generate Fake Data | The `generate` settings | Unchanged in substance |
| `montecarlo.tex` | Monte Carlo Simulations | `random`; `systematics` | `sim` settings; restarts and convergence checks from the dashboard |
| `functions.tex` | Built-in functions | Afwhm, base, brokenpowerlaw, chebyshev, constant, gaussian, legendre, linear, polynomial, powerlaw, random, … | Several functions since (see the reference lists below); GPU kernels for voigt |
| `newfunctions.tex` | Writing your own function | The Base function; arbitrary and polynomial functions | `functions/base.Base`, `user.py`, and the `new-alfunc` skill's recipe |
| `troubleshoot.tex` | Troubleshooting | Lists of errors and warnings (to be completed); FAQ | Messages now go through `msgs`; the dashboard's pre-flight check explains many before a run |
| `comments.tex` | Comments/Questions/Additions? | Contact | GitHub issues |
| `contact.tex`, `bibliography.tex` | Contact; references | — | Kept |

`to_be_completed.txt` lists v1.0's unfinished parts: the grey boxes' width, a journal
reference (Section 1), Table 1, the atomic data note (Section 3.2), "Your first fit"
(Section 4), the examples (Section 5), the proton-to-electron mass ratio in voigt
(Section 8), the user function file (Section 9), and the lists of errors and warnings
(Sections 10.1, 10.2). Each is taken up in the stage that writes its chapter.

### Since v1.0

- **`doc/ALIS_workflow.md`** (1,216 lines): preparing fitting regions (`prepfit` and
  snips), data, the model file section by section (settings, data, fixing, tying and
  limiting, emission, absorption, zero level, variables, links), synthetic data,
  running ALIS, project bundles, and the dashboard as a user meets it (§4.3). It is
  the most complete and current description, and the main source for the new
  chapters.
- **`doc/installing.rst`**: installing v2 (reStructuredText already).
- **The docstrings**, in numpydoc form throughout `alis/` and `alis/dashboard/`.
- **`examples/`**: 26 example fits (CNabs to vsigma), each a `.mod` with its data and
  a reference output; `context/fitting_examples/` holds real fits, not shipped.
- **The dashboard's documents**: `claude_prompts/ALIS_v2_dashboard_prompts.md` (the
  decisions), the stage documents, and the review pages' screenshot scripts
  (`doc/dashboard/stage4–6/take_screenshots.py` and the others).
- **`CHANGELOG.md`** and `tests/README.md` (the regression harness).
- **No Sphinx configuration yet**: there is no `conf.py` or `.readthedocs.yaml`, so
  the `build-docs` skill has nothing to build until Documentation Stage 1.

## The readers

1. **A new user** who has a spectrum and wants a first fit: install, a tour of the
   dashboard, a worked example from spectrum to results. Needs no prior ALIS.
2. **A user of ALIS v1** with `.mod` files and snips: what changed, how to run an old
   fit from the command line or open it in the dashboard (Task 7.1's Open a fit),
   and the new settings.
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
   - *New, one page per tab:* Data, Regions, Components, Fit (Inspect, Results,
     Compare), with the `.mod` panel, blinding, the history, preferences and keys.
   - Each page gets screenshots made by script (below).
   - Opening an earlier fit (Open a fit (.mod)…, Load snips…).
5. **Preparing data** — WF §0–1 (`prepfit`, snips, fitting regions), tex
   `description`.
6. **The model file** — WF §2 and tex `description`.
   - Each block: settings, data, the model's emission, absorption and zero level,
     variables and links.
   - Fixing, tying and limiting.
   - *New:* bundles, and `run datadirc`.
7. **Running ALIS** — WF §4, the `run_alis` command line (`--set`, `--extract`,
   `--pack`), outputs (`.mod.out`, `_fit.dat`, the covariance, the report), GPUs and
   memory (WF §2.1).
8. **Plotting** — tex `plotting`; `plot` and `plotscript` settings; the plot script.
9. **Synthetic data** — tex `generatedata`, WF §3.
10. **Simulations and convergence** — tex `montecarlo`; the `sim` settings;
    restarts and convergence checks (the Fit tab's tools).
11. **Examples** — tex `examples` and `examples/`: one page per example, its purpose,
    its `.mod`, and its fit drawn.
12. **Reference: model functions** — tex `functions`, made from the code (below).
13. **Reference: settings** — made from `alis/config.py` (below).
14. **Writing your own function** — tex `newfunctions`, `functions/base.py`,
    `functions/user.py`; adding a GPU kernel (`voigt_gpu.py` as the model).
15. **Troubleshooting** — tex `troubleshoot`; the lists of errors and warnings, made
    from the `msgs.error` and `msgs.warn` calls (DQ.4); the pre-flight check's
    findings; an FAQ.
16. **For developers** — the tests (`tests/README.md`), the regression harness, the
    dashboard's code layout, and how the documentation is built.
17. **Changes, contact and references** — `CHANGELOG.md`, tex `comments`, `contact`,
    `bibliography`.

## The reference lists

Both lists are made from the code by a script (`doc/source/make_reference.py`,
Documentation Stage 3), so that they stay complete. The pages it writes are checked
in CI against the code: a function or a setting the code has and the pages do not
fails the build. The lists below are this document's own, as of 2026-10-31.

### Model functions

Every function the registry builds (`alis.main.build_funcarray`), by module of
`alis/functions/`:

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
them.

### Settings

Every setting of `alis/config.py`, by section:

- **run** — atomic, backend, bintype, blind, cache, capvalue, convcriteria,
  convergence, convnostop, datadirc, datatype, gputhresh, last_update, limpar, logn,
  modname, ncpus, ngpus, nsubmax, nsubmin, nsubpix, renew_subpix, shmem,
  warn_subpix.
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
metadata, so they cannot drift.

## The tools

- **Sphinx**, with sources in Markdown through MyST (DQ.1). `doc/ALIS_workflow.md` and
  the new chapters are Markdown already. `doc/installing.rst` stays
  reStructuredText, which Sphinx reads beside it.
- **The theme:** `sphinx-rtd-theme` or `furo` (DQ.2).
- **The API pages:** `sphinx.ext.autodoc` with `numpydoc` (or `napoleon`), for the
  developer chapter alone. The user chapters do not depend on docstrings.
- **The v1.0 chapters:** converted from LaTeX by `pandoc`, then edited by hand
  (DQ.3). Equations stay as LaTeX math (MathJax).
- **readthedocs:** `.readthedocs.yaml`, `doc/source/conf.py`, and a `docs` extra in
  `pyproject.toml`. The site builds from `master` once merged (DQ.6).
- **How the build is checked:**
  - `sphinx-build -W -b html doc/source doc/_build/html`: no warning;
  - `sphinx-build -b linkcheck`: no broken link;
  - the reference lists against the code (above);
  - each code block of a `.mod` that the pages show is parsed by ALIS in a test.

  The `build-docs` skill runs the first two. A CI job runs all four.

## The dashboard's screenshots

The review pages' scripts already drive the window off-screen at 1440×900 and save
each tab, on real fits (`doc/dashboard/stage4–6/take_screenshots.py`,
`take_close_screenshots.py`). Documentation Stage 4 gathers them into one script,
`doc/source/screenshots.py`, which writes every screenshot the pages use. The
screenshots are made again before each release, so that they show the dashboard as
it is. They are written into the documentation's folder, not committed twice (DQ.7).

## The stages of the work

Each stage gets its own document, written when the stage before it is done, as the
dashboard's stages were. Each ends with RJC's review of the pages it built.

| Stage | Builds | Check |
|---|---|---|
| Documentation 1 | The skeleton: `conf.py`, `.readthedocs.yaml`, the `docs` extra, the outline's 17 pages (titles and sources), the build in CI | `build-docs` builds with no warning; readthedocs builds the branch |
| Documentation 2 | The user chapters from the existing text: Introduction, Installing, Preparing data, The model file, Running ALIS, Plotting, Synthetic data, Simulations (from WF and the tex chapters) | Every tex chapter and WF section has its place; every `.mod` block shown parses; `to_be_completed.txt`'s items for these chapters are done |
| Documentation 3 | The references: `make_reference.py`, the function and settings pages, the lists of errors and warnings | The pages list every function, setting and message the code has (a test), and the build fails when one is missing |
| Documentation 4 | The dashboard chapter and Your first fit, with `screenshots.py` | Every tab, sub-tab and dialog has its page and screenshot; the screenshots are made by the script alone |
| Documentation 5 | Examples, Writing your own function, Troubleshooting, For developers; the review and the release | Every example has its page; linkcheck passes; RJC's review applied; readthedocs serves `master` |

## Skills to use

- `build-docs`: each stage's build.
- `run-example`: the examples' pages and their figures.
- `gui-dev`: the dashboard's screenshots.
- `new-alfunc`: checking the Writing your own function chapter against its recipe.
- `atomic-data`: the atomic data's section.

## Context

- `doc/tex_files/` (the v1.0 manual) and `to_be_completed.txt`.
- `doc/ALIS_workflow.md`, `doc/installing.rst`, `CHANGELOG.md`, `tests/README.md`.
- `alis/config.py` (the settings and their metadata) and `alis/functions/`.
- `claude_prompts/ALIS_v2_dashboard_prompts.md` and the dashboard's stage documents.
- `doc/dashboard/` (the screenshot scripts of the review pages).

## Queries

*Raised by Claude on 2026-10-31 (Task 7.14 of `dashboard_stage7.md`). Each gives
Claude's lean.*

**DQ.1 — The source format.** Markdown (through MyST) or reStructuredText for the
pages. Markdown is what `doc/ALIS_workflow.md` is written in and is easier to edit;
reStructuredText is Sphinx's own, with richer cross-references.

My lean: Markdown (MyST), with `doc/installing.rst` kept as it is.

**DQ.2 — The theme.** `sphinx-rtd-theme` (readthedocs' familiar look) or `furo` (a
cleaner, more modern one, with a dark mode).

My lean: `furo`.

**DQ.3 — The v1.0 manual.** Convert its chapters with `pandoc` and edit them, or
rewrite each chapter from WF and the code, using the tex chapters as a checklist.
Much of tex `description` and `plotting` is out of date.

My lean: rewrite from WF, keeping tex text where it is still right (functions,
generatedata, montecarlo, newfunctions), converted with `pandoc`.

**DQ.4 — The lists of errors and warnings.** Made from the code (the `msgs.error` and
`msgs.warn` calls, grouped by module, each with a hand-written explanation), or
written by hand for the common ones only.

My lean: made from the code, so that none is missed, with explanations written for
the commonest first.

**DQ.5 — The acknowledgements.** v1.0 has the chapter commented out.

My lean: a short page, with the funding and the people you name.

**DQ.6 — Where the site lives.** readthedocs under `alis` (if free) or another name,
built from `master`, with a version per release tag.

My lean: as described, once the v2 branch is merged into `master`.

**DQ.7 — The screenshots in the repository.** Commit them under `doc/source/_static/`,
or make them in the readthedocs build (which needs Qt and a display there).

My lean: commit them, made by `screenshots.py` before each release; the build only
reads them.

## Prompts

