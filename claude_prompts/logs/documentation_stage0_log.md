# Documentation Stage 0 — log

*The plan's first draft (2026-10-31) and its update at dashboard Stage 14's close
(2026-11-26) are logged in `dashboard_stage7_log.md` (Task 7.14) and
`dashboard_stage14_log.md` (Prompt 11). This log starts with Prompt 1.*

## Prompt 1 (2026-11-26): RJC's responses applied, and the plan of the stages

**Read:**
- RJC's responses to DQ.1–DQ.14 (from `git diff` of `documentation_stage0.md`).
- `doc/ALIS_workflow.md`, all of it.
- The start of `claude_prompts/ALIS_v2_dashboard_prompts.md` (goals, F1–F14,
  S1–S31) and D87.
- `dashboard_stage14.md` (introduction and design) and its log's close.
- `dashboard_stage15.md`, the template `dashboard_stageX.md`, and `deferred_work.md`
  (the Priorities, §1.2 and §6).
- The code: `alis/logger.py` (`msgs`), `alis/config.py` (`settings_help`,
  `RENAMED_SETTINGS`, `REMOVED_SETTINGS`), `alis/scripts/run_alis.py`'s parser,
  `alis/dashboard/actions.py` (68 actions), the dialogs in `alis/dashboard/qt/`,
  `alis/prepfit/specplot.py`, `alis/functions/user.py`, and the places where the
  dashboard reads ALIS's messages (`model.py`, `results.py`).
- The repository: `pyproject.toml`, `.github/workflows/ci.yml`, `MANIFEST.in`, the
  README, the `build-docs` skill, `doc/installing.rst`, `doc/tex_files/` (its
  introduction and `to_be_completed.txt`), `tests/manifest.py`, and
  `context/publications_using_alis/` (how ALIS is cited).

**Found** (each now in "What exists" or a query):
- `doc/installing.rst` describes v1 (Python 2.7 or 3.3, `python setup.py
  develop`), not v2. The plan had called it v2's. The Installing page replaces it
  (DQ.1).
- WF §4.3 gives F10 for Keyboard shortcuts; `actions.py` has F1. The code is the
  authority, and the page will say F1.
- WF §10 calls `atomic.ecsv` a VOTable file. The VOTable form is the legacy
  `atomic.xml`.
- `prepfit` is in the package as well as in `examples/`: `alis/prepfit/specplot.py`
  (deprecated, untested). The dashboard's text names it in four places
  (`actions.py`: Load snips…'s and Adjust fit regions: one spectrum's tooltips;
  `qt/regions.py`: the Regions tab's hint; `loadfit.py`: Load snips…'s message for a
  file that names no transition). Nothing imports `alis.prepfit`; the tests use the
  name only in a helper, `prepfit_snip`. Raised as DQ.16.
- `msgs.error` and `msgs.warn` take the message alone. The dashboard reads ALIS's
  messages by their text (the `.mod` panel's problems, `model.py`; the run's
  warnings, `results.py`). Codes printed with the messages would change that text.
  Raised as DQ.15.
- Counts confirmed: 335 errors and 122 warnings in 40 modules (one warning is
  `prepfit`'s); 43 regression cases, 27 from 22 shipped examples (26 fits and
  `generate`); 23 dialog classes and the Column roles dialog.
- The tex introduction's citation is a placeholder, and its licence text is not the
  BSD 3-Clause of `LICENSE`. The papers in `context/publications_using_alis/` cite
  Cooke et al. 2014, ApJ, 781, 31 (`Coo14`). Raised as DQ.17.
- `MANIFEST.in` names a `docs/` tree, from the astropy template; Stage 1 points it at
  `doc/source`.
- Here, Sphinx 8.2.3 is installed; `myst-parser`, `furo` and `pandoc` are not
  (`pandoc` is no longer needed, DQ.3).

**Changed in `documentation_stage0.md`:**
- The header notes Prompt 1.
- "What exists" is corrected as above, with the v1.0 manual used as a checklist only.
- "The readers" assigns readers 1–3 to the User guide and reader 4 to the Developer
  guide.
- New section, "How the pages are written": the rules for succinct pages, from
  DQ.14's response.
- "The site's outline" is rewritten in two parts:
  - the User guide: Getting started, The dashboard, Fitting with ALIS, Reference,
    About;
  - the Developer guide.

  It ends with a list of what is not on the site. "Preparing data" is gone (DQ.9);
  Upgrading from ALIS v1 stays, without the manual (DQ.11).
- "The reference pages" groups the model functions on five pages by kind, and
  describes the codes of the messages (DQ.4, DQ.15).
- "The tools": furo, MyST, the extension `alisdoc.py`, the `alis` lexer, no
  `pandoc`, HTML alone, and `alis-fitting.readthedocs.io`.
- "The dashboard's screenshots": shipped examples and synthetic data alone (DQ.7).
- "The stages of the work" is replaced by "The plan of the work". It covers how each
  stage runs, and six stages, each with its tasks and checks:
  1. the skeleton;
  2. getting started and the dashboard;
  3. fitting with ALIS;
  4. the reference made from the code;
  5. errors, warnings and troubleshooting;
  6. examples, the Developer guide and the release.

  It ends with what comes after the documentation. The order changed: the dashboard
  now comes before the reference.
- Each response to DQ.1–DQ.14 has an "Applied" note. DQ.10's note gives how
  "penultimate" is read: the task just before the review.
- DQ.15–DQ.17 raised, each with a lean.

**Not changed yet:** D87 in `ALIS_v2_dashboard_prompts.md`, `deferred_work.md` §1.2
and `dashboard_stage15.md`'s introduction still say "Documentation Stages 1–5".
Stage 1's task 6 brings them into step once DQ.15 settles the number of stages.

No code was changed. Waiting for RJC's responses to DQ.15–DQ.17.

## Prompt 2 (2026-11-26): DQ.15–DQ.17 applied, and five stages

**Read:** RJC's responses to DQ.15–DQ.17 and Prompt 2. To make Stage 1's task
complete, every tracked mention of `prepfit` (`git grep -i prepfit`), with
`pyproject.toml`'s lint sections, `alis/data/` and `atomic_README`, and the
`msgs.error` and `msgs.warn` calls (to see how messages are built).

**The responses:**
- DQ.15: no codes for the errors and warnings.
- DQ.16 (a): `prepfit` removed in Documentation Stage 1.
- DQ.17 (a): cite Cooke et al. (2014), ApJ, 781, 31, until RJC prepares a citation of
  v2.

**Found** (corrections to Prompt 1, each now in the stage document):
- `prepfit` has a test: `tests/test_prepfit_deprecated.py` imports
  `alis.prepfit.specplot` in a subprocess. Prompt 1 said it was untested and that
  nothing imported it.
- Beyond the four texts a user sees (two tooltips in `actions.py`, the Adjust fit
  regions ▾ menu's tooltip in `qt/regions.py`, `loadfit.NO_NAME`), it is named in
  comments and docstrings in `lines.py`, `loadfit.py`, `modes.py`, `snips.py`,
  `qt/plots.py` and `qt/regions.py`. Most describe the snip's format as
  "`prepfit`'s four-column format".
- It is also named in:
  - the tests: the helper `prepfit_snip`, two tests named for it, and several
    docstrings;
  - `tests/README.md`, WF (§0 and five later places), and the `gui-dev` and
    `gui-component` skills;
  - `deferred_work.md` §1.1, §5, §7.3 and §7.8;
  - the CHANGELOG (a Deprecated entry and two mentions in Added).
- `alis/data/atomic.dat` is read by `prepfit` alone. Despite its name,
  `convert_datFormat_to_xmlFormat.py` holds its lines in the script and does not
  read it. Raised as DQ.19.
- Many messages are built from pieces (`"Keyword '" + name + "' is unknown for -"`),
  so a test of a quoted message matches each fixed piece, not the whole text.
- Without `prepfit`'s warning, there are 335 errors and 121 warnings.
- D87, `deferred_work.md` §1.2 and `dashboard_stage15.md` say "Documentation Stages
  1–5", which is right again with five stages. Stage 1's task 6 now updates only
  §1.2's note that the queries are open.

**Changed in `documentation_stage0.md`:**
- The header notes Prompt 2.
- "What exists":
  - a `prepfit` entry listing everything Stage 1 removes;
  - the `troubleshoot.tex` row, and the note on `to_be_completed.txt`: no lists of
    errors and warnings are made.
- The outline:
  - the Errors and warnings page is gone, so Atomic data, Upgrading from ALIS v1 and
    About ALIS are now 19–21;
  - Troubleshooting quotes the common messages, as ALIS prints them, with what to do;
  - About ALIS gives the 2014 citation and GitHub's issues;
  - Contributing drops the code for every new message;
  - "Not on the site" names the codes.
- "The reference pages": Errors and warnings now says how Troubleshooting picks its
  messages and keeps them in step (`alis-message` blocks, and a test against the
  calls).
- "The tools": the messages' directive is gone, and the `alis-message` lexer is new;
  `test_docs.py` checks the command line's options and Troubleshooting's quotes.
- The plan has five stages:
  - the stage of the codes is gone;
  - Troubleshooting joins Stage 3, with the WF sections it draws on;
  - the old Stage 6 is now Stage 5.
- Stage 1: About ALIS's citation and `CITATION.cff` in detail, task 5 lists every
  change for `prepfit`, and the checks are adjusted (`git grep -i prepfit`, the
  batches, `cffconvert`).
- Stage 2: the Fit page describes the pre-flight check's findings and Warnings and
  options to resolve…, the dashboard's suggestions that DQ.15's response relies on.
- Stage 3: Troubleshooting, and the test of its quotes.
- Stage 4: the Command line page's completeness test, and the Messages page if DQ.18
  is answered (b).
- The skills, the context, and the stage numbers in earlier Applied notes.
- Applied notes under DQ.4 (reverted), DQ.15, DQ.16 and DQ.17.
- DQ.18 (a page of every message, without codes) and DQ.19 (`atomic.dat`), each
  with a lean.

"Each stage design documents" in Prompt 2 is read as the stage sections of this
plan. `documentation_stage1.md` is written at RJC's prompt, as each dashboard stage's
document was.

No code was changed. Waiting for RJC's responses to DQ.18 and DQ.19.

## Prompt 2, run again (2026-11-26): DQ.18 and DQ.19 applied

**Read:**
- RJC's responses to DQ.18 and DQ.19. They were the document's only change: its
  line count was the same.
- `dashboard_stage15.md`'s structure, to place the item RJC asked for.
- `alis/dashboard/lines.py`'s `atomic()`, `actions.py`'s Help menu, and
  `atomic.ecsv`'s columns.

**The responses:**
- DQ.18 (a): no page lists every message.
- DQ.19 (a): `atomic.dat` is removed with `prepfit`. RJC also asked for a Help-menu
  item, Atomic data, in dashboard Stage 15: a dialog that lists every transition's
  atomic data, filtered by element, ion and wavelength range.

**Changed in `documentation_stage0.md`:**
- The header notes this run.
- Every passage that depended on DQ.18 (b) is gone: the outline's note, "The
  reference pages", the stages' rationale, and Stage 4's extra task.
- Stage 1, task 5 removes `atomic.dat`, and `atomic_README`'s first lines then name
  `atomic.ecsv` and `atomic.xml`. A new check: no file outside the history names
  `atomic.dat`.
- "After the documentation" notes Stage 15's item 17.
- There are Applied notes under DQ.18 and DQ.19, and the queries' preamble says that
  none is open.

**Changed in `dashboard_stage15.md`:**
- A new section, "Asked for by RJC", with item 17, Help → Atomic data. It gives what
  the dialog shows, and what it would take:
  - the logic in `alis/dashboard/`, reading the project's table through
    `lines.atomic(project.registry)`, with the filter;
  - a read-only dialog, outside the history and with nothing for the blinding gate
    to hide;
  - the action `help.atomic_data`, with its tests;
  - the pages it changes.
- A note under the introduction says the item is in the stage whatever Q15.1 and
  Q15.2 choose.

No query is open. The plan has five stages, and Documentation Stage 1's document
(`documentation_stage1.md`) is next, at RJC's prompt. No code was changed.

## Prompt 3 (2026-11-26): Documentation Stage 1's document

**Read:**
- the plan's Stage 1, and the rules of "How the pages are written" and "How each
  stage runs";
- the stage template (`dashboard_stageX.md`) and `dashboard_stage14.md`'s form;
- for the build: `pyproject.toml` (the `docs` extra, the lint lists, the version),
  `MANIFEST.in`, `.gitignore`, `pytest.ini`, `.github/workflows/ci.yml`,
  `.pre-commit-config.yaml`, `alis/__init__.py`, `LICENSE`, the README, and the
  `build-docs` skill;
- for the `alis` lexer: WF §2 and `examples/metal_line_abs/model/*.mod`;
- for `prepfit`'s removal: every line of `git grep -i prepfit`, WF §0, the two
  skills, `deferred_work.md` §1.1 and the Priorities, and `alis/data/`;
- for `CITATION.cff`: `context/publications_using_alis/deuterium_2014.tex` (the
  paper's authors and title), read only.

**Found:**
- PyPI has Sphinx 9.2.0, furo 2025.12.19, myst-parser 5.2.0 and cffconvert 2.0.0
  today. Sphinx 8.2.3 is installed here.
- `prepfit`'s lint entries are eight (black three, isort three, ruff two).
- `prepfit_snip` is used on 69 lines of four test files. `write_snip` and `cut_snip`
  are already the dashboard's names, so the helper becomes `earlier_snip`.
- The tests of Load snips…'s message match "names no transition", which the
  rewording keeps.
- `deferred_work.md` §1.1 is already marked "can be closed".
- The paper's DOI is not in the preprint. DQ1.3 asks RJC to confirm it.

**Written:** `claude_prompts/documentation_stage1.md`, which has:
- RJC's steps, before Task 1.1 and after Task 1.4;
- the design:
  - the build, file by file;
  - the 36 pages and their addresses, and what a page not yet written holds;
  - the front page and the sidebar;
  - About ALIS and `CITATION.cff`;
  - the checks (CI's `docs` job, and `tests/test_docs.py` with `PAGES` as the one
    list of pages);
  - the `build-docs` skill;
  - `prepfit`'s removal, with the four texts reworded, before and after;
- Tasks 1.1–1.8;
- the skills and the context;
- DQ1.1 (the pages' addresses), DQ1.2 (the sidebar) and DQ1.3 (About ALIS: "RJC",
  an ORCID, and the DOI), each with a lean.

The sidebar of DQ1.2 (a) departs from the plan's two captions. If RJC chooses (a),
`documentation_stage0.md`'s outline is brought into step at Stage 1's close.

No code was changed. Waiting for RJC's responses to DQ1.1–DQ1.3.
