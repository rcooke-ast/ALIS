# Context for ALIS software development

ALIS is a versatile software package that allows you to perform a model
fit to spectroscopic data. We are using Claude to assist with development
of the ALIS software package. This document outlines the setup for using
Claude in this context.

Note that we will refer to the "current version" of ALIS as the version of the
code that is currently in the main branch of the ALIS repository, which Claude
has never made changes to. The "new version" of ALIS is the version that the
Claude and I will be working on together, which will include new features,
code restructuring, unit tests, performance improvements, and documentation.
The "working version" of ALIS is the version that is currently being worked on,
which will become the new version of ALIS once all changes have been implemented
and fully tested.

## Tasks

For all tasks, if Claude has any queries, the queries will be written in the Queries
section below, and I will provide responses to them. Queries and their responses must
be reviewed by Claude before a task will be performed. This is to ensure that Claude
has a full understanding of the context needed to perform a task.

Before executing any task, Claude should decide if there are skills that would be useful
for the task, based on the skills listed in the Skills section below. If there are skills
that would be useful, Claude should generate new skills, and then use those skills to
perform the task. If there are no skills that would be useful, Claude can perform the task
without using existing or generating new skills.

1. Given your knowledge of the code base, please review the `examples/` folder to see simple example
model files that can be used as input to ALIS. Then, review the example model files provided in
the `context/fitting_examples/VMP_DLA/J0903p2628/model/` folder. Once you understand the purpose
of this one simple context example, generate a `context.txt` file in the `J0903p2628/` folder
that describes the purpose of this example, providing details that are relevant to the context
required by Claude. If there are missing details that are needed to understand the purpose of
this example, please provide a list of points that need clarification in the `context.txt` file,
and I will update it. This real world example will be used as context for Claude to understand the purpose of ALIS,
and to provide examples of how ALIS is used, that may help during the development of the new version
of ALIS. The `context.txt` file should be written in a way that is understandable to someone who has
never used ALIS before, and should provide a description of the purpose of the example, and how it
is used to perform a model fit to spectroscopic data.

2. Review the `context.md` file generated in Task 1. Given the knowledge you have acquired so
for about the code base, please now write `context.md` files for the example model files in the
`context/fitting_examples/VMP_DLA/J0814p5029/model/`
and
`context/fitting_examples/VMP_DLA/J0035m0918/model/`
directories. If you have queries about these model fitting examples, please write them in a
queries section at the bottom of the `context.md` file, and I will provide responses to them.
The `context.md` files should be written in a way that is understandable to someone who has never
used ALIS before, and should provide a description of the purpose of the example, and how it
is used to perform a model fit to spectroscopic data.

3. Review the `context.md` files generated so far for the `fitting_examples`. Given the knowledge
you have acquired so far about the code base, please now write `context.md` files for the example model files in the
`context/fitting_examples/helium34/*/model/`
directory. If you have queries about these model fitting examples, please write them in a
queries section at the bottom of the `context.md` file, and I will provide responses to them.
The `context.md` files should be written in a way that is understandable to someone who has never
used ALIS before, and should provide a description of the purpose of the example, and how it
is used to perform a model fit to spectroscopic data. For further context, I have added two
latex files including the information about the publications associated with the helium34
example model fits. The latex files are located in the `context/publications_using_alis/` directory,
and are named `helium3_2022.tex` and `helium3_2026.tex`. The latex files contain the relevant information
about the publications, and can be used to provide context for the helium34 example model fits.

4. Review the `context.md` files generated so far for the `fitting_examples`. Given the knowledge
you have acquired so far about the code base, please now write one `context.md` files for the example model files in the
`context/fitting_examples/Temperature/*/model/`
directory. Note, this directory contains multiple different models, but they all work the same
way in principle, so we only require one `context.md` file to describe all models. If you have
queries about these model fitting examples, please write them in a
queries section at the bottom of the `context.md` file, and I will provide responses to them.
The `context.md` files should be written in a way that is understandable to someone who has never
used ALIS before, and should provide a description of the purpose of the example, and how it
is used to perform a model fit to spectroscopic data. There are no publications associated with these
example model fits.

5. Review the `context.md` files generated so far for the `fitting_examples`. Given the knowledge
you have acquired so far about the code base, please now write one `context.md` files for the example model files in the
`context/fitting_examples/DH/*/model/`
directory. Note, this directory contains multiple different models, but they all work the same
way in principle, so we only require one `context.md` file to describe all models. If you have
queries about these model fitting examples, please write them in a
queries section at the bottom of the `context.md` file, and I will provide responses to them.
The `context.md` files should be written in a way that is understandable to someone who has never
used ALIS before, and should provide a description of the purpose of the example, and how it
is used to perform a model fit to spectroscopic data. For further context, I have added several
latex files including the information about the publications associated with the helium34
example model fits. The latex files are located in the `context/publications_using_alis/` directory,
and are named `deuterium_*.tex`. The latex files contain the relevant information
about the publications, and can be used to provide context for the `DH` example model fits.

6. Review all `context.md` files generated so far for the `fitting_examples`, including the manual
updates I have now made to these files. Given the knowledge you have acquired so far about the
code base and analysis from the publications, please now write one `context.md` files for the
most advanced usage of ALIS, which is an example model file in the
`context/fitting_examples/DH_orders/*/model_orders/` directory.
This example model fit is an extension to the model fit Claude has already contextualised
in the `context/fitting_examples/DH/Q1243p307/model/` directory. In the `DH_orders` example
model fit, the same data are used as in the `Q1243p307` example model fit, but the data are
split into different orders, and each order is fitted separately. The results of the individual
order fits are being fit simultaneous to produce a final result. The advantage of this approach
is that the errors are determined from the individual order fits, and are therefore more robust
than the errors determined from the single fit to the combined spectrum
(as in the `context/fitting_examples/DH/Q1243p307/model/` directory). In particular,
there is no covariance between the pixels in the `DH_orders/` version of the model, but there
are correlated pixels in the `DH/` model. The disadvantage of the `DH_orders/` model is that
it is more computationally expensive, contains more parameters, is more likely to suffer from
the chi-squared minimization falling into local minima, and takes much longer to run. Eventually,
this model will benefit from GPU support, which will speed up the fitting procedure.
If you have queries about these model fitting examples, please write them in a queries section
at the bottom of the `context.md` file, and I will provide responses to them.

## Context

Claude and I will be working together to implement a major update to the
ALIS software package. The update will include new features, code restructuring,
unit tests, performance improvements, and documentation. Claude will assist with
code generation, refactoring, and documentation. The entire current functionality
of the ALIS software package will be preserved, and all changes will be fully tested.
Below contains a list of the major changes to be implemented in the update. Additional
context will be provided during the development process as needed.

**Context for the new version of ALIS**

*  **Examples**: When developing the new version of ALIS, Claude can request specific examples of certain functionality, which I will provide through simple example model fits with the current version of ALIS. I will also provide specific examples that are commonly used with ALIS, latex files of papers that have previously used ALIS.

**Code quality and maintainability**

*  **Circular import removal**: In the current version of ALIS, the file alis.py is the main code that gets called. However, during the chi-squared minimization, this same code gets called (a circular import). This will be avoided in the new version of ALIS.
 
* **Model instantiation**: The current version of ALIS contains model fitting functions that are provided as classes in files such as `alfunc_BLAH.py`. Currently, these classes are the individual model fitting functions that are used by ALIS, but this is not taking full advantage of instantiating a class once when the model is initially loaded, and then used for the remainder of the fitting procedure. In the new version of ALIS, each model fitting function will be initially loaded, and no subsequent loads will be required. The new version of ALIS will make optimal use of the model function classes.
 
* **Avoid passing around `self`/`slf`**: In the new version of ALIS, the ClassMain will be removed from `alis.py` because it forces the entire code to carry around the `self` instance (or, in some functions, this is called `slf`).

*  **Compatibility**: The new version of ALIS will support all of its current features.

*  **Type annotations**: The codebase has no Python type hints. Adding them throughout (function
   signatures, return types, key data structures) would enable static analysis with `mypy`, improve
   IDE autocompletion, and make the code self-documenting without adding prose comments.

*  **Replace nested dictionaries with dataclasses**: The code relies heavily on nested dictionaries
   with string keys (`argflag`, `modpass`, `fdict`) that are difficult to inspect, refactor,
   and type-check. Python `dataclasses` (or `attrs`/`pydantic`) would provide structure,
   IDE autocompletion, and optional validation at the boundaries where data enter the system.

*  **Replace custom messaging with Python's `logging` module**: The `almsgs.msgs()` system is
   a custom print-based logger. Python's built-in `logging` module provides log levels, file
   handlers, formatters, and is straightforward for users to configure (e.g., suppressing output
   when ALIS is used as a library inside another script).

*  **Remove Python 2 compatibility cruft**: The codebase still carries `from __future__ import
   absolute_import, division, print_function` and `try: input = raw_input` stubs in many files.
   Since Python 2 is end-of-life, these can be removed, simplifying the code.

**Fitting and uncertainty**

*  **Fit diagnostics and residual analysis**: Automated tools to evaluate fit quality beyond the
   total chi-squared — e.g., per-region reduced chi-squared, flagging of poorly-fit wavelength
   intervals, and a summary "fit quality report" printed after convergence. This complements
   the convergence checks already planned.

*  **Profile caching / memoization**: Model components whose parameters are held fixed between
   iterations (tied parameters, fixed components) could be cached and only recomputed when their
   parameters change. For models with many fixed components, this could substantially reduce
   per-iteration computation time.

*  **Chi-squared minimization**: The current version of the code uses chi-squared minimization, and often this involves a large number of model parameters (sometimes in the hundreds). There is always a concern that ALIS will converge to a result that still remembers the initial starting parameters. Claude and I will brainstorm ideas that could potentially be implemented to ensure proper convergence and checks that models do not remember the initial parameter values.

**Infrastructure and packaging**

*  **GPU support**: Currently all models run on the CPU only. The new version of ALIS will support multiprocessed GPU models.

*  **Making ALIS more Modular**: The new version of ALIS will make it easier to write new functions. To achieve this, ALIS will need to be made more modular, so that it is easier to write unit tests, and allow for both CPU and GPU functionality.

*  **Unit tests and examples**: The new version of ALIS will contain unit tests, and a larger suite of examples that will be used to check that no bugs are introduced into the code as part of ongoing development. 

*  **Improved GUI to prepare fit and run/inspect fitting**: The ALIS repository There is a prepfit code that is designed to prepare a fitting procedure outside of ALIS. The new version of ALIS will contain a GUI that allows the end user to both prepare and run a fit, and iterate this procedure from within a single GUI.

*  **Packaging modernisation (`pyproject.toml`)**: Replace the current `setup.py` with a
   `pyproject.toml` following PEP 517/518/621. This is now the standard for Python packaging
   and would make it straightforward to publish ALIS to PyPI and conda-forge, lowering the
   barrier to installation for new users.

*  **Continuous integration (CI)**: Add GitHub Actions workflows for automated test runs on
    every push and pull request, plus linting (e.g., `ruff` or `flake8`) and code-coverage
    reporting. This would catch regressions early and enforce code style automatically.

*  **Pre-commit configuration**: Add a `.pre-commit-config.yaml` (black, ruff/isort) so that
    code style is enforced automatically before commits, keeping the codebase consistent without
    manual review overhead.

*  **Semantic versioning and a CHANGELOG**: The current version is pinned at `0.1.dev0`.
    Establishing semantic versioning (MAJOR.MINOR.PATCH) and maintaining a `CHANGELOG.md` would
    allow users to understand what changed between releases and plan upgrades.

**Data and I/O**

*  **Model parameter file modernisation**: Add new functionality to support a model file format
    with YAML or TOML, in addition to supporting the current text based options. The YAML and TOML
    formats are easier to parse, validate, generate programmatically, and edit with standard tools,
    while remaining human-readable.

*  **Atomic data modernisation**: The current atomic data are stored in the `atomic.xml` file. Claude and I will brainstorm if there are better alternatives to the xml file. One problem with the current version of ALIS is that the `nrows` parameter needs to be updated manually when each new piece of atomic data are added.

*  **Plotting script outputs**: The new version of ALIS will provide the option to output python plotting scripts to make publication quality plots with matplotlib.

**Usability**

*  **CLI modernisation**: Replace the current argument parsing with `argparse` (or `click`/`typer`)
    to provide a self-documenting command-line interface with `--help` output, tab completion,
    and consistent option naming conventions.

*  **Contribution guide**: Add a `CONTRIBUTING.md` describing how to set up the development
    environment, run the test suite, and submit changes. This is especially important as the
    new version becomes more modular and easier to extend.

* **Documentation**: The new version of ALIS will have more complete documentation, and will be provided on readthedocs, rather than compiled with a latex into a pdf. The latex files of the current version of the documentation is located in the `docs/tex_files/` directory. The content in these latex files is out of date with respect to the code. When generating new documentation, Claude should use the current version of the code as the source of truth, and not the latex files. The latex files are only provided for reference, and to provide context for the current version of the documentation. The new version of ALIS will have a more complete set of examples, and will include a tutorial that walks the user through the process of using ALIS to perform a model fit to spectroscopic data.

* **Type annotations**: The codebase has no Python type hints. Adding them throughout (function
   signatures, return types, key data structures) would enable static analysis with `mypy`, improve
   IDE autocompletion, and make the code self-documenting without adding prose comments.


## Skills

Suggested [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills)
to add for ALIS development. All skills should be prepared according to the structure:
`.claude/skills/<name>/SKILL.md`.

### Already present (examples)

| Skill | Description | Location |
|-------|-------------|----------|
| `critical-partner` | Constructive disagreement — surfaces untested assumptions and argues the strongest opposing case | [.claude/skills/critical-partner/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/critical-partner/SKILL.md) |
| `grill-me` | Relentless design interview — walks every branch of the decision tree, one question at a time | [.claude/skills/grill-me/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/grill-me/SKILL.md) |

### Suggested additions

**ALIS-specific skills (to be created)**

| Skill | Description | Location |
|-------|-------------|----------|
| `run-tests` | Run the ALIS pytest suite, optionally filtering by module or test name; reports pass/fail counts and tracebacks for any failures | [.claude/skills/run-tests/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/run-tests/SKILL.md) |
| `run-example` | Run a named example from `examples/` with the current version of ALIS and verify it converges to the expected result | [.claude/skills/run-example/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/run-example/SKILL.md) |
| `new-alfunc` | Scaffold a new model-function module (`alfunc_<name>.py`) from the `alfunc_base` interface, filling in all required methods and docstrings | [.claude/skills/new-alfunc/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/new-alfunc/SKILL.md) |
| `build-docs` | Build the Sphinx documentation locally (ReadTheDocs target) and report any warnings or broken cross-references | [.claude/skills/build-docs/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/build-docs/SKILL.md) |
| `check-fit` | Parse ALIS output files (`.mod.out`) and summarise fit quality: per-region reduced chi-squared, free-parameter count, and any convergence warnings | [.claude/skills/check-fit/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/check-fit/SKILL.md) |
| `profile-fit` | Profile an ALIS fit with `cProfile` / `line_profiler`, identify the top bottlenecks, and suggest optimisation targets ahead of GPU porting | [.claude/skills/profile-fit/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/profile-fit/SKILL.md) |

**GUI development**

| Skill | Description | Location |
|-------|-------------|----------|
| `gui-dev` | Launch and exercise the ALIS prepfit GUI, verify that a specific GUI interaction works correctly, and report any errors or visual regressions | [.claude/skills/gui-dev/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/gui-dev/SKILL.md) |
| `gui-component` | Scaffold a new GUI widget or panel for the ALIS prepfit/fitting GUI, following the existing design patterns in `alis/prepfit/` | [.claude/skills/gui-component/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/gui-component/SKILL.md) |

**GPU support**

| Skill | Description | Location |
|-------|-------------|----------|
| `port-to-gpu` | Port an existing `call_CPU` model function to `call_GPU` using CuPy or numba CUDA, following the `alfunc_base.Base` interface; verifies numerical equivalence against the CPU version | [.claude/skills/port-to-gpu/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/port-to-gpu/SKILL.md) |
| `gpu-benchmark` | Benchmark CPU vs GPU throughput for a given ALIS model function or full fit, and report the speedup | [.claude/skills/gpu-benchmark/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/gpu-benchmark/SKILL.md) |

**Unit test generation**

| Skill | Description | Location |
|-------|-------------|----------|
| `gen-tests` | Generate comprehensive pytest unit tests for a specified ALIS module or function, covering normal operation, boundary conditions, and known edge cases | [.claude/skills/gen-tests/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/gen-tests/SKILL.md) |
| `test-coverage` | Run pytest with coverage, identify untested code paths in a given module, and suggest the highest-value tests to add next | [.claude/skills/test-coverage/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/test-coverage/SKILL.md) |

**Convergence and atomic data**

| Skill | Description | Location |
|-------|-------------|----------|
| `convergence-check` | Re-run an ALIS fit from multiple randomised starting points and report whether all runs converge to the same solution, flagging parameter degeneracies | [.claude/skills/convergence-check/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/convergence-check/SKILL.md) |
| `atomic-data` | Add, validate, or convert atomic data entries for the ALIS atomic database, checking for duplicates and verifying units | [.claude/skills/atomic-data/SKILL.md](https://github.com/rcooke-ast/ALIS/blob/alis_v2/.claude/skills/atomic-data/SKILL.md) |

## Prompts

1. Perform Step 1 under Tasks.

2. Review the `context.txt` file generated in Task 1, and convert it to Markdown format.
Ensure that the Markdown file is well-structured, with appropriate headings, subheadings,
and formatting to enhance readability. Include any relevant code snippets or examples that
illustrate the purpose of the example model files in the `J0903p2628/` folder. I have
responded to the queries in the `context.txt` file, and you should incorporate those
responses into the Markdown document. There is no need to include the queries and
responses in the Markdown file, but ensure that all relevant information from the
queries and responses is included in the final Markdown document.

3. Perform Step 2 under Tasks.

4. I have responded to the queries in the `context.md` files generated in Task 2, and you
should incorporate those responses into the Markdown documents. There is no need to include
the queries and responses in the Markdown files, but ensure that all relevant information
from the queries and responses is included in the final Markdown documents.

5. Perform Step 3 under Tasks.

6. While I respond to your queries from Task 3, please execute Task 4.

7. I have updated the `context.md` files (that were generated in Task 3 and 4) with some
corrections, and responded to the queries in the `context.md` files generated in Task 3 and Task 4, and you
should incorporate those responses into the Markdown documents. There is no need to include
the queries and responses in the Markdown files, but ensure that all relevant information
from the queries and responses is included in the final Markdown documents.

8. Please execute step 5 under Tasks. These examples provide important context for the
development of the new version of ALIS. Most of the funtionality you have now seen before,
but make sure that these features are properly understood, so that they can be preserved
in the new version of ALIS.

9. I have responded to the queries in the `context.md` files generated in Task 5, and you
should incorporate those responses into the Markdown documents. There is no need to include
the queries and responses in the Markdown files, but ensure that all relevant information
from the queries and responses is included in the final Markdown documents. In the meantime,
I have also included new data in the `context/fitting_examples/DH/` directory, which you
should use to update the `context.md` files. This includes the D I absorber with the current
highest precision (`J1419p0829`). This is important in the context of measuring D/H. Please
review the `context.md` file, and the new data, and update the `context.md` with the appropriate
changes. If you have any new queries, please respond in the same way as before.

10. Please execute step 6 under Tasks.
