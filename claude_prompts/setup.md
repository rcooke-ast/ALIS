# Setting up the Claude experience for ALIS software development

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

1. Please generate a CLAUDE.md file to be placed in the parent directory.
The file should indicate that I will perform all git operations,
and that Claude should not run any state-changing git commands. It should
also indicate that this is a Python project and that Claude should follow
standard Python conventions (PEP 8) and match the style of surrounding code.

2. Given your understanding of the current version of ALIS and the Context provided
above for the major update to the ALIS software, provide an additional list of
suggested improvements that could be made to ALIS. I will then decide if
these updates should be included in the new version of ALIS. Please provide these
suggestions as a separate list in the Queries section below.

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

## Queries

### Suggested Additional Improvements (Prompt 2)

The following suggestions are in addition to the changes already listed in the Context section above.
Please indicate which of these should be included in the new version of ALIS.

**Code quality and maintainability**

1. **Type annotations**: The codebase has no Python type hints. Adding them throughout (function
   signatures, return types, key data structures) would enable static analysis with `mypy`, improve
   IDE autocompletion, and make the code self-documenting without adding prose comments.

2. **Replace nested dictionaries with dataclasses**: The code relies heavily on nested dictionaries
   with string keys (`argflag`, `modpass`, `fdict`) that are difficult to inspect, refactor,
   and type-check. Python `dataclasses` (or `attrs`/`pydantic`) would provide structure,
   IDE autocompletion, and optional validation at the boundaries where data enter the system.

3. **Replace custom messaging with Python's `logging` module**: The `almsgs.msgs()` system is
   a custom print-based logger. Python's built-in `logging` module provides log levels, file
   handlers, formatters, and is straightforward for users to configure (e.g., suppressing output
   when ALIS is used as a library inside another script).

4. **Remove Python 2 compatibility cruft**: The codebase still carries `from __future__ import
   absolute_import, division, print_function` and `try: input = raw_input` stubs in many files.
   Since Python 2 is end-of-life, these can be removed, simplifying the code.

~~5. **Use `astropy.constants` for physical constants**: Values such as the speed of light
   (299792.458 km/s) are hardcoded in multiple places. Using `astropy.constants.c` ensures
   consistency, correct units handling, and is immediately recognisable to astronomers.~~

**Fitting and uncertainty**

~~6. **MCMC / Bayesian posterior sampling**: Offer optional integration with `emcee` or `dynesty`
   as an alternative (or follow-up) to chi-squared minimisation. This would give full posterior
   distributions and robust uncertainties — especially valuable for the complex, high-dimensional
   parameter spaces that ALIS routinely handles.~~

7. **Fit diagnostics and residual analysis**: Automated tools to evaluate fit quality beyond the
   total chi-squared — e.g., per-region reduced chi-squared, flagging of poorly-fit wavelength
   intervals, and a summary "fit quality report" printed after convergence. This complements
   the convergence checks already planned.

8. **Profile caching / memoization**: Model components whose parameters are held fixed between
   iterations (tied parameters, fixed components) could be cached and only recomputed when their
   parameters change. For models with many fixed components, this could substantially reduce
   per-iteration computation time.

**Infrastructure and packaging**

9. **Packaging modernisation (`pyproject.toml`)**: Replace the current `setup.py` with a
   `pyproject.toml` following PEP 517/518/621. This is now the standard for Python packaging
   and would make it straightforward to publish ALIS to PyPI and conda-forge, lowering the
   barrier to installation for new users.

10. **Continuous integration (CI)**: Add GitHub Actions workflows for automated test runs on
    every push and pull request, plus linting (e.g., `ruff` or `flake8`) and code-coverage
    reporting. This would catch regressions early and enforce code style automatically.

11. **Pre-commit configuration**: Add a `.pre-commit-config.yaml` (black, ruff/isort) so that
    code style is enforced automatically before commits, keeping the codebase consistent without
    manual review overhead.

12. **Semantic versioning and a CHANGELOG**: The current version is pinned at `0.1.dev0`.
    Establishing semantic versioning (MAJOR.MINOR.PATCH) and maintaining a `CHANGELOG.md` would
    allow users to understand what changed between releases and plan upgrades.

**Data and I/O**

13. **Model parameter file modernisation**: Consider replacing the current custom text-based
    model file format with YAML or TOML. These formats are easier to parse, validate, generate
    programmatically, and edit with standard tools, while remaining human-readable.

~~14. **Memory-efficient handling of large spectra**: Large echelle datasets (hundreds of orders,
    millions of pixels) currently require all data to be loaded into memory at once. Lazy loading
    via `numpy.memmap` or `astropy.io.fits` memory-mapped I/O could reduce peak memory use for
    large datasets.~~

**Usability**

15. **CLI modernisation**: Replace the current argument parsing with `argparse` (or `click`/`typer`)
    to provide a self-documenting command-line interface with `--help` output, tab completion,
    and consistent option naming conventions.

16. **Contribution guide**: Add a `CONTRIBUTING.md` describing how to set up the development
    environment, run the test suite, and submit changes. This is especially important as the
    new version becomes more modular and easier to extend.


## Logs



## Prompts

1. Perform Step 1 under Tasks.

2. Perform Step 2 under Tasks.

3. I have included some of the additional Context suggestions you have offered
in the Context Section of this file. Given your understanding of the code base,
and the Context provided above for
the major update to the ALIS software, provide a list of suggested skills to 
add for Claude.  Provide the list in the Skills section above and include URLs
to their locations on GitHub. I have included two skills already as examples.

4. The Skills you have added are good choices. Can you suggest additional skills
that would be useful for the development of ALIS, including GUI construction,
GPU support, generation of unit tests? Are there any additional skills that
would be useful, given the Context provided above in the context section?
Please provide the list in the Skills section above and include URLs to their locations on GitHub.

5. Proceed to generate each of these skills.