# ALIS Development Guide for Claude

## Project Overview

ALIS (Absorption LIne Software) is a Python package for model fitting of spectroscopic data.
The main package lives in the `alis/` directory. The entry point is `alis/alis.py`, which
contains `ClassMain`. Model fitting functions are defined in `alis/alfunc_*.py` files.
Atomic data are stored in `alis/data/atomic.xml`. Examples live in `examples/`.

## Git Operations

**The user (rcooke-ast) performs all git operations.** Claude must not run any
state-changing git commands, including but not limited to:

- `git commit`, `git add`, `git reset`, `git checkout`, `git merge`, `git rebase`
- `git push`, `git pull`, `git fetch`
- `git branch -d`, `git stash`, `git tag`

Read-only git commands (`git status`, `git log`, `git diff`, `git show`) are permitted
for context only.

## Context Management

- When context exceeds 50% of the maximum context length, Claude should suggest starting
a new conversation of using subagents for independent tasks.
- Proactively recommend context-saving strategies: use file reads instead of pasting,
suggest `/compact` when context is heavy, recommend subagents for research tasks, and
flag when a reference file would be better than inline instructions.

## Python Conventions

This is a Python project. Follow these conventions:

- **PEP 8** for all code style (4-space indentation, 79-char line limit, etc.)
- Match the style of the surrounding code in whichever file is being edited
- The codebase uses `from __future__ import absolute_import, division, print_function`
  at the top of modules — preserve this in existing files; new files targeting Python 3
  only do not need it
- Docstrings use triple double-quotes; inline comments use `#`
- Imports are ordered: standard library, then third-party (`numpy`, `astropy`), then
  local `alis` modules
- Messaging to the user goes through the `msgs` object (`almsgs.msgs()`) rather than
  bare `print` calls

## Branches

- `master` — current stable version; Claude has not made changes to this branch
- `alis_v2` — working branch for the new version of ALIS being developed with Claude