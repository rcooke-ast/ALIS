# Stage 6 log -- Usability, CLI, and lint

Stage 6 as executed is **6.1 -> 6.5 -> 6.7 -> 6.4**. 6.2 (GUI) and 6.6
(interactive plot-script generation) went to a separate GUI design document
(Q6.2, Q6.4); 6.3 (Sphinx/ReadTheDocs) is deferred until v2.0.0 is ready to
release (Q6.3); 6.7 is new, from Q6.5.

Everything knowingly left unfinished -- by this stage and by the whole refactor
-- is now in `claude_prompts/deferred_work.md` (Q6.17/Q6.23).

## Task 6.1 -- CLI modernisation [COMPLETE]

The premise the doc was written on had been overtaken: `run_alis` already used
`argparse`. What was still hand-rolled was the other half of the path, and that
is what this task fixed.

**1. The model filename came from `sys.argv[-1]`, not from the parser.**
Fixed 2026-08-04 (Prompt 2): `args.alis_modfile`. Verified end to end --
`run_alis fit_spectra.mod -f -w -p 0` now fits (chi-squared 357.874162, matching
the reference) where before it reported "The filename does not exist" **and
exited 0**, so the failure was invisible to any caller checking the return code.
That is why a subprocess-based harness never caught it. `tests/test_cli.py` is
new; restoring the old line fails 4 of its tests.

**2. `msgs.error` now exits 1** (Q6.11). Same change applied to the five
`sys.exit()` calls that follow a `msgs.bug` or an error `print`
(`functions/base.py` x2, `legendre.py`, `chebyshev.py`, `prepfit/specplot.py`).
Deliberately left at 0, because they are successful or user-requested exits:
`logger.signal_handler` (Ctrl+C), the two onefits menu exits after printing or
extracting the model, and the GUI's own quit.

**3. The command line now beats the model file** (Q6.12). Settings arrive in
three passes -- defaults, command line, then the model file's `par` block -- so
the *file* had the last word and an explicit flag was silently discarded.
Measured: 44 of the 48 shipped examples set `plot dims`, so `run_alis -p 0` did
not suppress plotting; what made the harness headless was `MPLBACKEND=Agg`, not
the `-p 0` its own comment credited. `load.reapply_cli_overrides` re-applies the
command line after `load_input`.

*This surfaced a second dead feature.* With `-p 0` finally taking effect, the
fit aborted at the plotting stage with "Panel plot dimensions passed
incorrectly" -- `plot dims 0` has been advertised by `--help` and passed by the
harness for years, and nothing ever implemented it. `plot.make_plots_all` now
returns quietly on `0`.

**4. The command line is recorded in the `.mod.out`** (Q6.12/Q6.20/Q6.25), so
`run_alis model.mod.out` reproduces the run that produced it. The settings are
written **live**, not commented:

```
# --- applied from the command line ---
plot dims 0   #[cli] was 3x3
out fits True   #[cli] was False
run backend cpu   #[cli] was auto
# The following were also given on the command line, but describe
# that run rather than this model, so they are recorded only:
#   out overwrite True    (was False)
# --- end of command-line overrides ---
```

Only settings that describe the *model* are made live. `plot only` would stop
the re-run fitting at all, `out modelname` would make it clobber the original
output, `out overwrite` would make it overwrite silently, and the `sim`
counters would make it redo the whole simulation set -- those are recorded as
comments instead. Machine-specific choices (`ncpus`, `ngpus`) are not persisted
either. The split lives in `load.CLI_SETTING_MAP` as a third field per flag.

**The marker has to be on the line, not around the block.** The first attempt
delimited the block with comment lines and stripped between them; that silently
did nothing, because `load_input` drops comment-only lines when it reads a file,
so by the time the writer sees `_parlines` the delimiters are gone. Caught by
running three generations of `.mod.out` and finding the block duplicated. Each
live setting now carries `#[cli]` as a trailing comment, which survives into
`_parlines` because the whole raw line is kept, and which `set_params` ignores
because it reads the third whitespace token.

**Previous overrides are carried forward, not dropped** (Q6.25). Re-fitting a
`.mod.out` with no flags would otherwise produce a file that no longer
reproduces the run it describes -- generation 2 *ran* with the inherited
settings but recorded none of them. This run's overrides replace the earlier
ones key by key. Verified over three generations: gen 1 records three
overrides, gen 2 carries all three, gen 3 (`-p 1x1`) carries two and replaces
`plot dims`.

`set_params` no longer warns about a duplicate when the later line carries
`#[cli]` -- that block deliberately shadows the model's own setting, which is
not the accident the warning exists to catch.

**5. `--set 'section key value'`** (Q6.7), repeatable, accepting any setting, so
the CLI cannot fall behind the config as it did through Stages 4 and 5. It
routes through `load.set_params`, and an unknown section or key is an error that
points at `--list-settings`.

**6. `--list-settings`** (Q6.24) prints every section, key and default from the
`ArgFlag` dataclass and exits. It replaces the deleted `settings.alis` as the
place to find out what can be set, works from an installed wheel, and cannot go
stale.

**7. Flag rationalisation** (Q6.1 revised, Q6.19). `-g/--gpu` was a
`store_true`, so `run ngpus` became the boolean `True` where an int is declared;
it is now `-g/--ngpus N`. The count is **required**: with `nargs="?"` a bare
`-g` swallows the model file ("invalid int value: 'fit.mod'"), because argparse
cannot tell an optional value from the positional. `-x` and `-v` gained
`choices`. `--help` now takes its version from `alis.__version__` (the banner
said "v1.0" against `2.0.0.dev0`) and colours the title only when stdout is a
terminal, so the ANSI escapes no longer appear literally in a pipe.

**8. `optarg` tidied.** It located `data/settings.alis` by splitting a path on
`'/'` -- the anti-pattern Stage 5.2 replaced in `load_atomic` -- and carried a
dead Python-2 `getopt` block (`except getopt.GetoptError, err:`) as a string
literal. Both gone. The Namespace -> `argflag` copying is now a table
(`CLI_SETTING_MAP`) rather than fourteen `if`s. The unreachable `else` branch of
`run_alis.main` (guarded by a hard-coded `debug = True`) went with it.

## Task 6.1b -- one source of truth for the defaults [COMPLETE]

`alis/data/settings.alis` is **deleted** (Q6.21). It restated 55 of the 86
settings and was read *over* the `ArgFlag` dataclass on every run, so the two
could drift -- and had, on four settings, two of which change fits:

| setting | dataclass was | file said | now |
|---|---|---|---|
| `chisq fstep` | 1.0 | **20.0** | 20.0 |
| `chisq maxiter` | 20000 | **2000** | 2000 |
| `run ngpus` | `None` | 0 | 0 |
| `generate skyfrac` | 0.0 | **0.1** | 0.1 |

The file won, so the dataclass defaults were documentation that lied. The
dataclass adopted the file's values first (behaviour-neutral, since the file was
already winning), and the file was then removed. RJC confirmed `fstep = 20.0` is
deliberate. `load_settings` still accepts a path, so a user's own settings file
keeps working; it is now just an ordinary set of `par` lines.

## Task 6.5 -- lint, and the real bugs behind it [COMPLETE]

Taken as Q6.22 **option (b)**: the 39 legacy modules were removed from ruff's
`extend-exclude`, so the linter guards them from here on, while the black/isort
reformat -- 43% of 19,867 lines -- stays a separate decision, tracked in
`deferred_work.md`.

**The rule set is now pinned explicitly**, `select = ["E4","E7","E9","F"]`.
Ruff's *default* changed between the v0.6.9 the pre-commit hook installs and
current releases (0.16 adds UP/C4/SIM/PLR and reports ~1,200 more findings), so
leaving it implicit made the gate depend on which ruff happened to be installed.

**All 36 F821 undefined names are gone.**

- **SuperMongo deleted** (Q6.6/Q6.8/Q6.9), which was 22 of the 36 and ~200
  lines: `prep_arrs` (`plot.py`), `save_smfiles` (`save.py`), the `out sm`
  branch in `main.py` that raised "not implemented yet" *after* the fit had
  finished, two checks in `load.py`, `OutConfig.sm`, and the settings line. No
  `.mod` file, test or doc referenced it.
- **`imp.load_source` replaced with `importlib`** (Q6.17). The `systmodule=`
  hook had been dead since Python 3.12 removed `imp`; the `NameError` was
  swallowed by a bare `except` and reported as "Could not import module
  <theirs>", so the user was told their own file was at fault. The `except` is
  narrow now, so a module that exists but raises reports its own error.
- **Wavelength-dependent resolution refused rather than crashed** (Q6.16 option
  c). The branch in `afwhm`/`vfwhm`/`voigtconv`/`vsigma` referred to an
  undefined `szflx` and raised `NameError` on entry, so it had never run. RJC
  chose to support a single value honestly rather than enable untested numerics.
- **`convergence.py`**: `ans = nput(...)` was a typo for `input(...)`, in the
  live "file exists, overwrite?" prompt.
- **`chebyshev.py`**: `sys.exit()` with no `import sys`, reachable at order >= 10.
- **`lsfspline.py`**: an undefined `sidlist` in the `specid` keyword branch, now
  the comma-split value as in every sibling function.
- **`constant.py` / `linear.py`**: the last dead PyCUDA `SourceModule`
  scaffolding.

**All 7 E711 `== None` comparisons fixed.** Each operand was checked first: none
is a numpy array, so `is None` is safe (`arr == None` returns an array where
`arr is None` returns a bool, which is why the doc warned against a blind fix).

**Two more real defects, found by the rules the un-exclusion switched on:**
- `minimise.py` -- `__str__` of the fit result listed `'params'` twice in one
  dict literal, so one intended key was silently missing (F601).
- `save.py` -- `"{0:s}:{0:s}".format(i, num)` formatted the column *name* twice
  and dropped the index, so every onefits column header read `wave:wave`
  instead of `wave:0` (F523).

**E722 is ignored for now, with the reason recorded** (Q6.15 option b). 63 bare
`except:` remain, 40 of them in `load.py`, and several are load-bearing rather
than lazy -- `load_fits` uses one to work out which of three file formats it
has, `load_ascii` to decide a column is absent. Narrowing them changes which
errors are swallowed, in code no test covers on the failure path. Documented
once in `pyproject.toml` rather than 63 times, and listed in `deferred_work.md`.

## Task 6.7 -- documentation catch-up [COMPLETE]

- **`doc/ALIS_workflow.md`**: `settings.alis` replaced by `--list-settings` and
  `--set`, `atomic.xml` -> `atomic.ecsv` throughout, `bufferpix` added to the
  load sequence, the `alload.*` names corrected to `load.*`, and the settings
  precedence stated (the command line wins).
- **`README.md`**: GitHub, licence (BSD 3-Clause, confirmed against `LICENSE`
  and `pyproject.toml`), Python version and CI badges. The **black** badge is
  deliberately absent until the reformat happens, or it would advertise
  something untrue.
- **`claude_prompts/deferred_work.md`**: new, and the main deliverable here --
  every feature deferred, removed or known-broken across Stages 0-6, with why,
  where the intent is recorded, and what finishing it involves.

## Task 6.4 -- unit tests [COMPLETE]

`pytest -m unit`: **717 -> 736 passed, 31 skipped**. New coverage:
`--set` (accepted, rejected on an unknown setting, rejected on a fragment),
`--list-settings`, `-g` taking a count, the persist/record split, the override
block (live vs commented, carried forward, stripped from `_parlines`), the
precedence rule, the `#[cli]` shadowing not warning, `import_user_module`
(works, missing, raises), and the refused per-pixel resolution.

## Gate

- `pytest -m unit`: **736 passed, 31 skipped**.
- `pytest -m "unit or fast"`: **791 passed, 31 skipped, 0 failed** (7:08).
- `ruff check alis/ tests/`: clean, now including all 39 previously-excluded
  modules.
- **`pytest --run-slow`: 859 passed, 31 skipped, 0 failed (2:21:30)** --
  including the `slow`, `gpu` and `machine_dependent` batches. The baseline
  taken before the stage (2026-08-05, Q6.13) was 840 passed, 31 skipped, and the
  unit batch grew 717 -> 736, so **859 - 840 = 19 is exactly the new tests**:
  no pre-existing regression case moved, and the skip count is unchanged, so
  nothing stopped running either.

  That is the number that matters, because the stage deleted `settings.alis` and
  moved `chisq fstep`/`maxiter` into the dataclass, flipped the CLI/model-file
  precedence, made `-p 0` take effect for the first time, added the `#[cli]`
  block to every `.mod.out`, and removed ~200 lines of SuperMongo -- and not one
  fit changed.

**No golden reference was regenerated, and none needed to be.** The fast batch
is green with the `.mod.out` files now carrying the override block, because
`compare_mod_out` compares chi-squared, DOF and the model sections and ignores
the settings block. Regenerating would make the references *representative* of
what the writer now emits, but it is not required for the gate -- which is worth
knowing, because Q6.26 step 5 budgeted 2.5 hours for it.
