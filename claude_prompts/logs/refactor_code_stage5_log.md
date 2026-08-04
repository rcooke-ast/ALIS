# Stage 5 log -- Data and I/O modernisation

## Task 5.6 -- Atomic masses were read off by one [COMPLETE]
##
## (Resolved 2026-08-03: RJC regenerated all 42 references with the new
## `regen_harness.sh`, and the full harness -- `pytest --run-slow` -- came back
## 613 passed, 31 skipped, 0 failed in 2:23:48, including the slow, gpu (40/40)
## and machine_dependent batches. The "not done, deliberately" note at the end
## of this entry is therefore now historical.)

Found while auditing 5.2, disputed, then confirmed by RJC with an independent
generate-and-fit test (a H I line generated at b = 10 km/s with no thermal
broadening, refitted with zero turbulence and pure thermal broadening).

**The defect.** `load_atomic` published `Element` and `AtomicMass` as two lists
de-duplicated *independently* -- `Element` by isotope string, `AtomicMass` by
mass **value**. Every line-profile function reads them as a per-isotope mapping:

```python
m = np.where(atomic['Element'] == isotope); mass = atomic['AtomicMass'][m][0]
```

12 sites do this (4 each in `voigt.py`, `splineabs.py`, `lineemission.py`) and
nothing else reads either array. The two were 80 and 82 entries long, because
`1H`/`1Ly`/`1TL` share a single mass value and four isotopes carried two, so the
index found in `Element` addressed the wrong slot.

**Where the bug was, and why.** In the producer. Every other `AtomicData` field
is indexed by a *thing* -- a transition (815 entries) or an isotope (80).
`AtomicMass` was indexed by "the order in which a distinct float first
appeared", which corresponds to no entity and therefore cannot be addressed by
anything. An array with a meaningless index space is a defect wherever it is
built, independent of what consumers do with it. Confirmed experimentally: a
control that corrected *only* `load_atomic`'s two output arrays, leaving all 12
call sites untouched, took the check from 132 mismatches to 0.

**Effect.** 132 of the 167 isotope/ion combinations in `atomic.xml` were handed
another isotope's mass, which feeds the thermal width
`b = sqrt(b_turb^2 + 0.01662892444*T/m)` (`voigt.py:336`):

| ion | mass used | correct | b error |
|---|---|---|---|
| 1H | 2.0141018 (deuterium) | 1.00783 | -29.1% |
| 16O | 20.1800000 (neon) | 15.9994 | -9.9% |
| 14N | 19.0000000 | 14.00307 | -13.0% |
| 12C | 14.0030740 (nitrogen) | 12.00000 | -6.9% |
| 13C3 (molecular) | 200.0 | 13 | -63.6% |

D/H was largely insulated -- 1H and 2H both shift to the next isotope up, so
b(H)/b(D) was only 0.3% out -- which is very likely why this survived so long.
It is H-versus-metal at a tied temperature that was ~20% wrong.

**Fix 1: `alis/data/atomic.xml`.** Four isotopes carried two `AtomicMass`
values, mixing the isotopic mass with the element's standard atomic weight, and
they did *not* split by ion stage (`24Mg II`, `28Si II` and `56Fe II` each
appeared with both). RJC chose the isotopic value throughout:

| isotope | was | now | rows |
|---|---|---|---|
| 24Mg | 24.305 | 23.985041700 | 20 |
| 28Si | 28.0855 | 27.9769265325 | 19 |
| 47Ti | 47.867 | 46.9517631 | 2 |
| 56Fe | 55.845 | 55.9349375 | 39 |

Edited structurally -- only the 2nd `<TD>` of rows belonging to those four
isotopes -- because values like `55.845` also occur in other columns and a
string replace would have been unsafe. Verified: exactly 80 lines changed, four
distinct substitutions with those counts, 815 rows and the `nrows` attribute
unchanged, and no isotope now carries more than one mass. The new values are the
file's own existing literals, so no digits were re-entered (RJC's concern about
transcription errors when the ECSV lands).

**Fix 2: `load.load_atomic`.** The two arrays are now built in one pass, so
`Element[i]` is the isotope whose mass is `AtomicMass[i]`. All 12 consumer sites
were left untouched -- they are correct once the contract holds, and editing 12
places in 3 files would add risk to a change that already moves every reference.
A warning now fires once per isotope if a file gives one isotope two masses,
which guards a user-supplied file and 5.2's ECSV.

**Tests.** New `tests/test_atomic_mass.py`, 20 `unit` tests, grown from RJC's
`context/atomic/test_voigt_atomic.py`. Two layers, deliberately:
- *format-specific* -- the mapping `load_atomic` publishes agrees with the raw
  file (uses a VOTable parser, so 5.2 must revisit these);
- *format-agnostic* -- the mass that actually reaches the profile is the one
  `load_atomic` published. It drives ALIS's own parser with one model line per
  isotope/ion, reads the `b` that `set_vars` returns and inverts it to recover
  the mass. It never touches the file, so it carries over to `atomic.ecsv`
  unchanged, which is what RJC asked these tests to protect.
Both layers were verified to bite: reverting `load_atomic` to the old logic
gives **12 failures and 6 errors**; with the fix, 20 pass.

**Gate.**
- `pytest -m unit`: **490 passed, 31 skipped** (470 before; +20 from
  `test_atomic_mass.py`).
- `context/atomic/test_voigt_atomic.py`: passes on the *real* code path now
  (its `--fixed` control is consequently a no-op and also passes).
- Lint (ruff / isort / black) clean.
- **Stage 0 fast batch: 39 failed, 24 passed.** This is the expected and
  intended consequence -- the thermal width has changed, so every reference
  produced with the old masses has moved. Every failing example contains a
  voigt-family component (`generate`, `lls`, `lsf_hst`, `lsf_multigauss`,
  `metal_line_abs`, `polynomial`, `powerlaw`, `random`, `spline`, `voigtconv`);
  the 24 that still pass are the models with no thermal broadening. The
  `generate` case moved too, so the *generated* spectra change as well as the
  fits. `metal_line_abs` starting chi-squared: 512.368 -> 476.036; best fit
  358.112 -> 357.874.

**Not done, deliberately: the references have not been regenerated.** That
rewrites every golden file in the repository and is not reversible by
inspection, so it is RJC's call whether it happens now in one labelled commit
or is held for a release (the options put to him in Q5.15). Until then the fast
batch is red, and that redness is the correct signal.

## Task 5.4 -- Output-writer round-trip faithfulness [COMPLETE]

Six items, of which two turned out to be the same bug and one needed a different
design from the one agreed in Q5.16.

**1. `lsf` keyword echo.** The writer emitted `resolution=lsf(name=STIS,...)`;
the reader needs colons. `load_data` splits each data-line token on `=` and keeps
only field 1, so an `=` inside the parentheses truncates the value and the
function reads back as `lsf(name`. Fixed where the resolution string is composed
(`save.py`), converting `=` -> `:` in the parameter list only. Safe for every
convolution function: the five that take keywords (`lsf`, `lsffile`,
`lsfspline`, `apod`, `multivfwhm`) all undo it with `instr.replace(":", "=")` on
the way in, and purely numeric ones contain no `=`. Not applied to `shift=`,
where no loader does the reverse.

**2+3. `damping=` and `splineabs`'s empty `locations=` are one bug, and it is not
in the writer.** `parout` already refuses to echo a parameter the user did not
give -- it reads `mp['mkey'][i]['input'][parid]`
(`alis/functions/base.py:500-505`), a tri-state 0/not given, 1/positional,
2/named. That record was wrong. Since Stage 2 each function is instantiated
**once** and reused for every line of its type; `load()` records named parameters
in the *instance's* `self._keywd['input']` and then **shallow**-copies `_keywd`
into `mp['mkey']` (`base.py:384`). The record is never cleared between lines,
*and* every line ends up sharing one `input` dict.

Confirmed directly rather than by reading: on `helium34/Her36` all 12 voigt lines
had `id()`-identical `input` dicts and every one reported `damping` as named,
though only 1 line names it. The writer was echoing, per line, the **union of the
parameters named on any line of that function**.

Fixed in `load.call_function_load`, which hands the instance a fresh `input` dict
per call -- clearing the record and de-aliasing the previous line's snapshot in
one move. All four `load()` call sites route through it. Verified on
`examples/splineabs/fit_spectra_linear`, where line 23 names
`ColDensScale=1.0E13 logN=False` and line 24 does not: after the fix line 23
keeps them and line 24 drops them, which is what a faithful re-read needs.

There was no narrower fix available for `damping`. Its `_fixpar` is `True` with
`_defpar` 0.0, so an absent `damping` is fixed at 0.0 while a written-out
`damping=0.0000000` re-reads as **free**; and there is no per-token "fixed"
marker to annotate with, since the suffix letters are *tie* labels and fixing is
done by a whole-function `fix voigt damping True` line.

**4. Pixel-load buffer -- Q5.16's encoding does not work.** The agreed fix was to
write the loaded wavelength range as `loadrange=`. It fails: `load_data` widens
an *explicit* `loadrange` by the same 10-sigma resolution rule
(`alis/load.py:935`), so recording the loaded range re-inflates it --
`metal_line_abs` went 392 px -> **414 px** on re-read. Making an explicit
`loadrange` authoritative instead was rejected: the only models that use one
(`helium34/Her36`, `helium34/HD319718`) are *in the harness*, so it would have
changed their loaded pixels and hence their fits, not just their echo.

The buffer is therefore written as a resolution-independent **pixel count**,
reusing the `bufferpix` keyword approved as Q5.16 item 2. Both sides are stored
(`bufferpix=[left,right]`) because the resolution rule extends by a *wavelength*
and so does not cover the same number of pixels either side -- `lsf_hst` measures
`[3492,3490]`. Round-trip verified **exact** (identical `sha256` of the loaded
wavelength array) on `metal_line_abs` and on `summed_coldens`, which has a
genuinely fitted resolution (`vfwhm(7.0va)` -> `vfwhm(7.000va)`). Lines with an
explicit `loadrange` -- including the 1231 `loadrange=all` in this repo, e.g.
`voigtconv` -- get no `bufferpix` and are left untouched.

**5. `bufferpix` as a user keyword.** Added to the data-line keywords, accepting
`N` or `[left,right]`, with **no default** so the resolution rule stays in charge
unless asked. Warns when the requested buffer is narrower than the resolution
needs, since that truncates the convolution near the fitrange edges.

**6. `save_covar` filename.** `filename.rstrip(fnspl[-1])+'png'` replaced with
`os.path.splitext(filename)[0]+'.png'`. `rstrip` strips a character *set*, so
`out covar mycovar` wrote the image to a file literally called `png`.

**Blast radius -- 14 of 41 references move, not the 6 predicted in Q5.20.** The
prediction was made from the *symptom* (`damping=`/`locations=` tokens); the root
cause is general, so inherited `blind=`, `logN=` and `ColDensScale=` move too.
Measured by running the full gate:

- `pytest -m "unit or fast"`: **553 passed, 9 failed** (one of those 9,
  `test_fit_report`, was my own debris -- a stray `.mod.out.out` left by running
  ALIS *on* a `.mod.out` during testing; removed, now 5/5).
- `pytest -m "medium or slow" --run-slow`: **5 passed, 6 failed**.

The 14: `examples/{blind,metal_line_abs/fit_spectra_linear,spline/fit_spectra_splineContAbs,splineabs/fit_spectra_linear}`,
`helium34/{Her36,tet02OriA,HD319718}`, and
`DH/{HS0105p1619,J1358p0349,J0814p5029,J1358p6522_original,J1419p0829,J1558m0031_FINAL_MODEL,Q0913p072}`.

**The change is cosmetic, and that was checked rather than assumed.** For the 5
cases with an in-place run available -- covering all three categories (`damping`,
`locations`, inherited keywords) -- the harness's own parser reports
chi-squared, DOF and every parameter value **byte-identical**, with only keyword
tokens differing. No non-keyword token moved anywhere.

**Regeneration (RJC approved on being shown the true figure).** Done with
`regen_harness.sh --clean`. It ran in two passes, and the second was necessary:

- The first pass covered the 14 cases that *failed the gate*. That was not
  enough. `examples/lsf_hst` never failed -- its change is in the `data read`
  line, which `compare_mod_out` does not compare -- so its reference still
  carried the broken `lsf(name=` syntax while the `.mod.out.reference_adjusted`
  that mode (b) fell back on had just been deleted. Mode (b) would have crashed
  on it. Caught by reading the regenerated file rather than trusting the gate.
- The second pass therefore regenerated **all** of `examples/`, so every data
  line matches what the writer now emits. **23 references** changed in total.

Verified file by file against the committed versions with the harness's own
parser: **21 cosmetic** (chi-squared, DOF and every parameter byte-identical;
only keyword tokens differ), **3 timestamp-only** -- restored with
`git show HEAD:<path> > <path>`, so the review diff carries no noise -- and
**1 flagged**, `examples/random`, which draws its start from
`uniform(-2.0,0.0)` at load time and therefore re-draws on every run (initial
chi-squared 532.774 -> 542.000). The gate tolerates that through its parameter
tolerance and passed the case; regenerating merely re-pins the draw.

The strongest evidence that no fit moved: across the pre-regeneration
`fast` and `medium or slow` runs, **all 741 comparison failures were
"token count differs"** -- not one chi-squared, DOF, value or error mismatch.

Confirmed by construction too: `lineemission`, `powerlaw` and `voigtconv` got
no `bufferpix` at all, because all three set `loadrange=all` and the writer
leaves an explicit range alone.

**Workarounds removed.** `_ZERO_DAMPING_RE` is gone from `tests/alisrun.py`, and
both `.mod.out.reference_adjusted` files are deleted; `alisrun.py:349` falls back
to the plain reference, which now carries the corrected syntax.

**Gate: `pytest --run-slow` -> 613 passed, 31 skipped, 0 failed (2:24:08)** --
identical to the pre-5.4 baseline of 613/31/0, including the `slow`, `gpu`
(40/40) and `machine_dependent` batches. Six of the regenerated context cases are
machine-dependent, so they are re-pinned to this machine, as RJC's 5.6
regeneration also was.

**Lint.** `alis/load.py` and `alis/save.py` are both on ruff's and black's
exclusion lists (the Stage 6.3 to-do list), so the linters skip them; the
additions match the files' existing style, where 295 of 2083 lines already
exceed 88 characters.

## Task 5.2 -- Atomic data modernisation [COMPLETE]

`alis/data/atomic.ecsv` replaces `atomic.xml` as the file ALIS reads. 830 rows:
the 815 of `atomic.xml` transcribed unchanged, plus exactly the 15 3He I lines
RJC selected in Q5.8.

**The converter is shipped, not a one-off** (Q5.17):
`alis/data/convert_xmlFormat_to_ecsvFormat.py`, alongside the existing
`convert_datFormat_to_xmlFormat.py`. Users need it for their own files.

- Unit strings are taken **verbatim from the VOTable's FIELD attributes**, not
  from astropy's parsed `field.unit`, which normalises `1.66053886x10-24g` to
  `1.66054e-24 g` and loses digits from what is meant to document the file's own
  convention. ECSV stores an unrecognised unit string as written and reads it
  back unchanged, so the original survives. Recognised units are normalised to
  exactly equivalent forms (`s-1` -> `1 / s`, `0.1nm` -> `0.1 nm`).
- The data lines are padded so the columns line up. ECSV's space delimiter
  tolerates runs of spaces, so alignment costs nothing at read time and makes
  the table readable in an editor -- the point of leaving VOTable.
- `--append-from` / `--append-wave` selects specific transitions from a second
  VOTable, and **errors unless each requested wavelength matches exactly one
  row**. That is how the 15 3He lines were added, and it is auditable: the exact
  command is in this log's git history.
- The script verifies its own output cell by cell and exits non-zero on any
  difference.

**The 16th 3He line was excluded on purpose.** `atomic_rjc2.xml` has 16 3He rows
absent from the default, not 15. The extra one is 10834.374576 (f=0.17974),
which is a near-duplicate of the default's own 10834.374575 (f=0.0239733) --
same line, differing in the 10th decimal place, with an irreconcilable
oscillator strength. RJC's "no other 3He change" covers it; selecting by
explicit wavelength rather than by set difference is what keeps it out.

**Loader.** `atomic.ecsv` is the default (`config.RunConfig.atomic`,
`data/settings.alis`). `read_atomic_table` dispatches on the extension: ECSV
normally, VOTable still accepted so a user's own `.xml` keeps working, with a
deprecation warning that says what to do instead. 5.6 is not regressed --
`Element`/`AtomicMass` are still built as one isotope -> mass mapping.

**Data-directory lookup tidied.** `load_atomic` located `alis/data/` by
splitting `argflag['run']['prognm']` on `'/'`; it now uses the module's own
location (`load.atomic_datadir`). `prognm` is no longer consulted, so
`tests/test_atomic_mass.py` stopped setting it to `alis/alis.py` -- a file that
has not existed since Stage 2, which the tests never noticed because only the
directory part was ever used.

**Validation -- and one deviation from the doc, deliberately.** The duplicate
check (Element+Ion+RestWave ignoring MassNumber) is implemented as
`load.duplicate_transitions`, but it runs in the **converter**, not on every
load. Running it on load would warn on every single fit, because the shipped
file legitimately contains four such groups -- 6/7Li (different `fval`), 12/13C
and 54/56/57/58Fe (identical `fval` and `Gamma`). Those are real isotopes, not
the mislabelling the check exists to catch, and a warning that always fires
trains users to ignore it. Conversion is also when a bad row would actually be
introduced. What does run on load is a missing-column check, which fails early
and legibly instead of as a KeyError inside the profile calculation.

**Repointed the dead `run atomic` lines** (Q5.13 option b): 13 occurrences
across 11 context models, `atomic_rjc.xml` -> `atomic.ecsv`. (The doc said ten
models; the count had grown.)

**Tests.** `tests/test_atomic_mass.py` 20 -> 29. Only the format-specific half
was touched -- `_file_masses` now dispatches on extension. **The
format-agnostic half needed no edits**, which was the doc's stated condition for
the conversion being faithful. New:

- every cell of the 815 shared rows is identical between the two files. This
  compares the raw VOTable `.array`, not `.to_table()`: astropy *masks* NaN in a
  double column and a masked cell compares equal to nothing, which first showed
  up as 1483 spurious differences. The file's own value is NaN -- it literally
  contains `<TD>NaN</TD>` for all 815 K values and 664 of the q values.
- the ECSV is the XML plus exactly the 15 agreed rows, each present by wavelength;
- both formats load to the same isotope -> mass mapping;
- `duplicate_transitions` finds exactly the four known groups, plus a mutation
  test that a genuinely duplicated line is reported;
- the VOTable path warns and names the converter; the ECSV path does not warn;
- a missing column is fatal;
- the data directory is found without `prognm`.

Verified to bite: perturbing one `fval` in `atomic.ecsv` by one digit in the
last decimal place (1 cell of 8150) fails the fidelity test.

The `logmsgs` fixture attaches a handler to the shared 'alis' logger, as
`test_logger.py` does. Neither `capsys` nor `capfd` sees `msgs` output -- the
logger's stderr handler binds its stream at import, before pytest's capture.

**No fit can have changed, and that was checked rather than argued.** All 85
model files in `examples/` and `context/fitting_examples/` were scanned: not one
has a `fitrange` within 50 A of any added 3He line (the nearest work is
`helium34` at 3187-3190, 3888-3891 and 10827-10838; the added lines are at
515/522/537/6680/7067).

**Gate.**
- `pytest -m unit`: **499 passed, 31 skipped** (490 before; +9 new).
- `pytest -m "unit or fast"`: **571 passed, 31 skipped, 0 failed**.
- `pytest --run-slow`: **622 passed, 31 skipped, 0 failed** (2:22:16) --
  613 + the 9 new tests, with every pre-existing case unchanged, including the
  `slow`, `gpu` (40/40) and `machine_dependent` batches.
- ruff / black / isort clean on both new files.
- No golden file was regenerated for this task, and none needed to be. That is
  the point worth keeping: 5.2 swapped the atomic data file that every fit in
  the repository reads, and not one reference moved.

## Task 5.3 -- Plotting-script output [COMPLETE]

`alis/plotscript.py` emits a standalone matplotlib script beside the fit. ALIS
already draws its own PDF (`out plots`); this is different -- it writes a
*script*, so a publication figure can be edited without re-running the fit or
working inside ALIS. Stage 6.6 drives the same emitter from the GUI.

**Settings: a new `plotscript` section**, as RJC asked in Q5.18 --
`format` (`none`/`metals`/`DH`, default `none`), `numcol`, `residuals`,
`velrange`, `ylim`, `figsize`, `fontsize`, `labels`, `filename`, `overwrite`.
As predicted in the audit, `load.set_params` needed no change: it dispatches on
`linspl[0] in argflag.keys()`, so adding `PlotScriptConfig` to `ArgFlag` was
enough. The fields that must accept a word *or* a number/list (`numcol`,
`velrange`, `ylim`, `figsize`) default to `None`, since `set_params` converts by
the type of the default and only the `None` branch infers int/float/list/str.

**How the panels are found.** Each absorption component is asked which of its
transitions fall in a snip's fitrange, using the `set_vars` call the fit itself
uses -- so a panel shows what was actually modelled rather than a guess from the
wavelength range. Its return columns are
`[coldens, redshift, b, restwave, fvalue, gamma]`, which gives the centre
wavelength, the label and the strength in one go.

**One panel per transition, not per snip.** `metal_line_abs` fits O I 1302 and
Si II 1304 in a single fitrange; they are ~460 km/s apart, far wider than a
velocity panel shows, so one panel per snip silently dropped a line. Transitions
closer together than the velocity window *are* merged, keeping the strongest as
the centre -- those are blends, which the reference figures also draw in one
panel.

**Self-contained, and checked to be.** The reference figures in
`context/plotting/` all `import plotting_routines as pr`, which is not
distributed, and they are Python 2 (`xrange`). What is emitted is Python 3 with
the handful of helpers inlined. The test runs the emitted script in a subprocess
with `PYTHONPATH=""`, so a lingering dependency on ALIS or on `plotting_routines`
fails there rather than in a user's hands.

**Layouts.** `metals` is three columns with an ion's transitions adjacent. `DH`
is two columns: with one dataset the series runs *down* the left column and
continues down the right (the emitted order is column-major, since matplotlib
fills row-major); with two datasets the columns are the datasets and the rows
the transitions, so the same line can be compared across them; with more than
two the user edits the script, as agreed in Q5.11.

**Lyman labels.** The DH models write the series as the pseudo-element `Ly` with
the member as its "ion stage" (`1Ly_a` is Ly-alpha), which came out as "Ly a".
Now rendered as Ly-alpha/beta/gamma/delta and numbered after that, so `1Ly_g`
gives "Ly7" -- matching the "to about Ly7" in RJC's Q5.9 description.

**Two fixes found by looking at the output rather than the code.** The shared
axis labels first overlaid the panels: the reference scripts' frameless
`add_subplot(111)` trick needs hand-tuned margins, so the emitted script uses
`supxlabel`/`supylabel` and reserves margins specified in *inches* (converted to
figure fractions) so the spacing holds at any panel count. `bbox_inches='tight'`
then had to go, since it cropped away the margins just reserved.

**Never fatal.** `write_plotscript` catches its own errors and warns: a fit that
has completed should not be lost to a plotting problem. It also warns when
`out fits` is off, since the emitted script reads the `*_fit.dat` files that
setting writes.

**Tests.** New `tests/test_plotscript.py`, 10 `unit` tests: the settings parse
(including `auto` and comma-list forms), the three layout orderings, and three
end-to-end tests that run a real fit -- `none` emits nothing, the emitted script
runs standalone and draws a non-empty figure, and both transitions of a shared
snip get their own panel.

Verified by eye as well as by assertion: the `metal_line_abs` figure was
rendered and inspected -- two panels, data as black steps, model in red, dashed
continuum and zero-level guides, each line centred at v = 0.

**Gate.**
- `pytest -m unit`: **509 passed, 31 skipped** (499 before; +10).
- `pytest -m "unit or fast"`: **581 passed, 31 skipped, 0 failed** (571 before).
- ruff / black / isort clean on both new files. Note the rule set that actually
  gates is the one pinned in `.pre-commit-config.yaml` (ruff **v0.6.9**, whose
  defaults are E4/E7/E9/F). A newer local ruff reports ~50 further findings on
  these files, but it reports them on the already-cleaned modules too
  (`gpu.py`, `report.py`, `shared_arrays.py`, `test_logger.py`), so that is a
  repo-wide question for Stage 6.3, not a 5.3 one.
- No golden file moved: `plotscript format` defaults to `none`, so no existing
  model emits anything.

**Not exercised on real data: the one- and two-dataset `DH` orderings.** The
only DH model to hand (`DH/J1358p0349`) has **10** datasets, which is the
">2, the user edits the script" branch of Q5.11. Both other orderings are
unit-tested against RJC's Q5.9/Q5.11 descriptions, but no model in the harness
has exactly one or two datasets covering a Lyman series, so they have not been
seen end to end.

## Task 5.5 -- Unit tests for the stage's stable surface [COMPLETE]
##
## (Finished 2026-08-04. The first half -- the `.mod` -> `.mod.out` -> `.mod`
## round trip -- is written up immediately below; the file-format loaders, the
## model parser and the writer helpers follow it.)

Parts of 5.5 were delivered with the tasks they belong to: the atomic
loader/converter by `tests/test_atomic_mass.py` (29 tests, Task 5.2) and the
plotting emitter by `tests/test_plotscript.py` (10, Task 5.3). Checked before
adding anything: `tests/test_load_units.py` already covers `cpucheck`,
`get_binsize`, `getis`, `load_tied`, `pinfl_changed` and `set_params`, and none
of the three overlaps.

**Done here: `tests/test_writer_round_trip.py`, 79 tests, 4 seconds, no fits.**
This is the check the doc singles out -- a saved `.mod.out` is documented to be
a valid `.mod`, the Stage 0 harness carried `_ZERO_DAMPING_RE` and two
hand-fixed `.mod.out.reference_adjusted` files *only* because nothing pinned it,
and 5.4 deleted those workarounds.

No fit is run: the committed `.mod.out.reference` files **are** saved models, so
re-reading them is the round trip, and only the `model read` block is parsed, so
no spectrum is touched. That is what makes 40 real models affordable as a `unit`
test. It covers `examples/` and, when present, `context/fitting_examples/` --
which is untracked, so on a clean checkout the test still runs on `examples/`.

Two invariants per model:
- the reader **accepts** the saved file (the `examples/lsf_hst` failure, where
  the echoed `resolution=lsf(name=STIS,...)` crashed it outright);
- the **free-parameter count is unchanged** by the round trip. DOF = Npix -
  Nfree and the pixels do not move, so this is exactly the quantity the
  `damping=` bug shifted (helium34: 618 against 621).

**Blind models are excluded, and the exclusion is itself asserted.** Five
references -- `examples/blind`, and DH's `J0814p5029`, `J1358p0349`,
`J1558m0031_FINAL_MODEL`, `Q0913p072` -- are rejected by the reader, which is
*correct*: `run blind` replaces the parameters with `------ BLIND MODEL ------`,
so a blind output is meant not to be re-readable. A separate test asserts each
of those contains the blind marker **and** that it fails to parse, so a model
cannot quietly leave the round-trip suite by breaking for some other reason.

**Verified to bite.** Re-introducing the pre-5.4 writer behaviour -- stamping a
suffixless `damping=0.0000000` onto every voigt line that does not name one --
takes `helium34/Her36` from 56 free parameters to **67**, and the test fails.
The delta of 11 is exactly the 11 of its 12 voigt lines that do not name
damping, which is the mechanism diagnosed in Task 5.4. A companion unit test
pins that mechanism directly, so the round-trip test is known to remain capable
of catching it.

**Gate.** `pytest -m unit`: **588 passed, 31 skipped** (509 before; +79).
ruff (E4/E7/E9/F, the pinned v0.6.9 set) / black / isort clean.

### The rest of 5.5 (2026-08-04): the loaders, the parser and the writer

Three new files, **94 tests, 5 seconds, no fits and no golden files**. Nothing
in `alis/` changed -- this task is tests only.

| file | tests | covers |
|---|---|---|
| `tests/test_load_files.py` | 41 | `load_ascii`, `load_fits`, `load_datafile`, `load_userdata`, `load_data` |
| `tests/test_load_model.py` | 31 | the `.mod` model-block parser: named/positional parameters, ties, `fix`, `lim` |
| `tests/test_save_helpers.py` | 22 | `print_model`, `modlines`, `save_model`'s data line, `save_covar`, and a save-then-reload round trip |

Checked first, so nothing was written twice: `test_load_units.py` covers
`cpucheck` / `get_binsize` / `getis` / `load_tied` / `pinfl_changed` /
`set_params`; `test_atomic_mass.py` the atomic loader; `test_plotscript.py` the
emitter; `test_writer_round_trip.py` the re-reading half of the round trip.
None overlaps.

**The writer half of the round trip was the gap worth closing.**
`test_writer_round_trip.py` re-reads the 40 committed `.mod.out.reference`
files, but it cannot *run* the writer -- those files are the output of fits
that take hours. `test_save_helpers.py` builds a model over a 500-pixel
synthetic spectrum, calls `save_model(save=False, getlines=True)` to get a full
`.mod.out` in milliseconds, and feeds it straight back in. That closes the loop,
and it is what makes the 5.4 fixes testable at all rather than only observable
in a reference file.

The strongest test in the set is the one that moves the resolution: the fit is
told it ended on `vfwhm(40.0)` when it started at `vfwhm(7.0)`, the model is
written and re-read, and the loaded wavelength array must be **identical**
(`np.array_equal`, not `allclose`). Its control strips `bufferpix` from the same
file -- the pre-5.4 writer's output -- and the pixel count changes. That is the
5.4 `bufferpix` argument, reproduced in two seconds instead of by inspection.

**Every test was checked to bite.** Five mutations were applied to `alis/`, each
run against all 94, and each reverted:

| mutation | tests that failed |
|---|---|
| writer stops converting `=` -> `:` in the resolution keywords | 1 |
| `save_covar` back to `filename.rstrip(fnspl[-1]) + 'png'` | 1 |
| writer stops emitting `bufferpix` | 3 |
| `call_function_load` stops clearing `_keywd['input']` | 6 |
| `load_data` ignores `bufferpix` | 7 |

Two are worth recording. The `rstrip` mutation did not merely misname the file:
with an absolute `out covar` path and no extension it stripped the path itself
and wrote the correlation matrix to **`png.png` in the working directory** --
worse than the "a file called `png`" in the task doc, and confirmation the fix
was not cosmetic. And the `call_function_load` mutation fails the *writer*
tests as well as the parser ones, which is the two ends of the 5.4 `damping=`
bug meeting in the middle.

**The `lsf` colon rule is now pinned at both ends**, which it was not before.
`test_writer_round_trip.py` claims to cover the `lsf_hst` failure, but it parses
only the `model read` block; the `resolution=lsf(name=STIS,...)` string lives in
the *data* line, which that test never loads. So the writer's `=` -> `:`
substitution had no test. It has two now: one that writes a model with an `lsf`
resolution and asserts no `=` survives inside the parentheses, and one that
loads the `=` form and finds the reader has kept `lsf(name`. Both are marked
`linetools` and auto-skip where that optional package is absent. A third,
dependency-free test shows the same truncation on `label=left=right`, so the
underlying tokenizer rule is pinned even in CI without linetools.

**One defect found, reported, and then fixed by RJC: `load_fits` returned a
continuum of zeros where `load_ascii` returns ones.** `load_subpixels`
(`alis/load.py:1889`) treats an all-ones continuum as "none supplied" and skips
the interpolation; anything else is taken as a real continuum and multiplies the
model (`model_eval.py:572`). So a FITS spectrum loaded without a `continuum`
column got a continuum of 0.0 and the model was multiplied by zero. No shipped
example or context fit loads FITS data -- all 85 use ascii -- so nothing in the
harness exercised it. It was first recorded in a labelled test rather than
fixed, on the grounds that Stage 5 must not move a fit; **RJC then changed the
three `np.zeros` to `np.ones` himself (2026-08-04)**, which tripped that test
exactly as it was designed to.

**A companion slip went with it, and had to be fixed for the fix to hold.** The
zero-level branch of `load_fits` reads its column, warns that it will not be
used, and then substitutes the default -- but it assigned to `contin`, not
`zeroin`. That was invisible while the continuum was zeros either way; once it
is ones, a `zerolevel` column put the continuum straight back to zeros and
defeated the fix. Corrected to `zeroin`, mirroring the continuum branch three
lines above.

The test was rewritten accordingly, and is now stronger than what it replaced:
`test_every_loader_agrees_on_the_nothing_supplied_sentinels` compares
`load_fits`, `load_ascii` and `load_userdata` against **each other** rather than
against a literal, so they cannot drift apart again, and
`test_load_fits_keeps_the_continuum_default_when_a_zerolevel_is_given` covers
the companion slip (reverting it fails that test).

**Three smaller things found while writing. One was then fixed on RJC's
instruction (see the next section); the other two are recorded only.**
- `1e4` is not a number to ALIS. `check_tied_param` strips leading
  `+-.0123456789` and treats the rest as a tie label, and its scientific-notation
  escape only recognises `E+`/`e+`/`E-`/`e-` -- so `1e4` parses as the value
  **1.0** tied to a label `e4`. Two lines written `1e4` silently share one free
  parameter. Bit these tests before the literals were changed to `10000.0`;
  `1.0E+04` is the form the writer itself emits. **Now rejected at the parse
  (2026-08-04).**
- `load_input(textstr=...)` returns lines *without* their newlines (it splits on
  `"\n"`), while `load_input(filename=...)` keeps them -- and `save_model`
  concatenates `_parlines` assuming they end in one, so settings run together
  as `run ngpus 0run backend cpu`. Only the interactive onefits menu uses
  `textstr`, and only to plot, so it is not live. The tests write to a real file
  and re-read it, which is what a user does anyway.
- The **first** data line of a block must carry `columns=`, and the message if
  it does not is wrong. `colspl` is assigned only inside the `columns` branch of
  `load_data`'s loop, so omitting it on the first line raises `NameError`, which
  the bare `except:` around the load turns into "Error reading in file". Later
  lines are fine -- they inherit the previous line's `colspl`, which is why 131
  of the 181 data lines in `examples/` can omit the keyword. Not pinned by a
  test: cementing "raises SystemExit with a misleading message" is not worth it.

**Runtime.** The three files together take **2.4 s**. `atomic_data` is loaded
once per session; `build_funcarray` is 0.3 ms, so a fresh registry per test is
free; and no test runs a fit, opens a subprocess or reads a golden file.

**`logmsgs` moved to `tests/conftest.py`**, as the task doc asked once a third
caller appeared (there are four now). A session-scoped `atomic_data` fixture
went with it, so `atomic.ecsv` is read once for the whole run instead of once
per module. `test_atomic_mass.py`'s local copy of `logmsgs` was removed; nothing
else in it changed.

Every test builds its **own** function registry. `load_model` writes `fix` and
`lim` lines straight onto the shared function *instances* (`_fixpar`,
`_limited`, `_limits`), so a shared registry would carry one test's `fix` into
the next. `build_funcarray` costs 0.3 ms, so this is free.

**Gate.**
- `pytest -m unit`: **682 passed, 31 skipped** (588 before; +94). 701 after the
  `load_fits` fix and 5.7's check landed.
- `pytest -m "unit or fast"`: **754 passed, 31 skipped, 0 failed** (7:20) --
  660 before, so the entire delta is the 94 new tests and no regression case
  moved.
- ruff (E4/E7/E9/F, the pinned v0.6.9 set) / black / isort clean on all five
  touched test files.
- No golden file moved, and at this point no file under `alis/` had changed, so
  the Stage 0 gate could not have moved either. (The `load_fits` continuum fix
  and 5.7 came afterwards; neither can move a fit, for the reasons recorded
  above and below.)
- `tests/README.md` updated: the unit section now names the Stage 4/5 files and
  records that the I/O deferral from `refactor_code_unit_tests.md` is
  discharged.

## Task 5.7 -- An unsigned exponent is now an error, not a silent tie [COMPLETE]

Added 2026-08-04 at RJC's request, on the `1e4` finding above.

**The rule, exactly as RJC specified it.** A tie label that is `e` or `E`
followed by digits **and nothing else** is rejected. Everything else is left
alone:

| token | tie label | verdict |
|---|---|---|
| `1e4`, `1.0e345`, `5.0E45`, `2.0E5`, `7E054` | `e4`, `e345`, `E45`, `E5`, `E054` | **error** |
| `1.0E+04`, `1.0e-03`, `3.0e+1`, `1.0E-34567`, `2.0E+4` | none -- consumed as the exponent | number |
| `1.0E5t`, `1.0e345j`, `1.0e293e` | `E5t`, `e345j`, `e293e` | ordinary tie label |
| `5.0da`, `8000.0TA` | `da`, `TA` | ordinary tie label |

A signed exponent never reaches the check: the existing `E+`/`e+`/`E-`/`e-`
branch has already stripped it and left an empty label. A label that merely
*starts* `E`+digits cannot be a number, so it is not ambiguous and is untouched.
That is the whole of the rule -- reject only the case where the two readings
genuinely collide.

**Where it went.** One `check_tie_label` in `alis/functions/base.py`, called
from **25 sites**: the 20 copies of `check_tied_param` across 17 modules
(`shift.py` has four), and the 5 `getminmax` parameter loops, which parse a
resolution string with the same micro-syntax -- so `resolution=vfwhm(1e4)` is
caught as well as a model line. `lineemission.py` needed `base` adding to its
imports; it had only imported `voigt`.

The message names the model, states what ALIS *did* read, and gives both
remedies:

```
[ERROR] :: Ambiguous parameter '1e4' for the 'voigt' model
           ALIS reads this as the value 1 tied to a parameter labelled 'e4',
           not as scientific notation. If you meant a number, give the exponent
           a sign (1E+4 or 1E-4); if you meant a tie label, rename it so
           it is not 'e' followed only by digits.
```

**Nothing in the repository trips it, and that was measured rather than
assumed.** All 152 `.mod` / `.mod.out.reference` / `.mod.out.reference_adjusted`
files were scanned with the parser's own `lstrip` logic, over both
whitespace-separated model tokens and the comma-separated arguments inside
`resolution=`/`shift=` parentheses: **zero** matches. The check can therefore
only ever fire on input that was already being misread, so no fit can move.

**The writer cannot emit an ambiguous token either.** Every `_svfmt` is a `g`
or (via `gtoef`) an `E` format, and Python's `%g`/`%E` always write the exponent
with a sign -- `1e+04`, `1.00000000E+04`. Checked across magnitudes. The
round-trip tests re-read written output, so this is pinned as well as reasoned.

**Tests: 18 new** in `test_load_model.py` (14) and `test_load_files.py` (4) --
the five rejected forms with an assertion on the message content, the signed
forms still parsing as numbers, the `E5t`-style labels still being labels, and
the resolution-string path. Plus one that matters more than the rest:

- `test_every_model_function_rejects_an_unsigned_exponent` drives **all 32**
  functions in the registry with `1e4` and requires each to exit *with the
  "Ambiguous" message*. Reading the source would not do -- `check_tied_param` is
  copy-pasted 20 times, so patching one says nothing about the other 19 -- and
  an exit-only assertion would pass vacuously on the functions that reject a
  bare parameter list for a missing required keyword.

**Verified to bite.** Making `_AMBIGUOUS_EXPONENT` match nothing fails 7 of them.

**Gate.**
- `pytest -m unit`: **701 passed, 31 skipped** (682 before; +18 for 5.7, +1 for
  the `load_fits` zero-level test).
- `pytest -m "unit or fast"`: **773 passed, 31 skipped, 0 failed** (7:42).
- ruff clean. `alis/functions/*` are on black's and isort's exclusion lists
  (the Stage 6.3/6.5 to-do), so the formatters skip them; the additions match
  each file's existing style.
- **Not run: the full `--run-slow` harness** (2.5 h). It cannot be affected --
  the check only calls `msgs.error`, never alters a value, and no model file in
  the repository matches it -- but it is the one gate not exercised.
