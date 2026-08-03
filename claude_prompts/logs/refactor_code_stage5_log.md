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
