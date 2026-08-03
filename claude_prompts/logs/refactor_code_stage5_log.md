# Stage 5 log -- Data and I/O modernisation

## Task 5.6 -- Atomic masses were read off by one [COMPLETE, pending a decision
## on reference regeneration]

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
