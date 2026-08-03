# Prompt file for ALIS software refactoring -- STAGE 5

> **Data and I/O modernisation.** Add YAML/TOML model-file support *alongside*
> the existing text format (never removing text support), modernise the atomic
> data file so it needs no manual `nrows` maintenance, and add an option to emit
> standalone matplotlib plotting scripts. Depends on Stage 2; independent of
> Stages 3/4. All existing `.mod` files must still parse and produce identical
> results (Stage 0 gate).

## Tasks

> Complete **in the order below** (RJC, Q5.10): the writer round-trip
> (5.4) first, because it is the one with a user-visible correctness
> payoff and 5.5's round-trip tests need it in place; then the atomic
> data (5.2), the plotting script (5.3), and the unit tests (5.5). **5.6 is
> DONE** (2026-08-02/03): it was added mid-stage, fixed, and its references
> regenerated, so it no longer blocks 5.2 — but 5.2 rewrites the same loader
> and must preserve what 5.6 established.
> **The numbers are kept as they were** rather than renumbered, so that
> references from Stage 6.6, the appendices below and the logs stay
> valid. 5.1 is dropped and is listed last for the record.
> Log each in `ALIS/claude_prompts/logs/refactor_code_stage5_log.md`.

**5.4 — Output-writer round-trip faithfulness (deferred from Stage 0).**
> A saved `.mod.out` is documented to be a valid `.mod` input, but the Stage 0
> fixed-parameter gate found several cases where re-reading it does not
> reproduce the fit. RJC asked (Q0.13 pt 4, Q0.14) that these be fixed here.
> Each fix must keep the Stage 0 suite green, and — where the fix makes the
> plain `.mod.out.reference` a faithful re-input again — the corresponding
> hand-fixed `<name>.mod.out.reference_adjusted` in `tests/` (currently
> `examples/lsf_hst`, `examples/spline/…_splineContAbs`) should be removed so
> the gate uses the plain reference.
>
> **Which references 5.4 moves (measured 2026-08-03) — see Q5.20.** Not all of
> them, but not none either. `compare.compare_mod_out` ignores the `data read`
> block entirely and compares chi-squared, DOF and the model/convolution/shift
> sections; within a section `_compare_value_line` (`tests/compare.py:281-298`)
> compares **token by token**, failing on a differing token count, keyword name
> or suffix. So:
> - **Data-line fixes are free.** The `lsf` `=`->`:` echo and the new
>   `loadrange=` are both in the data line, which is not compared. **0 references
>   move.**
> - **Model-line fixes are not.** Dropping `locations=` changes the token count;
>   giving `damping=0.0000000` its fixed-status suffix changes the suffix. Both
>   trip the comparison. **6 of 41 references move:** the three `helium34`
>   (`Her36`, `HD319718`, `tet02OriA`) for damping, and
>   `examples/splineabs/{fit_spectra,fit_spectra_linear}` plus
>   `examples/spline/fit_spectra_splineContAbs` for locations.
>
> These are cheap to regenerate (the six are seconds-to-~75s each, not the
> 2.5-hour full harness), and the change is **cosmetic by construction** — the
> writer emits a different keyword token for the same fit. So verify rather than
> assume: after regenerating, diff each new reference against the old and confirm
> that *only* the `damping=`/`locations=` tokens differ, with chi-squared, DOF and
> every parameter value byte-identical. If anything else moves, the writer fix has
> changed the fit and must be reverted.
- **`lsf` keyword echo.** The writer emits `resolution=lsf(name=STIS,grating=…)`
  but the reader requires colons (`lsf(name:STIS,grating:…)`); re-reading a saved
  `.mod.out` crashes. Emit the reader's syntax. (`examples/lsf_hst`.) Confirmed
  2026-08-03 against `fit_spectra.mod.out.reference_adjusted`, whose only
  difference from the plain reference is exactly that `=` -> `:` swap **inside
  the `lsf(...)` parentheses only** — the surrounding `specid=`/`fitrange=`
  keywords keep their `=`. Note this is the sub-keyword separator inside a
  resolution function's argument list, so the fix belongs where the resolution
  string is composed (`save.py:537-540`), not in a blanket substitution.
- **`splineabs` `locations=` echo.** The writer emits an empty `locations=`
  keyword whose element count no longer matches the parameters; re-reading
  crashes. Emit a valid (or omitted) `locations=`. (`examples/…_splineContAbs`.)
  **Omission is the intended answer** — RJC's hand-fixed
  `fit_spectra_splineContAbs.mod.out.reference_adjusted` drops the keyword and
  its trailing whitespace entirely (2 lines, one live and one commented), so
  match that rather than synthesising a locations list. Verified 2026-08-03 by
  diffing the adjusted reference against the plain one.
- **Explicit default `damping=0.0000000`.** The writer prints the voigt default
  damping as a *suffixless* keyword, which on re-read becomes a **free**
  parameter (implicit default is fixed) and changes the DOF (helium34: DOF 618
  vs 621). The Stage 0 harness strips it as a workaround; fix the writer so the
  echo round-trips the free/fixed status, then drop the workaround
  (`alisrun.make_fixedparam_mod`).
- **Best-accepted vs rejected step. [NOT IN SCOPE — RJC, Q5.4: option (c).]**
  `examples/brokenpowerlaw` stays excluded from the fixed-parameter gate; the
  model is unlikely ever to be needed. Left here as a record of the suspected
  cause (the writer may emit a final *rejected* trial step rather than the best
  accepted point when convergence triggers on `atol`; Q0.12).
- **Covariance-PNG filename uses `str.rstrip` as a suffix strip.**
  `save.save_covar` (`alis/save.py:596`) derives the correlation-matrix image
  name at lines 657-658 with `filename.rstrip(fnspl[-1]) + 'png'`, where
  `fnspl = filename.split('.')`.
  `str.rstrip` removes any trailing characters *in that set*, not the suffix.
  It happens to be correct for every current covariance filename (the `.`
  terminates the strip), so this is not an active bug — but it breaks for a
  covariance name with no extension (`out covar mycovar` writes to `png`).
  Replace with `removesuffix`/`os.path.splitext` while tidying the writer.
  (Found during Stage 4 prep, 2026-07-30; deliberately deferred here.)
- **Fitted-vs-starting resolution in the echo.** The writer records the *fitted*
  resolution in the data line (e.g. `vfwhm(0.075va)`); because ALIS sizes the
  pixel-load buffer from the resolution at load time, re-reading loads a
  different pixel set.
- **Design superseded (Q5.5 -> Q5.16, RJC 2026-08-02).** Q5.5 originally asked
  for a data-line keyword defaulting to **30 extra pixels per side**. Measurement
  killed the default: the current buffer is not a pixel count at all, but
  **10 sigma of the instrumental profile in wavelength** (`vfwhm.getminmax`,
  `alis/functions/vfwhm.py:122`, called from `load.load_data` at
  `alis/load.py:867` and `:886`). That
  comes to 10-12 pixels for the narrow-resolution examples but **~3491**
  (`lsf_hst`) and **~4697** (`voigtconv`) for the broad ones — so a flat 30 would
  change the loaded pixels for *every* example (breaking the bitwise gate) and
  truncate the broad-LSF convolutions by two orders of magnitude. RJC agreed to
  all three parts of the replacement:
  1. **Write the concrete `loadrange` into `.mod.out`.** This is the actual fix
     for the round-trip: record the range that was *loaded*, so re-reading
     selects the same pixels whatever the fitted resolution became. It changes
     no first-run behaviour, so the Stage 0 gate stays **bitwise** and no
     reference moves.
  2. **Extra-pixels keyword as an optional override, with no default** — the
     resolution rule stays in charge unless the user asks otherwise.
  3. **Keep the "insufficient buffer" warning**, based on the resolution width
     rather than a flat 30.
  *Where the fix goes.* `save.save_model` builds the output data line at
  `alis/save.py:518-554` by copying the **input** line and substituting only
  `resolution=` and `shift=` (lines 537-546); every other keyword, `loadrange=`
  included, passes through verbatim. So today a model with no `loadrange=` in it
  writes none, and the re-read recomputes the buffer from the fitted resolution.
  The fix is to insert/replace `loadrange=[wmin,wmax]` in that loop from the
  loaded extent (`slf._wavefull[sp][ll:lu]`).
  *Expressiveness — checked 2026-08-03, the design holds.* A single `loadrange=`
  keyword per data line is only enough if a line maps to exactly one loaded
  range. Measured by instrumenting `load_data` and counting `_posnfull` snips
  against live data lines:

  | model | data lines | snips |
  |---|---|---|
  | metal_line_abs, lsf_hst, voigtconv, CNabs, summed_coldens, lls | 1 | 1 |
  | DH/Q1243p307 | 51 | 51 |
  | DH_orders/Q1243p307 | 351 | 351 |

  One data line = one snip in every case, and the writer's loop emits one output
  line per input line (`spmatch` breaks on the first `specid` keyword, and
  specids are unique), so the mapping is 1:1 end to end.

**5.2 — Atomic data modernisation.**
- Replace `atomic.xml` with a human-readable, whitespace-aligned ECSV table
  (`astropy` is already a dependency): self-describing header, obvious rows, no
  manual `nrows`. Provide a converter from the current `atomic.xml`, and
  validation (units, duplicates).
- **Decided (Q5.6, RJC 2026-08-01; amended by Q5.17, RJC 2026-08-02):**
  `atomic.ecsv` becomes the **default**, and `atomic.xml` is kept in the
  repository. Q5.6 originally said the XML would no longer be read at all; Q5.17
  supersedes that — **the VOTable reader stays on the load path** so a user's own
  `.xml` still works, and reading one emits a **deprecation warning**. That
  warning should tell the user what to do instead: ask RJC to include the new
  atomic data in the shipped ECSV, or run the converter.
- **Ship the converter as a script in `alis/data/`** (Q5.17), not as a one-off
  used to produce `atomic.ecsv` and then discarded — users need it to bring their
  own XML forward. It is also what the row-by-row equality check below runs on.
- **Merge decision — settled (Q5.8, RJC 2026-08-01). `atomic.ecsv` is
  `atomic.xml` plus exactly 15 rows, and nothing else:**
  - take the **15 3He I lines** from `atomic_rjc2.xml` (now at
    `context/atomic/atomic_rjc2.xml`, see the audit note below) — 515.6440,
    522.2406,
    537.0581, 6680.4953/4974/4977, and 7067.0362/1114/1159/1225/1455/2207/2318/
    5795/6888 (the 3He isotope structure of He I lambda-6678 and lambda-7065);
  - **no other 3He change** — the rest of the default's 3He is correct;
  - **do not** add `1H I 911.7633` (the Lyman-limit marker row);
  - **keep the default's O I lambda-918-920 wavelengths**; the new files' O I is
    to be ignored;
  - merge nothing on the strength of a differing MassNumber — those rows are
    duplicates mislabelled with radioactive isotopes;
  - `atomic.xml` is otherwise authoritative for every value.
- **Repoint the dead `run atomic` lines (Q5.13, option (b)).** The ten context
  models that name `atomic_rjc.xml` should be changed to `run atomic
  atomic.ecsv`, which demonstrates the setting and keeps them working. Verify
  the Stage 0 gate stays green across that change: the ECSV differs from the
  `atomic.xml` they actually load today only by the 15 3He rows, which none of
  those fits uses.
- **Constraint (RJC, Q5.14): the conversion must not change a single value.**
  `atomic.xml` -> `atomic.ecsv` is a faithful transcription plus the 15 3He rows
  already agreed, and nothing else. The conversion needs a row-by-row equality
  check against the XML so a transcription error cannot slip in.
  **Amended by 5.6 (2026-08-02):** the second half of this bullet used to say the
  isotopes carrying two different `AtomicMass` values (24Mg, 28Si, 47Ti, 56Fe)
  were to be left as they are. That is no longer the state of the file — 5.6
  changed 80 rows so each isotope carries **one** mass (RJC chose the isotopic
  value in all four cases). Transcribe the **current** `atomic.xml`; the
  duplicate-mass condition no longer exists to preserve.
- **Do not regress 5.6 while rewriting the loader.** `load_atomic` now builds
  `Element`/`AtomicMass` as a single isotope -> mass mapping; the ECSV path must
  produce the same mapping, not two independently de-duplicated lists.
  `tests/test_atomic_mass.py` is deliberately split so most of it survives this
  change untouched — only the **format-specific** half (which parses the VOTable
  directly, via `_file_masses`) needs a matching ECSV reader adding. The
  format-agnostic half, which recovers the mass from the broadening ALIS actually
  computes, must pass **unaltered** across the switch; if it needs editing to
  pass, the conversion has changed a value.
- Validation must include a **duplicate check keyed on Element+Ion+RestWave
  ignoring MassNumber** — that is exactly the check that would have caught the
  mislabelled isotopes. It should *report*, not modify.
- **Tidy the data-directory lookup while in there.** `load_atomic`
  (`alis/load.py:432-443`) locates `alis/data/` by string-splitting
  `argflag['run']['prognm']` on `'/'` and re-joining everything before the last
  separator. It works, but it is path-separator-specific and makes `prognm` look
  like it must name a real file when only its directory is ever used — to the
  point that `tests/test_atomic_mass.py` sets it to `alis/alis.py`, which has not
  existed since the Stage 2 reorganisation, and the tests pass anyway. Replace
  with `os.path.dirname` / `importlib.resources`, and fix the test's path so it
  stops advertising a dead filename. Keep `run atomic <file>` resolving both an
  explicit path and a bare name in `alis/data/`, exactly as now.

**5.3 — Plotting-script output.**
- Add an option to emit a **self-contained** matplotlib script that reproduces
  the publication-quality figure for a fit, so users can tweak the plot outside
  ALIS. Reference figures: `context/plotting/` (RJC, Q5.7). Note those reference
  scripts `import plotting_routines as pr`, which is **not** in the directory —
  "self-contained" means the emitted script must not depend on it, so whatever
  it needs from `pr` has to be inlined.
- **Settings: a new `plotscript` section (Q5.18, RJC 2026-08-02).** Not a single
  `out plotscript` value — RJC asked for a section with customisable sub-settings,
  because these all become editable from the interactive GUI in Stage 6.6, and
  building the surface here is the point. The settings parser needs no change to
  support this: `load.set_params` (`alis/load.py:186-229`) dispatches on
  `linspl[0] in argflag.keys()`, so adding a `PlotScriptConfig` dataclass to
  `ArgFlag` (`alis/config.py:320-334`, alongside `run`/`chisq`/`plot`/`out`/
  `sim`/`generate`) makes `plotscript format DH` parse automatically.
  - `plotscript format none|metals|DH` — **default `none`**; anything else emits
    a script.
  - `plotscript numcol auto|<int>` — `auto` uses the per-format default (3 for
    `metals`, 2 for `DH`).
  - `plotscript residuals True|False` — residual strip under each panel.
  - `plotscript velrange auto|<vmin>,<vmax>` — e.g. `-150,150`.
  - RJC invited further settings. Suggested, all defaulting to the current
    hard-coded behaviour of the `context/plotting/` references so the emitted
    script is unchanged unless asked: `plotscript filename` (output path, default
    `<model>.mod.plot.py`), `plotscript ylim`, `plotscript fontsize`,
    `plotscript figsize`, `plotscript labels True|False` (ion/transition
    annotation per panel), and `plotscript overwrite True|False` matching
    `out overwrite`.
  - *Parser note for typed values:* `set_params` reads exactly one whitespace-
    delimited token (`linspl[2]`) and converts by the **type of the default**. A
    field that must accept both `auto` and `3`, or both `auto` and `-150,150`,
    should default to `None` — that branch (lines 211-223) infers list/float/int/
    str — with normalisation in `check_argflag`. Values containing a space will
    not parse, so keep them comma-separated.
- **Modes (Q5.7/Q5.9):**
  - `none` — emit nothing;
  - `metals` — panels **three columns** wide, as many rows as needed, with the
    same element and ion in adjacent panels;
  - `DH` — **two columns**, with the meaning of the columns depending on how
    many datasets cover the transitions (Q5.9 + Q5.11):
    - **one dataset**: the Lyman series runs *down* the left column and
      continues *down* the right — top-left Ly-alpha, Ly-beta, ... to about Ly7;
      top-right Ly8, Ly9, and on. The split point moves with how many Lyman
      lines the fit contains.
    - **two datasets**: the columns are the *datasets* and the rows are the
      series — Ly-alpha of dataset 1 top-left and of dataset 2 top-right,
      Ly-beta on the second row, and so on, so the same transition can be
      compared across datasets at a glance. This is what
      `context/plotting/DH_Lya-Ly7.py` already does with `GridSpec(7, 2)`.
    - **more than two datasets**: the user chooses which to plot, by editing the
      emitted script. Co-adding (which is what DH_orders would want) is
      **explicitly not required now** — RJC will revisit it with the interactive
      GUI (Stage 6.6).
- Panels follow the `context/plotting/metals*.py` convention: velocity space
  about each transition, flux normalised by the continuum with the zero level
  subtracted, dashed continuum and zero-level guides.
- A Stage 6 task has been added (6.6) for generating this interactively from the
  GUI after a fit completes.

**5.6 — Atomic masses are read off by one.
[DONE 2026-08-03 — RJC confirmed the bug independently and chose the isotopic
mass for all four conflicting isotopes. `atomic.xml` made self-consistent (80
rows), `load_atomic` now builds Element/AtomicMass as one mapping, and
`tests/test_atomic_mass.py` (20 unit tests, in two layers so they survive the
ECSV switch) asserts it end to end. Unit batch 470 -> 490.
**References regenerated by RJC** with the new `regen_harness.sh` (42 cases), and
the **full harness is green**: `pytest --run-slow` gives 613 passed, 31 skipped,
0 failed in 2:23:48 — including the `slow`, `gpu` (40/40, `ngpus` 1 and 4) and
`machine_dependent` batches. All 31 skips are the structural "no GPU
implementation" ones. See Q5.15 and `logs/refactor_code_stage5_log.md`.]**
> RJC's Q5.14 response is that the lookup is by element name, so a shared mass
> number cannot cause a problem. That is what the code *intends*; the measurement
> below is what it *does*. Re-checked end-to-end rather than by reading the code.

- **Probe inserted at the point of use** (`alis/functions/voigt.py:336`, the line that computes
  the Doppler width) and `examples/metal_line_abs` run normally:

```
[MASSPROBE] ion=16O_I    mass_used=20.18            T=8000.0
[MASSPROBE] ion=28Si_II  mass_used=27.9769265325    T=8000.0
```

  16O is given **20.18** — neon's standard atomic weight. Oxygen is 15.9994.
- **Mechanism.** `load_atomic` does not build a per-element mass array. It builds
  two lists de-duplicated *independently*: `Element` from the isotope strings and
  `AtomicMass` from the mass *values*. Lookup is then
  `m = where(Element == ion); mass = AtomicMass[m][0]` — an index from one list
  used in the other. They align only by coincidence. Three isotopes (`1TL`,
  `1H`, `1Ly`) share one mass value, which costs two entries and offsets
  everything after them; the isotopes that carry *two* mass values (24Mg, 28Si,
  47Ti, 56Fe) add entries back, which is why 28Si happens to come out right.
- **13 of the 14 ions used by the shipped examples and context fits get the
  wrong mass** (12C, 13C, 14N, 16O, 1H, 24Mg, 27Al, 28Si, 2H, 32S, 3He, 4He,
  56Fe wrong; 58Ni right).
- **Why this has plausibly never shown up.** What a fit constrains is the
  *ratio* of thermal widths at a shared temperature, and for H vs D the error
  almost cancels — 1H and 2H are both shifted to the next isotope up:

| ratio | ALIS | correct | error |
|---|---|---|---|
| b(1H)/b(2H) | 1.4097 | 1.4137 | **-0.3%** |
| b(1H)/b(16O) | 3.1653 | 3.9844 | **-20.6%** |
| b(1H)/b(12C) | 2.6368 | 3.4506 | -23.6% |
| b(12C)/b(16O) | 1.2005 | 1.1547 | +4.0% |

  So D/H — the flagship use, and the one RJC's intuition is calibrated on — is
  insulated to 0.3%. It is **H-versus-metal** at a tied temperature where the
  relative widths are wrong by ~20%, which is exactly what the `TA`-tied fits do.
- **The fix is small** (build one isotope -> mass mapping instead of pairing two
  lists) but changes numbers, so every Stage 0 reference moves. It does **not**
  require editing the atomic data, which RJC has ruled out — the data are fine;
  it is the loader that mis-pairs them.

**5.5 — Unit tests for this stage's stable surface (do last).**
- Following the cross-cutting unit-test policy
  (`claude_prompts/refactor_code_unit_tests.md`), add `unit`-marked tests for the
  Stage 5 stable surface once the I/O is reshaped: this is where the deferred
  file-format loaders (`load_fits`/`load_ascii`/`load_data`/model parsing) get
  their unit tests, using small synthetic fixtures, plus the atomic-data
  loader/converter (5.2), the plotting-script emitter (5.3), and the
  output-writer round-trip helpers (5.4). (The YAML/TOML reader/writer that
  originally appeared here is gone with 5.1.) Keep them fast and isolated (no
  full fits); the existing `unit` CI job picks them up automatically.
- Include a **`.mod` -> `.mod.out` -> `.mod` round-trip test** over the shipped
  examples: it is the thing 5.4 is fixing, and the Stage 0 harness's
  `_ZERO_DAMPING_RE` / `reference_adjusted` workarounds only exist because
  nothing pins it. Cover the Stage 4 settings the writer now echoes
  (`run backend`/`ngpus`/`gputhresh`/`shmem`) while doing so.

**5.1 — YAML/TOML model files. [DROPPED — RJC, 2026-08-01]**
- RJC's answers to Q5.2/Q5.3 keep `.mod` as the format, and the side-by-side he
  asked for (appendix below) supports that: YAML would leave the `1.0da`
  micro-syntax hand-parsed anyway, expand DH_orders from 1252 to ~4000 lines,
  and add a second reader/writer pair to keep in agreement with the first —
  which 5.4 exists because it does not yet round-trip faithfully.
- Original scope, for the record: YAML/TOML parsing and writing mapping to the
  same internal representation, text remaining the default.
- Revisit only if models need to be generated/consumed by other tools; the
  cheaper answer then is a documented `.mod` <-> dict API in `alis/load.py`
  (which 5.5's parser tests will pin down anyway), not a second file format.


## Audit before commencing (2026-08-01)

Filenames and symbols in this doc were checked against the code as it now
stands (end of Stage 4).

**Corrected:** `alload.py` -> `alis/load.py`, `alplot.py` -> `alis/plot.py`
(renamed in Stage 2). Everything else named here still exists: `alis/save.py`'s
`save_covar`, `load_data`/`load_ascii`/`load_fits` in `alis/load.py`,
`alisrun.make_fixedparam_mod`, `alis/data/atomic.xml`, `doc/ALIS_workflow.md`
(the cited sections are S5 "Output Files" and S10 "Appendix: Atomic Data File"),
`claude_prompts/refactor_code_unit_tests.md`, and the `atomic-data` /
`run-tests` skills.

**Every 5.4 defect re-verified as still present** -- none was fixed in passing
by Stages 1-4:

| item | evidence today |
|---|---|
| `lsf` keyword echo | reference has `lsf(name=STIS,grating=E140H,slit=0.2x0.09)`, the hand-adjusted re-input has `name:`/`grating:`/`slit:` |
| `splineabs locations=` echo | reference emits a trailing empty `locations=`; the adjusted file drops it |
| explicit `damping=0.0000000` | `_ZERO_DAMPING_RE` workaround still live in `tests/alisrun.py:71` |
| best-accepted vs rejected step | `examples/brokenpowerlaw` still in `FIXEDPARAM_EXCLUDE` |
| `rstrip` as suffix strip | `alis/save.py:657-658` unchanged |
| both `.mod.out.reference_adjusted` files | still present (`examples/lsf_hst`, `examples/spline/...splineContAbs`) |

`atomic.xml` still carries a hand-maintained `nrows="815"` on its `<TABLE>`
element, so 5.2's motivation stands.

**Stage 4 additions the writer now echoes**, which 5.4's round-trip work should
cover: `run backend`, `run ngpus`, `run gputhresh`, `run shmem`. All are
ordinary settings and re-read cleanly, so no new defect -- but any round-trip
test written in 5.5 should include them.

### Audit addendum (2026-08-01, second pass): the new `atomic_*.xml` files broke the suite

The three files RJC added to `alis/data/` were **load-bearing in the wrong
direction**. Ten `.mod` files across seven context objects — every DH and
VMP_DLA fit — carry `run atomic atomic_rjc.xml`, and `load.load_atomic` resolves
that name against `alis/data/`. While the file was absent those fits warned and
silently fell back to `atomic.xml`; the moment it was present they resolved to
it and **failed outright**:

```
[ERROR] :: Element 1Ly not found in atomic data file
```

`atomic.xml` and `atomic_rjc2.xml` carry the `Ly` pseudo-element (`1Ly`);
`atomic_rjc.xml` and `atomic_mtm.xml` do **not**. So simply dropping the files
into the package directory broke DH_orders, DH/HS0105p1619, DH/J0814p5029,
DH/J1358p6522, DH/Q1243p307, VMP_DLA/J0814p5029 and VMP_DLA/J1358p6522.

**Action taken (Q5.12 gave permission to move them):** the three files are now
in **`context/atomic/`**, outside the package. DH_orders was re-checked and is
back to its reference chi-squared of 92636.74284838137. This also settles the
packaging concern — `pyproject.toml` ships `alis = ["data/*"]` wholesale, so
they would otherwise have gone into every wheel.

**Consequence worth noting:** those ten models *request* `atomic_rjc.xml` and
have never actually received it — their committed references were produced with
`atomic.xml` (they could not have been produced with `atomic_rjc.xml`, which
crashes them). Since the two files disagree on the O I lambda-918-920
wavelengths by up to 4.5 km/s, the `run atomic` lines are not only dead but
misleading. See Q5.13.

### Audit addendum (2026-08-02, third pass)

Re-checked after 5.6 landed. Everything the doc names is still correct:
`save.save_covar`'s `rstrip` is still at `alis/save.py:657-658`, the
`_ZERO_DAMPING_RE` workaround is still at `tests/alisrun.py:71`, both
`.mod.out.reference_adjusted` files still exist, `brokenpowerlaw` is still
excluded, and `load_data`/`load_ascii`/`load_fits` are unchanged. The three
`atomic_*.xml` are now in `context/atomic/`.

**One consequence of the regeneration worth recording for 5.4.** The two
`.mod.out.reference_adjusted` files are hand-made and cannot be produced by a
run, so `regen_harness.sh` leaves them alone -- which left them holding the old
parameters while the plain references held the new ones, and the mode-(b)
`lsf_hst` test failed until they were refreshed by re-applying the same hand-fix
to the new plain reference. 5.4 removes the need for both files, and should
delete them.

## Appendix (answers Q5.3) -- what `metal_line_abs` looks like in YAML

RJC asked for a side-by-side before deciding on 5.1. Here is the whole of
`examples/metal_line_abs/model/fit_spectra.mod` (26 lines) rendered two ways.

### The current `.mod` (26 lines)

```
run blind False
run datadirc ../data/
chisq ftol 1.0E-10
chisq atol 0.001
chisq miniter 10
chisq maxiter 1000
out fits True
out plots fit_spectra.pdf

data read
  OI_SiII.dat  specid=0  fitrange=[1301.0,1305.0]  resolution=vfwhm(7.0VA)  columns=[wave,flux,error]  label=OI_SiII
data end

model read
  fix voigt temperature True
  emission
    legendre 1.0   0.01   0.01    scale=[1.0,1.0,1.0]   specid=0
  absorption
    voigt   ion=16O_I   14.0    0.0    1.0da   8000TA   specid=0
    voigt   ion=28Si_II 13.0    0.0    1.0da   8000TA   specid=0
model end
```

### (a) "Thin" YAML -- keeps the parameter micro-syntax (~40 lines)

```yaml
run: {blind: false, datadirc: ../data/}
chisq: {ftol: 1.0e-10, atol: 0.001, miniter: 10, maxiter: 1000}
out: {fits: true, plots: fit_spectra.pdf}

data:
  - file: OI_SiII.dat
    specid: 0
    fitrange: [1301.0, 1305.0]
    resolution: vfwhm(7.0VA)
    columns: [wave, flux, error]
    label: OI_SiII

model:
  fix: [[voigt, temperature, true]]
  emission:
    - {function: legendre, parameters: [1.0, 0.01, 0.01],
       scale: [1.0, 1.0, 1.0], specid: 0}
  absorption:
    - {function: voigt, ion: 16O_I,   parameters: [14.0, 0.0, "1.0da", "8000TA"], specid: 0}
    - {function: voigt, ion: 28Si_II, parameters: [13.0, 0.0, "1.0da", "8000TA"], specid: 0}
```

### (b) "Explicit" YAML -- every parameter self-describing (~50 lines for 3 components)

```yaml
  absorption:
    - function: voigt
      ion: 16O_I
      specid: 0
      parameters:
        ColDens:     {value: 14.0}
        redshift:    {value: 0.0}
        bturb:       {value: 1.0,  tie: da}
        temperature: {value: 8000, tie: TA, fixed: true}
```

### Assessment

**Recommendation: do not do 5.1. RJC'''s instinct in Q5.2/Q5.3 is right, and
these are the reasons rather than just agreement.**

1. **The gain YAML normally brings -- a standard parser -- is mostly illusory
   here.** ALIS'''s expressiveness lives in the token `1.0da`: value, free/fixed
   (by case), and tie-label, in one word. Variant (a) keeps that as a *string*,
   so `load.py` still hand-parses it; YAML has parsed the punctuation but not
   the thing that is actually hard. Variant (b) removes the micro-syntax but at
   4 lines per parameter.
2. **Size goes the wrong way on the fits that matter.** metal_line_abs: 26 ->
   ~40 lines (a) or ~50 (b). DH_orders, the real workload, is 1252 lines with
   351 data lines and 138 model components; in (a) each data line becomes ~9
   YAML lines, giving **~4,000 lines**, and in (b) the model block alone
   multiplies by ~5. A `.mod` model block is a *table* of absorption lines,
   column-aligned, which is how the components are compared by eye.
3. **It is additive, not a replacement.** The stage brief requires text support
   forever, so YAML means two readers and two writers kept in agreement, plus
   round-trip tests both ways in 5.5 -- on top of 5.4, which exists because the
   *one* writer ALIS already has does not round-trip faithfully. Fixing that
   first is worth more than adding a second format.
4. **No user demand is recorded.** Q5.2/Q5.3 are the only mentions, and RJC'''s
   answer to both is that `.mod` is fine.

**What would change the answer:** a need to generate or consume models
programmatically from other tools (pipelines, a GUI, a survey system). If that
arrives, the cheaper route is a documented `.mod` <-> dict API in `load.py` --
which 5.5'''s parser unit tests will effectively pin down anyway -- and let
callers serialise that dict to whatever they like, rather than a second
first-class file format inside ALIS.

## Appendix (answers Q5.6) -- the three new `atomic_*.xml` files

Compared against the default `atomic.xml` (815 transitions, 168 species).
Additions only, as asked. `atomic_mtm.xml` has no `SolarAbundance` column; the
other two match the default schema.

| file | transitions |
|---|---|
| `atomic.xml` (default) | 815 |
| `atomic_mtm.xml` | 513 |
| `atomic_rjc.xml` | 545 |
| `atomic_rjc2.xml` | 587 |

**Headline: almost nothing in these files is new atomic data.** All three are
*smaller* than the default, and once the isotope label is set aside essentially
every transition already exists in `atomic.xml`: 510 of 513 (mtm), 541 of 545
(rjc), 577 of 587 (rjc2) match a default line of the same Element+Ion at the
same wavelength.

**The apparent "new species" are a MassNumber off-by-one, not new physics.**
The new files label lines with a mass number one below the stable isotope the
default uses:

| element | default | new files | stable isotope |
|---|---|---|---|
| Na | 23 | 22 | 23 |
| Al | 27 | 26 | 27 |
| Mn | 55 | 54 | 55 |
| Fe (mtm only) | 56 | 55 | 56 |
| O (mtm only) | 16 | 15 | 16 |
| Cr | 50/52/53/54 | 51 | 52 |
| Zn | 64/65/66/67/68/70 | 65 | 64/66 |

So `atomic_mtm.xml`'s apparent "18 new species / 115 new transitions" --
including a complete 15O set -- is a labelling artefact. 15O has a two-minute
half-life, as do 22Na, 26Al, 54Mn and 55Fe; none belongs in a QSO absorption
line list. **Do not merge these as new species.**

### The one substantive addition: 15 3He I lines, in `atomic_rjc2.xml` only

The default already carries 28 3He I lines and rjc2 carries 25, so rjc2 is *not*
a superset -- it is an overlapping compilation. Fifteen of its 3He I lines are
at wavelengths the default does not have:

```
3He I   515.6440  f=0.015045     3He I  7067.0362  f=0.069519
3He I   522.2406  f=0.029873     3He I  7067.1114  f=0.069519
3He I   537.0581  f=0.073460     3He I  7067.1159  f=0.069518
3He I  6680.4953  f=0.284110     3He I  7067.1225  f=0.069518
3He I  6680.4974  f=0.189410     3He I  7067.1455  f=0.069519
3He I  6680.4977  f=0.236760     3He I  7067.2207  f=0.069519
                                 3He I  7067.2318  f=0.069518
                                 3He I  7067.5795  f=0.069528
                                 3He I  7067.6888  f=0.069528
```

These are the 3He isotope/hyperfine structure of He I lambda-6678 and
lambda-7065, where the default has only the 4He components -- exactly what a
3He/4He measurement needs (cf. the `helium34` context fits). **This is the part
worth merging.**

*Method note:* the first pass matched wavelengths within 0.05 A and reported
only 7 new He lines. That tolerance is too coarse for isotope structure, whose
components sit *inside* 0.05 A of the parent line by construction; the 15 above
come from an exact-wavelength comparison restricted to 3He.

### Also new, in all three: `1H I 911.7633, f = 6.3e-18`

The Lyman limit, with a vanishingly small oscillator strength -- evidently a
series-limit/continuum marker rather than a transition. `atomic_rjc.xml` and
`atomic_rjc2.xml` also carry it under the `IB` ion label. Worth a decision on
whether ALIS wants such a marker row at all.

### Not additions, but disagreements that matter more than the additions

**O I lambda-918-920 -- the default and all three new files disagree.** Same
four lines, identical f-values, different wavelengths:

| f | `atomic.xml` | new files | difference |
|---|---|---|---|
| 6.14e-4 | 918.0531 | 918.0440 | 0.0091 A (3.0 km/s) |
| 1.32e-4 | **918.2341** | **919.2220** | **0.99 A (322 km/s)** |
| 7.92e-4 | 919.6717 | 919.6580 | 0.0137 A (4.5 km/s) |
| 1.77e-4 | 919.9142 | 919.9170 | 0.0028 A (0.9 km/s) |

The second row is a ~1 A disagreement on a line the f-values say is the same
transition -- one of the two is a transcription error. The others are 3-4.5
km/s, which is not negligible for D/H work. This needs resolving on its merits,
independently of any merge.

**Na I lambda-3303 doublet.** mtm and rjc give 3303.3690/3303.9780; the default
and rjc2 give 3303.3190/3303.9290 -- 0.05 A, 4.5 km/s apart. rjc2 agrees with
the default, so mtm/rjc look like the older values.

**Smaller refinements.** 35 (mtm), 69 (rjc), 69 (rjc2) lines differ from the
default by under 0.05 A. Median shift 0.05-0.14 km/s, but 9 lines in rjc/rjc2
exceed 1 km/s (all O I, above) and one in mtm does (B II 1362.473 vs 1362.463,
2.2 km/s).

### Recommendation

1. Merge the **15 3He I lines from `atomic_rjc2.xml`**.
2. Decide separately on the `1H I 911.7633` marker row.
3. Merge **nothing** on the strength of a differing MassNumber -- those rows are
   duplicates of lines ALIS already has, mislabelled with radioactive isotopes.
4. Treat the **O I lambda-918-920 wavelengths** as a data-quality question to
   settle before 5.2 freezes the ECSV, since it shifts fitted redshifts.

## Skills to use for this stage

- `atomic-data` — add/validate/convert atomic entries.
- `run-tests` — Stage 0 gate (parsing parity between text and YAML/TOML).

## Context

- `alis/load.py` (`.mod` parser), `alis/data/atomic.xml` and its VOTable format,
  `alis/plot.py` (current plotting). (Both were named `alload.py`/`alplot.py`
  when this doc was written; Stage 2 renamed them.)
- `doc/ALIS_workflow.md` §"Atomic Data File" and §"Output Files".
- Plan Q7/Q13 (ECSV candidate; whitespace-separated, easy to add/validate).

## Queries

**Q5.1 — Atomic format confirmation.** Confirm ECSV for the atomic data (per Q13),
and whether the old `atomic.xml` should be retained as a supported input during a
deprecation period or replaced outright (with a one-off converter).

**Response:** ECSV is a good choice for the atomic data file format. It is human-readable,
self-describing, and aligns well with the existing dependencies in ALIS. The old `atomic.xml`
can be retained as a supported input during a deprecation period to allow users time to
transition to the new format. A one-off converter should be provided to facilitate this
transition.

**Q5.2 — YAML/TOML dependencies.** YAML support needs `PyYAML` at runtime for
users who choose YAML, and TOML *writing* needs a small writer lib (e.g.
`tomli-w`) — both would be new runtime/optional deps (plan Q12). Do you want:
(a) TOML-only (stdlib read, optional writer), (b) YAML too (add `PyYAML` as an
optional extra), or (c) both as optional extras?

**Response:** I think the current .mod files are OK, and we don't need to switch to
YAML/TOML unless there is a strong reason to change the file format. If there's no
clear need to using YAML, then I would leave it as is.

**Q5.3 — Schema.** Should the YAML/TOML schema be a faithful 1:1 mapping of the
text `.mod` structure (settings / data / model / link blocks), or a cleaner
redesigned schema (with a converter between them)?

**Response:** I think we should continue to use the current .mod file format as
the primary format. If you strongly think YAML is a better alternative, could
you please provide a comparison example of the two formats (perhaps alter the
`metal_line_abs` example to show how it would look in YAML)? This will help us evaluate the
benefits of switching to YAML and whether it is worth the effort to implement. I will
then decide if we should pursue this conversion, or leave the .mod format as is.

**Q5.4 — Is the "best-accepted vs rejected step" item in scope here?** 5.4 flags
`examples/brokenpowerlaw` (reference evaluates to chi2 374.98 against a recorded
338.15) as a *fitting-engine* concern and says to "coordinate with Stage 3" —
but Stage 3 is complete. Do you want me to (a) investigate and fix it in
`minimise.py` during Stage 5, (b) diagnose only and write it up as a Stage 3
follow-up task, or (c) leave brokenpowerlaw excluded from the fixed-parameter
gate as it is now? My recommendation is (b): the diagnosis is cheap, but
changing which parameter vector the minimiser reports is a numerics change that
would move every reference, and that deserves its own gated task rather than
riding along inside an I/O stage.

**Response:** For now, I propose option (c), since the `brokenpowerlaw` model is unlikely to ever be needed.

**Q5.5 — May 5.4 change which pixels a re-read model loads?** The last 5.4 item
asks whether the loader should size the pixel-load buffer independently of the
(fittable) resolution. It should — a saved model reloading a different pixel set
is the underlying bug — but doing so changes the loaded data for the affected
examples, so their references would have to be **deliberately regenerated** and
the Stage 0 gate would not be bitwise across that commit. Is that acceptable
for this specific fix (with the regeneration called out in the log), or would
you rather leave the loader alone and keep the per-example `vfwhm` workaround?

**Response:** We could introduce a new parameter in the model file to specify the
number of additional pixels to load on either side of the fit range. This makes it
independent of the fittable resolution and allows users to control the pixel load
buffer size. We can keep the existing `vfwhm` workaround for now, but this should
only be used as a check. A warning message should be provided if the number of
pixels loaded is insufficient for the fit range, and users should be encouraged to
adjust the parameter accordingly. This way, we can avoid changing the loaded data
for the affected examples and keep the Stage 0 gate bitwise across commits. It
also adds a check that will ensure users are made aware if the loaded number of
pixels might be insufficient for the fit range. By default, the number of extra
pixels should be set to 30 on each side of the fitrange. If 30 pixels is not
available, then a warning message should also be provided to the user.

**Q5.6 — When does ECSV become the shipped default?** Your Q5.1 answer keeps
`atomic.xml` readable during a deprecation period. Do you want 5.2 to (a) ship
`atomic.ecsv` as the default and keep the XML reader as a fallback, or (b) ship
both but keep XML as the default until a later release? (a) is what exercises
the new path in every run and every test; (b) is more conservative. I lean (a),
with the XML reader kept and unit-tested so an existing user file still loads.

**Response:** I recommend that we don't delete the file, but it won't be loaded
or used for the time being. We can keep it as a reference for users who may want
to see the old format, but the new ECSV format will be the default for all operations.
Also, I have included three new `atomic_*.xml` files in the `alis/data/` directory.
Could you please inspect these files (`atomic_mtm.xml`, `atomic_rjc.xml`, `atomic_rjc2.xml`), and compare them to the `atomic.xml` file to
see what is different between all of these atomic data? Provide a summary of the
differences, and I will make a decision if any of these should be merged into
the `atomic.ecsv` file. Note, the default is `atomic.xml`, and I am interested to
know if there are any atomic data that are in one of the new files but not in the
default `atomic.xml` file. I am not interested in knowing about the atomic data
that are in the default `atomic.xml` file but not in the new files.

**Q5.7 — What should the 5.3 plotting script contain?** Two shapes: (a) a script
that reads the `_fit.dat` files ALIS already writes (small, but only works
beside those files), or (b) a self-contained script with the data embedded as
arrays (portable, one file to send a collaborator, but large for DH_orders-sized
fits). Also: should it reproduce the multi-panel `plot dims` layout, or emit one
figure per region? I lean (a) plus the existing multi-panel layout, since that
is the figure users already recognise.

**Response:** I have provided some context on the plotting script in the
`context/plotting/` directory. The plotting script should be self-contained and
should produce something similar to these files. There should be three modes
offered initially: (1) `none`, meaning nothing is output; (2) `metals` which will
display metals in a series of panels three columns wide, and however many rows
are required to provide all panels. Note that elements and ions should be grouped
together, so that the same element and ion are shown in adjacent subpanels. Please
also add a Task to the Stage 6 GUI to include a functionality that the user may
interactively generate this script after the fitting procedure is complete. We
will discuss further details of this functionality in Stage 6.

**Q5.8 — Two atomic-data questions the comparison raised (see the appendix).**
(a) All three new files carry `1H I 911.7633` with `f = 6.3e-18` — the Lyman
limit, effectively a marker row rather than a transition. Do you want it in
`atomic.ecsv`? (b) More importantly: the default and *all three* new files
disagree on the O I lambda-918-920 wavelengths, including one line where the
f-values say it is the same transition but the wavelengths differ by 0.99 A
(918.2341 vs 919.2220, ~322 km/s) — one of them must be a transcription error.
The other three differ by 0.9-4.5 km/s. Which set should the ECSV carry? This
shifts fitted redshifts, so I would rather not choose it for you.

**Response:** Here is a response to your questions:
(a) No, let's not include it in `atomic.ecsv`.
(b) The O I lines are trustworthy in the atomic.xml file. The O I lines from `atomic.xml` should be added to the ECSV file, and the O I lines in the new files should be ignored.
For the 3He I lines, we should include the 15 new lines from `atomic_rjc2.xml` in the ECSV file (The 515, 522, 537, 6680.*, 7067.*). No other 3He I lines should be added/changed (The default `atomic.xml` is otherwise correct). No other changes are needed at this stage. We should otherwise consider `atomic.xml` as the default values.

**Q5.9 — The third plotting mode.** Your Q5.7 answer says "three modes offered
initially" but names two (`none`, `metals`). Looking at `context/plotting/`, the
scripts split into a `metals*.py` family and a `DH_*.py` / `blends_*.py` family
(Lyman-series panels for D/H). Is the third mode the Lyman-series one — and if
so what should it be called and how should its panels be laid out (one column
per transition of the series, or the same three-wide grid)?

**Response:** My apologies for missing that! You are correct, the third mode should
be called something like D/H, and it should contain two columns. The top left should
show Lya, then the second on the left should show Lyb, and so forth, likely down to
Ly7. Then, the top right shows Ly8, the second on the right should show Ly9, and so
forth down the rest of the Lyman series. It may be necessary to change the exact numbers,
depending on how many Lyman series lines are included in the fit, but there should be
two columns and structured something similar to that listed above.

**Q5.10 — Ordering now that 5.1 is dropped.** With YAML/TOML gone, the stage is
5.2 (atomic/ECSV), 5.3 (plotting script), 5.4 (writer round-trip), 5.5 (tests).
5.4 is the one with a user-visible correctness payoff (a saved `.mod.out` that
re-reads faithfully) and it is what 5.5's round-trip tests need in place. Shall
I reorder to **5.4 first**, then 5.2, then 5.3? Or keep the numbering as it is?

**Response:** Yes, please change the task order according to your suggestion.

**Q5.11 — D/H panels when several datasets cover the same transition.** Your
layout runs the Lyman series down the left column and on down the right. But
`DH_orders` — the flagship D/H fit — has 351 spectra, several covering the *same*
transition from different spectrographs/orders (the `prochaska` / `kirkman`
labels), which is exactly why `context/plotting/DH_Lya-Ly7.py` uses its second
column for a second dataset. With both columns spent on the series, where should
the extra datasets go? Options: (a) co-add/overplot them in the one panel for
that transition, (b) one panel per dataset, flowing down the two columns so a
transition may occupy several panels, or (c) plot only a nominated dataset and
let the user edit the emitted script. I lean (a) for the default with the script
written so (b) is an easy edit — but this is your figure convention, not mine.

**Response:** That's correct. If there's more than one dataset I usually plot the
same transition from different datasets side by side. So, Lya of one dataset would be
on the top left while Lya of the second dataset would be on the top right. Then, Lyb
of the first dataset would be on the second row left, and Lyb of the second dataset
would be on the second row right, and so forth. This way, the user can easily compare the
same transition from different datasets. If there are more than two datasets, then the
user can choose which datasets to plot in the emitted script. For the DH_orders fit,
we should probably coadd the datasets for the same transition, and then plot the coadded
dataset in the emitted script. However, there is no need to include this functionality
at this time. I will later return to an interactive GUI that will allow the user to
make these decisions more interactively.

**Q5.12 — Where should the three reference `atomic_*.xml` files live?** They are
currently untracked in `alis/data/`, and `pyproject.toml` ships that directory
wholesale (`alis = ["data/*"]`), so committing them there would put ~320 kB of
unused reference data into every wheel. Since 5.2 merges 15 rows out of them and
then never reads them again, shall I move them to `context/atomic/` (alongside
the other reference material, outside the package) and leave only
`atomic.ecsv` + the retained-but-unread `atomic.xml` in `alis/data/`?

**Response:** I will remove the three `atomic_*.xml` files from `alis/data/` once the refactoring is complete. I will not include them in the repo. Feel free to move them to context/atomic/ if you prefer.

**Q5.13 — The dead `run atomic atomic_rjc.xml` lines.** Ten context models name
an atomic file they have never actually loaded (see the audit addendum): it was
absent, so they fell back to `atomic.xml`, and now that it exists it crashes
them for want of the `Ly` pseudo-element. Once 5.2 lands `atomic.ecsv`, do you
want me to (a) delete the `run atomic` line from those ten models so they use
the shipped default explicitly, (b) repoint them at `atomic.ecsv`, or (c) leave
them alone? (a) and (b) are equivalent in effect and both keep the references
valid; (c) leaves a line that will break again the moment anyone puts a file of
that name on the path. I lean (a).

**Response:** Let's go with (b), because it displays the functionality, and it won't break the examples.

**Q5.14 — The atomic-mass off-by-one (new task 5.6).** Auditing 5.2 turned up
that `load_atomic` pairs two independently de-duplicated lists positionally, so
64 of 80 isotopes get the wrong atomic mass — 1H is given deuterium's mass, 16O
is given neon's — and that mass sets the thermal broadening
(`b = sqrt(b_turb^2 + 0.0166*T/m)`). Every fit with a `temperature` is affected,
including the shipped examples. The fix is a few lines; the consequence is that
**every Stage 0 reference moves**, so the bitwise gate cannot hold across it.
How do you want to proceed: (a) fix it in Stage 5 and regenerate all references
in one clearly-labelled commit, (b) fix it but hold the regeneration for a
separate release, or (c) something else? I would not want to ship the ECSV
conversion (5.2) without deciding, because 5.2 rewrites the same loader and it
would be strange to leave the bug in place while touching that code.

**Response:** I don't think your proposed bug is correct. ALIS is loading the
atomic data, and when the Voigt model is constructed, it first searches for the
element name (which is the mass number and the element, e.g. `2H` for deuterium),
then it finds the atomic mass associated with that element. So, having multiple
different elements with the same mass number should not cause any issues.
Therefore, I don't think we need to make any changes to the atomic mass
handling in ALIS. However, you mentioned that some of the isotopes used in the
atomic data file use conflicting mass numbers. This is because some of the
transitions take into account isotopes, while some do not. Please do no make
any changes to the atomic data listed in the atomic data file. This is an important
input into ALIS and we cannot afford to introduce any transcription errors when
converting from xml to ecsv.

**Q5.15 — Atomic mass, re-verified (task 5.6).** Your Q5.14 answer is that the
lookup is by element name so a shared mass number is harmless. That is what the
code intends, but I have now measured it end-to-end rather than read it: a probe
on `alis/functions/voigt.py:336` during a normal `metal_line_abs` run prints
`ion=16O_I mass_used=20.18` — neon's weight, not oxygen's. The cause is that
`load_atomic` de-duplicates `Element` and `AtomicMass` *separately*, so the
index found in one list is used in the other and they align only by luck.
13 of the 14 ions in the shipped fits are affected.

To be fair to your intuition, D/H is nearly immune — 1H and 2H shift together,
so b(H)/b(D) is only 0.3% off. It is H-versus-metal at a tied temperature that
is ~20% off. **No atomic-data edit is involved**; the data are fine, the loader
mis-pairs them.

If you would still rather leave it, say so and I will drop it and note it as
accepted behaviour. If you want it fixed, the choice is the same as before:
regenerate the references in one labelled commit, or hold that for a release.

**Response:** OK, this bug is now resolved. Thanks for identifying the issue,
and helping to find a solution using the script I started in the `context/atomic/` directory.

**Q5.16 — The fixed pixel buffer (5.4) would break more than it fixes.** Your
Q5.5 design adds a data-line keyword for the number of extra pixels either side
of the fitrange, defaulting to 30. Before implementing I measured what the
*current* rule loads. It is not a pixel count at all: `load_data` calls the
resolution function's `getminmax`, which extends the fitrange by **10 sigma of
the instrumental profile in wavelength** (`alis/functions/vfwhm.py:122`).
Across the shipped
examples that comes to:

| example | buffer pixels per side |
|---|---|
| CNabs | 10 |
| metal_line_abs, lls, summed_coldens | 12 |
| lsf_hst | ~3491 |
| voigtconv | ~4697 |

So a fixed 30 would (a) change the loaded pixels for *every* example, breaking
the bitwise gate you have just regenerated, and (b) cut `lsf_hst` and
`voigtconv` by **two orders of magnitude** -- and those are exactly the cases
with broad instrumental profiles, where the buffer exists so the convolution is
not truncated at the fitrange edges. 30 pixels would corrupt the model there.

There is a way to fix the actual bug -- a saved model reloading a different
pixel set -- with **no change to the buffer rule and no reference movement**:
have the writer record the **loaded wavelength range** in the `.mod.out` data
line. `loadrange=` already exists as a keyword and is already echoed when the
user sets it, but it is never written with the range the resolution rule
produced. Emit that, and re-reading a `.mod.out` loads exactly the same pixels
whatever the fitted resolution became.

Proposed: (1) write the concrete `loadrange` into `.mod.out` -- fixes the
round-trip, bitwise-safe; (2) add your extra-pixels keyword as an **optional
override** for users who want explicit control, with no default so the
resolution rule stays in charge; (3) keep the "insufficient buffer" warning you
asked for, based on the resolution width rather than a flat 30. Does that meet
what you were after, or do you specifically want the flat default?

**Response:** I agree on all three suggestions.

**Q5.17 — Should `load_atomic` still read VOTable after 5.2?** Q5.1 says
`atomic.xml` should be "retained as a supported input during a deprecation
period"; Q5.6 says it should be kept in the repo but "won't be loaded or used".
Those point different ways: the first keeps a VOTable branch in `load_atomic`
(so a user's own `.xml` still works), the second lets 5.2 delete it and read
ECSV only. Which do you want? I lean on keeping the reader -- it is ~10 lines,
`run atomic` accepts any filename, and users will have their own XML files --
with ECSV as the shipped default and a deprecation warning when an XML is read.

**Response:** OK, I agree with your suggestion. Let's maintain support for the
xml format for now, and warn users if they are using a format that is about to
be deprecated. We should also encourage users to request RJC to include new
atomic information in the ECSV file or provide a python script in `alis/data/`
that converts an xml file into an ecsv file.

**Q5.18 — What setting selects the plotting mode (5.3)?** The three modes need
a name in the model file. `out plots` is already taken (the PDF ALIS itself
writes), so I suggest a sibling: `out plotscript none|metals|DH`, written beside
the `.mod.out` as `<model>.mod.plot.py`. Happy to use another name/section if
you have a preference -- and should the script be emitted on every run that
sets it, or only when the fit completes successfully?

**Response:** We should set `out plotscript none` as the default, and users can
override this option in their `.mod` file. If `out plotscript` is not `none` then
a plotting script will be produced. Another alternative is to provide a little more
control to the user, by changing the three word output to something like
`plotscript format none|metals|DH` and then their can be customisable settings like
`plotscript numcol auto` (which would automatically select the number of columns, or, users can enter a number `3` that would set 3 columns as the default)
`plotscript residuals True` (Show the residuals at the bottom of each panel)
`plotscript velrange auto` (automatically set the velocity range of the plot, or users can select a two column list `-150,150`)
These are just some examples, and if you think other settings would be helpful, feel free to add additional options. All of these settings will become customisable in a later stage when we incorporate an interactive plotscript tool as part of the GUI, so preparing some of that here might be beneficial. 

**Q5.19 — `CLAUDE.md` names three files that no longer exist.** Not a Stage 5
task, but it is the standing instruction file I read every session, so it is
worth correcting before it misleads a later stage:

| `CLAUDE.md` says | actual |
|---|---|
| "The entry point is `alis/alis.py`, which contains `ClassMain`" | `ClassMain` is at `alis/main.py:67`; `alis/alis.py` does not exist |
| "Model fitting functions are defined in `alis/alfunc_*.py` files" | `alis/functions/*.py` |
| "messaging ... through the `msgs` object (`almsgs.msgs()`)" | `almsgs.py` does not exist; `msgs()` is at `alis/logger.py:161` |

("Atomic data are stored in `alis/data/atomic.xml`" is still right, and becomes
`atomic.ecsv` in 5.2.) Shall I update those three lines? They are one-line
corrections and I would rather not edit your instruction file uninvited. The
same reorganisation is presumably unrecorded in the Stage 6+ prompt files too --
say the word and I will sweep those for stale paths in the same pass.

**Response:** Yes, please fix those three lines in CLAUDE.md. Also, if you can do
a sweep of the Stage 6+ prompt files, that would be great. Thanks!

**Outcome (2026-08-03).** Both done. The sweep of the forward-looking prompt
files turned up more than paths:

- **`CLAUDE.md`** -- three lines fixed (`alis/main.py`, `alis/functions/*.py`,
  and the `from alis import logger` / `logger.msgs()` idiom).
- **`refactor_code_stage6.md`** -- the 6.3 lint measurements had drifted and one
  claim was wrong. Re-measured: E701 1215 (was 1217), F821 **37** (was 39),
  E722/F841 unchanged; `== None` is **5 live**, not 6 (the sixth is commented
  out); `plot.py`'s undefined `slf` is **700-735, 23 refs**, not ~698-725/~20;
  and `SourceModule` is **no longer in `voigt.py`** (Stage 4's GPU port removed
  it -- that is exactly the 39 -> 37 drop), leaving `constant.py:38` and
  `linear.py:39`. Also recorded that `--isolated` is required to reproduce any of
  this, since every file concerned sits in ruff's `extend-exclude`.
- **A new latent bug found during that check** -- `alis/load.py:611` calls
  `imp.load_source(...)`, but `imp` is neither imported nor importable (removed
  from the stdlib in Python 3.12; ALIS targets 3.13). It is swallowed by a bare
  `except` and reported as a missing *user* module, so `systmodule=` is dead and
  misreports whose fault it is. Written up under 6.3.
- **`context.md` / `setup.md`** -- the `new-alfunc` and `port-to-gpu` rows in the
  skills tables still described the pre-Stage-2 `alfunc_<name>.py` /
  `alfunc_base` convention (4 rows, now corrected). The `SKILL.md` files
  themselves were already right. The surrounding prose in those two files
  describes the *pre-refactor* code deliberately ("In the current version of
  ALIS, the file alis.py ...") and was left alone, as were `stage2.md`,
  `ALIS_v2_code_plan.md` and everything under `logs/`, which are historical
  records. `stageX.md` and `refactor_code_unit_tests.md` were already clean.

**Q5.20 — 5.4 moves 6 of the 41 references; may I regenerate them?** You asked
in Q5.5 that I say so rather than regenerate silently, so: the Q5.16 `loadrange`
work is genuinely reference-neutral, but two of 5.4's other items are not.
`compare_mod_out` skips the `data read` block, so the `lsf` `=`->`:` fix and the
new `loadrange=` cost nothing. The model section, though, is compared token by
token — so dropping the empty `locations=` (token count) and giving
`damping=0.0000000` its fixed-status suffix (suffix mismatch) both fail against
the current references. Affected:

| fix | references |
|---|---|
| `damping=` | `helium34/{Her36,HD319718,tet02OriA}` |
| `locations=` | `examples/splineabs/{fit_spectra,fit_spectra_linear}`, `examples/spline/fit_spectra_splineContAbs` |

That is minutes of runtime, not the 2.5-hour full harness. The change is cosmetic
by construction — the same fit, a different keyword token — so I would verify
rather than assume: regenerate those six, then diff each against its predecessor
and confirm **only** the `damping=`/`locations=` tokens differ, with chi-squared,
DOF and every parameter value byte-identical. Anything else moving would mean the
writer fix changed the fit, and I would revert rather than accept it.

Three ways to go: (a) I regenerate those six myself as part of 5.4 and report the
diffs; (b) I make the code change and you run `regen_harness.sh` for those cases,
as you did for 5.6; (c) I implement only the reference-neutral items now
(`lsf`, `loadrange`, the `rstrip` tidy) and we hold `damping`/`locations` for a
release. I would suggest (b) — it keeps regeneration in your hands, which is
where it has been all stage, and the six are quick.

Note the two `.reference_adjusted` files cannot be dropped until *both* the
`lsf` and `locations` fixes are in, since one exists for each.

**Response:** Let's go with option (a), that you regenerate those six references
yourself as part of 5.4. I can use the git diffs to verify the changes.

## Prompts

1. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing (please check all filenames mentioned in this document reflect all updates to the code so far, and update as needed). Ask further queries if needed.

2. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing (please check all filenames mentioned in this document reflect all updates to the code so far, and update as needed). Ask further queries if needed. I am currently running the fast and medium test suite in the background. Assuming it passes, please check in the meantime if you have any further queries about the tasks in this stage.

3. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing. Ask further queries if needed.

