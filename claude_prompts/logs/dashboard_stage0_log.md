# Dashboard Stage 0 log (mockups)

### 2026-10-04 (Task 0.1: what the mockups show)

Read `J1358p6522.mod`, its reference outputs, and the Q1243p307 model and snips.
The choices below are what every page is built from.

**J1358p6522** (`context/fitting_examples/VMP_DLA/J1358p6522/`). The system z is
taken as 3.067258, the best-fit z of the main H I component.
- **Data tab:** the full spectrum, `J1358p6522_fluxcal.dat` (3 columns, 40,402
  pixels, 3650–6305 Å), with the system's transitions marked. The column-role
  dialog (S3) is shown for `VMP_DLA/J0814p5029/data/J0814p5029_HIRES.dat`, a
  real five-column file (wave, flux, error, an unlabelled fourth column, and a
  continuum).
- **Regions tab (A and B):** H I 926.2 (Ly7, snip `H2`). It has everything the
  tab must show: shared pixels with H I 923.2 and 930.7 (S26), O I 925.4 inside
  the snip, and Lyα-forest interlopers at z ≈ 2.09–2.10 for the line IDs (S5).
  Its continuum is the 5-coefficient Legendre of `H2`.
- **Components tab:** O I in six transitions (921.9, 925.4, 971.7, 976.4,
  1039.2, 1302.2), and a second page with H I and D I in the eight Lyman
  lines. The component matrix has c1 (H I, D I, O I) and c2 (H I only, at
  −7.1 km/s). The 57 `1Ly_a` interlopers form one collapsed "Lyα forest"
  group.
- **Blinding:** the D I column density is shown blinded on every page.
- **Fit tab:** badges for all 12 snips, computed from the reference `_fit.dat`
  files with ALIS's own runs test (`alis/report.py`); values and errors from
  the `.mod.out` and its `# Errors` block; correlations from the covariance
  matrix.
- **Plot tab:** the DH preset, with the eight Lyman lines in a 4×2 grid.

**Q1243p307** (`context/fitting_examples/DH/Q1243p307/`).
- **Data tab:** three datasets, each with its own resolution label and shift:
  HIRES (16 snips plus 7 for other systems; the reference), PROCHASKA (16;
  shift fixed at 0) and KIRKMAN (12; shift fixed at +83.5 km/s). Four other
  systems (z = 2.05, 2.18, 2.40, 2.44).
- **Regions tab (C):** O I 1302.2 in all three datasets, with the 97 pixels it
  shares with Si II 1304.4 in each. KIRKMAN's snips are cut in its own frame
  (its snip starts 1.29 Å, 84 km/s, redder than HIRES), so that strip is drawn
  in the HIRES frame by removing the +83.5 km/s shift.

**Things the real models do that the dashboard will meet** (for Q0.3):
- In J1358p6522, D I's b_turb (8.03 km/s) is not tied to H I's (11.20 km/s),
  although both have T fixed at 0. D22 says isotopes must share b_turb and T.
  Q1243p307 ties them (`da`, `ta`).
- In J1358p6522, each of the six O I lines has its own free z (spread
  ±0.7 km/s), while N and b are tied. D20 shares z across a component.

**Other facts used by the build script:**
- The reference `_fit.dat` files have seven columns: wave, flux, error, fit
  mask, continuum, zero level, model. Outside the fit range the continuum and
  model hold the sentinel −9.999999999×10⁹. (`ALIS_workflow.md` §5.1 describes
  the four-column case.)
- The Legendre continuum is reproduced to 3×10⁻⁴ by evaluating the
  `.mod.out` coefficients times their `scale=` factors on the snip's
  wavelength range mapped to [−1, 1]. The mockups therefore draw it across the
  whole snip, not only the fitted pixels.
- The covariance file holds all 450 parameter slots, with zeros for fixed and
  tied ones. Parameters are matched to their slots by their errors.

### 2026-10-04 (Tasks 0.2–0.9: the mockups, built and published)

**0.2 — Build script.** `doc/dashboard/mockups/build_mockups.py` reads the two
fits and writes every page. Run it from that directory:
`python build_mockups.py` (pages next to the script) or
`python build_mockups.py --publish DIR` (also one combined page for the
Artifact viewer). It uses numpy and ALIS's own readers (`load.read_atomic_table`,
`report._runs_z`); it adds no dependency and never runs a fit. Plots are inline
SVG, downsampled to a min/max envelope per screen pixel (Q0.1). Lines are at
most 88 characters, and `ruff check` passes.

**0.3 — Common frame.** One 1440×900 window (Q0.2): title bar, menu, toolbar
(undo/redo, autosave, mode, a "Blinded: N(D I)" pill, export), the five tabs
with status markers (F12), the `.mod` panel (open on Regions A, Components A
and the H I/D I page; collapsed elsewhere) with the real model text and the
current line highlighted (S16), and a status bar with run 4 in the background
(F6). The D I column density is masked on every page (D24).

**0.4–0.8 — Pages written** (all in `doc/dashboard/mockups/`):

| Page | File | Data |
|---|---|---|
| Data: new project | `data_new.html` | J1358p6522 spectrum; S3 dialog on J0814p5029_HIRES.dat |
| Data: three datasets | `data_datasets.html` | Q1243p307 datasets and coverage |
| Regions A | `regions_a.html` | H I 926.2 (snip H2), coverage list (S2) |
| Regions B | `regions_b.html` | H I 926.2, whole-spectrum strip |
| Regions C | `regions_c.html` | Q1243p307 O I 1302.2 × 3 datasets, normalised |
| Components A | `components_a.html` | O I, 3×2 grid, matrix |
| Components B | `components_b.html` | O I, one column, cards |
| Components A, H I + D I | `components_hd.html` | 8 Lyman lines, D22 notice |
| Fit | `fit.html` | badges, results, correlations, history |
| Plot | `plot.html` | DH preset, 4×2 Lyman lines |

Plus `index.html`, which links them with a key.

**0.9 — Checked and published.** Each page was rendered with headless Chrome
at 1440×900 and corrected: gaps in the spectrum are no longer bridged, line
IDs no longer overlap, tick labels of neighbouring panels no longer clash, and
the Fit tab's panes fit the window. Published as one private page for
comments: https://claude.ai/artifact/J43w9ERNESDo9hez9o918B (the source is
written by `--publish`, not kept in the repository).

**Findings from building the mockups** (raised as Q0.3–Q0.5 in
`dashboard_stage0.md`):
- **Runs test on resampled spectra.** Computed exactly as `alis/report.py`
  does, 9 of J1358p6522's 12 snips are flagged (red) only by the runs test
  (runs z −3.9 to −9.6), although χ²ν is 0.65–1.12. Negative z means too few
  sign changes: the residuals are correlated between neighbouring pixels, as
  expected for a resampled spectrum. Per-pixel runs-test badges would be red
  almost everywhere on such data. (Q0.4)
- **Isotopes in an imported model.** As noted under 0.1, J1358p6522 gives D I
  its own b_turb, which D22 forbids. The H I/D I page shows a notice with
  "Tie D I to H I" and "Keep as imported". (Q0.3)
- **Snips outside their source spectrum.** `J1358p6522_fluxcal.dat` has no
  data between 4535 and 5593 Å, but the Lyα (H8, 4750–5150 Å) and O I 1302
  (O4) snips lie there. Those snips came from another file, so matching snips
  to "the" source spectrum (F13, S23) would fail for this fit. (Q0.5)
- **Coverage list.** Ranking by fλ alone hides most of the transitions that
  matter here (high-order Lyman lines and weak O I lines are low in fλ). The
  list therefore shows the strongest 14 and then always lists every
  transition with a snip.

**Not done:** Task 0.10 (RJC's review).

### 2026-10-04 (Prompt 2: draft 2 of the mockups, after RJC's review)

Read RJC's responses to Q0.3–Q0.5 and the 12 comment threads on the published
page. All were clear enough to act on, so draft 2 was built straight away; the
two questions RJC put to Claude are answered by draft 2 and raised for
confirmation as Q0.6 (when a transition becomes a snip) and Q0.7 (palette).
Q0.8 follows from the Q0.5 response (tying settings between file rows). The
review outcome and its changes to D5, D13, D14, S2, S18, S24 and S25 are
recorded under Task 0.10 in `dashboard_stage0.md`.

**Pages in draft 2** (`doc/dashboard/mockups/`; draft 1 is kept in `draft1/`):

| Page | File | Main change |
|---|---|---|
| Data: new project | `data_new.html` | type z or identify a feature; dialog for every file |
| Data: three datasets | `data_datasets.html` | one row per file (Q0.5) |
| Regions | `regions.html` | A; transition list per element and ion stage |
| Regions, three datasets | `regions_datasets.html` | stacked strips inside A's layout |
| Components: O I | `components.html` | A's panels (3 columns), B's cards |
| Components: H I + D I | `components_hd.html` | 2 columns, scrolling; D I only with H I |
| Fit: Inspect | `fit_inspect.html` | View menu, ‹ ›, zoom/pan, χ²ν and runs marks |
| Fit: Results | `fit_results.html` | full table, larger correlations, statistics |
| Plot | `plot.html` | panel grid with drag and drop, "Not shown" tray |

`regions_b` and `components_b` are retired; `regions_a`, `regions_c`,
`components_a` and `fit` were replaced by the pages above.

**Colour-blind-safe palette.** Every colour now comes from Okabe–Ito: data
black, model vermilion, continuum blue, fit regions bluish green, other snip's
regions sky blue, shared pixels reddish-purple hatching, snip edges orange,
H I vermilion and D I blue (no red/green pair). Fit marks use ✓ ! ✕ and tab
markers ✓ ↻ ! as well as colour. The Plot preview keeps the published
figures' colours (Q0.7).

**Things found while building draft 2:**
- Ranking candidates across ions by fλ, as draft 1's transition list did,
  also failed for "identify a feature": the Balmer lines in
  `atomic_rjc.xml` (Hα, Hβ) and Mg II outrank Lyβ, so the right answer was
  missing. Candidates are now the strongest lines of common ions in a fixed
  order (H I first), among lines that give a positive z. The click on the
  Lyβ trough gives z = 3.06726.
- O I 971.7 is two tabulated lines 0.0006 Å apart (971.7382 and 971.7376 Å,
  plus a third, weaker one), which the list showed as two identical rows.
  Lines closer than 0.05 Å now share one row ("O I 971.7 ×2").
- Snip names used rounded wavelengths (976.45 → "976.5"); the build script
  now uses the tabulated values (976.448 → "976.4").

**Build script.** `build_mockups.py` was changed with an AST-based swap of
whole definitions; the draft 1 page builders were removed. `ruff check`
passes and every line is within 88 characters. All nine pages were rendered at
1440×900 with headless Chrome and checked.

**Published** as version 2 of the same page:
https://claude.ai/artifact/J43w9ERNESDo9hez9o918B. Replies were posted on all
11 threads sent to Claude. Nine are resolved. Two stay open until RJC answers
Q0.6 (transition list) and Q0.7 (palette). One thread ("one row per dataset…
This looks good to me!") was not sent to Claude, so it was left untouched.

### 2026-10-04 (Prompt 3: draft 3 of the mockups, after RJC's review of draft 2)

Read RJC's responses to Q0.6–Q0.8 and the 15 new comment threads on draft 2.
All were clear, so draft 3 was built straight away. The review outcome is
recorded under Task 0.10 in `dashboard_stage0.md`, and two new questions are
raised there: Q0.9 (hiding the difference of a blinded parameter when
comparing runs) and Q0.10 (where the publication style file lives).

**Pages in draft 3** (`doc/dashboard/mockups/`; drafts 1 and 2 are kept in
`draft1/` and `draft2/`):

| Page | File | Main change |
|---|---|---|
| Data: new project | `data_new.html` | Element → Ion → Transition menus in "Identify a feature" |
| Data: three datasets | `data_datasets.html` | note that rows can be tied (Q0.8) |
| Regions: one dataset | `regions.html` | shared layout, System menu, SNIP/CLEAR, vertical key |
| Regions: three datasets | `regions_datasets.html` | same layout; Continuum and Snip boxes for the selected strip |
| Components | `components.html`, `components_hd.html` | unchanged |
| Fit: Inspect | `fit_inspect.html` | Compare sub-tab; Columns setting for groups |
| Fit: Results | `fit_results.html` | run 4 (the full model) in the history |
| Fit: Compare | `fit_compare.html` | new: two real runs compared |
| Plot | `plot.html` | colour-blind-safe defaults, residuals, script colour variables |

**The Compare page uses two real fits.** `DH/J1358p6522/model/
J1358p6522_original.mod.out.reference` is a full model of the same absorber
(metals, Lyα forest, T free, D/H as a ratio). Its Lyα and O I 1302 fits have
the same pixels as the simplified model's (9,696 and 283), so both models can
be drawn on the same data. Findings shown on the page:
- the total log N(H I) agrees: 20.490 (2 components) and 20.491
  (3 components);
- log N(O I) agrees: 14.848 ± 0.023 and 14.844 ± 0.016;
- the main component's b_turb differs (11.20 and 1.19 km/s), because the full
  model fits T = 11,732 ± 765 K where the simplified one fixes T = 0;
- the FWHM differs by 2.2σ (6.931 ± 0.171 and 6.480 ± 0.118 km/s);
- the full model has a Lyα interloper near −2,250 km/s that the simplified
  model lacks, visible in the model-difference strip.
In the mockup history these are run 3 and run 4; run 5 is the one running.

**Other details:**
- **Regions list:** transitions with no data are no longer listed (outside
  every file, or in a gap; "no data" for Q1243p307's imported snips). A
  transition that has a snip stays, flagged (J1358p6522's Lyα, flagged "gap",
  Q0.5).
- **Identify a feature:** the Transition menu lists only lines that put the
  click at z > 0, so the Balmer lines are excluded.
- **Plot preview:** colour-blind-safe colours and residual strips. The serif
  font imitates the LaTeX style; the code box shows the top of the script
  with its colour variables.

**Build script:** changed with the same AST-based swap; `ruff check` passes
and every line is within 88 characters. All changed pages were rendered at
1440×900 with headless Chrome and checked. Fixes made after rendering:
- the three-dataset Snip box overflowed the window, so the strip view was
  shortened (it scrolls);
- the code box clipped long comments;
- a label overlapped the spectrum.

**Published** as version 3 of https://claude.ai/artifact/J43w9ERNESDo9hez9o918B.
Replies were posted on the 15 new threads. All 17 threads sent to Claude are
resolved, including the two left open in draft 2, whose questions RJC
answered in the doc. One thread was never sent to Claude, so it was left as
it is.
