# Dashboard Stage 12 log

Feedback from users, after Stage 11's list closed (Q11.2 (a)). The plan is in
`claude_prompts/dashboard_stage12.md`.

### 2026-11-16 (Prompt 1: the design, Tasks 12.1–12.12, and Q12.3–Q12.8)

**Read.**
- This stage's document: Feedback points 1–8 (RJC's MWC137 project,
  `context/feedback01/MWC137/`), the three items carried over from Stage 11, and RJC's
  responses to Q12.1 (items 1 and 3 wait; item 2 joins, the emptied dataset removed
  only by the user, with a warning) and Q12.2 ((a), another stage of feedback next).
- `ALIS_v2_dashboard_prompts.md`'s decisions D1–D128 in full, with QF.21 and Q1.8 of
  `dashboard_stage1.md` (the test for pixels fitted twice); `dashboard_stage11.md` in
  full and its log's Prompts 1–5, 7 and the close; `deferred_work.md`'s priorities
  and §7; the `gui-component` and `gui-dev` skills (their Stage 7–11 additions).
- The code each point touches: `load.find_shared_pixels`, `snips.shared_pixels`,
  `markers` (`fitted_twice`, `data_marker`, `regions_marker`), `preflight`;
  `components` (`include`, `draggable`, `drag_target`), `qt/components.py`
  (`refresh`, `_fill_panels`, `_fill_cards`, `selection_changed`, `_toggle_ion`,
  `readout`, `DragKeys`); `qt/regions.py` (`_fill_strips`, the Snip box's
  resolution); `qt/plots.py` (`SpectrumBox.wheelEvent`, `ZoomBar`); `bundle.py`
  (`add_source`, `embed_source`) and `sources.py`; `resolution.py` and
  `alis/functions/lsfspline.py`; `project.py` (`MAIN_ISOTOPE`, `ion_name`).

**Reproduced** on a copy of `mwc137_relinked.model` in the session's scratchpad
(scratch scripts and scratch `gui` tests; no code changed).
- *Point 1.* `shared_pixels` finds specids 6 and 7 sharing 147 pixels (73% of the
  smaller); ALIS's own warning is on line 32. The two files share one grid and agree
  digit for digit at 297 of reg1's 336 pixels, differing only in the core
  (7701.28–7702.10 Å). Specid 6 is in "39KI7701 (unassigned)", specid 7 in
  emsmwcs137.
- *Point 2.* `components.include(a2, ion)`: Li I works; Ti II, Ca I and Fe I raise
  "Component a2 is excluded from every ion: include it in one first" (a2 has no
  live line to copy z, b and T from).
- *Point 3.* A direct include keeps the card list's scroll; a real mouse click on a
  card's Excluded ions link (QTest) moves it from 1496 to 0, the click selecting the
  card (`selection_changed` scrolls to it before the cards are laid out). The
  audit, each tab's scroll bars and views moved, then one step (on the tab or on
  another): Data keeps all; Regions resets its list of spectra, its transition
  list (sideways) and the Snip box's table, keeping the spectra's ranges (D114);
  Components resets every panel's ranges to ⌂; Fit · Inspect resets its view and
  the progress chart; the drop-downs keep their choices everywhere. The window
  always opens at 1440 × 900.
- *Point 5.* `draggable(a7, "39K_I")`: log N False, b True; K I's log N is `ka`, from
  `ka(lia,lika) = lia - lika`, `lika` a free variable used by that link alone.
- *Point 6.* The bundle's sources have `"embedded": null`; `embed_source` exists in
  `alis/bundle.py` and `sources.py` reads an embedded copy, but no window action
  makes one. Deflated, `emsmwcs137.dat` is 12.8 MB (0.9 s) and `EM_MWC__137.dat`
  2.5 MB (0.2 s), against a 1.7 MB bundle.
- *Point 8.* `ion_name` gives "Rb I" for both `85Rb_I` and `87Rb_I`; a1's Excluded
  ions read "Li I, Rb I, Rb I". Of the elements with several isotopes in
  `atomic.xml`, rubidium alone is missing from `MAIN_ISOTOPE`. The navigator lists
  Rb I once (87Rb I has no live line); no notice is raised.
- *Item 2.* The Data tab's marker has no case for an unassigned dataset with no
  snips.

**Written.**
- `dashboard_stage12.md`:
  - the Design: how the feedback becomes tasks; "12.1–12.12", what was found for
    each point, and what stays as it is;
  - Tasks 12.1–12.12, each with what it builds or fixes and its check: Rb's isotope
    names (bug), including an ion a component never had (bug), a linked column
    density in a drag, snips the user says are different data (bug), an emptied
    unassigned dataset that says so, every view keeping its place, zooming the flux
    axis with ⌂x and ⌂y, source spectra kept in the project, lsfspline's knots (the
    logic, then the dialog), the review, and the close (against Stage 11's counts:
    2235/119 unit, 301 `gui`, 149 `fast`; ALIS outside `alis/dashboard/` unchanged
    against 1e4c2f4);
  - Q12.3 (point 1: lean, the user says so per pair), Q12.4 (point 3 between
    sessions: lean, the window's size and position in the preferences), Q12.5
    (point 4's key: lean, ⇧ Shift held rather than Y), Q12.6 (point 5: lean, the
    link's single free input moves, else the readout says why), Q12.7 (point 6:
    lean, a copy per spectrum by choice, and Save a copy with spectra…), Q12.8
    (point 7: lean (a) for where, the centre knot, the smoothing's start and what
    the fit frees);
  - notes under the responses to Q12.1 and Q12.2, the header's and the Queries'.
- `deferred_work.md`: §7.8 (Load snips… for a file that names no transition) and §7.9
  (Open a fit's question in the singular), Q12.1 (b), and low priority 23.

No code was changed. Waiting on RJC's responses to Q12.3–Q12.8; Tasks 12.3, 12.4 and
12.6–12.10 are written with the leans, and are revised where a response departs.

### 2026-11-16 (Prompt 2: RJC's responses to Q12.3–Q12.8 applied)

Asked as "prompt 23"; the stage document has Prompts 1 and 2, and RJC's responses
were in, so it was taken as Prompt 2.

**Read.** RJC's responses to Q12.3–Q12.8; `session.py`'s view state (`set_view`,
`view_of`, `close`, which keeps a changed view in the bundle without unsaved
changes) and each tab's use of it; `resolution.convert`; `lsfspline.call_CPU`'s
smoothing (a Gaussian of σ = smoothing/√2); D48 and D114.

**Applied** in `dashboard_stage12.md`:
- *Q12.3 (Ignore warning)*, Task 12.4: Ignore warning in the offer for pixels fitted
  twice, for the pair, as one step; `shared_pixels` and ALIS's test unchanged, the
  dashboard's warnings leaving the pair out; the warning back when the pair's shared
  pixels change; Warn again in the Snip box's list of warnings ignored, or Undo.
- *Q12.4 (b)*, Task 12.6: the window's size and position in `dashboard.json`; each
  view's ranges and each list's place in `ui/view.json`, kept by what each shows and
  by the item at a list's top; read only at Save, a clean close or a recovery copy,
  never on a scroll or a zoom, and restored once at open. The check times the
  reading (under 5 ms on 20 snips) and checks that zooming and scrolling write
  nothing (RJC's condition of no lag). D48 joins the decisions the close refines.
- *Q12.5 (b)*, *Q12.6 (a) with (b)*, *Q12.7 (a)*: Tasks 12.7, 12.3 and 12.8 as
  written.
- *Q12.8 (a) for all four*, Task 12.9: the smoothing starts at the resolution before
  as a FWHM in km/s (`vfwhm`; `vsigma` and `Afwhm` converted), or 5 km/s where none
  converts or the value is hidden. The note under the response says that ALIS's
  smoothing Gaussian is 1.67 times wider than the smoothing.
- Notes under each response, and the header's, the Tasks' and the Queries' notes.

**Checked.** Tasks 12.1–12.12 are numbered in order, each with what it builds or
fixes and its check; 12.12 closes the stage. No further query was needed. No code
was changed. The next prompt starts Task 12.1.

### 2026-11-16 (Prompt 3: the delta function, and lsfspline's smoothing as a FWHM)

**Read.** Prompt 3; `alis/functions/lsfspline.py` in full (`call_CPU` smooths the
knots' lines with a Gaussian of σ = smoothing/√2; `set_vars` asks for
√2 · nsubpix · pixel / smoothing sub-pixels); `vfwhm.set_vars` (2√(2 ln 2) · nsubpix ·
pixel / FWHM); `load.load_subpixels` and `get_binsize` (the sub-pixels per snip, the
most any function asks for, between `nsubmin` 5 and `nsubmax` 21, each pixel split
evenly); `config.py`'s settings; the regression harness's discovery
(`tests/manifest.py`): no case uses `lsfspline`, since `examples/lsf_spline` has no
references and no fit of `context/fitting_examples/` uses it.

**Checked** with a scratch script
(`/private/tmp/claude-501/-Users-rcooke-Software-ALIS/482188ad-3a54-42ff-be7f-241bdc95cabd/scratchpad/s12/delta_check.py`):
ALIS's `lsfspline` on Prompt 3's five knots, on a grid of 0.1 km/s, gives a Gaussian of
FWHM 1.65–1.67 times the smoothing (2√(ln 2) = 1.665), as Prompt 3 (ii) says.

**Written** in `dashboard_stage12.md`:
- Task 12.9: ALIS's smoothing as a FWHM (`call_CPU` and `set_vars`; old models smooth
  0.60 times as wide, said in `CHANGELOG.md` and the docstring; no harness reference
  moves); `delta_knots` (spix from the snip's smallest pixel and `nsubmax`, edges at
  six times the FWHM and at least twice spix, the edges following the FWHM while the
  five knots stand; ±spix free as inner knots, Q12.8 (iv)); the kernel on the snip's
  finest sub-pixels; the buffer's σ from the FWHM; the checks: a new
  `tests/test_lsfspline.py` (a delta function gives a Gaussian of the FWHM to 10⁻⁶
  of its peak; the sub-pixels asked for match `vfwhm`'s; the knots' shape unchanged)
  and the delta function in `test_dashboard_lsfknots.py`.
- Task 12.10: Delta Function, with its question (Confirm, Cancel), and the Smoothing,
  FWHM field; its checks.
- Task 12.11: the dialog after Delta Function. Task 12.12: `CHANGELOG.md`, decisions
  from Q12.1–Q12.10 and Prompt 3, and ALIS outside `alis/dashboard/` changed in
  `lsfspline.py` alone.
- The Design's notes on ALIS (how the feedback becomes tasks, point 7, what stays);
  a note under Q12.8's response; Q12.9 (the sub-pixels in spix: lean `nsubmax`; the
  edges following the FWHM: lean yes) and Q12.10 (`examples/lsf_spline` as a
  regression case: lean yes); the header's, the Tasks' and the Queries' notes.

No code was changed. Waiting on RJC's responses to Q12.9 and Q12.10; Tasks 12.9 and
12.10 are written with the leans.

### 2026-11-16 (Prompt 4: the responses to Q12.9 and Q12.10, and Q12.11)

**Read.** RJC's responses: Q12.9 (i) spix from `nsubmin`, never finer than ALIS's
sub-pixels, and ALIS's default `nsubmin` raised from 5 to 10; (ii) (a), the edges
following the FWHM. Q12.10 (b), `examples/lsf_spline` a regression case, deferred if
it is too slow or does not reproduce. Also where `nsubmin` is named (`config.py`;
`doc/ALIS_workflow.md`'s table; `doc/tex_files/description.tex`; nine
`helium34` models set `run nsubmin 100`), and the dashboard's use of ALIS's
sub-pixels (`profile.py` and `continuum.py` call `load.load_subpixels`).

**Checked** with a scratch script
(`/private/tmp/claude-501/-Users-rcooke-Software-ALIS/482188ad-3a54-42ff-be7f-241bdc95cabd/scratchpad/s12/spix_width.py`):
for a FWHM of 2 km/s on 2 km/s pixels, `lsfspline` asks for 12 sub-pixels. With spix
a tenth of a pixel (`nsubmin` 10), the delta function's kernel has FWHM 2.000 km/s
on 10 sub-pixels and 2.010 km/s (+0.5%) on 12; with a fifth (`nsubmin` 5), 2.040
km/s (+2.0%) on 12.

**Applied** in Task 12.9: spix from the model's `nsubmin` (or ALIS's default); the
default 10 in `config.py` and the two documents; the regression references that the
new default moves regenerated and logged with their χ²; `examples/lsf_spline` a case
of the `examples` batch, its data generated again, or deferred as RJC's response
says; the kernel drawn on the sub-pixels ALIS uses for the snip with the dialog's
smoothing, worked out again when the smoothing changes. Task 12.12's check: ALIS
outside the dashboard changes in `lsfspline.py` and `config.py` alone; `CHANGELOG.md`
gives the new default. Notes under the two responses, and the header's, the Tasks'
and the Queries'.

**Raised** Q12.11: (i) with spix from `nsubmin`, the kernel is exactly Gaussian only
where ALIS uses `nsubmin` sub-pixels (lean: spix from ALIS's sub-pixels for the snip
with the dialog's FWHM, which keeps both aims); (ii) the default of 10 changes every
fit's sub-pixels, moves the harness's references and slows fits of broad lines, and
the delta function does not need it (lean: keep 5). Task 12.9 is written with the
responses, and is revised if Q12.11's response departs.

**Checked** the list: Tasks 12.1–12.12 in order, each with what it builds or fixes
and its check; 12.12 closes the stage. No code was changed.

### 2026-11-16 (Prompt 4, run again: the response to Q12.11)

**Read.** RJC's response to Q12.11: (i) the knots are as the user defines them, and
never change with the sub-pixels during a fit; Delta Function sets their starting
values; no two knots may be closer than one sub-pixel at `nsubmin`, the separation
Delta Function gives; (ii) (a), ALIS's default `nsubmin` becomes 10.

**Applied** in `dashboard_stage12.md`:
- Task 12.9: spix is one sub-pixel at ALIS's coarsest, the snip's pixel as ALIS
  sizes it (`get_binsize`, its largest) divided by `nsubmin`, and for a dataset its
  snips' largest (changed from the smallest, so that no knot is finer than ALIS's
  sub-pixels in any snip); the rules refuse two knots closer than spix; a line
  written by hand that breaks a rule is read as it is; the knots never change with
  the sub-pixels; the checks (closer than spix refused, exactly spix accepted).
- Task 12.10: a drag stops spix from a neighbour; a double-click within spix adds no
  knot; knots that break a rule are marked, and Apply greyed until they are apart;
  their checks.
- Task 12.12: `CHANGELOG.md` gives the default of 10. The note under the response;
  the header's, the Tasks' and the Queries' notes.

Read "at least `nsubmin` sub-pixels" as one sub-pixel at `nsubmin` (the pixel
divided by `nsubmin`), the separation the response gives Delta Function's ±spix;
the note under the response says so.

**Checked** the list: Tasks 12.1–12.12 in order, each with what it builds or fixes
and its check; 12.12 closes the stage. Every query, Q12.1–Q12.11, is answered. No
further query was needed. No code was changed. The next prompt (5) starts Task 12.1.

### 2026-11-16 (RJC's question: the knots at ±spix after Delta Function)

RJC asked whether the Tasks make clear that the two knots at ±spix can still be
dragged after Delta Function, which only sets a very narrow profile in place of the
default triangle. It was intended (they are inner knots, dragged as any other, Task
12.10), but only implied. Made explicit: Task 12.10's Delta Function says that the
button sets the starting values only, and that the ±spix knots are dragged up,
down, outwards and inwards (to spix from the centre), deleted, or joined by others;
a new check drags and deletes them after Delta Function. Task 12.9's note on the
delta function says the same. No code was changed.
