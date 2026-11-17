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

## Prompt 5: the tasks (2026-11-16)

*Batches* run on a snapshot of the tree after each task (so that work went on
meanwhile), with pytest 9.0.2 locally and in a scratch venv with CI's pytest 9.2.0
(script `/private/tmp/claude-501/-Users-rcooke-Software-ALIS/482188ad-3a54-42ff-be7f-241bdc95cabd/scratchpad/batches.sh`).
Stage 11's close: unit 2235 passed, 119 skipped (9.2.0: 2234, 120); gui 301.

### Task 12.1 — Each isotope its own name (point 8)

**The bug.** Of the elements with several isotopes in ALIS's atomic data (C, H, He,
Li, Mg, Si, Ti, Cr, Fe, Zn, Rb), rubidium alone was missing from
`project.MAIN_ISOTOPE`, so `ion_name` named `85Rb_I` and `87Rb_I` both "Rb I".

**The fix.** `"Rb": 85` joins the table (`alis/dashboard/project.py`), with a note
that every element with several isotopes is there.

**Tests** (each failed before the fix):
- `test_dashboard_project.py`: `test_each_rubidium_isotope_has_its_own_name` ("Rb I"
  and "87Rb I"), `test_every_element_with_isotopes_has_its_main_one` (the table
  against the shipped atomic data; before, `['Rb']` missing).
- `test_dashboard_components.py`: `test_rubidium_s_isotopes_are_named_apart` (the
  navigator's row is "Rb I", "with 87Rb I", where it read "with Rb I"; a component
  excluded from both lists "Rb I, 87Rb I"), on a new synthetic fit,
  `dashboard_synthetic.write_rubidium_fit` (two components with lines of both
  isotopes around Rb I 7802 at z = 0; `write_ratio_fit` takes a `grid`).
- `test_dashboard_qt_components.py`: `test_each_rubidium_isotope_is_listed_once`
  (the card reads "Rb I, 87Rb I"; with Rb taken out of the table, "Rb I, Rb I").

**MWC137** (scratch `rb_mwc137.py` on a copy of `mwc137_relinked.model`): the
navigator reads "Rb I · 1 snip · 3 comp. · with 87Rb I"; each card names Rb I and
87Rb I once (a1's Excluded ions: "Li I, Rb I, 87Rb I, Fe I"). No notice names
rubidium: 87Rb I has no live line, so D22 has nothing to check.

**Batches:** unit 2238 passed, 119 skipped (9.2.0: 2237, 120); gui 302 passed
(9.2.0: 302). Pre-commit passes.

### Task 12.2 — Include an ion a component never had, when excluded from every ion (point 2)

**The bug.** `components.include` refused an ion the component never had when it
had no live line (`comp.main is None`): "Component a2 is excluded from every ion:
include it in one first".

**The fix** (`alis/dashboard/components.py`). The new line takes the component's z,
b_turb and T, as written and with their labels, from its first commented-out line in
the model's order (`_excluded_line`, which reads that line as it would be
uncommented), and a first guess of log N as before; it goes after the component's
last commented-out line (`_add_line_maker` takes an `anchor`). Before that, the
check that no live line now carries one of its free labels (`_check_untaken`, Q7.6)
runs, as for an include of an ion it had. `_main_word` is split into `_par_word`,
used for both. A component with neither a live nor a commented-out line still gets
the old refusal.

**Tests:**
- `test_dashboard_components.py`:
  `test_an_ion_never_had_is_included_in_a_component_excluded_from_every_ion`
  (J013301m400628's shape with O I in a2; a1 excluded from H I and D I takes O I
  with `ra`, `doppa`, `tempa` and their values, the guess's log N; H I and D I stay
  excluded; ALIS reads it; one undo gives back the text). It fails on the code of
  Task 12.1 with the old refusal.
- `test_dashboard_qt_components.py`: `test_a_greyed_card_includes_an_ion_it_never_had`
  (a click on O I in the greyed card is one step, asks nothing, and the card is
  greyed only for H I; on O I it is enabled). It fails on the old code (no step).

**MWC137** (scratch `a2_include.py`): a2 takes each ion it never had, Ti II, Ca I,
Ca II, 87Rb I and Fe I, each line with `re`, `de` and `TZERO` and a first guess
(11.60, 9.29, 12.26, 8.91 and 11.73); the model reads without error, and one undo
gives back the text.

**Batches** after 12.2: unit 2239 passed, 119 skipped (9.2.0: 2238, 120); gui 303
passed (9.2.0: 303). Pre-commit passes.

### Task 12.3 — A column density that a link computes, in a drag (point 5; Q12.6 (a), (b))

**What was built** (`alis/dashboard/components.py`):
- `link_input(project, comp, ion)`: for an ion's log N that a link computes, the
  link's one input that is free (neither fixed nor linked), written on `variable`
  lines alone, and named by no other link; or None, and the reason, "log N is
  computed by the link ka = lia - lika" (the link's target and expression as
  written).
- `drag`: a linked log N asks `link_input`; its input's value is found so that the
  link gives the log N dragged (`_solve_input`: the secant method on ALIS's own link,
  `profile.values`, exact in one step for a sum or a difference), within the input's
  bounds, which stop the drag (D63). The change is the input's, so the step writes it
  and b's. Without an input, the reason is the `DragError`.
- `draggable(comp, ion, project)`: log N is draggable when its link has such an
  input. `linked_coldens` and `link_logn` (the log N a link computes with a drag's
  values) serve the readout.

**The Components tab** (`qt/components.py`): a drag starts from the log N the link
computes (`C.shown`); the readout reads "log N 14.62 (dh -4.62) · b_turb 25.10 km/s",
with ", at its limit" when the input is held at a bound; where log N stays, "b_turb
… · log N is computed by the link nha = nda - FIXDH"; a hidden input, or a hidden
log N, shows only Δχ² (D24).

**Tests:**
- `test_dashboard_components.py`: `test_a_linked_log_n_moves_its_link_s_one_free_input`
  (replaces `test_a_linked_parameter_cannot_be_dragged`: on `examples/summed_coldens`,
  na = log10(10**nf − …) and the drag moves `nf`, a nonlinear link, so that it gives
  12.2; a redshift that a link computes is still refused);
  `test_a_drag_moves_a_link_s_input_so_the_link_gives_the_log_n` (D I's log N by
  `nda = nha + dh`: dh moves to −4.3 for 10.2, b moves; ALIS reads the step; H I's
  log N stays; one undo); `test_a_limit_on_the_link_s_input_stops_the_drag`;
  `test_a_link_with_no_single_input_leaves_log_n` (a fixed other input, and a free
  input that a second link names: the reason, not draggable, `DragError`). They fail
  on the code of Task 12.2.
- `test_dashboard_qt_components.py`:
  `test_a_drag_of_a_linked_log_n_moves_the_link_s_input` (the readout; one step;
  `dh` written, D I's log N unchanged), `test_a_drag_of_a_log_n_its_link_holds_says_why`,
  `test_a_drag_of_a_hidden_link_input_shows_only_dchi2`. On Task 12.2's code the
  readout reads "b_turb 26.12 km/s" in each.

**MWC137** (scratch `a7_drag.py`): a7's K I: input `lika`, draggable; a drag to
log N 12.80 (from 12.50) moves `lika` to −2.80; the step reads without error, K I's
log N is 12.80, and Li I's stays 10.0.

### Task 12.4 — Ignore warning, for two snips that share pixels (point 1; Q12.3)

**What was built.**
- `snips.py`: `IGNORED_SHARED`, the key of `ui/project.json` that lists the pairs
  ignored; `shared_key(a, b)`, a pair's record: the two snips' files and a digest of
  each one's fitted pixels (wavelength, flux and error), so that a change to either
  snip's pixels or fit range warns again; `is_ignored`, `warned_pixels` and
  `ignored_pixels` (the pairs of `shared_pixels`, which is unchanged, with and
  without the ignored ones); `ignore_step` ("Ignore the warning for 6 and 7") and
  `warn_again_step`, each one step of `ui/project.json`.
- The warnings leave an ignored pair out: `markers._shared` (the Regions tab's
  marker and `fitted_twice`, so the pre-flight note too); `validate.without_ignored`,
  which drops ALIS's own warning ("… are fitted twice, by the snips", matched by the
  two files' names) from `validate.check` and from the `.mod` panel's problems
  (filtered as they are shown, so that a step of `ui/project.json` alone takes
  effect); the Regions tab's hatching (`_shared_spans`) and offer.
- The Regions tab's Snip box: **Ignore warning** beside Keep them in this snip only
  and Merge the snips… (its tooltip: until either snip's pixels or fit range change);
  under it, while a pair is ignored, "Warnings ignored: 147 pixels shared with snip
  6 (…)" and **Warn again**.

**Tests:**
- `test_dashboard_snips.py`, on `dashboard_synthetic.write_twin_snips` (two files
  on one grid, alike digit for digit but in a core, each fitted in
  [4254, 4264] Å; with `third`, a copy of the first):
  `test_ignore_warning_keeps_the_test_and_leaves_the_pair_unwarned` (the pair found
  before and after; the marker, the pre-flight check and ALIS's warning before, none
  after; the record's files; one undo gives back the state and the warnings),
  `test_a_wider_fit_range_warns_again` (a region added to one snip; Warn again takes
  the record out), `test_ignoring_one_pair_leaves_the_others_warned` (three snips).
- `test_dashboard_markers.py`: `test_a_pair_whose_warning_is_ignored_leaves_the_marker`.
- `test_dashboard_qt_regions.py`:
  `test_ignore_warning_for_two_snips_alike_but_in_a_core` (the button, one step, the
  box and the hatching gone, the list and Warn again, two undos, no design
  reference).

**MWC137** (scratch `ignore_mwc137.py`): before, the pair (6, 7, 147 pixels) is
found and warned about by the marker (attention), the pre-flight check and ALIS
(7698.022 to 7704.682 Å); after Ignore warning, it is still found, and the marker is
complete, with no pre-flight note and no ALIS warning.

### Task 12.5 — An emptied unassigned dataset stays, and says so (item 2 carried over; Q12.1)

**What was built.**
- `datasets.py`: `empty_unassigned(row)` (unassigned, with no snips) and
  `empty_advice(row, will=False)`: "other (unassigned) holds no snips, and has no
  source spectrum: remove it with Remove… (right-click its row on the Data tab), or
  give it snips with File → Load snips…." ("will hold" in CLEAR's preview). The
  dataset's Contents cell reads "0 snips · unassigned: no source spectrum !".
- `snips.clear_plan`: CLEAR of an unassigned dataset's last snip adds the advice to
  the preview's notes (the dialog shows them); a dataset with a source gets none.
- `markers.data_marker`: ! with the advice while such a dataset is there.
- The Data tab: the Contents cell's tooltip is the advice. The task's "Remove
  dataset…" is the row's right-click Remove…, as the advice says; it takes the
  dataset away as before (checked, not changed).

**Tests:**
- `test_dashboard_loadfit.py` (where Stage 11's tests make an unassigned dataset,
  with their helpers): `test_clear_of_an_unassigned_dataset_s_last_snip_keeps_it_and_says_so`
  (the preview's advice, none for a dataset with a source; the row kept with no
  snips; the Data tab's marker ! with the advice, ✓ before and after undo; the
  Contents cell; ALIS reads the model).
- `test_dashboard_qt_data.py`: `test_an_emptied_unassigned_dataset_says_so_until_removed`
  (the row's text and tooltip, the marker, then Remove… and ✓; no design reference).
- The existing tests of CLEAR and of unassigned datasets pass unchanged.

**Pre-commit, a note:** in zsh an unquoted `$F` is one word, so a `pre-commit run
--files $F` checks nothing ("no files to check"); the runs that counted named the
files, or used `${=F}`. Every file changed in 12.1–12.5 passes.

**Batches** after 12.3: unit 2243 passed, 119 skipped (9.2.0: 2242, 120); gui 306
(9.2.0: 306). After 12.4 and 12.5 (the snapshot also held 12.6's first part, the
Components tab's cards and panels): unit 2248 passed, 119 skipped (9.2.0: 2247,
120); gui 310 (9.2.0: 310).

### Task 12.6 — Every view keeps its place (point 3; Q12.4 (b))

**The helper** (`qt/widgets.py`): `scroll_place(area, pending=False)` and
`restore_place(area, place)` keep any scroll area's (or list's, or table's) place
across a rebuild. The place is given back at once and again each time the area's
range changes (the old contents going, then the new laid out), and twice more
after the layout pass, until the user scrolls (`actionTriggered`), something else
moves the bar, or 1.5 s pass. A capture made while a place is still waiting takes
that place (a second refresh before the layout pass would otherwise keep 0: found
by the Regions test, the window refreshing twice). `top_item` and `item_place` keep
a list's place as the item at its top and how far into it, so that a project
reopens on the same card or spectrum in a window of another size. `held_place` is
the context manager.

**Within a session:**
- Components: a click on a card selects it without scrolling to it; a selection
  made elsewhere (a tick, the `.mod` panel's cursor) is scrolled to after the layout
  pass (`selection_changed(scroll)`, `_show_selected_card` on a zero timer); the card
  list keeps its place as the cards are rebuilt. The panels keep their ranges
  through every step, a drag's included (`_panel_places`, `_restore_panels`, by each
  panel's snip and transition); another ion or dataset keeps the velocity range,
  its flux range at home; a new half-range (the Range menu) sends them home; ← and →
  keep their history while the same panels are drawn; the panels' scroll is kept
  while they are the same.
- Regions: the list of spectra (D114's ranges were kept already), the transition
  list (sideways too, while it lists the same transitions) and the Snip box's
  table keep their places.
- Fit · Inspect and Fit · Compare: the panels keep their ranges through every step,
  on the tab or another (`fit.panel_places`, `restore_panels`, shared); one ion's
  panels in velocity keep the shared velocity range from one ion to the next.
- The progress chart cannot be zoomed by the user (its mouse is off), so it follows
  its points, as the rule asks of a view the user has not zoomed: no change.

**Between sessions (Q12.4 (b)):**
- The window's size and position: a preference `window.geometry` (`[x, y, width,
  height, maximised]`), not shown in the Preferences dialog (`Spec.shown`,
  `preferences.SHOWN`; the dialog lists `SHOWN`); restored as the window is made,
  on the main screen when no screen shows the place; kept as it closes. The
  preferences' `_copy` held every list as rows of settings, and now keeps a list of
  numbers as it is.
- Each tab's places in `ui/view.json` under `places` (`session.PLACES`), by tab
  (`data`, `regions`, `components`, `inspect`, `compare`): `window.keep_places`
  reads them from the views only before Save, Save as, a clean close and a recovery
  copy; never as they move, so nothing lags, and they leave no unsaved changes (the
  view is not part of the project's state; a clean close keeps it, as it keeps the
  tab). Each tab takes its stored places as its session opens and applies them
  once, on its first drawing; a tab not drawn since gives back what was stored.
  What the project no longer has opens at ⌂.

**Tests** (the first four fail on the code of Task 12.3):
- `test_dashboard_qt_components.py`:
  `test_a_click_on_a_card_s_list_keeps_the_list_where_it_is` (a real click with
  QTest on the last card's Excluded ions near the bottom: before, the list went to
  0 from 2181), `test_the_panels_keep_their_zoom_through_a_step_and_a_drag` (two
  steps and a drag; another ion keeps the velocity range, its y at home; ⌂).
- `test_dashboard_qt_regions.py`: `test_the_lists_keep_their_places_through_a_step`
  (five spectra, the list at its end; the transition list scrolled sideways).
- `test_dashboard_qt_fit.py`: `test_inspect_keeps_its_view_through_a_step` (a step
  while Inspect is shown, one while another tab is; ⌂).
- `test_dashboard_qt_session.py`: `test_a_project_opens_where_it_was_left` (a fit of
  20 snips and 12 components: a panel zoomed, the panels and cards scrolled, an
  Inspect panel zoomed; zooming and scrolling call no `set_view` and leave no
  unsaved changes; `keep_places` under 5 ms; closed, the window made 1180 × 820,
  opened: the same ranges, the same card at the top, the panels scrolled, Inspect's
  zoom), `test_what_the_project_no_longer_has_opens_at_home`.
- `test_dashboard_session.py`: `test_the_places_survive_save_close_and_a_recovery_copy`.
- `test_dashboard_qt_window.py`: `test_the_window_opens_where_it_was_closed` (from
  the preferences, kept as it closes, on the main screen when off every screen, not
  in the dialog, a too small place refused).
- D114's test (`test_the_view_stays_where_the_user_put_it`) passes unchanged.

**Batches** after 12.6: unit 2249 passed, 119 skipped (9.2.0: 2248, 120); gui 315
passed (9.2.0: 315). Pre-commit passes.

### Task 12.7 — Zooming the flux axis, and ⌂ for each axis (point 4; Q12.5 (b))

**What was built** (`qt/plots.py`):
- `SpectrumBox.wheelEvent`: with ⇧ Shift held (Qt's modifiers with the wheel event,
  so no key is taken from the keyboard), the wheel zooms the flux axis about the
  cursor; without it, the wavelength (or velocity) axis, as before (D38). On a Mac
  ⇧ turns a mouse wheel's scroll sideways; pyqtgraph's zoom reads the delta in either
  orientation, so it zooms the same. The residual strips keep their own scale.
- `SpectrumView.home_x` and `home_y` (⌂y takes the flux axis home for the
  wavelengths shown: the view's home flux range, or the robust range of the data
  shown); `home` is the two.
- `ZoomBar`: ⌂x and ⌂y beside ⌂ (tooltips `HOME_X_TIP`, `HOME_Y_TIP`, the latter
  naming ⇧ Shift and the wheel), each a step of ← and →, acting on every view the bar
  controls (⌂x on the first of linked views).
- The shortcut sheet (F10) says what the wheel, ⇧ Shift, ⌂, ⌂x and ⌂y do
  (`actions.MOUSE_NOTE`, under its table).

**Tests:** in `test_dashboard_qt_plots.py`,
`test_shift_and_the_wheel_zoom_the_flux_axis` (y zooms about the cursor and x stays;
without ⇧, x zooms and y stays), `test_home_x_and_home_y_take_one_axis_home` (each
changes its axis alone, and ← takes each back), `test_home_y_acts_on_every_view_of_its_bar`;
in `test_dashboard_qt_window.py`, `test_every_zoom_bar_has_a_home_for_each_axis`
(the Data, Regions, Components, Inspect and Compare bars, the sheet's note, no design
reference). `test_the_wheel_zooms_wavelength_only` passes unchanged (its stand-in
event has no modifiers, which the box allows).

### Task 12.8 — Source spectra kept in the project (point 6; Q12.7 (a))

**What was built.**
- `session.py`: `Session.keep_copy(ident)` (`bundle.embed_source`: a spectrum present
  and unchanged) and `remove_copy(ident)` (`unembed_source`; refused while the file
  is not found, which would leave no spectrum: "relink it before removing the copy
  kept in the project"). Like Relink…, neither is a step of the history; the
  project has unsaved changes until Save (the sources are part of its state).
  `save_copy_with_spectra(path)`: a copy of the project (as Save would write it,
  with its view) with every source kept in it, written to another file; the open
  project, its file and its state as they were; a spectrum missing or changed is
  not kept, and is returned to be named.
- Save had not written the kept copies: `Session._owned` (and `_merge`, a
  recovery copy over the disk's, and `Project.save`) wrote the sources' list but not
  the `sources/` members. They now keep the members of the project's bundle.
- The Data tab: the row's right-click menu offers **Keep a copy in the project**, or
  **Remove the copy** once kept; the Checksum cell reads "✓ unchanged · kept in the
  project", or "kept in the project (its file is not found)" where the copy is read
  (`datasets.checksum(state, kept)`, `is_kept`). The task named a "Source column";
  the cell is the table's Checksum column.
- **File → Save a copy with spectra…** (`file.save_with_spectra`), with an open
  project: the copy's file chosen (`<name>_with_spectra.model` to start), then the
  user is told where it was written, and which spectra could not be kept.
- The relink question adds: "(A project written with File → Save a copy with
  spectra… keeps its spectra, and needs no relinking.)"
- A kept spectrum, its file gone, is read from the copy (`sources.spectrum`, as
  since Stage 3): SNIP, cutting again and the faint source spectrum use it.

**Tests:**
- `test_dashboard_session.py`:
  `test_a_copy_kept_in_the_project_is_read_when_the_file_is_gone` (not a step,
  unsaved until Save; the file deleted, the state is embedded with no problem, the
  spectrum is read; Remove the copy refused), `test_remove_the_copy_takes_it_out_at_the_next_save`,
  `test_save_a_copy_with_spectra_leaves_the_open_project_as_it_was` (the file and
  state unchanged; the copy's sources all kept, and no relinking with the spectrum
  gone; a changed spectrum named; the project's own file refused).
- `test_dashboard_qt_relink.py` (its setup has a project of one spectrum):
  `test_a_kept_spectrum_asks_no_relinking` (Keep a copy, the cell, Save, close, the
  file deleted, opened: nothing asked, the marker ✓, the cell; Remove the copy
  refused, told), `test_the_relink_question_names_save_a_copy_with_spectra`,
  `test_save_a_copy_with_spectra_from_the_file_menu`.
- `test_bundle.py`'s tests of embedding are not touched (ALIS's `bundle.py` is
  unchanged).

### Task 12.9 — in progress, paused at Q12.12 (2026-11-17)

**Done so far.**
- ALIS (`alis/functions/lsfspline.py`, excluded from the linters): the smoothing is
  the FWHM of the Gaussian, σ = FWHM / 2√(2 ln 2) in `call_CPU` (it was
  smoothing/√2); `set_vars` asks for `vfwhm`'s sub-pixels, 2√(2 ln 2) · nsubpix ·
  pixel / FWHM (it was √2 · …); the class docstring says what changed for a model
  written before. `alis/config.py`: `nsubmin` 10 (was 5), with
  `doc/ALIS_workflow.md`'s table and `doc/tex_files/description.tex`.
- `tests/test_lsfspline.py` (`unit`): a delta function smoothed by 2, 5 or 10 km/s is
  a normalised Gaussian of that FWHM to 10⁻⁶ of its peak, on grids linear in
  wavelength and in log wavelength (on the old code, 1.665 times as wide, as the
  Prompt 3 scratch check found); RJC's nine knots smoothed by 2.14 km/s are their
  lines convolved with that Gaussian, and with no smoothing their lines alone; the
  sub-pixels asked for equal `vfwhm`'s for 1, 2.14, 5 and 20 km/s. 11 pass.

**Found:** ALIS's `lsfspline` puts its kernel one sub-pixel to the red of its line,
with or without smoothing, on both grids (a spike at index 800 of 1601 comes out at
801; `vfwhm` keeps it at 800). The cause is the last convolution's slice,
`ret[idf:idf + ysize]` with `idf = int(iysize / 2 - 1)`, one element before the
kernel's centre, `iysize // 2`. It predates the stage. The tests compare shapes
about the kernel's own peak (`centred`), and
`test_the_kernel_is_centred_on_its_line` is a strict expected failure that a fix
turns into a pass. **Q12.12** asks whether to fix it in Task 12.9 (lean: yes) or to
defer it; development is paused here (Prompt 5's rule), since the regression
references of `examples/lsf_spline`, the dashboard's kernel and Tasks 12.10–12.12
follow from the answer.

**The `fast` batch with the change** (on a snapshot, pytest 9.0.2, 19 minutes): 27
failed, 122 passed. No case uses `lsfspline`, so all are `nsubmin`'s: mode (a) in 15
cases (9 shipped examples, each failing as a bundle too, and 6 context fits), mode
(b) in `lls` and `powerlaw`, and `examples/generate`'s data. Most by little; some
fits change their result (J0903p2628 a parameter 143.6 → 128.2, 3.5σ;
HS0105p1619 χ² 2086.3 → 2137.0; J1558m0031 118 values; `splineabs` linear a 1σ
error 113.2 → 6.2). **Q12.13** asks whether to regenerate the 16 cases' references
as Q12.11's response has it, or to keep the default 5 (lean); the working tree has
`nsubmin` 10 until the response.

**Not yet done in 12.9:** the regression references, if the default stays 10; `examples/lsf_spline` as
a case; the dashboard's `lsfknots.py` (knots, rules, delta function, kernel, step,
buffer) and its tests.

### Prompt 6 (2026-11-17): the responses to Q12.12 and Q12.13, and Task 12.9 finished

**Responses.** Q12.12 (a): fix `lsfspline`'s offset, checked for even and odd
numbers of knots; `examples/lsf_spline`'s references after it. Q12.13 (b): ALIS's
default `nsubmin` stays 5, noted in `deferred_work.md`. Applied: notes under both
responses, Task 12.9's text and checks, Task 12.12's check (ALIS changes in
`lsfspline.py` alone; Q12.1–Q12.13 answered); `deferred_work.md` §7.10 (`nsubmin` 10,
with what it moved) and §7.11 (below), priority 24. No further query.

**Task 12.9, finished.**
- ALIS: `nsubmin` back to 5 (`config.py` and the two documents as at Stage 11's
  close). `lsfspline.call_CPU` slices its last convolution from the kernel's centre,
  `iysize // 2` (it was `int(iysize / 2 - 1)`), so the model is centred on its line;
  the docstring says so. `tests/test_lsfspline.py`: the expected failure is a test
  now, for three, four, six and nine knots, with and without smoothing, on 1601 and
  1600 pixels (centroid within 10⁻³ of a sub-pixel, the two sides alike), with
  `test_a_delta_function_peaks_at_its_line`; the shape tests compare with no
  re-centring. 28 pass; all 28 fail on the code of Task 12.6.
- `examples/lsf_spline`: its data generated again (`generate_spectra.mod`, the new
  smoothing and centring). Its fit takes 9 s (0.002 h; reduced χ² 1.0003), and with
  references written from it, the minimisation and the bundle passed, but the
  evaluation at the best fit (mode (b)) stopped: "locations must be set". ALIS writes
  `lsfspline` into a `.mod.out` without its `locations` (`SplineLSF.parout` writes
  the values only), so the output model cannot be read back. As Q12.10's response
  allows, the case waits (`deferred_work.md` §7.11); its references were removed, and
  the regenerated data kept.
- `alis/dashboard/lsfknots.py` (new, no Qt): `Knots`; `knots_of` (a dataset's or a
  snip's `lsfspline`, or the three default knots smoothed by the resolution before
  as a FWHM in km/s, 5 km/s where none converts or it is hidden; a hidden
  `lsfspline` refused); `pixel_kms` (ALIS's `get_binsize`) and `spix` (the largest
  pixel of the snip or the dataset's snips over the model's `nsubmin`);
  `subpixels` and `model_subpixels` (ALIS's sub-pixellation for a snip with a
  smoothing); `delta_knots`, `is_delta`, `with_smoothing` (the edges follow the FWHM
  while the five knots stand; at least two sub-pixels out); `check_knots` (three
  knots, order, no two closer than spix, heights within [0, 1], the edges at 0, the
  centre at (0, 1) inside them, a smoothing ≥ 0); `half_width`; `kernel` (ALIS's
  `SplineLSF` on a spike, on sub-pixels of pixel/n, scaled to a peak of 1); `fwhm`;
  `labels`, `word` and `knots_step` (a dataset's lines sharing one set of labels, or
  a snip's own; the smoothing and the inner heights free, the edges and the centre
  upper case; `bintype` as `RS.bins_patches` gives; one step, `keep_structure`).
- `snips.buffers`: for `lsfspline`, the kernel's reach (its outer knot and three σ
  of the smoothing), where ALIS's `getminmax` takes the first value as a FWHM.

**Tests** (`tests/test_dashboard_lsfknots.py`, 21): RJC's nine knots read, written as
the dataset's and read back, one undo; another function's start (`vfwhm`, `vsigma`,
`Afwhm` converted at the snip's wavelength, a hidden one 5 km/s); a hidden
`lsfspline` refused; the delta function (spix, edges, rules, following the FWHM, a
Gaussian of the FWHM on `nsubmin` sub-pixels, a little wider on 21); a dataset's
spix; each rule's refusal; two knots one sub-pixel apart accepted, closer refused;
the kernel (a triangle of FWHM 10 km/s, wider smoothed, ALIS's function on a spike);
the sub-pixels; the step for a snip of its own and for a dataset (ALIS reads it, the
fixed and free heights, one undo); the buffer of a kernel 200 km/s wide.

### Task 12.10 — Set LSFSpline knots… (point 7, Prompt 3; Q12.8, Q12.9, Q12.11)

**What was built.**
- `resolution.py`: `Function.knots` (True for `lsfspline`), so that `offered` is
  True for it: the Snip box's and the Data tab's menus no longer grey it (the other
  functions of several parameters stay greyed). `KNOTS_TEXT` ("Set LSFSpline
  knots…") and `KNOTS_NOTE`. `datasets._resolution_words` refuses `lsfspline` (its
  knots are set in their dialog); `snip_resolution_step` gives a snip that takes its
  own copy of a dataset's `lsfspline` labels of its own (`lsfknots.labels`).
- `qt/knots.py` (new): `KnotsDialog` and its `KnotsBox`. The plot (pyqtgraph, D2):
  the knots as squares (grey fixed, blue free, the selected one highlighted, red
  where two are closer than one sub-pixel), a thin faint black line joining them,
  the kernel as ALIS makes it in red (normal weight), and its FWHM marked at half
  the peak with its value ("FWHM 9.61 km/s"). The Smoothing, FWHM (km/s) field;
  Delta Function (after the question "All the knots will be cleared, and replaced by
  a delta function.", Confirm or Cancel); Delete knot (also the right-click menu and
  the Delete key). A drag moves the knot it starts on (`move_knot`: the centre not
  at all, an edge sideways only, every knot at least spix from its neighbours, its
  height within [0, 1]); a double-click adds a knot between the edges, none within
  spix of another (the note says why). The kernel is drawn on the sub-pixels ALIS
  uses for the snip with the dialog's smoothing (`lsfknots.subpixels` with the
  model's ask). Knots that break a rule are marked, the rule named under the plot,
  and Apply greyed until they are apart. Apply makes `lsfknots.knots_step`; Cancel
  changes nothing.
- The Snip box: **Set LSFSpline knots…** beside the menu while the snip's own
  resolution is `lsfspline` (greyed, with the reason as its tooltip, while its
  values are hidden); choosing `lsfspline` opens the dialog at once
  (`set_lsf_knots`; the tab already had a `set_knots`, for the continuum's knots,
  which a first version replaced: caught by the Regions tests, and renamed). The
  note "set in the .mod panel" is no longer shown for `lsfspline`.
- The Data tab: choosing `lsfspline` as a dataset's function opens the dialog; the
  dataset's resolution menu offers Set LSFSpline knots… while it is `lsfspline`.

**Tests** (`tests/test_dashboard_qt_knots.py`, 7, in a file of their own rather than
the Regions and Data tests' files): the Data tab's dialog with the three default
knots, Cancel, Apply as one step, the menu's entry, one undo; the Snip box's button
and RJC's nine knots, no design reference; a knot within the rules (up to 1, down to
0, sideways to spix from its neighbour, an edge sideways only, the centre fixed, the
view's drag); knots added and deleted (none within spix; neither an edge nor the
centre); knots written too close, marked and Apply greyed until one is moved; the
FWHM shown is `lsfknots.fwhm`'s and follows the smoothing; Delta Function (the
question, Cancel, Confirm, the five knots, a Gaussian of the field's FWHM to 1%, the
edges following the FWHM, then the +spix knot up, outwards and inwards to spix, the
edges staying, the −spix knot deleted, Apply writing the knots as moved). In
`test_dashboard_datasets.py` and `test_dashboard_qt_regions.py`, the checks that
`lsfspline` is greyed now say it is offered, through its dialog alone.

**Batches** (run by the session of Prompts 5 and 6 on snapshots; their counts were
logged at Prompt 7, when the runs had finished):
- after 12.9 (`nsubmin` 5, `lsfspline` as now): unit 2300 passed, 1 failed, 119
  skipped (9.2.0: 2299, 1, 120); gui 322 passed (9.2.0: 322); `fast` 149 passed,
  the regression harness and its `test_bundle` mode included (9.0.2, 18 minutes);
- after 12.10 (the snapshot is the tree as Prompt 7 found it): unit 2300 passed, 1
  failed, 119 skipped (9.2.0: 2299, 1, 120); gui 329 passed (9.2.0: 329).

The one failure, in both, was `test_dashboard_actions.py::test_the_menus_of_the_design`:
its File menu was the one before Task 12.8 added Save a copy with spectra…. Fixed at
Prompt 7 (below).

## Prompt 7 (2026-11-17): the context cleared; Task 12.11, the review

**Read.** This document and this log in full; `dashboard_stage11.md` and its log
(Prompt 6 to the close: how Task 11.10's review was built and its nine comments
applied); `dashboard_stage0.md`'s plan and "After v1"; the opening of Stages 1–10;
`doc/dashboard/stage11/` (the builder of the last review) and the skeleton's helpers;
the `gui` tests of Tasks 12.1–12.10, whose scenes the screenshots reuse. No query
was needed before Task 12.11.

**Before the review.**
- *A test not updated in Task 12.8.* `test_the_menus_of_the_design` lists the File
  menu; Save a copy with spectra… joins it after Save as…. The unit batch then
  passes: 2301 passed, 119 skipped (9.2.0: 2300 passed, 120 skipped). The tree is
  otherwise the snapshot of the gui batch after 12.10 (329 passed in both), so that
  batch was not run again.
- *Pre-commit* passes on every changed `.py` file (50, `lsfspline.py` excluded from
  the linters as before) and on the two new scripts.

**Task 12.11 — The review.** Draft 1 published, waiting on RJC's comments.
- `doc/dashboard/stage12/take_screenshots.py OUTDIR [SCREEN ...]`: twelve screens,
  23 PNGs, all of the synthetic scenes of the stage's `gui` tests
  (`dashboard_synthetic.write_rubidium_fit`, `write_linked_fit`, `write_ratio_fit`
  with the links of the drag tests, `write_twin_snips`; `dashboard_fits`' spectrum
  copies; the metal example packed; the knots dialog's NINE and too-close lines).
  Dialogs and menus are drawn on their own; questions and messages as the window's
  message box draws them. Two ways found to draw them off-screen: the Data tab's row
  menu runs `QMenu.exec`, which a patch of the PySide class did not replace, so a
  timer grabs the open popup and closes it; the knots dialog keeps the `ask` it is
  made with, so the script gives it one that reads the window's at call time.
- `doc/dashboard/stage12/build_review.py SHOTS OUT.html`: the earlier pages'
  stylesheet, several screenshots per section, and a menu, a message or the zoom bar
  shown at its own size rather than stretched. Eleven sections: Rb I and 87Rb I
  (12.1); a greyed card before and after taking O I (12.2); the drag's readout with
  the link's input, and the reason where the link holds (12.3); Ignore warning and
  Warnings ignored (12.4); CLEAR's preview and the emptied row (12.5); a view zoomed
  in flux and the zoom bar with ⌂x and ⌂y (12.7); the row's menu, the File menu, the
  Checksum cell, what Save a copy with spectra… says, and the relink question
  (12.8); the Snip box with Set LSFSpline knots…, the dialog on RJC's nine knots
  (FWHM 4.69 km/s at a smoothing of 2.14), Delta Function's question and its five
  knots (edges ±37.71 km/s, inner knots ±0.28 km/s, FWHM 6.29 km/s for 6.285), and
  two knots written 0.01 km/s apart (12.10). "Also done": Task 12.6, with its eight
  tests named, since a still cannot show a place kept; Task 12.9's change to ALIS;
  `nsubmin` and `examples/lsf_spline` deferred; the ⇧ Shift wheel's test.
- Seen in the screenshots, and put to RJC on the page and as **Q12.14**: after Ignore
  warning, the Snip box's table still reads 401 in its Shared px column for the pair
  (lean: keep the count, with "(ignored)" after it). Also seen, by design: while
  knots break a rule, the dialog draws no kernel (`KnotsBox.redraw`).
- Published as a private page: https://claude.ai/artifact/L7piPoxPkHSdJ3gxxV1gPX
  (version 1), from the session scratchpad's `stage12_review.html`.

- *RJC's comment on draft 1* (the Ignore warning section's question, Q12.14): keep
  the number of overlap pixels listed; Warn again is a good idea (to change one's
  mind and merge, say); the number should follow a change to the overlap. Applied in
  `qt/regions.py` (`_shared_cell`, `IGNORED_TIP`): the Shared px cell reads "401
  (ignored)", or "450 (401 ignored)" where another pair is still warned about, with
  a tooltip; it is counted from the pixels as they are, as before, so it follows
  them (and a change brings the warning back). New `gui` test
  `test_the_snips_table_counts_an_ignored_pair_s_pixels` (`test_dashboard_qt_
  regions.py`): the mark and tooltip; both snips' fit ranges widened, the count
  grows, unmarked, and the warning is back; undone, marked again; Warn again
  unmarks it. It fails on draft 1's code at the mark. The response did not say
  whether to mark the count; the mark is the lean, noted under Q12.14 to be taken
  out if RJC prefers the bare number. Screens retaken, page republished as version
  2, the thread resolved (its auto-reply already stood).
- *RJC's comment on draft 2* (the Data tab after CLEAR, Task 12.5): why does the
  emptied dataset read "no source spectrum !"; it should be carried so that its snip
  can be cut out again. The thread's auto-reply guessed at a bug in the source link;
  there is none. An unassigned dataset has never had a source spectrum (Stage 11:
  its snips match none of the project's spectra and are loaded as they are); the
  scene's snip was cut from a noisy copy the project does not hold, and the cell read
  "unassigned: no source spectrum" before CLEAR too. But a scratch script
  (`reload_check.py`) found that the advice of Task 12.5 does not do what it says:
  Load snips… of the same file, after CLEAR, puts the snip in a new dataset, "other
  (unassigned) 2", and leaves the emptied one. Asked as **Q12.15** (lean: Load
  snips… puts an unassigned snip back into the empty unassigned dataset of the same
  name). The page's CLEAR section explains the source and asks the question (draft
  3, version 3). The thread is left open for RJC's answer: the auto-reply stands
  and another reply was not posted. Work paused here (Prompt 5's rule).
- *RJC's comment on draft 3* (the zoom bar, Task 12.7): ⌂x and ⌂y look like Δx and
  Δy; use ↔ and ↕. Applied: the buttons' glyphs (`qt/plots.py`, `ZoomBar`; the keys
  `home_x` and `home_y` and the tooltips as they were), the shortcut sheet's note
  (`actions.MOUSE_NOTE`), the docstrings, and the checks of
  `test_home_x_and_home_y_take_one_axis_home` and
  `test_every_zoom_bar_has_a_home_for_each_axis` (46 tests of the plots, window and
  actions pass). The ↔ already used as the edges' drag cursor is a different thing,
  and stays. Every screen retaken (each window shows a zoom bar), page republished
  as version 4 (draft 4), the thread resolved (its auto-reply stood). Task 12.7's
  text in this stage's document keeps ⌂x and ⌂y as written; the close's documents
  use ↔ and ↕.
- *RJC's response to Q12.15* (in the stage document; a page comment pointed to it):
  RJC's reading of what happens is right, and the rule wanted is new: every dataset
  keeps its data, assigned or unassigned, and is treated alike; an unassigned
  dataset may hold several snips, or each snip its own. Answered under the response
  (Stage 11 made unassigned datasets sourceless on purpose, Q11.9 (3) and D12, so the
  snip's file was the only copy, and CLEAR takes it with the data line). Written:
  **Task 12.12**, every dataset keeps its data (the close renumbered 12.13, its
  checks naming 12.12 and Q12.1–Q12.16), and **Q12.16** on four points (the data
  kept as the dataset's source, a copy kept in the project; every dataset without a
  source, those of a fit opened without a spectrum too; several snips joined, the
  first kept where they overlap; projects saved before given theirs at open), leans
  (a). Checked on the way: the bundle keeps a source's copy (`bundle.add_source`,
  `_embed`), and Save writes the `sources/` members since Task 12.8. Whether
  commenting a data line out keeps its file was not traced; Task 12.12 checks it.
  The page's CLEAR section says what follows (draft 5, version 5); the page thread
  that pointed to the response is resolved, and the first thread on this dataset
  stays open until Task 12.12 is built. Work paused for Q12.16.

## 2026-11-17 (RJC's response to Q12.16; Task 12.12)

RJC (in chat, after Prompt 7): Q12.16 answered (the leans: (a) for (i), (ii) and
(iv), (iii) as written); implement the next steps. Applied note under the response.

### Task 12.12 — Every dataset keeps its data (RJC's comment on the review; Q12.15, Q12.16)

**What was built.**
- `datasets.py`: `kept_table(project, row)` (a dataset's snips' pixels, in
  wavelength order; where two overlap, the first snip in the model kept; columns
  wave, flux, error, and the bad-pixel mask when a snip has one) and
  `keep_data_step(project)`: every dataset with snips and no source is given its
  data as a source entry marked `from_snips`, with its copy in the bundle
  (`sources/s<n>_<name>.dat`), its row's `source` and `columns` in
  `ui/project.json` (a guessed structure stays guessed), one step.
  `from_snips(row)`; `checksum(..., snips=True)` reads "kept in the project · its
  snips' pixels"; `empty_unassigned` asks for no source too, so Task 12.5's advice
  and ! are left for a dataset emptied before this change.
- `project.py`: `Project.apply` puts a step's `sources/` members in the bundle
  (not the model's files), so Undo takes the copy back with the source entry.
- `sources.py`: `FROM_SNIPS`, `from_snips(entry)`; such a source is read from the
  bundle alone (never looked for on disk, so never missing or changed), and
  `describe` says "<dataset>: its snips' pixels, kept in the project."
- `loadfit.py`: Load snips… and Load a fit (Open a fit's ways in) end their step
  with `keep_data_step`; their notes say that the unassigned snips' pixels are kept
  as the dataset's data.
- `session.py`: as a session is made (Open, Open a fit without its spectrum, a
  recovery copy), `keep_data_step` is applied to the project, not as a step; the
  disk's state is noted as read, before it, so that `check_disk` stays SAME.
  `remove_copy` refuses a source made from snips.
- `snips.clear_plan`: the advice only for an unassigned dataset with no source.
- The Data tab: the Checksum cell; no Keep a copy or Remove the copy for data kept
  from snips.
- SNIP needed no change: it cuts from any dataset with a spectrum
  (`snips.covering`).

**Q12.17, raised while building** (the stage document): Q12.16 (iv)'s lean said
"unsaved changes until Save". Built so, every bundle packed from a `.mod` and every
imported fit saved before opened as changed (the `session` tests' packed metal
example failed seven tests on it: dirty at open, recovery copies, `check_disk`
CHANGED). Built instead: the data kept at open are part of reading the file, with
no unsaved changes, and the next Save writes them (lean (a)); (b) is a one-line
change.

**Tests** (made first where they could fail before):
- `test_dashboard_loadfit.py`: `test_a_snip_loaded_unassigned_keeps_its_data` (the
  source, its pixels the snip's, EMBEDDED and no problem, the cells; one undo takes
  dataset, entry and member); `test_clear_leaves_an_unassigned_dataset_its_data_and_
  snip_cuts_again` (replaces Task 12.5's test: no advice, the data kept, ✓, SNIP
  covers it and cuts the snip's own pixels again); `test_commenting_a_snip_out_
  leaves_its_dataset_its_data`; `test_a_dataset_emptied_before_its_data_were_kept_
  says_so` (Task 12.5's advice, for a row with neither); `test_snips_in_one_dataset_
  join_their_pixels` (two O I snips 0.4 Å apart in one dataset of an imported fit:
  sorted, the first's pixels where they overlap, the second's beyond; one step, one
  undo). Changed for the rule: the notes of `test_load_snips_gives_each_file_a_fate`
  and `test_a_mod_whose_snip_is_not_from_the_spectra_loads_it_unassigned`, and
  `test_unassigned_snips_load_without_a_source` renamed `..._with_their_data`.
- `test_dashboard_session.py`: `test_a_bundle_without_its_spectra_keeps_its_data_as_
  it_opens` (the packed metal example: its dataset's data at open, not a step, not
  dirty, `check_disk` SAME, still guessed; Save writes it; opened again, nothing
  added).
- `test_dashboard_qt_data.py`: `test_an_emptied_unassigned_dataset_says_so_until_
  removed` became `test_an_emptied_unassigned_dataset_keeps_its_data` (the row's
  cells after CLEAR, ✓, the menu without Keep/Remove copy, SNIP on the Regions tab,
  Remove…).

**MWC137** (scratch `mwc_keep.py`, on a copy of `mwc137_relinked.model`): opened, not
dirty, no step; its three unassigned datasets get their data (39KI7701 (unassigned):
336 px, the snip's); after CLEAR of snip 6, the dataset keeps them and the marker is
✓; SNIP of K I 7701 covers it and cuts 336 px, 7697.87–7705.18 Å.

**Batches** (snapshot after the tests above): unit 2306 passed, 119 skipped (9.2.0:
2305, 120); gui 327 passed, 3 failed under both. The three were what Q12.16 (ii)
changes, a fit opened without its spectrum having data now:
`test_without_a_source_only_snipped_transitions` (D12's Regions tab: the snipped
transitions alone) became `test_a_fit_opened_without_its_spectrum_lists_the_
transitions_on_its_data` (J1358: 22 transitions on its data, 8 snipped, Ly9 among
the others; Load snips… enabled; "The transitions on the data"), and
`test_open_a_fit_with_no_project_asks_first` and `test_q1243_tie_reaches_the_model_
and_undo_removes_it` assert the data kept where they asserted no source. The three
pass; pre-commit passes on every file changed.

**The review**: the CLEAR section rewritten (the Data tab after CLEAR, and SNIP
again on the Regions tab, which the script refreshes after SNIP before the shot),
with Q12.17 asked there; every screen retaken (the imported scenes have their
data); page republished as version 6 (draft 6); RJC's first thread on the emptied
dataset resolved.

## 2026-11-17 (RJC's response to Q12.17)

RJC (in the stage document, after Prompt 7): revert Q12.16 (iv) and implement (b);
few `.model` files exist before v2 is released, users can start new ones, and it
avoids the complications of unsaved changes. Read as Q12.16 (iv) (b), "only
datasets loaded from now on" (the decision named, and the one that removes the
question of unsaved changes); the reading is in the applied note.
- `session.py`: `Session.__init__` is as it was at the stage's start (no data at
  open, no note of the disk before them); `Session.import_model` (Open a fit without
  its spectrum) applies `keep_data_step` before the session is made. The loads keep
  theirs as before (`loadfit`).
- `test_dashboard_session.py`: `test_a_fit_opened_without_its_spectrum_keeps_its_
  data` (the metal example imported: its dataset's data, no step, still guessed;
  Save writes them; opened again, the same and clean) and `test_a_project_saved_
  before_opens_as_it_is` (the packed metal example: no data added, clean,
  `check_disk` SAME) replace the test of data at open. Session and loadfit tests:
  69 pass.
- MWC137 (scratch `mwc_open.py`, on copies): `mwc137_relinked.model` opens as it
  was, clean, its three unassigned datasets with no data; `alis2.mod` opened again
  without its spectra gives every dataset its data (snip 6's, "39KI7701", 336 px),
  which stay after CLEAR of snip 6, and SNIP of K I 7701 covers it.
- The review: the CLEAR section says so, the question of Q12.17 is gone; the zoom
  screen retaken (it opens a packed bundle, again without data); draft 7, version
  7.

Batches after Q12.17's change (snapshot): unit 2307 passed, 119 skipped (9.2.0:
2306, 120); gui 330 passed (9.2.0: 330).

## 2026-11-17 (RJC: "This looks good, please proceed to Task 12.13 and close the stage")

### Task 12.13 — Close the stage

Done (CI on both platforms waits for RJC's push).
- *Coverage* (`pytest --cov` on what changed, from their tests): `lsfknots.py` 92%,
  `snips.py` 86%, `components.py` 88%, `datasets.py` 82%, `sources.py` 83%,
  `qt/knots.py` 85%. The untested paths of this stage: a kept dataset's bad-pixel
  mask (`kept_table`), `keep_data_step`'s rewrite of a confirmed structure that
  lacks a row, `sources.describe` for kept data, the knots' refusals ("Cut a snip of
  … first", "Its snips cannot be read"), and the dialog's raw mouse events,
  right-click menu and Delete key. Added: `test_kept_data_carry_the_snip_s_bad_
  pixels` (`test_dashboard_loadfit.py`: the mask kept with the data, and a snip cut
  again from them keeps pixels 10–12 bad), `describe`'s text in `test_a_snip_
  loaded_unassigned_keeps_its_data`, and the Delete key in `test_knots_are_added_
  and_deleted`. The others are wording or pyqtgraph's own event plumbing.
- *The documents*: `doc/ALIS_workflow.md` §4.3 ("Since Stage 12"); `CHANGELOG.md`
  (Added: Stage 12; Fixed: rubidium's names, Save writing kept spectra, `lsfspline`
  centred; Changed: `lsfspline`'s smoothing a FWHM, a model written before smoothed
  0.60 times as wide, multiply by 1.665 to keep it); `tests/README.md` (a Stage 12
  row); the `gui-component` skill's "Stage 12 additions"; decisions D129–D138 in
  `ALIS_v2_dashboard_prompts.md` (refining D8, D12, D19 and Q1.8, D22 and D37, D38,
  D48, D63 and D65, D64 and D79, D67, D89, D114, D122 and Q11.9, F13);
  `dashboard_stage0.md` (Stage 12 done, Stage 13 next, "After v1" 5a and 5b);
  `deferred_work.md` §1.4, §7.8 and §7.9 stay, with §7.10 and §7.11 from Task
  12.9.
- *Found and not done* → `dashboard_stage13.md`'s "Carried over from Stage 12", with
  Q13.1: two datasets of a fit opened without its spectrum named alike (`alis2.mod`:
  two "39KI7701", grouped apart by their resolution labels, named by the same
  `label=`); an unassigned dataset emptied before Task 12.12, which Load snips…
  does not fill. Checked and not carried: the Regions tab redraws after SNIP in the
  window (a probe `gui` test; the stale screenshot came from the script's
  stand-ins).
- *What follows* (Q12.2 (a)): `claude_prompts/dashboard_stage13.md` written, its
  Feedback section for RJC, with Q13.2.
- *Pre-commit, an error and its repair*: a first run over every changed file took
  its list from `git ls-files --others` too, so it ran on 31 of RJC's untracked
  scripts under `context/`, and reformatted 29 (2 failed to parse). They have no git
  copy; each was copied back from the `fast` batch's snapshot, taken just before
  (timestamps kept), and checked byte for byte. Pre-commit then passes on the
  stage's 54 changed files alone, with `context/` untouched. Claude's memory
  (`lint-with-pre-commit`) now says how to build the list.
- *The checks:*
  - Feedback point 1: Task 12.4; point 2: 12.2; point 3: 12.6; point 4: 12.7; point
    5: 12.3; point 6: 12.8; point 7: 12.9 and 12.10; point 8: 12.1; item 2 carried
    over: 12.5 and 12.12 (RJC's comment on the review). Each is logged above as done
    with its check passing. Items 1 and 3: `deferred_work.md` §7.8 and §7.9.
  - Q12.1–Q12.17 are answered in the Queries section.
  - The review (12.11) is applied: drafts 1–7, every thread resolved.
  - Batches (pytest 9.0.2; CI's 9.2.0 in a scratch venv), against Stage 11's close
    (2235 unit passed, 119 skipped (9.2.0: 2234, 120); 301 gui; 149 fast):
    - unit: 2308 passed, 119 skipped (9.2.0: 2307 passed, 120 skipped);
    - gui: 330 passed (9.2.0: 330), on the tree before the Delete-key check, whose
      file passes alone under both;
    - fast, the regression harness and its `test_bundle` mode included: 149 passed
      (9.2.0: 149 passed).
  - CI on both platforms, the gui job within its 90 minutes: waits for RJC's push.
  - No design reference in the window's text (D44): the `gui` tests' checks pass;
    pre-commit passes on every changed file of the stage.
  - ALIS outside `alis/dashboard/` changes only as Task 12.9 says: `git diff --stat
    1e4c2f4 -- alis ':!alis/dashboard'` lists `alis/functions/lsfspline.py` alone,
    and the regression harness passes unchanged. This stage's tests are synthetic
    and need none of RJC's files; MWC137 was checked with scratch scripts.

Stage 12 closed on 2026-11-17, CI on both platforms waiting for RJC's push. The next
stage is `dashboard_stage13.md` (Q13.1, Q13.2).
