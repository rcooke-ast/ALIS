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
