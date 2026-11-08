# Dashboard Stage 10 log

Feedback from users, after Stage 8's list closed, and RJC's own on the Figure tab
(D99, Q10.2). The plan is in `claude_prompts/dashboard_stage10.md`.

### 2026-11-08 (Prompt 1: the design, the tasks, and queries Q10.4–Q10.8)

**Read.**
- This stage's document: points 1–13 of the Feedback list, the three items carried
  over from Stage 9, and RJC's responses to Q10.1–Q10.3 (all three items join;
  the list as written is the stage's; a closed list, with Stage 11 next).
- `ALIS_v2_dashboard_prompts.md` in full: F1–F14, S1–S31, D1–D112 and QF.1–QF.36.
- `dashboard_stage8.md` in full (how feedback became tasks last time), and the end of
  Stage 9's log (the close, the carried items, RJC's Prompts 4 and 5).
- `deferred_work.md` §7 (nothing there overlaps the points).
- The code each point touches: `qt/regions.py` (the list, the strips, the tools, SNIP,
  the edges, the line IDs), `qt/plots.py` (the regions, the edges, the faint
  spectrum, the line IDs, the view range), `snips.py` (`snip_step`, `covering`,
  `strips`, `edges_step`), `continuum.py` (`fit`, `guess`), `lines.py` (`line_ids`,
  `_labelled_ions`, `good_pixels`, `line_name`), `qt/window.py` (`_run`, `refresh`),
  `actions.py` (the Regions tab's actions), `qt/components.py` (the card's rows, the
  temperature button), `figscript.py` (`label_text`, the residuals, `CORNERS`),
  `figpresets.py` (`dh`), and `alis/plotscript.py`'s labels.
- RJC's project `context/feedback01/J0035m0918.model` (three datasets: ESPRESSO,
  UVES2010 and J0035m0918), whose bundle finds its three source spectra at the paths
  it recorded.

**Reproduced** (scratch scripts, no code changed).
- *Point 8.* `snips.snip_step` on Al II 1670.79 and Si II 1526.71 raises the user's
  `ContinuumError`. UVES2010 has every error ≤ 0 (−10³², flux 1) from 4988.34 to
  5691.00 Å (15,804 pixels); both lines fall there (5580.6 and 5099.3 Å at z =
  2.34009). `covering` takes the dataset since its source has pixels there, "good or
  bad". ESPRESSO (300 good pixels) and J0035m0918 (239 of 240) would have been cut.
- *Point 7.* `lines.line_ids` on J0035m0918 at Lyα gives `Lyα 1215.7` alone, with
  nothing counted; at Lyβ, Lyβ and two O I lines. D I (4059.35 Å and 3424.9 Å) is
  neither labelled nor counted, since only main isotopes and named ones are.
- *Point 12.* `figscript.label_text` gives C II λ1335, S II λ1260 and Al II λ1671.
  RJC's reference scripts round down (C II λ1334, S II λ1259, N III λ989).

**Found in the code.**
- *Point 10.* Every action runs `window.refresh`, which redraws the tab on screen;
  `RegionsTab._fill_strips` keeps the x range but restores the y range only when one
  dataset is shown. So with RJC's three datasets, any action (Tweak included) resets
  each spectrum's y range.
- *Point 6.* A movable region's edge is 1 px with pyqtgraph's default hover pen (red,
  1 px); pyqtgraph 0.14 picks such a line within 1.5 px.
- *Point 2.* The faint spectrum is cut once (±0.6 of the snip's width), so zooming
  out shows nothing more; moving an edge recuts the snip and the context with it.
- *Point 13.* The residual strip (−0.3 or −0.2, 0.05 per σ) and a lower label (0.07)
  are placed in flux, so a panel's own flux range can hide them.
- *Points 1, 3, 4, 5, 9, 11* are as the list describes them; see the stage
  document's Design.

**Written** in `dashboard_stage10.md`:
- the Design's "10.1–10.17 — The points of the list": what was found for each point
  and carried item, and what stays as it is (the decisions refined: D21, D36, D53,
  D54, D59, D78, D93, D106, D110, D111);
- Tasks 10.1–10.17, each with its check, in this order: the Regions tab's bugs (SNIP
  over a dataset with no usable data; the isotopes' line IDs; the view kept), its
  left pane (the line IDs' choice, the list's order, the system's snips), its tools
  and spectra (the fit regions' menu, their edges near the cursor, the whole source
  drawn for the view, two handles on a snip's edges), the Components tab (T's
  value), the Figure tab (labels rounded down, residuals and labels kept in a
  panel, the DH preset's first member, the spacing between rows, the untested
  paths), and the close (with Stage 11's document, Q10.3 (a));
- Q10.4–Q10.8, each with its lean: Load snips… moved to the File menu (b); D59's
  labels kept as a fourth choice and the default (b); the second handle takes every
  edge to the same place (a); the whole source drawn for the view (b); `plotscript.py`
  rounded down too (b).

No code was changed. Waiting on RJC's responses to Q10.4–Q10.8; the tasks are then
revised where a response departs from its lean.

### 2026-11-08 (Prompt 2: RJC's responses to Q10.4–Q10.8, and the tasks checked)

- **The responses.** RJC took every lean: (b) for Q10.4, Q10.5, Q10.7 and Q10.8, (a)
  for Q10.6. Q10.7's response adds that no pixel may be cut from a snip by the view:
  a user zoomed in may mean to fit pixels out of sight. Q10.8's says the change
  should be everywhere.
- **Checked against the code** (every name, test, figure and claim of Tasks
  10.1–10.17):
  - pyqtgraph 0.14 draws about five samples per screen pixel (`autoDownsampleFactor`
    5), ten points by peaks, not two as Q10.7 (b) said. Every curve of the spectra
    already has `clipToView` and peak downsampling (`plots._curve`).
  - The clipping bisects x with `bisect_left`, and `plots.step_curve` puts a NaN in
    x at each gap: over 0–999 with a NaN at 500, the bisection for 700 gives 500. A
    whole source has gaps, so the faint curve could stop short of the view's edge.
  - The snip's edge is one `InfiniteLine` with its `<|>` marker at the middle,
    dragged from anywhere along it.
  - A typed value is written in every place of its label (`edit.set_value`), so a
    tie keeps one value; D93 splits a shared label for a mode. Task 10.11 as written
    split it for a typed value too, which breaks a free tie.
  - The line ticks are placed in flux (`TICK_HEIGHT` 1.03–1.15), like the residuals
    and the label of point 13.
  - The only labels that round a wavelength to a whole number are
    `figscript.label_text` (the label and the panel's placeholder) and both forms of
    `plotscript.py`. The specids already take the integer part (`snips.py`,
    `modes.py`), and `lines.line_name` gives one decimal.
  - The region tools have no keys; Delete follows `_linked()` (every dataset unless
    tweaking). Load snips… is greyed without a source spectrum, and shows the first
    snip loaded.
  - The isotopes of 10.2 are in the table (¹³C II 1334.519, 3.0 km/s from ¹²C II's;
    ³He; ²⁴⁻²⁶Mg II within about 1 km/s). J101155p294141's shared `ZEROT` is
    `SHARED_T` in `test_dashboard_components.py`, not `tests/dashboard_synthetic.py`.
    UVES2010 has 110,712 pixels; Stage 9's figures (94% and 92%, the batch counts,
    CI's 90 minutes) match its log.
- **Revised.** Design (points 2, 11, 12, 13; what stays), and Tasks 10.5 (shortest
  first; the tests in fλ order), 10.6 (Q10.4 (a)'s conditional gone; Load snips…
  as now, from the File menu), 10.7 (no keys; Delete), 10.9 (the data, not the
  settings; drawing never cuts data; x finite at gaps; the faint curve made once;
  the points per pixel), 10.10 (where the drag begins decides the grip), 10.11
  (Q10.10; `SHARED_T`), 10.12 (what "everywhere" covers), 10.13 (the ticks, Q10.9)
  and 10.17 (Q10.9–Q10.10 in the decisions; the files this stage needs). Notes under
  Q10.7 and Q10.8 say what was applied, and correct Q10.7 (b)'s two points per
  pixel.
- **Raised** Q10.9 (the ticks: lean (b), by fractions of the panel) and Q10.10 (T
  typed on a shared label: lean (c), by the mode). Tasks 10.11 and 10.13 are written
  with the leans.

No code was changed. Waiting on RJC's responses to Q10.9 and Q10.10; the tasks are
then ready to execute, 10.1 first.

### 2026-11-08 (Prompt 3: RJC's responses to Q10.9 and Q10.10)

- **The responses.** RJC took both leans: Q10.9 (b), the ticks placed by fractions of
  the panel "so that they are always visible regardless of the flux range"; Q10.10
  (c), a T typed under a fixed mode acts as Fixed at… (the component alone, D93),
  and under T free or pure thermal is written in every place of its label.
- **Checked.** The two claims of Task 10.11 not checked at Prompt 2: voigt limits T
  below at 0 (`functions/voigt.py`, `_limited`), and D24 lets the user type over a
  masked value. No other part of the document had changed.
- **Revised.** Tasks 10.11 and 10.13 were written with the leans, so only their
  wording changed: the conditionals on Q10.9 and Q10.10 go, and 10.13 checks the
  ticks for each residual choice. Design (point 13; what stays), the notes of the
  header, the Design, the Tasks and the Queries say the responses are in.
- **The tasks.** 10.1–10.17 are numbered in order, each with its check; points 1–13
  and the three items carried over each have one task, and 10.17 closes the stage.
- No new queries. The tasks are ready to execute, 10.1 first. No code was changed.

### 2026-11-08 (Prompt 4: the tasks, in order)

**Task 10.1 — SNIP passes over a dataset with no usable data (point 8).** Done.
- *Tests first* (`test_dashboard_snips.py`): `unusable_copy` writes
  metal_line_abs's spectrum with flux 1 and error −10³² within ±350 km/s of O I 1302,
  as UVES2010 has in its gap. Before the fix, `covering` took the second dataset.
  - `test_snip_passes_over_a_dataset_with_no_usable_data`: the first dataset alone
    is cut, the second named, and still shown (`strips`); ALIS reads the model once
    it has a region, and two undos restore every byte.
  - `test_snip_is_refused_when_no_dataset_has_usable_data`: a `SnipError` naming
    both.
  - `test_a_first_guess_that_cannot_be_fitted_is_a_constant`: good pixels in O I's
    core (12) and two beside it. The guess's passes of rising order then have too
    few pixels outside the cores (a third-order pass needs four), and the snip
    starts with a constant through the median of its good pixels. (Good pixels all
    in the cores do not fail: `usable` then fits them all.)
  - `test_j0035_snip_passes_over_uves2010`: Al II 1670 and Si II 1526 are cut from
    ESPRESSO and J0035m0918, and UVES2010 is named. It passes here; it is skipped
    where the bundle or its sources are absent.
- *The fix* (`snips.py`): `covering` now needs the transition on the source's pixels,
  as before, and at least `MIN_PIXELS` (10) good pixels within the window
  (`usable_pixels`, by `lines.good_pixels`). `unusable` gives the rows passed over,
  and `unusable_text` says why (errors zero or negative, marked bad, or not finite).
  `snip_step` takes a `report` dict (skipped, constant, message) and is refused with
  the message when no dataset is left. The first guess's `ContinuumError` is caught
  in `add_snip_makers`, and `continuum.constant_guess` writes a Legendre of order 0
  through the median (Load snips…, which uses the same makers, gains this too).
- *The Regions tab*: SNIP and a region drawn where there is no snip show the message
  after the step, and catch `ContinuumError` as a message. A region drawn on the
  dataset passed over goes to the others under Add regions to all. The strip of a
  dataset with fewer than 10 good pixels reads "· no usable data here".
  `test_snip_passes_over_a_dataset_with_no_usable_data` (`test_dashboard_qt_regions.py`)
  checks the titles, the one SNIP message, and the region drawn on the second strip.

**Task 10.2 — The isotopes' line IDs (point 7).** Done.
- *Tests first*: `test_the_isotopes_of_the_ion_shown_are_labelled`
  (`test_dashboard_lines.py`): with H I shown and a system naming H I alone, D I's
  Lyα and Lyβ are labelled, −81.6 km/s from H I's, coloured as hydrogen; with O I
  shown, or no ion (the other tabs), as before. `test_an_isotope_within_a_few_km_s_
  shares_its_line_s_label`: ¹²C II and ²⁴Mg II shown add no label and no count.
  In the Regions tab, Lyβ on metal_line_abs's spectrum at z = 0.2695 draws
  "D Lyβ 1025.4", and on J0035m0918 D I is drawn beside Lyα and Lyβ in every strip.
- *The change* (`lines.py`): `line_ids(..., shown=)` adds the shown ion's isotopes
  (`isotopes_of`: one element and stage, any mass) and labels them as the ion's own
  lines are (named, or strong). One within `GROUP_KMS` of a main-isotope line of
  the ion is neither labelled nor counted. `_draw_ids` passes the ion shown.

**Task 10.3 — The view stays where the user put it (point 10).** Done.
- *Test first*: `test_the_view_stays_where_the_user_put_it[3]` failed at the first
  tool chosen (the last spectrum's y range back at its home); `[1]` passed before
  and after. With the last spectrum zoomed by its box, choosing each tool, drawing
  a region, adding a knot and an undo leave every spectrum's x and y ranges as they
  were, to 10⁻⁹; ← and → still step through the zooms; ⌂ takes each spectrum home;
  a transition shown again opens at home.
- *The change*: `_fill_strips` keeps every strip's ranges by dataset while the same
  transition is redrawn, and gives a spectrum that appears the others' x range and
  its own home y range. `ZoomBar.attach(..., keep_history=)` keeps ← and → across a
  redraw of the same spectra, where it had cleared them at every action.

**Task 10.4 — What the line IDs label (point 4, Q10.5 (b)).** Done.
- *Tests first*: `test_what_the_line_ids_label` (`test_dashboard_lines.py`), on
  metal_line_abs at z = 0 with O I 1302 shown: Every line counts nothing and labels
  Si II 1304; This transition labels O I 1302 alone and counts the rest; This ion
  labels O I's lines in view; the default equals the call without a choice; the
  transition out of view is counted with the rest; with H I shown, This transition
  labels H I's and D I's Lyβ. `test_the_line_ids_menu` (`test_dashboard_qt_regions.py`):
  the four entries, the default, the labels and counts on both spectra, no step,
  and This ion read back after the project is closed and opened.
- *The change*: `lines.LABELLED` and `line_ids(..., which=, rest=)`. This transition
  and This ion take Every line's IDs and keep the shown system's lines of the ion
  and its isotopes (`_shown_only`): for the transition, the one within `GROUP_KMS`
  of the rest wavelength shown, and each isotope's nearest within 300 km/s. The
  rest are counted. The Key pane's "Line IDs:" menu sets `state["line_ids"]`, kept
  in `ui/view.json` with the tab's view.

*Batches after Tasks 10.1–10.2* (pytest 9.2.0, in a scratch venv, as CI runs them):
`unit` 2180 passed, 120 skipped (2176 at Stage 9's close, plus 4); `gui` 259 passed
(256, plus 3), in 9 min 19 s.

**Task 10.5 — The transitions by wavelength or by strength (point 9).** Done.
- *Tests first*: `test_the_list_by_wavelength_or_by_strength` (`test_dashboard_lines.py`):
  on a flat source covering Si II 1190, 1193, 1260 and 1304, the rows are in order
  of rest wavelength by default, and in fλ order by strength, the same rows. In the
  Regions tab (same name): the Order menu, the note, the transition shown kept when
  the order changes, ↓ to the next row in the order shown, and the order read back
  after the project is closed and opened.
- *The change*: `lines.transitions(..., order=)` with `WAVELENGTH` (the default; a
  row of near-coincident lines by its first line) and `STRENGTH`. The Order menu
  under the list sets `state["order"]`, kept in `ui/view.json`. With no transition
  chosen yet, the strongest row is still selected first, whatever the order: by
  wavelength, the first row would be the shortest (Ly31 on J1358p6522).
- The task named ‹ › for stepping the list; those step the element and the ion.
  The list's keys are ↑ ↓, which follow the order shown, and are what the test
  checks.
- The tests that read the list in fλ order now choose by strength:
  `test_j1358_hydrogen_list_matches_the_mockup` (lines), and `test_j1358_hydrogen_
  list` and `test_without_a_source_only_snipped_transitions` (Regions tab).

**Task 10.6 — The system's snips in one list, and Load snips… moved (point 5,
Q10.4 (b)).** Done.
- *Tests first* (`test_dashboard_qt_regions.py`): `test_the_snips_of_this_system`, on
  two datasets: "No snips yet", greyed; SNIP adds "O I 1302.2 · 2 datasets", then
  Si II 1304 its own, in order; choosing O I shows it (Element, Ion, list, spectra);
  CLEAR of both its snips leaves "— choose a snip (1) —" and Si II; undo brings
  O I back with one dataset; system b (z = 0.0338) lists only its Si II 1260, and
  system a not that. On J1358p6522, each snipped transition of the primary is
  listed once, in order, with one dataset. `test_load_snips_places_them_or_loads_
  nothing` now runs from the File menu (after Open a fit (.mod)…, no button on the
  tab), and from the Data tab, which gives way to the Regions tab; the action is
  greyed for J1358p6522, which has no source spectrum.
- *The change*: "Snips of this system:" under the list (`_fill_snips_menu`, filled
  at every redraw from the snips' transitions; `_snip_chosen` calls
  `show_transition`). `file.load_snips` replaces `regions.load_snips`; the window's
  `do_file_load_snips` brings the Regions tab forward and runs its loader, and the
  window greys it, with the reason, while no source spectrum can be read.
- *Found and fixed*: `show_transition` and a system chosen replaced the tab's state
  whole, so the Model's Start/Best fit, and now the line IDs' choice and the list's
  order, went back to their defaults whenever the coverage grid, Load snips… or
  the System box showed a transition. They are now kept (`SETTINGS`).

*Batch after Tasks 10.3–10.6*: `unit` 2184 passed, 120 skipped, 1 failed:
`test_the_menus_of_the_design` lists the File menu, which now has Load snips… after
Open a fit (.mod)…; the list was updated (7 passed).

**Task 10.7 — Adjust fit regions: one menu (point 3).** Done.
- *Tests first*: `test_adjust_fit_regions` (`test_dashboard_qt_regions.py`), on two
  datasets: the label and its three entries, off by default; one spectrum greyed
  until the selected dataset has a snip; all spectra adds a region to both, one
  spectrum to the selected alone; Delete removes from the selected alone on one
  spectrum, and from both when off; off pans; Add knot sets the menu to off, and
  all spectra turns Add knot off; neither old name is in any text of the window.
  `test_a_drawn_region_reaches_the_mask_column` and the two-dataset region test now
  choose through the menu; `test_the_key_and_the_tools` checks that SNIP and CLEAR
  are the tools row's only buttons (and counts the Key's Line IDs row).
- *The change*: the tools row reads SNIP, CLEAR | Adjust fit regions: [menu]. The
  actions `regions.draw` and `regions.tweak` stay, renamed "Adjust fit regions: all
  spectra" and "… one spectrum", for the shortcut sheet; each entry triggers its
  action, and `_sync_regions_menu` shows the tool that is on (off when Add knot is
  chosen). One spectrum's entry is greyed as its action is. The tooltips, the Snip
  box's "No fit region yet: Adjust fit regions ▾ draws one.", and the docstrings
  (`regions.py`, `fit.py`, `results.py`, `snips.py`) no longer say Tweak or Add
  regions to all.

**Task 10.8 — A fit region's edge shown near the cursor (point 6).** Done.
- *Test first*: `test_a_fit_region_s_edge_is_shown_near_the_cursor`
  (`test_dashboard_qt_plots.py`): a movable region's edges have no pen (they had a
  1 px green one); their hover pen is 3 px in the new darker green; the cursor is
  ↔; the scene picks each 4 and 5.5 px from it and not 10 px; hovered it is drawn 3
  px, and not when the cursor leaves; a drag begun 5 px away moves it
  (`regionMoved`), drawn throughout; not movable, it is neither drawn nor moved.
- *The change* (`plots.py`): `_show_near_cursor` gives the edge `COLOURS["fit_edge"]`
  (#00694d) as its hover pen, 3 px, and a reach of `EDGE_PICK` = 6 px. pyqtgraph has
  no setting for the reach: it picks a line within (marker size + half its widest
  pen + 1) px, so the edge's `_maxMarkerSize` is set to 3.5, with a comment. A drag
  keeps the hover pen. The snip's orange edges are unchanged.

*Batch after Tasks 10.3–10.6*, continued: `gui` 262 passed, 2 failed (9 min 32 s):
`test_the_key_and_the_tools` (the Key's count and the tools row, rewritten with
Task 10.7) and `test_keys_only_while_the_tab_has_the_focus` (by wavelength, Lyβ,
chosen first as the strongest, is J1358p6522's last row, so ↑ reaches Lyγ; the test
now steps up, then down). Both pass.

**Task 10.9 — More of the spectrum as the user zooms out (point 2, Q10.7 (b)).** Done.
- *Tests first*:
  - `test_a_curve_with_a_gap_is_drawn_to_the_view_s_edges` (`test_dashboard_qt_plots.py`):
    500 pixels, a gap, 500 more, viewed over 300–900. Before the fix the curve drawn
    stopped at 499.5, the gap: pyqtgraph's clipping bisects x, and met the NaN that
    `step_curve` put there. `test_steps_are_centred_and_break_at_gaps` now wants the
    break in the flux alone, with x finite and in order (and `broken_line` too).
    Two tests that found the breaks by a NaN in x look for it in the flux.
  - `test_the_whole_source_is_drawn_for_the_view` (`test_dashboard_qt_regions.py`),
    on a synthetic source of 1250–1360 Å with O I 1302: the faint curve holds the
    whole source, clipped and reduced by peaks; zoomed out to ten snip widths it is
    drawn to both ends with at most 12 points per pixel; zoomed in to 50 pixels
    every pixel in view is drawn, and the snip's file and regions are unchanged; a
    region drawn and an undo reuse the faint curve; an edge dragged five widths out
    extends the snip there.
- *The change*: the faint curve is the source's whole step curve (`_context_of`,
  made once per source object and kept), before SNIP as after; `set_context(steps=)`
  draws it as given. Its flux is tamed (`tame`), since a wild pixel anywhere in a
  whole source would make Qt drop the curve. `step_curve` and `broken_line` break a
  gap with x at the gap's edge and a NaN in the values, for every curve they draw.
- *Timing on J0035m0918* (O I 1302 in three datasets, 1440×900, off-screen; the
  median of five): redraw of the tab 0.071 s before, 0.072 s after; one frame zoomed
  out to the whole range (3128–9466 Å) about 1 ms before and after. After, the faint
  curves hold 220,132, 221,424 and 154,864 points, and draw 4,114, 5,750 and 2,840 in
  a 633 px view.
- *Corrected*: the task's "about twice the plot's width" was Q10.7's error; the
  check allows the ten points per pixel pyqtgraph draws (noted at Prompt 2).

**Task 10.10 — Two handles on each of a snip's edges (point 1, Q10.6 (a)).** Done.
- *Tests first*:
  - `test_an_edge_moved_in_every_dataset_goes_to_one_velocity` (`test_dashboard_snips.py`):
    three datasets shifted by 0, +10 and −15 km/s, with a region in each. Out to
    −400 km/s from the first: every low edge at −400 km/s in its own frame, to one
    pixel; the high edges unchanged; ALIS reads the model; one undo restores every
    byte. Out to −490 km/s: the third stops at its source's first pixel (1300.0 Å)
    and is named.
  - `test_a_snip_s_edge_has_a_grip_for_every_snip` (`test_dashboard_qt_plots.py`): one
    marker without `every`, two with it; a drag begun at the middle or low on the
    line sends `edgeMoved`, one begun at the top grip `edgesMoved`; an edge moved
    quietly sends nothing.
  - `test_the_top_grip_moves_every_snip_s_edge` (`test_dashboard_qt_regions.py`): with
    three datasets each edge has two grips; the others' edges follow during the drag
    and the snips are unchanged until release; release makes one step ("Move the low
    edge of O I 1302.2 in 3 datasets") that undo takes back; the middle grip moves
    one snip; with one dataset, one grip.
- *The change*: `snips.edges_all_step` composes one `edges_step` per snip, each edge
  at the dragged velocity moved by its dataset's shift; it stops where a dataset's
  data end (its source, or without one its snip) and leaves a snip whose file is
  shared, naming both. `plots.EdgeLine` decides the grip where the drag begins
  (`TOP` 0.93 for the ▼ marker, `GRIP` 0.86); `move_edge_quietly` lets the other
  spectra follow. The key reads "snip edges: ◀▶ this snip, ▼ every snip".

**Task 10.11 — T's value on a card (point 11, Q10.10 (c)).** Done.
- *Tests first*:
  - `test_a_t_typed_on_a_card` (`test_dashboard_components.py`), J101155p294141's
    shape: 12000 typed on a1 under T free is its start, still free, one step that
    undo takes back; a word and −5 K are refused (T's own limit, 0 K); under pure
    thermal 0.5 is refused with its 1 K limit, and 8000 is written with the mode
    kept; under 100 K fixed, 250 gives a fixed 250.
  - `test_a_t_typed_on_a_shared_label`: on `SHARED_T` (a1, a2 and a forest line on
    `ZEROT`, fixed at 0 K), 250 on a2 is Fixed at… with its note, and a1's and the
    forest line's lines are unchanged byte for byte; two components tied on a free
    `tsh`, 9000 typed on a1 writes 9000 on both, the tie kept, and the description
    says "Set T of a1 (and of a2, on its label tsh)".
  - `test_dashboard_qt_components.py`: `test_t_s_value_on_a_card` (the field, the
    badge without Free or Fixed, a typed value and its undo, the refusal shown and
    nothing changed, the fixed value, a hidden T masked in the field and its badge
    and typed over); `test_t_typed_on_a_shared_fixed_label_asks_first` (the preview,
    then `ZEROT` kept by the others); `test_a_t_a_link_computes_is_read_only`.
- *The change*: `components.temperature_value_plan` (fixed modes go through
  `temperature_plan(…, "fixed", value)`; free and pure thermal check `Param.limits`,
  ALIS's own and any `lim param`, then `E.set_value` in every place of the label).
  The card's T row has the value field and badge (`_value_row(…, typed=)`), its
  badge without Free and Fixed, and the mode button after them; `_typed_temperature`
  shows the preview when the plan has notes, as a mode does.
- *Found and fixed*: a fixed mode (0 K, 100 K, 10⁴ K or Fixed at…) on an unlabelled
  T, as RJC's J101155p294141 writes them, raised `LabelNeeded`, which the card did
  not catch: the window reported an unexpected error. `temperature_plan` now ties
  the component's T under a label of the scheme first, as pure thermal does.

*Batch after Task 10.10*: `unit` 2186 passed, 120 skipped (2 min 0 s).

**Task 10.12 — A transition's label rounds its wavelength down (point 12, Q10.8
(b)).** Done.
- *Tests first*: `test_a_label_rounds_its_wavelength_down` (`test_dashboard_figscript.py`):
  C II 1334.5323 → λ1334, S II 1259.519 → λ1259, N III 989.799 → λ989, Al II
  1670.7886 → λ1670, Si II 1260.4221 → λ1260, 1000.0 → λ1000, 1239.9999999 → λ1240.
  `test_a_panel_label_rounds_its_wavelength_down` (`test_plotscript.py`): the same
  in both of `run_alis`'s forms ("H I 1025", "Lyβ 1025" for `1Ly_b`). Before, both
  rounded to the nearest whole number.
- *The change*: `plotscript.whole_angstroms` (the integer part, with a 10⁻⁶ Å guard)
  for `plotscript._pretty_ion`, and `figscript.label_text` uses it, so the label the
  script writes and the panel's placeholder agree.
- The drawn figure was not read back for this: metal_line_abs's panels (O I
  1302.17, Si II 1304.37) round alike either way, and the script's label is
  `label_text`'s (`figscript.py`, the panels' description).
- The regression harness runs with the batches below, since `plotscript.py` is
  ALIS outside the dashboard.

**Task 10.13 — The residuals and a label stay in a panel (point 13, Q10.9 (b)).** Done.
- *Test first*: `test_the_residuals_and_a_label_stay_in_a_panel[none|strip|inside]`
  (`test_dashboard_figscript.py`), on metal_line_abs's METALS figure with a label in
  the lower left. At the default range: the label at 0.07 in flux, the ticks at 1.03
  to 1.15, and every residual point at its old flux (offset + 0.05 per σ), to
  10⁻⁹. Ranged (0.5, 1.2): the panel keeps that range; the label and the bands lie
  inside it; every residual point is at the same place in the panel as before.
  Ranged (−0.1, 1.1): the ticks lie inside. The read-back now takes each line
  through its own transform into data coordinates, and gives each text's and each
  residual's place up the panel.
- *The change* (`figscript.py`): `panel_fractions` works out, from the figure's
  default flux range for its residuals, the fractions of a panel that put the
  residuals, a lower label and the ticks where they were in flux. The script writes
  them as `RESIDUAL_OFFSET`, `RESIDUAL_SCALE`, `LOWER_CORNER` (used by `CORNERS`)
  and `TICK_HEIGHT`, and draws them with `ax.get_xaxis_transform()` (x in data, y
  a fraction of the panel); every corner is now in fractions of the panel.
- *Found and fixed*: a panel's own flux range did not stand. The script set the y
  ticks after the range, and matplotlib widens a range to show every tick it is
  given, so (0.5, 1.2) with the default ticks (0, 0.5, 1) was drawn as (0, 1.2).
  The ticks are now set first.

*Batch after Task 10.10*, continued: `gui` 269 passed, 1 failed (10 min 4 s):
`test_j1358_cards_as_in_the_mockup` lists each card's value fields, which now have
T's (Task 10.11); the list was updated, and the Components tab's 44 tests pass.

**Task 10.14 — The DH preset starts at a chosen member (item 1 carried over).** Done.
- *Tests first*: `test_dh_from_a_chosen_member` (`test_dashboard_figpresets.py`): the
  synthetic D/H fit offers Lyα and Lyβ; from Lyβ, one dataset gives Lyβ alone and
  the figure keeps it; from Lyα, today's layout; two datasets give Lyβ in each
  column; a start below every member is refused; a new figure reads None (Lyα);
  `set_option` refuses `dh_from`. `test_j1358p6522_from_ly8` (skipped where absent):
  Ly8 first, and nothing below it. `test_the_dh_preset_from_a_member`
  (`test_dashboard_qt_figure.py`): the From menu shows with DH alone, lists "from
  Lyα" and "from Lyβ", Apply uses the one chosen as one step that undo takes back,
  and the menu shows the figure's member again.
- *The change*: `FG.DEFAULTS["dh_from"]` (None, so a figure saved before reads as
  Lyα, through `_complete`); `figpresets.dh_members`, and `dh(…, start=)` keeping the
  members from that rest wavelength up the series, passed through `layout` and
  `apply`. The Figure tab's From menu sits beside the preset; Lyα applies the preset
  as before.

**Task 10.15 — Room below a middle row's own x title (item 2 carried over).** Done.
- *Tests first*: `test_room_below_a_middle_row_s_own_x_title` (`test_dashboard_figscript.py`):
  metal_line_abs in two rows (7 × 3 in), Si II above Si II but in observed
  wavelength, with its own x numbers and title. At the old spacing, set by hand,
  the title's box (read back from matplotlib) reaches below the top of the panel
  under it; worked out, it clears it, and `spacing_of` shows the same values. With
  the panel back in velocity and no title of its own, the spacing is today's
  (0.05, 0.1); set by hand, it is the figure's. `test_the_spacing_worked_out_or_set`
  (`test_dashboard_figure.py`): None by default, a pair kept, the refusals outside
  0–2, a figure saved before Stage 10 with the old default read as Automatic and
  one with its own kept, and the old default set by hand since kept.
  `test_the_spacing_is_worked_out_unless_set` (`test_dashboard_qt_figure.py`): the
  Automatic box, the values it shows, unticked and edited, and ticked again.
- *The change*: `FG.DEFAULTS["spacing"]` is None, with `FG.SPACING` the scripts'
  (0.05, 0.1). `figscript.auto_spacing` keeps it unless a panel with a panel
  directly below it has its own x numbers (0.25 in) or x title (0.3 in), then
  gives the rows that room plus 0.05 in, as matplotlib's `hspace`, at most 2. A
  panel over an empty cell is numbered too, but nothing meets it, so it adds no
  room. `spacing_of` serves the Layout pane, and `_layout_panels` is shared with
  `margins_of`. `_complete` reads a figure with no `dh_from` (saved before this
  stage) and the old default as Automatic, once, so that the default set by hand
  later is not taken back.

**Task 10.16 — The paths Stage 9 left untested (item 3 carried over).** Done.
- *Tests*: `test_a_drag_begun_on_a_cell` (`test_dashboard_qt_figure.py`): `QDrag` is
  replaced, while the test runs, by a subclass whose `exec` records the drag's
  parent and data. A press and a move short of Qt's start distance make no drag;
  past it, one drag, belonging to the tab, carrying "0,1"; a press on an empty
  cell makes none. `test_a_size_or_spacing_out_of_range_is_refused`: the Layout
  pane shows each refusal (a size of 40 or 0.2 in, a spacing of 2.5 or −0.1), and
  the figure and the history are unchanged. `test_a_size_out_of_range_is_refused`
  (`test_dashboard_figure.py`): each bound of the size, with its message, and the
  bounds themselves accepted, for the size and the spacing.
- *Found and fixed*: a cell's drag began after a fixed 8 px, short of Qt's start
  distance (10 px here, and a user's setting): the test's 8 px move began one. It
  now uses `QApplication.startDragDistance()`.
- *Coverage* (`test-coverage`, the four figure test files): `figure.py` 96% (94% at
  Stage 9's close), `qt/figure.py` 94% (92%), `figscript.py` 96% (94%),
  `figpresets.py` 91% (91%).

**Task 10.17 — Close the stage.** In progress.
- *The documents* (written while the batches ran): `doc/ALIS_workflow.md` (File →
  Load snips…; the Regions tab's order, snips list, Adjust fit regions and the
  edges' grips; T typed on a card; a "Since Stage 10" paragraph after the Figure
  tab's); `CHANGELOG.md` (`plotscript`'s labels rounded down, under Changed);
  `tests/README.md` (a Stage 10 row); the `gui-component` skill ("Stage 10
  additions"); `ALIS_v2_dashboard_prompts.md` (D113–D117, each naming what it
  refines: D21, D36, D53, D54, D55, D59, D78, D93, D106, D110, D111);
  `dashboard_stage0.md` (Stage 10 built, closing after the review; Stage 11 in the
  table and the "After v1" list); `dashboard_stage11.md` (Q10.3 (a): its Feedback
  section for RJC, one item carried over, the line IDs' choice on the other tabs,
  with Q11.1 and Q11.2).
- *Still to do*: the review page (`doc/dashboard/stage10/`), RJC's review applied,
  and the close's checks.
