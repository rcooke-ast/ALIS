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
