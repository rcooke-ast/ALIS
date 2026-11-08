# Prompt file for ALIS software dashboard creation -- STAGE 10

> **Feedback from users, after Stage 8's list closed.** Stage 8 took a closed list of
> users' feedback, and Stage 9 built the Figure tab. Stage 10 takes the feedback
> that has arrived since Stage 8's list closed (Q8.4, D99), and any on the Figure tab.
> RJC collects each point in the Feedback section below. Claude turns the list into a
> design, queries and tasks, each with its check, and builds them in order, with a
> review before the stage closes.
>
> The rules of Stages 3–9 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D112 at the close of Stage 9).
> "Q0.*n*" to "Q9.*n*" refer to `dashboard_stage0.md` to `dashboard_stage9.md`. What
> follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-08 (Stage 9's Prompt 6). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is.*

*The Design, Tasks 10.1–10.17 and Q10.4–Q10.8 were written by Claude on 2026-11-08
(Prompt 1), from points 1–13 of the Feedback list and the three items carried over
(Q10.1 (a)). No code was changed. They were revised the same day (Prompt 2) after
RJC's responses to Q10.4–Q10.8, which took every lean, and after each task was
checked against the code; Q10.9 and Q10.10 were raised then. RJC took both leans
(Prompt 3), and the tasks are final.*

## Feedback

*Collected by RJC from users of the dashboard: one point per item, numbered, added as
it arrives. For each point:*
- *what the user did and saw, and what they expected instead;*
- *its kind: **(bug)**, **(change)** or **(new)**;*
- *the tab or dialog it concerns;*
- *where it can be reproduced: a project, or the model and data, and the steps. A
  user's files are referenced from here, and each test that needs them is skipped
  where they are absent (D99);*
- *who raised it, only if they agree to be named.*

*A point already on the list is not added twice: a note goes under the first, saying
how many users raised it. Claude reads this list and does not add to it.*

1. **(change)** Regions Tab: When a user drags the snip handles, it would be good to have two handles. One handle drags the edges for just this snip, and the other handle drags the edges for all snips simultaneously. This would make it easier to adjust multiple snips at once.
2. **(change)** Regions Tab: It appears like the default amount of the spectrum available to snip is too small and cannot be adjusted. However, dragging the handles into a region that doesn't show data makes new data appear. It would be better if there was a way to show more data as the user zooms out. Then more spectrum becomes available to snip when they release the drag. If you think something else might work better, please suggest it. We don't want to continuously update the amount of spectrum available during a drag, as it might slow down the user interface. We also don't want to show all pixels of the entire spectrum by default, as this could also slow down the user interface. We want to show a reasonable amount of spectrum by default, and then allow the user to zoom out to see more spectrum if they want to.
3. **(change)** Regions Tab: There are two options at the top of the Regions window: "Add regions to all" and "Tweak dataset regions" with a label that says "Fit regions:". I think it would be more intuitive to have a label that says: "Adjust fit regions:" with a drop down list that has three options: "all spectra", "one spectrum", "off". The "all spectra" option would allow the user to adjust the regions for all spectra simultaneously. The "one spectrum" option would allow the user to adjust the regions for just one spectrum. the "off" option would turn off the ability to adjust regions. This would make it more clear what the user is doing when they adjust regions.
4. **(change)** Regiosn Tab: Each panel displays the number of lines not labelled. It would be good to have a button that toggles between showing all lines and only a limited number of lines. For example, there could be an option that shows the main line only that is selected for this ion, or all lines from the selected ion that appear in this window's wavelength range, or all lines from all ions covered by the shown wavelength range. This would make it easier to see the main lines and not be overwhelmed by too many lines.
5. **(change)** Regions Tab: We no longer need the button that says "Load snips". This button can be removed, since there is a way to load the snips from a `.mod` file instead. Something that would be beneficial, instead, is a list box that contains *all* available snips for this System (every ion and transition that has been SNIPped). This allows the user to see what transitions have already been snipped, and to select them for display. When a snip is selected from this drop down list box, the corresponding element/ion should be updated to match, near the top of the Transitions box. Note that this list box should only contain snips that are available for this System only, and not for other Systems. If the user updates the System from the System box, the list of available snips should be updated to match the new System. This would make it easier for the user to see what snips are available for this System, and to select them for display and edit. Every time a new SNIP is created, it should be added to this list box. If a snip is deleted, it should be removed from this list box. This would make it easier for the user to manage all of their snips in one place.
6. **(change)** Regions Tab: When the cursor is near the edge of a region, there is a faint red/orange line that appears that allows the user to drag the edge of the region. It would be good if this line was a lot more visible, about as thick as the orange "Snip edges" lines. The line should only be shown when the cursor is near the edge of a region, and not when the cursor is far away from the edge. This would make it easier for the user to see where they can drag the edges of a region.
7. **(bug)** Regions Tab: When one of the H I lines is selected, the isotopes of H I and not being shown. It would be good to show the tick marks of an isotope.
8. **(bug)** Regions Tab: I encountered a bug when pressing the SNIP button:
```
The ALIS dashboard has encountered an unexpected error.
             Traceback (most recent call last):
               File "/Users/rcooke/Software/ALIS/alis/dashboard/qt/window.py", line 700, in _run
                 handler()
                 ~~~~~~~^^
               File "/Users/rcooke/Software/ALIS/alis/dashboard/qt/regions.py", line 2228, in do_snip
                 step = SN.snip_step(
                     self.session.project,
                 ...<5 lines>...
                     **self._options(),
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/snips.py", line 671, in snip_step
                 step = compose(
                     project, "", add_snip_makers(news, max_order, line_kms, clip, counted)
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/snips.py", line 101, in compose
                 step = maker(project)
               File "/Users/rcooke/Software/ALIS/alis/dashboard/snips.py", line 528, in make
                 best = CO.guess(
                     proj, snip, "legendre", max_order, clip=clip, line_kms=line_kms
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/continuum.py", line 810, in guess
                 use = continuum_pixels(
                     project, snip, function, knots, xrange, clip, line_kms, max_order
                 )
               File "/Users/rcooke/Software/ALIS/alis/dashboard/continuum.py", line 661, in continuum_pixels
                 keep = fit(
                        ~~~^
                     project.registry,
                     ^^^^^^^^^^^^^^^^^
                 ...<9 lines>...
                     start=keep,
                     ^^^^^^^^^^^
                 ).keep
                 ^
               File "/Users/rcooke/Software/ALIS/alis/dashboard/continuum.py", line 520, in fit
                 raise ContinuumError("There are too few good pixels to fit the continuum.")
             alis.dashboard.continuum.ContinuumError: There are too few good pixels to fit the continuum.
```
This appears to be a problem with the continuum fitting in a region where no data, or bad data exist. If it is just one dataset where the problem is, we could alert the user to there being an issue with dataset `<dataset name>` and that they should check the data. We should still allow them to proceed using any other datasets that do not have concerns. This error occurred when I was trying to press the SNIP button for the Al II 1670 transition after loading three datasets. The corresponding data are provided as context in the following folder: `context/feedback01/`. I suspect the error is with the `UVES2010` data. I've included the `J0035m0918.model` file in that directory. This also happens with the Si II 1526 line.
9. **(change)** Regions Tab: Currently the transitions are sorted by strength. It would be good to have a way to sort the transitions by wavelength, or by strength. This would make it easier for the user to find the transition they are looking for. The default should be ordered by wavelength.
10. **(change)** Regions Tab: When the user selects the "Zoom box" and zooms into a region of data they are interested in, and then selects the "Tweak dataset regions" option, the y-axis range resets to the full range of the data. It would be good if the y-axis range didn't reset automatically. After all, there is a "Home" button that allows the user to reset the ranges themselves, if they wish. We should not automatically change how the user has positions the y-axis and x-axis ranges.
11. **(change)** Components Tab: When the temperature is set to "T free", there is no way to set the temperature value. Like all other variables, there should be a way to set the temperature value, when it is either free or fixed. This would make it easier for the user to set the temperature value, without having to adjust the `.mod` file directly.
12. **(bug)** Plot Tab: Each panel shows a transition's label, with the element, ion stage, and wavelength. The current version of the Plot tab *rounds* the wavelength to the nearest integer. The standard in the field is to round down, so C II λ1334.5323 becomes C II λ1334 (but it is currently incorrectly shown as C II λ1335). 
13. **(bug)** Plot Tab: If I change the y-axis scale of a panel, the residuals and the label are no longer shown (they are probably drawn, but are not visible due to the different y-axis range shown).

## Carried over from Stage 9

*Found by Claude during Stage 9 and not done there. RJC decides whether each joins
this stage (Q10.1).*

1. **(change)** The DH preset always starts at Lyα. A figure of the higher members,
   as `DH_Ly8-Ly14*.py` draw them, needs its cells filled by dragging; the preset
   could instead start from a member the user chooses.
2. **(change)** A panel's own x title in a middle row (a wavelength panel among
   velocity ones, as `DH_Lyseries.py` has) can meet the panel below it. The margins
   are worked out from the panels (D111), but the spacing between rows is not; the
   user widens it by hand.
3. **(tests)** The paths Stage 9 left untested: a drag begun with the mouse on a cell
   (the drop side is tested with real drop events, but `QDrag.exec` cannot run
   off-screen); and the refusals of a figure size or spacing out of range.

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D112 and the stage documents. A point that asks for what a decision ruled out
  becomes a query that quotes the decision, not a task.
- **A bug becomes a test first.** It is reproduced from the user's project where there
  is one, or from a synthetic project otherwise (`tests/dashboard_synthetic.py`, and
  the helpers of `test_dashboard_figpresets.py` and `test_dashboard_figscript.py` for
  the Figure tab). The test fails before the fix and passes after it.
- **Outside the dashboard.** A point about ALIS itself, or about the documentation
  (planned in `documentation_stage0.md`), becomes a query: the dashboard's stages
  change ALIS only where a query agrees.
- **Users who disagree.** Two points that ask for opposite things become one query,
  with both described.
- **Grouping and order.** Points are grouped by the part of the window they touch
  (Data, Regions, Components, Fit, Figure, the `.mod` panel, Export), and a bug comes
  before the work that builds on it.
- **Shown before it closes.** Every change a user sees is shown on a review page,
  built by `doc/dashboard/stage10/` with the earlier pages' stylesheet, and RJC's
  comments on it are applied before the stage closes.

### 10.1–10.17 — The points of the list

*Written by Claude on 2026-11-08 (Prompt 1), from points 1–13 of the Feedback list,
the three items carried over from Stage 9 (Q10.1 (a)), RJC's J0035m0918 project in
`context/feedback01/`, and the code at the start of the stage. Each point's design is
in its task below. "Plot tab" in points 12 and 13 is the Figure tab (D102). Revised
the same day (Prompt 2) after RJC's responses to Q10.4–Q10.8, which agreed with every
lean, and again (Prompt 3) after the responses to Q10.9 and Q10.10, which did too.*

**What was found.** Each bug was reproduced, or its cause found in the code, before
its task was written.
- **Point 8** (J0035m0918: SNIP on Al II 1670 and on Si II 1526). Reproduced without
  the window: `snips.snip_step` stops with the user's traceback for both lines.
  - UVES2010's source has no data from 4988 to 5691 Å: 15,804 pixels with flux 1 and
    error −10³². Both lines fall in that gap, at 5580.6 and 5099.3 Å.
  - `snips.covering` counts a dataset as covering a transition when its source has
    pixels there, "good or bad". SNIP therefore cuts UVES2010's 240 pixels, all with
    an error ≤ 0, and the continuum's first guess (`continuum.guess`) has no pixel to
    fit. The error is a `ContinuumError`, not a `SnipError`, so the window reports it
    as unexpected. The snips of ESPRESSO (300 good pixels) and J0035m0918 (239 of
    240) are lost with it.
  - The Regions tab shows UVES2010's spectrum there too, flat at 1, titled "no snip"
    (`snips.strips` uses the same rule). The transition list and the Data tab's
    coverage count good pixels only (`lines.good_pixels`), so they were right.
  - The bundle finds the three source spectra at the paths it recorded, in RJC's
    research folder, outside `context/`. If the spectra are copied beside the bundle
    in `context/feedback01/`, the bundle's relative paths find them wherever
    `context/` is. Either way, the test that needs them is skipped where they are
    absent (D99).
- **Point 10.** Every action redraws the tab on screen (`window._run`, then
  `refresh`), including choosing Tweak dataset regions. The Regions tab then rebuilds
  its spectra (`_fill_strips`). It keeps the x range, but it keeps the y range only
  when there is one dataset. With two or more, each spectrum's y range goes back to
  its home after every action: a tool chosen, a region drawn, a knot added, an undo.
- **Point 7.** The line IDs (`lines.line_ids`) label every main isotope, but other
  isotopes only when the system names them (`_labelled_ions`). On J0035m0918, whose
  system has no D I, Lyα and Lyβ are labelled. D I's lines, 82 km/s bluer, are
  neither labelled nor counted.
- **Point 4.** The Regions tab labels each system's ions and strong lines, and counts
  the others ("+ 2 lines not labelled", D59). D59's switch for every line is on the
  Data tab only. The Regions tab offers no choice.
- **Point 9.** The list is ranked by fλ, "strongest first" (D36).
- **Point 5.** Load snips… sits under the list (Task 7.1, D78). It is the only way to
  load snips that come without a model. A `.mod` brings its own snips (Open a fit
  (.mod)…, Load a .mod…).
- **Point 3.** The fit regions' two tools are checkable buttons after the label "Fit
  regions:" (D54). Add knot, in the Continuum box, is the third of the exclusive
  tools.
- **Point 6.** A fit region's edge is a pyqtgraph line 1 px wide. It is drawn green
  while a region tool is on, and under the cursor it takes pyqtgraph's default hover
  pen (red, 1 px). In pyqtgraph 0.14 a line that thin is picked only within 1.5 px,
  so the edge is hard to find, and faint when found. The snip's orange edges are 3 px
  wide (5 px under the cursor), and their marker widens the pick to about 15 px.
- **Point 2.** The faint spectrum around a snip (`set_context`) is cut once, 0.6 of
  the snip's width either side. Before SNIP, the data shown are the ±300 km/s window
  plus half its width either side. Zooming out shows nothing beyond these. Moving an
  edge outwards recuts the snip, and the faint spectrum is cut again around the new
  extent: that is why "new data appear". Every curve of the spectra is already drawn
  clipped to the view and downsampled by peaks (`plots._curve`); only the data the
  faint curve is given are short. The clipping bisects the curve's x, and the steps
  break at a gap with a NaN in x, which a bisection can meet (`bisect_left` over
  0–999 with a NaN at 500 finds 700 at 500).
- **Point 1.** Each edge is a single handle, and moves only its own snip's edge
  (`edges_step`). The other datasets' snips keep theirs.
- **Point 11.** A card's T row is only the mode button (`_temperature_button`). Its
  value is shown only in a fixed mode, and set only through Fixed at…. z, b_turb and
  log N each have a value field and a badge (`_value_row`). A value typed there is
  written in every place of its parameter's label (`edit.set_value`), so a tie keeps
  one value. A temperature mode, Fixed at… included, changes its own component alone,
  splitting a shared label first (D93).
- **Point 12.** `figscript.label_text` writes `λ{rest:.0f}`, giving C II λ1335, S II
  λ1260 (1259.52) and Al II λ1671. RJC's scripts in `context/plotting_examples/`
  write C II λ1334, S II λ1259 and N III λ989. ALIS's own plotting script
  (`alis/plotscript.py`, for `run_alis`) rounds in the same way, in both its forms
  ("H I 1026", and "Lyβ 1026" for an ion named as a Lyman member). These are the
  only labels that round a wavelength to a whole number: the snips' specids already
  take its integer part (`snips.py`, `modes.py`), and the other names of a
  transition give one decimal ("O I 1302.2", `lines.line_name`).
- **Point 13.** The residual strip is drawn at a fixed flux (−0.3, or −0.2 inside
  the panel; 0.05 per σ). A label in a lower corner sits at a flux of 0.07, just
  above the zero level (D110). A panel's own flux range that starts above about
  −0.4 cuts off the strip, and one that starts above 0.07 hides the label as
  well. The line ticks are placed in flux too (1.03 to 1.15, just above the
  continuum), so a range that ends below 1.15 cuts them. Point 13 does not name
  them, but they take the same fix (Q10.9 (b)).
- **The items carried over** are as Stage 9's log describes them. The DH preset
  (`figpresets.dh`) always fills from the strongest member. The spacing between rows
  is the figure's `[0.05, 0.1]` until it is set by hand. The cells' drag start and
  the refusals of a size or a spacing out of range are untested.

**What stays as it is.**
- ALIS outside `alis/dashboard/`, except `alis/plotscript.py`'s labels (Q10.8 (b)).
- The bundle's format and the snip format (D4). `ui/view.json` gains the line IDs'
  choice and the list's order. A figure in `ui/project.json` gains the DH preset's
  first member and an automatic spacing. A project saved before reads them with
  their defaults.
- The other tabs' line IDs and views, which no point is about.
- The decisions D1–D112, except where a point or a query refines one: D36 (the
  list's order), D53 (what SNIP cuts), D54 (the fit regions' tools), D59 (the line
  IDs), D78 (Load snips…, Q10.4), D21 and D93 (T's value, Q10.10 (c)), D106 and
  D110 (labels, residuals and ticks, Q10.9 (b)), and D111 (the spacing).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage10_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (10.1, 10.2, …), and per item carried over
from Stage 9 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*Tasks 10.1–10.17 were written by Claude on 2026-11-08 (Prompt 1), and revised the
same day (Prompt 2) after RJC's responses to Q10.4–Q10.8. Every response took the
lean, so the designs stand. Each task was then checked against the code, which
changed the tasks below. RJC's responses to Q10.9 and Q10.10 (Prompt 3) took the
leans that 10.11 and 10.13 were written with, so only their wording changed:*
- *10.5: the order by wavelength runs from the shortest, and the tests that read the
  list in fλ order follow;*
- *10.6: the conditional for Q10.4 (a) goes;*
- *10.7: the tools have no keys, and Delete follows the menu;*
- *10.9: pyqtgraph's clipping and downsampling are already on, so the change is the
  data the faint curve is given. RJC's response adds that drawing never cuts data.
  The steps' breaks at a gap keep x finite, since a NaN in x stops pyqtgraph's
  clipping short, and the faint curve is made once rather than at every redraw;*
- *10.10: the edge is one line, dragged from anywhere along it, so where a drag
  begins decides the grip;*
- *10.11: a shared T label under a free mode, and its test's shape (Q10.10 (c));*
- *10.12: what "everywhere" (Q10.8's response) covers;*
- *10.13: the line ticks (Q10.9 (b)).*

*Each task names its point of the list, or its item carried over, and the query that
decides its design. The order puts a bug before the work that builds on it, and
keeps work on one part of the window together:*
- *the Regions tab's bugs first (10.1–10.3), since the later work there redraws the
  same spectra and line IDs;*
- *the Regions tab's left pane: the line IDs' choice, the list's order and the
  system's snips (10.4–10.6);*
- *the Regions tab's tools and spectra: the fit regions' menu, their edges, more of
  the spectrum, and the snip's two handles (10.7–10.10);*
- *the Components tab (10.11);*
- *the Figure tab: its two bugs, then the items carried over (10.12–10.16);*
- *the close (10.17).*

**10.1 — SNIP passes over a dataset with no usable data (point 8, bug;
`snips.py`, `qt/regions.py`; D53, D86).**
- **The bug.** SNIP cuts every dataset whose source has pixels at the transition,
  usable or not. A stretch whose errors are all zero or negative (UVES2010 at Al II
  1670 and Si II 1526) leaves the continuum's first guess no pixel to fit. The error
  then stops SNIP for every dataset, and is reported as unexpected (see Design).
- **The fix.**
  - A dataset covers a transition when its source has at least `MIN_PIXELS` (10)
    good pixels within the window. Good pixels are those of `lines.good_pixels`:
    finite, with a positive error, and not marked bad. `snips.covering` and
    `snips.strips` both use this rule.
  - SNIP cuts the datasets that cover the transition. Once the step is made, one
    message names the others: "J0035m0918_UVES2010 has no usable pixels within
    ±300 km/s of Al II 1670.8 (their errors are zero or negative), so it was not cut.
    Check its data." The step's description says how many datasets were cut. When no
    dataset is left, SNIP is refused with the same words, as a `SnipError`.
  - The first guess never stops SNIP. A dataset with too few good pixels for it
    (all inside line cores, say) is still cut, with a constant continuum through the
    median of its good pixels, and the message says so.
  - The Regions tab still shows such a dataset's spectrum, so that the user sees
    why. Its title reads "· no usable data here" in place of "· no snip".
  - SNIP, and Add regions to all (which snips first), report a `ContinuumError` as a
    message, as Auto first guess does (D86).
- **Check.** Made into tests first.
  - In `test_dashboard_snips.py`, on a synthetic project of two datasets made from
    `metal_line_abs`'s spectrum, the second with every error ≤ 0 within ±300 km/s of
    O I 1302:
    - SNIP cuts the first dataset alone and names the second. ALIS reads the model,
      and one undo restores every byte;
    - with both datasets bad, SNIP is refused with the message;
    - a dataset whose good pixels all lie in the line's core gets a constant
      continuum, and the message says so.
  - In `test_dashboard_qt_regions.py`, on the same project: SNIP shows the message,
    never the unexpected-error dialog. The second spectrum's title says it has no
    usable data.
  - RJC's project (skipped where its sources are absent): SNIP on Al II 1670, and on
    Si II 1526, cuts ESPRESSO and J0035m0918 and names UVES2010.

**10.2 — The isotopes' line IDs (point 7, bug; `lines.py`, `qt/regions.py`; D59).**
- **The bug.** With H I shown, D I's lines are neither labelled nor counted unless
  the system names D I (see Design).
- **The fix.** The Regions tab always labels the isotopes of the ion it shows, as it
  labels that ion's own lines: D I with H I, ³He with ⁴He, ¹³C with ¹²C, and so on.
  They take the colour of the ion's kind (H I and D I share one in the key, D59).
  - `lines.line_ids` is told the ion shown. The other tabs' line IDs are as now.
  - An isotope's line closer to its main isotope's line than `GROUP_KMS` (7 km/s)
    shares that line's label rather than adding one on top of it (¹³C II 1334, 3
    km/s from ¹²C II's; Mg II's isotopes).
  - Every choice of Task 10.4 includes them.
- **Check.** Made into tests first.
  - In `test_dashboard_lines.py`, on a synthetic system that names H I alone: with
    H I shown, D I's Lyα and Lyβ are labelled, 82 km/s bluer than H I's; with
    another ion shown, they are not. ¹³C II and Mg II's isotopes add no label.
  - In `test_dashboard_qt_regions.py`: with Lyβ shown, D I's label is drawn.
  - RJC's project (skipped where absent): D I is labelled beside Lyα and Lyβ.

**10.3 — The view stays where the user put it (point 10, change;
`qt/regions.py`).**
- **Now.** With two or more datasets, every action sets each spectrum's y range back
  to its home (see Design).
- **The change.** A redraw of the same transition keeps each spectrum's own x and y
  ranges, whatever the number of datasets. This holds after a tool is chosen, a
  region is drawn or moved, a knot is added, after SNIP, CLEAR, undo and redo, and
  after an edit in the `.mod` panel.
  - The ranges are kept by dataset. A spectrum that appears (SNIP on a dataset not
    shown before) takes the others' x range and its own home y range.
  - Only ⌂, back and forward change the ranges. A new transition, or switching
    between Flux and Normalised, starts at home, as now.
- **Check.** Made into tests first, in `test_dashboard_qt_regions.py`, on a
  synthetic project of three datasets:
  - zoom into the second spectrum with the zoom box. Then choose each region tool,
    draw a region, add a knot, and undo. Every spectrum's x and y ranges are
    unchanged, to 10⁻⁹;
  - ⌂ returns each spectrum to its home;
  - with one dataset, the same holds;
  - a new transition opens at home.

**10.4 — What the line IDs label (point 4, change; `lines.py`, `qt/regions.py`;
D59; Q10.5).**
- **Now.** The Regions tab labels each system's ions and strong lines (D59), with no
  choice.
- **The change (Q10.5 (b)).** A Line IDs menu in the Key pane, beside the
  line IDs' entries in the key, offers:
  - **This transition:** the transition shown, and its isotopes' lines (Task 10.2);
  - **This ion:** every line in view of the ion shown, and of its isotopes;
  - **The model's ions and strong lines:** as now (D59), and the default;
  - **Every line:** every line of every system, and of the forest, in view (D59's
    switch).

  Every choice counts the lines in view that are left unlabelled ("+ 12 lines not
  labelled"). The choice applies to every spectrum of the tab, and is kept with the
  project's view (`ui/view.json`). It is a view setting, not a step of the history.
- **Check.** Made into tests first.
  - In `test_dashboard_lines.py`, on `metal_line_abs`'s spectrum with O I 1302 shown
    (Si II 1304 beside it): This transition labels O I 1302 alone and counts Si II;
    This ion labels O I's lines in view; the default is as before, byte for byte;
    Every line counts nothing.
  - In `test_dashboard_qt_regions.py`: the menu changes the labels on every
    spectrum, and its choice is read back after the project is saved and opened.

**10.5 — The transitions in order of wavelength or strength (point 9, change;
`lines.py`, `qt/regions.py`; D36).**
- **Now.** The list is ranked by fλ, strongest first (D36).
- **The change.** An Order menu under the list: by wavelength, shortest first (the
  default, refining D36), or by strength (fλ, as now).
  - Near-coincident lines that share a row are placed by the row's first line.
  - The transition shown stays selected when the order changes, and ‹ › step
    through the list in the order shown.
  - The note under the list names the order ("The transitions on the data, by
    wavelength. ● has a snip.").
  - The choice is kept with the project's view.
- **Check.** Made into tests first.
  - In `test_dashboard_lines.py`: on a synthetic source that covers four Si II
    lines, the rows are in order of rest wavelength by default, and in fλ order by
    strength (as before).
  - In `test_dashboard_qt_regions.py`: the menu's two orders; the transition shown
    stays selected; ‹ › follow the order shown; the choice is read back after the
    project is saved and opened.
  - The tests that read the list in fλ order (J1358p6522's Lyα then Lyβ, for one)
    choose by strength, or read it by wavelength.

**10.6 — The system's snips in one list, and Load snips… moved (point 5, change;
`lines.py`, `qt/regions.py`, `actions.py`; D78; Q10.4).**
- **Now.** Each ion's snipped transitions are marked ● in its own list, so finding
  every snip of a system means stepping through its elements. Load snips… sits
  under the list (see Design).
- **The change.**
  - Under the transition list, where Load snips… was, "Snips of this system" is a
    drop-down. It lists every transition of the chosen system that has a snip, of
    any ion, in order of rest wavelength, each with the number of datasets it is cut
    from ("Al II 1670.8 · 2 datasets").
  - It shows the transition on screen when that has a snip, and "— choose a snip
    (9) —" otherwise. A system with no snip shows "No snips yet", greyed.
  - Choosing an entry shows that transition: the Element, the Ion and the list
    follow, as they do for the Data tab's coverage grid (`show_transition`).
  - It is filled afresh at every redraw. A SNIP therefore adds its transition, and
    a CLEAR of its last dataset takes it out, as do undo and the `.mod` panel.
    Choosing another System lists that system's snips.
  - Load snips… (Q10.4 (b)): the button goes, as point 5 asks. The action moves to
    the File menu, after Open a fit (.mod)…, and works as it does now (D78): greyed
    while the project has no source spectrum, each snip checked against the spectrum
    it was cut from, nothing loaded if one file fails, and the first snip loaded
    shown. From another tab, the Regions tab is brought forward to show it.
- **Check.** Made into tests first, in `test_dashboard_qt_regions.py`:
  - on a synthetic project of two systems, and on J1358p6522 (skipped where absent):
    the list holds each snipped transition of the chosen system once, in order,
    with its number of datasets. The other system lists its own;
  - choosing an entry sets the Element, the Ion and the transition, and the spectra
    show it;
  - SNIP adds an entry, CLEAR of its last snip removes it, and undo brings it back;
  - Load snips… is no longer on the tab, and is in the File menu after Open a fit
    (.mod)…. `test_load_snips_places_them_or_loads_nothing` runs from the menu, and
    the action is still greyed for J1358p6522, which has no source spectrum.

**10.7 — Adjust fit regions: one menu (point 3, change; `qt/regions.py`,
`actions.py`; D54).**
- **Now.** Two checkable buttons after "Fit regions:" (D54).
- **The change (refining D54).** "Adjust fit regions:" is followed by a menu of three
  entries:
  - **all spectra:** what Add regions to all does (a drag adds a region in every
    dataset of the transition, and a right-drag leaves pixels out);
  - **one spectrum:** what Tweak dataset regions does (the same in the selected
    dataset alone);
  - **off:** neither. Drags pan or zoom, as the toolbar sets.

  The menu is off by default, and goes to off whenever Add knot is chosen; choosing
  all spectra or one spectrum turns Add knot off. One spectrum is greyed while the
  selected dataset has no snip, as Tweak is now. The two actions stay, renamed to
  the entries, for the Regions menu and the shortcut sheet (they have no keys); each
  chooses its entry. Delete removes the region last clicked from every dataset,
  under all spectra or off, and from the selected dataset alone under one spectrum,
  as now; the right-click menu offers both, as now. Its Exclude this pixel, the
  tooltips, and the docstrings say "one spectrum" where they said Tweak.
- **Check.** Made into tests first, in `test_dashboard_qt_regions.py` (the tests of
  the two tools rewritten to the menu):
  - the label and the three entries are there, and the two buttons are gone;
  - all spectra adds a dragged region to every dataset; one spectrum adds it to the
    selected dataset alone; off pans;
  - Add knot sets the menu to off, and all spectra turns Add knot off;
  - Delete removes a region from every dataset under all spectra and off, and from
    the selected dataset alone under one spectrum;
  - "Tweak dataset regions" is in no text of the window.

**10.8 — A fit region's edge shown near the cursor (point 6, change;
`qt/plots.py`).**
- **Now.** The edge is 1 px wide, picked within 1.5 px, and faint (see Design).
- **The change.** While the menu of Task 10.7 is on all spectra or one spectrum:
  - a fit region's edge is not drawn until the cursor is within 6 px of it. It is
    then drawn 3 px wide, as the snip's edges are, in a darker shade of the fit
    regions' green, so that it is not taken for a snip's orange edge;
  - the cursor becomes ↔ over it, and the edge stays drawn while it is dragged;
  - it disappears once the cursor leaves.

  With the menu off, the edges are neither drawn nor picked, as now. The snip's
  edges are unchanged.
- **Check.** Made into tests first, in `test_dashboard_qt_plots.py`:
  - a movable region's edges have no visible pen;
  - the cursor 4 px from an edge draws it 3 px wide, with the ↔ cursor; 10 px away,
    it does not;
  - the pick reaches 6 px either side of the edge at the view's scale, and a drag
    begun 5 px from it moves it (`regionMoved`);
  - with the menu off, the edges cannot be picked.

**10.9 — More of the spectrum as the user zooms out (point 2, change;
`qt/plots.py`, `qt/regions.py`; Q10.7 (b)).**
- **Now.** The faint spectrum is cut once, close around the snip or the window, so
  zooming out shows nothing beyond it (see Design).
- **The change (Q10.7 (b)).**
  - The faint spectrum beside a snip is the dataset's whole source, in place of its
    cut of 0.6 of the snip's width either side. pyqtgraph draws only the part in
    view, reduced to the screen's resolution: every curve of the spectra already has
    its `clipToView`, and `autoDownsample` with the peak method, which keeps a narrow
    line (`plots._curve`). Before SNIP, the same faint source surrounds the
    ±300 km/s window, which is drawn as now.
  - Clipping and downsampling change what is drawn, never the data (RJC's response
    to Q10.7). The snip, its regions, its continuum, SNIP and the fit use every
    pixel, whatever the view. Zoomed in, every pixel in view is drawn, since
    downsampling starts only past five samples per screen pixel (pyqtgraph's
    default).
  - Each source's faint curve is made once and kept while the source is unchanged,
    since every action redraws the tab (Design, point 10).
  - pyqtgraph clips by bisecting the curve's x, and `plots.step_curve` breaks the
    steps at a gap with a NaN in x as well as in flux. A bisection that meets that
    NaN stops short, so the curve ends before the view's edge. A whole source has
    gaps, so the breaks keep x finite (the gap's edge) and put the NaN in flux
    alone; `connect="finite"` still breaks the line there. `step_curve` is shared,
    so every spectrum the dashboard draws in steps gains this, and its tests follow.
  - The view opens as now, on the snip or the window. Zooming out, whether with the
    wheel, the zoom buttons, the zoom box or a pan, shows the source as far as it
    goes, so an edge can be dragged out over data at any distance.
  - Nothing is cut or recomputed during a drag, and the cost of each frame depends
    on the plot's width in screen pixels, not on the spectrum's length.
  - Normalised shows no faint spectrum, as now, since the continuum covers the snip
    alone.
- **Check.**
  - In `test_dashboard_qt_regions.py`, on a synthetic snip of a long source:
    - the faint curve holds the whole source, with clipping and peak downsampling
      on (it holds 0.6 of the snip's width either side before the change);
    - zoomed out to ten times the snip's width, the curve drawn reaches both ends
      of the view, with no more than about ten points per pixel of the plot's
      width (five samples per pixel, two points each by peaks);
    - in `test_dashboard_qt_plots.py`, a source with gaps, viewed over a range
      whose bisection meets a gap, is drawn to both ends of the view (it stops at
      the gap before the change), and its x is finite throughout;
    - zoomed in to 50 pixels, every pixel in view is drawn, and the snip's file
      and regions are byte for byte as before the zoom;
    - an edge dragged five widths out extends the snip there;
    - a region drawn, and an undo, reuse the faint curve rather than making it
      again.
  - On J0035m0918 (skipped where absent; 110,712 pixels in UVES2010): the time to
    redraw the tab, and to draw one frame zoomed out to the whole spectrum, are
    measured before and after the change and logged.

**10.10 — Two handles on each of a snip's edges (point 1, change; `qt/plots.py`,
`qt/regions.py`, `snips.py`; Q10.6).**
- **Now.** Each edge has one handle, which moves its own snip's edge (see Design).
- **The change.**
  - Each orange edge has two grips. The middle grip (◀▶, as now) moves this
    dataset's snip, as does the line grabbed anywhere else along it, as now. A
    second grip near the top of the line, drawn differently, moves the same edge of
    every snip of the transition. The edge stays one line: where the drag begins
    decides which grip it is. The tooltips say which is which, and the key's "snip
    edges" entry names both. The second grip is shown only when the transition is
    cut from two or more datasets.
  - While the second grip is dragged, the other spectra's edges follow it. On
    release, one step moves every snip's edge (Q10.6 (a)): to the same
    velocity, moved by each dataset's shift as a copied region is (D55). A dataset
    whose source does not reach that far stops at its end, and the step says so.
    Each continuum's range follows its snip, as `edges_step` does now.
  - A snip whose file other data lines also read cannot have its edges moved
    (`edges_step` refuses it). It is left as it is, and named.
- **Check.** Made into tests first.
  - In `test_dashboard_snips.py`: the step that moves the low edge of every snip of
    a transition in three datasets with different shifts puts each new edge at the
    same velocity, to one pixel. It is one step, one undo restores every byte, and
    ALIS reads the model. A source that ends first stops there, and is named.
  - In `test_dashboard_qt_regions.py`: with three datasets each edge has two grips,
    and with one dataset one grip. Dragging the middle grip, or the line below the
    top grip, moves one snip (the existing edge tests stand). Dragging the second
    grip moves all three, and the other spectra's edges follow it during the drag.

**10.11 — T's value on a card (point 11, change; `qt/components.py`,
`components.py`; D21, D81, D93; Q10.10 (c)).**
- **Now.** A card's T row is only the mode button, and T's value is set only through
  Fixed at… (see Design).
- **The change.** The T row has a value field and a badge beside the mode button, as
  z and b_turb have.
  - Under T free and pure thermal, the field sets T's starting value. Under a fixed
    mode it sets the fixed value, as Fixed at… does. Each value typed is one step.
  - The limits hold (D81). A value below 1 K under pure thermal, or outside T's
    `lim param` or voigt's own limit, is refused with the reason (D86).
  - A T label that lines outside the component share (Q10.10 (c)):
    - under a fixed mode, the field is Fixed at…: the component alone (D93). Its
      lines take a new label first, and the others keep theirs, unchanged. A fixed
      shared label such as `ZEROT` is a shorthand for one value, so the forest keeps
      its 0 K. A label that a link uses is refused, as D93 refuses it;
    - under T free or pure thermal, the field sets the start value as b_turb's
      does, in every place of the label, so a tie the user made stays one value.
    - Either way, the step's description names the other components on the label.
  - A T computed by a link is shown read-only (D65). A hidden T is masked, and may
    be typed over (D24).
  - The mode menu and the drag's readout are unchanged.
- **Check.** Made into tests first, in `test_dashboard_qt_components.py` and
  `test_dashboard_components.py`:
  - on a synthetic component under T free, typing 12000 writes 12000.0 as T's start
    on the line of every ion, still free. It is one step, and one undo restores
    every byte;
  - the same under pure thermal, where 0.5 is refused and the limit named;
  - under 100 K fixed, typing 250 writes a fixed 250;
  - on J101155p294141_FIXDH's shape (`SHARED_T` in `test_dashboard_components.py`:
    a1, a2 and a forest line on `ZEROT`, fixed at 0 K), a2's T typed leaves a1's T
    and the forest line's `ZEROT` unchanged, byte for byte;
  - two components tied on one free T label: typing 9000 on either writes 9000 on
    both, the tie kept, and the description names the other;
  - a hidden T is masked in the field and its tooltip, and a T given by a link is
    read-only.

**10.12 — A transition's label rounds its wavelength down (point 12, bug;
`figscript.py`, `qt/figure.py`, `alis/plotscript.py`; Q10.8 (b)).**
- **The bug.** A panel's label rounds the rest wavelength to the nearest integer (C
  II λ1335). The convention, and RJC's scripts, round down (C II λ1334).
- **The fix.** The label's wavelength is the integer part of the rest wavelength,
  with a guard of 10⁻⁶ Å so that 1239.9999999 reads as 1240. This applies to the
  label the script writes and to the placeholder in the panel's options. Lyman lines
  keep their names, with no number. A label the user typed is left as typed.
  - Everywhere (Q10.8's response): `alis/plotscript.py` does the same for
    `run_alis`, in both its forms ("H I 1025", and "Lyβ 1025" for an ion named as a
    Lyman member). These and the Figure tab's are the only labels that round a
    wavelength to a whole number. The snips' specids already take its integer part,
    and the one-decimal names ("O I 1302.2") are not rounded to a whole number, so
    they stay (Design, point 12).
- **Check.** Made into tests first.
  - In `test_dashboard_figscript.py`: C II 1334.5323 gives C II λ1334, S II 1259.519
    gives λ1259, N III 989.799 gives λ989, Al II 1670.7886 gives λ1670, Si II
    1260.4221 gives λ1260, and 1000.0 gives λ1000. The drawn figure's label is read
    back from matplotlib.
  - In `test_plotscript.py`: the same rule in both of `run_alis`'s forms (H_I
    1025.7223 gives "H I 1025"); and the regression harness, its `test_bundle` mode
    included, since ALIS outside the dashboard changes.

**10.13 — The residuals and a label stay inside a panel whatever its flux range
(point 13, bug; `figscript.py`; D106, D110; Q10.9 (b)).**
- **The bug.** The residual strip and a label in a lower corner are placed at fixed
  fluxes, so a panel's own flux range can leave them out of view (see Design).
- **The fix (refining D110).** Both are placed by fractions of the panel's height,
  not by flux.
  - The fractions are chosen so that a panel at its default flux range draws them
    exactly where they are now: the strip at −0.3 (or −0.2 inside the panel), 0.05
    per σ, and the label at 0.07, above the zero level (D110).
  - A panel with its own flux range keeps them at the same places in the panel, and
    the same size, so they stay in view whatever the range.
  - The line ticks follow the same rule (Q10.9 (b)): at the default range they stay
    at 1.03 to 1.15, and a panel with its own range keeps them at the same place in
    the panel, so they are always visible.
  - The comments on the script's `RESIDUAL_OFFSET`, `RESIDUAL_SCALE`, `CORNERS` and
    `TICK_HEIGHT` say that they are fractions of the panel. The upper corners
    already are.
- **Check.** Made into tests first, in `test_dashboard_figscript.py`, read back from
  the drawn figure:
  - RJC's case first: with the flux range (0.5, 1.2), the strip and the label lie
    outside the axes before the change, and inside after it, for each of the three
    residual choices;
  - at the default range, every residual point and the label are where they were
    before the change, to 10⁻⁹ in flux, for each of the three residual choices;
  - the ticks (Q10.9 (b)): with the flux range (−0.1, 1.1), they lie partly
    outside the axes before the change and inside after it, and at the default
    range they are at 1.03 to 1.15, to 10⁻⁹, for each of the three residual
    choices.

**10.14 — The DH preset starts at a chosen member (item 1 carried over, change;
`figpresets.py`, `figure.py`, `qt/figure.py`).**
- **Now.** The DH preset always starts at Lyα. A figure of the higher members, as
  `DH_Ly8-Ly14*.py` draw them, needs its cells filled by dragging.
- **The change.** When DH is the chosen preset, a From menu beside it lists the
  system's Lyman members that have a snip, with Lyα first and the default. Apply
  fills the grid from that member up the series, in two columns for one dataset or
  one column per dataset, as now. The figure keeps the member with its preset, so
  the menu shows it again. A figure saved before reads as Lyα.
- **Check.** Made into tests first.
  - In `test_dashboard_figpresets.py`, with its synthetic DH project: From Ly8 gives
    Ly8, Ly9 and so on, placed as `DH_Ly8-Ly14*.py` place them, with no lower member.
    Two datasets give one column each, from Ly8. A member with no snip is not
    offered. From Lyα gives today's layout unchanged.
  - In `test_dashboard_qt_figure.py`: the menu appears with DH alone, lists the
    members, and Apply uses the one chosen, as one step that undo takes back.

**10.15 — Room below a middle row's own x title (item 2 carried over, change;
`figscript.py`, `figure.py`, `qt/figure.py`; D111).**
- **Now.** The margins are worked out from the panels (D111), but the spacing between
  rows is not. A panel's own x title in a middle row (a wavelength panel among
  velocity ones, as `DH_Lyseries.py` has) can meet the panel below.
- **The change (refining D111).** The spacing between rows is worked out from the
  panels, as the margins are, unless it is set. A row whose panels have their own x
  title, or x numbers, gets the room they take before the next row. Layout's Spacing
  gains Automatic, as Margins has: the values worked out are shown, and unticking
  Automatic keeps them as the figure's own. The script holds them in `SPACING`, as
  now. A figure saved with the old default spacing reads as Automatic, and one with
  any other spacing keeps it as set. The spacing between columns is as now.
- **Check.** Made into tests first.
  - In `test_dashboard_figscript.py`: in a figure of two rows whose upper row has a
    wavelength panel with its own x title above velocity panels, the title's box
    (read back from matplotlib) overlaps the lower row's axes before the change, and
    not after it. A figure with no own x title keeps today's spacing, and a spacing
    set by hand is the figure's.
  - In `test_dashboard_figure.py`: the option, its refusal outside 0–2, and figures
    saved before.
  - In `test_dashboard_qt_figure.py`: the Automatic control and the values it shows.

**10.16 — The paths Stage 9 left untested (item 3 carried over, tests;
`qt/figure.py`, `figure.py`).**
- **The tests,** for:
  - a drag begun with the mouse on a cell. `QDrag.exec` blocks off-screen, so the
    test replaces it while it runs, and checks what a press and a move past Qt's
    start distance make: a drag that belongs to the tab and carries the cell's data.
    A shorter move, or a press on an empty cell, makes none;
  - the refusals of a figure size and a spacing out of range: in `figure.py` (each
    bound, with its message) and in the Layout pane (the message is shown, the
    figure is unchanged, and no step is made).
- A fault that a test finds is fixed in the same task, and logged.
- **Check.** `test-coverage` on `figure.py` and `qt/figure.py`: those lines are
  covered, and the figures go in the log (94% and 92% at Stage 9's close).

**10.17 — Close the stage.**
- **The review.** Publish the screenshots of what changed for RJC's review, on a new
  page with the earlier pages' stylesheet (built by `doc/dashboard/stage10/`), and
  apply the review. The page shows:
  - SNIP on J0035m0918's Al II 1670, with its message and UVES2010's spectrum (10.1);
  - D I labelled beside Lyβ, and each choice of the line IDs (10.2, 10.4);
  - the list in order of wavelength, and the system's snips (10.5, 10.6);
  - the Adjust fit regions menu, a region's edge under the cursor, and a snip's two
    grips (10.7, 10.8, 10.10);
  - a view zoomed out over the source around a snip (10.9);
  - a card's T field in each mode (10.11);
  - a figure with labels rounded down, a panel with its own flux range (its
    residuals, label and ticks), the DH preset from Ly8, and the room under a
    middle row's title (10.12–10.15).

  Task 10.3 cannot be seen in a still image, so the page names its test.
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md` (with ALIS's
  change to `plotscript.py`, Q10.8 (b)), `tests/README.md` and the
  `gui-component` skill up to date. Record the decisions in
  `ALIS_v2_dashboard_prompts.md` (D113 onwards, from Q10.4–Q10.10 and the review),
  saying which earlier decisions they refine: D21, D36, D53, D54, D59, D78, D93,
  D106, D110 and D111. Update `dashboard_stage0.md`: Stage 10 done, and the "After
  v1" list.
- **What follows (Q10.3 (a)).** Write `claude_prompts/dashboard_stage11.md` for the
  feedback that arrived after this list closed, as Stage 9's close wrote this
  document, unless RJC names another stage to come first.
- **Check.**
  - Every point of the list, and every item carried over, has its task logged as
    done with its check passing, or its query answered.
  - The `unit`, `gui` and `fast` batches pass. Their counts go in the log, against
    Stage 9's close: 2176 unit tests passed with 120 skipped, 256 `gui` tests, and
    149 `fast` tests (all under pytest 9.2.0).
  - CI passes on both platforms, the `gui` job within its 90 minutes (Stage 9 raised
    the limit; Ubuntu's job was near 56 minutes by estimate).
  - The tests that need RJC's files pass here, and are skipped on CI. This stage's
    need J0035m0918's sources (10.1, 10.2, 10.9) and J1358p6522 (10.5, 10.6).
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44).
  - Task 10.12 changes ALIS outside `alis/dashboard/` (Q10.8 (b)), so the
    regression harness runs again at the close, with its `test_bundle` mode.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs and the Figure
  tab (its "Stage 9 additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D112.
- `claude_prompts/dashboard_stage8.md` and its log: how users' feedback was handled
  last time, from the list to the close.
- `claude_prompts/dashboard_stage9.md` and its log: the Figure tab, and RJC's feedback
  on it (Prompts 4 and 5).
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 9's builder is `doc/dashboard/stage9/`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude with this document (2026-11-08). Each gives Claude's lean.*

**Q10.1 — The items carried over from Stage 9.** Three ways:
- **(a) All three join this stage**, after the feedback's tasks.
- **(b) Only the tests (3)** join. The two changes wait for a user to ask for them,
  in `deferred_work.md` §7.
- **(c) None join.** All three go to `deferred_work.md` §7.

My lean: (a). They are small, and items 1 and 2 are what a user drawing D/H figures
meets first.

**Response:** I agree with (a).

**Q10.2 — Your feedback on the Figure tab.** You said that you would try Q9.12's
options in the scripts and send what to add, change or remove. Two ways:
- **(a) In this list**, as points marked "(Figure tab)", with the users' points.
- **(b) In a prompt of their own**, before the list is read.

My lean: (a). One list gives one design, one order and one review.

**Response:** I have included several items in this design document. Let's limit the changes to this list for now. If I receive further feedback, I can add these specifically in a future development stage. 

**Q10.3 — When the stage closes, and what follows it.** Stage 8's list was closed
before its design was written (Q8.4 (a)), and the Plot tab came next (Q8.5 (a)).
Two ways for Stage 10:
- **(a) The same.** The list closes when you judge it long enough. Feedback that
  arrives later goes to Stage 11's document, written at this stage's close, unless
  you name another stage to come first: one of the items marked "Later" (re-attaching
  to a run, command search, the `link` editor, reproducing an imported fit, Orders
  mode), the documentation (`documentation_stage0.md`), or the deferred work.
- **(b) An open stage.** Stage 10 builds points in batches as they arrive, with a
  review for each batch.

My lean: (a). A closed list gives one order, one review and a close, as Stage 8 had.

**Response:** I agree with option (a).

*Q10.4–Q10.8 were raised on 2026-11-08 (Prompt 1), from the Feedback list (Design,
"What was found"). Each gives Claude's lean, and the tasks are written with it. RJC
answered them the same day (Prompt 2), taking every lean.*

**Q10.4 — Load snips… (point 5, Task 10.6).** Point 5 asks for the button to go,
since a `.mod` brings its own snips (D78). The button is also the only way to load
snips that have no model yet: `prepfit`'s, with the fit regions chosen there, or an
Export's. Two ways:
- **(a) Removed.** The button, its action and `loadfit.load_snips_plan` go, with
  their tests. Snips come with a `.mod` (Open a fit (.mod)…, Load a .mod…), or are
  cut with SNIP and their regions drawn again.
- **(b) Moved.** The button goes, as the point asks. Load snips… moves to the File
  menu, after Open a fit (.mod)…, and works as now.

My lean: (b). The tab gains the room for the list of snips either way, and a user
with `prepfit`'s snips and no model keeps the regions they chose there.

**Response:** I agree with option (b). The button will be removed and the functionality will be included in the File menu, and the functionality will remain as it is currently.

**Q10.5 — What the line IDs label (points 4 and 7, Tasks 10.2 and 10.4).** Point 4
names three choices: the transition shown, the lines of its ion, and every line.
Today's labels (D59: each system's ions and strong lines, the rest counted) are none
of the three. Three ways:
- **(a) The point's three choices.** D59's mix goes. Every line is the default.
- **(b) The three, and D59's as a fourth,** which stays the default.
- **(c) As (b), but with This ion as the default.**

In each, the lines left unlabelled are counted, the isotopes of the ion shown are
labelled (point 7), and the choice is kept with the project's view.

My lean: (b). D59's mix labels the other systems' lines, the interlopers and blends
(S5), which the transition or ion alone would hide. The default view then stays as
users know it, and the menu thins it when it is too busy.

**Response:** I agree with option (b). There will be four choices, with the default being the current labeling scheme.

**Q10.6 — Where the second handle takes the other snips' edges (point 1, Task
10.10).** Two ways:
- **(a) To the same place.** Each snip's edge goes to the velocity of the edge
  dragged, moved by its dataset's shift as a copied region is (D55). The edges then
  line up across the stacked spectra, as SNIP cuts them.
- **(b) By the same amount.** Each snip's edge moves by the drag's change in
  velocity, so a snip whose edges were set apart stays apart.

My lean: (a). It is what the stacked spectra show while the handle is dragged, and a
snip that needs its own edge keeps the middle grip.

**Response:** I agree with option (a).

**Q10.7 — More of the spectrum when zoomed out (point 2, Task 10.9).** Point 2 asks
for more of the spectrum as the user zooms out, without recutting it during a drag,
and without drawing every pixel of the source from the start. It invites another
suggestion. Two ways:
- **(a) Recut once the view settles.** The faint spectrum is cut afresh for the
  range shown, plus a margin, once the view stops changing: when a pan or zoom-box
  drag is released, 250 ms after the last wheel step or zoom button, and after ⌂,
  back and forward. Each of these needs its own signal, and the linked spectra must
  be recut together. A view too wide to draw quickly is thinned.
- **(b) The whole source, drawn for the view.** The faint spectrum is the dataset's
  whole source, and pyqtgraph draws only the part in view, reduced to the screen's
  resolution (`clipToView`, and `autoDownsample` by peaks, which keeps narrow
  lines). Nothing is ever recut. A frame costs about the same whatever the
  spectrum's length, since at most about two points per screen pixel are drawn. The
  view opens as now.

My lean: (b). It meets both of the point's limits: what is drawn depends on the
screen, not on the source, and the data never run out wherever the user zooms or
drags an edge. It is also a small change to one curve, where (a) needs every way of
changing the view handled. The task measures both redraw times on J0035m0918.

**Response:** I agree with option (b). I didn't realise that pyqtgraph could handle the whole source and only draw what is in view. This will be a more efficient solution, and reduces the risk of errors (e.g. if the user zooms in close, we don't want any pixels physically cut from the SNIP, because there might be pixels that the user intends to fit that is not shown in a zoomed in view).

*Applied (Prompt 2): Task 10.9 says that clipping and downsampling change what is
drawn, never the data, and checks it zoomed in. Two corrections to (b) as written
above, found in pyqtgraph 0.14's code: it draws up to about ten points per screen
pixel (five samples, two points each by peaks), not two, so the cost still depends
on the screen and not on the source; and every curve of the spectra already has
both settings, so the change is the data the faint curve is given. Task 10.9 also
keeps x finite at the steps' gaps, since a NaN there stops the clipping short.*

**Q10.8 — Rounding down in ALIS's own plotting script (point 12, Task 10.12).**
`alis/plotscript.py`, which `run_alis` uses, rounds its labels to the nearest
integer as the Figure tab does ("H I 1026" for Lyβ). Two ways:
- **(a) The Figure tab only.** ALIS outside the dashboard is unchanged.
- **(b) Both.** `plotscript.py` rounds down too. ALIS outside the dashboard changes,
  so the regression harness runs, and `CHANGELOG.md` says so.

My lean: (b). One convention for both scripts, and the change is one line with its
test.

**Response:** I agree, the change should be everywhere.

*Applied (Prompt 2): "everywhere" is the Figure tab's labels and placeholder, and
both of `plotscript.py`'s forms. A search of ALIS found no other label that rounds a
wavelength to a whole number. The snips' specids already take the integer part, and
the one-decimal names of the lists and line IDs ("O I 1302.2") stay (Task 10.12).*

*Q10.9 and Q10.10 were raised on 2026-11-08 (Prompt 2), when Tasks 10.13 and 10.11
were checked against the code. Each task is written with the lean. RJC answered them
the same day (Prompt 3), taking each lean.*

**Q10.9 — The line ticks in a panel with its own flux range (point 13, Task
10.13).** Point 13 names the residuals and the label. The ticks above the spectrum
are placed in flux as well, from 1.03 to 1.15, so a panel whose range ends below
1.15 cuts them, and one that ends below 1.03 hides them. A panel of weak lines at
(0.8, 1.1), say, loses their tops. Two ways:
- **(a) As now, in flux.** The ticks stay just above the continuum, as the reference
  scripts place them. A user who sets a range widens it to show them. The stage
  keeps to the list (Q10.2's response).
- **(b) By fractions of the panel,** as Task 10.13 places the residuals and the
  label: the same at the default range, and in the same place in a panel with its
  own range. A panel zoomed into a line's core, at (−0.05, 0.4), then has its ticks
  over the spectrum rather than out of view.

My lean: (b). It is the same fault as point 13, fixed by the same line of the same
task, and panels of a figure with different ranges then have their ticks at one
height.

**Response:** I agree, option (b) is best. The ticks should be placed by fractions of the panel, so that they are always visible regardless of the flux range.

**Q10.10 — A T value typed on a label other components share (point 11, Task
10.11).** A value typed on a card is written in every place of its parameter's label
(`edit.set_value`), so a tie keeps one value: z, b_turb and log N work so. A
temperature mode, Fixed at… included, changes its own component alone and splits a
shared label first (D93, Q8.10 (a)). The T field meets both rules, and RJC's
J101155p294141_FIXDH has a1, a2 and three lines of the forest on `ZEROT`, fixed at
0 K. Three ways:
- **(a) As the other fields:** every place of the label. 250 typed on a2's T changes
  a1's T and the forest lines' to 250 as well, and the description names them.
- **(b) As the temperature modes (D93):** always the component alone. The forest
  keeps its 0 K, but typing a starting value on a T tied to another component's
  breaks the tie; one Tie to… restores it.
- **(c) By the mode.** Under a fixed mode, the field is Fixed at… (D93): the
  component alone, so `ZEROT`'s other lines keep their 0 K. Under T free or pure
  thermal, it sets the starting value in every place of the label, as b_turb's
  does, so a tie stays one value.

My lean: (c). A fixed shared label is a shorthand for one value, which D93 already
splits. A free one is a tie, and typing a starting value should not undo it. Task
10.11 was first written with (b), which breaks a tie.

**Response:** I agree with option (c).

## Prompts

1. Please read the list of changes and bug fixes, and the work of Stages 0–9 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Feedback`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all of the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (10.1, 10.2, etc.). The final Task (currently labelled 10.N) should be to close the stage, and should include the checks that will verify that the stage is complete.
2. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (10.1, 10.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
3. I have responded to your next set of queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (10.1, 10.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.