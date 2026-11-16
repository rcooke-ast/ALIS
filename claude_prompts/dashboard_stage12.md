# Prompt file for ALIS software dashboard creation -- STAGE 12

> **Feedback from users, after Stage 11's list closed.** Stage 11 took a closed list
> of users' feedback (a fit of several spectra, and the GPU, which it deferred) and
> the items Stage 10 carried over (Q11.2 (a)). Stage 12 takes the feedback that
> arrives after that list closed. RJC collects each point in the Feedback section
> below. Claude turns the list into a design, queries and tasks, each with its
> check, and builds them in order, with a review before the stage closes.
>
> The rules of Stages 3–11 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> project is one step of the history. ALIS outside `alis/dashboard/` changes only
> where a query agrees. A question that comes up while a task is built pauses the
> work: it is asked in the Queries section, and its answer is logged there (Stage 11,
> Prompt 2).
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D128 at the close of Stage 11).
> "Q0.*n*" to "Q11.*n*" refer to `dashboard_stage0.md` to `dashboard_stage11.md`.
> What follows v1 is listed in `dashboard_stage0.md` ("After v1").

*Written by Claude on 2026-11-13 (Stage 11's Task 11.11). No code was changed. The
Feedback section is RJC's to fill in; the design and the tasks are written once it
is. The items carried over are what Stage 11 found and did not do. Prompt 1
(2026-11-16) wrote the design and Tasks 12.1–12.12 from Feedback points 1–8 and the
item carried over that joins (Q12.1: item 2), with Q12.3–Q12.8; items 1 and 3 wait
in `deferred_work.md` §7.8 and §7.9. No code was changed. Prompt 2 (2026-11-16)
applied RJC's responses to Q12.3–Q12.8 to Tasks 12.4, 12.6 and 12.9, and checked the
list; no further query was needed, and no code was changed. Prompt 3 (2026-11-16)
added the delta function and lsfspline's smoothing as a FWHM to Tasks 12.9 and
12.10, with Q12.9 and Q12.10; no code was changed. Prompt 4 (2026-11-16) applied
the responses to Q12.9 and Q12.10 to Task 12.9 (spix from `nsubmin`, its default 10;
`examples/lsf_spline` a regression case), with Q12.11 on two consequences. Prompt
4, run again (2026-11-16), applied the response to Q12.11 to Tasks 12.9 and 12.10;
no further query was needed, and no code was changed.*

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

1. **(bug)** In some cases, the wavelengths of two datasets are identical, and the dashboard things that the wavelength regions overlap, but they are actually different datasets. For example, the context file `context/feedback01/MWC137/mwc137.model` provides an example where two datasets (specid=6 and specid=7) have overlapping data, but they are actually different datasets. The dashboard should be able to distinguish between these two datasets and not treat them as overlapping. This can be reproduced by loading the context file and checking the datasets with specid=6 and specid=7.

2. **(bug)** In the Components tab, it is not possible to click the excluded component a2 to make it included. Instead, it informs the user that they should add it to an ion first. I believe there should be the functionality to simply click the ion in the Excluded list to add this component to an ion. Alternatively, there needs to be a more helpful message to the user about how they can add this component to an ion. This can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model` and going to the Components tab, then trying to click on an ion in the excluded component a2.

3. **(change)** On the Components tab, if I scroll down to the bottom of the components list, and include an ion that is currently excluded, the page refreshes and the scroll changes so that the components list navigates to the top of the components list. This means that the user needs to scroll down to the bottom of the components list again to add another component. This is not ideal, and can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model`, going to the components tab, scrolling down to the components near the bottom of the list, including an ion into a component, and watch the menu scroll to the top of the components list. Overall, it would be good if none of the windows/scrollbars/menus/zooms reset to the default values. If the user has scrolled down, or zoomed in, or changed the window size, it would be good if those values were remembered and not reset to the default values. We must avoid changing things to the default value unless it is absolutely necessary. Furthermore, there is always an option for the user to go to the "Home" view by clicking on the "Home" button that appears on some plots. Otherwise, the user should have complete control over the zoom and scroll values of all windows.

4. **(new)** It would be good if there were a way to zoom into all spectra in both directions. Currently, a scroll only allows the wavelength to zoom in about the mouse cursor. Maybe we should make it so that the user can hold down a key and while the key is down (e.g. 'y') then scrolling would zoom in the y direction about the mouse cursor. Similarly, it would be good if there was a button that allows the user to reset the default x separately from the default y. This could be something like the Home button that appears on all plotting windows, but instead have an x and y button that resets the x and y axes separately.

5. **(change)** In the Component Tab, dragging below the continuum for K I 7701 (in the `emsmwcs137` dataset) for component a7 does not work. The b-value is allowed to change, but the column density does not change. This can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model`, going to the Components tab, and dragging below the continuum for K I 7701 for component a7. This is probably because the column density is linked to another variable. Is there a way that we can allow the column density to change so that the linked variable can be updated, or, a way that we can change the linked variable so that the a7 column density for K I could be modified? Alternatively, if the column density is linked to a another variable and we cannot update this easily, then perhaps when the user attempts to drag the column density, the text that would have displayed the new column density value could instead make it clear that the column density is linked. This would make it clear to the user why they cannot change the column density.

6. **(bug)** When loading the context file `context/feedback01/MWC137/mwc137.model` the user is presented with a message: "Some source spectra of this project are missing or have changed. Relink them now?" This suggests that the files are not included in the `.model` file. If a user wishes to share their `.model` file with another user, they would also need to share the source spectra files. It would be good if the `.model` file could include the source spectra files, so that a user can share their `.model` file with another user without needing to share the source spectra files separately. This can be reproduced by loading the context file `context/feedback01/MWC137/mwc137.model` and observing the message about missing or changed source spectra. I have also provided another context file `context/feedback01/MWC137/mwc137_relinked.model` where I have relinked the source spectra to the files that are on this machine, so the message does not appear.

7. On the Regions Tab, if the user selects `lsfspline` as their line spread function, we should allow a button to appear beside the list box. This button should have a label `Set LSFSpline knots...` that when clicked, opens a separate dialog to build the starting parameters of the lsfspline with draggable/insertable knots. The key starting values are a smoothing parameter, and three knots. The first knot should be fixed at a value of 0.0 and be located at the left edge (-10 km/s). The second knot should be fixed at a value of 1.0 at 0.0 km/s. Finally, the third knot should be located at the right edge (+10 km/s) with a fixed value of 0.0. There should be an option to add knots, delete knots, or move/drag knots around in both x and y directions. The y-values must be between 0 and 1. Knots cannot be allowed to have the same x value, but they can have the same yvalue. The x-values can be anything between the leftmost knot and the rightmost know. The edge knots should be moveable in the x direction only. The plot should clearly display the knots, together with a faint/thin black line that connects the knots (ordered from the leftmost knot to the right most knot). There should also be displayed the smoothed version of the linearly interpolated spline (a normal weight red line) that looks like the actual LSF kernel used by ALIS to convolve the data. Finally, it would be useful if the FWHM of the LSF was indicated on the plot, together with the value, so that the user has a rough idea of the full width at half maximum of the smoothed curve used for the LSF.

8. **(bug)** On the Components Tab, Rb I appears twice in the included/excluded list of ions. Is this because 87RbI is commented out? For further context, please see the following file that will show an example of this behaviour: `context/feedback01/MWC137/mwc137.model`

## Carried over from Stage 11

*Found by Claude during Stage 11 and not done there. RJC decides whether each joins
this stage (Q12.1).*

1. **(change)** Load snips… stops the whole load when a file cannot be read or its
   name names no transition (D122, Q7.4): such a file has no transition to be placed
   at. The dialog that assigns snips could take it instead, on the right, if the
   transition were guessed from the system's ions or chosen by the user. Asked on
   Stage 11's review page; no answer was given before the stage closed.
2. **(bug)** An unassigned dataset whose last snip is cleared (CLEAR) stays on the
   Data tab with no snips and no source ("0 snips · unassigned: no source
   spectrum"), since Load snips… needs such a row to exist before its first snip is
   added. Remove dataset takes it away. It could go with its last snip.
3. **(change)** Open a fit's question is RJC's wording for every fit (D123), with
   "spectra"; a fit whose snips all share one dataset was asked in the singular
   before. Kept unless RJC wants the singular back for those (asked at the close,
   2026-11-12; the stage closed on RJC's word).

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D128 and the stage documents. A point that asks for what a decision ruled out
  becomes a query that quotes the decision, not a task.
- **A bug becomes a test first.** It is reproduced from the user's project where there
  is one, then made into a synthetic test (`tests/dashboard_synthetic.py`,
  `tests/dashboard_fits.py`) that fails before the fix and passes after it. The
  user's files are checked with scratch scripts, logged (D99).
- **Outside the dashboard.** A point about ALIS itself becomes a query: the
  dashboard's stages change ALIS only where a query agrees. No point of the Feedback
  list needs ALIS to change; Prompt 3 asks for lsfspline's smoothing as a FWHM, and
  Q12.9's response for `nsubmin`'s default of 10 (Task 12.9).
- **Grouping and order.** Points are grouped by the part of the window they touch,
  and a bug comes before the work that builds on it.
- **Shown before it closes.** Every change a user sees is shown on a review page,
  built by `doc/dashboard/stage12/` with the earlier pages' stylesheet, and RJC's
  comments on it are applied before the stage closes.

### 12.1–12.12 — The Feedback list and the item carried over

*Written by Claude on 2026-11-16 (Prompt 1), from Feedback points 1–8, the item
carried over that joins (Q12.1: item 2, with RJC's response), and the code at the
start of the stage (292dba9). Each point's design is in its task below.*

**What was found.** Each point was reproduced off-screen on RJC's
`mwc137_relinked.model` (a copy in the session's scratchpad), with scratch scripts
and scratch `gui` tests, before its task was written. No code was changed.
- **Point 1** (specids 6 and 7 taken as overlapping).
  - The two snips are `emsmwcs137_39K_I_7701.1_reg1.dat` (specid 6, in the dataset
    "39KI7701 (unassigned)") and `…_reg2.dat` (specid 7, in emsmwcs137). They share
    one wavelength grid, and their flux and error agree digit for digit at 297 of
    reg1's 336 pixels: every pixel outside the line's core (7701.28–7702.10 Å), where
    they differ.
  - By the test of D19 and Q1.8 (the same wavelength and flux/error ratio, in runs
    of three or more), the 147 pixels fitted in both outside the core are the same
    pixels of the same data. The dashboard says so in four places: the hatching and
    the offer to merge on the Regions tab, the tab's marker, a note of the pre-flight
    check, and ALIS's own warning when it loads the data (line 32 of the `.mod`
    panel).
  - Two independent exposures on one grid agree nowhere, which is how Q1.8's test
    tells them apart. These agree almost everywhere, so no test of the values can
    tell them from one snip cut twice: only the user knows (Q12.3).
- **Point 2** (a2 cannot be included). a2 is excluded from every ion: its Li I, Na I,
  K I and Rb I lines are commented out, and its greyed card lists every ion of the
  system under Excluded ions (D79). A click on one of those four brings its line
  back. A click on Ti II, Ca I, Ca II or Fe I, which a2 never had, is refused:
  "Component a2 is excluded from every ion: include it in one first." A new ion's
  line copies z, b and T from the component's live line (`components.include`), and
  a2 has none.
- **Point 3** (the list scrolls to the top).
  - Reproduced with a real click: a click on a card's Excluded ions link near the
    bottom of the list moves the list to its top (scroll 1496 → 0). The click
    selects the card's component, and the selected card is scrolled into view as
    the cards are rebuilt, before they are laid out, while every card is still at
    the top. The same include made without the click keeps the list's place.
  - An audit of every tab: each scroll bar and view was moved, then one ordinary step
    was made, on that tab or on another. The Data tab kept everything. The Regions
    tab kept its spectra's ranges (D114), but its list of spectra, its transition
    list (sideways) and the Snip box's table went back to their start. On the
    Components tab, every panel went back to ⌂ (±200 km/s) after every step, a drag
    included. Fit · Inspect's view and progress chart went back to ⌂. The drop-downs
    kept their choices on every tab.
  - The window opens at 1440 × 900 each time; its size is not kept between sessions.
- **Point 4** (zooming in y). The wheel zooms the wavelength (or velocity) axis only,
  and so do + and − (D38); ⌂ returns both axes. Qt reports only modifier keys with a
  wheel event, so a held letter needs an event filter, as the Components tab's N and
  B have (`DragKeys`): pressed while the pointer is over a panel, the key is the
  panel's, and nothing else receives it (Q12.5).
- **Point 5** (a7's K I column density). K I's log N in a7 is `ka`, computed by the
  link `ka(lia,lika) = lia - lika`: Li I's log N in a7 less `lika`, a free `variable`
  that only this link uses. Links are never dragged (D65), so the drag moves b alone,
  and the readout names b alone without saying why (Q12.6).
- **Point 6** (the source spectra).
  - Embedding is built in the project file: `alis/bundle.py` keeps a copy of a
    spectrum (`embed_source`, D8), and the dashboard reads an embedded copy and never
    asks to relink it (`sources.EMBEDDED`, Stage 3). The window has no way to make
    one, so every project references its spectra by path and checksum alone.
  - RJC's two spectra would add about 15 MB to the 1.7 MB bundle (`emsmwcs137.dat`,
    32 MB of text, compresses to 12.8 MB; `EM_MWC__137.dat` to 2.5 MB), and about a
    second to each Save and to each minute's recovery copy (D48) (Q12.7).
- **Point 7** (lsfspline's knots).
  - The Snip box's resolution menu greys `lsfspline`, and so does the Data tab's: a
    function of several parameters is set in the `.mod` panel (D89).
  - ALIS's `lsfspline` (`alis/functions/lsfspline.py`): its first value is a
    smoothing in km/s (at least 0), and each other value the kernel's height at a
    knot, held within [0, 1] by ALIS; the knots' velocities are the keyword
    `locations`. The kernel is the knots joined by straight lines, smoothed by a
    Gaussian of σ = smoothing/√2, and normalised. (Prompt 3 (ii): Task 12.9 makes
    the smoothing the Gaussian's FWHM.)
  - RJC's `alis2.mod` uses nine knots from −10 to +10 km/s, the edges fixed at 0 and
    the centre at 1 (`LFIX`, `CFIX`, `RFIX`), the smoothing and the other heights
    free (Q12.8).
- **Point 8** (Rb I twice).
  - 87Rb I's lines being commented out is not the cause. The dashboard's table of
    each element's main isotope (`project.MAIN_ISOTOPE`) stops at zinc, so
    `ion_name` names both rubidium isotopes "Rb I": a1's Excluded ions read "Li I,
    Rb I, Rb I".
  - The same table groups an isotope under its element (D37) and checks that isotopes
    share z, b and T (D22), so neither applies to rubidium either (from the code;
    MWC137's 87Rb I has no live line). Rubidium is the only element with two
    isotopes in ALIS's atomic data that the table lacks.
- **Item 2** (Q12.1's response). CLEAR of an unassigned dataset's last snip leaves the
  dataset ("0 snips · unassigned: no source spectrum"), as RJC's response wants.
  Nothing tells the user that it is empty, and the Data tab's marker shows ✓. A snip
  cannot be cut from it, since it has no source spectrum; Load snips… gives it
  snips.

**What stays as it is.**
- The test for pixels fitted twice (D19, Q1.8), and ALIS's warning when `run_alis`
  loads the data.
- Links are written in the `.mod` panel (D65), whatever Q12.6 decides for the drag.
- A project references its spectra by path and checksum, and Relink… is as now
  (F13), whatever Q12.7 decides.
- The other functions of several parameters (`voigtconv`, `multivfwhm`) are set in
  the `.mod` panel (D89).
- The Figure tab's preview, drawn by its script (D105), has no zoom bar.
- ALIS outside `alis/dashboard/` does not change, except `lsfspline`'s smoothing
  (Prompt 3 (ii)) and `nsubmin`'s default (Q12.9's response), in Task 12.9.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage12_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests,
> also under CI's pytest (memory `ci-test-versions`); run the `fast` batch before the
> stage closes. A task that changes ALIS outside `alis/dashboard/` also runs the
> regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (12.1, 12.2, …), and per item carried over
from Stage 11 that joins the stage, written by Claude once the list is filled in.
Each has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*The last task is the review (a page built by `doc/dashboard/stage12/`), and the one
after it closes the stage, as Stage 11's Tasks 11.10 and 11.11 did.*

*Tasks 12.1–12.12 were written by Claude on 2026-11-16 (Prompt 1), with the leans
of Q12.3–Q12.8. They were revised at Prompt 2 (2026-11-16) with RJC's responses:
12.4 (Q12.3: Ignore warning, the test unchanged), 12.6 (Q12.4 (b): each view's
ranges and each list's place kept in `ui/view.json`, with no lag) and 12.9 (Q12.8
(iii): the smoothing starts at the resolution before, or 5 km/s); 12.3, 12.7, 12.8
and 12.10 are as the leans. At Prompt 3 (2026-11-16), 12.9 took lsfspline's
smoothing as a FWHM in ALIS, and 12.9 and 12.10 the delta function. At Prompt 4,
12.9 took RJC's responses to Q12.9 and Q12.10: spix from `nsubmin`, ALIS's default
`nsubmin` 10, and `examples/lsf_spline` a regression case. At the second Prompt 4,
12.9 and 12.10 took the response to Q12.11: no two knots closer than spix, the
knots never changed with the sub-pixels, and the default of 10 kept. The order puts
a bug before the work that builds on it, and keeps work on one part of the window
together:*
- *the Components tab: point 8's names, point 2's include, point 5's drag
  (12.1–12.3);*
- *the Regions and Data tabs: point 1's pixels, and item 2's emptied dataset (12.4,
  12.5);*
- *every view: point 3's places, then point 4's zoom, which builds on them (12.6,
  12.7);*
- *the project file: point 6's spectra (12.8);*
- *point 7, lsfspline's knots: the logic, then the dialog (12.9, 12.10);*
- *the review (12.11), and the close (12.12).*

**12.1 — Each isotope its own name (point 8, bug; `project.py`; D22, D37).**
- **The bug.** `project.MAIN_ISOTOPE` has no rubidium, so `ion_name` names `85Rb_I`
  and `87Rb_I` both "Rb I", and a card's lists read "Rb I, Rb I". The two isotopes
  are neither grouped under their element (D37) nor checked for sharing z, b and T
  (D22).
- **The fix.** Rubidium's main isotope (85) joins the table, so that `87Rb_I` is
  "87Rb I", as `13C_IV` is "13C IV". A test keeps the table complete: every element
  with more than one isotope in ALIS's atomic data has its main isotope there.
- **Check.** Made into tests first:
  - in `test_dashboard_project.py`, `ion_name` gives "Rb I" and "87Rb I" (before the
    fix, "Rb I" twice), and every element with two or more isotopes in
    `alis/data/atomic.xml` is in the table (before, rubidium is not);
  - in `test_dashboard_components.py`, on a synthetic system with 85Rb I and 87Rb I
    lines, the navigator lists 87Rb I inside Rb I, and a component excluded from
    both lists "Rb I, 87Rb I";
  - in `test_dashboard_qt_components.py`, the card's lists name each once;
  - RJC's MWC137 is checked with a scratch script (logged): each list names Rb I
    once, and any notice that D22 now raises for the rubidium isotopes is named in
    the log.

**12.2 — Include an ion that a component never had, when it is excluded from every
ion (point 2, bug; `components.py`; D62, D64, D79).**
- **The bug.** A component excluded from every ion lists every ion of its system
  under Excluded ions (D79), but a click on an ion it never had is refused ("…
  include it in one first"): the new line would copy z, b and T from the
  component's live line, and it has none. On MWC137, a2 takes back Li I, Na I, K I
  and Rb I, and refuses Ti II, Ca I, Ca II and Fe I.
- **The fix.** The new line takes the component's z, b and T, with their labels,
  from its commented-out lines (the first of them in the model's order, whose values
  its card shows), and a first guess of log N, as for any component (D62). The
  component is then included in that ion and stays excluded from the others. The
  refusals that remain keep their own reasons (a label of its now on a live line,
  Q7.6).
- **Check.** Made into tests first:
  - in `test_dashboard_components.py`, on a synthetic system with a component
    excluded from its two ions (as a2), including a third ion of the system gives a
    live line with the component's z, b and T labels and values, and a first-guess
    log N (before the fix, the refusal);
  - its other ions stay excluded, ALIS reads the model, and one undo gives back the
    text;
  - in `test_dashboard_qt_components.py`, a click on that ion in the greyed card
    includes it with no warning, and the card is no longer greyed.

**12.3 — A column density that a link computes, in a drag (point 5;
`components.py`, `qt/components.py`; D24, D63, D65; Q12.6).**
- **What** (Q12.6 (a), with (b) where (a) has no single input to move).
  - **The link's own input moves.** Where the dragged ion's log N is computed by a
    link with exactly one free input that no other line or link uses (`lika` in
    `ka(lia,lika) = lia - lika`), a drag below the continuum moves that input so that
    the link gives the log N the drag asks for. The input is found as ALIS evaluates
    the link (`profile.values`), by a search on that one value within its limits; a
    limit stops the drag, as log N's own bounds do (D63). The drag is one step, which
    writes the input's value and b's.
  - **The readout** names both: "log N 13.10 (lika −2.80) · b_turb 3.40 km/s".
  - **Otherwise** (an input that other lines use, such as `lia`, no free input, or
    several), log N stays, and the readout says why: "log N is computed by the link
    ka = lia − lika". A hidden value shows only the change in χ², as now (D24).
- **Check.** New `unit` tests in `test_dashboard_components.py`, on a synthetic fit
  with a link of that form:
  - a drag below the continuum moves the variable so that the link gives the dragged
    log N (before, log N is unchanged), and b moves as before;
  - the variable's limit stops the drag, and the readout says so;
  - a link whose inputs are both other lines' column densities leaves log N, and the
    readout gives the reason;
  - a hidden variable shows only Δχ²;
  - ALIS reads the model, and one undo gives back the text;
  - a `gui` test in `test_dashboard_qt_components.py`: the readout during the drag.

**12.4 — Ignore warning, for two snips that share pixels (point 1, bug; `snips.py`,
`markers.py`, `preflight.py`, `validate.py`, `qt/regions.py`; D19, D67, Q1.8;
Q12.3).**
- **What** (Q12.3's response: an Ignore warning button for the pair; how shared
  pixels are found does not change).
  - **Ignore warning.** Beside Merge the snips… and Keep in one snip only, the
    Regions tab's offer for pixels fitted twice has Ignore warning, for that pair of
    snips. It records the pair in `ui/project.json` (the two snips' files and a
    digest of their shared pixels, as D67 keeps a notice), as one step that Undo
    takes back.
  - **The test is unchanged.** `snips.shared_pixels` and `load.find_shared_pixels`
    find the pair as now (D19, Q1.8). The warnings leave out a pair the user has
    ignored: the hatching, the offer and the Regions tab's marker, the pre-flight
    note, and ALIS's own warning among the `.mod` panel's problems (matched by the
    two files).
  - **A changed pair warns again.** A change to either snip that changes the digest
    (its pixels or its fit range) brings the warning back: the user ignored the
    pixels they were warned about, not whatever the pair shares later.
  - **Taking it back:** Undo, or Warn again beside the pair, which the Snip box
    lists under "Warnings ignored" while the pair is ignored.
  - Every other pair is warned about as now, and `run_alis` on its own still warns.
- **Check.** Made into tests first, in `test_dashboard_snips.py` and
  `test_dashboard_markers.py`, on two synthetic snips on one grid that agree outside
  a core (the shape of RJC's specids 6 and 7):
  - the pair is found as now, before and after it is ignored (`shared_pixels` is
    unchanged);
  - once ignored, `markers.fitted_twice`, the Regions tab's marker, the pre-flight
    check and `validate.check`'s problems leave it out (before, each lists it);
  - a wider fit range on one snip, which adds shared pixels, brings the warning back;
  - one undo gives back `ui/project.json` and the warning;
  - another pair of the same project is still warned about;
  - in `test_dashboard_qt_regions.py`, the offer has Ignore warning, the hatching
    goes, the Snip box lists the pair, and Warn again brings the hatching back.

**12.5 — An emptied unassigned dataset stays, and says so (item 2 carried over,
Q12.1's response; `markers.py`, `snips.py`, `qt/data.py`, `qt/regions.py`; D122).**
- **What.** A dataset is removed only by the user (Q12.1's response).
  - **CLEAR's preview**, when it takes an unassigned dataset's last snip, says what
    is left: "emsmwcs137 (unassigned) will hold no snips, and has no source spectrum:
    remove it with Remove dataset… on the Data tab, or give it snips with Load
    snips…".
  - **The Data tab** marks such a dataset: its row's state reads "0 snips ·
    unassigned: no source spectrum", with ! and the preview's advice as its tooltip,
    and the tab's marker is ! with the same advice.
  - Remove dataset… takes it away, as now.
- **Check.** New tests:
  - in `test_dashboard_markers.py`, the Data tab's marker after CLEAR of an
    unassigned dataset's last snip is ! and names the dataset (before, ✓);
  - in `test_dashboard_snips.py`, CLEAR's plan keeps the row and carries the advice;
    the plan for a dataset with a source carries none;
  - in `test_dashboard_qt_data.py`, the row's state and tooltip, then Remove
    dataset…;
  - the existing tests of CLEAR and of unassigned datasets pass unchanged.

**12.6 — Every view keeps its place (point 3; `session.py`, `qt/components.py`,
`qt/regions.py`, `qt/data.py`, `qt/fit.py`, `qt/fitcompare.py`, `qt/plots.py`,
`qt/widgets.py`, `qt/window.py`; D48, D114; Q12.4).**
- **The rule.** Nothing the dashboard does moves a scroll bar or a view's ranges; only
  the user does, with the bars, the mouse, ⌂, ← or →. A view that shows another
  transition or snip on a wavelength axis starts at its home; views on a velocity axis
  (the Components tab's panels, Inspect in velocity) keep their velocity range from
  one ion or snip to the next. A view the user has not zoomed (the progress chart
  during a fit) keeps following what it shows.
- **The fixes**, where the audit found them (see the Design):
  - **Components:** the card list keeps its place when the cards are rebuilt; a card
    selected by a click on it is not scrolled to, and one selected elsewhere (a tick,
    the `.mod` panel's cursor) is scrolled to once the cards are laid out. The panels
    keep their ranges through every step, a drag's included, and their scroll.
  - **Regions:** the list of spectra, the transition list and the Snip box's table
    keep their places.
  - **Fit · Inspect:** the view and the progress chart keep their ranges through
    every step.
  - One helper (in `qt/widgets.py`) keeps a scroll area's place across a rebuild,
    restored after the layout pass, for every tab to use.
- **Between sessions** (Q12.4 (b)): a project opens where it was left.
  - **The window's size and position** are kept in the preferences (`dashboard.json`,
    D48), since they belong to the screen. A position on no screen now (a display
    unplugged) opens the window on the main screen.
  - **Each view's ranges and each list's place** are kept in the project's
    `ui/view.json`, beside its tab, transition and ion: the Data tab's view, the
    Regions tab's spectra (D114) and lists, the Components tab's velocity range and
    each panel's flux range with the card list, and Fit's Inspect and Compare. A view
    is kept by what it shows (a spectrum's specid, a panel's transition and dataset,
    a snip), and a list by the item at its top (a card's component, a spectrum), so
    that it opens on the same item in a window of another size. What the project no
    longer has opens at ⌂.
  - **No lag** (RJC's condition). A scroll or a zoom writes nothing: the places are
    read from the views only when the project is saved or closed, or a recovery copy
    is written (a few dozen numbers, read in well under a millisecond). Places alone
    leave no unsaved changes; a clean close keeps them in the bundle, as it keeps a
    changed tab now. They are restored once, when the project opens, after its first
    layout.
- **Check.** `gui` tests, one for each place, made from the audit's script (each fails
  before the fix):
  - in `test_dashboard_qt_components.py`, a click on a card's Excluded ions link near
    the bottom of the list leaves the list where it was (before, at its top); the
    panels' zoom survives a step and a drag (before, ⌂), and another ion keeps the
    velocity range;
  - in `test_dashboard_qt_regions.py`, the list of spectra and the transition list
    keep their places through a step;
  - in `test_dashboard_qt_fit.py`, Inspect's ranges survive a step, made on the Fit
    tab or on another;
  - in `test_dashboard_qt_window.py`, the window's size and position come back from
    the preferences, and a position on no screen opens on the main screen;
  - in `test_dashboard_qt_session.py`, a project zoomed and scrolled on each tab,
    closed and reopened, shows the same ranges and the same items at the top of its
    lists, also in a window of another size; a transition removed meanwhile opens at
    ⌂; zooming and scrolling call no `set_view` and leave no unsaved changes, and
    reading the places of a synthetic project of 20 snips takes under 5 ms;
  - in `test_dashboard_session.py`, the places survive Save, a clean close and a
    recovery copy, and a bundle without them opens at ⌂, as now;
  - ⌂ still returns each view home, and the existing tests of D114's ranges pass
    unchanged.

**12.7 — Zooming the flux axis, and ⌂ for each axis (point 4; `qt/plots.py`; D38;
Q12.5).**
- **What** (Q12.5 (b)).
  - **The wheel with ⇧ Shift held** zooms the flux axis about the cursor, in every
    spectrum view: the Data, Regions and Components tabs, and Fit's Inspect and
    Compare. Without it, the wheel zooms the wavelength (or velocity) axis about the
    cursor, as now. The residual strips keep their own scale.
  - **⌂x and ⌂y** beside ⌂ on every zoom bar: ⌂x returns the wavelength (or velocity)
    axis home, and ⌂y the flux axis of each view the bar controls; ⌂ returns both, as
    now. Each is a step of ← and →.
  - The zoom bar's tooltips and the shortcut sheet (F10) say so.
- **Check.** `gui` tests in `test_dashboard_qt_plots.py`:
  - a wheel event with ⇧ held zooms y about the cursor and leaves x (before, x
    zooms); without it, x zooms and y stays;
  - ⌂x changes x alone and ⌂y y alone, and ← takes each back;
  - each tab's zoom bar has the buttons, acting on every view it controls;
  - the window's text has no design reference (D44).

**12.8 — Source spectra kept in the project (point 6; `session.py`, `qt/data.py`,
`qt/window.py`, `qt/dialogs.py`; D8, F13; Q12.7).**
- **What** (Q12.7 (a): the user chooses to save the project with its spectra or
  without them).
  - **For each spectrum**, the Data tab's row offers Keep a copy in the project, and
    once kept, Remove the copy (`bundle.embed_source`, `unembed_source`); its Source
    column says "kept in the project". As Relink… does, it changes where the spectrum
    is read from and not the model, so it is not a step of the history; the project
    has unsaved changes until Save.
  - **File → Save a copy with spectra…** writes a copy of the project with every
    spectrum kept in it, for sharing; the open project is unchanged. A spectrum that
    is missing or changed cannot be kept, and the dialog says which.
  - **A project with kept copies**, opened on another computer, reads them and asks
    nothing about relinking them (as now, `sources.EMBEDDED`); SNIP, cutting again
    and the faint source spectrum read the copy.
  - **The relink question**, when a spectrum is missing, adds that a project saved
    with Save a copy with spectra… needs no relinking.
- **Check.** New tests:
  - in `test_dashboard_session.py`, a kept copy survives Save and Open; with the
    spectrum's path gone, the project reports it as kept and `sources.spectrum` reads
    it; Remove the copy takes it out at the next Save;
  - Save a copy with spectra… writes a bundle whose sources are all kept, and leaves
    the open project's file and state as they were;
  - in `test_dashboard_qt_data.py` and `test_dashboard_qt_relink.py`, the row's
    actions, and no relink question for a kept spectrum whose path is gone;
  - `test_bundle.py`'s tests of embedding pass unchanged.

**12.9 — lsfspline's knots: the smoothing as a FWHM, the delta function, the kernel
and the step (point 7, Prompt 3; `alis/functions/lsfspline.py`, `resolution.py`, a
new `lsfknots.py`, `snips.py`, `alis/config.py`; D60, D89, D92;
Q12.8–Q12.11).**
- **ALIS's smoothing becomes a FWHM** (Prompt 3 (ii); ALIS outside the dashboard,
  as RJC asks). `lsfspline`'s first value is now the FWHM of the Gaussian that
  smooths the knots, in km/s.
  - **The change.** In `call_CPU`, σ = smoothing / 2√(2 ln 2), instead of
    smoothing / √2 (a FWHM 1.665 times the smoothing). In `set_vars`, the sub-pixels
    asked for become `vfwhm`'s, 2√(2 ln 2) · nsubpix · pixel / FWHM, instead of
    √2 · nsubpix · pixel / smoothing.
  - **Existing models.** A model written before gives a kernel smoothed 0.60 times
    as wide as it did. `CHANGELOG.md` and the function's docstring say so; nothing
    converts old models.
  - **ALIS's fewest sub-pixels per pixel** (`run nsubmin`, Q12.9's response) goes
    from 5 to 10 by default (`config.py`, `doc/ALIS_workflow.md`'s table,
    `doc/tex_files/description.tex`), so that spix is a tenth of a pixel unless the
    model says otherwise (Q12.11 (ii): the cost to every fit accepted).
  - **The regression harness.** No case uses `lsfspline` (`examples/lsf_spline` has
    no references, and no fit of `context/fitting_examples/` uses it). The new
    `nsubmin` moves the references of every case with a snip that used fewer than
    10 sub-pixels; each is regenerated with the change, and listed in the log with
    its χ² before and after.
  - **`examples/lsf_spline` becomes a case** (Q12.10 (b)): its data are generated
    again with the new smoothing, the generated files' golden copies go in
    `reference_fits/`, and the fit's `.mod.out.reference` is written. If the case is
    too slow for CI's `examples` batch (over a minute), or does not reproduce, it
    waits in `deferred_work.md`, and the unit tests stand (RJC's response).
- **What the dashboard builds.** No Qt; used by the dialog of 12.10.
  - **The knots** of a data line's `lsfspline` (`knots_of`): their velocities and
    heights, and the smoothing. A line with another function starts from RJC's
    three: (−10, 0), (0, 1) and (+10, 0) km/s.
  - **The delta function** (`delta_knots`, Prompt 3 (i)): five knots, (−edge, 0),
    (−spix, 0), (0, 1), (+spix, 0) and (+edge, 0).
    - spix is the velocity of one sub-pixel at ALIS's coarsest: the snip's pixel in
      km/s (ALIS's `get_binsize`, its largest), divided by the fewest sub-pixels
      ALIS uses (the model's `nsubmin`, or ALIS's default; Q12.9 (i)'s response).
      For a dataset, it is the largest over its snips. No knot is then finer than
      ALIS's sub-pixels in any snip, and spix is the closest that two knots may be
      (the rules below, Q12.11 (i)).
    - Where ALIS uses `nsubmin` sub-pixels for the snip, the knots' lines give it
      one non-zero point, at the centre. Where it uses more, the knots stay as they
      are (they are not changed with the sub-pixels, in the dialog or in a fit), and
      the kernel ALIS makes is a little wider than the FWHM, as the dialog shows
      (Q12.11 (i)).
    - edge is six times the FWHM (the smoothing), and at least twice spix, so that
      the five knots stay apart. The edges follow the smoothing while the knots are
      the delta function's five (Q12.9 (ii)).
    - The edges and the centre are fixed at heights 0, 1 and 0, as before; ±spix
      are inner knots, free in the fit (Q12.8 (iv)).
    - With a smoothing f, and `nsubmin` sub-pixels, ALIS's kernel is then a
      Gaussian of FWHM f, sampled on the sub-pixels. Delta Function sets the knots'
      starting values, a very narrow profile in place of the default triangle; the
      user may then move, add or remove knots within the rules, the two at ±spix
      included (Task 12.10).
  - **The smoothing's start** (Q12.8 (iii) and its response): the snip's or the
    dataset's resolution before, as a FWHM in km/s (`resolution.convert`: `vfwhm` as
    it is, `vsigma` and `Afwhm` at the snip's wavelength converted); 5 km/s where it
    has none that converts (`lsffile`, `lsf`, `apod`, another function of several
    parameters) or where the value is hidden, so that the dialog shows nothing of
    it.
  - **The rules** (`check_knots`): every height within [0, 1]; no two knots closer
    than spix, one sub-pixel at `nsubmin` (Q12.11 (i): a knot closer to its
    neighbour may have no effect in the fit); every knot between the two edges; the
    edges at height 0, moved in velocity only; the centre fixed at (0, 1) (Q12.8
    (ii)); three knots at least. A line whose knots break a rule (two knots closer
    than spix, written by hand) is read as it is, and the dialog names the rule.
  - **The kernel** (`kernel`), as ALIS makes it: ALIS's own `lsfspline` function
    applied to a spike on the snip's sub-pixels at its wavelength, as many as ALIS
    uses for the snip with the dialog's smoothing (the most of what the model's
    lines ask, `profile`'s state, and what the smoothing asks, between `nsubmin` and
    `nsubmax`), so that the dialog draws ALIS's kernel (as D60 draws ALIS's model);
    and its FWHM (`fwhm`), the width at half its peak, interpolated between points.
    The sub-pixels are worked out again when the smoothing changes, not during a
    drag.
  - **The step** (`knots_step`): the data line's resolution becomes
    `lsfspline(smoothing, heights…; locations:velocities…)`, for one snip (its own,
    D89) or for a dataset (every line that takes the dataset's, with shared labels,
    so that it is one kernel; Q12.8 (i)). What the fit frees is Q12.8 (iv)'s: the
    smoothing and the inner heights free, the edges and the centre fixed (upper-case
    labels, as in RJC's `alis2.mod`), the labels from the dataset's scheme
    (`edit.new_row_labels`); `bintype=` where the model's differs (D92). One step.
  - **The buffer:** the kernel's half width (its outer edge, plus three σ of the
    smoothing, σ = FWHM / 2√(2 ln 2)) is what the Snip box's check of each side's
    buffer asks for (`snips.buffers`), as other functions' widths are.
- **Check.**
  - **ALIS's function**, in a new `tests/test_lsfspline.py` (`unit`), made into tests
    first:
    - the delta function's knots with a smoothing of 2, 5 or 10 km/s give a
      normalised Gaussian of that FWHM, sampled on the grid, to 10⁻⁶ of its peak, on
      grids linear in wavelength and in log wavelength (before the change, a FWHM
      1.665 times the smoothing);
    - `set_vars` asks for as many sub-pixels as `vfwhm` with the same FWHM;
    - RJC's nine knots with a smoothing f give the kernel that the old code gave with
      f / 2√(ln 2): only the smoothing's meaning changed;
    - `config`'s `nsubmin` is 10, and a model's `run nsubmin 5` still gives 5;
    - `test_function_interface.py` passes; the regression harness and its
      `test_bundle` mode pass, with the references that `nsubmin` moves regenerated
      and logged, and `examples/lsf_spline` a case of the `examples` batch (or
      deferred, as above).
  - **The dashboard's**, in a new `test_dashboard_lsfknots.py` (`unit`):
    - the nine knots of RJC's `lsfspline` line are read, and written back unchanged;
    - the delta function for a snip: spix from its pixel (`get_binsize`) and
      `nsubmin` (the model's, or 10), the edges at six times the FWHM (at least
      twice spix), the rules accept the five knots, and the kernel's FWHM is the
      smoothing's where ALIS uses `nsubmin` sub-pixels, and wider where a narrow
      line asks for more; for a dataset, spix is its snips' largest;
    - two knots closer than spix are refused, and two exactly spix apart are
      accepted;
    - a line with `vfwhm` starts from the three default knots, with that FWHM as the
      smoothing; one with `vsigma` or `Afwhm` with its FWHM in km/s; one with
      `lsffile`, or a hidden value, with 5 km/s;
    - each rule refuses what breaks it, with its reason;
    - the kernel equals ALIS's function on a spike; the three default knots with no
      smoothing give a triangle of FWHM 10 km/s, and a smoothing widens it;
    - the step for a snip and for a dataset: ALIS reads the model, and one undo
      gives back the text;
    - the buffer check takes the kernel's half width;
    - a hidden lsfspline value refuses the knots, with the reason.

**12.10 — Set LSFSpline knots… (point 7, Prompt 3; `qt/dialogs.py`,
`qt/regions.py`, `qt/data.py`; D2, D43, D89; Q12.8, Q12.9).**
- **What.**
  - **lsfspline is offered** in the Snip box's resolution menu, and in the Data
    tab's (Q12.8 (i)), no longer greyed. With it chosen, **Set LSFSpline knots…**
    appears beside the menu; choosing lsfspline opens the dialog at once.
  - **The dialog** draws, with pyqtgraph (D2): the knots as squares; a thin, faint
    black line joining them in order of velocity; the smoothed kernel as a red line
    of normal weight; and its FWHM, marked at half the peak with its value ("FWHM
    9.6 km/s"). Beside the plot, the Smoothing, FWHM (km/s) field. The x axis is
    velocity in km/s, signed (D43), and the y axis runs from 0 to 1.
  - **Delta Function** (Prompt 3 (i)), a button beside the plot, asks first: "All
    the knots will be cleared, and replaced by a delta function." with Confirm and
    Cancel. Confirm puts 12.9's five knots in place of every knot, at the
    smoothing's FWHM: a very narrow profile in place of the default triangle. The
    red line is then a Gaussian of that FWHM, and the FWHM marked is the field's.
    The button only sets the knots' starting values:
    - the user sets the FWHM in the field, and the edges follow it (at ±6 FWHM)
      until a knot is moved, added or removed (Q12.9 (ii));
    - the two knots at ±spix are inner knots like any other. Each is dragged up and
      down (heights within [0, 1]) and sideways (outwards, or inwards to spix from
      the centre), or deleted, and knots are added beside them;
    - once a knot is moved, the profile is the user's, and the red line and the
      FWHM follow it.
  - **The knots:** a knot is dragged (the edges sideways only; the centre not at
    all); a double-click on the plot adds a knot there; Delete knot (also the
    right-click menu and the Delete key) removes the selected inner knot. A drag stops
    at 12.9's rules (heights within [0, 1], at least spix from the neighbouring
    knots), and the line, the kernel and the FWHM follow during the drag. A
    double-click within spix of a knot adds none, and the status line says why.
  - **Knots that break a rule** when the dialog opens (written by hand) are marked
    in red, the rule is named under the plot, and Apply is greyed until they are
    apart.
  - **Apply** makes 12.9's step; Cancel changes nothing. Values are shown through the
    gate; while one is hidden, the button is greyed and its tooltip says why.
- **Check.** `gui` tests in `test_dashboard_qt_regions.py` and
  `test_dashboard_qt_data.py`:
  - choosing lsfspline shows the button and opens the dialog with the three default
    knots; a snip with RJC's nine knots shows nine;
  - a knot dragged up stops at 1 and down at 0, and one dragged sideways stops spix
    from its neighbour; an edge moves sideways only, and the centre does not move;
  - a double-click within spix of a knot adds none; a line with two knots closer
    than spix opens with them marked and Apply greyed, until one is moved;
  - a double-click adds a knot, Delete knot removes it, and neither an edge nor the
    centre can be removed;
  - the FWHM shown is `lsfknots.fwhm`'s, and changes with the smoothing;
  - Delta Function asks first: Cancel keeps the knots, and Confirm gives the five
    knots of 12.9, whose FWHM shown is the field's (to 1%); a new FWHM moves the
    edges to six times it, and after a knot is moved they stay;
  - after Delta Function, the knot at +spix dragged up rises (the red line and the
    FWHM change with it), dragged outwards moves, and dragged inwards stops at spix
    from the centre; the knot at −spix is deleted with Delete knot; Apply writes
    the knots as moved;
  - Apply writes the line as one step, which one undo takes back; Cancel changes
    nothing;
  - the window's text has no design reference (D44).

**12.11 — The review.**
- **What.** A review page built by `doc/dashboard/stage12/`, with the earlier pages'
  stylesheet. Screenshots at 1440×900 of:
  - a card's lists with 87Rb I, and a2 included in Ti II (12.1, 12.2);
  - a drag's readout on a column density that a link computes (12.3);
  - the Regions tab's offer for pixels fitted twice, with Ignore warning, and the
    Snip box's Warnings ignored (12.4);
  - CLEAR's preview and the Data tab's row for an emptied unassigned dataset (12.5);
  - the zoom bar with ⌂x and ⌂y, and a view zoomed in y (12.7);
  - the Data tab's Keep a copy in the project, and Save a copy with spectra… (12.8);
  - the Snip box with Set LSFSpline knots…, the dialog on RJC's nine knots, and
    after Delta Function (12.10).

  Where a still cannot show it (the places kept, 12.6), the page names the test. It
  is published as a private page, and RJC's comments are applied.
- **Check.** The page builds from screenshots of the `gui` tests' scenes (synthetic
  projects, `tests/dashboard_fits.py`). Every task of 12.1–12.10 that a user sees has
  its picture, or its test named. RJC's comments are applied and logged.

**12.12 — Close the stage.**
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill (its "Stage 12 additions") up to
  date; `CHANGELOG.md` says that `lsfspline`'s smoothing is now a FWHM, and what
  that does to a model written before, and gives `nsubmin`'s new default of 10.
  Record the decisions in
  `ALIS_v2_dashboard_prompts.md` (D129 onwards, from Q12.1–Q12.10, Prompt 3 and the
  review), saying which earlier decisions they refine: D8, D19 and Q1.8, D22 and
  D37, D38, D48, D63 and D65, D64 and D79, D89, D114, D122, and F13.
  Update `dashboard_stage0.md`: Stage 12 done, and the "After v1" list. Update
  `deferred_work.md`: §1.4 (the GPU), §7.8 and §7.9 (Q12.1 (b)) stay, and anything
  this stage leaves undone is added.
- **Found and not done.** What was found during the stage and not built goes to Stage
  13's "Carried over from Stage 12", with a query, as Stage 11's close did.
- **What follows (Q12.2 (a)).** Write `claude_prompts/dashboard_stage13.md` for the
  feedback that arrives after this list closes, as Stage 11's close wrote this
  document.
- **Check.**
  - Every point of the Feedback list and the item carried over has its task logged
    as done with its check passing: point 1 in 12.4, point 2 in 12.2, point 3 in 12.6,
    point 4 in 12.7, point 5 in 12.3, point 6 in 12.8, point 7 in 12.9 and 12.10,
    point 8 in 12.1, and item 2 in 12.5; items 1 and 3 are in `deferred_work.md` §7.8
    and §7.9.
  - Q12.1–Q12.11, and any query raised while the tasks were built, are answered in the
    Queries section.
  - The review of 12.11 is applied.
  - The `unit`, `gui` and `fast` batches pass, also under CI's pytest. Their counts go
    in the log, against Stage 11's close: 2235 unit tests passed with 119 skipped
    (2234 and 120 under pytest 9.2.0), 301 `gui` tests, and 149 `fast` tests, the
    regression harness and its `test_bundle` mode included.
  - CI passes on both platforms, the `gui` job within its 90 minutes.
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44), and pre-commit passes on every
    changed file.
  - ALIS outside `alis/dashboard/` changes only as Task 12.9 says (`git diff --stat
    1e4c2f4 -- alis ':!alis/dashboard'` lists `alis/functions/lsfspline.py`, and
    `alis/config.py` for `nsubmin`, and nothing else, against Stage 11's close), and
    the regression harness passes, its regenerated references logged.
    This stage's tests are synthetic and need none of RJC's files; MWC137 is
    checked with scratch scripts, logged.

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view, the Fit tab's runs, the Figure
  tab, and the lessons of Stages 10 and 11 (their "additions").
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- `dataviz`: where a point touches a figure's colours or marks.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D128.
- `claude_prompts/dashboard_stage11.md` and its log: how users' feedback was handled
  last time, from the list to the close, the design pausing for RJC's queries.
- `claude_prompts/deferred_work.md`: what was deferred (§1.4, the GPU kept on the
  device, a series of its own; §7, from the dashboard's stages).
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 11's builder is `doc/dashboard/stage11/`; its synthetic fits come from
  `tests/dashboard_fits.py`.
- The users' files, where the Feedback list names them (D99).

## Queries

*Raised by Claude: Q12.1 and Q12.2 with this document (2026-11-13), Q12.3–Q12.8 at
Prompt 1 (2026-11-16), all answered by RJC and applied at Prompt 2; Q12.9 and Q12.10
at Prompt 3, answered and applied at Prompt 4; Q12.11 at Prompt 4, answered and
applied when Prompt 4 was run again. Each gives Claude's lean. A question that comes
up while a task is built pauses the work; it is asked here, and its answer is logged
here (Stage 11, Prompt 2).*

**Q12.1 — The items carried over from Stage 11.** For each, two ways:
- **(a) It joins this stage**, after the feedback's tasks.
- **(b) It waits** for a user to ask for it, in `deferred_work.md` §7.

My lean: (b) for item 1 (a file that names no transition has nowhere to go without a
guess the user would have to check) and item 3 (RJC's wording fits every fit); (a)
for item 2, a small fix that leaves no empty dataset behind.

**Response:** I agree with your lean. For item 2, we need to make sure a dataset (even if it is unassigned and contains no SNIPs) is only removed explicitly by the user. A warning message could instead be displayed to the user so they know they can remove the dataset, as it currently contains no SNIPs and is unassigned. If they wish, they could alternatively cut a SNIP from this unassigned dataset.

*Applied (Prompt 1, 2026-11-16): item 2 joins the stage as Task 12.5. CLEAR keeps the
emptied dataset, as now, and the user is told that it holds no snips and has no
source, and how to remove it or give it snips. A snip cannot be cut from it, since
it has no source spectrum, so the advice names Load snips…, which gives it snips.
Items 1 and 3 wait in `deferred_work.md` §7.8 and §7.9.*

**Q12.2 — What follows this stage.** Two ways:
- **(a) Another stage of feedback** (Stage 13) for what arrives after this list
  closes, as now.
- **(b) Another series first**: the GPU kept on the device (`deferred_work.md` §1.4),
  the documentation (`documentation_stage0.md`), or an item marked "Later" in
  `dashboard_stage0.md`.

My lean: (a), unless RJC names one of (b).

**Response:** I agree with option (a), for now.

*Applied (Prompt 1): Task 12.12 writes `dashboard_stage13.md` for the feedback that
arrives after this list closes.*

**Q12.3 — Two snips whose pixels agree, which the user says are different data
(point 1, Task 12.4).** Specids 6 and 7 of MWC137 agree digit for digit at 297 of
336 pixels, and differ only in K I 7701's core. By Q1.8's test, the 147 pixels both
fit outside the core are the same pixels of the same data, and ALIS warns as the
dashboard does. Q1.8's response says that pixels are shared "only if they are
extracted from the same parent data"; the values of these two snips say they are,
and the user says they are not. Three ways:
- **(a) The user says so, for each pair.** The Regions tab's offer for pixels fitted
  twice gains a third choice, They are different data, kept with the project. That
  pair is then no longer warned about anywhere in the dashboard, until either snip's
  pixels or fit range change. `run_alis` on its own still warns.
- **(b) By dataset.** Two snips in different datasets are never taken as the same
  pixels, unless the datasets share a source spectrum. MWC137's pair is cleared with
  no click, since the dashboard put specid 6 in a dataset of its own when it loaded
  the fit. But a snip loaded unassigned (with no source) would then never be checked
  against another dataset's snips, though it may have been cut from the same
  spectrum.
- **(c) As now.** The warning stays, and the fit runs (QF.21 (c)).

My lean: (a). The test keeps its protection for every pair the user has not looked
at, and the user's word is kept for the pairs they have.

**Response:** Let's introduce a button that says "Ignore warning" that the user can click to remove the warning for that pair of SNIPs. This would allow the user to indicate that they are aware of the issue and have chosen to ignore it, without changing the underlying logic of how shared pixels are determined.

*Applied (Prompt 2, 2026-11-16) in Task 12.4: the offer for pixels fitted twice has
Ignore warning, for the pair, as one step. `shared_pixels` and ALIS find the pair as
now; the dashboard's warnings (the hatching, the offer, the marker, the pre-flight
note, ALIS's warning in the `.mod` panel) leave it out. The pair warns again if
either snip's pixels or fit range change what it shares, and Warn again, in the
Snip box's list of warnings ignored, or Undo, takes it back.*

**Q12.4 — Places kept between sessions (point 3, Task 12.6).** Point 3 asks that
scroll bars, zooms and the window's size be remembered. Within a session, Task 12.6
keeps every place through every action. Between sessions, two ways:
- **(a) The window's size and position**, kept in the preferences
  (`~/.alis/dashboard.json`) and restored when the dashboard opens. Each project's
  view is as now: its tab, transition, ion, Fit view and the `.mod` panel's width.
- **(b) As (a), and each view's zoom and each list's scroll**, kept in the project's
  `ui/view.json`, so that a project opens exactly where it was left.

My lean: (a). The window's size belongs to the screen it is on, and a zoom to the
look being taken; a project reopened days later starts from ⌂, while the tab,
transition and ion it shows are kept as now.

**Response:** I prefer option (b), to also store each view's zoom and each list's scroll in the project's `ui/view.json`. This would provide a more seamless user experience, allowing users to pick up exactly where they left off when reopening a project. The only concern I have is if this causes any lag in the interaction with the GUI. Provided no significant lag is introduced, I think this would be a valuable feature.

*Applied (Prompt 2) in Task 12.6: the window's size and position in the
preferences, and each view's ranges and each list's place in `ui/view.json`, kept by
what each shows. A scroll or a zoom writes nothing; the places are read only at
Save, a clean close or a recovery copy, and restored once when the project opens.
The task's check times the reading (under 5 ms on 20 snips) and checks that zooming
and scrolling write nothing.*

**Q12.5 — The key that zooms the flux axis (point 4, Task 12.7).** Two ways:
- **(a) Y held**, as point 4 suggests. Qt reports only modifier keys with a wheel
  event, so a held Y needs the Components tab's way with N and B: pressed while the
  pointer is over a spectrum, the key is the view's and nothing else receives it. A
  "y" typed in the `.mod` panel while the pointer rests on a spectrum would be taken
  by the view.
- **(b) ⇧ Shift held.** Qt gives its state with each wheel event, so nothing is
  taken from the keyboard, and it works the same with a mouse wheel and a trackpad
  (on a Mac, ⇧ turns a mouse wheel's scroll sideways, which the view reads as the
  same zoom).

My lean: (b). ⌂x and ⌂y are as point 4 asks, either way.

**Response:** I agree with option (b). Using ⇧ Shift held for zooming the flux axis is more intuitive and consistent with common user interface conventions. It also avoids potential conflicts with other keyboard shortcuts and ensures that the zoom functionality works seamlessly across different input devices, such as a mouse wheel and a trackpad.

*Applied (Prompt 2): Task 12.7 as written.*

**Q12.6 — A column density that a link computes (point 5, Task 12.3).** D65: links
are written in the `.mod` panel only, and the Components tab never drags or types
what they compute. Point 5 asks whether a drag could still change it. Three ways:
- **(a) The link's own input moves**, where there is exactly one: free, and used by
  no other line or link (`lika` in `ka = lia − lika`, the log Li/K ratio). The drag
  moves it so that the link gives the dragged log N, and Li I's log N stays. Where
  there is not exactly one such input, as (b).
- **(b) The readout only.** log N stays, and the readout says why ("log N is computed
  by the link ka = lia − lika").
- **(c) As now.**

My lean: (a), with (b) where (a) has no single input to move. It refines D65 for
drags only: links are still written in the `.mod` panel.

**Response:** I agree with option (a) with the fallback to (b) when there is no single input to move. This approach allows for more intuitive interaction when adjusting column densities, while still providing clear feedback to the user when adjustments cannot be made due to the nature of the link. It maintains consistency with the existing behavior of links being written in the `.mod` panel, while enhancing the user experience during drags.

*Applied (Prompt 2): Task 12.3 as written.*

**Q12.7 — Source spectra in the project (point 6, Task 12.8).** D8 offers to embed a
spectrum "for archiving". The project file can hold one, and the dashboard reads it,
but the window has no way to put one there. Three ways:
- **(a) The user's choice, for each spectrum:** Keep a copy in the project, on the
  Data tab; and File → Save a copy with spectra…, which writes a copy of the project
  with every spectrum kept in it, for sharing, and leaves the open project as it is.
- **(b) Always:** every spectrum is kept in the project as it is added, and those of
  older projects at their next Save.
- **(c) As (a), with a preference** to keep a copy of every spectrum added to a new
  project (off to start with).

My lean: (a). MWC137's two spectra would add about 15 MB to its 1.7 MB file, and
about a second to each Save and to each minute's recovery copy. Most projects stay
on one computer, and Save a copy with spectra… serves the sharing that point 6 asks
for.

**Response:** I agree with option (a), that the user should be able to choose if they wish to save a `.model` file with datasets, or without datasets.

*Applied (Prompt 2): Task 12.8 as written. Save keeps a spectrum in the file only
where the user chose Keep a copy in the project; Save a copy with spectra… writes a
file with all of them.*

**Q12.8 — Set LSFSpline knots… (point 7, Tasks 12.9 and 12.10).** Four points:
- *(i) Where it is offered.*
  - **(a)** In the Snip box (a snip's own resolution, as point 7 asks) and in the Data
    tab's Resolution (a dataset's: one kernel, shared by its snips).
  - **(b)** In the Snip box only.
- *(ii) The centre knot* (0 km/s, height 1).
  - **(a)** Fixed: neither moved nor removed. ALIS normalises the kernel, so the
    centre's height only sets the scale; and moving it sideways would shift the
    line, which the dataset's shift already does.
  - **(b)** Moved sideways, its height fixed at 1.
- *(iii) The smoothing's starting value*, for a snip that had another function.
  - **(a)** The snip's resolution before, in km/s (2.14 km/s for most of MWC137's
    snips).
  - **(b)** Zero: the knots' straight lines alone (from the three default knots, a
    triangle of FWHM 10 km/s).
  - **(c)** A value you name.
- *(iv) What the fit frees.*
  - **(a)** As your `alis2.mod`: the smoothing and the inner knots' heights free; the
    edges and the centre fixed. The knots' velocities are fixed, since ALIS reads
    them as a keyword.
  - **(b)** Every height fixed, the smoothing free.

My lean: (a) for all four.

**Response:** I agree with option (a) for all four points. If the resolution is already specified for a dataset/snip, then we should use that value as an estimate. If the resolution is not available for a dataset/snip, the default value should be 5 km/s.

*Applied (Prompt 2) in Tasks 12.9 and 12.10. The smoothing starts at the snip's or
the dataset's resolution as a FWHM in km/s: `vfwhm` as it is, and `vsigma` and
`Afwhm` converted at the snip's wavelength. Where none converts (`lsffile`, `lsf`,
`apod`, another function of several parameters), or the value is hidden, it starts
at 5 km/s. ALIS smooths with a Gaussian of σ = smoothing/√2 (FWHM 1.67 times the
smoothing), so the kernel starts a little wider than the resolution before. The fit
frees the smoothing, and the dialog shows the kernel's FWHM.*

*Prompt 3 (ii) (2026-11-16): Task 12.9 makes the smoothing the Gaussian's FWHM, so
the smoothing's start is the resolution before exactly.*

**Q12.9 — The delta function (Prompt 3 (i), Tasks 12.9 and 12.10).** Two points:
- *(i) The sub-pixels in spix.* ALIS has no fixed number of sub-pixels. For each
  snip, it uses the most that any line or the resolution asks for, between `nsubmin`
  (5) and `nsubmax` (21) (`load.load_subpixels`). The number comes from the model's
  b and FWHM values, so it changes as they do, and during a fit with `renew_subpix`.
  The knots at ±spix give ALIS one non-zero point only while no sub-pixel lies closer
  than spix to the centre. If the sub-pixels later get finer, the delta function
  becomes a narrow triangle.
  - **(a) `nsubmax`:** spix is the snip's smallest pixel in km/s divided by
    `nsubmax` (the model's, or 21), the finest sub-pixel ALIS can use. The kernel is
    then a Gaussian of the FWHM whatever the sub-pixels become. The cost is ±spix
    knots closer to the centre (0.1 km/s for a pixel of 2 km/s).
  - **(b) The snip's sub-pixels now,** as ALIS works them out for the model as it is
    (the number ALIS's log reports). The kernel is exact until a change to the
    model asks for more sub-pixels.
- *(ii) The edges when the FWHM changes after Confirm.* The edges are at six times
  the FWHM.
  - **(a) They follow the FWHM** while the knots are the delta function's five. Once
    a knot is moved, added or removed, they stay where they are.
  - **(b) They are placed once,** at Confirm, from the FWHM then; a new FWHM moves no
    knot.

My lean: (a) for both. (i) keeps the kernel exactly Gaussian, as Prompt 3 asks. With
(ii), the user sets the FWHM after Confirm, and the knots stay as Prompt 3 defines
them. The edges change nothing in the kernel (it is zero between ±spix and the
edges), but they set the buffer each side needs.

**Response:** For point (i), I actually think it makes more sense to adopt the value of nsubmin for spix. I think we want to avoid choosing a value that is finer than whatever the subpixellation value is. Otherwise, we will be trying to generate an LSF on a subpixellation scale that is finer than specified. We should probably reset the default nsubmin to be 10, so that the minimum subpixellation is 10 by default. For point (ii), I agree with option (a). The edges should follow the FWHM set by the user if they change it.

*Applied (Prompt 4, 2026-11-16) in Task 12.9. spix is the snip's smallest pixel
divided by `nsubmin` (the model's, or ALIS's default, raised to 10); the kernel is
drawn on the sub-pixels ALIS uses for the snip. The edges follow the FWHM while the
five knots stand (Task 12.10, as written). Q12.11 asks RJC to confirm two
consequences.*

**Q12.10 — lsfspline in the regression harness (Prompt 3 (ii), Task 12.9).** Prompt
3 expects the change to move the regression harness's references. The harness has no
case that uses `lsfspline`: `examples/lsf_spline` has no `.mod.out.reference` and no
golden data, and no fit in `context/fitting_examples/` uses the function. So no
reference moves, and nothing would catch a later change to the function. Two ways:
- **(a) Unit tests of the function alone** (Task 12.9's check: the delta function
  gives a Gaussian of the FWHM, and the knots' shape is unchanged).
- **(b) As (a), and `examples/lsf_spline` becomes a case.** After the change, its
  generated data goes to `reference_fits/` and its fit's `.mod.out.reference` is
  written. The harness then checks `lsfspline` as it checks the other functions, in
  CI's `examples` batch (a small fit of one spectrum).

My lean: (b). It keeps the function covered as every other shipped example is.

**Response:** I agree with option (b). Let's attempt to capture this in the regression harness. If we encounter problems with the regression harness, (e.g. the fit takes too long) then we can delay this to a future task, and just implement the unit tests for now.

*Applied (Prompt 4, 2026-11-16) in Task 12.9: `examples/lsf_spline` becomes a case,
its data generated again with the new smoothing. It waits in `deferred_work.md` if
it takes over a minute in CI's `examples` batch or does not reproduce, and the unit
tests stand.*

**Q12.11 — Two consequences of `nsubmin` for spix (Q12.9's response, Task 12.9).**
Q12.9's response takes spix from `nsubmin`, never finer than ALIS's sub-pixels, and
raises ALIS's default `nsubmin` from 5 to 10. Task 12.9 is written so. Two
consequences, for RJC to confirm:
- *(i) The kernel is exactly Gaussian only where ALIS uses `nsubmin` sub-pixels.*
  ALIS uses the most sub-pixels that any line or the resolution asks for, and
  `lsfspline` asks for 2√(2 ln 2) · nsubpix · pixel / FWHM. For a FWHM of 2 km/s on
  2 km/s pixels, that is 12. The knots at ±spix then span more than one sub-pixel,
  and the kernel is the Gaussian smoothed by a narrow triangle. With `nsubmin` 10 it
  is 0.5% wider than the FWHM; with 5, 2.0% wider (a scratch calculation, logged).
  The dialog draws ALIS's kernel on ALIS's sub-pixels, so the FWHM it shows is the
  width the fit uses.
  - **(a) As the response:** spix from `nsubmin`. The kernel is the Gaussian where
    ALIS uses `nsubmin` sub-pixels, and a little wider where it uses more.
  - **(b) spix from ALIS's sub-pixels for the snip,** as ALIS works them out with
    the dialog's FWHM (12 in the example). It is never finer than ALIS's sub-pixels,
    and the kernel is exactly the Gaussian for the model as it is. It is fixed when
    written: a later line that asks for more sub-pixels makes the kernel a little
    wider, as in (a).
- *(ii) The default `nsubmin` of 10 changes every fit.* Each snip's model is worked
  out on at least 10 sub-pixels per pixel, not 5:
  - the regression harness's references move wherever a snip used fewer than 10,
    and are regenerated;
  - a fit of broad lines, which used 5, takes up to twice as long to evaluate;
  - the `lsfspline` delta function needs no new default, since `lsfspline` asks for
    the sub-pixels its own FWHM needs (12 above).
  - **(a) The default becomes 10,** as the response says, with `CHANGELOG.md`.
  - **(b) The default stays 5;** spix uses the model's `nsubmin`, and a user who
    wants finer sub-pixels sets `run nsubmin 10` in the model.

My lean: (b) for both. (i)(b) keeps both aims of the response and Prompt 3: never
finer than ALIS's sub-pixels, and exactly Gaussian. For (ii), the delta function
does not need the new default, which would cost every user time and change every
fit's references. (a) for (ii) remains if RJC wants finer sub-pixels for every fit
in its own right.

**Response:* For point (i) lsfspline should always use the knots as they are defined; the knots should not change with the sub-pixel scale during the fit. The "Delta Function" button just sets the initial values of the knots. In principle, the user is free to change these points, but any knot that differs by `nsubmin` sub-pixels from the nearest knot may not have any effect in the fit. Therefore, I propose that we keep every know separately by *at least* `nsubmin` sub-pixels. This is why I propose the "Delta Function" button to set the initial values of the knots to be this minimum allowable value. The user can then adjust the knots as they see fit, but any knot that is closer than `nsubmin` sub-pixels to another knot should not be allowed. For point (ii), I agree with option (a).

*Applied (Prompt 4, run again, 2026-11-16) in Tasks 12.9 and 12.10. The closest two
knots may be is spix, read as one sub-pixel at `nsubmin`: the pixel divided by
`nsubmin`, the separation that Delta Function gives ±spix. spix is taken from the
snip's pixel as ALIS sizes it (`get_binsize`, its largest), and for a dataset from
its snips' largest, so that no knot is finer than ALIS's sub-pixels in any snip. The
knots are what the user sets, and never change with the sub-pixels, in the dialog or
in a fit. A drag stops spix from a neighbour, and a double-click within spix adds
no knot. A line written by hand with knots closer than spix opens with them marked,
and Apply is greyed until they are apart. ALIS's default `nsubmin` becomes 10.*

## Prompts

1. Please read the list of changes and bug fixes, and the work of Stages 0–11 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Feedback`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (12.1, 12.2, etc.). The final Task (currently labelled 12.N) should be to close the stage, and should include the checks that will verify that the stage is complete.
2. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (12.1, 12.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
3. Regarding Task 12.9 (Q12.8), there are two things to clarify: (i) We should introduce a button called `Delta Function`. By clicking this button, a warning should appear to warn the user that all knots are about to be cleared, and replaced by a delta function, with two options to "Confirm" or "Cancel". If "Confirm" is selected, all knots are removed, and there would be five knots automatically inserted. There will be three inner knots with y-values (0.0, 1.0, 0.0), and x-values equal to (-spix, 0.0, +spix), where spix is the subpixellation scale (i.e. the velocity of one pixel divided by the number of subpixels). Then, there will be two outer knots with y-values equal to (0.0, 0.0), and x-values equal to (-edge, +edge), where edge is six times the velocity FWHM. Note that the y-values at (-edge, 0.0, +edge) should all be fixed at (0.0, 1.0, 0.0), as before. In this case, the user should be able to set the FWHM. For a delta function defined in this way and convolved with a user-input FWHM (i.e. the smoothing scale) should produce a line spread function that is exactly a Gaussian with the user-input FWHM as the line's FWHM. (ii) The LSFSpline smoothing parameter is not currently the FWHM of a Gaussian when the knots are a delta function. We should modify the LSFSpline code so that the smoothing parameter is the FWHM of the smoothing function. This will change the gate, so we may need to regenerate that test and output. Please include this in Task 12.9. If you have any queries about these points, or anything else to be developed during this stage, please ask in the Queries section, and I will respond.
4. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please check that the ordered list of tasks is correctly written. Each task should be numbered sequentially (12.1, 12.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.
5. Please execute the tasks in order. There's no need to pause between tasks, unless you have queries that require my thoughts and responses. Please check your context limit between tasks. If you reach 75% context, please pause and ask for my guidance on the next steps. If you have queries during development, please pause the development and log the queries and responses in the Queries section.