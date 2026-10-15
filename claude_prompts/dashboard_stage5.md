# Prompt file for ALIS software dashboard creation -- STAGE 5

> **The Components tab (step 4).** The third tab of the skeleton is filled in. At the
> end of the stage, a user can take a project with snips and continua (Stage 4) to a
> model that is ready to fit, without typing in the `.mod` panel:
> - **The ion navigator (S30, D37):** each system and its ions, with the number of
>   snips and components; isotopes inside their element (D I with H I); one
>   Lyα forest group for the project; Add ion… and Add system….
> - **The panels (D37, S13):** every snip that holds a transition of the chosen ion
>   (Q5.19), in a chosen dataset or in all of them, on a common velocity axis relative
>   to the system's z at the snip's strongest transition (±200 km/s by default), in
>   1–4 columns; the data normalised by their continuum,
>   the model as it will be fitted, and the residuals underneath; each component
>   marked by a tick, other systems' lines drawn but locked.
> - **Components (S11, S12, D20–D22):** "+ Add component" and a click add a
>   component at that velocity, with a line for every ion of the system and each
>   log N first-guessed from the apparent optical depth; a click on a tick selects a
>   component; a drag moves it (velocity), or deepens and widens it (log N and b)
>   together, with the model updated live in every panel of the ion; the cards set
>   every value free, fixed or tied, the temperature mode and the blinding, and
>   exclude a component from an ion or include it again.
> - **Blends, links and removal (S14, S15, S27, D23):** interlopers in one click;
>   links and limits written in the `.mod` panel read, shown and honoured; a
>   component removed after a confirmation that lists every line.
> - **Imported models (Q0.3):** a notice, with a one-click fix, for components whose
>   ions do not share z (D20) or isotopes that do not share z, b and T (D22).
>
> Everything the tab shows comes from the project model, through the blinding gate,
> and every action is one step of the history, as in Stages 3 and 4. The logic needs
> no Qt and goes in `alis/dashboard/` (two new modules: `profile.py` and
> `components.py`), tested with pytest alone; the windows go in `alis/dashboard/qt/`.
> ALIS outside `alis/dashboard/` does not change, unless a query agrees otherwise.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md`. "Q0.*n*" to "Q4.*n*" refer to
> `dashboard_stage0.md` to `dashboard_stage4.md`. The plan for all stages is in
> `dashboard_stage0.md`.

## Design

*Written by Claude on 2026-10-08, from the design documents (D20–D24, D37, D43, D44,
S11–S16, S27, S30; QF.8, QF.10, QF.20, QF.25, QF.28), the mockups of the Components tab
(`doc/dashboard/mockups/components.html`, `components_hd.html`), RJC's review of them
(Task 0.10 and Q0.3 of `dashboard_stage0.md`) and what Stages 2–4 taught. Updated on
2026-10-14 (Prompt 2) to follow RJC's responses to Q5.1–Q5.18; each part names the
queries that settled it.*

### What a user can do at the end of this stage

1. **New project, Data and Regions** (Stages 3 and 4): the spectra, the systems, the
   snips with their fit regions and continua.
2. **Components:** choose an ion in the navigator; its transitions are shown side by
   side on one velocity axis, in the dataset chosen above them or in all of them.
   "+ Add component" and a click add a component there, with each ion's log N guessed
   from the data. Dragging its tick sets its velocity, and dragging its handle its
   log N and b, and the model follows in every panel. The cards set each value free,
   fixed or tied, the temperature and the blinding, and exclude a component from an
   ion or include it again. Interlopers are added in one click; links and limits
   written in the `.mod` panel are read and honoured.
3. **Fit** (Stage 6) follows. Until then, `run_alis project.model` fits, and the
   panels show the model that will be fitted.

### Layers

```
alis/dashboard/qt/   components.py  the Components tab: the navigator, the panels and
                                    the cards
                     plots.py       the spectrum view gains a model curve, residuals,
                                    component ticks and a handle
                     regions.py     the Regions tab's Model switch (Q5.8)
alis/dashboard/      profile.py     the model of each snip, as ALIS computes it, from
                                    the text or from values given during a drag; χ²
                                    on the fitted pixels
                     components.py  adding components and ions, the first guess of
                                    log N (AOD), temperature modes, drags as values
                                    and their bounds, excluding and including,
                                    interlopers, the notices' fixes
```

The names may change as the code is written; the log records the names as built.

- **No logic in the widgets**, every action one step of the `History`, every value
  through the `Gate` (Stages 3 and 4). A drag is one step (`history.group`).
- **The model drawn is ALIS's.** `profile.py` builds ALIS's own state for the
  project, as `continuum._load_with_alis` does, and evaluates
  `alis.model_eval.model_func` over every pixel of each snip (`output=1`), with the
  links applied as ALIS applies them (`load.load_tied`). What is drawn is what ALIS
  fits: the same atomic data, sub-pixellation and convolution. The panels draw that
  model divided by the continuum (`continuum.evaluate`). Components are marked by
  ticks, so no curve of one component alone is needed (`getemab` no longer returns
  the parts, since refactor Stage 3.5.4). A drag changes the values given to the
  evaluation, not the text; the text changes once, when the drag ends.
- **Blinded profiles may be drawn** (D24): a blinded component's curve, tick and
  handle are shown and can be dragged, but no number of it is: its card shows the
  mask (`▒▒▒▒`), and a drag reports only the change in χ² since it began (QF.20(d)).
- **A component is in every ion of its system** (D20) by default, and can be excluded
  from one (Q5.11, Q5.15): its line for that ion is then commented out. Isotopes share
  z, b and T with their element (D22). An imported model that breaks either gets a
  notice with a one-click fix, and may still be fitted as imported (Q0.3).

### The ion navigator (S30, D37; Q5.7, Q5.14)

- **Groups:** the primary system first, then each other system, each with its ions:
  "H I · 8 snips · 2 comp. · with D I" (Q5.19); isotopes only inside their element.
  Last, one **Lyα forest** group for the whole project: every `1Ly_a` and `1H_IB`
  line, wherever it falls ("Lyα forest · 57 components in 11 snips").
- **Add ion…** lists first the ions with a transition inside any snip, whatever the
  snip was cut for (O I 1302 in a Si II 1304 snip), then every other ion of the atomic
  data. The new ion gets a line in every component of the system, its log N guessed
  (below), with the specids of the snips that hold its transitions (Q4.3). An ion
  whose transitions fall in no snip lists the system's snips (ALIS needs a specid on
  every line when there are several), and the tab's marker shows "!" until a snip
  covers one. **Add system…**: as on the Data tab (`datasets.py`).
- **Keys** (the tab's own, as in Stage 4): ↑ ↓ the previous or next ion. No key
  removes a component (Q5.15(c)).

### The panels (D37, S13; Q5.2, Q5.3, Q5.8, Q5.14)

- **The dataset:** a Dataset menu above the panels lists each dataset that covers one
  of the ion's transitions (the reference dataset first), then "All datasets". It is
  one setting for the tab, kept in `ui/view.json`, and absent when the project has one
  dataset.
- **One panel per snip** (Q5.19) that holds a transition of the ion at the system's
  z, in the chosen dataset, or in every dataset ("All datasets"), ordered by the
  strength (fλ) of each snip's strongest transition, then by dataset. Zero velocity
  is that strongest transition, whose component ticks are dragged; the components
  are drawn again, thin, at the snip's other transitions. Titled "O I 1302.2 · HIRES",
  "O I 988.8 (+2) · HIRES" with two others inside, and with the snip named when it
  was cut for another transition ("O I 929.5 (+2) · Ly7 926.2 snip · HIRES"). A
  transition that leads a panel of another dataset but that the chosen one does not
  hold is listed as such ("O I 988.8: not in PROCHASKA"). Every dataset is drawn in the model's
  frame (its shift removed), so a component's tick is at the same velocity in each.
- **Layout:** a column setting of 1–4 for the whole tab, and vertical scrolling;
  ±200 km/s by default, with a menu of ranges (D37).
- **What each panel shows:** the data divided by the Regions tab's continuum, as
  steps; the model divided by the same continuum, so that the absorption is drawn
  relative to 1 (the data are never divided by the absorption); the fit regions
  shaded; the residuals, (data − model)/σ, in a strip underneath with ±1σ lines; the
  velocity axis relative to the system's z (D16), signed (D43).
- **The components:** a tick at each component's velocity, labelled with its id (a1,
  a2…); an excluded component's tick is faint and dashed. Other systems' lines and the
  forest's are drawn but locked (S30), so that blends stay visible.
- **The Lyα forest group:** a panel for each snip that holds one of its lines, in the
  velocity frame of the transition that snip was cut for (its record in
  `ui/project.json`), with a tick where each line falls.
- **The Regions tab's Model switch** (Q5.8), beside Normalised and off by default,
  draws the same model on the Regions tab's spectra; Stage 6 adds the best fit.
- Built on Stage 4's `SpectrumView` (velocity axis, zoom bar, linked x axes); the
  panels' x axes are linked in velocity.

### Adding, selecting and dragging (S11–S13, D20, D21; Q5.4–Q5.6, Q5.12–Q5.17)

- **Add component:** "+ Add component" arms one click. The next click in a panel adds
  a component at that velocity; the cursor shows that the button is armed, and Esc
  cancels. The component takes the next labels (`edit.new_component`), a line for
  every ion of the system with the specids of the snips that hold its transitions,
  b = 5 km/s (a preference), the default temperature mode (0 K fixed, a preference),
  and each ion's log N from the first guess. One step.
- **Selecting** (Q5.23): a click selects the component nearest in velocity (its thin
  ticks at the snip's other transitions count; other systems' and the forest's
  locked lines do not). Its card is highlighted and scrolled into view, and its
  lines are highlighted in the `.mod` panel (S16). Esc clears the selection. A click
  never adds a component unless + Add component is armed.
- **The first guess of log N** is made for every ion of the new component, the
  clicked one included; the clicked transition gives only the velocity.
  - Each of the ion's transitions inside any snip, in each dataset that covers it,
    gives N = 3.768×10¹⁴ ∫τ_a dv / (fλ) over ±0.5 FWHM_line around the component's
    velocity, divided by 0.76 (the fraction of a Gaussian within ±½ FWHM).
    FWHM_line = √(FWHM_inst² + FWHM_V²), FWHM_V = 2√ln 2 · b, with the ion's total b
    and that dataset's FWHM in km/s at the line's wavelength.
  - τ_a = ln(M/F), M being the current model without the new component, so that a
    blend or a neighbouring component is not counted twice.
  - An estimate counts when its absorption is detected at 3σ or more and is not
    saturated (flux within 3σ of zero). The start is the lowest of those, since a
    blend only adds absorption.
  - When nothing is detected, the start is the lowest 3σ upper limit. When every
    detected transition is saturated, it is 15.0 (a preference) or the highest lower
    limit, whichever is larger, and the card says so. With no data at the velocity, it
    is 12.0 (a preference), and the card says "no data at this velocity".
- **Dragging** (Q5.23): a drag moves the component nearest where it starts, which it
  selects, by the drag's movement (never to the pointer):
  - above the continuum (by the ticks), its z, shared by its ions (D20);
  - below it, log N and b together: log N by 2 dex over the panel's height
    (downwards deepens), and b by how much farther from the component the pointer
    is. N held keeps b as it is, B held keeps log N (pressed over a panel, the keys
    are the panel's). There is no handle;
  - a right drag pans the panels, and the wheel zooms;
  - b is the line's total b: b_turb = √(b² − b_th²) for the temperature mode, and in
    the pure thermal mode the drag sets T instead;
  - **bounds:** limits written in the `.mod` (`lim voigt ColDens …`, `lim param …`)
    bound the drag. Where none is written, b_turb stops at 0.5 km/s and T at 0
    (voigt.py's limits), and log N at a range of the dashboard's, 8.0 to 23.0 (a
    preference). A value typed on a card is taken as typed;
  - a linked parameter is not dragged and cannot be typed (Q5.9). Dragging a
    parameter that others are linked to moves them too, as ALIS computes them;
  - an excluded component can be selected but not dragged in that ion's panels;
  - the model is evaluated again as the drag moves (S13), in every panel of the ion
    and in the other panels where the component's ions are drawn. The panel's corner
    shows log N and b through the gate (only the change in χ² for a blinded
    component). The text changes once, when the drag ends, as one step.
- **Live preview speed:** ALIS's own model at every position, throttled to the rate
  the machine allows (a preference). Task 5.1 measures the cost on J1358p6522's H I.
  If it is too slow, a drag evaluates with ALIS's sub-pixels capped at 5
  (`run nsubmin`), and with the model's own setting when the drag ends.

### The cards (S11, D20–D22, D37; Q5.9, Q5.14, Q5.15)

One group of cards per system, and one for the Lyα forest; in each, one card per
component, in order of velocity:
- **Header:** "a1 · v = 0.0 km/s", Remove (✕) and Blind….
- **Exclude/Include:** one button for the navigator's ion, "Exclude from O I" or
  "Include in O I". An excluded component's card is greyed, except for that button.
  A component that never had the ion (one an imported model left out) shows the same,
  and Include adds a line with a first guess.
- **z** and **b_turb**, each with its state badge (● free, ■ fixed, ⛓ tied, as the
  Data tab's), a click on which offers Free, Fixed and Tie to <label> (`edit.fix`,
  `free`, `tie`), and the limits (`lim`, `edit.set_limits`). A linked parameter has a
  badge of its own, with its expression in the tooltip (through the gate), and is
  locked.
- **Temperature:** a menu of the modes of D21: 0, 100 or 10⁴ K fixed, "Fixed at…"
  (any value, as imported models have), free T, or pure thermal (b_turb fixed at 0,
  T free).
- **Each ion's log N**, for the ions the component is in, with its badge and the
  first guess's note (saturated, no data); isotopes nested under their element
  ("↳ D I"). The system's other ions, excluded or never in it, on one line,
  "Excluded: D I, O I"; a click on one includes it (Q5.20).
- **A Lyα forest card:** the wavelength of its line, log N and b.
- **Values through the gate:** a blinded value shows `▒▒▒▒`; typing over it is allowed
  and hides it again once made (QF.20(c)).

### Blends and links (S14, S15; Q5.9, Q5.14, Q5.17)

- **Limits** are set on the cards' menus, or written in the `.mod` panel; either way
  they bound drags.
- **Links** are written in the `.mod` panel; the tab reads them, shows a linked
  parameter on its card and locks it, and a drag moves what is linked to it. There is
  no link dialog: S15's link editor and its templates are after v1.
- **Interlopers (S14):** "Add interloper…" in a panel's right-click menu adds a line
  to the Lyα forest group (`1Ly_a` or `1H_IB`) at the clicked wavelength, or a
  component to another system (chosen from the line IDs there); fixed or free; one
  step.

### Excluding, including and removing (S27, D23; Q5.11, Q5.15, Q5.18)

- **Exclude** comments out the ion's line in the component (a `#` in front of it),
  after a preview of the lines it changes, as one step; the line keeps its values in
  the `.mod`.
  - The lines that use a label only the excluded line carries (a link, a
    `fix param` or `lim param` line, a ratio line) are commented out with it, each
    ending with a note such as `# excluded with 1H_I za3`. ALIS would otherwise stop
    on them.
  - A commented-out voigt line that carries a component's redshift label is that
    component's excluded line for its ion, whether the dashboard or the user commented
    it out. The label ties it to its component, since ids follow the order in z.
  - A blinded line stays hidden when commented out, and the labels of excluded lines
    stay taken (`edit.used_labels`), so that a new parameter never takes them.
- **Include** removes the `#` from the line and from the lines excluded with it, as
  one step. Log N is as it was; z, b and T are rewritten to the component's current
  values, since they are tied and must agree; the specids become those of the ion's
  snips now.
- **The last ion:** excluding a component from its last ion asks first, then removes
  the component.
- **An ion given as a ratio** (D I as `2H_I/1H_I`) is one line for every component it
  applies to, so excluding it from one component comments it out for all of them; the
  preview says so, as removal does (`remove.plan_ion`).
- **Remove (✕)** removes a component from every ion, after a confirmation that lists
  every line it deletes (`remove.plan_component`, S27), its commented-out lines and
  their notes included. One step. It is the only way to remove a component.

### Imported models (Q0.3, D20, D22)

The notices of `project.notices` ("O I has a separate free z in each of its 6 lines";
"D I has its own b turb") are shown above the panels, each with its one-click fix
("Tie them to a1", "Tie D I to H I") and "Keep as imported". The Components marker
keeps its "!" while a notice stands (Stage 3).

### Cross-highlighting (S16)

- **From the tab:** a selected card, tick, ion or panel highlights its lines in the
  `.mod` panel (`project.lines_of`, `panel.select_items`).
- **From the `.mod` panel:** the cursor on a voigt line selects its ion and its
  component; on a `link` or `lim` line, the parameter it constrains.

### Markers, preferences and actions

- **Markers:** Stage 3's rules stand (notices; "the regions changed since the
  components were last edited"). One is added: "!" while an ion of a system has no
  snip of any of its transitions (it constrains nothing; Q5.7).
- **New preferences** (`preferences.py`): the velocity range (±200 km/s), the
  starting b (5 km/s), and of a line of the Lyα forest (20 km/s, Q5.22), the default
  temperature mode (0 K fixed), two columns to start with (Q5.21; never more
  columns than panels), the first guess when
  every transition is saturated (15.0) and with no data (12.0), the bounds of log N in
  a drag (8.0 to 23.0), and the live preview's rate.
- **New actions** (`actions.py`, scope "Components"): ↑ ↓ the ion, Add component, Add
  ion…, Add interloper….

### Not in this stage

- The best-fit model, the run's results and the initial/final toggle (Stage 6: D25,
  S18–S21); the pre-flight check (S17, Stage 6).
- S15's link editor and its templates (Q5.9), and the Plot tab (after v1).
- The stacked spectrum of a transition (Q4.20: with Orders mode, after v1).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage5_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests;
> run the `fast` batch before the stage closes.

**5.1 — The model of each snip (`profile.py`, no Qt; S13, D24, Q5.5).**
- The model of a snip as ALIS computes it (`model_func`, every pixel, the links
  applied through `load.load_tied`), from the project's text and from values given
  during a drag; divided by the continuum for drawing; χ² on the fitted pixels; what
  is cached between evaluations; a cap on the sub-pixels for drags, used if the
  measurement calls for it.
- **Check** (unit tests): the model equals ALIS's own (`model_eval.model_func` on the
  state `run_alis` builds) to 1e-10 for every model of `examples/`; a changed value
  changes only the snips it applies to; a linked parameter follows a dragged one as in
  ALIS; no hidden value is returned by anything but the curve (the gate's `strings`
  test); the evaluation of J1358p6522's H I snips is timed and logged, and the drag's
  method is chosen from it.

**5.2 — Components (`components.py`, no Qt; S11, S12, S14, D20–D23; Q5.6, Q5.7,
Q5.12, Q5.13, Q5.15–Q5.18).**
- Adding a component (labels, a line for every ion of the system, specids, b, the
  temperature mode, the first guess of log N); adding an ion to a system; values from
  a drag and their bounds; temperature modes, "Fixed at…" included; excluding and
  including; interlopers; the notices' fixes.
- **Check** (unit tests): each is one step that ALIS reads, and one undo restores the
  text byte for byte; the first guess recovers log N of a synthetic unsaturated line
  to 0.05 dex, sets aside an undetected weak transition, and gives 15.0 or the lower
  limit when every transition is saturated; excluding then including restores the
  text byte for byte when nothing else changed; an exclusion that leaves a link
  without its label comments the link out with it, and Include restores both; ✕
  deletes a component's commented-out lines; a new component never takes an excluded
  line's label; J1358p6522's two notices are fixed by their fixes.

**5.3 — The ion navigator (`qt/components.py`; S30, D37; Q5.7, Q5.14).**
- The groups (the primary system, the other systems, the Lyα forest), ions with their
  counts, isotopes inside their element, Add ion… and Add system…, and the keys.
- **Check** (`gui` tests): J1358p6522's navigator as in the mockup (H I with D I, O I,
  the Lyα forest group); Q1243p307's systems one group each; Add ion… lists O I first
  where O I 1302 lies in a Si II 1304 snip; ↑ ↓ only with the tab's focus.

**5.4 — The panels (D37, S13; Q5.2, Q5.3, Q5.8, Q5.14).**
- One panel per snip that holds the ion, of the chosen dataset or of every dataset
  (Q5.19); the Dataset menu; the column setting and the velocity range; the normalised data and model, the
  residuals, the ticks (excluded ones faint and dashed, other systems' locked), the fit
  regions; the Lyα forest group's panels; the Regions tab's Model switch.
- **Check** (`gui` tests): J1358p6522's H I in two columns (eight panels, scrolling);
  Q1243p307's O I 1302 in each of its three datasets and in "All datasets", a tick at
  the same velocity in each; the model drawn equals `profile.py`'s; the Regions tab's
  Model switch draws it.

**5.5 — Adding, selecting and dragging (S11–S13; Q5.4, Q5.12, Q5.15, Q5.17).**
- "+ Add component" and its click; selecting a tick (its card, its `.mod` lines, its
  handle); the tick and handle drags, with Shift; the bounds; the live preview; one
  step per drag; the change in χ² for a blinded component.
- **Check** (`gui` tests): on a new `OI_SiII.dat` project, one click on O I 1302 adds
  one component with an O I and a Si II line, each log N within 0.1 dex of the
  example's best fit; a click with the button not armed selects or clears, and never
  adds; a drag is one step, and one undo restores the text; a drag stops at a `lim`
  and at b_turb = 0.5 km/s; dragging a blinded component shows no value.

**5.6 — The cards (S11, D20–D22; Q5.9, Q5.14, Q5.15).**
- The groups of cards; the header, Exclude/Include, z and b with their badges and
  limits, linked parameters, the temperature menu with "Fixed at…", the ions' log N
  and the first guess's notes, isotopes, ✕ (confirmed), Blind….
- **Check** (`gui` tests): every badge, menu and button is one step; an excluded card
  is greyed while its Include button stays enabled; excluding the last ion asks first;
  on `examples/blind`, no hidden value on a card; J1358p6522's cards as in the mockup.

**5.7 — Review of the Components tab (Q5.1).**
- Publish the tab's screenshots beside its mockups, on a new page, for RJC's comments,
  and apply them before the blends, links and notices are built.

**5.8 — Blends, links and notices (S14, S15, Q0.3; Q5.9, Q5.14).**
- The interlopers; links read from the `.mod` and shown (the badge, the tooltip, the
  locked parameter, drags that move what is linked); the notices with their fixes.
- **Check** (`gui` tests): an interloper added in a panel's menu is drawn, joins the
  Lyα forest group, and is one step; a parameter linked in the `.mod` panel shows its
  badge and expression and has no handle, and dragging what it is linked to moves it;
  J1358p6522's notices fixed in one click each, and their marker's "!" gone.

**5.9 — Cross-highlighting, markers, actions (S16, F12).**
- Both directions of cross-highlighting; the new marker rule; the actions and the
  preferences.
- **Check** (`gui` tests): the cursor on a voigt line selects its component's card; a
  selected card or tick highlights its lines; the new "!" and its tooltip.

**5.10 — From a spectrum to a fit (end to end).**
- The `fast` test of Task 4.12, with the absorption added on the Components tab
  instead of `edit.add_ion`: one click on O I 1302 (one component with an O I and a
  Si II line, sharing z and b, D20), then its temperature set to 8000 K with "Fixed
  at…", as the reference has it. `run_alis project.model` then fits, and the column
  densities agree with `fit_spectra.mod.out.reference` within their errors.

**5.11 — Close the stage.**
- Publish the screenshots of the tab beside the mockups for RJC's review, and apply
  the review.
- `doc/ALIS_workflow.md`: a section on the Components tab. Also update `CHANGELOG.md`,
  `tests/README.md` and the `gui-component` skill.
- Run `test-coverage` on the new modules, and the `unit`, `gui` and `fast` batches.
- Record in this document what Stage 6 receives, add the stage's decisions to
  `ALIS_v2_dashboard_prompts.md`, and update the stage table in `dashboard_stage0.md`
  (S15's link editor moves to after v1, Q5.9).

## Status (2026-10-15)

*Written by Claude in Prompt 4, at Task 5.11, and brought up to date in Prompt 5. The
details are in `claude_prompts/logs/dashboard_stage5_log.md`.*

**Stage 5 is closed** (Prompt 5 of this document). Tasks 5.1–5.11 are done. The tab
was reviewed halfway (Q5.1, Q5.19–Q5.22, and RJC's comment on the columns) and at the
close (Q5.23: a click selects the nearest component, a drag edits the component
nearest where it starts, N and B hold one of log N and b, a right drag pans); every
comment is applied, and draft 4 of the page shows the stage's final state:
https://claude.ai/artifact/7hNzToHBQmpTPXa2RTpoQr. The stage's decisions are D60–D68
of `ALIS_v2_dashboard_prompts.md`; S15's `link` editor moves to after v1 (Q5.9, the
stage table of `dashboard_stage0.md`). Stage 6's document is to be written.
- **Tests.** 3 new test files (`test_dashboard_profile.py`, 36 tests;
  `test_dashboard_components.py`, 31; `test_dashboard_qt_components.py`, 27 `gui`),
  and the end-to-end fit now through the Components tab. Coverage of the new modules
  over their tests: 91% (`components` 92%, `profile` 91%, `qt/components` 91%); two
  tests were added for what it found untested (a card's Tie, Untie, Limits… and
  Clear limits; a click in the Lyα forest's panels).
- **The model drawn** is ALIS's to 1e-10 on every example, J1358p6522 and Q1243p307;
  J1358p6522's eight H I snips evaluate in 30 ms, so a drag needs no sub-pixel cap.
- **The end-to-end fit** (5.10): one click on O I 1302, then 8000 K with Fixed at…,
  fits to O I 13.984 ± 0.015 and Si II 13.044 ± 0.049 (the reference: 13.985 ± 0.016
  and 13.041 ± 0.050).
- **The batches:** after 5.8, `unit` 1879 passed and `gui` 139 passed; after 5.9,
  `unit` 1881 and `gui` 142 passed. At Task 5.11: `unit` 1881 passed, 0 failed;
  `gui` 144 passed; `fast` 112 passed, 0 failed (13 min 42 s). After Q5.23
  (Prompt 5), at the close: `unit` 1881 passed, 0 failed; `gui` 146 passed;
  `fast` 112 passed, 0 failed (13 min 31 s).
- **ALIS outside `alis/dashboard/`** is unchanged in this stage. Inside it, besides
  the new modules: `edit.used_labels` counts the labels of commented-out lines,
  `remove.plan_component` deletes a component's excluded lines,
  `text.ModelText.comments_in`, `project` (excluded lines, notices kept as imported,
  `Notice.fix_text`), `actions` (the tab's actions; a key may serve two tabs),
  `markers` (the "!" of an ion in no snip), `preferences` (`components.*`), and in
  `qt/`: `plots` (model, residuals, ticks, handle), `regions` (the Model switch),
  `modpanel` (`select_items(move=False)`, `cursor_line`), `dialogs` (`IonDialog`).

### What Stage 6 receives

The Fit tab of Stage 6 is built in the frame of `qt/tabs.py`, as the `gui-component`
skill describes (it now documents the model, ticks and drags), on what Stage 5 adds:
- **The model of each snip** (`profile.py`): `of(project)` (cached on the project's
  state) and `Profile.evaluate(values=None, snips=None)`, a `Curve` per snip (model,
  continuum, normalised data, residuals, χ² on the fitted pixels), equal to
  `model_func`. Given the best-fit parameters (`values`), it draws the best fit, for
  the initial/final toggle (D25), and the Regions tab's Model switch can show it
  (Q5.8: "Stage 6 adds the best fit").
- **The spectrum view** (`qt/plots.py`): `set_model`, `ResidualView`, `set_ticks`
  (kinds `own`, `echo`, `excluded`, `isotope`, `other`, `forest`), and `set_edits`
  (a left drag gives `dragged`, a right drag pans).
- **The components** (`components.py`): `param_state` (free, fixed, tied, linked),
  `linked_values` and `shown` (what links compute), `draggable`, `bounds`,
  `unsnipped_ions` and the project's notices, which a pre-flight check (S17) can
  list; `at_line` for what a `.mod` line describes.
- **The end-to-end test** (`test_dashboard_qt_end_to_end.py`) takes a spectrum to a
  model on the three tabs, then fits it with `run_alis`; the Fit tab can replace that
  last step.

## Skills to use for this stage

- `gui-component`: the Components tab, wired to the project, the history, the
  validator and the gate; it documents the spectrum view (Stage 4).
- `gui-dev`: driving the window off-screen and taking the screenshots at 1440×900.
- `atomic-data`: the transitions of an ion, their f-values and the isotopes.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task; the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of `profile.py` and `components.py`.
- `check-fit`: the end-to-end fit of Task 5.10 against its reference.
- `profile-fit`: the cost of one model evaluation (Task 5.1, Q5.5).
- Claude Code's `artifact-design` skill: the review pages (5.7, 5.11).

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`:
  - D7, D16, D20–D24, D37, D43, D44, D49, D52–D59;
  - QF.8, QF.10, QF.20, QF.25 and QF.28;
  - F5, F8, F12, S11–S16, S27 and S30.
- `claude_prompts/dashboard_stage0.md`: Q0.3 (imported models that break D20 or D22),
  and the reviews of the Components drafts (Task 0.10).
- `claude_prompts/dashboard_stage2.md`: Q2.3 (labels), Q2.4 (an imported fit's
  structure); the project's components, notices and removal plans.
- `claude_prompts/dashboard_stage4.md`: "What Stage 5 receives", Q4.12 (what the
  spectra show), Q4.19–Q4.21.
- `claude_prompts/deferred_work.md` §7.2 (the best-fit model on the Regions tab).
- `doc/ALIS_workflow.md`, sections 2.3 (labels), 2.5 (the absorption block) and 9
  (the Voigt parameters).
- **The code:**
  - `alis/dashboard/` (the project model: `project.Component`, `System`,
    `notices`; `edit.new_component`, `add_ion`, `fix`, `free`, `tie`, `set_limits`,
    `set_command`; `remove.plan_component`, `plan_ion`; `blinding.Gate`; Stage 4's
    `snips.py`, `lines.py` and `continuum.py`);
  - `alis/dashboard/qt/tabs.py` (the frame of `ComponentsTab`) and `qt/plots.py`;
  - `alis/model_eval.py` (`model_func`, `FitState`) and `alis/main.py`
    (`ClassMain`), for the model as ALIS computes it;
  - `alis/functions/voigt.py` and `vfwhm.py` (the profile and the convolution).
- **The mockups:** `doc/dashboard/mockups/components.html` and `components_hd.html`
  (built by `build_mockups.py`), published at
  https://claude.ai/artifact/J43w9ERNESDo9hez9o918B.
- **The data:** `examples/metal_line_abs/` (a source spectrum and its model; the
  end-to-end fit), `context/fitting_examples/VMP_DLA/J1358p6522/` (H I with D I, O I,
  a Lyα forest; its notices), `context/fitting_examples/DH/Q1243p307` (three
  datasets) and `examples/blind` (a hidden line).

## Queries

*Raised by Claude on 2026-10-08, while writing this document. Each gives Claude's lean,
and the Design section and the tasks follow the leans.*

**Q5.1 — A review halfway through the stage.** As in Stage 4 (Q4.1), I propose to
publish the tab's screenshots for your comments once the navigator, the panels, the
dragging and the cards are built (Task 5.7), before the constraints and blends.

My lean: yes, a review at Task 5.7, and another at the close.

**Response:** Yes, I agree with this plan.

**Q5.2 — Panels with several datasets.** D37 shows every transition of an ion as one
panel each. With several datasets (Q1243p307's HIRES, PROCHASKA and KIRKMAN), a
transition is in each:
- **(a)** one panel per transition and dataset ("O I 1302.2 · HIRES"), ordered by
  transition, then dataset; or
- **(b)** one panel per transition, showing a dataset chosen in a menu above the
  panels.

My lean: (a), so that a component is checked against every dataset at once.

**Response:** I think (b) makes this most similar to the other parts of the dashboard,
and reduces the risk of overwhelming the user with too many panels. I lean towards (b) for clarity.
If there's an option to choose the dataset that is being displayed, then one option could be
"Show all datasets" which would then display all datasets. This would give the user flexibility
to view either a single dataset or all datasets at once.

**Q5.3 — What the panels show.** I propose the data divided by the continuum, with
the model divided by it (the continuum is the Regions tab's), and the residuals,
(data − model)/σ, in a strip underneath; the model is the one that will be fitted
(the starting values), not a run's best fit, which Stage 6 shows (D25).

My lean: as proposed.

**Response:** I agree that we should use the continuum that was defined in the regions tab. The data should be divided by the continuum model, and then the absorption components should be defined relative to 1. Just to be clear, I don't think we should divide the data by the absorption (i.e. the components) model. What we really need to show is the continuum normalised data, with a components model overplotted, and the residuals (data - model)/sigma in a strip underneath. This will allow the user to see how well the model fits the data, and if new components are needed.

**Q5.4 — The drag gestures.** Your description of step (4) has a click giving the
redshift, a horizontal drag the Doppler parameter and a vertical drag the column
density, per ion; D21 has the horizontal drag set b_turb with T held. I propose two
handles per component in the selected ion's panels:
- **(a)** a tick at the top of each panel, at the component's velocity: a horizontal
  drag moves the component (z, shared by its ions, D20);
- **(b)** a handle at the line's core: a vertical drag sets log N of this ion, a
  horizontal drag sets b_turb (T held).

My lean: as proposed. Holding Shift while dragging could make a drag finer (one tenth
of the movement).

**Response:** Let's implement your suggestion so I can test how it feels when working with the data. I'm concerned that the log N dragging will be difficult when the line is saturated.

**Q5.5 — The live preview's speed.** The model drawn during a drag is ALIS's own
(`model_eval.model_func`), evaluated for the snips of the ion at each new handle
position. On J1358p6522's eight H I snips that may take tens of milliseconds, or more
with sub-pixels and several datasets.
- **(a)** Evaluate with ALIS at every position, throttled to the rate the machine
  allows (the curve lags a little behind a fast drag); or
- **(b)** draw a quicker approximation during the drag (the Voigt profile without
  ALIS's sub-pixels), and ALIS's own model when the drag ends.

My lean: (a): the curve drawn is always the one ALIS will fit. Task 5.1 measures the
cost; if it is too slow, I will raise it again before Task 5.5.

**Response:** Let's go with (a) for now, and we can revisit if performance becomes an issue. This is just a quick initial guess, after all, so speed and response to the user is most important. I suspect option (b) might be better, but with a small number of subpixels (e.g. 5).

**Q5.6 — A new component's starting values.** I propose b = 10 km/s (a preference),
the temperature mode "0 K, fixed" (as the mockups show; a preference), and the AOD
guess over ±(b + FWHM) around the clicked velocity, from the weakest unsaturated
transition of each ion.

My lean: as proposed.

(a) I propose a default value, b = 5 km/s (which can be set as a preference). For the AOD,
perhaps we can simply use the AOD of the clicked transition. Due to line blending, I propose that we only use +/-0.5 * FWHM_line, this should contain ~76% of the column density, so we can scale the column density by a factor of 1/0.76 to get the starting value. Note that FWHM_line represents the FWHM of the instrument added in quadrature with the FWHM of the instrument, i.e. FWHM_line = sqrt(FWHM_instrument^2 + FWHM_voigt^2), and FWHM_voigt = 2 * sqrt(ln(2)) * b.

**Q5.7 — Adding an ion.** D20 shares a component across every ion of its system. Add
ion… would give the new ion a line in every component of the system, its log N
guessed by AOD. Should an ion with no snip be offered (its lines constrain nothing
until a snip is cut)?

My lean: only ions with a snip of one of their transitions are offered; the Regions
tab cuts the snips first.

**Response:** Sometimes, an ion may wish to be included in a snip of another transition. For example, the O I 1302 line may wish to be included in the same snip as the Si II 1304 line. In this case, the user may want to add O I even if there is no snip of O I. So I think we should allow the user to add an ion even if there is no snip of that ion.

**Q5.8 — The model on the Regions tab** (`deferred_work.md` §7.2). Once `profile.py`
exists, the Regions tab could draw the model too (a "Model" switch beside Normalised,
off by default), which helps when moving a snip's edges or regions around a blend.

My lean: yes, the switch, with the starting model in Stage 5 and the best fit added
in Stage 6.

**Response:** I agree with this idea.

**Q5.9 — The link editor.** S15's link editor: a dialog that writes one `link` line,
with the labels autocompleted from the model and its dependencies checked before it
is written. The templates (D/H through `ion=2H_I/1H_I`, D/H through `variable` and
`link`, ³He/⁴He) are after v1 (the stage table).

My lean: the dialog in Stage 5, the templates after v1.

**Response:** It will be difficult to set up link templates. For now, let's only allow the user to set the links manually in the mod file editor. The dashboard should still be able to read and understanding these links, but the user will have to set them manually. We can revisit this after v1.

**Q5.10 — Generic absorbers.** J1358p6522 has 57 Lyα forest components in 11 snips.
I propose to list them in the navigator as one group per system ("Lyα forest · 57
components in 11 snips"), showing the snips that hold them as panels, with the same
cards; S14's interlopers add to this group.

My lean: as proposed.

**Response:** I agree. I think there should be a set of component cards for each system identified. For example, there are a set of component cards for the primary absorber. There are also a separate set of component cards for each generic absorber identified (usually, a generic absorber will have 2 or more transitions identified). Then, there are a host of Lya forest lines that could appear anywhere in the primary absorber transitions or generic absorber transitions. Those component cards can be grouped together in a separate set of component cards.

**Q5.11 — The tab's keys.** ↑ ↓ for the previous or next ion (as the transitions on
the Regions tab), and Delete to remove the selected component (after its preview).

My lean: as proposed.

**Response:** We need to allow that a component may appear in one ion but not another. Therefore, the component cards set the total allowed components for the system, but each ion can include or exclude components from the listed component cards. So, by default, a component is included in all ions of the system, but the user can exclude a component from a specific ion. To delete a component (so that it is not considered for any ions/transitions) should be reserved as a feature only in the component cards. If a component card is deleted, a warning message should be presented to the user to confirm that they wish to delete the component from all ions/transitions. If a component is excluded from a specific ion, then the component card should be greyed out for that ion, and the user should be able to re-include it if desired.

*Raised by Claude on 2026-10-14 (Prompt 1), after reading the responses above. How I
read each response comes first, then four new queries, each with a lean. The Design
section and the tasks still read as they were written, and will be brought in line
once these are answered.*

**Notes on the responses.**
- **Q5.1.** A review at Task 5.7, and another at the close.
- **Q5.2.** (b), with your "All datasets":
  - a **Dataset** menu above the panels lists each dataset that covers one of the
    ion's transitions, then "All datasets". The reference dataset is shown first;
  - one panel per transition of the chosen dataset; "All datasets" gives one panel
    per transition and dataset, ordered by transition, then dataset;
  - one setting for the whole tab (kept in `ui/view.json`), and no menu when the
    project has one dataset;
  - every dataset is drawn in the model's frame (its shift removed, as the Regions tab
    moves regions between datasets), so a component's tick is at the same velocity in
    each;
  - a transition that the chosen dataset does not cover is left out, and the panels
    say so ("O I 1039.2: not in KIRKMAN").
- **Q5.3.** As you describe:
  - the data are divided by the Regions tab's continuum;
  - over them, the model that will be fitted is divided by the same continuum, so the
    absorption is drawn relative to 1. The data are never divided by the absorption;
  - the residuals, (data − model)/σ, are in a strip underneath.
- **Q5.4.** Built as proposed, with Shift for finer drags, so that you can try it. Your
  concern about log N on a saturated line is Q5.12.
- **Q5.5.** (a): ALIS's own model at every position, throttled. Task 5.1 measures the
  cost on J1358p6522's H I.
  - If it is too slow, the drag can use ALIS's own model with fewer sub-pixels, as you
    suggest. ALIS chooses 5 (`run nsubmin`) to 21 (`run nsubmax`) sub-pixels per pixel
    from the narrowest line in the snip.
  - Capping them at 5 during a drag, then evaluating with the model's own setting when
    the drag ends, keeps ALIS's code and atomic data. I would raise it with the
    measurement.
- **Q5.6.** b = 5 km/s (a preference). The log N guess follows your formula:
  - the apparent optical depth of the clicked transition, over ±0.5 FWHM_line around
    the clicked velocity, divided by 0.76 (erf(√ln 2) = 0.761, the fraction of a
    Gaussian within ±½ FWHM);
  - tried on `examples/metal_line_abs`, with b = 5 km/s and its 7 km/s FWHM: O I 1302
    gives 13.95 and Si II 1304 gives 12.98. The true values are 14.0 and 13.0, and the
    best fit is 13.985 and 13.041.

  Some details remain: Q5.13.
- **Q5.7.** An ion can be added without a snip of its own.
  - **Add ion…** first lists the ions with a transition inside any snip, whatever the
    snip was cut for. This covers your O I 1302 in the Si II 1304 snip, and O I 925
    inside J1358p6522's Ly7 snip. Every other ion of the atomic data follows.
  - **Its specids:** the new line takes the specids of the snips that hold its
    transitions (Stage 4's rule, Q4.3).
  - **An ion whose transitions fall in no snip** constrains nothing. ALIS needs a
    specid on a line when there are several snips, so its line lists the system's
    snips. The Components tab shows "!" (the Design's new marker rule) until a snip
    covers one of its transitions.
- **Q5.8.** A "Model" switch on the Regions tab, off by default, draws the starting
  model from `profile.py`. It is added to Task 5.4; Stage 6 adds the best fit.
- **Q5.9.** No link dialog: links are written in the `.mod` panel, and the tab reads
  them.
  - A linked parameter has its own badge on its card, with its expression in the
    tooltip (through the gate).
  - It has no handle and cannot be typed, because the link computes it.
  - Dragging a parameter that others are linked to moves them too, as ALIS computes
    them.
  - Task 5.8 loses the dialog. S15's link editor moves to after v1, with the
    templates (the stage table is updated at the close).
- **Q5.10 and Q5.11.** Q5.14 and Q5.15 below.
- **A correction to Task 5.10.** It says "a click on O I 1302 and on Si II 1304".
  - With D20, one click on O I 1302 adds a component with a line for both O I and
    Si II, since both ions have a transition in the merged snip. Si II's log N is
    guessed from its 1304 line. A second click would add a second component.
  - The test makes one click, then sets the temperature on the card to the
    reference's, 8000 K fixed. The temperature menu therefore also offers "Fixed at…"
    for a value that is not a preset, as imported models have.
- **Found while reading the code.** The Design says that `profile.py` takes the
  emission and absorption parts from `model_func(..., getemab=True)`.
  - Since refactor Stage 3.5.4, `getemab` returns only the cache of components
    (`model_eval.py:617`).
  - Nothing in ALIS needs to change. The panels draw the whole model divided by the
    continuum, which `model_func(output=1)` and `continuum.evaluate` give.
  - The mockups mark the components with ticks, not with curves of their own, so
    Task 5.1's "absorption of one component or ion alone" is not needed.

**Q5.12 — Dragging log N when a line is saturated (your Q5.4 response).** If the
handle's height set log N through the depth of the line it is dragged in, a saturated
line would hardly respond: J1358p6522's Lyβ core stays at zero whatever log N. I
propose that a drag changes log N by a fixed amount per distance, whatever the line's
shape:
- a vertical drag changes log N by 2 dex over the panel's height (downwards deepens);
  a horizontal drag changes b_turb by the velocity moved (to the right widens); Shift
  makes either ten times finer;
- the first few pixels of a drag decide which of the two it changes, so that one drag
  never changes both by accident;
- while dragging, the handle follows the cursor and every panel of the ion shows the
  model. The panel's corner shows log N and b through the gate (only Δχ² for a blinded
  component). On a saturated line the change shows in the wings, and in the ion's
  weaker transitions;
- the card's values can still be typed, for an exact number.

My lean: as proposed.

**Response:** Overall this sounds good. Dragging should allow both N and b to change simultaneously. Holding shift should allow for only N or b to change. Finer control is not required at this stage. We would need to be careful about the b value to ensure that a negative velocity does not result in a negative b value. Perhaps the hortizontal drag should be the position of the handle relative to the centre of the line, so that the b value is always positive. When the b value is zero or close to zero, we should make sure that the minimum value is set to the default minimum value specified in voigt.py. Similarly, the log N value should not be allowed to go below the minimum value specified in voigt.py. The log N value should also not be allowed to go above the maximum value specified in voigt.py.

**Q5.13 — The first guess of log N (your Q5.6 response).** Four points to settle:
- **(a) A saturated clicked transition.** When the flux within the window falls to
  within 3σ of zero, the clicked transition gives only a lower limit. I propose to use
  the ion's strongest transition that is not saturated there instead. When every
  transition is saturated, the clicked one is used, and the card says that the value
  is a lower limit.
- **(b) The component's other ions.** A new component has a line for every ion of the
  system (D20). The clicked transition gives the clicked ion's log N. Each other ion's
  comes from its own strongest transition that is not saturated at that velocity, with
  the same window. The strongest gives the most signal; S12 said "the weakest
  unsaturated".
- **(c) Absorption already in the model.** A click on absorption that an existing
  component already accounts for (a blend, or a second component beside the first)
  would count that absorption again. I propose to measure the optical depth against
  the current model rather than the continuum: τ_a = ln(M/F), where M is the model
  without the new component. The guess then covers only what the model does not yet
  explain.
- **(d) No data at that velocity.** An ion with no transition on the data at the
  component's velocity starts at log N = 12.0 (a preference), and its card says "no
  data at this velocity".

In each, FWHM_line uses the ion's total b (b_turb, and the thermal part of the
temperature mode for the ion's mass) and the FWHM of the clicked dataset (converted to
km/s at the line's wavelength when the resolution is not given in km/s).

My lean: (a) to (d) as proposed.

**Response:** (a) I agree with this approach. If there are no unsaturated transitions, then we should use a default value for saturated transitions log N = 15.0. (b) If a component is added for one ion, it would be good to automatically add this component for all other ions. The following standard procedure should be applied to all ions: First, a column density is estimated for this component using all available transitions (that are covered by any snip) of this ion. The starting column density will correspond to the lowest log N value derived in this way. (c) I agree with this approach. (d) I agree with this approach.

**Q5.14 — Groups of component cards (your Q5.10 response).** In the code, a "generic
absorber" is a line with no ion of its own (`1Ly_a`, `1H_IB`), and a "system" is a
redshift with identified ions (the Data tab's systems, D15). I read your three kinds of
cards as:
- **(i) the primary system:** its components, one card each;
- **(ii) each other system** (your "generic absorber identified", usually with two or
  more transitions, such as Q1243p307's systems at z = 2.05 to 2.44): its own
  components, one card each, and its own ions in the navigator;
- **(iii) one "Lyα forest" group for the whole project:** every `1Ly_a` and `1H_IB`
  line, wherever it falls, one card each (the wavelength of its Lyα, log N and b).
  Selecting the group shows a panel for each snip that holds one of its lines, in that
  snip's own velocity frame (its transition at its system's z). The lines' ticks are
  where their Lyα falls.

"Add interloper…", in a panel's right-click menu, then adds a line to (iii) at the
clicked wavelength, or a component to a system of (ii).

My lean: as read.

**Response:** I agree with your understanding, and I agree that we should implement it as you mention. You are correct that I meant  system (instead of generic absorber).

**Q5.15 — Including and excluding a component in an ion (your Q5.11 response).** In the
`.mod`, a component is the voigt lines that share its redshift label, one per ion, so
"c3 is included in O I" means that c3 has an O I line. I propose:
- **(a) Exclude** removes that ion's line from the component, as one step (previewed
  when something depends on it, such as a link or a ratio). **Include** adds it back,
  with a new first guess of log N (Q5.13).
  - The alternative is to comment the line out, so that Include restores its old
    values. The `.mod` would then fill with commented lines that ALIS skips, and the
    old value is rarely the right one once the other components have changed.
- **(b) Where:**
  - on the card, for the ion selected in the navigator: "Exclude from O I" on the
    ion's row; an excluded component's card is greyed, with "Include in O I";
  - in a panel, in the right-click menu of a component's tick. An excluded
    component's tick is drawn faint and dashed in the ion's panels, so that you can
    see where it is.
- **(c) Removing a component** from every ion is done only with the card's ✕, after a
  confirmation that names every line it removes, dependents included (S27). The
  Delete key does not remove components, unlike the Design's proposal (Q5.11); ↑ ↓
  stay.
- **(d) The last ion.** A component excluded from its last ion has no line left, so it
  is removed: Exclude then asks first, as ✕ does.

My lean: (a) remove the line; (b), (c) and (d) as proposed.

**Response:** (a) I like the idea of being able to preview the things that it depends on, but I think it might be better to comment it out. If the user wishes to remove the line from the .mod file, they can do so manually. This way, the user can always see what the previous values were and can revert back to them if needed. (b) I like the idea of having the option to exclude/include from both the card and the panel. For the card, there should be one button that says "Exclude from O I" or "Include in O I" depending on the current state. If excluded, the card should be greyed out (but not the button that says "Include in O I"; this button should appear active so users know they can click on it). For the panel, users should have the ability to select a component, and this would highlight that component in the cards menu, allowing users to include/exclude it from there. By selecting the component, this also gives users the opportunity to adjust/edit the components starting log N, z, and b values. Excluded components should have tick marks that are drawn faint and dashed in the ion's panels. Note that the ability to select components should not be confused with adding components. If users wish to add a new component (that doesn't already have a component card) they will need to select the "Add component" button and then click on the panel to set the redshift location and automatically select the first guesses of log N and b. (c) I agree with this. (d) Yes, a component excluded from its last ion should be alerted to the user, and ask if they wish to remove this component to confirm this action.

*Raised by Claude on 2026-10-14 (Prompt 2), after reading the responses to Q5.12 to
Q5.15. How I read them comes first, then three new queries, each with a lean. The
Design section and the tasks will be brought in line once these are answered.*

**Notes on the responses.**
- **Q5.12.** One handle, on the selected component (Q5.15(b)), drawn on the ion's line
  one b to the side of the component's velocity.
  - A drag changes log N and b together. Log N changes by 2 dex over the panel's
    height (downwards deepens). b is the handle's distance from the component's
    velocity, on either side, so it cannot go negative.
  - With Shift, only the one the drag starts along changes (its first few pixels
    decide). There is no finer mode.
  - The handle gives the line's total b. b_turb follows for the temperature mode,
    b_turb = √(b² − b_th²); in the pure thermal mode the handle sets T instead.
  - b_turb stops at 0.5 km/s, voigt.py's lower limit, and T at 0. A `lim` on the
    parameter also bounds its drag. voigt.py sets no limits on log N: Q5.17.
  - The tick still moves the component's z.
- **Q5.13.** (a) and (b) make one rule, applied to every ion of a new component,
  the clicked one included:
  - each of the ion's transitions inside any snip, in each dataset that covers it,
    gives an estimate over its own window (with that dataset's FWHM), measured
    against the current model (c);
  - saturated estimates (flux within 3σ of zero) are set aside. The start is the
    lowest of the others, since a blend only adds absorption;
  - when every one is saturated, the start is log N = 15.0 (a preference). An ion
    with no data at the velocity starts at 12.0, and its card says "no data at this
    velocity" (d);
  - the clicked transition gives only the velocity.

  How noise in a weak transition enters "the lowest" is Q5.16.
- **Q5.14.** As read: the primary system, each other system, and one Lyα forest
  group. "System" means an identified absorber, in the code and in the tab.
- **Q5.15.**
  - **(a) Exclude** comments out the ion's line in the component (a `#` in front of
    it), after a preview of the lines it changes. The line keeps its values in the
    `.mod`.
    - A commented-out voigt line that carries a component's redshift label is that
      component's excluded line for its ion, whether the dashboard or you commented
      it out. Component ids (a1, a2…) follow the order in z and change when a
      component is added, so the label is what ties the line to its component.
    - **Include** removes the `#`. Log N is as it was. z, b and T are rewritten to
      the component's current values, since they are tied and must agree, and the
      specids become those of the ion's snips now.
    - What else an exclusion comments out, and what ✕ does with commented-out lines,
      is Q5.18. A blinded line stays hidden when commented out (the bundle already
      hides commented-out lines with `blind=True`).
  - **An ion the component never had** (one added to the system later) gets the same
    greyed card with "Include in O I", which adds a line with a first guess (Q5.13).
    This replaces the Design's "O I not in this component · + Add". **Add ion…** adds
    the ion to every component of its system, since a component is in every ion by
    default.
  - **(b) The card** has one button for the navigator's ion: "Exclude from O I" or
    "Include in O I". An excluded card is greyed, except for that button.
  - **Selecting.** A click on a component's tick selects it:
    - its card is highlighted and scrolled into view, and its lines are highlighted in
      the `.mod` panel (S16);
    - only the selected component has a handle. Dragging a tick moves its z, and
      selects it;
    - a click elsewhere in a panel clears the selection. It never adds a component;
    - an excluded component's tick, faint and dashed, can be selected (its card then
      offers Include), but it has no handle in that ion.
  - **Adding.** "+ Add component" arms one click. The next click in a panel adds the
    component at that velocity, with its first guesses. The cursor shows that the
    button is armed, and Esc cancels.
  - **(c) and (d)** As agreed. ✕ removes a component from every ion, after a
    confirmation. There is no Delete key, and ↑ ↓ stay. Excluding a component's last
    ion asks to remove the component.
  - **An ion given as a ratio** (D I as `2H_I/1H_I`) is one line for every component
    it applies to. Excluding it from one component therefore comments it out for all
    of them, and the preview says so, as removal does now (`remove.plan_ion`).

**Q5.16 — Which estimates "the lowest" takes (your Q5.13(b) response).** A transition
too weak to show the line gives an estimate that is mostly noise, and the lowest of
all the estimates would often pick it. I simulated 2000 noise draws of Si II at
log N = 13.0 and b = 5 km/s, with a 7 km/s FWHM and S/N 20 per 2.5 km/s pixel:
- 1304 and 1526 absorb 24% and 37% at the centre. They give 13.03 and 13.02, each
  within about ±0.05;
- 1260 absorbs 93%. That is within 3σ of zero, so it counts as saturated, and rightly:
  it gives 12.89;
- 1808 (f = 0.0021) absorbs 0.9%. In 36% of the draws it shows no net absorption, so
  it has no logarithm. In the rest it gives anything from 12.9 to 13.8 (the 16th to
  84th percentiles);
- the lowest of 1304, 1526 and 1808 (setting aside the draws with no logarithm) is at
  least 0.2 dex too low in 8% of the draws, and at least 0.5 dex too low in 4%.

I propose:
- **(a)** a transition counts only when its absorption is detected at 3σ or more
  (∫τ_a dv over the window, against its error from the data's σ) and is not
  saturated. The start is the lowest of those: 13.00 above;
- **(b)** when no transition of the ion is detected, the start is the lowest 3σ upper
  limit, which comes from the ion's most sensitive transition: 11.45 from 1260 above,
  if Si II showed nothing;
- **(c)** when every detected transition is saturated, the start is 15.0 or the
  highest of their lower limits, whichever is larger. Every Lyman line of a DLA's H I
  is saturated, and their lower limits can exceed 15.0.

My lean: (a), (b) and (c) as proposed.

**Response:** I agree with this approach.

**Q5.17 — The limits of log N in a drag (your Q5.12 response).** voigt.py limits
b_turb (at least 0.5 km/s) and T (at least 0), but not the column density: its
`_limited` is `[0, 0]` for `ColDens`. There is therefore no minimum or maximum of
log N to take from it. Two ways:
- **(a)** the dashboard bounds a drag to a range of its own, kept as a preference:
  log N from 8.0 to 23.0 (voigt.py's default column density is 8.1). A typed value is
  taken as typed, and ALIS's fits are unchanged;
- **(b)** voigt.py gains limits on `ColDens`. Every fit then respects them, and they
  must differ under `logN=False`, where the column density is not a logarithm.

In either case, a `lim` on the parameter also bounds its drag.

My lean: (a).

**Response:** I agree with option (a). Also note that users can change the limits of log N and the b value if they wish in the .mod file. If the .mod file has the limits altered, then the dashboard should read those limits and apply them to the drag. If the user wishes to change the limits of log N or b, they can do so in the .mod file.

**Q5.18 — What an exclusion leaves behind (your Q5.15(a) response).**
- **(a) Lines that use only the excluded line's labels.** Usually there are none: the
  component's z, b and T labels stay on its other ions' lines. But a link, a
  `fix param` or `lim param` line, or a ratio line can use a label that only the
  excluded line carries, such as a D/H link on H I's column density. ALIS stops on a
  link, `fix param` or `lim param` that names a label no line carries.
  - I propose to comment them out with the line, listed in the preview. Each ends
    with a note such as `# excluded with 1H_I za3`, so that Include brings them back
    together.
  - The alternative is to refuse the exclusion until you change them in the `.mod`
    panel.
- **(b) Removing a component (✕)** deletes its lines. I propose that it also deletes
  its commented-out lines, listed in the confirmation, since no component would own
  them any more, and a later component could take the same labels. The alternative is
  to keep them, with new components skipping any label used in a comment.

My lean: (a) comment them out with the line, and bring them back together; (b) delete
them with the component.

**Response:** (a) I agree with your lean, that we should comment them out with the line, and bring them back together. (b) I agree with this approach.

*Raised by Claude on 2026-10-14 (Prompt 3, at Task 5.7's review), with the review
page: https://claude.ai/artifact/7hNzToHBQmpTPXa2RTpoQr. Comments on the page are
welcome as well as responses here.*

**Q5.19 — Which transitions get a panel.** Every transition inside a snip would give
J1358p6522's H I 22 panels (Ly8 to the Lyman limit share one snip) and its O I 32 (weak
lines in the Lyman-series snips). I give a panel to each transition a snip was cut
for, and to any other inside a snip whose fλ is at least a tenth of the strongest of
those (of the strongest inside a snip, when none was cut for the ion, as O I 1302 in a
Si II 1304 snip); transitions within 50 km/s share a panel ("O I 988.8 ×2"). That gives
8 and 6, as the mockup has them, and the navigator counts the same.

My lean: as built.

**Response** (RJC, on the review page, 2026-10-15): "Transitions don't get a panel. A
snip gets a panel. For example, if O I is selected, all snips that contain an O I
absorption feature are shown. If 1 snip contains 5 O I transitions, there will be one
panel only for this snip, and this snip will contain all 5 transitions. In terms of
editing components, only one transition (the strongest in this snip) will have
editable components, but changes to this transition will be displayed for all
transitions (in this snip, and all other snips with O I)."

*Done by Claude on 2026-10-15 (Prompt 3):* one panel per snip that holds a transition
of the ion at the system's z (`components.ion_snips`), in the chosen dataset or in all
of them; zero velocity at the strongest transition (fλ) inside the snip, whose
component ticks are dragged; the components drawn again, thin and unlabelled, at the
snip's other transitions (clicking one selects its component); panels ordered by the
strength of each one's strongest transition, then by dataset. Titles: "O I 988.8 (+2) ·
HIRES" for a snip holding three O I lines, and the snip named when it was cut for
another transition ("O I 929.5 (+2) · Ly7 926.2 snip · …"), as two snips can hold the
same one. The navigator counts snips ("H I · 8 snips · 2 comp. · with D I"). The x
axes stay linked at ±200 km/s about each snip's strongest transition; the others in a
wide snip (Ly8 to the Lyman limit, 3400 km/s) are seen with − or the Range menu.
J1358p6522: H I 8 panels, O I 11 (its four snips and seven Lyman-series snips holding
O I lines).

**Q5.20 — The ions on a card.** Each card lists every ion of its system, with "not
here" for an ion the component has no line of (Q1243p307's cards list eight ions, most
"not here" for a component of H I alone). The alternative is to list only the ions the
component has, and the ion shown in the navigator.

My lean: every ion of the system, as built, so that a component's place in each ion is
read at a glance; I can make "not here" rows quieter if they crowd.

**Response** (RJC, on the review page, 2026-10-15): "It is good to be explicit that a
component is or is not included for a given ion. However, rather than have a new line
for each case that's "not here", perhaps we can have: Included: <comma separated list
of included ions> Excluded: <comma separated list of excluded ions>"

*Done by Claude on 2026-10-15 (Prompt 3):* a card has a log N row for each ion the
component is in (isotopes under their element): those rows are the included ions,
named, so no separate "Included:" line repeats them. The system's other ions, excluded
or never in the component, are on one line, "Excluded: D I, O I", the ion shown in
bold; a click on one includes it (one step, with a first guess for an ion it never
had). I can add an "Included:" line above the log N rows if you would like both.

**Q5.21 — Columns with the .mod panel open.** With the .mod panel beside the tab, three
columns of panels are narrow (the Q1243p307 screenshot). The preference starts at three
(Stage 3's default, from the mockup).

My lean: start at two.

**Response** (RJC, on the review page, 2026-10-15): "Agreed, let's make the default 2
panels."

*Done by Claude on 2026-10-15 (Prompt 3):* `components.columns` starts at 2, whether
or not the .mod panel is open; 1–4 can still be chosen, and the choice is kept with the
project.

**Response** (RJC, on the review page, 2026-10-15, on the "one click" screen): "When
there is only one snip but 2 columns requested, the dashboard should take:
`min(Nsnips, Ncolumns)`. In this case, even though 2 columns are the default and
requested, because there's only 1 snip this snip will be displayed as if columns=1. If
another snip was available for O I, then since columns=2, this would mean that two
columns would be adopted."

*Done by Claude on 2026-10-15 (Prompt 3):* the grid has min(panels, Columns) columns;
with Q5.19's rule a panel is a snip, so this is min(Nsnips, Ncolumns), the snips shown
(of the chosen dataset, or of all of them). The Columns box keeps the setting, which
takes effect again with a second panel.

**Q5.22 — A new line of the Lyα forest.** It starts at b = 20 km/s and T = 0 K fixed,
with the forest's own temperature label when its lines share one (`ZEROT` in
J1358p6522). b of a new component is 5 km/s (Q5.6), too narrow for the forest.

My lean: 20 km/s, as a preference.

**Response** (RJC, on the review page, 2026-10-15): "20 km/s is a good choice for a
default value."

*Done by Claude on 2026-10-15 (Prompt 3):* a new preference, `components.forest_b`
(20 km/s), in place of the constant; T = 0 K fixed and the forest's shared temperature
label are unchanged.

*Raised by Claude on 2026-10-15 (Prompt 4, at Task 5.11's review), from RJC's comment
on the review page (thread "a click on its tick").*

**Q5.23 — Selecting and dragging anywhere in a panel.** You asked that a click select
the component nearest the clicked velocity, and that a drag above the normalised
continuum (by the ticks) move z, while a drag below it changes log N and b. This
replaces the ticks' and the handle's drags of Q5.4 and Q5.12. To settle:
- **(a) A click** selects the component nearest in velocity, however far (its thin
  ticks at the snip's other transitions count; other systems' and the forest's
  locked lines do not). With + Add component armed, a click still adds one, and
  never otherwise (Q5.15(b)); the page's auto-reply said otherwise, wrongly. Esc
  clears the selection.
- **(b) The component a drag moves** is the one nearest where the drag starts, which
  it also selects. Values follow the drag's movement and never jump to the pointer:
  z by the horizontal movement; log N by 2 dex over the panel's height (down
  deepens); b by how far the pointer moves away from (or towards) the component.
- **(c) Below the continuum, log N and b change together**, as Q5.12 agreed, with
  Shift holding the drag to the direction it starts along; or each drag is held to
  one direction from its start, as the auto-reply suggested.
- **(d) The handle** is no longer needed: the selected component's tick is
  highlighted, and the panel's corner shows log N and b during a drag.
- **(e) Panning.** A left drag pans the panels today (✥ in the zoom bar). If a drag
  edits, panning needs another gesture: (i) a right drag pans (a right click still
  opens Add interloper's menu); (ii) ✥ switches the panels to panning and a new
  button back to editing; (iii) no panning in these panels (the Range menu, + and −,
  the wheel's zoom).
- **(f) Saturated cores** (the auto-reply's question): below the continuum covers
  most of a saturated panel, which is fine, since a drag anywhere there changes
  log N vertically and b horizontally; b need not be grabbed from the wings.

My lean: (a), (b) and (d) as written; (c) together, as agreed, with Shift; (e)(i).

**Response:** (a), (b) and (d) as written. (c) I agree that dragging in both directions should change both log N and b. I suspect the shift key together with whatever direction is moved first might be a little klunky. Therefore, let's make two separate keys, 'n' only allows log N to vary and keeps the Doppler parameter fixed at the current value, while holding 'b' only allows the Doppler parameter to vary and keeps log N fixed at the current value. For (e), A right drag should pan, and the wheel should be used for zooming in and out. (f) Sounds fine to me.

*Done by Claude on 2026-10-15 (Prompt 5):* no further questions. A click selects the
component nearest in velocity (`ComponentsTab.nearest`); Esc disarms, then clears the
selection. A drag (`SpectrumView.set_edits`, the `dragged` signal) moves the
component nearest where it starts, by its movement: above the continuum its z (its
ticks follow), below it log N and b together; `DragKeys`, an application event
filter, holds N and B while they are pressed over a panel (Qt reports only
modifiers' states), N keeping b and B keeping log N; pressing or letting go of one
mid-drag goes on from where the drag is, so nothing jumps. The handle, the ticks'
own drags and Shift are gone. A right drag pans the panels; the wheel zooms the
velocity axis, as before (D38). Draft 4 of the review page shows it.

## Prompts

1. Please read the `ALIS_v2_code_plan.md` and `ALIS_v2_dashboard_prompts.md` files, and the work carried out during stages 0-4; see their design documents (`dashboard_stage0.md`, `dashboard_stage1.md`, `dashboard_stage2.md`, `dashboard_stage3.md`, `dashboard_stage4.md`) and the logs (`dashboard_stage0_log.md`, `dashboard_stage1_log.md`,  `dashboard_stage2_log.md`,  `dashboard_stage3_log.md`, `dashboard_stage4_log.md`) to understand the work that has been implemented until now. Then, please review the ALIS code to understand the current state of ALIS. Finally, read this document, including my responses to your queries. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. If everything is clear, then please notify me when you are ready to start implementing Stage 5, so that I can confirm before beginning.

2. Please read the responses to your queries in the Queries section of this document, and provide any further queries you may have. If everything is clear, then please notify me when you are ready to start implementing Stage 5.

3. If you have any further queries, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order, with a pause at the agreed task so that I can provide design feedback on the html file. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

4. I have reviewed the progress, and together with comments made and queries addressed, we are no ready to continue to the next phase of this stage. If you have further queries before beginning, please ask them in the Queries section of this document, and I will provide responses. Once everything is clear about the implementation of this stage, please execute the tasks in numerical order, starting with Task 5.8. If you have questions during development, please pause the development, ask questions and I will respond (please log these questions and answers in the Queries section).

5. I have responded to query Q5.23. Please review my response, and if you have further questions, please ask them before continuing with development. Once everything is clear, please address these final fixes before closing Stage 5.

