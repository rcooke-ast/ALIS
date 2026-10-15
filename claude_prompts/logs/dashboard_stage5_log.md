# Dashboard Stage 5 log

The Components tab (step 4): the ion navigator, the panels of an ion's transitions on
a velocity axis with the model and residuals, adding and dragging components with a
live preview evaluated by ALIS, the component cards, blends, removal, imported models'
notices, and cross-highlighting. The plan is in `claude_prompts/dashboard_stage5.md`.

### 2026-10-14 (Prompt 1: reading, and queries Q5.12–Q5.15)

**Reading.** The code plan, the dashboard prompts (D1–D59, the QF queries), the Stage
0–4 documents and logs, this stage's document with RJC's responses to Q5.1–Q5.11,
`deferred_work.md` §7, `doc/ALIS_workflow.md` §2.3–2.8 and §9, and the code Stage 5
builds on: `alis/dashboard/project.py` (`Component`, `System`, the notices and their
fixes), `edit.py` (`new_component`, `add_ion`, `fix`/`free`/`tie`, limits),
`continuum.py` (`_load_with_alis`, `evaluate`), `snips.py` (`snips_of`, the specid rule
of SNIP), `lines.py` (`snip_transitions`), `remove.py` (`plan_component`, `plan_ion`),
`markers.py`, `blinding.py` (the `Gate`), `qt/tabs.py` (`ComponentsTab`) and
`qt/plots.py` (`SpectrumView`); ALIS's `model_eval.py` (`model_func`, `FitState`),
`main.py` (how `ClassMain` builds the state and applies links), `load.load_subpixels`,
and `functions/base.py` (the `specid` keyword); and the mockups' builders of the
navigator, the panels and the cards.

**Facts that shaped the notes and queries.**
- `model_func(..., getemab=True)` no longer returns the emission and absorption parts
  (dropped in refactor Stage 3.5.4; it returns `[None, None, compcache]`), so the
  Design's way of drawing them does not exist. The panels need only the whole model
  (`output=1`, every pixel of the snip) divided by the continuum, and the mockups mark
  components with ticks, not curves, so nothing in ALIS needs to change.
- Links reach the model through `load.load_tied` (`main.py:197-210`), applied to the
  parameters before an evaluation; a drag of a parameter that others are linked to
  must apply it too.
- ALIS sub-pixellates each snip with 5 (`run nsubmin`) to 21 (`run nsubmax`) sub-pixels
  per pixel, chosen from the narrowest line in it (`load_subpixels`), so a drag that
  narrows b can change the grid; capping it during a drag is the cheap fallback for
  Q5.5.
- With more than one snip, ALIS requires `specid=` on every model line
  (`base.py:475`), so an ion with no transition in any snip (Q5.7) still needs one.
- RJC's AOD rule (Q5.6) on `examples/metal_line_abs` (true log N 14.0 and 13.0; best
  fit 13.985 and 13.041), with b = 5 km/s and FWHM 7 km/s: O I 1302 gives 13.95, Si II
  1304 gives 12.98 (over ±0.5 FWHM_line, divided by 0.76). Wider windows give 13.93
  for O I; with b = 10 km/s, 14.02 and 13.12.
- With D20, one click adds a line for every ion of the system, so Task 5.10's "a
  click on O I 1302 and on Si II 1304" would make two components; the reference model
  has one (b tied, T = 8000 K fixed).

**Baseline.** `pytest -m unit`: 1814 passed, 120 skipped, 0 failed (90 s);
`pytest -m gui`: 117 passed (89 s). The same as at the close of Stage 4.

**The responses.** Recorded as "Notes on the responses" in the document:
- a review at 5.7 and at the close (Q5.1);
- one panel per transition of a chosen dataset, with "All datasets" (Q5.2);
- the continuum-normalised data with the normalised model and residuals (Q5.3);
- the drag gestures as proposed (Q5.4), the saturation concern being Q5.12;
- ALIS's model at every position, measured in Task 5.1, with fewer sub-pixels during
  a drag as the fallback (Q5.5);
- b = 5 km/s and RJC's AOD window (Q5.6), its details being Q5.13;
- ions addable without a snip of their own, those with a transition in any snip listed
  first (Q5.7);
- the Regions tab's Model switch, added to Task 5.4 (Q5.8);
- no link dialog; links read and shown, linked parameters locked (Q5.9);
- the correction to Task 5.10, and the `getemab` finding.

**New queries,** each with a lean:
- **Q5.12:** dragging log N by a fixed amount per distance, so that a saturated line
  can still be adjusted, with the axis of a drag chosen by its first pixels;
- **Q5.13:** the AOD guess when the clicked transition is saturated, for the
  component's other ions, against the current model (blends), and with no data;
- **Q5.14:** the three groups of cards (the primary system, each other system, one
  Lyα-forest group), and the terms "generic absorber" and "system";
- **Q5.15:** including and excluding a component in an ion (removing the line, or
  commenting it out), where it is done, the card's ✕ the only removal, and no Delete
  key.

### 2026-10-14 (Prompt 2: the responses to Q5.12–Q5.15, and Q5.16–Q5.18)

**The responses.** Recorded as "Notes on the responses" (Prompt 2) in the document:
- **Q5.12:** one drag changes log N (2 dex per panel height) and b together; Shift
  keeps it to one; no finer mode. b is the handle's distance from the component's
  velocity, so never negative; b_turb stops at voigt.py's 0.5 km/s. Only the selected
  component has a handle (from Q5.15(b)).
- **Q5.13:** one rule for every ion of a new component, the clicked one included: an
  estimate from each transition inside any snip, against the current model; saturated
  ones set aside; the lowest of the rest; 15.0 when all are saturated; 12.0 with no
  data.
- **Q5.14:** as read (three groups; "system" for an identified absorber).
- **Q5.15:** Exclude comments the ion's line out (`#`), Include restores it with its
  log N; one toggle button on the card; a click on a tick selects the component (card,
  `.mod` lines, handle) and never adds one; "+ Add component" arms one click; ✕ is the
  only removal, with a confirmation; excluding the last ion asks to remove the
  component.

**Checked for the new queries.**
- `voigt.py` limits `bturb` (`_limits` 0.5 km/s) and `temperature` (0), but not
  `ColDens` (`_limited` `[0, 0]`; its default is 8.1), so the log N bounds RJC asked
  for (Q5.12) have no source there.
- The `.mod` text keeps commented-out lines (`text.COMMENT`), so excluding by `#` needs
  no new parsing of the file, only recognition of a commented voigt line by its
  redshift label. Component ids (`a1`…) are assigned in order of z
  (`project.py:1259`), so the label, not the id, ties an excluded line to its
  component.
- `bundle.blind_line_numbers` already counts commented-out lines with `blind=True`, so
  an excluded blinded line stays hidden.
- ALIS stops on a link with an undefined variable (`load.load_tied`) and on a
  `fix param` or `lim param` naming a label that does not exist (`load.py:1942`,
  `1976`); `remove._Removal.settle` already finds such dependents for a removal.
- A simulation of the AOD rule (2000 noise draws; Si II at log N 13.0, b 5 km/s,
  FWHM 7 km/s, S/N 20 per 2.5 km/s pixel): 1304 and 1526 give 13.03 and 13.02 (about
  ±0.05); 1260 (93% deep) counts as saturated and gives 12.89; 1808 (0.9% deep) has
  no logarithm in 36% of draws and spans 12.9–13.8 otherwise. The lowest of 1304,
  1526 and 1808 is at least 0.2 dex low in 8% of draws (0.5 dex in 4%); restricted to
  3σ detections it is 13.00. The 3σ upper limits are 11.45 (1260), 12.31 (1526),
  12.54 (1304) and 14.04 (1808).

**New queries,** each with a lean:
- **Q5.16:** which estimates "the lowest" takes: only 3σ detections that are not
  saturated; the lowest 3σ upper limit when nothing is detected; 15.0 or the highest
  lower limit when all are saturated;
- **Q5.17:** the bounds of log N in a drag: a preference of the dashboard (8.0–23.0),
  or new limits in voigt.py;
- **Q5.18:** what an exclusion leaves behind: links, `fix`/`lim param` lines and ratio
  lines that only the excluded line's labels serve (commented out with it and
  restored together), and the commented-out lines of a removed component (deleted with
  it).

### 2026-10-14 (Prompt 2 again: Q5.16–Q5.18 answered; the Design and Tasks updated)

**The responses.**
- **Q5.16:** as proposed. Only transitions detected at 3σ or more and not saturated
  count, and the start is the lowest of them. With no detection, the start is the
  lowest 3σ upper limit; when everything is saturated, 15.0 or the highest lower
  limit, whichever is larger.
- **Q5.17:** (a), the dashboard's own bounds on log N in a drag (8.0 to 23.0, a
  preference). Limits written in the `.mod` (`lim voigt …` or `lim param …`, on log N
  or b) replace them and bound the drag; the user changes limits there.
- **Q5.18:** (a) the lines that only the excluded line's labels serve are commented
  out with it, with a note, and Include restores them together; (b) ✕ deletes a
  component's commented-out lines with it.

**The document brought in line with Q5.1–Q5.18.** The summary at the top, the Design
and the Tasks were rewritten (the Queries are unchanged). The main changes from the
first version:
- the navigator's groups (the primary system, the other systems, one Lyα forest
  group), and Add ion… offering every ion (those in a snip first);
- the Dataset menu and "All datasets"; the panels' contents as in Q5.3; the Regions
  tab's Model switch (Task 5.4);
- "+ Add component" arms one click, and a click on a tick selects; the first guess for
  every ion by the rule of Q5.13 and Q5.16; one handle on the selected component,
  moving log N and b together, Shift for one of them, bounded by the `.mod`'s limits
  or the preferences;
- the cards' Exclude/Include button, linked parameters' badge, and "Fixed at…" for
  the temperature;
- no link dialog (S15's editor and templates after v1); no Delete key;
- excluding by commenting out, with its dependents, and ✕ as the only removal;
- `profile.py` without `getemab` (the whole model divided by the continuum);
- Task 5.10 with one click on O I 1302 and the temperature set with "Fixed at…".

Everything is clear; ready to begin Stage 5 at RJC's confirmation.

### 2026-10-14 (Prompt 3: Tasks 5.1 onwards)

**5.1 The model of each snip (`alis/dashboard/profile.py`).**
- `Profile(project)` builds ALIS's state in memory as `ClassMain` does up to the fit
  (`main.py`, to "NOW PREPARE TO FIT THE DATA!"): `load_input`, `load_data` (the
  project's files), `load_model`, `load_links`, `load_parinfo`,
  `load_par_influence`, `load_subpixels`, then the links appended to `mlnk` and
  applied to `p0` (`load_tied`). `out wavecorr` is switched off, so no file is
  written.
- `evaluate(values=None, snips=None)` calls `model_eval.model_func(..., output=1)`
  and returns a `Curve` per snip: the snip's pixels, flux and error, ALIS's model
  placed on them by wavelength (NaN where ALIS loads no pixel: it loads only the fit
  range and the convolution's buffer, e.g. 392 of `examples/blind`'s 554), the
  continuum (`continuum.evaluate`), the fitted pixels (as `build_fit_windows`
  finds them) and χ². `normalised`, `data_normalised` and `residuals` follow.
- **Some snips only:** `model_func` evaluates a snip when `ddpid` is in
  `_pinfl[0][sp][sn]`, and nothing else depends on `ddpid` since Stage 4.0 (the
  `set_vars` calls no longer take it). The profile sets `_pinfl` to a table of its
  own that names the snips wanted with the number −1 (`SELECTED`), and passes
  `ddpid=-1`. No change to ALIS.
- `values(changes)` sets parameters by ALIS's number and applies the links;
  `affected(changes, values)` names the snips a change reaches: the lines that carry
  the parameters or are linked to them (`p[n]` in the links, followed through), a
  ratio line of a changed ion's denominator, and, for a voigt line, the snips where
  one of its transitions falls before or after the change (ALIS's own test in
  `voigt.set_vars`: the observed wavelength within the shifted pixels).
- `profile.of(project)` keeps the profile while the text and files are unchanged.
- **Measured:** the model equals ALIS's to 0 (bit for bit) on every fit model of
  `examples/` and on J1358p6522 and Q1243p307, and χ² equals ALIS's starting χ²
  (`examples/random` is skipped: its `random` line draws new values at each read).
  J1358p6522: all 12 snips in 32 ms, its 8 H I snips in 30 ms; Q1243p307: all 51
  snips in 228 ms (about 4.5 ms a snip). A drag evaluates only the ion's snips, so
  the sub-pixel cap of Q5.5 is not needed, and was not built.
- Tests: `tests/test_dashboard_profile.py` (36, one skipped): ALIS's own state is
  built by a `ClassMain` whose `main` stops where the fit starts, on the files on
  disk; a change reaches exactly the snips that change (J1358p6522, every line's
  log N, z and b); a linked parameter follows (`examples/summed_coldens`: `rb` from
  `ra`, `na` from `nb`…`nf`), and the model with a dragged value equals the model
  with it written; no hidden value in what the profile returns.
- Batches: `unit` 1850 passed, 121 skipped; `gui` 117 passed.

**5.2 Components (`alis/dashboard/components.py`), with additions to `project.py`,
`edit.py`, `remove.py`, `text.py` and `preferences.py`.**
- `new_component(project, system, z)`: the labels of `edit.new_component`, a line for
  every ion of the system (`system_ions`: its components' ions, excluded ones too,
  and its line IDs; not a ratio ion, nor the forest's), each with the specids of
  the snips that hold one of its transitions (`snips_holding`, Q4.3), or the
  system's snips (Q5.7); b_turb 5 km/s and T 0 K fixed (preferences); log N from
  the first guess. One step, `(step, guesses)`; the lines follow the system's
  others in the text.
- `first_guess(project, ion, z, bturb, T, curves)`: an `Estimate` per transition and
  snip, over ±½ FWHM_line (FWHM from `vfwhm`, `vsigma` or `Afwhm`; else two pixels),
  τ_a = ln(M/F) against the current model (`profile`), the flux clipped at its
  error. Detected (∫τ dv ≥ 3σ) and not saturated (no pixel within 3σ of zero):
  the lowest; else max(15.0, the saturated lower limits); else the lowest 3σ upper
  limit; else 12.0. b is the total b, b² = b_turb² + 0.01662892444 T/A, ALIS's own
  constant (`voigt.parin`).
- On `metal_line_abs`: O I 13.954 and Si II 12.980 (best fit 13.985 and 13.041), the
  same as computed from the file in Prompt 1. A new project must be given the
  file's column roles: by D13's rule its fourth column is read as a continuum, and
  the guess is then 0.3 dex off.
- On a synthetic Si II spectrum (b 5 km/s, FWHM 7 km/s): noiseless, 1304 and 1526
  give 12.97 and 12.96 for 13.0 (the window holds whole pixels, about 10 of its
  10.9 km/s, so the guess is about 0.03 dex low); at S/N 20 each estimate scatters
  by about 0.06 dex, so the lowest of two can be 0.15 low; at S/N 50, 1260 (93%
  deep) no longer counts as saturated, and its AOD (0.1 dex low from unresolved
  saturation) would be the lowest. Both are the agreed rule's own behaviour (Q5.16).
- `add_ion(project, system, ion)`: a line in every component, with its z, b_turb and
  T words and labels, and its own first guess; `addable_ions` lists the ions of
  chemical elements (`lines.atomic`), in the order of the periodic table, those
  with a transition inside a snip first.
- Drags: `drag(project, comp, ion, z, logn, b)` gives `(Param, value)` pairs, each
  within `bounds` (the model's limits, voigt.py's b_turb ≥ 0.5 km/s among them, as
  `Param.limits` holds them; log N 8–23 where the model sets none); the total b
  gives b_turb at the component's T, or T in the pure thermal mode; a linked
  parameter raises `DragError`. `drag_step` writes the values as one step named
  "Drag component a1", with no value in it (D24).
- `temperature(project, comp, mode, value)`: preset/fixed, free and thermal; going
  into or out of the pure thermal mode keeps the line's total b.
- Excluding (Q5.15, Q5.18): `exclusion` plans with `remove._Removal`'s own search for
  dependents, then comments out the ion's line (`#` in front) and the dependents
  (`#` in front and `  # excluded with 16O_I za1` after); `include` takes the `#`
  and the notes away, then rewrites z, b and T to the component's words and the
  specids to the snips now, only where they differ, so that exclude then include
  gives back the text byte for byte. An ion the component never had gets a new line
  with a first guess. The last ion raises, and `remove.plan_component` (✕) now
  deletes the component's commented-out lines too.
- `project.py`: `commented_lines(mtext)` reads the commented-out lines of the model
  and link blocks (`text.ModelText.comments_in`, new: a comment line belongs to no
  block); each `Component` has `excluded` (ion → line) and `excluded_with` (the
  noted lines). `edit.used_labels` now counts the labels on commented-out lines
  (`commented_labels`), so that no new label takes one.
- `interloper(project, wave, ion, fixed)`: a `1Ly_a` (or `1H_IB`) line at the clicked
  Lyα wavelength, b 20 km/s, T 0 fixed with the forest's own label when its lines
  share one (`ZEROT` in J1358p6522), else `TLY1`; fixed: every parameter labelled
  (`NLY1`, `ZLY1`, `BLY1`, upper case).
- Notices: `project.keep_notice` ("Keep as imported") stores a 16-character digest of
  the notice's kind and lines in `ui/project.json` (`kept_notices`), never a line's
  text, which may hold a hidden value; such notices are no longer shown.
- New preferences: `components.start_b` (5), `temperature` (0), `saturated_logn`
  (15), `nodata_logn` (12), `logn_min` (8), `logn_max` (23), `preview_rate`
  (20 per s).
- Tests: `tests/test_dashboard_components.py` (22). Batches: `unit` 1872 passed;
  `gui` 117 passed.

**5.3 The ion navigator (`alis/dashboard/qt/components.py`, new; `tabs.py` imports
it).**
- `components.navigator(project)`: each system (the primary first) with a `NavIon`
  per element and stage (its main isotope's ion, and its isotopes: "with D I"), and
  a `NavForest` (the project's `1Ly_a`/`1H_IB` lines and the snips they apply to).
- **The panels' rule** (`panel_transitions`), which also gives the navigator's
  count: the transitions inside a snip that a snip was cut for (SNIP's record, or
  its guess), and the others whose fλ is at least a tenth of the strongest of
  those (of the strongest inside a snip, when none was cut for the ion, as O I 1302
  in a Si II 1304 snip); transitions within 50 km/s share a panel ("O I 988.8 ×2").
  Every transition inside a snip would give J1358p6522's H I 22 panels (Ly8 to the
  Lyman limit share one snip) and its O I 32 (weak lines in the Lyman-series
  snips); the rule gives 8 and 6, as the mockup has them.
- The tab: a tree of systems and ions; each ion's name, and under it what it has,
  muted (a delegate draws the detail column under the name); the first ion of the
  primary system is shown by default; what is shown is kept in the view state
  (`ui/view.json`, "components"). Add ion… (`dialogs.IonDialog`, new: the ions with
  a transition inside a snip first, then the others, filtered as typed) and Add
  system… (as on the Data tab). On a system with no component yet, Add ion… names
  the ion among the system's line IDs, so that its first component has a line of it.
- Actions (`actions.py`, scope "Components"): ↑ ↓, Add ion…, Add system….
  `duplicate_shortcuts` now lets two tabs each give a key their own action (↑ on
  the Regions and Components tabs act only while their tab has the focus); the
  window test of shortcuts follows the same rule.
- Tests (`tests/test_dashboard_qt_components.py`): J1358p6522's navigator as in the
  mockup ("H I · 8 transitions · 2 comp. · with D I", "O I · 6 transitions · 1
  comp.", "Lyα forest · 57 comp. in 10 snips"; the mockup said 11 snips, the model
  has 10); Q1243p307's five systems, the primary first; ↑ ↓ only with the tab's
  focus; Add ion… with O I 1302 inside a Si II 1304 snip; Add system….
- Batches: `unit` 1872 passed; `gui` 122 passed.

**5.4 The panels (`qt/components.py`, `qt/plots.py`; the Regions tab's Model switch in
`qt/regions.py`).**
- `components.panels(project, system, ion, dataset)`: a `Panel` per transition of the
  chosen dataset (or per transition and dataset), titled "O I 1302.2 · HIRES" (Lyman
  lines by name, "Lyβ 1025.7"), with its pixels as velocities in the model's frame
  (the row's shift removed, `continuum.model_wave`) and its fit regions; what the
  dataset does not cover ("O I 988.8 ×2: not in PROCHASKA"). `panel_datasets` gives
  the Dataset menu (the reference first). `forest_panels`: a panel per snip that
  holds a forest line, in the frame of the transition it was cut for.
- `panel_ticks`: every component of the system at its velocity ("own"; "excluded"
  for one without the ion, faint and dashed), the isotopes' lines beside them (D I
  at −81.6 km/s), and the lines of other systems and of the forest that fall in the
  panel and apply to its snip (grey, locked).
- `plots.py`: `SpectrumView.set_model` (ALIS's model, divided by the continuum with
  the data), `set_ticks` (short lines at the panel's top, labels at three heights
  so that close ones can be read; `tickClicked`, `tickMoved` for 5.5), `x_title`, and
  `ResidualView` (steps, ±1σ dashed, ±4σ, its x axis linked to its panel's).
- The tab: Dataset (hidden with one dataset), Columns (1–4; the preference, 3) and
  Range (±100 to ±2000 km/s; the preference, 200) above the panels, kept in the view
  state; one zoom bar for every panel; the model of the panels' snips evaluated once
  (`profile.of(project).evaluate(snips=...)`); a model that cannot be evaluated is
  said above the panels, which then show the data over the continuum alone.
- The Regions tab's **Model** switch (`regions.model`, beside Normalised): the
  model as it will be fitted on each snip, divided by the continuum when Normalised
  is on.
- Seen on a screenshot (J1358p6522, two columns): panels removed from the grid kept
  showing until the event loop ran (`deleteLater`), so they are now hidden and
  unparented at once; the navigator and the cards have minimum widths (170, 250),
  for the `.mod` panel open beside the tabs.
- Tests: J1358p6522's H I in two columns (eight panels, scrolling, ±200 km/s, ticks
  a1, a2 and D I); the model drawn equals `profile.py`'s; Q1243p307's O I 1302 in each
  dataset and in "All datasets", the ticks at the same velocities; the Model switch.

**5.5 Adding, selecting and dragging (`qt/components.py`, `qt/plots.py`).**
- "+ Add component" (`components.add_component`, checkable, at the head of the
  panels' tools) arms one click: the next click in a panel adds a component at that
  velocity (`components.new_component`), or, in the Lyα forest, a line of the forest
  at that wavelength (`interloper`); the cursor is a cross while armed; Esc cancels
  (a shortcut on the tab). The new component is selected, and its first guesses'
  notes kept for its card.
- A click on a tick selects its component (`tickClicked`); the click then reaches
  the panel too, and a flag stops it clearing the selection. A click elsewhere
  clears it, and never adds a component unless armed.
- The selected component has a handle in each panel of the ion (`set_handle`, a
  pyqtgraph `TargetItem`), on the model one total b to the side of its velocity. A
  drag: log N + 2 dex × (start height − height)/(panel's height), b = the handle's
  distance from the component's velocity; with Shift (read from the keyboard's
  modifiers), the first 2% of the panel's width or height moved chooses the one it
  changes, and the handle is held to that axis (`place_handle`). A tick dragged
  moves z. `components.drag` bounds each value; a linked one says why it cannot.
- The preview: ALIS's model of the panels' snips with the drag's values
  (`profile.values`, `evaluate(snips=...)`), at most at the preference's rate (a
  single-shot timer); the panel's corner says log N, b_turb or T, and v, or only
  Δχ² when a value is hidden (`readout`). The step is written when the drag ends
  ("Drag component a1").
- Found on the screenshots: a refresh during a drag (the window's timers run while
  the mouse is held) rebuilt the panels and took the handle away, so the tab now
  holds a refresh until the drag ends.
- Tests: one click at O I 1302 on a new `OI_SiII.dat` project adds one component with
  an O I and a Si II line, each log N within 0.1 dex of the best fit; an unarmed
  click selects or clears and never adds; a handle drag is one step (log N and b as
  computed from the movement) and undo restores the text; a tick drag moves z; a
  drag stops at b_turb = 0.5 km/s and log N = 23; a blinded drag shows only Δχ², and
  no hidden value reaches any string of the window.

**5.6 The cards (`qt/components.py`; `components.param_state`, `plan_interloper`;
`Exclusion.preview`; `dialogs.PlanDialog` says "comment out").**
- The cards of what the navigator shows: the system's components (a title, then a
  card each, in order of velocity), or the forest's lines (a card each: its Lyα's
  wavelength, log N and b, ✕).
- A card: "a1 · v = −7.2 km/s" (the velocity through the gate), Blind… and ✕ in the
  header, Exclude from / Include in the ion shown on a row of its own; z and b_turb,
  each a field to type in and a badge (● free, ■ fixed, ⛓ tied — its label used
  outside the component, the places in the tooltip —, ƒ computed by a link, its
  expression in the tooltip, through the gate, and the field read-only); the badge's
  menu: Free, Fixed, Tie to (the same parameter of the system's other components),
  Untie, Limits…, Clear limits; a parameter that needs a label for these is asked
  for one (the scheme's suggestion). The temperature: its mode ("0 K · fixed",
  "8000 K · fixed", "T free", "pure thermal") and a menu of the modes, "Fixed at…"
  among them. Then every ion of the system (the navigator's rows), isotopes under
  their element ("↳ D I"), each log N with its badge, or "excluded" / "not here".
- A component without the ion shown is greyed but for its Include button. Exclude
  previews the lines when more than the ion's own line is commented out; the last
  ion asks to remove the component (`exclude_last`). ✕ shows the removal's plan.
  Blind… asks, then sets `blind=True` on the component's lines.
- A typed value is one step named "Set log N of O I in a1" (`edit.set_value`'s own
  description names the value, which may be hidden, D24); typing over ▒▒▒▒ keeps
  the value hidden.
- Seen on the screenshots, with the .mod panel open beside the tab: the panels' tools
  are now in two rows (Add component and the zoom bar; Dataset, Columns, Range), the
  cards no longer scroll sideways (the toggle on its own row), a panel can be 110 px
  wide, and with more than 6 components in a panel only the selected one's tick is
  labelled.
- Tests: J1358p6522's cards as in the mockup; every badge, menu and button one step
  (Fix and Free z, T fixed at 8000 K, a typed log N, Exclude and Include, Blind…, ✕);
  the last ion asks first; on `examples/blind`, no hidden value on a card, and typing
  over the mask keeps it hidden.
- Batches after 5.5 and 5.6: `unit` 1872 passed; `gui` 135 passed.

**5.7 Review of the Components tab (Q5.1).**
- `doc/dashboard/stage5/take_screenshots.py` (ten screens) and `build_review.py`, in
  the style of the earlier review pages; published, private, at
  https://claude.ai/artifact/7hNzToHBQmpTPXa2RTpoQr: J1358p6522's O I and H I beside
  the two mockups, then a click on a new project, a drag, an exclusion, Q1243p307's
  datasets, the forest, a hidden component, Add ion… and the Regions tab's Model
  switch.
- Four questions are on the page: the panels' rule, which ions each card lists, two
  columns by default with the .mod panel open, and the forest's starting b.
- Paused here for RJC's comments, as agreed (Q5.1).

**5.7 RJC's comments on the review page (2026-10-15).**
- Q5.19, which transitions get a panel: "A snip gets a panel." Done: one panel per
  snip that holds a transition of the ion at the system's z
  (`components.ion_snips`, replacing `panel_transitions` and its fλ ≥ 0.1 and 50 km/s
  rules), zero velocity at the snip's strongest transition, whose ticks are dragged;
  the components drawn again, thin and unlabelled (`"echo"` ticks), at the snip's
  other transitions, and a click on one selects its component. Panels by the
  strength of their strongest transition, then dataset. Titles "O I 988.8 (+2) ·
  HIRES", naming the snip when it was cut for another transition, since two snips
  can hold the same one ("O I 929.5 (+2) · Ly7 926.2 snip · …"); a long title is cut
  at the panel's edge, whole in its tooltip. The navigator counts snips ("8 snips").
- J1358p6522: H I 8 panels (Ly8 to the Lyman limit one, "(+14)"), O I 11.
  Q1243p307: O I 26 panels in all datasets, 8 in PROCHASKA with "O I 988.8: not in
  PROCHASKA".
- Tests: three new unit tests (a snip gets one panel; every snip holding the ion is
  shown, titles distinct; only the strongest transition's ticks are dragged, the
  echoes at the right velocities); GUI tests updated for the navigator's wording and
  the Ly8 panel's movable ticks. Stage 5 tests: 79 passed, 1 skipped.
- Q5.20, the ions on a card: "Included: … / Excluded: …" rather than a "not here"
  row per ion. Done: the log N rows are only for the ions the component is in
  (isotopes under their element only when it is in the element too), and the
  system's other ions are on one line, "Excluded: D I, O I" (`card.excluded`), the
  ion shown in bold, each a link that includes it (`_toggle_ion`, one step). The
  log N rows stand for "Included", so no separate line repeats them; offered on the
  page.
- Q5.21, columns: `components.columns` starts at 2. Preference tests updated (3 is
  now the user's non-default value in the override test).
- Q5.22, forest b: a new preference `components.forest_b` (20 km/s) in place of
  `components.FOREST_B`.
- Tests: `test_the_excluded_line_includes_an_ion` (new), J1358p6522's a1 card reads
  "Excluded: D I, O I", the two-column default and its choice kept.
- RJC on the "one click" screen: min(Nsnips, Ncolumns) columns. Done in
  `_fill_panels` (`ncol = max(1, min(columns, panels))`; a panel is a snip now);
  the Columns box keeps the setting. Test: one O I panel fills the width with the
  setting at two.
- Also, seen on the new Lyman-limit screenshot at ±2000 km/s: the forest's "Lyα"
  labels overlapped, so locked lines go unlabelled when a panel has more than six
  of them (as components already did).
- Batches after Q5.19–Q5.22 (before the last two changes): `unit` 1875 passed;
  `gui` 136 passed; the tab's GUI tests again after them: 19 passed.
- The Excluded line keeps each ion's name whole when it wraps (non-breaking spaces).
- Draft 2 of the review page published at the same URL (version 2): the four
  questions answered in place ("Decided on this page"), so that the threads stay
  anchored; screenshots in the two-column default, a new "Ly8 to the Lyman limit"
  screen (one column, ±2000 km/s, the .mod panel closed). Each of RJC's five threads
  answered with what was done, and resolved.

## Prompt 4 (2026-10-15): from Task 5.8

No further queries before starting: RJC's review and the answers to Q5.19–Q5.22 are
applied (draft 2 of the review page).

**5.8 Blends, links and notices (S14, S15, Q0.3; Q5.9, Q5.14).**
- Interlopers: `components.interloper_choices(project, wave, shown)` gives Add
  interloper's choices at a clicked wavelength: a line of the Lyα forest whose Lyα
  is there (`1Ly_a`, or `1H_IB` for every Lyman line), free or fixed; then the
  components of other systems whose transitions (their line IDs, `lines.line_ids`,
  restricted to each system's ions) fall within 300 km/s of the click, nearest first,
  at most eight ("Lyα 1215.7 of z = 2.39869, at +37 km/s"). From a system's panels
  that system is not offered; from the forest's, every system is.
  `components.add_interloper` makes the step: `interloper` for the forest,
  `new_component` (a line for every ion of that system, D20) for a system.
- In the tab: a panel's right-click menu lists the choices (`context_actions`, as
  the Regions tab's); **Add interloper…** (`components.add_interloper`, checkable,
  beside + Add component, which it disarms) arms one click that opens the same
  menu there. The design listed it as an action; a click is needed for the
  wavelength, so it arms one, as + Add component does.
- Links: `components.draggable(comp, ion)` says what a drag can move: a component's
  tick is fixed when a link computes its z; its handle moves only what no link
  computes (log N, or b, or T in the pure thermal mode), and is absent when links
  compute both. `components.linked_values` gives what ALIS computes for every
  linked parameter (`Profile.values`, the links applied); cards and ticks show
  that, through the gate (which already hides what a link computes from a hidden
  value), since the value written for a linked parameter is not the one fitted:
  after a drag of `ra` in `summed_coldens`, `rb`'s tick is where ALIS puts it.
- Notices: `components.notices_for(project, system, ion)`: the notices of the
  system's components about the ion or its isotopes, shown above the panels
  (`NoticeRow`, at the top of the scrolled panels) with the fix and Keep as
  imported, each one step. `Notice.fix_text` names the fix ("Tie O I's z to a2",
  "Tie D I to H I"), and `Project.fix_notice` uses it as the step's name.
- Tests: unit (interloper choices on Q1243p307, a step and its undo; the forest's
  choices on metal_line_abs; links on summed_coldens; J1358p6522's notices and
  fixes); GUI (an interloper from the menu joins the forest and is drawn locked;
  Add interloper… arms one click; a linked log N on its card, the linked z's tick
  fixed, the handle moving b alone, rb's tick following ra's drag; J1358p6522's two
  notices fixed in one click each, the marker's "!" gone, Keep as imported).
- Batches after 5.8: `unit` 1879 passed; `gui` 139 passed.

**5.9 Cross-highlighting, markers, actions (S16, F12; Q5.7).**
- From the `.mod` panel: `components.at_line(project, num)` says what a line
  describes: a voigt line's component and ion (the navigator's: H I for a D I line),
  a ratio line's ion, a line of the Lyα forest; for a link, or a `fix`/`lim param`
  line, the parameter it constrains (through its label). The tab's `show_items`
  reads the panel's cursor line (`ModPanel.cursor_line`, new), shows that ion (or
  the forest) and selects the component.
- From the tab: a component selected (tick or card), an ion chosen in the
  navigator (its lines and its isotopes'), or a click in a panel (its snip's lines)
  highlights the lines in the `.mod` panel. `ModPanel.select_items(items,
  move=False)` (new argument) highlights and scrolls the lines into view without
  moving the cursor or taking the focus: the tab's ↑ ↓ need its focus, and moving
  the cursor would have the tab follow it back. Other tabs keep `move=True`. A
  refresh does not highlight, so typing in the `.mod` is never scrolled away.
- The new marker rule (Q5.7): `components.unsnipped_ions` lists the ions with
  lines that no snip holds; the Components marker shows "!" with "C II (z =
  0.00000) is in no snip, so its lines constrain nothing: cut a snip of one of its
  transitions on the Regions tab." (after the notices, before "the regions have
  changed").
- Actions: Add interloper… joined the tab's actions in 5.8; the preferences of Stage
  5 are in the Preferences dialog, which lists every `SPECS` entry.
- Tests: unit (`at_line` on J1358p6522's lines and on summed_coldens's link; the
  marker's message for an ion in no snip); GUI (the `.mod` cursor selects a2 and
  shows O I; a card, an ion and a panel highlight their lines, the cursor left
  where it was; a forest line shows the forest; a link line selects what it
  computes; the "!" and its tooltip, gone with the undo).

**5.10 From a spectrum to a fit (end to end).**
- `tests/test_dashboard_qt_end_to_end.py` (fast, examples): Task 4.12's test, with
  the absorption now added on the Components tab instead of `edit.add_ion`: O I
  chosen, + Add component, one click at 0 km/s on O I 1302 gives one component with
  an O I and a Si II line sharing z and b (one label each, D20); "Fixed at…" 8000 K
  on its card's temperature menu, as the example's model has it; the Components
  marker is ✓. `run_alis` then fits the bundle.
- Fitted against `fit_spectra.mod.out.reference`: O I 13.9843 ± 0.0155 against
  13.9852 ± 0.0155; Si II 13.0436 ± 0.049 against 13.0412 ± 0.050. 5 s.

**5.11 Close the stage.**
- Draft 3 of the review page, at the same URL (version 3): the finished tab, with
  three new screens (Add interloper's menu on Q1243p307, which offers Lyγ of z =
  2.39869 and two Fe II lines of z = 2.05277; summed_coldens's links, a1's log N
  12.0375 as ALIS computes it and a3's z with ƒ; an ion in no snip, the tab's "!"),
  and the notices and cross-highlighting in the J1358p6522 screens. The answered
  questions keep their place on the page. Waiting for RJC's review.
- `doc/ALIS_workflow.md` §4.3: "Building the model: the Components tab".
  `CHANGELOG.md`, `tests/README.md` and the `gui-component` skill ("Models, ticks and
  drags") updated.
- `test-coverage` over the Stage 5 tests: `components` 92%, `profile` 91%,
  `qt/components` 91% (91% overall). Two tests added for the largest untested
  paths: a card's Tie to, Untie, Limits… and Clear limits, each one step, the limits
  bounding drags; and a click in the Lyα forest's panels adding a forest line. Left
  untested: a label asked for when an imported value without one is fixed, and a
  drag refused by a link (no longer offered, `draggable`).
- `ALIS_v2_dashboard_prompts.md`: D60–D68, the stage's decisions. `dashboard_stage0.md`:
  S15's `link` editor and templates after v1 (Q5.9). This document's Status section,
  with what Stage 6 receives.
- Seen while closing: the Preferences dialog, with Stage 5's ten preferences, was
  934 px tall at least, more than a 900-pixel screen (D43); its form now scrolls,
  and the dialog opens at most 720 px tall.
- Batches at the close: `unit` 1881 passed, 0 failed; `gui` 144 passed; `fast` 112
  passed, 0 failed (13 min 42 s). The Preferences dialog's change came during the
  `gui` batch; its window and session tests were run again after it: 20 passed.
- RJC's comment on the review page (thread "a click on its tick", 10:08): a click
  selects the nearest component; a drag above the continuum moves z, below it log N
  and b. It changes the gestures agreed in Q5.4 and Q5.12, and leaves open how the
  panels pan, so it is Q5.23 in the Queries section, with my lean; development is
  paused for the response. The page's auto-reply had said that a click far from every
  component adds one: it does not (Q5.15(b)); corrected on the thread.

## Prompt 5 (2026-10-15): Q5.23, the final fixes

No further questions on RJC's response to Q5.23.
- `qt/plots.py`: `SpectrumBox.edits` / `SpectrumView.set_edits(True)`: in pan mode a
  left drag emits `dragged(x0, y0, x, y, state)` ("start", "move", "finish") instead
  of panning, and a right drag pans (`translateBy`) in any mode; the wheel zooms the
  velocity axis as before (D38). The handle (`set_handle`, `handle`, `place_handle`,
  `handleMoved`) and the ticks' own drags (`movable`, `tickMoved`) are removed.
- `qt/components.py`: `nearest(widget, x)`, the component nearest a velocity by its
  ticks (thin ones at other transitions included, locked lines not); a click
  (unarmed) selects it, or, with none, highlights the panel's snip; Esc disarms an
  armed button, else clears the selection. `_box_dragged`: a drag selects and moves
  the component nearest where it starts, by its movement: above the continuum
  (normalised flux > 1) its z, its ticks in every panel moving with it; below it,
  log N (2 dex over the panel's height) and b (by how much farther from that
  component's tick the pointer is) together. `DragKeys`, an application event
  filter: N or B pressed while the pointer is over a panel (or a panel has the
  focus) is held, and consumed, so it is typed nowhere else; N keeps b, B keeps
  log N; a key pressed or let go mid-drag re-bases the drag where it is, so nothing
  jumps. An excluded component, or a z a link computes, is selected but not
  dragged. The zoom bar's ✥ tooltip says that a drag edits and a right drag pans.
- Tests: rewritten for the new gestures (a click selects the nearest, Esc; a drag
  below the continuum is one step with log N and b by its movement, above it z by
  its movement; bounds; blinded, only Δχ²; N and B; a right drag pans and a left
  drag edits, through the view box; an excluded component's drags do nothing; the
  Ly8 panel's thin ticks select their component; the linked parameter's drags).
- Docs: the module docstrings, `doc/ALIS_workflow.md`, the `gui-component` skill,
  `CHANGELOG.md`, `tests/README.md`, D63 of `ALIS_v2_dashboard_prompts.md` and this
  stage's Design (Selecting, Dragging); Q5.23's "Done" note.
- Draft 4 of the review page published at the same URL (version 4); RJC's thread
  answered and resolved. No thread is open.
- Batches at the close: `unit` 1881 passed, 0 failed; `gui` 146 passed; `fast` 112
  passed, 0 failed (13 min 31 s). Stage 5 is closed.
