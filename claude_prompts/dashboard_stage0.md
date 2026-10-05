# Prompt file for ALIS software dashboard creation -- STAGE 0

> **Mockups of the dashboard layout.** Before any design work, draw static mockups of
> every tab from real data, so that the layout can be judged and changed cheaply (QF.2,
> QF.18(a), QF.31). The pages are pictures of the layout, viewed in a browser, not
> working controls. Stage 0 must not change any code under `alis/`. It adds only
> `doc/dashboard/mockups/` (a build script and the pages it writes). The stage ends
> when RJC has chosen the Regions and Components arrangements. `dashboard_stage1.md` is
> then written, using what the review taught us.
>
> As agreed in QF.34 and QF.36 (D33), this document also holds the plan for all of the
> stages, and the v1/later split of every proposed item.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to the Decisions list, the Functionality
> items and the Queries in `claude_prompts/ALIS_v2_dashboard_prompts.md`.

## Dashboard plan

### Stages

Each stage gets its own document, written when the previous stage is done (D33). A
stage may be split when its document is written, as Stage 3.5 was in the refactor.

| Stage | Contents | Code touched |
|---|---|---|
| 0 | Mockups of all five tabs (this document). | `doc/dashboard/` only |
| 1 | Changes to ALIS itself: the bundle (D8); `run_alis project.model` and `--extract` (D9); plain-file export (D10); in-memory loading of several data lines (D9); detecting shared pixels, with a warning in `run_alis` (D19); removing onefits (D11). Checked by the refactor's regression harness. | `alis/` |
| 2 | The project model, with no Qt: the text-sync layer that maps datasets, systems, snips and components to their `.mod` lines and makes targeted edits (D7), tested by opening every context model and writing it back unchanged (QF.5); the validator (F5); the blinding gate (F8); undo/redo (F2); removal with dependents (S27); the mode interface, with Voigt mode only (F11); the logic of cross-highlighting (S16) and of the out-of-date marker (F12). | `alis/dashboard/` |
| 3 | Qt skeleton of the chosen layout: the window, five tabs and their status markers (F12), the `.mod` panel, the status bar, open/save/autosave (F1), opening an existing fit (F4), relinking moved spectra (F13), the shortcut sheet (F10), the user's preferences file (Q2.2) and `run_alisgui` (F14). | `alis/dashboard/` |
| 4 | Data and Regions tabs (steps 1–3). | `alis/dashboard/` |
| 5 | Components tab (step 4). | `alis/dashboard/` |
| 6 | Fit tab (step 6). This completes v1. | `alis/dashboard/` |
| Later | The Plot tab (step 8), Orders mode, and the later items below. | |

### v1 and later

Every accepted item, and the stage that builds it. Where two stages are given, the
first builds the logic and the second the interface.

| Item | Stage | Item | Stage |
|---|---|---|---|
| F1 Never lose work | 1, 3 | S9 Normalised view | 4 |
| F2 Undo/redo | 2, 3 | S10 Shared continua | 4 |
| F3 Plain-file export (D10) | 1, 3 | S11 Component matrix | 5 |
| F4 Open an existing fit | 2, 3 | S12 First guess of log N (AOD) | 5 |
| F5 Validate as you type | 2, 3 | S13 Live preview | 5 |
| F6 Background fit runner | 6 | S14 One-click blends | 5 |
| F6 Re-attach after closing | Later | S15 Limits and `link` editor | 5 |
| F7 Run history (D26) | 6 | S15 Constraint templates | Later |
| F8 Blinding gate | 2 | S16 Cross-highlighting | 2, 4–5 |
| F9 Settings from `ArgFlag` | 6 | S17 Pre-flight check | 6 |
| F10 Shortcut sheet | 3 | S18 Fit-quality badges | 6 |
| F10 Command search | Later | S19 Results table, correlations | 6 |
| F11 Modes as plug-ins | 2 | S20 Continue from best fit | 6 |
| F12 Tab status markers | 2, 3 | S21 Convergence tools | 6 |
| F13 Relink moved spectra | 1, 3 | S22 Plot preview (D28) | Later |
| F14 One launcher | 3 | S23 Attach a source spectrum | Later |
| S2 Transition coverage | 4 | S24 "Set z here" | 4 |
| S3 Column roles (D13) | 4 | S25 Dataset panel (D14) | 4 |
| S5 Line identifications | 4 | S26 Showing shared pixels | 1, 4 |
| S6 Snip extent and mask | 4 | S27 Remove anything | 2, 4–5 |
| S7 Mask helpers | 4 | S28 Reproduce an imported fit | Later |
| S8 Automatic continuum | 4 | S29 Review clipped pixels | 6 |
| S30 Ion navigator | 5 | S31 Results from `run_alis` | 6 |

Also later: Orders mode (D29–D32) and FITS input (D13). S1 and S4 are not needed.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage0_log.md`
> (QF.16(b)).

**0.1 — Choose what the mockups show.** Read the two fits and record in the log which
transitions, ions and components each page will show.
- **`context/fitting_examples/VMP_DLA/J1358p6522/`** is the main example. It has a
  full spectrum, `J1358p6522_fluxcal.dat` (three columns, about 40,000 pixels), 12
  snips, `J1358p6522.mod` and reference outputs (`.mod.out`, `_fit.dat` files and a
  covariance matrix). The system is at z = 3.06726, with:
  - H I and D I (`1H_I`, `2H_I`) in eight Lyman-series snips, which shows isotopes
    sharing z, b_turb and T (D22);
  - O I in six transitions: 976, 1039 and 1302 have their own snips, while 921.9,
    925.0 and 971.7 sit inside the H I 923, 926 and 972 snips;
  - 57 Lyα-forest interlopers (`1Ly_a`), the generic absorbers of D15;
  - real shared pixels: H I 923/926, 926/930 and 930/937 share 18–23 pixels each
    (S26).
- **`context/fitting_examples/DH/Q1243p307/`** shows several datasets. Its snips come
  from three datasets: the new HIRES data (no prefix), `kirkman` and `prochaska`. The
  model is `Q1243p307_converge_newstart76.mod`, which also has contaminant systems
  (`zabs2p05` to `zabs2p44`). There is no full spectrum, so its views are limited to
  the snip extents, as for an imported fit (D12). O I 1302.2 and Si II 1304.4 share 97
  pixels in each of the three datasets (S26).
- The model names its atomic data file (`run atomic atomic_rjc.xml` in
  J1358p6522). The mockups use the same file, as the dashboard will.

**0.2 — The build script.** Write `doc/dashboard/mockups/build_mockups.py`, which reads
the data and writes every page, so that the mockups can be regenerated after each
round of comments.
- Plots are drawn from the data, downsampled to about one point per screen pixel. The
  method (inline SVG or images) is Q0.1.
- Models and residuals come from the reference outputs (`_fit.dat`; columns described
  in `doc/ALIS_workflow.md` §5.1), so ALIS is not run.
- The script lives under `doc/`, so it is not part of the `alis` package and adds no
  dependency. It follows the Coding section of the prompts document, with lines of at
  most 88 characters (D3).

**0.3 — The common frame.** Every page uses the same window, so that the tabs can be
compared:
- a menu bar, and undo/redo;
- the five tabs, with status markers (F12): for example, Components marked out of date
  because a region changed;
- the collapsible `.mod` panel (D5), open on some pages and closed on others. It shows
  the real `.mod` text, with one line highlighted to show cross-highlighting (S16);
- the status bar, showing a fit running in the background (F6);
- blinded values masked (`▒▒▒▒`) wherever they appear (D24). On the J1358p6522 pages
  one component (the D I column density, say) is blinded, so that the masking is
  reviewed on every tab.

**0.4 — Data tab.** Two pages:
- **New project (J1358p6522).** The full spectrum, with the system list (z, and the
  history of "set z here", S24), the global-blind switch (D24), and the column-role
  dialog (S3) shown as an inset.
- **Several datasets (Q1243p307).** The dataset panel (S25): one row per dataset, with
  its reference flag, FWHM, shift, zero level, file and checksum status (F13). A
  coverage chart shows which transitions each dataset covers.

**0.5 — Regions tab: three alternatives.** Each shows one large panel for the current
transition (D5), with the data, the continuum, its order with +/− buttons and the change
in χ² (S8), the fit regions, the snip edges as handles separate from the mask (S6),
masked pixels (S7), line IDs for every system (S5), shared pixels hatched (S26) and the
normalised-view toggle (S9).
- **A — List and panel (J1358p6522).** The transition coverage list (S2) at the left,
  ranked by strength and flagged, with the large panel beside it.
- **B — Overview and panel (J1358p6522).** A strip of the whole spectrum across the top,
  marking every snip, with the transitions as a row of buttons and the large panel
  below.
- **C — Several datasets (Q1243p307, QF.36(c)).** O I 1302.2 with its three datasets
  stacked on a shared wavelength axis, each with its own regions (D18). The pixels it
  shares with Si II 1304.4 are hatched, with the two fixes offered (D19).

**0.6 — Components tab: two or three alternatives (J1358p6522).** Each shows the ion
navigator (S30), equal panels of every transition of the chosen ion on a velocity axis
relative to the system z (D5), the component positions marked in every panel, other
ions' components drawn but locked, residuals under each panel (S13), and the component
matrix (S11) with free/fixed/tied toggles and each component's temperature mode (D21).
- **A — Grid.** The navigator at the left, O I's six transitions as a 3×2 grid, and the
  matrix below.
- **B — Column.** The navigator as a list across the top, the panels in one column (as
  in the publication figures), and the matrix in a sidebar.
- A second page of whichever arrangement is preferred shows H I with D I, to show how
  isotopes are presented (D22).
- **C (optional).** Add a third arrangement only if building A and B suggests a
  distinct one.

**0.7 — Fit tab (J1358p6522).** One page, using the reference outputs:
- run controls, including settings generated from `ArgFlag` (F9), and live progress
  (χ² against iteration);
- the panels with fit-quality badges (S18), computed in the build script as `report.py`
  computes them (χ²_ν and the runs test per snip);
- the results table (S19) from `J1358p6522.mod.out.reference`, with flags, and the
  correlation matrix from `J1358p6522.covar.reference`;
- the initial/final model toggle; the run history with its commit button (F7, D26);
  and a run made by `run_alis project.model` waiting to be inspected (S31).

**0.8 — Plot tab (built after v1, designed now).** One page: the grid and figure size,
the presets (metals, DH, blends, helium; D28), transitions assigned to panels, and a
preview drawn in the style of `context/plotting_examples/DH_Lya-Ly7_J1358.py`.

**0.9 — Index, check and publish.** Write `doc/dashboard/mockups/index.html`, which
links every page and says, for each alternative, what it is testing and what RJC is
asked to decide. Check every page at the agreed window size (Q0.2). Publish the pages
as one private page that RJC can comment on (QF.31(d)).

**0.10 — Review.** RJC chooses the Regions and Components arrangements, or asks for
changes. Record the choice, and any change it makes to D1–D33 or to the plan above, in
this document and in the log. Stage 0 then ends.

  *Review of draft 1 (2026-10-04), from RJC's comments on the published page and
  the responses to Q0.3–Q0.5. Draft 2 applies all of it.*
  - **Regions:** arrangement A. Several datasets are stacked, one strip each,
    never overlaid, and scroll when there are many (Regions C).
  - **Components:** A's panels with B's cards for editing, a user-set number of
    columns (1–4), and vertical scrolling. Isotopes appear only with their
    element (D I with H I).
  - **Fit:** split into Inspect and Results sub-tabs. Inspect has a View menu
    (all snips, one ion, or one snip) with ‹ ›, and matplotlib-style zoom and
    pan.
  - **Plot:** transitions are arranged by dragging them on a picture of the
    grid, with a "Not shown" tray.
  - **Data:** z is typed, or found by clicking a feature and choosing its
    transition; there is no redshift history.
  - **Changes to the decisions:**
    - D13: the column-role dialog opens for every file, with the roles filled
      in by the rule.
    - D14 and S25 (Q0.5): each file is one row of the dataset panel, with its
      own reference flag, FWHM, shift, zero level, file and checksum. The
      coverage chart is per file.
    - D5: the layout above.
    - S2: the transition list shows one element and ion stage at a time,
      ranked by fλ within the ion.
    - S18 (Q0.4): χ²ν and the runs test are shown separately, each with its
      own mark, using ALIS's per-pixel test unchanged; amber marks 2–3σ.
    - S24: as under Data above.
    - Imported models that break D20 or D22 (Q0.3): warn, offer a one-click
      fix, and allow the fit as imported.
    - All dashboard colours come from a colour-blind-safe palette
      (Okabe–Ito), and every coloured state also has an icon (Q0.7).

  *Review of draft 2 (2026-10-04), from RJC's comments and the responses to
  Q0.6–Q0.8. Draft 3 applies all of it.*
  - **Regions:**
    - One layout for one or several datasets: one strip per dataset, with the
      Continuum and Snip boxes showing the selected strip.
    - A System menu (primary by default) above the element and ion stage.
    - Only transitions that fall on data are listed; one with a snip stays,
      flagged.
    - SNIP and CLEAR buttons (Q0.6); "Mask brush" is renamed "Mask pixels";
      "Set z here" is removed (z is set in the Data tab).
    - The key lists one item per line.
  - **Data:** "Identify a feature" uses cascading Element → Ion → Transition
    menus, and can set the primary z or start a new system. One row per
    dataset is enough, and rows can be tied to each other (Q0.8).
  - **Components:** one column setting for the whole tab; ±200 km/s is the
    default velocity range.
  - **Fit:**
    - A Compare sub-tab next to Results (F7).
    - Inspect uses the Components column setting when it shows a group.
  - **Plot:**
    - Residuals on by default, and colour-blind-safe colours by default
      (Q0.7).
    - The written script defines every colour as a variable, and uses the
      publication style in `context/misc/matplotlibrc`, which nothing else
      in the dashboard uses.

  *Review of draft 3 (2026-10-04), from RJC's comments and the responses to
  Q0.9 and Q0.10. Draft 4 applies all of it.*
  - **No design references in the dashboard:** the shipped dashboard never
    refers to the design documents (D/F/S/QF/Q numbers). They stay in the
    design documents and in the notes around the mockups.
  - **Plot:**
    - There is no script box; users edit the written script outside the
      dashboard.
    - The publication style ships with ALIS and is the default, and another
      style file can be chosen.
    - The script uses LaTeX for text when LaTeX is installed, and
      matplotlib's own text otherwise (Q0.10).
  - **Fit, Compare:**
    - The models are viewed with the same View menu, ‹ › and Columns setting
      as Inspect.
    - A blinded parameter keeps its values and difference hidden, but shows
      the size of the change in σ (Q0.9; sign: Q0.11).
  - **Confirmed as they are:** the key under the transition list; one column
    setting for the whole Components tab; ±200 km/s; selecting a dataset by
    clicking its strip; the Results sub-tab; one row per dataset in the Data
    tab.

  *Review of draft 4 (2026-10-04), from RJC's comments and the response to
  Q0.11. Draft 5 applies all of it.*
  - **Data:** both Data pages share one layout. On the left are Systems (with
    "Identify a feature…") and Blinding, with Coverage in between when there
    are several files. On the right are the table of files and a spectrum
    view that can be zoomed and panned. Without source spectra, the view
    shows the selected dataset's snips, each scaled to its own peak.
  - **Fit:**
    - Committing a run asks for the user's own description, which can be
      edited later.
    - The χ² progress chart switches between a linear and a log axis.
    - A blinded change is shown as its size in σ, without a sign (Q0.11).
  - **Components:** the ion navigator has no explanatory text.
  - **Regions:** "click a spectrum to select it" (not "strip").

  *Review of draft 5 (2026-10-04). Draft 6, the final draft, applies it.*
  - **Velocity axes:** positive ticks are labelled with "+", negative ticks with
    "−", and 0 without a sign.
  - **Zoom toolbars:** every zoom toolbar gains "+" and "−" buttons that zoom
    along the wavelength axis only. The Compare sub-tab now has the toolbar too.
  - **Run descriptions:** they stay where they are, typed in the run history
    table (confirmed).

  *Stage 0 closed (2026-10-04). RJC confirmed the close in Prompt 7, noting that the
  design may be adjusted after feedback from users, and `dashboard_stage1.md` was
  then written.*
  - **Where the outcome is recorded:**
    - in `ALIS_v2_dashboard_prompts.md`, as decisions D34–D44, which refine D5,
      D13, D14, S2, S18, S24 and S25;
    - in `ALIS_v2_code_plan.md`, where Stage 6.2 and Query 8 now point to the
      dashboard, and a new section lists the changes the dashboard needs in
      ALIS itself.
  - **Where the mockups are:** draft 6 is in `doc/dashboard/mockups/`, drafts
    1, 2, 4 and 5 are in subfolders there, and draft 3 is kept only as version 3
    of the published page.
  - **Carried forward:**
    - **Stage 1 (changes to ALIS itself):**
      - the bundle, including several source files per dataset (Q0.5);
      - `run_alis project.model` and `--extract`;
      - the plain export;
      - in-memory loading of several data lines;
      - the shared-pixel warning;
      - removing onefits.
    - **Before Stage 3:** rewrite the `gui-dev` and `gui-component` skills for
      PySide6 and pyqtgraph.
    - **Stage 6:** matching the same physical quantity across different
      parametrisations when runs are compared (D41).
    - **With the Plot tab, after v1:** ship `alis/data/alis_publication.mplstyle`
      and extend `alis/plotscript.py` (D42).

## Skills to use for this stage

- `atomic-data`: transition wavelengths and f-values, for the coverage list (S2), the
  line IDs (S5) and the velocity axes.
- `check-fit`: summarising the reference `.mod.out` for the Fit tab.
- Claude Code's `artifact-design` skill: publishing the commentable page (0.9).
- `gui-dev` and `gui-component` are not used here. They describe `prepfit`'s Qt5 and
  matplotlib GUI, and will need rewriting for PySide6 and pyqtgraph (D2) before
  Stage 3.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the Functionality section, especially
  D1–D33; and QF.18, QF.30, QF.31 and QF.36 for the layout and the mockup plan.
- `doc/ALIS_workflow.md`: the steps that the tabs follow; §0 for snips and `prepfit`;
  §5.1 for the output columns.
- `alis/prepfit/specplot.py`: today's region selection, which the Regions tab replaces
  (D4).
- `alis/report.py`: how the fit-quality numbers behind the badges are computed.
- `context/plotting_examples/`: the reference figures for the panel style and the Plot
  tab; the `*_J1358.py` scripts use the same object.
- The two data directories named in 0.1.

## Queries

*Raised by Claude on 2026-10-04, while writing this document. Each gives Claude's
lean.*

**Q0.1 — How the mockups are drawn, and what is committed.**
- **(a)** Plots as inline SVG written by the build script with numpy, so each page is
  one self-contained HTML file that opens in any browser, offline, with no JavaScript
  library. The alternative is matplotlib images embedded in the pages: quicker to
  write, but they look less like a live application and make larger files.
- **(b)** Commit both the build script and the pages it writes. After downsampling,
  each page should be a few hundred kilobytes.

My lean: (a) inline SVG; (b) commit both.

**Response:** RJC agrees with the lean.

**Q0.2 — Window size and look.** The mockups need one window size to be judged at. I
suggest 1440×900 (a 13–14 inch laptop), checked at 1920×1080. The look would be
neutral and light, close to Qt's Fusion style, rather than a new visual design; a dark
theme can be decided in Stage 3. My lean: as suggested.

**Response:** RJC agrees with the lean.

*Raised by Claude on 2026-10-04, while building the mockups (Prompt 1). Each
comes from the real fits, and gives Claude's lean. Details are in
`claude_prompts/logs/dashboard_stage0_log.md`.*

**Q0.3 — Imported models that break isotope sharing (D22).** J1358p6522 fits D I
with its own b_turb (8.03 km/s), not tied to H I's (11.20 km/s); both have T
fixed at 0. D22 makes sharing compulsory. When such a model is opened (F4), the
dashboard could:
- **(a)** warn, and offer a one-click "Tie D I to H I", but allow the fit to run
  as imported, as for shared pixels (QF.21(c)); or
- **(b)** refuse to run until the isotopes are tied.

The same question arises for J1358p6522's O I, which has a separate free z in
each of its six lines, while D20 shares z across a component.

My lean: (a), for both. The `components_hd` and `components_a` mockups show it.

**Response:** RJC agrees with the lean.

**Q0.4 — The runs test on resampled spectra (S18).** Computed exactly as
`alis/report.py` does, 9 of J1358p6522's 12 snips are flagged only by the runs
test (runs z −3.9 to −9.6), although χ²ν is 0.65–1.12. The negative z means too
few sign changes: neighbouring residuals are correlated, as expected when a
spectrum has been resampled. Badges built on a per-pixel runs test would
therefore be red on almost every snip of such data. Options:
- **(a)** run the test on residuals averaged over one resolution element (or
  over a correlation length measured from the error spectrum);
- **(b)** keep ALIS's per-pixel test, but let the badge use χ²ν only, with the
  runs test shown in the full report;
- **(c)** leave it as is.

Either (a) or (b) would also change `out report`, so command-line fits would
behave the same way.

My lean: (a), in Stage 1 with the other changes to ALIS, keeping today's
per-pixel value in the report for comparison.

**Response:** I think leave it as is for now (show the reduced chi-squared and the runs test in the report). More information is best at this time, and the user can decide what is important to consider. 

**Q0.5 — Snips that are not in the dataset's file (F13, S23).**
`J1358p6522_fluxcal.dat` has no data between 4535 and 5593 Å, but the Lyα (H8)
and O I 1302 (O4) snips lie in that range, so they were cut from another file.
A dataset therefore cannot assume that one file holds every snip. Should a
dataset be allowed to hold several source files (for example, two instrument
settings), each with its own path and checksum, with each snip matched to the
file that contains it?

My lean: yes; it changes only the data model of Stage 2 and the dataset row
of the Data tab.

**Response:** RJC agrees with the lean. Each new file should have its own row in the dataset panel, with its own reference flag, FWHM, shift, zero level, file and checksum status (F13). The coverage chart should show which transitions each file covers.

*Raised by Claude on 2026-10-04, while making draft 2 (Prompt 2). Draft 2
already follows each lean, so a "yes" needs no further change.*

**Q0.6 — When does a transition become a snip?** You asked whether the user must
tick a box as well as clicking a transition. Draft 2 removes the tick box:
- clicking a transition only shows it in the panel;
- drawing its first fit region makes it a snip (marked ● in the list), with the
  snip's extent set around the regions and adjustable with the edge handles
  (S6);
- removing its last fit region asks whether to remove the snip as well (S27).

My lean: as described. The `regions` mockup shows it.

**Response:** RJC agrees with the lean. But, I'm also comfortable with a button that says "SNIP" so the user has something definitive they click to create a snip. The user can click the transition to view it, and then click "SNIP" to create the snip. Then, the user can have a CLEAR button that clears all drawn regions and removes it from being a SNIP. 

**Q0.7 — Colour-blind-safe colours (your comment on the badges).** Draft 2
moves every dashboard colour to the Okabe–Ito palette, and gives every
coloured state an icon as well (fit marks ✓ ! ✕, tab markers ✓ ↻ !). The
Plot tab's preview still uses the colours of your published DH figures (red
H I and green D I ticks), because the preview is what the script will draw.
- **(a)** Keep the published colours as the defaults of the plot presets,
  with the colour-blind-safe palette as a style option; or
- **(b)** make the colour-blind-safe palette the default of the presets too.

My lean: (a). The figures should look like your published ones unless you
choose otherwise; everything else in the dashboard is colour-blind-safe.

**Response:** The default should be the colour-blind-safe palette. The user can change the colours in the output plotting script, if they wish. To allow this, please specify variables in the python plotting script that is output, so that the user can change the colours of the ticks and lines in the output figure, if they wish. I have also added a matplotlibrc file to the `context/misc/` directory that contains the settings that have previously been used to generate published figures (with latex renderings). These settings should *only* be used for the output plotting script, and not for any other aspects of the dashboard.

**Q0.8 — Files that share settings (your Q0.5 response).** With one row per
file, two files from one observation (two settings of the same instrument)
would get separate FWHMs and, by D14, a free shift for the second. Should a
row's FWHM, shift and zero level be tieable to another row's, using the same
free/fixed/tied badges as the components? For example, "same as
J1358p6522_fluxcal.dat" would give one FWHM and no shift.

My lean: yes, with D14's defaults unchanged until the user ties them.

**Response:** Yes, I agree with the lean. The user can tie the FWHM, shift and zero level of one file to another file, if they wish. This will be useful for files that share settings, such as two settings of the same instrument.

*Raised by Claude on 2026-10-04, while making draft 3 (Prompt 3). Draft 3
already follows each lean.*

**Q0.9 — Blinded parameters in the Compare sub-tab.** Comparing two runs
shows each shared parameter's values and their difference. For a blinded
parameter (the D/H of J1358p6522 in the mockups), the two values are hidden
(D24), but the difference between runs would reveal how a modelling choice
moves the blinded result, which can bias those choices. Should the
difference of a blinded parameter also be hidden until unblinding?

My lean: yes, hide it, as in the `fit_compare` mockup. The models can still be
compared through the profiles, the χ² and the unblinded parameters.

**Response:** I agree that we should hide the values, however, perhaps for
blinded parameters we can show how much it changed in units of standard deviations (σ) of the uncertainty. This way, the user can see how much it changed relative to the uncertainty, without revealing the actual values. For example, if a value changed by 2.34σ, the user would see "Δ = 2.34σ" instead of the actual difference. This provides useful information for model comparison while maintaining the blinding.

**Q0.10 — Where the publication style lives.** The written plotting script
uses the style in `context/misc/matplotlibrc` (Q0.7). Since `context/` is not
part of the ALIS package, should ALIS ship that style (for example as
`alis/data/alis_publication.mplstyle`, with the script calling
`plt.style.use` on it), or should the script point to a style file the user
names?

My lean: ship it with ALIS, and let the user choose another file in the Plot
tab. Text rendered with LaTeX needs a LaTeX installation; if none is found,
the script falls back to matplotlib's own text rendering.

**Response:** I agree with the lean. The ALIS package should ship with the publication style file, and the user can choose another style file in the Plot tab if they wish. The script should check for a LaTeX installation and fall back to matplotlib's text rendering if none is found.

*Raised by Claude on 2026-10-04, while making draft 4 (Prompt 4). Draft 4
already follows the lean.*

**Q0.11 — The sign of a blinded change (your Q0.9 response).** Draft 4 shows
the change of a blinded parameter in σ, as you suggested. In J1358p6522,
log (D/H) changes by 0.4σ between run 3 (derived from N(D I) − N(H I), with
their covariance, σ = 0.011) and run 4 (fitted directly, σ = 0.011). Should the
sign be shown as well (for example "+0.4σ")? The sign says which way the
blinded value moved, which could steer modelling choices towards a preferred
answer.

My lean: show the size only, as in the `fit_compare` mockup.

Building this also showed that comparing runs must match the same physical
quantity across different parametrisations: run 3 has no D/H parameter, so
the dashboard derives it from the two column densities. The design
documents for the Fit tab (Stage 6) should cover this.

**Response:** I agree with the lean. The sign of the change should not be shown for blinded parameters, only the size of the change in units of σ. This maintains the blinding while still providing useful information about how much the parameter changed relative to its uncertainty.

## Prompts

1. Read this doc, check my responses to the queries, and ask more queries if needed. If there are no further queries, please execute the tasks in order, logging each in `ALIS/claude_prompts/logs/dashboard_stage0_log.md`.

2. Thanks for designing several great options for the dashboard layout! I have responded to several queries and made comments on the mockups. Please review my comments, ask further queries if anything is unclear. Once everything is clear, please make any changes you think are needed to the dashboard view so that I can inspect the mockups again, as a second draft.

3. Thanks for the great adjustments to the dashboard mockups! I have responded to several queries and made comments on the mockups. Please review my comments, ask further queries if anything is unclear. Once everything is clear, please make any changes you think are needed to the dashboard view so that I can inspect the mockups again, as a third draft.

4. Thanks for the great adjustments to the dashboard mockups! I have responded to several queries and made comments on the mockups. Please review my comments, ask further queries if anything is unclear. Once everything is clear, please make any changes you think are needed to the dashboard view so that I can inspect the mockups again, as a fourth draft.

5. Thanks again for the great adjustments to the dashboard mockups! I have responded to several queries and made comments on the mockups. Please review my comments, ask further queries if anything is unclear. Once everything is clear, please make any changes you think are needed to the dashboard view so that I can inspect the mockups again, as a fifth draft.

6. Thanks again for the great adjustments to the dashboard mockups! I have responded to several queries and made comments on the mockups. Please review my comments, ask further queries if anything is unclear. Once everything is clear, please make any changes you think are needed to the dashboard. My suggestions are very minor now, so once you have implemented them (and if you do not require further input from me), please update the `ALIS_v2_dashboard_prompts.md` and `ALIS_v2_code_plan.md` documents to reflect the final design decisions that we've made here. Please confirm is you are then happy to close Stage 0 and move on to Stage 1.

7. This is a great design. We may need to make some adjustments to the dashboard as I receive feedback from users, but I am happy to close Stage 0 and move on to Stage 1. Please update the `ALIS_v2_dashboard_prompts.md` and `ALIS_v2_code_plan.md` documents to reflect the final design decisions that we've made here. Then, please generate the design document for stage 1.
