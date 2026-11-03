# Prompt file for ALIS software dashboard creation -- STAGE 8

> **Feedback from users.** Stage 7 made the changes RJC found by fitting real
> problems with version 1. Stage 8 does the same for the feedback of other users:
> RJC collects each point in the Feedback section below; Claude turns the list into
> a design, queries and tasks, each with its check, and builds them in order, with a
> review before the stage closes.
>
> The rules of Stages 3–7 hold: the logic needs no Qt and goes in `alis/dashboard/`,
> tested with pytest alone; the windows go in `alis/dashboard/qt/`; everything shown
> comes from the project model through the blinding gate, and every change to the
> model is one step of the history. ALIS outside `alis/dashboard/` changes only where
> a query agrees.
>
> "D*n*", "F*n*", "S*n*" and "QF.*n*" refer to
> `claude_prompts/ALIS_v2_dashboard_prompts.md` (D1–D88 at the close of Stage 7).
> "Q0.*n*" to "Q7.*n*" refer to `dashboard_stage0.md` to `dashboard_stage7.md`. What
> follows v1 is listed in `dashboard_stage0.md` ("After v1").

## Feedback

*Collected by RJC from users of the dashboard: one point per item, numbered, added as
it arrives. For each point:*
- *what the user did and saw, and what they expected instead;*
- *its kind: **(bug)**, **(change)** or **(new)**;*
- *where it can be reproduced: a project, or the model and data, and the steps (see
  Q8.1 for where the files go);*
- *who raised it, only if they agree to be named.*

*A point already on the list is not added twice: a note goes under the first, saying
how many users raised it. Claude reads this list and does not add to it.*

1. **(new)** It would be good if the dashboard could handle different choices for the FWHM. At the moment, the dashboard only allows users to specify the vFWHM value, but some users may wish to choose a different functional form, or they may wish to load an LSF from a file. There is an example that loads an LSF from a file in the `examples/lsf_file/` directory that we could test this functionality with. Ultimately, each snip would need to have the option to choose the FWHM functional form, and the dashboard would need to be able to handle this. The most obvious place to do this is in the Regions tab, in the same place that we allow each snip to have its own shift and zerolevel. We should allow the user to choose between all the LSF options that ALIS currently has built in. In the case of a file, the user will need to be able to specify the file path, and the dashboard will need to handle this. If there are multi parameter LSFs, we should instruct the user to set the parameters in the .mod editor until this functional form is fully supported. For now, any LSF function in ALIS that has only 1 parameter, or load 1 file, should be supported in the dashboard. Everything else should be supported in the .mod editor, and we should instruct the user to use this for now. This is a new feature request, and it is not a bug. It is a change to the dashboard that would allow users to have more flexibility in their fits.

2. **(new)** When I load a `.model` file into ALIS, the dashboard does not allow me to go to the `Plot` tab (it is greyed out). Is there a way to activate the `Plot` tab once at least one fit has been made? I think it may have been set for deferred work, and now is the time to incorporate this feature. Please write a design document (stage 9) to develop this feature.

3. **(change)** In the Regions tab, the edges of the green highlighted regions that indicate the pixels included in the fit end halfway between the square pixel. It is therefore not clear from this tab if a pixel is included in the fit or not. It would be better if the edges of the green highlighted regions were aligned with the edges of the square pixels, so that it is clear which pixels are included in the fit. Every pixel that is fully highlighted within the green regions should indicate the pixels that are included in the fit. This is a change to the Regions tab, and it is not a bug. It is a change to the dashboard that would make it clearer which pixels are included in the fit.

4. **(bug)** The CI tests on GitHub are almost passing. Everything now passes except for the Mac OS GUI tests. I have copied the outputs to the following file for context: `context/misc/GitHub_gui_macos-latest_py313.txt`. Please investigate the cause of this failure and fix it, while preserving the functionality of the fixes that now work for all platforms.

5. **(bug)** If there's too many characters for the redshift (or any parameter, for that matter) in the `.mod` file, the component cards don't display the parameters correctly. I suspect the best approach is to show the leftmost characters of the number (at this stage, the dashboard shows the rightmost characters). This is a bug in the dashboard that needs to be fixed.

6. **(bug)** I have added an example `.model` file to be used as context to debug a possible bug in the dashboard. The file is located at:
`context/fitting_examples/Temperature/J101155p294141/model`
On the components tab, when I drag component `a2`, the temperature cannot be decreased to arbitrarily low values. The minimum I can get the temperature to is about 50000K. I suspect there is a bug when I converted this to a pure thermal model (perhaps there was an issue in making the turbulence zero?).

## Carried over from Stage 7

*Found by Claude during Stage 7 and not fixed there. RJC decides whether each joins
this stage (Q8.2).*

1. **(bug)** A `.mod.out` written by a plain ALIS blind run holds the line
   `------ BLIND MODEL ------`. The dashboard's parser refuses it, so such a run
   (examples/blind's reference output, say) shows no best-fit values. Runs made from
   the dashboard keep the real line, hidden in the bundle, and are not affected.
2. **(change)** Pure thermal (Task 7.5): a b_turb limit set before pure thermal was
   chosen is replaced by `[0.0,None]`. When the mode is left, the line is taken out
   rather than given back (Undo gives it back).
3. **(change)** Export's Corrected wavelength (Prompt 8): `vshiftscale` and
   `polyshift` scale about the middle pixel of the whole snip file, which can differ
   slightly from ALIS's when it loads only part of a snip.
4. **(tests)** The paths Stage 7 left untested: `label_owner` finding no component;
   `_shared_ratios`' skipped lines; and, in `export.py` (90% covered), a run still in
   progress, a run that wrote no `.mod.out`, and a snip whose file cannot be read.

## Design

*Written by Claude once the Feedback list is filled in: for each point, what changes,
in which modules, and what stays as it is. Points that need RJC's decision become
queries below, each with a lean.*

### How the feedback becomes tasks

- **Read against what was decided.** Each point is read against the decisions
  D1–D88 and the stage documents. A point that asks for what a decision ruled out
  becomes a query that quotes the decision, not a task.
- **A bug becomes a test first.** It is reproduced from the user's project where there
  is one, or from a synthetic project in `tests/dashboard_synthetic.py` otherwise. The
  test fails before the fix and passes after it.
- **Outside the dashboard.** A point about ALIS itself, or about the documentation
  (planned in `documentation_stage0.md`), becomes a query: the dashboard's stages
  change ALIS only where a query agrees.
- **Users who disagree.** Two points that ask for opposite things become one query,
  with both described.
- **Grouping and order.** Points are grouped by the part of the window they touch,
  and a bug comes before the work that builds on it, as in Stage 7.
- **Shown before it closes.** Every change a user sees is shown on a review page (as
  Stage 7's, built by `doc/dashboard/stage8/`), and RJC's comments on it are applied
  before the stage closes.

### 8.1–8.13 — The points of the list

*Written by Claude on 2026-11-02 (Prompt 1), from points 1–6 of the Feedback list,
the four items carried over from Stage 7 (Q8.2 (a)), Q8.3's response, RJC's
J101155p294141 project, `examples/lsf_file/`, the CI log in `context/misc/`, and the
code at the start of the stage. Each point's design is in its task below. Revised the
same day (Prompt 2) after RJC's responses to Q8.6–Q8.11, which agreed with every
lean.*

**What was found.** Each bug was reproduced before its task was written.
- **Point 6** (J101155p294141, a2). Two causes, neither of them pure thermal's b_turb
  of 0. With b_turb fixed at 0, `components.drag` gives T = 61 K for b = 1 km/s, as it
  should.
  - **The drag.** A drag below the continuum changes b by how much farther from the
    component's tick the pointer is (Q5.23). So the narrowest b it can reach is the
    starting b less the distance at which the drag began, and moving past the tick
    widens the line again. Off-screen, on RJC's project, a drag begun 5 km/s from
    a2's tick stops at 51,062 K, and one begun 10 km/s away stops at 34,985 K. a3's
    tick is 22 km/s from a2's, so a drag begun farther out moves a3 instead.
  - **A T label shared with other components.** a2's T carries the label `zerot`.
    So do a1's T and the T of three lines of the Lyα forest. In
    `J101155p294141_FIXDH.mod` all five were `0.0ZEROT`, fixed at 0 K. Pure thermal
    (`components.temperature`) keeps a label when all the component's lines share
    it, and frees it there. On a synthetic model of that shape, pure thermal on a2
    writes `6060.7zerot` on a1 and on the forest line too. The forest's T is then
    free, and a drag or a fit of a2's T moves all five lines. The project's
    70,168.8 K is their common best fit. Free T and the fixed presets change a shared
    label in the same way.
  - **Also found.** a3's b_turb reads `0.0BX1`, below its own `lim param BX1
    [0.01,None]`, while both runs' input models had `0.01BX1`. The pre-flight check
    therefore refuses Run for the project as saved (D81). Its note "starts below
    1 km/s, where a fit can find it hard to move" is given for four b_turb that are
    fixed, which a fit does not move.
- **Point 5.** A `QLineEdit` made with its text has its cursor at the end. A value
  wider than its field therefore shows its last characters (`qt/components.py`,
  `_value_row`). The Data tab's redshift, the Snip box's shift and the Run box's
  settings are made in the same way.
- **Point 3.** `project.mask_regions` gives each fit region from its first pixel's
  centre to its last pixel's centre. The data are drawn as steps out to each pixel's
  edges (`qt/plots.py`, `step_curve`), so the green covers half of each end pixel.
  The same spans shade the Regions tab's "other" and "shared" pixels, and the fitted
  pixels on the Components and Fit tabs.
- **Point 4.** The macOS `gui` job stops with a Bus error in pyqtgraph's
  `PlotCurveItem.paint` (its `QPainter.drawPath`). This happens 105 tests into the
  batch, while pytest-qt processes events in the teardown of
  `test_new_restarts_replace_the_kept_ones_after_a_question`
  (`test_dashboard_qt_fit_tools.py`). CI installed PySide6 6.12.0. This Mac (arm64,
  as `macos-latest`) has PySide6 6.11.2 and pyqtgraph 0.14.0, and the `gui` batch
  passes here. The Ubuntu `gui` job passes too. Not yet reproduced.
- **Point 1.** ALIS has nine functions for `resolution=`:
  - **one parameter:** `vfwhm` (FWHM in km/s), `Afwhm` (FWHM in Å) and `vsigma`
    (σ in km/s);
  - **one file:** `lsffile` (a kernel read from a text file; its one parameter only
    switches the convolution on);
  - **one parameter with keywords:** `lsf` (HST's COS and STIS through linetools, an
    optional dependency) and `apod` (an apodisation of the kind named);
  - **more parameters:** `voigtconv` (2), `multivfwhm` (3n + 1) and `lsfspline` (one
    per knot, with their `locations`).

  The dashboard writes only `vfwhm`, a dataset's FWHM on the Data tab ("FWHM
  (km/s)"). The cell shows any other function's first value as km/s, so an `Afwhm`
  of 0.042 Å reads as 0.042 km/s. A bundle refuses `lsffile` ("reads a file from
  disk", `bundle.unsupported`, Q1.5). ALIS reads the file by a path relative to where
  it runs, and with `index:-1` (its default) it picks one of the file's kernels at
  random. `examples/lsf_file/` generates its data with `lsffile(name:test_lsf.dat)`
  (a Gaussian of 0.05 Å FWHM) and fits them with `Afwhm`.
  - *Found in Task 8.8:* each function needs its own bins (`run bintype`, or a data
    line's `bintype=`): `Afwhm` Å, `apod` Hz, the others km/s.
- **Point 2.** The Plot tab is shown but disabled, by design (`qt/tabs.py`,
  `PlotTab`; D42, D77): it was left for after v1, and is Stage 9 (Q8.4, Q8.5).
- **Item 1 carried over.** `examples/blind`'s `.mod.out` holds
  `------ BLIND MODEL ------` in place of its blinded Si II line. `runs.Run.parsed`
  refuses it, so Fit · Results shows no best-fit value for the run.
- **Q8.3's response.** The pre-flight check refuses a model with no line in its
  absorption section (`preflight._nothing_to_fit`, D85). A snip with no fitted pixel
  is only marked (`markers.py`); it is not refused.
- **Done before this prompt** (2026-11-01, the log): RJC's report of shifts and zero
  levels missing from Fit · Results. The Instrument group is open by default, a
  closed group says how many rows it hides, and shifts are named by their snips.

**What stays as it is.**
- ALIS outside `alis/dashboard/`, except that a bundle can run `lsffile` (Q8.7, Task
  8.7);
- the bundle's format, except an LSF file kept as one of its files (Q8.7), and the
  snip format (D4);
- the decisions D1–D88, except where a query refines one: D14 and D83 (Q8.6, Q8.8),
  D21 and D81 (Q8.10, Q8.11), D63 (Q8.9), D85 (Q8.3).

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/dashboard_stage8_log.md`.
> After every task, run the `unit` batch, the dashboard tests and the `gui` tests; run
> the `fast` batch before the stage closes. A task that changes ALIS outside
> `alis/dashboard/` also runs the regression harness, its `test_bundle` mode included.

*One task per point of the Feedback list (8.1, 8.2, …), and per item carried over
from Stage 7 that joins the stage, written by Claude once the list is filled in. Each
has:*
- *what it builds or fixes, naming its point;*
- ***Check:** the test that fails before the change and passes after it (for a bug,
  the bug made into a test first), and what else is run.*

*Tasks 8.1–8.14 were written by Claude on 2026-11-02 (Prompt 1), and revised the same
day (Prompt 2) after RJC's responses to Q8.6–Q8.11. Every response took the lean, so
the tasks stand as written, but for two changes: 8.8 tells a user who chooses `lsf`
how to install linetools (Q8.8's response), and 8.10 no longer waits on ALIS, which
was found to fit a continuum alone. Each task names its point of the list, or its
item carried over. The order puts a bug before the work that builds on it, and keeps
work on one part of the window together:*
- *CI first (8.1), so that every later task is checked on both platforms;*
- *the Components tab: the cards' values, the temperature modes and the drag
  (8.2–8.5);*
- *the Regions tab: the shaded pixels, then the resolution of a snip (8.6–8.8);*
- *the Fit tab and Export (8.9–8.11);*
- *the tests left by Stage 7 (8.12);*
- *the Plot tab's design document last (8.13), before the close.*

**8.1 — The macOS `gui` job (point 4, bug; the window tests, CI).**
- **The bug.** The macOS `gui` job stops with a Bus error in pyqtgraph's curve
  painting during a test's teardown, with PySide6 6.12.0 (see Design). The Ubuntu
  job passes, and so does this Mac, with 6.11.2.
- **First, reproduce it.** On this Mac, in a scratch environment with CI's versions:
  PySide6 6.12.0, pytest-qt 4.5.0, and the pyqtgraph that
  `pip install -e ".[gui,gui-test,dev]"` chooses. Run the failing test alone, then
  its file, then the whole batch. Then find which curve is being painted, and why its
  data or its window are no longer valid by then: a window or dialog the test leaves
  open, say, or a curve whose arrays are replaced while a paint is pending.
- **The fix** goes at the cause, in the dashboard or in the tests' teardown
  (`dashboard_qt_helpers.py`), so that it holds on every platform. The fixes of
  Stage 7 stay as they are: the fitting process's standard input (Q7.17), the
  `linetools` marker (Q7.19), the pinned lint (Q7.20) and
  `--skip-machine-dependent` on Ubuntu (Q7.21).
- **If the fault is Qt's own** (a few lines with no dashboard code crash too), the
  remedy becomes a query, with what was found. It might be a bound on PySide6's
  version, on macOS or everywhere, or a change to how the tests paint.
- **Check.**
  - Made into a test first where it can be: the crash is reproduced here with 6.12.0,
    and gone after the fix. The `gui` batch passes here with 6.12.0 and with 6.11.2.
  - On CI, the `gui`, `unit` and `examples` jobs pass on both platforms. RJC pushes,
    and saves the logs in `context/misc/` if any job fails.

*Paused on 2026-11-02 (Prompt 3): the crash does not reproduce here, even with CI's
versions and order (Q8.12).*

*Done on 2026-11-03 (Prompt 3), as Q8.12's response asks. RJC's next push passed the
macOS `gui` job with no change, so the crash is set aside and watched for; if it comes
back, the scratch environments of Q8.12 are the place to start. The same push failed
the `lint` job (`context/misc/GitHub_lint.txt`), and that is fixed here. The first push
of a new branch has an all-zeros `github.event.before`. The step checked its base with
`git rev-parse --verify`, which accepts any well-formed 40-digit name without looking it
up, so `git diff` then stopped with "bad object". The check is now
`git rev-parse --verify --quiet "$BASE^{commit}"`, which looks the commit up, so a new
branch skips lint as the step meant it to. Each later push lints its changed files as
before. Checked by running the step's shell on four bases: all zeros and empty both
skip; this branch's first commit finds no changed Python files; Stage 7's closing
commit lists the files changed since.*

*Prompt 4 (2026-11-03): RJC's push of Tasks 8.1–8.7 failed the `gui` job on both
platforms, at one test of Task 8.6. CI's pytest 9.2.0 refuses a list of pairs in
`pytest.approx`, which this machine's 9.0.2 accepts. The test now compares a flat list,
as does one more found by running the `gui` batch under 9.2.0 here (it needs
`context/`, so CI skips it). The window tests are now run here under pytest 9.2.0, in
a scratch environment, as well as under the conda env's.*

**8.2 — A card's value shows its start (point 5, bug; `qt/components.py`,
`qt/widgets.py`).**
- **The bug.** A value longer than its field shows its last characters, so a long
  redshift loses its leading digits.
- **The fix.** Every value field shows its first characters. Its cursor goes to the
  start when the field is filled, and again when it loses the focus. One helper in
  `qt/widgets.py` does this for the cards, the Data tab's redshift, the Snip box's
  shift and the Run box's settings. A field's tooltip gives the whole value, masked
  while it is hidden (D24).
- **Check.** Made into a `gui` test first (`test_dashboard_qt_components.py`). A
  component's z is written with 16 digits, as in RJC's J101155p294141
  (2.5007852870610208), on a card narrower than its text.
  - The field's cursor is at its start, and the text shown begins "2.50078".
  - After a new value is typed and the field is left, the field shows its start
    again.
  - The same holds for the Data tab's redshift and the Snip box's shift.
  - A hidden value is still masked, in its field and in its tooltip.

**8.3 — A temperature mode changes its component alone (point 6, bug;
`components.py`; D21, D81; Q8.10).**
- **The bug.** A component's temperature mode (a fixed preset, Free T or pure
  thermal) writes to its T and b_turb labels. When another component or a line of
  the forest shares one of those labels, the mode changes them too. Pure thermal on
  a2 of J101155p294141 freed the forest's `ZEROT` (see Design).
- **The fix (Q8.10).** Before the mode is set, a T or b_turb label that lines outside
  the component share is replaced on the component's lines by a new label of the
  scheme, as Task 7.2 gives a redshift label.
  - The label is new: no line uses it, commented-out lines included
    (`edit.used_labels`).
  - The other lines keep the old label and their values. The step's preview names
    them ("The T of a1 and of 3 lines of the Lyα forest keep ZEROT, fixed at 0 K").
  - A label shared only within the component (by its ions, D20) is kept.
- RJC's project keeps its text. Untie on a2's T badge already gives a2's T a label of
  its own.
- **Check.** Made into tests first, in `test_dashboard_components.py`, on a synthetic
  model of J101155p294141's shape: two H I components and a forest line, all
  `0.0ZEROT`.
  - Pure thermal on a2 gives a2 a new T label, free, with its `lim param …
    [1.0,None]`. a1's and the forest line's `0.0ZEROT` are unchanged, byte for byte,
    and ALIS reads the model.
  - The same holds for Free T and a fixed preset, and for a shared b_turb label.
  - A T label shared by the component's own ions (H I and D I) stays one label.
  - One undo restores every byte.
  - RJC's project (skipped when absent): Untie on a2's T, then pure thermal, leaves
    a1 and the forest lines as written.

**8.4 — A drag narrows a line down to its bound (point 6, bug; `qt/components.py`;
D63, D81; Q8.9).**
- **The bug.** A drag across can narrow a line only by the distance at which it
  began, and widens it again past the tick (see Design). On RJC's project, a T
  dragged from 70,169 K stops near 50,000 K.
- **The fix (Q8.9).**
  - Moving the pointer toward the component narrows the line, and moving it away
    widens it, at 1 km/s of b per km/s, as now. The side is set where the drag
    begins.
  - Past the tick the line keeps narrowing, until b_turb or T reaches its bound
    (`components.bounds`: voigt.py's own, the `lim` lines, T's 1 K under pure
    thermal).
  - The readout says when the value is at its bound ("T 1 K, at its limit").
- **Check.** Made into tests first, in `test_dashboard_qt_components.py`.
  - A synthetic pure-thermal H I has b_turb fixed at 0 and T free at 70,000 K. A
    drag begun 5 km/s from its tick and ended 40 km/s past it writes T at its bound
    of 1 K, as one step. A drag ended at the tick narrows b by 5 km/s, as now.
  - A drag away from the tick widens by the distance moved, on either side.
    `test_a_drag_is_one_step` stands as written.
  - A turbulent line's b_turb stops at 0.5 km/s, voigt.py's own.
  - RJC's project (skipped when absent): a drag of a2 begun 5 km/s from its tick and
    ended 20 km/s past it writes T near 5,000 K.

**8.5 — Pure thermal gives back the limit it replaced (item 2 carried over, change;
`components.py`, `preflight.py`; D81; Q8.11).**
- **Now.** Pure thermal sets the lower bound of an existing `lim param` of b_turb to
  0.0. Leaving the mode takes the line out, so the user's own limit is lost; only
  Undo gives it back.
- **The change (Q8.11).**
  - Pure thermal writes the earlier limit as a comment on the line it changes:
    `lim param BX1 [0.0,None]   # pure thermal; was [0.01,None]`.
  - Leaving the mode gives the line back as it was, without the comment.
  - A line that pure thermal wrote itself has no such comment, and is taken out as
    now.
  - T's `[1.0,None]` is handled in the same way.
- **Found on the way** (see Design). The pre-flight check's note "starts below
  1 km/s, where a fit can find it hard to move" is given only for a free b_turb.
- **Check.** Made into tests first.
  - In `test_dashboard_components.py`, with `lim param BX1 [0.01,None]` written:
    - pure thermal, then Free T, gives the line back byte for byte;
    - so does pure thermal, then a fixed preset;
    - a line with no earlier limit is taken out, as now;
    - each undo restores every byte, and ALIS reads every model, comment included.
  - In `test_dashboard_preflight.py`: a b_turb fixed at 0.3 km/s gives no note, and a
    free one at 0.3 km/s does.

**8.6 — Fit regions shaded to the pixels' edges (point 3, change; `qt/plots.py`,
`project.py`).**
- **Now.** A fit region is shaded from its first pixel's centre to its last pixel's
  centre, so half of each end pixel's step lies outside the green.
- **The change.** Every shaded run of pixels goes from its first pixel's left edge to
  its last pixel's right edge, where `step_curve` draws them: half way to the
  neighbours, and at a gap as far as the pixel's other edge. A pixel is fitted
  exactly when its whole step is green.
  - One helper, beside `step_curve`, gives the edges of a run of pixels.
  - It is used wherever pixels are shaded: the Regions tab's fit, other and shared
    pixels; the Components tab's panels; Inspect and Compare; and the residual
    strip's pixels not fitted.
  - The regions as the model holds them (`mask_regions`, the fit range) do not
    change.
- **Dragging an edge** (Tweak dataset regions, `regionMoved`) still takes the pixels
  whose centres lie inside it. The edge then settles on a pixel's edge, so the
  shading always shows what is fitted.
- **Check.** Made into tests first.
  - Without Qt: the helper's edges, for runs at the start and end of a snip, of one
    pixel, and beside a gap, on a uniform and a non-uniform grid, equal
    `step_curve`'s.
  - In `test_dashboard_qt_regions.py`, on a synthetic snip:
    - each fit region's shading ends at the step edges of its first and last pixels,
      to 10⁻⁹ Å;
    - a pixel half outside a region is neither fitted nor shaded;
    - an edge dragged into half of a pixel settles on that pixel's edge, and the
      mask follows the pixel's centre.
  - The same check on the Components and Fit tabs' panels (velocity axes), and on
    the residual strip.

**8.7 — An LSF file a project can run (point 1, new; in ALIS, `bundle.py`,
`functions/lsffile.py`; in the dashboard, `export.py`; Q8.7).**
- **Now.** A bundle refuses `lsffile`, since it would read a file from disk (Q1.5).
  The dashboard therefore cannot fit a model that uses one.
- **The change (Q8.7).** The LSF file goes into the project, as the snips do.
  - **In the bundle.** The file is stored as one of the bundle's files
    (`files/lsf/<name>`), by name and checksum, and the data line names it as the
    snips are named. `bundle.unsupported` refuses `lsffile` only when its file is
    not in the bundle.
  - **In ALIS.** `lsffile` takes its kernel from the bundle in memory, as
    `load_data` takes the snips (D9). In a plain `.mod`, it reads the file from disk,
    as now.
  - **In the dashboard.** Its drawing of the model (D60), its first guess of log N
    and its buffer check use the same kernel. Where they need a width, it is the
    kernel's FWHM (`components.fwhm_kms`).
  - **Several kernels.** When a file holds its kernels in several columns, the
    dashboard writes `index:` (the first column by default), so that a fit gives the
    same result every time. Outside the dashboard, ALIS's own default (−1, a kernel
    at random) stays as it is.
  - **Export** (D88). Everything run_alis needs writes the file beside the snips,
    and the data line names it there, so `run_alis` runs the export as it runs the
    bundle. `run_alis --extract` does the same.
- **Check.** Made into tests first.
  - In ALIS (`tests/test_bundle.py`): `examples/lsf_file`'s data and
    `generate_spectra.mod`'s `lsffile(name:test_lsf.dat)`, packed into a bundle,
    give the same best fit as the plain `.mod` run beside the file.
  - A bundle whose `lsffile` names a file it does not hold is still refused, with
    the reason.
  - A file of two kernels is fitted with the kernel that `index:` names, the same in
    every run.
  - In the dashboard (`test_dashboard_export.py`): the export, run with `run_alis`,
    gives the bundle's result.
  - The regression harness, with its `test_bundle` mode, since ALIS outside the
    dashboard changes.

*Done on 2026-11-03 (Prompt 3); see the log. Two things were found in ALIS's
`lsffile`, and neither was changed, since Q8.7 agreed to no more: it stops under
`run bintype A`, so the tests fit `examples/lsf_file` with ALIS's default km/s bins;
and `getminmax` turns the kernel's width into a velocity at a fixed 1309 Å, so at
longer wavelengths it asks for more buffer than it needs. Both are for RJC at the
close.*

**8.8 — A snip's own resolution (point 1, new; `model.py`, `edit.py`, `project.py`,
`datasets.py`, `snips.py`, `qt/regions.py`, `qt/data.py`; D14, D35, D83; Q8.6,
Q8.8).**
- **Now.** Every snip's resolution is its dataset's FWHM, a `vfwhm` set on the Data
  tab. Any other function can be written only in the `.mod` panel.
- **The change (Q8.6, Q8.8).** The Snip box gains Resolution, beside Shift and Zero
  level: As the dataset (the default) or Its own. Its own overrides the dataset's
  resolution for that snip alone (Q8.6's response). It offers ALIS's functions in
  one menu:
  - `vfwhm`, `Afwhm` and `vsigma`: the value is typed with its unit (km/s, Å, km/s)
    and badged (free, fixed, tied), as the dataset's FWHM is;
  - `lsffile`: a file chosen in a dialog, copied into the project (Task 8.7). The box
    names it, with its kernel's FWHM in km/s. A file of several kernels offers the
    column. A file that cannot be read is refused, with the reason;
  - `lsf` and `apod` (Q8.8): their keywords typed in one field as ALIS writes them,
    starting from ALIS's defaults (`name:COS,grating:G130M,life_position:1,`
    `cen_wave:1309`; `kind:uniform`), and checked as ALIS reads them. `lsf` needs
    linetools (see below);
  - `voigtconv`, `multivfwhm` and `lsfspline`: listed but greyed, with the note
    "<name> has several parameters: set them in the .mod panel". A snip whose model
    already uses one shows its name, the same note, and a link that selects its line
    in the `.mod` panel (D68). Its own resolution is then kept as written.

  Each change is one step. Back to As the dataset restores the dataset's word.
- **`lsf` without linetools** (Q8.8's response). The user is told what `lsf` needs
  before it can fail them:
  - the menu marks it "needs linetools" where linetools is not installed;
  - choosing it asks first: "lsf needs linetools, which is not installed here.
    Install it with `pip install linetools`, then open the dashboard again." The
    user can choose it anyway, or cancel;
  - while it is chosen and linetools is missing, the Snip box shows the same note;
    the pre-flight check refuses Run with it; and the panels show it in place of the
    snip's model, never a traceback;
  - a model that already uses `lsf` (Load a .mod…, Open a fit (.mod)…, the `.mod`
    panel) gets the same note.
- **The dataset (Q8.6).** The Data tab's FWHM column becomes Resolution.
  - Its cell shows the function with its unit: "6.285 km/s", "0.042 Å", "σ 7.0
    km/s", "file test_lsf.dat".
  - The same menu sets the dataset's function, with Free, Fixed, Tie and Reset as
    now.
  - SNIP gives a new snip its dataset's resolution.
  - The cell says how many snips have their own, as for the shift (Task 7.13).
  - The Add a spectrum dialog keeps its FWHM in km/s (D46). Another function is
    chosen on the Data tab once the file is added.
- **What is read** (as Task 7.13 reads a shift). A resolution on one data line that
  the row's other lines do not share is that snip's own, whatever wrote it. A
  project's rows are kept in `ui/project.json`, so a snip's own label never splits
  its row. An imported fit with no rows recorded still groups its data lines by their
  resolution label (Q2.4), as now.
- **What follows from it.** Whatever needs a width takes it from each snip's own
  function: the first guess of log N (±½ FWHM), the buffer marker, and the line
  IDs' joining of close lines. A function with no simple width (`lsf`, `apod`, the
  multi-parameter ones) counts as two pixels, as now.
- **Check.** Made into tests first (`test_dashboard_datasets.py`,
  `test_dashboard_snips.py`, `test_dashboard_qt_regions.py`,
  `test_dashboard_qt_data.py`).
  - `examples/lsf_file`: a project of `OI_SiII.dat`, whose snip is given `lsffile`
    with `test_lsf.dat` in the Snip box, is run from the window. Its log N of O I and
    Si II agree with `fit_spectra.mod.out.reference`'s within 3σ.
  - Each of `vfwhm`, `Afwhm`, `vsigma`, `lsf` and `apod` gives a model ALIS reads,
    with the word on that data line alone. Back to As the dataset gives the text
    before, byte for byte, and every undo is exact.
  - A synthetic snip made with an FWHM in Å, given a free `Afwhm`, fits it back
    within 3σ.
  - The multi-parameter functions are greyed. A model with `voigtconv` on one snip
    is read with the note, and its link selects the line.
  - Each function set as a dataset's Resolution reaches every snip that takes the
    dataset's, and SNIP gives a new snip that function.
  - `examples/lsf_hst`'s model, loaded with Load a .mod… (skipped without
    linetools), reads `lsf` and its keywords as its snip's resolution.
  - With linetools made absent (in the test, wherever it is installed): the menu's
    mark, the question and its install line, the Snip box's note, the pre-flight
    refusal, and the panels' note in place of the model. The same holds for
    `examples/lsf_hst`'s model loaded with Load a .mod….
  - A hidden resolution value is masked in the box and in the cell
    (`test_dashboard_qt_blinding.py`).

*Done on 2026-11-03 (Prompt 4); see the log. One thing the task did not foresee: ALIS
needs bins that suit each function (`Afwhm` Å, `apod` Hz, the others km/s), and stops
otherwise. The dashboard writes `bintype=` on a data line whose function needs bins
other than the model's `run bintype`, and SNIP gives a new snip its dataset's.
`examples/lsf_file` was fitted from the window through `lsffile`, and then with a free
`Afwhm`, which gave back the kernel's 0.05 Å.*

**8.9 — A blind run's `.mod.out` (item 1 carried over, bug; `runs.py`, `model.py`).**
- **The bug.** A `.mod.out` written by a plain ALIS run with blinded lines holds
  `------ BLIND MODEL ------` in place of each, and the dashboard's parser refuses
  it (see Design). Such a run, opened with its fit (D12), shows no best-fit value.
- **The fix.** The parser reads the line as the hidden line it replaces, in the
  order of the input model.
  - The other lines' values are read as usual.
  - The hidden line's values are masked wherever a best fit is shown (Results,
    Compare), and its errors are shown where the `.mod.out` lists them (D24).
  - Inspect's Final cannot draw a line it has no values for. It draws nothing for
    the run, and says why.
- **Check.** Made into tests first.
  - In `test_dashboard_runs.py`: `examples/blind`'s reference `.mod.out` is read.
    The continuum's values are as written. O I's values are masked (offset by its
    blind range) and Si II's hidden.
  - In `test_dashboard_qt_blinding.py`: `examples/blind`, opened with its fit and
    its `.mod.out`, shows the run in Fit · Results without an error, and no hidden
    value in the window's text.

*Done on 2026-11-03 (Prompt 4); see the log. Results keeps such a run, its hidden rows
masked with their errors, and Final says why it draws nothing.*

**8.10 — A continuum alone can be fitted (Q8.3 (b), change; `preflight.py`,
`qt/fit.py`; D85).**
- **Now.** Run, Save and Run and Convergence check… are greyed while the model has
  no component (D85). A snip with no absorption line, or a continuum checked before
  components are added, cannot be fitted from the window.
- **The change (Q8.3 (b)).** The pre-flight check refuses:
  - a project with no snip, as now;
  - a project none of whose snips has a fitted pixel (until now only marked);
  - a model with no component only when none of its continuum's parameters is free.
    The message says so ("Nothing is free to fit: add a component, or free the
    continuum").

  Results, Inspect and Compare show a run with no component: the continuum's rows,
  the fit statistics and the marks.
- **ALIS needs no change.** ALIS fits a continuum alone. This was checked on
  2026-11-02 (Prompt 2) with metal_line_abs's Legendre over 1301.0–1301.6 Å: χ²
  fell from 47.2 to 41.7, the same with an empty `absorption` header and without
  one.
- **Check.** Made into tests first.
  - In `test_dashboard_preflight.py`: a snip with a free Legendre and no absorption
    line passes. With the continuum fixed, it is refused with the message. A project
    whose snips have no fitted pixel is refused.
  - In `test_dashboard_qt_fit.py`, on metal_line_abs with no component: Run is
    enabled, the run completes, and Results lists the continuum's coefficients and
    χ²ν. `test_run_is_greyed_while_there_is_nothing_to_fit` and
    `test_run_waits_for_a_component` are rewritten to the new rule.

*Done on 2026-11-03 (Prompt 4); see the log. Results, Inspect and Compare needed no
change; Compare, which never lists a continuum (D41), shows the statistics and the
FWHM.*

**8.11 — Export's corrected wavelength on ALIS's own pixels (item 3 carried over,
change; `export.py`).**
- **Now.** `vshiftscale` and `polyshift` scale about the middle pixel of the array
  they are given. Export gives them the snip's whole file. ALIS gives them the pixels
  it loads: the fit range with its buffer, or `loadrange`. Where the two differ, the
  corrected wavelength is slightly off.
- **The change.** Every pixel of the file is shifted about the reference ALIS uses:
  the middle of the pixels it loads for the snip, as the drawn model has them
  (`profile.py`, D60). If ALIS's shift functions cannot take the reference from
  outside, `export.py` computes it as they do, in one place, tested against them.
- **Check.** Made into a test first (`test_dashboard_export.py`). A synthetic snip is
  loaded in part (a fit range in its middle, with a buffer), with `vshiftscale` and
  with `polyshift`.
  - On the pixels ALIS loads, the corrected wavelength equals ALIS's own shifted
    wavelength, to 10⁻¹⁰ Å.
  - Beyond them, it follows the same formula.
  - `vshift` and `Ashift`, which have no reference pixel, are unchanged.

*Done on 2026-11-03 (Prompt 4); see the log. ALIS's reference is the middle of the
sub-pixel grid it shifts; `export.scale_shift` computes the two functions about it,
tested against `shift.py`.*

**8.12 — The paths Stage 7 left untested (item 4 carried over, tests).**
- **The tests,** for:
  - `components.label_owner` finding no component, and `_shared_ratios`' skipped
    lines;
  - in `export.py` (90% covered): a run still in progress, a run that wrote no
    `.mod.out`, and a snip whose file cannot be read. Each tells the user what
    happened, never with a traceback (D86).
- A fault that a test finds is fixed in the same task, and logged.
- **Check.** `test-coverage` on `components.py` and `export.py`: those lines are
  covered, and `export.py` is above 90%. The figures go in the log.

*Done on 2026-11-03 (Prompt 4); see the log. One fault found and fixed: Export skipped
a snip whose file cannot be read without a word, and then stopped with "ALIS cannot
evaluate the model"; it now stops first, naming the snip.*

**8.13 — The Plot tab's design document (point 2, new; `dashboard_stage9.md`; D28,
D42; Q8.4, Q8.5).**
- **The aim.** Point 2 asks for the Plot tab, enabled once a fit has been made, and
  for its design to be written as Stage 9's document. Stage 9 is the Plot tab, and
  feedback that arrives after this list closed goes to Stage 10 (Q8.4, Q8.5). The tab
  is built in Stage 9. This task writes its document, as Task 7.14 wrote the
  documentation's.
- **The document,** `claude_prompts/dashboard_stage9.md`, covers:
  - **what exists:** D28, D42, S22 and the Plot tab's mockup (Stage 0);
    `alis/plotscript.py`, which writes a plotting script from a model; RJC's scripts
    in `context/plotting_examples/` (D28's `context/plotting/`): the Lyman series of
    D/H, metals, blends and model fits; and `context/misc/matplotlibrc`, the shipped
    style's source (D42);
  - **when the tab is enabled** (point 2): once the project has a run, or on its
    starting model too, and what it shows when the shown run changes;
  - **the layout, the panels and the presets** (metals, DH, blends and helium, D28),
    and what each preset takes from the examples;
  - **the script:** what it writes, how it is run outside the dashboard, and how the
    script itself draws the preview (D42, S22);
  - **blinding:** what a figure of a blinded project may show (D24);
  - **the tasks** of Stage 9, each with its check, and queries with leans for RJC.
- **Check.**
  - A table in the document matches every script in `context/plotting_examples/` to
    a preset, or says why it needs none.
  - Every element of D28, D42 and S22 has its place in the document's tasks.
  - No code changes.

*Done on 2026-11-03 (Prompt 4): `claude_prompts/dashboard_stage9.md`, with eight
draft tasks and Q9.1–Q9.6 for RJC; see the log.*

**8.14 — Close the stage.**
- **The review.** Publish the screenshots of what changed for RJC's review, on a new
  page with the earlier pages' stylesheet, and apply the review. The page shows:
  - the cards' values (8.2);
  - a temperature mode's preview on a shared label, a drag down to 1 K, and a limit
    given back (8.3–8.5);
  - the shaded pixels on each tab (8.6);
  - the Snip box's Resolution with each kind of function, the Data tab's Resolution
    column, and `examples/lsf_file` fitted with its LSF file (8.7, 8.8);
  - a blind run's results, and a continuum fitted alone (8.9, 8.10);
  - Fit · Results' Instrument group (RJC's report of 2026-11-01).
- **The documents.** Bring `doc/ALIS_workflow.md` §4.3, `CHANGELOG.md` (with ALIS's
  change for `lsffile` in a bundle, Task 8.7), `tests/README.md` and the
  `gui-component` skill up to date.
- **The decisions.** Record them in `ALIS_v2_dashboard_prompts.md` (D89 onwards, from
  Q8.1–Q8.11 and the review), saying which earlier decisions they refine: D14, D21,
  D63, D81, D83, D85, and Q1.5's refusal of `lsffile`. Update
  `dashboard_stage0.md`: Stage 8 done, Stage 9 the Plot tab (8.13's document), Stage
  10 the feedback that arrived after the list closed (Q8.4), and the "After v1"
  list.
- **Check.**
  - Every point of the list has its task logged as done with its check passing, or
    its query answered.
  - The `unit`, `gui` and `fast` batches pass. Their counts go in the log, against
    the start of the stage (the log of 2026-11-01): 2090 unit tests passed with 119
    skipped, and 212 `gui` tests passed; and the `fast` batch of Stage 7's close.
  - CI passes on both platforms (Task 8.1).
  - `test-coverage` runs on what changed. The untested paths go in the log, and the
    most valuable tests are added.
  - The window's text has no design reference (D44).
  - Task 8.7 changes ALIS outside `alis/dashboard/` (Q8.7), so the regression
    harness runs again at the close, with its `test_bundle` mode.

*Draft 1 of the review on 2026-11-03 (Prompt 4):
https://claude.ai/artifact/Qkycw3e1TQR6oD8HGP5TY9. The documents and D89–D99 are
written. Batches under CI's pytest 9.2.0: `unit` 2128 passed, 120 skipped (2090 and
119 at the start of the stage); `gui` 229 passed (212); `fast` 149 passed (143 at Stage
7's close), the regression harness and its `test_bundle` mode included. Waiting on
your push (CI), your comments on the page, and Q8.13.*

## Skills to use for this stage

- `gui-component`: any panel or widget changed or added. It documents the project
  model, the history, the gate, the spectrum view and the Fit tab's runs.
- `gui-dev`: driving the window off-screen, and the screenshots at 1440×900.
- `run-tests`: the `unit`, dashboard and `gui` tests after each task, and the `fast`
  batch at the end.
- `gen-tests` and `test-coverage`: the tests of what changed.
- `check-fit`: where a point touches a fit's results.
- Claude Code's `artifact-design` skill: the review page.

## Context

- `claude_prompts/ALIS_v2_dashboard_prompts.md`: the decisions D1–D88.
- `claude_prompts/dashboard_stage7.md` and its log: how the changes from real use
  were handled, and the review and its comments (the Export dialog, D88).
- `claude_prompts/deferred_work.md` §7: what was deferred from the dashboard's stages.
- `doc/ALIS_workflow.md` §4.3: the dashboard as a user meets it.
- The review pages of the earlier stages (`stage-review-pages` in Claude's memory).
  Stage 7's builder is `doc/dashboard/stage7/`.
- `tests/dashboard_synthetic.py`: the synthetic fits a bug's test can be built on.
- The users' files, where the Feedback list names them (Q8.1).

## Queries

*Raised by Claude with this document. Each gives Claude's lean.*

*Q8.1–Q8.5 were raised on 2026-11-01 (Stage 7's Prompt 10).*

**Q8.1 — Where a user's files go.** A point is easiest to fix with the user's project,
or its model and data. Two ways:
- **(a) In `context/feedback/`.** One folder per point, named by its number
  (`context/feedback/3/`), not committed, as `context/fitting_examples/` is not. Users'
  data stay out of the repository, and each test that needs them is skipped where
  they are absent.
- **(b) Described only.** The point says what was done, and Claude builds a synthetic
  project for each test.

My lean: (a), with (b) for a point that comes with no files.

**Response:** For now, I have referenced the location of the `.model` files in the Feedback section.

**Q8.2 — The items carried over from Stage 7.** Three ways:
- **(a) All four join this stage**, after the feedback's tasks.
- **(b) Only the bug (1)** joins. The other three go to `deferred_work.md` §7.
- **(c) None join.** All four go to `deferred_work.md` §7.

My lean: (a). They are small, and items 1 and 2 are known faults.

**Response:** I agree with option (a).

**Q8.3 — A continuum alone.** Run is greyed while the model has no component (your
review of Stage 7, D85). So a user who fits a continuum alone, to check it before
adding components, cannot run from the window. Two ways:
- **(a) Keep it**, as you asked in the review.
- **(b) Allow a continuum alone.** Run is still greyed with no snip or no fit region,
  but not for a model with no component when a continuum parameter is free.

My lean: (a), unless a user's feedback asks for (b).

**Response:** I think (b) is the correct approach. Sometimes, a snip may not have an absorption line, and the user may wish to simply fit a continuum only.

**Q8.4 — When the stage starts and closes.** Feedback arrives over time. Two ways:
- **(a) A closed list.** Stage 8 starts once you judge the list long enough. It
  designs, builds and closes as Stage 7 did. Feedback that arrives later goes to
  Stage 9's document, written at the close, as this one was.
- **(b) An open stage.** Stage 8 stays open and builds points in batches as they
  arrive, with a review for each batch.

My lean: (a). A closed list gives one order, one review and a close.

**Response:** I agree with option (a). It is better to have a closed list for this stage, and any new feedback can be addressed in Stage 10 (Stage 9 will be the Plot tab).

**Q8.5 — The rest of "After v1".** The Plot tab, the items marked "Later", the
documentation and the deferred work are listed in `dashboard_stage0.md` ("After v1").
Two ways:
- **(a) Stage 8 is the feedback alone.** The Plot tab and the rest get stages of their
  own, in the order you give the list.
- **(b) Stage 8 also starts the Plot tab.**

My lean: (a). A stage with one source keeps its review readable.

**Response:** I agree with option (a). I mentioned the Plot tab in the Feedback section, but it is better to have a separate stage for that feature.

*Q8.6–Q8.11 were raised on 2026-11-02 (Prompt 1), from the Feedback list (Design,
"What was found"). RJC answered them the same day (Prompt 2), taking every lean.*

**Q8.6 — Where a resolution is chosen (point 1, Task 8.8).** Point 1 asks for a
resolution function per snip, in the Snip box. D14 has one FWHM per dataset, "which
can be freed or overridden per snip". Two ways:
- **(a) Per snip only.** The Snip box's Its own offers every function. The Data tab's
  FWHM stays a `vfwhm` in km/s.
- **(b) Per snip and per dataset.** The Data tab's FWHM column becomes Resolution,
  with the same choice of function, and a new snip takes its dataset's. A dataset of
  COS spectra, say, is then set once rather than snip by snip.

My lean: (b).

**Response:** I agree with option (b). However, we should also allow the user to set the resolution per snip as well, in case they want to override the dataset's resolution for a specific snip.

*Applied (Prompt 2): Task 8.8 keeps both. The Data tab sets a dataset's resolution,
and the Snip box's Its own overrides it for one snip.*

**Q8.7 — The LSF file (point 1, Tasks 8.7 and 8.8).** A bundle refuses `lsffile`
today, since it reads a file from disk (Q1.5). Two ways:
- **(a) In the project.** The file is copied into the bundle, as the snips are, and
  ALIS's `lsffile` reads it from there, in memory. The project stays one file that
  runs on any machine (D1, D8, D9). This changes ALIS outside the dashboard
  (`bundle.py`, `functions/lsffile.py`), so the regression harness runs.
- **(b) On disk.** The bundle records the file's path and checksum, as it does a
  source spectrum's (D8), and a run reads it from there. The change is smaller, but a
  project copied to another machine runs only with the file at the same path.

My lean: (a).

**Response:** I agree with option (a). It is better to have the LSF file included in the project so that it can be run on any machine without needing to worry about file paths. 

**Q8.8 — `lsf` and `apod` (point 1, Task 8.8).** Point 1 asks for every function of
one parameter or one file in the dashboard. `lsf` (HST's COS and STIS, through
linetools) and `apod` have one parameter, but they need keywords. `lsf` needs the
instrument, the grating, and either `cen_wave` and `life_position` (COS) or `slit`
(STIS). `apod` needs the kind of apodisation. Two ways:
- **(a) In the Snip box,** with one field for the keywords, written as ALIS writes
  them and checked as ALIS reads them. `lsf` is greyed where linetools is not
  installed.
- **(b) In the `.mod` panel,** with the functions of several parameters, until they
  are fully supported.

My lean: (a). Point 1 counts parameters, and the field asks no more of the user than
the `.mod` panel does.

**Response:** I agree with option (a). Let's also make sure that the user is made aware that they need to install linetools for the `lsf` function to work, if that is the option they choose.

*Applied (Prompt 2): in Task 8.8, `lsf` is no longer greyed where linetools is
missing. It is marked "needs linetools", and choosing it asks first, with the install
line (`pip install linetools`). While it is chosen, the Snip box, the pre-flight
check and the panels repeat the note, and a loaded model that uses `lsf` gets it
too.*

**Q8.9 — What a drag across moves (point 6, Task 8.4).** Q5.23 has a drag change b
by how much farther from the component the pointer is. A drag begun close to the
tick therefore cannot narrow the line much: on RJC's project, a drag begun 5 km/s
from a2's tick stops at 51,062 K. Three ways:
- **(a) Toward narrows, away widens.** The side is set where the drag begins. Moving
  on past the tick keeps narrowing, until b_turb or T reaches its bound. The rate
  stays at 1 km/s of b per km/s.
- **(b) In proportion.** b is scaled by the pointer's distance from the tick over the
  distance at which the drag began, so halfway to the tick halves b. This allows fine
  narrowing, but a drag begun close to the tick is very sensitive.
- **(c) Keep it as it is,** and say in the status bar how far a drag can narrow.

My lean: (a). Widening keeps the feel of Q5.23, and any T down to its limit can be
reached from either side of the line.

**Response:** I agree with option (a).

**Q8.10 — A temperature mode on a shared label (point 6, Task 8.3).** A component's
T or b_turb label may be shared with other components, or with the forest. RJC's a2
shared `ZEROT` with a1 and three lines of the forest, and pure thermal freed it for
all five. Two ways:
- **(a) The component alone.** The component's lines take a new label first, and the
  other lines keep theirs, unchanged. The preview names them. A tie the user wants
  back is one Tie to….
- **(b) Ties are followed.** A fixed shared label (`ZEROT`) is split as in (a), since
  it is a shorthand for one value. A free shared label is a tie the user made, so
  every component on it follows the mode, and the preview names them.

My lean: (a). A card's menu changes its own component, and the preview says what
stays.

**Response:** I agree with option (a).

**Q8.11 — The limit pure thermal replaces (item 2 carried over, Task 8.5).** To give
the limit back, the dashboard has to keep it while the mode is on. Two ways:
- **(a) A comment on the line:** `lim param BX1 [0.0,None]   # pure thermal; was
  [0.01,None]`. It stays with the model (D7), in exports and in a run's copy of the
  input, and it is read back when the mode is left. If the comment is edited away in
  the `.mod` panel, the line is taken out, as now.
- **(b) In `ui/project.json`,** by the label. The model text stays as ALIS would write
  it, but an edit of the line in the `.mod` panel can leave the record out of date.

My lean: (a).

**Response:** I agree with option (a).

*Q8.12 was raised on 2026-11-02 (Prompt 3), during Task 8.1. Development is paused
until it is answered.*

**Q8.12 — The macOS crash, not reproduced here (point 4, Task 8.1).** What was
tried, on this Mac (arm64, like `macos-latest`), with the tests in CI's order and the
23 tests CI skipped left out (they need `context/`):
- PySide6 6.12.0, as CI installed it, over this machine's other packages;
- the same with numpy 2.5.4 (released on 10 October), which CI would also install;
- an environment built as CI builds its own, from PyPI alone: Python 3.13.18, numpy
  2.5.4, PySide6 6.12.0, pyqtgraph 0.14.0, pytest 9.2.0, pytest-qt 4.5.0.

All 86 tests up to and including the crashing file pass each time. What was ruled
out:
- pyqtgraph builds its curves by writing into Qt's own memory, but 300 random paths,
  some with gaps, were built correctly under 6.12.0;
- the window's only worker thread (the `.mod` panel's full check) never touches Qt;
- the restarts chart draws markers only, with no line, so the crashing curve is
  another of the window's own.

The crash came in the first event-processing pass after the test body, before any
fixture teardown, so the window was alive and painting. PySide6 6.12.0 was released
on 8 October, after the last stage that changed the `gui` job (Stage 3, 6 October).
Whether the macOS job ever passed with 6.11 is not known here. Three ways:
- **(a) One diagnostic push first.** The macOS `gui` job gains a second entry, the
  same but with `PySide6==6.11.2`. You push, and send the logs if either fails:
  - if 6.11.2 passes and 6.12.0 fails again at the same test, the trigger is
    PySide6 6.12. The job then keeps the bound `PySide6<6.12` on macOS, with an
    entry in `deferred_work.md` §7 to lift it and a check against each new PySide6;
  - if both fail, the fault is ours. A second traceback narrows it, and Task 8.1
    goes on from there;
  - if both pass, it was intermittent, and the job runs each test file in its own
    process, so that one fault cannot spread through the batch.
- **(b) Bound PySide6 now,** `PySide6<6.12` for the macOS `gui` job only, with the
  same entry in `deferred_work.md`. This is quicker, but the cause stays unknown.
- **(c) One process per test file** on macOS, without the bound.

My lean: (a). One push separates PySide6 from our code, and you keep the 6.12.0
entry's result either way.

**Response:** I pushed changes to this document (that shouldn't have affected the Mac os gui tests) and the latest results pass. So, maybe it ewas just a glitch on GitHub. For now, let's ignore it, and see if it shows up again at some point. If it does, we can investigate further. I did get a problem with the lint tests (I've put the output in `GitHub_lint.txt`), so we will need to investigate this before closing this stage.

*Q8.13 was raised on 2026-11-03 (Prompt 4), at the close (Task 8.14). The stage's
review page waits on your comments as well.*

**Q8.13 — Two things found in ALIS's resolution functions (Tasks 8.7, 8.8).** Neither
was changed, since Q8.7 agreed to no more than an LSF file in a bundle:
- **`lsffile`'s buffer.** `getminmax` turns the kernel's width into a velocity at a
  fixed 1309 Å, so at longer wavelengths it asks for more buffer than a snip needs
  (and at shorter, less). The dashboard's buffer marker and the validator follow it.
- **`apod`'s buffer.** `getminmax` reads apod's value as a FWHM in Hz, so on a
  spectrum in Å it asks for about 130 Å of buffer (metal_line_abs's Si II snip), and
  the Snip box marks the buffer too narrow.

(Task 8.7's other finding, that `lsffile` stops under `run bintype A`, is ALIS's rule
that each function needs its own bins, D92; the dashboard now writes them.) Two ways:
- **(a) Fix both in ALIS**, in Stage 10 or with the documentation: `lsffile` takes the
  wavelength from the snip's data, and `apod` asks for a buffer in the bins it is given.
  The regression harness runs.
- **(b) Leave them**, with an entry in `deferred_work.md` §7.

My lean: (a), for `lsffile` at least, since the dashboard now offers it; `apod` is for
spectra in frequency, so its buffer may be right for the data it was written for.

## Prompts

1. Please read the list of changes and bug fixes, and the work of Stages 0–7 (their documents and logs). I have drafted a list of bugs, improvements, and additions to the dashboard (see Section `Feedback`). Please review these suggestions, and if anything is unclear, ask in the Queries section, and I will respond; when everything is clear, please write an ordered list of tasks. Please write all of the tasks, each with a description of what it builds or fixes, and the check that will verify it. Each task should be numbered sequentially (7.2, 7.3, etc.). The final Task (currently labelled 8.N) should be to close the stage, and should include the checks that will verify that the stage is complete.

2. I have responded to your queries. Please review my responses, and if you have any further queries, please ask them in the Queries section. When everything is clear, please write an ordered list of tasks. Each task should be numbered sequentially (8.1, 8.2, etc.), and should include a description of what it builds or fixes, and the check that will verify it. The final Task (currently labelled 8.N) should be to close the stage, and should include the checks that will verify that the stage is complete. Please note that we can continue this thread provided we are below 75% context usage. If we exceed that, we will need to start a new thread to continue the discussion.

3. Please execute the tasks in order. There's no need to pause between tasks, unless you have queries that require my thoughts and responses. If you have queries during development, please pause the development and log the queries and responses in the Queries section.

4. Please read this document, and the associated log (`dashboard_stage8_log.md`), together with the work of Stages 0–7 (their documents and logs) to understand the work that has been implemented so far. I have pushed recent changes to the code based on Tasks 8.1-8.7. The GitHub lint tests now pass. However, the Mac OS gui and Ubunutu gui tests fail (see the output as described in `GitHub_gui_macos-latest_py313.txt` and `GitHub_gui_Ubuntu-latest_py313.txt` respectively). It is the same error. Please look into these failures as part of Stage 8. Also, please continue with the development of Stage 8, starting with Task 8.8, and continue to execute all remaining tasks of this stage. Please ensure that all tasks are completed in order, and that all checks are performed to verify the completion of each task. If you encounter any issues or have any questions during development, please log them in the Queries section for my review and response.
