# Dashboard Stage 2 log

The project model, with no Qt: the text-sync layer, opening an existing fit, the
validator, the blinding gate, undo/redo, removal with dependents, and the modes. The
plan is in `claude_prompts/dashboard_stage2.md`.

### 2026-10-05 (Prompt 1: queries, and Tasks 2.1–2.2)

**Reading.** The code plan, the dashboard prompts (D1–D44, the QF queries), the Stage 0
and Stage 1 documents and logs, and the code Stage 2 builds on: `load.load_input`,
`load_data`, `load_model` and `load_links`; `functions/base.py` and `voigt.py`
(the per-line loaders, the ratio syntax `ion=2H_I/1H_I`); `bundle.py`; `logger.py`;
the round-trip and bundle tests; and the mockup loaders.

**Queries.** RJC's responses to Q2.1–Q2.9 were read. Claude's answers to the notes on
Q2.2–Q2.5 and Q2.7 were added to the document, and three new queries were asked in the
session; RJC took every lean:
- **Q2.2 follow-up:** the user preferences file (outside the bundle) is built in
  Stage 3;
- **Q2.10:** a `fitrange=[lo,hi]` line keeps that form while it has one region, and
  switches to a mask column when it needs a second;
- **Q2.11:** unblinding switches blinding off in the text, is logged, and cannot be
  undone.

The Design section was updated to follow them, and to say how Q2.5 is done: a handler
on the `alis` logger collects ALIS's warnings and errors during the full check, and an
unexpected exception prints its traceback to the terminal.

**Baseline.** `pytest -m unit`: 857 passed, 85 skipped, 0 failed (46 s).

**2.1 The package.**
- `alis/dashboard/` with the nine modules of Q2.1, each with a docstring, and a package
  docstring that describes the layers.
- `pyproject.toml`: a `gui` extra (PySide6, qtpy, pyqtgraph), unused in this stage.
  `hypothesis` was added to the `dev` extra for the property tests; it was already
  installed here.
- Test: `tests/test_dashboard_no_qt.py` imports every module of the package (except a
  future `qt/` subpackage) in a fresh interpreter and fails if `qtpy`, PySide6, PyQt6,
  PyQt5, PySide2 or pyqtgraph is loaded. A second test checks that the same probe does
  see `qtpy` when it is imported (qtpy and PyQt6 are installed here), so the guard
  cannot pass vacuously.

**2.2 The model text (`text.py`).**
- `ModelText` splits a model into lines (`\r\n`, `\n` or `\r`, as universal newlines
  read a model file) and words, and classifies each line exactly as `load_input` does:
  blank, comment, long comment, hidden placeholder, setting, block marker, data line,
  section, `fix`, `lim`, model function, or link. It reports structural problems: a
  missing `data end` or `model end`, a block that is never opened, `<--#` before
  `#-->`, two markers on one line, and text after `<--#` (which ALIS reads wrongly).
- Every word is a `Token` with its line and columns. `split_value` splits `5.0da` as
  ALIS's `check_tied_param` does (so `1e4` is the value 1 with the label `e4`);
  `split_keyword` and `function_args` split `resolution=vfwhm(6.974va)`.
- A `Patch` replaces a span and has an exact inverse; a `Change` is several patches in
  order. Patches replace a word (keeping the next word's column, Q2.6), a line, or a
  range of lines (insertion and deletion, including after a last line that has no
  ending). `Patch.map_line` follows a line through an edit, for hidden lines and
  selections. `layout_like` lays out a new line in the columns of a neighbour.
- Tests (`tests/test_dashboard_text.py`, 288 with the import test):
  - all 126 `.mod` and `.mod.out.reference` files come back byte for byte;
  - for each of them that ALIS reads on its own, the settings, data, model and link
    lines are exactly `load_input`'s four lists;
  - property tests (hypothesis): any word patch changes only its own span and its own
    line, and any sequence of word, line, insert and delete patches is undone exactly.

**2.3 The parsed model (`model.py`).**
- **How it agrees with ALIS.** The data lines, settings, model lines, `fix`/`lim`
  commands and links are read from `text.py`'s words. What the parameters *mean* is
  then left to ALIS: `load_model`, `load_links` and `load_parinfo` are run on fresh
  function instances (a `fix` command changes the instance it is applied to), with
  `_specid`, `_resn` and `_shft` made from the data lines as `load_data` makes them.
  Their tables are matched to the words, so that every `Param` knows its value and
  label tokens, its ALIS number, whether it defines the value, and whether it is
  fixed, linked or limited. Agreement with ALIS holds by construction; the test
  checks the matching. Timing: 0.03 s for J1358p6522, 0.1 s for Q1243p307, 0.4 s for
  the 351-line DH_orders model (`build_funcarray` costs 0.3 ms).
- **ALIS, quietly.** `alis_quietly()` puts a filter on the `alis` logger that
  collects the messages of the current thread (other threads pass) and turns the
  exit of `msgs.error` into `AlisStop`. This is how Q2.5 is done, with no change to
  ALIS.
- **Problems on their lines.** Checks that ALIS makes before it reads (unknown
  settings, data-line keywords, fitranges, unknown functions and ions, `fix`/`lim`
  forms, link forms) are made from the words. An error inside `load_model` is placed
  by reading the model functions one at a time, as its first pass does; other
  messages by the line or the name they quote. A warning that quotes nothing (`blind
  should be of type boolean`) is placed the same way. An unknown ion is an error,
  but does not stop the reading: ALIS reads `DH/J1558m0031_dlaonly`, whose `26Al_II`
  is not in the atomic table, and fails only when it fits.
- **The label table.** Each label's places, the `fix param`/`lim param` commands and
  links that name it, and its state: free, tied (free, in several places), fixed or
  linked.
- **Found:** ALIS reads a value with `float(word.rstrip(label))`, which strips the
  label's *characters*. `11.2878481n1a` is therefore read as `11.287848`: the label's
  `1` also takes the value's last digit. In all the models this happens twice, both
  in `examples/CNabs/model/fit_spectra.mod.out.reference` (so re-reading that output
  loses a digit). The parsed model keeps ALIS's reading as `value` and what is written
  as `written`. Raised as a query (fixing it changes ALIS).
- **Tests** (`tests/test_dashboard_model.py`, 118 pass): for 92 models (every model
  and reference output that ALIS reads on its own with its data from memory), the
  parsed model agrees with ALIS's full reading (`load_input`, `load_data` from memory,
  `load_model`, `load_links`, `load_parinfo`) on `p0`, `tpar`, `mtie`, `mfix` and
  `mtyp`, the number of free parameters (neither fixed nor linked), each label's
  places, and each parameter's line (`modpass['line']`); and every parameter is
  matched to the right word. 34 are skipped: 23 data generators, 5 Temperature
  models whose data are not in the folder, `J1558m0031_zerolevel` (ALIS does not load
  it), and 5 blind reference outputs. Unit tests cover labels and states, named and
  positional parameters, the ratio syntax, positions, and a problem of each kind on
  its line.
- The `unit` batch: 1263 passed, 0 failed.

**2.4 Targeted edits (`edit.py`).** Every operation takes a parsed model and returns a
`Change` to the text.
- **Values:** `set_value` writes a value's digits only, in every place of its label
  by default (ALIS uses the first; the validator warns when they differ), keeping the
  next word's column. `format_value` writes the shortest exact form, always with a
  decimal point and a signed exponent, and gives an unsigned exponent typed by the user
  a sign (ALIS reads `1e4ta` as the value 1 with the label `e4ta`).
- **Free, fixed, tied (S11):** `fix` and `free` change the label's case, renaming it
  everywhere it is used (its places, `fix param`/`lim param`, links). When a
  function-wide `fix` (such as `fix voigt temperature True` in `examples/blind`)
  overrides the case, the change also writes `fix param <label> True/False`; this is
  found by reading the result again. `free` on a parameter that ALIS fixes because it
  is not written (voigt's `damping`) writes it out. `tie` gives one place another
  label and that label's value; `untie` removes the label, or asks for one if the
  parameter is fixed.
- **Labels (Q2.3):** `new_component` gives `za1`, `ba1`, `ta1` (the first number none
  of the three uses), `new_row_labels` gives `fwhm<n>` and `vs<n>`, and an upper-case
  label is fixed. A parameter with no label that has to be fixed, tied or limited
  raises `LabelNeeded`, with a suggestion from the scheme, or none for a column
  density (RJC, Q2.3). `check_label` refuses labels that are not identifiers, and
  `e`/`E` followed by digits.
- **Limits (S15):** `set_limits` writes or updates `lim param <label> [lo,hi]` after the
  model's other commands; `clear_limits` removes it.
- **Keywords:** `set_keyword` replaces a value or adds `name=value` after the line's last
  word, with the line's own gap; `remove_keyword` removes it with the space before it.
- **New lines:** `add_ion` (after the component's last line), `add_continuum`,
  `add_zero_level` and `add_data_line`, each laid out in the columns of the nearest line
  of its kind. A missing section (J1358p6522 has no `zerolevel`) is made, in ALIS's
  order.
- **Datasets (D35):** `set_fwhm`, `set_shift` (adding `shift=vshift(...)` with a row
  label where a line has none), `set_zero_level` (None switches it off), and `tie_rows`
  for the FWHM, the shift and the zero level.
- **Tests** (`tests/test_dashboard_edit.py`, 38 pass) on `examples/metal_line_abs`,
  `examples/blind`, J1358p6522, Q1243p307 and the D/H J1358p6522 model. Each test
  checks, by a line diff, that only the lines it should have changed did, and reads
  the result with the parsed model, i.e. with ALIS's own loaders, checking the
  structure intended (labels, states, limits, specids, layout columns).
- The `unit` batch: 1301 passed, 0 failed.

**2.5 The project (`project.py`).**
- **State and steps.** A `Project` holds a bundle in memory: the model text with its
  hidden lines restored, the files tree, `ui/project.json`, and the lines hidden besides
  those with `blind=True` (for S20). Every change is a `Step` (a text `Change`, files as
  `(old, new)` bytes, `ui/project.json` as `(old, new)`, hidden lines as `(old, new)`)
  with an exact inverse and a `then` that joins steps (for the history). Hidden lines
  follow edits through `Patch.map_line`. After each step the text is read again and
  the concepts are rebuilt; a text that does not read keeps the last concepts and
  pauses panel edits (`require_reading`, QF.3).
- **Opening.** `Project.open` reads a bundle; `Project.import_model` packs a plain
  `.mod` with `bundle.pack` (F4). `to_bundle`/`save` hide the hidden lines again
  (`bundle.hide_lines`, numbered as `pack` numbers them, so an unchanged project gives
  the same bytes) and keep the members ALIS and the project do not use.
- **Rows (D35):** from `ui/project.json`, else inferred (Q2.4): data lines that share a
  resolution label, then one row per file stem; named by the lines' `label=`, else
  the stem. The reference row is the first with no shift or a shift fixed at 0.
- **Systems and components (D15, D20):** components are the voigt lines that share a
  redshift label, joined by untied lines within 0.1 km/s; systems group components
  within 500 km/s, the one with most ions being primary; `1Ly_a` and `1H_IB` are
  generic absorbers; a ratio line (`2H_I/1H_I`) joins every component of its
  denominator ion that shares a snip. Each component has its ions, isotopes (D22) and
  temperature mode (D21: preset, fixed, thermal, free or linked).
- **Notices:** the general notice that the structure was inferred (RJC, Q2.4); an
  "untied" notice for a component whose lines do not share its redshift label (D20);
  an "isotopes" notice for isotopes that do not share z, b_turb and T (D22). Each fix
  is one change: `tie_component` writes the component's label (or a new one of the
  scheme) on every line, with the first line's value; `tie_isotope` gives the main
  isotope's parameter a label of the scheme where it has none and ties the others to
  it.
- **Regions (Q2.7, Q2.10):** read from the mask column, the `fitrange=[lo,hi]` token,
  or the whole snip (`all`). `set_regions` rewrites only the mask characters of the
  lines whose mask changes; on a `[lo,hi]` line it edits the token for one region and,
  for more, adds a mask column to the file (the other columns' text unchanged) and
  switches the line to `fitrange=columns` with `fitrange:<n>` in `columns=`. A file
  read by other data lines too is copied first. `set_snip_edges` cuts a snip from its
  own lines; moving an edge outwards needs the source spectrum (Stage 4).
- **Q2.9:** `lines_of(item)` and `items_at(line)` map snips, rows, components, systems
  and notices to their `.mod` lines and back (S16); `model_changed_since_run` compares
  the stored model's SHA-256 with `runs/latest/run.json` (F12).
- **What the two mockup fits give:**
  - J1358p6522: one system (z = 3.0672596), components at −7.2 and 0 km/s, the main
    one with H I, D I and O I (D I its isotope, six O I lines), T preset at 0 K,
    57 generic absorbers, one row of 12 snips; notices: inferred, O I's redshift
    untied (D20), D I's b_turb not H I's (D22).
  - Q1243p307: the rows HIRES (23 snips, reference), PROCHASKA (16) and KIRKMAN (12),
    with their labels `vh`, `vp`, `vk`, KIRKMAN's shift `SHK` and three zero levels;
    the primary system at z = 2.5257 with 18 components, and four others at
    z = 2.05, 2.18, 2.40 and 2.44 (as the mockups show); five D22 notices (D I's T or
    b_turb not H I's in five components).
- **Tests** (`tests/test_dashboard_project.py`, 19 pass): the two fits as above, with
  both fixes applied and checked; a small model that breaks D20 (O I and Si II at one
  untied z) and D22 (³He and ⁴He with different b_turb), each fixed; regions on a
  masked snip (only the mask characters change; undo restores the bytes), on a
  `[lo,hi]` line (one region, then two, checked by ALIS's own `load_data` on the edited
  bundle in memory), and on a shared file; snip edges; confirming the structure and
  reopening the saved bundle; the blind line hidden again on writing back (the same
  bytes as `bundle.pack`); the model-changed test; pausing; hidden lines followed
  through an edit and its undo.
- The `unit` batch: 1320 passed, 0 failed.

**2.6 The validator (`validate.py`, F5).**
- **`quick(pm, data)`**, at every keystroke, from the parsed model and the snips'
  pixels: the text's structural problems and the parsed model's own (unknown settings,
  functions, ions, data-line keywords, fitranges, `fix`/`lim` and link forms, and
  ALIS's error when it stops); a `1e4`-style label, found from the words even where
  ALIS has stopped earlier; a value ALIS reads differently from how it is written (the
  `rstrip` quirk of Task 2.3); a model line naming a specid with no data line; a
  specid that no emission model covers; places of one label written with different
  values (warning on each later place); a fitted range with no pixels (a
  `[lo,hi]` with no pixel in it, or an all-zero mask); a buffer narrower than the
  convolution needs (warning), from ALIS's own `getminmax`, as ALIS checks
  `bufferpix`. The pixel checks are skipped for a model that generates its data.
  Problems are de-duplicated against ALIS's own messages and put in line order. A
  quick check of J1358p6522 takes 0.04 s, of Q1243p307 0.09 s.
- **`full(text, data, registry)`**, after a pause: ALIS's own `load_input`,
  `load_data` (data in memory), `load_model`, `load_links`, `load_parinfo`,
  `load_par_influence` and `load_subpixels`, run in `alis_quietly`. This finds what
  only a fit would otherwise find, such as an isotope missing from the atomic data
  (`set_vars`). Each message is placed by the data file or specid it names, the line
  it quotes, or the ion whose isotope it names. ALIS's "For a blind analysis ..."
  warnings are dropped (a bundle keeps its outputs, hidden). `bundle.unsupported`
  gives a warning for what the dashboard cannot run (generated data, `lsffile`, ...).
  J1358p6522 takes 0.06 s, Q1243p307 0.2 s.
- **Unexpected errors (RJC, Q2.5):** any other exception is written to the terminal
  with its traceback, through `msgs.bug`, and reported as "The ALIS dashboard has
  encountered an unexpected error ... The terminal window shows the details."
- **Found:** ALIS's `Afwhm.getminmax` scales the fitted range by
  `1 ± Nsig·σ`, with σ in Ångström, as if it were a fraction: for FWHM 0.1 Å it asks
  for ±40% of the wavelength (63,655 km/s for `examples/lsf_file`). ALIS only loads
  more data than it needs, but the validator would have warned falsely, so it uses the
  additive width for `Afwhm`. Raised with the query of Task 2.3.
- **Tests** (`tests/test_dashboard_validate.py`, 66 pass): each problem of F5's list
  made in a small model and found on its line (structure: a long comment closed before
  it opens, a missing `model end`; an unknown function and ion; a `1e3` label; a
  specid with no data line; a specid with no continuum; a fitrange and a mask with no
  pixels; tied labels with different values; a buffer too narrow on one side); a
  misread value; the full check placing a missing data file and an unknown isotope;
  ALIS's messages kept off the terminal; an unexpected error with its traceback on the
  terminal; every one of the 47 models in `examples/` (fits and generators) gives no
  errors; J1358p6522 as a project gives no errors and its shared-pixel warnings each
  on a line. No message names a design document (checked by pattern in every test).
- The `unit` batch: 1386 passed, 0 failed.

**2.7 The blinding gate (`blinding.py`, F8, D24, QF.20).**
- **What is hidden.** The values of the lines with `blind=True` and of the lines
  hidden by S20. Hiding follows the model: a label whose value is set on a hidden line
  is hidden wherever it is used; a parameter computed by a link from a hidden one is
  hidden (`dhtie(dhrand) = dhrand` hides `dhtie` once `dhrand` is blinded); so is what
  is derived from a hidden value (a component's velocity from its redshift, a hidden
  label's `lim param`/`fix param` values). On a hidden line every value is masked, and
  every number-valued keyword; `blindrange` and `blindseed` are masked everywhere.
- **`Gate`.** `value` (starting values), `word` (`▒▒▒▒da`), `best_fit` (masked under
  global blind, `run blind True`, as well, QF.20(a)), `error` and `chi2_change` (shown),
  `sigma_change` (a hidden parameter's change between runs as its size in σ, without
  sign, Q0.9/Q0.11), `keyword`, `velocity`, `redshift`, `system_z`, and `message`/
  `problem`, which mask the numbers of a message that mentions a hidden value, label or
  line (keeping its line numbers).
- **The `.mod` panel.** `Gate.view()` gives a `MaskedView`: the text with each hidden
  value replaced by `▒▒▒▒`, and `to_real`, which turns an edit of the panel's text into
  an edit of the real text. An edit that touches a mask replaces the whole value with
  what is typed, which stays hidden because its line still is (QF.20(c)); deleting
  part of a mask with nothing typed changes nothing.
- **`unblind(project, confirmed=True, note=...)`** refuses without confirmation; writes
  `blind=False` and `run blind False`; empties the extra hidden lines; calls
  `bundle.unblind`, which restores the hidden lines and records the time and note in
  the manifest; logs it through `msgs`; and clears the history (Q2.11).
- **Found while testing:** an inferred system took its redshift from the component
  with most ions, which might be a hidden one, and every other component's velocity
  would then reveal it. An inferred system now takes its redshift from a component
  whose redshift is not hidden, when it has one (`project.py`).
- **`strings(project, gate, problems)`** lists every string the panels show of a
  project's values through the gate: the `.mod` panel, the parameter tables, tooltips,
  best-fit and error cells, the rows, systems, component cards (z, velocity, N, b, T,
  temperature mode), keywords, notices and problems.
- **Tests** (`tests/test_dashboard_blinding.py`, 9 pass):
  - for `examples/blind` (its Si II line given distinctive values, 13.4729 and
    z = 0.000173) and for `DH/J1358p6522_original` (with `blind=True` and the value
    −4.5873 on `dhrand`, set by the dashboard's own edits), every rendering of every
    hidden value (as written, `repr`, and 3–8 significant figures; and the hidden
    component's velocity, 51.86 km/s) that no visible value shares is searched for in
    every string, including the validator's problems: none is found, while the
    visible values are shown;
  - `blindrange`/`blindseed` are masked; messages are masked but keep line numbers;
  - the hidden line is hidden again when the project is saved (no member of the zip
    holds the value; the model member has `<hidden:1>`; reopening restores it);
  - typing over a mask stores the typed value, still masked; an edit elsewhere maps to
    the right place of the real text;
  - global blind shows starting values and errors, and masks best-fit values;
  - unblinding is refused without confirmation, then logged and final.
- The `unit` batch: 1395 passed, 0 failed.

**2.8 History (`history.py`, F2).**
- A `History` makes and keeps the steps of one project: `do(step)`, `edit(change)`
  (a panel's edit), `type(change)` (typing in the `.mod` panel), `undo`, `redo`, their
  descriptions for the menus, and `clear` (after unblinding).
- Typing steps that follow each other within `pause` seconds (1 s) are joined into
  one; a pause starts a new step. Everything done inside `with history.group(...)`
  (a drag) is one step; groups may be nested. A new step forgets what was undone. At
  most 1000 steps are kept. Nothing is saved in the bundle (Q2.8).
- Steps are joined with `Step.then`: the text changes in order, and the first old and
  last new value of each file, of `ui/project.json` and of the hidden lines, so the
  joined step's inverse is exact.
- **Tests** (`tests/test_dashboard_history.py`, 7 pass): one step undone and redone;
  typing grouped by pause; a drag of four value changes as one step; a new step
  clearing the redo list; a region edit (a snip's file), a confirmed structure
  (`ui/project.json`) and a notice's fix on J1358p6522, all undone to the original;
  and a hypothesis property test: any sequence of up to ten operations (panel edits,
  typing, a fitrange region, two regions that add a mask column to the file,
  `ui/project.json`, hidden lines, drags, a text that stops reading, and undo and redo
  between them), undone in full, gives back the original text, files, project data
  and hidden lines, byte for byte, and redone gives the same result again. 40 examples
  run every time; 300 were run once, and passed.
- Hypothesis keeps a cache in `.hypothesis/` at the repository root even with its
  example database switched off (`database=None` is set in the tests). `.hypothesis/`
  was added to `.gitignore`, beside the other test caches.
- The `unit` batch: 1402 passed, 0 failed.

**2.9 Removal (`remove.py`, S27, D23).**
- `plan_component`, `plan_ion`, `plan_snip`, `plan_system` and `plan_dataset` each make a
  `Plan`: the lines to delete and to edit (each with its reason, its text now and after),
  the snip files that leave the bundle, the new `ui/project.json` (a system or row
  removed from it), notes, and the change to the text. `Plan.step(project)` makes it
  one undoable step; `Plan.preview(gate)` gives its lines through the blinding gate.
- What goes is followed to a fixed point: a data line's specid, when no other data line
  has it; the specids dropped from other lines, and a line left with none deleted
  (continua, zero levels, blends); `fix param`/`lim param` lines and links that use a
  label no parameter uses any more; a ratio line (`2H_I/1H_I`) with no line of its
  denominator left on its snips; a `variable` whose links are all removed. Removing an
  ion given as a ratio removes it from every component it applies to, and the plan
  says which.
- Notes: a tie left with one member, and a label whose starting value moves to another
  line with a different value. Labels are not renamed or removed.
- A snip file leaves the bundle unless a data line that stays names it, or a
  commented-out data line in the data block does (it can be switched back on, as
  `bundle.pack` keeps it). The commented copy of an older input at the end of
  Q1243p307's model names every snip, and must not keep them.
- **Tests** (`tests/test_dashboard_remove.py`, 8 pass); after each removal ALIS's full
  set of loaders (the validator's full check, with the data in memory) reads the model
  without error, and undoing it gives back the text, the files and
  `ui/project.json`:
  - D/H J1358p6522: removing D I (a ratio line for three H I components) deletes the
    ratio line, the link `dhtie(dhrand) = dhrand` and the `variable` `dhrand`, and keeps
    the commented-out `random` line;
  - Q1243p307: removing KIRKMAN deletes its 12 data lines (with their shift `SHK` and
    resolution `vk`), its 12 continua and its zero level, drops its specids from the
    voigt lines, and removes its 12 snip files;
  - a component, a snip (with its continuum and the blends on it only), a system, a
    system of a confirmed `ui/project.json`, the notes of a tie left with one member,
    and the preview of a plan through the gate.
- The `unit` batch: 1410 passed, 0 failed.

**2.10 Modes (`modes.py`, F11).**
- `Mode` is the interface: `load` (the data loader), `display` (the display spectra),
  `regions_to_mask` (regions to fitted data), `new_project` (the model template) and
  `plot_presets`. `VoigtMode` is the only mode (`get("voigt")`); Orders mode will be a
  second subclass (D29–D32).
- **Loader (D13, S3):** `column_roles` gives the dialog's starting roles: wave, flux,
  error; a fourth column of 0s and 1s is a mask, any other a continuum. A continuum is
  multiplied into the flux and error, with a note (QF.7).
- **Display and regions:** each dataset is its own display spectrum; a region maps
  straight onto its snip's mask (`apply_regions` uses `Project.set_regions`).
- **Template:** `new_project(source, z, [(ion, rest)], fwhm=..., window=300,
  region=60, coldens=13.5, bturb=10)` cuts one snip per transition (±300 km/s, in
  `prepfit`'s four-column format at full precision, with the ±60 km/s fit region as
  its mask), and writes a model laid out as ALIS's own: the settings (a fixed list until
  the preferences file of Stage 3), a data line per snip with a fixed FWHM
  (`vfwhm(<fwhm>FWHM1)`, D14), a Legendre continuum per snip starting at the median
  flux outside the fit region (S8 replaces it in Stage 4), and a voigt line per ion
  with the labels of component `a1` (`za1`, `ba1`, `TA1`: T fixed at 0 K, D21). The
  bundle has the default atomic table, the source recorded by path and checksum, and
  `ui/project.json` (mode, the primary system and its line IDs, the reference row and
  its column roles), so the project has no notices.
- **Plot presets** (D28): metals, DH, blends and helium, with their grids.
- **Tests** (`tests/test_dashboard_modes.py`, 7 pass): the column roles; a continuum
  multiplied in; regions to masks; the presets; a new project read as intended
  (labels, fixed FWHM and T, a ±60 km/s region to within a pixel, the source's
  checksum, `ui/project.json`, no problems from the validator); transitions off the
  spectrum and unknown ions refused; and two runs of ALIS on a new project, as
  `run_alis project.model` runs it: from `J1358p6522_fluxcal.dat` with O I 1039 at
  z = 3.0672596 (12 iterations, χ² from 3974 to 210; the O I line has Lyα blends that
  this one-component model lacks), and from a three-column spectrum made from
  `examples/metal_line_abs` with O I 1302 at z = 0 (so that CI runs it too). These two
  run a fit, so they are marked `fast` rather than `unit`.
- While trying the template, a script run from standard input (`python -`) hung: ALIS's
  worker processes are spawned (macOS), and cannot import `<stdin>`. This is how
  Python's multiprocessing works, not a dashboard problem; the tests run from files.
- The `unit` batch: 1415 passed, 0 failed.

**2.11 Closing the stage.**
- **Skills.** `gui-dev` and `gui-component` were rewritten for the dashboard: the
  `gui` extra, `qtpy` and pyqtgraph, headless runs (`QT_QPA_PLATFORM=offscreen`,
  pytest-qt), the projects to try, and the rules Stage 2 sets for every panel: Qt only
  in `alis/dashboard/qt/`; every action an edit through the `History` (a drag in one
  `group`); `LabelNeeded` asks the user; every value through the blinding `Gate`;
  problems from `validate`; panels paused while the text does not read; the look of
  D43 and no design references (D44). `prepfit` is described as the old GUI.
- **Coverage** (the `test-coverage` skill, over the dashboard tests): 93% at first.
  Following the skill, the uncovered lines that matter were tested rather than the
  number chased: saving over an existing bundle keeps its runs (Q1.7); a ratio line
  goes when its denominator does (the cascade of `remove._ratios`, which the D I test
  did not reach because it deletes the ratio line itself); a mask written as integers
  keeps its style. Now 94% (3125 statements): `text` 97%, `model` 95%, `edit` 90%,
  `project` 92%, `validate` 94%, `blinding` 95%, `history` 97%, `remove` 97%, `modes`
  95%. Most of what is left is error branches for malformed input.
- **Documents.** The stage document records the status and what Stage 3 receives, and
  raises Q2.12 (two small faults in ALIS, not blocking). `CHANGELOG.md` (Unreleased:
  the dashboard package, the `gui` extra); `tests/README.md` (the dashboard tests);
  `dashboard_stage0.md`'s stage table (S16 and F12 have their logic in Stage 2; the
  user's preferences file is in Stage 3). No string shown to the user in
  `alis/dashboard/` names a design document (checked over every string constant that
  is not a docstring).
- **The batches:** `unit` 1418 passed, 0 failed (60 s); `fast` 110 passed, 0 failed
  (13 min 41 s).
- **ALIS outside `alis/dashboard/` is unchanged.**

**Stage 2 is complete.** Q2.12 is open for RJC.
