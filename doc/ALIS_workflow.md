# ALIS Workflow Guide

**Version:** 0.3  
**Date:** 2026-07-12  
**Authors:** RJC and Claude

---

## Overview

ALIS (Absorption LIne Software) fits models to spectroscopic data using chi-squared
minimization. The core workflow is:

1. Prepare one or more data files (ASCII or FITS)
2. Write a model file (`.mod`) that describes the settings, data, and model
3. Run ALIS: `run_alis myfit.mod`
4. Inspect the output: a best-fit model file (`.mod.out`), optional fit data files,
   and an optional PDF plot

The sections below describe each step in detail. The primary reference for ALIS syntax
is the source code (in `alis/`) and the fitting examples in `context/fitting_examples/`.
The LaTeX documentation in `doc/tex_files/` provides useful background but is
incomplete and may be out of date in places.

---

## 1. Data Preparation

ALIS reads ASCII (column-delimited) or FITS data files. The minimum required columns are:

| Column | Description                                           |
|--------|-------------------------------------------------------|
| `wave` | Wavelength array (must have units of Å)               |
| `flux` | Normalised or unnormalised flux                       |
| `error` | 1σ flux uncertainty associated with the `flux` column |

Optional columns (read using the `columns=` keyword in the data block):

| Column | Description |
|--------|-------------|
| `fitrange` | Integer mask: 1 = include in fit, 0 = exclude |
| `continuum` | Pre-computed continuum normalisation (multiplied into model) |
| `zerolevel` | Pre-computed zero-level offset (added to model) |

Note that if the input data files do not contain information in one of the optional columns, the corresponding model block can still include a `zerolevel` or `continuum` model. ALIS will then generate these models as part of the fitting procedure, and store them in the corresponding columns of the output data files.

Wavelength units default to Å (`bintype=km/s` means *constant-velocity* pixels, not
a velocity unit — this is the default for echelle data; `bintype=A` for constant-Å pixels).

---

## 2. The Model File

The model file (`.mod`) is a plain-text file with four sections read in order:

```
# Comments begin with '#'. Block comments use '#--> ... <--#'

<three-argument global settings>

data read
  <data file specifications>
data end

model read
  <global model limits>
  emission
    <emission models>
  absorption
    <absorption models>
  zerolevel
    <zero-level models>
model end

link read
  <parameter link expressions>
link end
```

### 2.1 Global Settings

Three-argument commands of the form `<category> <keyword> <value>` that override
the built-in defaults. Run **`run_alis --list-settings`** to see every setting
with its default; the defaults live in `alis/config.py` (`ArgFlag`) and nowhere
else, so that listing cannot go out of date. (There used to be a shipped
`alis/data/settings.alis`; it was a second copy of the same defaults, the two
had drifted apart on four settings, and it was removed in v2.) Any setting can
also be given on the command line with `--set`, e.g. `--set 'run backend cpu'`,
which beats the model file. Examples:

```
run  ncpus        4          # CPUs to use (-1 = all bar one, -2 = all bar two)
run  backend      auto       # Jacobian backend: cpu, gpu, or auto (time both)
run  ngpus        0          # GPUs to use (0 = CPU backend; see below)
run  gputhresh    10000      # Smallest model group worth sending to a GPU
run  nsubpix      5          # Sub-pixels per 1σ for profile integration
run  blind        False      # Global blind analysis flag
run  convergence  False      # Run convergence checker
run  convcriteria 0.2        # Convergence criterion (in units of param 1σ)
run  atomic       atomic_rjc.xml  # Atomic data file
chisq atol        0.01       # Stop when χ² fractional change < atol
chisq fstep       1.3        # Levenberg-Marquardt step size factor
chisq miniter     10         # Minimum iterations before convergence test
chisq maxiter     3000       # Maximum iterations
out  covar        fit.covar  # Write covariance matrix to this file
out  fits         True       # Write best-fit data columns to file
out  model        True       # Write best-fit model parameters to .out
out  plots        fit.pdf    # Save plot to PDF (empty string = no save)
plot dims         3x1        # Plot grid: ROWSxCOLUMNS
plot fits         True       # Overplot best-fit model on data
plot fitregions   True       # Shade the fitted wavelength regions
plot ticklabels   True       # Show component ID labels above profiles
```

Key `run` sub-settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `ncpus` | −1 | CPU count; −1 = all bar one, −2 = all bar two |
| `backend` | auto | Which backend computes the Jacobian: `cpu`, `gpu` or `auto` |
| `ngpus` | 0 | GPU count; 0 keeps the CPU backend (see *Fitting on GPUs*) |
| `gputhresh` | 10000 | Smallest model group (sub-pixels × profiles) sent to a GPU |
| `shmem` | True | Share the workers' read-only arrays via `/dev/shm` (see *Memory during a fit*) |
| `nsubpix` | 5 | Sub-pixel oversampling (fixed count) |
| `nsubmin`/`nsubmax` | 5/21 | Min/max sub-pixel oversampling (adaptive) |
| `blind` | True | Global blind analysis |
| `convergence` | False | Enable convergence testing |
| `datatype` | default | Data format: `default`, `HIRESredux`, `UVESpopler` |

#### Fitting on GPUs

The parallel backend is an either-or choice made once per fit: the Jacobian's
derivative columns are computed **either** by a pool of `ncpus` CPU workers **or**
by a pool of `ngpus` GPU workers (one CUDA device each), never both. The CPU is
the default; `run ngpus N` (N > 0) opts in, and ALIS prints a notice at start-up
when it finds GPUs you are not using.

`run backend` decides how the choice is made:

| Value | Behaviour |
|---|---|
| `auto` (default) | CPU, unless `run ngpus N` (N > 0) is set — then ALIS warms both backends, times **one Jacobian on each** at the starting parameters, and commits the whole fit to the faster one. |
| `cpu` | The CPU pool, `ngpus` ignored. Nothing is probed, so this is also the cheapest start-up and the most reproducible setting. |
| `gpu` | The GPU pool: `ngpus` devices, or every device present if `ngpus` is unset. |

**Do not assume the GPU is faster.** Whether it wins depends on your model and
on how many devices you have against how many cores. On a 12-core, 4-GPU machine
one Jacobian of the 351-spectrum `DH_orders` model takes 110 s on the CPU and
177 s on the GPUs — so `auto` correctly keeps that fit on the CPU. Per *worker*
the GPU is about 1.9× faster; there are simply three times fewer of them. The
probe costs one discarded Jacobian on the losing backend, which is ~1% of a
75-iteration fit.

Use `backend cpu` or `backend gpu` when you need reproducibility: `auto` can
pick differently between runs when the two are close, and the CPU and GPU paths
agree only to ~1e-12, not bit for bit.

GPU support needs the optional extra and a CUDA toolkit matching your driver
(the toolkit is *not* installed by pip):

```
pip install "alis[gpu]"
```

If no usable GPU is found, `run ngpus N` warns and falls back to the CPU, so a
model file written on a GPU machine still runs everywhere.

Only model functions with a GPU implementation are dispatched to the device
(currently `voigt`); everything else is computed on the CPU as usual, within the
same fit. A group of profiles is sent to the device only if it is large enough
to be worth a kernel launch — `sub-pixels × profiles ≥ gputhresh` — because
below roughly 10⁴ pixel-components the launch and transfer cost more than the
kernel saves. Lower `gputhresh` to push more work onto the device (`0` forces
every supported group there); raise it to keep small snips on the CPU.

The GPU and CPU paths agree to better than 10⁻¹² in the profile, but they are
not bit-for-bit identical, so a fit repeated on a different backend will differ
in the last digits. Pick one backend (`run backend cpu` or `gpu`) for a piece of
work if you need reproducible output.

#### Memory during a fit

Each derivative worker needs the same read-only arrays — the data, the sub-pixel
grids and the cached model components — and used to be handed its own copy of
them. On a large fit that is the single biggest thing ALIS holds in memory:
DH_orders shipped 6.4 GB per Jacobian and kept about 0.5 GB resident in every
one of its 12 workers.

With `run shmem True` (the default) those arrays are placed in one shared
segment instead, and the workers read them in place. Nothing changes in the
result — the fit is bit-for-bit the same — but on DH_orders the total footprint
drops from 13.1 GB to 7.7 GB and the segment is released when the fit ends.

Set `run shmem False` if `/dev/shm` is small; a container's default is 64 MB,
which is not enough for a large fit. ALIS also falls back on its own, with a
warning, if a segment cannot be created — the fit then behaves as it did before,
carrying the arrays with each chunk of work.

### 2.2 Data Block

Each line between `data read` and `data end` specifies one data file and its
properties. All properties must be specified on a single line (no line continuation):

```
../data/myspectrum.dat
  specid=HI_Lya
  fitrange=columns
  loadrange=[3500.0,3700.0]
  resolution=vfwhm(6.280vh)
  shift=vshift(0.0SHP)
  columns=[wave:0,flux:1,error:2,fitrange:3,continuum:4,zerolevel:5]
  loadrange=all
  label=HIRES
  plotone=False
```

**Key data block keywords:**

| Keyword | Description |
|---------|-------------|
| `specid=<str>` | Links data to model components with the same `specid` |
| `fitrange=[lo,hi]` | Wavelength range used in χ² (or `all`, or `columns`) |
| `loadrange=[lo,hi]` | Wavelength range to load and generate model over (or `all`) |
| `resolution=vfwhm(x)` | Instrumental resolution: FWHM in km/s |
| `resolution=Afwhm(x)` | Instrumental resolution: FWHM in Å |
| `resolution=vsigma(x)` | Instrumental resolution: sigma in km/s |
| `resolution=columns` | Read resolution (FWHM km/s) from a data column |
| `shift=vshift(x)` | Velocity shift in km/s (for heliocentric correction etc.) |
| `shift=Ashift(x)` | Wavelength shift in Å |
| `shift=vshiftscale(v,s)` | Simultaneous velocity shift and wavelength scale factor |
| `columns=[...]` | Map column names to zero-indexed column positions |
| `loadall=True` | Load all pixels (deprecated synonym for `loadrange=all`) |
| `loadrange=all` | Load all pixels in the file |
| `bintype=km/s` | Pixels have constant velocity width (default) |
| `bintype=A` | Pixels have constant wavelength width |
| `nsubpix=N` | Override global sub-pixel setting for this file |
| `plotone=True/False` | Plot in its own panel, not grouped with `plot dims` |
| `label=<str>` | Label shown on the plot panel |
| `yrange=[lo,hi]` | Override automatic y-axis range |

**The `specid` system:** Any model component with `specid=X` is applied only to the
data file(s) that also have `specid=X`. Multiple data files can share the same `specid`
(the model is applied to all of them). If no `specid` is given in the model, it applies
to all data.

**Fixing/tying the resolution and shift:** The parameter label appended to the value
controls whether it is free, tied, or fixed (see §2.3). For example,
`resolution=vfwhm(6.280vh)` makes the FWHM a *free parameter* labelled `vh`.
`resolution=vfwhm(6.280VH)` fixes it at 6.280 km/s. Two different data files sharing
`resolution=vfwhm(6.280vh)` will have their resolution *tied* to the same free parameter.

### 2.3 Parameter Specification: Fixing, Tying, and Limiting

ALIS identifies parameters by a label string that immediately follows the numeric value
(no spaces). The case of the label controls its behaviour:

| Label case | Behaviour |
|------------|-----------|
| **lowercase** (e.g., `0.5ra`) | Free parameter, labelled `ra`; all instances of `ra` share the same free parameter (tied) |
| **UPPERCASE** (e.g., `0.5RA`) | Fixed parameter, labelled `RA`; all instances of `RA` are fixed at the *first* value encountered |
| No label | Free parameter, independent of all others |

**Scientific notation must carry a sign on the exponent** — write `1.0E+04` or
`1.0e-3`, not `1e4`. Because the label follows the value with no separator,
`1e4` is indistinguishable from "the value 1.0, labelled `e4`", and that is how
ALIS would read it: two lines both written `1e4` would silently share one free
parameter set to 1.0 rather than both being 10000. ALIS therefore rejects a
label that is `e`/`E` followed only by digits, and tells you to add the sign or
rename the label. A label that merely begins that way (`E5t`, `e345j`) cannot
be a number and is accepted as an ordinary label. The same applies inside a
resolution or shift function — `resolution=vfwhm(1e4)` is rejected too.

**Global `fix` and `lim` commands** (inside the `model read` section, before `emission`):

```
fix voigt temperature True        # Fix temperature for all voigt lines
fix vfwhm value True              # Fix all resolution parameters
lim voigt bturb [0.01,None]       # b-parameter ≥ 0.01 km/s, no upper limit
lim voigt ColDens [8.0,22.0]      # log N between 8 and 22
lim constant value [None,None]    # No limits on zerolevel constants
lim param jc [20.0,21.0]          # Limit single parameter labelled 'jc'
fix param tval True               # Fix single parameter labelled 'tval'
```

These commands affect *all subsequent* model instances of that type. A second `fix`
or `lim` command for the same function/parameter overrides the previous one for all
*subsequent* lines (it does not retroactively change earlier lines).

### 2.4 Model Block — Emission

The `emission` keyword separates emission models. All lines after `emission` (until
`absorption`, `zerolevel`, or `model end`) are treated as additive emission.

**Built-in emission models:**

| Function | Description |
|----------|-------------|
| `constant` | Scalar constant: `constant 1.0CONST specid=0` |
| `legendre` | Legendre polynomial: `legendre p0 p1 p2 ... scale=[s0,s1,s2] specid=0 min=lo max=hi` |
| `chebyshev` | Chebyshev polynomial (up to order 9) |
| `polynomial` | Standard polynomial |
| `powerlaw` | Power-law continuum: `powerlaw amplitude index` |
| `brokenpowerlaw` | Broken power-law with 5 parameters |
| `gaussian` | Gaussian emission line: `gaussian amplitude redshift dispersion wave=H_I_1215` |
| `line_emission` | Emission line tied to atomic data: `line_emission ion=16O_III_5007 IntFlux redshift b` |
| `spline` | Spline function |
| `linear` | Linear function |

The `legendre` function accepts optional `min=` and `max=` keywords that specify the
wavelength bounds over which the polynomial is normalised. This is required when multiple
specids spanning different wavelength ranges share a single Legendre emission model (e.g.,
when all per-order specids for a given transition/dataset share one polynomial). Note that
in principle, any model function can be used as an emission model, but the above are the most common.

### 2.5 Model Block — Absorption

The `absorption` keyword begins the absorption sub-block. Absorption acts multiplicatively
on the preceding emission.

**The Voigt profile (`voigt`):**

```
voigt ion=1H_I  ColDens  redshift  bturb  temperature  [blind=False]  [specid=0]
```

Parameters (in order):
1. `ColDens` — log₁₀(N / cm⁻²)
2. `redshift` — z (absolute redshift of the absorber)
3. `bturb` — turbulent Doppler b-parameter (km/s)
4. `temperature` — kinetic temperature (K)

The `ion=` keyword must specify one of the ions in the `atomic.ecsv` file, using the format
`<MassNumber><ElementSymbol>_<IonStage>` (e.g., `1H_I`, `2H_I`, `28Si_II`). Special
pseudo-ions `1Ly_a` and `1H_IB` are available as fictitious absorbers for forest modelling.

**Multiple absorption components:** Each `voigt` line defines one absorption component.
Components at different redshifts or with different column densities are listed as separate
lines. Parameters are tied/fixed with the label convention (§2.3). For example, to tie the
redshift of two components `a` and `b`:

```
voigt ion=1H_I  19.5hia  2.5256ra  5.0da  1.0E4ta  specid=0
voigt ion=2H_I  14.9dha  2.5256ra  5.0da  9.0E3ta  specid=0
```

Here the redshift `ra`, b-parameter `da`, and kinetic temperature `ta` are shared (tied) between H I and D I.
**Important note:** Even though the kinetic temperature for 2H_I is set to a different numerical value than the kinetic temperature for the 1H_I line, the fact that they both have the same suffix `ta` means that they are tied together, and will both start with the value `1.0E4`, which is the first instance where `ta` is used. The second instance of `ta` will be ignored, and the kinetic temperature for both lines will be set to `1.0E4` at the start of the fit.

**The `ZEROT` label:** Setting temperature to `0.0ZEROT` fixes all lines with `ZEROT` at
T = 0 K, which means broadening is entirely turbulent. This is the conventional way to
mark components whose thermal broadening is negligible or unknown.

**Ion ratio syntax:** Instead of specifying a column density directly, a voigt component
can be defined as a logarithmic offset from another ion's column density:

```
voigt ion=28Si_II/16O_I  -1.0  specid=0
```

This sets log₁₀ N(Si II) = log₁₀ N(O I) − 1.0. The single parameter is the offset
(log abundance ratio). This is useful for enforcing physically-motivated abundance ratios
between species (e.g., tying a metal to hydrogen or to another metal). It also works
when generating synthetic data with `generate data True`.

**Linear column density mode:** For very high column densities or special cases, the
column density parameter can be expressed in linear (not logarithmic) units:

```
voigt ion=16O_I  1.0  0.0  5da  8.0E3TA  specid=0  logN=False  ColDensScale=1.0E14
```

Here `logN=False` means the parameter value is N / ColDensScale rather than log₁₀(N).
With `ColDensScale=1.0E14`, a parameter value of 1.0 corresponds to N = 10¹⁴ cm⁻².

**Lyman forest modelling:** `ion=1Ly_a` (or `ion=1H_IB`) is used for fictitious absorbers
representing blended Lyman-alpha forest lines. The profile shape is identical to a Voigt
profile, but the `ion=` pseudo-label signals that this is not a physical species from the
atomic database.

### 2.6 Model Block — Zerolevel

The `zerolevel` keyword begins the zero-level sub-block. Models here are *added* to the
emission×absorption product:

```
zerolevel
  constant 0.01  specid=Ly8h_FR0000,...
```

When fitting saturated absorption troughs where the true zero level is uncertain, or when
different detectors have a non-zero dark current contribution, a `constant` zerolevel model
allows ALIS to fit the offset. The `lim constant value [None,None]` command removes any
lower or upper bound on the zerolevel constant, allowing negative values.

### 2.7 Variable Models

```
variable -4.62dhrand  specid=...
```

The `variable` model holds a single free parameter that does not directly generate a
spectrum. It is used in combination with the `link` block to compute derived parameters
(e.g., the D/H ratio `dhrand`, which feeds into the D I column densities via links).

### 2.8 Link Block

The link block defines algebraic dependencies between model parameters:

```
link read
  dha(hia,dhrand) = hia + dhrand
  dhb(hib,dhrand) = hib + dhrand
link end
```

Syntax: `<linked_param>(<dependency1>,<dependency2>,...) = <expression>`

- The parameter on the left-hand side is *computed* from the right-hand side and cannot
  be a free parameter. ALIS will not report an error for the linked parameter.
- Expressions may use `+`, `-`, `*`, `/`, `**`, parentheses, numeric literals, and any
  parameter label defined in the model block.
- `numpy` functions are also available in expressions (e.g., `numpy.log10(...)`).

Links are useful for:
- Enforcing a constant D/H ratio across all components: `dha = hia + dhrand`
- Constraining flux ratios of emission lines
- Linking redshifts between different instruments with a velocity offset:
  `raa(ra,zoffs) = ra + zoffs`
- Calculating total (summed) column densities of nearby components. The summed column density is often more well-determined than the individual component column densities for absorption lines that are very close together.

---

## 3. Generating Synthetic Data

Before fitting, ALIS can generate synthetic (fake) spectroscopic data. This is useful
for testing that a model can recover known input parameters, or for planning observations.
Setting `generate data True` in the global settings causes ALIS to generate data instead
of fitting.

```
generate data      True
generate pixelsize 2.5     # Pixel size (km/s for bintype=km/s, Å for bintype=A)
generate peaksnr   20      # Signal-to-noise ratio at the model peak (0.0 = perfect data)
generate skyfrac   0.02    # Sky background fraction (relative to model peak)
generate overwrite True    # Overwrite existing output file
```

The generated spectrum has Gaussian noise with standard deviation 1/peaksnr at the
continuum level. ALIS writes the output to the file specified in the `data read` block
(the existing file is used as the wavelength grid if it already exists, or a new grid
is created). The output data file contains: wavelength, noisy flux, 1σ error, and the
noiseless model.

---

## 4. Running ALIS

```bash
run_alis myfit.mod
```

Internally, the call sequence is:

1. **`load.optarg`** — apply the parsed command line to the defaults
2. **`load.load_settings`** — the built-in defaults (`alis/config.py`)
3. **`load.load_input`** — parse the `.mod` file (settings, data, model, link sections); the command line is then re-applied over it, so an explicit flag wins
4. **`load.load_atomic`** — load `atomic.ecsv` atomic data
5. **`load.load_data`** — read all data files; apply `loadrange`, `bufferpix`, `fitrange`, column mapping
6. **`alload.load_model`** — parse emission, absorption, zerolevel, variable, and fix/lim commands
7. **`alload.load_links`** — parse link expressions
8. **`alload.load_subpixels`** — set up sub-pixel grids per spectrum
9. **`alcsmin.alfit`** — Levenberg-Marquardt chi-squared minimization (with multiprocessing)
10. **`alsave.save_model`** — write `<filename>.mod.out` with best-fit parameters and errors
11. **`alsave.save_modelfits`** — (if `out fits True`) write best-fit data columns to file
12. **`alplot`** — generate plot and save to PDF (if `out plots <file>`)

The minimizer (`alcsmin.alfit`, wrapping a customised version of `mpfit`) uses the
Levenberg-Marquardt algorithm. Each iteration, model profiles are computed for all spectra
in parallel using Python multiprocessing. The number of processes is set by `run ncpus`
— or, when `run ngpus` is greater than zero, by `run ngpus`, with one CUDA device bound
per worker (see *Fitting on GPUs* above).

### 4.1 Worked example: `examples/metal_line_abs/`

The `examples/metal_line_abs/` directory provides the simplest complete ALIS workflow.
The data (`OI_SiII.dat`) was generated by `generate_spectra.mod` using:
- A flat continuum (`constant 1.0`)
- O I 1302 Å: log N = 14.0, z = 0.0, b_turb = 5.0 km/s, T = 8000 K
- Si II 1304 Å: log N(Si II)/N(O I) = −1.0 (via the `ion=28Si_II/16O_I` ratio syntax)
- S/N ≈ 20 per pixel, spectral resolution 7.0 km/s FWHM

Running `run_alis fit_spectra.mod` from the `model/` subdirectory prints iteration progress
to screen, showing the chi-squared and current parameter values at each iteration:

```
ITERATION 1  CHI-SQUARED =  485.01  DOF = 360  (REDUCED = 1.347)
ITERATION 2  CHI-SQUARED =  416.08  DOF = 360  (REDUCED = 1.156)
...
ITERATION 5  CHI-SQUARED =  355.75  DOF = 360  (REDUCED = 0.988)
Reason for convergence: The relative reduction in the sum of squares is less than atol
```

**Fit statistics** (from `fit_spectra.mod.out` header):

| Statistic | Value |
|-----------|-------|
| Initial χ² | 485.0 |
| Best-fit χ² | 355.8 |
| DOF | 360 |
| Reduced χ² | 0.988 |
| Iterations | 6 |
| Run time | ~0.003 hours |

**Best-fit parameters vs true values:**

| Parameter | Starting | Best-fit | Error | True value |
|-----------|----------|---------|-------|------------|
| log N(O I) | 14.0 | 13.989 | 0.017 | 14.0 |
| log N(Si II) | 13.0 | 12.933 | 0.060 | 13.0 |
| b_turb (tied) | 1.0 km/s | 4.633 km/s | 0.355 | 5.0 km/s |
| Legendre p0 | 1.0 | 1.001 | 0.003 | 1.0 (flat) |

The fit recovers the true parameters well. The O I column density is within 0.7σ of
the true value, and the Si II column density within 1.1σ. The b-parameter (4.633 km/s)
is within 1σ of the true value (5.0 km/s). The reduced χ² = 0.988 ≈ 1 confirms the
fit is statistically consistent with the noise.

The residuals (normalised by the 1σ flux error) have mean = 0.000 and standard
deviation = 0.983 — confirming that the model describes the data to within the noise.

#### 4.1.1 The plot layout (`fit_spectra.pdf`)

The output PDF contains one panel (since there is only one data file). The layout has
two sub-panels stacked vertically:

- **Upper sub-panel** — the spectrum: observed flux (black), best-fit model (red),
  and continuum model (blue dashed line). Both absorption troughs are visible:
  O I 1302 Å (deep, reaching ~0.29 in normalised flux) and Si II 1304 Å (shallower,
  reaching ~0.81). The model traces both features cleanly.

- **Lower sub-panel** — residuals in two parts:
  - Top: ±1σ error spectrum shown as a shaded grey band, and the zerolevel line
    (green dashed).
  - Bottom (darker grey): The normalised residuals, (data − model) / σ, are shown as a
    blue histogram, where the scatter is consistent with white noise. The dark grey band
    indicates ±1σ.

The grey band in the residual panel shading covers the `fitrange`; the unshaded regions
at the edges of the loaded wavelength range are outside the `fitrange` and not
included in the chi-squared.

#### 4.1.2 Importance of starting parameter values

The first run of this example (before the bug fix in `fit_spectra.mod`) used
b_turb = 0.5 km/s as a starting value. That fit converged to a degenerate
local minimum: log N(O I) = 16.47 (true 14.0), b_turb = 0.500 km/s (unchanged
from the start), with reduced χ² = 1.07. The reduced χ² was consistent with a good
fit, yet the recovered column density was two orders of magnitude too high. This
illustrates a fundamental degeneracy in Voigt profile fitting: a very narrow, highly
saturated line can produce an indistinguishable profile from a broader, less saturated
line at a given spectral resolution. Correcting the starting b-value to 1.0 km/s was
sufficient to escape the local minimum and find the correct global solution.

---

## 5. Output Files

| File | Condition | Description |
|------|-----------|-------------|
| `<model>.mod.out` | `out model True` (default) | Best-fit parameters and 1σ errors, plus χ², DOF, reduced χ², runtime, convergence reason |
| `<data>_fit.dat` | `out fits True` | Best-fit model columns written to file for each input data file |
| `<model>.pdf` | `out plots <filename>` | PDF of data + best-fit model |
| `<model>.covar` | `out covar <filename>` | Parameter covariance matrix |
| `<model>.conv` | `out convtest <filename>` | Convergence test results |

The `.mod.out` file has the same format as the `.mod` file (with best-fit values substituted
in), prefixed by a block of comment lines (`#`) recording the fit statistics. This means
`.mod.out` can be used directly as a new `.mod` starting point with optimal parameters.

### 5.1 Fit data file format (`_fit.dat`)

When `out fits True` is set, ALIS writes one output data file per input spectrum. The
file name is derived from the input data file name by appending `_fit` before the
extension (e.g. `OI_SiII.dat` → `OI_SiII_fit.dat`). The file has four columns:

| Column | Content |
|--------|---------|
| 1 | Wavelength (Å) |
| 2 | Observed flux (same as input) |
| 3 | 1σ flux error (same as input) |
| 4 | Best-fit model flux |

The `loadrange` setting controls how many pixels are written. Pixels within the
`loadrange` but **outside** the `fitrange` are included in the file, but the
best-fit model column (column 4) is set to the sentinel value −9.999999999×10⁹.
This distinguishes pixels that were included in the χ² minimization (column 4
contains a real model value) from those that were loaded for context only.

Example from `OI_SiII_fit.dat`:
- 392 rows loaded (wider than the `fitrange=[1301.0,1305.0]` to include convolution buffer)
- 368 rows with real model values (~1.0 in continuum, ~0.3 at O I trough, ~0.8 at Si II trough)
- 24 rows at the edges with sentinel value −9.999999999×10⁹ (outside fitrange)

---

## 6. Monte Carlo Convergence Testing

ALIS provides two mechanisms to validate that the fit has found the global minimum, rather
than a local one.

### 5.1 Internal convergence check (`run convergence True`)

When `run convergence True` is set, after the main fit converges, ALIS re-runs the fit
from perturbed starting parameters (perturbed by a fraction of the fit error). If the
new fit converges to the same parameter values (within `run convcriteria` × 1σ), the
solution is considered converged. The number of re-runs is controlled by
`run convcriteria`. Results are written to the file specified by `out convtest`.

### 5.2 External convergence testing (newstart)

For complex models with many parameters (e.g., the D/H DLA fits), the internal check
may be insufficient. A common approach is to run ALIS many times with different random
starting parameters (drawn from a multivariate normal centred on the best-fit values,
using the covariance matrix written by `out covar`). The `sim newstart True` setting
enables this; it draws new starting parameters from the covariance matrix of a previous
fit. The Monte Carlo runs are stored in a `sims/` subdirectory. The file naming convention
in the fitting examples (e.g., `Q1243p307_converge_newstart76.mod`) records the run index.

A Python script external to ALIS is typically used to launch many such runs in parallel
and collect the results to build a distribution of best-fit D/H values.

---

## 7. Current ALIS Code Structure

The source files in `alis/` perform the following roles:

| File | Role |
|------|------|
| `alis.py` | Entry point: `ClassMain`, `myfunct_wrap`, `alis()` function. Contains the model evaluation function `myfunct` called at each chi-squared iteration. |
| `alload.py` | All data/model/settings loading: parses `.mod` file, reads data files, sets up parameter arrays, constructs `modpass`/`parin`/`fdict` dictionaries. |
| `alcsmin.py` | Chi-squared minimization: wraps `mpfit` (Levenberg-Marquardt), handles multiprocessing for parallel model evaluation. |
| `alconv.py` | Convolution of model profiles with instrumental broadening functions. |
| `alplot.py` | Generates matplotlib plots of data and best-fit models. |
| `alsave.py` | Writes output files: `.mod.out`, fit data files, covariance matrix. |
| `alsims.py` | Monte Carlo simulation management. |
| `alshift.py` | Velocity/wavelength shift application to spectra. |
| `alutils.py` | Utility functions (wavelength conversion, error handling, etc.). |
| `almsgs.py` | Custom message/logging system. |
| `alfunc_base.py` | Base class for all model functions; defines the interface. |
| `alfunc_voigt.py` | Voigt profile model (absorption). |
| `alfunc_legendre.py` | Legendre polynomial (emission/continuum). |
| `alfunc_gaussian.py` | Gaussian (emission). |
| `alfunc_constant.py` | Constant model (emission/zerolevel). |
| `alfunc_vfwhm.py` | Gaussian instrumental broadening (velocity FWHM). |
| `alfunc_variable.py` | Variable parameter (for use with links). |
| … | Additional function files with `alfunc_` prefix |
| `data/atomic.ecsv` | Atomic data (transitions, wavelengths, f-values, Gamma). `data/atomic.xml` is the superseded VOTable form, still readable |

### Key data structures (current v1)

ALIS v1 passes state primarily via the `ClassMain` instance (`self` / `slf`), which
carries a large set of nested dictionaries:

| Dictionary | Contents |
|------------|----------|
| `_argflag` | All run/chisq/plot/out settings |
| `_datlines`, `_modlines`, `_lnklines` | Raw lines from the `.mod` file |
| `_specdata` | Loaded spectral data arrays per specid |
| `_parin` | Parameter array (initial values, fix flags, limits) |
| `_modpass` | Per-iteration model evaluation state |
| `_fdict` | Serialised `ClassMain` state for multiprocessing |

A known issue is that `myfunct_wrap` (the function called at every chi-squared iteration)
instantiates a new `ClassMain` object and copies the entire state into it via
`instance.__dict__.update(fdict)`. This is a workaround for Python multiprocessing's
requirement that the function be picklable, and is the primary circular import issue
targeted in ALIS v2.

---

## 8. Known Issues Targeted in ALIS v2

The following are the key technical issues identified for the ALIS v2 rewrite:

1. **Circular import**: `myfunct_wrap` creates a new `ClassMain` instance at every
   chi-squared iteration, importing and instantiating the entire class. This is slow
   and architecturally fragile.

2. **ClassMain / `self` passing**: The entire program state is carried in a single
   class instance. This makes unit testing difficult and tightly couples all modules.

3. **Nested dictionaries**: `_argflag`, `_modpass`, `_fdict` etc. are opaque nested
   dicts. Python `dataclasses` (or `attrs`/`pydantic`) would make these inspectable
   and type-checkable.

4. **Custom messaging**: `almsgs.msgs()` should be replaced with Python's standard
   `logging` module.

5. **No type annotations**: Adding type hints throughout would enable `mypy` and
   improve IDE autocompletion.

6. **Python 2 compatibility stubs**: `from __future__ import ...` and `raw_input`
   stubs should be removed.

7. **GPU support**: Currently CPU-only. `run ngpus` is a placeholder; Voigt profile
   computation on GPU is a high-priority v2 feature for large per-order models.

8. **Model function instantiation**: `alfunc_*.py` classes are instantiated at every
   model evaluation; they should be instantiated once and reused.

9. **No unit tests**: All current testing is end-to-end via example model runs.

10. **CLI**: The entry point uses a custom argument parser; `argparse` / `typer` would
    give self-documenting `--help` output.

---

## 9. Appendix: ALIS Voigt Profile Parameters

The `voigt` function takes four positional parameters and several keywords:

```
voigt ion=<ion>  <ColDens><label>  <redshift><label>  <bturb><label>  <temperature><label>
      [blind=False]  [specid=<id>]  [damping=<value><label>]
```

| Parameter | Name | Units | Default |
|-----------|------|-------|---------|
| p0 | `ColDens` | log₁₀(cm⁻²) | 0.0 |
| p1 | `redshift` | dimensionless | 0.0 |
| p2 | `bturb` | km/s | 0.0 |
| p3 | `temperature` | K | 0.0 |

The total line-of-sight velocity dispersion is:
```
b_total = sqrt(b_turb² + 2 k T / m)
```
where m is the mass of the ion. At T = 0 K (ZEROT), the broadening is purely turbulent.

The oscillator strength, wavelength, and damping constant are read from `atomic.ecsv` for
the specified `ion`. The `damping=` keyword allows an override of the natural damping
constant (used for e.g. custom telluric line shapes in the He I* models).

---

## 10. Appendix: Atomic Data File (`atomic.ecsv`)

The file `alis/data/atomic.ecsv` (or a custom variant, including a legacy `.xml`) is a
VOTable-format XML file with the following columns:

| Column | Description |
|--------|-------------|
| 1 | Mass number |
| 2 | Atomic mass (amu) |
| 3 | Solar isotopic abundance |
| 4 | Element name |
| 5 | Ionisation stage |
| 6 | Vacuum wavelength (Å) |
| 7 | Oscillator strength (f-value) |
| 8 | Transition probability (s⁻¹) |
| 9 | q-value (fine-structure constant variation) |
| 10 | K-value (proton-to-electron mass ratio variation) |

The `nrows` attribute in the file header must be updated manually whenever new rows are
added. This is a known maintenance issue targeted in v2.