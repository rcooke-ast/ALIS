---
name: run-example
description: Run a named ALIS example end-to-end (generate synthetic data, then fit it) and verify the fit converges correctly.
---

Run one of the ALIS example fits from the `examples/` directory. Each example has two stages: generating synthetic data and then fitting it.

## Available examples

Located in `examples/`:
- `lls` — Lyman-limit system (HI absorption + photo-ionisation cross-section)
- `metal_line_abs` — metal absorption lines (with variants: linear continuum, thermal broadening)
- `emission_line_ratio` — emission line ratio fitting
- `CNabs` — CN absorption
- `lsf_multigauss` — multi-Gaussian LSF
- `lsf_spline` — spline LSF
- `lsf_file` — LSF loaded from a file
- `lsf_hst` — HST line-spread function
- `spline` — spline continuum (with absorption variant)
- `splineabs` — spline absorption
- `summed_coldens` — summed column densities

Each example directory contains `model/generate_spectra.mod` (to produce synthetic data) and one or more `model/fit_spectra*.mod` files (to fit).

## Steps

1. Identify the example name from the user's request. If unspecified, ask which example to run.

2. Check that the directory `examples/<name>/` exists and contains the expected `.mod` files.

3. Run from the example directory so that relative data paths inside the `.mod` file resolve correctly:
   ```
   cd /Users/rcooke/Software/ALIS/examples/<name>
   ```

4. Generate synthetic data (skip if data files already exist):
   ```python
   python -c "
   from alis import alis
   with open('model/generate_spectra.mod') as f:
       alis.main(parlines=f.readlines())
   "
   ```

5. Run the fit:
   ```python
   python -c "
   from alis import alis
   with open('model/fit_spectra.mod') as f:
       alis.main(parlines=f.readlines())
   "
   ```

6. Verify convergence: check that the fit printed a final reduced chi-squared near 1.0 and no error messages were raised.

7. Report: example name, final reduced chi-squared, number of free parameters, and whether the fit converged.

## Notes

- Some examples have multiple `.mod` variants (e.g. `fit_spectra_linear.mod`); run the one specified by the user, defaulting to `fit_spectra.mod`.
- Do not modify any `.mod` files or output data files without explicit instruction.
