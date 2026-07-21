---
name: check-fit
description: Parse ALIS output files (.mod.out) and summarise fit quality — per-region reduced chi-squared, free-parameter count, and convergence warnings.
---

Read and summarise the results of an ALIS fit from its output files.

## ALIS output conventions

After a fit, ALIS writes results to a file alongside the model file (typically `<modelfile>.mod.out`). The file contains best-fit parameter values, 1-sigma uncertainties, and fit statistics including the total chi-squared and degrees of freedom.

## Steps

1. Identify the output file. If the user does not specify one, search for `*.mod.out` files in the current directory and under `examples/`.

2. Read the output file and extract:
   - Final reduced chi-squared (χ²/dof) overall and per spectral region / `specid`
   - Number of free parameters and number of data points (degrees of freedom)
   - Best-fit parameter values and their 1-sigma uncertainties
   - Any convergence warnings (parameters at their limits, non-positive-definite covariance, fit did not converge)

3. Summarise in a compact table:

   | Region (specid) | Data points | Free params | χ²/dof | Status |
   |-----------------|------------|------------|--------|--------|

4. Flag potential issues:
   - Parameters at their lower or upper bound → possible convergence issue
   - Parameters with uncertainty larger than their value → poorly constrained
   - χ²/dof >> 1 → unmodelled features, underestimated errors, or wrong model
   - χ²/dof << 1 → overestimated errors or too many free parameters

5. Give an overall fit quality assessment and, where χ²/dof deviates significantly from 1, suggest the most likely causes.

## Notes

- If the output file format is unfamiliar or has changed, read it in full before summarising.
- Do not modify any output or model files.
- The reduced chi-squared target is approximately 1.0; values between 0.9 and 1.2 are generally acceptable depending on the noise model.
