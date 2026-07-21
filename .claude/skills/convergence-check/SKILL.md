---
name: convergence-check
description: Re-run an ALIS fit from multiple randomised starting points and report whether all runs converge to the same solution, flagging parameter degeneracies.
---

Test the robustness of an ALIS fit by running it repeatedly with randomised initial parameter values and checking whether all runs reach the same chi-squared minimum.

## Steps

1. Identify the model file to test. If unspecified, ask which fit to check.

2. Parse the model file to identify all free parameters, their initial values, and their allowed bounds.

3. For each trial (default: 10 trials, configurable by the user):
   - Set a reproducible random seed (`numpy.random.seed(trial_index)`)
   - Randomly perturb each free parameter within its allowed range using a uniform distribution (default: ±50% of the initial value, clamped to the parameter bounds)
   - Write a temporary model file with the perturbed starting values
   - Run the fit and record: final reduced χ², best-fit parameter values, and minimiser convergence status

4. Compare results across all trials:
   - Are all final χ²/dof values within 1% of each other?
   - Do all trials return consistent best-fit parameter values (within 2σ of the median)?
   - Are any parameters systematically different between trials?

5. Report a summary table:

   | Trial | Seed | Final χ²/dof | Converged? |
   |-------|------|-------------|-----------|

   Then list any parameters that vary significantly between trials (potential degeneracies or multiple minima).

6. Give an overall verdict:
   - **Robust**: all trials agree — the fit is trustworthy
   - **Degenerate**: trials diverge — flag the specific parameters and suggest remedies (tighten bounds, fix degenerate parameters, add data constraints)
   - **Multiple minima**: trials converge to distinct χ² values — the model may be over-parameterised

## Notes

- Use `numpy.random.seed` for reproducibility; report the seed used for each trial so any trial can be reproduced.
- Perturb parameters by ±50% of their initial value by default, staying within the declared parameter bounds.
- Clean up temporary model files after the check.
- This skill directly supports the convergence-checking goals described in the ALIS v2 Context.
