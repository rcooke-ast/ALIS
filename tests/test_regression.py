"""
ALIS regression test suite (Stage 0.2).

The suite is data-driven: every test case is discovered by
``manifest.discover_cases()`` (Stage 0.1). Each fit case yields up
to two tests:

- ``test_minimisation`` (mode a): run ``run_alis <fit>.mod`` in a
  disposable copy of the example and compare the produced
  ``.mod.out`` (parameters, 1-sigma errors, chi-squared, DOF), every
  paired ``_fit.dat`` model column, and -- where a golden copy
  exists -- the covariance matrix. Marked with the case's runtime
  batch (fast / medium / slow).
- ``test_fixed_param`` (mode b, "option A"): run the case's
  ``.mod.out.reference`` itself with ``chisq miniter 0`` and
  ``chisq maxiter 0`` and compare the evaluated model against the
  same golden ``reference_fits/`` files, with a tighter chi-squared
  tolerance. Always marked fast (runs on every commit); blinded
  cases are excluded (their .mod.out values are hidden or offset).

The ``generate`` special case runs its data-generation model and
compares the created data file against its golden copy.

No test ever writes inside the repository: each run is staged in a
pytest tmp_path copy of the example directory (see alisrun.py).
"""

import shutil

import pytest

from alisrun import (
    make_fixedparam_mod,
    run_alis,
    stage_case,
)
from compare import (
    CHISQ_RTOL_FIXEDPARAM,
    CHISQ_RTOL_MINIMISATION,
    FITDAT_FIXEDPARAM_PEAKFRAC,
    FITDAT_RTOL_FIXEDPARAM,
    compare_covar,
    compare_fit_dat,
    compare_generated_data,
    compare_mod_out,
)
from manifest import discover_cases

_CASES = discover_cases()
_FIT_CASES = [c for c in _CASES if c.kind == "fit"]
_GEN_CASES = [c for c in _CASES if c.kind == "generate"]

# Cases excluded from the fixed-parameter gate (mode b) in addition
# to the blinded cases (details: Q0.12 / Q0.14 / Q0.15 in
# claude_prompts/refactor_code_stage0.md). Every excluded case still
# runs its minimisation test (mode a).
#
# Most Task-0.2 exclusions were resolved by RJC: the load-buffer
# cases had their vfwhm fixed (capitalised) and were regenerated
# (lls, metal_line_abs_thermal, spline/splineCont, splineabs x2,
# VMP_DLA/J0035m0918); the two echo round-trip crashes now ship a
# hand-fixed .mod.out.reference_adjusted re-input (lsf_hst,
# splineContAbs); metal_line_abs_linear's stale reference was fixed;
# and HS0105p1619 now runs. Those are back in the gate.
#
# The sharp-feature cases run in the gate under the Q0.15 model-
# column criterion (|new-ref|/max(reference_model) <
# FITDAT_FIXEDPARAM_PEAKFRAC). After RJC regenerated J0903 and
# Q1243 they reproduce the goldens exactly; splineContAbs is 1.6e-3
# of peak (kept). Only tophat (1.0e-2 of peak -- a print-truncated
# sharp edge) exceeds it, and RJC was content to drop tophat from
# the gate (Q0.15); its minimisation test (mode a) still runs.
#
# What remains excluded from mode (b):
# - "reference point": brokenpowerlaw's committed .mod.out.reference
#   does not evaluate to its own recorded chi-squared (Q0.12);
# - "sharp edge": tophat exceeds the peak-relative tolerance (Q0.15).
FIXEDPARAM_EXCLUDE = {
    "examples/brokenpowerlaw/model/fit_spectra":
        "reference point (chi2 374.98 vs recorded 338.15); Q0.12",
    "examples/tophat/model/fit_spectra":
        "sharp edge (1.0e-2 of peak > tol); Q0.15",
}

# Cases whose *minimisation* test (mode a) does not reproduce the
# committed reference on the current code: a fresh from-scratch refit
# of these three hard, many-parameter real-world fits lands at a
# slightly (HS0105p1619) or substantially (Q1243 newstart76, J0903)
# different point, and for J0903 at a *better* chi-squared (5057.6 vs
# the recorded 5112.8) -- i.e. the committed reference is not the true
# minimum. This is a reference-quality / minimiser-reproducibility
# issue, independent of the Q0.12-Q0.15 fixed-parameter work, and is
# raised as Q0.17 for RJC to decide (regenerate the references, relax
# the mode-(a) tolerances for real-world fits, or accept as known
# divergence). Skipped (not xfailed) because each takes minutes to
# run. Their fixed-parameter gate (mode b) still runs where enabled
# (Q1243 newstart76 and J0903 pass it; HS0105p1619 is covered too now
# the out-covar strip removes its timeout).
MINIMISATION_KNOWN_DIVERGENCE = {
    "context/fitting_examples/DH/HS0105p1619/model/HS0105p1619":
        "refit diverges: errors ~25%, covar 0.28%, model <=0.2%; "
        "Q0.17",
    "context/fitting_examples/DH/Q1243p307/model/"
    "Q1243p307_converge_newstart76":
        "refit diverges at saturated H I cores; Q0.17",
    "context/fitting_examples/VMP_DLA/J0903p2628/model/J0903p2628":
        "refit finds better chi2 (5057.6 vs 5112.8); Q0.17",
}


def _params(cases, batch=None):
    """
    Wrap manifest cases as pytest parameters with batch markers.

    Parameters
    ----------
    cases : list[RegressionCase]
        The cases to parametrise over.
    batch : str | None
        Marker to apply to every case (e.g. "fast" for the
        fixed-parameter gate); None uses each case's own batch.

    Returns
    -------
    list
        pytest.param objects with ids and markers.

    Generated by RJC and Claude.
    """
    params = []
    for case in cases:
        mark = getattr(pytest.mark, batch or case.batch)
        params.append(pytest.param(case, id=case.name, marks=mark))
    return params


def _assert_clean(failures, case, note):
    """
    Fail the test with every collected comparison mismatch.

    Parameters
    ----------
    failures : list[str]
        Messages collected from the compare functions.
    case : RegressionCase
        The case under test (for the failure header).
    note : str
        Which mode was being tested.

    Generated by RJC and Claude.
    """
    if failures:
        body = "\n  - ".join(failures)
        pytest.fail(
            f"{case.name} [{note}]: {len(failures)} comparison "
            f"failure(s):\n  - {body}",
            pytrace=False,
        )


@pytest.mark.parametrize(
    "case",
    _params(
        [
            c
            for c in _FIT_CASES
            if c.name not in MINIMISATION_KNOWN_DIVERGENCE
        ]
    ),
)
def test_minimisation(case, tmp_path):
    """
    Mode (a): full fit, compared against all golden references.

    Parameters
    ----------
    case : RegressionCase
        The manifest entry to run.
    tmp_path : Path
        Pytest per-test temporary directory.

    Generated by RJC and Claude.
    """
    staged = stage_case(case, tmp_path)
    mod = staged.staged(case.mod_file)
    run_alis(mod, runtime_hrs=case.runtime_hrs)
    failures = []
    out_mod_out = mod.parent / (mod.name + ".out")
    failures += compare_mod_out(
        out_mod_out, case.mod_out_reference, CHISQ_RTOL_MINIMISATION
    )
    for pair in case.data_pairs:
        for ref in pair.reference_fits:
            failures += compare_fit_dat(
                staged.fit_output_for(ref), ref
            )
    if case.covar_reference is not None:
        failures += compare_covar(
            staged.staged(case.covar_output), case.covar_reference
        )
    _assert_clean(failures, case, "minimisation")
    # Keep the staged copy only when the test fails (for debugging).
    shutil.rmtree(staged.root, ignore_errors=True)


@pytest.mark.parametrize(
    "case",
    _params(
        [
            c
            for c in _FIT_CASES
            if not c.is_blind and c.name not in FIXEDPARAM_EXCLUDE
        ],
        batch="fast",
    ),
)
def test_fixed_param(case, tmp_path):
    """
    Mode (b): fixed-parameter evaluation at the best-fit values.

    Runs the .mod.out.reference with miniter = maxiter = 0 and
    checks that the evaluated model reproduces the existing golden
    reference_fits/ files and the reference chi-squared (0.1%).

    Parameters
    ----------
    case : RegressionCase
        The manifest entry to run (blind cases are excluded).
    tmp_path : Path
        Pytest per-test temporary directory.

    Generated by RJC and Claude.
    """
    staged = stage_case(case, tmp_path)
    mod = make_fixedparam_mod(staged)
    # The evaluation itself is fast, but data loading of the large
    # cases is not free -- reuse the reference runtime for the
    # timeout headroom rather than the base timeout alone.
    run_alis(mod, runtime_hrs=case.runtime_hrs)
    failures = []
    out_mod_out = mod.parent / (mod.name + ".out")
    # Per Stage 0.2b the gate is chi-squared + DOF + model column;
    # a zero-iteration run does not reproduce the 1-sigma errors.
    failures += compare_mod_out(
        out_mod_out,
        case.mod_out_reference,
        CHISQ_RTOL_FIXEDPARAM,
        compare_params=False,
    )
    for pair in case.data_pairs:
        for ref in pair.reference_fits:
            failures += compare_fit_dat(
                staged.fit_output_for(ref),
                ref,
                rtol=FITDAT_RTOL_FIXEDPARAM,
                peakfrac=FITDAT_FIXEDPARAM_PEAKFRAC,
            )
    _assert_clean(failures, case, "fixed-param")
    # Keep the staged copy only when the test fails (for debugging).
    shutil.rmtree(staged.root, ignore_errors=True)


@pytest.mark.parametrize("case", _params(_GEN_CASES, batch="fast"))
def test_generate(case, tmp_path):
    """
    The "generate" special case: data generation, no fit.

    Runs generate_spectra.mod and compares the created data file
    against its golden copy in reference_fits/.

    Parameters
    ----------
    case : RegressionCase
        The generate manifest entry.
    tmp_path : Path
        Pytest per-test temporary directory.

    Generated by RJC and Claude.
    """
    staged = stage_case(case, tmp_path)
    mod = staged.staged(case.mod_file)
    run_alis(mod, runtime_hrs=case.runtime_hrs)
    failures = []
    for pair in case.data_pairs:
        for ref in pair.reference_fits:
            failures += compare_generated_data(
                staged.staged(pair.input_file), ref
            )
    _assert_clean(failures, case, "generate")
    # Keep the staged copy only when the test fails (for debugging).
    shutil.rmtree(staged.root, ignore_errors=True)
