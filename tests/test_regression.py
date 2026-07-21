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
    FITDAT_ERRFRAC_FIXEDPARAM,
    FITDAT_ERRFRAC_MINIMISATION,
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
# The sharp-feature cases run in the gate under the Q0.22 error-based
# model-column check (|new-ref| < FITDAT_ERRFRAC_FIXEDPARAM * error).
#
# What remains excluded from mode (b):
# - "reference point": brokenpowerlaw's committed .mod.out.reference
#   does not evaluate to its own recorded chi-squared (Q0.12);
# - "sharp edge": tophat's print-truncated edge lands between pixels,
#   giving a ~14x-the-error model jump at the edge pixel, far beyond
#   any sane error fraction (RJC was content to drop it; Q0.15). Its
#   minimisation test (mode a) still runs.
FIXEDPARAM_EXCLUDE = {
    "examples/brokenpowerlaw/model/fit_spectra":
        "reference point (chi2 374.98 vs recorded 338.15); Q0.12",
    "examples/tophat/model/fit_spectra":
        "sharp edge (~14x error at edge pixel); Q0.15",
}

# Fixed-parameter evals normally all run on every commit (batch
# "fast"). The DH_orders eval is the exception: a single evaluation of
# the 351-spectra model takes ~4.3 min, so RJC placed it in the
# nightly "medium" batch (Q0.19) to keep the every-commit batch fast.
FIXEDPARAM_BATCH_OVERRIDE = {
    "context/fitting_examples/DH_orders/Q1243p307/model_orders/"
    "Q1243p307_orders": "medium",
}

# Minimisation (mode a) tests skipped because a fresh refit does not
# reproduce the committed reference. Earlier rounds were resolved by
# RJC (HS0105p1619/J0903p2628/Q1243_newstart76 in Q0.17; five
# DH/helium34 fits in Q0.21 by regeneration; J1358p6522_original in
# Q0.22 by adopting the error-based model-column check).
#
# DH/J0814p5029 (Q0.24) is different: it is a blind + *random* D/H fit
# of a complex H I Lyman forest that is degenerate/multi-modal. A
# fresh refit reaches the *same chi-squared* (0 chi-squared failures)
# but at meaningfully different parameters and model -- the H I cores
# differ by up to 0.74 of the pixel error, and 161 one-sigma errors
# differ > 10%. The random start lands at a different-but-equivalent
# solution each run, so regeneration will not fix it (unlike the
# deterministic cases). It only ran here because --run-slow first
# exercised the slow batch. Skipped pending Q0.24 (fix the random D/H
# start to the reference value for a deterministic test, or compare
# chi-squared/DOF only for this case).
MINIMISATION_KNOWN_DIVERGENCE = {
    "context/fitting_examples/DH/J0814p5029/model/J0814p5029":
        "degenerate blind+random D/H fit: same chi-squared but "
        "different params/model (H I cores to 0.74*error); Q0.24",
}


def _params(cases, batch=None, overrides=None):
    """
    Wrap manifest cases as pytest parameters with batch markers.

    Parameters
    ----------
    cases : list[RegressionCase]
        The cases to parametrise over.
    batch : str | None
        Marker to apply to every case (e.g. "fast" for the
        fixed-parameter gate); None uses each case's own batch.
    overrides : dict[str, str] | None
        Per-case marker overrides keyed by case name; takes
        precedence over ``batch`` (e.g. keep the one very slow
        fixed-parameter eval out of the every-commit batch).

    Returns
    -------
    list
        pytest.param objects with ids and markers.

    Generated by RJC and Claude.
    """
    overrides = overrides or {}
    params = []
    for case in cases:
        chosen = overrides.get(case.name, batch or case.batch)
        mark = getattr(pytest.mark, chosen)
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
                staged.fit_output_for(ref), ref,
                errfrac=FITDAT_ERRFRAC_MINIMISATION,
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
        overrides=FIXEDPARAM_BATCH_OVERRIDE,
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
                errfrac=FITDAT_ERRFRAC_FIXEDPARAM,
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
