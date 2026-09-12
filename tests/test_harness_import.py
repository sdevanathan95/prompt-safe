"""Import-only sanity check: eval.harness must be importable (and its
CHEAP_MODEL_BY_PROVIDER/build_pipeline referenceable) without an API key
set or any network call. Actual runs are explicit and out of scope for
tests/ — see eval/harness.py's module docstring.
"""

from eval.harness import (
    CHEAP_MODEL_BY_PROVIDER,
    CaseResult,
    build_pipeline,
    run_suite_subset,
)


def test_cheap_model_defaults_present_for_both_providers():
    assert "openai" in CHEAP_MODEL_BY_PROVIDER
    assert "anthropic" in CHEAP_MODEL_BY_PROVIDER


def test_case_result_and_run_suite_subset_are_importable():
    assert CaseResult is not None
    assert callable(build_pipeline)
    assert callable(run_suite_subset)


def test_the_run_options_are_real_parameters_not_module_constants():
    """Each option a full run depends on must be settable per run."""
    import inspect

    params = inspect.signature(run_suite_subset).parameters
    for name in ("response_channel", "alignment_model", "lazy_masked_run", "results_path"):
        assert name in params, name
    assert params["response_channel"].default is False

