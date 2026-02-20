"""
Binding-surface compatibility tests for local rlm_core wrapper exports.
"""

import rlm_core


def test_core_compatibility_helpers_exported():
    assert hasattr(rlm_core, "version")
    assert hasattr(rlm_core, "version_tuple")
    assert hasattr(rlm_core, "has_feature")
    assert hasattr(rlm_core, "available_features")
    assert hasattr(rlm_core, "quick_hallucination_check")
    assert hasattr(rlm_core, "__version__")


def test_optional_adversarial_exports_are_consistent():
    optional_names = [
        "AdversarialConfig",
        "ValidationContext",
        "ValidationResult",
        "IssueSeverity",
    ]
    present = [hasattr(rlm_core, name) for name in optional_names]

    # Optional bundle should either be fully present or fully absent.
    assert all(present) or not any(present)

    if all(present):
        for name in optional_names:
            assert name in rlm_core.__all__
    else:
        for name in optional_names:
            assert name not in rlm_core.__all__
