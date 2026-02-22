from __future__ import annotations

import tess_vetter.api as api


def test_legacy_dynamic_registry_tracks_expected_categories() -> None:
    registry = api.get_legacy_dynamic_export_policy_registry()

    assert registry["generate_control"] == "variadic_dispatch"
    assert registry["plot_odd_even"] == "plotting_variadic"
    assert registry["ExportFormat"] == "typing_alias_artifact"
    assert registry["vet"] == "alias_artifact"


def test_legacy_dynamic_helpers_are_consistent() -> None:
    names = api.list_legacy_dynamic_exports()
    assert names == sorted(names)
    assert "plot_odd_even" in names
    assert "generate_control" in names
    assert api.is_legacy_dynamic_export("generate_control") is True
    assert api.is_legacy_dynamic_export("vet_candidate") is False


def test_actionable_vs_unloadable_classification_helpers() -> None:
    assert api.is_agent_actionable_export("vet_candidate") is True
    assert api.is_agent_actionable_export("run_periodogram") is True
    assert api.is_agent_actionable_export("plot_odd_even") is False
    assert api.is_agent_actionable_export("generate_control") is False
    assert api.is_agent_actionable_export("vet") is False
    assert api.is_agent_actionable_export("does_not_exist") is False

    assert api.is_unloadable_export("plot_odd_even") is True
    assert api.is_unloadable_export("vet_candidate") is False
