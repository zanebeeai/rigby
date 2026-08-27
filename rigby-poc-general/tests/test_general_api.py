"""The health surface must expose base-tree identity and never a secret."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rigby_general.app import create_app, health_payload, provider_readiness
from rigby_general.config import GeneralSettings, base_tree_fingerprint


def test_health_reports_the_base_tree_fingerprint() -> None:
    """A path dependency on an untracked tree needs some identity."""

    client = TestClient(create_app(GeneralSettings()))
    body = client.get("/api/v3/health").json()

    assert body["service"] == "rigby-general"
    assert body["base_tree"]["package"] == "rigby_v2"
    assert len(body["base_tree"]["sha256"]) == 64
    assert body["base_tree"]["file_count"] > 0


def test_base_tree_fingerprint_is_stable_within_a_process() -> None:
    assert base_tree_fingerprint().sha256 == base_tree_fingerprint().sha256


def test_health_declares_only_the_supported_morphology_classes() -> None:
    body = health_payload(GeneralSettings())

    assert body["supported_morphology_classes"] == [
        "dexterous_effector",
        "fixed_base_arm",
        "fixed_base_bimanual",
    ]


def test_provider_readiness_reports_presence_not_values() -> None:
    """Follows the v2 rule: booleans and a fingerprint, never the routing itself."""

    env = {
        "OPENAI_API_KEY": "sk-should-never-appear",
        "OPENAI_PLANNER_MODEL": "some-planner-model",
    }
    payload = provider_readiness(env)
    serialized = repr(payload)

    assert payload["openai_key_present"] is True
    assert payload["routing_configured"]["OPENAI_PLANNER_MODEL"] is True
    assert payload["routing_configured"]["OPENAI_JUDGE_MODEL"] is False
    assert "sk-should-never-appear" not in serialized
    assert "some-planner-model" not in serialized


def test_blank_key_is_not_present() -> None:
    assert provider_readiness({"OPENAI_API_KEY": "   "})["openai_key_present"] is False


def test_settings_reject_an_incompatible_render_rate() -> None:
    import pytest

    with pytest.raises(ValueError, match="integer divisor"):
        GeneralSettings(physics_hz=240, render_fps=7)
