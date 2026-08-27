from __future__ import annotations

import math

import pytest

from rigby_v2.benchmark import BenchmarkCaseKind, load_benchmark_manifest
from rigby_v2.guardrails import (
    AssetIntakeV1,
    CapabilityFirewallError,
    FirewallOutcome,
    FirewallReason,
    validate_asset,
    validate_prompt,
)
from rigby_v2.planner import OpenAISemanticPlanner

pytestmark = pytest.mark.fast


class _ForbiddenResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs):
        del kwargs
        self.calls += 1
        raise AssertionError("model boundary must not be reached")


class _ForbiddenClient:
    def __init__(self) -> None:
        self.responses = _ForbiddenResponses()


def test_sealed_adversarial_prompts_are_typed_without_reading_labels() -> None:
    manifest = load_benchmark_manifest()
    adversarial = tuple(
        case
        for case in manifest.cases
        if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL
    )
    decisions = tuple(validate_prompt(case.prompt) for case in adversarial)

    assert len(decisions) == 100
    assert all(
        decision.outcome.value == case.expected_outcome.value
        for case, decision in zip(adversarial, decisions, strict=True)
    )
    assert len({decision.decision_hash for decision in decisions}) == 100

    # Tags and expected labels are audit data, not classifier inputs.
    first = adversarial[0]
    relabeled = first.model_copy(
        update={"tags": ("deliberately", "wrong"), "expected_outcome": "infeasible"}
    )
    assert validate_prompt(relabeled.prompt) == validate_prompt(first.prompt)


def test_all_300_supported_prompts_and_legitimate_near_misses_remain_allowed() -> None:
    manifest = load_benchmark_manifest()
    supported = tuple(
        case for case in manifest.cases if case.kind is BenchmarkCaseKind.SUPPORTED
    )
    near_misses = (
        "Fold the rigid hinged panel along its modeled hinge.",
        "Tilt a rigid cup containing no simulated liquid.",
        "Use both hands to lift one rigid tray.",
        "Perform a robot-style wave in local simulation.",
        "With feet planted, touch a target 50 centimeters away.",
        "Keep the left hand still while lifting the block with the right hand.",
        "Load embedded mesh assets/tools/hammer.obj as a visual mesh.",
        "Use a concave visual mesh with convex decomposition for its dynamic collider.",
        "Open the articulated drawer, press the button, and release.",
    )

    assert len(supported) == 300
    assert all(validate_prompt(case.prompt).outcome is FirewallOutcome.ALLOW for case in supported)
    assert all(validate_prompt(prompt).outcome is FirewallOutcome.ALLOW for prompt in near_misses)


@pytest.mark.parametrize(
    ("asset", "reason"),
    (
        (AssetIntakeV1(asset_paths=("../../secret.obj",)), FirewallReason.PATH_TRAVERSAL),
        (AssetIntakeV1(native_plugins=("evil.dll",)), FirewallReason.PLUGIN_INJECTION),
        (AssetIntakeV1(mass_kg=math.nan), FirewallReason.NONFINITE_OR_INVALID_INERTIA),
        (
            AssetIntakeV1(dynamic=True, collision_representation="raw_concave_mesh"),
            FirewallReason.RAW_CONCAVE_DYNAMIC_COLLIDER,
        ),
    ),
)
def test_structured_asset_intake_fails_closed(asset, reason) -> None:
    decision = validate_asset(asset)
    assert decision.outcome is FirewallOutcome.INVALID_ASSET
    assert decision.reason is reason


def test_valid_embedded_convex_asset_is_allowed() -> None:
    decision = validate_asset(
        AssetIntakeV1(
            asset_paths=("meshes/tool-hull.obj",),
            mass_kg=1.2,
            inertia_diagonal_kg_m2=(0.02, 0.03, 0.04),
            dynamic=True,
            collision_representation="convex_decomposition",
        )
    )
    assert decision.outcome is FirewallOutcome.ALLOW


def test_real_semantic_planner_hook_rejects_all_100_before_model_call() -> None:
    client = _ForbiddenClient()
    planner = OpenAISemanticPlanner(client=client, model="must-not-run")
    adversarial = (
        case
        for case in load_benchmark_manifest().cases
        if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL
    )
    for case in adversarial:
        with pytest.raises(CapabilityFirewallError) as caught:
            planner.plan(
                prompt=case.prompt,
                rig=None,  # type: ignore[arg-type]
                rig_artifact_sha256="not-consumed",
                scene=None,  # type: ignore[arg-type]
                retrieval_release="not-consumed",
                seed=case.seed,
            )
        assert caught.value.outcome.value == case.expected_outcome.value
    assert client.responses.calls == 0

