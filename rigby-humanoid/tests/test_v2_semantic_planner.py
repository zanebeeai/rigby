from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.lib._pydantic import to_strict_json_schema

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.contracts import (
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
)
from rigby_v2.library import CompactExample
from rigby_v2.hashing import hash_file
from rigby_v2.planner import OpenAISemanticPlanner, PlannedMotionV1, PlannerError
from rigby_v2.rigging import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene

pytestmark = pytest.mark.medium


def _program(joint: str = "spine_flex") -> MotionProgramV2:
    return MotionProgramV2(
        program_id="model-authored-id",
        source_text="model-authored-text",
        duration_s=1.0,
        rig_id="model-invented-rig",
        scene_id="model-invented-scene",
        seed=999,
        phases=(
            MotionPhaseV2(
                phase_id="action",
                kind=PhaseKind.ACTION,
                start_s=0.0,
                end_s=1.0,
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="posture",
                target=joint,
                owner="planner",
                keyframes=(
                    MotionKeyframeV2(time_s=0.0, joint_values={joint: 0.0}),
                    MotionKeyframeV2(time_s=1.0, joint_values={joint: 0.01}),
                ),
            ),
        ),
    )


class FakeResponses:
    def __init__(self, program: MotionProgramV2) -> None:
        self.program = program
        self.calls = []

    def parse(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=PlannedMotionV1(
                program=self.program,
                rationale="Use a conservative balanced motion with an explicit action phase.",
            )
        )


def test_structured_planner_binds_rig_scene_seed_and_compact_rag_context(
    tmp_path: Path,
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "drawer",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    responses = FakeResponses(_program())
    planner = OpenAISemanticPlanner(
        client=SimpleNamespace(responses=responses),
        model="explicit-planner-model",
    )

    plan = planner.plan(
        prompt="Open the drawer carefully.",
        rig=rig.manifest,
        rig_artifact_sha256=rig.reference.sha256,
        scene=scene.manifest,
        retrieval_release="release-7",
        retrieval_examples=(
            CompactExample("record-a", "Example: setup, contact, pull, settle."),
            CompactExample("record-b", "Example: preshape the hand before contact."),
        ),
        seed=17,
    )

    assert len(responses.calls) == 1
    assert responses.calls[0]["text_format"] is PlannedMotionV1
    assert responses.calls[0]["store"] is False
    assert plan.base_program.rig_id == rig.manifest.rig_id
    assert plan.base_program.scene_id == scene.manifest.scene_id
    assert plan.base_program.seed == 17
    assert plan.base_program.source_text == "Open the drawer carefully."
    assert plan.base_program.metadata["retrieval_record_ids"] == (
        "record-a",
        "record-b",
    )
    assert len(plan.retrieval_examples) == 2
    assert all(len(value) == 64 for value in plan.retrieval_examples)


def test_planner_rejects_invented_physical_dof(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("small", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "drawer",
        profile="small",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    planner = OpenAISemanticPlanner(
        client=SimpleNamespace(responses=FakeResponses(_program("invented_joint"))),
        model="explicit-planner-model",
    )
    with pytest.raises(PlannerError, match="invented rig DOFs"):
        planner.plan(
            prompt="Open the drawer.",
            rig=rig.manifest,
            rig_artifact_sha256=rig.reference.sha256,
            scene=scene.manifest,
            retrieval_release="release-1",
            seed=1,
        )


def test_planner_response_contract_is_openai_strict_schema_compatible() -> None:
    schema = to_strict_json_schema(PlannedMotionV1)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            assert value.get("additionalProperties") is not True
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)


def test_sealed_live_planner_smoke_uses_strict_unstored_response() -> None:
    root = Path(__file__).resolve().parents[1] / "assets" / "v2" / "benchmark"
    report = root / "live_planner_smoke.json"
    expected = report.with_suffix(".json.sha256").read_text(encoding="utf-8").split()[0]
    assert hash_file(report) == expected
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["passed"] is True
    assert value["response_stored"] is False
    assert value["strict_schema_accepted"] is True
    assert value["persisted_contract_validated"] is True
    assert value["invented_dofs"] == 0
