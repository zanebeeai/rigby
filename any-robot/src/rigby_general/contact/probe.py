"""Run a grasp attempt and record it the way a prompt run is recorded.

Contact needs its own entry point, and the reason is worth stating rather than
hiding behind a shared function signature. A prompt run goes
planner -> binder -> grounder, and none of those steps apply here: contact
schemas are *unafforded* until a grasp certifies, so asking a robot to "pick it
up" in words is genuinely refused. A probe deliberately bypasses that to exercise
the physics directly, which is a different kind of evidence and is labelled as
such (``kind = "contact_probe"``).

Both kinds land in the same store, because the question a person actually has --
"what happened when it tried?" -- is the same either way, and splitting the
record in two would mean the failed grasps quietly stop being visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..contracts import EffectorV1, RobotAssetManifestV1
from ..errors import RigbyGeneralError
from ..grounding.grounder import figure_site_for
from ..grounding.workspace import build_workspace_frame
from ..scenes.block import GraspScene, build_grasp_scene
from ..trace import RunTrace
from .grasp import GraspResult, attempt_grasp


PROBE_PROMPT = "pick up the block"


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    result: GraspResult | None
    scene: GraspScene | None
    trace: RunTrace


def probe_grasp(
    manifest: RobotAssetManifestV1,
    mjcf_xml: str,
    model,
    *,
    prompt: str = PROBE_PROMPT,
) -> ProbeOutcome:
    """Build a scene for this robot, attempt a pick, and record every step."""

    from ..run import _robot_summary

    started = time.perf_counter()
    trace = RunTrace(prompt=prompt, robot_id=manifest.rig_id, kind="contact_probe")
    trace.robot = _robot_summary(manifest)

    morphology = manifest.morphology
    grippers = morphology.grasping_effectors
    if not grippers:
        trace.record(
            "effector",
            "refused",
            0.0,
            "nothing on this robot was measured to close, so there is nothing to "
            "grasp with",
            effectors=[item.kind.value for item in morphology.effectors],
        )
        trace.failure = {
            "stage": "effector",
            "code": "no_gripper",
            "detail": "no effector on this robot closes; a grasp is not defined for it",
        }
        trace.elapsed_seconds = time.perf_counter() - started
        return ProbeOutcome(result=None, scene=None, trace=trace)

    effector: EffectorV1 = grippers[0]
    trace.record(
        "effector",
        "ok",
        0.0,
        f"{effector.kind.value} with a measured aperture of "
        f"{effector.max_aperture_m * 1000:.0f} mm",
        effector_name=effector.name,
        kind=effector.kind.value,
        members=list(effector.member_bodies),
        opposition_groups=[list(group) for group in effector.opposition_groups],
        grip_joints=list(effector.grip_joints),
        aperture_m=effector.max_aperture_m,
    )

    chain = next(
        item for item in morphology.chains if item.chain_id == effector.chain_id
    )
    site = figure_site_for(manifest, effector.chain_id)
    frame = build_workspace_frame(model, morphology, chain, figure_site=site)

    mark = time.perf_counter()
    try:
        scene = build_grasp_scene(manifest, mjcf_xml, effector, frame)
    except RigbyGeneralError as error:
        trace.record(
            "scene", "refused", (time.perf_counter() - mark) * 1000.0, str(error)
        )
        trace.failure = {
            "stage": "scene",
            "code": error.code.value,
            "detail": str(error),
        }
        trace.elapsed_seconds = time.perf_counter() - started
        return ProbeOutcome(result=None, scene=None, trace=trace)

    trace.record(
        "scene",
        "ok",
        (time.perf_counter() - mark) * 1000.0,
        f"{scene.block_half_extent_m * 2000:.0f} mm block, "
        f"{scene.block_mass_kg * 1000:.0f} g, placed inside the measured envelope",
        block_size_m=round(scene.block_half_extent_m * 2, 5),
        block_mass_kg=round(scene.block_mass_kg, 5),
        block_position_m=[round(float(v), 4) for v in scene.block_position_m],
        support_height_m=round(scene.support_height_m, 4),
        sized_from="measured gripper aperture",
        placed_from="measured reach envelope",
    )

    mark = time.perf_counter()
    result = attempt_grasp(manifest, scene, effector, frame)
    elapsed_ms = (time.perf_counter() - mark) * 1000.0

    required_lift = 0.8 * 2 * scene.block_half_extent_m
    trace.grasp = {
        "certified": result.certified,
        "block_size_m": round(scene.block_half_extent_m * 2, 5),
        "block_mass_kg": round(scene.block_mass_kg, 5),
        "opposition_achieved": result.opposition_achieved,
        "lift_height_m": round(result.lift_height_m, 5),
        "required_lift_m": round(required_lift, 5),
        "final_height_m": round(result.final_height_m, 5),
        "grip_force_n": round(result.peak_force_n, 3),
        "carry_offset_m": round(result.carry_offset_m, 5),
        "penetration_m": round(result.max_penetration_m, 5),
        "duration_s": result.duration_s,
        "equality_constraints": int(scene.model.neq),
        "violations": [
            {
                "code": violation.code,
                "detail": violation.detail,
                "measured": round(violation.measured, 5),
                "limit": round(violation.limit, 5),
            }
            for violation in result.violations
        ],
    }

    trace.record(
        "closure",
        "ok" if result.opposition_achieved else "refused",
        elapsed_ms,
        (
            f"opposing members both made contact at {result.peak_force_n:.1f} N"
            if result.opposition_achieved
            else "opposing members never both made contact"
        ),
        opposition_achieved=result.opposition_achieved,
        grip_force_n=round(result.peak_force_n, 3),
        penetration_m=round(result.max_penetration_m, 5),
        control="force, not position",
    )
    trace.record(
        "grasp_gates",
        "ok" if result.certified else "refused",
        0.0,
        (
            f"held: lifted {result.lift_height_m * 1000:.0f} mm of a required "
            f"{required_lift * 1000:.0f} mm, no weld"
            if result.certified
            else "; ".join(
                f"{violation.code}" for violation in result.violations
            )
        ),
        **trace.grasp,
    )

    trace.accepted = result.certified
    if not result.certified:
        first = result.violations[0]
        trace.failure = {
            "stage": "grasp_gates",
            "code": first.code,
            "detail": first.detail,
        }
    trace.elapsed_seconds = time.perf_counter() - started
    return ProbeOutcome(result=result, scene=scene, trace=trace)
