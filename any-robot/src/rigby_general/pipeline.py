"""End-to-end robot intake: a path in, a measured and simulable robot out.

One function so there is exactly one order in which these steps happen. The order
is load-bearing:

1. Scan the raw text and refuse anything unsafe **before** a parser sees it. Once
   MuJoCo has read the file, a plugin has already been resolved and an absolute
   mesh path has already been followed.
2. Compile as delivered, so morphology is measured against what was uploaded
   rather than against something already modified.
3. Measure. Nothing before this point knows where the gripper is.
4. Write back sites and actuators, and recompile.
"""

from __future__ import annotations

import mujoco
import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path

from .contracts import RobotAssetManifestV1, RobotMorphologyV1
from .errors import GeneralFailureCode, MorphologyError
from .ingest import (
    FinalizedRobot,
    check_integrity,
    check_rest_contacts,
    finalize,
    load_model,
)
from .ingest.integrity import (
    IntegrityReport,
    raise_for_violations,
    scan_source_text,
)
from .ingest.loader import release_sandbox, urdf_velocity_limits
from .morphology import analyze


MINIMUM_POSITIONING_DOF = 2
"""Below this, a chain traces a line or a circle rather than reaching a point."""


@dataclass(frozen=True, slots=True)
class IngestedRobot:
    manifest: RobotAssetManifestV1
    morphology: RobotMorphologyV1
    integrity: IntegrityReport
    mjcf_xml: str
    finalized: FinalizedRobot

    @property
    def robot_id(self) -> str:
        return self.manifest.rig_id


STRUCTURAL_OVERLAP_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0)
"""Where along each joint's range a parent-child overlap is re-checked."""


def _structural_overlaps(
    model: mujoco.MjModel,
) -> tuple[tuple[str, str], ...]:
    """Parent-child pairs authored inside one another, which no pose can undo.

    The exclusion set was measured for guarding IK, and there dropping adjacent
    links is right: they meet at the joint, and listing them would drown the
    real signal. Physics needs the opposite question asked. The SO-ARM101's
    shoulder is authored 26 mm inside the base it turns on, and the only joint
    between them is a vertical pan -- rotating it cannot separate them, so the
    penetration is there at every configuration the arm can adopt. Nothing
    excluded it, so the solver spent all 5252 steps of every attempt pushing
    them apart: the controller demanded 3.2 kN*m against a 10 N*m limit,
    clipped on 1262 of 1441 steps, and the hand tracked 131 degrees behind a
    path whose own IK residual was one millimetre.

    A pair that overlaps at *every* configuration of the joints between them is
    a fact about the upload, not about the motion, which is the same test the
    non-adjacent pairs already get.
    """

    from .morphology.measure import joint_range, neutral_qpos

    data = mujoco.MjData(model)
    rest = neutral_qpos(model)
    geoms: dict[int, list[int]] = {}
    for geom in range(model.ngeom):
        if int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]):
            geoms.setdefault(int(model.geom_bodyid[geom]), []).append(geom)

    found: list[tuple[str, str]] = []
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        if parent == 0 or not geoms.get(body) or not geoms.get(parent):
            continue
        joints = [
            int(model.body_jntadr[body]) + offset
            for offset in range(int(model.body_jntnum[body]))
        ]
        samples = []
        for fraction in STRUCTURAL_OVERLAP_SAMPLES:
            data.qpos[:] = rest
            for joint in joints:
                low, high = joint_range(model, joint)
                data.qpos[int(model.jnt_qposadr[joint])] = low + fraction * (
                    high - low
                )
            mujoco.mj_kinematics(model, data)
            samples.append(
                min(
                    float(mujoco.mj_geomDistance(model, data, one, other, 1.0, None))
                    for one in geoms[body]
                    for other in geoms[parent]
                )
            )
        if samples and max(samples) < 0.0:
            first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)
            if first and second:
                found.append(tuple(sorted((first, second))))
    return tuple(sorted(set(found)))


def ingest_robot(source_path: Path, *, robot_id: str | None = None) -> IngestedRobot:
    """Accept a URDF or MJCF and return everything measured about it."""

    identity = robot_id or source_path.parent.name

    # Step 1 happens before step 2 for a reason. By the time MuJoCo has parsed
    # the file, a plugin directive has already been resolved and an absolute or
    # ``package://`` mesh path has already been followed -- and the failure
    # surfaces as an unhelpful parse error rather than as the security refusal it
    # actually is.
    raise_for_violations(identity, scan_source_text(source_path.read_bytes()))

    loaded = load_model(source_path, robot_id=robot_id)
    try:
        return _measure(loaded)
    finally:
        # Only now: the spec is recompiled during finalize, and it resolves mesh
        # paths against the staged directory.
        release_sandbox(loaded)


def _measure(loaded) -> IngestedRobot:
    report = check_integrity(loaded)
    report.raise_for_status(loaded.robot_id)

    velocity_limits = (
        urdf_velocity_limits(loaded.source_bytes)
        if loaded.source_format == "urdf"
        else {}
    )
    morphology = analyze(loaded, velocity_limits=velocity_limits)

    if not morphology.is_supported:
        raise MorphologyError(
            GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
            f"{loaded.robot_id}: morphology class "
            f"{morphology.morphology_class.value!r} is not supported here",
            details={"morphology_class": morphology.morphology_class.value},
        )

    # Placing an effector somewhere takes at least two independent positioning
    # axes. One is a slide or a turntable: it traces a line or a circle and
    # cannot reach a point off it.
    #
    # This is the check that distinguishes a manipulator from a mechanism that
    # merely moves. Without it a cart on a rail ingests happily as a
    # "fixed_base_arm" whose "tool tip" is a freely swinging pole and whose
    # measured reach is fifteen metres -- every number technically correct and
    # the whole description meaningless.
    best = max((chain.positioning_dof for chain in morphology.chains), default=0)
    # A hand is exempt, and the exemption is narrow: it must have been *measured*
    # to close, with opposing groups. That is the difference between the two
    # things this check used to conflate -- a cart on a rail positions nothing and
    # holds nothing, while a five-fingered hand positions nothing and holds
    # everything. What it affords is limited downstream rather than here.
    holds = any(
        effector.can_grasp and len(effector.opposition_groups) >= 2
        for effector in morphology.effectors
    )
    if best < MINIMUM_POSITIONING_DOF and not holds:
        raise MorphologyError(
            GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
            f"{loaded.robot_id}: the best chain has {best} positioning "
            f"degree(s) of freedom, and placing an effector needs at least "
            f"{MINIMUM_POSITIONING_DOF}. This is a mechanism that moves, not one "
            "that positions.",
            details={
                "positioning_dof": best,
                "required": MINIMUM_POSITIONING_DOF,
                "chains": [
                    {"chain_id": chain.chain_id, "positioning_dof": chain.positioning_dof}
                    for chain in morphology.chains
                ],
            },
        )

    contacts, authored_overlaps = check_rest_contacts(loaded.model, morphology)
    raise_for_violations(loaded.robot_id, contacts)
    report = replace(report, notes=report.notes + authored_overlaps)

    finalized = finalize(
        loaded,
        morphology,
        ungateable_pairs=_structural_overlaps(loaded.model) + tuple(
            (subject.split("|", 1)[0], subject.split("|", 1)[1])
            for subject in (note.subject for note in authored_overlaps)
            if "|" in subject
        ),
    )
    return IngestedRobot(
        manifest=finalized.manifest,
        morphology=finalized.morphology,
        integrity=report,
        mjcf_xml=finalized.mjcf_xml,
        finalized=finalized,
    )
