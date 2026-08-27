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
        ungateable_pairs=tuple(
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
