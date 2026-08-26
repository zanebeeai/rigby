"""Prompt to certified motion, end to end, with a trace of how it got there.

The whole pipeline in one place, in the order it happens:

    prompt
      -> firewall       is this even the kind of thing the system does?
      -> planner        body-neutral schema program (no metres, no joints)
      -> binder         does *this* robot have a certified primitive for it?
      -> grounder       schema terms resolved against measured scale
      -> compiler       the certified v2 motion compiler, unchanged
      -> gates          simulate three times, check, agree
      -> result         a motion, or a typed refusal naming the reason

Every step can refuse, and each refusal says something different: the firewall
refuses requests outside the system's scope, the planner refuses wording it does
not recognise, the binder refuses a schema this body has no certified primitive
for, the grounder refuses a term outside measured limits, and the gates refuse a
motion the robot did not actually perform. A single "unsupported" would collapse
five genuinely different answers into one.

Each step also *records* what it decided, whether or not it refused. Rigby's own
lesson, restated: a pipeline that reports only its verdict cannot be debugged,
and one that reports only its failures cannot be trusted about its successes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
from rigby_v2.motion.compiler import compile_motion_program
from rigby_v2.motion.errors import MotionCompilationError
from rigby_v2.motion.trajectory import CandidateTrajectoryV1

from .binding import BoundMotion, bind
from .contracts import RobotAssetManifestV1
from .errors import GeneralFailureCode, RigbyGeneralError
from .gates import CertificationResult, GatePolicy, certify
from .planner import OfflineSchemaPlanner
from .primitives.library import BindingFailure, PrimitiveRecord
from .schema.inventory import SchemaInventory, afforded_entries
from .schema.program import MotionSchemaProgramV1
from .trace import RunTrace


SAMPLE_HZ = 240


@dataclass(frozen=True, slots=True)
class RunResult:
    """What came back, whether or not it succeeded."""

    prompt: str
    robot_id: str
    accepted: bool
    schema_program: MotionSchemaProgramV1 | None = None
    bound: BoundMotion | None = None
    trajectory: CandidateTrajectoryV1 | None = None
    certification: CertificationResult | None = None
    failure_code: str | None = None
    failure_stage: str | None = None
    failure_detail: str | None = None
    elapsed_seconds: float = 0.0
    trace: RunTrace | None = None

    @property
    def duration_s(self) -> float:
        if self.bound is None:
            return 0.0
        return self.bound.grounded.program.duration_s

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "prompt": self.prompt,
            "robot_id": self.robot_id,
            "accepted": self.accepted,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.schema_program is not None:
            payload["schema_keys"] = list(self.schema_program.canonical_keys)
            payload["role_normalized_hash"] = self.schema_program.role_normalized_hash()
        if self.bound is not None:
            payload["duration_s"] = round(self.duration_s, 3)
            payload["bound_schema_keys"] = list(self.bound.program.canonical_keys)
            payload["certified_primitives_used"] = self.bound.used_certified_primitives
            payload["region_substitutions"] = self.bound.substitutions
        if self.certification is not None:
            payload["tracking_error_m"] = round(
                float(self.certification.trace.tracking_error_m.max()), 6
            )
            payload["replay_agreement"] = len(set(self.certification.replay_hashes)) == 1
        if not self.accepted:
            payload["failure"] = {
                "code": self.failure_code,
                "stage": self.failure_stage,
                "detail": self.failure_detail,
            }
        return payload


def _robot_summary(manifest: RobotAssetManifestV1) -> dict[str, object]:
    morphology = manifest.morphology
    scale = morphology.scale
    frame = morphology.intrinsic_frame
    return {
        "robot_id": manifest.rig_id,
        "morphology_class": morphology.morphology_class.value,
        "dof": len(manifest.dofs),
        "effectors": [
            {
                "name": effector.name,
                "kind": effector.kind.value,
                "aperture_m": effector.max_aperture_m,
            }
            for effector in morphology.effectors
        ],
        "reach_radius_m": scale.reach_radius_m,
        "characteristic_length_m": scale.characteristic_length_m,
        "neutral_speed_mps": scale.neutral_speed_mps,
        "payload_kg": scale.payload_kg,
        "total_mass_kg": scale.total_mass_kg,
        "intrinsic_frame": {
            "source": frame.source.value,
            "confidence": frame.confidence,
            "evidence": list(frame.evidence),
        },
    }


def _segment_rows(program: MotionSchemaProgramV1) -> list[dict[str, object]]:
    """The schema program laid out as Talmy's slots, one row per segment."""

    rows = []
    for segment in program.segments:
        schema = segment.motion_schema
        rows.append(
            {
                "segment_id": segment.segment_id,
                "schema": schema.canonical_key,
                "vector": schema.vector.value if schema.vector else None,
                "conformation": (
                    schema.conformation.value if schema.conformation else None
                ),
                "deixis": schema.deixis.value if schema.deixis else None,
                "contour": schema.contour.value if schema.contour else None,
                "stative": schema.stative.value if schema.stative else None,
                "figure": segment.figure.role.value,
                "ground": segment.ground.role.value,
                "remove": segment.region.remove.value,
                "frame": segment.frame.value,
                "boundary": segment.boundary.value,
                "manner": {
                    axis: getattr(segment.manner, axis)
                    for axis in (
                        "speed",
                        "effort",
                        "smoothness",
                        "rhythm",
                        "amplitude",
                        "repetition",
                        "precision",
                    )
                },
                "repetition_count": segment.manner.repetition_count,
            }
        )
    return rows


def answer(
    prompt: str,
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    inventory: SchemaInventory,
    records: tuple[PrimitiveRecord, ...],
    failures: tuple[BindingFailure, ...],
    *,
    planner: OfflineSchemaPlanner | None = None,
    policy: GatePolicy | None = None,
    seed: int = 0,
) -> RunResult:
    started = time.perf_counter()
    resolved_planner = planner or OfflineSchemaPlanner(inventory)
    trace = RunTrace(prompt=prompt, robot_id=manifest.rig_id)
    trace.robot = _robot_summary(manifest)

    def refuse(stage: str, code: str, detail: str, **extra) -> RunResult:
        trace.accepted = False
        trace.failure = {"stage": stage, "code": code, "detail": detail[:600]}
        trace.elapsed_seconds = time.perf_counter() - started
        return RunResult(
            prompt=prompt,
            robot_id=manifest.rig_id,
            accepted=False,
            failure_stage=stage,
            failure_code=code,
            failure_detail=detail[:400],
            elapsed_seconds=trace.elapsed_seconds,
            trace=trace,
            **extra,
        )

    afforded = afforded_entries(inventory, manifest.morphology)

    # -- planning (the firewall runs inside it, before recognition) --------
    mark = time.perf_counter()
    try:
        schema_program = resolved_planner.plan(prompt, afforded=afforded)
    except RigbyGeneralError as error:
        elapsed = (time.perf_counter() - mark) * 1000.0
        stage = (
            "firewall"
            if error.code is GeneralFailureCode.UNSUPPORTED_MORPHOLOGY
            else "planning"
        )
        trace.record(
            stage,
            "refused",
            elapsed,
            str(error),
            failure_code=error.code.value,
            afforded_schemas=len(afforded),
            **{k: v for k, v in error.details.items() if k != "prompt"},
        )
        return refuse(stage, error.code.value, str(error))

    planning_ms = (time.perf_counter() - mark) * 1000.0
    trace.schema_program = {
        **schema_program.model_dump(mode="json"),
        "canonical_keys": list(schema_program.canonical_keys),
        "segments_readable": _segment_rows(schema_program),
    }
    trace.role_normalized_hash = schema_program.role_normalized_hash()
    trace.record(
        "firewall", "ok", 0.0, "in scope: the request names motion this system performs"
    )
    trace.record(
        "planning",
        "ok",
        planning_ms,
        f"read as {len(schema_program.segments)} segment(s): "
        + ", ".join(item.entry_id for item in resolved_planner.last_trace),
        cues=[
            {
                "clause": item.clause,
                "entry_id": item.entry_id,
                "remove": item.remove,
                "manner": item.manner,
            }
            for item in resolved_planner.last_trace
        ],
        afforded_schemas=len(afforded),
        role_normalized_hash=trace.role_normalized_hash,
    )

    # -- binding ----------------------------------------------------------
    mark = time.perf_counter()
    try:
        bound = bind(
            schema_program, manifest, model, inventory, records, failures, seed=seed
        )
    except RigbyGeneralError as error:
        # Merged rather than splatted alongside the explicit keys: an
        # UnbindableSegment already carries its own ``failure_code`` in
        # ``details``, and passing both crashed the refusal path -- the one path
        # whose whole job is to not crash.
        detail = dict(error.details)
        detail.setdefault("failure_code", error.code.value)
        detail["library_size"] = len(records)
        trace.record(
            "binding",
            "refused",
            (time.perf_counter() - mark) * 1000.0,
            str(error),
            **detail,
        )
        return refuse(
            "binding", error.code.value, str(error), schema_program=schema_program
        )

    trace.bindings = [
        {
            "segment_id": item.segment_id,
            "segment_key": item.segment_key,
            "entry_id": item.entry_id,
            "certified_primitive": item.record.record_id if item.record else None,
            "certified_remove": item.record.remove if item.record else None,
            "substituted": item.substituted,
        }
        for item in bound.bindings
    ]
    trace.record(
        "binding",
        "ok",
        (time.perf_counter() - mark) * 1000.0,
        f"{bound.used_certified_primitives} certified primitive(s) matched"
        + (
            f", {bound.substitutions} region substitution(s)"
            if bound.substitutions
            else ""
        ),
        library_size=len(records),
    )

    grounded_program = bound.grounded.program
    trace.grounded = {
        "duration_s": round(grounded_program.duration_s, 4),
        "phases": [
            {
                "phase_id": phase.phase_id,
                "kind": phase.kind.value,
                "start_s": phase.start_s,
                "end_s": phase.end_s,
                "energy": phase.energy,
            }
            for phase in grounded_program.phases
        ],
        "tracks": [
            {
                "track_id": track.track_id,
                "target_site": track.target,
                "owner": track.owner,
                "keyframes": len(track.keyframes),
            }
            for track in grounded_program.tracks
        ],
        "waypoints": bound.grounded.waypoint_count,
        "figure_sites": list(bound.grounded.figure_sites),
        "grounded_against": dict(grounded_program.metadata.get("grounded_against", {})),
        "program_sha256": grounded_program.content_hash(),
        "inventory_sha256": grounded_program.metadata.get("inventory_sha256"),
    }
    trace.record(
        "grounding",
        "ok",
        0.0,
        f"{bound.grounded.waypoint_count} waypoints over "
        f"{grounded_program.duration_s:.1f}s, against a measured reach of "
        f"{manifest.morphology.scale.reach_radius_m:.3f} m",
        **trace.grounded["grounded_against"],
    )

    # -- compilation ------------------------------------------------------
    mark = time.perf_counter()
    try:
        trajectory = compile_motion_program(
            grounded_program, model, manifest, sample_hz=SAMPLE_HZ
        )
    except MotionCompilationError as error:
        trace.record(
            "compilation",
            "refused",
            (time.perf_counter() - mark) * 1000.0,
            str(error),
            failure_code=error.reason.value,
        )
        return refuse(
            "compilation",
            error.reason.value,
            str(error),
            schema_program=schema_program,
            bound=bound,
        )

    trace.record(
        "compilation",
        "ok",
        (time.perf_counter() - mark) * 1000.0,
        f"{trajectory.qpos.shape[0]} frames at {SAMPLE_HZ} Hz",
        frames=int(trajectory.qpos.shape[0]),
        sample_hz=SAMPLE_HZ,
    )

    # -- certification ----------------------------------------------------
    mark = time.perf_counter()
    result = certify(
        model,
        manifest,
        trajectory,
        site_name=bound.grounded.figure_sites[0],
        policy=policy,
    )
    certification_ms = (time.perf_counter() - mark) * 1000.0
    trace.certification = {
        "certified": result.certified,
        "repeats": result.repeats,
        "replay_agreement": len(set(result.replay_hashes)) == 1,
        "replay_hashes": [digest[:16] for digest in result.replay_hashes],
        "tracking_error_m": round(float(result.trace.tracking_error_m.max()), 6),
        "base_drift_m": round(result.trace.base_drift_m, 9),
        "unexpected_contacts": [list(pair) for pair in result.trace.unexpected_contacts],
        "violations": [
            {
                "code": violation.code.value,
                "detail": violation.detail,
                "measured": round(violation.measured, 6),
                "limit": round(violation.limit, 6),
                "subject": violation.subject,
            }
            for violation in result.violations
        ],
    }

    if not result.certified:
        first = result.violations[0]
        trace.record(
            "certification",
            "refused",
            certification_ms,
            f"{first.code.value}: {first.detail}",
            **trace.certification,
        )
        return refuse(
            "certification",
            "deterministic_gate_failed",
            f"{first.code.value}: {first.detail}",
            schema_program=schema_program,
            bound=bound,
            trajectory=trajectory,
            certification=result,
        )

    trace.record(
        "certification",
        "ok",
        certification_ms,
        f"{result.repeats} identical replays, tracking within "
        f"{trace.certification['tracking_error_m'] * 1000:.1f} mm",
        **trace.certification,
    )
    trace.accepted = True
    trace.elapsed_seconds = time.perf_counter() - started

    return RunResult(
        prompt=prompt,
        robot_id=manifest.rig_id,
        accepted=True,
        schema_program=schema_program,
        bound=bound,
        trajectory=trajectory,
        certification=result,
        elapsed_seconds=trace.elapsed_seconds,
        trace=trace,
    )


def unsupported_reason(result: RunResult) -> str:
    """A one-line explanation suitable for showing a person."""

    if result.accepted:
        return "accepted"
    # Prefer whatever the refusing stage actually said. A recognized-but-
    # unafforded schema and an unrecognized phrase both surface at the planner,
    # and collapsing them into one sentence throws away the more useful half:
    # "this arm has no second hand to pass it to" is a different answer from
    # "I do not know that word".
    if result.failure_detail:
        return result.failure_detail
    if result.failure_stage == "planning":
        return "nothing in that request names a motion this system knows"
    if result.failure_stage == "binding":
        return "this robot has no certified primitive for it"
    if result.failure_stage == "compilation":
        return "the motion could not be compiled for this robot"
    return "the robot did not perform the motion within its gates"


_ = GeneralFailureCode  # re-exported for callers matching on failure codes
