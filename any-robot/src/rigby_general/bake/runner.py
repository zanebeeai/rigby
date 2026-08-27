"""Run every afforded binding and keep the ones that survive.

The bake is where a robot stops being a description and becomes a set of things
it can actually do. Each candidate is ground, compiled, simulated three times and
gated; what passes becomes a certified primitive and what fails becomes a
recorded refusal.

Failures are kept deliberately. A schema that will not certify on this body is
not noise to be discarded -- it is the answer to a question the planner will
later be asked, and having measured it once is what lets the system say *why*
rather than merely refusing.

The budget is a real ceiling, not a suggestion. A bake that runs out of time
stops and says so, and the library it produced is marked incomplete rather than
being presented as a full one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
from rigby_core.motion.compiler import compile_motion_program
from rigby_core.motion.errors import MotionCompilationError

from ..config import base_tree_fingerprint
from ..contracts import RobotAssetManifestV1
from ..errors import GeneralFailureCode, GroundingError, RigbyGeneralError
from ..gates import GatePolicy, certify
from ..grounding import ground
from ..primitives.library import (
    BakeStage,
    BakeSummary,
    BindingFailure,
    PrimitiveRecord,
)
from ..schema.inventory import SchemaInventory
from .enumerate import BindingCandidate, enumerate_bindings


SAMPLE_HZ = 240

# A binding that fails only because it moved too fast is not an infeasible
# binding, it is a binding asked to run at the wrong pace. The analytic velocity
# margin in the grounder gets the timing close, but closed-loop overshoot varies
# with the body and the path, so the bake measures the real overshoot and
# re-grounds slower. Bounded, because a binding that still fails after being
# slowed three times is failing for some other reason.
# Measured: raising this from one to three quadrupled bake time on the smallest
# arm and recovered a single extra primitive. The bindings that fail here mostly
# do not fail *because* of pace -- a tiny arm with generous angular limits
# overshoots in angle no matter how slowly the Cartesian path is walked -- so one
# retry catches the genuinely rushed cases and stopping there costs nothing real.
MAX_PACE_RETRIES = 1
PACE_MARGIN = 1.15
_PACE_GATES = frozenset({"velocity_limit"})


@dataclass(frozen=True, slots=True)
class BakeResult:
    records: tuple[PrimitiveRecord, ...]
    failures: tuple[BindingFailure, ...]
    summary: BakeSummary


def bake_robot(
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    inventory: SchemaInventory,
    *,
    budget_seconds: int = 1_200,
    max_attempts: int = 500,
    policy: GatePolicy | None = None,
    contact_scene: bool = False,
    progress=None,
) -> BakeResult:
    """Ground, compile, simulate and gate every binding this body affords."""

    started = time.perf_counter()
    candidates = enumerate_bindings(
        inventory, manifest.morphology, contact_scene=contact_scene
    )[:max_attempts]

    records: list[PrimitiveRecord] = []
    failures: list[BindingFailure] = []
    attempted = 0
    complete = True
    fingerprint = base_tree_fingerprint().sha256

    for candidate in candidates:
        if time.perf_counter() - started > budget_seconds:
            complete = False
            break
        attempted += 1

        outcome = _attempt(
            candidate, manifest, model, inventory, policy=policy, fingerprint=fingerprint
        )
        if isinstance(outcome, PrimitiveRecord):
            records.append(outcome)
        else:
            failures.append(outcome)

        if progress is not None:
            progress(candidate, outcome, attempted, len(candidates))

    elapsed = time.perf_counter() - started
    summary = BakeSummary(
        robot_id=manifest.rig_id,
        attempted=attempted,
        certified=len(records),
        elapsed_seconds=elapsed,
        budget_seconds=budget_seconds,
        max_attempts=max_attempts,
        complete=complete and attempted == len(candidates),
        inventory_sha256=inventory.sha256,
        base_tree_sha256=fingerprint,
        afforded_entry_ids=tuple(
            sorted({candidate.entry_id for candidate in candidates})
        ),
        covered_entry_ids=tuple(sorted({record.entry_id for record in records})),
    )
    return BakeResult(
        records=tuple(records), failures=tuple(failures), summary=summary
    )


def _attempt(
    candidate: BindingCandidate,
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    inventory: SchemaInventory,
    *,
    policy: GatePolicy | None,
    fingerprint: str,
) -> PrimitiveRecord | BindingFailure:
    def failed(
        stage: BakeStage,
        code: str,
        detail: str,
        *,
        gate: str | None = None,
        measurements: dict[str, float] | None = None,
    ) -> BindingFailure:
        return BindingFailure(
            robot_id=manifest.rig_id,
            entry_id=candidate.entry_id,
            schema_key=candidate.schema_key,
            segment_key=candidate.segment_key,
            remove=candidate.remove.value,
            stage=stage,
            failure_code=code,
            detail=detail[:400],
            failed_gate=gate,
            measurements=measurements or {},
        )

    scale = 1.0
    for attempt in range(MAX_PACE_RETRIES + 1):
        outcome = _attempt_once(
            candidate,
            manifest,
            model,
            inventory,
            policy=policy,
            fingerprint=fingerprint,
            duration_scale=scale,
            failed=failed,
        )
        if isinstance(outcome, PrimitiveRecord):
            return outcome
        if attempt == MAX_PACE_RETRIES or outcome.failed_gate not in _PACE_GATES:
            return outcome
        overshoot = outcome.measurements.get("measured", 0.0) / max(
            outcome.measurements.get("limit", 1.0), 1e-9
        )
        if overshoot <= 1.0:
            return outcome
        scale *= overshoot * PACE_MARGIN
    return outcome  # pragma: no cover - loop always returns


def _attempt_once(
    candidate: BindingCandidate,
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    inventory: SchemaInventory,
    *,
    policy: GatePolicy | None,
    fingerprint: str,
    duration_scale: float,
    failed,
) -> PrimitiveRecord | BindingFailure:
    try:
        grounded = ground(
            candidate.program, manifest, model, inventory, duration_scale=duration_scale
        )
    except GroundingError as error:
        return failed(BakeStage.GROUNDING, error.code.value, str(error))
    except RigbyGeneralError as error:
        stage = (
            BakeStage.AFFORDANCE
            if error.code is GeneralFailureCode.UNAFFORDED_SCHEMA
            else BakeStage.GROUNDING
        )
        return failed(stage, error.code.value, str(error))

    try:
        trajectory = compile_motion_program(
            grounded.program, model, manifest, sample_hz=SAMPLE_HZ
        )
    except MotionCompilationError as error:
        return failed(BakeStage.COMPILATION, error.reason.value, str(error))
    except (ValueError, KeyError) as error:
        return failed(BakeStage.COMPILATION, "invalid_contract", str(error))

    try:
        result = certify(
            model,
            manifest,
            trajectory,
            site_name=grounded.figure_sites[0],
            policy=policy,
        )
    except (ValueError, KeyError) as error:  # pragma: no cover - defensive
        return failed(BakeStage.SIMULATION, "simulation_failed", str(error))

    if not result.certified:
        first = result.violations[0]
        return failed(
            BakeStage.CERTIFICATION,
            "deterministic_gate_failed",
            f"{first.code.value}: {first.detail}",
            gate=first.code.value,
            measurements={
                "measured": round(first.measured, 6),
                "limit": round(first.limit, 6),
            },
        )

    return PrimitiveRecord(
        robot_id=manifest.rig_id,
        entry_id=candidate.entry_id,
        schema_key=candidate.schema_key,
        segment_key=candidate.segment_key,
        remove=candidate.remove.value,
        figure_role=candidate.program.segments[0].figure.role.value,
        ground_role=candidate.program.segments[0].ground.role.value,
        figure_site=grounded.figure_sites[0],
        duration_s=round(grounded.program.duration_s, 4),
        waypoints=grounded.waypoint_count,
        program=grounded.program.model_dump(mode="json"),
        program_sha256=grounded.program.content_hash(),
        trace_sha256=result.replay_hashes[0],
        measurements={
            "tracking_error_m": round(
                float(result.trace.tracking_error_m.max()), 6
            ),
            "base_drift_m": round(result.trace.base_drift_m, 9),
            "duration_s": round(grounded.program.duration_s, 4),
            "duration_scale": round(duration_scale, 4),
        },
        inventory_sha256=inventory.sha256,
        base_tree_sha256=fingerprint,
    )
