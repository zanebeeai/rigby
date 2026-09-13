"""Compose two certified skills: check the boundary, repair what a path can
repair, verify again, and only then run the second.

The first skill runs and leaves a boundary state. The neutral checker holds
it against the second skill's initiation set. If it is compatible the second
runs. If every violation admits a path repair, the repair runs on physics --
a move of the offending joints inside their margins, a settle until the
body is still, a fresh observation -- and the boundary is measured and
checked again before the second skill may begin; a repair that does not
leave the boundary compatible within its bound is a typed failure. If any
violation admits no path -- the second skill expects to hold what is not
held, or to be free of what is held, or a resource is still owned -- the
composition is rejected before anything moves, and no smooth motion is
proposed in its place.

Everything is one physics record: the first skill, the repairs and the
second skill continue the same world on the same clock, and the record
replays. Over the whole of it the joint-limit and velocity gates the free
motion certification applies are evaluated again, so a second skill that
begins where the first left a joint past its limit cannot be called a
success by the second skill's own certificate alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np
from rigby_core.simulation.recording import PhysicsRecorder
from rigby_core.skills import BoundaryStateV1, BoundaryVerdictV1, ContactMode, InitiationSetV1, Repair, TransitionCostV1, check_boundary

from ..contact.transfer import PHASES, TransferStart, attempt_transfer
from ..contracts import EffectorV1, RobotAssetManifestV1
from ..gates.certify import GatePolicy
from ..grounding import ik
from ..grounding.workspace import WorkspaceFrame
from .boundary import arm_joint_names, initiation_for, measure_boundary
from .repair import Executed, joint_move, settle, track


@dataclass
class SkillOutcome:
    skill_id: str
    executed: bool
    certified: bool
    gate: str | None
    continuation: TransferStart | None
    phases: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    physics_s: float = 0.0
    interrupted: bool = False
    detail: str = ""
    measurements: dict = field(default_factory=dict)


@dataclass
class Skill:
    skill_id: str
    initiation: InitiationSetV1
    run: Callable[[TransferStart | None, PhysicsRecorder | None, Callable[[float], bool] | None], SkillOutcome]
    keeps_resources: tuple[str, ...] = ()
    """Resources the skill still owns when it ends (a skill that hands off
    its manipulator to nobody)."""


def transfer_skill(model, manifest: RobotAssetManifestV1, scene, effector: EffectorV1, frame: WorkspaceFrame, *, skill_id: str,
                   phase_range: tuple[str, str] = (PHASES[0], PHASES[-1]), keeps_resources: tuple[str, ...] = ()) -> Skill:
    """The G06 transfer, whole or a contiguous part of it, as a skill: an
    acquisition (approach through carry) begins free and ends holding; a
    placement (lower through dwell) begins holding and ends resting."""

    first = phase_range[0]
    begins_holding = PHASES.index(first) > PHASES.index("close")
    initiation = initiation_for(skill_id, ContactMode.HOLDING if begins_holding else ContactMode.FREE, effector, requires_object=begins_holding,
                                requires_resting=not begins_holding)

    def run(resume, recorder, should_stop):
        result = attempt_transfer(manifest, scene, effector, frame, recorder=recorder, resume=resume, should_stop=should_stop, phase_range=phase_range)
        return SkillOutcome(skill_id=skill_id, executed=result.executed, certified=result.certified, gate=result.failed_gate, continuation=result.continuation(),
                            phases=[p.name for p in result.phases], violations=[v.code for v in result.violations],
                            physics_s=(result.final_time_s - float(result.times_s[0])) if result.executed else 0.0, interrupted=result.interrupted,
                            detail=result.violations[0].detail[:200] if result.violations else "",
                            measurements={"lift_height_m": result.lift_height_m, "hold_s": result.hold_s, "placement_dwell_s": result.placement_dwell_s,
                                          "placement_success": result.placement_success, "max_penetration_m": result.max_penetration_m, "peak_force_n": result.peak_force_n,
                                          "path_seed": result.path_seed})
    return Skill(skill_id=skill_id, initiation=initiation, run=run, keeps_resources=keeps_resources)


def joint_move_skill(model, manifest: RobotAssetManifestV1, effector: EffectorV1, frame: WorkspaceFrame, guard: "ik.CollisionGuard | None", *, skill_id: str,
                     targets: dict[str, float], holding: bool = False, stop_fraction: float | None = None, speed_fraction: float = 0.35,
                     keeps_resources: tuple[str, ...] = ()) -> Skill:
    """A free-space motion as a skill: a quintic joint-space move to
    ``targets`` (by declared joint name), guarded. ``stop_fraction`` ends
    the move part way, still moving -- the way a skill interrupted mid-flight
    leaves a boundary with velocity in it."""

    arm = arm_joint_names(model, effector, frame)
    initiation = initiation_for(skill_id, ContactMode.HOLDING if holding else ContactMode.FREE, effector, requires_object=holding)

    def run(resume, recorder, should_stop):
        if resume is None:
            from ..contact.grasp import _scene_rest_qpos

            resume = TransferStart(qpos=np.asarray(_scene_rest_qpos(model, manifest), dtype=float), qvel=np.zeros(model.nv), time_s=0.0, state=None)
        stopper = should_stop
        if stop_fraction is not None:
            deadline: dict = {}

            def stopper(now: float) -> bool:  # noqa: F811
                if should_stop is not None and should_stop(now):
                    return True
                return "at" in deadline and now >= deadline["at"]

            executed = _joint_move_with_stop(model, manifest, effector, arm, resume, recorder, targets, guard, holding, speed_fraction, stop_fraction, deadline, stopper)
        else:
            executed = joint_move(model, manifest, effector, arm, resume, recorder, targets, guard, holding=holding, speed_fraction=speed_fraction)
        # A free motion certifies itself by the free-motion gates over its
        # own record: a move that pushed a joint past its limit is not a
        # certified skill, whatever boundary it left.
        own_gates = joint_limit_violations(model, manifest, arm, executed.qpos, executed.qvel, scope=skill_id) if executed.executed and len(executed.qpos) else []
        certified = executed.executed and executed.refusal is None and (not executed.interrupted or stop_fraction is not None) and not own_gates
        gate = executed.refusal or (own_gates[0]["code"] if own_gates else None)
        return SkillOutcome(skill_id=skill_id, executed=executed.executed, certified=certified, gate=gate, continuation=executed.start if executed.executed else None,
                            phases=["move"] if executed.executed else [], violations=[executed.refusal] if executed.refusal else [g["code"] for g in own_gates],
                            physics_s=executed.physics_s, interrupted=executed.interrupted, detail=executed.detail or (f"{own_gates[0]['subject']} {own_gates[0]['measured']:.3f} against {own_gates[0]['limit']:.3f}" if own_gates else ""),
                            measurements={"joint_travel_rad": executed.joint_travel_rad, "peak_speed_fraction": executed.peak_speed_fraction, "own_gate_violations": own_gates})
    return Skill(skill_id=skill_id, initiation=initiation, run=run, keeps_resources=keeps_resources)


def _joint_move_with_stop(model, manifest, effector, arm, resume, recorder, targets, guard, holding, speed_fraction, stop_fraction, deadline, stopper) -> Executed:
    """A joint move that stops at ``stop_fraction`` of its own duration."""

    from rigby_core.motion.timing import QuinticSegment

    from .repair import GUARD_SAMPLES, MIN_MOVE_S, reference_of

    by_joint = {dof.joint: dof for dof in manifest.dofs}
    arm_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm])
    start_arm = np.array(resume.qpos, dtype=float)[arm_adr]
    end_arm = start_arm.copy()
    for index, name in enumerate(arm):
        dof = by_joint[name]
        if dof.name in targets:
            end_arm[index] = float(targets[dof.name])
    if guard is not None:
        full = np.array(resume.qpos, dtype=float)
        for fraction in np.linspace(0.0, 1.0, GUARD_SAMPLES):
            sample = full.copy()
            sample[arm_adr] = start_arm + fraction * (end_arm - start_arm)
            inside = ik.penetrations(model, guard, sample)
            if inside:
                first, second, depth = inside[0]
                return Executed(executed=False, start=resume, physics_s=0.0, joint_travel_rad=0.0, peak_speed_fraction=0.0, refusal="self_collision_path",
                                detail=f"{first} inside {second} by {depth * 1000:.1f} mm at {fraction:.2f} of the move")
    limits = np.array([by_joint[n].velocity_limit for n in arm], dtype=float)
    duration = max(MIN_MOVE_S, float(np.max(np.abs(end_arm - start_arm) / (speed_fraction * limits))))
    deadline["at"] = resume.time_s + stop_fraction * duration
    segment = QuinticSegment(start_arm, end_arm, duration)
    return track(model, manifest, effector, arm, resume, recorder, reference_of(segment, duration), duration, holding=holding, should_stop=stopper)


@dataclass
class RepairRecord:
    kind: Repair
    executed: bool
    refusal: str | None
    physics_s: float
    joint_travel_rad: float
    peak_speed_fraction: float
    detail: str
    verdict_after: BoundaryVerdictV1 | None = None
    boundary_after: BoundaryStateV1 | None = None
    qpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    qvel: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)


@dataclass
class CompositionRecord:
    first: SkillOutcome
    boundary: BoundaryStateV1 | None
    contact: dict
    verdicts: list[BoundaryVerdictV1]
    repairs: list[RepairRecord]
    validated: bool
    rejected: bool
    rejection: str | None
    re_verified: bool
    second: SkillOutcome | None
    composed_success: bool
    gate_violations: list[dict]
    cost: TransitionCostV1
    final: TransferStart | None
    belief_age_s: float

    @property
    def compatible_before_second(self) -> bool:
        return bool(self.verdicts) and self.verdicts[-1].compatible

    def summary(self) -> dict:
        return {
            "first": {"skill": self.first.skill_id, "executed": self.first.executed, "certified": self.first.certified, "gate": self.first.gate, "phases": self.first.phases, "physics_s": self.first.physics_s},
            "boundary": None if self.boundary is None else {"contact_mode": self.boundary.contact_mode.value, "held": dict(self.boundary.held), "belief_age_s": self.boundary.belief_age_s,
                                                              "joints": [{"name": j.name, "position": j.position, "velocity": j.velocity, "minimum": j.minimum, "maximum": j.maximum} for j in self.boundary.joints]},
            "contact": self.contact,
            "verdicts": [{"compatible": v.compatible, "repair": v.repair.value, "violations": [{"code": x.code, "subject": x.subject, "measured": x.measured, "limit": x.limit, "repair": x.repair.value, "detail": x.detail} for x in v.violations]} for v in self.verdicts],
            "repairs": [{"kind": r.kind.value, "executed": r.executed, "refusal": r.refusal, "physics_s": r.physics_s, "joint_travel_rad": r.joint_travel_rad, "peak_speed_fraction": r.peak_speed_fraction, "detail": r.detail,
                         "compatible_after": None if r.verdict_after is None else r.verdict_after.compatible} for r in self.repairs],
            "validated": self.validated, "rejected": self.rejected, "rejection": self.rejection, "re_verified": self.re_verified,
            "second": None if self.second is None else {"skill": self.second.skill_id, "executed": self.second.executed, "certified": self.second.certified, "gate": self.second.gate, "phases": self.second.phases, "physics_s": self.second.physics_s, "measurements": self.second.measurements},
            "composed_success": self.composed_success, "gate_violations": self.gate_violations,
            "cost": {"physics_s": self.cost.physics_s, "joint_travel_rad": self.cost.joint_travel_rad, "peak_speed_fraction": self.cost.peak_speed_fraction, "repairs": [r.value for r in self.cost.repairs]},
        }


def joint_limit_violations(model, manifest: RobotAssetManifestV1, arm_joints: tuple[str, ...], qpos: np.ndarray, qvel: np.ndarray | None = None,
                           policy: GatePolicy | None = None, *, scope: str = "record") -> list[dict]:
    """The free-motion certification's joint position gate over ``qpos`` and,
    when ``qvel`` is given, its velocity gate; ``scope`` names what was gated."""

    policy = policy or GatePolicy()
    by_joint = {dof.joint: dof for dof in manifest.dofs}
    found = []
    for name in arm_joints:
        dof = by_joint[name]
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        values = qpos[:, int(model.jnt_qposadr[joint])]
        low, high = float(values.min()), float(values.max())
        if low < dof.minimum - policy.limit_margin:
            found.append({"code": "joint_position_limit", "subject": dof.name, "measured": low, "limit": dof.minimum, "scope": scope})
        if high > dof.maximum + policy.limit_margin:
            found.append({"code": "joint_position_limit", "subject": dof.name, "measured": high, "limit": dof.maximum, "scope": scope})
        if qvel is not None and len(qvel):
            peak = float(np.abs(qvel[:, int(model.jnt_dofadr[joint])]).max())
            if peak > dof.velocity_limit * (1.0 + policy.effort_margin_fraction):
                found.append({"code": "velocity_limit", "subject": dof.name, "measured": peak, "limit": dof.velocity_limit, "scope": scope})
    return found


def compose(model, manifest: RobotAssetManifestV1, effector: EffectorV1, frame: WorkspaceFrame, first: Skill, second: Skill, recorder: PhysicsRecorder | None, *,
            guard: "ik.CollisionGuard | None", validate: bool = True, max_repairs: int = 2, belief_age_s: float = 0.0, first_resume: TransferStart | None = None,
            should_stop: Callable[[float], bool] | None = None) -> CompositionRecord:
    """Run ``first``, check its boundary against ``second``, repair and
    re-verify within ``max_repairs``, then run ``second``; or, with
    ``validate`` off, run ``second`` straight from wherever ``first`` ended,
    which is what the check is there to prevent."""

    arm = arm_joint_names(model, effector, frame)
    outcome = first.run(first_resume, recorder, should_stop)
    empty_cost = TransitionCostV1(physics_s=0.0, joint_travel_rad=0.0, peak_speed_fraction=0.0)
    if not outcome.executed or outcome.continuation is None:
        return CompositionRecord(first=outcome, boundary=None, contact={}, verdicts=[], repairs=[], validated=validate, rejected=True,
                                 rejection=f"first_skill_{outcome.gate or 'not_executed'}", re_verified=False, second=None, composed_success=False,
                                 gate_violations=[], cost=empty_cost, final=None, belief_age_s=belief_age_s)
    current = outcome.continuation
    age = float(belief_age_s)
    boundary, contact = measure_boundary(model, manifest, effector, frame, current, belief_age_s=age, owned=first.keeps_resources)
    verdicts = [check_boundary(boundary, second.initiation)]
    repairs: list[RepairRecord] = []
    rejected, rejection = False, None
    holding = boundary.contact_mode is ContactMode.HOLDING
    if validate:
        while not verdicts[-1].compatible and len(repairs) < max_repairs:
            verdict = verdicts[-1]
            if verdict.rejected:
                rejected = True
                rejection = next(v.code for v in verdict.violations if not v.repairable_by_path)
                break
            if verdict.repair is Repair.JOINT_MOVE:
                # The verdict names the nearest admissible position; the move
                # aims one more margin inside it, because a tracking error of
                # a few hundredths of a radian on arrival would leave the joint
                # inside the margin and the boundary still incompatible.
                targets = {}
                for name, value in verdict.joint_targets.items():
                    joint = boundary.joint(name)
                    margin = second.initiation.limit_margin_fraction * joint.range
                    targets[name] = value - margin if value > 0.5 * (joint.minimum + joint.maximum) else value + margin
                executed = joint_move(model, manifest, effector, arm, current, recorder, targets, guard, holding=holding)
            elif verdict.repair is Repair.SETTLE:
                executed = settle(model, manifest, effector, arm, current, recorder, speed_fraction=second.initiation.speed_fraction, holding=holding)
            elif verdict.repair is Repair.OBSERVE:
                age = 0.0
                executed = Executed(executed=True, start=current, physics_s=0.0, joint_travel_rad=0.0, peak_speed_fraction=0.0, detail="belief refreshed from the scene")
            else:  # pragma: no cover - rejected above
                rejected, rejection = True, verdict.repair.value
                break
            record = RepairRecord(kind=verdict.repair, executed=executed.executed, refusal=executed.refusal, physics_s=executed.physics_s,
                                  joint_travel_rad=executed.joint_travel_rad, peak_speed_fraction=executed.peak_speed_fraction, detail=executed.detail,
                                  qpos=executed.qpos, qvel=executed.qvel)
            if not executed.executed:
                repairs.append(record)
                rejected, rejection = True, f"repair_refused:{executed.refusal}"
                break
            current = executed.start
            boundary, contact = measure_boundary(model, manifest, effector, frame, current, belief_age_s=age, owned=first.keeps_resources)
            verdict_after = check_boundary(boundary, second.initiation)
            record.verdict_after, record.boundary_after = verdict_after, boundary
            repairs.append(record)
            verdicts.append(verdict_after)
            holding = boundary.contact_mode is ContactMode.HOLDING
        if not rejected and not verdicts[-1].compatible:
            rejected = True
            rejection = "repair_budget_exhausted:" + ",".join(sorted({v.code for v in verdicts[-1].violations}))
    re_verified = bool(repairs) and verdicts[-1].compatible
    second_outcome = None
    second_rows_from = None
    if validate and rejected:
        pass
    else:
        second_rows_from = max(0, len(recorder.rows["qpos"]) - 1) if recorder is not None else None
        second_outcome = second.run(current, recorder, should_stop)
        if second_outcome.continuation is not None:
            current = second_outcome.continuation
    # The position gate over the whole record: a second skill that begins
    # where the first left a joint past its limit is not a success. The
    # velocity gate over the transitions only: the skills' own motion is
    # certified by their own gates, the repair by this one.
    # Each transition segment is gated on its own motion, its first sample
    # excluded: that sample is the boundary it was asked to repair. The
    # second skill's segment is gated on position over its whole extent.
    gate_violations: list[dict] = []
    for index, repair in enumerate(repairs):
        if len(repair.qvel) > 1:
            gate_violations += joint_limit_violations(model, manifest, arm, repair.qpos[1:], repair.qvel[1:], scope=f"repair:{index}:{repair.kind.value}")
    if second_outcome is not None and recorder is not None and second_rows_from is not None and len(recorder.rows["qpos"]) > second_rows_from:
        gate_violations += joint_limit_violations(model, manifest, arm, np.asarray(recorder.rows["qpos"][second_rows_from:]), None, scope="second")
    cost = TransitionCostV1(physics_s=float(sum(r.physics_s for r in repairs)), joint_travel_rad=float(sum(r.joint_travel_rad for r in repairs)),
                            peak_speed_fraction=float(max((r.peak_speed_fraction for r in repairs), default=0.0)), repairs=tuple(r.kind for r in repairs))
    composed_success = bool(outcome.certified and second_outcome is not None and second_outcome.certified and not gate_violations and (not validate or verdicts[-1].compatible))
    return CompositionRecord(first=outcome, boundary=boundary, contact=contact, verdicts=verdicts, repairs=repairs, validated=validate, rejected=rejected,
                             rejection=rejection, re_verified=re_verified, second=second_outcome, composed_success=composed_success,
                             gate_violations=gate_violations, cost=cost, final=current, belief_age_s=age)
