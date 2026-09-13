"""The skill store bound to bodies: contexts from sessions, validation sets, reuse, restart.

The neutral store (``rigby_core.skills.store``) compares contexts and keeps
certificates; this module says what a context *is* for a TransferObject
session on physics, and runs the episodes a certificate rests on. Six
facets are computed from the session as it stands, never from what a caller
says it intends: the body from the ingested manifest; the controller from
the configuration every leaf and hold actually runs with, closure included;
the sensors from the configuration the live sensing was built from and the
policy the conditionals were bound with; the geometry from the object's
size and the fixtures' sizes and heights (positions are free, within the
ranges); the friction assumption from the range the certificate claims; the
evidence schema from the protocol and record versions the bundles are
sealed under. Beside the facets go the values a run brings -- the object's
friction, mass and distance from the mount, the destination's distance --
which a certificate's ranges are checked against.

A validation set is drawn the way the G06 roster drew its trials, from a
generator seeded apart from every development set, over seeds no
development set used. A restart opens a fresh session on the checkpoint's
physical state and learns what it can believe from the sensors alone:
whether the closure is engaged (from the contact the object exerts on the
members, as the contact sensor reads it), whether the object is held, placed
or merely visible (from the same conditionals every verification uses),
and where it is. Nothing from the checkpoint's belief is copied.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rigby_core.hashing import content_hash
from rigby_core.skills import (
    Belief,
    CheckpointV1,
    CheckpointingRuntime,
    ContextDimension,
    ContextFacetV1,
    ExecutionContextV1,
    Interrupt,
    RangeV1,
    SkillLibraryV1,
    ValidationSetV1,
    execute,
)
from rigby_core.simulation.recording import STATE_SPEC

from ..contact.closure import ClosureConfig
from ..contact.placement import PlacementGoal
from ..contact.transfer import TransferStart
from ..gates.control import ControllerConfig
from ..scenes.environment import EnvironmentV1, FixtureV1, SceneObjectV1
from ..sensing import configuration, decide_live
from .transfer_object import STILL_WINDOW_S, TransferObjectRuntime, TransferObjectSession, predicates_for_transfer
from .transfer_runtime import SimulationClock


EVIDENCE_SCHEMA = {"episode_protocol": "rigby.transfer-object-episode/1", "bundle_layout": "rigby.evidence-bundle/G01", "execution_record": "1.0", "certificate": "1.0"}
"""What the sealed evidence is written as; a certificate is conditioned on it."""

CLAIMED_RANGES = (
    RangeV1(quantity="object_friction", low=1.0, high=1.8, units="coefficient"),
    RangeV1(quantity="object_mass_kg", low=0.004, high=0.016, units="kg"),
    RangeV1(quantity="object_distance_m", low=0.55, high=0.80, units="m"),
    RangeV1(quantity="destination_distance_m", low=0.55, high=0.80, units="m"),
)
"""What a TransferObject certificate claims to cover. The validation set
samples the object's friction and mass over the G06 draw family and the
fixed fixtures; the distances are asserted from the fixed world's geometry
with room for the reuse layouts, and the reuse runs are what test them."""

FRICTION_ASSUMPTION = {"object_friction_range": [1.0, 1.8], "basis": "the G06 draw family, 0.8 to 1.2 of a 1.4 coefficient, with margin"}
GOAL_MARGIN_M = 0.06
GOAL_HEIGHT_M = 0.09
DEVELOPMENT_SETS = ("g10-transfer-v1:nominal-seeds-0-99", "g06-transfer-v1:roster-seeds-0-99")
GRIP_ENGAGED_N = 0.1
"""The contact the closure has to exert on the object for a restart to call it engaged: the G09 policy's contact threshold."""


# -- contexts --------------------------------------------------------------------------


def horizontal_distance(environment: EnvironmentV1, position_m) -> float:
    mount = np.asarray(environment.robot_mount_m, dtype=float)[:2]
    return float(np.linalg.norm(np.asarray(position_m, dtype=float)[:2] - mount))


def fixture_named(environment: EnvironmentV1, name: str) -> FixtureV1:
    return next(f for f in environment.fixtures if f.name == name)


def goal_for(environment: EnvironmentV1, *, destination: str = "platform") -> PlacementGoal:
    """The placement region the registered G06 goal draws around its platform,
    drawn around wherever this environment's platform stands."""

    platform = fixture_named(environment, destination)
    top = float(platform.position_m[2] + platform.size_m[2])
    x, y = float(platform.position_m[0]), float(platform.position_m[1])
    return PlacementGoal(region_minimum_m=(x - GOAL_MARGIN_M, y - GOAL_MARGIN_M, top - 0.001), region_maximum_m=(x + GOAL_MARGIN_M, y + GOAL_MARGIN_M, top + GOAL_HEIGHT_M),
                         dwell_s=2.0, maximum_linear_speed_mps=0.01, maximum_angular_speed_radps=0.1)


def context_of(session: TransferObjectSession, *, friction_assumption: dict | None = None, evidence_schema: dict | None = None, ranges: tuple[RangeV1, ...] = CLAIMED_RANGES) -> ExecutionContextV1:
    """The six facets from the session as it stands, and the values this run brings."""

    manifest = session.robot.manifest
    environment = session.environment
    cube = environment.objects[0]
    sensors = configuration(session.configuration_id, tuple(session.effectors.values()))
    facets = (
        ContextFacetV1.of(ContextDimension.BODY, manifest.rig_id, {"rig_id": manifest.rig_id, "manifest_sha256": manifest.content_hash(), "effector": session.effector.chain_id}),
        ContextFacetV1.of(ContextDimension.CONTROLLER, f"computed-torque {session.controller_config.natural_frequency_hz:g} Hz zeta {session.controller_config.damping_ratio:g}",
                          {"arm": "rigby_general.gates.control.ComputedTorqueController", "arm_config": asdict(session.controller_config),
                           "closure": "rigby_general.contact.closure.ClosureController", "closure_config": asdict(ClosureConfig())}),
        ContextFacetV1.of(ContextDimension.SENSORS, session.configuration_id, {"configuration_id": session.configuration_id, "configuration_sha256": sensors.content_hash(), "policy_sha256": session.policy_sha256}),
        ContextFacetV1.of(ContextDimension.GEOMETRY, f"{cube.name} {cube.size_m[0] * 2000:.0f} mm on {len(environment.fixtures)} fixtures",
                          {"object_size_m": [float(v) for v in cube.size_m], "fixtures": [{"name": f.name, "size_m": [float(v) for v in f.size_m], "top_m": round(float(f.position_m[2] + f.size_m[2]), 6)} for f in environment.fixtures],
                           "robot_mount_m": [float(v) for v in environment.robot_mount_m]}),
        ContextFacetV1.of(ContextDimension.FRICTION, "friction " + "-".join(f"{v:g}" for v in (friction_assumption or FRICTION_ASSUMPTION)["object_friction_range"]), dict(friction_assumption or FRICTION_ASSUMPTION)),
        ContextFacetV1.of(ContextDimension.EVIDENCE_SCHEMA, (evidence_schema or EVIDENCE_SCHEMA)["episode_protocol"], dict(evidence_schema or EVIDENCE_SCHEMA)),
    )
    values = {
        "object_friction": float(cube.friction), "object_mass_kg": float(cube.mass_kg),
        "object_distance_m": horizontal_distance(environment, cube.position_m),
        "destination_distance_m": horizontal_distance(environment, fixture_named(environment, "platform").position_m),
    }
    return ExecutionContextV1(facets=facets, ranges=ranges, values=values)


# -- validation sets and layouts -------------------------------------------------------------------


def draw_validation_set(body: str, *, set_id: str, seeds: tuple[int, ...], rng_seed: int, threshold: int, independent_of: tuple[str, ...] = DEVELOPMENT_SETS,
                        translation_half_width_m=(0.02, 0.02, 0.0), mass_multiplier_range=(0.8, 1.2), friction_multiplier_range=(0.8, 1.2)) -> tuple[ValidationSetV1, list[dict]]:
    """The G06 draw rule over new seeds from a generator seeded apart."""

    rng = np.random.default_rng(rng_seed)
    draws = []
    for seed in seeds:
        translation = [float(rng.uniform(-w, w)) if w > 0 else 0.0 for w in translation_half_width_m]
        draws.append({"seed": int(seed), "translation_m": translation, "mass_multiplier": float(rng.uniform(*mass_multiplier_range)), "friction_multiplier": float(rng.uniform(*friction_multiplier_range))})
    episodes = tuple(f"{body}-{set_id}-{seed:04d}" for seed in seeds)
    validation = ValidationSetV1(set_id=set_id, body=body, episodes=episodes, draws_sha256=content_hash(draws), independent_of=independent_of, threshold=threshold)
    return validation, draws


def perturbed(environment: EnvironmentV1, draw: dict) -> EnvironmentV1:
    cube = environment.objects[0]
    moved = SceneObjectV1(name=cube.name, size_m=cube.size_m, mass_kg=round(cube.mass_kg * draw["mass_multiplier"], 9),
                          position_m=(cube.position_m[0] + draw["translation_m"][0], cube.position_m[1] + draw["translation_m"][1], cube.position_m[2] + draw["translation_m"][2]),
                          friction=round(cube.friction * draw["friction_multiplier"], 6), rgba=cube.rgba)
    return environment.model_copy(update={"objects": (moved,)})


def with_fixture_moved(environment: EnvironmentV1, name: str, *, dx: float = 0.0, dy: float = 0.0) -> EnvironmentV1:
    fixtures = []
    for fixture in environment.fixtures:
        if fixture.name == name:
            fixtures.append(fixture.model_copy(update={"position_m": (fixture.position_m[0] + dx, fixture.position_m[1] + dy, fixture.position_m[2])}))
        else:
            fixtures.append(fixture)
    objects = environment.objects
    if name == "bench":
        cube = environment.objects[0]
        objects = (cube.model_copy(update={"position_m": (cube.position_m[0] + dx, cube.position_m[1] + dy, cube.position_m[2])}),)
    return environment.model_copy(update={"fixtures": tuple(fixtures), "objects": objects})


def layouts(environment: EnvironmentV1) -> dict[str, EnvironmentV1]:
    """Three object and layout instances no certificate was validated on,
    each inside the claimed ranges: the destination moved away, the two
    fixtures mirrored about the mount's forward axis, and a heavier, slicker
    cube on a bench moved nearer."""

    bench, platform = fixture_named(environment, "bench"), fixture_named(environment, "platform")
    far = with_fixture_moved(environment, "platform", dy=0.06)
    far = far.model_copy(update={"environment_id": environment.environment_id + "+platform_far", "description": environment.description + " The platform stands six centimetres further from the mount."})
    mid = 0.5 * (bench.position_m[0] + platform.position_m[0])
    mirrored = with_fixture_moved(with_fixture_moved(environment, "bench", dx=2 * (mid - bench.position_m[0])), "platform", dx=2 * (mid - platform.position_m[0]))
    mirrored = mirrored.model_copy(update={"environment_id": environment.environment_id + "+mirrored", "description": environment.description + " The bench and the platform have swapped sides."})
    near = with_fixture_moved(with_fixture_moved(environment, "bench", dy=-0.05), "platform", dx=0.04)
    cube = near.objects[0]
    near = near.model_copy(update={"environment_id": environment.environment_id + "+near_heavy", "description": environment.description + " The bench is five centimetres nearer, the platform four to the right; the cube is a third heavier and a tenth slicker.",
                                   "objects": (cube.model_copy(update={"mass_kg": round(cube.mass_kg * 1.3, 9), "friction": round(cube.friction * 0.9, 6)}),)})
    return {"platform_far": far, "mirrored": mirrored, "near_heavy": near}


# -- running one episode ---------------------------------------------------------------------------


@dataclass
class EpisodeRun:
    session: TransferObjectSession
    runtime: Any
    tree: Any
    record: Any
    wall_s: float
    checkpoint: CheckpointV1 | None = None
    reconstruction: dict[str, Any] | None = None


def world_state_of(session: TransferObjectSession) -> dict[str, Any]:
    """The full integration state and the clock, as a leaf would leave them."""

    if session.state is None:
        return {"time_s": float(session.time_s), "qpos": None, "qvel": None, "state": None}
    return {"time_s": float(session.state.time_s), "qpos": [float(v) for v in session.state.qpos], "qvel": [float(v) for v in session.state.qvel],
            "state": None if session.state.state is None else [float(v) for v in session.state.state]}


def run_tree(session: TransferObjectSession, library: SkillLibraryV1, root: str, arguments: dict[str, str], *, checkpoint_after: int | None = None, belief: Belief | None = None) -> EpisodeRun:
    tree = library.expand(root, arguments)
    interrupt = Interrupt()
    inner = TransferObjectRuntime(session, interrupt)
    runtime: Any = inner
    if checkpoint_after is not None:
        runtime = CheckpointingRuntime(inner, interrupt, checkpoint_after, world_state=lambda: world_state_of(session), tree_sha256=tree.content_hash(), library_sha256=library.content_hash())
    started = time.perf_counter()
    record = execute(tree, library, runtime, predicates_for_transfer(), clock=SimulationClock(session), interrupt=interrupt, belief=belief or Belief())
    return EpisodeRun(session=session, runtime=inner, tree=tree, record=record, wall_s=time.perf_counter() - started, checkpoint=getattr(runtime, "checkpoint", None))


# -- restart -----------------------------------------------------------------------------------------


def reopen(checkpoint: CheckpointV1, zoo_id: str, source: Path, environment: EnvironmentV1, goal: PlacementGoal, policy: dict, *, configuration_name: str, controller_config: ControllerConfig | None = None,
           seed_label: str = "") -> TransferObjectSession:
    """A fresh session -- new sensing streams, new recorder, empty results --
    standing on the checkpoint's physical state at the checkpoint's clock."""

    session = TransferObjectSession.open(zoo_id, source, environment, goal, policy, configuration_name=configuration_name, seed_label=seed_label, controller_config=controller_config)
    world = checkpoint.world_state
    if world.get("qpos") is not None:
        session.adopt(TransferStart(qpos=np.asarray(world["qpos"], dtype=float), qvel=np.asarray(world["qvel"], dtype=float), time_s=float(world["time_s"]),
                                    state=None if world.get("state") is None else np.asarray(world["state"], dtype=float)))
    return session


def reconstruct(session: TransferObjectSession, *, arguments: dict[str, str]) -> dict[str, Any]:
    """What a restarted executor may believe, from the sensors alone.

    First the closure: the contact the object exerts on the members, read as
    the contact sensor reads it on the restored state, says whether the
    gripper is engaged; the hold keeps whatever it finds. Then the arm holds
    still for the observation window while the declared sensors sample, and
    the conditionals decide held, placed and reachable; the object's position
    is the mean of the frames that showed it still. The facts returned are
    exactly those the tree's predicates read."""

    model = session.model
    data = mujoco.MjData(model)
    if session.state is not None and session.state.state is not None:
        mujoco.mj_setState(model, data, np.asarray(session.state.state, dtype=float), STATE_SPEC)
    else:
        data.qpos[:] = session.current_qpos()
    mujoco.mj_forward(model, data)
    grip_n = session.sensing.grip_force_now(data, session.effector.chain_id)
    engaged = grip_n >= GRIP_ENGAGED_N
    spent = session.hold_still(STILL_WINDOW_S + 0.1, holding=engaged)
    held = session.decide("held")
    placed = session.decide("stably_placed")
    reach = session.decide("reachable")
    position = session.sensing.mean_object_position(session.time_s, STILL_WINDOW_S)
    drift = session.sensing.object_drift_mps(session.time_s, STILL_WINDOW_S)
    facts: dict[str, Any] = {}
    obj, dest = arguments["object"], arguments["destination"]
    if held.decision.value == "pass":
        facts[f"held:{obj}"] = True
    if placed.decision.value == "pass":
        facts[f"placed:{obj}:{dest}"] = True
        facts[f"held:{obj}"] = False
    if position is not None and (drift is None or drift <= 0.02):
        facts[f"known:{obj}"] = True
        facts[f"pose:{obj}"] = [float(v) for v in position]
        facts[f"reach:{obj}"] = reach.decision.value == "pass"
        facts[f"reach_verdict:{obj}"] = reach.decision.value
    trail = {"grip_force_n": round(grip_n, 4), "closure_engaged": engaged, "observation_s": spent,
             "held": {"decision": held.decision.value, "reason": held.reason, "sensors": list(held.sensors_used)},
             "stably_placed": {"decision": placed.decision.value, "reason": placed.reason, "sensors": list(placed.sensors_used)},
             "reachable": {"decision": reach.decision.value, "reason": reach.reason, "sensors": list(reach.sensors_used)},
             "object_position_m": None if position is None else [round(float(v), 4) for v in position], "object_drift_mps": None if drift is None else round(float(drift), 4),
             "facts": dict(facts)}
    session.events.append({"time_s": session.time_s, "event": "reconstruction", "detail": trail})
    return {"facts": facts, "trail": trail}


def restart(checkpoint: CheckpointV1, library: SkillLibraryV1, root: str, arguments: dict[str, str], zoo_id: str, source: Path, environment: EnvironmentV1, goal: PlacementGoal, policy: dict, *,
            configuration_name: str, controller_config: ControllerConfig | None = None, seed_label: str = "") -> EpisodeRun:
    """Reopen on the checkpoint's world, reconstruct, run the tree from its root."""

    session = reopen(checkpoint, zoo_id, source, environment, goal, policy, configuration_name=configuration_name, controller_config=controller_config, seed_label=seed_label)
    reconstruction = reconstruct(session, arguments=arguments)
    run = run_tree(session, library, root, arguments, belief=Belief(dict(reconstruction["facts"])))
    run.reconstruction = reconstruction
    return run


# -- helpers for the campaign ---------------------------------------------------------------------------


def rows_digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()


__all__ = [
    "CLAIMED_RANGES", "DEVELOPMENT_SETS", "EVIDENCE_SCHEMA", "FRICTION_ASSUMPTION", "GRIP_ENGAGED_N", "EpisodeRun", "context_of", "draw_validation_set", "goal_for",
    "horizontal_distance", "layouts", "perturbed", "reconstruct", "reopen", "restart", "rows_digest", "run_tree", "with_fixture_moved", "world_state_of",
]
