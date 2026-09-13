"""A sealed physical bundle, opened for observation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.simulation.recording import PhysicsRecord

from ..contact.placement import PlacementGoal
from ..contracts import EffectorV1, RobotAssetManifestV1, SiteSemantic
from ..scenes.environment import SUPPORT_PREFIX, EnvironmentV1


OBJECT_GEOM = "scene_block_geom"
OBJECT_JOINT = "scene_block_free"


@dataclass
class Episode:
    """One recorded run: its model, its record and what the task said."""

    root: Path
    model: mujoco.MjModel
    record: PhysicsRecord
    manifest: RobotAssetManifestV1
    environment: EnvironmentV1
    goal: PlacementGoal
    task: dict
    outcome: dict
    execution: dict
    metadata: dict
    _data: mujoco.MjData = field(repr=False, default=None)

    @classmethod
    def load(cls, root: Path, *, default_goal: PlacementGoal | None = None) -> "Episode":
        """Open a sealed bundle. A composed or tree run's task carries no
        placement goal of its own (it composes against the registered G06
        fixture by hash); ``default_goal`` supplies that registered goal."""

        root = Path(root)
        manifest_json = json.loads((root / "manifest.json").read_bytes())
        model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
        record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
        if record.content_hash() != manifest_json["metadata"]["trace_sha256"]:
            raise ValueError(f"{root}: the trace does not match its manifest")
        task = json.loads((root / "task.json").read_bytes())
        env = EnvironmentV1.model_validate(task["environment"])
        if "placement_goal" in task:
            g = task["placement_goal"]
            goal = PlacementGoal(region_minimum_m=tuple(g["region_minimum_m"]), region_maximum_m=tuple(g["region_maximum_m"]), dwell_s=g["dwell_s"],
                                 maximum_linear_speed_mps=g["maximum_linear_speed_mps"], maximum_angular_speed_radps=g["maximum_angular_speed_radps"])
        elif default_goal is not None:
            goal = default_goal
        else:
            raise KeyError(f"{root}: the task carries no placement goal and none was supplied")
        robot = RobotAssetManifestV1.model_validate_json((root / "robot.json").read_bytes())
        episode = cls(root=root, model=model, record=record, manifest=robot, environment=env, goal=goal, task=task,
                      outcome=json.loads((root / "outcome.json").read_bytes()), execution=json.loads((root / "execution.json").read_bytes()),
                      metadata=manifest_json["metadata"])
        episode._data = mujoco.MjData(model)
        return episode

    # -- identity -----------------------------------------------------------
    @property
    def episode_id(self) -> str:
        return str(self.metadata["robot_id"])

    @property
    def zoo_id(self) -> str:
        return str(self.metadata.get("rig_id") or self.manifest.rig_id)

    @property
    def trace_sha256(self) -> str:
        return str(self.metadata["trace_sha256"])

    @property
    def times(self) -> np.ndarray:
        return self.record.arrays["time_s"]

    @property
    def start_s(self) -> float:
        return float(self.times[0])

    @property
    def end_s(self) -> float:
        return float(self.times[-1])

    @property
    def refusal(self) -> bool:
        return len(self.times) == 1

    @property
    def track(self) -> str:
        return str(self.task.get("track", "strict_fixed_world"))

    def phases(self) -> list[dict]:
        return list(self.execution.get("phases", []))

    # -- the body -------------------------------------------------------------
    @cached_property
    def effectors(self) -> tuple[EffectorV1, ...]:
        return tuple(self.manifest.morphology.grasping_effectors)

    def grasp_site(self, effector: EffectorV1) -> int:
        """The site the transfer drives: the effector's grasp point."""

        for semantic in (SiteSemantic.GRASP_POINT, SiteSemantic.GRASP_CENTER):
            for site in self.manifest.morphology.sites:
                if site.semantic is semantic and site.name.startswith(effector.chain_id):
                    identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site.name)
                    if identifier >= 0:
                        return identifier
        for name in effector.site_names:
            identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if identifier >= 0:
                return identifier
        raise KeyError(f"{effector.chain_id}: no site in the model")

    def member_geoms(self, effector: EffectorV1) -> dict[str, frozenset[int]]:
        out = {}
        for body in effector.member_bodies:
            identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if identifier < 0:
                out[body] = frozenset()
                continue
            start, count = int(self.model.body_geomadr[identifier]), int(self.model.body_geomnum[identifier])
            out[body] = frozenset(range(start, start + count))
        return out

    def robot_geoms(self) -> frozenset[int]:
        return frozenset(g for g in range(self.model.ngeom) if not self.is_world_geom(g))

    def is_world_geom(self, geom: int) -> bool:
        body = int(self.model.geom_bodyid[geom])
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        return body == 0 or name.startswith((SUPPORT_PREFIX, "scene_", "env_object_"))

    # -- the object and the world ----------------------------------------------
    @cached_property
    def object_geom(self) -> int:
        identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
        if identifier < 0:
            raise KeyError("the episode has no scene object")
        return identifier

    @cached_property
    def object_qpos(self) -> int:
        return int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT)])

    @cached_property
    def object_dof(self) -> int:
        return int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT)])

    @cached_property
    def object_half_extent_m(self) -> np.ndarray:
        return np.array(self.model.geom_size[self.object_geom], dtype=float)

    @cached_property
    def support_geoms(self) -> frozenset[int]:
        return frozenset(g for g in range(self.model.ngeom) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(SUPPORT_PREFIX))

    @cached_property
    def support_top_m(self) -> float:
        """The top of the support the object began on: the highest fixture top under it."""

        start = self.record.arrays["qpos"][0][self.object_qpos: self.object_qpos + 3]
        tops = [float(f.position_m[2] + f.size_m[2]) for f in self.environment.fixtures
                if abs(f.position_m[0] - start[0]) <= f.size_m[0] + 0.05 and abs(f.position_m[1] - start[1]) <= f.size_m[1] + 0.05]
        return max(tops) if tops else 0.0

    # -- the record -------------------------------------------------------------
    def index_at(self, time_s: float) -> int:
        """The last recorded sample at or before ``time_s``."""

        return int(max(0, np.searchsorted(self.times, time_s + 1e-9, side="right") - 1))

    def kinematics(self, index: int) -> mujoco.MjData:
        """Positions only, from the recorded configuration; never physics."""

        self._data.qpos[:] = self.record.arrays["qpos"][index]
        mujoco.mj_kinematics(self.model, self._data)
        return self._data

    def object_position(self, index: int) -> np.ndarray:
        return np.array(self.record.arrays["qpos"][index][self.object_qpos: self.object_qpos + 3], dtype=float)

    def object_velocity(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        dof = self.object_dof
        qvel = self.record.arrays["qvel"][index]
        return np.array(qvel[dof: dof + 3], dtype=float), np.array(qvel[dof + 3: dof + 6], dtype=float)

    def contacts(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The recorded contacts describing the interval that ended at sample
        ``index``: geom pairs, geometry rows (dist, pos, frame) and wrenches."""

        arrays = self.record.arrays
        lo, hi = int(arrays["contact_offsets"][index]), int(arrays["contact_offsets"][index + 1])
        return arrays["contact_geom"][lo:hi], arrays["contact_geometry"][lo:hi], arrays["contact_wrench"][lo:hi]

    def member_forces(self, effector: EffectorV1, index: int) -> dict[str, float]:
        """Normal force on each member of ``effector`` from the object, from the recorded contacts."""

        geoms = self.member_geoms(effector)
        forces = {body: 0.0 for body in geoms}
        pairs, _, wrenches = self.contacts(index)
        obj = self.object_geom
        for (a, b), wrench in zip(pairs, wrenches):
            if obj not in (int(a), int(b)):
                continue
            other = int(b) if int(a) == obj else int(a)
            for body, members in geoms.items():
                if other in members:
                    forces[body] += abs(float(wrench[0]))
                    break
        return forces

    def group_forces(self, effector: EffectorV1, index: int) -> list[float]:
        forces = self.member_forces(effector, index)
        return [max((forces.get(m, 0.0) for m in group), default=0.0) for group in effector.opposition_groups]

    def object_contacts(self, index: int) -> tuple[bool, bool, float]:
        """Whether the object touches a support, whether any robot geom touches
        it, and the largest robot normal force on it, at sample ``index``."""

        pairs, geometry, wrenches = self.contacts(index)
        obj = self.object_geom
        support = robot = False
        peak = 0.0
        for (a, b), row, wrench in zip(pairs, geometry, wrenches):
            if obj not in (int(a), int(b)):
                continue
            other = int(b) if int(a) == obj else int(a)
            touching = float(row[0]) <= 0.0
            if other in self.support_geoms:
                support |= touching
            elif not self.is_world_geom(other):
                robot |= touching or abs(float(wrench[0])) > 0.0
                peak = max(peak, abs(float(wrench[0])))
        return support, robot, peak

    def object_extent(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """The object's axis-aligned bounds at sample ``index`` from its recorded pose."""

        data = self.kinematics(index)
        centre = np.array(data.geom_xpos[self.object_geom], dtype=float)
        half = np.abs(np.array(data.geom_xmat[self.object_geom], dtype=float).reshape(3, 3)) @ self.object_half_extent_m
        return centre - half, centre + half
