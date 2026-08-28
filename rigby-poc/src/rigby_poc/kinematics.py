from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .models import BonePose, Quat

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _glb_document(path: Path) -> tuple[dict, str]:
    """The parsed JSON chunk, and the sha256 of the bytes it was parsed from.

    The digest is taken here, of the exact bytes read, rather than by re-reading
    the file later. Re-reading would return the digest of whatever is on disk
    *now* while the caller holds transforms parsed from what was there *then* --
    which agrees with the declared hash exactly when it should not. Plan 04
    §6.1d.
    """

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    magic, version, _ = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("rig asset is not a glTF 2.0 binary")
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            return json.loads(payload.rstrip(b" \x00")), digest
        offset += 8 + length
    raise ValueError("rig asset does not contain a JSON chunk")


def _quat_matrix(q) -> np.ndarray:
    """A quaternion's rotation matrix, in xyzw order.

    The same thing ``Rotation.from_quat(q).as_matrix()`` returns, without
    constructing a scipy object to get it. That construction was 24 seconds of
    a two-minute profile across 547,000 calls: the rig is evaluated for every
    candidate the controller scores, and building an object per bone per
    candidate is most of what evaluating it costs.
    """
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    return np.array([
        [1.0 - (y * ys + z * zs), x * ys - w * zs, x * zs + w * ys],
        [x * ys + w * zs, 1.0 - (x * xs + z * zs), y * zs - w * xs],
        [x * zs - w * ys, y * zs + w * xs, 1.0 - (x * xs + y * ys)],
    ], dtype=float)


@dataclass(frozen=True)
class _NodeRest:
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        # The rest pose never changes, so the parts of the transform that do not
        # depend on the delta are built once here rather than three million
        # times during a run.
        object.__setattr__(self, "_scaled", self.rotation * self.scale)
        base = np.eye(4, dtype=float)
        base[:3, :3] = self._scaled
        base[:3, 3] = self.translation
        object.__setattr__(self, "_base", base)

    def matrix(self, delta_rotation: np.ndarray | None = None, delta_translation: np.ndarray | None = None) -> np.ndarray:
        if delta_rotation is None and delta_translation is None:
            # Most nodes in any given frame are unposed -- a probe changes a
            # finger, not the skeleton -- and for those the answer is a constant.
            # Returned rather than rebuilt; every caller composes it into a new
            # array and none writes through it.
            return self._base
        matrix = np.empty((4, 4), dtype=float)
        matrix[3, :3] = 0.0
        matrix[3, 3] = 1.0
        matrix[:3, :3] = (
            self._scaled if delta_rotation is None
            else (self.rotation @ delta_rotation) * self.scale
        )
        matrix[:3, 3] = (
            self.translation if delta_translation is None
            else self.translation + delta_translation
        )
        return matrix


class RigKinematics:
    """Evaluate Rigby's exact source-rig hierarchy for deterministic gates."""

    def __init__(self, asset_path: Path, profile_path: Path) -> None:
        document, self.asset_sha256 = _glb_document(asset_path)
        """sha256 of the GLB bytes this instance was parsed from.

        Load-bearing for provenance, not diagnostics. Every anatomical frame and
        every range-of-motion verdict derives from these transforms, and the
        evidence a grader sees comes from the browser's independent parse of the
        same file. Comparing this against ``render_provenance.asset_sha256`` is
        parse against parse; comparing either against the *declared* hash is not,
        and is green precisely in the case that matters.
        """

        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.nodes = document["nodes"]
        self.parents = {
            child: parent
            for parent, node in enumerate(self.nodes)
            for child in node.get("children", [])
        }
        node_by_name = {
            node.get("name"): index
            for index, node in enumerate(self.nodes)
            if node.get("name")
        }
        self.node_by_name = node_by_name
        self.node_by_canonical = {
            canonical: node_by_name[source]
            for canonical, source in profile["bone_map"].items()
            if source in node_by_name
        }
        self.canonical_by_node = {
            node_index: canonical
            for canonical, node_index in self.node_by_canonical.items()
        }
        self.rest = [self._node_rest(node) for node in self.nodes]
        self.rest_world = self._rest_world_matrices()

    def _rest_world_matrices(self) -> list[np.ndarray]:
        local = [rest.matrix() for rest in self.rest]
        world: list[np.ndarray | None] = [None] * len(local)

        def resolve(index: int) -> np.ndarray:
            cached = world[index]
            if cached is not None:
                return cached
            parent = self.parents.get(index)
            value = local[index] if parent is None else resolve(parent) @ local[index]
            world[index] = value
            return value

        return [resolve(index) for index in range(len(local))]

    @staticmethod
    def _node_rest(node: dict) -> _NodeRest:
        if "matrix" in node:
            matrix = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
            scale = np.linalg.norm(matrix[:3, :3], axis=0)
            rotation = matrix[:3, :3] / np.maximum(scale, 1e-12)
            return _NodeRest(matrix[:3, 3], rotation, scale)
        return _NodeRest(
            np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=float),
            _quat_matrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0])),
            np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=float),
        )

    def world_matrices(self, bones: Mapping[str, BonePose]) -> list[np.ndarray]:
        local: list[np.ndarray] = []
        for node_index, rest in enumerate(self.rest):
            canonical = self.canonical_by_node.get(node_index)
            pose = bones.get(canonical) if canonical else None
            delta_rotation = (
                _quat_matrix(pose.rotation.as_list()) if pose is not None else None
            )
            delta_translation = None
            if pose is not None and pose.position is not None:
                value = np.asarray(pose.position.as_list(), dtype=float)
                # The pelvis is a child of the source rig's -90-degree X root.
                # Convert Rigby's Y-up application delta to that local frame.
                delta_translation = (
                    np.asarray([value[0], -value[2], value[1]], dtype=float)
                    if canonical == "hips"
                    else value
                )
            local.append(rest.matrix(delta_rotation, delta_translation))

        world: list[np.ndarray | None] = [None] * len(local)

        def resolve(index: int) -> np.ndarray:
            cached = world[index]
            if cached is not None:
                return cached
            parent = self.parents.get(index)
            value = local[index] if parent is None else resolve(parent) @ local[index]
            world[index] = value
            return value

        return [resolve(index) for index in range(len(local))]

    def canonical_positions(self, bones: Mapping[str, BonePose]) -> dict[str, np.ndarray]:
        world = self.world_matrices(bones)
        return {
            canonical: world[node_index][:3, 3].copy()
            for canonical, node_index in self.node_by_canonical.items()
        }

    def fingertip_positions(
        self,
        bones: Mapping[str, BonePose],
        hand: str,
    ) -> dict[str, np.ndarray]:
        """Return exact source-rig leaf pivots for all five fingertips."""

        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        suffix = "l" if hand == "left" else "r"
        source_stems = {
            "thumb": "thumb_04_leaf",
            "index": "index_04_leaf",
            "middle": "middle_04_leaf",
            "ring": "ring_04_leaf",
            "little": "pinky_04_leaf",
        }
        world = self.world_matrices(bones)
        return {
            digit: world[self.node_by_name[f"{stem}_{suffix}"]][:3, 3].copy()
            for digit, stem in source_stems.items()
        }

    def canonical_world_rotation(self, bones: Mapping[str, BonePose], canonical: str) -> np.ndarray:
        return self.world_matrices(bones)[self.node_by_canonical[canonical]][:3, :3].copy()

    def leg_reach(self, side: str) -> float:
        lower_index = self.node_by_canonical[f"{side}LowerLeg"]
        foot_index = self.node_by_canonical[f"{side}Foot"]
        return float(
            np.linalg.norm(self.rest[lower_index].translation)
            + np.linalg.norm(self.rest[foot_index].translation)
        )

    def world_delta_quat(self, canonical: str, world_delta: np.ndarray) -> Quat:
        """Convert an application-world rotation into a source local delta."""

        node_index = self.node_by_canonical[canonical]
        rest_world_rotation = self.rest_world[node_index][:3, :3]
        local_delta = rest_world_rotation.T @ world_delta @ rest_world_rotation
        return self._quat(local_delta)

    @staticmethod
    def _orientation_along(segment: np.ndarray, lateral_hint: np.ndarray) -> np.ndarray:
        local_y = segment / max(float(np.linalg.norm(segment)), 1e-12)
        lateral = lateral_hint / max(float(np.linalg.norm(lateral_hint)), 1e-12)
        local_x = lateral - local_y * float(np.dot(lateral, local_y))
        if float(np.linalg.norm(local_x)) < 1e-6:
            lateral = np.asarray([0.0, 0.0, 1.0], dtype=float)
            local_x = lateral - local_y * float(np.dot(lateral, local_y))
        local_x /= max(float(np.linalg.norm(local_x)), 1e-12)
        local_z = np.cross(local_x, local_y)
        local_z /= max(float(np.linalg.norm(local_z)), 1e-12)
        return np.column_stack((local_x, local_y, local_z))

    @staticmethod
    def _quat(matrix: np.ndarray) -> Quat:
        value = Rotation.from_matrix(matrix).as_quat()
        return Quat(x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3]))

    @staticmethod
    def _minimal_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the shortest rotation taking one direction onto another."""

        source = source / max(float(np.linalg.norm(source)), 1e-12)
        target = target / max(float(np.linalg.norm(target)), 1e-12)
        cross = np.cross(source, target)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if cross_norm < 1e-10:
            if dot > 0.0:
                return np.eye(3, dtype=float)
            axis = np.cross(source, np.asarray([1.0, 0.0, 0.0]))
            if float(np.linalg.norm(axis)) < 1e-8:
                axis = np.cross(source, np.asarray([0.0, 1.0, 0.0]))
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            return Rotation.from_rotvec(axis * math.pi).as_matrix()
        axis = cross / cross_norm
        return Rotation.from_rotvec(axis * math.acos(dot)).as_matrix()

    def solve_leg(
        self,
        bones: Mapping[str, BonePose],
        side: str,
        ankle_target: np.ndarray,
        *,
        foot_world_rotation: np.ndarray | None = None,
    ) -> dict[str, Quat]:
        """Return exact local delta rotations for a planted/swinging ankle target."""

        upper_name = f"{side}UpperLeg"
        lower_name = f"{side}LowerLeg"
        foot_name = f"{side}Foot"
        upper_index = self.node_by_canonical[upper_name]
        lower_index = self.node_by_canonical[lower_name]
        foot_index = self.node_by_canonical[foot_name]
        parent_index = self.parents[upper_index]

        neutral_world = self.world_matrices(bones)
        parent_world = neutral_world[parent_index]
        hip = neutral_world[upper_index][:3, 3]
        upper_length = float(np.linalg.norm(self.rest[lower_index].translation))
        lower_length = float(np.linalg.norm(self.rest[foot_index].translation))
        reach = ankle_target - hip
        distance = float(np.linalg.norm(reach))
        minimum = abs(upper_length - lower_length) + 1e-5
        maximum = upper_length + lower_length - 1e-5
        distance = float(np.clip(distance, minimum, maximum))
        direction = reach / max(float(np.linalg.norm(reach)), 1e-12)
        along = (
            upper_length * upper_length
            - lower_length * lower_length
            + distance * distance
        ) / (2.0 * distance)
        bend_height = math.sqrt(max(0.0, upper_length * upper_length - along * along))
        lateral = parent_world[:3, :3] @ self.rest[upper_index].rotation[:, 0]
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        bend_direction = np.cross(direction, lateral)
        bend_direction /= max(float(np.linalg.norm(bend_direction)), 1e-12)
        knee = hip + direction * along + bend_direction * bend_height

        upper_world_rotation = self._orientation_along(knee - hip, lateral)
        upper_local_rotation = parent_world[:3, :3].T @ upper_world_rotation
        upper_delta = self.rest[upper_index].rotation.T @ upper_local_rotation

        lower_world_rotation = self._orientation_along(ankle_target - knee, lateral)
        lower_local_rotation = upper_world_rotation.T @ lower_world_rotation
        lower_delta = self.rest[lower_index].rotation.T @ lower_local_rotation

        desired_foot_world = (
            foot_world_rotation
            if foot_world_rotation is not None
            else neutral_world[foot_index][:3, :3]
        )
        foot_local_rotation = lower_world_rotation.T @ desired_foot_world
        foot_delta = self.rest[foot_index].rotation.T @ foot_local_rotation
        return {
            upper_name: self._quat(upper_delta),
            lower_name: self._quat(lower_delta),
            foot_name: self._quat(foot_delta),
        }

    def solve_arm(
        self,
        bones: Mapping[str, BonePose],
        side: str,
        hand_target: np.ndarray,
        *,
        hand_world_rotation: np.ndarray | None = None,
        bend_hint_world: np.ndarray | None = None,
    ) -> dict[str, Quat]:
        """Return exact local rotations for a planted or reaching hand target.

        Unlike the lightweight authoring-space arm solve, this evaluates the
        complete posed hierarchy first.  It therefore remains correct when
        the pelvis and chest are horizontal, as in a crawl or push-up.
        """

        upper_name = f"{side}UpperArm"
        lower_name = f"{side}LowerArm"
        hand_name = f"{side}Hand"
        upper_index = self.node_by_canonical[upper_name]
        lower_index = self.node_by_canonical[lower_name]
        hand_index = self.node_by_canonical[hand_name]
        parent_index = self.parents[upper_index]

        current_world = self.world_matrices(bones)
        parent_world = current_world[parent_index]
        shoulder = current_world[upper_index][:3, 3]
        current_elbow = current_world[lower_index][:3, 3]
        upper_length = float(np.linalg.norm(self.rest[lower_index].translation))
        lower_length = float(np.linalg.norm(self.rest[hand_index].translation))
        reach = hand_target - shoulder
        raw_distance = float(np.linalg.norm(reach))
        minimum = abs(upper_length - lower_length) + 1e-5
        maximum = upper_length + lower_length - 1e-5
        distance = float(np.clip(raw_distance, minimum, maximum))
        direction = reach / max(raw_distance, 1e-12)
        reachable_target = shoulder + direction * distance
        along = (
            upper_length * upper_length
            - lower_length * lower_length
            + distance * distance
        ) / (2.0 * distance)
        bend_height = math.sqrt(max(0.0, upper_length * upper_length - along * along))

        # Preserve the authored elbow side as the pole. This keeps the exact
        # solve continuous with the incoming pose instead of choosing an
        # arbitrary world-axis bend when the torso rotates.
        bend_direction = (
            np.asarray(bend_hint_world, dtype=float).copy()
            if bend_hint_world is not None
            else current_elbow - shoulder
        )
        bend_direction -= direction * float(np.dot(bend_direction, direction))
        if float(np.linalg.norm(bend_direction)) < 1e-6:
            bend_direction = np.asarray([0.0, -1.0, 0.0], dtype=float)
            bend_direction -= direction * float(np.dot(bend_direction, direction))
        bend_direction /= max(float(np.linalg.norm(bend_direction)), 1e-12)
        elbow = shoulder + direction * along + bend_direction * bend_height

        upper_rest_direction = self.rest[lower_index].translation.copy()
        upper_rest_direction /= max(
            float(np.linalg.norm(upper_rest_direction)),
            1e-12,
        )
        upper_base_world = parent_world[:3, :3] @ self.rest[upper_index].rotation
        upper_alignment = self._minimal_alignment(
            upper_base_world @ upper_rest_direction,
            elbow - shoulder,
        )
        upper_world_rotation = upper_alignment @ upper_base_world
        upper_local_rotation = parent_world[:3, :3].T @ upper_world_rotation
        upper_delta = self.rest[upper_index].rotation.T @ upper_local_rotation

        lower_rest_direction = self.rest[hand_index].translation.copy()
        lower_rest_direction /= max(
            float(np.linalg.norm(lower_rest_direction)),
            1e-12,
        )
        lower_base_world = upper_world_rotation @ self.rest[lower_index].rotation
        lower_alignment = self._minimal_alignment(
            lower_base_world @ lower_rest_direction,
            reachable_target - elbow,
        )
        lower_world_rotation = lower_alignment @ lower_base_world
        lower_local_rotation = upper_world_rotation.T @ lower_world_rotation
        lower_delta = self.rest[lower_index].rotation.T @ lower_local_rotation

        desired_hand_world = (
            hand_world_rotation
            if hand_world_rotation is not None
            else current_world[hand_index][:3, :3]
        )
        hand_local_rotation = lower_world_rotation.T @ desired_hand_world
        hand_delta = self.rest[hand_index].rotation.T @ hand_local_rotation
        return {
            upper_name: self._quat(upper_delta),
            lower_name: self._quat(lower_delta),
            hand_name: self._quat(hand_delta),
        }


@lru_cache(maxsize=1)
def rig_kinematics() -> RigKinematics:
    """The one rig this process uses.

    **The cache is load-bearing for provenance, not a performance detail.** It
    is what makes "one skeleton per process" true, which is what lets
    ``asset_sha256`` on this instance describe every frame derived anywhere in
    the process. Clearing it between tests is the most natural thing for a
    fixture author to reach for and it would silently break that. ``conftest.py``
    carries the same warning and a test asserts the cache is never cleared --
    do not add one without talking to lane ``anatomy``.
    """

    return RigKinematics(
        PROJECT_ROOT / "assets" / "models" / "human-male.glb",
        PROJECT_ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json",
    )
