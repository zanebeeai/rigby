"""The rig's actual skinned hand surface, for contact against scene objects.

Every grip the compiler has ever produced was placed by a rule about *pivots* --
a wrist offset, a knuckle centroid, a fingertip distance. Pivots are not the
thing that touches. The skeleton's metacarpal-head plane sits 37.6 mm inside
the palm's own surface, so a grip seated on the pivot plane buries a third of
a 60 mm block in the hand, and every pivot-based measure reports it as clear.

This module reads the skinning arrays out of the same GLB the rest of the
pipeline poses, so contact is measured against the surface a viewer actually
sees rather than against a proxy for it.

The GLB is parsed a second time here rather than threaded out of
:class:`~rigby_poc.kinematics.RigKinematics`, which keeps that class's parse
untouched; :func:`hand_mesh` asserts the two digests agree, so "one skeleton
per process" still means one asset per process and a mismatch is an error
rather than a silent divergence between the posed skeleton and the skin it
carries.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .kinematics import PROJECT_ROOT, rig_kinematics
from .models import BonePose

#: glTF componentType -> (struct code, byte width).
_COMPONENT: dict[int, tuple[str, int]] = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}

#: glTF accessor type -> component count.
_COMPONENTS_PER: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}

#: A vertex belongs to a group when the group's bones carry more than this
#: share of its skin weight. Half is the natural cut: it makes membership
#: unique, so no vertex is counted against two digits.
_DOMINANT_WEIGHT = 0.5

FINGER_SEGMENTS: tuple[str, ...] = ("Proximal", "Intermediate", "Distal")
THUMB_SEGMENTS: tuple[str, ...] = ("Metacarpal", "Proximal", "Distal")


def _chunks(raw: bytes) -> tuple[dict, bytes]:
    """The JSON document and the binary buffer of a GLB."""

    magic, version, _ = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("rig asset is not a glTF 2.0 binary")
    offset, document, binary = 12, None, None
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \x00"))
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length
    if document is None or binary is None:
        raise ValueError("rig asset is missing a JSON or binary chunk")
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    """One glTF accessor as a dense ``(count, components)`` array."""

    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    code, width = _COMPONENT[accessor["componentType"]]
    count = _COMPONENTS_PER[accessor["type"]]
    stride = view.get("byteStride") or width * count
    base = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    layout = "<" + code * count
    values = [
        struct.unpack_from(layout, binary, base + row * stride)
        for row in range(accessor["count"])
    ]
    return np.asarray(values, dtype=float if code == "f" else np.int64)


@dataclass(frozen=True)
class HandMesh:
    """One hand's skinned surface, grouped by the part that carries it."""

    #: Rest-pose vertex positions, one row per vertex of this hand.
    rest: np.ndarray
    #: Skin joint indices and weights, aligned with :attr:`rest`.
    joint_indices: np.ndarray
    joint_weights: np.ndarray
    #: ``skin.joints`` resolved to node indices, and their inverse binds.
    skin_nodes: tuple[int, ...]
    inverse_bind: np.ndarray
    #: Row selections into :attr:`rest`, keyed by ``"Palm"`` and digit name.
    groups: Mapping[str, np.ndarray]
    side: str

    def world(self, bones: Mapping[str, BonePose]) -> np.ndarray:
        """Skin this hand into world space for one pose."""

        matrices = rig_kinematics().world_matrices(bones)
        palette = np.stack(
            [
                np.asarray(matrices[node]) @ self.inverse_bind[slot]
                for slot, node in enumerate(self.skin_nodes)
            ]
        )
        blended = np.einsum(
            "vk,vkij->vij",
            self.joint_weights,
            palette[self.joint_indices],
        )
        homogeneous = np.hstack([self.rest, np.ones((len(self.rest), 1))])
        return np.einsum("vij,vj->vi", blended, homogeneous)[:, :3]

    def local(self, bones: Mapping[str, BonePose]) -> np.ndarray:
        """Skin this hand into the wrist's own frame for one pose."""

        kinematics = rig_kinematics()
        wrist = kinematics.world_matrices(bones)[
            kinematics.node_by_canonical[f"{self.side}Hand"]
        ]
        return (self.world(bones) - wrist[:3, 3]) @ wrist[:3, :3]


@lru_cache(maxsize=2)
def hand_mesh(side: str) -> HandMesh:
    """The skinned surface of one hand; ``side`` in {left, right}."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    path = Path(PROJECT_ROOT) / "assets" / "models" / "human-male.glb"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    kinematics = rig_kinematics()
    if digest != kinematics.asset_sha256:
        raise ValueError(
            "hand mesh and posed skeleton came from different rig assets: "
            f"{digest} against {kinematics.asset_sha256}"
        )
    document, binary = _chunks(raw)

    primitive = document["meshes"][0]["primitives"][0]
    positions = _accessor(document, binary, primitive["attributes"]["POSITION"])
    indices = _accessor(
        document, binary, primitive["attributes"]["JOINTS_0"]
    ).astype(int)
    weights = _accessor(document, binary, primitive["attributes"]["WEIGHTS_0"])

    skin = document["skins"][0]
    skin_nodes = tuple(int(node) for node in skin["joints"])
    # glTF stores matrices column-major; transpose each into row-major.
    inverse_bind = (
        _accessor(document, binary, skin["inverseBindMatrices"])
        .reshape(-1, 4, 4)
        .transpose(0, 2, 1)
    )
    slot_of_node = {node: slot for slot, node in enumerate(skin_nodes)}

    def share(canonicals: tuple[str, ...]) -> np.ndarray:
        slots = {
            slot_of_node[kinematics.node_by_canonical[name]]
            for name in canonicals
            if name in kinematics.node_by_canonical
            and kinematics.node_by_canonical[name] in slot_of_node
        }
        if not slots:
            return np.zeros(len(positions))
        member = np.isin(indices, list(slots))
        return np.where(member, weights, 0.0).sum(axis=1)

    digits = {
        "Thumb": THUMB_SEGMENTS,
        "Index": FINGER_SEGMENTS,
        "Middle": FINGER_SEGMENTS,
        "Ring": FINGER_SEGMENTS,
        "Little": FINGER_SEGMENTS,
    }
    shares = {"Palm": share((f"{side}Hand",))}
    for digit, segments in digits.items():
        shares[digit] = share(tuple(f"{side}{digit}{s}" for s in segments))

    keep = np.zeros(len(positions), dtype=bool)
    for value in shares.values():
        keep |= value > _DOMINANT_WEIGHT
    rows = np.nonzero(keep)[0]
    renumber = {int(v): i for i, v in enumerate(rows)}
    groups = {
        name: np.asarray(
            [renumber[int(v)] for v in np.nonzero(value > _DOMINANT_WEIGHT)[0]],
            dtype=int,
        )
        for name, value in shares.items()
    }
    return HandMesh(
        rest=positions[rows],
        joint_indices=indices[rows],
        joint_weights=weights[rows],
        skin_nodes=skin_nodes,
        inverse_bind=inverse_bind,
        groups=groups,
        side=side,
    )


def box_signed_distance(
    points: np.ndarray,
    centre: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Signed distance from points to an oriented box; negative is inside."""

    local = np.abs((points - centre) @ rotation) - half_extents
    outside = np.linalg.norm(np.maximum(local, 0.0), axis=1)
    return outside + np.minimum(local.max(axis=1), 0.0)


def box_escape(
    points: np.ndarray,
    centre: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    along: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth inside an oriented box and the way out, per point.

    Returns ``(depth, direction)``: ``depth`` is positive for points inside the
    box (zero outside) and ``direction`` is the world-space unit vector along
    which moving the point by ``depth`` puts it on a face.

    Without ``along`` the way out is the nearest face. With ``along`` (a world
    unit vector) every inside point leaves in that one direction, however far
    the face is. A support surface needs the latter: a hand pushed through a
    thin table top has vertices nearer its underside than its top, and the
    nearest face for those is *down*, which is not where a hand gets out of a
    table.
    """

    local = (points - centre) @ rotation
    slack = half_extents - np.abs(local)
    inside = np.all(slack > 0.0, axis=1)
    rows = np.arange(len(points))
    if along is None:
        axis = np.argmin(slack, axis=1)
        depth = np.where(inside, slack[rows, axis], 0.0)
        sign = np.sign(local[rows, axis])
        sign[sign == 0.0] = 1.0
        local_direction = np.zeros_like(points)
        local_direction[rows, axis] = sign
        return depth, local_direction @ rotation.T
    unit = np.asarray(along, dtype=float)
    unit = unit / np.linalg.norm(unit)
    local_unit = unit @ rotation
    # Distance to leave through the face the ray meets first.
    exits = np.full(len(points), np.inf)
    for axis in range(3):
        if abs(local_unit[axis]) < 1e-9:
            continue
        face = half_extents[axis] if local_unit[axis] > 0.0 else -half_extents[axis]
        exits = np.minimum(exits, (face - local[:, axis]) / local_unit[axis])
    depth = np.where(inside, exits, 0.0)
    return depth, np.tile(unit, (len(points), 1))


__all__ = ["HandMesh", "box_escape", "box_signed_distance", "hand_mesh"]
