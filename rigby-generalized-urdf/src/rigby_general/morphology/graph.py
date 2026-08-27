"""Kinematic structure recovered from a compiled model.

Talmy's Figure/Ground assignment criteria -- the Figure is the smaller, movable,
to-be-located element; the Ground is the larger, more permanent reference -- do
not need to be imposed on a robot. A kinematic tree already satisfies them: the
base is the ultimate Ground, and every link is a Figure relative to its parent.
So this module does not invent a reference hierarchy, it reads the one the URDF
already encodes.

Two structural facts are derived here and used everywhere downstream:

*The base* is the single body hanging off the world. Everything welded to it is
still the base as far as physics is concerned, but the named root link is what
the ``BASE`` role and the ``BASE_DRIFT`` gate point at.

*An effector cluster* is a set of leaf bodies that share a branch point. That is
the whole definition -- no name matching. Two leaves under one parent is a jaw
shape; three is a hand shape; a lone leaf is a tip. Whether any of them actually
*closes* is a separate, behavioural question answered in :mod:`.measure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import mujoco

from ..errors import GeneralFailureCode, MorphologyError
from ..ingest.loader import body_names, joint_names


ACTUATED_JOINT_TYPES = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


@dataclass(frozen=True, slots=True)
class EffectorCluster:
    """Leaves sharing a branch point, plus the joints that move them."""

    attach_body: int
    """Where the cluster hangs off the chain -- the palm, in gripper terms."""

    member_bodies: tuple[int, ...]
    """The leaf bodies themselves, sorted by name for determinism."""

    interior_joints: tuple[int, ...]
    """Actuated joints strictly below ``attach_body``: closure candidates."""


class KinematicGraph:
    """A read-only view of body and joint structure, indexed for traversal."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.body_names = body_names(model)
        self.joint_names = joint_names(model)

        children: dict[int, list[int]] = {index: [] for index in range(model.nbody)}
        for index in range(1, model.nbody):
            children[int(model.body_parentid[index])].append(index)
        self._children = {
            key: tuple(sorted(value, key=lambda i: self.body_names[i]))
            for key, value in children.items()
        }

        world_children = self._children[0]
        if not world_children:
            raise MorphologyError(
                GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
                "The model has no body attached to the world",
            )
        if len(world_children) > 1:
            names = [self.body_names[index] for index in world_children]
            raise MorphologyError(
                GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
                "The model has more than one root body, so which one is the robot "
                f"is ambiguous: {names}. Upload a single robot, not a scene.",
                details={"roots": names},
            )
        self.base_body = world_children[0]

    # -- traversal ---------------------------------------------------------

    def children(self, body: int) -> tuple[int, ...]:
        return self._children[body]

    def parent(self, body: int) -> int:
        return int(self.model.body_parentid[body])

    def is_leaf(self, body: int) -> bool:
        return not self._children[body]

    def path_from_base(self, body: int) -> tuple[int, ...]:
        """Bodies from the base down to ``body`` inclusive."""

        trail: list[int] = []
        current = body
        while current != 0:
            trail.append(current)
            if current == self.base_body:
                break
            current = self.parent(current)
        else:  # pragma: no cover - only for a body outside the robot subtree
            raise MorphologyError(
                GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
                f"body {self.body_names[body]!r} is not below the base",
            )
        return tuple(reversed(trail))

    def subtree(self, body: int) -> tuple[int, ...]:
        collected: list[int] = []
        stack = [body]
        while stack:
            current = stack.pop()
            collected.append(current)
            stack.extend(self._children[current])
        return tuple(sorted(collected))

    # -- joints ------------------------------------------------------------

    @cached_property
    def actuated_joints(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self.model.njnt)
            if self.model.jnt_type[index] in ACTUATED_JOINT_TYPES
        )

    def joints_of_body(self, body: int) -> tuple[int, ...]:
        start = int(self.model.body_jntadr[body])
        count = int(self.model.body_jntnum[body])
        if start < 0 or count == 0:
            return ()
        return tuple(
            index
            for index in range(start, start + count)
            if self.model.jnt_type[index] in ACTUATED_JOINT_TYPES
        )

    def joints_on_path(self, bodies: tuple[int, ...]) -> tuple[int, ...]:
        collected: list[int] = []
        for body in bodies:
            collected.extend(self.joints_of_body(body))
        return tuple(collected)

    def qpos_address(self, joint: int) -> int:
        return int(self.model.jnt_qposadr[joint])

    def dof_address(self, joint: int) -> int:
        return int(self.model.jnt_dofadr[joint])

    # -- clusters ----------------------------------------------------------

    def branch_point(self, leaf: int) -> int:
        """The nearest ancestor with more than one child, else the leaf's parent.

        A single-leaf chain has no branch, so the parent stands in. That keeps a
        rigid tool tip and a two-finger jaw on the same footing: both are "the
        thing hanging off the end", and only the closure test distinguishes them.
        """

        current = self.parent(leaf)
        while current not in (0, self.base_body):
            if len(self._children[current]) > 1:
                return current
            current = self.parent(current)
        return self.parent(leaf)

    @cached_property
    def clusters(self) -> tuple[EffectorCluster, ...]:
        grouped: dict[int, list[int]] = {}
        for body in range(1, self.model.nbody):
            if not self.is_leaf(body):
                continue
            grouped.setdefault(self.branch_point(body), []).append(body)

        clusters: list[EffectorCluster] = []
        for attach, leaves in grouped.items():
            members = tuple(sorted(leaves, key=lambda i: self.body_names[i]))
            interior = tuple(
                joint
                for body in self.subtree(attach)
                if body != attach
                for joint in self.joints_of_body(body)
            )
            clusters.append(
                EffectorCluster(
                    attach_body=attach,
                    member_bodies=members,
                    interior_joints=interior,
                )
            )
        return tuple(
            sorted(clusters, key=lambda cluster: self.body_names[cluster.attach_body])
        )

    def chain_bodies(self, cluster: EffectorCluster) -> tuple[int, ...]:
        """Base down to the cluster attachment point."""

        return self.path_from_base(cluster.attach_body)

    def geoms_of_body(self, body: int) -> tuple[int, ...]:
        start = int(self.model.body_geomadr[body])
        count = int(self.model.body_geomnum[body])
        if start < 0 or count == 0:
            return ()
        return tuple(range(start, start + count))

    def collidable_geoms_of_body(self, body: int) -> tuple[int, ...]:
        return tuple(
            index
            for index in self.geoms_of_body(body)
            if self.model.geom_contype[index] or self.model.geom_conaffinity[index]
        )
