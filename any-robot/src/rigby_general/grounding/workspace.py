"""A per-chain frame in which every schema target can be expressed.

This is where magnitude neutrality actually cashes out. A schema says "distal,
straight out"; a workspace frame says what that is in metres *for this arm*. The
same three numbers -- a fraction of reach, an azimuth, an elevation -- describe a
0.39 m desktop arm and a 2.0 m long-reach arm, and neither the planner nor the
schema program ever sees the difference.

The frame is built from measurements, not conventions:

``out``  the horizontal bearing from the chain's root to the mean of its own
         reachable tip positions. Where this arm is built to point. It needs no
         operator confirmation because it is a fact about the mechanism, which is
         why free-space schemas ground the moment a robot is uploaded.
``up``   against gravity.
``side`` ``up`` cross ``out``, completing a right-handed frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..contracts import KinematicChainV1, RobotMorphologyV1
from ..morphology import envelope as envelope_module


@dataclass(frozen=True, slots=True)
class WorkspaceFrame:
    """Where a chain can put its effector, in its own terms."""

    chain_id: str
    origin: np.ndarray
    out: np.ndarray
    side: np.ndarray
    up: np.ndarray
    reach_m: float
    """Furthest reach in any direction. Kept for reporting; targets use the
    envelope, because a reachable set is not a ball."""

    envelope: np.ndarray
    inner: np.ndarray
    """The near surface of the reachable shell. See ``point``."""

    home: np.ndarray
    """Where the figure site rests before anything is asked of it."""

    figure_site: str

    def directional_reach(self, azimuth_deg: float, elevation_deg: float) -> float:
        """How far this chain reaches along one bearing, as measured."""

        return max(
            envelope_module.directional_reach(
                self.envelope, azimuth_deg, elevation_deg
            ),
            1e-4,
        )

    def inner_reach(self, azimuth_deg: float, elevation_deg: float) -> float:
        """How close to the root this chain can bring its effector on a bearing."""

        near = envelope_module.inner_reach(self.inner, azimuth_deg, elevation_deg)
        far = self.directional_reach(azimuth_deg, elevation_deg)
        # Never let the near surface meet or cross the far one; a degenerate
        # shell would collapse every remove onto a single radius.
        return float(max(0.0, min(near, 0.9 * far)))

    def bearing_of(self, point: np.ndarray) -> tuple[float, float]:
        """Azimuth and elevation of a world point, in this chain's frame."""

        offset = np.asarray(point, dtype=float) - self.origin
        radius = float(np.linalg.norm(offset))
        if radius < 1e-9:
            return 0.0, 0.0
        return (
            math.degrees(math.atan2(float(offset @ self.side), float(offset @ self.out))),
            math.degrees(math.asin(max(-1.0, min(1.0, float(offset @ self.up) / radius)))),
        )

    def point(
        self,
        *,
        radius_fraction: float,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ) -> np.ndarray:
        """A world point at a fraction of the reachable shell in that direction.

        Zero is the closest the effector can be brought to the chain root on this
        bearing, and one is the furthest -- not zero and the maximum. The
        difference only shows on a real arm: a target at ``0.1`` of an iiwa7's
        maximum reach is 110 mm from the base, and the tool cannot get within
        173 mm of it, so the whole of ``adjacent`` was landing inside the
        machine. On an arm that folds down to nothing the two readings agree.
        """

        near = self.inner_reach(azimuth_deg, elevation_deg)
        far = self.directional_reach(azimuth_deg, elevation_deg)
        radius = near + radius_fraction * (far - near)
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        horizontal = math.cos(elevation)
        direction = (
            horizontal * math.cos(azimuth) * self.out
            + horizontal * math.sin(azimuth) * self.side
            + math.sin(elevation) * self.up
        )
        return self.origin + radius * direction

    def point_at_reach_fraction(
        self,
        *,
        radius_fraction: float,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ) -> np.ndarray:
        """A world point at a fraction of the *maximum* reach on a bearing.

        Deliberately distinct from :meth:`point`, which spans the reachable
        shell. Both are useful and they mean different things.

        ``point`` is the language-facing one: a degree of remove is a position
        within what the arm can reach, so its zero is the near surface.

        This one is for building scenes. Where to stand a block is a question
        about the arm's outward extent, and the fractions that answer it were
        calibrated against that extent. Quietly re-basing them on the shell moved
        every block 60 mm further out and turned a certified grasp into jaws
        closing inside the block.
        """

        radius = radius_fraction * self.directional_reach(azimuth_deg, elevation_deg)
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        horizontal = math.cos(elevation)
        direction = (
            horizontal * math.cos(azimuth) * self.out
            + horizontal * math.sin(azimuth) * self.side
            + math.sin(elevation) * self.up
        )
        return self.origin + radius * direction

    def clamp(self, point: np.ndarray, *, margin: float = 0.95) -> np.ndarray:
        """Pull a point back inside the reach measured along its own bearing.

        Used only for waypoints a schema derives from another waypoint -- an arc
        apex, say. A *requested* target that lands outside reach is a grounding
        failure and must be reported, never quietly shortened.
        """

        offset = np.asarray(point, dtype=float) - self.origin
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            azimuth, elevation = 0.0, 0.0
        else:
            azimuth, elevation = self.bearing_of(point)
        limit = margin * self.directional_reach(azimuth, elevation)
        floor = self.inner_reach(azimuth, elevation)
        if distance < 1e-9:
            # Dead on the root: there is no bearing to push out along, so use
            # the frame's own forward direction.
            return self.origin + max(floor, 0.0) * self.out
        if distance < floor:
            return self.origin + offset * (floor / distance)
        if distance <= limit:
            return point
        return self.origin + offset * (limit / distance)

    def clamp_rising(self, point: np.ndarray, *, margin: float = 0.95) -> np.ndarray:
        """Pull a point inside reach by giving up radius before height.

        ``clamp`` scales the whole offset, so a target outside the envelope comes
        back *down* as well as in. For a lift that is the wrong trade: the height
        is the thing being demonstrated and the radius is free, and an arm asked
        to raise something it is already holding draws it inward rather than
        setting it back down. A block near the reach limit -- which is where an
        authored world tends to put it -- had its lift target scaled back to
        almost the height it started at, and the attempt then failed
        `object_not_lifted` while holding the block perfectly well.

        Keeps the requested height and takes the furthest radius along the same
        bearing that reach allows, falling back to ``clamp`` only when no radius
        at that height is reachable -- the case where the height itself is out of
        range, and giving up radius cannot buy it back.
        """

        point = np.asarray(point, dtype=float)
        offset = point - self.origin
        horizontal = np.array([offset[0], offset[1], 0.0])
        radius = float(np.linalg.norm(horizontal))
        if radius < 1e-9 or self.contains(point, margin=margin):
            return point if radius >= 1e-9 else self.clamp(point, margin=margin)

        direction = horizontal / radius
        rise = np.array([0.0, 0.0, offset[2]])
        low, high = 0.0, radius
        best: "np.ndarray | None" = None
        for _ in range(32):
            middle = 0.5 * (low + high)
            candidate = self.origin + direction * middle + rise
            if self.contains(candidate, margin=margin):
                best = candidate
                low = middle
            else:
                high = middle
        return best if best is not None else self.clamp(point, margin=margin)

    def contains(self, point: np.ndarray, *, margin: float = 1.0) -> bool:
        """Inside the reachable shell -- outside it *and* outside the hole.

        This tested only the outer bound, so a point in the arm's own inner
        hole -- the volume too close in for it to fold into -- reported as
        reachable. The workspace is a shell, not a ball: `inner_reach` is
        measured alongside `directional_reach` for exactly this reason, and
        `object_inside_reach_hole` exists as a refusal because the hole is real.
        Every caller asking "can the arm be here" was getting yes for a region
        it cannot occupy, which is why clamping a path into the envelope left
        the unreachable points untouched and the solver went on failing on them.
        """

        offset = np.asarray(point, dtype=float) - self.origin
        distance = float(np.linalg.norm(offset))
        azimuth, elevation = self.bearing_of(point)
        if distance < self.inner_reach(azimuth, elevation) - 1e-9:
            return False
        return bool(
            distance <= margin * self.directional_reach(azimuth, elevation) + 1e-9
        )


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return fallback
    return vector / norm


def build_workspace_frame(
    model: mujoco.MjModel,
    morphology: RobotMorphologyV1,
    chain: KinematicChainV1,
    *,
    figure_site: str,
) -> WorkspaceFrame:
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(morphology_rest_qpos(model), dtype=float)
    mujoco.mj_kinematics(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chain.root_body)
    if root_id < 0:  # pragma: no cover - defensive
        raise KeyError(f"chain root body {chain.root_body!r} is not in the model")
    origin = np.array(data.xpos[root_id], dtype=float)

    frame = morphology.intrinsic_frame
    up = _unit(
        np.array([frame.up.x, frame.up.y, frame.up.z], dtype=float),
        np.array([0.0, 0.0, 1.0]),
    )

    # The working direction was stored at measurement time precisely so that the
    # envelope below is read in the frame it was built in. Recomputing it here
    # would risk the two quietly disagreeing about which way "out" is.
    out = _unit(
        np.array(
            [
                chain.working_direction.x,
                chain.working_direction.y,
                chain.working_direction.z,
            ],
            dtype=float,
        ),
        np.array([1.0, 0.0, 0.0]),
    )
    side = _unit(np.cross(up, out), np.array([0.0, 1.0, 0.0]))

    grid = envelope_module.unflatten(
        chain.reach_envelope_m,
        chain.envelope_azimuth_bins,
        chain.envelope_elevation_bins,
    )
    # A chain measured before the inner surface existed has none recorded; an
    # all-zero grid reproduces the old ball-from-the-origin behaviour exactly.
    inner = (
        envelope_module.unflatten(
            chain.reach_inner_m,
            chain.envelope_azimuth_bins,
            chain.envelope_elevation_bins,
        )
        if chain.reach_inner_m
        else np.zeros_like(grid)
    )

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, figure_site)
    if site_id < 0:  # pragma: no cover - defensive
        raise KeyError(f"figure site {figure_site!r} is not in the model")
    home = np.array(data.site_xpos[site_id], dtype=float)

    return WorkspaceFrame(
        chain_id=chain.chain_id,
        origin=origin,
        out=out,
        side=side,
        up=up,
        reach_m=float(chain.reach_radius_m),
        envelope=grid,
        inner=inner,
        home=home,
        figure_site=figure_site,
    )


def morphology_rest_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Rest configuration, clamped into every joint limit."""

    from ..morphology.measure import neutral_qpos

    return neutral_qpos(model)
