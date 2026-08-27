"""The hand reduced to an opposition quad, and how much object sits inside it.

Every grasp objective this repository has tried measured the digits one at a
time: how far is the thumb from the surface, how far is the index. That answers
"is a finger near the object" and never answers "is the object *between* the
fingers", which is the thing a grasp actually needs. A hand can have five digits
each 2 mm from the block and still be about to knock it away, and that is
precisely what the traces kept showing.

This takes the other view, the one grasp research calls the *virtual finger*:
collapse the four fingers into a single opposing member, and the hand becomes two
vectors.

    thumb base ----------------> thumb tip
        |                            |
        |        the aperture        |
        v                            v
    finger base ---------------> finger tip     (bases and tips averaged over
                                                 index, middle, ring, little)

Joining base to base and tip to tip closes a quadrilateral -- wide at the tips
when the hand is open, narrowing as it closes. That quad is the surface the
grasp will sweep when the fingers come together, so the question "will this
grasp work" becomes a question about area: **how much of the object lies inside
the quad**. An object fully inside is enclosed. An object outside it is about to
be pushed, whatever the individual digit distances say.

Two things fall out of the same geometry. The **closure direction** is the tip-to-
tip vector: that is the way the aperture shuts. And the **overlap fraction** is a
scalar objective a search can maximise, which the per-digit distances never were,
because improving one digit routinely made another worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .kinematics import rig_kinematics
from .models import BonePose, SceneObject
from .primitives import (
    MAX_WRIST_PITCH_RAD,
    MAX_WRIST_TWIST_RAD,
    MAX_WRIST_YAW_RAD,
)

#: The four members the virtual finger averages over.
_FINGERS = ("Index", "Middle", "Ring", "Little")
_TIPS = ("index", "middle", "ring", "little")

#: Samples per axis across the quad. 24x24 resolves the block's 60 mm face to
#: about 3 mm, well under the millimetre-scale differences a grasp turns on,
#: and costs no simulation.
_SAMPLES = 24

#: Digit capsules are 6.5-7.5 mm in radius; an approach that clears the surface
#: by less than this is already touching.
_APPROACH_MARGIN_M = 0.010


@dataclass(frozen=True)
class ApertureQuad:
    """The opposition quad, in world space, corners in winding order."""

    thumb_base: np.ndarray
    thumb_tip: np.ndarray
    finger_tip: np.ndarray
    finger_base: np.ndarray

    @property
    def corners(self) -> np.ndarray:
        return np.stack(
            [self.thumb_base, self.thumb_tip, self.finger_tip, self.finger_base]
        )

    @property
    def closure_direction(self) -> np.ndarray:
        """Tip to tip: the way the aperture shuts."""
        delta = self.finger_tip - self.thumb_tip
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1e-9 else np.zeros(3)

    @property
    def tip_span_m(self) -> float:
        return float(np.linalg.norm(self.finger_tip - self.thumb_tip))

    @property
    def base_span_m(self) -> float:
        return float(np.linalg.norm(self.finger_base - self.thumb_base))

    @property
    def area_m2(self) -> float:
        """Two triangles, so a non-planar quad is still measured honestly."""
        a, b, c, d = self.corners
        return 0.5 * (
            float(np.linalg.norm(np.cross(b - a, c - a)))
            + float(np.linalg.norm(np.cross(c - a, d - a)))
        )

    def sample(self, resolution: int = _SAMPLES) -> np.ndarray:
        """Points spread over the quad by bilinear interpolation."""
        u = np.linspace(0.0, 1.0, resolution)
        v = np.linspace(0.0, 1.0, resolution)
        grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
        gu = grid_u[..., None]
        gv = grid_v[..., None]
        # thumb edge (base->tip) blended toward the finger edge (base->tip)
        thumb_edge = self.thumb_base * (1 - gu) + self.thumb_tip * gu
        finger_edge = self.finger_base * (1 - gu) + self.finger_tip * gu
        return (thumb_edge * (1 - gv) + finger_edge * gv).reshape(-1, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": [[float(v) for v in c] for c in self.corners],
            "closure_direction": [float(v) for v in self.closure_direction],
            "tip_span_m": self.tip_span_m,
            "base_span_m": self.base_span_m,
            "area_m2": self.area_m2,
        }


#: A digit centre-line rests this far outside the surface when the digit touches
#: it: the physics capsules are 6.5-7.5 mm in radius, so a tip *at* the surface
#: is already pressed into it by its own thickness.
_CONTACT_STANDOFF_M = 0.005

#: Beyond this from the contact standoff a tip is neither touching nor about to.
_SEATING_TOLERANCE_M = 0.018


def _seating(distance: float | np.ndarray) -> float:
    """1.0 when a tip rests on the surface, falling to 0 as it hovers or buries.

    Symmetric on purpose. Hovering means no force and burying means the digit
    was already inside before it closed, and both produce the same failure --
    nothing to press against.
    """
    error = abs(float(distance) - _CONTACT_STANDOFF_M)
    return float(max(0.0, 1.0 - error / _SEATING_TOLERANCE_M))


def _height(points: np.ndarray) -> np.ndarray:
    """World height of each point. The rig's up axis is y."""
    array = np.atleast_2d(np.asarray(points, dtype=float))
    return array[:, 1]


def _obstructed(
    points: np.ndarray,
    support_height_m: float,
    obstacles: tuple[SceneObject, ...],
) -> float:
    """Total depth by which these points intrude into anything not being grasped.

    Stated as one rule over the whole scene rather than a test for the table,
    because the table was never the point. The planner checked clearance against
    the object it was reaching for and against nothing else, so it happily chose
    a placement that drove every finger through whatever the object was standing
    on -- measured, the little and ring fingertips reached 9.7 cm below the
    tabletop and stayed in contact for over 400 steps. A hand buried in the
    scenery fights it for the entire approach.

    The support surface is a half-space and the other scene objects are boxes,
    so both are expressed as depths and summed. A rule that named the table
    would break on the first object placed on a shelf, a rung or the floor.
    """
    total = 0.0
    if support_height_m:
        total += float(np.sum(np.maximum(0.0, support_height_m - _height(points))))
    for obstacle in obstacles:
        total += float(np.sum(np.maximum(0.0, -_box_distance(points, obstacle))))
    return total


def _palm_samples(pose: dict[str, BonePose], side: str) -> np.ndarray:
    """Points filling the palm box, in world space.

    The palm was the one part of the hand the plan never looked at, and it is
    the largest: 8.9 x 12.7 x 2.4 cm against a 6 cm block. Unchecked, the chosen
    placement buried it in the object, and the simulation duly knocked the block
    9.07 cm across the table during the approach -- before a single finger had
    closed. Every grasp measured after that moment was measured against a block
    that was no longer there.

    Geometry is taken from ``physics`` rather than restated here. The two
    disagreeing is the entire failure mode this guards against, so they read
    from one definition. The import is deferred because ``physics`` pulls in
    MuJoCo and planning must stay importable without it.
    """
    from .models import Hand
    from .physics import _frame_hand_landmarks, _palm_transform
    from .models import ClipFrame

    hand = Hand.LEFT if side == "left" else Hand.RIGHT
    landmarks = _frame_hand_landmarks(
        ClipFrame(time_s=0.0, bones=pose, objects={}), hand
    )
    centre, rotation = _palm_transform(landmarks, hand)
    across = float(
        np.linalg.norm(
            landmarks[f"{side}IndexProximal"] - landmarks[f"{side}LittleProximal"]
        )
    )
    along = float(
        np.linalg.norm(landmarks[f"{side}MiddleProximal"] - landmarks[f"{side}Hand"])
    )
    half = np.asarray(
        [max(0.025, across * 0.55), max(0.028, along * 0.58), 0.012], dtype=float
    )
    axis = np.linspace(-1.0, 1.0, 3)
    grid = np.array([[x, y, z] for x in axis for y in axis for z in axis])
    return (grid * half) @ rotation.T + centre


def aperture_quads(bones: dict[str, BonePose], hand: str) -> dict[str, ApertureQuad]:
    """One quad per finger, each opposed to the thumb.

    Averaging the four fingers into a single virtual finger says whether the
    object is in *an* aperture; it cannot say which opposition pair is holding
    it. Four independent quads can, and they disagree often: a block can sit
    squarely between thumb and index while the little finger is nowhere near it,
    and a mean over the four hides both facts inside one middling number.
    """
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(bones)
    tips = kinematics.fingertip_positions(bones, hand)
    thumb_base = positions[f"{hand}ThumbMetacarpal"].copy()
    thumb_tip = tips["thumb"].copy()
    return {
        tip: ApertureQuad(
            thumb_base=thumb_base,
            thumb_tip=thumb_tip,
            finger_tip=tips[tip].copy(),
            finger_base=positions[f"{hand}{name}Proximal"].copy(),
        )
        for name, tip in zip(_FINGERS, _TIPS)
    }


def thumb_convergence(bones: dict[str, BonePose], hand: str) -> np.ndarray:
    """Where the thumb should travel: the mean tip-to-tip vector.

    The average of the vectors from the thumb tip to each fingertip. Closing
    along it drives the thumb toward the centre of the group it opposes rather
    than toward any one finger.
    """
    kinematics = rig_kinematics()
    tips = kinematics.fingertip_positions(bones, hand)
    thumb_tip = tips["thumb"]
    mean = np.mean([tips[t] - thumb_tip for t in _TIPS], axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-9 else np.zeros(3)


def aperture_quad(bones: dict[str, BonePose], hand: str) -> ApertureQuad:
    """Collapse a posed hand into its opposition quad."""
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(bones)
    tips = kinematics.fingertip_positions(bones, hand)
    finger_base = np.mean(
        [positions[f"{hand}{name}Proximal"] for name in _FINGERS], axis=0
    )
    finger_tip = np.mean([tips[name] for name in _TIPS], axis=0)
    return ApertureQuad(
        thumb_base=positions[f"{hand}ThumbMetacarpal"].copy(),
        thumb_tip=tips["thumb"].copy(),
        finger_tip=finger_tip,
        finger_base=finger_base,
    )


def _box_distance(points: np.ndarray, item: SceneObject) -> np.ndarray:
    """Signed distance to the object's surface; negative inside."""
    from scipy.spatial.transform import Rotation

    centre = np.asarray(item.transform.translation.as_list(), dtype=float)
    rotation = Rotation.from_quat(item.transform.rotation.as_list()).as_matrix()
    half = np.asarray(
        [item.dimensions_m.x / 2, item.dimensions_m.y / 2, item.dimensions_m.z / 2]
    )
    local = np.abs((points - centre) @ rotation) - half
    outside = np.linalg.norm(np.maximum(local, 0.0), axis=-1)
    inside = np.minimum(np.max(local, axis=-1), 0.0)
    return outside + inside


def _inside_box(points: np.ndarray, item: SceneObject) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    centre = np.asarray(item.transform.translation.as_list(), dtype=float)
    rotation = Rotation.from_quat(item.transform.rotation.as_list()).as_matrix()
    half = np.asarray(
        [item.dimensions_m.x / 2, item.dimensions_m.y / 2, item.dimensions_m.z / 2]
    )
    local = (points - centre) @ rotation
    return np.all(np.abs(local) <= half, axis=-1)


def overlap(
    quad: ApertureQuad,
    item: SceneObject,
    *,
    resolution: int = _SAMPLES,
) -> dict[str, Any]:
    """How much of the aperture is filled by the object.

    ``covered_fraction`` is the objective worth maximising: it is bounded in
    [0, 1], it rises smoothly as the hand comes to surround the object, and
    unlike a per-digit distance it cannot be improved for one digit at another's
    expense.
    """
    points = quad.sample(resolution)
    inside = _inside_box(points, item)
    fraction = float(np.count_nonzero(inside)) / max(1, inside.size)

    centre = np.asarray(item.transform.translation.as_list(), dtype=float)
    corners = quad.corners
    quad_centre = corners.mean(axis=0)
    return {
        "covered_fraction": fraction,
        "covered_area_m2": fraction * quad.area_m2,
        "aperture_area_m2": quad.area_m2,
        "tip_span_m": quad.tip_span_m,
        "object_centre_offset_m": float(np.linalg.norm(quad_centre - centre)),
        # A grasp needs the object between the tips, not merely touching the
        # quad somewhere: an object clipped by the base edge is in the palm's
        # way, not in the aperture.
        "object_between_tips": bool(
            _inside_box(
                np.stack([(quad.thumb_tip + quad.finger_tip) / 2.0]), item
            )[0]
        ),
        "closure_direction": [float(v) for v in quad.closure_direction],
    }


def aperture_report(
    bones: dict[str, BonePose], hand: str, item: SceneObject
) -> dict[str, Any]:
    """Every opposition pair, measured separately, plus what they add up to."""
    quads = aperture_quads(bones, hand)
    per_finger = {name: overlap(quad, item) for name, quad in quads.items()}
    covered = {name: r["covered_fraction"] for name, r in per_finger.items()}
    holding = [name for name, r in per_finger.items() if r["object_between_tips"]]
    return {
        "per_finger": {name: {**r, "quad": quads[name].to_dict()} for name, r in per_finger.items()},
        "best_finger": max(covered, key=covered.get) if covered else None,
        "best_covered_fraction": max(covered.values(), default=0.0),
        "mean_covered_fraction": float(np.mean(list(covered.values()))) if covered else 0.0,
        # An opposition pair only counts when the object is actually between the
        # tips; coverage alone can be satisfied by a quad the object merely
        # clips at one corner.
        "holding_fingers": holding,
        "opposition_pair_count": len(holding),
        "thumb_convergence": [float(v) for v in thumb_convergence(bones, hand)],
    }


# ---------------------------------------------------------------- planning

@dataclass(frozen=True)
class GraspPlan:
    """Where to put the wrist so the object is inside the aperture."""

    wrist_target: np.ndarray
    wrist_pitch: float
    wrist_yaw: float
    wrist_roll: float
    covered_fraction: float
    closure_direction: np.ndarray
    tip_span_m: float
    evaluations: int
    #: How many of the four thumb-finger pairs have the object between their
    #: tips when the hand is closed. Zero is a hand that shuts beside the object.
    opposition_pairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "aperture_grasp_plan_v1",
            "wrist_target_m": [float(v) for v in self.wrist_target],
            "wrist_pitch": self.wrist_pitch,
            "wrist_yaw": self.wrist_yaw,
            "wrist_roll": self.wrist_roll,
            "covered_fraction": self.covered_fraction,
            "closure_direction": [float(v) for v in self.closure_direction],
            "tip_span_m": self.tip_span_m,
            "evaluations": self.evaluations,
            "opposition_pairs": self.opposition_pairs,
        }


@lru_cache(maxsize=64)
def _cached_plan(signature: tuple, hand: Any, shoulder_key: tuple) -> "GraspPlan":
    """Memoised by the only things the search actually depends on.

    The plan is a function of the object's pose and size, the hand, and the
    shoulder it hangs from -- not of the primitive being compiled. Recomputing
    it per compile put a 17 s search inside every grab in the corpus, which took
    the suite from minutes to over an hour.
    """
    item, base_pose, shoulder, support, obstacles = _PLAN_INPUTS[signature]
    return _plan_grasp_pose(
        base_pose, hand, item, shoulder,
        support_height_m=support, obstacles=obstacles,
    )


#: Non-hashable arguments, held by the signature the cache is keyed on.
_PLAN_INPUTS: dict[tuple, Any] = {}


def plan_grasp_pose(
    base_pose: dict[str, BonePose],
    hand: Any,
    item: SceneObject,
    shoulder: Any,
    support_height_m: float = 0.0,
    obstacles: tuple[SceneObject, ...] = (),
    **kwargs: Any,
) -> "GraspPlan":
    """Cached entry point. See :func:`_plan_grasp_pose` for the search itself."""
    if kwargs:
        return _plan_grasp_pose(
            base_pose, hand, item, shoulder,
            support_height_m=support_height_m, obstacles=obstacles, **kwargs
        )
    translation = item.transform.translation
    rotation = item.transform.rotation
    # The base pose is part of the key: it is the posture the arm solution is
    # measured against, so two rigs (or two rest postures) must not share a plan.
    posture = tuple(
        sorted(
            (name, pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)
            for name, pose in base_pose.items()
        )
    )
    signature = (
        item.id,
        hash(posture),
        (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w),
        (item.dimensions_m.x, item.dimensions_m.y, item.dimensions_m.z),
        hand.value,
        (shoulder.x, shoulder.y, shoulder.z),
        round(float(support_height_m), 5),
        tuple(o.id for o in obstacles),
    )
    _PLAN_INPUTS[signature] = (
        item, base_pose, shoulder, float(support_height_m), tuple(obstacles)
    )
    return _cached_plan(signature, hand, (shoulder.x, shoulder.y, shoulder.z))


def _plan_grasp_pose(
    base_pose: dict[str, BonePose],
    hand: Any,
    item: SceneObject,
    shoulder: Any,
    *,
    support_height_m: float = 0.0,
    obstacles: tuple[SceneObject, ...] = (),
    coarse_step_m: float = 0.045,
    refine_step_m: float = 0.018,
    resolution: int = 8,
) -> GraspPlan:
    """Place the wrist so the closed hand's apertures contain the object.

    Two corrections over scoring one averaged quad on an open hand.

    **Four quads, not one.** Averaging the fingers into a virtual finger reports
    a middling number for a hand gripping firmly with two fingers and missing
    with two, and the placement it picks is a compromise between opposition
    pairs that never existed.

    **Scored on the CLOSED hand.** The aperture shrinks as the hand shuts, so
    the placement that is best for an open hand is not the one holding the
    object at contact -- measured directly, the open-hand version put peak
    coverage in ``lift``, one phase after the fingers had already closed. At the
    winning placement the closed hand holds the block with index and middle at
    0.86 coverage while the open hand holds nothing at all, because an open
    hand's tips are meant to be wider than the object. Requiring both would
    score every real grasp as a failure. The open shape still contributes,
    weighted low, so closure captures the object rather than sweeping it aside.
    """
    from scipy.spatial.transform import Rotation

    from .models import HandShape, PrimitiveParameters, Vec3
    from .primitives import arm_pose_from_target, hand_pose

    kinematics = rig_kinematics()
    side = hand.value
    centre = np.asarray(item.transform.translation.as_list(), dtype=float)
    neutral = PrimitiveParameters()
    shapes = (HandShape.OPEN, HandShape.FIST)

    def arm_pose(target: np.ndarray, shape) -> dict[str, BonePose]:
        arm, _ = arm_pose_from_target(
            hand,
            shoulder,
            Vec3(x=float(target[0]), y=float(target[1]), z=float(target[2])),
            neutral,
            present_hand=False,
        )
        pose = dict(base_pose)
        pose.update({k: BonePose(rotation=v) for k, v in arm.items()})
        pose.update(
            {k: BonePose(rotation=v) for k, v in hand_pose(hand, shape, neutral).items()}
        )
        if shape is HandShape.OPEN:
            # The approach must be cleared in the pose the hand actually flies,
            # which is the closure's, not ``hand_pose(OPEN)``. Those differ in
            # the thumb -- 0.18 opposition against 0.9 -- and the difference is
            # the whole width of the thumb's swing. Clearing a pose the hand
            # never adopts is not clearing anything: the plan reported the
            # corridor free while the thumb's shafts struck the block.
            from .force_closure import _APPROACH_OPPOSITION, digit_rotations

            pose.update(
                {
                    k: BonePose(rotation=v)
                    for k, v in digit_rotations(
                        hand,
                        {d: 0.02 for d in ("thumb", "index", "middle", "ring", "little")},
                        _APPROACH_OPPOSITION,
                    ).items()
                    if "Thumb" in k
                }
            )
        return pose

    # Each shape's four quads, in the hand bone's frame, measured once. The
    # hand is rigid in that frame, so a candidate costs a rotation of sixteen
    # points rather than a fresh forward-kinematics pass.
    reference_target = centre + np.asarray([-0.10, 0.0, -0.10])
    local: dict[Any, dict[str, np.ndarray]] = {}
    palm_local: dict[Any, np.ndarray] = {}
    for shape in shapes:
        pose = arm_pose(reference_target, shape)
        rotation = kinematics.canonical_world_rotation(pose, f"{side}Hand")
        origin = kinematics.canonical_positions(pose)[f"{side}Hand"]
        local[shape] = {
            name: (quad.corners - origin) @ rotation
            for name, quad in aperture_quads(pose, side).items()
        }
        palm_local[shape] = (_palm_samples(pose, side) - origin) @ rotation

    # Two resolutions. The coarse pass only has to find the right region, and
    # spending the full 75-angle set on every one of 180 positions was most of a
    # 44 s compile. The fine set is paid for once, at the winner.
    coarse_wrist = [
        (p, y, r) for p in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0) for r in (-1.0, 0.0, 1.0)
    ]
    wrist_options = [
        (p, y, r)
        for p in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for y in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for r in (-1.0, 0.0, 1.0)
    ]
    deltas = {
        wrist: Rotation.from_euler(
            "xyz",
            [
                wrist[0] * MAX_WRIST_PITCH_RAD,
                wrist[2] * MAX_WRIST_TWIST_RAD,
                wrist[1] * MAX_WRIST_YAW_RAD,
            ],
        ).as_matrix()
        for wrist in wrist_options
    }

    evaluations = 0

    def measure(shape, world, origin, samples: int):
        held, covered, quads = 0, [], {}
        pierced = 0.0
        seating: list[tuple[float, float]] = []
        for name, corners in local[shape].items():
            placed = corners @ world.T + origin
            quad = ApertureQuad(*[placed[i] for i in range(4)])
            result = overlap(quad, item, resolution=samples)
            covered.append(result["covered_fraction"])
            held += int(result["object_between_tips"])
            # Sampled ALONG the digits, not at their endpoints. A digit is a
            # capsule: the thumb can pass straight through the block with both
            # its base and its tip outside, which an endpoint test scores as
            # clear. Measured on the placement an endpoint test chose: the
            # thumb seated at its opening curl carrying 134 N while all four
            # fingers closed to their limit and reported 0.0 N. It was buried
            # in the block before closure began, and no closure loop can
            # recover an approach that starts inside.
            span = np.linspace(0.0, 1.0, 9)[:, None]
            thumb_line = placed[0] * (1 - span) + placed[1] * span
            finger_line = placed[3] * (1 - span) + placed[2] * span
            # Distance, not an inside/outside test, and with a margin: the
            # physics digits are capsules of 6.5-7.5 mm radius, so a centre-line
            # that merely clears the surface is already in collision. A binary
            # test passed the placement whose thumb then seated at its opening
            # curl carrying 163 N.
            clearance = _box_distance(
                np.vstack([thumb_line, finger_line]), item
            )
            # A DEPTH, not a count. Counting samples below the margin saturates:
            # once every candidate has some sample inside it the term is a
            # constant and the search cannot tell a corridor with 0.91 cm of
            # room from one with 0.5 cm. Measured, that is exactly what
            # happened -- no placement anywhere in the search cleared the
            # margin, so raising the penalty from 0.5 to 3.0 changed the winner
            # not at all, because it was multiplying a constant.
            pierced += float(
                np.sum(np.maximum(0.0, _APPROACH_MARGIN_M - clearance))
            )
            seating.append(
                (
                    _seating(_box_distance(quad.thumb_tip, item)),
                    _seating(_box_distance(quad.finger_tip, item)),
                )
            )
            quads[name] = quad
        palm = palm_local[shape] @ world.T + origin
        # The table was invisible to this search. Measured on the placement it
        # chose, every finger was driven through the tabletop -- the little and
        # ring fingertips reached 9.7 cm BELOW it and stayed in contact for over
        # 400 steps. A hand buried in the surface is fighting it for the whole
        # approach, and those contact forces are a large part of what threw the
        # block. Clearance against the object was checked from the beginning;
        # clearance against the thing the object is sitting on never was.
        below = _obstructed(palm, support_height_m, obstacles)
        for _name, corners in local[shape].items():
            placed_line = corners @ world.T + origin
            span_t = np.linspace(0.0, 1.0, 9)[:, None]
            below += _obstructed(
                np.vstack([
                    placed_line[0] * (1 - span_t) + placed_line[1] * span_t,
                    placed_line[3] * (1 - span_t) + placed_line[2] * span_t,
                ]),
                support_height_m,
                obstacles,
            )
        palm_pierced = float(
            np.sum(np.maximum(0.0, _APPROACH_MARGIN_M - _box_distance(palm, item)))
        )
        thumb_seat = float(np.mean([t for t, _f in seating])) if seating else 0.0
        finger_seat = float(np.mean([f for _t, f in seating])) if seating else 0.0
        return (
            held,
            float(np.mean(covered)),
            quads,
            pierced,
            (thumb_seat, finger_seat),
            palm_pierced,
            below,
        )

    def best_at(offset: np.ndarray, angles=None, samples: int | None = None):
        nonlocal evaluations
        angles = angles if angles is not None else wrist_options
        samples = samples if samples is not None else resolution
        target = centre + offset
        frames = {}
        for shape in shapes:
            pose = arm_pose(target, shape)
            frames[shape] = (
                kinematics.canonical_world_rotation(pose, f"{side}Hand"),
                kinematics.canonical_positions(pose)[f"{side}Hand"],
            )
        found = (-1.0, None, None, (0, 0.0))
        for wrist in angles:
            delta = deltas[wrist]
            evaluations += 1
            open_rotation, open_origin = frames[HandShape.OPEN]
            closed_rotation, closed_origin = frames[HandShape.FIST]
            (
                _open_held, open_covered, _open_quads, open_pierced,
                _open_seat, open_palm, open_below,
            ) = measure(
                HandShape.OPEN, open_rotation @ delta, open_origin, samples
            )
            (
                held, covered, quads, _closed_pierced,
                (thumb_seat, finger_seat), closed_palm, closed_below,
            ) = measure(HandShape.FIST, closed_rotation @ delta, closed_origin, samples)
            # The open hand is the approach pose, so it must be clear. The
            # closed hand is allowed to be inside the object -- that is what
            # gripping it means.
            #
            # Seating is weighted separately from containment because the two
            # come apart, and measurably did: the placement chosen without it
            # reported four opposition pairs while the thumb tip hovered 1.19 cm
            # off the block on every one of them and the ring and little fingers
            # were driven 1.6-2.0 cm through the far face. Every pair was
            # one-sided. Containment asks whether the object lies between the
            # tips; only seating asks whether the tips are on it, which is the
            # difference between a cage and a grip.
            #
            # The FINGERS decide the placement, not the thumb. One wrist cannot
            # seat both: at FIST the four tip spans are 8.6/7.2/5.9/5.9 cm on a
            # 6 cm block, so a wrist moved to bring the thumb in drives the ring
            # and little fingers through the far face, and one moved to spare
            # them leaves the thumb in free air. The thumb does not need the
            # wrist's help -- swept over its own curl and opposition from a
            # fixed wrist it travels from 6.57 cm clear to 0.32 cm buried, which
            # is the whole block and then some. So the wrist places the cage and
            # the thumb travels to it, which is also how a hand does it.
            #
            # The thumb's own seating stays in the score at a low weight as a
            # tiebreak, to prefer placements it reaches comfortably. What keeps
            # the thumb on the far side at all is ``held`` -- containment is
            # measured tip to tip, so a placement with the thumb on the fingers'
            # side scores no pairs.
            score = (
                held
                + covered
                + 0.25 * thumb_seat
                + 1.0 * finger_seat
                + 0.25 * open_covered
                # Weighted to dominate, not to trade. At 0.5 a placement could
                # buy an extra tenth of coverage by giving up the corridor, and
                # it did: the winner left the thumb 1.6 mm of margin against a
                # 2 mm mean tracking error, so the shaft struck the block on the
                # way in and the grasp was decided before it began.
                - 120.0 * open_pierced
                # The palm is checked on both shapes and weighted to dominate.
                # A grasp whose palm is inside the object is not a grasp that
                # squeezes it, it is a shove, and the object is gone before the
                # fingers arrive. No amount of coverage compensates.
                - 200.0 * (open_palm + closed_palm)
                - 200.0 * (open_below + closed_below)
            )
            if score > found[0]:
                found = (score, wrist, quads, (held, covered))
        return found

    best = (-1.0, np.zeros(3), (0.0, 0.0, 0.0), None, (0, 0.0))
    # Symmetric about the object. The old box reached from -0.18 to +0.04 in z
    # and -0.16 to +0.08 in x, which only ever offered placements in front of
    # and inboard of the block -- the one family of approaches whose corridor
    # the thumb crosses. A search that cannot propose coming at it from the far
    # side or from above cannot find the clearance that is missing.
    for dx in np.arange(-0.17, 0.171, coarse_step_m):
        for dy in np.arange(-0.08, 0.161, coarse_step_m):
            for dz in np.arange(-0.18, 0.181, coarse_step_m):
                offset = np.asarray([dx, dy, dz])
                score, wrist, quads, detail = best_at(offset, coarse_wrist, 6)
                if score > best[0]:
                    best = (score, offset, wrist, quads, detail)

    # Re-score the winning region at full angular and sample resolution.
    score, offset, wrist, quads, detail = best_at_full = (
        (lambda r: (r[0], best[1], r[1], r[2], r[3]))(best_at(best[1]))
    )
    for dx in (-refine_step_m, 0.0, refine_step_m):
        for dy in (-refine_step_m, 0.0, refine_step_m):
            for dz in (-refine_step_m, 0.0, refine_step_m):
                candidate = offset + np.asarray([dx, dy, dz])
                value, candidate_wrist, candidate_quads, candidate_detail = best_at(
                    candidate
                )
                if value > score:
                    score, offset, wrist, quads, detail = (
                        value,
                        candidate,
                        candidate_wrist,
                        candidate_quads,
                        candidate_detail,
                    )

    held, covered = detail
    # Closure follows the mean tip-to-tip vector over the four pairs, which is
    # where the thumb should travel to meet the group it opposes.
    directions = [q.closure_direction for q in quads.values()] if quads else []
    mean_direction = np.mean(directions, axis=0) if directions else np.zeros(3)
    norm = float(np.linalg.norm(mean_direction))
    closure = mean_direction / norm if norm > 1e-9 else np.zeros(3)
    span = float(np.mean([q.tip_span_m for q in quads.values()])) if quads else 0.0

    return GraspPlan(
        wrist_target=centre + offset,
        wrist_pitch=wrist[0],
        wrist_yaw=wrist[1],
        wrist_roll=wrist[2],
        covered_fraction=covered,
        closure_direction=closure,
        tip_span_m=span,
        evaluations=evaluations,
        opposition_pairs=held,
    )
