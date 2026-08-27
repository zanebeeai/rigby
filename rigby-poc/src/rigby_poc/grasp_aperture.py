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

#: Intrusion beyond this into scenery is as bad as it needs to get.
_OBSTACLE_TOLERANCE_M = 0.020

#: How far back along its own approach the hand is judged for clearance.
_APPROACH_STANDOFF_M = 0.060


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
    """(see below) Returned as a mean fraction of the tolerance, not a raw depth."""
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
    # Normalised by a tolerance so the result is dimensionless. Summing raw
    # metres made this term reach ~1.0 while coverage tops out at 1 and
    # containment at 4, so at the weights it was given the search stopped
    # optimising the grasp at all and minimised clearance alone: coverage
    # 0.009, no opposition pair, the apertures skimming above the block.
    depths = []
    if support_height_m:
        depths.append(np.maximum(0.0, support_height_m - _height(points)))
    for obstacle in obstacles:
        depths.append(np.maximum(0.0, -_box_distance(points, obstacle)))
    if not depths:
        return 0.0
    worst = np.max(np.vstack(depths), axis=0)
    return float(np.mean(np.minimum(1.0, worst / _OBSTACLE_TOLERANCE_M)))


def _palm_normal(pose: dict[str, BonePose], side: str) -> np.ndarray:
    """The palm's outward normal in world space."""
    from .models import ClipFrame, Hand
    from .physics import _frame_hand_landmarks, _palm_transform

    hand = Hand.LEFT if side == "left" else Hand.RIGHT
    landmarks = _frame_hand_landmarks(
        ClipFrame(time_s=0.0, bones=pose, objects={}), hand
    )
    _centre, rotation = _palm_transform(landmarks, hand)
    return np.asarray(rotation[:, 2], dtype=float)


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


#: Thumb poses considered when squaring the apertures, as (curl, opposition).
#:
#: Opposition is floored rather than swept to zero. Squareness alone picks
#: opposition 0.0 -- the thumb straight out alongside the fingers, which is the
#: squarest aperture and also directly in the corridor the hand travels down.
#: Measured, that placement batted the block 172 cm across the room. The
#: approach sweep put the thumb's clearest opposition near 0.9, so the choice is
#: made among poses that keep the corridor open rather than over all of them.
_THUMB_CANDIDATES = tuple(
    (curl, oppose)
    for curl in (0.15, 0.30, 0.45, 0.60)
    for oppose in (0.6, 0.75, 0.9, 1.05)
)


def square_thumb_pose(base_pose: dict[str, BonePose], hand: Any) -> tuple[float, float]:
    """The thumb pose that stands the apertures most squarely to the palm.

    Chosen once, and separately from the placement, because the two are
    independent: the quads and the palm are both built from the hand, so moving
    the wrist rotates both together and the angle between them does not change.
    Measured directly -- adding an aperture-squareness term to the placement
    score left the winner byte-identical, because no placement in the search
    could affect it.

    The thumb can. Swept over its own curl and opposition the apertures range
    from 16 degrees to the palm to 82, and the authored FIST sits near the
    bottom of that: at its opposition of 0.86 and curl of 0.72 the four
    apertures stand at 19, 19, 25 and 63 degrees. An aperture lying flat in the
    palm is not an opening, and an object "inside" it is really just resting
    against the palm -- which is what the coverage number had been reporting as
    a grasp.
    """
    from .force_closure import digit_rotations
    from .models import HandShape
    from .primitives import HAND_SHAPES

    side = hand.value
    fist = HAND_SHAPES[HandShape.FIST]
    curls = {d: float(fist.curls[d.title()]) for d in ("index", "middle", "ring", "little")}
    best = (-1.0, _THUMB_CANDIDATES[0])
    for curl, oppose in _THUMB_CANDIDATES:
        pose = dict(base_pose)
        for name, rotation in digit_rotations(
            hand, {"thumb": curl, **curls}, oppose
        ).items():
            pose[name] = BonePose(rotation=rotation)
        normal = _palm_normal(pose, side)
        scores = []
        for quad in aperture_quads(pose, side).values():
            corners = quad.corners
            face = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            size = float(np.linalg.norm(face))
            if size < 1e-12:
                continue
            scores.append(1.0 - abs(float(np.dot(face / size, normal))))
        value = float(np.mean(scores)) if scores else 0.0
        if value > best[0]:
            best = (value, (curl, oppose))
    return best[1]


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
    from .primitives import HAND_SHAPES, arm_pose_from_target, hand_pose

    kinematics = rig_kinematics()
    side = hand.value
    centre = np.asarray(item.transform.translation.as_list(), dtype=float)
    neutral = PrimitiveParameters()


    def arm_pose(target: np.ndarray, shape, hand_key=None) -> dict[str, BonePose]:
        arm, _ = arm_pose_from_target(
            hand,
            shoulder,
            Vec3(x=float(target[0]), y=float(target[1]), z=float(target[2])),
            neutral,
            present_hand=False,
        )
        pose = dict(base_pose)
        pose.update({k: BonePose(rotation=v) for k, v in arm.items()})
        shape_params = (
            grip_params[hand_key[1]]
            if hand_key is not None and hand_key[0] == "grip"
            else neutral
        )
        pose.update(
            {
                k: BonePose(rotation=v)
                for k, v in hand_pose(hand, shape, shape_params).items()
            }
        )
        # The closed reference hand stays hand_pose(FIST). Rebuilding it from
        # digit_rotations with the same nominal curls is NOT the same hand --
        # hand_pose applies a 0.75 scale and blends opposition, digit_rotations
        # does neither -- and swapping them moved the chosen placement enough to
        # put the palm back through the table and the block back to being
        # shoved 7.5 cm instead of 0.35.
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
    # NOT APPLIED, and the reason is the thing to fix next.
    #
    # square_thumb_pose picks a thumb that stands the apertures up off the palm,
    # and that part works: swept over its own DOFs the apertures move between 16
    # and 82 degrees to the palm, against the authored FIST's 19/19/25/63. But
    # the compiler hands the frames to close_until_contact, which replaces every
    # digit with its own closure state, so a thumb chosen here never reaches the
    # rendered clip -- measured, the aperture angles came back byte-identical
    # after the plan chose a different thumb entirely.
    #
    # Applying it anyway only destabilised the search: chosen for squareness
    # alone it picks opposition 0.0, the thumb straight out in the approach
    # corridor, and the block was batted 172 cm. Flooring opposition kept the
    # corridor clear but still left the closure overwriting the choice.
    #
    # The fix is to make the closure start from the plan's hand rather than from
    # its own, which is a change to the closure and not to this search.
    thumb_pose = (
        float(HAND_SHAPES[HandShape.FIST].curls["Thumb"]),
        float(HAND_SHAPES[HandShape.FIST].thumb_opposition),
    )
    reference_target = centre + np.asarray([-0.10, 0.0, -0.10])
    # Grip candidates, as per-digit curl displacements. Negative opens a digit
    # relative to the preset, positive closes it. A loose cage suits a block
    # wider than the fist's tip span; a tight one suits something small.
    GRIP_CANDIDATES = (
        ("fist", PrimitiveParameters()),
        ("loose", PrimitiveParameters(
            index_curl=-0.45, middle_curl=-0.45, ring_curl=-0.45, little_curl=-0.45)),
        ("wide", PrimitiveParameters(
            index_curl=-0.7, middle_curl=-0.7, ring_curl=-0.7, little_curl=-0.7,
            finger_splay=0.4)),
        ("deep", PrimitiveParameters(
            index_curl=-0.25, middle_curl=-0.25, ring_curl=-0.35, little_curl=-0.35,
            thumb_curl=0.25)),
    )
    shape_keys = [(HandShape.OPEN, ("open", None))] + [
        (HandShape.FIST, ("grip", name)) for name, _p in GRIP_CANDIDATES
    ]
    grip_params = {name: params for name, params in GRIP_CANDIDATES}
    grip_keys = [("grip", name) for name, _p in GRIP_CANDIDATES]
    open_key = ("open", None)

    local: dict[Any, dict[str, np.ndarray]] = {}
    palm_local: dict[Any, np.ndarray] = {}
    palm_normal_local: dict[Any, np.ndarray] = {}
    # The closed hand is SEARCHED, not assumed.
    #
    # Every version of this planner has scored placements against exactly one
    # closed hand -- hand_pose(FIST) -- and asked only where to put it. But the
    # aperture's whole geometry comes from the hand shape: at FIST the four tip
    # spans are 8.6/7.2/5.9/5.9 cm against a 6 cm block, so some fingers cannot
    # reach it and others are driven through it, and no placement fixes that
    # because placement cannot change a span. Searching one shape's placements
    # very thoroughly was answering a question that had no good answer.
    #
    # Now that hand_pose gives the per-digit parameters full authority, the
    # planner can ask for a hand that fits this object instead of the nearest
    # named one.
    for shape, hand_key in shape_keys:
        pose = arm_pose(reference_target, shape, hand_key)
        rotation = kinematics.canonical_world_rotation(pose, f"{side}Hand")
        origin = kinematics.canonical_positions(pose)[f"{side}Hand"]
        local[hand_key] = {
            name: (quad.corners - origin) @ rotation
            for name, quad in aperture_quads(pose, side).items()
        }
        palm_local[hand_key] = (_palm_samples(pose, side) - origin) @ rotation
        palm_normal_local[hand_key] = _palm_normal(pose, side) @ rotation

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
                np.mean(
                    np.minimum(
                        1.0,
                        np.maximum(0.0, _APPROACH_MARGIN_M - clearance)
                        / _APPROACH_MARGIN_M,
                    )
                )
            )
            seating.append(
                (
                    _seating(_box_distance(quad.thumb_tip, item)),
                    _seating(_box_distance(quad.finger_tip, item)),
                )
            )
            quads[name] = quad
        # The convergence point: where the five tips are trying to meet. If it
        # lands inside the object then the object is in the grasp, and if it
        # does not then no amount of closing will find it. Measured on the
        # placement chosen without it, the point never got inside the block at
        # any moment of the clip -- closest +0.10 cm, and that during the LIFT,
        # after closing was over. At the end of the close it was +1.08 cm out.
        #
        # Unlike the aperture's angle to the palm, this one the placement CAN
        # move: the tips are rigid in the hand frame, so the point travels with
        # the wrist.
        tips = np.vstack([quads[next(iter(quads))].thumb_tip] +
                         [q.finger_tip for q in quads.values()]) if quads else None
        convergence = (
            float(_box_distance(tips.mean(axis=0), item)) if tips is not None else 1.0
        )
        palm = palm_local[shape] @ world.T + origin
        # The table was invisible to this search. Measured on the placement it
        # chose, every finger was driven through the tabletop -- the little and
        # ring fingertips reached 9.7 cm BELOW it and stayed in contact for over
        # 400 steps. A hand buried in the surface is fighting it for the whole
        # approach, and those contact forces are a large part of what threw the
        # block. Clearance against the object was checked from the beginning;
        # clearance against the thing the object is sitting on never was.
        below = _obstructed(palm, support_height_m, obstacles)
        below_samples = 1
        for _name, corners in local[shape].items():
            placed_line = corners @ world.T + origin
            span_t = np.linspace(0.0, 1.0, 9)[:, None]
            below_samples += 1
            below += _obstructed(
                np.vstack([
                    placed_line[0] * (1 - span_t) + placed_line[1] * span_t,
                    placed_line[3] * (1 - span_t) + placed_line[2] * span_t,
                ]),
                support_height_m,
                obstacles,
            )
        palm_pierced = float(
            np.mean(
                np.minimum(
                    1.0,
                    np.maximum(0.0, _APPROACH_MARGIN_M - _box_distance(palm, item))
                    / _APPROACH_MARGIN_M,
                )
            )
        )
        below /= max(below_samples, 1)
        # How square the aperture stands to the palm. An aperture lying flat in
        # the palm is not an opening at all: the object it "contains" is sitting
        # against the palm rather than between the digits, which is what the
        # coverage number was quietly reporting as a grasp. Measured on the
        # placement chosen without this, the four apertures stood at 19, 19, 25
        # and 63 degrees to the palm -- essentially coplanar with it.
        palm_normal = world @ palm_normal_local[shape]
        squareness = []
        for quad in quads.values():
            corners = quad.corners
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            size = float(np.linalg.norm(normal))
            if size < 1e-12:
                continue
            # 1.0 when the aperture plane is perpendicular to the palm.
            squareness.append(
                1.0 - abs(float(np.dot(normal / size, palm_normal)))
            )
        square = float(np.mean(squareness)) if squareness else 0.0
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
            square,
            convergence,
        )

    def best_at(offset: np.ndarray, angles=None, samples: int | None = None):
        nonlocal evaluations
        angles = angles if angles is not None else wrist_options
        samples = samples if samples is not None else resolution
        target = centre + offset
        frames = {}
        for shape, hand_key in shape_keys:
            pose = arm_pose(target, shape, hand_key)
            frames[hand_key] = (
                kinematics.canonical_world_rotation(pose, f"{side}Hand"),
                kinematics.canonical_positions(pose)[f"{side}Hand"],
            )
        # Clearance is a property of the CORRIDOR, not of the destination. At
        # the destination the hand is supposed to be wrapped around the object,
        # so charging it for being near one is charging it for grasping: the
        # winning placement held the palm 9.93 cm away and the nearest fingertip
        # 4.20 cm away, never reaching the block at all, because standing off
        # scored better than closing on it. The open hand is therefore measured
        # backed away along its own approach, at the pose it actually passes
        # through, while coverage and seating stay measured where the hand ends.
        approach = offset / max(float(np.linalg.norm(offset)), 1e-9)
        standoff_pose = arm_pose(target + approach * _APPROACH_STANDOFF_M, HandShape.OPEN)
        approach_frame = (
            kinematics.canonical_world_rotation(standoff_pose, f"{side}Hand"),
            kinematics.canonical_positions(standoff_pose)[f"{side}Hand"],
        )
        found = (-1.0, None, None, (0, 0.0))
        for wrist in angles:
            delta = deltas[wrist]
            evaluations += 1
            open_rotation, open_origin = frames[open_key]
            # Reward the open hand where it ends, but judge its clearance
            # where it travels.
            (
                _open_held, open_covered, _open_quads, _end_pierced,
                _open_seat, _end_palm, _end_below, _end_square, _end_conv,
            ) = measure(open_key, open_rotation @ delta, open_origin, samples)
            approach_rotation, approach_origin = approach_frame
            (
                _a_held, _a_covered, _a_quads, open_pierced,
                _a_seat, open_palm, open_below, _a_square, _a_conv,
            ) = measure(open_key, approach_rotation @ delta, approach_origin, samples)
            grip_best = None
            for grip_key in grip_keys:
                closed_rotation, closed_origin = frames[grip_key]
                candidate = measure(
                    grip_key, closed_rotation @ delta, closed_origin, samples
                )
                # Rank grips the way placements are ranked, so the pair is
                # chosen together rather than the grip being fixed first.
                rank = candidate[0] + candidate[1] - 40.0 * max(0.0, candidate[8])
                if grip_best is None or rank > grip_best[0]:
                    grip_best = (rank, grip_key, candidate)
            (
                held, covered, quads, _closed_pierced,
                (thumb_seat, finger_seat), closed_palm, closed_below, closed_square, convergence,
            ) = grip_best[2]
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
            # Arriving is the PATH; clearing the scenery is MANNER. Talmy's
            # distinction, and a real one here rather than a flourish: a reach
            # is FIGURE=hand, GROUND=object, PATH=TO, and a TO path is defined
            # by terminating at its Ground. Scored as one weighted sum against
            # the clearance terms, not arriving could win -- and it did. The
            # chosen placement held the palm 9.93 cm from the block and the
            # nearest fingertip 4.20 cm away, never reaching it at all, because
            # standing off cost nothing while closing in cost margin.
            #
            # Containment is therefore lexicographic over everything else. A
            # placement that puts the object between no thumb-finger pair is not
            # a worse grasp, it is not a grasp, and no amount of clearance
            # redeems it. Manner ranks only the placements that satisfy the path.
            # The gate is CONVERGENCE, not containment.
            #
            # Both express "the object is in the grasp", but they are not equally
            # strong and the weaker one was rejecting the right answer. ``held``
            # asks whether the object lies between some thumb-finger pair, which
            # a hand can satisfy while its tips are still spread wide around
            # nothing. Convergence asks where the five tips are actually trying
            # to meet, and requires that point to be inside the object.
            #
            # Measured, that difference decided everything: searched for
            # convergence alone the hand reaches 2.91 cm INSIDE the block, but
            # that placement scores ``held == 0`` and was being discarded at
            # -1000 before any of the clearance terms saw it. The best the
            # surviving placements ever managed was +0.08 cm -- outside, and
            # only during the lift, after closing had finished.
            #
            # So convergence gates and containment becomes one term among the
            # rest. A placement whose tips converge outside the object is not a
            # worse grasp, it is a grasp aimed at the wrong place.
            # REVERTED to containment as the gate. Convergence-inside sounds
            # like the stronger test and measured worse: gating on it rejected
            # every candidate the real search produces, and the fallback ranking
            # took the block from 7.63 cm displaced to 13.89.
            #
            # The reason is in the placement the isolated search found. It puts
            # the convergence point 2.91 cm inside the block by putting ALL FIVE
            # FINGERTIPS inside it -- -0.51, -0.64, -0.61, -1.64 and -1.93 cm --
            # with the palm 2.9 cm below the tabletop. That satisfies "the point
            # where the tips meet is within the object" exactly, and it is not a
            # grasp, it is a hand pushed through a box.
            #
            # The criterion needs the tips ON the surface while their meeting
            # point is inside, which is a different statement from either test
            # alone. Left as containment until that is written properly.
            if held == 0:
                score = -1000.0 + covered + 0.25 * open_covered
                if score > found[0]:
                    found = (score, wrist, quads, (held, covered))
                continue
            score = (
                held
                + covered
                + 0.25 * thumb_seat
                + 1.0 * finger_seat
                - 40.0 * max(0.0, convergence)
                # Squareness is MEASURED (see square_thumb_pose) but not scored
                # here. No placement can change it -- the quads and the palm are
                # both built from the hand, so moving the wrist turns both
                # together -- so as a term in the placement score it is close to
                # a constant, and the small variation it does have pulled the
                # winner off a placement that kept the hand out of the table.
                # It belongs on the thumb pose, not on the wrist.
                + 0.25 * open_covered
                # Weighted to dominate, not to trade. At 0.5 a placement could
                # buy an extra tenth of coverage by giving up the corridor, and
                # it did: the winner left the thumb 1.6 mm of margin against a
                # 2 mm mean tracking error, so the shaft struck the block on the
                # way in and the grasp was decided before it began.
                - 3.0 * open_pierced
                # The palm is checked on both shapes and weighted to dominate.
                # A grasp whose palm is inside the object is not a grasp that
                # squeezes it, it is a shove, and the object is gone before the
                # fingers arrive. No amount of coverage compensates.
                - 4.0 * open_palm
                # Scenery clearance is required of the APPROACH, not of the
                # grasp -- the same distinction already made about the object
                # itself, and for the same reason. A hand closing on a block
                # that is standing on a table brings its fingertips down to
                # table level, because that is where the bottom of the block is.
                # Charging the closed hand for that pushed the placement up and
                # off the object: coverage 0.009, the thumb 4.04 cm clear, the
                # apertures skimming above a block they never enclosed.
                - 4.0 * open_below
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
