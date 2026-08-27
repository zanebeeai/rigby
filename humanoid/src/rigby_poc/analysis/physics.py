"""Plan 10 §3.1 — the physics layer (L2).

Built on a **centre of mass** derived from standard segment mass fractions
applied to :meth:`RigKinematics.canonical_positions`. §3.1 calls that "one new
primitive that unlocks the whole layer"; it is one of **two**. Five of the eight
checks §3.1 lists -- balance, foot skate, root consistency, duty factor and
cadence -- need to know **which foot is on the ground in which frame**, and that
does not exist as a general primitive either. Today it lives inside the
full-body compile path as bespoke per-family metrics, which is precisely what
§3.1 says lifting the layer is meant to end: "a new locomotion primitive
inherits them instead of having to reimplement them". Both primitives are here.

**This layer reads frames, never `metrics`.** The same shape as
:func:`~rigby_poc.analysis.anatomy.rom.rom_checks`, and for a reason beyond
symmetry: `MutationSpec.apply` transforms frames only, so a mutated clip carries
the compiler's *pre-mutation* metrics (measured by lane `groundtruth` on four
full-body cases). A metric-derived check scored against a mutated clip is
reading the unmutated one and reports a confident, stable, wrong answer. A
frames-derived check is live under mutation by construction.

**The ground plane is not the origin.** §3.1 specifies ground penetration as "no
collider below y = 0". The source rig's neutral pose stands with its lowest toe
at **y = 0.015201 m**, so an origin datum under-reports every frame of every
clip by 15.2 mm -- in the permissive direction, which for a penetration check
means it under-reports by more than most real penetrations would be. The datum
is :attr:`AnalysisContext.ground_height`, which already derived it correctly and
which `analysis/full_body/ground.py` documents. §3.1 corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics import RigKinematics, rig_kinematics
from ..models import BonePose, ClipFrame
from ..thresholds import value_of
from .contract import PHYSICS, CheckResult, skipped, upper_bound_check
from .rig import rig_profile

#: Segment mass as a fraction of total body mass, and where along the segment
#: that mass acts, as a fraction of segment length from the proximal joint.
#:
#: Source: Winter, *Biomechanics and Motor Control of Human Movement*, 4th ed.,
#: Table 4.1 (Dempster's cadaver regressions). Cited rather than tuned -- these
#: are the numbers the field uses, and inventing rig-specific ones would put an
#: unmeasured judgement inside the instrument.
#:
#: **The trunk mapping is the approximation and it is PROVISIONAL.** Winter
#: segments the trunk as pelvis / abdomen / thorax (0.142 / 0.139 / 0.216,
#: summing to his 0.497 trunk). This rig's spine chain is
#: hips -> spine -> chest -> upperChest, four joints for three segments, so
#: `upperChest` is folded into the thorax segment rather than given mass of its
#: own. A rig whose chest and upperChest are far apart would place thorax mass
#: slightly low. Marked here rather than hidden because the balance checks are
#: sensitive to trunk mass placement and nothing in this repository calibrates it.
SEGMENTS: tuple[tuple[str, str, float, float], ...] = (
    # (proximal bone, distal bone, mass fraction, CoM fraction from proximal)
    ("hips", "spine", 0.142, 0.500),
    ("spine", "chest", 0.139, 0.500),
    ("chest", "neck", 0.216, 0.500),
    ("neck", "head", 0.081, 0.500),
    ("leftUpperArm", "leftLowerArm", 0.028, 0.436),
    ("leftLowerArm", "leftHand", 0.016, 0.430),
    ("rightUpperArm", "rightLowerArm", 0.028, 0.436),
    ("rightLowerArm", "rightHand", 0.016, 0.430),
    ("leftUpperLeg", "leftLowerLeg", 0.100, 0.433),
    ("leftLowerLeg", "leftFoot", 0.0465, 0.433),
    ("leftFoot", "leftToes", 0.0145, 0.500),
    ("rightUpperLeg", "rightLowerLeg", 0.100, 0.433),
    ("rightLowerLeg", "rightFoot", 0.0465, 0.433),
    ("rightFoot", "rightToes", 0.0145, 0.500),
)

#: The hands carry Winter's 0.006 each at the wrist joint, because the rig's
#: hand has no non-finger child to span to. A point mass at the wrist places it
#: ~9 cm proximal of a real hand CoM; at 0.6% of body mass each that is a
#: sub-millimetre shift in whole-body CoM and is not worth a finger traversal.
POINT_MASSES: tuple[tuple[str, float], ...] = (
    ("leftHand", 0.006),
    ("rightHand", 0.006),
)

TOTAL_MASS_FRACTION = sum(m for _p, _d, m, _c in SEGMENTS) + sum(
    m for _b, m in POINT_MASSES
)


class PhysicsError(ValueError):
    """The physics layer was asked for something the clip cannot support."""


def center_of_mass(positions: dict[str, np.ndarray]) -> np.ndarray:
    """Whole-body CoM for one frame's canonical joint positions, in metres.

    ``positions`` is one element of :attr:`AnalysisContext.world_positions`.
    Raises rather than substituting a default when a segment's bone is absent:
    a CoM computed over some of the body is not a CoM, and returning one anyway
    is the absence-becoming-a-value shape this repository keeps re-finding.
    """

    total = np.zeros(3, dtype=float)
    for proximal, distal, mass, along in SEGMENTS:
        try:
            head = np.asarray(positions[proximal], dtype=float)
            tail = np.asarray(positions[distal], dtype=float)
        except KeyError as error:
            raise PhysicsError(
                f"cannot compute a centre of mass: segment "
                f"{proximal}->{distal} is missing {error.args[0]!r} from this "
                "frame's pose"
            ) from error
        total += mass * (head + along * (tail - head))
    for bone, mass in POINT_MASSES:
        try:
            total += mass * np.asarray(positions[bone], dtype=float)
        except KeyError as error:
            raise PhysicsError(
                f"cannot compute a centre of mass: {error.args[0]!r} is missing "
                "from this frame's pose"
            ) from error
    return total / TOTAL_MASS_FRACTION


def center_of_mass_series(
    world_positions: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """``(n, 3)`` CoM track over a clip. Empty input gives an empty array."""

    if not world_positions:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([center_of_mass(p) for p in world_positions], dtype=float)


@dataclass(frozen=True)
class FootContacts:
    """Per-frame ground contact for both feet, and the datum it was decided on.

    ``left`` and ``right`` are boolean arrays over frames. ``clearance_*`` keep
    the metres so a caller can report *how far* rather than only whether -- the
    same reason ``RomViolation`` is a record and not a boolean.
    """

    left: np.ndarray
    right: np.ndarray
    left_clearance_m: np.ndarray
    right_clearance_m: np.ndarray
    threshold_m: float

    @property
    def frames(self) -> int:
        return int(self.left.size)

    @property
    def airborne(self) -> np.ndarray:
        """Frames where neither foot is in contact."""

        return ~(self.left | self.right)

    @property
    def duty_factor(self) -> dict[str, float]:
        """Share of frames each foot spends in contact."""

        if not self.frames:
            return {"left": 0.0, "right": 0.0}
        return {
            "left": float(np.count_nonzero(self.left)) / self.frames,
            "right": float(np.count_nonzero(self.right)) / self.frames,
        }


def foot_contacts(
    world_positions: list[dict[str, np.ndarray]],
    *,
    ground_height: float,
    threshold_m: float,
) -> FootContacts:
    """Which foot is on the ground in which frame.

    Contact is decided on **toe** clearance above ``ground_height``, not on the
    ankle, because the toe is the lowest part of the foot in the rig's neutral
    pose and is what actually meets the floor.

    ``threshold_m`` is a band rather than an equality: a compiled clip's planted
    foot does not sit at exactly the ground height on every frame, and testing
    for zero would report a foot in contact on no frame at all -- a check that
    is structurally incapable of firing.
    """

    if threshold_m <= 0.0:
        raise PhysicsError(
            f"a contact threshold must be a positive band, got {threshold_m}; "
            "testing for exact ground height reports contact on no frame"
        )
    clearances = {}
    for side in ("left", "right"):
        bone = f"{side}Toes"
        missing = [i for i, p in enumerate(world_positions) if bone not in p]
        if missing:
            raise PhysicsError(
                f"cannot decide foot contact: {bone!r} is missing from "
                f"{len(missing)} of {len(world_positions)} frames"
            )
        clearances[side] = np.asarray(
            [float(p[bone][1]) - ground_height for p in world_positions],
            dtype=float,
        )
    return FootContacts(
        left=clearances["left"] <= threshold_m,
        right=clearances["right"] <= threshold_m,
        left_clearance_m=clearances["left"],
        right_clearance_m=clearances["right"],
        threshold_m=threshold_m,
    )


def lowest_joint_height(
    world_positions: list[dict[str, np.ndarray]], *, ground_height: float
) -> np.ndarray:
    """Per-frame height of the lowest canonical joint above ``ground_height``.

    Every canonical bone, not only the toes: a knee driven through the floor is
    a penetration the toes do not see, and it is the case a locomotion primitive
    is most likely to produce.
    """

    if not world_positions:
        return np.zeros(0, dtype=float)
    bones = tuple(rig_profile()["bone_map"])
    return np.asarray(
        [
            min(float(pose[bone][1]) for bone in bones if bone in pose) - ground_height
            for pose in world_positions
        ],
        dtype=float,
    )


def ground_height_of(kinematics: RigKinematics) -> float:
    """The floor for a given skeleton: its lowest neutral-pose toe.

    **Not ``y = 0``.** Measured on this rig it is ``0.015201 m``, so an origin
    datum reads every clip as 15.2 mm further above the floor than it is -- the
    permissive direction for a penetration check, which would sit at zero and
    have that zero mean something about the datum rather than the motion. Plan
    10 §3.1 specified the origin; ``analysis/full_body/ground.py`` and
    ``AnalysisContext.ground_height`` already had it right.

    **This is the single definition of the floor in the repository, and that is
    enforced rather than intended.** ``AnalysisContext.ground_height`` calls it,
    so the two cannot be a copy that agrees on the day it is written and
    silently diverges afterwards -- lane `groundtruth`'s point, and the correct
    one: deliberateness is not a mechanism. It takes the skeleton as an argument
    precisely so the context's injected kinematics still decides.
    """

    neutral = {name: BonePose() for name in rig_profile()["bone_map"]}
    positions = kinematics.canonical_positions(neutral)
    return min(float(positions["leftToes"][1]), float(positions["rightToes"][1]))


def ground_height() -> float:
    """:func:`ground_height_of` for the process-wide rig."""

    return ground_height_of(rig_kinematics())


def world_positions_of(frames: list[ClipFrame]) -> list[dict[str, np.ndarray]]:
    """One forward-kinematics pass over the clip, shared by every physics check.

    ``test_forward_kinematics_is_evaluated_once_per_frame`` pins the invariant
    that the position pass runs once per frame however many checks read it.
    That guard wraps ``analyze()`` rather than ``validate()``, so this layer sits
    outside it -- which is a reason to honour the invariant deliberately here,
    not a licence to ignore it. Every check below takes the result as an
    argument instead of recomputing it.
    """

    kinematics = rig_kinematics()
    return [kinematics.canonical_positions(frame.bones) for frame in frames]


def _horizontal(point: np.ndarray) -> np.ndarray:
    """Ground-plane projection: x and z. y is up in this rig."""

    return np.asarray([point[0], point[2]], dtype=float)


def ground_penetration_check(
    world: list[dict[str, np.ndarray]], *, floor: float
) -> CheckResult:
    """No canonical joint below the floor. Plan 10 §3.1 ``physics.ground.penetration``.

    Reuses ``safety.penetration_max_m`` rather than adding a bound: it is the
    repository's existing penetration tolerance and a second one would be two
    numbers for one question, which is what plan 08 exists to end.
    """

    if not world:
        return skipped(
            "physics.ground.penetration",
            PHYSICS,
            detail="the clip has no frames, so nothing was measured",
        )
    depth = -float(np.min(lowest_joint_height(world, ground_height=floor)))
    return upper_bound_check(
        "physics.ground.penetration",
        PHYSICS,
        max(0.0, depth),
        float(value_of("safety.penetration_max_m")),
        scale=0.05,
        detail="a canonical joint passes below the floor",
    )


#: A foot's contact patch, as ratios of the rig's own measured ankle-to-toe
#: horizontal span (0.1614 m on this rig). Derived from the rig rather than
#: written as metres so the model follows whatever skeleton is loaded.
#:
#: PROVISIONAL. The ratios come from Winter's segment-length anthropometry --
#: foot length is ~0.152 of stature and foot breadth ~0.055 -- reduced to the
#: ankle-to-toe span this rig actually exposes. The rig has **no foot width at
#: all**: both feet are a two-joint chain, so without a modelled patch the
#: support polygon of a single-foot stance degenerates to a line segment and the
#: balance margin is non-positive by construction -- a check unable to pass in
#: the single-support case, which is the case it exists for.
SUPPORT_HEEL_RATIO = 0.40
SUPPORT_HALF_WIDTH_RATIO = 0.29


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull of a small 2-D point set, counter-clockwise."""

    unique = np.unique(np.round(points, 9), axis=0)
    if unique.shape[0] < 3:
        return unique
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    ordered = unique[order]

    def half(seq: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for point in seq:
            while len(out) >= 2:
                first = out[-1] - out[-2]
                second = point - out[-2]
                if first[0] * second[1] - first[1] * second[0] > 0:
                    break
                out.pop()
            out.append(point)
        return out

    lower = half(ordered)
    upper = half(ordered[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _margin_to_hull(hull: np.ndarray, point: np.ndarray) -> float:
    """Signed distance from ``point`` to a convex hull. Positive inside.

    A degenerate hull of fewer than three points cannot contain anything, so the
    margin is the negated distance to it -- zero at best. That is the honest
    answer for a support region with no area, and it is why the footprint above
    is modelled rather than taken as the bare joints.
    """

    if hull.shape[0] == 0:
        return float("-inf")
    if hull.shape[0] < 3:
        far = np.min(np.linalg.norm(hull - point, axis=1))
        return -float(far)
    distances = []
    inside = True
    count = hull.shape[0]
    for index in range(count):
        start = hull[index]
        end = hull[(index + 1) % count]
        edge = end - start
        length = float(np.hypot(edge[0], edge[1]))
        if length <= 1e-12:
            continue
        offset = point - start
        cross = float(edge[0] * offset[1] - edge[1] * offset[0]) / length
        if cross < 0.0:
            inside = False
        projected = float(edge[0] * offset[0] + edge[1] * offset[1]) / (length * length)
        nearest = start + max(0.0, min(1.0, projected)) * edge
        distances.append(float(np.linalg.norm(point - nearest)))
    if not distances:
        return float("-inf")
    closest = min(distances)
    return closest if inside else -closest


def support_polygon(
    pose: dict[str, np.ndarray], *, left: bool, right: bool
) -> np.ndarray:
    """Ground-plane support region for one frame, as a convex hull.

    Each foot in contact contributes a modelled rectangular patch rather than
    its two joints -- see :data:`SUPPORT_HEEL_RATIO`.
    """

    corners: list[np.ndarray] = []
    for side, contacting in (("left", left), ("right", right)):
        if not contacting:
            continue
        ankle = _horizontal(pose[f"{side}Foot"])
        toe = _horizontal(pose[f"{side}Toes"])
        axis = toe - ankle
        length = float(np.hypot(axis[0], axis[1]))
        if length <= 1e-9:
            corners.append(ankle)
            continue
        forward = axis / length
        lateral = np.asarray([-forward[1], forward[0]], dtype=float)
        heel = ankle - SUPPORT_HEEL_RATIO * length * forward
        half = SUPPORT_HALF_WIDTH_RATIO * length
        corners.extend(
            [
                heel + half * lateral,
                heel - half * lateral,
                toe + half * lateral,
                toe - half * lateral,
            ]
        )
    if not corners:
        return np.zeros((0, 2), dtype=float)
    return _convex_hull(np.asarray(corners, dtype=float))


def foot_skate_check(
    world: list[dict[str, np.ndarray]],
    contacts: FootContacts,
    *,
    threshold_m: float,
) -> CheckResult:
    """No horizontal toe motion while that foot is in contact. §3.1.

    Per-frame displacement rather than total travel across the contact: a foot
    that slides 2 mm on each of forty frames and one that jumps 80 mm once are
    different defects, and the sum cannot tell them apart. Same reason
    ``RomViolation`` carries both a peak and an integral.

    This gives ``physics.foot_drift_max_m`` its first real producer. That
    threshold is marked ``UNMEASURED_GATE`` in ``thresholds.v1.json`` and its own
    rationale says the gate is vacuous: ``safety_metrics`` emits the literal 0.0
    for ``foot_drift_m`` and the evidence layer compares that literal against the
    bound, so it has never been able to fail and has therefore never been
    validated.
    """

    if contacts is None or contacts.frames < 2:
        return skipped(
            "physics.contact.foot_skate",
            PHYSICS,
            detail="fewer than two frames, so there is no displacement to measure",
        )
    worst = 0.0
    for side, mask in (("left", contacts.left), ("right", contacts.right)):
        bone = f"{side}Toes"
        for index in range(1, contacts.frames):
            if not (bool(mask[index]) and bool(mask[index - 1])):
                continue
            step = _horizontal(world[index][bone]) - _horizontal(world[index - 1][bone])
            worst = max(worst, float(np.hypot(step[0], step[1])))
    return upper_bound_check(
        "physics.contact.foot_skate",
        PHYSICS,
        worst,
        threshold_m,
        scale=0.02,
        detail="a planted foot slides along the ground between frames",
    )


def physics_checks(frames: list[ClipFrame], *, fps: float) -> list[CheckResult]:
    """Every physics verdict for a clip. Plan 10 §3.1.

    **Always emits every id it owns**, skipping where it measured nothing rather
    than omitting the entry. A check that emits nothing on an inapplicable clip
    is indistinguishable from a check that does not exist, and the mutation
    registry counts emissions by equality, so a vanishing id and a skipping id
    are different facts. Same rule ``rom_checks`` follows with its 156 skips on
    the zero-frame case.

    ``fps`` is accepted and currently unused by both shipped checks -- foot skate
    is a per-frame displacement and penetration is per-frame -- and is on the
    signature because the deferred gait and ballistic checks need it and adding
    it later would change every call site. Stated rather than left to look like
    an oversight.

    **Two of §3.1's eight checks ship here.** The other six are deferred with
    measured reasons, recorded in TRACKING and in plan 10 §3.1; the short form is
    that ``balance`` has the wrong criterion for dynamic motion, ``ballistic``
    and ``momentum`` need a flight phase the corpus has in 5 of 46 cases (and
    those are a ladder climb and a pushup, not flight), and the two ``gait``
    checks need an applicability rule this PR does not build.
    """

    del fps  # see the docstring: kept for the deferred checks' signatures
    floor = ground_height()
    world = world_positions_of(frames)
    contacts = (
        foot_contacts(
            world,
            ground_height=floor,
            threshold_m=float(value_of("physics.foot_contact_band_m")),
        )
        if world
        else None
    )
    return [
        ground_penetration_check(world, floor=floor),
        foot_skate_check(
            world,
            contacts,
            threshold_m=float(value_of("physics.foot_drift_max_m")),
        ),
    ]


__all__ = [
    "POINT_MASSES",
    "SEGMENTS",
    "SUPPORT_HALF_WIDTH_RATIO",
    "SUPPORT_HEEL_RATIO",
    "TOTAL_MASS_FRACTION",
    "FootContacts",
    "PhysicsError",
    "center_of_mass",
    "center_of_mass_series",
    "foot_contacts",
    "foot_skate_check",
    "ground_height",
    "ground_height_of",
    "ground_penetration_check",
    "lowest_joint_height",
    "physics_checks",
    "support_polygon",
    "world_positions_of",
]
