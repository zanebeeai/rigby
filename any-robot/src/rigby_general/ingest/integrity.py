"""Refuse models that are unsafe to load or impossible to simulate honestly.

Every check here mirrors a rule the v2 firewall already applies to assets
(``asset.path.external.v1``, ``asset.plugin.native.v1``,
``asset.inertial.finite-positive.v1``, ``asset.dynamic-collider.convex-only.v1``),
restated for a model that arrives from outside rather than from the repository.

The security checks run on the *source text*, before MuJoCo parses it. Once the
parser has run it is too late: a plugin directive has already been resolved and
an absolute mesh path has already been read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from xml.etree import ElementTree as ET

import mujoco

from ..errors import GeneralFailureCode, ModelIngestError
from .loader import LoadedModel, body_names, geom_names


class IntegrityRule(StrEnum):
    EXTERNAL_PATH = "asset.path.external.v1"
    NATIVE_PLUGIN = "asset.plugin.native.v1"
    FINITE_POSITIVE_INERTIA = "asset.inertial.finite-positive.v1"
    CONVEX_DYNAMIC_COLLIDER = "asset.dynamic-collider.convex-only.v1"
    HAS_COLLIDER = "asset.collider.present.v1"
    FIXED_BASE = "asset.base.fixed.v1"
    REST_POSE_PENETRATION = "asset.rest.non-penetrating.v1"


@dataclass(frozen=True, slots=True)
class IntegrityViolation:
    rule: IntegrityRule
    detail: str
    subject: str


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    violations: tuple[IntegrityViolation, ...]
    body_count: int
    actuated_joint_count: int
    collider_count: int
    total_mass_kg: float
    notes: tuple[IntegrityViolation, ...] = ()
    """Defects recorded rather than refused. Presently only collision
    hulls that overlap in every configuration; see check_rest_contacts."""

    @property
    def passed(self) -> bool:
        return not self.violations

    def raise_for_status(self, robot_id: str) -> None:
        raise_for_violations(robot_id, self.violations)


def raise_for_violations(
    robot_id: str, violations: tuple[IntegrityViolation, ...]
) -> None:
    if not violations:
        return
    first = violations[0]
    raise ModelIngestError(
        _RULE_FAILURE[first.rule],
        f"{robot_id}: {first.detail}",
        details={
            "rule": first.rule.value,
            "subject": first.subject,
            "violations": [
                {
                    "rule": violation.rule.value,
                    "subject": violation.subject,
                    "detail": violation.detail,
                }
                for violation in violations
            ],
        },
    )


_RULE_FAILURE: dict[IntegrityRule, GeneralFailureCode] = {
    IntegrityRule.EXTERNAL_PATH: GeneralFailureCode.UNSAFE_ASSET,
    IntegrityRule.NATIVE_PLUGIN: GeneralFailureCode.UNSAFE_ASSET,
    IntegrityRule.CONVEX_DYNAMIC_COLLIDER: GeneralFailureCode.UNSAFE_ASSET,
    IntegrityRule.FINITE_POSITIVE_INERTIA: GeneralFailureCode.DEGENERATE_INERTIA,
    IntegrityRule.HAS_COLLIDER: GeneralFailureCode.UNREADABLE_MODEL,
    IntegrityRule.FIXED_BASE: GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
    IntegrityRule.REST_POSE_PENETRATION: GeneralFailureCode.UNSAFE_ASSET,
}

# Attributes that can name a file on disk, across both URDF and MJCF.
_PATH_ATTRIBUTES = ("filename", "file", "meshdir", "texturedir", "assetdir")

_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _is_external(reference: str) -> bool:
    """True when a path escapes the upload directory or names a remote resource.

    ``package://`` is included: it resolves through a ROS workspace outside the
    upload, so it cannot be honoured here even though it is idiomatic in URDF.
    """

    candidate = reference.strip()
    if not candidate:
        return False
    if _URI_SCHEME.match(candidate):
        return True
    posix = PurePosixPath(candidate)
    windows = PureWindowsPath(candidate)
    if posix.is_absolute() or windows.is_absolute():
        return True
    if candidate.startswith("\\\\"):
        return True
    return ".." in posix.parts or ".." in windows.parts


_INERTIA_TRIANGLE_TOLERANCE = 1e-9


def _urdf_inertia_violations(root: ET.Element) -> list[IntegrityViolation]:
    """Principal moments must satisfy the triangle inequality A + B >= C.

    A rigid body's inertia tensor is not three free numbers. No mass
    distribution produces principal moments where one exceeds the sum of the
    other two, so a tensor that does describes no object -- simulating it gives
    torques that no real arm would feel, and every primitive certified against
    it is certified against a fiction.

    MuJoCo agrees and refuses to compile such a model. It has a
    ``balanceinertia`` flag that quietly rewrites the tensor to fit; the loader
    leaves that off on purpose, since repairing a defect in place hides it while
    changing the dynamics. The check is here, on the source, so the refusal can
    name the link and the size of the violation instead of surfacing as a parse
    error.
    """

    import numpy as np

    violations: list[IntegrityViolation] = []
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia")
        if inertia is None:
            continue

        def moment(key: str) -> float:
            try:
                return float(inertia.get(key, "0"))
            except ValueError:
                return math.nan

        tensor = np.array(
            [
                [moment("ixx"), moment("ixy"), moment("ixz")],
                [moment("ixy"), moment("iyy"), moment("iyz")],
                [moment("ixz"), moment("iyz"), moment("izz")],
            ]
        )
        name = link.get("name") or "?"
        if not np.all(np.isfinite(tensor)):
            violations.append(
                IntegrityViolation(
                    IntegrityRule.FINITE_POSITIVE_INERTIA,
                    f"link {name!r} has a non-finite inertia tensor",
                    subject=name,
                )
            )
            continue
        if not np.any(tensor):
            continue

        first, second, third = sorted(float(v) for v in np.linalg.eigvalsh(tensor))
        if third <= 0.0:
            continue
        if first + second < third * (1.0 - _INERTIA_TRIANGLE_TOLERANCE):
            shortfall = (third - (first + second)) / third
            violations.append(
                IntegrityViolation(
                    IntegrityRule.FINITE_POSITIVE_INERTIA,
                    f"link {name!r} has principal moments "
                    f"({first:.6g}, {second:.6g}, {third:.6g}) that violate "
                    f"A + B >= C by {shortfall * 100.0:.1f}% -- no rigid body has "
                    "this inertia, so anything certified against it would be "
                    "certified against dynamics that cannot occur",
                    subject=name,
                )
            )
    return violations


def scan_source_text(source: bytes) -> tuple[IntegrityViolation, ...]:
    """Security scan of the raw upload, before any parser touches it."""

    violations: list[IntegrityViolation] = []
    try:
        root = ET.fromstring(source)
    except ET.ParseError as error:
        raise ModelIngestError(
            GeneralFailureCode.UNREADABLE_MODEL,
            f"The model is not well-formed XML: {error}",
            details={"line": getattr(error, "lineno", None)},
        ) from error

    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in ("plugin", "extension") or element.get("plugin"):
            violations.append(
                IntegrityViolation(
                    IntegrityRule.NATIVE_PLUGIN,
                    "The model requests a native plugin, which cannot be loaded "
                    "from an untrusted upload",
                    subject=tag,
                )
            )
        for attribute in _PATH_ATTRIBUTES:
            reference = element.get(attribute)
            if reference and _is_external(reference):
                violations.append(
                    IntegrityViolation(
                        IntegrityRule.EXTERNAL_PATH,
                        f"{tag}/@{attribute} points outside the upload: {reference!r}",
                        subject=reference,
                    )
                )

    if root.tag.rsplit("}", 1)[-1] == "robot":
        violations.extend(_urdf_inertia_violations(root))
    return tuple(violations)


def _inertia_violations(model: mujoco.MjModel) -> list[IntegrityViolation]:
    violations: list[IntegrityViolation] = []
    names = body_names(model)
    for index in range(1, model.nbody):
        name = names[index]
        mass = float(model.body_mass[index])
        inertia = model.body_inertia[index]

        if model.body_dofnum[index] == 0:
            # A welded body carries no state; a zero-mass static frame is a
            # legitimate way to write a mount point, not a defect.
            continue

        if not math.isfinite(mass) or mass <= 0.0:
            violations.append(
                IntegrityViolation(
                    IntegrityRule.FINITE_POSITIVE_INERTIA,
                    f"body {name!r} has non-positive or non-finite mass {mass!r}",
                    subject=name,
                )
            )
            continue

        values = [float(value) for value in inertia]
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            violations.append(
                IntegrityViolation(
                    IntegrityRule.FINITE_POSITIVE_INERTIA,
                    f"body {name!r} has non-positive or non-finite inertia {values!r}",
                    subject=name,
                )
            )
            continue

        # The inertia tensor of any real solid satisfies the triangle
        # inequality. A violation means the numbers were invented, and no
        # simulation of them describes an object that could exist.
        a, b, c = sorted(values)
        if a + b < c * (1.0 - 1e-6):
            violations.append(
                IntegrityViolation(
                    IntegrityRule.FINITE_POSITIVE_INERTIA,
                    f"body {name!r} violates the inertia triangle inequality: {values!r}",
                    subject=name,
                )
            )
    return violations


def _collider_violations(model: mujoco.MjModel) -> list[IntegrityViolation]:
    violations: list[IntegrityViolation] = []
    names = geom_names(model)
    dynamic_colliders = 0

    for index in range(model.ngeom):
        if model.geom_contype[index] == 0 and model.geom_conaffinity[index] == 0:
            continue
        body = int(model.geom_bodyid[index])
        if model.body_dofnum[body] > 0:
            dynamic_colliders += 1
        if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[index])
        if mesh_id < 0:
            continue
        # graphadr < 0 means the compiler stored no convex hull for the mesh, so
        # contacts against it would be resolved against raw triangles.
        if int(model.mesh_graphadr[mesh_id]) < 0 and model.body_dofnum[body] > 0:
            violations.append(
                IntegrityViolation(
                    IntegrityRule.CONVEX_DYNAMIC_COLLIDER,
                    f"geom {names[index]!r} collides using a mesh with no convex hull",
                    subject=names[index],
                )
            )

    if dynamic_colliders == 0:
        violations.append(
            IntegrityViolation(
                IntegrityRule.HAS_COLLIDER,
                "No moving body carries a collision geom, so no contact could "
                "ever be measured",
                subject="model",
            )
        )
    return violations


def _base_violations(model: mujoco.MjModel) -> list[IntegrityViolation]:
    for index in range(model.njnt):
        if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE:
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                or f"joint_{index}"
            )
            return [
                IntegrityViolation(
                    IntegrityRule.FIXED_BASE,
                    f"joint {name!r} is a free joint; this package certifies "
                    "fixed-base robots only and has no balance controller",
                    subject=name,
                )
            ]
    return []


# Any real contact counts, not only a deep one. MuJoCo reports a contact when
# surfaces are within the collision margin, so a pair merely resting against
# each other shows a distance near zero rather than a large negative one -- and
# that is exactly the case that jams a joint at full torque while looking
# harmless in the numbers.
# Any real contact counts, not only a deep one. MuJoCo reports a contact when
# surfaces are within the collision margin, so a pair merely resting against each
# other shows a distance near zero rather than a large negative one -- and that is
# exactly the case that jams a joint at full torque while looking harmless.
_REST_CONTACT_TOLERANCE_M = 0.0


SEPARABILITY_SAMPLES = 9


def _bodies_touch(
    model: mujoco.MjModel, data: mujoco.MjData, first: int, second: int
) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        if float(contact.dist) > _REST_CONTACT_TOLERANCE_M:
            continue
        pair = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        if pair == {first, second}:
            return True
    return False


def _joints_between(model: mujoco.MjModel, first: int, second: int) -> tuple[int, ...]:
    """Every joint on the tree path linking two bodies.

    These are exactly the joints that can change the two bodies' relative pose;
    no other joint in the model can.
    """

    def ancestry(body: int) -> list[int]:
        chain, current = [], body
        while current > 0:
            chain.append(current)
            current = int(model.body_parentid[current])
        return chain

    first_line, second_line = ancestry(first), ancestry(second)
    shared = set(first_line) & set(second_line)
    path = [body for body in first_line + second_line if body not in shared]

    joints: list[int] = []
    for body in path:
        for offset in range(int(model.body_jntnum[body])):
            joint = int(model.body_jntadr[body]) + offset
            if model.jnt_type[joint] in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                joints.append(joint)
    return tuple(joints)


def _separable(model: mujoco.MjModel, rest, first: int, second: int) -> bool:
    """Can any joint motion pull these two links apart?

    This is the line between two defects that look identical at rest and are not
    remotely the same problem. An arm slumped against its own torso is a
    mechanism fighting itself: rotate the shoulder and the contact is gone, and a
    model that starts there burns its whole actuator budget pushing out of it.
    Two wrist links whose convex collision hulls overlap near the joint are stuck
    together at *every* configuration -- that is coarse hull authoring, ordinary
    in real robot descriptions, and no controller can fix it.

    Refusing both meant refusing real arms for a modelling habit. So: separable
    is refused, inseparable is recorded.
    """

    import numpy as np

    data = mujoco.MjData(model)
    base = np.asarray(rest, dtype=float)
    for joint in _joints_between(model, first, second):
        if model.jnt_limited[joint]:
            low, high = (float(value) for value in model.jnt_range[joint])
        else:
            low, high = -math.pi, math.pi
        address = int(model.jnt_qposadr[joint])
        for step in range(SEPARABILITY_SAMPLES):
            data.qpos[:] = base
            data.qpos[address] = low + (high - low) * step / (SEPARABILITY_SAMPLES - 1)
            mujoco.mj_forward(model, data)
            if not _bodies_touch(model, data, first, second):
                return True
    return False


def check_rest_contacts(
    model: mujoco.MjModel, morphology
) -> tuple[tuple[IntegrityViolation, ...], tuple[IntegrityViolation, ...]]:
    """Refuse links pressed together at rest, with two exceptions.

    Returns ``(refused, recorded)``.

    Deliberately run *after* morphology rather than during the load, and the
    reason is a real robot the earlier version rejected. Running it at load time
    meant it had no idea which bodies belong to a gripper -- and a gripper parked
    closed has its fingertips overlapping by design. The PR2 gripper's tips
    overlap by 10 mm at rest, which is not a broken model, it is a closed hand.

    The second exception is geometric rather than semantic, and cost a second
    real arm: see :func:`_separable`. A pair no joint motion can part is
    recorded, not refused.

    What remains genuinely wrong is a contact the mechanism must *fight*: an arm
    resting against a torso, where the shoulder saturates at full torque and
    never moves.
    """

    from ..morphology.measure import neutral_qpos

    members: set[frozenset[str]] = set()
    for effector in morphology.effectors:
        bodies = set(effector.member_bodies)
        for first in bodies:
            for second in bodies:
                if first != second:
                    members.add(frozenset((first, second)))

    rest = neutral_qpos(model)
    data = mujoco.MjData(model)
    # The rest pose everything downstream uses, not the raw zero pose --
    # otherwise this reports a penetration the relaxation already resolved.
    data.qpos[:] = rest
    mujoco.mj_forward(model, data)

    adjacency = {
        (
            min(body, int(model.body_parentid[body])),
            max(body, int(model.body_parentid[body])),
        )
        for body in range(1, model.nbody)
    }

    refused: list[IntegrityViolation] = []
    recorded: list[IntegrityViolation] = []
    seen: set[tuple[str, str]] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        depth = float(contact.dist)
        if depth > _REST_CONTACT_TOLERANCE_M:
            continue
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if first == second or (min(first, second), max(first, second)) in adjacency:
            continue
        names = tuple(
            sorted(
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or "?",
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or "?",
                )
            )
        )
        if names in seen or frozenset(names) in members:
            continue
        seen.add(names)

        if _separable(model, rest, first, second):
            refused.append(
                IntegrityViolation(
                    IntegrityRule.REST_POSE_PENETRATION,
                    f"{names[0]!r} and {names[1]!r} are already in contact in the "
                    f"rest pose (separation {depth * 1000.0:.2f} mm), and joint "
                    "motion can part them, so the mechanism would have to fight "
                    "its way out",
                    subject=f"{names[0]}|{names[1]}",
                )
            )
        else:
            recorded.append(
                IntegrityViolation(
                    IntegrityRule.REST_POSE_PENETRATION,
                    f"{names[0]!r} and {names[1]!r} overlap by "
                    f"{-depth * 1000.0:.2f} mm in every configuration, so their "
                    "collision hulls are authored intersecting; recorded rather "
                    "than refused",
                    subject=f"{names[0]}|{names[1]}",
                )
            )
    return tuple(refused), tuple(recorded)


def check_integrity(loaded: LoadedModel) -> IntegrityReport:
    """Run every rule and report all violations, not just the first."""

    violations: list[IntegrityViolation] = list(scan_source_text(loaded.source_bytes))
    model = loaded.model
    violations.extend(_base_violations(model))
    violations.extend(_inertia_violations(model))
    violations.extend(_collider_violations(model))

    actuated = sum(
        1
        for index in range(model.njnt)
        if model.jnt_type[index]
        in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    )
    colliders = sum(
        1
        for index in range(model.ngeom)
        if model.geom_contype[index] or model.geom_conaffinity[index]
    )

    return IntegrityReport(
        violations=tuple(violations),
        body_count=int(model.nbody) - 1,
        actuated_joint_count=actuated,
        collider_count=colliders,
        total_mass_kg=float(model.body_mass.sum()),
    )
