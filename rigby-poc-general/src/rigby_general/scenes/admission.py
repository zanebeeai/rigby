"""Whether a given robot can attempt a given object, and if not, exactly why.

With the object authored in world coordinates rather than derived from the arm,
two questions become answerable that a derived scene cannot even pose. Both are
decided from measurements already taken at ingest, before any simulation runs, so
a robot that cannot attempt something is told so in milliseconds rather than
after a six-second rollout.

The refusals are the interesting output. "This arm reaches 385 mm and the crate
is 1350 mm away" is a fact about a pairing, and it is the thing a derived scene
structurally cannot produce, because it places the object inside the envelope by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import EffectorV1, RobotAssetManifestV1
from ..grounding.workspace import WorkspaceFrame
from .environment import EnvironmentV1, SceneObjectV1


# A jaw has to open wider than the object is thick, or it closes on the outside
# of it. Ten percent of headroom is enough to admit a grasp and tight enough to
# refuse one that would only just fail.
APERTURE_MARGIN = 1.10

# Reach is measured to where the effector site can go. Asking for the very edge
# of that is asking for a fully extended arm at zero manipulability, which will
# not track a descent.
REACH_USABLE_FRACTION = 0.92


@dataclass(frozen=True, slots=True)
class ObjectAdmission:
    object_name: str
    admitted: bool
    reason: str
    code: str | None
    distance_m: float
    reach_limit_m: float
    span_m: float
    aperture_m: float

    def as_dict(self) -> dict:
        return {
            "object": self.object_name,
            "admitted": self.admitted,
            "code": self.code,
            "reason": self.reason,
            "distance_m": round(self.distance_m, 4),
            "reach_limit_m": round(self.reach_limit_m, 4),
            "span_m": round(self.span_m, 4),
            "aperture_m": round(self.aperture_m, 4),
        }


def admit_object(
    manifest: RobotAssetManifestV1,
    effector: EffectorV1,
    frame: WorkspaceFrame,
    item: SceneObjectV1,
) -> ObjectAdmission:
    """Can this gripper, on this arm, attempt this particular object?"""

    aperture = float(effector.max_aperture_m or 0.0)
    span = item.span_m

    target = np.asarray(item.position_m, dtype=float)
    offset = target - frame.origin
    distance = float(np.linalg.norm(offset))
    azimuth, elevation = frame.bearing_of(target)
    reach = frame.directional_reach(azimuth, elevation) * REACH_USABLE_FRACTION
    near = frame.inner_reach(azimuth, elevation)

    if span * APERTURE_MARGIN > aperture:
        return ObjectAdmission(
            item.name,
            False,
            f"the jaw opens {aperture * 1000:.0f} mm and the object is "
            f"{span * 1000:.0f} mm across",
            "object_too_wide",
            distance,
            reach,
            span,
            aperture,
        )
    if distance > reach:
        return ObjectAdmission(
            item.name,
            False,
            f"the object is {distance * 1000:.0f} mm away and this arm reaches "
            f"{reach * 1000:.0f} mm on that bearing",
            "object_out_of_reach",
            distance,
            reach,
            span,
            aperture,
        )
    if distance < near:
        return ObjectAdmission(
            item.name,
            False,
            f"the object is {distance * 1000:.0f} mm away and this arm cannot "
            f"bring its effector closer than {near * 1000:.0f} mm",
            "object_inside_reach_hole",
            distance,
            reach,
            span,
            aperture,
        )
    return ObjectAdmission(
        item.name,
        True,
        f"{span * 1000:.0f} mm across, {distance * 1000:.0f} mm away, within a "
        f"{reach * 1000:.0f} mm reach and a {aperture * 1000:.0f} mm jaw",
        None,
        distance,
        reach,
        span,
        aperture,
    )


def admit_environment(
    manifest: RobotAssetManifestV1,
    effector: EffectorV1,
    frame: WorkspaceFrame,
    environment: EnvironmentV1,
) -> tuple[ObjectAdmission, ...]:
    return tuple(
        admit_object(manifest, effector, frame, item)
        for item in environment.objects
    )
