"""A command as a Talmy motion situation, and the object path it specifies.

Talmy partitions a motion situation into four components (1975, p. 181): the
FIGURE that moves, the GROUND it moves with respect to, the PATH that is the
respect in which it moves, and the MOTION itself -- a closed verb set of MOVE
and BEL, where a located state is the limiting case of motion. MANNER enters
from outside the situation and adjoins the motion verb.

The reason that partition is worth having here is that it inverts the pipeline's
current direction of travel. Today the compiler authors body motion and the
object's trajectory is whatever falls out of the physics; nothing anywhere states
what the object was supposed to do. A motion situation states exactly that, and
:func:`path_contour` turns it into a sequence of world-space waypoints and
tangents -- a contour the Figure must follow.

That contour is a specification, so it is also a loss. :func:`contour_error`
measures how far a compiled clip's actual object trajectory departs from it,
which is the objective a primitive-fitting search would minimise. Nothing here
does the fitting; this module defines what "right" means so that a search has
something to be right about.

What this module deliberately does not do is guess. If a command does not
resolve to a Figure and a Path, it returns ``None`` rather than a default
situation, because a confidently wrong specification is worse than none.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .models import ClipFrame, SceneManifest, SceneObject

CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "talmy_paths.v1.json"
)

MOTION_VERBS = ("MOVE", "BEL")


class TalmyError(ValueError):
    """A motion situation that cannot be built from the catalog."""


@lru_cache(maxsize=1)
def path_catalog() -> dict[str, Any]:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    paths = document["paths"]
    if not paths:
        raise TalmyError("the path catalog is empty")
    for name, spec in paths.items():
        if spec["motion"] not in MOTION_VERBS:
            raise TalmyError(
                f"path {name!r} names motion verb {spec['motion']!r}; "
                f"Talmy's verb set is closed to {MOTION_VERBS}"
            )
        waypoints = spec["contour"]["waypoints"]
        times = [float(w["at"]) for w in waypoints]
        if times != sorted(times) or times[0] != 0.0 or times[-1] != 1.0:
            raise TalmyError(f"path {name!r} has a contour that is not ordered over [0, 1]")
    return document


@dataclass(frozen=True)
class MotionSituation:
    """One command partitioned into Talmy's components.

    ``motion`` is the deep verb; ``manner`` is the component that adjoins it,
    giving Talmy's ``Mm``. ``ground`` is optional because several paths take the
    Figure's own support as their implicit Ground.
    """

    figure: str
    path: str
    motion: str
    ground: str | None = None
    manner: str | None = None
    source_text: str = ""

    def as_notation(self) -> str:
        """Talmy's phrase-marker shorthand: N(F) V(Mm) Pl(P) N(G)."""
        verb = f"{self.motion}{'+' + self.manner if self.manner else ''}"
        ground = f" N({self.ground})" if self.ground else ""
        return f"N({self.figure}) V({verb}) Pl({self.path}){ground}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure": self.figure,
            "ground": self.ground,
            "path": self.path,
            "motion": self.motion,
            "manner": self.manner,
            "notation": self.as_notation(),
            "source_text": self.source_text,
        }


def _match(text: str, aliases: dict[str, list[str]]) -> str | None:
    """Longest alias wins, so 'put down' never resolves as 'put on'."""
    lowered = f" {text.lower().strip()} "
    best: tuple[int, str] | None = None
    for name, words in aliases.items():
        for alias in words:
            if re.search(rf"\b{re.escape(alias.lower())}\b", lowered):
                if best is None or len(alias) > best[0]:
                    best = (len(alias), name)
    return best[1] if best else None


def _resolve_object(text: str, scene: SceneManifest) -> SceneObject | None:
    lowered = text.lower()
    for item in scene.objects:
        if re.search(rf"\b{re.escape(item.id.lower())}\b", lowered):
            return item
        if item.kind and re.search(rf"\b{re.escape(item.kind.lower())}\b", lowered):
            return item
    # "box" is the everyday word for the scene's block; the catalog is a
    # vocabulary of paths, not of nouns, so this stays a scene-level synonym.
    if re.search(r"\bbox\b", lowered):
        return scene.object_by_id("block")
    return None


def interpret(text: str, scene: SceneManifest) -> MotionSituation | None:
    """Partition a command, or return None if it does not resolve."""
    catalog = path_catalog()
    path = _match(text, {k: v["aliases"] for k, v in catalog["paths"].items()})
    if path is None:
        return None
    figure = _resolve_object(text, scene)
    if figure is None:
        return None
    spec = catalog["paths"][path]
    ground: str | None = None
    if spec["requires_ground"]:
        for item in scene.objects:
            if item.id != figure.id and re.search(
                rf"\b{re.escape(item.id.lower())}\b", text.lower()
            ):
                ground = item.id
                break
        if ground is None:
            return None
    manner = _match(text, {k: v["aliases"] for k, v in catalog["manners"].items()})
    return MotionSituation(
        figure=figure.id,
        ground=ground,
        path=path,
        motion=spec["motion"],
        manner=manner,
        source_text=text,
    )


@dataclass
class PathContour:
    """The trajectory the Figure must follow, as points and tangents.

    ``tangents`` are unit vectors, undefined-and-zero where the contour is
    stationary, which is the honest representation of Talmy's located site: a
    site has a position and no direction of travel.
    """

    figure: str
    path: str
    times: np.ndarray
    points: np.ndarray
    tangents: np.ndarray
    frame_origin: np.ndarray
    situation: MotionSituation = field(repr=False)

    def sample(self, progress: float) -> tuple[np.ndarray, np.ndarray]:
        alpha = float(np.clip(progress, 0.0, 1.0))
        index = int(np.searchsorted(self.times, alpha, side="right"))
        upper = min(max(index, 1), len(self.times) - 1)
        lower = upper - 1
        span = max(self.times[upper] - self.times[lower], 1e-12)
        blend = (alpha - self.times[lower]) / span
        point = self.points[lower] * (1 - blend) + self.points[upper] * blend
        tangent = self.tangents[lower] * (1 - blend) + self.tangents[upper] * blend
        norm = float(np.linalg.norm(tangent))
        return point, (tangent / norm if norm > 1e-9 else np.zeros(3))

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure": self.figure,
            "path": self.path,
            "situation": self.situation.to_dict(),
            "waypoints": [
                {
                    "at": float(t),
                    "position": [float(v) for v in p],
                    "tangent": [float(v) for v in d],
                }
                for t, p, d in zip(self.times, self.points, self.tangents)
            ],
        }


def _support_socket(ground: SceneObject):
    """The Ground's own declaration of where it affords support, if it has one.

    ``role == "support"`` first, then any socket that bears weight. Returns
    ``None`` when the object declares nothing, which is the signal to fall back
    to its bounding box.
    """
    sockets = list(getattr(ground, "sockets", ()) or ())
    if not sockets:
        return None
    supports = [s for s in sockets if getattr(s, "role", None) == "support"]
    if supports:
        return max(
            supports, key=lambda s: float(s.transform.translation.as_list()[1])
        )
    bearing = [s for s in sockets if getattr(s, "supports_body_weight", False)]
    if bearing:
        # Highest weight-bearing socket: a ladder's top rung is where you would
        # put something down, not its lowest.
        return max(
            bearing, key=lambda s: float(s.transform.translation.as_list()[1])
        )
    return None


def path_contour(situation: MotionSituation, scene: SceneManifest) -> PathContour:
    """The world-space contour a motion situation specifies for its Figure."""
    catalog = path_catalog()
    spec = catalog["paths"][situation.path]
    figure = scene.object_by_id(situation.figure)
    if figure is None:
        raise TalmyError(f"figure {situation.figure!r} is not in the scene")

    origin = np.asarray(figure.transform.translation.as_list(), dtype=float)
    target = origin
    if situation.ground:
        ground = scene.object_by_id(situation.ground)
        if ground is None:
            raise TalmyError(f"ground {situation.ground!r} is not in the scene")
        base = np.asarray(ground.transform.translation.as_list(), dtype=float)
        if situation.path == "ONTO":
            # Prefer a declared support socket over the bounding box. The box is
            # wrong for anything that is not a plinth: "put the block on the
            # ladder" resolved to 2.14 m, the top of a 2.1 m tall bounding box,
            # rather than onto a rung. Sockets are the scene's own statement of
            # where an object affords being supported.
            socket = _support_socket(ground)
            anchor = base
            if socket is not None:
                anchor = base + np.asarray(
                    socket.transform.translation.as_list(), dtype=float
                )
                target = anchor + np.asarray(
                    [0.0, figure.dimensions_m.y / 2, 0.0]
                )
            else:
                target = base + np.asarray(
                    [0.0, ground.dimensions_m.y / 2 + figure.dimensions_m.y / 2, 0.0]
                )
        else:
            target = base

    waypoints = spec["contour"]["waypoints"]
    times = np.asarray([float(w["at"]) for w in waypoints], dtype=float)
    points = np.zeros((len(waypoints), 3), dtype=float)
    for index, waypoint in enumerate(waypoints):
        offset = np.asarray(waypoint["offset"], dtype=float)
        blend = times[index] if situation.ground else 0.0
        anchor = origin * (1.0 - blend) + target * blend
        points[index] = anchor + offset

    tangents = np.zeros_like(points)
    for index in range(len(points)):
        lower = max(0, index - 1)
        upper = min(len(points) - 1, index + 1)
        delta = points[upper] - points[lower]
        norm = float(np.linalg.norm(delta))
        tangents[index] = delta / norm if norm > 1e-9 else np.zeros(3)

    return PathContour(
        figure=situation.figure,
        path=situation.path,
        times=times,
        points=points,
        tangents=tangents,
        frame_origin=origin,
        situation=situation,
    )


def contour_error(
    contour: PathContour,
    frames: list[ClipFrame],
) -> dict[str, Any]:
    """How far the Figure's actual trajectory departed from its specification.

    This is the objective a primitive-fitting search minimises. It is reported
    as a peak and a mean rather than a single number because the two fail
    differently: a large mean is a trajectory with the wrong shape, while a
    large peak with a small mean is a trajectory that is right except at the one
    moment that matters.
    """
    if not frames:
        return {}
    samples = [f for f in frames if contour.figure in f.objects]
    if not samples:
        return {}
    duration = max(samples[-1].time_s, 1e-9)
    errors: list[float] = []
    endpoint_error = 0.0
    for frame in samples:
        actual = np.asarray(frame.objects[contour.figure].translation.as_list())
        expected, _tangent = contour.sample(frame.time_s / duration)
        errors.append(float(np.linalg.norm(actual - expected)))
    endpoint_error = errors[-1]
    return {
        "talmy_path": contour.path,
        "talmy_notation": contour.situation.as_notation(),
        "contour_sample_count": len(errors),
        "contour_mean_error_m": float(np.mean(errors)),
        "contour_peak_error_m": float(np.max(errors)),
        "contour_endpoint_error_m": float(endpoint_error),
        "contour_specified_displacement_m": float(
            np.linalg.norm(contour.points[-1] - contour.points[0])
        ),
        "contour_actual_displacement_m": float(
            np.linalg.norm(
                np.asarray(samples[-1].objects[contour.figure].translation.as_list())
                - np.asarray(samples[0].objects[contour.figure].translation.as_list())
            )
        ),
    }
