from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .evidence import lookup
from .models import GateResult, Status


def rig_forward_vector(profile: dict[str, Any]) -> tuple[float, float, float]:
    raw = str(profile.get("forward_axis", "")).strip().upper()
    signs = {"X": (1.0, 0.0, 0.0), "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
             "Y": (0.0, 1.0, 0.0), "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
             "Z": (0.0, 0.0, 1.0), "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0)}
    if raw not in signs:
        raise ValueError(f"unsupported rig forward axis {raw!r}")
    return signs[raw]


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict) and all(isinstance(value.get(key), (int, float)) for key in ("x", "y", "z")):
        return float(value["x"]), float(value["y"]), float(value["z"])
    if isinstance(value, list) and len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    return None


def _projection(point: tuple[float, float, float], origin: tuple[float, float, float], forward: tuple[float, float, float]) -> float:
    return sum((point[index] - origin[index]) * forward[index] for index in range(3))


def _head_origin(profile: dict[str, Any]) -> tuple[float, float, float]:
    pivots = profile.get("coordinate_calibration", {}).get("rest_world_pivots_m", {})
    return _vec3(pivots.get("head")) or _vec3(pivots.get("upperChest")) or (0.0, 0.0, 0.0)


def _shoulder_origin(profile: dict[str, Any], hand: str) -> tuple[float, float, float] | None:
    pivots = profile.get("coordinate_calibration", {}).get("rest_world_pivots_m", {})
    key = "leftUpperArm" if hand == "left" else "rightUpperArm"
    return _vec3(pivots.get(key))


def _arm_reach(profile: dict[str, Any]) -> float | None:
    lengths = profile.get("coordinate_calibration", {}).get("arm_lengths_m", {})
    upper, lower = lengths.get("upper"), lengths.get("lower")
    if isinstance(upper, (int, float)) and isinstance(lower, (int, float)):
        return float(upper) + float(lower)
    return None


def semantic_forward_gate(items: list[dict[str, Any]], profile: dict[str, Any], criteria: dict[str, Any]) -> GateResult:
    forward = rig_forward_vector(profile)
    origin = _head_origin(profile)
    violations: list[str] = []
    missing: list[str] = []
    projections: dict[str, float] = {}
    reach_distances: dict[str, float] = {}
    minimum = float(criteria["min_forward_projection_m"])
    max_reach = _arm_reach(profile)
    reach_tolerance = float(criteria.get("reach_tolerance_m", 0.0))
    for item in items:
        item_id = str(item.get("id", "unknown"))
        program = item.get("program", {})
        intent = lookup(program, ["intent"])
        response = item.get("response")
        body = getattr(response, "body", None) if response is not None else item.get("clip", {})
        observables = lookup(body or {}, ["slider_observables", "observables"])
        names = criteria["gesture_target_observables"]
        wrist_target = None
        if isinstance(observables, dict) and all(isinstance(observables.get(name), (int, float)) for name in names):
            wrist_target = tuple(float(observables[name]) for name in names)
        hand = str(lookup(program, ["hand", "handedness"]) or "right").lower()
        shoulder = _shoulder_origin(profile, hand)
        if wrist_target is None or shoulder is None or max_reach is None:
            missing.append(f"{item_id}: selected-hand wrist target or calibrated arm reach is missing")
        else:
            reach = math.sqrt(sum((wrist_target[index] - shoulder[index]) ** 2 for index in range(3)))
            reach_distances[item_id] = reach
            if reach > max_reach + reach_tolerance:
                violations.append(
                    f"{item_id}: wrist target is {reach:.3f} m from {hand} shoulder; arm reach is {max_reach:.3f} m"
                )
        if intent == "gesture":
            if wrist_target is None:
                continue
            projection = _projection(wrist_target, origin, forward)
            projections[item_id] = projection
            if projection < minimum:
                violations.append(f"{item_id}: gesture target projects {projection:.3f} m along rig forward")
        elif intent in {"grab", "grasp"}:
            object_id = lookup(program, ["object_id", "target_object_id"]) or "block"
            scene_objects = item.get("scene", {}).get("objects", [])
            target = next((value for value in scene_objects if value.get("id") == object_id), None)
            scene_point = _vec3(target.get("transform", {}).get("translation")) if isinstance(target, dict) else None
            frame_points = []
            for frame in item.get("clip", {}).get("frames", []):
                transform = frame.get("objects", {}).get(object_id) if isinstance(frame, dict) else None
                point = _vec3(transform.get("translation")) if isinstance(transform, dict) else None
                if point is not None:
                    frame_points.append(point)
            points = ([scene_point] if scene_point is not None else []) + frame_points
            if not points:
                missing.append(f"{item_id}: pickup object trajectory is missing")
                continue
            projection = min(_projection(point, origin, forward) for point in points)
            projections[item_id] = projection
            if projection < minimum:
                violations.append(f"{item_id}: pickup trajectory enters {projection:.3f} m along rig forward")
    expected = len(items)
    fraction = min(len(projections), len(reach_distances)) / expected if expected else 0.0
    if violations:
        status = Status.FAIL
    elif missing or fraction < criteria["required_fraction"]:
        status = Status.UNVERIFIED
    else:
        status = Status.PASS
    return GateResult(
        "semantic_forward_space",
        status,
        f"{min(len(projections), len(reach_distances))}/{expected} clips were both in the rig's forward half-space and within calibrated arm reach",
        {
            "rig_forward": list(forward), "head_origin_m": list(origin), "projections_m": projections,
            "reach_distances_m": reach_distances, "max_arm_reach_m": max_reach,
            "verified_fraction": fraction,
        },
        criteria,
        failures=(violations + missing)[:60],
    )


def _declared_neutral_gaze() -> list[float] | None:
    """The neutral gaze from `config/camera.v1.json`, the single source since 08c."""
    config = Path(__file__).resolve().parents[1] / "config" / "camera.v1.json"
    if not config.is_file():
        return None
    try:
        document = json.loads(config.read_text(encoding="utf-8"))
        value = document["camera"]["ego_neutral_gaze"]["value"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return [float(component) for component in value] if len(value) == 3 else None


def inspect_egocentric_camera_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    start = source.find('if (this.mode === "ego")')
    end = source.find("this.renderer.render", start)
    section = source[start:end] if start >= 0 and end > start else ""
    vectors: list[list[float]] = []
    for match in re.finditer(r"add\(new THREE\.Vector3\(([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)\)\)", section):
        vectors.append([float(match.group(index)) for index in (1, 2, 3)])
    positions = [tuple(float(match.group(index)) for index in (1, 2, 3)) for match in re.finditer(
        r"egoCamera\.position\.set\(([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)\)", section
    )]
    targets = [tuple(float(match.group(index)) for index in (1, 2, 3)) for match in re.finditer(
        r"egoCamera\.lookAt\(([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)\)", section
    )]
    vectors.extend([[target[i] - position[i] for i in range(3)] for position, target in zip(positions, targets)])
    sources = {path.name: source}
    imported = re.search(r'import\s+\{[^}]*computeEgoCameraPose[^}]*\}\s+from\s+["\'](\./[^"\']+)["\']', source)
    if imported:
        camera_path = path.parent / f"{imported.group(1).removeprefix('./')}.ts"
        if camera_path.is_file():
            camera_source = camera_path.read_text(encoding="utf-8")
            sources[camera_path.name] = camera_source
            neutral = re.search(
                r"NEUTRAL_GAZE\s*=\s*new THREE\.Vector3\(([-+0-9.]+),\s*([-+0-9.]+),\s*([-+0-9.]+)\)",
                camera_source,
            )
            if neutral:
                vectors.insert(0, [float(neutral.group(index)) for index in (1, 2, 3)])
            elif re.search(r"NEUTRAL_GAZE\s*=\s*new THREE\.Vector3\(\.\.\.egoNeutralGaze\)", camera_source):
                # Plan 08 §3.3 replaced the hand-written literal with a generated import,
                # which is the point of that PR and which silently emptied this gate: the
                # regex above finds numeric literals, and there are no longer any to find.
                # A source inspector reading constants out of code is defeated by exactly
                # the refactor that centralises those constants, and it fails *open* --
                # one branch instead of two, reported as unverified rather than as wrong.
                #
                # Following the import to config/camera.v1.json rather than parsing the
                # generated TypeScript: the generator's --check job already proves the two
                # agree on every CI run, so the JSON is the same fact with less parsing.
                declared = _declared_neutral_gaze()
                if declared is not None:
                    vectors.insert(0, declared)
    source_digest = hashlib.sha256(
        "\n".join(f"{name}\n{sources[name]}" for name in sorted(sources)).encode("utf-8")
    ).hexdigest()
    return {"vectors": vectors, "source_sha256": source_digest, "sources": sorted(sources)}


def camera_orientation_gate(evidence: dict[str, Any], profile: dict[str, Any], criteria: dict[str, Any]) -> GateResult:
    forward = rig_forward_vector(profile)
    vectors = evidence.get("vectors", [])
    dots: list[float] = []
    failures: list[str] = []
    threshold = math.cos(math.radians(float(criteria["max_angle_from_rig_forward_deg"])))
    for index, raw in enumerate(vectors):
        vector = _vec3(raw)
        if vector is None:
            continue
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            failures.append(f"ego camera branch {index} has a zero view vector")
            continue
        dot = sum(vector[axis] / norm * forward[axis] for axis in range(3))
        dots.append(dot)
        if dot < threshold:
            failures.append(f"ego camera branch {index} has forward dot {dot:.3f}, below {threshold:.3f}")
    minimum_branches = int(criteria.get("minimum_camera_branches", 1))
    if failures:
        status = Status.FAIL
    elif len(dots) < minimum_branches:
        status = Status.UNVERIFIED
        failures.append(f"only {len(dots)} camera branches were discoverable; {minimum_branches} required")
    else:
        status = Status.PASS
    return GateResult(
        "egocentric_camera_orientation",
        status,
        f"{len(dots)} egocentric camera branches checked against the rig's forward axis",
        {"rig_forward": list(forward), "view_vectors": vectors, "forward_dots": dots, "source_sha256": evidence.get("source_sha256")},
        criteria,
        failures=failures or ([] if dots else ["no egocentric camera view vectors were discoverable"]),
    )
