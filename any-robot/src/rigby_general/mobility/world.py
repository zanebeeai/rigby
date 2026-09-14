"""The common course the three mobile bodies share, and the feasibility map that says which of them can do what on it.

One world, frozen before any controller is tuned, independent of which
locomotion controller later succeeds: a level floor, a start pad, a
corridor between two walls, an object station (a low platform with a
cube on it), a delivery tray, a ramp on the way back, and two side
branches the bodies are expected to disagree about -- a narrow doorway
and a step. The feasibility analysis reads each body's manifest and
settling measurements (footprint, stance heights, reach, support
friction, wheel radius) against the course's dimensions and gives every
body x element a verdict with the numbers that decided it. The expert
baseline beside it is labelled by hand and kept apart, so agreement
between the two is something to report rather than assume.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..evidence.capture import json_bytes
from .contracts import MobileBodyManifestV1, StanceMeasurementV1


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str
    """pad | wall | platform | object | tray | ramp | doorway | step"""
    position_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    """half extents"""
    note: str = ""
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.55, 1.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass_kg: float | None = None
    """An object (free) rather than a fixture (static)."""


@dataclass(frozen=True)
class Course:
    course_id: str
    description: str
    floor_friction: float
    features: tuple[Feature, ...]
    route: tuple[tuple[float, float], ...]
    """Waypoints of the main route: start -> station -> tray -> start."""
    goals: tuple[dict, ...]
    branches: tuple[dict, ...]
    """Side elements with their governing dimension: the doorway's width, the step's height, the ramp's grade."""

    def as_json(self) -> dict:
        return {"schema": "rigby.mobile-course/1", "course_id": self.course_id, "description": self.description, "floor_friction": self.floor_friction,
                "features": [{"name": f.name, "kind": f.kind, "position_m": list(f.position_m), "size_m": list(f.size_m), "note": f.note, "rgba": list(f.rgba), "euler": list(f.euler), "mass_kg": f.mass_kg} for f in self.features],
                "route_m": [list(p) for p in self.route], "goals": list(self.goals), "branches": list(self.branches)}

    def sha256(self) -> str:
        return hashlib.sha256(json_bytes(self.as_json())).hexdigest()

    def mjcf_features(self) -> str:
        """The course's features as MJCF worldbody children (the floor is the body's own)."""

        parts = []
        for f in self.features:
            pos = " ".join(f"{v:.4f}" for v in f.position_m)
            size = " ".join(f"{v:.4f}" for v in f.size_m)
            rgba = " ".join(f"{v:.3f}" for v in f.rgba)
            euler = " ".join(f"{v:.5f}" for v in f.euler)
            if f.mass_kg is not None:
                parts.append(f'<body name="course_{f.name}" pos="{pos}"><freejoint name="course_{f.name}_free"/><geom name="course_{f.name}_geom" type="box" size="{size}" mass="{f.mass_kg:.4f}" rgba="{rgba}" friction="1.4 0.005 0.0001"/></body>')
            else:
                parts.append(f'<geom name="course_{f.name}" type="box" pos="{pos}" size="{size}" euler="{euler}" rgba="{rgba}" friction="{self.floor_friction:.3f} 0.005 0.0001"/>')
        return "\n    ".join(parts)


def course_v1() -> Course:
    ramp_grade_deg = 6.0
    ramp_length = 1.2
    ramp_rise = ramp_length * math.tan(math.radians(ramp_grade_deg))
    features = (
        Feature("start_pad", "pad", (0.0, 0.0, 0.002), (0.45, 0.45, 0.002), "where every body starts and returns to", (0.25, 0.55, 0.35, 1.0)),
        Feature("corridor_left", "wall", (1.6, 0.75, 0.25), (1.0, 0.05, 0.25), "the corridor's left wall; 1.4 m clear between the walls' inner faces", (0.6, 0.6, 0.62, 1.0)),
        Feature("corridor_right", "wall", (1.6, -0.75, 0.25), (1.0, 0.05, 0.25), "the corridor's right wall", (0.6, 0.6, 0.62, 1.0)),
        Feature("station", "platform", (3.4, 0.0, 0.03), (0.25, 0.25, 0.03), "the object station: a platform 6 cm tall", (0.3, 0.45, 0.75, 1.0)),
        Feature("cube", "object", (3.4, 0.0, 0.075), (0.015, 0.015, 0.015), "a 30 mm cube on the station, top at 9 cm", (0.85, 0.45, 0.15, 1.0), mass_kg=0.05),
        Feature("tray", "tray", (3.4, 2.2, 0.06), (0.2, 0.2, 0.01), "the delivery tray's floor, 5 cm up on a plinth", (0.35, 0.35, 0.4, 1.0)),
        Feature("tray_plinth", "platform", (3.4, 2.2, 0.025), (0.22, 0.22, 0.025), "the tray's plinth", (0.4, 0.4, 0.45, 1.0)),
        Feature("tray_rim_n", "tray", (3.4, 2.4, 0.09), (0.2, 0.01, 0.02), "the tray's rim, 2 cm above its floor", (0.35, 0.35, 0.4, 1.0)),
        Feature("tray_rim_s", "tray", (3.4, 2.0, 0.09), (0.2, 0.01, 0.02), "", (0.35, 0.35, 0.4, 1.0)),
        Feature("tray_rim_e", "tray", (3.6, 2.2, 0.09), (0.01, 0.2, 0.02), "", (0.35, 0.35, 0.4, 1.0)),
        Feature("tray_rim_w", "tray", (3.2, 2.2, 0.09), (0.01, 0.2, 0.02), "", (0.35, 0.35, 0.4, 1.0)),
        Feature("ramp", "ramp", (1.8, 2.2, ramp_rise / 2 - 0.005), (ramp_length / 2, 0.6, 0.01), f"a {ramp_grade_deg:.0f} degree ramp on the way back from the tray, rising {ramp_rise * 100:.0f} cm over {ramp_length:.1f} m",
                (0.55, 0.5, 0.45, 1.0), euler=(0.0, -math.radians(ramp_grade_deg), 0.0)),
        Feature("ramp_top", "platform", (0.85, 2.2, ramp_rise / 2), (0.35, 0.6, ramp_rise / 2), "the landing at the top of the ramp", (0.55, 0.5, 0.45, 1.0)),
        Feature("doorway_left", "doorway", (1.6, -2.2, 0.3), (0.05, 0.4, 0.3), "a doorway 0.7 m wide on a side branch", (0.65, 0.55, 0.5, 1.0)),
        Feature("doorway_right", "doorway", (1.6, -3.3, 0.3), (0.05, 0.4, 0.3), "", (0.65, 0.55, 0.5, 1.0)),
        Feature("step", "step", (3.4, -2.4, 0.025), (0.5, 0.4, 0.025), "a 5 cm step on a side branch", (0.5, 0.45, 0.4, 1.0)),
    )
    goals = (
        {"goal_id": "travel", "for": "G17", "text": "leave the start pad, pass the corridor to the station, stop within 0.3 m of it for 2 s, turn around, return to the start pad and stop", "route": "start -> station -> start", "success": "base centre within 0.3 m of each waypoint in turn, stable for 2 s at the end, no fall"},
        {"goal_id": "retrieve", "for": "G18", "text": "approach the station, stabilise, acquire the cube, carry it to the tray, place it inside the rim, return to the start pad", "route": "start -> station -> tray -> ramp -> start", "success": "the cube at rest inside the tray's rim, the body back on the start pad and stable, nothing knocked over"},
    )
    branches = (
        {"branch_id": "corridor", "element": "clearance", "value_m": 1.4, "rule": "body footprint across the direction of travel + 0.1 m margin must fit"},
        {"branch_id": "doorway", "element": "clearance", "value_m": 0.7, "rule": "same rule; a side branch the bodies are expected to disagree about"},
        {"branch_id": "ramp", "element": "traction", "grade_deg": ramp_grade_deg, "rule": "support friction must exceed tan(grade) with a factor of two; a legged body must also keep its stance height on the slope"},
        {"branch_id": "step", "element": "step", "value_m": 0.05, "rule": "a wheel climbs a step lower than half its radius; a leg steps up to a third of its length; a crawler clears a step lower than its mantle's rise"},
        {"branch_id": "station", "element": "reach", "object_top_m": 0.09, "rule": "the grasp site must reach the cube's top with the body in a stance it can hold"},
        {"branch_id": "tray", "element": "reach", "rim_top_m": 0.11, "rule": "the grasp site must reach over the rim to the tray floor at 0.07 m"},
    )
    return Course(course_id="mobile-course-v1", description="A level floor with a start pad, a walled corridor to an object station, a delivery tray on a plinth, a gentle ramp on the way back, and two side branches: a narrow doorway and a step.",
                  floor_friction=1.0, features=features, route=((0.0, 0.0), (3.0, 0.0), (3.4, 2.2), (1.2, 2.2), (0.0, 0.0)), goals=goals, branches=branches)


def _stance_heights(stances: tuple[StanceMeasurementV1, ...], manifest: MobileBodyManifestV1) -> tuple[float, float]:
    """The lowest and highest base heights among the stances measured statically stable (the declared standing height counts for a body whose working stance needs a controller)."""

    heights = [s.base_height_m for s in stances if s.statically_stable]
    if manifest.stances and not any(s.stance == manifest.working_stance and s.statically_stable for s in stances):
        heights.append(manifest.expected_standing_height_m)
    return (min(heights), max(heights)) if heights else (manifest.expected_standing_height_m, manifest.expected_standing_height_m)


def feasibility(course: Course, manifest: MobileBodyManifestV1, stances: tuple[StanceMeasurementV1, ...], *, arm_base_height_above_base_m: float, wheel_radius_m: float | None, leg_length_m: float | None,
                mantle_rise_m: float | None) -> dict:
    """Every course element judged for one body from its measured numbers."""

    verdicts = []
    across = min(manifest.footprint_m)
    low, high = _stance_heights(stances, manifest)
    friction = min(m.friction for m in manifest.support_members)
    reach = max((m.reach_m for m in manifest.manipulators), default=0.0)
    for branch in course.branches:
        if branch["element"] == "clearance":
            fits = across + 0.1 <= branch["value_m"]
            verdicts.append({"branch_id": branch["branch_id"], "feasible": fits, "measured": {"footprint_across_m": round(across, 3), "clearance_m": branch["value_m"]}, "why": f"footprint {across:.2f} m + 0.1 m margin {'fits' if fits else 'does not fit'} {branch['value_m']:.1f} m"})
        elif branch["element"] == "traction":
            need = 2.0 * math.tan(math.radians(branch["grade_deg"]))
            ok = friction >= need
            verdicts.append({"branch_id": branch["branch_id"], "feasible": ok, "measured": {"support_friction": round(friction, 3), "required": round(need, 3)}, "why": f"support friction {friction:.2f} against 2 x tan({branch['grade_deg']:.0f} deg) = {need:.2f}"})
        elif branch["element"] == "step":
            h = branch["value_m"]
            if manifest.base_kind.value == "wheeled":
                limit, basis = 0.5 * (wheel_radius_m or 0.0), f"half the wheel radius ({(wheel_radius_m or 0.0) * 100:.0f} cm)"
            elif manifest.base_kind.value == "legged":
                limit, basis = (leg_length_m or 0.0) / 3.0, f"a third of the leg length ({(leg_length_m or 0.0) * 100:.0f} cm)"
            else:
                limit, basis = (mantle_rise_m or 0.0), f"the mantle's rise ({(mantle_rise_m or 0.0) * 100:.0f} cm)"
            ok = h <= limit
            verdicts.append({"branch_id": branch["branch_id"], "feasible": ok, "measured": {"step_m": h, "limit_m": round(limit, 3)}, "why": f"a {h * 100:.0f} cm step against {basis}: {limit * 100:.1f} cm"})
        elif branch["element"] == "reach":
            target = branch.get("object_top_m", branch.get("rim_top_m"))
            # the arm hangs from its base: the lowest point it reaches is base height + arm base offset - reach; the highest is + reach
            lowest = low + arm_base_height_above_base_m - reach
            highest = high + arm_base_height_above_base_m + reach
            ok = lowest <= target <= highest and reach > 0
            verdicts.append({"branch_id": branch["branch_id"], "feasible": ok, "measured": {"target_m": target, "reachable_band_m": [round(lowest, 3), round(highest, 3)], "reach_m": round(reach, 3), "stance_heights_m": [round(low, 3), round(high, 3)]},
                             "why": f"target at {target:.2f} m; the grasp site spans {lowest:.2f} to {highest:.2f} m over the stances the body can hold"})
    return {"robot_id": manifest.robot_id, "verdicts": verdicts, "all_main_route_feasible": all(v["feasible"] for v in verdicts if v["branch_id"] in ("corridor", "ramp", "station", "tray"))}


__all__ = ["Course", "Feature", "course_v1", "feasibility"]
