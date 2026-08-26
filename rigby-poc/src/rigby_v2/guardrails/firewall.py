"""Deterministic, model-free capability and security intake firewall."""

from __future__ import annotations

import math
import re
from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath

from .models import (
    AssetIntakeV1,
    CapabilityDecisionV1,
    CapabilityFirewallError,
    FirewallOutcome,
    FirewallReason,
)


_SPACE = re.compile(r"\s+")
_DISTANCE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(millimeters?|mm|centimeters?|cm|meters?|m|kilometers?|km)\b")


def _normalize(prompt: str) -> str:
    return _SPACE.sub(" ", prompt.casefold().strip())


def _decision(
    prompt: str,
    outcome: FirewallOutcome,
    reason: FirewallReason,
    rule_id: str,
    *facts: str,
) -> CapabilityDecisionV1:
    return CapabilityDecisionV1(
        prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
        outcome=outcome,
        reason=reason,
        rule_id=rule_id,
        matched_facts=tuple(sorted(set(facts))),
    )


def _path_is_external(value: str) -> bool:
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
        or normalized.startswith(("file:", "http:", "https:", "//"))
    )


def _prompt_path_attack(text: str) -> str | None:
    tokens = re.findall(r"[^\s\"']+", text)
    for token in tokens:
        stripped = token.strip(".,;:()[]{}")
        if (
            "../" in stripped
            or "..\\" in stripped
            or re.match(r"^[a-zA-Z]:[\\/]", stripped)
            or stripped.startswith(("/", "\\\\", "file://"))
        ):
            return stripped
    return None


def _fixed_feet_distance_m(text: str) -> float | None:
    feet_fixed = any(
        phrase in text
        for phrase in (
            "without moving the feet",
            "without moving your feet",
            "feet planted",
            "feet fixed",
            "feet motionless",
        )
    )
    if not feet_fixed:
        return None
    scale = {
        "millimeter": 0.001,
        "millimeters": 0.001,
        "mm": 0.001,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "cm": 0.01,
        "meter": 1.0,
        "meters": 1.0,
        "m": 1.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "km": 1000.0,
    }
    distances = [float(value) * scale[unit] for value, unit in _DISTANCE.findall(text)]
    return max(distances) if distances else None


def validate_prompt(prompt: str) -> CapabilityDecisionV1:
    """Classify a planning prompt without consulting benchmark metadata."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("capability validation requires a nonempty prompt")
    text = _normalize(prompt)

    attacked_path = _prompt_path_attack(text)
    if attacked_path is not None:
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.PATH_TRAVERSAL,
            "asset.path.external.v1",
            attacked_path,
        )
    if re.search(r"\b(?:enable|load|execute|inject)\b.{0,40}\b(?:arbitrary\s+)?(?:native\s+)?plugin\b", text) or re.search(
        r"\bplugin\b.{0,30}\b(?:\.dll|\.so|\.dylib|evil[_-]?plugin)\b", text
    ):
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.PLUGIN_INJECTION,
            "asset.plugin.native.v1",
            "native_plugin_request",
        )
    invalid_scalar = re.search(
        r"\b(?:nan|[+-]?inf(?:inity)?)\b.{0,24}\b(?:mass|inertia)\b|\b(?:mass|inertia)\b.{0,24}\b(?:nan|[+-]?inf(?:inity)?)\b",
        text,
    )
    zero_or_negative_inertia = re.search(
        r"\b(?:zero|negative|-\s*\d+(?:\.\d+)?)\s+inertia\b|\binertia\b.{0,12}\b(?:zero|negative)\b",
        text,
    )
    if invalid_scalar or zero_or_negative_inertia:
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.NONFINITE_OR_INVALID_INERTIA,
            "asset.inertial.finite-positive.v1",
            "invalid_mass_or_inertia",
        )
    if (
        re.search(r"\b(?:raw|unprocessed|unrestricted)\b.{0,30}\bconcave\b", text)
        and re.search(r"\b(?:dynamic|moving|free)\b", text)
        and re.search(r"\b(?:collider|collision|mesh)\b", text)
    ):
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.RAW_CONCAVE_DYNAMIC_COLLIDER,
            "asset.dynamic-collider.convex-only.v1",
            "raw_concave",
            "dynamic",
        )

    if re.search(r"\b(?:deformable|soft[- ]body)\b", text) or (
        re.search(r"\b(?:cloth|fabric|towel|garment)\b", text)
        and re.search(r"\b(?:fold|drape|wrinkle|crumple|stretch|animate|simulate)\w*\b", text)
    ):
        return _decision(
            prompt,
            FirewallOutcome.UNSUPPORTED,
            FirewallReason.DEFORMABLE_SIMULATION,
            "capability.deformable.v1",
            "deformable_material",
        )
    if re.search(r"\bfree[- ]surface\b", text) or (
        re.search(r"\b(?:liquid|fluid|water|juice|oil)\b", text)
        and re.search(r"\b(?:pour|slosh|splash|spill|flow|transfer)\w*\b", text)
    ):
        return _decision(
            prompt,
            FirewallOutcome.UNSUPPORTED,
            FirewallReason.FLUID_SIMULATION,
            "capability.fluid.v1",
            "free_surface_fluid",
        )
    if re.search(
        r"\b(?:two|2|multiple|several)\s+(?:independent\s+)?(?:people|persons|humans|characters|actors)\b|\b(?:crowd|multi[- ]character)\b",
        text,
    ):
        return _decision(
            prompt,
            FirewallOutcome.UNSUPPORTED,
            FirewallReason.MULTI_CHARACTER,
            "capability.single-character.v1",
            "multiple_independent_characters",
        )
    if (
        re.search(r"\b(?:real|physical|uncalibrated)\s+robot\b", text)
        and re.search(r"\b(?:execute|deploy|run|send|control)\w*\b", text)
    ) or re.search(r"\bdirectly\s+on\b.{0,24}\brobot\b", text):
        return _decision(
            prompt,
            FirewallOutcome.UNSUPPORTED,
            FirewallReason.DIRECT_ROBOT_EXECUTION,
            "capability.simulation-only.v1",
            "direct_robot_hardware",
        )

    fixed_distance = _fixed_feet_distance_m(text)
    if fixed_distance is not None and fixed_distance > 2.0:
        return _decision(
            prompt,
            FirewallOutcome.INFEASIBLE,
            FirewallReason.UNREACHABLE_TARGET,
            "feasibility.fixed-base-reach.v1",
            f"distance_m={fixed_distance:.9g}",
        )
    both_hands_fixed = re.search(
        r"\b(?:keep|hold)\s+both\s+hands\s+(?:motionless|still|fixed)\b|\bwithout\s+moving\s+(?:either|both)\s+hands\b",
        text,
    )
    bimanual_action = re.search(
        r"\b(?:lift|move|pick\s+up|carry|open|rotate|manipulate)\w*\b.{0,45}\b(?:with|using)\s+both\s+hands\b",
        text,
    )
    object_fixed_and_moved = re.search(
        r"\bkeep\s+(?:the\s+)?object\s+(?:motionless|fixed|still)\b.{0,60}\b(?:move|lift|rotate|translate)\s+(?:the\s+)?(?:same\s+)?object\b",
        text,
    )
    if (both_hands_fixed and bimanual_action) or object_fixed_and_moved:
        return _decision(
            prompt,
            FirewallOutcome.INFEASIBLE,
            FirewallReason.CONTRADICTORY_CONSTRAINTS,
            "feasibility.constraint-contradiction.v1",
            "same_effector_fixed_and_required",
        )
    return _decision(
        prompt,
        FirewallOutcome.ALLOW,
        FirewallReason.SUPPORTED,
        "supported.rigid-articulated.v1",
    )


def validate_asset(asset: AssetIntakeV1, *, prompt: str = "structured asset intake") -> CapabilityDecisionV1:
    """Validate structured asset data using the same stable outcome contract."""

    external = next((path for path in asset.asset_paths if _path_is_external(path)), None)
    if external is not None:
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.PATH_TRAVERSAL,
            "asset.path.external.v1",
            external,
        )
    if asset.native_plugins:
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.PLUGIN_INJECTION,
            "asset.plugin.native.v1",
            *asset.native_plugins,
        )
    invalid_mass = asset.mass_kg is not None and (
        not math.isfinite(asset.mass_kg) or asset.mass_kg <= 0.0
    )
    invalid_inertia = asset.inertia_diagonal_kg_m2 is not None and any(
        not math.isfinite(value) or value <= 0.0
        for value in asset.inertia_diagonal_kg_m2
    )
    if invalid_mass or invalid_inertia:
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.NONFINITE_OR_INVALID_INERTIA,
            "asset.inertial.finite-positive.v1",
            "invalid_mass_or_inertia",
        )
    if asset.dynamic and asset.collision_representation == "raw_concave_mesh":
        return _decision(
            prompt,
            FirewallOutcome.INVALID_ASSET,
            FirewallReason.RAW_CONCAVE_DYNAMIC_COLLIDER,
            "asset.dynamic-collider.convex-only.v1",
            "raw_concave",
            "dynamic",
        )
    return _decision(
        prompt,
        FirewallOutcome.ALLOW,
        FirewallReason.SUPPORTED,
        "supported.asset.v1",
    )


def enforce_planning_intake(prompt: str) -> CapabilityDecisionV1:
    decision = validate_prompt(prompt)
    if decision.outcome is not FirewallOutcome.ALLOW:
        raise CapabilityFirewallError(decision)
    return decision

