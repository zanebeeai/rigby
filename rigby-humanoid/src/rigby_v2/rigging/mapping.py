"""Validated skeleton-to-DOF manifest access for the canonical human."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "assets" / "v2" / "rig_manifest.json"


@dataclass(frozen=True, slots=True)
class BodySizeProfile:
    name: str
    linear_scale: float
    nominal_height_m: float
    nominal_mass_kg: float


@dataclass(frozen=True, slots=True)
class RigManifest:
    schema_version: str
    model_path: Path
    canonical_frame: str
    source_frame: str
    free_root_joint: str
    skeleton_to_dofs: dict[str, tuple[str, ...]]
    fixed_visual_bones: dict[str, str]
    semantic_sites: dict[str, str]
    profiles: dict[str, BodySizeProfile]

    def dofs_for_bone(self, canonical_bone: str) -> tuple[str, ...]:
        try:
            return self.skeleton_to_dofs[canonical_bone]
        except KeyError as error:
            raise KeyError(f"unknown canonical bone: {canonical_bone}") from error

    def profile(self, name: str) -> BodySizeProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            raise KeyError(f"unknown body-size profile: {name}") from error


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


@lru_cache(maxsize=4)
def load_rig_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> RigManifest:
    resolved = path.resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    bone_map = data.get("skeleton_to_dofs")
    fixed_bones = data.get("fixed_visual_bones")
    sites = data.get("semantic_sites")
    profiles_data = data.get("body_size_profiles")
    if not isinstance(bone_map, dict) or not bone_map:
        raise ValueError("manifest skeleton_to_dofs must be a non-empty object")
    if not isinstance(fixed_bones, dict) or not all(
        isinstance(bone, str) and bone and isinstance(reason, str) and reason
        for bone, reason in fixed_bones.items()
    ):
        raise ValueError(
            "manifest fixed_visual_bones must map bone names to audit reasons"
        )
    if set(fixed_bones) & set(bone_map):
        raise ValueError("visual bones cannot be both physical and fixed")
    if not isinstance(sites, dict) or not sites:
        raise ValueError("manifest semantic_sites must be a non-empty object")
    if not isinstance(profiles_data, dict) or set(profiles_data) != {
        "small",
        "medium",
        "large",
    }:
        raise ValueError(
            "manifest must define small, medium, and large body-size profiles"
        )

    skeleton_to_dofs: dict[str, tuple[str, ...]] = {}
    for bone, dofs in bone_map.items():
        if (
            not isinstance(bone, str)
            or not isinstance(dofs, list)
            or not all(isinstance(item, str) and item for item in dofs)
        ):
            raise ValueError(
                "skeleton_to_dofs entries must map names to joint-name lists"
            )
        skeleton_to_dofs[bone] = tuple(dofs)

    profiles = {
        name: BodySizeProfile(
            name=name,
            linear_scale=float(value["linear_scale"]),
            nominal_height_m=float(value["nominal_height_m"]),
            nominal_mass_kg=float(value["nominal_mass_kg"]),
        )
        for name, value in profiles_data.items()
    }
    if any(profile.linear_scale <= 0.0 for profile in profiles.values()):
        raise ValueError("body-size scales must be positive")

    model_path = (resolved.parent / _require_string(data, "model_path")).resolve()
    return RigManifest(
        schema_version=_require_string(data, "schema_version"),
        model_path=model_path,
        canonical_frame=_require_string(data, "canonical_frame"),
        source_frame=_require_string(data, "source_frame"),
        free_root_joint=_require_string(data, "free_root_joint"),
        skeleton_to_dofs=skeleton_to_dofs,
        fixed_visual_bones={str(key): str(value) for key, value in fixed_bones.items()},
        semantic_sites={str(key): str(value) for key, value in sites.items()},
        profiles=profiles,
    )
