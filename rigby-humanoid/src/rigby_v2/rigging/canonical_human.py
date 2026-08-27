"""Load and uniformly profile Rigby's canonical MuJoCo human."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco

from .mapping import BodySizeProfile, RigManifest, load_rig_manifest


# The foot boxes declare a 1 mm collision margin.  At geometric touch that
# margin is fully compressed and produces substantially more than body weight
# (for medium: 1218.8 N versus 723.4 N), so the unforced canonical rest starts
# with +7.83 m/s^2 vertical root acceleration.  These audited, profile-specific
# clearances put the soft contact at its static load point while remaining
# inside the declared margin and without changing geometry or friction.
_CONTACT_EQUILIBRIUM_CLEARANCE_M = {
    "small": 0.00063,
    "medium": 0.00061,
    "large": 0.00059,
}

_INERTIAL_DATA_NAME = "canonical_inertials.v1.json"
_INERTIAL_COLUMNS = (
    "mass_kg",
    "ipos_x_m",
    "ipos_y_m",
    "ipos_z_m",
    "iquat_w",
    "iquat_x",
    "iquat_y",
    "iquat_z",
    "principal_i_x_kg_m2",
    "principal_i_y_kg_m2",
    "principal_i_z_kg_m2",
)


def _audited_inertial_data(model_path: Path) -> dict[str, Any]:
    """Load and cryptographically validate the versioned inertial source."""

    path = model_path.with_name(_INERTIAL_DATA_NAME)
    data = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "schema_version",
        "columns",
        "reference_profile",
        "reference_linear_scale",
        "bodies",
        "provenance",
    )
    audit_payload = {key: data.get(key) for key in required}
    encoded = json.dumps(
        audit_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_audit = hashlib.sha256(encoded).hexdigest()
    if data.get("audit_sha256") != observed_audit:
        raise ValueError("canonical inertial data failed its audit hash")
    if data.get("schema_version") != "canonical_inertials.v1":
        raise ValueError("unsupported canonical inertial schema")
    if tuple(data.get("columns", ())) != _INERTIAL_COLUMNS:
        raise ValueError("canonical inertial columns do not match v1")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("canonical inertial provenance is missing")
    source_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if provenance.get("explicit_canonical_source_sha256") != source_hash:
        raise ValueError("canonical MJCF does not match inertial audit provenance")
    return data


def _format_values(values: list[float]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def _apply_explicit_inertials(
    root: ET.Element,
    profile: BodySizeProfile,
    data: dict[str, Any],
) -> None:
    """Materialize the audited medium inertials under uniform scale laws."""

    compiler = root.find("./compiler")
    if compiler is None or compiler.attrib.get("inertiafromgeom") != "auto":
        raise ValueError("canonical MJCF must use scoped automatic inertia")
    reference_scale = float(data.get("reference_linear_scale", 0.0))
    if data.get("reference_profile") != "medium" or reference_scale <= 0.0:
        raise ValueError("canonical inertial reference profile is invalid")
    bodies = data.get("bodies")
    if not isinstance(bodies, dict):
        raise ValueError("canonical inertial bodies must be an object")
    xml_bodies = {
        body.attrib.get("name"): body
        for body in root.iter("body")
        if body.attrib.get("name")
    }
    if set(xml_bodies) != set(bodies):
        raise ValueError("canonical MJCF bodies do not match inertial audit")
    scale = profile.linear_scale / reference_scale
    for name, body in xml_bodies.items():
        raw = bodies[name]
        if (
            not isinstance(raw, list)
            or len(raw) != len(_INERTIAL_COLUMNS)
            or not all(isinstance(value, (int, float)) for value in raw)
        ):
            raise ValueError(f"invalid canonical inertial record for {name!r}")
        values = [float(value) for value in raw]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite canonical inertial record for {name!r}")
        mass = values[0] * scale**3
        ipos = [value * scale for value in values[1:4]]
        iquat = values[4:8]
        inertia = [value * scale**5 for value in values[8:11]]
        if mass <= 0.0 or any(value <= 0.0 for value in inertia):
            raise ValueError(f"non-positive canonical inertia for {name!r}")
        if body.find("./inertial") is not None:
            raise ValueError(f"canonical body {name!r} already declares inertia")
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "mass": f"{mass:.17g}",
                    "pos": _format_values(ipos),
                    "quat": _format_values(iquat),
                    "diaginertia": _format_values(inertia),
                },
            ),
        )


def _scale_numbers(raw: str, factor: float) -> str:
    return " ".join(f"{float(value) * factor:.9g}" for value in raw.split())


def _apply_anatomical_actuator_limits(root: ET.Element) -> None:
    """Bound small distal joints well below whole-body actuator limits."""

    digits = ("thumb", "index", "middle", "ring", "little")
    arms = ("shoulder", "elbow", "forearm")
    for motor in root.findall("./actuator/motor"):
        joint = motor.attrib.get("joint", "")
        if any(token in joint for token in digits):
            motor.set("ctrlrange", "-3 3")
        elif "wrist" in joint:
            motor.set("ctrlrange", "-5 5")
        elif any(token in joint for token in arms):
            motor.set("ctrlrange", "-60 60")


def _apply_rest_limit_margin(root: ET.Element) -> None:
    """Keep neutral flexion joints away from a unilateral hard stop."""

    for joint in root.iter("joint"):
        raw_range = joint.attrib.get("range")
        if raw_range is None:
            continue
        lower, _ = (float(value) for value in raw_range.split())
        if lower == 0.0:
            joint.set("ref", "5")


def _apply_contact_equilibrium_clearance(
    root: ET.Element,
    profile: BodySizeProfile,
) -> None:
    try:
        clearance = _CONTACT_EQUILIBRIUM_CLEARANCE_M[profile.name]
    except KeyError as error:
        raise ValueError(
            f"canonical contact equilibrium is not audited for profile {profile.name!r}"
        ) from error
    pelvis = root.find("./worldbody/body[@name='pelvis']")
    if pelvis is None or "pos" not in pelvis.attrib:
        raise ValueError("canonical pelvis position is missing")
    position = [float(value) for value in pelvis.attrib["pos"].split()]
    if len(position) != 3:
        raise ValueError("canonical pelvis position must contain three values")
    position[2] += clearance
    pelvis.set("pos", " ".join(f"{value:.9g}" for value in position))


def xml_for_profile(
    profile: str | BodySizeProfile = "medium",
    *,
    manifest: RigManifest | None = None,
) -> str:
    """Return canonical MJCF with profile geometry and mass scaling applied.

    Per-body mass, inertial pose, and principal inertia come from the versioned
    audited source next to the MJCF.  Uniform profiles apply the dimensional
    laws mass=s^3, inertial position=s, and inertia=s^5.  MuJoCo geometry is no
    longer an implicit source of canonical-body inertia.
    """

    rig = manifest or load_rig_manifest()
    selected = rig.profile(profile) if isinstance(profile, str) else profile
    root = ET.fromstring(rig.model_path.read_text(encoding="utf-8"))
    inertial_data = _audited_inertial_data(rig.model_path)
    root.set("model", f"rigby_canonical_human_{selected.name}")
    _apply_anatomical_actuator_limits(root)
    _apply_rest_limit_margin(root)
    if selected.linear_scale != 1.0:
        for element in root.iter():
            if element.tag == "body" and "pos" in element.attrib:
                element.set("pos", _scale_numbers(element.attrib["pos"], selected.linear_scale))
            elif element.tag in {"geom", "site"}:
                for attribute in ("pos", "fromto", "size"):
                    if attribute in element.attrib:
                        element.set(
                            attribute,
                            _scale_numbers(element.attrib[attribute], selected.linear_scale),
                        )
    _apply_explicit_inertials(root, selected, inertial_data)
    _apply_contact_equilibrium_clearance(root, selected)
    return ET.tostring(root, encoding="unicode")


def load_canonical_human(
    profile: str | BodySizeProfile = "medium",
    *,
    manifest: RigManifest | None = None,
) -> mujoco.MjModel:
    """Compile and return the requested canonical human profile."""

    return mujoco.MjModel.from_xml_string(xml_for_profile(profile, manifest=manifest))
