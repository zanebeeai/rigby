from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_v2.rigging import load_rig_manifest
from rigby_v2.rigging.canonical_human import xml_for_profile


def _implicit_reference(explicit_xml: str) -> mujoco.MjModel:
    """Reconstruct the audited pre-migration compiler behavior."""

    root = ET.fromstring(explicit_xml)
    compiler = root.find("./compiler")
    assert compiler is not None
    compiler.set("inertiafromgeom", "true")
    for body in root.iter("body"):
        inertial = body.find("./inertial")
        assert inertial is not None
        body.remove(inertial)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("profile", ("small", "medium", "large"))
def test_explicit_inertials_are_equivalent_to_audited_compiled_values(
    profile: str,
) -> None:
    xml = xml_for_profile(profile)
    root = ET.fromstring(xml)
    compiler = root.find("./compiler")
    assert compiler is not None
    # Auto preserves composition for new pack bodies, while the assertion
    # below proves every canonical body bypasses inference with explicit data.
    assert compiler.attrib["inertiafromgeom"] == "auto"
    bodies = tuple(root.iter("body"))
    assert len(bodies) == 47
    assert all(body.find("./inertial") is not None for body in bodies)

    explicit = mujoco.MjModel.from_xml_string(xml)
    implicit = _implicit_reference(xml)
    assert explicit.nbody == implicit.nbody
    if profile == "medium":
        # The captured reference profile reproduces the compiled arrays byte
        # for byte; scaled profiles are judged at strict floating tolerance.
        for explicit_values, implicit_values in (
            (explicit.body_mass, implicit.body_mass),
            (explicit.body_ipos, implicit.body_ipos),
            (explicit.body_iquat, implicit.body_iquat),
            (explicit.body_inertia, implicit.body_inertia),
        ):
            assert explicit_values.tobytes() == implicit_values.tobytes()
    np.testing.assert_allclose(
        explicit.body_mass,
        implicit.body_mass,
        rtol=2e-15,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        explicit.body_ipos,
        implicit.body_ipos,
        rtol=2e-15,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        explicit.body_iquat,
        implicit.body_iquat,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        explicit.body_inertia,
        implicit.body_inertia,
        rtol=3e-15,
        atol=1e-18,
    )


def _copy_audit_assets(destination: Path) -> tuple[Path, Path]:
    manifest = load_rig_manifest()
    xml_path = destination / "canonical_human.xml"
    data_path = destination / "canonical_inertials.v1.json"
    xml_path.write_bytes(manifest.model_path.read_bytes())
    data_path.write_bytes(
        manifest.model_path.with_name("canonical_inertials.v1.json").read_bytes()
    )
    return xml_path, data_path


def test_inertial_data_tampering_fails_closed(tmp_path: Path) -> None:
    xml_path, data_path = _copy_audit_assets(tmp_path)
    data_path.write_text(
        data_path.read_text(encoding="utf-8").replace(
            "11.2220831178881",
            "11.3220831178881",
            1,
        ),
        encoding="utf-8",
    )
    manifest = replace(load_rig_manifest(), model_path=xml_path)
    with pytest.raises(ValueError, match="audit hash"):
        xml_for_profile("medium", manifest=manifest)


def test_canonical_source_tampering_breaks_provenance(tmp_path: Path) -> None:
    xml_path, _ = _copy_audit_assets(tmp_path)
    xml_path.write_text(
        xml_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    manifest = replace(load_rig_manifest(), model_path=xml_path)
    with pytest.raises(ValueError, match="audit provenance"):
        xml_for_profile("medium", manifest=manifest)
