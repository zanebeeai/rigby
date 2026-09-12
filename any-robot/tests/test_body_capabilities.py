"""Capability provenance and source-name invariance on real compiled bodies."""

import importlib.util
import json
from pathlib import Path
import random
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
from rigby_core.motion.compiler import compile_motion_program

from rigby_general.bake.enumerate import build_candidate
from rigby_general.capabilities.intake import ingest_capability_body
from rigby_general.errors import GeneralFailureCode, ModelIngestError, RigbyGeneralError
from rigby_general.grounding import ground
from rigby_general.pipeline import ingest_robot
from rigby_general.schema.inventory import load_inventory
from rigby_general.schema.program import Remove

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/g03_geometry"
ZOO = sorted((ROOT / "assets/general/zoo").glob("*/robot.urdf"))


@pytest.fixture(scope="module")
def third_party(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("third_party_package", FIXTURES.parent / "g03_third_party/package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract(tmp_path_factory.mktemp("third-party-bodies"))


def unordered_xml(element, inverse=None):
    inverse = inverse or {}
    clone = ET.Element(element.tag, dict(sorted((k, inverse.get(v, v)) for k, v in element.attrib.items())))
    clone.text = (element.text or "").strip() or None
    clone.extend(sorted((unordered_xml(c, inverse) for c in element), key=lambda n: ET.tostring(n)))
    return clone


def renamed_variant(source: Path, index: int, destination: Path):
    root = ET.fromstring(source.read_bytes())
    original = ET.tostring(unordered_xml(root))
    names = [node.get("name") for node in root if node.tag in {"link", "joint"}]
    randomizer = random.Random(20260912 + index)
    shuffled = names.copy()
    randomizer.shuffle(shuffled)
    prefixes = ("camera", "imu", "left", "right", "finger", "wheel", "vacuum", "opaque")
    mapping = {name: f"{prefixes[index % len(prefixes)]}_{slot:04d}" for slot, name in enumerate(shuffled)}
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            node.set(key, mapping.get(value, value))
        children = list(node)
        randomizer.shuffle(children)
        node[:] = children
    inverse = {value: key for key, value in mapping.items()}
    assert ET.tostring(unordered_xml(root, inverse)) == original, "Variant must preserve every mechanical declaration"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8", newline="\n")
    return inverse


def reference(body):
    robot = body.robot
    inventory = load_inventory()
    candidate = build_candidate(inventory.by_id("reach_to_point"), Remove.MEDIAL)
    grounded = ground(candidate.program, robot.manifest, robot.finalized.model, inventory)
    motion = compile_motion_program(grounded.program, robot.finalized.model, robot.manifest, sample_hz=60)
    times = np.linspace(0, float(motion.times_s[-1]), 101)
    qpos = np.array([motion.sample(float(t)).qpos for t in times])
    data = mujoco.MjData(robot.finalized.model)
    positions = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(robot.finalized.model, data)
        positions.append(data.xpos[1:].copy())
    return times, qpos, np.asarray(positions), grounded.figure_sites


@pytest.mark.parametrize("source", ZOO, ids=lambda p: p.parent.name)
def test_twenty_name_and_order_permutations_preserve_capabilities_and_motion(source, tmp_path):
    baseline = ingest_capability_body(source)
    expected = reference(baseline)
    source_hashes = set()
    for index in range(20):
        variant = tmp_path / str(index) / "robot.urdf"
        renamed_variant(source, index, variant)
        actual = ingest_capability_body(variant)
        source_hashes.add(actual.manifest.source_urdf_sha256)
        assert actual.canonical.xml == baseline.canonical.xml
        assert actual.manifest.capability_hash() == baseline.manifest.capability_hash()
        assert actual.robot.mjcf_xml == baseline.robot.mjcf_xml
        result = reference(actual)
        assert result[3] == expected[3], "The same physical effector must remain selected"
        for measured, wanted in zip(result[:3], expected[:3]):
            assert np.array_equal(measured, wanted), "Reference clock, joint motion and every body transform must agree exactly"
    assert len(source_hashes) == 20


@pytest.mark.parametrize("source", ZOO, ids=lambda p: p.parent.name)
def test_canonicalization_preserves_uploaded_kinematics_and_inertia(source):
    original = ingest_robot(source)
    canonical = ingest_capability_body(source)
    before, after = original.finalized.model, canonical.robot.finalized.model
    a, b = mujoco.MjData(before), mujoco.MjData(after)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        a.qpos[:] = original.manifest.rest_qpos
        b.qpos[:] = canonical.robot.manifest.rest_qpos
        for changed, uploaded in canonical.manifest.joint_aliases.items():
            one = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_JOINT, uploaded)
            two = mujoco.mj_name2id(after, mujoco.mjtObj.mjOBJ_JOINT, changed)
            if one < 0:  # Fixed URDF joints have no simulator coordinate.
                assert two < 0
                continue
            assert np.array_equal(before.jnt_range[one], after.jnt_range[two])
            assert np.array_equal(before.jnt_axis[one], after.jnt_axis[two])
            low, high = before.jnt_range[one]
            value = low + fraction*(high-low)
            a.qpos[before.jnt_qposadr[one]] = value
            b.qpos[after.jnt_qposadr[two]] = value
        mujoco.mj_kinematics(before, a)
        mujoco.mj_kinematics(after, b)
        for changed, uploaded in canonical.manifest.link_aliases.items():
            one = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, uploaded)
            two = mujoco.mj_name2id(after, mujoco.mjtObj.mjOBJ_BODY, changed)
            assert np.allclose(a.xpos[one], b.xpos[two], atol=1e-12, rtol=0)
            assert np.allclose(a.xmat[one], b.xmat[two], atol=1e-12, rtol=0)
            assert before.body_mass[one] == after.body_mass[two]
            assert np.array_equal(before.body_inertia[one], after.body_inertia[two])


@pytest.mark.parametrize("kind", ("rigid_tool", "opposing_jaws", "three_digits"))
def test_sites_and_effectors_match_frozen_independent_geometry(kind):
    spec = importlib.util.spec_from_file_location("independent_geometry", FIXTURES / "verify_geometry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()  # Standard-library geometry and frozen digests, no Rigby oracle.
    labels = json.loads((FIXTURES / kind / "expected.json").read_bytes())
    body = ingest_capability_body(FIXTURES / kind / "robot.urdf")
    robot, model = body.robot, body.robot.finalized.model
    assert [e.kind.value for e in robot.morphology.effectors] == [labels["expected_effector_kind"]]
    assert max(c.positioning_dof for c in robot.morphology.chains) == labels["expected_positioning_dof"]
    data = mujoco.MjData(model)
    for changed, uploaded in body.manifest.joint_aliases.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, changed)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = labels["configuration"][uploaded]
    mujoco.mj_kinematics(model, data)
    matched = set()
    for site in robot.morphology.sites:
        uploaded = body.manifest.link_aliases[site.body]
        eligible = [(name, point) for name, point in labels["points"].items()
                    if point["body"] == uploaded and (site.semantic.value in point["semantics"]
                        or (site.semantic.value == "tip" and "tip_if_selected" in point["semantics"]))]
        if site.semantic.value == "grasp_point":
            eligible = [("gripping_region", labels["gripping_region"]["region_centre"])]
        assert eligible, (kind, site.semantic, uploaded)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name)
        errors = [(np.linalg.norm(data.site_xpos[sid] - point["world_m"]), name) for name, point in eligible]
        error, name = min(errors)
        assert error <= labels["position_tolerance_m"], (kind, site.name, float(error))
        matched.add((name, site.semantic.value))
    required = {(name, semantic) for name, point in labels["points"].items()
                for semantic in point["semantics"] if semantic != "tip_if_selected"}
    assert required <= matched, "Do not pass by omitting a required site"
    if kind != "rigid_tool":
        assert ("gripping_region", "grasp_point") in matched
    for item in body.manifest.capabilities:
        if item.capability == "physical_grasp_transport":
            assert item.status == "unknown" and not item.enabled


@pytest.mark.parametrize("attribute,value", [("upper", "-0.1"), ("upper", "-0.05"), ("lower", "nan"), ("velocity", "-1"), ("effort", "inf")])
def test_invalid_source_limits_are_typed_refusals_before_measurement(tmp_path, attribute, value):
    tree = ET.parse(FIXTURES / "rigid_tool/robot.urdf")
    tree.getroot().find("joint[@name='position_x']/limit").set(attribute, value)
    path = tmp_path / "robot.urdf"
    path.write_text(ET.tostring(tree.getroot(), encoding="unicode"), encoding="utf-8", newline="\n")
    with pytest.raises(ModelIngestError) as error:
        ingest_capability_body(path)
    assert error.value.code == GeneralFailureCode.INVALID_JOINT_LIMIT


def test_missing_capability_and_invalid_inertia_are_typed(tmp_path):
    body = ingest_capability_body(FIXTURES / "rigid_tool/robot.urdf")
    for capability in ("suction", "rolling", "dynamic_balance", "hardware_rgb", "physical_payload_capacity"):
        with pytest.raises(RigbyGeneralError) as error:
            body.require(capability)
        assert error.value.code == GeneralFailureCode.UNAFFORDED_SCHEMA
    tree = ET.parse(FIXTURES / "rigid_tool/robot.urdf")
    tree.getroot().find("link[@name='tool']/inertial/inertia").set("ixx", "-1")
    path = tmp_path / "robot.urdf"
    path.write_text(ET.tostring(tree.getroot(), encoding="unicode"), encoding="utf-8", newline="\n")
    with pytest.raises(ModelIngestError) as error:
        ingest_capability_body(path)
    assert error.value.code == GeneralFailureCode.DEGENERATE_INERTIA


def test_grasp_centroid_must_not_cross_a_central_palm_obstacle(tmp_path):
    tree = ET.parse(FIXTURES / "opposing_jaws/robot.urdf")
    palm = tree.getroot().find("link[@name='palm']")
    for kind in ("collision", "visual"):
        shape = ET.SubElement(palm, kind)
        ET.SubElement(shape, "origin", xyz="0 0 0.045")
        ET.SubElement(ET.SubElement(shape, "geometry"), "box", size="0.004 0.006 0.010")
    path = tmp_path / "robot.urdf"
    path.write_text(ET.tostring(tree.getroot(), encoding="unicode"), encoding="utf-8", newline="\n")
    body = ingest_capability_body(path)
    points = [s for s in body.robot.morphology.sites if s.semantic.value == "grasp_point"]
    assert points, "The unobstructed side regions still provide geometric point witnesses"
    for site in points:
        assert body.manifest.link_aliases[site.body] == "palm"
        local = np.array([site.position_m.x, site.position_m.y, site.position_m.z]) - [0, 0, .045]
        assert not np.all(np.abs(local) <= [.002, .003, .005]), "An average of free rays is not necessarily free"


def test_continuous_joint_sampling_bounds_are_not_declared_position_limits(tmp_path):
    tree = ET.parse(FIXTURES / "rigid_tool/robot.urdf")
    node = tree.getroot().find("joint[@name='position_x']")
    node.set("type", "continuous")
    node.find("axis").set("xyz", "0 1 0")
    for name in ("lower", "upper"):
        node.find("limit").attrib.pop(name)
    path = tmp_path / "robot.urdf"
    path.write_text(ET.tostring(tree.getroot(), encoding="unicode"), encoding="utf-8", newline="\n")
    body = ingest_capability_body(path)
    renamed = next(k for k, v in body.manifest.joint_aliases.items() if v == "position_x")
    fact = next(f for f in body.manifest.facts if f.fact_id == "joint." + renamed)
    assert fact.basis == "simulation_assumption"
    assert fact.value["source_position_range"] is None
    assert fact.value["measurement_interval_is_assumed"]
    assert not fact.value["position_is_bounded"]


def test_archived_inspection_replays_without_source_and_detects_tampering(tmp_path):
    from rigby_general.capabilities.inspection import capture_inspection, verify_inspection

    destination = tmp_path / "evidence"
    capture_inspection(FIXTURES / "three_digits/robot.urdf", destination)
    verified = verify_inspection(destination)
    assert verified["kinematic_samples_replayed"] == 128
    assert verified["site_replay_max_error_m"] == 0
    assert not verified["physics_replayed"]
    path = destination / "canonical.urdf"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Changed inspection payload"):
        verify_inspection(destination)


@pytest.mark.parametrize("name", ("iiwa7", "kuka_lwr", "so101"))
def test_three_available_third_party_descriptions_have_explicit_assumptions(name, third_party):
    body = ingest_capability_body(third_party / name / "robot.urdf")
    assert body.robot.integrity.passed
    assert body.manifest.assets
    assert body.manifest.facts and body.manifest.limitations
    assert not any(c.enabled for c in body.manifest.capabilities if c.capability in {"physical_grasp_transport", "dynamic_balance", "rolling", "suction"})


def test_source_mimic_is_refused_instead_of_silently_replaced_by_independent_motors(third_party):
    with pytest.raises(ModelIngestError) as error:
        ingest_capability_body(third_party / "panda/robot.urdf")
    assert error.value.code == GeneralFailureCode.UNSUPPORTED_COUPLING
    assert error.value.details["model_invalid"] is False
