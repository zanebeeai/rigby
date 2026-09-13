"""Same resolved world across bodies, with adversarial changes to real models."""

from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import pytest
from rigby_core.hashing import content_hash

from rigby_general.benchmark.contracts import BenchmarkWorldV1, RootGoalV1, RegionV1
from rigby_general.benchmark.world import (
    BenchmarkBodyManifestV1, BodyCameraMountV1, BenchmarkRefusal, compile_world,
    normalize_world, resolved_world_manifest, validate_binding, verify_world,
)
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import load_environment

ZOO = Path(__file__).resolve().parents[1] / "assets/general/zoo"


@pytest.fixture(scope="module")
def world():
    return BenchmarkWorldV1(environment=load_environment("desk_bench"), goal=RootGoalV1(
        object_names=("cube",), target=RegionV1(minimum_m=(0.2, -0.08, 0.05), maximum_m=(0.35, 0.08, 0.2))))


@pytest.fixture(scope="module")
def robots():
    return [ingest_robot(path) for path in sorted(ZOO.glob("*/robot.urdf"))]


def test_same_compiled_world_survives_all_six_body_swaps(world, robots):
    assert len(robots) == 6
    expected = compile_world(world).world_sha256
    compiled = [compile_world(world, body=BenchmarkBodyManifestV1(robot=r.manifest), robot_xml=r.finalized.mjcf_xml) for r in robots]
    assert {r.world_sha256 for r in compiled} == {expected}
    assert len({r.compiled_model_sha256 for r in compiled}) == 6
    for r in compiled:
        verify_world(r, expected)


@pytest.mark.parametrize("mutation", ["geometry", "mass", "inertia", "placement", "friction", "contact", "lighting", "camera", "gravity", "timestep", "initial_object_pose"])
def test_real_compiled_mutations_fail_world_validation(world, mutation):
    c = compile_world(world)
    m = c.model
    geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "world/cube_geom")
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "world/cube")
    if mutation == "geometry": m.geom_size[geom, 0] += 0.001
    elif mutation == "mass": m.body_mass[body] *= 1.1
    elif mutation == "inertia": m.body_inertia[body, 0] *= 1.1
    elif mutation == "placement": m.body_pos[body, 0] += 0.001
    elif mutation == "friction": m.geom_friction[geom, 0] *= 1.1
    elif mutation == "contact": m.geom_conaffinity[geom] = 0
    elif mutation == "lighting": m.light_ambient[0, 0] += 0.01
    elif mutation == "camera": m.cam_fovy[0] += 1
    elif mutation == "gravity": m.opt.gravity[2] -= 0.01
    elif mutation == "timestep": m.opt.timestep *= 0.5
    elif mutation == "initial_object_pose": m.qpos0[0] += 0.01
    with pytest.raises(BenchmarkRefusal) as error:
        verify_world(c, c.world_sha256)
    assert error.value.code == "world_changed"


@pytest.mark.parametrize("field", ["goal", "start_zone", "sensor_policy"])
def test_non_geometry_contract_changes_also_change_world_hash(world, field):
    c = compile_world(world)
    raw = world.model_dump(mode="json")
    if field == "goal": raw[field]["dwell_s"] += 1
    elif field == "start_zone": raw[field]["position_tolerance_m"] += 0.001
    else: raw[field]["depth"] = True
    changed = BenchmarkWorldV1.model_validate(raw)
    assert content_hash(resolved_world_manifest(c.model, changed)) != c.world_sha256


def test_body_sensor_mount_is_pinned_separately_from_world(world, robots):
    r = robots[0]
    mount = BodyCameraMountV1(channel="ego", body=r.morphology.base_body,
                             position_m=(0, 0, 0.15), quaternion_wxyz=(1, 0, 0, 0), fovy_degrees=55,
                             provenance="declared_simulated_sensor")
    body = BenchmarkBodyManifestV1(robot=r.manifest, camera_mounts=(mount,))
    a = compile_world(world, body=body, robot_xml=r.finalized.mjcf_xml)
    moved = mount.model_copy(update={"position_m": (0, 0, 0.2)})
    b = compile_world(world, body=body.model_copy(update={"camera_mounts": (moved,)}), robot_xml=r.finalized.mjcf_xml)
    assert a.world_sha256 == b.world_sha256
    assert content_hash(a.body_manifest) != content_hash(b.body_manifest)
    assert a.compiled_model_sha256 != b.compiled_model_sha256
    sensor = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot/ego")
    a.model.cam_pos[sensor, 0] += 0.01
    with pytest.raises(BenchmarkRefusal) as error:
        verify_world(a, a.world_sha256)
    assert error.value.code == "body_or_model_changed"


def test_robot_cannot_import_its_own_world_fixture(world, robots):
    r = robots[0]
    xml = ET.fromstring(r.finalized.mjcf_xml)
    ET.SubElement(xml.find("worldbody"), "geom", type="plane", size="2 2 0.1")
    with pytest.raises(BenchmarkRefusal) as error:
        compile_world(world, body=BenchmarkBodyManifestV1(robot=r.manifest), robot_xml=ET.tostring(xml, encoding="unicode"))
    assert error.value.code == "robot_contains_world"


def test_strict_mode_rejects_fitting_and_semantic_substitution(world):
    normalized, record = normalize_world(world, 0.5)
    semantics = {"predicate": "TransferObject", "object": "cube"}
    validate_binding(mode="capability_normalized", authored=world, resolved=normalized,
                     requested_semantics=semantics, bound_semantics=semantics, normalization=record)
    with pytest.raises(BenchmarkRefusal) as error:
        validate_binding(mode="strict_fixed_world", authored=world, resolved=normalized,
                         requested_semantics=semantics, bound_semantics=semantics, normalization=record)
    assert error.value.code == "world_fitting_forbidden"
    with pytest.raises(BenchmarkRefusal) as error:
        validate_binding(mode="strict_fixed_world", authored=world, resolved=world,
                         requested_semantics=semantics, bound_semantics={"predicate": "Reach"})
    assert error.value.code == "semantic_substitution"
    modified = normalized.model_dump(mode="json")
    modified["environment"]["objects"][0]["mass_kg"] *= 0.5
    with pytest.raises(BenchmarkRefusal) as error:
        validate_binding(mode="capability_normalized", authored=world, resolved=BenchmarkWorldV1.model_validate(modified),
                         requested_semantics=semantics, bound_semantics=semantics, normalization=record)
    assert error.value.code == "undocumented_normalization"


def test_robot_mount_cannot_change_after_registration(world, robots):
    r = robots[0]
    c = compile_world(world, body=BenchmarkBodyManifestV1(robot=r.manifest), robot_xml=r.finalized.mjcf_xml)
    root = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, "robot/"+r.morphology.base_body)
    c.model.body_pos[root, 0] += 0.05
    with pytest.raises(BenchmarkRefusal) as error:
        verify_world(c, c.world_sha256)
    assert error.value.code == "body_or_model_changed"
