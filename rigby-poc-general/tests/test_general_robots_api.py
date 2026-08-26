"""Uploading a robot, and refusing one."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rigby_general.app import create_app
from rigby_general.config import GeneralSettings
from rigby_general.robots import RobotRegistry, normalize_robot_id


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = GeneralSettings(
        project_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        robot_root=tmp_path / "robots",
    )
    return TestClient(create_app(settings))


def upload(client: TestClient, robot_id: str) -> dict:
    payload = (ZOO_ROOT / robot_id / "robot.urdf").read_bytes()
    response = client.post(
        "/api/v3/robots",
        files={"file": ("robot.urdf", payload, "application/xml")},
        params={"robot_id": robot_id},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_measures_the_robot(client: TestClient) -> None:
    body = upload(client, "zoo_jaw_arm")

    assert body["robot_id"] == "zoo_jaw_arm"
    assert body["morphology_class"] == "fixed_base_arm"
    assert body["dof"] == 8
    assert [e["kind"] for e in body["effectors"]] == ["parallel_jaw"]
    assert body["scale"]["reach_radius_m"] > 1.0


def test_uploaded_robots_are_listed_and_retrievable(client: TestClient) -> None:
    upload(client, "zoo_tool_arm")
    upload(client, "zoo_dual_arm")

    listed = client.get("/api/v3/robots").json()["robots"]
    assert sorted(item["robot_id"] for item in listed) == [
        "zoo_dual_arm",
        "zoo_tool_arm",
    ]

    single = client.get("/api/v3/robots/zoo_dual_arm").json()
    assert single["morphology_class"] == "fixed_base_bimanual"
    assert len(single["effectors"]) == 2


def test_model_xml_carries_the_derived_sites(client: TestClient) -> None:
    upload(client, "zoo_hand_arm")
    xml = client.get("/api/v3/robots/zoo_hand_arm/model.xml").text

    assert "<site" in xml
    assert "palm_grasp_center" in xml


def test_unknown_robot_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v3/robots/not_a_robot").status_code == 404


def test_a_malformed_upload_is_a_typed_refusal(client: TestClient) -> None:
    """Not a 500. The caller is told which rule the file broke."""

    response = client.post(
        "/api/v3/robots",
        files={"file": ("robot.urdf", b"<robot name='x'><link/>", "application/xml")},
        params={"robot_id": "broken"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["failure_code"] == "unreadable_model"


def test_an_external_mesh_reference_is_refused(client: TestClient) -> None:
    """A ``package://`` path resolves through a workspace outside the upload."""

    urdf = b"""<?xml version="1.0"?>
<robot name="escapee">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision><geometry><mesh filename="package://somewhere/else/mesh.stl"/></geometry></collision>
  </link>
</robot>
"""
    response = client.post(
        "/api/v3/robots",
        files={"file": ("robot.urdf", urdf, "application/xml")},
        params={"robot_id": "escapee"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["failure_code"] == "unsafe_asset"
    assert detail["rule"] == "asset.path.external.v1"


# --------------------------------------------------------------------------
# Frame confirmation
# --------------------------------------------------------------------------


def test_frame_starts_unconfirmed_and_can_be_corrected(client: TestClient) -> None:
    body = upload(client, "zoo_jaw_arm")
    assert body["intrinsic_frame"]["source"] == "derived"
    assert body["intrinsic_frame"]["confidence"] == 0.0

    confirmed = client.post(
        "/api/v3/robots/zoo_jaw_arm/frame", json={"front": [0.0, 1.0, 0.0]}
    ).json()

    assert confirmed["intrinsic_frame"]["source"] == "operator_confirmed"
    assert confirmed["intrinsic_frame"]["confidence"] == 1.0
    assert confirmed["intrinsic_frame"]["front"] == [0.0, 1.0, 0.0]


def test_confirmation_survives_a_reload(client: TestClient) -> None:
    upload(client, "zoo_tool_arm")
    client.post("/api/v3/robots/zoo_tool_arm/frame", json={"front": [1.0, 0.0, 0.0]})

    reloaded = client.get("/api/v3/robots/zoo_tool_arm").json()
    assert reloaded["intrinsic_frame"]["source"] == "operator_confirmed"


def test_a_vertical_front_is_refused(client: TestClient) -> None:
    upload(client, "zoo_tool_arm")
    response = client.post(
        "/api/v3/robots/zoo_tool_arm/frame", json={"front": [0.0, 0.0, 1.0]}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Store safety
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Zoo Jaw Arm", "zoo_jaw_arm"),
        ("panda-v2", "panda-v2"),
        ("  UR5e  ", "ur5e"),
    ],
)
def test_robot_ids_are_normalized(raw: str, expected: str) -> None:
    assert normalize_robot_id(raw) == expected


@pytest.mark.parametrize("raw", ["../../etc", "..\\..\\windows", "a/b/c"])
def test_path_separators_are_stripped_not_carried(raw: str) -> None:
    """Ids become directory names, so a traversal must not survive normalization.

    Sanitizing rather than rejecting: an uploaded filename is allowed to be
    untidy. What matters is that nothing resembling a path comes out the far
    side, which the identifier pattern then guarantees.
    """

    normalized = normalize_robot_id(raw)
    assert "/" not in normalized
    assert "\\" not in normalized
    assert ".." not in normalized


@pytest.mark.parametrize("raw", ["...", "/", "", "   ", "!!!"])
def test_ids_that_sanitize_to_nothing_are_refused(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_robot_id(raw)


def test_registry_refuses_a_traversal_id(tmp_path: Path) -> None:
    registry = RobotRegistry(tmp_path)
    with pytest.raises(ValueError, match="unsafe robot id"):
        registry.directory("../escape")
