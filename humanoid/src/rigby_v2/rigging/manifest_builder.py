from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import (
    ArtifactRefV1,
    BodyInertiaSpecV1,
    ColliderSpecV1,
    DofSpecV1,
    RigAssetManifestV1,
    SiteSpecV1,
    Vec3,
)
from .canonical_human import xml_for_profile
from .mapping import PROJECT_ROOT, load_rig_manifest


_GEOM_SHAPES = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_MESH: "convex_mesh",
}


@dataclass(frozen=True, slots=True)
class StagedCanonicalRig:
    manifest: RigAssetManifestV1
    reference: ArtifactRefV1


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, kind, index)
    if value is None:
        raise ValueError(f"Unnamed {kind.name} at index {index}")
    return value


def _site_semantic(site_name: str, semantic_name: str) -> str:
    if semantic_name.endswith("Tip"):
        return "fingertip"
    if semantic_name.endswith("Palm"):
        return "palm"
    if semantic_name.endswith("Foot"):
        return "foot"
    if semantic_name == "gaze":
        return "gaze"
    return "task"


def build_canonical_rig_manifest(
    profile: str = "medium",
    *,
    artifacts: ContentAddressedArtifactStore,
    include_visual_asset: bool = True,
) -> RigAssetManifestV1:
    xml = xml_for_profile(profile)
    model = mujoco.MjModel.from_xml_string(xml)
    mapping = load_rig_manifest()
    mjcf = artifacts.put_bytes(
        xml.encode("utf-8"),
        media_type="application/mjcf+xml",
        filename=f"canonical-human-{profile}.xml",
    )
    visual: ArtifactRefV1 | None = None
    if include_visual_asset:
        visual_path = PROJECT_ROOT / "assets" / "models" / "human-male.glb"
        visual = artifacts.put_file(visual_path, media_type="model/gltf-binary")

    dofs: list[DofSpecV1] = []
    actuator_order: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_order.append(_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
        minimum, maximum = (float(value) for value in model.jnt_range[joint_id])
        control_limit = max(
            abs(float(value)) for value in model.actuator_ctrlrange[actuator_id]
        )
        is_digit = any(
            token in joint_name
            for token in ("thumb", "index", "middle", "ring", "little")
        )
        dofs.append(
            DofSpecV1(
                name=joint_name,
                joint=joint_name,
                minimum=minimum,
                maximum=maximum,
                velocity_limit=30.0 if is_digit else 25.0,
                acceleration_limit=4_000.0 if is_digit else 8_000.0,
                effort_limit=control_limit,
            )
        )

    bodies = tuple(
        BodyInertiaSpecV1(
            name=_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            mass_kg=float(model.body_mass[body_id]),
            center_of_mass_m=Vec3(
                x=float(model.body_ipos[body_id, 0]),
                y=float(model.body_ipos[body_id, 1]),
                z=float(model.body_ipos[body_id, 2]),
            ),
            diagonal_inertia_kg_m2=Vec3(
                x=float(model.body_inertia[body_id, 0]),
                y=float(model.body_inertia[body_id, 1]),
                z=float(model.body_inertia[body_id, 2]),
            ),
        )
        for body_id in range(1, model.nbody)
    )

    colliders: list[ColliderSpecV1] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
        if body_id == 0 or geom_type not in _GEOM_SHAPES:
            continue
        colliders.append(
            ColliderSpecV1(
                name=_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                body=_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                group="human",
                shape=_GEOM_SHAPES[geom_type],
            )
        )

    semantic_by_site = {
        site: semantic for semantic, site in mapping.semantic_sites.items()
    }
    sites: list[SiteSpecV1] = []
    for site_id in range(model.nsite):
        site_name = _name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        semantic_name = semantic_by_site.get(site_name, site_name)
        sites.append(
            SiteSpecV1(
                name=site_name,
                body=_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site_id])
                ),
                semantic=_site_semantic(site_name, semantic_name),
            )
        )

    xml_root = ET.fromstring(xml)
    exclusions = tuple(
        (element.attrib["body1"], element.attrib["body2"])
        for element in xml_root.findall("./contact/exclude")
    )
    visual_body_map: dict[str, str] = {}
    for bone, joints in mapping.skeleton_to_dofs.items():
        if joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])
            visual_body_map[bone] = _name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id])
            )
        elif bone == "head":
            visual_body_map[bone] = "head"
    source_profile = json.loads(
        (
            PROJECT_ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json"
        ).read_text(encoding="utf-8")
    )
    return RigAssetManifestV1(
        rig_id=f"rigby-canonical-human-{profile}",
        mjcf=mjcf,
        visual_asset=visual,
        body_size_profile=profile,  # type: ignore[arg-type]
        dofs=tuple(dofs),
        actuator_order=tuple(actuator_order),
        colliders=tuple(colliders),
        bodies=bodies,
        sites=tuple(sites),
        rest_qpos=tuple(float(value) for value in model.qpos0),
        visual_skeleton_map={
            bone: tuple(joints) for bone, joints in mapping.skeleton_to_dofs.items()
        },
        visual_body_map=visual_body_map,
        visual_node_map={
            bone: str(source_profile["bone_map"][bone])
            for bone in (*mapping.skeleton_to_dofs, *mapping.fixed_visual_bones)
        },
        fixed_visual_bones=tuple(mapping.fixed_visual_bones),
        adjacent_collision_exclusions=exclusions,
        asset_licenses={
            "visual_asset": "CC0-1.0",
            "canonical_mjcf": "Rigby project generated asset",
        },
    )


def stage_canonical_rig(
    profile: str = "medium",
    *,
    artifacts: ContentAddressedArtifactStore,
) -> StagedCanonicalRig:
    manifest = build_canonical_rig_manifest(profile, artifacts=artifacts)
    reference = artifacts.put_json(
        manifest, filename=f"canonical-human-{profile}-manifest.json"
    )
    return StagedCanonicalRig(manifest=manifest, reference=reference)
