"""Content-addressed staging for certified object-pack scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import (
    ArtifactRefV1,
    CollisionModel,
    CoordinateFrame,
    Pose,
    Quaternion,
    QuaternionConvention,
    QuaternionMeaning,
    QuaternionOrder,
    RigAssetManifestV1,
    SceneAffordanceV1,
    SceneJointV2,
    SceneManifestV2,
    SceneObjectV2,
    SceneStatePredicateV1,
    SupportSurfaceV2,
    Vec3,
)
from ..rigging.canonical_human import xml_for_profile
from .compiler import CompiledScene, compile_scene
from .loader import PACK_ROOT, load_object_pack
from .models import CollisionKind


@dataclass(frozen=True, slots=True)
class StagedScene:
    manifest: SceneManifestV2
    reference: ArtifactRefV1
    compiled: CompiledScene


def _vec3(values: tuple[float, float, float]) -> Vec3:
    return Vec3(x=values[0], y=values[1], z=values[2])


def _pose(
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> Pose:
    return Pose(
        position=_vec3(position),
        rotation=Quaternion(
            values=quaternion_wxyz,
            convention=QuaternionConvention(
                order=QuaternionOrder.WXYZ,
                frame=CoordinateFrame.WORLD,
                meaning=QuaternionMeaning.ABSOLUTE,
            ),
        ),
    )


def stage_object_pack_scene(
    pack_id: str,
    *,
    profile: str,
    rig_reference: ArtifactRefV1,
    artifacts: ContentAddressedArtifactStore,
    pack_root: Path = PACK_ROOT,
) -> StagedScene:
    """Compile and stage a scene whose rig manifest is already in the same CAS."""

    rig = RigAssetManifestV1.model_validate_json(artifacts.read_bytes(rig_reference))
    if rig.body_size_profile != profile:
        raise ValueError(
            f"Rig profile {rig.body_size_profile!r} does not match scene profile {profile!r}"
        )
    canonical_xml = xml_for_profile(profile)
    if artifacts.read_bytes(rig.mjcf) != canonical_xml.encode("utf-8"):
        raise ValueError("Scene compiler requires the matching canonical rig MJCF artifact")

    pack = load_object_pack(pack_id, root=pack_root)
    compiled = compile_scene(pack, profile=profile, pack_root=pack_root)
    pack_reference = artifacts.put_json(
        pack, filename=f"{pack.pack_id}-object-pack.json"
    )
    mjcf_reference = artifacts.put_bytes(
        compiled.xml.encode("utf-8"),
        media_type="application/mjcf+xml",
        filename=f"{pack.pack_id}-{profile}.xml",
    )
    mjz_reference = artifacts.put_bytes(
        compiled.mjz_bytes,
        media_type="application/vnd.mujoco.mjz",
        filename=f"{pack.pack_id}-{profile}.mjz",
    )
    if mjcf_reference.sha256 != compiled.xml_sha256:
        raise RuntimeError("Staged MJCF hash differs from compiled scene")
    if mjz_reference.sha256 != compiled.mjz_sha256:
        raise RuntimeError("Staged MJZ hash differs from compiled scene")

    objects = tuple(
        SceneObjectV2(
            object_id=obj.object_id,
            asset=pack_reference,
            pose=_pose(obj.pos, obj.quat_wxyz),
            dynamic=obj.dynamic,
            articulated=obj.articulated,
            mass_kg=obj.mass_kg,
            collision_model=(
                CollisionModel.CONVEX_DECOMPOSITION
                if any(
                    part.collision_kind is CollisionKind.CONVEX_DECOMPOSITION
                    for part in obj.parts
                )
                else CollisionModel.PRIMITIVE
            ),
            joints=tuple(
                SceneJointV2(
                    name=part.joint.name,
                    kind=part.joint.type,
                    axis=_vec3(part.joint.axis),
                    range=part.joint.range,
                )
                for part in obj.parts
                if part.joint is not None
            ),
            affordances=tuple(
                f"{obj.object_id}.{value.name}" for value in obj.affordances
            ),
            material=obj.material.name,
        )
        for obj in pack.objects
    )
    supports = tuple(
        SupportSurfaceV2(
            surface_id=support.support_id,
            pose=_pose(support.pos, (1.0, 0.0, 0.0, 0.0)),
            size_m=_vec3(support.size),
            friction=support.material.friction,
        )
        for support in pack.supports
    )
    affordances = tuple(
        SceneAffordanceV1(
            name=name,
            kind=str(value["kind"]),
            site=str(value["site"]),
            allowed_effectors=tuple(str(item) for item in value["allowed_effectors"]),
        )
        for name, value in sorted(compiled.affordances.items())
    )
    predicates = tuple(
        SceneStatePredicateV1.model_validate(value)
        for value in compiled.state_predicates
    )
    effector_geoms = {
        "left_palm": {
            collider.name
            for collider in rig.colliders
            if collider.name.startswith("left_")
            and any(
                token in collider.name
                for token in ("palm", "thumb", "index", "middle", "ring", "little")
            )
        },
        "right_palm": {
            collider.name
            for collider in rig.colliders
            if collider.name.startswith("right_")
            and any(
                token in collider.name
                for token in ("palm", "thumb", "index", "middle", "ring", "little")
            )
        },
    }
    effector_geoms["both_palms"] = (
        effector_geoms["left_palm"] | effector_geoms["right_palm"]
    )
    allowed_pairs: set[tuple[str, str]] = {
        tuple(sorted(("ground", "left_foot_collision"))),
        tuple(sorted(("ground", "right_foot_collision"))),
    }
    support_geoms = {f"support__{support.support_id}" for support in pack.supports}
    for obj in pack.objects:
        object_geoms = {
            f"obj__{obj.object_id}__{part.part_id}__{geom.name}"
            for part in obj.parts
            for geom in part.geoms
        }
        for geom in object_geoms:
            allowed_pairs.add(tuple(sorted((geom, "ground"))))
            allowed_pairs.update(tuple(sorted((geom, support))) for support in support_geoms)
        for affordance in obj.affordances:
            for effector in affordance.allowed_effectors:
                allowed_pairs.update(
                    tuple(sorted((human_geom, object_geom)))
                    for human_geom in effector_geoms[str(effector)]
                    for object_geom in object_geoms
                )
    manifest = SceneManifestV2(
        scene_id=f"rigby-{pack.pack_id}-{profile}",
        rig_asset_hash=rig_reference.sha256,
        objects=objects,
        supports=supports,
        pack_id=pack.pack_id,
        compiled_mjcf=mjcf_reference,
        compiled_mjz=mjz_reference,
        semantic_sites=compiled.semantic_sites,
        affordances=affordances,
        state_predicates=predicates,
        allowed_contact_pairs=tuple(sorted(allowed_pairs)),
        metadata={
            "object_pack_artifact": pack_reference.model_dump(mode="json"),
            "scene_compiler": "rigby_v2.scenes.compile_scene",
            "profile": profile,
        },
    )
    reference = artifacts.put_json(
        manifest, filename=f"{pack.pack_id}-{profile}-scene-manifest.json"
    )
    return StagedScene(manifest=manifest, reference=reference, compiled=compiled)
