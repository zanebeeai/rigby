"""Derive deterministic certification policy from persisted manifests."""

from __future__ import annotations

from rigby_core.contracts import RigAssetManifestV1, SceneManifestV2

from .models import CertificationPolicy, ContactPair, SupportFoot


def policy_from_manifests(
    rig: RigAssetManifestV1,
    scene: SceneManifestV2,
    *,
    require_export_reimport: bool = True,
) -> CertificationPolicy:
    foot_specs = []
    for body in ("left_foot", "right_foot"):
        geoms = frozenset(
            collider.name for collider in rig.colliders if collider.body == body
        )
        if not geoms:
            raise ValueError(f"Rig manifest does not declare collision geometry for {body}")
        foot_specs.append(SupportFoot(body_name=body, geom_names=geoms))
    actuator_effort = {
        actuator: dof.effort_limit
        for actuator, dof in zip(rig.actuator_order, rig.dofs, strict=True)
    }
    return CertificationPolicy(
        pelvis_body="pelvis",
        torso_body="torso",
        support_feet=tuple(foot_specs),
        ground_geom_names=frozenset({"ground"}),
        allowed_contact_pairs=frozenset(
            ContactPair.of(first, second)
            for first, second in scene.allowed_contact_pairs
        ),
        joint_velocity_limits={dof.joint: dof.velocity_limit for dof in rig.dofs},
        joint_acceleration_limits={
            dof.joint: dof.acceleration_limit for dof in rig.dofs
        },
        actuator_effort_limits=actuator_effort,
        joint_effort_limits={dof.joint: dof.effort_limit for dof in rig.dofs},
        joint_power_limits={
            dof.joint: dof.effort_limit * dof.velocity_limit for dof in rig.dofs
        },
        require_export_reimport=require_export_reimport,
    )
