"""Portable body-inspection evidence, explicitly separate from physical trials."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rigby_core.hashing import hash_file

from .contracts import BodyCapabilityManifestV1
from .intake import ingest_capability_body


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def capture_inspection(source: Path, destination: Path, *, title: str | None = None) -> dict:
    """Save the measured model and sampled kinematics; never step physics."""
    if destination.exists():
        raise ValueError("Choose a new inspection destination; evidence is not overwritten")
    body = ingest_capability_body(source)
    destination.mkdir(parents=True)
    write_json(destination / "body-manifest.json", body.manifest.model_dump(mode="json"))
    write_json(destination / "morphology-measurements.json", body.robot.morphology.model_dump(mode="json"))
    (destination / "canonical.urdf").write_text(body.canonical.xml, encoding="utf-8", newline="\n")
    (destination / "runtime.xml").write_text(body.robot.mjcf_xml, encoding="utf-8", newline="\n")
    model = body.robot.finalized.model
    mujoco.mj_saveModel(model, str(destination / "model.mjb"))
    data = mujoco.MjData(model)
    # These samples visualize the approximate kinematic envelope, not a
    # collision-free, dynamically reachable or physically certified workspace.
    rng = np.random.default_rng(20260912)
    samples = np.tile(np.asarray(body.robot.manifest.rest_qpos), (128, 1))
    for index in range(model.njnt):
        low, high = model.jnt_range[index]
        if not model.jnt_limited[index]:
            low, high = -np.pi, np.pi
        samples[:, model.jnt_qposadr[index]] = rng.uniform(low, high, len(samples))
    sites = []
    for qpos in samples:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        sites.append(data.site_xpos.copy())
    np.savez_compressed(destination / "inspection.npz", qpos=samples, sites=np.asarray(sites),
                        rest_qpos=np.asarray(body.robot.manifest.rest_qpos))
    metadata = {"schema": "body.inspection.v1", "title": title or source.stem,
        "mujoco_version": mujoco.__version__, "capability_sha256": body.manifest.capability_hash(),
        "sample_seed": 20260912, "sample_count": len(samples), "physics_steps": 0,
        "workspace_scope": "Joint-limit sampling; no collision, contact or dynamics certificate",
        "media_scope": "Camera orbit of a static rest pose with derived sites, chains and sampled kinematic workspace",
        "fps": 10, "frame_count": 60, "source_path_hint": source.name}
    write_json(destination / "inspection.json", metadata)
    files = {p.name: hash_file(p) for p in sorted(destination.iterdir()) if p.is_file()}
    write_json(destination / "payloads.json", files)
    return metadata


def verify_inspection(root: Path) -> dict:
    recorded = json.loads((root / "payloads.json").read_bytes())
    for name, digest in recorded.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or hash_file(path) != digest:
            raise ValueError(f"Changed inspection payload: {name}")
    manifest = BodyCapabilityManifestV1.model_validate_json((root / "body-manifest.json").read_bytes())
    metadata = json.loads((root / "inspection.json").read_bytes())
    if manifest.capability_hash() != metadata["capability_sha256"]:
        raise ValueError("Capability hash does not match the inspection")
    if metadata["mujoco_version"] != mujoco.__version__:
        raise ValueError("Use the recorded MuJoCo version to load its binary model")
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(root / "inspection.npz", allow_pickle=False) as samples:
        error = 0.0
        for qpos, expected in zip(samples["qpos"], samples["sites"], strict=True):
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            error = max(error, float(np.max(np.abs(data.site_xpos - expected))))
    if error != 0:
        raise ValueError(f"Recorded kinematics do not replay exactly: {error}")
    return {"payloads_verified": len(recorded), "site_replay_max_error_m": error,
            "physics_replayed": False, "kinematic_samples_replayed": metadata["sample_count"]}


PALETTE = ((0.1, 0.5, 0.95, 1.0), (0.2, 0.8, 0.4, 1.0), (0.75, 0.35, 0.8, 1.0))


def _dot(scene, position, radius, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                       np.asarray(position), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def render_inspection(root: Path, destination: Path) -> dict:
    """Re-render the archived inspection without source assets or model calls."""
    verified = verify_inspection(root)
    if destination.exists():
        raise ValueError("Choose a new render destination")
    metadata = json.loads((root / "inspection.json").read_bytes())
    manifest = BodyCapabilityManifestV1.model_validate_json((root / "body-manifest.json").read_bytes())
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(root / "inspection.npz", allow_pickle=False) as samples:
        data.qpos[:] = samples["rest_qpos"]
        cloud = samples["sites"].copy()
    mujoco.mj_forward(model, data)  # Visualization reconstruction only.
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 400
    model.vis.headlight.ambient[:] = 0.7
    model.vis.headlight.diffuse[:] = 0.8
    low, high = data.xpos[1:].min(axis=0), data.xpos[1:].max(axis=0)
    reach = max(float(np.linalg.norm(high-low)), manifest.robot.morphology.scale.reach_radius_m, 0.12)
    model.site_rgba[:] = (1., .5, .05, 1.)
    model.site_size[:] = reach * .008
    model.geom_rgba[:, 3] = .55
    centre = (low + high) / 2
    camera = mujoco.MjvCamera()
    camera.lookat[:] = centre
    camera.distance = reach * 2.3
    camera.elevation = -18
    chains = manifest.robot.morphology.chains
    tip_ids = []
    for chain in chains:
        effector_sites = {name for e in manifest.robot.morphology.effectors if e.chain_id == chain.chain_id for name in e.site_names}
        selected = next((s for s in manifest.robot.sites if s.name in effector_sites and s.semantic.value == "tip"), None)
        if selected:
            tip_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, selected.name))
    frames, frame_map = [], []
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    with mujoco.Renderer(model, height=400, width=640) as renderer:
        for index in range(metadata["frame_count"]):
            camera.azimuth = 45 + index * 360 / metadata["frame_count"]
            renderer.update_scene(data, camera=camera)
            scene = renderer.scene
            for number, chain in enumerate(chains):
                color = PALETTE[number % len(PALETTE)]
                ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in chain.bodies]
                for a, b in zip(ids[:-1], ids[1:]):
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                       np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, reach * .005, data.xpos[a], data.xpos[b])
                    scene.ngeom += 1
            for number, sid in enumerate(tip_ids):
                color = (*PALETTE[number % len(PALETTE)][:3], .22)
                for point in cloud[::4, sid]:
                    _dot(scene, point, reach * .004, color)
            for point in data.site_xpos:
                _dot(scene, point, reach * .008, (1., .5, .05, 1.))
            canvas = Image.new("RGB", (640, 480), (248, 249, 251))
            canvas.paste(Image.fromarray(renderer.render()), (0, 80))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 7), f"{metadata['title']} | BODY INSPECTION | frame {index+1}/{metadata['frame_count']}", fill="black", font=font)
            draw.text((10, 28), "Static rest pose / orbiting camera / zero physics steps", fill="black", font=small)
            draw.text((10, 45), "Color lines: chains | orange: derived sites | faint dots: kinematic samples", fill="black", font=small)
            draw.text((10, 62), "Physical grasp, payload, balance and mobility remain unverified.", fill="black", font=small)
            frames.append(canvas)
            frame_map.append({"frame": index, "camera_azimuth_degrees": camera.azimuth,
                              "playback_seconds": index / metadata["fps"], "robot_state": "archived_rest_qpos"})
    destination.mkdir(parents=True)
    frames[0].save(destination / "poster.png")
    frames[0].save(destination / "inspection.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // metadata["fps"], loop=0, optimize=False)
    write_json(destination / "frames.json", frame_map)
    write_json(destination / "verification.json", {**verified, "frame_count": len(frames),
               "fps": metadata["fps"], "renderer_sha256": hash_file(Path(__file__))})
    return {**verified, "frame_count": len(frames), "gif_sha256": hash_file(destination / "inspection.gif")}
