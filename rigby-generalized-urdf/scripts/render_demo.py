"""Render what a robot actually did, from a plain-language prompt.

Renders the *simulated* trace rather than the planned trajectory. That
distinction matters: the plan is what was asked for and the trace is what the
robot did, and only the second is evidence. Every certified motion has both, and
showing the plan would quietly hide exactly the failures the gates exist to
catch.

Two camera roles per clip, following the v1 demo convention: an orbit view that
shows the whole arm and where it went, and a closer task view of the effector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from rigby_general.config import GeneralSettings
from rigby_general.pipeline import ingest_robot
from rigby_general.primitives import PrimitiveLibrary
from rigby_general.run import answer
from rigby_general.schema.inventory import load_inventory


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "media"

PANEL_WIDTH = 480
PANEL_HEIGHT = 360
TARGET_FPS = 10
MAX_FRAMES = 90


@dataclass(frozen=True, slots=True)
class CameraRole:
    name: str
    azimuth: float
    elevation: float
    distance_scale: float


CAMERA_ROLES = (
    CameraRole("orbit", azimuth=135.0, elevation=-20.0, distance_scale=2.6),
    CameraRole("task", azimuth=55.0, elevation=-12.0, distance_scale=1.5),
)


def _camera(
    model: mujoco.MjModel, role: CameraRole, centre: np.ndarray, reach: float
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = centre
    camera.distance = role.distance_scale * max(reach, 0.2)
    camera.azimuth = role.azimuth
    camera.elevation = role.elevation
    return camera


def render_trace(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    times: np.ndarray,
    *,
    centre: np.ndarray,
    reach: float,
) -> list[Image.Image]:
    # A trace with no samples has nothing to render. That happens for real --
    # a stative certified in zero steps, or an attempt refused before the first
    # one -- and it is not an error, so return no frames rather than crashing
    # the whole build on the last robot in the list.
    if len(times) == 0 or len(qpos) == 0:
        return []

    duration = float(times[-1]) or 1.0
    wanted = min(MAX_FRAMES, max(8, int(duration * TARGET_FPS)))
    indices = np.linspace(0, len(qpos) - 1, wanted).astype(int)

    data = mujoco.MjData(model)
    frames: list[Image.Image] = []

    with mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH) as renderer:
        for index in indices:
            data.qpos[:] = qpos[index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            panels = []
            for role in CAMERA_ROLES:
                renderer.update_scene(data, camera=_camera(model, role, centre, reach))
                panels.append(np.asarray(renderer.render()))

            frames.append(Image.fromarray(np.concatenate(panels, axis=1)))
    return frames


def write_gif(frames: list[Image.Image], path: Path) -> bool:
    """Write the clip, or report that there was nothing to write.

    A trace with no samples produces no frames, which is a real outcome rather
    than an error -- so say there is no clip instead of writing a broken one or
    taking the whole build down on the last robot in the list.
    """

    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = [
        frame.convert("P", palette=Image.ADAPTIVE, colors=96) for frame in frames
    ]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / TARGET_FPS),
        loop=0,
        optimize=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot")
    parser.add_argument("prompt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=ZOO_ROOT)
    parser.add_argument("--library", type=Path, default=None)
    arguments = parser.parse_args()

    settings = GeneralSettings.from_env()
    library = PrimitiveLibrary(arguments.library or Path("robots"))
    inventory = load_inventory()

    robot = ingest_robot(
        arguments.root / arguments.robot / "robot.urdf", robot_id=arguments.robot
    )
    records = library.load(arguments.robot)
    failures = library.load_failures(arguments.robot)
    if not records:
        print(f"no baked library for {arguments.robot}; run scripts/bake_robot.py first")
        return 1

    result = answer(
        arguments.prompt,
        robot.manifest,
        robot.finalized.model,
        inventory,
        records,
        failures,
    )
    print(json.dumps(result.summary(), indent=2))
    if not result.accepted:
        return 2

    scale = robot.morphology.scale
    centre = np.array(
        [
            scale.workspace_centroid_m.x,
            scale.workspace_centroid_m.y,
            scale.workspace_centroid_m.z,
        ]
    )
    trace = result.certification.trace
    frames = render_trace(
        robot.finalized.model,
        trace.qpos,
        trace.times_s,
        centre=centre,
        reach=scale.reach_radius_m,
    )

    slug = "-".join(arguments.prompt.lower().split())[:48].strip("-")
    out = arguments.out or (DEFAULT_OUT / f"{arguments.robot}-{slug}.gif")
    write_gif(frames, out)
    print(f"\nwrote {out}  ({len(frames)} frames, {result.duration_s:.1f}s of motion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
