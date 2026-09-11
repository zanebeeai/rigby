"""Build the geometry and motion the 3D viewer plays: one file per robot.

Two kinds of payload, and they cost very different amounts to produce.

*Scenes* are free -- the kinematic tree and its meshes fall straight out of the
compiled model.

*Primitive tracks* are not. Every certified primitive has to be recompiled and
rolled forward to know what the arm actually did, which is the expensive half of
a bake. So this runs as its own step and writes a cache the studio build picks
up if it is there, rather than making every studio rebuild pay for it.

The rollout here is a single pass, not the three the certifier runs. That is a
deliberate and stated difference: certification needs repeats because a
trajectory that certifies once and differs on the next run is not reproducible,
but the viewer is showing one playback, and it already only ever shows
primitives that survived the three-repeat gate at bake time. This is a picture of
a certified motion, not a fresh claim that it certifies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rigby_core.motion.compiler import compile_motion_program
from rigby_core.motion.errors import MotionCompilationError
from rigby_general.contracts import RobotAssetManifestV1  # noqa: F401 - typing only
from rigby_general.contact import probe_grasp
from rigby_general.gates import simulate
from rigby_general.pipeline import ingest_robot
from rigby_general.primitives import PrimitiveLibrary
from rigby_general.viewer import build_scene, sample_track

from rigby_core.contracts import MotionProgramV2


ROOT = Path(__file__).resolve().parents[1]
ROOTS = (
    ROOT / "assets" / "general" / "zoo",
    ROOT / "assets" / "general" / "exotic",
    ROOT / "assets" / "general" / "irl",
)
SAMPLE_HZ = 240


def discover() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            source = directory / "robot.urdf"
            if source.is_file():
                found[directory.name] = source
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", nargs="*", help="robot ids; default is all")
    parser.add_argument("--library", type=Path, default=ROOT / "robots")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "viewer")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="playback samples per primitive; the cost is linear in this",
    )
    parser.add_argument(
        "--scenes-only",
        action="store_true",
        help="skip the expensive rollouts and refresh geometry only",
    )
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    library = PrimitiveLibrary(arguments.library)
    sources = discover()
    wanted = arguments.robot or sorted(sources)

    for robot_id in wanted:
        source = sources.get(robot_id)
        if source is None:
            print(f"{robot_id:<18} no model found")
            continue

        started = time.perf_counter()
        try:
            robot = ingest_robot(source, robot_id=robot_id)
        except Exception as error:  # noqa: BLE001 - a refused robot has no scene
            print(f"{robot_id:<18} not ingestable: {type(error).__name__}")
            continue

        model = robot.finalized.model
        payload: dict[str, object] = {
            "robot_id": robot_id,
            "scene": build_scene(model),
            "rest_qpos": [round(float(v), 7) for v in robot.manifest.rest_qpos],
            "reach_m": round(robot.morphology.scale.reach_radius_m, 4),
            "primitives": [],
        }

        if not arguments.scenes_only:
            payload["primitives"] = _primitive_tracks(
                robot, model, library, robot_id, arguments.max_frames
            )
            # The grasp probe runs on a different model -- the robot plus a block
            # and the plate it stands on -- so it needs its own scene. Without
            # one the probes were the only demos in the studio with no playback
            # at all, which is backwards: a failed grasp is the more informative
            # clip, because you can watch the block get knocked away instead of
            # reading a gate code and guessing.
            probe = _probe_track(robot, arguments.max_frames, source.parent)
            if probe is not None:
                payload["probe"] = probe

        target = arguments.out / f"{robot_id}.json"
        target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        size = target.stat().st_size / 1024
        print(
            f"{robot_id:<18} {len(payload['primitives']):>3} primitive tracks  "
            f"{size:>7.0f} KB  {time.perf_counter() - started:5.1f}s"
        )
    return 0


def _primitive_tracks(
    robot, model, library: PrimitiveLibrary, robot_id: str, max_frames: int
) -> list[dict]:
    tracks: list[dict] = []
    for record in library.load(robot_id):
        try:
            program = MotionProgramV2.model_validate(record.program)
            trajectory = compile_motion_program(
                program, model, robot.manifest, sample_hz=SAMPLE_HZ
            )
            rollout = simulate(
                model, robot.manifest, trajectory, site_name=record.figure_site
            )
        except (MotionCompilationError, ValueError, KeyError) as error:
            # A stored primitive that will not replay is worth saying so about;
            # silently omitting it would make the library look smaller than the
            # bake reported and give no clue why.
            print(f"{'':18} !! {record.entry_id}@{record.remove}: {error}")
            continue

        track = sample_track(
            rollout.times_s, rollout.qpos, max_frames=max_frames
        )
        tracks.append(
            {
                "record_id": record.record_id,
                "entry_id": record.entry_id,
                "remove": record.remove,
                "schema_key": record.schema_key,
                "segment_key": record.segment_key,
                "figure_site": record.figure_site,
                "waypoints": record.waypoints,
                "measurements": record.measurements,
                "track": track,
            }
        )
    return tracks




def _probe_track(robot, max_frames: int, asset_root: Path | None = None) -> dict | None:
    """The grasp probe's own scene and rollout, block included."""

    try:
        outcome = probe_grasp(
            robot.manifest,
            robot.mjcf_xml,
            robot.finalized.model,
            asset_root=asset_root,
        )
    except Exception as error:  # noqa: BLE001 - a robot with no gripper has none
        print(f"{'':18} .. no grasp probe: {type(error).__name__}: {error}")
        return None

    if outcome.result is None or outcome.scene is None:
        # Refused before anything moved -- there is nothing to watch, and the
        # trace already says which stage refused it.
        return None

    scene_model = outcome.scene.model
    return {
        "scene": build_scene(scene_model),
        "rest_qpos": [round(float(v), 7) for v in scene_model.qpos0],
        "accepted": bool(outcome.trace.accepted),
        "block_position_m": [
            round(float(v), 5) for v in outcome.scene.block_position_m
        ],
        "track": sample_track(
            outcome.result.times_s, outcome.result.qpos, max_frames=max_frames
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
