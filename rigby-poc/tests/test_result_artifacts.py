from __future__ import annotations

import json
from pathlib import Path

from evals.glb import check_glb
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, PlanRequest, default_scene
from rigby_poc.planner import OfflinePlanner
from rigby_poc.store import ResultStore


ROOT = Path(__file__).resolve().parents[1]


def test_representative_persisted_results_are_complete_and_roundtrip(
    tmp_path: Path,
) -> None:
    profile = json.loads((ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json").read_text(encoding="utf-8"))
    aliases = {source: canonical for canonical, source in profile["bone_map"].items()}
    scene = default_scene()
    store = ResultStore(tmp_path / "results")
    for prompt in (
        "Throw up a hang-ten sign with your right hand.",
        "Step over the hurdle with your right foot.",
    ):
        program = OfflinePlanner().plan(
            PlanRequest(text=prompt, scene=scene, provider="offline")
        ).program
        request = CompileRequest(scene=scene, program=program, persist=True)
        clip = compile_motion(request)
        assert clip.success, clip.failure
        store.persist(request, clip)

    result_dirs = sorted(
        path
        for path in (tmp_path / "results").glob(
            "[0-9][0-9][0-9][0-9][0-9][0-9]-*"
        )
        if path.is_dir()
    )
    assert len(result_dirs) == 2
    required = {"request.json", "scene.json", "program.json", "clip.json", "metrics.json", "provenance.json"}
    complete_provenance = 0
    successful_exports = 0
    for result_dir in result_dirs:
        assert required <= {path.name for path in result_dir.iterdir()}
        clip = json.loads((result_dir / "clip.json").read_text(encoding="utf-8"))
        animation = result_dir / "animation.glb"
        assert clip.get("success")
        assert animation.is_file(), f"{result_dir.name} is a successful clip without a GLB"
        check = check_glb(
            animation.read_bytes(), clip,
            translation_tolerance=0.0001, rotation_tolerance=0.0001,
            node_aliases=aliases,
        )
        assert check.valid
        assert check.compared_samples > 0
        transform_failures = [failure for failure in check.failures if "incomplete Rigby provenance" not in failure]
        assert transform_failures == []
        complete_provenance += int(check.has_provenance)
        successful_exports += 1
    assert successful_exports == 2
    assert complete_provenance == 2
