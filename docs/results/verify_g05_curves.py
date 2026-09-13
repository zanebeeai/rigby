"""Verify the curve diagnostic hashes, references and saved physical traces."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from rigby_core.contracts import MotionProgramV2


ROOT = Path(__file__).resolve().parent


def main():
    root = ROOT / "g05-curves-diagnostic"
    validation = json.loads((ROOT / "g05-curves-validation.json").read_bytes())
    assert hashlib.sha256((root / "payloads.json").read_bytes()).hexdigest() == validation["payload_manifest_sha256"]
    files = json.loads((root / "payloads.json").read_bytes())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "payloads.json"}
    assert actual == set(files)
    for name, expected in files.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    report = json.loads((root / "report.json").read_bytes())
    assert report["source_commit"] == validation["source_commit"]
    assert report["native_clock_commit"] == validation["native_clock_commit"]
    assert not report["goal_G05_complete"] and report["api_calls"] == 0
    script = ROOT.parents[1] / "any-robot/scripts/audit_bounded_joint_curves.py"
    spec = importlib.util.spec_from_file_location("curve_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    intervals = 0
    successes = 0
    for body in report["bodies"]:
        folder = root / body["robot_id"]
        before = MotionProgramV2.model_validate_json((folder / "program-before.json").read_bytes())
        after = MotionProgramV2.model_validate_json((folder / "program-after.json").read_bytes())
        assert before.duration_s == after.duration_s == body["authored_duration_s"]
        assert before.phases == after.phases
        assert [t.keyframes for t in before.tracks] == [t.keyframes for t in after.tracks]
        for label, program in (("before", before), ("after", after)):
            recomputed = module.interval_extrema(program)
            assert recomputed == json.loads((folder / f"extrema-{label}.json").read_bytes())
            if label == "after":
                assert max(r["endpoint_range_excess"] for r in recomputed) < 1e-9
                intervals += len(recomputed)
        assert body["fresh_exact_region_leaves"] == 2 and body["leaf_failures"] == 0
        assert body["duration_scales"] == [1., 1.]
        assert body["region_substitutions"] == 0
        assert body["role_normalized_hash"] == "09191df11fd0af2d0172d92356e8beaabba12206e701ab1684c730ad7fccd1e6"
        assert body["repeats"] == 3 and len(set(body["replay_hashes"])) == 1
        with np.load(folder / "rollout.npz", allow_pickle=False) as arrays:
            assert np.all(np.diff(arrays["time_s"]) > 0)
            assert np.allclose(np.diff(arrays["time_s"]), body["native_timestep_s"], atol=1e-12, rtol=0)
            assert arrays["time_s"][-1] == body["actual_physics_duration_s"]
            assert abs(body["actual_physics_duration_s"]-body["authored_duration_s"]) <= body["native_timestep_s"]+1e-9
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
            payload = b"".join(np.ascontiguousarray(arrays[key], dtype=np.float64).tobytes()
                               for key in ("time_s", "qpos", "qvel", "ctrl"))
            assert hashlib.sha256(payload).hexdigest() == body["replay_hashes"][0]
        if body["accepted"]:
            successes += 1
            assert body["violations"] == []
        else:
            assert body["robot_id"] == "zoo_dual_arm"
            assert body["failure"]["code"] == "deterministic_gate_failed"
            assert {v["code"] for v in body["violations"]} == {"joint_position_limit", "actuator_effort_limit"}
    assert len(report["bodies"]) == 6 and successes == 5
    result = {"payloads_verified": len(files), "scalar_intervals_verified": intervals,
              "native_physics_saved_trace_hashes_verified": 6, "canonical_successes": successes,
              "runtime_failures": 1, "G05_complete": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
