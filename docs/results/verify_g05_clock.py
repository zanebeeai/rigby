"""Check the published clock diagnostic without running bakes or model calls."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    root = ROOT / "g05-clock-diagnostic"
    validation = json.loads((ROOT / "g05-clock-validation.json").read_bytes())
    assert hashlib.sha256((root / "payloads.json").read_bytes()).hexdigest() == validation["payload_manifest_sha256"]
    files = json.loads((root / "payloads.json").read_bytes())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "payloads.json"}
    assert actual == set(files)
    for name, expected in files.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    report = json.loads((root / "report.json").read_bytes())
    assert report["source_commit"] == validation["source_commit"]
    assert not report["goal_G05_complete"]
    for probe in report["clock_probes"]:
        path = root / f"{probe['runner']}-{round(1/probe['timestep_s'])}hz.npz"
        with np.load(path, allow_pickle=False) as arrays:
            measured = float(np.max(np.abs(arrays["physical_times_s"] - arrays["reported_times_s"])))
            assert measured == probe["max_clock_disagreement_s"]
            assert len(arrays["physical_times_s"]) - 1 == probe["physics_steps"]
            if probe["runner"] == "after":
                assert measured == 0
    successes = 0
    for body in report["bodies"]:
        assert body["fresh_exact_region_bakes"] == 2 and body["bake_failures"] == 0
        assert body["duration_scales"] == [1., 1.]
        assert body["region_substitutions"] == 0
        assert body["role_normalized_hash"] == "09191df11fd0af2d0172d92356e8beaabba12206e701ab1684c730ad7fccd1e6"
        if not body["accepted"]:
            assert body["robot_id"] == "zoo_dual_arm"
            assert body["failure"]["code"] == "joint_limit_violation"
            continue
        successes += 1
        assert body["repeats"] == 3 and len(set(body["replay_hashes"])) == 1 and body["violations"] == []
        with np.load(root / body["robot_id"] / "rollout.npz", allow_pickle=False) as arrays:
            assert np.all(np.diff(arrays["time_s"]) > 0)
            assert np.allclose(np.diff(arrays["time_s"]), body["native_timestep_s"], atol=1e-12, rtol=0)
            assert arrays["time_s"][-1] == body["actual_physics_duration_s"]
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
            payload = b"".join(np.ascontiguousarray(arrays[key], dtype=np.float64).tobytes()
                               for key in ("time_s", "qpos", "qvel", "ctrl"))
            assert hashlib.sha256(payload).hexdigest() == body["replay_hashes"][0]
    assert successes == 5 and len(report["bodies"]) == 6
    spec = importlib.util.spec_from_file_location("independent_quintic", ROOT / "g05-transition-diagnostic/polynomial_repro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extrema = module.reproduce()
    result = {"payloads_verified": len(files), "native_clock_probes": 3, "legacy_clock_probes": 3,
              "fresh_leaves_certified": 12, "canonical_compositions_certified": successes,
              "independent_polynomial_overshoots_reproduced": len(extrema), "G05_complete": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
