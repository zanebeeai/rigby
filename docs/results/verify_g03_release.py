"""Verify the complete G03 release and replay its saved kinematic inspections."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from rigby_general.capabilities.inspection import render_inspection, verify_inspection


REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent / "g03-release"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT, *, rerender=False):
    validation = json.loads((Path(__file__).parent / "g03-validation.json").read_bytes())
    assert digest(root / "release-index.json") == validation["release_index_sha256"]
    index = json.loads((root / "release-index.json").read_bytes())
    actual_paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "release-index.json"}
    assert actual_paths == set(index["files"])
    for name, expected in index["files"].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()) and digest(path) == expected, name
    audit = json.loads((root / "audit.json").read_bytes())
    assert audit["source_commit"] == validation["source_commit"]
    assert not audit["tracked_tree_dirty"]
    assert audit["physics_task_trials"] == 0 and audit["api_calls"] == 0
    inspections, frames, sites = 0, 0, 0
    for entry in audit["admissions"]:
        name = entry["body"]
        if entry["outcome"] != "admitted":
            assert name == "panda" and entry["code"] == "unsupported_coupling"
            assert entry["details"]["model_invalid"] is False
            continue
        replay = verify_inspection(root / "bodies" / name)
        assert replay["site_replay_max_error_m"] == 0
        inspections += 1
        sites += replay["kinematic_samples_replayed"]
        with Image.open(root / "media" / name / "inspection.gif") as gif:
            assert gif.n_frames == entry["frame_count"] == 60
            durations = []
            for number in range(gif.n_frames):
                gif.seek(number)
                durations.append(gif.info["duration"])
            assert durations == [100] * 60
            frames += gif.n_frames
    assert inspections == 12 and frames == 720 and sites == 1536
    spec = importlib.util.spec_from_file_location("independent_g03_labels", REPO / "any-robot/tests/fixtures/g03_geometry/verify_geometry.py")
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    assert independent.verify() == json.loads((root / "independent-geometry.json").read_bytes())
    geometry = json.loads((root / "geometry.json").read_bytes())
    assert len(geometry) == 3 and all(g["max_site_error_m"] <= .001 and g["all_required_label_semantics_present"] for g in geometry)
    cases = json.loads((root / "invariance.json").read_bytes())
    assert len(cases) == 120
    bodies = {c["body"] for c in cases}
    assert len(bodies) == 6
    for body in bodies:
        rows = [c for c in cases if c["body"] == body]
        assert {c["variant"] for c in rows} == set(range(20))
        assert len({c["source_sha256"] for c in rows}) == 20
        with np.load(root / "invariance" / body / "baseline.npz", allow_pickle=False) as baseline:
            for case in rows:
                path = root / "invariance" / body / f"variant-{case['variant']:02d}"
                assert digest(path / "robot.urdf") == case["source_sha256"]
                assert case["capabilities_exact"] and case["runtime_mjcf_exact"] and case["figure_sites_exact"]
                assert case["inverse_renamed_source_mechanics_exact"] and case["reference_clock_exact"]
                with np.load(path / "reference.npz", allow_pickle=False) as actual:
                    assert set(actual.files) == set(baseline.files)
                    assert all(np.array_equal(actual[key], baseline[key]) for key in actual.files)
    rerendered = False
    if rerender:
        with tempfile.TemporaryDirectory(prefix="g03-independent-rerender-") as temporary:
            # Repeat a full archived inspection with no source URDF/mesh access.
            render_inspection(root / "bodies/three_digits", Path(temporary) / "media")
            assert digest(Path(temporary) / "media/inspection.gif") == digest(root / "media/three_digits/inspection.gif")
            rerendered = True
    return {"payloads_verified": len(index["files"]), "bodies_replayed": inspections,
            "kinematic_samples_replayed": sites, "inspection_frames_verified": frames,
            "invariance_cases_verified": len(cases), "full_gif_rerender_byte_identical": rerendered,
            "physical_task_success_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rerender", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, rerender=args.rerender), indent=2, sort_keys=True))
