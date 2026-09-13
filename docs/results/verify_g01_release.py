"""Offline acceptance audit of the committed G01 release (run from repo root).

PYTHONPATH must include core/src and any-robot/src. Requires ffprobe on PATH.
The optional --rerendered path compares an independently rendered compact
episode with the release. Fresh results go to --out; the release is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from rigby_core.evidence import EvidenceIntegrityError, verify_bundle
from rigby_core.evidence_archive import unpack_bundle
from rigby_core.simulation.recording import PhysicsRecord
from rigby_general.evidence.capture import replay_bundle


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit_union(paths: list[Path]) -> dict:
    states: dict[str, set[str]] = {}
    for path in paths:
        for case in ET.parse(path).iter("testcase"):
            name = case.get("classname", "") + "::" + case.get("name", "")
            state = next((s for s in ("failure", "error", "skipped") if case.find(s) is not None), "passed")
            states.setdefault(name, set()).add(state)
    assert not any(value & {"failure", "error"} for value in states.values())
    return {
        "files": [p.as_posix() for p in paths],
        "unique_passed": sum("passed" in value for value in states.values()),
        "unresolved_skips": sorted(k for k, value in states.items() if value == {"skipped"}),
        "skips_resolved_by_followup": sum(value == {"passed", "skipped"} for value in states.values()),
        "deselected_tests_not_in_junit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("docs/results/g01-release"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rerendered", type=Path)
    args = parser.parse_args()
    index = json.loads((args.release / "index.json").read_bytes())
    result = {"schema": "rigby.g01-validation/1", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "release_index_sha256": sha(args.release / "index.json"), "cases": [], "tampering": []}
    results = Path("any-robot/results")
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g01-audit-", dir=results) as temporary:
        for row in index["cases"]:
            root = Path(temporary) / row["case"]
            digest = unpack_bundle(args.release / row["archive"], root,
                                   archive_sha256=row["archive_sha256"], expected_digest=row["physical_manifest_sha256"])
            manifest = verify_bundle(root, digest)
            record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
            repeats = json.loads((root / "repeats.json").read_bytes())
            execution = json.loads((root / "execution.json").read_bytes())
            task = json.loads((root / "task.json").read_bytes())
            source = json.loads((root / "source.json").read_bytes())
            assert repeats["count"] == 3 and repeats["agree"]
            assert repeats["independent_prompt_runs"] and repeats["one_recording_per_prompt_run"]
            assert len(repeats["reference_data_hashes"]) == 3 and len(set(repeats["reference_data_hashes"])) == 1
            assert len(repeats["pipeline_trace_hashes"]) == 3 and len(set(repeats["pipeline_trace_hashes"])) == 1
            assert repeats["physical_hashes"] == [record.content_hash()] * 3
            assert repeats["tolerance"] == {"absolute": 0.0, "relative": 0.0}
            assert all(x == execution["nominal_repeat_outcomes"][0] for x in execution["nominal_repeat_outcomes"])
            assert task["requested_semantics"] == task["bound_semantics"]
            assert all(r["agrees"] for r in repeats["recorded_control_replays"])
            assert sha(root / "world.json") == row["world_sha256"]
            media = args.release / row["media"]
            verify_bundle(media, row["media_manifest_sha256"])
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames,duration", "-of", "json", str(media / "episode.mp4"),
            ]))["streams"][0]
            assert int(probe["nb_read_frames"]) == row["video_frames"]
            assert abs(float(probe["duration"]) - row["video_duration_s"]) < 1e-5
            result["cases"].append({"case": row["case"], "archive_restored": True, "manifest_sha256": digest,
                                    "media_verified": True, "decoded_video_frames": int(probe["nb_read_frames"]),
                                    "repeat_count": 3, "repeat_state_action_hashes_equal": True,
                                    "independent_prompt_references_and_pipeline_traces_equal": True,
                                    "repeat_nominal_outcomes_equal": True, "requested_semantics_preserved": True,
                                    "baked_leaf_count": len(execution["baked_primitives"]),
                                    "dependencies": source["dependencies"], "source_commit": source["commit"]})
            if row["case"] == "zoo_compact_arm":
                result["restored_compact_replay"] = replay_bundle(root, digest)
                assert result["restored_compact_replay"]["agrees"]
                for name in ("trace.npz", "model.mjb", "world.json", "outcome.json"):
                    path = root / name
                    original = path.read_bytes()
                    altered = bytearray(original)
                    altered[len(altered) // 2] ^= 1
                    path.write_bytes(altered)
                    try:
                        verify_bundle(root, digest)
                    except EvidenceIntegrityError as exc:
                        result["tampering"].append({"payload": name, "same_size_bit_flip_detected": True, "error": str(exc)})
                    else:
                        raise AssertionError(f"Undetected tampering: {name}")
                    finally:
                        path.write_bytes(original)
                    verify_bundle(root, digest)
                if args.rerendered:
                    result["offline_rerender"] = {"path": args.rerendered.as_posix(), "files": {}}
                    for name in ("initial.png", "final.png", "episode.mp4"):
                        expected, actual = sha(media / name), sha(args.rerendered / name)
                        assert expected == actual, f"Rerender differs: {name}"
                        result["offline_rerender"]["files"][name] = {"byte_identical": True, "sha256": actual}
    preview = json.loads((args.release / "preview-manifest.json").read_bytes())
    for name, digest in preview["files"].items():
        assert sha(args.release / name) == digest
    result["preview_integrity_valid"] = True
    result["anyrobot_tests"] = junit_union([Path("docs/results") / name for name in
                                           ("g01-anyrobot-tests.xml", "g01-asset-followup-tests.xml", "g01-repeat-regression-tests.xml")])
    result["core_tests"] = junit_union([Path("docs/results") / name for name in
                                       ("g01-core-tests.xml", "g01-archive-and-neutrality-tests.xml")])
    result["known_core_exclusion"] = "test_the_digest_is_stable_across_processes: Windows subprocess environment fix lives in independent G00 PR 27"
    result["all_checks_passed"] = True
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"checked_cases": len(result["cases"]), "tamper_checks": len(result["tampering"]),
                      "anyrobot": result["anyrobot_tests"], "core": result["core_tests"], "all_checks_passed": True}))


if __name__ == "__main__":
    main()
