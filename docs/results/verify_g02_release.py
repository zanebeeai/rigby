"""Verify the committed G02 release and detect real payload mutations on copies."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from xml.etree import ElementTree as ET

import numpy as np
from rigby_core.hashing import canonical_json_bytes, hash_file
from rigby_general.benchmark.audit import verify_release

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "docs/results/g02-release"
INDEX_SHA256 = "4f0088702fbc4c845838f62fafff62c4ac8a1de53ce380d2f728bba567752303"


def main():
    verified = verify_release(RELEASE, expected_index_sha256=INDEX_SHA256)
    conformance = json.loads((RELEASE / "world-conformance.json").read_bytes())
    crosswalk = json.loads((ROOT / "docs/results/g02-acceptance-crosswalk.json").read_bytes())
    assert hash_file(ROOT / crosswalk["source"]) == crosswalk["source_sha256"]
    assert len(crosswalk["requirements"]) == 16 and not crosswalk["thresholds_modified"]
    rows = []
    for path in sorted((RELEASE / "probes").glob("*/*/result.json")):
        result = json.loads(path.read_bytes())
        packet = json.loads((path.parent / "initial-policy-observation.json").read_bytes())
        inputs = json.loads((path.parent / "trial-input.json").read_bytes())
        truth = json.loads((path.parent / "final-evaluator-truth.json").read_bytes())
        rgb = np.asarray(next(r["values"] for r in packet["readings"] if r["channel"] == "rgb"))
        assert rgb.size == 96*96*3 and 0 <= rgb.min() < rgb.max() <= 255
        assert packet["access"] == "declared_sensors" and truth["access"] == "evaluator_only"
        assert not {"object_poses", "success_labels", "simulator_state"} & packet.keys()
        assert inputs["source_commit"] == json.loads((RELEASE / "provenance.json").read_bytes())["source_commit"]
        rows.append({"body": path.parent.parent.name, "mode": result["mode"], "world_sha256": result["world_sha256"],
                     "normalization": inputs["normalization"], "steps": result["physical_steps"],
                     "simulation_s": result["elapsed_simulation_s"], "status": result["status"],
                     "scored": result["scored"], "root_success": result["root_success"],
                     "rgb_values": int(rgb.size), "rgb_min": int(rgb.min()), "rgb_max": int(rgb.max()),
                     "worker_closed_before_privileged_output": result["worker_closed_before_privileged_output"]})
    strict = {r["world_sha256"] for r in rows if r["mode"] == "strict_fixed_world"}
    normalized = {r["world_sha256"] for r in rows if r["mode"] == "capability_normalized"}
    assert len(strict) == 1 and len(normalized) == 6 and not strict & normalized
    tampering = []
    targets = ["worlds/development-0.json", "probes/zoo_compact_arm/strict_fixed_world/result.json",
               "probes/zoo_compact_arm/strict_fixed_world/initial-policy-observation.json",
               "registration/feasibility-rules.v1.json"]
    with tempfile.TemporaryDirectory(prefix="rigby-g02-integrity-") as temporary:
        copy = Path(temporary) / "release"
        shutil.copytree(RELEASE, copy)
        for name in targets:
            path = copy / name
            original = path.read_bytes()
            mutated = bytearray(original)
            mutated[len(mutated)//2] ^= 1
            try:
                path.write_bytes(mutated)
                try:
                    verify_release(copy, expected_index_sha256=INDEX_SHA256)
                except ValueError as error:
                    assert "payload changed" in str(error)
                    tampering.append({"payload": name, "same_size_bit_flip_detected": True})
                else:
                    raise AssertionError("Mutation went undetected")
            finally:
                path.write_bytes(original)
            verify_release(copy, expected_index_sha256=INDEX_SHA256)
    test_reports = []
    for name in ("g02-core-tests.xml", "g02-anyrobot-tests.xml", "g02-evaluation-regressions.xml"):
        path = ROOT / "docs/results" / name
        suite = ET.parse(path).getroot().find("testsuite")
        attributes = dict(suite.attrib)
        assert int(attributes["failures"]) == int(attributes["errors"]) == int(attributes["skipped"]) == 0
        test_reports.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hash_file(path), **attributes})
    evidence = {"schema": "benchmark.g02-validation.v1", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Benchmark contracts and observation access only; no manipulation/locomotion/VLM/robustness claim",
        "source_commit": json.loads((RELEASE / "provenance.json").read_bytes())["source_commit"],
        "release": verified, "registered_world_variants": len(conformance["variants"]),
        "strict_probe_world_sha256": next(iter(strict)), "distinct_normalized_worlds": len(normalized),
        "probes": rows, "tamper_checks": tampering, "test_reports": test_reports,
        "known_test_exclusion": {"path": "core/tests/test_core_hashing.py", "test": "test_the_digest_is_stable_across_processes",
            "reason": "Existing Windows subprocess environment issue; fix is separately validated in G00 PR 27. No assertion changed or new skip introduced here."},
        "crosswalk_sha256": hash_file(ROOT / "docs/results/g02-acceptance-crosswalk.json"),
        "existing_acceptance_requirements_preserved": 16,
        "video_required_by_G02": False, "linux_ci_validated": False,
        "observation_boundary_limit": "Trusted API and clean interpreter state, not protection against hostile same-user OS access",
        "research_scoring_enabled": False}
    (ROOT / "docs/results/g02-validation.json").write_bytes(canonical_json_bytes(evidence))
    print(json.dumps({"release": verified, "tamper_checks_detected": len(tampering),
                      "strict_probe_world_sha256": next(iter(strict)), "distinct_normalized_worlds": len(normalized),
                      "test_counts": [int(r["tests"]) for r in test_reports]}, indent=2))


if __name__ == "__main__":
    main()
