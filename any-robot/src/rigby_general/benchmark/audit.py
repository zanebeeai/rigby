"""Reproduce and verify the public G02 contract audit, without scored trials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess

import mujoco
import numpy as np
import pydantic
from rigby_core.hashing import canonical_json_bytes, content_hash, hash_file

from ..pipeline import ingest_robot
from .protocol import load_registration, public_variant
from .runner import run_trial
from .world import BenchmarkBodyManifestV1, compile_world

ROOT = Path(__file__).resolve().parents[4]
REGISTRATION = ROOT / "any-robot/assets/general/research-protocols/transfer-bench-v1-r1"
REGISTRATION_SHA256 = "1dab70141652d41aae1e511937696f7c9bbf9cb891aecd19ed7565734f32fc49"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def verify_release(directory: Path, *, expected_index_sha256: str) -> dict:
    """Verify a committed index and every payload before reading its claims."""
    directory = directory.resolve()
    if hash_file(directory / "index.json") != expected_index_sha256:
        raise ValueError("Evidence index differs from the external digest")
    index = json.loads((directory / "index.json").read_bytes())
    if index.get("schema") != "benchmark.g02-evidence.v1":
        raise ValueError("Unknown evidence schema")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} - {"index.json"}
    if actual != set(index["files"]):
        raise ValueError("Evidence inventory changed")
    for name, digest in index["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Invalid evidence path")
        path = directory.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(directory) or any(p.is_symlink() for p in (path, *path.parents) if p != directory.parent):
            raise ValueError("Evidence links are not admitted")
        if hash_file(path) != digest:
            raise ValueError(f"Evidence payload changed: {name}")
    conformance = json.loads((directory / "world-conformance.json").read_bytes())
    if len(conformance["variants"]) != 140 or conformance["body_count"] != 6:
        raise ValueError("Audit coverage differs from the registered engineering roster")
    comparisons = 0
    for row in conformance["variants"]:
        if len(row["bodies"]) != 6 or any(b["world_sha256"] != row["world_sha256"] for b in row["bodies"]):
            raise ValueError("A body changed the fixed world")
        if len({b["compiled_model_sha256"] for b in row["bodies"]}) != 6:
            raise ValueError("Embodiment models were not distinct")
        comparisons += len(row["bodies"])
    results = [json.loads(p.read_bytes()) for p in sorted((directory / "probes").glob("*/*/result.json"))]
    if len(results) != 12:
        raise ValueError("Expected six bodies in both disclosed modes")
    for row in results:
        if row["scored"] or row["root_success"] or row["status"] != "protocol_probe_complete":
            raise ValueError("An infrastructure probe was relabeled or did not finish")
        if row["physical_steps"] != 100 or abs(row["elapsed_simulation_s"] - 0.2) > 1e-12:
            raise ValueError("Probe physics duration differs")
        if not row["worker_closed_before_privileged_output"]:
            raise ValueError("Observation boundary was not closed")
    return {"verified": True, "files": len(index["files"]), "body_world_comparisons": comparisons,
            "probes": len(results), "scored_runs": 0, "root_successes": 0,
            "index_sha256": expected_index_sha256}


def create_release(directory: Path) -> dict:
    """Compile all public seed/body pairs, then run twelve short sensor probes."""
    if directory.exists():
        raise FileExistsError(directory)
    protocol = load_registration(REGISTRATION, expected_registration_sha256=REGISTRATION_SHA256)
    # Require the implementation and protocol to be committed before publishing
    # evidence. Unrelated output files may remain untracked while we generate.
    code_roots = ["core/src", "any-robot/src", "any-robot/assets/general/research-protocols"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *code_roots], cwd=ROOT, check=True, capture_output=True)
    if subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *code_roots], cwd=ROOT, text=True).strip():
        raise ValueError("Commit source and protocol before producing the release")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = sorted((ROOT / "any-robot/assets/general/zoo").glob("*/robot.urdf"))
    if len(paths) != 6:
        raise ValueError("The public zoo roster must contain six source URDFs")
    robots = [(p, ingest_robot(p)) for p in paths]
    directory.mkdir(parents=True)
    for path, robot in robots:
        write_json(directory / "bodies" / (path.parent.name + ".json"), BenchmarkBodyManifestV1(robot=robot.manifest))
    variants = []
    for split in protocol.splits:
        for seed in split.seeds:
            world, draw = public_variant(protocol, split_id=split.split_id, seed=seed)
            reference = compile_world(world)
            rows = []
            for path, robot in robots:
                compiled = compile_world(world, body=BenchmarkBodyManifestV1(robot=robot.manifest),
                                         robot_xml=robot.finalized.mjcf_xml, asset_root=path.parent)
                if compiled.world_sha256 != reference.world_sha256:
                    raise ValueError("Body swap altered the registered world")
                rows.append({"body": path.parent.name, "world_sha256": compiled.world_sha256,
                             "body_manifest_sha256": content_hash(compiled.body_manifest),
                             "compiled_model_sha256": compiled.compiled_model_sha256})
            write_json(directory / "worlds" / f"{split.split_id}-{seed}.json", reference.manifest)
            variants.append({"split": split.split_id, "seed": seed, "draw": draw,
                             "world_sha256": reference.world_sha256, "bodies": rows})
        print(json.dumps({"phase": "fixed_world_conformance", "split": split.split_id,
                          "variants_completed": len(variants), "body_count": len(robots)}), flush=True)
    write_json(directory / "world-conformance.json", {"body_count": 6, "variants": variants,
               "scope": "Resolved-model invariance; these compilations do not establish task capability."})
    for path, _ in robots:
        for mode in protocol.modes:
            result = run_trial(registration=REGISTRATION, expected_registration_sha256=REGISTRATION_SHA256,
                robot_source=path, split_id="development", seed=0, mode=mode,
                policy="rigby_general.benchmark.policies:zero_command_probe",
                destination=directory / "probes" / path.parent.name / mode, probe_steps=100)
            print(json.dumps({"phase": "sensor_policy_probe", "body": path.parent.name,
                              "mode": mode, "status": result["status"]}), flush=True)
    for path in sorted(REGISTRATION.iterdir()):
        if path.is_file():
            target = directory / "registration" / path.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(path.read_bytes())
    write_json(directory / "provenance.json", {"source_commit": source_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "mujoco": mujoco.__version__, "numpy": np.__version__, "pydantic": pydantic.__version__,
        "registration_sha256": REGISTRATION_SHA256, "protocol_sha256": content_hash(protocol),
        "dependency_lock_sha256": hash_file(ROOT / "uv.lock"), "public_zoo_only": True,
        "scored_runs": 0, "existing_sealed_manifests_read": False,
        "observation_threat_model": "Trusted policy API and clean Python state; not a hostile-code OS sandbox",
        "physical_replay_available": False,
        "scope": "G02 contract infrastructure only; G01 integration required before research scoring."})
    index = {"schema": "benchmark.g02-evidence.v1", "source_commit": source_commit,
             "files": {p.relative_to(directory).as_posix(): hash_file(p)
                       for p in sorted(directory.rglob("*")) if p.is_file()}}
    write_json(directory / "index.json", index)
    return verify_release(directory, expected_index_sha256=hash_file(directory / "index.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-index-sha256")
    args = parser.parse_args()
    if args.command == "create":
        result = create_release(args.directory)
    else:
        if not args.expected_index_sha256:
            parser.error("Verification requires an externally retained index digest")
        result = verify_release(args.directory, expected_index_sha256=args.expected_index_sha256)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
