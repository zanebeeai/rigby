"""Register a protocol before runs and resolve deterministic public variants."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rigby_core.hashing import canonical_json_bytes, content_hash, hash_file

from .contracts import BenchmarkProtocolV1, BenchmarkWorldV1
from .world import BenchmarkRefusal


def split_hashes(protocol: BenchmarkProtocolV1) -> dict:
    return {split.split_id: content_hash(split) for split in protocol.splits}


def register_protocol(protocol: BenchmarkProtocolV1, destination: Path) -> dict:
    """Write a no-overwrite registration; manifest is published last.

The registration is self-verifying against its recorded hashes. Retain its
digest in a commit or external ledger before accepting scored runs.
"""
    protocol = BenchmarkProtocolV1.model_validate_json(protocol.model_dump_json())
    destination.mkdir(parents=True, exist_ok=False)
    payloads = {"benchmark-protocol.v1.json": canonical_json_bytes(protocol),
                "split-hashes.json": canonical_json_bytes(split_hashes(protocol))}
    for name, raw in payloads.items():
        (destination / name).write_bytes(raw)
    result = {"schema": "benchmark.registration.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": content_hash(protocol), "split_hashes": split_hashes(protocol),
              "files": {name: hash_file(destination / name) for name in payloads},
              "scored_runs_started": False, "existing_sealed_manifests_read": False,
              "scope": "Public engineering protocol; confirmatory sealed identities remain with their independent custodian."}
    (destination / "registration.json").write_bytes(canonical_json_bytes(result))
    return result


def load_registration(directory: Path, *, expected_registration_sha256: str) -> BenchmarkProtocolV1:
    if hash_file(directory / "registration.json") != expected_registration_sha256:
        raise BenchmarkRefusal("registration_changed", "Registration does not match its external digest")
    registration = json.loads((directory / "registration.json").read_bytes())
    if set(registration["files"]) != {"benchmark-protocol.v1.json", "split-hashes.json"}:
        raise BenchmarkRefusal("registration_inventory", "Registration file inventory is not recognized")
    for name, expected in registration["files"].items():
        if hash_file(directory / name) != expected:
            raise BenchmarkRefusal("protocol_changed", name)
    protocol = BenchmarkProtocolV1.model_validate_json((directory / "benchmark-protocol.v1.json").read_bytes())
    if content_hash(protocol) != registration["protocol_sha256"] or split_hashes(protocol) != registration["split_hashes"]:
        raise BenchmarkRefusal("protocol_changed", "Protocol or split meaning changed")
    return protocol


def public_variant(protocol: BenchmarkProtocolV1, *, split_id: str, seed: int) -> tuple[BenchmarkWorldV1, dict]:
    split = next((s for s in protocol.splits if s.split_id == split_id), None)
    if split is None or seed not in split.seeds:
        raise BenchmarkRefusal("unregistered_trial", "Split and seed must be registered before execution")
    rng = np.random.Generator(np.random.PCG64(seed))
    raw = protocol.world.model_dump(mode="json")
    policy = protocol.perturbations
    draws = []
    for item in sorted(raw["environment"]["objects"], key=lambda x: x["name"]):
        delta = rng.uniform(-np.asarray(policy.object_translation_half_width_m), policy.object_translation_half_width_m)
        mass = float(rng.uniform(*policy.mass_multiplier_range))
        friction = float(rng.uniform(*policy.friction_multiplier_range))
        item["position_m"] = (np.asarray(item["position_m"])+delta).tolist()
        item["mass_kg"] *= mass
        item["friction"] *= friction
        draws.append({"object": item["name"], "translation_m": delta.tolist(), "mass_multiplier": mass, "friction_multiplier": friction})
    world = BenchmarkWorldV1.model_validate(raw)
    return world, {"split_id": split_id, "split_sha256": content_hash(split), "seed": seed,
                   "generator": "numpy.PCG64", "numpy": np.__version__, "draws": draws,
                   "world_sha256": content_hash(world), "redraws": 0}
