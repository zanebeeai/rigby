"""Verify the G10 TransferObject evidence: the registration, the campaign rows, the bundles, the media, D10.

Recomputes every claim the report makes from the committed files: the
world, the library, the policy and the corpus against their registration;
the per-episode rows against the corpus (every registered episode run
once, in order); the nominal successes per body, the recoveries per class
per body, the false completions, the cap and the retry budgets re-derived
from the rows; every committed media bundle's digests and frame map, and
every local physical bundle's digests (and its physics replay with
--replay); the D10 index against its files.

    python docs/results/verify_g10.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g10-transfer-v1"
G06 = REPO / "any-robot/assets/general/research-protocols/g06-transfer-v1"
G09 = REPO / "any-robot/assets/general/research-protocols/g09-conditionals-v1"
CAMPAIGN = ROOT / "g10-campaign"
LOCAL = REPO / "any-robot/results/g10-campaign"
D10 = ROOT / "g10-d10"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bundle(physical: Path, expected: str, *, replay: bool) -> dict:
    manifest = json.loads((physical / "manifest.json").read_bytes())
    assert sha256(physical / "manifest.json") == expected, physical
    for name, info in manifest["files"].items():
        assert sha256(physical / name) == info["sha256"], (physical, name)
    if replay:
        import mujoco
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        model = mujoco.MjModel.from_binary_path(str(physical / "model.mjb"))
        record = PhysicsRecord.from_bytes((physical / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        assert replay_physics(model, record)["agrees"], physical
    return manifest


def check_media(media: Path, source: str | None = None) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    if source is not None:
        assert manifest["metadata"]["source_bundle_sha256"] == source
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"]
    if manifest["metadata"].get("refusal_slate"):
        assert manifest["metadata"]["playback_duration_s"] > 0
    else:
        assert abs(manifest["metadata"]["playback_duration_s"] - manifest["metadata"]["simulation_duration_s"]) <= 2.0 / manifest["metadata"]["fps"]
    for name in ("episode.mp4", "preview.gif", "frames.json"):
        assert (media / name).is_file(), (media, name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    validation = json.loads((ROOT / "g10-validation.json").read_bytes())

    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        path = PROTOCOL / name if not name.startswith(("g06/", "g09/")) else (G06 if name.startswith("g06/") else G09) / name.split("/", 1)[1]
        assert sha256(path) == digest, name
    listing = json.dumps(registration["files"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert hashlib.sha256(listing.encode("utf-8")).hexdigest() == registration["registration_sha256"] == validation["registration_sha256"]
    corpus = json.loads((PROTOCOL / "corpus.json").read_bytes())
    assert corpus["scored_runs_started"] is False and corpus["generation_calls"] == 0
    assert corpus["episode_cap_s"] == 120.0 and corpus["retry_budget"] == 3
    from rigby_core.skills.examples import transfer_object_library

    assert transfer_object_library().content_hash() == corpus["library_sha256"]
    library = json.loads((PROTOCOL / "library.json").read_bytes())
    assert library["library_id"] == "transfer_object_v1"
    assert all(s["loop"]["max_attempts"] <= 3 for s in library["skills"] if s["kind"] == "repeat_until")
    assert next(s for s in library["skills"] if s["skill_id"] == "transfer_object")["timeout_s"] == 120.0
    episodes = corpus["episodes"]
    assert len(episodes) == 3 * (100 + 3 * 20)

    summary = json.loads((CAMPAIGN / "summary.json").read_bytes())
    assert summary["registration_sha256"] == registration["registration_sha256"]
    assert summary["provenance"]["scored"] is True and summary["provenance"]["generation_calls"] == 0 and summary["provenance"]["commit"] == validation["campaign_commit"]
    rows = json.loads((CAMPAIGN / "trials.json").read_bytes())
    assert [r["episode_id"] for r in rows] == [e["episode_id"] for e in episodes]
    per = {}
    false_completions = 0
    for row, entry in zip(rows, episodes):
        assert row["zoo_id"] == entry["zoo_id"] and row["class"] == entry["class"] and row["seed"] == entry["seed"] and row["draw"] == entry["draw"]
        assert row["skill_success"] == (row["verdict"] == "success")
        assert row["false_completion"] == (row["skill_success"] and not row["oracle"]["placed"])
        assert row["within_cap"] == (row["physics_s"] <= corpus["episode_cap_s"] + 1e-9)
        assert (row["attempts"]["place_until_placed"] or 0) <= 3 and all(a <= 3 for a in row["attempts"]["acquire_until_held"])
        assert row["recovered"] == (row["class"] != "nominal" and row["skill_success"])
        if row["class"] != "nominal":
            # A disturbance fires at a leaf; an episode that failed before
            # reaching that leaf never received it and cannot have recovered.
            assert row["disturbance"] or not row["skill_success"], "a recovered episode received its disturbance"
        if not row["skill_success"]:
            assert "bundle" in row, "every failure is sealed and rendered"
        false_completions += row["false_completion"]
        cell = per.setdefault(row["zoo_id"], {}).setdefault(row["class"], {"episodes": 0, "successes": 0, "within_cap": 0})
        cell["episodes"] += 1
        cell["successes"] += row["skill_success"]
        cell["within_cap"] += row["within_cap"]
        if "bundle" in row:
            media = CAMPAIGN / row["zoo_id"] / row["class"] / f"seed-{row['seed']:03d}" / "media"
            manifest = check_media(media, row["bundle"]["sha256"])
            assert manifest["sha256"] == row["media_sha256"] if "sha256" in manifest else True
            physical = LOCAL / row["zoo_id"] / row["class"] / f"seed-{row['seed']:03d}" / "physical"
            if (physical / "manifest.json").is_file():
                bundle = check_bundle(physical, row["bundle"]["sha256"], replay=args.replay)
                assert bundle["metadata"]["trace_sha256"] == row["bundle"]["trace_sha256"]
    for body, classes in per.items():
        for kind, cell in classes.items():
            claimed = summary["per_body"][body][kind]
            assert claimed["episodes"] == cell["episodes"] and claimed["successes"] == cell["successes"] and claimed["within_cap"] == cell["within_cap"], (body, kind)
            assert validation["per_body"][body][kind]["successes"] == cell["successes"]
    assert summary["false_completions"] == false_completions == validation["false_completions"]

    index = json.loads((D10 / "index.json").read_bytes())
    assert index["generation_calls"] == 0
    assert (D10 / index["three_body"]["video"]).is_file() and (D10 / index["three_body"]["preview"]["gif"]).is_file()
    assert len(index["three_body"]["episodes"]) == 3
    for case in index["cases"]:
        manifest = check_media(D10 / case["name"] / "overlay")
        assert manifest["metadata"]["source_bundle_sha256"] == case["physical_sha256"]
        assert sha256(D10 / case["video"]) == manifest["files"]["episode.mp4"]["sha256"]
        assert sha256(D10 / case["preview"]) == manifest["files"]["preview.gif"]["sha256"]
        assert sha256(D10 / case["frames"]) == manifest["files"]["frames.json"]["sha256"]
        trail = json.loads((D10 / case["name"] / "overlay" / "trail.json").read_bytes())
        assert len(trail) == manifest["metadata"]["frame_count"]
        row = next(r for r in rows if r["episode_id"] == case["episode_id"])
        assert case["verdict"] == row["verdict"] and case["recovered"] == row["recovered"]
    assert {c["class"] for c in index["cases"]} == {"slip", "occlusion"}
    print(json.dumps({"verified": True, "episodes": len(rows), "false_completions": false_completions,
                      "per_body": {b: {k: f"{v['successes']}/{v['episodes']}" for k, v in c.items()} for b, c in per.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
