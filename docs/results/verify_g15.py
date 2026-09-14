"""Verify the G15 clearance evidence: the registration, every episode row, the chain-length curves, the paired disturbed comparison, D15.

Recomputes every claim the report makes from the committed files: the
corpus, policy, world and generated libraries against the registration;
per body and chain length the fifty registered seeds run once each in
order, root success recounted from the rows with the oracle's judgement
and false completions beside it, the Wilson interval re-derived; per body
the thirty disturbed seeds run once through the tree and once through the
flat twin, paired by seed with the same disturbance, the gain in
percentage points re-derived; every episode's per-object outcomes
consistent with its verdict (no averaging of children over an unfinished
root); every episode within its cap and every tree at least four deep with
every transfer bound to a distinct object; the D15 index against its media
bundles and frame maps; the baseline (protocol v1) rows against their own
registration, and the before/after pairs against their media. With
--replay, every local segment bundle of the D15 and before/after episodes
is replayed on native physics and must agree exactly.

    python docs/results/verify_g15.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g15-clearance-v3"
PROTOCOL_V2 = REPO / "any-robot/assets/general/research-protocols/g15-clearance-v2"
PROTOCOL_V1 = REPO / "any-robot/assets/general/research-protocols/g15-clearance-v1"
CAMPAIGN = ROOT / "g15-clearance"
BASELINE = ROOT / "g15-clearance-v1"
D15 = ROOT / "g15-d15"
BEFORE_AFTER = ROOT / "g15-before-after"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def check_media(media: Path) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"], media
    return manifest


def check_segment(bundle: Path, expected: str, *, replay: bool) -> None:
    assert sha256(bundle / "manifest.json") == expected, bundle
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(bundle / name) == info["sha256"], (bundle, name)
    if replay:
        import mujoco
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        record = PhysicsRecord.from_bytes((bundle / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        if len(record.arrays["time_s"]) > 1:
            model = mujoco.MjModel.from_binary_path(str(bundle / "model.mjb"))
            assert replay_physics(model, record)["agrees"], bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "core/src"))
    sys.path.insert(0, str(REPO / "any-robot/src"))
    sys.path.insert(0, str(REPO / "any-robot/scripts"))
    from rigby_core.skills.clearance import clear_work_area_library
    from rigby_general.evidence.capture import json_bytes
    from rigby_general.skills.clear_work_area import LAYOUT, LAYOUT_V1, build_clearance_world
    import g10_corpus as g10

    validation = json.loads((ROOT / "g15-validation.json").read_bytes())
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    assert sha256(PROTOCOL / "corpus.json") == registration["files"]["corpus.json"]
    assert sha256(g10.G09 / "policy.json") == registration["files"]["g09/policy.json"]
    assert registration["registration_sha256"] == validation["registration_sha256"]
    corpus = json.loads((PROTOCOL / "corpus.json").read_bytes())
    base = g10.environment()
    assert registration["layout"] == LAYOUT.name == "v3" and corpus["layout"] == LAYOUT.as_json()
    assert sha256(PROTOCOL / "qualification.json") == registration["files"]["qualification.json"] == corpus["qualification_sha256"]
    qualification = json.loads((PROTOCOL / "qualification.json").read_bytes())["positions"]
    for x, y in LAYOUT.slots_m:
        assert all(qualification[f"slot:{x:.2f},{y:.2f}"][b]["verdict"] == "success" for b in corpus["bodies"]), (x, y)
    for dx, dy in LAYOUT.cell_offsets_m:
        x, y = LAYOUT.platform_centre_m[0] + dx, LAYOUT.platform_centre_m[1] + dy
        assert all(qualification[f"cell:{x:.2f},{y:.2f}"][b]["verdict"] == "success" for b in corpus["bodies"]), (x, y)
    registration_v2 = json.loads((PROTOCOL_V2 / "registration.json").read_bytes())
    assert sha256(PROTOCOL_V2 / "corpus.json") == registration_v2["files"]["corpus.json"], "the superseded v2 registration is kept as registered"
    assert LAYOUT.pitch_m >= 0.10, "ten centimetres between neighbours with the jaw opened to the object's width"
    for count in corpus["chain_lengths"]:
        world = build_clearance_world(base, count)
        assert hashlib.sha256(json_bytes(world.environment.model_dump(mode="json"))).hexdigest() == registration["worlds"][str(count)] == corpus["worlds"][str(count)]["environment_sha256"]
        for flat in (False, True):
            library = clear_work_area_library(world.objects, tuple(world.cells[o] for o in world.objects), flat=flat)
            assert library.content_hash() == registration["libraries"][f"{count}-{'flat' if flat else 'tree'}"]
            assert library.library_id.startswith("clear_work_area_v3_")
            tree = library.expand("clear_work_area", {"effector": "x"})
            assert tree.max_depth >= 4
            bound = [n.arguments["object"] for n in tree.root.walk() if n.skill_id == "transfer_object"]
            assert bound == list(world.objects), "one transfer per object, generated, not scripted"
            assert [c.skill for c in library.skill("clear_pass_and_check").children] == ["clear_pass", "stand_clear", "observe_work_area"], "every pass ends with the arm standing clear and a look"
    targets = {int(k): v for k, v in corpus["targets"]["nominal"].items()}
    # the baseline: protocol v1 rows against their own registration, complete or stopped where the revision found them
    registration_v1 = json.loads((PROTOCOL_V1 / "registration.json").read_bytes())
    assert sha256(PROTOCOL_V1 / "corpus.json") == registration_v1["files"]["corpus.json"]
    corpus_v1 = json.loads((PROTOCOL_V1 / "corpus.json").read_bytes())
    assert corpus_v1["nominal"] == corpus["nominal"] and corpus_v1["disturbed"] == corpus["disturbed"], "the draws are unchanged between the protocols"
    for count in corpus_v1["chain_lengths"]:
        world = build_clearance_world(base, count, layout=LAYOUT_V1)
        for flat in (False, True):
            library = clear_work_area_library(world.objects, tuple(world.cells[o] for o in world.objects), flat=flat, observe_each_pass=False)
            assert library.content_hash() == registration_v1["libraries"][f"{count}-{'flat' if flat else 'tree'}"], "the v1 structure is reproduced exactly by the legacy flag"
    for cell in validation["baseline"]["cells"]:
        rows = json.loads((BASELINE / cell["name"] / "trials.json").read_bytes())
        provenance = json.loads((BASELINE / cell["name"] / "provenance.json").read_bytes())
        assert provenance["registration_sha256"] == registration_v1["registration_sha256"] and provenance["generation_calls"] == 0
        assert cell["episodes"] == len(rows) and cell["successes"] == sum(1 for r in rows if r["skill_success"]) and cell["false_completions"] == sum(1 for r in rows if r["false_completion"])
        assert [r["seed"] for r in rows] == [d["seed"] for d in corpus_v1["nominal"]][: len(rows)]
        assert cell["complete"] == (len(rows) == len(corpus_v1["nominal"]))

    # -- nominal chain-length curves -------------------------------------------------------------
    for body in corpus["bodies"]:
        for count in corpus["chain_lengths"]:
            rows = json.loads((CAMPAIGN / f"nominal-{body}-{count}" / "trials.json").read_bytes())
            summary = json.loads((CAMPAIGN / f"nominal-{body}-{count}" / "summary.json").read_bytes())
            assert summary["provenance"]["scored"] is True and summary["provenance"]["registration_sha256"] == registration["registration_sha256"] and summary["provenance"]["generation_calls"] == 0
            assert all(r["layout"] == "v3" and r["library_id"] == f"clear_work_area_v3_tree_{count}" for r in rows)
            assert [r["seed"] for r in rows] == [d["seed"] for d in corpus["nominal"]], (body, count)
            assert all(r["executor"] == "tree" and r["objects"] == count for r in rows)
            successes = sum(1 for r in rows if r["skill_success"])
            false_completions = sum(1 for r in rows if r["false_completion"])
            oracle_root = sum(1 for r in rows if r["oracle"]["root_success"])
            for r in rows:
                assert r["within_cap"], (body, count, r["episode_id"])
                assert r["tree"]["max_depth"] >= 4 and r["tree"]["transfer_instances"] == count
                if r["skill_success"]:
                    assert len(r["placed_by_belief"]) == count and r["work_area_clear_observed"] is True, "root success needs every object placed and the area observed clear"
                assert r["placed_by_oracle"] <= count
            cell = validation["nominal"][body][str(count)]
            assert cell["successes"] == successes and cell["episodes"] == len(rows) and cell["false_completions"] == false_completions and cell["oracle_root"] == oracle_root
            low, high = wilson(successes, len(rows))
            assert abs(cell["wilson95"][0] - low) < 1e-9 and abs(cell["wilson95"][1] - high) < 1e-9
            assert cell["target"] == targets[count] and cell["met"] == (successes / len(rows) >= targets[count])
            for r in rows:
                if "sealed" in r and args.replay and r["episode_id"] in validation.get("d15_episodes", []):
                    for seg in r["sealed"]["segments"]:
                        if Path(seg["bundle"]).exists():
                            check_segment(Path(seg["bundle"]), seg["sha256"], replay=True)

    # -- paired disturbed comparison ---------------------------------------------------------------
    for body in corpus["bodies"]:
        rows = json.loads((CAMPAIGN / f"disturbed-{body}" / "trials.json").read_bytes())
        by_seed: dict[int, dict[str, dict]] = {}
        for r in rows:
            by_seed.setdefault(r["seed"], {})[r["executor"]] = r
        assert sorted(by_seed) == [d["seed"] for d in corpus["disturbed"]]
        for draw in corpus["disturbed"]:
            pair = by_seed[draw["seed"]]
            assert set(pair) == {"tree", "flat"}
            for executor, r in pair.items():
                assert r["disturbance"] == {"object": f"cube_{draw['object_index'] + 1:02d}", "kind": draw["kind"]} and r["objects"] == corpus["disturbed_count"]
                assert r["within_cap"] and r["cap_s"] == corpus["budgets"]["cap_s"][str(corpus["disturbed_count"])]
        tree_successes = sum(1 for p in by_seed.values() if p["tree"]["skill_success"])
        flat_successes = sum(1 for p in by_seed.values() if p["flat"]["skill_success"])
        gain = 100.0 * (tree_successes - flat_successes) / len(by_seed)
        cell = validation["disturbed"][body]
        assert cell["tree"] == tree_successes and cell["flat"] == flat_successes and abs(cell["gain_pp"] - gain) < 1e-9 and cell["pairs"] == len(by_seed)
        assert cell["met"] == (gain >= corpus["targets"]["recovery_gain_pp"])
        assert cell["false_completions"] == sum(1 for p in by_seed.values() for r in p.values() if r["false_completion"])
        for pair in by_seed.values():
            for r in pair.values():
                if "sealed" in r and args.replay and r["episode_id"] in validation.get("d15_episodes", []):
                    for seg in r["sealed"]["segments"]:
                        if Path(seg["bundle"]).exists():
                            check_segment(Path(seg["bundle"]), seg["sha256"], replay=True)

    # -- D15 -------------------------------------------------------------------------------------------------
    index = json.loads((D15 / "index.json").read_bytes())
    assert index["generation_calls"] == 0
    for clearance in index["clearances"]:
        if clearance.get("episode_id") is None:
            continue
        manifest = check_media(D15 / Path(clearance["video"]).parent)
        assert sha256(D15 / Path(clearance["video"]).parent / "manifest.json") == clearance["media_sha256"]
        assert manifest["metadata"]["frame_count"] == clearance["frame_count"] and len(manifest["metadata"]["source_segments"]) == clearance["segments"]
        assert clearance["objects"] == clearance["segments"] or clearance["segments"] >= clearance["objects"]
    assert [c["objects"] for c in index["clearances"]] == corpus["chain_lengths"]
    # -- before/after ------------------------------------------------------------------------------------
    pairs = json.loads((BEFORE_AFTER / "index.json").read_bytes())["pairs"]
    assert set(pairs) == {"pitch", "look", "closure"} and validation["before_after"] == {name: {half: {"verdict": p[half]["verdict"], "false_completion": p[half]["false_completion"], "placed_by_oracle": p[half]["placed_by_oracle"]} for half in ("before", "after")} for name, p in pairs.items()}
    assert pairs["pitch"]["before"]["layout"] == "v1" and pairs["pitch"]["after"]["layout"] == "v3" and pairs["pitch"]["before"]["observe_each_pass"] and pairs["pitch"]["after"]["observe_each_pass"]
    assert pairs["look"]["before"]["layout"] == pairs["look"]["after"]["layout"] == "v1" and not pairs["look"]["before"]["observe_each_pass"] and pairs["look"]["after"]["observe_each_pass"]
    assert pairs["closure"]["before"]["layout"] == pairs["closure"]["after"]["layout"] == "v2" and not pairs["closure"]["before"]["closure_fix"] and pairs["closure"]["after"]["closure_fix"]
    assert all(p[half].get("closure_fix", True) for name, p in pairs.items() if name != "closure" for half in ("before", "after"))
    assert not pairs["pitch"]["before"]["opening_sized"] and pairs["pitch"]["after"]["opening_sized"]
    assert all(not p[half]["opening_sized"] for name, p in pairs.items() if name != "pitch" for half in ("before", "after")), "the look and closure pairs hold the opening fixed"
    for name, pair in pairs.items():
        for half in ("before", "after"):
            manifest = check_media(BEFORE_AFTER / Path(pair[half]["video"]).parent)
            assert manifest["metadata"]["frame_count"] == pair[half]["frame_count"]
            row = json.loads((BEFORE_AFTER / pair[half]["row"]).read_bytes())
            assert row["verdict"] == pair[half]["verdict"] and row["seed"] == pair["seed"] and row["draw_sha256"] == hashlib.sha256(json_bytes(next(d for d in corpus["nominal"] if d["seed"] == pair["seed"]))).hexdigest()
            assert row["library_id"] == ("clear_work_area_v3_tree_" if pair[half]["observe_each_pass"] else "clear_work_area_v1_tree_") + str(pair["objects"])
            if args.replay:
                for seg in row["sealed"]["segments"]:
                    if Path(seg["bundle"]).exists():
                        check_segment(Path(seg["bundle"]), seg["sha256"], replay=True)
        assert (BEFORE_AFTER / pair["pair"]["video"]).is_file() and (BEFORE_AFTER / pair["pair"]["preview"]["gif"]).is_file()
    if "disturbance" in index:
        for executor in ("tree", "flat"):
            manifest = check_media(D15 / Path(index["disturbance"][executor]["video"]).parent)
            assert manifest["metadata"]["frame_count"] == index["disturbance"][executor]["frame_count"]
            assert index["disturbance"][executor]["episode_id"] in validation.get("d15_episodes", [])
        assert (D15 / index["disturbance"]["pair"]["video"]).is_file() and (D15 / index["disturbance"]["pair"]["preview"]["gif"]).is_file()
    print(json.dumps({"verified": True, "nominal": {b: {n: f"{c['successes']}/{c['episodes']}" for n, c in v.items()} for b, v in validation["nominal"].items()}, "disturbed": {b: f"{c['tree']}/{c['flat']} (+{c['gain_pp']:.0f} pp)" for b, c in validation["disturbed"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
