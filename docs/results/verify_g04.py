"""Verify the G04 semantic evidence: the fixture's registration, every case's score, the spend, D04.

Recomputes every claim the report makes from the committed files: the
fixture, labels and inventory against the registration, and the model's
system prompt and reply schema against the code as it stands; every stored
reading re-scored against its expected program with the campaign's own
rules, the per-kind and per-family counts re-derived; the body agreement
rows; the call log's tokens and cost summed against the ledger figure, with
every evidence file scanned for anything shaped like a key; the D04 index
against its media bundles and frame maps, the stated distance against the
measured travel, and the two qualitative prompts against their per-body
travel. With --replay, every local D04 physical bundle is replayed on native
physics and must agree exactly.

    python docs/results/verify_g04.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FIXTURE = REPO / "any-robot/assets/general/research-protocols/g04-semantics-v1"
RESULTS = ROOT / "g04-semantics"
D04 = ROOT / "g04-d04"
LOCAL = REPO / "any-robot/results/g04-d04"
KEY_SHAPES = (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), re.compile(r"AIza[0-9A-Za-z_\-]{20,}"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_media(media: Path) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"], media
    assert abs(manifest["metadata"]["playback_duration_s"] - len(frames) / manifest["metadata"]["fps"]) < 1e-6
    return manifest


def no_keys(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for shape in KEY_SHAPES:
        assert not shape.search(text), f"{path} carries something shaped like a key"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "any-robot/scripts"))
    sys.path.insert(0, str(REPO / "any-robot/src"))
    sys.path.insert(0, str(REPO / "core/src"))
    import g04_semantics_campaign as campaign
    from rigby_general.planner.model_planner import reply_schema, system_prompt
    from rigby_general.schema.inventory import INVENTORY_PATH, load_inventory

    inventory = load_inventory()
    for entry in inventory.entries:
        campaign._ENTRY_BY_BINDING[entry.binding_key] = entry.entry_id
    validation = json.loads((ROOT / "g04-validation.json").read_bytes())

    # -- registration -------------------------------------------------------------
    registration = json.loads((FIXTURE / "registration.json").read_bytes())
    for name in ("semantic-cases.json", "labels.json"):
        assert sha256(FIXTURE / name) == registration["files"][name], name
    assert sha256(INVENTORY_PATH) == registration["files"]["schema_inventory.v1.json"]
    assert hashlib.sha256(system_prompt(inventory).encode("utf-8")).hexdigest() == registration["system_prompt_sha256"], "the system prompt changed after registration"
    assert hashlib.sha256(json.dumps(reply_schema(inventory), sort_keys=True).encode("utf-8")).hexdigest() == registration["reply_schema_sha256"]
    assert registration["registration_sha256"] == validation["fixture_registration"]
    model = registration["selected_model"]
    assert model == validation["model"]

    # -- the fixture --------------------------------------------------------------
    fixture = json.loads((FIXTURE / "semantic-cases.json").read_bytes())
    cases = fixture["cases"]
    by_id = {case["case_id"]: case for case in cases}
    canonical = [case for case in cases if case["kind"] == "canonical"]
    language = [case for case in cases if case["kind"] == "language"]
    assert len(canonical) == 60 and len(language) == 120
    assert all(case["label_provenance"] == "internal" for case in cases)
    families = {case["family"] for case in language}
    assert {"paraphrase", "minimal_contrast", "negation", "ambiguous_deixis", "explicit_quantity"} <= families
    entries_spanned = {seg["entry_id"] for case in canonical for seg in case["expected"].get("segments", [])}
    assert entries_spanned == {entry.entry_id for entry in inventory.entries}, "the canonical cases must span the inventory"
    labels = json.loads((FIXTURE / "labels.json").read_bytes())
    assert labels["provenance"] == "internal" and labels["independent_review"]["status"] == "pending"
    assert set(labels["cases"]) == set(by_id)

    # -- every stored reading re-scored ------------------------------------------------
    stored = json.loads((RESULTS / "cases.json").read_bytes())
    summary = json.loads((RESULTS / "summary.json").read_bytes())
    assert summary["model"] == model and summary["fixture_registration"] == registration["registration_sha256"]
    recount = {}
    for planner_name, rows in stored.items():
        assert [row["case_id"] for row in rows] == [case["case_id"] for case in cases], f"{planner_name}: every case once, in order"
        for row in rows:
            case = by_id[row["case_id"]]
            verdict = campaign.score(case, campaign.expected_program(case, inventory), row["actual"])
            assert verdict["exact"] == row["exact"] and verdict["prohibited_rejected"] == row["prohibited_rejected"] and verdict["error"] == row["error"], (planner_name, row["case_id"])
            if "hash" in row["actual"]:
                assert row["actual"]["planner_id"] == ("offline-recognizer-v1" if planner_name == "offline" else "model-schema-planner-v1")
        recount[planner_name] = {kind: {"exact": sum(1 for r in rows if r["kind"] == kind and r["exact"]), "cases": sum(1 for r in rows if r["kind"] == kind),
                                        "prohibited_cases": sum(1 for r in rows if r["kind"] == kind and by_id[r["case_id"]]["prohibited"]),
                                        "prohibited_rejected": sum(1 for r in rows if r["kind"] == kind and by_id[r["case_id"]]["prohibited"] and r["prohibited_rejected"])}
                                 for kind in ("canonical", "language")}
    for planner_name, claimed_key in (("offline", "offline"), ("model", "model_planner")):
        claimed = summary[claimed_key]["by_kind"]
        for kind, cell in recount[planner_name].items():
            assert claimed[kind]["exact"] == cell["exact"] and claimed[kind]["cases"] == cell["cases"], (planner_name, kind)
            assert claimed[kind]["prohibited_cases"] == cell["prohibited_cases"] and claimed[kind]["prohibited_rejected"] == cell["prohibited_rejected"], (planner_name, kind)
    model_cells = recount["model"]
    assert model_cells["canonical"]["exact"] == validation["model_planner"]["canonical_exact"] == 60, "canonical symbolic cases must be exact"
    assert model_cells["language"]["exact"] == validation["model_planner"]["language_exact"]
    assert model_cells["language"]["prohibited_rejected"] == model_cells["language"]["prohibited_cases"] == validation["model_planner"]["prohibited_cases"], "every prohibited substitution rejected"
    assert recount["offline"]["canonical"]["exact"] == validation["offline"]["canonical_exact"]
    assert recount["offline"]["language"]["exact"] == validation["offline"]["language_exact"]

    # -- errors and unsupported coverage published apart --------------------------------
    confusion = json.loads((RESULTS / "confusion.json").read_bytes())
    unsupported = json.loads((RESULTS / "unsupported.json").read_bytes())
    model_rows = stored["model"]
    errors = {r["case_id"] for r in model_rows if not r["exact"] and not (r["error"] or "").startswith("refusal_instead_of_reading")}
    refused = {r["case_id"] for r in model_rows if (r["error"] or "").startswith("refusal_instead_of_reading")}
    assert {e["case_id"] for e in confusion["model"]["errors"]} == errors
    assert {u["case_id"] for u in unsupported["model"]} == refused
    assert errors.isdisjoint(refused)

    # -- body agreement ---------------------------------------------------------------------
    agreement = json.loads((RESULTS / "agreement.json").read_bytes())
    for planner_name, rows in agreement.items():
        assert [r["case_id"] for r in rows] == [c["case_id"] for c in canonical]
        for row in rows:
            case = by_id[row["case_id"]]
            needed = [s["entry_id"] for s in case["expected"]["segments"]]
            for body, cell in row["bodies"].items():
                if planner_name == "model":
                    assert cell["agrees"], (planner_name, row["case_id"], body)
                if not cell["applicable"]:
                    assert cell["missing"] and set(cell["missing"]) <= set(needed)
            assert row["all_agree"] == all(cell["agrees"] for cell in row["bodies"].values())
        assert summary["agreement"][planner_name]["all_agree"] == sum(1 for r in rows if r["all_agree"])
    assert summary["agreement"]["model"]["all_agree"] == 60 == validation["agreement"]["model_all_agree"]

    # -- the spend --------------------------------------------------------------------------------
    logs = [RESULTS / "calls.jsonl", *sorted(RESULTS.glob("round-*/calls.jsonl")), D04 / "calls.jsonl"]
    paid_calls, dollars, prompt_tokens, completion_tokens = 0, 0.0, 0, 0
    for log in logs:
        if not log.exists():
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for field in ("model", "transport", "purpose", "prompt_tokens", "completion_tokens", "cost_usd", "rates_usd_per_mtok", "cached", "paid"):
                assert field in row, (log, field)
            if row["paid"]:
                paid_calls += 1
                dollars += row["cost_usd"]
                prompt_tokens += row["prompt_tokens"]
                completion_tokens += row["completion_tokens"]
                assert row["transport"] == "openai" and not row["tokens_estimated"], "a paid call is an OpenAI call with reported tokens"
    assert paid_calls == validation["spend"]["paid_calls"], (paid_calls, validation["spend"]["paid_calls"])
    assert abs(dollars - validation["spend"]["dollars"]) < 1e-4, (dollars, validation["spend"]["dollars"])
    assert dollars < validation["spend"]["soft_cap_usd"]
    assert summary["fallback_used"] is False and summary["stopped"] is None
    for path in list(RESULTS.rglob("*.json")) + list(RESULTS.rglob("*.jsonl")) + list(FIXTURE.rglob("*.json")) + [ROOT / "g04-report.md", ROOT / "g04-validation.json"]:
        no_keys(path)
    if (D04 / "index.json").exists():
        no_keys(D04 / "index.json")

    # -- D04 ------------------------------------------------------------------------------------------
    index = json.loads((D04 / "index.json").read_bytes())
    assert index["fixture_registration"] == registration["registration_sha256"]
    assert len(index["clips"]) == 9 and (D04 / index["tile"]["video"]).is_file() and (D04 / index["tile"]["preview"]["gif"]).is_file()
    tile_frames = json.loads((D04 / index["tile"]["frames"]).read_bytes())
    clip_maps = [json.loads((D04 / clip["frames"]).read_bytes()) for clip in index["clips"]]
    assert len(tile_frames) == max(len(m) for m in clip_maps), "the tile runs as long as its longest clip"
    for frame, row in enumerate(tile_frames):
        assert row["frame"] == frame and set(row["clips"]) == {f"{c['body']}/{c['slug']}" for c in index["clips"]}
    travel = {}
    for clip in index["clips"]:
        manifest = check_media(D04 / clip["media"])
        assert manifest["metadata"]["source_bundle_sha256"] == clip["physical_sha256"]
        measurements = json.loads((D04 / clip["media"] / "measurements.json").read_bytes())
        assert measurements["final_travel_m"] == clip["measurements"]["final_travel_m"]
        travel.setdefault(clip["slug"], {})[clip["body"]] = (clip["outcome"], measurements["final_travel_m"], clip["reading"]["hash"])
        if args.replay:
            physical = Path(clip["physical_bundle"])
            if physical.exists():
                from rigby_core.simulation.recording import PhysicsRecord, replay_physics
                import mujoco
                assert sha256(physical / "manifest.json") == clip["physical_sha256"]
                record = PhysicsRecord.from_bytes((physical / "trace.npz").read_bytes())
                model_ = mujoco.MjModel.from_binary_path(str(physical / "model.mjb"))
                assert replay_physics(model_, record)["agrees"], physical
    for slug, per_body in travel.items():
        hashes = {h for _, _, h in per_body.values() if h}
        assert len(hashes) == 1, f"{slug}: one reading on every body"
    five = travel["five_cm"]
    for body, (outcome, metres, _) in five.items():
        if outcome == "success":
            assert abs(metres - 0.05) <= 0.004, (body, metres)
    for slug in ("little", "edge"):
        metres = sorted(m for outcome, m, _ in travel[slug].values() if outcome == "success")
        assert len(metres) >= 2 and metres[-1] - metres[0] > 0.01, f"{slug}: two distinct body-relative distances"
    assert index["calls"]["paid_calls"] == validation["d04"]["paid_calls"]

    print(json.dumps({"verified": True, "model": model, "canonical_exact": f"{model_cells['canonical']['exact']}/60", "language_exact": f"{model_cells['language']['exact']}/120",
                      "prohibited_rejected": f"{model_cells['language']['prohibited_rejected']}/{model_cells['language']['prohibited_cases']}",
                      "paid_calls": paid_calls, "dollars": round(dollars, 4), "d04_travel_m": {slug: {b: round(m, 4) for b, (_, m, _) in per.items()} for slug, per in travel.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
