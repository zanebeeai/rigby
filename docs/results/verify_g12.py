"""Verify the G12 acquisition evidence: the registration, every search attempt, every held-out trial, the classification, the store, D12.

Recomputes every claim the report makes from the committed files: the
protocol against its registration; per problem, the attempts' digest
against the outcome's, the ceiling never exceeded, the confirmation only
after a full development pass, the settled vector inside every bound; the
held-out rows against the registered draws (every seed once, in order),
the successes recounted and the acceptance re-derived from the threshold;
the classification re-derived from the attempts and the acceptance (an
instantiation only when attempt 0 passed with nothing moved, a discovery
only when a searched vector passed, and neither when the acceptance
failed); the persisted store's certificate for an acquired or instantiated
vector, its controller facet carrying that vector; every media bundle's
digests and frame maps; the D12 index against its panels. With --replay,
every local physical bundle is replayed on native physics and must agree.

    python docs/results/verify_g12.py [--replay]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = REPO / "any-robot/assets/general/research-protocols/g12-acquisition-v1"
STORE = REPO / "any-robot/assets/general/skill-store-v1"
CAMPAIGN = ROOT / "g12-acquisition"
D12 = ROOT / "g12-d12"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_media(media: Path) -> dict:
    manifest = json.loads((media / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(media / name) == info["sha256"], (media, name)
    frames = json.loads((media / "frames.json").read_bytes())
    assert len(frames) == manifest["metadata"]["frame_count"], media
    return manifest


def check_physical(physical: Path, expected: str, *, replay: bool) -> None:
    assert sha256(physical / "manifest.json") == expected, physical
    manifest = json.loads((physical / "manifest.json").read_bytes())
    for name, info in manifest["files"].items():
        assert sha256(physical / name) == info["sha256"], (physical, name)
    task = json.loads((physical / "task.json").read_bytes())
    assert task.get("manual_trajectory_edits", 0) == 0 and task.get("retry_limit", 0) == 0
    if replay:
        import mujoco
        from rigby_core.simulation.recording import PhysicsRecord, replay_physics

        record = PhysicsRecord.from_bytes((physical / "trace.npz").read_bytes())
        assert record.content_hash() == manifest["metadata"]["trace_sha256"]
        if len(record.arrays["time_s"]) > 1:
            model = mujoco.MjModel.from_binary_path(str(physical / "model.mjb"))
            assert replay_physics(model, record)["agrees"], physical


def check_row(row: dict, root: Path, *, replay: bool) -> None:
    if "media" in row:
        manifest = check_media(root / row["media"])
        assert manifest["metadata"]["source_bundle_sha256"] == row["bundle"]["sha256"]
        assert sha256(root / row["media"] / "manifest.json") == row["media_sha256"]
    if "bundle" in row:
        physical = Path(row["bundle"]["bundle"])
        if physical.exists():
            check_physical(physical, row["bundle"]["sha256"], replay=replay)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "core/src"))
    sys.path.insert(0, str(REPO / "any-robot/src"))
    from rigby_core.skills import AcquisitionProblemV1, AttemptV1, SkillStoreV1, attempts_digest, classify
    from rigby_core.skills.examples import transfer_object_library

    validation = json.loads((ROOT / "g12-validation.json").read_bytes())
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        assert sha256(PROTOCOL / name) == digest, name
    assert registration["registration_sha256"] == validation["registration_sha256"]
    assert transfer_object_library().content_hash() == registration["library_sha256"]
    problems = {p["problem_id"]: AcquisitionProblemV1.model_validate(p) for p in json.loads((PROTOCOL / "problems.json").read_bytes())["problems"]}
    draws = json.loads((PROTOCOL / "draws.json").read_bytes())["problems"]
    assert set(problems) == set(validation["problems"]) == {"jaw-slick", "hand-fixed-world", "long-heavy-large"}
    families = {p.body_family for p in problems.values()}
    assert len(families) >= 2
    store = SkillStoreV1.load(STORE)
    index = json.loads((D12 / "index.json").read_bytes())
    by_problem = {p["problem_id"]: p for p in index["problems"]}

    for problem_id, problem in problems.items():
        report = json.loads((CAMPAIGN / problem_id / "problem.json").read_bytes())
        assert report["provenance"]["registration_sha256"] == registration["registration_sha256"] and report["provenance"]["generation_calls"] == 0 and report["provenance"]["manual_trajectory_edits"] == 0
        attempts = [AttemptV1.model_validate(a) for a in report["attempts"]]
        outcome = report["outcome"]
        budget = report["search"]["budget"]
        assert attempts_digest(attempts) == outcome["attempts_sha256"]
        assert len(attempts) <= problem.ceiling.attempts and budget["attempts"] == len(attempts)
        assert budget["worker_minutes"] <= problem.ceiling.worker_minutes + 2.0, "one attempt may overrun the ceiling by its own length, no more"
        assert [a.index for a in attempts] == list(range(len(attempts)))
        assert attempts[0].parameters == problem.defaults(), "attempt 0 asks whether search is needed at all"
        for attempt in attempts:
            for spec in problem.parameters:
                assert spec.low <= attempt.parameters[spec.name] <= spec.high, (problem_id, attempt.index, spec.name)
            assert [e.seed for e in attempt.episodes] == list(problem.development_seeds[: len(attempt.episodes)])
            if attempt.confirmation:
                assert attempt.certified == attempt.of, "confirmation only after a full development pass"
                assert [e.seed for e in attempt.confirmation] == list(problem.confirmation_seeds)
        best = attempts[outcome["best_attempt"]] if outcome["best_attempt"] is not None else None
        confirmed = best is not None and best.certified == best.of and bool(best.confirmation) and all(e.certified for e in best.confirmation)
        assert report["search"]["confirmed"] == confirmed
        # held-out
        holdout = report["holdout"]
        registered = draws[problem_id]["holdout"]
        assert holdout["set"] == registered["set"]
        rows = holdout["rows"]
        assert [r["draw"] for r in rows] == registered["draws"] and len(rows) == problem.acceptance.trials
        successes = sum(1 for r in rows if r["certified"])
        assert holdout["successes"] == successes and holdout["accepted"] == (successes >= problem.acceptance.threshold)
        assert all("bundle" in r for r in rows if not r["certified"]), "every held-out failure is sealed"
        for r in rows:
            check_row(r, CAMPAIGN / problem_id, replay=args.replay)
        # classification
        status, kind, changed = classify(problem, best if confirmed else None, accepted=holdout["accepted"] and confirmed)
        assert outcome["status"] == status.value and outcome["kind"] == (None if kind is None else kind.value) and tuple(outcome["parameters_changed"]) == changed, (problem_id, outcome["status"], status)
        if status.value == "instantiated":
            assert best.index == 0 and not changed
        if status.value == "acquired":
            assert changed and best.index > 0
        if status.value in ("budget_exhausted", "acceptance_failed"):
            assert outcome["limiting_capability"]
        assert validation["problems"][problem_id] == {"status": status.value, "kind": None if kind is None else kind.value, "attempts": len(attempts), "worker_minutes": round(budget["worker_minutes"], 2),
                                                        "holdout": f"{successes}/{len(rows)}", "changed": list(changed)}
        for name, shown in report.get("shown_attempts", {}).items():
            check_row(shown, CAMPAIGN / problem_id, replay=args.replay)
            assert shown["parameters"] == attempts[shown["attempt"]].parameters
        # the store
        if status.value in ("acquired", "instantiated"):
            certificate = store.certificate(report["store"]["certificate_id"])
            assert certificate.status.value == report["store"]["status"] == "promoted"
            facet = certificate.context.facet("controller") if hasattr(certificate.context, "facet") else None
            from rigby_core.skills import ContextDimension
            facet = certificate.context.facet(ContextDimension.CONTROLLER)
            settled = report["search"]["settled_parameters"]
            assert abs(facet.detail["duration_scale"] - settled["duration_scale"]) < 1e-9
            assert abs(facet.detail["closure_config"]["grip_safety_factor"] - settled["closure.grip_safety_factor"]) < 1e-9
            assert certificate.validation.successes == successes and certificate.validation.set_id == problem.acceptance.set_id
            reuse = report["reuse"]
            assert reuse["retrieval"]["chosen"] == certificate.certificate_id and reuse["executed"]
            check_row(reuse, CAMPAIGN / problem_id, replay=args.replay)
        # D12
        d12 = by_problem[problem_id]
        assert len(d12["panels"]) == 5 and d12["status"] == status.value
        tile = d12["tile"]
        assert (D12 / tile["video"]).is_file() and (D12 / tile["preview_gif"]).is_file()
        frames = json.loads((D12 / tile["frames"]).read_bytes())
        assert len(frames) == tile["frame_count"]
        for panel in d12["panels"]:
            assert (D12 / panel["video"]).is_file()
    print(json.dumps({"verified": True, "problems": validation["problems"], "store_version": store.version}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
