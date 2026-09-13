"""Run the registered G12 protocol: search within the ceiling, test on the held-out draws, classify, store, reuse.

For each frozen problem, in its own process:

1. The search proposes parameter vectors of the contact transfer family and
   runs the development draws single-shot with every hard gate on; a vector
   that certifies them all is confirmed on the confirmation draws. Every
   attempt is logged as it happens. The search stops when confirmed or at
   the ceiling (200 attempts or 30 simulator minutes).
2. The settled vector runs the fifty held-out draws once; failures and the
   first successes are sealed and rendered. At least forty must certify.
3. The outcome is classified: an instantiation when the defaults passed
   with nothing moved, a discovery when a searched vector passed, budget
   exhausted or acceptance failed otherwise, with the limiting capability
   named. Nothing is rebranded.
4. An acquired or instantiated vector is promoted into the persisted G11
   store as a certificate whose controller facet carries it, on the
   held-out outcome as its independent validation, and the transfer tree is
   run once from the store in a new layout under it, sealed and rendered,
   as the reuse D12 shows.

Attempts that the demo shows -- the defaults failing, a failed searched
attempt, the confirming attempt -- are re-run with the recorder afterwards,
which is exact because the physics is deterministic. No API or model calls.

    python any-robot/scripts/g12_acquisition_campaign.py jaw-slick --out docs/results/g12-acquisition --local any-robot/results/g12-acquisition
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.skills import AcquisitionOutcomeV1, AcquisitionProblemV1, AcquisitionStatus, AttemptV1, CertificateV1, CostV1, RangeV1, SkillStoreV1, ValidationOutcomeV1, ValidationSetV1
from rigby_core.skills.examples import transfer_object_library

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import load_policy
from rigby_general.skills import seal_tree_run
from rigby_general.skills.acquisition_runtime import PROTOCOL as EPISODE_PROTOCOL, acquire, configs_of, outcome_of, problem_world, run_single_shot, seal_single_shot, summarize
from rigby_general.skills.skill_store import CLAIMED_RANGES, DEVELOPMENT_SETS, EVIDENCE_SCHEMA, context_of, goal_for, perturbed, rows_digest, run_tree
from rigby_general.skills.transfer_object import TransferObjectSession

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402
import g12_protocol as protocol  # noqa: E402
from g11_skill_store_campaign import STORE, episode_row  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RENDER_SUCCESSES = 2
LIMITING = {
    "hand-fixed-world": "the multifinger hand's digits are 47 mm long against a 30 mm cube and its closure is a single-parameter squeeze, so the only grasp that clears the bench is a fingertip pinch; "
                        "within the contact transfer family no closure force, rate, tracking bandwidth or slowing of the lift gave that pinch a margin against the load of the lift. A wrap grasp, a compliant pad or a two-parameter closure is a different primitive, not a retuned one.",
    "jaw-slick": "within the contact transfer family no closure force, rate, tracking bandwidth or slowing of the lift held a cube at half the registered friction through the hold and carry on the jaw arm",
    "long-heavy-large": "within the contact transfer family no parameter vector carried the grown, heavier cube on the long arm",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registration() -> dict:
    registration = json.loads((protocol.PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        if sha256_of(protocol.PROTOCOL / name) != digest:
            raise SystemExit(f"{name} does not match its registration; refusing to run")
    if transfer_object_library().content_hash() != registration["library_sha256"]:
        raise SystemExit("the transfer_object library no longer hashes to the registration")
    if hashlib.sha256(json_bytes(g10.environment().model_dump(mode="json"))).hexdigest() != registration["environment_sha256"]:
        raise SystemExit("the world no longer hashes to the registration")
    return registration


def claimed_ranges_for(world: EnvironmentV1, friction_assumption: dict | None) -> tuple[RangeV1, ...]:
    """What a certificate acquired in this world claims: the G11 ranges, with
    the friction range the problem assumed and a mass range that covers the
    problem's cube with a fifth to spare; the held-out draws sample both."""

    cube = world.objects[0]
    ranges = []
    for bound in CLAIMED_RANGES:
        if bound.quantity == "object_friction" and friction_assumption:
            low, high = friction_assumption["object_friction_range"]
            ranges.append(RangeV1(quantity="object_friction", low=float(low), high=float(high), units=bound.units))
        elif bound.quantity == "object_mass_kg":
            ranges.append(RangeV1(quantity="object_mass_kg", low=min(bound.low, cube.mass_kg * 0.8), high=max(bound.high, cube.mass_kg * 1.25), units=bound.units))
        else:
            ranges.append(bound)
    return tuple(ranges)


def render_with_retry(bundle_dir: Path, media_dir: Path, digest: str) -> dict:
    """The render is a pure function of the sealed bundle; ffmpeg has died
    mid-pipe under load before, so a failed render is retried after a pause
    that grows, with the interpreter's garbage collected in between."""

    last = None
    for pause in (3.0, 8.0, 20.0, 45.0):
        if media_dir.exists():
            shutil.rmtree(media_dir)
        try:
            return render_bundle(bundle_dir, media_dir, expected_digest=digest)
        except Exception as error:  # noqa: BLE001 - every failure mode seen so far was transient
            last = error
            gc.collect()
            time.sleep(pause)
    raise RuntimeError(f"rendering {media_dir} failed four times: {last!r}")


def seal_trial(problem: AcquisitionProblemV1, source: Path, world: EnvironmentV1, goal, policy: dict, parameters: dict, *, seed_label: str, name: str, local: Path, out: Path, caption: str, task_extra: dict) -> tuple[dict, dict]:
    session, result, wall = run_single_shot(problem.body, source, world, goal, policy, parameters, seed_label=seed_label, record=True)
    bundle_dir = local / name / "physical"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    sealed = seal_single_shot(bundle_dir, session=session, result=result, label=seed_label, caption=caption, parameters=parameters, task_extra=task_extra)
    media = render_with_retry(bundle_dir, out / name / "media", sealed["sha256"])
    row = {"bundle": sealed, "media_sha256": media["sha256"], "frames": media["frame_count"], "media": (out / name / "media").relative_to(out).as_posix()}
    return row, outcome_of(0, result, wall).model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--skip-store", action="store_true", help="do not promote into the persisted store or run the reuse (engineering smoke)")
    parser.add_argument("--store-only", action="store_true", help="resume from problem.json: redo only the store promotion and the reuse")
    args = parser.parse_args()
    registration = load_registration()
    problems = json.loads((protocol.PROTOCOL / "problems.json").read_bytes())
    draws = json.loads((protocol.PROTOCOL / "draws.json").read_bytes())
    problem = next(AcquisitionProblemV1.model_validate(p) for p in problems["problems"] if p["problem_id"] == args.problem)
    policy = load_policy(g10.G09 / "policy.json")
    base = g10.environment()
    world = problem_world(problem, base, problems["changes"])
    goal = goal_for(world)
    source = ROOT / "assets/general/zoo" / problem.body / "robot.urdf"
    out = args.out / problem.problem_id
    local = args.local / problem.problem_id
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__, "registration_sha256": registration["registration_sha256"],
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0, "manual_trajectory_edits": 0}
    development = draws["problems"][problem.problem_id]["development"]
    by_seed = {int(d["seed"]): d for d in development["draws"]}
    if args.store_only:
        report = json.loads((out / "problem.json").read_bytes())
        attempts = [AttemptV1.model_validate(a) for a in report["attempts"]]
        best = attempts[report["search"]["best_attempt"]] if report["search"]["best_attempt"] is not None else None
        settled = report["search"]["settled_parameters"]
        rows = report["holdout"]["rows"]
        successes = report["holdout"]["successes"]
        budget = report["search"]["budget"]
        holdout_set = ValidationSetV1.model_validate(report["holdout"]["set"])
        outcome = AcquisitionOutcomeV1.model_validate(report["outcome"])
        report.pop("store", None)
        report.pop("reuse", None)
        return store_and_reuse(registration, problems, problem, source, world, goal, policy, report, attempts, settled, rows, successes, budget, holdout_set, outcome, out, local)
    report: dict = {"schema": "g12.problem.v1", "problem": problem.model_dump(mode="json"), "world": world.model_dump(mode="json"), "provenance": provenance, "attempts": []}

    # -- 1. the search -----------------------------------------------------------------------------
    def on_attempt(attempt, worker_total):
        report["attempts"].append(attempt.model_dump(mode="json"))
        (out / "attempts.json").write_bytes(json_bytes({"problem_id": problem.problem_id, "attempts": report["attempts"], "worker_minutes_so_far": worker_total / 60.0}))
        print(json.dumps({"attempt": attempt.index, "certified": f"{attempt.certified}/{attempt.of}", "confirmation": [e.certified for e in attempt.confirmation], "improved": attempt.improved,
                          "sigma": round(attempt.sigma, 3), "worker_min": round(worker_total / 60.0, 2), "gates": [e.failed_gate for e in attempt.episodes if not e.certified]}), flush=True)

    attempts, best, search, budget = acquire(problem, source, world, goal, policy, by_seed, search_seed=problems["search"]["seed"], on_attempt=on_attempt)
    confirmed = best is not None and best.certified == best.of and best.confirmation and all(e.certified for e in best.confirmation)
    settled = dict(best.parameters) if best is not None else problem.defaults()
    report["search"] = {"budget": budget, "confirmed": bool(confirmed), "best_attempt": None if best is None else best.index, "settled_parameters": settled, "provenance": search.provenance().model_dump(mode="json")}
    print(json.dumps({"search": budget, "confirmed": bool(confirmed), "best": None if best is None else best.index}), flush=True)

    # -- 2. the held-out test ------------------------------------------------------------------------
    holdout = draws["problems"][problem.problem_id]["holdout"]
    holdout_set = ValidationSetV1.model_validate(holdout["set"])
    rows = []
    rendered = 0
    for draw in holdout["draws"]:
        trial_world = perturbed(world, draw)
        label = f"{problem.problem_id}-holdout-{draw['seed']}"
        session, result, wall = run_single_shot(problem.body, source, trial_world, goal, policy, settled, seed_label=label, record=True)
        episode = outcome_of(int(draw["seed"]), result, wall).model_dump(mode="json")
        row = {"episode_id": label, "draw": draw, **episode}
        keep = not result.certified or rendered < RENDER_SUCCESSES
        if keep:
            bundle_dir = local / "holdout" / label / "physical"
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            sealed = seal_single_shot(bundle_dir, session=session, result=result, label=label, caption=f"{problem.body} | {problem.problem_id} | held-out {draw['seed']} | {'certified' if result.certified else result.failed_gate}",
                                      parameters=settled, task_extra={"held_out": True, "registration_sha256": registration["registration_sha256"], "draw": draw})
            media = render_with_retry(bundle_dir, out / "holdout" / label / "media", sealed["sha256"])
            row.update({"bundle": sealed, "media_sha256": media["sha256"], "frames": media["frame_count"], "media": (out / "holdout" / label / "media").relative_to(out).as_posix()})
            if result.certified:
                rendered += 1
        rows.append(row)
        print(json.dumps({"holdout": draw["seed"], "certified": result.certified, "gate": result.failed_gate, "physics_s": round(episode["physics_s"], 2), "kept": keep}), flush=True)
        (out / "holdout.json").write_bytes(json_bytes({"set": holdout["set"], "rows": rows}))
    successes = sum(1 for r in rows if r["certified"])
    accepted = successes >= problem.acceptance.threshold
    taxonomy: dict[str, int] = {}
    for r in rows:
        if not r["certified"]:
            taxonomy[r["failed_gate"] or "failed"] = taxonomy.get(r["failed_gate"] or "failed", 0) + 1
    report["holdout"] = {"set": holdout["set"], "trials": len(rows), "successes": successes, "threshold": problem.acceptance.threshold, "accepted": accepted, "taxonomy": taxonomy, "rows_sha256": rows_digest(rows),
                         "physics_s": sum(r["physics_s"] for r in rows), "wall_s": sum(r["wall_s"] for r in rows), "rows": rows}

    # -- 3. classification ------------------------------------------------------------------------------
    outcome = summarize(problem, attempts, best, search, budget, accepted=accepted and bool(confirmed), limiting=LIMITING.get(problem.problem_id, ""))
    report["outcome"] = outcome.model_dump(mode="json")
    print(json.dumps({"outcome": outcome.status.value, "kind": None if outcome.kind is None else outcome.kind.value, "holdout": f"{successes}/{len(rows)}", "changed": list(outcome.parameters_changed)}), flush=True)

    # -- the attempts the demo shows, re-run with the recorder ----------------------------------------------
    shown = {"defaults": attempts[0] if attempts else None}
    failed_searched = next((a for a in attempts[1:] if a.certified < a.of), None)
    if failed_searched is not None:
        shown["failed_search"] = failed_searched
    if confirmed:
        shown["confirming"] = best
    report["shown_attempts"] = {}
    for name, attempt in shown.items():
        if attempt is None:
            continue
        draw = by_seed[int(attempt.episodes[0].seed)]
        row, episode = seal_trial(problem, source, perturbed(world, draw), goal, policy, dict(attempt.parameters), seed_label=f"{problem.problem_id}-attempt-{attempt.index:03d}-{draw['seed']}", name=f"attempts/{name}",
                                  local=local, out=out, caption=f"{problem.body} | {problem.problem_id} | attempt {attempt.index} ({name.replace('_', ' ')}) | {'certified' if attempt.episodes[0].certified else attempt.episodes[0].failed_gate}",
                                  task_extra={"attempt": attempt.index, "shown_as": name, "registration_sha256": registration["registration_sha256"], "draw": draw})
        if episode["certified"] != attempt.episodes[0].certified:
            raise SystemExit(f"re-running attempt {attempt.index} with the recorder did not reproduce its outcome")
        report["shown_attempts"][name] = {"attempt": attempt.index, "parameters": dict(attempt.parameters), **row, "episode": episode}
    (out / "problem.json").write_bytes(json_bytes(report))

    if args.skip_store:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "problem.json").write_bytes(json_bytes(report))
        return 0
    return store_and_reuse(registration, problems, problem, source, world, goal, policy, report, attempts, settled, rows, successes, budget, holdout_set, outcome, out, local)


def store_and_reuse(registration, problems, problem, source, world, goal, policy, report, attempts, settled, rows, successes, budget, holdout_set, outcome, out, local) -> int:
    # -- 4. store and reuse ------------------------------------------------------------------------------------
    if outcome.status in (AcquisitionStatus.ACQUIRED, AcquisitionStatus.INSTANTIATED):
        library = transfer_object_library()
        controller, closure, duration_scale = configs_of(settled)
        friction_assumption = problems["friction_assumptions"].get(problem.problem_id)
        probe = TransferObjectSession.open(problem.body, source, world, goal, policy, seed_label="context", controller_config=controller, closure_config=closure, duration_scale=duration_scale)
        ranges = claimed_ranges_for(world, friction_assumption)
        context = context_of(probe, friction_assumption=friction_assumption, ranges=ranges)
        validation = ValidationOutcomeV1(set_id=holdout_set.set_id, episodes=len(rows), successes=successes, false_completions=0, threshold=problem.acceptance.threshold,
                                         evidence=tuple(r["bundle"]["sha256"] for r in rows if "bundle" in r), rows_sha256=rows_digest(rows))
        cost = CostV1(physics_s=float(budget["physics_minutes"] * 60.0 + report["holdout"]["physics_s"]), wall_s=float(budget["wall_minutes"] * 60.0 + report["holdout"]["wall_s"]),
                      episodes=sum(len(a.episodes) + len(a.confirmation) for a in attempts) + len(rows), generation_calls=0, dollars=0.0)
        store = SkillStoreV1.load(STORE).add_library(library)
        candidate = CertificateV1.candidate(skill_id="transfer_object", library=library, context=context, validation=validation, cost=cost)
        store, decided = store.promote(candidate, holdout_set, development_sets=DEVELOPMENT_SETS + (f"g12-development-{problem.problem_id}",))
        store.save(STORE)
        report["store"] = {"certificate_id": decided.certificate_id, "status": decided.status.value, "reason": decided.reason, "controller_facet": context.facet("controller").model_dump(mode="json") if context.facet("controller") else None,
                           "claimed_ranges": [r.model_dump(mode="json") for r in ranges], "store_version": store.version}
        print(json.dumps({"store": decided.status.value, "certificate": decided.certificate_id[:12]}), flush=True)
        # reuse: the transfer tree from the store in the mirrored layout, under the acquired parameters
        layouts_payload = json.loads((ROOT / "assets/general/research-protocols/g11-skill-store-v1/layouts.json").read_bytes())
        layout = problem_world(problem, EnvironmentV1.model_validate(layouts_payload["layouts"]["mirrored"]["environment"]), problems["changes"])
        session = TransferObjectSession.open(problem.body, source, layout, goal_for(layout), policy, seed_label=f"{problem.problem_id}-reuse", controller_config=controller, closure_config=closure, duration_scale=duration_scale)
        trace = store.retrieve("transfer_object", context_of(session, friction_assumption=friction_assumption, ranges=ranges))
        reuse = {"retrieval": trace.model_dump(mode="json"), "executed": trace.chosen is not None}
        if trace.chosen is not None:
            certificate = store.certificate(trace.chosen)
            run = run_tree(session, store.library(certificate.library_sha256), "transfer_object", {"object": "cube", "destination": "platform", "effector": session.effector.chain_id})
            episode = episode_row(run, episode_id=f"{problem.problem_id}-reuse", body=problem.body, draw=None, goal=goal_for(layout))
            bundle_dir = local / "reuse" / "physical"
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            sealed = seal_tree_run(bundle_dir, session=session, library=store.library(certificate.library_sha256), tree=run.tree, record=run.record, label=f"{problem.problem_id}-reuse",
                                   caption=f"{problem.body} | {problem.problem_id} | reuse from the store in the mirrored layout | certificate {certificate.certificate_id[:12]} | {run.record.verdict.value}",
                                   runtime_calls=run.runtime.calls, goal="G12", protocol=EVIDENCE_SCHEMA["episode_protocol"],
                                   task_extra={"layout": "mirrored", "certificate_id": certificate.certificate_id, "parameters": settled, "registration_sha256": registration["registration_sha256"]},
                                   outcome_extra={"skill_success": episode["skill_success"], "oracle_placement": episode["oracle"], "false_completion": episode["false_completion"], "attempts": episode["attempts"]},
                                   observation={"policy": "declared_sensors_with_oracle_labels_kept_apart", "inputs": ["joint encoders", "grasp point through the model", "contact force per opposition group", "cameras by ray visibility"], "vlm": False})
            media = render_with_retry(bundle_dir, out / "reuse" / "media", sealed["sha256"])
            reuse.update({"certificate_id": certificate.certificate_id, "episode": {k: v for k, v in episode.items() if k not in ("leaf_calls", "verdict_trail", "events")}, "bundle": sealed,
                          "media_sha256": media["sha256"], "frames": media["frame_count"], "media": (out / "reuse" / "media").relative_to(out).as_posix()})
            print(json.dumps({"reuse": run.record.verdict.value, "reason": run.record.root.reason, "oracle_placed": episode["oracle"]["placed"]}), flush=True)
        report["reuse"] = reuse
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "problem.json").write_bytes(json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
