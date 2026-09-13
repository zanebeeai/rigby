"""Run the registered G11 protocol: promote, reuse, invalidate and revalidate, interrupt and restart.

Four phases, each writing its own file under --out and its sealed bundles
under --local, the persisted store under the assets tree:

``promote``
    Every enabled body runs its independent validation set once; the
    outcomes for transfer_object and its two subskills become candidate
    certificates, promoted only where the set passes; the store is saved.

``reuse``
    The store is loaded from disk and, for every body, every layout and
    every skill, queried with the context the session actually stands in;
    a valid promoted certificate is reused (its validation episodes are not
    re-run, and that is logged as the cache hit) and the skill runs from
    the persisted library; a query no certificate covers is refused typed.

``invalidate``
    Five context changes, one per dimension, each applied to a copy of the
    promoted store from the same baseline: the dimension's facet changes,
    every certificate that differs on it is invalidated, and the change's
    revalidation set runs under the new context on the jaw arm; a passing
    outcome issues a superseding version, a failing one leaves the
    certificate invalidated. The geometry change is also applied to the
    persisted store, as D11's physical-context change.

``restart``
    On every body, the nominal episode is interrupted as each leaf
    completes; a fresh session opens on the checkpoint's physical state,
    reconstructs its belief from the sensors alone and runs the tree from
    its root; it completes or stops with a typed reason.

Every episode is one physics record; failures and the first successes are
sealed and rendered; every episode keeps its row. No API or model calls.

    python any-robot/scripts/g11_skill_store_campaign.py promote --out docs/results/g11-store --local any-robot/results/g11-store
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from dataclasses import asdict
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
from rigby_core.skills import (
    CertificateStatus,
    CertificateV1,
    ContextDimension,
    CostV1,
    SkillLibraryV1,
    SkillStoreV1,
    ValidationOutcomeV1,
    ValidationSetV1,
    Verdict,
)
from rigby_core.skills.examples import transfer_object_library

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.gates.control import ControllerConfig
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import load_policy
from rigby_general.skills import TransferObjectSession, seal_tree_run
from rigby_general.skills.skill_store import (
    EVIDENCE_SCHEMA,
    FRICTION_ASSUMPTION,
    DEVELOPMENT_SETS,
    context_of,
    goal_for,
    perturbed,
    restart,
    rows_digest,
    run_tree,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402
import g11_protocol as protocol  # noqa: E402
from g10_transfer_campaign import attempts_of, oracle_placed  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
STORE = ROOT / "assets/general/skill-store-v1"
EPISODE_PROTOCOL = EVIDENCE_SCHEMA["episode_protocol"]
RENDER_SUCCESSES = 2
ARGS = {"object": "cube", "destination": "platform"}


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


def provenance(registration: dict, phase: str) -> dict:
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)), "platform": platform.platform(), "python": sys.version,
            "mujoco": mujoco.__version__, "numpy": np.__version__, "registration_sha256": registration["registration_sha256"], "phase": phase,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0}


def source_of(body: str) -> Path:
    return ROOT / "assets/general/zoo" / body / "robot.urdf"


def open_session(body: str, environment: EnvironmentV1, policy: dict, *, configuration: str, controller: ControllerConfig | None, seed_label: str) -> TransferObjectSession:
    return TransferObjectSession.open(body, source_of(body), environment, goal_for(environment), policy, configuration_name=configuration, seed_label=seed_label, controller_config=controller)


def seal_and_render(run, *, local: Path, out: Path, name: str, label: str, caption: str, library: SkillLibraryV1, task_extra: dict, outcome_extra: dict, render: bool) -> dict:
    bundle_dir = local / name / "physical"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    sealed = seal_tree_run(bundle_dir, session=run.session, library=library, tree=run.tree, record=run.record, label=label, caption=caption, runtime_calls=run.runtime.calls, goal="G11",
                           protocol=EPISODE_PROTOCOL, task_extra=task_extra, outcome_extra=outcome_extra,
                           observation={"policy": "declared_sensors_with_oracle_labels_kept_apart", "inputs": ["joint encoders", "grasp point through the model", "contact force per opposition group", "cameras by ray visibility"], "vlm": False})
    row = {"bundle": sealed}
    if render:
        media_dir = out / name / "media"
        last = None
        for attempt in range(3):
            # ffmpeg has died mid-pipe under load before (G10); the render is
            # a pure function of the sealed bundle, so it is simply retried.
            if media_dir.exists():
                shutil.rmtree(media_dir)
            try:
                media = render_bundle(bundle_dir, media_dir, expected_digest=sealed["sha256"])
                break
            except (BrokenPipeError, OSError, RuntimeError, MemoryError) as error:
                last = error
                gc.collect()
                time.sleep(2.0)
        else:
            raise RuntimeError(f"rendering {name} failed three times: {last!r}")
        row["media_sha256"] = media["sha256"]
        row["frames"] = media["frame_count"]
        row["media"] = media_dir.relative_to(out).as_posix()
    return row


def episode_row(run, *, episode_id: str, body: str, draw: dict | None, goal) -> dict:
    session, record = run.session, run.record
    physical = session.recorder.finish() if session.recorder.rows["time_s"] else None
    oracle = oracle_placed(session.model, physical, goal, dwell_s=goal.dwell_s) if physical is not None else {"placed": False, "samples": 0}
    success = record.verdict is Verdict.SUCCESS
    subskills = {}
    for node in record.root.walk():
        if node.skill_id in ("acquire_until_held", "observe_object") and node.skill_id not in subskills:
            subskills[node.skill_id] = node.verdict.value
    return {"episode_id": episode_id, "zoo_id": body, "draw": draw, "verdict": record.verdict.value, "root_reason": record.root.reason, "interrupted": record.interrupted,
            "interrupt_reason": record.interrupt_reason, "physics_s": float(session.time_s), "wall_seconds": run.wall_s, "attempts": attempts_of(record), "leaf_calls": run.runtime.calls,
            "verdict_trail": run.runtime.verdicts, "events": session.events, "skill_success": success, "oracle": oracle,
            # A subskill places nothing; the oracle's placement judgement is a false-completion check for the transfer only.
            "false_completion": bool(success and not oracle["placed"] and run.tree.root.skill_id == "transfer_object"),
            "subskill_verdicts": subskills, "effector": session.effector.chain_id}


def outcome_of(set_: ValidationSetV1, rows: list[dict], *, success_of) -> ValidationOutcomeV1:
    successes = sum(1 for r in rows if success_of(r))
    return ValidationOutcomeV1(set_id=set_.set_id, episodes=len(rows), successes=successes, false_completions=sum(1 for r in rows if r["false_completion"] and success_of(r)),
                               threshold=set_.threshold, evidence=tuple(r["bundle"]["sha256"] for r in rows if "bundle" in r), rows_sha256=rows_digest(rows))


def cost_of(rows: list[dict]) -> CostV1:
    return CostV1(physics_s=float(sum(r["physics_s"] for r in rows)), wall_s=float(sum(r["wall_seconds"] for r in rows)), episodes=len(rows), generation_calls=0, dollars=0.0)


SUCCESS_OF = {
    "transfer_object": lambda r: r["skill_success"],
    "acquire_until_held": lambda r: r["subskill_verdicts"].get("acquire_until_held") == "success",
    "observe_object": lambda r: r["subskill_verdicts"].get("observe_object") == "success",
}


def run_set(body: str, set_payload: dict, environment: EnvironmentV1, policy: dict, library: SkillLibraryV1, *, configuration: str, controller: ControllerConfig | None, out: Path, local: Path,
            phase: str, task_extra: dict) -> tuple[ValidationSetV1, list[dict]]:
    """Every episode of a validation set, sealed where it failed and for the first successes."""

    validation = ValidationSetV1.model_validate(set_payload["set"])
    rows = []
    rendered = 0
    goal = goal_for(environment)
    for episode_id, draw in zip(validation.episodes, set_payload["draws"], strict=True):
        world = perturbed(environment, draw)
        session = open_session(body, world, policy, configuration=configuration, controller=controller, seed_label=episode_id)
        run = run_tree(session, library, "transfer_object", {**ARGS, "effector": session.effector.chain_id})
        row = episode_row(run, episode_id=episode_id, body=body, draw=draw, goal=goal)
        keep = not row["skill_success"] or rendered < RENDER_SUCCESSES
        if keep:
            caption = f"{body} | {phase} | {episode_id} | {run.record.verdict.value}" + (f" | {run.record.root.reason}" if run.record.root.reason else "")
            row.update(seal_and_render(run, local=local, out=out, name=f"{phase}/{episode_id}", label=episode_id, caption=caption, library=library,
                                       task_extra={"validation_set": validation.set_id, "draw": draw, "sensor_configuration": configuration, "controller": asdict(controller or ControllerConfig()), **task_extra},
                                       outcome_extra={"skill_success": row["skill_success"], "oracle_placement": row["oracle"], "false_completion": row["false_completion"], "attempts": row["attempts"], "subskill_verdicts": row["subskill_verdicts"]},
                                       render=True))
            if row["skill_success"]:
                rendered += 1
        rows.append(row)
        print(json.dumps({"episode": episode_id, "verdict": run.record.verdict.value, "reason": run.record.root.reason, "physics_s": round(row["physics_s"], 2), "oracle_placed": row["oracle"]["placed"], "wall": round(run.wall_s, 1), "kept": keep}), flush=True)
    return validation, rows


def certificates_from(rows: list[dict], validation: ValidationSetV1, library: SkillLibraryV1, context, *, version: int = 1, supersedes: dict[str, str] | None = None) -> dict[str, CertificateV1]:
    cost = cost_of(rows)
    return {skill: CertificateV1.candidate(skill_id=skill, library=library, context=context, validation=outcome_of(validation, rows, success_of=SUCCESS_OF[skill]), cost=cost, version=version,
                                            supersedes=(supersedes or {}).get(skill)) for skill in protocol.SKILLS}


# -- phases --------------------------------------------------------------------------------------


def phase_promote(args, registration: dict, sets: dict, policy: dict, library: SkillLibraryV1) -> None:
    environment = g10.environment()
    store = SkillStoreV1(store_id="rigby-skill-store-v1").add_library(library)
    report = {"schema": "g11.promotion.v1", "provenance": provenance(registration, "promote"), "bodies": {}}
    for body in protocol.BODIES:
        if args.bodies and body not in args.bodies.split(","):
            continue
        validation, rows = run_set(body, sets["validation"][body], environment, policy, library, configuration=protocol.CONFIGURATION, controller=None, out=args.out, local=args.local,
                                   phase=f"promotion/{body}", task_extra={"registration_sha256": registration["registration_sha256"]})
        session = open_session(body, environment, policy, configuration=protocol.CONFIGURATION, controller=None, seed_label="context")
        context = context_of(session)
        decisions = {}
        for skill, candidate in certificates_from(rows, validation, library, context).items():
            store, decided = store.promote(candidate, validation, development_sets=DEVELOPMENT_SETS)
            decisions[skill] = {"certificate_id": decided.certificate_id, "status": decided.status.value, "reason": decided.reason, "successes": decided.validation.successes,
                                "episodes": decided.validation.episodes, "threshold": decided.validation.threshold, "false_completions": decided.validation.false_completions,
                                "cost": decided.cost.model_dump(mode="json"), "evidence": list(decided.validation.evidence)}
            print(json.dumps({"body": body, "skill": skill, "status": decided.status.value, "successes": f"{decided.validation.successes}/{decided.validation.episodes}"}), flush=True)
        report["bodies"][body] = {"validation_set": validation.model_dump(mode="json"), "context": context.model_dump(mode="json"), "rows": rows, "decisions": decisions}
        (args.out / "promotion.json").write_bytes(json_bytes(report))
    if STORE.exists():
        shutil.rmtree(STORE)
    digests = store.save(STORE)
    report["store"] = {"path": STORE.relative_to(REPO).as_posix(), "version": store.version, "files": digests, "history": [e.model_dump(mode="json") for e in store.history]}
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "promotion.json").write_bytes(json_bytes(report))


def phase_reuse(args, registration: dict, layouts_payload: dict, policy: dict) -> None:
    store = SkillStoreV1.load(STORE)
    report = {"schema": "g11.reuse.v1", "provenance": provenance(registration, "reuse"), "store_version_loaded": store.version, "store_path": STORE.relative_to(REPO).as_posix(), "runs": [], "cache": []}
    ingested: dict[str, float] = {}
    for body in protocol.BODIES:
        if args.bodies and body not in args.bodies.split(","):
            continue
        for layout_name, layout in layouts_payload["layouts"].items():
            environment = EnvironmentV1.model_validate(layout["environment"])
            goal = goal_for(environment)
            for skill in protocol.SKILLS:
                label = f"{body}-{layout_name}-{skill}"
                started = time.perf_counter()
                session = open_session(body, environment, policy, configuration=protocol.CONFIGURATION, controller=None, seed_label=label)
                open_s = time.perf_counter() - started
                query = context_of(session)
                trace = store.retrieve(skill, query)
                store = store.note_retrieval(trace)
                cache = {"query": label, "certificate_hit": trace.chosen is not None, "certificate_id": trace.chosen, "library_from_disk": None, "validation_episodes_not_rerun": 0, "body_ingest_s": round(open_s, 2)}
                row = {"body": body, "layout": layout_name, "skill": skill, "retrieval": trace.model_dump(mode="json"), "context_values": query.values}
                if trace.chosen is None:
                    row.update({"executed": False, "refusal": trace.reason})
                    print(json.dumps({"reuse": label, "refused": trace.reason}), flush=True)
                else:
                    certificate = store.certificate(trace.chosen)
                    library = store.library(certificate.library_sha256)
                    cache["library_from_disk"] = (STORE / "libraries" / f"{certificate.library_sha256}.json").relative_to(REPO).as_posix()
                    cache["validation_episodes_not_rerun"] = certificate.validation.episodes
                    run = run_tree(session, library, skill, {**ARGS, "effector": session.effector.chain_id} if skill != "observe_object" else {"object": "cube"})
                    episode = episode_row(run, episode_id=label, body=body, draw=None, goal=goal)
                    caption = f"{body} | reuse {layout_name} | {skill} | {run.record.verdict.value}" + (f" | {run.record.root.reason}" if run.record.root.reason else "")
                    episode.update(seal_and_render(run, local=args.local, out=args.out, name=f"reuse/{label}", label=label, caption=caption, library=library,
                                                   task_extra={"layout": layout_name, "certificate_id": certificate.certificate_id, "certificate_version": certificate.version, "registration_sha256": registration["registration_sha256"], "sensor_configuration": protocol.CONFIGURATION},
                                                   outcome_extra={"skill_success": episode["skill_success"], "oracle_placement": episode["oracle"], "false_completion": episode["false_completion"], "attempts": episode["attempts"]},
                                                   render=True))
                    row.update({"executed": True, "episode": episode, "certificate_id": certificate.certificate_id, "certificate_version": certificate.version, "library_sha256": certificate.library_sha256})
                    print(json.dumps({"reuse": label, "certificate": certificate.certificate_id[:12], "verdict": run.record.verdict.value, "reason": run.record.root.reason, "physics_s": round(episode["physics_s"], 2), "oracle_placed": episode["oracle"]["placed"]}), flush=True)
                report["runs"].append(row)
                report["cache"].append(cache)
                (args.out / "reuse.json").write_bytes(json_bytes(report))
    store.save(STORE)
    report["store_version_after"] = store.version
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "reuse.json").write_bytes(json_bytes(report))


def changed_world(change: dict, environment: EnvironmentV1) -> EnvironmentV1:
    if change["dimension"] == "geometry":
        cube = environment.objects[0]
        half = change["object_half_size_m"]
        grown = cube.model_copy(update={"size_m": (half, half, half), "mass_kg": round(cube.mass_kg * change["mass_scale"], 9), "position_m": (cube.position_m[0], cube.position_m[1], cube.position_m[2] + (half - cube.size_m[2]))})
        return environment.model_copy(update={"objects": (grown,)})
    return environment


def phase_invalidate(args, registration: dict, sets: dict, changes: dict, policy: dict, library: SkillLibraryV1) -> None:
    base = g10.environment()
    body = protocol.CHANGE_BODY
    # Every change starts from the same baseline: the persisted store as the
    # reuse phase left it, snapshotted here so a change applied to the
    # persisted store (geometry, for D11) cannot leak into the next test.
    baseline = args.out / "invalidation" / "baseline-store"
    if not baseline.exists():
        shutil.copytree(STORE, baseline)
    # One change per process keeps the renderer's memory from accumulating
    # across thirty episodes; the report merges what earlier processes wrote.
    existing = json.loads((args.out / "invalidation.json").read_bytes()) if (args.out / "invalidation.json").exists() else {}
    report = {"schema": "g11.invalidation.v1", "provenance": provenance(registration, "invalidate"), "body": body, "baseline_store": baseline.relative_to(args.out).as_posix(),
              "baseline_store_version": SkillStoreV1.load(baseline).version, "changes": dict(existing.get("changes", {})), "provenance_by_change": dict(existing.get("provenance_by_change", {}))}
    for name, change in changes["changes"].items():
        if args.changes and name not in args.changes.split(","):
            continue
        store = SkillStoreV1.load(baseline)
        dimension = ContextDimension(change["dimension"])
        environment = changed_world(change, base)
        configuration = change.get("configuration", protocol.CONFIGURATION)
        controller = ControllerConfig(natural_frequency_hz=change["natural_frequency_hz"], damping_ratio=change["damping_ratio"]) if dimension is ContextDimension.CONTROLLER else None
        friction_assumption = sets["revalidation"][name]["friction_assumption"] if dimension is ContextDimension.FRICTION else None
        evidence_schema = {**EVIDENCE_SCHEMA, "episode_protocol": change["episode_protocol"]} if dimension is ContextDimension.EVIDENCE_SCHEMA else None
        probe = open_session(body, environment, policy, configuration=configuration, controller=controller, seed_label=f"context-{name}")
        new_context = context_of(probe, friction_assumption=friction_assumption, evidence_schema=evidence_schema)
        facet = new_context.facet(dimension)
        before = {c.certificate_id: c.status.value for c in store.certificates}
        store, affected = store.invalidate(dimension, facet, reason=change["declared"])
        retrieval_after = store.retrieve("transfer_object", new_context)
        entry = {"change": change, "facet": facet.model_dump(mode="json"), "affected": list(affected), "status_before": before, "status_after_invalidation": {c.certificate_id: c.status.value for c in store.certificates},
                 "retrieval_under_new_context_before_revalidation": retrieval_after.model_dump(mode="json")}
        print(json.dumps({"change": name, "affected": len(affected)}), flush=True)
        # revalidate the jaw arm's certificates under the new context
        set_payload = sets["revalidation"][name]
        validation, rows = run_set(body, set_payload, environment, policy, library, configuration=configuration, controller=controller, out=args.out, local=args.local, phase=f"invalidation/{name}",
                                   task_extra={"change": name, "registration_sha256": registration["registration_sha256"], "friction_assumption": friction_assumption or FRICTION_ASSUMPTION, "evidence_schema": evidence_schema or EVIDENCE_SCHEMA})
        jaw_certificates = {c.skill_id: c for c in store.certificates if c.status is CertificateStatus.INVALIDATED and c.certificate_id in affected and c.context.facet(ContextDimension.BODY).name == probe.robot.manifest.rig_id}
        decisions = {}
        for skill in protocol.SKILLS:
            old = jaw_certificates.get(skill)
            if old is None:
                decisions[skill] = {"note": "no invalidated certificate of this skill on this body"}
                continue
            outcome = outcome_of(validation, rows, success_of=SUCCESS_OF[skill])
            store, decided = store.revalidate(old.certificate_id, library=library, context=new_context, validation=outcome, validation_set=validation, cost=cost_of(rows), development_sets=DEVELOPMENT_SETS)
            decisions[skill] = {"invalidated": old.certificate_id, "new_certificate_id": decided.certificate_id, "version": decided.version, "status": decided.status.value, "reason": decided.reason,
                                "successes": outcome.successes, "episodes": outcome.episodes, "threshold": outcome.threshold, "false_completions": outcome.false_completions}
            print(json.dumps({"change": name, "skill": skill, "revalidation": decided.status.value, "successes": f"{outcome.successes}/{outcome.episodes}"}), flush=True)
        entry.update({"revalidation_set": validation.model_dump(mode="json"), "rows": rows, "decisions": decisions, "retrieval_under_new_context_after": store.retrieve("transfer_object", new_context).model_dump(mode="json"),
                      "retrieval_under_old_context_after": store.retrieve("transfer_object", context_of(open_session(body, base, policy, configuration=protocol.CONFIGURATION, controller=None, seed_label="old"))).model_dump(mode="json"),
                      "status_after": {c.certificate_id: c.status.value for c in store.certificates}, "history_tail": [e.model_dump(mode="json") for e in store.history[-8:]]})
        evidence_store = args.out / "invalidation" / name / "store"
        if evidence_store.exists():
            shutil.rmtree(evidence_store)
        store.save(evidence_store)
        entry["store_saved_to"] = evidence_store.relative_to(args.out).as_posix()
        report["changes"][name] = entry
        report["provenance_by_change"][name] = report["provenance"]
        (args.out / "invalidation.json").write_bytes(json_bytes(report))
    if "geometry" in (args.changes.split(",") if args.changes else changes["changes"]) and "geometry" in report["changes"]:
        # D11's physical-context change lives on in the persisted store; the
        # other four tests leave it as the baseline.
        geometry_store = SkillStoreV1.load(args.out / report["changes"]["geometry"]["store_saved_to"])
        if STORE.exists():
            shutil.rmtree(STORE)
        geometry_store.save(STORE)
        report["changes"]["geometry"]["applied_to_persisted_store"] = True
        report["persisted_store_version_after"] = geometry_store.version
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "invalidation.json").write_bytes(json_bytes(report))


def phase_restart(args, registration: dict, sets: dict, restart_payload: dict, policy: dict, library: SkillLibraryV1) -> None:
    environment = g10.environment()
    goal = goal_for(environment)
    report = {"schema": "g11.restart.v1", "provenance": provenance(registration, "restart"), "boundaries": restart_payload["boundaries"], "runs": []}
    for body in protocol.BODIES:
        if args.bodies and body not in args.bodies.split(","):
            continue
        draw = sets["validation"][body]["draws"][0]
        world = perturbed(environment, draw)
        for boundary in restart_payload["boundaries"]:
            k = boundary["index"]
            label = f"{body}-restart-after-{k:02d}-{boundary['skill_id']}"
            session = open_session(body, world, policy, configuration=protocol.CONFIGURATION, controller=None, seed_label=label)
            arguments = {**ARGS, "effector": session.effector.chain_id}
            first = run_tree(session, library, "transfer_object", arguments, checkpoint_after=k)
            checkpoint = first.checkpoint
            row = {"body": body, "boundary": boundary, "label": label, "draw": draw, "first": episode_row(first, episode_id=label + "-before", body=body, draw=draw, goal=goal),
                   "checkpoint": None if checkpoint is None else {**checkpoint.model_dump(mode="json"), "world_state": {"time_s": checkpoint.world_state.get("time_s"), "qpos_len": len(checkpoint.world_state.get("qpos") or [])}}}
            render_first = body == "zoo_jaw_arm" or first.record.verdict is not Verdict.INTERRUPTED
            row["first"].update(seal_and_render(first, local=args.local, out=args.out, name=f"restart/{label}/before", label=label + "-before", caption=f"{body} | interrupted after leaf {k} ({boundary['skill_id']}) | {first.record.verdict.value}",
                                                library=library, task_extra={"boundary": boundary, "registration_sha256": registration["registration_sha256"], "sensor_configuration": protocol.CONFIGURATION},
                                                outcome_extra={"checkpoint": row["checkpoint"]}, render=render_first))
            if checkpoint is None:
                row.update({"restarted": False, "reason": "the tree finished before the requested boundary"})
                report["runs"].append(row)
                continue
            second = restart(checkpoint, library, "transfer_object", arguments, body, source_of(body), world, goal, policy, configuration_name=protocol.CONFIGURATION, seed_label=label + "-after")
            after = episode_row(second, episode_id=label + "-after", body=body, draw=draw, goal=goal)
            completed = second.record.verdict is Verdict.SUCCESS
            explicit_stop = (not completed) and bool(second.record.root.reason or second.record.verdict is not Verdict.SUCCESS)
            caption = f"{body} | restarted after leaf {k} ({boundary['skill_id']}) from observation | {second.record.verdict.value}" + (f" | {second.record.root.reason}" if second.record.root.reason else "")
            after.update(seal_and_render(second, local=args.local, out=args.out, name=f"restart/{label}/after", label=label + "-after", caption=caption, library=library,
                                         task_extra={"boundary": boundary, "checkpoint_id": checkpoint.checkpoint_id, "reconstruction": second.reconstruction["trail"], "registration_sha256": registration["registration_sha256"], "sensor_configuration": protocol.CONFIGURATION},
                                         outcome_extra={"skill_success": after["skill_success"], "oracle_placement": after["oracle"], "false_completion": after["false_completion"], "completed": completed, "explicit_stop": explicit_stop,
                                                        "reconstruction": second.reconstruction["trail"]}, render=True))
            row.update({"restarted": True, "reconstruction": second.reconstruction["trail"], "after": after, "completed": completed, "explicit_stop": explicit_stop,
                        "leaves_after_restart": [c.get("leaf") for c in second.runtime.calls], "physics_continuous": abs(float(second.reconstruction["trail"]["observation_s"]) >= 0.0)})
            print(json.dumps({"restart": label, "before": first.record.verdict.value, "after": second.record.verdict.value, "reason": second.record.root.reason, "completed": completed,
                              "closure_engaged": second.reconstruction["trail"]["closure_engaged"], "facts": list(second.reconstruction["facts"]), "oracle_placed": after["oracle"]["placed"]}), flush=True)
            report["runs"].append(row)
            (args.out / "restart.json").write_bytes(json_bytes(report))
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "restart.json").write_bytes(json_bytes(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("promote", "reuse", "invalidate", "restart"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--bodies", default=None)
    parser.add_argument("--changes", default=None)
    args = parser.parse_args()
    registration = load_registration()
    sets = json.loads((protocol.PROTOCOL / "validation-sets.json").read_bytes())
    policy = load_policy(g10.G09 / "policy.json")
    library = transfer_object_library()
    args.out.mkdir(parents=True, exist_ok=True)
    args.local.mkdir(parents=True, exist_ok=True)
    if args.phase == "promote":
        phase_promote(args, registration, sets, policy, library)
    elif args.phase == "reuse":
        phase_reuse(args, registration, json.loads((protocol.PROTOCOL / "layouts.json").read_bytes()), policy)
    elif args.phase == "invalidate":
        phase_invalidate(args, registration, sets, json.loads((protocol.PROTOCOL / "changes.json").read_bytes()), policy, library)
    else:
        phase_restart(args, registration, sets, json.loads((protocol.PROTOCOL / "restart.json").read_bytes()), policy, library)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
