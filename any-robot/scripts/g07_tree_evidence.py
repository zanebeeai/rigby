"""G07 evidence: the contract schema, the nested example, its execution records, and four tree runs on physics.

Three kinds of artifact. The schemas of the library, the task tree and the
execution record, as JSON Schema, with the example library and its expanded
tree. The execution records of the example against a scripted runtime, one
per verdict the executor can reach -- success on the first alternative,
success by falling through to the second, failure by spending the loop's
budget, failure by losing progress, undecided when the verifying
observation sees nothing, and interruption mid-leaf -- so the record format
is shown on every path. And the same tree bound to real bodies in the G06
fixed world, each run one continuous physics record, sealed and rendered:
the bimanual dual arm whose first manipulator is refused by the guard and
whose second completes the transfer; the multifinger hand, whose every
attempt fails until the budget is spent; the jaw arm interrupted from
outside mid-lift; and the compact arm, refused on every alternative before
anything moves. No API or model calls.

    python any-robot/scripts/g07_tree_evidence.py --out docs/results/g07-trees
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rigby_core.skills import (
    Belief,
    ExecutionRecordV1,
    Interrupt,
    LeafOutcome,
    MonotonicClock,
    SkillLibraryV1,
    TaskTreeV1,
    Verdict,
    execute,
)
from rigby_core.skills.examples import clear_bench_library

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.skills import BodySession, TransferRuntime, predicates_for, seal_tree_run
from rigby_general.skills.transfer_runtime import SimulationClock


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g06_transfer_campaign as campaign  # noqa: E402


# --------------------------------------------------------------------------
# the scripted runtime: every verdict, recorded
# --------------------------------------------------------------------------


@dataclass
class ScriptedRuntime:
    primitives: dict[str, tuple[Verdict, bool | None]]
    placement_observable: bool = True
    inventory: dict[str, Any] = field(default_factory=lambda: {"known:cube": True, "reach:cube": True, "placed:cube:platform": False})
    stop_on_call: int | None = None
    interrupt: Interrupt | None = None
    calls: int = 0

    def run_primitive(self, context) -> LeafOutcome:
        self.calls += 1
        if self.stop_on_call is not None and self.calls >= self.stop_on_call and self.interrupt is not None:
            self.interrupt.request("stopped from outside")
        context.clock.advance(12.0)
        if context.should_stop():
            return LeafOutcome(Verdict.INTERRUPTED, "stopped mid-motion")
        verdict, placed = self.primitives.get(context.node.arguments["effector"], (Verdict.FAILURE, False))
        facts = {} if placed is None else {"placed:cube:platform": placed}
        return LeafOutcome(verdict, "" if verdict is Verdict.SUCCESS else "grasp_not_achieved", facts=facts, evidence={"effector": context.node.arguments["effector"]})

    def observe(self, context):
        context.clock.advance(0.5)
        if context.node.skill_id == "observe_inventory":
            return dict(self.inventory)
        if not self.placement_observable:
            return None
        return {"placed:cube:platform": context.belief.get("placed:cube:platform")}


SCRIPTED_PREDICATES = {
    "object_known": lambda belief, args: belief.get("known:" + args[0]),
    "object_placed": lambda belief, args: belief.get("placed:" + args[0] + ":" + args[1]),
    "object_in_reach": lambda belief, args: belief.get("reach:" + args[0]),
}
ARGUMENTS = {"object": "cube", "destination": "platform", "effector": "primary", "alternate": "secondary"}
S, F = Verdict.SUCCESS, Verdict.FAILURE


def scripted_scenarios(library: SkillLibraryV1, tree: TaskTreeV1) -> list[tuple[str, ExecutionRecordV1, str]]:
    scenarios = []

    def run(name: str, runtime: ScriptedRuntime, expected: Verdict, note: str, lib=library, tr=tree, interrupt=None):
        record = execute(tr, lib, runtime, SCRIPTED_PREDICATES, clock=MonotonicClock(), interrupt=interrupt)
        if record.verdict is not expected:
            raise SystemExit(f"scenario {name}: expected {expected.value}, got {record.verdict.value} ({record.root.reason})")
        scenarios.append((name, record, note))

    run("success-first-alternative", ScriptedRuntime({"primary": (S, True)}), S, "the first alternative places the object; the loop stops after one attempt")
    run("success-second-alternative", ScriptedRuntime({"primary": (F, False), "secondary": (S, True)}), S, "the first alternative fails typed; the selector falls through and the second places it")
    run("failure-budget-exhausted", ScriptedRuntime({}), F, "both alternatives fail on both attempts; the loop spends its budget of two")
    run("failure-no-progress", ScriptedRuntime({}, inventory={"known:cube": True, "reach:cube": False, "placed:cube:platform": False}), F, "the progress predicate fails after the first attempt; the loop stops before its budget")
    payload = json.loads(library.model_dump_json())
    for definition in payload["skills"]:
        if definition["skill_id"] == "transfer":
            definition["effects"] = []
    only_observed = SkillLibraryV1.model_validate(payload)
    run("unknown-nothing-observed", ScriptedRuntime({"primary": (S, None)}, placement_observable=False), Verdict.UNKNOWN,
        "the transfer claims no effect of its own and the verifying observation sees nothing; undecided propagates and the attempt is not repeated",
        lib=only_observed, tr=only_observed.expand("clear_bench", ARGUMENTS))
    interrupt = Interrupt()
    run("interrupted-mid-leaf", ScriptedRuntime({"primary": (S, True)}, stop_on_call=1, interrupt=interrupt), Verdict.INTERRUPTED,
        "an interruption arrives while the first transfer runs; the leaf, the selector, the sequence, the loop and the root all record it", interrupt=interrupt)
    return scenarios


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------


def physical_run(out: Path, *, name: str, zoo_id: str, library: SkillLibraryV1, env, goal, effector: str, alternate: str, stop_at_s: float | None, caption: str) -> dict:
    source = REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf"
    session = BodySession.open(zoo_id, source, env, goal)
    if effector not in session.effectors or alternate not in session.effectors:
        raise SystemExit(f"{zoo_id}: effectors {sorted(session.effectors)} do not include {effector!r} and {alternate!r}")
    arguments = {"object": "cube", "destination": "platform", "effector": effector, "alternate": alternate}
    tree = library.expand("clear_bench", arguments)
    interrupt = Interrupt()
    runtime = TransferRuntime(session, interrupt, stop_at_s=stop_at_s)
    record = execute(tree, library, runtime, predicates_for(session), clock=SimulationClock(session), interrupt=interrupt, belief=Belief())
    physical = out / "physical" / name / "physical"
    sealed = seal_tree_run(physical, session=session, library=library, tree=tree, record=record, label=f"{zoo_id}-{name}", caption=caption, runtime_calls=runtime.calls)
    media = render_bundle(physical, out / "physical" / name / "media", expected_digest=sealed["sha256"])
    (out / "physical" / name / "record.json").write_bytes(record.model_dump_json(indent=2).encode("utf-8"))
    summary = {"name": name, "zoo_id": zoo_id, "arguments": arguments, "verdict": record.verdict.value, "root_reason": record.root.reason,
               "interrupted": record.interrupted, "interrupt_reason": record.interrupt_reason, "tree_sha256": tree.content_hash(),
               "max_depth": tree.max_depth, "nodes": [{"node": r.node_id, "skill": r.skill_id, "kind": r.kind.value, "verdict": r.verdict.value, "reason": r.reason, "attempts": r.attempts}
                                                      for r in record.root.walk()],
               "leaves": runtime.calls, "bundle": sealed, "media_sha256": media["sha256"], "frames": media["frame_count"],
               "simulation_duration_s": media["simulation_duration_s"], "physics_end_s": record.ended_s}
    print(json.dumps({"run": name, "verdict": record.verdict.value, "reason": record.root.reason, "sim_s": round(media["simulation_duration_s"], 2)}), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a fresh destination")
    args.out.mkdir(parents=True)
    index: dict = {"goal": "G07", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)), "generation_calls": 0}

    # -- schemas, the example and its tree -------------------------------
    library = clear_bench_library()
    tree = library.expand("clear_bench", ARGUMENTS)
    (args.out / "schema").mkdir()
    for model in (SkillLibraryV1, TaskTreeV1, ExecutionRecordV1):
        (args.out / "schema" / f"{model.__name__}.schema.json").write_bytes(json_bytes(model.model_json_schema()))
    (args.out / "example").mkdir()
    (args.out / "example" / "clear_bench_v1.library.json").write_bytes(library.model_dump_json(indent=2).encode("utf-8"))
    (args.out / "example" / "clear_bench.tree.json").write_bytes(tree.model_dump_json(indent=2).encode("utf-8"))
    index["library"] = {"library_id": library.library_id, "sha256": library.content_hash(), "definition_depth": library.definition_depth("clear_bench"),
                        "skills": [{"skill_id": s.skill_id, "kind": s.kind.value, "children": [c.skill for c in s.children]} for s in library.skills]}
    index["tree"] = {"sha256": tree.content_hash(), "max_depth": tree.max_depth, "nodes": sum(1 for _ in tree.root.walk()),
                     "kinds": sorted({n.kind.value for n in tree.root.walk()})}
    reloaded = SkillLibraryV1.model_validate_json((args.out / "example" / "clear_bench_v1.library.json").read_bytes())
    index["round_trip"] = {"library_equal": reloaded == library, "library_sha256_equal": reloaded.content_hash() == library.content_hash(),
                           "tree_sha256_equal": reloaded.expand("clear_bench", ARGUMENTS).content_hash() == tree.content_hash()}

    # -- scripted scenarios ------------------------------------------------
    (args.out / "scripted").mkdir()
    index["scripted"] = []
    for name, record, note in scripted_scenarios(library, tree):
        (args.out / "scripted" / f"{name}.record.json").write_bytes(record.model_dump_json(indent=2).encode("utf-8"))
        index["scripted"].append({"name": name, "verdict": record.verdict.value, "root_reason": record.root.reason, "note": note,
                                  "nodes": [{"node": r.node_id, "skill": r.skill_id, "verdict": r.verdict.value, "reason": r.reason} for r in record.root.walk()],
                                  "record_sha256": record.content_hash()})
        print(json.dumps({"scripted": name, "verdict": record.verdict.value, "reason": record.root.reason}), flush=True)

    # -- physics ----------------------------------------------------------
    env, goal, feasibility, roster, registration = campaign.load_registration()
    index["environment"] = {"environment_id": env.environment_id, "registration_sha256": registration["registration_sha256"]}
    runs = []
    dual = BodySession.open("zoo_dual_arm", REPO / "any-robot/assets/general/zoo/zoo_dual_arm/robot.urdf", env, goal)
    chains = sorted(dual.effectors)
    # The manipulator the guard refuses on this fixture is the first
    # alternative, so the selector has something to fall through from.
    refused = next((c for c in chains if c != chains[0]), chains[0])
    runs.append(physical_run(args.out, name="dual-selector", zoo_id="zoo_dual_arm", library=library, env=env, goal=goal, effector=refused, alternate=chains[0], stop_at_s=None,
                             caption=f"zoo_dual_arm | clear_bench tree | first alternative {refused} refused by the guard, second {chains[0]} transfers"))
    for zoo_id, name, stop, caption in (
        ("zoo_hand_arm", "hand-budget", None, "zoo_hand_arm | clear_bench tree | every attempt fails, the loop spends its budget"),
        ("zoo_jaw_arm", "jaw-interrupt", 6.0, "zoo_jaw_arm | clear_bench tree | interrupted from outside at 6 s of physics"),
        ("zoo_compact_arm", "compact-refused", None, "zoo_compact_arm | clear_bench tree | every alternative refused before motion"),
    ):
        session = BodySession.open(zoo_id, REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf", env, goal)
        only = sorted(session.effectors)[0]
        runs.append(physical_run(args.out, name=name, zoo_id=zoo_id, library=library, env=env, goal=goal, effector=only, alternate=only, stop_at_s=stop, caption=caption))
    index["physical"] = runs
    expected = {"dual-selector": "success", "hand-budget": "failure", "jaw-interrupt": "interrupted", "compact-refused": "failure"}
    for run in runs:
        if run["verdict"] != expected[run["name"]]:
            raise SystemExit(f"{run['name']}: expected {expected[run['name']]}, got {run['verdict']} ({run['root_reason']})")
    files = {p.relative_to(args.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "index.json"}
    index["files"] = files
    (args.out / "index.json").write_bytes(json_bytes(index))
    print(json.dumps({"physical": [(r["name"], r["verdict"], r["root_reason"]) for r in runs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
