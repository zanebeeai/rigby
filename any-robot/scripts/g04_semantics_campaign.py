"""G04: score the planners against the registered semantic fixture.

Three modes, in the order the authorization requires them:

``dry-run``
    Every prompt of the fixture through the model planner over a mock
    transport that answers with the offline recognizer's reading and counts
    tokens from characters. No provider is contacted. Writes the projected
    cost of the whole fixture for every candidate model.

``select``
    The 60 canonical cases through each candidate model in order of price,
    stopping at the first whose readings are exact on all of them. Every
    reply is cached by hash and every call logged with its tokens and cost.

``score``
    All 180 cases through the offline recognizer and the chosen model; then
    every canonical case on every zoo body, where a reading must agree across
    the bodies that afford it and refuse, typed and naming the entry, on the
    bodies that do not. Errors and unsupported coverage are published in
    their own files. Optionally the explicit-quantity cases are grounded on
    one body so the unsupported list carries grounding's refusals too.

The fixture is registered by hash before the first scored run and every mode
refuses a fixture that no longer hashes to its registration. Keys are read
from a ``.env`` file and never written anywhere.

    python any-robot/scripts/g04_semantics_campaign.py dry-run --out docs/results/g04-semantics
    python any-robot/scripts/g04_semantics_campaign.py select --out docs/results/g04-semantics
    python any-robot/scripts/g04_semantics_campaign.py score --model gpt-5-nano --out docs/results/g04-semantics --ground-quantities
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rigby_general.errors import GeneralFailureCode, RigbyGeneralError
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.planner.model_planner import (
    PUBLISHED_RATES_USD_PER_MTOK,
    RATES_RECORDED_ON,
    REASONING_ALLOWANCE_TOKENS,
    Budget,
    BudgetStop,
    CallLog,
    base_model,
    GeminiTransport,
    MockTransport,
    ModelSchemaPlanner,
    ModelUnavailable,
    OpenAITransport,
    ResponseCache,
    cost_usd,
    reply_schema,
    system_prompt,
)
from rigby_general.planner.schema_planner import _boundary_for, _frame_for
from rigby_general.schema.inventory import INVENTORY_PATH, afforded_entries, load_inventory
from rigby_general.schema.program import (
    Concurrency,
    Dimensionality,
    Flexion,
    MannerV1,
    MemberGroup,
    MotionSchemaProgramV1,
    PostureV1,
    RegionV1,
    Remove,
    RoleBindingV1,
    SegmentLinkV1,
    SegmentV1,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIXTURE = ROOT / "assets/general/research-protocols/g04-semantics-v1"
CACHE = FIXTURE / "cache"
CANDIDATES = ("gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5-mini@low", "gpt-4.1", "gpt-5@low")
"""Cheapest first, by the recorded rates; the first to read every canonical case exactly is kept."""
ZOO = ("zoo_compact_arm", "zoo_dual_arm", "zoo_hand_arm", "zoo_jaw_arm", "zoo_long_arm", "zoo_tool_arm")
MANNER_AXES = ("speed", "effort", "smoothness", "rhythm", "amplitude", "repetition", "precision")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8"))


def load_env(path: Path) -> None:
    """Put a .env file's keys in the environment; values never leave the process."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# -- registration ------------------------------------------------------------------


def registration_payload(model: str | None) -> dict:
    files = {name: sha256_of(FIXTURE / name) for name in ("semantic-cases.json", "labels.json")}
    files["schema_inventory.v1.json"] = sha256_of(INVENTORY_PATH)
    inventory = load_inventory()
    return {
        "schema": "g04.registration.v1",
        "fixture_id": "g04-semantics-v1",
        "files": files,
        "system_prompt_sha256": hashlib.sha256(system_prompt(inventory).encode("utf-8")).hexdigest(),
        "reply_schema_sha256": hashlib.sha256(json.dumps(reply_schema(inventory), sort_keys=True).encode("utf-8")).hexdigest(),
        "candidate_models": list(CANDIDATES),
        "rates_recorded_on": RATES_RECORDED_ON,
        "rates_usd_per_mtok": {name: list(PUBLISHED_RATES_USD_PER_MTOK[base_model(name)]) for name in CANDIDATES},
        "selected_model": model,
        "label_provenance": "internal",
        "note": "registered before any scored run; the campaign refuses a fixture, inventory, system prompt or reply schema that does not hash to this",
    }


def register(model: str | None = None) -> dict:
    path = FIXTURE / "registration.json"
    payload = registration_payload(model)
    if path.exists():
        existing = json.loads(path.read_bytes())
        for key in ("files", "system_prompt_sha256", "reply_schema_sha256"):
            if existing[key] != payload[key]:
                raise SystemExit(f"the fixture no longer hashes to its registration ({key}); re-register deliberately and retain the old run")
        if model is not None and existing.get("selected_model") not in (None, model):
            raise SystemExit(f"the registration names {existing['selected_model']!r}; a different model needs a new registration")
        if model is not None and existing.get("selected_model") is None:
            existing["selected_model"] = model
            existing["model_selected_at_utc"] = datetime.now(timezone.utc).isoformat()
            existing["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in existing.items() if k != "registration_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()
            json_dump(path, existing)
        return existing
    payload["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["registration_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    json_dump(path, payload)
    return payload


# -- expected readings ------------------------------------------------------------------


def expected_program(case: dict, inventory) -> tuple[MotionSchemaProgramV1, list[dict]] | None:
    expected = case["expected"]
    if "segments" not in expected:
        return None
    segments, quantities = [], []
    for index, item in enumerate(expected["segments"]):
        entry = inventory.by_id(item["entry_id"])
        posture = None
        if item.get("posture"):
            p = item["posture"]
            posture = PostureV1(group=MemberGroup(p["group"]), selected_count=p["selected_count"], selected=Flexion(p["selected"]), remainder=Flexion(p["remainder"]), opposing=Flexion(p["opposing"]))
        manner = MannerV1(**{axis: int(item["manner"].get(axis, 0)) for axis in MANNER_AXES}, repetition_count=item.get("repetition_count"))
        segments.append(SegmentV1(
            segment_id=f"s{index}", motion_schema=entry.schema,
            figure=RoleBindingV1(role=entry.figure_role), ground=RoleBindingV1(role=entry.ground_role),
            region=RegionV1(remove=Remove(item["remove"]), dimensionality=Dimensionality.POINT),
            frame=_frame_for(entry), manner=manner, posture=posture, boundary=_boundary_for(entry),
        ))
        for quantity in item.get("quantities", []):
            quantities.append({"segment_id": f"s{index}", "text": quantity["text"], "value_m": float(quantity["value_m"])})
    links = tuple(SegmentLinkV1(from_segment=a.segment_id, to_segment=b.segment_id, relation=Concurrency.SEQUENCE) for a, b in zip(segments, segments[1:]))
    program = MotionSchemaProgramV1(program_id=f"expected-{case['case_id']}", source_text=case["prompt"], segments=tuple(segments), links=links)
    return program, quantities


def readable(program: MotionSchemaProgramV1, quantities) -> list[dict]:
    by_segment: dict[str, list] = {}
    for quantity in quantities:
        segment_id = quantity["segment_id"] if isinstance(quantity, dict) else quantity.segment_id
        text = quantity["text"] if isinstance(quantity, dict) else quantity.text
        value = quantity["value_m"] if isinstance(quantity, dict) else quantity.value
        by_segment.setdefault(segment_id, []).append({"text": text, "value_m": round(float(value), 6)})
    rows = []
    for segment in program.segments:
        entry_id = _entry_id_of(segment)
        rows.append({
            "segment_id": segment.segment_id, "entry_id": entry_id, "remove": segment.region.remove.value,
            "manner": {axis: getattr(segment.manner, axis) for axis in MANNER_AXES if getattr(segment.manner, axis)},
            "repetition_count": segment.manner.repetition_count,
            "posture": segment.posture.canonical_key if segment.posture else None,
            "quantities": by_segment.get(segment.segment_id, []),
        })
    return rows


_ENTRY_BY_BINDING: dict[str, str] = {}


def _entry_id_of(segment: SegmentV1) -> str:
    key = f"{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}"
    return _ENTRY_BY_BINDING.get(key, key)


# -- one reading ------------------------------------------------------------------------


def read(planner, prompt: str, afforded) -> dict:
    """The planner's reading of one prompt, or its typed refusal, or a stop."""

    started = time.perf_counter()
    try:
        planned = planner.plan_request(prompt, afforded=afforded)
    except RigbyGeneralError as error:
        return {"refusal": error.code.value, "detail": str(error)[:300], "details": {k: v for k, v in error.details.items() if k != "prompt"}, "seconds": time.perf_counter() - started}
    except (ModelUnavailable, BudgetStop) as error:
        return {"stopped": type(error).__name__, "detail": str(error)[:300], "seconds": time.perf_counter() - started}
    return {
        "hash": planned.program.role_normalized_hash(), "program": readable(planned.program, planned.quantities),
        "quantities": [{"segment_id": q.segment_id, "text": q.text, "value_m": q.value} for q in planned.quantities],
        "model": planned.model, "cached": planned.cached, "planner_id": planned.planner_id, "seconds": time.perf_counter() - started,
    }


def score(case: dict, expected, actual: dict) -> dict:
    """Exactness, the prohibited check, and a name for what went wrong."""

    verdict = {"exact": False, "prohibited_rejected": True, "error": None}
    if "stopped" in actual:
        verdict["error"] = "stopped"
        return verdict
    if expected is None:
        allowed = case["expected"]["refusal"]
        if "refusal" in actual and actual["refusal"] in allowed:
            verdict["exact"] = True
        elif "refusal" in actual:
            verdict["error"] = f"refusal_code:{actual['refusal']}"
        else:
            verdict["error"] = "reading_instead_of_refusal"
        return verdict
    program, quantities = expected
    if "refusal" in actual:
        verdict["error"] = f"refusal_instead_of_reading:{actual['refusal']}"
        return verdict  # a refusal contains nothing prohibited
    same_hash = actual["hash"] == program.role_normalized_hash()
    expected_q = sorted((q["segment_id"], q["text"].lower(), round(q["value_m"], 6)) for q in quantities)
    actual_q = sorted((q["segment_id"], q["text"].lower(), round(q["value_m"], 6)) for q in actual["quantities"])
    same_quantities = expected_q == actual_q
    verdict["exact"] = same_hash and same_quantities
    if not verdict["exact"]:
        verdict["error"] = classify(readable(program, quantities), actual["program"], same_quantities)
    for item in case["prohibited"]:
        if item["kind"] == "quantity" and actual["quantities"]:
            verdict["prohibited_rejected"] = False
        elif item["kind"] == "entry" and any(row["entry_id"] == item["entry_id"] for row in actual["program"]):
            verdict["prohibited_rejected"] = False
        elif item["kind"] == "remove" and any(row["remove"] == item["remove"] for row in actual["program"]):
            verdict["prohibited_rejected"] = False
        elif item["kind"] == "speed_above" and any(row["manner"].get("speed", 0) > item["value"] for row in actual["program"]):
            verdict["prohibited_rejected"] = False
        elif item["kind"] == "posture":
            forbidden = PostureV1(group=MemberGroup(item["posture"]["group"]), selected_count=item["posture"]["selected_count"], selected=Flexion(item["posture"]["selected"]), remainder=Flexion(item["posture"]["remainder"]), opposing=Flexion(item["posture"]["opposing"])).canonical_key
            if any(row["posture"] == forbidden for row in actual["program"]):
                verdict["prohibited_rejected"] = False
    return verdict


def classify(expected_rows: list[dict], actual_rows: list[dict], same_quantities: bool) -> str:
    if len(expected_rows) != len(actual_rows):
        return f"segment_count:{len(expected_rows)}->{len(actual_rows)}"
    for expected_row, actual_row in zip(expected_rows, actual_rows):
        if expected_row["entry_id"] != actual_row["entry_id"]:
            return f"entry:{expected_row['entry_id']}->{actual_row['entry_id']}"
        if expected_row["remove"] != actual_row["remove"]:
            return f"remove:{expected_row['remove']}->{actual_row['remove']}"
        if expected_row["manner"] != actual_row["manner"]:
            return f"manner:{expected_row['manner']}->{actual_row['manner']}"
        if expected_row["repetition_count"] != actual_row["repetition_count"]:
            return f"repetition_count:{expected_row['repetition_count']}->{actual_row['repetition_count']}"
        if expected_row["posture"] != actual_row["posture"]:
            return f"posture:{expected_row['posture']}->{actual_row['posture']}"
    if not same_quantities:
        return "quantity"
    return "other"


# -- the runs ------------------------------------------------------------------------------


def planner_for(mode: str, model: str, inventory, *, log: CallLog, budget: Budget, purpose: str, allow_fallback: bool):
    if mode == "mock":
        return ModelSchemaPlanner(inventory, model=model, transport=MockTransport(inventory), cache=None, log=log, budget=budget, purpose=purpose)
    transport = OpenAITransport()
    fallback = None
    if allow_fallback and os.environ.get("GEMINI_API_KEY", "").strip():
        fallback = ("gemini-2.0-flash", GeminiTransport())
    return ModelSchemaPlanner(inventory, model=model, transport=transport, cache=ResponseCache(CACHE), log=log, budget=budget, purpose=purpose, fallback=fallback)


def run_cases(planner, cases: list[dict], inventory, *, label: str) -> tuple[list[dict], bool]:
    rows, stopped = [], False
    for case in cases:
        expected = expected_program(case, inventory)
        actual = read(planner, case["prompt"], inventory.entries) if not stopped else {"stopped": "not_attempted", "detail": "the planner stopped earlier in this run", "seconds": 0.0}
        if "stopped" in actual and actual["stopped"] != "not_attempted":
            stopped = True
        verdict = score(case, expected, actual)
        rows.append({"case_id": case["case_id"], "kind": case["kind"], "family": case["family"], "prompt": case["prompt"],
                     "expected": (readable(*expected) if expected else {"refusal": case["expected"]["refusal"]}),
                     "actual": actual, **verdict, "planner": label})
    return rows, stopped


def summarize(rows: list[dict]) -> dict:
    def block(items):
        exact = sum(1 for r in items if r["exact"])
        prohibited = [r for r in items if any(True for _ in r.get("prohibited_items", [])) or r.get("has_prohibited")]
        return {"cases": len(items), "exact": exact, "exact_fraction": round(exact / len(items), 4) if items else None,
                "prohibited_cases": len(prohibited), "prohibited_rejected": sum(1 for r in prohibited if r["prohibited_rejected"]),
                "errors": sorted({r["error"] for r in items if r["error"]}), "stopped": sum(1 for r in items if r["error"] == "stopped")}
    by_kind = {kind: block([r for r in rows if r["kind"] == kind]) for kind in ("canonical", "language")}
    by_family = {}
    for family in sorted({r["family"] for r in rows}):
        by_family[family] = block([r for r in rows if r["family"] == family])
    return {"by_kind": by_kind, "by_family": by_family}


def with_prohibited_flag(rows: list[dict], cases: dict[str, dict]) -> list[dict]:
    for row in rows:
        row["has_prohibited"] = bool(cases[row["case_id"]]["prohibited"])
    return rows


def body_agreement(planner, cases: list[dict], inventory, robots: dict, *, label: str) -> list[dict]:
    """A canonical reading agrees on every body that affords it and refuses, typed, elsewhere."""

    rows = []
    for case in cases:
        expected = expected_program(case, inventory)
        neutral = read(planner, case["prompt"], inventory.entries)
        per_body = {}
        for body, robot in robots.items():
            afforded = afforded_entries(inventory, robot.morphology)
            afforded_ids = {entry.entry_id for entry in afforded}
            needed = [item["entry_id"] for item in case["expected"].get("segments", [])]
            applicable = bool(needed) and all(entry_id in afforded_ids for entry_id in needed)
            actual = read(planner, case["prompt"], afforded)
            if applicable:
                agrees = "hash" in actual and "hash" in neutral and actual["hash"] == neutral["hash"] and (expected is None or actual["hash"] == expected[0].role_normalized_hash())
            else:
                missing = [entry_id for entry_id in needed if entry_id not in afforded_ids]
                agrees = actual.get("refusal") == GeneralFailureCode.UNAFFORDED_SCHEMA.value and (not missing or actual.get("details", {}).get("entry_id") in missing) if needed else ("refusal" in actual)
            per_body[body] = {"applicable": applicable, "agrees": bool(agrees), "actual": actual.get("hash") or actual.get("refusal") or actual.get("stopped"), "missing": [e for e in needed if e not in afforded_ids]}
        rows.append({"case_id": case["case_id"], "prompt": case["prompt"], "neutral": neutral.get("hash") or neutral.get("refusal"), "bodies": per_body,
                     "applicable_bodies": sum(1 for b in per_body.values() if b["applicable"]), "all_agree": all(b["agrees"] for b in per_body.values()), "planner": label})
    return rows


def ground_quantities(cases: list[dict], planner, *, body: str) -> list[dict]:
    """Every explicit-quantity case through bind and ground on one body."""

    from rigby_general.evidence.composition import run_prompt

    robot = ingest_robot(ROOT / "assets/general/zoo" / body / "robot.urdf", robot_id=body)
    rows = []
    for case in cases:
        runs, _, _, _ = run_prompt(robot, case["prompt"], repeats=1, planner=planner)
        run = runs[0]
        rows.append({
            "case_id": case["case_id"], "prompt": case["prompt"], "body": body, "accepted": run.accepted,
            "requested_quantities": run.trace.requested_quantities, "failure_stage": run.failure_stage, "failure_code": run.failure_code,
            "failure_detail": run.failure_detail, "measurement": next((s.get("measurement") for s in [] ), None),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("dry-run", "select", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--spent-before", type=float, default=0.0, help="dollars already spent under the authorization, from the ledger")
    parser.add_argument("--soft-cap", type=float, default=18.0)
    parser.add_argument("--hard-ceiling", type=float, default=20.0)
    parser.add_argument("--allow-fallback", action="store_true", help="permit the free-plan Gemini fallback when OpenAI is unavailable (never for a scored run)")
    parser.add_argument("--ground-quantities", action="store_true")
    parser.add_argument("--ground-body", default="zoo_jaw_arm")
    args = parser.parse_args()

    env_path = args.env or (REPO / ".env" if (REPO / ".env").exists() else REPO.parent / "rigby" / ".env")
    load_env(env_path)
    inventory = load_inventory()
    for entry in inventory.entries:
        _ENTRY_BY_BINDING[entry.binding_key] = entry.entry_id
    fixture = json.loads((FIXTURE / "semantic-cases.json").read_bytes())
    cases = fixture["cases"]
    by_id = {case["case_id"]: case for case in cases}
    canonical = [case for case in cases if case["kind"] == "canonical"]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()

    if args.mode == "dry-run":
        registration = register(None)
        log = CallLog(out / "dry-run-calls.jsonl")
        if log.path.exists():
            log.path.unlink()
        budget = Budget(hard_ceiling_usd=args.hard_ceiling, soft_cap_usd=args.soft_cap, spent_before_usd=args.spent_before)
        planner = planner_for("mock", "gpt-5-nano", inventory, log=log, budget=budget, purpose="dry_run", allow_fallback=False)
        rows, _ = run_cases(planner, cases, inventory, label="mock(offline reading)")
        with_prohibited_flag(rows, by_id)
        prompt_tokens = sum(r["prompt_tokens"] for r in log.rows)
        completion_tokens = sum(r["completion_tokens"] for r in log.rows)
        per_model = {}
        for model in args.candidates.split(","):
            allowance = REASONING_ALLOWANCE_TOKENS.get(base_model(model), 0) * (4 if "@" in model else 1)
            completion_with_reasoning = completion_tokens + len(cases) * allowance
            once = cost_usd(model, prompt_tokens, completion_with_reasoning)
            per_model[model] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_with_reasoning, "usd_once": round(once, 4),
                                "usd_twice_with_margin": round(2.0 * once * 1.5, 4), "rates_usd_per_mtok": list(PUBLISHED_RATES_USD_PER_MTOK[base_model(model)])}
        all_candidates_once = sum(v["usd_once"] for v in per_model.values())
        # The protocol as run: every candidate reads the 60 canonical cases (worst
        # case: none passes early), then the dearest candidate reads the whole
        # fixture once for scoring; the body-agreement pass is served from cache.
        canonical_share = len(canonical) / len(cases)
        selection_worst = sum(v["usd_once"] * canonical_share for v in per_model.values())
        scoring_worst = max(v["usd_once"] for v in per_model.values())
        protocol_worst = selection_worst + scoring_worst
        projection = {
            "schema": "g04.projection.v1", "started_at_utc": started, "commit": commit, "fixture_registration": registration["registration_sha256"],
            "calls": len(log.rows), "cases": len(cases), "tokens_estimated_from_characters": True, "chars_per_token": 3.5,
            "prompt_tokens_total": prompt_tokens, "completion_tokens_total": completion_tokens,
            "per_model": per_model, "all_candidates_once_usd": round(all_candidates_once, 4),
            "protocol": {"selection_canonical_only_every_candidate_usd": round(selection_worst, 4), "scoring_full_fixture_dearest_candidate_usd": round(scoring_worst, 4),
                         "worst_case_usd": round(protocol_worst, 4), "worst_case_with_margin_usd": round(protocol_worst * 1.5, 4)},
            "soft_cap_usd": args.soft_cap, "fits_under_soft_cap": protocol_worst * 1.5 + args.spent_before < args.soft_cap,
            "mock_reading_summary": summarize(rows),
        }
        json_dump(out / "projection.json", projection)
        json_dump(out / "dry-run-cases.json", rows)
        print(json.dumps({k: projection[k] for k in ("calls", "prompt_tokens_total", "completion_tokens_total", "all_candidates_once_usd", "protocol", "fits_under_soft_cap")}, indent=1))
        print(json.dumps(projection["mock_reading_summary"]["by_kind"], indent=1))
        return 0

    log = CallLog(out / "calls.jsonl")
    budget = Budget(hard_ceiling_usd=args.hard_ceiling, soft_cap_usd=args.soft_cap, spent_before_usd=args.spent_before)

    if args.mode == "select":
        registration = register(None)
        results = {"schema": "g04.selection.v1", "started_at_utc": started, "commit": commit, "candidates": [], "selected": None, "stopped": None}
        for model in args.candidates.split(","):
            try:
                planner = planner_for("openai", model, inventory, log=log, budget=budget, purpose=f"select:{model}", allow_fallback=False)
            except ModelUnavailable as error:
                results["stopped"] = {"reason": error.reason, "detail": str(error)[:300]}
                break
            rows, stopped = run_cases(planner, canonical, inventory, label=f"model:{model}")
            with_prohibited_flag(rows, by_id)
            exact = sum(1 for r in rows if r["exact"])
            results["candidates"].append({
                "model": model, "canonical_exact": exact, "of": len(rows), "stopped": stopped,
                "spend": planner.spend.__dict__, "failures": [{"case_id": r["case_id"], "prompt": r["prompt"], "error": r["error"]} for r in rows if not r["exact"]],
            })
            json_dump(out / f"selection-{model}.json", rows)
            print(f"{model}: {exact}/{len(rows)} canonical exact; ${planner.spend.dollars:.4f}; stopped={planner.spend.stopped}")
            if stopped:
                results["stopped"] = {"reason": planner.spend.stopped, "model": model}
                break
            if exact == len(rows):
                results["selected"] = model
                break
        results["totals"] = log.totals()
        json_dump(out / "selection.json", results)
        if results["selected"]:
            register(results["selected"])
        print(json.dumps({"selected": results["selected"], "stopped": results["stopped"], "totals": results["totals"]}, indent=1))
        return 0

    # -- score ---------------------------------------------------------------------
    if not args.model:
        raise SystemExit("score needs --model")
    registration = register(args.model)
    offline = OfflineSchemaPlanner(inventory)
    offline_rows, _ = run_cases(offline, cases, inventory, label="offline-recognizer-v1")
    with_prohibited_flag(offline_rows, by_id)
    try:
        planner = planner_for("openai", args.model, inventory, log=log, budget=budget, purpose=f"score:{args.model}", allow_fallback=args.allow_fallback)
    except ModelUnavailable as error:
        planner = None
        model_rows, stopped = [], True
        stop_detail = {"reason": error.reason, "detail": str(error)[:300]}
    else:
        model_rows, stopped = run_cases(planner, cases, inventory, label=f"model:{args.model}")
        with_prohibited_flag(model_rows, by_id)
        stop_detail = {"reason": planner.spend.stopped} if stopped else None

    robots = {body: ingest_robot(ROOT / "assets/general/zoo" / body / "robot.urdf", robot_id=body) for body in ZOO}
    agreement = {"offline": body_agreement(offline, canonical, inventory, robots, label="offline-recognizer-v1")}
    if planner is not None and not stopped:
        agreement["model"] = body_agreement(planner, canonical, inventory, robots, label=f"model:{args.model}")

    grounding = None
    if args.ground_quantities:
        quantity_cases = [case for case in cases if case["family"] == "explicit_quantity"]
        grounding = {"body": args.ground_body, "offline": ground_quantities(quantity_cases, offline, body=args.ground_body)}
        if planner is not None and not stopped:
            grounding["model"] = ground_quantities(quantity_cases, planner, body=args.ground_body)

    # errors and unsupported coverage, separately
    def errors_of(rows):
        return [{"case_id": r["case_id"], "family": r["family"], "prompt": r["prompt"], "error": r["error"], "expected": r["expected"], "actual": r["actual"].get("program") or r["actual"].get("refusal") or r["actual"].get("stopped")}
                for r in rows if not r["exact"] and not (r["error"] or "").startswith("refusal_instead_of_reading") and r["error"] != "stopped"]

    def unsupported_of(rows):
        return [{"case_id": r["case_id"], "family": r["family"], "prompt": r["prompt"], "refusal": r["actual"].get("refusal"), "detail": r["actual"].get("detail")}
                for r in rows if (r["error"] or "").startswith("refusal_instead_of_reading")]

    def confusion_of(rows):
        matrix: dict[str, dict[str, int]] = {}
        for r in rows:
            if "segments" not in by_id[r["case_id"]]["expected"] or "program" not in r["actual"]:
                continue
            for expected_row, actual_row in zip(r["expected"], r["actual"]["program"]):
                matrix.setdefault(expected_row["entry_id"], {}).setdefault(actual_row["entry_id"], 0)
                matrix[expected_row["entry_id"]][actual_row["entry_id"]] += 1
        return matrix

    confusion = {
        "schema": "g04.confusion.v1",
        "offline": {"errors": errors_of(offline_rows), "entry_matrix": confusion_of(offline_rows), "by_error": _count([r["error"] for r in offline_rows if r["error"]])},
        "model": {"errors": errors_of(model_rows), "entry_matrix": confusion_of(model_rows), "by_error": _count([r["error"] for r in model_rows if r["error"]])} if model_rows else None,
    }
    unsupported = {
        "schema": "g04.unsupported.v1",
        "note": "language the label reads as a motion and the planner refused, plus stated quantities grounding refuses typed; counted apart from errors",
        "offline": unsupported_of(offline_rows),
        "model": unsupported_of(model_rows) if model_rows else None,
        "grounding": ([{"case_id": g["case_id"], "prompt": g["prompt"], "body": g["body"], "planner": name, "failure_stage": g["failure_stage"], "failure_code": g["failure_code"], "detail": g["failure_detail"]}
                       for name, rows_ in (grounding or {}).items() if name != "body" for g in rows_ if not g["accepted"]] if grounding else None),
    }
    summary = {
        "schema": "g04.summary.v1", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
        "fixture_registration": registration["registration_sha256"], "model": args.model, "label_provenance": "internal",
        "offline": summarize(offline_rows), "model_planner": summarize(model_rows) if model_rows else None,
        "stopped": stop_detail, "fallback_used": bool(planner and planner.spend.fallback_used),
        "agreement": {name: {"cases": len(rows_), "all_agree": sum(1 for r in rows_ if r["all_agree"]), "applicable_body_cases": sum(r["applicable_bodies"] for r in rows_)} for name, rows_ in agreement.items()},
        "spend": {"totals": log.totals(), "budget": budget.__dict__, "planner": planner.spend.__dict__ if planner else None},
        "grounding": {name: {"cases": len(rows_), "accepted": sum(1 for g in rows_ if g["accepted"])} for name, rows_ in (grounding or {}).items() if name != "body"} if grounding else None,
    }
    json_dump(out / "cases.json", {"offline": offline_rows, "model": model_rows})
    json_dump(out / "agreement.json", agreement)
    json_dump(out / "confusion.json", confusion)
    json_dump(out / "unsupported.json", unsupported)
    if grounding:
        json_dump(out / "grounding.json", grounding)
    json_dump(out / "summary.json", summary)
    print(json.dumps({"offline": summary["offline"]["by_kind"], "model": summary["model_planner"]["by_kind"] if summary["model_planner"] else None, "agreement": summary["agreement"], "spend": summary["spend"]["totals"], "stopped": stop_detail}, indent=1))
    return 0


def _count(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    sys.exit(main())
