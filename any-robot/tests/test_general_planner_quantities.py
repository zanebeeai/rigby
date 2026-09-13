"""Stated quantities beside the metric-free program, and a model held to the contract.

'5 cm' is a number the person said; 'a little' is a region the body resolves.
The first has to survive the metric-free IR untouched and end as five
centimetres of travel; the second has to stay a qualitative remove. A model
that reads the request may copy a quantity the request states and may not
report one it does not: these tests hand the planner such replies through a
mock transport and check that the contract, not the model, has the last word.
No model call is made anywhere here.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_general.errors import GeneralFailureCode, RigbyGeneralError
from rigby_general.evidence.composition import run_prompt
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner, extract_quantities
from rigby_general.planner.model_planner import (
    Budget,
    BudgetStop,
    CallLog,
    MockTransport,
    ModelReply,
    ModelSchemaPlanner,
    ModelUnavailable,
    ResponseCache,
    cost_usd,
    reply_schema,
)
from rigby_general.schema.inventory import afforded_entries, load_inventory
from rigby_general.schema.program import Remove, assert_metric_free


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


@pytest.fixture(scope="module")
def jaw_arm():
    return ingest_robot(ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf", robot_id="zoo_jaw_arm")


@pytest.fixture(scope="module")
def afforded(inventory, jaw_arm):
    return afforded_entries(inventory, jaw_arm.morphology)


def farthest_travel_m(robot, run) -> float:
    model = robot.finalized.model
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, run.bound.grounded.figure_sites[0])
    trajectory = run.trajectory
    data.qpos[:] = trajectory.qpos[0]
    mujoco.mj_kinematics(model, data)
    origin = np.array(data.site_xpos[site])
    farthest = 0.0
    for qpos in trajectory.qpos[::5]:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        farthest = max(farthest, float(np.linalg.norm(np.array(data.site_xpos[site]) - origin)))
    return farthest


# --------------------------------------------------------------------------
# The offline recognizer keeps a stated distance beside the program
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clause, expected",
    [
        ("reach out 5 cm", ("5 cm", 0.05)),
        ("reach out five centimetres", ("five centimetres", 0.05)),
        ("move 10 centimeters forward", ("10 centimeters", 0.10)),
        ("go 0.2 m out", ("0.2 m", 0.2)),
        ("reach out 2 inches", ("2 inches", 0.0508)),
        ("extend 30 mm", ("30 mm", 0.03)),
    ],
)
def test_extract_quantities_reads_stated_distances(clause, expected):
    (quantity,) = extract_quantities([clause])
    assert quantity.segment_id == "s0"
    assert quantity.text == expected[0]
    assert quantity.value == pytest.approx(expected[1])
    assert quantity.unit == "m"
    assert quantity.provenance == "user_stated"


@pytest.mark.parametrize("clause", ["reach out a little", "reach in front of you", "wave twice", "put it in the box"])
def test_extract_quantities_finds_nothing_in_qualitative_wording(clause):
    assert extract_quantities([clause]) == ()


def test_offline_plan_request_keeps_the_program_metric_free(inventory, afforded):
    planner = OfflineSchemaPlanner(inventory)
    planned = planner.plan_request("reach out 5 cm then come back", afforded=afforded)
    assert_metric_free(planned.program)
    assert [q.segment_id for q in planned.quantities] == ["s0"]
    assert planned.distances_m() == {"s0": pytest.approx(0.05)}
    assert planned.program.segments[0].region.remove is Remove.MEDIAL
    little = planner.plan_request("reach out a little", afforded=afforded)
    assert little.quantities == ()
    assert little.program.segments[0].region.remove is Remove.PROXIMAL


# --------------------------------------------------------------------------
# Grounding: five centimetres is five centimetres, a little is body-relative
# --------------------------------------------------------------------------


def test_stated_five_centimetres_grounds_to_five_centimetres_of_travel(jaw_arm):
    runs, _, _, _ = run_prompt(jaw_arm, "reach out 5 cm", repeats=1)
    run = runs[0]
    assert run.accepted, (run.failure_stage, run.failure_detail)
    assert run.trace.requested_quantities[0]["text"] == "5 cm"
    assert farthest_travel_m(jaw_arm, run) == pytest.approx(0.05, abs=0.004)


def test_trace_json_carries_the_quantities_and_the_planner(jaw_arm):
    runs, _, _, _ = run_prompt(jaw_arm, "reach out 5 cm", repeats=1)
    payload = runs[0].trace.to_json()
    assert payload["requested_quantities"] == [{"segment_id": "s0", "kind": "distance", "text": "5 cm", "value": 0.05, "unit": "m", "provenance": "user_stated"}]
    assert payload["planner"] == {"planner_id": "offline-recognizer-v1", "model": None, "cached": False}


def test_qualitative_little_grounds_from_the_body_not_the_number(jaw_arm):
    runs, _, _, _ = run_prompt(jaw_arm, "reach out a little", repeats=1)
    run = runs[0]
    assert run.accepted, (run.failure_stage, run.failure_detail)
    assert run.trace.requested_quantities == []
    assert farthest_travel_m(jaw_arm, run) > 0.1


def test_stated_distance_beyond_reach_is_refused_typed(jaw_arm):
    runs, _, _, _ = run_prompt(jaw_arm, "reach out 4 m", repeats=1)
    run = runs[0]
    assert not run.accepted
    assert run.failure_stage == "binding"
    assert "outside what this body was measured to reach" in run.failure_detail


def test_stated_distance_on_an_oscillation_is_refused_not_reread(jaw_arm):
    runs, _, _, _ = run_prompt(jaw_arm, "wave 5 cm", repeats=1)
    run = runs[0]
    assert not run.accepted
    assert run.failure_stage == "binding"
    assert "amplitude or a span" in run.failure_detail


# --------------------------------------------------------------------------
# The model planner: the contract judges the model's reply
# --------------------------------------------------------------------------


def planner_with(inventory, *, canned=None, cache=None, log=None, budget=None, model="gpt-5-nano"):
    transport = MockTransport(inventory, canned=canned)
    return ModelSchemaPlanner(inventory, model=model, transport=transport, cache=cache, log=log, budget=budget, purpose="test"), transport


def test_mock_reading_matches_the_offline_program(inventory, afforded):
    planner, _ = planner_with(inventory)
    offline = OfflineSchemaPlanner(inventory)
    for prompt in ("reach out 5 cm", "wave twice slowly then come back", "make a fist", "reach out as far as you can"):
        model_program = planner.plan_request(prompt, afforded=afforded)
        offline_program = offline.plan_request(prompt, afforded=afforded)
        assert model_program.program.role_normalized_hash() == offline_program.program.role_normalized_hash(), prompt
        assert [(q.segment_id, q.text, q.value) for q in model_program.quantities] == [(q.segment_id, q.text, q.value) for q in offline_program.quantities]
        assert model_program.planner_id == "model-schema-planner-v1"
        assert model_program.model == "gpt-5-nano"
        assert_metric_free(model_program.program)


def test_reply_schema_enumerates_only_the_inventory(inventory):
    schema = reply_schema(inventory)
    entry_ids = schema["properties"]["segments"]["items"]["properties"]["entry_id"]["enum"]
    assert set(entry_ids) == {entry.entry_id for entry in inventory.entries}
    assert schema["additionalProperties"] is False


def test_invented_distance_is_a_prohibited_substitution(inventory, afforded):
    reply = {"segments": [{"clause": "reach out a little", "entry_id": "reach_to_point", "remove": "proximal",
                           "manner": {"speed": 0, "effort": 0, "smoothness": 0, "rhythm": 0, "amplitude": 0, "repetition": 0, "precision": 0, "repetition_count": None},
                           "posture": None, "quantities": [{"text": "5 cm", "value": 5, "unit": "cm"}]}], "unsupported_reason": None}
    planner, _ = planner_with(inventory, canned={"reach out a little": json.dumps(reply)})
    with pytest.raises(RigbyGeneralError) as caught:
        planner.plan_request("reach out a little", afforded=afforded)
    assert caught.value.code is GeneralFailureCode.PROHIBITED_SUBSTITUTION


def test_relabelled_distance_is_a_prohibited_substitution(inventory, afforded):
    reply = {"segments": [{"clause": "reach out 5 cm", "entry_id": "reach_to_point", "remove": "medial",
                           "manner": {"speed": 0, "effort": 0, "smoothness": 0, "rhythm": 0, "amplitude": 0, "repetition": 0, "precision": 0, "repetition_count": None},
                           "posture": None, "quantities": [{"text": "5 cm", "value": 50, "unit": "cm"}]}], "unsupported_reason": None}
    planner, _ = planner_with(inventory, canned={"reach out 5 cm": json.dumps(reply)})
    with pytest.raises(RigbyGeneralError) as caught:
        planner.plan_request("reach out 5 cm", afforded=afforded)
    assert caught.value.code is GeneralFailureCode.PROHIBITED_SUBSTITUTION


def test_unknown_entry_is_an_invented_binding(inventory, afforded):
    reply = {"segments": [{"clause": "reach out", "entry_id": "teleport", "remove": "medial",
                           "manner": {"speed": 0, "effort": 0, "smoothness": 0, "rhythm": 0, "amplitude": 0, "repetition": 0, "precision": 0, "repetition_count": None},
                           "posture": None, "quantities": []}], "unsupported_reason": None}
    planner, _ = planner_with(inventory, canned={"reach out": json.dumps(reply)})
    with pytest.raises(RigbyGeneralError) as caught:
        planner.plan_request("reach out", afforded=afforded)
    assert caught.value.code is GeneralFailureCode.INVENTED_BINDING


def test_unafforded_entry_is_refused_after_the_reading(inventory, afforded):
    planner, _ = planner_with(inventory)
    with pytest.raises(RigbyGeneralError) as caught:
        planner.plan_request("beckon it over", afforded=afforded)
    assert caught.value.code is GeneralFailureCode.UNAFFORDED_SCHEMA
    assert caught.value.details["entry_id"] == "draw_hither"


def test_out_of_scope_requests_never_reach_the_model(inventory, afforded):
    planner, transport = planner_with(inventory)
    with pytest.raises(RigbyGeneralError) as caught:
        planner.plan_request("drive across the room", afforded=afforded)
    assert caught.value.code is GeneralFailureCode.UNSUPPORTED_MORPHOLOGY
    assert transport.calls == 0


def test_cache_serves_the_second_reading_for_nothing(inventory, afforded, tmp_path):
    class Recording:
        name = "openai"
        paid = True

        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            reply = self.inner.complete(**kwargs)
            return ModelReply(text=reply.text, model=kwargs["model"], transport=self.name, prompt_tokens=2000, completion_tokens=100)

    transport = Recording(MockTransport(inventory))
    cache = ResponseCache(tmp_path / "cache")
    log = CallLog(tmp_path / "calls.jsonl")
    budget = Budget(hard_ceiling_usd=20.0, soft_cap_usd=18.0)
    planner = ModelSchemaPlanner(inventory, model="gpt-5-nano", transport=transport, cache=cache, log=log, budget=budget, purpose="test")
    first = planner.plan_request("reach out 5 cm", afforded=afforded)
    second = planner.plan_request("reach out 5 cm", afforded=afforded)
    assert transport.calls == 1
    assert first.cached is False and second.cached is True
    assert first.program.role_normalized_hash() == second.program.role_normalized_hash()
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [row["paid"] for row in rows] == [True, False]
    assert rows[0]["cost_usd"] == pytest.approx(cost_usd("gpt-5-nano", 2000, 100))
    assert log.totals()["paid_calls"] == 1
    assert budget.spent_usd == pytest.approx(rows[0]["cost_usd"])
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_soft_cap_stops_before_the_call(inventory, afforded):
    class Paid:
        name = "openai"
        paid = True

        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            raise AssertionError("must not be called")

    transport = Paid()
    planner = ModelSchemaPlanner(inventory, model="gpt-5-nano", transport=transport, budget=Budget(soft_cap_usd=18.0, spent_before_usd=17.9999), purpose="test")
    with pytest.raises(BudgetStop):
        planner.plan_request("reach out", afforded=afforded)
    assert transport.calls == 0
    assert planner.spend.stopped == "budget"


def test_billing_refusal_stops_spending_and_uses_no_fallback_unless_given(inventory, afforded):
    class Broke:
        name = "openai"
        paid = True

        def complete(self, **kwargs):
            raise ModelUnavailable("HTTP 429: insufficient_quota", reason="billing", transport="openai")

    planner = ModelSchemaPlanner(inventory, model="gpt-5-nano", transport=Broke(), purpose="test")
    with pytest.raises(ModelUnavailable) as caught:
        planner.plan_request("reach out", afforded=afforded)
    assert caught.value.reason == "billing"
    assert planner.spend.stopped == "billing"
    with pytest.raises(ModelUnavailable):
        planner.plan_request("wave", afforded=afforded)


def test_fallback_transport_is_recorded_as_used(inventory, afforded):
    class Down:
        name = "openai"
        paid = True

        def complete(self, **kwargs):
            raise ModelUnavailable("HTTP 503", reason="unavailable", transport="openai")

    fallback = MockTransport(inventory)
    planner = ModelSchemaPlanner(inventory, model="gpt-5-nano", transport=Down(), purpose="test", fallback=("gemini-2.0-flash", fallback))
    planned = planner.plan_request("reach out 5 cm", afforded=afforded)
    assert planned.model == "gemini-2.0-flash"
    assert planner.spend.fallback_used is True
    assert planner.spend.dollars == 0.0


def test_unknown_model_cannot_be_called(inventory):
    with pytest.raises(KeyError):
        ModelSchemaPlanner(inventory, model="gpt-99", transport=MockTransport(inventory))


def test_model_reading_grounds_through_the_same_pipeline(inventory, jaw_arm):
    planner, transport = planner_with(inventory)
    runs, _, _, _ = run_prompt(jaw_arm, "reach out 5 cm", repeats=1, planner=planner)
    run = runs[0]
    assert run.accepted, (run.failure_stage, run.failure_detail)
    assert run.trace.planner == {"planner_id": "model-schema-planner-v1", "model": "gpt-5-nano", "cached": False}
    assert farthest_travel_m(jaw_arm, run) == pytest.approx(0.05, abs=0.004)
