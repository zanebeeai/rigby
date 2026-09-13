"""Acquisition bound to a body: parameter vectors become configurations, problems become worlds, and one single-shot trial runs and seals.

A parameter vector of the contact transfer family turns into the arm
controller, the closure configuration and the duration scale the transfer
takes, with the defaults reproducing the G06 primitive exactly. A problem's
declared change turns into a world: half friction leaves the cube where it
was and halves its coefficient; a grown cube keeps its base on the bench
and scales its mass by its volume. On the long arm one single-shot trial
at the defaults certifies, and a slowed vector runs longer in physics than
the defaults do. Sealed, the trial replays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rigby_core.skills import AcquisitionProblemV1
from rigby_general.contact.closure import ClosureConfig
from rigby_general.gates.control import ControllerConfig
from rigby_general.sensing import load_policy
from rigby_general.skills.acquisition_runtime import configs_of, outcome_of, problem_world, run_single_shot, seal_single_shot
from rigby_general.skills.skill_store import goal_for


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g10_corpus as g10  # noqa: E402
import g12_protocol as protocol  # noqa: E402


def test_defaults_reproduce_the_g06_configurations():
    controller, closure, scale = configs_of({p.name: p.default for p in protocol.PARAMETERS})
    assert controller == ControllerConfig() and closure == ClosureConfig() and scale == 1.0
    controller, closure, scale = configs_of({"closure.grip_safety_factor": 200.0, "arm.natural_frequency_hz": 7.0, "duration_scale": 2.5})
    assert closure.grip_safety_factor == 200.0 and controller.natural_frequency_hz == 7.0 and scale == 2.5


def test_problem_worlds_apply_the_declared_change_and_nothing_else():
    base = g10.environment()
    by_id = {p.problem_id: p for p in protocol.PROBLEMS}
    slick = problem_world(by_id["jaw-slick"], base, protocol.CHANGES)
    assert slick.objects[0].friction == pytest.approx(0.7) and slick.objects[0].position_m == base.objects[0].position_m and slick.objects[0].size_m == base.objects[0].size_m
    same = problem_world(by_id["hand-fixed-world"], base, protocol.CHANGES)
    assert same.content_hash() == base.content_hash()
    grown = problem_world(by_id["long-heavy-large"], base, protocol.CHANGES)
    assert grown.objects[0].size_m == (0.0175, 0.0175, 0.0175)
    assert grown.objects[0].position_m[2] == pytest.approx(base.objects[0].position_m[2] + 0.0025)
    assert grown.objects[0].mass_kg == pytest.approx(base.objects[0].mass_kg * (0.0175 / 0.015) ** 3 * 1.5, rel=1e-6)
    assert grown.fixtures == base.fixtures


def test_a_single_shot_trial_on_the_long_arm_certifies_and_seals(tmp_path):
    base = g10.environment()
    policy = load_policy(g10.G09 / "policy.json")
    source = ROOT / "assets/general/zoo/zoo_long_arm/robot.urdf"
    defaults = {p.name: p.default for p in protocol.PARAMETERS}
    session, result, wall = run_single_shot("zoo_long_arm", source, base, goal_for(base), policy, defaults, seed_label="t", record=True)
    assert result.certified, result.failed_gate
    outcome = outcome_of(0, result, wall)
    assert outcome.certified and outcome.physics_s > 5.0
    sealed = seal_single_shot(tmp_path / "physical", session=session, result=result, label="t", caption="test", parameters=defaults)
    assert sealed["replay_agrees"] and sealed["outcome"] == "success"
    slowed = dict(defaults, duration_scale=2.0)
    _, slow_result, _ = run_single_shot("zoo_long_arm", source, base, goal_for(base), policy, slowed, seed_label="slow")
    assert slow_result.executed and float(slow_result.times_s[-1] - slow_result.times_s[0]) > outcome.physics_s * 1.3
