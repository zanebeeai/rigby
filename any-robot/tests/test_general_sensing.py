"""Declared sensors over a recorded transfer, and what the monitor may and may not know.

One jaw-arm transfer in the registered fixed world is recorded and sealed
once per module. Pinned on it: a camera reports occlusion, never a
position, when the hand stands between it and the cube; the same six
contracts bind to every configuration and decide only where their kinds
exist; degraded evidence (a screen, a frozen sensor, an absent sensor, a
sparse sensor) never yields a pass at an instant the full evidence
passed; a privileged oracle sensor in the configuration changes no verdict;
and the oracle labels agree with the decided verdicts through the hold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rigby_core.skills import Decision, EvidenceKind, SampleQuality

from rigby_general.contact.placement import PlacementGoal
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import CONFIGURATIONS, Episode, EvidenceStreams, OracleLabeler, configuration, conditionals_for, decide, load_policy
from rigby_general.sensing.corrupt import absent, occluded_by_screen, sensors_of, sparse, stale


ROOT = Path(__file__).resolve().parents[1]
ZOO_ROOT = ROOT / "assets" / "general" / "zoo"
G06 = ROOT / "assets" / "general" / "research-protocols" / "g06-transfer-v1"
POLICY = ROOT / "assets" / "general" / "research-protocols" / "g09-conditionals-v1" / "policy.json"
sys.path.insert(0, str(ROOT / "scripts"))
import g06_transfer_campaign as campaign  # noqa: E402


@pytest.fixture(scope="module")
def episode(tmp_path_factory) -> Episode:
    env = EnvironmentV1.model_validate_json((G06 / "environment.json").read_bytes())
    raw = json.loads((G06 / "goal.json").read_bytes())
    goal = PlacementGoal(region_minimum_m=tuple(raw["region_minimum_m"]), region_maximum_m=tuple(raw["region_maximum_m"]), dwell_s=raw["dwell_s"],
                         maximum_linear_speed_mps=raw["maximum_linear_speed_mps"], maximum_angular_speed_radps=raw["maximum_angular_speed_radps"])
    source = ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf"
    robot = ingest_robot(source, robot_id="zoo_jaw_arm")
    result, recorder, scene, _ = campaign.run_one(robot, source, env, goal, record=True)
    assert result.certified, result.violations
    destination = tmp_path_factory.mktemp("sensing") / "physical"
    campaign.seal(destination, label="zoo_jaw_arm-test", robot=robot, source=source, scene=scene, env=env, goal=goal, result=result, recorder=recorder,
                  track="strict_fixed_world", trial={"kind": "test"}, caption="test")
    return Episode.load(destination)


@pytest.fixture(scope="module")
def policy() -> dict:
    return load_policy(POLICY)


def hold_instant(episode: Episode) -> float:
    phase = next(p for p in episode.phases() if p["name"] == "hold")
    return 0.5 * (phase["start_s"] + phase["end_s"])


def test_a_camera_reports_occlusion_not_a_position_when_the_hand_hides_the_cube(episode: Episode) -> None:
    overhead = EvidenceStreams.build(episode, configuration("overhead_contact", episode.effectors)).streams["camera:overhead"]
    front = EvidenceStreams.build(episode, configuration("front_contact", episode.effectors)).streams["camera:front"]
    hold = hold_instant(episode)
    during = [i for i, t in enumerate(overhead.times) if hold - 0.5 <= t <= hold]
    assert during and all(overhead.quality[i] is SampleQuality.OCCLUDED for i in during)
    assert all("x" not in overhead.values[i] for i in during), "an occluded sample carries no position"
    seen = [i for i, t in enumerate(front.times) if hold - 0.5 <= t <= hold]
    assert sum(front.quality[i] is SampleQuality.VALID for i in seen) >= 0.8 * len(seen)
    # An approach from rest is visible from above; the camera is not blind, the hand is in the way later.
    early = [i for i, t in enumerate(overhead.times) if t <= episode.start_s + 1.0]
    assert all(overhead.quality[i] is SampleQuality.VALID for i in early)


def test_the_same_contracts_bind_to_every_configuration(episode: Episode, policy: dict) -> None:
    effector = episode.effectors[0]
    conditionals = conditionals_for(episode, effector, policy)
    hold = hold_instant(episode)
    verdicts = {}
    for name in CONFIGURATIONS:
        streams = EvidenceStreams.build(episode, configuration(name, episode.effectors))
        verdicts[name] = {p: decide(c, streams, hold) for p, c in conditionals.items()}
    # Decided where the kinds exist, unknown with the missing source named where they do not; the contract is the same object throughout.
    assert verdicts["front_contact"]["held"].decision is Decision.PASS and verdicts["front_contact"]["opposition_established"].decision is Decision.PASS
    assert verdicts["contact_only"]["held"].decision is Decision.PASS
    assert verdicts["contact_only"]["reachable"].reason == "missing_source:object_pose"
    assert verdicts["front_vision_only"]["held"].reason == "missing_source:contact_force"
    assert verdicts["overhead_contact"]["reachable"].reason == "occluded:camera:overhead"
    assert verdicts["front_vision_only"]["reachable"].decision is Decision.PASS
    for p in conditionals:
        assert conditionals[p].rule.policy == policy["policy_id"]


def test_thresholds_come_from_the_policy_and_task_not_the_robot(episode: Episode, policy: dict) -> None:
    conditionals = conditionals_for(episode, episode.effectors[0], policy)
    assert conditionals["held"].rule.parameters["contact_force_n"] == policy["rules"]["contact_force_n"]
    assert conditionals["held"].rule.parameters["half_height_m"] == pytest.approx(float(episode.object_half_extent_m[2]))
    assert conditionals["stably_placed"].rule.parameters["max_speed_mps"] == episode.goal.maximum_linear_speed_mps
    assert "calibration" in policy and policy["calibration"]["episodes"], "the policy records what it was calibrated on"
    for key in ("contact_force_n", "held", "moving_with_robot", "stably_placed", "reachable"):
        assert any(k.endswith(key) for k in policy["calibration"]["results"]), key


def test_degraded_evidence_never_passes_where_full_evidence_did(episode: Episode, policy: dict) -> None:
    effector = episode.effectors[0]
    conditionals = conditionals_for(episode, effector, policy)
    streams = EvidenceStreams.build(episode, configuration("front_contact", episode.effectors))
    hold = hold_instant(episode)
    held, opposition = conditionals["held"], conditionals["opposition_established"]
    reachable = conditionals["reachable"]
    assert decide(held, streams, hold).decision is Decision.PASS and decide(reachable, streams, hold).decision is Decision.PASS
    contact = sensors_of(streams, EvidenceKind.CONTACT_FORCE)
    assert decide(held, stale(streams, contact, hold - 0.3), hold).reason.startswith("stale:contact:palm")
    assert decide(opposition, absent(streams, EvidenceKind.CONTACT_FORCE), hold).reason == "missing_source:contact_force"
    assert decide(held, sparse(streams, contact, 6.0), hold).reason == "insufficient_samples:contact_force"
    screened = occluded_by_screen(streams, "camera:front", (-0.08, 1.0, 0.41), (0.3, 0.005, 0.3))
    verdict = decide(reachable, screened, hold)
    assert verdict.decision is Decision.UNKNOWN and verdict.reason == "occluded:camera:front" and verdict.fallback == "re_observe"
    assert all(q is SampleQuality.OCCLUDED for q in screened.streams["camera:front"].quality), "the screen hides the fixtures from the front camera throughout"


def test_an_oracle_sensor_in_the_configuration_changes_nothing(episode: Episode, policy: dict) -> None:
    effector = episode.effectors[0]
    conditionals = conditionals_for(episode, effector, policy)
    plain = EvidenceStreams.build(episode, configuration("front_contact", episode.effectors))
    with_oracle = EvidenceStreams.build(episode, configuration("front_contact_with_oracle", episode.effectors))
    assert "oracle" in with_oracle.streams and with_oracle.streams["oracle"].sensor.oracle
    for now in np.arange(episode.start_s + 0.5, episode.end_s, 1.0):
        for p, c in conditionals.items():
            a, b = decide(c, plain, float(now)), decide(c, with_oracle, float(now))
            assert (a.decision, a.reason, a.detail, a.sensors_used) == (b.decision, b.reason, b.detail, b.sensors_used), (p, now)
            assert "oracle" not in b.sensors_used


def test_oracle_labels_agree_with_decided_verdicts_through_the_transfer(episode: Episode, policy: dict) -> None:
    effector = episode.effectors[0]
    conditionals = conditionals_for(episode, effector, policy)
    streams = EvidenceStreams.build(episode, configuration("front_contact", episode.effectors))
    labeler = OracleLabeler(episode)
    hold, dwell = hold_instant(episode), episode.end_s - 0.1
    for p, now, expected in (("held", hold, True), ("opposition_established", hold, True), ("stably_placed", dwell, True), ("area_clear", dwell, False), ("held", dwell, False), ("area_clear", episode.start_s + 1.0, True)):
        label = labeler.label(p, effector, now, conditionals[p].window.duration_s)
        verdict = decide(conditionals[p], streams, now)
        assert label.value is expected, (p, now, label)
        assert verdict.decision is (Decision.PASS if expected else Decision.FAIL), (p, now, verdict)
