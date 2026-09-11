"""The offline-scoring seam: same code path, same payload, both checked.

Plan 10 §8.1. `--offline-scoring` is a stand-in for the real caller, and a
harness fails in a specific way — it can run the real path on a payload the real
caller would never have sent. So the replay is asserted twice over: the scorer is
the *identical function* a live run calls, and the payload it is handed is the
one that was actually recorded.

Reassigned to this lane with 10d; the design is lane `groundtruth`'s.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from evals.calibration import ReplayError, ReplayStore, graders_from_record, prompt_versions_agree
from rigby_poc.judge import GRADER_OUTPUT_MODELS, VLMJudge, assemble_split_score
from rigby_poc.judge_claims import claim_specs
from rigby_poc.judge_prompts import GRADER_CORES, GRADER_NAMES

pytestmark = pytest.mark.medium


INTENT = "gesture"


class FakeUsage:
    def model_dump(self, mode: str = "json") -> dict[str, int]:
        return {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}


class FakeResponses:
    def __init__(self, verdicts: dict[str, str] | None = None) -> None:
        self.calls: list[dict] = []
        self.verdicts = verdicts or {}

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["input"][0]["content"][0]["text"]
        grader = next(
            name for name in GRADER_NAMES if system.startswith(GRADER_CORES[name][:60])
        )
        payload: dict[str, object] = {
            "claims": [
                {
                    "id": spec.id,
                    "verdict": self.verdicts.get(grader, "yes"),
                    "confidence": 0.9,
                    "snapshot_id": "01-ego",
                }
                for spec in claim_specs(grader, intent=INTENT)
            ],
            "summary": f"{grader} saw nothing notable.",
        }
        if grader == "semantic":
            payload["suggested_adjustment"] = "Preserve the pose."
        return SimpleNamespace(
            id=f"resp_{grader}",
            model=kwargs["model"],
            usage=FakeUsage(),
            output_parsed=GRADER_OUTPUT_MODELS[grader].model_validate(payload),
        )


class FakeClient:
    def __init__(self, verdicts: dict[str, str] | None = None) -> None:
        self.responses = FakeResponses(verdicts)


def _manifest(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, view in enumerate(("ego", "orbit"), start=1):
        buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), color=(9 * index, 20, 40)).save(buffer, format="PNG")
        image = buffer.getvalue()
        (tmp_path / f"{view}.png").write_bytes(image)
        snapshots.append(
            {
                "id": f"01-{view}",
                "phase": "hold",
                "label": "presented_pose",
                "view": view,
                "requested_time_s": 0.5,
                "rendered_time_s": 0.5,
                "path": f"{view}.png",
                "sha256": hashlib.sha256(image).hexdigest(),
                "width_px": 1600,
                "height_px": 900,
                "camera": {
                    "view": view,
                    "width_px": 1600,
                    "height_px": 900,
                    "vertical_fov_deg": 94 if view == "ego" else 46,
                },
            }
        )
    path = tmp_path / "evidence-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "result_id": "replay-result",
                "intent": INTENT,
                "prompt": "Throw up a hang-ten sign.",
                "capture_contract": {
                    "raw_canvas_only": True,
                    "width_px": 1600,
                    "height_px": 900,
                    "egocentric_vertical_fov_deg": 94.0,
                    "device_pixel_ratio": 1,
                    "ui_overlay_included": False,
                },
                "snapshots": snapshots,
            }
        ),
        encoding="utf-8",
    )
    return path


def _live_record(tmp_path: Path, verdicts: dict[str, str] | None = None) -> dict:
    judge = VLMJudge(client=FakeClient(verdicts), model="test-vlm", grader_mode="split")
    return judge.score(_manifest(tmp_path / "evidence"))


# ------------------------------------------------------- same code path


def test_replaying_a_recorded_case_reproduces_the_live_score_exactly(tmp_path: Path) -> None:
    """The whole point: the offline number IS the online number.

    Not `pytest.approx` and not a subset — the full score payload, because a
    replay that reproduces most of a score is a replay that has forked somewhere.
    """
    live = _live_record(tmp_path)
    store = ReplayStore(tmp_path / "store")
    store.write_record("case-1", live)

    replayed, _ = assemble_split_score(store.parts("case-1"), intent=INTENT)

    assert replayed.model_dump(mode="json") == live["call"]["parsed"]


def test_a_failing_case_replays_as_failing(tmp_path: Path) -> None:
    live = _live_record(tmp_path, verdicts={"anatomy": "no"})
    store = ReplayStore(tmp_path / "store")
    store.write_record("case-bad", live)

    replayed, verdicts = assemble_split_score(store.parts("case-bad"), intent=INTENT)

    assert replayed.accept is False
    assert replayed.model_dump(mode="json") == live["call"]["parsed"]
    assert verdicts["anatomical_naturalness"].score == 1


def test_there_is_no_scoring_function_in_the_replay_module() -> None:
    """The absence of a wrapper is the guarantee (plan 10 §8.1).

    A convenience wrapper around `assemble_split_score` is where an `if offline`
    branch would eventually be added, and a fork there means the offline number
    is not the online number. Asserted structurally so the wrapper cannot be
    reintroduced as a tidy-up.
    """
    from evals.calibration import replay

    source = Path(replay.__file__).read_text(encoding="utf-8")
    assert "assemble_split_score" not in source.replace(
        "`judge.assemble_split_score`", ""
    ).replace("judge.assemble_split_score", "")


# ---------------------------------------------------------- same payload


def test_each_stored_payload_is_the_one_its_own_grader_answered(tmp_path: Path) -> None:
    """The second half of §8.1: a harness can run the real path on a payload the
    real caller would never have sent.

    The failure this rules out is a payload landing under the wrong grader —
    which would replay cleanly, score plausibly, and measure the anatomy grader's
    claims against the timing grader's answers. Matched by the claim ids each
    grader was actually asked, not by position in the record.
    """
    judge = VLMJudge(client=FakeClient(), model="test-vlm", grader_mode="split")
    live = judge.score(_manifest(tmp_path / "evidence"))

    store = ReplayStore(tmp_path / "store")
    store.write_record("case-1", live)
    stored = store.parts("case-1")

    for name in GRADER_NAMES:
        assert stored[name] == live["graders"][name]["call"]["parsed"]
        answered = {claim["id"] for claim in stored[name]["claims"]}
        asked = {spec.id for spec in claim_specs(name, intent=INTENT)}
        assert answered == asked, f"{name} holds another grader's claim set"


def test_a_payload_stored_under_the_wrong_grader_is_caught(tmp_path: Path) -> None:
    # The same check, shown to bite: swap two graders' payloads and the claim
    # ids no longer match the questions that grader was asked.
    judge = VLMJudge(client=FakeClient(), model="test-vlm", grader_mode="split")
    live = judge.score(_manifest(tmp_path / "evidence"))
    live["graders"]["timing"]["call"]["parsed"] = live["graders"]["anatomy"]["call"]["parsed"]

    store = ReplayStore(tmp_path / "store")
    store.write_record("case-1", live)
    stored = store.parts("case-1")

    answered = {claim["id"] for claim in stored["timing"]["claims"]}
    asked = {spec.id for spec in claim_specs("timing", intent=INTENT)}
    assert answered != asked


# ------------------------------------------------------------- the store


def test_a_partial_case_raises_rather_than_scoring_four_graders(tmp_path: Path) -> None:
    live = _live_record(tmp_path)
    store = ReplayStore(tmp_path / "store")
    store.write_record("case-1", live)
    (tmp_path / "store" / "case-1" / "timing.json").unlink()

    with pytest.raises(ReplayError, match="no recorded payload for grader timing"):
        store.parts("case-1")


def test_the_key_is_case_and_grader_not_case_grader_and_dimension(tmp_path: Path) -> None:
    """`semantic` owns two dimensions from one claim set, so dimension is not a key.

    Keying by dimension would let the second silently overwrite the first, and
    the replayed score would be subtly wrong rather than absent.
    """
    live = _live_record(tmp_path)
    store = ReplayStore(tmp_path / "store")
    paths = store.write_record("case-1", live)

    assert len(paths) == len(GRADER_NAMES)
    assert len({path.name for path in paths}) == len(GRADER_NAMES)
    semantic = json.loads((tmp_path / "store" / "case-1" / "semantic.json").read_text())
    ids = {claim["id"] for claim in semantic["parsed"]["claims"]}
    assert {"requested_action_is_present", "action_is_recognizable_unprompted"} <= ids


def test_a_non_split_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="needs a split_motion_judgment"):
        graders_from_record("case-1", {"kind": "unary_motion_judgment"})


def test_an_unsafe_case_id_cannot_escape_the_store(tmp_path: Path) -> None:
    store = ReplayStore(tmp_path / "store")
    for unsafe in ("../escape", "a/b", ""):
        with pytest.raises(ReplayError, match="unsafe case id"):
            store.parts(unsafe)


def test_cases_are_listed_from_disk(tmp_path: Path) -> None:
    live = _live_record(tmp_path)
    store = ReplayStore(tmp_path / "store")
    store.write_record("b-case", live)
    store.write_record("a-case", live)
    assert store.cases() == ["a-case", "b-case"]


# ------------------------------------------------------- prompt versions


def test_recordings_spanning_a_prompt_change_are_visible(tmp_path: Path) -> None:
    """A calibration result is only meaningful against a stated prompt version.

    Two shas for one grader means the recordings span a prompt change, and
    pooling them reports one calibration for two instruments.
    """
    live = _live_record(tmp_path)
    store = ReplayStore(tmp_path / "store")
    store.write_record("case-1", live)
    store.write_record("case-2", live)
    assert all(len(shas) == 1 for shas in prompt_versions_agree(store, store.cases()).values())

    drifted = json.loads((tmp_path / "store" / "case-2" / "timing.json").read_text())
    drifted["prompt"]["sha256"] = "f" * 64
    (tmp_path / "store" / "case-2" / "timing.json").write_text(json.dumps(drifted))

    seen = prompt_versions_agree(store, store.cases())
    assert len(seen["timing"]) == 2
    assert len(seen["anatomy"]) == 1
