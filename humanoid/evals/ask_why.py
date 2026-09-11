"""Ask the model why it repeated a goal it had been told could not be moved.

Every mechanism built to catch this fired correctly. The planner was woken on
the stall rather than a clock, told which goal had stopped moving, told in plain
words that "no move of the named packages improves this number at all", and
shown a climbing count of how often it had asked. It asked again. Six times.

That leaves a question the transcripts cannot answer: was the signal not read,
not believed, not understood, or read and reasonably overruled? Only the model
can say, so this asks it -- with the same numbers it was holding at the moment,
and its own words back.

A limitation, stated rather than hidden: the interview replays the situation
JSON but NOT the two camera images, because reproducing those means
re-simulating the run frame by frame. So this can tell us how the model reads
the readings; it cannot rule out that the pictures were driving the choice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

INTERVIEW = """You are being asked about a decision you made while directing a
robot arm. This is not a trick and nothing turns on defending it -- an honest
account of your reasoning is worth more here than a good justification.

You will be shown what you were told at that moment and what you then asked for.
The specific thing being asked about: you were told the goal you had been
driving was STALLED, that no available move could improve it at all, and how
many times you had already asked for that same number. You asked for it again.

Answer plainly:
  "read_it": did you take in the stalled/no-move signal at all?
  "believed_it": did you believe it, or did you think the search was wrong?
  "why_again": in one or two sentences, why you asked for the same number again
  "what_would_have_helped": what, in the payload, would have made you choose
                            differently

Reply as JSON with exactly those four keys."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--metric", default="")
    parser.add_argument("--max-questions", type=int, default=3)
    args = parser.parse_args()

    document = json.loads(Path(args.source).read_text(encoding="utf-8"))
    decisions = [e for e in document.get("transcript", []) if "target" in e]

    # The decisions worth asking about: told it was stalled, told nothing helps,
    # and asked for that same metric again anyway.
    guilty = []
    for index, entry in enumerate(decisions[1:], start=1):
        before = decisions[index - 1]
        situation = entry.get("predicament") or {}
        if situation.get("woken_because") != "stalled":
            continue
        if "out of options" not in str(situation.get("search_says", "")):
            continue
        if entry.get("target") != before.get("target"):
            continue
        if args.metric and entry.get("target") != args.metric:
            continue
        guilty.append((before, entry))

    print(f"  decisions that repeated a stalled goal: {len(guilty)}")
    if not guilty:
        return 0

    from rigby_poc.gripper.decision.planner import _load_env

    _load_env()
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("RIGBY_VLM_MODEL", "gpt-4o")

    for before, entry in guilty[: args.max_questions]:
        asked = {
            "what_you_were_told": {
                "predicament": entry.get("predicament"),
                "sensed": entry.get("sensed"),
                "your_body_right_now": entry.get("your_body_right_now"),
            },
            "what_you_had_just_been_driving": {
                "target": before.get("target"), "value": before.get("value"),
                "your_reason_then": before.get("why"),
            },
            "what_you_then_asked_for": {
                "target": entry.get("target"), "value": entry.get("value"),
                "your_reason": entry.get("why"),
            },
        }
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": INTERVIEW},
                      {"role": "user",
                       "content": json.dumps(asked, sort_keys=True)}],
            response_format={"type": "json_object"},
            max_completion_tokens=400,
        )
        said = json.loads(response.choices[0].message.content or "{}")
        print()
        print(f"  --- t={entry.get('t')}  asked {entry.get('target')}"
              f" = {entry.get('value')} again")
        for key in ("read_it", "believed_it", "why_again",
                    "what_would_have_helped"):
            print(f"    {key:<24} {said.get(key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
