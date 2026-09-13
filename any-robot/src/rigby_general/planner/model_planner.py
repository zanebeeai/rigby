"""A language model chooses among the closed class; the contract judges it.

This is the only module in the repository that issues a model call. Every
other planner is the offline recognizer, and every scored result before G04
came from it. The model is asked one question per request -- which entries of
the sealed inventory this request names, with what remove, manner, posture
and stated quantities -- and it answers inside a JSON schema whose enumerations
are the inventory's own ids. It cannot name a joint, a metre or a schema that
does not exist, because the schema has no field for one, and what it does say
is validated against the same ``MotionSchemaProgramV1`` contract and the same
inventory the offline planner is held to. The reading is body-neutral on
purpose: the model never learns which body will answer, so one call serves
every body and the affordance check runs afterwards, exactly as offline.

Three things are kept beside every call because money and provenance are both
finite. Every response is cached by the hash of the model, the system prompt,
the schema and the request, so a re-run of a fixture costs nothing and reads
the same words. Every call that reaches a provider is logged with its model,
its token counts as the provider reported them, its purpose and its cost at
the published rates recorded below. And a budget object is consulted before
each paid call, so the planner stops itself at the soft cap rather than
discovering the ceiling on the invoice.

Transports are pluggable so the same planner runs against a provider, a
free-plan fallback, or a mock that counts tokens without spending any:

* :class:`OpenAITransport` -- the scored route, strict structured output.
* :class:`GeminiTransport` -- fallback only, JSON mode, never a scored run.
* :class:`MockTransport` -- the offline recognizer's reading wrapped as a
  reply, with token counts estimated from characters, for dry runs and tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..errors import GeneralFailureCode, RigbyGeneralError
from ..guardrails import screen
from ..schema.inventory import SchemaEntry, SchemaInventory
from ..schema.program import (
    Concurrency,
    Dimensionality,
    Flexion,
    MannerV1,
    MemberGroup,
    MotionSchemaProgramV1,
    POSTURE_REMOVE,
    PostureV1,
    RegionV1,
    Remove,
    RoleBindingV1,
    SegmentLinkV1,
    SegmentV1,
)
from .quantities import PlannedRequestV1, QuantityKind, RequestedQuantityV1
from .schema_planner import (
    OfflineSchemaPlanner,
    PlannerTrace,
    _SEQUENCE_SPLIT,
    _boundary_for,
    _frame_for,
)


# -- published rates -----------------------------------------------------------
#
# USD per million tokens, (input, output), as published on OpenAI's pricing page
# and recorded here on the date below. Output includes reasoning tokens where a
# model bills them. Cached-input discounts are ignored: every input token is
# charged at the full rate, so the figure here can only overstate the invoice.
# A model with no row cannot be called, because a call whose cost cannot be
# tracked is a call outside the authorization.
PUBLISHED_RATES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.0-flash": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
}
"""Gemini rows are the free plan: the key is a free-plan key and the fallback is never scored."""
RATES_RECORDED_ON = "2026-09-13"
CHARS_PER_TOKEN_ESTIMATE = 3.5
"""Conservative for English prose and JSON (four is the usual figure)."""
REASONING_ALLOWANCE_TOKENS = {"gpt-5-nano": 256, "gpt-5-mini": 256, "gpt-5": 512}
"""Output tokens a reasoning model may bill beyond the visible reply at minimal effort; used by projections only."""

UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254}


def base_model(model: str) -> str:
    """``gpt-5-mini@low`` names gpt-5-mini at reasoning effort ``low``; rates are by the base name."""

    return model.split("@", 1)[0]


def reasoning_effort(model: str) -> str | None:
    return model.split("@", 1)[1] if "@" in model else None


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PUBLISHED_RATES_USD_PER_MTOK.get(base_model(model))
    if rates is None:
        raise KeyError(f"no published rate recorded for {model!r}; it cannot be called under the budget")
    return prompt_tokens * rates[0] / 1e6 + completion_tokens * rates[1] / 1e6


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


# -- replies, transports -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelReply:
    text: str
    model: str
    transport: str
    prompt_tokens: int
    completion_tokens: int
    """Every billed output token, reasoning included."""
    estimated: bool = False
    """True when the counts were estimated rather than reported by a provider."""


class ModelUnavailable(RuntimeError):
    """The provider refused to serve: billing, quota, authentication or outage.

    ``reason`` is ``billing`` when the refusal names quota or billing, which the
    authorization treats as a stop, and ``unavailable`` otherwise.
    """

    def __init__(self, message: str, *, reason: str, transport: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.transport = transport


class BudgetStop(RuntimeError):
    """The next call would cross the soft cap; nothing was spent."""


class Transport(Protocol):
    name: str
    paid: bool

    def complete(self, *, model: str, system: str, user: str, schema: dict, purpose: str) -> ModelReply: ...


def _billing_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("insufficient_quota", "billing", "quota", "payment", "402"))


class OpenAITransport:
    """Strict structured output over the official SDK. The scored route."""

    name = "openai"
    paid = True

    def __init__(self, api_key: str | None = None, *, timeout_s: float = 60.0) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise ModelUnavailable("OPENAI_API_KEY is not set", reason="unavailable", transport=self.name)
        import openai  # imported here so the module loads without the SDK

        self._openai = openai
        self._client = openai.OpenAI(api_key=key, timeout=timeout_s, max_retries=2)

    def complete(self, *, model: str, system: str, user: str, schema: dict, purpose: str) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": base_model(model),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "schema_reading", "strict": True, "schema": schema}},
        }
        if base_model(model).startswith("gpt-5"):
            kwargs["reasoning_effort"] = reasoning_effort(model) or "minimal"
        else:
            kwargs["temperature"] = 0.0
            kwargs["seed"] = 0
        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._openai.APIStatusError as error:  # type: ignore[attr-defined]
            body = str(getattr(error, "message", "") or error)
            reason = "billing" if error.status_code in (402,) or _billing_refusal(body) else "unavailable"
            raise ModelUnavailable(f"{model}: HTTP {error.status_code}: {body[:200]}", reason=reason, transport=self.name) from error
        except self._openai.APIError as error:  # type: ignore[attr-defined]
            raise ModelUnavailable(f"{model}: {str(error)[:200]}", reason="unavailable", transport=self.name) from error
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            text = json.dumps({"segments": [], "unsupported_reason": f"model refusal: {choice.message.refusal}"[:300]})
        else:
            text = choice.message.content or ""
        usage = response.usage
        completion = int(usage.completion_tokens or 0)
        return ModelReply(text=text, model=response.model or model, transport=self.name,
                          prompt_tokens=int(usage.prompt_tokens or 0), completion_tokens=completion)


class GeminiTransport:
    """JSON mode over the REST API of a free-plan key. Fallback only.

    Gemini's schema dialect is narrower than the strict one used above, so the
    schema is described in the system text and the reply is validated exactly
    as any other. Nothing produced here is scored; the campaign records where
    it was used.
    """

    name = "gemini"
    paid = False

    def __init__(self, api_key: str | None = None, *, timeout_s: float = 60.0) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        if not key.strip():
            raise ModelUnavailable("GEMINI_API_KEY is not set", reason="unavailable", transport=self.name)
        self._key = key
        self._timeout_s = timeout_s

    def complete(self, *, model: str, system: str, user: str, schema: dict, purpose: str) -> ModelReply:
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system + "\n\nReply with one JSON object matching this schema exactly:\n" + json.dumps(schema)}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
        }
        try:
            response = httpx.post(url, params={"key": self._key}, json=payload, timeout=self._timeout_s)
        except httpx.HTTPError as error:
            raise ModelUnavailable(f"{model}: {error}", reason="unavailable", transport=self.name) from error
        if response.status_code != 200:
            body = response.text[:200]
            raise ModelUnavailable(f"{model}: HTTP {response.status_code}: {body}", reason="billing" if _billing_refusal(body) else "unavailable", transport=self.name)
        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelUnavailable(f"{model}: reply carried no text", reason="unavailable", transport=self.name) from error
        usage = data.get("usageMetadata", {})
        return ModelReply(text=text, model=model, transport=self.name,
                          prompt_tokens=int(usage.get("promptTokenCount", 0)), completion_tokens=int(usage.get("candidatesTokenCount", 0)))


class MockTransport:
    """The offline recognizer's reading, returned as a model would return it.

    Token counts are estimated from characters, so a dry run over a fixture
    projects what the real run would cost without a single paid call. Canned
    replies keyed by the user text let a test hand the planner a reply the
    recognizer would never produce -- an invented quantity, an unknown entry --
    to check that validation refuses it.
    """

    name = "mock"
    paid = False

    def __init__(self, inventory: SchemaInventory, *, canned: dict[str, str] | None = None) -> None:
        self._offline = OfflineSchemaPlanner(inventory)
        self._inventory = inventory
        self._canned = dict(canned or {})
        self.calls = 0

    def complete(self, *, model: str, system: str, user: str, schema: dict, purpose: str) -> ModelReply:
        self.calls += 1
        prompt = user.split("Request: ", 1)[-1].strip()
        if prompt in self._canned:
            text = self._canned[prompt]
        else:
            text = self._offline_reading(prompt)
        prompt_tokens = estimate_tokens(system) + estimate_tokens(json.dumps(schema)) + estimate_tokens(user)
        return ModelReply(text=text, model=model, transport=self.name, prompt_tokens=prompt_tokens,
                          completion_tokens=estimate_tokens(text), estimated=True)

    def _offline_reading(self, prompt: str) -> str:
        try:
            planned = self._offline.plan_request(prompt, afforded=self._inventory.entries)
        except RigbyGeneralError as error:
            return json.dumps({"segments": [], "unsupported_reason": str(error)[:200]})
        by_segment: dict[str, list[dict]] = {}
        for quantity in planned.quantities:
            by_segment.setdefault(quantity.segment_id, []).append(
                {"text": quantity.text, "value": _value_in_unit(quantity), "unit": _unit_of(quantity)}
            )
        segments = []
        for segment, trace in zip(planned.program.segments, self._offline.last_trace, strict=True):
            posture = None
            if segment.posture is not None:
                posture = {"group": segment.posture.group.value, "selected_count": segment.posture.selected_count,
                           "selected": segment.posture.selected.value, "remainder": segment.posture.remainder.value,
                           "opposing": segment.posture.opposing.value}
            segments.append({
                "clause": trace.clause, "entry_id": trace.entry_id, "remove": segment.region.remove.value,
                "manner": {axis: getattr(segment.manner, axis) for axis in MANNER_AXES} | {"repetition_count": segment.manner.repetition_count},
                "posture": posture, "quantities": by_segment.get(segment.segment_id, []),
            })
        return json.dumps({"segments": segments, "unsupported_reason": None})


def _unit_of(quantity: RequestedQuantityV1) -> str:
    text = quantity.text.lower()
    for unit, pattern in (("mm", r"mm|millimet"), ("cm", r"cm|centimet"), ("in", r"\bin\b|inch"), ("m", r"\bm\b|metre|meter")):
        if re.search(pattern, text):
            return unit
    return "m"


def _value_in_unit(quantity: RequestedQuantityV1) -> float:
    unit = _unit_of(quantity)
    return round(quantity.value / UNIT_TO_M[unit], 6)


# -- cache, log, budget --------------------------------------------------------


class ResponseCache:
    """Replies by the hash of everything that determined them."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(*, model: str, system: str, schema: dict, prompt: str) -> str:
        material = "\n\x1e".join((model, system, json.dumps(schema, sort_keys=True), prompt))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_bytes())

    def put(self, key: str, entry: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        path.write_bytes((json.dumps(entry, indent=2, sort_keys=True) + "\n").encode("utf-8"))


class CallLog:
    """One line per call that reached a transport, mock and cache hits included.

    A cache hit is logged with ``cached: true`` and zero cost so the evidence
    shows every reading's origin; only lines with ``paid: true`` count toward
    the authorization's spend.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.rows: list[dict] = []

    def record(self, row: dict) -> None:
        self.rows.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))

    def totals(self) -> dict:
        paid = [row for row in self.rows if row["paid"]]
        return {
            "calls_logged": len(self.rows),
            "paid_calls": len(paid),
            "cached_hits": sum(1 for row in self.rows if row["cached"]),
            "mock_calls": sum(1 for row in self.rows if row["transport"] == "mock"),
            "prompt_tokens_paid": sum(row["prompt_tokens"] for row in paid),
            "completion_tokens_paid": sum(row["completion_tokens"] for row in paid),
            "dollars": round(sum(row["cost_usd"] for row in paid), 6),
        }


@dataclass
class Budget:
    """The authorization's numbers, consulted before every paid call."""

    hard_ceiling_usd: float = 20.0
    soft_cap_usd: float = 18.0
    spent_before_usd: float = 0.0
    spent_here_usd: float = 0.0
    paid_calls: int = 0

    @property
    def spent_usd(self) -> float:
        return self.spent_before_usd + self.spent_here_usd

    def admit(self, projected_call_usd: float) -> None:
        if self.spent_usd + projected_call_usd > self.soft_cap_usd:
            raise BudgetStop(
                f"spent ${self.spent_usd:.4f}; the next call (about ${projected_call_usd:.4f}) would cross the soft cap of ${self.soft_cap_usd:.2f}"
            )

    def charge(self, dollars: float) -> None:
        self.spent_here_usd += dollars
        self.paid_calls += 1


# -- the reading's schema ------------------------------------------------------

MANNER_AXES = ("speed", "effort", "smoothness", "rhythm", "amplitude", "repetition", "precision")
READING_REMOVES = (Remove.ADJACENT, Remove.PROXIMAL, Remove.MEDIAL, Remove.DISTAL)
"""What a request can say about remove. ``coincident`` is in the contract for the grounder's own use; no wording names it."""
REMOVE_GLOSS = {
    "adjacent": "right here, right there, touching, against, up against, adjacent",
    "proximal": "a little, a bit, slightly, near, close, nearby, barely",
    "medial": "UNMARKED (the default for every entry), or halfway, midway, part-way, moderate",
    "distal": "far, far out, way out, out there, distant, all the way, fully, maximum, as far as possible",
}


def reply_schema(inventory: SchemaInventory) -> dict:
    """The strict JSON schema of one reading: enumerations are the inventory's ids."""

    entry_ids = sorted(entry.entry_id for entry in inventory.entries)
    manner_properties = {axis: {"type": "integer", "description": "-2..2 relative to the robot's own neutral; 0 when unmarked"} for axis in MANNER_AXES}
    manner_properties["repetition_count"] = {"type": ["integer", "null"], "description": "an explicit count of times, else null"}
    flexion = {"type": "string", "enum": [item.value for item in Flexion]}
    posture = {
        "type": "object",
        "properties": {
            "group": {"type": "string", "enum": [item.value for item in MemberGroup]},
            "selected_count": {"type": ["integer", "null"]},
            "selected": flexion, "remainder": flexion, "opposing": flexion,
        },
        "required": ["group", "selected_count", "selected", "remainder", "opposing"],
        "additionalProperties": False,
    }
    quantity = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "the distance exactly as the request wrote it"},
            "value": {"type": "number"},
            "unit": {"type": "string", "enum": sorted(UNIT_TO_M)},
        },
        "required": ["text", "value", "unit"],
        "additionalProperties": False,
    }
    segment = {
        "type": "object",
        "properties": {
            "clause": {"type": "string", "description": "the words of the request this segment reads"},
            "entry_id": {"type": "string", "enum": entry_ids},
            "remove": {"type": "string", "enum": [item.value for item in READING_REMOVES]},
            "manner": {"type": "object", "properties": manner_properties, "required": list(manner_properties), "additionalProperties": False},
            "posture": {"anyOf": [posture, {"type": "null"}]},
            "quantities": {"type": "array", "items": quantity},
        },
        "required": ["clause", "entry_id", "remove", "manner", "posture", "quantities"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "segments": {"type": "array", "items": segment},
            "unsupported_reason": {"type": ["string", "null"], "description": "why nothing here names a motion this system knows; null when segments were read"},
        },
        "required": ["segments", "unsupported_reason"],
        "additionalProperties": False,
    }


def system_prompt(inventory: SchemaInventory) -> str:
    """The closed class, described once; the model chooses, it does not invent."""

    lines = [
        "You read a plain-language motion request for a robot whose body you do not know, and you answer only with entries of a closed inventory.",
        "The request is split into segments at sequence markers (then, and then, next, after that, afterwards, finally, and finally, followed by, before returning). Read each segment as exactly one inventory entry, in order. A request without a marker is one segment, whatever else it says.",
        "A motion the request says NOT to do is not a segment. A segment that names no motion in the inventory is omitted; if no segment names one, return no segments and say why in unsupported_reason.",
        "Never name anything outside the inventory. quantities is [] unless the request itself contains a number (digits or a number word) followed by a unit of length (mm, cm, m, inches); then copy that distance exactly as written with its value and unit. Words like 'a little', 'far', 'in front of you' are never quantities.",
        "",
        "Inventory (entry_id: reading):",
    ]
    for entry in inventory.entries:
        schema = entry.schema
        if schema.stative is not None:
            shape = f"stative {schema.stative.value}"
        else:
            shape = f"path {schema.vector.value} {schema.conformation.value}, {schema.contour.value}, deixis {schema.deixis.value}"
        lines.append(f"- {entry.entry_id}: {entry.gloss} [{shape}; figure {entry.figure_role.value}, ground {entry.ground_role.value}]")
    lines += [
        "",
        "remove (how far, in the robot's own terms, never in metres): " + "; ".join(f"{name} = {gloss}" for name, gloss in REMOVE_GLOSS.items()) + ".",
        "The remove is medial whenever the request does not mark how far with one of those words. This holds for every entry: a surface descent or lift-off, a contact approach, a pick-up, a press, a handover, a gaze, a wrist turn, a hold, a beckon or a shoo are all medial unless the words mark otherwise. The entry itself carries the contact or the surface; do not turn that into adjacent. A posture (configure_effector) always has remove " + POSTURE_REMOVE.value + ".",
        "reach_to_edge is only for 'as far as you can', 'as far as possible', 'full extension', 'fully extend', 'maximum reach', 'all the way out', 'to the edge of your reach', and its remove is always distal. 'far', 'far out', 'way out', 'out there' are reach_to_point with remove distal. 'in front of you', 'ahead', 'out', 'across the workspace', 'across the bench', 'over to it' say where, not how far or how large: remove medial, no amplitude. A manner word ('very slowly', 'quickly') says nothing about remove.",
        "manner: ordinals -2..2 relative to the robot's own neutral, 0 when unmarked. Mark an axis only for an explicit word of manner from the lists below; the verb itself carries none ('sweep', 'shoo', 'press', 'lift', 'beckon', 'wave' mark no speed, effort or amplitude), and neither does the place ('across the workspace' is not wide). speed: as fast as you can, flat out = +2; quickly, fast, swiftly, rapidly, briskly = +1; slowly, gently, gradually, carefully, take your time, easy = -1; very slowly, crawl, inch = -2. effort: hard, forcefully, firmly, strongly, vigorously = +1; gently, softly, lightly, delicately, carefully = -1 (so 'gently' and 'carefully' set both speed -1 and effort -1). smoothness: smoothly, fluidly, evenly, steadily = +1; jerkily, abruptly, sharply, staccato = -1. amplitude: huge, enormous, sweeping, expansive = +2; wide, big, large, broad, generous = +1; small, tight, narrow, slight, subtle, little (as in 'a little wave') = -1; tiny, minute, barely = -2. 'a little', 'a bit', 'slightly' said of how far to go are the remove proximal and mark no amplitude. precision: precisely, exactly, accurately = +1; roughly, approximately, about, casually = -1. rhythm and repetition stay 0 unless the request marks them. repetition_count is an explicit number of times (once = 1, twice = 2, three times = 3, a couple of times = 2, a few times = 3) else null; a number that is part of a distance or a posture ('three fingers') is not a count.",
        "Statives are motions this system knows and are never 'unsupported': 'turn the wrist', 'rotate the tool', 'orient' is orient_effector; 'hold still', 'stay', 'freeze' is hold_still; 'thumbs up', 'fist', 'open your hand', 'peace sign', 'point with one finger', 'one finger up', 'three fingers up' is configure_effector with a posture. 'point with a finger' or 'point with one finger' is the posture, never reach_to_point; only 'point to' or 'point at' a place with the arm is reach_to_point.",
        "posture: only for configure_effector, and null for every other entry. selected_count members of the group take `selected`, the rest of the group take `remainder`, the opposing side takes `opposing`; `selected` is 'extended' unless the request says the selected members are flexed. Peace sign / two fingers: selected_count 2, selected extended, remainder flexed, opposing flexed. One finger / point with a finger: 1, extended, remainder flexed, opposing flexed. Three fingers: 3, extended, remainder flexed, opposing flexed. Thumbs up: 0, extended, remainder flexed, opposing extended. Fist / close the hand / clench: 0, extended, remainder flexed, opposing flexed. Open hand / spread the fingers / flat hand / high five: 0, extended, remainder extended, opposing extended.",
        "Readings that matter: 'reach', 'extend', 'stretch', 'move', 'go', 'point', 'put', 'place' toward a place is reach_to_point; 'wave', 'shake', 'wiggle', 'wag' is oscillate_about_point; 'back and forth', 'side to side', 'to and fro' along a line is oscillate_along_line; 'sweep', 'scan', 'across', 'traverse' is traverse_line, over an arc traverse_line_arced; 'circle', 'orbit', 'loop around' is circle_axis; 'come back', 'return', 'retract', 'withdraw', 'pull back', 'go home' is retract_from_point; 'lift', 'raise', 'come up', 'go up', 'off the table' is retract_from_surface; 'lower', 'descend', 'go down', 'come down', 'onto the table' is descend_to_surface, over an arc arc_to_surface; 'pick up', 'grab', 'grasp', 'take hold' is transport_object; 'touch it', 'close in on it', 'until you touch' is approach_to_contact; 'beckon', 'come here', 'draw it in', 'bring it toward you' is draw_hither; 'shoo', 'push it away', 'wave it off', 'send it away from you' is push_thither; 'move away', 'back off', 'back away' with no deictic centre is move_away; 'approach', 'move toward', 'closer' is move_toward; 'swing', 'arc over', 'over the top' to a point is swing_to_point; 'reach into', 'inside the' is enter_volume; 'pass through', 'via the waypoint' is pass_through_point; 'hold still', 'stay', 'freeze', 'do not move' is hold_still; 'turn the wrist', 'orient', 'rotate the tool' is orient_effector; 'look at', 'aim at', 'aim the camera', 'track' is track_with_gaze; 'hand it over', 'pass it across', 'to the other hand' is hand_across; 'press', 'push down on', 'bear on' is press.",
        "",
        "Examples (conventions only, not the request). 'stretch your arm out' -> reach_to_point, medial, all manner 0, posture null, quantities []. 'give it a gentle shake' -> oscillate_about_point, medial, speed -1, effort -1. 'put the tool down on the bench, then back off a bit' -> descend_to_surface medial; move_away proximal. 'slide 12 cm to the right' -> reach_to_point, medial, quantities [{text '12 cm', value 12, unit cm}].",
    ]
    return "\n".join(lines)


# -- the planner ---------------------------------------------------------------


@dataclass
class PlannerSpend:
    """What this planner instance has spent and read, for the evidence."""

    model: str
    transport: str
    calls: int = 0
    cached_hits: int = 0
    paid_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dollars: float = 0.0
    fallback_used: bool = False
    stopped: str | None = None
    """``billing``, ``budget`` or ``unavailable`` once the planner stopped spending."""


class ModelSchemaPlanner:
    """Plan through a model, validate through the contract, spend through a budget."""

    planner_id = "model-schema-planner-v1"

    def __init__(
        self,
        inventory: SchemaInventory,
        *,
        model: str,
        transport: Transport,
        cache: ResponseCache | None = None,
        log: CallLog | None = None,
        budget: Budget | None = None,
        purpose: str = "planning",
        fallback: tuple[str, Transport] | None = None,
    ) -> None:
        if base_model(model) not in PUBLISHED_RATES_USD_PER_MTOK:
            raise KeyError(f"no published rate recorded for {model!r}; record one before calling it")
        self.inventory = inventory
        self.model = model
        self.transport = transport
        self.cache = cache
        self.log = log if log is not None else CallLog(None)
        self.budget = budget if budget is not None else Budget()
        self.purpose = purpose
        self.fallback = fallback
        self.system = system_prompt(inventory)
        self.schema = reply_schema(inventory)
        self.spend = PlannerSpend(model=model, transport=transport.name)
        self.last_trace: tuple[PlannerTrace, ...] = ()
        self.last_clauses: tuple[str, ...] = ()
        self.last_reply: dict | None = None
        self._entries = {entry.entry_id: entry for entry in inventory.entries}

    # -- the SchemaPlanner protocol -------------------------------------------

    def plan(self, prompt: str, *, afforded: tuple[SchemaEntry, ...]) -> MotionSchemaProgramV1:
        return self.plan_request(prompt, afforded=afforded).program

    def plan_request(self, prompt: str, *, afforded: tuple[SchemaEntry, ...]) -> PlannedRequestV1:
        text = prompt.strip()
        if not text:
            raise RigbyGeneralError(GeneralFailureCode.INVALID_CONTRACT, "the prompt is empty")
        decision = screen(text)
        if not decision.allowed:
            raise RigbyGeneralError(
                GeneralFailureCode.UNSUPPORTED_MORPHOLOGY, decision.reason,
                details={**decision.as_details(), "prompt": text[:200]},
            )
        reply, cached, model_used, transport_used = self.reading(text)
        return self._program_from(text, reply, afforded=afforded, cached=cached, model=model_used, transport=transport_used)

    # -- the call ---------------------------------------------------------------

    def reading(self, prompt: str) -> tuple[dict, bool, str, str]:
        """The model's reply for this request: from the cache when it has one."""

        user = f"Request: {prompt}"
        key = ResponseCache.key(model=self.model, system=self.system, schema=self.schema, prompt=prompt)
        self.spend.calls += 1
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.spend.cached_hits += 1
                self.log.record(self._row(prompt, key, hit["model"], hit["transport"], hit["usage"]["prompt_tokens"], hit["usage"]["completion_tokens"], cached=True, paid=False, cost=0.0, estimated=bool(hit["usage"].get("estimated", False))))
                self.last_reply = hit["reply"]
                return hit["reply"], True, hit["model"], hit["transport"]
        if self.spend.stopped is not None:
            raise ModelUnavailable(f"the planner stopped spending: {self.spend.stopped}", reason=self.spend.stopped, transport=self.transport.name)
        model, transport = self.model, self.transport
        if transport.paid:
            projected = cost_usd(model, estimate_tokens(self.system) + estimate_tokens(json.dumps(self.schema)) + estimate_tokens(user), 400 + REASONING_ALLOWANCE_TOKENS.get(base_model(model), 0))
            try:
                self.budget.admit(projected)
            except BudgetStop:
                self.spend.stopped = "budget"
                raise
        try:
            reply = transport.complete(model=model, system=self.system, user=user, schema=self.schema, purpose=self.purpose)
        except ModelUnavailable as error:
            if error.reason == "billing":
                self.spend.stopped = "billing"
            if self.fallback is None:
                raise
            fallback_model, fallback_transport = self.fallback
            self.spend.fallback_used = True
            model, transport = fallback_model, fallback_transport
            reply = transport.complete(model=model, system=self.system, user=user, schema=self.schema, purpose=self.purpose + ":fallback")
        dollars = cost_usd(model, reply.prompt_tokens, reply.completion_tokens) if transport.paid else 0.0
        if transport.paid:
            self.budget.charge(dollars)
            self.spend.paid_calls += 1
        self.spend.prompt_tokens += reply.prompt_tokens
        self.spend.completion_tokens += reply.completion_tokens
        self.spend.dollars += dollars
        try:
            parsed = json.loads(reply.text)
        except json.JSONDecodeError as error:
            parsed = {"segments": [], "unsupported_reason": f"reply was not JSON: {reply.text[:120]}", "_invalid": True}
            _ = error
        self.log.record(self._row(prompt, key, reply.model, transport.name, reply.prompt_tokens, reply.completion_tokens, cached=False, paid=transport.paid, cost=dollars, estimated=reply.estimated))
        if self.cache is not None and not parsed.get("_invalid") and transport.name != "mock":
            self.cache.put(key, {
                "key": key, "model": reply.model, "requested_model": model, "transport": transport.name, "prompt": prompt,
                "system_sha256": hashlib.sha256(self.system.encode("utf-8")).hexdigest(),
                "schema_sha256": hashlib.sha256(json.dumps(self.schema, sort_keys=True).encode("utf-8")).hexdigest(),
                "reply": parsed, "usage": {"prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens, "estimated": reply.estimated},
                "cost_usd": dollars, "purpose": self.purpose, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        self.last_reply = parsed
        return parsed, False, reply.model, transport.name

    def _row(self, prompt: str, key: str, model: str, transport: str, prompt_tokens: int, completion_tokens: int, *, cached: bool, paid: bool, cost: float, estimated: bool) -> dict:
        return {
            "at_utc": datetime.now(timezone.utc).isoformat(), "model": model, "transport": transport, "purpose": self.purpose,
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "cache_key": key,
            "prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens), "tokens_estimated": estimated,
            "cached": cached, "paid": paid, "cost_usd": round(cost, 8), "rates_recorded_on": RATES_RECORDED_ON,
            "rates_usd_per_mtok": list(PUBLISHED_RATES_USD_PER_MTOK.get(base_model(model), PUBLISHED_RATES_USD_PER_MTOK.get(base_model(self.model)))),
        }

    # -- validation -------------------------------------------------------------

    def _program_from(self, text: str, reply: dict, *, afforded: tuple[SchemaEntry, ...], cached: bool, model: str, transport: str) -> PlannedRequestV1:
        if reply.get("_invalid") or not isinstance(reply.get("segments"), list):
            raise RigbyGeneralError(GeneralFailureCode.INVALID_CONTRACT, "the model's reply was not a reading", details={"reason": "reply_invalid", "prompt": text[:200]})
        available = {entry.entry_id: entry for entry in afforded}
        lowered = text.lower()
        segments: list[SegmentV1] = []
        traces: list[PlannerTrace] = []
        quantities: list[RequestedQuantityV1] = []
        clauses: list[str] = []
        for index, item in enumerate(reply["segments"][:8]):
            entry_id = str(item.get("entry_id", ""))
            entry = self._entries.get(entry_id)
            if entry is None:
                raise RigbyGeneralError(GeneralFailureCode.INVENTED_BINDING, f"the model named a schema the inventory does not have: {entry_id!r}", details={"entry_id": entry_id, "prompt": text[:200]})
            if entry_id not in available:
                raise RigbyGeneralError(GeneralFailureCode.UNAFFORDED_SCHEMA, f"this robot has no certified {entry_id!r} primitive", details={"entry_id": entry_id, "clause": str(item.get("clause", ""))[:200]})
            try:
                remove = Remove(str(item.get("remove", "")))
                manner_payload = dict(item.get("manner") or {})
                count = manner_payload.pop("repetition_count", None)
                manner = MannerV1(**{axis: int(manner_payload.get(axis, 0)) for axis in MANNER_AXES}, repetition_count=count if count is None else int(count))
                posture = None
                if entry.takes_posture and item.get("posture"):
                    posture_payload = dict(item["posture"])
                    selected_count = posture_payload.get("selected_count")
                    posture = PostureV1(
                        group=MemberGroup(posture_payload.get("group", MemberGroup.OPPOSED.value)),
                        selected_count=selected_count,
                        # With no member selected the selected pole names nothing; it is
                        # normalised to the contract's default so two readings of the
                        # same fist hash alike instead of differing on an empty field.
                        selected=Flexion.EXTENDED if selected_count == 0 else Flexion(posture_payload.get("selected", Flexion.EXTENDED.value)),
                        remainder=Flexion(posture_payload.get("remainder", Flexion.FLEXED.value)),
                        opposing=Flexion(posture_payload.get("opposing", Flexion.FREE.value)),
                    )
            except (ValueError, TypeError) as error:
                raise RigbyGeneralError(GeneralFailureCode.INVALID_CONTRACT, f"the model's reading of segment {index} is outside the contract: {error}", details={"reason": "reply_invalid", "segment": index, "prompt": text[:200]}) from error
            if posture is not None:
                remove = POSTURE_REMOVE
            segment_id = f"s{index}"
            for raw in item.get("quantities") or ():
                quantity_text = str(raw.get("text", "")).strip()
                unit = str(raw.get("unit", "")).lower()
                if not quantity_text or quantity_text.lower() not in lowered or unit not in UNIT_TO_M:
                    raise RigbyGeneralError(
                        GeneralFailureCode.PROHIBITED_SUBSTITUTION,
                        f"the model reported a distance the request never stated: {quantity_text!r}",
                        details={"segment": segment_id, "text": quantity_text, "prompt": text[:200]},
                    )
                try:
                    value = float(raw.get("value"))
                except (TypeError, ValueError) as error:
                    raise RigbyGeneralError(GeneralFailureCode.INVALID_CONTRACT, "a stated quantity carried no number", details={"reason": "reply_invalid", "prompt": text[:200]}) from error
                stated = _stated_value(quantity_text)
                if stated is None or not math.isclose(stated, value, rel_tol=1e-6, abs_tol=1e-9):
                    raise RigbyGeneralError(
                        GeneralFailureCode.PROHIBITED_SUBSTITUTION,
                        f"the model changed a stated distance: {quantity_text!r} read as {value}",
                        details={"segment": segment_id, "text": quantity_text, "value": value, "prompt": text[:200]},
                    )
                quantities.append(RequestedQuantityV1(segment_id=segment_id, kind=QuantityKind.DISTANCE, text=quantity_text, value=value * UNIT_TO_M[unit], unit="m"))
            segments.append(SegmentV1(
                segment_id=segment_id, motion_schema=entry.schema,
                figure=RoleBindingV1(role=entry.figure_role), ground=RoleBindingV1(role=entry.ground_role),
                region=RegionV1(remove=remove, dimensionality=Dimensionality.POINT),
                frame=_frame_for(entry), manner=manner, posture=posture, boundary=_boundary_for(entry),
            ))
            clause = str(item.get("clause", ""))[:200]
            clauses.append(clause)
            traces.append(PlannerTrace(clause=clause, entry_id=entry_id, remove=remove.value,
                                       manner={axis: getattr(manner, axis) for axis in ("speed", "effort", "amplitude", "repetition") if getattr(manner, axis)}))
        if not segments:
            raise RigbyGeneralError(
                GeneralFailureCode.UNAFFORDED_SCHEMA, "nothing in that request names a motion this system knows",
                details={"prompt": text[:200], "model_reason": str(reply.get("unsupported_reason") or "")[:200]},
            )
        links = tuple(SegmentLinkV1(from_segment=a.segment_id, to_segment=b.segment_id, relation=Concurrency.SEQUENCE) for a, b in zip(segments, segments[1:]))
        self.last_trace = tuple(traces)
        self.last_clauses = tuple(clauses)
        program = MotionSchemaProgramV1(
            program_id=f"model-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
            source_text=text, segments=tuple(segments), links=links,
        )
        return PlannedRequestV1(program=program, quantities=tuple(quantities), planner_id=self.planner_id, model=model, cached=cached)


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "half a": 0.5,
}


def _stated_value(text: str) -> float | None:
    """The number a quantity's own words carry, so a model cannot relabel '5 cm' as 50."""

    match = re.match(r"^\s*(\d+(?:\.\d+)?|[a-z ]+?)\s*(?:mm|millimet\w*|cm|centimet\w*|m|metres?|meters?|in|inch|inches)\b", text.strip().lower())
    if not match:
        return None
    token = match.group(1).strip()
    if re.match(r"^\d", token):
        return float(token)
    return float(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else None


__all__ = [
    "Budget", "BudgetStop", "CallLog", "GeminiTransport", "MockTransport", "ModelReply", "ModelSchemaPlanner",
    "ModelUnavailable", "OpenAITransport", "PUBLISHED_RATES_USD_PER_MTOK", "PlannerSpend", "RATES_RECORDED_ON",
    "ResponseCache", "cost_usd", "estimate_tokens", "reply_schema", "system_prompt",
]
