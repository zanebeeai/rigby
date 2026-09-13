"""Explicit quantities a request states, kept beside the metric-free program.

The schema IR has no place for a number, on purpose: a planner that cannot
write "0.42 m" cannot invent a reach. But a person who says "five
centimetres" has stated one, and erasing it would be the other failure.
So a stated quantity travels beside the program, not inside it: the
segment it belongs to, the text exactly as written, its value in base
units, and its provenance -- always the user, never a model. A model
planner may only copy a quantity that appears verbatim in the request;
anything else is a prohibited substitution and is refused.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import Field
from rigby_core.contracts import Contract

from ..schema.program import MotionSchemaProgramV1


class QuantityKind(StrEnum):
    DISTANCE = "distance"


class RequestedQuantityV1(Contract):
    segment_id: str = Field(min_length=1)
    kind: QuantityKind
    text: str = Field(min_length=1)
    """The quantity exactly as the request wrote it."""
    value: float = Field(gt=0.0)
    unit: str = Field(min_length=1)
    """The base unit the value is in: metres for a distance."""
    provenance: Literal["user_stated"] = "user_stated"


class PlannedRequestV1(Contract):
    """A program and the quantities the request stated beside it."""

    program: MotionSchemaProgramV1
    quantities: tuple[RequestedQuantityV1, ...] = ()
    planner_id: str = Field(min_length=1)
    model: str | None = None
    cached: bool = False

    def distances_m(self) -> dict[str, float]:
        return {q.segment_id: q.value for q in self.quantities if q.kind is QuantityKind.DISTANCE}


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "half a": 0.5, "a": 1, "an": 1,
}
_UNITS_TO_M = {
    "mm": 0.001, "millimetre": 0.001, "millimetres": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimetre": 0.01, "centimetres": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
    "inch": 0.0254, "inches": 0.0254, "in": 0.0254,
}
_QUANTITY = re.compile(
    r"\b(?P<number>\d+(?:\.\d+)?|" + "|".join(sorted((re.escape(w) for w in _NUMBER_WORDS), key=len, reverse=True)) + r")\s*"
    r"(?P<unit>millimetres?|millimeters?|mm|centimetres?|centimeters?|cm|metres?|meters?|m|inches|inch|in)\b",
    re.IGNORECASE,
)


def extract_quantities(clauses: list[str]) -> tuple[RequestedQuantityV1, ...]:
    """Every stated distance in each clause, attributed to that clause's segment."""

    found: list[RequestedQuantityV1] = []
    for index, clause in enumerate(clauses):
        for match in _QUANTITY.finditer(clause):
            number = match.group("number").lower()
            unit = match.group("unit").lower()
            if unit == "in" and not re.search(r"\b\d+(?:\.\d+)?\s*in\b", match.group(0).lower()):
                continue  # "in" the word, not the unit
            if unit == "m" and number in ("a", "an"):
                continue  # "a metre" is not a stated quantity in a motion request
            value = float(number) if re.match(r"^\d", number) else float(_NUMBER_WORDS[number])
            if number in ("a", "an"):
                continue
            found.append(RequestedQuantityV1(segment_id=f"s{index}", kind=QuantityKind.DISTANCE, text=match.group(0), value=value * _UNITS_TO_M[unit], unit="m"))
    return tuple(found)


def quantities_verbatim(quantities: tuple[RequestedQuantityV1, ...], prompt: str) -> tuple[RequestedQuantityV1, ...]:
    """The quantities whose text does not appear in the prompt: a model that
    reports a distance the request never stated has substituted one."""

    lowered = prompt.lower()
    return tuple(q for q in quantities if q.text.lower() not in lowered)
