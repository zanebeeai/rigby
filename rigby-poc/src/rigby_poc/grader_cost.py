"""Token cost per grader, and how much of it is actually attributable.

A grader-cost figure has three populations in it and they are invisible in a
total.  A dispatch that *raises* carries no usage anywhere — the provider never
answered — while a response that *arrives* and then fails validation does. So
tokens go unrecorded in three distinct ways:

1. **Escalation after a bad parse.** The predicate fires on `output_parsed is
   None`, so the primary answered and its tokens exist.  Attributable.
2. **Escalation after a network error.** Nothing answered.  Not attributable.
3. **Transient retries.** `_parse_response_with_retry` retries connection,
   timeout and server errors twice; each is a separate dispatch with its own
   span, and only the one that eventually succeeds carries usage.  This one can
   happen on a call that ultimately succeeded and looks entirely healthy.

The undercount is therefore not noise.  It is concentrated in exactly the calls
that failed hardest, and averaging over it reports a cost that is systematically
low wherever the system was struggling most.

So cost is reported with **tokens attributed / dispatches made** beside it.  Near
1.0 means the figure is sound; a drop means the undercount is concentrated
somewhere and the report must say where rather than average over it.

Instrumentation for all three arrived in PR 01b; this reads it and adds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from evals.calibration_stats import proportion


@dataclass(frozen=True)
class CallCost:
    """One routed model call: what it spent and what could be attributed."""

    label: str
    #: Every HTTP request the call actually made, from `routing.dispatch_count`.
    dispatches: int
    #: Attempt records that carry a usage block.
    attributed_attempts: int
    input_tokens: int
    output_tokens: int
    transient_retries: int

    @property
    def unattributed_dispatches(self) -> int:
        return max(0, self.dispatches - self.attributed_attempts)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _usage_tokens(attempt: Mapping[str, Any]) -> tuple[int, int] | None:
    usage = attempt.get("usage")
    if not isinstance(usage, Mapping) or not usage:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return input_tokens, output_tokens


def call_cost(label: str, record: Mapping[str, Any]) -> CallCost:
    """Read one judge or grader record. Pure; no model, no transcript reader."""
    routing = record.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError(f"{label}: record carries no routing block")
    dispatches = routing.get("dispatch_count")
    if not isinstance(dispatches, int):
        raise ValueError(f"{label}: routing carries no dispatch_count")
    attempts = record.get("attempts") or []
    attributed = 0
    input_tokens = output_tokens = 0
    for attempt in attempts:
        tokens = _usage_tokens(attempt) if isinstance(attempt, Mapping) else None
        if tokens is None:
            continue
        attributed += 1
        input_tokens += tokens[0]
        output_tokens += tokens[1]
    if attributed > dispatches:
        # More usage blocks than HTTP requests is impossible from the provider's
        # side, so it is an instrumentation fault rather than provider behaviour.
        # Raising makes the report a self-check on its own instrumentation.
        raise ValueError(
            f"{label}: {attributed} attributed attempts exceed {dispatches} dispatches"
        )
    retries = routing.get("transient_retry_count")
    return CallCost(
        label=label,
        dispatches=dispatches,
        attributed_attempts=attributed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        transient_retries=retries if isinstance(retries, int) else 0,
    )


def cost_report(costs: Sequence[CallCost]) -> dict[str, Any]:
    """Cost with its attribution ratio, its n, and where the shortfall sits.

    The ratio is the number that licenses the cost figure. It is reported as a
    proportion with an interval rather than a bare fraction, so it satisfies the
    same standard every other published rate in the repo has to meet.
    """
    if not costs:
        raise ValueError("cost_report needs at least one call")
    dispatches = sum(item.dispatches for item in costs)
    attributed = sum(item.attributed_attempts for item in costs)
    shortfall = sorted(
        (item.label for item in costs if item.unattributed_dispatches),
    )
    return {
        "n_calls": len(costs),
        "dispatches": dispatches,
        "attributed_attempts": attributed,
        "attribution_ratio": proportion(attributed, dispatches),
        "input_tokens": sum(item.input_tokens for item in costs),
        "output_tokens": sum(item.output_tokens for item in costs),
        "total_tokens": sum(item.total_tokens for item in costs),
        "transient_retries": sum(item.transient_retries for item in costs),
        #: Named, not counted: a ratio below 1.0 means the cost figure is low by
        #: an unknown amount concentrated in these calls.
        "calls_with_unattributed_dispatches": shortfall,
        "cost_figure_is_complete": attributed == dispatches,
    }


def report_records(records: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    return cost_report([call_cost(label, record) for label, record in records])
