"""Acquire a missing motor skill inside a bounded search, and say what kind of thing was found.

An acquisition problem is frozen before any search runs: the effect wanted
(a skill of the library with its arguments), the body it is wanted on, the
change to the world that makes the existing implementation fail, the
controller family the answer may come from, the parameters of that family
with their bounds and defaults, the development draws the search may look
at, the held-out draws it may not, the acceptance threshold, and the
ceiling in attempts and simulator minutes. The search proposes a parameter
vector, the body's runtime runs the development episodes with it, and the
search is told how many certified. It stops when a vector certifies every
development episode and its confirmation, or when the ceiling is reached.

What was found is then classified, because the catalog separates three
things a library can grow by. An *instantiation* is the existing
implementation at its defaults passing the acceptance test: no search was
needed and none is claimed. A *composition* is a new arrangement of leaves
that already exist. A *discovery* is a parameter vector the existing
implementation did not have -- new contact timing, trajectory or controller
parameters -- found by the search and passing the acceptance test. A search
that exhausts its ceiling is a reported result with a hypothesis about the
limiting capability, and the skill stays unpromoted; nothing here rebrands
an existing motion as discovery.

The optimizer is a seeded (1+lambda) evolution strategy over the bounded
parameters in normalized coordinates, log-scaled where a parameter spans
orders of magnitude, with a step size that contracts when a generation
brings no improvement. It is deterministic given its seed and the outcomes
it is told, so the whole search replays from its provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from ..contracts import Contract
from ..hashing import content_hash


class ProblemKind(StrEnum):
    INSTANTIATION = "instantiation"
    COMPOSITION = "composition"
    DISCOVERY = "discovery"


class AcquisitionStatus(StrEnum):
    ACQUIRED = "acquired"
    INSTANTIATED = "instantiated"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ACCEPTANCE_FAILED = "acceptance_failed"


class ParameterSpecV1(Contract):
    name: str = Field(min_length=1)
    low: float
    high: float
    default: float
    units: str = Field(min_length=1)
    scale: Literal["linear", "log"] = "linear"
    description: str = ""

    @model_validator(mode="after")
    def ordered_and_contains_default(self) -> Self:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.high > self.low):
            raise ValueError(f"{self.name}: bounds must be finite and increasing")
        if not (self.low <= self.default <= self.high):
            raise ValueError(f"{self.name}: the default lies outside the bounds")
        if self.scale == "log" and self.low <= 0.0:
            raise ValueError(f"{self.name}: a log-scaled parameter needs a positive lower bound")
        return self


class EffectV1(Contract):
    skill_id: str = Field(min_length=1)
    arguments: dict[str, str] = {}


class CeilingV1(Contract):
    attempts: int = Field(ge=1)
    worker_minutes: float = Field(gt=0.0)


class AcceptanceV1(Contract):
    set_id: str = Field(min_length=1)
    trials: int = Field(ge=1)
    threshold: int = Field(ge=1)

    @model_validator(mode="after")
    def threshold_within(self) -> Self:
        if self.threshold > self.trials:
            raise ValueError("the threshold cannot exceed the trials")
        return self


class AcquisitionProblemV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    problem_id: str = Field(min_length=1)
    body: str = Field(min_length=1)
    body_family: str = Field(min_length=1)
    effect: EffectV1
    world_change: str = Field(min_length=1)
    """What makes the existing implementation fail, in words; the campaign carries the actual world."""
    hypothesis: ProblemKind
    """What kind of answer is expected; the outcome records what kind was found."""
    controller_family: str = Field(min_length=1)
    parameters: tuple[ParameterSpecV1, ...] = Field(min_length=1)
    development_seeds: tuple[int, ...] = Field(min_length=1)
    confirmation_seeds: tuple[int, ...] = ()
    acceptance: AcceptanceV1
    ceiling: CeilingV1
    notes: str = ""

    @model_validator(mode="after")
    def distinct(self) -> Self:
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be distinct")
        if set(self.development_seeds) & set(self.confirmation_seeds):
            raise ValueError("confirmation seeds must be distinct from development seeds")
        return self

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.parameters}


class EpisodeOutcomeV1(Contract):
    seed: int
    certified: bool
    failed_gate: str | None = None
    physics_s: float = Field(ge=0.0)
    wall_s: float = Field(ge=0.0)
    measurements: dict[str, float] = {}


class AttemptV1(Contract):
    index: int = Field(ge=0)
    parameters: dict[str, float]
    episodes: tuple[EpisodeOutcomeV1, ...]
    certified: int = Field(ge=0)
    of: int = Field(ge=1)
    confirmation: tuple[EpisodeOutcomeV1, ...] = ()
    improved: bool = False
    sigma: float = Field(ge=0.0)
    physics_s: float = Field(ge=0.0)
    wall_s: float = Field(ge=0.0)

    @property
    def score(self) -> float:
        return self.certified + (sum(1 for e in self.confirmation if e.certified) if self.confirmation else 0.0)


class SearchProvenanceV1(Contract):
    algorithm: Literal["one-plus-lambda-es"] = "one-plus-lambda-es"
    seed: int
    offspring: int = Field(ge=1)
    sigma0: float = Field(gt=0.0)
    contraction: float = Field(gt=0.0, lt=1.0)
    minimum_sigma: float = Field(gt=0.0)
    bounds_sha256: str = Field(min_length=64, max_length=64)
    note: str = "deterministic given the seed and the outcomes observed; the whole search replays from the attempts"


class AcquisitionOutcomeV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    problem_id: str
    status: AcquisitionStatus
    kind: ProblemKind | None
    """What was found: instantiation when the defaults passed, discovery when a searched vector did; None when nothing passed."""
    attempts: int = Field(ge=0)
    physics_minutes: float = Field(ge=0.0)
    wall_minutes: float = Field(ge=0.0)
    ceiling_hit: str | None = None
    best_attempt: int | None = None
    best_parameters: dict[str, float] = {}
    parameters_changed: tuple[str, ...] = ()
    """Which parameters the best vector moved from their defaults by more than five percent of the range."""
    limiting_capability: str = ""
    """For an exhausted search: the hypothesis, in words, about what the family cannot do."""
    provenance: SearchProvenanceV1 | None = None
    attempts_sha256: str = Field(min_length=64, max_length=64)


# -- the search ---------------------------------------------------------------------


def _to_unit(spec: ParameterSpecV1, value: float) -> float:
    if spec.scale == "log":
        return (math.log(value) - math.log(spec.low)) / (math.log(spec.high) - math.log(spec.low))
    return (value - spec.low) / (spec.high - spec.low)


def _from_unit(spec: ParameterSpecV1, unit: float) -> float:
    unit = min(1.0, max(0.0, unit))
    if spec.scale == "log":
        return float(math.exp(math.log(spec.low) + unit * (math.log(spec.high) - math.log(spec.low))))
    return float(spec.low + unit * (spec.high - spec.low))


@dataclass
class EvolutionSearch:
    """A (1+lambda) evolution strategy in the unit cube of the parameters."""

    problem: AcquisitionProblemV1
    seed: int = 0
    offspring: int = 4
    sigma0: float = 0.25
    contraction: float = 0.85
    minimum_sigma: float = 0.02
    rng: np.random.Generator = field(init=False)
    incumbent: dict[str, float] = field(init=False)
    incumbent_score: float = field(default=-1.0, init=False)
    sigma: float = field(init=False)
    generation_count: int = field(default=0, init=False)
    generation_improved: bool = field(default=False, init=False)
    improved_ever: bool = field(default=False, init=False)
    """Whether any searched vector has ever beaten the defaults; until one has,
    the objective is a plateau and narrowing the step would only slow the walk."""
    proposals: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.incumbent = self.problem.defaults()
        self.sigma = self.sigma0

    def provenance(self) -> SearchProvenanceV1:
        return SearchProvenanceV1(seed=self.seed, offspring=self.offspring, sigma0=self.sigma0, contraction=self.contraction, minimum_sigma=self.minimum_sigma,
                                  bounds_sha256=content_hash([p.model_dump(mode="json") for p in self.problem.parameters]))

    def propose(self) -> dict[str, float]:
        """The defaults first (attempt 0 asks whether search is needed at all),
        then Gaussian steps around the incumbent, clipped to the bounds."""

        self.proposals += 1
        if self.proposals == 1:
            return dict(self.incumbent)
        proposal = {}
        for spec in self.problem.parameters:
            unit = _to_unit(spec, self.incumbent[spec.name]) + float(self.rng.normal(0.0, self.sigma))
            proposal[spec.name] = _from_unit(spec, unit)
        return proposal

    def observe(self, parameters: dict[str, float], score: float) -> bool:
        """Tell the search what a proposal scored; returns whether it became the incumbent."""

        improved = score > self.incumbent_score
        if improved:
            self.incumbent = dict(parameters)
            self.incumbent_score = score
            if self.proposals > 1:
                self.improved_ever = True
        if self.proposals == 1:
            return improved  # the defaults set the baseline; they are not a generation
        self.generation_improved = self.generation_improved or improved
        self.generation_count += 1
        if self.generation_count >= self.offspring:
            # Once something has beaten the defaults, a generation that brings
            # nothing narrows the search around the incumbent; on the plateau
            # before that, the step stays wide.
            if self.improved_ever and not self.generation_improved:
                self.sigma = max(self.minimum_sigma, self.sigma * self.contraction)
            self.generation_count = 0
            self.generation_improved = False
        return improved


def classify(problem: AcquisitionProblemV1, best_attempt: AttemptV1 | None, *, accepted: bool) -> tuple[AcquisitionStatus, ProblemKind | None, tuple[str, ...]]:
    """What kind of thing the search found, from what it did."""

    if best_attempt is None:
        return AcquisitionStatus.BUDGET_EXHAUSTED, None, ()
    changed = tuple(spec.name for spec in problem.parameters
                    if abs(_to_unit(spec, best_attempt.parameters[spec.name]) - _to_unit(spec, spec.default)) > 0.05)
    if not accepted:
        return AcquisitionStatus.ACCEPTANCE_FAILED, None, changed
    if best_attempt.index == 0 and not changed:
        return AcquisitionStatus.INSTANTIATED, ProblemKind.INSTANTIATION, ()
    return AcquisitionStatus.ACQUIRED, ProblemKind.DISCOVERY, changed


def attempts_digest(attempts: list[AttemptV1] | tuple[AttemptV1, ...]) -> str:
    return content_hash([a.model_dump(mode="json") for a in attempts])


__all__ = [
    "AcceptanceV1", "AcquisitionOutcomeV1", "AcquisitionProblemV1", "AcquisitionStatus", "AttemptV1", "CeilingV1", "EffectV1", "EpisodeOutcomeV1", "EvolutionSearch",
    "ParameterSpecV1", "ProblemKind", "SearchProvenanceV1", "attempts_digest", "classify",
]
