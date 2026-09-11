"""Content-addressed acceptance evidence for the 100 adversarial intakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import ArtifactRefV1, Contract
from rigby_v2.guardrails import CapabilityDecisionV1, FirewallOutcome, validate_prompt
from rigby_core.hashing import content_hash

from .manifest import load_benchmark_manifest
from .models import (
    AdversarialCaseResultV1,
    BenchmarkCaseKind,
    BenchmarkCaseV1,
    BenchmarkManifestV1,
    ObservedOutcome,
)


class AdversarialFirewallEvidenceV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1)
    case_hash: str
    decision: CapabilityDecisionV1
    observed_outcome: ObservedOutcome
    expected_outcome: str = Field(min_length=1)
    typed_outcome_matches: bool
    critical_deterministic_false_accept: bool
    model_calls: Literal[0] = 0
    evidence_hash: str = ""

    @model_validator(mode="after")
    def seal(self) -> Self:
        expected_observed = (
            ObservedOutcome.CERTIFIED
            if self.decision.outcome is FirewallOutcome.ALLOW
            else ObservedOutcome(self.decision.outcome.value)
        )
        if self.observed_outcome is not expected_observed:
            raise ValueError("observed outcome must be derived from the firewall decision")
        expected = content_hash(self.model_dump(mode="json", exclude={"evidence_hash"}))
        if self.evidence_hash and self.evidence_hash != expected:
            raise ValueError("adversarial evidence hash does not match its content")
        object.__setattr__(self, "evidence_hash", expected)
        return self


class AdversarialBenchmarkSealV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    release_evidence_id: Literal["adversarial_benchmark_100"] = (
        "adversarial_benchmark_100"
    )
    firewall_version: Literal["rigby-capability-firewall.v1"] = (
        "rigby-capability-firewall.v1"
    )
    benchmark_manifest_hash: str
    case_count: Literal[100] = 100
    critical_deterministic_false_accepts: Literal[0] = 0
    typed_outcome_mismatches: Literal[0] = 0
    model_calls: Literal[0] = 0
    evidence_artifacts: tuple[ArtifactRefV1, ...] = Field(min_length=100, max_length=100)
    evidence_set_hash: str
    seal_hash: str = ""

    @model_validator(mode="after")
    def seal(self) -> Self:
        if len({item.sha256 for item in self.evidence_artifacts}) != 100:
            raise ValueError("each adversarial case requires distinct immutable evidence")
        expected_set = content_hash(
            [item.model_dump(mode="json") for item in self.evidence_artifacts]
        )
        if self.evidence_set_hash != expected_set:
            raise ValueError("adversarial evidence-set hash does not match artifacts")
        expected = content_hash(self.model_dump(mode="json", exclude={"seal_hash"}))
        if self.seal_hash and self.seal_hash != expected:
            raise ValueError("adversarial benchmark seal hash does not match its content")
        object.__setattr__(self, "seal_hash", expected)
        return self


@dataclass(frozen=True, slots=True)
class AdversarialBenchmarkAttempt:
    results: tuple[AdversarialCaseResultV1, ...]
    evidence: tuple[ArtifactRefV1, ...]
    critical_false_accepts: int
    typed_outcome_mismatches: int
    model_calls: int
    seal: AdversarialBenchmarkSealV1 | None
    seal_artifact: ArtifactRefV1 | None

    @property
    def sealed(self) -> bool:
        return self.seal is not None and self.seal_artifact is not None


def _observed(decision: CapabilityDecisionV1) -> ObservedOutcome:
    if decision.outcome is FirewallOutcome.ALLOW:
        # ALLOW is a critical deterministic false accept for an adversarial row.
        return ObservedOutcome.CERTIFIED
    return ObservedOutcome(decision.outcome.value)


def evaluate_adversarial_cases(
    cases: tuple[BenchmarkCaseV1, ...],
    *,
    artifacts: ContentAddressedArtifactStore,
    benchmark_manifest_hash: str,
    require_exact_100: bool = True,
) -> AdversarialBenchmarkAttempt:
    """Run prompt-only firewall decisions, then audit against frozen labels."""

    if require_exact_100 and len(cases) != 100:
        raise ValueError("adversarial release acceptance requires exactly 100 cases")
    if not cases or any(
        case.kind is not BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL for case in cases
    ):
        raise ValueError("adversarial acceptance accepts only adversarial cases")

    results = []
    references = []
    false_accepts = 0
    mismatches = 0
    model_calls = 0
    for case in cases:
        # Classification deliberately receives only prompt text.  The expected
        # outcome, tags, category, and case ID are unavailable to the firewall.
        decision = validate_prompt(case.prompt)
        is_false_accept = decision.outcome is FirewallOutcome.ALLOW
        typed_match = decision.outcome.value == case.expected_outcome.value
        false_accepts += int(is_false_accept)
        mismatches += int(not typed_match)
        observed = _observed(decision)
        evidence = AdversarialFirewallEvidenceV1(
            case_id=case.case_id,
            case_hash=case.case_hash,
            decision=decision,
            observed_outcome=observed,
            expected_outcome=case.expected_outcome.value,
            typed_outcome_matches=typed_match,
            critical_deterministic_false_accept=is_false_accept,
        )
        references.append(
            artifacts.put_json(
                evidence,
                filename=f"{case.case_id}-firewall-evidence.json",
            )
        )
        results.append(
            AdversarialCaseResultV1(
                case_id=case.case_id,
                case_hash=case.case_hash,
                observed_outcome=observed,
                critical_judge_false_accept=is_false_accept,
            )
        )

    seal = None
    seal_artifact = None
    if (
        require_exact_100
        and len(cases) == 100
        and false_accepts == 0
        and mismatches == 0
        and model_calls == 0
    ):
        evidence_refs = tuple(references)
        seal = AdversarialBenchmarkSealV1(
            benchmark_manifest_hash=benchmark_manifest_hash,
            evidence_artifacts=evidence_refs,
            evidence_set_hash=content_hash(
                [item.model_dump(mode="json") for item in evidence_refs]
            ),
        )
        seal_artifact = artifacts.put_json(
            seal, filename="adversarial_benchmark_100.json"
        )
    return AdversarialBenchmarkAttempt(
        results=tuple(results),
        evidence=tuple(references),
        critical_false_accepts=false_accepts,
        typed_outcome_mismatches=mismatches,
        model_calls=model_calls,
        seal=seal,
        seal_artifact=seal_artifact,
    )


def run_adversarial_benchmark_100(
    *,
    artifacts: ContentAddressedArtifactStore,
    manifest: BenchmarkManifestV1 | None = None,
) -> AdversarialBenchmarkAttempt:
    selected = manifest or load_benchmark_manifest()
    cases = tuple(
        case
        for case in selected.cases
        if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL
    )
    return evaluate_adversarial_cases(
        cases,
        artifacts=artifacts,
        benchmark_manifest_hash=selected.content_hash(),
    )
