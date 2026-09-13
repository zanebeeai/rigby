"""A versioned store of skills and the certificates that say where they are valid.

A skill definition is body-neutral; a certificate is not. A certificate says
that one definition, from one library, passed an independent validation set
in one context -- one body, one controller, one sensor configuration, one
geometry, one range of friction and mass, one evidence schema -- and it
carries the cost of finding that out and the hashes of the evidence that
shows it. Everything a later run needs to decide whether that certificate
applies to it is in the context, and the decision is made by comparing
contexts, never by looking at how similar a request seems to the one that
was validated. Retrieval ranks candidates by similarity so a person can see
what nearly applied; validity is a separate, exact verdict, and a retrieval
whose best match is invalid returns no certificate.

Promotion is a gate, not a record: a candidate becomes promoted only when a
validation set that is independent of the sets the skill was developed on
passes its threshold. A candidate that fails stays a candidate, with the
outcome attached, and neither retrieval similarity nor a saved success can
promote it. When a dimension of the context changes -- a body's geometry,
the controller, the sensors, the friction assumptions, the evidence schema
-- every certificate whose facet for that dimension differs is invalidated
with the dimension named, and a certificate returns to service only through
a new validation outcome under the new context, which produces a new
version that supersedes the old one. The store keeps every version and
every decision in its history.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from ..contracts import Contract
from ..hashing import content_hash
from .contract import RangeV1, SkillLibraryV1


class ContextDimension(StrEnum):
    """The dimensions a certificate's validity is conditioned on."""

    BODY = "body"
    CONTROLLER = "controller"
    SENSORS = "sensors"
    GEOMETRY = "geometry"
    FRICTION = "friction"
    EVIDENCE_SCHEMA = "evidence_schema"


class ContextFacetV1(Contract):
    """One dimension of a context: a name for people, a digest for the check,
    the detail the digest was taken over."""

    dimension: ContextDimension
    name: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    detail: dict[str, Any] = {}

    @classmethod
    def of(cls, dimension: ContextDimension, name: str, detail: dict[str, Any]) -> "ContextFacetV1":
        return cls(dimension=dimension, name=name, sha256=content_hash(detail), detail=detail)


class ExecutionContextV1(Contract):
    """Where a skill runs or ran: exact facets, and ranges on the quantities
    a run may vary within (a certificate) or the values it has (a query)."""

    facets: tuple[ContextFacetV1, ...]
    ranges: tuple[RangeV1, ...] = ()
    """What a certificate claims to cover."""
    values: dict[str, float] = {}
    """What a query brings; checked against a certificate's ranges."""

    @model_validator(mode="after")
    def one_facet_per_dimension(self) -> Self:
        dimensions = [f.dimension for f in self.facets]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("a context carries one facet per dimension")
        quantities = [r.quantity for r in self.ranges]
        if len(set(quantities)) != len(quantities):
            raise ValueError("a context carries one range per quantity")
        return self

    def facet(self, dimension: ContextDimension) -> ContextFacetV1 | None:
        for candidate in self.facets:
            if candidate.dimension is dimension:
                return candidate
        return None

    def range(self, quantity: str) -> RangeV1 | None:
        for candidate in self.ranges:
            if candidate.quantity == quantity:
                return candidate
        return None


class ValidityDifferenceV1(Contract):
    """One reason a certificate does not apply to a query."""

    kind: Literal["facet", "range", "missing"]
    dimension: ContextDimension | None = None
    quantity: str | None = None
    certified: str = ""
    queried: str = ""


class ValidityVerdictV1(Contract):
    valid: bool
    differences: tuple[ValidityDifferenceV1, ...] = ()
    similarity: float = Field(ge=0.0, le=1.0)
    """The fraction of facets that match: a ranking aid, never a verdict."""


def compare(certified: ExecutionContextV1, queried: ExecutionContextV1) -> ValidityVerdictV1:
    """Exact on every facet the certificate carries, inside every range it
    claims; a query that omits a facet or a value the certificate has is
    not covered by it."""

    differences: list[ValidityDifferenceV1] = []
    matched = 0
    for facet in certified.facets:
        other = queried.facet(facet.dimension)
        if other is None:
            differences.append(ValidityDifferenceV1(kind="missing", dimension=facet.dimension, certified=facet.name))
        elif other.sha256 != facet.sha256:
            differences.append(ValidityDifferenceV1(kind="facet", dimension=facet.dimension, certified=facet.name, queried=other.name))
        else:
            matched += 1
    for bound in certified.ranges:
        value = queried.values.get(bound.quantity)
        if value is None or not math.isfinite(value):
            differences.append(ValidityDifferenceV1(kind="missing", quantity=bound.quantity, certified=f"[{bound.low:g}, {bound.high:g}] {bound.units}"))
        elif not (bound.low <= value <= bound.high):
            differences.append(ValidityDifferenceV1(kind="range", quantity=bound.quantity, certified=f"[{bound.low:g}, {bound.high:g}] {bound.units}", queried=f"{value:g}"))
    similarity = matched / len(certified.facets) if certified.facets else 1.0
    return ValidityVerdictV1(valid=not differences, differences=tuple(differences), similarity=similarity)


class ValidationSetV1(Contract):
    """The episodes a candidate has to pass, fixed before it runs them."""

    set_id: str = Field(min_length=1)
    body: str = Field(min_length=1)
    episodes: tuple[str, ...] = Field(min_length=1)
    draws_sha256: str = Field(min_length=64, max_length=64)
    independent_of: tuple[str, ...] = Field(min_length=1)
    """The development sets whose seeds this set shares none of."""
    threshold: int = Field(ge=1)
    """Successes required, of ``len(episodes)``."""

    @model_validator(mode="after")
    def threshold_within_set(self) -> Self:
        if self.threshold > len(self.episodes):
            raise ValueError("a validation set cannot require more successes than it has episodes")
        return self


class ValidationOutcomeV1(Contract):
    set_id: str = Field(min_length=1)
    episodes: int = Field(ge=1)
    successes: int = Field(ge=0)
    false_completions: int = Field(ge=0)
    threshold: int = Field(ge=1)
    evidence: tuple[str, ...] = ()
    """Content digests of the sealed bundles the outcome rests on."""
    rows_sha256: str = Field(min_length=64, max_length=64)
    """Digest of every episode's row, so the counts can be recomputed."""

    @property
    def passed(self) -> bool:
        return self.successes >= self.threshold and self.false_completions == 0


class CostV1(Contract):
    physics_s: float = Field(ge=0.0)
    wall_s: float = Field(ge=0.0)
    episodes: int = Field(ge=0)
    generation_calls: int = Field(ge=0)
    dollars: float = Field(ge=0.0)


class CertificateStatus(StrEnum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


class CertificateV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    certificate_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    library_id: str = Field(min_length=1)
    library_sha256: str = Field(min_length=64, max_length=64)
    definition_sha256: str = Field(min_length=64, max_length=64)
    context: ExecutionContextV1
    validation: ValidationOutcomeV1
    cost: CostV1
    status: CertificateStatus = CertificateStatus.CANDIDATE
    version: int = Field(default=1, ge=1)
    decided_at_utc: str = ""
    reason: str = ""
    supersedes: str | None = None
    invalidated_by: tuple[ContextDimension, ...] = ()

    @classmethod
    def candidate(cls, *, skill_id: str, library: SkillLibraryV1, context: ExecutionContextV1, validation: ValidationOutcomeV1, cost: CostV1, version: int = 1, supersedes: str | None = None) -> "CertificateV1":
        definition = library.skill(skill_id)
        material = {"skill": skill_id, "library": library.content_hash(), "definition": definition.content_hash(), "context": content_hash(context), "validation": content_hash(validation), "version": version}
        return cls(certificate_id=content_hash(material)[:32], skill_id=skill_id, library_id=library.library_id, library_sha256=library.content_hash(),
                   definition_sha256=definition.content_hash(), context=context, validation=validation, cost=cost, version=version, supersedes=supersedes)


class StoreEventV1(Contract):
    at_utc: str
    event: Literal["promoted", "rejected", "invalidated", "revalidated", "revalidation_failed", "retrieved", "library_added"]
    certificate_id: str | None = None
    skill_id: str | None = None
    detail: dict[str, Any] = {}


class RetrievalMatchV1(Contract):
    certificate_id: str
    skill_id: str
    version: int
    status: CertificateStatus
    verdict: ValidityVerdictV1


class RetrievalTraceV1(Contract):
    """Every certificate of the skill, ranked by similarity, each with its
    exact verdict; the chosen one is the most similar *valid* promoted
    certificate, and there may be none."""

    skill_id: str
    query: ExecutionContextV1
    matches: tuple[RetrievalMatchV1, ...]
    chosen: str | None
    reason: str


class SkillStoreV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    store_id: str = Field(min_length=1)
    version: int = Field(default=0, ge=0)
    libraries: dict[str, SkillLibraryV1] = {}
    """By content digest: a certificate names the library it was issued against."""
    certificates: tuple[CertificateV1, ...] = ()
    history: tuple[StoreEventV1, ...] = ()

    # -- lookup -----------------------------------------------------------------

    def certificate(self, certificate_id: str) -> CertificateV1:
        for candidate in self.certificates:
            if candidate.certificate_id == certificate_id:
                return candidate
        raise KeyError(certificate_id)

    def library(self, sha256: str) -> SkillLibraryV1:
        return self.libraries[sha256]

    def certificates_for(self, skill_id: str) -> tuple[CertificateV1, ...]:
        return tuple(c for c in self.certificates if c.skill_id == skill_id)

    # -- mutation (every method returns a new store) ---------------------------------------

    def _with(self, *, certificates: tuple[CertificateV1, ...] | None = None, event: StoreEventV1, libraries: dict[str, SkillLibraryV1] | None = None) -> "SkillStoreV1":
        return self.model_copy(update={
            "version": self.version + 1,
            "certificates": self.certificates if certificates is None else certificates,
            "libraries": self.libraries if libraries is None else libraries,
            "history": (*self.history, event),
        })

    def add_library(self, library: SkillLibraryV1) -> "SkillStoreV1":
        digest = library.content_hash()
        if digest in self.libraries:
            return self
        return self._with(libraries={**self.libraries, digest: library}, event=StoreEventV1(at_utc=_now(), event="library_added", detail={"library_id": library.library_id, "sha256": digest}))

    def promote(self, candidate: CertificateV1, validation_set: ValidationSetV1, *, development_sets: tuple[str, ...]) -> tuple["SkillStoreV1", CertificateV1]:
        """Promote only when the validation set is independent of every set
        the skill was developed on and its outcome passes. A failing or a
        dependent candidate is kept as a candidate with the reason attached."""

        if candidate.library_sha256 not in self.libraries:
            raise KeyError(f"the store holds no library {candidate.library_sha256[:12]}; add it before promoting against it")
        if candidate.validation.set_id != validation_set.set_id:
            raise ValueError("the candidate's outcome is not this validation set's")
        reasons = []
        overlap = set(validation_set.independent_of) & set(development_sets)
        if not overlap or set(development_sets) - set(validation_set.independent_of):
            reasons.append("validation_set_not_independent_of_every_development_set")
        if validation_set.threshold != candidate.validation.threshold:
            reasons.append("threshold_mismatch")
        if not candidate.validation.passed:
            reasons.append(f"validation_failed:{candidate.validation.successes}/{candidate.validation.episodes}<{candidate.validation.threshold}"
                           + (f",false_completions={candidate.validation.false_completions}" if candidate.validation.false_completions else ""))
        at = _now()
        if reasons:
            decided = candidate.model_copy(update={"status": CertificateStatus.CANDIDATE, "decided_at_utc": at, "reason": "; ".join(reasons)})
            store = self._with(certificates=(*self.certificates, decided), event=StoreEventV1(at_utc=at, event="rejected", certificate_id=decided.certificate_id, skill_id=decided.skill_id, detail={"reasons": reasons}))
            return store, decided
        promoted = candidate.model_copy(update={"status": CertificateStatus.PROMOTED, "decided_at_utc": at,
                                                "reason": f"independent set {validation_set.set_id} passed {candidate.validation.successes}/{candidate.validation.episodes} (threshold {validation_set.threshold}), no false completion"})
        certificates = list(self.certificates)
        if promoted.supersedes is not None:
            certificates = [c.model_copy(update={"status": CertificateStatus.SUPERSEDED}) if c.certificate_id == promoted.supersedes else c for c in certificates]
        store = self._with(certificates=(*certificates, promoted), event=StoreEventV1(at_utc=at, event="revalidated" if promoted.supersedes else "promoted", certificate_id=promoted.certificate_id, skill_id=promoted.skill_id,
                                                                                    detail={"validation_set": validation_set.set_id, "successes": candidate.validation.successes, "episodes": candidate.validation.episodes, "supersedes": promoted.supersedes}))
        return store, promoted

    def retrieve(self, skill_id: str, query: ExecutionContextV1) -> RetrievalTraceV1:
        matches = []
        for certificate in self.certificates_for(skill_id):
            matches.append(RetrievalMatchV1(certificate_id=certificate.certificate_id, skill_id=skill_id, version=certificate.version, status=certificate.status, verdict=compare(certificate.context, query)))
        matches.sort(key=lambda m: (-m.verdict.similarity, -m.version))
        chosen = next((m for m in matches if m.status is CertificateStatus.PROMOTED and m.verdict.valid), None)
        if chosen is not None:
            reason = f"promoted certificate {chosen.certificate_id} v{chosen.version} matches every facet and range"
        elif matches:
            best = matches[0]
            reason = (f"no valid promoted certificate; the most similar ({best.certificate_id}, {best.status.value}, similarity {best.verdict.similarity:.2f}) differs on "
                      + ", ".join(d.dimension.value if d.dimension else d.quantity or d.kind for d in best.verdict.differences) if best.verdict.differences else
                      f"no valid promoted certificate; the most similar ({best.certificate_id}) is {best.status.value}")
        else:
            reason = "no certificate for this skill"
        return RetrievalTraceV1(skill_id=skill_id, query=query, matches=tuple(matches), chosen=None if chosen is None else chosen.certificate_id, reason=reason)

    def invalidate(self, dimension: ContextDimension, changed_to: ContextFacetV1, *, reason: str = "") -> tuple["SkillStoreV1", tuple[str, ...]]:
        """Every promoted certificate whose facet for ``dimension`` is not the
        new one is invalidated, and the dimension is named on it."""

        affected = []
        certificates = []
        for certificate in self.certificates:
            facet = certificate.context.facet(dimension)
            if certificate.status is CertificateStatus.PROMOTED and (facet is None or facet.sha256 != changed_to.sha256):
                affected.append(certificate.certificate_id)
                certificates.append(certificate.model_copy(update={"status": CertificateStatus.INVALIDATED, "invalidated_by": (*certificate.invalidated_by, dimension),
                                                                   "reason": f"{dimension.value} changed to {changed_to.name}" + (f": {reason}" if reason else "")}))
            else:
                certificates.append(certificate)
        at = _now()
        store = self._with(certificates=tuple(certificates), event=StoreEventV1(at_utc=at, event="invalidated", detail={"dimension": dimension.value, "changed_to": changed_to.name, "affected": affected, "reason": reason}))
        return store, tuple(affected)

    def revalidate(self, invalidated_id: str, *, library: SkillLibraryV1, context: ExecutionContextV1, validation: ValidationOutcomeV1, validation_set: ValidationSetV1, cost: CostV1,
                   development_sets: tuple[str, ...]) -> tuple["SkillStoreV1", CertificateV1]:
        """A new version under the new context, promoted only if its own
        validation passes; otherwise the old certificate stays invalidated
        and the failed attempt is kept as a candidate."""

        old = self.certificate(invalidated_id)
        if old.status is not CertificateStatus.INVALIDATED:
            raise ValueError(f"{invalidated_id} is {old.status.value}, not invalidated")
        candidate = CertificateV1.candidate(skill_id=old.skill_id, library=library, context=context, validation=validation, cost=cost, version=old.version + 1, supersedes=old.certificate_id)
        store = self.add_library(library)
        store, decided = store.promote(candidate, validation_set, development_sets=development_sets)
        if decided.status is not CertificateStatus.PROMOTED:
            store = store._with(event=StoreEventV1(at_utc=_now(), event="revalidation_failed", certificate_id=decided.certificate_id, skill_id=old.skill_id, detail={"invalidated": invalidated_id, "reason": decided.reason}))
        return store, decided

    def note_retrieval(self, trace: RetrievalTraceV1) -> "SkillStoreV1":
        return self._with(event=StoreEventV1(at_utc=_now(), event="retrieved", certificate_id=trace.chosen, skill_id=trace.skill_id, detail={"reason": trace.reason, "candidates": len(trace.matches)}))

    # -- persistence -----------------------------------------------------------------

    def save(self, root: Path) -> dict[str, str]:
        """``store.json`` with the index and history, one file per certificate
        and per library; returns the digests written."""

        root = Path(root)
        (root / "certificates").mkdir(parents=True, exist_ok=True)
        (root / "libraries").mkdir(parents=True, exist_ok=True)
        digests: dict[str, str] = {}
        for certificate in self.certificates:
            path = root / "certificates" / f"{certificate.certificate_id}.json"
            path.write_bytes(_json(certificate.model_dump(mode="json")))
            digests[f"certificates/{certificate.certificate_id}.json"] = certificate.content_hash()
        for digest, library in self.libraries.items():
            path = root / "libraries" / f"{digest}.json"
            path.write_bytes(library.model_dump_json(indent=2).encode("utf-8"))
            digests[f"libraries/{digest}.json"] = digest
        index = {
            "schema_version": self.schema_version, "store_id": self.store_id, "version": self.version, "saved_at_utc": _now(),
            "libraries": sorted(self.libraries), "certificates": [c.certificate_id for c in self.certificates],
            "history": [e.model_dump(mode="json") for e in self.history], "files": digests,
        }
        (root / "store.json").write_bytes(_json(index))
        return digests

    @classmethod
    def load(cls, root: Path) -> "SkillStoreV1":
        root = Path(root)
        index = json.loads((root / "store.json").read_bytes())
        libraries = {digest: SkillLibraryV1.model_validate_json((root / "libraries" / f"{digest}.json").read_bytes()) for digest in index["libraries"]}
        for digest, library in libraries.items():
            if library.content_hash() != digest:
                raise ValueError(f"library {digest[:12]} does not hash to its name")
        certificates = tuple(CertificateV1.model_validate_json((root / "certificates" / f"{cid}.json").read_bytes()) for cid in index["certificates"])
        for certificate in certificates:
            if certificate.library_sha256 not in libraries:
                raise ValueError(f"certificate {certificate.certificate_id} names a library the store does not hold")
        return cls(store_id=index["store_id"], version=index["version"], libraries=libraries, certificates=certificates,
                   history=tuple(StoreEventV1.model_validate(e) for e in index["history"]))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


__all__ = [
    "CertificateStatus", "CertificateV1", "ContextDimension", "ContextFacetV1", "CostV1", "ExecutionContextV1", "RetrievalMatchV1", "RetrievalTraceV1",
    "SkillStoreV1", "StoreEventV1", "ValidationOutcomeV1", "ValidationSetV1", "ValidityDifferenceV1", "ValidityVerdictV1", "compare",
]
