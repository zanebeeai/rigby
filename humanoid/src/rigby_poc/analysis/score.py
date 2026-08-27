"""Plan 10 §3.3 — the deterministic composite score, and the oracle it feeds.

Plan 10 §6 names two trajectory metrics that need a per-clip score: selection
regret ("was the winner the best available candidate?") and repair efficacy.
`evals/trajectory.py` ships both with an injected ``oracle`` and no default
because this module did not exist. It is the producer.

Three things had to be true of it, and each rules out an obvious design.

**It cannot be the judge.** ``flywheel._candidate_score`` reads
``judgment.overall``, which is the quantity the winner is the argmax of, so
regret computed with it is identically zero by construction. §3.3 exists to
give selection regret an oracle the selector did not produce.

**It cannot be a fold over severities.** ``CheckResult`` forbids a non-zero
severity on anything but a fail, so a severity score is 0.0 for a clip that
passes everything. Measured over the 47-case corpus: severities alone take
**12 distinct values across 47 cases, and 91.1% of the total severity mass is
two DOFs** -- ``left`` and ``rightLowerArm.abduction``, which fail on 46 of 47
cases each and are a compiler defect rather than a property of the clip. A
severity oracle is a readout of that one defect. :attr:`CheckResult.headroom`
is the signed extension that lets a clean clip be ranked at all.

**It cannot be a flat mean over the checks.** ``validate()`` emits 160-167
verdicts per case and **156 of them are ``anatomy.rom.*``** -- between 93% and
97% of the surface, so an unweighted mean is the ROM layer wearing a composite's
name, and one bone's two DOFs move it more than every non-anatomy check
combined. The fold here is balanced at three levels instead: checks within a
family, families within a layer, layers within the clip. One DOF is then 1/156
of ``anatomy.rom``, which is one family of the ``anatomy`` layer, which is one
layer of the score.

**A clip that measured almost nothing is not a clip that scored well.** The
first draft of this module guarded the layer level and shipped the trap one
level down: ``knownbad-eigenvalues-unsupported`` compiles to zero frames, so all
156 range-of-motion checks skip, and the four contract checks that remain are
categorical passes. It scored **+1.0000 -- the best score in the corpus -- on
2.5% of the check surface**, outranking all 41 known-good cases. A mean is only
comparable against another mean over the same surface, so
:class:`CompositeScore` carries its :attr:`~CompositeScore.coverage` and refuses
to publish a score below :data:`MINIMUM_COVERAGE`.

**An absent stratum is absent, not perfect.** A layer with nothing to say is
left out of the fold and named in :attr:`CompositeScore.missing_layers` rather
than scored 1.0, and a clip that measured nothing at all scores ``None``. Plan
10's physics and signal layers are unbuilt at the time of writing, so this is
the ordinary case rather than a corner of it; the mean over an empty collection
is the failure ``docs/testing.md`` opens with, and a score that silently rose
because a layer stopped being emitted is the same shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import LAYERS, CheckResult

#: The share of a clip's checks that must have measured something before a fold
#: over them is comparable to a fold over another clip's.
#:
#: Not a tuning knob and not a quality bar -- it separates "this clip scored
#: well" from "this clip was barely looked at". Measured over the 47-case
#: corpus, coverage takes exactly **three** values: 0.0250 (the one clip that
#: compiles to no frames), 0.99375 and 1.0. The constant sits in a gap **39
#: times wider** than any tolerance it needs, so it is a margin guard rather
#: than a fitted threshold: if a future clip lands anywhere near 0.5, that is
#: news about the corpus and this line should fail rather than absorb it.
MINIMUM_COVERAGE = 0.5

#: Relative weight of each layer in the fold. Equal by default, and that is a
#: decision rather than a default: nothing in this repository establishes that a
#: range-of-motion excursion matters more or less to a viewer than a jerk
#: spike, and inventing a ratio would put an unmeasured judgement inside the
#: instrument that plan 10 §2.5 exists to keep judgement out of. Override at the
#: call site, and say why there.
DEFAULT_LAYER_WEIGHTS: dict[str, float] = {layer: 1.0 for layer in LAYERS}


def check_family(check_id: str) -> str:
    """The stratum a check belongs to within its layer.

    The first two dot-separated components: ``anatomy.rom.leftLowerArm.abduction``
    is ``anatomy.rom``, ``signal.angular.jerk`` is ``signal.angular``. Verified
    over the corpus's 176 distinct ids: they fall into 10 families and no family
    spans two layers.

    Note the family prefix is **not** the layer. ``contract.camera.active_hand_visibility``
    (``gesture.py``) declares ``layer="signal"`` while its id says ``contract``,
    so the layer stratum keys off the declared field and only the family keys
    off the id. Renaming the check would move the id surface, so the
    disagreement is reported rather than papered over here.
    """

    parts = check_id.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else check_id


@dataclass(frozen=True)
class FamilyScore:
    """One check family's mean headroom, and how much of it was measured."""

    family: str
    layer: str
    score: float
    contributing: int
    unmeasured: int


@dataclass(frozen=True)
class LayerScore:
    layer: str
    score: float
    families: tuple[FamilyScore, ...]
    contributing: int
    unmeasured: int


@dataclass(frozen=True)
class CompositeScore:
    """A clip's deterministic score, with the evidence it was folded from.

    ``score`` is signed and in ``[-1, 1]``: negative means something failed,
    positive is room to spare. It is ``None`` when the clip cannot be scored
    comparably -- nothing measured at all, or coverage below
    :data:`MINIMUM_COVERAGE` -- rather than folded to 0.0, because 0.0 is a real
    score meaning "everything sat exactly on its bound", and rather than folded
    over what little was measured, because that is how a zero-frame clip came
    top of the corpus.

    The strata are kept rather than summarised away so a caller can say *why* a
    clip scored what it did. Per-family attribution is what
    ``docs/evaluation.md:64`` has named as missing since the beginning.
    """

    score: float | None
    layers: tuple[LayerScore, ...]
    contributing: int
    unmeasured: int
    missing_layers: tuple[str, ...]

    @property
    def coverage(self) -> float:
        """Share of this clip's checks that measured anything, in ``[0, 1]``.

        1.0 when every check reported a headroom. A clip with no checks at all
        has nothing to be a share of and is 0.0.
        """

        total = self.contributing + self.unmeasured
        return self.contributing / total if total else 0.0

    @property
    def scoreable(self) -> bool:
        return self.score is not None

    @property
    def normalised(self) -> float | None:
        """``score`` mapped onto ``[0, 1]``, higher better -- the Oracle contract.

        ``evals.trajectory.Oracle`` is documented as "a per-clip score in [0, 1]
        where higher is better", so the sign convention is converted at the
        boundary rather than by redefining the contract.

        ``None`` propagates. A caller wiring this into
        ``evals.trajectory.selection_regret`` must drop an unscoreable clip in
        its ``clip_of`` -- which already skips a ``None`` -- rather than
        substituting a number for it. Substituting 0.0 would rank an unmeasured
        candidate as the worst available and substituting 1.0 as the best; both
        are claims about a clip nothing looked at.
        """

        return None if self.score is None else (self.score + 1.0) / 2.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def composite_score(
    checks: Sequence[CheckResult],
    *,
    weights: Mapping[str, float] | None = None,
) -> CompositeScore:
    """Fold a clip's check verdicts into one signed score.

    Balanced at three levels -- see the module docstring for why a flat mean is
    not usable here. ``weights`` re-weights the layer level only; a caller that
    wants to weight families should say so in its own terms rather than
    flattening this.
    """

    layer_weights = dict(DEFAULT_LAYER_WEIGHTS if weights is None else weights)

    by_family: dict[str, list[CheckResult]] = {}
    for check in checks:
        by_family.setdefault(check_family(check.id), []).append(check)

    families: list[FamilyScore] = []
    for family, members in sorted(by_family.items()):
        scored = [c.headroom for c in members if c.headroom is not None]
        if not scored:
            # Every member abstained. The family is absent from the fold; it is
            # not a family that scored zero.
            continue
        layers = {c.layer for c in members}
        if len(layers) != 1:
            raise ValueError(
                f"check family {family!r} spans more than one layer: "
                f"{sorted(layers)}. A family is a stratum within a layer, so "
                "this would make the fold's denominators ambiguous."
            )
        families.append(
            FamilyScore(
                family=family,
                layer=next(iter(layers)),
                score=_mean(scored),
                contributing=len(scored),
                unmeasured=len(members) - len(scored),
            )
        )

    by_layer: dict[str, list[FamilyScore]] = {}
    for entry in families:
        by_layer.setdefault(entry.layer, []).append(entry)

    layer_scores = tuple(
        LayerScore(
            layer=layer,
            score=_mean([f.score for f in members]),
            families=tuple(members),
            contributing=sum(f.contributing for f in members),
            unmeasured=sum(f.unmeasured for f in members),
        )
        for layer, members in sorted(by_layer.items())
    )

    contributing = sum(entry.contributing for entry in layer_scores)
    unmeasured = sum(1 for c in checks if c.headroom is None)
    missing = tuple(layer for layer in LAYERS if layer not in by_layer)

    if not layer_scores:
        return CompositeScore(None, (), 0, unmeasured, missing)

    total_checks = contributing + unmeasured
    if total_checks and contributing / total_checks < MINIMUM_COVERAGE:
        # Scored strata are kept: the caller still gets to see what little was
        # measured and why the clip was refused. Only the top-level number is
        # withheld, because that is the one another clip's number gets compared
        # against.
        return CompositeScore(None, layer_scores, contributing, unmeasured, missing)

    total_weight = sum(layer_weights.get(entry.layer, 0.0) for entry in layer_scores)
    if total_weight <= 0.0:
        raise ValueError(
            "every layer that produced a score has zero weight, so the fold has "
            f"no denominator: scored {[e.layer for e in layer_scores]}"
        )
    score = (
        sum(layer_weights.get(e.layer, 0.0) * e.score for e in layer_scores)
        / total_weight
    )
    return CompositeScore(score, layer_scores, contributing, unmeasured, missing)


def clip_composite(clip: Any, program: Any) -> CompositeScore:
    """:func:`composite_score` over a compiled clip, via the public check surface.

    Goes through :func:`rigby_poc.analysis.validate_clip` rather than assembling
    checks itself, so the oracle scores the same verdicts every other consumer
    reads. Plan 10 §8.1's rule for ``--offline-scoring`` is the same rule: a
    harness that reaches past the production path measures the harness.
    """

    from . import validate_clip

    return composite_score(validate_clip(clip, program))


__all__ = [
    "DEFAULT_LAYER_WEIGHTS",
    "MINIMUM_COVERAGE",
    "CompositeScore",
    "FamilyScore",
    "LayerScore",
    "check_family",
    "clip_composite",
    "composite_score",
]
