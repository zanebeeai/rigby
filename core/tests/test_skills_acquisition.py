"""Bounded acquisition: a frozen problem, a seeded search inside its bounds, and an honest classification.

No body, no physics. A scripted world certifies an episode only when the
search brings a parameter into a region the defaults do not occupy; the
search finds it within the ceiling and the outcome is a discovery. A world
the defaults already satisfy is an instantiation, with no parameter
changed. A world no vector satisfies exhausts the ceiling, and nothing is
rebranded. The search is deterministic given its seed, and every proposal
stays inside the bounds.
"""

from __future__ import annotations

import pytest

from rigby_core.skills.acquisition import (
    AcceptanceV1,
    AcquisitionProblemV1,
    AcquisitionStatus,
    AttemptV1,
    CeilingV1,
    EffectV1,
    EpisodeOutcomeV1,
    EvolutionSearch,
    ParameterSpecV1,
    ProblemKind,
    attempts_digest,
    classify,
)


def problem(**kwargs) -> AcquisitionProblemV1:
    base = dict(
        problem_id="scripted", body="scripted_arm", body_family="scripted", effect=EffectV1(skill_id="transfer_object", arguments={"object": "cube"}),
        world_change="a cube at half friction", hypothesis=ProblemKind.DISCOVERY, controller_family="contact.transfer",
        parameters=(ParameterSpecV1(name="grip_safety_factor", low=10.0, high=400.0, default=80.0, units="ratio", scale="log"),
                    ParameterSpecV1(name="duration_scale", low=1.0, high=4.0, default=1.0, units="ratio")),
        development_seeds=(4000, 4001), confirmation_seeds=(4002, 4003), acceptance=AcceptanceV1(set_id="holdout", trials=50, threshold=40), ceiling=CeilingV1(attempts=200, worker_minutes=30.0),
    )
    base.update(kwargs)
    return AcquisitionProblemV1(**base)


def run_search(problem_: AcquisitionProblemV1, certifies, *, seed=0) -> tuple[list[AttemptV1], AttemptV1 | None]:
    """A scripted acquisition loop: ``certifies(parameters, seed)`` is the world."""

    search = EvolutionSearch(problem_, seed=seed)
    attempts: list[AttemptV1] = []
    best = None
    for index in range(problem_.ceiling.attempts):
        parameters = search.propose()
        for spec in problem_.parameters:
            assert spec.low <= parameters[spec.name] <= spec.high, (spec.name, parameters[spec.name])
        episodes = tuple(EpisodeOutcomeV1(seed=s, certified=certifies(parameters, s), physics_s=15.0, wall_s=4.0) for s in problem_.development_seeds)
        confirmation = ()
        if all(e.certified for e in episodes):
            confirmation = tuple(EpisodeOutcomeV1(seed=s, certified=certifies(parameters, s), physics_s=15.0, wall_s=4.0) for s in problem_.confirmation_seeds)
        attempt = AttemptV1(index=index, parameters=parameters, episodes=episodes, certified=sum(e.certified for e in episodes), of=len(episodes), confirmation=confirmation,
                            sigma=search.sigma, physics_s=15.0 * (len(episodes) + len(confirmation)), wall_s=4.0 * (len(episodes) + len(confirmation)))
        improved = search.observe(parameters, attempt.score)
        attempt = attempt.model_copy(update={"improved": improved})
        attempts.append(attempt)
        if improved or best is None:
            best = attempt if best is None or attempt.score > best.score else best
        if attempt.score == len(episodes) + len(confirmation) and confirmation:
            return attempts, attempt
    return attempts, best


def test_the_search_finds_a_region_the_defaults_do_not_occupy():
    attempts, found = run_search(problem(), lambda p, seed: p["grip_safety_factor"] >= 200.0 and p["duration_scale"] >= 1.8)
    assert found is not None and found.score == 4
    assert found.parameters["grip_safety_factor"] >= 200.0 and found.parameters["duration_scale"] >= 1.8
    assert len(attempts) <= 200 and attempts[0].parameters == problem().defaults()
    status, kind, changed = classify(problem(), found, accepted=True)
    assert status is AcquisitionStatus.ACQUIRED and kind is ProblemKind.DISCOVERY
    assert set(changed) == {"grip_safety_factor", "duration_scale"}


def test_defaults_that_pass_are_an_instantiation_not_a_discovery():
    attempts, found = run_search(problem(hypothesis=ProblemKind.INSTANTIATION), lambda p, seed: True)
    assert len(attempts) == 1 and found.index == 0
    status, kind, changed = classify(problem(), found, accepted=True)
    assert status is AcquisitionStatus.INSTANTIATED and kind is ProblemKind.INSTANTIATION and changed == ()


def test_a_world_nothing_satisfies_exhausts_the_ceiling_and_stays_unacquired():
    small = problem(ceiling=CeilingV1(attempts=12, worker_minutes=30.0))
    attempts, best = run_search(small, lambda p, seed: False)
    assert len(attempts) == 12 and all(a.certified == 0 for a in attempts)
    status, kind, _ = classify(small, None, accepted=False)
    assert status is AcquisitionStatus.BUDGET_EXHAUSTED and kind is None


def test_a_found_vector_that_fails_acceptance_is_not_acquired():
    _, found = run_search(problem(), lambda p, seed: p["duration_scale"] >= 1.5)
    status, kind, changed = classify(problem(), found, accepted=False)
    assert status is AcquisitionStatus.ACCEPTANCE_FAILED and kind is None and "duration_scale" in changed


def test_search_is_deterministic_and_stays_in_bounds():
    first, _ = run_search(problem(), lambda p, seed: p["duration_scale"] >= 3.0, seed=7)
    second, _ = run_search(problem(), lambda p, seed: p["duration_scale"] >= 3.0, seed=7)
    assert attempts_digest(first) == attempts_digest(second)
    third, _ = run_search(problem(), lambda p, seed: p["duration_scale"] >= 3.0, seed=8)
    assert attempts_digest(third) != attempts_digest(first)
    assert EvolutionSearch(problem(), seed=7).provenance().bounds_sha256 == EvolutionSearch(problem(), seed=8).provenance().bounds_sha256


def test_step_size_stays_wide_on_the_plateau_and_contracts_after_an_improvement():
    search = EvolutionSearch(problem(), seed=1, offspring=2)
    search.propose()
    search.observe(problem().defaults(), 0.0)
    before = search.sigma
    for _ in range(4):
        search.observe(search.propose(), 0.0)
    assert search.sigma == before, "nothing has beaten the defaults yet: the walk stays wide"
    search.observe(search.propose(), 1.0)
    search.observe(search.propose(), 0.0)
    for _ in range(2):
        search.observe(search.propose(), 0.0)
    assert search.sigma < before, "after an improvement, a barren generation narrows the step"


def test_problem_contract_refuses_bad_bounds_and_shared_seeds():
    with pytest.raises(ValueError):
        ParameterSpecV1(name="x", low=1.0, high=0.5, default=0.7, units="u")
    with pytest.raises(ValueError):
        ParameterSpecV1(name="x", low=0.0, high=1.0, default=0.5, units="u", scale="log")
    with pytest.raises(ValueError):
        problem(confirmation_seeds=(4000,))
    with pytest.raises(ValueError):
        AcceptanceV1(set_id="h", trials=10, threshold=11)
