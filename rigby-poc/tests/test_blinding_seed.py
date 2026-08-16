"""The judge's presentation order must be derived from content, not position.

Plan 07 §1.4: seeding five-way blinding with `70_000 + round_index` puts recipe
*k* in the same A-E slot in round 0 of every run of every prompt, so any
positional bias in the model becomes a reproducible preference for one recipe.
"""

from __future__ import annotations

import random

from evals.flywheel import blinding_seed


RECIPE_COUNT = 5
LABELS = ("A", "B", "C", "D", "E")


def _slot_of_recipe(seed: int, recipe_index: int, *, recipe_count: int = RECIPE_COUNT) -> int:
    """Reproduce `VLMJudge.rank_five`'s shuffle and report where a recipe lands."""
    randomized = list(range(recipe_count))
    random.Random(seed).shuffle(randomized)
    return randomized.index(recipe_index)


def _result_ids(prompt: str) -> list[str]:
    return [f"{prompt}-candidate-{index}" for index in range(RECIPE_COUNT)]


def test_seed_is_stable_for_identical_content() -> None:
    ids = _result_ids("throw a right jab")
    first = blinding_seed("throw a right jab", ids)
    second = blinding_seed("throw a right jab", list(ids))
    assert first == second


def test_seed_ignores_the_order_the_caller_holds_candidates_in() -> None:
    ids = _result_ids("throw a right jab")
    assert blinding_seed("throw a right jab", ids) == blinding_seed(
        "throw a right jab", list(reversed(ids))
    )


def test_seed_differs_across_prompts() -> None:
    ids = _result_ids("shared")
    seeds = {
        prompt: blinding_seed(prompt, ids)
        for prompt in ("throw a right jab", "throw a left hook", "wave hello")
    }
    assert len(set(seeds.values())) == len(seeds)


def test_seed_differs_across_candidate_sets() -> None:
    prompt = "throw a right jab"
    assert blinding_seed(prompt, _result_ids("run-a")) != blinding_seed(
        prompt, _result_ids("run-b")
    )


def test_recipe_zero_is_not_always_in_slot_a_in_round_zero() -> None:
    # Round 0 of many distinct prompts — the exact situation the fixed seed
    # made degenerate.
    slots = [
        _slot_of_recipe(blinding_seed(f"prompt-{index}", _result_ids(f"prompt-{index}")), 0)
        for index in range(200)
    ]
    assert slots.count(0) < len(slots)
    assert set(slots) == set(range(RECIPE_COUNT))


def test_the_old_fixed_seed_was_degenerate() -> None:
    # Guards the regression this PR fixes: `70_000 + round_index` pinned one
    # recipe to slot A across every prompt, which is what the test above rules
    # out for the content-derived seed.
    round_zero_slots = {_slot_of_recipe(70_000, 0) for _ in range(50)}
    assert len(round_zero_slots) == 1


def test_recipe_slots_are_broadly_distributed() -> None:
    counts = [0] * RECIPE_COUNT
    trials = 500
    for index in range(trials):
        prompt = f"prompt-{index}"
        slot = _slot_of_recipe(blinding_seed(prompt, _result_ids(prompt)), 0)
        counts[slot] += 1
    expected = trials / RECIPE_COUNT
    # Generous band: this asserts the seed is not degenerate, not that sha256 is
    # a good PRNG.
    assert all(0.6 * expected <= count <= 1.4 * expected for count in counts), counts


def test_pairwise_seed_is_symmetric_in_the_pair() -> None:
    prompt = "throw a right jab"
    assert blinding_seed(prompt, ["champion", "challenger"]) == blinding_seed(
        prompt, ["challenger", "champion"]
    )


def test_seed_fits_the_shuffle_contract() -> None:
    seed = blinding_seed("throw a right jab", _result_ids("throw a right jab"))
    assert isinstance(seed, int)
    assert 0 <= seed < 2**64
