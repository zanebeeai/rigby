"""Seed a platform column in a corpus ``expected.json`` payload, consistently.

**Why this exists.** 03d keyed the ``environment`` block by platform, like the three
digest maps, and added a validator: a file whose ``environment`` keys and digest keys
disagree is rejected at load, before any test assertion runs. That validator is
correct and it is the point of 03d.

What it exposed is that several tests seed *this platform's* digest column directly --
to corrupt a hash, or to stage a stale one -- and never touch ``environment``. On a
**blessed** machine the key is already present in both maps, so adding it to the
digests changes nothing about their agreement and the tests pass. On a platform nobody
has blessed the key is new, only the digest maps gain it, the two disagree, and every
one of those tests dies inside `load_corpus` with a validation error rather than at its
own assertion.

That is the same shape 03b catalogued and it caught this lane twice: **the
unblessed-platform path is unreachable from a blessed machine, so tests accumulate
assumptions there and a second machine finds them all at once.** Three tests failed
this way on Windows CI at ``8e4edf0``. The comments at those sites even reason
carefully about the unblessed case -- "writing this platform's key explicitly makes the
test say what it means everywhere" -- and were right about the digests and silent about
the environment, because on the machine they were written on there was nothing to say.

So the repair is not three call-site edits. It is one helper that makes the coupled
edit the only convenient edit, because the next person seeding a digest column will
otherwise reproduce this exactly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

#: The three per-platform digest maps 03d keeps in step with ``environment``.
DIGEST_FIELDS = ("motion_sha256", "metrics_sha256", "observables_sha256")


def seed_platform_column(
    payload: dict[str, Any], key: str, value: str
) -> dict[str, Any]:
    """Give ``key`` a digest of ``value`` in all three maps, and an environment entry.

    Returns the same payload, mutated in place and also returned for chaining.

    The environment entry is **cloned from one the file already carries** rather than
    constructed here. A synthetic entry would drift from the real schema the moment a
    field is added to ``BlessEnvironment``, and the resulting failure would surface as
    a validation error in whatever test happened to seed a column next -- a long way
    from the cause. Cloning an existing entry means this helper has no schema knowledge
    to go stale.
    """
    for name in DIGEST_FIELDS:
        payload[name] = {**payload[name], key: value}
    payload["environment"] = _with_environment_for(payload.get("environment"), key)
    return payload


def _with_environment_for(environment: Any, key: str) -> dict[str, Any]:
    if not isinstance(environment, dict) or not environment:
        raise ValueError(
            "cannot seed a platform column on a payload with no environment block to "
            "clone; 03d requires every blessed platform to record how it blessed, and "
            "this helper deliberately does not invent one"
        )
    if key in environment:
        return environment
    template = deepcopy(next(iter(environment.values())))
    if isinstance(template, dict):
        template["platform_key"] = key
    return {**environment, key: template}


def platform_columns_agree(payload: dict[str, Any]) -> bool:
    """Whether the payload would survive 03d's validator.

    Used by the regression test rather than by the seeding path, so the property is
    asserted somewhere that fails loudly instead of only being maintained by
    convention.
    """
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return False
    return all(set(payload[name]) == set(environment) for name in DIGEST_FIELDS)


__all__ = ["DIGEST_FIELDS", "platform_columns_agree", "seed_platform_column"]
