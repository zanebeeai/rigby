"""Unscored observation-path probe; no manipulation capability is implied."""

from rigby_core.observations import PolicyAction, PolicyObservation


def zero_command_probe(observation: PolicyObservation) -> PolicyAction:
    positions = [r for r in observation.readings if r.channel.endswith("_position")]
    if not positions:
        raise ValueError("Zero-command probe needs one position channel per actuator")
    return PolicyAction(values=tuple(0.0 for _ in positions))
