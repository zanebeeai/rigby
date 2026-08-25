"""The 28 specs from ``evals/corruptions.py``, ported as the severe tier.

Plan 06 section 2 non-goals: "Replacing the existing 28 specs. They are absorbed as
the severe tier of their families."  This module is that absorption, and it is a
port rather than a rewrite -- the transforms delegate to ``evals.corruptions`` so the
motion produced is bit-identical to what the previous calibration used.

**Three things the port deliberately does not carry across.**  ``corrupt_clip`` sets
``success=False``, attaches a ``Failure`` naming the corruption, and writes
``metrics["deliberate_corruption"]``.  Those made the mutated artifact announce
itself, and ``evals/calibrate_judge.py:224`` then recorded the outcome as the **conjunction**
of the grader's accept flag with ``structural_valid``, and the corruption was built to
force the second term false -- so the recorded outcome was not a function of the
grader's verdict.  See plan 06 section 6.5.  :func:`_strip_labels` removes all
three.

**Why they are all severe.**  The mildest wrist spec is 0.85 rad, about 49 degrees.
Plan 06 section 1.3: a grader that detects a 49-degree wrist error has demonstrated
almost nothing.  Every one of these lands at ``Tier.SEVERE``, which is the honest
label and makes a report covering only this tier visibly a report covering only this
tier.

**Why they are gesture-only.**  Every spec needs a ``present``, ``hold``, ``shake``
or ``recover`` phase in ``phase_ranges_s``, which only the gesture path emits.  That
is exactly the per-``(spec, case)`` applicability plan 06 section 6.1 requires, so
the guard is derived from the phases the clip actually has rather than from a list of
case ids.
"""

from __future__ import annotations

from evals.corruptions import CorruptionSpec, corrupt_clip, corruption_specs
from rigby_poc.models import ClipResult, Hand

from .family import MutationFamily, Tier
from .inject import is_static
from .spec import Applicability, MutationSpec

#: Which family each legacy ``kind`` belongs to, and what it should trip.
#: ``wrong_joint_shake`` is deliberately ``anatomy`` with **no deterministic target**
#: -- its own docstring says the magnitude stays inside the structural wrist limit on
#: purpose, so it is a visual-anatomy defect for a model grader rather than a
#: deterministic-gate failure.  It is therefore carried as a semantic mutation with
#: an asserted program-level difference, per plan 06 section 6.1.
#:
#: **Corrected in 06b: the four ids here were names no check emits.**  06a took them
#: from plan 06 section 3.3 -- ``anatomy.wrist.swing``, ``anatomy.wrist.twist``,
#: ``signal.min_jerk`` and ``signal.sparc`` -- and measured over all 47 corpus cases
#: :func:`rigby_poc.analysis.validate` emits none of them; ``min_jerk`` and ``sparc``
#: occur nowhere in the repository outside that one declaration.  All 28 ported specs
#: therefore pointed at checks that could not fire, and the detection matrix would
#: have rendered them identically to "no detector exists" -- a total detection
#: failure by the deterministic layer, as a result rather than as a visible gap.
#: The real ids are below, and :func:`evals.mutations.checks.require_known_targets`
#: now refuses a spec that names one that is not emitted.
LEGACY_FAMILIES: dict[str, tuple[MutationFamily, tuple[str, ...]]] = {
    # One check covers both swing and twist; there is no separate id per axis.
    "wrist_rotation": (MutationFamily.ANATOMY, ("anatomy.wrist.swing_twist_limit",)),
    "fist_shape": (MutationFamily.SEMANTIC, ()),
    "open_middle_fingers": (MutationFamily.SEMANTIC, ()),
    # The legacy timing corruptions perturb when the motion happens, which this
    # repo measures as angular kinematics and frame-to-frame discontinuity.
    "timing": (
        MutationFamily.TIMING,
        (
            "signal.angular.jerk",
            "signal.angular.acceleration",
            "contract.clip.rotational_discontinuities",
        ),
    ),
    "wrong_joint_shake": (MutationFamily.SEMANTIC, ()),
}

#: The program-level difference each semantic legacy kind produces, asserted
#: independently so the negative is valid without a deterministic oracle.
LEGACY_ASSERTS: dict[str, tuple[str, ...]] = {
    "fist_shape": ("hand_shape.thumb_and_little_curled",),
    "open_middle_fingers": ("hand_shape.middle_three_extended",),
    "wrong_joint_shake": ("shake.oscillation_moved_from_forearm_to_wrist",),
}

#: Phase kinds each legacy transform needs present in ``phase_ranges_s``.
LEGACY_PHASES: dict[str, tuple[str, ...]] = {
    "wrist_rotation": (),
    "fist_shape": (),
    "open_middle_fingers": (),
    "wrong_joint_shake": ("shake",),
    "timing": ("present",),
}

#: ``remove_hold`` additionally needs both a hold and a recover phase.
REMOVE_HOLD_PHASES = ("present", "hold", "recover")

#: Labels the legacy transform stamps onto the clip, removed on the way out.
STRIPPED_METRIC_KEYS = ("deliberate_corruption",)


def _phases(clip: ClipResult) -> set[str]:
    return {
        str(item["kind"])
        for item in clip.metrics.get("phase_ranges_s", [])
        if isinstance(item, dict) and "kind" in item
    }


def _required_phases(legacy: CorruptionSpec) -> tuple[str, ...]:
    if legacy.kind == "timing" and legacy.mode == "remove_hold":
        return REMOVE_HOLD_PHASES
    return LEGACY_PHASES[legacy.kind]


def _strip_labels(mutated: ClipResult, base: ClipResult) -> ClipResult:
    """Undo the legacy transform's self-identification.

    A mutated clip must be indistinguishable from an ordinary one except for the
    mutation.  ``success`` is restored from the base rather than forced true: the
    base's value is what an unmutated compile said, and the mutation is not entitled
    to an opinion about it.
    """
    metrics = {
        key: value for key, value in mutated.metrics.items() if key not in STRIPPED_METRIC_KEYS
    }
    return mutated.model_copy(
        update={"metrics": metrics, "success": base.success, "failure": base.failure}
    )


def _hand_of(clip: ClipResult) -> Hand:
    """Which hand the legacy transform should act on.

    Read off the clip rather than passed in: the legacy signature takes a hand
    because it was always called with one from a live request, and a mutation that
    guesses wrong perturbs the arm that is not moving.
    """
    active = clip.metrics.get("active_hands")
    if isinstance(active, list) and active:
        return Hand(str(active[0]))
    handedness = clip.slider_observables.get("handedness")
    if isinstance(handedness, (int, float)):
        return Hand.LEFT if handedness < 0 else Hand.RIGHT
    return Hand.RIGHT


def _make_guard(legacy: CorruptionSpec):
    required = _required_phases(legacy)

    def guard(clip: ClipResult, _spec: MutationSpec) -> Applicability:
        present = _phases(clip)
        missing = [kind for kind in required if kind not in present]
        if missing:
            return Applicability(
                False,
                f"needs {'/'.join(required)} phase(s); this clip has "
                f"{sorted(present) or 'none'}",
            )
        prefix = _hand_of(clip).value
        hand_bone = f"{prefix}Hand"
        if not any(hand_bone in frame.bones for frame in clip.frames):
            return Applicability(False, f"clip carries no {hand_bone} bone")
        # Applicable either way; tagged so a static target is reported as a
        # capability result rather than pooled into a threshold curve.
        return Applicability(True, static_target=is_static(clip, hand_bone))

    return guard


def _make_transform(legacy: CorruptionSpec):
    def transform(clip: ClipResult, _spec: MutationSpec) -> ClipResult:
        return _strip_labels(corrupt_clip(clip, _hand_of(clip), legacy), clip)

    return transform


def legacy_spec(legacy: CorruptionSpec) -> MutationSpec:
    family, targets = LEGACY_FAMILIES[legacy.kind]
    return MutationSpec(
        id=f"legacy-{legacy.id}",
        family=family,
        targets=targets,
        severity=1.0,
        tier=Tier.SEVERE,
        params={
            "legacy_kind": legacy.kind,
            "legacy_id": legacy.id,
            "axis": legacy.axis,
            "angle_rad": legacy.angle_rad,
            "mode": legacy.mode,
        },
        transform=_make_transform(legacy),
        guard=_make_guard(legacy),
        asserts=LEGACY_ASSERTS.get(legacy.kind, ()),
    )


def legacy_specs() -> list[MutationSpec]:
    """All 28, in the new shape, at ``Tier.SEVERE``."""
    return [legacy_spec(legacy) for legacy in corruption_specs()]
