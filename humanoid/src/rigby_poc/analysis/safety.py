"""Clip-level safety metrics: NaNs, discontinuities, root drift.

Moved verbatim from ``compiler._safety_metrics``. Every compile path calls it,
so it is the one check that applies to every intent.

**No longer joint limits.** This module used to walk ``profile["joint_limits_rad"]``
and count per-bone excursions into ``joint_limit_violations``. That check fired
on **0 of 47** corpus cases while the per-DOF range-of-motion layer found a
violation on 46 of 47 over the same clips and the same bones -- inert rather
than permissive, so tightening it would have moved nothing. It is deleted here
because ``anatomy.rom.<bone>.<dof>`` replaces it and, since 04d, is reachable
from :func:`rigby_poc.analysis.validate`. Deleting it before that wiring landed
would have opened a window with no joint-limit gating at all; plan 04's ordering
constraint exists for that reason and this is its second half.

**Root drift has its own check now, and that is a correction rather than a
tidy-up.** It used to increment ``joint_limit_violations``, so a clip whose hips
travelled too far was reported as exceeding a *joint limit* -- a number under a
name that does not describe it, read by ten sites in ``compiler.py``. The
quantity is unchanged; only the label and the addressability are.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from ..models import ClipFrame
from ..thresholds import value_of
from .contract import CONTRACT, CheckResult, count_check, skipped, upper_bound_check

#: How a clip's root translation is judged. ``fixed`` is the float-noise
#: epsilon every hands-and-arms path compiles under; ``bounded`` is the strike
#: envelope -- root motion is real but must stay inside a stated bound rather
#: than being waved through; ``free`` is whole-body and sequence, where the
#: program legitimately travels and there is no whole-clip bound to apply.
#: Three states, not a bool, because the 2026-08-29 ruling on legs-in-strikes
#: rejected both binary outcomes: joining ``free`` deletes the strike root
#: gate entirely, and staying ``fixed`` (1e-06) forbids a 5 cm crouch by
#: 50,000x. A ``Literal`` rather than an enum so every state is enumerable by
#: the type and a router with a missing branch is a type error, not a silent
#: fold -- the same closure argument the check-status router records.
RootDriftPolicy = Literal["fixed", "bounded", "free"]


def safety_metrics(
    frames: list[ClipFrame],
    *,
    allow_root_motion: bool = False,
) -> dict[str, Any]:
    if not frames:
        return {
            "joint_limit_violations": 0,
            "root_drift_m": 0.0,
            "foot_drift_m": 0.0,
            "nan_count": 0,
            "discontinuities": 0,
            "max_frame_rotation_delta_rad": 0.0,
            "quaternion_norm_max_error": 0.0,
            "safety_derivation": "no frames",
        }
    discontinuity_rad = value_of("signal.discontinuity_rad")
    root_drift_max = value_of("physics.root_drift_max_m")
    all_quats = np.asarray(
        [pose.rotation.as_list() for frame in frames for pose in frame.bones.values()], dtype=float
    )
    all_positions = np.asarray(
        [
            pose.position.as_list()
            for frame in frames
            for pose in frame.bones.values()
            if pose.position is not None
        ],
        dtype=float,
    )
    nan_count = int(np.count_nonzero(~np.isfinite(all_quats)))
    if all_positions.size:
        nan_count += int(np.count_nonzero(~np.isfinite(all_positions)))
    norm_error = float(np.max(np.abs(np.linalg.norm(all_quats, axis=1) - 1.0)))
    max_delta = 0.0
    discontinuities = 0
    for previous, current in zip(frames, frames[1:]):
        frame_max = 0.0
        for key in previous.bones:
            a = np.asarray(previous.bones[key].rotation.as_list())
            b = np.asarray(current.bones[key].rotation.as_list())
            delta = 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))
            frame_max = max(frame_max, delta)
        max_delta = max(max_delta, frame_max)
        discontinuities += int(frame_max > discontinuity_rad)
    fixed_keys = ("hips", "leftFoot", "rightFoot", "leftToes", "rightToes")
    fixed_delta = max(
        2.0 * math.acos(float(np.clip(abs(frame.bones[key].rotation.w), 0.0, 1.0)))
        for frame in frames
        for key in fixed_keys
    )
    hips_positions = np.asarray(
        [
            frame.bones["hips"].position.as_list()
            if frame.bones["hips"].position is not None
            else [0.0, 0.0, 0.0]
            for frame in frames
        ],
        dtype=float,
    )
    root_drift = max(
        (float(np.linalg.norm(position - hips_positions[0])) for position in hips_positions),
        default=0.0,
    )
    return {
        # Zero from this module now. The gesture/strike/grab path adds its wrist
        # swing/twist violations to it in ``hand.py``, which is the only thing
        # left in the aggregate and the only thing its name ever described.
        "joint_limit_violations": 0,
        "root_drift_m": root_drift,
        "foot_drift_m": 0.0,
        "fixed_root_foot_max_rotation_delta_rad": fixed_delta,
        "nan_count": nan_count,
        "discontinuities": discontinuities,
        "max_frame_rotation_delta_rad": max_delta,
        "quaternion_norm_max_error": norm_error,
        "safety_derivation": (
            "computed over every frame/local delta quaternion and authored hips translation; "
            + ("root motion is explicitly enabled" if allow_root_motion else "root motion must remain fixed")
        ),
    }


def root_drift_limit_m(*, policy: RootDriftPolicy) -> float | None:
    """The bound a clip's root drift is judged against, or ``None`` if unbounded.

    Derived rather than published. An earlier version of this put it in
    ``ClipResult.metrics`` so the bound would travel beside the measurement,
    which is the reporting discipline this repo keeps insisting on -- and it
    moved the golden corpus's ``metrics_sha256`` on **all 47 cases**, because
    that digest is taken over the whole metrics dict. A key nothing had before
    is a digest change for everything, whatever its value. The measurement is
    unchanged; only where the bound is looked up moved.

    Every policy state is handled explicitly and anything else raises: a new
    state falling through to a permissive default is how a gate stops applying
    without anyone deciding it should.
    """

    if policy == "free":
        return None
    if policy == "bounded":
        return float(value_of("physics.strike_root_drift_max_m"))
    if policy == "fixed":
        return float(value_of("physics.root_drift_max_m"))
    raise ValueError(f"unhandled root-drift policy: {policy!r}")


def clip_contract_violations(
    metrics: dict[str, Any], *, policy: RootDriftPolicy
) -> int:
    """What the compiler's clip-level gates have always actually meant.

    Every gate in ``compiler.py`` that reads ``joint_limit_violations`` is asking
    one question -- *did this clip break a whole-clip contract* -- and until this
    module stopped folding root drift into that counter, the answer happened to
    be spelled with a joint-limit name. Ten sites read it. Unfolding the counter
    without giving them this leaves all ten quietly not rejecting a drifted
    clip, which is the failure mode plan 04's ordering constraint exists to
    prevent, one quantity over.

    So the counter now means only joint limits and the gates read this instead.
    The value is unchanged for every clip: ``joint_limit_violations`` is 0 on all
    47 corpus cases and root drift has never exceeded its bound on any of them,
    so this is a rename with a gate behind it rather than a behaviour change.
    """

    limit = root_drift_limit_m(policy=policy)
    drifted = limit is not None and float(metrics.get("root_drift_m", 0.0)) > limit
    return int(metrics.get("joint_limit_violations", 0)) + int(drifted)


def _root_drift_check(
    metrics: dict[str, Any], *, policy: RootDriftPolicy
) -> CheckResult:
    """The root-drift verdict, addressable under its own name.

    A program whose policy is ``free`` has no bound to be judged against, so
    that is a **skip** rather than a pass. Reporting it as a pass would put a
    verdict in the output that is not derived from what it claims to describe
    -- the not-measured / measured-negative conflation four lanes hit
    separately in this push, and the same distinction ``rom_checks`` makes for
    a bone the clip never posed. ``bounded`` is NOT a skip: there is a real
    bound and this is the gate that applies it, which is the entire point of
    the third state -- strikes keep a whole-clip root gate while the pelvis is
    allowed to move inside the stated envelope.
    """

    limit = root_drift_limit_m(policy=policy)
    if limit is None:
        return skipped(
            "contract.clip.root_drift",
            CONTRACT,
            detail="the program enables root motion; there is no bound to apply",
        )
    return upper_bound_check(
        "contract.clip.root_drift",
        CONTRACT,
        float(metrics.get("root_drift_m", 0.0)),
        limit,
        detail=(
            "the root travelled further than this intent's bounded envelope permits"
            if policy == "bounded"
            else "the root travelled further than a fixed-root clip permits"
        ),
    )


def safety_checks(
    metrics: dict[str, Any], *, policy: RootDriftPolicy = "fixed"
) -> list[CheckResult]:
    """Report the safety metrics as individually addressable checks.

    ``policy`` is :func:`rigby_poc.analysis.context.root_drift_policy` of the
    program the clip compiled from; it decides which root-drift bound applies,
    if any. It defaults to ``fixed`` -- the strictest state and the one every
    path except strike, whole-body and sequence uses -- because a wrong default
    here silences a gate rather than raising.
    """

    return [
        count_check(
            "contract.clip.non_finite_transforms",
            CONTRACT,
            int(metrics.get("nan_count", 0)),
            scale=1,
            detail="clip contains non-finite transforms",
        ),
        count_check(
            "contract.clip.joint_limit_violations",
            CONTRACT,
            int(metrics.get("joint_limit_violations", 0)),
            detail="clip exceeds a joint limit",
        ),
        _root_drift_check(metrics, policy=policy),
        count_check(
            "contract.clip.rotational_discontinuities",
            CONTRACT,
            int(metrics.get("discontinuities", 0)),
            detail="clip contains rotational discontinuities",
        ),
    ]
