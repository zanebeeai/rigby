"""Plan 10 §3.2's primitives, validated against the literature rather than themselves.

Everything here is synthetic — analytic profiles, no compile, no corpus — so it
is the ``fast`` tier and says nothing about any real clip.

**The SPARC tests exist because the first implementation was wrong and passed
every internal check.** It ran, returned a float, and ordered a smooth bell
above a jittered one — monotonic, plausible, and off by a factor of 17, because
the frequency step used ``1/cutoff_hz`` where SPARC normalises the axis across
the band. Nothing internal to the code could have caught that: it needed an
external number to compare against. So the assertions below are anchored to
published magnitudes, not to the implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.analysis.signal_quality import (
    MIN_TRAVEL_M,
    SignalError,
    active_end_effector,
    bell_correlation,
    bone_activity,
    is_flat,
    minimum_jerk_profile,
    spectral_arc_length,
    speed_profile,
)
from rigby_poc.models import BonePose, ClipFrame, Quat

pytestmark = pytest.mark.fast

FPS = 30.0


def _rotated_frames(
    angles_rad: list[float], bone: str = "leftUpperArm"
) -> list[ClipFrame]:
    frames = []
    for index, angle in enumerate(angles_rad):
        half = angle / 2.0
        frames.append(
            ClipFrame(
                time_s=index / FPS,
                bones={
                    bone: BonePose(
                        rotation=Quat(x=np.sin(half), y=0.0, z=0.0, w=np.cos(half))
                    )
                },
            )
        )
    return frames


# -- the minimum-jerk ideal -------------------------------------------------------------


def test_the_minimum_jerk_profile_is_the_symmetric_bell_flash_and_hogan_derive() -> (
    None
):
    """``30τ²(1-τ)²``: zero at both ends, peak at the midpoint, symmetric.

    Checked against the closed form rather than against the code: this is the
    shape a human reach follows, and it is the shape constant-velocity
    interpolation conspicuously does not.
    """

    bell = minimum_jerk_profile(101)
    assert bell[0] == pytest.approx(0.0)
    assert bell[-1] == pytest.approx(0.0)
    assert int(np.argmax(bell)) == 50
    assert float(np.max(bell)) == pytest.approx(1.0)
    assert bell == pytest.approx(bell[::-1], abs=1e-12)


def test_a_profile_correlates_perfectly_with_itself_and_not_with_a_ramp() -> None:
    bell = minimum_jerk_profile(60)
    assert bell_correlation(bell) == pytest.approx(1.0)
    ramp = np.linspace(0.0, 1.0, 60)
    assert bell_correlation(ramp) < 0.2


def test_a_constant_velocity_profile_is_flat_rather_than_uncorrelated() -> None:
    """The single most diagnostic case in §3.2, and it must not become a number.

    A naive correlation returns 0.0 for a flat profile — or a divide-by-zero
    warning and a nan — and 0.0 is indistinguishable from an ordinary bad reach.
    "Constant velocity" is a named state here, which is what makes it reportable.
    """

    flat = np.full(60, 2.5)
    assert is_flat(flat)
    with pytest.raises(SignalError, match="flat speed profile"):
        bell_correlation(flat)
    assert not is_flat(minimum_jerk_profile(60))


# -- SPARC, anchored to published magnitudes ---------------------------------------------


def test_sparc_scores_a_minimum_jerk_bell_near_the_published_value() -> None:
    """The assertion that caught a factor-of-17 bug the code could not see.

    Balasubramanian et al. (2015) report about -1.5 for a single smooth reach.
    The bound here is deliberately loose — discretisation and zero-padding move
    it — but it is an *external* anchor, and the broken implementation scored
    -34.277 against it.
    """

    value = spectral_arc_length(minimum_jerk_profile(60), fps=FPS)
    assert -3.0 < value < -1.0, f"a smooth bell scored {value:.3f}"


def test_sparc_degrades_monotonically_as_jitter_is_added() -> None:
    """Ordering alone is not enough — the broken version also ordered correctly.

    So this asserts ordering *and* that the effect is large enough to be a
    measurement. The broken implementation moved 0.03 across the whole sweep
    while ordering the ends correctly.

    **Averaged over seeds, and that is a finding rather than defensiveness.**
    Written first with one seed and a 0.05 step, this test failed: σ=0.05 scored
    -1.958 against σ=0's -1.965, i.e. slightly *smoother*. The step is real but
    smaller than the between-draw variance of a 60-sample profile, so a
    single-draw assertion at that spacing is a coin flip — the same shape as
    asserting a timing ratio inside its own noise. Ten seeds, and the smallest
    step kept above the noise floor.
    """

    bell = minimum_jerk_profile(60)
    levels = (0.0, 0.10, 0.20, 0.30)
    means = []
    for sd in levels:
        rng = np.random.default_rng(20260825)
        samples = [
            spectral_arc_length(bell + rng.normal(0.0, sd, 60), fps=FPS)
            for _ in range(10)
        ]
        means.append(float(np.mean(samples)))

    assert means == sorted(means, reverse=True), means
    assert means[0] - means[-1] > 0.5, (
        f"jitter moved mean SPARC by only {means[0] - means[-1]:.3f}; the broken "
        "implementation moved it by 0.03 while ordering correctly"
    )


def test_sparc_scores_two_submovements_worse_than_one() -> None:
    """Submovement count is the property SPARC is built to expose."""

    tau = np.linspace(0.0, 1.0, 60)
    single = np.exp(-(((tau - 0.5) / 0.12) ** 2))
    double = np.exp(-(((tau - 0.3) / 0.08) ** 2)) + np.exp(-(((tau - 0.7) / 0.08) ** 2))
    assert spectral_arc_length(double, fps=FPS) < spectral_arc_length(single, fps=FPS)


def test_sparc_refuses_a_profile_it_cannot_normalise() -> None:
    with pytest.raises(SignalError, match="zero mean"):
        spectral_arc_length(np.zeros(60), fps=FPS)
    with pytest.raises(SignalError, match="positive frame rate"):
        spectral_arc_length(minimum_jerk_profile(60), fps=0.0)


# -- speed and activity ------------------------------------------------------------------


def test_a_joint_moving_at_a_constant_rate_has_a_constant_speed() -> None:
    world = [
        {"leftHand": np.asarray([i * 0.01, 0.0, 0.0], dtype=float)} for i in range(30)
    ]
    speed = speed_profile(world, "leftHand", fps=FPS)
    assert speed.size == 29
    assert speed == pytest.approx(np.full(29, 0.01 * FPS))
    assert is_flat(speed)


def test_a_missing_joint_raises_rather_than_yielding_a_short_profile() -> None:
    world = [{"leftHand": np.zeros(3)} for _ in range(10)]
    del world[4]["leftHand"]
    with pytest.raises(SignalError, match="leftHand"):
        speed_profile(world, "leftHand", fps=FPS)


def test_bone_activity_measures_rotation_travelled_not_quaternion_spread() -> None:
    """A bone oscillating about its rest pose is moving, and must not read as still.

    Quaternion components are not a linear space, so their variance is not a
    rotation. A bone swinging symmetrically about rest has near-zero component
    variance and a large geodesic total — and it is the case ``dead_limb`` most
    needs to get right, since a limb that jitters in place is not frozen.
    """

    swing = [0.4 * np.sin(i * 0.5) for i in range(40)]
    frames = _rotated_frames(swing)
    total = bone_activity(frames, ("leftUpperArm",))["leftUpperArm"]
    assert total > 1.0, f"an oscillating bone accumulated only {total:.4f} rad"

    still = bone_activity(_rotated_frames([0.0] * 40), ("leftUpperArm",))[
        "leftUpperArm"
    ]
    assert still == pytest.approx(0.0, abs=1e-9)


def test_bone_activity_reports_zero_for_a_bone_the_clip_never_poses() -> None:
    assert (
        bone_activity(_rotated_frames([0.1, 0.2]), ("rightFoot",))["rightFoot"] == 0.0
    )
    assert bone_activity([], ("leftUpperArm",)) == {"leftUpperArm": 0.0}


def test_a_hand_that_barely_moves_is_not_an_end_effector() -> None:
    """Correlating numerical noise against a bell produces a shaped non-measurement."""

    still = [
        {"leftHand": np.asarray([i * 1e-5, 0.0, 0.0]), "rightHand": np.zeros(3)}
        for i in range(40)
    ]
    assert active_end_effector(still, fps=FPS) is None

    moving = [
        {"leftHand": np.asarray([i * 0.01, 0.0, 0.0]), "rightHand": np.zeros(3)}
        for i in range(40)
    ]
    picked = active_end_effector(moving, fps=FPS)
    assert picked is not None and picked[0] == "leftHand"
    assert float(np.sum(picked[1])) / FPS >= MIN_TRAVEL_M
