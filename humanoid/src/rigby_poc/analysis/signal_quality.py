"""Plan 10 §3.2 — the signal-quality layer (L3).

The quantitative naturalness proxies: the part usually assumed to need a model.
§3.2's claim is that ``min_jerk`` and ``dead_limb`` together will likely catch
more real "looks wrong" cases than a VLM does, at zero marginal cost and with a
number that can be regression-tested.

**Named ``signal_quality`` rather than ``signal``** so the module never has to be
reasoned about next to the standard library's ``signal``. Absolute imports make
the shadowing harmless in fact, but "harmless once you check" is a cost paid by
every future reader.

**Ids are three segments, and that is a correction to §3.2 rather than a
preference.** The section writes ``signal.min_jerk``, ``signal.sparc`` and so on,
two segments each. Every check id in this repository has at least three, and the
composite's stratum key is the first *two* -- so a two-segment id becomes its own
single-member family. Seven of them would make ``signal.min_jerk`` alone weigh as
much as all three ``signal.angular.*`` checks combined in the fold, purely as an
artefact of how it was written down. Grouped here so the strata mean something:
``signal.smoothness`` holds both of the smoothness measures, which are two views
of one property.

**This layer reads frames, never ``metrics``** -- the same rule as the physics
layer, and for the same reason: ``MutationSpec.apply`` transforms frames only, so
a metric-derived check scored against a mutated clip reads the unmutated one.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import ClipFrame
from ..thresholds import value_of
from .contract import SIGNAL, CheckResult, lower_bound_check, skipped
from .physics import world_positions_of

#: End effectors whose speed profile the smoothness measures read.
END_EFFECTORS = ("leftHand", "rightHand")

#: Frames below which a speed profile carries no shape worth correlating.
#: A four-frame clip has two speed samples and one acceleration sample; a
#: correlation against a bell over two points is 1.0 or -1.0 by construction and
#: says nothing. Ten is the smallest window in which the ideal profile has a
#: distinguishable rise, peak and fall at this corpus's frame rates.
MIN_PROFILE_FRAMES = 10

#: Total travel below which an end effector is treated as not having moved, in
#: metres. A hand that drifts 3 mm over a clip has a speed profile made of
#: numerical noise, and correlating noise against a bell produces a number with
#: the shape of a measurement and none of the content.
MIN_TRAVEL_M = 0.02


class SignalError(ValueError):
    """The signal layer was asked for something the clip cannot support."""


def speed_profile(
    world: list[dict[str, np.ndarray]], bone: str, *, fps: float
) -> np.ndarray:
    """Per-frame scalar speed of one joint, in metres per second.

    Forward differences, so the result is one shorter than the frame count.
    """

    if fps <= 0.0:
        raise SignalError(f"a speed profile needs a positive frame rate, got {fps}")
    missing = [index for index, pose in enumerate(world) if bone not in pose]
    if missing:
        raise SignalError(
            f"cannot build a speed profile: {bone!r} is missing from "
            f"{len(missing)} of {len(world)} frames"
        )
    if len(world) < 2:
        return np.zeros(0, dtype=float)
    track = np.asarray([np.asarray(pose[bone], dtype=float) for pose in world])
    return np.linalg.norm(np.diff(track, axis=0), axis=1) * fps


def minimum_jerk_profile(samples: int) -> np.ndarray:
    """The ideal speed profile for a point-to-point reach, normalised to peak 1.

    Flash and Hogan (1985): minimising jerk over a fixed duration gives the
    position ``x(t) = 10τ³ - 15τ⁴ + 6τ⁵`` for ``τ = t/T``, whose derivative is
    the symmetric bell ``30τ²(1-τ)²``. This is the shape a human reach follows
    and the shape constant-velocity interpolation conspicuously does not -- which
    §3.2 names as "the biggest tell of procedural motion".
    """

    if samples < 2:
        raise SignalError(
            f"a minimum-jerk profile needs at least 2 samples, got {samples}"
        )
    tau = np.linspace(0.0, 1.0, samples)
    bell = 30.0 * tau**2 * (1.0 - tau) ** 2
    peak = float(np.max(bell))
    return bell / peak if peak > 0.0 else bell


def bell_correlation(speed: np.ndarray) -> float:
    """Pearson correlation of a speed profile against the minimum-jerk bell.

    1.0 is a textbook reach. A constant-velocity interpolation has zero variance
    in its speed, so the correlation is undefined rather than low -- that case is
    reported by :func:`is_flat` and must not be folded in here as a number.
    """

    values = np.asarray(speed, dtype=float)
    if values.size < 2:
        raise SignalError("a correlation needs at least two speed samples")
    ideal = minimum_jerk_profile(values.size)
    if float(np.std(values)) <= 0.0 or float(np.std(ideal)) <= 0.0:
        raise SignalError(
            "a flat speed profile has no correlation with anything; call is_flat first"
        )
    return float(np.corrcoef(values, ideal)[0, 1])


def is_flat(speed: np.ndarray, *, relative_tolerance: float = 1e-6) -> bool:
    """Whether a speed profile has no variation to correlate.

    Constant-velocity interpolation lands here, and it is the single most
    diagnostic case in §3.2 -- so it is a named state rather than a correlation
    of zero, which is what a naive implementation returns for it and which is
    indistinguishable from an ordinary bad reach.
    """

    values = np.asarray(speed, dtype=float)
    if values.size < 2:
        return True
    scale = float(np.mean(np.abs(values)))
    return float(np.std(values)) <= relative_tolerance * max(scale, 1e-12)


def spectral_arc_length(
    speed: np.ndarray, *, fps: float, cutoff_hz: float = 10.0, padding: int = 4
) -> float:
    """SPARC: arc length of the normalised speed spectrum. Less negative is smoother.

    Balasubramanian et al. (2015), *On the analysis of movement smoothness*. The
    measure is duration- and amplitude-robust by construction -- the spectrum is
    normalised by its own DC magnitude and the arc length is taken over a fixed
    frequency band -- which is why §3.2 picks it over a jerk integral, whose value
    depends on how long the movement took.

    Typical values: about -1.5 for a single smooth reach, more negative as
    submovements and jitter add spectral content.
    """

    values = np.asarray(speed, dtype=float)
    if values.size < 2:
        raise SignalError("SPARC needs at least two speed samples")
    if fps <= 0.0:
        raise SignalError(f"SPARC needs a positive frame rate, got {fps}")
    length = int(2 ** (np.ceil(np.log2(values.size)) + padding))
    spectrum = np.abs(np.fft.rfft(values, n=length))
    peak = float(spectrum[0]) if spectrum.size else 0.0
    if peak <= 0.0:
        raise SignalError("a speed profile with zero mean has no normalised spectrum")
    spectrum = spectrum / peak
    freqs = np.fft.rfftfreq(length, d=1.0 / fps)
    band = freqs <= cutoff_hz
    if int(np.count_nonzero(band)) < 2:
        raise SignalError(
            f"the {cutoff_hz} Hz band holds fewer than two spectral bins at "
            f"{fps} fps; SPARC is not defined here"
        )
    magnitudes = spectrum[band]
    # The frequency axis is normalised across the band, so the step is
    # 1/(bins-1) and the arc length is dimensionless. Using 1/cutoff_hz here
    # instead -- which is dimensionally plausible and was the first thing
    # written -- makes the sum scale with the bin count: a smooth bell scored
    # -34.3 against its true -1.5, and adding heavy jitter moved it by 0.03.
    # It ran, returned a float, and ordered two profiles the right way round.
    dx = 1.0 / float(magnitudes.size - 1)
    steps = np.diff(magnitudes)
    return -float(np.sum(np.sqrt(dx**2 + steps**2)))


def bone_activity(frames: list[ClipFrame], bones: tuple[str, ...]) -> dict[str, float]:
    """Total rotational travel of each bone over the clip, in radians.

    The sum of per-frame rotation magnitudes rather than the variance of the
    quaternion: a quaternion's components are not a linear space, so their
    variance is not a rotation and would rank a bone oscillating about its rest
    pose as still. Summed geodesic steps is the rotation the bone actually
    performed.
    """

    if not frames:
        return dict.fromkeys(bones, 0.0)
    totals: dict[str, float] = {}
    for bone in bones:
        present = [frame for frame in frames if bone in frame.bones]
        if len(present) < 2:
            totals[bone] = 0.0
            continue
        rotations = Rotation.from_quat(
            [frame.bones[bone].rotation.as_list() for frame in present]
        )
        steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
        totals[bone] = float(np.sum(steps))
    return totals


def active_end_effector(
    world: list[dict[str, np.ndarray]], *, fps: float
) -> tuple[str, np.ndarray] | None:
    """The hand that moved furthest, with its speed profile, or ``None``.

    A clip where neither hand travels :data:`MIN_TRAVEL_M` has no end-effector
    movement to assess, and the honest answer is that nothing was measured. A
    smoothness score computed from sub-millimetre numerical noise is the
    absence-becoming-a-value shape with extra arithmetic.
    """

    best: tuple[str, np.ndarray] | None = None
    best_travel = 0.0
    for bone in END_EFFECTORS:
        if any(bone not in pose for pose in world):
            continue
        speed = speed_profile(world, bone, fps=fps)
        if speed.size < MIN_PROFILE_FRAMES:
            continue
        travel = float(np.sum(speed)) / fps
        if travel >= MIN_TRAVEL_M and travel > best_travel:
            best, best_travel = (bone, speed), travel
    return best


#: The four limb chains ``signal.activity.dead_limb`` watches.
LIMB_CHAINS: dict[str, tuple[str, ...]] = {
    "left_arm": ("leftUpperArm", "leftLowerArm", "leftHand"),
    "right_arm": ("rightUpperArm", "rightLowerArm", "rightHand"),
    "left_leg": ("leftUpperLeg", "leftLowerLeg", "leftFoot"),
    "right_leg": ("rightUpperLeg", "rightLowerLeg", "rightFoot"),
}

#: Total limb rotation below which the whole clip is treated as having no motion
#: to assess, in radians. A clip where every limb is still is not a clip with a
#: dead limb; it is a clip with nothing happening, and the honest verdict is a
#: skip.
MIN_BODY_ACTIVITY_RAD = 0.10


def dead_limb_check(frames: list[ClipFrame], *, threshold_rad: float) -> CheckResult:
    """No limb chain is frozen while the body is moving. Plan 10 §3.2.

    §3.2 calls this "very cheap, very high yield" and it is both. Measured over
    the corpus it fails **39 of 46** compiling cases, and the shape is specific:
    it is **always both legs and never an arm**. The compiler animates the upper
    body only on gesture, composite, strike and object paths, leaving the legs at
    exactly their rest rotation for the whole clip.

    That is reported at 39 of 46 rather than tuned away. The criterion is right
    and the subject is bad, which is the case plan 10 §10.5 says to expect and
    §Common forbids tuning around.
    """

    if not frames:
        return skipped(
            "signal.activity.dead_limb",
            SIGNAL,
            detail="the clip has no frames, so no limb was measured",
        )
    bones = tuple(bone for chain in LIMB_CHAINS.values() for bone in chain)
    activity = bone_activity(frames, bones)
    totals = {
        name: sum(activity[bone] for bone in chain)
        for name, chain in LIMB_CHAINS.items()
    }
    if sum(totals.values()) < MIN_BODY_ACTIVITY_RAD:
        return skipped(
            "signal.activity.dead_limb",
            SIGNAL,
            detail=(
                "no limb moves in this clip, so there is no active body against "
                "which a limb could be dead"
            ),
        )
    quietest = min(totals, key=lambda name: totals[name])
    return lower_bound_check(
        "signal.activity.dead_limb",
        SIGNAL,
        float(totals[quietest]),
        threshold_rad,
        scale=threshold_rad,
        detail=f"{quietest} does not move while the rest of the body does",
    )


def sparc_check(
    world: list[dict[str, np.ndarray]], *, fps: float, floor: float
) -> CheckResult:
    """Spectral arc length of the active end effector's speed profile. §3.2.

    **This is an outlier bound, not a naturalness gate, and the distinction is
    the whole of what can honestly be claimed.** SPARC itself is validated -- it
    scores a minimum-jerk bell at -1.97 against the literature's ~-1.5, degrades
    monotonically with added jitter, and scores two submovements worse than one.
    What is *not* validated is any particular value meaning "unnatural", because
    this corpus carries no smoothness ground truth: measured against the only
    labels it has, SPARC separates structurally-valid from known-bad clips at
    **AUC 0.546 against a 0.500 baseline, n = 205 pairs** -- chance.

    So the bound is set at the corpus's own Tukey lower fence rather than at a
    quality claim, in the same spirit as the 90 s CI rot ceiling in
    ``docs/testing.md``: it is not the budget, it is there to catch something
    far outside anything seen so far. One case sits below it today.
    """

    if not world:
        return skipped(
            "signal.smoothness.sparc",
            SIGNAL,
            detail="the clip has no frames, so there is no speed profile",
        )
    picked = active_end_effector(world, fps=fps)
    if picked is None:
        return skipped(
            "signal.smoothness.sparc",
            SIGNAL,
            detail=(
                f"neither hand travels {MIN_TRAVEL_M} m, so the speed profile is "
                "numerical noise rather than a movement"
            ),
        )
    bone, speed = picked
    try:
        value = spectral_arc_length(speed, fps=fps)
    except SignalError as error:
        return skipped("signal.smoothness.sparc", SIGNAL, detail=str(error))
    return lower_bound_check(
        "signal.smoothness.sparc",
        SIGNAL,
        value,
        floor,
        scale=abs(floor),
        detail=f"{bone} speed profile is far less smooth than any measured clip",
    )


def signal_checks(frames: list[ClipFrame], *, fps: float) -> list[CheckResult]:
    """Every signal-quality verdict for a clip. Plan 10 §3.2.

    Always emits every id it owns, skipping where it measured nothing -- the
    rule ``rom_checks`` and ``physics_checks`` follow, and the one the mutation
    registry's emission counts depend on.

    **Two of §3.2's seven ship.** ``min_jerk`` is deferred because its criterion
    is a *single point-to-point reach* and these clips are multi-submovement:
    correlating a whole clip against one bell gives a negative correlation on 32
    of 46 cases, and longer clips trend *less* negative, which is averaging
    rather than signal. The other four need event or gait structure the corpus
    does not carry -- see plan 10 §3.2.
    """

    world = world_positions_of(frames)
    return [
        dead_limb_check(
            frames, threshold_rad=float(value_of("signal.dead_limb_min_rotation_rad"))
        ),
        sparc_check(world, fps=fps, floor=float(value_of("signal.sparc_floor"))),
    ]


__all__ = [
    "END_EFFECTORS",
    "LIMB_CHAINS",
    "MIN_BODY_ACTIVITY_RAD",
    "MIN_PROFILE_FRAMES",
    "MIN_TRAVEL_M",
    "SignalError",
    "active_end_effector",
    "bell_correlation",
    "bone_activity",
    "dead_limb_check",
    "is_flat",
    "minimum_jerk_profile",
    "signal_checks",
    "sparc_check",
    "spectral_arc_length",
    "speed_profile",
    "world_positions_of",
]
