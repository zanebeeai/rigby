"""Turn a simulated rollout into something small enough to ship and scrub.

A certified run is simulated at 240 Hz. Sending that is pointless -- nobody
watches a 240 Hz playback, and the studio would carry tens of megabytes of joint
angles nobody looks at. What a viewer needs is enough samples that the motion
reads as continuous and the timeline scrubs smoothly.

Sampling is uniform in *time* rather than every Nth step, so a track keeps its
real duration and a pause at 4.2 seconds lands where the robot actually was at
4.2 seconds. The final frame is always included: the end of a motion is the
frame most worth stopping on, and dropping it to keep the stride even would trim
exactly the pose the gates were judging.
"""

from __future__ import annotations

import numpy as np


DEFAULT_MAX_FRAMES = 120


def sample_track(
    times_s,
    qpos,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    decimals: int = 5,
) -> dict:
    """Uniformly resample a rollout down to at most ``max_frames`` poses."""

    times = np.asarray(times_s, dtype=float).reshape(-1)
    values = np.asarray(qpos, dtype=float)
    if times.size == 0 or values.size == 0:  # pragma: no cover - defensive
        return {"times": [], "qpos": [], "nq": 0, "duration_s": 0.0}
    if values.ndim == 1:
        values = values.reshape(1, -1)

    count = min(int(max_frames), int(times.size))
    if count <= 1:
        picked = np.array([times.size - 1])
    else:
        wanted = np.linspace(times[0], times[-1], count)
        picked = np.unique(np.searchsorted(times, wanted).clip(0, times.size - 1))
        if picked[-1] != times.size - 1:
            picked = np.append(picked, times.size - 1)

    sampled_times = times[picked]
    sampled_qpos = values[picked]
    return {
        "nq": int(sampled_qpos.shape[1]),
        "frames": int(sampled_qpos.shape[0]),
        "duration_s": round(float(times[-1]), 4),
        "times": [round(float(v), 4) for v in sampled_times],
        # Flattened: frames * nq, read row-major by the viewer. A list of lists
        # roughly doubles the JSON size for no gain.
        "qpos": [round(float(v), decimals) for v in sampled_qpos.reshape(-1)],
    }
