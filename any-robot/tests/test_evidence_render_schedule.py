"""A long episode must survive video sampling as a full-duration episode."""

import numpy as np
import pytest

from rigby_general.evidence.render import frame_schedule


def test_long_episode_has_no_ninety_frame_cap_and_keeps_the_last_state():
    times = np.arange(15001, dtype=float) * 0.002
    indices, requested = frame_schedule(times, fps=12)
    assert len(indices) == 361
    assert indices[0] == 0 and indices[-1] == len(times) - 1
    assert requested[-1] == times[-1]
    assert np.max(requested - times[indices]) <= 0.002 + 1e-10
    assert 30 <= len(indices) / 12 <= 30 + 2 / 12


def test_reference_clock_cannot_replace_physical_clock():
    # Existing arm simulation advances its reference at 240 Hz but steps this
    # model at 500 Hz. Rendering must use recorded physics time, not the plan.
    physical = np.arange(7323, dtype=float) * 0.002
    indices, _ = frame_schedule(physical, fps=12)
    assert len(indices) == 177
    assert len(indices) / 12 < 15.0
    assert physical[-1] == pytest.approx(14.644)


def test_nonzero_initial_time_and_refusal_slate():
    indices, times = frame_schedule(np.array([12.0, 12.1, 12.2]), fps=10)
    assert indices[0] == 0 and indices[-1] == 2
    assert times[0] == 12.0 and times[-1] == 12.2
    indices, times = frame_schedule(np.array([0.0]), fps=12)
    assert len(indices) == 24 and not indices.any() and not times.any()


@pytest.mark.parametrize("times,fps", [([], 12), ([0.0, 0.0], 12), ([1.0, 0.0], 12), ([float('nan')], 12), ([0.0], 0)])
def test_invalid_schedule_is_refused(times, fps):
    with pytest.raises(ValueError):
        frame_schedule(np.asarray(times), fps)
