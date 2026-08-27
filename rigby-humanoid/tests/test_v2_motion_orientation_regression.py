from __future__ import annotations

import numpy as np
import pytest

from rigby_v2.motion.refinement import _orientation_error

pytestmark = pytest.mark.fast


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )


def test_orientation_residual_does_not_vanish_at_180_degrees() -> None:
    residual = _orientation_error(np.eye(3), _rotation_x(np.pi))

    assert np.linalg.norm(residual) == pytest.approx(np.pi)
