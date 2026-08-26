"""Model geometry and joint tracks, shaped for a browser."""

from .scene import build_scene, forward_kinematics
from .track import sample_track

__all__ = ["build_scene", "forward_kinematics", "sample_track"]
