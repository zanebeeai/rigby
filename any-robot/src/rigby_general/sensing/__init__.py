"""Declared sensors over recorded physics, and the labels they are judged by.

A recorded episode holds every physical state a run passed through. What a
monitor may see of it is what its declared sensors report: joint encoders,
contact force on the gripper's members, a camera that reports an object's
position when its rays reach the object and reports occlusion when they hit
something else, the task's own geometry and the manipulator's calibrated
reach. The oracle labeler reads the full state and never feeds the
evaluator; it is what the evaluator's verdicts are scored against.
"""

from .bind import conditionals_for, decide, load_policy, policy_digest
from .corrupt import absent, occluded_by_screen, sparse, stale
from .episode import Episode
from .labels import OracleLabeler
from .sensors import CONFIGURATIONS, CameraSpec, EvidenceStreams, configuration

__all__ = ["CONFIGURATIONS", "CameraSpec", "Episode", "EvidenceStreams", "OracleLabeler", "absent", "conditionals_for", "configuration", "decide",
           "load_policy", "occluded_by_screen", "policy_digest", "sparse", "stale"]
