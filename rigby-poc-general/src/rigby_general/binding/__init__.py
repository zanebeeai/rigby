"""Bind body-neutral schema programs to a robot's certified primitives."""

from .binder import BoundMotion, SegmentBinding, UnbindableSegment, bind, coverage_report

__all__ = [
    "BoundMotion",
    "SegmentBinding",
    "UnbindableSegment",
    "bind",
    "coverage_report",
]
