"""Synthetic ground truth: graded, deterministic perturbations of compiled clips.

Under a no-human-evaluation policy this is the primary source of labelled data for
the whole suite (plan 06 section 1.1).  Start at :class:`~evals.mutations.spec.MutationSpec`,
whose docstring carries the four contract rules -- each written because breaking it
produces a plausible, well-formed, monotonic result that measures the harness rather
than the check.
"""

from __future__ import annotations

from .anatomy import hinge_off_axis_sweep, is_hinge, rom_sweep
from .checks import known_check_ids, require_known_targets, rom_detected
from .clipping import limb_through_torso_sweep
from .compose import compose, directional_pair, strata_b_pairs
from .family import MutationFamily, Tier
from .inject import add_dof, bone_dof_series, is_static, peak_dof, signed_magnitude
from .legacy import legacy_spec, legacy_specs
from .signal import jitter_sweep
from .timing import freeze_sweep, snap_sweep
from .spec import (
    APPLICABLE,
    Applicability,
    MutationSpec,
    NotApplicable,
    Severity,
)
from .sweep import (
    DEFAULT_LEVELS,
    degrees_sweep,
    families_covered,
    subperceptual_floor,
    sweep,
    tier_for,
)

__all__ = [
    "APPLICABLE",
    "DEFAULT_LEVELS",
    "Applicability",
    "MutationFamily",
    "MutationSpec",
    "NotApplicable",
    "Severity",
    "Tier",
    "add_dof",
    "bone_dof_series",
    "compose",
    "degrees_sweep",
    "freeze_sweep",
    "hinge_off_axis_sweep",
    "is_hinge",
    "jitter_sweep",
    "known_check_ids",
    "limb_through_torso_sweep",
    "directional_pair",
    "families_covered",
    "is_static",
    "legacy_spec",
    "legacy_specs",
    "peak_dof",
    "require_known_targets",
    "rom_detected",
    "rom_sweep",
    "signed_magnitude",
    "snap_sweep",
    "strata_b_pairs",
    "subperceptual_floor",
    "sweep",
    "tier_for",
]
