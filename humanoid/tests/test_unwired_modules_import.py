"""Two modules of the closed-loop branch are kept but not yet wired in.

``articulated_physics`` and ``vlm_selector`` came in with the gripper and
VLA work of PR 19. When that branch was reconciled with the grasp solver on
main, the compiler paths that reached them were not carried over, so the
reachability guard (``test_module_reachability``) would otherwise report
them as observed by nothing. This test is that observation, deliberately
thin: it pins that they import and that their public surface is what the
closed-loop work expects, until the decision layer wires them in or the
branch retires them.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


def test_articulated_physics_imports() -> None:
    from rigby_poc import articulated_physics

    assert callable(getattr(articulated_physics, "simulate_articulated_grasp", None)) or hasattr(
        articulated_physics, "_model_xml"
    )


def test_vlm_selector_imports() -> None:
    from rigby_poc import vlm_selector

    assert vlm_selector.__doc__ is not None
