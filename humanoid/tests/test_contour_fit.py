"""The search that fits primitives to a Talmy contour.

Driven by a synthetic compile function with a known optimum rather than the real
compiler, so these assert properties of the *search* and run in milliseconds. A
test that ran MuJoCo would be measuring the grasp instead, and would take
minutes to tell you less.

The properties that matter are safety properties, not convergence ones. A search
that sometimes fails to improve is disappointing; a search that returns
something worse than it was given, or spends past its budget, or reports
exhaustion as convergence, is unusable.
"""

from __future__ import annotations

import pytest

from rigby_poc.contour_fit import (
    GRAB_KNOBS,
    Knob,
    _current_value,
    _with_value,
    fit_to_contour,
)
from rigby_poc.models import (
    ClipFrame,
    ClipResult,
    PlanRequest,
    PrimitiveKind,
    Provenance,
    Transform,
    Vec3,
    default_scene,
)
from rigby_poc.planner import plan_motion

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast


def _clip(frames) -> ClipResult:
    """A minimal valid ClipResult. Only ``frames`` is read by the objective."""
    return ClipResult(
        success=True,
        fps=30,
        duration_s=1.0,
        frames=frames,
        metrics={},
        slider_observables={},
        parametric_observables={},
        provenance=Provenance(
            rig_id="test", rig_asset="test", compiler_version="test",
            physics_engine="none", physics_version="0", planner_provider="offline",
            planner_model="none", model_calls=0, seed=0,
            physics_model={}, coordinate_frames={},
        ),
    )


def _program(text: str = "pick up box"):
    scene = default_scene()
    return scene, plan_motion(
        PlanRequest(text=text, scene=scene, provider="offline")
    ).program


def _synthetic_compiler(scene, target_lift: float, calls: list):
    """A compile whose object rises by whatever ``lift_height_m`` asks for.

    The optimum is therefore known exactly: the loss is minimised when
    ``lift_height_m`` equals the contour's specified 0.12 m rise.
    """
    block = scene.object_by_id("block")
    start = block.transform.translation

    def compile_fn(program):
        calls.append(program)
        lift = next(
            p.parameters.lift_height_m
            for p in program.primitives
            if p.kind == PrimitiveKind.LIFT
        )
        frames = []
        steps = 21
        for index in range(steps):
            alpha = index / (steps - 1)
            frames.append(
                ClipFrame(
                    time_s=alpha,
                    bones={},
                    objects={
                        "block": Transform(
                            translation=Vec3(
                                x=start.x, y=start.y + lift * alpha, z=start.z
                            )
                        )
                    },
                )
            )
        return _clip(frames)

    return compile_fn


def test_the_search_finds_the_known_optimum() -> None:
    """Started deliberately wrong, because the planner's default is already right.

    The offline planner authors a 0.10 m lift and the contour specifies 0.12 m,
    so a search started from the default has almost nothing to find and would
    pass this test without demonstrating anything. Starting at 0.50 m gives the
    search a real distance to close.
    """
    scene, program = _program()
    program = _with_value(
        program,
        Knob("lift", "lift_height_m",
             (PrimitiveKind.LIFT, PrimitiveKind.HOLD, PrimitiveKind.RECOVER),
             0.04, 0.0, 0.8),
        0.50,
    )
    calls: list = []
    result = fit_to_contour(
        program, scene, _synthetic_compiler(scene, 0.12, calls), budget=60
    )
    assert result is not None
    assert result.improved
    assert result.loss < result.baseline_loss
    fitted = next(
        p.parameters.lift_height_m
        for p in result.program.primitives
        if p.kind == PrimitiveKind.LIFT
    )
    # The contour specifies a 0.12 m rise; the search should close most of
    # the 0.38 m it started away from it.
    assert fitted < 0.30, f'search barely moved: {fitted}'
    assert result.loss < result.baseline_loss * 0.5


def test_the_search_never_returns_something_worse_than_it_was_given() -> None:
    """The incumbent is replaced only on a strict improvement.

    Checked against a compiler whose loss is unaffected by every knob, which is
    the case where a search that accepted ties would wander.
    """
    scene, program = _program()

    def flat(program_):
        block = scene.object_by_id("block")
        t = block.transform.translation
        return _clip([
            ClipFrame(time_s=i / 10.0, bones={},
                      objects={"block": Transform(translation=t)})
            for i in range(11)
        ])

    result = fit_to_contour(program, scene, flat, budget=20)
    assert result is not None
    assert result.loss <= result.baseline_loss
    assert not result.improved
    assert result.program == program, "no improvement means no change"


def test_the_search_respects_its_budget_and_says_when_it_ran_out() -> None:
    scene, program = _program()
    calls: list = []
    result = fit_to_contour(
        program, scene, _synthetic_compiler(scene, 0.12, calls), budget=7
    )
    assert result is not None
    assert result.evaluations <= 7
    assert len(calls) <= 7
    assert result.exhausted_budget is True, (
        "a search that stops on budget must not report it as convergence"
    )


def test_an_unresolvable_command_is_not_fitted_to_an_invented_target() -> None:
    scene, program = _program()
    bogus = program.model_copy(update={"source_text": "wave hello"})
    calls: list = []
    assert (
        fit_to_contour(bogus, scene, _synthetic_compiler(scene, 0.12, calls))
        is None
    )
    assert not calls, "nothing should be compiled without a specification"


def test_knobs_stay_inside_their_declared_bounds() -> None:
    scene, program = _program()
    calls: list = []
    fit_to_contour(program, scene, _synthetic_compiler(scene, 0.12, calls), budget=60)
    by_field = {k.field: k for k in GRAB_KNOBS}
    for candidate in calls:
        for primitive in candidate.primitives:
            for field, knob in by_field.items():
                value = float(getattr(primitive.parameters, field))
                assert knob.low <= value <= knob.high, (
                    f"{field}={value} escaped [{knob.low}, {knob.high}]"
                )


def test_a_knob_only_touches_the_phases_it_names() -> None:
    _scene, program = _program()
    knob = Knob("lift_height", "lift_height_m", (PrimitiveKind.LIFT,), 0.04, 0.0, 0.8)
    changed = _with_value(program, knob, 0.42)
    for original, updated in zip(program.primitives, changed.primitives):
        if updated.kind == PrimitiveKind.LIFT:
            assert updated.parameters.lift_height_m == pytest.approx(0.42)
        else:
            assert updated.parameters.lift_height_m == pytest.approx(
                original.parameters.lift_height_m
            )
    assert _current_value(changed, knob) == pytest.approx(0.42)


def test_the_search_is_deterministic() -> None:
    scene, program = _program()
    runs = []
    for _ in range(2):
        calls: list = []
        result = fit_to_contour(
            program, scene, _synthetic_compiler(scene, 0.12, calls), budget=40
        )
        assert result is not None
        runs.append((result.loss, result.evaluations, len(result.trace)))
    assert runs[0] == runs[1]
