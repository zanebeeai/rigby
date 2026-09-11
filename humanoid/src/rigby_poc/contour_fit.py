"""Fit motion primitives so the object follows the path its command specified.

:mod:`rigby_poc.talmy` turns a command into a contour the Figure must follow and
scores a clip against it. That score is a loss, and this module descends it.

The search is a **pattern search** (Hooke-Jeeves style): try each knob at plus
and minus one step, keep any improvement, shrink the step when a full sweep
finds none. It is chosen over a gradient method because the objective is not
differentiable -- every evaluation runs a rigid-body simulation with contact,
where an infinitesimal parameter change can flip whether a contact happens at
all -- and over a random search because the budget is tens of evaluations, not
thousands. One evaluation is a full compile plus a full MuJoCo pass.

Two properties matter more than the search being clever:

* **It never returns something worse than it started with.** The incumbent is
  only replaced on a strict improvement, so the caller can always use the
  result.
* **It reports what it spent and what it dropped.** A search that quietly
  exhausts its budget and reports its best-so-far as though it converged is the
  shape this repository keeps cataloguing, so the trace says which it was.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import ClipResult, MotionProgram, PrimitiveKind, SceneManifest
from .talmy import MotionSituation, PathContour, contour_error, interpret, path_contour

CompileFn = Callable[[MotionProgram], ClipResult]


@dataclass(frozen=True)
class Knob:
    """One searchable parameter, and the primitives it applies to.

    ``kinds`` is empty for a knob that applies to every primitive; otherwise the
    change lands only on the phases named, which is what keeps "how high to
    lift" from also changing the reach.
    """

    name: str
    field: str
    kinds: tuple[PrimitiveKind, ...]
    step: float
    low: float
    high: float


#: The knobs that move the object. Deliberately small: every entry costs
#: evaluations, and parameters that only change how the motion *looks* cannot
#: reduce a contour error and would spend budget proving it.
GRAB_KNOBS: tuple[Knob, ...] = (
    Knob("lift_height", "lift_height_m",
         (PrimitiveKind.LIFT, PrimitiveKind.HOLD, PrimitiveKind.RECOVER),
         0.04, 0.0, 0.8),
    Knob("approach_depth", "arm_depth",
         (PrimitiveKind.CONTACT, PrimitiveKind.CLOSE), 0.15, -1.0, 1.0),
    Knob("approach_lateral", "lateral_offset",
         (PrimitiveKind.CONTACT, PrimitiveKind.CLOSE), 0.15, -1.0, 1.0),
    Knob("approach_height", "arm_height",
         (PrimitiveKind.CONTACT, PrimitiveKind.CLOSE), 0.15, -1.0, 1.0),
    Knob("grip_closure", "finger_curl",
         (PrimitiveKind.CLOSE, PrimitiveKind.LIFT, PrimitiveKind.HOLD), 0.15, -1.0, 1.0),
    Knob("thumb_opposition", "thumb_opposition",
         (PrimitiveKind.CLOSE, PrimitiveKind.LIFT, PrimitiveKind.HOLD), 0.15, 0.0, 1.0),
)


@dataclass
class FitResult:
    program: MotionProgram
    clip: ClipResult
    loss: float
    baseline_loss: float
    evaluations: int
    improved: bool
    exhausted_budget: bool
    contour: PathContour = field(repr=False)
    trace: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "talmy_contour_fit_v1",
            "path": self.contour.path,
            "notation": self.contour.situation.as_notation(),
            "baseline_loss_m": self.baseline_loss,
            "loss_m": self.loss,
            "improvement_m": self.baseline_loss - self.loss,
            "evaluations": self.evaluations,
            "improved": self.improved,
            "exhausted_budget": self.exhausted_budget,
            "accepted_steps": [t for t in self.trace if t["accepted"]],
        }


def _with_value(
    program: MotionProgram, knob: Knob, value: float
) -> MotionProgram:
    primitives = []
    for primitive in program.primitives:
        if knob.kinds and primitive.kind not in knob.kinds:
            primitives.append(primitive)
            continue
        primitives.append(
            primitive.model_copy(
                update={
                    "parameters": primitive.parameters.model_copy(
                        update={knob.field: value}
                    )
                }
            )
        )
    return program.model_copy(update={"primitives": primitives})


def _current_value(program: MotionProgram, knob: Knob) -> float:
    for primitive in program.primitives:
        if not knob.kinds or primitive.kind in knob.kinds:
            return float(getattr(primitive.parameters, knob.field))
    return 0.0


def objective(clip: ClipResult, contour: PathContour) -> float:
    """The loss: how far the object's trajectory departed from its contour.

    Endpoint error is weighted alongside the mean because the two fail
    differently. A trajectory that tracks the contour and then drops the object
    at the last moment has a small mean and has not performed the command.
    """
    error = contour_error(contour, clip.frames)
    if not error:
        return float("inf")
    return float(error["contour_mean_error_m"]) + float(
        error["contour_endpoint_error_m"]
    )


def fit_to_contour(
    program: MotionProgram,
    scene: SceneManifest,
    compile_fn: CompileFn,
    *,
    situation: MotionSituation | None = None,
    knobs: tuple[Knob, ...] = GRAB_KNOBS,
    budget: int = 40,
    shrink: float = 0.5,
    minimum_step: float = 0.02,
) -> FitResult | None:
    """Search primitive parameters for the lowest contour error.

    Returns ``None`` when the command does not resolve to a motion situation,
    because there is then no specification to fit to and any "improvement"
    would be against an invented target.
    """
    if situation is None:
        situation = interpret(program.source_text, scene)
    if situation is None:
        return None
    contour = path_contour(situation, scene)

    incumbent = program
    clip = compile_fn(incumbent)
    loss = objective(clip, contour)
    baseline = loss
    best_clip = clip
    evaluations = 1
    trace: list[dict[str, Any]] = []
    scale = 1.0

    while evaluations < budget and scale * min(k.step for k in knobs) >= minimum_step:
        swept_an_improvement = False
        for knob in knobs:
            if evaluations >= budget:
                break
            for direction in (1.0, -1.0):
                if evaluations >= budget:
                    break
                # Keep walking a direction while it keeps paying. Taking one
                # step per knob per sweep makes the search cost O(distance /
                # step) full simulations to cross a wide parameter, which at
                # seconds per evaluation is the difference between converging
                # and running out of budget part-way.
                moved_this_direction = False
                while evaluations < budget:
                    candidate_value = _current_value(incumbent, knob) + (
                        direction * knob.step * scale
                    )
                    if not (knob.low <= candidate_value <= knob.high):
                        break
                    candidate = _with_value(incumbent, knob, candidate_value)
                    candidate_clip = compile_fn(candidate)
                    candidate_loss = objective(candidate_clip, contour)
                    evaluations += 1
                    accepted = candidate_loss < loss - 1e-9
                    trace.append(
                        {
                            "knob": knob.name,
                            "value": candidate_value,
                            "loss_m": candidate_loss,
                            "accepted": accepted,
                            "evaluation": evaluations,
                        }
                    )
                    if not accepted:
                        break
                    incumbent, loss, best_clip = candidate, candidate_loss, candidate_clip
                    swept_an_improvement = True
                    moved_this_direction = True
                if moved_this_direction:
                    break
        if not swept_an_improvement:
            scale *= shrink
    return FitResult(
        program=incumbent,
        clip=best_clip,
        loss=loss,
        baseline_loss=baseline,
        evaluations=evaluations,
        improved=loss < baseline - 1e-9,
        exhausted_budget=evaluations >= budget,
        contour=contour,
        trace=trace,
    )
