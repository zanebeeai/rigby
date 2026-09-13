"""Stable machine-readable failure vocabulary for the general pipeline.

Extends ``rigby_core.errors.FailureCode`` with the failures that only exist once a
robot is no longer fixed: an unreadable model, an unmeasurable morphology, a
schema the body cannot afford, and a term that will not ground.

Every one of these is a *typed* refusal. The design rule inherited from v2 holds
here: a request either produces a certified motion or names precisely why it did
not. Nothing is silently approximated.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from rigby_core.errors import FailureCode as V2FailureCode


class GeneralFailureCode(StrEnum):
    """Failures specific to accepting an arbitrary body."""

    INVALID_CONTRACT = "invalid_contract"
    UNREADABLE_MODEL = "unreadable_model"
    """The uploaded URDF or MJCF would not compile."""

    UNSAFE_ASSET = "unsafe_asset"
    """External path reference, native plugin, or non-convex dynamic collider."""

    DEGENERATE_INERTIA = "degenerate_inertia"
    """A body carries non-finite or non-positive mass or inertia."""

    INVALID_JOINT_LIMIT = "invalid_joint_limit"
    """A declared joint range or velocity/effort bound is invalid."""

    UNSUPPORTED_MORPHOLOGY = "unsupported_morphology"
    """A floating base or a topology outside the supported classes."""

    UNSUPPORTED_COUPLING = "unsupported_coupling"
    """Source joint coupling cannot yet be preserved by measurement/control."""

    NO_EFFECTOR = "no_effector"
    """No chain terminates in anything that could act on the world."""

    FRAME_UNCONFIRMED = "frame_unconfirmed"
    """An intrinsic-frame schema was requested before a person confirmed front."""

    UNAFFORDED_SCHEMA = "unafforded_schema"
    """This body has no certified primitive for a requested schema."""

    UNGROUNDABLE = "ungroundable"
    """A magnitude-neutral term resolved outside this robot's measured limits."""

    INVENTED_BINDING = "invented_binding"
    """The planner named a site, object, or schema the robot does not have."""

    METRIC_LEAK = "metric_leak"
    """A schema program carried a metric value it structurally must not hold."""

    PROHIBITED_SUBSTITUTION = "prohibited_substitution"
    """The planner reported a quantity the request never stated, or changed one it did."""

    BAKE_BUDGET_EXHAUSTED = "bake_budget_exhausted"
    """The primitive bake hit its time or attempt ceiling before completing."""

    INTERNAL_ERROR = "internal_error"


FailureCode = GeneralFailureCode | V2FailureCode


class RigbyGeneralError(RuntimeError):
    def __init__(
        self,
        code: GeneralFailureCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ModelIngestError(RigbyGeneralError):
    """An uploaded asset could not be turned into a simulable model."""


class MorphologyError(RigbyGeneralError):
    """A model compiled, but its morphology could not be measured or supported."""


class GroundingError(RigbyGeneralError):
    """A body-neutral term could not be resolved against this robot."""

    def __init__(
        self,
        message: str,
        *,
        schema_key: str | None = None,
        measurement: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        if schema_key is not None:
            payload["schema_key"] = schema_key
        if measurement is not None:
            payload["measurement"] = measurement
        super().__init__(GeneralFailureCode.UNGROUNDABLE, message, details=payload)


class BakeBudgetExhausted(RigbyGeneralError):
    """The bake stopped early; the library is incomplete and says so."""

    def __init__(
        self,
        message: str,
        *,
        attempted: int,
        certified: int,
        elapsed_seconds: float,
    ) -> None:
        super().__init__(
            GeneralFailureCode.BAKE_BUDGET_EXHAUSTED,
            message,
            details={
                "attempted": attempted,
                "certified": certified,
                "elapsed_seconds": elapsed_seconds,
            },
        )


class RobotNotFoundError(KeyError):
    pass
