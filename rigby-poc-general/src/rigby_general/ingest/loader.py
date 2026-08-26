"""Turn an uploaded URDF or MJCF into a simulable MuJoCo model.

Ingest runs in two passes, and the split matters. The first pass compiles the
model *as delivered* so morphology can be measured from it. Only then does the
second pass write the sites and actuators the measurement decided on. Doing it
the other way round would mean guessing where a gripper is before looking.

Three compiler settings are deliberately overridden:

``fusestatic = False``
    MuJoCo fuses a jointless root link into the world body. That is physically
    correct for a bolted-down arm, but it erases the base link's name -- and the
    ``BASE`` ground role and the ``BASE_DRIFT`` gate both need something to point
    at. Keeping it costs nothing: the body still has zero degrees of freedom.

``discardvisual = False``
    Visual geometry is what the evidence renderer shows the judge. Discarding it
    would leave the demo looking at collision proxies.

``balanceinertia = False``
    Left off on purpose. A body whose inertia violates the triangle inequality is
    a defect in the upload, and silently repairing it would hide the defect while
    changing the dynamics the primitive is later certified against. Integrity
    reports it instead.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco

from ..errors import GeneralFailureCode, ModelIngestError


SourceFormat = Literal["urdf", "mjcf"]

_URDF_SUFFIXES = frozenset({".urdf", ".URDF"})
_MJCF_SUFFIXES = frozenset({".xml", ".mjcf", ".XML"})


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A compiled model plus the spec it came from, ready for measurement."""

    robot_id: str
    source_format: SourceFormat
    source_path: Path
    source_bytes: bytes
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    sandbox: Path | None = None
    """Scratch directory holding the staged model and its meshes.

    Kept alive rather than cleaned up inside :func:`load_model`, because the spec
    resolves mesh paths against the directory it was read from and is recompiled
    later, when sites and actuators are added. Deleting it early works fine for a
    model built from primitives and fails on the first one with meshes -- the
    second compile cannot open a file the first one read happily. The caller
    releases it with :func:`release_sandbox`."""

    @property
    def mjcf_xml(self) -> str:
        return self.spec.to_xml()


def detect_source_format(path: Path) -> SourceFormat:
    suffix = path.suffix
    if suffix in _URDF_SUFFIXES:
        return "urdf"
    if suffix in _MJCF_SUFFIXES:
        return "mjcf"
    raise ModelIngestError(
        GeneralFailureCode.UNREADABLE_MODEL,
        f"Unrecognised model extension {suffix!r}; expected .urdf, .xml, or .mjcf",
        details={"path": path.name},
    )


def apply_compiler_policy(spec: mujoco.MjSpec) -> None:
    """Normalize compiler settings. See the module docstring for the reasoning."""

    spec.compiler.fusestatic = False
    spec.compiler.discardvisual = False
    spec.compiler.balanceinertia = False
    spec.compiler.autolimits = True


def load_model(
    source_path: Path,
    *,
    robot_id: str | None = None,
    sandbox: bool = True,
) -> LoadedModel:
    """Compile an uploaded model without letting it read outside its own folder.

    ``sandbox`` copies the model and its sibling assets into a scratch directory
    before compiling, so a relative ``../../`` mesh reference resolves to nothing
    instead of reaching into the filesystem. Absolute references are rejected
    outright by :mod:`.integrity`, which runs on the source text first.
    """

    resolved = source_path.resolve()
    if not resolved.is_file():
        raise ModelIngestError(
            GeneralFailureCode.UNREADABLE_MODEL,
            f"No model file at {resolved}",
            details={"path": str(resolved)},
        )

    source_format = detect_source_format(resolved)
    source_bytes = resolved.read_bytes()
    identity = robot_id or resolved.parent.name

    compile_path = resolved
    scratch: str | None = None
    try:
        if sandbox:
            scratch = tempfile.mkdtemp(prefix="rigby-ingest-")
            staged = Path(scratch) / resolved.parent.name
            shutil.copytree(resolved.parent, staged)
            compile_path = staged / resolved.name

        try:
            spec = mujoco.MjSpec.from_file(str(compile_path))
        except ValueError as error:
            raise ModelIngestError(
                GeneralFailureCode.UNREADABLE_MODEL,
                f"MuJoCo could not parse the model: {error}",
                details={"path": resolved.name, "format": source_format},
            ) from error

        apply_compiler_policy(spec)

        try:
            model = spec.compile()
        except ValueError as error:
            raise ModelIngestError(
                GeneralFailureCode.UNREADABLE_MODEL,
                f"MuJoCo could not compile the model: {error}",
                details={"path": resolved.name, "format": source_format},
            ) from error
    except BaseException:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise

    if model.nbody < 2:
        raise ModelIngestError(
            GeneralFailureCode.UNREADABLE_MODEL,
            "The model contains no bodies besides the world",
            details={"path": resolved.name},
        )

    return LoadedModel(
        robot_id=identity,
        source_format=source_format,
        source_path=resolved,
        source_bytes=source_bytes,
        spec=spec,
        model=model,
        sandbox=Path(scratch) if scratch else None,
    )


def release_sandbox(loaded: LoadedModel) -> None:
    """Drop the staged copy once nothing will recompile the spec again."""

    if loaded.sandbox is not None:
        shutil.rmtree(loaded.sandbox, ignore_errors=True)


def urdf_velocity_limits(source: bytes) -> dict[str, float]:
    """Recover ``<limit velocity=...>`` per joint.

    MuJoCo maps a URDF joint's ``lower``/``upper`` onto the joint range and its
    ``effort`` onto the actuator force range, but drops ``velocity`` entirely --
    there is nowhere on ``MjsJoint`` to keep it. It is needed here because
    ``neutral_speed`` is measured from it, and neutral speed is what every
    magnitude-neutral ``speed`` ordinal grounds against.
    """

    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(source)
    except ET.ParseError:
        return {}
    if root.tag.rsplit("}", 1)[-1] != "robot":
        return {}

    limits: dict[str, float] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        raw = limit.get("velocity")
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0.0:
            limits[name] = value
    return limits


def body_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or f"body_{index}"
        for index in range(model.nbody)
    )


def joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}"
        for index in range(model.njnt)
    )


def geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom_{index}"
        for index in range(model.ngeom)
    )
