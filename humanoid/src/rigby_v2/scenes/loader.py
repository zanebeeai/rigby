"""Trusted pack discovery and content-addressed scene-asset loading."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from rigby_core.errors import FailureCode, RigbyV2Error
from rigby_core.hashing import hash_file

from .models import ObjectPackSpec


PACK_ROOT = Path(__file__).resolve().parents[3] / "assets" / "v2" / "object_packs"
_PACK_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_PACK_BYTES = 512 * 1024
MAX_COLLISION_ASSET_BYTES = 10 * 1024 * 1024
MAX_MESH_VERTICES = 100_000
MAX_MESH_FACES = 200_000
MAX_TOTAL_MESH_ASSET_BYTES = 64 * 1024 * 1024
MAX_TOTAL_MESH_VERTICES = 250_000
MAX_TOTAL_MESH_FACES = 500_000


class SceneAssetError(RigbyV2Error):
    def __init__(
        self, message: str, *, details: dict[str, object] | None = None
    ) -> None:
        super().__init__(FailureCode.UNSUPPORTED_ASSET, message, details=details)


def resolve_content_asset(root: Path, relative_path: str, expected_sha256: str) -> Path:
    """Resolve and verify a mesh asset without permitting path escape."""

    if "\\" in relative_path or not _SHA256.fullmatch(expected_sha256):
        raise SceneAssetError("mesh asset reference is not canonical")
    pure = PurePosixPath(relative_path)
    expected_parts = ("sha256", expected_sha256, pure.name)
    if (
        pure.is_absolute()
        or pure.parts != expected_parts
        or pure.name in {"", ".", ".."}
    ):
        raise SceneAssetError(
            "mesh assets must use sha256/<digest>/<basename> paths",
            details={"path": relative_path},
        )
    if pure.suffix.lower() not in {".obj", ".stl"}:
        raise SceneAssetError("only content-addressed OBJ and STL mesh assets are supported")
    safe_root = root.resolve()
    candidate = (safe_root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(safe_root)
    except ValueError as error:
        raise SceneAssetError("mesh asset escapes the object-pack root") from error
    if not candidate.is_file() or candidate.stat().st_size > MAX_COLLISION_ASSET_BYTES:
        raise SceneAssetError("mesh asset is missing or exceeds the size limit")
    if hash_file(candidate) != expected_sha256:
        raise SceneAssetError("mesh asset content does not match its SHA-256 address")
    return candidate


def _validate_obj_payload(
    value: bytes, *, require_collision_piece: bool
) -> tuple[int, int]:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SceneAssetError("OBJ mesh asset must be UTF-8 text") from error
    vertices = 0
    faces = 0
    for raw_line in text.splitlines():
        line = raw_line.lstrip()
        if not line or line.startswith("#"):
            continue
        keyword = line.split(maxsplit=1)[0].lower()
        if keyword in {"mtllib", "call", "csh"}:
            raise SceneAssetError("OBJ mesh asset contains an external reference")
        vertices += keyword == "v"
        faces += keyword == "f"
    if require_collision_piece and (vertices < 4 or faces < 4):
        raise SceneAssetError("OBJ collision asset is not a closed convex piece")
    if not require_collision_piece and (vertices < 3 or faces < 1):
        raise SceneAssetError("OBJ visual mesh has insufficient geometry")
    if vertices > MAX_MESH_VERTICES or faces > MAX_MESH_FACES:
        raise SceneAssetError("OBJ mesh asset exceeds mesh complexity limits")
    return vertices, faces


def _validate_stl_payload(value: bytes) -> tuple[int, int]:
    # A binary STL is exactly 84 bytes plus 50 bytes per declared triangle.
    if len(value) >= 84:
        triangles = int.from_bytes(value[80:84], "little")
        if len(value) == 84 + triangles * 50:
            if triangles == 0 or triangles > MAX_MESH_FACES:
                raise SceneAssetError("STL mesh asset exceeds mesh complexity limits")
            return 0, triangles
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as error:
        raise SceneAssetError("STL mesh asset is malformed") from error
    facets = sum(
        line.lstrip().lower().startswith("facet normal") for line in text.splitlines()
    )
    if facets == 0 or facets > MAX_MESH_FACES:
        raise SceneAssetError("STL mesh asset exceeds mesh complexity limits")
    return 0, facets


def validate_collision_asset_payload(path: Path, value: bytes) -> tuple[int, int]:
    if path.suffix.lower() == ".obj":
        return _validate_obj_payload(value, require_collision_piece=True)
    elif path.suffix.lower() == ".stl":
        return _validate_stl_payload(value)
    else:  # resolve_content_asset already rejects this; keep this function fail-closed.
        raise SceneAssetError("unsupported collision asset format")


def validate_visual_asset_payload(path: Path, value: bytes) -> tuple[int, int]:
    """Validate a raw render mesh without asserting collision convexity."""

    if path.suffix.lower() == ".obj":
        return _validate_obj_payload(value, require_collision_piece=False)
    if path.suffix.lower() == ".stl":
        return _validate_stl_payload(value)
    raise SceneAssetError("unsupported visual mesh asset format")


def load_object_pack(pack_id: str, *, root: Path = PACK_ROOT) -> ObjectPackSpec:
    if not _PACK_ID.fullmatch(pack_id):
        raise SceneAssetError("object pack ID is invalid")
    safe_root = root.resolve()
    path = (safe_root / f"{pack_id}.json").resolve()
    try:
        path.relative_to(safe_root)
    except ValueError as error:
        raise SceneAssetError("object pack path escapes the pack root") from error
    if not path.is_file() or path.stat().st_size > MAX_PACK_BYTES:
        raise SceneAssetError("object pack is missing or exceeds the size limit")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        pack = ObjectPackSpec.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise SceneAssetError(
            "object pack manifest is invalid", details={"error": str(error)}
        ) from error
    if pack.pack_id != pack_id:
        raise SceneAssetError("object pack ID does not match its filename")
    return pack


def _asset_references(
    pack: ObjectPackSpec,
) -> tuple[dict[str, str], dict[str, str]]:
    collision: dict[str, str] = {}
    visual: dict[str, str] = {}
    for obj in pack.objects:
        for part in obj.parts:
            for geom in part.geoms:
                if geom.asset_path is None or geom.asset_sha256 is None:
                    continue
                previous = collision.setdefault(geom.asset_path, geom.asset_sha256)
                if previous != geom.asset_sha256:
                    raise SceneAssetError("mesh asset path has conflicting hashes")
            for mesh in part.visual_meshes:
                previous = visual.setdefault(mesh.asset_path, mesh.asset_sha256)
                if previous != mesh.asset_sha256:
                    raise SceneAssetError("visual mesh asset path has conflicting hashes")
    overlap = sorted(set(collision) & set(visual))
    if overlap:
        raise SceneAssetError(
            "render-only visual mesh cannot also be classified as collision geometry",
            details={"paths": overlap},
        )
    return collision, visual


def load_scene_assets(
    pack: ObjectPackSpec, *, root: Path = PACK_ROOT
) -> dict[str, bytes]:
    """Load all declared scene meshes with shared and aggregate security budgets."""

    collision, visual = _asset_references(pack)
    assets: dict[str, bytes] = {}
    total_bytes = 0
    total_vertices = 0
    total_faces = 0
    for role, references in (("collision", collision), ("visual", visual)):
        for asset_path, asset_sha256 in sorted(references.items()):
            path = resolve_content_asset(root, asset_path, asset_sha256)
            payload = path.read_bytes()
            if role == "collision":
                vertices, faces = validate_collision_asset_payload(path, payload)
            else:
                vertices, faces = validate_visual_asset_payload(path, payload)
            total_bytes += len(payload)
            total_vertices += vertices
            total_faces += faces
            if (
                total_bytes > MAX_TOTAL_MESH_ASSET_BYTES
                or total_vertices > MAX_TOTAL_MESH_VERTICES
                or total_faces > MAX_TOTAL_MESH_FACES
            ):
                raise SceneAssetError(
                    "object pack mesh assets exceed aggregate size or complexity limits"
                )
            assets[asset_path] = payload
    return assets


def load_collision_assets(
    pack: ObjectPackSpec, *, root: Path = PACK_ROOT
) -> dict[str, bytes]:
    """Compatibility helper returning collision assets after cross-role validation."""

    collision, _ = _asset_references(pack)
    loaded = load_scene_assets(pack, root=root)
    return {path: loaded[path] for path in collision}
