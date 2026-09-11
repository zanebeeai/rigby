"""Declarative, security-checked MuJoCo scene and object-pack compiler."""

from .compiler import CompiledScene, compile_scene, load_compiled_scene
from .loader import (
    PACK_ROOT,
    SceneAssetError,
    load_object_pack,
    load_scene_assets,
    resolve_content_asset,
)
from .models import ObjectPackSpec, VisualMeshSpec
from .staging import StagedScene, stage_object_pack_scene

__all__ = [
    "CompiledScene",
    "ObjectPackSpec",
    "VisualMeshSpec",
    "PACK_ROOT",
    "SceneAssetError",
    "StagedScene",
    "compile_scene",
    "load_compiled_scene",
    "load_object_pack",
    "load_scene_assets",
    "resolve_content_asset",
    "stage_object_pack_scene",
]
