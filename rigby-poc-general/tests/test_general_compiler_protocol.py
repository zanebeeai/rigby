"""Guard the structural contract with the certified v2 compiler.

``rigby_v2.motion.compiler.compile_motion_program`` annotates its ``rig``
parameter as ``RigAssetManifestV1``, whose validator refuses ``free_root=False``.
A fixed-base arm can never satisfy that, so this package passes its own
``RobotAssetManifestV1`` instead and relies on the compiler reading only four
members.

That reliance is invisible at runtime until it breaks, so it is asserted here
instead: if a future base tree reads a fifth member, this test names it rather
than letting a bake fail with ``AttributeError`` after twenty minutes.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from rigby_general.contracts import CompilerRigProtocol


EXPECTED_RIG_MEMBERS = {"rig_id", "dofs", "rest_qpos", "content_hash"}


def _rig_members_read_by(module_path: Path) -> set[str]:
    """Every attribute read from a local named ``rig`` in the module."""

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    members: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "rig":
                members.add(node.attr)
    return members


def _compiler_source_path() -> Path:
    from rigby_v2.motion import compiler

    return Path(inspect.getfile(compiler))


def test_compiler_reads_only_the_declared_rig_members() -> None:
    observed = _rig_members_read_by(_compiler_source_path())

    unexpected = observed - EXPECTED_RIG_MEMBERS
    assert not unexpected, (
        "rigby_v2 motion compiler now reads rig members this package does not "
        f"supply: {sorted(unexpected)}. Add them to RobotAssetManifestV1 and to "
        "CompilerRigProtocol."
    )
    assert observed, "expected the compiler to read the rig at all"


def test_protocol_members_match_the_expected_set() -> None:
    declared = {
        name
        for name in dir(CompilerRigProtocol)
        if not name.startswith("_")
    }
    assert declared == EXPECTED_RIG_MEMBERS


def test_robot_manifest_satisfies_the_protocol(manifest) -> None:
    assert isinstance(manifest, CompilerRigProtocol)
    assert isinstance(manifest.rig_id, str)
    assert manifest.dofs
    assert manifest.rest_qpos
    assert len(manifest.content_hash()) == 64
