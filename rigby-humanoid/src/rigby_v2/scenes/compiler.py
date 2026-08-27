"""Safe composition of canonical humans and declarative object packs."""

from __future__ import annotations

import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import mujoco

from rigby_v2.errors import ArtifactIntegrityError
from rigby_v2.rigging.canonical_human import xml_for_profile

from .loader import PACK_ROOT, load_object_pack, load_scene_assets
from .models import (
    CollisionGeomSpec,
    ObjectPackSpec,
    ObjectSpec,
    PartSpec,
    VisualMeshSpec,
)


MAX_MJZ_BYTES = 64 * 1024 * 1024
MAX_MJZ_MEMBERS = 512
MAX_MJZ_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_MJZ_COMPRESSION_RATIO = 200.0
MAX_MJCF_BYTES = 8 * 1024 * 1024


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _name(*parts: str) -> str:
    return "obj__" + "__".join(parts)


@dataclass(frozen=True, slots=True)
class CompiledScene:
    pack_id: str
    profile: str
    xml: str
    xml_sha256: str
    mjz_bytes: bytes
    mjz_sha256: str
    sha256: str
    assets: dict[str, bytes]
    render_only_geoms: tuple[str, ...]
    semantic_sites: dict[str, str]
    affordances: dict[str, dict[str, object]]
    state_predicates: tuple[dict[str, object], ...]

    def load_model(self) -> mujoco.MjModel:
        """Validate the scene artifact and compile exclusively from its MJZ."""

        actual_hash = hashlib.sha256(self.mjz_bytes).hexdigest()
        if actual_hash != self.mjz_sha256 or self.sha256 != self.mjz_sha256:
            raise ArtifactIntegrityError(
                "compiled scene MJZ hash mismatch",
                details={"expected": self.mjz_sha256, "actual": actual_hash},
            )
        _validate_mjz_members(self.mjz_bytes)
        try:
            spec = mujoco.from_zip(io.BytesIO(self.mjz_bytes))
            return spec.compile()
        except (ValueError, RuntimeError) as error:
            raise ArtifactIntegrityError(
                "compiled scene MJZ could not be loaded", details={"error": str(error)}
            ) from error


def _validate_mjz_members(value: bytes) -> tuple[str, ...]:
    if len(value) > MAX_MJZ_BYTES:
        raise ArtifactIntegrityError(
            "compiled scene MJZ exceeds the archive size limit"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            entries = tuple(archive.infolist())
            members = tuple(item.filename for item in entries)
            if len(entries) > MAX_MJZ_MEMBERS:
                raise ArtifactIntegrityError("compiled scene MJZ has too many members")
            if len(set(members)) != len(members):
                raise ArtifactIntegrityError(
                    "compiled scene MJZ contains duplicate members"
                )
            total_uncompressed = sum(item.file_size for item in entries)
            if total_uncompressed > MAX_MJZ_UNCOMPRESSED_BYTES:
                raise ArtifactIntegrityError(
                    "compiled scene MJZ exceeds the uncompressed size limit"
                )
            for item in entries:
                if (
                    item.file_size
                    and item.file_size / max(item.compress_size, 1)
                    > MAX_MJZ_COMPRESSION_RATIO
                ):
                    raise ArtifactIntegrityError(
                        "compiled scene MJZ exceeds the compression-ratio limit"
                    )
            xml_entries = tuple(
                item for item in entries if item.filename.lower().endswith(".xml")
            )
            if len(xml_entries) == 1:
                if xml_entries[0].file_size > MAX_MJCF_BYTES:
                    raise ArtifactIntegrityError(
                        "compiled scene MJCF exceeds the size limit"
                    )
                xml_payload = archive.read(xml_entries[0])
            else:
                xml_payload = b""
    except (OSError, zipfile.BadZipFile) as error:
        raise ArtifactIntegrityError(
            "compiled scene artifact is not a valid MJZ"
        ) from error
    if not members or sum(name.lower().endswith(".xml") for name in members) != 1:
        raise ArtifactIntegrityError(
            "compiled scene MJZ must contain exactly one MJCF document"
        )
    for name in members:
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or not path.name
            or ":" in path.parts[0]
        ):
            raise ArtifactIntegrityError(
                "compiled scene MJZ contains an unsafe member path",
                details={"member": name},
            )
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as error:
        raise ArtifactIntegrityError("compiled scene MJCF is malformed") from error
    if root.find(".//include") is not None:
        raise ArtifactIntegrityError(
            "compiled scene MJCF cannot contain external includes"
        )
    if root.find(".//extension") is not None or root.find(".//plugin") is not None:
        raise ArtifactIntegrityError("compiled scene MJCF cannot load plugins")
    compiler = root.find("compiler")
    if compiler is not None and any(
        key in compiler.attrib for key in ("assetdir", "meshdir", "texturedir")
    ):
        raise ArtifactIntegrityError(
            "compiled scene MJCF cannot declare external asset directories"
        )
    member_set = {name.replace("\\", "/") for name in members}
    for element in root.iter():
        file_reference = element.attrib.get("file")
        if file_reference is None:
            continue
        normalized = file_reference.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or not path.name
            or ":" in path.parts[0]
            or normalized not in member_set
        ):
            raise ArtifactIntegrityError(
                "compiled scene MJCF references a non-embedded asset",
                details={"file": file_reference},
            )
    return members


def _compile_spec_and_mjz(xml: str, assets: dict[str, bytes]) -> bytes:
    """Parse, compile, and archive through MuJoCo's native MjSpec API."""

    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    spec.compile()
    stream = io.BytesIO()
    mujoco.to_zip(spec, stream)
    value = stream.getvalue()
    _validate_mjz_members(value)
    # Prove the serialized artifact, rather than only the in-memory source,
    # remains independently loadable before it is returned to the caller.
    mujoco.from_zip(io.BytesIO(value)).compile()
    return value


def _append_geom(
    body: ET.Element,
    obj: ObjectSpec,
    part: PartSpec,
    geom: CollisionGeomSpec,
    *,
    mass: float,
    asset: ET.Element,
) -> None:
    attributes = {
        "name": _name(obj.object_id, part.part_id, geom.name),
        "pos": _numbers(geom.pos),
        "quat": _numbers(geom.quat_wxyz),
        "mass": f"{mass:.9g}",
        "friction": _numbers(obj.material.friction),
        "rgba": _numbers(obj.material.rgba),
        "contype": "1",
        "conaffinity": "1",
    }
    if geom.margin_m > 0.0:
        attributes["margin"] = f"{geom.margin_m:.9g}"
    if geom.contact_time_constant_s is not None:
        attributes["solref"] = _numbers(
            (geom.contact_time_constant_s, geom.contact_damping_ratio or 1.0)
        )
    if geom.type == "convex_mesh":
        mesh_name = _name(obj.object_id, part.part_id, geom.name, "mesh")
        ET.SubElement(asset, "mesh", name=mesh_name, file=geom.asset_path or "")
        attributes.update({"type": "mesh", "mesh": mesh_name})
    else:
        attributes.update({"type": geom.type, "size": _numbers(geom.size)})
    ET.SubElement(body, "geom", attributes)


def _append_visual_mesh(
    body: ET.Element,
    obj: ObjectSpec,
    part: PartSpec,
    mesh: VisualMeshSpec,
    *,
    asset: ET.Element,
) -> str:
    """Attach a source mesh to rendering without mass or collision semantics."""

    mesh_name = _name(obj.object_id, part.part_id, "visual", mesh.name, "mesh")
    ET.SubElement(
        asset,
        "mesh",
        name=mesh_name,
        file=mesh.asset_path,
        scale=_numbers(mesh.scale),
    )
    geom_name = _name(obj.object_id, part.part_id, "visual", mesh.name)
    ET.SubElement(
        body,
        "geom",
        name=geom_name,
        type="mesh",
        mesh=mesh_name,
        pos=_numbers(mesh.pos),
        quat=_numbers(mesh.quat_wxyz),
        rgba=_numbers(obj.material.rgba),
        mass="0",
        contype="0",
        conaffinity="0",
        group="2",
    )
    return geom_name


def _append_object(
    worldbody: ET.Element,
    asset: ET.Element,
    obj: ObjectSpec,
    render_only_geoms: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    children: dict[str | None, list[PartSpec]] = {}
    for part in obj.parts:
        children.setdefault(part.parent, []).append(part)
    body_elements: dict[str, ET.Element] = {}

    def append_part(part: PartSpec, parent: ET.Element) -> None:
        position = obj.pos if part.parent is None else part.pos
        quaternion = obj.quat_wxyz if part.parent is None else part.quat_wxyz
        body = ET.SubElement(
            parent,
            "body",
            name=_name(obj.object_id, part.part_id),
            pos=_numbers(position),
            quat=_numbers(quaternion),
        )
        body_elements[part.part_id] = body
        if part.parent is None and obj.dynamic:
            ET.SubElement(body, "freejoint", name=_name(obj.object_id, "free"))
        if part.joint is not None:
            ET.SubElement(
                body,
                "joint",
                name=_name(obj.object_id, part.joint.name),
                type=part.joint.type,
                axis=_numbers(part.joint.axis),
                range=_numbers(part.joint.range),
                damping=f"{part.joint.damping:.9g}",
                frictionloss=f"{part.joint.frictionloss:.9g}",
            )
        geom_mass = part.mass_kg / len(part.geoms)
        for geom in part.geoms:
            _append_geom(body, obj, part, geom, mass=geom_mass, asset=asset)
        for visual_mesh in part.visual_meshes:
            render_only_geoms.append(
                _append_visual_mesh(
                    body, obj, part, visual_mesh, asset=asset
                )
            )
        for child in sorted(
            children.get(part.part_id, ()), key=lambda value: value.part_id
        ):
            append_part(child, body)

    root = children[None][0]
    append_part(root, worldbody)

    sites: dict[str, str] = {}
    for site in obj.sites:
        compiled_name = _name(obj.object_id, site.name)
        ET.SubElement(
            body_elements[site.part],
            "site",
            name=compiled_name,
            type="sphere",
            pos=_numbers(site.pos),
            size=f"{site.size_m:.9g}",
            rgba="1 0.2 0.1 0.8",
            group="3",
        )
        sites[f"{obj.object_id}.{site.name}"] = compiled_name
    affordances = {
        f"{obj.object_id}.{value.name}": {
            "kind": value.kind,
            "site": sites[f"{obj.object_id}.{value.site}"],
            "allowed_effectors": list(value.allowed_effectors),
        }
        for value in obj.affordances
    }
    return sites, affordances


def compile_scene(
    pack: str | ObjectPackSpec,
    *,
    profile: str = "medium",
    pack_root: Path = PACK_ROOT,
) -> CompiledScene:
    selected = load_object_pack(pack, root=pack_root) if isinstance(pack, str) else pack
    assets = load_scene_assets(selected, root=pack_root)
    root = ET.fromstring(xml_for_profile(profile))
    root.set("model", f"rigby_scene_{selected.pack_id}_{profile}")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("canonical human MJCF has no worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = list(root).index(worldbody)
        root.insert(worldbody_index, asset)

    semantic_sites: dict[str, str] = {}
    for support in selected.supports:
        support_name = f"support__{support.support_id}"
        ET.SubElement(
            worldbody,
            "geom",
            name=support_name,
            type="box",
            pos=_numbers(support.pos),
            size=_numbers(support.size),
            friction=_numbers(support.material.friction),
            rgba=_numbers(support.material.rgba),
            contype="1",
            conaffinity="1",
        )
        if support.site_name is not None:
            compiled_site = f"support__{support.support_id}__{support.site_name}"
            ET.SubElement(
                worldbody,
                "site",
                name=compiled_site,
                type="box",
                pos=_numbers(support.pos),
                size=_numbers(support.size),
                rgba="0.1 0.8 0.2 0.18",
                group="3",
            )
            semantic_sites[f"{support.support_id}.{support.site_name}"] = compiled_site

    affordances: dict[str, dict[str, object]] = {}
    predicates: list[dict[str, object]] = []
    render_only_geoms: list[str] = []
    for obj in sorted(selected.objects, key=lambda value: value.object_id):
        object_sites, object_affordances = _append_object(
            worldbody, asset, obj, render_only_geoms
        )
        semantic_sites.update(object_sites)
        affordances.update(object_affordances)
        predicates.extend(
            {
                "object_id": obj.object_id,
                "name": predicate.name,
                "target": (
                    _name(obj.object_id, predicate.target)
                    if any(
                        part.joint and part.joint.name == predicate.target
                        for part in obj.parts
                    )
                    else object_sites[f"{obj.object_id}.{predicate.target}"]
                ),
                "operator": predicate.operator,
                "values": list(predicate.values),
                "units": predicate.units,
            }
            for predicate in obj.state_predicates
        )

    metadata = {
        "schema_version": selected.schema_version,
        "pack_id": selected.pack_id,
        "semantic_sites": semantic_sites,
        "affordances": affordances,
        "state_predicates": predicates,
        "render_only_geoms": render_only_geoms,
    }
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(
        custom,
        "text",
        name="rigby_scene_metadata",
        data=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    )
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    mjz_bytes = _compile_spec_and_mjz(xml, assets)
    mjz_sha256 = hashlib.sha256(mjz_bytes).hexdigest()
    return CompiledScene(
        pack_id=selected.pack_id,
        profile=profile,
        xml=xml,
        xml_sha256=hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        mjz_bytes=mjz_bytes,
        mjz_sha256=mjz_sha256,
        sha256=mjz_sha256,
        assets=assets,
        render_only_geoms=tuple(render_only_geoms),
        semantic_sites=semantic_sites,
        affordances=affordances,
        state_predicates=tuple(predicates),
    )


def load_compiled_scene(
    pack: str | ObjectPackSpec,
    *,
    profile: str = "medium",
    pack_root: Path = PACK_ROOT,
) -> tuple[CompiledScene, mujoco.MjModel]:
    compiled = compile_scene(pack, profile=profile, pack_root=pack_root)
    return compiled, compiled.load_model()
