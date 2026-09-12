"""Give a URDF structural identities before name-sensitive numerical analysis.

The uploaded names remain aliases. Sorting/renaming never changes a numeric
geometry, inertia, limit, axis, transmission ratio or kinematic relationship.
This intake profile admits the URDF elements whose references it can preserve;
unrecognized extensions receive a typed refusal rather than a lossy rewrite.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from xml.etree import ElementTree as ET

from ..errors import GeneralFailureCode, ModelIngestError
from ..ingest.integrity import raise_for_violations, scan_source_text


@dataclass(frozen=True)
class CanonicalURDF:
    xml: str
    link_aliases: dict[str, str]
    joint_aliases: dict[str, str]
    source_sha256: str
    canonical_sha256: str


def _refuse(detail: str):
    raise ModelIngestError(GeneralFailureCode.INVALID_CONTRACT, detail,
                          details={"profile": "structural_urdf.v1"})


def _ordered(element):
    """URDF child declarations have no execution order; preserve all values."""
    result = ET.Element(element.tag, dict(sorted(element.attrib.items())))
    result.text = (element.text or "").strip() or None
    children = [_ordered(child) for child in element]
    result.extend(sorted(children, key=lambda node: ET.tostring(node, encoding="unicode")))
    return result


def _descriptor(element, *, joint=False):
    copy = deepcopy(element)
    copy.attrib.pop("name", None)
    for node in copy.iter():
        if node.tag in {"visual", "collision"}:
            node.attrib.pop("name", None)
        if joint and node.tag in {"parent", "child"}:
            node.attrib.pop("link", None)
        if joint and node.tag == "mimic":
            # The referenced identity is rewritten after the whole tree has
            # structural labels. Ratio and offset still distinguish mechanics.
            node.attrib.pop("joint", None)
    return ET.tostring(_ordered(copy), encoding="unicode")


def canonicalize_urdf(source: bytes) -> CanonicalURDF:
    raise_for_violations("uploaded_body", scan_source_text(source))
    try:
        root = ET.fromstring(source)
    except ET.ParseError as error:
        raise ModelIngestError(GeneralFailureCode.UNREADABLE_MODEL, str(error)) from error
    if root.tag != "robot":
        _refuse("Structural intake currently requires a URDF robot; other formats need their own reference-preserving profile")
    unknown = {child.tag for child in root} - {"link", "joint", "material", "transmission"}
    if unknown:
        _refuse(f"URDF extensions need explicit reference support: {sorted(unknown)}")
    links = {node.get("name"): node for node in root.findall("link")}
    joints = {node.get("name"): node for node in root.findall("joint")}
    if None in links or None in joints or len(links) != len(root.findall("link")) or len(joints) != len(root.findall("joint")):
        _refuse("Every link and joint must have a unique name")
    children = {name: [] for name in links}
    parent_of = {}
    for name, node in joints.items():
        parent, child = node.find("parent"), node.find("child")
        if parent is None or child is None or parent.get("link") not in links or child.get("link") not in links:
            _refuse("A joint names a missing parent or child link")
        parent, child = parent.get("link"), child.get("link")
        if child in parent_of:
            _refuse("URDF link has more than one parent")
        parent_of[child] = parent
        children[parent].append((name, child))
        mimic = node.find("mimic")
        if mimic is not None and mimic.get("joint") not in joints:
            _refuse("Mimic constraint names an absent joint")
    roots = set(links) - set(parent_of)
    if len(roots) != 1:
        _refuse("URDF requires exactly one connected root")
    signatures, visiting = {}, set()

    def signature(link):
        if link in visiting:
            _refuse("URDF kinematic tree contains a cycle")
        if link not in signatures:
            visiting.add(link)
            edges = sorted((_descriptor(joints[j], joint=True), signature(c)) for j, c in children[link])
            value = repr((_descriptor(links[link]), edges)).encode("utf-8")
            signatures[link] = hashlib.sha256(value).hexdigest()
            visiting.remove(link)
        return signatures[link]

    initial = next(iter(roots))
    signature(initial)
    if len(signatures) != len(links):
        _refuse("URDF contains disconnected links or a disconnected cycle")
    link_map, joint_map = {}, {}

    def visit(link):
        link_map[link] = f"body_{len(link_map):04d}"
        ordered = sorted(children[link], key=lambda edge: (_descriptor(joints[edge[0]], joint=True), signatures[edge[1]]))
        keys = [(_descriptor(joints[j], joint=True), signatures[c]) for j, c in ordered]
        if len(keys) != len(set(keys)):
            _refuse("Indistinguishable sibling subtrees require an explicit symmetry-orbit contract; source names cannot break the tie")
        for joint, child in ordered:
            joint_map[joint] = f"axis_{len(joint_map):04d}"
            visit(child)

    visit(initial)
    result = deepcopy(root)
    result.set("name", "structural_body")
    # Exact schema references, never substring replacement of identifiers.
    for node in result.findall("link"):
        node.set("name", link_map[node.get("name")])
        for child in node.iter():
            if child.tag in {"visual", "collision"}:
                child.attrib.pop("name", None)
    for node in result.findall("joint"):
        node.set("name", joint_map[node.get("name")])
        for tag in ("parent", "child"):
            node.find(tag).set("link", link_map[node.find(tag).get("link")])
        mimic = node.find("mimic")
        if mimic is not None:
            mimic.set("joint", joint_map[mimic.get("joint")])
    transmissions = []
    for node in result.findall("transmission"):
        for joint in node.findall("joint"):
            if joint.get("name") not in joint_map:
                _refuse("Transmission references an absent joint")
            joint.set("name", joint_map[joint.get("name")])
        node.attrib.pop("name", None)
        for actuator in node.findall("actuator"):
            actuator.attrib.pop("name", None)
        transmissions.append(node)
    for index, node in enumerate(sorted(transmissions, key=lambda n: ET.tostring(_ordered(n), encoding="unicode"))):
        node.set("name", f"transmission_{index:04d}")
        for offset, actuator in enumerate(node.findall("actuator")):
            actuator.set("name", f"drive_{index:04d}_{offset:04d}")
    xml = ET.tostring(_ordered(result), encoding="unicode")
    return CanonicalURDF(xml, {value: key for key, value in link_map.items()},
                         {value: key for key, value in joint_map.items()},
                         hashlib.sha256(source).hexdigest(), hashlib.sha256(xml.encode("utf-8")).hexdigest())
