"""Build the two robots that exist on the operator's bench, not just in a file.

`assets/general/irl/` is for hardware someone can actually go and touch. The
point of simulating them here is to see a motion before it is uploaded to a real
arm, so the provenance of each number matters more than usual -- a demo that
looks right on a model with invented link lengths is worse than no demo.

So each robot records where its numbers came from, and the two differ sharply:

``kuka_kr6``  Kinematics are authoritative. Joint origins, axes and limits come
              from ROS-Industrial's `kuka_kr6_support`, which is KUKA's own
              geometry for the KR 6 R900 sixx, and the collision meshes are the
              real ones. Two things are *not* from the vendor and are marked as
              such: the inertias, which that package omits entirely and which
              MuJoCo infers from mesh volume (giving 54.6 kg against a published
              52 kg, so the approximation is close), and the gripper, which is a
              generic parallel jaw standing in for whatever is bolted to the
              flange.

``uhand2``    No vendor URDF exists. Every dimension here is read off Hiwonder's
              published specification for the uHand 2.0 and built from
              primitives. It is a dimensional stand-in, not a scan, and anything
              measured from it should be read as "roughly, for a hand this size".

Neither is committed: same reference-only policy as the exotic set. Run this
script to produce them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "general" / "irl"

KR6_REPO = "ros-industrial/kuka_experimental"
KR6_REF = "melodic-devel"
KR6_BASE = (
    f"https://raw.githubusercontent.com/{KR6_REPO}/{KR6_REF}/kuka_kr6_support"
)


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read() if response.status == 200 else None
    except Exception:  # noqa: BLE001 - a missing asset is reported, not raised
        return None


# --------------------------------------------------------------------------
# KUKA KR 6 R900 sixx
# --------------------------------------------------------------------------


def expand_xacro(text: str) -> str:
    """Resolve the macro by hand rather than shelling out to ROS.

    The file only uses three xacro features: a `prefix` parameter, includes for
    colour materials, and `radians()` in the joint limits. None needs a
    dependency to resolve, and requiring a ROS installation to read a robot
    description would be a strange thing to build into an uploader.
    """

    text = re.sub(r"<xacro:include[^>]*/>", "", text)
    text = re.sub(r"<xacro:material[^>]*/>", "", text)
    text = re.sub(r"<xacro:macro[^>]*>", "", text)
    text = text.replace("</xacro:macro>", "")
    text = text.replace(' xmlns:xacro="http://wiki.ros.org/xacro"', "")
    text = text.replace("${prefix}", "")
    text = re.sub(
        r"\$\{(-?)radians\(([\d.]+)\)\}",
        lambda m: f"{(-1 if m.group(1) else 1) * math.radians(float(m.group(2))):.6f}",
        text,
    )
    leftover = re.findall(r"\$\{[^}]*\}|<xacro:[^>]*>", text)
    if leftover:
        raise RuntimeError(f"xacro left unresolved: {leftover[:3]}")
    return text


def add_kr6_gripper(text: str) -> str:
    """Bolt a generic parallel jaw to the flange.

    Explicitly generic. The arm carries whatever the operator fitted, and this
    stands in for it at a plausible size for a 6 kg payload: 90 mm of stroke,
    fingers that protrude past the palm so a top-down grasp is geometrically
    possible at all.
    """

    root = ET.fromstring(text)

    def link(name, size, mass, origin=(0, 0, 0), rgba="0.15 0.15 0.17 1"):
        node = ET.SubElement(root, "link", name=name)
        for kind in ("visual", "collision"):
            element = ET.SubElement(node, kind)
            ET.SubElement(element, "origin", xyz=" ".join(f"{v:.5f}" for v in origin))
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", size=" ".join(f"{v:.5f}" for v in size))
            if kind == "visual":
                material = ET.SubElement(element, "material", name=f"{name}_mat")
                ET.SubElement(material, "color", rgba=rgba)
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.5f}" for v in origin))
        ET.SubElement(inertial, "mass", value=f"{mass:.5f}")
        extent = [max(v, 1e-3) for v in size]
        ixx = mass * (extent[1] ** 2 + extent[2] ** 2) / 12.0
        iyy = mass * (extent[0] ** 2 + extent[2] ** 2) / 12.0
        izz = mass * (extent[0] ** 2 + extent[1] ** 2) / 12.0
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{ixx:.8f}", ixy="0", ixz="0",
            iyy=f"{iyy:.8f}", iyz="0", izz=f"{izz:.8f}",
        )
        return node

    def joint(name, kind, parent, child, origin, axis, lower, upper, effort, velocity):
        node = ET.SubElement(root, "joint", name=name, type=kind)
        ET.SubElement(node, "origin", xyz=" ".join(f"{v:.5f}" for v in origin))
        ET.SubElement(node, "parent", link=parent)
        ET.SubElement(node, "child", link=child)
        ET.SubElement(node, "axis", xyz=" ".join(f"{v:.1f}" for v in axis))
        ET.SubElement(
            node, "limit",
            lower=f"{lower}", upper=f"{upper}",
            effort=f"{effort}", velocity=f"{velocity}",
        )

    # tool0 is the flange in every ROS-Industrial support package.
    link("gripper_palm", (0.09, 0.11, 0.055), 0.60, origin=(0.0275, 0, 0))
    joint("gripper_mount", "fixed", "tool0", "gripper_palm",
          (0, 0, 0), (1, 0, 0), 0, 0, 0, 0)
    for side, direction in (("left", 1.0), ("right", -1.0)):
        # Fingers reach 70 mm beyond the palm face, so the jaw can close around
        # something standing on a table rather than the palm reaching it first.
        link(f"gripper_finger_{side}", (0.07, 0.016, 0.030), 0.08,
             origin=(0.035, 0, 0), rgba="0.85 0.45 0.10 1")
        joint(f"gripper_grip_{side}", "prismatic", "gripper_palm",
              f"gripper_finger_{side}",
              (0.055, direction * 0.053, 0), (0, -direction, 0),
              0.0, 0.045, 120.0, 0.35)
    return ET.tostring(root, encoding="unicode")


def build_kr6(out: Path) -> dict | None:
    macro = fetch(f"{KR6_BASE}/urdf/kr6r900sixx_macro.xacro")
    if macro is None:
        print("kuka_kr6        FAILED to fetch the xacro")
        return None

    text = expand_xacro(macro.decode("utf-8"))

    # MuJoCo reads OBJ and STL. The visual meshes are COLLADA, so the collision
    # mesh stands in for both -- recorded here because it changes what is shown.
    text = re.sub(
        r"package://kuka_kr6_support/meshes/([^\"]+)/visual/([^\"]+)\.dae",
        r"meshes/\1/\2.stl", text,
    )
    text = re.sub(
        r"package://kuka_kr6_support/meshes/([^\"]+)/collision/([^\"]+)\.stl",
        r"meshes/\1/\2.stl", text,
    )
    text = add_kr6_gripper(text)

    directory = out / "kuka_kr6"
    directory.mkdir(parents=True, exist_ok=True)
    references = sorted(set(re.findall(r'filename="(meshes/[^"]+)"', text)))
    fetched, missing = 0, []
    for reference in references:
        family, name = reference.replace("meshes/", "").split("/")
        target = directory / "meshes" / family / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size:
            fetched += 1
            continue
        data = None
        for sub in ("collision", "visual"):
            data = fetch(f"{KR6_BASE}/meshes/{family}/{sub}/{name}")
            if data:
                break
        if data is None:
            missing.append(reference)
            continue
        target.write_bytes(data)
        fetched += 1

    payload = text.encode("utf-8")
    (directory / "robot.urdf").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (directory / "robot.urdf.sha256").write_text(f"{digest}  robot.urdf\n", "ascii")
    print(
        f"kuka_kr6        ok  {len(payload):>7} bytes  {fetched}/{len(references)} meshes"
        + (f"  {len(missing)} MISSING" if missing else "")
    )
    return {
        "robot_id": "kuka_kr6",
        "hardware": "KUKA KR 6 R900 sixx (Agilus)",
        "upstream": KR6_REPO,
        "upstream_ref": KR6_REF,
        "license_id": "BSD-3-Clause",
        "license_source_url": f"https://github.com/{KR6_REPO}/blob/{KR6_REF}/LICENSE",
        "bundled": False,
        "sha256": digest,
        "kinematics": "vendor, via ROS-Industrial kuka_kr6_support",
        "meshes": "vendor collision STLs; visual DAE unsupported by MuJoCo",
        "inertia": "INFERRED by MuJoCo from mesh volume; the package carries none. "
                   "Yields 54.6 kg against a published 52 kg.",
        "gripper": "GENERIC stand-in, not vendor hardware: 90 mm stroke parallel jaw",
        "meshes_missing": missing,
    }


# --------------------------------------------------------------------------
# Hiwonder uHand 2.0
# --------------------------------------------------------------------------


# The four fingers stand along the top edge of the palm and curl forward; the
# thumb sits low on the side and curls back across them. That opposition is the
# whole reason a hand can hold anything, and it has to be in the geometry -- five
# digits all curling the same way close on nothing, which is what a first pass at
# this produced.
#
# mount: (x, y, z) on the palm.  axis: which way the servo swings the digit.
UHAND_DIGITS = (
    # name,     mount,                    axis,          proximal, distal
    ("index",   (-0.008,  0.026, 0.090),  (0, 1.0, 0),   0.048,    0.036),
    ("middle",  (-0.008,  0.009, 0.090),  (0, 1.0, 0),   0.052,    0.038),
    ("ring",    (-0.008, -0.009, 0.090),  (0, 1.0, 0),   0.048,    0.036),
    ("little",  (-0.008, -0.026, 0.090),  (0, 1.0, 0),   0.040,    0.030),
    ("thumb",   ( 0.012,  0.038, 0.062),  (0, -1.0, 0),  0.044,    0.034),
)


def build_uhand(out: Path) -> dict:
    """A dimensional stand-in for the uHand 2.0, built from published specs.

    Six servos: five finger flexions and one wrist rotation. Every length below
    is read off Hiwonder's specification sheet rather than measured off the
    hardware, so this is the right shape and roughly the right size and is not a
    scan of the object on the bench.
    """

    robot = ET.Element("robot", name="uhand2")

    def link(name, size, mass, origin=(0, 0, 0), rgba="0.20 0.21 0.24 1"):
        node = ET.SubElement(robot, "link", name=name)
        for kind in ("visual", "collision"):
            element = ET.SubElement(node, kind)
            ET.SubElement(element, "origin", xyz=" ".join(f"{v:.5f}" for v in origin))
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", size=" ".join(f"{v:.5f}" for v in size))
            if kind == "visual":
                material = ET.SubElement(element, "material", name=f"{name}_mat")
                ET.SubElement(material, "color", rgba=rgba)
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.5f}" for v in origin))
        ET.SubElement(inertial, "mass", value=f"{mass:.5f}")
        e = [max(v, 2e-3) for v in size]
        ET.SubElement(
            inertial, "inertia",
            ixx=f"{mass * (e[1]**2 + e[2]**2) / 12:.9f}", ixy="0", ixz="0",
            iyy=f"{mass * (e[0]**2 + e[2]**2) / 12:.9f}", iyz="0",
            izz=f"{mass * (e[0]**2 + e[1]**2) / 12:.9f}",
        )

    def joint(name, kind, parent, child, origin, axis, lower, upper, effort, velocity):
        node = ET.SubElement(robot, "joint", name=name, type=kind)
        ET.SubElement(node, "origin", xyz=" ".join(f"{v:.5f}" for v in origin))
        ET.SubElement(node, "parent", link=parent)
        ET.SubElement(node, "child", link=child)
        ET.SubElement(node, "axis", xyz=" ".join(f"{v:.2f}" for v in axis))
        ET.SubElement(
            node, "limit", lower=f"{lower:.4f}", upper=f"{upper:.4f}",
            effort=f"{effort}", velocity=f"{velocity}",
        )

    link("base", (0.070, 0.070, 0.020), 0.35, origin=(0, 0, 0.01))
    link("forearm", (0.052, 0.052, 0.075), 0.22, origin=(0, 0, 0.0375),
         rgba="0.20 0.21 0.24 1")
    joint("wrist_rotate", "revolute", "base", "forearm",
          (0, 0, 0.020), (0, 0, 1), -1.5708, 1.5708, 8.0, 3.0)

    # Palm: the only body the fingers oppose against.
    link("palm", (0.028, 0.085, 0.090), 0.16, origin=(0, 0, 0.045),
         rgba="0.24 0.25 0.28 1")
    joint("palm_fixed", "fixed", "forearm", "palm",
          (0, 0, 0.075), (0, 0, 1), 0, 0, 0, 0)

    for name, mount, axis, proximal, distal in UHAND_DIGITS:
        # One servo per digit drives the proximal segment; the distal segment
        # follows it as a fixed link, which is how a tendon-driven finger of this
        # class actually behaves at the resolution anything here can measure.
        link(f"{name}_proximal", (0.016, 0.015, proximal), 0.010,
             origin=(0, 0, proximal / 2), rgba="0.88 0.88 0.90 1")
        joint(f"{name}_flex", "revolute", "palm", f"{name}_proximal",
              mount, axis, 0.0, 1.5708, 3.5, 4.0)
        link(f"{name}_distal", (0.014, 0.013, distal), 0.007,
             origin=(0, 0, distal / 2), rgba="0.88 0.88 0.90 1")
        joint(f"{name}_tip_fixed", "fixed", f"{name}_proximal", f"{name}_distal",
              (0, 0, proximal), (1, 0, 0), 0, 0, 0, 0)

    directory = out / "uhand2"
    directory.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(robot, encoding="unicode").encode("utf-8")
    (directory / "robot.urdf").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (directory / "robot.urdf.sha256").write_text(f"{digest}  robot.urdf\n", "ascii")
    print(f"uhand2          ok  {len(payload):>7} bytes  6 actuated joints, no meshes")
    return {
        "robot_id": "uhand2",
        "hardware": "Hiwonder uHand 2.0",
        "upstream": None,
        "license_id": "n/a -- authored here",
        "bundled": False,
        "sha256": digest,
        "kinematics": "SPEC-DERIVED. No vendor URDF exists; every dimension is "
                      "read off Hiwonder's published specification and built "
                      "from primitives. A dimensional stand-in, not a scan.",
        "meshes": "none; primitive geometry only",
        "inertia": "boxes at plausible density",
        "gripper": "the hand itself: five flexing fingers opposing one palm",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", default=None)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    records = []
    if arguments.only is None or "kuka_kr6" in arguments.only:
        record = build_kr6(arguments.out)
        if record:
            records.append(record)
    if arguments.only is None or "uhand2" in arguments.only:
        records.append(build_uhand(arguments.out))

    index_path = arguments.out / "references.json"
    existing = {}
    if index_path.is_file():
        try:
            for row in json.loads(index_path.read_text("utf-8"))["robots"]:
                existing[row["robot_id"]] = row
        except Exception:  # noqa: BLE001
            existing = {}
    for row in records:
        existing[row["robot_id"]] = row
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "role": "hardware_the_operator_owns",
                "note": (
                    "Robots that exist on a bench somewhere. Each record states "
                    "which numbers came from the vendor and which were inferred "
                    "or authored, because a motion is uploaded to the real arm "
                    "and a plausible-looking model with invented dimensions is "
                    "worse than none."
                ),
                "robots": [existing[k] for k in sorted(existing)],
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(records)} robot(s) -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
