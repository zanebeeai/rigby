"""Build explicitly, or verify/extract pinned offline third-party test inputs.

Only `build` accesses the network. Tests use the committed archive and verify
every entry. The three admitted source descriptions and their referenced meshes
are byte-identical to the pinned upstream files. Panda is a URDF-only coupling
refusal fixture: its historical package:// localization is explicitly recorded.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from xml.etree import ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parent
BULLET = "63c4d67e337017f9d8b298c900e9aabdb69296e7"
SO_ARM = "7629d2ad9853d10fb903093a33ef6114099d97e5"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()


def build(source_root):
    references = {r["robot_id"]: r for r in json.loads((source_root / "references.json").read_bytes())["references"]}
    payloads, records, checks = {}, [], []
    for name in ("iiwa7", "kuka_lwr", "so101", "panda"):
        reference = references[name]
        commit = SO_ARM if name == "so101" else BULLET
        url_root = f"https://raw.githubusercontent.com/{reference['upstream']}/{commit}"
        source = (source_root / name / "robot.urdf").read_bytes()
        upstream = fetch(f"{url_root}/{reference['upstream_path']}")
        assert digest(source) == reference["packaged_sha256"]
        if name != "panda":
            assert source == upstream, name
        else:
            assert digest(upstream) == reference["upstream_sha256"]
        payloads[f"{name}/robot.urdf"] = source
        meshes = sorted({n.get("filename") for n in ET.fromstring(source).iter("mesh")}) if name != "panda" else []
        for relative in meshes:
            content = (source_root / name / relative).read_bytes()
            url = f"{url_root}/{Path(reference['upstream_path']).parent.as_posix()}/{relative}"
            checks.append((f"{name}/{relative}", content, url))
            payloads[f"{name}/{relative}"] = content
        license_path = "LICENSE" if name == "so101" else "LICENSE.txt"
        payloads[f"{name}/LICENSE.upstream.txt"] = fetch(f"{url_root}/{license_path}")
        records.append({"body": name, "upstream": reference["upstream"], "commit": commit,
            "source_url": f"{url_root}/{reference['upstream_path']}", "license_url": f"{url_root}/{license_path}",
            "license_id": reference["license_id"], "upstream_urdf_sha256": digest(upstream),
            "packaged_urdf_sha256": digest(source), "mesh_count": len(meshes),
            "packaging_changes": "URDF basename only; URDF and all referenced meshes byte-identical" if name != "panda" else
                "Historical package:// prefixes localized; meshes omitted because coupling is refused before loading"})
    def check(item):
        name, content, url = item
        assert fetch(url) == content, name
        return {"path": name, "sha256": digest(content), "source_url": url}
    with ThreadPoolExecutor(max_workers=6) as pool:
        verified_meshes = list(pool.map(check, checks))
    payloads["ATTRIBUTION.txt"] = ("Third-party models: bulletphysics/bullet3 (Zlib) and TheRobotStudio/SO-ARM100 (Apache-2.0).\n"
        "Rigby does not author these models. Original license texts accompany each model.\n"
        "See provenance.json for exact revisions, file hashes, URLs and packaging changes.\n").encode()
    archive = ROOT / "source-packages.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in sorted(payloads.items()):
            entry = zipfile.ZipInfo(name, (2026, 9, 12, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, content, compresslevel=9)
    metadata = {"schema": "g03.third_party_inputs.v1", "archive_sha256": digest(archive.read_bytes()),
        "files": {k: digest(v) for k, v in sorted(payloads.items())}, "references": records,
        "upstream_mesh_verification": verified_meshes}
    (ROOT / "provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")
    return {"archive_bytes": archive.stat().st_size, "files": len(payloads), "upstream_meshes_verified": len(checks)}


def extract(destination):
    metadata = json.loads((ROOT / "provenance.json").read_bytes())
    archive = ROOT / "source-packages.zip"
    assert digest(archive.read_bytes()) == metadata["archive_sha256"]
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(metadata["files"])
        for name, expected in metadata["files"].items():
            output = (destination / name).resolve()
            assert output.is_relative_to(destination)
            content = bundle.read(name)
            assert digest(content) == expected
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "extract"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(build(args.path) if args.command == "build" else extract(args.path))
