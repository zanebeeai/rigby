"""Produce G03 admission, geometry, invariance and replayable inspection evidence.

Uses public zoo sources, independently frozen analytic fixtures, and the pinned
offline third-party archive. It does not open sealed benchmark labels/prompts.
The output is an engineering capability audit, not physical task performance.
"""

import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import mujoco
import numpy as np
from rigby_core.hashing import hash_file
from rigby_general.audit import _scan_for_robot_identifiers
from rigby_general.capabilities.intake import ingest_capability_body
from rigby_general.capabilities.inspection import capture_inspection, render_inspection, write_json
from rigby_general.errors import ModelIngestError


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIXTURES = ROOT / "tests/fixtures/g03_geometry"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def geometry_check(source):
    labels = json.loads((source.parent / "expected.json").read_bytes())
    body = ingest_capability_body(source)
    model = body.robot.finalized.model
    data = mujoco.MjData(model)
    for changed, uploaded in body.manifest.joint_aliases.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, changed)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = labels["configuration"][uploaded]
    mujoco.mj_kinematics(model, data)
    matched, points = set(), []
    for site in body.robot.morphology.sites:
        uploaded = body.manifest.link_aliases[site.body]
        eligible = [(name, point) for name, point in labels["points"].items() if point["body"] == uploaded
            and (site.semantic.value in point["semantics"] or
                 (site.semantic.value == "tip" and "tip_if_selected" in point["semantics"]))]
        if site.semantic.value == "grasp_point":
            eligible = [("gripping_region", labels["gripping_region"]["region_centre"])]
        assert eligible
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name)
        error, label = min((float(np.linalg.norm(data.site_xpos[sid] - p["world_m"])), n) for n, p in eligible)
        assert error <= labels["position_tolerance_m"]
        matched.add((label, site.semantic.value))
        points.append({"site": site.name, "semantic": site.semantic.value, "label": label,
                       "position_world_m": data.site_xpos[sid].tolist(), "error_m": error})
    required = {(name, semantic) for name, p in labels["points"].items() for semantic in p["semantics"] if semantic != "tip_if_selected"}
    assert required <= matched
    kinds = [e.kind.value for e in body.robot.morphology.effectors]
    assert kinds == [labels["expected_effector_kind"]]
    return {"fixture": source.parent.name, "source_sha256": hash_file(source),
        "independent_label_sha256": hash_file(source.parent / "expected.json"), "sites": points,
        "max_site_error_m": max(p["error_m"] for p in points), "effector_classes": kinds,
        "all_required_label_semantics_present": True, "position_tolerance_m": labels["position_tolerance_m"]}


def gallery(destination, admissions):
    cards = []
    for item in admissions:
        name = item["body"]
        if item["outcome"] != "admitted":
            cards.append(f'<section><h2>{html.escape(name)}</h2><p>Typed refusal: {html.escape(item["code"])}. '
                         'This is a runtime limitation, not a physical infeasibility certificate.</p></section>')
            continue
        manifest = json.loads((destination / "bodies" / name / "body-manifest.json").read_bytes())
        morphology = manifest["robot"]["morphology"]
        rows = ''.join(f'<tr><td>{html.escape(c["capability"])}</td><td>{html.escape(c["subject"])}</td>'
                       f'<td>{html.escape(c["status"])}</td><td>{html.escape(c["condition"])}</td></tr>' for c in manifest["capabilities"])
        kinds = ', '.join(e['kind'] for e in morphology['effectors'])
        cards.append(f'''<section><h2>{html.escape(name)}</h2><p>{len(morphology['chains'])} chains; {len(morphology['joints'])} joints; {html.escape(kinds)}.</p>
<img loading="lazy" src="media/{name}/inspection.gif" alt="Rotating body inspection for {html.escape(name)}">
<p><a href="bodies/{name}/body-manifest.json">Capability manifest and source aliases</a> · <a href="bodies/{name}/morphology-measurements.json">Measured workspace and sites</a> · <a href="media/{name}/frames.json">Replay frame map</a></p>
<details><summary>Capabilities and named uncertainties</summary><table><thead><tr><th>Capability</th><th>Subject</th><th>Status</th><th>Evidence needed</th></tr></thead><tbody>{rows}</tbody></table></details></section>''')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rigby body capability inspection</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#17202b;background:#fafafa}h1{font-size:30px}h2{font-size:23px}section{padding:24px 0;border-top:1px solid #ccd1d8}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px;vertical-align:top;border-bottom:1px solid #ddd}a{color:#165da0}summary{cursor:pointer}code{font-size:13px}</style>
<h1>Rigby body capability inspection</h1><p>Six procedural zoo bodies, three unchanged third-party descriptions and three independent analytic fixtures. Each six-second GIF orbits an archived static pose. Orange markers show derived sites; colored lines show chains; faint dots show sampled kinematics. Zero physics steps are executed.</p>
<p>Joint-name invariance and geometric measurements are engineering evidence. Physical grasping, payload capacity, hardware sensing, dynamic balance and locomotion require their own trials. Mesh colliders use the existing convex approximation; sampled workspace does not establish collision-free reachability.</p>
<p><a href="audit.json">Full audit</a> · <a href="geometry.json">Independent site errors</a> · <a href="invariance.json">120 renaming/permutation results</a> · <a href="contact-feasibility.json">Independent contact witnesses and unknown cases</a></p>'''
    (destination / "index.html").write_text(document + ''.join(cards) + '</html>\n', encoding="utf-8", newline="\n")


def main(destination):
    if destination.exists():
        raise ValueError("Choose a new immutable audit destination")
    destination.mkdir(parents=True)
    helpers = module(ROOT / "tests/test_body_capabilities.py", "g03_test_helpers")
    independent = module(FIXTURES / "verify_geometry.py", "g03_independent_geometry")
    packages = module(ROOT / "tests/fixtures/g03_third_party/package.py", "g03_third_party")
    independent_result = independent.verify()
    geometry = [geometry_check(FIXTURES / name / "robot.urdf") for name in independent_result["results"]]
    write_json(destination / "independent-geometry.json", independent_result)
    write_json(destination / "geometry.json", geometry)
    admissions, invariance, models = [], [], []
    with tempfile.TemporaryDirectory(prefix="g03-third-party-") as temporary:
        inputs = packages.extract(Path(temporary))
        sources = [(p.parent.name, p, "procedural_zoo") for p in helpers.ZOO]
        sources += [(n, inputs / n / "robot.urdf", "third_party") for n in ("iiwa7", "kuka_lwr", "so101", "panda")]
        sources += [(n, FIXTURES / n / "robot.urdf", "independent_geometry") for n in independent_result["results"]]
        for name, source, category in sources:
            print(f"Inspecting {name}", flush=True)
            try:
                metadata = capture_inspection(source, destination / "bodies" / name, title=name)
            except ModelIngestError as error:
                admissions.append({"body": name, "category": category, "outcome": "refused", "code": error.code.value,
                                   "detail": str(error), "details": error.details, "source_sha256": hash_file(source)})
                continue
            media = render_inspection(destination / "bodies" / name, destination / "media" / name)
            admissions.append({"body": name, "category": category, "outcome": "admitted", **metadata, "media": media})
            if category != "procedural_zoo":
                continue
            helpers.test_canonicalization_preserves_uploaded_kinematics_and_inertia(source)
            baseline = ingest_capability_body(source)
            expected = helpers.reference(baseline)
            directory = destination / "invariance" / name
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "baseline.npz", time_s=expected[0], qpos=expected[1], body_positions_m=expected[2])
            for index in range(20):
                variant = directory / f"variant-{index:02d}" / "robot.urdf"
                inverse = helpers.renamed_variant(source, index, variant)
                actual = ingest_capability_body(variant)
                result = helpers.reference(actual)
                assert actual.canonical.xml == baseline.canonical.xml
                assert actual.manifest.capability_hash() == baseline.manifest.capability_hash()
                assert actual.robot.mjcf_xml == baseline.robot.mjcf_xml
                assert result[3] == expected[3]
                assert all(np.array_equal(a, b) for a, b in zip(result[:3], expected[:3]))
                np.savez_compressed(variant.parent / "reference.npz", time_s=result[0], qpos=result[1], body_positions_m=result[2])
                write_json(variant.parent / "inverse-aliases.json", inverse)
                invariance.append({"body": name, "variant": index, "source_sha256": actual.manifest.source_urdf_sha256,
                    "canonical_sha256": actual.manifest.canonical_urdf_sha256, "capability_sha256": actual.manifest.capability_hash(),
                    "inverse_renamed_source_mechanics_exact": True, "runtime_mjcf_exact": True, "capabilities_exact": True,
                    "figure_sites_exact": True, "reference_clock_exact": True, "qpos_max_error": 0., "body_positions_max_error_m": 0.,
                    "reference_samples": len(result[0]), "reference_duration_s": float(result[0][-1]), "physics_steps": 0})
            models.append({"body": name, "uploaded_kinematics_verified_configurations": 5,
                           "link_pose_tolerance": 1e-12, "joint_axes_and_ranges_exact": True, "link_mass_and_inertia_exact": True})
    assert len(invariance) == 120
    assert sum(a["outcome"] == "admitted" and a["category"] == "third_party" for a in admissions) >= 3
    scan = _scan_for_robot_identifiers(ROOT / "src/rigby_general", [p.parent.name for p in helpers.ZOO])
    assert not scan
    contact = {"scope": "Independent analytic box-contact feasibility only; no dynamic lift or retention claims",
        "independent_fixture_results": independent_result["results"],
        "bodies": [{"body": a["body"], "physical_contact_feasibility": "unknown; requires G06 object-specific independent checks",
                    "task_infeasible": False} for a in admissions if a["category"] != "independent_geometry"]}
    write_json(destination / "contact-feasibility.json", contact)
    write_json(destination / "invariance.json", invariance)
    write_json(destination / "uploaded-model-equivalence.json", models)
    write_json(destination / "static-scan.json", {"zoo_identifier_hits_in_shared_source": scan,
        "scope": "Existing exact zoo-ID scan; new profile maps all uploaded joint/link labels to structural identities before shared measurement/control",
        "legacy_api_name_invariance_claimed": False, "dynamic_adversarial_name_variants": len(invariance)})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip())
    audit = {"schema": "g03.audit.v1", "source_commit": commit, "tracked_tree_dirty": dirty,
        "source_hashes": {p.relative_to(REPO).as_posix(): hash_file(p) for p in sorted((ROOT / "src").rglob("*.py"))},
        "platform": platform.platform(), "mujoco": mujoco.__version__, "numpy": np.__version__,
        "admissions": admissions, "permutations": len(invariance), "geometry_site_max_error_m": max(g["max_site_error_m"] for g in geometry),
        "total_inspection_frames": sum(a["media"]["frame_count"] for a in admissions if a["outcome"] == "admitted"),
        "physics_task_trials": 0, "api_calls": 0, "sealed_labels_opened": False}
    write_json(destination / "audit.json", audit)
    gallery(destination, admissions)
    files = {p.relative_to(destination).as_posix(): hash_file(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(destination / "release-index.json", {"schema": "g03.release.v1", "files": files})
    print(json.dumps({"release": str(destination), "source_commit": commit, "frames": audit["total_inspection_frames"],
                      "files": len(files), "release_index_sha256": hash_file(destination / "release-index.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    main(parser.parse_args().destination)
