from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .archive import utc_now
from .criteria import POC_ROOT, load_criteria
from .evidence import gesture_gate
from .models import Status
from .report import render_report


ACCEPTANCE_ROOT = POC_ROOT / "results" / "acceptance-runs"
EXPECTED_CLIPS = [f"s{index:02d}" for index in range(1, 21)]
REQUIRED_REVIEW_GATES = {"gesture_diversity", "semantic_forward_space", "egocentric_camera_orientation"}
DIVERSITY_OBSERVABLES = (
    "duration_s", "easing", "wrist_lateral_m", "wrist_height_m", "wrist_depth_m",
    "wrist_pitch_rad", "wrist_yaw_rad", "wrist_roll_rad", "elbow_swivel_rad", "torso_rotation_rad",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return sha256_value(
        [{"path": str(item.relative_to(path)).replace("\\", "/"), "sha256": sha256_file(item)} for item in files]
    )


def motion_sha256(clip_path: Path) -> str:
    clip = json.loads(clip_path.read_text(encoding="utf-8"))
    canonical_motion = {
        key: clip[key]
        for key in ("schema_version", "fps", "duration_s", "frames", "contacts")
    }
    return sha256_value(canonical_motion)


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_review_manifest(run_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {record["clip_id"]: record for record in records}
    manifest_records: list[dict[str, Any]] = []
    for clip_id in EXPECTED_CLIPS:
        source = by_id.get(clip_id, {})
        result_id = source.get("result_id")
        result_root = POC_ROOT / "results" / result_id if isinstance(result_id, str) else None
        clip_path = result_root / "clip.json" if result_root else None
        animation_path = result_root / "animation.glb" if result_root else None
        scene_path = result_root / "scene.json" if result_root else None
        program_path = result_root / "program.json" if result_root else None
        prompt = source.get("prompt")
        program: dict[str, Any] = {}
        if program_path and program_path.is_file():
            program = json.loads(program_path.read_text(encoding="utf-8"))
            if isinstance(program.get("source_text"), str):
                prompt = program["source_text"]
        clip = json.loads(clip_path.read_text(encoding="utf-8")) if clip_path and clip_path.is_file() else {}
        observables = clip.get("slider_observables", {}) if isinstance(clip, dict) else {}
        available = bool(clip_path and animation_path and scene_path and clip_path.is_file() and animation_path.is_file() and scene_path.is_file())
        avatar_path: Path | None = None
        if available and scene_path:
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            asset_uri = scene.get("rig", {}).get("asset_uri")
            avatar_path = POC_ROOT / asset_uri if isinstance(asset_uri, str) else None
            available = bool(avatar_path and avatar_path.is_file())
        manifest_records.append(
            {
                "clip_id": clip_id,
                "content_group": "diverse_hang_ten",
                "prompt": prompt,
                "variation_seed": program.get("seed"),
                "motion_profile": program.get("motion_profile"),
                "motion_descriptor": {
                    key: observables.get(key)
                    for key in DIVERSITY_OBSERVABLES
                },
                "result_id": result_id,
                "result_url": f"/api/v1/results/{result_id}" if result_id else None,
                "animation_url": f"/api/v1/results/{result_id}/animation.glb" if result_id else None,
                "available": available,
                "clip_sha256": sha256_file(clip_path) if available and clip_path else None,
                "motion_sha256": motion_sha256(clip_path) if available and clip_path else None,
                "animation_sha256": sha256_file(animation_path) if available and animation_path else None,
                "avatar_sha256": sha256_file(avatar_path) if available and avatar_path else None,
            }
        )
    viewer_root = POC_ROOT / "frontend" / "dist"
    if not viewer_root.is_dir():
        viewer_root = POC_ROOT / "frontend" / "src"
    core = {
        "schema_version": "1.1",
        "acceptance_run_id": run_id,
        "blinded": False,
        "prompt_visible": True,
        "content_group": "diverse_hang_ten",
        "review_viewer_sha256": sha256_tree(viewer_root),
        "records": manifest_records,
    }
    root_hash = sha256_value(core)
    return {**core, "manifest_sha256": root_hash, "manifest_root_sha256": root_hash}


def write_review_bundle(run_root: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    manifest = build_review_manifest(run_root.name, records)
    manifest_path = run_root / "blinded-gesture-manifest.json"
    template_path = run_root / "gesture-review-template.json"
    template = {
        "schema_version": "1.1",
        "acceptance_run_id": run_root.name,
        "manifest_sha256": manifest["manifest_sha256"],
        "instructions": "Rate each exact clip from 1 to 5 in both views with its source prompt visible.",
        "records": [
            {
                "clip_id": record["clip_id"],
                "result_id": record["result_id"],
                "content_group": record["content_group"],
                "clip_sha256": record["clip_sha256"],
                "motion_sha256": record["motion_sha256"],
                "animation_sha256": record["animation_sha256"],
                "avatar_sha256": record["avatar_sha256"],
                "egocentric_ratings": [],
                "orbit_ratings": [],
                "reviewer_notes": [],
            }
            for record in manifest["records"]
        ],
    }
    template["review_binding_sha256"] = sha256_value(
        {key: template[key] for key in ("schema_version", "acceptance_run_id", "manifest_sha256", "records")}
    )
    _write_new_json(manifest_path, manifest)
    _write_new_json(template_path, template)
    return manifest_path, template_path


def _load_verified_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "blinded-gesture-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "manifest_root_sha256"}}
    if digest != sha256_value(core) or manifest.get("manifest_root_sha256") != digest:
        raise ValueError("manifest digest does not match its contents")
    if manifest.get("acceptance_run_id") != run_root.name:
        raise ValueError("manifest belongs to a different acceptance run")
    if [record.get("clip_id") for record in manifest.get("records", [])] != EXPECTED_CLIPS:
        raise ValueError("manifest does not contain ordered s01 through s20")
    viewer_root = POC_ROOT / "frontend" / "dist"
    if not viewer_root.is_dir():
        viewer_root = POC_ROOT / "frontend" / "src"
    if sha256_tree(viewer_root) != manifest.get("review_viewer_sha256"):
        raise ValueError("review viewer changed after the manifest was created")
    for record in manifest["records"]:
        result_id = record.get("result_id")
        result_root = POC_ROOT / "results" / str(result_id)
        if not record.get("available") or not result_root.is_dir():
            raise ValueError(f"{record['clip_id']} has no available result")
        if sha256_file(result_root / "clip.json") != record.get("clip_sha256"):
            raise ValueError(f"{record['clip_id']} clip artifact changed after blinding")
        if motion_sha256(result_root / "clip.json") != record.get("motion_sha256"):
            raise ValueError(f"{record['clip_id']} canonical motion changed after blinding")
        if sha256_file(result_root / "animation.glb") != record.get("animation_sha256"):
            raise ValueError(f"{record['clip_id']} animation artifact changed after blinding")
        scene = json.loads((result_root / "scene.json").read_text(encoding="utf-8"))
        avatar = POC_ROOT / scene["rig"]["asset_uri"]
        if sha256_file(avatar) != record.get("avatar_sha256"):
            raise ValueError(f"{record['clip_id']} avatar changed after blinding")
        if record.get("content_group") != manifest.get("content_group"):
            raise ValueError(f"{record['clip_id']} content group does not match the manifest")
        program = json.loads((result_root / "program.json").read_text(encoding="utf-8"))
        if record.get("prompt") != program.get("source_text"):
            raise ValueError(f"{record['clip_id']} prompt changed after manifest creation")
        if record.get("motion_profile") != program.get("motion_profile"):
            raise ValueError(f"{record['clip_id']} motion profile changed after manifest creation")
        if record.get("variation_seed") != program.get("seed"):
            raise ValueError(f"{record['clip_id']} variation seed changed after manifest creation")
        clip = json.loads((result_root / "clip.json").read_text(encoding="utf-8"))
        observables = clip.get("slider_observables", {})
        descriptor = {key: observables.get(key) for key in DIVERSITY_OBSERVABLES}
        if record.get("motion_descriptor") != descriptor:
            raise ValueError(f"{record['clip_id']} motion descriptor changed after manifest creation")
    return manifest


def bind_review(run_id: str, review_path: Path) -> Path:
    run_root = ACCEPTANCE_ROOT / run_id
    manifest = _load_verified_manifest(run_root)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != "1.1":
        raise ValueError("review submission must use the exact-run schema version 1.1")
    if review.get("acceptance_run_id") != run_id:
        raise ValueError("review submission belongs to a different acceptance run")
    if review.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("review submission belongs to a different blinded manifest")
    raw_records = review.get("records", [])
    if [record.get("clip_id") for record in raw_records] != EXPECTED_CLIPS:
        raise ValueError("review does not contain ordered s01 through s20")
    manifest_by_id = {record["clip_id"]: record for record in manifest["records"]}
    bound_records = []
    for record in raw_records:
        clip_id = record["clip_id"]
        for field in ("egocentric_ratings", "orbit_ratings"):
            ratings = record.get(field)
            if not isinstance(ratings, list) or not ratings or any(
                not isinstance(rating, (int, float)) or isinstance(rating, bool) or not 1 <= rating <= 5
                for rating in ratings
            ):
                raise ValueError(f"{clip_id} has incomplete or invalid {field}")
        source = manifest_by_id[clip_id]
        bound_records.append(
            {
                "clip_id": clip_id,
                "result_id": source["result_id"],
                "content_group": source["content_group"],
                "clip_sha256": source["clip_sha256"],
                "motion_sha256": source["motion_sha256"],
                "animation_sha256": source["animation_sha256"],
                "avatar_sha256": source["avatar_sha256"],
                "egocentric_ratings": record["egocentric_ratings"],
                "orbit_ratings": record["orbit_ratings"],
                "reviewer_notes": record.get("reviewer_notes", []),
            }
        )
    core = {
        "schema_version": "1.1",
        "acceptance_run_id": run_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "records": bound_records,
    }
    bound = {**core, "bound_review_sha256": sha256_value(core), "bound_at": utc_now()}
    submissions = run_root / "review-submissions"
    sequence = len(list(submissions.glob("*-bound-review.json"))) + 1 if submissions.exists() else 1
    output = submissions / f"{sequence:06d}-bound-review.json"
    _write_new_json(output, bound)
    return output


def _load_verified_bound_review(run_root: Path, review_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    review_path = review_path.resolve()
    submissions = (run_root / "review-submissions").resolve()
    if submissions not in review_path.parents:
        raise ValueError("only a bound review archived under this run may be finalized")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    core = {key: review[key] for key in ("schema_version", "acceptance_run_id", "manifest_sha256", "records")}
    if review.get("bound_review_sha256") != sha256_value(core):
        raise ValueError("bound review digest does not match its contents")
    if review.get("acceptance_run_id") != run_root.name or review.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("bound review belongs to a different run or manifest")
    manifest_records = {record["clip_id"]: record for record in manifest["records"]}
    if [record.get("clip_id") for record in review["records"]] != EXPECTED_CLIPS:
        raise ValueError("bound review record set is invalid")
    for record in review["records"]:
        expected = manifest_records[record["clip_id"]]
        for key in ("result_id", "content_group", "clip_sha256", "motion_sha256", "animation_sha256", "avatar_sha256"):
            if record.get(key) != expected.get(key):
                raise ValueError(f"{record['clip_id']} is bound to a different artifact")
    return review


def _recompute_certification(index: dict[str, Any], required_runs: int) -> None:
    streak = 0
    for run in reversed([item for item in index.get("runs", []) if not item.get("synthetic")]):
        if not run.get("passed"):
            break
        streak += 1
    index["consecutive_passes"] = streak
    index["certified"] = streak >= required_runs


def finalize_review(run_id: str, bound_review_path: Path) -> Path:
    run_root = ACCEPTANCE_ROOT / run_id
    manifest = _load_verified_manifest(run_root)
    review = _load_verified_bound_review(run_root, bound_review_path, manifest)
    index_path = ACCEPTANCE_ROOT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    matches = [record for record in index.get("runs", []) if record.get("id") == run_id]
    if len(matches) != 1:
        raise ValueError("acceptance ledger does not contain exactly one target run")
    ledger_record = matches[0]
    if ledger_record.get("review_finalized"):
        raise ValueError("this run already has finalized human review evidence")
    original_report_path = ACCEPTANCE_ROOT / ledger_record["report"]
    original_report = json.loads(original_report_path.read_text(encoding="utf-8"))
    archived_gate_names = {gate.get("gate") for gate in original_report.get("gates", [])}
    missing_required_gates = REQUIRED_REVIEW_GATES - archived_gate_names
    if missing_required_gates:
        raise ValueError(
            f"automated report predates required review gates: {sorted(missing_required_gates)}"
        )
    gesture_items = []
    for record in manifest["records"]:
        result_root = POC_ROOT / "results" / record["result_id"]
        gesture_items.append(
            {
                "id": record["clip_id"],
                "program": json.loads((result_root / "program.json").read_text(encoding="utf-8")),
                "clip": json.loads((result_root / "clip.json").read_text(encoding="utf-8")),
            }
        )
    criteria = load_criteria()
    gesture = gesture_gate(gesture_items, review, criteria["gesture"])
    revised = deepcopy(original_report)
    gate_indexes = [index for index, gate in enumerate(revised["gates"]) if gate.get("gate") == "gesture"]
    if len(gate_indexes) != 1:
        raise ValueError("archived report does not contain exactly one gesture gate")
    revised["gates"][gate_indexes[0]] = gesture.to_dict()
    revised["passed"] = bool(
        not revised.get("synthetic") and revised["gates"]
        and all(gate.get("status") == Status.PASS.value for gate in revised["gates"])
    )
    revised["review_finalization"] = {
        "finalized_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "bound_review_sha256": review["bound_review_sha256"],
        "original_report_sha256": sha256_file(original_report_path),
    }
    revisions = run_root / "revisions"
    revision_number = len([path for path in revisions.glob("*-human-review") if path.is_dir()]) + 1 if revisions.exists() else 1
    revision_root = revisions / f"{revision_number:06d}-human-review"
    revision_root.mkdir(parents=True, exist_ok=False)
    submitted_path = revision_root / "submitted-review.json"
    report_path = revision_root / "report.json"
    audit_path = revision_root / "audit.json"
    _write_new_json(submitted_path, review)
    _write_new_json(report_path, revised)
    audit = {
        "schema_version": "1.0",
        "event": "human_review_finalized",
        "acceptance_run_id": run_id,
        "created_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "bound_review_sha256": review["bound_review_sha256"],
        "original_report": str(original_report_path.relative_to(POC_ROOT)),
        "original_report_sha256": sha256_file(original_report_path),
        "revised_report_sha256": sha256_file(report_path),
        "changed_gate": "gesture",
    }
    _write_new_json(audit_path, audit)
    html_path = revision_root / "report.html"
    html_path.write_text(render_report(revised), encoding="utf-8")
    ledger_record.update(
        {
            "passed": revised["passed"],
            "report": str(report_path.relative_to(ACCEPTANCE_ROOT)).replace("\\", "/"),
            "html": str(html_path.relative_to(ACCEPTANCE_ROOT)).replace("\\", "/"),
            "review_finalized": True,
            "review_manifest_sha256": manifest["manifest_sha256"],
            "review_evidence_sha256": review["bound_review_sha256"],
            "original_report": str(original_report_path.relative_to(ACCEPTANCE_ROOT)).replace("\\", "/"),
        }
    )
    _recompute_certification(index, criteria["runs_required"])
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bind and finalize exact-run blinded gesture review evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bind = subparsers.add_parser("bind", help="bind downloaded ratings to one immutable run manifest")
    bind.add_argument("--run-id", required=True)
    bind.add_argument("--input", required=True, type=Path)
    finalize = subparsers.add_parser("finalize", help="recompute only the archived run's gesture gate")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = bind_review(args.run_id, args.input) if args.command == "bind" else finalize_review(args.run_id, args.input)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
