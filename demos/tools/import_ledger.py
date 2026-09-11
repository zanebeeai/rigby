"""Seed the registry from the results ledger (docs/results), once.

The ledger catalogued media that existed before the registry did: the
any-robot zoo demo GIFs with their demo-manifest, the humanoid README GIFs
with theirs, and Angelo's gripper recordings. This turns each into a
registry entry that references the media where it already lives, with the
best provenance the source records carry: the humanoid demo-manifest names a
compiler commit; the zoo manifest names none, so the entry says so.

Re-running is safe: an entry whose id already exists is left alone.

    uv run python demos/tools/import_ledger.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_tools import REGISTRY, ROOT, SCHEMA, sha256_file, slug  # noqa: E402

LEDGER_MEDIA = ROOT / "docs" / "results" / "media"


def entry(*, title, prompt, tier, embodiment, kind, who, produced, commit, branch, how, files, outcome=None, notes="", tags=(), role="clip"):
    digest = hashlib.sha256()
    media = []
    for f in files:
        p = ROOT / f
        h = sha256_file(p)
        digest.update(bytes.fromhex(h))
        media.append({"path": f, "bytes": p.stat().st_size, "sha256": h, "role": role})
    did = f"{produced[:10]}-{slug(prompt)}-{digest.hexdigest()[:8]}"
    out = REGISTRY / f"{did}.json"
    if out.exists():
        return False
    e = {
        "schema": SCHEMA, "id": did, "title": title, "prompt": prompt, "tier": tier, "embodiment": embodiment,
        "kind": kind, "who": who, "produced_at": produced,
        "source": {"commit": commit, "branch": branch, "dirty": False, "how": how},
        "outcome": outcome, "notes": notes, "media": media, "payload": None, "tags": list(tags),
    }
    out.write_text(json.dumps(e, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> int:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    n = 0
    # any-robot zoo demos
    zoo = json.loads((LEDGER_MEDIA / "any-robot-zoo-demos" / "demo-manifest.json").read_text(encoding="utf-8"))
    for c in zoo["clips"]:
        n += entry(
            title=f"{c['robot_id']}: {c['prompt']}",
            prompt=c["prompt"], tier="any-robot", embodiment=c["robot_id"], kind="gif", who="zanebeeai",
            produced="2026-08-27T00:00:00Z", commit="unrecorded", branch="feat/general-package (rigby-poc-general)",
            how="uv run python scripts/build_demos.py  # any-robot, pre-split layout; demo-manifest.json records no commit",
            files=[f"docs/results/media/any-robot-zoo-demos/{c['gif']}"],
            outcome={"state": "ok", "text": f"{c['certified_primitives_used']} certified primitive(s), tracking error {c['tracking_error_m'] * 1000:.1f} mm, {c['dof']} DOF"},
            notes=f"role-normalized hash {c['role_normalized_hash'][:12]}: the same words read the same way on every body.",
            tags=("zoo", "ledger-import"),
        )
    # humanoid README demos
    hm = json.loads((ROOT / "humanoid" / "docs" / "media" / "demo-manifest.json").read_text(encoding="utf-8"))
    for d in hm["demos"]:
        n += entry(
            title=d["prompt"][:60] + ("…" if len(d["prompt"]) > 60 else ""),
            prompt=d["prompt"], tier="humanoid", embodiment="mesh2motion-human-vrm1", kind="gif", who="zanebeeai",
            produced=f"{hm['generated_on']}T00:00:00Z", commit=hm["compiler_commit"], branch="main",
            how=f"cd humanoid && uv run python -m evals.render_demo_gif {d['result_id']} docs/media/{d['file']}",
            files=[f"humanoid/docs/media/{d['file']}"],
            outcome={"state": "ok", "text": f"{d['intent']}, {d['duration_s']} s at {d['clip_fps']} fps, {d['planner_provider']}/{d['planner_model']} seed {d['seed']}"},
            notes=f"result id {d['result_id']} in the humanoid archive (rigby-poc/results).",
            tags=("readme", "ledger-import"),
        )
    for c in hm.get("comparisons", []):
        n += entry(
            title="Before and after: the humeral-roll fix",
            prompt=c["prompt"], tier="humanoid", embodiment="mesh2motion-human-vrm1", kind="gif", who="tpypan",
            produced=f"{hm['generated_on']}T00:00:00Z", commit=f"{c['before_commit']}..{c['after_commit']}", branch="main",
            how=f"cd humanoid && uv run python -m evals.render_comparison_gif <before-result> <after-result> docs/media/{c['file']} --view {c['view']}",
            files=[f"humanoid/docs/media/{c['file']}"],
            outcome={"state": "ok", "text": c["note"]},
            notes=f"program source {c['program_source']}.", tags=("readme", "comparison", "ledger-import"), role="comparison",
        )
    # gripper recordings
    g = "docs/results/media/gripper-milestones/"
    videos = [
        ("videos--01-lift-and-place-known-positions.webm", "Pick the block off the bench and place it in the bin, object pose read from the simulator",
         "pick up the block and put it in the bin", {"state": "ok", "text": "placed, 28.9 cm peak lift, 0.036 mm deepest penetration; the reference that proves the arm can do the task"}),
        ("videos--02-lift-and-place-camera-only.webm", "The same task with one wrist camera as the only percept",
         "pick up the block and put it in the bin", {"state": "ok", "text": "placed; 9 of 12 placements and sizes across the sweep; second look brings 38 mm error to 7 mm"}),
        ("videos--gripper-two-cameras.webm", "Pick-and-place seen through the corner camera with the wrist camera inset",
         "pick up the block and put it in the bin", {"state": "ok", "text": "the inset is the image the controller segments, frame for frame"}),
        ("videos--gripper-vlm-directed.webm", "A VLM names the numbers, a greedy search chases them",
         "pick up the block and put it in the bin", {"state": "ok", "text": "model-directed run recorded as the model saw it"}),
        ("videos--03-cabinet-attempt-not-working.webm", "The cabinet: the arm finds the handle and closes on it, and the door does not open",
         "open the cabinet and take out what is inside", {"state": "failed", "text": "unfinished, kept because it is the honest state of it"}),
    ]
    for f, title, prompt, outcome in videos:
        n += entry(
            title=title, prompt=prompt, tier="humanoid", embodiment="gripper", kind="video", who="AngeloWhey",
            produced="2026-09-09T00:00:00Z", commit="f00861d", branch="grasp/auto-lift",
            how="npm run dev --prefix frontend; open gripper.html?run=<id> and screen-record; recorder settings not committed",
            files=[g + f], outcome=outcome,
            notes="Torque-driven three-segment gripper; runs in milestones/runs/*.json on the branch.", tags=("gripper", "ledger-import"),
        )
    print(f"{n} entries written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
