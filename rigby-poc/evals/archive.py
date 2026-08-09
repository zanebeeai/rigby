from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:48] or "animation"


class ResultArchive:
    """Append-only animation archive; acceptance runs have a separate ledger."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / "index.json"
        if not self.index_path.exists():
            _write_json(
                self.index_path,
                {"schema_version": "1.0", "next_sequence": 1, "results": []},
            )

    def _load(self) -> dict[str, Any]:
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            highest = max((int(item["id"].split("-", 1)[0]) for item in value), default=0)
            return {"schema_version": "1.0", "next_sequence": highest + 1, "results": value}
        return value

    def add(
        self,
        *,
        label: str,
        request: dict[str, Any],
        scene: dict[str, Any],
        program: dict[str, Any],
        clip: dict[str, Any],
        metrics: dict[str, Any],
        provenance: dict[str, Any],
        glb: bytes | None = None,
    ) -> str:
        index = self._load()
        sequence = int(index.get("next_sequence", 1))
        result_id = f"{sequence:06d}-{_slug(label)}"
        result_dir = self.root / result_id
        if result_dir.exists():
            raise FileExistsError(f"refusing to overwrite archived result {result_id}")
        result_dir.mkdir(parents=False)
        values = {
            "request.json": request,
            "scene.json": scene,
            "program.json": program,
            "clip.json": clip,
            "metrics.json": metrics,
            "provenance.json": provenance,
        }
        for name, value in values.items():
            _write_json(result_dir / name, value)
        if glb is not None:
            (result_dir / "animation.glb").write_bytes(glb)
        record = {
            "id": result_id,
            "sequence": sequence,
            "created_at": utc_now(),
            "label": label,
            "path": result_id,
            "has_glb": glb is not None,
            "request": f"{result_id}/request.json",
            "scene": f"{result_id}/scene.json",
            "program": f"{result_id}/program.json",
            "clip": f"{result_id}/clip.json",
            "metrics": f"{result_id}/metrics.json",
            "provenance": f"{result_id}/provenance.json",
        }
        if glb is not None:
            record["animation"] = f"{result_id}/animation.glb"
        index.setdefault("results", []).append(record)
        index["next_sequence"] = sequence + 1
        _write_json(self.index_path, index)
        return result_id

    def attach_glb(self, result_id: str, data: bytes) -> None:
        path = self.root / result_id / "animation.glb"
        if path.exists():
            if path.read_bytes() == data:
                return
            raise FileExistsError(f"refusing to overwrite {path} with different bytes")
        path.write_bytes(data)
        index = self._load()
        matches = [record for record in index.get("results", []) if record.get("id") == result_id]
        if len(matches) != 1:
            path.unlink(missing_ok=True)
            raise KeyError(f"archive index does not contain exactly one {result_id}")
        matches[0]["has_glb"] = True
        matches[0]["animation"] = f"{result_id}/animation.glb"
        _write_json(self.index_path, index)


class AcceptanceLedger:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / "index.json"
        if not self.index_path.exists():
            _write_json(
                self.index_path,
                {
                    "schema_version": "1.0",
                    "next_sequence": 1,
                    "certified": False,
                    "consecutive_passes": 0,
                    "runs": [],
                },
            )

    def record(self, report: dict[str, Any], html: str, required_runs: int) -> dict[str, Any]:
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        sequence = int(index.get("next_sequence", 1))
        run_id = f"{sequence:06d}"
        run_dir = self.root / run_id
        run_dir.mkdir()
        report = dict(report)
        report["run_id"] = run_id
        _write_json(run_dir / "report.json", report)
        (run_dir / "report.html").write_text(html, encoding="utf-8")
        index.setdefault("runs", []).append(
            {
                "id": run_id,
                "completed_at": report["completed_at"],
                "passed": bool(report.get("passed")),
                "synthetic": bool(report.get("synthetic")),
                "report": f"{run_id}/report.json",
                "html": f"{run_id}/report.html",
            }
        )
        index["next_sequence"] = sequence + 1
        real_runs = [r for r in index["runs"] if not r.get("synthetic")]
        streak = 0
        for run in reversed(real_runs):
            if not run.get("passed"):
                break
            streak += 1
        index["consecutive_passes"] = streak
        index["certified"] = streak >= required_runs
        _write_json(self.index_path, index)
        return index
