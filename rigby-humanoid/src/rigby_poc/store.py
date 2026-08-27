from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiler import PROJECT_ROOT, RIG_PROFILE
from .exporter import export_glb
from .models import ClipResult, CompileRequest, ResultSummary


#: The sequence number a result directory is named for. Six digits, then the slug.
_SEQUENCE_PREFIX = re.compile(r"^(\d{6})-")


class ResultStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT / "results"
        self.index_path = self.root / "index.json"
        self._lock = threading.Lock()

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]
        return slug or "motion"

    def _index_document(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"schema_version": "1.0", "next_sequence": 1, "results": []}
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return {"schema_version": "1.0", "next_sequence": len(value) + 1, "results": value}
        return value

    def _next_sequence(self, recorded: int) -> int:
        """The next id, advanced past every sequence already on disk.

        ``persist`` creates the directory, writes seven files, and updates
        ``index.json`` **last**. A run killed in between leaves a directory the
        index does not know about while ``next_sequence`` still points at it, so
        the store hands out an id that already exists -- and keeps handing out
        that same id, because only a successful persist advances the counter.
        The worktree cannot pass the suite again until a human deletes the
        folder, and it surfaces as a ``FileExistsError`` raised under
        ``flywheel.py`` naming neither this store nor the kill that caused it.

        Healing means skipping the id, not reusing it. ``exist_ok=True`` would
        be the wrong fix twice over: it writes a live result into a dead one's
        directory, so the seven files describe two different runs, and it turns
        a loud failure into a silent corruption. The orphan is left alone
        precisely because a killed run's directory is of unknown completeness --
        deleting it is ``prune_run.py``'s decision, not this write path's.

        The scan is keyed on the sequence number rather than the whole
        directory name. A slug is derived from the prompt, so an orphan from a
        *different* prompt does not collide on ``mkdir`` at all -- it silently
        yields two directories sharing one sequence number, with the index
        claiming that number for whichever wrote second. That is the quieter
        half of the same bug.
        """

        highest = 0
        for child in self.root.iterdir():
            match = _SEQUENCE_PREFIX.match(child.name)
            if match is not None and child.is_dir():
                highest = max(highest, int(match.group(1)))
        return max(recorded, highest + 1)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def persist(self, request: CompileRequest, clip: ClipResult) -> str:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            index_document = self._index_document()
            index = index_document["results"]
            next_number = self._next_sequence(int(index_document["next_sequence"]))
            result_id = f"{next_number:06d}-{self._slug(request.program.source_text)}"
            folder = self.root / result_id
            folder.mkdir(parents=False, exist_ok=False)
            self._write_json(folder / "request.json", request.model_dump(mode="json"))
            self._write_json(folder / "scene.json", request.scene.model_dump(mode="json"))
            self._write_json(folder / "program.json", request.program.model_dump(mode="json"))
            self._write_json(folder / "clip.json", clip.model_dump(mode="json"))
            self._write_json(folder / "metrics.json", clip.metrics)
            self._write_json(folder / "provenance.json", clip.provenance.model_dump(mode="json"))
            # Typed compilation failures are still useful, replayable evidence,
            # but they have no frames and therefore cannot produce a valid GLB.
            # Persist their diagnostics without converting the failure into a
            # server-side exporter exception.
            if clip.success and clip.frames:
                rig_profile = json.loads(RIG_PROFILE.read_text(encoding="utf-8"))
                export_glb(
                    clip,
                    request.scene,
                    folder / "animation.glb",
                    PROJECT_ROOT / request.scene.rig.asset_uri,
                    rig_profile,
                )
            summary = ResultSummary(
                id=result_id,
                created_at=datetime.now(UTC).isoformat(),
                prompt=request.program.source_text,
                intent=request.program.intent,
                success=clip.success,
                duration_s=clip.duration_s,
                metrics=clip.metrics,
            )
            index.append(summary.model_dump(mode="json"))
            index_document["next_sequence"] = next_number + 1
            self._write_json(self.index_path, index_document)
            return result_id

    def list(self) -> list[ResultSummary]:
        summaries: list[ResultSummary] = []
        for item in reversed(self._index_document()["results"]):
            value = dict(item)
            if not value.get("duration_s"):
                clip_path = self.root / str(value.get("id", "")) / "clip.json"
                if clip_path.is_file():
                    value["duration_s"] = json.loads(clip_path.read_text(encoding="utf-8")).get("duration_s", 0.0)
            summaries.append(ResultSummary.model_validate(value))
        return summaries

    def detail(self, result_id: str) -> dict[str, Any] | None:
        folder = self.root / result_id
        if not folder.is_dir() or folder.parent.resolve() != self.root.resolve():
            return None
        animation = folder / "animation.glb"
        return {
            "id": result_id,
            "scene": json.loads((folder / "scene.json").read_text(encoding="utf-8")),
            "program": json.loads((folder / "program.json").read_text(encoding="utf-8")),
            "clip": json.loads((folder / "clip.json").read_text(encoding="utf-8")),
            "animation_url": f"/api/v1/results/{result_id}/animation.glb" if animation.is_file() else None,
        }

    def animation_path(self, result_id: str) -> Path | None:
        path = self.root / result_id / "animation.glb"
        if path.is_file() and path.parent.parent.resolve() == self.root.resolve():
            return path
        return None
