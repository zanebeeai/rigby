"""Secret-safe discovery for the supported benchmark's external-model authority."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rigby_v2.hashing import content_hash


_DOTENV_LINE = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
)


@dataclass(frozen=True, slots=True)
class LiveModelConfiguration:
    """Resolved values whose representation and diagnostic never reveal values."""

    api_key: str = field(repr=False)
    planner_model: str = field(repr=False)
    judge_model: str = field(repr=False)
    base_url: str = field(repr=False)
    sources: tuple[str, ...]

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.planner_model and self.judge_model)

    @property
    def missing_authority(self) -> tuple[str, ...]:
        missing = []
        if not self.api_key:
            missing.append("api_credential")
        if not self.planner_model:
            missing.append("planner_model_configuration")
        if not self.judge_model:
            missing.append("judge_model_configuration")
        return tuple(missing)

    @property
    def configuration_fingerprint(self) -> str:
        """Bind non-secret routing configuration without persisting its values."""

        return content_hash(
            {
                "planner_model": self.planner_model,
                "judge_model": self.judge_model,
                "base_url": self.base_url,
                "credential_present": bool(self.api_key),
            }
        )

    def diagnostic(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "configured": self.configured,
            "api_credential_present": bool(self.api_key),
            "planner_model_present": bool(self.planner_model),
            "judge_model_present": bool(self.judge_model),
            "base_url_present": bool(self.base_url),
            "configuration_sources": list(self.sources),
            "missing_authority": list(self.missing_authority),
            "configuration_fingerprint": self.configuration_fingerprint,
        }


def _dotenv_values(paths: Sequence[Path]) -> tuple[dict[str, str], bool]:
    values: dict[str, str] = {}
    found = False
    for candidate in paths:
        path = candidate.resolve()
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            continue
        found = True
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _DOTENV_LINE.fullmatch(raw_line)
            if match is None:
                continue
            key, raw_value = match.groups()
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values.setdefault(key, value.strip())
    return values, found


def discover_live_model_configuration(
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Sequence[Path] = (),
) -> LiveModelConfiguration:
    """Resolve process-first configuration while exposing only presence metadata."""

    process = dict(os.environ if environ is None else environ)
    dotenv, dotenv_found = _dotenv_values(dotenv_paths)

    def resolve(*keys: str) -> tuple[str, str | None]:
        for key in keys:
            value = process.get(key, "").strip()
            if value:
                return value, "process_environment"
        for key in keys:
            value = dotenv.get(key, "").strip()
            if value:
                return value, "workspace_dotenv"
        return "", None

    api_key, api_source = resolve("OPENAI_API_KEY")
    planner_model, planner_source = resolve(
        "OPENAI_PLANNER_MODEL", "RIGBY_V2_PLANNER_MODEL"
    )
    judge_model, judge_source = resolve(
        "OPENAI_JUDGE_MODEL", "RIGBY_V2_JUDGE_MODEL"
    )
    base_url, base_url_source = resolve("OPENAI_BASE_URL")
    sources = {
        value
        for value in (api_source, planner_source, judge_source, base_url_source)
        if value is not None
    }
    if dotenv_found:
        sources.add("workspace_dotenv_discovered")
    return LiveModelConfiguration(
        api_key=api_key,
        planner_model=planner_model,
        judge_model=judge_model,
        base_url=base_url,
        sources=tuple(sorted(sources)),
    )


__all__ = ["LiveModelConfiguration", "discover_live_model_configuration"]
