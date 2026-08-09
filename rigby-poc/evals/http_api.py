from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ApiResponse:
    status: int | None
    body: dict[str, Any] | None
    binary: bytes | None
    elapsed_ms: float
    error: str | None = None
    content_type: str | None = None

    def evidence(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "body": self.body,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


class RigbyApi:
    def __init__(self, base_url: str, timeout_s: float = 30.0, provider: str = "openai"):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.provider = provider

    def _post(self, path: str, payload: dict[str, Any]) -> ApiResponse:
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=encoded,
            headers={"Content-Type": "application/json", "Accept": "application/json, model/gltf-binary"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
                elapsed = (time.perf_counter() - started) * 1000
                content_type = response.headers.get_content_type()
                if content_type in {"model/gltf-binary", "application/octet-stream"}:
                    return ApiResponse(response.status, None, raw, elapsed, content_type=content_type)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    return ApiResponse(response.status, None, None, elapsed, f"invalid JSON: {exc}", content_type)
                return ApiResponse(response.status, body, None, elapsed, content_type=content_type)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            elapsed = (time.perf_counter() - started) * 1000
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = {"error": raw.decode("utf-8", errors="replace")[:1000]}
            return ApiResponse(exc.code, body, None, elapsed, f"HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return ApiResponse(None, None, None, elapsed, str(exc))

    def plan(self, prompt: str, scene: dict[str, Any]) -> ApiResponse:
        return self._post("/api/v1/plan", {"text": prompt, "scene": scene, "provider": self.provider})

    def compile(
        self, program: dict[str, Any], scene: dict[str, Any], parameter_overrides: dict[str, Any] | None = None,
        *, persist: bool = True,
    ) -> ApiResponse:
        payload: dict[str, Any] = {"program": program, "scene": scene, "persist": persist}
        if parameter_overrides:
            payload["parameter_overrides"] = parameter_overrides
        return self._post("/api/v1/compile", payload)

    def export(self, result_id: str) -> ApiResponse:
        return self._post("/api/v1/export", {"result_id": result_id})
