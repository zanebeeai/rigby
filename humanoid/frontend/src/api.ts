import { downloadBlob, unwrapClip, unwrapProgram, unwrapResults } from "./motion";
import type {
  ClipResult,
  CompileRequest,
  ExportRequest,
  MotionProgram,
  PlanRequest,
  PipelineRun,
  PipelineRunRequest,
  ResultSummary,
} from "./types";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch(path, { ...init, signal: controller.signal });
    if (!response.ok) {
      const text = await response.text();
      let detail: unknown = text;
      try {
        detail = JSON.parse(text);
      } catch {
        // Preserve the server's plain-text error.
      }
      throw new ApiError(`Request failed (${response.status})`, response.status, detail);
    }
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("The backend did not respond within 30 seconds.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await request(path, init);
  return (await response.json()) as T;
}

function post(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const api = {
  async plan(body: PlanRequest): Promise<MotionProgram> {
    return unwrapProgram(await json<unknown>("/api/v1/plan", post(body)));
  },

  async compile(body: CompileRequest): Promise<ClipResult> {
    return unwrapClip(await json<unknown>("/api/v1/compile", post(body)));
  },

  async listResults(): Promise<ResultSummary[]> {
    return unwrapResults(await json<unknown>("/api/v1/results"));
  },

  async getResult(id: string): Promise<ClipResult> {
    return unwrapClip(await json<unknown>(`/api/v1/results/${encodeURIComponent(id)}`));
  },

  async exportResult(body: ExportRequest): Promise<void> {
    const response = await request("/api/v1/export", post(body));
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("model/gltf-binary") || contentType.includes("application/octet-stream")) {
      downloadBlob(await response.blob(), `rigby-${body.result_id}.glb`);
      return;
    }
    const payload = (await response.json()) as Record<string, unknown>;
    const url = String(payload.url ?? payload.download_url ?? payload.glb_url ?? "");
    if (url) {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `rigby-${body.result_id}.glb`;
      anchor.click();
      return;
    }
    throw new ApiError("Export completed without a downloadable artifact.", undefined, payload);
  },

  async registerDemo(resultId: string): Promise<DemoRegistration> {
    return json<DemoRegistration>(`/api/v1/results/${encodeURIComponent(resultId)}/demo`, post({}));
  },

  async startPipeline(body: PipelineRunRequest): Promise<PipelineRun> {
    return json<PipelineRun>("/api/v1/pipeline-runs", post(body));
  },

  async getPipeline(runId: string): Promise<PipelineRun> {
    return json<PipelineRun>(`/api/v1/pipeline-runs/${encodeURIComponent(runId)}`);
  },
};

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail as Record<string, unknown> | string | undefined;
    if (typeof detail === "string" && detail) return detail;
    if (detail && typeof detail === "object") {
      return String(detail.detail ?? detail.message ?? detail.error ?? error.message);
    }
  }
  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

/** What the server says after it wrote a registry entry for a result. */
export interface DemoRegistration {
  id: string;
  registry: string;
  who: string;
  source: { commit: string; branch: string; dirty: boolean; how: string };
  index: string;
}
