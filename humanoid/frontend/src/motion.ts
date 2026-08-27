import type {
  ClipFailure,
  ClipResult,
  ContactEvent,
  MotionFrame,
  MotionProgram,
  ResultSummary,
} from "./types";

function objectRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function finite(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function arrayValue<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function vec3(value: unknown): [number, number, number] | undefined {
  if (Array.isArray(value) && value.length >= 3) return [finite(value[0], 0), finite(value[1], 0), finite(value[2], 0)];
  const item = objectRecord(value);
  if ([item.x, item.y, item.z].every((part) => typeof part === "number")) {
    return [finite(item.x, 0), finite(item.y, 0), finite(item.z, 0)];
  }
  return undefined;
}

function quat(value: unknown): [number, number, number, number] | undefined {
  if (Array.isArray(value) && value.length >= 4) return [finite(value[0], 0), finite(value[1], 0), finite(value[2], 0), finite(value[3], 1)];
  const item = objectRecord(value);
  if ([item.x, item.y, item.z, item.w].every((part) => typeof part === "number")) {
    return [finite(item.x, 0), finite(item.y, 0), finite(item.z, 0), finite(item.w, 1)];
  }
  return undefined;
}

function normalizeFrame(value: unknown, index: number, fps: number): MotionFrame {
  const frame = objectRecord(value);
  const bones: MotionFrame["bones"] = {};
  for (const [name, rawPose] of Object.entries(objectRecord(frame.bones ?? frame.bone_rotations))) {
    const pose = objectRecord(rawPose);
    bones[name] = { rotation: quat(pose.rotation ?? rawPose), position: vec3(pose.position ?? pose.translation) };
  }
  const objects: NonNullable<MotionFrame["objects"]> = {};
  for (const [name, rawPose] of Object.entries(objectRecord(frame.objects))) {
    const pose = objectRecord(rawPose);
    objects[name] = {
      position: vec3(pose.position ?? pose.translation),
      rotation: quat(pose.rotation),
      scale: vec3(pose.scale),
    };
  }
  return {
    time: finite(frame.time ?? frame.time_s, index / fps),
    bones,
    objects,
  };
}

export function unwrapProgram(payload: unknown): MotionProgram {
  const root = objectRecord(payload);
  const candidate = root.motion_program ?? root.program ?? root.data ?? root;
  return objectRecord(candidate) as MotionProgram;
}

export function unwrapClip(payload: unknown): ClipResult {
  const root = objectRecord(payload);
  const raw = objectRecord(root.clip_result ?? root.clip ?? root.result ?? root.data ?? root);
  const rootProgram = objectRecord(root.program);
  const fps = finite(raw.fps, 30);
  const frames = arrayValue<unknown>(raw.frames ?? raw.keyframes).map((frame, index) => normalizeFrame(frame, index, fps));
  const duration = finite(
    raw.duration ?? raw.duration_s,
    frames.length > 0 ? frames[frames.length - 1]?.time ?? frames.length / fps : 0,
  );
  const failureRaw = objectRecord(raw.failure ?? raw.error);
  const failure: ClipFailure | null = Object.keys(failureRaw).length
    ? {
        code: String(failureRaw.code ?? "compile_failed"),
        message: String(failureRaw.message ?? failureRaw.detail ?? "Compilation failed"),
        details: failureRaw.details,
      }
    : null;
  return {
    ...raw,
    id: root.result_id ? String(root.result_id) : root.id ? String(root.id) : raw.id ? String(raw.id) : undefined,
    status: String(raw.status ?? (raw.success === false || failure ? "failed" : "success")),
    duration,
    fps,
    frames,
    metrics: objectRecord(raw.metrics) as ClipResult["metrics"],
    sliderObservables: objectRecord(raw.sliderObservables ?? raw.slider_observables) as Record<string, number>,
    contacts: arrayValue<unknown>(raw.contacts ?? raw.contact_events).map((value) => {
      const contact = objectRecord(value);
      return {
        ...contact,
        time: finite(contact.time ?? contact.time_s, 0),
        position: vec3(contact.position),
        bodyA: contact.bodyA ? String(contact.bodyA) : contact.hand ? String(contact.hand) : undefined,
        bodyB: contact.bodyB ? String(contact.bodyB) : contact.object_id ? String(contact.object_id) : undefined,
      } as ContactEvent;
    }),
    failure,
    provenance: objectRecord(raw.provenance),
    program: raw.program
      ? (objectRecord(raw.program) as MotionProgram)
      : Object.keys(rootProgram).length
        ? (rootProgram as MotionProgram)
        : undefined,
    prompt: raw.prompt
      ? String(raw.prompt)
      : root.prompt
        ? String(root.prompt)
        : rootProgram.source_text
          ? String(rootProgram.source_text)
          : undefined,
    createdAt: raw.createdAt ? String(raw.createdAt) : undefined,
    created_at: raw.created_at ? String(raw.created_at) : undefined,
    glbUrl: raw.glbUrl ? String(raw.glbUrl) : undefined,
    glb_url: raw.glb_url ? String(raw.glb_url) : undefined,
  };
}

export function unwrapResults(payload: unknown): ResultSummary[] {
  const root = objectRecord(payload);
  const rows = Array.isArray(payload)
    ? payload
    : arrayValue(root.results ?? root.items ?? root.data);
  return rows.map((row, index) => {
    const item = objectRecord(row);
    return {
      ...item,
      id: String(item.id ?? item.result_id ?? `result-${index + 1}`),
      prompt: item.prompt ? String(item.prompt) : undefined,
      status: item.status ? String(item.status) : item.success === false ? "failed" : item.success === true ? "success" : undefined,
      createdAt: item.createdAt ? String(item.createdAt) : undefined,
      created_at: item.created_at ? String(item.created_at) : undefined,
      duration: finite(item.duration ?? item.duration_s, 0),
      metrics: objectRecord(item.metrics) as ResultSummary["metrics"],
    };
  });
}

export function frameAt(clip: ClipResult | null, time: number): MotionFrame | null {
  if (!clip?.frames.length) return null;
  let best = clip.frames[0] ?? null;
  for (const frame of clip.frames) {
    if (frame.time > time) break;
    best = frame;
  }
  return best;
}

export function formatMetric(value: unknown): string {
  if (typeof value === "number") {
    if (Math.abs(value) > 0 && Math.abs(value) < 0.01) return value.toExponential(2);
    return Number.isInteger(value) ? value.toString() : value.toFixed(3);
  }
  if (typeof value === "boolean") return value ? "Pass" : "Fail";
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return value == null ? "—" : String(value);
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
