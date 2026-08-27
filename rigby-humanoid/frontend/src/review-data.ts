export interface ReviewManifestRecord {
  clip_id: string;
  prompt?: string;
  motion_sha256?: string;
  motion_profile?: Record<string, string>;
  motion_descriptor?: Record<string, number>;
  result_id?: string;
  result_url?: string;
  clip_url?: string;
  animation_url?: string;
  available?: boolean;
  clip?: unknown;
}

export interface ReviewManifest {
  schema_version: string;
  acceptance_run_id?: string;
  manifest_sha256?: string;
  content_group?: string;
  records: ReviewManifestRecord[];
}

export interface ReviewScore {
  egocentric: number | null;
  orbit: number | null;
  note: string;
}

export interface ReviewExport {
  schema_version: "1.1";
  acceptance_run_id: string;
  manifest_sha256: string;
  instructions: string;
  records: Array<{
    clip_id: string;
    egocentric_ratings: number[];
    orbit_ratings: number[];
    reviewer_notes: string[];
  }>;
}

export interface ReviewRunDraft {
  manifest: ReviewManifest;
  scores: Record<string, ReviewScore>;
}

function objectRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" ? value as Record<string, unknown> : {};
}

export function normalizeReviewManifest(value: unknown): ReviewManifest {
  const root = objectRecord(value);
  const rawRecords = Array.isArray(root.records) ? root.records : Array.isArray(root.clips) ? root.clips : [];
  const records = rawRecords.map((raw) => {
    const item = objectRecord(raw);
    return {
      clip_id: String(item.clip_id ?? item.id ?? item.opaque_id ?? ""),
      prompt: item.prompt ? String(item.prompt) : undefined,
      motion_sha256: item.motion_sha256 ? String(item.motion_sha256) : undefined,
      motion_profile: Object.keys(objectRecord(item.motion_profile)).length
        ? objectRecord(item.motion_profile) as Record<string, string>
        : undefined,
      motion_descriptor: Object.keys(objectRecord(item.motion_descriptor)).length
        ? objectRecord(item.motion_descriptor) as Record<string, number>
        : undefined,
      result_id: item.result_id ? String(item.result_id) : item.result_ref ? String(item.result_ref) : undefined,
      result_url: item.result_url ? String(item.result_url) : undefined,
      clip_url: item.clip_url ? String(item.clip_url) : undefined,
      animation_url: item.animation_url ? String(item.animation_url) : undefined,
      available: item.available === undefined ? true : item.available === true,
      clip: item.clip,
    };
  });
  const expected = Array.from({ length: 20 }, (_, index) => `s${String(index + 1).padStart(2, "0")}`);
  if (records.length !== expected.length || records.some((record, index) => record.clip_id !== expected[index])) {
    throw new Error("The review manifest must contain the ordered opaque IDs s01 through s20.");
  }
  if (records.some((record) => !record.result_id && !record.result_url && !record.clip_url && !record.clip)) {
    throw new Error("Every review clip must include a result reference.");
  }
  if (records.some((record) => record.available === false)) {
    throw new Error("The acceptance run contains unavailable gesture clips.");
  }
  const contentGroup = typeof root.content_group === "string" ? root.content_group : undefined;
  if (contentGroup === "diverse_hang_ten") {
    const hashes = records.map((record) => record.motion_sha256).filter(Boolean);
    if (hashes.length !== records.length || new Set(hashes).size !== records.length) {
      throw new Error("The diverse review manifest must contain 20 unique compiled motions.");
    }
  }
  const schemaVersion = String(root.schema_version ?? "1.0");
  const acceptanceRunId = typeof root.acceptance_run_id === "string" ? root.acceptance_run_id : undefined;
  const manifestSha256 = typeof root.manifest_sha256 === "string" ? root.manifest_sha256 : undefined;
  if (schemaVersion === "1.1" && (!acceptanceRunId || !manifestSha256)) {
    throw new Error("The bound review manifest is missing its acceptance run or digest.");
  }
  return { schema_version: schemaVersion, acceptance_run_id: acceptanceRunId, manifest_sha256: manifestSha256, content_group: contentGroup, records };
}

export function emptyScores(manifest: ReviewManifest): Record<string, ReviewScore> {
  return Object.fromEntries(manifest.records.map((record) => [record.clip_id, { egocentric: null, orbit: null, note: "" }]));
}

export function completedCount(scores: Record<string, ReviewScore>): number {
  return Object.values(scores).filter((score) => score.egocentric !== null && score.orbit !== null).length;
}

export function queueCompletedCount(runs: ReviewRunDraft[]): number {
  return runs.reduce((total, run) => total + completedCount(run.scores), 0);
}

export function queueExports(runs: ReviewRunDraft[]): ReviewExport[] {
  return runs.map((run) => reviewExport(run.manifest, run.scores));
}

export function validateReviewQueue(manifests: ReviewManifest[]): void {
  const diverse = manifests.filter((manifest) => manifest.content_group === "diverse_hang_ten");
  if (diverse.length < 2) return;
  const hashes = diverse.flatMap((manifest) => manifest.records.map((record) => record.motion_sha256).filter(Boolean));
  const expected = diverse.reduce((total, manifest) => total + manifest.records.length, 0);
  if (hashes.length !== expected || new Set(hashes).size !== expected) {
    throw new Error("The combined review queue must contain a different compiled animation for every LLM call.");
  }
}

export function reviewExport(manifest: ReviewManifest, scores: Record<string, ReviewScore>): ReviewExport {
  return {
    schema_version: "1.1",
    acceptance_run_id: manifest.acceptance_run_id ?? "",
    manifest_sha256: manifest.manifest_sha256 ?? "",
    instructions: "The source prompt was shown with each motion. Each anonymous 1-5 rating evaluates prompt fidelity and naturalness in the named view.",
    records: manifest.records.map((record) => {
      const score = scores[record.clip_id];
      return {
        clip_id: record.clip_id,
        egocentric_ratings: score?.egocentric === null || score?.egocentric === undefined ? [] : [score.egocentric],
        orbit_ratings: score?.orbit === null || score?.orbit === undefined ? [] : [score.orbit],
        reviewer_notes: score?.note.trim() ? [score.note.trim()] : [],
      };
    }),
  };
}

export function resultReference(record: ReviewManifestRecord): { kind: "inline"; value: unknown } | { kind: "url"; value: string } {
  if (record.clip) return { kind: "inline", value: record.clip };
  if (record.clip_url) return { kind: "url", value: record.clip_url };
  if (record.result_url) return { kind: "url", value: record.result_url };
  return { kind: "url", value: `/api/v1/results/${encodeURIComponent(record.result_id ?? "")}` };
}
