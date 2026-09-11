import { describe, expect, it } from "vitest";
import { completedCount, emptyScores, normalizeReviewManifest, queueCompletedCount, queueExports, reviewExport, validateReviewQueue } from "./review-data";

function manifest() {
  return {
    schema_version: "1.1",
    acceptance_run_id: "000005",
    manifest_sha256: "abc123",
    records: Array.from({ length: 20 }, (_, index) => ({
      clip_id: `s${String(index + 1).padStart(2, "0")}`,
      prompt: `Show prompt ${index + 1}`,
      result_id: `opaque-result-${index}`,
    })),
  };
}

describe("prompt-aware review data", () => {
  it("requires the exact opaque acceptance IDs", () => {
    expect(normalizeReviewManifest(manifest()).records).toHaveLength(20);
    expect(normalizeReviewManifest(manifest()).records[0]?.prompt).toBe("Show prompt 1");
    expect(() => normalizeReviewManifest({ records: manifest().records.slice(1) })).toThrow(/s01 through s20/);
    expect(() => normalizeReviewManifest({ ...manifest(), manifest_sha256: undefined })).toThrow(/missing its acceptance run or digest/);
  });

  it("rejects duplicate motions in a diverse review manifest", () => {
    const records = manifest().records.map((record, index) => ({ ...record, motion_sha256: `motion-${index}` }));
    expect(normalizeReviewManifest({ ...manifest(), content_group: "diverse_hang_ten", records }).records).toHaveLength(20);
    records[19]!.motion_sha256 = records[0]!.motion_sha256;
    expect(() => normalizeReviewManifest({ ...manifest(), content_group: "diverse_hang_ten", records })).toThrow(/20 unique compiled motions/);
  });

  it("rejects repeated outputs across two LLM-call review runs", () => {
    const firstRecords = manifest().records.map((record, index) => ({ ...record, motion_sha256: `first-${index}` }));
    const secondRecords = manifest().records.map((record, index) => ({ ...record, motion_sha256: `second-${index}` }));
    const first = normalizeReviewManifest({ ...manifest(), content_group: "diverse_hang_ten", records: firstRecords });
    const second = normalizeReviewManifest({ ...manifest(), acceptance_run_id: "000006", content_group: "diverse_hang_ten", records: secondRecords });
    expect(() => validateReviewQueue([first, second])).not.toThrow();
    second.records[19]!.motion_sha256 = first.records[0]!.motion_sha256;
    expect(() => validateReviewQueue([first, second])).toThrow(/different compiled animation for every LLM call/);
  });

  it("exports the acceptance review schema without result references", () => {
    const normalized = normalizeReviewManifest(manifest());
    const scores = emptyScores(normalized);
    scores.s01 = { egocentric: 4, orbit: 5, note: "Readable motion" };
    const output = reviewExport(normalized, scores);
    expect(output.acceptance_run_id).toBe("000005");
    expect(output.manifest_sha256).toBe("abc123");
    expect(output.records[0]).toEqual({
      clip_id: "s01",
      egocentric_ratings: [4],
      orbit_ratings: [5],
      reviewer_notes: ["Readable motion"],
    });
    expect(JSON.stringify(output)).not.toContain("opaque-result");
    expect(completedCount(scores)).toBe(1);
  });

  it("keeps repeated opaque IDs independent across a two-run queue", () => {
    const first = normalizeReviewManifest(manifest());
    const second = normalizeReviewManifest(manifest());
    const firstScores = emptyScores(first);
    const secondScores = emptyScores(second);
    firstScores.s01 = { egocentric: 4, orbit: 4, note: "First run" };
    secondScores.s01 = { egocentric: 5, orbit: 5, note: "Second run" };
    const runs = [{ manifest: first, scores: firstScores }, { manifest: second, scores: secondScores }];
    expect(queueCompletedCount(runs)).toBe(2);
    expect(queueExports(runs)[0]?.records[0]?.reviewer_notes).toEqual(["First run"]);
    expect(queueExports(runs)[1]?.records[0]?.reviewer_notes).toEqual(["Second run"]);
  });
});
