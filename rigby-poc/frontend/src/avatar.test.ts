import { describe, expect, it } from "vitest";
import {
  AvatarAssetError,
  loadAvatarAsset,
  sha256Hex,
  type AvatarAssetIO,
} from "./avatar";

const BYTES = new TextEncoder().encode("glTF-ish bytes").buffer as ArrayBuffer;

function io(overrides: Partial<AvatarAssetIO<string>> = {}): AvatarAssetIO<string> {
  return {
    fetchBytes: async () => BYTES,
    digest: async () => "a".repeat(64),
    parse: async () => "humanoid",
    ...overrides,
  };
}

async function reason(promise: Promise<unknown>): Promise<string> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof AvatarAssetError) return error.reason;
    return `unexpected:${String(error)}`;
  }
  return "resolved";
}

describe("loadAvatarAsset", () => {
  it("returns the parsed root alongside the hash of the bytes it parsed", async () => {
    const asset = await loadAvatarAsset("/assets/models/human-male.glb", io());
    expect(asset.root).toBe("humanoid");
    expect(asset.sha256).toBe("a".repeat(64));
    expect(asset.bytes).toBe(BYTES.byteLength);
    expect(asset.url).toBe("/assets/models/human-male.glb");
  });

  it("throws rather than substituting when the asset cannot be fetched", async () => {
    const failing = io({
      fetchBytes: async () => {
        throw new Error("HTTP 404 Not Found");
      },
    });
    await expect(reason(loadAvatarAsset("/missing.glb", failing))).resolves.toBe("fetch_failed");
    await expect(loadAvatarAsset("/missing.glb", failing)).rejects.toThrow(/HTTP 404/);
  });

  it("rejects an empty asset before it reaches the parser", async () => {
    let parsed = false;
    const empty = io({
      fetchBytes: async () => new ArrayBuffer(0),
      parse: async () => {
        parsed = true;
        return "humanoid";
      },
    });
    await expect(reason(loadAvatarAsset("/empty.glb", empty))).resolves.toBe("empty_asset");
    expect(parsed).toBe(false);
  });

  it("refuses to load an asset it cannot hash", async () => {
    const unhashable = io({
      digest: async () => {
        throw new Error("SubtleCrypto is unavailable");
      },
    });
    await expect(reason(loadAvatarAsset("/insecure.glb", unhashable))).resolves.toBe(
      "digest_unavailable",
    );
  });

  it("loads without a hash when hashing is optional, so the studio still degrades gracefully", async () => {
    const unhashable = io({
      digest: async () => {
        throw new Error("SubtleCrypto is unavailable");
      },
    });
    const asset = await loadAvatarAsset("/insecure.glb", unhashable, { requireDigest: false });
    expect(asset.root).toBe("humanoid");
    expect(asset.sha256).toBeNull();
  });

  it("still fails on a fetch error even when hashing is optional", async () => {
    const failing = io({
      fetchBytes: async () => {
        throw new Error("HTTP 404");
      },
    });
    await expect(
      reason(loadAvatarAsset("/missing.glb", failing, { requireDigest: false })),
    ).resolves.toBe("fetch_failed");
  });

  it("reports a parse failure as an asset failure", async () => {
    const corrupt = io({
      parse: async () => {
        throw new Error("Unsupported glTF version");
      },
    });
    await expect(reason(loadAvatarAsset("/corrupt.glb", corrupt))).resolves.toBe("parse_failed");
  });

  it("hashes the bytes before parsing, so a swapped body is a different hash", async () => {
    const seen: string[] = [];
    const ordered = io({
      digest: async () => {
        seen.push("digest");
        return "b".repeat(64);
      },
      parse: async () => {
        seen.push("parse");
        return "humanoid";
      },
    });
    await loadAvatarAsset("/assets/models/human-male.glb", ordered);
    expect(seen).toEqual(["digest", "parse"]);
  });
});

describe("sha256Hex", () => {
  it("matches the known digest of the empty input", async () => {
    // The canonical SHA-256 of zero bytes; a wrong hex encoding fails this immediately.
    expect(await sha256Hex(new ArrayBuffer(0))).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("pads single-digit bytes, so the digest is always 64 characters", async () => {
    const digest = await sha256Hex(new TextEncoder().encode("rigby").buffer as ArrayBuffer);
    expect(digest).toMatch(/^[0-9a-f]{64}$/);
  });
});
