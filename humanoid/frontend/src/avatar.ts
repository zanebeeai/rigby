/**
 * Loading the humanoid asset, with the identity of the loaded bytes attached.
 *
 * Evidence capture is only meaningful if the body in the frame is the body that was
 * compiled. `GLTFLoader.loadAsync` cannot say which bytes it parsed, so the fetch and
 * the parse are separated here: the bytes are hashed before they are handed to the
 * parser, and the hash travels with the loaded asset into the evidence manifest.
 *
 * Nothing in this module imports three.js, so it is exercised directly under vitest
 * without a WebGL context.
 */

export type AvatarAssetFailure =
  | "fetch_failed"
  | "empty_asset"
  | "digest_unavailable"
  | "parse_failed";

export class AvatarAssetError extends Error {
  readonly reason: AvatarAssetFailure;

  constructor(message: string, reason: AvatarAssetFailure) {
    super(message);
    this.name = "AvatarAssetError";
    this.reason = reason;
  }
}

export interface AvatarAssetIO<TRoot> {
  fetchBytes(url: string): Promise<ArrayBuffer>;
  digest(bytes: ArrayBuffer): Promise<string>;
  parse(bytes: ArrayBuffer, url: string): Promise<TRoot>;
}

export interface LoadedAvatarAsset<TRoot> {
  root: TRoot;
  url: string;
  /** Null only when hashing was optional and unavailable; capture never accepts null. */
  sha256: string | null;
  bytes: number;
}

export interface LoadAvatarOptions {
  /**
   * Fail the load when the bytes cannot be hashed. Capture requires this: an asset it
   * cannot identify is an asset it cannot attest to. The interactive studio does not,
   * so a non-secure origin without SubtleCrypto still gets its humanoid.
   */
  requireDigest?: boolean;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Fetch, hash and parse the avatar asset, or throw `AvatarAssetError`.
 *
 * Every failure is a throw. A caller that wants a degraded body — the interactive
 * studio does — substitutes one explicitly, so the substitution is never invisible.
 */
export async function loadAvatarAsset<TRoot>(
  url: string,
  io: AvatarAssetIO<TRoot>,
  options: LoadAvatarOptions = {},
): Promise<LoadedAvatarAsset<TRoot>> {
  const requireDigest = options.requireDigest ?? true;
  let raw: ArrayBuffer;
  try {
    raw = await io.fetchBytes(url);
  } catch (error) {
    throw new AvatarAssetError(`humanoid asset ${url} could not be fetched: ${describe(error)}`, "fetch_failed");
  }
  if (raw.byteLength === 0) {
    throw new AvatarAssetError(`humanoid asset ${url} is empty`, "empty_asset");
  }
  let sha256: string | null;
  try {
    sha256 = await io.digest(raw);
  } catch (error) {
    if (requireDigest) {
      throw new AvatarAssetError(
        `humanoid asset ${url} could not be hashed: ${describe(error)}`,
        "digest_unavailable",
      );
    }
    sha256 = null;
  }
  let root: TRoot;
  try {
    root = await io.parse(raw, url);
  } catch (error) {
    throw new AvatarAssetError(`humanoid asset ${url} could not be parsed: ${describe(error)}`, "parse_failed");
  }
  return { root, url, sha256, bytes: raw.byteLength };
}

export async function fetchAssetBytes(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`.trim());
  return await response.arrayBuffer();
}

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  // SubtleCrypto is restricted to secure contexts. http://127.0.0.1 and http://localhost
  // qualify, which covers every capture target; a non-loopback plain-http origin does
  // not, and capture refuses to proceed rather than record an unverified asset.
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error("SubtleCrypto is unavailable; asset hashing needs https or a loopback origin");
  }
  const digest = await subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
