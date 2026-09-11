import "./capture.css";
import {
  activeObstacleObjectId,
  activeTaskObjectId,
  shouldShowTaskEnvironment,
  shouldShowTaskSupportSurface,
  supportSurfaceFor,
} from "./environment";
import {
  captureHeightPx,
  captureWidthPx,
  egoVerticalFovDeg,
} from "./generated/camera";
import { frameAt, unwrapClip } from "./motion";
import { RigbyScene, type RenderProvenance } from "./scene";
import {
  DEFAULT_PARAMETERS,
  type SceneManifest,
  type BlockParameters,
  type CameraMode,
  type ClipResult,
  type MotionProgram,
} from "./types";

interface CaptureState {
  status: "loading" | "ready" | "error";
  result_id?: string;
  prompt?: string;
  requested_time_s?: number;
  rendered_time_s?: number;
  camera?: Record<string, number | string>;
  render_provenance?: RenderProvenance & { user_agent: string };
  error?: string;
}

declare global {
  interface Window {
    __RIGBY_CAPTURE__: CaptureState;
    /**
     * Re-aim and re-pose an already-loaded capture page.
     *
     * Capture used to reload the whole page — and re-fetch the 521 KB humanoid GLB —
     * once per sample point per view. Seeking on a live page keeps a single asset load
     * per candidate, which is both far faster and the reason an asset failure is one
     * loud error instead of a per-snapshot coin flip.
     */
    __RIGBY_SEEK__?: (timeS: number, view: CameraMode) => Promise<CaptureState>;
  }
}

const WIDTH = captureWidthPx;
const HEIGHT = captureHeightPx;
const EGO_FOV_DEG = egoVerticalFovDeg;
const mount = document.querySelector<HTMLElement>("#capture-app");
if (!mount) throw new Error("Capture mount not found");
const captureHost = mount;

window.__RIGBY_CAPTURE__ = { status: "loading" };

function blockParameters(sceneValue: unknown, preferredObjectId?: string | null): BlockParameters {
  const scene = sceneValue as {
    objects?: Array<{
      id?: string;
      kind?: string;
      transform?: { translation?: { x?: number; y?: number; z?: number } };
      dimensions_m?: { x?: number; y?: number; z?: number };
      mass_kg?: number;
      friction?: number;
    }>;
  };
  const block = scene.objects?.find((item) => item.id === preferredObjectId)
    ?? scene.objects?.find((item) => item.kind !== "table");
  const translation = block?.transform?.translation;
  const dimensions = block?.dimensions_m;
  return {
    kind: block?.kind ?? "block",
    width: dimensions?.x ?? DEFAULT_PARAMETERS.block.width,
    height: dimensions?.y ?? DEFAULT_PARAMETERS.block.height,
    depth: dimensions?.z ?? DEFAULT_PARAMETERS.block.depth,
    mass: block?.mass_kg ?? DEFAULT_PARAMETERS.block.mass,
    friction: block?.friction ?? DEFAULT_PARAMETERS.block.friction,
    position: [
      translation?.x ?? DEFAULT_PARAMETERS.block.position[0],
      translation?.y ?? DEFAULT_PARAMETERS.block.position[1],
      translation?.z ?? DEFAULT_PARAMETERS.block.position[2],
    ],
  };
}

/**
 * Aim the camera for one view.
 *
 * This runs on every seek, not once per page load, and that is deliberate: the orbit
 * framing call also clears `RigbyScene`'s accumulated root-tracking offset, so a seek
 * starts from the same camera state a fresh page load would.
 */
function applyView(
  scene: RigbyScene,
  view: CameraMode,
  program: MotionProgram | undefined,
  taskObjectId: string | null | undefined,
  obstacleObjectId: string | null | undefined,
): void {
  scene.setCamera(view);
  if (view === "ego") {
    scene.setEgoFieldOfView(EGO_FOV_DEG);
    return;
  }
  if (program?.intent === "full_body") {
    // Preserve head-to-toe evidence with enough ground margin to judge
    // support, clearance, and penetration throughout root-tracked motion.
    // Bias the diagnostic view toward the side of the avatar.  A mostly
    // frontal view foreshortens stride length and can make a valid run look
    // like a two-footed hop even when the feet are far apart in depth.
    scene.setOrbitFraming(
      taskObjectId === "ladder"
        ? [2.7, 1.55, -1.8]
        : obstacleObjectId
          ? [3.0, 1.45, -2.0]
          : [3.0, 1.45, 2.0],
      taskObjectId === "ladder"
        ? [0, 1.20, 0.42]
        : obstacleObjectId
          ? [0, 1.0, 0.35]
          : [0, 1.0, 0.2],
    );
  } else {
    scene.setOrbitFraming([1.6, 1.5, 1.9], [0, 1.25, 0.25]);
  }
}

function failed(message: string): CaptureState {
  const state: CaptureState = { status: "error", error: message };
  window.__RIGBY_CAPTURE__ = state;
  document.body.dataset.captureStatus = "error";
  document.body.dataset.captureError = message;
  return state;
}

async function initialize(): Promise<void> {
  try {
    const query = new URLSearchParams(window.location.search);
    const resultId = query.get("result")?.trim();
    const view = query.get("view") as CameraMode | null;
    const requestedTime = Number(query.get("time"));
    const deterministicRender = query.get("render") === "deterministic";
    if (!resultId) throw new Error("A result id is required");
    if (view !== "ego" && view !== "orbit") throw new Error("View must be ego or orbit");
    if (!Number.isFinite(requestedTime) || requestedTime < 0) throw new Error("Time must be non-negative");

    const response = await fetch(`/api/v1/results/${encodeURIComponent(resultId)}`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`Result could not be loaded (${response.status})`);
    const payload = await response.json() as Record<string, unknown>;
    const clip = unwrapClip(payload);
    if (!clip.frames.length) throw new Error("Result has no animation frames");

    const program = payload.program as MotionProgram | undefined;
    const obstacleObjectId = activeObstacleObjectId(program);
    const taskObjectId = activeTaskObjectId(program);
    const parameters = blockParameters(payload.scene, taskObjectId);
    const scene = new RigbyScene(captureHost, parameters, {
      strictAssets: true,
      deterministicRender,
    });
    scene.updateSupportSurface(
      supportSurfaceFor(program, payload.scene as Pick<SceneManifest, "objects"> | undefined),
    );
    scene.setTaskEnvironmentVisible(
      shouldShowTaskEnvironment(program),
      shouldShowTaskSupportSurface(program),
    );
    scene.setObjectId(taskObjectId);

    const seek = async (timeS: number, seekView: CameraMode): Promise<CaptureState> => {
      if (!Number.isFinite(timeS) || timeS < 0) throw new Error("Time must be non-negative");
      if (seekView !== "ego" && seekView !== "orbit") throw new Error("View must be ego or orbit");
      const renderedFrame = frameAt(clip as ClipResult, Math.min(timeS, clip.duration));
      if (!renderedFrame) throw new Error("No animation frame was selected");
      document.body.dataset.captureStatus = "seeking";
      applyView(scene, seekView, program, taskObjectId, obstacleObjectId);
      await scene.prepareCapture(WIDTH, HEIGHT, seekView, renderedFrame);
      const state: CaptureState = {
        status: "ready",
        result_id: resultId,
        prompt: clip.prompt ?? clip.program?.source_text,
        requested_time_s: timeS,
        rendered_time_s: renderedFrame.time,
        camera: scene.captureContract(),
        render_provenance: {
          ...scene.renderProvenance(),
          user_agent: window.navigator.userAgent,
        },
      };
      window.__RIGBY_CAPTURE__ = state;
      document.body.dataset.captureStatus = "ready";
      document.body.dataset.captureView = seekView;
      return state;
    };

    window.__RIGBY_SEEK__ = async (timeS, seekView) => {
      try {
        return await seek(timeS, seekView);
      } catch (error) {
        return failed(error instanceof Error ? error.message : "Seek failed");
      }
    };

    await seek(requestedTime, view);
  } catch (error) {
    failed(error instanceof Error ? error.message : "Capture failed");
  }
}

void initialize();
