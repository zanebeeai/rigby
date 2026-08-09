import "./capture.css";
import {
  activeObstacleObjectId,
  activeTaskObjectId,
  shouldShowTaskEnvironment,
  shouldShowTaskSupportSurface,
} from "./environment";
import { frameAt, unwrapClip } from "./motion";
import { RigbyScene } from "./scene";
import {
  DEFAULT_PARAMETERS,
  type BlockParameters,
  type CameraMode,
  type MotionProgram,
} from "./types";

interface CaptureState {
  status: "loading" | "ready" | "error";
  result_id?: string;
  prompt?: string;
  requested_time_s?: number;
  rendered_time_s?: number;
  camera?: Record<string, number | string>;
  error?: string;
}

declare global {
  interface Window {
    __RIGBY_CAPTURE__: CaptureState;
  }
}

const WIDTH = 1600;
const HEIGHT = 900;
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
  const block = scene.objects?.find((item) => item.id === preferredObjectId) ?? scene.objects?.[0];
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

async function initialize(): Promise<void> {
  try {
    const query = new URLSearchParams(window.location.search);
    const resultId = query.get("result")?.trim();
    const view = query.get("view") as CameraMode | null;
    const requestedTime = Number(query.get("time"));
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
    const renderedFrame = frameAt(clip, Math.min(requestedTime, clip.duration));
    if (!renderedFrame) throw new Error("No animation frame was selected");

    const program = payload.program as MotionProgram | undefined;
    const obstacleObjectId = activeObstacleObjectId(program);
    const taskObjectId = activeTaskObjectId(program);
    const parameters = blockParameters(payload.scene, taskObjectId);
    const scene = new RigbyScene(captureHost, parameters);
    scene.setTaskEnvironmentVisible(
      shouldShowTaskEnvironment(program),
      shouldShowTaskSupportSurface(program),
    );
    scene.setObjectId(taskObjectId);
    scene.setCamera(view);
    if (view === "ego") scene.setEgoFieldOfView(94);
    else if (program?.intent === "full_body") {
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
    await scene.prepareCapture(WIDTH, HEIGHT, view, renderedFrame);

    const state: CaptureState = {
      status: "ready",
      result_id: resultId,
      prompt: clip.prompt ?? clip.program?.source_text,
      requested_time_s: requestedTime,
      rendered_time_s: renderedFrame.time,
      camera: scene.captureContract(),
    };
    window.__RIGBY_CAPTURE__ = state;
    document.body.dataset.captureStatus = "ready";
    document.body.dataset.captureView = view;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Capture failed";
    window.__RIGBY_CAPTURE__ = { status: "error", error: message };
    document.body.dataset.captureStatus = "error";
    document.body.dataset.captureError = message;
  }
}

void initialize();
