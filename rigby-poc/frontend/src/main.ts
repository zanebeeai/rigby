import "./styles.css";
import { api, errorMessage } from "./api";
import {
  activeTaskObjectId,
  shouldShowTaskEnvironment,
  shouldShowTaskSupportSurface,
} from "./environment";
import { DEMO_CLIP, DEMO_PROGRAM, DEMO_SUMMARY } from "./mock";
import { formatMetric, frameAt } from "./motion";
import {
  candidateProgressBadge,
  isTerminalWithoutWinner,
  pipelineStageLabel,
} from "./pipeline-state";
import { RigbyScene } from "./scene";
import {
  DEFAULT_PARAMETERS,
  type BlockParameters,
  type CameraMode,
  type ClipResult,
  type ContactEvent,
  type MotionParameters,
  type MotionPrimitive,
  type MotionProgram,
  type ParameterOverrides,
  type PipelineRun,
  type ResultSummary,
} from "./types";

type NumericKey = Exclude<keyof MotionParameters, "handedness" | "block">;
type BlockNumericKey = Exclude<keyof BlockParameters, "position" | "kind">;

interface SliderSpec {
  label: string;
  key: NumericKey | BlockNumericKey | "blockX" | "blockY" | "blockZ";
  group: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  block?: boolean;
}

const sliderSpecs: SliderSpec[] = [
  { group: "Timing", label: "Duration", key: "duration", min: 0.8, max: 4, step: 0.1, unit: "s" },
  { group: "Timing", label: "Hold", key: "holdDuration", min: 0.25, max: 3, step: 0.05, unit: "s" },
  { group: "Trajectory", label: "Presentation arc", key: "pathArc", min: -1, max: 1, step: 0.05 },
  { group: "Trajectory", label: "Wrist flourish", key: "wristFlourish", min: -1, max: 1, step: 0.05 },
  { group: "Trajectory", label: "Forearm twist amplitude", key: "wristShakeAmplitude", min: 0, max: 1, step: 0.01 },
  { group: "Trajectory", label: "Forearm twist cycles", key: "wristShakeCycles", min: 0, max: 6, step: 0.25 },
  { group: "Trajectory", label: "Path amplitude", key: "trajectoryAmplitude", min: 0, max: 0.2, step: 0.005, unit: "m" },
  { group: "Trajectory", label: "Path repetitions", key: "trajectoryCycles", min: 0, max: 8, step: 0.25 },
  { group: "Trajectory", label: "Axial forearm roll", key: "axialRotationAmplitude", min: 0, max: 1, step: 0.01 },
  { group: "Whole body", label: "Travel distance", key: "bodyDistance", min: 0, max: 3, step: 0.05, unit: "m" },
  { group: "Whole body", label: "Turn", key: "bodyTurn", min: -180, max: 180, step: 5, unit: "°" },
  { group: "Whole body", label: "Vertical height", key: "bodyHeight", min: 0, max: 0.65, step: 0.01, unit: "m" },
  { group: "Whole body", label: "Steps / repetitions", key: "bodyCycles", min: 0, max: 8, step: 0.25 },
  { group: "Whole body", label: "Intensity", key: "bodyIntensity", min: 0, max: 1, step: 0.01 },
  { group: "Object flight", label: "Travel distance", key: "objectDistance", min: 0.2, max: 2.5, step: 0.05, unit: "m" },
  { group: "Object flight", label: "Arc height", key: "objectApexHeight", min: 0.05, max: 1.2, step: 0.05, unit: "m" },
  { group: "Object flight", label: "Spin", key: "objectSpinTurns", min: -3, max: 3, step: 0.1, unit: "turns" },
  { group: "Object flight", label: "Catch height", key: "objectContactHeight", min: 0.85, max: 1.65, step: 0.01, unit: "m" },
  { group: "Object flight", label: "Catch depth", key: "objectContactDepth", min: 0.18, max: 0.58, step: 0.01, unit: "m" },
  { group: "Object flight", label: "Landing height", key: "objectLandingHeight", min: 0.02, max: 1.3, step: 0.01, unit: "m" },
  { group: "Arm pose", label: "Lateral offset", key: "armLateral", min: -0.3, max: 0.3, step: 0.005, unit: "m" },
  { group: "Arm pose", label: "Height offset", key: "armHeight", min: -0.3, max: 0.3, step: 0.005, unit: "m" },
  { group: "Arm pose", label: "Depth offset", key: "armDepth", min: -0.3, max: 0.3, step: 0.005, unit: "m" },
  { group: "Arm pose", label: "Elbow swivel", key: "elbowSwivel", min: -90, max: 90, step: 1, unit: "°" },
  { group: "Arm pose", label: "Torso participation", key: "torsoParticipation", min: 0, max: 1, step: 0.01 },
  { group: "Wrist", label: "Pitch", key: "wristPitch", min: -180, max: 180, step: 1, unit: "°" },
  { group: "Wrist", label: "Yaw", key: "wristYaw", min: -180, max: 180, step: 1, unit: "°" },
  { group: "Wrist", label: "Roll", key: "wristRoll", min: -180, max: 180, step: 1, unit: "°" },
  { group: "Hand", label: "Thumb curl", key: "thumbCurl", min: 0, max: 1, step: 0.01 },
  { group: "Hand", label: "Thumb opposition", key: "thumbOpposition", min: 0, max: 1, step: 0.01 },
  { group: "Hand", label: "Index curl", key: "indexCurl", min: 0, max: 1, step: 0.01 },
  { group: "Hand", label: "Middle curl", key: "middleCurl", min: 0, max: 1, step: 0.01 },
  { group: "Hand", label: "Ring curl", key: "ringCurl", min: 0, max: 1, step: 0.01 },
  { group: "Hand", label: "Little curl", key: "littleCurl", min: 0, max: 1, step: 0.01 },
  { group: "Hand", label: "Finger spread", key: "fingerSpread", min: 0, max: 1, step: 0.01 },
  { group: "Physical grasp", label: "Grip force", key: "gripForce", min: 0, max: 1, step: 0.01 },
  { group: "Physical grasp", label: "Lift height", key: "liftHeight", min: 0.03, max: 0.25, step: 0.005, unit: "m" },
  { group: "Block", label: "Width", key: "width", min: 0.04, max: 0.09, step: 0.005, unit: "m", block: true },
  { group: "Block", label: "Height", key: "height", min: 0.04, max: 0.14, step: 0.005, unit: "m", block: true },
  { group: "Block", label: "Depth", key: "depth", min: 0.04, max: 0.14, step: 0.005, unit: "m", block: true },
  { group: "Block", label: "Mass", key: "mass", min: 0.1, max: 0.5, step: 0.01, unit: "kg", block: true },
  { group: "Block", label: "Friction", key: "friction", min: 0.4, max: 1.5, step: 0.05, block: true },
  { group: "Block placement", label: "Left / right", key: "blockX", min: -0.6, max: 0.6, step: 0.01, unit: "m", block: true },
  { group: "Block placement", label: "Height", key: "blockY", min: 0.84, max: 1.25, step: 0.01, unit: "m", block: true },
  { group: "Block placement", label: "Forward distance (+Z)", key: "blockZ", min: 0.15, max: 0.29, step: 0.01, unit: "m", block: true },
];

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("App mount not found");

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></div>
        <div><strong>Rigby</strong><span>Motion Studio</span></div>
      </div>
      <div class="topbar-actions">
        <div class="connection" id="connection" role="status"><i></i><span>Connecting</span></div>
        <button class="button button-secondary" id="results-toggle" aria-controls="results-drawer" aria-expanded="false">
          <span aria-hidden="true">◫</span> Results
        </button>
        <button class="button button-primary" id="export" disabled>Export GLB</button>
      </div>
    </header>

    <main class="workspace" id="workspace">
      <aside class="prompt-panel" aria-label="Motion prompt">
        <div class="panel-heading">
          <p class="eyebrow">01 · Direct</p>
          <h1>Describe the motion</h1>
          <p>Use plain language. Rigby proposes five motions, checks them, and automatically selects the strongest result.</p>
        </div>
        <label class="prompt-label" for="prompt">Instruction</label>
        <div class="prompt-wrap">
          <textarea id="prompt" rows="5" maxlength="400" placeholder="Grab the block in front of you">Grab the block in front of you</textarea>
          <span id="prompt-count">34 / 400</span>
        </div>
        <button class="button generate-button" id="generate">
          <span class="button-label">Generate 5 & choose</span><kbd>⌘ ↵</kbd>
        </button>
        <div class="prompt-examples" aria-label="Prompt examples">
          <span>Try</span>
          <button data-prompt="Throw up a hang-ten sign with your right hand">Hang ten</button>
          <button data-prompt="Grab the block in front of you with your left hand">Left grab</button>
          <button data-prompt="Throw a left hook">Left hook</button>
        </div>

        <section class="run-status" aria-live="polite">
          <div class="run-status-head"><span id="run-icon">○</span><strong id="run-title">Ready to generate</strong></div>
          <p id="run-detail">The active block is within the configured workspace.</p>
          <div class="progress-track" id="progress-track"><span></span></div>
        </section>

        <section class="pipeline-trace" id="pipeline-trace" aria-live="polite">
          <div class="pipeline-trace-head">
            <div><p class="eyebrow">Autonomous run</p><strong id="pipeline-title">Waiting for a prompt</strong></div>
            <span id="pipeline-count">0 events</span>
          </div>
          <div class="pipeline-stages" id="pipeline-stages">
            <div data-stage="planning"><i>1</i><span>Understand</span></div>
            <div data-stage="candidates"><i>2</i><span>Propose 5</span></div>
            <div data-stage="visual_evidence"><i>3</i><span>Inspect</span></div>
            <div data-stage="vlm_judge"><i>4</i><span>Judge</span></div>
            <div data-stage="finalize"><i>5</i><span>Finalize</span></div>
          </div>
          <div class="candidate-progress" id="candidate-progress">
            <p class="empty-state">Candidate details will arrive here.</p>
          </div>
          <ol class="pipeline-events" id="pipeline-events"></ol>
        </section>

        <details class="data-disclosure" open>
          <summary>Motion program <span id="program-intent">No plan</span></summary>
          <pre id="program-json">Generate a motion to inspect its semantic program.</pre>
        </details>
      </aside>

      <section class="stage" aria-label="Motion preview">
        <div class="stage-toolbar">
          <div class="segmented" role="group" aria-label="Camera view">
            <button data-camera="orbit" class="active" aria-pressed="true">Orbit</button>
            <button data-camera="ego" aria-pressed="false">First person</button>
          </div>
          <label class="switch-label">
            <input type="checkbox" id="debug-toggle" />
            <span class="switch"><i></i></span>
            Contacts & sockets
          </label>
        </div>
        <div class="viewport" id="viewport" tabindex="0" aria-label="3D motion viewport. Drag to orbit, scroll to zoom.">
          <div class="viewport-label"><span>LIVE</span><span id="frame-readout">Frame 000 · 0.00s</span></div>
          <div class="viewport-help">Drag to orbit · Scroll to zoom · Space to play</div>
        </div>
        <div class="timeline-panel">
          <button class="play-button" id="play" aria-label="Play animation">▶</button>
          <span class="timecode" id="current-time">0:00.00</span>
          <div class="timeline-wrap">
            <input id="timeline" type="range" min="0" max="2.8" step="0.001" value="0" aria-label="Animation timeline" />
            <div class="timeline-phases" id="timeline-phases"></div>
          </div>
          <span class="timecode muted" id="total-time">0:02.80</span>
          <label class="loop-label"><input type="checkbox" id="loop" checked /> Loop</label>
        </div>
      </section>

      <aside class="inspector" aria-label="Motion controls and verification">
        <div class="inspector-head">
          <div><p class="eyebrow">02 · Refine</p><h2>Parameters</h2></div>
          <button class="icon-button" id="reset-params" title="Reset parameters" aria-label="Reset all parameters">↺</button>
        </div>
        <label class="select-control" for="handedness"><span>Handedness</span>
          <select id="handedness"><option value="right">Right</option><option value="left">Left</option></select>
        </label>
        <div id="parameter-controls"></div>

        <section class="inspection-section" id="verification-panel">
          <div class="section-title"><h3>Verification</h3><span class="status-chip neutral" id="verification-chip">Not run</span></div>
          <div id="metrics" class="metric-grid"><p class="empty-state">Metrics appear after compilation.</p></div>
        </section>
        <section class="inspection-section failure-panel" id="failure-panel" hidden>
          <div class="section-title"><h3>Failure</h3><span class="status-chip failed">Blocked</span></div>
          <strong id="failure-code"></strong><p id="failure-message"></p>
        </section>
        <details class="inspection-section provenance-panel">
          <summary><h3>Provenance</h3><span>⌄</span></summary>
          <dl id="provenance"><div><dt>Status</dt><dd>Not generated</dd></div></dl>
        </details>
      </aside>
    </main>

    <aside class="results-drawer" id="results-drawer" aria-label="Saved results" aria-hidden="true">
      <div class="drawer-head"><div><p class="eyebrow">Replay library</p><h2>Sequential results</h2></div><button class="icon-button" id="results-close" aria-label="Close results">×</button></div>
      <p class="drawer-copy">Open a compiled run without asking the planner again.</p>
      <div id="results-list" class="results-list"></div>
    </aside>
    <div class="drawer-scrim" id="drawer-scrim"></div>
    <div class="toast" id="toast" role="status" aria-live="polite"></div>
  </div>
`;

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`Missing UI element: ${selector}`);
  return found;
}

const ui = {
  prompt: element<HTMLTextAreaElement>("#prompt"),
  promptCount: element<HTMLElement>("#prompt-count"),
  generate: element<HTMLButtonElement>("#generate"),
  export: element<HTMLButtonElement>("#export"),
  handedness: element<HTMLSelectElement>("#handedness"),
  controls: element<HTMLElement>("#parameter-controls"),
  runIcon: element<HTMLElement>("#run-icon"),
  runTitle: element<HTMLElement>("#run-title"),
  runDetail: element<HTMLElement>("#run-detail"),
  progress: element<HTMLElement>("#progress-track"),
  programIntent: element<HTMLElement>("#program-intent"),
  programJson: element<HTMLElement>("#program-json"),
  play: element<HTMLButtonElement>("#play"),
  timeline: element<HTMLInputElement>("#timeline"),
  timelinePhases: element<HTMLElement>("#timeline-phases"),
  currentTime: element<HTMLElement>("#current-time"),
  totalTime: element<HTMLElement>("#total-time"),
  frameReadout: element<HTMLElement>("#frame-readout"),
  loop: element<HTMLInputElement>("#loop"),
  metrics: element<HTMLElement>("#metrics"),
  verificationChip: element<HTMLElement>("#verification-chip"),
  failurePanel: element<HTMLElement>("#failure-panel"),
  failureCode: element<HTMLElement>("#failure-code"),
  failureMessage: element<HTMLElement>("#failure-message"),
  provenance: element<HTMLElement>("#provenance"),
  connection: element<HTMLElement>("#connection"),
  resultsDrawer: element<HTMLElement>("#results-drawer"),
  resultsToggle: element<HTMLButtonElement>("#results-toggle"),
  resultsClose: element<HTMLButtonElement>("#results-close"),
  drawerScrim: element<HTMLElement>("#drawer-scrim"),
  resultsList: element<HTMLElement>("#results-list"),
  toast: element<HTMLElement>("#toast"),
  pipelineTrace: element<HTMLElement>("#pipeline-trace"),
  pipelineTitle: element<HTMLElement>("#pipeline-title"),
  pipelineCount: element<HTMLElement>("#pipeline-count"),
  pipelineStages: element<HTMLElement>("#pipeline-stages"),
  candidateProgress: element<HTMLElement>("#candidate-progress"),
  pipelineEvents: element<HTMLOListElement>("#pipeline-events"),
};

const scene = new RigbyScene(element<HTMLElement>("#viewport"), DEFAULT_PARAMETERS.block);
let parameters = structuredClone(DEFAULT_PARAMETERS);
let program: MotionProgram | null = null;
let clip: ClipResult | null = null;
let currentTime = 0;
let playing = false;
let playStartedAt = 0;
let playStartedFrom = 0;
let compileTimer = 0;
let compileRevision = 0;
let toastTimer = 0;
const dirtyOverrides = new Set<keyof ParameterOverrides>();

function sceneManifest() {
  const [x, y, z] = parameters.block.position;
  return {
    schema_version: "1.0" as const,
    rig: {
      id: "mesh2motion-human-vrm1",
      asset_uri: "assets/models/human-male.glb",
      profile_uri: "config/rig_profiles/mesh2motion-human-vrm1.json",
      fixed_root: true,
    },
    objects: [
      {
        id: "block",
        kind: "block" as const,
        transform: {
          translation: { x, y, z },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
        dimensions_m: { x: parameters.block.width, y: parameters.block.height, z: parameters.block.depth },
        mass_kg: parameters.block.mass,
        friction: parameters.block.friction,
        sockets: [
          {
            id: "front_center",
            transform: {
              translation: { x: 0, y: 0, z: -parameters.block.depth / 2 },
              rotation: { x: 0, y: 0, z: 0, w: 1 },
            },
            approach_normal: { x: 0, y: 0, z: -1 },
            grasp_span_m: Math.min(parameters.block.width, 0.12),
          },
        ],
      },
      {
        id: "ladder",
        kind: "ladder" as const,
        transform: {
          translation: { x: 0, y: 1.05, z: 0.58 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
        dimensions_m: { x: 0.46, y: 2.10, z: 0.08 },
        mass_kg: 18,
        friction: 0.9,
        sockets: Array.from({ length: 9 }, (_, index) => ({
          id: `rung_${index + 1}`,
          transform: {
            translation: { x: 0, y: -0.82 + index * 0.205, z: -0.05 },
            rotation: { x: 0, y: 0, z: 0, w: 1 },
          },
          approach_normal: { x: 0, y: 0, z: -1 },
          grasp_span_m: 0.04,
          role: "climb_contact" as const,
          supports_body_weight: true,
        })),
      },
      {
        id: "hurdle",
        kind: "hurdle" as const,
        transform: {
          translation: { x: 0, y: 0.07, z: 0.55 },
          rotation: { x: 0, y: 0, z: 0, w: 1 },
        },
        dimensions_m: { x: 0.22, y: 0.10, z: 0.10 },
        mass_kg: 4,
        friction: 1.2,
        sockets: [
          {
            id: "top_center",
            transform: {
              translation: { x: 0, y: 0.05, z: 0 },
              rotation: { x: 0, y: 0, z: 0, w: 1 },
            },
            approach_normal: { x: 0, y: 1, z: 0 },
            grasp_span_m: 0.10,
            role: "support" as const,
            supports_body_weight: false,
          },
        ],
      },
    ],
    fps: 30,
    reachable_radius_m: 0.72,
  };
}

function parameterOverrides(): Partial<ParameterOverrides> {
  const [blockX, blockY, blockZ] = parameters.block.position;
  const all: ParameterOverrides = {
    hand: parameters.handedness,
    duration_s: parameters.duration,
    arm_height: parameters.armHeight,
    arm_depth: parameters.armDepth,
    lateral_offset: parameters.armLateral,
    wrist_pitch: parameters.wristPitch / 180,
    wrist_yaw: parameters.wristYaw / 180,
    wrist_roll: parameters.wristRoll / 180,
    elbow_swivel: parameters.elbowSwivel / 90,
    torso_participation: parameters.torsoParticipation,
    path_arc: parameters.pathArc,
    wrist_flourish: parameters.wristFlourish,
    wrist_shake_amplitude: parameters.wristShakeAmplitude,
    wrist_shake_cycles: parameters.wristShakeCycles,
    trajectory_amplitude_m: parameters.trajectoryAmplitude,
    trajectory_cycles: parameters.trajectoryCycles,
    axial_rotation_amplitude: parameters.axialRotationAmplitude,
    body_distance_m: parameters.bodyDistance,
    body_turn_degrees: parameters.bodyTurn,
    body_height_m: parameters.bodyHeight,
    body_cycles: parameters.bodyCycles,
    body_intensity: parameters.bodyIntensity,
    object_distance_m: parameters.objectDistance,
    object_apex_height_m: parameters.objectApexHeight,
    object_spin_turns: parameters.objectSpinTurns,
    object_contact_height_m: parameters.objectContactHeight,
    object_contact_depth_m: parameters.objectContactDepth,
    object_landing_height_m: parameters.objectLandingHeight,
    thumb_curl: parameters.thumbCurl,
    index_curl: parameters.indexCurl,
    middle_curl: parameters.middleCurl,
    ring_curl: parameters.ringCurl,
    little_curl: parameters.littleCurl,
    finger_splay: parameters.fingerSpread,
    thumb_opposition: parameters.thumbOpposition,
    grip_force: parameters.gripForce,
    lift_height_m: parameters.liftHeight,
    hold_duration_s: parameters.holdDuration,
    block_x: blockX,
    block_y: blockY,
    block_z: blockZ,
    block_width_m: parameters.block.width,
    block_height_m: parameters.block.height,
    block_depth_m: parameters.block.depth,
    block_mass_kg: parameters.block.mass,
    block_friction: parameters.block.friction,
  };
  return Object.fromEntries(
    [...dirtyOverrides].map((key) => [key, all[key]]),
  ) as Partial<ParameterOverrides>;
}

function overrideKeyForSlider(spec: SliderSpec): keyof ParameterOverrides {
  const keys: Record<SliderSpec["key"], keyof ParameterOverrides> = {
    duration: "duration_s",
    holdDuration: "hold_duration_s",
    armLateral: "lateral_offset",
    armHeight: "arm_height",
    armDepth: "arm_depth",
    elbowSwivel: "elbow_swivel",
    torsoParticipation: "torso_participation",
    pathArc: "path_arc",
    wristFlourish: "wrist_flourish",
    wristShakeAmplitude: "wrist_shake_amplitude",
    wristShakeCycles: "wrist_shake_cycles",
    trajectoryAmplitude: "trajectory_amplitude_m",
    trajectoryCycles: "trajectory_cycles",
    axialRotationAmplitude: "axial_rotation_amplitude",
    bodyDistance: "body_distance_m",
    bodyTurn: "body_turn_degrees",
    bodyHeight: "body_height_m",
    bodyCycles: "body_cycles",
    bodyIntensity: "body_intensity",
    objectDistance: "object_distance_m",
    objectApexHeight: "object_apex_height_m",
    objectSpinTurns: "object_spin_turns",
    objectContactHeight: "object_contact_height_m",
    objectContactDepth: "object_contact_depth_m",
    objectLandingHeight: "object_landing_height_m",
    wristPitch: "wrist_pitch",
    wristYaw: "wrist_yaw",
    wristRoll: "wrist_roll",
    thumbCurl: "thumb_curl",
    thumbOpposition: "thumb_opposition",
    indexCurl: "index_curl",
    middleCurl: "middle_curl",
    ringCurl: "ring_curl",
    littleCurl: "little_curl",
    fingerSpread: "finger_splay",
    gripForce: "grip_force",
    liftHeight: "lift_height_m",
    width: "block_width_m",
    height: "block_height_m",
    depth: "block_depth_m",
    mass: "block_mass_kg",
    friction: "block_friction",
    blockX: "block_x",
    blockY: "block_y",
    blockZ: "block_z",
  };
  return keys[spec.key];
}

function getSliderValue(spec: SliderSpec): number {
  if (spec.key === "blockX") return parameters.block.position[0];
  if (spec.key === "blockY") return parameters.block.position[1];
  if (spec.key === "blockZ") return parameters.block.position[2];
  if (spec.block) return parameters.block[spec.key as BlockNumericKey];
  return parameters[spec.key as NumericKey];
}

function setSliderValue(spec: SliderSpec, value: number): void {
  if (spec.key === "blockX") parameters.block.position[0] = value;
  else if (spec.key === "blockY") parameters.block.position[1] = value;
  else if (spec.key === "blockZ") parameters.block.position[2] = value;
  else if (spec.block) parameters.block[spec.key as BlockNumericKey] = value;
  else parameters[spec.key as NumericKey] = value;
}

function displayValue(value: number, spec: SliderSpec): string {
  const places = spec.step >= 1 ? 0 : spec.step >= 0.1 ? 1 : spec.step >= 0.01 ? 2 : 3;
  return `${value.toFixed(places)}${spec.unit ? ` ${spec.unit}` : ""}`;
}

function renderControls(): void {
  ui.controls.replaceChildren();
  const groups = new Map<string, SliderSpec[]>();
  for (const spec of sliderSpecs) groups.set(spec.group, [...(groups.get(spec.group) ?? []), spec]);
  for (const [name, specs] of groups) {
    const details = document.createElement("details");
    details.className = "control-group";
    details.open = ["Timing", "Object flight", "Arm pose", "Hand", "Physical grasp", "Block"].includes(name);
    details.innerHTML = `<summary><span>${name}</span><i>⌄</i></summary>`;
    const body = document.createElement("div");
    body.className = "control-group-body";
    for (const spec of specs) {
      const value = getSliderValue(spec);
      const control = document.createElement("label");
      control.className = "range-control";
      control.innerHTML = `
        <span>${spec.label}</span><output>${displayValue(value, spec)}</output>
        <input type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${value}" aria-label="${spec.label}" />
      `;
      const input = control.querySelector<HTMLInputElement>("input");
      const output = control.querySelector<HTMLOutputElement>("output");
      input?.addEventListener("input", () => {
        const next = Number(input.value);
        setSliderValue(spec, next);
        dirtyOverrides.add(overrideKeyForSlider(spec));
        if (output) output.value = displayValue(next, spec);
        if (spec.block) scene.updateBlock(parameters.block);
        scheduleCompile();
      });
      body.append(control);
    }
    details.append(body);
    ui.controls.append(details);
  }
}

function setBusy(busy: boolean, phase = "Planning motion"): void {
  ui.generate.disabled = busy;
  ui.generate.classList.toggle("is-loading", busy);
  ui.generate.querySelector(".button-label")!.textContent = busy ? phase : "Generate 5 & choose";
  ui.progress.classList.toggle("active", busy);
}

function setConnection(connected: boolean): void {
  ui.connection.classList.toggle("connected", connected);
  ui.connection.classList.toggle("offline", !connected);
  ui.connection.querySelector("span")!.textContent = connected ? "Backend ready" : "Backend offline";
}

function showToast(message: string, tone: "success" | "error" = "success"): void {
  window.clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.className = `toast visible ${tone}`;
  toastTimer = window.setTimeout(() => ui.toast.classList.remove("visible"), 3800);
}

function setRunState(title: string, detail: string, state: "idle" | "working" | "pass" | "fail"): void {
  ui.runTitle.textContent = title;
  ui.runDetail.textContent = detail;
  ui.runIcon.textContent = state === "pass" ? "✓" : state === "fail" ? "!" : state === "working" ? "◌" : "○";
  ui.runIcon.className = state;
}

function recordValue(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" ? value as Record<string, unknown> : {};
}

function listValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

const pipelineStageOrder = ["planning", "candidates", "visual_evidence", "vlm_judge", "finalize"];

function normalizedPipelineStage(stage: string): string {
  if (stage === "structural_checks") return "candidates";
  if (stage === "repair") return "vlm_judge";
  if (stage === "queued") return "planning";
  return stage;
}

function renderCandidateProgress(run: PipelineRun): void {
  const trace = recordValue(run.trace);
  const rounds = listValue(trace.rounds).map(recordValue);
  const candidateMap = new Map<string, { round: number; value: Record<string, unknown> }>();
  for (const event of run.events) {
    if (!["candidate_started", "candidate_compiled", "capture_started", "capture_ready"].includes(event.event)) continue;
    const round = Number(event.data.round ?? 1);
    const index = Number(event.data.candidate_index ?? 0);
    if (!index) continue;
    const key = `${round}-${index}`;
    const prior = candidateMap.get(key)?.value ?? {};
    const value: Record<string, unknown> = { ...prior, candidate_index: index };
    if (event.data.recipe) value.recipe = { name: event.data.recipe, purpose: event.data.purpose };
    if (event.event === "candidate_compiled") Object.assign(value, event.data);
    if (event.event === "capture_started") value.capture_started = true;
    if (event.event === "capture_ready") value.evidence_manifest = event.data.evidence_manifest ?? true;
    candidateMap.set(key, { round, value });
  }
  rounds.forEach((round, roundIndex) => {
    const roundNumber = Number(round.round ?? roundIndex + 1);
    listValue(round.candidates).forEach((candidate) => {
      const value = recordValue(candidate);
      const index = Number(value.candidate_index ?? 0);
      const prior = candidateMap.get(`${roundNumber}-${index}`)?.value ?? {};
      candidateMap.set(`${roundNumber}-${index}`, { round: roundNumber, value: { ...prior, ...value } });
    });
  });
  const candidates = [...candidateMap.values()];
  ui.candidateProgress.replaceChildren();
  if (!candidates.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = run.status === "unsupported"
      ? "No candidates were generated because this motion is outside the current prototype's supported scope."
      : run.stage === "planning"
        ? "The motion plan is being authored."
        : "Candidate details will arrive here.";
    ui.candidateProgress.append(empty);
    return;
  }
  for (const { round, value } of candidates) {
    const recipe = recordValue(value.recipe);
    const judgment = recordValue(value.judgment);
    const selected = value.result_id === run.winner_result_id;
    const structurallyValid = value.structural_valid !== false && value.compile_success !== false;
    const captured = Boolean(value.evidence_manifest);
    const judged = Object.keys(judgment).length > 0;
    const card = document.createElement("article");
    card.className = `candidate-card ${selected ? "selected" : ""} ${!structurallyValid ? "rejected" : ""}`;
    const candidateResultId = typeof value.result_id === "string" ? value.result_id : null;
    if (candidateResultId) {
      // Every candidate is written to disk whether or not it passed, so any of
      // them can be watched. Filtering decides what is selected, not what exists.
      card.classList.add("inspectable");
      card.title = "Click to watch this candidate";
      card.style.cursor = "pointer";
      card.addEventListener("click", () => {
        void loadResultForInspection(
          candidateResultId,
          structurallyValid ? `Inspecting ${candidateResultId}` : `Rejected candidate ${candidateResultId}`,
          String(listValue(value.structural_failures).join("; ") || "shown for inspection"),
        ).catch(() => showToast(`Could not open ${candidateResultId}.`, "error"));
      });
    }
    const top = document.createElement("div");
    const index = Number(value.candidate_index ?? 0);
    const title = document.createElement("strong");
    title.textContent = `R${round} · ${index || "?"} · ${String(recipe.name ?? "candidate").replaceAll("_", " ")}`;
    const badge = document.createElement("span");
    badge.textContent = candidateProgressBadge(value, selected);
    top.append(title, badge);
    const detail = document.createElement("p");
    detail.textContent = value.rejection_stage
      ? String(value.rejection_reason ?? "Not selected for the final judging batch.")
      : judged
        ? `Overall ${String(judgment.overall ?? "–")}/5 · anatomy ${String(judgment.anatomical_naturalness ?? "–")}/5 · semantic ${String(judgment.semantic_match ?? "–")}/5`
        : String(recipe.purpose ?? (captured ? "Full-FOV evidence ready." : "Deterministic checks in progress."));
    card.append(top, detail);
    ui.candidateProgress.append(card);
  }
}

function renderPipeline(run: PipelineRun): void {
  ui.pipelineTitle.textContent = pipelineStageLabel(run.stage, run.status);
  ui.pipelineCount.textContent = `${run.events.length} event${run.events.length === 1 ? "" : "s"}`;
  const normalized = normalizedPipelineStage(run.stage);
  const activeIndex = pipelineStageOrder.indexOf(normalized);
  ui.pipelineStages.querySelectorAll<HTMLElement>("[data-stage]").forEach((item, index) => {
    const terminalWithoutWinner = isTerminalWithoutWinner(run.status);
    item.classList.toggle("active", index === activeIndex && run.status !== "completed" && !terminalWithoutWinner);
    item.classList.toggle("complete", index < activeIndex || run.status === "completed");
    item.classList.toggle("failed", terminalWithoutWinner && index === Math.max(0, activeIndex));
  });

  const planned = [...run.events].reverse().find((event) => event.event === "plan_ready");
  const plannedProgram = recordValue(planned?.data.program);
  if (Object.keys(plannedProgram).length && program?.source_text !== plannedProgram.source_text) {
    program = plannedProgram as MotionProgram;
    const plannedHand = program.hand ?? program.handedness;
    if (plannedHand) {
      parameters.handedness = plannedHand;
      ui.handedness.value = plannedHand;
    }
    renderProgram();
  }

  renderCandidateProgress(run);
  ui.pipelineEvents.replaceChildren();
  for (const event of run.events.slice(-24).reverse()) {
    const row = document.createElement("li");
    const marker = document.createElement("i");
    const body = document.createElement("div");
    const message = document.createElement("strong");
    const meta = document.createElement("span");
    message.textContent = event.message;
    const time = new Date(event.at);
    const stageLabel = event.event === "pipeline_unsupported"
      ? "Unsupported motion"
      : pipelineStageLabel(event.stage, event.event === "pipeline_finished" ? run.status : undefined);
    meta.textContent = `${stageLabel} · ${Number.isNaN(time.valueOf()) ? "" : time.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })}`;
    body.append(message, meta);
    row.append(marker, body);
    ui.pipelineEvents.append(row);
  }
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function watchPipeline(initial: PipelineRun): Promise<PipelineRun> {
  let run = initial;
  localStorage.setItem("rigby.activePipelineRun", run.run_id);
  const url = new URL(window.location.href);
  url.searchParams.set("pipelineRun", run.run_id);
  window.history.replaceState({}, "", url);
  renderPipeline(run);
  while (["queued", "running"].includes(run.status)) {
    const latestEvent = run.events.at(-1);
    setBusy(true, pipelineStageLabel(run.stage, run.status));
    setRunState(
      pipelineStageLabel(run.stage, run.status),
      latestEvent?.message ?? "The autonomous run is progressing.",
      "working",
    );
    await wait(700);
    run = await api.getPipeline(run.run_id);
    renderPipeline(run);
  }
  return run;
}

/** Load any saved result into the viewer. Used for accepted winners and for
 *  rejected candidates alike -- a failure you cannot watch is not evidence. */
async function loadResultForInspection(resultId: string, label: string, detail: string): Promise<boolean> {
  const loaded = await api.getResult(resultId);
  clip = loaded;
  program = loaded.program ?? program;
  currentTime = 0;
  setPlaying(false);
  renderProgram();
  renderClip();
  setRunState(label, detail, "fail");
  return false;
}

async function showPipelineOutcome(run: PipelineRun): Promise<boolean> {
  if (run.status !== "completed" || !run.winner_result_id) {
    const trace = recordValue(run.trace);
    const inspectionId = run.inspection_result_id
      ?? (typeof trace.inspection_result_id === "string" ? trace.inspection_result_id : null);
    const inspectionReason = run.inspection_reason
      ?? (typeof trace.inspection_reason === "string" ? trace.inspection_reason : "");

    // A rejection nobody can watch is indistinguishable from the pipeline
    // breaking. When every candidate failed, still load the closest one so the
    // failure is on screen -- labelled as a failure, never as a result.
    if (inspectionId) {
      try {
        const detail = inspectionReason
          ? `No candidate passed. Showing the closest one so you can watch it fail: ${inspectionReason}`
          : "No candidate passed. Showing the closest one so you can watch it fail.";
        showToast(detail, "error");
        return await loadResultForInspection(inspectionId, "Rejected — shown for inspection", detail);
      } catch {
        // fall through to the empty state below
      }
    }

    clip = null;
    currentTime = 0;
    setPlaying(false);
    renderClip();
    const message = run.error?.message ?? (
      listValue(trace.rankings).length === 0
        ? "The pipeline could not assemble a complete five-candidate judging batch."
        : "None of the candidates passed the automatic visual-quality threshold."
    );
    setRunState(run.status === "unsupported" ? "Unsupported motion" : "No animation selected", message, "fail");
    showToast(message, "error");
    return false;
  }
  const result = await api.getResult(run.winner_result_id);
  clip = result;
  program = result.program ?? program;
  if (!program) throw new Error("The selected animation did not include its motion program.");
  currentTime = 0;
  setPlaying(false);
  renderProgram();
  renderClip();
  const judgedCount = Number(run.summary?.judged_candidate_count ?? run.summary?.candidate_count ?? 5);
  const attemptedCount = Number(run.summary?.attempted_candidate_count ?? judgedCount);
  const screeningDetail = attemptedCount > judgedCount
    ? ` after screening ${attemptedCount} proposals`
    : "";
  setRunState(
    "Final animation ready",
    `Selected from ${judgedCount} judged candidates${screeningDetail} · ${clip.frames.length} frames · ${clip.duration.toFixed(2)} seconds.`,
    "pass",
  );
  showToast("The automatic judge selected the final animation.");
  void refreshResults();
  return true;
}

async function resumeSavedPipeline(): Promise<void> {
  const linkedRunId = new URLSearchParams(window.location.search).get("pipelineRun");
  const runId = linkedRunId ?? localStorage.getItem("rigby.activePipelineRun");
  if (!runId) return;
  setBusy(true, "Resuming autonomous run…");
  try {
    const run = await watchPipeline(await api.getPipeline(runId));
    ui.prompt.value = run.prompt;
    updatePromptCount();
    await showPipelineOutcome(run);
    setConnection(true);
  } catch (error) {
    showToast(`Could not resume the saved run: ${errorMessage(error)}`, "error");
  } finally {
    localStorage.removeItem("rigby.activePipelineRun");
    setBusy(false);
  }
}

function renderProgram(): void {
  const cyclic = program?.primitives?.find((primitive) => primitive.kind === "cycle");
  if (cyclic?.parameters) {
    const amplitude = Number(cyclic.parameters.trajectory_amplitude_m);
    const cycles = Number(cyclic.parameters.trajectory_cycles);
    const axialAmplitude = Number(cyclic.parameters.axial_rotation_amplitude);
    if (Number.isFinite(amplitude)) parameters.trajectoryAmplitude = amplitude;
    if (Number.isFinite(cycles)) parameters.trajectoryCycles = cycles;
    if (Number.isFinite(axialAmplitude)) parameters.axialRotationAmplitude = axialAmplitude;
  }
  const bodyPrimitive = program?.primitives?.find((primitive) => primitive.kind === "body" && primitive.body);
  if (bodyPrimitive?.body) {
    parameters.bodyDistance = Number(bodyPrimitive.body.distance_m ?? 0);
    parameters.bodyTurn = Number(bodyPrimitive.body.turn_degrees ?? 0);
    parameters.bodyHeight = Number(bodyPrimitive.body.height_m ?? 0);
    parameters.bodyCycles = Number(bodyPrimitive.body.cycles ?? 1);
    parameters.bodyIntensity = Number(bodyPrimitive.body.intensity ?? 0.65);
  }
  if (program?.object_motion) {
    parameters.objectDistance = Number(program.object_motion.distance_m ?? 0.85);
    parameters.objectApexHeight = Number(program.object_motion.apex_height_m ?? 0.35);
    parameters.objectSpinTurns = Number(program.object_motion.spin_turns ?? 0.35);
    parameters.objectContactHeight = Number(program.object_motion.contact_height_m ?? 1.25);
    parameters.objectContactDepth = Number(program.object_motion.contact_depth_m ?? 0.38);
    parameters.objectLandingHeight = Number(program.object_motion.landing_height_m ?? 0.04);
  }
  ui.programIntent.textContent = String(program?.intent ?? "No plan").replaceAll("_", " ");
  scene.setTaskEnvironmentVisible(
    shouldShowTaskEnvironment(program),
    shouldShowTaskSupportSurface(program),
  );
  ui.programJson.textContent = program ? JSON.stringify(program, null, 2) : "Generate a motion to inspect its semantic program.";
  const taskObjectId = activeTaskObjectId(program);
  const taskObject = sceneManifest().objects.find((item) => item.id === taskObjectId);
  if (taskObject) {
    scene.updateBlock({
      kind: taskObject.kind,
      width: taskObject.dimensions_m.x,
      height: taskObject.dimensions_m.y,
      depth: taskObject.dimensions_m.z,
      mass: taskObject.mass_kg,
      friction: taskObject.friction,
      position: [
        taskObject.transform.translation.x,
        taskObject.transform.translation.y,
        taskObject.transform.translation.z,
      ],
    });
  }
  scene.setObjectId(taskObjectId);
  const isBilateral = (program?.hands?.length ?? 0) > 1;
  ui.handedness.disabled = isBilateral;
  ui.handedness.title = isBilateral
    ? "This motion coordinates both hands; handedness is authored per phase."
    : "Choose the active hand.";
  renderControls();
  renderPhases();
}

function renderPhases(): void {
  ui.timelinePhases.replaceChildren();
  const primitives = program?.primitives ?? [];
  const primitiveDuration = (item: MotionPrimitive): number =>
    Number(item.parameters?.duration_s ?? item.duration ?? 0);
  const total = primitives.reduce((sum, item) => sum + primitiveDuration(item), 0) || parameters.duration;
  let cursor = 0;
  for (const primitive of primitives) {
    const duration = primitiveDuration(primitive) || total / Math.max(1, primitives.length);
    const marker = document.createElement("span");
    marker.style.left = `${(cursor / total) * 100}%`;
    marker.title = String(primitive.label ?? primitive.kind ?? primitive.type);
    ui.timelinePhases.append(marker);
    cursor += duration;
  }
}

function renderClip(): void {
  const duration = Math.max(clip?.duration ?? parameters.duration, 0.001);
  ui.timeline.max = String(duration);
  ui.totalTime.textContent = formatTime(duration);
  ui.export.disabled = !clip?.id || clip.id === "local-demo" || clip.status === "failed";
  currentTime = Math.min(currentTime, duration);
  renderMetrics();
  updateFrame();
}

function renderMetrics(): void {
  ui.metrics.replaceChildren();
  const graspOnlyMetrics = new Set([
    "lift_height_m", "lost_table_contact", "opposing_contacts", "hold_duration_s",
    "vertical_drift_m", "palm_relative_slip_m", "opposing_contact_ratio", "grasp_attempts",
  ]);
  const metrics = Object.entries(clip?.metrics ?? {}).filter(
    ([key]) => program?.intent === "grab" || !graspOnlyMetrics.has(key),
  );
  const passed = clip && !clip.failure && clip.status !== "failed";
  ui.verificationChip.textContent = clip ? (passed ? "Passed" : "Failed") : "Not run";
  ui.verificationChip.className = `status-chip ${clip ? (passed ? "passed" : "failed") : "neutral"}`;
  if (!metrics.length) ui.metrics.innerHTML = `<p class="empty-state">Metrics appear after compilation.</p>`;
  for (const [key, value] of metrics) {
    const row = document.createElement("div");
    const positive = key === "weld_used" && typeof value === "boolean" ? !value : value;
    const booleanClass = typeof positive === "boolean" ? (positive ? "metric-pass" : "metric-fail") : "";
    const display = key === "weld_used" && typeof value === "boolean" ? (value ? "Fail" : "Pass") : formatMetric(value);
    row.innerHTML = `<span>${key.replaceAll("_", " ")}</span><strong class="${booleanClass}">${display}</strong>`;
    ui.metrics.append(row);
  }
  const observables = Object.entries(clip?.sliderObservables ?? {});
  if (observables.length) {
    const label = document.createElement("p");
    label.className = "metric-subheading";
    label.textContent = "Slider observables";
    ui.metrics.append(label);
    for (const [key, value] of observables) {
      const row = document.createElement("div");
      row.innerHTML = `<span>${key.replaceAll("_", " ")}</span><strong>${formatMetric(value)}</strong>`;
      ui.metrics.append(row);
    }
  }

  ui.failurePanel.hidden = !clip?.failure;
  if (clip?.failure) {
    ui.failureCode.textContent = clip.failure.code.replaceAll("_", " ");
    ui.failureMessage.textContent = clip.failure.message;
  }

  const provenance = clip?.provenance ?? {};
  ui.provenance.replaceChildren();
  const entries: Array<[string, unknown]> = [
    ["Result", clip?.id ?? "Unsaved"],
    ["Model", provenance.model ?? provenance.planner_model],
    ["Provider", provenance.planner_provider],
    ["Seed", provenance.seed],
    ["Rig", provenance.rigVersion ?? provenance.rig_version ?? provenance.rig_id],
    ["Compiler", provenance.compilerVersion ?? provenance.compiler_version],
    ["Physics", [provenance.physics_engine, provenance.physics_version].filter(Boolean).join(" ") || undefined],
    ["Model calls", provenance.model_calls],
    ["Generated", provenance.generatedAt ?? provenance.generated_at ?? clip?.createdAt ?? clip?.created_at],
  ];
  for (const [label, value] of entries.filter(([, value]) => value !== undefined)) {
    const row = document.createElement("div");
    row.innerHTML = `<dt>${label}</dt><dd>${String(value)}</dd>`;
    ui.provenance.append(row);
  }
}

function activeContacts(time: number): ContactEvent[] {
  return (clip?.contacts ?? []).filter((contact) => {
    if (contact.time === undefined && contact.frame === undefined) return true;
    const eventTime = contact.time ?? (contact.frame ?? 0) / (clip?.fps ?? 30);
    return Math.abs(eventTime - time) < 0.12;
  });
}

function updateFrame(): void {
  if (clip && playing) {
    const elapsed = (performance.now() - playStartedAt) / 1000;
    currentTime = playStartedFrom + elapsed;
    if (currentTime >= clip.duration) {
      if (ui.loop.checked) {
        currentTime %= clip.duration || 1;
        playStartedAt = performance.now();
        playStartedFrom = currentTime;
      } else {
        currentTime = clip.duration;
        setPlaying(false);
      }
    }
  }
  const frame = frameAt(clip, currentTime);
  scene.applyFrame(frame, activeContacts(currentTime));
  ui.timeline.value = String(currentTime);
  ui.currentTime.textContent = formatTime(currentTime);
  const frameNumber = Math.round(currentTime * (clip?.fps ?? 30));
  ui.frameReadout.textContent = `Frame ${String(frameNumber).padStart(3, "0")} · ${currentTime.toFixed(2)}s`;
  requestAnimationFrame(updateFrame);
}

function formatTime(time: number): string {
  const minutes = Math.floor(time / 60);
  return `${minutes}:${(time % 60).toFixed(2).padStart(5, "0")}`;
}

function setPlaying(value: boolean): void {
  if (value && !clip) return;
  playing = value;
  playStartedAt = performance.now();
  playStartedFrom = currentTime >= (clip?.duration ?? 0) ? 0 : currentTime;
  if (playing) currentTime = playStartedFrom;
  ui.play.textContent = playing ? "❚❚" : "▶";
  ui.play.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
}

async function generate(): Promise<void> {
  const prompt = ui.prompt.value.trim();
  if (!prompt) {
    ui.prompt.focus();
    showToast("Describe a motion before generating.", "error");
    return;
  }
  clip = null;
  program = null;
  currentTime = 0;
  setPlaying(false);
  renderProgram();
  renderClip();
  setBusy(true, "Starting autonomous run…");
  setRunState("Starting the pipeline", "The planner will create five candidates before the visual judge selects one.", "working");
  try {
    dirtyOverrides.clear();
    const started = await api.startPipeline({
      text: prompt,
      scene: sceneManifest(),
      provider: "auto",
      max_rounds: 2,
    });
    setConnection(true);
    await showPipelineOutcome(await watchPipeline(started));
  } catch (error) {
    setConnection(false);
    const message = errorMessage(error);
    setRunState("Generation failed", message, "fail");
    showToast(message, "error");
  } finally {
    localStorage.removeItem("rigby.activePipelineRun");
    setBusy(false);
  }
}

async function compileNow(parameterEdit: boolean): Promise<void> {
  if (!program) return;
  const revision = ++compileRevision;
  if (parameterEdit) setRunState("Recompiling parameters", "No planning request was made.", "working");
  try {
    const result = await api.compile({ program, scene: sceneManifest(), parameter_overrides: parameterOverrides(), persist: true });
    if (revision !== compileRevision) return;
    clip = { ...result, program: result.program ?? program, prompt: result.prompt ?? ui.prompt.value.trim() };
    setConnection(true);
    currentTime = 0;
    setPlaying(false);
    renderClip();
    if (clip.failure || clip.status === "failed") {
      setRunState("Constraint failure", clip.failure?.message ?? "The clip did not pass verification.", "fail");
    } else {
      setRunState("Verified motion", `${clip.frames.length} frames · ${clip.duration.toFixed(2)} seconds · deterministic compile`, "pass");
    }
  } catch (error) {
    if (revision !== compileRevision) return;
    setConnection(false);
    const message = errorMessage(error);
    setRunState("Compilation failed", message, "fail");
    if (parameterEdit) showToast(message, "error");
    else throw error;
  }
}

function scheduleCompile(): void {
  window.clearTimeout(compileTimer);
  compileTimer = window.setTimeout(() => void compileNow(true), 260);
}

function openResults(open: boolean): void {
  ui.resultsDrawer.classList.toggle("open", open);
  ui.drawerScrim.classList.toggle("visible", open);
  ui.resultsDrawer.setAttribute("aria-hidden", String(!open));
  ui.resultsToggle.setAttribute("aria-expanded", String(open));
  if (open) void refreshResults();
}

async function refreshResults(): Promise<void> {
  ui.resultsList.innerHTML = `<p class="empty-state loading-line">Loading results…</p>`;
  let results: ResultSummary[] = [];
  try {
    results = await api.listResults();
    setConnection(true);
  } catch {
    setConnection(false);
  }
  const rows = [DEMO_SUMMARY, ...results.filter((result) => result.id !== DEMO_SUMMARY.id)];
  ui.resultsList.replaceChildren();
  for (const result of rows) {
    const button = document.createElement("button");
    button.className = "result-card";
    button.innerHTML = `
      <span class="result-card-top"><i class="${result.status === "failed" ? "failed" : "passed"}"></i><time>${formatResultDate(result.createdAt ?? result.created_at)}</time></span>
      <strong>${result.prompt ?? "Untitled motion"}</strong>
      <span>${result.id === "local-demo" ? "Bundled replay fixture" : `Result ${result.id}`} · ${(result.duration ?? 0).toFixed(2)}s</span>
    `;
    button.addEventListener("click", () => void loadResult(result));
    ui.resultsList.append(button);
  }
}

function formatResultDate(value: string | undefined): string {
  if (!value) return "Unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return date.valueOf() === 0 ? "Local sample" : date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

async function loadResult(summary: ResultSummary): Promise<void> {
  try {
    const result = summary.id === "local-demo" ? DEMO_CLIP : await api.getResult(summary.id);
    clip = result;
    dirtyOverrides.clear();
    program = result.program ?? (summary.id === "local-demo" ? DEMO_PROGRAM : null);
    if (result.prompt) ui.prompt.value = result.prompt;
    updatePromptCount();
    renderProgram();
    currentTime = 0;
    renderClip();
    setRunState("Replaying saved result", `Loaded ${result.id ?? summary.id} without invoking the planner.`, result.failure ? "fail" : "pass");
    openResults(false);
    showToast("Saved result loaded for replay.");
  } catch (error) {
    showToast(errorMessage(error), "error");
  }
}

function updatePromptCount(): void {
  ui.promptCount.textContent = `${ui.prompt.value.length} / 400`;
}

ui.generate.addEventListener("click", () => void generate());
ui.prompt.addEventListener("input", updatePromptCount);
ui.prompt.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    void generate();
  }
});
document.querySelectorAll<HTMLButtonElement>("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    ui.prompt.value = button.dataset.prompt ?? "";
    updatePromptCount();
    ui.prompt.focus();
  });
});
document.querySelectorAll<HTMLButtonElement>("[data-camera]").forEach((button) => {
  button.addEventListener("click", () => {
    const camera = button.dataset.camera as CameraMode;
    scene.setCamera(camera);
    document.querySelectorAll<HTMLButtonElement>("[data-camera]").forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
  });
});
element<HTMLInputElement>("#debug-toggle").addEventListener("change", (event) => {
  scene.setDebug((event.currentTarget as HTMLInputElement).checked);
});
ui.handedness.addEventListener("change", () => {
  parameters.handedness = ui.handedness.value as MotionParameters["handedness"];
  dirtyOverrides.add("hand");
  scheduleCompile();
});
element<HTMLButtonElement>("#reset-params").addEventListener("click", () => {
  parameters = structuredClone(DEFAULT_PARAMETERS);
  dirtyOverrides.clear();
  if (program?.hand ?? program?.handedness) parameters.handedness = (program.hand ?? program.handedness)!;
  ui.handedness.value = parameters.handedness;
  scene.updateBlock(parameters.block);
  renderControls();
  scheduleCompile();
  showToast("Parameters reset to the motion defaults.");
});
ui.play.addEventListener("click", () => setPlaying(!playing));
ui.timeline.addEventListener("input", () => {
  setPlaying(false);
  currentTime = Number(ui.timeline.value);
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLTextAreaElement) && !(event.target instanceof HTMLSelectElement)) {
    event.preventDefault();
    setPlaying(!playing);
  }
  if (event.key === "Escape") openResults(false);
});
ui.resultsToggle.addEventListener("click", () => openResults(!ui.resultsDrawer.classList.contains("open")));
ui.resultsClose.addEventListener("click", () => openResults(false));
ui.drawerScrim.addEventListener("click", () => openResults(false));
ui.export.addEventListener("click", async () => {
  if (!clip || !program) return;
  ui.export.disabled = true;
  try {
    if (!clip.id) throw new Error("This clip has not been persisted and cannot be exported.");
    await api.exportResult({ result_id: clip.id });
    showToast("GLB export is ready.");
  } catch (error) {
    showToast(errorMessage(error), "error");
  } finally {
    ui.export.disabled = Boolean(clip.failure) || !clip.id || clip.id === "local-demo";
  }
});

renderControls();
updatePromptCount();
renderClip();
void refreshResults();
void resumeSavedPipeline();
