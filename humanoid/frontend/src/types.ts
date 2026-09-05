export type Handedness = "left" | "right";
export type CameraMode = "ego" | "orbit";

export interface BlockParameters {
  kind?: string;
  width: number;
  height: number;
  depth: number;
  mass: number;
  friction: number;
  position: [number, number, number];
}

/** The surface objects rest on, read from the scene manifest's `table` object. */
export interface SupportSurfaceParameters {
  width: number;
  height: number;
  depth: number;
  position: [number, number, number];
}

export interface MotionParameters {
  handedness: Handedness;
  duration: number;
  armLateral: number;
  armHeight: number;
  armDepth: number;
  wristPitch: number;
  wristYaw: number;
  wristRoll: number;
  elbowSwivel: number;
  torsoParticipation: number;
  pathArc: number;
  wristFlourish: number;
  wristShakeAmplitude: number;
  wristShakeCycles: number;
  trajectoryAmplitude: number;
  trajectoryCycles: number;
  axialRotationAmplitude: number;
  bodyDistance: number;
  bodyTurn: number;
  bodyHeight: number;
  bodyCycles: number;
  bodyIntensity: number;
  objectDistance: number;
  objectApexHeight: number;
  objectSpinTurns: number;
  objectContactHeight: number;
  objectContactDepth: number;
  objectLandingHeight: number;
  thumbCurl: number;
  thumbOpposition: number;
  indexCurl: number;
  middleCurl: number;
  ringCurl: number;
  littleCurl: number;
  fingerSpread: number;
  gripForce: number;
  liftHeight: number;
  holdDuration: number;
  block: BlockParameters;
}

export interface MotionPrimitive {
  type?: string;
  kind?: string;
  duration?: number;
  hand_shape?: string | null;
  object_id?: string | null;
  socket_id?: string | null;
  parameters?: Record<string, unknown>;
  label?: string | null;
  trajectory?: "linear" | "arc" | "circle" | "oscillate" | "hold" | null;
  trajectory_plane?: "frontal" | "sagittal" | "horizontal" | null;
  effectors?: Array<{
    hand: Handedness;
    target_x: number;
    target_y: number;
    target_z: number;
    elbow_swivel?: number;
    wrist_pitch?: number;
    wrist_yaw?: number;
    wrist_roll?: number;
    phase_offset_cycles?: number;
    hand_shape?: string;
  }>;
  intra_hand_contact?: {
    hand: Handedness;
    driver_digit: "thumb" | "index" | "middle" | "ring" | "little";
    target_digit: "thumb" | "index" | "middle" | "ring" | "little";
    maximum_distance_m: number;
  } | null;
  gaze_target?: {
    hand?: Handedness | null;
    object_id?: string | null;
    maximum_angle_deg: number;
  } | null;
  body?: {
    action: "hold" | "step" | "walk" | "run" | "turn" | "crouch" | "jump" | "kick" | "dance" | "climb" | "rotate" | "pose";
    direction_x: number;
    direction_z: number;
    distance_m: number;
    turn_degrees: number;
    height_m: number;
    cycles: number;
    intensity: number;
    lead_side: Handedness;
    obstacle_mode?: "none" | "over" | "around";
    obstacle_object_id?: string | null;
    support_object_id?: string | null;
    climb_direction?: "up" | "down";
    obstacle_clearance_m?: number;
    path_lateral_offset_m?: number;
  } | null;
  [key: string]: unknown;
}

export interface MotionProgram {
  schema_version?: string;
  source_text?: string;
  version?: string;
  intent?: string;
  handedness?: Handedness;
  hand?: Handedness;
  hands?: Handedness[];
  strike_type?: "hook" | "jab" | "cross" | "uppercut" | null;
  object_action?: "throw" | "catch" | "push" | "pull" | "roll" | "spin" | "place" | "drop" | "handoff" | null;
  object_motion?: {
    style: "overhand" | "underhand" | "sidearm" | "toss";
    direction_x: number;
    direction_z: number;
    distance_m: number;
    apex_height_m: number;
    spin_turns: number;
    contact_height_m: number;
    contact_depth_m: number;
    landing_height_m: number;
  } | null;
  steps?: MotionProgram[];
  objectId?: string | null;
  object_id?: string | null;
  primitives?: MotionPrimitive[];
  assertions?: Array<Record<string, unknown> | string>;
  parameters?: Record<string, unknown>;
  seed?: number;
  unsupportedReason?: string | null;
  unsupported_reason?: string | null;
  [key: string]: unknown;
}

export interface BonePose {
  rotation?: [number, number, number, number];
  position?: [number, number, number];
}

export interface ObjectPose {
  position?: [number, number, number];
  rotation?: [number, number, number, number];
  scale?: [number, number, number];
}

export interface MotionFrame {
  time: number;
  bones: Record<string, BonePose | [number, number, number, number]>;
  objects?: Record<string, ObjectPose>;
  block?: ObjectPose;
}

export type MetricValue = number | string | boolean | null;

export interface ContactEvent {
  time_s?: number;
  time?: number;
  frame?: number;
  position?: [number, number, number];
  normal?: [number, number, number];
  bodyA?: string;
  bodyB?: string;
  body_a?: string;
  body_b?: string;
  active?: boolean;
  [key: string]: unknown;
}

export interface ClipFailure {
  code: string;
  message: string;
  details?: unknown;
}

export interface ClipProvenance {
  model?: string;
  seed?: number;
  rigVersion?: string;
  rig_version?: string;
  generatedAt?: string;
  generated_at?: string;
  compilerVersion?: string;
  compiler_version?: string;
  rig_id?: string;
  planner_provider?: string;
  planner_model?: string;
  physics_engine?: string;
  physics_version?: string;
  model_calls?: number;
  [key: string]: unknown;
}

export interface ClipResult {
  id?: string;
  status?: "success" | "failed" | string;
  duration: number;
  fps: number;
  frames: MotionFrame[];
  metrics: Record<string, MetricValue>;
  sliderObservables?: Record<string, number>;
  contacts: ContactEvent[];
  failure?: ClipFailure | null;
  provenance: ClipProvenance;
  program?: MotionProgram;
  prompt?: string;
  createdAt?: string;
  created_at?: string;
  glbUrl?: string;
  glb_url?: string;
  [key: string]: unknown;
}

export interface ResultSummary {
  id: string;
  prompt?: string;
  status?: string;
  createdAt?: string;
  created_at?: string;
  duration?: number;
  metrics?: Record<string, MetricValue>;
  thumbnail?: string;
  [key: string]: unknown;
}

export interface PlanRequest {
  text: string;
  scene: SceneManifest;
  provider: "auto" | "openai" | "offline";
}

export interface CompileRequest {
  program: MotionProgram;
  scene: SceneManifest;
  parameter_overrides: Partial<ParameterOverrides>;
  persist: boolean;
}

export interface ParameterOverrides {
  hand: Handedness;
  duration_s: number;
  arm_height: number;
  arm_depth: number;
  lateral_offset: number;
  wrist_pitch: number;
  wrist_yaw: number;
  wrist_roll: number;
  elbow_swivel: number;
  torso_participation: number;
  path_arc: number;
  wrist_flourish: number;
  wrist_shake_amplitude: number;
  wrist_shake_cycles: number;
  trajectory_amplitude_m: number;
  trajectory_cycles: number;
  axial_rotation_amplitude: number;
  body_distance_m: number;
  body_turn_degrees: number;
  body_height_m: number;
  body_cycles: number;
  body_intensity: number;
  object_distance_m: number;
  object_apex_height_m: number;
  object_spin_turns: number;
  object_contact_height_m: number;
  object_contact_depth_m: number;
  object_landing_height_m: number;
  thumb_curl: number;
  index_curl: number;
  middle_curl: number;
  ring_curl: number;
  little_curl: number;
  finger_splay: number;
  thumb_opposition: number;
  grip_force: number;
  lift_height_m: number;
  hold_duration_s: number;
  block_x: number;
  block_y: number;
  block_z: number;
  block_width_m: number;
  block_height_m: number;
  block_depth_m: number;
  block_mass_kg: number;
  block_friction: number;
}

export interface ExportRequest {
  result_id: string;
}

export interface PipelineEvent {
  sequence: number;
  at: string;
  event: string;
  stage: string;
  message: string;
  data: Record<string, unknown>;
}

export interface PipelineRun {
  run_id: string;
  prompt: string;
  status: "queued" | "running" | "completed" | "unsupported" | "no_acceptable_candidate" | "failed";
  stage: string;
  winner_result_id?: string | null;
  trace_url?: string | null;
  error?: { type?: string; message?: string } | null;
  events: PipelineEvent[];
  trace?: Record<string, unknown> | null;
  summary?: Record<string, unknown>;
}

export interface PipelineRunRequest {
  text: string;
  scene: SceneManifest;
  provider: "auto" | "openai" | "offline";
  max_rounds: 1 | 2;
}

export interface SceneManifest {
  schema_version: "1.0";
  rig: {
    id: string;
    asset_uri: string;
    profile_uri: string;
    fixed_root: boolean;
  };
  objects: Array<{
    id: string;
    kind: string;
    transform: {
      translation: { x: number; y: number; z: number };
      rotation: { x: number; y: number; z: number; w: number };
    };
    dimensions_m: { x: number; y: number; z: number };
    mass_kg: number;
    friction: number;
    sockets: Array<{
      id: string;
      transform: {
        translation: { x: number; y: number; z: number };
        rotation: { x: number; y: number; z: number; w: number };
      };
      approach_normal: { x: number; y: number; z: number };
      grasp_span_m: number;
      role?: "grasp" | "climb_contact" | "support";
      supports_body_weight?: boolean;
    }>;
  }>;
  fps: number;
  reachable_radius_m: number;
}

export const DEFAULT_PARAMETERS: MotionParameters = {
  handedness: "right",
  duration: 2.8,
  armLateral: 0,
  armHeight: 0,
  armDepth: 0,
  wristPitch: 0,
  wristYaw: 0,
  wristRoll: 0,
  elbowSwivel: 0,
  torsoParticipation: 0.3,
  pathArc: 0,
  wristFlourish: 0,
  wristShakeAmplitude: 0,
  wristShakeCycles: 0,
  trajectoryAmplitude: 0,
  trajectoryCycles: 0,
  axialRotationAmplitude: 0,
  bodyDistance: 0,
  bodyTurn: 0,
  bodyHeight: 0,
  bodyCycles: 1,
  bodyIntensity: 0.65,
  objectDistance: 0.85,
  objectApexHeight: 0.35,
  objectSpinTurns: 0.35,
  objectContactHeight: 1.25,
  objectContactDepth: 0.38,
  objectLandingHeight: 0.04,
  thumbCurl: 0.15,
  thumbOpposition: 0.65,
  indexCurl: 0.15,
  middleCurl: 0.7,
  ringCurl: 0.75,
  littleCurl: 0.1,
  fingerSpread: 0.35,
  gripForce: 0.75,
  liftHeight: 0.1,
  holdDuration: 1,
  block: {
    width: 0.07,
    height: 0.08,
    depth: 0.07,
    mass: 0.25,
    friction: 0.8,
    position: [0, 1.05, 0.29],
  },
};
