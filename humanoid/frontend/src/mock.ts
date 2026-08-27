import type { ClipResult, MotionProgram, ResultSummary } from "./types";

export const DEMO_PROGRAM: MotionProgram = {
  version: "1.0",
  intent: "grab",
  handedness: "right",
  object_id: "block",
  primitives: [
    { type: "reach", duration: 0.8 },
    { type: "preshape", duration: 0.35 },
    { type: "contact", duration: 0.2 },
    { type: "close", duration: 0.3 },
    { type: "lift", duration: 0.55 },
    { type: "hold", duration: 1 },
  ],
  assertions: ["block_lifted", "opposing_contacts", "root_fixed"],
  seed: 7,
};

export const DEMO_CLIP: ClipResult = {
  id: "local-demo",
  status: "success",
  prompt: "Grab the block in front of you",
  duration: 2.8,
  fps: 30,
  frames: Array.from({ length: 85 }, (_, index) => {
    const time = index / 30;
    const lift = Math.max(0, Math.min(1, (time - 1.25) / 0.55));
    return {
      time,
      bones: {},
      objects: {
        block: {
          position: [0, 1.05 + lift * 0.1, 0.29],
          rotation: [0, 0, 0, 1],
        },
      },
    };
  }),
  metrics: {
    verification_passed: true,
    com_lift_m: 0.1,
    hold_drift_m: 0.006,
    palm_slip_m: 0.009,
    max_penetration_m: 0.002,
    compile_seconds: 0.42,
  },
  contacts: [
    { time: 1.18, position: [-0.035, 1.05, 0.255], bodyA: "right_palm", bodyB: "block" },
    { time: 1.18, position: [0.035, 1.05, 0.255], bodyA: "right_fingers", bodyB: "block" },
  ],
  provenance: {
    model: "Local replay fixture",
    seed: 7,
    rigVersion: "mesh2motion-human-vrm1",
    compilerVersion: "demo",
    generatedAt: new Date(0).toISOString(),
  },
  sliderObservables: {
    right_thumb_curl: 0.15,
    right_index_curl: 0.15,
    right_middle_curl: 0.7,
    right_ring_curl: 0.75,
    right_little_curl: 0.1,
  },
  program: DEMO_PROGRAM,
  createdAt: new Date(0).toISOString(),
};

export const DEMO_SUMMARY: ResultSummary = {
  id: "local-demo",
  prompt: DEMO_CLIP.prompt,
  status: DEMO_CLIP.status,
  createdAt: DEMO_CLIP.createdAt,
  duration: DEMO_CLIP.duration,
  metrics: DEMO_CLIP.metrics,
};
