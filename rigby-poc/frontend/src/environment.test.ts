import { describe, expect, it } from "vitest";

import {
  activeObstacleObjectId,
  activeTaskObjectId,
  shouldShowTaskEnvironment,
  shouldShowTaskSupportSurface,
} from "./environment";
import type { MotionProgram } from "./types";

describe("task environment visibility", () => {
  it("keeps ordinary full-body previews uncluttered", () => {
    expect(shouldShowTaskEnvironment({ intent: "full_body", primitives: [] })).toBe(false);
    expect(shouldShowTaskSupportSurface({ intent: "full_body", primitives: [] })).toBe(false);
  });

  it("shows and selects the exact obstacle referenced by a body primitive", () => {
    const program: MotionProgram = {
      intent: "full_body",
      primitives: [{
        kind: "body",
        body: {
          action: "step",
          direction_x: 0,
          direction_z: 1,
          distance_m: 0.6,
          turn_degrees: 0,
          height_m: 0,
          cycles: 1,
          intensity: 0.6,
          lead_side: "right",
          obstacle_mode: "over",
          obstacle_object_id: "hurdle",
        },
      }],
    };

    expect(activeObstacleObjectId(program)).toBe("hurdle");
    expect(shouldShowTaskEnvironment(program)).toBe(true);
    expect(shouldShowTaskSupportSurface(program)).toBe(false);
  });

  it("continues to show the environment for object interactions", () => {
    expect(shouldShowTaskEnvironment({ intent: "grab" })).toBe(true);
    expect(shouldShowTaskSupportSurface({ intent: "grab" })).toBe(true);
  });

  it("selects a weight-bearing climb affordance without showing the table", () => {
    const program: MotionProgram = {
      intent: "full_body",
      primitives: [{
        kind: "body",
        body: {
          action: "climb",
          direction_x: 0,
          direction_z: 1,
          distance_m: 0,
          turn_degrees: 0,
          height_m: 0.62,
          cycles: 3,
          intensity: 0.72,
          lead_side: "right",
          support_object_id: "ladder",
          climb_direction: "up",
        },
      }],
    };

    expect(activeTaskObjectId(program)).toBe("ladder");
    expect(shouldShowTaskEnvironment(program)).toBe(true);
    expect(shouldShowTaskSupportSurface(program)).toBe(false);
  });

  it("hides unrelated scene objects for strikes and gestures", () => {
    expect(shouldShowTaskEnvironment({ intent: "strike" })).toBe(false);
    expect(shouldShowTaskSupportSurface({ intent: "gesture" })).toBe(false);
  });

  it("finds referenced objects inside composite sequences", () => {
    const program: MotionProgram = {
      intent: "sequence",
      steps: [{
        intent: "object_interaction",
        primitives: [{ kind: "reach", object_id: "parcel" }],
      }],
    };
    expect(activeTaskObjectId(program)).toBe("parcel");
    expect(shouldShowTaskEnvironment(program)).toBe(true);
  });
});
