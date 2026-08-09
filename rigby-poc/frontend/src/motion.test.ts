import { describe, expect, it } from "vitest";
import { frameAt, unwrapClip, unwrapProgram, unwrapResults } from "./motion";

describe("API response normalization", () => {
  it("unwraps program envelopes", () => {
    expect(unwrapProgram({ motion_program: { intent: "hang_ten" } }).intent).toBe("hang_ten");
  });

  it("normalizes clip defaults and selects timeline frames", () => {
    const clip = unwrapClip({ clip_result: { fps: 20, frames: [{ bones: {} }, { bones: {} }] } });
    expect(clip.frames[1]?.time).toBe(0.05);
    expect(frameAt(clip, 1)).toBe(clip.frames[1]);
  });

  it("normalizes the backend clip contract", () => {
    const clip = unwrapClip({
      result_id: "000001-grab-block",
      clip: {
        success: true,
        fps: 30,
        duration_s: 1,
        frames: [{
          time_s: 0,
          bones: { rightUpperArm: { rotation: { x: 0, y: 0.2, z: 0, w: 0.98 } } },
          objects: { block: { translation: { x: 0, y: 1.05, z: 0.29 }, rotation: { x: 0, y: 0, z: 0, w: 1 } } },
        }],
        metrics: { passed: true },
        slider_observables: { right_thumb_curl: 0.25, right_index_curl: 0.5 },
        contacts: [{ time_s: 0.5, hand: "right", object_id: "block", position: { x: 0, y: 1, z: 0.255 } }],
        provenance: { rig_id: "mesh2motion-human-vrm1" },
      },
    });
    expect(clip.id).toBe("000001-grab-block");
    expect(clip.duration).toBe(1);
    expect(clip.frames[0]?.bones.rightUpperArm).toEqual({ rotation: [0, 0.2, 0, 0.98], position: undefined });
    expect(clip.frames[0]?.objects?.block?.position).toEqual([0, 1.05, 0.29]);
    expect(clip.contacts[0]?.position).toEqual([0, 1, 0.255]);
    expect(clip.sliderObservables?.right_thumb_curl).toBe(0.25);
    expect(clip.sliderObservables?.right_index_curl).toBe(0.5);
  });

  it("normalizes result collection envelopes", () => {
    const result = unwrapResults({ items: [{ result_id: "abc", prompt: "Reach", duration_s: 1.25 }] })[0];
    expect(result?.id).toBe("abc");
    expect(result?.duration).toBe(1.25);
  });

  it("preserves replay metadata from the result-detail envelope", () => {
    const clip = unwrapClip({
      id: "000008-hang-ten",
      program: { intent: "gesture", source_text: "Throw up a hang-ten sign" },
      clip: { success: true, fps: 30, duration_s: 2.4, frames: [] },
    });
    expect(clip.id).toBe("000008-hang-ten");
    expect(clip.program?.intent).toBe("gesture");
    expect(clip.prompt).toBe("Throw up a hang-ten sign");
  });
});
