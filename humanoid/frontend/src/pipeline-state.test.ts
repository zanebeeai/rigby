import { describe, expect, it } from "vitest";

import {
  candidateProgressBadge,
  isTerminalWithoutWinner,
  pipelineStageLabel,
} from "./pipeline-state";

describe("pipeline terminal state", () => {
  it("does not describe a failed selection as a ready animation", () => {
    expect(pipelineStageLabel("finalize", "no_acceptable_candidate")).toBe("No animation selected");
    expect(isTerminalWithoutWinner("no_acceptable_candidate")).toBe(true);
    expect(pipelineStageLabel("finalize", "completed")).toBe("Final animation ready");
  });

  it("labels structurally valid candidates excluded from the judging batch", () => {
    expect(candidateProgressBadge({
      compile_success: true,
      structural_valid: true,
      rejection_stage: "batch_diversity_selection",
    }, false)).toBe("Not selected");
  });

  it("keeps selected and structurally rejected states distinct", () => {
    expect(candidateProgressBadge({ structural_valid: true }, true)).toBe("Winner");
    expect(candidateProgressBadge({ structural_valid: false }, false)).toBe("Filtered");
  });
});
