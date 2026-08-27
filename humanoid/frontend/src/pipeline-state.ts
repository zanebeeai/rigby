import type { PipelineRun } from "./types";

export function isTerminalWithoutWinner(status: PipelineRun["status"]): boolean {
  return ["failed", "unsupported", "no_acceptable_candidate"].includes(status);
}

export function pipelineStageLabel(stage: string, status?: PipelineRun["status"]): string {
  if (stage === "finalize") {
    if (status === "no_acceptable_candidate") return "No animation selected";
    if (status === "unsupported") return "Motion not supported";
    if (status === "failed") return "Run stopped";
    return "Final animation ready";
  }
  const labels: Record<string, string> = {
    queued: "Queued",
    planning: "Understanding the request",
    candidates: "Building candidate pool",
    structural_checks: "Checking physical constraints",
    visual_evidence: "Capturing full-FOV evidence",
    vlm_judge: "Visual judge comparing candidates",
    repair: "Applying a bounded repair",
    failed: "Run stopped",
  };
  return labels[stage] ?? stage.replaceAll("_", " ");
}

export function candidateProgressBadge(
  value: Record<string, unknown>,
  selected: boolean,
): string {
  const judgment = value.judgment;
  const judged = judgment !== null && typeof judgment === "object" && Object.keys(judgment).length > 0;
  const structurallyValid = value.structural_valid !== false && value.compile_success !== false;
  if (selected) return "Winner";
  if (!structurallyValid) return "Filtered";
  if (value.rejection_stage) return "Not selected";
  if (judged) return (judgment as Record<string, unknown>).accept ? "Accepted" : "Not chosen";
  if (value.evidence_manifest) return "Captured";
  if (value.capture_started) return "Capturing";
  if (value.perceptually_rankable) return "Selected for judging";
  return "Compiling";
}
