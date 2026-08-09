import type { MotionProgram } from "./types";

function programsInSequence(program: MotionProgram | null | undefined): MotionProgram[] {
  if (!program) return [];
  return [program, ...(program.steps ?? []).flatMap((step) => programsInSequence(step))];
}

export function activeObstacleObjectId(program: MotionProgram | null | undefined): string | null {
  const obstacle = programsInSequence(program)
    .flatMap((item) => item.primitives ?? [])
    .find(
      (primitive) => primitive.body?.obstacle_mode && primitive.body.obstacle_mode !== "none",
    )?.body;
  return obstacle?.obstacle_object_id?.trim() || null;
}

export function activeTaskObjectId(program: MotionProgram | null | undefined): string | null {
  const obstacleObjectId = activeObstacleObjectId(program);
  if (obstacleObjectId) return obstacleObjectId;
  for (const item of programsInSequence(program)) {
    const supportObjectId = item.primitives
      ?.find((primitive) => primitive.body?.support_object_id)
      ?.body?.support_object_id;
    if (supportObjectId?.trim()) return supportObjectId.trim();
    const directId = item.objectId ?? item.object_id;
    if (typeof directId === "string" && directId.trim()) return directId.trim();
    const primitiveId = item.primitives?.find((primitive) => primitive.object_id)?.object_id;
    if (primitiveId?.trim()) return primitiveId.trim();
  }
  return null;
}

export function shouldShowTaskEnvironment(program: MotionProgram | null | undefined): boolean {
  return programsInSequence(program).some(
    (item) =>
      item.intent === "grab"
      || item.intent === "object_interaction"
      || Boolean(item.object_action)
      || Boolean(item.object_motion)
      || activeTaskObjectId(item) !== null,
  );
}

export function shouldShowTaskSupportSurface(program: MotionProgram | null | undefined): boolean {
  const taskObjectId = activeTaskObjectId(program);
  return shouldShowTaskEnvironment(program)
    && activeObstacleObjectId(program) === null
    && taskObjectId !== "ladder";
}
