import type { MotionProgram, SceneManifest, SupportSurfaceParameters } from "./types";

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

/**
 * The support surface to draw for this program, or null to draw none.
 *
 * The geometry comes from the manifest's `table` object -- the same body the
 * compiler plans against -- so the viewer never carries its own copy of it.
 * A scene without a table draws no table, whatever the program wants.
 */
export function supportSurfaceFor(
  program: MotionProgram | null | undefined,
  scene: Pick<SceneManifest, "objects"> | null | undefined,
): SupportSurfaceParameters | null {
  if (!shouldShowTaskSupportSurface(program)) return null;
  const table = scene?.objects?.find((item) => item.kind === "table");
  if (!table) return null;
  const { translation } = table.transform;
  return {
    width: table.dimensions_m.x,
    height: table.dimensions_m.y,
    depth: table.dimensions_m.z,
    position: [translation.x, translation.y, translation.z],
  };
}
