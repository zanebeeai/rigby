import * as THREE from "three";
import { egoEyeOffsetM, egoNeutralGaze } from "./generated/camera";

// Rigby application space is glTF Y-up with the humanoid's anatomical front in +Z.
// Three.js cameras look down local -Z, but lookAt() rotates that local axis toward
// the world-space +Z gaze below.
const NEUTRAL_EYE_OFFSET = new THREE.Vector3(...egoEyeOffsetM);
const NEUTRAL_GAZE = new THREE.Vector3(...egoNeutralGaze);

export interface EgoCameraPose {
  position: THREE.Vector3;
  target: THREE.Vector3;
  forward: THREE.Vector3;
}

export interface OrbitCameraPose {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

/**
 * Follow horizontal root travel without cancelling vertical body motion.
 *
 * Tracking Y makes a crouch or jump remain at the same screen height, hiding
 * the very displacement the orbit view is meant to verify.  The ground is the
 * visual reference for full-body skills, so camera height and target height
 * stay fixed while X/Z follow locomotion.
 */
export function trackOrbitRoot(
  position: THREE.Vector3,
  target: THREE.Vector3,
  previousRoot: THREE.Vector3,
  nextRoot: THREE.Vector3,
): OrbitCameraPose {
  const delta = nextRoot.clone().sub(previousRoot);
  delta.y = 0;
  return {
    position: position.clone().add(delta),
    target: target.clone().add(delta),
  };
}

/**
 * Derive an egocentric camera from an animated head without inheriting the
 * source rig's rest-axis convention. Only the animation delta is applied to
 * Rigby's canonical +Z eye position and slightly downward gaze.
 */
export function computeEgoCameraPose(
  headWorldPosition: THREE.Vector3,
  headWorldRotation: THREE.Quaternion,
  headRestWorldRotation: THREE.Quaternion,
): EgoCameraPose {
  const headDelta = headWorldRotation.clone().multiply(headRestWorldRotation.clone().invert()).normalize();
  const position = headWorldPosition.clone().add(NEUTRAL_EYE_OFFSET.clone().applyQuaternion(headDelta));
  const gaze = NEUTRAL_GAZE.clone().applyQuaternion(headDelta);
  return {
    position,
    target: position.clone().add(gaze),
    forward: gaze.normalize(),
  };
}

export function applyEgoCameraPose(camera: THREE.PerspectiveCamera, pose: EgoCameraPose): void {
  camera.position.copy(pose.position);
  camera.up.set(0, 1, 0);
  camera.lookAt(pose.target);
  camera.updateMatrixWorld(true);
}

export function pointIsVisible(camera: THREE.PerspectiveCamera, point: THREE.Vector3): boolean {
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const viewProjection = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  return new THREE.Frustum().setFromProjectionMatrix(viewProjection).containsPoint(point);
}
