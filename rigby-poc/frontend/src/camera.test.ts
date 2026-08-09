import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { applyEgoCameraPose, computeEgoCameraPose, pointIsVisible, trackOrbitRoot } from "./camera";
import { DEFAULT_PARAMETERS } from "./types";

const HEAD_REST = new THREE.Vector3(0, 1.5685, 0.0114);

function cameraFor(pose: ReturnType<typeof computeEgoCameraPose>): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(94, 16 / 9, 0.015, 40);
  applyEgoCameraPose(camera, pose);
  return camera;
}

describe("egocentric camera convention", () => {
  it("faces humanoid-forward +Z regardless of the source head rest axis", () => {
    const sourceRest = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0.35, Math.PI));
    const pose = computeEgoCameraPose(HEAD_REST, sourceRest, sourceRest);
    const camera = cameraFor(pose);
    const renderedForward = camera.getWorldDirection(new THREE.Vector3());

    expect(pose.forward.z).toBeGreaterThan(0.8);
    expect(pose.forward.y).toBeLessThan(0);
    expect(renderedForward.dot(pose.forward)).toBeGreaterThan(0.9999);
  });

  it("applies animated head rotation as a delta from the calibrated rest pose", () => {
    const rest = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.2, -0.4, 0.1));
    const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
    const animated = yaw.clone().multiply(rest);
    const pose = computeEgoCameraPose(HEAD_REST, animated, rest);

    expect(pose.forward.x).toBeGreaterThan(0.8);
    expect(Math.abs(pose.forward.z)).toBeLessThan(0.1);
  });

  it("keeps the canonical object-in-front position inside the ego frustum", () => {
    const rest = new THREE.Quaternion();
    const camera = cameraFor(computeEgoCameraPose(HEAD_REST, rest, rest));
    const blockCenter = new THREE.Vector3(...DEFAULT_PARAMETERS.block.position);
    const behindAvatar = new THREE.Vector3(0, 1.2, -0.5);

    expect(DEFAULT_PARAMETERS.block.position[2]).toBeGreaterThan(HEAD_REST.z);
    expect(pointIsVisible(camera, blockCenter)).toBe(true);
    expect(pointIsVisible(camera, behindAvatar)).toBe(false);
  });
});

describe("orbit root tracking", () => {
  it("tracks horizontal root travel while preserving vertical motion", () => {
    const pose = trackOrbitRoot(
      new THREE.Vector3(1.6, 1.5, 1.9),
      new THREE.Vector3(0, 1.25, 0.25),
      new THREE.Vector3(0.2, 0.1, 0.4),
      new THREE.Vector3(-0.3, 0.45, 1.2),
    );

    expect(pose.position.x).toBeCloseTo(1.1);
    expect(pose.position.y).toBeCloseTo(1.5);
    expect(pose.position.z).toBeCloseTo(2.7);
    expect(pose.target.x).toBeCloseTo(-0.5);
    expect(pose.target.y).toBeCloseTo(1.25);
    expect(pose.target.z).toBeCloseTo(1.05);
    const framing = pose.position.clone().sub(pose.target);
    expect(framing.x).toBeCloseTo(1.6);
    expect(framing.y).toBeCloseTo(0.25);
    expect(framing.z).toBeCloseTo(1.65);
  });
});
