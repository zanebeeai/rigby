import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { applyEgoCameraPose, computeEgoCameraPose, trackOrbitRoot } from "./camera";
import type {
  BlockParameters,
  CameraMode,
  ContactEvent,
  MotionFrame,
} from "./types";

const DEG = Math.PI / 180;
const BONE_MAP: Record<string, string> = {
  hips: "pelvis", spine: "spine_01", chest: "spine_02", upperChest: "spine_03", neck: "neck_01", head: "head",
  leftShoulder: "clavicle_l", leftUpperArm: "upperarm_l", leftLowerArm: "lowerarm_l", leftHand: "hand_l",
  rightShoulder: "clavicle_r", rightUpperArm: "upperarm_r", rightLowerArm: "lowerarm_r", rightHand: "hand_r",
  leftUpperLeg: "thigh_l", leftLowerLeg: "calf_l", leftFoot: "foot_l", leftToes: "ball_l",
  rightUpperLeg: "thigh_r", rightLowerLeg: "calf_r", rightFoot: "foot_r", rightToes: "ball_r",
  leftThumbMetacarpal: "thumb_01_l", leftThumbProximal: "thumb_02_l", leftThumbDistal: "thumb_03_l",
  leftIndexProximal: "index_01_l", leftIndexIntermediate: "index_02_l", leftIndexDistal: "index_03_l",
  leftMiddleProximal: "middle_01_l", leftMiddleIntermediate: "middle_02_l", leftMiddleDistal: "middle_03_l",
  leftRingProximal: "ring_01_l", leftRingIntermediate: "ring_02_l", leftRingDistal: "ring_03_l",
  leftLittleProximal: "pinky_01_l", leftLittleIntermediate: "pinky_02_l", leftLittleDistal: "pinky_03_l",
  rightThumbMetacarpal: "thumb_01_r", rightThumbProximal: "thumb_02_r", rightThumbDistal: "thumb_03_r",
  rightIndexProximal: "index_01_r", rightIndexIntermediate: "index_02_r", rightIndexDistal: "index_03_r",
  rightMiddleProximal: "middle_01_r", rightMiddleIntermediate: "middle_02_r", rightMiddleDistal: "middle_03_r",
  rightRingProximal: "ring_01_r", rightRingIntermediate: "ring_02_r", rightRingDistal: "ring_03_r",
  rightLittleProximal: "pinky_01_r", rightLittleIntermediate: "pinky_02_r", rightLittleDistal: "pinky_03_r",
};

function makeLimb(
  parent: THREE.Object3D,
  size: [number, number, number],
  position: [number, number, number],
  material: THREE.Material,
): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(size[0], size[1], 8, 12), material);
  mesh.scale.set(1, 1, size[2]);
  mesh.position.set(...position);
  mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}

function makeFallbackAvatar(): THREE.Group {
  const avatar = new THREE.Group();
  avatar.name = "RigbyFallbackAvatar";
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x96a2ac, roughness: 0.72 });
  const jointMaterial = new THREE.MeshStandardMaterial({ color: 0x27313a, roughness: 0.8 });

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.19, 0.38, 12, 20), bodyMaterial);
  torso.position.y = 1.25;
  torso.scale.z = 0.62;
  torso.castShadow = true;
  avatar.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.125, 24, 18), bodyMaterial);
  head.position.y = 1.68;
  head.name = "Head";
  head.castShadow = true;
  avatar.add(head);

  for (const side of [-1, 1]) {
    makeLimb(avatar, [0.065, 0.24, 0.85], [side * 0.27, 1.34, 0], bodyMaterial).rotation.z = side * 8 * DEG;
    makeLimb(avatar, [0.055, 0.25, 0.82], [side * 0.33, 1.06, -0.01], bodyMaterial).rotation.z = side * -6 * DEG;
    const hand = new THREE.Mesh(new THREE.BoxGeometry(0.105, 0.15, 0.045), jointMaterial);
    hand.position.set(side * 0.35, 0.86, -0.015);
    hand.castShadow = true;
    avatar.add(hand);
    makeLimb(avatar, [0.075, 0.36, 1], [side * 0.105, 0.65, 0], bodyMaterial);
    makeLimb(avatar, [0.065, 0.38, 0.95], [side * 0.105, 0.24, 0.01], bodyMaterial);
  }
  return avatar;
}

function poseFromFrame(frame: MotionFrame | null, objectId: string): NonNullable<MotionFrame["block"]> | null {
  if (!frame) return null;
  return frame.objects?.[objectId] ?? frame.objects?.block ?? frame.block ?? null;
}

export class RigbyScene {
  private readonly scene = new THREE.Scene();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly orbitCamera = new THREE.PerspectiveCamera(46, 1, 0.01, 40);
  private readonly egoCamera = new THREE.PerspectiveCamera(94, 1, 0.015, 40);
  private readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;
  private readonly readyPromise: Promise<void>;
  private readonly avatarRoot = new THREE.Group();
  private readonly fallbackAvatar = makeFallbackAvatar();
  private readonly block: THREE.Group;
  private readonly ladder: THREE.Group;
  private readonly table: THREE.Mesh;
  private readonly socketMarker: THREE.Mesh;
  private readonly contactMarkers = new THREE.Group();
  private readonly taskEnvironment = new THREE.Group();
  private boneLookup = new Map<string, THREE.Object3D>();
  private restRotations = new Map<string, THREE.Quaternion>();
  private restPositions = new Map<string, THREE.Vector3>();
  private headBone: THREE.Object3D | null = null;
  private headRestWorldRotation: THREE.Quaternion | null = null;
  private readonly trackedRootOffset = new THREE.Vector3();
  private mode: CameraMode = "orbit";
  private debugVisible = false;
  private blockParams: BlockParameters;
  private objectId = "block";

  constructor(
    private readonly host: HTMLElement,
    blockParams: BlockParameters,
  ) {
    this.blockParams = structuredClone(blockParams);
    this.scene.background = new THREE.Color(0x0b1016);
    this.scene.fog = new THREE.FogExp2(0x0b1016, 0.075);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.host.append(this.renderer.domElement);

    this.orbitCamera.position.set(2.15, 1.45, 2.65);
    this.controls = new OrbitControls(this.orbitCamera, this.renderer.domElement);
    this.controls.target.set(0, 1, 0.2);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI * 0.54;

    this.avatarRoot.add(this.fallbackAvatar);
    this.scene.add(this.avatarRoot);

    const blockMaterial = new THREE.MeshStandardMaterial({
      color: 0xefa868,
      roughness: 0.5,
      metalness: 0.08,
    });
    const blockBody = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), blockMaterial);
    blockBody.castShadow = true;
    blockBody.receiveShadow = true;
    const orientationStripe = new THREE.Mesh(
      new THREE.BoxGeometry(0.16, 0.018, 1.025),
      new THREE.MeshStandardMaterial({
        color: 0x284f70,
        roughness: 0.58,
        metalness: 0.04,
      }),
    );
    orientationStripe.position.y = 0.508;
    orientationStripe.castShadow = true;
    const orientationMarker = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.024, 0.22),
      new THREE.MeshStandardMaterial({
        color: 0x55dff5,
        emissive: 0x123b46,
        emissiveIntensity: 0.55,
        roughness: 0.54,
        metalness: 0.04,
      }),
    );
    orientationMarker.position.set(0.31, 0.512, 0.31);
    orientationMarker.castShadow = true;
    this.block = new THREE.Group();
    this.block.name = "RigbyOrientedTaskObject";
    this.block.add(blockBody, orientationStripe, orientationMarker);
    this.ladder = new THREE.Group();
    this.ladder.name = "RigbyLadderAffordance";
    this.taskEnvironment.add(this.block, this.ladder);

    this.socketMarker = new THREE.Mesh(
      new THREE.TorusGeometry(0.032, 0.005, 8, 24),
      new THREE.MeshBasicMaterial({ color: 0x5fe5c4, depthTest: false }),
    );
    this.socketMarker.rotation.x = Math.PI / 2;
    this.taskEnvironment.add(this.socketMarker, this.contactMarkers);
    this.setDebug(false);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(16, 16),
      new THREE.MeshStandardMaterial({ color: 0x111921, roughness: 0.92 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    this.table = new THREE.Mesh(
      new THREE.BoxGeometry(1.05, 0.055, 0.72),
      new THREE.MeshStandardMaterial({ color: 0x303a43, roughness: 0.82 }),
    );
    this.table.position.set(0, 0.985, 0.5);
    this.table.castShadow = true;
    this.table.receiveShadow = true;
    this.taskEnvironment.add(this.table);
    this.scene.add(this.taskEnvironment);

    const key = new THREE.DirectionalLight(0xffe2c5, 3.4);
    key.position.set(3.5, 5.2, 3.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x75b8ff, 2.4);
    rim.position.set(-3, 3.5, -4);
    this.scene.add(rim, new THREE.HemisphereLight(0xb9d9ff, 0x1a2229, 1.5));

    const grid = new THREE.GridHelper(16, 32, 0x2c4555, 0x182630);
    (grid.material as THREE.Material).opacity = 0.36;
    (grid.material as THREE.Material).transparent = true;
    this.scene.add(grid);

    this.updateBlock(blockParams);
    this.readyPromise = this.loadAvatar();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
    this.animate();
  }

  setCamera(mode: CameraMode): void {
    this.mode = mode;
    this.controls.enabled = mode === "orbit";
  }

  async prepareCapture(
    width: number,
    height: number,
    mode: CameraMode,
    frame: MotionFrame | null,
  ): Promise<void> {
    await this.readyPromise;
    this.mode = mode;
    this.controls.enabled = false;
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.orbitCamera.aspect = width / height;
    this.egoCamera.aspect = width / height;
    this.orbitCamera.updateProjectionMatrix();
    this.egoCamera.updateProjectionMatrix();
    this.applyFrame(frame);
    this.renderOnce();
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    this.renderOnce();
  }

  captureContract(): Record<string, number | string> {
    return {
      view: this.mode,
      width_px: this.renderer.domElement.width,
      height_px: this.renderer.domElement.height,
      aspect_ratio: this.egoCamera.aspect,
      vertical_fov_deg: this.mode === "ego" ? this.egoCamera.fov : this.orbitCamera.fov,
      device_pixel_ratio: 1,
    };
  }

  setEgoFieldOfView(degrees: number): void {
    this.egoCamera.fov = THREE.MathUtils.clamp(degrees, 40, 110);
    this.egoCamera.updateProjectionMatrix();
  }

  setOrbitFraming(position: [number, number, number], target: [number, number, number]): void {
    this.orbitCamera.position.fromArray(position);
    this.controls.target.fromArray(target);
    this.trackedRootOffset.set(0, 0, 0);
    this.controls.update();
  }

  setDebug(visible: boolean): void {
    this.debugVisible = visible;
    this.socketMarker.visible = visible;
    this.contactMarkers.visible = visible;
  }

  setTaskEnvironmentVisible(visible: boolean, supportSurfaceVisible = visible): void {
    this.taskEnvironment.visible = visible;
    this.table.visible = visible && supportSurfaceVisible;
  }

  setObjectId(objectId: string | null | undefined): void {
    this.objectId = objectId || "block";
  }

  updateBlock(params: BlockParameters): void {
    this.blockParams = structuredClone(params);
    this.block.scale.set(params.width, params.height, params.depth);
    this.block.position.set(...params.position);
    this.rebuildLadder(params);
    this.block.visible = params.kind !== "ladder";
    this.ladder.visible = params.kind === "ladder";
    this.socketMarker.position.set(
      params.position[0],
      params.position[1],
      params.position[2] - params.depth * 0.51,
    );
  }

  private rebuildLadder(params: BlockParameters): void {
    for (const child of [...this.ladder.children]) {
      this.ladder.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) material.dispose();
      }
    }
    const material = () => new THREE.MeshStandardMaterial({
      color: 0xc98b52,
      roughness: 0.62,
      metalness: 0.08,
    });
    const railWidth = Math.min(0.055, params.width * 0.12);
    for (const side of [-1, 1]) {
      const rail = new THREE.Mesh(
        new THREE.BoxGeometry(railWidth, params.height, Math.max(0.045, params.depth * 0.62)),
        material(),
      );
      rail.position.x = side * (params.width * 0.5 - railWidth * 0.65);
      rail.castShadow = true;
      rail.receiveShadow = true;
      this.ladder.add(rail);
    }
    const rungCount = Math.max(3, Math.min(16, Math.round(params.height / 0.205)));
    const rungSpan = Math.max(0.12, params.width - railWidth * 1.5);
    for (let index = 0; index < rungCount; index += 1) {
      const rung = new THREE.Mesh(
        new THREE.BoxGeometry(rungSpan, 0.035, Math.max(0.05, params.depth * 0.72)),
        material(),
      );
      rung.position.y = -params.height * 0.5 + 0.16 + index * (
        (params.height - 0.32) / Math.max(1, rungCount - 1)
      );
      rung.position.z = -params.depth * 0.18;
      rung.castShadow = true;
      rung.receiveShadow = true;
      this.ladder.add(rung);
    }
    this.ladder.position.set(...params.position);
  }

  applyFrame(frame: MotionFrame | null, contacts: ContactEvent[] = []): void {
    const nextRootOffset = new THREE.Vector3();
    for (const [canonical, restRotation] of this.restRotations) {
      const bone = this.boneLookup.get(canonical.toLowerCase());
      bone?.quaternion.copy(restRotation);
      const restPosition = this.restPositions.get(canonical);
      if (bone && restPosition) bone.position.copy(restPosition);
    }
    if (frame) {
      for (const [name, poseValue] of Object.entries(frame.bones ?? {})) {
        const bone = this.boneLookup.get(name.toLowerCase());
        if (!bone) continue;
        const pose = Array.isArray(poseValue) ? { rotation: poseValue } : poseValue;
        if (pose.rotation?.length === 4) {
          const delta = new THREE.Quaternion().fromArray(pose.rotation).normalize();
          const rest = this.restRotations.get(name) ?? this.restRotations.get(name.toLowerCase());
          bone.quaternion.copy(rest ?? new THREE.Quaternion()).multiply(delta).normalize();
        }
        if (pose.position?.length === 3) {
          const rest = this.restPositions.get(name) ?? this.restPositions.get(name.toLowerCase());
          const delta = new THREE.Vector3().fromArray(pose.position);
          if (name.toLowerCase() === "hips") nextRootOffset.copy(delta);
          // Clip root translation is authored in application/world axes
          // (Y-up, +Z forward). The source GLB root is rotated -90° about X,
          // so pelvis-local translation is [x, -z, y].
          if (name.toLowerCase() === "hips") delta.set(delta.x, -delta.z, delta.y);
          bone.position.copy(rest ?? new THREE.Vector3()).add(delta);
        }
      }
    }
    const trackedOrbit = trackOrbitRoot(
      this.orbitCamera.position,
      this.controls.target,
      this.trackedRootOffset,
      nextRootOffset,
    );
    this.orbitCamera.position.copy(trackedOrbit.position);
    this.controls.target.copy(trackedOrbit.target);
    this.trackedRootOffset.copy(nextRootOffset);
    const blockPose = poseFromFrame(frame, this.objectId);
    const activeObject = this.blockParams.kind === "ladder" ? this.ladder : this.block;
    this.block.visible = this.blockParams.kind !== "ladder";
    this.ladder.visible = this.blockParams.kind === "ladder";
    if (blockPose?.position) activeObject.position.fromArray(blockPose.position);
    else activeObject.position.set(...this.blockParams.position);
    if (blockPose?.rotation) activeObject.quaternion.fromArray(blockPose.rotation).normalize();
    else activeObject.quaternion.identity();
    if (activeObject === this.block) {
      if (blockPose?.scale) this.block.scale.fromArray(blockPose.scale);
      else this.block.scale.set(this.blockParams.width, this.blockParams.height, this.blockParams.depth);
    } else if (blockPose?.scale) {
      this.ladder.scale.fromArray(blockPose.scale);
    } else {
      this.ladder.scale.set(1, 1, 1);
    }

    this.socketMarker.position.copy(activeObject.position);
    this.socketMarker.position.z -= this.blockParams.depth * 0.51;
    if (this.debugVisible) this.renderContacts(contacts);
  }

  private renderContacts(contacts: ContactEvent[]): void {
    this.contactMarkers.clear();
    for (const contact of contacts.slice(0, 12)) {
      if (!contact.position) continue;
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 12, 8),
        new THREE.MeshBasicMaterial({ color: contact.active === false ? 0xf05f68 : 0x62e7c6, depthTest: false }),
      );
      marker.position.fromArray(contact.position);
      this.contactMarkers.add(marker);
      if (contact.normal) {
        const arrow = new THREE.ArrowHelper(
          new THREE.Vector3().fromArray(contact.normal).normalize(),
          marker.position,
          0.09,
          0x62e7c6,
        );
        this.contactMarkers.add(arrow);
      }
    }
  }

  private async loadAvatar(): Promise<void> {
    const loader = new GLTFLoader();
    try {
      const gltf = await loader.loadAsync("/assets/models/human-male.glb");
      const avatar = gltf.scene;
      avatar.name = "RigbyHumanoid";
      avatar.traverse((node) => {
        if (node.name) this.boneLookup.set(node.name.toLowerCase(), node);
        if (node instanceof THREE.Mesh) {
          node.castShadow = true;
          node.receiveShadow = true;
        }
      });
      for (const [canonical, source] of Object.entries(BONE_MAP)) {
        const bone = this.boneLookup.get(source.toLowerCase());
        if (!bone) continue;
        this.boneLookup.set(canonical.toLowerCase(), bone);
        this.restRotations.set(canonical, bone.quaternion.clone());
        this.restRotations.set(canonical.toLowerCase(), bone.quaternion.clone());
        this.restPositions.set(canonical, bone.position.clone());
        this.restPositions.set(canonical.toLowerCase(), bone.position.clone());
      }
      this.headBone = this.boneLookup.get("head") ?? null;
      this.avatarRoot.remove(this.fallbackAvatar);
      this.avatarRoot.add(avatar);
      avatar.updateWorldMatrix(true, true);
      this.headRestWorldRotation = this.headBone?.getWorldQuaternion(new THREE.Quaternion()) ?? null;
    } catch {
      // The procedural avatar keeps the UI useful when the asset server is offline.
      this.fallbackAvatar.traverse((node) => {
        if (node.name) this.boneLookup.set(node.name.toLowerCase(), node);
      });
      this.headBone = this.boneLookup.get("head") ?? null;
      this.fallbackAvatar.updateWorldMatrix(true, true);
      this.headRestWorldRotation = this.headBone?.getWorldQuaternion(new THREE.Quaternion()) ?? null;
    }
  }

  private resize(): void {
    const width = Math.max(1, this.host.clientWidth);
    const height = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(width, height, false);
    this.orbitCamera.aspect = width / height;
    this.egoCamera.aspect = width / height;
    this.orbitCamera.updateProjectionMatrix();
    this.egoCamera.updateProjectionMatrix();
  }

  private renderOnce(): void {
    if (this.mode === "ego") {
      if (this.headBone && this.headRestWorldRotation) {
        this.headBone.updateWorldMatrix(true, false);
        const pose = computeEgoCameraPose(
          this.headBone.getWorldPosition(new THREE.Vector3()),
          this.headBone.getWorldQuaternion(new THREE.Quaternion()),
          this.headRestWorldRotation,
        );
        applyEgoCameraPose(this.egoCamera, pose);
      } else {
        this.egoCamera.position.set(0, 1.66, 0.08);
        this.egoCamera.lookAt(0, 1.25, 1.2);
      }
    } else {
      this.controls.update();
    }
    this.renderer.render(this.scene, this.mode === "ego" ? this.egoCamera : this.orbitCamera);
  }

  private readonly animate = (): void => {
    requestAnimationFrame(this.animate);
    this.renderOnce();
  };
}
