/**
 * Watch a gripper run.
 *
 * The humanoid viewer animates bone poses on a rigged mesh. This machine has no
 * skeleton and no mesh, so it gets its own page rather than being dressed up as
 * something it is not: each frame carries where its links actually are, and this
 * draws them.
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Link =
  | { kind: "segment"; from: number[]; to: number[]; radius: number; simulated: boolean }
  | { kind: "finger"; from: number[]; to: number[]; half: number[]; simulated: boolean }
  | { kind: "plate"; at: number[]; approach: number[]; across: number[]; simulated: boolean };

interface Frame {
  t: number;
  phase: number;
  links: Link[];
  block: number[];
  block_quat: number[];
  forces: Record<string, number>;
  opening_m: number;
  over_target_m: number;
  above_rim_m: number;
  in_target: boolean;
}

interface Clip {
  fps: number;
  pedestal?: { from: number[]; to: number[]; radius_m: number };
  bin?: {
    centre: number[]; inner_half_m: number[]; wall_m: number;
    riser_from: number[]; rim_height_m: number;
  };
  table_top_m: number;
  block_half_m: number[];
  phase_names: string[];
  achieved: Record<string, number>;
  frames: Frame[];
}

const canvas = document.getElementById("view") as HTMLCanvasElement;
const scrub = document.getElementById("scrub") as HTMLInputElement;
const playButton = document.getElementById("play") as HTMLButtonElement;
const timeLabel = document.getElementById("time") as HTMLElement;
const detail = document.getElementById("detail") as HTMLElement;
const summary = document.getElementById("summary") as HTMLElement;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x171a21);
const camera = new THREE.PerspectiveCamera(34, 1, 0.02, 20);
// Framed on the work, not the room: the whole point of watching this is
// the last few centimetres, and a wide shot of a table hides them.
camera.position.set(0.78, 1.32, 1.00);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.16, 0.90, 0.16);
controls.enableDamping = true;
controls.minDistance = 0.15;
controls.maxDistance = 3.0;
controls.update();

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(1.2, 2.2, 1.4);
scene.add(key);

// The arm is solid but visibly a different thing from the gripper: it is
// linkage, not collision geometry. Only the pads, the plate, the block and the
// table are in the physics, and the clip marks which is which.
const armMaterial = new THREE.MeshStandardMaterial({
  color: 0x6a768d, roughness: 0.5, metalness: 0.1,
});
const fingerMaterial = new THREE.MeshStandardMaterial({ color: 0x4fa3d1, roughness: 0.4 });
const plateMaterial = new THREE.MeshStandardMaterial({ color: 0x3d6f96, roughness: 0.5 });
const blockMaterial = new THREE.MeshStandardMaterial({ color: 0xd98c4a, roughness: 0.7 });

const root = new THREE.Group();
scene.add(root);
const parts: THREE.Mesh[] = [];
let block: THREE.Mesh | null = null;
let clip: Clip | null = null;
let playing = false;
let cursor = 0;

/** A box stretched between two points, which is every link this body has. */
function place(mesh: THREE.Mesh, from: THREE.Vector3, to: THREE.Vector3) {
  const middle = from.clone().add(to).multiplyScalar(0.5);
  const span = to.clone().sub(from);
  const length = Math.max(span.length(), 1e-5);
  mesh.position.copy(middle);
  mesh.scale.set(1, 1, length);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    span.normalize(),
  );
}

function vec(a: number[]): THREE.Vector3 {
  return new THREE.Vector3(a[0], a[1], a[2]);
}

function build(first: Frame, data: Clip) {
  const table = new THREE.Mesh(
    new THREE.BoxGeometry(1.8, 0.02, 1.8),
    new THREE.MeshStandardMaterial({ color: 0x2b3038, roughness: 0.95 }),
  );
  table.position.set(0, data.table_top_m - 0.01, 0.25);
  scene.add(table);

  // The plinth the arm stands on. Bolted flat to the table the arm could not
  // reach forward without swinging a link under the surface, so the mount is
  // part of the machine and worth seeing.
  if (data.pedestal) {
    const from = vec(data.pedestal.from);
    const to = vec(data.pedestal.to);
    const post = new THREE.Mesh(
      new THREE.CylinderGeometry(
        data.pedestal.radius_m, data.pedestal.radius_m * 1.25,
        Math.max(to.y - from.y, 0.01), 20),
      new THREE.MeshStandardMaterial({ color: 0x4a5262, roughness: 0.7 }),
    );
    post.position.set(from.x, (from.y + to.y) / 2, from.z);
    scene.add(post);
  }

  for (const link of first.links) {
    let mesh: THREE.Mesh;
    if (link.kind === "segment") {
      // A cylinder along its own length, which is what a linkage looks like.
      // Three.js cylinders stand along Y, so it is turned onto Z once here and
      // the stretch below then works in the same axis as every other link.
      const tube = new THREE.CylinderGeometry(link.radius, link.radius, 1, 16);
      tube.rotateX(Math.PI / 2);
      mesh = new THREE.Mesh(tube, armMaterial);
    } else if (link.kind === "finger") {
      mesh = new THREE.Mesh(
        new THREE.BoxGeometry(link.half[0] * 2, link.half[1] * 2, 1),
        fingerMaterial,
      );
    } else {
      mesh = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.08, 0.03), plateMaterial);
    }
    parts.push(mesh);
    root.add(mesh);
  }

  // The bin the block is going into: four walls, a floor and its riser. Static
  // scenery in the physics, so it is drawn once and never moved.
  if (data.bin) {
    const c = data.bin.centre;
    const inner = data.bin.inner_half_m;
    const wall = data.bin.wall_m;
    const binMaterial = new THREE.MeshStandardMaterial({
      color: 0x7b8598, roughness: 0.75,
    });
    const floor = new THREE.Mesh(
      new THREE.BoxGeometry((inner[0] + wall) * 2, wall * 2, (inner[2] + wall) * 2),
      binMaterial);
    floor.position.set(c[0], c[1] - inner[1] - wall, c[2]);
    scene.add(floor);
    const sides: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    for (const [dx, dz] of sides) {
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(
          dx ? wall * 2 : (inner[0] + wall * 2) * 2,
          inner[1] * 2,
          dx ? (inner[2] + wall * 2) * 2 : wall * 2),
        binMaterial);
      panel.position.set(
        c[0] + dx * (inner[0] + wall), c[1], c[2] + dz * (inner[2] + wall));
      scene.add(panel);
    }
    const riser = data.bin.riser_from;
    const post = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.055, c[1] - inner[1] - riser[1], 18),
      new THREE.MeshStandardMaterial({ color: 0x4a5262, roughness: 0.7 }));
    post.position.set(c[0], (riser[1] + c[1] - inner[1]) / 2, c[2]);
    scene.add(post);
  }

  const half = data.block_half_m;
  block = new THREE.Mesh(
    new THREE.BoxGeometry(half[0] * 2, half[1] * 2, half[2] * 2),
    blockMaterial,
  );
  scene.add(block);
}

function show(index: number) {
  if (!clip || !block) return;
  const frame = clip.frames[Math.max(0, Math.min(index, clip.frames.length - 1))];
  frame.links.forEach((link, i) => {
    const mesh = parts[i];
    if (!mesh) return;
    if (link.kind === "plate") {
      mesh.position.copy(vec(link.at));
      const forward = vec(link.approach).normalize();
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), forward);
    } else {
      place(mesh, vec(link.from), vec(link.to));
    }
  });
  block.position.copy(vec(frame.block));
  // Orientation as well as place. Without it a carried box slides around the
  // scene perfectly upright however the gripper turns, which reads as physics
  // that ignores rotation and is really a clip that never recorded it.
  const q = frame.block_quat;
  if (q && q.length === 4) {
    block.quaternion.set(q[1], q[2], q[3], q[0]);
  }

  const held = frame.forces["finger_left"] ?? 0;
  const other = frame.forces["finger_right"] ?? 0;
  timeLabel.textContent = `${frame.t.toFixed(2)} s`;
  detail.innerHTML =
    `<span class="phase">${clip.phase_names[frame.phase] ?? frame.phase}</span>` +
    ` &nbsp; opening <b>${(frame.opening_m * 100).toFixed(1)} cm</b>` +
    ` &nbsp; pads <b>${held.toFixed(1)}</b> / <b>${other.toFixed(1)} N</b>` +
    ` &nbsp; block height <b>${frame.block[1].toFixed(3)} m</b>` +
    ` &nbsp; over bin <b>${(frame.over_target_m * 100).toFixed(1)} cm</b>` +
    ` &nbsp; rim <b>${(frame.above_rim_m * 100).toFixed(1)} cm</b>` +
    (frame.in_target ? ` &nbsp; <span class="phase">IN THE BIN</span>` : "");
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }
}

let last = performance.now();
function loop(now: number) {
  resize();
  if (playing && clip) {
    if (now - last > 1000 / clip.fps) {
      last = now;
      cursor = (cursor + 1) % clip.frames.length;
      scrub.value = String(cursor);
      show(cursor);
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

scrub.addEventListener("input", () => {
  playing = false;
  playButton.textContent = "▶ Play";
  cursor = Number(scrub.value);
  show(cursor);
});
playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "❚❚ Pause" : "▶ Play";
});

fetch(`${import.meta.env.BASE_URL}gripper-run.json`)
  .then((response) => response.json())
  .then((data: Clip) => {
    clip = data;
    scrub.max = String(data.frames.length - 1);
    build(data.frames[0], data);
    const lift = (data.achieved.final_lift_m ?? 0) * 100;
    summary.textContent =
      `${data.frames.length} frames · lift held ${lift.toFixed(2)} cm`;
    show(0);
  })
  .catch(() => {
    summary.textContent = "no gripper run exported yet";
  });

requestAnimationFrame(loop);
