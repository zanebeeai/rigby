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
  | { kind: "segment"; from: number[]; to: number[]; radius: number }
  | { kind: "finger"; from: number[]; to: number[]; half: number[] }
  | { kind: "plate"; at: number[]; approach: number[]; across: number[] };

interface Frame {
  t: number;
  phase: number;
  links: Link[];
  block: number[];
  forces: Record<string, number>;
  opening_m: number;
}

interface Clip {
  fps: number;
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
camera.position.set(0.46, 0.98, 0.78);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.0, 0.80, 0.29);
controls.enableDamping = true;
controls.minDistance = 0.15;
controls.maxDistance = 3.0;
controls.update();

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(1.2, 2.2, 1.4);
scene.add(key);

const armMaterial = new THREE.MeshStandardMaterial({ color: 0x5a6b86, roughness: 0.55 });
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

  for (const link of first.links) {
    let mesh: THREE.Mesh;
    if (link.kind === "segment") {
      mesh = new THREE.Mesh(
        new THREE.BoxGeometry(link.radius * 2, link.radius * 2, 1),
        armMaterial,
      );
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

  const held = frame.forces["finger_left"] ?? 0;
  const other = frame.forces["finger_right"] ?? 0;
  timeLabel.textContent = `${frame.t.toFixed(2)} s`;
  detail.innerHTML =
    `<span class="phase">${clip.phase_names[frame.phase] ?? frame.phase}</span>` +
    ` &nbsp; opening <b>${(frame.opening_m * 100).toFixed(1)} cm</b>` +
    ` &nbsp; pads <b>${held.toFixed(1)}</b> / <b>${other.toFixed(1)} N</b>` +
    ` &nbsp; block height <b>${frame.block[1].toFixed(3)} m</b>`;
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
