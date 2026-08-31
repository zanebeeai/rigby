import "./gripper-live.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type RunStatus = "queued" | "running" | "completed" | "failed" | "aborted" | "interrupted";
type Vec3 = [number, number, number];
type Vec2 = [number, number];
type Quat = [number, number, number, number];
interface RunEvent { at: string; kind: string; message: string; detail?: unknown }
interface GripperRun {
  id: string; task: string; status: RunStatus; stage: string; message: string;
  started_at: string; updated_at: string; finished_at?: string | null;
  progress: number; elapsed_s: number; model_calls: number; can_abort: boolean;
  clip_url?: string | null; events: RunEvent[]; latest?: Record<string, unknown> | null;
  achieved?: Record<string, unknown> | null; error?: string | null;
}
type Link =
  | { kind: "segment"; from: Vec3; to: Vec3; radius: number; simulated: boolean }
  | { kind: "finger"; from: Vec3; to: Vec3; half: Vec2; simulated: boolean }
  | { kind: "plate"; at: Vec3; approach: Vec3; across: Vec3; simulated: boolean };
interface Frame {
  t: number; phase: number; links: Link[]; block: Vec3; block_quat: Quat;
  forces: Record<string, number>; opening_m: number; over_target_m: number;
  above_rim_m: number; object_seen?: boolean; holding?: boolean; in_target: boolean;
  penetration_mm: number; target?: string | null; target_value?: number | null;
  part?: string; move?: string; error?: number;
}
interface Clip {
  fps: number; task?: string; table_top_m: number; block_half_m: Vec3;
  phase_names: string[]; achieved: Record<string, unknown>; transcript?: unknown[];
  frames: Frame[]; pedestal?: { from: Vec3; to: Vec3; radius_m: number };
  bin?: { centre: Vec3; inner_half_m: Vec3; wall_m: number; riser_from: Vec3; rim_height_m: number };
}

function element<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (!value) throw new Error(`Missing ${selector}`);
  return value;
}
const ui = {
  prompt: element<HTMLTextAreaElement>("#prompt"), promptCount: element("#prompt-count"),
  run: element<HTMLButtonElement>("#run-button"), abort: element<HTMLButtonElement>("#abort-button"),
  connection: element("#connection"), runTitle: element("#run-title"), runBadge: element("#run-badge"),
  runMessage: element("#run-message"), progress: element<HTMLElement>("#progress-bar"),
  elapsed: element("#elapsed"), calls: element("#calls"), target: element("#target"), move: element("#move"),
  events: element<HTMLOListElement>("#events"), eventCount: element("#event-count"),
  history: element("#history"), historyList: element("#history-list"),
  historyToggle: element<HTMLButtonElement>("#history-toggle"), historyClose: element<HTMLButtonElement>("#history-close"),
  canvas: element<HTMLCanvasElement>("#view"), scrub: element<HTMLInputElement>("#scrub"),
  play: element<HTMLButtonElement>("#play"), follow: element<HTMLInputElement>("#follow"),
  time: element("#time"), liveDot: element("#live-dot"), stageLabel: element("#stage-label"),
  frameLabel: element("#frame-label"), measurements: element("#measurements"),
  outcome: element("#outcome"), outcomeDetail: element("#outcome-detail"), raw: element("#raw-state"),
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) }, ...init });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = String((await response.json()).detail ?? message); } catch { /* response was not JSON */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

// -- Physical replay -------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0f14);
scene.fog = new THREE.Fog(0x0a0f14, 2.0, 4.2);
const camera = new THREE.PerspectiveCamera(35, 1, 0.02, 20);
camera.position.set(0.78, 1.32, 1.0);
const controls = new OrbitControls(camera, ui.canvas);
controls.target.set(0.16, 0.9, 0.16); controls.enableDamping = true; controls.minDistance = .15; controls.maxDistance = 3;
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 1.15));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(1.2, 2.2, 1.4); key.castShadow = true; scene.add(key);
const world = new THREE.Group(); scene.add(world);
const armMaterial = new THREE.MeshStandardMaterial({ color: 0x69768c, roughness: .5, metalness: .12 });
const fingerMaterial = new THREE.MeshStandardMaterial({ color: 0x59c9ad, roughness: .38 });
const plateMaterial = new THREE.MeshStandardMaterial({ color: 0x318875, roughness: .5 });
const blockMaterial = new THREE.MeshStandardMaterial({ color: 0xe39b5a, roughness: .68 });
let parts: THREE.Mesh[] = []; let block: THREE.Mesh | null = null; let clip: Clip | null = null;
let cursor = 0; let playing = false; let lastTick = performance.now();
const vec = (values: Vec3) => new THREE.Vector3(values[0], values[1], values[2]);
function place(mesh: THREE.Mesh, from: THREE.Vector3, to: THREE.Vector3) {
  const span = to.clone().sub(from); const length = Math.max(span.length(), 1e-5);
  mesh.position.copy(from.clone().add(to).multiplyScalar(.5)); mesh.scale.set(1, 1, length);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), span.normalize());
}
function clearWorld() { world.clear(); parts = []; block = null; }
function clearReplay() {
  clearWorld(); clip = null; cursor = 0; playing = false; clipStamp = "";
  ui.play.textContent = "▶"; ui.scrub.value = "0"; ui.scrub.max = "0";
  ui.time.textContent = "0:00.00"; ui.frameLabel.textContent = "No clip loaded";
  const empty = document.createElement("p"); empty.className = "empty";
  empty.textContent = "Measurements appear when a clip starts.";
  ui.measurements.replaceChildren(empty);
}
function build(data: Clip) {
  if (!data.frames.length) return;
  clearWorld();
  const table = new THREE.Mesh(new THREE.BoxGeometry(1.8, .02, 1.8), new THREE.MeshStandardMaterial({ color: 0x293039, roughness: .95 }));
  table.position.set(0, data.table_top_m - .01, .25); table.receiveShadow = true; world.add(table);
  if (data.pedestal) {
    const from = vec(data.pedestal.from), to = vec(data.pedestal.to);
    const post = new THREE.Mesh(new THREE.CylinderGeometry(data.pedestal.radius_m, data.pedestal.radius_m * 1.2, Math.max(to.y - from.y, .01), 20), new THREE.MeshStandardMaterial({ color: 0x4a5262, roughness: .7 }));
    post.position.set(from.x, (from.y + to.y) / 2, from.z); world.add(post);
  }
  for (const link of data.frames[0]!.links) {
    let mesh: THREE.Mesh;
    if (link.kind === "segment") { const geometry = new THREE.CylinderGeometry(link.radius, link.radius, 1, 16); geometry.rotateX(Math.PI / 2); mesh = new THREE.Mesh(geometry, armMaterial); }
    else if (link.kind === "finger") mesh = new THREE.Mesh(new THREE.BoxGeometry(link.half[0] * 2, link.half[1] * 2, 1), fingerMaterial);
    else mesh = new THREE.Mesh(new THREE.BoxGeometry(.1, .08, .03), plateMaterial);
    mesh.castShadow = true; parts.push(mesh); world.add(mesh);
  }
  if (data.bin) {
    const c = data.bin.centre, inner = data.bin.inner_half_m, wall = data.bin.wall_m;
    const material = new THREE.MeshStandardMaterial({ color: 0x667486, roughness: .76 });
    const floor = new THREE.Mesh(new THREE.BoxGeometry((inner[0] + wall) * 2, wall * 2, (inner[2] + wall) * 2), material); floor.position.set(c[0], c[1] - inner[1] - wall, c[2]); world.add(floor);
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as Array<[number, number]>) {
      const panel = new THREE.Mesh(new THREE.BoxGeometry(dx ? wall * 2 : (inner[0] + wall * 2) * 2, inner[1] * 2, dx ? (inner[2] + wall * 2) * 2 : wall * 2), material);
      panel.position.set(c[0] + dx * (inner[0] + wall), c[1], c[2] + dz * (inner[2] + wall)); world.add(panel);
    }
  }
  const half = data.block_half_m; block = new THREE.Mesh(new THREE.BoxGeometry(half[0] * 2, half[1] * 2, half[2] * 2), blockMaterial); block.castShadow = true; world.add(block);
}
function measurement(label: string, value: string) { const row = document.createElement("div"); row.className = "measurement"; const a = document.createElement("span"); a.textContent = label; const b = document.createElement("strong"); b.textContent = value; row.append(a, b); return row; }
function show(index: number) {
  if (!clip || !block || !clip.frames.length) return;
  cursor = Math.max(0, Math.min(index, clip.frames.length - 1)); const frame = clip.frames[cursor]!;
  frame.links.forEach((link, i) => { const mesh = parts[i]; if (!mesh) return; if (link.kind === "plate") { mesh.position.copy(vec(link.at)); mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), vec(link.approach).normalize()); } else place(mesh, vec(link.from), vec(link.to)); });
  block.position.copy(vec(frame.block)); const q = frame.block_quat; if (q?.length === 4) block.quaternion.set(q[1], q[2], q[3], q[0]);
  ui.scrub.value = String(cursor); ui.time.textContent = formatSeconds(frame.t); ui.frameLabel.textContent = `Frame ${String(cursor + 1).padStart(3, "0")} · ${frame.t.toFixed(2)} s`;
  ui.measurements.replaceChildren(
    measurement("Current target", frame.target ? `${frame.target} → ${frame.target_value ?? ""}` : "waiting"),
    measurement("Chosen move", `${frame.part ?? "—"} / ${frame.move ?? "—"}`),
    measurement("Goal error", frame.error == null ? "—" : frame.error.toFixed(4)),
    measurement("Object visible", frame.object_seen ? "yes" : "no"),
    measurement("Holding", frame.holding ? "yes" : "no"),
    measurement("Jaw opening", `${(frame.opening_m * 100).toFixed(2)} cm`),
    measurement("Pad force", `${(frame.forces.finger_left ?? 0).toFixed(1)} / ${(frame.forces.finger_right ?? 0).toFixed(1)} N`),
    measurement("Penetration", `${frame.penetration_mm.toFixed(3)} mm`),
    measurement("Over target", `${(frame.over_target_m * 100).toFixed(1)} cm`),
  );
}
function loadClip(data: Clip, followEnd: boolean) {
  const rebuild = !clip || !block || clip.frames[0]?.links.length !== data.frames[0]?.links.length;
  clip = data; if (rebuild) build(data); ui.scrub.max = String(Math.max(0, data.frames.length - 1));
  if (followEnd || cursor >= data.frames.length) show(data.frames.length - 1); else show(cursor);
}
function formatSeconds(value: number) { const minutes = Math.floor(value / 60); return `${minutes}:${(value - minutes * 60).toFixed(2).padStart(5, "0")}`; }
function resize() { const width = ui.canvas.clientWidth, height = ui.canvas.clientHeight; if (ui.canvas.width !== width || ui.canvas.height !== height) { renderer.setSize(width, height, false); camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix(); } }
function animate(now: number) { resize(); if (playing && clip?.frames.length && now - lastTick >= 1000 / clip.fps) { lastTick = now; const next = cursor + 1; if (next >= clip.frames.length) { playing = false; ui.play.textContent = "▶"; } else show(next); } controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); }

// -- Persistent run UI -----------------------------------------------------
let selected: GripperRun | null = null; let runs: GripperRun[] = []; let clipStamp = ""; let polling = false;
function setConnection(ok: boolean) { ui.connection.className = `connection ${ok ? "connected" : "offline"}`; ui.connection.querySelector("span")!.textContent = ok ? "API connected" : "API unavailable"; }
function renderEvents(events: RunEvent[]) {
  ui.eventCount.textContent = `${events.length} event${events.length === 1 ? "" : "s"}`; ui.events.replaceChildren();
  if (!events.length) { const empty = document.createElement("li"); empty.className = "empty"; empty.textContent = "Model decisions will appear here while it works."; ui.events.append(empty); return; }
  for (const event of [...events].reverse()) { const item = document.createElement("li"), strong = document.createElement("strong"), time = document.createElement("span"); strong.textContent = event.message; time.textContent = `${event.kind.replaceAll("_", " ")} · ${new Date(event.at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })}`; item.append(strong, time); ui.events.append(item); }
}
function renderRun(run: GripperRun | null) {
  if (selected?.id !== run?.id) clearReplay();
  selected = run; ui.raw.textContent = JSON.stringify(run ?? {}, null, 2);
  if (!run) return;
  ui.runTitle.textContent = run.task; ui.runBadge.textContent = run.status; ui.runBadge.className = `status-badge ${run.status}`;
  ui.runMessage.textContent = run.error || run.message; ui.progress.style.width = `${Math.max(0, Math.min(1, run.progress || 0)) * 100}%`;
  ui.elapsed.textContent = `${Number(run.elapsed_s || 0).toFixed(1)} s`; ui.calls.textContent = String(run.model_calls ?? 0);
  ui.target.textContent = String(run.latest?.target ?? "—"); ui.move.textContent = run.latest ? `${String(run.latest.part ?? "—")} / ${String(run.latest.move ?? "—")}` : "—";
  ui.abort.hidden = !run.can_abort; ui.run.disabled = run.can_abort; renderEvents(run.events ?? []);
  const live = run.status === "running" || run.status === "queued"; ui.liveDot.className = `live-dot ${live ? "live" : run.status === "failed" ? "failed" : "idle"}`; ui.stageLabel.textContent = live ? "LIVE" : "REPLAY";
  const achieved = run.achieved; if (run.status === "completed") { const placed = achieved?.in_target === true; ui.outcome.textContent = placed ? "Placed in target" : "Task not completed"; ui.outcomeDetail.textContent = `Peak lift ${(Number(achieved?.peak_lift_m ?? 0) * 100).toFixed(1)} cm · deepest penetration ${Number(achieved?.deepest_penetration_mm ?? 0).toFixed(3)} mm.`; }
  else if (run.status === "aborted") { ui.outcome.textContent = "Aborted"; ui.outcomeDetail.textContent = "The isolated simulation process was terminated. The frames produced before abort remain replayable."; }
  else if (run.status === "failed") { ui.outcome.textContent = "Run failed"; ui.outcomeDetail.textContent = run.error || "See the run trace for the failure."; }
  else { ui.outcome.textContent = "In progress"; ui.outcomeDetail.textContent = "Physical success is evaluated from the simulated object state when the run ends."; }
}
function renderHistory() {
  ui.historyList.replaceChildren(); if (!runs.length) { const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = "No gripper runs yet."; ui.historyList.append(empty); return; }
  for (const run of runs) { const button = document.createElement("button"); button.type = "button"; button.className = `history-run${selected?.id === run.id ? " selected" : ""}`; const top = document.createElement("div"), title = document.createElement("strong"), status = document.createElement("span"), meta = document.createElement("p"); title.textContent = run.task; status.textContent = run.status; meta.textContent = `${new Date(run.started_at).toLocaleString()} · ${run.model_calls ?? 0} model calls`; top.append(title, status); button.append(top, meta); button.addEventListener("click", () => { selectRun(run); openHistory(false); }); ui.historyList.append(button); }
}
async function refreshClip(run: GripperRun) {
  if (!run.clip_url || run.updated_at === clipStamp) return; try { const response = await fetch(`${run.clip_url}?v=${encodeURIComponent(run.updated_at)}`, { cache: "no-store" }); if (!response.ok) return; const data = await response.json() as Clip; if (data.frames?.length) { loadClip(data, ui.follow.checked && run.can_abort); clipStamp = run.updated_at; } } catch { /* a worker can replace the file between polls */ }
}
async function selectRun(run: GripperRun) { clipStamp = ""; renderRun(run); renderHistory(); history.replaceState(null, "", `?run=${encodeURIComponent(run.id)}`); await refreshClip(run); }
async function refresh() {
  if (polling) return; polling = true;
  try { const data = await api<{ runs: GripperRun[] }>("/api/v1/gripper-runs"); setConnection(true); runs = data.runs; const wanted = selected?.id ?? new URLSearchParams(location.search).get("run"); const current = (wanted && runs.find((run) => run.id === wanted)) || runs.find((run) => run.can_abort) || runs[0] || null; if (current) { renderRun(current); await refreshClip(current); } renderHistory(); }
  catch { setConnection(false); }
  finally { polling = false; }
}
async function startRun() {
  const task = ui.prompt.value.trim(); if (!task) { ui.prompt.focus(); return; }
  ui.run.disabled = true;
  try { const run = await api<GripperRun>("/api/v1/gripper-runs", { method: "POST", body: JSON.stringify({ task, seconds: 32, max_model_calls: 10 }) }); runs = [run, ...runs.filter((item) => item.id !== run.id)]; await selectRun(run); }
  catch (error) { ui.runMessage.textContent = error instanceof Error ? error.message : "Could not start run"; ui.run.disabled = false; }
}
async function abortRun() { if (!selected?.can_abort) return; ui.abort.disabled = true; try { const run = await api<GripperRun>(`/api/v1/gripper-runs/${encodeURIComponent(selected.id)}/abort`, { method: "POST", body: "{}" }); renderRun(run); await refresh(); } finally { ui.abort.disabled = false; } }
function openHistory(open: boolean) { ui.history.classList.toggle("open", open); ui.history.setAttribute("aria-hidden", String(!open)); ui.historyToggle.setAttribute("aria-expanded", String(open)); }

ui.prompt.addEventListener("input", () => { ui.promptCount.textContent = `${ui.prompt.value.length} / 400`; });
ui.prompt.dispatchEvent(new Event("input")); document.querySelectorAll<HTMLButtonElement>("[data-prompt]").forEach((button) => button.addEventListener("click", () => { ui.prompt.value = button.dataset.prompt ?? ""; ui.prompt.dispatchEvent(new Event("input")); }));
ui.run.addEventListener("click", startRun); ui.abort.addEventListener("click", abortRun); ui.prompt.addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); startRun(); } });
ui.historyToggle.addEventListener("click", () => openHistory(!ui.history.classList.contains("open"))); ui.historyClose.addEventListener("click", () => openHistory(false));
ui.scrub.addEventListener("input", () => { playing = false; ui.play.textContent = "▶"; ui.follow.checked = false; show(Number(ui.scrub.value)); }); ui.play.addEventListener("click", () => { playing = !playing; ui.follow.checked = false; ui.play.textContent = playing ? "❚❚" : "▶"; });
ui.follow.addEventListener("change", () => { if (ui.follow.checked && clip?.frames.length) show(clip.frames.length - 1); });

requestAnimationFrame(animate); refresh(); window.setInterval(refresh, 750);
