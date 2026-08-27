import "./comparison.css";
import { downloadBlob, frameAt, unwrapClip } from "./motion";
import { RigbyScene } from "./scene";
import { DEFAULT_PARAMETERS, type ClipResult } from "./types";

type Choice = "A" | "B" | "tie";
interface ComparisonRecord {
  clip_id: string;
  prompt: string;
  candidates: Record<"A" | "B", { result_url: string }>;
}
interface ComparisonManifest {
  schema_version: string;
  kind: string;
  manifest_sha256: string;
  instructions: string;
  records: ComparisonRecord[];
}
interface Draft { choice: Choice | null; note: string; }

const mount = document.querySelector<HTMLElement>("#comparison-app");
if (!mount) throw new Error("Comparison mount not found");
mount.innerHTML = `
  <div class="comparison-shell">
    <header class="comparison-header">
      <div class="comparison-brand"><i><span></span><span></span><span></span></i><div><strong>Rigby</strong><small>Blinded motion comparison</small></div></div>
      <div class="comparison-progress"><p id="progress-text">Loading comparison…</p><div><i id="progress-bar"></i></div></div>
      <button id="download-comparison" class="comparison-button primary" disabled>Complete all comparisons</button>
    </header>
    <main class="comparison-main">
      <section class="comparison-workspace">
        <div class="prompt-banner"><span>Prompt</span><p id="comparison-prompt">Loading…</p><strong id="comparison-count">—</strong></div>
        <div class="candidate-grid">
          <article class="candidate"><header><strong>Candidate A</strong><span>Synchronized views</span></header><div class="candidate-views"><div class="comparison-view"><header>Egocentric · full FOV</header><div id="a-ego" class="comparison-canvas"></div></div><div class="comparison-view"><header>Orbit</header><div id="a-orbit" class="comparison-canvas"></div></div></div></article>
          <article class="candidate"><header><strong>Candidate B</strong><span>Synchronized views</span></header><div class="candidate-views"><div class="comparison-view"><header>Egocentric · full FOV</header><div id="b-ego" class="comparison-canvas"></div></div><div class="comparison-view"><header>Orbit</header><div id="b-orbit" class="comparison-canvas"></div></div></div></article>
        </div>
        <div class="playback"><button id="comparison-play" aria-label="Pause">❚❚</button><time id="current-time">0:00.00</time><input id="comparison-timeline" type="range" min="0" max="1" step="0.001" value="0" aria-label="Synchronized comparison timeline"/><time id="total-time">0:00.00</time><label><input id="comparison-loop" type="checkbox" checked/> Loop</label></div>
      </section>
      <aside class="preference-panel">
        <p class="eyebrow">Independent human preference</p><h1>Which motion is better?</h1>
        <p class="guidance">Use the shown prompt. Prefer recognizability, natural arm and wrist motion, timing, and full first-person visibility.</p>
        <div class="choice-stack"><button data-choice="A"><strong>Candidate A</strong><small>A is clearly better</small></button><button data-choice="B"><strong>Candidate B</strong><small>B is clearly better</small></button><button data-choice="tie"><strong>Tie</strong><small>No meaningful preference</small></button></div>
        <label class="note-field"><span>Optional note</span><textarea id="comparison-note" maxlength="400" placeholder="What decided your preference?"></textarea></label>
        <p id="comparison-warning" class="warning" hidden>Choose A, B, or Tie before continuing.</p>
        <div class="navigation"><button id="comparison-previous" class="comparison-button">Previous</button><button id="comparison-next" class="comparison-button primary">Save & next</button></div>
        <div id="comparison-dots" class="clip-dots"></div>
      </aside>
    </main>
    <div id="comparison-loading" class="loading-cover visible">Loading synchronized candidates…</div>
    <div id="comparison-error" class="error-box" hidden></div>
  </div>`;

function element<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (!value) throw new Error(`Missing comparison element: ${selector}`);
  return value;
}

const ui = {
  progressText: element<HTMLElement>("#progress-text"), progressBar: element<HTMLElement>("#progress-bar"),
  download: element<HTMLButtonElement>("#download-comparison"), prompt: element<HTMLElement>("#comparison-prompt"),
  count: element<HTMLElement>("#comparison-count"), play: element<HTMLButtonElement>("#comparison-play"),
  timeline: element<HTMLInputElement>("#comparison-timeline"), current: element<HTMLElement>("#current-time"),
  total: element<HTMLElement>("#total-time"), loop: element<HTMLInputElement>("#comparison-loop"),
  note: element<HTMLTextAreaElement>("#comparison-note"), warning: element<HTMLElement>("#comparison-warning"),
  previous: element<HTMLButtonElement>("#comparison-previous"), next: element<HTMLButtonElement>("#comparison-next"),
  dots: element<HTMLElement>("#comparison-dots"), loading: element<HTMLElement>("#comparison-loading"),
  error: element<HTMLElement>("#comparison-error"),
};

const scenes = {
  aEgo: new RigbyScene(element("#a-ego"), DEFAULT_PARAMETERS.block),
  aOrbit: new RigbyScene(element("#a-orbit"), DEFAULT_PARAMETERS.block),
  bEgo: new RigbyScene(element("#b-ego"), DEFAULT_PARAMETERS.block),
  bOrbit: new RigbyScene(element("#b-orbit"), DEFAULT_PARAMETERS.block),
};
for (const scene of [scenes.aEgo, scenes.bEgo]) { scene.setCamera("ego"); scene.setEgoFieldOfView(94); }
for (const scene of [scenes.aOrbit, scenes.bOrbit]) { scene.setCamera("orbit"); scene.setOrbitFraming([1.6, 1.5, 1.9], [0, 1.25, 0.25]); }

const manifestUrl = new URLSearchParams(location.search).get("manifest") ?? "/results/heldout-flywheel/complete/comparison-manifest.json";
let manifest: ComparisonManifest | null = null;
let drafts: Record<string, Draft> = {};
let index = 0;
let clips: Record<"A" | "B", ClipResult | null> = { A: null, B: null };
let currentTime = 0;
let playing = true;
let playStarted = performance.now();
let playFrom = 0;
let loadRevision = 0;

function formatTime(value: number): string { return `${Math.floor(value / 60)}:${(value % 60).toFixed(2).padStart(5, "0")}`; }
function storageKey(): string { return `rigby-preference-v1:${manifest?.manifest_sha256 ?? manifestUrl}`; }
function duration(): number { return Math.max(clips.A?.duration ?? 0, clips.B?.duration ?? 0); }
function currentRecord(): ComparisonRecord | undefined { return manifest?.records[index]; }
function completedCount(): number { return Object.values(drafts).filter((draft) => draft.choice !== null).length; }

function saveDraft(): void { if (manifest) localStorage.setItem(storageKey(), JSON.stringify(drafts)); renderProgress(); }
function restoreDrafts(): void {
  if (!manifest) return;
  drafts = Object.fromEntries(manifest.records.map((record) => [record.clip_id, { choice: null, note: "" }]));
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey()) ?? "{}") as Record<string, Draft>;
    for (const id of Object.keys(drafts)) {
      const savedDraft = saved[id];
      if (savedDraft && ["A", "B", "tie"].includes(String(savedDraft.choice))) {
        drafts[id] = { choice: savedDraft.choice, note: String(savedDraft.note ?? "").slice(0, 400) };
      }
    }
  } catch { /* Ignore malformed local drafts. */ }
}

function renderProgress(): void {
  if (!manifest) return;
  const done = completedCount(); const total = manifest.records.length;
  ui.progressText.textContent = `${done} of ${total} comparisons complete`;
  ui.progressBar.style.width = `${(done / total) * 100}%`;
  ui.download.disabled = done !== total;
  ui.download.textContent = done === total ? "Download completed review" : `Complete ${total - done} more`;
  ui.dots.replaceChildren();
  manifest.records.forEach((record, dotIndex) => {
    const button = document.createElement("button"); button.textContent = String(dotIndex + 1);
    button.className = `${drafts[record.clip_id]?.choice ? "complete" : ""} ${dotIndex === index ? "active" : ""}`;
    button.addEventListener("click", () => void selectRecord(dotIndex)); ui.dots.append(button);
  });
}

function renderChoice(): void {
  const draft = currentRecord() ? drafts[currentRecord()!.clip_id] : undefined;
  document.querySelectorAll<HTMLButtonElement>("[data-choice]").forEach((button) => button.classList.toggle("selected", button.dataset.choice === draft?.choice));
  ui.note.value = draft?.note ?? ""; ui.warning.hidden = true;
}

async function fetchClip(url: string): Promise<ClipResult> {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Candidate could not be loaded (${response.status}).`);
  return unwrapClip(await response.json());
}

async function selectRecord(nextIndex: number): Promise<void> {
  if (!manifest || nextIndex < 0 || nextIndex >= manifest.records.length) return;
  const revision = ++loadRevision; index = nextIndex; currentTime = 0; clips = { A: null, B: null }; ui.loading.classList.add("visible");
  const record = manifest.records[index]!; ui.prompt.textContent = record.prompt; ui.count.textContent = `${index + 1} / ${manifest.records.length}`;
  ui.previous.disabled = index === 0; ui.next.textContent = index === manifest.records.length - 1 ? "Save choice" : "Save & next"; renderChoice(); renderProgress();
  try {
    const [a, b] = await Promise.all([fetchClip(record.candidates.A.result_url), fetchClip(record.candidates.B.result_url)]);
    if (revision !== loadRevision) return; clips = { A: a, B: b }; ui.timeline.max = String(Math.max(duration(), .001)); ui.total.textContent = formatTime(duration()); setPlaying(true);
  } catch (error) { if (revision === loadRevision) { ui.error.hidden = false; ui.error.textContent = error instanceof Error ? error.message : "Comparison failed to load."; } }
  finally { if (revision === loadRevision) ui.loading.classList.remove("visible"); }
}

function setPlaying(value: boolean): void { playing = value; playStarted = performance.now(); playFrom = currentTime >= duration() ? 0 : currentTime; if (playing) currentTime = playFrom; ui.play.textContent = playing ? "❚❚" : "▶"; ui.play.setAttribute("aria-label", playing ? "Pause" : "Play"); }
function updateFrame(): void {
  if (playing && duration() > 0) { currentTime = playFrom + (performance.now() - playStarted) / 1000; if (currentTime >= duration()) { if (ui.loop.checked) { currentTime %= duration(); playStarted = performance.now(); playFrom = currentTime; } else { currentTime = duration(); setPlaying(false); } } }
  const a = frameAt(clips.A, Math.min(currentTime, clips.A?.duration ?? 0)); const b = frameAt(clips.B, Math.min(currentTime, clips.B?.duration ?? 0));
  scenes.aEgo.applyFrame(a); scenes.aOrbit.applyFrame(a); scenes.bEgo.applyFrame(b); scenes.bOrbit.applyFrame(b);
  ui.timeline.value = String(currentTime); ui.current.textContent = formatTime(currentTime); requestAnimationFrame(updateFrame);
}

document.querySelectorAll<HTMLButtonElement>("[data-choice]").forEach((button) => button.addEventListener("click", () => { const record = currentRecord(); if (!record) return; drafts[record.clip_id]!.choice = button.dataset.choice as Choice; ui.warning.hidden = true; saveDraft(); renderChoice(); }));
ui.note.addEventListener("input", () => { const record = currentRecord(); if (record) { drafts[record.clip_id]!.note = ui.note.value; saveDraft(); } });
ui.previous.addEventListener("click", () => void selectRecord(index - 1));
ui.next.addEventListener("click", () => { const record = currentRecord(); if (!record || !drafts[record.clip_id]?.choice) { ui.warning.hidden = false; return; } saveDraft(); if (manifest && index < manifest.records.length - 1) void selectRecord(index + 1); });
ui.play.addEventListener("click", () => setPlaying(!playing)); ui.timeline.addEventListener("input", () => { setPlaying(false); currentTime = Number(ui.timeline.value); });
ui.download.addEventListener("click", () => { if (!manifest || completedCount() !== manifest.records.length) return; downloadBlob(new Blob([`${JSON.stringify({ schema_version: "1.0", manifest_sha256: manifest.manifest_sha256, records: manifest.records.map((record) => ({ clip_id: record.clip_id, choice: drafts[record.clip_id]!.choice, note: drafts[record.clip_id]!.note })) }, null, 2)}\n`], { type: "application/json" }), "rigby-blinded-comparison-review.json"); });

async function initialize(): Promise<void> {
  try { const response = await fetch(manifestUrl, { headers: { Accept: "application/json" } }); if (!response.ok) throw new Error("Comparison manifest is unavailable."); manifest = await response.json() as ComparisonManifest; if (!Array.isArray(manifest.records) || manifest.records.length < 1) throw new Error("Comparison manifest must contain at least one prompt."); restoreDrafts(); renderProgress(); await selectRecord(0); }
  catch (error) { ui.loading.classList.remove("visible"); ui.error.hidden = false; ui.error.textContent = error instanceof Error ? error.message : "Comparison could not start."; }
}

requestAnimationFrame(updateFrame); void initialize();
