import "./review.css";
import { downloadBlob, frameAt, unwrapClip } from "./motion";
import { RigbyScene } from "./scene";
import {
  emptyScores,
  normalizeReviewManifest,
  queueCompletedCount,
  queueExports,
  resultReference,
  validateReviewQueue,
  type ReviewManifest,
  type ReviewRunDraft,
  type ReviewScore,
} from "./review-data";
import { DEFAULT_PARAMETERS, type ClipResult } from "./types";

const mount = document.querySelector<HTMLDivElement>("#review-app");
if (!mount) throw new Error("Review mount not found");

mount.innerHTML = `
  <div class="review-shell">
    <header class="review-header">
      <div class="review-brand"><i aria-hidden="true"><span></span><span></span><span></span></i><div><strong>Rigby</strong><small>Diverse prompt-aware review</small></div></div>
      <div class="review-progress" role="status" aria-live="polite"><span id="review-progress-text">Loading manifest…</span><div><i id="review-progress-bar"></i></div></div>
      <button id="download-review" class="review-button primary" disabled>Download completed review</button>
    </header>
    <main class="review-main">
      <section class="review-workspace" aria-label="Synchronized prompt-aware motion views">
        <div class="prompt-banner"><span>PROMPT</span><p id="review-prompt">Loading prompt…</p><strong id="opaque-id">—</strong></div>
        <div class="view-grid">
          <article class="review-view"><header><span>View A</span><strong>Egocentric</strong></header><div id="ego-view" class="review-canvas" aria-label="Egocentric animation view"></div></article>
          <article class="review-view"><header><span>View B</span><strong>Orbit · drag / zoom</strong></header><div id="orbit-view" class="review-canvas" aria-label="Orbit animation view"></div></article>
          <div id="clip-loading" class="clip-loading"><span></span><strong>Loading clip</strong></div>
        </div>
        <div class="review-playback">
          <button id="review-play" class="round-button" aria-label="Pause animation">❚❚</button>
          <time id="review-time">0:00.00</time>
          <input id="review-timeline" type="range" min="0" max="1" value="0" step="0.001" aria-label="Synchronized review timeline" />
          <time id="review-duration">0:00.00</time>
          <label><input type="checkbox" id="review-loop" checked /> Loop</label>
        </div>
      </section>
      <aside class="score-panel" aria-label="Motion ratings">
        <p class="review-eyebrow">Independent human score</p>
        <h1>Rate this motion</h1>
        <p class="score-guidance">Judge how faithfully the motion matches the shown target, style, timing, and wrist directions—and how natural it looks in each view.</p>
        <div id="rating-controls"></div>
        <label class="note-field" for="review-note"><span>Optional reviewer note</span><textarea id="review-note" rows="4" maxlength="500" placeholder="Note visible motion qualities only…"></textarea><small id="note-count">0 / 500</small></label>
        <div id="score-warning" class="score-warning" role="alert" hidden>Choose a rating for both views before continuing.</div>
        <div class="review-navigation">
          <button id="review-previous" class="review-button secondary">Previous</button>
          <button id="review-next" class="review-button primary">Save & next</button>
        </div>
        <div class="clip-dots" id="clip-dots" aria-label="Review progress by clip"></div>
        <p class="privacy-note"><span aria-hidden="true">◉</span> The downloaded artifact contains only opaque clip IDs, scores, and notes.</p>
      </aside>
    </main>
    <div id="review-error" class="review-error" role="alert" hidden><strong>Review unavailable</strong><p></p><button class="review-button secondary" onclick="location.reload()">Retry</button></div>
  </div>
`;

function element<T extends Element>(selector: string): T {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`Missing review element: ${selector}`);
  return found;
}

const ui = {
  progressText: element<HTMLElement>("#review-progress-text"),
  progressBar: element<HTMLElement>("#review-progress-bar"),
  download: element<HTMLButtonElement>("#download-review"),
  prompt: element<HTMLElement>("#review-prompt"),
  opaqueId: element<HTMLElement>("#opaque-id"),
  loading: element<HTMLElement>("#clip-loading"),
  play: element<HTMLButtonElement>("#review-play"),
  timeline: element<HTMLInputElement>("#review-timeline"),
  time: element<HTMLElement>("#review-time"),
  duration: element<HTMLElement>("#review-duration"),
  loop: element<HTMLInputElement>("#review-loop"),
  ratings: element<HTMLElement>("#rating-controls"),
  note: element<HTMLTextAreaElement>("#review-note"),
  noteCount: element<HTMLElement>("#note-count"),
  warning: element<HTMLElement>("#score-warning"),
  previous: element<HTMLButtonElement>("#review-previous"),
  next: element<HTMLButtonElement>("#review-next"),
  dots: element<HTMLElement>("#clip-dots"),
  error: element<HTMLElement>("#review-error"),
};

const egoScene = new RigbyScene(element<HTMLElement>("#ego-view"), DEFAULT_PARAMETERS.block);
const orbitScene = new RigbyScene(element<HTMLElement>("#orbit-view"), DEFAULT_PARAMETERS.block);
egoScene.setCamera("ego");
egoScene.setEgoFieldOfView(94);
orbitScene.setCamera("orbit");
orbitScene.setOrbitFraming([1.6, 1.5, 1.9], [0, 1.25, 0.25]);

interface ReviewRunSession extends ReviewRunDraft {
  url: string;
}

const requestedManifestUrls = new URLSearchParams(window.location.search).getAll("manifest").filter(Boolean);
let runs: ReviewRunSession[] = [];
let currentQueueIndex = 0;
let currentClip: ClipResult | null = null;
let currentTime = 0;
let playing = true;
let playStartedAt = performance.now();
let playStartedFrom = 0;
let loadRevision = 0;

function formatTime(time: number): string {
  return `${Math.floor(time / 60)}:${(time % 60).toFixed(2).padStart(5, "0")}`;
}

function validScore(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 5 ? value : null;
}

function restoreScores(url: string, manifest: ReviewManifest): Record<string, ReviewScore> {
  const blank = emptyScores(manifest);
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey(url)) ?? "{}") as Record<string, Partial<ReviewScore>>;
    for (const id of Object.keys(blank)) {
      blank[id] = {
        egocentric: validScore(saved[id]?.egocentric),
        orbit: validScore(saved[id]?.orbit),
        note: typeof saved[id]?.note === "string" ? saved[id].note!.slice(0, 500) : "",
      };
    }
  } catch {
    // A malformed local draft is ignored rather than contaminating review output.
  }
  return blank;
}

function saveDraft(): void {
  const run = currentRun();
  if (run) localStorage.setItem(storageKey(run.url), JSON.stringify(run.scores));
  renderProgress();
}

function storageKey(url: string): string {
  return `rigby-prompt-review-v2:${url}`;
}

function totalClips(): number {
  return runs.reduce((total, run) => total + run.manifest.records.length, 0);
}

function queuePosition(index = currentQueueIndex): { runIndex: number; clipIndex: number } {
  let cursor = index;
  for (let runIndex = 0; runIndex < runs.length; runIndex += 1) {
    const count = runs[runIndex]?.manifest.records.length ?? 0;
    if (cursor < count) return { runIndex, clipIndex: cursor };
    cursor -= count;
  }
  const runIndex = Math.max(0, runs.length - 1);
  return { runIndex, clipIndex: Math.max(0, (runs[runIndex]?.manifest.records.length ?? 1) - 1) };
}

function currentRun(): ReviewRunSession | undefined {
  return runs[queuePosition().runIndex];
}

function currentRecord() {
  const position = queuePosition();
  return runs[position.runIndex]?.manifest.records[position.clipIndex];
}

function ratingControl(view: "egocentric" | "orbit", label: string): HTMLElement {
  const section = document.createElement("fieldset");
  section.className = "rating-group";
  section.innerHTML = `<legend><span>${label}</span><small>recognizability + naturalness</small></legend><div class="rating-scale"></div><div class="rating-labels"><span>Unreadable</span><span>Excellent</span></div>`;
  const scale = section.querySelector<HTMLElement>(".rating-scale")!;
  for (let rating = 1; rating <= 5; rating += 1) {
    const labelElement = document.createElement("label");
    labelElement.innerHTML = `<input type="radio" name="${view}-rating" value="${rating}" /><span>${rating}</span>`;
    labelElement.querySelector("input")?.addEventListener("change", () => {
      const id = currentRecord()?.clip_id;
      const run = currentRun();
      if (!id || !run?.scores[id]) return;
      run.scores[id][view] = rating;
      ui.warning.hidden = true;
      saveDraft();
    });
    scale.append(labelElement);
  }
  return section;
}

ui.ratings.append(ratingControl("egocentric", "View A · Egocentric"), ratingControl("orbit", "View B · Orbit"));

function renderRatings(): void {
  const id = currentRecord()?.clip_id;
  const score = id ? currentRun()?.scores[id] : undefined;
  for (const view of ["egocentric", "orbit"] as const) {
    document.querySelectorAll<HTMLInputElement>(`input[name="${view}-rating"]`).forEach((input) => {
      input.checked = Number(input.value) === score?.[view];
    });
  }
  ui.note.value = score?.note ?? "";
  ui.noteCount.textContent = `${ui.note.value.length} / 500`;
  ui.warning.hidden = true;
}

function renderProgress(): void {
  if (!runs.length) return;
  const done = queueCompletedCount(runs);
  const total = totalClips();
  ui.progressText.textContent = `${done} of ${total} clips complete`;
  ui.progressBar.style.width = `${(done / total) * 100}%`;
  ui.download.disabled = done !== total;
  ui.download.textContent = done === total ? `Download ${runs.length === 1 ? "completed review" : `${runs.length} review files`}` : `Complete ${total - done} more`;
  ui.dots.replaceChildren();
  let queueIndex = 0;
  runs.forEach((run) => run.manifest.records.forEach((record) => {
    const index = queueIndex++;
    const button = document.createElement("button");
    const complete = run.scores[record.clip_id]?.egocentric !== null && run.scores[record.clip_id]?.orbit !== null;
    button.className = `${complete ? "complete" : ""} ${index === currentQueueIndex ? "active" : ""}`;
    button.textContent = String(index + 1);
    button.title = `Open clip ${index + 1}${complete ? ", rated" : ", not rated"}`;
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => void selectClip(index));
    ui.dots.append(button);
  }));
}

async function fetchClip(index: number): Promise<ClipResult> {
  const position = queuePosition(index);
  const record = runs[position.runIndex]?.manifest.records[position.clipIndex];
  if (!record) throw new Error("No review manifest is loaded.");
  const reference = resultReference(record);
  if (reference.kind === "inline") return unwrapClip(reference.value);
  const response = await fetch(reference.value, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Clip ${index + 1} could not be loaded.`);
  return unwrapClip(await response.json());
}

async function selectClip(index: number): Promise<void> {
  const total = totalClips();
  if (!runs.length || index < 0 || index >= total) return;
  const revision = ++loadRevision;
  currentQueueIndex = index;
  currentClip = null;
  currentTime = 0;
  ui.loading.classList.add("visible");
  ui.opaqueId.textContent = `Clip ${String(index + 1).padStart(2, "0")} / ${total}`;
  ui.prompt.textContent = currentRecord()?.prompt ?? "Loading prompt…";
  ui.prompt.removeAttribute("title");
  ui.previous.disabled = index === 0;
  ui.next.textContent = index === total - 1 ? "Save rating" : "Save & next";
  renderRatings();
  renderProgress();
  try {
    const clip = await fetchClip(index);
    if (revision !== loadRevision) return;
    if (!clip.frames.length) throw new Error(`Clip ${index + 1} has no animation frames.`);
    currentClip = clip;
    const prompt = currentRecord()?.prompt ?? clip.prompt ?? clip.program?.source_text ?? "Prompt unavailable";
    ui.prompt.textContent = prompt;
    ui.prompt.title = prompt;
    ui.timeline.max = String(Math.max(clip.duration, 0.001));
    ui.duration.textContent = formatTime(clip.duration);
    setPlaying(true);
  } catch (error) {
    if (revision !== loadRevision) return;
    showError(error instanceof Error ? error.message : "The clip could not be loaded.");
  } finally {
    if (revision === loadRevision) ui.loading.classList.remove("visible");
  }
}

function setPlaying(next: boolean): void {
  playing = next;
  playStartedAt = performance.now();
  playStartedFrom = currentTime >= (currentClip?.duration ?? 0) ? 0 : currentTime;
  if (playing) currentTime = playStartedFrom;
  ui.play.textContent = playing ? "❚❚" : "▶";
  ui.play.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
}

function updateFrame(): void {
  if (currentClip && playing) {
    currentTime = playStartedFrom + (performance.now() - playStartedAt) / 1000;
    if (currentTime >= currentClip.duration) {
      if (ui.loop.checked) {
        currentTime %= currentClip.duration || 1;
        playStartedAt = performance.now();
        playStartedFrom = currentTime;
      } else {
        currentTime = currentClip.duration;
        setPlaying(false);
      }
    }
  }
  const frame = frameAt(currentClip, currentTime);
  egoScene.applyFrame(frame);
  orbitScene.applyFrame(frame);
  ui.timeline.value = String(currentTime);
  ui.time.textContent = formatTime(currentTime);
  requestAnimationFrame(updateFrame);
}

function currentComplete(): boolean {
  const id = currentRecord()?.clip_id;
  const score = id ? currentRun()?.scores[id] : undefined;
  return score?.egocentric !== null && score?.egocentric !== undefined && score.orbit !== null && score.orbit !== undefined;
}

function showError(message: string): void {
  ui.error.hidden = false;
  ui.error.querySelector("p")!.textContent = message;
}

async function initialize(): Promise<void> {
  try {
    let urls = [...requestedManifestUrls];
    if (urls.length > 2) throw new Error("A review queue supports at most two acceptance manifests.");
    if (!urls.length) {
      const ledgerResponse = await fetch("/results/acceptance-runs/index.json", { headers: { Accept: "application/json" } });
      if (!ledgerResponse.ok) throw new Error("No acceptance-run ledger is available. Open this page with ?manifest=<manifest URL>.");
      const ledger = await ledgerResponse.json() as { runs?: Array<{ id?: string }> };
      urls = (ledger.runs ?? []).slice().reverse().flatMap((run) => run.id ? [`/results/acceptance-runs/${encodeURIComponent(run.id)}/blinded-gesture-manifest.json`] : []);
    }
    const loaded: ReviewRunSession[] = [];
    for (const url of urls) {
      const response = await fetch(url, { headers: { Accept: "application/json" } });
      if (!response.ok) {
        if (requestedManifestUrls.length) throw new Error("A requested gesture review manifest is unavailable.");
        continue;
      }
      const runManifest = normalizeReviewManifest(await response.json());
      loaded.push({ url, manifest: runManifest, scores: restoreScores(url, runManifest) });
      if (!requestedManifestUrls.length && loaded.length === 2) break;
    }
    validateReviewQueue(loaded.map((run) => run.manifest));
    runs = requestedManifestUrls.length ? loaded : loaded.reverse();
    if (!runs.length) throw new Error("No acceptance run is available for review.");
    renderProgress();
    await selectClip(0);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The review could not start.");
  }
}

ui.note.addEventListener("input", () => {
  const id = currentRecord()?.clip_id;
  const run = currentRun();
  if (id && run?.scores[id]) {
    run.scores[id].note = ui.note.value;
    saveDraft();
  }
  ui.noteCount.textContent = `${ui.note.value.length} / 500`;
});
ui.previous.addEventListener("click", () => void selectClip(currentQueueIndex - 1));
ui.next.addEventListener("click", () => {
  if (!currentComplete()) {
    ui.warning.hidden = false;
    ui.warning.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  saveDraft();
  if (currentQueueIndex < totalClips() - 1) void selectClip(currentQueueIndex + 1);
});
ui.play.addEventListener("click", () => setPlaying(!playing));
ui.timeline.addEventListener("input", () => {
  setPlaying(false);
  currentTime = Number(ui.timeline.value);
});
ui.download.addEventListener("click", () => {
  if (!runs.length || queueCompletedCount(runs) !== totalClips()) return;
  queueExports(runs).forEach((output, index) => {
    const runId = runs[index]?.url.match(/acceptance-runs\/([^/]+)/)?.[1] ?? `set-${String(index + 1).padStart(2, "0")}`;
    downloadBlob(new Blob([`${JSON.stringify(output, null, 2)}\n`], { type: "application/json" }), `hangten-prompt-review-${runId}.json`);
  });
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !(event.target instanceof HTMLTextAreaElement) && !(event.target instanceof HTMLInputElement)) {
    event.preventDefault();
    setPlaying(!playing);
  }
});

requestAnimationFrame(updateFrame);
void initialize();
