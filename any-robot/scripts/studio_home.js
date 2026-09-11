// The studio's front door: pick a robot, pick a world, ask for something.
//
// The page itself is a static file, so browsing works with no server at all --
// every robot, world and past run is already inlined. Submitting a prompt is the
// one thing that cannot be: it has to plan, ground, compile, simulate and gate,
// which is the pipeline, not a page. So the console talks to the local API when
// it is running and says exactly how to start it when it is not, rather than
// presenting a button that silently does nothing.

'use strict';

// Matches RIGBY_GENERAL_API_PORT, whose default is 8020. Hardcoding the wrong
// port produces a Check button that reports the pipeline down while it is up,
// which is a worse failure than no button at all.
const API_BASE = 'http://127.0.0.1:8020';

let homeRobot = null;
let homeEnvironment = '';
let homeStatus = null;
let apiOnline = null;   // null = not yet checked

function robotDirectory() {
  // Everything the studio knows about, whether or not it has geometry.
  const rows = new Map();
  (DATA.intake || []).forEach((entry) => {
    rows.set(entry.robot_id, {
      robot_id: entry.robot_id,
      origin: entry.origin,
      accepted: entry.accepted,
      morphology_class: entry.morphology_class,
      dofs: entry.dofs,
      reach_m: entry.reach_m,
      mass_kg: entry.mass_kg,
      effectors: entry.effectors || [],
      code: entry.code,
      detail: entry.detail,
    });
  });
  Object.keys(VIEWER.robots || {}).forEach((id) => {
    if (!rows.has(id)) {
      rows.set(id, { robot_id: id, origin: 'unknown', accepted: true, effectors: [] });
    }
    rows.get(id).playable = true;
  });
  return [...rows.values()].sort((a, b) => a.robot_id.localeCompare(b.robot_id));
}

function libraryFor(robotId) {
  return (DATA.libraries || {})[robotId] || null;
}

function renderHome() {
  const robots = robotDirectory();
  const usable = robots.filter(r => r.accepted !== false);
  if (!homeRobot && usable.length) homeRobot = usable[0].robot_id;

  const environments = DATA.environments || [];
  const chosen = robots.find(r => r.robot_id === homeRobot) || null;
  const library = chosen ? libraryFor(chosen.robot_id) : null;

  const card = (r) => {
    const on = r.robot_id === homeRobot;
    const refused = r.accepted === false;
    const kinds = (r.effectors || []).map(e => e.kind).join(', ');
    return `<button class="rcard ${on ? 'on' : ''} ${refused ? 'refused' : ''}"
      ${refused ? 'disabled' : `data-pick-robot="${vesc(r.robot_id)}"`}>
      <div class="thumb" data-thumb="robot" data-thumb-id="${vesc(r.robot_id)}"></div>
      <div class="rname mono">${vesc(r.robot_id)}</div>
      <div class="rmeta">${vesc(r.origin)}${r.playable ? ' · playable' : ''}</div>
      ${refused
        ? `<div class="rrefused">refused: ${vesc(r.code || 'unusable')}</div>`
        : `<div class="rspecs">
             <span>${vesc(r.morphology_class || '')}</span>
             <span>${r.dofs ?? '?'} dof</span>
             <span>${r.reach_m ? r.reach_m.toFixed(2) + ' m' : ''}</span>
           </div>
           <div class="rgrip">${kinds ? vesc(kinds) : 'no effector'}</div>`}
    </button>`;
  };

  const envCard = (e) => `<button class="ecard ${homeEnvironment === e.environment_id ? 'on' : ''}"
      data-pick-env="${vesc(e.environment_id)}">
      <div class="thumb wide" data-thumb="env" data-thumb-id="${vesc(e.environment_id)}"></div>
      <div class="mono">${vesc(e.environment_id)}</div>
      <div class="note">${vesc(e.description)}</div>
      <div class="rspecs">${(e.objects || []).map(o =>
        `<span>${vesc(o.name)} · ${(o.span_m * 1000).toFixed(0)} mm</span>`).join('')}</div>
    </button>`;

  const status = homeStatus
    ? `<div class="runstatus ${homeStatus.kind}">${homeStatus.html}</div>`
    : '';

  const apiLine = apiOnline === true
    ? '<span class="pass">pipeline reachable</span>'
    : apiOnline === false
      ? `<span class="bad">pipeline not running</span> &mdash; start it with
         <code>uv run rigby-general</code>, then press Check again`
      : '<span class="na">pipeline not checked</span>';

  return `
  <section>
    <h2>Run something</h2>
    <div class="card">
      Pick a robot, optionally a world to put it in, and ask for something in
      plain language.
      <div class="note">Browsing here needs nothing running &mdash; every robot,
        world and past run is already in this file. Submitting a prompt does: it
        has to plan, ground, compile, simulate and gate, which is the pipeline
        rather than a page. Status: ${apiLine}
        <button class="tbtn" data-act="ping">Check</button></div>
    </div>
  </section>

  <section>
    <h2>1 &middot; Robot</h2>
    <div class="rgrid">${robots.map(card).join('')}</div>
    <div class="note">Robots that were refused at intake are shown and disabled;
      the reason is on the card and the full measurement is in the Intake view.</div>
  </section>

  ${chosen && library ? `<section>
    <h2>What ${vesc(chosen.robot_id)} can be asked for</h2>
    <div class="card">
      <b>${library.certified_count}</b> certified primitives of
      ${library.attempted} attempted
      ${library.coverage !== null && library.coverage !== undefined
        ? `&middot; ${Math.round(library.coverage * 100)}% schema coverage` : ''}
      <div class="note">A prompt can only be answered out of this library. If it
        is empty, every prompt refuses at the binding stage, and that is the
        honest answer rather than a failure of the request.</div>
    </div>
  </section>` : ''}

  <section>
    <h2>2 &middot; World <span class="note">optional</span></h2>
    <div class="egrid">
      <button class="ecard ${homeEnvironment === '' ? 'on' : ''}" data-pick-env="">
        <div class="thumb wide thumb-none"></div>
        <div class="mono">free space</div>
        <div class="note">No objects. The arm moves through its own workspace,
          which is what the baked primitives describe.</div>
      </button>
      ${environments.map(envCard).join('')}
    </div>
  </section>

  <section>
    <h2>3 &middot; Prompt</h2>
    <div class="card">
      <div class="promptrow">
        <input id="home-prompt" type="text" class="promptbox"
          placeholder="reach out as far as you can and then come back"
          value="">
        <button class="tbtn go" data-act="run">Run</button>
      </div>
      <div class="chips">${[
        'reach out as far as you can and then come back',
        'trace a big circle',
        'wave three times',
        'sweep slowly across in front of you',
        'hold still',
        'pick it up and hold it',
      ].map(p => `<button class="chip" data-prompt="${vesc(p)}">${vesc(p)}</button>`).join('')}</div>
      ${status}
      <div id="home-preview"></div>
    </div>
  </section>

  <section>
    <h2>4 &middot; Or add a robot</h2>
    <div class="card">
      <div class="promptrow">
        <input id="home-file" type="file" accept=".urdf,.xml,.mjcf" class="filebox">
        <button class="tbtn go" data-act="upload">Upload</button>
      </div>
      <div class="note">The upload goes through the same intake every robot here
        did: scanned before it is parsed, compiled as delivered, measured, and
        either accepted with its morphology or refused naming the measurement
        that failed. Meshes must be beside the model or it will be refused for
        reaching outside its own folder &mdash; which is the sandbox working.</div>
    </div>
  </section>`;
}

// -- pipeline progress -------------------------------------------------------

/* The stages in the order run.py executes them, with what each one is deciding.
   Listing them up front means a run that stops early still shows the stages it
   never reached, greyed -- which is the difference between "it failed" and "it
   failed here, and these were never tried". */
const PIPELINE = [
  ['firewall',      'is this the kind of thing the system does?'],
  ['planning',      'words to a body-neutral schema'],
  ['binding',       'does this body have a certified primitive?'],
  ['grounding',     'schema terms resolved against measured scale'],
  ['compilation',   'waypoints to a time-sampled trajectory'],
  ['certification', 'simulate three times and agree'],
];

/* A stage's own record, rendered as the reason rather than the code. The
   detail is a bag of measurements whose keys differ per stage, so the useful
   ones are named explicitly and the rest are shown as they come. */
function stageDetail(name, detail, failure) {
  if (!detail || typeof detail !== 'object') return '';
  const bits = [];
  if (name === 'planning') {
    (detail.cues || []).forEach((c) => {
      bits.push(`clause <b>&ldquo;${vesc(c.clause)}&rdquo;</b> &rarr; <code>${
        vesc(c.entry_id || '?')}</code>${c.remove ? ' @' + vesc(c.remove) : ''}`);
    });
    if (detail.afforded_schemas !== undefined) {
      bits.push(`${detail.afforded_schemas} schemas afforded by this body`);
    }
    if (failure && detail.clause) {
      // The clause that stopped it, which for a multi-clause prompt is the
      // whole question: the rest were never read.
      bits.push(`stopped on clause <b>&ldquo;${vesc(detail.clause)}&rdquo;</b>`);
    }
    if (failure && detail.entry_id) {
      bits.push(`needed <code>${vesc(detail.entry_id)}</code>, which this body has
        no certified primitive for`);
    }
  } else if (name === 'binding') {
    if (detail.library_size !== undefined) {
      bits.push(`${detail.library_size} certified primitives in this library`);
    }
  } else if (name === 'grounding') {
    if (detail.reach_radius_m) bits.push(`reach ${(detail.reach_radius_m*1000).toFixed(0)} mm`);
    if (detail.neutral_speed_mps) bits.push(`neutral pace ${detail.neutral_speed_mps.toFixed(3)} m/s`);
  } else if (name === 'compilation') {
    if (detail.frames) bits.push(`${detail.frames} frames @ ${detail.sample_hz || 240} Hz`);
  } else if (name === 'certification') {
    if (detail.tracking_error_m !== undefined) {
      bits.push(`tracking ${(detail.tracking_error_m*1000).toFixed(1)} mm`);
    }
    if (detail.base_drift_m !== undefined) {
      bits.push(`base drift ${(detail.base_drift_m*1000).toFixed(2)} mm`);
    }
    if (detail.repeats) {
      bits.push(`${detail.repeats} replays ${detail.replay_agreement ? 'agreed' : 'DISAGREED'}`);
    }
  }
  if (detail.failure_code) bits.push(`<code>${vesc(detail.failure_code)}</code>`);
  return bits.length ? `<div class="st-detail">${bits.join(' &middot; ')}</div>` : '';
}

function pipelineMarkup(body) {
  const trace = body.trace || {};
  const stages = trace.stages || [];
  const byName = new Map(stages.map(s => [s.name, s]));
  const failedAt = (trace.failure || {}).stage;
  let reached = true;

  // A stage that recorded nothing but is followed by one that did has plainly
  // run -- a silent pass, not a skip. Only the stages after the failure were
  // genuinely never reached, and conflating the two would misreport the run.
  const lastRecorded = PIPELINE.reduce(
    (acc, [name], i) => (byName.has(name) ? i : acc), -1);

  const rows = PIPELINE.map(([name, what], i) => {
    const s = byName.get(name);
    let cls = 'skipped', mark = '&middot;';
    if (s) {
      cls = s.status === 'ok' ? 'ok' : 'bad';
      mark = s.status === 'ok' ? '&#10003;' : '&#10005;';
    } else if (i < lastRecorded) {
      cls = 'ok'; mark = '&#10003;';
    } else if (!reached) {
      cls = 'never';
    }
    if (s && s.status !== 'ok') reached = false;
    const detail = s ? stageDetail(name, s.detail, s.status !== 'ok') : '';
    const note = !s && !reached
      ? '<div class="st-detail">never reached &mdash; the run stopped earlier</div>'
      : '';
    return `<div class="st ${cls}">
      <div class="st-mark">${mark}</div>
      <div class="st-body">
        <div class="st-name">${vesc(name)}<span class="st-what">${what}</span></div>
        ${detail}${note}
      </div>
    </div>`;
  }).join('');

  const f = trace.failure;
  const banner = f
    ? `<div class="runbanner bad">
         <b>Refused at ${vesc(f.stage)}</b>
         <code>${vesc(f.code || '')}</code>
         <div>${vesc(f.detail || '')}</div>
       </div>`
    : `<div class="runbanner ok"><b>Certified</b>
         <div>simulated three times with agreeing state hashes</div></div>`;

  return `<div class="pipeline">${banner}
    <div class="stages">${rows}</div>
    <div class="note">Trace <code>${vesc(body.trace_id || '')}</code> &mdash; stored,
      and now listed under Runs.</div>
  </div>`;
}


/* A run submitted here is a run like any other: it belongs in the list with the
   rest. The page is built ahead of time, so the trace is pushed into the same
   array the Runs view reads -- newest first, matching how the build sorts. */
function adoptRun(body) {
  if (!body || !body.trace) return;
  const trace = body.trace;
  const at = DATA.traces.findIndex(t => t.trace_id === trace.trace_id);
  if (at >= 0) DATA.traces.splice(at, 1);
  DATA.traces.unshift(trace);

  // Geometry and rollout, so the Runs view can play it without a rebuild.
  //
  // The scene that comes back is the scene the run actually happened in: for a
  // world run that is the robot *and* the fixtures *and* the object, and its
  // qpos includes the block's free joint. Filing it under the robot's bare
  // entry would throw all of that away and play the arm alone in empty space,
  // which is what the preview did -- so it is registered as its own world and
  // referenced by name.
  VIEWER.robots[trace.robot_id] = VIEWER.robots[trace.robot_id] || {};
  const entry = VIEWER.robots[trace.robot_id];
  if (body.scene && !entry.scene) {
    entry.scene = body.scene;
    if (body.rest_qpos) entry.rest_qpos = body.rest_qpos;
  }
  if (body.scene) {
    VIEWER.envs = VIEWER.envs || {};
    VIEWER.envs[runSceneRef(trace.trace_id)] =
      { scene: body.scene, rest_qpos: body.rest_qpos, robot_id: trace.robot_id };
  }
  if (body.track) VIEWER.runs[trace.trace_id] = { track: body.track };

  const header = document.querySelector('.sub');
  if (header) {
    header.textContent = DATA.traces.length + ' runs \u00b7 every stage recorded';
  }
}

/* Scenes from submitted runs are filed alongside the authored worlds, under a
   key that cannot collide with one. */
function runSceneRef(traceId) { return 'run:' + traceId; }

/* The motion, where it was asked for. A refused run has no rollout, and the
   absence is stated rather than shown as an empty player. */
function mountRunPreview(body) {
  const host = document.getElementById('home-preview');
  if (!host) return;
  if (!body.track || !body.scene) {
    host.innerHTML = body.accepted
      ? '<div class="note">No rollout came back for this run.</div>'
      : '<div class="note">Nothing to play &mdash; the run refused before it was simulated.</div>';
    return;
  }
  // Reference the run's own scene rather than the robot's bare one.
  host.innerHTML = playerMarkup(
    body.robot_id, '', 'env|' + runSceneRef(body.trace_id)) + demoBarMarkup(body.trace_id);
  mountPending();
  const bar = host.querySelector('[data-save-demo]');
  if (bar) bar.addEventListener('click', () => void saveDemo(body.trace_id, bar));
  const node = host.querySelector('[data-viewer]');
  if (node && node._handle) {
    node._handle.setTrack(body.track);
    node._handle.frameCamera();
  }
}

// -- selection thumbnails ----------------------------------------------------

const THUMB_CACHE = new Map();
let thumbQueue = Promise.resolve();

/* One offscreen renderer, reused. Sixty live contexts is more than a browser
   will hand out, so scenes are drawn one at a time and kept as data URLs. */
function renderThumb(key, scene, rest) {
  if (THUMB_CACHE.has(key)) return Promise.resolve(THUMB_CACHE.get(key));
  thumbQueue = thumbQueue.then(() => new Promise((resolve) => {
    let host = null;
    try {
      host = document.createElement('div');
      host.style.cssText =
        'position:fixed;left:-9999px;top:0;width:320px;height:200px;pointer-events:none';
      document.body.appendChild(host);
      const handle = mountViewer(host, scene, { rest });
      if (!handle) throw new Error('no renderer');
      handle.frameCamera();
      // One frame, then read it back. rAF twice so the resize observer has
      // settled the canvas before the draw that gets captured.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        let url = '';
        try { url = handle.canvas.toDataURL('image/png'); } catch (_) { url = ''; }
        THUMB_CACHE.set(key, url);
        try {
          const gl = handle.canvas.getContext('webgl2');
          const lose = gl && gl.getExtension('WEBGL_lose_context');
          if (lose) lose.loseContext();
        } catch (_) { /* the context is going away regardless */ }
        host.remove();
        resolve(url);
      }));
    } catch (_) {
      if (host) host.remove();
      THUMB_CACHE.set(key, '');
      resolve('');
    }
  }));
  return thumbQueue;
}

/* Fill every placeholder currently on screen. Called after each render, so a
   card that appears when the robot changes gets its picture too. */
function paintThumbs() {
  document.querySelectorAll('[data-thumb]:not([data-painted])').forEach((node) => {
    const kind = node.dataset.thumb;
    const id = node.dataset.thumbId;
    let scene = null, rest = null, key = '';
    if (kind === 'robot') {
      const entry = (VIEWER.robots || {})[id];
      if (entry && entry.scene) { scene = entry.scene; rest = entry.rest_qpos; key = 'r:' + id; }
    } else {
      // The selected robot standing in that world -- which is the question a
      // world thumbnail is actually being asked.
      const entry = (VIEWER.envs || {})[id + '.' + homeRobot];
      if (entry && entry.scene) {
        scene = entry.scene; rest = entry.rest_qpos; key = 'e:' + id + '.' + homeRobot;
      }
    }
    node.setAttribute('data-painted', '1');
    if (!scene) { node.classList.add('thumb-none'); return; }
    renderThumb(key, scene, rest).then((url) => {
      if (url) node.style.backgroundImage = 'url(' + url + ')';
      else node.classList.add('thumb-none');
    });
  });
}

async function pingApi() {
  try {
    const response = await fetch(`${API_BASE}/api/v3/health`, { mode: 'cors' });
    apiOnline = response.ok;
  } catch (_) {
    apiOnline = false;
  }
  render();
}

function setStatus(kind, html) {
  homeStatus = { kind, html };
  render();
}

async function submitPrompt() {
  const field = document.getElementById('home-prompt');
  const prompt = (field && field.value || '').trim();
  if (!prompt) { setStatus('bad', 'Type a prompt first.'); return; }
  if (!homeRobot) { setStatus('bad', 'Pick a robot first.'); return; }

  setStatus('busy', `Running <code>${vesc(prompt)}</code> on
    <code>${vesc(homeRobot)}</code>&hellip; this simulates and gates, so it takes
    a few seconds.`);
  try {
    const response = await fetch(`${API_BASE}/api/v3/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        robot_id: homeRobot,
        environment: homeEnvironment || null,
      }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail || {};
      setStatus('bad', `Refused: <code>${vesc(detail.failure_code || response.status)}</code>
        ${vesc(detail.message || JSON.stringify(detail).slice(0, 200))}`);
      return;
    }
    adoptRun(body);
    setStatus(body.accepted ? 'pass' : 'warn', pipelineMarkup(body));
    mountRunPreview(body);
  } catch (error) {
    apiOnline = false;
    setStatus('bad', `Could not reach the pipeline at <code>${API_BASE}</code>.
      Start it with <code>uv run rigby-general</code> and try again.`);
  }
}

async function submitUpload() {
  const field = document.getElementById('home-file');
  const file = field && field.files && field.files[0];
  if (!file) { setStatus('bad', 'Choose a .urdf file first.'); return; }

  setStatus('busy', `Ingesting <code>${vesc(file.name)}</code>&hellip;`);
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch(`${API_BASE}/api/v3/robots`, {
      method: 'POST', body: form,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail || {};
      setStatus('bad', `Refused: <code>${vesc(detail.failure_code || response.status)}</code>
        ${vesc(detail.message || '')}`);
      return;
    }
    setStatus('pass', `Accepted as <code>${vesc(body.robot_id || file.name)}</code>.
      Bake its library with
      <code>uv run python scripts/bake_robot.py ${vesc(body.robot_id || '')}</code>,
      then rebuild the studio.`);
  } catch (error) {
    apiOnline = false;
    setStatus('bad', `Could not reach the pipeline at <code>${API_BASE}</code>.`);
  }
}

function wireHome() {
  // Placeholders are rebuilt on every render, so painting here covers the
  // first paint and every change of selection -- including a robot change,
  // which alters what the world thumbnails should be showing.
  paintThumbs();
  document.querySelectorAll('[data-pick-robot]').forEach((node) => {
    node.onclick = () => { homeRobot = node.dataset.pickRobot; render(); };
  });
  document.querySelectorAll('[data-pick-env]').forEach((node) => {
    node.onclick = () => { homeEnvironment = node.dataset.pickEnv; render(); };
  });
  document.querySelectorAll('[data-prompt]').forEach((node) => {
    node.onclick = () => {
      const field = document.getElementById('home-prompt');
      if (field) { field.value = node.dataset.prompt; field.focus(); }
    };
  });
  const ping = document.querySelector('[data-act="ping"]');
  if (ping) ping.onclick = pingApi;
  const run = document.querySelector('[data-act="run"]');
  if (run) run.onclick = submitPrompt;
  const upload = document.querySelector('[data-act="upload"]');
  if (upload) upload.onclick = submitUpload;
  const field = document.getElementById('home-prompt');
  if (field) {
    field.onkeydown = (e) => { if (e.key === 'Enter') submitPrompt(); };
  }
}


/* Every run worth keeping is one click from the shared registry. The server
   writes demos/registry/<id>.json with who asked, the commit, the branch and
   the command, copies the clip beside it and regenerates demos/index.html;
   committing the three is the person's own act, on their own branch. */
function demoBarMarkup(traceId) {
  return `<div class="demo-bar">
    <button type="button" class="demo-save" data-save-demo="${traceId}">Save as demo</button>
    <span class="demo-status" data-demo-status></span>
  </div>`;
}

async function saveDemo(traceId, button) {
  const status = button.parentElement.querySelector('[data-demo-status]');
  button.disabled = true;
  status.textContent = 'registering\u2026';
  try {
    const response = await fetch(`${API_BASE}/api/v3/results/${encodeURIComponent(traceId)}/demo`, {
      method: 'POST', mode: 'cors', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.statusText);
    status.textContent = `saved ${body.registry} as ${body.who} on ${body.source.branch}; commit it with the code`;
  } catch (error) {
    status.textContent = `not saved: ${error.message}`;
    button.disabled = false;
  }
}
