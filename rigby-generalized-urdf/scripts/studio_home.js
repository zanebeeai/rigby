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
    if (body.accepted) {
      setStatus('pass', `Certified. Trace <code>${vesc(body.trace_id)}</code>.
        Rebuild the studio to watch it:
        <code>uv run python scripts/build_studio.py</code>`);
    } else {
      setStatus('warn', `Refused at <b>${vesc(body.failure_stage)}</b>. That is a
        result: the stage and the measurement are in the trace
        <code>${vesc(body.trace_id)}</code>.`);
    }
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
