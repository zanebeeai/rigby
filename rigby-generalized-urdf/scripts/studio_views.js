// Studio views built on the viewer: the player, the motion tree, the library
// breakdown, and the explorer.
//
// These render markup as strings, like the rest of the studio, and mount live
// viewers into it afterwards. `mountPending` closes that gap: any element
// carrying `data-viewer` gets a renderer attached once it is in the DOM.

'use strict';

const VIEWERS = [];

function vesc(value) {
  return String(value === null || value === undefined ? '' : value)
    .replace(/[&<>"']/g, (c) => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
    ));
}

// A robot has two scenes: itself, and itself standing in a grasp scene with a
// block and a support plate. They are different models -- different body counts,
// and the block carries a free joint -- so the player has to be told which.
function sceneSource(robotId, ref) {
  const entry = VIEWER.robots[robotId];
  if (!entry) return null;
  if (ref && ref.indexOf('probe|') === 0) {
    return entry.probe
      ? { scene: entry.probe.scene, rest: entry.probe.rest_qpos }
      : null;
  }
  if (ref && ref.indexOf('env|') === 0) {
    // An authored world: the robot standing in it, with its fixtures and every
    // object, not just the one being attempted.
    const world = (VIEWER.envs || {})[ref.slice(4)];
    return world ? { scene: world.scene, rest: world.rest_qpos } : null;
  }
  return { scene: entry.scene, rest: entry.rest_qpos };
}

function trackFor(reference) {
  if (!reference) return null;
  const parts = reference.split('|');
  if (parts[0] === 'run') return VIEWER.runs[parts[1]] || null;
  if (parts[0] === 'probe') {
    const robot = VIEWER.robots[parts[1]];
    return robot && robot.probe ? robot.probe.track : null;
  }
  if (parts[0] === 'envtrial') {
    const world = (VIEWER.envs || {})[parts[1]];
    if (!world) return null;
    const found = (world.attempts || []).find(a => a.trace_id === parts[2]);
    return found ? found.track : null;
  }
  if (parts[0] === 'prim') {
    const robot = VIEWER.robots[parts[1]];
    if (!robot) return null;
    const found = (robot.primitives || []).find(p => p.record_id === parts[2]);
    return found ? found.track : null;
  }
  return null;
}

// -- the player --------------------------------------------------------------

function playerMarkup(robotId, trackRef, sceneRef) {
  if (!sceneSource(robotId, sceneRef)) {
    return '<div class="viewer-error">No geometry exported for <code>' +
      vesc(robotId) + '</code>. Run <code>python scripts/export_viewer.py ' +
      vesc(robotId) + '</code>.</div>';
  }
  return `
  <div class="player" data-viewer="${vesc(robotId)}" data-track="${vesc(trackRef || '')}"
       data-scene="${vesc(sceneRef || '')}">
    <div class="viewer-host"></div>
    <div class="transport">
      <button class="tbtn" data-act="play" title="Play / pause">&#9654;</button>
      <input class="scrub" type="range" min="0" max="1000" value="0" step="1">
      <span class="clock">&mdash;</span>
      <select class="speed" title="Playback speed">
        <option value="0.25">0.25&times;</option>
        <option value="0.5">0.5&times;</option>
        <option value="1" selected>1&times;</option>
        <option value="2">2&times;</option>
        <option value="4">4&times;</option>
      </select>
      <label class="tlab"><input type="checkbox" class="loop" checked> loop</label>
      <button class="tbtn" data-act="frame"
        title="Recentre and refit the camera">&#9678;</button>
    </div>
  </div>`;
}

function mountPending() {
  VIEWERS.length = 0;
  document.querySelectorAll('[data-viewer]').forEach((node) => {
    const robotId = node.dataset.viewer;
    const source = sceneSource(robotId, node.dataset.scene);
    if (!source) return;
    const scene = source.scene;
    const host = node.querySelector('.viewer-host');

    const clock = node.querySelector('.clock');
    const scrub = node.querySelector('.scrub');
    const playBtn = node.querySelector('[data-act="play"]');
    const bars = document.querySelector('.timeline');

    const handle = mountViewer(host, scene, {
      rest: source.rest,
      onTime: (time, playing) => {
        const track = handle && handle.track;
        const duration = track ? track.duration_s : 0;
        if (clock) {
          clock.textContent = duration
            ? time.toFixed(2) + ' / ' + duration.toFixed(2) + ' s' : '—';
        }
        if (scrub && duration) {
          scrub.value = String(Math.round(time / duration * 1000));
        }
        if (playBtn) playBtn.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
        if (bars) highlightPhase(bars, time);
      },
    });
    if (!handle) return;

    const track = trackFor(node.dataset.track);
    if (track) handle.setTrack(track);

    if (playBtn) playBtn.onclick = () => handle.toggle();
    if (scrub) {
      scrub.oninput = () => {
        const duration = handle.track ? handle.track.duration_s : 0;
        handle.seek(Number(scrub.value) / 1000 * duration);
      };
    }
    const speed = node.querySelector('.speed');
    if (speed) speed.onchange = () => handle.setSpeed(Number(speed.value));
    const loop = node.querySelector('.loop');
    if (loop) loop.onchange = () => handle.setLoop(loop.checked);
    const frameBtn = node.querySelector('[data-act="frame"]');
    if (frameBtn) frameBtn.onclick = () => handle.frameCamera();

    node._handle = handle;
    VIEWERS.push(handle);
  });
}

function highlightPhase(bars, time) {
  bars.querySelectorAll('.phase').forEach((bar) => {
    const start = Number(bar.dataset.start);
    const end = Number(bar.dataset.end);
    bar.classList.toggle('active', time >= start && time < end);
  });
}

// A prompt run replays its own rollout; a contact probe replays the grasp it
// ran, in the scene it ran in. Returns '' when there is nothing exported, and
// the caller falls back to the rendered clip.
function playerForTrace(t) {
  if (t.kind === 'environment_trial') {
    if (!t.trial) return '';
    const key = t.trial.environment + '.' + t.robot_id;
    const world = (VIEWER.envs || {})[key];
    if (!world) return '';
    return playerMarkup(
      t.robot_id, 'envtrial|' + key + '|' + t.trace_id, 'env|' + key);
  }
  if (t.kind === 'contact_probe') {
    const entry = VIEWER.robots[t.robot_id];
    if (!entry || !entry.probe) return '';
    return playerMarkup(t.robot_id, 'probe|' + t.robot_id, 'probe|' + t.robot_id);
  }
  return VIEWER.runs[t.trace_id]
    ? playerMarkup(t.robot_id, 'run|' + t.trace_id)
    : '';
}

// -- (B) the motion tree over the animation ---------------------------------

const SLOT_ORDER = ['vector', 'conformation', 'deixis', 'contour', 'stative'];
const REMOVES = ['adjacent', 'proximal', 'medial', 'distal'];

function renderMotionTree(t) {
  const program = t.schema_program;
  if (!program || !program.segments || !program.segments.length) return '';
  const grounded = t.grounded;
  if (!grounded) return '';

  const bindings = new Map((t.bindings || []).map(b => [b.segment_id, b]));
  const phases = grounded.phases || [];
  const total = grounded.duration_s || 0;
  const links = program.links || [];

  const bars = phases.map((phase) => {
    const width = total ? ((phase.end_s - phase.start_s) / total * 100) : 0;
    const isRecovery = phase.phase_id === 'phase_recovery';
    const index = isRecovery ? -1 : Number(phase.phase_id.replace('phase_', ''));
    const segment = index < 0 ? null : program.segments[index];
    const binding = segment ? bindings.get(segment.segment_id) : null;
    const label = isRecovery
      ? 'recovery'
      : vesc(binding ? binding.entry_id : segment.segment_id);
    return `<div class="phase ${vesc(phase.kind)}" style="width:${width.toFixed(3)}%"
      data-start="${phase.start_s}" data-end="${phase.end_s}"
      data-seek="${phase.start_s}"
      title="${vesc(phase.phase_id)} &middot; ${(phase.end_s - phase.start_s).toFixed(2)} s">
      <span>${label}</span></div>`;
  }).join('');

  const rows = program.segments.map((segment, index) => {
    const binding = bindings.get(segment.segment_id);
    const phase = phases.find(p => p.phase_id === 'phase_' + index) || null;
    const schema = segment.motion_schema || {};
    const slots = SLOT_ORDER.filter(k => schema[k]).map(k =>
      `<div class="slot"><div class="k">${k}</div><div class="v">${vesc(schema[k])}</div></div>`
    ).join('');
    const manner = segment.manner || {};
    const active = Object.keys(manner)
      .filter(k => k !== 'repetition_count' && manner[k])
      .map(k => k + ' ' + (manner[k] > 0 ? '+' : '') + manner[k]);
    if (manner.repetition_count) active.push('×' + manner.repetition_count);
    const link = links.find(l => l.to_segment === segment.segment_id);

    return `<div class="seg">
      <div class="seg-head">
        <b class="mono">${vesc(segment.segment_id)}</b>
        <span class="rel">${link
          ? vesc(link.relation) + ' after ' + vesc(link.from_segment)
          : 'start'}</span>
        ${phase ? `<span class="span" data-seek="${phase.start_s}">${
          phase.start_s.toFixed(2)}&ndash;${phase.end_s.toFixed(2)} s</span>` : ''}
      </div>
      <div class="seg-body">
        <div class="slots">${slots}
          <div class="slot"><div class="k">remove</div><div class="v">${
            vesc((segment.region || {}).remove)}</div></div>
          <div class="slot"><div class="k">frame</div><div class="v">${
            vesc(segment.frame)}</div></div>
          <div class="slot"><div class="k">boundary</div><div class="v">${
            vesc(segment.boundary)}</div></div>
        </div>
        <div class="note">manner: ${active.length ? vesc(active.join(', ')) : 'neutral'}</div>
        ${binding ? `<div class="bound">backed by <code>${vesc(binding.entry_id)}@${
          vesc(binding.certified_remove)}</code>
          <span class="hash">${vesc(binding.certified_primitive || 'none')}</span>
          ${binding.substituted ? '<div class="warn">substituted &mdash; the certified region differs from the one asked for</div>' : ''}
        </div>` : '<div class="bound"><span class="bad">no certified primitive backed this segment</span></div>'}
      </div>
    </div>`;
  }).join('');

  return `
  <section>
    <h2>Motion tree</h2>
    <div class="timeline">${bars}</div>
    <div class="note">Each bar is one phase of the compiled program, to scale in
      real time; click to jump the playback there. The last is recovery &mdash;
      the grounder retraces the outbound path rather than cutting home, which is
      what makes each primitive independently reusable.</div>
    <div class="segs">${rows}</div>
  </section>`;
}

// -- (A) the robot's whole primitive library, shown per run -----------------

function renderLibraryBreakdown(robotId, used) {
  const lib = DATA.libraries[robotId];
  if (!lib) return '';
  const inventory = DATA.inventory || [];
  const certified = lib.certified || {};
  const refused = lib.refused || {};
  const afforded = new Set(lib.afforded || []);
  const usedSet = new Set(used || []);

  const rows = inventory.map((entry) => {
    const isAfforded = afforded.has(entry.id);
    const have = certified[entry.id] || [];
    const cells = REMOVES.map((remove) => {
      if (!isAfforded) return '<td class="cell na">&mdash;</td>';
      const ok = have.indexOf(remove) >= 0;
      return '<td class="cell ' + (ok ? 'yes' : 'no') + '">' +
        (ok ? '&#9679;' : '&#183;') + '</td>';
    }).join('');
    const why = !isAfforded
      ? '<span class="na">not afforded &mdash; needs ' +
        vesc((entry.requires || []).join(', ') || 'nothing') + '</span>'
      : (have.length ? '' : '<span class="bad">' +
        vesc(refused[entry.id] || 'no certified region') + '</span>');
    return `<tr class="${usedSet.has(entry.id) ? 'used' : ''}">
      <td class="mono">${vesc(entry.id)}${usedSet.has(entry.id)
        ? ' <span class="tag">used here</span>' : ''}
        <div class="note">${vesc(entry.gloss)}</div></td>
      ${cells}<td>${why}</td></tr>`;
  }).join('');

  const pct = (v) => v === null || v === undefined ? '&mdash;' : Math.round(v * 100) + '%';

  return `
  <section>
    <h2>Base primitives for ${vesc(robotId)}</h2>
    <div class="card">
      <b>${lib.certified_count} certified</b> of ${lib.attempted} attempted
      &middot; yield ${pct(lib.yield)} &middot; schema coverage ${pct(lib.coverage)}
      <div class="note">Every row is one schema from the sealed inventory; every
        column a degree of remove. A filled dot is a primitive this body actually
        holds. None of it was authored &mdash; the set is the cross product of
        what the robot was measured to afford.</div>
    </div>
    <table class="matrix"><thead><tr><th>schema</th>${
      REMOVES.map(r => '<th class="mono">' + r + '</th>').join('')
    }<th>if empty, why</th></tr></thead><tbody>${rows}</tbody></table>
  </section>`;
}

// -- (C) the explorer --------------------------------------------------------

let explorerRobot = null;
let explorerPrimitive = null;

function renderExplorer() {
  const ids = Object.keys(VIEWER.robots).sort();
  if (!ids.length) {
    return `<section><div class="card">No viewer geometry exported yet. Run
      <code>uv run python scripts/export_viewer.py</code>.</div></section>`;
  }
  if (!explorerRobot || !VIEWER.robots[explorerRobot]) explorerRobot = ids[0];
  const entry = VIEWER.robots[explorerRobot];
  const primitives = entry.primitives || [];
  if (!explorerPrimitive || !primitives.some(p => p.record_id === explorerPrimitive)) {
    explorerPrimitive = primitives.length ? primitives[0].record_id : null;
  }

  const grouped = new Map();
  primitives.forEach((primitive) => {
    if (!grouped.has(primitive.entry_id)) grouped.set(primitive.entry_id, []);
    grouped.get(primitive.entry_id).push(primitive);
  });
  const list = [...grouped.entries()].sort().map((pair) => {
    const items = pair[1].slice().sort(
      (a, b) => REMOVES.indexOf(a.remove) - REMOVES.indexOf(b.remove));
    return `<div class="pgroup">
      <div class="pname mono">${vesc(pair[0])}</div>
      <div class="pchips">${items.map(p =>
        `<button class="chip ${p.record_id === explorerPrimitive ? 'on' : ''}"
          data-primitive="${vesc(p.record_id)}"
          title="${p.track.duration_s.toFixed(2)} s &middot; ${p.waypoints} waypoints">${
          vesc(p.remove)}</button>`).join('')}</div>
    </div>`;
  }).join('');

  const joints = entry.scene.joints.map((joint) => {
    const range = joint.range || [-Math.PI, Math.PI];
    const value = entry.rest_qpos[joint.qposadr] || 0;
    return `<div class="jrow">
      <label class="mono">${vesc(joint.name)}</label>
      <input type="range" class="jslider" data-qposadr="${joint.qposadr}"
        min="${range[0]}" max="${range[1]}" step="0.001" value="${value}">
      <span class="jval mono" data-for="${joint.qposadr}">${value.toFixed(3)}</span>
    </div>`;
  }).join('');

  const chosen = primitives.find(p => p.record_id === explorerPrimitive);
  const measured = chosen ? (chosen.measurements || {}) : {};

  return `
  <section>
    <h2>Explorer</h2>
    <div class="card">
      Every uploaded robot, and every base primitive baked for it. Pick a
      primitive to watch it, or drive the joints yourself &mdash; the viewer runs
      the same forward kinematics MuJoCo does, checked against it to under a
      picometre, so a pose dialled in by hand is a pose the robot would hold.
      Drag to orbit, right-drag or shift-drag to pan, scroll to zoom.
      <div class="note">Moving a joint detaches playback; pick a primitive again
        to reattach it.</div>
    </div>
    <div class="robotpick">${ids.map(id =>
      `<button class="chip ${id === explorerRobot ? 'on' : ''}" data-robot="${
        vesc(id)}">${vesc(id)}</button>`).join('')}</div>
  </section>
  <section class="explorer">
    <div class="ecol">
      ${playerMarkup(explorerRobot, explorerPrimitive
        ? 'prim|' + explorerRobot + '|' + explorerPrimitive : '')}
      ${chosen ? `<div class="card" style="margin-top:10px">
        <b class="mono">${vesc(chosen.entry_id)}@${vesc(chosen.remove)}</b>
        <div class="note">${vesc(chosen.segment_key)}</div>
        <div class="slots">
          <div class="slot"><div class="k">duration</div><div class="v">${
            chosen.track.duration_s.toFixed(2)} s</div></div>
          <div class="slot"><div class="k">waypoints</div><div class="v">${
            chosen.waypoints}</div></div>
          <div class="slot"><div class="k">tracking</div><div class="v">${
            (measured.tracking_error_m || 0).toFixed(4)} m</div></div>
          <div class="slot"><div class="k">site</div><div class="v">${
            vesc(chosen.figure_site)}</div></div>
        </div></div>` : ''}
    </div>
    <div class="ecol">
      <h3>Joints</h3>
      <div class="joints">${joints}</div>
      <button class="tbtn wide" data-act="rest">Return to rest pose</button>
      <h3 class="spaced">Base primitives (${primitives.length})</h3>
      <div class="plist">${list ||
        '<div class="note">None certified for this robot.</div>'}</div>
    </div>
  </section>`;
}

function wireExplorer() {
  document.querySelectorAll('[data-robot]').forEach((node) => {
    node.onclick = () => {
      explorerRobot = node.dataset.robot;
      explorerPrimitive = null;
      render();
    };
  });
  document.querySelectorAll('[data-primitive]').forEach((node) => {
    node.onclick = () => {
      explorerPrimitive = node.dataset.primitive;
      render();
      // Without this the player stays scrolled off the top and picking a
      // primitive looks like it did nothing at all.
      const player = document.querySelector('.explorer .player');
      if (player) player.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    };
  });

  const player = document.querySelector('.explorer [data-viewer]');
  const handle = player && player._handle;
  if (!handle) return;

  const entry = VIEWER.robots[explorerRobot];
  const pose = (entry.rest_qpos || []).slice();
  document.querySelectorAll('.jslider').forEach((slider) => {
    slider.oninput = () => {
      const address = Number(slider.dataset.qposadr);
      pose[address] = Number(slider.value);
      handle.setPose(pose);
      const readout = document.querySelector('.jval[data-for="' + address + '"]');
      if (readout) readout.textContent = Number(slider.value).toFixed(3);
    };
  });
  const reset = document.querySelector('[data-act="rest"]');
  if (reset) {
    reset.onclick = () => {
      document.querySelectorAll('.jslider').forEach((slider) => {
        const address = Number(slider.dataset.qposadr);
        pose[address] = entry.rest_qpos[address] || 0;
        slider.value = String(pose[address]);
        const readout = document.querySelector('.jval[data-for="' + address + '"]');
        if (readout) readout.textContent = pose[address].toFixed(3);
      });
      handle.setPose(pose);
    };
  }
}

function wireSeeks() {
  const handle = VIEWERS[0];
  if (!handle) return;
  document.querySelectorAll('[data-seek]').forEach((node) => {
    node.onclick = () => handle.seek(Number(node.dataset.seek));
  });
}

// -- worlds: the authored environments, and who could do what in them --------

const REFUSAL_LABEL = {
  object_too_wide: 'too wide for the jaw',
  object_out_of_reach: 'out of reach',
  object_inside_reach_hole: 'inside the reach hole',
  scene_not_buildable: 'scene would not compile',
};

function renderWorlds() {
  const trials = DATA.traces.filter(t => t.kind === 'environment_trial' && t.trial);
  if (!trials.length) {
    return `<section><div class="card">No environment trials recorded. Run
      <code>uv run python scripts/run_trials.py</code>.</div></section>`;
  }

  const worlds = new Map();
  trials.forEach((t) => {
    const id = t.trial.environment;
    if (!worlds.has(id)) {
      worlds.set(id, { description: t.trial.description, rows: [] });
    }
    worlds.get(id).rows.push(t);
  });

  const attempted = trials.filter(t => (t.trial.admission || {}).admitted);
  const held = trials.filter(t => t.accepted);

  const sections = [...worlds.entries()].sort().map((pair) => {
    const [id, world] = pair;
    const rows = world.rows.slice().sort((a, b) =>
      (a.robot_id + a.trial.object).localeCompare(b.robot_id + b.trial.object));
    const objects = [...new Set(rows.map(r => r.trial.object))];
    return `<section>
      <h2>${vesc(id)}</h2>
      <div class="card">
        ${vesc(world.description)}
        <div class="slots">
          <div class="slot"><div class="k">objects</div><div class="v">${
            objects.map(vesc).join(', ')}</div></div>
          <div class="slot"><div class="k">attempted</div><div class="v">${
            rows.filter(r => (r.trial.admission||{}).admitted).length}/${rows.length}</div></div>
          <div class="slot"><div class="k">held</div><div class="v">${
            rows.filter(r => r.accepted).length}</div></div>
        </div>
      </div>
      <table><thead><tr><th>robot</th><th>object</th><th class="num">span</th>
        <th class="num">distance</th><th class="num">reach</th><th>outcome</th></tr></thead>
      <tbody>${rows.map((r) => {
        const a = r.trial.admission || {};
        const outcome = r.accepted
          ? '<span class="pass">HELD</span>'
          : (a.admitted
              ? '<span class="bad">' + vesc((r.failure||{}).code || 'failed') + '</span>'
              : '<span class="na">' + vesc(
                  REFUSAL_LABEL[a.code] || a.code || 'refused') + '</span>');
        const nudged = (r.trial.bystanders || []).filter(b => b.disturbed);
        const idx = DATA.traces.indexOf(r);
        return `<tr>
          <td class="mono"><a href="#" data-open="${idx}">${vesc(r.robot_id)}</a></td>
          <td class="mono">${vesc(r.trial.object)}</td>
          <td class="num">${(r.trial.object_span_m*1000).toFixed(0)} mm</td>
          <td class="num">${(a.distance_m*1000||0).toFixed(0)} mm</td>
          <td class="num">${(a.reach_limit_m*1000||0).toFixed(0)} mm</td>
          <td>${outcome}${nudged.length
            ? '<div class="note bad">disturbed ' +
              nudged.map(b => vesc(b.name)).join(', ') + '</div>' : ''}</td>
        </tr>`;
      }).join('')}</tbody></table>
    </section>`;
  }).join('');

  return `
  <section>
    <h2>Worlds</h2>
    <div class="card">
      Four environments authored in absolute metres, before any robot and with no
      knowledge of what would be asked. The robot is mounted where the world says
      and the object stays where the file put it.
      <div class="note">That is what makes the refusals possible. A scene derived
        from the arm places its object inside the reach envelope by construction,
        so it can never report that something is too far away, too wide to hold,
        or too close to fold onto. Admission decides all three from ingest
        measurements alone, before anything is simulated.</div>
      <div class="slots">
        <div class="slot"><div class="k">pairings</div><div class="v">${trials.length}</div></div>
        <div class="slot"><div class="k">admitted</div><div class="v">${attempted.length}</div></div>
        <div class="slot"><div class="k">held</div><div class="v">${held.length}</div></div>
        <div class="slot"><div class="k">refused</div><div class="v">${
          trials.length - attempted.length}</div></div>
      </div>
    </div>
  </section>
  ${sections}`;
}

function wireWorlds() {
  document.querySelectorAll('[data-open]').forEach((node) => {
    node.onclick = (e) => {
      e.preventDefault();
      view = 'runs';
      document.querySelectorAll('button.tab').forEach(x =>
        x.setAttribute('aria-selected', String(x.dataset.view === 'runs')));
      select(Number(node.dataset.open));
    };
  });
}
