// A small WebGL2 robot viewer: forward kinematics in the browser, no dependency.
//
// Why hand-rolled rather than three.js: the studio is a single file opened from
// disk with no server, and every byte of a library would have to be inlined into
// it. What is actually needed here is one shader, six parametric shapes, and an
// orbit camera -- a few hundred lines against six hundred kilobytes.
//
// The forward kinematics below must agree with MuJoCo's. It is a direct port of
// `rigby_general.viewer.scene.forward_kinematics`, which is checked against
// `mj_kinematics` in the test suite; if you change one, change both. Measured
// agreement across every robot in the studio is under a picometre.

'use strict';

// -- small matrix helpers (column-major, WebGL convention) -------------------

function mat4Identity() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

// Builds a 4x4 from a row-major 3x3 rotation and a translation.
function mat4Compose(rot, pos) {
  return new Float32Array([
    rot[0], rot[3], rot[6], 0,
    rot[1], rot[4], rot[7], 0,
    rot[2], rot[5], rot[8], 0,
    pos[0], pos[1], pos[2], 1,
  ]);
}

function mat4Perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f; out[11] = -1;
  out[10] = (far + near) / (near - far);
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function mat4LookAt(eye, target, up) {
  const z = normalize([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function cross(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/n, v[1]/n, v[2]/n];
}

// Row-major 3x3, matching the Python reference.
function quatToMat(q) {
  const [w, x, y, z] = q;
  return [
    1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),
    2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),
    2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y),
  ];
}

function mat3Multiply(a, b) {
  const out = new Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r*3+c] = a[r*3]*b[c] + a[r*3+1]*b[3+c] + a[r*3+2]*b[6+c];
    }
  }
  return out;
}

function mat3Apply(m, v) {
  return [
    m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
    m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
    m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
  ];
}

function axisAngleMat(axis, angle) {
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  if (n < 1e-12) return [1,0,0, 0,1,0, 0,0,1];
  const [x, y, z] = [axis[0]/n, axis[1]/n, axis[2]/n];
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
    t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
    t*x*z - s*y, t*y*z + s*x, t*z*z + c,
  ];
}

// -- forward kinematics ------------------------------------------------------
// Port of rigby_general.viewer.scene.forward_kinematics. A body's frame is its
// parent's composed with its fixed offset; then each of its joints displaces it,
// rotating about an anchor or sliding along an axis, both in the body's frame.

function forwardKinematics(scene, qpos) {
  const bodies = scene.bodies;
  const poses = new Array(bodies.length);
  poses[0] = { pos: [0,0,0], rot: [1,0,0, 0,1,0, 0,0,1] };

  if (!scene._jointsByBody) {
    const map = new Map();
    for (const joint of scene.joints) {
      if (!map.has(joint.body)) map.set(joint.body, []);
      map.get(joint.body).push(joint);
    }
    scene._jointsByBody = map;
  }

  for (let i = 1; i < bodies.length; i++) {
    const body = bodies[i];
    const base = poses[body.parent] || poses[0];
    let rot = mat3Multiply(base.rot, quatToMat(body.quat));
    const offset = mat3Apply(base.rot, body.pos);
    let pos = [base.pos[0]+offset[0], base.pos[1]+offset[1], base.pos[2]+offset[2]];

    const joints = scene._jointsByBody.get(i);
    if (joints) {
      for (const joint of joints) {
        if (joint.type === 'free') {
          // A loose object -- the block a grasp probe lifts. Its seven values
          // are a world pose outright, not a displacement of the parent frame.
          const a = joint.qposadr;
          pos = [qpos[a] ?? 0, qpos[a+1] ?? 0, qpos[a+2] ?? 0];
          rot = quatToMat([qpos[a+3] ?? 1, qpos[a+4] ?? 0, qpos[a+5] ?? 0, qpos[a+6] ?? 0]);
          continue;
        }
        const value = (qpos[joint.qposadr] ?? 0) - joint.ref;
        if (joint.type === 'hinge') {
          const spin = axisAngleMat(joint.axis, value);
          const spun = mat3Apply(spin, joint.pos);
          const shift = mat3Apply(rot, [
            joint.pos[0]-spun[0], joint.pos[1]-spun[1], joint.pos[2]-spun[2],
          ]);
          pos = [pos[0]+shift[0], pos[1]+shift[1], pos[2]+shift[2]];
          rot = mat3Multiply(rot, spin);
        } else {
          const slide = mat3Apply(rot, [
            joint.axis[0]*value, joint.axis[1]*value, joint.axis[2]*value,
          ]);
          pos = [pos[0]+slide[0], pos[1]+slide[1], pos[2]+slide[2]];
        }
      }
    }
    poses[i] = { pos, rot };
  }
  return poses;
}

// -- parametric geometry -----------------------------------------------------
// MuJoCo geom types the exporter marks drawable. Sizes follow MuJoCo's
// conventions: box size is a half-extent, capsule/cylinder size is
// (radius, half-length).

function buildBox(hx, hy, hz) {
  const p = [[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
             [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]];
  const faces = [
    [0,3,2,1,[0,0,-1]], [4,5,6,7,[0,0,1]],
    [0,1,5,4,[0,-1,0]], [2,3,7,6,[0,1,0]],
    [1,2,6,5,[1,0,0]],  [3,0,4,7,[-1,0,0]],
  ];
  const vert = [], norm = [], index = [];
  for (const [a,b,c,d,n] of faces) {
    const base = vert.length / 3;
    for (const i of [a,b,c,d]) { vert.push(...p[i]); norm.push(...n); }
    index.push(base, base+1, base+2, base, base+2, base+3);
  }
  return { vert, norm, index };
}

function buildSphereLike(rx, ry, rz, segments = 20, rings = 14) {
  const vert = [], norm = [], index = [];
  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI;
    for (let s = 0; s <= segments; s++) {
      const theta = (s / segments) * Math.PI * 2;
      const nx = Math.sin(phi) * Math.cos(theta);
      const ny = Math.sin(phi) * Math.sin(theta);
      const nz = Math.cos(phi);
      vert.push(nx*rx, ny*ry, nz*rz);
      norm.push(...normalize([nx/rx, ny/ry, nz/rz]));
    }
  }
  const stride = segments + 1;
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r*stride + s, b = a + stride;
      index.push(a, b, a+1, a+1, b, b+1);
    }
  }
  return { vert, norm, index };
}

function buildCylinder(radius, halfLength, capsule, segments = 24) {
  const vert = [], norm = [], index = [];
  // Side wall.
  for (let s = 0; s <= segments; s++) {
    const t = (s / segments) * Math.PI * 2;
    const nx = Math.cos(t), ny = Math.sin(t);
    vert.push(nx*radius, ny*radius, -halfLength); norm.push(nx, ny, 0);
    vert.push(nx*radius, ny*radius,  halfLength); norm.push(nx, ny, 0);
  }
  for (let s = 0; s < segments; s++) {
    const a = s*2;
    index.push(a, a+1, a+2, a+1, a+3, a+2);
  }
  if (capsule) {
    // Hemispherical caps, offset to the ends of the shaft.
    for (const sign of [-1, 1]) {
      const start = vert.length / 3;
      const rings = 8;
      for (let r = 0; r <= rings; r++) {
        const phi = (r / rings) * (Math.PI / 2);
        for (let s = 0; s <= segments; s++) {
          const theta = (s / segments) * Math.PI * 2;
          const nx = Math.cos(phi) * Math.cos(theta);
          const ny = Math.cos(phi) * Math.sin(theta);
          const nz = Math.sin(phi) * sign;
          vert.push(nx*radius, ny*radius, nz*radius + sign*halfLength);
          norm.push(nx, ny, nz);
        }
      }
      const stride = segments + 1;
      for (let r = 0; r < rings; r++) {
        for (let s = 0; s < segments; s++) {
          const a = start + r*stride + s, b = a + stride;
          index.push(a, b, a+1, a+1, b, b+1);
        }
      }
    }
  } else {
    for (const sign of [-1, 1]) {
      const centre = vert.length / 3;
      vert.push(0, 0, sign*halfLength); norm.push(0, 0, sign);
      const start = vert.length / 3;
      for (let s = 0; s <= segments; s++) {
        const t = (s / segments) * Math.PI * 2;
        vert.push(Math.cos(t)*radius, Math.sin(t)*radius, sign*halfLength);
        norm.push(0, 0, sign);
      }
      for (let s = 0; s < segments; s++) {
        if (sign > 0) index.push(centre, start+s, start+s+1);
        else index.push(centre, start+s+1, start+s);
      }
    }
  }
  return { vert, norm, index };
}

function buildPlane(hx, hy) {
  const x = hx > 0 ? hx : 8, y = hy > 0 ? hy : 8;
  return {
    vert: [-x,-y,0, x,-y,0, x,y,0, -x,y,0],
    norm: [0,0,1, 0,0,1, 0,0,1, 0,0,1],
    index: [0,1,2, 0,2,3],
  };
}

function buildMesh(mesh) {
  const vert = mesh.vert.slice();
  const index = mesh.face.slice();
  // Face normals accumulated per vertex: the exporter ships positions and
  // triangles only, and a mesh drawn flat-shaded from face normals looks
  // faceted enough to misread as a different part.
  const norm = new Array(vert.length).fill(0);
  for (let i = 0; i < index.length; i += 3) {
    const [a, b, c] = [index[i]*3, index[i+1]*3, index[i+2]*3];
    const u = [vert[b]-vert[a], vert[b+1]-vert[a+1], vert[b+2]-vert[a+2]];
    const v = [vert[c]-vert[a], vert[c+1]-vert[a+1], vert[c+2]-vert[a+2]];
    const n = cross(u, v);
    for (const base of [a, b, c]) {
      norm[base] += n[0]; norm[base+1] += n[1]; norm[base+2] += n[2];
    }
  }
  for (let i = 0; i < norm.length; i += 3) {
    const n = normalize([norm[i], norm[i+1], norm[i+2]]);
    norm[i] = n[0]; norm[i+1] = n[1]; norm[i+2] = n[2];
  }
  return { vert, norm, index };
}

function geometryFor(scene, geom) {
  const s = geom.size;
  switch (geom.type) {
    case 0: return buildPlane(s[0], s[1]);
    case 2: return buildSphereLike(s[0], s[0], s[0]);
    case 3: return buildCylinder(s[0], s[1], true);
    case 4: return buildSphereLike(s[0], s[1], s[2]);
    case 5: return buildCylinder(s[0], s[1], false);
    case 6: return buildBox(s[0], s[1], s[2]);
    case 7: return buildMesh(scene.meshes[geom.mesh]);
    default: return null;
  }
}

// -- renderer ----------------------------------------------------------------

const VERTEX_SHADER = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vNormal;
out vec3 vWorld;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorld = world.xyz;
  vNormal = mat3(uModel) * aNormal;
  gl_Position = uViewProj * world;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorld;
uniform vec4 uColor;
uniform vec3 uEye;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;
  vec3 key = normalize(vec3(0.45, 0.7, 1.0));
  vec3 fill = normalize(vec3(-0.6, -0.3, 0.35));
  float lambert = max(dot(n, key), 0.0) * 0.78 + max(dot(n, fill), 0.0) * 0.22;
  vec3 view = normalize(uEye - vWorld);
  float spec = pow(max(dot(reflect(-key, n), view), 0.0), 26.0) * 0.28;
  float ambient = 0.30 + 0.10 * (n.z * 0.5 + 0.5);
  outColor = vec4(uColor.rgb * (ambient + lambert) + vec3(spec), uColor.a);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || 'shader failed to compile');
  }
  return shader;
}

class RobotRenderer {
  constructor(canvas, scene) {
    this.canvas = canvas;
    this.scene = scene;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: true });
    if (!gl) throw new Error('WebGL2 is unavailable in this browser');
    this.gl = gl;

    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || 'program failed to link');
    }
    this.program = program;
    this.uModel = gl.getUniformLocation(program, 'uModel');
    this.uViewProj = gl.getUniformLocation(program, 'uViewProj');
    this.uColor = gl.getUniformLocation(program, 'uColor');
    this.uEye = gl.getUniformLocation(program, 'uEye');

    this.parts = [];
    for (const geom of scene.geoms) {
      const geometry = geometryFor(scene, geom);
      if (!geometry) continue;
      this.parts.push({
        geom,
        vao: this._upload(geometry),
        count: geometry.index.length,
        local: mat4Compose(quatToMat(geom.quat), geom.pos),
      });
    }

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  _upload(geometry) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const positions = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positions);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.vert), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const normals = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normals);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(geometry.norm), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const indices = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(geometry.index), gl.STATIC_DRAW);

    gl.bindVertexArray(null);
    return vao;
  }

  // Where the robot sits, so the camera can frame it without being told.
  bounds(qpos) {
    const poses = forwardKinematics(this.scene, qpos);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const part of this.parts) {
      if (part.geom.type === 0) continue;  // an 8 m ground plane is not the subject
      const pose = poses[part.geom.body];
      if (!pose) continue;
      const centre = [
        pose.pos[0] + mat3Apply(pose.rot, part.geom.pos)[0],
        pose.pos[1] + mat3Apply(pose.rot, part.geom.pos)[1],
        pose.pos[2] + mat3Apply(pose.rot, part.geom.pos)[2],
      ];
      const r = Math.max(...part.geom.size.filter(v => v > 0), 0.02);
      for (let i = 0; i < 3; i++) {
        lo[i] = Math.min(lo[i], centre[i] - r);
        hi[i] = Math.max(hi[i], centre[i] + r);
      }
    }
    if (!isFinite(lo[0])) { lo = [-0.5,-0.5,0]; hi = [0.5,0.5,1]; }
    return { lo, hi };
  }

  draw(qpos, camera) {
    const gl = this.gl;
    const canvas = this.canvas;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const poses = forwardKinematics(this.scene, qpos);
    const eye = camera.eye();
    const view = mat4LookAt(eye, camera.target, [0, 0, 1]);
    const proj = mat4Perspective(camera.fov, width / height, camera.near, camera.far);
    const viewProj = mat4Multiply(proj, view);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, viewProj);
    gl.uniform3fv(this.uEye, new Float32Array(eye));

    for (const part of this.parts) {
      const pose = poses[part.geom.body];
      if (!pose) continue;
      const world = mat4Multiply(mat4Compose(pose.rot, pose.pos), part.local);
      gl.uniformMatrix4fv(this.uModel, false, world);
      gl.uniform4fv(this.uColor, new Float32Array(part.geom.rgba));
      gl.bindVertexArray(part.vao);
      gl.drawElements(gl.TRIANGLES, part.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
  }
}

// -- orbit camera ------------------------------------------------------------

class OrbitCamera {
  constructor(target, distance) {
    this.target = target;
    this.distance = distance;
    this.azimuth = 2.3;
    this.elevation = 0.42;
    this.fov = 0.85;
    this.near = 0.01;
    this.far = 200;
  }
  eye() {
    const h = Math.cos(this.elevation) * this.distance;
    return [
      this.target[0] + h * Math.cos(this.azimuth),
      this.target[1] + h * Math.sin(this.azimuth),
      this.target[2] + Math.sin(this.elevation) * this.distance,
    ];
  }
  attach(canvas, onChange) {
    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', (e) => {
      dragging = false;
      try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      this.azimuth -= (e.clientX - lastX) * 0.01;
      this.elevation = Math.max(-1.45, Math.min(1.45,
        this.elevation + (e.clientY - lastY) * 0.01));
      lastX = e.clientX; lastY = e.clientY;
      onChange();
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance = Math.max(0.15, Math.min(80,
        this.distance * Math.exp(e.deltaY * 0.0012)));
      onChange();
    }, { passive: false });
  }
}

// -- playback ----------------------------------------------------------------

function sampleTrack(track, time) {
  const times = track.times, nq = track.nq, values = track.qpos;
  const n = times.length;
  if (n === 0) return [];
  if (time <= times[0]) return values.slice(0, nq);
  if (time >= times[n-1]) return values.slice((n-1)*nq, n*nq);

  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= time) lo = mid; else hi = mid;
  }
  const span = times[hi] - times[lo];
  const t = span > 1e-12 ? (time - times[lo]) / span : 0;
  const out = new Array(nq);
  for (let i = 0; i < nq; i++) {
    const a = values[lo*nq + i], b = values[hi*nq + i];
    out[i] = a + (b - a) * t;
  }
  return out;
}

// Mounts a viewer into `host`. Returns handles the surrounding page uses to
// drive it: a timeline, a joint panel, whatever the tab needs.
function mountViewer(host, scene, options = {}) {
  host.innerHTML = '';
  host.classList.add('viewer');

  const canvas = document.createElement('canvas');
  canvas.className = 'viewer-canvas';
  host.appendChild(canvas);

  let renderer;
  try {
    renderer = new RobotRenderer(canvas, scene);
  } catch (error) {
    host.innerHTML = `<div class="viewer-error">3D playback unavailable: ${error.message}</div>`;
    return null;
  }

  const rest = options.rest || scene.qpos0 || new Array(scene.nq).fill(0);
  let pose = rest.slice();

  const box = renderer.bounds(rest);
  const centre = [
    (box.lo[0]+box.hi[0])/2, (box.lo[1]+box.hi[1])/2, (box.lo[2]+box.hi[2])/2,
  ];
  const extent = Math.max(
    box.hi[0]-box.lo[0], box.hi[1]-box.lo[1], box.hi[2]-box.lo[2], 0.2);
  // Framed from the robot's own extent. 2.1 left an arm looking like a speck in
  // the middle of a large empty box; 1.45 fills the frame while still leaving
  // room for the motion to travel outside the rest pose.
  const camera = new OrbitCamera(centre, Math.max(extent * 1.45, 0.25));

  let needsDraw = true;
  const invalidate = () => { needsDraw = true; };
  camera.attach(canvas, invalidate);

  const state = {
    track: options.track || null,
    time: 0,
    playing: false,
    speed: 1,
    loop: true,
    onTime: options.onTime || null,
  };

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (state.playing && state.track) {
      state.time += dt * state.speed;
      const duration = state.track.duration_s || 0;
      if (state.time >= duration) {
        if (state.loop) state.time = 0;
        else { state.time = duration; state.playing = false; }
      }
      pose = sampleTrack(state.track, state.time);
      if (state.onTime) state.onTime(state.time, state.playing);
      needsDraw = true;
    }
    if (needsDraw) {
      renderer.draw(pose, camera);
      needsDraw = false;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const observer = new ResizeObserver(invalidate);
  observer.observe(canvas);

  return {
    canvas,
    camera,
    scene,
    get time() { return state.time; },
    get playing() { return state.playing; },
    get track() { return state.track; },
    setTrack(track) {
      state.track = track;
      state.time = 0;
      pose = track ? sampleTrack(track, 0) : rest.slice();
      if (state.onTime) state.onTime(0, state.playing);
      invalidate();
    },
    seek(time) {
      if (!state.track) return;
      state.time = Math.max(0, Math.min(state.track.duration_s || 0, time));
      pose = sampleTrack(state.track, state.time);
      if (state.onTime) state.onTime(state.time, state.playing);
      invalidate();
    },
    play() { if (state.track) { state.playing = true; last = performance.now(); } },
    pause() { state.playing = false; },
    toggle() { state.playing ? this.pause() : this.play(); },
    setSpeed(value) { state.speed = value; },
    setLoop(value) { state.loop = value; },
    setPose(values) { pose = values.slice(); state.track = null; invalidate(); },
    resetPose() { pose = rest.slice(); invalidate(); },
    frameCamera() {
      const b = renderer.bounds(pose);
      camera.target = [(b.lo[0]+b.hi[0])/2, (b.lo[1]+b.hi[1])/2, (b.lo[2]+b.hi[2])/2];
      invalidate();
    },
    invalidate,
  };
}
