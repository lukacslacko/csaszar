"""Generate a standalone HTML viewer for the Csaszar polyhedron.

Reads csaszar.json (produced by csaszar.py) and writes viewer.html with the
vertex and face data inlined. Open viewer.html in any browser to pan / rotate
the polyhedron (click-drag = orbit, right-drag = pan, scroll = zoom).
"""
import json
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "csaszar.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "viewer.html"

data = json.load(open(src))
verts = data["vertices"]
faces = data["faces"]

html = """<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Csaszar polyhedron</title>
<style>
  html,body { margin:0; height:100%; background:#111; color:#ccc; font-family:system-ui,sans-serif; }
  #hud { position:fixed; top:10px; left:12px; font-size:13px; opacity:.8; }
  button { background:#333; color:#ccc; border:1px solid #555; padding:3px 8px; margin-right:4px; cursor:pointer; }
</style>
</head><body>
<div id="hud">
  <div>Csaszar polyhedron &mdash; drag to rotate, scroll to zoom</div>
  <div style="margin-top:4px;">
    <button onclick="setView('faces')">faces</button>
    <button onclick="setView('wire')">wireframe</button>
    <button onclick="setView('both')">both</button>
    <button onclick="toggleVerts()">vertices</button>
    <button onclick="toggleSpin()">spin</button>
  </div>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const VERTS = __VERTS__;
const FACES = __FACES__;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 100);
camera.position.set(3, 2.2, 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(3, 5, 2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xaaccff, 0.4);
fill.position.set(-2, -1, -3);
scene.add(fill);

// ---- face mesh (translucent, double-sided) ----
const posArr = new Float32Array(FACES.length * 9);
for (let i = 0; i < FACES.length; i++) {
  const f = FACES[i];
  for (let j = 0; j < 3; j++) {
    const v = VERTS[f[j]];
    posArr[i*9 + j*3    ] = v[0];
    posArr[i*9 + j*3 + 1] = v[1];
    posArr[i*9 + j*3 + 2] = v[2];
  }
}
const faceGeom = new THREE.BufferGeometry();
faceGeom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
faceGeom.computeVertexNormals();
const faceMat = new THREE.MeshPhongMaterial({
  color: 0x66aaff, opacity: 0.6, transparent: true,
  side: THREE.DoubleSide, flatShading: true,
});
const faceMesh = new THREE.Mesh(faceGeom, faceMat);
scene.add(faceMesh);

// ---- all 21 edges (K_7) in white ----
const edgeSet = new Set();
for (const f of FACES) {
  for (const [a,b] of [[f[0],f[1]],[f[1],f[2]],[f[0],f[2]]]) {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    edgeSet.add(key);
  }
}
const edgePos = [];
for (const k of edgeSet) {
  const [a,b] = k.split(',').map(Number);
  edgePos.push(...VERTS[a], ...VERTS[b]);
}
const edgeGeom = new THREE.BufferGeometry();
edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff });
const edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
scene.add(edgeLines);

// ---- vertex spheres with labels ----
const vertGroup = new THREE.Group();
const sphereGeom = new THREE.SphereGeometry(0.035, 16, 16);
for (let i = 0; i < VERTS.length; i++) {
  const m = new THREE.Mesh(sphereGeom,
    new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
  m.position.fromArray(VERTS[i]);
  vertGroup.add(m);
}
scene.add(vertGroup);
vertGroup.visible = true;

// ---- controls ----
let mode = 'both';
let spinning = false;
window.setView = function(m) {
  mode = m;
  faceMesh.visible = (m !== 'wire');
  edgeLines.visible = (m !== 'faces');
};
window.toggleVerts = () => { vertGroup.visible = !vertGroup.visible; };
window.toggleSpin = () => { spinning = !spinning; };

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

function tick() {
  requestAnimationFrame(tick);
  if (spinning) {
    scene.rotation.y += 0.004;
  }
  controls.update();
  renderer.render(scene, camera);
}
tick();
</script>
</body></html>
"""

html = html.replace("__VERTS__", json.dumps(verts))
html = html.replace("__FACES__", json.dumps(faces))
open(dst, "w").write(html)
print(f"wrote {dst}  (open it in a browser)")
