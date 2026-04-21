"""Viewer for K_12 result that highlights the remaining intersecting
(edge, triangle) pairs in red."""
import json
import sys
from itertools import combinations

import numpy as np


def signed_vol6(A, B, C, D):
    return np.dot(B - A, np.cross(C - A, D - A))


def crosses(P, Q, A, B, C, tol=1e-9):
    vp = signed_vol6(P, A, B, C); vq = signed_vol6(Q, A, B, C)
    vab = signed_vol6(P, Q, A, B); vbc = signed_vol6(P, Q, B, C); vca = signed_vol6(P, Q, C, A)
    plane = vp * vq < -tol
    inside = (vab > tol and vbc > tol and vca > tol) or \
             (vab < -tol and vbc < -tol and vca < -tol)
    return plane and inside


def find_intersections(verts, faces):
    N = len(verts)
    EDGES = list(combinations(range(N), 2))
    bad_faces = set()
    bad_edges = set()
    pairs = []
    for face in faces:
        fset = set(face)
        A, B, C = verts[face[0]], verts[face[1]], verts[face[2]]
        for e in EDGES:
            if fset & set(e): continue
            P, Q = verts[e[0]], verts[e[1]]
            if crosses(P, Q, A, B, C):
                bad_faces.add(tuple(face))
                bad_edges.add(e)
                pairs.append([list(face), list(e)])
    return bad_faces, bad_edges, pairs


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>K_12 genus-6 attempt</title>
<style>
  html,body { margin:0; height:100%; background:#111; color:#ccc; font-family:system-ui,sans-serif; }
  #hud { position:fixed; top:10px; left:12px; font-size:13px; opacity:.85; }
  #hud b { color:#ffcc33; }
  button { background:#333; color:#ccc; border:1px solid #555; padding:3px 8px; margin-right:4px; cursor:pointer; }
</style>
</head>
<body>
<div id="hud">
  <div><b>K_12 genus-6 pseudo-manifold</b> &mdash; __NIX__ self-intersections</div>
  <div>red edges = pierce at least one face; red faces = pierced by at least one edge</div>
  <div style="margin-top:4px;">
    <button onclick="setView('both')">faces+wire</button>
    <button onclick="setView('wire')">wire only</button>
    <button onclick="setView('faces')">faces only</button>
    <button onclick="toggleSpin()">spin</button>
  </div>
</div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const VERTS = __VERTS__;
const FACES = __FACES__;
const BAD_FACES = new Set(__BAD_FACES__.map(f => f.slice().sort().join(',')));
const BAD_EDGES = new Set(__BAD_EDGES__.map(e => e.slice().sort().join(',')));

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 100);
camera.position.set(3,2.2,3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const k = new THREE.DirectionalLight(0xffffff, 0.9); k.position.set(3,5,2); scene.add(k);
const f = new THREE.DirectionalLight(0xaaccff, 0.4); f.position.set(-2,-1,-3); scene.add(f);

// Split faces into two buffers: ok and bad.
function buildFaces(isBad) {
  const list = FACES.filter(f => BAD_FACES.has(f.slice().sort().join(',')) === isBad);
  if (!list.length) return null;
  const pos = new Float32Array(list.length * 9);
  for (let i=0; i<list.length; i++) {
    const fc = list[i];
    for (let j=0; j<3; j++) {
      const v = VERTS[fc[j]];
      pos[i*9+j*3] = v[0]; pos[i*9+j*3+1] = v[1]; pos[i*9+j*3+2] = v[2];
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshPhongMaterial({
    color: isBad ? 0xff5555 : 0x66aaff, opacity: isBad ? 0.8 : 0.45,
    transparent: true, side: THREE.DoubleSide, flatShading: true
  }));
}
const okFaces = buildFaces(false);
const badFaces = buildFaces(true);
if (okFaces) scene.add(okFaces);
if (badFaces) scene.add(badFaces);

// Edges (K_12 = 66 edges total)
const edgeSet = new Set();
for (const f of FACES) for (const [a,b] of [[f[0],f[1]],[f[1],f[2]],[f[0],f[2]]]) {
  edgeSet.add(a<b ? `${a},${b}` : `${b},${a}`);
}
const okEdgePos = [], badEdgePos = [];
for (const key of edgeSet) {
  const [a,b] = key.split(',').map(Number);
  const target = BAD_EDGES.has(key) ? badEdgePos : okEdgePos;
  target.push(...VERTS[a], ...VERTS[b]);
}
function lines(pos, color) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
}
const okEdges = lines(okEdgePos, 0xffffff);
const badEdges = lines(badEdgePos, 0xff3333);
okEdges.material.linewidth = 1;
badEdges.material.linewidth = 3;
scene.add(okEdges); scene.add(badEdges);

// Vertices
const vg = new THREE.Group();
const sg = new THREE.SphereGeometry(0.035, 16, 16);
for (let i=0; i<VERTS.length; i++) {
  const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
  m.position.fromArray(VERTS[i]); vg.add(m);
}
scene.add(vg);

let mode='both', spinning=false;
window.setView = (m) => { mode=m;
  if (okFaces) okFaces.visible = (m!=='wire');
  if (badFaces) badFaces.visible = (m!=='wire');
  okEdges.visible = (m!=='faces');
  badEdges.visible = (m!=='faces');
};
window.toggleSpin = () => { spinning = !spinning; };
addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
(function tick() {
  requestAnimationFrame(tick);
  if (spinning) scene.rotation.y += 0.004;
  controls.update();
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "n12.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "n12_viewer.html"
    d = json.load(open(src))
    verts = np.asarray(d["vertices"], dtype=np.float64)
    faces = [tuple(f) for f in d["faces"]]
    bad_faces, bad_edges, pairs = find_intersections(verts, faces)
    print(f"{src}: {len(pairs)} intersecting (edge, triangle) pairs, "
          f"{len(bad_edges)} distinct bad edges, {len(bad_faces)} distinct bad faces")
    html = (HTML
            .replace("__VERTS__", json.dumps(verts.tolist()))
            .replace("__FACES__", json.dumps([list(f) for f in faces]))
            .replace("__BAD_FACES__", json.dumps([list(f) for f in bad_faces]))
            .replace("__BAD_EDGES__", json.dumps([list(e) for e in bad_edges]))
            .replace("__NIX__", str(len(pairs))))
    open(dst, "w").write(html)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
