"""Weight-scan for the polish step, with an interactive result viewer.

Given an already-computed polyhedron (from csaszar.json), this script runs the
polish optimizer under a grid of (w_dihedral, w_collinear) weights in a single
batched pass, and emits an HTML page with a scatter plot (min edge-pair angle
vs max dihedral) where clicking a point shows the corresponding polyhedron in
3D. The dihedral loss here is SYMMETRIC — cos^2(dihedral) so both creased
(dihedral -> 0) and flat (dihedral -> pi) pairs are penalized equally.

Usage:
  JAX_PLATFORMS=cpu python scan_polish.py [--input csaszar.json]
                                           [--steps 1500] [--grid 7]
"""

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

from csaszar import (
    N,
    build_pair_index, build_dihedral_index, build_edge_pair_index,
    make_loss, dihedral_cos, edge_pair_cos2,
    pca_normalize_batch, count_real_intersections,
)


def make_symmetric_polish_loss(face_idx, edge_idx, dih_idx, ep_idx):
    """Loss: intersection penalty + w_dih * smooth_max cos^2(dihedral)
                                   + w_coll * smooth_max cos^2(edge-pair).
    Using cos^2 instead of cos means we push dihedrals toward pi/2 from both
    sides (so pairs of faces are neither creased nor flattened), and edge pairs
    toward pi/2 (so no three vertices become collinear)."""
    intersect = make_loss(face_idx, edge_idx)

    def loss(V, tau_ix, tau_dih, tau_coll, w_dih, w_coll):
        l_ix = intersect(V, tau_ix)
        cos_d = dihedral_cos(V, dih_idx)
        cos2_d = cos_d * cos_d
        smx_d = tau_dih * jax.scipy.special.logsumexp(cos2_d / tau_dih)
        cos2_ep = edge_pair_cos2(V, ep_idx)
        smx_e = tau_coll * jax.scipy.special.logsumexp(cos2_ep / tau_coll)
        return l_ix + w_dih * smx_d + w_coll * smx_e
    return loss


def scan(V0, faces, weight_pairs, steps=1500, lr=1e-3,
         tau_ix=5e-4, tau_dih_start=0.3, tau_dih_end=0.03,
         tau_coll_start=0.3, tau_coll_end=0.03,
         chunk_size=150, log_every=300, use_scan=True):
    face_idx, edge_idx = build_pair_index(faces)
    dih_idx = build_dihedral_index(faces)
    ep_idx = build_edge_pair_index(N)

    loss_fn = make_symmetric_polish_loss(face_idx, edge_idx, dih_idx, ep_idx)
    grad_fn = jax.grad(loss_fn)

    B = len(weight_pairs)
    V0_np = np.asarray(V0, dtype=np.float32)
    V_init = np.broadcast_to(V0_np[None], (B, N, 3)).copy()
    w_dih_arr = jnp.asarray([w[0] for w in weight_pairs], dtype=jnp.float32)
    w_coll_arr = jnp.asarray([w[1] for w in weight_pairs], dtype=jnp.float32)

    beta1, beta2, ae = 0.9, 0.999, 1e-8

    def one_step(V, m, s, step_num, td, tc):
        g = jax.vmap(grad_fn, in_axes=(0, None, None, None, 0, 0))(
            V, tau_ix, td, tc, w_dih_arr, w_coll_arr)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s

    one_step_jit = jax.jit(one_step)

    @jax.jit
    def run_chunk_scan(V, m, s, start_f, tdc, tcc):
        def body(carry, xs):
            V_, m_, s_ = carry
            sn, td, tc = xs
            return one_step(V_, m_, s_, sn, td, tc), None
        step_nums = start_f + jnp.arange(1, tdc.shape[0] + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s), (step_nums, tdc, tcc))
        return final

    def run_chunk_loop(V, m, s, start, tdc, tcc):
        for i in range(tdc.shape[0]):
            V, m, s = one_step_jit(V, m, s,
                                   jnp.float32(start + i + 1),
                                   tdc[i], tcc[i])
        return V, m, s

    rchunk = run_chunk_scan if use_scan else run_chunk_loop

    V = pca_normalize_batch(V_init)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)

    tds = (tau_dih_start *
           (tau_dih_end / tau_dih_start) ** np.linspace(0, 1, steps)).astype(np.float32)
    tcs = (tau_coll_start *
           (tau_coll_end / tau_coll_start) ** np.linspace(0, 1, steps)).astype(np.float32)

    done = 0
    while done < steps:
        k = min(chunk_size, steps - done)
        tdc = jnp.asarray(tds[done: done + k])
        tcc = jnp.asarray(tcs[done: done + k])
        if use_scan:
            V, m, s = rchunk(V, m, s, jnp.float32(done), tdc, tcc)
        else:
            V, m, s = rchunk(V, m, s, done, tdc, tcc)
        V = pca_normalize_batch(V)
        done += k
        if done % log_every < chunk_size:
            V_np = np.asarray(V)
            n_clean = sum(count_real_intersections(V_np[i], faces)[0] == 0 for i in range(B))
            print(f"  step {done:5d}  clean={n_clean}/{B}")

    V_np = np.asarray(V)
    out = []
    for i, (wd, wc) in enumerate(weight_pairs):
        v = V_np[i]
        n_ix, _ = count_real_intersections(v, faces)
        cos_d = np.clip(np.asarray(dihedral_cos(v, dih_idx)), -1.0, 1.0)
        ang_d = np.degrees(np.arccos(cos_d))
        cos2_ep = np.clip(np.asarray(edge_pair_cos2(v, ep_idx)), 0.0, 1.0)
        dev_ep = np.degrees(np.arcsin(np.sqrt(np.clip(1.0 - cos2_ep, 0.0, 1.0))))
        out.append({
            "w_dihedral": float(wd),
            "w_collinear": float(wc),
            "vertices": v.tolist(),
            "real_intersections": int(n_ix),
            "min_dihedral": float(ang_d.min()),
            "max_dihedral": float(ang_d.max()),
            "mean_dihedral": float(ang_d.mean()),
            "min_edge_pair_dev": float(dev_ep.min()),
            "mean_edge_pair_dev": float(dev_ep.mean()),
        })
    return out


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Csaszar polish scan</title>
<style>
  html,body { margin:0; height:100%; overflow:hidden; background:#111; color:#ccc; font-family:system-ui,sans-serif; }
  #app { display:flex; height:100%; }
  #plotcol { flex:1; min-width:0; padding:6px; box-sizing:border-box; }
  #plot { width:100%; height:100%; }
  #viewcol { flex:1; position:relative; border-left:1px solid #333; }
  #cv { width:100%; height:100%; display:block; }
  #info { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.72); padding:8px 10px; border-radius:4px; font-size:12px; pointer-events:none; line-height:1.35; }
  #info b { color:#ffcc33; }
  #buttons { position:absolute; bottom:10px; left:10px; font-size:12px; }
  button { background:#333; color:#ccc; border:1px solid #555; padding:3px 8px; margin-right:4px; cursor:pointer; }
</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
</head>
<body>
<div id="app">
  <div id="plotcol"><div id="plot"></div></div>
  <div id="viewcol">
    <canvas id="cv"></canvas>
    <div id="info">click a point &larr;</div>
    <div id="buttons">
      <button onclick="setView('both')">faces+wire</button>
      <button onclick="setView('faces')">faces only</button>
      <button onclick="setView('wire')">wire only</button>
      <button onclick="toggleSpin()">spin</button>
    </div>
  </div>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const RESULTS = __RESULTS__;
const FACES = __FACES__;

// ---------------- scatter plot ----------------
const clean = RESULTS.map(r => r.real_intersections === 0);
const xs = RESULTS.map(r => r.min_edge_pair_dev);
const ys = RESULTS.map(r => r.max_dihedral);
const colors = RESULTS.map(r => r.min_dihedral);
const text = RESULTS.map((r, i) =>
  `<b>run ${i}</b>&nbsp;&nbsp;w_dih=${r.w_dihedral.toFixed(3)}&nbsp;&nbsp;w_coll=${r.w_collinear.toFixed(3)}<br>` +
  `min edge-pair dev = <b>${r.min_edge_pair_dev.toFixed(1)}°</b><br>` +
  `dihedrals: ${r.min_dihedral.toFixed(1)}° .. ${r.max_dihedral.toFixed(1)}° (mean ${r.mean_dihedral.toFixed(1)}°)<br>` +
  `real intersections: ${r.real_intersections}`);

const symbols = clean.map(c => c ? 'circle' : 'x');

const trace = {
  x: xs, y: ys,
  text: text,
  mode: 'markers',
  type: 'scatter',
  hovertemplate: '%{text}<extra></extra>',
  marker: {
    size: 13,
    color: colors,
    colorscale: 'Viridis',
    showscale: true,
    cmin: 0,
    symbol: symbols,
    colorbar: {
      title: { text: 'min<br>dihedral°', font: { color: '#ccc' } },
      tickfont: { color: '#ccc' },
      outlinewidth: 0
    },
    line: { color: '#333', width: 0.5 },
  },
};

const layout = {
  title: { text: 'Polish weight scan — sweet spot: high x AND low y', font: { color: '#ddd' } },
  xaxis: { title: { text: 'min edge-pair angle from collinearity  (°, higher = better)', font: { color: '#ccc' } }, gridcolor: '#333', zerolinecolor: '#555', tickfont: { color: '#ccc' } },
  yaxis: { title: { text: 'max dihedral angle  (°, lower = better, away from 180°)', font: { color: '#ccc' } }, gridcolor: '#333', zerolinecolor: '#555', tickfont: { color: '#ccc' }, range: [0, 181] },
  paper_bgcolor: '#111', plot_bgcolor: '#1a1a1a', font: { color: '#ccc' },
  margin: { t: 40, r: 10, b: 52, l: 60 },
  hoverlabel: { bgcolor: '#222', bordercolor: '#444', font: { color: '#ccc' } },
};

const plotDiv = document.getElementById('plot');
Plotly.newPlot(plotDiv, [trace], layout, { responsive: true }).then(() => {
  plotDiv.on('plotly_click', (data) => {
    const i = data.points[0].pointIndex;
    selectRun(i);
  });
});

// ---------------- three.js ----------------
const cv = document.getElementById('cv');
const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2.9, 2.1, 2.9);
const controls = new OrbitControls(camera, cv);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(3,5,2); scene.add(key);
const fill = new THREE.DirectionalLight(0xaaccff, 0.4); fill.position.set(-2,-1,-3); scene.add(fill);

const root = new THREE.Group(); scene.add(root);
let faceMesh = null, edgeLines = null, vertGroup = null;
let mode = 'both', spinning = false;

function setPoly(verts) {
  while (root.children.length) {
    const c = root.children.pop();
    c.traverse?.(n => { n.geometry?.dispose?.(); n.material?.dispose?.(); });
    c.geometry?.dispose?.();
    c.material?.dispose?.();
  }
  const pos = new Float32Array(FACES.length * 9);
  for (let i = 0; i < FACES.length; i++) {
    const f = FACES[i];
    for (let j = 0; j < 3; j++) {
      const v = verts[f[j]];
      pos[i*9+j*3] = v[0]; pos[i*9+j*3+1] = v[1]; pos[i*9+j*3+2] = v[2];
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.computeVertexNormals();
  faceMesh = new THREE.Mesh(geom, new THREE.MeshPhongMaterial({
    color: 0x66aaff, opacity: 0.6, transparent: true, side: THREE.DoubleSide, flatShading: true
  }));
  root.add(faceMesh);

  const edgeSet = new Set();
  for (const f of FACES) for (const [a,b] of [[f[0],f[1]],[f[1],f[2]],[f[0],f[2]]]) {
    edgeSet.add(a < b ? `${a},${b}` : `${b},${a}`);
  }
  const ep = [];
  for (const k of edgeSet) {
    const [a,b] = k.split(',').map(Number);
    ep.push(...verts[a], ...verts[b]);
  }
  const eg = new THREE.BufferGeometry();
  eg.setAttribute('position', new THREE.Float32BufferAttribute(ep, 3));
  edgeLines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0xffffff }));
  root.add(edgeLines);

  vertGroup = new THREE.Group();
  const sg = new THREE.SphereGeometry(0.03, 16, 16);
  for (const v of verts) {
    const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
    m.position.fromArray(v);
    vertGroup.add(m);
  }
  root.add(vertGroup);
  applyMode();
}

function applyMode() {
  if (!faceMesh) return;
  faceMesh.visible = (mode !== 'wire');
  edgeLines.visible = (mode !== 'faces');
}

window.setView = (m) => { mode = m; applyMode(); };
window.toggleSpin = () => { spinning = !spinning; };

function resize() {
  const c = document.getElementById('viewcol');
  const w = c.clientWidth, h = c.clientHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', resize);
setTimeout(resize, 0);

function selectRun(i) {
  const r = RESULTS[i];
  setPoly(r.vertices);
  document.getElementById('info').innerHTML =
    `<b>run ${i}</b><br>` +
    `w_dihedral = ${r.w_dihedral.toFixed(3)}<br>` +
    `w_collinear = ${r.w_collinear.toFixed(3)}<br>` +
    `min edge-pair dev = ${r.min_edge_pair_dev.toFixed(2)}°<br>` +
    `dihedral: ${r.min_dihedral.toFixed(1)}° .. ${r.max_dihedral.toFixed(1)}° (mean ${r.mean_dihedral.toFixed(1)}°)<br>` +
    (r.real_intersections > 0 ? `<span style="color:#f77">intersections: ${r.real_intersections}</span>` : `intersections: 0`);
}

// pick a reasonable default: maximize min_edge_pair_dev - 0.5 * max_dihedral^2... actually
// prefer a clean run with highest (x - 0.2*y) as a proxy for "sweet spot".
let bestI = 0, bestScore = -Infinity;
for (let i = 0; i < RESULTS.length; i++) {
  if (RESULTS[i].real_intersections !== 0) continue;
  const s = RESULTS[i].min_edge_pair_dev - 0.2 * Math.max(0, RESULTS[i].max_dihedral - 120);
  if (s > bestScore) { bestScore = s; bestI = i; }
}
selectRun(bestI);

function tick() {
  requestAnimationFrame(tick);
  if (spinning) root.rotation.y += 0.004;
  controls.update();
  renderer.render(scene, camera);
}
tick();
</script>
</body>
</html>
"""


def make_html(results, faces, out_path):
    html = (HTML
            .replace("__RESULTS__", json.dumps(results))
            .replace("__FACES__", json.dumps([list(f) for f in faces])))
    open(out_path, "w").write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="csaszar.json",
                    help="JSON with 'vertices' and 'faces' keys")
    ap.add_argument("--output", default="scan_viewer.html")
    ap.add_argument("--results-json", default="scan_results.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--grid", type=int, default=7,
                    help="N for an NxN grid of (w_dih, w_coll) weights")
    ap.add_argument("--w-min", type=float, default=0.03)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk-size", type=int, default=150)
    ap.add_argument("--no-scan", action="store_true")
    args = ap.parse_args()

    print(f"JAX backend: {jax.devices()}")

    data = json.load(open(args.input))
    V0 = np.asarray(data["vertices"], dtype=np.float32)
    faces = [tuple(f) for f in data["faces"]]
    print(f"loaded {args.input}: {V0.shape[0]} vertices, {len(faces)} faces")

    grid_vals = np.geomspace(args.w_min, args.w_max, args.grid).astype(np.float32)
    weight_pairs = [(float(wd), float(wc)) for wd in grid_vals for wc in grid_vals]
    print(f"running {len(weight_pairs)} configurations "
          f"(w grid {grid_vals.tolist()})")

    t0 = time.time()
    results = scan(V0, faces, weight_pairs,
                   steps=args.steps, lr=args.lr,
                   chunk_size=args.chunk_size,
                   use_scan=not args.no_scan)
    dt = time.time() - t0
    print(f"scan done in {dt:.1f}s")

    clean_results = [r for r in results if r["real_intersections"] == 0]
    print(f"\nclean runs: {len(clean_results)}/{len(results)}")
    if clean_results:
        best_coll = max(clean_results, key=lambda r: r["min_edge_pair_dev"])
        best_dih = min(clean_results, key=lambda r: r["max_dihedral"])
        print(f"  best min-edge-pair-dev:    {best_coll['min_edge_pair_dev']:.2f}°  "
              f"(w_dih={best_coll['w_dihedral']:.2f}, w_coll={best_coll['w_collinear']:.2f}, "
              f"max_dih={best_coll['max_dihedral']:.1f}°)")
        print(f"  lowest max-dihedral:       {best_dih['max_dihedral']:.2f}°  "
              f"(w_dih={best_dih['w_dihedral']:.2f}, w_coll={best_dih['w_collinear']:.2f}, "
              f"min_ep={best_dih['min_edge_pair_dev']:.1f}°)")

    with open(args.results_json, "w") as fh:
        json.dump({"faces": [list(f) for f in faces], "results": results}, fh)
    make_html(results, faces, args.output)
    print(f"\nwrote {args.results_json} and {args.output}")
    print(f"open {args.output} in a browser and click points to explore")


if __name__ == "__main__":
    main()
