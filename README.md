# Csaszar polyhedron — rediscovery by optimization

Rediscover the [Csaszar polyhedron](https://en.wikipedia.org/wiki/Cs%C3%A1sz%C3%A1r_polyhedron) from scratch: place 7 points at random, eagerly pick 14 triangular faces that look like a torus triangulation with no edge-through-face crossings, then optimize vertex positions in parallel with JAX + Adam until the embedding is clean. An optional second phase polishes the shape so faces aren't creased or flattened and no three vertices become collinear. An interactive scatter plot lets you scan the polish tradeoff and pick your favourite shape.

Every pair of the 7 vertices is connected by an edge (the surface is *neighborly*), the polyhedron has Euler characteristic V − E + F = 7 − 21 + 14 = 0, and it is topologically a torus — the Csaszar polyhedron is the only known non-tetrahedral polyhedron with these properties.

## Install

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The code uses [JAX](https://docs.jax.dev). On Apple Silicon you can install `jax-metal` to run on the GPU, but for this 21-DoF-per-instance problem CPU is faster because per-op dispatch dominates. The snippets below set `JAX_PLATFORMS=cpu` explicitly.

## 1. Generate + polish a polyhedron

```sh
JAX_PLATFORMS=cpu python csaszar.py --batch 256 --steps 2000 --polish-steps 2000
```

What happens:

1. Greedy face selection finds a valid 14-triangle manifold triangulation of the torus (random 7-point cloud, each triangle scored by how many non-incident segments pierce it, picked in increasing order subject to every edge ending up with exactly 2 faces and every vertex link being a cycle).
2. Phase 1 — intersection removal. 256 independent random initializations optimize a smooth interior-intersection penalty (product-of-sigmoids replaced with a `softplus(smooth_min(...))` margin so gradients don't die for deep crossings). Between chunks of Adam steps, vertex coordinates are PCA-normalized (center, rotate to principal axes, scale each axis to [−1, 1]) to keep the optimization well-scaled and stop the polyhedron from collapsing onto a plane.
3. Phase 2 — polish. Over every clean instance from phase 1, a combined objective runs in parallel: the intersection penalty is kept active at a sharp fixed τ, a smooth-max over cos(dihedral) pushes the smallest dihedral up, and a smooth-max over cos²(angle) for every pair of edges sharing a vertex pushes each pair away from collinearity. The best polished instance (by `min(min_dihedral, min_edge_pair_deviation)`) is saved.

Outputs: `csaszar_vertices.npy`, `csaszar_faces.npy`, `csaszar.obj`, `csaszar.json`.

Useful flags: `--no-polish`, `--polish-weight-dihedral`, `--polish-weight-collinear`, `--polish-steps`, `--no-scan` (replaces `lax.scan` with a Python loop — needed on jax-metal).

## 2. View the saved polyhedron

```sh
python make_viewer.py        # writes viewer.html with csaszar.json inlined
open viewer.html             # orbit / zoom with the mouse, buttons for view modes
```

The OBJ works in any 3D viewer (Blender, MeshLab).

## 3. Explore the polish tradeoff — interactive scatter

The default polish uses `cos(dihedral)` which pushes dihedrals toward π (flat). If that flattens faces too much or collapses edges toward collinearity, use the scanning tool. It re-polishes an existing polyhedron under a grid of weights using a **symmetric** `cos²(dihedral)` loss (so both creased and flattened face pairs are penalized) plus the same `cos²(edge-pair)` collinearity term.

```sh
JAX_PLATFORMS=cpu python scan_polish.py --steps 1500 --grid 7
open scan_viewer.html
```

The scatter plot shows every run as one dot:
- **x** = minimum angle any edge-pair-at-a-vertex makes with collinearity (°) — higher is better.
- **y** = maximum dihedral angle (°) — lower is better, away from 180°.
- **color** = minimum dihedral angle (°) — avoid dark points, those have creased face pairs.
- **shape** = ○ for clean, × for has intersections.

Click any point to render that polyhedron on the right. The info panel shows the weights and angle stats. Defaults pick a clean run with a reasonable sweet-spot score.

Flags: `--input any.json`, `--grid N` (N×N grid of weights), `--w-min / --w-max` (log-spaced grid range), `--steps`, `--lr`, `--no-scan`.

Pick your favourite and write it back as the canonical output:

```sh
python - <<'PY'
import json, numpy as np
d = json.load(open('scan_results.json'))
faces = [tuple(f) for f in d['faces']]
# choose by weight (approximate match)
w_dih, w_coll = 0.949, 0.300
r = min(d['results'], key=lambda r: (r['w_dihedral']-w_dih)**2 + (r['w_collinear']-w_coll)**2)
v = np.asarray(r['vertices'], dtype=np.float32)
np.save('csaszar_vertices.npy', v)
np.save('csaszar_faces.npy', np.asarray(faces, dtype=np.int32))
with open('csaszar.obj','w') as fh:
    for p in v: fh.write(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    for f in faces: fh.write(f'f {f[0]+1} {f[1]+1} {f[2]+1}\n')
json.dump({'vertices': v.tolist(), 'faces':[list(f) for f in faces], **r}, open('csaszar.json','w'), indent=2)
PY
python make_viewer.py && open viewer.html
```

## Files

| file              | purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `csaszar.py`      | greedy face selection + two-phase parallel optimization (fixed V=7)     |
| `neighborly.py`   | generalized phase-1-only pipeline parameterized by `--n V`              |
| `scan_polish.py`  | weight-grid scan of the symmetric polish loss, writes `scan_viewer.html` |
| `make_viewer.py`  | render a `*.json` into a standalone HTML viewer                         |
| `csaszar.obj`     | OBJ export of the saved polyhedron                                      |
| `csaszar.json`    | vertex coordinates + faces + angle statistics                           |

## Notes on the objectives

- **Interior intersection predicate.** Segment PQ crosses triangle ABC in its interior iff the three products (−v_P · v_Q), (v_{AB} · v_{BC}), (v_{BC} · v_{CA}) of signed tetrahedron volumes are all positive. No division, so stable. The smooth loss is `τ · softplus(smooth_min(...)/τ)` which behaves linearly in the magnitude of the violation (non-vanishing gradient for deep crossings) and turns into a hard zero cleanly once any one of the three products flips sign.
- **Dihedral objective.** Default `csaszar.py` polish uses the smooth-max over `cos(dihedral)` to push the smallest dihedral up. The scan script uses `cos²(dihedral)` so both extremes (creased and flattened) are penalized symmetrically, with a minimum at π/2.
- **Collinearity objective.** For every (vertex, pair-of-incident-edges) triple — 7 · C(6, 2) = 105 in total — penalize cos² of the angle between the two edge directions. This prevents 3+ vertices from lining up and prevents vertex links from degenerating.
- **Gauge fix.** The intersection and angle objectives are affine-invariant, so the polyhedron can drift in scale/rotation without changing the loss. Between chunks of Adam steps we therefore run a PCA normalization (center, rotate to principal axes, scale each axis to [−1, 1]) — a projection onto a canonical gauge that also prevents the polyhedron from degenerating onto a plane.

## Bonus: the next rung — V=12, genus 6

`neighborly.py` is a generalized version of the pipeline that takes `--n V`. After V=7 (Csaszar, genus 1) the next vertex count for which `K_V` can be neighborly-triangulated on an orientable surface is **V=12**, giving `E=66`, `F=44`, `chi=-10`, **genus 6** (six holes).

```sh
JAX_PLATFORMS=cpu python neighborly.py --n 12 --batch 256 --steps 4000 \
    --structure-tries 500 --allow-pseudomanifold --out n12
python make_viewer.py n12.json n12_viewer.html && open n12_viewer.html
```

Status as of this repo: **open problem, not solved**. Manifold triangulations of `K_12` on genus 6 exist combinatorially (Ringel) but are a tiny sliver of the ways to pick 44 triangles with each edge in 2 faces — the greedy sampler from `csaszar.py` finds 44-face pseudo-manifolds (pinched vertices) roughly 3% of the time and true 2-manifolds essentially never. `--allow-pseudomanifold` lets you proceed with pinched face sets anyway; parallel Adam drives those down to **27–33 self-intersections** depending on the specific combinatorial structure, but never to 0 — the pinched-vertex cones cannot separate geometrically. Running the same pipeline against a true manifold triangulation (not produced by this repo) is the next step for someone who wants to push on this.

The default `neighborly.py --n 7` reproduces the Csaszar flow end-to-end.
