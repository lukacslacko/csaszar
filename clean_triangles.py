"""Minimize edge-triangle interior intersections across ALL possible
(edge, non-incident-triangle) pairs, regardless of face selection.

For N=7 there are C(7,2)*C(5,3) = 21 * 10 = 210 such pairs.
For N=12 there are C(12,2)*C(10,3) = 66 * 120 = 7920 such pairs.

After optimization, count the triangles that remain "clean" (pierced by no
edge), and check whether a valid K_N neighborly triangulation of the
target genus can be extracted from the clean subset.

Supports Z_k symmetry in the same way as symmetric.py. Optionally runs
across 8 workers.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import combinations

import numpy as np


SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def _pin():
    for k, v in SINGLE_THREAD_ENV.items():
        os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Numpy geometry
# ---------------------------------------------------------------------------

def _sv(A, B, C, D):
    return np.dot(B - A, np.cross(C - A, D - A))

def _svvec(X, Y, Z, W):
    return np.einsum('...j,...j->...', Y - X, np.cross(Z - X, W - X))

def seg_crosses_tri(P, Q, A, B, C, tol=1e-3):
    """Robust predicate. Returns True if the segment PQ pierces the interior
    of triangle ABC, OR if the (segment, triangle) configuration is
    degenerate/near-degenerate (any signed volume has magnitude < tol).
    Conservative — degenerate cases are reported as intersections so the
    optimizer pushes away from them."""
    vp = _sv(P, A, B, C); vq = _sv(Q, A, B, C)
    vab = _sv(P, Q, A, B); vbc = _sv(P, Q, B, C); vca = _sv(P, Q, C, A)
    # Any near-zero signed volume means the configuration is near a
    # coplanar / degenerate case (e.g., vertices sharing a plane).
    if min(abs(vp), abs(vq), abs(vab), abs(vbc), abs(vca)) < tol:
        return True
    plane = vp * vq < 0
    inside = (vab > 0 and vbc > 0 and vca > 0) or \
             (vab < 0 and vbc < 0 and vca < 0)
    return plane and inside


def count_all_intersections(V, TRIANGLES, EDGES, tol=1e-3):
    counts = np.zeros(len(TRIANGLES), dtype=np.int32)
    offenders = []
    for ti, tri in enumerate(TRIANGLES):
        A, B, C = V[list(tri)]
        for e in EDGES:
            if set(e) & set(tri): continue
            P, Q = V[e[0]], V[e[1]]
            if seg_crosses_tri(P, Q, A, B, C, tol):
                counts[ti] += 1
                offenders.append((tri, e))
    return counts, offenders


# ---------------------------------------------------------------------------
# Greedy extraction of valid face set from clean triangles
# ---------------------------------------------------------------------------

def _tri_edges(face):
    a, b, c = face
    return (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c))))


def _link_is_cycle(v, faces_containing):
    link_edges, link_verts = [], set()
    for face in faces_containing:
        opp = [x for x in face if x != v]
        link_edges.append(tuple(sorted(opp))); link_verts.update(opp)
    if not link_verts: return False
    deg = {u: 0 for u in link_verts}
    for a, b in link_edges:
        deg[a] += 1; deg[b] += 1
    if any(d != 2 for d in deg.values()): return False
    adj = {u: [] for u in link_verts}
    for a, b in link_edges:
        adj[a].append(b); adj[b].append(a)
    start = next(iter(link_verts)); seen = {start}; stk = [start]
    while stk:
        u = stk.pop()
        for w in adj[u]:
            if w not in seen: seen.add(w); stk.append(w)
    return len(seen) == len(link_verts)


def try_extract_polyhedron(clean_triangles, N, EDGES, target_F, rng=None,
                            require_manifold=True, n_tries=200):
    """Try `n_tries` different random orderings of clean_triangles, greedily
    selecting a valid face set."""
    if rng is None:
        rng = np.random.default_rng(0)
    clean_list = list(clean_triangles)
    for _ in range(n_tries):
        rng.shuffle(clean_list)
        edge_deg = {e: 0 for e in EDGES}
        selected = []
        for tri in clean_list:
            te = _tri_edges(tri)
            if any(edge_deg[e] >= 2 for e in te):
                continue
            selected.append(tri)
            for e in te: edge_deg[e] += 1
            if len(selected) == target_F:
                break
        if len(selected) != target_F or any(d != 2 for d in edge_deg.values()):
            continue
        if not require_manifold:
            return selected, True
        incident = {v: [] for v in range(N)}
        for face in selected:
            for v in face: incident[v].append(face)
        if all(_link_is_cycle(v, incident[v]) for v in range(N)):
            return selected, True
    return None, False


# ---------------------------------------------------------------------------
# JAX loss
# ---------------------------------------------------------------------------

def build_all_pair_index(N, TRIANGLES, EDGES):
    """All (non-incident triangle, edge) pairs: C(N,3) * C(N-3, 2) pairs total."""
    fi, ei = [], []
    for tri in TRIANGLES:
        tset = set(tri)
        for e in EDGES:
            if tset & set(e):
                continue
            fi.append(tri); ei.append(e)
    return np.asarray(fi, dtype=np.int32), np.asarray(ei, dtype=np.int32)


def make_all_loss(face_idx, edge_idx, margin, vol_margin):
    """Robust soft loss. Two additive components per pair:

      - crossing penalty: smooth_min(conds) + margin, softplus'd. conds are
        the three sign-product volumes we want strictly negative. Margin > 0
        forces the min to be at most -margin, guaranteeing real clearance
        instead of a knife-edge near-zero configuration.
      - degeneracy penalty: min(|vp|,|vq|,|vab|,|vbc|,|vca|) should be at
        least vol_margin. If any individual signed volume is near zero the
        configuration is degenerate and we penalise.
    """
    import jax
    import jax.numpy as jnp
    def _sv_j(A, B, C, D):
        return jnp.dot(B - A, jnp.cross(C - A, D - A))

    def _pair(fxy, exy, tau):
        A, B, C = fxy[0], fxy[1], fxy[2]
        P, Q = exy[0], exy[1]
        vp = _sv_j(P, A, B, C); vq = _sv_j(Q, A, B, C)
        vab = _sv_j(P, Q, A, B); vbc = _sv_j(P, Q, B, C); vca = _sv_j(P, Q, C, A)
        conds = jnp.stack([-vp * vq, vab * vbc, vbc * vca])
        smin = -tau * jax.scipy.special.logsumexp(-conds / tau)
        cross_pen = tau * jax.nn.softplus((smin + margin) / tau)
        # Degeneracy penalty: use absolute values of volumes and penalize
        # whichever is smallest below vol_margin.
        abs_vols = jnp.stack([jnp.abs(vp), jnp.abs(vq), jnp.abs(vab),
                              jnp.abs(vbc), jnp.abs(vca)])
        vol_smin = -tau * jax.scipy.special.logsumexp(-abs_vols / tau)
        degen_pen = tau * jax.nn.softplus((vol_margin - vol_smin) / tau)
        return cross_pen + degen_pen

    fi_j = jnp.asarray(face_idx); ei_j = jnp.asarray(edge_idx)
    def loss(V, tau):
        fv = V[fi_j]; ev = V[ei_j]
        return jnp.sum(jax.vmap(lambda f, e: _pair(f, e, tau))(fv, ev))
    return loss


# ---------------------------------------------------------------------------
# Symmetric helpers (copied from symmetric.py for subprocess ease)
# ---------------------------------------------------------------------------

def build_sigma(N, n_axis, k):
    sigma = list(range(N))
    for start in range(n_axis, N, k):
        for j in range(k):
            sigma[start + j] = start + ((j + 1) % k)
    return sigma


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def project_symmetric_batch(verts, N, n_axis, k):
    v = np.asarray(verts, dtype=np.float32).copy()
    theta = 2.0 * np.pi / k
    R_inv = np.stack([rotation_z(-j * theta) for j in range(k)])
    R = np.stack([rotation_z(j * theta) for j in range(k)])
    for a in range(n_axis):
        v[:, a, 0] = 0.0; v[:, a, 1] = 0.0
    for start in range(n_axis, N, k):
        canonical = np.zeros((v.shape[0], 3), dtype=np.float32)
        for j in range(k):
            canonical += np.einsum('ij,bj->bi', R_inv[j], v[:, start + j, :])
        canonical /= k
        for j in range(k):
            v[:, start + j, :] = np.einsum('ij,bj->bi', R[j], canonical)
    return v


def normalize_symmetric_batch(verts, N, n_axis, k, min_scale=1e-2):
    v = np.asarray(verts, dtype=np.float32).copy()
    mean_z = v[:, :, 2].mean(axis=1, keepdims=True)
    v[:, :, 2] -= mean_z
    if k == 2:
        xy = v[:, :, :2]
        cov = np.einsum('bni,bnj->bij', xy, xy) / xy.shape[1]
        _, evecs = np.linalg.eigh(cov)
        R = np.swapaxes(evecs, -1, -2)
        v[:, :, :2] = np.einsum('bij,bnj->bni', R, xy)
        max_abs = np.maximum(np.max(np.abs(v), axis=1, keepdims=True), min_scale)
        v /= max_abs
    elif k >= 3:
        max_xy = np.maximum(
            np.max(np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2), axis=1, keepdims=True),
            min_scale)[..., None]
        v[:, :, :2] /= max_xy
        max_z = np.maximum(np.max(np.abs(v[:, :, 2:3]), axis=1, keepdims=True), min_scale)
        v[:, :, 2:3] /= max_z
    else:
        # No symmetry: full PCA normalization
        v -= v.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(v, full_matrices=False)
        v = np.matmul(v, np.swapaxes(Vt, -1, -2))
        max_abs = np.maximum(np.max(np.abs(v), axis=1, keepdims=True), min_scale)
        v /= max_abs
    return v


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one(args):
    _pin()
    (seed, N, k, n_axis, batch, steps, lr, tau_start, tau_end,
     chunk_size, margin, vol_margin, hard_tol) = args
    import jax, jax.numpy as jnp

    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    target_F = N * (N - 1) // 3

    face_idx, edge_idx = build_all_pair_index(N, TRIANGLES, EDGES)
    loss_fn = make_all_loss(face_idx, edge_idx, margin, vol_margin)
    grad_fn = jax.grad(loss_fn)

    rng = np.random.default_rng(seed)
    V0 = rng.standard_normal((batch, N, 3)).astype(np.float32)
    if k >= 2:
        V0 = project_symmetric_batch(V0, N, n_axis, k)
        V0 = normalize_symmetric_batch(V0, N, n_axis, k)
    else:
        V0 = normalize_symmetric_batch(V0, N, 0, 1)

    beta1, beta2, ae = 0.9, 0.999, 1e-8

    def one_step(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_fn, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s

    @jax.jit
    def run_chunk(V, m, s, start_f, tau_chunk):
        def body(carry, x):
            V_, m_, s_ = carry; sn, tv = x
            return one_step(V_, m_, s_, sn, tv), None
        step_nums = start_f + jnp.arange(1, tau_chunk.shape[0] + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s), (step_nums, tau_chunk))
        return final

    V = jnp.asarray(V0, dtype=jnp.float32)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)
    tau_sched = (tau_start *
                 (tau_end / tau_start) ** np.linspace(0, 1, steps)).astype(np.float32)

    done = 0
    best_per_instance = np.full(batch, 10**9, dtype=np.int32)
    best_verts = np.asarray(V)
    while done < steps:
        chunk_k = min(chunk_size, steps - done)
        tau_chunk = jnp.asarray(tau_sched[done: done + chunk_k])
        V, m, s = run_chunk(V, m, s, jnp.float32(done), tau_chunk)
        Vn = np.asarray(V)
        if k >= 2:
            Vn = project_symmetric_batch(Vn, N, n_axis, k)
        Vn = normalize_symmetric_batch(Vn, N, n_axis if k >= 2 else 0, k if k >= 2 else 1)
        V = jnp.asarray(Vn, dtype=jnp.float32)
        # Vectorized ROBUST batch counting.
        fv = Vn[:, face_idx]        # (B, P, 3, 3)
        ev = Vn[:, edge_idx]        # (B, P, 2, 3)
        A = fv[..., 0, :]; Bb = fv[..., 1, :]; C = fv[..., 2, :]
        P = ev[..., 0, :]; Q = ev[..., 1, :]
        vp = _svvec(P, A, Bb, C); vq = _svvec(Q, A, Bb, C)
        vab = _svvec(P, Q, A, Bb); vbc = _svvec(P, Q, Bb, C); vca = _svvec(P, Q, C, A)
        # Degenerate: any individual volume smaller than hard_tol → flag as bad.
        abs_min = np.minimum.reduce([np.abs(vp), np.abs(vq), np.abs(vab),
                                      np.abs(vbc), np.abs(vca)])
        degen = abs_min < hard_tol
        plane = vp * vq < 0
        pos = (vab > 0) & (vbc > 0) & (vca > 0)
        neg = (vab < 0) & (vbc < 0) & (vca < 0)
        actual_cross = plane & (pos | neg)
        all_counts = (actual_cross | degen).sum(axis=-1).astype(np.int32)
        improved = all_counts < best_per_instance
        best_per_instance = np.where(improved, all_counts, best_per_instance)
        best_verts = np.where(improved[:, None, None], Vn, best_verts)
        done += chunk_k

    # Pick best
    i_star = int(best_per_instance.argmin())
    v = best_verts[i_star]
    counts, offenders = count_all_intersections(v, TRIANGLES, EDGES, tol=hard_tol)
    clean_tris = [tri for ti, tri in enumerate(TRIANGLES) if counts[ti] == 0]
    poly_faces, poly_ok = try_extract_polyhedron(
        clean_tris, N, EDGES, target_F, rng=np.random.default_rng(seed + 42))
    return {
        "seed": seed,
        "total_intersections": int(counts.sum()),
        "clean_triangles": [list(t) for t in clean_tris],
        "n_clean": len(clean_tris),
        "n_triangles_total": len(TRIANGLES),
        "poly_ok": poly_ok,
        "poly_faces": [list(f) for f in (poly_faces or [])],
        "vertices": v.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--k", type=int, default=1,
                    help="Symmetry order (1 = no symmetry, 2 = Z_2, ...)")
    ap.add_argument("--axis-vertices", type=int, default=-1)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="smooth_min(conds) must drop below -margin for a pair to count as clean.")
    ap.add_argument("--vol-margin", type=float, default=0.05,
                    help="Each signed volume magnitude must exceed vol_margin to avoid degenerate penalty.")
    ap.add_argument("--hard-tol", type=float, default=1e-2,
                    help="Hard intersection check tolerance. Pairs with any signed volume below this "
                         "(in magnitude) count as intersecting.")
    ap.add_argument("--out", default="clean_tri")
    args = ap.parse_args()

    N, k = args.n, args.k
    if args.axis_vertices == -1:
        n_axis = 1 if (N % k == 1 and k > 1) else 0
    else:
        n_axis = args.axis_vertices
    if k > 1 and (N - n_axis) % k != 0:
        print(f"invalid: (N - n_axis) = {N - n_axis} not divisible by k = {k}")
        return

    n_pairs = len(list(combinations(range(N), 3))) * len(list(combinations(range(N - 3), 2)))
    print(f"N={N}, k={k}, n_axis={n_axis}, total (tri, edge) pairs = {n_pairs}")

    work = [(args.start_seed + i, N, k, n_axis, args.batch, args.steps,
             args.lr, args.tau_start, args.tau_end, args.chunk_size,
             args.margin, args.vol_margin, args.hard_tol)
            for i in range(args.seeds)]

    ctx = mp.get_context("spawn")
    t0 = time.time()
    best = None
    all_results = []
    def is_better(new, cur):
        # Prefer (poly_ok=True, low_ix) over (poly_ok=False, low_ix).
        if new["poly_ok"] and not cur["poly_ok"]:
            return True
        if not new["poly_ok"] and cur["poly_ok"]:
            return False
        return new["total_intersections"] < cur["total_intersections"]
    with ctx.Pool(args.workers, initializer=_pin) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, work)):
            all_results.append(r)
            marker = ""
            if best is None or is_better(r, best):
                best = r; marker = "  ***"
            poly_str = f"poly={'YES' if r['poly_ok'] else 'no'}"
            print(f"[{i+1}/{len(work)}] seed {r['seed']:3d}  "
                  f"total_ix={r['total_intersections']:4d}  "
                  f"clean_tris={r['n_clean']}/{r['n_triangles_total']}  "
                  f"{poly_str}  cur_best={best['total_intersections']}"
                  f"{'/poly' if best['poly_ok'] else ''}{marker}",
                  flush=True)

    dt = time.time() - t0
    print(f"\nTotal {dt:.1f}s, best total_ix = {best['total_intersections']} "
          f"(seed {best['seed']}, clean tris = {best['n_clean']}, "
          f"poly={best['poly_ok']})")

    # Save
    v = np.asarray(best["vertices"], dtype=np.float32)
    np.save(f"{args.out}_vertices.npy", v)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": N, "k": k, "n_axis": n_axis,
            "vertices": v.tolist(),
            "total_intersections": best["total_intersections"],
            "n_clean": best["n_clean"],
            "n_triangles_total": best["n_triangles_total"],
            "clean_triangles": best["clean_triangles"],
            "poly_ok": best["poly_ok"],
            "poly_faces": best["poly_faces"],
            "best_seed": best["seed"],
        }, fh, indent=2)
    print(f"wrote {args.out}.{{npy,json}}")


if __name__ == "__main__":
    main()
