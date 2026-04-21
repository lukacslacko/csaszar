"""Incremental construction for N=7.

Fix vertices 0..3 as a regular tetrahedron; commit the two faces (0,1,2) and
(0,1,3) sharing edge (0,1). Then place v4, v5, v6 one at a time, each as a
normal-distributed sample around the tetrahedron centroid. Reject a candidate
if ANY edge from it to an already-placed vertex crosses one of the two
committed faces in the interior. Backtrack after --max-tries attempts at any
step.

After all 7 vertices are placed, count clean triangles (same robust predicate
as clean_triangles.py) and try to extract a valid Csaszar.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import combinations

import numpy as np


def _pin():
    for k, v in [
        ("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"), ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ]:
        os.environ.setdefault(k, v)


FIXED_FACES = ((0, 1, 2), (0, 1, 3))


def regular_tetrahedron():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2.0, 0.0])
    d = np.array([0.5, np.sqrt(3) / 6.0, np.sqrt(6) / 3.0])
    return np.array([a, b, c, d], dtype=np.float64)


def seg_crosses_tri(P, Q, A, B, C, tol=1e-9):
    def sv(X, Y, Z, W):
        return np.dot(Y - X, np.cross(Z - X, W - X))
    vp = sv(P, A, B, C); vq = sv(Q, A, B, C)
    vab = sv(P, Q, A, B); vbc = sv(P, Q, B, C); vca = sv(P, Q, C, A)
    if min(abs(vp), abs(vq), abs(vab), abs(vbc), abs(vca)) < tol:
        return True  # treat degenerate as bad
    plane = vp * vq < 0
    inside = (vab > 0 and vbc > 0 and vca > 0) or \
             (vab < 0 and vbc < 0 and vca < 0)
    return plane and inside


def edge_ok(v_new, verts_so_far, fixed_faces):
    """Check that edges from v_new to each existing vertex don't cross any
    fixed face in its interior."""
    for j, v_j in enumerate(verts_so_far):
        for face in fixed_faces:
            if j in face:
                continue
            A, B, C = verts_so_far[face[0]], verts_so_far[face[1]], verts_so_far[face[2]]
            if seg_crosses_tri(v_j, v_new, A, B, C):
                return False
    return True


def build(rng, stddev=2.0, max_tries=100, target_n=7, max_backtracks=200):
    """Returns (verts, stats) or (None, stats) on failure."""
    verts = list(regular_tetrahedron())
    center = np.mean(verts, axis=0)
    tries_left = [None] * 4 + [max_tries] * (target_n - 4)
    # tries_left[i] = remaining tries for vertex i
    stats = {"backtracks": 0, "rejected_edges": 0}

    i = 4
    while i < target_n:
        if tries_left[i] == 0:
            # backtrack
            stats["backtracks"] += 1
            if stats["backtracks"] >= max_backtracks:
                return None, stats
            if i <= 4:
                # Can't go back past 4; restart from scratch.
                tries_left = [None] * 4 + [max_tries] * (target_n - 4)
                verts = list(regular_tetrahedron())
                i = 4
                continue
            # Drop vertex i-1 and decrement its tries.
            verts.pop()
            i -= 1
            tries_left[i] -= 1
            # Refresh tries for the dropped-below position too? No, only dec prev.
            # Reset tries for i+1 (so we can re-try from scratch when we come back).
            for k in range(i + 1, target_n):
                tries_left[k] = max_tries
            continue

        candidate = center + rng.normal(scale=stddev, size=3)
        tries_left[i] -= 1
        if edge_ok(candidate, verts, FIXED_FACES):
            verts.append(candidate)
            i += 1
        else:
            stats["rejected_edges"] += 1
    return np.array(verts, dtype=np.float64), stats


# ---- counting + extraction (copy-simplified from clean_triangles) ----

def _svvec(X, Y, Z, W):
    return np.einsum('...j,...j->...', Y - X, np.cross(Z - X, W - X))


def count_clean(verts, N, TRIANGLES, EDGES, hard_tol=1e-2):
    fi, ei = [], []
    for tri in TRIANGLES:
        ts = set(tri)
        for e in EDGES:
            if ts & set(e): continue
            fi.append(tri); ei.append(e)
    face_idx = np.asarray(fi); edge_idx = np.asarray(ei)
    tri_idx = {t: i for i, t in enumerate(TRIANGLES)}
    pair_to_tri = np.array([tri_idx[tuple(f)] for f in face_idx], dtype=np.int32)

    fv = verts[face_idx]; ev = verts[edge_idx]
    A = fv[:, 0]; Bb = fv[:, 1]; C = fv[:, 2]
    P = ev[:, 0]; Q = ev[:, 1]
    vp = _svvec(P, A, Bb, C); vq = _svvec(Q, A, Bb, C)
    vab = _svvec(P, Q, A, Bb); vbc = _svvec(P, Q, Bb, C); vca = _svvec(P, Q, C, A)
    abs_min = np.minimum.reduce([np.abs(vp), np.abs(vq), np.abs(vab),
                                  np.abs(vbc), np.abs(vca)])
    degen = abs_min < hard_tol
    plane = vp * vq < 0
    pos = (vab > 0) & (vbc > 0) & (vca > 0)
    neg = (vab < 0) & (vbc < 0) & (vca < 0)
    hits = (plane & (pos | neg)) | degen
    tri_hits = np.zeros(len(TRIANGLES), dtype=np.int32)
    np.add.at(tri_hits, pair_to_tri, hits.astype(np.int32))
    clean_tris = [TRIANGLES[i] for i in range(len(TRIANGLES)) if tri_hits[i] == 0]
    return clean_tris, int(hits.sum())


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


def try_extract(clean_tris, N, EDGES, target_F, rng, must_include=None,
                n_tries=500, require_manifold=True):
    must_include = [tuple(sorted(f)) for f in (must_include or [])]
    clean_list = [tuple(sorted(t)) for t in clean_tris]
    for _ in range(n_tries):
        rng.shuffle(clean_list)
        edge_deg = {e: 0 for e in EDGES}
        selected = list(must_include)
        for tri in must_include:
            a, b, c = tri
            for u, v in [(a, b), (b, c), (a, c)]:
                edge_deg[tuple(sorted((u, v)))] += 1
        if any(d > 2 for d in edge_deg.values()):
            return None, False
        for tri in clean_list:
            if tri in selected: continue
            a, b, c = tri
            te = [tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))]
            if any(edge_deg[e] >= 2 for e in te): continue
            selected.append(tri)
            for e in te: edge_deg[e] += 1
            if len(selected) == target_F: break
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


def _worker(args):
    _pin()
    (seed, stddev, max_tries, max_backtracks) = args
    N = 7
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    target_F = 14
    rng = np.random.default_rng(seed)
    verts, stats = build(rng, stddev=stddev, max_tries=max_tries,
                          max_backtracks=max_backtracks)
    if verts is None:
        return {"seed": seed, "status": "build_failed", **stats}
    clean, n_hits = count_clean(verts, N, TRIANGLES, EDGES)
    poly, ok = try_extract(clean, N, EDGES, target_F, rng,
                            must_include=list(FIXED_FACES))
    return {"seed": seed, "status": "ok",
            "n_clean": len(clean), "n_hits": n_hits,
            "poly_ok": ok, "poly_faces": [list(f) for f in (poly or [])],
            "vertices": verts.tolist(), **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stddev", type=float, default=2.0)
    ap.add_argument("--max-tries", type=int, default=100)
    ap.add_argument("--max-backtracks", type=int, default=500)
    ap.add_argument("--out", default="incremental7")
    args = ap.parse_args()

    work = [(i, args.stddev, args.max_tries, args.max_backtracks)
            for i in range(args.seeds)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers, initializer=_pin) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, work)):
            results.append(r)
    dt = time.time() - t0

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    with_poly = [r for r in ok if r["poly_ok"]]

    print(f"\n{args.seeds} seeds in {dt:.1f}s ({args.seeds/dt:.0f} seeds/s)")
    print(f"  build succeeded: {len(ok)} / {args.seeds}")
    print(f"  build failed:    {len(failed)} / {args.seeds}")
    if ok:
        cleans = np.array([r["n_clean"] for r in ok])
        print(f"  clean tris: min={cleans.min()}, max={cleans.max()}, mean={cleans.mean():.1f}, median={int(np.median(cleans))}")
        for t in [14, 20, 25, 28, 30, 33, 35]:
            n = (cleans >= t).sum()
            print(f"    # seeds with clean >= {t:2d}: {n:5d} / {len(ok)} ({100 * n / len(ok):.1f}%)")
    print(f"  polyhedron extracted: {len(with_poly)} / {len(ok)}")

    # Save best
    best = max(ok, key=lambda r: (r["poly_ok"], r["n_clean"]), default=None)
    if best is not None:
        v = np.asarray(best["vertices"], dtype=np.float32)
        faces = [tuple(f) for f in best["poly_faces"]] if best["poly_ok"] else []
        with open(f"{args.out}.json", "w") as fh:
            json.dump({
                "seed": best["seed"], "n_clean": best["n_clean"],
                "poly_ok": best["poly_ok"],
                "vertices": v.tolist(),
                "faces": [list(f) for f in faces],
            }, fh, indent=2)
        print(f"wrote {args.out}.json (best: seed {best['seed']}, "
              f"clean {best['n_clean']}, poly {best['poly_ok']})")


if __name__ == "__main__":
    main()
