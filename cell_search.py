"""Incremental N=7 construction with cell-based vertex placement.

Every three already-placed vertices span a plane. Those planes partition
R^3 into cells. For a new vertex to be placed, we need to avoid the
"forbidden" cells — the ones where placing it would cause some new edge
to pierce a committed face in its interior. Two cells can be topologically
distinguished by the sign-vector of a point in them relative to all of
the triple-planes; sampling many candidate points and grouping by sign
vector gives us a sampling approximation of the arrangement.

Two search modes:
  - `random`: pick one representative from a uniformly random feasible cell.
  - `exhaustive`: iterate ALL feasible cells at every level, recursing to
    the next vertex. Reports the total number of topologically distinct
    configurations (and how many extract to Csaszar).
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

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
        return True
    plane = vp * vq < 0
    inside = (vab > 0 and vbc > 0 and vca > 0) or \
             (vab < 0 and vbc < 0 and vca < 0)
    return plane and inside


def point_is_feasible(P, verts, committed_faces):
    """Does placing a new vertex at P create any crossing edges with committed faces?"""
    for j, v_j in enumerate(verts):
        for face in committed_faces:
            if j in face:
                continue
            A, B, C = verts[face[0]], verts[face[1]], verts[face[2]]
            if seg_crosses_tri(v_j, P, A, B, C):
                return False
    return True


def plane_sign_vectors(candidates, verts):
    """Given M candidate points and current vertex list, compute the sign
    vector for each candidate w.r.t. every plane defined by a triple of
    already-placed vertices. Returns (K, M) matrix of +/-1/0."""
    triples = list(combinations(range(len(verts)), 3))
    V = np.asarray(verts)
    signs = np.zeros((candidates.shape[0], len(triples)), dtype=np.int8)
    for i, (a, b, c) in enumerate(triples):
        A, B, C = V[a], V[b], V[c]
        n = np.cross(B - A, C - A)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue  # degenerate — all zero
        n /= norm
        d = -np.dot(n, A)
        vals = candidates @ n + d
        sv = np.sign(vals).astype(np.int8)
        sv[np.abs(vals) < 1e-6] = 0
        signs[:, i] = sv
    return signs, triples


def group_by_sign(signs):
    """Return a dict sig_tuple -> list of candidate indices."""
    groups = {}
    for i in range(signs.shape[0]):
        key = tuple(int(x) for x in signs[i])
        groups.setdefault(key, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

def feasible_cell_reps(verts, committed_faces, rng, n_samples=4000,
                       spread=5.0):
    """Sample candidates, group by sign vector, return one representative
    per feasible cell (and keep a bit of their spread for downstream picks).

    Returns list of (sign_key, representative_point, cell_size)."""
    center = np.mean(verts, axis=0)
    cands = center + rng.normal(scale=spread, size=(n_samples, 3))
    signs, _ = plane_sign_vectors(cands, verts)
    groups = group_by_sign(signs)
    feasible = []
    for key, idxs in groups.items():
        # Skip cells containing any on-plane ("0") coordinate — they are
        # boundary and ambiguous.
        if 0 in key:
            continue
        # Feasibility: test the centroid-ish representative (first cand).
        rep = cands[idxs[0]]
        if point_is_feasible(rep, verts, committed_faces):
            feasible.append((key, rep, len(idxs)))
    return feasible


# ---------------------------------------------------------------------------
# Trial building: random vs exhaustive
# ---------------------------------------------------------------------------

def try_one_random(rng, stddev=5.0, n_samples=4000):
    """Single random trial: at every level pick a random feasible cell and
    a random candidate within it."""
    verts = list(regular_tetrahedron())
    fixed_faces = [(0, 1, 2), (0, 1, 3)]
    per_step_stats = []
    while len(verts) < 7:
        feasible = feasible_cell_reps(
            verts, fixed_faces, rng, n_samples=n_samples, spread=stddev)
        per_step_stats.append(len(feasible))
        if not feasible:
            return None, per_step_stats
        _, rep, _ = feasible[rng.integers(0, len(feasible))]
        verts.append(rep)
    return np.array(verts), per_step_stats


def exhaustive_search(rng, stddev=5.0, n_samples=4000, max_leaves=2000):
    """Enumerate all feasible cells at every level up to max_leaves leaves.
    Returns list of candidate vertex arrays (each N=7)."""
    fixed_faces = [(0, 1, 2), (0, 1, 3)]
    start = list(regular_tetrahedron())
    # DFS stack: (partial_verts, depth)
    solutions = []
    per_level_branching = [[], [], []]  # list of branching factors at each level
    stack = [(start,)]
    while stack:
        (verts,) = stack.pop()
        if len(verts) == 7:
            solutions.append(np.array(verts))
            if len(solutions) >= max_leaves:
                break
            continue
        feasible = feasible_cell_reps(
            verts, fixed_faces, rng, n_samples=n_samples, spread=stddev)
        lvl = len(verts) - 4
        per_level_branching[lvl].append(len(feasible))
        # Reverse order to make DFS deterministic
        for key, rep, size in feasible:
            stack.append((verts + [rep],))
    return solutions, per_level_branching


# ---------------------------------------------------------------------------
# Clean triangle counting and Csaszar extraction (reused from incremental.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_random(args):
    _pin()
    (seed, stddev, n_samples) = args
    N = 7
    TRIANGLES = list(combinations(range(N), 3))
    EDGES = list(combinations(range(N), 2))
    rng = np.random.default_rng(seed)
    verts, per_step = try_one_random(rng, stddev=stddev, n_samples=n_samples)
    if verts is None:
        return {"seed": seed, "status": "no_feasible", "per_step": per_step}
    clean, _ = count_clean(verts, N, TRIANGLES, EDGES)
    poly, ok = try_extract(clean, N, EDGES, 14, rng,
                            must_include=[(0, 1, 2), (0, 1, 3)])
    return {"seed": seed, "status": "ok",
            "n_clean": len(clean), "poly_ok": ok,
            "per_step_feasible_cells": per_step,
            "vertices": verts.tolist(),
            "poly_faces": [list(f) for f in (poly or [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stddev", type=float, default=5.0)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--mode", choices=["random", "exhaustive"], default="random")
    ap.add_argument("--out", default="cells7")
    args = ap.parse_args()

    if args.mode == "random":
        work = [(i, args.stddev, args.n_samples) for i in range(args.seeds)]
        t0 = time.time()
        ctx = mp.get_context("spawn")
        results = []
        with ctx.Pool(args.workers, initializer=_pin) as pool:
            for r in pool.imap_unordered(_worker_random, work):
                results.append(r)
        dt = time.time() - t0
        ok = [r for r in results if r["status"] == "ok"]
        polys = [r for r in ok if r["poly_ok"]]
        print(f"\n{args.seeds} seeds in {dt:.1f}s ({args.seeds/dt:.0f} seeds/s)")
        print(f"  successful builds: {len(ok)}/{args.seeds}")
        if ok:
            c = np.array([r["n_clean"] for r in ok])
            print(f"  clean tris: min={c.min()}, max={c.max()}, mean={c.mean():.1f}, median={int(np.median(c))}")
            for t in [14, 20, 25, 28, 30, 33, 35]:
                n = (c >= t).sum()
                print(f"    # with clean >= {t:2d}: {n:5d} / {len(ok)} ({100 * n / len(ok):.1f}%)")
            # Per-step feasible cell counts
            fc = [r["per_step_feasible_cells"] for r in ok]
            for k in range(3):
                per_level = [s[k] for s in fc if len(s) > k]
                if per_level:
                    print(f"  level {k} (place v{4+k}): feasible cells — "
                          f"min={min(per_level)} max={max(per_level)} mean={np.mean(per_level):.1f}")
        print(f"  polyhedra extracted: {len(polys)}/{len(ok)}")

        # Save best
        best = max(ok, key=lambda r: (r["poly_ok"], r["n_clean"]), default=None)
        if best is not None:
            with open(f"{args.out}.json", "w") as fh:
                json.dump({
                    "seed": best["seed"], "n_clean": best["n_clean"],
                    "poly_ok": best["poly_ok"],
                    "vertices": best["vertices"],
                    "faces": best["poly_faces"],
                }, fh, indent=2)
            print(f"wrote {args.out}.json")
    else:
        # Exhaustive mode on a single seed (so we can count topologically
        # distinct configurations without the sampling randomness overwhelming
        # the count).
        rng = np.random.default_rng(0)
        t0 = time.time()
        solutions, branching = exhaustive_search(
            rng, stddev=args.stddev, n_samples=args.n_samples, max_leaves=200000)
        dt = time.time() - t0
        print(f"\nexhaustive: {len(solutions)} leaves in {dt:.1f}s")
        for lvl, bs in enumerate(branching):
            if bs:
                print(f"  level {lvl} branching: "
                      f"min={min(bs)}, max={max(bs)}, mean={np.mean(bs):.1f}, samples={len(bs)}")
        N = 7
        TRIANGLES = list(combinations(range(N), 3))
        EDGES = list(combinations(range(N), 2))
        polys = 0
        best_clean = 0
        for verts in solutions:
            clean, _ = count_clean(verts, N, TRIANGLES, EDGES)
            best_clean = max(best_clean, len(clean))
            _, ok = try_extract(clean, N, EDGES, 14, rng,
                                must_include=[(0, 1, 2), (0, 1, 3)],
                                n_tries=100)
            if ok: polys += 1
        print(f"  max clean = {best_clean}, polyhedra extracted = {polys} / {len(solutions)}")


if __name__ == "__main__":
    main()
