"""Deterministic cell enumeration for N=12 / K_12 polyhedron search.

Stage 1 (orchestrator): enumerate ALL topologically distinct feasible
placements of v4, v5, v6 using a deterministic (non-random) cell
enumeration. Report the count at each level and save the depth-7 states.

Stage 2 (workers): each of 8 workers takes a disjoint slice of the
saved depth-7 states and extends the exploration further down using the
same deterministic enumeration, reporting progress periodically.

Cell enumeration strategy (deterministic):

  - Every 3 already-placed vertices span a plane; planes partition R^3
    into cells. A cell is uniquely characterised by the sign vector of
    a point in it w.r.t. every plane.
  - Generate a fixed dense set of candidate points:
      (i) a uniform 3D grid in a bounded box,
      (ii) every arrangement vertex (intersection of 3 planes) inside
           the box, perturbed in all ±axis directions.
  - Group candidates by sign vector. Each non-empty group = one cell
    (possibly missing pure-interior cells that no sample happens to
    land in, but the fixed grid resolution bounds the diameter of
    misses).
  - Discard ambiguous (on-plane) samples (any entry of the sign vector
    being 0).
  - Keep only cells whose representative point doesn't cause any edge
    from it to an already-placed vertex to pierce the 2 committed
    tetrahedron faces in their interiors.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter
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
# Base geometry
# ---------------------------------------------------------------------------

def regular_tetrahedron():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2.0, 0.0])
    d = np.array([0.5, np.sqrt(3) / 6.0, np.sqrt(6) / 3.0])
    return np.array([a, b, c, d], dtype=np.float64)


FIXED_FACES = ((0, 1, 2), (0, 1, 3))


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
    for j, v_j in enumerate(verts):
        for face in committed_faces:
            if j in face:
                continue
            A, B, C = verts[face[0]], verts[face[1]], verts[face[2]]
            if seg_crosses_tri(v_j, P, A, B, C):
                return False
    return True


# ---------------------------------------------------------------------------
# Deterministic cell enumeration
# ---------------------------------------------------------------------------

def _planes_from_verts(verts):
    V = np.asarray(verts)
    triples = list(combinations(range(len(V)), 3))
    normals, offsets = [], []
    for (a, b, c) in triples:
        A, B, C = V[a], V[b], V[c]
        n = np.cross(B - A, C - A)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue  # skip degenerate triples
        n /= norm
        d = -np.dot(n, A)
        normals.append(n); offsets.append(d)
    return np.asarray(normals), np.asarray(offsets)


def _arrangement_vertices(normals, offsets, box):
    """Compute intersection points of every triple of planes that falls
    inside the bounding box. Linear solve per triple."""
    M = len(normals)
    pts = []
    for (a, b, c) in combinations(range(M), 3):
        A = np.stack([normals[a], normals[b], normals[c]])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        try:
            p = np.linalg.solve(A, -np.array([offsets[a], offsets[b], offsets[c]]))
        except np.linalg.LinAlgError:
            continue
        if np.max(np.abs(p)) < box:
            pts.append(p)
    return np.asarray(pts) if pts else np.zeros((0, 3))


def enumerate_feasible_cells(verts, committed_faces,
                              box=6.0, grid_n=30, pert=0.05):
    """Deterministic enumeration of feasible cells.

    Dense sample = grid(grid_n per axis in box) + (6 perturbations of every
    arrangement vertex inside the box). Group by sign vector, check
    feasibility of representative."""
    normals, offsets = _planes_from_verts(verts)
    if len(normals) == 0:
        return []

    # 1. Uniform grid
    axis = np.linspace(-box, box, grid_n)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing='ij')
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # 2. Arrangement vertices with ±axis perturbations
    arr_vs = _arrangement_vertices(normals, offsets, box)
    if len(arr_vs):
        off = np.array([[+pert, 0, 0], [-pert, 0, 0],
                         [0, +pert, 0], [0, -pert, 0],
                         [0, 0, +pert], [0, 0, -pert]])
        perturbed = (arr_vs[:, None, :] + off[None, :, :]).reshape(-1, 3)
        samples = np.concatenate([grid, perturbed], axis=0)
    else:
        samples = grid

    vals = samples @ normals.T + offsets      # (n_samples, M)
    signs = np.sign(vals).astype(np.int8)
    signs[np.abs(vals) < 1e-7] = 0

    cells = {}
    for i in range(samples.shape[0]):
        s = signs[i]
        if 0 in s:
            continue
        key = tuple(int(x) for x in s)
        cells.setdefault(key, []).append(i)

    feasible = []
    for key, idxs in cells.items():
        rep = samples[idxs[0]]
        if point_is_feasible(rep, verts, committed_faces):
            feasible.append((key, rep, len(idxs)))
    return feasible


# ---------------------------------------------------------------------------
# Clean-triangle / extraction helpers (for worker stage reporting)
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
    return [TRIANGLES[i] for i in range(len(TRIANGLES)) if tri_hits[i] == 0]


# ---------------------------------------------------------------------------
# Orchestrator: enumerate to target_depth levels
# ---------------------------------------------------------------------------

def orchestrate(target_n_initial, target_depth, box, grid_n, pert,
                report_every=5.0):
    """DFS down to `target_depth` total vertices; return list of partial
    configurations at that depth together with per-level stats."""
    stats = {
        "cells_per_level": {i: [] for i in range(target_depth - target_n_initial + 1)},
        "t_start": time.time(),
    }
    start_verts = list(regular_tetrahedron())
    stack = []

    # Initial enumeration
    init_cells = enumerate_feasible_cells(
        start_verts, FIXED_FACES, box=box, grid_n=grid_n, pert=pert)
    stats["cells_per_level"][0].append(len(init_cells))
    print(f"Level 0 (placing v{target_n_initial}): {len(init_cells)} feasible cells")
    stack.append({"verts": start_verts, "cells": list(init_cells), "next": 0})

    frontier = []  # list of completed depth-(target_depth) configs

    last_report = time.time()
    while stack:
        frame = stack[-1]
        if frame["next"] >= len(frame["cells"]):
            stack.pop()
            continue
        _, rep, _ = frame["cells"][frame["next"]]
        frame["next"] += 1

        new_verts = frame["verts"] + [rep]
        level = len(new_verts) - target_n_initial

        if len(new_verts) == target_depth:
            frontier.append(list(new_verts))
            if time.time() - last_report > report_every:
                elapsed = time.time() - stats["t_start"]
                per_level_totals = {
                    lvl: {"visits": len(cs),
                          "total": sum(cs),
                          "mean": sum(cs) / max(1, len(cs)),
                          "min": min(cs) if cs else 0,
                          "max": max(cs) if cs else 0}
                    for lvl, cs in stats["cells_per_level"].items()
                    if cs
                }
                print(f"\n[{elapsed:.1f}s] frontier = {len(frontier)} "
                      f"depth-{target_depth} configs found so far")
                for lvl, info in per_level_totals.items():
                    print(f"  level {lvl}: visited {info['visits']} parents, "
                          f"cells per visit: min={info['min']}, max={info['max']}, "
                          f"mean={info['mean']:.1f}, total={info['total']}")
                last_report = time.time()
            continue

        cells = enumerate_feasible_cells(
            new_verts, FIXED_FACES, box=box, grid_n=grid_n, pert=pert)
        stats["cells_per_level"][level].append(len(cells))
        if cells:
            stack.append({"verts": new_verts, "cells": list(cells), "next": 0})

    elapsed = time.time() - stats["t_start"]
    print(f"\n=== ORCHESTRATOR DONE in {elapsed:.1f}s ===")
    print(f"Total depth-{target_depth} configurations: {len(frontier)}")
    for lvl, cs in stats["cells_per_level"].items():
        if not cs: continue
        print(f"  level {lvl}: visited {len(cs)} parents, "
              f"cells/visit min={min(cs)}, max={max(cs)}, mean={sum(cs)/len(cs):.1f}, "
              f"total={sum(cs)}")
    return frontier, stats


# ---------------------------------------------------------------------------
# Worker: continue the DFS from a given depth-7 state
# ---------------------------------------------------------------------------

def _worker(args):
    _pin()
    (chunk_id, verts_list, target_n, box, grid_n, pert, time_limit) = args
    target_F = target_n * (target_n - 1) // 3
    TRIANGLES = list(combinations(range(target_n), 3))
    EDGES = list(combinations(range(target_n), 2))
    t0 = time.time()
    completed = 0
    best_clean = 0
    leaves = 0
    dead_ends = 0
    polys = 0

    for start_idx, start_verts in enumerate(verts_list):
        if time.time() - t0 > time_limit:
            break
        stack = [{"verts": list(start_verts),
                  "cells": enumerate_feasible_cells(start_verts, FIXED_FACES,
                                                     box=box, grid_n=grid_n, pert=pert),
                  "next": 0}]
        while stack:
            if time.time() - t0 > time_limit:
                break
            frame = stack[-1]
            if frame["next"] >= len(frame["cells"]):
                stack.pop(); continue
            _, rep, _ = frame["cells"][frame["next"]]
            frame["next"] += 1
            new_verts = frame["verts"] + [rep]
            if len(new_verts) == target_n:
                completed += 1
                V = np.array(new_verts, dtype=np.float64)
                clean = count_clean(V, target_n, TRIANGLES, EDGES)
                if len(clean) > best_clean:
                    best_clean = len(clean)
                leaves += 1
                continue
            cells = enumerate_feasible_cells(new_verts, FIXED_FACES,
                                              box=box, grid_n=grid_n, pert=pert)
            if cells:
                stack.append({"verts": new_verts, "cells": cells, "next": 0})
            else:
                dead_ends += 1
    return {
        "chunk_id": chunk_id,
        "seeds_processed": start_idx + 1 if verts_list else 0,
        "completed": completed, "leaves": leaves,
        "best_clean": best_clean, "dead_ends": dead_ends,
        "polys": polys,
        "time": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--orchestrator-depth", type=int, default=7,
                    help="Total vertex count to enumerate to before launching workers.")
    ap.add_argument("--box", type=float, default=6.0,
                    help="Half-side of the sampling box.")
    ap.add_argument("--grid-n", type=int, default=30,
                    help="Grid resolution per axis for sampling.")
    ap.add_argument("--pert", type=float, default=0.05,
                    help="Magnitude of perturbation around arrangement vertices.")
    ap.add_argument("--report-every", type=float, default=5.0)
    ap.add_argument("--save-frontier", default="frontier.pkl")
    ap.add_argument("--workers", type=int, default=0,
                    help="After orchestration, launch this many workers (0 = orchestrate only).")
    ap.add_argument("--worker-time-limit", type=float, default=300.0)
    args = ap.parse_args()

    _pin()
    print(f"Orchestrator: N={args.n}, depth={args.orchestrator_depth}, "
          f"box={args.box}, grid_n={args.grid_n}")
    frontier, stats = orchestrate(
        target_n_initial=4, target_depth=args.orchestrator_depth,
        box=args.box, grid_n=args.grid_n, pert=args.pert,
        report_every=args.report_every,
    )
    with open(args.save_frontier, "wb") as fh:
        pickle.dump(frontier, fh)
    print(f"wrote {args.save_frontier} ({len(frontier)} depth-{args.orchestrator_depth} configs)")

    if args.workers > 0 and frontier:
        print(f"\n=== LAUNCHING {args.workers} WORKERS ===")
        chunk_size = (len(frontier) + args.workers - 1) // args.workers
        chunks = [frontier[i:i+chunk_size] for i in range(0, len(frontier), chunk_size)]
        work = [(i, chunks[i], args.n, args.box, args.grid_n, args.pert,
                 args.worker_time_limit)
                for i in range(len(chunks))]
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ctx.Pool(args.workers, initializer=_pin) as pool:
            for r in pool.imap_unordered(_worker, work):
                print(f"  chunk {r['chunk_id']}: processed {r['seeds_processed']}, "
                      f"leaves {r['leaves']}, best_clean {r['best_clean']}, "
                      f"time {r['time']:.1f}s")
        print(f"Workers total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
