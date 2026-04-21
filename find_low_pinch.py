"""Pre-scan: generate many pseudo-manifold face sets for K_12 via greedy and
rank them by "pinch count" — the number of vertices whose link is not a
single cycle. Structures with fewer pinches are closer to being true manifolds
and potentially have lower intersection-count floors."""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from itertools import combinations

import numpy as np


def _pinch_stats(faces, N):
    incident = {v: [] for v in range(N)}
    for face in faces:
        for v in face:
            incident[v].append(face)
    pinch_count = 0
    total_excess_components = 0
    for v in range(N):
        link_edges, link_verts = [], set()
        for face in incident[v]:
            opp = [x for x in face if x != v]
            link_edges.append(tuple(sorted(opp)))
            link_verts.update(opp)
        if not link_verts:
            continue
        # Check degrees
        deg = {u: 0 for u in link_verts}
        for a, b in link_edges:
            deg[a] += 1; deg[b] += 1
        if any(d != 2 for d in deg.values()):
            pinch_count += 1
            total_excess_components += 5   # penalty
            continue
        # Components
        adj = {u: [] for u in link_verts}
        for a, b in link_edges:
            adj[a].append(b); adj[b].append(a)
        seen, comps = set(), 0
        for start in link_verts:
            if start in seen: continue
            comps += 1
            stk = [start]; seen.add(start)
            while stk:
                u = stk.pop()
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w); stk.append(w)
        if comps > 1:
            pinch_count += 1
            total_excess_components += comps - 1
    return pinch_count, total_excess_components


def _one_worker(args):
    N, seed, num_tries = args
    from neighborly import (
        make_tri_edge_disjoint_pairs, greedy_select_faces,
    )
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    target_F = N * (N - 1) // 3
    tri_arr, edge_arr, pair_t, pair_e = make_tri_edge_disjoint_pairs(
        N, TRIANGLES, EDGES)

    rng = np.random.default_rng(seed)
    found = []
    for t in range(num_tries):
        verts = rng.random((N, 3)) * 2 - 1
        faces, ok = greedy_select_faces(
            verts, N, EDGES, TRIANGLES, target_F,
            tri_arr, edge_arr, pair_t, pair_e, rng,
            require_manifold=False)
        if ok:
            pc, xc = _pinch_stats(faces, N)
            found.append({
                "seed_attempt": seed * num_tries + t,
                "faces": [list(f) for f in faces],
                "pinch_count": pc,
                "excess_components": xc,
                "init_verts": verts.tolist(),
            })
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tries-per-worker", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="low_pinch.pkl")
    args = ap.parse_args()

    N = args.n
    print(f"Searching for low-pinch pseudo-manifolds on K_{N}...")
    work = [(N, w * 100003, args.tries_per_worker) for w in range(args.workers)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    all_found = []
    with ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_one_worker, work):
            all_found.extend(res)
            print(f"  worker done: +{len(res)} face sets "
                  f"(total {len(all_found)} so far, {time.time() - t0:.1f}s)",
                  flush=True)
    print(f"\nTotal: {len(all_found)} face sets in {time.time() - t0:.1f}s")

    if not all_found:
        print("No face sets found.")
        return

    # Distribution of pinch counts
    from collections import Counter
    pc_dist = Counter(r["pinch_count"] for r in all_found)
    xc_dist = Counter(r["excess_components"] for r in all_found)
    print("\nPinch count distribution (lower is closer to manifold):")
    for k in sorted(pc_dist.keys()):
        print(f"  pinch_count={k:2d}: {pc_dist[k]} structures")
    print("\nExcess-components distribution:")
    for k in sorted(xc_dist.keys()):
        print(f"  excess={k:2d}: {xc_dist[k]} structures")

    # Sort by (pinch_count, excess_components) ascending
    all_found.sort(key=lambda r: (r["pinch_count"], r["excess_components"]))
    with open(args.out, "wb") as fh:
        pickle.dump(all_found, fh)
    print(f"\nWrote {args.out} with {len(all_found)} face sets (sorted by pinch_count)")
    print("Top 10 (fewest pinches):")
    for i, r in enumerate(all_found[:10]):
        print(f"  [{i}]  pinch={r['pinch_count']}, excess={r['excess_components']}")


if __name__ == "__main__":
    main()
