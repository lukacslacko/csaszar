"""Try many pseudo-manifold face sets for K_12 and report the lowest
intersection count reached by the optimizer on any of them.

Runs in-process so each structure reuses the JAX XLA cache as much as
possible; each structure still needs its own jit (face/edge index arrays
differ), but Python startup + import is paid once."""

import argparse
import json
import time
from itertools import combinations

import numpy as np

from neighborly import (
    make_tri_edge_disjoint_pairs,
    greedy_select_faces,
    count_intersections_given_faces,
    optimize_batch,
    write_obj,
)


def find_structure_once(N, EDGES, TRIANGLES, target_F, tri_arr, edge_arr,
                         pair_t, pair_e, seed, max_tries=500,
                         require_manifold=False):
    rng = np.random.default_rng(seed)
    for t in range(max_tries):
        verts = rng.random((N, 3)) * 2 - 1
        faces, ok = greedy_select_faces(verts, N, EDGES, TRIANGLES, target_F,
                                         tri_arr, edge_arr, pair_t, pair_e, rng,
                                         require_manifold=require_manifold)
        if ok:
            return faces
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--structure-tries", type=int, default=500)
    ap.add_argument("--require-manifold", action="store_true")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--out", default="n12_best")
    ap.add_argument("--early-stop-zero", action="store_true",
                    help="Stop scanning as soon as any structure reaches 0 intersections")
    args = ap.parse_args()

    N = 12
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    target_F = N * (N - 1) // 3
    print(f"K_{N}, target F = {target_F}, genus = 6")

    tri_arr, edge_arr, pair_t, pair_e = make_tri_edge_disjoint_pairs(
        N, TRIANGLES, EDGES)

    results = []
    best = None
    t_start = time.time()

    for i in range(args.structures):
        seed = args.start_seed + i
        print(f"\n=== structure {i + 1}/{args.structures} (seed {seed}) ===")
        t0 = time.time()
        faces = find_structure_once(N, EDGES, TRIANGLES, target_F,
                                     tri_arr, edge_arr, pair_t, pair_e,
                                     seed, args.structure_tries,
                                     require_manifold=args.require_manifold)
        if faces is None:
            print(f"  no face set found in {args.structure_tries} draws; skipping")
            continue
        print(f"  found face set ({len(faces)} faces) in {time.time() - t0:.1f}s")

        rng = np.random.default_rng(seed + 100003)
        V0 = rng.standard_normal((args.batch, N, 3)).astype(np.float32)
        t1 = time.time()
        best_verts, best_ix = optimize_batch(
            V0, faces, EDGES,
            steps=args.steps, lr=args.lr,
            tau_start=args.tau_start, tau_end=args.tau_end,
            chunk_size=args.chunk_size, log_every=args.steps,
            use_scan=True,
        )
        opt_time = time.time() - t1
        i_star = int(best_ix.argmin())
        n_ix = int(best_ix[i_star])
        v = best_verts[i_star]
        n_zero = int((best_ix == 0).sum())
        print(f"  optimizer done in {opt_time:.1f}s — best_n_ix={n_ix}, zero_instances={n_zero}/{args.batch}")

        results.append({
            "seed": seed, "n_ix": n_ix, "n_zero": n_zero,
            "vertices": v.tolist(), "faces": [list(f) for f in faces],
        })
        if best is None or n_ix < best["n_ix"]:
            best = results[-1]
            print(f"  *** NEW BEST: {n_ix} intersections ***")
        if args.early_stop_zero and n_ix == 0:
            break

    print(f"\ntotal time: {time.time() - t_start:.1f}s")
    if not results:
        print("no results")
        return

    print("\n==== summary (sorted by best n_ix) ====")
    for r in sorted(results, key=lambda r: r["n_ix"]):
        print(f"  seed {r['seed']:4d}  n_ix = {r['n_ix']:4d}  zero_seen = {r['n_zero']}")

    print(f"\n==== best: seed {best['seed']}, {best['n_ix']} intersections ====")

    v = np.asarray(best["vertices"], dtype=np.float32)
    faces = [tuple(f) for f in best["faces"]]
    np.save(f"{args.out}_vertices.npy", v)
    np.save(f"{args.out}_faces.npy", np.asarray(faces, dtype=np.int32))
    write_obj(f"{args.out}.obj", v, faces)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": N, "genus": 6,
            "vertices": v.tolist(),
            "faces": [list(f) for f in faces],
            "real_intersections": best["n_ix"],
            "best_seed": best["seed"],
            "summary": [{"seed": r["seed"], "n_ix": r["n_ix"], "n_zero": r["n_zero"]}
                        for r in sorted(results, key=lambda r: r["n_ix"])],
        }, fh, indent=2)
    print(f"wrote {args.out}_vertices.npy, {args.out}_faces.npy, "
          f"{args.out}.obj, {args.out}.json")


if __name__ == "__main__":
    main()
