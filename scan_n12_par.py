"""Parallel scan: run many pseudo-manifold K_12 optimization runs across 8
workers and collect the lowest intersection count.

Each worker is a separate process that limits JAX / BLAS / OpenMP to a single
thread so the 8 workers don't fight each other. Each worker picks up the next
available seed, finds a pseudo-manifold face set for that seed, runs the
parallel-Adam optimizer, and returns the lowest intersection count."""

import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import combinations

import numpy as np


def _worker_setup():
    # Force single-threaded BLAS/OpenMP per worker so 8 processes don't fight.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "XLA_FLAGS"):
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS",
                          "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def _run_one(args):
    _worker_setup()
    seed, steps, batch, tau_start, tau_end, lr, chunk_size, structure_tries = args
    # Import inside worker to pick up the env settings.
    from itertools import combinations
    import numpy as np
    from neighborly import (
        make_tri_edge_disjoint_pairs, greedy_select_faces,
        count_intersections_given_faces, optimize_batch,
    )

    N = 12
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    target_F = N * (N - 1) // 3
    tri_arr, edge_arr, pair_t, pair_e = make_tri_edge_disjoint_pairs(
        N, TRIANGLES, EDGES)

    rng = np.random.default_rng(seed)
    faces = None
    for t in range(structure_tries):
        verts = rng.random((N, 3)) * 2 - 1
        f, ok = greedy_select_faces(verts, N, EDGES, TRIANGLES, target_F,
                                     tri_arr, edge_arr, pair_t, pair_e, rng,
                                     require_manifold=False)
        if ok:
            faces = f
            break
    if faces is None:
        return {"seed": seed, "status": "no_structure"}

    rng = np.random.default_rng(seed + 100003)
    V0 = rng.standard_normal((batch, N, 3)).astype(np.float32)
    t0 = time.time()
    best_verts, best_ix = optimize_batch(
        V0, faces, EDGES,
        steps=steps, lr=lr,
        tau_start=tau_start, tau_end=tau_end,
        chunk_size=chunk_size, log_every=steps,
        use_scan=True,
    )
    dt = time.time() - t0
    i_star = int(best_ix.argmin())
    n_ix = int(best_ix[i_star])
    v = best_verts[i_star]
    n_zero = int((best_ix == 0).sum())
    return {
        "seed": seed, "status": "done", "n_ix": n_ix, "n_zero": n_zero,
        "dt": dt, "vertices": v.tolist(), "faces": [list(f) for f in faces],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=32,
                    help="Total number of seeds (structures) to try.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-seed", type=int, default=20)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--structure-tries", type=int, default=500)
    ap.add_argument("--out", default="n12_best")
    args = ap.parse_args()

    work = [(args.start_seed + i, args.steps, args.batch,
             args.tau_start, args.tau_end, args.lr, args.chunk_size,
             args.structure_tries)
            for i in range(args.seeds)]
    t_global = time.time()
    results = []
    best = None

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_setup) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, work)):
            if r["status"] != "done":
                print(f"[{i+1}/{len(work)}] seed {r['seed']}: {r['status']}")
                continue
            marker = ""
            if best is None or r["n_ix"] < best["n_ix"]:
                best = r
                marker = "  ***"
            print(f"[{i+1}/{len(work)}] seed {r['seed']:4d}  n_ix={r['n_ix']:4d}  "
                  f"dt={r['dt']:.1f}s  cur_best={best['n_ix']}{marker}")
            results.append(r)

    dt_all = time.time() - t_global
    print(f"\nTotal wall time: {dt_all:.1f}s over {args.workers} workers")
    if best is None:
        print("No successful run.")
        return

    print("\n==== summary (sorted by n_ix) ====")
    for r in sorted(results, key=lambda r: r["n_ix"]):
        print(f"  seed {r['seed']:4d}  n_ix = {r['n_ix']:4d}  zero_seen = {r['n_zero']}")

    print(f"\n==== best: seed {best['seed']}, {best['n_ix']} intersections ====")
    v = np.asarray(best["vertices"], dtype=np.float32)
    faces = [tuple(f) for f in best["faces"]]
    np.save(f"{args.out}_vertices.npy", v)
    np.save(f"{args.out}_faces.npy", np.asarray(faces, dtype=np.int32))
    with open(f"{args.out}.obj", "w") as fh:
        fh.write("# K_12 genus-6 pseudo-manifold (best of parallel scan)\n")
        for p in v: fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for f in faces: fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": 12, "genus": 6,
            "vertices": v.tolist(),
            "faces": [list(f) for f in faces],
            "real_intersections": best["n_ix"],
            "best_seed": best["seed"],
            "summary": sorted(
                [{"seed": r["seed"], "n_ix": r["n_ix"], "n_zero": r["n_zero"]}
                 for r in results],
                key=lambda r: r["n_ix"]),
        }, fh, indent=2)
    print(f"wrote {args.out}_vertices.npy, {args.out}_faces.npy, "
          f"{args.out}.obj, {args.out}.json")


if __name__ == "__main__":
    main()
