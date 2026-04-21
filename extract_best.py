"""Load a trajectory.py JSON, take the top-K seeds by final clean count,
re-run each briefly to collect the final vertex positions, and attempt
polyhedron extraction from each."""
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
        ("NUMEXPR_NUM_THREADS", "1"), ("JAX_PLATFORMS", "cpu"),
        ("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"),
    ]:
        os.environ.setdefault(k, v)


def _run_one(args):
    _pin()
    (seed, N, steps, lr, tau_start, tau_end, margin, vol_margin, hard_tol) = args
    import jax, jax.numpy as jnp
    from clean_triangles import (
        build_all_pair_index, make_all_loss, normalize_symmetric_batch,
        count_all_intersections, try_extract_polyhedron,
    )

    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    target_F = N * (N - 1) // 3
    face_idx, edge_idx = build_all_pair_index(N, TRIANGLES, EDGES)
    loss_fn = make_all_loss(face_idx, edge_idx, margin, vol_margin)
    grad_fn = jax.grad(loss_fn)

    rng = np.random.default_rng(seed)
    V = rng.standard_normal((1, N, 3)).astype(np.float32)
    V = normalize_symmetric_batch(V, N, 0, 1)
    V = jnp.asarray(V, dtype=jnp.float32)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)
    beta1, beta2, ae = 0.9, 0.999, 1e-8
    tau_sched = (tau_start * (tau_end / tau_start) ** np.linspace(0, 1, steps)).astype(np.float32)

    @jax.jit
    def step_jit(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_fn, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s

    for step in range(1, steps + 1):
        V, m, s = step_jit(V, m, s, jnp.float32(step), tau_sched[step - 1])

    Vn = normalize_symmetric_batch(np.asarray(V), N, 0, 1)
    v = Vn[0]
    counts, _ = count_all_intersections(v, TRIANGLES, EDGES, tol=hard_tol)
    clean_tris = [tri for ti, tri in enumerate(TRIANGLES) if counts[ti] == 0]
    # Check edge coverage: each edge must appear in >=2 clean triangles.
    edge_cov = {e: 0 for e in EDGES}
    for tri in clean_tris:
        a, b, c = tri
        for u, v_ in ((a, b), (b, c), (a, c)):
            edge_cov[tuple(sorted((u, v_)))] += 1
    edges_covered_2 = sum(1 for cnt in edge_cov.values() if cnt >= 2)
    faces_m, ok_m = try_extract_polyhedron(
        clean_tris, N, EDGES, target_F,
        rng=np.random.default_rng(seed + 42),
        require_manifold=True, n_tries=2000)
    faces_pm, ok_pm = try_extract_polyhedron(
        clean_tris, N, EDGES, target_F,
        rng=np.random.default_rng(seed + 42),
        require_manifold=False, n_tries=2000)
    best_faces = faces_m if ok_m else (faces_pm if ok_pm else [])
    return {"seed": seed, "n_clean": len(clean_tris),
            "edges_covered_2": edges_covered_2,
            "poly_ok": ok_m, "pseudo_ok": ok_pm,
            "vertices": v.tolist(),
            "poly_faces": [list(f) for f in (best_faces or [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="traj12_1k.json")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--vol-margin", type=float, default=0.01)
    ap.add_argument("--hard-tol", type=float, default=0.002)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="top_extract")
    args = ap.parse_args()

    d = json.load(open(args.trajectory))
    records = d["records"]
    records.sort(key=lambda r: r["history"][-1][2][0], reverse=True)
    top = records[:args.top_k]
    print(f"Top {len(top)} seeds by final clean count (from {args.trajectory}):")
    for r in top[:10]:
        print(f"  seed {r['seed']}: final clean = {r['history'][-1][2][0]}")
    if len(top) > 10:
        print(f"  ... and {len(top) - 10} more")

    work = [(r["seed"], args.n, args.steps, args.lr,
             args.tau_start, args.tau_end,
             args.margin, args.vol_margin, args.hard_tol)
            for r in top]

    t0 = time.time()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers, initializer=_pin) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, work)):
            results.append(r)
            print(f"[{i+1}/{len(work)}] seed {r['seed']:4d}  clean={r['n_clean']:3d}  "
                  f"edges_cov>=2={r['edges_covered_2']:2d}/66  "
                  f"mfold={'Y' if r['poly_ok'] else 'n'} "
                  f"pseudo={'Y' if r['pseudo_ok'] else 'n'}", flush=True)

    dt = time.time() - t0
    print(f"\nTotal {dt:.1f}s")
    results.sort(key=lambda r: (r["poly_ok"], r["n_clean"]), reverse=True)
    n_poly = sum(1 for r in results if r["poly_ok"])
    n_clean_max = max(r["n_clean"] for r in results)
    print(f"{n_poly}/{len(results)} extracted a valid polyhedron; max clean = {n_clean_max}")

    # Save the best extractable one
    best = next((r for r in results if r["poly_ok"]), results[0])
    v = np.asarray(best["vertices"], dtype=np.float32)
    faces = [tuple(f) for f in best["poly_faces"]] if best["poly_ok"] else []
    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": args.n, "vertices": v.tolist(),
            "faces": [list(f) for f in faces],
            "best_seed": best["seed"],
            "n_clean": best["n_clean"],
            "poly_ok": best["poly_ok"],
            "n_poly_extracted": n_poly,
            "max_clean_seen": n_clean_max,
        }, fh, indent=2)
    print(f"wrote {args.out}.json (best: seed {best['seed']}, clean {best['n_clean']}, poly {best['poly_ok']})")


if __name__ == "__main__":
    main()
