"""Deep intensive optimization of a single pseudo-manifold structure.

Takes an existing face set (read from a JSON) and runs 8 parallel workers,
each with its own random initialization, a longer step budget, and a warm /
cold annealing schedule. Reports the minimum intersections found across all
workers and re-saves the best result."""

import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import combinations

import numpy as np


def _worker_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS",
                          "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def _do_run(args):
    _worker_env()
    (seed, faces_json_list, steps, batch, lr, tau_start, tau_end,
     chunk_size, restart_every, tag) = args

    from itertools import combinations
    import numpy as np
    from neighborly import (
        build_pair_index, make_loss, pca_normalize_batch,
        count_intersections_given_faces,
    )
    import jax
    import jax.numpy as jnp

    N = 12
    EDGES = list(combinations(range(N), 2))
    faces = [tuple(f) for f in faces_json_list]

    face_idx, edge_idx = build_pair_index(faces, EDGES)
    loss_single = make_loss(face_idx, edge_idx)
    grad_single = jax.grad(loss_single)

    beta1, beta2, ae = 0.9, 0.999, 1e-8

    def one_step(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_single, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s

    @jax.jit
    def run_chunk(V, m, s, start_f, tau_chunk):
        def body(carry, x):
            V_, m_, s_ = carry; step_num, tau_val = x
            return one_step(V_, m_, s_, step_num, tau_val), None
        step_nums = start_f + jnp.arange(1, tau_chunk.shape[0] + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s), (step_nums, tau_chunk))
        return final

    rng = np.random.default_rng(seed)
    V0 = rng.standard_normal((batch, N, 3)).astype(np.float32)
    V = pca_normalize_batch(V0)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)

    tau_sched = (tau_start *
                 (tau_end / tau_start) ** np.linspace(0, 1, steps)).astype(np.float32)

    best_verts_global = None
    best_ix_global = 10**9
    step_done = 0
    while step_done < steps:
        k = min(chunk_size, steps - step_done)
        tau_chunk = jnp.asarray(tau_sched[step_done: step_done + k])
        V, m, s = run_chunk(V, m, s, jnp.float32(step_done), tau_chunk)
        V = pca_normalize_batch(V)
        step_done += k

        V_np = np.asarray(V)
        n_ix_all = count_intersections_given_faces(V_np, faces, EDGES)
        i_star = int(n_ix_all.argmin())
        if int(n_ix_all[i_star]) < best_ix_global:
            best_ix_global = int(n_ix_all[i_star])
            best_verts_global = V_np[i_star].copy()

        # ES-style restart: replace bottom 25% with perturbed top 25%
        if restart_every and step_done < steps and (step_done % restart_every == 0):
            order = np.argsort(n_ix_all)
            n_top = max(1, batch // 4)
            top_V = V_np[order[:n_top]]
            bot_slots = order[-n_top:]
            noise = rng.normal(scale=0.15, size=(n_top, N, 3)).astype(np.float32)
            V = np.asarray(V).copy()
            V[bot_slots] = top_V + noise
            V = pca_normalize_batch(V)
            m_np = np.asarray(m).copy()
            s_np = np.asarray(s).copy()
            m_np[bot_slots] = 0.0
            s_np[bot_slots] = 0.0
            m = jnp.asarray(m_np); s = jnp.asarray(s_np)

    return {
        "seed": seed,
        "tag": tag,
        "faces": faces_json_list,
        "best_n_ix": best_ix_global,
        "best_vertices": best_verts_global.tolist() if best_verts_global is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--input", default=None,
                   help="JSON with 'vertices' and 'faces' (only 'faces' is used). "
                        "One structure, K workers each with different random init.")
    g.add_argument("--pickle", default=None,
                   help="Pickle from find_low_pinch.py; dispatches the top --top-k "
                        "structures across workers.")
    ap.add_argument("--top-k", type=int, default=8,
                    help="When reading a pickle, run this many top-ranked structures.")
    ap.add_argument("--skip-k", type=int, default=0,
                    help="When reading a pickle, skip this many top-ranked structures "
                         "(use to resume after an earlier --top-k N pass).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds-per-worker", type=int, default=1,
                    help="When reading a single JSON, # random inits per worker.")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=1e-3)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--restart-every", type=int, default=1500,
                    help="Every N steps, replace worst 25%% of instances with "
                         "perturbed copies of the best 25%%. 0 disables.")
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--out", default="n12_deep")
    args = ap.parse_args()

    structures = []  # list of (tag, faces_list)
    if args.pickle:
        import pickle as _pickle
        with open(args.pickle, "rb") as fh:
            ranked = _pickle.load(fh)
        slice_ = ranked[args.skip_k: args.skip_k + args.top_k]
        print(f"Read {len(ranked)} structures from {args.pickle}; "
              f"using {args.skip_k}..{args.skip_k + len(slice_)}:")
        for i, r in enumerate(slice_):
            idx = args.skip_k + i
            tag = f"pickle#{idx}_pinch{r['pinch_count']}_excess{r['excess_components']}"
            print(f"  {tag}")
            structures.append((tag, [list(f) for f in r["faces"]]))
    else:
        inp = args.input or "n12_21.json"
        data = json.load(open(inp))
        faces = [list(f) for f in data["faces"]]
        print(f"Input {inp}: current_n_ix={data.get('real_intersections', '?')}, "
              f"{len(faces)} faces")
        structures.append((os.path.basename(inp), faces))

    work = []
    if args.pickle:
        # One work item per structure (one seed each).
        for i, (tag, faces) in enumerate(structures):
            seed = args.start_seed + i
            work.append((seed, faces, args.steps, args.batch, args.lr,
                         args.tau_start, args.tau_end, args.chunk_size,
                         args.restart_every, tag))
    else:
        for w in range(args.workers):
            for k in range(args.seeds_per_worker):
                seed = args.start_seed + w * 1000 + k
                work.append((seed, structures[0][1], args.steps, args.batch, args.lr,
                             args.tau_start, args.tau_end, args.chunk_size,
                             args.restart_every, structures[0][0]))

    print(f"running {len(work)} deep optimizations across {args.workers} workers, "
          f"{args.steps} steps each")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    best = None
    with ctx.Pool(args.workers, initializer=_worker_env) as pool:
        for i, r in enumerate(pool.imap_unordered(_do_run, work)):
            marker = ""
            if best is None or r["best_n_ix"] < best["best_n_ix"]:
                best = r
                marker = "  ***"
            print(f"[{i+1}/{len(work)}] tag={r.get('tag','')}: n_ix={r['best_n_ix']}"
                  f"{marker}", flush=True)
    dt = time.time() - t0
    print(f"\ntotal {dt:.1f}s")
    print(f"best: seed {best['seed']}, n_ix = {best['best_n_ix']}")

    if best and best["best_vertices"]:
        v = np.asarray(best["best_vertices"], dtype=np.float32)
        faces_t = [tuple(f) for f in best["faces"]]
        np.save(f"{args.out}_vertices.npy", v)
        np.save(f"{args.out}_faces.npy", np.asarray(faces_t, dtype=np.int32))
        with open(f"{args.out}.obj", "w") as fh:
            fh.write(f"# K_12 genus-6 (deep optimize, best={best['best_n_ix']} ix)\n")
            for p in v: fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for f in faces_t: fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        with open(f"{args.out}.json", "w") as fh:
            json.dump({
                "n": 12, "genus": 6,
                "vertices": v.tolist(),
                "faces": [list(f) for f in faces_t],
                "real_intersections": best["best_n_ix"],
                "tag": best.get("tag", ""),
                "best_seed": best["seed"],
            }, fh, indent=2)
        print(f"wrote {args.out}_vertices.npy, {args.out}_faces.npy, {args.out}.obj, {args.out}.json")


if __name__ == "__main__":
    main()
