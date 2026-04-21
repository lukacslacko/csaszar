"""Scan many seeds for initial clean-triangle count, then for the top- and
bottom-scoring 32 run the de-intersection optimization and plot loss
trajectories."""
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


def _svvec(X, Y, Z, W):
    return np.einsum('...j,...j->...', Y - X, np.cross(Z - X, W - X))


def initial_clean_count(seed, N, face_idx, edge_idx, pair_to_tri,
                         n_triangles, hard_tol=2e-3):
    """Count intersection-free triangles at a random seed init."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3)).astype(np.float32)
    v -= v.mean(axis=0)
    v /= max(np.max(np.abs(v)), 1e-2)
    fv = v[face_idx]; ev = v[edge_idx]
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
    tri_hits = np.zeros(n_triangles, dtype=np.int32)
    np.add.at(tri_hits, pair_to_tri, hits.astype(np.int32))
    return int((tri_hits == 0).sum())


def _scan_batch(args):
    _pin()
    seeds, N, face_idx, edge_idx, pair_to_tri, n_triangles, hard_tol = args
    return [(s, initial_clean_count(s, N, face_idx, edge_idx, pair_to_tri,
                                      n_triangles, hard_tol))
            for s in seeds]


def _optimize_and_record(args):
    _pin()
    (seed, label, N, steps, lr, tau_start, tau_end, margin,
     vol_margin, hard_tol) = args
    import jax, jax.numpy as jnp
    from clean_triangles import (
        build_all_pair_index, make_all_loss,
        normalize_symmetric_batch,
    )

    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    face_idx, edge_idx = build_all_pair_index(N, TRIANGLES, EDGES)
    pair_to_tri = np.array([TRIANGLES.index(tuple(f)) for f in face_idx],
                           dtype=np.int32)
    loss_fn = make_all_loss(face_idx, edge_idx, margin, vol_margin)
    grad_fn = jax.grad(loss_fn)

    rng = np.random.default_rng(seed)
    V0 = rng.standard_normal((1, N, 3)).astype(np.float32)
    V0 = normalize_symmetric_batch(V0, N, 0, 1)
    V = jnp.asarray(V0, dtype=jnp.float32)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)
    beta1, beta2, ae = 0.9, 0.999, 1e-8
    tau_sched = (tau_start * (tau_end / tau_start) ** np.linspace(0, 1, steps)).astype(np.float32)

    @jax.jit
    def step_jit(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_fn, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s

    @jax.jit
    def eval_loss(V, tau_val):
        return loss_fn(V[0], tau_val)

    def clean_at(V_arr):
        fv = V_arr[:, face_idx]; ev = V_arr[:, edge_idx]
        A = fv[..., 0, :]; Bb = fv[..., 1, :]; C = fv[..., 2, :]
        P = ev[..., 0, :]; Q = ev[..., 1, :]
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
        np.add.at(tri_hits, pair_to_tri, hits[0].astype(np.int32))
        return int((tri_hits == 0).sum())

    loss_trajectory = []
    clean_trajectory = []

    # Record initial
    tau0 = float(tau_sched[0])
    loss_trajectory.append(float(eval_loss(V, tau0)))
    clean_trajectory.append(clean_at(np.asarray(V)))

    for step in range(1, steps + 1):
        tau_val = tau_sched[min(step - 1, len(tau_sched) - 1)]
        V, m, s = step_jit(V, m, s, jnp.float32(step), tau_val)
        loss_trajectory.append(float(eval_loss(V, tau_val)))
        Vn = normalize_symmetric_batch(np.asarray(V), N, 0, 1)
        V = jnp.asarray(Vn, dtype=jnp.float32)
        clean_trajectory.append(clean_at(Vn))

    return {"seed": seed, "label": label,
            "loss": loss_trajectory, "clean": clean_trajectory}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--scan-seeds", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--vol-margin", type=float, default=0.01)
    ap.add_argument("--hard-tol", type=float, default=2e-3)
    ap.add_argument("--out", default="filter_study")
    args = ap.parse_args()

    N = args.n
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    n_triangles = len(TRIANGLES)
    # Build pair index once
    fi, ei = [], []
    for tri in TRIANGLES:
        ts = set(tri)
        for e in EDGES:
            if ts & set(e): continue
            fi.append(tri); ei.append(e)
    face_idx = np.asarray(fi, dtype=np.int32)
    edge_idx = np.asarray(ei, dtype=np.int32)
    tri_idx = {t: i for i, t in enumerate(TRIANGLES)}
    pair_to_tri = np.array([tri_idx[tuple(f)] for f in face_idx], dtype=np.int32)

    # Step 1: scan all seeds for initial clean count.
    print(f"Scanning {args.scan_seeds} seeds for initial clean count...")
    t0 = time.time()
    chunks = 32
    per_chunk = args.scan_seeds // chunks
    work = [(list(range(i * per_chunk, (i + 1) * per_chunk)),
             N, face_idx, edge_idx, pair_to_tri, n_triangles, args.hard_tol)
            for i in range(chunks)]
    scan_results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_scan_batch, work):
            scan_results.extend(r)
    scan_results.sort(key=lambda x: x[1])
    n_init_mean = np.mean([c for _, c in scan_results])
    print(f"  scan done in {time.time() - t0:.1f}s; mean init clean = {n_init_mean:.1f}, "
          f"min = {scan_results[0][1]}, max = {scan_results[-1][1]}")

    bottom = scan_results[: args.top_k]
    top = scan_results[-args.top_k:]
    print(f"  bottom {args.top_k}: clean {bottom[0][1]} .. {bottom[-1][1]}")
    print(f"  top    {args.top_k}: clean {top[0][1]} .. {top[-1][1]}")

    # Step 2+3: optimize top and bottom seeds, record loss.
    print(f"\nOptimizing top {args.top_k} and bottom {args.top_k} seeds, "
          f"{args.opt_steps} steps each...")
    work_opt = []
    for seed, cl in top:
        work_opt.append((seed, "top", N, args.opt_steps, args.lr,
                         args.tau_start, args.tau_end, args.margin,
                         args.vol_margin, args.hard_tol))
    for seed, cl in bottom:
        work_opt.append((seed, "bottom", N, args.opt_steps, args.lr,
                         args.tau_start, args.tau_end, args.margin,
                         args.vol_margin, args.hard_tol))

    t0 = time.time()
    trajectories = []
    with ctx.Pool(args.workers, initializer=_pin) as pool:
        for i, r in enumerate(pool.imap_unordered(_optimize_and_record, work_opt)):
            trajectories.append(r)
            print(f"[{i+1}/{len(work_opt)}] seed {r['seed']:4d} ({r['label']:6s})  "
                  f"loss {r['loss'][0]:.2f} -> {r['loss'][-1]:.2f}  "
                  f"clean {r['clean'][0]} -> {r['clean'][-1]}",
                  flush=True)
    print(f"Optimization done in {time.time() - t0:.1f}s")

    # Plot.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xs = list(range(args.opt_steps + 1))
    for r in trajectories:
        color = 'tab:green' if r['label'] == 'top' else 'tab:red'
        axes[0].plot(xs, r['loss'], color=color, alpha=0.3, linewidth=1)
        axes[1].plot(xs, r['clean'], color=color, alpha=0.3, linewidth=1)
    # mean curves
    loss_top = np.array([r['loss'] for r in trajectories if r['label'] == 'top'])
    loss_bot = np.array([r['loss'] for r in trajectories if r['label'] == 'bottom'])
    clean_top = np.array([r['clean'] for r in trajectories if r['label'] == 'top'])
    clean_bot = np.array([r['clean'] for r in trajectories if r['label'] == 'bottom'])
    axes[0].plot(xs, loss_top.mean(axis=0), color='tab:green', linewidth=2.5, label='top mean')
    axes[0].plot(xs, loss_bot.mean(axis=0), color='tab:red', linewidth=2.5, label='bottom mean')
    axes[1].plot(xs, clean_top.mean(axis=0), color='tab:green', linewidth=2.5, label='top mean')
    axes[1].plot(xs, clean_bot.mean(axis=0), color='tab:red', linewidth=2.5, label='bottom mean')
    axes[0].set_xlabel('optimization step')
    axes[0].set_ylabel('soft loss (intersection + degeneracy)')
    axes[0].set_title(f'Loss trajectories (top/bot {args.top_k} of {args.scan_seeds} seeds)')
    axes[0].set_yscale('log')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('optimization step')
    axes[1].set_ylabel(f'# clean triangles (of {n_triangles})')
    axes[1].set_title('Clean-triangle trajectories')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].axhline(y=44, color='gray', linestyle='--', alpha=0.5, label='44 (target)')

    fig.tight_layout()
    plot_path = f"{args.out}.png"
    fig.savefig(plot_path, dpi=100)
    print(f"wrote {plot_path}")

    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": N, "scan_seeds": args.scan_seeds,
            "init_distribution": {
                "mean": float(n_init_mean),
                "min": int(scan_results[0][1]),
                "max": int(scan_results[-1][1]),
            },
            "top_seeds": [{"seed": s, "init_clean": c} for s, c in top],
            "bottom_seeds": [{"seed": s, "init_clean": c} for s, c in bottom],
            "trajectories": trajectories,
        }, fh)
    print(f"wrote {args.out}.json")


if __name__ == "__main__":
    main()
