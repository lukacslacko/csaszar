"""Record per-step clean-triangle counts for many seeds and plot their
trajectories to see how much the intersection optimization helps."""
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


def _svvec(X, Y, Z, W):
    return np.einsum('...j,...j->...', Y - X, np.cross(Z - X, W - X))


def count_clean_per_instance(Vn, face_idx, edge_idx, hard_tol):
    """Given a batch of vertex configurations, return per-instance
    counts of (clean triangles, total edge-triangle intersections)."""
    fv = Vn[:, face_idx]; ev = Vn[:, edge_idx]
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
    hits = (plane & (pos | neg)) | degen    # (B, n_pairs)
    n_ix = hits.sum(axis=-1).astype(np.int32)
    # Clean triangles: aggregate hits per triangle (use face_idx as group key).
    # But we want per-instance per-triangle — do that by counting per triangle.
    # face_idx has shape (n_pairs, 3). Same triangle appears n_edges_non_incident times.
    # Group by triangle vertex tuple.
    # For simplicity, iterate.
    B = Vn.shape[0]
    # Build mapping from pair index to triangle index (0..n_triangles-1)
    return n_ix


def _run_one(args):
    _pin()
    (seed, N, k, n_axis, batch, steps, lr, tau_start, tau_end,
     record_every, margin, vol_margin, hard_tol) = args

    import jax, jax.numpy as jnp
    from clean_triangles import (
        build_all_pair_index, make_all_loss,
        project_symmetric_batch, normalize_symmetric_batch,
    )

    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    face_idx, edge_idx = build_all_pair_index(N, TRIANGLES, EDGES)

    # Precompute which pair belongs to which triangle
    tri_idx_of = {t: i for i, t in enumerate(TRIANGLES)}
    pair_to_tri = np.array([tri_idx_of[tuple(f)] for f in face_idx], dtype=np.int32)

    loss_fn = make_all_loss(face_idx, edge_idx, margin, vol_margin)
    grad_fn = jax.grad(loss_fn)

    rng = np.random.default_rng(seed)
    V0 = rng.standard_normal((batch, N, 3)).astype(np.float32)
    if k >= 2:
        V0 = project_symmetric_batch(V0, N, n_axis, k)
    V0 = normalize_symmetric_batch(V0, N, n_axis if k >= 2 else 0, k if k >= 2 else 1)

    beta1, beta2, ae = 0.9, 0.999, 1e-8
    V = jnp.asarray(V0, dtype=jnp.float32)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)

    tau_sched = (tau_start * (tau_end / tau_start) ** np.linspace(0, 1, max(1, steps))).astype(np.float32)

    def one_step(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_fn, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s
    one_step_jit = jax.jit(one_step)

    # Record step 0 before any optimization
    def record_state(V_arr):
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
        n_ix_per_instance = hits.sum(axis=-1).astype(np.int32)
        # Per-instance per-triangle dirty count; clean triangles = count 0.
        B = V_arr.shape[0]
        clean_per_instance = np.zeros(B, dtype=np.int32)
        # Aggregate hits per triangle using np.add.at.
        for b in range(B):
            tri_hit = np.zeros(len(TRIANGLES), dtype=np.int32)
            np.add.at(tri_hit, pair_to_tri, hits[b].astype(np.int32))
            clean_per_instance[b] = int((tri_hit == 0).sum())
        return n_ix_per_instance, clean_per_instance

    history = []  # list of (step, n_ix_array, clean_array)

    Vn = np.asarray(V)
    n_ix, clean = record_state(Vn)
    history.append((0, n_ix.tolist(), clean.tolist()))

    for step in range(1, steps + 1):
        V, m, s = one_step_jit(V, m, s, jnp.float32(step), tau_sched[min(step - 1, len(tau_sched) - 1)])
        if step % record_every == 0 or step == steps:
            Vn = np.asarray(V)
            if k >= 2:
                Vn = project_symmetric_batch(Vn, N, n_axis, k)
            Vn = normalize_symmetric_batch(Vn, N, n_axis if k >= 2 else 0, k if k >= 2 else 1)
            V = jnp.asarray(Vn, dtype=jnp.float32)
            n_ix, clean = record_state(Vn)
            history.append((step, n_ix.tolist(), clean.tolist()))

    return {"seed": seed, "batch": batch, "history": history,
            "n_triangles_total": len(TRIANGLES),
            "n_pairs_total": len(face_idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--axis-vertices", type=int, default=-1)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size per seed (1 means one trajectory per seed).")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--vol-margin", type=float, default=0.05)
    ap.add_argument("--hard-tol", type=float, default=1e-2)
    ap.add_argument("--out", default="trajectory")
    args = ap.parse_args()

    N, k = args.n, args.k
    if args.axis_vertices == -1:
        n_axis = 1 if (N % k == 1 and k > 1) else 0
    else:
        n_axis = args.axis_vertices

    work = [(args.start_seed + i if hasattr(args, 'start_seed') else i,
             N, k, n_axis, args.batch, args.steps, args.lr,
             args.tau_start, args.tau_end, args.record_every,
             args.margin, args.vol_margin, args.hard_tol)
            for i in range(args.seeds)]

    print(f"Running {args.seeds} seeds, {args.steps} steps each, recording every {args.record_every}, "
          f"batch {args.batch}, across {args.workers} workers")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers, initializer=_pin) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, work)):
            results.append(r)
            final_clean = r["history"][-1][2]
            init_clean = r["history"][0][2]
            print(f"[{i+1}/{len(work)}] seed {r['seed']}: "
                  f"clean_tris {init_clean[0]} -> {final_clean[0]} (out of {r['n_triangles_total']})",
                  flush=True)

    print(f"Total time: {time.time() - t0:.1f}s")

    # Collect trajectories: a (seeds, records) array of clean counts.
    steps_recorded = [h[0] for h in results[0]["history"]]
    # Each result has batch trajectories. Flatten to list of (label, y_array).
    curves = []
    for r in results:
        for bi in range(args.batch):
            y = [h[2][bi] for h in r["history"]]
            curves.append((f"seed {r['seed']}.{bi}", steps_recorded, y))
    n_tri = results[0]["n_triangles_total"]

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, xs, ys in curves:
        ax.plot(xs, ys, alpha=0.45, linewidth=1)
    # Aggregate mean and spread
    ys_matrix = np.array([c[2] for c in curves])
    mean_curve = ys_matrix.mean(axis=0)
    ax.plot(steps_recorded, mean_curve, 'k-', linewidth=2.5, label='mean')
    ax.axhline(n_tri, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label=f'n_triangles_total = {n_tri}')
    ax.set_xlabel('optimization step')
    ax.set_ylabel(f'# intersection-free triangles (of {n_tri})')
    ax.set_title(f'Clean-triangle trajectories — N={args.n}, '
                 f'{len(curves)} trajectories, {args.steps} steps')
    ax.grid(alpha=0.3)
    ax.legend()
    plot_path = f"{args.out}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100)
    print(f"wrote {plot_path}")

    # Also save raw data
    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": args.n, "k": args.k, "n_axis": n_axis,
            "steps": args.steps, "batch": args.batch, "seeds": args.seeds,
            "n_triangles_total": n_tri,
            "n_pairs_total": results[0]["n_pairs_total"],
            "records": [{"seed": r["seed"], "history": r["history"]} for r in results],
        }, fh)
    print(f"wrote {args.out}.json")


if __name__ == "__main__":
    main()
