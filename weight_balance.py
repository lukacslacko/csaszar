"""Study how the balance between the intersection penalty and the
degeneracy penalty affects optimization. 32 seeds for each weight
combination, 2000 steps, plot loss and clean-triangle trajectories."""
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


def _make_weighted_loss(face_idx, edge_idx, margin, vol_margin, w_cross, w_degen):
    import jax
    import jax.numpy as jnp

    def _sv_j(A, B, C, D):
        return jnp.dot(B - A, jnp.cross(C - A, D - A))

    def _pair(fxy, exy, tau):
        A, B, C = fxy[0], fxy[1], fxy[2]
        P, Q = exy[0], exy[1]
        vp = _sv_j(P, A, B, C); vq = _sv_j(Q, A, B, C)
        vab = _sv_j(P, Q, A, B); vbc = _sv_j(P, Q, B, C); vca = _sv_j(P, Q, C, A)
        conds = jnp.stack([-vp * vq, vab * vbc, vbc * vca])
        smin = -tau * jax.scipy.special.logsumexp(-conds / tau)
        cross_pen = tau * jax.nn.softplus((smin + margin) / tau)
        abs_vols = jnp.stack([jnp.abs(vp), jnp.abs(vq), jnp.abs(vab),
                              jnp.abs(vbc), jnp.abs(vca)])
        vol_smin = -tau * jax.scipy.special.logsumexp(-abs_vols / tau)
        degen_pen = tau * jax.nn.softplus((vol_margin - vol_smin) / tau)
        return w_cross * cross_pen + w_degen * degen_pen

    fi_j = jnp.asarray(face_idx); ei_j = jnp.asarray(edge_idx)
    def loss(V, tau):
        fv = V[fi_j]; ev = V[ei_j]
        return jnp.sum(jax.vmap(lambda f, e: _pair(f, e, tau))(fv, ev))
    return loss


def _run_one(args):
    _pin()
    (seed, w_cross, w_degen, N, steps, lr, tau_start, tau_end,
     margin, vol_margin, hard_tol, record_every) = args
    import jax, jax.numpy as jnp
    from clean_triangles import normalize_symmetric_batch

    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
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

    loss_fn = _make_weighted_loss(face_idx, edge_idx, margin, vol_margin,
                                    w_cross, w_degen)
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

    loss_traj = []
    clean_traj = []
    steps_rec = []

    tau0 = float(tau_sched[0])
    loss_traj.append(float(eval_loss(V, tau0)))
    clean_traj.append(clean_at(np.asarray(V)))
    steps_rec.append(0)

    for step in range(1, steps + 1):
        tau_val = tau_sched[min(step - 1, len(tau_sched) - 1)]
        V, m, s = step_jit(V, m, s, jnp.float32(step), tau_val)
        if step % record_every == 0 or step == steps:
            loss_traj.append(float(eval_loss(V, tau_val)))
            Vn = normalize_symmetric_batch(np.asarray(V), N, 0, 1)
            V = jnp.asarray(Vn, dtype=jnp.float32)
            clean_traj.append(clean_at(Vn))
            steps_rec.append(step)

    return {"seed": seed, "w_cross": w_cross, "w_degen": w_degen,
            "steps": steps_rec, "loss": loss_traj, "clean": clean_traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--record-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--vol-margin", type=float, default=0.01)
    ap.add_argument("--hard-tol", type=float, default=2e-3)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=0.05)
    ap.add_argument("--out", default="balance")
    args = ap.parse_args()

    # Weight combinations to try.
    combos = [
        (1.0, 0.0),   # pure intersection
        (1.0, 0.1),   # mostly intersection
        (1.0, 0.3),   # weakly degeneracy
        (1.0, 1.0),   # default / equal
        (1.0, 3.0),   # strong degeneracy
        (0.1, 1.0),   # mostly degeneracy
        (0.0, 1.0),   # pure degeneracy
    ]

    work = []
    for (wc, wd) in combos:
        for i in range(args.seeds):
            work.append((i, wc, wd, args.n, args.steps, args.lr,
                         args.tau_start, args.tau_end,
                         args.margin, args.vol_margin, args.hard_tol,
                         args.record_every))
    print(f"Running {len(work)} optimizations ({len(combos)} weight combos x {args.seeds} seeds, "
          f"{args.steps} steps each)")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    trajectories = []
    with ctx.Pool(args.workers, initializer=_pin) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, work)):
            trajectories.append(r)
            if i % 10 == 0 or i == len(work) - 1:
                print(f"[{i+1}/{len(work)}] w_cross={r['w_cross']:.1f}, "
                      f"w_degen={r['w_degen']:.1f}, seed {r['seed']}: "
                      f"loss {r['loss'][0]:.2f}->{r['loss'][-1]:.2f}  "
                      f"clean {r['clean'][0]}->{r['clean'][-1]}",
                      flush=True)
    dt = time.time() - t0
    print(f"Total: {dt:.1f}s")

    # Plot: grid of (w_cross, w_degen) -> mean clean trajectory
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_triangles = 220 if args.n == 12 else 35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(combos)))
    for idx, (wc, wd) in enumerate(combos):
        runs = [r for r in trajectories if r["w_cross"] == wc and r["w_degen"] == wd]
        if not runs:
            continue
        clean_arr = np.array([r["clean"] for r in runs])
        loss_arr = np.array([r["loss"] for r in runs])
        xs = runs[0]["steps"]
        label = f"w_cross={wc}, w_degen={wd}"
        axes[0].plot(xs, loss_arr.mean(axis=0), color=colors[idx], linewidth=2, label=label)
        axes[0].fill_between(xs,
                              np.percentile(loss_arr, 25, axis=0),
                              np.percentile(loss_arr, 75, axis=0),
                              color=colors[idx], alpha=0.1)
        axes[1].plot(xs, clean_arr.mean(axis=0), color=colors[idx], linewidth=2, label=label)
        axes[1].fill_between(xs,
                              np.percentile(clean_arr, 25, axis=0),
                              np.percentile(clean_arr, 75, axis=0),
                              color=colors[idx], alpha=0.1)
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("soft loss")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Loss (mean ± IQR) — {args.seeds} seeds per combo")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel(f"# clean triangles (of {n_triangles})")
    axes[1].axhline(44, color='gray', ls='--', alpha=0.5, label='44 (K_12 target)')
    axes[1].set_title(f"Clean triangles (mean ± IQR)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=100)
    print(f"wrote {args.out}.png")

    # Summary table
    print("\n=== summary (mean final values across 32 seeds) ===")
    print(f"{'w_cross':>8} {'w_degen':>8} {'final_loss':>12} {'final_clean':>12} {'best_clean':>12}")
    for (wc, wd) in combos:
        runs = [r for r in trajectories if r["w_cross"] == wc and r["w_degen"] == wd]
        if not runs: continue
        final_loss = np.mean([r["loss"][-1] for r in runs])
        final_clean = np.mean([r["clean"][-1] for r in runs])
        best_clean = max(r["clean"][-1] for r in runs)
        print(f"{wc:>8.2f} {wd:>8.2f} {final_loss:>12.2f} {final_clean:>12.2f} {best_clean:>12d}")

    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "n": args.n, "seeds": args.seeds, "steps": args.steps,
            "weight_combos": [list(c) for c in combos],
            "trajectories": trajectories,
        }, fh)
    print(f"wrote {args.out}.json")


if __name__ == "__main__":
    main()
