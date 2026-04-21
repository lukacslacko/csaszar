"""Polish an existing N=7 polyhedron JSON and re-render it."""
import argparse
import json
import numpy as np
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="polished")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64,
                    help="Random perturbations around input vertices.")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="Stddev of gaussian perturbation around input.")
    ap.add_argument("--w-dih", type=float, default=1.0)
    ap.add_argument("--w-coll", type=float, default=0.3)
    args = ap.parse_args()

    from csaszar import (dihedral_cos, edge_pair_cos2,
                          build_dihedral_index, build_edge_pair_index,
                          build_pair_index, pca_normalize_batch,
                          count_real_intersections)
    from scan_polish import make_symmetric_polish_loss
    import jax

    data = json.load(open(args.input))
    v0 = np.asarray(data["vertices"], dtype=np.float32)
    faces = [tuple(f) for f in data["faces"]]
    assert len(v0) == 7, f"polish_once expects N=7; got {len(v0)}"

    rng = np.random.default_rng(0)
    V_batch = np.tile(v0, (args.batch, 1, 1))
    V_batch[1:] += rng.normal(scale=args.noise, size=V_batch[1:].shape).astype(np.float32)
    V_batch = V_batch.astype(np.float32)

    print(f"Polishing {args.input} ({args.batch} perturbations, {args.steps} steps) "
          f"with symmetric cos^2 loss...")

    # Symmetric cos^2 loss — penalises dihedrals near BOTH 0° and 180°.
    face_idx, edge_idx = build_pair_index(faces)
    dih_idx = build_dihedral_index(faces)
    ep_idx = build_edge_pair_index(7)
    loss_fn = make_symmetric_polish_loss(face_idx, edge_idx, dih_idx, ep_idx)
    grad_fn = jax.grad(loss_fn)

    beta1, beta2, ae = 0.9, 0.999, 1e-8
    lr, tau_ix = 1e-3, 5e-4

    def one_step(V, m, s, step_num, tau_dih, tau_coll):
        g = jax.vmap(grad_fn, in_axes=(0, None, None, None, None, None))(
            V, tau_ix, tau_dih, tau_coll, args.w_dih, args.w_coll)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s

    @jax.jit
    def run_chunk(V, m, s, start_f, tau_dih_chunk, tau_coll_chunk):
        def body(carry, x):
            V_, m_, s_ = carry; sn, td, tc = x
            return one_step(V_, m_, s_, sn, td, tc), None
        step_nums = start_f + jnp.arange(1, tau_dih_chunk.shape[0] + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s),
                                (step_nums, tau_dih_chunk, tau_coll_chunk))
        return final

    V = pca_normalize_batch(jnp.asarray(V_batch))
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)
    tau_dih_sched = (0.3 * (0.03 / 0.3) ** np.linspace(0, 1, args.steps)).astype(np.float32)
    tau_coll_sched = tau_dih_sched.copy()

    chunk_size = 200
    done = 0
    while done < args.steps:
        cs = min(chunk_size, args.steps - done)
        tdc = jnp.asarray(tau_dih_sched[done: done + cs])
        tcc = jnp.asarray(tau_coll_sched[done: done + cs])
        V, m, s = run_chunk(V, m, s, jnp.float32(done), tdc, tcc)
        V = pca_normalize_batch(V)
        done += cs
    polished_verts = np.asarray(V)

    # Pick best clean instance by smoothest dihedrals (min max deviation from 90°).
    best_idx = None; best_smoothness = 1e18
    for i in range(args.batch):
        n_ix, _ = count_real_intersections(polished_verts[i], faces)
        if n_ix != 0:
            continue
        cos_d = np.clip(np.asarray(dihedral_cos(polished_verts[i], dih_idx)), -1, 1)
        ang = np.degrees(np.arccos(cos_d))
        # Penalty: max over dihedrals of |angle - 90°|
        sm = float(np.max(np.abs(ang - 90.0)))
        if sm < best_smoothness:
            best_smoothness = sm; best_idx = i
    if best_idx is None:
        best_idx = 0
    final_v = polished_verts[best_idx]

    cos_d = np.clip(np.asarray(dihedral_cos(final_v, dih_idx)), -1, 1)
    ang_d = np.degrees(np.arccos(cos_d))
    cos2 = np.clip(np.asarray(edge_pair_cos2(final_v, ep_idx)), 0, 1)
    ep_d = np.degrees(np.arcsin(np.sqrt(np.clip(1 - cos2, 0, 1))))
    n_ix, offenders = count_real_intersections(final_v, faces)
    print(f"n_ix={n_ix}")
    print(f"  dihedral:      {ang_d.min():.2f}° .. {ang_d.max():.2f}°  (mean {ang_d.mean():.2f}°)")
    print(f"  edge-pair dev: {ep_d.min():.2f}° .. {ep_d.max():.2f}°  (mean {ep_d.mean():.2f}°)")

    np.save(f"{args.output}_vertices.npy", final_v)
    np.save(f"{args.output}_faces.npy", np.asarray(faces, dtype=np.int32))
    with open(f"{args.output}.obj", "w") as fh:
        fh.write(f"# polished from {args.input}\n")
        for p in final_v: fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for f in faces: fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    with open(f"{args.output}.json", "w") as fh:
        json.dump({
            "vertices": final_v.tolist(),
            "faces": [list(f) for f in faces],
            "real_intersections": int(n_ix),
            "min_dihedral_deg": float(ang_d.min()),
            "max_dihedral_deg": float(ang_d.max()),
            "mean_dihedral_deg": float(ang_d.mean()),
            "dihedral_degrees": ang_d.tolist(),
            "min_edge_pair_deviation_deg": float(ep_d.min()),
            "mean_edge_pair_deviation_deg": float(ep_d.mean()),
            "edge_pair_deviation_degrees": ep_d.tolist(),
            "polished_from": args.input,
        }, fh, indent=2)
    print(f"wrote {args.output}.{{npy,obj,json}}")


if __name__ == "__main__":
    main()
