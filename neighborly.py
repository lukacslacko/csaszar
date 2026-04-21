"""Generalized polyhedron rediscovery for arbitrary vertex count.

For V = 7 we get the Csaszar polyhedron (torus). For V = 12 we get the next
candidate on the neighborly-orientable-surface ladder: E = 66 edges, F = 44
faces, chi = -10, genus 6. Whether a simplicial polyhedron realizing this
combinatorial type exists in R^3 without self-intersections is an open problem.

Pipeline (same as the phase-1 of csaszar.py, parameterized by N):

 1. Place N points in the unit cube at random.
 2. Greedy face selection: score each of C(N,3) triangles by how many of the
    non-incident C(N,2)-3 edges pierce its interior. Repeatedly pick the
    lowest-scoring triangle that doesn't yet push any edge above 2 incident
    faces. Require the result to be a closed 2-manifold (every edge used
    twice, every vertex link a single cycle).
 3. Parallel Adam on the signed-volume smooth interior-intersection penalty,
    periodic PCA normalization, batch of random restarts. Report the fewest
    intersections reached.

No polishing — we just want to see how far down the optimizer can push the
intersection count.
"""

import argparse
import json
import time
from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Exact geometry
# ---------------------------------------------------------------------------

def signed_vol6_vec(X, Y, Z, W):
    """6 * signed volume of tetrahedron, batched over leading dims."""
    return np.einsum('...j,...j->...', Y - X, np.cross(Z - X, W - X))


def _sgn_vol(A, B, C, D):
    return np.dot(B - A, np.cross(C - A, D - A))


def seg_crosses_tri_interior(P, Q, A, B, C, tol=1e-9):
    vp = _sgn_vol(P, A, B, C)
    vq = _sgn_vol(Q, A, B, C)
    vab = _sgn_vol(P, Q, A, B)
    vbc = _sgn_vol(P, Q, B, C)
    vca = _sgn_vol(P, Q, C, A)
    plane = vp * vq < -tol
    inside = (vab > tol and vbc > tol and vca > tol) or \
             (vab < -tol and vbc < -tol and vca < -tol)
    return plane and inside


# ---------------------------------------------------------------------------
# Vectorized intersection counting (numpy)
# ---------------------------------------------------------------------------

def make_tri_edge_disjoint_pairs(N, TRIANGLES, EDGES):
    """Index arrays of all (tri, edge) pairs with disjoint vertex sets."""
    tri_arr = np.asarray(TRIANGLES, dtype=np.int32)
    edge_arr = np.asarray(EDGES, dtype=np.int32)
    tset = np.zeros((len(TRIANGLES), N), dtype=bool)
    for i, t in enumerate(TRIANGLES):
        tset[i, list(t)] = True
    eset = np.zeros((len(EDGES), N), dtype=bool)
    for i, e in enumerate(EDGES):
        eset[i, list(e)] = True
    touch = np.einsum('in,jn->ij', tset.astype(np.int32), eset.astype(np.int32))
    pair_t, pair_e = np.where(touch == 0)
    return tri_arr, edge_arr, pair_t.astype(np.int32), pair_e.astype(np.int32)


def count_intersections_per_triangle(verts, tri_arr, edge_arr, pair_t, pair_e, tol=1e-9):
    """For every triangle in `tri_arr`, count how many non-incident edges
    (listed in the pair arrays) pierce its interior. Returns (T,) int."""
    tri_v = verts[tri_arr[pair_t]]       # (P, 3, 3)
    edge_v = verts[edge_arr[pair_e]]     # (P, 2, 3)
    A = tri_v[:, 0]; B = tri_v[:, 1]; C = tri_v[:, 2]
    P = edge_v[:, 0]; Q = edge_v[:, 1]
    vp = signed_vol6_vec(P, A, B, C)
    vq = signed_vol6_vec(Q, A, B, C)
    vab = signed_vol6_vec(P, Q, A, B)
    vbc = signed_vol6_vec(P, Q, B, C)
    vca = signed_vol6_vec(P, Q, C, A)
    plane = vp * vq < -tol
    pos = (vab > tol) & (vbc > tol) & (vca > tol)
    neg = (vab < -tol) & (vbc < -tol) & (vca < -tol)
    hits = plane & (pos | neg)
    counts = np.zeros(tri_arr.shape[0], dtype=np.int32)
    np.add.at(counts, pair_t, hits.astype(np.int32))
    return counts


def count_intersections_given_faces(verts_batch, faces, EDGES, tol=1e-9):
    """verts_batch: (B, N, 3). Returns (B,) counts."""
    face_vert_idx = []
    edge_vert_idx = []
    for face in faces:
        fset = set(face)
        for e in EDGES:
            if fset & set(e):
                continue
            face_vert_idx.append(face)
            edge_vert_idx.append(e)
    fvi = np.asarray(face_vert_idx, dtype=np.int32)
    evi = np.asarray(edge_vert_idx, dtype=np.int32)
    fv = verts_batch[:, fvi]     # (B, P, 3, 3)
    ev = verts_batch[:, evi]     # (B, P, 2, 3)
    A = fv[..., 0, :]; Bb = fv[..., 1, :]; C = fv[..., 2, :]
    P = ev[..., 0, :]; Q = ev[..., 1, :]
    vp = signed_vol6_vec(P, A, Bb, C)
    vq = signed_vol6_vec(Q, A, Bb, C)
    vab = signed_vol6_vec(P, Q, A, Bb)
    vbc = signed_vol6_vec(P, Q, Bb, C)
    vca = signed_vol6_vec(P, Q, C, A)
    plane = vp * vq < -tol
    pos = (vab > tol) & (vbc > tol) & (vca > tol)
    neg = (vab < -tol) & (vbc < -tol) & (vca < -tol)
    hits = plane & (pos | neg)
    return hits.sum(axis=-1)


# ---------------------------------------------------------------------------
# Greedy face selection with manifold check
# ---------------------------------------------------------------------------

def _tri_edges(tri):
    a, b, c = tri
    return (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c))))


def vertex_link_is_disk(vertex, faces_containing):
    if not faces_containing:
        return False
    link_edges, link_verts = [], set()
    for face in faces_containing:
        opp = [x for x in face if x != vertex]
        link_edges.append(tuple(sorted(opp)))
        link_verts.update(opp)
    deg = {v: 0 for v in link_verts}
    for (a, b) in link_edges:
        deg[a] += 1; deg[b] += 1
    if any(d != 2 for d in deg.values()):
        return False
    adj = {v: [] for v in link_verts}
    for (a, b) in link_edges:
        adj[a].append(b); adj[b].append(a)
    start = next(iter(link_verts))
    seen = {start}; stack = [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w); stack.append(w)
    return len(seen) == len(link_verts)


def greedy_select_faces(verts, N, EDGES, TRIANGLES, target_F,
                         tri_arr, edge_arr, pair_t, pair_e, rng,
                         require_manifold=True):
    """Greedy pick of `target_F` triangles that look like a closed manifold
    triangulation. We prune with three local budgets:

      - edge degree cap: each edge supports at most 2 selected faces (hard),
      - vertex degree cap: each vertex supports at most target_F*3/N faces,
      - edge slack-aware ordering: whenever we revisit the next candidate,
        we prefer triangles on edges that have fewer "still available" peers
        (edge remaining-budget MRV), falling back on intersection count and
        a random tiebreak.
    """
    counts = count_intersections_per_triangle(verts, tri_arr, edge_arr, pair_t, pair_e)
    tiebreak = rng.random(len(TRIANGLES))

    faces_per_vertex = (target_F * 3) // N
    edge_deg = {e: 0 for e in EDGES}
    vert_deg = [0] * N

    # Triangles per edge, for urgency tracking.
    edge_tris = {e: set() for e in EDGES}
    tri_edges_of = [None] * len(TRIANGLES)
    for ti, tri in enumerate(TRIANGLES):
        te = _tri_edges(tri)
        tri_edges_of[ti] = te
        for e in te:
            edge_tris[e].add(ti)

    # Which triangles are still candidates.
    candidates = set(range(len(TRIANGLES)))
    selected = []

    def disable_tri(ti):
        candidates.discard(ti)
        for e in tri_edges_of[ti]:
            edge_tris[e].discard(ti)

    while len(selected) < target_F and candidates:
        # Pick next by: (lowest slack on its most-urgent edge,
        #                intersection count, random tiebreak).
        best, best_key = None, None
        for ti in candidates:
            te = tri_edges_of[ti]
            # How many candidate triangles remain on each of its edges?
            slack = min(len(edge_tris[e]) for e in te)
            key = (slack, int(counts[ti]), tiebreak[ti])
            if best_key is None or key < best_key:
                best, best_key = ti, key
        ti = best
        tri = TRIANGLES[ti]
        te = tri_edges_of[ti]
        if any(edge_deg[e] >= 2 for e in te) or any(vert_deg[v] >= faces_per_vertex for v in tri):
            disable_tri(ti)
            continue
        selected.append(tri)
        for e in te:
            edge_deg[e] += 1
            if edge_deg[e] == 2:
                # Edge is full — any triangle still using it becomes infeasible.
                for other in list(edge_tris[e]):
                    if other != ti:
                        disable_tri(other)
        for v in tri:
            vert_deg[v] += 1
            if vert_deg[v] == faces_per_vertex:
                for other in list(candidates):
                    if v in TRIANGLES[other] and other != ti:
                        disable_tri(other)
        disable_tri(ti)

    if len(selected) != target_F or any(d != 2 for d in edge_deg.values()):
        return selected, False
    if not require_manifold:
        return selected, True
    incident = {v: [] for v in range(N)}
    for face in selected:
        for v in face:
            incident[v].append(face)
    for v in range(N):
        if not vertex_link_is_disk(v, incident[v]):
            return selected, False
    return selected, True


def _find_once(seed_and_args):
    (seed, N, EDGES, TRIANGLES, target_F, tri_arr, edge_arr, pair_t, pair_e,
     tries, require_manifold) = seed_and_args
    rng = np.random.default_rng(seed)
    for t in range(tries):
        verts = rng.random((N, 3)) * 2 - 1
        faces, ok = greedy_select_faces(verts, N, EDGES, TRIANGLES, target_F,
                                         tri_arr, edge_arr, pair_t, pair_e, rng,
                                         require_manifold=require_manifold)
        if ok:
            return t + 1, verts, faces
    return tries, None, None


def find_structure(N, EDGES, TRIANGLES, target_F,
                    tri_arr, edge_arr, pair_t, pair_e,
                    seed, max_tries, n_workers=1, verbose=True,
                    require_manifold=True):
    if n_workers <= 1:
        rng = np.random.default_rng(seed)
        reports = max(1, max_tries // 10)
        for t in range(max_tries):
            verts = rng.random((N, 3)) * 2 - 1
            faces, ok = greedy_select_faces(verts, N, EDGES, TRIANGLES, target_F,
                                             tri_arr, edge_arr, pair_t, pair_e, rng,
                                             require_manifold=require_manifold)
            if ok:
                if verbose: print(f"  greedy hit after {t + 1} draws")
                return verts, faces
            if verbose and (t + 1) % reports == 0:
                print(f"  ...{t + 1}/{max_tries} draws, still searching")
        return None, None

    import multiprocessing as mp
    per_worker = max_tries // n_workers
    args_list = [(seed + i * 97_003, N, EDGES, TRIANGLES, target_F,
                  tri_arr, edge_arr, pair_t, pair_e, per_worker, require_manifold)
                 for i in range(n_workers)]
    with mp.Pool(n_workers) as pool:
        for done, verts, faces in pool.imap_unordered(_find_once, args_list):
            if faces is not None:
                if verbose: print(f"  greedy hit in ~{done} draws (one of {n_workers} workers)")
                pool.terminate()
                return verts, faces
    return None, None


# ---------------------------------------------------------------------------
# JAX objective (same as csaszar.py)
# ---------------------------------------------------------------------------

def _signed_vol6_j(A, B, C, D):
    return jnp.dot(B - A, jnp.cross(C - A, D - A))


def _pair_penalty(face_xyz, edge_xyz, tau):
    A, B, C = face_xyz[0], face_xyz[1], face_xyz[2]
    P, Q = edge_xyz[0], edge_xyz[1]
    vp = _signed_vol6_j(P, A, B, C)
    vq = _signed_vol6_j(Q, A, B, C)
    vab = _signed_vol6_j(P, Q, A, B)
    vbc = _signed_vol6_j(P, Q, B, C)
    vca = _signed_vol6_j(P, Q, C, A)
    conds = jnp.stack([-vp * vq, vab * vbc, vbc * vca])
    smin = -tau * jax.scipy.special.logsumexp(-conds / tau)
    return tau * jax.nn.softplus(smin / tau)


def build_pair_index(faces, EDGES):
    face_idx, edge_idx = [], []
    for face in faces:
        fset = set(face)
        for e in EDGES:
            if fset & set(e):
                continue
            face_idx.append(face)
            edge_idx.append(e)
    return (jnp.asarray(face_idx, dtype=jnp.int32),
            jnp.asarray(edge_idx, dtype=jnp.int32))


def make_loss(face_idx, edge_idx):
    def loss(V, tau):
        fv = V[face_idx]; ev = V[edge_idx]
        return jnp.sum(jax.vmap(lambda f, e: _pair_penalty(f, e, tau))(fv, ev))
    return loss


def pca_normalize_batch(verts_batch, min_scale=1e-2):
    v = np.asarray(verts_batch, dtype=np.float32)
    centered = v - v.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    rotated = np.matmul(centered, np.swapaxes(Vt, -1, -2))
    max_abs = np.maximum(np.max(np.abs(rotated), axis=1, keepdims=True), min_scale)
    return jnp.asarray(rotated / max_abs, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Batched optimization
# ---------------------------------------------------------------------------

def optimize_batch(verts_batch, faces, EDGES,
                   steps=5000, lr=5e-3,
                   tau_start=0.5, tau_end=1e-3,
                   chunk_size=250, log_every=500, use_scan=True):
    face_idx, edge_idx = build_pair_index(faces, EDGES)
    loss_single = make_loss(face_idx, edge_idx)
    grad_single = jax.grad(loss_single)
    loss_batch_jit = jax.jit(jax.vmap(loss_single, in_axes=(0, None)))

    beta1, beta2, ae = 0.9, 0.999, 1e-8

    def one_step(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_single, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        V = V - lr * (m / (1 - beta1 ** step_num)) \
              / (jnp.sqrt(s / (1 - beta2 ** step_num)) + ae)
        return V, m, s
    one_step_jit = jax.jit(one_step)

    @jax.jit
    def run_chunk_scan(V, m, s, start_f, tau_chunk):
        def body(carry, x):
            V_, m_, s_ = carry; step_num, tau_val = x
            return one_step(V_, m_, s_, step_num, tau_val), None
        step_nums = start_f + jnp.arange(1, tau_chunk.shape[0] + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s), (step_nums, tau_chunk))
        return final

    def run_chunk_loop(V, m, s, start, tau_chunk):
        for i in range(tau_chunk.shape[0]):
            V, m, s = one_step_jit(V, m, s, jnp.float32(start + i + 1), tau_chunk[i])
        return V, m, s

    rchunk = run_chunk_scan if use_scan else run_chunk_loop

    B = verts_batch.shape[0]
    V = pca_normalize_batch(verts_batch)
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)

    tau_sched = (tau_start *
                 (tau_end / tau_start) ** np.linspace(0, 1, steps)).astype(np.float32)

    best_verts = np.asarray(V)
    best_ix = np.full(B, 10**9, dtype=np.int32)

    done = 0
    while done < steps:
        k = min(chunk_size, steps - done)
        tau_chunk = jnp.asarray(tau_sched[done: done + k])
        if use_scan:
            V, m, s = rchunk(V, m, s, jnp.float32(done), tau_chunk)
        else:
            V, m, s = rchunk(V, m, s, done, tau_chunk)
        V = pca_normalize_batch(V)
        done += k

        V_np = np.asarray(V)
        n_ix_all = count_intersections_given_faces(V_np, faces, EDGES)
        improved = n_ix_all < best_ix
        best_ix = np.where(improved, n_ix_all, best_ix)
        best_verts = np.where(improved[:, None, None], V_np, best_verts)

        if done % log_every < chunk_size or done == steps:
            losses = np.asarray(loss_batch_jit(V, tau_chunk[-1]))
            print(f"  step {done:5d}  tau={float(tau_chunk[-1]):.4f}  "
                  f"min_loss={float(losses.min()):.3g}  "
                  f"mean_loss={float(losses.mean()):.3g}  "
                  f"min_ix={int(n_ix_all.min())}  mean_ix={float(n_ix_all.mean()):.2f}  "
                  f"best_ever={int(best_ix.min())}  zero_seen={int((best_ix == 0).sum())}/{B}")

    return best_verts, best_ix


def write_obj(path, verts, faces):
    with open(path, 'w') as fh:
        fh.write(f'# Neighborly polyhedron on {len(verts)} vertices\n')
        for p in verts:
            fh.write(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        for f in faces:
            fh.write(f'f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--structure-tries', type=int, default=50000)
    ap.add_argument('--max-structure-seeds', type=int, default=10)
    ap.add_argument('--structure-workers', type=int, default=1)
    ap.add_argument('--allow-pseudomanifold', action='store_true',
                    help="Accept pseudo-manifold face sets (edges are in 2 faces but "
                         "some vertex link is a union of cycles instead of one cycle). "
                         "Manifold K_12 triangulations are very rare under the greedy; "
                         "pseudo-manifolds reach 44 faces easily and are still valid "
                         "face sets for the intersection question.")
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--tau-start', type=float, default=0.5)
    ap.add_argument('--tau-end', type=float, default=1e-3)
    ap.add_argument('--chunk-size', type=int, default=250)
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--no-scan', action='store_true')
    ap.add_argument('--out', default='neighborly')
    args = ap.parse_args()

    print(f"JAX backend: {jax.devices()}")

    N = args.n
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    if (N * (N - 1)) % 6 != 0:
        print(f"K_{N} is not neighborly-triangulable (N*(N-1) not div by 6).")
        return
    target_F = N * (N - 1) // 3
    chi = N - len(EDGES) + target_F
    if chi % 2 != 0:
        print(f"chi={chi} is odd; no orientable realization possible for N={N}.")
        return
    genus = (2 - chi) // 2
    print(f"Target: V={N}, E={len(EDGES)}, F={target_F}, chi={chi}, genus={genus}")

    tri_arr, edge_arr, pair_t, pair_e = make_tri_edge_disjoint_pairs(N, TRIANGLES, EDGES)
    print(f"  {len(pair_t)} (triangle, non-incident-edge) pairs")

    # Find a manifold triangulation.
    faces = None
    t0 = time.time()
    for restart in range(args.max_structure_seeds):
        this_seed = args.seed + restart * 1009
        print(f"\n== structure search (seed {this_seed}) ==")
        _, faces = find_structure(N, EDGES, TRIANGLES, target_F,
                                   tri_arr, edge_arr, pair_t, pair_e,
                                   this_seed, args.structure_tries,
                                   n_workers=args.structure_workers,
                                   require_manifold=not args.allow_pseudomanifold)
        if faces is not None:
            break
    if faces is None:
        print(f"\nGave up after {args.max_structure_seeds * args.structure_tries} "
              f"total greedy draws in {time.time() - t0:.1f}s.")
        return
    print(f"found {len(faces)} faces in {time.time() - t0:.1f}s")
    for f in faces[:10]:
        print(f"  {f}")
    if len(faces) > 10:
        print(f"  ... and {len(faces) - 10} more")

    # Batched random restarts
    rng = np.random.default_rng(args.seed + 100003)
    V0 = rng.standard_normal((args.batch, N, 3)).astype(np.float32)
    init_counts = count_intersections_given_faces(V0, faces, EDGES)
    print(f"\n== optimizer: {args.batch} restarts, {args.steps} steps ==")
    print(f"  initial intersections: min={init_counts.min()} "
          f"mean={init_counts.mean():.1f} max={init_counts.max()}")

    t0 = time.time()
    best_verts_per, best_ix_per = optimize_batch(
        V0, faces, EDGES,
        steps=args.steps, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end,
        chunk_size=args.chunk_size, log_every=args.log_every,
        use_scan=not args.no_scan,
    )
    dt = time.time() - t0
    print(f"optimizer done in {dt:.1f}s ({args.steps / dt:.1f} steps/s)")

    i_star = int(best_ix_per.argmin())
    v_best = best_verts_per[i_star]
    best_n = int(best_ix_per[i_star])
    n_zero = int((best_ix_per == 0).sum())
    print(f"\n== best ==")
    print(f"  fewest intersections seen: {best_n}")
    print(f"  instances that reached 0:  {n_zero} / {args.batch}")

    np.save(f"{args.out}_vertices.npy", v_best)
    np.save(f"{args.out}_faces.npy", np.asarray(faces, dtype=np.int32))
    write_obj(f"{args.out}.obj", v_best, faces)
    with open(f"{args.out}.json", 'w') as fh:
        json.dump({
            "n": N, "genus": genus, "chi": chi,
            "faces": [list(f) for f in faces],
            "vertices": v_best.tolist(),
            "real_intersections": best_n,
            "batch_size": args.batch,
            "n_zero_instances": n_zero,
            "per_instance_best_intersections": best_ix_per.tolist(),
        }, fh, indent=2)
    print(f"\nwrote {args.out}_vertices.npy, {args.out}_faces.npy, "
          f"{args.out}.obj, {args.out}.json")


if __name__ == '__main__':
    main()
