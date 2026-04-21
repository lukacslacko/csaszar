"""Symmetry-restricted search: require Z_k rotation symmetry around the z-axis.

k = 2 gives 180° rotation. k = 3 gives 120°. Etc.

Vertex parameterization:
  - `n_axis` vertices sit on the z-axis (x = y = 0), fixed by the rotation.
  - The remaining N - n_axis vertices come in orbits of size k, each generated
    by applying the rotation R = R_z(2π/k) to a representative. The script
    numbers the orbit (n_axis + k*p, n_axis + k*p + 1, ..., n_axis + k*p + k-1).

Face selection picks whole sigma-orbits. Every optimization step projects
the vertex coordinates back onto the symmetric subspace, then renormalizes
only along the symmetry-invariant directions:

  - shift all z-coords by their mean (the xy centroid is already zero by
    symmetry),
  - for k=2, rotate every xy-pair by a common 2D rotation to diagonalise
    the xy covariance (the only rotational gauge compatible with Z_2);
    for k>=3 the xy cloud is already rotation-isotropic so no rotation,
  - scale axes to fit in [-1,1]^3: for k=2, independently; for k>=3,
    scale x and y together to preserve the rotational symmetry.

Optional `--polish` phase adds a symmetric polish loss (smooth-max of
cos^2(dihedral) and cos^2(edge-pair-angle), with the intersection penalty
kept active) on top of a clean result.
"""

import argparse
import json
import time
from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------

def build_sigma(N, n_axis, k):
    """Permutation sigma[v] = image of v under the group generator (rotation
    by 2π/k around the z-axis). Axis vertices are fixed."""
    if (N - n_axis) % k != 0:
        raise ValueError(f"N - n_axis = {N - n_axis} not divisible by k = {k}")
    sigma = list(range(N))
    for orbit_start in range(n_axis, N, k):
        for j in range(k):
            sigma[orbit_start + j] = orbit_start + ((j + 1) % k)
    return sigma


def orbit_of(v, sigma):
    """Return the sigma-orbit of vertex v as a tuple."""
    out = [v]
    cur = sigma[v]
    while cur != v:
        out.append(cur); cur = sigma[cur]
    return tuple(out)


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def project_symmetric_batch(verts_batch, N, n_axis, k):
    """Project each batch sample to the Z_k-symmetric subspace.

    For each orbit of size k, compute the canonical representative as the
    average of R^{-j} · v_{orbit_start + j}, then reconstruct v_{orbit_start + j} = R^j · canonical.
    Axis vertices are forced to (0, 0, z)."""
    v = np.asarray(verts_batch, dtype=np.float32).copy()    # (B, N, 3)
    theta = 2.0 * np.pi / k
    R_inv = np.stack([rotation_z(-j * theta) for j in range(k)])   # (k, 3, 3)
    R = np.stack([rotation_z(j * theta) for j in range(k)])        # (k, 3, 3)
    for a in range(n_axis):
        v[:, a, 0] = 0.0; v[:, a, 1] = 0.0
    for orbit_start in range(n_axis, N, k):
        canonical = np.zeros((v.shape[0], 3), dtype=np.float32)
        for j in range(k):
            canonical += np.einsum('ij,bj->bi', R_inv[j],
                                   v[:, orbit_start + j, :])
        canonical /= k
        for j in range(k):
            v[:, orbit_start + j, :] = np.einsum('ij,bj->bi', R[j], canonical)
    return jnp.asarray(v, dtype=jnp.float32)


def normalize_symmetric_batch(verts_batch, N, n_axis, k, min_scale=1e-2):
    """Gauge-fix: shift z to mean 0, xy-PCA if k==2, scale axes to [-1, 1]."""
    v = np.asarray(verts_batch, dtype=np.float32).copy()    # (B, N, 3)
    # Shift z only (xy centroid is 0 by symmetry after projection).
    mean_z = v[:, :, 2].mean(axis=1, keepdims=True)
    v[:, :, 2] -= mean_z
    if k == 2:
        # 2D PCA on xy (common rotation applied to all xy pairs).
        xy = v[:, :, :2]
        cov = np.einsum('bni,bnj->bij', xy, xy) / xy.shape[1]
        _, evecs = np.linalg.eigh(cov)
        R = np.swapaxes(evecs, -1, -2)
        v[:, :, :2] = np.einsum('bij,bnj->bni', R, xy)
        # Independent per-axis scaling (preserves Z_2).
        max_abs = np.maximum(np.max(np.abs(v), axis=1, keepdims=True), min_scale)
        v /= max_abs
    else:
        # Z_k for k >= 3: xy covariance is isotropic. Scale x & y together.
        max_xy = np.maximum(
            np.max(np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2), axis=1, keepdims=True),
            min_scale)[..., None]                        # (B, 1, 1)
        v[:, :, :2] /= max_xy
        max_z = np.maximum(np.max(np.abs(v[:, :, 2:3]), axis=1, keepdims=True),
                           min_scale)
        v[:, :, 2:3] /= max_z
    return jnp.asarray(v, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Geometry predicates (numpy)
# ---------------------------------------------------------------------------

def _sv(A, B, C, D):
    return np.dot(B - A, np.cross(C - A, D - A))

def _svvec(X, Y, Z, W):
    return np.einsum('...j,...j->...', Y - X, np.cross(Z - X, W - X))

def seg_crosses_tri(P, Q, A, B, C, tol=1e-9):
    vp = _sv(P, A, B, C); vq = _sv(Q, A, B, C)
    vab = _sv(P, Q, A, B); vbc = _sv(P, Q, B, C); vca = _sv(P, Q, C, A)
    plane = vp * vq < -tol
    inside = (vab > tol and vbc > tol and vca > tol) or \
             (vab < -tol and vbc < -tol and vca < -tol)
    return plane and inside


def count_intersections_batch(verts_batch, faces, EDGES, tol=1e-9):
    fvi, evi = [], []
    for face in faces:
        fset = set(face)
        for e in EDGES:
            if fset & set(e): continue
            fvi.append(face); evi.append(e)
    fvi = np.asarray(fvi); evi = np.asarray(evi)
    fv = verts_batch[:, fvi]; ev = verts_batch[:, evi]
    A = fv[..., 0, :]; B = fv[..., 1, :]; C = fv[..., 2, :]
    P = ev[..., 0, :]; Q = ev[..., 1, :]
    vp = _svvec(P, A, B, C); vq = _svvec(Q, A, B, C)
    vab = _svvec(P, Q, A, B); vbc = _svvec(P, Q, B, C); vca = _svvec(P, Q, C, A)
    plane = vp * vq < -tol
    pos = (vab > tol) & (vbc > tol) & (vca > tol)
    neg = (vab < -tol) & (vbc < -tol) & (vca < -tol)
    return (plane & (pos | neg)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Greedy face selection
# ---------------------------------------------------------------------------

def _sigma_face(face, sigma):
    return tuple(sorted(sigma[v] for v in face))

def _orbit_face(face, sigma):
    """Return the tuple (sorted) of sorted-face-triples forming the orbit."""
    seen = set()
    cur = tuple(sorted(face))
    while cur not in seen:
        seen.add(cur)
        cur = _sigma_face(cur, sigma)
    return tuple(sorted(seen))

def _tri_edges(face):
    a, b, c = face
    return (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c))))

def _vertex_link_is_cycle(v, faces_containing):
    link_edges, link_verts = [], set()
    for face in faces_containing:
        opp = [x for x in face if x != v]
        link_edges.append(tuple(sorted(opp))); link_verts.update(opp)
    if not link_verts: return False
    deg = {u: 0 for u in link_verts}
    for a, b in link_edges:
        deg[a] += 1; deg[b] += 1
    if any(d != 2 for d in deg.values()): return False
    adj = {u: [] for u in link_verts}
    for a, b in link_edges:
        adj[a].append(b); adj[b].append(a)
    start = next(iter(link_verts)); seen = {start}; stk = [start]
    while stk:
        u = stk.pop()
        for w in adj[u]:
            if w not in seen: seen.add(w); stk.append(w)
    return len(seen) == len(link_verts)


def greedy_select_symmetric(verts, N, EDGES, TRIANGLES, target_F, sigma, rng,
                             require_manifold=True):
    counts = np.zeros(len(TRIANGLES), dtype=np.int32)
    for ti, tri in enumerate(TRIANGLES):
        A, B, C = verts[list(tri)]
        for e in EDGES:
            if set(e) & set(tri): continue
            P, Q = verts[e[0]], verts[e[1]]
            if seg_crosses_tri(P, Q, A, B, C):
                counts[ti] += 1
    tri_idx_of = {tuple(t): i for i, t in enumerate(TRIANGLES)}

    orbits, seen = [], set()
    for ti, tri in enumerate(TRIANGLES):
        tup = tuple(sorted(tri))
        if tup in seen: continue
        orbit = list(_orbit_face(tup, sigma))
        for t in orbit: seen.add(t)
        orbit_count = sum(counts[tri_idx_of[t]] for t in orbit)
        orbits.append((orbit, orbit_count))

    tiebreak = rng.random(len(orbits))
    order = sorted(range(len(orbits)), key=lambda i: (orbits[i][1], tiebreak[i]))
    edge_deg = {e: 0 for e in EDGES}
    selected = []
    for oi in order:
        orbit = orbits[oi][0]
        inc = {}
        for tri in orbit:
            for e in _tri_edges(tri): inc[e] = inc.get(e, 0) + 1
        if any(edge_deg[e] + v > 2 for e, v in inc.items()):
            continue
        for tri in orbit:
            selected.append(tri)
            for e in _tri_edges(tri): edge_deg[e] += 1
        if len(selected) >= target_F: break

    if len(selected) != target_F or any(d != 2 for d in edge_deg.values()):
        return selected, False
    if not require_manifold:
        return selected, True
    incident = {v: [] for v in range(N)}
    for face in selected:
        for v in face: incident[v].append(face)
    for v in range(N):
        if not _vertex_link_is_cycle(v, incident[v]):
            return selected, False
    return selected, True


# ---------------------------------------------------------------------------
# JAX objectives
# ---------------------------------------------------------------------------

def _sv_j(A, B, C, D):
    return jnp.dot(B - A, jnp.cross(C - A, D - A))


def _pair_penalty(face_xyz, edge_xyz, tau):
    A, B, C = face_xyz[0], face_xyz[1], face_xyz[2]
    P, Q = edge_xyz[0], edge_xyz[1]
    vp = _sv_j(P, A, B, C); vq = _sv_j(Q, A, B, C)
    vab = _sv_j(P, Q, A, B); vbc = _sv_j(P, Q, B, C); vca = _sv_j(P, Q, C, A)
    conds = jnp.stack([-vp * vq, vab * vbc, vbc * vca])
    smin = -tau * jax.scipy.special.logsumexp(-conds / tau)
    return tau * jax.nn.softplus(smin / tau)


def build_pair_index(faces, EDGES):
    fi, ei = [], []
    for face in faces:
        fset = set(face)
        for e in EDGES:
            if fset & set(e): continue
            fi.append(face); ei.append(e)
    return jnp.asarray(fi, dtype=jnp.int32), jnp.asarray(ei, dtype=jnp.int32)


def make_loss(face_idx, edge_idx):
    def loss(V, tau):
        fv = V[face_idx]; ev = V[edge_idx]
        return jnp.sum(jax.vmap(lambda f, e: _pair_penalty(f, e, tau))(fv, ev))
    return loss


def build_dihedral_index(faces, N):
    edge_to_faces = {}
    for face in faces:
        a, b, c = face
        for u, v in [(a, b), (b, c), (a, c)]:
            e = (min(u, v), max(u, v))
            edge_to_faces.setdefault(e, []).append(face)
    i_arr, j_arr, k_arr, l_arr = [], [], [], []
    for (i, j), fs in edge_to_faces.items():
        if len(fs) != 2: continue
        f1, f2 = fs
        k = next(v for v in f1 if v != i and v != j)
        l = next(v for v in f2 if v != i and v != j)
        i_arr.append(i); j_arr.append(j); k_arr.append(k); l_arr.append(l)
    return (jnp.array(i_arr, dtype=jnp.int32),
            jnp.array(j_arr, dtype=jnp.int32),
            jnp.array(k_arr, dtype=jnp.int32),
            jnp.array(l_arr, dtype=jnp.int32))


def build_edge_pair_index(N):
    v_arr, a_arr, b_arr = [], [], []
    for v in range(N):
        others = [i for i in range(N) if i != v]
        for p, a in enumerate(others):
            for b in others[p + 1:]:
                v_arr.append(v); a_arr.append(a); b_arr.append(b)
    return (jnp.array(v_arr, dtype=jnp.int32),
            jnp.array(a_arr, dtype=jnp.int32),
            jnp.array(b_arr, dtype=jnp.int32))


def dihedral_cos(V, di):
    i_a, j_a, k_a, l_a = di
    pi = V[i_a]; pj = V[j_a]; pk = V[k_a]; pl = V[l_a]
    e = pj - pi
    u = pk - pi
    v = pl - pi
    def one(e, u, v):
        e2 = jnp.dot(e, e) + 1e-12
        u_perp = u - (jnp.dot(u, e) / e2) * e
        v_perp = v - (jnp.dot(v, e) / e2) * e
        num = jnp.dot(u_perp, v_perp)
        den = jnp.sqrt(jnp.dot(u_perp, u_perp) * jnp.dot(v_perp, v_perp)) + 1e-12
        return num / den
    return jax.vmap(one)(e, u, v)


def edge_pair_cos2(V, ep):
    v_a, a_a, b_a = ep
    pv = V[v_a]; pa = V[a_a]; pb = V[b_a]
    def one(pv, pa, pb):
        da = pa - pv; db = pb - pv
        num = jnp.dot(da, db)
        den = jnp.dot(da, da) * jnp.dot(db, db) + 1e-12
        return (num * num) / den
    return jax.vmap(one)(pv, pa, pb)


def make_polish_loss(face_idx, edge_idx, dih_idx, ep_idx):
    intersect = make_loss(face_idx, edge_idx)
    def polish(V, tau_ix, tau_dih, tau_coll, w_dih, w_coll):
        l_ix = intersect(V, tau_ix)
        cos_d = dihedral_cos(V, dih_idx)
        cos2_d = cos_d * cos_d
        smx_d = tau_dih * jax.scipy.special.logsumexp(cos2_d / tau_dih)
        cos2_e = edge_pair_cos2(V, ep_idx)
        smx_e = tau_coll * jax.scipy.special.logsumexp(cos2_e / tau_coll)
        return l_ix + w_dih * smx_d + w_coll * smx_e
    return polish


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def run_adam_scan(V, faces, EDGES, N, n_axis, k,
                   steps, lr, tau_start, tau_end,
                   chunk_size, log_every, polish_opts=None):
    """Adam + periodic project/normalize. `polish_opts` is either None for
    plain intersection optimization, or a dict
        {tau_ix, tau_dih_start/end, tau_coll_start/end, w_dih, w_coll}
    for phase-2 polish."""
    face_idx, edge_idx = build_pair_index(faces, EDGES)
    dih_idx = build_dihedral_index(faces, N)
    ep_idx = build_edge_pair_index(N)

    if polish_opts is None:
        loss_single = make_loss(face_idx, edge_idx)
        grad_single = jax.grad(loss_single)
        def one_step(V, m, s, step_num, tau):
            g = jax.vmap(grad_single, in_axes=(0, None))(V, tau)
            m = BETA1 * m + (1 - BETA1) * g
            s = BETA2 * s + (1 - BETA2) * g * g
            V = V - lr * (m / (1 - BETA1 ** step_num)) \
                  / (jnp.sqrt(s / (1 - BETA2 ** step_num)) + ADAM_EPS)
            return V, m, s
    else:
        polish_loss = make_polish_loss(face_idx, edge_idx, dih_idx, ep_idx)
        grad_polish = jax.grad(polish_loss)
        tau_ix = polish_opts["tau_ix"]
        w_dih, w_coll = polish_opts["w_dih"], polish_opts["w_coll"]
        def one_step(V, m, s, step_num, tau_pair):
            tau_dih, tau_coll = tau_pair[0], tau_pair[1]
            g = jax.vmap(grad_polish, in_axes=(0, None, None, None, None, None))(
                V, tau_ix, tau_dih, tau_coll, w_dih, w_coll)
            m = BETA1 * m + (1 - BETA1) * g
            s = BETA2 * s + (1 - BETA2) * g * g
            V = V - lr * (m / (1 - BETA1 ** step_num)) \
                  / (jnp.sqrt(s / (1 - BETA2 ** step_num)) + ADAM_EPS)
            return V, m, s

    @jax.jit
    def run_chunk(V, m, s, start_f, tau_chunk):
        def body(carry, x):
            V_, m_, s_ = carry; sn, tv = x
            return one_step(V_, m_, s_, sn, tv), None
        step_nums = start_f + jnp.arange(1, tau_chunk.shape[0] + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s), (step_nums, tau_chunk))
        return final

    loss_batch_jit = jax.jit(jax.vmap(make_loss(face_idx, edge_idx),
                                       in_axes=(0, None)))

    if polish_opts is None:
        tau_sched_a = (tau_start *
                       (tau_end / tau_start) ** np.linspace(0, 1, steps)).astype(np.float32)
        tau_sched = tau_sched_a[:, None]  # (steps, 1) same shape as polish for JIT
    else:
        tau_dih_sched = (polish_opts["tau_dih_start"] *
                         (polish_opts["tau_dih_end"] / polish_opts["tau_dih_start"])
                         ** np.linspace(0, 1, steps)).astype(np.float32)
        tau_coll_sched = (polish_opts["tau_coll_start"] *
                          (polish_opts["tau_coll_end"] / polish_opts["tau_coll_start"])
                          ** np.linspace(0, 1, steps)).astype(np.float32)
        tau_sched = np.stack([tau_dih_sched, tau_coll_sched], axis=1)  # (steps, 2)

    B = V.shape[0]
    m = jnp.zeros_like(V); s = jnp.zeros_like(V)
    best_verts = np.asarray(V)
    best_ix = np.full(B, 10**9, dtype=np.int32)

    done = 0
    while done < steps:
        chunk_k = min(chunk_size, steps - done)
        tau_chunk = jnp.asarray(tau_sched[done: done + chunk_k])
        if polish_opts is None:
            # Take only the first column, scalar tau per step
            tau_pass = tau_chunk[:, 0]
            V, m, s = run_chunk(V, m, s, jnp.float32(done), tau_pass)
        else:
            V, m, s = run_chunk(V, m, s, jnp.float32(done), tau_chunk)
        V = project_symmetric_batch(V, N, n_axis, k)
        V = normalize_symmetric_batch(V, N, n_axis, k)
        done += chunk_k

        V_np = np.asarray(V)
        n_ix = count_intersections_batch(V_np, faces, EDGES)
        improved = n_ix < best_ix
        best_ix = np.where(improved, n_ix, best_ix)
        best_verts = np.where(improved[:, None, None], V_np, best_verts)

        if done % log_every < chunk_k or done == steps:
            if polish_opts is None:
                losses = np.asarray(loss_batch_jit(V, tau_chunk[-1, 0]))
            else:
                losses = np.asarray(loss_batch_jit(V, polish_opts["tau_ix"]))
            n_zero = int((n_ix == 0).sum())
            print(f"  step {done:5d}  tau={float(tau_chunk[-1, 0]):.4f}  "
                  f"ix_loss={float(losses.min()):.3g}..{float(losses.mean()):.3g}  "
                  f"min_ix={int(n_ix.min())}  mean_ix={float(n_ix.mean()):.2f}  "
                  f"best_ever={int(best_ix.min())}  zero_seen={n_zero}/{B}")
    return best_verts, best_ix


BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _initial_symmetric_batch(N, n_axis, k, batch, rng):
    V = rng.standard_normal((batch, N, 3)).astype(np.float32)
    return np.asarray(project_symmetric_batch(jnp.asarray(V), N, n_axis, k))


def compute_dihedrals(v, faces, N):
    di = build_dihedral_index(faces, N)
    cos_d = np.clip(np.asarray(dihedral_cos(v, di)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_d))


def compute_edge_pair_devs(v, N):
    ep = build_edge_pair_index(N)
    cos2 = np.clip(np.asarray(edge_pair_cos2(v, ep)), 0.0, 1.0)
    return np.degrees(np.arcsin(np.sqrt(np.clip(1 - cos2, 0, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--k", type=int, default=2,
                    help="Symmetry order: Z_k rotation around the z-axis.")
    ap.add_argument("--axis-vertices", type=int, default=-1,
                    help="Vertices on the rotation axis (must satisfy (N - n_axis) %% k == 0). "
                         "-1 picks 1 for odd N, 0 for even N.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--structure-tries", type=int, default=2000)
    ap.add_argument("--max-structure-seeds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=1e-3)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--allow-pseudomanifold", action="store_true")
    ap.add_argument("--polish", action="store_true",
                    help="After phase 1 finds clean results, run polish "
                         "(max symmetric dihedral/collinearity).")
    ap.add_argument("--polish-steps", type=int, default=2000)
    ap.add_argument("--polish-lr", type=float, default=1e-3)
    ap.add_argument("--polish-tau-ix", type=float, default=5e-4)
    ap.add_argument("--polish-tau-dih-start", type=float, default=0.3)
    ap.add_argument("--polish-tau-dih-end", type=float, default=0.03)
    ap.add_argument("--polish-tau-coll-start", type=float, default=0.3)
    ap.add_argument("--polish-tau-coll-end", type=float, default=0.03)
    ap.add_argument("--polish-w-dih", type=float, default=1.0)
    ap.add_argument("--polish-w-coll", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    N, k = args.n, args.k
    if args.axis_vertices == -1:
        n_axis = 1 if N % k == 1 else 0
    else:
        n_axis = args.axis_vertices
    EDGES = list(combinations(range(N), 2))
    TRIANGLES = list(combinations(range(N), 3))
    if (N * (N - 1)) % 6 != 0:
        print(f"N={N} has no neighborly triangulation")
        return
    if (N - n_axis) % k != 0:
        print(f"Cannot have Z_{k} symmetry with n_axis={n_axis}; "
              f"(N - n_axis) = {N - n_axis} not divisible by {k}.")
        return
    target_F = N * (N - 1) // 3
    chi = N - len(EDGES) + target_F
    genus = (2 - chi) // 2 if chi <= 2 else None
    sigma = build_sigma(N, n_axis, k)
    print(f"Target: V={N}, E={len(EDGES)}, F={target_F}, chi={chi}, genus={genus}")
    print(f"Z_{k} symmetry: axis={n_axis}, orbits={(N - n_axis)//k}")
    print(f"sigma = {sigma}")

    # Find structure.
    faces = None
    for restart in range(args.max_structure_seeds):
        seed = args.seed + restart * 997
        rng = np.random.default_rng(seed)
        print(f"\n== structure search (seed {seed}) ==")
        for t in range(args.structure_tries):
            raw = rng.random((N, 3)) * 2 - 1
            verts = np.asarray(project_symmetric_batch(
                jnp.asarray(raw)[None], N, n_axis, k))[0]
            f, ok = greedy_select_symmetric(
                verts, N, EDGES, TRIANGLES, target_F, sigma, rng,
                require_manifold=not args.allow_pseudomanifold)
            if ok:
                faces = f; break
        if faces is not None:
            print(f"  hit on seed {seed} after {t + 1} draws")
            break
    if faces is None:
        print(f"failed to find a Z_{k}-symmetric triangulation")
        return

    print(f"\n{len(faces)} faces (orbit groups):")
    shown = set()
    for face in faces:
        tup = tuple(sorted(face))
        if tup in shown: continue
        orbit = list(_orbit_face(tup, sigma))
        for t in orbit: shown.add(t)
        if len(orbit) == 1:
            print(f"  fixed: {orbit[0]}")
        else:
            print(f"  orbit ({len(orbit)}): {orbit}")

    # Phase 1: intersection removal.
    rng2 = np.random.default_rng(args.seed + 100003)
    V0 = _initial_symmetric_batch(N, n_axis, k, args.batch, rng2)
    init_counts = count_intersections_batch(V0, faces, EDGES)
    print(f"\n== phase 1 ({args.batch} x {args.steps}) ==")
    print(f"  initial ix: min={init_counts.min()} mean={init_counts.mean():.1f} "
          f"max={init_counts.max()}")
    V_proj = project_symmetric_batch(V0, N, n_axis, k)
    V_norm = normalize_symmetric_batch(V_proj, N, n_axis, k)
    t0 = time.time()
    best_verts, best_ix = run_adam_scan(
        V_norm, faces, EDGES, N, n_axis, k,
        steps=args.steps, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end,
        chunk_size=args.chunk_size, log_every=args.log_every)
    print(f"  phase 1 done in {time.time() - t0:.1f}s")
    i_star = int(best_ix.argmin())
    n_zero = int((best_ix == 0).sum())
    print(f"  best phase-1 n_ix = {int(best_ix[i_star])}  (zero-instances: {n_zero})")

    # Phase 2: polish (optional).
    final_v = best_verts[i_star]
    final_n_ix = int(best_ix[i_star])
    phase2_ran = False
    if args.polish and n_zero > 0:
        print(f"\n== phase 2 polish ==")
        mask = (best_ix == 0)
        V_clean = best_verts[mask]
        # Apply symmetric projection+normalize to be safe
        V_clean = np.asarray(project_symmetric_batch(jnp.asarray(V_clean), N, n_axis, k))
        V_clean = np.asarray(normalize_symmetric_batch(jnp.asarray(V_clean), N, n_axis, k))
        polish_opts = dict(
            tau_ix=args.polish_tau_ix,
            tau_dih_start=args.polish_tau_dih_start,
            tau_dih_end=args.polish_tau_dih_end,
            tau_coll_start=args.polish_tau_coll_start,
            tau_coll_end=args.polish_tau_coll_end,
            w_dih=args.polish_w_dih,
            w_coll=args.polish_w_coll,
        )
        t0 = time.time()
        polished_verts, polished_ix = run_adam_scan(
            jnp.asarray(V_clean, dtype=jnp.float32), faces, EDGES, N, n_axis, k,
            steps=args.polish_steps, lr=args.polish_lr,
            tau_start=args.polish_tau_dih_start, tau_end=args.polish_tau_dih_end,
            chunk_size=args.chunk_size, log_every=args.log_every,
            polish_opts=polish_opts)
        print(f"  phase 2 done in {time.time() - t0:.1f}s")

        # Score: largest min(min_dih, min_edge_pair) across clean polished.
        best_score = -1; best_idx = None
        for i, v in enumerate(polished_verts):
            if polished_ix[i] != 0: continue
            angs = compute_dihedrals(v, faces, N)
            eps = compute_edge_pair_devs(v, N)
            sc = min(float(angs.min()), float(eps.min()))
            if sc > best_score:
                best_score = sc; best_idx = i
        if best_idx is not None:
            final_v = polished_verts[best_idx]
            final_n_ix = 0
            phase2_ran = True
            angs = compute_dihedrals(final_v, faces, N)
            eps = compute_edge_pair_devs(final_v, N)
            print(f"  polished: dihedral {angs.min():.2f}° .. {angs.max():.2f}° "
                  f"(mean {angs.mean():.2f}°)")
            print(f"  polished: edge-pair dev min={eps.min():.2f}° max={eps.max():.2f}° "
                  f"mean={eps.mean():.2f}°")

    # Report.
    if args.out:
        np.save(f"{args.out}_vertices.npy", final_v)
        np.save(f"{args.out}_faces.npy", np.asarray(faces, dtype=np.int32))
        with open(f"{args.out}.obj", "w") as fh:
            fh.write(f"# Z_{k}-symmetric N={N} (ix={final_n_ix}{', polished' if phase2_ran else ''})\n")
            for p in final_v:
                fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for f in faces:
                fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        payload = {
            "n": N, "k": k, "n_axis": n_axis, "sigma": sigma,
            "genus": genus, "chi": chi,
            "vertices": final_v.tolist(),
            "faces": [list(f) for f in faces],
            "real_intersections": final_n_ix,
            "polished": phase2_ran,
        }
        if phase2_ran:
            angs = compute_dihedrals(final_v, faces, N)
            eps = compute_edge_pair_devs(final_v, N)
            payload.update({
                "min_dihedral_deg": float(angs.min()),
                "mean_dihedral_deg": float(angs.mean()),
                "max_dihedral_deg": float(angs.max()),
                "min_edge_pair_dev_deg": float(eps.min()),
                "max_edge_pair_dev_deg": float(eps.max()),
                "mean_edge_pair_dev_deg": float(eps.mean()),
            })
        with open(f"{args.out}.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}.{{npy,obj,json}}")


if __name__ == "__main__":
    main()
