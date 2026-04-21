"""Rediscover the Csaszar polyhedron.

Place 7 points in the unit cube at random, greedily pick 14 triangular faces
that minimize edge-through-face intersections subject to each edge being used
by exactly two faces, then relax the vertex positions with gradient descent
(JAX / Metal) against a smooth intersection-badness objective, periodically
re-normalizing via PCA so the polyhedron can't collapse to a lower-dimensional
shape.

Combinatorics:
  V = 7, E = C(7,2) = 21, F = 14, Euler characteristic V - E + F = 0 (torus).
  Every pair of vertices is an edge (the surface is neighborly).
"""

import os
# NB: Metal backend works (use JAX_PLATFORMS=metal with --no-scan), but for
# this small problem the per-dispatch overhead dominates and CPU is faster.
# If jax-metal is installed the default platform is already metal; set
# JAX_PLATFORMS=cpu explicitly to force CPU if you want the fast path.

import argparse
import json
import random
from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp

N = 7
EDGES = list(combinations(range(N), 2))                 # 21
TRIANGLES = list(combinations(range(N), 3))             # 35
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


# ---------------------------------------------------------------------------
# Exact geometry for combinatorial face selection (numpy, no autograd)
# ---------------------------------------------------------------------------

def signed_vol6(A, B, C, D):
    """6 * signed volume of tetrahedron ABCD."""
    return np.dot(B - A, np.cross(C - A, D - A))


def segment_crosses_triangle_interior(P, Q, A, B, C, tol=1e-9):
    """Robust predicate: does the open segment PQ cross the open triangle ABC?

    Uses the five signed-volume / orientation determinants so there is no
    division. Strictly inside on all counts means a true interior crossing.
    """
    vp  = signed_vol6(P, A, B, C)
    vq  = signed_vol6(Q, A, B, C)
    vab = signed_vol6(P, Q, A, B)
    vbc = signed_vol6(P, Q, B, C)
    vca = signed_vol6(P, Q, C, A)
    plane = vp * vq < -tol
    inside = (vab > tol and vbc > tol and vca > tol) or \
             (vab < -tol and vbc < -tol and vca < -tol)
    return plane and inside


# ---------------------------------------------------------------------------
# Greedy face selection
# ---------------------------------------------------------------------------

def intersection_counts(verts):
    """For each of the 35 triangles, count how many of the 18 non-incident
    segments cross its interior."""
    counts = np.zeros(len(TRIANGLES), dtype=np.int32)
    for ti, tri in enumerate(TRIANGLES):
        A, B, C = verts[list(tri)]
        for e in EDGES:
            if set(e) & set(tri):
                continue
            P, Q = verts[e[0]], verts[e[1]]
            if segment_crosses_triangle_interior(P, Q, A, B, C):
                counts[ti] += 1
    return counts


def _triangle_edges(tri):
    a, b, c = tri
    return (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c))))


def vertex_link_is_disk(vertex, faces_containing):
    """Check the star of `vertex` forms a cone (i.e. link is a single cycle).

    Each incident face contributes an edge of the link (the opposite edge of
    the triangle). A closed manifold disk means every link-vertex has degree
    exactly 2 and the link graph is connected.
    """
    # Link edges: for face (v, a, b) the link edge is (a, b)
    link_edges = []
    link_verts = set()
    for face in faces_containing:
        opp = [x for x in face if x != vertex]
        link_edges.append(tuple(sorted(opp)))
        link_verts.update(opp)
    # Each link vertex must appear in exactly 2 link edges.
    deg = {v: 0 for v in link_verts}
    for (a, b) in link_edges:
        deg[a] += 1
        deg[b] += 1
    if any(d != 2 for d in deg.values()):
        return False
    # Connectivity (single cycle).
    adj = {v: [] for v in link_verts}
    for (a, b) in link_edges:
        adj[a].append(b)
        adj[b].append(a)
    start = next(iter(link_verts))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(link_verts)


def greedy_select_faces(verts, rng):
    """Eager algorithm described in the task: pick triangles in order of
    fewest interior intersections (random tiebreak), skipping any triangle
    that would give one of its edges a third face. Stops when 14 faces are
    selected. Returns faces and whether the result is a valid 2-manifold
    triangulation of the torus (every edge used twice, every vertex link a
    cycle).
    """
    counts = intersection_counts(verts)
    tiebreak = rng.random(len(TRIANGLES))
    order = sorted(range(len(TRIANGLES)), key=lambda i: (int(counts[i]), tiebreak[i]))

    edge_deg = {e: 0 for e in EDGES}
    selected = []
    for idx in order:
        tri = TRIANGLES[idx]
        te = _triangle_edges(tri)
        if any(edge_deg[e] >= 2 for e in te):
            continue
        selected.append(tri)
        for e in te:
            edge_deg[e] += 1
        if len(selected) == 14:
            break

    every_edge_twice = len(selected) == 14 and all(d == 2 for d in edge_deg.values())
    if not every_edge_twice:
        return selected, False

    # Check vertex links are disks (i.e. manifold).
    incident = {v: [] for v in range(N)}
    for face in selected:
        for v in face:
            incident[v].append(face)
    for v in range(N):
        if not vertex_link_is_disk(v, incident[v]):
            return selected, False
    return selected, True


def find_combinatorial_structure(seed, max_tries=2000, verbose=True):
    rng = np.random.default_rng(seed)
    for t in range(max_tries):
        verts = rng.random((N, 3)) * 2 - 1
        faces, ok = greedy_select_faces(verts, rng)
        if ok:
            if verbose:
                print(f"  greedy found valid torus triangulation after {t+1} draws")
            return verts, faces
    return None, None


# ---------------------------------------------------------------------------
# JAX objective: smooth intersection badness
# ---------------------------------------------------------------------------

def _signed_vol6_j(A, B, C, D):
    return jnp.dot(B - A, jnp.cross(C - A, D - A))


def _pair_penalty(face_xyz, edge_xyz, tau):
    """Smooth penalty for segment-crosses-triangle-interior.

    The segment PQ crosses the open triangle ABC iff every one of
        (-vp * vq),  (vab * vbc),  (vbc * vca)
    is strictly positive (these are the three sign-product conditions on the
    plane and triangle side tests from signed_vol6). Equivalently
        m := min( -vp*vq,  vab*vbc,  vbc*vca )
    is positive iff the crossing occurs, and the magnitude of m measures how
    deep the crossing is. We report a smooth version of max(0, m):
        penalty = tau * softplus( smooth_min(...) / tau )
    so the gradient stays linear (non-vanishing) even for deep crossings, and
    goes to zero cleanly once any of the three conditions flips sign."""
    A, B, C = face_xyz[0], face_xyz[1], face_xyz[2]
    P, Q = edge_xyz[0], edge_xyz[1]
    vp  = _signed_vol6_j(P, A, B, C)
    vq  = _signed_vol6_j(Q, A, B, C)
    vab = _signed_vol6_j(P, Q, A, B)
    vbc = _signed_vol6_j(P, Q, B, C)
    vca = _signed_vol6_j(P, Q, C, A)
    conds = jnp.stack([-vp * vq, vab * vbc, vbc * vca])
    smin = -tau * jax.scipy.special.logsumexp(-conds / tau)
    return tau * jax.nn.softplus(smin / tau)


def build_pair_index(faces):
    """For every (face, edge-not-touching-that-face) pair, precompute the
    6 vertex indices needed. Each face has 21 - 3 - 3*4 = 6 disjoint edges,
    so 14 * 6 = 84 pairs in total."""
    face_idx, edge_idx = [], []
    for face in faces:
        fset = set(face)
        for e in EDGES:
            if fset & set(e):
                continue
            face_idx.append(face)
            edge_idx.append(e)
    return jnp.array(face_idx, dtype=jnp.int32), jnp.array(edge_idx, dtype=jnp.int32)


def make_loss(face_idx, edge_idx):
    def loss(verts, tau):
        fv = verts[face_idx]           # (P, 3, 3)
        ev = verts[edge_idx]           # (P, 2, 3)
        return jnp.sum(jax.vmap(lambda f, e: _pair_penalty(f, e, tau))(fv, ev))
    return loss


# ---------------------------------------------------------------------------
# Dihedral angle machinery (for the polish phase)
# ---------------------------------------------------------------------------

def build_dihedral_index(faces):
    """For each interior edge (i, j) shared by two faces, record the endpoints
    i, j and the two opposite-vertex indices k, l (one from each face). Returns
    four int32 arrays of length 21."""
    edge_to_faces = {}
    for face in faces:
        a, b, c = face
        for u, v in [(a, b), (b, c), (a, c)]:
            e = (min(u, v), max(u, v))
            edge_to_faces.setdefault(e, []).append(face)
    i_arr, j_arr, k_arr, l_arr = [], [], [], []
    for (i, j), fs in edge_to_faces.items():
        assert len(fs) == 2, f"edge {(i, j)} incident to {len(fs)} faces"
        f1, f2 = fs
        k = next(v for v in f1 if v != i and v != j)
        l = next(v for v in f2 if v != i and v != j)
        i_arr.append(i); j_arr.append(j); k_arr.append(k); l_arr.append(l)
    return (jnp.array(i_arr, dtype=jnp.int32),
            jnp.array(j_arr, dtype=jnp.int32),
            jnp.array(k_arr, dtype=jnp.int32),
            jnp.array(l_arr, dtype=jnp.int32))


def _dihedral_cos_one(pi, pj, pk, pl):
    """cos of the geometric dihedral angle at edge (pi -> pj) between faces
    (i, j, k) and (i, j, l). Uses the projected-opposite-vertex formulation so
    the answer is in [-1, 1]: -1 means the two triangles are coplanar and
    flattened open, +1 means they are folded back on top of each other
    ("almost touching")."""
    e = pj - pi
    u = pk - pi
    v = pl - pi
    e2 = jnp.dot(e, e) + 1e-12
    u_perp = u - (jnp.dot(u, e) / e2) * e
    v_perp = v - (jnp.dot(v, e) / e2) * e
    num = jnp.dot(u_perp, v_perp)
    den = jnp.sqrt(jnp.dot(u_perp, u_perp) * jnp.dot(v_perp, v_perp)) + 1e-12
    return num / den


def dihedral_cos(verts, dihedral_idx):
    """Return cos(dihedral) for every one of the 21 shared edges."""
    i_a, j_a, k_a, l_a = dihedral_idx
    pi = verts[i_a]; pj = verts[j_a]; pk = verts[k_a]; pl = verts[l_a]
    return jax.vmap(_dihedral_cos_one)(pi, pj, pk, pl)


def build_edge_pair_index(n=N):
    """For every unordered triple (v, a, b) with a != v, b != v, a < b, record
    the three vertex indices. At each of the 7 vertices this gives C(6,2)=15
    pairs-of-edges-meeting-at-that-vertex, so 7*15 = 105 triples in total.
    The angle between edges va and vb is what we want to keep bounded away
    from 0 and pi."""
    v_arr, a_arr, b_arr = [], [], []
    for v in range(n):
        others = [i for i in range(n) if i != v]
        for p, a in enumerate(others):
            for b in others[p + 1:]:
                v_arr.append(v); a_arr.append(a); b_arr.append(b)
    return (jnp.array(v_arr, dtype=jnp.int32),
            jnp.array(a_arr, dtype=jnp.int32),
            jnp.array(b_arr, dtype=jnp.int32))


def _edge_pair_cos2_one(pv, pa, pb):
    """cos^2 of the angle between edges v->a and v->b. In [0, 1]; equals 1
    exactly when the two edges are collinear (either both pointing the same
    direction, or one being the continuation of the other through v)."""
    da = pa - pv
    db = pb - pv
    num = jnp.dot(da, db)
    den = jnp.dot(da, da) * jnp.dot(db, db) + 1e-12
    return (num * num) / den


def edge_pair_cos2(verts, ep_idx):
    v_a, a_a, b_a = ep_idx
    pv = verts[v_a]; pa = verts[a_a]; pb = verts[b_a]
    return jax.vmap(_edge_pair_cos2_one)(pv, pa, pb)


def make_polish_loss(face_idx, edge_idx, dihedral_idx, edge_pair_idx):
    """Combined objective: keep the embedding clean (intersection penalty),
    push the largest cos(dihedral) down (maximize the smallest dihedral),
    and push the largest cos^2(edge-pair angle) down (keep every pair of
    edges meeting at a common vertex away from collinearity, which also
    prevents 3+ vertices from becoming collinear)."""
    intersect = make_loss(face_idx, edge_idx)

    def polish_loss(verts, tau_ix, tau_dih, tau_coll, w_dih, w_coll):
        l_cross = intersect(verts, tau_ix)
        coses = dihedral_cos(verts, dihedral_idx)
        soft_max_cos_dih = tau_dih * jax.scipy.special.logsumexp(coses / tau_dih)
        cos2 = edge_pair_cos2(verts, edge_pair_idx)
        soft_max_cos2 = tau_coll * jax.scipy.special.logsumexp(cos2 / tau_coll)
        return l_cross + w_dih * soft_max_cos_dih + w_coll * soft_max_cos2
    return polish_loss


# ---------------------------------------------------------------------------
# Gauge fixing: PCA normalization so the polyhedron stays spread out
# ---------------------------------------------------------------------------

def pca_normalize(verts, min_scale=1e-2):
    """Center at origin, rotate so principal axes align with world axes, and
    rescale so each axis lies in [-1, 1]. Because the badness objective is
    affine-invariant this does not change the loss; it just keeps the
    optimization well-conditioned and prevents the polyhedron from degenerating
    onto a plane (by ensuring all three principal axes have unit extent).

    Done in numpy because jax-metal has no SVD / eigh kernels. The result is
    returned as a jax array so the optimizer step stays on device."""
    v = np.asarray(verts, dtype=np.float32)
    centered = v - v.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ Vt.T
    max_abs = np.maximum(np.max(np.abs(rotated), axis=0), min_scale)
    return jnp.asarray(rotated / max_abs, dtype=jnp.float32)


def pca_normalize_batch(verts_batch, min_scale=1e-2):
    """Batched PCA-normalize. Shape (B, N, 3) -> (B, N, 3)."""
    v = np.asarray(verts_batch, dtype=np.float32)
    centered = v - v.mean(axis=1, keepdims=True)
    # numpy SVD with batch dimension on the leading axis.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)   # Vt: (B, 3, 3)
    rotated = np.matmul(centered, np.swapaxes(Vt, -1, -2))     # (B, N, 3)
    max_abs = np.maximum(np.max(np.abs(rotated), axis=1, keepdims=True), min_scale)
    return jnp.asarray(rotated / max_abs, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

def polish_batch(verts_batch, faces, steps=3000, lr=1e-3,
                 tau_ix=5e-4, tau_dih_start=0.3, tau_dih_end=0.03,
                 tau_coll_start=0.3, tau_coll_end=0.03,
                 w_dih=1.0, w_coll=1.0,
                 chunk_size=200, log_every=500, use_scan=True):
    """Phase-2 optimization: keep the intersection penalty active (at fixed
    sharp tau_ix), maximize the smallest dihedral angle (smooth-max over
    cos(dihedral)), and keep every pair of edges that share a vertex away
    from collinearity (smooth-max over cos^2(edge-pair angle)). Same batched
    Adam-with-PCA-normalize loop as `optimize_batch`."""
    face_idx, edge_idx = build_pair_index(faces)
    dih_idx = build_dihedral_index(faces)
    ep_idx = build_edge_pair_index(N)

    polish_loss = make_polish_loss(face_idx, edge_idx, dih_idx, ep_idx)
    grad_single = jax.grad(polish_loss)
    loss_batch_jit = jax.jit(jax.vmap(
        polish_loss, in_axes=(0, None, None, None, None, None)))

    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    def one_step(V, m, s, step_num, tau_dih_val, tau_coll_val):
        g = jax.vmap(grad_single, in_axes=(0, None, None, None, None, None))(
            V, tau_ix, tau_dih_val, tau_coll_val, w_dih, w_coll)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        bc1 = 1.0 - beta1 ** step_num
        bc2 = 1.0 - beta2 ** step_num
        V = V - lr * (m / bc1) / (jnp.sqrt(s / bc2) + adam_eps)
        return V, m, s

    one_step_jit = jax.jit(one_step)

    @jax.jit
    def run_chunk_scan(V, m, s, step_start_f, tau_dih_chunk, tau_coll_chunk):
        def body(carry, inputs):
            V_, m_, s_ = carry
            step_num, tau_dih_val, tau_coll_val = inputs
            return one_step(V_, m_, s_, step_num, tau_dih_val, tau_coll_val), None
        steps_in_chunk = tau_dih_chunk.shape[0]
        step_nums = step_start_f + jnp.arange(1, steps_in_chunk + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s),
                                (step_nums, tau_dih_chunk, tau_coll_chunk))
        return final

    def run_chunk_loop(V, m, s, step_start, tau_dih_chunk, tau_coll_chunk):
        for i in range(tau_dih_chunk.shape[0]):
            V, m, s = one_step_jit(V, m, s,
                                   jnp.float32(step_start + i + 1),
                                   tau_dih_chunk[i], tau_coll_chunk[i])
        return V, m, s

    run_chunk = run_chunk_scan if use_scan else run_chunk_loop

    B = verts_batch.shape[0]
    V = pca_normalize_batch(verts_batch)
    m = jnp.zeros_like(V)
    s = jnp.zeros_like(V)

    tau_dih_sched = (tau_dih_start *
                     (tau_dih_end / tau_dih_start) ** np.linspace(0.0, 1.0, steps)).astype(np.float32)
    tau_coll_sched = (tau_coll_start *
                      (tau_coll_end / tau_coll_start) ** np.linspace(0.0, 1.0, steps)).astype(np.float32)

    # Track the best (clean) instance by largest min-dihedral-angle-in-degrees.
    best_verts = np.asarray(V)
    best_min_deg = np.full(B, -1.0, dtype=np.float32)

    step_done = 0
    # Track per-instance: "score" = min(min_dihedral_deg, min_edge_pair_angle_deg)
    # since we want BOTH quantities large. We only credit clean instances.
    best_score_deg = np.full(B, -1.0, dtype=np.float32)
    while step_done < steps:
        k = min(chunk_size, steps - step_done)
        tau_dih_chunk = jnp.asarray(tau_dih_sched[step_done: step_done + k], dtype=jnp.float32)
        tau_coll_chunk = jnp.asarray(tau_coll_sched[step_done: step_done + k], dtype=jnp.float32)
        if use_scan:
            step_start_f = jnp.asarray(float(step_done), dtype=jnp.float32)
            V, m, s = run_chunk(V, m, s, step_start_f, tau_dih_chunk, tau_coll_chunk)
        else:
            V, m, s = run_chunk(V, m, s, step_done, tau_dih_chunk, tau_coll_chunk)
        V = pca_normalize_batch(V)
        step_done += k

        V_np = np.asarray(V)
        n_ix_all = np.empty(B, dtype=np.int32)
        min_dih_deg = np.empty(B, dtype=np.float32)
        min_ep_deg = np.empty(B, dtype=np.float32)
        for i in range(B):
            n_ix_all[i], _ = count_real_intersections(V_np[i], faces)
            cos_d = np.clip(np.asarray(dihedral_cos(V_np[i], dih_idx)), -1.0, 1.0)
            min_dih_deg[i] = np.degrees(np.arccos(cos_d.max()))
            cos2_ep = np.clip(np.asarray(edge_pair_cos2(V_np[i], ep_idx)), 0.0, 1.0)
            # Worst-case is whichever edge-pair has the largest cos^2; the
            # "deviation from collinearity" angle is arccos(sqrt(cos2)) if cos>0
            # or pi - that if cos<0. Either way the smallest deviation is
            # arcsin(sqrt(1 - max(cos2))).
            min_ep_deg[i] = np.degrees(np.arcsin(np.sqrt(max(0.0, 1.0 - cos2_ep.max()))))

        score = np.minimum(min_dih_deg, min_ep_deg)
        clean = (n_ix_all == 0)
        improved = clean & (score > best_score_deg)
        best_score_deg = np.where(improved, score, best_score_deg)
        best_verts = np.where(improved[:, None, None], V_np, best_verts)

        if step_done % log_every < chunk_size or step_done == steps:
            if clean.any():
                cd_min = float(min_dih_deg[clean].min())
                cd_max = float(min_dih_deg[clean].max())
                ce_min = float(min_ep_deg[clean].min())
                ce_max = float(min_ep_deg[clean].max())
                cs_max = float(score[clean].max())
            else:
                cd_min = cd_max = ce_min = ce_max = cs_max = float("nan")
            losses = np.asarray(loss_batch_jit(
                V, tau_ix, tau_dih_chunk[-1], tau_coll_chunk[-1], w_dih, w_coll))
            print(f"  step {step_done:5d}  "
                  f"tau_dih={float(tau_dih_chunk[-1]):.3f}  "
                  f"tau_coll={float(tau_coll_chunk[-1]):.3f}  "
                  f"loss={float(losses.mean()):.3g}  "
                  f"clean={int(clean.sum())}/{B}  "
                  f"min_dih[{cd_min:5.1f}..{cd_max:5.1f}]  "
                  f"min_edgepair[{ce_min:5.1f}..{ce_max:5.1f}]  "
                  f"best_score={cs_max:5.1f}")

    return best_verts, best_score_deg


def count_real_intersections(verts, faces):
    cnt = 0
    offenders = []
    for face in faces:
        A, B, C = verts[list(face)]
        for e in EDGES:
            if set(e) & set(face):
                continue
            P, Q = verts[e[0]], verts[e[1]]
            if segment_crosses_triangle_interior(P, Q, A, B, C):
                cnt += 1
                offenders.append((face, e))
    return cnt, offenders


def optimize(verts, faces, steps=4000, lr=5e-3,
             eps_start=0.5, eps_end=1e-3, log_every=100):
    face_idx, edge_idx = build_pair_index(faces)
    loss_fn = jax.jit(make_loss(face_idx, edge_idx))
    grad_fn = jax.jit(jax.grad(make_loss(face_idx, edge_idx)))

    # Adam (by hand — no optax dependency).
    v = jnp.asarray(verts, dtype=jnp.float32)
    v = pca_normalize(v)
    m = jnp.zeros_like(v)
    s = jnp.zeros_like(v)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    history = []
    for step in range(1, steps + 1):
        frac = (step - 1) / max(1, steps - 1)
        eps = eps_start * (eps_end / eps_start) ** frac

        g = grad_fn(v, eps)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** step)
        s_hat = s / (1 - beta2 ** step)
        v = v - lr * m_hat / (jnp.sqrt(s_hat) + adam_eps)

        v = pca_normalize(v)

        if step % log_every == 0 or step == 1 or step == steps:
            L = float(loss_fn(v, eps))
            n_ix, _ = count_real_intersections(np.array(v), faces)
            history.append((step, eps, L, n_ix))
            print(f"  step {step:5d}  eps={eps:.5f}  soft_loss={L:8.4f}  real_ix={n_ix}")
            if n_ix == 0 and L < 1e-3:
                # Don't bother finishing if we're clearly already clean.
                print("  clean polyhedron, stopping early")
                break

    return np.array(v), history


# ---------------------------------------------------------------------------
# Batched / parallel optimization
# ---------------------------------------------------------------------------

def optimize_batch(verts_batch, faces, steps=5000, lr=5e-3,
                   tau_start=0.5, tau_end=1e-3,
                   chunk_size=250, log_every=500, use_scan=True):
    """Run `B = verts_batch.shape[0]` Adam trajectories in parallel, sharing a
    single combinatorial structure. When `use_scan` is true, each chunk is one
    jitted `lax.scan` kernel (fastest on CPU/CUDA). On jax-metal `scan` hangs,
    so `use_scan=False` falls back to a Python loop over a jitted per-step
    kernel. Between chunks we hop to the host to PCA-normalize (jax-metal has
    no eigh/svd) and to count real intersections for logging."""
    face_idx, edge_idx = build_pair_index(faces)
    loss_single = make_loss(face_idx, edge_idx)
    grad_single = jax.grad(loss_single)

    loss_batch_jit = jax.jit(jax.vmap(loss_single, in_axes=(0, None)))

    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    def one_step(V, m, s, step_num, tau_val):
        g = jax.vmap(grad_single, in_axes=(0, None))(V, tau_val)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        bc1 = 1.0 - beta1 ** step_num
        bc2 = 1.0 - beta2 ** step_num
        V = V - lr * (m / bc1) / (jnp.sqrt(s / bc2) + adam_eps)
        return V, m, s

    one_step_jit = jax.jit(one_step)

    @jax.jit
    def run_chunk_scan(V, m, s, step_start_f, tau_chunk):
        def body(carry, inputs):
            V_, m_, s_ = carry
            step_num, tau_val = inputs
            return one_step(V_, m_, s_, step_num, tau_val), None
        steps_in_chunk = tau_chunk.shape[0]
        step_nums = step_start_f + jnp.arange(1, steps_in_chunk + 1, dtype=jnp.float32)
        final, _ = jax.lax.scan(body, (V, m, s), (step_nums, tau_chunk))
        return final

    def run_chunk_loop(V, m, s, step_start, tau_chunk):
        for i in range(tau_chunk.shape[0]):
            V, m, s = one_step_jit(V, m, s,
                                   jnp.float32(step_start + i + 1),
                                   tau_chunk[i])
        return V, m, s

    run_chunk = run_chunk_scan if use_scan else run_chunk_loop

    B = verts_batch.shape[0]
    V = pca_normalize_batch(verts_batch)
    m = jnp.zeros_like(V)
    s = jnp.zeros_like(V)

    # Geometric anneal over `steps`.
    tau_sched = tau_start * (tau_end / tau_start) ** np.linspace(0.0, 1.0, steps).astype(np.float32)

    best_verts = np.asarray(V)
    best_ix = np.full(B, 10**9, dtype=np.int32)   # smallest-yet real intersections per instance
    global_best = (10**9, None)                   # (n_ix, verts)
    any_zero = False

    step_done = 0
    log_buf = []
    while step_done < steps:
        k = min(chunk_size, steps - step_done)
        tau_chunk = jnp.asarray(tau_sched[step_done: step_done + k], dtype=jnp.float32)
        if use_scan:
            step_start_f = jnp.asarray(float(step_done), dtype=jnp.float32)
            V, m, s = run_chunk(V, m, s, step_start_f, tau_chunk)
        else:
            V, m, s = run_chunk(V, m, s, step_done, tau_chunk)
        V = pca_normalize_batch(V)
        step_done += k

        V_np = np.asarray(V)
        n_ix_all = np.empty(B, dtype=np.int32)
        for i in range(B):
            n_ix_all[i], _ = count_real_intersections(V_np[i], faces)

        improved = n_ix_all < best_ix
        best_ix = np.where(improved, n_ix_all, best_ix)
        best_verts = np.where(improved[:, None, None], V_np, best_verts)
        cur_min = int(n_ix_all.min())
        if cur_min < global_best[0]:
            i_star = int(n_ix_all.argmin())
            global_best = (cur_min, V_np[i_star].copy())

        losses = np.asarray(loss_batch_jit(V, tau_chunk[-1]))
        zero_count = int((n_ix_all == 0).sum())
        mean_ix = float(n_ix_all.mean())
        log_buf.append((step_done, float(tau_chunk[-1]), float(losses.min()),
                        float(losses.mean()), cur_min, mean_ix, zero_count))

        if step_done % log_every < chunk_size or step_done == steps:
            print(f"  step {step_done:5d}  tau={float(tau_chunk[-1]):.5f}  "
                  f"min_loss={float(losses.min()):.4g}  mean_loss={float(losses.mean()):.4g}  "
                  f"min_ix={cur_min}  mean_ix={mean_ix:.2f}  "
                  f"zero_instances={zero_count}/{B}")

        if zero_count > 0 and not any_zero:
            any_zero = True
            # Could stop early, but let the anneal finish so we get good margins.

    return best_verts, best_ix, global_best, log_buf


# ---------------------------------------------------------------------------
# IO / reporting
# ---------------------------------------------------------------------------

def write_obj(path, verts, faces):
    with open(path, "w") as fh:
        fh.write("# Csaszar polyhedron\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--structure-tries", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--tau-start", type=float, default=0.5)
    ap.add_argument("--tau-end", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--no-scan", action="store_true",
                    help="use a Python step loop instead of lax.scan (needed on jax-metal)")
    ap.add_argument("--out", type=str, default="csaszar")
    # Polish-phase knobs
    ap.add_argument("--no-polish", action="store_true",
                    help="skip the second phase that maximizes the min dihedral angle")
    ap.add_argument("--polish-steps", type=int, default=3000)
    ap.add_argument("--polish-lr", type=float, default=1e-3)
    ap.add_argument("--polish-tau-ix", type=float, default=5e-4,
                    help="sharpness of the intersection penalty during polish")
    ap.add_argument("--polish-tau-dih-start", type=float, default=0.3)
    ap.add_argument("--polish-tau-dih-end", type=float, default=0.03)
    ap.add_argument("--polish-tau-coll-start", type=float, default=0.3)
    ap.add_argument("--polish-tau-coll-end", type=float, default=0.03)
    ap.add_argument("--polish-weight-dihedral", type=float, default=1.0,
                    help="weight on the dihedral smooth-max term")
    ap.add_argument("--polish-weight-collinear", type=float, default=1.0,
                    help="weight on the edge-pair collinearity smooth-max term")
    args = ap.parse_args()

    print(f"JAX backend: {jax.devices()}")

    print("\n== finding combinatorial structure ==")
    v_anchor, faces = find_combinatorial_structure(args.seed, args.structure_tries)
    if faces is None:
        print("failed to find a manifold triangulation")
        return
    print(f"  found: {len(faces)} faces")
    for f in faces:
        print(f"    {f}")

    # Batched random initializations. Use standard-normal coords so the initial
    # covariance is already near I; the PCA normalize then only fixes the gauge.
    rng = np.random.default_rng(args.seed + 1_000_003)
    V0 = rng.standard_normal(size=(args.batch, N, 3)).astype(np.float32)

    # Initial intersection stats
    init_counts = np.array([count_real_intersections(V0[i], faces)[0] for i in range(args.batch)])
    print(f"\n== launching {args.batch} parallel optimizations ==")
    print(f"  initial intersection distribution: min={init_counts.min()}, "
          f"mean={init_counts.mean():.1f}, max={init_counts.max()}")

    best_verts_per_instance, best_ix_per_instance, global_best, log_buf = optimize_batch(
        V0, faces,
        steps=args.steps, lr=args.lr,
        tau_start=args.tau_start, tau_end=args.tau_end,
        chunk_size=args.chunk_size, log_every=args.log_every,
        use_scan=not args.no_scan,
    )

    # Phase 1 results
    n_success = int((best_ix_per_instance == 0).sum())
    i_star = int(best_ix_per_instance.argmin())
    best_n_ix = int(best_ix_per_instance[i_star])
    v_best = best_verts_per_instance[i_star]
    v_best = np.asarray(pca_normalize(jnp.asarray(v_best)))
    print("\n== phase 1: intersection removal ==")
    print(f"  instances that reached 0 intersections: {n_success} / {args.batch}")
    print(f"  best instance has: {best_n_ix} real intersections")

    polished = None
    if n_success > 0 and not args.no_polish:
        # Feed every clean instance into phase 2 in parallel.
        mask = (best_ix_per_instance == 0)
        V_clean = best_verts_per_instance[mask]
        dih_idx_cpu = build_dihedral_index(faces)
        init_min_deg = []
        for v in V_clean:
            cos_arr = np.clip(np.asarray(dihedral_cos(v, dih_idx_cpu)), -1.0, 1.0)
            init_min_deg.append(np.degrees(np.arccos(cos_arr.max())))
        init_min_deg = np.asarray(init_min_deg)
        print(f"\n== phase 2: maximize min dihedral (over {len(V_clean)} clean instances) ==")
        print(f"  initial min dihedral (deg): min={init_min_deg.min():.2f} "
              f"mean={init_min_deg.mean():.2f} max={init_min_deg.max():.2f}")

        polished_batch, polished_score_deg = polish_batch(
            V_clean, faces,
            steps=args.polish_steps, lr=args.polish_lr,
            tau_ix=args.polish_tau_ix,
            tau_dih_start=args.polish_tau_dih_start,
            tau_dih_end=args.polish_tau_dih_end,
            tau_coll_start=args.polish_tau_coll_start,
            tau_coll_end=args.polish_tau_coll_end,
            w_dih=args.polish_weight_dihedral,
            w_coll=args.polish_weight_collinear,
            chunk_size=args.chunk_size, log_every=args.log_every,
            use_scan=not args.no_scan,
        )
        # Pick the polished instance with the best (largest) min(min-dihedral,
        # min-edge-pair-angle-away-from-collinear).
        valid = (polished_score_deg > 0)
        if valid.any():
            j_star = int(np.where(valid, polished_score_deg, -1).argmax())
            v_polished = polished_batch[j_star]
            v_polished = np.asarray(pca_normalize(jnp.asarray(v_polished)))
            polished_n_ix, polished_offenders = count_real_intersections(v_polished, faces)
            cos_arr = np.clip(np.asarray(dihedral_cos(v_polished, dih_idx_cpu)), -1.0, 1.0)
            angs_deg = np.degrees(np.arccos(cos_arr))
            ep_idx_cpu = build_edge_pair_index(N)
            cos2_ep = np.clip(np.asarray(edge_pair_cos2(v_polished, ep_idx_cpu)), 0.0, 1.0)
            # Angle of each edge pair away from collinearity.
            ep_dev_deg = np.degrees(np.arcsin(np.sqrt(np.clip(1.0 - cos2_ep, 0.0, 1.0))))
            print(f"\n  best polished instance:")
            print(f"    real intersections: {polished_n_ix}")
            print(f"    dihedral:         min={angs_deg.min():.2f}  mean={angs_deg.mean():.2f}  max={angs_deg.max():.2f}  deg")
            print(f"    edge-pair dev:    min={ep_dev_deg.min():.2f}  mean={ep_dev_deg.mean():.2f}  max={ep_dev_deg.max():.2f}  deg")
            if polished_n_ix == 0:
                polished = (v_polished, polished_n_ix, angs_deg, ep_dev_deg, polished_offenders)

    # Pick which result to save: polished if available and clean, else phase-1.
    if polished is not None:
        v_best, n_ix_final, angs_deg, ep_dev_deg, offenders = polished
    else:
        n_ix_final, offenders = count_real_intersections(v_best, faces)
        dih_idx_cpu = build_dihedral_index(faces)
        ep_idx_cpu = build_edge_pair_index(N)
        cos_arr = np.clip(np.asarray(dihedral_cos(v_best, dih_idx_cpu)), -1.0, 1.0)
        angs_deg = np.degrees(np.arccos(cos_arr))
        cos2_ep = np.clip(np.asarray(edge_pair_cos2(v_best, ep_idx_cpu)), 0.0, 1.0)
        ep_dev_deg = np.degrees(np.arcsin(np.sqrt(np.clip(1.0 - cos2_ep, 0.0, 1.0))))

    print("\n== saved result ==")
    print(f"  real intersections: {n_ix_final}")
    print(f"  dihedral:       min={angs_deg.min():.2f}  mean={angs_deg.mean():.2f}  max={angs_deg.max():.2f}  deg")
    print(f"  edge-pair dev:  min={ep_dev_deg.min():.2f}  mean={ep_dev_deg.mean():.2f}  max={ep_dev_deg.max():.2f}  deg")

    np.save(f"{args.out}_vertices.npy", v_best)
    np.save(f"{args.out}_faces.npy", np.array(faces, dtype=np.int32))
    write_obj(f"{args.out}.obj", v_best, faces)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({
            "vertices": v_best.tolist(),
            "faces": [list(f) for f in faces],
            "real_intersections": n_ix_final,
            "min_dihedral_deg": float(angs_deg.min()),
            "mean_dihedral_deg": float(angs_deg.mean()),
            "dihedral_degrees": angs_deg.tolist(),
            "min_edge_pair_deviation_deg": float(ep_dev_deg.min()),
            "mean_edge_pair_deviation_deg": float(ep_dev_deg.mean()),
            "edge_pair_deviation_degrees": ep_dev_deg.tolist(),
            "n_zero_instances_phase1": n_success,
            "batch_size": args.batch,
            "offenders": [[list(a), list(b)] for a, b in offenders],
        }, fh, indent=2)
    print(f"\nwrote {args.out}_vertices.npy, {args.out}_faces.npy, {args.out}.obj, {args.out}.json")


if __name__ == "__main__":
    main()
