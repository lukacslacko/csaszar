"""Search Z_6-symmetric rotation systems on K_12. The rotation at vertex v is
determined by its parity:
    rotation[v] = (rot_even + v)  if v is even
    rotation[v] = (rot_odd  + v)  if v is odd
so we have TWO independent base permutations (rot_even, rot_odd), each a
cyclic order of {1, 2, ..., 11}. The pair together is invariant under shift
by 2, giving the Z_6 subgroup of Z_12.

Pure random over (11!)^2 is hopeless, but linked ansatze (rot_odd = f(rot_even)
for some simple f) cut the search to 11! and give a reasonable hit chance.
"""

import argparse
import itertools
import time
import numpy as np


def trace_faces_z6(rot_even, rot_odd, N=12):
    """Z_6-symmetric rotation system defined by rot_even, rot_odd."""
    pos_even = {d: i for i, d in enumerate(rot_even)}
    pos_odd = {d: i for i, d in enumerate(rot_odd)}
    L = len(rot_even)
    cap = N * (N - 1)

    def succ(vertex, x):
        if vertex % 2 == 0:
            return rot_even[(pos_even[x] + 1) % L]
        return rot_odd[(pos_odd[x] + 1) % L]

    visited = set()
    faces = []
    for a0 in range(N):
        rot_a = rot_even if a0 % 2 == 0 else rot_odd
        for d in rot_a:
            b0 = (a0 + d) % N
            if (a0, b0) in visited:
                continue
            face = []
            a, b = a0, b0
            for _ in range(cap + 1):
                if (a, b) in visited:
                    break
                visited.add((a, b))
                face.append(a)
                diff = (a - b) % N
                next_d = succ(b, diff)
                a, b = b, (b + next_d) % N
            faces.append(face)
    return faces


def all_triangular(rot_even, rot_odd, N=12):
    faces = trace_faces_z6(rot_even, rot_odd, N)
    return all(len(f) == 3 for f in faces), faces


def canonical_tris(faces):
    seen = set()
    out = []
    for f in faces:
        t = tuple(sorted(f))
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def link_is_cycle(vertex, faces_containing):
    link_edges, link_verts = [], set()
    for face in faces_containing:
        opp = [x for x in face if x != vertex]
        link_edges.append(tuple(sorted(opp)))
        link_verts.update(opp)
    deg = {v: 0 for v in link_verts}
    for (a, b) in link_edges:
        deg[a] += 1; deg[b] += 1
    if any(d != 2 for d in deg.values()): return False
    adj = {v: [] for v in link_verts}
    for (a, b) in link_edges:
        adj[a].append(b); adj[b].append(a)
    start = next(iter(link_verts))
    seen = {start}; stk = [start]
    while stk:
        u = stk.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w); stk.append(w)
    return len(seen) == len(link_verts)


def verify_manifold(unique_faces, N=12):
    edges = set()
    ec = {}
    for (a, b, c) in unique_faces:
        for u, v in ((a, b), (b, c), (a, c)):
            e = tuple(sorted((u, v)))
            edges.add(e)
            ec[e] = ec.get(e, 0) + 1
    if not all(c == 2 for c in ec.values()): return False
    if len(edges) != N * (N - 1) // 2: return False
    incident = {v: [] for v in range(N)}
    for f in unique_faces:
        for v in f:
            incident[v].append(f)
    for v in range(N):
        if not link_is_cycle(v, incident[v]): return False
    return True


def search(rng, tries, ansatz, N=12, verbose=True):
    """ansatz is a name: 'indep', 'shift1', 'reverse', 'reverse_shift'."""
    base = list(range(1, N))
    report_every = max(1, tries // 20)
    t0 = time.time()
    best = 0
    for t in range(tries):
        rot_even = list(base)
        rng.shuffle(rot_even)
        if ansatz == "shift1":
            rot_odd = [((x + 1 - 1) % (N - 1)) + 1 for x in rot_even]  # identity
        elif ansatz == "reverse":
            rot_odd = list(reversed(rot_even))
        elif ansatz == "negate":
            rot_odd = [(N - x) % N if (N - x) % N != 0 else N - 1 for x in rot_even]
            # Make sure it's a permutation of 1..N-1
            if len(set(rot_odd)) != N - 1: continue
        elif ansatz == "shift":
            shift = rng.integers(0, N - 1)
            rot_odd = [((x - 1 + shift) % (N - 1)) + 1 for x in rot_even]
        elif ansatz == "indep":
            rot_odd = list(base)
            rng.shuffle(rot_odd)
        else:
            raise ValueError(f"unknown ansatz {ansatz}")
        faces = trace_faces_z6(rot_even, rot_odd, N)
        n_tri = sum(1 for f in faces if len(f) == 3)
        if n_tri > best:
            best = n_tri
        if all(len(f) == 3 for f in faces):
            unique = canonical_tris(faces)
            if verify_manifold(unique, N):
                if verbose:
                    print(f"  HIT at try {t + 1}: rot_even={rot_even}, rot_odd={rot_odd}")
                    print(f"  {len(unique)} faces, verified manifold")
                return rot_even, rot_odd, unique
            else:
                if verbose:
                    print(f"  try {t + 1}: all-triangular but not manifold — keep searching")
        if verbose and (t + 1) % report_every == 0:
            print(f"  ...{t + 1}/{tries}  best_triangle_count={best}  "
                  f"elapsed={time.time() - t0:.1f}s  ({ansatz})")
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tries', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ansatz', default='indep',
                    choices=['indep', 'reverse', 'shift', 'shift1'])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Z_6-symmetric K_12 search: ansatz={args.ansatz}, tries={args.tries}")
    rot_even, rot_odd, faces = search(rng, args.tries, args.ansatz)
    if faces is None:
        print("No hit.")
        return
    import json
    out = "k12_manifold.json"
    json.dump({"n": 12, "rot_even": rot_even, "rot_odd": rot_odd,
               "faces": faces}, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
