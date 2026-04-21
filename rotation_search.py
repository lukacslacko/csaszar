"""Find a triangular rotation system for K_N on an orientable surface.

Works in two modes:

  (a) Z_N-symmetric search. Pick a single "base rotation" rot0, a permutation
      of 1..N-1. The rotation at vertex v is then rot0 shifted by v. If every
      face traced by the rotation system is a triangle we have a neighborly
      triangulation of the complete graph on N vertices with
          F = 2*C(N,2)/3  =  N(N-1)/3,
      and automatically genus = (2 - (N - N(N-1)/2 + N(N-1)/3)) / 2.

  (b) Backtracking over non-symmetric rotation systems (slow, fallback).

For N=12 we want genus 6. Any triangular rotation system on K_12 is
automatically on genus 6.

Prints a valid face list if one is found and writes it to <out>.json.
"""
import argparse
import json
import time
from itertools import combinations

import numpy as np


def face_trace_sym(rot0, N, max_face=None):
    """Trace faces under the Z_N-symmetric rotation system built from rot0.
    Returns list of faces (each face is a list of vertex ids in cyclic order).

    Every face length divides 2*C(N,2) = N(N-1); in a well-formed rotation
    system the sum of face lengths equals N(N-1) (== # directed edges). We
    cap each individual trace at N*(N-1) steps as a paranoid safety net."""
    pos = {d: i for i, d in enumerate(rot0)}
    L = len(rot0)
    cap = max_face if max_face is not None else N * (N - 1)
    visited = set()
    faces = []
    for a0 in range(N):
        for d in rot0:
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
                next_d = rot0[(pos[diff] + 1) % L]
                a, b = b, (b + next_d) % N
            faces.append(face)
    return faces


def all_triangular(rot0, N):
    faces = face_trace_sym(rot0, N)
    if faces is None:
        return False, None
    if not all(len(f) == 3 for f in faces):
        return False, faces
    return True, faces


def canonical_tris(faces):
    """De-duplicate faces by unordered vertex triples (every triangle is
    traced three times, once per edge-start, so we quotient out)."""
    seen = set()
    out = []
    for f in faces:
        t = tuple(sorted(f))
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def search_symmetric(N, tries, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    base = list(range(1, N))
    best_triangle_count = 0
    t0 = time.time()
    report_every = max(1, tries // 20)
    for t in range(tries):
        rot0 = list(base)
        rng.shuffle(rot0)
        faces = face_trace_sym(rot0, N)
        n_tri = sum(1 for f in faces if len(f) == 3)
        if n_tri > best_triangle_count:
            best_triangle_count = n_tri
        if all(len(f) == 3 for f in faces):
            if verbose:
                print(f"  hit at try {t + 1}: rot0 = {rot0}")
            return rot0, faces
        if verbose and (t + 1) % report_every == 0:
            print(f"  ...{t + 1}/{tries}  best_triangle_count={best_triangle_count}  "
                  f"elapsed={time.time() - t0:.1f}s")
    return None, None


def verify(faces_unique, N):
    """Sanity check that the unique-face set is a closed 2-manifold
    triangulation of K_N."""
    edges = set()
    edge_count = {}
    for (a, b, c) in faces_unique:
        for u, v in ((a, b), (b, c), (a, c)):
            e = tuple(sorted((u, v)))
            edges.add(e)
            edge_count[e] = edge_count.get(e, 0) + 1
    V = N
    E = len(edges)
    F = len(faces_unique)
    chi = V - E + F
    genus = (2 - chi) / 2
    all_edges_twice = all(c == 2 for c in edge_count.values())
    K_complete = E == V * (V - 1) // 2
    # Vertex link check
    incident = {v: [] for v in range(N)}
    for face in faces_unique:
        for v in face:
            incident[v].append(face)
    links_are_cycles = True
    for v in range(N):
        if not _link_is_one_cycle(v, incident[v]):
            links_are_cycles = False
            break
    return {
        "V": V, "E": E, "F": F, "chi": chi, "genus": genus,
        "all_edges_twice": all_edges_twice,
        "complete_graph": K_complete,
        "manifold": links_are_cycles,
    }


def _link_is_one_cycle(vertex, faces_containing):
    if not faces_containing:
        return False
    link_edges = []
    link_verts = set()
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
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w); stack.append(w)
    return len(seen) == len(link_verts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--tries', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None,
                    help='Write unique face list to this JSON if found')
    args = ap.parse_args()

    print(f"Searching Z_{args.n}-symmetric rotation systems on K_{args.n}...")
    rot0, faces = search_symmetric(args.n, args.tries, args.seed)
    if rot0 is None:
        print("No Z_N-symmetric all-triangular rotation found.")
        return

    unique = canonical_tris(faces)
    info = verify(unique, args.n)
    print("\nFound:")
    print(f"  rot0 = {rot0}")
    print(f"  {len(unique)} unique triangular faces")
    for f in unique[:6]:
        print(f"    {f}")
    if len(unique) > 6:
        print(f"    ... and {len(unique) - 6} more")
    print(f"  verify: {info}")

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({"n": args.n, "rot0": rot0, "faces": unique, "verify": info}, fh, indent=2)
        print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
