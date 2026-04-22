# cells4 — purely combinatorial arrangement search

## Results (2026-04-22)

**At N = 7 in a 10-minute enumeration**:
- 22,788 leaves completed
- 7,133 clean polyhedra found (= valid K_7 face-sets realised by our
  coord witnesses, every edge in exactly 2 faces, every vertex in
  exactly 6 faces)
- **120 distinct face-sets** after deduplication
- 889 combinatorial prunes, 4,096 LP witness failures
- Best clean-triangle count per leaf: 28 out of 35 K_7 triangles.

The 120 distinct face-sets is exactly **7! / |Aut(Csaszar)| = 5040/42
= 120**, matching the known order-42 automorphism group of the Csaszar
polyhedron.  Every polyhedron we found is a re-labelling of the
Csaszar polyhedron — the unique K_7 torus triangulation that embeds in
R^3 as a non-self-intersecting polyhedron.  The combinatorial pipeline
recovered the full labeling orbit.

First distinct polyhedron's faces (path [0, 0, 13] from the tet
interior cell):
```
0-1-2, 0-1-5, 0-2-3, 0-3-4, 0-4-6, 0-5-6, 1-2-4, 1-3-5,
1-3-6, 1-4-6, 2-3-6, 2-4-5, 2-5-6, 3-4-5
```

## Decisions (user, 2026-04-22 night)

1. **No face committing at all** for the first cut. Skip entirely; no anchor faces.
2. **No coordinates in the hot path**, ever. Hot path = chirotope extension, cell split, piercing update. Coordinates appear only in (a) a test harness that realises the arrangement in R³ and verifies combinatorial outputs, and (b) the final visualiser.
3. **Initial 4-vertex tetrahedron is hard-coded purely combinatorially**: its chirotope, its 15 cells (with sign vectors), and the observation that piercing data is empty at N = 4 (no 5 disjoint vertices → no edge and triangle can share zero vertices).
4. **Staging**: get the core combinatorial machinery right first (cells, cell-split, piercing), test against a coordinate reference, then run N = 7 and identify combinatorial polyhedra. Visualise with the coordinate harness.

## Goal at N = 7

Enumerate every reachable arrangement (every combinatorial position of v_4, …, v_6 picked one cell at a time). For each end-state, check whether the clean (unpierced) triangles admit a 2-factor covering every edge exactly twice — a polyhedron. Expect a small, enumerable search space at N = 7; face commitment buys essentially nothing at that scale, so we just don't do it.

## Pure-combinatorial scope

Hot-path primitives (no coordinates):

- **Chirotope** χ: sign on ordered 4-tuples of placed vertex indices, antisymmetric.
- **Plane** = canonical sorted triple (α, β, γ) of placed indices.
- **Cell** = sign vector σ over the current plane list; σ[k] ∈ {+1, −1} encodes which halfspace of plane k the cell lies in.
- **Arrangement** = (planes, chirotope, cells, face-lattice incidences enough to do the split).
- **Piercing table** = for each (edge, triangle) pair with disjoint vertex sets, a bit derived from χ via the four-sign pierce test reused from cells2.

Conventions pinned at implementation time:
- χ(0, 1, 2, 3) = +1 (matches the user's "0 ∈ 123" statement after verifying parity against sv).
- σ[plane(α, β, γ)] = +1 iff the cell lies in halfspace αβγ (= where α → β → γ appears CCW).

## Four operations the core must support

1. `init_tetrahedron()` → (chirotope with the 24 χ entries on {0,1,2,3}, four planes, 15 cells, empty piercing).
2. `extend_chirotope(cell_id)` → choose a cell for v_new; derive χ(new, α, β, γ) for every placed triple (α, β, γ) from σ_C.
3. `add_plane(plane, chirotope)` → for each existing cell, decide cut / +side / −side; split cut cells, extend uncut ones. **This is the hard combinatorial step.**
4. `update_piercings(new_vertex)` → recompute the piercing table for the new edges / triangles involving v_new; combinatorial 4-sign test on χ.

The test harness shadows each operation with a coordinate realisation and asserts equivalence.

## User's matrix formulation → oriented matroid extension

The user's intuition (2026-04-22 night, after Phase A commit): cells can
be described by {−1, 0, +1} matrices where each column corresponds to a
placed vertex and each row to a plane-side constraint; a new vertex adds
a new column that references the existing matrix; cell-cut feasibility
becomes a satisfiability check on integer sign matrices, no floats.

This is exactly the **oriented matroid** framework expressed matrixwise:
- Each row of the matrix is a "covector" (equivalently, signs of the
  cell on a set of bounding plane-sides).
- Extension by a new vertex ↔ adding a new element to the OM ↔ adding a
  new "localisation" (the σ of the cell that hosts v_new).
- The satisfiability question "is this extended sign pattern realised by
  some 3-D point?" is OM **realisability of the extension**.

For realizable OMs — ours, by construction — the test reduces to a
finite propagation on 4-tuple chirotope signs via the Grassmann–Plücker
3-term identity. Explicitly, for any 5 placed indices p_1, …, p_5 in
3-D, there's an affine relation with signed coefficients

    λ_i = (−1)^{i−1} · χ(p_1, …, p̂_i, …, p_5)

and the OM axiom demands the 5 λ_i's have mixed signs. This pins down
any single unknown χ entry given the other four. Our unknown when
asking "does plane P = plane(v_new, v_a, v_b) cut cell C'?" is

    χ(v_new, v_a, v_b, p)   with p some representative of C'

which lies in a 5-tuple (v_new, v_a, v_b, v_c, p) for each placed v_c,
giving one equation per v_c. Combined with the cell's σ' (known
χ(p, placed triple) values), propagation resolves the unknown for every
cell — or detects the mixed-sign case (cell is cut).

**Why this isn't a one-night implementation**: the propagation has to be
a fixed point of dozens of GP identities simultaneously, plus careful
treatment of the 0-sign (degenerate) cases where a corner sits exactly
on the new plane. It needs unit tests against the coordinate harness at
every chirotope entry and extends to maintaining a face-lattice
structure across the 6 plane additions of a single vertex-step. Rough
estimate: 7–10 focused days, with high error risk around sign
conventions.

## The combinatorial split — plan

For each cell C' with sign vector σ' and new plane P:

- Compute sign of P at each **corner** of C' (arrangement vertices on C's boundary).
  - Corner is a **placed vertex** v_k: sign(P, v_k) = χ(new, a, b, k) with the convention-fixed sign factor.
  - Corner is a **generic 3-plane intersection** V = plane(A) ∩ plane(B) ∩ plane(C): sign(P, V) = sign of the 4×4 determinant det[A; B; C; P] over plane 4-vectors, expandable combinatorially via Grassmann–Plücker 5-point identities.
- Verdict:
  - All corner signs match and are nonzero → C' is not cut; extend σ' with that sign.
  - Mixed nonzero signs → C' is cut; emit two children σ' ‖ +P and σ' ‖ −P.
  - Signs include zero (corner is on P, e.g. corner = v_a or v_b) → treat the zero as compatible with either side; if the remaining corners all match a single sign, that's the verdict; else cut.

Corners per cell are maintained via an explicit face lattice (0-faces = arrangement vertices, 1-faces = edges, 2-faces = facets, 3-faces = cells, with incidences). Adding a plane P updates the lattice incrementally:
- For each old edge (V1, V2) with sign(P, V1) ≠ sign(P, V2), a new 0-face is created at P ∩ edge, and the edge splits into two.
- For each old 2-face F, a new 1-face is created where P crosses F.
- For each old cell C that has mixed corner signs, a new 2-face is created on P inside C, and the cell splits.

Maintaining this lattice is the bulk of the work. The test harness verifies the lattice by comparing cell counts, corner sets per cell, and piercing verdicts against the coordinate realisation.

## Files (planned)

```
rust_search/src/cells4/
  main.rs            CLI (enumerate | visualise-stub)
  chirotope.rs       Chirotope store; 4-tuple sign; extension from σ_C
  cell.rs            SignVec, Cell, Corner
  lattice.rs         face lattice maintenance
  arrangement.rs     top-level Arrangement; init_tetrahedron, extend_chirotope, add_plane
  piercing.rs        4-sign pierce test; incremental update
  coords.rs          TEST-ONLY coordinate realisation
  tet.rs             hard-coded N=4 chirotope, 15 cells, empty piercing
  enumerate.rs       N=7 iterator over reachable arrangements, 2-factor extractor
```

## Roadmap (phased)

- **Phase A — scaffolding**: types, hard-coded tet, combinatorial piercing (trivial at N=4), coordinate test harness, chirotope extension on `extend_chirotope`. Verify 15 cells match geometric count.
- **Phase B — face lattice**: incidences on top of cells; verify the tet's face lattice (4 arrangement-vertex 0-faces, 6 edges/1-faces, their incidences) against coordinates.
- **Phase C — combinatorial split**: sign(P, V) for every V in the lattice, via χ for placed vertices and a Grassmann–Plücker 5-point formula for generic vertices; implement `add_plane` with incremental lattice update; verify cell counts at N=5 vs a coordinate-based cell enumerator.
- **Phase D — N=5, 6, 7 iteration**: recursively enumerate all paths; at each leaf run the 2-factor polyhedron extractor on clean triangles; report hits.
- **Phase E — visualise**: coordinate realisation of a specific combinatorial arrangement, output JSON/OBJ for the browser viewer that already exists in the repo.

Phase A is tonight's goal; B/C/D/E are subsequent milestones and will be staged with dedicated commits and their own tests.
