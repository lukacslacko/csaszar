# cells4 — purely combinatorial arrangement search

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
