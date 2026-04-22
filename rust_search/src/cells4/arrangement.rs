//! Combinatorial Arrangement.  The central structure tying chirotope,
//! planes, cells, and (future) face lattice together.
//!
//! Phase A (this commit) only supports init + chirotope extension on
//! vertex-add; the cell-splitting step is the next milestone and is left
//! unimplemented here.  See DESIGN.md for the full design.

use crate::cell::{Cell, Plane, PlaneId};
use crate::chirotope::{Chirotope, VertId};

pub struct Arrangement {
    pub n_placed: u32,
    pub chi: Chirotope,
    pub planes: Vec<Plane>,
    /// Cells currently realised.  At N=4 this is the 15 hard-coded tet
    /// cells.  After add_vertex (once split is implemented), this grows.
    pub cells: Vec<Cell>,
    /// Index of plane by canonical sorted triple, for O(1) lookup.
    pub plane_of_triple: std::collections::HashMap<(VertId, VertId, VertId), PlaneId>,
}

impl Arrangement {
    pub fn new_tetrahedron() -> Self {
        let chi = crate::tet::tet_chirotope();
        let planes = crate::tet::tet_planes();
        let cells = crate::tet::tet_cells();
        let mut plane_of_triple = std::collections::HashMap::new();
        for (pid, p) in planes.iter().enumerate() {
            plane_of_triple.insert(p.triple, pid as PlaneId);
        }
        Arrangement { n_placed: 4, chi, planes, cells, plane_of_triple }
    }

    /// Extend the chirotope to include a new vertex placed in `cell`.
    /// Every old triple (α, β, γ) yields one new chirotope entry
    /// χ(new, α, β, γ) = σ_cell[plane(α, β, γ)].
    ///
    /// This does NOT yet add the new planes or split cells — those are
    /// the unimplemented combinatorial-split step.
    pub fn extend_chirotope(&mut self, cell: &Cell) {
        let new_id = self.n_placed;
        for (pid, plane) in self.planes.iter().enumerate() {
            let (alpha, beta, gamma) = plane.triple;
            let sign = cell.sign.get(pid);
            // σ = χ(new, α, β, γ) with α<β<γ canonical.
            self.chi.set(new_id, alpha, beta, gamma, sign);
        }
        // Note: n_placed is incremented only once the full add_vertex is
        // implemented (which also adds planes and splits cells).  For the
        // current Phase A scaffolding, we leave n_placed unchanged; tests
        // inspect the new chirotope entries directly.
    }

    pub fn n_cells(&self) -> usize { self.cells.len() }
}

/// Sign of the new plane P = plane(v_new, v_a, v_b) at placed vertex
/// v_k, reading from the extended chirotope.  Return 0 iff v_k is on
/// the plane (i.e. v_k ∈ {v_new, v_a, v_b}).
///
/// By convention σ[plane(α, β, γ)] at point p = +1 iff χ(p, α, β, γ) > 0,
/// with (α, β, γ) sorted ascending.  For the new plane we sort
/// {v_new, v_a, v_b} and query the corresponding 4-point chirotope entry.
pub fn sign_of_new_plane_at_placed(
    chi: &crate::chirotope::Chirotope,
    v_new: VertId, v_a: VertId, v_b: VertId,
    v_k: VertId,
) -> i8 {
    if v_k == v_new || v_k == v_a || v_k == v_b { return 0; }
    // σ[plane(α, β, γ)] at p = +1 ⇔ χ(p, α, β, γ) > 0 where (α, β, γ)
    // is the sorted canonical triple.  Sort {v_new, v_a, v_b} before
    // querying χ so we get the canonical σ convention.
    let mut triple = [v_new, v_a, v_b];
    triple.sort();
    chi.get(v_k, triple[0], triple[1], triple[2])
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{chirotope_from_coords, sign_of_point_vs_plane, tet_coords, Vec3};
    use crate::tet::N_TET_CELLS;

    #[test]
    fn tetrahedron_has_15_cells() {
        let arr = Arrangement::new_tetrahedron();
        assert_eq!(arr.n_cells(), N_TET_CELLS);
        assert_eq!(arr.planes.len(), 4);
    }

    /// For every placed tet vertex v_k, the combinatorial
    /// `sign_of_new_plane_at_placed` must match the sign you'd get by
    /// evaluating the geometric plane through (v_new, v_a, v_b) at v_k.
    /// This pins down the primitive that the combinatorial split will
    /// call in a loop.
    #[test]
    fn sign_at_placed_vertex_matches_geometry() {
        use crate::coords::{chirotope_from_coords, sign_of_point_vs_plane, tet_coords, Vec3};

        // Extend coords with a specific v_new, then compare combinatorial
        // sign-at-placed-vertex to the geometric sign-of-plane at each
        // placed tet vertex for every pair (a, b).
        let mut coords = tet_coords().to_vec();
        let v_new: Vec3 = [0.3, 0.2, 0.1]; // inside the tet, in interior cell
        coords.push(v_new);
        let chi = chirotope_from_coords(&coords);
        let new_id = 4u32;

        for a in 0..4u32 {
            for b in (a + 1)..4u32 {
                // Sort the new plane's triple as canonical.
                let mut triple = [new_id, a, b];
                triple.sort();
                for k in 0..4u32 {
                    let combo = super::sign_of_new_plane_at_placed(&chi, new_id, a, b, k);
                    if k == a || k == b {
                        assert_eq!(combo, 0, "v_{} is on plane({}, {}, {}); expected 0", k, new_id, a, b);
                    } else {
                        let geom = sign_of_point_vs_plane(
                            &coords[k as usize], &coords,
                            triple[0], triple[1], triple[2],
                        );
                        assert_eq!(combo, geom,
                            "sign mismatch at v_{} vs plane({},{},{}): combo={}, geom={}",
                            k, triple[0], triple[1], triple[2], combo, geom);
                    }
                }
            }
        }
    }

    #[test]
    fn extend_chirotope_matches_coords() {
        // Place a 5th vertex at a known position, e.g. inside the
        // tet-interior cell.  Find which cell it's in (via coords),
        // extend the chirotope combinatorially, and verify the new
        // χ entries match what coords would have given.
        let coords = tet_coords();
        let v5: Vec3 = [0.0, 0.0, 0.0]; // tet centroid — inside interior cell
        let planes_iter: Vec<(VertId, VertId, VertId)> =
            vec![(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];

        // Coord-truth: the sign vector of v5.
        let coord_sigma: Vec<i8> = planes_iter.iter().map(|&(a, b, c)| {
            sign_of_point_vs_plane(&v5, &coords, a, b, c)
        }).collect();

        // Find the cell in the arrangement with this sign pattern.
        let arr = Arrangement::new_tetrahedron();
        let cell = arr.cells.iter().find(|c| {
            (0..4).all(|i| c.sign.get(i) == coord_sigma[i])
        }).expect("v5 at centroid should land in an arrangement cell");
        assert_eq!(cell.label, "interior");

        // Extend.
        let mut arr2 = Arrangement::new_tetrahedron();
        arr2.extend_chirotope(cell);

        // Ground truth from coords with v5 appended.
        let mut coords5 = coords.to_vec();
        coords5.push(v5);
        let chi_truth = chirotope_from_coords(&coords5);

        // Compare χ(4, α, β, γ) for each old plane.
        let new_id = 4;
        for (a, b, c) in planes_iter {
            assert_eq!(
                arr2.chi.get(new_id, a, b, c),
                chi_truth.get(new_id, a, b, c),
                "mismatch at χ(4, {}, {}, {})", a, b, c,
            );
        }
    }
}
