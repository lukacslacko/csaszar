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
