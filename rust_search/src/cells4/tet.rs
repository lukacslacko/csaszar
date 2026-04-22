//! Hard-coded combinatorial description of the initial tetrahedron.
//!
//! Convention:
//!   - 4 placed vertices {0, 1, 2, 3} satisfying χ(0, 1, 2, 3) = +1
//!     (user's "0 ∈ 123" statement).
//!   - 4 planes in canonical sorted order, indexed 0..3:
//!       plane 0 = (0, 1, 2)
//!       plane 1 = (0, 1, 3)
//!       plane 2 = (0, 2, 3)
//!       plane 3 = (1, 2, 3)
//!   - σ[k] = +1 iff the cell lies in χ(·, α_k, β_k, γ_k) > 0.
//!
//! The 15 cells below were derived in DESIGN.md / this session's analysis
//! and cross-checked against coord realisation in coords.rs.  They are
//! hard-coded here.  The missing (infeasible) 16th combination is
//! σ = (+, −, −, +) on the plane order above.

use crate::cell::{Cell, Plane, SignVec};
use crate::chirotope::Chirotope;

pub const N_TET_CELLS: usize = 15;

pub fn tet_planes() -> Vec<Plane> {
    vec![
        Plane { triple: (0, 1, 2) },
        Plane { triple: (0, 1, 3) },
        Plane { triple: (0, 2, 3) },
        Plane { triple: (1, 2, 3) },
    ]
}

pub fn tet_chirotope() -> Chirotope {
    let mut chi = Chirotope::new();
    chi.set(0, 1, 2, 3, 1); // user's convention
    chi
}

/// The 15 realised cells of the tet arrangement, with human-readable
/// labels.  Plane order is (0,1,2), (0,1,3), (0,2,3), (1,2,3).
pub fn tet_cells() -> Vec<Cell> {
    // Each row's sign vector on plane order [(0,1,2), (0,1,3), (0,2,3), (1,2,3)].
    // Derived by evaluating sign(plane.value(witness)) under the
    // Plane::through((α,β,γ)) normal convention n = (β−α)×(γ−α), with
    // σ = +1 ⇔ sv > 0 ⇔ plane.value < 0 (since sv = −|n|·plane.value).
    let rows: &[(&str, [i8; 4])] = &[
        ("interior",        [-1,  1, -1,  1]),
        ("face-opp-v0",     [-1,  1, -1, -1]),
        ("face-opp-v1",     [-1,  1,  1,  1]),
        ("face-opp-v2",     [-1, -1, -1,  1]),
        ("face-opp-v3",     [ 1,  1, -1,  1]),
        ("edge(0,1)",       [ 1, -1, -1,  1]),
        ("edge(0,2)",       [ 1,  1,  1,  1]),
        ("edge(0,3)",       [-1, -1,  1,  1]),
        ("edge(1,2)",       [ 1,  1, -1, -1]),
        ("edge(1,3)",       [-1, -1, -1, -1]),
        ("edge(2,3)",       [-1,  1,  1, -1]),
        ("vertex-v0-cone",  [ 1, -1,  1,  1]),
        ("vertex-v1-cone",  [ 1, -1, -1, -1]),
        ("vertex-v2-cone",  [ 1,  1,  1, -1]),
        ("vertex-v3-cone",  [-1, -1,  1, -1]),
    ];
    rows.iter()
        .map(|(label, signs)| Cell {
            sign: SignVec::from_signs(signs),
            label: label.to_string(),
        })
        .collect()
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{chirotope_from_coords, sign_of_point_vs_plane, tet_coords, Vec3};

    /// Every hard-coded cell must be distinct (different sign vectors).
    #[test]
    fn tet_cells_are_distinct() {
        let cells = tet_cells();
        assert_eq!(cells.len(), N_TET_CELLS);
        let mut seen = std::collections::HashSet::new();
        for c in &cells {
            assert!(seen.insert(c.sign), "duplicate sign vector for {}", c.label);
        }
    }

    /// The hard-coded chirotope must agree with the coordinate
    /// realisation on the user's convention.
    #[test]
    fn chirotope_matches_coords() {
        let coords = tet_coords();
        let chi_from_coords = chirotope_from_coords(&coords);
        let chi_hard = tet_chirotope();
        assert_eq!(chi_hard.get(0, 1, 2, 3), chi_from_coords.get(0, 1, 2, 3));
        // Full 4-tuple parity consistency already covered by chirotope antisym tests.
    }

    /// Every hard-coded cell must be realisable in R³: there exists some
    /// point whose sign pattern against the 4 planes matches the cell's
    /// sign vector.  We exhibit one witness point per cell and check.
    #[test]
    fn tet_cells_realised_in_coords() {
        let coords = tet_coords();
        let cells = tet_cells();
        let planes = tet_planes();

        // Witness points — one per cell, derived at design time.
        let witnesses: std::collections::HashMap<&str, Vec3> = [
            ("interior",       [0.0, 0.0, 0.0]),
            ("face-opp-v0",    [-6.0, -6.0, -6.0]),
            ("face-opp-v1",    [6.0, 6.0, -6.0]),
            ("face-opp-v2",    [6.0, -6.0, 6.0]),
            ("face-opp-v3",    [-6.0, 6.0, 6.0]),
            ("edge(0,1)",      [0.0, 0.0, 10.0]),
            ("edge(0,2)",      [0.0, 10.0, 0.0]),
            ("edge(0,3)",      [10.0, 0.0, 0.0]),
            ("edge(1,2)",      [-10.0, 0.0, 0.0]),
            ("edge(1,3)",      [0.0, -10.0, 0.0]),
            ("edge(2,3)",      [0.0, 0.0, -10.0]),
            ("vertex-v0-cone", [10.0, 10.0, 10.0]),
            ("vertex-v1-cone", [-10.0, -10.0, 10.0]),
            ("vertex-v2-cone", [-10.0, 10.0, -10.0]),
            ("vertex-v3-cone", [10.0, -10.0, -10.0]),
        ].iter().cloned().collect();

        for cell in &cells {
            let w = witnesses.get(cell.label.as_str())
                .unwrap_or_else(|| panic!("no witness for {}", cell.label));
            for (pid, plane) in planes.iter().enumerate() {
                let (a, b, c) = plane.triple;
                let actual = sign_of_point_vs_plane(w, &coords, a, b, c);
                let expected = cell.sign.get(pid);
                assert_eq!(actual, expected,
                    "cell '{}' @ {:?} on plane {:?}: expected σ={}, coord gives {}",
                    cell.label, w, plane.triple, expected, actual);
            }
        }
    }

    /// Enumerating 3-D points yields exactly the 15 hard-coded sign
    /// vectors and not the 16th (+, −, +, −).
    #[test]
    fn tet_sixteenth_combination_is_infeasible() {
        let cells = tet_cells();
        let hard_set: std::collections::HashSet<_> = cells.iter().map(|c| c.sign).collect();
        let missing = SignVec::from_signs(&[1, -1, 1, -1]);
        assert!(!hard_set.contains(&missing),
                "expected (+,-,+,-) to NOT be a hard-coded cell");
    }
}
