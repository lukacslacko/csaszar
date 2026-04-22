//! First slice of the combinatorial split decision.
//!
//! Given a cell C' and a new plane P = plane(v_new, v_a, v_b), decide
//! whether P cuts C' using only the signs of P at the cell's *placed*
//! vertex corners.  Three outcomes:
//!
//!   NotCut(+1)  — all non-zero corner signs agree at +1; zero corners,
//!                 if any, sit on the new plane (v_k ∈ {v_new, v_a, v_b}).
//!   NotCut(-1)  — symmetrically.
//!   Cut         — some corner is + and another is -.
//!   Undecidable — corners provide only sign 0 information (wedge cells
//!                 whose corners are the plane's pivot vertices), or a
//!                 mix of one sign and zero which can't be resolved
//!                 from placed corners alone.
//!
//! Undecidable cases need the full face-lattice / Grassmann–Plücker
//! treatment described in DESIGN.md.  Phase B-pre does not attempt
//! them; phase C will.

use crate::arrangement::sign_of_new_plane_at_placed;
use crate::cell::Cell;
use crate::chirotope::{Chirotope, VertId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDecision {
    /// Cell stays intact; new-plane bit in its σ extends by this sign.
    NotCut(i8),
    /// Cell is split into +/- children.
    Cut,
    /// Placed-corner signs are insufficient to decide.
    Undecidable(UndecidableCause),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndecidableCause {
    /// All corners are on the new plane (e.g. wedge around line(v_a, v_b)).
    AllZero,
    /// Some corners are +1 and some are 0, no −1.
    PositiveAndZero,
    /// Some corners are −1 and some are 0, no +1.
    NegativeAndZero,
}

pub fn decide_split(
    chi: &Chirotope,
    cell: &Cell,
    v_new: VertId,
    v_a: VertId,
    v_b: VertId,
) -> SplitDecision {
    let mut has_pos = false;
    let mut has_neg = false;
    let mut has_zero = false;
    for &v_k in &cell.placed_corners {
        let s = sign_of_new_plane_at_placed(chi, v_new, v_a, v_b, v_k);
        if s > 0 { has_pos = true; }
        else if s < 0 { has_neg = true; }
        else { has_zero = true; }
    }
    match (has_pos, has_neg, has_zero) {
        (true, true, _)       => SplitDecision::Cut,
        (true, false, false)  => SplitDecision::NotCut(1),
        (false, true, false)  => SplitDecision::NotCut(-1),
        (true, false, true)   => SplitDecision::Undecidable(UndecidableCause::PositiveAndZero),
        (false, true, true)   => SplitDecision::Undecidable(UndecidableCause::NegativeAndZero),
        (false, false, true)  => SplitDecision::Undecidable(UndecidableCause::AllZero),
        (false, false, false) => unreachable!("cell has no placed corners — shouldn't happen for tet"),
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrangement::Arrangement;
    use crate::coords::{chirotope_from_coords, sign_of_point_vs_plane, tet_coords, Vec3};

    /// For every pair (v_a, v_b) from the tet and a v_new placed in the
    /// tet interior, tally how many of the 15 cells the decision
    /// resolves by placed corners alone, and verify that Decisions
    /// that DO resolve agree with the geometric split outcome.
    #[test]
    fn placed_corner_split_is_consistent_with_geometry() {
        let mut coords = tet_coords().to_vec();
        let v_new: Vec3 = [0.3, 0.2, 0.1]; // interior of tet
        coords.push(v_new);
        let chi = chirotope_from_coords(&coords);
        let new_id = 4u32;

        let mut arr = Arrangement::new_tetrahedron();
        // Populate extended chirotope from coords (since placing v_4 in
        // the interior cell would do this combinatorially, but we go
        // direct for the test).
        arr.chi = chi.clone();

        let mut total_cells = 0;
        let mut decided = 0;
        let mut undec = 0;

        for a in 0..4u32 {
            for b in (a + 1)..4u32 {
                // For each pair (a, b), new plane = plane(4, a, b).
                let mut triple = [new_id, a, b];
                triple.sort();
                for cell in &arr.cells {
                    total_cells += 1;
                    let decision = decide_split(&chi, cell, new_id, a, b);

                    // Cross-check any resolved decision against geometry:
                    //   find how the new plane actually splits the cell by
                    //   evaluating sign(plane.value) at a witness point
                    //   inside the cell.  For tet cells we can't easily
                    //   pick a witness here without duplicating tet.rs
                    //   data, so only check corner consistency.
                    match decision {
                        SplitDecision::NotCut(sign) => {
                            decided += 1;
                            // All non-zero corners must match sign.
                            for &v_k in &cell.placed_corners {
                                let s = sign_of_point_vs_plane(
                                    &coords[v_k as usize], &coords,
                                    triple[0], triple[1], triple[2],
                                );
                                assert!(s == 0 || s == sign,
                                    "NotCut({}) but corner v_{} has sign {}", sign, v_k, s);
                            }
                        }
                        SplitDecision::Cut => {
                            decided += 1;
                            // Must have corners of both signs geometrically.
                            let mut saw_pos = false;
                            let mut saw_neg = false;
                            for &v_k in &cell.placed_corners {
                                let s = sign_of_point_vs_plane(
                                    &coords[v_k as usize], &coords,
                                    triple[0], triple[1], triple[2],
                                );
                                if s > 0 { saw_pos = true; }
                                if s < 0 { saw_neg = true; }
                            }
                            assert!(saw_pos && saw_neg,
                                "Cut decision but geometry shows only one sign");
                        }
                        SplitDecision::Undecidable(_) => {
                            undec += 1;
                        }
                    }
                }
            }
        }

        eprintln!(
            "N=5 add_vertex at interior: {} / {} (cell, plane)-pairs decided by placed corners; {} undecidable",
            decided, total_cells, undec,
        );
        // Sanity: at least some decisions must resolve (it'd be a bug
        // if the primitive never yielded an answer).
        assert!(decided > 0, "placed-corner primitive must resolve SOME pairs");
        assert_eq!(decided + undec, total_cells);
    }
}
