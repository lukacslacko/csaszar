//! Combinatorial piercing: does edge (i, j) pierce the interior of
//! triangle (a, b, c) when {i, j} ∩ {a, b, c} = ∅?
//!
//! Purely a function of the chirotope.  Uses the four-sign test derived
//! in cells2's feasibility module and re-expressed here in χ terms:
//!
//!   edge (i, j) pierces triangle (a, b, c) iff
//!     sign(sv(i, a, b, c)) ≠ sign(sv(j, a, b, c))       (plane-sep)
//!     AND
//!     sign(sv(p,q,r,s)) agree on the 3 wedge-tests where
//!     the relevant 4-tuples are (i, j, a, b), (i, j, b, c), (i, j, a, c)
//!     with one of {i, j} playing the role of "apex".  We use a clean
//!     form: the three signs must all be equal (all + or all −).
//!
//! χ(i, j, k, l) = sign(sv(v_i, v_j, v_k, v_l)) by convention, so the
//! whole pierce test is a finite set of chirotope queries.

use crate::chirotope::{Chirotope, VertId};

/// Returns true iff edge (i, j) pierces triangle (a, b, c).  The caller
/// guarantees {i, j} ∩ {a, b, c} = ∅ and i ≠ j, a < b < c.
pub fn edge_pierces_triangle(
    chi: &Chirotope,
    i: VertId, j: VertId,
    a: VertId, b: VertId, c: VertId,
) -> bool {
    debug_assert!(i != j);
    debug_assert!(a < b && b < c);
    debug_assert!(i != a && i != b && i != c);
    debug_assert!(j != a && j != b && j != c);

    // Plane-separation: i and j on opposite sides of plane(a, b, c).
    let s_i = chi.get(i, a, b, c);
    let s_j = chi.get(j, a, b, c);
    if s_i == 0 || s_j == 0 { return true; }  // degenerate — treat as hit
    if s_i == s_j { return false; } // same side → no cross

    // Inside-triangle: signs of sv(i, j, x, y) for (x,y) ∈ edges of abc
    // must all agree.  Equivalently χ(i, j, a, b), χ(i, j, b, c),
    // χ(i, j, c, a) all equal.  Note the third tuple uses (c, a) order.
    let ab = chi.get(i, j, a, b);
    let bc = chi.get(i, j, b, c);
    let ca = chi.get(i, j, c, a);
    if ab == 0 || bc == 0 || ca == 0 { return true; }
    (ab > 0 && bc > 0 && ca > 0) || (ab < 0 && bc < 0 && ca < 0)
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{chirotope_from_coords, sv, tet_coords, Vec3};

    fn seg_crosses_tri_coords(
        p: &Vec3, q: &Vec3, a: &Vec3, b: &Vec3, c: &Vec3, tol: f64,
    ) -> bool {
        let vp = sv(p, a, b, c);
        let vq = sv(q, a, b, c);
        let vab = sv(p, q, a, b);
        let vbc = sv(p, q, b, c);
        let vca = sv(p, q, c, a);
        let min_abs = vp.abs().min(vq.abs()).min(vab.abs())
            .min(vbc.abs()).min(vca.abs());
        if min_abs < tol { return true; }
        let plane = vp * vq < 0.0;
        let inside = (vab > 0.0 && vbc > 0.0 && vca > 0.0)
                  || (vab < 0.0 && vbc < 0.0 && vca < 0.0);
        plane && inside
    }

    /// At N = 4 there are no disjoint (edge, triangle) pairs, so piercing
    /// is vacuously empty.  Sanity check by enumerating all disjoint
    /// pairs and asserting the set is empty.
    #[test]
    fn tetrahedron_has_no_piercings() {
        let n = 4u32;
        let mut pairs = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                for a in 0..n {
                    for b in (a + 1)..n {
                        for c in (b + 1)..n {
                            let edge_set = [i, j];
                            let tri_set = [a, b, c];
                            if edge_set.iter().any(|x| tri_set.contains(x)) { continue; }
                            pairs += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(pairs, 0, "expected zero disjoint (edge, triangle) pairs at N=4");
    }

    /// For N = 5, every piercing verdict from `edge_pierces_triangle`
    /// must match the geometric `seg_crosses_tri` on coords.  Builds a
    /// chirotope from coords and exhaustively compares.
    #[test]
    fn n5_piercing_matches_geometry() {
        // Add a 5th vertex at a position clearly distinct from the tet.
        let mut coords: Vec<Vec3> = tet_coords().to_vec();
        coords.push([0.0, 0.0, 5.0]); // above the tet

        let chi = chirotope_from_coords(&coords);
        let n = coords.len() as u32;
        let mut checked = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                for a in 0..n {
                    for b in (a + 1)..n {
                        for c in (b + 1)..n {
                            let edge_set = [i, j];
                            let tri_set = [a, b, c];
                            if edge_set.iter().any(|x| tri_set.contains(x)) { continue; }
                            let combo = edge_pierces_triangle(&chi, i, j, a, b, c);
                            let geom = seg_crosses_tri_coords(
                                &coords[i as usize], &coords[j as usize],
                                &coords[a as usize], &coords[b as usize], &coords[c as usize],
                                1e-9);
                            assert_eq!(combo, geom,
                                "pierce mismatch: edge ({},{}) vs tri ({},{},{}); \
                                 combo={} geom={}",
                                i, j, a, b, c, combo, geom);
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "no disjoint pairs at N=5?");
    }
}
