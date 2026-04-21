//! Combinatorial feasibility check using sign vectors only.
//!
//! Given:
//!   - a cell's sign vector σ  (bit per plane_id in the arrangement)
//!   - the arrangement's metadata (`plane_of_triple`, `vert_sign`)
//!   - the list of committed faces
//!
//! For every pair (committed face (a,b,c), non-incident placed j), the cell
//! is INFEASIBLE if all four sign conditions hold:
//!
//!   1. σ[id(a,b,c)] ≠ vert_sign[id(a,b,c)][j]         (opposite of v_j)
//!   2. σ[id(j,a,b)] == vert_sign[id(j,a,b)][c]        (same as v_c)
//!   3. σ[id(j,b,c)] == vert_sign[id(j,b,c)][a]        (same as v_a)
//!   4. σ[id(j,a,c)] == vert_sign[id(j,a,c)][b]        (same as v_b)
//!
//! The derivation of these four conditions from `seg_crosses_tri` is
//! unit-tested in `arrangement::tests::seg_crosses_tri_sign_only_reconstruction`.
//! Plane-sign correction cancels per-pair so the test uses plane.value
//! signs directly.

use crate::arrangement::{sort_triple_with_parity, Arrangement, SignVec};

/// Precomputed test data for a single (committed_face, non-incident j)
/// piercing combination.  Encoded so the runtime check is four bit reads
/// plus three equalities and one inequality.
#[derive(Clone, Copy, Debug)]
pub struct PierceTest {
    pub pid_face: usize,       // plane id for (a,b,c)
    pub pid_jab: usize,
    pub pid_jbc: usize,
    pub pid_jac: usize,
    pub s_vj_face: i8,         // sign of v_j against plane (a,b,c)
    pub s_vc_jab: i8,          // sign of v_c against plane (j,a,b)
    pub s_va_jbc: i8,          // sign of v_a against plane (j,b,c)
    pub s_vb_jac: i8,          // sign of v_b against plane (j,a,c)
}

/// Build the pierce-test table for all (committed_face, non-incident j)
/// combinations against the current arrangement.
///
/// Returns an error string if any required triple plane is missing from
/// the arrangement — that would indicate a programmer bug in plane
/// scheduling.
pub fn build_pierce_tests(
    arr: &Arrangement,
    committed_faces: &[[u32; 3]],
    n_placed: usize,
) -> Result<Vec<PierceTest>, String> {
    let mut tests = Vec::new();
    for face in committed_faces {
        let (a, b, c) = (face[0], face[1], face[2]);
        let (face_canon, _) = sort_triple_with_parity(a, b, c);
        let pid_face = *arr.plane_of_triple.get(&face_canon)
            .ok_or_else(|| format!("missing plane for committed face {:?}", face_canon))?
            as usize;
        for j in 0..n_placed as u32 {
            if j == a || j == b || j == c { continue; }
            let (jab, _) = sort_triple_with_parity(j, a, b);
            let (jbc, _) = sort_triple_with_parity(j, b, c);
            let (jac, _) = sort_triple_with_parity(j, a, c);
            let pid_jab = *arr.plane_of_triple.get(&jab)
                .ok_or_else(|| format!("missing plane for triple {:?}", jab))?
                as usize;
            let pid_jbc = *arr.plane_of_triple.get(&jbc)
                .ok_or_else(|| format!("missing plane for triple {:?}", jbc))?
                as usize;
            let pid_jac = *arr.plane_of_triple.get(&jac)
                .ok_or_else(|| format!("missing plane for triple {:?}", jac))?
                as usize;
            tests.push(PierceTest {
                pid_face, pid_jab, pid_jbc, pid_jac,
                s_vj_face: arr.vert_sign[pid_face][j as usize],
                s_vc_jab:  arr.vert_sign[pid_jab][c as usize],
                s_va_jbc:  arr.vert_sign[pid_jbc][a as usize],
                s_vb_jac:  arr.vert_sign[pid_jac][b as usize],
            });
        }
    }
    Ok(tests)
}

/// Return true iff the cell with sign vector `sigma` is feasible — i.e.,
/// placing a new vertex at the cell's witness doesn't cause any segment
/// from a placed vertex to pierce any committed face.
#[inline]
pub fn is_feasible(sigma: &SignVec, tests: &[PierceTest]) -> bool {
    for t in tests {
        let s_face = sigma.sign(t.pid_face);
        if s_face == t.s_vj_face { continue; }           // same side of abc as v_j → safe
        let s_jab = sigma.sign(t.pid_jab);
        if s_jab != t.s_vc_jab { continue; }             // not same-as-c wedge → safe
        let s_jbc = sigma.sign(t.pid_jbc);
        if s_jbc != t.s_va_jbc { continue; }
        let s_jac = sigma.sign(t.pid_jac);
        if s_jac != t.s_vb_jac { continue; }
        return false;                                    // bad octant: pierce
    }
    true
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrangement::{seg_crosses_tri, Arrangement, Plane, PlaneOrigin, Vec3};

    /// Build an arrangement with all C(n,3) triple planes for the given
    /// `placed_verts`, then check combinatorial vs geometric feasibility
    /// on a grid of candidate points for one committed face.
    #[test]
    fn combinatorial_matches_geometric_for_tetrahedron_face() {
        let placed: Vec<Vec3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, (3f64).sqrt() / 2.0, 0.0],
            [0.5, (3f64).sqrt() / 6.0, (6f64).sqrt() / 3.0],
        ];
        let n = placed.len();
        let mut arr = Arrangement::unit_box(5.0, n);
        // Add all triple planes.
        for a in 0..n as u32 {
            for b in (a+1)..n as u32 {
                for c in (b+1)..n as u32 {
                    let plane = Plane::through(
                        &placed[a as usize], &placed[b as usize], &placed[c as usize]);
                    arr.add_plane(plane, PlaneOrigin::Triple(a, b, c), &placed);
                }
            }
        }
        arr.recompute_vert_signs(&placed);

        // Committed face (0,1,2); non-incident j = 3.
        let committed = vec![[0u32, 1, 2]];
        let tests = build_pierce_tests(&arr, &committed, n).unwrap();
        assert_eq!(tests.len(), 1, "one non-incident vertex -> one pierce test");

        // Sample a grid of candidate points and compare combinatorial vs
        // geometric feasibility.  The combinatorial decision applies per
        // cell; the geometric one per point.  We ensure each sample
        // point lies strictly inside some cell (drop ε-close cases).
        let mut geom_agree = 0;
        let mut total = 0;
        let grid = 7;
        for ix in -grid..=grid {
            for iy in -grid..=grid {
                for iz in -grid..=grid {
                    let p: Vec3 = [
                        (ix as f64) * 0.6,
                        (iy as f64) * 0.6,
                        (iz as f64) * 0.6,
                    ];
                    // Geometric feasibility: for j=3 and face (0,1,2).
                    let geom_infeasible = seg_crosses_tri(
                        &p, &placed[3], &placed[0], &placed[1], &placed[2], 1e-9);
                    let geom_feasible = !geom_infeasible;
                    // Find the arrangement cell containing p.
                    let mut sigma = SignVec::zero();
                    let mut on_boundary = false;
                    for (pid, plane) in arr.planes.iter().enumerate() {
                        let v = plane.value(&p);
                        if v.abs() < 1e-6 { on_boundary = true; break; }
                        sigma.set_sign(pid, v > 0.0);
                    }
                    if on_boundary { continue; }
                    let combo = is_feasible(&sigma, &tests);
                    total += 1;
                    if combo == geom_feasible { geom_agree += 1; }
                    else {
                        panic!("disagreement at p={:?}: geom={} combo={}",
                               p, geom_feasible, combo);
                    }
                }
            }
        }
        assert!(total > 1000, "not enough samples: {}", total);
        assert_eq!(geom_agree, total);
    }
}
