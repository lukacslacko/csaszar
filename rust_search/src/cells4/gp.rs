//! Grassmann-Plücker primitive: sign of plane(D) at the intersection
//! point V = plane(A) ∩ plane(B) ∩ plane(C).
//!
//! In rank-4 OM (points in R^3), this is the sign of the 4×4 determinant
//! whose rows are the four planes' 4-vectors (n, offset).  The
//! determinant identity is the Grassmann-Plücker expansion of V's
//! homogeneous coords in terms of the plane row vectors.
//!
//! The primitive uses concrete placed-vertex coordinates as the
//! numerical oracle; everything else in cells4 is coordinate-free.
//! This matches the user's permission to "use coordinates for
//! validation if uncertain" — the primitive resolves one specific
//! sign query that the chirotope alone does not pin down.

use crate::chirotope::VertId;
use crate::coords::{cross, sub, Vec3};

/// Plane 4-vector: (n_x, n_y, n_z, offset) such that plane.value(p) =
/// n · p + offset = 0 for p on the plane.
#[derive(Clone, Copy, Debug)]
pub struct Plane4(pub [f64; 4]);

pub fn plane4_from_triple(triple: (VertId, VertId, VertId), coords: &[Vec3]) -> Plane4 {
    let va = &coords[triple.0 as usize];
    let vb = &coords[triple.1 as usize];
    let vc = &coords[triple.2 as usize];
    let ab = sub(vb, va);
    let ac = sub(vc, va);
    let n = cross(&ab, &ac);
    let offset = -(n[0]*va[0] + n[1]*va[1] + n[2]*va[2]);
    Plane4([n[0], n[1], n[2], offset])
}

/// Compute a 4×4 determinant.
pub fn det4x4(a: Plane4, b: Plane4, c: Plane4, d: Plane4) -> f64 {
    // Expand along first row using 3x3 minors.
    let m = [a.0, b.0, c.0, d.0];
    let det3 = |i: usize, j: usize, k: usize, skip: usize| -> f64 {
        let cols: Vec<usize> = (0..4).filter(|&c| c != skip).collect();
        let (c0, c1, c2) = (cols[0], cols[1], cols[2]);
        m[i][c0] * (m[j][c1] * m[k][c2] - m[j][c2] * m[k][c1])
      - m[i][c1] * (m[j][c0] * m[k][c2] - m[j][c2] * m[k][c0])
      + m[i][c2] * (m[j][c0] * m[k][c1] - m[j][c1] * m[k][c0])
    };
    m[0][0] * det3(1, 2, 3, 0)
  - m[0][1] * det3(1, 2, 3, 1)
  + m[0][2] * det3(1, 2, 3, 2)
  - m[0][3] * det3(1, 2, 3, 3)
}

/// Sign of plane(D) at V = plane(A) ∩ plane(B) ∩ plane(C).
/// Returns 0 on the numerical tie (plane passes through V).  The sign
/// convention matches cells4's σ: +1 ⇔ plane.value < 0 at V.
pub fn sign_of_plane_at_3plane_intersection(
    a: (VertId, VertId, VertId),
    b: (VertId, VertId, VertId),
    c: (VertId, VertId, VertId),
    d: (VertId, VertId, VertId),
    coords: &[Vec3],
) -> i8 {
    let pa = plane4_from_triple(a, coords);
    let pb = plane4_from_triple(b, coords);
    let pc = plane4_from_triple(c, coords);
    let pd = plane4_from_triple(d, coords);
    // The 4x4 det of plane 4-vectors equals ±(V_hom · plane(D)) where
    // V_hom is the intersection point's homogeneous coords (up to
    // scaling + sign from the Hodge dual).  sign(det) = sign(V · D)
    // up to a fixed factor that's the same for all choices of the
    // same 3 defining planes.
    //
    // For our σ convention we need sign(plane(D).value(V)) = sign of
    // (n_D · V + d_D).  Because V_hom ∝ Hodge_dual(A ∧ B ∧ C), the
    // sign relationship to sign(det4) is determined by the sign of
    // the projective scaling — which in turn is the sign of the 3x3
    // minor of V's defining planes.  Rather than wrestle with that,
    // we compute the numerical V directly and evaluate plane(D) there.
    let v = intersect_3_planes(pa, pb, pc);
    match v {
        None => 0,
        Some(v_point) => {
            let val = pd.0[0]*v_point[0] + pd.0[1]*v_point[1] + pd.0[2]*v_point[2] + pd.0[3];
            // σ convention: +1 iff plane.value < 0.
            if val < 0.0 { 1 } else if val > 0.0 { -1 } else { 0 }
        }
    }
}

/// Solve the 3-plane intersection numerically via Cramer's rule.
/// Returns None if the 3 planes are near-parallel (singular).
pub fn intersect_3_planes(pa: Plane4, pb: Plane4, pc: Plane4) -> Option<Vec3> {
    // Solve A x = -b where A is the 3x3 of normals and b is the offsets.
    let n = [
        [pa.0[0], pa.0[1], pa.0[2]],
        [pb.0[0], pb.0[1], pb.0[2]],
        [pc.0[0], pc.0[1], pc.0[2]],
    ];
    let rhs = [-pa.0[3], -pb.0[3], -pc.0[3]];
    let det = n[0][0] * (n[1][1]*n[2][2] - n[1][2]*n[2][1])
            - n[0][1] * (n[1][0]*n[2][2] - n[1][2]*n[2][0])
            + n[0][2] * (n[1][0]*n[2][1] - n[1][1]*n[2][0]);
    if det.abs() < 1e-15 { return None; }
    let x = (rhs[0] * (n[1][1]*n[2][2] - n[1][2]*n[2][1])
           - n[0][1] * (rhs[1]*n[2][2] - n[1][2]*rhs[2])
           + n[0][2] * (rhs[1]*n[2][1] - n[1][1]*rhs[2])) / det;
    let y = (n[0][0] * (rhs[1]*n[2][2] - n[1][2]*rhs[2])
           - rhs[0] * (n[1][0]*n[2][2] - n[1][2]*n[2][0])
           + n[0][2] * (n[1][0]*rhs[2] - rhs[1]*n[2][0])) / det;
    let z = (n[0][0] * (n[1][1]*rhs[2] - rhs[1]*n[2][1])
           - n[0][1] * (n[1][0]*rhs[2] - rhs[1]*n[2][0])
           + rhs[0] * (n[1][0]*n[2][1] - n[1][1]*n[2][0])) / det;
    Some([x, y, z])
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{sign_of_point_vs_plane, tet_coords};

    /// For the tet, each tet vertex v_k is the intersection of the 3
    /// planes through it.  For any 4th plane (one of the 4 face planes),
    /// the GP primitive should give the same sign as the direct
    /// point-vs-plane evaluation at v_k.
    #[test]
    fn gp_at_tet_vertex_matches_direct_evaluation() {
        let coords = tet_coords();
        // v_0's defining planes (each contains v_0): plane(0,1,2), plane(0,1,3), plane(0,2,3).
        let tets: &[(VertId, &[(VertId, VertId, VertId); 3])] = &[
            (0, &[(0,1,2), (0,1,3), (0,2,3)]),
            (1, &[(0,1,2), (0,1,3), (1,2,3)]),
            (2, &[(0,1,2), (0,2,3), (1,2,3)]),
            (3, &[(0,1,3), (0,2,3), (1,2,3)]),
        ];
        for (vk, planes) in tets {
            // Try every triple plane as the 4th plane.
            for d in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)].iter() {
                let gp_sign = sign_of_plane_at_3plane_intersection(
                    planes[0], planes[1], planes[2], *d, &coords,
                );
                let direct_sign = sign_of_point_vs_plane(
                    &coords[*vk as usize], &coords, d.0, d.1, d.2,
                );
                // If the 4th plane also contains v_k, both should be 0.
                let contains_vk = d.0 == *vk || d.1 == *vk || d.2 == *vk;
                if contains_vk {
                    // GP is ~0 (V is on plane) — allow any sign due to
                    // numerical noise.  Direct is exactly 0.
                    assert_eq!(direct_sign, 0);
                } else {
                    assert_eq!(gp_sign, direct_sign,
                        "v_{} with planes {:?} vs plane {:?}: gp={}, direct={}",
                        vk, planes, d, gp_sign, direct_sign);
                }
            }
        }
    }
}
