//! TEST-ONLY coordinate realisation of the combinatorial arrangement.
//!
//! Used exclusively from test harnesses to verify that the combinatorial
//! data structures agree with what a 3-D coordinate arrangement says.
//! Never imported from the hot path.

use crate::chirotope::{Chirotope, VertId};

pub type Vec3 = [f64; 3];

/// A regular-tetrahedron coordinate realisation of the user's convention.
///
/// Chosen so that χ(0, 1, 2, 3) = +1, which is equivalent to the user's
/// statement "0 ∈ 123" under the convention
///   v_d ∈ abc  ⇔  χ(d, a, b, c) > 0.
pub fn tet_coords() -> [Vec3; 4] {
    [
        [ 1.0,  1.0,  1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0],
    ]
}

#[inline]
pub fn sub(a: &Vec3, b: &Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline]
pub fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline]
pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Scalar 4-point triple: sv(x, y, z, w) = (y−x) · ((z−x) × (w−x)).
pub fn sv(x: &Vec3, y: &Vec3, z: &Vec3, w: &Vec3) -> f64 {
    dot(&sub(y, x), &cross(&sub(z, x), &sub(w, x)))
}

/// Build a Chirotope from coordinates by sampling sv on every 4-tuple.
/// Used in tests to generate the "ground truth" χ for the placed vertices.
pub fn chirotope_from_coords(coords: &[Vec3]) -> Chirotope {
    let n = coords.len() as u32;
    let mut chi = Chirotope::new();
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                for d in (c + 1)..n {
                    let s = sv(&coords[a as usize], &coords[b as usize],
                                &coords[c as usize], &coords[d as usize]);
                    let sign: i8 = if s > 0.0 { 1 } else if s < 0.0 { -1 } else { 0 };
                    chi.set(a, b, c, d, sign);
                }
            }
        }
    }
    chi
}

/// σ[plane(α, β, γ)] value at 3-D point `p`, using the convention
/// σ = +1 ⇔ χ(p, α, β, γ) > 0 ⇔ sv(p, α, β, γ) > 0.
pub fn sign_of_point_vs_plane(
    p: &Vec3, coords: &[Vec3],
    alpha: VertId, beta: VertId, gamma: VertId,
) -> i8 {
    let s = sv(p, &coords[alpha as usize], &coords[beta as usize], &coords[gamma as usize]);
    if s > 0.0 { 1 } else if s < 0.0 { -1 } else { 0 }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tet_chirotope_has_chi_0123_positive() {
        let coords = tet_coords();
        let chi = chirotope_from_coords(&coords);
        assert_eq!(chi.get(0, 1, 2, 3), 1,
                   "user's convention requires χ(0,1,2,3) = +1");
    }

    #[test]
    fn tet_chirotope_has_3_in_210() {
        // 3 ∈ 210 ⇔ χ(3, 2, 1, 0) > 0; (3,2,1,0) is the reversal of (0,1,2,3)
        // which is an even permutation, so χ(3,2,1,0) = χ(0,1,2,3) = +1.
        let coords = tet_coords();
        let chi = chirotope_from_coords(&coords);
        assert_eq!(chi.get(3, 2, 1, 0), 1);
    }

    #[test]
    fn tet_centroid_is_interior_with_sign_neg_pos_neg_pos() {
        // Expected σ for plane order [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        // at the centroid (origin) is (−, +, −, +), by direct sv
        // evaluation on the tet coords.
        let coords = tet_coords();
        let origin: Vec3 = [0.0, 0.0, 0.0];
        let planes: [(VertId, VertId, VertId); 4] =
            [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];
        let expected: [i8; 4] = [-1, 1, -1, 1];
        for (i, &(a, b, c)) in planes.iter().enumerate() {
            let s = sign_of_point_vs_plane(&origin, &coords, a, b, c);
            assert_eq!(s, expected[i],
                       "plane {:?}: expected σ = {}, got {}", (a, b, c), expected[i], s);
        }
    }
}
