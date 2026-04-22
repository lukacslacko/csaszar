//! Cell = sign vector over the arrangement's planes.
//!
//! Bits are indexed by PlaneId.  Bit = 1 ↔ cell lies in the plane's
//! "canonical positive" halfspace.  We define: the canonical positive
//! halfspace of plane(α, β, γ) (with α < β < γ sorted) is the set
//! { p : χ(p, α, β, γ) > 0 }, using the chirotope convention from
//! chirotope.rs.

use crate::chirotope::VertId;

pub type PlaneId = u32;

pub const MAX_PLANES: usize = 256;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct SignVec {
    pub bits: [u64; 4],
}

impl SignVec {
    pub const fn zero() -> Self { Self { bits: [0; 4] } }

    #[inline]
    pub fn get(&self, idx: usize) -> i8 {
        if (self.bits[idx >> 6] >> (idx & 63)) & 1 != 0 { 1 } else { -1 }
    }

    #[inline]
    pub fn set(&mut self, idx: usize, positive: bool) {
        let w = idx >> 6;
        let b = idx & 63;
        if positive {
            self.bits[w] |= 1 << b;
        } else {
            self.bits[w] &= !(1 << b);
        }
    }

    /// Construct from a slice of +1 / -1 values (indexed by plane id).
    pub fn from_signs(signs: &[i8]) -> Self {
        let mut s = Self::zero();
        for (i, &sign) in signs.iter().enumerate() {
            s.set(i, sign > 0);
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct Plane {
    /// Canonical sorted triple (α < β < γ) of placed-vertex indices.
    pub triple: (VertId, VertId, VertId),
}

impl Plane {
    pub fn new(a: VertId, b: VertId, c: VertId) -> Self {
        let mut t = [a, b, c];
        t.sort();
        Plane { triple: (t[0], t[1], t[2]) }
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub sign: SignVec,
    /// Friendly human name (e.g. "interior", "edge(0,1)", "v_3-cone")
    /// for the hard-coded tet; empty for derived cells.
    pub label: String,
    /// Indices of placed vertices that are corners (0-faces) on the
    /// boundary of this cell.  For the tet's hard-coded cells, these
    /// are the tet vertices lying on the cell's closure; for a derived
    /// cell, this is populated from the parent cell's corners minus
    /// those cut away by the splitting plane.  Generic
    /// 3-plane-intersection corners (not placed vertices) are tracked
    /// separately — see DESIGN.md for the plan.
    pub placed_corners: Vec<VertId>,
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signvec_roundtrip() {
        let s = SignVec::from_signs(&[1, -1, 1, -1]);
        assert_eq!(s.get(0), 1);
        assert_eq!(s.get(1), -1);
        assert_eq!(s.get(2), 1);
        assert_eq!(s.get(3), -1);
    }

    #[test]
    fn signvec_beyond_slice_reads_negative() {
        let s = SignVec::from_signs(&[1, 1, 1]);
        assert_eq!(s.get(5), -1); // untouched bit
    }

    #[test]
    fn plane_triple_canonicalised() {
        let p = Plane::new(2, 0, 1);
        assert_eq!(p.triple, (0, 1, 2));
    }
}
