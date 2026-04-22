//! Chirotope: sign function on ordered 4-tuples of placed vertex indices.
//!
//! Stored keyed by sorted 4-tuples; antisymmetric lookups compute the
//! sort parity and flip the stored sign accordingly.

use std::collections::HashMap;

pub type VertId = u32;

#[derive(Clone, Debug, Default)]
pub struct Chirotope {
    signs: HashMap<(VertId, VertId, VertId, VertId), i8>,
}

/// Sort a 4-tuple by bubble sort; return (sorted, parity).  Parity = +1
/// if the sort required an even number of swaps, -1 if odd.
pub fn sort4_with_parity(
    a: VertId, b: VertId, c: VertId, d: VertId,
) -> ((VertId, VertId, VertId, VertId), i8) {
    let mut arr = [a, b, c, d];
    let mut parity: i8 = 1;
    for i in 0..4 {
        for j in 0..(3 - i) {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                parity = -parity;
            }
        }
    }
    ((arr[0], arr[1], arr[2], arr[3]), parity)
}

impl Chirotope {
    pub fn new() -> Self { Self::default() }

    /// χ(a, b, c, d) for any ordering of distinct indices.
    pub fn get(&self, a: VertId, b: VertId, c: VertId, d: VertId) -> i8 {
        let (key, parity) = sort4_with_parity(a, b, c, d);
        let stored = self.signs.get(&key).copied().unwrap_or(0);
        parity * stored
    }

    /// Set χ at the given 4-tuple (any ordering); internally stored sorted.
    pub fn set(&mut self, a: VertId, b: VertId, c: VertId, d: VertId, sign: i8) {
        let (key, parity) = sort4_with_parity(a, b, c, d);
        self.signs.insert(key, parity * sign);
    }

    pub fn n_entries(&self) -> usize { self.signs.len() }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_parity_identity() {
        let (s, p) = sort4_with_parity(0, 1, 2, 3);
        assert_eq!(s, (0, 1, 2, 3));
        assert_eq!(p, 1);
    }

    #[test]
    fn sort_parity_single_swap() {
        let (s, p) = sort4_with_parity(1, 0, 2, 3);
        assert_eq!(s, (0, 1, 2, 3));
        assert_eq!(p, -1);
    }

    #[test]
    fn sort_parity_double_swap() {
        let (s, p) = sort4_with_parity(1, 0, 3, 2);
        assert_eq!(s, (0, 1, 2, 3));
        assert_eq!(p, 1);
    }

    #[test]
    fn sort_parity_reversal() {
        // (3,2,1,0) to (0,1,2,3): 2 transpositions (swap 0/3, swap 1/2) = even parity.
        let (s, p) = sort4_with_parity(3, 2, 1, 0);
        assert_eq!(s, (0, 1, 2, 3));
        assert_eq!(p, 1);
    }

    #[test]
    fn chirotope_antisymmetric_queries() {
        let mut chi = Chirotope::new();
        chi.set(0, 1, 2, 3, 1);
        assert_eq!(chi.get(0, 1, 2, 3), 1);
        assert_eq!(chi.get(1, 0, 2, 3), -1); // single swap
        assert_eq!(chi.get(0, 2, 1, 3), -1);
        assert_eq!(chi.get(3, 2, 1, 0), 1);  // reversal = 2 swaps
    }
}
