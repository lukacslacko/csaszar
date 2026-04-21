//! cells3: an experimental "more combinatorial" arrangement.
//!
//! Design intent (see also the discussion with the user the night of
//! 2026-04-22): cells are represented purely by their sign vector over
//! the arrangement's planes.  No explicit polytope vertices, and — the
//! goal — no witness points either; only the sign vector.  Splitting a
//! cell by a new plane becomes a pure LP feasibility test: given the
//! cell's sign constraints, is the extension "+new_plane" realisable?
//! Is "-new_plane"?  If both, the cell is cut.
//!
//! Why LP rather than a fully combinatorial (oriented-matroid)
//! extension test?  The user's intuition is correct that the NEW
//! arrangement's combinatorial structure is determined by σ_C alone
//! (the cell v_new lives in), modulo a measure-zero set.  But deriving
//! the split decision for each old cell C' purely combinatorially from
//! (σ_C, σ', chirotope-of-placed-vertices) requires Grassmann–Plücker
//! relations from oriented matroid theory: the sign of
//! sv(v_new, v_a, v_b, p) for p ∈ C' is determined by chi(v_new, v_a,
//! v_b, V) for each arrangement vertex V of C', but translating that
//! into a closed-form check is a multi-week implementation.
//!
//! A cleaner "cyclic order" combinatorial argument DOES handle the
//! subset of cells that touch line(v_a, v_b): those cells are split
//! iff v_new's angular position around the line lies in the cell's
//! wedge.  Cells not touching the line still need LP in this design.
//!
//! So cells3 uses LP feasibility (iterative Kaczmarz-style projection,
//! ~60 LOC, no dependencies) as the split oracle.  Slower than cells2's
//! zone-theorem splits but conceptually simpler, and the cell memory is
//! strictly sign-vector (no persisted witness).  Intended for small-N
//! verification of the combinatorial direction, not large-N runtime.

pub type Vec3 = [f64; 3];

// --------------------------------------------------------------------------
// Scalar primitives (copied from cells2; identical semantics)
// --------------------------------------------------------------------------

#[inline] pub fn sub(a: &Vec3, b: &Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] pub fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

#[inline]
pub fn sv(x: &Vec3, y: &Vec3, z: &Vec3, w: &Vec3) -> f64 {
    dot(&sub(y, x), &cross(&sub(z, x), &sub(w, x)))
}

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: [f64; 3],
    pub offset: f64,
}

impl Plane {
    pub fn through(a: &Vec3, b: &Vec3, c: &Vec3) -> Self {
        let ab = sub(b, a);
        let ac = sub(c, a);
        let n = cross(&ab, &ac);
        let norm = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
        let n = [n[0]/norm, n[1]/norm, n[2]/norm];
        let offset = -(n[0]*a[0] + n[1]*a[1] + n[2]*a[2]);
        Plane { normal: n, offset }
    }
    #[inline]
    pub fn value(&self, p: &Vec3) -> f64 {
        self.normal[0]*p[0] + self.normal[1]*p[1] + self.normal[2]*p[2] + self.offset
    }
}

pub fn seg_crosses_tri(p: &Vec3, q: &Vec3, a: &Vec3, b: &Vec3, c: &Vec3, tol: f64) -> bool {
    let vp = sv(p, a, b, c);
    let vq = sv(q, a, b, c);
    let vab = sv(p, q, a, b);
    let vbc = sv(p, q, b, c);
    let vca = sv(p, q, c, a);
    let min_abs = vp.abs().min(vq.abs()).min(vab.abs()).min(vbc.abs()).min(vca.abs());
    if min_abs < tol { return true; }
    let plane = vp * vq < 0.0;
    let inside = (vab > 0.0 && vbc > 0.0 && vca > 0.0)
              || (vab < 0.0 && vbc < 0.0 && vca < 0.0);
    plane && inside
}

pub fn sort_triple_with_parity(x: u32, y: u32, z: u32) -> ((u32, u32, u32), i8) {
    let mut a = [x, y, z];
    let mut parity: i8 = 1;
    if a[0] > a[1] { a.swap(0, 1); parity = -parity; }
    if a[1] > a[2] { a.swap(1, 2); parity = -parity; }
    if a[0] > a[1] { a.swap(0, 1); parity = -parity; }
    ((a[0], a[1], a[2]), parity)
}

// --------------------------------------------------------------------------
// SignVec (identical to cells2)
// --------------------------------------------------------------------------

pub const MAX_PLANES: usize = 256;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct SignVec { pub bits: [u64; 4] }

impl SignVec {
    pub const fn zero() -> Self { Self { bits: [0; 4] } }
    #[inline]
    pub fn sign(&self, idx: usize) -> i8 {
        if (self.bits[idx >> 6] >> (idx & 63)) & 1 != 0 { 1 } else { -1 }
    }
    #[inline]
    pub fn set_sign(&mut self, idx: usize, positive: bool) {
        let w = idx >> 6; let b = idx & 63;
        if positive { self.bits[w] |= 1 << b; } else { self.bits[w] &= !(1 << b); }
    }
}

// --------------------------------------------------------------------------
// Simple LP feasibility via greedy Kaczmarz projection
// --------------------------------------------------------------------------
//
// Given a set of half-space constraints a_k · x ≥ b_k + eps (strict
// interior), find a point x ∈ R^3 satisfying all of them.  If the
// constraints are infeasible, returns None (detected when iteration
// stalls with unresolved violations).
//
// This is NOT a sound-and-complete LP solver (infeasibility detection
// is heuristic), but it's sufficient for our cell-split tests where
// the constraints come from a realisable hyperplane arrangement
// restricted to a bounded box.

pub fn lp_feasible(
    a_list: &[[f64; 3]],
    b_list: &[f64],
    bounds: f64,     // |x_i| ≤ bounds; passed in as box half-side
    eps: f64,
    max_iter: usize,
) -> Option<Vec3> {
    assert_eq!(a_list.len(), b_list.len());
    let n = a_list.len();
    // Initial point: origin (inside the box).
    let mut x: Vec3 = [0.0, 0.0, 0.0];
    // Pre-add bounding-box constraints (|x_i| < bounds - eps).
    // We treat these inline rather than through a_list so callers don't
    // have to manufacture them.
    let bbox_val = |x: &Vec3| -> Option<(usize, f64, [f64; 3])> {
        // Return (axis*2 + sign_flag, violation, constraint-normal)
        // if any bbox constraint is violated; else None.
        for i in 0..3 {
            if x[i] > bounds - eps {
                let mut nrm = [0.0; 3]; nrm[i] = -1.0;
                let viol = x[i] - (bounds - eps);
                return Some((2*i,   viol, nrm));
            }
            if x[i] < -bounds + eps {
                let mut nrm = [0.0; 3]; nrm[i] = 1.0;
                let viol = (-bounds + eps) - x[i];
                return Some((2*i+1, viol, nrm));
            }
        }
        None
    };

    for _ in 0..max_iter {
        // Check bbox first — these are easy fixes.
        if let Some((_id, _viol, nrm)) = bbox_val(&x) {
            // Project x to stay within bbox by moving along nrm.
            // Compute current value and snap axis.
            for i in 0..3 {
                if x[i] > bounds - eps { x[i] = bounds - eps; }
                if x[i] < -bounds + eps { x[i] = -bounds + eps; }
            }
            let _ = nrm;
            continue;
        }
        // Find worst-violated constraint.
        let mut worst_idx: Option<usize> = None;
        let mut worst_viol: f64 = 0.0;
        for i in 0..n {
            let val = a_list[i][0]*x[0] + a_list[i][1]*x[1] + a_list[i][2]*x[2];
            let need = b_list[i] + eps;
            if val < need {
                let v = need - val;
                if v > worst_viol { worst_viol = v; worst_idx = Some(i); }
            }
        }
        match worst_idx {
            None => return Some(x),
            Some(i) => {
                let a = &a_list[i];
                let norm_sq = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
                if norm_sq < 1e-20 { return None; }
                // Move x toward the constraint boundary by worst_viol/|a|.
                let step = worst_viol / norm_sq;
                x[0] += step * a[0];
                x[1] += step * a[1];
                x[2] += step * a[2];
            }
        }
    }
    // Final feasibility re-check with looser tolerance.
    for i in 0..n {
        let val = a_list[i][0]*x[0] + a_list[i][1]*x[1] + a_list[i][2]*x[2];
        if val < b_list[i] + eps * 0.1 { return None; }
    }
    Some(x)
}

/// Convert "σ-constrained cell" into a list of LP constraints.
pub fn cell_constraints(
    sigma: &SignVec,
    planes: &[Plane],
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let mut a_list: Vec<[f64; 3]> = Vec::with_capacity(planes.len());
    let mut b_list: Vec<f64> = Vec::with_capacity(planes.len());
    for (k, p) in planes.iter().enumerate() {
        let s = sigma.sign(k) as f64;
        // s·(n·x + o) > 0  ↔  (s·n)·x > -s·o.
        a_list.push([s * p.normal[0], s * p.normal[1], s * p.normal[2]]);
        b_list.push(-s * p.offset);
    }
    (a_list, b_list)
}

// --------------------------------------------------------------------------
// Arrangement with sign-vector-only cells, LP-based splits
// --------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum PlaneOrigin {
    Box,
    Triple(u32, u32, u32),
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub sign: SignVec,
}

pub struct Arrangement {
    pub planes: Vec<Plane>,
    pub origins: Vec<PlaneOrigin>,
    pub plane_of_triple: std::collections::HashMap<(u32, u32, u32), u32>,
    pub vert_sign: Vec<Vec<i8>>,
    pub cells: Vec<Cell>,
    pub box_size: f64,
}

impl Arrangement {
    pub fn unit_box(box_size: f64, n_placed: usize) -> Self {
        let mut arr = Arrangement {
            planes: Vec::new(),
            origins: Vec::new(),
            plane_of_triple: std::collections::HashMap::new(),
            vert_sign: Vec::new(),
            cells: Vec::new(),
            box_size,
        };
        let axes: [([f64; 3], f64); 6] = [
            ([ 1.0,  0.0,  0.0], -box_size),
            ([-1.0,  0.0,  0.0], -box_size),
            ([ 0.0,  1.0,  0.0], -box_size),
            ([ 0.0, -1.0,  0.0], -box_size),
            ([ 0.0,  0.0,  1.0], -box_size),
            ([ 0.0,  0.0, -1.0], -box_size),
        ];
        for (normal, offset) in axes {
            arr.planes.push(Plane { normal, offset });
            arr.origins.push(PlaneOrigin::Box);
            arr.vert_sign.push(vec![0i8; n_placed]);
        }
        arr.cells.push(Cell { sign: SignVec::zero() });
        arr
    }

    pub fn recompute_vert_signs(&mut self, placed_verts: &[Vec3]) {
        for (pid, plane) in self.planes.iter().enumerate() {
            let mut row = Vec::with_capacity(placed_verts.len());
            for v in placed_verts {
                let x = plane.value(v);
                row.push(if x > 0.0 { 1 } else { -1 });
            }
            self.vert_sign[pid] = row;
        }
    }

    /// Add a plane, splitting cells via LP feasibility tests.
    pub fn add_plane(
        &mut self,
        new_plane: Plane,
        origin: PlaneOrigin,
        placed_verts: &[Vec3],
    ) {
        let new_pid = self.planes.len();
        let eps = 1e-6;
        let max_iter = 200;
        let box_size = self.box_size;

        let mut new_cells: Vec<Cell> = Vec::with_capacity(self.cells.len() * 2);
        let n = new_plane.normal;
        let o = new_plane.offset;

        for cell in self.cells.drain(..) {
            // Build LP constraints for this cell's base sign vector.
            let (mut a_list, mut b_list) = cell_constraints(&cell.sign, &self.planes);

            // Try extension +new_plane: need  n·x + o > 0,
            //   i.e.,  n·x > -o.
            a_list.push([n[0], n[1], n[2]]);
            b_list.push(-o);
            let feas_pos = lp_feasible(&a_list, &b_list, box_size, eps, max_iter).is_some();
            a_list.pop(); b_list.pop();

            // Try extension -new_plane: need  n·x + o < 0,
            //   i.e.,  (-n)·x > o.
            a_list.push([-n[0], -n[1], -n[2]]);
            b_list.push(o);
            let feas_neg = lp_feasible(&a_list, &b_list, box_size, eps, max_iter).is_some();
            a_list.pop(); b_list.pop();

            match (feas_pos, feas_neg) {
                (true, true) => {
                    // Cell is cut; produce two children.
                    let mut sp = cell.sign; sp.set_sign(new_pid, true);
                    let mut sn = cell.sign; sn.set_sign(new_pid, false);
                    new_cells.push(Cell { sign: sp });
                    new_cells.push(Cell { sign: sn });
                }
                (true, false) => {
                    let mut s = cell.sign; s.set_sign(new_pid, true);
                    new_cells.push(Cell { sign: s });
                }
                (false, true) => {
                    let mut s = cell.sign; s.set_sign(new_pid, false);
                    new_cells.push(Cell { sign: s });
                }
                (false, false) => {
                    // Neither side feasible means the cell itself was
                    // infeasible — LP noise or empty cell.  Drop it.
                }
            }
        }
        self.cells = new_cells;

        // Register new plane.
        self.planes.push(new_plane);
        self.origins.push(origin);
        if let PlaneOrigin::Triple(a, b, c) = origin {
            let (canonical, _) = sort_triple_with_parity(a, b, c);
            self.plane_of_triple.insert(canonical, new_pid as u32);
        }
        let mut row = Vec::with_capacity(placed_verts.len());
        for v in placed_verts {
            let x = new_plane.value(v);
            row.push(if x > 0.0 { 1 } else { -1 });
        }
        self.vert_sign.push(row);
    }

    /// Find a 3D witness point inside the given sign vector's cell.
    pub fn witness_for(&self, sigma: &SignVec) -> Option<Vec3> {
        let (a_list, b_list) = cell_constraints(sigma, &self.planes);
        lp_feasible(&a_list, &b_list, self.box_size, 1e-4, 500)
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_trivial_feasible() {
        // x > 0, y > 0, z > 0 inside box of size 10.  Feasible.
        let a = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = vec![0.0, 0.0, 0.0];
        let res = lp_feasible(&a, &b, 10.0, 1e-4, 100);
        assert!(res.is_some());
        let p = res.unwrap();
        assert!(p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0, "got {:?}", p);
    }

    #[test]
    fn lp_infeasible() {
        // x > 0 AND x < -1 simultaneously.  Infeasible.
        let a = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let b = vec![0.0, 1.0];
        let res = lp_feasible(&a, &b, 10.0, 1e-4, 100);
        assert!(res.is_none(), "infeasible system should return None, got {:?}", res);
    }

    #[test]
    fn arrangement_unit_box_one_cell() {
        let arr = Arrangement::unit_box(10.0, 0);
        assert_eq!(arr.cells.len(), 1);
    }

    #[test]
    fn arrangement_single_split_via_lp() {
        let mut arr = Arrangement::unit_box(10.0, 0);
        let p = Plane { normal: [0.0, 0.0, 1.0], offset: 0.0 };
        arr.add_plane(p, PlaneOrigin::Box, &[]);
        assert_eq!(arr.cells.len(), 2);
    }

    #[test]
    fn arrangement_three_orthogonal_splits_via_lp() {
        let mut arr = Arrangement::unit_box(10.0, 0);
        let planes = [
            Plane { normal: [1.0, 0.0, 0.0], offset: 0.0 },
            Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 },
            Plane { normal: [0.0, 0.0, 1.0], offset: 0.0 },
        ];
        for (i, p) in planes.iter().enumerate() {
            arr.add_plane(*p, PlaneOrigin::Box, &[]);
            let expected = 1 << (i + 1);
            assert_eq!(arr.cells.len(), expected,
                       "after {} planes expected {} cells, got {}",
                       i + 1, expected, arr.cells.len());
        }
    }

    #[test]
    fn witness_roundtrip() {
        let mut arr = Arrangement::unit_box(10.0, 0);
        let p = Plane { normal: [0.0, 0.0, 1.0], offset: 0.0 };
        arr.add_plane(p, PlaneOrigin::Box, &[]);
        for cell in &arr.cells {
            let w = arr.witness_for(&cell.sign).expect("should have a witness");
            // Verify witness satisfies every plane's sign.
            for (k, plane) in arr.planes.iter().enumerate() {
                let expected = cell.sign.sign(k);
                let v = plane.value(&w);
                let got = if v > 0.0 { 1 } else { -1 };
                assert_eq!(got, expected,
                           "plane {}: cell sign {}, witness gives {} (v={})",
                           k, expected, got, v);
            }
        }
    }
}
