//! Plane arrangement with sign-vector cells (no stored polytope vertices).
//!
//! Cells carry only: a 256-bit sign vector (one bit per plane, +1/-1) plus
//! a single witness point used downstream to place a new search vertex.
//!
//! The geometric invariant we rely on throughout:
//!
//!   sv(p, a, b, c) == -||n|| * Plane::through(a, b, c).value(p)
//!
//! i.e. the scalar triple `sv(p, a, b, c) = (a - p) . ((b - p) x (c - p))`
//! and `Plane::through(a,b,c).value(p)` have OPPOSITE signs (and magnitudes
//! differing by the norm of (b-a) x (c-a)).  Every sign convention in this
//! module is expressed directly in terms of `sv(...)` — which is what the
//! pierce test in `seg_crosses_tri` uses — so no per-plane "correction
//! factor" appears downstream.  Permutation parity of triples IS handled
//! explicitly.

pub type Vec3 = [f64; 3];

// --------------------------------------------------------------------------
// Scalar primitives
// --------------------------------------------------------------------------

#[inline] pub fn sub(a: &Vec3, b: &Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] pub fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Scalar triple `sv(x, y, z, w) = (y - x) . ((z - x) x (w - x))`.
/// Six times the signed volume of tetrahedron (x, y, z, w).  Zero iff the
/// four points are coplanar.
#[inline]
pub fn sv(x: &Vec3, y: &Vec3, z: &Vec3, w: &Vec3) -> f64 {
    dot(&sub(y, x), &cross(&sub(z, x), &sub(w, x)))
}

// --------------------------------------------------------------------------
// Plane
// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------
// Pierce test (kept bitwise-identical to cells/main.rs seg_crosses_tri for
// parity tests; all downstream combinatorial tests agree with this one)
// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------
// Permutation parity for reordered triples
// --------------------------------------------------------------------------

/// Sort (x, y, z) ascending; return sorted tuple and the parity of the
/// permutation (input -> sorted), i.e. +1 for even, -1 for odd.  This is
/// the sign correction for `sv(p, x, y, z) = parity * sv(p, x', y', z')`
/// where (x', y', z') is the sorted canonical form.
pub fn sort_triple_with_parity(x: u32, y: u32, z: u32) -> ((u32, u32, u32), i8) {
    let mut a = [x, y, z];
    let mut parity: i8 = 1;
    // Bubble sort with swap counting — 3 elements so at most 3 comparisons.
    if a[0] > a[1] { a.swap(0, 1); parity = -parity; }
    if a[1] > a[2] { a.swap(1, 2); parity = -parity; }
    if a[0] > a[1] { a.swap(0, 1); parity = -parity; }
    ((a[0], a[1], a[2]), parity)
}

// --------------------------------------------------------------------------
// SignVec: up to 256 signs packed in [u64; 4]
// --------------------------------------------------------------------------

pub const MAX_PLANES: usize = 256;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct SignVec {
    pub bits: [u64; 4],
}

impl SignVec {
    pub const fn zero() -> Self { Self { bits: [0; 4] } }

    /// Returns +1 if the bit at `idx` is set, otherwise -1.
    /// Caller must ensure `idx` has been explicitly set via `set_sign` once
    /// after the plane was created; uninitialised slots read as -1.
    #[inline]
    pub fn sign(&self, idx: usize) -> i8 {
        if (self.bits[idx >> 6] >> (idx & 63)) & 1 != 0 { 1 } else { -1 }
    }

    /// Set the sign at `idx` to +1 if `positive`, else -1.
    #[inline]
    pub fn set_sign(&mut self, idx: usize, positive: bool) {
        let w = idx >> 6; let b = idx & 63;
        if positive { self.bits[w] |= 1 << b; } else { self.bits[w] &= !(1 << b); }
    }
}

// --------------------------------------------------------------------------
// 2D arrangement on a cutting plane — Vertex-Sector method
// --------------------------------------------------------------------------

/// A 2D line  a*x + b*y + c = 0  with a² + b² = 1.  `plane_id` refers to
/// the original 3D plane this line came from; it indexes into the full
/// arrangement's plane list and the output sign vectors.
#[derive(Clone, Copy, Debug)]
pub struct Line2D {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub plane_id: u32,
}

impl Line2D {
    pub fn from_abc(a: f64, b: f64, c: f64, plane_id: u32) -> Option<Self> {
        let norm = (a*a + b*b).sqrt();
        if norm < 1e-12 { return None; }
        Some(Line2D { a: a/norm, b: b/norm, c: c/norm, plane_id })
    }
    #[inline]
    pub fn value(&self, x: f64, y: f64) -> f64 {
        self.a * x + self.b * y + self.c
    }
}

/// Intersect two 2D lines; return None if parallel.
pub fn intersect_2d(l1: &Line2D, l2: &Line2D) -> Option<(f64, f64)> {
    let det = l1.a * l2.b - l2.a * l1.b;
    if det.abs() < 1e-12 { return None; }
    let x = (l1.b * l2.c - l2.b * l1.c) / det;
    let y = (l2.a * l1.c - l1.a * l2.c) / det;
    Some((x, y))
}

/// Result of zone enumeration on a cutting plane: one entry per 2D face
/// of the line arrangement on that plane.  Each entry's sign vector gives
/// the signs w.r.t. every line's source plane (i.e. a bit per `plane_id`).
/// The 2D witness is a point strictly in the face's interior.
pub struct Zone {
    pub face_signatures: std::collections::HashSet<SignVec>,
    pub face_witness: std::collections::HashMap<SignVec, (f64, f64)>,
}

fn normalize_angle(a: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut b = a % tau;
    if b < 0.0 { b += tau; }
    b
}

/// Enumerate the 2D arrangement's faces via Gemini's Vertex-Sector method.
///
/// `lines` are the projected 2D lines (already normalised).  `num_bits` is
/// the number of bits to write in each output SignVec — typically the
/// highest `plane_id + 1` used by the caller.
///
/// Degeneracies are handled numerically with ε-clustering of intersection
/// points; this is adequate for Step 2 correctness tests on generic inputs.
/// Combinatorial vertex identification (for the pathological v_i / v_j
/// cases in the N=12 search) is added in Step 3.
pub fn compute_zone(lines: &[Line2D], num_bits: usize) -> Zone {
    let eps_inc = 1e-8;      // "line is incident at V" if |d| < eps_inc
    let eps_dedup = 1e-6;    // cluster points within this distance as the same vertex
    let witness_off = 1e-4;  // offset along bisector to get an interior witness

    // --- Step 1: collect all pairwise intersections as vertex candidates.
    let mut raw: Vec<(f64, f64)> = Vec::new();
    for i in 0..lines.len() {
        for j in (i+1)..lines.len() {
            if let Some(p) = intersect_2d(&lines[i], &lines[j]) {
                raw.push(p);
            }
        }
    }

    // Dedup with ε-clustering.
    let mut vertices: Vec<(f64, f64)> = Vec::new();
    for (x, y) in raw.iter() {
        let merged = vertices.iter().any(|(vx, vy)| {
            let dx = vx - x; let dy = vy - y;
            (dx*dx + dy*dy).sqrt() < eps_dedup
        });
        if !merged { vertices.push((*x, *y)); }
    }

    // --- Step 2: at each vertex, enumerate angular sectors.
    let mut face_signatures: std::collections::HashSet<SignVec> =
        std::collections::HashSet::new();
    let mut face_witness: std::collections::HashMap<SignVec, (f64, f64)> =
        std::collections::HashMap::new();

    for (vx, vy) in &vertices {
        // Classify each line as incident or not.
        let mut incident: Vec<usize> = Vec::new();
        let mut non_inc_pos: Vec<(usize, bool)> = Vec::new();
        for (idx, l) in lines.iter().enumerate() {
            let d = l.value(*vx, *vy);
            if d.abs() < eps_inc {
                incident.push(idx);
            } else {
                non_inc_pos.push((idx, d > 0.0));
            }
        }

        if incident.is_empty() { continue; }

        // Build rays from each incident line: each line contributes two
        // rays emanating from V, in directions ±(−b, a) (perpendicular to
        // the normal).  Record the angle and whether this is the "+dir"
        // ray or the "−dir" ray.
        let mut rays: Vec<f64> = Vec::new();
        for &idx in &incident {
            let l = &lines[idx];
            // Perpendicular to (a,b) is (-b, a); angle is atan2(a, -b).
            let ang1 = normalize_angle(l.a.atan2(-l.b));
            let ang2 = normalize_angle(ang1 + std::f64::consts::PI);
            rays.push(ang1);
            rays.push(ang2);
        }
        rays.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Enumerate sectors between consecutive sorted rays.
        let n = rays.len();
        for i in 0..n {
            let a1 = rays[i];
            let a2 = if i + 1 < n { rays[i+1] } else { rays[0] + 2.0 * std::f64::consts::PI };
            let bis = (a1 + a2) / 2.0;
            let bx = bis.cos();
            let by = bis.sin();

            // Build the sign vector for this sector.
            let mut sv_bits = SignVec::zero();
            for &(idx, positive) in &non_inc_pos {
                let pid = lines[idx].plane_id as usize;
                if pid < num_bits { sv_bits.set_sign(pid, positive); }
            }
            for &idx in &incident {
                let l = &lines[idx];
                let s = l.a * bx + l.b * by;
                let pid = l.plane_id as usize;
                if pid < num_bits { sv_bits.set_sign(pid, s > 0.0); }
            }
            face_signatures.insert(sv_bits);
            // Witness: walk from V in direction (bx, by) until we're about
            // to cross a non-incident line, then step halfway.  This puts
            // the witness in the deep interior of the 2D face rather than
            // arbitrarily close to V.
            let mut max_t = 1.0; // default small step if face is bounded at infinity
            for &(idx, _positive) in &non_inc_pos {
                let l = &lines[idx];
                let denom = l.a * bx + l.b * by;
                if denom.abs() < 1e-12 { continue; }
                let t = -(l.a * vx + l.b * vy + l.c) / denom;
                if t > 1e-7 && t < max_t { max_t = t; }
            }
            let step = 0.5 * max_t;
            let wx = vx + step * bx;
            let wy = vy + step * by;
            face_witness.entry(sv_bits).or_insert((wx, wy));
        }
    }

    Zone { face_signatures, face_witness }
}

// --------------------------------------------------------------------------
// 3D arrangement with sign-vector cells
// --------------------------------------------------------------------------
//
// Sign convention for the entire module:
//
//   sign_vector[plane_id] = sign( planes[plane_id].value(point) )
//
// NOT the sv-scalar-triple sign.  The two differ by `-||n||` per plane,
// but the feasibility test — once derived — compares *pairs* of signs on
// the same plane, so the per-plane factor cancels and no correction is
// needed anywhere.  The sv/plane.value correspondence is unit-tested
// above (`sv_vs_plane_sign_correction_is_minus_one`); it is implicit
// here.

use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub enum PlaneOrigin {
    /// An axis-aligned bounding-box plane.  Not a triple plane; never
    /// referenced by the combinatorial feasibility check.
    Box,
    /// A plane through three placed vertices, canonicalised to sorted
    /// index order.  Indices are into the `placed_verts` list.
    Triple(u32, u32, u32),
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub sign: SignVec,
    pub witness: Vec3,
}

pub struct Arrangement {
    pub planes: Vec<Plane>,
    pub origins: Vec<PlaneOrigin>,
    /// Sorted triple (a,b,c) -> plane_id.  Populated only for `Triple`
    /// planes; box planes don't appear.
    pub plane_of_triple: HashMap<(u32, u32, u32), u32>,
    /// vert_sign[plane_id][placed_vertex_idx] ∈ {-1, +1} — the sign of
    /// `planes[plane_id].value(placed_verts[placed_vertex_idx])`.
    pub vert_sign: Vec<Vec<i8>>,
    pub cells: Vec<Cell>,
}

impl Arrangement {
    /// Create an arrangement initialised with an axis-aligned bounding
    /// box of half-side `box_size` centred at the origin.  One initial
    /// cell with witness at the origin.
    pub fn unit_box(box_size: f64, n_placed: usize) -> Self {
        let mut arr = Arrangement {
            planes: Vec::new(),
            origins: Vec::new(),
            plane_of_triple: HashMap::new(),
            vert_sign: Vec::new(),
            cells: Vec::new(),
        };
        // 6 box planes: +x, -x, +y, -y, +z, -z.  Convention: normal
        // points OUT of the box; inside cell's signs are all -1 (since
        // plane.value(origin) = -box_size for each).
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
            arr.vert_sign.push(vec![0i8; n_placed]); // filled in once verts known
        }
        // Initial cell: witness = origin, all 6 bits = -1 (inside the box).
        let initial = Cell {
            sign: SignVec::zero(),   // all -1 by convention
            witness: [0.0, 0.0, 0.0],
        };
        arr.cells.push(initial);
        arr
    }

    /// Fill in `vert_sign` for a set of placed vertices; must be called
    /// after all planes have been added and before feasibility queries.
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

    /// Add a plane to the arrangement; split every affected cell.
    ///
    /// `placed_verts` is required only to initialise `vert_sign[new_pid]`.
    pub fn add_plane(
        &mut self,
        new_plane: Plane,
        origin: PlaneOrigin,
        placed_verts: &[Vec3],
    ) {
        let new_pid = self.planes.len();

        // Compute an orthonormal basis on the new plane plus an origin
        // point on the plane.
        let (p_origin, e1, e2) = plane_basis(&new_plane);

        // Project every existing plane onto new_plane as a 2D line.
        // Planes that project to a degenerate normal (parallel to
        // new_plane) are remembered separately — their sign at new_plane
        // is constant across the plane.
        let mut lines_2d: Vec<Line2D> = Vec::new();
        let mut parallel: Vec<(u32, bool)> = Vec::new();
        for (qid, q) in self.planes.iter().enumerate() {
            let a = q.normal[0]*e1[0] + q.normal[1]*e1[1] + q.normal[2]*e1[2];
            let b = q.normal[0]*e2[0] + q.normal[1]*e2[1] + q.normal[2]*e2[2];
            let c = q.normal[0]*p_origin[0] + q.normal[1]*p_origin[1]
                  + q.normal[2]*p_origin[2] + q.offset;
            if let Some(line) = Line2D::from_abc(a, b, c, qid as u32) {
                lines_2d.push(line);
            } else {
                parallel.push((qid as u32, c > 0.0));
            }
        }

        // Run the zone enumerator.
        let zone = compute_zone(&lines_2d, self.planes.len());

        // Decorate each face signature with the parallel bits (all 2D
        // faces share the same constant sign against parallel planes).
        let mut cut_set: HashSet<SignVec> = HashSet::with_capacity(zone.face_signatures.len());
        let mut cut_witness_lifted: HashMap<SignVec, Vec3> = HashMap::new();
        let normal_p = new_plane.normal;
        let lift_eps = 1e-4;
        for sig in &zone.face_signatures {
            let mut s = *sig;
            for &(pid, sign) in &parallel {
                if (pid as usize) < self.planes.len() { s.set_sign(pid as usize, sign); }
            }
            cut_set.insert(s);
            // Corresponding 2D witness in this face.
            if let Some(&(wx, wy)) = zone.face_witness.get(sig) {
                let w3 = [
                    p_origin[0] + wx*e1[0] + wy*e2[0],
                    p_origin[1] + wx*e1[1] + wy*e2[1],
                    p_origin[2] + wx*e1[2] + wy*e2[2],
                ];
                cut_witness_lifted.insert(s, w3);
            }
        }

        // Extend every existing cell.  Cut cells split into two; uncut
        // cells gain a single sign bit.
        let mut new_cells: Vec<Cell> = Vec::with_capacity(self.cells.len() + cut_set.len());
        for cell in self.cells.drain(..) {
            if let Some(w3) = cut_witness_lifted.get(&cell.sign) {
                // Cut: produce +P and -P children.
                let mut sig_pos = cell.sign;
                sig_pos.set_sign(new_pid, true);
                let mut sig_neg = cell.sign;
                sig_neg.set_sign(new_pid, false);
                let w_pos = [
                    w3[0] + lift_eps * normal_p[0],
                    w3[1] + lift_eps * normal_p[1],
                    w3[2] + lift_eps * normal_p[2],
                ];
                let w_neg = [
                    w3[0] - lift_eps * normal_p[0],
                    w3[1] - lift_eps * normal_p[1],
                    w3[2] - lift_eps * normal_p[2],
                ];
                new_cells.push(Cell { sign: sig_pos, witness: w_pos });
                new_cells.push(Cell { sign: sig_neg, witness: w_neg });
            } else {
                // Uncut: extend with the witness's side.
                let s = new_plane.value(&cell.witness);
                let mut sig_new = cell.sign;
                sig_new.set_sign(new_pid, s > 0.0);
                new_cells.push(Cell { sign: sig_new, witness: cell.witness });
            }
        }
        self.cells = new_cells;

        // Register the new plane.
        self.planes.push(new_plane);
        self.origins.push(origin);
        if let PlaneOrigin::Triple(a, b, c) = origin {
            let (canonical, _parity) = sort_triple_with_parity(a, b, c);
            self.plane_of_triple.insert(canonical, new_pid as u32);
        }
        // New row in vert_sign for this plane.
        let mut row = Vec::with_capacity(placed_verts.len());
        for v in placed_verts {
            let x = new_plane.value(v);
            row.push(if x > 0.0 { 1 } else { -1 });
        }
        self.vert_sign.push(row);
    }
}

/// Produce an orthonormal (e1, e2) basis on `plane` plus a specific
/// origin point that lies on the plane.
fn plane_basis(plane: &Plane) -> (Vec3, Vec3, Vec3) {
    let n = plane.normal;
    // Origin on the plane: closest point from [0,0,0] is -offset * n.
    let p_origin = [-plane.offset * n[0], -plane.offset * n[1], -plane.offset * n[2]];
    // Build e1 by taking any vector not parallel to n.
    let pick = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1_raw = [
        pick[0] - (n[0]*pick[0] + n[1]*pick[1] + n[2]*pick[2]) * n[0],
        pick[1] - (n[0]*pick[0] + n[1]*pick[1] + n[2]*pick[2]) * n[1],
        pick[2] - (n[0]*pick[0] + n[1]*pick[1] + n[2]*pick[2]) * n[2],
    ];
    let norm = (e1_raw[0]*e1_raw[0] + e1_raw[1]*e1_raw[1] + e1_raw[2]*e1_raw[2]).sqrt();
    let e1 = [e1_raw[0]/norm, e1_raw[1]/norm, e1_raw[2]/norm];
    let e2 = cross(&n, &e1);
    (p_origin, e1, e2)
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny xorshift PRNG (same as cells/main.rs) for deterministic tests
    // without pulling in rand.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self { Rng(seed.max(1)) }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x; x
        }
        fn gauss(&mut self) -> f64 {
            // Marsaglia polar method.
            loop {
                let u = (self.next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0;
                let v = (self.next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0;
                let s = u*u + v*v;
                if s > 0.0 && s < 1.0 {
                    return u * (-2.0 * s.ln() / s).sqrt();
                }
            }
        }
        fn point(&mut self) -> Vec3 { [self.gauss(), self.gauss(), self.gauss()] }
    }

    /// Sanity: sv(p,a,b,c) and Plane::through(a,b,c).value(p) have
    /// OPPOSITE signs on all non-degenerate inputs.
    #[test]
    fn sv_vs_plane_sign_correction_is_minus_one() {
        let mut rng = Rng::new(0xC5A5);
        let mut checked = 0;
        for _ in 0..10_000 {
            let p = rng.point();
            let a = rng.point();
            let b = rng.point();
            let c = rng.point();
            let s = sv(&p, &a, &b, &c);
            let v = Plane::through(&a, &b, &c).value(&p);
            if s.abs() < 1e-8 || v.abs() < 1e-8 { continue; }
            assert!(s * v < 0.0,
                    "expected sv and plane.value to have opposite signs, got sv={}, v={}", s, v);
            checked += 1;
        }
        assert!(checked > 9_500, "rejected too many near-degenerate samples: {}", checked);
    }

    /// Permutation parity: for any reordering (x,y,z) of (a,b,c),
    /// sv(p, x, y, z) == parity * sv(p, a, b, c) where parity is the
    /// sign of the permutation.
    #[test]
    fn sv_permutation_parity() {
        let mut rng = Rng::new(0xDEAD);
        let ids = [0u32, 1, 2];
        // All 6 permutations of (0, 1, 2) with their parities.
        let perms_and_parity: [([u32; 3], i8); 6] = [
            ([0, 1, 2],  1),
            ([0, 2, 1], -1),
            ([1, 0, 2], -1),
            ([1, 2, 0],  1),
            ([2, 0, 1],  1),
            ([2, 1, 0], -1),
        ];
        for _ in 0..200 {
            let p = rng.point();
            let v = [rng.point(), rng.point(), rng.point()];
            let base = sv(&p, &v[0], &v[1], &v[2]);
            if base.abs() < 1e-8 { continue; }
            for (perm, expected_parity) in perms_and_parity.iter() {
                let got = sv(&p, &v[perm[0] as usize], &v[perm[1] as usize], &v[perm[2] as usize]);
                let ratio = got / base;
                assert!((ratio - *expected_parity as f64).abs() < 1e-9,
                        "perm {:?} gave ratio {}, expected {}", perm, ratio, expected_parity);
                // And cross-check sort_triple_with_parity gives the same sign.
                let (_canonical, parity) = sort_triple_with_parity(perm[0], perm[1], perm[2]);
                // Going input -> sorted is the inverse of ids -> perm.
                // Inverse of a permutation has the same parity, so expected_parity == parity.
                assert_eq!(parity, *expected_parity,
                           "sort_triple_with_parity disagrees for perm {:?}", perm);
                let _ = ids; // silence unused
            }
        }
    }

    /// seg_crosses_tri vs a sign-only re-derivation: given just the signs
    /// of the five relevant sv values, the inside-triangle-and-plane test
    /// gives the same verdict.  This pins down the sign-only logic we will
    /// use in the combinatorial feasibility check.
    #[test]
    fn seg_crosses_tri_sign_only_reconstruction() {
        let mut rng = Rng::new(0xF00D);
        let tol = 1e-9;
        let mut agreements = 0;
        let mut non_degenerate = 0;
        for _ in 0..10_000 {
            let p = rng.point();
            let q = rng.point();
            let a = rng.point();
            let b = rng.point();
            let c = rng.point();
            let vp = sv(&p, &a, &b, &c);
            let vq = sv(&q, &a, &b, &c);
            let vab = sv(&p, &q, &a, &b);
            let vbc = sv(&p, &q, &b, &c);
            let vca = sv(&p, &q, &c, &a);
            let min_abs = vp.abs().min(vq.abs()).min(vab.abs()).min(vbc.abs()).min(vca.abs());
            if min_abs < 1e-6 { continue; }
            non_degenerate += 1;
            let geom = seg_crosses_tri(&p, &q, &a, &b, &c, tol);
            // Sign-only reconstruction.
            let plane_ok = (vp > 0.0) != (vq > 0.0);
            let all_pos = vab > 0.0 && vbc > 0.0 && vca > 0.0;
            let all_neg = vab < 0.0 && vbc < 0.0 && vca < 0.0;
            let inside = all_pos || all_neg;
            let combo = plane_ok && inside;
            assert_eq!(geom, combo, "mismatch on non-degenerate inputs");
            agreements += 1;
        }
        assert!(non_degenerate > 9_000,
                "too many degenerate samples skipped: {}", non_degenerate);
        assert_eq!(agreements, non_degenerate);
    }

    /// Three lines forming a triangle: expected 7 cells (1 interior + 3
    /// edge strips + 3 corners).
    #[test]
    fn zone_three_lines_triangle() {
        // Lines of a triangle with vertices (0,0), (3,0), (0,3):
        //   L0: y = 0             -> 0*x + 1*y + 0 = 0
        //   L1: x = 0             -> 1*x + 0*y + 0 = 0
        //   L2: x + y - 3 = 0
        let lines = vec![
            Line2D::from_abc(0.0, 1.0,  0.0, 0).unwrap(),
            Line2D::from_abc(1.0, 0.0,  0.0, 1).unwrap(),
            Line2D::from_abc(1.0, 1.0, -3.0, 2).unwrap(),
        ];
        let zone = compute_zone(&lines, 3);
        assert_eq!(zone.face_signatures.len(), 7,
                   "3 generic lines should make 7 faces, got {}", zone.face_signatures.len());
    }

    /// Two parallel lines: produces 3 faces (two half-planes plus the
    /// strip in between).  There are no vertices, so our current
    /// vertex-sector enumeration returns 0 faces.  This is a known
    /// limitation that a bounding box would fix; documenting it so Step
    /// 3 remembers to add one (the 3D box is always present in the real
    /// arrangement so the real use-case never has fewer than a box's
    /// worth of lines).
    #[test]
    fn zone_two_parallel_lines_yields_nothing_without_bbox() {
        let lines = vec![
            Line2D::from_abc(1.0, 0.0, -1.0, 0).unwrap(),  // x = 1
            Line2D::from_abc(1.0, 0.0, -2.0, 1).unwrap(),  // x = 2
        ];
        let zone = compute_zone(&lines, 2);
        assert_eq!(zone.face_signatures.len(), 0);
    }

    /// Three concurrent lines meeting at origin: 6 sectors.
    #[test]
    fn zone_three_concurrent_lines() {
        let lines = vec![
            Line2D::from_abc(0.0, 1.0, 0.0, 0).unwrap(),     // y=0
            Line2D::from_abc(1.0, 0.0, 0.0, 1).unwrap(),     // x=0
            Line2D::from_abc(1.0, 1.0, 0.0, 2).unwrap(),     // x+y=0
        ];
        let zone = compute_zone(&lines, 3);
        assert_eq!(zone.face_signatures.len(), 6,
                   "3 concurrent lines should make 6 wedges, got {}", zone.face_signatures.len());
    }

    /// Four generic lines in general position: 11 faces.
    ///
    /// General formula: n lines in general position create
    /// 1 + n + C(n,2) = 1 + 4 + 6 = 11 faces.  (Every pair intersects at
    /// a distinct point, so there are no degeneracies and every cell has
    /// a vertex on its boundary.)
    #[test]
    fn zone_four_lines_general_position() {
        let lines = vec![
            Line2D::from_abc(0.0, 1.0,  0.0, 0).unwrap(),    // y=0
            Line2D::from_abc(1.0, 0.0,  0.0, 1).unwrap(),    // x=0
            Line2D::from_abc(1.0, 1.0, -3.0, 2).unwrap(),    // x+y=3
            Line2D::from_abc(1.0, -1.0, 1.0, 3).unwrap(),    // x-y=-1
        ];
        let zone = compute_zone(&lines, 4);
        assert_eq!(zone.face_signatures.len(), 11,
                   "4 generic lines should make 11 faces, got {}", zone.face_signatures.len());
    }

    /// Witnesses returned by the zone enumerator must lie strictly on
    /// the claimed side of each line.
    #[test]
    fn zone_witnesses_respect_signatures() {
        let lines = vec![
            Line2D::from_abc(0.0, 1.0,  0.0, 0).unwrap(),
            Line2D::from_abc(1.0, 0.0,  0.0, 1).unwrap(),
            Line2D::from_abc(1.0, 1.0, -3.0, 2).unwrap(),
            Line2D::from_abc(1.0, -1.0, 1.0, 3).unwrap(),
        ];
        let zone = compute_zone(&lines, 4);
        for (sig, (wx, wy)) in &zone.face_witness {
            for l in &lines {
                let d = l.value(*wx, *wy);
                assert!(d.abs() > 1e-6, "witness sits on line {}: d={}", l.plane_id, d);
                let expected = sig.sign(l.plane_id as usize);
                let got = if d > 0.0 { 1 } else { -1 };
                assert_eq!(got, expected,
                           "witness sign mismatch for line {}: expected {}, got {} (d={})",
                           l.plane_id, expected, got, d);
            }
        }
    }

    /// Empty arrangement starts with one "inside the box" cell whose
    /// witness is the origin and whose sign vector is all zeros.
    #[test]
    fn arrangement_unit_box_one_cell() {
        let arr = Arrangement::unit_box(10.0, 0);
        assert_eq!(arr.planes.len(), 6);
        assert_eq!(arr.cells.len(), 1);
        assert_eq!(arr.cells[0].witness, [0.0, 0.0, 0.0]);
    }

    /// Add a plane through origin: splits 1 cell into 2.
    #[test]
    fn arrangement_single_split() {
        let mut arr = Arrangement::unit_box(10.0, 0);
        // Plane z = 0, normal (0, 0, 1).
        let p = Plane { normal: [0.0, 0.0, 1.0], offset: 0.0 };
        arr.add_plane(p, PlaneOrigin::Box, &[]);
        assert_eq!(arr.cells.len(), 2,
                   "plane through origin should split 1 cell into 2, got {}",
                   arr.cells.len());
    }

    /// Add 3 mutually orthogonal planes through origin inside the box:
    /// 1 -> 2 -> 4 -> 8 cells (octants).
    #[test]
    fn arrangement_three_orthogonal_planes() {
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
                       "after {} orthogonal planes, expected {} cells, got {}",
                       i + 1, expected, arr.cells.len());
        }
    }

    /// General formula: n generic planes in general position crossing
    /// through a bounded box produce at most 1 + n + C(n,2) + C(n,3)
    /// cells inside the box, BUT only planes that actually pierce the
    /// box contribute.  Here we use 4 random planes through points near
    /// origin so they all cut through the box.
    #[test]
    fn arrangement_four_generic_planes() {
        let mut arr = Arrangement::unit_box(10.0, 0);
        let planes = [
            Plane { normal: normalize([1.0, 0.2, 0.1]), offset: 0.0 },
            Plane { normal: normalize([0.3, 1.0, -0.2]), offset: 0.0 },
            Plane { normal: normalize([-0.1, 0.4, 1.0]), offset: 0.0 },
            Plane { normal: normalize([1.0, -1.0, 0.5]), offset: -0.2 },
        ];
        for p in planes.iter() {
            arr.add_plane(*p, PlaneOrigin::Box, &[]);
        }
        // 1 + 4 + C(4,2) + C(4,3) = 1 + 4 + 6 + 4 = 15 cells for planes
        // in general position.
        assert_eq!(arr.cells.len(), 15,
                   "expected 15 cells for 4 generic planes, got {}",
                   arr.cells.len());
    }

    fn normalize(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/n, v[1]/n, v[2]/n]
    }

    /// Witnesses of all cells in the arrangement must satisfy every
    /// plane's sign constraint.
    #[test]
    fn arrangement_witnesses_respect_signs() {
        let mut arr = Arrangement::unit_box(10.0, 0);
        let planes = [
            Plane { normal: normalize([1.0, 0.2, 0.1]), offset: 0.0 },
            Plane { normal: normalize([0.3, 1.0, -0.2]), offset: 0.0 },
            Plane { normal: normalize([-0.1, 0.4, 1.0]), offset: 0.0 },
        ];
        for p in planes.iter() {
            arr.add_plane(*p, PlaneOrigin::Box, &[]);
        }
        for cell in &arr.cells {
            for (pid, plane) in arr.planes.iter().enumerate() {
                let v = plane.value(&cell.witness);
                let expected = cell.sign.sign(pid);
                assert!(v.abs() > 1e-7,
                        "witness of cell with sig {:?} lies on plane {}: v={}",
                        cell.sign, pid, v);
                let got = if v > 0.0 { 1 } else { -1 };
                assert_eq!(got, expected,
                           "plane {}: cell witness sign {} but sig bit {}",
                           pid, got, expected);
            }
        }
    }

    /// SignVec round-trip.
    #[test]
    fn signvec_get_set() {
        let mut sv = SignVec::zero();
        for i in 0..MAX_PLANES {
            assert_eq!(sv.sign(i), -1, "uninitialised bit should read as -1");
        }
        for i in 0..MAX_PLANES {
            sv.set_sign(i, i % 3 == 0);
        }
        for i in 0..MAX_PLANES {
            assert_eq!(sv.sign(i), if i % 3 == 0 { 1 } else { -1 }, "bit {}", i);
        }
    }
}
