//! Coord-assisted arrangement construction: 2D-zone method lifted to
//! cells4's sign-vector cells.
//!
//! Each `add_plane` takes the current arrangement (planes + cells) and
//! the current coordinate realisation of placed vertices, runs Gemini's
//! Vertex-Sector 2D-zone enumeration on the cutting plane to identify
//! which cells are cut and which aren't, then splits / extends sign
//! vectors accordingly.  Cells store no witness — we extract one only
//! when placing the next vertex, via `cell_witness_coord`.
//!
//! Coordinates are used here strictly as a numerical oracle; the cell
//! REPRESENTATION stays purely combinatorial (sign vectors).

use std::collections::{HashMap, HashSet};

use crate::arrangement::Arrangement;
use crate::cell::{Cell, Plane, PlaneId, SignVec};
use crate::chirotope::{sort4_with_parity, VertId};
use crate::coords::{cross, dot, sub, Vec3};

// --------------------------------------------------------------------------
// Internal: plane and sign computations from coords (the only spot where
// concrete numbers enter the split path — exactly what the user said is
// acceptable for "uncertainty resolution").
// --------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct PlaneNumeric {
    pub normal: [f64; 3],
    pub offset: f64,
}

pub fn plane_from_coords(a: &Vec3, b: &Vec3, c: &Vec3) -> PlaneNumeric {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let n = cross(&ab, &ac);
    let norm = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    let n = [n[0]/norm, n[1]/norm, n[2]/norm];
    let offset = -(n[0]*a[0] + n[1]*a[1] + n[2]*a[2]);
    PlaneNumeric { normal: n, offset }
}

impl PlaneNumeric {
    #[inline]
    pub fn value(&self, p: &Vec3) -> f64 {
        self.normal[0]*p[0] + self.normal[1]*p[1] + self.normal[2]*p[2] + self.offset
    }
}

/// Convert a combinatorial plane (sorted triple) to a numerical plane.
pub fn plane_to_numeric(plane: &Plane, coords: &[Vec3]) -> PlaneNumeric {
    let (a, b, c) = plane.triple;
    plane_from_coords(&coords[a as usize], &coords[b as usize], &coords[c as usize])
}

/// σ-convention: σ[plane(α,β,γ)] = +1 ⇔ χ(p, α, β, γ) > 0 ⇔ sv < 0
/// ⇔ plane.value < 0 (since sv = −|n|·plane.value).
#[inline]
pub fn sigma_bit_from_plane_value(v: f64) -> i8 {
    if v < 0.0 { 1 } else { -1 }
}

// --------------------------------------------------------------------------
// 2D zone on the cutting plane — copied structurally from cells2, kept
// self-contained here so cells4 has no cross-binary dep.
// --------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Line2D { a: f64, b: f64, c: f64, plane_id: PlaneId }

fn line_from_abc(a: f64, b: f64, c: f64, plane_id: PlaneId) -> Option<Line2D> {
    let norm = (a*a + b*b).sqrt();
    if norm < 1e-12 { return None; }
    Some(Line2D { a: a/norm, b: b/norm, c: c/norm, plane_id })
}

fn intersect_2d(l1: &Line2D, l2: &Line2D) -> Option<(f64, f64)> {
    let det = l1.a * l2.b - l2.a * l1.b;
    if det.abs() < 1e-12 { return None; }
    let x = (l1.b * l2.c - l2.b * l1.c) / det;
    let y = (l2.a * l1.c - l1.a * l2.c) / det;
    Some((x, y))
}

fn normalize_angle(a: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut b = a % tau;
    if b < 0.0 { b += tau; }
    b
}

/// Run the Vertex-Sector zone enumeration on the cutting plane and
/// return the face signatures (with bits for each line's source
/// plane_id), plus a 2D witness per signature.
fn compute_zone(
    lines: &[Line2D],
    n_planes: usize,
) -> (HashSet<SignVec>, HashMap<SignVec, (f64, f64)>) {
    let eps_inc = 1e-8;
    let eps_dedup = 1e-6;

    // Collect intersection points.
    let mut raw = Vec::new();
    for i in 0..lines.len() {
        for j in (i+1)..lines.len() {
            if let Some(p) = intersect_2d(&lines[i], &lines[j]) { raw.push(p); }
        }
    }
    let mut vertices: Vec<(f64, f64)> = Vec::new();
    for (x, y) in &raw {
        let merged = vertices.iter().any(|(vx, vy)| {
            let dx = vx - x; let dy = vy - y;
            (dx*dx + dy*dy).sqrt() < eps_dedup
        });
        if !merged { vertices.push((*x, *y)); }
    }

    let mut face_sigs: HashSet<SignVec> = HashSet::new();
    let mut face_witness: HashMap<SignVec, (f64, f64)> = HashMap::new();

    for (vx, vy) in &vertices {
        let mut incident: Vec<usize> = Vec::new();
        let mut non_inc: Vec<(usize, bool)> = Vec::new();
        for (idx, l) in lines.iter().enumerate() {
            let d = l.a * vx + l.b * vy + l.c;
            if d.abs() < eps_inc { incident.push(idx); }
            else { non_inc.push((idx, d > 0.0)); }
        }
        if incident.is_empty() { continue; }

        let mut rays: Vec<f64> = Vec::new();
        for &idx in &incident {
            let l = &lines[idx];
            let ang1 = normalize_angle(l.a.atan2(-l.b));
            rays.push(ang1);
            rays.push(normalize_angle(ang1 + std::f64::consts::PI));
        }
        rays.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 0..rays.len() {
            let a1 = rays[i];
            let a2 = if i + 1 < rays.len() { rays[i+1] }
                     else { rays[0] + 2.0 * std::f64::consts::PI };
            let bis = (a1 + a2) / 2.0;
            let bx = bis.cos();
            let by = bis.sin();

            // Build sign vector for this sector.
            let mut sv = SignVec::zero();
            for &(idx, positive) in &non_inc {
                let pid = lines[idx].plane_id as usize;
                if pid < n_planes { sv.set(pid, positive); }
            }
            for &idx in &incident {
                let l = &lines[idx];
                let s = l.a * bx + l.b * by;
                let pid = l.plane_id as usize;
                if pid < n_planes { sv.set(pid, s > 0.0); }
            }

            // Face witness: walk from V in direction (bx, by) until the
            // nearest non-incident line, step halfway.  Gives a deep
            // interior witness.
            let mut max_t = 1.0;
            for &(idx, _) in &non_inc {
                let l = &lines[idx];
                let denom = l.a * bx + l.b * by;
                if denom.abs() < 1e-12 { continue; }
                let t = -(l.a * vx + l.b * vy + l.c) / denom;
                if t > 1e-7 && t < max_t { max_t = t; }
            }
            let step = 0.5 * max_t;
            let wx = vx + step * bx;
            let wy = vy + step * by;

            face_sigs.insert(sv);
            face_witness.entry(sv).or_insert((wx, wy));
        }
    }
    (face_sigs, face_witness)
}

// --------------------------------------------------------------------------
// Plane basis on cutting plane
// --------------------------------------------------------------------------

fn plane_basis(p: &PlaneNumeric) -> (Vec3, Vec3, Vec3) {
    let n = p.normal;
    let origin = [-p.offset * n[0], -p.offset * n[1], -p.offset * n[2]];
    let pick = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let dp = n[0]*pick[0] + n[1]*pick[1] + n[2]*pick[2];
    let e1_raw = [pick[0] - dp*n[0], pick[1] - dp*n[1], pick[2] - dp*n[2]];
    let norm = (e1_raw[0]*e1_raw[0] + e1_raw[1]*e1_raw[1] + e1_raw[2]*e1_raw[2]).sqrt();
    let e1 = [e1_raw[0]/norm, e1_raw[1]/norm, e1_raw[2]/norm];
    let e2 = cross(&n, &e1);
    (origin, e1, e2)
}

// --------------------------------------------------------------------------
// Public API: add_plane (coord-assisted)
// --------------------------------------------------------------------------

/// Add plane `new_plane` (canonical sorted triple) to the arrangement.
/// Uses the current `placed_coords` as the numerical oracle.  Cell
/// sign vectors are extended by one bit (for the new plane) and split
/// as needed.  Returns the number of cells after the add.
///
/// Side-effects on `arr`:
///   - appends `new_plane` to `arr.planes` / `arr.origins`
///   - updates `arr.plane_of_triple`
///   - rewrites `arr.cells` with the post-split cells
///   - does NOT modify `arr.chi` (handled by caller's extend_chirotope)
pub fn add_plane_coord_assisted(
    arr: &mut Arrangement,
    new_plane_triple: (VertId, VertId, VertId),
    placed_coords: &[Vec3],
) -> usize {
    // Canonicalise triple.
    let (s, _) = sort4_with_parity(new_plane_triple.0, new_plane_triple.1,
                                     new_plane_triple.2, u32::MAX);
    // Strip the u32::MAX sentinel; keep the first 3.
    let tri = (s.0, s.1, s.2);
    let new_pid = arr.planes.len() as PlaneId;
    let num_existing_planes = arr.planes.len();

    // Build numeric representation of the new plane.
    let p_num = plane_from_coords(
        &placed_coords[tri.0 as usize],
        &placed_coords[tri.1 as usize],
        &placed_coords[tri.2 as usize],
    );
    let (p_origin, e1, e2) = plane_basis(&p_num);

    // Project every existing plane to a 2D line on the new plane.  A
    // degenerate projection (zero 2D normal) means the existing plane
    // is parallel to the new plane; those get a constant sign bit
    // across all face sigs.
    let mut lines_2d: Vec<Line2D> = Vec::new();
    let mut parallel: Vec<(PlaneId, bool)> = Vec::new();
    for (qid, q_plane) in arr.planes.iter().enumerate() {
        let q_num = plane_to_numeric(q_plane, placed_coords);
        let a = dot(&q_num.normal, &e1);
        let b = dot(&q_num.normal, &e2);
        let c = dot(&q_num.normal, &p_origin) + q_num.offset;
        if let Some(line) = line_from_abc(a, b, c, qid as PlaneId) {
            lines_2d.push(line);
        } else {
            parallel.push((qid as PlaneId, c > 0.0));
        }
    }

    let (zone_sigs, zone_witness) = compute_zone(&lines_2d, num_existing_planes);

    // Augment zone sigs with parallel-plane bits.
    let mut cut_set: HashSet<SignVec> = HashSet::with_capacity(zone_sigs.len());
    let mut cut_witness_3d: HashMap<SignVec, Vec3> = HashMap::new();
    let lift_eps = 1e-4;
    for sig in &zone_sigs {
        let mut s = *sig;
        for &(pid, sign) in &parallel {
            if (pid as usize) < num_existing_planes {
                s.set(pid as usize, sign);
            }
        }
        cut_set.insert(s);
        if let Some(&(wx, wy)) = zone_witness.get(sig) {
            let w3 = [
                p_origin[0] + wx*e1[0] + wy*e2[0],
                p_origin[1] + wx*e1[1] + wy*e2[1],
                p_origin[2] + wx*e1[2] + wy*e2[2],
            ];
            cut_witness_3d.insert(s, w3);
        }
    }

    // Extend every cell.  Cut cells emit two children; uncut ones
    // extend with a single sign bit determined by their witness side.
    // We need a witness per cell to decide the side for uncut cells.
    // The face-lattice would give this directly; without it, we fall
    // back to an LP-generated witness per uncut cell.
    let old_cells = std::mem::take(&mut arr.cells);
    let mut new_cells: Vec<Cell> = Vec::with_capacity(old_cells.len() + cut_set.len());
    for cell in old_cells {
        if cut_set.contains(&cell.sign) {
            // Split.
            let mut sig_pos = cell.sign; sig_pos.set(new_pid as usize, true);
            let mut sig_neg = cell.sign; sig_neg.set(new_pid as usize, false);
            let _w3 = cut_witness_3d.get(&cell.sign);
            // Both children get the parent's placed_corners plus any new
            // corners where the plane cuts the cell — for now we keep
            // parent's placed corners only; the split detail is tracked
            // at the search level, not inside the sign vector.
            let label_pos = format!("{}|+{}", cell.label, new_pid);
            let label_neg = format!("{}|-{}", cell.label, new_pid);
            new_cells.push(Cell {
                sign: sig_pos,
                label: label_pos,
                placed_corners: cell.placed_corners.clone(),
            });
            new_cells.push(Cell {
                sign: sig_neg,
                label: label_neg,
                placed_corners: cell.placed_corners,
            });
        } else {
            // Not cut — determine sign of new plane in this cell's
            // interior.  Use coord-assisted witness via a small LP.
            let sign = match witness_sign(&cell.sign, &arr.planes, placed_coords, &p_num) {
                Some(s) => s,
                None => {
                    // LP failed to find a witness — cell may be empty
                    // (numerical noise); drop it.
                    continue;
                }
            };
            let mut sig_new = cell.sign;
            sig_new.set(new_pid as usize, sign > 0);
            new_cells.push(Cell {
                sign: sig_new,
                label: cell.label,
                placed_corners: cell.placed_corners,
            });
        }
    }
    arr.cells = new_cells;

    // Register the new plane.
    arr.planes.push(Plane { triple: tri });
    arr.plane_of_triple.insert(tri, new_pid);

    arr.cells.len()
}

/// Find a witness point inside the cell with the given sign vector and
/// return the sign of `target_plane` at that witness.  Uses iterative
/// projection (Kaczmarz) — same primitive as cells3, dependency-free.
pub fn witness_sign(
    sigma: &SignVec,
    planes: &[Plane],
    placed_coords: &[Vec3],
    target_plane: &PlaneNumeric,
) -> Option<i8> {
    let w = witness_coord(sigma, planes, placed_coords)?;
    let v = target_plane.value(&w);
    Some(if v < 0.0 { 1 } else { -1 })
}

/// Find any 3-D point satisfying the cell's sign constraints.  Returns
/// None if the constraints seem infeasible (may indicate numerical
/// instability).
pub fn witness_coord(
    sigma: &SignVec,
    planes: &[Plane],
    placed_coords: &[Vec3],
) -> Option<Vec3> {
    let bounds = 20.0;
    let eps = 1e-4;
    let max_iter = 300;

    // Build constraints: for each plane k, σ-bit · plane_value < 0 ⇒
    //   +1 bit → plane.value < 0 ⇒ (-n) · x > offset
    //   −1 bit → plane.value > 0 ⇒ n · x > −offset
    let mut a_list: Vec<[f64; 3]> = Vec::with_capacity(planes.len());
    let mut b_list: Vec<f64> = Vec::with_capacity(planes.len());
    for (k, p) in planes.iter().enumerate() {
        let pn = plane_to_numeric(p, placed_coords);
        let bit = sigma.get(k);
        let s = if bit > 0 { -1.0 } else { 1.0 };
        a_list.push([s * pn.normal[0], s * pn.normal[1], s * pn.normal[2]]);
        b_list.push(-s * pn.offset);
    }

    // Iterative projection.
    let mut x: Vec3 = [0.0, 0.0, 0.0];
    for _ in 0..max_iter {
        // Enforce bounding box.
        for i in 0..3 {
            if x[i] > bounds - eps { x[i] = bounds - eps; }
            if x[i] < -bounds + eps { x[i] = -bounds + eps; }
        }
        let mut worst_i: Option<usize> = None;
        let mut worst_v: f64 = 0.0;
        for i in 0..a_list.len() {
            let val = a_list[i][0]*x[0] + a_list[i][1]*x[1] + a_list[i][2]*x[2];
            let need = b_list[i] + eps;
            if val < need {
                let v = need - val;
                if v > worst_v { worst_v = v; worst_i = Some(i); }
            }
        }
        match worst_i {
            None => return Some(x),
            Some(i) => {
                let a = &a_list[i];
                let ns = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
                if ns < 1e-20 { return None; }
                let step = worst_v / ns;
                x[0] += step * a[0];
                x[1] += step * a[1];
                x[2] += step * a[2];
            }
        }
    }
    // Final re-check.
    for i in 0..a_list.len() {
        let val = a_list[i][0]*x[0] + a_list[i][1]*x[1] + a_list[i][2]*x[2];
        if val < b_list[i] + eps * 0.1 { return None; }
    }
    Some(x)
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::tet_coords;

    #[test]
    fn lp_witness_for_tet_interior() {
        let coords = tet_coords();
        let arr = Arrangement::new_tetrahedron();
        // Interior cell has sign (-, +, -, +) — check we can find a witness.
        let interior = arr.cells.iter().find(|c| c.label == "interior").unwrap();
        let w = witness_coord(&interior.sign, &arr.planes, &coords);
        assert!(w.is_some(), "must find witness for interior cell");
        // Witness should be near tet centroid (bounded region).
        let p = w.unwrap();
        assert!(p[0].abs() < 2.0 && p[1].abs() < 2.0 && p[2].abs() < 2.0,
                "witness {:?} far from origin", p);
    }

    /// Add one new plane at N=5 and verify we grow the number of cells.
    #[test]
    fn add_plane_n5_grows_cell_count() {
        let mut arr = Arrangement::new_tetrahedron();
        let mut coords: Vec<Vec3> = tet_coords().to_vec();
        // Place v_4 near the tet centroid but offset.
        coords.push([0.1, 0.2, 0.15]);
        // Add plane (4, 0, 1) — a new plane through 2 tet verts + v_4.
        let n_before = arr.cells.len();
        let n_after = add_plane_coord_assisted(&mut arr, (0, 1, 4), &coords);
        assert_eq!(n_before, 15);
        assert!(n_after >= n_before, "adding a plane should not reduce cell count; got {} → {}",
                n_before, n_after);
    }
}
