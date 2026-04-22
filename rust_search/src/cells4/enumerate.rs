//! Recursive path enumeration toward N=7 with combinatorial piercing
//! and polyhedron extraction.
//!
//! At each depth we pick every cell of the current arrangement in turn,
//! place the next vertex inside it, incrementally add C(N, 2) new
//! triple-planes through the new vertex, and recurse.  At N = target,
//! we run a purely combinatorial pierce-table build followed by a
//! 2-factor extraction on the clean triangles.
//!
//! The hot path reads sign vectors and the chirotope; coords are used
//! only in the splitter's LP witness generation (zone.rs) — every
//! decision outside `add_plane_coord_assisted` is sign-based.

use std::collections::HashMap;
use std::time::Instant;

use crate::arrangement::Arrangement;
use crate::cell::{Cell, PlaneId};
use crate::chirotope::VertId;
use crate::coords::{tet_coords, Vec3};
use crate::piercing::edge_pierces_triangle;
use crate::zone::{add_plane_coord_assisted, witness_coord};

#[derive(Default, Debug, Clone)]
pub struct EnumerateStats {
    pub paths_completed: u64,
    pub polyhedra_found: u64,
    pub best_clean_count: usize,
    pub total_leaves: u64,
    pub lp_failures: u64,
    pub polyhedra: Vec<PolyResult>,
}

#[derive(Debug, Clone)]
pub struct PolyResult {
    pub path: Vec<usize>,         // cell-index choices v_4, v_5, …
    pub placed_coords: Vec<Vec3>, // for visualisation
    pub faces: Vec<[VertId; 3]>,  // selected triangles forming the polyhedron
    pub clean_count: usize,
}

/// Add v_new by picking cell `cell_index`; extends chirotope, places
/// v_new at the cell's LP-witness, adds all new triple-planes.
/// Returns Ok(()) on success, Err(reason) on LP failure.
pub fn add_vertex_from_cell(
    arr: &mut Arrangement,
    cell_index: usize,
    placed_coords: &mut Vec<Vec3>,
) -> Result<(), &'static str> {
    if cell_index >= arr.cells.len() { return Err("cell index out of range"); }
    let cell: Cell = arr.cells[cell_index].clone();

    // Choose the new coord as the LP witness of the cell.
    let new_coord = match witness_coord(&cell.sign, &arr.planes, placed_coords) {
        Some(c) => c,
        None => return Err("LP failed to find witness for chosen cell"),
    };

    // Extend chirotope: χ(v_new, α, β, γ) = σ_cell[plane(α,β,γ)] for each existing plane.
    arr.extend_chirotope(&cell);

    let v_new = placed_coords.len() as u32;
    placed_coords.push(new_coord);

    // Add planes through v_new and every pair of existing vertices.
    for a in 0..v_new {
        for b in (a + 1)..v_new {
            add_plane_coord_assisted(arr, (a, b, v_new), placed_coords);
        }
    }
    arr.n_placed = v_new + 1;
    Ok(())
}

/// Combinatorial piercing table at a given N.
pub fn build_piercings(arr: &Arrangement, n: u32) -> HashMap<([VertId; 2], [VertId; 3]), bool> {
    let mut table = HashMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for a in 0..n {
                for b in (a + 1)..n {
                    for c in (b + 1)..n {
                        if i == a || i == b || i == c { continue; }
                        if j == a || j == b || j == c { continue; }
                        let pierced = edge_pierces_triangle(&arr.chi, i, j, a, b, c);
                        table.insert(([i, j], [a, b, c]), pierced);
                    }
                }
            }
        }
    }
    table
}

/// From the piercing table, identify clean triangles (not pierced by
/// any non-incident edge) and run a 2-factor extractor looking for a
/// K_N-polyhedron: every edge covered by exactly two clean triangles.
pub fn extract_polyhedron(
    pierce: &HashMap<([VertId; 2], [VertId; 3]), bool>,
    n: u32,
) -> (usize, Option<Vec<[VertId; 3]>>) {
    let target_f = (n as usize) * (n as usize - 1) / 3;

    // Enumerate all triangles.
    let mut triangles: Vec<[VertId; 3]> = Vec::new();
    for a in 0..n { for b in (a+1)..n { for c in (b+1)..n {
        triangles.push([a, b, c]);
    }}}

    // A triangle is clean iff no non-incident edge pierces it.
    let mut clean = Vec::new();
    for &t in &triangles {
        let [a, b, c] = t;
        let mut is_clean = true;
        for x in 0..n {
            for y in (x + 1)..n {
                if x == a || x == b || x == c || y == a || y == b || y == c { continue; }
                if *pierce.get(&([x, y], [a, b, c])).unwrap_or(&false) {
                    is_clean = false; break;
                }
            }
            if !is_clean { break; }
        }
        if is_clean { clean.push(t); }
    }
    let n_clean = clean.len();
    if n_clean < target_f { return (n_clean, None); }

    // Edge-coverage upper bound: every edge must be in ≥ 2 clean triangles.
    let mut per_edge: HashMap<[VertId; 2], Vec<usize>> = HashMap::new();
    for (idx, &[a, b, c]) in clean.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (a, c)] {
            let key = if u < v { [u, v] } else { [v, u] };
            per_edge.entry(key).or_default().push(idx);
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            if per_edge.get(&[a, b]).map(|v| v.len()).unwrap_or(0) < 2 {
                return (n_clean, None);
            }
        }
    }

    // Backtracking: select a subset of clean triangles of size target_f
    // such that every edge appears exactly twice.
    let mut selected = vec![false; clean.len()];
    let mut edge_count: HashMap<[VertId; 2], u8> = HashMap::new();
    let mut count = 0usize;
    let mut budget: u64 = 2_000_000;
    let ok = backtrack(&mut selected, &clean, &per_edge, n as usize,
                        &mut edge_count, &mut count, target_f, 0, &mut budget);
    if ok {
        let out: Vec<[VertId; 3]> = selected.iter().enumerate()
            .filter(|&(_, &b)| b).map(|(i, _)| clean[i]).collect();
        return (n_clean, Some(out));
    }
    (n_clean, None)
}

fn backtrack(
    selected: &mut [bool], clean: &[[VertId; 3]],
    per_edge: &HashMap<[VertId; 2], Vec<usize>>, n_verts: usize,
    edge_count: &mut HashMap<[VertId; 2], u8>,
    count: &mut usize, target: usize, start: usize, budget: &mut u64,
) -> bool {
    if *budget == 0 { return false; }
    *budget -= 1;
    if *count == target {
        // Every edge of K_N must be covered exactly twice (not just the
        // ones we happened to touch).  Enumerate all C(n,2) edges
        // explicitly.
        for a in 0..n_verts as VertId {
            for b in (a + 1)..n_verts as VertId {
                let ec = *edge_count.get(&[a, b]).unwrap_or(&0);
                if ec != 2 { return false; }
            }
        }
        return true;
    }
    if start >= clean.len() { return false; }

    // Forward-check: every edge with count < 2 must have enough
    // remaining clean triangles starting at `start`.
    for a in 0..n_verts as VertId {
        for b in (a + 1)..n_verts as VertId {
            let ec = *edge_count.get(&[a, b]).unwrap_or(&0);
            if ec >= 2 { continue; }
            let need = 2 - ec;
            let tris = per_edge.get(&[a, b]);
            let remaining = if let Some(v) = tris {
                v.iter().filter(|&&i| i >= start && !selected[i]).count()
            } else { 0 };
            if (remaining as u8) < need { return false; }
        }
    }

    // Include branch.
    {
        let [a, b, c] = clean[start];
        let keys = [
            if a < b { [a, b] } else { [b, a] },
            if b < c { [b, c] } else { [c, b] },
            if a < c { [a, c] } else { [c, a] },
        ];
        let mut ok = true;
        for k in &keys {
            if *edge_count.get(k).unwrap_or(&0) >= 2 { ok = false; break; }
        }
        if ok {
            for k in &keys { *edge_count.entry(*k).or_insert(0) += 1; }
            selected[start] = true; *count += 1;
            if backtrack(selected, clean, per_edge, n_verts, edge_count,
                          count, target, start + 1, budget) { return true; }
            selected[start] = false; *count -= 1;
            for k in &keys { *edge_count.entry(*k).or_insert(0) -= 1; }
        }
    }

    // Skip branch.
    backtrack(selected, clean, per_edge, n_verts, edge_count,
               count, target, start + 1, budget)
}

/// Depth-first enumeration of every path from N=4 up to `target`.
/// Each path = sequence of cell indices (one per vertex added).  At a
/// leaf we build the combinatorial pierce table and try to extract a
/// polyhedron.  `time_limit_secs` bounds wall time; `max_paths` caps
/// the depth-target enumeration explosion.
pub fn enumerate(target: u32, time_limit_secs: f64, max_paths: u64) -> EnumerateStats {
    let t0 = Instant::now();
    let mut stats = EnumerateStats::default();
    let arr0 = Arrangement::new_tetrahedron();
    let coords0: Vec<Vec3> = tet_coords().to_vec();
    let path0: Vec<usize> = Vec::new();
    enumerate_rec(arr0, coords0, path0, target, &mut stats, t0, time_limit_secs, max_paths);
    stats
}

fn enumerate_rec(
    arr: Arrangement,
    coords: Vec<Vec3>,
    path: Vec<usize>,
    target: u32,
    stats: &mut EnumerateStats,
    t0: Instant,
    time_limit: f64,
    max_paths: u64,
) {
    if t0.elapsed().as_secs_f64() > time_limit { return; }
    if stats.total_leaves >= max_paths { return; }

    if arr.n_placed == target {
        stats.total_leaves += 1;
        let table = build_piercings(&arr, target);
        let (n_clean, poly) = extract_polyhedron(&table, target);
        if n_clean > stats.best_clean_count { stats.best_clean_count = n_clean; }
        if let Some(faces) = poly {
            stats.polyhedra_found += 1;
            stats.paths_completed += 1;
            stats.polyhedra.push(PolyResult {
                path: path.clone(),
                placed_coords: coords.clone(),
                faces,
                clean_count: n_clean,
            });
        } else {
            stats.paths_completed += 1;
        }
        return;
    }

    let n_cells = arr.cells.len();
    for ci in 0..n_cells {
        let mut arr_next = arr.clone();
        let mut coords_next = coords.clone();
        let mut path_next = path.clone();
        path_next.push(ci);
        match add_vertex_from_cell(&mut arr_next, ci, &mut coords_next) {
            Ok(()) => enumerate_rec(arr_next, coords_next, path_next, target, stats,
                                      t0, time_limit, max_paths),
            Err(_) => { stats.lp_failures += 1; }
        }
        if t0.elapsed().as_secs_f64() > time_limit { return; }
        if stats.total_leaves >= max_paths { return; }
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_place_v4_interior() {
        let mut arr = Arrangement::new_tetrahedron();
        let mut coords: Vec<Vec3> = tet_coords().to_vec();
        let interior_idx = arr.cells.iter().position(|c| c.label == "interior").unwrap();
        add_vertex_from_cell(&mut arr, interior_idx, &mut coords).expect("interior place should succeed");
        assert_eq!(arr.n_placed, 5);
        assert_eq!(coords.len(), 5);
        // With 5 placed vertices, arrangement should have C(5,3) = 10 triple-planes.
        assert_eq!(arr.planes.len(), 10);
    }

    #[test]
    fn n5_enumerate_no_polyhedron() {
        // At N=5 target_F = 5*4/3 — not an integer, so no polyhedron
        // possible.  The extractor should return no hits.
        let stats = enumerate(5, 5.0, 100);
        assert_eq!(stats.polyhedra_found, 0,
                   "N=5 can't have a polyhedron (F = 5*4/3 not integer)");
        // But we should have enumerated some leaves.
        assert!(stats.total_leaves > 0);
    }
}
