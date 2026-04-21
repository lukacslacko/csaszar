//! Backtracking search using the sign-vector arrangement.
//!
//! Mirrors `cells::backtrack_search` in `rust_search/src/main.rs` but
//! replaces the vertex-based `build_arrangement` + `cell_is_feasible`
//! with the cells2 combinatorial-sign-vector-based machinery.
//!
//! The outer algorithm is identical; full arrangement rebuilds happen at
//! each level (same semantics as the reference `cells`).

use std::collections::HashMap;
use std::time::Instant;

use crate::arrangement::{
    seg_crosses_tri, Arrangement, Plane, PlaneOrigin, Vec3,
};
use crate::feasibility::{build_pierce_tests, is_feasible};

// --------------------------------------------------------------------------
// Utilities (copied from cells binary; kept identical for deterministic
// seed compatibility when doing parity tests).
// --------------------------------------------------------------------------

pub struct Xorshift { state: u64 }
impl Xorshift {
    pub fn new(seed: u64) -> Self { Xorshift { state: seed.max(1) } }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }
    pub fn pick(&mut self, n: usize) -> usize { (self.next_u64() as usize) % n }
}

pub fn shuffle<T>(v: &mut Vec<T>, rng: &mut Xorshift) {
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        v.swap(i, j);
    }
}

pub fn regular_tetrahedron() -> Vec<Vec3> {
    let s3 = (3f64).sqrt();
    let s6 = (6f64).sqrt();
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, s3 / 2.0, 0.0],
        [0.5, s3 / 6.0, s6 / 3.0],
    ]
}

// Per-edge face count for the incremental face-commitment code, stored
// in a flat 12x12 (max N=12) to avoid hash overhead.
#[derive(Clone, Debug)]
pub struct EdgeFaceCount { data: [u8; 144] }
impl EdgeFaceCount {
    pub fn new() -> Self { EdgeFaceCount { data: [0; 144] } }
    #[inline] fn idx(a: u32, b: u32) -> usize {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        (a as usize) * 12 + (b as usize)
    }
    pub fn get(&self, a: u32, b: u32) -> u8 { self.data[Self::idx(a, b)] }
    pub fn inc(&mut self, a: u32, b: u32) { self.data[Self::idx(a, b)] += 1; }
    pub fn dec(&mut self, a: u32, b: u32) { self.data[Self::idx(a, b)] -= 1; }
    pub fn inc_face(&mut self, f: [u32; 3]) { self.inc(f[0], f[1]); self.inc(f[1], f[2]); self.inc(f[0], f[2]); }
    pub fn dec_face(&mut self, f: [u32; 3]) { self.dec(f[0], f[1]); self.dec(f[1], f[2]); self.dec(f[0], f[2]); }
}

pub fn current_rss_bytes() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            #[cfg(target_os = "macos")]
            { return usage.ru_maxrss as u64; }
            #[cfg(not(target_os = "macos"))]
            { return (usage.ru_maxrss as u64).saturating_mul(1024); }
        }
    }
    0
}

// --------------------------------------------------------------------------
// New-face conflict check (geometric; called only when committing a face,
// so the cell-count-per-call is tiny)
// --------------------------------------------------------------------------

fn triangles_cross(t1: [u32; 3], t2: [u32; 3], verts: &[Vec3]) -> bool {
    let contains = |t: [u32; 3], v: u32| t[0] == v || t[1] == v || t[2] == v;
    let tol = 1e-9;
    for &(a, b) in &[(t1[0], t1[1]), (t1[1], t1[2]), (t1[0], t1[2])] {
        if contains(t2, a) || contains(t2, b) { continue; }
        if seg_crosses_tri(&verts[a as usize], &verts[b as usize],
                            &verts[t2[0] as usize], &verts[t2[1] as usize], &verts[t2[2] as usize],
                            tol) {
            return true;
        }
    }
    for &(a, b) in &[(t2[0], t2[1]), (t2[1], t2[2]), (t2[0], t2[2])] {
        if contains(t1, a) || contains(t1, b) { continue; }
        if seg_crosses_tri(&verts[a as usize], &verts[b as usize],
                            &verts[t1[0] as usize], &verts[t1[1] as usize], &verts[t1[2] as usize],
                            tol) {
            return true;
        }
    }
    false
}

fn new_face_conflicts(f: [u32; 3], committed: &[[u32; 3]], verts: &[Vec3]) -> bool {
    for other in committed {
        if triangles_cross(f, *other, verts) { return true; }
    }
    false
}

// --------------------------------------------------------------------------
// Polyhedron extract (copied verbatim from cells binary; same semantics)
// --------------------------------------------------------------------------

fn combo(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    let mut r = 1usize;
    for i in 0..k { r = r * (n - i) / (i + 1); }
    r
}

pub fn extract_polyhedron(
    verts: &[Vec3],
    committed_faces: &[[u32; 3]],
) -> (usize, Option<Vec<[u32; 3]>>) {
    let n = verts.len();
    let target_f = n * (n - 1) / 3;
    let tol = 1e-9;

    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(combo(n, 3));
    for a in 0..n { for b in (a+1)..n { for c in (b+1)..n {
        triangles.push([a as u32, b as u32, c as u32]);
    }}}

    let mut clean: Vec<bool> = vec![true; triangles.len()];
    for (ti, &[a, b, c]) in triangles.iter().enumerate() {
        let ta = &verts[a as usize]; let tb = &verts[b as usize]; let tc = &verts[c as usize];
        for x in 0..n {
            for y in (x+1)..n {
                let xu = x as u32; let yu = y as u32;
                if xu == a || xu == b || xu == c { continue; }
                if yu == a || yu == b || yu == c { continue; }
                if seg_crosses_tri(&verts[x], &verts[y], ta, tb, tc, tol) {
                    clean[ti] = false; break;
                }
            }
            if !clean[ti] { break; }
        }
    }

    let clean_set: Vec<[u32; 3]> = triangles.iter().enumerate()
        .filter(|&(i, _)| clean[i]).map(|(_, t)| *t).collect();
    let n_clean = clean_set.len();
    if n_clean < target_f { return (n_clean, None); }

    let mut tris_per_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (idx, &[a, b, c]) in clean_set.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (a, c)] {
            let key = if u < v { (u, v) } else { (v, u) };
            tris_per_edge.entry(key).or_default().push(idx);
        }
    }
    for a in 0..n as u32 {
        for b in (a+1)..n as u32 {
            if tris_per_edge.get(&(a, b)).map(|v| v.len()).unwrap_or(0) < 2 {
                return (n_clean, None);
            }
        }
    }

    let clean_index: HashMap<[u32; 3], usize> =
        clean_set.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let mut forced: Vec<usize> = Vec::new();
    for f in committed_faces {
        let mut s = *f; s.sort();
        if let Some(&idx) = clean_index.get(&s) { forced.push(idx); } else {
            return (n_clean, None);
        }
    }

    let mut selected: Vec<bool> = vec![false; clean_set.len()];
    let mut edge_count: HashMap<(u32, u32), u8> = HashMap::new();
    for &idx in &forced {
        selected[idx] = true;
        let [a, b, c] = clean_set[idx];
        for (u, v) in [(a, b), (b, c), (a, c)] {
            let key = if u < v { (u, v) } else { (v, u) };
            let e = edge_count.entry(key).or_insert(0);
            *e += 1;
            if *e > 2 { return (n_clean, None); }
        }
    }

    let n_selected_forced = forced.len();
    let mut count = n_selected_forced;
    let mut budget: u64 = 2_000_000;
    let found = polyhedron_backtrack(
        &mut selected, &clean_set, &tris_per_edge, n,
        &mut edge_count, &mut count, target_f, 0, &mut budget,
    );
    if found {
        let out: Vec<[u32; 3]> = selected.iter().enumerate()
            .filter(|&(_, &b)| b).map(|(i, _)| clean_set[i]).collect();
        return (n_clean, Some(out));
    }
    (n_clean, None)
}

fn polyhedron_backtrack(
    selected: &mut [bool],
    clean: &[[u32; 3]],
    tris_per_edge: &HashMap<(u32, u32), Vec<usize>>,
    n_verts: usize,
    edge_count: &mut HashMap<(u32, u32), u8>,
    count: &mut usize,
    target: usize,
    start: usize,
    budget: &mut u64,
) -> bool {
    if *budget == 0 { return false; }
    *budget -= 1;
    if *count == target {
        return edge_count.values().all(|&c| c == 2);
    }
    if start >= clean.len() { return false; }

    for a in 0..n_verts as u32 {
        for b in (a+1)..n_verts as u32 {
            let ec = *edge_count.get(&(a, b)).unwrap_or(&0);
            if ec >= 2 { continue; }
            let need = 2 - ec;
            let tris = tris_per_edge.get(&(a, b));
            let remaining = if let Some(v) = tris {
                v.iter().filter(|&&i| i >= start && !selected[i]).count()
            } else { 0 };
            if (remaining as u8) < need { return false; }
        }
    }

    if !selected[start] {
        let [a, b, c] = clean[start];
        let keys = [
            if a < b { (a, b) } else { (b, a) },
            if b < c { (b, c) } else { (c, b) },
            if a < c { (a, c) } else { (c, a) },
        ];
        let mut ok = true;
        for k in &keys {
            if *edge_count.get(k).unwrap_or(&0) >= 2 { ok = false; break; }
        }
        if ok {
            for k in &keys { *edge_count.entry(*k).or_insert(0) += 1; }
            selected[start] = true; *count += 1;
            if polyhedron_backtrack(selected, clean, tris_per_edge, n_verts, edge_count,
                          count, target, start + 1, budget) {
                return true;
            }
            selected[start] = false; *count -= 1;
            for k in &keys { *edge_count.entry(*k).or_insert(0) -= 1; }
        }
    }
    polyhedron_backtrack(selected, clean, tris_per_edge, n_verts, edge_count,
               count, target, start + 1, budget)
}

// --------------------------------------------------------------------------
// Combinatorial pruning (upper-bound on clean triangles per edge)
// --------------------------------------------------------------------------
//
// Copied from cells::cannot_reach_polyhedron — this stays geometric because
// it operates on partial vertex lists and hasn't been reformulated as pure
// sign-vector logic.  Runs in O(n^5) with n ≤ 12, so ~160k ops; negligible.

pub fn cannot_reach_polyhedron(verts: &[Vec3], n_target: usize) -> bool {
    let placed = verts.len();
    if placed < 3 { return false; }
    let tol = 1e-9;

    let mut alive_count: HashMap<(u32, u32), u32> = HashMap::new();
    for a in 0..placed {
        for b in (a + 1)..placed {
            alive_count.insert((a as u32, b as u32), 0);
        }
    }

    for a in 0..placed {
        for b in (a + 1)..placed {
            for c in (b + 1)..placed {
                let ta = &verts[a]; let tb = &verts[b]; let tc = &verts[c];
                let mut alive = true;
                'outer: for x in 0..placed {
                    if x == a || x == b || x == c { continue; }
                    for y in (x + 1)..placed {
                        if y == a || y == b || y == c { continue; }
                        if seg_crosses_tri(&verts[x], &verts[y], ta, tb, tc, tol) {
                            alive = false; break 'outer;
                        }
                    }
                }
                if alive {
                    *alive_count.get_mut(&(a as u32, b as u32)).unwrap() += 1;
                    *alive_count.get_mut(&(b as u32, c as u32)).unwrap() += 1;
                    *alive_count.get_mut(&(a as u32, c as u32)).unwrap() += 1;
                }
            }
        }
    }

    let future = (n_target - placed) as u32;
    for (_, &alive) in alive_count.iter() {
        if alive + future < 2 { return true; }
    }
    false
}

// --------------------------------------------------------------------------
// Feasible-cells-at: full rebuild + combinatorial feasibility filter
// --------------------------------------------------------------------------

pub fn build_full_arrangement(verts: &[Vec3], box_size: f64) -> Arrangement {
    let n = verts.len();
    let mut arr = Arrangement::unit_box(box_size, n);
    for a in 0..n as u32 {
        for b in (a+1)..n as u32 {
            for c in (b+1)..n as u32 {
                let plane = Plane::through(
                    &verts[a as usize], &verts[b as usize], &verts[c as usize]);
                arr.add_plane(plane, PlaneOrigin::Triple(a, b, c), verts);
            }
        }
    }
    arr.recompute_vert_signs(verts);
    arr
}

/// Return the list of witnesses of all feasible cells, subject to the
/// combinatorial pierce test against committed_faces.
pub fn feasible_witnesses(
    arr: &Arrangement,
    committed: &[[u32; 3]],
    n_placed: usize,
) -> Vec<Vec3> {
    let tests = match build_pierce_tests(arr, committed, n_placed) {
        Ok(t) => t,
        Err(msg) => { eprintln!("  [pierce-test build error] {}", msg); return Vec::new(); }
    };
    let mut out = Vec::new();
    for cell in &arr.cells {
        if is_feasible(&cell.sign, &tests) { out.push(cell.witness); }
    }
    out
}

// --------------------------------------------------------------------------
// Backtracking search — mirrors cells::backtrack_search
// --------------------------------------------------------------------------

struct LevelState {
    committed_pushed: usize,
    cells_to_try: Vec<Vec3>,
}

pub fn backtrack_search(
    seed: u64, target: usize, box_size: f64, commit_until: usize,
    deadline_secs: f64, mem_limit_bytes: u64,
) {
    backtrack_search_inner(seed, target, box_size, commit_until,
                            deadline_secs, mem_limit_bytes, false);
}

pub fn backtrack_search_verbose(
    seed: u64, target: usize, box_size: f64, commit_until: usize,
    deadline_secs: f64, mem_limit_bytes: u64,
) {
    backtrack_search_inner(seed, target, box_size, commit_until,
                            deadline_secs, mem_limit_bytes, true);
}

pub fn backtrack_search_inner(
    seed: u64, target: usize, box_size: f64, commit_until: usize,
    deadline_secs: f64, mem_limit_bytes: u64, verbose: bool,
) {
    let t0 = Instant::now();
    let mut rng = Xorshift::new(seed);
    let mut verts = regular_tetrahedron();
    let mut committed: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3]];
    let mut edge_fc = EdgeFaceCount::new();
    for &f in &committed { edge_fc.inc_face(f); }

    let mut stack: Vec<LevelState> = Vec::new();
    let mut visits: u64 = 0;
    let mut prunes: u64 = 0;
    let mut target_reached: u64 = 0;
    let mut polys_found: u64 = 0;
    let mut best_clean: usize = 0;
    let mut last_report = t0;

    // Level 0: feasible cells for placing v_4.
    let arr = build_full_arrangement(&verts, box_size);
    let mut cells0 = feasible_witnesses(&arr, &committed, verts.len());
    if verbose {
        println!("  [lvl0] placed={} cells={} feasible={}",
                 verts.len(), arr.cells.len(), cells0.len());
    }
    shuffle(&mut cells0, &mut rng);
    stack.push(LevelState { committed_pushed: 0, cells_to_try: cells0 });

    let mut stop_reason = "search exhausted";

    loop {
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > deadline_secs { stop_reason = "time limit"; break; }
        let rss = current_rss_bytes();
        if rss > mem_limit_bytes { stop_reason = "memory limit"; break; }

        if last_report.elapsed().as_secs_f64() >= 2.0 {
            println!(
                "  t={:>6.1}s  rss={:>5.2}GB  depth={:>2}  stack={:>2}  visits={:>8}  prunes={:>7}  tgt={:>5}  polys={}  best_clean={}",
                elapsed, rss as f64 / (1024.0 * 1024.0 * 1024.0),
                verts.len(), stack.len(), visits, prunes, target_reached,
                polys_found, best_clean,
            );
            last_report = Instant::now();
        }

        if stack.is_empty() { stop_reason = "search exhausted"; break; }

        if stack.last().unwrap().cells_to_try.is_empty() {
            let ls = stack.pop().unwrap();
            for _ in 0..ls.committed_pushed {
                let f = committed.pop().unwrap();
                edge_fc.dec_face(f);
            }
            if verts.len() > 4 { verts.pop(); }
            continue;
        }

        let cand = stack.last_mut().unwrap().cells_to_try.pop().unwrap();
        visits += 1;
        verts.push(cand);
        let i = (verts.len() - 1) as u32;

        // Random face commitment for v_i when i < commit_until.
        let mut committed_this_level = 0usize;
        if (i as usize) < commit_until {
            let mut potential: Vec<(u32, u32)> = Vec::new();
            for j in 0..i {
                for k in (j + 1)..i {
                    if edge_fc.get(j, k) < 2 { potential.push((j, k)); }
                }
            }
            shuffle(&mut potential, &mut rng);
            for &(j, k) in &potential {
                if edge_fc.get(j, k) >= 2 { continue; }
                if edge_fc.get(i, j) >= 2 { continue; }
                if edge_fc.get(i, k) >= 2 { continue; }
                let face = [i, j, k];
                if new_face_conflicts(face, &committed, &verts) { continue; }
                if rng.next_u64() & 1 == 1 {
                    edge_fc.inc_face(face);
                    committed.push(face);
                    committed_this_level += 1;
                }
            }
        }

        // Combinatorial prune (identical to cells binary).
        if cannot_reach_polyhedron(&verts, target) {
            prunes += 1;
            for _ in 0..committed_this_level {
                let f = committed.pop().unwrap();
                edge_fc.dec_face(f);
            }
            verts.pop();
            continue;
        }

        // Leaf: extract polyhedron.
        if verts.len() == target {
            target_reached += 1;
            let (n_clean, poly) = extract_polyhedron(&verts, &committed);
            if n_clean > best_clean { best_clean = n_clean; }
            if let Some(p) = poly {
                polys_found += 1;
                println!("  [!] polyhedron #{} found at visit {} (t={:.1}s), {} faces",
                         polys_found, visits, t0.elapsed().as_secs_f64(), p.len());
            }
            for _ in 0..committed_this_level {
                let f = committed.pop().unwrap();
                edge_fc.dec_face(f);
            }
            verts.pop();
            continue;
        }

        // Build next level's arrangement + feasible witnesses.
        let t_build = Instant::now();
        let arr = build_full_arrangement(&verts, box_size);
        let mut next_cells = feasible_witnesses(&arr, &committed, verts.len());
        let build_elapsed = t_build.elapsed().as_secs_f64();
        if build_elapsed > 1.0 || verbose {
            eprintln!("  [build] placed={} cells={} feasible={} t={:.2}s",
                      verts.len(), arr.cells.len(), next_cells.len(), build_elapsed);
        }
        shuffle(&mut next_cells, &mut rng);
        stack.push(LevelState { committed_pushed: committed_this_level, cells_to_try: next_cells });
    }

    let dt = t0.elapsed().as_secs_f64();
    let rss = current_rss_bytes();
    println!("\n=== Backtrack (cells2) done ({}) ===", stop_reason);
    println!("  elapsed: {:.2}s  peak_rss: {:.2}GB", dt, rss as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  visits (cells picked):      {}", visits);
    println!("  depth target reached:       {}", target_reached);
    println!("  combinatorial prunes:       {}", prunes);
    println!("  polyhedra found:            {}", polys_found);
    println!("  best clean-tri count seen:  {}", best_clean);
}
