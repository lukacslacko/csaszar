// Hybrid cell-based + face-commit search.
//
// Two knobs:
//
//   - `target_depth`: how many vertices to place in total.
//   - `commit_until`: for each vertex v_i with i < commit_until, we
//     also enumerate all valid face-subsets at v_i and commit them;
//     for i >= commit_until we only pick the cell (no new face
//     commitment).
//
// The initial committed faces (0,1,2) and (0,1,3) of the anchoring
// tetrahedron are always committed.  Feasibility of a cell always
// checks against every committed face — so even after commit_until we
// still reap the benefit of tighter pruning from earlier commitments.

use std::time::Instant;

type Vec3 = [f64; 3];

// ============================================================================
// Primitives
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: [f64; 3],
    offset: f64,
}

impl Plane {
    fn through(a: &Vec3, b: &Vec3, c: &Vec3) -> Self {
        let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let n = [
            ab[1]*ac[2] - ab[2]*ac[1],
            ab[2]*ac[0] - ab[0]*ac[2],
            ab[0]*ac[1] - ab[1]*ac[0],
        ];
        let norm = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
        let n = [n[0]/norm, n[1]/norm, n[2]/norm];
        let offset = -(n[0]*a[0] + n[1]*a[1] + n[2]*a[2]);
        Plane { normal: n, offset }
    }
    #[inline]
    fn value(&self, p: &Vec3) -> f64 {
        self.normal[0]*p[0] + self.normal[1]*p[1] + self.normal[2]*p[2] + self.offset
    }
}

#[inline] fn sub(a: &Vec3, b: &Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn dot(a: &Vec3, b: &Vec3) -> f64 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
#[inline] fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

fn seg_crosses_tri(p: &Vec3, q: &Vec3, a: &Vec3, b: &Vec3, c: &Vec3, tol: f64) -> bool {
    #[inline]
    fn sv(x: &Vec3, y: &Vec3, z: &Vec3, w: &Vec3) -> f64 {
        dot(&sub(y, x), &cross(&sub(z, x), &sub(w, x)))
    }
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

// ============================================================================
// Cell representation + splitting
// ============================================================================

#[derive(Debug, Clone)]
struct Cell {
    facets: Vec<(u32, i8)>,
    vertices: Vec<Vec3>,
    vertex_facets: Vec<Vec<u32>>,
}

impl Cell {
    fn unit_box(planes: &mut Vec<Plane>, b: f64) -> Cell {
        let mut facets = Vec::with_capacity(6);
        let axes: [[f64; 3]; 6] = [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ];
        for n in axes.iter() {
            let id = planes.len() as u32;
            planes.push(Plane { normal: *n, offset: -b });
            facets.push((id, -1));
        }
        let mut vertices: Vec<Vec3> = Vec::with_capacity(8);
        for &x in &[-b, b] { for &y in &[-b, b] { for &z in &[-b, b] { vertices.push([x, y, z]); }}}
        let mut vertex_facets: Vec<Vec<u32>> = Vec::with_capacity(8);
        for v in &vertices {
            let mut vf = Vec::with_capacity(3);
            vf.push(if v[0] > 0.0 { 0 } else { 1 });
            vf.push(if v[1] > 0.0 { 2 } else { 3 });
            vf.push(if v[2] > 0.0 { 4 } else { 5 });
            vertex_facets.push(vf);
        }
        Cell { facets, vertices, vertex_facets }
    }
    fn centroid(&self) -> Vec3 {
        let n = self.vertices.len() as f64;
        let mut c = [0.0; 3];
        for v in &self.vertices { c[0] += v[0]; c[1] += v[1]; c[2] += v[2]; }
        [c[0]/n, c[1]/n, c[2]/n]
    }

    fn split_by(self, plane: &Plane, plane_id: u32, tol: f64) -> SplitResult {
        let n_v = self.vertices.len();
        let mut vals: Vec<f64> = Vec::with_capacity(n_v);
        let mut signs: Vec<i8> = Vec::with_capacity(n_v);
        let mut has_pos = false; let mut has_neg = false;
        for v in &self.vertices {
            let x = plane.value(v);
            vals.push(x);
            let s = if x > tol { 1 } else if x < -tol { -1 } else { 0 };
            signs.push(s);
            if s > 0 { has_pos = true; }
            if s < 0 { has_neg = true; }
        }
        if !has_pos && !has_neg { return SplitResult::NoSplit(self); }
        if !has_neg {
            let Cell { mut facets, vertices, mut vertex_facets } = self;
            let new_idx = facets.len() as u32;
            facets.push((plane_id, 1));
            for (i, &s) in signs.iter().enumerate() { if s == 0 { vertex_facets[i].push(new_idx); } }
            return SplitResult::NoSplit(Cell { facets, vertices, vertex_facets });
        }
        if !has_pos {
            let Cell { mut facets, vertices, mut vertex_facets } = self;
            let new_idx = facets.len() as u32;
            facets.push((plane_id, -1));
            for (i, &s) in signs.iter().enumerate() { if s == 0 { vertex_facets[i].push(new_idx); } }
            return SplitResult::NoSplit(Cell { facets, vertices, vertex_facets });
        }
        let old_facet_count = self.facets.len() as u32;
        let new_plane_facet_local = old_facet_count;
        let mut crossing_verts: Vec<Vec3> = Vec::new();
        let mut crossing_vf: Vec<Vec<u32>> = Vec::new();
        for i in 0..n_v {
            for j in (i+1)..n_v {
                if signs[i] * signs[j] >= 0 { continue; }
                let mut shared: Vec<u32> = Vec::new();
                for &f in &self.vertex_facets[i] {
                    if self.vertex_facets[j].contains(&f) { shared.push(f); }
                }
                if shared.len() < 2 { continue; }
                let t = vals[i] / (vals[i] - vals[j]);
                let vi = &self.vertices[i]; let vj = &self.vertices[j];
                let new_v = [
                    vi[0]*(1.0-t) + vj[0]*t,
                    vi[1]*(1.0-t) + vj[1]*t,
                    vi[2]*(1.0-t) + vj[2]*t,
                ];
                shared.push(new_plane_facet_local);
                crossing_verts.push(new_v);
                crossing_vf.push(shared);
            }
        }
        let est = n_v + crossing_verts.len();
        let mut pos_verts = Vec::with_capacity(est);
        let mut pos_vf = Vec::with_capacity(est);
        let mut neg_verts = Vec::with_capacity(est);
        let mut neg_vf = Vec::with_capacity(est);
        let Cell { facets, vertices, vertex_facets } = self;
        for ((v, vf), &s) in vertices.into_iter().zip(vertex_facets.into_iter()).zip(signs.iter()) {
            match s {
                1 => { pos_verts.push(v); pos_vf.push(vf); }
                -1 => { neg_verts.push(v); neg_vf.push(vf); }
                _ => {
                    let mut vf_with = vf.clone();
                    vf_with.push(new_plane_facet_local);
                    pos_verts.push(v); pos_vf.push(vf_with.clone());
                    neg_verts.push(v); neg_vf.push(vf_with);
                }
            }
        }
        for (cv, cvf) in crossing_verts.into_iter().zip(crossing_vf.into_iter()) {
            pos_verts.push(cv); pos_vf.push(cvf.clone());
            neg_verts.push(cv); neg_vf.push(cvf);
        }
        let mut pos_facets = facets.clone(); pos_facets.push((plane_id, 1));
        let mut neg_facets = facets; neg_facets.push((plane_id, -1));
        SplitResult::Split(
            Cell { facets: pos_facets, vertices: pos_verts, vertex_facets: pos_vf },
            Cell { facets: neg_facets, vertices: neg_verts, vertex_facets: neg_vf },
        )
    }
}

#[derive(Debug)]
enum SplitResult { NoSplit(Cell), Split(Cell, Cell) }

#[derive(Debug, Clone)]
struct Arrangement { planes: Vec<Plane>, cells: Vec<Cell> }

impl Arrangement {
    fn new(box_size: f64) -> Self {
        let mut planes = Vec::new();
        let box_cell = Cell::unit_box(&mut planes, box_size);
        Arrangement { planes, cells: vec![box_cell] }
    }
    fn add_plane(&mut self, plane: Plane, tol: f64) {
        let t_plane = Instant::now();
        let mut last_inner = t_plane;
        let plane_id = self.planes.len() as u32;
        self.planes.push(plane);
        let old = std::mem::take(&mut self.cells);
        let old_len = old.len();
        let mut new_cells: Vec<Cell> = Vec::with_capacity(old_len * 2);
        let mut max_seen_verts: usize = 0;
        for (idx, c) in old.into_iter().enumerate() {
            let pre_v = c.vertices.len();
            if pre_v > max_seen_verts { max_seen_verts = pre_v; }
            let t_cell = Instant::now();
            match c.split_by(&plane, plane_id, tol) {
                SplitResult::NoSplit(c) => {
                    let dt = t_cell.elapsed().as_secs_f64();
                    if dt > 0.5 {
                        eprintln!("    [split] slow NoSplit cell idx={}/{} v_before={} v_after={} t={:.2}s",
                                  idx, old_len, pre_v, c.vertices.len(), dt);
                    }
                    new_cells.push(c);
                }
                SplitResult::Split(a, b) => {
                    let dt = t_cell.elapsed().as_secs_f64();
                    if dt > 0.5 {
                        eprintln!("    [split] slow Split cell idx={}/{} v_before={} v_out={}+{} t={:.2}s",
                                  idx, old_len, pre_v, a.vertices.len(), b.vertices.len(), dt);
                    }
                    new_cells.push(a); new_cells.push(b);
                }
            }
            if last_inner.elapsed().as_secs_f64() >= 1.0 {
                eprintln!("    [inner] cells_done={}/{} new_cells_so_far={} max_v={} t_plane={:.2}s",
                          idx + 1, old_len, new_cells.len(), max_seen_verts,
                          t_plane.elapsed().as_secs_f64());
                last_inner = Instant::now();
            }
        }
        self.cells = new_cells;
    }
}

fn current_rss_bytes() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            // macOS: ru_maxrss is in bytes; Linux: kilobytes.
            #[cfg(target_os = "macos")]
            { return usage.ru_maxrss as u64; }
            #[cfg(not(target_os = "macos"))]
            { return (usage.ru_maxrss as u64).saturating_mul(1024); }
        }
    }
    0
}

/// Build arrangement.  If `mem_limit_bytes` is Some, RSS is polled
/// after every added plane; exceeding it returns None.  No hard cell
/// or vertex count caps — we rely on the RSS check to abort before OOM.
/// Emits a heartbeat line to stderr if the build takes >1s, showing
/// per-plane progress every 500ms.
fn build_arrangement(
    verts: &[Vec3], box_size: f64, mem_limit_bytes: Option<u64>,
) -> Option<Arrangement> {
    let t_start = Instant::now();
    let mut last_print = t_start;
    let mut arr = Arrangement::new(box_size);
    let n = verts.len();
    let total_planes = n * (n - 1) * (n - 2) / 6;
    let mut plane_idx: usize = 0;
    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                arr.add_plane(Plane::through(&verts[i], &verts[j], &verts[k]), 1e-9);
                plane_idx += 1;
                if let Some(lim) = mem_limit_bytes {
                    if current_rss_bytes() > lim { return None; }
                }
                let elapsed = t_start.elapsed().as_secs_f64();
                if elapsed > 1.0 && last_print.elapsed().as_secs_f64() >= 0.5 {
                    eprintln!("  [arr placed={}] plane {}/{} cells={} t={:.2}s rss={:.2}GB",
                              n, plane_idx, total_planes, arr.cells.len(), elapsed,
                              current_rss_bytes() as f64 / (1024.0*1024.0*1024.0));
                    last_print = Instant::now();
                }
            }
        }
    }
    let total = t_start.elapsed().as_secs_f64();
    if total > 1.0 {
        eprintln!("  [arr done placed={}] cells={} t={:.2}s", n, arr.cells.len(), total);
    }
    Some(arr)
}

// ============================================================================
// Edge-face counter
// ============================================================================

#[derive(Debug, Clone)]
struct EdgeFaceCount { data: [u8; 144] }
impl EdgeFaceCount {
    fn new() -> Self { EdgeFaceCount { data: [0; 144] } }
    #[inline] fn idx(a: u32, b: u32) -> usize {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        (a as usize) * 12 + (b as usize)
    }
    #[inline] fn get(&self, a: u32, b: u32) -> u8 { self.data[Self::idx(a, b)] }
    #[inline] fn inc(&mut self, a: u32, b: u32) { self.data[Self::idx(a, b)] += 1; }
    #[inline] fn dec(&mut self, a: u32, b: u32) { self.data[Self::idx(a, b)] -= 1; }
    fn inc_face(&mut self, f: [u32; 3]) { self.inc(f[0], f[1]); self.inc(f[1], f[2]); self.inc(f[0], f[2]); }
    fn dec_face(&mut self, f: [u32; 3]) { self.dec(f[0], f[1]); self.dec(f[1], f[2]); self.dec(f[0], f[2]); }
}

// ============================================================================
// Feasibility
// ============================================================================

fn cell_is_feasible(p: &Vec3, verts: &[Vec3], committed_faces: &[[u32; 3]]) -> bool {
    let tol = 1e-9;
    for (j_usize, vj) in verts.iter().enumerate() {
        let j = j_usize as u32;
        for face in committed_faces {
            if face[0] == j || face[1] == j || face[2] == j { continue; }
            let a = &verts[face[0] as usize];
            let b = &verts[face[1] as usize];
            let c = &verts[face[2] as usize];
            if seg_crosses_tri(vj, p, a, b, c, tol) { return false; }
        }
    }
    true
}

fn new_face_conflicts(f: [u32; 3], committed_faces: &[[u32; 3]], verts: &[Vec3]) -> bool {
    for other in committed_faces {
        if triangles_cross(f, *other, verts) { return true; }
    }
    false
}

/// Upper-bound pruning check.  For each edge on placed vertices, count
/// "alive" triangles — triangles on placed vertices not yet pierced by
/// any non-incident placed edge.  An edge needs at least 2 clean
/// triangles in any final polyhedron, drawn from alive placed ones
/// plus future triangles (one per unplaced vertex).  If alive+future < 2
/// for any edge, no polyhedron can possibly be extracted.
fn cannot_reach_polyhedron(verts: &[Vec3], n_target: usize) -> bool {
    let placed = verts.len();
    if placed < 3 { return false; }
    let tol = 1e-9;

    // alive_count[(a,b)] = number of alive placed triangles containing edge (a,b)
    let mut alive_count: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();
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

// ============================================================================
// Options + statistics
// ============================================================================

// ============================================================================
// Leaf: try to extract a K_N polyhedron
// ============================================================================
//
// At a depth-target leaf we have `verts.len() == N` and some set of
// committed faces (at least the two anchor ones).  To determine if a
// polyhedron can be built from this vertex configuration:
//
//   1. Compute the pierces matrix — for every (edge, non-incident
//      triangle) pair, does the edge pierce the triangle's interior?
//   2. A triangle is "clean" iff no non-incident edge pierces it.
//      Only clean triangles can be faces of the final polyhedron.
//   3. Coverage check: every edge must appear in >= 2 clean triangles
//      (since every edge of a K_N neighborly triangulation is in
//      exactly 2 faces).  If not, give up immediately.
//   4. Combinatorial face selection: backtrack over clean triangles,
//      tracking per-edge face count (must land at exactly 2 for
//      every edge and exactly `target_F` total faces).  The already
//      committed faces are forced picks; the search explores which
//      of the remaining clean triangles to include.

fn combo(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    let mut r = 1usize;
    for i in 0..k { r = r * (n - i) / (i + 1); }
    r
}

fn extract_polyhedron(
    verts: &[Vec3],
    committed_faces: &[[u32; 3]],
) -> (usize, Option<Vec<[u32; 3]>>) {
    let n = verts.len();
    let target_f = n * (n - 1) / 3;
    let tol = 1e-9;

    // Enumerate all triangles (sorted vertex triples).
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(combo(n, 3));
    for a in 0..n { for b in (a+1)..n { for c in (b+1)..n {
        triangles.push([a as u32, b as u32, c as u32]);
    }}}

    // For each triangle, check every non-incident edge.  Mark the
    // triangle "clean" only if no such edge pierces its interior.
    let mut clean: Vec<bool> = vec![true; triangles.len()];
    for (ti, &[a, b, c]) in triangles.iter().enumerate() {
        let A = &verts[a as usize]; let B = &verts[b as usize]; let C = &verts[c as usize];
        for x in 0..n {
            for y in (x+1)..n {
                let xu = x as u32; let yu = y as u32;
                if xu == a || xu == b || xu == c { continue; }
                if yu == a || yu == b || yu == c { continue; }
                if seg_crosses_tri(&verts[x], &verts[y], A, B, C, tol) {
                    clean[ti] = false;
                    break;
                }
            }
            if !clean[ti] { break; }
        }
    }

    let clean_set: Vec<[u32; 3]> = triangles.iter().enumerate()
        .filter(|&(i, _)| clean[i]).map(|(_, t)| *t).collect();
    let n_clean = clean_set.len();
    if n_clean < target_f { return (n_clean, None); }

    // Coverage check: each edge must have at least 2 clean triangles.
    let mut tris_per_edge: std::collections::HashMap<(u32, u32), Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, &[a, b, c]) in clean_set.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (a, c)] {
            let key = if u < v { (u, v) } else { (v, u) };
            tris_per_edge.entry(key).or_default().push(idx);
        }
    }
    // Every edge of K_N must be covered.
    for a in 0..n as u32 {
        for b in (a+1)..n as u32 {
            if tris_per_edge.get(&(a, b)).map(|v| v.len()).unwrap_or(0) < 2 {
                return (n_clean, None);
            }
        }
    }

    // Backtracking selection.
    // `must_include`: committed faces that MUST be in the final set.
    // Filter out any that aren't in clean_set (shouldn't happen if the
    // commitments were ever valid, but we guard anyway).
    let clean_index: std::collections::HashMap<[u32; 3], usize> =
        clean_set.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let mut forced: Vec<usize> = Vec::new();
    for f in committed_faces {
        let mut s = *f; s.sort();
        if let Some(&idx) = clean_index.get(&s) { forced.push(idx); } else {
            return (n_clean, None); // committed face isn't clean — impossible
        }
    }

    let mut selected: Vec<bool> = vec![false; clean_set.len()];
    let mut edge_count: std::collections::HashMap<(u32, u32), u8> =
        std::collections::HashMap::new();
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
    let t_ext = Instant::now();
    let mut last_print = t_ext;
    let found = backtrack(&mut selected, &clean_set, &tris_per_edge, n,
                  &mut edge_count, &mut count, target_f, 0, &mut budget,
                  t_ext, &mut last_print);
    let total = t_ext.elapsed().as_secs_f64();
    if total > 1.0 {
        eprintln!("  [ext done] clean={} forced={} budget_used={} t={:.2}s found={}",
                  n_clean, n_selected_forced, 2_000_000 - budget, total, found);
    }
    if found {
        let out: Vec<[u32; 3]> = selected.iter().enumerate()
            .filter(|&(_, &b)| b).map(|(i, _)| clean_set[i]).collect();
        return (n_clean, Some(out));
    }
    (n_clean, None)
}

fn backtrack(
    selected: &mut [bool],
    clean: &[[u32; 3]],
    tris_per_edge: &std::collections::HashMap<(u32, u32), Vec<usize>>,
    n_verts: usize,
    edge_count: &mut std::collections::HashMap<(u32, u32), u8>,
    count: &mut usize,
    target: usize,
    start: usize,
    budget: &mut u64,
    t_ext: Instant,
    last_print: &mut Instant,
) -> bool {
    if *budget == 0 { return false; }
    *budget -= 1;
    // Heartbeat every ~50k budget decrements, once elapsed > 1s.
    if *budget % 50_000 == 0 {
        let elapsed = t_ext.elapsed().as_secs_f64();
        if elapsed > 1.0 && last_print.elapsed().as_secs_f64() >= 0.5 {
            eprintln!("  [ext bt] t={:.2}s budget_left={} count={}/{} start={}/{}",
                      elapsed, *budget, *count, target, start, clean.len());
            *last_print = Instant::now();
        }
    }
    if *count == target {
        return edge_count.values().all(|&c| c == 2);
    }
    if start >= clean.len() { return false; }

    // Forward-check: for every edge with count < 2, the edges'
    // remaining clean triangles starting at `start` must be enough
    // to bring the count to 2.
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

    // Include branch
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
            if backtrack(selected, clean, tris_per_edge, n_verts, edge_count,
                          count, target, start + 1, budget, t_ext, last_print) {
                return true;
            }
            selected[start] = false; *count -= 1;
            for k in &keys { *edge_count.entry(*k).or_insert(0) -= 1; }
        }
    }
    // Skip branch
    backtrack(selected, clean, tris_per_edge, n_verts, edge_count,
               count, target, start + 1, budget, t_ext, last_print)
}

#[derive(Clone, Copy)]
struct Opts {
    target: usize,
    commit_until: usize,     // commit faces for v_i iff i < commit_until
    box_size: f64,
    first_only: bool,
}

struct Stats {
    cells_at_level: Vec<Vec<usize>>,
    face_subsets_at_level: Vec<Vec<usize>>,
    complete: u64,
    nodes_expanded: u64,
    best_clean: usize,
    poly_found: u64,
    best_poly: Option<Vec<[u32; 3]>>,
}
impl Stats {
    fn new() -> Self { Stats {
        cells_at_level: Vec::new(), face_subsets_at_level: Vec::new(),
        complete: 0, nodes_expanded: 0,
        best_clean: 0, poly_found: 0, best_poly: None,
    }}
    fn record_cells(&mut self, lvl: usize, n: usize) {
        while self.cells_at_level.len() <= lvl { self.cells_at_level.push(Vec::new()); }
        self.cells_at_level[lvl].push(n);
    }
    fn record_subsets(&mut self, lvl: usize, n: usize) {
        while self.face_subsets_at_level.len() <= lvl { self.face_subsets_at_level.push(Vec::new()); }
        self.face_subsets_at_level[lvl].push(n);
    }
}

// ============================================================================
// DFS
// ============================================================================

fn dfs(
    verts: &mut Vec<Vec3>,
    committed_faces: &mut Vec<[u32; 3]>,
    edge_face_count: &mut EdgeFaceCount,
    opts: &Opts,
    stats: &mut Stats,
) {
    if opts.first_only && stats.complete > 0 { return; }

    let i = verts.len() as u32;
    if verts.len() == opts.target {
        stats.complete += 1;
        let (n_clean, poly) = extract_polyhedron(verts, committed_faces);
        if n_clean > stats.best_clean { stats.best_clean = n_clean; }
        if poly.is_some() {
            stats.poly_found += 1;
            if stats.best_poly.is_none() { stats.best_poly = poly; }
        }
        return;
    }

    let lvl = (verts.len() - 4) as usize;

    let arr = match build_arrangement(verts, opts.box_size, None) {
        Some(a) => a,
        None => {
            stats.record_cells(lvl, 0);
            eprintln!("  [warn] arrangement cap hit at placed={}", verts.len());
            return;
        }
    };
    let mut feasible: Vec<Vec3> = Vec::new();
    for cell in &arr.cells {
        let c = cell.centroid();
        if cell_is_feasible(&c, verts, committed_faces) { feasible.push(c); }
    }
    stats.record_cells(lvl, feasible.len());

    for cand in feasible {
        if opts.first_only && stats.complete > 0 { return; }
        stats.nodes_expanded += 1;
        verts.push(cand);

        // At vertex index i we're placing v_i.  If i < commit_until we
        // enumerate all valid face subsets; otherwise just go straight
        // to recursing on the next vertex (no new face commitment).
        if (i as usize) < opts.commit_until {
            let mut potential: Vec<(u32, u32)> = Vec::new();
            for j in 0..i {
                for k in (j + 1)..i {
                    if edge_face_count.get(j, k) < 2 {
                        potential.push((j, k));
                    }
                }
            }
            let mut subsets_for_this_cell: u64 = 0;
            subsets_dfs(
                i, 0, &potential,
                verts, committed_faces, edge_face_count,
                &mut subsets_for_this_cell,
                opts, stats,
            );
            stats.record_subsets(lvl, subsets_for_this_cell as usize);
        } else {
            // No face commitment for this vertex.
            dfs(verts, committed_faces, edge_face_count, opts, stats);
        }

        verts.pop();
    }
}

fn subsets_dfs(
    i: u32, idx: usize, potential: &[(u32, u32)],
    verts: &mut Vec<Vec3>,
    committed_faces: &mut Vec<[u32; 3]>,
    edge_face_count: &mut EdgeFaceCount,
    subsets_at_cell: &mut u64,
    opts: &Opts,
    stats: &mut Stats,
) {
    if opts.first_only && stats.complete > 0 { return; }
    if idx == potential.len() {
        *subsets_at_cell += 1;
        dfs(verts, committed_faces, edge_face_count, opts, stats);
        return;
    }

    // In `first_only` mode, bias the order to alternate include/skip
    // at every other potential face, so the first subset picked is
    // "middle-density" rather than all-empty or all-full.
    // Specifically: at even `idx` try INCLUDE first; at odd `idx` try
    // SKIP first.  This gives roughly half the faces committed (among
    // the ones that pass the geometric/edge-budget constraints).
    let include_first = opts.first_only && (idx % 2 == 0);

    let skip_branch = |
        committed_faces: &mut Vec<[u32; 3]>,
        edge_face_count: &mut EdgeFaceCount,
        subsets_at_cell: &mut u64,
        stats: &mut Stats,
        verts: &mut Vec<Vec3>,
    | {
        subsets_dfs(i, idx + 1, potential, verts, committed_faces, edge_face_count,
                     subsets_at_cell, opts, stats);
    };

    let include_branch = |
        committed_faces: &mut Vec<[u32; 3]>,
        edge_face_count: &mut EdgeFaceCount,
        subsets_at_cell: &mut u64,
        stats: &mut Stats,
        verts: &mut Vec<Vec3>,
    | -> bool {
        let (j, k) = potential[idx];
        let face = [i, j, k];
        if edge_face_count.get(j, k) >= 2 { return false; }
        if edge_face_count.get(i, j) >= 2 { return false; }
        if edge_face_count.get(i, k) >= 2 { return false; }
        if new_face_conflicts(face, committed_faces, verts) { return false; }
        edge_face_count.inc_face(face);
        committed_faces.push(face);
        subsets_dfs(i, idx + 1, potential, verts, committed_faces, edge_face_count,
                     subsets_at_cell, opts, stats);
        committed_faces.pop();
        edge_face_count.dec_face(face);
        true
    };

    if include_first {
        include_branch(committed_faces, edge_face_count, subsets_at_cell, stats, verts);
        if opts.first_only && stats.complete > 0 { return; }
        skip_branch(committed_faces, edge_face_count, subsets_at_cell, stats, verts);
    } else {
        skip_branch(committed_faces, edge_face_count, subsets_at_cell, stats, verts);
        if opts.first_only && stats.complete > 0 { return; }
        include_branch(committed_faces, edge_face_count, subsets_at_cell, stats, verts);
    }
}

// ============================================================================
// main
// ============================================================================

// ============================================================================
// Random descent
// ============================================================================

/// Tiny xorshift PRNG.  No external dependencies.
struct Xorshift { state: u64 }
impl Xorshift {
    fn new(seed: u64) -> Self { Xorshift { state: seed.max(1) } }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x; x
    }
    fn pick(&mut self, n: usize) -> usize { (self.next_u64() as usize) % n }
}

struct DescentResult {
    level_reached: usize,
    per_level_cells: Vec<usize>,
    n_clean: Option<usize>,
    poly_ok: bool,
    time_secs: f64,
    committed_at_level: Vec<usize>,
    pruned_at_level: Option<usize>, // Some(k) if combinatorial prune fired at level k
}

/// One random descent: at each level, enumerate feasible cells and
/// pick exactly ONE uniformly at random.  After placing v_i, if
/// i < commit_until, randomly commit a subset of the potential new
/// faces (i, j, k) for j<k<i (subject to edge-budget and geometric
/// conflict constraints).  No backtracking — if a level has 0 feasible
/// cells, the descent fails at that level.  At depth `target` we run
/// the polyhedron extraction.
fn random_descent(
    seed: u64,
    target: usize,
    box_size: f64,
    commit_until: usize,
    use_prune: bool,
) -> DescentResult {
    let t0 = Instant::now();
    let mut rng = Xorshift::new(seed);
    let mut verts = regular_tetrahedron();
    let mut committed_faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3]];
    let mut edge_face_count = EdgeFaceCount::new();
    for &f in &committed_faces { edge_face_count.inc_face(f); }
    let mut per_level_cells = Vec::new();
    let mut committed_at_level: Vec<usize> = Vec::new();

    while verts.len() < target {
        let arr = match build_arrangement(&verts, box_size, None) {
            Some(a) => a,
            None => {
                eprintln!("  [warn] seed {} arrangement cap hit at placed={}",
                          seed, verts.len());
                return DescentResult {
                    level_reached: verts.len(),
                    per_level_cells,
                    n_clean: None,
                    poly_ok: false,
                    time_secs: t0.elapsed().as_secs_f64(),
                    committed_at_level,
                    pruned_at_level: None,
                };
            }
        };
        let mut feasible: Vec<Vec3> = Vec::new();
        for cell in &arr.cells {
            let c = cell.centroid();
            if cell_is_feasible(&c, &verts, &committed_faces) { feasible.push(c); }
        }
        per_level_cells.push(feasible.len());
        if feasible.is_empty() {
            return DescentResult {
                level_reached: verts.len(),
                per_level_cells,
                n_clean: None,
                poly_ok: false,
                time_secs: t0.elapsed().as_secs_f64(),
                committed_at_level,
                pruned_at_level: None,
            };
        }
        let pick = rng.pick(feasible.len());
        verts.push(feasible[pick]);
        let i = (verts.len() - 1) as u32;

        // Random face commitment for v_i when i < commit_until.
        let mut committed_this_level = 0usize;
        if (i as usize) < commit_until {
            let mut potential: Vec<(u32, u32)> = Vec::new();
            for j in 0..i {
                for k in (j + 1)..i {
                    if edge_face_count.get(j, k) < 2 { potential.push((j, k)); }
                }
            }
            for m in (1..potential.len()).rev() {
                let r = (rng.next_u64() as usize) % (m + 1);
                potential.swap(m, r);
            }
            for &(j, k) in &potential {
                if edge_face_count.get(j, k) >= 2 { continue; }
                if edge_face_count.get(i, j) >= 2 { continue; }
                if edge_face_count.get(i, k) >= 2 { continue; }
                let face = [i, j, k];
                if new_face_conflicts(face, &committed_faces, &verts) { continue; }
                if rng.next_u64() & 1 == 1 {
                    edge_face_count.inc_face(face);
                    committed_faces.push(face);
                    committed_this_level += 1;
                }
            }
        }
        committed_at_level.push(committed_this_level);

        // Combinatorial upper-bound pruning.
        if use_prune && cannot_reach_polyhedron(&verts, target) {
            return DescentResult {
                level_reached: verts.len(),
                per_level_cells,
                n_clean: None,
                poly_ok: false,
                time_secs: t0.elapsed().as_secs_f64(),
                committed_at_level,
                pruned_at_level: Some(verts.len()),
            };
        }
    }
    let (n_clean, poly) = extract_polyhedron(&verts, &committed_faces);
    DescentResult {
        level_reached: target,
        per_level_cells,
        n_clean: Some(n_clean),
        poly_ok: poly.is_some(),
        time_secs: t0.elapsed().as_secs_f64(),
        committed_at_level,
        pruned_at_level: None,
    }
}

fn regular_tetrahedron() -> Vec<Vec3> {
    let s3 = (3f64).sqrt();
    let s6 = (6f64).sqrt();
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, s3 / 2.0, 0.0],
        [0.5, s3 / 6.0, s6 / 3.0],
    ]
}

fn run_random(
    count: u64, seed_base: u64, target: usize, box_size: f64,
    commit_until: usize, use_prune: bool,
) {
    let t0 = Instant::now();
    let mut per_level_cells: Vec<Vec<usize>> = Vec::new();
    let mut committed_per_level: Vec<Vec<usize>> = Vec::new();
    let mut clean_counts: Vec<usize> = Vec::new();
    let mut reached_target: u64 = 0;
    let mut poly_found: u64 = 0;
    let mut failed: u64 = 0;
    let mut pruned: u64 = 0;
    let mut prune_level_hist: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();
    let mut level_reached_hist: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();

    for i in 0..count {
        let r = random_descent(seed_base.wrapping_add(i), target, box_size,
                                 commit_until, use_prune);
        *level_reached_hist.entry(r.level_reached).or_insert(0) += 1;
        for (lvl, &cells) in r.per_level_cells.iter().enumerate() {
            while per_level_cells.len() <= lvl { per_level_cells.push(Vec::new()); }
            per_level_cells[lvl].push(cells);
        }
        for (lvl, &committed) in r.committed_at_level.iter().enumerate() {
            while committed_per_level.len() <= lvl { committed_per_level.push(Vec::new()); }
            committed_per_level[lvl].push(committed);
        }
        if let Some(k) = r.pruned_at_level {
            pruned += 1;
            *prune_level_hist.entry(k).or_insert(0) += 1;
        }
        if r.level_reached != target { failed += 1; continue; }
        reached_target += 1;
        if r.poly_ok { poly_found += 1; }
        if let Some(nc) = r.n_clean { clean_counts.push(nc); }
        if i % 8 == 0 || i == count - 1 {
            println!("  [{}/{}]  reached_target={}  poly_found={}  pruned={}  elapsed={:.1}s",
                     i + 1, count, reached_target, poly_found, pruned,
                     t0.elapsed().as_secs_f64());
        }
    }

    let dt = t0.elapsed().as_secs_f64();
    println!("\n=== Random descent done in {:.2}s ({} attempts) ===", dt, count);
    println!("  reached depth {}: {}  (failed: {})", target, reached_target, failed);
    println!("  polyhedra found: {}", poly_found);
    if use_prune {
        println!("  pruned by combinatorial check: {}", pruned);
        if !prune_level_hist.is_empty() {
            let mut keys: Vec<&usize> = prune_level_hist.keys().collect();
            keys.sort();
            println!("  prune-level histogram:");
            for k in keys {
                println!("    placed {} verts: {} descents pruned",
                         k, prune_level_hist[k]);
            }
        }
    }
    println!("  level_reached histogram:");
    let mut keys: Vec<&usize> = level_reached_hist.keys().collect();
    keys.sort();
    for k in keys {
        println!("    depth {}: {} descents", k, level_reached_hist[k]);
    }

    println!("  per-level feasible cell counts (across ALL {} descents):", count);
    for (lvl, cs) in per_level_cells.iter().enumerate() {
        if cs.is_empty() { continue; }
        let mut sorted = cs.clone(); sorted.sort();
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let p50 = sorted[sorted.len() / 2];
        let mean = cs.iter().sum::<usize>() as f64 / cs.len() as f64;
        println!("    lvl {} (v{}): min={} p50={} max={} mean={:.1}",
                 lvl, lvl + 4, min, p50, max, mean);
    }

    if committed_per_level.iter().any(|v| v.iter().any(|&x| x > 0)) {
        println!("  faces committed at each level (min/p50/max/mean, n):");
        for (lvl, cs) in committed_per_level.iter().enumerate() {
            if cs.is_empty() { continue; }
            let mut sorted = cs.clone(); sorted.sort();
            let min = sorted[0];
            let max = sorted[sorted.len() - 1];
            let p50 = sorted[sorted.len() / 2];
            let mean = cs.iter().sum::<usize>() as f64 / cs.len() as f64;
            println!("    lvl {} (v{}): min={} p50={} max={} mean={:.2}",
                     lvl, lvl + 4, min, p50, max, mean);
        }
    }

    if !clean_counts.is_empty() {
        let mut sc = clean_counts.clone(); sc.sort();
        println!("  clean-triangle counts at leaves ({} depth-{} descents):", clean_counts.len(), target);
        println!("    min={}  p10={}  p25={}  p50={}  p75={}  p90={}  max={}",
                 sc[0], sc[sc.len()/10], sc[sc.len()/4], sc[sc.len()/2],
                 sc[sc.len()*3/4], sc[sc.len()*9/10], sc[sc.len()-1]);
        // Histogram bucketed by 10.
        let mut hist: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();
        for &v in &clean_counts {
            let bucket = (v / 10) * 10;
            *hist.entry(bucket).or_insert(0) += 1;
        }
        println!("  clean-count histogram (10-wide buckets):");
        for (bucket, n) in hist {
            let bar = "#".repeat((n as usize).min(50));
            println!("    {:3}-{:3}: {:4}  {}", bucket, bucket + 9, n, bar);
        }
    }
}

// ============================================================================
// Backtracking random search with prune
// ============================================================================

struct LevelState {
    committed_pushed: usize, // face commits added when placing the vertex at this level
    cells_to_try: Vec<Vec3>, // remaining feasible cells to try at this level
}

fn shuffle_vec<T>(v: &mut Vec<T>, rng: &mut Xorshift) {
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        v.swap(i, j);
    }
}

fn feasible_cells_at(
    verts: &[Vec3], committed: &[[u32; 3]], box_size: f64, mem_limit: Option<u64>,
) -> Option<Vec<Vec3>> {
    let arr = build_arrangement(verts, box_size, mem_limit)?;
    let mut feasible = Vec::new();
    for cell in &arr.cells {
        let c = cell.centroid();
        if cell_is_feasible(&c, verts, committed) { feasible.push(c); }
    }
    Some(feasible)
}

fn backtrack_search(
    seed: u64, target: usize, box_size: f64, commit_until: usize,
    deadline_secs: f64, mem_limit_bytes: u64,
) {
    let t0 = Instant::now();
    let mut rng = Xorshift::new(seed);
    let mut verts = regular_tetrahedron();
    let mut committed: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3]];
    let mut edge_fc = EdgeFaceCount::new();
    for &f in &committed { edge_fc.inc_face(f); }

    let mut stack: Vec<LevelState> = Vec::new();
    let mut visits: u64 = 0;          // cells picked (vertex placements attempted)
    let mut prunes: u64 = 0;          // combinatorial prune aborts
    let mut cap_hits: u64 = 0;        // arrangement cap aborts
    let mut target_reached: u64 = 0;  // times depth target reached
    let mut polys_found: u64 = 0;
    let mut best_clean: usize = 0;
    let mut best_poly: Option<Vec<[u32; 3]>> = None;
    let mut depth_hist: std::collections::BTreeMap<usize, u64> =
        std::collections::BTreeMap::new();
    let mut last_report = t0;

    // Prime the search: level-0 feasible cells.
    match feasible_cells_at(&verts, &committed, box_size, Some(mem_limit_bytes)) {
        None => { println!("  [fatal] level-0 arrangement cap hit"); return; }
        Some(mut cells) => {
            shuffle_vec(&mut cells, &mut rng);
            stack.push(LevelState { committed_pushed: 0, cells_to_try: cells });
        }
    }

    let mut stop_reason = "search exhausted";

    loop {
        // Deadline & memory checks.
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > deadline_secs { stop_reason = "time limit"; break; }
        let rss = current_rss_bytes();
        if rss > mem_limit_bytes { stop_reason = "memory limit"; break; }

        if last_report.elapsed().as_secs_f64() >= 2.0 {
            println!(
                "  t={:>6.1}s  rss={:>5.2}GB  depth={:>2}  stack={:>2}  visits={:>8}  prunes={:>7}  caps={:>5}  tgt={:>5}  polys={}  best_clean={}",
                elapsed, rss as f64 / (1024.0 * 1024.0 * 1024.0),
                verts.len(), stack.len(), visits, prunes, cap_hits,
                target_reached, polys_found, best_clean,
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
            if !stack.is_empty() {
                // The previous level's vertex came from stack parent; pop it now
                // (root tetrahedron vertices are never popped).
                if verts.len() > 4 { verts.pop(); }
            } else {
                if verts.len() > 4 { verts.pop(); }
            }
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
            shuffle_vec(&mut potential, &mut rng);
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

        // Combinatorial prune.
        if cannot_reach_polyhedron(&verts, target) {
            prunes += 1;
            *depth_hist.entry(verts.len()).or_insert(0) += 1;
            for _ in 0..committed_this_level {
                let f = committed.pop().unwrap();
                edge_fc.dec_face(f);
            }
            verts.pop();
            continue;
        }

        // Leaf: extract.
        if verts.len() == target {
            target_reached += 1;
            let (n_clean, poly) = extract_polyhedron(&verts, &committed);
            if n_clean > best_clean { best_clean = n_clean; }
            if let Some(p) = poly {
                polys_found += 1;
                if best_poly.is_none() { best_poly = Some(p.clone()); }
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

        // Descend.
        match feasible_cells_at(&verts, &committed, box_size, Some(mem_limit_bytes)) {
            None => {
                cap_hits += 1;
                *depth_hist.entry(verts.len()).or_insert(0) += 1;
                for _ in 0..committed_this_level {
                    let f = committed.pop().unwrap();
                    edge_fc.dec_face(f);
                }
                verts.pop();
            }
            Some(mut cells) => {
                shuffle_vec(&mut cells, &mut rng);
                stack.push(LevelState { committed_pushed: committed_this_level, cells_to_try: cells });
            }
        }
    }

    let dt = t0.elapsed().as_secs_f64();
    let rss = current_rss_bytes();
    println!("\n=== Backtrack done ({}) ===", stop_reason);
    println!("  elapsed: {:.2}s  peak_rss: {:.2}GB", dt, rss as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  visits (cells picked):      {}", visits);
    println!("  depth target reached:       {}", target_reached);
    println!("  combinatorial prunes:       {}", prunes);
    println!("  arrangement cap aborts:     {}", cap_hits);
    println!("  polyhedra found:            {}", polys_found);
    println!("  best clean-tri count seen:  {}", best_clean);
    println!("  depth-at-abort histogram (prune + cap):");
    for (k, v) in depth_hist.iter() {
        println!("    depth {:2}: {} aborts", k, v);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Backtrack mode: cells backtrack <seed> <target> <box> <commit_until> [time_secs] [mem_gb]
    if args.get(1).map(|s| s.as_str()) == Some("backtrack") {
        let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12345);
        let target: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);
        let box_size: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6.0);
        let commit_until: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
        let time_secs: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(300.0);
        let mem_gb: f64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(8.0);
        let mem_limit = (mem_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        println!("Backtrack: seed={} target={} box={} commit_until={} time={}s mem_limit={}GB",
                 seed, target, box_size, commit_until, time_secs, mem_gb);
        backtrack_search(seed, target, box_size, commit_until, time_secs, mem_limit);
        return;
    }
    // Random mode: cells random <count> [seed] [target] [box] [commit_until] [prune]
    if args.get(1).map(|s| s.as_str()) == Some("random") {
        let count: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
        let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12345);
        let target: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(12);
        let box_size: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(6.0);
        let commit_until: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
        let use_prune: bool = args.get(7).map(|s| s == "prune" || s == "1" || s == "true").unwrap_or(false);
        println!("Random descent: count={}, seed_base={}, target={}, box={}, commit_until={}, prune={}",
                 count, seed, target, box_size, commit_until, use_prune);
        run_random(count, seed, target, box_size, commit_until, use_prune);
        return;
    }

    let target = args.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(6);
    let commit_until = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(target);
    let box_size: f64 = args.get(3).and_then(|s| s.parse::<f64>().ok()).unwrap_or(6.0);
    let first_only = args.get(4).map(|s| s == "first").unwrap_or(false);

    let start = Instant::now();
    let mut verts = regular_tetrahedron();
    let mut committed_faces: Vec<[u32; 3]> = vec![[0, 1, 2], [0, 1, 3]];
    let mut edge_face_count = EdgeFaceCount::new();
    for &f in &committed_faces { edge_face_count.inc_face(f); }
    let mut stats = Stats::new();
    let opts = Opts { target, commit_until, box_size, first_only };

    dfs(&mut verts, &mut committed_faces, &mut edge_face_count, &opts, &mut stats);
    let elapsed = start.elapsed();

    println!("=== DONE in {:.3}s  (target={}, commit_until={}, box={}, first_only={}) ===",
             elapsed.as_secs_f64(), target, commit_until, box_size, first_only);
    println!("  nodes_expanded = {}", stats.nodes_expanded);
    println!("  complete configs = {}", stats.complete);
    println!("  best clean = {}", stats.best_clean);
    println!("  polyhedra found = {}", stats.poly_found);
    if let Some(ref f) = stats.best_poly {
        println!("  first polyhedron: {} faces", f.len());
    }
    for (lvl, cs) in stats.cells_at_level.iter().enumerate() {
        if cs.is_empty() { continue; }
        let total: usize = cs.iter().sum();
        let min = *cs.iter().min().unwrap();
        let max = *cs.iter().max().unwrap();
        let mean = total as f64 / cs.len() as f64;
        println!(
            "  cells lvl {} (v{}): visits={} min={} max={} mean={:.1} total={}",
            lvl, lvl + 4, cs.len(), min, max, mean, total
        );
    }
    for (lvl, ss) in stats.face_subsets_at_level.iter().enumerate() {
        if ss.is_empty() { continue; }
        let total: usize = ss.iter().sum();
        let min = *ss.iter().min().unwrap();
        let max = *ss.iter().max().unwrap();
        let mean = total as f64 / ss.len() as f64;
        println!(
            "  subsets lvl {} (v{}): cells={} min={} max={} mean={:.2} total={}",
            lvl, lvl + 4, ss.len(), min, max, mean, total
        );
    }
}
