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
        let plane_id = self.planes.len() as u32;
        self.planes.push(plane);
        let old = std::mem::take(&mut self.cells);
        let mut new_cells: Vec<Cell> = Vec::with_capacity(old.len() * 2);
        for c in old {
            match c.split_by(&plane, plane_id, tol) {
                SplitResult::NoSplit(c) => new_cells.push(c),
                SplitResult::Split(a, b) => { new_cells.push(a); new_cells.push(b); }
            }
        }
        self.cells = new_cells;
    }
}

fn build_arrangement(verts: &[Vec3], box_size: f64) -> Arrangement {
    let mut arr = Arrangement::new(box_size);
    let n = verts.len();
    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                arr.add_plane(Plane::through(&verts[i], &verts[j], &verts[k]), 1e-9);
            }
        }
    }
    arr
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

// ============================================================================
// Options + statistics
// ============================================================================

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
}
impl Stats {
    fn new() -> Self { Stats {
        cells_at_level: Vec::new(), face_subsets_at_level: Vec::new(),
        complete: 0, nodes_expanded: 0,
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
    if verts.len() == opts.target { stats.complete += 1; return; }

    let lvl = (verts.len() - 4) as usize;

    let arr = build_arrangement(verts, opts.box_size);
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
    // skip this potential face
    subsets_dfs(i, idx + 1, potential, verts, committed_faces, edge_face_count,
                 subsets_at_cell, opts, stats);
    // include this potential face, if constraints pass
    let (j, k) = potential[idx];
    let face = [i, j, k];
    if edge_face_count.get(j, k) >= 2 { return; }
    if edge_face_count.get(i, j) >= 2 { return; }
    if edge_face_count.get(i, k) >= 2 { return; }
    if new_face_conflicts(face, committed_faces, verts) { return; }
    edge_face_count.inc_face(face);
    committed_faces.push(face);
    subsets_dfs(i, idx + 1, potential, verts, committed_faces, edge_face_count,
                 subsets_at_cell, opts, stats);
    committed_faces.pop();
    edge_face_count.dec_face(face);
}

// ============================================================================
// main
// ============================================================================

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
