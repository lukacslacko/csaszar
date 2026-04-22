//! Topological sanity checks on a found polyhedron.

use crate::chirotope::VertId;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TopoStats {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub euler: i64,
    pub genus: Option<i64>,
    pub crosscaps: Option<i64>,
    pub faces_per_vertex: HashMap<VertId, usize>,
    pub faces_per_edge: HashMap<[VertId; 2], usize>,
    pub all_edges_covered_twice: bool,
    pub is_2_manifold: bool,
    pub non_manifold_vertices: Vec<VertId>,
}

/// Check if the link of vertex v is a single cycle.  For a 2-manifold
/// triangulation, this must be true at every vertex.
pub fn vertex_link_is_cycle(v: VertId, faces: &[[VertId; 3]]) -> bool {
    // Collect edges of the link (pairs of "other" vertices).
    let mut link_edges: Vec<(VertId, VertId)> = Vec::new();
    for f in faces {
        if !f.contains(&v) { continue; }
        let others: Vec<VertId> = f.iter().filter(|&&x| x != v).copied().collect();
        debug_assert_eq!(others.len(), 2);
        link_edges.push((others[0], others[1]));
    }
    if link_edges.is_empty() { return false; }
    let link_verts: HashSet<VertId> =
        link_edges.iter().flat_map(|&(a, b)| [a, b]).collect();
    // Degree of each link vertex must be 2 (for a simple cycle).
    let mut deg: HashMap<VertId, usize> = HashMap::new();
    for &(a, b) in &link_edges {
        *deg.entry(a).or_insert(0) += 1;
        *deg.entry(b).or_insert(0) += 1;
    }
    if deg.values().any(|&d| d != 2) { return false; }
    // Connectedness: BFS should reach all link vertices.
    let mut adj: HashMap<VertId, Vec<VertId>> = HashMap::new();
    for &(a, b) in &link_edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let start = *link_verts.iter().next().unwrap();
    let mut seen: HashSet<VertId> = HashSet::new();
    seen.insert(start);
    let mut stack = vec![start];
    while let Some(u) = stack.pop() {
        for &w in adj.get(&u).map(|v| v.as_slice()).unwrap_or(&[]) {
            if seen.insert(w) { stack.push(w); }
        }
    }
    seen.len() == link_verts.len()
}

pub fn topology(faces: &[[VertId; 3]]) -> TopoStats {
    let mut verts: HashSet<VertId> = HashSet::new();
    let mut edges: HashSet<[VertId; 2]> = HashSet::new();
    let mut faces_per_vertex: HashMap<VertId, usize> = HashMap::new();
    let mut faces_per_edge: HashMap<[VertId; 2], usize> = HashMap::new();
    for f in faces {
        for &v in f {
            verts.insert(v);
            *faces_per_vertex.entry(v).or_insert(0) += 1;
        }
        for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[0], f[2])] {
            let key = if u < v { [u, v] } else { [v, u] };
            edges.insert(key);
            *faces_per_edge.entry(key).or_insert(0) += 1;
        }
    }
    let v = verts.len();
    let e = edges.len();
    let f_count = faces.len();
    let euler = v as i64 - e as i64 + f_count as i64;
    // Orientable: χ = 2 - 2g ⇒ g = (2 - χ)/2
    let genus = if (2 - euler) % 2 == 0 { Some((2 - euler) / 2) } else { None };
    // Non-orientable: χ = 2 - k ⇒ k = 2 - χ  (any integer)
    let crosscaps = Some(2 - euler);
    let all_twice = faces_per_edge.values().all(|&c| c == 2);

    // 2-manifold check: every vertex link is a single cycle.
    let mut non_manifold_vertices: Vec<VertId> = Vec::new();
    for &v in &verts {
        if !vertex_link_is_cycle(v, faces) {
            non_manifold_vertices.push(v);
        }
    }
    non_manifold_vertices.sort();
    let is_manifold = all_twice && non_manifold_vertices.is_empty();

    TopoStats {
        vertices: v,
        edges: e,
        faces: f_count,
        euler,
        genus,
        crosscaps,
        faces_per_vertex,
        faces_per_edge,
        all_edges_covered_twice: all_twice,
        is_2_manifold: is_manifold,
        non_manifold_vertices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_cycle_for_csaszar_is_true() {
        let faces: Vec<[VertId; 3]> = vec![
            [0,1,2], [0,1,5], [0,2,3], [0,3,4], [0,4,6], [0,5,6],
            [1,2,4], [1,3,5], [1,3,6], [1,4,6], [2,3,6], [2,4,5],
            [2,5,6], [3,4,5],
        ];
        for v in 0..7u32 {
            assert!(vertex_link_is_cycle(v, &faces),
                    "v_{} link isn't a cycle in Csaszar", v);
        }
    }

    #[test]
    fn n9_first_found_is_pseudo_manifold() {
        // From the N=9 parallel 10-min run: this face-set satisfies
        // the 2-factor property but has a non-manifold vertex v_0
        // whose link splits into two 4-cycles {1-2, 1-8, 2-3, 3-8}
        // and {4-6, 4-7, 5-6, 5-7}.
        let faces: Vec<[VertId; 3]> = vec![
            [0,1,2], [0,1,8], [0,2,3], [0,3,8], [0,4,6], [0,4,7],
            [0,5,6], [0,5,7], [1,2,6], [1,3,4], [1,3,7], [1,4,5],
            [1,5,6], [1,7,8], [2,3,5], [2,4,7], [2,4,8], [2,5,7],
            [2,6,8], [3,4,6], [3,5,8], [3,6,7], [4,5,8], [6,7,8],
        ];
        let topo = topology(&faces);
        assert_eq!(topo.faces, 24);
        assert!(topo.all_edges_covered_twice, "2-factor property OK");
        assert!(!topo.is_2_manifold, "should fail manifold filter");
        assert!(topo.non_manifold_vertices.contains(&0),
                "v_0 should be a non-manifold vertex");
    }

    #[test]
    fn first_found_csaszar_is_genus_1() {
        // Same face list as the first distinct polyhedron we find at N=7.
        let faces: Vec<[VertId; 3]> = vec![
            [0,1,2], [0,1,5], [0,2,3], [0,3,4], [0,4,6], [0,5,6],
            [1,2,4], [1,3,5], [1,3,6], [1,4,6], [2,3,6], [2,4,5],
            [2,5,6], [3,4,5],
        ];
        let topo = topology(&faces);
        assert_eq!(topo.vertices, 7);
        assert_eq!(topo.edges, 21);
        assert_eq!(topo.faces, 14);
        assert_eq!(topo.euler, 0);
        assert_eq!(topo.genus, Some(1), "expected genus 1 (torus)");
        assert!(topo.all_edges_covered_twice);
        for (_, &count) in topo.faces_per_vertex.iter() {
            assert_eq!(count, 6, "each vertex in a K_7 torus triangulation has face-degree 6");
        }
    }
}
