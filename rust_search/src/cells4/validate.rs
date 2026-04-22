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
    pub faces_per_vertex: HashMap<VertId, usize>,
    pub faces_per_edge: HashMap<[VertId; 2], usize>,
    pub all_edges_covered_twice: bool,
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
    let genus = if (2 - euler) % 2 == 0 { Some((2 - euler) / 2) } else { None };
    let all_twice = faces_per_edge.values().all(|&c| c == 2);
    TopoStats {
        vertices: v,
        edges: e,
        faces: f_count,
        euler,
        genus,
        faces_per_vertex,
        faces_per_edge,
        all_edges_covered_twice: all_twice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
