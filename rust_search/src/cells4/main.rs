//! cells4: purely combinatorial arrangement search.
//!
//! See DESIGN.md for the full design.  Phase A (scaffolding + tet
//! hardcode + combinatorial piercing + coord test harness) is in this
//! commit.  The combinatorial cell-split step is the next milestone.

mod arrangement;
mod cell;
mod chirotope;
mod coords;
mod enumerate;
mod gp;
mod piercing;
mod split;
mod tet;
mod validate;
mod zone;

use arrangement::Arrangement;

fn write_obj(p: &enumerate::PolyResult, path: &str) -> std::io::Result<()> {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(&mut s, "# cells4 — found Csaszar polyhedron candidate").ok();
    writeln!(&mut s, "# clean count: {}", p.clean_count).ok();
    writeln!(&mut s, "# path:        {:?}", p.path).ok();
    for c in &p.placed_coords {
        writeln!(&mut s, "v {:.6} {:.6} {:.6}", c[0], c[1], c[2]).ok();
    }
    for f in &p.faces {
        // OBJ is 1-indexed.
        writeln!(&mut s, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1).ok();
    }
    std::fs::write(path, s)
}

fn serde_json_encode(polys: &[enumerate::PolyResult]) -> Result<String, std::fmt::Error> {
    use std::fmt::Write;
    let mut out = String::from("[\n");
    for (pi, p) in polys.iter().enumerate() {
        write!(out, "  {{\n    \"path\": {:?},\n    \"clean_count\": {},\n    \"coords\": [",
               p.path, p.clean_count)?;
        for (i, c) in p.placed_coords.iter().enumerate() {
            if i > 0 { out.push_str(", "); }
            write!(out, "[{:.6}, {:.6}, {:.6}]", c[0], c[1], c[2])?;
        }
        out.push_str("],\n    \"faces\": [");
        for (i, f) in p.faces.iter().enumerate() {
            if i > 0 { out.push_str(", "); }
            write!(out, "[{}, {}, {}]", f[0], f[1], f[2])?;
        }
        out.push_str("]\n  }");
        if pi + 1 < polys.len() { out.push_str(","); }
        out.push('\n');
    }
    out.push_str("]\n");
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some("smoke") {
        let arr = Arrangement::new_tetrahedron();
        println!("cells4 smoke test");
        println!("  tet placed:  {}", arr.n_placed);
        println!("  tet planes:  {}", arr.planes.len());
        println!("  tet cells:   {}", arr.n_cells());
        println!("  χ entries:   {}", arr.chi.n_entries());
        println!();
        println!("  cells by label:");
        for c in &arr.cells {
            let signs: Vec<char> = (0..4).map(|i| if c.sign.get(i) > 0 { '+' } else { '-' }).collect();
            println!("    {:18}  σ = ({},{},{},{})",
                     c.label, signs[0], signs[1], signs[2], signs[3]);
        }
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("enumerate") {
        let target: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
        let time_limit: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(60.0);
        let max_paths: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
        println!("cells4 enumerate: target=N={} time_limit={}s max_paths={}",
                 target, time_limit, max_paths);
        let parallel = args.iter().any(|a| a == "--parallel");
        let stats = if parallel {
            println!("  (parallel mode: 8 threads, 1 per initial tet cell subset)");
            enumerate::enumerate_parallel(target, time_limit, max_paths / 8)
        } else {
            enumerate::enumerate(target, time_limit, max_paths)
        };
        println!();
        println!("  total leaves:       {}", stats.total_leaves);
        println!("  paths completed:    {}", stats.paths_completed);
        println!("  LP failures:        {}", stats.lp_failures);
        println!("  combinatorial prunes: {}", stats.prunes);
        println!("  best clean count:   {}", stats.best_clean_count);
        println!("  polyhedra found:    {}", stats.polyhedra_found);
        let distinct = stats.distinct_polyhedra();
        println!("  distinct (by face-set): {}", distinct.len());
        if !distinct.is_empty() {
            println!();
            println!("  first distinct polyhedron:");
            let p = &distinct[0];
            println!("    path:        {:?}", p.path);
            println!("    clean count: {}", p.clean_count);
            println!("    faces ({}):", p.faces.len());
            for f in &p.faces {
                println!("      [{}, {}, {}]", f[0], f[1], f[2]);
            }
            println!();
            println!("  coords of first distinct polyhedron:");
            for (i, c) in p.placed_coords.iter().enumerate() {
                println!("    v{}: [{:.6}, {:.6}, {:.6}]", i, c[0], c[1], c[2]);
            }
            // Dump all distinct polys to a json file for later.
            if let Ok(json) = serde_json_encode(&distinct) {
                let _ = std::fs::write("/tmp/cells4_polys.json", json);
                println!();
                println!("  wrote /tmp/cells4_polys.json ({} polyhedra)", distinct.len());
            }
            // First distinct also gets an OBJ for quick viewing.
            if write_obj(&distinct[0], "/tmp/cells4_first.obj").is_ok() {
                println!("  wrote /tmp/cells4_first.obj (vertices + faces)");
            }
            // Topology check on the first distinct.
            let topo = validate::topology(&distinct[0].faces);
            println!();
            println!("  topology of first distinct polyhedron:");
            println!("    V = {}, E = {}, F = {}", topo.vertices, topo.edges, topo.faces);
            println!("    Euler characteristic: {}", topo.euler);
            println!("    genus: {:?}", topo.genus);
            println!("    every edge in 2 faces: {}", topo.all_edges_covered_twice);
            let degrees: std::collections::BTreeMap<_,_> = topo.faces_per_vertex.iter().collect();
            let all_six = degrees.values().all(|&&c| c == 6);
            println!("    every vertex in 6 faces: {}", all_six);

            // Cross-check: all distinct polyhedra share the same topology.
            let mut by_genus: std::collections::BTreeMap<Option<i64>, u32> =
                std::collections::BTreeMap::new();
            for p in &distinct {
                let t = validate::topology(&p.faces);
                *by_genus.entry(t.genus).or_insert(0) += 1;
            }
            println!();
            println!("  all distinct polyhedra — genus distribution:");
            for (g, c) in &by_genus {
                println!("    genus {:?}: {} polyhedra", g, c);
            }
        }
        return;
    }

    println!("cells4 — purely combinatorial arrangement search");
    println!("  usage:");
    println!("    cells4 smoke                              # print tet summary");
    println!("    cells4 enumerate [N] [time_s] [max_paths] # DFS over N=7 paths");
    println!("    cargo test --bin cells4                    # run unit tests");
}
