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
mod piercing;
mod split;
mod tet;
mod zone;

use arrangement::Arrangement;

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
        let stats = enumerate::enumerate(target, time_limit, max_paths);
        println!();
        println!("  total leaves:       {}", stats.total_leaves);
        println!("  paths completed:    {}", stats.paths_completed);
        println!("  LP failures:        {}", stats.lp_failures);
        println!("  best clean count:   {}", stats.best_clean_count);
        println!("  polyhedra found:    {}", stats.polyhedra_found);
        if stats.polyhedra_found > 0 {
            println!();
            println!("  first polyhedron:");
            let p = &stats.polyhedra[0];
            println!("    path:        {:?}", p.path);
            println!("    clean count: {}", p.clean_count);
            println!("    faces ({}):", p.faces.len());
            for f in &p.faces {
                println!("      [{}, {}, {}]", f[0], f[1], f[2]);
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
