//! cells4: purely combinatorial arrangement search.
//!
//! See DESIGN.md for the full design.  Phase A (scaffolding + tet
//! hardcode + combinatorial piercing + coord test harness) is in this
//! commit.  The combinatorial cell-split step is the next milestone.

mod arrangement;
mod cell;
mod chirotope;
mod coords;
mod piercing;
mod tet;

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

    println!("cells4 — purely combinatorial arrangement search (scaffold)");
    println!("  usage: cells4 smoke           # print tetrahedron summary");
    println!("         cargo test --bin cells4 # run unit tests");
}
