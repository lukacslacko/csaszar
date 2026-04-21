//! cells2: plane-arrangement search with sign-vector cells.
//!
//! See /Users/lukacs/.claude/plans/why-are-the-vertices-mossy-moonbeam.md
//! for the full design rationale.  Briefly:
//!
//!   - Cells are represented only by their sign vector over the current
//!     arrangement planes plus a single witness point.
//!   - Splitting by a new plane is done via the 2D arrangement induced on
//!     the cutting plane itself ("zone" method), never by tracking
//!     explicit polytope vertices.
//!   - Feasibility (does placing a new vertex cause any committed-face
//!     piercing) is a combinatorial sign lookup; no geometric predicates
//!     in the hot loop.

mod arrangement;
mod backtrack;
mod feasibility;

use backtrack::{backtrack_search, backtrack_search_verbose};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Backtrack mode:
    // cells2 backtrack <seed> <target> <box> <commit_until> [time_secs] [mem_gb]
    if args.get(1).map(|s| s.as_str()) == Some("backtrack") {
        let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12345);
        let target: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);
        let box_size: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6.0);
        let commit_until: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
        let time_secs: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(300.0);
        let mem_gb: f64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(8.0);
        let mem_limit = (mem_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        println!("cells2 backtrack: seed={} target={} box={} commit_until={} time={}s mem_limit={}GB",
                 seed, target, box_size, commit_until, time_secs, mem_gb);
        backtrack_search(seed, target, box_size, commit_until, time_secs, mem_limit);
        return;
    }

    // Verbose mode — same args, plus per-level build info.
    if args.get(1).map(|s| s.as_str()) == Some("backtrack-verbose") {
        let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12345);
        let target: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7);
        let box_size: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
        let commit_until: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
        let time_secs: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(60.0);
        let mem_gb: f64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(4.0);
        let mem_limit = (mem_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        println!("cells2 backtrack (verbose): seed={} target={} box={} cu={}",
                 seed, target, box_size, commit_until);
        backtrack_search_verbose(seed, target, box_size, commit_until,
                                   time_secs, mem_limit);
        return;
    }

    println!("cells2 — sign-vector arrangement search");
    println!("  usage: cells2 backtrack <seed> <target> <box> <commit_until> [time_secs] [mem_gb]");
    println!("  tests: cargo test --bin cells2");
}
