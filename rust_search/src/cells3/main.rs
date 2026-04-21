//! cells3: experimental sign-vector-only arrangement with LP-based splits.
//!
//! Philosophical intent: the arrangement's cells are stored PURELY as
//! sign vectors — no persisted witness, no explicit polytope vertices.
//! Splitting a cell by a new plane is decided by an iterative-projection
//! LP (no coordinates leak into the Cell struct itself).
//!
//! Witness points for feasibility picks are generated on demand via the
//! same LP.  The oriented-matroid "purely combinatorial" split (without
//! any LP at all) is discussed in the module header of arrangement.rs
//! but not implemented here — it requires Grassmann-Plücker machinery
//! beyond a one-night exploration.
//!
//! Performance note: LP per cell per plane is O(cells × planes × iters),
//! which at N=12 is far too slow for real searches.  Use cells3 to
//! validate the combinatorial direction at small N (≤ 7) and as a
//! scaffold for a future fully-combinatorial implementation.

mod arrangement;
mod backtrack;
mod feasibility;

use backtrack::backtrack_search;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some("backtrack") {
        let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12345);
        let target: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7);
        let box_size: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3.0);
        let commit_until: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
        let time_secs: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(60.0);
        let mem_gb: f64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(4.0);
        let mem_limit = (mem_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        println!("cells3 backtrack: seed={} target={} box={} cu={} time={}s",
                 seed, target, box_size, commit_until, time_secs);
        backtrack_search(seed, target, box_size, commit_until, time_secs, mem_limit);
        return;
    }

    println!("cells3 — LP-based sign-vector arrangement (experimental)");
    println!("  usage: cells3 backtrack <seed> <target> <box> <commit_until> [time_secs] [mem_gb]");
    println!("  tests: cargo test --bin cells3");
}
