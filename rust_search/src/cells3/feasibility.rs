//! Combinatorial feasibility check (identical logic to cells2's).
//! See cells2/feasibility.rs for the derivation of the 4-sign pierce test.

use crate::arrangement::{sort_triple_with_parity, Arrangement, SignVec};

#[derive(Clone, Copy, Debug)]
pub struct PierceTest {
    pub pid_face: usize,
    pub pid_jab: usize,
    pub pid_jbc: usize,
    pub pid_jac: usize,
    pub s_vj_face: i8,
    pub s_vc_jab: i8,
    pub s_va_jbc: i8,
    pub s_vb_jac: i8,
}

pub fn build_pierce_tests(
    arr: &Arrangement,
    committed_faces: &[[u32; 3]],
    n_placed: usize,
) -> Result<Vec<PierceTest>, String> {
    let mut tests = Vec::new();
    for face in committed_faces {
        let (a, b, c) = (face[0], face[1], face[2]);
        let (face_canon, _) = sort_triple_with_parity(a, b, c);
        let pid_face = *arr.plane_of_triple.get(&face_canon)
            .ok_or_else(|| format!("missing plane for committed face {:?}", face_canon))?
            as usize;
        for j in 0..n_placed as u32 {
            if j == a || j == b || j == c { continue; }
            let (jab, _) = sort_triple_with_parity(j, a, b);
            let (jbc, _) = sort_triple_with_parity(j, b, c);
            let (jac, _) = sort_triple_with_parity(j, a, c);
            let pid_jab = *arr.plane_of_triple.get(&jab)
                .ok_or_else(|| format!("missing plane for triple {:?}", jab))? as usize;
            let pid_jbc = *arr.plane_of_triple.get(&jbc)
                .ok_or_else(|| format!("missing plane for triple {:?}", jbc))? as usize;
            let pid_jac = *arr.plane_of_triple.get(&jac)
                .ok_or_else(|| format!("missing plane for triple {:?}", jac))? as usize;
            tests.push(PierceTest {
                pid_face, pid_jab, pid_jbc, pid_jac,
                s_vj_face: arr.vert_sign[pid_face][j as usize],
                s_vc_jab:  arr.vert_sign[pid_jab][c as usize],
                s_va_jbc:  arr.vert_sign[pid_jbc][a as usize],
                s_vb_jac:  arr.vert_sign[pid_jac][b as usize],
            });
        }
    }
    Ok(tests)
}

#[inline]
pub fn is_feasible(sigma: &SignVec, tests: &[PierceTest]) -> bool {
    for t in tests {
        let s_face = sigma.sign(t.pid_face);
        if s_face == t.s_vj_face { continue; }
        let s_jab = sigma.sign(t.pid_jab);
        if s_jab != t.s_vc_jab { continue; }
        let s_jbc = sigma.sign(t.pid_jbc);
        if s_jbc != t.s_va_jbc { continue; }
        let s_jac = sigma.sign(t.pid_jac);
        if s_jac != t.s_vb_jac { continue; }
        return false;
    }
    true
}
