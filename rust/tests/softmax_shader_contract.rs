//! Narrow contract tests for `shaders/glsl/softmax_f32.comp` (issue #74).
//!
//! Two halves, both deliberately host-free:
//!
//! 1. **Source contract.** The properties the Design Review fixed for this kernel are properties
//!    of its *text*: the push-constant pair, the descriptor set, the specialization constant, the
//!    shared-memory budget, the absence of every optional Vulkan feature, the barrier pairing,
//!    and the rule that no invocation leaves the workgroup between two barriers. A GPU is not
//!    required to check any of them and none of them should wait for one.
//!
//! 2. **Portable reduction model.** A faithful CPU re-execution of the kernel's four passes —
//!    including idle lanes, the fixed 256-slot shared array and the power-of-two tree — compared
//!    against a straightforward softmax oracle. This is what catches the identity bug: at
//!    `axis_size = 128` under `local_size_x = 256`, invocations 128..255 own no element, and if
//!    their shared slot is seeded with `0.0` instead of `-INF` every all-negative row silently
//!    normalises against a maximum of zero. The `wrong_identity` control below reproduces exactly
//!    that failure against the same oracle, so the passing test is not vacuous.
//!
//! Nothing here touches the proof ledger, the registry or any shared document; the host-side
//! integration (shape flattening, the power-of-two assertion, the `maxComputeWorkGroupCount[0]`
//! refusal) is Mouse's and is tested with Mouse's code.

use std::path::PathBuf;

/// The default (and maximum supported) workgroup width, mirroring `local_size_x = 256`.
const DEFAULT_LOCAL_SIZE: usize = 256;
/// Bit pattern of -INF, the identity of `max` — the same constant the shader spells out.
const NEG_INF: f32 = f32::NEG_INFINITY;

fn shader_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("glsl")
        .join("softmax_f32.comp")
}

fn shader_source() -> String {
    let p = shader_path();
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("shaders/glsl/softmax_f32.comp must be readable ({p:?}): {e}"))
}

/// Strip `//` and `/* */` comments.
///
/// Every "this shader does not use X" assertion below has to run against code only: the header of
/// the shader explains at length *why* it uses no subgroup operation and no 64-bit integer, and a
/// naive substring search over the raw file would fail on the explanation rather than on the code.
fn code_only(src: &str) -> String {
    let bytes: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
            while i < bytes.len() && bytes[i] != '\n' {
                i += 1;
            }
        } else if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == '*' && bytes[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

// ── 1. Source contract ───────────────────────────────────────────────────────────────────────

#[test]
fn the_shader_exists_where_the_build_looks_for_it() {
    // `build.rs` compiles every `shaders/glsl/*.comp` directly; the path *is* the wiring.
    assert!(
        shader_path().is_file(),
        "shaders/glsl/softmax_f32.comp is missing, so build.rs compiles no softmax module at all"
    );
}

#[test]
fn it_declares_exactly_the_agreed_push_constants_in_order() {
    let code = code_only(&shader_source());
    let start = code
        .find("layout(push_constant)")
        .expect("the kernel must declare its own push constants");
    let end = code[start..]
        .find("} pc;")
        .map(|o| start + o)
        .expect("the push-constant block must be named `pc`");
    let block = &code[start..end];

    let row = block
        .find("row_count")
        .expect("push constant `row_count` is part of the contract");
    let axis = block
        .find("axis_size")
        .expect("push constant `axis_size` is part of the contract");
    assert!(
        row < axis,
        "push-constant order is (row_count, axis_size); the host encodes them by offset"
    );

    let members = block.matches("uint ").count();
    assert_eq!(
        members, 2,
        "the block must be exactly (uint row_count, uint axis_size) — 8 bytes, no padding word: {block}"
    );
}

#[test]
fn it_declares_the_agreed_descriptor_set() {
    let code = code_only(&shader_source());
    assert!(
        code.contains("layout(set = 0, binding = 0) readonly  buffer In")
            || code.contains("layout(set = 0, binding = 0) readonly buffer In"),
        "binding 0 of set 0 must be the readonly input storage buffer"
    );
    assert!(
        code.contains("layout(set = 0, binding = 1) writeonly buffer Out"),
        "binding 1 of set 0 must be the writeonly output storage buffer"
    );
    assert_eq!(
        code.matches("layout(set = 0, binding").count(),
        2,
        "the kernel takes exactly two descriptors; a third would change the layout Mouse builds"
    );
    assert_eq!(
        code.matches("uniform ").count(),
        1,
        "the push-constant block is the only uniform block; a UBO would be a third descriptor"
    );
}

#[test]
fn the_workgroup_width_is_specializable_and_defaults_to_256() {
    let code = code_only(&shader_source());
    assert!(
        code.contains("layout(local_size_x_id = 0, local_size_x = 256) in;"),
        "local_size_x must be specialization constant 0 with a default of 256 (the Vulkan 1.1 \
         maxComputeWorkGroupSize[0] floor)"
    );
}

#[test]
fn shared_memory_is_one_kib_and_nothing_else() {
    let code = code_only(&shader_source());
    let shared: Vec<&str> = code
        .lines()
        .filter(|l| l.trim_start().starts_with("shared "))
        .collect();
    assert_eq!(
        shared.len(),
        1,
        "exactly one shared allocation is budgeted; found {shared:?}"
    );
    assert!(
        shared[0].contains("float shmem[256]"),
        "the shared array is 256 floats: {}",
        shared[0]
    );
    let bytes = DEFAULT_LOCAL_SIZE * std::mem::size_of::<f32>();
    assert_eq!(bytes, 1024, "256 floats is 1 KiB");
    assert!(
        bytes <= 16 * 1024,
        "1 KiB must fit the Vulkan 1.1 maxComputeSharedMemorySize floor of 16 KiB"
    );
}

#[test]
fn it_uses_no_optional_vulkan_feature_and_no_extension() {
    let code = code_only(&shader_source());
    for banned in [
        "subgroup",   // GL_KHR_shader_subgroup*, not guaranteed for compute on 1.1
        "int64",      // shaderInt64
        "uint64",     //   "
        "float16_t",  // shaderFloat16 / 16-bit storage
        "#extension", // any extension at all
        "#include",   // no shared include: the digest closure is this file alone
        "atomic",     // no atomics in the reduction
        "GL_EXT",
        "GL_KHR",
        "GL_GOOGLE",
    ] {
        assert!(
            !code
                .to_ascii_lowercase()
                .contains(&banned.to_ascii_lowercase()),
            "softmax_f32.comp must stay on the Vulkan 1.1 core floor, but its code mentions `{banned}`"
        );
    }
    assert!(
        code.contains("#version 450"),
        "the shader targets GLSL 4.50 like every other kernel in this tree"
    );
}

#[test]
fn the_frozen_shared_include_is_untouched_by_this_kernel() {
    // `shaders/include/indexing.glsl` is frozen: editing it moves the source digest of every
    // module that includes it. This kernel is closed over its own text precisely so that it
    // cannot widen that blast radius.
    let code = code_only(&shader_source());
    assert!(
        !code.contains("indexing.glsl") && !code.contains("op_codes.glsl"),
        "softmax_f32.comp must not pull in a shared include"
    );
}

#[test]
fn idle_lanes_are_seeded_with_negative_infinity_not_zero() {
    let code = code_only(&shader_source());
    assert!(
        code.contains("uintBitsToFloat(0xFF800000u)"),
        "-INF must be spelled as a bit pattern, not synthesised by a division by zero"
    );
    assert!(
        code.contains("float partial_max = NEG_INF;"),
        "the max pass must seed NEG_INF so a lane that owns no element contributes the identity \
         of `max`; seeding 0.0 makes every all-negative row normalise against a phantom maximum"
    );
    assert!(
        code.contains("float partial_sum = 0.0;"),
        "the sum pass must seed 0.0, the identity of `+`"
    );
}

#[test]
fn every_barrier_is_paired_with_a_shared_memory_barrier() {
    let code = code_only(&shader_source());
    let stmts: Vec<String> = code
        .split(';')
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let mut pairs = 0;
    for (i, s) in stmts.iter().enumerate() {
        if s.ends_with("barrier()") && !s.ends_with("memoryBarrierShared()") {
            let next = stmts.get(i + 1).map(String::as_str).unwrap_or("");
            assert!(
                next.ends_with("memoryBarrierShared()"),
                "barrier() #{i} is not followed by memoryBarrierShared(); found `{next}`"
            );
            pairs += 1;
        }
    }
    // Two reductions (each: seed barrier + in-loop barrier) plus the read-before-reuse barrier
    // that separates them: five pairs, and the fifth is the load-bearing one.
    assert_eq!(
        pairs, 5,
        "expected five barrier/memoryBarrierShared pairs: seed+loop for the max reduction, the \
         read-before-reuse barrier, and seed+loop for the sum reduction"
    );
}

#[test]
fn shared_memory_is_not_reused_without_a_barrier_in_between() {
    let code = code_only(&shader_source());
    let read = code
        .find("float row_max = shmem[0];")
        .expect("the max reduction ends by reading shmem[0]");
    let reuse = code[read..]
        .find("shmem[tid] = partial_sum;")
        .map(|o| read + o)
        .expect("the sum reduction reuses the same shared array");
    let between = &code[read..reuse];
    assert!(
        between.contains("barrier()") && between.contains("memoryBarrierShared()"),
        "reusing shmem for the sum pass without a barrier after reading shmem[0] is a race that a \
         single-subgroup workgroup hides completely"
    );
}

#[test]
fn no_invocation_leaves_the_workgroup_between_barriers() {
    let code = code_only(&shader_source());
    let first_barrier = code
        .find("barrier()")
        .expect("the kernel contains barriers");
    let returns: Vec<usize> = code.match_indices("return").map(|(i, _)| i).collect();
    assert_eq!(
        returns.len(),
        1,
        "the only permitted early exit is the workgroup-uniform row guard"
    );
    assert!(
        returns[0] < first_barrier,
        "an early return after a barrier breaks the uniform-control-flow requirement of \
         OpControlBarrier (GLSL 4.50 §8.16)"
    );
    let guard_line = code
        .lines()
        .find(|l| l.contains("return"))
        .expect("the guard line exists");
    assert!(
        guard_line.contains("row >= pc.row_count"),
        "the guard must depend on gl_WorkGroupID alone so it is uniform across the workgroup: {guard_line}"
    );
}

#[test]
fn the_degenerate_row_paths_are_guarded() {
    let code = code_only(&shader_source());
    assert!(
        code.contains("isinf(row_max)"),
        "a fully masked row makes row_max -INF and exp(x - row_max) NaN; that must be guarded"
    );
    assert!(
        code.contains("row_sum > 0.0"),
        "the reciprocal of the row sum must be guarded against 0 (and, being a `>` test, against NaN)"
    );
    assert!(
        code.contains("uniform_p"),
        "the guarded path must produce the uniform distribution, not a zero row"
    );
}

// ── 2. Portable reduction model ──────────────────────────────────────────────────────────────

/// A faithful CPU re-execution of `softmax_f32.comp`.
///
/// `local_size` is the specialized workgroup width; `max_identity` is the value an idle lane
/// writes into the shared slot during the max pass, exposed only so that the control test can
/// inject the wrong one and show this suite would notice.
fn simulate_row(row: &[f32], local_size: usize, max_identity: f32) -> Vec<f32> {
    assert!(
        local_size.is_power_of_two() && local_size <= DEFAULT_LOCAL_SIZE,
        "the host asserts a power-of-two workgroup no wider than the shared array"
    );
    let n = row.len();

    // Pass 1 + 2: partial maxima, then the shared-memory tree.
    let mut shmem = vec![f32::NAN; DEFAULT_LOCAL_SIZE];
    for (tid, slot) in shmem.iter_mut().enumerate().take(local_size) {
        let mut partial = max_identity;
        let mut i = tid;
        while i < n {
            partial = if row[i] > partial { row[i] } else { partial };
            i += local_size;
        }
        *slot = partial;
    }
    let mut stride = local_size / 2;
    while stride > 0 {
        for tid in 0..stride {
            shmem[tid] = if shmem[tid + stride] > shmem[tid] {
                shmem[tid + stride]
            } else {
                shmem[tid]
            };
        }
        stride /= 2;
    }
    let row_max = shmem[0];
    let finite_max = row_max.is_finite();

    let exp_of = |x: f32| -> f32 {
        if finite_max {
            (x - row_max).exp()
        } else if x == row_max {
            1.0
        } else {
            0.0
        }
    };

    // Pass 3: partial sums, then the same tree with `+`.
    for (tid, slot) in shmem.iter_mut().enumerate().take(local_size) {
        let mut partial = 0.0f32;
        let mut i = tid;
        while i < n {
            partial += exp_of(row[i]);
            i += local_size;
        }
        *slot = partial;
    }
    let mut stride = local_size / 2;
    while stride > 0 {
        for tid in 0..stride {
            shmem[tid] += shmem[tid + stride];
        }
        stride /= 2;
    }
    let row_sum = shmem[0];

    // Mirrors the shader's `bool degenerate = !(row_sum > 0.0);` — the comparison is written
    // first and negated second so that NaN falls on the degenerate side, exactly as it does in
    // GLSL, where every ordered comparison against NaN is false.
    let sum_is_usable = row_sum > 0.0;
    let degenerate = !sum_is_usable;
    let inv_sum = if degenerate { 0.0 } else { 1.0 / row_sum };
    let uniform_p = if n > 0 { 1.0 / n as f32 } else { 0.0 };

    // Pass 4.
    let mut out = vec![0.0f32; n];
    for tid in 0..local_size {
        let mut i = tid;
        while i < n {
            out[i] = if degenerate {
                uniform_p
            } else {
                exp_of(row[i]) * inv_sum
            };
            i += local_size;
        }
    }
    out
}

fn simulate(row: &[f32], local_size: usize) -> Vec<f32> {
    simulate_row(row, local_size, NEG_INF)
}

/// The oracle: textbook max-subtracted softmax in f64, with the same documented rule for a
/// non-finite maximum (indicator of the maximum), which is the limit of softmax in both
/// directions and is what a fully masked row must produce.
fn oracle(row: &[f32]) -> Vec<f32> {
    if row.is_empty() {
        return Vec::new();
    }
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = if m.is_finite() {
        row.iter().map(|&x| ((x - m) as f64).exp()).collect()
    } else {
        row.iter()
            .map(|&x| if x == m { 1.0 } else { 0.0 })
            .collect()
    };
    let s: f64 = e.iter().sum();
    if s > 0.0 {
        e.iter().map(|&v| (v / s) as f32).collect()
    } else {
        vec![1.0 / row.len() as f32; row.len()]
    }
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: element {i} is {g}, oracle says {w} (tol {tol})"
        );
    }
}

/// Deterministic, dependency-free pseudo-random f32s in roughly [-8, 8).
fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 16.0 - 8.0
        })
        .collect()
}

#[test]
fn row_128_under_workgroup_256_matches_the_oracle() {
    // The configuration the Design Review called out: half the workgroup owns no element.
    let row = pseudo_random(128, 0xC0FFEE);
    assert_close(&simulate(&row, 256), &oracle(&row), 1e-6, "128 under 256");
}

#[test]
fn an_all_negative_row_under_a_wider_workgroup_would_expose_a_zero_identity() {
    // Half the workgroup owns no element at `axis_size = 128, local_size_x = 256`.
    //
    // Being precise about *how* a `0.0` idle-lane seed breaks, because the obvious story is
    // wrong: softmax is shift-invariant, so subtracting an inflated maximum is algebraically
    // harmless. What it is not is *representably* harmless. The whole point of subtracting the
    // maximum is to move the exponent range back to where f32 can hold it; seeding an idle lane
    // with `0.0` clamps `row_max` to at least zero, and a row of large negative logits — the
    // shape a masked or heavily biased attention row actually has — then exponentiates to
    // exactly zero in every lane. `row_sum` becomes 0, the degenerate guard fires, and a sharply
    // peaked distribution is returned as a flat one. Silently.
    let row: Vec<f32> = (0..128).map(|i| -500.0 + i as f32).collect();
    let want = oracle(&row);

    assert_close(
        &simulate(&row, 256),
        &want,
        1e-6,
        "negative row, -INF identity",
    );
    assert!(
        want.iter().any(|p| *p > 0.5),
        "precondition: this row's true softmax is peaked, not uniform"
    );

    let wrong = simulate_row(&row, 256, 0.0);
    let differs = wrong
        .iter()
        .zip(&want)
        .any(|(g, w)| (g - w).abs() > 1e-6 || !g.is_finite());
    assert!(
        differs,
        "control failed: seeding idle lanes with 0.0 produced the same answer, so this suite \
         would not have caught the identity bug it exists to catch"
    );
    assert!(
        wrong.iter().all(|p| (*p - 1.0 / 128.0).abs() < 1e-9),
        "control precondition: the 0.0 seed is expected to underflow the whole row to the \
         uniform fallback, got {wrong:?}"
    );
}

#[test]
fn a_fully_masked_row_is_uniform_not_merely_not_nan() {
    for n in [1usize, 7, 128, 256, 300] {
        let row = vec![f32::NEG_INFINITY; n];
        let got = simulate(&row, 256);
        let want = oracle(&row);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "a fully masked row of {n} produced a non-finite value: {got:?}"
        );
        assert_close(&got, &want, 0.0, "fully masked row");
        for v in &got {
            assert_eq!(
                *v,
                1.0 / n as f32,
                "a fully masked row of {n} must be exactly uniform 1/N"
            );
        }
        let total: f32 = got.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "a fully masked row of {n} must still sum to 1, got {total}"
        );
    }
}

#[test]
fn a_partially_masked_row_ignores_the_masked_entries() {
    let mut row = pseudo_random(128, 0x5EED);
    for (i, v) in row.iter_mut().enumerate() {
        if i % 3 == 0 {
            *v = f32::NEG_INFINITY;
        }
    }
    let got = simulate(&row, 256);
    assert_close(&got, &oracle(&row), 1e-6, "partially masked row");
    for (i, v) in got.iter().enumerate() {
        if i % 3 == 0 {
            assert_eq!(*v, 0.0, "masked position {i} must receive exactly zero");
        }
    }
}

#[test]
fn a_positive_infinity_logit_collapses_to_the_one_hot_limit() {
    let mut row = pseudo_random(64, 0xBEEF);
    row[13] = f32::INFINITY;
    let got = simulate(&row, 256);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "+INF row produced NaN: {got:?}"
    );
    assert_eq!(got[13], 1.0, "the +INF entry takes the whole mass");
    assert_close(&got, &oracle(&row), 0.0, "+INF row");
}

#[test]
fn large_magnitudes_do_not_overflow_because_the_maximum_is_subtracted() {
    for scale in [1.0e3f32, 1.0e6, 1.0e30] {
        let row: Vec<f32> = (0..200).map(|i| scale + i as f32).collect();
        let got = simulate(&row, 256);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "scale {scale} overflowed: {got:?}"
        );
        assert_close(&got, &oracle(&row), 1e-6, "large magnitudes");
    }
}

#[test]
fn every_power_of_two_workgroup_agrees_with_the_oracle() {
    // The tree reduction is only correct for a power-of-two width; the host asserts it, and this
    // sweeps the whole legal range including the degenerate width of 1.
    for lsz in [1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        for n in [1usize, 3, 31, 32, 128, 255, 256, 257, 1024] {
            let row = pseudo_random(n, (lsz as u32) << 8 | n as u32);
            let got = simulate(&row, lsz);
            assert_close(&got, &oracle(&row), 2e-6, &format!("lsz={lsz} n={n}"));
            let total: f32 = got.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-3,
                "lsz={lsz} n={n}: row must sum to 1, got {total}"
            );
        }
    }
}

#[test]
fn an_empty_axis_writes_nothing_and_divides_by_nothing() {
    let got = simulate(&[], 256);
    assert!(
        got.is_empty(),
        "axis_size == 0 must produce no output element"
    );
}

#[test]
fn a_row_longer_than_the_workgroup_is_covered_exactly_once() {
    // The strided loop must partition the row: every element written, none written twice. A
    // stride bug shows up here as a hole (0.0 left behind) rather than as a wrong sum.
    let n = 1000;
    let row = pseudo_random(n, 0xA11CE);
    let got = simulate(&row, 256);
    assert_eq!(got.len(), n);
    assert!(
        got.iter().all(|v| *v > 0.0),
        "a finite row has no zero-probability element; a hole means the grid-stride loop skipped one"
    );
    assert_close(&got, &oracle(&row), 2e-6, "n=1000 under 256");
}
