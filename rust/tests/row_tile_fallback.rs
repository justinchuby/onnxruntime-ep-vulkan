//! The row-tile fallback knob, exercised through the process environment.
//!
//! Why this file exists rather than a unit test in `src/ops/quant.rs`
//! ------------------------------------------------------------------
//! `tests/layering.rs` forbids `unsafe` anywhere under `src/ops/` — an op handler that needs it has
//! skipped a layer (DESIGN.md §4.2). Mutating process environment on current Rust requires
//! `unsafe`, so the *clamping* is tested purely in `ops::quant::clamp_max_rows` and the *plumbing*
//! — that `gemv_tile` actually consults `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` at all — is tested
//! here, outside that layer.
//!
//! This is not a redundant pair. A refactor that stopped reading the variable would leave every
//! unit test green, because they pass the ceiling in directly. This test is the one that fails.
//!
//! `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS=1` is the documented operational fallback for issue #7:
//! it restores the pre-tile decode geometry without a rebuild. A fallback nobody has demonstrated
//! is a paragraph, not a fallback.

// `tile()` below drives `gemv_tile_choice` — the exact function `matmul_nbits_gemv` calls — not
// `gemv_tile` directly. A prior version of this test called `gemv_tile` directly, which made it a
// dead seam relative to production: a change that broke `matmul_nbits_gemv`'s delegation to
// `gemv_tile_choice` (e.g. calling something else, or calling `gemv_tile` with the wrong
// arguments) would have left every assertion here green, because none of them touched the actual
// function production calls. Driving `gemv_tile_choice` with `ONNXRUNTIME_EP_VULKAN_GEMV_TILE`
// left unset makes this test observe the real delegation: it fails if that call is ever replaced.
use onnxruntime_vulkan_ep::ops::quant::{ENV_GEMV_TILE, gemv_tile_choice, gemv_workgroup};

const VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS";

// One Phi-3.5 projection shape: K=3072, N=3072, q4, block 32, fp16 activations.
const K: u64 = 3072;
const N: u64 = 3072;
const BITS: u32 = 4;
const A_BYTES: u64 = 2;

fn tile(m: u64) -> (u32, u32) {
    gemv_tile_choice(m, N, K, BITS, A_BYTES, gemv_workgroup(K / 32))
        .expect(
            "ONNXRUNTIME_EP_VULKAN_GEMV_TILE is unset throughout this file; gemv_tile_choice \
                 must fall through to gemv_tile verbatim and never refuse",
        )
        .0
}

/// Everything in one test, deliberately: `cargo test` runs the tests in a file on several threads
/// and the environment is process-wide, so two tests that both write this variable would race.
#[test]
fn the_environment_can_pin_the_row_tile_back_to_the_decode_geometry() {
    let wg = gemv_workgroup(K / 32);
    // SAFETY: this is the only test in this binary and it is the only writer of these variables
    // in the process; both are removed again before the test returns. `ENV_GEMV_TILE` is not this
    // test's subject — it is cleared so a stray value in the ambient environment cannot make
    // `gemv_tile_choice` refuse or substitute a tile, which would defeat the point of driving the
    // real seam here.
    unsafe { std::env::remove_var(VAR) };
    unsafe { std::env::remove_var(ENV_GEMV_TILE) };

    let default_decode = tile(1);
    let default_prefill = tile(8);
    assert_eq!(default_decode.1, 1, "M=1 must never take a row tile");
    assert!(
        default_prefill.1 > 1,
        "with no override an 8-row prefill must tile, got {default_prefill:?}"
    );

    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "1") };
    assert_eq!(
        tile(8),
        default_decode,
        "{VAR}=1 must reproduce the pre-issue-#7 geometry exactly — this is the documented \
         fallback, and it has to reach the same (cols, rows) M=1 gets"
    );
    assert_eq!(
        tile(1),
        default_decode,
        "pinning must not disturb decode either"
    );

    // The knob is a ceiling, not a request: it can only ever make the tile smaller, so no value of
    // it can produce a geometry the shader would refuse.
    for value in ["1", "2", "3", "4", "64", "0", "", "nonsense"] {
        // SAFETY: as above.
        unsafe { std::env::set_var(VAR, value) };
        let (cols, rows) = tile(8);
        assert!(
            (1..=4).contains(&rows),
            "{VAR}={value:?} produced rows={rows}"
        );
        assert!(
            cols * rows <= 32,
            "{VAR}={value:?} produced an over-budget tile {cols}x{rows}"
        );
        assert_eq!(
            N % u64::from(cols),
            0,
            "{VAR}={value:?} produced a column tile that splits N"
        );
    }

    // SAFETY: as above.
    unsafe { std::env::remove_var(VAR) };
    assert_eq!(
        tile(8),
        default_prefill,
        "removing the override must restore the default, not leave the last value latched"
    );
    let _ = wg;
}
