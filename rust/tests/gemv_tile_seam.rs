//! `ONNXRUNTIME_EP_VULKAN_GEMV_TILE`, driven through the *actual production seam*.
//!
//! Every other test of this mechanism (`ops::quant::tests::*`) calls `gemv_tile_choice` or
//! `gemv_tile_choice_with` directly — real coverage of the parser, the legality function, and the
//! byte-traffic mathematics, but none of it proves that `matmul_nbits_gemv`, the handler
//! `registry::spec_for` actually returns for `MatMulNBits`, calls that function at all. A
//! refactor that quietly went back to calling `gemv_tile` directly (or called
//! `gemv_tile_choice_with` with the wrong arguments) would leave every one of those unit tests
//! green, because none of them go through `registry::spec_for` or `OpSpec::translate`.
//!
//! This file closes exactly that gap: it builds a `MatMulNBits` `NodeDesc`, resolves it through
//! `registry::spec_for` the same way the engine does, and drives the resulting `translate`
//! function through a `DispatchContext` recorder — so a broken delegation between
//! `matmul_nbits_gemv` and `gemv_tile_choice` fails *here*, not only in a unit test that assumes
//! the delegation is intact.
//!
//! Why one `#[test]`: `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` and `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS`
//! are process-wide, `cargo test` runs the tests within one file on several threads, and this file
//! has no other test to race with — mirrors `tests/row_tile_fallback.rs`'s own reasoning.

use onnxruntime_vulkan_ep::counters;
use onnxruntime_vulkan_ep::engine::{
    AttrValue, BufferView, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef, TensorDesc,
    TensorRef,
};
use onnxruntime_vulkan_ep::ops::quant::{ENV_GEMV_TILE, gemv_cols, gemv_tile, gemv_workgroup};
use onnxruntime_vulkan_ep::registry;

const MAX_ROWS_VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS";

// K=3072, N=3072, q4, block 32, fp16 activations — one real Phi-3.5 projection shape, chosen
// (rather than copied from `ops::quant`'s private `phi35_shaped_node`) so this file has no
// dependency on that private test helper; integration tests cannot see it regardless.
const K: i64 = 3072;
const N: i64 = 3072;
const BLOCK_SIZE: i64 = 32;
const BITS: i64 = 4;

/// Build a `MatMulNBits` node for `M` rows of activation, in the shape ORT's GenAI exporter
/// actually emits it (`DESIGN.md` §1.2, `OP_COVERAGE.md` MatMulNBits row).
fn matmul_nbits_node(m: i64) -> NodeDesc {
    use onnxruntime_vulkan_ep::engine::DType;
    let blocks = K / BLOCK_SIZE;
    let input = |name: &str, dtype: DType, shape: Vec<i64>, is_initializer: bool| TensorRef {
        name: name.to_string(),
        desc: Some(TensorDesc::new(dtype, shape)),
        is_initializer,
    };
    NodeDesc {
        op_type: "MatMulNBits".to_string(),
        domain: "com.microsoft".to_string(),
        since_version: 1,
        name: "gemv_tile_seam_probe".to_string(),
        attributes: [
            ("K".to_string(), AttrValue::Int(K)),
            ("N".to_string(), AttrValue::Int(N)),
            ("bits".to_string(), AttrValue::Int(BITS)),
            ("block_size".to_string(), AttrValue::Int(BLOCK_SIZE)),
        ]
        .into_iter()
        .collect(),
        inputs: vec![
            input("A", DType::F16, vec![1, m, K], false),
            input("B", DType::U8, vec![N, blocks, 16], true),
            input("scales", DType::F16, vec![N * blocks], true),
        ],
        outputs: vec![OutRef {
            name: "Y".to_string(),
            desc: Some(TensorDesc::new(DType::F16, vec![1, m, N])),
        }],
    }
}

/// Records every dispatch a handler asked for; nothing else about it is exercised here.
#[derive(Debug, Default)]
struct Recorder {
    next: u64,
    dispatches: Vec<KernelRequest>,
}

impl DispatchContext for Recorder {
    fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
        self.next += 1;
        Ok(BufferView::from_raw(self.next))
    }
    fn bind_output(&mut self, _o: &OutRef, _desc: TensorDesc) -> EpResult<BufferView> {
        self.next += 1;
        Ok(BufferView::from_raw(self.next))
    }
    fn alloc_temp(&mut self, _desc: TensorDesc) -> EpResult<BufferView> {
        self.next += 1;
        Ok(BufferView::from_raw(self.next))
    }
    fn dispatch(&mut self, k: KernelRequest) -> EpResult<()> {
        self.dispatches.push(k);
        Ok(())
    }
    fn read_const_i64(&self, _r: &TensorRef) -> Option<Vec<i64>> {
        None
    }
}

/// Run `MatMulNBits::translate` for one `M`, through `registry::spec_for` exactly as the engine
/// resolves it — this is the actual production seam, not a direct call to any op-module function.
fn run(m: i64) -> EpResult<Recorder> {
    let node = matmul_nbits_node(m);
    let spec = registry::spec_for(&node).expect("MatMulNBits must have a live row");
    let mut rec = Recorder::default();
    (spec.translate)(spec, &node, &mut rec)?;
    Ok(rec)
}

/// SAFETY: this is the only test in this binary and the only writer of `ENV_GEMV_TILE` /
/// `MAX_ROWS_VAR` in the process; both are removed again before the test returns.
unsafe fn set(var: &str, value: &str) {
    unsafe { std::env::set_var(var, value) };
}
unsafe fn unset(var: &str) {
    unsafe { std::env::remove_var(var) };
}

#[test]
fn the_registry_seam_honours_gemv_tile_end_to_end() {
    // SAFETY: see the fn-level comments above.
    unsafe {
        unset(ENV_GEMV_TILE);
        unset(MAX_ROWS_VAR);
    }

    // ── 1. Unset: production must dispatch exactly what `gemv_tile` (the pre-existing, un-
    //    duplicated selector) picks for this shape — the "no second default" half of the spec. ──
    //
    // Note on `counters::pipeline_variants()` / `gemv_tile_spec_constants()`: those are populated
    // by `vk/session.rs`'s real `CompileRecorder` (`record_pipeline_variant` is called from
    // exactly one place, the concrete engine's dispatch path — `rust/src/vk/session.rs`), not by
    // the abstract `DispatchContext::dispatch` this file's bare `Recorder` implements. That
    // witness function is unit-tested directly in `counters.rs`
    // (`gemv_tile_spec_constants_reads_indices_4_and_6_distinctly_from_the_packed_field`). What
    // this file proves instead is the fact upstream of it: that `matmul_nbits_gemv` — the exact
    // handler `registry::spec_for` returns — puts the chosen `(cols, rows)` into specialisation
    // constants 4 and 6 of the `KernelRequest` it dispatches, which is what `vk/session.rs` then
    // hands to `record_pipeline_variant` unmodified in the real engine.
    counters::reset();
    let wg = gemv_workgroup((K / BLOCK_SIZE) as u64);
    let expected_unset = gemv_tile(8, N as u64, K as u64, BITS as u32, 2, wg);
    let rec = run(8).expect("unset must never refuse");
    assert_eq!(rec.dispatches.len(), 1, "one dispatch per node");
    let sc = &rec.dispatches[0].spec_constants;
    assert_eq!(
        (sc[4], sc[6]),
        expected_unset,
        "unset ENV_GEMV_TILE must dispatch the exact (cols, rows) gemv_tile itself picks"
    );
    let json = counters::snapshot().to_json();
    assert!(
        json.contains("\"gemv_tile_refusals\": 0")
            && json.contains("\"gemv_tile_refusal_reason\": null"),
        "no refusal happened; json={json}"
    );

    // ── 2. Malformed: refuses before dispatch, latches the reason, never falls back. ──
    counters::reset();
    // SAFETY: see above.
    unsafe { set(ENV_GEMV_TILE, "not-a-tile") };
    let err = run(8).expect_err("a malformed ENV_GEMV_TILE must refuse rather than dispatch");
    let msg = err.to_string();
    assert!(
        msg.contains(ENV_GEMV_TILE) && msg.contains("MatMulNBits"),
        "the error must name both the variable and the refusing node: {msg}"
    );
    let json = counters::snapshot().to_json();
    assert!(
        json.contains("\"gemv_tile_refusals\": 1"),
        "exactly one refusal must be latched; json={json}"
    );
    assert!(
        !json.contains("\"gemv_tile_refusal_reason\": null"),
        "the refusal reason must be a real string, not the absent-value sentinel; json={json}"
    );

    // ── 3. Legal override at M>1: substituted whole, and it is (8, 4) — distinct from the
    //    incumbent (16, 2) the unset search happened to pick above, per issue #81. ──
    counters::reset();
    // SAFETY: see above.
    unsafe { set(ENV_GEMV_TILE, "8,4") };
    let rec = run(8).expect("(8,4) is legal for this shape");
    let sc = &rec.dispatches[0].spec_constants;
    assert_eq!(
        (sc[4], sc[6]),
        (8, 4),
        "an explicit legal override must be substituted verbatim, not blended with the search"
    );
    assert_ne!(
        (sc[4], sc[6]),
        expected_unset,
        "(8,4) must be observably distinct from whatever the unset search picked"
    );

    // ── 4. Decode null control: the same legal (8,4) override is still set, but M=1 must take
    //    the decode geometry regardless — the override never reaches decode. ──
    counters::reset();
    let rec = run(1).expect("decode never refuses on an override it does not apply");
    let sc = &rec.dispatches[0].spec_constants;
    assert_eq!(
        sc[6], 1,
        "M=1 must always take rows=1, even with a legal (8,4) override set: {sc:?}"
    );
    assert_eq!(
        sc[4],
        gemv_cols(N as u64, wg),
        "M=1's cols must be the plain gemv_cols seed, not the override's cols"
    );

    // ── 5. MAX_ROWS-override precedence: an explicit legal GEMV_TILE=8,4 outranks
    //    GEMV_MAX_ROWS=1 (F4) — legality is judged against the *compiled* ceiling, not the
    //    other knob's environment-adjusted one. ──
    counters::reset();
    // SAFETY: see above.
    unsafe { set(MAX_ROWS_VAR, "1") };
    let rec = run(8).expect("an explicit override is not refused by a contradicting MAX_ROWS");
    let sc = &rec.dispatches[0].spec_constants;
    assert_eq!(
        (sc[4], sc[6]),
        (8, 4),
        "GEMV_TILE=8,4 must win over GEMV_MAX_ROWS=1: rows=4 must reach the dispatch, not rows=1"
    );

    // ── 6. Unset again for a good measure: removing the override with GEMV_MAX_ROWS=1 still
    //    set must fall back to the *clamped* search, not to the override's last value latching. ──
    counters::reset();
    // SAFETY: see above.
    unsafe { unset(ENV_GEMV_TILE) };
    let rec = run(8).expect("unset must never refuse");
    let sc = &rec.dispatches[0].spec_constants;
    assert_eq!(
        sc[6], 1,
        "with GEMV_MAX_ROWS=1 and ENV_GEMV_TILE unset, gemv_tile's own clamp must cap rows at 1: {sc:?}"
    );

    // SAFETY: see above.
    unsafe {
        unset(ENV_GEMV_TILE);
        unset(MAX_ROWS_VAR);
    }
}
