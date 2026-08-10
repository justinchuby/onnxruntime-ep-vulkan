//! `ONNXRUNTIME_EP_VULKAN_GEMV_TILE`, exercised through the process environment and the **live
//! registry row**.
//!
//! Why this file exists rather than a unit test in `src/ops/quant.rs`
//! ------------------------------------------------------------------
//! Two independent reasons, and they are different reasons:
//!
//! 1. `tests/layering.rs` forbids `unsafe` anywhere under `src/ops/`, and mutating process
//!    environment on current Rust requires it. Same constraint that put
//!    `rust/tests/row_tile_fallback.rs` here.
//! 2. More importantly, the failure this file is built to catch is a **disconnected instrument**:
//!    a parser and a legality function that are individually perfect and never consulted by
//!    anything that dispatches. Every unit test in `ops::quant` passes the request in as an
//!    argument, so all of them stay green if the env read is deleted, if the registry row is
//!    pointed at a different handler, or if the specialisation vector stops carrying the tile.
//!    This test starts from `registry::spec_for` — the same lookup the EP itself performs — and
//!    ends at the specialisation constants a pipeline would be built from.
//!
//! The readback contract (issue #81)
//! ---------------------------------
//! `counters::record_pipeline_variant` is the shipping witness for *which pipeline ran*, and its
//! input is the `(shader stem, spec constants)` pair. So this test feeds the vector it recovered
//! from the handler into that function and asserts the variant string changes between arms: that
//! is the artifact an A/B is read out of, and if the request could not move it, the A/B would be
//! unreportable no matter how correct the selector was.
//!
//! `record_pipeline_variant` alone is not sufficient, which is the other half of the contract. A
//! requested `(16,2)` and an automatic `(16,2)` are the same pipeline and produce the *same*
//! variant string, and a refused request produces no variant string at all. Those two facts are
//! what `counters::gemv_tile_selections` and `counters::gemv_tile_refusal_forms` exist to record,
//! and both are asserted here.

use onnxruntime_vulkan_ep::counters;
use onnxruntime_vulkan_ep::engine::{
    AttrValue, BufferView, DType, DispatchContext, EpError, EpResult, KernelRequest, NodeDesc,
    OutRef, TensorDesc, TensorRef,
};
use onnxruntime_vulkan_ep::registry;

const VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";

/// One Phi-3.5 projection shape at a 128-token prefill: `K = N = 3072`, q4, block 32, fp16
/// activations. The shape matters — `M = 128` is `M mod 4 == 0`, which is exactly the tie case
/// where `(16,2)` and `(8,4)` name the same bytes and the byte model therefore keeps its
/// incumbent.
const K: i64 = 3072;
const N: i64 = 3072;
const M: i64 = 128;

/// Index of `QB_COLS` and `QB_ROWS` in the GEMV specialisation vector
/// `[wg, bits, block_size, has_zp, cols, packed, rows]`.
const COLS_INDEX: usize = 4;
const ROWS_INDEX: usize = 6;

#[derive(Default)]
struct Recorder {
    next: u64,
    temps: usize,
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
        self.temps += 1;
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

fn phi35_node() -> NodeDesc {
    let blocks = K / 32;
    let input = |name: &str, dtype: DType, shape: Vec<i64>| TensorRef {
        name: name.to_string(),
        desc: Some(TensorDesc::new(dtype, shape)),
        is_initializer: name != "A",
    };
    NodeDesc {
        op_type: "MatMulNBits".to_string(),
        name: "phi35_qkv_proj".to_string(),
        domain: "com.microsoft".to_string(),
        since_version: 1,
        attributes: [
            ("K".to_string(), AttrValue::Int(K)),
            ("N".to_string(), AttrValue::Int(N)),
            ("bits".to_string(), AttrValue::Int(4)),
            ("block_size".to_string(), AttrValue::Int(32)),
        ]
        .into_iter()
        .collect(),
        inputs: vec![
            input("A", DType::F16, vec![1, M, K]),
            input("B", DType::U8, vec![N, blocks, 16]),
            input("scales", DType::F16, vec![N * blocks]),
        ],
        outputs: vec![OutRef {
            name: "Y".to_string(),
            desc: Some(TensorDesc::new(DType::F16, vec![1, M, N])),
        }],
    }
}

/// Translate the node through the **registry lookup the EP itself uses**, not through a handler
/// named directly. A row that stopped resolving, or that resolved to a different handler, is a
/// live defect and this is where it surfaces.
fn translate(rec: &mut Recorder) -> EpResult<()> {
    let node = phi35_node();
    let spec = registry::spec_for(&node).expect("the MatMulNBits row must resolve from the graph");
    assert_eq!(spec.op_type, "MatMulNBits");
    (spec.translate)(spec, &node, rec)
}

fn tile_of(rec: &Recorder) -> (u32, u32) {
    assert_eq!(
        rec.dispatches.len(),
        1,
        "the GEMV is one dispatch; got {}",
        rec.dispatches.len()
    );
    let sc = &rec.dispatches[0].spec_constants;
    assert!(
        sc.len() > ROWS_INDEX,
        "the GEMV specialisation vector must carry cols and rows; got {sc:?}"
    );
    (sc[COLS_INDEX], sc[ROWS_INDEX])
}

/// The variant string the shipping witness would record for this dispatch, obtained by feeding it
/// the pair the handler actually produced.
fn record_variant(rec: &Recorder) -> String {
    let k = &rec.dispatches[0];
    counters::record_pipeline_variant(k.shader, &k.spec_constants);
    format!(
        "{}:{}",
        k.shader,
        k.spec_constants
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Everything in one test, deliberately, for the reason `row_tile_fallback.rs` gives: `cargo test`
/// runs a file's tests on several threads and the environment is process-wide, so two tests that
/// both write this variable would race and the failure would be intermittent.
#[test]
fn the_tile_request_reaches_the_live_dispatch_and_is_readable_back_from_it() {
    counters::reset();
    // SAFETY: this is the only test in this binary and the only writer of this variable in the
    // process; the value is removed again before the test returns.
    unsafe { std::env::remove_var(VAR) };

    // ── Polarity 1: unset is the control arm. ──────────────────────────────────────────────────
    let mut rec = Recorder::default();
    translate(&mut rec).expect("the control arm must translate");
    let control_tile = tile_of(&rec);
    assert_eq!(
        control_tile,
        (16, 2),
        "with no request the shipping selector must choose the wide tile at a {M}-row prefill"
    );
    let control_variant = record_variant(&rec);

    assert_eq!(
        counters::gemv_tile_request_counts(),
        (0, 0),
        "an unset variable is not a request and must not be counted as one"
    );
    assert_eq!(
        counters::gemv_tile_selections().expect("the selection log must not be poisoned"),
        vec!["AUTOMATIC cols=16 rows=2".to_string()],
        "the control arm must be recorded as AUTOMATIC, not merely as a tile"
    );

    // ── Polarity 2: a legal request reaches the pipeline. ──────────────────────────────────────
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "8,4") };
    let mut rec = Recorder::default();
    translate(&mut rec).expect("(8,4) is legal on this shape");
    let requested_tile = tile_of(&rec);
    assert_eq!(
        requested_tile,
        (8, 4),
        "the request must reach the specialisation constants the pipeline is built from — this is \
         the arm the byte model can never select, so if it does not arrive here it cannot be \
         measured at all"
    );
    let requested_variant = record_variant(&rec);
    assert_ne!(
        control_variant, requested_variant,
        "the two arms must produce different variant strings, or the artifact an A/B is read out \
         of cannot tell them apart:\n  control:   {control_variant}\n  requested: {requested_variant}"
    );
    assert!(
        counters::pipeline_variants().contains(&requested_variant),
        "the shipping readback witness must carry the requested arm"
    );
    assert_eq!(
        counters::gemv_tile_request_counts(),
        (1, 0),
        "an honoured request is counted honoured and nothing is refused"
    );

    // ── The distinguishability contract. ───────────────────────────────────────────────────────
    // A *requested* (16,2) builds the identical pipeline to the automatic one, so the variant
    // string cannot separate them. The selection log must.
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "16,2") };
    let mut rec = Recorder::default();
    translate(&mut rec).expect("(16,2) is legal on this shape");
    assert_eq!(tile_of(&rec), control_tile);
    assert_eq!(
        record_variant(&rec),
        control_variant,
        "same tile, same pipeline, same variant string — by construction"
    );
    let selections = counters::gemv_tile_selections().expect("not poisoned");
    assert!(
        selections.contains(&"AUTOMATIC cols=16 rows=2".to_string())
            && selections.contains(&"REQUESTED cols=16 rows=2".to_string()),
        "the same tile chosen by two different surfaces must be two distinguishable rows, else a \
         request that was silently ignored would read exactly like one that was honoured: {selections:?}"
    );

    // ── Polarity 3: refusals. Nothing is dispatched and the reason is typed. ───────────────────
    for (value, token) in [
        (" 8,4", "MALFORMED"),
        ("8,4,2", "MALFORMED"),
        ("08,4", "MALFORMED"),
        ("0,4", "MALFORMED"),
        ("nonsense", "MALFORMED"),
        ("16,4", "ILLEGAL"),
        ("32,1", "ILLEGAL"),
        ("3,1", "ILLEGAL"),
        ("8,8", "ILLEGAL"),
    ] {
        // SAFETY: as above.
        unsafe { std::env::set_var(VAR, value) };
        let mut rec = Recorder::default();
        let err = translate(&mut rec).expect_err("an unusable request must refuse");
        assert!(
            matches!(err, EpError::Unsupported(_)),
            "{VAR}={value:?} refused with {err:?}; a refusal of an operator's value is \
             Unsupported, not Internal"
        );
        let text = format!("{err:?}");
        assert!(
            text.contains(token),
            "{VAR}={value:?} must refuse as {token}, and the reason must survive into the error \
             an operator reads: {text}"
        );
        assert!(
            rec.dispatches.is_empty() && rec.temps == 0,
            "{VAR}={value:?} refused only after touching the queue: {} dispatch(es), {} temp(s). \
             A refusal that happens after work has been submitted is not a refusal.",
            rec.dispatches.len(),
            rec.temps
        );
    }

    let (honoured, refused) = counters::gemv_tile_request_counts();
    assert_eq!(honoured, 2, "the two legal requests");
    assert_eq!(refused, 9, "the nine unusable ones");
    let forms = counters::gemv_tile_refusal_forms().expect("not poisoned");
    assert!(
        forms.iter().any(|f| f.starts_with("MALFORMED"))
            && forms.iter().any(|f| f.starts_with("ILLEGAL")),
        "the two kinds of refusal must be distinguishable in the artifact — 'you typed it wrong' \
         and 'that cannot run on this shape' are different instructions to the operator: {forms:?}"
    );
    assert!(
        forms.len() >= 5,
        "distinct refusal reasons must not collapse into one row: {forms:?}"
    );

    // ── The control arm is restored, not latched. ──────────────────────────────────────────────
    // SAFETY: as above.
    unsafe { std::env::remove_var(VAR) };
    let mut rec = Recorder::default();
    translate(&mut rec).expect("removing the request must restore the control arm");
    assert_eq!(
        tile_of(&rec),
        control_tile,
        "removing the override must restore the shipping selector, not leave the last value latched"
    );

    counters::reset();
    assert_eq!(counters::gemv_tile_request_counts(), (0, 0));
}
