//! `ONNXRUNTIME_EP_VULKAN_GEMV_TILE`, exercised through the registry and the process environment.
//!
//! Why this file exists rather than a unit test in `src/ops/quant.rs`
//! ------------------------------------------------------------------
//! Two independent reasons, and both matter.
//!
//! 1. `tests/layering.rs` forbids `unsafe` anywhere under `src/ops/`, and mutating process
//!    environment on current Rust requires it. So the *pure* halves — the parser, the legality
//!    rules, the tie theorem, the unreachability of `(8, 4)` — are unit-tested in
//!    `ops::quant::tests`, and the fact that production **reads the variable at all** is tested
//!    here.
//!
//! 2. More importantly, this file reaches the handler the way ORT does: it looks the node up in
//!    the **registry** with `registry::spec_for` and calls the row's own `translate`. A test that
//!    called `matmul_nbits_gemv` directly would prove the function works and prove nothing about
//!    whether anything calls it. Issue #81 asks for a *production-delegated* surface rather than
//!    a test-only seam, and the difference between those two is exactly this lookup.
//!
//! This is not a redundant pair with the unit tests. A refactor that stopped consulting the
//! variable, or that reached the tile without going through the registry row, would leave every
//! unit test green. These are the ones that fail.

use onnxruntime_vulkan_ep::counters;
use onnxruntime_vulkan_ep::engine::{
    AttrValue, BufferView, DType, EpResult, KernelRequest, NodeDesc, OutRef, TensorDesc, TensorRef,
};
use onnxruntime_vulkan_ep::registry;

const VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";
const CEILING_VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS";

/// Index of `QB_COLS` and `QB_ROWS` in `q_gemv.comp`'s specialisation vector.
///
/// Named rather than spelled inline so that a reordering in `ops/quant.rs` is a compile-adjacent
/// edit here too, instead of a silently misread constant.
const COLS_INDEX: usize = 4;
const ROWS_INDEX: usize = 6;

/// The smallest `DispatchContext` that can witness a dispatch: it records the kernel requests and
/// hands out distinct buffer tokens, which is all a tile decision needs to be readable.
#[derive(Default)]
struct Recorder {
    next: u64,
    dispatches: Vec<KernelRequest>,
}

impl onnxruntime_vulkan_ep::engine::DispatchContext for Recorder {
    fn resolve(&mut self, _r: &TensorRef) -> EpResult<BufferView> {
        self.next += 1;
        Ok(BufferView::from_raw(self.next))
    }
    fn bind_output(&mut self, _o: &OutRef, _d: TensorDesc) -> EpResult<BufferView> {
        self.next += 1;
        Ok(BufferView::from_raw(self.next))
    }
    fn alloc_temp(&mut self, _d: TensorDesc) -> EpResult<BufferView> {
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

/// One Phi-3.5 projection node at `M` rows: `K = N = 3072`, q4, block 32, fp16 activations.
fn node(m: i64) -> NodeDesc {
    let (k, n) = (3072_i64, 3072_i64);
    let blocks = k / 32;
    let input = |name: &str, dtype: DType, shape: Vec<i64>| TensorRef {
        name: name.to_string(),
        desc: Some(TensorDesc::new(dtype, shape)),
        is_initializer: name != "A",
    };
    NodeDesc {
        op_type: "MatMulNBits".to_string(),
        name: "phi35".to_string(),
        domain: "com.microsoft".to_string(),
        since_version: 1,
        attributes: [
            ("K".to_string(), AttrValue::Int(k)),
            ("N".to_string(), AttrValue::Int(n)),
            ("bits".to_string(), AttrValue::Int(4)),
            ("block_size".to_string(), AttrValue::Int(32)),
        ]
        .into_iter()
        .collect(),
        inputs: vec![
            input("A", DType::F16, vec![1, m, k]),
            input("B", DType::U8, vec![n, blocks, 16]),
            input("scales", DType::F16, vec![n * blocks]),
        ],
        outputs: vec![OutRef {
            name: "Y".to_string(),
            desc: Some(TensorDesc::new(DType::F16, vec![1, m, n])),
        }],
    }
}

/// Drive the node through the **registry**, not through a function this file names.
fn translate(m: i64) -> Result<(u32, u32), String> {
    let desc = node(m);
    let spec = registry::spec_for(&desc).expect("MatMulNBits must have a registry row");
    let mut rec = Recorder::default();
    match (spec.translate)(spec, &desc, &mut rec) {
        Ok(()) => {
            assert_eq!(rec.dispatches.len(), 1, "one dispatch per node");
            let spec_constants = &rec.dispatches[0].spec_constants;
            Ok((spec_constants[COLS_INDEX], spec_constants[ROWS_INDEX]))
        }
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Everything in one test, deliberately: `cargo test` runs a file's tests on several threads and
/// the environment is process-wide, so two tests that both write this variable would race. The
/// same reasoning `rust/tests/row_tile_fallback.rs` states for the row ceiling.
#[test]
fn the_tile_request_reaches_the_pipeline_and_refuses_visibly_when_it_cannot() {
    // SAFETY: this is the only test in this binary and the only writer of these variables in the
    // process; both are removed again before the test returns.
    unsafe {
        std::env::remove_var(VAR);
        std::env::remove_var(CEILING_VAR);
    }

    // ── Control: unset ────────────────────────────────────────────────────────────────────────
    // The shipping selector's answer for a Phi-3.5 prefill, and the string issue #81 reads off
    // the approved-head artifacts: `...,16,1,2`. If this ever moves, every claim in the issue
    // about what the current tree builds is stale and the A/B has lost its control arm.
    counters::reset();
    let unset_prefill = translate(32).expect("the unset control must dispatch");
    let unset_decode = translate(1).expect("the unset control must dispatch at M=1 too");
    assert_eq!(unset_prefill, (16, 2), "the approved-head prefill tile");
    assert_eq!(unset_decode, (16, 1), "M=1 must never take a row tile");
    assert_eq!(
        counters::gemv_tile_surface(),
        "SELECTED",
        "with nothing set, every decision must be the byte model's"
    );
    assert_eq!(
        counters::gemv_tile_requests(),
        (0, 0),
        "an unset variable is not a request and not a refusal"
    );
    assert!(
        counters::gemv_tile_decisions()
            .iter()
            .all(|d| d.starts_with("SELECTED")),
        "{:?}",
        counters::gemv_tile_decisions()
    );

    // ── Positive polarity: the arm the issue exists to make reachable ─────────────────────────
    counters::reset();
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "8,4") };
    assert_eq!(
        translate(32).expect("(8,4) is legal at this shape"),
        (8, 4),
        "the requested tile must reach the specialisation vector verbatim"
    );
    assert_eq!(counters::gemv_tile_surface(), "REQUESTED");
    assert_eq!(counters::gemv_tile_requests(), (1, 0));
    assert_eq!(
        counters::gemv_tile_decisions(),
        vec!["REQUESTED cols=8 rows=4".to_string()],
        "the artifact must be able to tell a requested (8,4) from a selected one"
    );

    // The equal-traffic partner, requested rather than selected. Same pipeline as the control's
    // — that is the point of the pair — but a *different* recorded decision, which is the only
    // reason an A/B between them is readable at all.
    counters::reset();
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "16,2") };
    assert_eq!(translate(32).expect("(16,2) is legal"), (16, 2));
    assert_eq!(
        counters::gemv_tile_decisions(),
        vec!["REQUESTED cols=16 rows=2".to_string()]
    );

    // ── Negative polarity: refusal before dispatch, with the reason ───────────────────────────
    // Each row is a value an operator could plausibly type, paired with the form the counters
    // must carry. A refusal that reported the wrong reason would be worse than none: it would
    // send the reader to fix the wrong thing.
    for (value, form) in [
        ("", "SYNTAX-EMPTY"),
        ("8", "SYNTAX-NOT-A-PAIR"),
        (" 8,4", "SYNTAX-NOT-DECIMAL"),
        ("8,4 ", "SYNTAX-NOT-DECIMAL"),
        ("0,4", "SYNTAX-ZERO"),
        ("4294967296,4", "SYNTAX-OVERFLOW"),
        ("32,1", "TILE-COLS-ABOVE-MAX"),
        ("8,8", "TILE-ROWS-ABOVE-MAX"),
        ("16,4", "TILE-PRODUCT-ABOVE-MAX"),
        ("3,4", "TILE-NOT-A-POWER-OF-TWO"),
    ] {
        counters::reset();
        // SAFETY: as above.
        unsafe { std::env::set_var(VAR, value) };
        let err = translate(32).expect_err(&format!("{value:?} must refuse"));
        assert!(
            err.contains(form),
            "{value:?}: the error must name the form; got {err}"
        );
        assert_eq!(
            counters::gemv_tile_requests(),
            (0, 1),
            "{value:?}: a refusal is counted as a refusal"
        );
        assert_eq!(
            counters::gemv_tile_refusal_forms(),
            vec![form.to_string()],
            "{value:?}"
        );
        assert_eq!(
            counters::gemv_tile_surface(),
            "ALL-REFUSED",
            "{value:?}: nothing was selected, and that is not the same as never reached"
        );
        assert!(
            counters::gemv_tile_decisions().is_empty(),
            "{value:?}: a refused request must never record a decision — a silent fall back to \
             the automatic tile is the exact failure this surface exists to prevent"
        );
    }

    // A row tile at decode is refused rather than honoured: `rows == 1` at one row is the geometry
    // the ledger stands behind, and this surface may not move it.
    counters::reset();
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "8,4") };
    let decode = translate(1).expect_err("a row tile at M=1 must refuse");
    assert!(decode.contains("TILE-ROW-TILE-AT-DECODE"), "{decode}");

    // ── The two surfaces disagreeing is a refusal, not a silent winner ────────────────────────
    counters::reset();
    // SAFETY: as above.
    unsafe {
        std::env::set_var(VAR, "8,4");
        std::env::set_var(CEILING_VAR, "2");
    }
    let clash = translate(32).expect_err("a request above the ceiling must refuse");
    assert!(clash.contains("TILE-ROWS-ABOVE-CEILING"), "{clash}");

    // ── Removing the request restores the control exactly ─────────────────────────────────────
    // The property that makes this instrument safe to ship: it is inert unless armed.
    counters::reset();
    // SAFETY: as above.
    unsafe {
        std::env::remove_var(VAR);
        std::env::remove_var(CEILING_VAR);
    }
    assert_eq!(
        (translate(32).unwrap(), translate(1).unwrap()),
        (unset_prefill, unset_decode),
        "unsetting the variable must return the byte-for-byte control"
    );
    assert_eq!(counters::gemv_tile_requests(), (0, 0));
}
