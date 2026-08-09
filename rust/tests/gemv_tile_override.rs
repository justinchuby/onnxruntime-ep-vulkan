//! The `ONNXRUNTIME_EP_VULKAN_GEMV_TILE` override, exercised through the **live** path.
//!
//! What "live" means here, and why the distinction is the whole point of this file
//! -------------------------------------------------------------------------------
//! `src/ops/quant.rs`'s unit tests call the selector directly. They are true and they are worth
//! having, but they prove exactly one thing: *given* that the handler asks the selector, the
//! selector answers correctly. They cannot see the handler at all — an edit that deleted the
//! handler's call and hardcoded `(16, 2)` would leave every one of them green, and would leave
//! `row_tile_fallback.rs` green too, because that file also calls `gemv_tile` directly.
//!
//! This file closes that gap by starting from `registry::spec_for` — the same lookup `Compile`
//! uses — and driving the row's own `translate` pointer with a recording `DispatchContext`. What
//! it asserts is the **specialisation-constant vector the handler actually built**, which is the
//! value a pipeline is created from. Nothing in the chain from the registry row to
//! `KernelRequest.spec_constants` is stubbed or re-derived.
//!
//! Why it is outside `src/`
//! ------------------------
//! `tests/layering.rs` forbids `unsafe` under `src/ops/`, and mutating process environment on
//! current Rust requires it. Same reason `rust/tests/row_tile_fallback.rs` exists.
//!
//! Why it is one test function
//! ---------------------------
//! The environment is process-wide and `cargo test` runs a file's tests on several threads, so two
//! tests that both write this variable would race. `row_tile_fallback.rs` sets the same precedent.
//! Splitting this into readable phases is done with comments, not with `#[test]`.
//!
//! What this file does **not** prove
//! ---------------------------------
//! That `vkCreateComputePipelines` was called with the vector, or that a GPU executed it. That
//! requires a Vulkan device and lives in the device lanes. The seam this file reaches is the last
//! host-side point at which the numbers exist; `src/vk/session.rs` carries them from there to the
//! pipeline key. The `record_pipeline_variant` phase below feeds the **captured** vector through
//! the production recorder, which proves the recorder's key derivation over a real handler-built
//! vector — it does not prove that a session ran.

use onnxruntime_vulkan_ep::counters;
use onnxruntime_vulkan_ep::engine::{
    AttrValue, BufferView, DType, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef,
    TensorDesc, TensorRef,
};

const VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";
const CEILING_VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS";

/// Index of `QB_COLS` in `q_gemv.comp`'s specialisation vector.
///
/// The vector the handler builds is `[wg, bits, block_size, has_zp, cols, packed, rows]`. Named
/// rather than spelled inline so that a reordering in the handler is caught by a failing name
/// rather than by a silently-passing wrong index.
const COLS_INDEX: usize = 4;
/// Index of `QB_ROWS` in the same vector.
const ROWS_INDEX: usize = 6;

/// A `DispatchContext` that records the dispatches and nothing else.
///
/// Buffer handles are dense integers because nothing here inspects them; the only field under test
/// is `spec_constants`.
#[derive(Default)]
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

/// One Phi-3.5 projection node: `K = N = 3072`, q4, block 32, fp16 activations, `M` prefill rows.
///
/// `M = 4` is chosen for the tiling phases because it is a width at which `(16, 2)` and `(8, 4)`
/// name the same bytes — that is the case the override exists to reach, and asserting it at a
/// non-tying width would prove something weaker.
fn phi35_node(m: i64) -> NodeDesc {
    let (k, n) = (3072_i64, 3072_i64);
    let blocks = k / 32;
    let inp = |name: &str, dtype: DType, shape: Vec<i64>| TensorRef {
        name: name.to_string(),
        desc: Some(TensorDesc::new(dtype, shape)),
        is_initializer: name != "A",
    };
    NodeDesc {
        op_type: "MatMulNBits".to_string(),
        name: "gemv_tile_override_seam".to_string(),
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
            inp("A", DType::F16, vec![1, m, k]),
            inp("B", DType::U8, vec![n, blocks, 16]),
            inp("scales", DType::F16, vec![n * blocks]),
        ],
        outputs: vec![OutRef {
            name: "Y".to_string(),
            desc: Some(TensorDesc::new(DType::F16, vec![1, m, n])),
        }],
    }
}

/// Translate through the **registry row**, not through the handler by name.
///
/// `spec_for` is the function `Compile` calls, and the `translate` field it hands back is the same
/// pointer the op table holds. A change that unwired the row would fail here at the lookup; a
/// change that unwired the selector from the handler would fail at the assertions.
fn translate(node: &NodeDesc) -> Result<Vec<KernelRequest>, String> {
    let spec = onnxruntime_vulkan_ep::registry::spec_for(node)
        .expect("MatMulNBits must be a live registry row");
    let mut rec = Recorder::default();
    match (spec.translate)(spec, node, &mut rec) {
        Ok(()) => Ok(rec.dispatches),
        Err(e) => {
            assert!(
                rec.dispatches.is_empty(),
                "a refusal must happen before dispatch, but {} dispatch(es) were recorded",
                rec.dispatches.len()
            );
            Err(format!("{e:?}"))
        }
    }
}

fn tile_of(k: &KernelRequest) -> (u32, u32) {
    assert!(
        k.spec_constants.len() > ROWS_INDEX,
        "the q_gemv specialisation vector is shorter than the tile indices: {:?}",
        k.spec_constants
    );
    (k.spec_constants[COLS_INDEX], k.spec_constants[ROWS_INDEX])
}

#[test]
fn the_override_reaches_the_production_specialisation_constants() {
    // SAFETY: this is the only test in this binary, so it is the only writer of these variables in
    // the process; both are removed again before the test returns.
    unsafe {
        std::env::remove_var(VAR);
        std::env::remove_var(CEILING_VAR);
    }
    counters::reset();

    // ── Phase 1: unset is the default search, observed through the live handler ────────────
    let unset = translate(&phi35_node(4)).expect("unset must translate");
    assert_eq!(unset.len(), 1, "one dispatch per node");
    let default_tile = tile_of(&unset[0]);
    assert_eq!(
        default_tile,
        (16, 2),
        "with no override the handler must build the tile the byte-model search picks"
    );
    assert_eq!(
        counters::gemv_tile_override_refusals().as_deref(),
        Some(&[][..]),
        "an unset override must refuse nothing"
    );
    let default_constants = unset[0].spec_constants.clone();

    // ── Phase 2: the incumbent, asked for explicitly, is the identical vector ──────────────
    //
    // This is the control that makes phase 3 mean something. If asking for `(16, 2)` produced a
    // *different* vector from the default, then phase 3's difference would be attributable to the
    // override mechanism rather than to the tile.
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "16,2") };
    let incumbent = translate(&phi35_node(4)).expect("16,2 is legal here");
    assert_eq!(
        incumbent[0].spec_constants, default_constants,
        "asking for the tile the search would pick must change nothing at all"
    );

    // ── Phase 3: the equal-total candidate, unreachable any other way, is reached ──────────
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "8,4") };
    let overridden = translate(&phi35_node(4)).expect("8,4 is legal here");
    assert_eq!(
        tile_of(&overridden[0]),
        (8, 4),
        "the override must reach the production specialisation constants"
    );
    // Everything else about the dispatch is the same request: this is a tile change, not a
    // different kernel. The grid extents are expected to move — halving `cols` doubles the x
    // extent and doubling `rows` halves the y extent — which is itself the evidence that the
    // handler used the requested pair rather than merely reporting it.
    assert_eq!(overridden[0].shader, unset[0].shader);
    assert_eq!(overridden[0].push_constants, unset[0].push_constants);
    assert_eq!(
        overridden[0].workgroups[0],
        2 * unset[0].workgroups[0],
        "halving the column tile doubles the x extent: that is what a column tile is"
    );
    assert_eq!(
        overridden[0].workgroups[1], 1,
        "4 rows in tiles of 4 is one row-tile"
    );
    assert_eq!(
        unset[0].workgroups[1], 2,
        "4 rows in tiles of 2 is two row-tiles"
    );
    for (i, (a, b)) in default_constants
        .iter()
        .zip(overridden[0].spec_constants.iter())
        .enumerate()
    {
        if i != COLS_INDEX && i != ROWS_INDEX {
            assert_eq!(
                a, b,
                "specialisation constant {i} must not move with the tile"
            );
        }
    }
    assert_eq!(
        counters::gemv_tile_override_refusals().as_deref(),
        Some(&[][..]),
        "a legal override must refuse nothing"
    );

    // ── Phase 4: the recorder seam, over the vector the handler actually built ─────────────
    //
    // The pipeline key is derived from the effective specialisation vector in `vk/session.rs`.
    // That call needs a device. What can be proven without one is that the *production recorder*,
    // fed the vector this handler built, distinguishes the two tiles — i.e. that the override is
    // observable as a distinct pipeline rather than collapsing onto the default's key. The
    // remaining link (session -> recorder) is stated as a limitation in `docs/PERF.md` §26.4
    // rather than claimed here.
    counters::reset();
    assert!(counters::record_pipeline_variant(
        "q_gemv_f16",
        &default_constants
    ));
    assert!(counters::record_pipeline_variant(
        "q_gemv_f16",
        &overridden[0].spec_constants
    ));
    let variants = counters::pipeline_variants();
    assert_eq!(
        variants.len(),
        2,
        "the two tiles must be two pipelines, not one: {variants:?}"
    );
    assert!(
        variants.iter().any(|v| v.ends_with(",16,1,2")),
        "the default tile's key must name cols=16 rows=2: {variants:?}"
    );
    assert!(
        variants.iter().any(|v| v.ends_with(",8,1,4")),
        "the overridden tile's key must name cols=8 rows=4: {variants:?}"
    );
    counters::reset();

    // ── Phase 5: every refusal refuses before dispatch, and is observable ──────────────────
    //
    // The pairs below are one per rule that can be reached from this node's shape. A refusal that
    // silently fell back to the default tile would translate successfully and be caught by the
    // `is_err` assertion; a refusal that dispatched anyway would be caught inside `translate`.
    let refusals: [(&str, &str); 8] = [
        ("nonsense", "unparseable"),
        ("", "unparseable"),
        ("8,4,2", "unparseable"),
        ("2.5,4", "unparseable"),
        ("-8,4", "unparseable"),
        ("3,2", "cols_not_power_of_two"),
        ("32,1", "cols_above_cap"),
        ("16,4", "tile_above_accumulator_budget"),
    ];
    for (value, token) in refusals {
        // SAFETY: as above.
        unsafe { std::env::set_var(VAR, value) };
        if std::env::var_os(VAR).is_none() {
            // Windows removes a variable set to the empty string. The empty-string case is pinned
            // by the pure parser instead; skipping it here is honest, asserting it would not be.
            continue;
        }
        let err = translate(&phi35_node(4))
            .expect_err(&format!("{VAR}={value:?} must refuse, never fall back"));
        assert!(
            err.contains(token),
            "{VAR}={value:?} refused, but not by the expected rule: {err}"
        );
        let rows = counters::gemv_tile_override_refusals().expect("the counter must be readable");
        assert!(
            rows.iter().any(|r| r.starts_with(token)),
            "{VAR}={value:?} refused without recording a {token} row: {rows:?}"
        );
    }
    let recorded = counters::gemv_tile_override_refusals().expect("readable");
    assert!(
        !recorded.is_empty(),
        "the refusal counter must be the observable half of the refusal"
    );
    counters::reset();

    // ── Phase 6: precedence against the row-tile ceiling is refusal, not silent demotion ───
    //
    // SAFETY: as above.
    unsafe {
        std::env::set_var(CEILING_VAR, "2");
        std::env::set_var(VAR, "8,4");
    }
    let err = translate(&phi35_node(4))
        .expect_err("the ceiling must not be silently outranked by the exact request");
    assert!(
        err.contains("rows_above_ceiling_in_force"),
        "the two controls disagreeing must be reported as such: {err}"
    );
    // With the ceiling in force and no exact request, the search still runs and still succeeds.
    // SAFETY: as above.
    unsafe { std::env::remove_var(VAR) };
    let ceiled = translate(&phi35_node(4)).expect("the ceiling alone must not refuse");
    assert_eq!(tile_of(&ceiled[0]), (16, 2));

    // ── Phase 7: removing the override restores the default exactly ────────────────────────
    // SAFETY: as above.
    unsafe {
        std::env::remove_var(VAR);
        std::env::remove_var(CEILING_VAR);
    }
    counters::reset();
    let restored = translate(&phi35_node(4)).expect("restored must translate");
    assert_eq!(
        restored[0].spec_constants, default_constants,
        "removing the override must restore the default, not latch the last value"
    );
    assert_eq!(
        counters::gemv_tile_override_refusals().as_deref(),
        Some(&[][..]),
    );

    // ── Phase 8: decode is untouched, whatever the override says ───────────────────────────
    //
    // `M = 1` is the shipping decode geometry and the one arm of `q_gemv.comp` that is verbatim
    // pre-issue-#7 text. A row-tile request at decode width must refuse rather than quietly widen
    // the kernel that serves every generated token.
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "16,2") };
    let err = translate(&phi35_node(1)).expect_err("a row tile at decode width must refuse");
    assert!(err.contains("row_tile_at_decode_width"), "{err}");
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "16,1") };
    let decode = translate(&phi35_node(1)).expect("the decode tile itself is legal");
    assert_eq!(tile_of(&decode[0]), (16, 1));

    // SAFETY: as above — leave the process as it was found.
    unsafe {
        std::env::remove_var(VAR);
        std::env::remove_var(CEILING_VAR);
    }
    counters::reset();
}
