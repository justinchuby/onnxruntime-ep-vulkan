//! `ONNXRUNTIME_EP_VULKAN_GEMV_TILE`, exercised through the process environment and the real
//! `MatMulNBits` handler (issue #81).
//!
//! Why this file exists rather than a unit test in `src/ops/quant.rs`
//! ------------------------------------------------------------------
//! Two reasons, and the second is the one that matters.
//!
//! 1. `tests/layering.rs` forbids `unsafe` anywhere under `src/ops/` (DESIGN.md §4.2), and
//!    mutating the process environment on current Rust requires it. The unit tests therefore pass
//!    the request in as a parameter and prove the *rules*; this file proves the *plumbing*.
//!
//! 2. The unit tests call the selector. Nobody dispatches a selector. What a later A/B harness
//!    depends on is that the pair the selector chose arrives in the **specialisation constant
//!    vector** the engine hands to `vkCreateComputePipelines` — and that is produced by
//!    `matmul_nbits_gemv`, several dozen lines downstream of the selector, through a `Vec<u32>`
//!    whose fields are positional and unnamed. A refactor that reordered that vector, or that
//!    stopped consulting the environment at all, would leave every unit test in the tree green.
//!    This file is the one that fails.
//!
//! The claim being underwritten is narrow and worth stating exactly: *forcing `(8,4)` and forcing
//! `(16,2)` produce two different pipelines on the same node.* If they produced the same pipeline
//! the later measurement would be one run reported twice, and a null result from it would be
//! unfalsifiable rather than informative.
//!
//! Everything is in one `#[test]` for the reason `row_tile_fallback.rs` gives: `cargo test` runs
//! the tests in a binary on several threads, the environment is process-wide, and two tests that
//! both write this variable would race.

use onnxruntime_vulkan_ep::engine::{
    AttrValue, BufferView, DType, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef,
    TensorDesc, TensorRef,
};

const VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";
const CEILING: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS";

/// Index of `QB_COLS` in `q_gemv.comp`'s constant vector, and of `QB_ROWS`.
///
/// Named rather than inlined because the whole point of this file is that these are positional:
/// `[wg, bits, block_size, has_zp, cols, packed, rows]`.
const COLS: usize = 4;
const ROWS: usize = 6;

/// Records what the handler asked the engine to do. Nothing else; there is no device here.
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

/// One Phi-3.5 projection node: `K = N = 3072`, q4, block 32, fp16 activations, `M` rows.
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

/// Translate one node through the registry's live handler, returning what reached `dispatch`.
///
/// `Err` carries the handler's refusal *and* the fact that nothing was dispatched, because those
/// are two separate claims and the second is the one a refusal has to make good on.
fn translate(m: i64) -> Result<Vec<KernelRequest>, String> {
    let n = node(m);
    let spec = onnxruntime_vulkan_ep::registry::spec_for(&n).expect("MatMulNBits is live");
    let mut rec = Recorder::default();
    match (spec.translate)(spec, &n, &mut rec) {
        Ok(()) => Ok(rec.dispatches),
        Err(e) => {
            assert!(
                rec.dispatches.is_empty(),
                "the handler refused after dispatching {} kernel(s); a refusal that has already \
                 built a pipeline is not a pre-dispatch refusal",
                rec.dispatches.len()
            );
            Err(format!("{e:?}"))
        }
    }
}

/// The `(cols, rows)` actually specialised into the one pipeline this node produces.
fn dispatched_tile(m: i64) -> (u32, u32) {
    let d = translate(m).unwrap_or_else(|e| panic!("M={m} was refused: {e}"));
    assert_eq!(d.len(), 1, "M={m}: one MatMulNBits node is one dispatch");
    let c = &d[0].spec_constants;
    assert!(
        c.len() > ROWS,
        "M={m}: the constant vector is too short to hold the row tile: {c:?}"
    );
    (c[COLS], c[ROWS])
}

#[test]
fn the_environment_can_force_either_arm_of_the_tile_ab() {
    // SAFETY: this is the only test in this binary, so it is the only writer of these variables
    // in the process; both are removed again before it returns.
    unsafe {
        std::env::remove_var(VAR);
        std::env::remove_var(CEILING);
    }

    // ── The control: no request. This is the shipping geometry, and every later claim is
    // relative to it.
    let prefill = dispatched_tile(128);
    let decode = dispatched_tile(1);
    assert_eq!(
        prefill,
        (16, 2),
        "the shipping selector must still pick the incumbent tile at M=128"
    );
    assert_eq!(decode.1, 1, "decode must never take a row tile");
    let control = translate(128).expect("no request cannot refuse");
    let control_constants = control[0].spec_constants.clone();
    let control_groups = control[0].workgroups;

    // ── Arm A: force the incumbent. Byte-for-byte the control, which is what makes it a control
    // rather than a third arm.
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "16,2") };
    let forced_incumbent = translate(128).expect("(16,2) is legal at this shape");
    assert_eq!(
        forced_incumbent[0].spec_constants, control_constants,
        "forcing the tile the selector already picks must reproduce the control exactly"
    );
    assert_eq!(forced_incumbent[0].workgroups, control_groups);

    // ── Arm B: force the equal-traffic tile. The pipeline must differ, and differ in exactly the
    // two fields the tile names.
    // SAFETY: as above.
    unsafe { std::env::set_var(VAR, "8,4") };
    let arm_b = translate(128).expect("(8,4) is legal at this shape");
    let b_constants = arm_b[0].spec_constants.clone();
    assert_eq!((b_constants[COLS], b_constants[ROWS]), (8, 4));
    assert_ne!(
        b_constants, control_constants,
        "the two arms specialise identically, so they are one pipeline and there is no A/B"
    );
    assert_eq!(
        b_constants.len(),
        control_constants.len(),
        "the arms must differ in value, not in shape"
    );
    for (i, (a, b)) in control_constants.iter().zip(&b_constants).enumerate() {
        if i == COLS || i == ROWS {
            assert_ne!(
                a, b,
                "constant {i} is the tile and must differ between arms"
            );
        } else {
            assert_eq!(a, b, "constant {i} is not the tile and must not have moved");
        }
    }
    // Same shape, different grid: half the columns per group, half the row tiles.
    assert_eq!(
        arm_b[0].workgroups[0],
        control_groups[0] * 2,
        "cols 16 -> 8 must double the x extent"
    );
    assert_eq!(
        arm_b[0].workgroups[1] * 2,
        control_groups[1],
        "rows 2 -> 4 must halve the y extent"
    );

    // ── The null control survives the treatment: decode is untouched in both arms.
    assert_eq!(
        dispatched_tile(1),
        decode,
        "a forced prefill tile must not re-tile decode; the decode pipeline is the null control \
         and an override that moved it would make it a second treatment"
    );

    // ── Refusals. Each of these must fail *before* a pipeline exists — `translate` asserts that
    // nothing reached `dispatch` on the error path.
    for bad in [
        "",             // present and empty is a request, not an absence
        "8",            // no comma
        "8,",           // missing field
        ",4",           // missing field
        "8, 4",         // interior whitespace
        " 8,4",         // leading whitespace
        "8,4 ",         // trailing whitespace
        "8,4\n",        // trailing newline, the shell's favourite
        "+8,4",         // sign
        "08,4",         // leading zero
        "0x8,4",        // radix prefix
        "8,4,1",        // extra field
        "eight,four",   // not a number
        "4294967296,4", // u32 overflow
        "0,4",          // zero extent
        "8,0",          // zero extent
        "16,4",         // legal fields, over the accumulator budget
        "32,1",         // over GEMV_MAX_COLS
        "1,8",          // over GEMV_MAX_ROWS
        "5,2",          // does not divide N = 3072
    ] {
        // SAFETY: as above.
        unsafe { std::env::set_var(VAR, bad) };
        let refusal = translate(128).expect_err(&format!(
            "{VAR}={bad:?} was accepted and dispatched something"
        ));
        assert!(
            refusal.contains(VAR),
            "{VAR}={bad:?} was refused without naming the variable: {refusal}"
        );
    }

    // ── The two controls contradicting each other is two experiments, so neither runs.
    // SAFETY: as above.
    unsafe {
        std::env::set_var(VAR, "8,4");
        std::env::set_var(CEILING, "1");
    }
    let refusal = translate(128).expect_err("a request above the ceiling must refuse");
    assert!(
        refusal.contains(CEILING),
        "the refusal must name the control it conflicts with: {refusal}"
    );
    // SAFETY: as above.
    unsafe { std::env::remove_var(CEILING) };

    // ── Removing the request restores the control exactly, rather than latching the last value.
    // SAFETY: as above.
    unsafe { std::env::remove_var(VAR) };
    let restored = translate(128).expect("no request cannot refuse");
    assert_eq!(restored[0].spec_constants, control_constants);
    assert_eq!(restored[0].workgroups, control_groups);
}
