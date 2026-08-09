//! `ONNXRUNTIME_EP_VULKAN_GEMV_TILE`, exercised through the real translate handler.
//!
//! Why this file exists rather than a unit test in `src/ops/quant.rs`
//! ------------------------------------------------------------------
//! The same two reasons `rust/tests/row_tile_fallback.rs` gives:
//!
//! 1. `rust/tests/layering.rs` forbids `unsafe` anywhere under `src/ops/`, and mutating process
//!    environment on current Rust requires it. So the *grammar* and the *legality* live in
//!    `ops::quant` as pure functions with their own unit tests, and the *plumbing* — that the
//!    shipping handler consults the variable at all, and that what it selects reaches the
//!    specialisation vector — is tested here.
//! 2. The pair is not redundant. A refactor that stopped reading the variable, or read it and
//!    dropped the result, would leave every pure unit test green. This is the test that fails.
//!
//! What is under test, exactly
//! ---------------------------
//! The **production** path. `registry::spec_for()` finds the `MatMulNBits` row that ships, and
//! `spec.translate` is the same function pointer `vk/session.rs` calls. The `KernelRequest` it
//! produces carries the specialisation vector a pipeline would be built from, and that vector is
//! handed to `counters::record_pipeline_variant` — the *existing* readback witness, unchanged by
//! this change — exactly as `vk/session.rs` hands it the effective pair.
//!
//! That last step is a mirror of a production call site, so it is guarded rather than assumed:
//! `the_witness_this_file_reads_is_still_the_one_production_writes` reads `vk/session.rs` and
//! fails if the call it mirrors moves or changes shape. A mirror nobody checks is a restatement.
//!
//! What the recorded strings are, and are not
//! ------------------------------------------
//! The expectations below are computed from this build's own selector, not transcribed from a
//! benchmark record. They are host-side facts about which pipeline *would* be built. Two committed
//! artifacts — `bench/results/real_model_diagnostics.json` and `..._before_gqa.json`, 18 run
//! records each — independently record `q_gemv_matmul_nbits_f16:32,4,32,0,16,1,2` for the tiled
//! arm at the `prefill/M8` and `prefill/M128` cases and `…,16,1,1` at `prefill/M1` and
//! `decode/M1/past1024`, which is agreement rather than the source of the expectation.
//!
//! Why no GPU
//! ----------
//! Nothing here builds a pipeline, allocates device memory, or submits anything. The question the
//! instrument answers is *which kernel would be built*, and that is decided on the host before any
//! device is touched.

use onnxruntime_vulkan_ep::counters;
use onnxruntime_vulkan_ep::engine::{
    AttrValue, BufferView, DType, DispatchContext, EpResult, KernelRequest, NodeDesc, OutRef,
    TensorDesc, TensorRef,
};
use onnxruntime_vulkan_ep::ops::quant::{gemv_tile, gemv_workgroup};
use onnxruntime_vulkan_ep::registry;

const VAR: &str = "ONNXRUNTIME_EP_VULKAN_GEMV_TILE";

/// One Phi-3.5 projection shape: `K = N = 3072`, q4, block 32, fp16 activations, no zero points.
const K: i64 = 3072;
const N: i64 = 3072;

/// Puts the variable back on **every** exit path, including the unwind path.
///
/// A trailing `remove_var` at the bottom of a test does not run when an assertion above it panics,
/// and the value then leaks into whatever runs next.
struct EnvRestore(Option<std::ffi::OsString>);

impl EnvRestore {
    fn take() -> Self {
        Self(std::env::var_os(VAR))
    }
    fn set(value: &str) {
        // SAFETY: this binary runs exactly one test that touches the environment, so no other
        // thread in this process can observe the intermediate value.
        unsafe { std::env::set_var(VAR, value) };
    }
    fn clear() {
        // SAFETY: as above.
        unsafe { std::env::remove_var(VAR) };
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match self.0.take() {
            // SAFETY: as above.
            Some(prior) => unsafe { std::env::set_var(VAR, prior) },
            // SAFETY: as above.
            None => unsafe { std::env::remove_var(VAR) },
        }
    }
}

/// Captures the `KernelRequest` the shipping handler emits. Resolves nothing and allocates
/// nothing — a translate handler's whole output is the request.
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

fn phi35_node(m: i64) -> NodeDesc {
    let blocks = K / 32;
    let input = |name: &str, dtype: DType, shape: Vec<i64>| TensorRef {
        name: name.to_string(),
        desc: Some(TensorDesc::new(dtype, shape)),
        is_initializer: name != "A",
    };
    NodeDesc {
        op_type: "MatMulNBits".to_string(),
        name: "phi35_projection".to_string(),
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
            input("A", DType::F16, vec![1, m, K]),
            input("B", DType::U8, vec![N, blocks, 16]),
            input("scales", DType::F16, vec![N * blocks]),
        ],
        outputs: vec![OutRef {
            name: "Y".to_string(),
            desc: Some(TensorDesc::new(DType::F16, vec![1, m, N])),
        }],
    }
}

/// The whole recorded key for this shape at a given `(cols, rows)`.
///
/// Built from the module's own constants rather than typed out, so the string this file asserts
/// cannot drift away from the vector the handler emits without the handler itself changing. The
/// vector is `[wg, bits, block_size, has_zp, cols, packed, rows]`.
fn expected_key(cols: u32, rows: u32) -> String {
    let wg = gemv_workgroup((K / 32) as u64);
    format!("q_gemv_matmul_nbits_f16:{wg},4,32,0,{cols},1,{rows}")
}

/// Run the shipping translate handler for a Phi-3.5 `MatMulNBits` at `M = m`, and record the
/// resulting pipeline the way `vk/session.rs` records it. Returns the recorded key, or the
/// refusal text.
fn translate_and_record(m: i64) -> Result<String, String> {
    let node = phi35_node(m);
    let spec = registry::spec_for(&node).expect("MatMulNBits must have a shipping registry row");
    let mut rec = Recorder::default();
    (spec.translate)(spec, &node, &mut rec).map_err(|e| format!("{e:?}"))?;
    assert_eq!(
        rec.dispatches.len(),
        1,
        "MatMulNBits translates to exactly one dispatch"
    );
    let k = &rec.dispatches[0];
    // The mirror of `vk/session.rs`'s call, guarded below.
    counters::record_pipeline_variant(k.shader, &k.spec_constants);
    Ok(counters::pipeline_variants()
        .into_iter()
        .find(|v| v.starts_with("q_gemv"))
        .expect("a q_gemv pipeline variant must have been recorded"))
}

/// The assertion for a value the platform can hold but Rust cannot read as `&str`. Shared by both
/// arms of the `cfg` fork below so the *claim* is written once and only the construction differs.
fn assert_non_utf8_refuses(bad: std::ffi::OsString) {
    assert!(
        bad.to_str().is_none(),
        "this fixture must actually be un-transcodable, or the test proves nothing"
    );
    // SAFETY: this binary runs exactly one test that touches the environment.
    unsafe { std::env::set_var(VAR, &bad) };
    counters::reset();
    let err = translate_and_record(128)
        .expect_err("a non-UTF-8 value must refuse rather than read as unset");
    assert!(
        err.contains(VAR) && err.contains("UTF-8"),
        "the refusal must say the value was not valid UTF-8; got {err}"
    );
    assert!(
        counters::pipeline_variants().is_empty(),
        "a non-UTF-8 request must build no pipeline"
    );
}

/// Everything that touches the environment or the process-global counters, in one test.
///
/// `cargo test` runs the tests in a binary on several threads; the environment and
/// `counters::pipeline_variants()` are both process-wide, so two tests writing either would race.
#[test]
fn the_tile_request_reaches_the_specialisation_vector_or_refuses_before_it() {
    let _restore = EnvRestore::take();

    // ── THE UNSET CONTROL ────────────────────────────────────────────────────────────────
    // With the variable absent the handler must select what it selected before this variable
    // existed. The expectation is the searched tile computed from this build's own selector, so
    // the control cannot pass by agreeing with a number somebody typed.
    let wg = gemv_workgroup((K / 32) as u64);
    let searched_prefill = gemv_tile(128, N as u64, K as u64, 4, 2, wg);
    let searched_decode = gemv_tile(1, N as u64, K as u64, 4, 2, wg);
    assert_eq!(
        searched_prefill,
        (16, 2),
        "the shipping selector must still choose (16,2) at M=128 for this shape"
    );
    assert_eq!(
        searched_decode,
        (16, 1),
        "the shipping selector must still choose rows=1 at M=1"
    );

    EnvRestore::clear();
    counters::reset();
    assert_eq!(
        translate_and_record(128).expect("the unset control must not refuse"),
        expected_key(searched_prefill.0, searched_prefill.1),
        "unset must select the searched tile for this shape, byte for byte"
    );
    assert_eq!(
        counters::gemv_tile_request_refused(),
        "NONE",
        "nothing was requested, so nothing can have been refused"
    );

    // Decode is the other half of the control: `M = 1` takes the specialisation-constant arm that
    // holds the verbatim pre-row-tile kernel.
    counters::reset();
    assert_eq!(
        translate_and_record(1).expect("the decode control must not refuse"),
        expected_key(searched_decode.0, searched_decode.1),
        "M=1 must still take the decode arm when nothing is requested"
    );

    // ── POSITIVE POLARITY: the arm this instrument exists for ────────────────────────────
    // `(8,4)` names exactly the bytes `(16,2)` names, so no value of
    // `ONNXRUNTIME_EP_VULKAN_GEMV_MAX_ROWS` can select it — the strict-`<` improvement rule keeps
    // the incumbent on a tie. This asserts the pair actually reaches the pipeline key, which is
    // the only thing that makes an A/B between the two arms attributable.
    EnvRestore::set("8,4");
    counters::reset();
    assert_eq!(
        translate_and_record(128).expect("(8,4) is legal for this shape and must be honoured"),
        expected_key(8, 4),
        "the requested tile must appear at indices 4 and 6 of the recorded specialisation vector"
    );
    assert_eq!(
        counters::gemv_tile_request_refused(),
        "NONE",
        "an honoured request is not a refusal"
    );

    // The request is honoured wherever it is legal, including at `M = 1`. Deliberate, and the
    // reason the unset control above is the load-bearing one: an explicit request beats an
    // implicit policy, so decode is protected only by the variable being unset.
    counters::reset();
    assert_eq!(
        translate_and_record(1).expect("(8,4) is legal at M=1 too"),
        expected_key(8, 4),
        "an explicit request is honoured at M=1 as well; only the unset path protects decode"
    );

    // ── NEGATIVE POLARITY: illegal refuses, visibly, before any dispatch ─────────────────
    // `(16,4)` is the pair the selector itself refuses as strictly-better-and-illegal:
    // `cols*rows = 64` exceeds the accumulator budget `GEMV_MAX_TILE = 32`. Raising that budget is
    // the change this instrument exists to inform; it is emphatically not this change.
    for (value, expect) in [
        ("16,4", "GEMV_MAX_TILE"),
        ("32,1", "GEMV_MAX_COLS"),
        ("4,8", "GEMV_MAX_ROWS"),
        ("7,2", "does not divide"),
    ] {
        EnvRestore::set(value);
        counters::reset();
        let err = translate_and_record(128)
            .expect_err(&format!("{VAR}={value} must refuse rather than dispatch"));
        assert!(
            err.contains(expect),
            "{VAR}={value} must say which rule it broke; got {err}"
        );
        let recorded = counters::gemv_tile_request_refused();
        assert!(
            recorded.contains(VAR) && recorded.contains(expect),
            "the refusal must reach the diagnostics surface; got {recorded:?}"
        );
        assert!(
            counters::pipeline_variants().is_empty(),
            "a refused request must build no pipeline at all, got {:?}",
            counters::pipeline_variants()
        );
    }

    // A malformed value refuses on syntax rather than legality, and quotes the operator's text.
    for value in [
        "",
        "8",
        "8, 4",
        "8,4,2",
        "-8,4",
        "0,4",
        "8,4junk",
        "4294967296,4",
    ] {
        EnvRestore::set(value);
        counters::reset();
        let err = translate_and_record(128)
            .expect_err(&format!("{VAR}={value:?} must refuse rather than dispatch"));
        assert!(
            err.contains(VAR),
            "{VAR}={value:?} refusal must name the variable; got {err}"
        );
        assert!(
            counters::pipeline_variants().is_empty(),
            "a malformed request must build no pipeline"
        );
    }

    // A value that is not valid UTF-8 must refuse, not read as "unset". `std::env::var` collapses
    // NotPresent and NotUnicode into one Err, so a `.ok()` here would silently take the searched
    // tile while the operator believed their pair was in force — and nothing would be recorded as
    // refused.
    //
    // Both arms: an environment value is UTF-16 on Windows and bytes on everything else, so an
    // ill-formed one is built differently, but the *assertion* is identical.
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStringExt;
        // "8," + an unpaired surrogate + "4": well-formed UTF-16 storage, not transcodable.
        assert_non_utf8_refuses(std::ffi::OsString::from_wide(&[0x38, 0x2C, 0xD800, 0x34]));
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::ffi::OsStringExt;
        // "8," + a byte that begins no valid UTF-8 sequence + "4".
        assert_non_utf8_refuses(std::ffi::OsString::from_vec(vec![0x38, 0x2C, 0xFF, 0x34]));
    }

    // ── THE LATCH CONTROL ────────────────────────────────────────────────────────────────
    // Removing the variable must restore the default, not leave the last request latched. The
    // property a process-global cached selection would break.
    EnvRestore::clear();
    counters::reset();
    assert_eq!(
        translate_and_record(128).expect("the restored control must not refuse"),
        expected_key(searched_prefill.0, searched_prefill.1),
        "removing the request must restore the searched tile, not latch the last one"
    );
    assert_eq!(counters::gemv_tile_request_refused(), "NONE");
}

/// The mirror guard. This file records the pipeline variant the way `vk/session.rs` does; if that
/// call moves or changes shape, this file is testing a composition production no longer performs.
///
/// Reads the Rust rather than restating it. Touches no environment and no counter, so it is safe
/// to run beside the test above.
#[test]
fn the_witness_this_file_reads_is_still_the_one_production_writes() {
    let session = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("vk")
        .join("session.rs");
    let src = std::fs::read_to_string(&session).expect("vk/session.rs must be readable");
    assert!(
        src.contains("crate::counters::record_pipeline_variant(eff_shader, eff_spec_constants)"),
        "vk/session.rs no longer records the effective (shader, spec_constants) pair, so \
         `translate_and_record` in this file mirrors a call site that has moved. Re-read \
         session.rs and update the mirror before trusting anything in this file."
    );
}
