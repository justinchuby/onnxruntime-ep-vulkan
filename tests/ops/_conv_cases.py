"""The `Conv` attribute space, in one list, because the proof key does not contain it.

A ledger entry for `Conv` names the domain, opset window, dtypes, module, shape class and
arity — and **nothing else**. `group`, `strides`, `dilations` and `pads` are not key components,
so `evidence/proof_ledger.jsonl` carrying `ai.onnx::Conv/1+/f32,f32,f32>f32/metadata/static/n3`
says exactly nothing about whether a stride-2 asymmetric-pad depthwise convolution is right.

That gap is not a defect in the key — arity and dtype are what decide which *binding layout* and
which *compiled module* run, and those are what a proof is about. It is a gap in what a proof
covers, and the honest response is to name the uncovered axis and test it separately rather than
to pretend the entry is broader than it is.

Every case below is drawn from a form MobileNetV2-12 actually contains, or is the boundary
immediately outside one.
"""

from __future__ import annotations

#: ``(id, n, c, m, group, kernel, strides, dilations, pads, hw, bias)``
CONV_CASES = [
    # The 34 pointwise convolutions that make up two thirds of MobileNetV2.
    ("pointwise",        1, 4, 6, 1, (1, 1), (1, 1), (1, 1), (0, 0, 0, 0), (7, 7),  True),
    # The plain 3x3 with symmetric padding.
    ("k3_pad1",          1, 4, 6, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1), (8, 10), True),
    # Depthwise: `group == C == M`. 17 of MobileNetV2's convolutions are this shape, and it is
    # the case where a handler that ignored `group` would read the wrong input channel entirely
    # rather than merely accumulating too much.
    ("depthwise",        1, 6, 6, 6, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1), (8, 10), True),
    # Grouped but not depthwise: the case between the two, where `M/group != 1`.
    ("grouped_g2",       1, 4, 6, 2, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1), (8, 10), True),
    # Stride 2 with symmetric pads — the downsampling layers.
    ("stride2",          1, 4, 6, 1, (3, 3), (2, 2), (1, 1), (1, 1, 1, 1), (8, 10), True),
    # Stride 2 with an ASYMMETRIC pad. The begin pad shifts the read window and the end pad only
    # changes the output extent; a handler that used one for both is wrong exactly here.
    ("stride2_asym_pad", 1, 4, 6, 1, (3, 3), (2, 2), (1, 1), (0, 1, 1, 0), (9, 9),  True),
    # Dilation. Not in MobileNetV2; it is in every segmentation backbone, and it is the one
    # attribute that changes *where* the kernel reads without changing how much it reads.
    ("dilation2",        1, 4, 6, 1, (3, 3), (1, 1), (2, 2), (0, 0, 0, 0), (9, 9),  True),
    # Non-square kernel with non-square strides: the two axes must not be transposed.
    ("k1x3_s2x1",        1, 4, 6, 1, (1, 3), (2, 1), (1, 1), (0, 1, 0, 1), (8, 10), True),
    # No bias. A different arity, hence a different proof key, and the case where the inert
    # placeholder binding must not contribute.
    ("nobias",           1, 4, 6, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1), (8, 10), False),
    # Batch > 1: the outermost index of the flattened grid.
    ("batch3",           3, 4, 6, 1, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1), (8, 10), True),
    # Depthwise, stride 2, batch 2 — three of the above at once, which is what a real graph does.
    ("depthwise_s2_n2",  2, 6, 6, 6, (3, 3), (2, 2), (1, 1), (1, 1, 1, 1), (8, 10), True),
    # A kernel exactly as large as its padded input: OH == OW == 1, the degenerate-but-valid end.
    ("kernel_eq_input",  1, 4, 6, 1, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), (3, 3),  True),
]


def out_extent(extent: int, pad_b: int, pad_e: int, dil: int, k: int, stride: int) -> int:
    """The ONNX output-extent formula, restated here so the test does not ask the EP for it."""
    return (extent + pad_b + pad_e - ((k - 1) * dil + 1)) // stride + 1
