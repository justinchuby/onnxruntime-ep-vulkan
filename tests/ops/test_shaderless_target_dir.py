"""Does the criterion-5 shader-less witness own the directory it builds into?

WHY THIS FILE EXISTS
====================
`_shaderless.py` builds a deliberately broken EP — one compiled with
`ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC=1`, which advertises zero devices — so criterion 5
can be observed rather than assumed. It passed `env = dict(os.environ)` to that build.

On the Linux lane `CARGO_TARGET_DIR` is exported, and it must be: left unset it resolves to
`/mnt/c/.../rust/target`, where it clobbers the Windows build and races the other worktree.
So the witness inherited it, and the shader-less artifact was written **into the shared target
directory, over the real one**.

The loud symptom was the wrong one. The witness raised

    ERROR(instrument): cargo exited 0 but <scratch>/rust/target/release/lib...so does not exist

which reads as a build problem in the witness. The silent damage was that every later step in
the same pytest run then loaded an EP advertising zero devices:

    before the pin:  19 failed / 622 passed;  Criterion 10 gate exit 1, verdict reader exit 1,
                     `epctl --check-counters` exit 3 — reporting `counters ABI 4, but this
                     epctl understands 8`, which was the AGE of a snapshot left by an earlier
                     session, because the EP never ran and so never wrote a new one.
    after the pin:    8 failed / 633 passed;  all four of those steps exit 0.

One environment variable, thirteen reporters, and the message that named itself as the cause
named a version number instead.

WHAT THESE ASSERT
=================
Both are about the witness's *contract*, not about a run of it, so they are cheap and they
hold on every platform including ones with no Vulkan at all:

  1. the build environment pins `CARGO_TARGET_DIR` to the scratch tree, **even when the
     ambient environment points somewhere else** — the polarity that actually failed;
  2. the pinned directory is the one `artifact_paths()` reads, so the check that raises
     "cargo exited 0 but it does not exist" is looking where the build was told to write.

These are separate assertions on purpose. A witness that pins a directory nobody reads is
still broken, and it would still be broken silently.
"""

from __future__ import annotations

import os
from pathlib import Path

import _shaderless


def _build_env(monkeypatch, ambient: str | None) -> dict[str, str]:
    """The env `_shaderless` would hand to cargo, without running cargo.

    `build_shaderless_artifact` is not called: it compiles the crate twice and this question
    does not need a compiler. The environment construction is lifted verbatim instead, which
    means this test must be updated if that block moves — and that is the correct coupling,
    because the block IS the subject.
    """
    if ambient is None:
        monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    else:
        monkeypatch.setenv("CARGO_TARGET_DIR", ambient)
    env = dict(os.environ)
    env["ORT_INCLUDE_DIR"] = str(_shaderless.REPO / "third_party" / "onnxruntime" / "include")
    env["ONNXRUNTIME_EP_VULKAN_ALLOW_MISSING_GLSLC"] = "1"
    env["CARGO_TARGET_DIR"] = str((_shaderless.SCRATCH / "rust" / "target").resolve())
    return env


def test_the_shaderless_build_does_not_inherit_an_ambient_target_dir(monkeypatch) -> None:
    hostile = str(Path(os.sep) / "somewhere" / "else" / "shared-target")
    env = _build_env(monkeypatch, hostile)
    assert env["CARGO_TARGET_DIR"] != hostile, (
        "the shader-less build inherited CARGO_TARGET_DIR from the ambient environment. That is "
        "how a binary built to advertise zero devices got written over the shared release "
        "artifact on the Linux lane, and the run that discovered it reported the damage as "
        "`counters ABI 4 vs 8` four steps downstream."
    )
    assert Path(env["CARGO_TARGET_DIR"]).is_absolute()


def test_the_pinned_target_dir_is_the_one_the_witness_reads(monkeypatch) -> None:
    env = _build_env(monkeypatch, None)
    lib, epctl = _shaderless.artifact_paths()
    pinned = Path(env["CARGO_TARGET_DIR"]).resolve()
    for path in (lib, epctl):
        assert Path(path).resolve().parent == pinned / "release", (
            f"{path} is not under the pinned CARGO_TARGET_DIR {pinned}. A witness that builds "
            "into one directory and checks another reports 'cargo exited 0 but it does not "
            "exist' for a build that succeeded — which is the message this pin was added to "
            "stop, and it names the wrong subject."
        )


def test_the_source_of_the_pin_is_still_in_the_helper() -> None:
    """The two tests above lift the env block; this one asserts the block is still there.

    Without it, deleting the pin from `_shaderless.py` leaves both tests green, because they
    construct the environment themselves. That is the `BUILD_SKIPPED` shape: a check that
    passes because it stopped looking at the thing it is about.
    """
    text = Path(_shaderless.__file__).read_text(encoding="utf-8")
    assert 'env["CARGO_TARGET_DIR"]' in text, (
        "_shaderless.py no longer pins CARGO_TARGET_DIR. The tests in this file build the "
        "environment themselves and cannot see that; this assertion is the one that can."
    )
