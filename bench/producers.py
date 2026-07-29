"""Producer provenance — a benchmark artefact is relative to its producer, not to a model family.

Mouse established this for correctness in ``OP_COVERAGE.md`` §4.18: **op coverage is relative to
a producer, not to a model architecture.** Justin's own ``onnx-genai-models`` builder (``mobius``)
emits ``ai.onnx::Attention`` @ opset 23, ``ai.onnx::RMSNormalization`` and
``ai.onnx::RotaryEmbedding`` for a Qwen3 decoder layer, where the ORT GenAI builder emits the
``com.microsoft`` contrib equivalents. ``MatMulNBits`` is the only op both toolchains agree on.

The same is true of a *measurement*, and it is easier to be fooled by. If ``cases.py`` builds a
graph with one exporter and the result table says "Qwen3", the number is about that exporter's
graph — its op set, its fusion decisions, its KV-cache layout — and not about Qwen3. A reader who
compares it against someone else's "Qwen3" number is comparing two different programs. Worse, the
two producers can differ in whether an op is *claimable by our EP at all*: a graph of standard-domain
``RMSNormalization`` partitions differently from one of ``com.microsoft::SimplifiedLayerNormalization``,
so "the EP claimed 40% of the graph" is a statement about the producer as much as about the EP.

So provenance is not a comment here. Two things are structural:

1. Every :class:`~cases.Case` carries a :class:`Producer`, and it is recorded in the result JSON
   and in the environment record next to device, driver, OS and build flags.
2. **A case may not carry a model-family name unless a model producer is recorded for it.**
   :func:`assert_family_label_is_earned` raises at case-construction time, so a case named
   ``qwen3_decoder_layer`` built by the synthetic op-builder cannot be constructed at all. This
   is the same instinct as the ``Phase::Submit`` assertion: make the misleading thing impossible
   rather than discouraged.

Producer identity includes a **content digest**, not just a name. A builder that changed between
two runs is a different producer, for exactly the reason a driver update makes a different device
(``devices.py``): the graph changed, and attributing the change to the EP would be wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: Words that name a model family. A case whose name or tags contain one of these is claiming to
#: say something about that family, which requires a model producer to have built it.
#:
#: Deliberately over-broad: a false positive costs one explicit ``model_family=`` argument, and a
#: false negative ships a number labelled with a family nobody built.
MODEL_FAMILY_WORDS = frozenset(
    {
        "bert",
        "clip",
        "deepseek",
        "gemma",
        "gpt",
        "gpt2",
        "llama",
        "mistral",
        "mixtral",
        "phi",
        "qwen",
        "qwen2",
        "qwen3",
        "resnet",
        "sd",
        "smollm",
        "stablediffusion",
        "t5",
        "whisper",
        "yolo",
    }
)

#: Producer kinds. ``op`` builders make synthetic single-op or few-op graphs and can never earn a
#: model-family label; ``model`` builders export a real architecture and can.
KIND_OP = "op"
KIND_MODEL = "model"


class ProducerProvenanceError(ValueError):
    """Raised when a case would claim more provenance than it has.

    This is an error, not a warning, and it fires during case construction — before any timing
    happens — so a mislabelled case cannot reach a result file.
    """


@dataclass(frozen=True)
class Producer:
    """Who built the graph that was benchmarked, and at what version.

    ``digest`` is what makes this more than a label: it is a hash of the builder's own source (or
    of an explicitly supplied identity string), so a silent edit to the builder shows up as a
    different producer rather than as a performance change.
    """

    #: Short stable id, e.g. ``tests/ops/_models.py`` or ``onnx-genai-models/mobius``.
    name: str
    #: ``op`` or ``model`` — see :data:`KIND_OP` / :data:`KIND_MODEL`.
    kind: str
    #: Version of the producing toolchain. ``None`` is allowed only for ``op`` producers, where
    #: ``digest`` carries the identity; a ``model`` producer with no version cannot earn a family
    #: label.
    version: "str | None" = None
    #: Content digest of the builder source, or of an explicit identity string.
    digest: str = ""
    #: Opset imports the producer emits, ``{domain: version}``. ``""`` is the default (ai.onnx)
    #: domain. This is the field that distinguishes mobius from the ORT GenAI builder.
    opsets: "dict[str, int]" = field(default_factory=dict)
    #: The model family this producer exported, when it exported one. ``None`` for op builders.
    model_family: "str | None" = None
    notes: str = ""

    @property
    def fingerprint(self) -> str:
        """Identity for comparison purposes: name / version / digest.

        Two results whose case fingerprints differ are not comparable, in the same way and for
        the same reason as two results from different devices.
        """
        return f"{self.name}@{self.version or '-'}#{self.digest[:12] or '-'}"

    @property
    def can_claim_model_family(self) -> bool:
        """Whether this producer may put a model-family name on a case.

        Requires all three: it is a model exporter, it says which family, and it says which
        version of itself. A family label from an unversioned exporter is not reproducible.
        """
        return self.kind == KIND_MODEL and bool(self.model_family) and bool(self.version)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "digest": self.digest,
            "opsets": dict(self.opsets),
            "model_family": self.model_family,
            "fingerprint": self.fingerprint,
            "notes": self.notes,
        }

    def summary(self) -> str:
        ops = ", ".join(f"{d or 'ai.onnx'}={v}" for d, v in sorted(self.opsets.items()))
        return f"{self.fingerprint}" + (f" [{ops}]" if ops else "")


def digest_of_paths(paths: "list[Path]") -> str:
    """SHA-256 over the contents of ``paths``, in the order given.

    Missing files contribute their name and a marker rather than raising: a producer whose source
    we cannot read is still identifiable as *that* producer, and the missing-source fact is itself
    part of the identity.
    """
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


def digest_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_REPO = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def op_builder() -> Producer:
    """The synthetic op-graph builder the correctness tests use (``tests/ops/_models.py``).

    This is the only producer the harness has today. It builds single-op and few-op graphs, so it
    is honest about elementwise and ``MatMulNBits`` shapes and says **nothing** about any model
    family — which is why :func:`assert_family_label_is_earned` will reject a Qwen3-labelled case
    built by it.
    """
    src = _REPO / "tests" / "ops" / "_models.py"
    return Producer(
        name="tests/ops/_models.py",
        kind=KIND_OP,
        version=None,
        digest=digest_of_paths([src]),
        opsets={},
        model_family=None,
        notes=(
            "Synthetic single-op graphs shared with the correctness tests, so a benchmark cannot "
            "drift from what is tested. Says nothing about any model family."
        ),
    )


def mobius(version: str, *, opsets: "dict[str, int] | None" = None,
           model_family: str = "qwen3", notes: str = "") -> Producer:
    """Justin's ``onnx-genai-models`` builder (``mobius``).

    Emits **standard-domain** ops for a Qwen3 decoder layer — ``ai.onnx::Attention`` @ opset 23,
    ``ai.onnx::RMSNormalization``, ``ai.onnx::RotaryEmbedding`` — where the ORT GenAI builder emits
    the ``com.microsoft`` contrib equivalents (``OP_COVERAGE.md`` §4.18). Mouse also notes it
    avoids ``seqlens_k`` indirection and in-place KV-cache aliasing, which is why it is the path
    most likely to produce a model we can actually build and iterate on locally.

    ``version`` is required and has no default: an unversioned exporter cannot earn a family label
    (:attr:`Producer.can_claim_model_family`), because the graph it emits changes with it.
    """
    return Producer(
        name="onnx-genai-models/mobius",
        kind=KIND_MODEL,
        version=version,
        digest=digest_of_text(f"onnx-genai-models/mobius@{version}"),
        opsets=opsets or {"": 23},
        model_family=model_family,
        notes=notes or (
            "Standard-domain op set: ai.onnx::Attention@23, RMSNormalization, RotaryEmbedding. "
            "Not interchangeable with the ORT GenAI builder's com.microsoft graph."
        ),
    )


def ort_genai_builder(version: str, *, opsets: "dict[str, int] | None" = None,
                      model_family: str = "qwen3", notes: str = "") -> Producer:
    """The ORT GenAI model builder (``onnxruntime-genai``, ``builder.py``).

    Emits the ``com.microsoft`` contrib graph: ``GroupQueryAttention``,
    ``SimplifiedLayerNormalization``, ``RotaryEmbedding`` in the contrib domain, with ``seqlens_k``
    indirection and in-place KV-cache aliasing. A number from this producer and a number from
    :func:`mobius` for "the same model" are numbers about two different graphs.
    """
    return Producer(
        name="onnxruntime-genai/builder.py",
        kind=KIND_MODEL,
        version=version,
        digest=digest_of_text(f"onnxruntime-genai/builder.py@{version}"),
        opsets=opsets or {"": 21, "com.microsoft": 1},
        model_family=model_family,
        notes=notes or (
            "com.microsoft contrib graph (GroupQueryAttention, SimplifiedLayerNormalization). "
            "Not interchangeable with the mobius standard-domain graph."
        ),
    )


def family_words_in(*texts: str) -> "list[str]":
    """Model-family words appearing as whole tokens in ``texts``.

    Tokenised on non-alphanumerics so ``qwen3_decoder_layer`` matches ``qwen3`` while
    ``matmulnbits`` does not accidentally match anything.
    """
    found: "list[str]" = []
    for text in texts:
        token = ""
        for ch in (text or "").lower() + " ":
            if ch.isalnum():
                token += ch
            else:
                if token in MODEL_FAMILY_WORDS and token not in found:
                    found.append(token)
                token = ""
    return found


def assert_family_label_is_earned(name: str, tags: "list[str]", producer: Producer) -> None:
    """Refuse to construct a case that names a model family it cannot speak for.

    Raises :class:`ProducerProvenanceError` when ``name`` or ``tags`` contain a model-family word
    and ``producer`` is not a versioned exporter of that family. The failure is at construction
    time and is fatal — the whole point is that such a case never reaches a result file, never
    gets screenshotted, and never gets quoted.
    """
    words = family_words_in(name, *(tags or []))
    if not words:
        return
    if not producer.can_claim_model_family:
        raise ProducerProvenanceError(
            f"case {name!r} names the model family/families {words} but its producer "
            f"{producer.fingerprint!r} (kind={producer.kind!r}, version={producer.version!r}, "
            f"model_family={producer.model_family!r}) cannot speak for a model family.\n"
            f"A benchmark artefact is relative to its producer (OP_COVERAGE.md §4.18): a "
            f"synthetic op graph named after an architecture reports that graph's cost, not the "
            f"architecture's. Either rename the case after what it actually builds, or build it "
            f"with a versioned model exporter (producers.mobius(...) / "
            f"producers.ort_genai_builder(...))."
        )
    if producer.model_family and producer.model_family.lower() not in words:
        raise ProducerProvenanceError(
            f"case {name!r} names {words} but its producer exported "
            f"{producer.model_family!r}. A case must be named after the family that was actually "
            f"built."
        )


def describe(producers: "list[Producer]") -> str:
    """Human summary for the environment banner."""
    if not producers:
        return "producer: none recorded"
    return "\n".join(f"producer: {p.summary()}" for p in producers)


if __name__ == "__main__":  # pragma: no cover - manual use
    print(describe([op_builder(), mobius("0.0.0-example"), ort_genai_builder("0.0.0-example")]))
