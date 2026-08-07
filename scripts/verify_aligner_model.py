"""Prove the ONNX aligner we run is the model its author published.

Subtitles are timed by a **pre-converted** ONNX build of MMS-300m
(`onnx-community/mms-300m-1130-forced-aligner-ONNX`), pinned by revision, so that
serving alignments never requires PyTorch. That convenience comes with a question
the pin alone cannot answer: a third party did the PyTorch → ONNX conversion, and
"the bytes are equivalent to the author's own weights" is a claim, not a given.

This script turns it into a measurement. It loads the author's checkpoint
(`MahmoudAshraf/mms-300m-1130-forced-aligner`) with PyTorch, runs both models over
the same audio, and reports how far apart they are — both in raw logits and, the
part that actually decides an alignment, in which token wins each frame.

Argmax agreement is the figure to cite. Two models can differ in the last decimal
of every logit and still produce byte-identical alignments, because forced
alignment only ever compares tokens within a frame; conversely a single flipped
argmax is a real difference. At the time of writing the two agree on **100.00%** of
frames, with max absolute logit differences around 1e-4 on noise and 2e-3 on real
parliamentary audio.

PyTorch is needed here and nowhere else in the pipeline, so it is installed
ad-hoc rather than declared — the same arrangement as the eval harness's local
reranker:

    uv run --with torch --with transformers --with onnxscript --no-sync \\
      python scripts/verify_aligner_model.py [audio.wav]

With no audio argument only synthetic noise is compared, which is enough to detect
different weights but not representative of real speech; pass a 16 kHz mono wav
(any intervention's audio) for the second, more meaningful comparison.
"""

import sys
import wave

import numpy as np

TORCH_REPO = "MahmoudAshraf/mms-300m-1130-forced-aligner"
ONNX_REPO = "onnx-community/mms-300m-1130-forced-aligner-ONNX"
ONNX_REVISION = "2100fb247d8e"
ONNX_FILE = "onnx/model.onnx"
SAMPLE_RATE = 16000


def _load_wav(path):
    with wave.open(path, "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise SystemExit(
                f"{path}: expected 16 kHz mono, got {handle.getframerate()} Hz "
                f"/ {handle.getnchannels()} channel(s)")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _normalise(samples):
    """The feature extractor's own normalisation (``do_normalize: true``)."""
    return ((samples - samples.mean()) / (samples.std() + 1e-7)
            )[None, :].astype(np.float32)


def _report(label, ours, theirs):
    difference = np.abs(ours - theirs)
    agreement = (ours.argmax(-1) == theirs.argmax(-1)).mean() * 100
    print(f"  {label:34} max {difference.max():.3e}  mean {difference.mean():.3e}"
          f"  argmax agree {agreement:6.2f}%")
    return agreement


def main():
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC

    print(f"author's weights : {TORCH_REPO}")
    print(f"converted ONNX   : {ONNX_REPO}@{ONNX_REVISION}")

    model = Wav2Vec2ForCTC.from_pretrained(TORCH_REPO).eval()
    onnx_path = hf_hub_download(ONNX_REPO, ONNX_FILE, revision=ONNX_REVISION)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"parameters       : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
          f", vocab {model.config.vocab_size}")

    cases = [("synthetic noise (5 s)", np.random.default_rng(0).standard_normal(
        SAMPLE_RATE * 5).astype(np.float32))]
    if len(sys.argv) > 1:
        samples = _load_wav(sys.argv[1])[:SAMPLE_RATE * 30]
        cases.append((f"real audio ({len(samples) / SAMPLE_RATE:.0f} s)", samples))

    agreements = []
    for label, samples in cases:
        inputs = _normalise(samples)
        with torch.no_grad():
            reference = model(torch.from_numpy(inputs)).logits.numpy()
        converted = session.run(None, {"input_values": inputs})[0]
        print(f"\n{label}: {reference.shape[1]} frames")
        agreements.append(_report("converted vs author's weights",
                                  converted, reference))

    print()
    if min(agreements) == 100.0:
        print("EQUIVALENT: the conversion decides every frame exactly as the "
              "author's weights do.")
    else:
        print(f"DIVERGENT: argmax agreement fell to {min(agreements):.2f}% — the "
              "conversion is NOT interchangeable with the author's weights.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
