"""Unit tests for the pure passage chunker — no I/O."""

import pytest

from qhld_engine.domain.speeches.chunking import chunk_text

pytestmark = pytest.mark.unit


def test_empty_text_yields_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("Hola. Adiós.", 100, 10) == ["Hola. Adiós."]


def test_splits_on_sentence_boundaries_with_overlap():
    text = "AAAA. BBBB. CCCC. DDDD."  # four 5-char sentences
    chunks = chunk_text(text, target_chars=12, overlap_chars=5)

    assert len(chunks) > 1
    # every chunk is made of whole sentences (never a mid-sentence cut)
    allowed = {"AAAA.", "BBBB.", "CCCC.", "DDDD."}
    for chunk in chunks:
        assert set(chunk.split(" ")) <= allowed
    # overlap carries a sentence from one chunk into the next
    assert any(
        set(a.split(" ")) & set(b.split(" "))
        for a, b in zip(chunks, chunks[1:])
    )


def test_over_long_sentence_becomes_its_own_chunk():
    long_sentence = "B" * 40
    text = f"AA. {long_sentence}. CC."
    chunks = chunk_text(text, target_chars=10, overlap_chars=0)

    assert f"{long_sentence}." in chunks
