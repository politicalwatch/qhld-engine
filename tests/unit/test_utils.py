"""Unit tests for ``utils`` — DB-free pure helpers."""

import hashlib

import pytest

from utils import generateId

pytestmark = pytest.mark.unit


def test_generate_id_matches_sha1_of_joined_args():
    # generateId joins its string args and returns their sha1 hex digest.
    expected = hashlib.sha1("abc".encode("utf-8")).hexdigest()
    assert generateId("a", "b", "c") == expected


def test_generate_id_single_arg():
    expected = hashlib.sha1("hello".encode("utf-8")).hexdigest()
    assert generateId("hello") == expected


def test_generate_id_is_deterministic():
    assert generateId("x", "y") == generateId("x", "y")


def test_generate_id_returns_error_sentinel_on_bad_input():
    # Non-string args break the u''.join(...) and hit the bare-except path.
    assert generateId(None) == "ID_ERROR"
    assert generateId(1, 2) == "ID_ERROR"
