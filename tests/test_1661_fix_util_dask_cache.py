# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for PR #1661: utility-layer bugs (steps leakage, dask error
paths, LRU cache) and Py2-era cleanup.

Covers:
- maybe_steps leakage via uproot._util.regularize_files (finding 1)
- damerau_levenshtein distance values (finding 2)
- LRU cache under concurrent use (finding 7); its ordering is in
  tests/test_1832_cov_api_cache.py
"""

import threading

import pytest

import uproot
import uproot._util
from uproot.cache import LRUCache

# ---------------------------------------------------------------------------
# Finding 1: maybe_steps leakage in regularize_files
# ---------------------------------------------------------------------------


def test_regularize_files_steps_no_leakage():
    """
    When a dict value with 'steps' is followed by a plain-string value,
    the second entry must NOT inherit the first entry's steps.
    """
    import skhep_testdata

    path_a = skhep_testdata.data_path("uproot-HZZ.root")
    path_b = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")

    result = uproot._util.regularize_files(
        {
            path_a: {"object_path": "events", "steps": [0, 10, 100]},
            path_b: "sample",
        },
        steps_allowed=True,
    )
    # Only the first entry carries steps; the second must NOT inherit them
    assert result == [
        (path_a, "events", [[0, 10], [10, 100]]),
        (path_b, "sample"),
    ]


# ---------------------------------------------------------------------------
# Finding 2: damerau_levenshtein correct values
# ---------------------------------------------------------------------------


def test_damerau_levenshtein_transposition():
    d = uproot._util.damerau_levenshtein
    # Adjacent transpositions cost 1
    assert d("ab", "ba") == 1
    assert d("abcd", "abdc") == 1
    # Identical strings cost 0
    assert d("abc", "abc") == 0
    # Standard edit distances (with this implementation's cost model)
    assert d("a", "b") == 2  # different char → cost 2
    assert d("", "abc") == 3  # 3 insertions
    assert d("abc", "") == 3  # 3 deletions


def test_damerau_levenshtein_no_ratio_param():
    """Confirm the broken ratio parameter was removed."""
    import inspect

    sig = inspect.signature(uproot._util.damerau_levenshtein)
    assert "ratio" not in sig.parameters


# ---------------------------------------------------------------------------
# Finding 7: LRU cache correctness
# ---------------------------------------------------------------------------


def test_lru_cache_thread_safety():
    """Concurrent gets and sets should neither raise nor lose track of the contents."""
    c = LRUCache(50)
    errors = []

    def writer():
        try:
            for i in range(200):
                c[i] = i * 2
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for i in range(200):
                _ = c.get(i % 50)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"Thread-safety errors: {errors}"
    # all 200 keys were written, so the cache is full, and nothing got mismatched
    assert len(c) == c.current == 50
    assert all(value == key * 2 for key, value in c.items())
