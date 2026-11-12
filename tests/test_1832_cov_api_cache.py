# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pickle

import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.cache import LRUArrayCache, LRUCache


def test_lru_order_and_views():
    cache = LRUCache(3)
    assert (cache.limit, cache.current) == (3, 0)

    cache["a"], cache["b"], cache["c"] = 1, 2, 3
    assert cache["a"] == 1  # "a" is now the most recently used

    assert cache.keys() == ["b", "c", "a"]
    assert cache.values() == [2, 3, 1]
    assert cache.items() == [("b", 2), ("c", 3), ("a", 1)]
    assert list(cache) == ["b", "c", "a"]
    assert len(cache) == 3
    assert cache.current == 3

    cache["d"] = 4  # evicts "b", the least recently used
    assert cache.items() == [("c", 3), ("a", 1), ("d", 4)]
    assert cache.current == 3


def test_lru_replacing_and_deleting_keep_current_in_step():
    cache = LRUCache(2)
    cache["a"] = 1
    cache["b"] = 2

    cache["a"] = 10  # replacing moves "a" to the end without counting it twice
    assert cache.items() == [("b", 2), ("a", 10)]
    assert cache.current == 2

    del cache["b"]
    assert cache.items() == [("a", 10)]
    assert cache.current == 1
    with pytest.raises(KeyError, match="b"):
        del cache["b"]
    assert cache.current == 1


def test_lru_replacing_a_key_protects_it_from_eviction():
    cache = LRUCache(3)
    cache["a"], cache["b"], cache["c"] = 1, 2, 3
    cache["a"] = 99
    cache["d"] = 4  # evicts "b", now the least recently used
    assert cache.items() == [("c", 3), ("a", 99), ("d", 4)]


def test_lru_without_limit_never_evicts():
    cache = LRUCache(None)
    assert cache.limit is None
    for i in range(100):
        cache[i] = i
    assert len(cache) == cache.current == 100
    assert repr(cache).startswith("<LRUCache (no limit) at 0x")


def test_lru_repr_shows_how_full_it_is():
    cache = LRUCache(5)
    cache["a"] = 1
    assert repr(cache).startswith("<LRUCache (1/5 full) at 0x")


def test_lru_pickle_keeps_the_limit_but_not_the_data():
    cache = LRUCache(5)
    cache["a"] = 1
    out = pickle.loads(pickle.dumps(cache))
    assert out.limit == 5
    assert (out.current, out.keys()) == (0, [])
    out["b"] = 2  # the lock has been rebuilt
    assert out.items() == [("b", 2)]


def test_array_cache_evicts_by_bytes():
    cache = LRUArrayCache("24 B")
    assert (cache.limit, cache.current) == (24, 0)

    cache["a"] = np.zeros(2)  # 16 bytes
    cache["b"] = np.zeros(1)  # 8 bytes
    assert cache.current == 24
    assert cache.keys() == ["a", "b"]

    cache["c"] = np.zeros(2)  # needs 16 bytes: "a" goes, and "b" can stay
    assert cache.keys() == ["b", "c"]
    assert cache.current == 24

    cache["d"] = np.zeros(4)  # too big to keep, even in an empty cache
    assert (cache.keys(), cache.current) == ([], 0)


def test_array_cache_repr():
    limited = LRUArrayCache(100)
    limited["a"] = np.zeros(1)
    assert repr(limited).startswith("<LRUArrayCache (8/100 bytes full) at 0x")

    assert LRUArrayCache(None).limit is None
    assert repr(LRUArrayCache(None)).startswith("<LRUArrayCache (no limit) at 0x")


class _MemoryUsage:
    def __init__(self, usage):
        self.usage = usage

    def memory_usage(self):
        return self.usage


@pytest.mark.parametrize(
    ("what", "expected"),
    [
        (np.zeros(3, np.int32), 12),
        (_MemoryUsage(np.array([10, 20, 30])), 60),
        (_MemoryUsage(60), LRUArrayCache.default_nbytes),
        ("no size at all", LRUArrayCache.default_nbytes),
    ],
)
def test_array_cache_sizeof(what, expected):
    assert LRUArrayCache.sizeof(what) == expected


def test_array_cache_sizeof_pandas():
    pandas = pytest.importorskip("pandas")
    series = pandas.Series(np.zeros(4))
    assert LRUArrayCache.sizeof(series) == 32
    df = pandas.DataFrame({"x": np.zeros(4), "y": np.zeros(4, np.int32)})
    assert not hasattr(df, "nbytes")
    assert LRUArrayCache.sizeof(df) == df.memory_usage().sum()


def test_array_cache_in_use_returns_the_stored_array():
    cache = LRUArrayCache("1 MB")
    with uproot.open(
        skhep_testdata.data_path("uproot-Zmumu.root"), array_cache=cache
    ) as f:
        first = f["events/M"].array(library="np")
        assert cache.current == first.nbytes
        second = f["events/M"].array(library="np")
    assert second is first
    assert first[:2].tolist() == pytest.approx([82.4626915551, 83.6262040052])
