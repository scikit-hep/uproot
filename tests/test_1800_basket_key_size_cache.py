# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
The uncompressed basket sizes cached on a TBranch are read only once, stay
compact, are not pickled, and do not hide that a file is closed.
"""

from __future__ import annotations

import pickle

import numpy
import pytest
import skhep_testdata

import uproot


def _sample_path():
    return skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")


def test_partially_cached_branch_reads_only_missing_keys(monkeypatch):
    with uproot.open(_sample_path())["sample"] as tree:
        expected = tree.num_entries_for("1 kB")

    with uproot.open(_sample_path())["sample"] as tree:
        branches = list(tree.itervalues(recursive=True))
        missing = []
        for branch in branches:
            assert branch._num_normal_baskets > 1
            for num in range(branch._num_normal_baskets):
                if num % 2 == 0:
                    branch.basket_uncompressed_bytes(num)
                else:
                    missing.append(branch._basket_key_range(num))

        source = tree.file.source
        fetched = []
        original_chunks = source.chunks

        def chunks(ranges, notifications, **kwargs):
            fetched.append(ranges)
            return original_chunks(ranges, notifications, **kwargs)

        monkeypatch.setattr(source, "chunks", chunks)

        assert tree.num_entries_for("1 kB") == expected
        assert len(fetched) == 1
        assert sorted(fetched[0]) == sorted(missing)


def test_cache_is_a_lazily_allocated_int64_array():
    with uproot.open(_sample_path())["sample"] as tree:
        branch = next(tree.itervalues())
        assert branch._uncompressed_bytes_cache is None

        size = branch.basket_uncompressed_bytes(1)
        assert isinstance(size, int)

        cache = branch._uncompressed_bytes_cache
        assert cache.dtype == numpy.int64
        assert len(cache) == branch._num_normal_baskets
        assert cache[1] == size
        assert (numpy.delete(cache, 1) == -1).all()


def test_cache_is_not_pickled():
    path = skhep_testdata.data_path("uproot-HZZ.root")
    with uproot.open(path)["events"] as tree:
        expected = tree.num_entries_for("1 kB")
        assert next(tree.itervalues())._uncompressed_bytes_cache is not None

        # A pickle from before the cache existed has no such attribute either
        unpickled = pickle.loads(pickle.dumps(tree))
        for branch in unpickled.itervalues(recursive=True):
            assert "_uncompressed_bytes_cache" not in branch.__dict__
            assert branch._uncompressed_bytes_cache is None
        assert unpickled.num_entries_for("1 kB") == expected
        unpickled.close()


@pytest.mark.parametrize("read_before_closing", [False, True])
def test_closed_file_raises_regardless_of_cache(read_before_closing):
    tree = uproot.open(_sample_path())["sample"]
    branch = next(tree.itervalues())
    if read_before_closing:
        tree.num_entries_for("1 kB")
    tree.file.close()

    with pytest.raises(OSError, match="is closed"):
        tree.num_entries_for("1 kB")
    with pytest.raises(OSError, match="is closed"):
        branch.basket_uncompressed_bytes(0)
