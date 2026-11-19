# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import os
import shutil
import struct

import awkward as ak
import numpy as np
import skhep_testdata

import uproot
from uproot.writing._cascade import FreeSegments
from uproot.writing.writable import _collect_tree_basket_ranges, _validate_basket_key


def test_free_segments_merge_slices():
    assert FreeSegments._merge_slices((), ()) == ()
    assert FreeSegments._merge_slices(((10, 20),), ()) == ((10, 20),)
    assert FreeSegments._merge_slices((), [(10, 20)]) == ((10, 20),)

    # Adjacent intervals merge
    assert FreeSegments._merge_slices(((10, 20),), [(20, 30)]) == ((10, 30),)
    assert FreeSegments._merge_slices(((20, 30),), [(10, 20)]) == ((10, 30),)

    # Overlapping intervals merge
    assert FreeSegments._merge_slices(((10, 25),), [(20, 35)]) == ((10, 35),)

    # Subsumed intervals merge
    assert FreeSegments._merge_slices(((10, 50),), [(20, 30)]) == ((10, 50),)

    # Disjoint intervals stay separate and sorted
    assert FreeSegments._merge_slices(((10, 20),), [(30, 40)]) == ((10, 20), (30, 40))
    assert FreeSegments._merge_slices(((30, 40),), [(10, 20)]) == ((10, 20), (30, 40))

    # Empty or inverted intervals are ignored
    assert FreeSegments._merge_slices(((10, 20),), [(25, 25), (30, 25)]) == ((10, 20),)


def test_delete_ttree_frees_baskets_reproducer(tmp_path):
    filename = os.path.join(tmp_path, "leak.root")
    with uproot.recreate(filename) as f:
        f.mktree("keep", {"x": "float64"})
        f["keep"].extend({"x": np.zeros(10, dtype=np.float64)})

    sizes = []
    for _ in range(5):
        with uproot.update(filename) as f:
            t = f.mktree("t", {"x": "float64"})
            t.extend({"x": np.arange(100_000, dtype=np.float64)})
            del f["t"]
        sizes.append(os.path.getsize(filename))

    # Free segments table stabilizes after the first delete round
    assert sizes[1] == sizes[2] == sizes[3] == sizes[4]
    assert sizes[1] - sizes[0] < 200


def test_delete_ttree_with_multiple_branches(tmp_path):
    filename = os.path.join(tmp_path, "multibranch.root")
    with uproot.recreate(filename) as f:
        f.mktree("keep", {"k": "int32"})
        f["keep"].extend({"k": np.array([1, 2, 3], dtype=np.int32)})

    sizes = []
    for _ in range(3):
        with uproot.update(filename) as f:
            t = f.mktree("t", {"a": "int32", "b": "float64", "c": "int64"})
            t.extend(
                {
                    "a": np.arange(50_000, dtype=np.int32),
                    "b": np.arange(50_000, dtype=np.float64),
                    "c": np.arange(50_000, dtype=np.int64),
                }
            )
            del f["t"]
        sizes.append(os.path.getsize(filename))

    assert sizes[1] == sizes[2]
    assert sizes[1] - sizes[0] < 200


def test_delete_ttree_with_jagged_branches(tmp_path):
    filename = os.path.join(tmp_path, "jagged.root")
    with uproot.recreate(filename) as f:
        f.mktree("keep", {"k": "int32"})
        f["keep"].extend({"k": np.array([1], dtype=np.int32)})

    sizes = []
    for _ in range(3):
        with uproot.update(filename) as f:
            t = f.mktree("t", {"jag": ak.Array([[1, 2], [3], [4, 5, 6]])})
            t.extend({"jag": ak.Array([[7], [8, 9]] * 10_000)})
            del f["t"]
        sizes.append(os.path.getsize(filename))

    assert sizes[1] == sizes[2]
    assert sizes[1] - sizes[0] < 200


def test_delete_older_cycle_preserves_shared_baskets(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-physlite-rntuple_v1-0-0-0.root")
    test_path = os.path.join(tmp_path, "physlite_copy.root")
    shutil.copyfile(orig_path, test_path)

    # Deleting cycle 1 must not delete baskets shared with cycles 2 through 5
    with uproot.update(test_path) as f:
        del f["MetaData;1"]

    with uproot.open(test_path) as f:
        assert "MetaData;1" not in f
        assert "MetaData;2" in f
        m2 = f["MetaData;2"]
        assert m2.num_entries == 1

    # Delete remaining cycles until the last cycle frees all shared baskets
    with uproot.update(test_path) as f:
        for c in (2, 3, 4, 5):
            del f[f"MetaData;{c}"]

    with uproot.open(test_path) as f:
        assert "MetaData" not in f


def test_validate_basket_key(tmp_path):
    filename = os.path.join(tmp_path, "validate.root")
    with uproot.recreate(filename) as f:
        t = f.mktree("t", {"x": "int32"})
        t.extend({"x": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        key = f._cascading.data.get_key("t")
        tree = f._file._read_key(key, f)
        baskets = _collect_tree_basket_ranges(tree)
        assert len(baskets) == 1
        seek, stop = baskets[0]
        nbytes = stop - seek

        # Exact match succeeds
        assert _validate_basket_key(f._file.sink, seek, nbytes) is True
        # Wrong length fails
        assert _validate_basket_key(f._file.sink, seek, nbytes + 1) is False
        # Out of bounds fails safely
        assert _validate_basket_key(f._file.sink, 999_999, 100) is False
        # Negative seek fails safely
        assert _validate_basket_key(f._file.sink, -10, 100) is False


def test_delete_ttree_in_subdirectory(tmp_path):
    filename = os.path.join(tmp_path, "subdir.root")
    with uproot.recreate(filename) as f:
        d = f.mkdir("subdir")
        d.mktree("keep", {"k": "int32"})
        d["keep"].extend({"k": np.array([1, 2], dtype=np.int32)})

    sizes = []
    for _ in range(3):
        with uproot.update(filename) as f:
            d = f["subdir"]
            t = d.mktree("t", {"x": "float64"})
            t.extend({"x": np.arange(50_000, dtype=np.float64)})
            del d["t"]
        sizes.append(os.path.getsize(filename))

    assert sizes[1] == sizes[2]
    assert sizes[1] - sizes[0] < 200

    with uproot.open(filename) as f:
        assert "t" not in f["subdir"]
        assert "keep" in f["subdir"]


def test_delete_tree_unreadable_fallback(tmp_path, monkeypatch):
    filename = os.path.join(tmp_path, "fallback.root")
    with uproot.recreate(filename) as f:
        t = f.mktree("t", {"x": "int32"})
        t.extend({"x": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        # Simulate deserialization failure during del
        monkeypatch.setattr(
            f._file,
            "_read_key",
            lambda key, parent=None: (_ for _ in ()).throw(OSError("corrupt metadata")),
        )
        # del should fall back gracefully to freeing the key header without raising
        del f["t"]

    with uproot.open(filename) as f:
        assert "t" not in f


def test_free_segments_release_many_edge_cases(tmp_path):
    filename = os.path.join(tmp_path, "edges.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": "int32"})

    with uproot.update(filename) as f:
        fs = f._cascading.freesegments
        orig_slices = fs.data.slices

        # 1. Empty intervals list returns immediately
        fs.release_many([])
        assert fs.data.slices == orig_slices

        # 2. Intervals already included in free slices returns immediately
        fs.release_many(list(orig_slices))
        assert fs.data.slices == orig_slices

        # 3. Release when FreeSegments is not at the end
        fs._data.end = fs.data.end + 1000
        assert not fs.at_end

        # 3a. Releasing interval that merges into existing slice without growing slice records
        s0, s1 = orig_slices[0]
        fs.release_many([(s1, s1 + 50)])
        assert (s0, s1 + 50) in fs.data.slices

        # 3b. Releasing interval that forces FreeSegments to relocate to end
        fs.release_many([(s1 + 100, s1 + 200)])
        assert fs.at_end


def test_collect_tree_basket_ranges_edge_cases():
    # 1. Object without fBranches
    class NoBranches:
        def has_member(self, name):
            return False

    assert _collect_tree_basket_ranges(NoBranches()) == []

    # 2. Tree raising exception on fBranches
    class BrokenTree:
        def has_member(self, name):
            return True

        def member(self, name):
            raise RuntimeError("broken tree")

    assert _collect_tree_basket_ranges(BrokenTree()) == []

    # 3. Branch referencing external file via fFileName
    class ExternalBranch:
        def has_member(self, name):
            return name in ("fFileName", "fBasketSeek", "fBasketBytes")

        def member(self, name):
            if name == "fFileName":
                return "external.root"
            if name == "fBasketSeek":
                return [100]
            if name == "fBasketBytes":
                return [50]

    class TreeWithExternalBranch:
        def has_member(self, name):
            return name == "fBranches"

        def member(self, name):
            return [ExternalBranch()]

    assert _collect_tree_basket_ranges(TreeWithExternalBranch()) == []

    # 4. Branch that raises an exception during traversal
    class BrokenBranch:
        def has_member(self, name):
            raise RuntimeError("broken branch")

    class TreeWithBrokenBranch:
        def has_member(self, name):
            return name == "fBranches"

        def member(self, name):
            return [BrokenBranch()]

    assert _collect_tree_basket_ranges(TreeWithBrokenBranch()) == []


def test_validate_basket_key_edge_cases():
    # 1. 32-bit key header (fVersion <= 1000)
    class MockSink32:
        def read(self, seek, nbytes, insist=False):
            header = uproot.reading._key_format_small.pack(50, 4, 10, 0, 26, 1, seek, 0)
            data = header + b"\x07TBasket"
            return data[:nbytes]

    assert _validate_basket_key(MockSink32(), 100, 50) is True

    # 2. Truncated header (read returns fewer than header_size + 1)
    class MockSinkTruncatedHeader:
        def read(self, seek, nbytes, insist=False):
            if nbytes == 6:
                return struct.pack(">ih", 50, 1004)
            return b"short"

    assert _validate_basket_key(MockSinkTruncatedHeader(), 100, 50) is False

    # 3. Truncated class name string
    class MockSinkTruncatedClassname:
        def read(self, seek, nbytes, insist=False):
            if nbytes == 6:
                return struct.pack(">ih", 50, 1004)
            header = uproot.reading._key_format_big.pack(
                50, 1004, 10, 0, 34, 1, seek, 0
            )
            return header + b"\x10short"

    assert _validate_basket_key(MockSinkTruncatedClassname(), 100, 50) is False

    # 4. Read error raises exception
    class MockSinkError:
        def read(self, seek, nbytes, insist=False):
            raise OSError("io error")

    assert _validate_basket_key(MockSinkError(), 100, 50) is False


def test_delete_non_ttree_and_collection_failure(tmp_path, monkeypatch):
    filename = os.path.join(tmp_path, "mixed.root")
    with uproot.recreate(filename) as f:
        # Delete non-TTree object (histogram)
        f["h"] = (np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
        del f["h"]
        assert "h" not in f

        # Delete tree where _collect_tree_basket_ranges raises inside _del
        t = f.mktree("t", {"x": "int32"})
        t.extend({"x": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        monkeypatch.setattr(
            uproot.writing.writable,
            "_collect_tree_basket_ranges",
            lambda tree: (_ for _ in ()).throw(RuntimeError("forced failure")),
        )
        del f["t"]
        assert "t" not in f
