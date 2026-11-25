# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import copy
import os
import struct

import awkward as ak
import numpy as np
import pytest

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

    # Overlapping intervals raise RuntimeError
    with pytest.raises(RuntimeError, match="overlaps one already marked as free"):
        FreeSegments._merge_slices(((10, 25),), [(20, 35)])

    # Subsumed intervals raise RuntimeError
    with pytest.raises(RuntimeError, match="overlaps one already marked as free"):
        FreeSegments._merge_slices(((10, 50),), [(20, 30)])

    # Overlapping intervals in the input list raise RuntimeError
    with pytest.raises(RuntimeError, match="overlaps another segment to release"):
        FreeSegments._merge_slices((), [(10, 25), (20, 35)])

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
    filename = os.path.join(tmp_path, "multicycle.root")
    with uproot.recreate(filename) as f:
        t = f.mktree("t", {"x": "int32"})
        t.extend({"x": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        orig_key = f._cascading.data.get_key("t", 1)
        c2 = copy.copy(orig_key)
        c2._cycle = 2
        f._cascading.data.add_key(c2)
        f._cascading.write(f._file.sink)

    # Deleting cycle 1 must not delete baskets shared with cycle 2
    with uproot.update(filename) as f:
        del f["t;1"]

    with uproot.open(filename) as f:
        assert "t;1" not in f
        assert "t;2" in f
        assert f["t;2"]["x"].array().tolist() == list(range(10))

    # Deleting the last cycle frees all baskets
    with uproot.update(filename) as f:
        del f["t;2"]

    with uproot.open(filename) as f:
        assert "t" not in f


def test_delete_tree_with_alias_preserves_baskets(tmp_path):
    filename = os.path.join(tmp_path, "alias.root")
    with uproot.recreate(filename) as f:
        t = f.mktree("t", {"x": "int32"})
        t.extend({"x": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        orig_key = f._cascading.data.get_key("t", 1)
        alias_key = copy.copy(orig_key)
        alias_key._name = uproot.writing._cascade.String(None, "alias")
        f._cascading.data.add_key(alias_key)
        f._cascading.write(f._file.sink)

    # Deleting alias key must not corrupt or free t's baskets
    with uproot.update(filename) as f:
        del f["alias"]

    with uproot.open(filename) as f:
        assert "alias" not in f
        assert "t" in f
        assert f["t"]["x"].array().tolist() == list(range(10))

    # Deleting t then frees the shared baskets
    with uproot.update(filename) as f:
        del f["t"]

    with uproot.open(filename) as f:
        assert "t" not in f


def test_release_overlapping_segment_raises_error(tmp_path):
    filename = os.path.join(tmp_path, "overlap.root")
    with uproot.recreate(filename) as f:
        f["h"] = (np.arange(10.0), np.arange(11.0))
        f["k"] = (np.arange(10.0), np.arange(11.0))
        key = f._cascading.data.get_key("h", None)
        start = key.seek_location
        stop = key.seek_location + key.num_bytes + key.compressed_bytes
        del f["h"]
        with pytest.raises(RuntimeError, match="overlaps one already marked as free"):
            f._cascading.freesegments.release(start, stop)


def test_delete_ntupled_frees_baskets(tmp_path):
    filename = os.path.join(tmp_path, "ntd.root")
    with uproot.recreate(filename) as f:
        t = f.mktree("nt", {"x": "int32"})
        t.extend({"x": np.arange(1000, dtype=np.int32)})

    with uproot.update(filename) as f:
        key = f._cascading.data.get_key("nt", 1)
        key._classname = uproot.writing._cascade.String(None, "TNtupleD")
        f._cascading.write(f._file.sink)

    with uproot.update(filename) as f:
        del f["nt"]
        slices = f._cascading.freesegments._data.slices
        # Both key header and basket segment must be freed
        assert len(slices) >= 2


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

        # 1b. Non-empty intervals with zero or negative length return immediately
        fs.release_many([(100, 100), (200, 150)])
        assert fs.data.slices == orig_slices

        # 2. Intervals already overlapping with free slices raise RuntimeError
        with pytest.raises(RuntimeError, match="overlaps one already marked as free"):
            fs.release_many(list(orig_slices))

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

    # 5. Branch with nested sub-branches
    class SubBranch:
        def has_member(self, name):
            return name in ("fBasketSeek", "fBasketBytes")

        def member(self, name):
            if name == "fBasketSeek":
                return [200]
            if name == "fBasketBytes":
                return [60]

    class ParentBranch:
        def has_member(self, name):
            return name in ("fBranches", "fBasketSeek", "fBasketBytes")

        def member(self, name):
            if name == "fBranches":
                return [SubBranch()]
            if name == "fBasketSeek":
                return []
            if name == "fBasketBytes":
                return []

    class TreeWithSubBranches:
        def has_member(self, name):
            return name == "fBranches"

        def member(self, name):
            return [ParentBranch()]

    assert _collect_tree_basket_ranges(TreeWithSubBranches()) == [(200, 260)]

    # 6. Branch without fBasketSeek
    class BranchWithoutBaskets:
        def has_member(self, name):
            return False

    class TreeWithBranchWithoutBaskets:
        def has_member(self, name):
            return name == "fBranches"

        def member(self, name):
            return [BranchWithoutBaskets()]

    assert _collect_tree_basket_ranges(TreeWithBranchWithoutBaskets()) == []


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


def test_is_ttree_classname_streamer_hierarchy():
    from uproot.writing.writable import _is_ttree_classname

    class MockBase(uproot.streamers.Model_TStreamerBase):
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name

    class MockStreamer:
        def __init__(self, name, bases):
            self.name = name
            self.elements = [MockBase(b) for b in bases]

    assert _is_ttree_classname("TTree") is True
    assert _is_ttree_classname("Other", None) is False

    # Inheritance chain: CustomTree inherits from IntermediateTree inherits from TTree
    streamers = [
        object(),
        MockStreamer("CustomTree", ["IntermediateTree"]),
        MockStreamer("IntermediateTree", ["TTree"]),
    ]
    assert _is_ttree_classname("CustomTree", streamers) is True

    # Cycle or duplicate base in hierarchy
    streamers_cycle = [
        MockStreamer("CycleTree", ["BaseA", "BaseA"]),
        MockStreamer("BaseA", ["NotTree"]),
    ]
    assert _is_ttree_classname("CycleTree", streamers_cycle) is False


def test_delete_tree_with_broken_surviving_tree(tmp_path, monkeypatch):
    filename = os.path.join(tmp_path, "surviving_broken.root")
    with uproot.recreate(filename) as f:
        t1 = f.mktree("t1", {"x": "int32"})
        t1.extend({"x": np.arange(10, dtype=np.int32)})
        t2 = f.mktree("t2", {"y": "int32"})
        t2.extend({"y": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        orig_read_key = f._file._read_key

        def mock_read_key(key, directory, **kwargs):
            if key.name.string == "t2":
                raise OSError("corrupt surviving tree")
            return orig_read_key(key, directory, **kwargs)

        monkeypatch.setattr(f._file, "_read_key", mock_read_key)
        del f["t1"]
        assert "t1" not in f
        assert "t2" in f
        # Writing new data must not overwrite t2 baskets because basket freeing failed closed
        f["h"] = (np.full(10, 5.0), np.linspace(0, 1, 11))

    with uproot.open(filename) as f:
        assert f["t2"]["y"].array().tolist() == list(range(10))


def test_delete_tree_with_broken_subdirectory_fails_closed(tmp_path, monkeypatch):
    filename = os.path.join(tmp_path, "broken_subdir.root")
    with uproot.recreate(filename) as f:
        t1 = f.mktree("t1", {"x": "int32"})
        t1.extend({"x": np.arange(10, dtype=np.int32)})
        f.mkdir("sub")

    with uproot.update(filename) as f:
        orig_subdir = f._subdir

        def mock_subdir(key):
            raise OSError("broken directory")

        monkeypatch.setattr(f, "_subdir", mock_subdir)
        del f["t1"]
        monkeypatch.setattr(f, "_subdir", orig_subdir)
        assert "t1" not in f
        f["h"] = (np.full(10, 5.0), np.linspace(0, 1, 11))


def test_tree_basket_seeks_cache_and_invalidation(tmp_path):
    filename = os.path.join(tmp_path, "cache_invalidation.root")
    with uproot.recreate(filename) as f:
        t1 = f.mktree("t1", {"x": "int32"})
        t1.extend({"x": np.arange(10, dtype=np.int32)})
        t2 = f.mktree("t2", {"y": "int32"})
        t2.extend({"y": np.arange(10, dtype=np.int32)})
        t3 = f.mktree("t3", {"z": "int32"})
        t3.extend({"z": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        assert f._file._tree_basket_seeks_cache is None
        del f["t1"]
        # Cache populated after first delete
        assert f._file._tree_basket_seeks_cache is not None
        # Subsequent delete reuses cache
        del f["t2"]
        assert "t2" not in f

        # Invalidation on move_tree
        f._file._trees[123] = "dummy"
        f._file._move_tree(123, 456)
        assert f._file._tree_basket_seeks_cache is None

        # Extending with new fields invalidates cache
        f._file._tree_basket_seeks_cache = {"some": "cache"}
        t3 = f["t3"]
        t3.extend(
            {
                "z": np.arange(10, dtype=np.int32),
                "new_field": np.arange(10, dtype=np.float64),
            },
            accept_new_fields=True,
        )
        assert f._file._tree_basket_seeks_cache is None


def test_delete_empty_tree_has_no_baskets(tmp_path):
    filename = os.path.join(tmp_path, "empty_tree.root")
    with uproot.recreate(filename) as f:
        f.mktree("empty", {"x": "int32"})

    with uproot.update(filename) as f:
        del f["empty"]
        assert "empty" not in f
