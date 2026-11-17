# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import os
import shutil

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
