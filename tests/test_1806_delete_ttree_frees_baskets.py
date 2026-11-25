# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import copy
import os
import struct

import awkward as ak
import numpy as np
import pytest

import uproot
import uproot.writing.writable as writable
from uproot.writing._cascade import FreeSegments, String


def _basket_ranges(directory, name):
    key = directory._cascading.data.get_key(name, None)
    return writable._collect_tree_basket_ranges(
        directory._file._read_key(key, directory)
    )


def _is_free(f, start, stop):
    return any(
        s < stop and start < e for s, e in f._cascading.freesegments._data.slices
    )


@pytest.mark.parametrize(
    "branches, data",
    [
        ({"x": "float64"}, {"x": np.arange(100_000, dtype=np.float64)}),
        (
            {"a": "int32", "b": "float64", "c": "int64"},
            {
                "a": np.arange(50_000, dtype=np.int32),
                "b": np.arange(50_000, dtype=np.float64),
                "c": np.arange(50_000, dtype=np.int64),
            },
        ),
        (
            {"jag": ak.Array([[1, 2], [3], [4, 5, 6]])},
            {"jag": ak.Array([[7], [8, 9]] * 10_000)},
        ),
    ],
    ids=["flat", "multibranch", "jagged"],
)
@pytest.mark.parametrize("subdir", [None, "sub"])
def test_repeated_write_and_delete_does_not_grow_file(tmp_path, branches, data, subdir):
    filename = os.path.join(tmp_path, "leak.root")
    with uproot.recreate(filename) as f:
        d = f.mkdir(subdir) if subdir else f
        d.mktree("keep", {"k": "int32"}).extend({"k": np.arange(3, dtype=np.int32)})

    sizes = []
    for _ in range(3):
        with uproot.update(filename) as f:
            d = f[subdir] if subdir else f
            d.mktree("t", branches).extend(data)
            del d["t"]
        sizes.append(os.path.getsize(filename))

    # the free-segments record may grow once, then the file size is stable
    assert sizes[1] == sizes[2]
    assert sizes[1] - sizes[0] < 200

    with uproot.open(filename) as f:
        d = f[subdir] if subdir else f
        assert "t" not in d
        assert d["keep"]["k"].array().tolist() == [0, 1, 2]


@pytest.mark.parametrize("shared", ["cycle", "alias"])
def test_shared_key_keeps_baskets_until_last_reference(tmp_path, shared):
    filename = os.path.join(tmp_path, "shared.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": "int32"}).extend({"x": np.arange(10, dtype=np.int32)})

    # second key pointing at the same TTree record (and therefore same baskets)
    with uproot.update(filename) as f:
        other = copy.copy(f._cascading.data.get_key("t", 1))
        if shared == "cycle":
            other._cycle = 2
            first, second = "t;1", "t;2"
        else:
            other._name = String(None, "alias")
            first, second = "alias", "t"
        f._cascading.data.add_key(other)
        f._cascading.write(f._file.sink)

    with uproot.update(filename) as f:
        baskets = _basket_ranges(f, second.split(";")[0])
        del f[first]
        assert not any(_is_free(f, *b) for b in baskets)
        f["h"] = (np.full(10, 5.0), np.linspace(0, 1, 11))

    with uproot.open(filename) as f:
        assert first not in f
        assert f[second]["x"].array().tolist() == list(range(10))

    with uproot.update(filename) as f:
        del f[second]
        assert all(_is_free(f, *b) for b in baskets)


def test_delete_ttree_subclass_frees_baskets(tmp_path):
    filename = os.path.join(tmp_path, "ntd.root")
    with uproot.recreate(filename) as f:
        f.mktree("nt", {"x": "int32"}).extend({"x": np.arange(1000, dtype=np.int32)})

    with uproot.update(filename) as f:
        key = f._cascading.data.get_key("nt", 1)
        key._classname = String(None, "TNtupleD")
        f._cascading.write(f._file.sink)

    with uproot.update(filename) as f:
        baskets = _basket_ranges(f, "nt")
        del f["nt"]
        assert all(_is_free(f, *b) for b in baskets)


def test_delete_empty_tree(tmp_path):
    filename = os.path.join(tmp_path, "empty.root")
    with uproot.recreate(filename) as f:
        f.mktree("empty", {"x": "int32"})

    with uproot.update(filename) as f:
        del f["empty"]
        assert "empty" not in f


def _fail_read(name):
    def inject(monkeypatch, f):
        orig = f._file._read_key

        def read_key(key, *args, **kwargs):
            if key.name.string == name:
                raise OSError("corrupt")
            return orig(key, *args, **kwargs)

        monkeypatch.setattr(f._file, "_read_key", read_key)

    return inject


def _fail_collect(monkeypatch, f):
    def collect(tree):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(writable, "_collect_tree_basket_ranges", collect)


def _fail_subdir(monkeypatch, f):
    # root_directory builds a new WritableDirectory on each access, so the
    # directory walk can only be intercepted on the class
    def subdir(self, key):
        raise OSError("broken directory")

    monkeypatch.setattr(writable.WritableDirectory, "_subdir", subdir)


@pytest.mark.parametrize(
    "inject",
    [_fail_read("t1"), _fail_collect, _fail_read("t2"), _fail_subdir],
    ids=["deleted-tree-unreadable", "collect-fails", "other-tree-unreadable", "subdir"],
)
def test_delete_fails_closed(tmp_path, monkeypatch, inject):
    filename = os.path.join(tmp_path, "fail_closed.root")
    with uproot.recreate(filename) as f:
        f.mktree("t1", {"x": "int32"}).extend({"x": np.arange(10, dtype=np.int32)})
        f.mktree("t2", {"y": "int32"}).extend({"y": np.arange(10, dtype=np.int32)})
        f.mkdir("sub")

    with uproot.update(filename) as f:
        baskets = _basket_ranges(f, "t1")
        inject(monkeypatch, f)
        del f["t1"]
        monkeypatch.undo()
        assert "t1" not in f
        assert not any(_is_free(f, *b) for b in baskets)
        f["h"] = (np.full(10, 5.0), np.linspace(0, 1, 11))

    with uproot.open(filename) as f:
        assert f["t2"]["y"].array().tolist() == list(range(10))


def test_delete_non_tree_does_not_walk_subdirectories(tmp_path, monkeypatch):
    filename = os.path.join(tmp_path, "hist.root")
    with uproot.recreate(filename) as f:
        f["h"] = (np.full(10, 5.0), np.linspace(0, 1, 11))
        f.mkdir("sub")

    with uproot.update(filename) as f:
        _fail_subdir(monkeypatch, f)
        del f["h"]
        monkeypatch.undo()
        assert "h" not in f


def test_tree_basket_seeks_cache(tmp_path):
    filename = os.path.join(tmp_path, "cache.root")
    with uproot.recreate(filename) as f:
        for name in ("t1", "t2", "t3"):
            f.mktree(name, {"x": "int32"}).extend({"x": np.arange(10, dtype=np.int32)})

    def cached_names():
        return {name for _, name, _, _ in f._file._tree_basket_seeks_cache}

    with uproot.update(filename) as f:
        assert f._file._tree_basket_seeks_cache is None
        del f["t1"]
        assert cached_names() == {"t2", "t3"}

        # the second delete reuses the cache and still frees t2's baskets
        baskets = _basket_ranges(f, "t2")
        del f["t2"]
        assert all(_is_free(f, *b) for b in baskets)
        assert cached_names() == {"t3"}

        data = {"x": np.arange(10, dtype=np.int32), "y": np.arange(10.0)}
        f["t3"].extend(data, accept_new_fields=True)
        assert f._file._tree_basket_seeks_cache is None

        f._file._tree_basket_seeks_cache = {}
        f["t3"].extend(data)
        assert f._file._tree_basket_seeks_cache is None


def test_free_segments_merge_slices():
    merge = FreeSegments._merge_slices
    assert merge((), ()) == ()
    assert merge(((10, 20),), ()) == ((10, 20),)
    assert merge((), [(10, 20)]) == ((10, 20),)
    assert merge(((10, 20),), [(20, 30)]) == ((10, 30),)
    assert merge(((20, 30),), [(10, 20)]) == ((10, 30),)
    assert merge(((10, 20),), [(30, 40)]) == ((10, 20), (30, 40))
    assert merge(((30, 40),), [(10, 20)]) == ((10, 20), (30, 40))
    assert merge(((10, 20),), [(25, 25), (30, 25)]) == ((10, 20),)

    for slices, new in [(((10, 25),), [(20, 35)]), (((10, 50),), [(20, 30)])]:
        with pytest.raises(RuntimeError, match="overlaps one already marked as free"):
            merge(slices, new)
    with pytest.raises(RuntimeError, match="overlaps another segment to release"):
        merge((), [(10, 25), (20, 35)])


def test_free_segments_release(tmp_path):
    filename = os.path.join(tmp_path, "release.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": "int32"})

    with uproot.update(filename) as f:
        fs = f._cascading.freesegments
        orig = fs.data.slices

        fs.release_many([])
        fs.release_many([(100, 100), (200, 150)])
        assert fs.data.slices == orig

        with pytest.raises(RuntimeError, match="overlaps one already marked as free"):
            fs.release(*orig[0])

        # not at the end: merging into an existing slice keeps the record in place
        fs._data.end = fs.data.end + 1000
        assert not fs.at_end
        s0, s1 = orig[0]
        fs.release_many([(s1, s1 + 50)])
        assert (s0, s1 + 50) in fs.data.slices

        # a new slice makes the record grow, so it moves to the end
        fs.release_many([(s1 + 100, s1 + 200)])
        assert fs.at_end


class _Obj:
    def __init__(self, **members):
        self._members = members

    def has_member(self, name):
        return name in self._members

    def member(self, name):
        value = self._members[name]
        if isinstance(value, Exception):
            raise value
        return value


def test_collect_tree_basket_ranges():
    collect = writable._collect_tree_basket_ranges

    def tree(*branches):
        return _Obj(fBranches=list(branches))

    def branch(seeks=(), nbytes=(), **more):
        return _Obj(fBasketSeek=list(seeks), fBasketBytes=list(nbytes), **more)

    assert collect(_Obj()) == []
    assert collect(_Obj(fBranches=RuntimeError("broken tree"))) == []
    assert collect(tree(_Obj())) == []
    assert collect(tree(branch([100], [50], fFileName="external.root"))) == []
    assert collect(tree(_Obj(fBranches=RuntimeError("broken branch")))) == []
    # zero seeks (embedded baskets) and zero sizes are skipped
    b = branch([0, 300, 400], [10, 0, 20], fBranches=[], fFileName="")
    assert collect(tree(b)) == [(400, 420)]
    nested = branch(fBranches=[branch([200], [60])])
    assert collect(tree(nested)) == [(200, 260)]


class _Sink:
    def __init__(self, data):
        self.data = data

    def read(self, seek, nbytes, insist=False):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data[:nbytes]


def test_validate_basket_key(tmp_path):
    validate = writable._validate_basket_key

    filename = os.path.join(tmp_path, "validate.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": "int32"}).extend({"x": np.arange(10, dtype=np.int32)})

    with uproot.update(filename) as f:
        ((seek, stop),) = _basket_ranges(f, "t")
        sink = f._file.sink
        assert validate(sink, seek, stop - seek) is True
        assert validate(sink, seek, stop - seek + 1) is False
        assert validate(sink, 999_999, 100) is False
        assert validate(sink, -10, 100) is False

    small = uproot.reading._key_format_small.pack(50, 4, 10, 0, 26, 1, 100, 0)
    assert validate(_Sink(small + b"\x07TBasket"), 100, 50) is True
    big = uproot.reading._key_format_big.pack(50, 1004, 10, 0, 34, 1, 100, 0)
    assert validate(_Sink(big + b"\x07TBasket"), 100, 50) is True
    assert validate(_Sink(big + b"\x05TTree"), 100, 50) is False
    assert validate(_Sink(struct.pack(">ih", 50, 1004)), 100, 50) is False
    assert validate(_Sink(big + b"\x10short"), 100, 50) is False
    assert validate(_Sink(OSError("io error")), 100, 50) is False


def test_is_ttree_classname():
    is_ttree = writable._is_ttree_classname

    class Base(uproot.streamers.Model_TStreamerBase):
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name

    class Streamer:
        def __init__(self, name, bases):
            self.name = name
            self.elements = [Base(b) for b in bases]

    assert is_ttree("TTree") is True
    assert is_ttree("TNtupleD") is True
    assert is_ttree("Other", None) is False
    streamers = [
        object(),
        Streamer("CustomTree", ["IntermediateTree"]),
        Streamer("IntermediateTree", ["TTree"]),
        Streamer("NotATree", ["BaseA", "BaseA"]),
        Streamer("BaseA", ["TNamed"]),
    ]
    assert is_ttree("CustomTree", streamers) is True
    assert is_ttree("NotATree", streamers) is False
