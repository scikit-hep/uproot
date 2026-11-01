# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest

import uproot


def hist(values):
    return (
        np.array(values, dtype=np.float64),
        np.arange(len(values) + 1, dtype=np.float64),
    )


def test_delete_older_cycle(tmp_path):
    filename = os.path.join(tmp_path, "cycles.root")
    with uproot.recreate(filename) as f:
        f["h"] = hist([1.0, 2.0, 3.0])
        f["h"] = hist([4.0, 5.0, 6.0])

    with uproot.update(filename) as f:
        del f["h;1"]
        assert f.keys() == ["h;2"]
        assert "h" in f

    with uproot.open(filename) as f:
        assert f.keys() == ["h;2"]
        assert f["h;2"].values().tolist() == [4.0, 5.0, 6.0]


def test_delete_only_cycle(tmp_path):
    filename = os.path.join(tmp_path, "single.root")
    with uproot.recreate(filename) as f:
        f["h"] = hist([1.0])
        del f["h"]
        assert f.keys() == []
        assert "h" not in f

    with uproot.open(filename) as f:
        assert f.keys() == []


def test_delete_with_subdirectory(tmp_path):
    filename = os.path.join(tmp_path, "subdir.root")
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["h"] = hist([2.0])
        assert len(f) == 3
        del f["h"]
        assert "h" in f  # still found in the subdirectory
        assert "other" not in f
        del f["d"]["h"]
        assert "h" not in f
        assert len(f) == 1


def test_contains_with_directory_and_tree_of_same_name(tmp_path):
    filename = os.path.join(tmp_path, "same_name.root")
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f.mktree("d", {"x": np.int32})

    with uproot.update(filename) as f:
        assert f.keys(recursive=False) == ["d;1", "d;2"]
        assert "other" not in f
        assert "h" in f
        assert len(f) == 3


def test_extend_older_tree_cycle(tmp_path):
    filename = os.path.join(tmp_path, "tree.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": np.int32})
        f.mktree("t", {"x": np.int32})
        t1 = f["t;1"]
        for _ in range(20):
            t1.extend({"x": np.arange(3, dtype=np.int32)})

    with uproot.open(filename) as f:
        assert f["t;1"].num_entries == 60
        assert f["t;2"].num_entries == 0


def test_delete_middle_cycle_then_write(tmp_path):
    filename = os.path.join(tmp_path, "middle.root")
    with uproot.recreate(filename) as f:
        f["h"] = hist([1.0])
        f["h"] = hist([2.0])
        f["h"] = hist([3.0])
        del f["h;2"]
        assert f.keys() == ["h;1", "h;3"]
        f["h"] = hist([4.0])
        assert f.keys() == ["h;1", "h;3", "h;4"]

    with uproot.open(filename) as f:
        assert f.keys() == ["h;1", "h;3", "h;4"]


def test_recreate_name_after_deleting_all_cycles(tmp_path):
    filename = os.path.join(tmp_path, "all.root")
    with uproot.recreate(filename) as f:
        f["h"] = hist([1.0])
        f["h"] = hist([2.0])

    with uproot.update(filename) as f:
        del f["h;1"]
        del f["h;2"]
        assert "h" not in f
        f["h"] = hist([9.0])
        assert f.keys() == ["h;1"]

    with uproot.open(filename) as f:
        assert f["h"].values().tolist() == [9.0]


def test_deleted_tree_handle_raises(tmp_path):
    filename = os.path.join(tmp_path, "handle.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": np.int32})
        f.mktree("t", {"x": np.int32})
        t1 = f["t;1"]
        del f["t;1"]
        with pytest.raises(ValueError, match="deleted"):
            t1.extend({"x": np.arange(3, dtype=np.int32)})
        with pytest.raises(ValueError, match="deleted"):
            t1.add_branches({"y": np.array([], dtype=np.int32)})


def test_deleted_ntuple_handle_raises(tmp_path):
    filename = os.path.join(tmp_path, "ntuple_handle.root")
    with uproot.recreate(filename) as f:
        f.mkrntuple("n", {"x": np.int32})
        f.mkrntuple("n", {"x": np.int32})
        n1 = f["n;1"]
        del f["n;1"]
        f.mkrntuple("u", {"x": np.int32})
        with pytest.raises(ValueError, match="deleted"):
            n1.extend({"x": np.arange(7, dtype=np.int32)})
        with pytest.raises(ValueError, match="deleted"):
            n1.add_fields({"y": np.int32})

    with uproot.open(filename) as f:
        assert f["u"].num_entries == 0


def test_add_branches_keeps_handle(tmp_path):
    filename = os.path.join(tmp_path, "add_branches.root")
    with uproot.recreate(filename) as f:
        t = f.mktree("t", {"x": np.int32})
        t.add_branches({"y": np.array([], dtype=np.int32)})
        assert f["t"] is t
        del f["t"]
        f.mktree("u", {"x": np.int32, "y": np.int32})
        with pytest.raises(ValueError, match="deleted"):
            t.extend(
                {
                    "x": np.arange(7, dtype=np.int32),
                    "y": np.arange(7, dtype=np.int32),
                }
            )

    with uproot.open(filename) as f:
        assert f["u"].num_entries == 0


def test_add_fields_keeps_handle(tmp_path):
    filename = os.path.join(tmp_path, "add_fields.root")
    with uproot.recreate(filename) as f:
        n = f.mkrntuple("n", {"x": np.int32})
        n.add_fields({"y": np.int32})
        assert f["n"] is n
        del f["n"]
        f.mkrntuple("u", {"x": np.int32, "y": np.int32})
        with pytest.raises(ValueError, match="deleted"):
            n.extend(
                {
                    "x": np.arange(7, dtype=np.int32),
                    "y": np.arange(7, dtype=np.int32),
                }
            )

    with uproot.open(filename) as f:
        assert f["u"].num_entries == 0


def test_add_branches_older_cycle_then_delete(tmp_path):
    filename = os.path.join(tmp_path, "add_branches_older.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": np.int32})
        f.mktree("t", {"x": np.int32})
        f["t;1"].add_branches({"y": np.array([], dtype=np.int32)})

    with uproot.open(filename) as f:
        assert f["t;1"].keys() == ["x", "y"]
        assert f["t;2"].keys() == ["x"]

    with uproot.update(filename) as f:
        del f["t;1"]

    with uproot.open(filename) as f:
        assert f.keys() == ["t;2"]
        assert f["t;2"].keys() == ["x"]


def test_add_branches_older_cycle_then_delete_same_session(tmp_path):
    filename = os.path.join(tmp_path, "add_branches_session.root")
    with uproot.recreate(filename) as f:
        f.mktree("t", {"x": np.int32})
        f.mktree("t", {"x": np.int32})
        t1 = f["t;1"]
        t1.add_branches({"y": np.array([], dtype=np.int32)})
        del f["t;1"]
        with pytest.raises(ValueError, match="deleted"):
            t1.extend(
                {
                    "x": np.arange(3, dtype=np.int32),
                    "y": np.arange(3, dtype=np.int32),
                }
            )

    with uproot.open(filename) as f:
        assert f.keys() == ["t;2"]
        assert f["t;2"].keys() == ["x"]
        assert f["t;2"].num_entries == 0


def test_reload_keeps_column_encoding_error(tmp_path, monkeypatch):
    filename = os.path.join(tmp_path, "encoding.root")
    with uproot.recreate(filename) as f:
        n = f.mkrntuple("n", {"x": np.int32})
        load = uproot.writing.writable.WritableDirectory._load_ntuple_cascading
        monkeypatch.setattr(
            uproot.writing.writable.WritableDirectory,
            "_load_ntuple_cascading",
            lambda self, key: (load(self, key)[0], "unsupported encoding"),
        )
        n._reload_cascading()
        assert f["n"] is n
        with pytest.raises(ValueError, match="unsupported encoding"):
            n.extend({"x": np.arange(3, dtype=np.int32)})


def test_add_fields_older_cycle(tmp_path):
    filename = os.path.join(tmp_path, "add_fields_older.root")
    with uproot.recreate(filename) as f:
        f.mkrntuple("n", {"x": np.int32})
        f.mkrntuple("n", {"x": np.int32})
        n1 = f["n;1"]
        n1.add_fields({"y": np.int32})
        assert f["n;1"] is n1

    with uproot.open(filename) as f:
        assert f["n;1"].keys() == ["x", "y"]
        assert f["n;2"].keys() == ["x"]


def test_extend_older_ntuple_cycle(tmp_path):
    filename = os.path.join(tmp_path, "ntuple.root")
    with uproot.recreate(filename) as f:
        f.mkrntuple("n", {"x": np.int32})
        f.mkrntuple("n", {"x": np.int32})
        n1 = f["n;1"]
        n1.extend({"x": np.arange(3, dtype=np.int32)})

    with uproot.open(filename) as f:
        assert f["n;1"].num_entries == 3
        assert f["n;2"].num_entries == 0
