# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot


def hist(values):
    return (
        np.array(values, dtype=np.float64),
        np.arange(len(values) + 1, dtype=np.float64),
    )


@pytest.mark.parametrize("where", ["d/h", "/d/h", "d/h/", "d//h"])
def test_delete_by_path(tmp_path, where):
    filename = tmp_path / "delete_by_path.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["d/g"] = hist([2.0])

    with uproot.update(filename) as f:
        del f[where]
        assert f.keys() == ["d;1", "d/g;1"]

    with uproot.open(filename) as f:
        assert f.keys() == ["d;1", "d/g;1"]


def test_delete_by_path_selects_latest_cycle(tmp_path):
    filename = tmp_path / "delete_by_path_cycle.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["d/h"] = hist([2.0])

    with uproot.update(filename) as f:
        del f["d/h"]
        assert f.keys() == ["d;1", "d/h;1"]
        assert f["d/h;1"].values().tolist() == [1.0]

    with uproot.open(filename) as f:
        assert f.keys() == ["d;1", "d/h;1"]


def test_delete_by_path_with_cycle(tmp_path):
    filename = tmp_path / "delete_by_path_given_cycle.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["d/h"] = hist([2.0])

    with uproot.update(filename) as f:
        del f["d/h;1"]
        assert f.keys() == ["d;1", "d/h;2"]
        assert f["d/h;2"].values().tolist() == [2.0]


def test_delete_directory_by_path(tmp_path):
    filename = tmp_path / "delete_directory_by_path.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["g"] = hist([2.0])

    with uproot.update(filename) as f:
        del f["d/"]
        assert f.keys() == ["g;1"]

    with uproot.open(filename) as f:
        assert f.keys() == ["g;1"]


def test_delete_tree_by_path(tmp_path):
    filename = tmp_path / "delete_tree_by_path.root"
    with uproot.recreate(filename) as f:
        f["d/t"] = {"x": np.arange(3, dtype=np.int32)}
        f["d/t"] = {"x": np.arange(3, dtype=np.int32)}

    with uproot.update(filename) as f:
        del f["d/t;1"]
        assert f.keys() == ["d;1", "d/t;2"]

    with uproot.update(filename) as f:
        del f["d/t"]
        assert f.keys() == ["d;1"]

    with uproot.open(filename) as f:
        assert f.keys() == ["d;1"]


@pytest.mark.parametrize("where", ["d/nothere", "d/h;9", "d/t/x", "d/t:x"])
def test_delete_by_path_that_selects_nothing_raises(tmp_path, where):
    filename = tmp_path / "delete_by_path_missing.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["d/t"] = {"x": np.arange(3, dtype=np.int32)}
    expected = ["d;1", "d/h;1", "d/t;1"]

    with uproot.update(filename) as f:
        with pytest.raises(uproot.KeyInFileError):
            del f[where]
        assert f.keys() == expected

    with uproot.open(filename) as f:
        assert f.keys() == expected


def test_contains_cycle_and_path(tmp_path):
    filename = tmp_path / "contains_cycle_and_path.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
        f["h"] = hist([2.0])
        f["h"] = hist([3.0])

        assert "h;2" in f
        assert "h;9" not in f
        assert "d/h" in f
        assert "d/h;1" in f
        assert "/d/h" in f
        assert "d/nothere" not in f
        assert "d/h;foo" not in f
        assert "h/x" not in f
        assert "nothere/h" not in f
        assert "/" not in f
        assert "//" not in f

    with uproot.open(filename) as f:
        assert "h;2" in f
        assert "d/h" in f


def test_contains_does_not_load_trees(tmp_path):
    filename = tmp_path / "contains_no_load.root"
    with uproot.recreate(filename) as f:
        f["d/t"] = {"x": np.arange(3, dtype=np.int32)}

    with uproot.update(filename) as f:
        assert len(f._file._trees) == 0
        assert "d/t" in f
        assert "d/t;1" in f
        assert "d/t;9" not in f
        assert len(f._file._trees) == 0

    with open(filename, "r+b") as fo, uproot.update(fo) as f:
        assert "t" in f["d"]
        assert "d/t" in f
        assert "d/t;1" in f
        assert "d/t;9" not in f


@pytest.mark.parametrize("where", ["/", "//"])
def test_delete_root_path_raises(tmp_path, where):
    filename = tmp_path / "delete_root_path.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])
    expected = ["d;1", "d/h;1"]

    with uproot.update(filename) as f:
        with pytest.raises(uproot.KeyInFileError):
            del f[where]
        assert f.keys() == expected

    with uproot.open(filename) as f:
        assert f.keys() == expected


def test_get_root_path(tmp_path):
    filename = tmp_path / "get_root_path.root"
    with uproot.recreate(filename) as f:
        f["d/h"] = hist([1.0])

        assert f["/"] is f

    with uproot.open(filename) as f:
        assert f["/"] is f
