# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
import skhep_testdata

import uproot

awkward = pytest.importorskip("awkward")


def test_unequal_baskets(tmp_path):
    path = str(tmp_path / "hzz.root")
    with (
        open(skhep_testdata.data_path("uproot-HZZ.root"), "rb") as src,
        open(path, "wb") as dst,
    ):
        dst.write(src.read())

    with uproot.open(path) as f:
        before = f["events"].arrays()
        n = f["events"].num_entries

    with uproot.update(path) as f:
        f["events"].add_branches({"new": np.arange(n, dtype=np.float64)})

    with uproot.open(path) as f:
        after = f["events"].arrays()[before.fields]
        assert awkward.array_equal(before, after)
        np.testing.assert_array_equal(
            f["events"]["new"].array(library="np"), np.arange(n)
        )


def test_empty_tree_add_branches_then_extend(tmp_path):
    path = str(tmp_path / "empty.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"a": np.int64})
    with uproot.update(path) as f:
        f["t"].add_branches({"b": np.zeros(0)})
        f["t"].extend({"a": np.arange(3), "b": np.arange(3.0)})
    with uproot.open(path) as f:
        assert f["t"]["a"].array(library="np").tolist() == [0, 1, 2]
        assert f["t"]["b"].array(library="np").tolist() == [0.0, 1.0, 2.0]


def test_accept_new_fields_then_extend(tmp_path):
    path = str(tmp_path / "newfields.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"a": np.int64})
        f["t"].extend({"a": [0, 1, 2], "b": [0.0, 1.0, 2.0]}, accept_new_fields=True)
        f["t"].extend({"a": [3], "b": [3.0]})
    with uproot.open(path) as f:
        assert f["t"]["a"].array(library="np").tolist() == [0, 1, 2, 3]
        assert f["t"]["b"].array(library="np").tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "jagged",
    [
        awkward.Array([[1.0], [], [2.0, 3.0]]),
        [[1.0], [], [2.0, 3.0]],
    ],
)
def test_jagged_add_branches(tmp_path, jagged):
    path = str(tmp_path / "x.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"a": np.int64})
        f["t"].extend({"a": np.arange(3)})
    with uproot.update(path) as f:
        f["t"].add_branches({"j": jagged})
    with uproot.open(path) as f:
        assert f["t"]["a"].array(library="np").tolist() == [0, 1, 2]
        assert f["t"]["nj"].array(library="np").tolist() == [1, 0, 2]
        assert awkward.to_list(f["t"]["j"].array()) == [[1.0], [], [2.0, 3.0]]
