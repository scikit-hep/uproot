# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest

import uproot

EMPTY = {"x": np.array([], np.float64), "j": ak.Array([[1.0]])[:0]}
FULL = {"x": np.array([32.5, 37.4, 34.1]), "j": ak.Array([[1.0, 2.0], [], [3.0]])}
FULL_LIST = [
    {"x": 32.5, "nj": 2, "j": [1.0, 2.0]},
    {"x": 37.4, "nj": 0, "j": []},
    {"x": 34.1, "nj": 1, "j": [3.0]},
]


def _baskets(filename):
    with uproot.open(filename) as f:
        tree = f["tree"]
        num_baskets = {branch.num_baskets for branch in tree.values()}
        assert len(num_baskets) == 1
        return tree.num_entries, num_baskets.pop(), tree.arrays().tolist()


def test_empty_tree_and_zero_entry_extends(tmp_path):
    filename = os.path.join(tmp_path, "empty.root")
    with uproot.recreate(filename) as f:
        tree = f.mktree("tree", EMPTY, initial_basket_capacity=2)
        for _ in range(3):
            tree.extend(EMPTY)

    with uproot.open(filename) as f:
        tree = f["tree"]
        assert tree.member("fTotBytes") == tree.member("fZipBytes") == 0
        for branch in tree.values():
            assert branch.member("fWriteBasket") == 0
            assert branch.member("fMaxBaskets") == 2
            assert branch.member("fBasketSeek")[0] == 0
    assert _baskets(filename) == (0, 0, [])

    with uproot.update(filename) as f:
        f["tree"].extend(EMPTY)
    assert _baskets(filename) == (0, 0, [])

    with uproot.update(filename) as f:
        f["tree"].extend({"x": FULL["x"][:2], "j": FULL["j"][:2]})
        f["tree"].extend(EMPTY)
        f["tree"].extend({"x": FULL["x"][2:], "j": FULL["j"][2:]})
    assert _baskets(filename) == (3, 2, FULL_LIST)
    with uproot.open(filename) as f:
        assert f["tree"]["x"].member("fBasketEntry")[:3].tolist() == [0, 2, 3]


@pytest.mark.parametrize("empty_first", [True, False])
def test_hadd_with_root(tmp_path, empty_first):
    ROOT = pytest.importorskip("ROOT")

    inputs = []
    for name, data in [("empty", EMPTY), ("full", FULL)]:
        inputs.append(os.path.join(tmp_path, f"{name}.root"))
        with uproot.recreate(inputs[-1]) as f:
            f.mktree("tree", data)
    merged = os.path.join(tmp_path, "merged.root")

    merger = ROOT.TFileMerger(False, False)
    merger.SetPrintLevel(0)
    merger.OutputFile(merged, "RECREATE")
    for filename in inputs if empty_first else inputs[::-1]:
        merger.AddFile(filename)
    assert merger.Merge()

    file = ROOT.TFile.Open(merged)
    tree = file.Get("tree")
    rows = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        j = tree.GetLeaf("j")
        rows.append(
            {
                "x": tree.GetLeaf("x").GetValue(),
                "nj": int(tree.GetLeaf("nj").GetValue()),
                "j": [j.GetValue(k) for k in range(j.GetLen())],
            }
        )
    file.Close()

    assert rows == FULL_LIST
    assert _baskets(merged)[::2] == (3, FULL_LIST)
