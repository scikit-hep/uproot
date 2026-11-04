# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest

import uproot

ak = pytest.importorskip("awkward")


def test_rntuple_variant_multicluster_two_clusters(tmp_path):
    path = tmp_path / "var.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], "a"]), "y": [1, 2]}
        f["n"].extend({"x": ak.Array([[4], "b"]), "y": [3, 4]})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().tolist() == [
            {"x": [1], "y": 1},
            {"x": "a", "y": 2},
            {"x": [4], "y": 3},
            {"x": "b", "y": 4},
        ]
        assert n.arrays(entry_start=2).x.tolist() == [[4], "b"]
        assert n.arrays(entry_start=1, entry_stop=3).x.tolist() == ["a", [4]]


def test_rntuple_variant_multicluster_three_clusters_asymmetric(tmp_path):
    path = tmp_path / "var3.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[1], [2], "a"])}
        f["n"].extend({"x": ak.Array([[3], "b", "c"])})
        f["n"].extend({"x": ak.Array([[4], [5], "d"])})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [
            [1],
            [2],
            "a",
            [3],
            "b",
            "c",
            [4],
            [5],
            "d",
        ]
        assert n.arrays(entry_start=3).x.tolist() == [[3], "b", "c", [4], [5], "d"]
        assert n.arrays(entry_start=6).x.tolist() == [[4], [5], "d"]
        assert n.arrays(entry_start=2, entry_stop=7).x.tolist() == [
            "a",
            [3],
            "b",
            "c",
            [4],
        ]


def test_rntuple_variant_multicluster_nested_vector(tmp_path):
    path = tmp_path / "var_vec.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[[1], "a"], ["b"]])}
        f["n"].extend({"x": ak.Array([[[2], "c"], [[3], [4]]])})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [
            [[1], "a"],
            ["b"],
            [[2], "c"],
            [[3], [4]],
        ]
        assert n.arrays(entry_start=1).x.tolist() == [
            ["b"],
            [[2], "c"],
            [[3], [4]],
        ]
        assert n.arrays(entry_start=2).x.tolist() == [
            [[2], "c"],
            [[3], [4]],
        ]


def test_rntuple_variant_multicluster_virtual(tmp_path):
    path = tmp_path / "var_virtual.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([[10], "x"]), "y": [100, 200]}
        f["n"].extend({"x": ak.Array([[20], "y"]), "y": [300, 400]})

    with uproot.open(path) as f:
        n = f["n"]
        arr = n.arrays(virtual=True)
        assert arr.tolist() == [
            {"x": [10], "y": 100},
            {"x": "x", "y": 200},
            {"x": [20], "y": 300},
            {"x": "y", "y": 400},
        ]


def test_rntuple_variant_multicluster_scalars(tmp_path):
    path = tmp_path / "var_scalars.root"
    with uproot.recreate(path) as f:
        f["n"] = {"x": ak.Array([1, "first"])}
        f["n"].extend({"x": ak.Array([2, "second"])})

    with uproot.open(path) as f:
        n = f["n"]
        assert n.arrays().x.tolist() == [1, "first", 2, "second"]
        assert n.arrays(entry_start=1, entry_stop=3).x.tolist() == ["first", 2]
