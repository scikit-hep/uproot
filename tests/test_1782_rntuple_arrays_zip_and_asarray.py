# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import numpy as np

import uproot


def test_rntuple_arrays_zip(tmp_path):
    path = tmp_path / "test_zip.root"
    data = {
        "x": np.array([1, 2, 3], dtype=np.int64),
        "jet_pt": ak.Array([[1.0, 2.0], [], [3.0]]),
        "jet_eta": ak.Array([[0.1, 0.2], [], [0.3]]),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data
        file.mktree(
            "tree",
            {"x": "int64", "jet_pt": "var * float64", "jet_eta": "var * float64"},
        )
        file["tree"].extend(data)

    with uproot.open(path) as file:
        tree = file["tree"]
        ntuple = file["ntuple"]

        tree_zip = tree.arrays(["x", "jet_pt", "jet_eta"], how="zip")
        ntuple_zip = ntuple.arrays(how="zip")

        assert ntuple_zip.to_list() == tree_zip.to_list()
        assert ntuple_zip.type == tree_zip.type


def test_rntuple_arrays_zip_multiple_jagged_groups(tmp_path):
    path = tmp_path / "test_multi_zip.root"
    data = {
        "x": np.array([10, 20, 30], dtype=np.int64),
        "jet_pt": ak.Array([[1.0, 2.0], [], [3.0]]),
        "jet_eta": ak.Array([[0.1, 0.2], [], [0.3]]),
        "muon_pt": ak.Array([[100.0], [200.0, 300.0], []]),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        res = ntuple.arrays(how="zip")

        assert res.x.to_list() == [10, 20, 30]
        assert res.jet.pt.to_list() == [[1.0, 2.0], [], [3.0]]
        assert res.jet.eta.to_list() == [[0.1, 0.2], [], [0.3]]
        assert res.muon.pt.to_list() == [[100.0], [200.0, 300.0], []]


def test_rntuple_arrays_zip_sliced(tmp_path):
    path = tmp_path / "test_sliced_zip.root"
    data = {
        "x": np.arange(6, dtype=np.int64),
        "jet_pt": ak.Array([[1.0], [2.0, 3.0], [], [4.0], [5.0, 6.0], []]),
        "jet_eta": ak.Array([[0.1], [0.2, 0.3], [], [0.4], [0.5, 0.6], []]),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        full = ntuple.arrays(how="zip")
        sliced = ntuple.arrays(how="zip", entry_start=1, entry_stop=4)

        assert sliced.to_list() == full[1:4].to_list()


def test_rntuple_arrays_zip_empty(tmp_path):
    path = tmp_path / "test_empty_zip.root"
    data = {
        "x": np.array([1, 2], dtype=np.int64),
        "jet_pt": ak.Array([[1.0], [2.0]]),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        empty_slice = ntuple.arrays(how="zip", entry_start=0, entry_stop=0)
        assert len(empty_slice) == 0
        assert empty_slice.to_list() == []


def test_rntuple_numpy_asarray_ntuple(tmp_path):
    path = tmp_path / "test_asarray.root"
    data = {
        "x": np.array([1, 2, 3], dtype=np.int64),
        "y": np.array([4, 5, 6], dtype=np.int64),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data
        file.mktree("tree", {"x": "int64", "y": "int64"})
        file["tree"].extend(data)

    with uproot.open(path) as file:
        tree = file["tree"]
        ntuple = file["ntuple"]

        tree_arr = np.asarray(tree)
        ntuple_arr = np.asarray(ntuple)

        assert ntuple_arr.shape == tree_arr.shape
        assert ntuple_arr.dtype == tree_arr.dtype
        assert np.array_equal(ntuple_arr, tree_arr)


def test_rntuple_numpy_asarray_fields(tmp_path):
    path = tmp_path / "test_asarray_fields.root"
    data = {
        "x": np.array([1, 2, 3], dtype=np.int64),
        "jet_pt": ak.Array([[1.0, 2.0], [], [3.0]]),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data

    with uproot.open(path) as file:
        ntuple = file["ntuple"]

        arr_x = np.asarray(ntuple["x"])
        assert isinstance(arr_x, np.ndarray)
        assert arr_x.shape == (3,)
        assert arr_x.dtype == np.int64
        assert arr_x.tolist() == [1, 2, 3]

        arr_x_float = np.asarray(ntuple["x"], dtype=np.float64)
        assert arr_x_float.dtype == np.float64
        assert arr_x_float.tolist() == [1.0, 2.0, 3.0]

        arr_jet = np.asarray(ntuple["jet_pt"])
        assert isinstance(arr_jet, np.ndarray)
        assert arr_jet.shape == (3,)
        assert arr_jet.dtype == object
        assert arr_jet[0].tolist() == [1.0, 2.0]
        assert arr_jet[1].tolist() == []
        assert arr_jet[2].tolist() == [3.0]

        ntuple_arr = np.asarray(ntuple)
        assert isinstance(ntuple_arr, np.ndarray)
        assert ntuple_arr.shape == (2, 3)
        assert ntuple_arr.dtype == object


def test_rntuple_arrays_zip_strings_and_fixed_size_arrays(tmp_path):
    path = tmp_path / "test_strings_fixed.root"
    data = {
        "x": np.array([1, 2], dtype=np.int64),
        "dataset": ["abc", "de"],
        "pos": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        "arr_a": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "arr_b": np.array([[5, 6], [7, 8]], dtype=np.int64),
        "jet_pt": ak.Array([[1.0, 2.0], [3.0]]),
        "jet_tag": ak.Array([["tagA", "tagB"], ["tagC"]]),
    }
    with uproot.recreate(path) as file:
        file["ntuple"] = data

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        res = ntuple.arrays(how="zip")

        assert res.x.to_list() == [1, 2]
        assert res.dataset.to_list() == ["abc", "de"]
        assert res.pos.to_list() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert res.arr_a.to_list() == [[1, 2], [3, 4]]
        assert res.arr_b.to_list() == [[5, 6], [7, 8]]
        assert res.jet.pt.to_list() == [[1.0, 2.0], [3.0]]
        assert res.jet.tag.to_list() == [["tagA", "tagB"], ["tagC"]]

        assert "jagged0" not in res.fields
        assert "jagged1" not in res.fields
        assert "arr" not in res.fields
