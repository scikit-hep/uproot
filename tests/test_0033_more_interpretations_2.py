# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_awkward_strings():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert awkward.to_list(tree["string"].array(library="ak")) == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_pandas_strings():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(skhep_testdata.data_path("uproot-stl_containers.root"))[
        "tree"
    ] as tree:
        assert tree["string"].array(library="pd").values.tolist() == [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]


def test_leaflist_numpy():
    with uproot.open(skhep_testdata.data_path("uproot-leaflist.root"))[
        "tree/leaflist"
    ] as branch:
        result = branch.array(library="np")
        assert result.dtype.names == ("x", "y", "z")
        assert result.tolist() == [
            (1.1, 1, 97),
            (2.2, 2, 98),
            (3.3, 3, 99),
            (4.0, 4, 100),
            (5.5, 5, 101),
        ]
        assert branch.typename == "struct {double x; int32_t y; int8_t z;}"


def test_leaflist_awkward():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-leaflist.root"))[
        "tree/leaflist"
    ] as branch:
        result = branch.array(library="ak")
        assert str(awkward.type(result)) == "5 * {x: float64, y: int32, z: int8}"
        assert awkward.to_list(result) == [
            {"x": 1.1, "y": 1, "z": 97},
            {"x": 2.2, "y": 2, "z": 98},
            {"x": 3.3, "y": 3, "z": 99},
            {"x": 4.0, "y": 4, "z": 100},
            {"x": 5.5, "y": 5, "z": 101},
        ]


def test_leaflist_pandas():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(skhep_testdata.data_path("uproot-leaflist.root"))["tree"] as tree:
        result = tree["leaflist"].array(library="pd")

        assert result.ak["x"].tolist() == [1.1, 2.2, 3.3, 4.0, 5.5]
        assert result.ak["y"].tolist() == [1, 2, 3, 4, 5]
        assert result.ak["z"].tolist() == [97, 98, 99, 100, 101]

        result = tree.arrays("leaflist", library="pd")
        assert list(result.columns) == ["leaflist"]


def test_fixed_width():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as tree:
        assert tree["ai4"].array(library="np").tolist() == [
            [i, i + 1, i + 2] for i in range(-14, 16)
        ]
        assert tree["ai4"].typename == "int32_t[3]"


def test_fixed_width_awkward():
    awkward = pytest.importorskip("awkward")
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as tree:
        assert awkward.to_list(tree["ai4"].array(library="ak")) == [
            [i, i + 1, i + 2] for i in range(-14, 16)
        ]


def test_fixed_width_pandas():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as tree:
        result = tree["ai4"].array(library="pd").ak.array
        assert result[:, 0].tolist() == list(range(-14, 16))
        assert result[:, 1].tolist() == list(range(-13, 17))
        assert result[:, 2].tolist() == list(range(-12, 18))

        result = tree.arrays("ai4", library="pd")
        assert list(result.columns) == ["ai4"]
        assert result["ai4"].ak.array[:, 0].tolist() == list(range(-14, 16))
        assert result["ai4"].ak.array[:, 1].tolist() == list(range(-13, 17))
        assert result["ai4"].ak.array[:, 2].tolist() == list(range(-12, 18))


def test_fixed_width_pandas_2():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree"
    ] as tree:
        result = tree.arrays(
            "xyz", aliases={"xyz": "get('ArrayI32[10]')"}, library="pd"
        )
        assert list(result.columns) == ["xyz"]


def hook(self, **kwargs):
    print("ENTER")
    for k, v in kwargs.items():
        print(k, v)
