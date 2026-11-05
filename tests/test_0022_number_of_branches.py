# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test_branchname():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays("i4", library="np")["i4"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(["i4", "i8"], library="np")
        assert set(arrays.keys()) == {"i4", "i8"}
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(filter_name="/i[48]/", library="np")
        assert set(arrays.keys()) == {"i4", "i8"}
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(filter_name=["/i[12]/", "/i[48]/"], library="np")
        assert set(arrays.keys()) == {"i1", "i2", "i4", "i8"}
        assert arrays["i1"].tolist() == list(range(-15, 15))
        assert arrays["i2"].tolist() == list(range(-15, 15))
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(filter_name="i*", library="np")
        assert set(arrays.keys()) == {"i1", "i2", "i4", "i8"}
        assert arrays["i1"].tolist() == list(range(-15, 15))
        assert arrays["i2"].tolist() == list(range(-15, 15))
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))

        arrays = sample.arrays(["i4", "i8"], filter_name="u*", library="np")
        assert set(arrays.keys()) == {"i4", "i8"}
        assert arrays["i4"].tolist() == list(range(-15, 15))
        assert arrays["i8"].tolist() == list(range(-15, 15))


def test_tuple_branchname():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        arrays = sample.arrays(["i4", "i8"], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(-15, 15))
        assert arrays[1].tolist() == list(range(-15, 15))

        arrays = sample.arrays(["i4", "i4"], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(-15, 15))
        assert arrays[1].tolist() == list(range(-15, 15))


def test_interpretation():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample["i2"].array(">u2", library="np").tolist() == list(
            range(65521, 65536)
        ) + list(range(0, 15))

        arrays = sample.arrays({"i1": ">u1", "i2": ">u2"}, library="np")
        assert set(arrays.keys()) == {"i1", "i2"}
        assert arrays["i1"].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays["i2"].tolist() == list(range(65521, 65536)) + list(range(0, 15))

        arrays = sample.arrays([("i1", ">u1"), ("i2", ">u2")], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays[1].tolist() == list(range(65521, 65536)) + list(range(0, 15))

        arrays = sample.arrays({"i1": ">u1", "i2": None}, library="np")
        assert set(arrays.keys()) == {"i1", "i2"}
        assert arrays["i1"].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays["i2"].tolist() == list(range(-15, 15))

        arrays = sample.arrays([("i1", ">u1"), ("i2", None)], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(241, 256)) + list(range(0, 15))
        assert arrays[1].tolist() == list(range(-15, 15))

        with pytest.raises(ValueError, match="multiple interpretations"):
            sample.arrays([("i1", ">u1"), ("i1", None)], library="np", how=tuple)


def test_compute():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays("i4 + 100", library="np")["i4 + 100"].tolist() == list(
            range(85, 115)
        )

        arrays = sample.arrays(["i4 + 100", "i8 + 100"], library="np")
        assert set(arrays.keys()) == {"i4 + 100", "i8 + 100"}
        assert arrays["i4 + 100"].tolist() == list(range(85, 115))
        assert arrays["i8 + 100"].tolist() == list(range(85, 115))

        arrays = sample.arrays(["i4 + 100", "i4 + 200"], library="np")
        assert set(arrays.keys()) == {"i4 + 100", "i4 + 200"}
        assert arrays["i4 + 100"].tolist() == list(range(85, 115))
        assert arrays["i4 + 200"].tolist() == list(range(185, 215))

        arrays = sample.arrays(["i4 + 100", "i4 + 100"], library="np")
        assert set(arrays.keys()) == {"i4 + 100"}
        assert arrays["i4 + 100"].tolist() == list(range(85, 115))

        arrays = sample.arrays(["i4 + 100", "i4 + 100"], library="np", how=tuple)
        assert isinstance(arrays, tuple) and len(arrays) == 2
        assert arrays[0].tolist() == list(range(85, 115))
        assert arrays[1].tolist() == list(range(85, 115))


def test_cut():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays("i4 + 100", cut="i4 > 0", library="np")[
            "i4 + 100"
        ].tolist() == list(range(101, 115))

        arrays = sample.arrays(["i4 + 100", "i8 + 100"], cut="i4 > 0", library="np")
        assert set(arrays.keys()) == {"i4 + 100", "i8 + 100"}
        assert arrays["i4 + 100"].tolist() == list(range(101, 115))
        assert arrays["i8 + 100"].tolist() == list(range(101, 115))

        arrays = sample.arrays(["i4", "i8"], cut="i4 > 0", library="np")
        assert set(arrays.keys()) == {"i4", "i8"}
        assert arrays["i4"].tolist() == list(range(1, 15))
        assert arrays["i8"].tolist() == list(range(1, 15))


def test_aliases():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert sample.arrays(
            "whatever", aliases={"whatever": "i4 + 100"}, library="np"
        )["whatever"].tolist() == list(range(85, 115))

        arrays = sample.arrays(
            ["one", "two"], aliases={"one": "i4 + 100", "two": "i8 + 100"}, library="np"
        )
        assert set(arrays.keys()) == {"one", "two"}
        assert arrays["one"].tolist() == list(range(85, 115))
        assert arrays["two"].tolist() == list(range(85, 115))

        arrays = sample.arrays(
            ["one", "two"], aliases={"one": "i4 + 100", "two": "one"}, library="np"
        )
        assert set(arrays.keys()) == {"one", "two"}
        assert arrays["one"].tolist() == list(range(85, 115))
        assert arrays["two"].tolist() == list(range(85, 115))

        with pytest.raises(ValueError, match="recursively defined with aliases"):
            sample.arrays(
                ["one", "two"], aliases={"one": "two", "two": "one"}, library="np"
            )

        arrays = sample.arrays(
            ["one", "two"],
            cut="one > 100",
            aliases={"one": "i4 + 100", "two": "i8 + 100"},
            library="np",
        )
        assert set(arrays.keys()) == {"one", "two"}
        assert arrays["one"].tolist() == list(range(101, 115))
        assert arrays["two"].tolist() == list(range(101, 115))

        arrays = sample.arrays(
            ["i4"],
            cut="one > 100",
            aliases={"one": "i4 + 100", "two": "i8 + 100"},
            library="np",
        )
        assert set(arrays.keys()) == {"i4"}
        assert arrays["i4"].tolist() == list(range(1, 15))


def test_jagged():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        assert [x.tolist() for x in sample.arrays("Ai4", library="np")["Ai4"]] == [
            [],
            [-15],
            [-15, -13],
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
            [5, 7, 9, 11],
            [],
            [10],
            [10, 12],
            [10, 12, 14],
            [10, 12, 14, 16],
        ]
