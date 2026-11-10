# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys
from io import StringIO

import numpy
import pytest
import skhep_testdata

import uproot


def test_any_basket():
    interpretation = uproot.interpretation.numerical.AsDtype(">i4")

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/i4"] as branch:
        assert branch.basket(0).array(interpretation, library="np").tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
        ]
        assert branch.basket(1).array(interpretation, library="np").tolist() == [
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
        ]
        assert branch.basket(2).array(interpretation, library="np").tolist() == [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
        ]
        assert branch.basket(3).array(interpretation, library="np").tolist() == [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]
        assert branch.basket(4).array(interpretation, library="np").tolist() == [
            13,
            14,
        ]


def test_stitching_arrays():
    interpretation = uproot.interpretation.numerical.AsDtype("i8")
    expectation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    basket_arrays = [[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], [10], [11, 12, 13, 14]]
    basket_arrays = [numpy.array(x) for x in basket_arrays]
    entry_offsets = numpy.array([0, 5, 7, 7, 10, 11, 15])
    library = uproot.interpretation.library._libraries["np"]

    for start in range(16):
        for stop in range(15, -1, -1):
            actual = interpretation.final_array(
                basket_arrays, start, stop, entry_offsets, library, None, {}
            )
            assert expectation[start:stop] == actual.tolist()


def _names_entries_to_ranges_or_baskets(self, branch_names, entry_start, entry_stop):
    out = []
    for name in branch_names:
        branch = self[name]
        for basket_num, range_or_basket in branch.entries_to_ranges_or_baskets(
            entry_start, entry_stop
        ):
            out.append((branch, basket_num, range_or_basket))
    return out


def test_names_entries_to_ranges_or_baskets():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        out = _names_entries_to_ranges_or_baskets(sample, ["i4"], 0, 30)
        assert [x[1] for x in out] == [0, 1, 2, 3, 4]
        assert [x[2] for x in out] == [
            (6992, 7091),
            (16085, 16184),
            (25939, 26038),
            (35042, 35141),
            (40396, 40475),
        ]


def test_ranges_or_baskets_to_arrays():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        branch = sample["i4"]

        ranges_or_baskets = _names_entries_to_ranges_or_baskets(sample, ["i4"], 0, 30)
        branchid_interpretation = {
            branch.cache_key: uproot.interpretation.numerical.AsDtype(">i4")
        }
        entry_start, entry_stop = (0, 30)
        decompression_executor = uproot.source.futures.TrivialExecutor()
        interpretation_executor = uproot.source.futures.TrivialExecutor()
        library = uproot.interpretation.library._libraries["np"]

        arrays = {}
        uproot.behaviors.TBranch._ranges_or_baskets_to_arrays(
            sample,
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            library,
            arrays,
            False,
            {},
        )
        assert arrays[branch.cache_key].tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ]


@pytest.mark.parametrize(
    "handler",
    [uproot.source.file.MultithreadedFileSource, uproot.source.file.MemmapSource],
)
def test_branch_array_1(handler):
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        handler=handler,
    )["sample/i4"] as branch:
        assert branch.array(
            uproot.interpretation.numerical.AsDtype(">i4"), library="np"
        ).tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ]


@pytest.mark.parametrize(
    "handler",
    [uproot.source.file.MultithreadedFileSource, uproot.source.file.MemmapSource],
)
def test_branch_array_2(handler):
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        handler=handler,
    )["sample/i4"] as branch:
        assert branch.array(
            uproot.interpretation.numerical.AsDtype(">i4"),
            entry_start=3,
            entry_stop=-5,
            library="np",
        ).tolist() == [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]


@pytest.mark.parametrize(
    "handler",
    [uproot.source.file.MultithreadedFileSource, uproot.source.file.MemmapSource],
)
def test_branch_array_3(handler):
    executor = uproot.ThreadPoolExecutor()
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        handler=handler,
        interpretation_executor=executor,
        decompression_executor=executor,
    )["sample/i4"] as branch:
        assert branch.array(
            uproot.interpretation.numerical.AsDtype(">i4"),
            entry_start=3,
            entry_stop=-5,
            library="np",
        ).tolist() == [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]


@pytest.mark.parametrize(
    "handler",
    [uproot.source.file.MultithreadedFileSource, uproot.source.file.MemmapSource],
)
def test_branch_array_4(handler):
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        handler=handler,
    )["sample/i4"] as branch:
        with pytest.raises(
            ValueError,
            match=r"wrong number of bytes \(28\) for interpretation AsDtype\('>i8'\)",
        ):
            branch.array(uproot.interpretation.numerical.AsDtype(">i8"), library="np")


def test_cache():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        object_cache=100,
        array_cache="100 MB",
        handler=uproot.source.file.MemmapSource,
    ) as f:
        assert f.cache_key == "db4be408-93ad-11ea-9027-d201a8c0beef:/"
        assert f["sample"].cache_key == "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1"
        assert (
            f["sample/i4"].cache_key
            == "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1:i4(16)"
        )
        i4 = f["sample/i4"]
        assert list(f.file.array_cache) == []
        i4.array(uproot.interpretation.numerical.AsDtype(">i4"), library="np")
        assert list(f.file.array_cache) == [
            "db4be408-93ad-11ea-9027-d201a8c0beef:/sample;1:i4(16):i4:AsDtype(Bi4(),Li4()):0-30:np"
        ]

    with pytest.raises(OSError, match="memmap is closed"):
        i4.array(
            uproot.interpretation.numerical.AsDtype(">i4"), entry_start=3, library="np"
        )


def test_pandas():
    pandas = pytest.importorskip("pandas")
    executor = uproot.ThreadPoolExecutor()
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"),
        interpretation_executor=executor,
        decompression_executor=executor,
    )["sample/i4"] as branch:
        series = branch.array(
            uproot.interpretation.numerical.AsDtype(">i4"),
            entry_start=3,
            entry_stop=-5,
            library="pd",
        )
        assert isinstance(series, pandas.Series)
        assert series.values.tolist() == [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]
