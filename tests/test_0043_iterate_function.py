# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_num_entries_for():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as events:
        assert events.num_entries_for("1 kB") == 4
        assert events.num_entries_for("10 kB") == 37
        assert events.num_entries_for("0.1 MB") == 372
        assert events.num_entries == 2421


def test_num_entries_for_2():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        assert events.num_entries_for("1 kB") == 7
        assert events.num_entries_for("10 kB") == 70
        assert events.num_entries_for("0.1 MB") == 696
        assert events.num_entries == 2304


def test_iterate_1():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        for i, arrays in enumerate(events.iterate(step_size="0.1 MB", library="np")):
            if i in range(3):
                assert len(arrays["px1"]) == 696
            elif i == 3:
                assert len(arrays["px1"]) == 216
            else:
                assert False


def test_iterate_2():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        for i, arrays in enumerate(events.iterate("px1", step_size=1000, library="np")):
            if i == 0:
                assert len(arrays["px1"]) == 1000
                assert arrays["px1"][:5].tolist() == [
                    -41.1952876442,
                    35.1180497674,
                    35.1180497674,
                    34.1444372454,
                    22.7835819537,
                ]
            elif i == 1:
                assert len(arrays["px1"]) == 1000
                assert arrays["px1"][:5].tolist() == [
                    26.043758785,
                    26.043758785,
                    25.9962042016,
                    -44.4626620943,
                    28.2794901505,
                ]
            elif i == 2:
                assert len(arrays["px1"]) == 304
                assert arrays["px1"][:5].tolist() == [
                    -43.3783782352,
                    -43.3783782352,
                    -43.2444221651,
                    -20.2126675303,
                    43.7131175076,
                ]
            else:
                assert False


def test_iterate_pandas_1():
    pandas = pytest.importorskip("pandas")
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        for i, arrays in enumerate(events.iterate("px1", step_size=1000, library="pd")):
            if i == 0:
                assert arrays["px1"].index.values[0] == 0
                assert arrays["px1"].index.values[-1] == 999
            elif i == 1:
                assert arrays["px1"].index.values[0] == 1000
                assert arrays["px1"].index.values[-1] == 1999
            elif i == 2:
                assert arrays["px1"].index.values[0] == 2000
                assert arrays["px1"].index.values[-1] == 2303
            else:
                assert False


def test_iterate_pandas_2():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as events:
        for i, arrays in enumerate(
            events.iterate("Muon_Px", step_size=1000, library="pd")
        ):
            if i == 0:
                assert arrays.index.values[0] == 0
                assert arrays.index.values[-1] == 999
            elif i == 1:
                assert arrays.index.values[0] == 1000
                assert arrays.index.values[-1] == 1999
            elif i == 2:
                assert arrays.index.values[0] == 2000
                assert arrays.index.values[-1] == 2420
            else:
                assert False


def test_iterate_report_1():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        for i, (arrays, report) in enumerate(
            events.iterate("px1", step_size=1000, report=True, library="np")
        ):
            if i == 0:
                assert report.tree_entry_start == 0
                assert report.tree_entry_stop == 1000
                assert report.file_path == skhep_testdata.data_path("uproot-Zmumu.root")
            elif i == 1:
                assert report.tree_entry_start == 1000
                assert report.tree_entry_stop == 2000
                assert report.file_path == skhep_testdata.data_path("uproot-Zmumu.root")
            elif i == 2:
                assert report.tree_entry_start == 2000
                assert report.tree_entry_stop == 2304
                assert report.file_path == skhep_testdata.data_path("uproot-Zmumu.root")
            else:
                assert False


def test_iterate_report_2():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as events:
        for i, (arrays, report) in enumerate(
            events.iterate("Muon_Px", step_size=1000, report=True, library="np")
        ):
            if i == 0:
                assert report.tree_entry_start == 0
                assert report.tree_entry_stop == 1000
                assert report.file_path == skhep_testdata.data_path("uproot-HZZ.root")
            elif i == 1:
                assert report.tree_entry_start == 1000
                assert report.tree_entry_stop == 2000
                assert report.file_path == skhep_testdata.data_path("uproot-HZZ.root")
            elif i == 2:
                assert report.tree_entry_start == 2000
                assert report.tree_entry_stop == 2421
                assert report.file_path == skhep_testdata.data_path("uproot-HZZ.root")
            else:
                assert False


def test_function_iterate():
    files = [
        skhep_testdata.data_path(f"uproot-sample-{x}-uncompressed.root") + ":sample"
        for x in [
            "5.23.02",
            "5.24.00",
            "5.25.02",
            "5.26.00",
            "5.27.02",
            "5.28.00",
            "5.29.02",
            "5.30.00",
            "6.08.04",
            "6.10.05",
            "6.14.00",
            "6.16.00",
            "6.18.00",
            "6.20.04",
        ]
    ]
    expect = 0
    for arrays, report in uproot.iterate(files, "i8", report=True, library="np"):
        assert arrays["i8"][:5].tolist() == [-15, -14, -13, -12, -11]
        assert report.global_entry_start == expect
        assert report.global_entry_stop == expect + len(arrays["i8"])
        expect += len(arrays["i8"])


def test_function_iterate_pandas():
    pandas = pytest.importorskip("pandas")
    files = [
        skhep_testdata.data_path(f"uproot-sample-{x}-uncompressed.root") + ":sample"
        for x in [
            "5.23.02",
            "5.24.00",
            "5.25.02",
            "5.26.00",
            "5.27.02",
            "5.28.00",
            "5.29.02",
            "5.30.00",
            "6.08.04",
            "6.10.05",
            "6.14.00",
            "6.16.00",
            "6.18.00",
            "6.20.04",
        ]
    ]
    expect = 0
    for arrays, report in uproot.iterate(files, "i8", report=True, library="pd"):
        assert arrays["i8"].values[:5].tolist() == [-15, -14, -13, -12, -11]
        assert arrays.index.values[0] == expect
        assert report.global_entry_start == expect
        assert report.global_entry_stop == expect + len(arrays["i8"])
        expect += len(arrays["i8"])


def test_function_iterate_pandas_2():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("akimbo")
    files = [
        skhep_testdata.data_path("uproot-HZZ.root") + ":events",
        skhep_testdata.data_path("uproot-HZZ-uncompressed.root") + ":events",
        skhep_testdata.data_path("uproot-HZZ-zlib.root") + ":events",
        skhep_testdata.data_path("uproot-HZZ-lz4.root") + ":events",
    ]
    expect = 0
    for arrays, report in uproot.iterate(files, "Muon_Px", report=True, library="pd"):
        assert arrays["Muon_Px"].index.values[0] == expect
        expect += report.tree.num_entries
