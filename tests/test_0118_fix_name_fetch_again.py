# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_keys():
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree"
    ] as t:
        assert t.keys(filter_name="P3.Px") == ["evt/P3/P3.Px"]
        assert t.keys(filter_name="/P3.Px") == []
        assert t.keys(filter_name="P3/P3.Px") == []
        assert t.keys(filter_name="evt/P3/P3.Px") == ["evt/P3/P3.Px"]
        assert t.keys(filter_name="/evt/P3/P3.Px") == ["evt/P3/P3.Px"]
        assert t["evt"].keys(filter_name="P3.Px") == ["P3/P3.Px"]
        assert t["evt"].keys(filter_name="/P3.Px") == []
        assert t["evt"].keys(filter_name="P3/P3.Px") == ["P3/P3.Px"]
        assert t["evt"].keys(filter_name="/P3/P3.Px") == ["P3/P3.Px"]
        assert t["evt"].keys(filter_name="evt/P3/P3.Px") == []
        assert t["evt/P3"].keys(filter_name="P3.Px") == ["P3.Px"]
        assert t["evt/P3"].keys(filter_name="/P3.Px") == ["P3.Px"]
        assert t["evt/P3"].keys(filter_name="P3/P3.Px") == []


def test_numpy():
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree"
    ] as t:
        assert list(t.arrays(filter_name="P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t.arrays(filter_name="/P3.Px", library="np").keys()) == []
        assert list(t.arrays(filter_name="P3/P3.Px", library="np").keys()) == []
        assert list(t.arrays(filter_name="evt/P3/P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert list(t.arrays(filter_name="/evt/P3/P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert list(t["evt"].arrays(filter_name="P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert list(t["evt"].arrays(filter_name="/P3.Px", library="np").keys()) == []
        assert list(t["evt"].arrays(filter_name="P3/P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert list(t["evt"].arrays(filter_name="/P3/P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert (
            list(t["evt"].arrays(filter_name="evt/P3/P3.Px", library="np").keys()) == []
        )
        assert list(t["evt/P3"].arrays(filter_name="P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert list(t["evt/P3"].arrays(filter_name="/P3.Px", library="np").keys()) == [
            "P3.Px"
        ]
        assert (
            list(t["evt/P3"].arrays(filter_name="P3/P3.Px", library="np").keys()) == []
        )

        assert list(t.arrays("P3.Px", library="np").keys()) == ["P3.Px"]
        with pytest.raises(SyntaxError, match="invalid syntax"):
            t.arrays("/P3.Px", library="np")
        assert list(t.arrays("evt/P3/P3.Px", library="np").keys()) == ["evt/P3/P3.Px"]
        assert list(t.arrays("/evt/P3/P3.Px", library="np").keys()) == ["/evt/P3/P3.Px"]
        assert list(t["evt"].arrays("P3.Px", library="np").keys()) == ["P3.Px"]
        with pytest.raises(SyntaxError, match="invalid syntax"):
            t["evt"].arrays("/P3.Px", library="np")
        assert list(t["evt"].arrays("P3/P3.Px", library="np").keys()) == ["P3/P3.Px"]
        assert list(t["evt/P3"].arrays("P3.Px", library="np").keys()) == ["P3.Px"]
        assert list(t["evt/P3"].arrays("/P3.Px", library="np").keys()) == ["/P3.Px"]


def test_awkward():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree"
    ] as t:
        assert t.arrays(filter_name="P3.Px", library="ak").fields == ["P3.Px"]
        assert t.arrays(filter_name="/P3.Px", library="ak").fields == []
        assert t.arrays(filter_name="P3/P3.Px", library="ak").fields == []
        assert t.arrays(filter_name="evt/P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t.arrays(filter_name="/evt/P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="/P3.Px", library="ak").fields == []
        assert t["evt"].arrays(filter_name="P3/P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt"].arrays(filter_name="/P3/P3.Px", library="ak").fields == [
            "P3.Px"
        ]
        assert t["evt"].arrays(filter_name="evt/P3/P3.Px", library="ak").fields == []
        assert t["evt/P3"].arrays(filter_name="P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt/P3"].arrays(filter_name="/P3.Px", library="ak").fields == [
            "P3.Px"
        ]
        assert t["evt/P3"].arrays(filter_name="P3/P3.Px", library="ak").fields == []

        assert t.arrays("P3.Px", library="ak").fields == ["P3.Px"]
        with pytest.raises(SyntaxError, match="invalid syntax"):
            t.arrays("/P3.Px", library="ak")
        assert t.arrays("evt/P3/P3.Px", library="ak").fields == ["evt/P3/P3.Px"]
        assert t.arrays("/evt/P3/P3.Px", library="ak").fields == ["/evt/P3/P3.Px"]
        assert t["evt"].arrays("P3.Px", library="ak").fields == ["P3.Px"]
        with pytest.raises(SyntaxError, match="invalid syntax"):
            t["evt"].arrays("/P3.Px", library="ak")
        assert t["evt"].arrays("P3/P3.Px", library="ak").fields == ["P3/P3.Px"]
        assert t["evt/P3"].arrays("P3.Px", library="ak").fields == ["P3.Px"]
        assert t["evt/P3"].arrays("/P3.Px", library="ak").fields == ["/P3.Px"]


def test_pandas():
    pandas = pytest.importorskip("pandas")
    with uproot.open(skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root"))[
        "tree"
    ] as t:
        assert t.arrays(filter_name="P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t.arrays(filter_name="/P3.Px", library="pd").columns.tolist() == []
        assert t.arrays(filter_name="P3/P3.Px", library="pd").columns.tolist() == []
        assert t.arrays(filter_name="evt/P3/P3.Px", library="pd").columns.tolist() == [
            "P3.Px"
        ]
        assert t.arrays(filter_name="/evt/P3/P3.Px", library="pd").columns.tolist() == [
            "P3.Px"
        ]
        assert t["evt"].arrays(filter_name="P3.Px", library="pd").columns.tolist() == [
            "P3.Px"
        ]
        assert (
            t["evt"].arrays(filter_name="/P3.Px", library="pd").columns.tolist() == []
        )
        assert t["evt"].arrays(
            filter_name="P3/P3.Px", library="pd"
        ).columns.tolist() == ["P3.Px"]
        assert t["evt"].arrays(
            filter_name="/P3/P3.Px", library="pd"
        ).columns.tolist() == ["P3.Px"]
        assert (
            t["evt"].arrays(filter_name="evt/P3/P3.Px", library="pd").columns.tolist()
            == []
        )
        assert t["evt/P3"].arrays(
            filter_name="P3.Px", library="pd"
        ).columns.tolist() == ["P3.Px"]
        assert t["evt/P3"].arrays(
            filter_name="/P3.Px", library="pd"
        ).columns.tolist() == ["P3.Px"]
        assert (
            t["evt/P3"].arrays(filter_name="P3/P3.Px", library="pd").columns.tolist()
            == []
        )

        assert t.arrays("P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        with pytest.raises(SyntaxError, match="invalid syntax"):
            t.arrays("/P3.Px", library="pd")
        assert t.arrays("evt/P3/P3.Px", library="pd").columns.tolist() == [
            "evt/P3/P3.Px"
        ]
        assert t.arrays("/evt/P3/P3.Px", library="pd").columns.tolist() == [
            "/evt/P3/P3.Px"
        ]
        assert t["evt"].arrays("P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        with pytest.raises(SyntaxError, match="invalid syntax"):
            t["evt"].arrays("/P3.Px", library="pd")
        assert t["evt"].arrays("P3/P3.Px", library="pd").columns.tolist() == [
            "P3/P3.Px"
        ]
        assert t["evt/P3"].arrays("P3.Px", library="pd").columns.tolist() == ["P3.Px"]
        assert t["evt/P3"].arrays("/P3.Px", library="pd").columns.tolist() == ["/P3.Px"]
