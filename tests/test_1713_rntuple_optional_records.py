# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import pytest

import uproot


def test_optional_record_round_trip(tmp_path):
    arr = ak.Array([{"o": {"a": 1.0}}, {"o": None}])
    path = tmp_path / "opt_record.root"

    with uproot.recreate(path) as f:
        f["t"] = arr

    with uproot.open(path) as f:
        nt = f["t"]
        assert nt.arrays().tolist() == arr.tolist()
        assert nt["o"].array().tolist() == arr["o"].tolist()
        assert nt["o.a"].array().tolist() == [1.0, None]
        assert nt.arrays(filter_name="o")["o"].tolist() == arr["o"].tolist()
        assert nt["o"].typename == ""
        assert nt["o"].record.type_alias == "std::optional<>"
        assert [field.typename for field in nt.all_fields] == ["", "", "double"]


def test_optional_record_with_multiple_fields(tmp_path):
    arr = ak.Array([{"o": {"a": 1.0, "b": 2}}, {"o": None}])
    path = tmp_path / "opt_multi.root"

    with uproot.recreate(path) as f:
        f["t"] = arr

    with uproot.open(path) as f:
        nt = f["t"]
        assert nt.arrays().tolist() == arr.tolist()
        assert nt["o.a"].array().tolist() == [1.0, None]
        assert nt["o.b"].array().tolist() == [2, None]
        assert nt["o"].typename == ""
        assert nt["o"].record.type_alias == "std::optional<>"
        assert [field.typename for field in nt.all_fields] == [
            "",
            "",
            "double",
            "std::int64_t",
        ]


def test_optional_record_nested_tuple(tmp_path):
    arr = ak.Array([{"o": (1.0, {"x": 2})}, {"o": None}])
    path = tmp_path / "opt_tuple.root"

    with uproot.recreate(path) as f:
        f["t"] = arr

    with uproot.open(path) as f:
        nt = f["t"]
        assert nt.arrays().tolist() == arr.tolist()
        assert nt["o"].array().tolist() == arr["o"].tolist()
        assert nt["o"].typename == ""
        assert nt["o"].record.type_alias == "std::optional<>"
        assert [field.typename for field in nt.all_fields] == [
            "",
            "",
            "double",
            "",
            "std::int64_t",
        ]


def test_optional_record_nested_list(tmp_path):
    arr = ak.Array([{"o": [{"a": 1.0}]}, {"o": None}])
    path = tmp_path / "opt_list.root"

    with uproot.recreate(path) as f:
        f["t"] = arr

    with uproot.open(path) as f:
        nt = f["t"]
        assert nt.arrays().tolist() == arr.tolist()
        assert nt["o"].array().tolist() == arr["o"].tolist()
        assert nt["o"].typename == ""
        assert nt["o"].record.type_alias == "std::optional<>"
        assert [field.typename for field in nt.all_fields] == ["", "", "", "double"]


def test_optional_record_extend(tmp_path):
    arr1 = ak.Array([{"o": {"a": 1.0}}, {"o": None}])
    arr2 = ak.Array([{"o": None}, {"o": {"a": 2.0}}])
    path = tmp_path / "opt_extend.root"

    with uproot.recreate(path) as f:
        f["t"] = arr1

    with uproot.update(path) as f:
        f["t"].extend(arr2)

    with uproot.open(path) as f:
        expected = ak.concatenate([arr1, arr2])
        assert f["t"].arrays().tolist() == expected.tolist()


def test_optional_record_dask(tmp_path):
    pytest.importorskip("dask_awkward")

    arr1 = ak.Array([{"o": {"a": 1.0}}, {"o": None}])
    arr2 = ak.Array([{"o": None}, {"o": {"a": 2.0}}])
    path = tmp_path / "opt_dask.root"

    with uproot.recreate(path) as f:
        f["t"] = arr1

    with uproot.update(path) as f:
        f["t"].extend(arr2)

    expected = ak.concatenate([arr1, arr2])
    dask_arr = uproot.dask(str(path) + ":t")
    assert dask_arr.compute().tolist() == expected.tolist()


def test_optional_record_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63400:
        pytest.skip("ROOT version does not support RNTuple v1.0.0.0")

    filepath = str(tmp_path / "opt_record_root.root")
    arr = ak.Array([{"o": {"a": 1.0}}, {"o": None}])

    with uproot.recreate(filepath) as f:
        f["ntuple"] = arr

    if ROOT.gROOT.GetVersionInt() < 63600:
        RT = ROOT.Experimental.RNTupleReader.Open("ntuple", filepath)
    else:
        RT = ROOT.RNTupleReader.Open("ntuple", filepath)
    RT.PrintInfo()
    out = capfd.readouterr().out
    assert "ntuple" in out
