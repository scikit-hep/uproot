# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import pytest

import uproot


@pytest.fixture
def tuple_path(tmp_path):
    path = tmp_path / "tuple.root"
    with uproot.recreate(path) as file:
        file["tuple"] = ak.Array([(1.0, 2), (3.0, 4)])
    return path


def test_top_level_tuple_round_trip(tuple_path):
    with uproot.open(tuple_path) as file:
        ntuple = file["tuple"]
        form, _ = ntuple.to_akform()

        assert form.is_tuple
        # written like nested tuples, and renamed to match Awkward when read
        assert ntuple.field_names == ["_0", "_1"]
        assert ntuple.keys() == ["0", "1"]
        assert ntuple.arrays().tolist() == [(1.0, 2), (3.0, 4)]


def test_top_level_tuple_fields(tuple_path):
    with uproot.open(tuple_path) as file:
        ntuple = file["tuple"]

        assert ntuple["0"].array().tolist() == [1.0, 3.0]
        assert ntuple["1"].array().tolist() == [2, 4]

        # a subset of the fields is a record, so that its fields match the RFields
        for name, expected in [("0", [1.0, 3.0]), ("1", [2, 4])]:
            arrays = ntuple.arrays(filter_name=name)
            assert arrays.fields == [name]
            assert arrays[name].tolist() == expected


def test_top_level_tuple_subfields(tmp_path):
    path = tmp_path / "tuple.root"
    with uproot.recreate(path) as file:
        file["tuple"] = ak.Array([([{"x": 1}, {"x": 2}], 3), ([], 4)])

    with uproot.open(path) as file:
        ntuple = file["tuple"]

        assert ntuple.keys() == ["0", "0.x", "1"]
        assert ntuple["0"]["x"].array().tolist() == [[1, 2], []]
        assert ntuple.arrays().tolist() == [([{"x": 1}, {"x": 2}], 3), ([], 4)]


def test_top_level_tuple_dask(tuple_path):
    pytest.importorskip("dask_awkward")

    array = uproot.dask(f"{tuple_path}:tuple")
    assert array.compute().tolist() == [(1.0, 2), (3.0, 4)]
    assert array["1"].compute().tolist() == [2, 4]

    array = uproot.dask(f"{tuple_path}:tuple", filter_name="1")
    assert array.compute().tolist() == [{"1": 2}, {"1": 4}]


def test_top_level_tuple_extend(tuple_path):
    with uproot.update(tuple_path) as file:
        file["tuple"].extend({"0": [5.0], "1": [6]})
        file["tuple"].extend(ak.Array([(7.0, 8)]))

    with uproot.open(tuple_path) as file:
        assert file["tuple"].arrays().tolist() == [
            (1.0, 2),
            (3.0, 4),
            (5.0, 6),
            (7.0, 8),
        ]


@pytest.mark.parametrize(
    "array",
    [
        ak.Array([{"_2": 10, "_3": 20}]),
        ak.Array([{"_1": 10, "_0": 20}]),
    ],
)
def test_top_level_record_with_tuple_like_names(tmp_path, array):
    path = tmp_path / "record.root"
    with uproot.recreate(path) as file:
        file["record"] = array

    with uproot.open(path) as file:
        ntuple = file["record"]

        assert not ntuple.to_akform()[0].is_tuple
        assert ntuple.keys() == array.fields
        assert ntuple.arrays().tolist() == array.tolist()

    pytest.importorskip("dask_awkward")
    assert uproot.dask(f"{path}:record").compute().tolist() == array.tolist()


@pytest.mark.parametrize("is_list", [False, True])
def test_nested_tuple_fields(tmp_path, is_list):
    tuples = [(1.0, 2), (3.0, 4)]
    array = ak.Array([{"r": [t] if is_list else t} for t in tuples])
    path = tmp_path / "nested.root"
    with uproot.recreate(path) as file:
        file["nested"] = array

    def wrap(x):
        return [x] if is_list else x

    with uproot.open(path) as file:
        ntuple = file["nested"]

        assert ntuple.keys() == ["r", "r.0", "r.1"]
        assert ntuple.arrays().tolist() == array.tolist()
        assert ntuple["r.1"].array().tolist() == [wrap(2), wrap(4)]
        # a subset of the fields is a record, so that its fields match the RFields
        assert ntuple.arrays(filter_name="r.1").tolist() == [
            {"r": wrap({"1": 2})},
            {"r": wrap({"1": 4})},
        ]

    pytest.importorskip("dask_awkward")
    assert uproot.dask(f"{path}:nested").compute().tolist() == array.tolist()
    assert uproot.dask(f"{path}:nested")["r", "1"].compute().tolist() == [
        wrap(2),
        wrap(4),
    ]


@pytest.mark.parametrize(
    "record",
    [{"_2": 10, "_3": 20}, {"_1": 10, "_0": 20}, {"_0": 10, "x": 20}],
)
def test_nested_record_with_tuple_like_names(tmp_path, record):
    array = ak.Array([{"r": record}])
    path = tmp_path / "nested.root"
    with uproot.recreate(path) as file:
        file["nested"] = array

    last = list(record)[-1]
    with uproot.open(path) as file:
        ntuple = file["nested"]

        assert ntuple.keys() == ["r"] + [f"r.{name}" for name in record]
        assert ntuple.arrays().tolist() == array.tolist()
        assert ntuple[f"r.{last}"].array().tolist() == [record[last]]

    pytest.importorskip("dask_awkward")
    assert uproot.dask(f"{path}:nested").compute().tolist() == array.tolist()
    assert uproot.dask(f"{path}:nested")["r", last].compute().tolist() == [record[last]]
