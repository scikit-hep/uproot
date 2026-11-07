# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import os
import uproot
import pytest
import skhep_testdata

ak = pytest.importorskip("awkward")

# There are other tests for hidden keys in test_1406_improved_rntuple_methods.py

data = ak.Array(
    {
        "union1": [
            {"a": 1, "b": 1},
            "two",
            {"a": 3, "b": 3},
            "four",
            "five",
        ],
        "union2": [
            {"a": 1, "b": 1},
            {"a": 2, "b": 2, "c": 2},
            {"a": 3, "b": 3},
            {"a": 4, "b": 4, "c": 4},
            "five",
        ],
    }
)


def test_struct_with_union(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)

    obj = uproot.open(filepath)["ntuple"]
    assert obj.keys() == ["union1", "union2"]
    assert obj.arrays().to_list() == data.to_list()


def test_atomic():
    filepath = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    obj = uproot.open(filepath)["ntuple"]

    assert obj["atomic_int"].keys() == []
