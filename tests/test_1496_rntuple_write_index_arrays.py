# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import pytest

import uproot

import numpy as np

ak = pytest.importorskip("awkward")

record_array = ak.Array(
    [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}, {"x": 7, "y": 8}]
).layout
regular_array = ak.to_regular(np.arange(12).reshape(-1, 3)).layout
listoffset_array = ak.contents.ListOffsetArray(
    ak.index.Index64([0, 0, 1, 2]),
    record_array,
)
index_listoffset_array = ak.contents.IndexedArray(
    ak.index.Index64([1, 0, 2, 2]),
    listoffset_array,
)
listoffset_regular_array = ak.contents.ListOffsetArray(
    ak.index.Index64([0, 0, 1, 2, 2]),
    regular_array,
)
record_index64_array = ak.contents.IndexedArray(
    ak.index.Index64([1, 0, 2, 2]),
    record_array,
)
record_index32_array = ak.contents.IndexedArray(
    ak.index.Index32([1, 0, 2, 2]),
    record_array,
)
record_index64_option_array = ak.contents.IndexedOptionArray(
    ak.index.Index64([1, -1, 2, 2]),
    record_array,
)
record_index32_option_array = ak.contents.IndexedOptionArray(
    ak.index.Index32([1, -1, 2, 2]),
    record_array,
)
record_list64_array = ak.contents.ListArray(
    ak.index.Index64([1, 0, 2, 2]),
    ak.index.Index64([4, 4, 4, 4]),
    record_array,
)
record_list32_array = ak.contents.ListArray(
    ak.index.Index32([1, 0, 2, 2]),
    ak.index.Index32([4, 4, 4, 4]),
    record_array,
)

data = ak.Array(
    {
        "record_index64_array": record_index64_array,
        "record_index32_array": record_index32_array,
        "record_index64_option_array": record_index64_option_array,
        "record_index32_option_array": record_index32_option_array,
        "record_list64_array": record_list64_array,
        "record_list32_array": record_list32_array,
        "index_listoffset_array": index_listoffset_array,
        "listoffset_regular_array": listoffset_regular_array,
    }
)


def test_writing_and_reading(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)

    obj = uproot.open(filepath)["ntuple"]
    arrays = obj.arrays()

    for f in data.fields:
        assert arrays[f][:4].tolist() == data[f].tolist()
        assert arrays[f][4:].tolist() == data[f].tolist()
