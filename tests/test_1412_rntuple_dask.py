# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy
import pytest

import uproot

ak = pytest.importorskip("awkward")
dask = pytest.importorskip("dask")

data = ak.Array(
    {
        "ints": [1, 2, 3, 4, 5],
        "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
        "strings": ["one", "two", "three", "four", "five"],
    }
)


def test_dask(tmp_path):
    filepath1 = os.path.join(tmp_path, "test1.root")
    filepath2 = os.path.join(tmp_path, "test2.root")

    with uproot.recreate(filepath1) as file:
        file.mkrntuple("ntuple", data)

    with uproot.recreate(filepath2) as file:
        file.mkrntuple("ntuple", data)

    dask_arr = uproot.dask(f"{tmp_path}/test*.root:ntuple")

    arr = dask_arr.compute()

    assert ak.array_equal(arr[:5], data)
    assert ak.array_equal(arr[5:], data)
