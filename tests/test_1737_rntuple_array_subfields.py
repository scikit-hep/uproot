# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import os

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def _check_all_subfields(ntuple):
    subfields = [key for key in ntuple.keys() if "." in key]
    assert len(subfields) > 0
    for path in subfields:
        # reading the subfield directly must match reading its top-level field
        top, *rest = path.split(".")
        expected = ntuple[top].array()
        for name in rest:
            expected = expected[name]
        result = ntuple[path].array()
        assert str(result.type) == str(expected.type), path
        assert ak.array_equal(result, expected, equal_nan=True), path


def test_std_array_of_records():
    filename = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        pt = ntuple["array_lv.pt"].array()
        assert pt.tolist() == [[float(i)] * 3 for i in range(1, 6)]
        _check_all_subfields(ntuple)


def test_nested_std_arrays(tmp_path):
    y = numpy.arange(30, dtype=numpy.int64)
    records = ak.zip({"x": y + 0.5, "y": y})

    def array(content, size):
        return ak.to_regular(ak.unflatten(content, size), axis=1)

    arrays = {
        # std::array<struct>
        "arr": array(records[:15], 3),
        # std::array<std::vector<struct>>
        "arr_vec": array(ak.unflatten(records[:9], [0, 1, 2] * 3 + [0]), 2),
        # std::array<std::array<struct>>
        "arr_arr": array(array(records, 2), 3),
        # std::array<std::optional<struct>>
        "arr_opt": array(
            ak.Array([r if r["y"] % 3 else None for r in records[:10].tolist()]), 2
        ),
        # std::vector<std::array<struct>>
        "vec_arr": ak.unflatten(array(records[:20], 2), [0, 1, 2, 3, 4]),
    }
    # struct{std::array<struct>, int}
    arrays["rec_arr"] = ak.zip(
        {"a": arrays["arr"], "z": numpy.arange(5, dtype=numpy.int64)}, depth_limit=1
    )

    path = os.path.join(tmp_path, "test_1737.root")
    with uproot.recreate(path) as f:
        f.mkrntuple("ntuple", arrays)

    with uproot.open(path) as f:
        ntuple = f["ntuple"]
        assert str(ntuple["arr.x"].array().type) == "5 * 3 * float64"
        assert str(ntuple["arr_vec.y"].array().type) == "5 * 2 * var * int64"
        assert str(ntuple["arr_arr.x"].array().type) == "5 * 3 * 2 * float64"
        assert str(ntuple["rec_arr.a.x"].array().type) == "5 * 3 * float64"
        _check_all_subfields(ntuple)
