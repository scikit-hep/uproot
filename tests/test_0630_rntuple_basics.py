# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


def test_flat():
    filename = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ["one_integers", "two_floats"]
        assert [r.type_name for r in R.header.field_records] == [
            "std::int32_t",
            "float",
        ]
        assert R.header.checksum == R.footer.header_checksum
        assert R.arrays(entry_stop=3)["one_integers"].tolist() == [9, 8, 7]
        assert R.arrays(entry_stop=3)["two_floats"].tolist() == pytest.approx(
            [9.9, 8.8, 7.7]
        )
        assert R.arrays("one_integers", entry_stop=3).fields == ["one_integers"]
        assert R.arrays("one_integers", entry_stop=3)["one_integers"].tolist() == [
            9,
            8,
            7,
        ]
        assert R.arrays(entry_start=1, entry_stop=3)["one_integers"].tolist() == [8, 7]

    filename = skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.arrays(entry_stop=3)["one_integers"].tolist() == [50000, 49999, 49998]
        assert R.arrays(entry_start=-3)["one_integers"].tolist() == [3, 2, 1]


def test_jagged():
    filename = skhep_testdata.data_path(
        "test_int_vfloat_tlv_vtlv_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys(recursive=False) == [
            "one_integers",
            "two_v_floats",
            "three_LV",
            "four_v_LVs",
        ]
        arrays = R.arrays()
        assert arrays["one_integers"].tolist() == [9, 8, 7, 6, 5]
        assert arrays["two_v_floats"].tolist() == [
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0],
            [2.0, 1.0],
            [0.0, -1.0],
            [-2.0],
        ]
        assert arrays["three_LV"].fields == ["pt", "eta", "phi", "mass"]
        assert arrays["three_LV"]["pt"].tolist() == [19.0, 18.0, 17.0, 17.0, 16.0]
        assert arrays["four_v_LVs"]["pt"][:2].tolist() == [
            [19.0] * 4,
            [19.0] * 4 + [18.0] * 3,
        ]
