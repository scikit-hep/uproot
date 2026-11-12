# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import json
import os
import queue
import re
import sys

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def test_header(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        akform = ak.forms.RecordForm(
            [
                ak.forms.NumpyForm("float64"),
                ak.forms.NumpyForm("int32"),
                ak.forms.NumpyForm("bool"),
            ],
            ["one", "two", "three"],
        )
        file.mkrntuple("ntuple", akform)

    file = uproot.open(filepath)["ntuple"]

    header = file.header
    assert header.checksum == file.footer.header_checksum

    frs = header.field_records
    assert frs[0].parent_field_id == 0
    assert frs[1].parent_field_id == 1
    assert frs[2].parent_field_id == 2
    assert frs[0].field_name == "one"
    assert frs[1].field_name == "two"
    assert frs[2].field_name == "three"
    assert frs[0].type_name == "double"
    assert frs[1].type_name == "std::int32_t"
    assert frs[2].type_name == "bool"

    crs = header.column_records
    assert crs[0].type == uproot.const.rntuple_col_type_to_num_dict["real64"]
    assert crs[1].type == uproot.const.rntuple_col_type_to_num_dict["int32"]
    assert crs[2].type == uproot.const.rntuple_col_type_to_num_dict["bit"]
    assert crs[0].field_id == 0
    assert crs[1].field_id == 1
    assert crs[2].field_id == 2
    assert crs[0].nbits == uproot.const.rntuple_col_num_to_size_dict[crs[0].type]
    assert crs[1].nbits == uproot.const.rntuple_col_num_to_size_dict[crs[1].type]
    assert crs[2].nbits == uproot.const.rntuple_col_num_to_size_dict[crs[2].type]


def test_writable(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        akform = ak.forms.RecordForm(
            [
                ak.forms.NumpyForm("int32"),
            ],
            ["one"],
        )
        rn = file.mkrntuple("ntuple", akform)
        assert re.fullmatch(r"<WritableNTuple '/ntuple' at 0x[0-9a-f]{12,}>", repr(rn))
        assert type(file["ntuple"]).__name__ == "WritableNTuple"
        assert rn.num_entries == 0


def test_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63400:
        pytest.skip("ROOT version does not support RNTuple v1.0.0.0")

    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        akform = ak.forms.RecordForm(
            [
                ak.forms.NumpyForm("float64"),
                ak.forms.NumpyForm("int32"),
            ],
            ["one", "two"],
        )
        file.mkrntuple("ntuple", akform)
    if ROOT.gROOT.GetVersionInt() < 63600:
        RT = ROOT.Experimental.RNTupleReader.Open("ntuple", filepath)
    else:
        RT = ROOT.RNTupleReader.Open("ntuple", filepath)
    RT.PrintInfo()
    out = capfd.readouterr().out
    assert "* Field 1   : one (double)" in out
    assert "* Field 2   : two (std::int32_t)" in out
