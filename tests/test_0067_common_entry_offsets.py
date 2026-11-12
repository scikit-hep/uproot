# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_common_offsets():
    # this file has terrible branch alignment
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert f["ntuple;1"].common_entry_offsets() == [0, 75000]

    # this file has just one branch
    with uproot.open(skhep_testdata.data_path("uproot-foriter.root")) as f:
        assert f["foriter;1"].common_entry_offsets() == [
            0,
            6,
            12,
            18,
            24,
            30,
            36,
            42,
            46,
        ]

    with uproot.open(skhep_testdata.data_path("uproot-small-dy-nooffsets.root")) as f:
        assert f["tree;1"].common_entry_offsets() == [0, 200, 400, 501]
        assert f["tree;1"].common_entry_offsets(filter_name=["Jet_pt"]) == [
            0,
            200,
            397,
            400,
            501,
        ]
        assert f["tree;1"].common_entry_offsets(filter_name=["Jet_pt", "nJet"]) == [
            0,
            200,
            400,
            501,
        ]
