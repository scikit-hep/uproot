# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_read_efficiency_with_streamer():
    with uproot.open(skhep_testdata.data_path("uproot-issue38c.root")) as fp:
        eff = fp["TEfficiencyName"]
        assert eff
        assert [pair.members for pair in eff.members["fBeta_bin_params"]] == [
            {"first": -1.0, "second": -2.0},
            {"first": 2.0, "second": 4.0},
            {"first": 4.0, "second": 8.0},
            {"first": 8.0, "second": 16.0},
            {"first": 16.0, "second": 32.0},
            {"first": 32.0, "second": 64.0},
            {"first": 64.0, "second": 128.0},
            {"first": 128.0, "second": 256.0},
            {"first": 256.0, "second": 512.0},
            {"first": 512.0, "second": 1024.0},
            {"first": 1024.0, "second": 2048.0},
            {"first": -1.0, "second": -2.0},
            {"first": 1.0, "second": 1.0},
        ]


def test_read_efficiency_without_streamer(reset_classes):
    with uproot.open(skhep_testdata.data_path("uproot-issue209.root")) as fp:
        with pytest.raises(
            NotImplementedError,
            match=r"streamerless memberwise serialization of class AsVector\(pair<double,double>\)",
        ):
            fp["TEfficiencyName"]
