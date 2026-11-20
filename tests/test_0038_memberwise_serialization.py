# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import skhep_testdata

import uproot


def test_read_efficiency_with_streamer():
    with uproot.open(skhep_testdata.data_path("uproot-issue38c.root")) as fp:
        eff = fp["TEfficiencyName"]
        assert eff
        assert eff.member("fBeta_bin_params").tolist() == [
            (-1.0, -2.0),
            (2.0, 4.0),
            (4.0, 8.0),
            (8.0, 16.0),
            (16.0, 32.0),
            (32.0, 64.0),
            (64.0, 128.0),
            (128.0, 256.0),
            (256.0, 512.0),
            (512.0, 1024.0),
            (1024.0, 2048.0),
            (-1.0, -2.0),
            (1.0, 1.0),
        ]


def test_read_efficiency_without_streamer(reset_classes):
    # this file has no streamer for pair<double,double>
    with uproot.open(skhep_testdata.data_path("uproot-issue209.root")) as fp:
        eff = fp["TEfficiencyName"]
        assert len(eff.member("fBeta_bin_params")) == 0
        assert eff.member("fTotalHistogram").classname == "TH1D"
