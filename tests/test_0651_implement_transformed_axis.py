# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

hist = pytest.importorskip("hist")


def test(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    h_reg = hist.Hist(
        hist.axis.Regular(19, 0.1, 1e4, transform=hist.axis.transform.log)
    )
    h_var = hist.Hist(hist.axis.Variable(np.logspace(-1, 4, 20)))
    h_reg.fill([0.15, 1.5, 15, 150, 1500, 15000])
    h_var.fill([0.15, 1.5, 15, 150, 1500, 15000])

    expectation = [
        0.10000000000000002,
        0.1832980710832436,
        0.3359818286283784,
        0.6158482110660265,
        1.128837891684689,
        2.0691380811147897,
        3.79269019073225,
        6.951927961775609,
        12.742749857031336,
        23.357214690901213,
        42.81332398719395,
        78.47599703514618,
        143.84498882876622,
        263.6650898730361,
        483.2930238571756,
        885.8667904100829,
        1623.7767391887219,
        2976.35144163132,
        5455.594781168521,
        10000.00000000001,
    ]

    with uproot.writing.recreate(filename) as f:
        f["h_reg"] = h_reg
        f["h_var"] = h_var

    with uproot.open(filename) as f:
        assert f["h_reg"].to_hist().axes[0].edges.tolist() == pytest.approx(expectation)
        assert f["h_var"].to_hist().axes[0].edges.tolist() == pytest.approx(expectation)
