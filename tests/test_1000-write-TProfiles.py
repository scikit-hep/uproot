# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import os
import math
import numpy as np

pytest.importorskip("hist")
ROOT = pytest.importorskip("ROOT")


def test_write_TProfile(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h1 = ROOT.TProfile("h1", "title", 2, -3.14, 2.71)
    h1.Fill(-4, 10)
    h1.Fill(-3.1, 10)
    h1.Fill(-3.1, 20)
    h1.Fill(2.7, 20)
    h1.Fill(3, 20)

    hhist = uproot.from_pyroot(h1).to_hist()

    uhist = uproot.writing.identify.to_TProfile(
        fName="h1",
        fTitle="title",
        data=np.array([10, 30, 20, 20], np.float64),
        fEntries=5.0,
        fTsumw=3.0,
        fTsumw2=3.0,
        fTsumwx=-3.5,
        fTsumwx2=26.51,
        fTsumwy=50.0,
        fTsumwy2=900.0,
        fSumw2=np.array([100, 500, 400, 400], np.float64),
        fBinEntries=np.array([1, 2, 1, 1], np.float64),
        fBinSumw2=np.array([], np.float64),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=2,
            fXmin=-3.14,
            fXmax=2.71,
        ),
    )

    with uproot.recreate(newfile) as fin:
        fin["hhist"] = hhist
        fin["uhist"] = uhist

    f = ROOT.TFile(newfile)
    h2 = f.Get("hhist")
    h3 = f.Get("uhist")

    assert h1.GetEntries() == h2.GetEntries() == h3.GetEntries() == 5
    assert h1.GetSumOfWeights() == h2.GetSumOfWeights() == h3.GetSumOfWeights() == 35
    assert (
        h1.GetBinLowEdge(1)
        == h2.GetBinLowEdge(1)
        == h3.GetBinLowEdge(1)
        == pytest.approx(-3.14)
    )
    assert (
        h1.GetBinWidth(1)
        == h2.GetBinWidth(1)
        == h3.GetBinWidth(1)
        == pytest.approx((2.71 + 3.14) / 2)
    )
    assert (
        h1.GetBinContent(0)
        == h2.GetBinContent(0)
        == h3.GetBinContent(0)
        == pytest.approx(10)
    )
    assert (
        h1.GetBinContent(1)
        == h2.GetBinContent(1)
        == h3.GetBinContent(1)
        == pytest.approx(15)
    )
    assert (
        h1.GetBinContent(2)
        == h2.GetBinContent(2)
        == h3.GetBinContent(2)
        == pytest.approx(20)
    )
    assert (
        h1.GetBinContent(3)
        == h2.GetBinContent(3)
        == h3.GetBinContent(3)
        == pytest.approx(20)
    )
    assert (
        h1.GetBinError(0) == h2.GetBinError(0) == h3.GetBinError(0) == pytest.approx(0)
    )
    assert (
        h1.GetBinError(1)
        == h2.GetBinError(1)
        == h3.GetBinError(1)
        == pytest.approx(np.sqrt(12.5))
    )
    assert (
        h1.GetBinError(2) == h2.GetBinError(2) == h3.GetBinError(2) == pytest.approx(0)
    )
    assert (
        h1.GetBinError(3) == h2.GetBinError(3) == h3.GetBinError(3) == pytest.approx(0)
    )

    assert hhist[0].variance == pytest.approx(50.0)

    f.Close()
