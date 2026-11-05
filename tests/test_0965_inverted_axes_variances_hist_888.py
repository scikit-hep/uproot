# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import numpy
import uproot
import skhep_testdata

pytest.importorskip("hist")
ROOT = pytest.importorskip("ROOT")


def test_axes_of_variances_to_hist_2D_weighted():
    hroot2 = ROOT.TH2F("hroot2", "", 2, 0, 2, 3, 0, 3)
    hroot2.Sumw2()

    # each bin gets its own weight, to tell the axes apart
    for x in range(2):
        for y in range(3):
            hroot2.Fill(x + 0.5, y + 0.5, 1 + x + 10 * y)
            hroot2.Fill(x + 0.5, y + 0.5, 2)

    huproot2 = uproot.from_pyroot(hroot2)
    expected = [[(1 + x + 10 * y) ** 2 + 4.0 for y in range(3)] for x in range(2)]
    assert huproot2.variances().tolist() == expected
    assert huproot2.to_hist().variances().tolist() == expected


def test_axes_variances_to_hist_3D_weighted():
    hroot3 = ROOT.TH3F("hroot3", "", 3, 0, 3, 2, 0, 2, 5, 0, 5)
    hroot3.Sumw2()

    for x in range(3):
        for y in range(2):
            for z in range(5):
                hroot3.Fill(x + 0.5, y + 0.5, z + 0.5, 1 + x + 10 * y + 100 * z)
                hroot3.Fill(x + 0.5, y + 0.5, z + 0.5, 2)

    huproot3 = uproot.from_pyroot(hroot3)
    expected = numpy.array(
        [
            [
                [(1 + x + 10 * y + 100 * z) ** 2 + 4.0 for z in range(5)]
                for y in range(2)
            ]
            for x in range(3)
        ]
    )
    assert numpy.array_equal(huproot3.variances(), expected)
    assert numpy.array_equal(huproot3.to_hist().variances(), expected)


def test_users_2d_weighted_histogram():
    with uproot.open(skhep_testdata.data_path("uproot-issue-888.root")) as f:
        h = f["hrecoVsgen_ll_cHel_400mttbar"]
        assert h.variances().shape == (192, 768)
        assert h.variances()[96, 384] == pytest.approx(663.7100270851231)
        assert numpy.array_equal(h.variances(), h.to_hist().variances())
