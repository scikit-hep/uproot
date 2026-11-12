# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test_ttime_custom():
    filename = skhep_testdata.data_path("uproot-issue-1275.root")

    with uproot.open(filename) as file:
        spline = file["spline"]

        assert spline.member("fPoly")[0].member("fX") == 1.0
        assert spline.member("fPoly")[0].member("fY") == 1.0
        assert spline.member("fPoly")[0].member("fB") == 2.5
        assert spline.member("fPoly")[0].member("fC") == 0.0
        assert spline.member("fPoly")[0].member("fD") == 0.5

        assert spline.member("fPoly")[1].member("fX") == 2.0
        assert spline.member("fPoly")[1].member("fY") == 4.0
        assert spline.member("fPoly")[1].member("fB") == 4.0
        assert spline.member("fPoly")[1].member("fC") == 1.5
        assert spline.member("fPoly")[1].member("fD") == -0.5

        assert spline.member("fPoly")[2].member("fX") == 3.0
        assert spline.member("fPoly")[2].member("fY") == 9.0
        assert spline.member("fPoly")[2].member("fB") == 5.5
        assert spline.member("fPoly")[2].member("fC") == 1.0
        assert spline.member("fPoly")[2].member("fD") == 1.7142857142857144
