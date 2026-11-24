# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_recreate(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    f1 = ROOT.TFile(filename, "recreate")
    mat = ROOT.TMatrixD(3, 3)
    mat[0, 1] = 4
    mat[1, 0] = 8
    mat[2, 2] = 3
    mat.Write("mat")
    f1.Close()

    with uproot.open(filename) as f2:
        assert f2["mat"].member("fNrows") == 3
        assert f2["mat"].member("fNcols") == 3
        assert np.array_equal(
            f2["mat"].member("fElements"), [0, 4, 0, 8, 0, 0, 0, 0, 3]
        )
