# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import io
import os

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-359.root")) as file:
        matrix = file["covmatrixOCratio"]
        num_elements = matrix.member("fNrows") * (matrix.member("fNcols") + 1) // 2
        assert num_elements == 435
        assert len(matrix.member("fElements")) == num_elements
        assert matrix.member("fElements")[:3].tolist() == pytest.approx(
            [0.07898878, 0.00017309, 0.00930861], abs=1e-8
        )
        assert matrix.member("fElements").sum() == pytest.approx(3.8886480228275087)
