# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-569.root")) as f:
        assert f["MCTruthTree/MCTruthEvent/TObject/fBits"].array(
            library="np"
        ).tolist() == [50331648]
