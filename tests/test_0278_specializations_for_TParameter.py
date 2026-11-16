# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy as np
import skhep_testdata
from numpy.testing import assert_array_equal

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue64.root")) as file:
        p = file["events/nbevents"]
        assert p.value == 500
        assert p
        assert int(p) == 500
        assert float(p) == 500.0


def test_issue_707():
    with uproot.open(skhep_testdata.data_path("uproot-issue-707.root")) as file:
        p = file["NumberOfPrimariesEdep"]
        assert p.value == 100000004
