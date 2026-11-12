# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        batches = list(
            f["events"].iterate("px1", step_size=1000, cut="px1 > 0", library="np")
        )
        assert [len(arrays["px1"]) for arrays in batches] == [471, 509, 148]
        assert all(numpy.all(arrays["px1"] > 0) for arrays in batches)
