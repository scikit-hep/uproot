# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    awkward = pytest.importorskip("awkward")
    with uproot.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root")) as f:
        array = f["t/x"].array(entry_stop=2)
        assert str(awkward.type(array)) == "2 * var * var * float64"
