# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    # an empty TTree yields no batches, instead of dividing by zero
    assert (
        list(
            uproot.iterate(
                skhep_testdata.data_path("uproot-issue335.root") + ":empty_tree",
                ["var"],
                library="np",
            )
        )
        == []
    )
