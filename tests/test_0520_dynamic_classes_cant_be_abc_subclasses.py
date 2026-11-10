# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pickle
import os

import uproot
import pytest


def test_pickle(tests_directory):
    with open(os.path.join(tests_directory, "samples/h_dynamic.pkl"), "rb") as f:
        histogram = pickle.load(f)

    assert histogram.member("fName") == "h"
    assert len(list(histogram.axis(0))) == 100
    assert histogram.axis(0).edges().tolist() == pytest.approx(
        [i / 100 for i in range(101)]
    )
