# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy as np
import skhep_testdata
from numpy.testing import assert_array_equal

import uproot


def test_read_TProfile2D():
    file = skhep_testdata.data_path("uproot-issue-227a.root")

    with uproot.open(file) as h:
        T = h["hprof2d"]

    assert T.kind == "MEAN"
    assert_array_equal(T.axis("x").edges(), np.array([1.0, 2.0, 3.0]))
    assert_array_equal(T.axis("y").edges(), np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.sum(T.counts(flow=True)) == 12
    assert_array_equal(T.values().tolist(), [[1.0, 2.0, 0.0], [2.0, 4.0, 6.0]])


def test_read_TProfile3D():
    file = skhep_testdata.data_path("uproot-issue-227b.root")

    with uproot.open(file) as h:
        T = h["hprof3d"]

    assert T.kind == "MEAN"
    assert_array_equal(T.axis("x").edges(), np.array([1.0, 2.0, 3.0]))
    assert_array_equal(T.axis("y").edges(), np.array([1.0, 2.0, 3.0, 4.0]))
    assert_array_equal(T.axis("z").edges(), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.sum(T.counts(flow=True)) == 12
    assert_array_equal(
        T.values().tolist(),
        [
            [[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]],
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
        ],
    )
