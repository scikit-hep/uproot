# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for PR #1651: fix TProfile2D/3D weighted property and counts() shape.

Test files:
  uproot-issue-227a.root  — contains a TProfile2D named "hprof2d"  (2 x 3 bins)
  uproot-issue-227b.root  — contains a TProfile3D named "hprof3d"  (2 x 3 x 4 bins)
"""

import numpy
import pytest
import skhep_testdata

import uproot


@pytest.fixture(scope="module")
def h2():
    with uproot.open(skhep_testdata.data_path("uproot-issue-227a.root")) as f:
        yield f["hprof2d"]


@pytest.fixture(scope="module")
def h3():
    with uproot.open(skhep_testdata.data_path("uproot-issue-227b.root")) as f:
        yield f["hprof3d"]


# ---------------------------------------------------------------------------
# TProfile2D
# ---------------------------------------------------------------------------


def test_tprofile2d_weighted_value(h2):
    """Histogram was not filled with weights, so weighted should be False."""
    assert h2.weighted is False


def test_tprofile2d_counts(h2):
    """counts() has the shape of values(), with or without flow."""
    counts = numpy.array([[3.0, 2.0, 0.0], [4.0, 2.0, 1.0]])
    assert h2.counts().dtype == numpy.float64
    assert h2.counts(flow=False).tolist() == counts.tolist()
    assert h2.counts(flow=True).tolist() == numpy.pad(counts, 1).tolist()
    assert h2.counts(flow=False).shape == h2.values(flow=False).shape
    assert h2.counts(flow=True).shape == h2.values(flow=True).shape


# ---------------------------------------------------------------------------
# TProfile3D
# ---------------------------------------------------------------------------


def test_tprofile3d_weighted_value(h3):
    """Histogram was not filled with weights, so weighted should be False."""
    assert h3.weighted is False


def test_tprofile3d_counts(h3):
    """counts() has the shape of values(), with or without flow."""
    counts = numpy.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    assert h3.counts().dtype == numpy.float64
    assert h3.counts(flow=False).tolist() == counts.tolist()
    with_flow = numpy.pad(counts, 1)
    with_flow[2, 3, 5] = 2.0  # the sample has entries in an overflow bin of z
    assert h3.counts(flow=True).tolist() == with_flow.tolist()
    assert h3.counts(flow=False).shape == h3.values(flow=False).shape
    assert h3.counts(flow=True).shape == h3.values(flow=True).shape
