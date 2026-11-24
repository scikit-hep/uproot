# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

hist = pytest.importorskip("hist")


def test_no_flow(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    h = hist.Hist(hist.axis.Regular(20, 0, 20, flow=False)).fill(
        np.random.normal(10, 6, 1000)
    )

    with uproot.recreate(filename) as fout:
        fout["test"] = h

    with uproot.open(filename) as fin:
        h_read = fin["test"]

    assert np.array_equal(h.axes[0].edges, h_read.axes[0].edges())
    assert np.array_equal(h.values(flow=False), h_read.values(flow=False))


def test_yes_flow(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    h = hist.Hist(hist.axis.Regular(20, 0, 20, flow=True)).fill(
        np.random.normal(10, 6, 1000)
    )

    with uproot.recreate(filename) as fout:
        fout["test"] = h

    with uproot.open(filename) as fin:
        h_read = fin["test"]

    assert np.array_equal(h.axes[0].edges, h_read.axes[0].edges())
    assert np.array_equal(h.values(flow=False), h_read.values(flow=False))
    assert np.array_equal(h.values(flow=True), h_read.values(flow=True))
