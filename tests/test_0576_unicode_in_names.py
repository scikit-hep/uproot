# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

ROOT = pytest.importorskip("ROOT")


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    with uproot.recreate(filename) as file:
        file["\u03c0"] = np.histogram(np.random.normal(0, 1, 10))
        assert file["\u03c0"].name == "\u03c0"

    file2 = ROOT.TFile(filename)
    histogram = file2.Get("\u03c0")
    assert histogram.GetName() == "\u03c0"
