# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import os
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def test_flat_arrays_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63400:
        pytest.skip("ROOT version does not support RNTuple v1.0.0.0")

    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        data = ak.Array({"one": [1, 2, 3], "two": [1.1, 2.2, 3.3]})
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)

    if ROOT.gROOT.GetVersionInt() < 63600:
        RT = ROOT.Experimental.RNTupleReader.Open("ntuple", filepath)
    else:
        RT = ROOT.RNTupleReader.Open("ntuple", filepath)
    RT.PrintInfo()
    RT.Show(0)
    RT.Show(2)
    out = capfd.readouterr().out
    assert "* N-Tuple : ntuple" in out
    assert "* Entries : 3" in out
    assert "* Field 1   : one (std::int64_t)" in out
    assert "* Field 2   : two (double)" in out
    assert '  "one": 1,' in out
    assert '  "two": 1.1' in out
    assert '  "one": 3' in out
    assert '  "two": 3.3' in out
