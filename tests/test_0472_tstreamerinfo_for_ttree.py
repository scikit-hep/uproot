# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import io
import os

import numpy as np
import pytest
import skhep_testdata

import uproot

ROOT = pytest.importorskip("ROOT")


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    with uproot.recreate(filename) as f:
        f.mktree("t", {"b": np.array([1, 2, 3], np.int64)})

    with uproot.open(filename) as f:
        output = io.StringIO()
        f.file.show_streamers(stream=output)
        assert output.getvalue().startswith("TString (v2)\n\nTObject (v1)\n")
        assert "\nTTree (v20): " in output.getvalue()
        assert len(f.file.streamers) == 24
        assert f["t/b"].array(library="np").tolist() == [1, 2, 3]

    f = ROOT.TFile(filename)
    t = f.Get("t")
    assert [x.b for x in t] == [1, 2, 3]
