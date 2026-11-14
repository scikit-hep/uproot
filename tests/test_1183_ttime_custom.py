# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata

ROOT = pytest.importorskip("ROOT")


def test_ttime_custom():
    f = ROOT.TFile.Open(skhep_testdata.data_path("uproot-issue-861.root"))
    t1 = f.Get("RealTime_0").AsString()
    t2 = f.Get("LiveTime_0").AsString()
    f.Close()
    with uproot.open(skhep_testdata.data_path("uproot-issue-861.root")) as f:
        assert str(f["RealTime_0"].members["fMilliSec"]) == t1
        assert str(f["LiveTime_0"].members["fMilliSec"]) == t2
