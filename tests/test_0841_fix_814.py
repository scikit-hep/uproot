# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import numpy
import pytest
import uproot

ROOT = pytest.importorskip("ROOT")


def test_check_file_after_snapshot(tmp_path):
    filename = os.path.join(tmp_path, "tfile_with_tvector3_1.root")
    tfile = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("tree", "tree")
    tvector3 = ROOT.TVector3()
    tree.Branch("tvector3", tvector3)
    for x in range(10):
        tvector3.SetX(x)
        tree.Fill()
    tree.AutoSave()
    tfile.Write()
    array = uproot.open(filename + ":tree").arrays()
    assert array["tvector3"].fields == ["fX", "fY", "fZ"]
    assert array["tvector3"].fX.tolist() == list(range(10))
    assert array["tvector3"].fY.tolist() == [0.0] * 10

    tfile = ROOT.TFile(filename, "READ")
    tree = tfile.Get("tree")
    rdf = ROOT.RDataFrame(tree)
    branchlist = ROOT.std.vector(ROOT.std.string)()
    branchlist.push_back("tvector3")
    filename2 = os.path.join(tmp_path, "tfile_with_tvector3_2.root")
    rdf.Snapshot("tree", filename2, branchlist)
    array = uproot.open(filename2 + ":tree").arrays()
    assert array["tvector3"].fields == ["fX", "fY", "fZ"]
    assert array["tvector3"].fX.tolist() == list(range(10))
    assert array["tvector3"].fZ.tolist() == [0.0] * 10
