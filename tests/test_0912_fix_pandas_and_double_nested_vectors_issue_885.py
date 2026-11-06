# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot
import pytest
import skhep_testdata, os
import numpy as np
import awkward as ak

ROOT = pytest.importorskip("ROOT")


def test_pandas_and_double_nested_vectors_issue_885(tmp_path):
    filename = os.path.join(
        tmp_path, "uproot_test_pandas_and_double_nested_vectors.root"
    )
    f = ROOT.TFile(filename, "recreate")
    t = ROOT.TTree("mytree", "example tree")

    vec1 = ROOT.std.vector("double")()
    vec2 = ROOT.std.vector("double")()
    vec_vec = ROOT.std.vector(ROOT.std.vector("double"))()

    for i in range(3):
        vec1.push_back(i)
    for i in range(5):
        vec2.push_back(i)

    vec_vec.push_back(vec1)
    vec_vec.push_back(vec2)

    a = np.array([1, 2, 3, 4], dtype=np.uint32)
    avec = ROOT.std.vector("unsigned int")(a)

    b = np.array([[[0, 1, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.uint32)
    bvec = ROOT.std.vector("unsigned int")(b)

    t.Branch("2Dvector", vec_vec)
    t.Branch("1Dvector", avec)
    t.Branch("othervector", bvec)

    nentries = 25
    for i in range(nentries):
        t.Fill()

    f.Write()

    with uproot.open(filename)["mytree"] as fs:
        u = fs.arrays(["2Dvector", "1Dvector", "othervector"], library="pd")
        assert isinstance(u["2Dvector"][0], list)
        assert isinstance(u["1Dvector"][0], list)
        assert isinstance(u["othervector"][0], list)
        assert u["2Dvector"][0] == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0, 4.0]]
        assert u["1Dvector"][0] == [1, 2, 3, 4]
        assert u["othervector"][0] == [0, 1, 3, 4, 5, 6, 7, 8, 9]

        branch = fs["2Dvector"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = False
        u2 = branch.array(interp, library="pd")
        assert isinstance(u2[0], list)
