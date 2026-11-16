# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
import pytest
import uproot
import os
import numpy as np

ROOT = pytest.importorskip("ROOT")


def test_read_free_floating_list(tmp_path):
    newfile = os.path.join(tmp_path, "test_free_stl_list.root")
    f = ROOT.TFile(newfile, "recreate")

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([0, 1, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)

    alist = ROOT.std.list("double")(a)
    blist = ROOT.std.list("int")(b)
    avector = ROOT.std.vector("double")(v)

    f.WriteObject(alist, "alist")
    f.WriteObject(avector, "avector")
    f.WriteObject(blist, "blist")

    f.Write()
    f.Close()

    with uproot.open(newfile) as fs:
        a = fs["alist"]
        b = fs["blist"]
        v = fs["avector"]

        assert a.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert b.tolist() == [0, 1, 4, 5, 6, 7, 8, 9]
        assert isinstance(a, uproot.containers.STLList)
        assert isinstance(b, uproot.containers.STLList)
        assert isinstance(v, uproot.containers.STLVector)


def test_read_list_in_tree(tmp_path):
    newfile = os.path.join(tmp_path, "test_ttree_stl_list.root")
    f = ROOT.TFile(newfile, "recreate")
    t = ROOT.TTree("mytree", "example tree")

    alist = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64)
    treelist = ROOT.std.list("double")(alist)

    avector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    treevector = ROOT.std.vector("double")(avector)

    t.Branch("avector", treevector)
    t.Branch("alist", treelist)

    for i in range(3):
        t.Fill()

    f.Write()

    with uproot.open(newfile)["mytree"] as fs:

        assert fs["alist"].array().tolist() == [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ]
        assert fs["avector"].array().tolist() == [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
