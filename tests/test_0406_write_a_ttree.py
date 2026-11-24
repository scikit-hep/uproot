# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot

ROOT = pytest.importorskip("ROOT")


def test_basic(tmp_path):
    # original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    # f1 = ROOT.TFile(original, "recreate")
    # f1.SetCompressionLevel(0)
    # t1 = ROOT.TTree("t1", "title")
    # d1 = array.array("d", [0.0])
    # t1.Branch("branch1", d1, "branch1/D")

    # t1.Write()
    # f1.Close()

    # with uproot.open(original) as fin:
    #     fin["t1"]

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("t1", {"branch1": np.float64}, "title")

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("t1")

    assert t2.GetName() == "t1"
    assert t2.GetTitle() == "title"

    assert t2.GetBranch("branch1").GetName() == "branch1"
    assert t2.GetBranch("branch1").GetTitle() == "branch1/D"

    assert t2.GetBranch("branch1").GetLeaf("branch1").GetName() == "branch1"
    assert t2.GetBranch("branch1").GetLeaf("branch1").GetTitle() == "branch1"

    assert t2.GetLeaf("branch1").GetName() == "branch1"
    assert t2.GetLeaf("branch1").GetTitle() == "branch1"

    f2.Close()


def test_rename(tmp_path):
    newfile = os.path.join(tmp_path, "newfiley_file.root")

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree(
            "treey_tree",
            {"branchy_branch": np.float64},
            "titley_title",
        )

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("treey_tree")

    assert t2.GetName() == "treey_tree"
    assert t2.GetTitle() == "titley_title"

    assert t2.GetBranch("branchy_branch").GetName() == "branchy_branch"
    assert t2.GetBranch("branchy_branch").GetTitle() == "branchy_branch/D"

    assert (
        t2.GetBranch("branchy_branch").GetLeaf("branchy_branch").GetName()
        == "branchy_branch"
    )
    assert (
        t2.GetBranch("branchy_branch").GetLeaf("branchy_branch").GetTitle()
        == "branchy_branch"
    )

    assert t2.GetLeaf("branchy_branch").GetName() == "branchy_branch"
    assert t2.GetLeaf("branchy_branch").GetTitle() == "branchy_branch"

    f2.Close()


def test_2_branches(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree(
            "t1",
            {"branch1": np.float64, "branch2": np.float64},
            "title",
        )

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("t1")

    assert t2.GetName() == "t1"
    assert t2.GetTitle() == "title"

    for branchname in ["branch1", "branch2"]:
        assert t2.GetBranch(branchname).GetName() == branchname
        assert t2.GetBranch(branchname).GetTitle() == branchname + "/D"

        assert t2.GetBranch(branchname).GetLeaf(branchname).GetName() == branchname
        assert t2.GetBranch(branchname).GetLeaf(branchname).GetTitle() == branchname

        assert t2.GetLeaf(branchname).GetName() == branchname
        assert t2.GetLeaf(branchname).GetTitle() == branchname

    # FIXME: also test the following in a case where "t1" has a 64-bit TKey
    with uproot.open(newfile) as fin:
        t1 = fin["t1"]
        branch1 = t1["branch1"]
        branch2 = t1["branch2"]
        assert branch1.member("fLeaves")[0] is t1.member("fLeaves")[0]
        assert branch2.member("fLeaves")[0] is t1.member("fLeaves")[1]

    f2.Close()


def test_100_branches(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree(
            "t1",
            {"branch" + str(i): np.float64 for i in range(100)},
            "title",
        )

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("t1")

    assert t2.GetName() == "t1"
    assert t2.GetTitle() == "title"

    for branchname in ["branch" + str(i) for i in range(100)]:
        assert t2.GetBranch(branchname).GetName() == branchname
        assert t2.GetBranch(branchname).GetTitle() == branchname + "/D"

        assert t2.GetBranch(branchname).GetLeaf(branchname).GetName() == branchname
        assert t2.GetBranch(branchname).GetLeaf(branchname).GetTitle() == branchname

        assert t2.GetLeaf(branchname).GetName() == branchname
        assert t2.GetLeaf(branchname).GetTitle() == branchname

    f2.Close()


def test_branch_types(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree(
            "t1",
            {
                "typeO": np.bool_,
                "typeB": np.int8,
                "typeb": np.uint8,
                "typeS": np.int16,
                "types": np.uint16,
                "typeI": np.int32,
                "typei": np.uint32,
                "typeL": np.int64,
                "typel": np.uint64,
                "typeF": np.float32,
                "typeD": np.float64,
            },
            "title",
        )

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("t1")

    assert t2.GetName() == "t1"
    assert t2.GetTitle() == "title"

    for name, size, isunsigned in [
        ("typeO", 1, False),
        ("typeB", 1, False),
        ("typeb", 1, True),
        ("typeS", 2, False),
        ("types", 2, True),
        ("typeI", 4, False),
        ("typei", 4, True),
        ("typeL", 8, False),
        ("typel", 8, True),
        ("typeF", 4, False),
        ("typeD", 8, False),
    ]:
        assert t2.GetBranch(name).GetName() == name
        assert t2.GetBranch(name).GetTitle() == name + "/" + name[-1]

        assert t2.GetBranch(name).GetLeaf(name).GetName() == name
        assert t2.GetBranch(name).GetLeaf(name).GetTitle() == name

        assert t2.GetLeaf(name).GetName() == name
        assert t2.GetLeaf(name).GetTitle() == name
        assert t2.GetLeaf(name).GetLenType() == size
        assert t2.GetLeaf(name).IsUnsigned() == isunsigned

    f2.Close()


def test_basket(tmp_path):
    # original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    # f1 = ROOT.TFile(original, "recreate")
    # f1.SetCompressionLevel(0)
    # t1 = ROOT.TTree("t1", "title")
    # d1 = array.array("i", [0])
    # t1.Branch("branch1", d1, "branch1/I")

    # d1[0] = 5
    # t1.Fill()
    # d1[0] = 4
    # t1.Fill()
    # d1[0] = 3
    # t1.Fill()
    # d1[0] = 2
    # t1.Fill()
    # d1[0] = 1
    # t1.Fill()
    # d1[0] = 5
    # t1.Fill()

    # t1.Write()
    # f1.Close()

    # with uproot.open(original) as fin:
    #     t1 = fin["t1"]

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("t2", {"branch1": np.int32}, "title")
        tree.extend({"branch1": np.array([5, 4, 3, 2, 1, 5], ">i4")})

    with uproot.open(newfile) as fin2:
        t2 = fin2["t2"]
        assert t2["branch1"].array(library="np").tolist() == [5, 4, 3, 2, 1, 5]

    f2 = ROOT.TFile(newfile)
    t2 = f2.Get("t2")
    assert [x.branch1 for x in t2] == [5, 4, 3, 2, 1, 5]
    f2.Close()


def test_baskets_branches(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    b1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b2 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("t", {"b1": np.int32, "b2": np.float64}, "title")
        tree.extend({"b1": b1, "b2": b2})
        tree.extend({"b1": b1, "b2": b2})

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("t")
    assert t1.GetEntries() == len(b1) + len(b1)
    assert [x.b1 for x in t1] == b1 + b1
    assert [x.b2 for x in t1] == b2 + b2

    with uproot.open(newfile + ":t") as t2:
        assert t2.num_entries == len(b1) + len(b1)
        assert t2["b1"].array(library="np").tolist() == b1 + b1
        assert t2["b2"].array(library="np").tolist() == b2 + b2

    f1.Close()


def test_baskets_beyond_capacity(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    b1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b2 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("t", {"b1": np.int32, "b2": np.float64}, "title")

        assert tree._cascading._basket_capacity == 10

        for _ in range(5):
            fout["t"].extend({"b1": b1, "b2": b2})

        assert tree._cascading._basket_capacity == 10

        for _ in range(10):
            fout["t"].extend({"b1": b1, "b2": b2})

        assert tree._cascading._basket_capacity == 100

        for _ in range(90):
            fout["t"].extend({"b1": b1, "b2": b2})

        assert tree._cascading._basket_capacity == 1000

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("t")
    assert t1.GetEntries() == len(b1) * 105
    assert [x.b1 for x in t1] == b1 * 105
    assert [x.b2 for x in t1] == b2 * 105

    with uproot.open(newfile) as fin:
        assert fin.keys() == ["t;1"]  # same cycle number
        t2 = fin["t"]
        assert t2.num_entries == len(b1) * 105
        assert t2["b1"].array(library="np").tolist() == b1 * 105
        assert t2["b2"].array(library="np").tolist() == b2 * 105

    f1.Close()


def test_writable_vs_readable_tree(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(newfile, "recreate")
    f1.SetCompressionLevel(0)
    t1 = ROOT.TTree("t1", "title")
    d1 = array.array("i", [0])
    t1.Branch("branch1", d1, "branch1/I")

    d1[0] = 5
    t1.Fill()
    d1[0] = 4
    t1.Fill()
    d1[0] = 3
    t1.Fill()
    d1[0] = 2
    t1.Fill()
    d1[0] = 1
    t1.Fill()
    d1[0] = 5
    t1.Fill()

    t1.Write()
    f1.Close()

    b1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b2 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    with uproot.update(newfile) as fin:
        fin.mktree("t2", {"b1": np.int32, "b2": np.float64}, "title")

        for _ in range(5):
            fin["t2"].extend({"b1": b1, "b2": b2})

    f1 = ROOT.TFile(newfile)
    t2root = f1.Get("t2")
    assert t2root.GetEntries() == len(b1) * 5
    assert [x.b1 for x in t2root] == b1 * 5
    assert [x.b2 for x in t2root] == b2 * 5

    t1root = f1.Get("t1")
    assert [x.branch1 for x in t1root] == [5, 4, 3, 2, 1, 5]

    with uproot.open(newfile) as finagin:
        t2uproot = finagin["t2"]
        assert t2uproot.num_entries == len(b1) * 5
        assert t2uproot["b1"].array(library="np").tolist() == b1 * 5
        assert t2uproot["b2"].array(library="np").tolist() == b2 * 5

        assert finagin["t1/branch1"].array(library="np").tolist() == [5, 4, 3, 2, 1, 5]

    f1.Close()


def test_interface(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    branch1 = np.arange(10)
    branch2 = 1.1 * np.arange(10)

    entries, edges = np.histogram(branch2)

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"branch1": branch1, "branch2": branch2})
        fout["tree"].extend({"branch1": branch1, "branch2": branch2})
        fout["hist"] = (entries, edges)

    with uproot.open(newfile) as fin:
        assert fin["tree/branch1"].array(library="np").tolist() == branch1.tolist() * 2
        assert fin["tree/branch2"].array(library="np").tolist() == branch2.tolist() * 2
        assert fin["hist"].to_numpy()[0].tolist() == entries.tolist()
        assert fin["hist"].to_numpy()[1].tolist() == edges.tolist()
