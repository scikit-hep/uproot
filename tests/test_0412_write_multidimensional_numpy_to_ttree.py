# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")

WRONG_SHAPE = "must fill branches with a consistent shape"


def test_2dim(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        tree = fout.mktree("tree", {"branch": np.dtype((np.float64, (3,)))})
        tree.extend({"branch": np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])})
        with pytest.raises(ValueError, match=WRONG_SHAPE):
            tree.extend({"branch": np.array([7.7, 8.8, 9.9])})
        with pytest.raises(ValueError, match=WRONG_SHAPE):
            tree.extend({"branch": np.array([[7.7], [8.8], [9.9]])})

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert [np.asarray(x.branch).tolist() for x in t1] == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
    ]

    with uproot.open(newfile) as fin:
        assert fin["tree/branch"].array().tolist() == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    f1.Close()


@pytest.mark.parametrize("wrap", [np.array, list], ids=["numpy", "lists"])
def test_2dim_interface(tmp_path, wrap):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree("tree", {"branch": wrap([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])})
        fout["tree"].extend(
            {"branch": wrap([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])}
        )
        with pytest.raises(ValueError, match=WRONG_SHAPE):
            fout["tree"].extend({"branch": wrap([7.7, 8.8, 9.9])})
        with pytest.raises(ValueError, match=WRONG_SHAPE):
            fout["tree"].extend({"branch": wrap([[7.7], [8.8], [9.9]])})

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert [np.asarray(x.branch).tolist() for x in t1] == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    with uproot.open(newfile) as fin:
        assert fin["tree/branch"].array().tolist() == [
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]

    f1.Close()


def test_structured_array(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            np.array(
                [(1, 1.1), (2, 2.2), (3, 3.3)], [("x", np.int32), ("y", np.float64)]
            ),
        )
        fout["tree"].extend(
            np.array(
                [(4, 4.4), (5, 5.5), (6, 6.6)], [("x", np.int32), ("y", np.float64)]
            )
        )

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert t1.GetBranch("x").GetName() == "x"
    assert t1.GetBranch("y").GetName() == "y"
    assert [np.asarray(x.x).tolist() for x in t1] == [1, 2, 3, 4, 5, 6]
    assert [np.asarray(x.y).tolist() for x in t1] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    with uproot.open(newfile) as fin:
        assert fin["tree/x"].name == "x"
        assert fin["tree/y"].name == "y"
        assert fin["tree/x"].typename == "int32_t"
        assert fin["tree/y"].typename == "double"
        assert fin["tree/x"].array().tolist() == [1, 2, 3, 4, 5, 6]
        assert fin["tree/y"].array().tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    f1.Close()


def test_pandas(tmp_path):
    pandas = pytest.importorskip("pandas")

    newfile = os.path.join(tmp_path, "newfile.root")

    df1 = pandas.DataFrame({"x": [1, 2, 3], "y": [1.1, 2.2, 3.3]})
    df2 = pandas.DataFrame({"x": [4, 5, 6], "y": [4.4, 5.5, 6.6]})

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree("tree", df1)
        fout["tree"].extend(df2)

    f1 = ROOT.TFile(newfile)
    t1 = f1.Get("tree")

    assert t1.GetBranch("index").GetName() == "index"
    assert t1.GetBranch("x").GetName() == "x"
    assert t1.GetBranch("y").GetName() == "y"
    assert [np.asarray(x.x).tolist() for x in t1] == [1, 2, 3, 4, 5, 6]
    assert [np.asarray(x.y).tolist() for x in t1] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    with uproot.open(newfile) as fin:
        assert fin["tree/index"].name == "index"
        assert fin["tree/x"].name == "x"
        assert fin["tree/y"].name == "y"
        assert fin["tree/index"].typename.startswith("int")
        assert fin["tree/x"].typename.startswith("int")
        assert fin["tree/y"].typename == "double"
        assert fin["tree/x"].array().tolist() == [1, 2, 3, 4, 5, 6]
        assert fin["tree/y"].array().tolist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]

    f1.Close()


def test_histogram_interface(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as fin:
        h1d, h2d = fin["hpx"], fin["hpxpy"]
        h1d_entries_1, h1d_xedges_1 = h1d.to_numpy()
        h2d_entries_1, h2d_xedges_1, h2d_yedges_1 = h2d.to_numpy(dd=False)
        h2d_dd_entries_1, (h2d_dd_xedges_1, h2d_dd_yedges_1) = h2d.to_numpy(dd=True)

        with uproot.recreate(newfile) as fout:
            fout["h1d"] = h1d.to_numpy()
            fout["h2d"] = h2d.to_numpy(dd=False)
            fout["h2d_dd"] = h2d.to_numpy(dd=True)
            h1d_entries_2, h1d_xedges_2 = fout["h1d"].to_numpy()
            h2d_entries_2, h2d_xedges_2, h2d_yedges_2 = fout["h2d"].to_numpy(dd=False)
            h2d_dd_entries_2, (h2d_dd_xedges_2, h2d_dd_yedges_2) = fout[
                "h2d_dd"
            ].to_numpy(dd=True)

    with uproot.open(newfile) as finagin:
        assert np.array_equal(h1d_entries_1, h1d_entries_2)
        assert np.array_equal(h1d_xedges_1, h1d_xedges_2)

        assert np.array_equal(h2d_entries_1, h2d_entries_2)
        assert np.array_equal(h2d_xedges_1, h2d_xedges_2)
        assert np.array_equal(h2d_yedges_1, h2d_yedges_2)

        assert np.array_equal(h2d_dd_entries_1, h2d_dd_entries_2)
        assert np.array_equal(h2d_dd_xedges_1, h2d_dd_xedges_2)
        assert np.array_equal(h2d_dd_yedges_1, h2d_dd_yedges_2)


def test_ex_nihilo_TH1(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h1 = uproot.writing.identify.to_TH1x(
        fName="h1",
        fTitle="title",
        data=np.array([0, 2, 5, 0], np.float64),
        fEntries=7,
        fTsumw=1,
        fTsumw2=1,
        fTsumwx=1,
        fTsumwx2=1,
        fSumw2=np.array([0, 2, 5, 0], np.float64),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=2,
            fXmin=-3.14,
            fXmax=2.71,
        ),
    )

    with uproot.recreate(newfile) as fout:
        fout["out"] = h1
        fout["there"] = h1.to_numpy()

    f3 = ROOT.TFile(newfile)
    for name in "out", "there":
        h3 = f3.Get(name)
        assert h3.GetEntries() == pytest.approx(7)
        assert h3.GetBinLowEdge(1) == pytest.approx(-3.14)
        assert h3.GetBinWidth(1) == pytest.approx((2.71 - -3.14) / 2)
        assert h3.GetBinContent(0) == pytest.approx(0)
        assert h3.GetBinContent(1) == pytest.approx(2)
        assert h3.GetBinContent(2) == pytest.approx(5)
        assert h3.GetBinContent(3) == pytest.approx(0)
        assert h3.GetBinError(0) == pytest.approx(0)
        assert h3.GetBinError(1) == pytest.approx(np.sqrt(2))
        assert h3.GetBinError(2) == pytest.approx(np.sqrt(5))
        assert h3.GetBinError(3) == pytest.approx(0)

    f3.Close()


def test_ex_nihilo_TH2(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h1 = uproot.writing.identify.to_TH2x(
        fName="h1",
        fTitle="title",
        data=np.array(
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], np.float64
        ),
        fEntries=7,
        fTsumw=1,
        fTsumw2=1,
        fTsumwx=1,
        fTsumwx2=1,
        fTsumwy=-1,
        fTsumwy2=1,
        fTsumwxy=-1,
        fSumw2=np.array(
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], np.float64
        ),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=2,
            fXmin=-3.14,
            fXmax=2.71,
        ),
        fYaxis=uproot.writing.identify.to_TAxis(
            fName="yaxis",
            fTitle="",
            fNbins=3,
            fXmin=-5.0,
            fXmax=10.0,
        ),
    )

    with uproot.recreate(newfile) as fout:
        fout["out"] = h1
        fout["there"] = h1.to_numpy()

    f3 = ROOT.TFile(newfile)
    for name in "out", "there":
        h3 = f3.Get(name)
        assert h3.GetEntries() == 7
        assert h3.GetNbinsX() == 2
        assert h3.GetNbinsY() == 3
        assert h3.GetXaxis().GetBinLowEdge(1) == pytest.approx(-3.14)
        assert h3.GetXaxis().GetBinUpEdge(2) == pytest.approx(2.71)
        assert h3.GetYaxis().GetBinLowEdge(1) == pytest.approx(-5)
        assert h3.GetYaxis().GetBinUpEdge(3) == pytest.approx(10)
        assert [[h3.GetBinContent(i, j) for j in range(5)] for i in range(4)] == [
            pytest.approx([0, 0, 0, 0, 0]),
            pytest.approx([0, 0, 0, 2, 0]),
            pytest.approx([0, 5, 0, 0, 0]),
            pytest.approx([0, 0, 0, 0, 0]),
        ]

    f3.Close()


def test_ex_nihilo_TH3(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h1 = uproot.writing.identify.to_TH3x(
        fName="h1",
        fTitle="title",
        data=np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.float64,
        ),
        fEntries=7,
        fTsumw=1,
        fTsumw2=1,
        fTsumwx=1,
        fTsumwx2=1,
        fTsumwy=1,
        fTsumwy2=1,
        fTsumwxy=1,
        fTsumwz=1,
        fTsumwz2=1,
        fTsumwxz=1,
        fTsumwyz=1,
        fSumw2=np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.float64,
        ),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=2,
            fXmin=-3.14,
            fXmax=2.71,
        ),
        fYaxis=uproot.writing.identify.to_TAxis(
            fName="yaxis",
            fTitle="",
            fNbins=3,
            fXmin=-5.0,
            fXmax=10.0,
        ),
        fZaxis=uproot.writing.identify.to_TAxis(
            fName="zaxis",
            fTitle="",
            fNbins=1,
            fXmin=100.0,
            fXmax=200.0,
        ),
    )

    with uproot.recreate(newfile) as fout:
        fout["out"] = h1
        fout["there"] = h1.to_numpy()

    f3 = ROOT.TFile(newfile)
    for name in "out", "there":
        h3 = f3.Get(name)
        assert h3.GetEntries() == 7
        assert h3.GetNbinsX() == 2
        assert h3.GetNbinsY() == 3
        assert h3.GetNbinsZ() == 1
        assert h3.GetXaxis().GetBinLowEdge(1) == pytest.approx(-3.14)
        assert h3.GetXaxis().GetBinUpEdge(2) == pytest.approx(2.71)
        assert h3.GetYaxis().GetBinLowEdge(1) == pytest.approx(-5)
        assert h3.GetYaxis().GetBinUpEdge(3) == pytest.approx(10)
        assert h3.GetZaxis().GetBinLowEdge(1) == pytest.approx(100)
        assert h3.GetZaxis().GetBinUpEdge(1) == pytest.approx(200)
        approx = pytest.approx
        assert [
            [[h3.GetBinContent(i, j, k) for k in range(3)] for j in range(5)]
            for i in range(4)
        ] == [
            [[0, 0, 0], approx([0, 0, 0]), [0, 0, 0], approx([0, 0, 0]), [0, 0, 0]],
            [[0, 0, 0], approx([0, 0, 0]), [0, 0, 0], approx([0, 2, 0]), [0, 0, 0]],
            [[0, 0, 0], approx([0, 5, 0]), [0, 0, 0], approx([0, 0, 0]), [0, 0, 0]],
            [[0, 0, 0], approx([0, 0, 0]), [0, 0, 0], approx([0, 0, 0]), [0, 0, 0]],
        ]

    f3.Close()


def test_fromroot_TH1(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TH1D("h1", "title", 2, 3.0, 5.0)  # centers: 3.5 4.5
    h1.Sumw2()
    h1.Fill(3.5)
    h1.Fill(3.5)
    h1.Fill(4.5)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2
            fout["there"] = h2.to_numpy()

    with uproot.open(newfile) as finagin:
        assert np.array_equal(finagin["out"].member("fSumw2"), [0, 2, 1, 0])
        assert np.array_equal(finagin["there"].member("fSumw2"), [])
        assert {
            k: v for k, v in finagin["out"].all_members.items() if k.startswith("fTs")
        } == {
            k: v for k, v in finagin["there"].all_members.items() if k.startswith("fTs")
        }

    f1.Close()


def test_fromroot_TH2(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TH2D(
        "h1", "title", 2, 3.0, 5.0, 3, 10, 13
    )  # centers: 3.5 4.5; 10.5 11.5 12.5
    h1.Sumw2()
    h1.Fill(3.5, 10.5)
    h1.Fill(3.5, 10.5)
    h1.Fill(4.5, 11.5)
    h1.Fill(4.5, 12.5)

    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2
            fout["there"] = h2.to_numpy()

    with uproot.open(newfile) as finagin:
        assert np.array_equal(
            finagin["out"].member("fSumw2"),
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
            + [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert np.array_equal(finagin["there"].member("fSumw2"), [])
        assert {
            k: v for k, v in finagin["out"].all_members.items() if k.startswith("fTs")
        } == {
            k: v for k, v in finagin["there"].all_members.items() if k.startswith("fTs")
        }

    f1.Close()


def test_fromroot_TH3(tmp_path):
    original = os.path.join(tmp_path, "original.root")
    newfile = os.path.join(tmp_path, "newfile.root")

    f1 = ROOT.TFile(original, "recreate")
    h1 = ROOT.TH3D(
        "h1", "title", 2, 3.0, 5.0, 3, 10, 13, 4, 100, 104
    )  # centers: 3.5 4.5; 10.5 11.5 12.5; 100.5, 101.5, 102.5, 103.5
    h1.Sumw2()
    h1.Fill(3.5, 10.5, 100.5)
    h1.Fill(3.5, 10.5, 100.5)
    h1.Fill(3.5, 11.5, 102.5)
    h1.Fill(3.5, 12.5, 101.5)
    h1.Write()
    f1.Close()

    with uproot.open(original) as fin:
        h2 = fin["h1"]

        with uproot.recreate(newfile) as fout:
            fout["out"] = h2
            fout["there"] = h2.to_numpy()

    with uproot.open(newfile) as finagin:
        assert np.array_equal(
            finagin["out"].member("fSumw2"),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert np.array_equal(finagin["there"].member("fSumw2"), [])
        assert {
            k: v for k, v in finagin["out"].all_members.items() if k.startswith("fTs")
        } == {
            k: v for k, v in finagin["there"].all_members.items() if k.startswith("fTs")
        }

    f1.Close()
