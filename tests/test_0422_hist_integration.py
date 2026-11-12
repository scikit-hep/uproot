# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
hist = pytest.importorskip("hist")


def test_regular_1d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout["h1"] = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Weight()
            .fill([-2, 3, 3, 1, 99], weight=[1, 1, 5, 5, 3])
        )

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.values(flow=True).tolist() == pytest.approx(
            [0, 0, 0, 0, 1, 0, 0, 5, 0, 6, 0, 3]
        )
        assert h1.axis().member("fName") == "xaxis"
        assert h1.axis().member("fTitle") == "wee"
        assert h1.axis().member("fXmin") == -5
        assert h1.axis().member("fXmax") == 5
        assert len(h1.axis().member("fXbins")) == 0

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(9) == 6
    assert h2.GetBinContent(11) == 3
    assert h2.GetXaxis().GetName() == "xaxis"
    assert h2.GetXaxis().GetTitle() == "wee"
    assert h2.GetBinLowEdge(1) == pytest.approx(-5)
    assert h2.GetBinWidth(1) == pytest.approx(1)
    f.Close()


def test_variable_1d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout["h1"] = (
            hist.Hist.new.Var([-5, -3, 0, 1, 2, 10], name="wow", label="wee")
            .Weight()
            .fill([-2, 3, 3, 1, 99], weight=[1, 1, 5, 5, 3])
        )

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.values(flow=True).tolist() == pytest.approx([0, 0, 1, 0, 5, 6, 3])
        assert h1.axis().member("fName") == "xaxis"
        assert h1.axis().member("fTitle") == "wee"
        assert h1.axis().member("fXmin") == -5
        assert h1.axis().member("fXmax") == 10
        assert list(h1.axis().member("fXbins")) == pytest.approx([-5, -3, 0, 1, 2, 10])

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(5) == 6
    assert h2.GetBinContent(6) == 3
    assert h2.GetXaxis().GetName() == "xaxis"
    assert h2.GetXaxis().GetTitle() == "wee"
    assert h2.GetBinLowEdge(1) == pytest.approx(-5)
    assert h2.GetBinWidth(1) == pytest.approx(2)
    f.Close()


def test_regular_2d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        tmp = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Reg(8, 2, 10)
            .Weight()
            .fill([-2, 3, 3, 1, 99], [9, 9, 9, 4, 4], weight=[1, 1, 5, 5, 3])
        )
        asarray = tmp.values(flow=True)
        assert asarray[9, 8] == 6
        assert asarray[8, 9] == 0
        fout["h1"] = tmp

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.axis(0).member("fName") == "xaxis"
        assert h1.axis(0).member("fTitle") == "wee"
        assert h1.axis(0).member("fXmin") == -5
        assert h1.axis(0).member("fXmax") == 5
        assert h1.axis(1).member("fXmin") == 2
        assert h1.axis(1).member("fXmax") == 10
        assert np.allclose(asarray, h1.values(flow=True))
        assert np.allclose(asarray, h1.to_hist().values(flow=True))

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(9, 8) == 6
    assert h2.GetBinContent(8, 9) == 0
    f.Close()


def test_regular_3d(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        tmp = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Reg(8, 2, 10)
            .Reg(2, -2, 2)
            .Weight()
            .fill(
                [-2, 3, 3, 1, 99],
                [9, 9, 9, 4, 4],
                [1, 1, 1, -1, -1],
                weight=[1, 1, 5, 5, 3],
            )
        )
        asarray = tmp.values(flow=True)
        assert asarray[9, 8, 2] == 6
        assert asarray[8, 9, 2] == 0
        assert asarray[9, 8, 1] == 0
        fout["h1"] = tmp

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.axis(0).member("fName") == "xaxis"
        assert h1.axis(0).member("fTitle") == "wee"
        assert h1.axis(0).member("fXmin") == -5
        assert h1.axis(0).member("fXmax") == 5
        assert h1.axis(1).member("fXmin") == 2
        assert h1.axis(1).member("fXmax") == 10
        assert h1.axis(2).member("fXmin") == -2
        assert h1.axis(2).member("fXmax") == 2
        assert np.allclose(asarray, h1.values(flow=True))
        assert np.allclose(asarray, h1.to_hist().values(flow=True))

    f = ROOT.TFile(newfile)
    h2 = f.Get("h1")
    assert h2.GetEntries() == 15
    assert h2.GetBinContent(9, 8, 2) == 6
    assert h2.GetBinContent(8, 9, 2) == 0
    assert h2.GetBinContent(9, 8, 1) == 0
    f.Close()


def test_issue_0659(tmp_path):
    # https://github.com/scikit-hep/uproot/issues/659
    newfile = os.path.join(tmp_path, "newfile.root")

    cat_axis = hist.axis.IntCategory([10, 11, 12], label="Category")
    reg_axis = hist.axis.Regular(100, 0, 100, label="Random")
    reg_axis_z = hist.axis.Regular(50, 20, 30, label="RandomZ")
    h = hist.Hist(cat_axis)
    h2 = hist.Hist(cat_axis, reg_axis)
    h3 = hist.Hist(cat_axis, reg_axis, reg_axis_z)
    h.fill(np.random.randint(1, 4, 1000))
    h2.fill(np.random.randint(1, 4, 1000), np.random.normal(20, 5, 1000))
    h3.fill(
        np.random.randint(1, 4, 1000),
        np.random.normal(20, 5, 1000),
        np.random.normal(25, 2, 1000),
    )

    with uproot.recreate(newfile) as fout:
        fout["h"] = h
        fout["h2"] = h2
        fout["h3"] = h3

    with uproot.open(newfile) as fin:
        h_opened = fin["h"]
        assert h_opened.values(flow=False).shape == (3,)
        assert h_opened.values(flow=True).shape == (5,)
        assert h_opened.axis(0).edges().tolist() == [0.0, 1.0, 2.0, 3.0]

        h2_opened = fin["h2"]
        assert h2_opened.values(flow=False).shape == (3, 100)
        assert h2_opened.values(flow=True).shape == (5, 102)
        assert h2_opened.axis(0).edges().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert h2_opened.axis(1).edges().tolist() == list(map(float, range(101)))

        h3_opened = fin["h3"]
        assert h3_opened.values(flow=False).shape == (3, 100, 50)
        assert h3_opened.values(flow=True).shape == (5, 102, 52)
        assert h3_opened.axis(0).edges().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert h3_opened.axis(1).edges().tolist() == list(map(float, range(101)))
        assert h3_opened.axis(2).edges().tolist() == list(np.linspace(20, 30, 51))

        h_opened.to_hist()
        h2_opened.to_hist()
        h3_opened.to_hist()

    f = ROOT.TFile(newfile)
    h_opened2 = f.Get("h")
    h2_opened2 = f.Get("h2")
    h3_opened2 = f.Get("h3")

    assert h_opened2.GetBinContent(0) == 0.0
    assert h2_opened2.GetBinContent(0) == 0.0
    assert h3_opened2.GetBinContent(0) == 0.0

    f.Close()


def test_issue_722(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h = ROOT.TH1D("h", "h", 10, 0.0, 1.0)
    h.FillRandom("gaus", 10000)

    assert len(h.GetSumw2()) == 0

    fout = ROOT.TFile(newfile, "RECREATE")
    h.Write()
    fout.Close()

    # open with uproot
    with uproot.open(newfile) as fin:
        h1 = fin["h"]

    assert len(h1.axes) == 1
    assert h1.axis(0).edges().tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert len(h1.member("fSumw2")) == 0

    # convert to hist
    h2 = h1.to_hist()
    assert str(h2.storage_type) == "<class 'boost_histogram.storage.Double'>"

    # write and read again
    with uproot.recreate(newfile) as fout2:
        fout2["h"] = h2

    with uproot.open(newfile) as fin2:
        h3 = fin2["h"]

    assert len(h3.member("fSumw2")) == 0


def test_hist_weights_from_root(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h = ROOT.TH1D("h", "h", 20, 0.0, 5.0)
    for _ in range(1000):
        # fill with random values and random weights
        h.Fill(5.0 * np.random.random(), np.random.random())

    assert len(h.GetSumw2()) == 22  # 20 bins + 2, should not be 0 since we have weights

    fout = ROOT.TFile(newfile, "RECREATE")
    h.Write()
    fout.Close()

    with uproot.open(newfile) as fin:
        h1 = fin["h"]

    assert len(h1.member("fSumw2")) == 22

    h2 = h1.to_hist()
    assert str(h2.storage_type) == "<class 'boost_histogram.storage.Weight'>"

    # write and read again
    with uproot.recreate(newfile) as fout2:
        fout2["h"] = h2

    with uproot.open(newfile) as fin2:
        h3 = fin2["h"]

    assert len(h3.member("fSumw2")) == 22


def test_hist_weights_labels_from_root(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h_weights = ROOT.TH1D(
        "h_weights", "histogram with weights with labels", 12, 0.0, 5.0
    )
    h_noweights = ROOT.TH1D(
        "h_noweights", "histogram without weights with labels", 12, 0.0, 5.0
    )
    h_noweights_nolabels = ROOT.TH1D(
        "h_noweights_nolabels", "histogram without weights", 12, 0.0, 5.0
    )
    for _ in range(1000):
        # fill with random values and random weights
        h_weights.Fill(5.0 * np.random.random(), np.random.random())
        h_noweights.Fill(5.0 * np.random.random())
        h_noweights_nolabels.Fill(5.0 * np.random.random())

    assert h_weights.GetNbinsX() == h_noweights.GetNbinsX()
    for i in range(h_weights.GetNbinsX()):
        h_weights.GetXaxis().SetBinLabel(i + 1, f"label_{i}")
        h_noweights.GetXaxis().SetBinLabel(i + 1, f"label_{i}")

    assert len(h_weights.GetSumw2()) == 14  # 12 bins + 2
    assert len(h_noweights.GetSumw2()) == 0
    assert len(h_noweights_nolabels.GetSumw2()) == 0
    assert h_weights.GetXaxis().GetLabels().GetSize() == 12
    assert h_noweights.GetXaxis().GetLabels().GetSize() == 12

    fout = ROOT.TFile(newfile, "RECREATE")
    h_weights.Write()
    h_noweights.Write()
    h_noweights_nolabels.Write()
    fout.Close()

    with uproot.open(newfile) as fin:
        h_weights1 = fin["h_weights"]
        h_noweights1 = fin["h_noweights"]
        h_noweights_nolabels1 = fin["h_noweights_nolabels"]

    assert len(h_weights1.member("fSumw2")) == 14
    assert len(h_noweights1.member("fSumw2")) == 0
    assert len(h_noweights_nolabels1.member("fSumw2")) == 0

    h_weights2 = h_weights1.to_hist()
    h_noweights2 = h_noweights1.to_hist()
    h_noweights_nolabels2 = h_noweights_nolabels1.to_hist()
    assert str(h_weights2.storage_type) == "<class 'boost_histogram.storage.Weight'>"
    assert str(h_noweights2.storage_type) == "<class 'boost_histogram.storage.Double'>"
    assert (
        str(h_noweights_nolabels2.storage_type)
        == "<class 'boost_histogram.storage.Double'>"
    )


def test_hist_weights_2D(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h_2D_noweights_nolabels = ROOT.TH2D(
        "h_2D_noweights_nolabels", "2D", 20, 0.0, 5.0, 10, -10.0, 10.0
    )
    h_2D_weights_nolabels = ROOT.TH2D(
        "h_2D_weights_nolabels", "2D", 20, 0.0, 5.0, 10, -10.0, 10.0
    )
    h_2D_noweights_labels = ROOT.TH2D(
        "h_2D_noweights_labels", "2D", 20, 0.0, 5.0, 10, -10.0, 10.0
    )
    h_2D_weights_labels = ROOT.TH2D(
        "h_2D_weights_labels", "2D", 20, 0.0, 5.0, 10, -10.0, 10.0
    )

    for _ in range(1000):
        # fill with random values and random weights
        h_2D_noweights_nolabels.Fill(
            5.0 * np.random.random(), 10.0 * np.random.random()
        )
        h_2D_weights_nolabels.Fill(
            5.0 * np.random.random(), 10.0 * np.random.random(), np.random.random()
        )
        h_2D_noweights_labels.Fill(5.0 * np.random.random(), 10.0 * np.random.random())
        h_2D_weights_labels.Fill(
            5.0 * np.random.random(), 10.0 * np.random.random(), np.random.random()
        )

    assert (
        h_2D_noweights_nolabels.GetNbinsX()
        == h_2D_weights_nolabels.GetNbinsX()
        == h_2D_noweights_labels.GetNbinsX()
        == h_2D_weights_labels.GetNbinsX()
        == 20
    )
    assert (
        h_2D_noweights_nolabels.GetNbinsY()
        == h_2D_weights_nolabels.GetNbinsY()
        == h_2D_noweights_labels.GetNbinsY()
        == h_2D_weights_labels.GetNbinsY()
        == 10
    )
    for i in range(h_2D_weights_labels.GetNbinsX()):
        h_2D_weights_labels.GetXaxis().SetBinLabel(i + 1, f"label_{i}")
        h_2D_noweights_labels.GetXaxis().SetBinLabel(i + 1, f"label_{i}")
    for i in range(h_2D_noweights_labels.GetNbinsY()):
        # add y labels to this one
        h_2D_noweights_labels.GetYaxis().SetBinLabel(i + 1, f"label_{i}")

    assert (
        len(h_2D_weights_nolabels.GetSumw2())
        == len(h_2D_weights_labels.GetSumw2())
        == 264
    )
    assert (
        len(h_2D_noweights_nolabels.GetSumw2())
        == len(h_2D_noweights_labels.GetSumw2())
        == 0
    )

    assert (
        h_2D_weights_labels.GetXaxis().GetLabels().GetSize()
        == h_2D_noweights_labels.GetXaxis().GetLabels().GetSize()
        == 20
    )
    assert h_2D_noweights_labels.GetYaxis().GetLabels().GetSize() == 10

    fout = ROOT.TFile(newfile, "RECREATE")
    h_2D_noweights_nolabels.Write()
    h_2D_weights_nolabels.Write()
    h_2D_noweights_labels.Write()
    h_2D_weights_labels.Write()
    fout.Close()

    with uproot.open(newfile) as fin:
        h_2D_noweights_nolabels = fin["h_2D_noweights_nolabels"]
        h_2D_weights_nolabels = fin["h_2D_weights_nolabels"]
        h_2D_noweights_labels = fin["h_2D_noweights_labels"]
        h_2D_weights_labels = fin["h_2D_weights_labels"]

    h_2D_noweights_nolabels.to_hist()
    h_2D_weights_nolabels.to_hist()
    h_2D_noweights_labels.to_hist()
    h_2D_weights_labels.to_hist()


def test_hist_weights_3D(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    h_3D_noweights_nolabels = ROOT.TH3D(
        "h_3D_noweights_nolabels", "3D", 20, 0.0, 5.0, 10, -10.0, 10.0, 5, -5.0, 5.0
    )
    h_3D_weights_nolabels = ROOT.TH3D(
        "h_3D_weights_nolabels", "3D", 20, 0.0, 5.0, 10, -10.0, 10.0, 5, -5.0, 5.0
    )
    h_3D_noweights_labels = ROOT.TH3D(
        "h_3D_noweights_labels", "3D", 20, 0.0, 5.0, 10, -10.0, 10.0, 5, -5.0, 5.0
    )
    h_3D_weights_labels = ROOT.TH3D(
        "h_3D_weights_labels", "3D", 20, 0.0, 5.0, 10, -10.0, 10.0, 5, -5.0, 5.0
    )

    for _ in range(1000):
        # fill with random values and random weights
        h_3D_noweights_nolabels.Fill(
            5.0 * np.random.random(), 10.0 * np.random.random(), 2 * np.random.random()
        )
        h_3D_weights_nolabels.Fill(
            5.0 * np.random.random(),
            10.0 * np.random.random(),
            2 * np.random.random(),
            np.random.random(),
        )
        h_3D_noweights_labels.Fill(
            5.0 * np.random.random(), 10.0 * np.random.random(), 2 * np.random.random()
        )
        h_3D_weights_labels.Fill(
            5.0 * np.random.random(),
            10.0 * np.random.random(),
            2 * np.random.random(),
            np.random.random(),
        )

    assert (
        h_3D_noweights_nolabels.GetNbinsX()
        == h_3D_weights_nolabels.GetNbinsX()
        == h_3D_noweights_labels.GetNbinsX()
        == h_3D_weights_labels.GetNbinsX()
        == 20
    )
    assert (
        h_3D_noweights_nolabels.GetNbinsY()
        == h_3D_weights_nolabels.GetNbinsY()
        == h_3D_noweights_labels.GetNbinsY()
        == h_3D_weights_labels.GetNbinsY()
        == 10
    )
    assert (
        h_3D_noweights_nolabels.GetNbinsZ()
        == h_3D_weights_nolabels.GetNbinsZ()
        == h_3D_noweights_labels.GetNbinsZ()
        == h_3D_weights_labels.GetNbinsZ()
        == 5
    )
    for i in range(h_3D_noweights_labels.GetNbinsX()):
        h_3D_weights_labels.GetXaxis().SetBinLabel(i + 1, f"label_{i}")
    for i in range(h_3D_noweights_labels.GetNbinsZ()):
        # add z labels to this one
        h_3D_noweights_labels.GetZaxis().SetBinLabel(i + 1, f"label_{i}")

    assert (
        len(h_3D_weights_nolabels.GetSumw2())
        == len(h_3D_weights_labels.GetSumw2())
        == 1848
    )
    assert (
        len(h_3D_noweights_nolabels.GetSumw2())
        == len(h_3D_noweights_labels.GetSumw2())
        == 0
    )

    assert h_3D_weights_labels.GetXaxis().GetLabels().GetSize() == 20
    assert h_3D_noweights_labels.GetZaxis().GetLabels().GetSize() == 5

    fout = ROOT.TFile(newfile, "RECREATE")
    h_3D_noweights_nolabels.Write()
    h_3D_weights_nolabels.Write()
    h_3D_noweights_labels.Write()
    h_3D_weights_labels.Write()
    fout.Close()

    with uproot.open(newfile) as fin:
        h_3D_noweights_nolabels = fin["h_3D_noweights_nolabels"]
        h_3D_weights_nolabels = fin["h_3D_weights_nolabels"]
        h_3D_noweights_labels = fin["h_3D_noweights_labels"]
        h_3D_weights_labels = fin["h_3D_weights_labels"]

    h_3D_noweights_nolabels.to_hist()
    h_3D_weights_nolabels.to_hist()
    h_3D_noweights_labels.to_hist()
    h_3D_weights_labels.to_hist()
