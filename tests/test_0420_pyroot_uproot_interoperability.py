# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot

ROOT = pytest.importorskip("ROOT")


def test_from_pyroot():
    pyroot_hist = ROOT.TH1F("hist", "title", 10, -5, 5)
    uproot_hist = uproot.from_pyroot(pyroot_hist)
    assert isinstance(uproot_hist, uproot.Model)
    assert isinstance(uproot_hist, uproot.behaviors.TH1.TH1)
    assert isinstance(uproot_hist.values(), np.ndarray)
    assert uproot_hist.values().tolist() == [0.0] * 10
    assert isinstance(uproot_hist.axis().edges(), np.ndarray)
    assert uproot_hist.axis().edges().tolist() == np.linspace(-5, 5, 11).tolist()

    pyroot_vec = ROOT.TLorentzVector(1, 2, 3, 4)
    uproot_vec = uproot.from_pyroot(pyroot_vec)
    assert isinstance(uproot_vec, uproot.Model)
    assert uproot_vec.member("fP").member("fX") == 1
    assert uproot_vec.member("fP").member("fY") == 2
    assert uproot_vec.member("fP").member("fZ") == 3
    assert uproot_vec.member("fE") == 4

    # This part just tests if the buffer relocation works
    # which is not always triggered by TH1Fs
    pyroot_h1 = ROOT.TH1F("h1", "Histogram", 100, 0, 1)
    pyroot_h2 = ROOT.TH1D("h2", "Histogram", 100, 0, 1)
    pyroot_h3 = ROOT.TH2F("h3", "Histogram", 100, 0, 1, 100, 0, 1)
    pyroot_h4 = ROOT.TH2D("h4", "Histogram", 100, 0, 1, 100, 0, 1)
    pyroot_h5 = ROOT.TH3F("h5", "Histogram", 100, 0, 1, 100, 0, 1, 100, 0, 1)
    pyroot_h6 = ROOT.TH3D("h6", "Histogram", 100, 0, 1, 100, 0, 1, 100, 0, 1)
    h1 = uproot.from_pyroot(pyroot_h1)
    h2 = uproot.from_pyroot(pyroot_h2)
    h3 = uproot.from_pyroot(pyroot_h3)
    h4 = uproot.from_pyroot(pyroot_h4)
    h5 = uproot.from_pyroot(pyroot_h5)
    h6 = uproot.from_pyroot(pyroot_h6)
    for hist, classname, shape in [
        (h1, "TH1F", (100,)),
        (h2, "TH1D", (100,)),
        (h3, "TH2F", (100, 100)),
        (h4, "TH2D", (100, 100)),
        (h5, "TH3F", (100, 100, 100)),
        (h6, "TH3D", (100, 100, 100)),
    ]:
        assert hist.classname == classname
        assert hist.values().shape == shape
        assert hist.values().sum() == 0
        assert hist.axis(0).edges()[[0, -1]].tolist() == [0.0, 1.0]


def test_write_pyroot_TObjString(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout["something"] = ROOT.TObjString("wowie")

    with uproot.open(newfile) as fin:
        assert fin["something"] == "wowie"

    f = ROOT.TFile(newfile)
    assert str(f.Get("something")) == "wowie"
    f.Close()


def test_write_pyroot_TLorentzVector(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout["something"] = ROOT.TLorentzVector(1, 2, 3, 4)

    classes = dict(uproot.classes)
    with uproot.open(newfile, custom_classes=classes) as fin:
        uproot_vec = fin["something"]
        assert uproot_vec.member("fP").member("fX") == 1
        assert uproot_vec.member("fP").member("fY") == 2
        assert uproot_vec.member("fP").member("fZ") == 3
        assert uproot_vec.member("fE") == 4

    f = ROOT.TFile(newfile)
    pyroot_vec = f.Get("something")
    assert pyroot_vec.X() == 1
    assert pyroot_vec.Y() == 2
    assert pyroot_vec.Z() == 3
    assert pyroot_vec.T() == 4
    f.Close()


def test_write_pyroot_TH1F(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout["something"] = ROOT.TH1F("something", "wicked this way comes", 10, -5, 5)

    with uproot.open(newfile) as fin:
        uproot_hist = fin["something"]
        assert uproot_hist.values().tolist() == [0.0] * 10
        assert uproot_hist.axis().edges().tolist() == np.linspace(-5, 5, 11).tolist()

    f = ROOT.TFile(newfile)
    pyroot_hist = f.Get("something")
    assert [pyroot_hist.GetBinContent(i) for i in range(1, 11)] == [0.0] * 10
    assert pyroot_hist.GetBinLowEdge(1) == pytest.approx(-5)
    assert pyroot_hist.GetBinWidth(1) == pytest.approx(1)
    f.Close()


def test_convert_to_pyroot():
    uproot_tobjstring = uproot.to_writable("hello")
    uproot_histogram = uproot.to_writable(
        (np.array([3, 2, 1.0]), np.array([10, 15, 20, 25.0]))
    )

    pyroot_tobjstring = uproot_tobjstring.to_pyroot()
    pyroot_histogram = uproot_histogram.to_pyroot()

    assert str(pyroot_tobjstring) == "hello"
    assert pyroot_histogram.GetBinContent(1) == 3
    assert pyroot_histogram.GetBinContent(2) == 2
    assert pyroot_histogram.GetBinContent(3) == 1
    assert pyroot_histogram.GetBinLowEdge(1) == pytest.approx(10)
    assert pyroot_histogram.GetBinWidth(1) == pytest.approx(5)
