# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_volley(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    f1 = ROOT.TFile(filename, "recreate")
    f1.Close()

    assert set(uproot.open(filename).file.streamers) == set()

    with uproot.writing.update(filename) as f2:
        f2.file._cascading.streamers.write(f2.file.sink)

    assert set(uproot.open(filename).file.streamers) == set()

    f3 = ROOT.TFile(filename, "update")
    x = ROOT.TObjString("hello")
    x.Write()
    f3.Close()

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    with uproot.writing.update(filename) as f4:
        f4.file._cascading.streamers.write(f4.file.sink)

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    f5 = ROOT.TFile(filename, "update")
    y = ROOT.TH1F("hey", "there", 100, -5, 5)
    y.Write()
    f5.Close()

    # the TList of streamers has to move to the end of the file
    with uproot.writing.update(filename) as f6:
        f6.file.update_streamers([])

    streamers = set(uproot.open(filename).file.streamers)
    assert {"TObjString", "TH1F", "TH1", "TAxis"} <= streamers
    assert uproot.open(filename)["hey"].title == "there"
    assert uproot.open(filename)["hello"] == "hello"


def test_with_mkdir(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    f1 = ROOT.TFile(filename, "recreate")
    f1.Close()

    assert set(uproot.open(filename).file.streamers) == set()

    with uproot.writing.update(filename) as f2:
        f2.mkdir("one")

    assert set(uproot.open(filename).file.streamers) == set()

    f3 = ROOT.TFile(filename, "update")
    f3.cd("one")
    x = ROOT.TObjString("hello")
    x.Write()
    f3.Close()

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    with uproot.writing.update(filename) as f4:
        f4.mkdir("two")

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    f5 = ROOT.TFile(filename, "update")
    y = ROOT.TH1F("hey", "there", 100, -5, 5)
    y.Write()
    f5.Close()

    with uproot.writing.update(filename) as f6:
        f6.mkdir("three")

    with uproot.open(filename) as f7:
        assert set(f7.keys(cycle=False)) == {
            "one",
            "one/hello",
            "two",
            "hey",
            "three",
        }
        assert {"TObjString", "TH1F"} <= set(f7.file.streamers)


def test_add_streamers_1(tmp_path):
    has_TObjString = os.path.join(tmp_path, "has_TObjString.root")

    f_TObjString = ROOT.TFile(has_TObjString, "recreate")
    x = ROOT.TObjString("hello")
    x.Write()
    f_TObjString.Close()

    streamers = [uproot.open(has_TObjString).file.streamers["TObjString"][1]]

    filename = os.path.join(tmp_path, "testy.root")

    f1 = ROOT.TFile(filename, "recreate")
    f1.Close()

    assert set(uproot.open(filename).file.streamers) == set()

    with uproot.writing.update(filename) as f2:
        f2.file.update_streamers(streamers)

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    f3 = ROOT.TFile(filename, "update")
    y = ROOT.TObjString("there")
    y.Write()
    f3.Close()

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    with uproot.writing.update(filename) as f3:
        pass

    assert set(uproot.open(filename).file.streamers) == {"TObjString"}

    f4 = ROOT.TFile(filename, "update")
    z = ROOT.TH1F("histogram", "", 100, -5, 5)
    z.Write()
    f4.Close()

    assert uproot.open(filename)["histogram"].values().tolist() == [0] * 100


def test_add_streamers_2(tmp_path):
    has_histogram = os.path.join(tmp_path, "has_histogram.root")

    f_histogram = ROOT.TFile(has_histogram, "recreate")
    x = ROOT.TH1F("histogram", "", 100, -5, 5)
    x.Write()
    f_histogram.Close()

    streamers = [
        byversion
        for byname in uproot.open(has_histogram).file.streamers.values()
        for byversion in byname.values()
    ]

    filename = os.path.join(tmp_path, "testy.root")

    f1 = ROOT.TFile(filename, "recreate")
    f1.Close()

    assert set(uproot.open(filename).file.streamers) == set()

    with uproot.writing.update(filename) as f2:
        f2.file.update_streamers(streamers)

    assert set(uproot.open(filename).file.streamers) == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
    }

    f3 = ROOT.TFile(filename, "update")
    y = ROOT.TH1F("margotsih", "", 100, -5, 5)
    f3.Close()

    assert set(uproot.open(filename).file.streamers) == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
    }

    with uproot.writing.update(filename) as f3:
        pass

    assert set(uproot.open(filename).file.streamers) == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
    }

    f3 = ROOT.TFile(filename, "update")
    z = ROOT.TObjString("there")
    z.Write()
    f3.Close()

    f4 = ROOT.TFile(filename)
    assert str(f4.Get("there")) == "there"
    f4.Close()


def test_add_streamers_3(tmp_path):
    has_TObjString = os.path.join(tmp_path, "has_TObjString.root")

    f_TObjString = ROOT.TFile(has_TObjString, "recreate")
    x = ROOT.TObjString("hello")
    x.Write()
    f_TObjString.Close()

    streamers = [uproot.open(has_TObjString).file.streamers["TObjString"][1]]

    filename = os.path.join(tmp_path, "testy.root")

    f1 = ROOT.TFile(filename, "recreate")
    y = ROOT.TH1F("histogram", "", 100, -5, 5)
    y.Write()
    f1.Close()

    assert set(uproot.open(filename).file.streamers) == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
    }

    with uproot.writing.update(filename) as f2:
        f2.file.update_streamers(streamers)

    assert set(uproot.open(filename).file.streamers) == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
        "TObjString",
    }

    f2 = ROOT.TFile(filename, "update")
    assert {z.GetName() for z in f2.GetStreamerInfoList()} == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
        "TObjString",
    }
    w = ROOT.TObjString("wow")
    w.Write()
    f2.Close()

    assert set(uproot.open(filename).file.streamers) == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
        "TObjString",
    }

    f3 = ROOT.TFile(filename, "update")
    assert {z.GetName() for z in f3.GetStreamerInfoList()} == {
        "TObject",
        "TNamed",
        "TH1F",
        "TH1",
        "TAttLine",
        "TAttFill",
        "TAttMarker",
        "TAxis",
        "TAttAxis",
        "THashList",
        "TList",
        "TSeqCollection",
        "TCollection",
        "TString",
        "TObjString",
    }
    f3.Close()
