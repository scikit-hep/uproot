# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_deep_mkdir(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    with uproot.recreate(filename) as f1:
        f1.mkdir("/one/two/three")

    assert uproot.open(filename).keys() == ["one;1", "one/two;1", "one/two/three;1"]


def test_bulk_copy(tmp_path):
    source_filename = os.path.join(tmp_path, "source.root")
    dest_filename = os.path.join(tmp_path, "dest.root")

    f_histogram = ROOT.TFile(source_filename, "recreate")
    f_histogram.mkdir("subdir")
    f_histogram.cd("subdir")
    x = ROOT.TH1F("hist", "title", 100, -5, 5)
    x.Write()
    f_histogram.Close()

    with uproot.open(source_filename) as source:
        with uproot.recreate(dest_filename) as dest:
            dest.copy_from(source, filter_name="subdir/hist")

    with uproot.open(dest_filename) as dest:
        assert dest.keys() == ["subdir;1", "subdir/hist;1"]
        hist = dest["subdir/hist"]
        assert hist.name == "hist"
        assert hist.title == "title"
        assert hist.axis().low == -5

    f2 = ROOT.TFile(dest_filename, "update")
    h2 = f2.Get("subdir/hist")
    assert h2.GetName() == "hist"
    assert h2.GetTitle() == "title"
    assert h2.GetNbinsX() == 100

    y = ROOT.TObjString("hello")
    y.Write()
    f2.Close()

    assert set(uproot.open(dest_filename).keys()) == {
        "subdir;1",
        "subdir/hist;1",
        "hello;1",
    }
