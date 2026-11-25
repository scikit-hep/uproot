# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_get_subdir(tmp_path):
    empty_filename = os.path.join(tmp_path, "empty.root")
    nonempty_filename = os.path.join(tmp_path, "nonempty.root")

    empty = ROOT.TFile(empty_filename, "recreate")
    empty.mkdir("subdir")
    empty.Close()

    nonempty = ROOT.TFile(nonempty_filename, "recreate")
    nonempty.mkdir("subdir")
    nonempty.cd("subdir")
    x = ROOT.TObjString("hello")
    x.Write()
    nonempty.Close()

    with uproot.update(empty_filename) as f1:
        f1["subdir"].mkdir("another")

    with uproot.update(nonempty_filename) as f2:
        f2["subdir"].mkdir("another")

    assert uproot.open(empty_filename).keys() == ["subdir;1", "subdir/another;1"]
    assert uproot.open(nonempty_filename).keys() == [
        "subdir;1",
        "subdir/hello;1",
        "subdir/another;1",
    ]

    f3 = ROOT.TFile(empty_filename, "update")
    f3.cd("subdir/another")
    y = ROOT.TObjString("there")
    y.Write()
    f3.Close()

    f4 = ROOT.TFile(nonempty_filename, "update")
    f4.cd("subdir/another")
    z = ROOT.TObjString("you")
    z.Write()
    f4.Close()

    assert uproot.open(empty_filename).keys() == [
        "subdir;1",
        "subdir/another;1",
        "subdir/another/there;1",
    ]
    assert uproot.open(nonempty_filename).keys() == [
        "subdir;1",
        "subdir/hello;1",
        "subdir/another;1",
        "subdir/another/you;1",
    ]


def test_get_string(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    f1 = ROOT.TFile(filename, "recreate")
    x = ROOT.TObjString("hello")
    x.Write()
    f1.Close()

    with uproot.update(filename) as f2:
        assert str(f2["hello"]) == "hello"


def test_get_histogram(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    f1 = ROOT.TFile(filename, "recreate")
    x = ROOT.TH1F("name", "title", 100, -5, 5)
    x.Write()
    f1.Close()

    with uproot.update(filename) as f2:
        h = f2["name"]
        assert h.name == "name"
        assert h.title == "title"


def test_get_nested(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    f1 = ROOT.TFile(filename, "recreate")
    one = f1.mkdir("one")
    one.cd()
    two = one.mkdir("two")
    two.cd()
    x = ROOT.TObjString("hello")
    x.Write()
    f1.Close()

    with uproot.update(filename) as f2:
        assert str(f2["one/two/hello"]) == "hello"
        assert f2.keys() == ["one;1", "one/two;1", "one/two/hello;1"]
        assert f2.classnames() == {
            "one;1": "TDirectory",
            "one/two;1": "TDirectory",
            "one/two/hello;1": "TObjString",
        }
