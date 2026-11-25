# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot._util
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_subdirs(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    with uproot.writing.recreate(filename) as root_directory:
        subdir1 = root_directory.mkdir("wowzers")
        subdir2 = root_directory.mkdir("yikes")

    assert uproot.open(filename).classnames() == {
        "wowzers;1": "TDirectory",
        "yikes;1": "TDirectory",
    }

    f = ROOT.TFile(filename, "update")
    x = ROOT.TObjString("hello")
    x.Write()
    f.cd("wowzers")
    y = ROOT.TObjString("there")
    y.Write()
    f.cd("/yikes")
    z = ROOT.TObjString("you")
    z.Write()
    f.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers;1": "TDirectory",
        "yikes;1": "TDirectory",
        "hello;1": "TObjString",
        "wowzers/there;1": "TObjString",
        "yikes/you;1": "TObjString",
    }

    g = ROOT.TFile(filename, "update")
    w = ROOT.TObjString("one_more")
    w.Write()
    g.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers;1": "TDirectory",
        "yikes;1": "TDirectory",
        "hello;1": "TObjString",
        "one_more;1": "TObjString",
        "wowzers/there;1": "TObjString",
        "yikes/you;1": "TObjString",
    }


def test_subsubdir(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    with uproot.writing.recreate(filename) as root_directory:
        subdir1 = root_directory.mkdir("wowzers")
        subdir2 = subdir1.mkdir("yikes")

    assert uproot.open(filename).classnames() == {
        "wowzers;1": "TDirectory",
        "wowzers/yikes;1": "TDirectory",
    }

    f = ROOT.TFile(filename, "update")
    x = ROOT.TObjString("hello")
    x.Write()
    f.cd("/wowzers")
    y = ROOT.TObjString("there")
    y.Write()
    f.cd("/wowzers/yikes")
    z = ROOT.TObjString("you")
    z.Write()
    f.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers;1": "TDirectory",
        "wowzers/yikes;1": "TDirectory",
        "hello;1": "TObjString",
        "wowzers/there;1": "TObjString",
        "wowzers/yikes/you;1": "TObjString",
    }

    g = ROOT.TFile(filename, "update")
    g.cd("/wowzers/yikes")
    w = ROOT.TObjString("one_more")
    w.Write()
    g.Close()

    assert uproot.open(filename).classnames() == {
        "wowzers;1": "TDirectory",
        "wowzers/yikes;1": "TDirectory",
        "hello;1": "TObjString",
        "wowzers/there;1": "TObjString",
        "wowzers/yikes/you;1": "TObjString",
        "wowzers/yikes/one_more;1": "TObjString",
    }


def test_little_datime_functions():
    assert (
        uproot._util.datetime_to_code(uproot._util.code_to_datetime(1762860281))
        == 1762860281
    )
