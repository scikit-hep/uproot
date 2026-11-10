# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import awkward as ak
import uproot
import skhep_testdata
import numpy
import os


def test_colon_in_path_and_name(tmp_path):
    newfile = os.path.join(tmp_path, "test_colon_in_name.root")
    with uproot.recreate(newfile) as f:
        f["one:two"] = "together"
        array = ak.Array(["one", "two", "three"])
        f.mktree("one", {"two": array})

    with uproot.open(newfile) as f:
        assert f["one:two"] == "together"
        assert f["one"]["two"].array().tolist() == ["one", "two", "three"]


def test_colon_reading_in_path():
    with uproot.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    ) as f:
        assert f["tree:evt/P3/P3.Py"].array().tolist() == numpy.arange(100).tolist()
