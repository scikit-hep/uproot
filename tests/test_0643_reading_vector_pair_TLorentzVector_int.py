# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_numpy():
    with uproot.open(skhep_testdata.data_path("uproot-issue-643.root"))[
        "ntuple0/objects"
    ] as t:
        array = t["pf"].array(entry_stop=2, library="np")
        assert array[1][1].member("first").member("fE") == 326.0029230449897
        assert array[1][1].member("second") == 22


awkward = pytest.importorskip("awkward")


def test_awkward():
    with uproot.open(skhep_testdata.data_path("uproot-issue-643.root"))[
        "ntuple0/objects"
    ] as t:
        array = t["pf"].array(entry_stop=2, library="ak")
        assert array[1, 1, "first", "fE"] == 326.0029230449897
        assert array[1, 1, "second"] == 22
