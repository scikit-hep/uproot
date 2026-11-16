# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_fallback_reading():
    with uproot.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    ) as f:
        branch = f["tree:evt/P3/P3.Py"]
        assert branch.name == "P3.Py"
        assert branch.array(library="np", entry_stop=3).tolist() == [0.0, 1.0, 2.0]
        assert f.file._streamers is None
