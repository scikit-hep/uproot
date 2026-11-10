# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-750.root"))[
        "tout/Electron_eta"
    ] as branch:
        # basket 63 of this branch has no entries
        assert branch.num_baskets == 132
        assert branch.entry_offsets[63:65] == [148, 148]
        array = branch.array(library="np")
        assert array.shape == (312,)
        assert array[:3].tolist() == pytest.approx(
            [-0.65563965, -0.72460938, 0.05927277]
        )
        assert array[-3:].tolist() == pytest.approx(
            [-1.15307617, 0.95727539, -0.98120117]
        )
        assert array.sum() == pytest.approx(-44.476287841796875)
