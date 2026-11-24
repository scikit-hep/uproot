# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot


def test_numpy():
    with uproot.open(skhep_testdata.data_path("uproot-issue-589.root")) as file:
        array = file["events"]["MC_px"].array(library="np", entry_stop=2)
        assert array[0][7] == 30.399463653564453
        assert array[1][11] == 42.04872131347656


pytest.importorskip("awkward")


def test_awkward():
    with uproot.open(skhep_testdata.data_path("uproot-issue-589.root")) as file:
        array = file["events"]["MC_px"].array(library="ak", entry_stop=2)
        assert array[0][7] == 30.399463653564453
        assert array[1][11] == 42.04872131347656
