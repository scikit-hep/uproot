# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as file:
        tree = file["Events"]
        assert tree["Muon"].array(library="np").tolist() == [
            1,
            0,
            3,
            3,
            0,
            1,
            5,
            0,
            1,
            0,
        ]
