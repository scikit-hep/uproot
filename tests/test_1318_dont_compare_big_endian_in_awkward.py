# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import pytest
import awkward as ak

import uproot


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    nMuon_pt = ak.Array([1, 2, 3])
    Muon_pt = ak.Array([[1.1], [2.2, 3.3], [4.4, 5.5, 6.6]])

    with uproot.recreate(filename) as file:
        file.mktree("tree", {"nMuon_pt": nMuon_pt, "Muon_pt": Muon_pt})

    with uproot.open(filename) as file:
        tree = file["tree"]
        assert tree["nMuon_pt"].array().tolist() == [1, 2, 3]
        assert tree["Muon_pt"].array().tolist() == Muon_pt.tolist()
