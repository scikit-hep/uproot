# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


@pytest.mark.parametrize(
    "filename, tree, branch, message",
    [
        (
            "uproot-issue510b.root",
            "EDepSimEvents",
            "Event/Trajectories.Points",
            "memberwise serialization of AsArray",
        ),
        (
            "uproot-issue403.root",
            "Model",
            "Model.collimatorInfo",
            r"memberwise serialization of AsVector\(BDSOutputROOTEventCollimatorInfo\)",
        ),
        (
            "uproot-issue475.root",
            "Event/Elec/ElecEvent",
            "fElecChannels",
            r"memberwise serialization of AsVector\(nEXO::ElecChannel\)",
        ),
    ],
)
def test_memberwise_not_implemented(filename, tree, branch, message):
    with uproot.open(skhep_testdata.data_path(filename))[tree] as t:
        with pytest.raises(NotImplementedError, match=message):
            t[branch].array(library="np")
