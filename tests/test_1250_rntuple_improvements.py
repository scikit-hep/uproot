# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_field_class():
    filename = skhep_testdata.data_path("test_nested_structs_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        my_struct = obj["my_struct"]
        assert len(my_struct) == 2
        assert my_struct is f["ntuple/my_struct"]
        assert my_struct is f["ntuple"]["my_struct"]

        sub_struct = my_struct["sub_struct"]
        assert len(sub_struct) == 2
        assert sub_struct is f["ntuple/my_struct/sub_struct"]
        assert sub_struct is f["ntuple"]["my_struct"]["sub_struct"]

        sub_sub_struct = sub_struct["sub_sub_struct"]
        assert len(sub_sub_struct) == 2
        assert sub_sub_struct is f["ntuple/my_struct/sub_struct/sub_sub_struct"]
        assert (
            sub_sub_struct is f["ntuple"]["my_struct"]["sub_struct"]["sub_sub_struct"]
        )

        v = sub_sub_struct["v"]
        assert len(v) == 0
        assert v.path == "my_struct.sub_struct.sub_sub_struct.v"


def test_array_methods():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        nMuon_array = obj["nMuon"].array()
        Muon_pt_array = obj["Muon_pt"].array()
        assert nMuon_array.tolist() == [len(l) for l in Muon_pt_array]

        nMuon_arrays = obj["nMuon"].arrays()
        assert len(nMuon_arrays.fields) == 1
        assert len(nMuon_arrays) == 1000
        assert nMuon_arrays["nMuon"].tolist() == nMuon_array.tolist()


def test_iterate():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        expected_pt = [10.763696670532227, 15.736522674560547]
        expected_charge = [-1, -1]

        for step_size, lengths in [(100, [100] * 10), ("10 kB", [158] * 6 + [52])]:
            chunks = list(obj.iterate(step_size=step_size))
            assert [len(chunk) for chunk in chunks] == lengths
            assert chunks[0]["Muon_pt"][0].tolist() == expected_pt
            assert chunks[0]["Muon_charge"][0].tolist() == expected_charge

        Muon_pt = obj["Muon_pt"]
        for step_size, lengths in [(100, [100] * 10), ("5 kB", [286] * 3 + [142])]:
            chunks = list(Muon_pt.iterate(step_size=step_size))
            assert [len(chunk) for chunk in chunks] == lengths
            assert chunks[0]["Muon_pt"][0].tolist() == expected_pt
