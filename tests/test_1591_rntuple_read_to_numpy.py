# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata
import awkward as ak
import numpy as np
import uproot


def test_reading_into_numpy():
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        ak_arrays = obj.arrays(filter_name="n*")
        np_arrays = obj.arrays(library="np", filter_name="n*")
        assert isinstance(ak_arrays, ak.Array)
        assert isinstance(np_arrays, dict)
        assert all(isinstance(arr, np.ndarray) for arr in np_arrays.values())

        nmuon_ak_array = obj["nMuon"].array()
        nmuon_np_array = obj["nMuon"].array(library="np")
        assert isinstance(nmuon_ak_array, ak.Array)
        assert isinstance(nmuon_np_array, np.ndarray)

        assert nmuon_np_array[:5].tolist() == [2, 2, 1, 4, 4]
        assert ak.array_equal(nmuon_ak_array, nmuon_np_array)
        assert ak.array_equal(ak_arrays["nMuon"], np_arrays["nMuon"])
        assert ak.array_equal(ak_arrays["nMuon"], nmuon_ak_array)
        assert ak.array_equal(np_arrays["nMuon"], nmuon_np_array)
