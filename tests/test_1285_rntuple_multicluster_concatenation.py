# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata
import numpy as np

import uproot


def test_schema_extension():
    filename = skhep_testdata.data_path("test_index_multicluster_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        arrays = obj.arrays()
        int_vec_array = arrays["int_vector"]

        assert int_vec_array[:200].tolist() == [
            [i, i + j] for j in range(2) for i in range(100)
        ]
