# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata

import uproot
import awkward as ak


def test_concatenate_iterate_find_unique_RNTuple():
    """
    Test ``uproot.concatenate`` and ``uproot.iterate`` can automatically find
    the unique RNTuple in the given ROOT file.
    """
    path = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")

    with uproot.open(path) as f:
        arr_ref = f["ntuple"].arrays()

    arr_concat = uproot.concatenate(path)
    assert ak.array_equal(arr_ref, arr_concat)

    arr_iterate = ak.concatenate([arr for arr in uproot.iterate(path)])
    assert ak.array_equal(arr_ref, arr_iterate)
