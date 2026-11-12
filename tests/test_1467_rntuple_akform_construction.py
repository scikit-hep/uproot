# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def test_akform_logic():
    filepath = skhep_testdata.data_path(
        "cmsopendata2015_ttbar_19980_NANOAOD_RNTupleImporter_rntuple_v1-0-0-1.root"
    )

    with uproot.open(filepath) as file:
        obj = file["Events"]
        arrays = obj.arrays()

    # If the logic was wrong it would have crashed before getting here
    assert len(arrays.fields) == 969
    assert arrays.nJet[:5].tolist() == [8, 8, 7, 9, 7]
    assert arrays.event[:3].tolist() == [44727241, 44727242, 44727243]
    assert arrays.Jet_pt[1].tolist() == [
        81.6875,
        71.6875,
        52.15625,
        35.125,
        28.875,
        25.25,
        19.15625,
        18.125,
    ]
