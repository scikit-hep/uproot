# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import sys
from io import StringIO

import numpy
import pytest
import skhep_testdata

import uproot


def test_get_key():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        assert f["Events"].name == "Events"
        assert f["Events/Info"].name == "Info"
        assert f["Events/Info/evtNum"].name == "evtNum"
        assert f["Events"]["Info/evtNum"].name == "evtNum"
        assert f["Events"]["/Info/evtNum"].name == "evtNum"
        assert f["Events/evtNum"].name == "evtNum"
        assert f["Events"]["evtNum"].name == "evtNum"
        assert f["Events"]["/Info"].name == "Info"
        assert f["Events"]["/Info/"].name == "Info"
        with pytest.raises(uproot.KeyInFileError, match="not found: '/evtNum'"):
            f["Events"]["/evtNum"]


def test_basket_data():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    ) as f:
        assert f["sample/i4"].basket_key(3).fSeekKey == 35042
        assert f["sample/i4"].basket(3).data.view(">i4").tolist() == [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]


def test_compressed():
    with uproot.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        basket = f["Events/Muon.q"].basket(0)
        assert basket.data.view(">i4").tolist() == [
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
        ]
        assert basket.byte_offsets.tolist() == [0, 4, 4, 16, 28, 28, 32, 52, 52, 56, 56]


@pytest.mark.parametrize("mini", [False, True])
def test_recovery(mini):
    # flat array to recover:
    filename = skhep_testdata.data_path("uproot-issue21.root")
    with uproot.open(
        {"file:" + filename: "nllscan/mH"}, minimal_ttree_metadata=mini
    ) as branch:
        basket = branch.basket(0)
        assert basket.data.view(">f8").tolist()[:10] == [
            124.0,
            124.09089660644531,
            124.18180084228516,
            124.27269744873047,
            124.36360168457031,
            124.45449829101562,
            124.54550170898438,
            124.63639831542969,
            124.72730255126953,
            124.81819915771484,
        ]
        assert basket.byte_offsets is None
        assert branch.entry_offsets == [0, branch.num_entries]

    # jagged arrays to recover:

    # uproot-issue327.root DstTree: fTracks.fCharge
    # uproot-issue232.root fTreeV0: V0s.fV0pt MCparticles.nbodies
    # uproot-issue187.root fTreeV0: V0s.fV0pt MCparticles.nbodies
    # uproot-from-geant4.root Details: numgood, TrackedRays: Event phi
    filename = skhep_testdata.data_path("uproot-issue327.root")
    with uproot.open(
        {"file:" + filename: "DstTree/fTracks.fCharge"}, minimal_ttree_metadata=mini
    ) as branch:
        basket = branch.basket(0)
        assert basket.data.view("i1")[:10].tolist() == [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
        ]
        assert basket.byte_offsets[:10].tolist() == [
            0,
            2,
            37,
            56,
            60,
            81,
            82,
            112,
            112,
            112,
        ]
        assert branch.entry_offsets == [0, branch.num_entries]


def test_library_equality_is_safe():
    from uproot.interpretation.library import NumPy, Awkward, Pandas

    np_lib = NumPy()
    ak_lib = Awkward()
    pd_lib = Pandas()

    assert np_lib == NumPy()
    assert ak_lib == Awkward()
    assert pd_lib == Pandas()

    # Invalid comparisons must NOT raise
    assert not (np_lib == object())
    assert not (np_lib == "np")
