# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import skhep_testdata

import uproot


def test_atomic():
    filename = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("atomic_int")

        assert a.atomic_int.tolist() == [1, 2, 3]


def test_bitset():
    filename = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("bitset")

        assert len(a.bitset) == 3
        assert len(a.bitset[0]) == 42
        assert a.bitset[0].tolist()[:6] == [0, 1, 0, 1, 0, 1]
        assert all(a.bitset[0][6:] == 0)
        assert a.bitset[1].tolist()[:16] == [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
        assert all(a.bitset[1][16:] == 0)
        assert a.bitset[2].tolist()[:16] == [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ]
        assert all(a.bitset[2][16:] == 0)


def test_empty_struct():
    filename = skhep_testdata.data_path(
        "test_emptystruct_invalidvar_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("empty_struct")

        assert a.empty_struct.tolist() == [(), (), ()]


def test_invalid_variant():
    filename = skhep_testdata.data_path(
        "test_emptystruct_invalidvar_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj["variant"].array()

        assert a.tolist() == [1, None, {"i": 2}]
