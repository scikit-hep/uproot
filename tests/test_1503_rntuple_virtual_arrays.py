# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def test_rntuple_virtual_arrays_no_log():
    filename = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        eager = obj.arrays()
        virtual = obj.arrays(virtual=True, access_log=None)

    assert ak.to_packed(virtual).layout.is_equal_to(ak.to_packed(eager).layout)


def test_rntuple_virtual_arrays_with_log():
    log = []
    filename = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]
        eager = obj.arrays()
        virtual = obj.arrays(virtual=True, access_log=log)

    assert len(log) == 0
    assert ak.to_packed(virtual).layout.is_equal_to(ak.to_packed(eager).layout)
    assert len(log) == 44
    ak.materialize(virtual)
    assert len(log) == 44


def test_rntuple_virtual():
    log = []
    filename = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]["string"]
        eager = obj.array()
        virtual = obj.array(virtual=True, access_log=log)

    assert len(log) == 0
    assert ak.to_packed(virtual).layout.is_equal_to(ak.to_packed(eager).layout)
    assert len(log) == 2  # Offsets and data


def test_rntuple_virtual_arrays_nonsense_kwargs_combinations():
    path = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    with uproot.open(path) as file:
        obj = file["ntuple"]

        # virtual=True
        match = "cannot be used with 'virtual=True'"
        with pytest.raises(ValueError, match=match):
            obj.arrays(virtual=True, how="zip")

        with pytest.raises(ValueError, match=match):
            obj.arrays(virtual=True, library="numpy")

        with pytest.raises(ValueError, match=match):
            obj.arrays(virtual=True, expressions="foo")

        with pytest.raises(ValueError, match=match):
            obj.arrays(virtual=True, cut="foo")

        with pytest.raises(ValueError, match=match):
            obj.arrays(virtual=True, aliases="foo")

        # virtual=False
        match = "cannot be used with 'virtual=False'"
        with pytest.raises(ValueError, match=match):
            obj.arrays(virtual=False, access_log=[])
