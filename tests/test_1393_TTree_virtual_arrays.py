# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest

ak = pytest.importorskip("awkward")

import uproot
import skhep_testdata


def test_ttree_virtual_arrays_no_log():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path) as file:
        tree = file["events"]
        eager = tree.arrays()
        virtual = tree.arrays(virtual=True, access_log=None)
        assert ak.materialize(virtual).layout.is_equal_to(eager.layout)


def test_ttree_virtual_arrays_with_log():
    log = []
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path) as file:
        tree = file["events"]
        eager = tree.arrays()
        virtual = tree.arrays(virtual=True, access_log=log)

        assert len(log) == 0
        assert ak.materialize(virtual).layout.is_equal_to(eager.layout)
        assert len(log) == 21

        assert set(tree.keys()) == set({a.branch for a in log})


def test_ttree_virtual_arrays_single_branch():
    log = []
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path) as file:
        branch = file["events"]["Run"]
        eager = branch.arrays()
        virtual = branch.arrays(virtual=True, access_log=log)

        assert len(log) == 0
        assert ak.materialize(virtual).layout.is_equal_to(eager.layout)
        assert len(log) == 1

        assert {"Run"} == set({a.branch for a in log})


def test_ttree_virtual_arrays_nonsense_kwargs_combinations():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path) as file:
        tree = file["events"]

        # virtual=True
        match = "cannot be used with 'virtual=True'"
        with pytest.raises(ValueError, match=match):
            tree.arrays(virtual=True, how="zip")

        with pytest.raises(ValueError, match=match):
            tree.arrays(virtual=True, library="numpy")

        with pytest.raises(ValueError, match=match):
            tree.arrays(virtual=True, expressions="foo")

        with pytest.raises(ValueError, match=match):
            tree.arrays(virtual=True, cut="foo")

        with pytest.raises(ValueError, match=match):
            tree.arrays(virtual=True, aliases="foo")

        # virtual=False
        match = "cannot be used with 'virtual=False'"
        with pytest.raises(ValueError, match=match):
            tree.arrays(virtual=False, access_log=[])
