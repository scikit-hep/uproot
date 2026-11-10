# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
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


def test_concatenate_and_iterate_accept_already_open_rntuple():
    """
    An already-open RNTuple (a ``HasFields``, just as an open TTree is a
    ``HasBranches``) is accepted in ``files`` instead of being mistaken for an
    iterable of file paths, and reading it must not close the user's file.
    """
    path = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")

    with uproot.open(path) as f:
        ntuple = f["ntuple"]
        expected = ntuple.arrays()
        assert len(expected) > 1

        assert ak.array_equal(uproot.concatenate(ntuple), expected)
        assert ak.array_equal(uproot.concatenate([ntuple]), expected)

        chunks = list(uproot.iterate(ntuple, step_size=2))
        assert len(chunks) > 1
        assert ak.array_equal(ak.concatenate(chunks), expected)

        # the RNTuple was read in place: the user's file is still open
        assert not f.file.source.closed


def test_backend_and_interpreter_are_refused_for_ttree():
    """
    ``backend``/``interpreter`` are only implemented for RNTuples: a TTree must
    raise instead of quietly returning CPU arrays.
    """
    path = skhep_testdata.data_path("uproot-HZZ.root") + ":events"

    with pytest.raises(NotImplementedError, match="only implemented for RNTuples"):
        uproot.concatenate(path, backend="cuda", interpreter="gpu")
    with pytest.raises(NotImplementedError, match="only implemented for RNTuples"):
        next(uproot.iterate(path, backend="cuda", interpreter="gpu"))

    # either keyword alone is enough to ask for a non-CPU read
    with pytest.raises(NotImplementedError, match="only implemented for RNTuples"):
        uproot.concatenate(path, backend="cuda")
    with pytest.raises(NotImplementedError, match="only implemented for RNTuples"):
        next(uproot.iterate(path, interpreter="gpu"))

    # a mixed TTree/RNTuple read must not read the TTree on the CPU silently
    rntuple_path = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    with pytest.raises(NotImplementedError, match="only implemented for RNTuples"):
        uproot.concatenate([path, rntuple_path], backend="cuda", interpreter="gpu")
    with pytest.raises(NotImplementedError, match="only implemented for RNTuples"):
        next(uproot.iterate([path, rntuple_path], backend="cuda", interpreter="gpu"))
