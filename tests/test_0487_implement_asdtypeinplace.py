# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_inplace_tbranch_array():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        N = events.num_entries

        m_array = numpy.zeros(N, dtype=numpy.float64)
        m_i = events["M"].interpretation.inplace(m_array)
        m_res = events["M"].array(interpretation=m_i, library="np")

        assert m_res.base is m_array
        assert m_res[:5].tolist() == [
            82.4626915551,
            83.6262040052,
            83.3084646667,
            82.1493728809,
            90.4691230355,
        ]

        run_array = numpy.zeros(N, dtype=numpy.int64)
        run_i = events["Run"].interpretation.inplace(run_array)
        run_res = events["Run"].array(interpretation=run_i, library="np")

        assert run_res.base is run_array
        assert run_res[:5].tolist() == [148031, 148031, 148031, 148031, 148031]


def test_inplace_ttree_arrays():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as events:
        N = events.num_entries

        m_array = numpy.zeros(N, dtype=numpy.float64)
        m_i = events["M"].interpretation.inplace(m_array)

        run_array = numpy.zeros(N, dtype=numpy.int64)
        run_i = events["Run"].interpretation.inplace(run_array)

        events.arrays(dict(M=m_i, Run=run_i), library="np")

        assert m_array[:5].tolist() == [
            82.4626915551,
            83.6262040052,
            83.3084646667,
            82.1493728809,
            90.4691230355,
        ]

        assert run_array[:5].tolist() == [148031, 148031, 148031, 148031, 148031]
