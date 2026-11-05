# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest
import skhep_testdata

import uproot


def test_regular_axis_negative_index():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        axis = f["hpx"].axis()
        assert axis[-1] == pytest.approx((3.92, 4.0))
        assert axis[-100] == axis[0]
        # axis[i] matches the intervals that iteration yields for every bin
        bins = list(axis)
        for i in range(-100, 100):
            assert axis[i] == tuple(bins[i])
        with pytest.raises(IndexError, match="bin index 100 out of range for 100"):
            axis[100]
        with pytest.raises(IndexError, match="bin index -101 out of range for 100"):
            axis[-101]
        with pytest.raises(IndexError, match="only integers, slices"):
            axis[99.5]


def test_regular_axis_array_index():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        axis = f["hpx"].axis()
        low, high = axis[numpy.arange(3)]
        assert low == pytest.approx([-4.0, -3.92, -3.84])
        assert high == pytest.approx([-3.92, -3.84, -3.76])
        assert axis[numpy.int8(-1)] == axis[99]


def test_regular_axis_multidimensional_index():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        axis = f["hpx"].axis()
        index = numpy.arange(6).reshape(2, 3)
        low, high = axis[index]
        assert low.shape == high.shape == (2, 3)
        assert low == pytest.approx(axis.intervals()[index, 0])
        assert high == pytest.approx(axis.intervals()[index, 1])


def test_axis_contains_its_own_bins():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        axis = f["hpx"].axis()
        assert axis[0] in axis
        assert axis[-1] in axis
        assert (100.0, 200.0) not in axis


def test_variable_axis_negative_index(tmp_path):
    edges = numpy.array([0.0, 1.0, 3.0, 7.0])
    with uproot.recreate(tmp_path / "variable.root") as f:
        f["h"] = (numpy.array([1.0, 2.0, 3.0]), edges)

    with uproot.open(tmp_path / "variable.root") as f:
        axis = f["h"].axis()
        assert axis[-1] == (3.0, 7.0)
        assert axis[-3] == axis[0] == (0.0, 1.0)
        low, high = axis[numpy.array([[0, 2], [1, 0]])]
        assert low.tolist() == [[0.0, 3.0], [1.0, 0.0]]
        assert high.tolist() == [[1.0, 7.0], [3.0, 1.0]]
        with pytest.raises(IndexError, match="bin index 3 out of range for 3 bins"):
            axis[3]
        with pytest.raises(IndexError, match="bin index -4 out of range for 3 bins"):
            axis[-4]
