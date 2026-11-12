# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

boost_histogram = pytest.importorskip("boost_histogram")


@pytest.fixture
def curves():
    with uproot.open(skhep_testdata.data_path("uproot-issue-350.root")) as f:
        yield f["bhist"], f["berr"]


@pytest.mark.parametrize("to_sequence", [list, tuple])
def test_to_boost_accepts_sequence_edges(curves, to_sequence):
    roocurve, roocurve_err = curves
    edges = numpy.arange(0.0, 51.0)

    expected = roocurve.to_boost(edges)
    result = roocurve.to_boost(to_sequence(edges.tolist()))
    assert result.axes[0].edges == pytest.approx(edges)
    assert result.values() == pytest.approx(expected.values())

    expected = roocurve.to_boost(edges, error_curve=roocurve_err)
    result = roocurve.to_boost(to_sequence(edges.tolist()), error_curve=roocurve_err)
    assert result.values() == pytest.approx(expected.values())
    assert result.variances() == pytest.approx(expected.variances())


def test_to_boost_integer_edges(curves):
    roocurve, _ = curves
    result = roocurve.to_boost([0, 10, 20])
    expected = roocurve.interpolate(numpy.array([5.0, 15.0]))
    assert result.values() == pytest.approx(expected)
