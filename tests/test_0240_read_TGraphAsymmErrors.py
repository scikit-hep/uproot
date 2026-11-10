# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


@pytest.fixture
def graph():
    with uproot.open(skhep_testdata.data_path("uproot-issue-240.root")) as f:
        yield f["Expected limit 1lbb/Graph1D_y1"]


def test_interpretation(graph):
    assert graph.classname == "TGraphAsymmErrors"
    assert graph.behaviors[0] == uproot.behaviors.TGraphAsymmErrors.TGraphAsymmErrors

    assert "fX" in graph.all_members.keys()
    assert "fY" in graph.all_members.keys()
    assert "fEXlow" in graph.all_members.keys()
    assert "fEYlow" in graph.all_members.keys()
    assert "fEXhigh" in graph.all_members.keys()
    assert "fEYhigh" in graph.all_members.keys()


@pytest.mark.parametrize(
    "axis, member, first_values",
    [
        (-2, "fX", [840.7016, 837.2594]),
        (0, "fX", [840.7016, 837.2594]),
        ("x", "fX", [840.7016, 837.2594]),
        (-1, "fY", [0.0, 8.8392]),
        (1, "fY", [0.0, 8.8392]),
        ("y", "fY", [0.0, 8.8392]),
    ],
)
def test_values_single(graph, axis, member, first_values):
    values = graph.values(axis=axis)
    assert isinstance(values, numpy.ndarray)
    assert values.shape == (162,)
    assert values.tolist() == graph.member(member).tolist()
    assert values[:2].tolist() == pytest.approx(first_values)


@pytest.mark.parametrize("axis", [-2, -1, 0, 1, "x", "y"])
@pytest.mark.parametrize("which", ["low", "high", "mean", "diff"])
def test_errors_single(graph, axis, which):
    # this graph is a limit contour: all of its errors are zero
    errors = graph.errors(axis=axis, which=which)
    assert isinstance(errors, numpy.ndarray)
    assert errors.tolist() == [0.0] * 162


def test_values_double(graph):
    x, y = graph.values(axis="both")
    assert x.tolist() == graph.member("fX").tolist()
    assert y.tolist() == graph.member("fY").tolist()


@pytest.mark.parametrize("which", ["low", "high", "mean", "diff"])
def test_errors_double(graph, which):
    x, y = graph.errors(axis="both", which=which)
    assert x.tolist() == [0.0] * 162
    assert y.tolist() == [0.0] * 162


@pytest.mark.parametrize(
    "which, expected_x, expected_y",
    [
        ("low", [0.1, 0.2], [1.0, 2.0]),
        ("high", [0.3, 0.8], [4.0, 3.0]),
        ("mean", [0.2, 0.5], [2.5, 2.5]),
        ("diff", [0.2, 0.6], [3.0, 1.0]),
    ],
)
def test_errors_are_combined_from_both_sides(which, expected_x, expected_y):
    graph = uproot.as_TGraph(
        {
            "x": numpy.array([1.0, 2.0]),
            "y": numpy.array([3.0, 4.0]),
            "x_errors_low": numpy.array([0.1, 0.2]),
            "x_errors_high": numpy.array([0.3, 0.8]),
            "y_errors_low": numpy.array([1.0, 2.0]),
            "y_errors_high": numpy.array([4.0, 3.0]),
        }
    )
    assert graph.errors(which, "x") == pytest.approx(expected_x)
    assert graph.errors(which, "y") == pytest.approx(expected_y)
    x, y = graph.errors(which, "both")
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(expected_y)
