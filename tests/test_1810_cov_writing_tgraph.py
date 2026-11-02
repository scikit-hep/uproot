# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy as np
import pytest

import uproot
from uproot.writing.interpret import _as_TGraph

X = np.array([1.0, 2.0, 3.0])
Y = np.array([2.0, 4.0, 8.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"x_errors": X}, "requires both x_errors and y_errors"),
        ({"y_errors": X}, "requires both x_errors and y_errors"),
        (
            {"x_errors": X, "y_errors": X, "y_errors_high": X},
            "symmetrical errors OR asymmetrical errors, but not both",
        ),
        (
            {"x_errors_low": X, "x_errors_high": X, "y_errors_low": X},
            "requires all of the following",
        ),
        ({"title": "a;b"}, "can't contain ';'"),
        ({"xAxisLabel": "a;b"}, "can't contain ';'"),
        ({"yAxisLabel": "a;b"}, "can't contain ';'"),
        ({"minY": "low"}, "minY has to be None or a number, not <class 'str'>"),
        ({"maxY": "high"}, "maxY has to be None or a number, not <class 'str'>"),
        (
            {"x_errors": X[:2], "y_errors": X},
            "Length of all error arrays has to be the same",
        ),
        (
            {
                "x_errors_low": X,
                "x_errors_high": X[:2],
                "y_errors_low": X,
                "y_errors_high": X,
            },
            "Length of errors all error arrays has to be the same",
        ),
    ],
)
def test_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _as_TGraph(X, Y, **kwargs)


@pytest.mark.parametrize(
    "x, y, message",
    [
        (X, Y[:2], "Arrays x and y must have the same length!"),
        (X[:0], Y[:0], "x and y arrays can't be empty"),
        (X.reshape(3, 1), Y, "x has to be 1D, but is 2D!"),
        (X, Y.reshape(3, 1), "y has to be 1D, but is 2D!"),
    ],
)
def test_invalid_arrays(x, y, message):
    with pytest.raises(ValueError, match=message):
        _as_TGraph(x, y)


def test_explicit_y_range_round_trip(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = tmp_path / "graph.root"
    with uproot.recreate(path) as file:
        file["graph"] = _as_TGraph(X, Y, minY=-1.5, maxY=20)

    with uproot.open(path) as file:
        graph = file["graph"]
        assert graph.member("fMinimum") == -1.5
        assert graph.member("fMaximum") == 20

    root_file = ROOT.TFile.Open(str(path))
    graph = root_file.Get("graph")
    assert graph.GetN() == 3
    assert graph.GetMinimum() == -1.5
    assert graph.GetMaximum() == 20
    assert [graph.GetPointY(i) for i in range(3)] == Y.tolist()
    root_file.Close()


def test_symmetric_errors_make_a_TGraphErrors():
    graph = _as_TGraph(X, Y, x_errors=X / 10, y_errors=Y / 10)
    assert isinstance(graph, uproot.models.TGraph.Model_TGraphErrors_v3)
    assert graph.classname == "TGraphErrors"
    assert graph.member("fEX").tolist() == (X / 10).tolist()
    assert graph.member("fEY").tolist() == (Y / 10).tolist()
    assert graph.member("fX").tolist() == X.tolist()
    assert graph.member("fNpoints") == 3


def test_asymmetric_errors_make_a_TGraphAsymmErrors():
    graph = _as_TGraph(
        X,
        Y,
        x_errors_low=X / 10,
        x_errors_high=X / 20,
        y_errors_low=Y / 10,
        y_errors_high=Y / 20,
    )
    assert isinstance(graph, uproot.models.TGraph.Model_TGraphAsymmErrors_v3)
    assert graph.classname == "TGraphAsymmErrors"
    assert graph.member("fEXlow").tolist() == (X / 10).tolist()
    assert graph.member("fEXhigh").tolist() == (X / 20).tolist()
    assert graph.member("fEYlow").tolist() == (Y / 10).tolist()
    assert graph.member("fEYhigh").tolist() == (Y / 20).tolist()
    assert graph.member("fY").tolist() == Y.tolist()
