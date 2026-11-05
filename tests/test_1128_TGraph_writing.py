# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import pytest
import numpy as np

import uproot

pd = pytest.importorskip("pandas")
ROOT = pytest.importorskip("ROOT")


def test_saving_TGraph_to_file(tmp_path):
    newfile = os.path.join(tmp_path, "test_file.root")
    x = [i for i in range(10)]
    y = [i * 10 for i in range(10)]
    df = pd.DataFrame({"x": x, "y": y})

    with uproot.recreate(newfile) as f:
        f["myTGraph"] = uproot.as_TGraph(df)

    with uproot.open(newfile) as f:
        tgraph = f["myTGraph"]

        x_new = tgraph.values("x")
        y_new = tgraph.values("y")

        assert x_new.tolist() == x
        assert y_new.tolist() == y


def test_opening_TGraph_with_root(tmp_path):
    newfile = os.path.join(tmp_path, "test_file.root")
    x = [i for i in range(10)]
    y = [i * 10 for i in range(10)]
    df = pd.DataFrame({"x": x, "y": y})
    tGraphName = "myTGraph"
    title = "My TGraph"
    xLabel = "xLabel"
    yLabel = "yLabel"

    with uproot.recreate(newfile) as f:
        f[tGraphName] = uproot.as_TGraph(
            df, title=title, xAxisLabel=xLabel, yAxisLabel=yLabel
        )

    with ROOT.TFile.Open(newfile) as file:
        tgraph = file.Get(tGraphName)

        assert tgraph.GetTitle() == f"{title};{xLabel};{yLabel}"
        assert tgraph.GetXaxis().GetTitle() == xLabel
        assert tgraph.GetYaxis().GetTitle() == yLabel

        assert tgraph.GetN() == len(x)
        assert [tgraph.GetPointX(i) for i in range(len(x))] == x
        assert [tgraph.GetPointY(i) for i in range(len(y))] == y
