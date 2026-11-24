# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest

ROOT = pytest.importorskip("ROOT")


def test():
    h = ROOT.TProfile3D()
    assert __import__("uproot").from_pyroot(h).values().shape == (1, 1, 1)
