# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot

ak = pytest.importorskip("awkward")


def test(tmp_path):
    filename = os.path.join(str(tmp_path), "whatever.root")

    with uproot.recreate(filename) as file:
        file.mktree("tree", {"branch": ak.Array([[1, 2, 3], [4, 5, 6]])})

    with uproot.open(filename) as file:
        assert isinstance(file["tree/branch"].interpretation, uproot.AsJagged)
        assert file["tree/branch"].array().tolist() == [[1, 2, 3], [4, 5, 6]]
