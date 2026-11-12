# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import os
import numpy as np


def test(tmp_path):
    newfile = os.path.join(tmp_path, "example.root")

    with uproot.recreate(newfile) as f:
        f.mktree("t", {"x": np.array(["A", "B"]), "y": np.array([1, 2])})
        f["t"].extend({"x": np.array(["A", "B"]), "y": np.array([1, 2])})

    with uproot.open(newfile) as f:
        assert f["t"]["x"].array().tolist() == ["A", "B", "A", "B"]
        assert f["t"]["y"].array().tolist() == [1, 2, 1, 2]
