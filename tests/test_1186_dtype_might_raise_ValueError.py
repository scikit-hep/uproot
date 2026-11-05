# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import pytest
import numpy as np

import uproot

ak = pytest.importorskip("awkward")


def test(tmp_path):
    filename = os.path.join(tmp_path, "whatever.root")

    pt = ak.Array([[0.0, 11, 22], [], [33, 44], [55], [66, 77, 88, 99]])
    event_params = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.uint32
    )

    with uproot.recreate(filename) as f:
        f.mktree("t", {"event_params": "3 * uint32", "pt": "var * float64"})
        f["t"].extend({"event_params": event_params, "pt": pt})

    with uproot.open(filename) as f:
        assert f["t"]["event_params"].array().tolist() == event_params.tolist()
        assert f["t"]["pt"].array().tolist() == pt.tolist()
