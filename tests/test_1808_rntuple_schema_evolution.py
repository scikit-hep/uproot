# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import uproot


@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
def test_ntuple_add_fields_preserves_intervening_extends(tmp_path, automatic):
    path = tmp_path / "test.root"
    first = np.array([1, 2, 3], dtype=np.int32)
    second = np.array([4, 5], dtype=np.int32)

    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("n", {"x": np.int32})
        ntuple.add_fields({"y": np.int32})
        ntuple.extend({"x": first, "y": first + 10})

        if automatic:
            ntuple.extend(
                {"x": second, "y": second + 10, "z": second + 20},
                accept_new_fields=True,
            )
        else:
            ntuple.add_fields({"z": np.int32})

    with uproot.open(path) as f:
        ntuple = f["n"]
        if automatic:
            assert ntuple.num_entries == 5
            assert ak.to_list(ntuple["x"].array()) == [1, 2, 3, 4, 5]
            assert ak.to_list(ntuple["y"].array()) == [11, 12, 13, 14, 15]
            assert ak.to_list(ntuple["z"].array()) == [0, 0, 0, 24, 25]
        else:
            assert ntuple.num_entries == 3
            assert ak.to_list(ntuple["x"].array()) == [1, 2, 3]
            assert ak.to_list(ntuple["y"].array()) == [11, 12, 13]
            assert ak.to_list(ntuple["z"].array()) == [0, 0, 0]
