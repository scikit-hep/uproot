# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os
import pytest
import uproot
import awkward as ak
import numpy as np


def test_writing_ak_arrays_of_type_unknown(tmp_path):
    filename = os.path.join(tmp_path, "uproot_test_empty_type_unknown.root")
    ak_array = ak.Array([[], [], []])
    ak_array = ak.values_astype(ak_array, np.float64)

    tree = {"branch": ak_array}

    with uproot.recreate(filename) as file:
        file.mktree("test", tree)

    with uproot.open(filename) as file:
        assert file["test"].num_entries == 3
        assert file["test"].keys() == ["nbranch", "branch"]
        assert file["test/nbranch"].array().tolist() == [0, 0, 0]
        assert file["test/branch"].typename == "double[]"
        assert file["test/branch"].array().tolist() == [[], [], []]
