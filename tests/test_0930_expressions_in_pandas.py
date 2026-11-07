# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest
import uproot


def test_expressions_in_pandas(tmp_path):
    pandas = pytest.importorskip("pandas")
    filename = os.path.join(tmp_path, "uproot_test_pandas_expressions.root")
    # create tmp file
    with uproot.recreate(filename) as file:
        file.mktree("tree", {"b1": [1, 5, 9], "b2": [3, 6, 11]})

    with uproot.open(filename) as file:
        tree = file["tree"]

        # checking different options
        # arbitrary expressions work, like log
        df = tree.arrays(["log(b1)"], library="pd")
        assert df.columns.tolist() == ["log(b1)"]
        assert df["log(b1)"].tolist() == pytest.approx(
            [0.0, 1.6094379124341003, 2.1972245773362196]
        )

        # works with np
        expression = "where(b1 < b2, b1, b2)"
        assert tree.arrays([expression], library="np")[expression].tolist() == [1, 5, 9]

        # should not fail with pd
        df = tree.arrays([expression], library="pd")
        assert df.columns.tolist() == [expression]
        assert df[expression].tolist() == [1, 5, 9]
