# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot


@pytest.mark.parametrize("reader", ["iterate", "concatenate"])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("filter_branches", [False, True])
def test_open_tree(tmp_path, reader, as_list, filter_branches):
    path = tmp_path / "tree.root"
    data = {"x": np.arange(5), "y": np.arange(5) + 10}
    with uproot.recreate(path) as file:
        file.mktree("tree", data)

    with uproot.open(path) as file:
        tree = file["tree"]
        files = [tree] if as_list else tree
        options = {"library": "np"}
        if filter_branches:
            options["filter_branch"] = lambda branch: branch.name == "x"

        if reader == "iterate":
            chunks = list(uproot.iterate(files, step_size=2, **options))
            assert [len(chunk["x"]) for chunk in chunks] == [2, 2, 1]
            arrays = {
                name: np.concatenate([chunk[name] for chunk in chunks])
                for name in chunks[0]
            }
        else:
            arrays = uproot.concatenate(files, **options)

        assert set(arrays) == ({"x"} if filter_branches else set(data))
        for name, array in arrays.items():
            np.testing.assert_array_equal(array, data[name])

        # Reading through a caller-owned tree must not close its file.
        assert not file.file.closed
        np.testing.assert_array_equal(tree["x"].array(library="np"), data["x"])
