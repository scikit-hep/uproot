# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import pytest

import uproot


@pytest.mark.parametrize(
    ("method", "directory"), [("mktree", "tree"), ("mkrntuple", "ntpl")]
)
def test_repeated_same_subdir(tmp_path, method, directory):
    newfile = os.path.join(tmp_path, "newfile.root")
    names = ["x", "y", "z", "w"]
    expected = {directory, *(f"{directory}/{name}" for name in names)}

    with uproot.recreate(newfile) as f:
        for i, name in enumerate(names):
            getattr(f, method)(f"{directory}/{name}", {"x": [i, 2 * i, 3 * i]})

        assert set(f.keys(cycle=False)) == expected

    with uproot.open(newfile) as fin:
        assert set(fin.keys(cycle=False)) == expected
        for i, name in enumerate(names):
            values = fin[f"{directory}/{name}"]["x"].array(library="np")
            assert values.tolist() == [i, 2 * i, 3 * i]
