# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
awkward = pytest.importorskip("awkward")

COUNTS_DISAGREE = "branch 'nb' provided both as an explicit array and generated as a counter, and they disagree"


def test_awkward_as_numpy(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", {"b1": "var * float64", "b2": "var * float64"})
        fout["tree"].extend(
            {
                "b1": [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
                "b2": np.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], dtype="O"),
            }
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert fin["tree/b2"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


@pytest.mark.parametrize(
    "branch_types",
    [{"nb": "int32", "b": "var * float64"}, {"b": "var * float64", "nb": "int32"}],
    ids=["counter-first", "counter-last"],
)
@pytest.mark.parametrize("names", [("nb", "b"), ("b", "nb")], ids=["nb-b", "b-nb"])
def test_counter_shadows_branch_in_extend(tmp_path, branch_types, names):
    newfile = os.path.join(tmp_path, "newfile.root")

    def data(counts):
        arrays = {"nb": counts, "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]}
        return {name: arrays[name] for name in names}

    with uproot.recreate(newfile) as fout:
        fout.mktree("tree", branch_types)
        with pytest.raises(ValueError, match=COUNTS_DISAGREE):
            fout["tree"].extend(data([1, 2, 3]))
        fout["tree"].extend(data([3, 0, 2]))

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


@pytest.mark.parametrize("names", [("nb", "b"), ("b", "nb")], ids=["nb-b", "b-nb"])
def test_counter_shadows_branch_in_mktree(tmp_path, names):
    newfile = os.path.join(tmp_path, "newfile.root")

    def data(counts):
        arrays = {"nb": counts, "b": [[1.1, 2.2, 3.3], [], [4.4, 5.5]]}
        return {name: arrays[name] for name in names}

    with uproot.recreate(newfile) as fout:
        with pytest.raises(ValueError, match=COUNTS_DISAGREE):
            fout.mktree("tree", data([1, 2, 3]))
        fout.mktree("tree", data([3, 0, 2]))

    with uproot.open(newfile) as fin:
        assert fin["tree/nb"].array().tolist() == [3, 0, 2]
        assert fin["tree/b"].array().tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
