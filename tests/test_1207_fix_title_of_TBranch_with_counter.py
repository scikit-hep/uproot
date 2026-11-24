# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import awkward as ak

import uproot


def test(tmp_path):
    filename = os.path.join(tmp_path, "file.root")

    with uproot.recreate(filename) as file:
        file.mktree("tree", {"branch": ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])})

    with uproot.open(filename) as file:
        assert file["tree"]["branch"].title == "branch[nbranch]/D"
        assert (
            file["tree"]["branch"].member("fLeaves")[0].member("fTitle")
            == "branch[nbranch]"
        )
