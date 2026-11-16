# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_numpy(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        t1 = fout.mktree("t1", {"b": "f8"})
        t1.extend({"b": [1.1, 2.2, 3.3]})

        t2 = fout.mktree("t2", {"b": "f8"})
        t2.extend({"b": [4.4, 5.5]})

    with uproot.update(newfile) as fup:
        t3 = fup.mktree("t3", {"b": "f8"})
        t3.extend({"b": [6.6, 7.7]})

        t4 = fup.mktree("t4", {"b": "f8"})
        t4.extend({"b": [8.8, 9.9]})

    with uproot.open(newfile) as fin:
        assert fin["t1/b"].array(library="np").tolist() == [1.1, 2.2, 3.3]
        assert fin["t2/b"].array(library="np").tolist() == [4.4, 5.5]
        assert fin["t3/b"].array(library="np").tolist() == [6.6, 7.7]
        assert fin["t4/b"].array(library="np").tolist() == [8.8, 9.9]
