# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import uproot
import os
import numpy as np

ROOT = pytest.importorskip("ROOT")


def test_read_free_floating_vector(tmp_path):
    newfile = os.path.join(tmp_path, "test_freevec.root")
    f = ROOT.TFile(newfile, "recreate")
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    c = np.array([0, 1, 4, 5, 6, 7, 8, 9], dtype=np.uint32)
    avec = ROOT.std.vector("double")(a)
    bvec = ROOT.std.vector("unsigned int")(c)
    f.WriteObject(avec, "avec")
    f.WriteObject(bvec, "bvec")
    f.Write()
    f.Close()

    with uproot.open(newfile) as f:
        assert f["avec"].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert f["bvec"].tolist() == [0, 1, 4, 5, 6, 7, 8, 9]
