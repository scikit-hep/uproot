# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("dask_awkward")


def test():
    da = uproot.dask(skhep_testdata.data_path("uproot-issue-791.root") + ":tree")
    assert da[da.int_branch < 0].compute().tolist() == [
        {
            "int_branch": -1,
            "long_branch": -2,
            "float_branch": 3.299999952316284,
            "double_branch": 4.4,
            "bool_branch": True,
        }
    ]
