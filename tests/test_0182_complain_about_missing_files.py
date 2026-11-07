# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test():
    one = skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")
    two = skhep_testdata.data_path("uproot-sample-6.18.00-uncompressed.root")
    bad = one.replace(".root", "-DOES-NOT-EXIST.root")
    okay = one.replace(".root", "-DOES-NOT-EXIST-*.root")

    def num_entries(files):
        return [
            len(arrays["i4"])
            for arrays in uproot.iterate(files, "i4", step_size="1 TB", library="np")
        ]

    assert num_entries([one, two]) == [30, 30]

    with pytest.raises(FileNotFoundError, match="DOES-NOT-EXIST"):
        list(uproot.iterate([one, two, bad], library="np"))

    assert num_entries([one, two, okay]) == [30, 30]
