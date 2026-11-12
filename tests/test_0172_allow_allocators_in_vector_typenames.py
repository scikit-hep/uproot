# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue-172.root")) as f:
        t = f["events"]
        assert (
            t["rec_part_px_VecOps"].typename == "std::vector<float>"
        )  # without the allocator
        array = t["rec_part_px_VecOps"].array(library="np")
        assert [len(x) for x in array] == [24, 44, 49, 39, 36, 32, 43, 29, 29, 30]
        assert array[0][:2].tolist() == [-0.10093698650598526, -0.17859043180942535]
