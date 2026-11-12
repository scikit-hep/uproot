# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test():
    with uproot.open(
        skhep_testdata.data_path("uproot-Zmumu-uncompressed.root")
    ) as file:
        basket = file["events/px1"].basket(0)

    # without PR #519, this would be a segfault
    assert basket.data[0] == 192
