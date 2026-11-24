# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import uproot
import uproot.writing._cascade
import uproot.const


def test_freesegments_num_bytes_at_boundary():
    # End pointer is below 2GB
    data_small = uproot.writing._cascade.FreeSegmentsData(
        0, (), uproot.const.kStartBigFile - 5
    )
    assert data_small.num_bytes == 10  # small format size

    # End pointer is at or above 2GB
    data_big = uproot.writing._cascade.FreeSegmentsData(
        0, (), uproot.const.kStartBigFile + 5
    )
    assert data_big.num_bytes == 18  # big format size


def test_freesegments_num_bytes_with_location():
    # location is far below boundary, end is None (floating)
    # tentative end would be 0 + 0 + 10 = 10 (< 2GB)
    data_floating_small = uproot.writing._cascade.FreeSegmentsData(0, (), None)
    assert data_floating_small.num_bytes == 10

    # location is near boundary, end is None (floating)
    # tentative end would be (2GB - 5) + 0 + 10 = 2GB + 5 (>= 2GB)
    data_floating_big = uproot.writing._cascade.FreeSegmentsData(
        uproot.const.kStartBigFile - 5, (), None
    )
    assert data_floating_big.num_bytes == 18
