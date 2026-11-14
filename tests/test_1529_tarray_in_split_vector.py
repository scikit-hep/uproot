# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot

# entries: two tracks, no tracks, one track with an empty TArrayF, one track
EXPECTED = [
    [[1.0, 2.0, 3.0], [1.0]],
    [],
    [[]],
    [[1.0, 2.0]],
]


@pytest.fixture
def branch():
    with uproot.open(skhep_testdata.data_path("uproot-issue-1529.root")) as file:
        yield file["Test/Event/fTracks/fTracks.fPositions"]


def test_tarray_member_in_split_vector_ak(branch):
    array = branch.array(library="ak")
    assert str(array.type) == "4 * var * var * float32"
    assert array.tolist() == EXPECTED


def test_tarray_member_in_split_vector_np(branch):
    array = branch.array(library="np")
    assert [[list(tarray) for tarray in entry] for entry in array] == EXPECTED


def test_tarray_member_in_split_vector_entry_range(branch):
    array = branch.array(entry_start=1, entry_stop=3, library="ak")
    assert array.tolist() == EXPECTED[1:3]
