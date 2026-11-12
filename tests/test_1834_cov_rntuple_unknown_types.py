# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import pytest

import uproot

ak = pytest.importorskip("awkward")


@pytest.mark.parametrize(
    "chunk",
    [
        # "b" has lists of a fixed size where the RNTuple has variable-length lists
        {
            "a": ak.Array([[], []]),
            "b": ak.to_regular(ak.Array([[7], [8]])),
            "r": ak.Array([{"x": [], "y": []}] * 2),
        },
        # "r" has a field "z" in place of "y"
        {
            "a": ak.Array([[], []]),
            "b": ak.Array([[], []]),
            "r": ak.Array([{"x": [], "z": [1]}] * 2),
        },
    ],
)
def test_a_chunk_with_unknown_types_must_match_elsewhere(tmp_path, chunk):
    path = tmp_path / "file.root"
    data = {
        "a": ak.Array([[1], [2]]),
        "b": ak.Array([[3], [4]]),
        "r": ak.Array([{"x": [5], "y": [6]}] * 2),
    }
    with uproot.recreate(path) as file:
        ntuple = file.mkrntuple("ntuple", data)
        with pytest.raises(ValueError, match="not compatible"):
            ntuple.extend(chunk)

    with uproot.open(path) as file:
        assert file["ntuple"].arrays().to_list() == ak.Array(data).to_list()
