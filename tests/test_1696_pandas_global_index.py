# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""Regression tests for issue #1688: Pandas global indices.

``uproot.concatenate``/``uproot.iterate`` re-index each file's DataFrames so that
entry numbers are global rather than per-file, but ``how=dict`` was not handled
and a non-``RangeIndex`` (from a ``cut``) was shifted in place. ``RNTuple.arrays``
and ``RField.array`` built their Pandas index from a cluster-relative or
unregularized entry number.
"""

from __future__ import annotations

import numpy as np
import pytest

import uproot

pd = pytest.importorskip("pandas")


def _members(result, how):
    assert isinstance(result, pd.DataFrame if how is None else how)
    if how is None:
        return [result["x"]]
    elif how is dict:
        return list(result.values())
    else:
        return list(result)


@pytest.mark.parametrize("cut", [None, "x % 3 != 0"])
@pytest.mark.parametrize("how", [None, dict, tuple, list])
def test_multifile_index_is_global(tmp_path, how, cut):
    # every branch holds its global entry number, so index == values
    files = {}
    for i in range(2):
        path = str(tmp_path / f"file{i}.root")
        with uproot.recreate(path) as f:
            f.mktree("t", {"x": np.int64, "y": np.float64}).extend(
                {"x": np.arange(5 * i, 5 * i + 5), "y": np.arange(5 * i, 5 * i + 5)}
            )
        files[path] = "t"
    expected = [i for i in range(10) if cut is None or i % 3 != 0]

    result = uproot.concatenate(files, ["x", "y"], cut=cut, library="pd", how=how)
    for member in _members(result, how):
        assert member.index.tolist() == member.tolist() == expected

    chunks = list(
        uproot.iterate(files, ["x", "y"], cut=cut, library="pd", how=how, step_size=3)
    )
    for i in range(len(_members(chunks[0], how))):
        members = [_members(chunk, how)[i] for chunk in chunks]
        for member in members:
            assert member.index.tolist() == member.tolist()
        assert sum((member.index.tolist() for member in members), []) == expected


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [(None, None), (0, 3), (2, 6), (4, 6), (5, 8), (6, 7), (-2, None), (3, 100)],
)
def test_rntuple_index_is_global(tmp_path, entry_start, entry_stop):
    # two clusters, [0, 4) and [4, 8), and a TTree with the same content
    path = str(tmp_path / "file.root")
    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("nt", {"x": np.int64})
        tree = f.mktree("t", {"x": np.int64})
        for chunk in (np.arange(0, 4), np.arange(4, 8)):
            ntuple.extend({"x": chunk})
            tree.extend({"x": chunk})
    expected = list(range(8))[entry_start:entry_stop]

    kwargs = {"entry_start": entry_start, "entry_stop": entry_stop, "library": "pd"}
    with uproot.open(path) as f:
        from_ntuple = f["nt"].arrays(**kwargs)
        from_field = f["nt"]["x"].array(**kwargs)
        from_tree = f["t"].arrays(**kwargs)

    assert from_ntuple["x"].tolist() == from_field.tolist() == expected
    assert from_ntuple.index.tolist() == from_field.index.tolist() == expected
    assert from_tree.index.tolist() == expected
