# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
"""``HasBranches.common_entry_offsets`` over both ends of the branch family: a leaf, a
split parent without ``TBaskets``, a selection that matches nothing, an empty ``TTree``.
"""

from __future__ import annotations

import pytest
import skhep_testdata

import uproot


def _open(file, tree):
    return uproot.open(skhep_testdata.data_path(file) + ":" + tree)


def _shared(branches):
    sets = [set(b.entry_offsets) for b in branches if b.num_baskets]
    return sorted(set.intersection(*sets))


@pytest.mark.parametrize(
    "file, tree, branch",
    [
        ("uproot-HZZ.root", "events", "Muon_Px"),
        ("uproot-foriter.root", "foriter", "data"),
        ("uproot-hepdata-example.root", "ntuple", "px"),
    ],
)
def test_a_leaf_branch_reports_its_own_entry_offsets(file, tree, branch):
    leaf = _open(file, tree)[branch]
    assert not leaf.branches
    assert len(leaf.entry_offsets) > 2
    assert leaf.common_entry_offsets() == leaf.entry_offsets
    assert leaf.common_entry_offsets(filter_name="no_such*") == leaf.entry_offsets


def test_a_branch_without_baskets_constrains_nothing():
    tree = _open("uproot-small-evnt-tree-fullsplit.root", "tree")
    every = list(tree.itervalues(recursive=True))
    assert any(b.num_baskets == 0 for b in every)
    expected = _shared(every)
    assert expected[-1] == tree.num_entries > 0
    assert tree.common_entry_offsets() == expected
    evt = tree["evt"]
    assert evt.num_baskets == 0 and evt.entry_offsets == [0]
    assert evt.common_entry_offsets() == _shared(evt.itervalues(recursive=True))


def test_nothing_with_baskets_gives_zero():
    tree = _open("uproot-hepdata-example.root", "ntuple")
    assert tree.common_entry_offsets(filter_name="no_such*") == [0]
    empty = _open("uproot-empty.root", "tree")
    assert empty.num_entries == 0
    assert empty.common_entry_offsets() == [0]
    assert empty["x"].common_entry_offsets() == empty["x"].entry_offsets == [0]
