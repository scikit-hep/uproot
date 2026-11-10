# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``graphed_partitions(align_baskets=True)`` moves chunk boundaries onto ``TBasket``
boundaries, and the ``_basket_offsets`` rule it rests on."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed_helpers.baskets import (  # noqa: E402
    aligned_ranges,
    common_offsets,
    count_baskets,
    even_bounds,
    sized_bounds,
)
from uproot._graphed import graphed_partitions, read_graphed_partition  # noqa: E402


def _path(file, tree):
    return skhep_testdata.data_path(file) + ":" + tree


FORITER = _path("uproot-foriter.root", "foriter")
ZMUMU = _path("uproot-Zmumu.root", "events")
HEPDATA = _path("uproot-hepdata-example.root", "ntuple")
FULLSPLIT = _path("uproot-small-evnt-tree-fullsplit.root", "tree")
EMPTY = _path("uproot-empty.root", "tree")


def _ranges(parts):
    return [(p.entry_start, p.entry_stop) for p in parts]


def _every_branch(tree):
    return list(tree.itervalues(recursive=True))


def test_the_rule_over_both_ends_of_the_branch_family():
    _basket_offsets = uproot._graphed._basket_offsets
    assert _basket_offsets([], 10) is None
    assert _basket_offsets([[0]], 0) == [0]
    hep = uproot.open(HEPDATA)
    n = hep.num_entries
    px, random = hep["px"].common_entry_offsets(), hep["random"].common_entry_offsets()
    assert _basket_offsets([px], n) == hep["px"].entry_offsets
    assert _basket_offsets([px, random], n) == [0, n]
    evt = uproot.open(FULLSPLIT)["evt"]
    assert evt.num_baskets == 0 and evt.entry_offsets == [0]
    assert _basket_offsets([evt.common_entry_offsets()], evt.num_entries) == [
        0,
        evt.num_entries,
    ]


def test_a_branch_selection_follows_only_those_baskets():
    hep = uproot.open(HEPDATA)
    n = hep.num_entries
    px_only = graphed_partitions(HEPDATA, steps_per_file=5, align_baskets="px")
    assert _ranges(px_only) == aligned_ranges(
        even_bounds(n, 5), common_offsets([hep["px"]])
    )
    assert len(px_only) == 5
    for kwargs in ({"align_baskets": True}, {"align_baskets": ["px", "random"]}):
        assert _ranges(graphed_partitions(HEPDATA, steps_per_file=5, **kwargs)) == [
            (0, n)
        ]
    with pytest.raises(ValueError, match="no TBranch with TBaskets"):
        graphed_partitions(HEPDATA, steps_per_file=5, align_baskets="no_such*")


@pytest.mark.parametrize(
    "where, kwargs",
    [
        (FORITER, {"steps_per_file": 4}),
        (FORITER, {"steps_per_file": 5}),
        (FORITER, {"step_size": 10}),
        (ZMUMU, {"steps_per_file": 5}),
        (HEPDATA, {"steps_per_file": 5}),
        (FULLSPLIT, {"steps_per_file": 4}),
    ],
)
def test_aligned_ranges_follow_the_shared_basket_offsets(where, kwargs):
    tree = uproot.open(where)
    n = tree.num_entries
    if "step_size" in kwargs:
        bounds = sized_bounds(n, kwargs["step_size"])
    else:
        bounds = even_bounds(n, kwargs["steps_per_file"])
    expected = aligned_ranges(bounds, common_offsets(_every_branch(tree)))
    got = _ranges(graphed_partitions(where, align_baskets=True, **kwargs))
    assert got == expected
    assert got != _ranges(graphed_partitions(where, **kwargs)) or len(got) == 1


def test_the_contract_examples():
    ranges = {
        key: _ranges(
            graphed_partitions(FORITER, align_baskets=True, **{key[0]: key[1]})
        )
        for key in (("steps_per_file", 4), ("steps_per_file", 5), ("step_size", 10))
    }
    assert ranges == {
        ("steps_per_file", 4): [(0, 12), (12, 24), (24, 36), (36, 46)],
        ("steps_per_file", 5): [(0, 6), (6, 18), (18, 24), (24, 36), (36, 46)],
        ("step_size", 10): [(0, 12), (12, 18), (18, 30), (30, 42), (42, 46)],
    }
    assert _ranges(graphed_partitions(ZMUMU, align_baskets=True, steps_per_file=5)) == [
        (0, 2304)
    ]
    assert _ranges(
        graphed_partitions(HEPDATA, align_baskets=True, steps_per_file=5)
    ) == [(0, 75000)]
    assert _ranges(
        graphed_partitions(FULLSPLIT, align_baskets=True, steps_per_file=4)
    ) == [(0, 100)]
    for kwargs in ({"steps_per_file": 3}, {"step_size": 10}):
        assert graphed_partitions(EMPTY, align_baskets=True, **kwargs) == []
        assert graphed_partitions(EMPTY, **kwargs) == []


def test_each_basket_is_read_once():
    aligned = graphed_partitions(FORITER, align_baskets=True, steps_per_file=5)
    with count_baskets() as counts:
        for part in aligned:
            read_graphed_partition(part, ["data"])
    n_baskets = uproot.open(FORITER)["data"].num_baskets
    assert counts == {("data", i): 1 for i in range(n_baskets)}
    with count_baskets() as control:
        for part in graphed_partitions(FORITER, steps_per_file=5):
            read_graphed_partition(part, ["data"])
    assert max(control.values()) > 1


def _values(where, partitions, column):
    plan = graphed.aggregate_plan(
        getattr(uproot.graphed(where), column),
        reduce=lambda vals: [vals[0]],
        combine=lambda a, b: a + b,
        empty=list,
        partitions=partitions,
    )
    return ak.concatenate(SequentialRunner().run(plan).value)


def _counts(where, partitions, column):
    plan = graphed.aggregate_plan(
        getattr(uproot.graphed(where), column),
        reduce=lambda vals: int(ak.count_nonzero(vals[0] > 0)),
        combine=lambda a, b: a + b,
        empty=lambda: 0,
        partitions=partitions,
    )
    return SequentialRunner().run(plan).value


@pytest.mark.parametrize(
    "where, column", [(FORITER, "data"), (ZMUMU, "px1"), (HEPDATA, "px")]
)
def test_aligned_partitions_give_the_unaligned_result_bit_for_bit(where, column):
    whole = graphed_partitions(where, steps_per_file=1)
    aligned = graphed_partitions(where, align_baskets=True, steps_per_file=5)
    tree = uproot.open(where)
    assert _ranges(aligned) == aligned_ranges(
        even_bounds(tree.num_entries, 5), common_offsets(_every_branch(tree))
    )
    ref = _values(where, whole, column)
    got = _values(where, aligned, column)
    assert np.asarray(got).tobytes() == np.asarray(ref).tobytes()
    assert _counts(where, aligned, column) == _counts(where, whole, column)


def test_refusals():
    with pytest.raises(TypeError, match="open_files"):
        graphed_partitions(
            FORITER, steps_per_file=2, open_files=False, align_baskets=True
        )


def test_an_rntuple_refuses_alignment(tmp_path):
    path = os.path.join(tmp_path, "nt.root")
    with uproot.recreate(path) as f:
        f["nt"] = ak.Array([{"x": i} for i in range(6)])
    assert len(graphed_partitions(path + ":nt", steps_per_file=2)) == 2
    with pytest.raises(NotImplementedError, match="align_baskets"):
        graphed_partitions(path + ":nt", steps_per_file=2, align_baskets=True)


def test_the_default_is_unaligned():
    n = uproot.open(FORITER).num_entries
    for kwargs, bounds in (
        ({"steps_per_file": 5}, even_bounds(n, 5)),
        ({"step_size": 7}, sized_bounds(n, 7)),
    ):
        expected = list(zip(bounds, bounds[1:]))
        assert _ranges(graphed_partitions(FORITER, **kwargs)) == expected
        assert (
            _ranges(graphed_partitions(FORITER, align_baskets=False, **kwargs))
            == expected
        )
        assert (
            _ranges(graphed_partitions(FORITER, align_baskets=True, **kwargs))
            != expected
        )
