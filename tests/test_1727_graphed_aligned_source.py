# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
"""``uproot.graphed(align_baskets=True)``: blind partitions snap to the ``TBasket``
boundaries of the branches a partition reads, when it is read."""

from __future__ import annotations

import itertools
import os
import pickle

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed_helpers import mini_nanoaod as mini  # noqa: E402
from tests.graphed_helpers.baskets import (  # noqa: E402
    aligned_blind_lengths,
    common_offsets,
    count_baskets,
    even_bounds,
)
from uproot._graphed import _partition_range, _uproot_source  # noqa: E402


def _path(file, tree):
    return skhep_testdata.data_path(file) + ":" + tree


FORITER = _path("uproot-foriter.root", "foriter")
ZMUMU = _path("uproot-Zmumu.root", "events")
HEPDATA = _path("uproot-hepdata-example.root", "ntuple")
EMPTY = _path("uproot-empty.root", "tree")


def _source_of(array):
    return _uproot_source(array, "_source_of")[1]


def _lengths(out, steps_per_file):
    """Per-task output lengths, in key order."""
    plan = graphed.aggregate_plan(
        out,
        reduce=lambda vals: [len(vals[0])],
        combine=lambda a, b: a + b,
        empty=list,
        steps_per_file=steps_per_file,
    )
    return SequentialRunner().run(plan).value


def _outputs(out, steps_per_file):
    plan = graphed.aggregate_plan(
        out,
        reduce=lambda vals: [vals[0]],
        combine=lambda a, b: a + b,
        empty=list,
        steps_per_file=steps_per_file,
    )
    return ak.concatenate(SequentialRunner().run(plan).value)


def _buffers(array):
    form, length, buffers = ak.to_buffers(ak.to_packed(array))
    return (
        form.to_json(),
        length,
        {k: np.asarray(v).tobytes() for k, v in buffers.items()},
    )


def _rule_lengths(where, branches, steps):
    tree = uproot.open(where)
    offsets = common_offsets([tree[b] for b in branches])
    return aligned_blind_lengths(tree.num_entries, steps, offsets)


def _mini_nano(tmp_path):
    """Five ``extend`` calls: one ``TBasket`` per branch per call."""
    path = os.path.join(tmp_path, "nano.root")
    rng = np.random.default_rng(1727)
    with uproot.recreate(path) as f:
        for i, n in enumerate((7, 5, 9, 4, 6)):
            nj, nm = rng.integers(0, 4, n), rng.integers(0, 3, n)
            chunk = {
                "Jet": ak.unflatten(
                    ak.zip(
                        {
                            "pt": rng.uniform(10, 50, nj.sum()).astype("f4"),
                            "eta": rng.normal(size=nj.sum()).astype("f4"),
                        }
                    ),
                    nj,
                ),
                "Muon": ak.unflatten(
                    ak.zip({"pt": rng.uniform(5, 30, nm.sum()).astype("f4")}), nm
                ),
                "MET_pt": ak.Array(rng.uniform(0, 100, n).astype("f4")),
            }
            if i == 0:
                f.mktree("Events", {k: v.type for k, v in chunk.items()})
            f["Events"].extend(chunk)
    return path + ":Events"


def _rntuple(tmp_path):
    path = os.path.join(tmp_path, "nt.root")
    with uproot.recreate(path) as f:
        f["nt"] = ak.Array(
            [{"rec": {"x": i, "y": 2 * i}, "z": 3 * i} for i in range(6)]
        )
    return path + ":nt"


# ---- 1. snap at read time ------------------------------------------------------------
@pytest.mark.parametrize("steps", [4, 5])
def test_blind_steps_snap_to_basket_offsets_at_read_time(steps):
    g = uproot.graphed(FORITER, align_baskets=True)
    expected = _rule_lengths(FORITER, ["data"], steps)
    assert _lengths(g.data, steps) == expected
    assert expected == {4: [12, 12, 12, 10], 5: [6, 12, 6, 12, 10]}[steps]


# ---- 2. reconciliation over the read list --------------------------------------------
def _boundaries(lengths):
    return list(itertools.accumulate(lengths, initial=0))


def test_offsets_are_reconciled_over_the_branches_a_partition_reads():
    g = uproot.graphed(HEPDATA, align_baskets=True)
    tree = uproot.open(HEPDATA)
    for column in ("px", "random"):
        lengths = _lengths(getattr(g, column), 5)
        assert sum(1 for n in lengths if n) >= 2
        assert set(_boundaries(lengths)) <= set(tree[column].entry_offsets)
    both = _lengths(g.px + g.random, 5)
    assert sum(1 for n in both if n) == 1
    assert sum(both) == tree.num_entries


# ---- 3. the read family, both ends ---------------------------------------------------
def test_entry_range_over_missing_and_empty_read_lists():
    hep = uproot.open(HEPDATA)
    zmumu = uproot.open(ZMUMU)
    hep_src = _source_of(uproot.graphed(HEPDATA, align_baskets=True))
    zmumu_src = _source_of(uproot.graphed(ZMUMU, align_baskets=True))
    uri = HEPDATA.rsplit(":", 1)[0]
    for s in range(5):
        step = graphed.core.Partition.blind(uri, "ntuple", s, 5)
        px_only = hep_src.entry_range(step, hep, ["px"])
        assert hep_src.entry_range(step, hep, ["px", "no_such_branch"]) == px_only
        assert tuple(hep_src.entry_range(step, hep, ["no_such_branch"])) == tuple(
            _partition_range(step, hep.num_entries)
        )
        assert tuple(zmumu_src.entry_range(step, zmumu, [])) == tuple(
            _partition_range(step, zmumu.num_entries)
        )
    assert [
        b - a
        for a, b in (
            hep_src.entry_range(
                graphed.core.Partition.blind(uri, "ntuple", s, 5), hep, ["px"]
            )
            for s in range(5)
        )
    ] == _rule_lengths(HEPDATA, ["px"], 5)


def test_a_tree_without_baskets_reads_as_empty():
    g = uproot.graphed(EMPTY, align_baskets=True)
    assert _lengths(g.x, 3) == [0, 0, 0]
    assert _source_of(g).entry_range(
        graphed.core.Partition.blind(EMPTY.rsplit(":", 1)[0], "tree", 1, 3),
        uproot.open(EMPTY),
        ["x"],
    ) == (0, 0)


# ---- 4. each basket once -------------------------------------------------------------
@pytest.mark.parametrize("steps", [2, 3, 5, 20])
def test_each_basket_is_read_once_per_run(steps):
    g = uproot.graphed(HEPDATA, align_baskets=True)
    n_baskets = uproot.open(HEPDATA)["px"].num_baskets
    with count_baskets() as counts:
        lengths = _lengths(g.px, steps)
    assert counts == {("px", i): 1 for i in range(n_baskets)}
    assert lengths == _rule_lengths(HEPDATA, ["px"], steps)
    with count_baskets() as control:
        _lengths(uproot.graphed(HEPDATA, name="plain").px, 5)
    assert max(control.values()) > 1


# ---- 5. empty tasks read nothing -----------------------------------------------------
def test_empty_tasks_read_no_basket():
    g = uproot.graphed(ZMUMU, align_baskets=True)
    with count_baskets() as counts:
        lengths = _lengths(g.px1, 4)
    assert len(lengths) == 4 and lengths.count(0) == 3
    assert counts == {("px1", 0): 1}
    n = uproot.open(FORITER).num_entries
    steps = n + 4
    with count_baskets() as unaligned:
        lengths = _lengths(uproot.graphed(FORITER).data, steps)
    assert lengths.count(0) == steps - n
    assert (
        sum(unaligned.values()) == n
    )  # one basket per one-entry task, none for the empty ones


# ---- 6. bit-for-bit ------------------------------------------------------------------
@pytest.mark.parametrize(
    "where, column, branches",
    [
        (HEPDATA, "px", ["px"]),
        (FORITER, "data", ["data"]),
        (ZMUMU, "px1", ["px1"]),
    ],
)
def test_aligned_reads_are_bit_for_bit_flat(where, column, branches):
    aligned = getattr(uproot.graphed(where, align_baskets=True), column)
    plain = getattr(uproot.graphed(where, name="plain"), column)
    assert _lengths(aligned, 5) == _rule_lengths(where, branches, 5)
    ref = _buffers(_outputs(plain, 1))
    assert _buffers(_outputs(aligned, 5)) == ref
    assert _buffers(_outputs(plain, 5)) == ref
    assert sum(_lengths(aligned, 5)) == sum(_lengths(plain, 1))


def test_aligned_reads_are_bit_for_bit_mapped(tmp_path):
    where = _mini_nano(tmp_path)

    def jets(**kwargs):
        return uproot.graphed(
            where,
            filter_name=mini.FILTER,
            form_mapping=mini.MiniNanoMapping(),
            **kwargs,
        ).Jet.pt

    aligned, plain = jets(align_baskets=True), jets(name="plain")
    assert _lengths(aligned, 5) == _rule_lengths(where, ["nJet", "Jet_pt"], 5)
    assert _lengths(aligned, 5) != _lengths(plain, 5)
    ref = _buffers(_outputs(plain, 1))
    assert _buffers(_outputs(aligned, 5)) == ref
    assert _buffers(_outputs(plain, 5)) == ref
    count_ref = ak.num(_outputs(plain, 1)).to_numpy().tobytes()
    assert ak.num(_outputs(aligned, 5)).to_numpy().tobytes() == count_ref


# ---- 7. blind and counted; the writer reads the aligned range ------------------------
def test_aligned_plans_stay_blind_and_counted():
    g = uproot.graphed(HEPDATA, align_baskets=True)
    plan = graphed.aggregate_plan(
        g.px,
        reduce=lambda vals: [len(vals[0])],
        combine=lambda a, b: a + b,
        empty=list,
        steps_per_file=5,
    )
    assert len(plan.tasks) == 5
    assert all(task.partition.is_blind for task in plan.tasks)
    assert SequentialRunner().run(plan).value == _rule_lengths(HEPDATA, ["px"], 5)


def test_the_writer_reads_the_aligned_range(tmp_path):
    pytest.importorskip("graphed_executors.local")
    g = uproot.graphed(HEPDATA, align_baskets=True)
    with count_baskets() as counts:
        paths = uproot.graphed_write(
            gak.zip({"px": g.px}),
            os.path.join(tmp_path, "out"),
            steps_per_file=5,
            executor="thread",
            max_workers=1,
        )
    pieces = [uproot.open(p)["tree"].arrays() for p in paths]
    expected = [n for n in _rule_lengths(HEPDATA, ["px"], 5) if n]
    assert [len(p) for p in pieces] == expected
    ref = uproot.open(HEPDATA)["px"].array()
    assert (
        np.asarray(ak.concatenate([p.px for p in pieces])).tobytes()
        == np.asarray(ref).tobytes()
    )
    n_baskets = uproot.open(HEPDATA)["px"].num_baskets
    assert {k: v for k, v in counts.items() if k[0] == "px"} == {
        ("px", i): 1 for i in range(n_baskets)
    }


# ---- 8. refusals ---------------------------------------------------------------------
def test_explicit_steps_refuse_alignment_before_any_file_opens(tmp_path):
    files = {os.path.join(tmp_path, "no_such.root"): {"steps": [[0, 100]]}}
    with pytest.raises(FileNotFoundError):
        uproot.graphed(files)
    with pytest.raises(TypeError, match="align_baskets"):
        uproot.graphed(files, align_baskets=True)


def test_an_rntuple_refuses_alignment(tmp_path):
    where = _rntuple(tmp_path)
    g = uproot.graphed(where)
    assert _lengths(g.z, 2) == [3, 3]
    with pytest.raises(NotImplementedError, match="align_baskets"):
        uproot.graphed(where, align_baskets=True)
    form = g.session.form(g).tt.layout.form
    lazy = uproot.graphed(where, align_baskets=True, known_base_form=form, name="lazy")
    with pytest.raises(NotImplementedError, match="align_baskets"):
        _lengths(lazy.z, 2)


# ---- 8b. the fold runs once per file and read list -----------------------------------
def test_the_offset_fold_runs_once_per_file_and_read_list(monkeypatch):
    calls = []
    original = uproot.behaviors.TBranch.HasBranches.common_entry_offsets

    def counting(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        uproot.behaviors.TBranch.HasBranches, "common_entry_offsets", counting
    )
    g = uproot.graphed(HEPDATA, align_baskets=True)
    expected = _rule_lengths(HEPDATA, ["px"], 5)
    assert _lengths(g.px, 5) == expected
    assert calls == ["px"]
    assert _lengths(g.px, 5) == expected
    assert calls == ["px"]
    _lengths(g.px + g.random, 5)
    assert sorted(calls) == ["px", "px", "random"]
    assert pickle.loads(pickle.dumps(_source_of(g)))._offsets_memo == {}


# ---- 9. default unchanged ------------------------------------------------------------
def test_the_default_splits_evenly():
    n = uproot.open(FORITER).num_entries
    bounds = even_bounds(n, 5)
    even = [b - a for a, b in zip(bounds, bounds[1:])]
    assert _lengths(uproot.graphed(FORITER).data, 5) == even
    assert _lengths(uproot.graphed(FORITER, align_baskets=False).data, 5) == even
    aligned = uproot.graphed(FORITER, align_baskets=True, name="aligned")
    assert _lengths(aligned.data, 5) != even
