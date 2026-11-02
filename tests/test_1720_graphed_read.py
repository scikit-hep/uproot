# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed``: construction, file/tree resolution shared with ``uproot.dask``, the
whole-dataset loader, and ``graphed_head``."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _make_tree(path, n=10, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8", "y": "f8"})
        f[name].extend({"x": np.arange(n, dtype="f8"), "y": 2.0 * np.arange(n)})
    return path


def _sum_plan(out, **kwargs):
    return graphed.aggregate_plan(
        out,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        **kwargs,
    )


def _run(plan):
    return SequentialRunner().run(plan).value


def _source_of(array):
    _nid, src = uproot._graphed._uproot_source(array, "test")
    return src


# ---- construction ------------------------------------------------------------------------
def test_construction_reads_only_metadata(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    opened = []
    real_open = uproot.reading.ReadOnlyFile.__init__

    def counting(self, *args, **kwargs):
        opened.append(args[0])
        real_open(self, *args, **kwargs)

    uproot.reading.ReadOnlyFile.__init__ = counting
    try:
        g = uproot.graphed(path + ":events")
    finally:
        uproot.reading.ReadOnlyFile.__init__ = real_open
    assert opened == [path]  # one open for the form
    assert isinstance(g, graphed.Array)
    assert g.session.form(g).describe() == "## * {x: float64, y: float64}"
    assert g.session.backend.__class__.__name__ == "AwkwardBackend"


def test_np_library_is_not_implemented():
    with pytest.raises(NotImplementedError, match="library='ak'"):
        uproot.graphed(_zmumu(), library="np")


@pytest.mark.parametrize(
    "kwarg",
    [
        {"step_size": "1 MB"},
        {"steps_per_file": 2},
        {"allow_read_errors_with_report": True},
    ],
)
def test_dask_only_keywords_are_refused_by_name(kwarg):
    (name,) = kwarg
    with pytest.raises(TypeError, match=f"got {name}=, which belongs to uproot.dask"):
        uproot.graphed(_zmumu(), **kwarg)


def test_behavior_is_refused_beside_backend():
    from graphed.awkward import AwkwardBackend

    with pytest.raises(TypeError, match="behavior= is for the default backend"):
        uproot.graphed(_zmumu(), behavior={}, backend=AwkwardBackend())


# ---- resolution shared with uproot.dask ------------------------------------------------
def test_a_single_tree_file_needs_no_object_path(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    g = uproot.graphed(path)
    (partition,) = _source_of(g).partitions(1)
    assert partition.tree == "/events;1"  # the RESOLVED path, so a worker can open it
    assert _run(_sum_plan(g.x, steps_per_file=2)) == 45.0


def test_an_open_tree_is_reopened_by_path_in_the_workers(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    g = uproot.graphed(uproot.open(path)["events"])
    (partition,) = _source_of(g).partitions(1)
    assert (partition.uri, partition.tree) == (path, "/events;1")
    assert _run(_sum_plan(g.y, steps_per_file=3)) == 90.0
    # the same normalisation on the path that opens nothing itself
    form = g.session.form(g).tt.layout.form
    known = uproot.graphed(uproot.open(path)["events"], known_base_form=form)
    (partition,) = _source_of(known).partitions(1)
    assert (partition.uri, partition.tree) == (path, "/events;1")


def test_steps_for_some_but_not_all_files_are_refused(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"))
    opened = uproot.graphed(a + ":events")
    form = opened.session.form(opened).tt.layout.form
    with pytest.raises(TypeError, match="some but not all"):
        uproot.graphed(
            {a: {"object_path": "events", "steps": [[0, 5], [5, 10]]}, b: "events"},
            known_base_form=form,
        )


def test_a_leaf_branch_as_object_path_selects_that_branch(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    g = uproot.graphed(path + ":events/x")
    assert g.session.form(g).describe() == "## * {x: float64}"
    assert uproot.graphed_head(g.x, 3).tolist() == [0.0, 1.0, 2.0]


def test_a_grouped_branch_resolves_like_uproot_dask():
    pytest.importorskip("dask_awkward")
    path = skhep_testdata.data_path("uproot-issue-1502.root")  # `branch` is AsGrouped
    for full_paths in (False, True):
        g = uproot.graphed(path, full_paths=full_paths, name=f"g{full_paths}")
        ref = uproot.dask(path, full_paths=full_paths)
        assert g.session.form(g).tt.fields == ref.fields
        assert ak.array_equal(ak.Array(g.session.materialize(g)), ref.compute())


def test_a_nested_rntuple_keeps_its_nesting(tmp_path):
    path = os.path.join(tmp_path, "nt.root")
    with uproot.recreate(path) as f:
        f["nt"] = ak.Array(
            [{"rec": {"x": i, "y": 2 * i}, "z": 3 * i} for i in range(6)]
        )
    g = uproot.graphed(path + ":nt")
    assert g.session.form(g).describe() == "## * {rec: {x: int64, y: int64}, z: int64}"
    assert g.session.form(g.rec.x).describe() == "## * int64"
    assert uproot.graphed_head(g.rec.x, 4).tolist() == [0, 1, 2, 3]
    assert (
        ak.Array(g.session.materialize(g)).tolist()
        == uproot.open(path + ":nt").arrays().tolist()
    )
    assert _run(_sum_plan(g.rec.y, steps_per_file=3)) == 30.0
    # a collection of records: the leaf is read through the collection level
    jets = ak.Array(
        [{"jets": [{"pt": float(j)} for j in range(k % 3)]} for k in range(6)]
    )
    with uproot.recreate(os.path.join(tmp_path, "jets.root")) as f:
        f["nt"] = jets
    gj = uproot.graphed(os.path.join(tmp_path, "jets.root") + ":nt", name="jets")
    assert uproot.graphed_head(gj.jets.pt, 3).tolist() == [[], [0.0], [0.0, 1.0]]
    paths = uproot.graphed_write(
        gak.zip({"pt": gj.jets.pt}), os.path.join(tmp_path, "out"), executor="thread"
    )
    assert uproot.open(paths[0])["tree"].arrays().pt.tolist() == jets.jets.pt.tolist()


def test_allow_missing_drops_a_file_without_the_tree(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"), name="other")
    eager = uproot.graphed(
        [a + ":events", b + ":events"], allow_missing=True, open_files=True
    )
    assert len(_source_of(eager).partitions(1)) == 1
    lazy = uproot.graphed(
        [a + ":events", b + ":events"], allow_missing=True, name="lazy"
    )
    assert len(_source_of(lazy).partitions(1)) == 2  # b is not opened until it is read
    assert _run(_sum_plan(lazy.x)) == _run(_sum_plan(eager.x)) == 45.0
    with pytest.raises(uproot.KeyInFileError):
        uproot.graphed(b + ":events")
    # no object path and no TTree in the file at all
    nott = os.path.join(tmp_path, "nott.root")
    with uproot.recreate(nott) as f:
        f["h"] = np.histogram(np.arange(10.0))
    with pytest.raises(ValueError, match="no TTrees found"):
        uproot.graphed(nott)
    g = uproot.graphed([nott, a], allow_missing=True, name="g")
    assert len(_source_of(g).partitions(1)) == 1


def test_known_base_form_with_allow_missing_reads_a_missing_tree_as_empty(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"), name="other")
    opened = uproot.graphed(a + ":events")
    form = opened.session.form(opened).tt.layout.form
    g = uproot.graphed(
        [a + ":events", b + ":events"],
        known_base_form=form,
        allow_missing=True,
        name="g",
    )
    assert (
        len(_source_of(g).partitions(2)) == 4
    )  # nothing was opened: both files partition
    assert _run(_sum_plan(g.x, steps_per_file=2)) == 45.0  # b's chunks are empty


def test_explicit_steps_are_the_partitions(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    g = uproot.graphed({path: {"object_path": "events", "steps": [[0, 4], [4, 10]]}})
    parts = _source_of(g).partitions(1)
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 4), (4, 10)]
    assert _run(_sum_plan(g.x)) == 45.0
    with pytest.raises(TypeError, match="incompatible with steps_per_file"):
        _source_of(g).partitions(2)


def test_the_whole_dataset_loader_concatenates_every_file(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"), n=4)
    b = _make_tree(os.path.join(tmp_path, "b.root"), n=6)
    g = uproot.graphed([a + ":events", b + ":events"])
    assert ak.Array(g.session.materialize(g.x)).tolist() == [0.0, 1.0, 2.0, 3.0] + [
        float(i) for i in range(6)
    ]
    # under known_base_form nothing is opened up front: a missing tree is skipped at read time,
    # and a dataset with no tree at all loads as empty
    form = g.session.form(g).tt.layout.form
    missing = _make_tree(os.path.join(tmp_path, "c.root"), name="other") + ":events"
    partial = uproot.graphed(
        [missing, b + ":events"],
        known_base_form=form,
        allow_missing=True,
        name="partial",
    )
    assert ak.Array(partial.session.materialize(partial.x)).tolist() == [
        float(i) for i in range(6)
    ]
    none = uproot.graphed(
        [missing], known_base_form=form, allow_missing=True, name="none"
    )
    assert len(ak.Array(none.session.materialize(none.x))) == 0


class _CountingExecutor:
    def __init__(self):
        self.calls = 0

    def submit(self, fn, *args, **kwargs):
        import concurrent.futures

        self.calls += 1
        future = concurrent.futures.Future()
        future.set_result(fn(*args, **kwargs))
        return future


def test_decompression_executor_reaches_every_read():
    mark = _CountingExecutor()
    g = uproot.graphed(_zmumu(), decompression_executor=mark)
    _run(_sum_plan(g.px1, steps_per_file=2))
    assert mark.calls > 0


# ---- graphed_head ------------------------------------------------------------------------
def test_head_returns_the_first_rows_of_the_output(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    g = uproot.graphed(path + ":events")
    assert uproot.graphed_head(g.x[3:8], 3).tolist() == [3.0, 4.0, 5.0]
    assert uproot.graphed_head(g[g.x % 2 == 0], 2).tolist() == [
        {"x": 0.0, "y": 0.0},
        {"x": 2.0, "y": 4.0},
    ]
    assert len(uproot.graphed_head(g.x, 50)) == 10  # clamps to the first file
    assert uproot.graphed_head(gak.sum(g.x), 3) == 3.0  # a scalar peek over the prefix


def test_head_reads_only_the_first_file_and_the_needed_branches(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"), n=4)
    b = _make_tree(os.path.join(tmp_path, "b.root"), n=6)
    g = uproot.graphed([a + ":events", b + ":events"])
    source = _source_of(g)
    seen = []
    real_read = source.read_range
    source.read_range = lambda tree, keys, start, stop: (
        seen.append((tree.file.file_path, tuple(keys), start, stop)),
        real_read(tree, keys, start, stop),
    )[1]
    assert uproot.graphed_head(g.y * 2, 6).tolist() == [0.0, 4.0, 8.0, 12.0]
    assert seen == [(a, ("y",), 0, 4)]  # 6 asked, 4 in the first file: it stops there


@pytest.mark.filterwarnings("ignore:.*is deprecated; pass session=:DeprecationWarning")
def test_head_requires_one_uproot_source():
    from graphed.awkward import AwkwardBackend, from_awkward

    arr = from_awkward(graphed.Session(AwkwardBackend()), "mem", ak.Array({"x": [1.0]}))
    with pytest.raises(TypeError, match="exactly one uproot.graphed source"):
        uproot.graphed_head(arr.x)


# ---- the optional dependency ---------------------------------------------------------
def test_missing_graphed_reports_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_graphed(name, *args, **kwargs):
        if name == "graphed" or name.startswith("graphed."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_graphed)
    with pytest.raises(ModuleNotFoundError, match="pip install graphed"):
        uproot.extras.graphed()


def test_missing_graphed_executors_reports_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_executors(name, *args, **kwargs):
        if name.startswith("graphed_executors"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_executors)
    with pytest.raises(ModuleNotFoundError, match="pip install graphed-executors"):
        uproot.extras.graphed_executors()


def test_a_too_old_graphed_reports_the_floor(monkeypatch):
    monkeypatch.setattr(graphed, "__version__", "0.0.4")
    with pytest.raises(ModuleNotFoundError, match="needs graphed 0.0.5 or newer"):
        uproot.extras.graphed()


def test_a_recorded_op_points_at_the_analysts_line():
    g = uproot.graphed(_zmumu())
    out = g.px1 + g.py1  # <- this line
    prov = g.session.provenance(out)
    assert prov.filename == __file__ and prov.source == "g.px1 + g.py1"
