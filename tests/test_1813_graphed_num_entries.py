# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``_GraphedTTreeSource.num_entries`` and the files ``uproot.graphed`` opens outside a worker."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402


def _make_tree(path, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8"})
        f[name].extend({"x": np.arange(10.0)})
    return path


def _source_of(array):
    return uproot._graphed._uproot_source(array, "test")[1]


@pytest.fixture
def opened(monkeypatch):
    """Every ``ReadOnlyFile`` opened, held so nothing closes by collection."""
    files = []
    real = uproot.reading.ReadOnlyFile.__init__

    def recording(self, *args, **kwargs):
        files.append(self)
        real(self, *args, **kwargs)

    monkeypatch.setattr(uproot.reading.ReadOnlyFile, "__init__", recording)
    return files


def test_num_entries_counts_the_tree_a_blind_chunk_reads(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    source = _source_of(uproot.graphed(path + ":events"))
    assert [source.num_entries(p) for p in source.partitions(3)] == [10, 10, 10]


def test_num_entries_closes_the_file_and_a_missing_tree_is_zero(tmp_path, opened):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"), name="other")
    source = _source_of(
        uproot.graphed([a + ":events", b + ":events"], allow_missing=True)
    )
    opened.clear()
    assert [source.num_entries(p) for p in source.partitions(1)] == [10, 0]
    assert [f.closed for f in opened] == [True, True]


def test_num_entries_closes_the_file_when_the_tree_is_gone(tmp_path, opened):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    source = _source_of(uproot.graphed(path + ":events"))
    _make_tree(path, name="other")
    opened.clear()
    with pytest.raises(uproot.KeyInFileError):
        source.num_entries(source.partitions(1)[0])
    assert [f.closed for f in opened] == [True]


def test_num_entries_is_none_where_baskets_move_a_chunk(tmp_path):
    path = os.path.join(tmp_path, "t.root")
    with uproot.recreate(path) as f:
        f.mktree("events", {"x": "i8"})
        f["events"].extend({"x": np.arange(4)})
        f["events"].extend({"x": np.arange(4, 10)})
    source = _source_of(uproot.graphed(path + ":events", align_baskets=True))
    partitions = source.partitions(3)
    with uproot.open(path + ":events") as tree:
        ranges = [source.entry_range(p, tree, ["x"]) for p in partitions]
    assert ranges == [(0, 4), (4, 4), (4, 10)]
    assert [p.resolve(10).entry_start for p in partitions] == [0, 3, 6]
    assert [source.num_entries(p) for p in partitions] == [None, None, None]


def test_whole_dataset_loader_closes_every_file(tmp_path, opened):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = _make_tree(os.path.join(tmp_path, "b.root"))
    source = _source_of(uproot.graphed([a + ":events", b + ":events"]))
    opened.clear()
    assert ak.sum(source().x) == 90.0
    assert [f.closed for f in opened] == [True, True]


def test_graphed_head_reads_its_tree_then_closes_the_file(tmp_path, opened):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    array = uproot.graphed(path + ":events")
    opened.clear()
    assert ak.to_list(uproot._graphed.graphed_head(array.x, 3)) == [0.0, 1.0, 2.0]
    assert [f.closed for f in opened] == [True]


def test_graphed_head_closes_the_file_while_its_failure_is_held(
    tmp_path, opened, monkeypatch
):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    array = uproot.graphed(path + ":events")
    opened.clear()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(uproot.extras.graphed(), "evaluate_ir", boom)
    with pytest.raises(RuntimeError, match="boom") as failure:
        uproot._graphed.graphed_head(array.x, 3)
    assert [f.closed for f in opened] == [True]


def test_whole_dataset_loader_closes_the_file_while_its_failure_is_held(
    tmp_path, opened, monkeypatch
):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    source = _source_of(uproot.graphed(path + ":events"))
    opened.clear()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(source, "read_range", boom)
    with pytest.raises(RuntimeError, match="boom") as failure:
        source()
    assert [f.closed for f in opened] == [True]


def test_graphed_partitions_reads_each_tree_then_closes_its_file(tmp_path, opened):
    paths = [_make_tree(os.path.join(tmp_path, f"{n}.root")) for n in "ab"]
    parts = uproot._graphed.graphed_partitions(
        [path + ":events" for path in paths], step_size="1 kB", align_baskets=True
    )
    assert [(p.entry_start, p.entry_stop) for p in parts] == [(0, 10), (0, 10)]
    assert [f.closed for f in opened] == [True, True]


def test_graphed_partitions_leaves_a_given_tree_open(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    with uproot.open(path + ":events") as tree:
        uproot._graphed.graphed_partitions(tree, steps_per_file=2)
        assert not tree.file.closed


@pytest.mark.parametrize("library", ["ak", "np"])
@pytest.mark.parametrize(
    "handler", [uproot.source.file.MemmapSource, uproot.source.fsspec.FSSpecSource]
)
def test_read_graphed_partition_closes_the_file_it_opens(
    tmp_path, opened, library, handler
):
    path = os.path.join(tmp_path, "t.root")
    with uproot.recreate(path, compression=None) as f:
        f.mktree("events", {"x": "f8"})
        f["events"].extend({"x": np.arange(10.0)})
    opened.clear()
    partition = graphed.core.Partition.blind(path, "events", 1, 2)
    chunk = uproot._graphed.read_graphed_partition(
        partition, ["x"], library=library, handler=handler
    )
    assert [f.closed for f in opened] == [True]
    assert ak.to_list(chunk["x"]) == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_read_graphed_partition_leaves_a_given_tree_open(tmp_path):
    path = _make_tree(os.path.join(tmp_path, "t.root"))
    partition = graphed.core.Partition.blind(path, "events", 0, 2)
    with uproot.open(path + ":events") as tree:
        chunk = uproot._graphed.read_graphed_partition(partition, ["x"], tree=tree)
        assert not tree.file.closed
    assert ak.to_list(chunk.x) == [0.0, 1.0, 2.0, 3.0, 4.0]
