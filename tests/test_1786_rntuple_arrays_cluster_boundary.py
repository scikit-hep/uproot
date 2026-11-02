# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy
import pytest

import uproot
import uproot.models.RNTuple

ak = pytest.importorskip("awkward")


def _data(start, stop):
    return ak.Array(
        {
            "x": numpy.arange(start, stop, dtype=numpy.int64),
            "jagged": [list(range(i % 5)) for i in range(start, stop)],
            "struct": [{"a": i, "b": float(i)} for i in range(start, stop)],
        }
    )


@pytest.fixture
def ntuple(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        # two clusters, starting at entries 0 and 600
        writable = file.mkrntuple("ntuple", _data(0, 600))
        writable.extend(_data(600, 1000))
    with uproot.open(path) as file:
        yield file["ntuple"]


@pytest.fixture
def clusters_read(monkeypatch):
    read = []
    original = uproot.models.RNTuple.Model_ROOT_3a3a_RNTuple.read_cluster_range

    def spy(self, col_idx, start_cluster_idx, stop_cluster_idx, *args, **kwargs):
        read.append((int(start_cluster_idx), int(stop_cluster_idx)))
        return original(
            self, col_idx, start_cluster_idx, stop_cluster_idx, *args, **kwargs
        )

    monkeypatch.setattr(
        uproot.models.RNTuple.Model_ROOT_3a3a_RNTuple, "read_cluster_range", spy
    )
    return read


@pytest.mark.parametrize(
    ("entry_start", "entry_stop", "expected_clusters"),
    [
        (0, 600, {0}),  # entry_stop is exactly where cluster 1 starts
        (0, 599, {0}),
        (0, 601, {0, 1}),
        (100, 600, {0}),
        (600, 1000, {1}),
        (599, 601, {0, 1}),
        (0, 1000, {0, 1}),
        (0, 0, set()),
        (600, 600, set()),
        (300, 300, set()),  # an empty range inside a cluster
        (1000, 1000, set()),
    ],
)
def test_arrays_reads_only_needed_clusters(
    ntuple, clusters_read, entry_start, entry_stop, expected_clusters
):
    arrays = ntuple.arrays(entry_start=entry_start, entry_stop=entry_stop)
    assert {i for start, stop in clusters_read for i in range(start, stop)} == (
        expected_clusters
    )
    assert arrays.to_list() == _data(entry_start, entry_stop).to_list()


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [(0, 600), (600, 600), (0, 0), (300, 300), (100, 600), (599, 601)],
)
def test_arrays_virtual_at_cluster_boundary(ntuple, entry_start, entry_stop):
    arrays = ntuple.arrays(entry_start=entry_start, entry_stop=entry_stop, virtual=True)
    assert ak.materialize(arrays).to_list() == _data(entry_start, entry_stop).to_list()


def test_iterate_cluster_aligned_steps(ntuple, clusters_read):
    chunks = list(ntuple.iterate(step_size=600, filter_name="x"))
    assert [len(chunk) for chunk in chunks] == [600, 400]
    # one column, one cluster per chunk
    assert clusters_read == [(0, 1), (1, 2)]
    assert ak.concatenate(chunks).x.to_list() == list(range(1000))


@pytest.fixture
def ntuple_with_empty_cluster(tmp_path):
    path = tmp_path / "test.root"
    with uproot.recreate(path) as file:
        # three clusters, starting at entries 0, 600 and 600: the middle one is empty
        writable = file.mkrntuple("ntuple", _data(0, 600))
        writable.extend(_data(601, 602)[:0])
        writable.extend(_data(600, 1000))
    with uproot.open(path) as file:
        yield file["ntuple"]


@pytest.mark.parametrize(
    ("entry_start", "entry_stop", "expected_clusters"),
    [
        (0, 600, {0}),
        (600, 1000, {2}),
        (600, 600, set()),
        (599, 601, {0, 1, 2}),
        (0, 1000, {0, 1, 2}),
    ],
)
def test_arrays_with_empty_cluster(
    ntuple_with_empty_cluster, clusters_read, entry_start, entry_stop, expected_clusters
):
    assert [c.num_entries for c in ntuple_with_empty_cluster.cluster_summaries] == [
        600,
        0,
        400,
    ]
    arrays = ntuple_with_empty_cluster.arrays(
        entry_start=entry_start, entry_stop=entry_stop
    )
    assert all(start <= stop for start, stop in clusters_read)
    assert {i for start, stop in clusters_read for i in range(start, stop)} == (
        expected_clusters
    )
    assert arrays.to_list() == _data(entry_start, entry_stop).to_list()


def test_access_log_with_empty_cluster(ntuple_with_empty_cluster):
    log = []
    arrays = ntuple_with_empty_cluster.arrays(
        entry_start=600, entry_stop=600, virtual=True, access_log=log
    )
    assert ak.materialize(arrays).to_list() == []
    assert len(log) > 0
    assert all(access.cluster_start == access.cluster_stop for access in log)


def test_empty_cluster_range_buffers_are_zeroed(ntuple, monkeypatch):
    # fill new buffers with nonzero values, so any element left uninitialized shows up
    original_empty = numpy.empty

    def nonzero_empty(*args, **kwargs):
        out = original_empty(*args, **kwargs)
        out.fill(7)
        return out

    model = ntuple.ntuple
    monkeypatch.setattr(numpy, "empty", nonzero_empty)
    for col_idx in range(len(model.column_records)):
        for cluster_idx in (0, 1):
            buffer = model.read_cluster_range(col_idx, cluster_idx, cluster_idx)
            assert numpy.all(buffer == 0)
