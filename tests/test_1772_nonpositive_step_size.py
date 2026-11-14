# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
import skhep_testdata

import uproot

BAD_STEP_SIZES = [0, -1, -100, np.int64(-1), "0 MB", "-1 MB"]


@pytest.fixture
def hzz_path():
    return skhep_testdata.data_path("uproot-HZZ.root")


@pytest.fixture
def ttree(hzz_path):
    with uproot.open(hzz_path) as file:
        yield file["events"]


@pytest.fixture
def rntuple_path(tmp_path):
    filename = tmp_path / "ntuple.root"
    with uproot.recreate(filename) as file:
        file["ntuple"] = {"x": np.arange(10, dtype=np.float64)}
    return str(filename)


@pytest.fixture
def rntuple(rntuple_path):
    with uproot.open(rntuple_path) as file:
        yield file["ntuple"]


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_ttree_iterate(ttree, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(ttree.iterate(["NJet"], step_size=step_size))


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_ttree_iterate_with_empty_selection(ttree, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(ttree.iterate(filter_name="nonexistent*", step_size=step_size))


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_rntuple_iterate(rntuple, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(rntuple.iterate(step_size=step_size))


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_uproot_iterate_ttree(hzz_path, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(uproot.iterate({hzz_path: "events"}, ["NJet"], step_size=step_size))


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_uproot_iterate_rntuple(rntuple_path, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(uproot.iterate({rntuple_path: "ntuple"}, step_size=step_size))


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_uproot_iterate_raises_before_opening_files(tmp_path, step_size):
    missing = str(tmp_path / "does-not-exist.root")
    with pytest.raises(ValueError, match="step_size"):
        next(uproot.iterate({missing: "events"}, step_size=step_size))


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_uproot_dask(hzz_path, step_size):
    pytest.importorskip("dask_awkward")
    with pytest.raises(ValueError, match="step_size"):
        uproot.dask({hzz_path: "events"}, step_size=step_size)


@pytest.mark.parametrize("step_size", BAD_STEP_SIZES)
def test_graphed_partitions(hzz_path, step_size):
    pytest.importorskip("graphed.awkward")
    from uproot._graphed import graphed_partitions

    with pytest.raises(ValueError, match="step_size"):
        graphed_partitions({hzz_path: "events"}, step_size=step_size)


def test_graphed_partitions_numpy_integer_is_entries(hzz_path):
    pytest.importorskip("graphed.awkward")
    from uproot._graphed import graphed_partitions

    parts = graphed_partitions({hzz_path: "events"}, step_size=np.int64(1000))
    assert [(p.entry_start, p.entry_stop) for p in parts] == [
        (0, 1000),
        (1000, 2000),
        (2000, 2421),
    ]


def test_valid_step_sizes_still_work(ttree, rntuple):
    assert sum(len(a) for a in ttree.iterate(["NJet"], step_size=1000)) == 2421
    assert [len(a) for a in rntuple.iterate(step_size=3)] == [3, 3, 3, 1]
    # a positive memory size smaller than one entry still gives one entry per step
    assert [len(a) for a in rntuple.iterate(step_size="1 B")] == [1] * 10
