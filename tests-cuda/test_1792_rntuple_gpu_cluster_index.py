# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
)


def _data(start, stop):
    return ak.Array(
        {
            "x": numpy.arange(start, stop, dtype=numpy.int64),
            "jagged": [list(range(i % 5)) for i in range(start, stop)],
        }
    )


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "test.root"
    # the GPU interpreter can decompress ZSTD and LZ4, but not ZLIB
    with uproot.recreate(path, compression=uproot.ZSTD(1)) as file:
        # three clusters, starting at entries 0, 300 and 600
        writable = file.mkrntuple("ntuple", _data(0, 300))
        writable.extend(_data(300, 600))
        writable.extend(_data(600, 1000))
    return path


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [(300, 600), (600, 1000), (300, 1000), (450, 700), (999, 1000)],
)
def test_arrays_gpu_not_from_first_cluster(path, entry_start, entry_stop):
    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays(
            entry_start=entry_start,
            entry_stop=entry_stop,
            backend="cuda",
            interpreter="gpu",
        )
    assert (
        ak.to_backend(arrays, "cpu").to_list()
        == _data(entry_start, entry_stop).to_list()
    )


def test_iterate_gpu(path):
    with uproot.open(path) as file:
        chunks = list(
            file["ntuple"].iterate(step_size=300, backend="cuda", interpreter="gpu")
        )
    assert [len(chunk) for chunk in chunks] == [300, 300, 300, 100]
    assert ak.to_backend(ak.concatenate(chunks), "cpu").to_list() == (
        _data(0, 1000).to_list()
    )


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    # clusters start at 0, 350, 467 and 551; float_field starts at entry 200
    # and intvec_field at entry 400, so the ranges that start in cluster 1 pad
    # intvec_field and those that start later pad nothing
    [(0, 600), (350, 600), (360, 480), (400, 467), (470, 600), (551, 600)],
)
def test_arrays_gpu_late_columns_match_cpu(entry_start, entry_stop):
    filename = skhep_testdata.data_path("test_extension_columns_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as file:
        ntuple = file["ntuple"]
        expected = ntuple.arrays(
            entry_start=entry_start, entry_stop=entry_stop, interpreter="cpu"
        )
        arrays = ntuple.arrays(
            entry_start=entry_start,
            entry_stop=entry_stop,
            backend="cuda",
            interpreter="gpu",
        )
    assert ak.to_backend(arrays, "cpu").to_list() == expected.to_list()
