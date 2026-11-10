# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import numpy
import pytest

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
            "y": numpy.arange(start, stop, dtype=numpy.float64) * 0.5,
        }
    )


@pytest.fixture
def paths(tmp_path):
    """Two RNTuple files, each with two clusters, in a GPU-readable compression."""
    paths = []
    for i in range(2):
        path = tmp_path / f"test{i}.root"
        with uproot.recreate(path, compression=uproot.ZSTD(1)) as file:
            writable = file.mkrntuple("ntuple", _data(0, 250))
            writable.extend(_data(250, 500))
        paths.append(path)
    return paths


def test_iterate_multi_file_gpu(paths):
    """``uproot.iterate`` passes ``backend``/``interpreter`` through to every RNTuple."""
    chunks = list(
        uproot.iterate(
            [f"{path}:ntuple" for path in paths],
            step_size=200,
            backend="cuda",
            interpreter="gpu",
        )
    )
    assert [len(chunk) for chunk in chunks] == [200, 200, 100, 200, 200, 100]
    expected = ak.concatenate([_data(0, 500), _data(0, 500)])
    assert ak.to_backend(ak.concatenate(chunks), "cpu").to_list() == expected.to_list()


def test_concatenate_multi_file_gpu(paths):
    """``uproot.concatenate`` passes ``backend``/``interpreter`` through to every RNTuple."""
    array = uproot.concatenate(
        [f"{path}:ntuple" for path in paths], backend="cuda", interpreter="gpu"
    )
    expected = ak.concatenate([_data(0, 500), _data(0, 500)])
    assert ak.to_backend(array, "cpu").to_list() == expected.to_list()
