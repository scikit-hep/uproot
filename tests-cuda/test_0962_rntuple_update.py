# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
)


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_new_support_RNTuple_split_int32_reading(backend, interpreter, library):
    with uproot.open(
        skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays(backend=backend, interpreter=interpreter)
        assert len(df) == 5e4
        assert len(df.one_integers) == 5e4
        assert ak.all(df.one_integers == library.arange(5e4 + 1)[::-1][:-1])


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_new_support_RNTuple_bit_bool_reading(backend, interpreter, library):
    with uproot.open(skhep_testdata.data_path("test_bit_rntuple_v1-0-0-0.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays(backend=backend, interpreter=interpreter)
        assert ak.all(df.one_bit == library.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_new_support_RNTuple_split_int16_reading(backend, interpreter, library):
    with uproot.open(
        skhep_testdata.data_path("test_int_multicluster_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays(backend=backend, interpreter=interpreter)
        assert len(df.one_integers) == 1e8
        assert df.one_integers[0] == 2
        assert df.one_integers[-1] == 1
        assert ak.all(
            library.unique(library.array(df.one_integers[: len(df.one_integers) // 2]))
            == library.array([2])
        )
        assert ak.all(
            library.unique(
                library.array(df.one_integers[len(df.one_integers) / 2 + 1 :])
            )
            == library.array([1])
        )
