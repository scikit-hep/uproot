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
def test_atomic(backend, interpreter, library):
    filename = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("atomic_int", backend=backend, interpreter=interpreter)

        assert ak.all(a.atomic_int == library.array([1, 2, 3]))


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_bitset(backend, interpreter, library):
    filename = skhep_testdata.data_path("test_atomic_bitset_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("bitset", backend=backend, interpreter=interpreter)

        assert len(a.bitset) == 3
        assert len(a.bitset[0]) == 42
        assert ak.all(a.bitset[0][:6] == library.array([0, 1, 0, 1, 0, 1]))
        assert ak.all(a.bitset[0][6:] == 0)
        assert ak.all(
            a.bitset[1][:16]
            == library.array(
                [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ]
            )
        )
        assert ak.all(a.bitset[1][16:] == 0)
        assert ak.all(
            a.bitset[2][:16]
            == library.array(
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                ]
            )
        )
        assert ak.all(a.bitset[2][16:] == 0)


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_empty_struct(backend, interpreter, library):
    filename = skhep_testdata.data_path(
        "test_emptystruct_invalidvar_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("empty_struct", backend=backend, interpreter=interpreter)

        assert a.empty_struct.tolist() == [(), (), ()]


# cupy doesn't support None or object dtype like numpy; test cannot pass with GDS
def test_invalid_variant():
    filename = skhep_testdata.data_path(
        "test_emptystruct_invalidvar_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        a = obj.arrays("variant", backend="cuda", interpreter="cpu")

        assert a.variant.tolist() == [1, None, {"i": 2}]
