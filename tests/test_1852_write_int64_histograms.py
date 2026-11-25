# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot

hist = pytest.importorskip("hist")


def roundtrip(tmp_path, h):
    with uproot.recreate(tmp_path / "out.root") as fout:
        fout["h"] = h
    with uproot.open(tmp_path / "out.root") as fin:
        return fin["h"]


@pytest.mark.parametrize("storage", ["Int64", "AtomicInt64"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_write_integer_storage(tmp_path, storage, ndim):
    axes = [hist.axis.Regular(4, 0, 4, name=name) for name in "xyz"[:ndim]]
    h = hist.Hist(*axes, storage=getattr(hist.storage, storage)())
    h.fill(*([[0, 1, 1, 3]] * ndim))

    h_read = roundtrip(tmp_path, h)
    assert h_read.classname == f"TH{ndim}D"
    assert np.array_equal(h_read.values(), h.values())


class _Axis:
    def __init__(self, edges):
        self.edges = np.asarray(edges, dtype=np.float64)

    def __len__(self):
        return len(self.edges) - 1


class _Histogram:
    kind = "COUNT"

    def __init__(self, values):
        self._values = values
        self.axes = (_Axis(np.arange(len(values) + 1)),)

    def values(self):
        return self._values

    def variances(self):
        return None

    def counts(self):
        return self._values


@pytest.mark.parametrize(
    ("dtype", "classname"),
    [
        (np.uint8, "TH1D"),
        (np.uint64, "TH1D"),
        (np.int64, "TH1D"),
        (np.int8, "TH1C"),
        (np.int16, "TH1S"),
        (np.int32, "TH1I"),
        (np.float32, "TH1F"),
        (np.float64, "TH1D"),
    ],
)
def test_write_plottable_histogram_dtypes(tmp_path, dtype, classname):
    h = _Histogram(np.array([1, 2, 3], dtype=dtype))
    h_read = roundtrip(tmp_path, h)
    assert h_read.classname == classname
    assert np.array_equal(h_read.values(), [1, 2, 3])
