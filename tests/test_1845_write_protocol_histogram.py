# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot


class _Traits:
    def __init__(self, discrete):
        self.circular = False
        self.discrete = discrete


class ProtocolAxis:
    """Implements the PlottableAxis protocol without an ``edges`` attribute."""

    def __init__(self, edges=None, labels=None, pair=tuple):
        self._edges = None if edges is None else np.asarray(edges, dtype=np.float64)
        self._labels = labels
        self._pair = pair
        self.traits = _Traits(discrete=labels is not None)

    def __len__(self):
        return len(self._labels) if self._labels is not None else len(self._edges) - 1

    def __getitem__(self, index):
        if self._labels is not None:
            return self._labels[index]
        return self._pair((self._edges[index], self._edges[index + 1]))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __eq__(self, other):
        return list(self) == list(other)


class ProtocolHistogram:
    """Implements the PlottableHistogram protocol with ``ProtocolAxis`` axes."""

    def __init__(self, *axes, kind="COUNT"):
        self.axes = axes
        self.kind = kind
        shape = tuple(len(axis) for axis in axes)
        self._values = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)

    def values(self):
        return self._values

    def variances(self):
        return self._values

    def counts(self):
        return self._values


def roundtrip(tmp_path, h):
    with uproot.recreate(tmp_path / "protocol.root") as fout:
        fout["h"] = h
    with uproot.open(tmp_path / "protocol.root") as fin:
        return fin["h"]


@pytest.mark.parametrize(
    "edges",
    [
        [[0, 1, 2, 3]],
        [[0, 1, 5, 10]],
        [[0, 1, 2], [0, 2, 4, 8]],
        [[0, 1, 2], [0, 2, 4], [0, 3, 6, 9]],
    ],
    ids=["1d-regular", "1d-variable", "2d", "3d"],
)
def test_write_protocol_histogram(tmp_path, edges):
    h = ProtocolHistogram(*(ProtocolAxis(edges=e) for e in edges))
    h_read = roundtrip(tmp_path, h)
    for axis_read, axis_edges in zip(h_read.axes, edges, strict=True):
        assert np.array_equal(axis_read.edges(), axis_edges)
    assert np.array_equal(h_read.values(), h.values())
    assert np.array_equal(h_read.variances(), h.variances())


@pytest.mark.parametrize(
    "pair", [tuple, list, np.array], ids=["tuple", "list", "array"]
)
def test_write_protocol_histogram_pair_types(tmp_path, pair):
    h = ProtocolHistogram(ProtocolAxis(edges=[0, 1, 5, 10], pair=pair))
    h_read = roundtrip(tmp_path, h)
    assert np.array_equal(h_read.axes[0].edges(), [0, 1, 5, 10])
    assert h_read.member("fTsumwx") == pytest.approx(0.5 * 1 + 3 * 2 + 7.5 * 3)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [(["a", "b", "c"], ["a", "b", "c"]), ([1, 5, 10], ["1", "5", "10"])],
    ids=["str", "int"],
)
def test_write_protocol_histogram_discrete_axis(tmp_path, labels, expected):
    h = ProtocolHistogram(ProtocolAxis(labels=labels))
    h_read = roundtrip(tmp_path, h)
    assert np.array_equal(h_read.axes[0].edges(), [0, 1, 2, 3])
    assert h_read.axes[0].labels() == expected
    assert np.array_equal(h_read.values(), h.values())


def test_write_uhi_numpy_plottable_histogram(tmp_path):
    numpy_plottable = pytest.importorskip("uhi.numpy_plottable")
    h = numpy_plottable.NumPyPlottableHistogram(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 5.0, 10.0])
    )
    h_read = roundtrip(tmp_path, h)
    assert np.array_equal(h_read.axes[0].edges(), [0, 1, 5, 10])
    assert np.array_equal(h_read.values(), [1, 2, 3])


def test_write_protocol_profile_not_supported(tmp_path):
    h = ProtocolHistogram(ProtocolAxis(edges=[0, 1, 2, 3]), kind="MEAN")
    with pytest.raises(NotImplementedError, match=r"kind='MEAN'"):
        with uproot.recreate(tmp_path / "protocol.root") as fout:
            fout["h"] = h


@pytest.mark.parametrize(
    "edges",
    [[[0, 1, 2, 3]], [[0, 1, 5, 10]], [[0, 1, 2], [0, 2, 4, 8]]],
    ids=["1d-regular", "1d-variable", "2d"],
)
def test_protocol_histogram_matches_hist(tmp_path, edges):
    hist = pytest.importorskip("hist")

    h_protocol = ProtocolHistogram(*(ProtocolAxis(edges=e) for e in edges))
    h_hist = hist.Hist(*(hist.axis.Variable(e, flow=False) for e in edges))
    h_hist[...] = h_protocol.values()

    with uproot.recreate(tmp_path / "protocol.root") as fout:
        fout["h"] = h_protocol
    with uproot.recreate(tmp_path / "hist.root") as fout:
        fout["h"] = h_hist

    with uproot.open(tmp_path / "protocol.root") as fin:
        from_protocol = fin["h"]
        with uproot.open(tmp_path / "hist.root") as fin_hist:
            from_hist = fin_hist["h"]
            for axis_protocol, axis_hist in zip(
                from_protocol.axes, from_hist.axes, strict=True
            ):
                # evenly spaced edges are written without fXbins for a
                # PlottableHistogram, but for a hist Variable axis with them, so
                # compare the edges they describe
                assert np.array_equal(axis_protocol.edges(), axis_hist.edges())
            assert np.array_equal(
                from_protocol.values(flow=True), from_hist.values(flow=True)
            )
            for name in ["fEntries", "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"]:
                assert from_protocol.member(name) == pytest.approx(
                    from_hist.member(name)
                )
