# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy as np
import pytest

import uproot


class _Traits:
    def __init__(self, discrete):
        self.circular = False
        self.discrete = discrete


class ProtocolAxis:
    """Implements the PlottableAxis protocol without an ``edges`` attribute."""

    def __init__(self, edges=None, labels=None):
        self._edges = None if edges is None else np.asarray(edges, dtype=np.float64)
        self._labels = labels
        self.traits = _Traits(discrete=labels is not None)

    def __len__(self):
        return len(self._labels) if self._labels is not None else len(self._edges) - 1

    def __getitem__(self, index):
        if self._labels is not None:
            return self._labels[index]
        return (self._edges[index], self._edges[index + 1])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __eq__(self, other):
        return list(self) == list(other)


class ProtocolHistogram:
    """Implements the PlottableHistogram protocol with ``ProtocolAxis`` axes."""

    kind = "COUNT"

    def __init__(self, *axes):
        self.axes = axes
        shape = tuple(len(axis) for axis in axes)
        self._values = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)

    def values(self):
        return self._values

    def variances(self):
        return self._values

    def counts(self):
        return self._values


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
    filename = tmp_path / "protocol.root"

    with uproot.recreate(filename) as fout:
        fout["h"] = h

    with uproot.open(filename) as fin:
        h_read = fin["h"]
        for axis_read, axis_edges in zip(h_read.axes, edges, strict=True):
            assert np.array_equal(axis_read.edges(), axis_edges)
        assert np.array_equal(h_read.values(), h.values())
        assert np.array_equal(h_read.variances(), h.variances())


def test_write_protocol_histogram_discrete_axis(tmp_path):
    h = ProtocolHistogram(ProtocolAxis(labels=["a", "b", "c"]))
    filename = tmp_path / "protocol.root"

    with uproot.recreate(filename) as fout:
        fout["h"] = h

    with uproot.open(filename) as fin:
        h_read = fin["h"]
        assert np.array_equal(h_read.axes[0].edges(), [0, 1, 2, 3])
        assert np.array_equal(h_read.values(), h.values())


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
