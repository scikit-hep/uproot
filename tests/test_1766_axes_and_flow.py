# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
from uproot.writing.identify import (
    to_TAxis,
    to_TH1x,
    to_THashList,
    to_TObjString,
    to_TProfile,
)

hist = pytest.importorskip("hist")

ABC = [(1, "a"), (2, "b"), (3, "c")]


def _axis(nbins, labels):
    # labels: list of (bin number, label) in the order they were set,
    # as ROOT's TAxis::SetBinLabel stores them
    fLabels = to_THashList([to_TObjString(label) for _, label in labels])
    for (bin_number, _), label in zip(labels, fLabels):
        label._bases[0]._members["@fUniqueID"] = bin_number
    return to_TAxis(
        fName="xaxis", fTitle="", fNbins=nbins, fXmin=0, fXmax=nbins, fLabels=fLabels
    )


def _th1(nbins, labels, data):
    return to_TH1x(
        fName="h",
        fTitle="",
        data=np.array(data, np.float64),
        fEntries=2.0,
        fTsumw=2.0,
        fTsumw2=2.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=np.array(data, np.float64) ** 2,
        fXaxis=_axis(nbins, labels),
    )


def _tprofile(nbins, labels, data):
    data = np.array(data, np.float64)
    entries = (data != 0).astype(np.float64)
    return to_TProfile(
        fName="p",
        fTitle="",
        data=data,
        fEntries=2.0,
        fTsumw=2.0,
        fTsumw2=2.0,
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fTsumwy=data.sum(),
        fTsumwy2=(data**2).sum(),
        fSumw2=data**2,
        fBinEntries=entries,
        fBinSumw2=np.array([], np.float64),
        fXaxis=_axis(nbins, labels),
    )


def _write_and_read(tmp_path, obj):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = obj
    return uproot.open(filepath)["h"]


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_out_of_order(tmp_path, make):
    h = _write_and_read(
        tmp_path, make(3, [(3, "c"), (1, "a"), (2, "b")], [0, 10, 0, 30, 0])
    )
    axis = h.member("fXaxis")
    assert axis.labels() == ["a", "b", "c"]
    assert list(axis) == ["a", "b", "c"]
    assert axis[2] == "c"
    assert axis.traits.discrete

    # the labels are cached, but callers get a copy
    axis.labels().append("d")
    assert axis.labels() == ["a", "b", "c"]

    out = h.to_hist()
    assert list(out.axes[0]) == ["a", "b", "c"]
    assert out.values().tolist() == [10, 0, 30]
    assert out[{0: "c"}].value == 30


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_partial(tmp_path, make):
    h = _write_and_read(
        tmp_path, make(6, [(1, "a"), (2, "b")], [0, 10, 0, 0, 0, 50, 0, 0])
    )
    axis = h.member("fXaxis")
    assert axis.labels() is None
    assert not axis.traits.discrete

    with pytest.warns(UserWarning, match="labels for only 2 of its 6 bins") as record:
        out = h.to_hist()
    assert record[0].filename == __file__  # the user's code, not Uproot's
    assert out.axes[0].edges.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert out.values().tolist() == [10, 0, 0, 0, 50, 0]


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_same_bin(tmp_path, make):
    # every label has a valid bin number, but bin 2 has none: the labels can't
    # be assigned in the order they were set
    h = _write_and_read(
        tmp_path, make(3, [(1, "x"), (1, "y"), (3, "z")], [0, 10, 20, 30, 0])
    )
    axis = h.member("fXaxis")
    assert axis.labels() is None
    assert not axis.traits.discrete

    with pytest.warns(UserWarning, match="labels for only 2 of its 3 bins"):
        out = h.to_hist()
    assert out.values().tolist() == [10, 20, 30]


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_first_for_bin(tmp_path, make):
    # as in ROOT's TAxis::GetBinLabel, the first label with a bin's number wins
    h = _write_and_read(tmp_path, make(2, [(1, "a"), (2, "b"), (1, "c")], [0] * 4))
    assert h.member("fXaxis").labels() == ["a", "b"]


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_some_without_bin_numbers(tmp_path, make):
    # the label without a bin number is ignored, as in ROOT, rather than all of
    # them being taken in the order they were set
    h = _write_and_read(
        tmp_path, make(3, [(3, "c"), (1, "a"), (0, "x")], [0, 10, 20, 30, 0])
    )
    assert h.member("fXaxis").labels() is None
    with pytest.warns(UserWarning, match="labels for only 2 of its 3 bins"):
        out = h.to_hist()
    assert out.values().tolist() == [10, 20, 30]


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labels_without_bin_numbers(tmp_path, make):
    # one label for each bin: assume they were set in bin order
    h = _write_and_read(
        tmp_path, make(3, [(0, "a"), (0, "b"), (0, "c")], [0, 10, 20, 30, 0])
    )
    assert h.member("fXaxis").labels() == ["a", "b", "c"]

    # fewer labels than bins: not a labeled axis
    h = _write_and_read(tmp_path, make(3, [(0, "a"), (0, "b")], [0, 10, 20, 30, 0]))
    assert h.member("fXaxis").labels() is None
    with pytest.warns(UserWarning, match="labels for only 2 of its 3 bins"):
        out = h.to_hist()
    assert out.values().tolist() == [10, 20, 30]


@pytest.mark.parametrize("make", [_th1, _tprofile])
def test_labeled_underflow(tmp_path, make):
    # no warning when it's empty
    _write_and_read(tmp_path, make(3, ABC, [0, 1, 0, 0, 0])).to_hist()

    h = _write_and_read(tmp_path, make(3, ABC, [1, 1, 0, 0, 0]))
    with pytest.warns(UserWarning, match="underflow bin of labeled axis") as record:
        out = h.to_hist()
    assert record[0].filename == __file__
    assert out.values(flow=True).tolist() == [1, 0, 0, 0]
    if make is _tprofile:
        # ROOT's number of fills is kept, including the dropped underflow's
        assert _write_and_read(tmp_path, out).member("fEntries") == 2


def test_labels_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filepath = os.path.join(tmp_path, "test.root")
    f = ROOT.TFile(filepath, "RECREATE")
    h = ROOT.TH2D("h2_1766_labels", "", 2, 0, 2, 3, 0, 3)
    h.SetDirectory(f)
    for bin_number, label in [(2, "y"), (1, "x")]:
        h.GetXaxis().SetBinLabel(bin_number, label)
    for bin_number, label in [(3, "w"), (2, "v"), (1, "u")]:
        h.GetYaxis().SetBinLabel(bin_number, label)
    h.Fill(1.5, 2.5, 7)
    # filling a 3-bin axis with a 4th label doubles its bins, leaving 2 unlabeled
    p = ROOT.TProfile("p_1766_labels", "", 3, 0, 3)
    p.SetDirectory(f)
    for label, y in [("c", 1), ("a", 2), ("b", 3), ("d", 4)]:
        p.Fill(label, y)
    expected = [p.GetBinContent(i) for i in range(1, p.GetNbinsX() + 1)]
    f.Write()
    f.Close()

    with uproot.open(filepath) as f:
        h = f["h2_1766_labels"].to_hist()
        with pytest.warns(UserWarning, match="labels for only 4 of its 6 bins"):
            p = f["p_1766_labels"].to_hist()

    assert list(h.axes[0]) == ["x", "y"]
    assert list(h.axes[1]) == ["u", "v", "w"]
    assert h[{0: "y", 1: "w"}].value == 7
    assert h.values().sum() == 7

    assert p.axes[0].edges.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert p.values().tolist() == expected == [1, 2, 3, 4, 0, 0]


@pytest.mark.parametrize(
    ("axis", "fills", "expected"),
    [
        (hist.axis.Integer(0, 3, underflow=False), [0, 1, 5, 5], [0, 1, 1, 0, 2]),
        (hist.axis.Integer(0, 3, overflow=False), [-1, 0, 1], [1, 1, 1, 0, 0]),
        (hist.axis.StrCategory(["a", "b"]), ["a", "z", "z"], [0, 1, 0, 2]),
        (hist.axis.IntCategory([1, 2], overflow=False), [1, 1, 2], [0, 2, 1, 0]),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_partial_flow_bins_1d(tmp_path, axis, fills, expected, weighted):
    storage = hist.storage.Weight() if weighted else hist.storage.Double()
    h = hist.Hist(axis, storage=storage)
    h.fill(fills)
    out = _write_and_read(tmp_path, h)
    assert out.values(flow=True).tolist() == expected
    if weighted:
        assert out.variances(flow=True).tolist() == expected


def test_partial_flow_bins_2d(tmp_path):
    h = hist.Hist(
        hist.axis.Integer(0, 2, underflow=False),
        hist.axis.StrCategory(["a", "b"]),
    )
    h.fill([0, 5, 5], ["a", "b", "z"])
    out = _write_and_read(tmp_path, h)
    expected = np.zeros((4, 4))
    expected[1, 1] = 1  # x=0, y="a"
    expected[3, 2] = 1  # x overflow, y="b"
    expected[3, 3] = 1  # x overflow, y overflow
    assert out.values(flow=True).tolist() == expected.tolist()


def test_warning_location():
    # warnings from Uproot, including from its generated models (compiled as
    # "<dynamic>"), are attributed to the first caller outside of it
    code = compile("uproot._util.warn_outside_uproot('message')", "<dynamic>", "exec")
    with pytest.warns(UserWarning, match="message") as record:
        exec(code, {"uproot": uproot})
    assert record[0].filename == __file__
