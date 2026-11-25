# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import warnings

import numpy as np
import pytest

import uproot
from uproot.writing.identify import to_TAxis, to_TProfile

hist = pytest.importorskip("hist")

# (axis, has underflow, has overflow, bin index -> fill value)
AXES = {
    "Regular": (hist.axis.Regular(4, 0, 4), True, True, lambda i: i + 0.5),
    "Variable": (hist.axis.Variable([0, 1, 3, 6]), True, True, None),
    "Integer": (hist.axis.Integer(0, 4), True, True, lambda i: i),
    "Regular-noflow": (
        hist.axis.Regular(4, 0, 4, underflow=False, overflow=False),
        False,
        False,
        lambda i: i + 0.5,
    ),
    "Regular-nounderflow": (
        hist.axis.Regular(4, 0, 4, underflow=False),
        False,
        True,
        lambda i: i + 0.5,
    ),
    "IntCategory": (
        hist.axis.IntCategory([10, 20, 30]),
        False,
        True,
        lambda i: [10, 20, 30, 99][i],
    ),
    "IntCategory-noflow": (
        hist.axis.IntCategory([10, 20, 30], overflow=False),
        False,
        False,
        lambda i: [10, 20, 30][i],
    ),
    "StrCategory": (
        hist.axis.StrCategory(["a", "b", "c"]),
        False,
        True,
        lambda i: ["a", "b", "c", "zz"][i],
    ),
    "Boolean": (hist.axis.Boolean(), False, False, bool),
}


def _fill(name, storage, seed=0):
    axis, underflow, overflow, to_value = AXES[name]
    rng = np.random.default_rng(seed)
    n = len(axis)
    edges = axis.edges
    index = rng.integers(-1 if underflow else 0, (n if overflow else n - 1) + 1, 200)
    y = rng.normal(5, 2, len(index))
    if storage == "WeightedMean":
        w = rng.uniform(-0.5, 2, len(index))
    else:
        w = np.ones(len(index))

    # ROOT x-coordinates: bin centers, with flow bins just outside the edges
    centers = np.concatenate(
        [[edges[0] - 0.5], (edges[:-1] + edges[1:]) / 2, [edges[-1] + 0.5]]
    )
    x = centers[index + 1]

    h = hist.Hist(axis, storage=getattr(hist.storage, storage)())
    values = x if to_value is None else [to_value(i) for i in index]
    if storage == "WeightedMean":
        h.fill(values, sample=y, weight=w)
    else:
        h.fill(values, sample=y)
    return h, x, y, w, index


def _write_and_read(tmp_path, obj, name="p"):
    filepath = os.path.join(tmp_path, f"{name}.root")
    with uproot.recreate(filepath) as f:
        f[name] = obj
    return uproot.open(filepath)[name]


def _root_write_and_read(ROOT, tmp_path, obj):
    filepath = os.path.join(tmp_path, f"{obj.GetName()}.root")
    f = ROOT.TFile(filepath, "RECREATE")
    obj.Write()
    f.Close()
    return uproot.open(filepath)[obj.GetName()]


def _root_read(ROOT, tmp_path, obj, name):
    filepath = os.path.join(tmp_path, "uproot-written.root")
    with uproot.recreate(filepath) as f:
        f[name] = obj
    f = ROOT.TFile(filepath)
    out = f.Get(name)
    out.SetDirectory(ROOT.nullptr)
    f.Close()
    return out


def _weighted_profile(fSumw2, fErrorMode=0, fTsumw=1.0, fTsumwx=0.5):
    return to_TProfile(
        fName="p",
        fTitle="",
        data=np.array([0.0, 3.0, 0.0], np.float64),
        fEntries=3.0,
        fTsumw=fTsumw,
        fTsumw2=5.0,
        fTsumwx=fTsumwx,
        fTsumwx2=0.25,
        fTsumwy=3.0,
        fTsumwy2=fSumw2,
        fSumw2=np.array([0.0, fSumw2, 0.0], np.float64),
        fBinEntries=np.array([0.0, 1.0, 0.0], np.float64),
        fBinSumw2=np.array([0.0, 5.0, 0.0], np.float64),
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=1, fXmin=0, fXmax=1),
        fErrorMode=fErrorMode,
    )


@pytest.mark.parametrize("storage", ["Mean", "WeightedMean"])
@pytest.mark.parametrize("name", list(AXES))
def test_axes(tmp_path, name, storage):
    h, x, y, w, index = _fill(name, storage)
    n = len(h.axes[0])

    p = _write_and_read(tmp_path, h)
    expected = np.zeros(n + 2)
    np.add.at(expected, index + 1, w)
    assert np.asarray(p.member("fBinEntries")) == pytest.approx(expected)

    if storage == "Mean":
        assert len(p.member("fBinSumw2")) == 0
        assert p.member("fEntries") == len(index)
    else:
        assert len(p.member("fBinSumw2")) == n + 2
        assert p.member("fEntries") == pytest.approx(w.sum())

    inside = (index >= 0) & (index < n)
    assert p.member("fTsumw") == pytest.approx(w[inside].sum())
    assert p.member("fTsumwx") == pytest.approx((w * x)[inside].sum())
    assert p.member("fTsumwx2") == pytest.approx((w * x**2)[inside].sum())
    assert p.member("fTsumwy") == pytest.approx((w * y)[inside].sum())
    assert p.member("fTsumwy2") == pytest.approx((w * y**2)[inside].sum())

    out = p.to_hist()
    assert out.storage_type is h.storage_type
    assert out.values() == pytest.approx(h.values(), nan_ok=True)


@pytest.mark.parametrize("storage", ["Mean", "WeightedMean"])
@pytest.mark.parametrize("name", list(AXES))
def test_axes_pyroot(tmp_path, name, storage):
    ROOT = pytest.importorskip("ROOT")

    h, x, y, w, _ = _fill(name, storage)
    n = len(h.axes[0])

    # filled at bin centers, so the x statistics match too
    expected = ROOT.TProfile(
        f"expected_1766_write_{name}_{storage}",
        "",
        n,
        np.asarray(h.axes[0].edges, "f8"),
    )
    expected.SetDirectory(ROOT.nullptr)
    if storage == "WeightedMean":
        expected.Sumw2()
    for xi, yi, wi in zip(x, y, w):
        expected.Fill(xi, yi, wi)

    p = _root_read(ROOT, tmp_path, h, "p")
    for i in range(n + 2):
        assert p.GetBinContent(i) == pytest.approx(expected.GetBinContent(i))
        assert p.GetBinError(i) == pytest.approx(expected.GetBinError(i))
        assert p.GetBinEntries(i) == pytest.approx(expected.GetBinEntries(i))
        assert p.GetBinEffectiveEntries(i) == pytest.approx(
            expected.GetBinEffectiveEntries(i)
        )
    for axis in (1, 2):
        assert p.GetMean(axis) == pytest.approx(expected.GetMean(axis))
        assert p.GetStdDev(axis) == pytest.approx(expected.GetStdDev(axis))
        assert p.GetMeanError(axis) == pytest.approx(expected.GetMeanError(axis))
    assert p.GetEffectiveEntries() == pytest.approx(expected.GetEffectiveEntries())
    assert p.GetBinSumw2().GetSize() == expected.GetBinSumw2().GetSize()
    if storage == "Mean":
        assert p.GetEntries() == expected.GetEntries()


@pytest.mark.parametrize(
    ("fSumw2", "sum_sq_dev"),
    [
        # two fills with weights 2 and -1 and y = 3 +- sqrt(2.5), 3 +- 2 sqrt(2.5):
        # sum(w) = 1, sum(w^2) = 5, mean = 3, sum(w*y^2) = 4 < sum(w) * mean^2 = 9
        (4.0, 4.0 - 9.0),
        # only round-off is clamped
        (9.0 * (1 - 1e-15), 0.0),
    ],
)
def test_sum_of_squared_deviations(tmp_path, fSumw2, sum_sq_dev):
    p = _write_and_read(tmp_path, _weighted_profile(fSumw2))
    errors = p.errors(flow=True)
    h = p.to_hist()
    view = h.view(flow=True)
    assert view["_sum_of_weighted_deltas_squared"][1] == pytest.approx(sum_sq_dev)

    if sum_sq_dev != 0:
        roundtrip = _write_and_read(tmp_path, h, "roundtrip")
        assert np.asarray(roundtrip.member("fSumw2")) == pytest.approx([0, 4, 0])
        assert roundtrip.errors(flow=True) == pytest.approx(errors)


def test_negative_weights_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    rng = np.random.default_rng(0)
    p = ROOT.TProfile("p_1766_negw", "", 4, 0, 4)
    p.SetDirectory(ROOT.nullptr)
    p.Sumw2()
    for x, y, w in zip(
        rng.uniform(0, 4, 40), rng.normal(5, 2, 40), rng.uniform(-1, 1.2, 40)
    ):
        p.Fill(x, y, w)

    h = _root_write_and_read(ROOT, tmp_path, p).to_hist()
    q = _root_read(ROOT, tmp_path, h, "q")
    for i in range(6):
        assert q.GetBinContent(i) == pytest.approx(p.GetBinContent(i))
        assert q.GetBinError(i) == pytest.approx(p.GetBinError(i))

    # the statistics of the filled x values, not of the bin centers, are kept
    assert q.GetEntries() == p.GetEntries()
    p_stats, q_stats = np.zeros(13), np.zeros(13)
    p.GetStats(p_stats)
    q.GetStats(q_stats)
    assert q_stats == pytest.approx(p_stats)
    # ROOT's own fTsumw, fTsumw2, fTsumwx and fTsumwx2 are kept exactly
    assert q_stats[:4].tolist() == p_stats[:4].tolist()


@pytest.mark.parametrize(
    "case",
    [
        "unchanged",
        "Merge",
        "SetStatOverflows",
        "PutStats",
        "SetBinContent",
        "SetBinEntries",
        "Rebin",
        "labels",
    ],
)
def test_global_statistics_pyroot(tmp_path, case):
    ROOT = pytest.importorskip("ROOT")

    rng = np.random.default_rng(0)
    # labels make the axis [0, 10] when written back, so ROOT's x statistics
    # don't apply then
    xmax = 1 if case == "labels" else 10
    p = ROOT.TProfile(f"p_1766_stats_{case}", "", 10, 0, xmax)
    p.SetDirectory(ROOT.nullptr)
    if case == "SetStatOverflows":
        # ROOT's statistics then include the flow bins
        p.SetStatOverflows(ROOT.TH1.kConsider)
    low, high = (0, xmax) if case == "labels" else (-1, 11)
    for x, y in zip(rng.uniform(low, high, 200), rng.normal(5, 2, 200)):
        p.Fill(x, y)

    if case == "Merge":
        # merging with a weighted profile doesn't make fBinSumw2, so fTsumw2 (the
        # sum of squared weights) can't be recomputed from the bins
        other = ROOT.TProfile("other_1766_stats", "", 10, 0, 10)
        other.SetDirectory(ROOT.nullptr)
        other.Sumw2()
        for x, y, w in zip(
            rng.uniform(0, 10, 200), rng.normal(5, 2, 200), rng.uniform(0.5, 2, 200)
        ):
            other.Fill(x, y, w)
        others = ROOT.TList()
        others.Add(other)
        p.Merge(others)
    # these make ROOT's statistics disagree with the bins (or tell ROOT to
    # recompute them from the bins, which it does with fTsumw = 0)
    elif case == "PutStats":
        p.PutStats(np.zeros(13))
    elif case == "SetBinContent":
        p.SetBinContent(2, 100)
    elif case == "SetBinEntries":
        p.SetBinEntries(2, 1)
    elif case == "Rebin":
        # 10 bins don't divide into groups of 3, which ROOT warns about
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.Rebin(3)
    elif case == "labels":
        for i in range(10):
            p.GetXaxis().SetBinLabel(i + 1, f"label{i}")

    q = _root_read(
        ROOT, tmp_path, _root_write_and_read(ROOT, tmp_path, p).to_hist(), "q"
    )
    assert q.GetEntries() == p.GetEntries()
    if case in ("unchanged", "Merge"):
        assert q.GetMean(1) == pytest.approx(p.GetMean(1))
        assert q.GetStdDev(1) == pytest.approx(p.GetStdDev(1))
        assert q.GetEffectiveEntries() == pytest.approx(p.GetEffectiveEntries())
    else:
        # recomputed from the bin centers
        n = q.GetNbinsX()
        w = np.array([q.GetBinEntries(i) for i in range(1, n + 1)])
        x = np.array([q.GetBinCenter(i) for i in range(1, n + 1)])
        assert q.GetMean(1) == pytest.approx((w * x).sum() / w.sum())


def test_out_of_date_statistics(tmp_path):
    # ROOT's fTsumw = 0 means that its statistics must be recomputed from the bins
    p = _write_and_read(tmp_path, _weighted_profile(10.0, fTsumw=0.0, fTsumwx=7.0))
    roundtrip = _write_and_read(tmp_path, p.to_hist(), "roundtrip")
    assert roundtrip.member("fTsumw") == 1.0
    assert roundtrip.member("fTsumwx") == 0.5


@pytest.mark.parametrize(
    "axis", [hist.axis.Regular(3, 0, 3), hist.axis.StrCategory(["a", "b", "c"])]
)
def test_weighted_empty_bins(tmp_path, axis):
    h = hist.Hist(axis, storage=hist.storage.WeightedMean())
    h.fill([axis.value(0)] * 2, sample=[1.0, 3.0], weight=[2.0, 1.0])

    p = _write_and_read(tmp_path, h)
    out = p.to_hist()

    # fBinSumw2 is the sum of squared weights, which is 0 (not 0/0) in empty bins
    sumw2 = out.view(flow=True)["sum_of_weights_squared"]
    assert not np.isnan(sumw2).any()
    assert sumw2.sum() == 5.0
    assert out.counts() == pytest.approx(p.counts())
    assert out.counts() == pytest.approx([9.0 / 5.0, 0.0, 0.0])


def test_weighted_empty_bins_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    p = ROOT.TProfile("p_1766_empty", "", 3, 0, 3)
    p.SetDirectory(ROOT.nullptr)
    p.Sumw2()
    p.Fill(0.5, 1, 2)
    p.Fill(0.5, 3, 1)
    r = _root_read(
        ROOT, tmp_path, _root_write_and_read(ROOT, tmp_path, p).to_hist(), "r"
    )

    other = ROOT.TProfile("other_1766_empty", "", 3, 0, 3)
    other.SetDirectory(ROOT.nullptr)
    other.Sumw2()
    other.Fill(1.5, 2, 1.5)
    other.Fill(1.5, 4, 0.5)
    r.Add(other)

    for i in range(5):
        assert np.isfinite(r.GetBinError(i))
    assert r.GetBinEffectiveEntries(1) == pytest.approx(9.0 / 5.0)
    assert r.GetBinEffectiveEntries(2) == pytest.approx(4.0 / 2.5)


def test_zero_sum_of_weights(tmp_path):
    # boost-histogram's mean is undefined in a bin whose weights sum to zero
    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.WeightedMean())
    h.fill([0.5, 0.5, 1.5], sample=[1.0, 3.0, 2.0], weight=[1.0, -1.0, 1.0])

    p = _write_and_read(tmp_path, h)
    assert p.values(flow=True).tolist() == [0, 0, 2, 0]
    assert np.isfinite(p.member("fSumw2")).all()
    assert p.member("fTsumwy") == 2
    assert p.member("fTsumwy2") == 4

    # and from ROOT, where such a bin has a sum of weighted values (-2)
    p = to_TProfile(
        fName="p",
        fTitle="",
        data=np.array([0.0, -2.0, 2.0, 0.0], np.float64),
        fEntries=3.0,
        fTsumw=1.0,
        fTsumw2=3.0,
        fTsumwx=1.5,
        fTsumwx2=2.25,
        fTsumwy=0.0,
        fTsumwy2=-4.0,
        fSumw2=np.array([0.0, -8.0, 4.0, 0.0], np.float64),
        fBinEntries=np.array([0.0, 0.0, 1.0, 0.0], np.float64),
        fBinSumw2=np.array([0.0, 2.0, 1.0, 0.0], np.float64),
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=2, fXmin=0, fXmax=2),
    )
    p = _write_and_read(tmp_path, p, "from_root")
    with pytest.warns(UserWarning, match="zero sum of weights") as record:
        h = p.to_hist()
    assert record[0].filename == __file__
    assert h.view(flow=True)["_sum_of_weighted_deltas_squared"][1] == -8


def _modified(change):
    # a profile with statistics as if from ROOT: its fEntries (3) is not its sum
    # of weights (3.5), and its fTsumwx (1) is not from bin centers (3.25)
    def make(tmp_path):
        h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.WeightedMean())
        h.fill([0.5, 1.5, 1.5], sample=[1.0, 2.0, 4.0], weight=[2.0, 0.5, 1.0])
        h.metadata = {"fEntries": 3, "fTsumwx": 1.0, "fTsumwx2": 0.75}
        p = _write_and_read(tmp_path, h)
        # explicit statistics are used
        assert p.member("fEntries") == 3
        assert p.member("fTsumwx") == 1.0
        return change(p.to_hist())

    return make


@pytest.mark.parametrize(
    ("make", "fEntries", "fTsumwx"),
    [
        (_modified(lambda h: h), 3, 1.0),
        (_modified(lambda h: h.copy()), 3, 1.0),
        # the bins changed, so the statistics are recomputed from the bins
        (_modified(lambda h: h + h), 7, 6.5),
        (_modified(lambda h: h.fill([0.5], sample=[3.0], weight=[1.0])), 4.5, 3.75),
        (_modified(lambda h: h.fill([0.5], sample=[3.0], weight=[1e-9])), 3.5, 3.25),
    ],
    ids=["unchanged", "copy", "added", "filled", "filled-small-weight"],
)
def test_stale_statistics(tmp_path, make, fEntries, fTsumwx):
    p = _write_and_read(tmp_path, make(tmp_path), "roundtrip")
    assert p.member("fEntries") == pytest.approx(fEntries)
    assert p.member("fTsumwx") == pytest.approx(fTsumwx)


def test_unweighted_fEntries(tmp_path):
    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Mean())
    h.fill([0.5, 1.5, 5.0], sample=[1.0, 2.0, 4.0])
    p = _write_and_read(tmp_path, h).to_hist()
    assert _write_and_read(tmp_path, p + p, "roundtrip").member("fEntries") == 6


@pytest.mark.parametrize("error_mode", [0, 1, 2, 3])
def test_error_mode_roundtrip(tmp_path, error_mode):
    h = _write_and_read(tmp_path, _weighted_profile(10.0, error_mode)).to_hist()
    assert h.metadata["fErrorMode"] == error_mode
    roundtrip = _write_and_read(tmp_path, h, "roundtrip")
    assert roundtrip.member("fErrorMode") == error_mode


@pytest.mark.parametrize(
    ("value", "expected"), [("", 0), ("s", 1), ("i", 2), ("g", 3), (2, 2), (None, 0)]
)
def test_error_mode_metadata(tmp_path, value, expected):
    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Mean())
    h.fill([0.5], sample=[1.0])
    h.metadata = {"fErrorMode": value}
    assert _write_and_read(tmp_path, h).member("fErrorMode") == expected

    h.metadata = {"fErrorMode": "x"}
    with pytest.raises(ValueError, match="error_mode must be"):
        _write_and_read(tmp_path, h)


def test_non_dict_metadata(tmp_path):
    h = hist.Hist(
        hist.axis.Regular(3, 0, 3), storage=hist.storage.Mean(), metadata="info"
    )
    h.fill([1, 5], sample=[1.0, 2.0])
    assert _write_and_read(tmp_path, h).member("fEntries") == 2


def test_non_boost_mean_raises(tmp_path):
    h = hist.Hist(hist.axis.Regular(3, 0, 3), storage=hist.storage.Mean())

    class MeanHistogram:
        kind = "MEAN"
        axes = h.axes

        def values(self, flow=False):
            return h.values(flow=flow)

        def variances(self, flow=False):
            return h.variances(flow=flow)

        def counts(self, flow=False):
            return h.counts(flow=flow)

    with pytest.raises(NotImplementedError, match="boost-histogram and hist objects"):
        _write_and_read(tmp_path, MeanHistogram())
