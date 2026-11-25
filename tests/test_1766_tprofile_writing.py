# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import contextlib
import os
import warnings

import numpy as np
import pytest

import uproot
from uproot.writing.identify import to_TAxis, to_TProfile

hist = pytest.importorskip("hist")

# hist axes and what to fill each of their bins with, from the underflow (if
# any) to the overflow (if any), or None to fill at the bin centers
AXES = {
    "Regular": (hist.axis.Regular(4, 0, 4), None),
    "Variable": (hist.axis.Variable([0, 1, 3, 6]), None),
    "Integer": (hist.axis.Integer(0, 4), [-1, 0, 1, 2, 3, 4]),
    "Regular-noflow": (hist.axis.Regular(4, 0, 4, flow=False), None),
    "Regular-nounderflow": (hist.axis.Regular(4, 0, 4, underflow=False), None),
    "IntCategory": (hist.axis.IntCategory([10, 20, 30]), [10, 20, 30, 99]),
    "IntCategory-noflow": (hist.axis.IntCategory([10, 20], overflow=False), [10, 20]),
    "StrCategory": (hist.axis.StrCategory(["a", "b"]), ["a", "b", "zz"]),
    "Boolean": (hist.axis.Boolean(), [False, True]),
}


def _fill(name, storage):
    axis, values = AXES[name]
    n, edges = len(axis), axis.edges
    low = -1 if axis.traits.underflow else 0
    high = n if axis.traits.overflow else n - 1
    rng = np.random.default_rng(0)
    index = rng.integers(low, high + 1, 200)
    y = rng.normal(5, 2, 200)
    w = rng.uniform(-0.5, 2, 200) if storage == "WeightedMean" else np.ones(200)

    # ROOT x-coordinates: bin centers, with flow bins just outside the edges
    centers = np.concatenate(
        [[edges[0] - 0.5], (edges[:-1] + edges[1:]) / 2, [edges[-1] + 0.5]]
    )
    x = centers[index + 1]

    h = hist.Hist(axis, storage=getattr(hist.storage, storage)())
    h.fill(
        x if values is None else [values[i - low] for i in index], sample=y, weight=w
    )
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


def _profile(
    sumy2, sumy=3.0, entries=1.0, sumw2=5.0, fTsumw=None, fTsumwx=None, fErrorMode=0
):
    # one bin of [0, 1] with sums of weights (entries), of squared weights
    # (sumw2, or None if unweighted), and of weighted y and y^2 (sumy, sumy2)
    return to_TProfile(
        fName="p",
        fTitle="",
        data=np.array([0.0, sumy, 0.0]),
        fEntries=entries,
        fTsumw=entries if fTsumw is None else fTsumw,
        fTsumw2=entries if sumw2 is None else sumw2,
        fTsumwx=0.5 * entries if fTsumwx is None else fTsumwx,
        fTsumwx2=0.25 * entries,
        fTsumwy=sumy,
        fTsumwy2=sumy2,
        fSumw2=np.array([0.0, sumy2, 0.0]),
        fBinEntries=np.array([0.0, entries, 0.0]),
        fBinSumw2=np.array([] if sumw2 is None else [0.0, sumw2, 0.0]),
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
    for member, values in [
        ("fTsumw", w),
        ("fTsumwx", w * x),
        ("fTsumwx2", w * x**2),
        ("fTsumwy", w * y),
        ("fTsumwy2", w * y**2),
    ]:
        assert p.member(member) == pytest.approx(values[inside].sum())

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
        f"expected_1766_write_{name}_{storage}", "", n, np.asarray(h.axes[0].edges)
    )
    expected.SetDirectory(ROOT.nullptr)
    if storage == "WeightedMean":
        expected.Sumw2()
    for xi, yi, wi in zip(x, y, w):
        expected.Fill(xi, yi, wi)

    p = _root_read(ROOT, tmp_path, h, "p")
    for method in [
        "GetBinContent",
        "GetBinError",
        "GetBinEntries",
        "GetBinEffectiveEntries",
    ]:
        for i in range(n + 2):
            assert getattr(p, method)(i) == pytest.approx(getattr(expected, method)(i))
    for method in ["GetMean", "GetStdDev", "GetMeanError"]:
        for axis in (1, 2):
            assert getattr(p, method)(axis) == pytest.approx(
                getattr(expected, method)(axis)
            )
    assert p.GetEffectiveEntries() == pytest.approx(expected.GetEffectiveEntries())
    assert p.GetBinSumw2().GetSize() == expected.GetBinSumw2().GetSize()
    if storage == "Mean":
        assert p.GetEntries() == expected.GetEntries()


@pytest.mark.parametrize(
    ("kwargs", "expected", "warning"),
    [
        # two fills with weights 2 and -1 and y = 3 +- sqrt(2.5), 3 +- 2 sqrt(2.5):
        # sum(w) = 1, sum(w^2) = 5, mean = 3, sum(w*y^2) = 4 < sum(w) * mean^2 = 9
        ({"sumy2": 4.0}, -5.0, "from negative weights"),
        # only round-off is clamped,
        ({"sumy2": 9.0 * (1 - 1e-15)}, 0.0, None),
        # which grows with the number of fills that ROOT summed: here, of y = 2.3
        # with ROOT's sum of y^2 off by -5e-10 (relative)
        (
            {
                "sumy2": 2.3**2 * 1e7 * (1 - 5e-10),
                "sumy": 2.3e7,
                "entries": 1e7,
                "sumw2": None,
            },
            0.0,
            None,
        ),
        (
            {
                "sumy2": 2.3**2 * 10 * (1 - 5e-10),
                "sumy": 23.0,
                "entries": 10.0,
                "sumw2": None,
            },
            -(2.3**2) * 10 * 5e-10,
            "from bins set with SetBinContent",
        ),
    ],
    ids=["negative", "round-off", "round-off-many-fills", "few-fills"],
)
def test_sum_of_squared_deviations(tmp_path, kwargs, expected, warning):
    p = _write_and_read(tmp_path, _profile(**kwargs))
    with (
        pytest.warns(UserWarning, match=warning)
        if warning
        else contextlib.nullcontext()
    ):
        h = p.to_hist()
    view = h.view(flow=True)
    field = (
        "_sum_of_weighted_deltas_squared" if p.weighted else "_sum_of_deltas_squared"
    )
    assert view[field][1] == pytest.approx(expected)

    if expected != 0:
        # kept as it is, so that writing back to ROOT gives the same sums
        roundtrip = _write_and_read(tmp_path, h, "roundtrip")
        assert roundtrip.member("fSumw2")[1] == pytest.approx(kwargs["sumy2"])
        assert roundtrip.errors(flow=True) == pytest.approx(p.errors(flow=True))


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

    # (in bins where boost-histogram's variances() are NaN anyway)
    with pytest.warns(UserWarning, match="negative sum of squared deviations"):
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

    with warnings.catch_warnings():
        if case in ("SetBinContent", "SetBinEntries"):
            # changing a bin's sum of values or of weights, but not its sum of
            # squared values, can make its sum of squared deviations negative
            warnings.filterwarnings(
                "ignore", "some bins have a negative sum", UserWarning
            )
        h = _root_write_and_read(ROOT, tmp_path, p).to_hist()
    q = _root_read(ROOT, tmp_path, h, "q")
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


@pytest.mark.parametrize(
    ("fTsumw", "kept"),
    [
        # ROOT sums the fills and the bins in different orders, so with many fills
        # (and weights that cancel) its fTsumw differs from the bins' by round-off
        (1.0 + 1e-11, True),
        (1.0 - 1e-11, True),
        # but no more than that unless its statistics are out of date (fTsumw = 0
        # tells ROOT to recompute them from the bins)
        (1.0 + 1e-6, False),
        (0.0, False),
    ],
)
def test_statistics_round_off(tmp_path, fTsumw, kept):
    p = _write_and_read(tmp_path, _profile(10.0, fTsumw=fTsumw, fTsumwx=7.0))
    roundtrip = _write_and_read(tmp_path, p.to_hist(), "roundtrip")
    assert roundtrip.member("fTsumw") == (fTsumw if kept else 1.0)
    assert roundtrip.member("fTsumwx") == (7.0 if kept else 0.5)


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
    assert out.counts() == pytest.approx(p.counts()) == [9.0 / 5.0, 0.0, 0.0]


def test_weighted_empty_bins_pyroot(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    profiles = []
    for name in ["p_1766_empty", "other_1766_empty"]:
        profiles.append(ROOT.TProfile(name, "", 3, 0, 3))
        profiles[-1].SetDirectory(ROOT.nullptr)
        profiles[-1].Sumw2()
    p, other = profiles
    p.Fill(0.5, 1, 2)
    p.Fill(0.5, 3, 1)
    other.Fill(1.5, 2, 1.5)
    other.Fill(1.5, 4, 0.5)

    r = _root_read(
        ROOT, tmp_path, _root_write_and_read(ROOT, tmp_path, p).to_hist(), "r"
    )
    r.Add(other)
    for i in range(5):
        assert np.isfinite(r.GetBinError(i))
    assert r.GetBinEffectiveEntries(1) == pytest.approx(9.0 / 5.0)
    assert r.GetBinEffectiveEntries(2) == pytest.approx(4.0 / 2.5)


@pytest.mark.parametrize(
    ("storage", "weights", "warning"),
    [
        # boost-histogram's mean is undefined in a bin whose weights sum to zero
        ("WeightedMean", [1.0, -1.0], "weights that sum to zero"),
        # and stays so after more fills
        ("WeightedMean", [1.0, -1.0, 1.0], "undefined"),
        ("Mean", [0.0, 1.0], "undefined"),
    ],
)
def test_undefined_mean(tmp_path, storage, weights, warning):
    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=getattr(hist.storage, storage)())
    h.fill([0.5] * len(weights), sample=[1.0, 3.0, 2.0][: len(weights)], weight=weights)
    h.fill([1.5], sample=[2.0])
    assert not np.isfinite(h.view()["value"][0])

    with pytest.warns(UserWarning, match=warning) as record:
        p = _write_and_read(tmp_path, h)
    assert record[0].filename == __file__
    # written as empty, not as NaN
    assert p.values(flow=True).tolist() == [0, 0, 2, 0]
    assert np.asarray(p.member("fBinEntries")).tolist() == [0, 0, 1, 0]
    assert np.isfinite(p.member("fSumw2")).all()
    assert (p.member("fTsumwy"), p.member("fTsumwy2")) == (2, 4)


def test_zero_sum_of_weights_from_root(tmp_path):
    # where such a bin has a sum of weighted values (-2)
    p = _write_and_read(tmp_path, _profile(-8.0, sumy=-2.0, entries=0.0, sumw2=2.0))
    with pytest.warns(UserWarning) as record:
        h = p.to_hist()
    assert [str(x.message)[:30] for x in record] == [
        "some bins have a zero sum of w",
        "some bins have a negative sum ",
    ]
    assert all(x.filename == __file__ for x in record)
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
        assert (p.member("fEntries"), p.member("fTsumwx")) == (3, 1.0)
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
        # the total weight is the same, but not the bins' (3 and 0.5)
        (
            _modified(lambda h: h.fill([0.5, 1.5], sample=[3.0, 3.0], weight=[1, -1])),
            3.5,
            2.25,
        ),
    ],
    ids=["unchanged", "copy", "added", "filled", "filled-small-weight", "cancelling"],
)
def test_stale_statistics(tmp_path, make, fEntries, fTsumwx):
    p = _write_and_read(tmp_path, make(tmp_path), "roundtrip")
    assert p.member("fEntries") == pytest.approx(fEntries)
    assert p.member("fTsumwx") == pytest.approx(fTsumwx)


def test_stale_statistics_reordered(tmp_path):
    h = hist.Hist(hist.axis.StrCategory(["a", "b", "c"]), storage=hist.storage.Mean())
    h.fill(["a", "a", "c"], sample=[1.0, 2.0, 3.0])
    h = _write_and_read(tmp_path, h).to_hist()
    assert "fTsumwx" in h.metadata

    # the same bins in a different order: c = 1, b = 0, a = 2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "List indexing selection is experimental")
        reordered = h[["c", "b", "a"]]
    p = _write_and_read(tmp_path, reordered, "reordered")
    assert p.member("fTsumwx") == 1 * 0.5 + 2 * 2.5
    assert p.member("fTsumwx2") == 1 * 0.5**2 + 2 * 2.5**2


def test_unweighted_fEntries(tmp_path):
    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=hist.storage.Mean())
    h.fill([0.5, 1.5, 5.0], sample=[1.0, 2.0, 4.0])
    p = _write_and_read(tmp_path, h).to_hist()
    assert _write_and_read(tmp_path, p + p, "roundtrip").member("fEntries") == 6


@pytest.mark.parametrize("error_mode", [0, 1, 2, 3])
def test_error_mode_roundtrip(tmp_path, error_mode):
    h = _write_and_read(tmp_path, _profile(10.0, fErrorMode=error_mode)).to_hist()
    assert h.metadata["fErrorMode"] == error_mode
    roundtrip = _write_and_read(tmp_path, h, "roundtrip")
    assert roundtrip.member("fErrorMode") == error_mode


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", 0), ("s", 1), ("i", 2), ("G", 3), (2, 2), (None, 0)],
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


class _ProfileLike:
    # a non-boost-histogram MEAN histogram, which can be written only if it has
    # the members of an Uproot TProfile
    def __init__(self, model):
        self._model = model
        self._bases = model._bases
        self.member = model.member
        self.axes = (hist.axis.Variable(model.axes[0].edges()),)
        self.kind = "MEAN"

    def values(self, flow=False):
        return self._model.values(flow=flow)

    def variances(self, flow=False):
        return self._model.variances(flow=flow)

    def counts(self, flow=False):
        return self._model.counts(flow=flow)


@pytest.mark.parametrize("case", ["members", "old-version", "no-members"])
def test_profile_like(tmp_path, case):
    model = _write_and_read(tmp_path, _profile(10.0, fErrorMode=1), "model")
    obj = _ProfileLike(model)
    if case == "no-members":
        del obj._bases
        with pytest.raises(NotImplementedError, match="boost-histogram and hist"):
            _write_and_read(tmp_path, obj)
        return
    if case == "old-version":
        # class versions before 6 (ROOT < 5.26) have no fBinSumw2
        del model._members["fBinSumw2"]

    p = _write_and_read(tmp_path, obj)
    assert p.member("fErrorMode") == 1
    assert len(p.member("fBinSumw2")) == (0 if case == "old-version" else 3)
    assert p.values().tolist() == model.values().tolist()
