# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.writing.identify import to_TAxis, to_TProfile


def _profile(error_mode=0, fBinEntries=(0.0, 2.0, 2.0, 0.0)):
    # bin 1: y = {1, 3}; bin 2: y = {3, 3}
    return to_TProfile(
        fName="p",
        fTitle="",
        data=np.array([0.0, 4.0, 6.0, 0.0], np.float64),
        fEntries=4.0,
        fTsumw=4.0,
        fTsumw2=4.0,
        fTsumwx=4.0,
        fTsumwx2=5.0,
        fTsumwy=10.0,
        fTsumwy2=28.0,
        fSumw2=np.array([0.0, 10.0, 18.0, 0.0], np.float64),
        fBinEntries=np.array(fBinEntries, np.float64),
        fBinSumw2=np.array([], np.float64),
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=2, fXmin=0, fXmax=2),
        fErrorMode=error_mode,
    )


def _write_and_read(tmp_path, obj):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["p"] = obj
    return uproot.open(filepath)["p"]


def _key(f, classname):
    return next(k for k, v in f.classnames().items() if v == classname)


@pytest.mark.parametrize(
    ("error_mode", "names", "expected"),
    [
        (0, ["", 0, None], [1 / np.sqrt(2), 0]),
        (1, ["s", 1], [1, 0]),
        (2, ["i", 2], [1 / np.sqrt(2), 1 / np.sqrt(12 * 2)]),
        (3, ["g", 3], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
    ],
)
def test_error_mode(tmp_path, error_mode, names, expected):
    p = _write_and_read(tmp_path, _profile(error_mode))

    # the default follows the profile's fErrorMode, like ROOT's GetBinError
    assert p.errors() == pytest.approx(expected)
    assert p.variances() == pytest.approx(np.square(expected))
    for name in names:
        assert p.errors(error_mode=name) == pytest.approx(expected)
        assert p.errors(flow=True, error_mode=name)[1:-1] == pytest.approx(expected)

    with pytest.raises(ValueError, match="error_mode must be"):
        p.errors(error_mode="x")


@pytest.mark.parametrize("weights", ["unweighted", "weighted", "negative"])
@pytest.mark.parametrize("option", ["", "s", "i", "g"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_errors_pyroot(tmp_path, ndim, option, weights):
    ROOT = pytest.importorskip("ROOT")

    rng = np.random.default_rng(ndim)
    name = f"p_1766_errors_{ndim}_{option or 'mean'}_{weights}"
    if ndim == 1:
        p = ROOT.TProfile(name, "", 3, 0, 3, option)
    elif ndim == 2:
        p = ROOT.TProfile2D(name, "", 3, 0, 3, 2, 0, 2, option)
    else:
        p = ROOT.TProfile3D(name, "", 3, 0, 3, 2, 0, 2, 2, 0, 2, option)
    p.SetDirectory(ROOT.nullptr)
    if weights != "unweighted":
        p.Sumw2()
    low = -1 if weights == "negative" else 0.5
    for _ in range(2000):
        x = rng.uniform(-0.5, 3.5, 4)
        w = 1.0 if weights == "unweighted" else rng.uniform(low, 2)
        p.Fill(*x[:ndim], rng.normal(5, 2), w)

    filepath = os.path.join(tmp_path, "test.root")
    f = ROOT.TFile(filepath, "RECREATE")
    p.Write()
    f.Close()

    with uproot.open(filepath) as f:
        u = f[name]
        values = u.values(flow=True)
        errors = u.errors(flow=True)
        counts = u.counts(flow=True)
        assert u.weighted == (weights != "unweighted")
        assert u == u  # even with NaN errors (option "g" with negative weights)

    for index in np.ndindex(errors.shape):
        b = p.GetBin(*index)
        assert values[index] == pytest.approx(p.GetBinContent(b), rel=1e-9, abs=1e-9)
        assert errors[index] == pytest.approx(
            p.GetBinError(b), rel=1e-6, abs=1e-6, nan_ok=True
        )
        assert counts[index] == pytest.approx(p.GetBinEffectiveEntries(b), rel=1e-9)


def test_equality(tmp_path):
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert f["hprof"] == f["hprof"]
        assert not (f["hprof"] == f["hpx"])

    with uproot.open(skhep_testdata.data_path("uproot-issue-227a.root")) as f:
        key = _key(f, "TProfile2D")
        assert f[key] == f[key]

    # option "g" gives NaN errors for a negative sum of weights, as in ROOT
    p = _write_and_read(tmp_path, _profile(3, fBinEntries=(0.0, -2.0, 2.0, 0.0)))
    assert np.isnan(p.errors()[0])
    assert p == p


@pytest.mark.parametrize(
    "filename, classname",
    [
        ("uproot-hepdata-example.root", "TProfile"),
        ("uproot-issue-227a.root", "TProfile2D"),
        ("uproot-issue-227b.root", "TProfile3D"),
    ],
)
def test_missing_fBinSumw2(filename, classname):
    # TProfile class versions before 6 (ROOT < 5.26) have no fBinSumw2
    filepath = skhep_testdata.data_path(filename)
    with uproot.open(filepath) as f:
        key = _key(f, classname)
        p = f[key]
        expected_counts = np.asarray(p.member("fBinEntries"))
        values = p.values(flow=True)
        errors = p.errors(flow=True)
        assert not p.weighted

    # reopen to get an object without cached values and errors
    with uproot.open(filepath, object_cache=None) as f:
        p = f[key]
        del p._members["fBinSumw2"]

        assert p.counts(flow=True).reshape(-1, order="F") == pytest.approx(
            expected_counts
        )
        assert p.values(flow=True) == pytest.approx(values)
        assert p.errors(flow=True) == pytest.approx(errors)
        assert p.variances(flow=True) == pytest.approx(errors**2)
        if classname == "TProfile":
            pytest.importorskip("hist")
            assert p.to_hist().values(flow=True) == pytest.approx(values)


@pytest.mark.parametrize(
    "filename, classname",
    [
        ("uproot-hepdata-example.root", "TProfile"),
        ("uproot-issue-227a.root", "TProfile2D"),
        ("uproot-issue-227b.root", "TProfile3D"),
    ],
)
@pytest.mark.parametrize("flow", [False, True])
def test_to_numpy(filename, classname, flow):
    with uproot.open(skhep_testdata.data_path(filename)) as f:
        p = f[_key(f, classname)]
        values, *edges = p.to_numpy(flow=flow)
        dd_values, dd_edges = p.to_numpy(flow=flow, dd=True)

    assert values.tolist() == dd_values.tolist() == p.values(flow=flow).tolist()
    assert len(edges) == len(dd_edges) == len(p.axes)
    for axis, e, dd_e in zip(p.axes, edges, dd_edges):
        assert e.tolist() == dd_e.tolist() == axis.edges(flow=flow).tolist()
    assert values.shape == tuple(len(e) - 1 for e in edges)
