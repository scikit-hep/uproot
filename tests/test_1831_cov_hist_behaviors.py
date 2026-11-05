# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
The user-facing methods of histograms, profiles, axes, graphs, RooFit plots, and
TParameters, for the values and errors of objects whose content is known.
"""

import operator

import numpy
import pytest
import skhep_testdata

import uproot
import uproot.behaviors.TH1
import uproot.behaviors.TProfile
import uproot.writing.identify as identify


def replace_member(obj, name, value):
    if name in obj._members:
        obj._members[name] = value
        return True
    return any(replace_member(base, name, value) for base in obj.bases)


def delete_member(obj, name):
    if name in obj._members:
        del obj._members[name]
        return True
    return any(delete_member(base, name) for base in obj.bases)


def axis(name="xaxis", fNbins=3, **kwargs):
    return identify.to_TAxis(
        fName=name, fTitle="", fNbins=fNbins, fXmin=0.0, fXmax=3.0, **kwargs
    )


def labels(*names):
    return identify.to_THashList([identify.to_TObjString(x) for x in names])


def th1(data=(0, 1, 2, 3, 0), sumw2=(), dtype="f8", xaxis=None):
    return identify.to_TH1x(
        fName="h1",
        fTitle="title",
        data=numpy.array(data, dtype=dtype),
        fEntries=6.0,
        fTsumw=6.0,
        fTsumw2=6.0,
        fTsumwx=1.0,
        fTsumwx2=1.0,
        fSumw2=numpy.array(sumw2, dtype="f8"),
        fXaxis=axis() if xaxis is None else xaxis,
    )


def th2(sumw2=()):
    return identify.to_TH2x(
        fName="h2",
        fTitle="title",
        data=numpy.arange(25, dtype="f8") - 12,
        fEntries=6.0,
        fTsumw=6.0,
        fTsumw2=6.0,
        fTsumwx=1.0,
        fTsumwx2=1.0,
        fTsumwy=1.0,
        fTsumwy2=1.0,
        fTsumwxy=1.0,
        fSumw2=numpy.array(sumw2, dtype="f8"),
        fXaxis=axis(),
        fYaxis=axis("yaxis"),
    )


def th3(sumw2=()):
    return identify.to_TH3x(
        fName="h3",
        fTitle="title",
        data=numpy.arange(125, dtype="f8") - 62,
        fEntries=6.0,
        fTsumw=6.0,
        fTsumw2=6.0,
        fTsumwx=1.0,
        fTsumwx2=1.0,
        fTsumwy=1.0,
        fTsumwy2=1.0,
        fTsumwxy=1.0,
        fTsumwz=1.0,
        fTsumwz2=1.0,
        fTsumwxz=1.0,
        fTsumwyz=1.0,
        fSumw2=numpy.array(sumw2, dtype="f8"),
        fXaxis=axis(),
        fYaxis=axis("yaxis"),
        fZaxis=axis("zaxis"),
    )


def tprofile(fSumw2=(0.0, 5.0, 20.0, 0.0, 0.0), xaxis=None):
    # the first bin has entries 1 and 2, the second has 1, 1, 3, and 3, the third is empty
    return identify.to_TProfile(
        fName="p1",
        fTitle="title",
        data=numpy.array([0.0, 3.0, 8.0, 0.0, 0.0]),
        fEntries=6.0,
        fTsumw=6.0,
        fTsumw2=6.0,
        fTsumwx=1.0,
        fTsumwx2=1.0,
        fTsumwy=11.0,
        fTsumwy2=25.0,
        fSumw2=numpy.array(fSumw2, dtype="f8"),
        fBinEntries=numpy.array([0.0, 2.0, 4.0, 0.0, 0.0]),
        fBinSumw2=numpy.array([], dtype="f8"),
        fXaxis=axis() if xaxis is None else xaxis,
    )


def tprofile2d():
    data = numpy.zeros(25)
    entries = numpy.zeros(25)
    sumw2 = numpy.zeros(25)
    data[6], entries[6], sumw2[6] = 3.0, 2.0, 5.0  # x bin 1, y bin 1
    data[7], entries[7], sumw2[7] = 8.0, 4.0, 20.0  # x bin 2, y bin 1
    return identify.to_TProfile2D(
        fName="p2",
        fTitle="title",
        data=data,
        fEntries=6.0,
        fTsumw=6.0,
        fTsumw2=6.0,
        fTsumwx=1.0,
        fTsumwx2=1.0,
        fTsumwy=1.0,
        fTsumwy2=1.0,
        fTsumwxy=1.0,
        fTsumwz=11.0,
        fTsumwz2=25.0,
        fSumw2=sumw2,
        fBinEntries=entries,
        fBinSumw2=numpy.array([], dtype="f8"),
        fXaxis=axis(),
        fYaxis=axis("yaxis"),
    )


def tprofile3d():
    data = numpy.zeros(125)
    entries = numpy.zeros(125)
    sumw2 = numpy.zeros(125)
    data[31], entries[31], sumw2[31] = 3.0, 2.0, 5.0  # x bin 1, y bin 1, z bin 1
    data[32], entries[32], sumw2[32] = 8.0, 4.0, 20.0  # x bin 2, y bin 1, z bin 1
    return identify.to_TProfile3D(
        fName="p3",
        fTitle="title",
        data=data,
        fEntries=6.0,
        fTsumw=6.0,
        fTsumw2=6.0,
        fTsumwx=1.0,
        fTsumwx2=1.0,
        fTsumwy=1.0,
        fTsumwy2=1.0,
        fTsumwxy=1.0,
        fTsumwz=1.0,
        fTsumwz2=1.0,
        fTsumwxz=1.0,
        fTsumwyz=1.0,
        fTsumwt=11.0,
        fTsumwt2=25.0,
        fSumw2=sumw2,
        fBinEntries=entries,
        fBinSumw2=numpy.array([], dtype="f8"),
        fXaxis=axis(),
        fYaxis=axis("yaxis"),
        fZaxis=axis("zaxis"),
    )


@pytest.mark.parametrize(
    "make, accepted, message",
    [
        (th1, [(0, "fXaxis"), (-1, "fXaxis"), ("x", "fXaxis")], "axis must be 0"),
        (tprofile, [(0, "fXaxis"), ("x", "fXaxis")], "for a TProfile"),
        (th2, [(0, "fXaxis"), (-1, "fYaxis"), ("y", "fYaxis")], "for a TH2"),
        (tprofile2d, [(-2, "fXaxis"), (1, "fYaxis")], "for a TProfile2D"),
        (th3, [(-3, "fXaxis"), (-2, "fYaxis"), ("z", "fZaxis")], "for a TH3"),
        (
            tprofile3d,
            [(0, "fXaxis"), (1, "fYaxis"), (-1, "fZaxis")],
            "for a TProfile3D",
        ),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_axis_selection(make, accepted, message):
    hist = make()
    for which, member in accepted:
        assert hist.axis(which) is hist.member(member)
    assert hist.axes == tuple(
        hist.member(x) for x in ("fXaxis", "fYaxis", "fZaxis")[: len(hist.axes)]
    )
    with pytest.raises(ValueError, match=message):
        hist.axis("t")


@pytest.mark.parametrize(
    "make, kind, num_axes",
    [
        (th1, "COUNT", 1),
        (th2, "COUNT", 2),
        (th3, "COUNT", 3),
        (tprofile, "MEAN", 1),
        (tprofile2d, "MEAN", 2),
        (tprofile3d, "MEAN", 3),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_kind_and_dimensions(make, kind, num_axes):
    hist = make()
    assert hist.kind == kind
    assert len(hist.axes) == num_axes


def test_abstract_histogram_methods():
    class Bare(uproot.behaviors.TH1.Histogram):
        def __repr__(self):
            return "<bare histogram>"

    class BareProfile(uproot.behaviors.TProfile.Profile):
        def __repr__(self):
            return "<bare profile>"

    for call in (
        lambda: Bare().axes,
        lambda: Bare().axis(0),
        lambda: Bare().kind,
        lambda: Bare().values(),
    ):
        with pytest.raises(NotImplementedError, match="<bare histogram>"):
            call()

    for call in (lambda: BareProfile().counts(), lambda: BareProfile().values()):
        with pytest.raises(NotImplementedError, match="<bare profile>"):
            call()


def test_histogram_equality():
    reference = th1(sumw2=(0, 1, 2, 3, 0))
    assert reference == th1(sumw2=(0, 1, 2, 3, 0))
    assert reference != th1(sumw2=(0, 1, 2, 4, 0))  # variances differ
    assert reference != th1(data=(0, 1, 2, 4, 0), sumw2=(0, 1, 2, 3, 0))  # values
    wider = identify.to_TAxis("xaxis", "", 3, 0.0, 6.0)
    assert reference != th1(sumw2=(0, 1, 2, 3, 0), xaxis=wider)  # axes
    assert reference != th2()
    assert reference != "h1"


def test_counts_of_a_histogram_are_its_values():
    assert th1().counts().tolist() == [1.0, 2.0, 3.0]
    assert th1().counts(flow=True).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_th1_to_numpy_dd():
    values, (edges,) = th1().to_numpy(dd=True)
    assert values.tolist() == [1.0, 2.0, 3.0]
    assert edges.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_to_boost_storage_follows_the_data():
    boost_histogram = pytest.importorskip("boost_histogram")

    counts = th1(dtype="i4").to_boost()
    assert isinstance(counts.storage_type(), boost_histogram.storage.Int64)
    assert counts.view().tolist() == [1, 2, 3]

    doubles = th1(dtype="f8").to_boost()
    assert isinstance(doubles.storage_type(), boost_histogram.storage.Double)

    weights = th1(sumw2=(0, 7, 8, 9, 0)).to_boost()
    assert isinstance(weights.storage_type(), boost_histogram.storage.Weight)
    assert weights.variances().tolist() == [7.0, 8.0, 9.0]


def test_to_hist_with_explicit_metadata():
    pytest.importorskip("hist")
    hist = th1().to_hist(
        metadata={"histogram_name": "fName"}, axis_metadata={"axis_name": "fName"}
    )
    assert hist.values().tolist() == [1.0, 2.0, 3.0]
    assert hist.histogram_name == "h1"
    assert hist.axes[0].axis_name == "xaxis"


def test_profile_counts_flow_and_no_flow():
    # without fBinSumw2, the effective counts are the entries
    assert tprofile().counts().tolist() == [2.0, 4.0, 0.0]
    assert tprofile().counts(flow=True).tolist() == [0.0, 2.0, 4.0, 0.0, 0.0]


def test_profile_error_mode_integer_uses_sqrt12_for_zero_spread():
    # bin 1 has no spread: two entries of 1.5
    profile = tprofile(fSumw2=(0.0, 4.5, 20.0, 0.0, 0.0))
    assert profile.errors(error_mode="i")[0] == pytest.approx(1.0 / numpy.sqrt(24.0))


@pytest.mark.parametrize(
    "mode, expected",
    [
        (None, ""),
        ("", ""),
        (uproot.behaviors.TProfile._kERRORMEAN, ""),
        ("s", "S"),
        (uproot.behaviors.TProfile._kERRORSPREAD, "S"),
        ("i", "I"),
        (uproot.behaviors.TProfile._kERRORSPREADI, "I"),
        ("g", "G"),
        (uproot.behaviors.TProfile._kERRORSPREADG, "G"),
        ("anything else", "_"),
    ],
)
def test_profile_error_mode_names(mode, expected):
    assert uproot.behaviors.TProfile._error_mode_str(mode) == expected


def test_th1_values_and_variances_without_flow(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file["h"] = numpy.array([1.0, 2.0, 3.0]), numpy.array([0.0, 1.0, 2.0, 3.0])

    with uproot.open(path) as file:
        values, variances = file["h"]._values_variances(False)
    assert values.tolist() == [1, 2, 3]
    assert variances.tolist() == [1, 2, 3]


def test_profile_error_mode_is_validated():
    with pytest.raises(ValueError, match="error_mode must be None/0/''"):
        tprofile().errors(error_mode="wrong")


def test_profile_error_without_fSumw2():
    # the sum of squares is unknown: ROOT's formula sees 0, so the spread is the mean
    assert tprofile(fSumw2=()).errors(error_mode="s").tolist() == [1.5, 2.0, 0.0]


@pytest.mark.parametrize(
    "make, index",
    [
        (tprofile, (slice(0, 2),)),
        (tprofile2d, (slice(0, 2), 0)),
        (tprofile3d, (slice(0, 2), 0, 0)),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_profile_without_fsumw2_member(make, index):
    # an object with no fSumw2 at all, as from a very old version
    profile = make()
    assert delete_member(profile, "fSumw2")
    assert profile.errors(error_mode="s")[index].tolist() == [1.5, 2.0]


@pytest.mark.parametrize(
    "fSumw2, sum_sq_dev",
    [
        ((0.0, 5.0, 20.0, 0.0, 0.0), [0.5, 4.0, 0.0]),
        ((), [0.0, 0.0, 0.0]),  # the sum of squares is unknown
    ],
)
def test_profile_to_boost_sum_of_deltas(fSumw2, sum_sq_dev):
    boost_histogram = pytest.importorskip("boost_histogram")

    hist = tprofile(fSumw2=fSumw2).to_boost()
    assert isinstance(hist.storage_type(), boost_histogram.storage.Mean)
    assert hist.view()["count"].tolist() == [2.0, 4.0, 0.0]
    assert hist.view()["value"].tolist() == [1.5, 2.0, 0.0]
    assert hist.view()["_sum_of_deltas_squared"].tolist() == sum_sq_dev


def test_profile_to_boost_without_fsumw2_member():
    pytest.importorskip("boost_histogram")
    profile = tprofile()
    assert delete_member(profile, "fSumw2")
    assert profile.to_boost().view()["_sum_of_deltas_squared"].tolist() == [0.0] * 3


@pytest.mark.parametrize("make", [tprofile2d, tprofile3d], ids=lambda x: x.__name__)
def test_higher_dimensional_profiles_have_no_boost_equivalent(make):
    with pytest.raises(NotImplementedError, match="this one kinda doesn't exist"):
        make().to_boost()


def test_axis_traits_repr():
    xaxis = axis()
    assert repr(xaxis.traits) == f"AxisTraits({xaxis!r})"


def test_axis_equality():
    assert axis() == axis()
    assert axis() != "xaxis"
    assert axis() != axis(fNbins=4)
    assert axis() != identify.to_TAxis("xaxis", "", 3, 0.0, 6.0)

    abc = axis(fLabels=labels("a", "b", "c"))
    assert abc == axis(fLabels=labels("a", "b", "c"))
    assert abc != axis(fLabels=labels("a", "b", "d"))
    assert abc != axis()
    assert axis() != abc


def test_axis_width_of_variable_bins():
    variable = axis(fXbins=numpy.array([0.0, 0.5, 1.5, 3.0]))
    assert variable.width == pytest.approx(1.0)
    assert variable.widths().tolist() == [0.5, 1.0, 1.5]
    assert axis().width == 1.0


def test_axis_labels_with_flow():
    abc = axis(fLabels=labels("a", "b", "c"))
    assert abc.labels() == ["a", "b", "c"]
    assert abc.labels(flow=True) == ["underflow", "a", "b", "c", "overflow"]
    assert axis().labels(flow=True) is None


@pytest.fixture
def tgraph_asymm():
    path = skhep_testdata.data_path("uproot-issue-240.root")
    with uproot.open(path) as file:
        yield file["Expected limit 1lbb/Graph1D_y1"]


def test_tgraph_rejects_unknown_axis(tgraph_asymm):
    graph = tgraph_asymm.bases[0]
    x, y = graph.values("both")
    assert numpy.array_equal(x, graph.member("fX"))
    assert numpy.array_equal(y, graph.member("fY"))
    with pytest.raises(ValueError, match=r"or 'both' for a TGraph$"):
        graph.values("z")


def test_tgraph_asymm_errors_rejects_unknown_arguments(tgraph_asymm):
    message = r"or 'both' for a TGraphAsymmErrors$"
    with pytest.raises(ValueError, match=message):
        tgraph_asymm.values("z")
    with pytest.raises(ValueError, match=message):
        tgraph_asymm.errors("low", "z")
    with pytest.raises(
        ValueError, match="which must be 'low', 'high', 'mean', or 'diff'"
    ):
        tgraph_asymm.errors("sum")


@pytest.fixture
def rooplot():
    with uproot.open(skhep_testdata.data_path("uproot-issue-350.root")) as file:
        yield file


def test_roohist_to_numpy(rooplot):
    roohist = rooplot["data"]
    _, expected = roohist.values()
    values, edges = roohist.to_numpy()
    assert numpy.array_equal(values, expected)
    assert edges == pytest.approx(numpy.arange(0.0, 51.0))

    values, (edges,) = roohist.to_numpy(dd=True)
    assert numpy.array_equal(values, expected)
    assert edges == pytest.approx(numpy.arange(0.0, 51.0))


def test_roohist_to_numpy_rejects_gaps(rooplot):
    roohist = rooplot["data"]
    fEXlow = roohist.member("fEXlow").copy()
    fEXlow[1] = 0.25  # bin 1 no longer touches bin 0
    replace_member(roohist, "fEXlow", fEXlow)
    with pytest.raises(ValueError, match=r"bin_edges_low\[1:\] != bin_edges_high"):
        roohist.to_numpy()
    pytest.importorskip("boost_histogram")
    with pytest.raises(ValueError, match=r"bin_edges_low\[1:\] != bin_edges_high"):
        roohist.to_boost()


def test_roohist_to_hist(rooplot):
    hist = pytest.importorskip("hist")
    roohist = rooplot["data"]
    _, expected = roohist.values()
    for converted in (roohist.to_hist(), hist.Hist(roohist)):
        assert converted.values() == pytest.approx(expected)
        assert converted.axes[0].edges == pytest.approx(numpy.arange(0.0, 51.0))


def test_roocurve_to_hist(rooplot):
    pytest.importorskip("hist")
    curve = rooplot["bhist"]
    edges = numpy.arange(0.0, 51.0)
    hist = curve.to_hist(edges, rooplot["berr"])
    assert hist.axes[0].edges == pytest.approx(edges)
    assert hist.values() == pytest.approx(
        curve.interpolate((edges[1:] + edges[:-1]) / 2)
    )


def test_roocurve_methods_are_for_one_curve_type(rooplot):
    values, errors = rooplot["bhist"], rooplot["berr"]
    with pytest.raises(ValueError, match="Try interpolate_errors"):
        errors.interpolate([1.0])
    with pytest.raises(ValueError, match="Try interpolate."):
        values.interpolate_errors([1.0])
    with pytest.raises(ValueError, match="Try interpolate."):
        values.interpolate_asymm_errors([1.0])
    with pytest.raises(
        ValueError, match="to_boost should be called on the value curve"
    ):
        errors.to_boost(numpy.arange(0.0, 51.0))


def test_roocurve_error_band_must_be_ordered(rooplot):
    errors = rooplot["berr"]
    fX = errors.member("fX").copy()
    fX[1] = fX[3] + 0.5
    replace_member(errors, "fX", fX)
    with pytest.raises(ValueError, match="RooCurve x values are not increasing"):
        errors.interpolate_errors([1.0])


@pytest.fixture
def parameters():
    with uproot.open(skhep_testdata.data_path("uproot-issue243-new.root")) as file:
        yield file


def test_tparameter_boolean(parameters):
    parameter = parameters["isMC"]
    assert parameter.value is True
    assert bool(parameter) is True
    assert int(parameter) == 1
    assert float(parameter) == 1.0


def test_tparameter_integer(parameters):
    parameter = parameters["DSID"]
    assert parameter.value == 301488
    assert bool(parameter) is True
    assert int(parameter) == 301488
    assert operator.index(parameter) == 301488
    assert float(parameter) == 301488.0


def test_tparameter_floating(parameters):
    parameter = parameters["normalization"]
    assert parameter.value == 20000.0
    assert bool(parameter) is True
    assert int(parameter) == 20000
    assert float(parameter) == 20000.0


def test_th2poly(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = str(tmp_path / "th2poly.root")
    file = ROOT.TFile(path, "recreate")
    poly = ROOT.TH2Poly("poly", "title", 0.0, 4.0, -1.0, 1.0)
    for low, high in ((0.0, 2.0), (2.0, 4.0)):
        x = numpy.array([low, high, high, low])
        y = numpy.array([-1.0, -1.0, 1.0, 1.0])
        poly.AddBin(ROOT.TGraph(4, x, y))
    poly.Write()
    file.Close()

    with uproot.open(path) as opened:
        hist = opened["poly"]
        assert isinstance(hist, uproot.behaviors.TH2Poly.TH2Poly)
        assert hist.axes == (hist.member("fXaxis"), hist.member("fYaxis"))
        assert hist.axis(0).high == 4.0
        assert hist.axis("x") is hist.member("fXaxis")
        assert hist.axis(-2) is hist.member("fXaxis")
        assert hist.axis(1).low == -1.0
        assert hist.axis("y") is hist.member("fYaxis")
        assert hist.axis(-1) is hist.member("fYaxis")
        with pytest.raises(ValueError, match="axis must be 0"):
            hist.axis("z")
