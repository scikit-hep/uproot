# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest

import uproot
import uproot.behaviors.TH1
import uproot.behaviors.TProfile
import uproot.writing.identify as identify


def axis(name="xaxis", fNbins=3, **kwargs):
    return identify.to_TAxis(
        fName=name, fTitle="", fNbins=fNbins, fXmin=0.0, fXmax=3.0, **kwargs
    )


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


def th2(sumw2=(), data=None, dtype="f8"):
    return identify.to_TH2x(
        fName="h2",
        fTitle="title",
        data=numpy.arange(25, dtype="f8") - 12 if data is None else data,
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


def th3(sumw2=(), data=None, dtype="f8"):
    return identify.to_TH3x(
        fName="h3",
        fTitle="title",
        data=numpy.arange(125, dtype="f8") - 62 if data is None else data,
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


def test_profile_equality():
    assert tprofile() == tprofile()
    assert tprofile() != tprofile(fSumw2=(0.0, 5.0, 21.0, 0.0, 0.0))
    assert tprofile2d() == tprofile2d()
    assert tprofile3d() == tprofile3d()


def test_th1_variances_without_and_with_sumw2():
    # without fSumw2, ROOT's errors are sqrt(|content|), also for negative bins
    hist = th1(data=(0, 1, -2, 3, 0))
    assert hist.variances().tolist() == [1.0, 2.0, 3.0]
    assert hist.variances(flow=True).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]  # cached
    assert th1(sumw2=(0, 7, 8, 9, 0)).variances().tolist() == [7.0, 8.0, 9.0]


@pytest.mark.parametrize("dtype", ["i1", "i2", "i4"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_variances_of_the_lowest_integer_content(ndim, dtype):
    # the absolute value of the lowest integer doesn't fit in its own dtype
    lowest = numpy.iinfo(dtype).min
    data = numpy.zeros(5**ndim, dtype=dtype)
    data[len(data) // 2] = lowest  # the central bin, which isn't a flow bin
    hist = {1: th1, 2: th2, 3: th3}[ndim](data=data, dtype=dtype)

    variances = hist.variances(flow=True)
    assert variances.dtype == numpy.dtype(numpy.float64)
    assert variances.min() == 0
    assert variances.max() == -float(lowest)
    assert hist.errors().max() == numpy.sqrt(-float(lowest))


def test_th2_values_and_variances():
    hist = th2()
    expected = numpy.transpose((numpy.arange(25) - 12.0).reshape(5, 5))
    assert numpy.array_equal(hist.values(flow=True), expected)

    # without fSumw2, the variances are |values|, as ROOT's errors are
    variances = numpy.abs(expected)
    assert numpy.array_equal(hist.variances(flow=True), variances)
    assert numpy.array_equal(hist.variances(), variances[1:-1, 1:-1])

    sumw2 = numpy.arange(25, dtype="f8")
    weighted = th2(sumw2=sumw2).variances(flow=True)
    assert numpy.array_equal(weighted, numpy.transpose(sumw2.reshape(5, 5)))


def test_th3_values_and_variances():
    hist = th3()
    expected = numpy.transpose((numpy.arange(125) - 62.0).reshape(5, 5, 5))
    assert numpy.array_equal(hist.values(flow=True), expected)
    assert numpy.array_equal(hist.values(), expected[1:-1, 1:-1, 1:-1])

    # without fSumw2, the variances are |values|, as ROOT's errors are
    variances = numpy.abs(expected)
    assert numpy.array_equal(hist.variances(flow=True), variances)
    assert numpy.array_equal(hist.variances(), variances[1:-1, 1:-1, 1:-1])

    sumw2 = numpy.arange(125, dtype="f8")
    weighted = th3(sumw2=sumw2).variances(flow=True)
    assert numpy.array_equal(weighted, numpy.transpose(sumw2.reshape(5, 5, 5)))

    numpy_values, (x, y, z) = hist.to_numpy(dd=True)
    assert numpy.array_equal(numpy_values, expected[1:-1, 1:-1, 1:-1])
    for edges in (x, y, z):
        assert edges.tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "make, index",
    [
        (tprofile, (slice(0, 2),)),
        (tprofile2d, (slice(0, 2), 0)),
        (tprofile3d, (slice(0, 2), 0, 0)),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_profile_values_and_errors(make, index):
    hist = make()

    # bins with sums of 3 and 8, sums of squares of 5 and 20, and 2 and 4 entries
    expected_values = [1.5, 2.0]
    spread = numpy.sqrt([5 / 2 - 1.5**2, 20 / 4 - 2.0**2])
    neff = numpy.array([2.0, 4.0])
    expected_errors = {
        "": spread / numpy.sqrt(neff),
        "s": spread,
        "i": spread / numpy.sqrt(neff),
        "g": 1.0 / numpy.sqrt(neff),
    }

    assert hist.values()[index].tolist() == expected_values
    # the variances behind histogram equality are the squared errors
    values, variances = hist._values_variances(flow=False)
    assert values[index].tolist() == expected_values
    assert variances[index] == pytest.approx(expected_errors[""] ** 2)
    for mode, errors in expected_errors.items():
        assert hist.errors(error_mode=mode)[index] == pytest.approx(errors)
        assert hist.variances(error_mode=mode)[index] == pytest.approx(errors**2)
        # the second call comes from the object's cache
        assert hist.errors(error_mode=mode)[index] == pytest.approx(errors)

    flow = tuple(
        x + 1 if isinstance(x, int) else slice(x.start + 1, x.stop + 1) for x in index
    )
    assert hist.values(flow=True)[flow].tolist() == expected_values
    assert hist.errors(flow=True)[flow] == pytest.approx(expected_errors[""])
    assert hist.values(flow=True).shape == (5,) * len(hist.axes)
    assert hist.values().shape == (3,) * len(hist.axes)
    assert hist.errors().shape == (3,) * len(hist.axes)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_errors_without_sumw2_match_root(tmp_path, ndim):
    ROOT = pytest.importorskip("ROOT")

    path = tmp_path / "negative.root"
    root_file = ROOT.TFile.Open(str(path), "RECREATE")
    hist = getattr(ROOT, f"TH{ndim}D")("h", "", *([2, 0.0, 2.0] * ndim))
    for cell in range(hist.GetNcells()):
        hist.SetBinContent(cell, (cell % 5) - 2.0)  # negative, zero and positive
    assert hist.GetSumw2N() == 0
    expected = [hist.GetBinError(cell) ** 2 for cell in range(hist.GetNcells())]
    hist.Write()
    root_file.Close()

    with uproot.open(path) as file:
        variances = file["h"].variances(flow=True)
    # uproot's axes are x, y, z; ROOT's cells run over x fastest
    assert numpy.transpose(variances).reshape(-1).tolist() == pytest.approx(expected)
