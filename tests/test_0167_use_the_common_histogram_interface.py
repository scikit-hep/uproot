# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_axis():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert (
            f["hpx"].axes[0]
            == f["hpx"].axis(0)
            == f["hpx"].axis(-1)
            == f["hpx"].axis("x")
        )
        axis = f["hpx"].axis()
        assert len(axis) == 100
        assert axis[0] == (-4.0, -3.92)
        assert axis[1] == (-3.92, -3.84)
        assert [tuple(x) for x in axis][:3] == [
            (-4.0, -3.92),
            (-3.92, -3.84),
            (-3.84, -3.76),
        ]
        assert f["hpxpy"].axis(0) == f["hpxpy"].axis(1)
        assert axis.traits.circular is False
        assert axis.traits.discrete is False
        assert axis.low == -4
        assert axis.high == 4
        assert axis.width == 0.08
        assert axis.labels() is None
        assert axis.edges()[:5].tolist() == [-4.0, -3.92, -3.84, -3.76, -3.68]
        assert axis.edges(True)[:4].tolist() == [-numpy.inf, -4.0, -3.92, -3.84]
        assert axis.intervals()[:2].tolist() == [[-4.0, -3.92], [-3.92, -3.84]]
        assert axis.intervals(True)[:2].tolist() == [[-numpy.inf, -4.0], [-4.0, -3.92]]
        assert axis.centers()[:4].tolist() == [-3.96, -3.88, -3.8, -3.7199999999999998]
        assert axis.centers()[:3].tolist() == [-3.96, -3.88, -3.8]
        assert axis.centers(True)[:3].tolist() == [-numpy.inf, -3.96, -3.88]
        assert axis.widths()[:3].tolist() == [0.08, 0.08, 0.08]
        assert axis.widths(True)[:2].tolist() == [numpy.inf, 0.08000000000000007]
        assert (
            len(axis)
            == len(axis.edges()) - 1
            == len(axis.intervals())
            == len(axis.centers())
            == len(axis.widths())
        )
        assert (
            len(axis) + 2
            == len(axis.edges(flow=True)) - 1
            == len(axis.intervals(flow=True))
            == len(axis.centers(flow=True))
            == len(axis.widths(flow=True))
        )

    with uproot.open(skhep_testdata.data_path("uproot-issue33.root")) as f:
        assert f["cutflow"].axes[0] == f["cutflow"].axis(0) == f["cutflow"].axis("x")
        axis = f["cutflow"].axis()
        assert len(axis) == 7
        assert axis[0] == "Dijet"
        assert axis[1] == "MET"
        assert list(axis)[:3] == ["Dijet", "MET", "MuonVeto"]
        assert axis.traits.circular is False
        assert axis.traits.discrete is True
        assert axis.low == 0.0
        assert axis.high == 7.0
        assert axis.width == 1.0
        assert axis.labels()[:3] == ["Dijet", "MET", "MuonVeto"]
        assert list(axis) == axis.labels()
        assert axis.edges()[:5].tolist() == [0, 1, 2, 3, 4]
        assert axis.edges(True)[:4].tolist() == [-numpy.inf, 0, 1, 2]
        assert axis.intervals()[:2].tolist() == [[0, 1], [1, 2]]
        assert axis.intervals(True)[:2].tolist() == [[-numpy.inf, 0], [0, 1]]
        assert axis.centers()[:4].tolist() == [0.5, 1.5, 2.5, 3.5]
        assert axis.centers(True)[:3].tolist() == [-numpy.inf, 0.5, 1.5]
        assert axis.widths()[:3].tolist() == [1, 1, 1]
        assert axis.widths(True)[:3].tolist() == [numpy.inf, 1, 1]
        assert (
            len(axis.edges()) - 1
            == len(axis.intervals())
            == len(axis.centers())
            == len(axis.widths())
        )
        assert (
            len(axis.edges(flow=True)) - 1
            == len(axis.intervals(flow=True))
            == len(axis.centers(flow=True))
            == len(axis.widths(flow=True))
        )


def test_bins():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        hpx = f["hpx"]
        hpxpy = f["hpxpy"]
        hprof = f["hprof"]
        for hist in (hpx, hprof):
            assert hist.values().shape == hist.axis().centers().shape == (100,)
            assert (
                hist.values(flow=True).shape
                == hist.axis().centers(flow=True).shape
                == (102,)
            )
        assert (
            hpxpy.values().shape
            == (len(hpxpy.axis(0).centers()), len(hpxpy.axis(1).centers()))
            == (40, 40)
        )
        assert (
            hpxpy.values(flow=True).shape
            == (
                len(hpxpy.axis(0).centers(flow=True)),
                len(hpxpy.axis(1).centers(flow=True)),
            )
            == (42, 42)
        )
        # the histogram wasn't weighted
        assert numpy.array_equal(hpxpy.values(), hpxpy.variances())
        assert numpy.array_equal(hpxpy.values(flow=True), hpxpy.variances(flow=True))


def test_boost():
    boost_histogram = pytest.importorskip("boost_histogram")

    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        hpx = f["hpx"]
        hpxpy = f["hpxpy"]
        hprof = f["hprof"]

        assert hpx.to_boost() == boost_histogram.Histogram(hpx)

        assert hpx.to_boost().name == "hpx"
        assert hpx.to_boost().axes[0].name == "xaxis"

        assert hpxpy.to_boost().name == "hpxpy"
        assert hpxpy.to_boost().axes[0].name == "xaxis"
        assert hpxpy.to_boost().axes[1].name == "yaxis"

        assert hprof.to_boost().name == "hprof"
        assert hprof.to_boost().axes[0].name == "xaxis"


def test_boost_2():
    boost_histogram = pytest.importorskip("boost_histogram")

    with uproot.open(skhep_testdata.data_path("uproot-issue33.root")) as f:
        cutflow = f["cutflow"]
        converted = cutflow.to_boost()

    assert converted.name == "cutflow"
    assert isinstance(converted.storage_type(), boost_histogram.storage.Weight)
    assert isinstance(converted.axes[0], boost_histogram.axis.StrCategory)
    assert converted.axes[0].name == "xaxis"
    assert list(converted.axes[0]) == [
        "Dijet",
        "MET",
        "MuonVeto",
        "IsoMuonTrackVeto",
        "ElectronVeto",
        "IsoElectronTrackVeto",
        "IsoPionTrackVeto",
    ]
    assert converted.values().tolist() == [
        39551.0,
        27951.0,
        27911.0,
        27861.0,
        27737.0,
        27460.0,
        26751.0,
    ]
    assert converted.variances().tolist() == cutflow.variances().tolist()
    assert converted.variances()[0] == pytest.approx(39952.15079605)


def test_issue_0722():
    boost_histogram = pytest.importorskip("boost_histogram")

    with uproot.open(skhep_testdata.data_path("uproot-issue-722.root")) as f:
        hist = f["hist"]
        converted = hist.to_boost()

    # the histogram has 27 labelled bins, and many of the labels are repeated
    assert hist.axis().labels().count("550.0-800.0") == 9
    assert isinstance(converted.axes[0], boost_histogram.axis.StrCategory)
    assert list(converted.axes[0]) == hist.axis().labels()
    assert converted.values().tolist() == [0.0] * 27
    assert converted.variances()[:3].tolist() == [1144208.0, 590445.0, 244319.0]
    assert converted.variances().sum() == pytest.approx(hist.variances().sum())
