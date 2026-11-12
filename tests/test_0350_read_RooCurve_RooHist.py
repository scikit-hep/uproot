# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

boost_histogram = pytest.importorskip("boost_histogram")


@pytest.fixture(scope="module")
def datafile(tmpdir_factory):
    yield skhep_testdata.data_path("uproot-issue-350.root")


@pytest.fixture
def roohist(datafile):
    with uproot.open(datafile) as f:
        yield f["data"]


@pytest.fixture
def roocurve(datafile):
    with uproot.open(datafile) as f:
        yield f["bhist"]


@pytest.fixture
def roocurve_err(datafile):
    with uproot.open(datafile) as f:
        yield f["berr"]


def test_interpretation(roohist, roocurve, roocurve_err):
    assert roohist.classname == "RooHist"
    assert roohist.behaviors[0] == uproot.behaviors.RooHist.RooHist

    assert roocurve.classname == "RooCurve"
    assert roocurve.behaviors[0] == uproot.behaviors.RooCurve.RooCurve
    assert roocurve.curve_type == "VALUES"

    assert roocurve_err.classname == "RooCurve"
    assert roocurve_err.behaviors[0] == uproot.behaviors.RooCurve.RooCurve
    assert roocurve_err.curve_type == "ERRORS"


def test_to_boost(roohist, roocurve, roocurve_err):
    rh_boost = roohist.to_boost()
    assert rh_boost.axes[0].edges == pytest.approx(numpy.arange(0.0, 51.0))
    assert rh_boost.values() == pytest.approx(
        numpy.array(
            [
                33071.0,
                31911.0,
                30343.0,
                29856.0,
                28536.0,
                28121.0,
                26896.0,
                25812.0,
                25110.0,
                24471.0,
                23591.0,
                22420.0,
                22183.0,
                21280.0,
                20292.0,
                20049.0,
                19577.0,
                19113.0,
                18099.0,
                18021.0,
                17336.0,
                16917.0,
                16581.0,
                16087.0,
                15591.0,
                15110.0,
                14822.0,
                14018.0,
                13503.0,
                12897.0,
                12236.0,
                11986.0,
                11495.0,
                10911.0,
                10658.0,
                10365.0,
                9892.0,
                9704.0,
                9421.0,
                8902.0,
                8459.0,
                8207.0,
                8313.0,
                7540.0,
                7670.0,
                7139.0,
                6930.0,
                6952.0,
                6541.0,
                6509.0,
            ]
        )
    )

    rc_boost = roocurve.to_boost(rh_boost.axes[0].edges)
    assert (rc_boost.axes[0].edges == rh_boost.axes[0].edges).all()
    if hasattr(rc_boost, "storage_type"):
        assert rc_boost.storage_type == boost_histogram.storage.Double
    else:
        assert rc_boost._storage_type == boost_histogram.storage.Double

    rc_boost = roocurve.to_boost(rh_boost.axes[0].edges, error_curve=roocurve_err)
    assert (rc_boost.axes[0].edges == rh_boost.axes[0].edges).all()
    if hasattr(rc_boost, "storage_type"):
        assert rc_boost.storage_type == boost_histogram.storage.Weight
    else:
        assert rc_boost._storage_type == boost_histogram.storage.Weight


def test_interpolate(roocurve, roocurve_err):
    x = numpy.linspace(0.0, 50.0, 25)
    assert roocurve.interpolate(x) == pytest.approx(
        numpy.array(
            [
                0.0,
                30702.48387239,
                28876.37922894,
                27216.88435277,
                25410.15503657,
                23870.96973243,
                22431.78768639,
                20494.32470025,
                19711.55585783,
                18044.31928269,
                17038.95096061,
                15978.6456956,
                14571.88021957,
                13407.14686783,
                12542.6196236,
                11851.2158329,
                10917.3313527,
                10461.60283896,
                9810.5979089,
                9011.93796946,
                8310.6710814,
                7635.42636501,
                7231.40126899,
                7034.03153905,
                0.0,
            ]
        )
    )
    assert roocurve_err.interpolate_errors(x) == pytest.approx(
        numpy.array(
            [
                0.0,
                171.78121867,
                166.75282816,
                161.99516719,
                156.62312969,
                151.88614908,
                147.32328378,
                140.96278754,
                138.46663782,
                133.11834307,
                130.23098947,
                127.28119047,
                122.37485039,
                116.92086339,
                112.14695174,
                108.09250035,
                103.11762571,
                100.45395025,
                97.13368841,
                92.94880493,
                89.132786,
                85.30179688,
                82.91622731,
                81.70937157,
                0.0,
            ]
        )
    )
    yup, ydown = roocurve_err.interpolate_asymm_errors(x)
    assert yup == pytest.approx(
        numpy.array(
            [
                0.0,
                30874.26509106,
                29043.13205709,
                27378.87951996,
                25566.77816627,
                24022.85588151,
                22579.11097017,
                20635.28748779,
                19850.02249566,
                18177.43762576,
                17169.18195008,
                16105.92688607,
                14694.25506996,
                13524.06773121,
                12654.76657534,
                11959.30833324,
                11020.4489784,
                10562.05678921,
                9907.73159731,
                9104.88677439,
                8399.8038674,
                7720.72816188,
                7314.3174963,
                7115.74091062,
                0.0,
            ]
        )
    )
    assert ydown == pytest.approx(
        numpy.array(
            [
                0.0,
                30530.70265372,
                28709.62640078,
                27054.88918558,
                25253.53190688,
                23719.08358336,
                22284.46440262,
                20353.36191271,
                19573.08922001,
                17911.20093961,
                16908.71997115,
                15851.36450513,
                14449.50536918,
                13290.22600444,
                12430.47267185,
                11743.12333255,
                10814.21372699,
                10361.14888871,
                9713.46422048,
                8918.98916453,
                8221.5382954,
                7550.12456813,
                7148.48504169,
                6952.32216748,
                0.0,
            ]
        )
    )
