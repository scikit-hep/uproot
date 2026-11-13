# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

# Reference values computed with ROOT 6.40 (TEfficiency::GetEfficiency,
# GetEfficiencyErrorLow, GetEfficiencyErrorUp) for the histograms below, with
# confidence level 0.9. Keys are (statistic option, posterior mode, shortest
# interval); values are (efficiency, error low, error up) for bins 1-6.
TOTAL = np.array([0, 1, 5, 10, 20, 50], np.float64)
PASSED = np.array([0, 1, 0, 3, 20, 26], np.float64)
EDGES = np.arange(7, dtype=np.float64)

REFERENCE = {
    (0, False, False): (
        [0, 1, 0, 0.3, 1, 0.52],
        [0, 0.95, 0, 0.212735566086, 0.139108340668, 0.124587964731],
        [1, 0, 0.450719728347, 0.306624216105, 0, 0.122741209883],
    ),
    (1, False, False): (
        [0, 1, 0, 0.3, 1, 0.52],
        [0, 0, 0, 0.238361936005, 0, 0.116215631147],
        [1, 0, 0, 0.238361936005, 0, 0.116215631147],
    ),
    (2, False, False): (
        [0, 1, 0, 0.3, 1, 0.52],
        [0, 0.730134051216, 0, 0.173123416097, 0.119157837361, 0.114224796256],
        [1, 0, 0.351116500765, 0.25830020413, 0, 0.112171468825],
    ),
    (3, False, False): (
        [0, 1, 0, 0.3, 1, 0.52],
        [0, 0.776454956415, 0, 0.176407332637, 0.14128771121, 0.1142292129],
        [1, 0, 0.400990596051, 0.26158412067, 0, 0.112175885469],
    ),
    (4, False, False): (
        [0, 1, 0, 0.3, 1, 0.52],
        [0, 0.9, 0, 0.1845472252, 0.1154527748, 0.122080206387],
        [1, 0, 0.379272491671, 0.316874317639, 0, 0.123046761863],
    ),
    (5, False, False): (
        [0.5, 0.75, 0.0833333333333, 0.318181818182, 0.97619047619, 0.519607843137],
        [
            0.493844170298,
            0.521480186194,
            0.0829593363055,
            0.200852441502,
            0.0666669027283,
            0.114545621786,
        ],
        [
            0.493844170298,
            0.248457080697,
            0.222412251356,
            0.239944948303,
            0.0237124460509,
            0.113662228297,
        ],
    ),
    (6, False, False): (
        [
            0.5,
            0.666666666667,
            0.142857142857,
            0.333333333333,
            0.954545454545,
            0.519230769231,
        ],
        [
            0.45,
            0.443059868917,
            0.134344698246,
            0.198257860414,
            0.087491365572,
            0.113448209755,
        ],
        [
            0.45,
            0.308012767814,
            0.25018062614,
            0.231040854956,
            0.0430149881949,
            0.112598631147,
        ],
    ),
    (7, False, False): (
        [0.4, 0.5, 0.2, 0.333333333333, 0.88, 0.509090909091],
        [
            0.302388537114,
            0.310744622562,
            0.158976832493,
            0.180615710949,
            0.119801016129,
            0.110202095269,
        ],
        [
            0.35139537427,
            0.310744622562,
            0.229135547031,
            0.206672113191,
            0.0850466651663,
            0.109822586546,
        ],
    ),
    (8, False, False): (
        [0, 1, 0, 0.3, 1, 0.52],
        [0, 0.9, 0, 0.192555202357, 0.108749062754, 0.11605969768],
        [1, 0, 0.369042656384, 0.270574824698, 0, 0.114382962249],
    ),
    (7, True, False): (
        [0.333333333333, 0.5, 0.125, 0.307692307692, 0.913043478261, 0.509433962264],
        [
            0.265431546853,
            0.310744622698,
            0.109570205532,
            0.169086986655,
            0.127398742479,
            0.110319304132,
        ],
        [
            0.378857947997,
            0.310744622427,
            0.2511057189,
            0.214636107649,
            0.0656050096489,
            0.109704134021,
        ],
    ),
    (5, False, True): (
        [0.5, 0.75, 0.0833333333333, 0.318181818182, 0.97619047619, 0.519607843137],
        [
            0.493844170298,
            0.398642801974,
            0.0833333333333,
            0.221302056629,
            0.0408024789833,
            0.114017607404,
        ],
        [
            0.493844170298,
            0.25,
            0.143391195296,
            0.212170872922,
            0.0238095238095,
            0.114183734527,
        ],
    ),
}


@pytest.mark.parametrize("key", list(REFERENCE))
def test_intervals_match_root(key):
    pytest.importorskip("scipy")
    option, posterior_mode, shortest_interval = key
    eff = uproot.writing.to_TEfficiency(
        "eff",
        "",
        (PASSED, EDGES),
        (TOTAL, EDGES),
        fStatisticOption=option,
        fConfLevel=0.9,
        fBeta_alpha=2.0,
        fBeta_beta=3.0,
        posterior_mode=posterior_mode,
        shortest_interval=shortest_interval,
    )
    expected_values, expected_low, expected_high = REFERENCE[key]
    low, high = eff.errors()
    assert eff.values() == pytest.approx(expected_values, abs=1e-7)
    assert low == pytest.approx(expected_low, abs=1e-7)
    assert high == pytest.approx(expected_high, abs=1e-7)

    lower, upper = eff.intervals()
    assert lower == pytest.approx(eff.values() - low)
    assert upper == pytest.approx(eff.values() + high)


def test_read_with_pair_streamer():
    pytest.importorskip("scipy")
    with uproot.open(skhep_testdata.data_path("uproot-issue38c.root")) as f:
        eff = f["TEfficiencyName"]
        assert eff.statistic_option == "kFCP"
        assert eff.confidence_level == pytest.approx(0.682689492137)
        assert eff.passed.classname == "TH1F"
        assert eff.total.classname == "TH1F"
        assert eff.ndim == 1
        assert eff.values() == pytest.approx([0.5] * 10 + [0])
        low, high = eff.errors()
        expected = [
            0.314698893873,
            0.220457375638,
            0.150355854439,
            0.102132876997,
            0.0696995964941,
            0.0478967520932,
            0.0331343600714,
            0.0230501441233,
            0.0161051715799,
            0.0112898668505,
        ]
        assert low == pytest.approx([*expected, 0])
        assert high == pytest.approx([*expected, 1])


def test_read_without_pair_streamer(reset_classes):
    pytest.importorskip("scipy")
    with uproot.open(skhep_testdata.data_path("uproot-issue209.root")) as f:
        eff = f["TEfficiencyName"]
        assert eff.name == "TEfficiencyName"
        assert eff.title == "TEfficiencyTitle"
        assert eff.values()[:4] == pytest.approx(
            [0.196311639717, 0.346395156852, 0.33506189361, 0.312722103767]
        )
        low, high = eff.errors()
        assert low[:3] == pytest.approx(
            [0.00461720870276, 0.00502975589161, 0.00616004335835]
        )
        assert high[11:13] == pytest.approx([1, 0.841344746068])
        assert eff.values(flow=True).shape == (22,)


def test_write_roundtrip(tmp_path):
    filename = os.path.join(tmp_path, "eff.root")
    with uproot.open(skhep_testdata.data_path("uproot-issue38c.root")) as f:
        original = f["TEfficiencyName"]
        with uproot.recreate(filename) as out:
            out["eff"] = original

        with uproot.open(filename) as f2:
            eff = f2["eff"]
            assert eff.classname == "TEfficiency"
            assert eff.name == "eff"
            assert eff.passed.name == "eff_passed"
            assert eff.total.name == "eff_total"
            for member in (
                "fBeta_alpha",
                "fBeta_beta",
                "fConfLevel",
                "fStatisticOption",
                "fWeight",
            ):
                assert eff.member(member) == original.member(member)
            assert eff.member("@fBits") & 0x7C000 == original.member("@fBits") & 0x7C000
            assert eff.beta_bin_params.tolist() == original.beta_bin_params.tolist()
            assert np.array_equal(
                eff.passed.values(flow=True), original.passed.values(flow=True)
            )
            assert np.array_equal(
                eff.total.values(flow=True), original.total.values(flow=True)
            )
            assert np.array_equal(
                eff.total.axis().edges(), original.total.axis().edges()
            )


def test_to_TEfficiency_flags(tmp_path):
    hist = pytest.importorskip("hist")
    pytest.importorskip("scipy")

    rng = np.random.default_rng(12345)
    x = rng.uniform(0, 10, 500)
    y = rng.uniform(0, 5, 500)
    w = rng.uniform(0.5, 2.0, 500)
    accepted = rng.uniform(size=500) < x / 10

    def make(mask):
        return (
            hist.Hist.new.Var([0, 1, 3, 6, 10], name="x")
            .Reg(3, 0, 5, name="y")
            .Weight()
            .fill(x[mask], y[mask], weight=w[mask])
        )

    passed, total = make(accepted), make(slice(None))

    filename = os.path.join(tmp_path, "eff.root")
    with uproot.recreate(filename) as f:
        f["eff"] = uproot.writing.to_TEfficiency(
            "eff",
            "title",
            passed,
            total,
            fStatisticOption="kBJeffrey",
            fBeta_bin_params=[(1.0, 2.0), (3.0, 4.0)],
            posterior_mode=True,
        )

    with uproot.open(filename) as f:
        eff = f["eff"]
        assert eff.ndim == 2
        assert eff.statistic_option == "kBJeffrey"
        assert eff.uses_weights
        assert eff.uses_bayesian_stat
        assert eff.uses_posterior_mode
        assert eff.uses_shortest_interval
        # kBJeffrey fixes the prior to Beta(0.5, 0.5) and disables per-bin priors
        assert (eff.beta_alpha, eff.beta_beta) == (0.5, 0.5)
        assert eff.values().shape == (4, 3)
        assert np.array_equal(eff.passed.values(), passed.values())
        assert np.array_equal(eff.total.variances(), total.variances())
        low, high = eff.errors()
        assert np.all(low >= 0) and np.all(high >= 0)
        assert np.all(eff.values() - low >= 0) and np.all(eff.values() + high <= 1)


def test_to_TEfficiency_invalid():
    with pytest.raises(ValueError, match="same binning"):
        uproot.writing.to_TEfficiency("eff", "", (PASSED, EDGES), (TOTAL, EDGES * 2))
    with pytest.raises(ValueError, match="statistic option"):
        uproot.writing.to_TEfficiency(
            "eff", "", (PASSED, EDGES), (TOTAL, EDGES), fStatisticOption="kNope"
        )


def test_pyroot_reads_uproot_written(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    pytest.importorskip("scipy")

    filename = os.path.join(tmp_path, "eff.root")
    with uproot.recreate(filename) as f:
        for option in range(9):
            f[f"eff{option}"] = uproot.writing.to_TEfficiency(
                f"eff{option}",
                "",
                (PASSED, EDGES),
                (TOTAL, EDGES),
                fStatisticOption=option,
                fConfLevel=0.9,
                fBeta_alpha=2.0,
                fBeta_beta=3.0,
            )

    rootfile = ROOT.TFile(filename)
    try:
        for option in range(9):
            eff = rootfile.Get(f"eff{option}")
            ROOT.SetOwnership(eff, False)  # owned by (and deleted with) the file
            assert eff.ClassName() == "TEfficiency"
            assert eff.GetStatisticOption() == option
            # ROOT does not restore the interval function when reading
            eff.SetStatisticOption(eff.GetStatisticOption())
            expected = REFERENCE[(option, False, False)]
            for i in range(6):
                assert eff.GetEfficiency(i + 1) == pytest.approx(
                    expected[0][i], abs=1e-7
                )
                assert eff.GetEfficiencyErrorLow(i + 1) == pytest.approx(
                    expected[1][i], abs=1e-7
                )
                assert eff.GetEfficiencyErrorUp(i + 1) == pytest.approx(
                    expected[2][i], abs=1e-7
                )
    finally:
        rootfile.Close()


def test_uproot_reads_pyroot_written(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    pytest.importorskip("scipy")

    filename = os.path.join(tmp_path, "eff.root")
    rootfile = ROOT.TFile(filename, "recreate")
    eff = ROOT.TEfficiency("eff", "", 6, 0, 6)
    ROOT.SetOwnership(eff, False)  # owned by (and deleted with) the file
    for i, (total, passed) in enumerate(zip(TOTAL, PASSED)):
        for j in range(int(total)):
            eff.Fill(bool(j < passed), i + 0.5)
    eff.SetConfidenceLevel(0.9)
    eff.SetBetaAlpha(2.0)
    eff.SetBetaBeta(3.0)
    eff.SetStatisticOption(ROOT.TEfficiency.kBBayesian)
    eff.SetPosteriorMode()
    eff.Write()
    rootfile.Close()

    with uproot.open(filename) as f:
        eff = f["eff"]
        expected = REFERENCE[(7, True, False)]
        low, high = eff.errors()
        assert eff.values() == pytest.approx(expected[0], abs=1e-7)
        assert low == pytest.approx(expected[1], abs=1e-7)
        assert high == pytest.approx(expected[2], abs=1e-7)


# Feldman-Cousins bounds from ROOT 6.40 (TEfficiency::FeldmanCousins) at
# confidence level 0.683, for totals large enough that the binomial
# coefficients overflow if they are not computed in log space
FELDMAN_COUSINS_LARGE = {
    (1030, 515): (0.48397509288, 0.51602490712),
    (2000, 1000): (0.488748100586, 0.511251899414),
    (2000, 3): (0.000551928766072, 0.00265088025481),
    (10000, 9990): (0.998619371094, 0.999322344549),
}


@pytest.mark.parametrize("total,passed", list(FELDMAN_COUSINS_LARGE))
def test_feldman_cousins_large_total(total, passed):
    pytest.importorskip("scipy")
    from uproot.behaviors.TEfficiency import feldman_cousins

    lower, upper = FELDMAN_COUSINS_LARGE[(total, passed)]
    assert feldman_cousins(total, passed, 0.683, False) == pytest.approx(
        lower, abs=1e-8
    )
    assert feldman_cousins(total, passed, 0.683, True) == pytest.approx(upper, abs=1e-8)


def test_feldman_cousins_large_total_efficiency(tmp_path):
    pytest.importorskip("scipy")

    total = np.array([2000, 1030, 2000], np.float64)
    passed = np.array([1000, 515, 1000], np.float64)
    edges = np.arange(4, dtype=np.float64)
    filename = os.path.join(tmp_path, "eff.root")
    with uproot.recreate(filename) as f:
        f["eff"] = uproot.writing.to_TEfficiency(
            "eff",
            "",
            (passed, edges),
            (total, edges),
            fStatisticOption="kFFC",
            fConfLevel=0.683,
        )
    with uproot.open(filename) as f:
        lower, upper = f["eff"].intervals()
    expected = [FELDMAN_COUSINS_LARGE[(int(t), int(p))] for t, p in zip(total, passed)]
    assert lower == pytest.approx([x[0] for x in expected], abs=1e-8)
    assert upper == pytest.approx([x[1] for x in expected], abs=1e-8)


def test_to_TEfficiency_exported():
    assert uproot.writing.to_TEfficiency is uproot.writing.identify.to_TEfficiency


def test_histogram_names_without_name():
    eff = uproot.writing.to_TEfficiency(None, "", (PASSED, EDGES), (TOTAL, EDGES))
    out = []
    eff._serialize(out, True, None, np.uint32(0))
    data = b"".join(out)
    assert b"None_passed" not in data
    assert b"None_total" not in data
    assert b"\x06passed" in data
    assert b"\x05total" in data

    out = []
    eff._serialize(out, True, "myeff", np.uint32(0))
    data = b"".join(out)
    assert b"\x0cmyeff_passed" in data
    assert b"\x0bmyeff_total" in data


def test_uproot_reads_pyroot_written_objectwise(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filename = os.path.join(tmp_path, "eff.root")
    ROOT.TVirtualStreamerInfo.SetStreamMemberWise(False)
    try:
        rootfile = ROOT.TFile(filename, "recreate")
        eff = ROOT.TEfficiency("eff", "", 6, 0, 6)
        ROOT.SetOwnership(eff, False)  # owned by (and deleted with) the file
        eff.SetBetaBinParameters(1, 2.0, 3.0)
        eff.SetBetaBinParameters(4, 5.0, 6.0)
        eff.Write()
        rootfile.Close()
    finally:
        ROOT.TVirtualStreamerInfo.SetStreamMemberWise(True)

    with uproot.open(filename) as f:
        params = f["eff"].member("fBeta_bin_params")
    assert params.tolist() == [
        (1.0, 1.0),
        (2.0, 3.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (5.0, 6.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    ]
