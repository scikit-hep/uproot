# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
Coverage of the TEfficiency support of #1822, which was merged after the coverage
tests of #1814 were split into PRs.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.behaviors.TEfficiency
import uproot.models.TEfficiency

TOTAL = np.array([0, 1, 5, 10, 20, 50], np.float64)
PASSED = np.array([0, 1, 0, 3, 20, 26], np.float64)
EDGES = np.arange(7, dtype=np.float64)

kByteCount = 0x40000000
kMemberwise = 0x4000


def efficiency(**kwargs):
    return uproot.writing.to_TEfficiency(
        "eff", "", (PASSED, EDGES), (TOTAL, EDGES), **kwargs
    )


# ---- intervals -----------------------------------------------------------------------
def test_mid_p_interpolates_a_fractional_passed_count():
    pytest.importorskip("scipy")
    mid_p = uproot.behaviors.TEfficiency.mid_p
    for upper in (False, True):
        # as TEfficiency::MidPInterval, between 0 and 1 passed events
        p0, p1, p = mid_p(10, [0.0, 1.0, 0.25], 0.9, upper)
        assert p == pytest.approx((p1 - p0) * 0.25 + p0)
        assert p0 < p < p1


def test_beta_shortest_interval_without_a_distribution():
    pytest.importorskip("scipy")
    # as TEfficiency::BetaShortestInterval, a non-positive parameter has no
    # distribution, and the interval is everything
    lower, upper = uproot.behaviors.TEfficiency.beta_shortest_interval(
        0.68, [0.0, 2.0, -1.0], [3.0, 0.0, 1.0]
    )
    assert lower.tolist() == [0.0, 0.0, 0.0]
    assert upper.tolist() == [1.0, 1.0, 1.0]


# ---- the behavior --------------------------------------------------------------------
def test_axes_of_the_total_histogram():
    eff = efficiency()
    assert eff.ndim == 1
    assert len(eff.axes) == 1
    assert eff.axes[0].edges().tolist() == EDGES.tolist()
    assert eff.axis().edges().tolist() == EDGES.tolist()
    assert eff.axis(0) is not None


def test_an_unknown_statistic_option_is_clopper_pearson():
    # TEfficiency::GetEfficiencyErrorLow falls back to Clopper-Pearson
    eff = efficiency()
    eff._members["fStatisticOption"] = 99
    assert eff.statistic_option == "kFCP"
    eff._members["fStatisticOption"] = -1
    assert eff.statistic_option == "kFCP"


# ---- to_TEfficiency ------------------------------------------------------------------
def test_arguments_that_are_not_histograms():
    with pytest.raises(
        TypeError, match=r"passed histogram .*\(not a profile\), not str$"
    ):
        uproot.writing.to_TEfficiency("eff", "", "abc", (TOTAL, EDGES))
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        profile = f["hprof"]
        with pytest.raises(
            TypeError, match=r"total histogram .*\(not a profile\), not Model_TProfile"
        ):
            uproot.writing.to_TEfficiency("eff", "", (PASSED, EDGES), profile)


def test_histograms_of_different_dimensions():
    two_d = (np.ones((6, 2)), EDGES, np.arange(3, dtype=np.float64))
    with pytest.raises(ValueError, match="same number of dimensions"):
        uproot.writing.to_TEfficiency("eff", "", (PASSED, EDGES), two_d)


@pytest.mark.parametrize("option", [-1, 9, 100])
def test_statistic_option_numbers_out_of_range(option):
    with pytest.raises(ValueError, match=f"statistic option {option}"):
        efficiency(fStatisticOption=option)


def test_statistic_option_as_a_number():
    assert efficiency(fStatisticOption=2).statistic_option == "kFWilson"


def test_bin_priors_as_a_structured_array():
    params = np.array(
        [(1.0, 2.0), (3.0, 4.0)], dtype=[("first", "f4"), ("second", "f4")]
    )
    eff = efficiency(fBeta_bin_params=params, fStatisticOption="kBBayesian")
    assert eff.beta_bin_params.dtype == uproot.models.TEfficiency._beta_bin_params_dtype
    assert eff.beta_bin_params.tolist() == [(1.0, 2.0), (3.0, 4.0)]
    # with priors per bin and a general Bayesian option, they are used
    assert eff.member("@fBits") & uproot.behaviors.TEfficiency.kUseBinPrior


@pytest.mark.parametrize("option", ["kBJeffrey", "kBUniform"])
def test_bin_priors_are_not_used_with_a_fixed_prior(option):
    eff = efficiency(fBeta_bin_params=[(1.0, 2.0)], fStatisticOption=option)
    assert not eff.member("@fBits") & uproot.behaviors.TEfficiency.kUseBinPrior


@pytest.mark.parametrize("use_weights", [True, False])
def test_explicit_use_weights(use_weights):
    eff = efficiency(use_weights=use_weights)
    assert eff.uses_weights == use_weights


# ---- the model -----------------------------------------------------------------------
def read_beta_bin_params(data):
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        chunk = uproot.source.chunk.Chunk.wrap(
            f.file.source, np.frombuffer(data, np.uint8)
        )
        cursor = uproot.source.cursor.Cursor(0)
        out = uproot.models.TEfficiency._read_beta_bin_params(
            chunk, cursor, {}, f.file, f.file.detached, None
        )
    assert cursor.index == len(data)
    return out.tolist()


def with_header(payload, version):
    return struct.pack(">IH", kByteCount | (len(payload) + 2), version) + payload


def test_objectwise_pairs_with_a_version():
    # a pair with a nonzero version has no checksum after its header
    pairs = struct.pack(">i", 2) + b"".join(
        with_header(struct.pack(">dd", a, b), 1) for a, b in [(1.0, 2.0), (3.0, 4.0)]
    )
    assert read_beta_bin_params(with_header(pairs, 9)) == [(1.0, 2.0), (3.0, 4.0)]


def test_memberwise_pairs_with_a_version():
    # a pair class with a nonzero version has no checksum after its version
    payload = (
        struct.pack(">h", 1)
        + struct.pack(">i", 2)
        + struct.pack(">2d", 1.0, 3.0)
        + struct.pack(">2d", 2.0, 4.0)
    )
    assert read_beta_bin_params(with_header(payload, 9 | kMemberwise)) == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]


def test_bin_priors_given_as_plain_pairs_are_written():
    out = []
    uproot.models.TEfficiency._serialize_beta_bin_params(out, [[1.0, 2.0], [3.0, 4.0]])
    assert read_beta_bin_params(b"".join(out)) == [(1.0, 2.0), (3.0, 4.0)]


def test_class_rawstreamers_without_histograms():
    eff = efficiency()
    with_histograms = eff.class_rawstreamers
    eff._members["fPassedHistogram"] = None
    eff._members["fTotalHistogram"] = None
    without = eff.class_rawstreamers
    assert uproot.models.TEfficiency._rawstreamer_TEfficiency_v2 in without
    assert set(without) < set(with_histograms)


def test_serialize_without_a_header():
    eff = efficiency()
    with_header_ = []
    eff._serialize(with_header_, True, "eff", np.uint32(0))
    without_header = []
    eff._serialize(without_header, False, "eff", np.uint32(0))
    # the header is the byte count and version 2
    assert b"".join(with_header_)[6:] == b"".join(without_header)
    assert b"".join(with_header_)[4:6] == b"\x00\x02"
