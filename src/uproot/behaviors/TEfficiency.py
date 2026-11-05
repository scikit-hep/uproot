# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``TEfficiency``.

The efficiencies and their uncertainties are computed the same way as ROOT's
``TEfficiency::GetEfficiency``, ``TEfficiency::GetEfficiencyErrorLow``, and
``TEfficiency::GetEfficiencyErrorUp``, for all of ROOT's statistic options.
Computing uncertainties requires SciPy.
"""

from __future__ import annotations

import numpy

import uproot
import uproot.extras

# TEfficiency::EStatOption
statistic_options = (
    "kFCP",
    "kFNormal",
    "kFWilson",
    "kFAC",
    "kFFC",
    "kBJeffrey",
    "kBUniform",
    "kBBayesian",
    "kMidP",
)

# TEfficiency::EStatusBits (stored in TObject::fBits)
kIsBayesian = 1 << 14
kPosteriorMode = 1 << 15
kShortestInterval = 1 << 16
kUseBinPrior = 1 << 17
kUseWeights = 1 << 18


def _special():
    return uproot.extras.scipy().special


def _normal_quantile(p, sigma):
    return sigma * _special().ndtri(p)


def _beta_quantile(p, a, b):
    return _special().betaincinv(a, b, p)


def _beta_quantile_c(p, a, b):
    return _special().betainccinv(a, b, p)


def clopper_pearson(total, passed, level, upper):
    """
    Clopper-Pearson interval boundary, as ``TEfficiency::ClopperPearson``.
    """
    total, passed = numpy.broadcast_arrays(
        numpy.asarray(total, numpy.float64), numpy.asarray(passed, numpy.float64)
    )
    alpha = (1.0 - level) / 2
    with numpy.errstate(all="ignore"):
        if upper:
            return numpy.where(
                passed == total,
                1.0,
                _beta_quantile(1 - alpha, passed + 1, total - passed),
            )
        else:
            return numpy.where(
                passed == 0, 0.0, _beta_quantile(alpha, passed, total - passed + 1.0)
            )


def normal(total, passed, level, upper):
    """
    Normal-approximation interval boundary, as ``TEfficiency::Normal``.
    """
    total, passed = numpy.broadcast_arrays(
        numpy.asarray(total, numpy.float64), numpy.asarray(passed, numpy.float64)
    )
    alpha = (1.0 - level) / 2
    with numpy.errstate(all="ignore"):
        average = passed / total
        sigma = numpy.sqrt(average * (1 - average) / total)
        delta = _normal_quantile(1 - alpha, sigma)
        if upper:
            out = numpy.where(average + delta > 1, 1.0, average + delta)
        else:
            out = numpy.where(average - delta < 0, 0.0, average - delta)
    return numpy.where(total == 0, 1.0 if upper else 0.0, out)


def wilson(total, passed, level, upper):
    """
    Wilson interval boundary, as ``TEfficiency::Wilson``.
    """
    total, passed = numpy.broadcast_arrays(
        numpy.asarray(total, numpy.float64), numpy.asarray(passed, numpy.float64)
    )
    alpha = (1.0 - level) / 2
    with numpy.errstate(all="ignore"):
        average = passed / total
        kappa = _normal_quantile(1 - alpha, 1)
        mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa)
        delta = (
            kappa
            / (total + kappa * kappa)
            * numpy.sqrt(total * average * (1 - average) + kappa * kappa / 4)
        )
        if upper:
            out = numpy.where(mode + delta > 1, 1.0, mode + delta)
        else:
            out = numpy.where(mode - delta < 0, 0.0, mode - delta)
    return numpy.where(total == 0, 1.0 if upper else 0.0, out)


def agresti_coull(total, passed, level, upper):
    """
    Agresti-Coull interval boundary, as ``TEfficiency::AgrestiCoull``.
    """
    total, passed = numpy.broadcast_arrays(
        numpy.asarray(total, numpy.float64), numpy.asarray(passed, numpy.float64)
    )
    alpha = (1.0 - level) / 2
    kappa = _normal_quantile(1 - alpha, 1)
    with numpy.errstate(all="ignore"):
        mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa)
        delta = kappa * numpy.sqrt(mode * (1 - mode) / (total + kappa * kappa))
        if upper:
            return numpy.where(mode + delta > 1, 1.0, mode + delta)
        else:
            return numpy.where(mode - delta < 0, 0.0, mode - delta)


def _feldman_cousins_rho_set(rho, ntot, target):
    special = _special()
    x = numpy.arange(ntot + 1, dtype=numpy.float64)
    prob = special.binom(ntot, x) * rho**x * (1 - rho) ** (ntot - x)
    with numpy.errstate(all="ignore"):
        rho_hat = x / ntot
        term1 = (rho / rho_hat) ** x
        term2 = ((1 - rho) / (1 - rho_hat)) ** (ntot - x)
        lratio = numpy.where((term1 == 0) | (term2 == 0), 0.0, term1 * term2)
    lratio[0] = (1 - rho) ** ntot
    lratio[ntot] = rho**ntot
    order = numpy.argsort(-lratio, kind="stable")
    cumulative = numpy.cumsum(prob[order])
    # add terms (in order of decreasing likelihood ratio) until 'target' is reached
    num = numpy.searchsorted(cumulative, target, side="left") + 1
    accepted = order[:num]
    return accepted.min(), accepted.max()


def _feldman_cousins_scalar(total, passed, level):
    ntot = int(total)
    target = level
    tol = 1e-9

    rho_min, rho_max, rho = 0.0, 1.0, 0.0
    while abs(rho_max - rho_min) > tol:
        rho = (rho_min + rho_max) / 2
        _, x_r = _feldman_cousins_rho_set(rho, ntot, target)
        if x_r < passed:
            rho_min = rho
        else:
            rho_max = rho
    lower = rho

    rho_min, rho_max, rho = 0.0, 1.0, 0.0
    while abs(rho_max - rho_min) > tol:
        rho = (rho_min + rho_max) / 2
        x_l, _ = _feldman_cousins_rho_set(rho, ntot, target)
        if x_l > passed:
            rho_max = rho
        else:
            rho_min = rho
    upper = rho

    return lower, upper


def feldman_cousins(total, passed, level, upper):
    """
    Feldman-Cousins interval boundary, as ``TEfficiency::FeldmanCousins``.
    """
    total, passed = numpy.broadcast_arrays(
        numpy.asarray(total, numpy.float64), numpy.asarray(passed, numpy.float64)
    )
    out = numpy.empty(total.shape, numpy.float64)
    for index in numpy.ndindex(total.shape):
        lo, hi = _feldman_cousins_scalar(total[index], passed[index], level)
        out[index] = hi if upper else lo
    return out


def _mid_p_scalar(total, passed, level, upper):
    special = _special()
    alpha = 1.0 - level
    alpha_min = alpha / 2
    tol = 1e-9

    if 0 < passed < 1:
        p0 = _mid_p_scalar(total, 0.0, level, upper)
        p1 = _mid_p_scalar(total, 1.0, level, upper)
        return (p1 - p0) * passed + p0

    a, b = passed + 1.0, total - passed + 1.0
    pmin, pmax, p = 0.0, 1.0, 0.0
    while abs(pmax - pmin) > tol:
        p = (pmin + pmax) / 2
        # 0.5 * beta_pdf(p, passed + 1, total - passed + 1) / (total + 1)
        v = 0.5 * numpy.exp(
            (a - 1) * numpy.log(p) + (b - 1) * numpy.log1p(-p) - special.betaln(a, b)
        )
        v /= total + 1
        if passed - 1 >= 0:
            v += special.betaincc(passed, total - passed + 1, p)
        vmin = alpha_min if upper else 1.0 - alpha_min
        if v > vmin:
            pmin = p
        else:
            pmax = p
    return p


def mid_p(total, passed, level, upper):
    """
    Mid-P Lancaster interval boundary, as ``TEfficiency::MidPInterval``.
    """
    total, passed = numpy.broadcast_arrays(
        numpy.asarray(total, numpy.float64), numpy.asarray(passed, numpy.float64)
    )
    out = numpy.empty(total.shape, numpy.float64)
    for index in numpy.ndindex(total.shape):
        out[index] = _mid_p_scalar(total[index], passed[index], level, upper)
    return out


_frequentist_boundaries = {
    0: clopper_pearson,
    1: normal,
    2: wilson,
    3: agresti_coull,
    4: feldman_cousins,
    8: mid_p,
}


def beta_mean(a, b):
    """
    Mean of the beta distribution, as ``TEfficiency::BetaMean``.
    """
    a, b = numpy.broadcast_arrays(
        numpy.asarray(a, numpy.float64), numpy.asarray(b, numpy.float64)
    )
    with numpy.errstate(all="ignore"):
        return numpy.where((a <= 0) | (b <= 0), 0.0, a / (a + b))


def beta_mode(a, b):
    """
    Mode of the beta distribution, as ``TEfficiency::BetaMode``.
    """
    a, b = numpy.broadcast_arrays(
        numpy.asarray(a, numpy.float64), numpy.asarray(b, numpy.float64)
    )
    with numpy.errstate(all="ignore"):
        out = (a - 1.0) / (a + b - 2.0)
    small = (a <= 1) | (b <= 1)
    out = numpy.where(small & (a < b), 0.0, out)
    out = numpy.where(small & (a > b), 1.0, out)
    out = numpy.where(small & (a == b), 0.5, out)
    return numpy.where((a <= 0) | (b <= 0), 0.0, out)


def beta_central_interval(level, a, b, upper):
    """
    Boundary of the central interval of the beta distribution, as
    ``TEfficiency::BetaCentralInterval``.
    """
    a, b = numpy.broadcast_arrays(
        numpy.asarray(a, numpy.float64), numpy.asarray(b, numpy.float64)
    )
    valid = (a > 0) & (b > 0)
    with numpy.errstate(all="ignore"):
        if upper:
            return numpy.where(valid, _beta_quantile((1 + level) / 2, a, b), 1.0)
        else:
            return numpy.where(valid, _beta_quantile((1 - level) / 2, a, b), 0.0)


def _beta_shortest_interval_scalar(level, a, b):
    scipy = uproot.extras.scipy()

    if not (numpy.isfinite(a) and numpy.isfinite(b)):
        # only happens for bins that ROOT does not compute (e.g. weighted bins
        # with no entries); the caller masks them out
        return numpy.nan, numpy.nan
    if a <= 0 or b <= 0:
        return 0.0, 1.0

    mode = float(beta_mode(a, b))
    if mode == 0.0:
        return 0.0, float(_beta_quantile(level, a, b))
    if mode == 1.0:
        return float(_beta_quantile_c(level, a, b)), 1.0
    if a == b and a <= 1.0:
        return (
            float(beta_central_interval(level, a, b, False)),
            float(beta_central_interval(level, a, b, True)),
        )

    def interval_length(lower):
        plow = scipy.special.betainc(a, b, lower)
        return _beta_quantile(plow + level, a, b) - lower

    lower_max = float(_beta_quantile_c(level, a, b))
    result = scipy.optimize.minimize_scalar(
        interval_length,
        bounds=(0.0, lower_max),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x), float(result.x + result.fun)


def beta_shortest_interval(level, a, b):
    """
    Shortest interval of the beta distribution, as
    ``TEfficiency::BetaShortestInterval``. Returns a 2-tuple of lower and upper
    boundaries.
    """
    a, b = numpy.broadcast_arrays(
        numpy.asarray(a, numpy.float64), numpy.asarray(b, numpy.float64)
    )
    lower = numpy.empty(a.shape, numpy.float64)
    upper = numpy.empty(a.shape, numpy.float64)
    for index in numpy.ndindex(a.shape):
        lower[index], upper[index] = _beta_shortest_interval_scalar(
            level, a[index], b[index]
        )
    return lower, upper


class TEfficiency:
    """
    Behaviors for TEfficiency: get the passed and total histograms, and the
    efficiencies with their (asymmetric) uncertainties as NumPy arrays.
    """

    @property
    def name(self):
        """
        The name of the efficiency object.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        The title of the efficiency object.
        """
        return self.member("fTitle")

    @property
    def passed(self):
        """
        The histogram of events that passed the selection
        (``fPassedHistogram``), as an Uproot histogram.
        """
        return self.member("fPassedHistogram")

    @property
    def total(self):
        """
        The histogram of all events (``fTotalHistogram``), as an Uproot
        histogram.
        """
        return self.member("fTotalHistogram")

    @property
    def axes(self):
        """
        A tuple of the axes of the (total) histogram.
        """
        return self.total.axes

    def axis(self, axis=0):
        """
        Returns an axis of the (total) histogram; see
        :ref:`uproot.behaviors.TH1.Histogram.axis`.
        """
        return self.total.axis(axis)

    @property
    def ndim(self):
        """
        The number of dimensions of the efficiency (1, 2, or 3).
        """
        return len(self.total.axes)

    @property
    def statistic_option(self):
        """
        The name of the method used to compute confidence intervals, one of
        ``"kFCP"`` (Clopper-Pearson), ``"kFNormal"``, ``"kFWilson"``,
        ``"kFAC"`` (Agresti-Coull), ``"kFFC"`` (Feldman-Cousins),
        ``"kBJeffrey"``, ``"kBUniform"``, ``"kBBayesian"``, or ``"kMidP"``.
        """
        option = self.member("fStatisticOption")
        if 0 <= option < len(statistic_options):
            return statistic_options[option]
        return "kFCP"

    @property
    def confidence_level(self):
        """
        The confidence level of the intervals (``fConfLevel``).
        """
        return self.member("fConfLevel")

    @property
    def beta_alpha(self):
        """
        Global alpha parameter of the beta prior (``fBeta_alpha``).
        """
        return self.member("fBeta_alpha")

    @property
    def beta_beta(self):
        """
        Global beta parameter of the beta prior (``fBeta_beta``).
        """
        return self.member("fBeta_beta")

    @property
    def beta_bin_params(self):
        """
        Per-bin (alpha, beta) parameters of the beta prior
        (``fBeta_bin_params``), as a structured array with fields ``"first"``
        and ``"second"``, indexed by ROOT global bin number.
        """
        return self.member("fBeta_bin_params")

    def _test_bit(self, bit):
        return bool(self.member("@fBits") & bit)

    @property
    def uses_bayesian_stat(self):
        """
        True if Bayesian statistics are used for intervals.
        """
        return self._test_bit(kIsBayesian)

    @property
    def uses_posterior_mode(self):
        """
        True if the posterior mode (rather than mean) is used as the efficiency
        estimate (Bayesian statistics only).
        """
        return self._test_bit(kPosteriorMode) and self._test_bit(kIsBayesian)

    @property
    def uses_shortest_interval(self):
        """
        True if the shortest (rather than central) interval is used (Bayesian
        statistics only).
        """
        return self._test_bit(kShortestInterval) and self._test_bit(kIsBayesian)

    @property
    def uses_weights(self):
        """
        True if the histograms were filled with weights.
        """
        return self._test_bit(kUseWeights)

    def _global_bins(self, shape):
        # ROOT global bin numbers (bin = x + (nx+2) * (y + (ny+2) * z)), in the
        # same array layout as values(flow=True)
        indices = numpy.indices(shape)
        out = numpy.zeros(shape, numpy.int64)
        stride = 1
        for dim, index in enumerate(indices):
            out += index * stride
            stride *= shape[dim]
        return out

    def _beta_parameters(self, shape):
        alpha = numpy.full(shape, self.member("fBeta_alpha"), numpy.float64)
        beta = numpy.full(shape, self.member("fBeta_beta"), numpy.float64)
        if self._test_bit(kUseBinPrior):
            params = self.member("fBeta_bin_params")
            global_bins = self._global_bins(shape)
            mask = global_bins < len(params)
            alpha[mask] = params["first"][global_bins[mask]]
            beta[mask] = params["second"][global_bins[mask]]
        return alpha, beta

    def _compute(self, errors):
        total = numpy.asarray(self.total.values(flow=True), numpy.float64)
        passed = numpy.asarray(self.passed.values(flow=True), numpy.float64)
        level = self.member("fConfLevel")
        bayesian = self._test_bit(kIsBayesian)
        weights = self._test_bit(kUseWeights)

        if weights:
            total_w2 = numpy.asarray(self.total.variances(flow=True), numpy.float64)

        if bayesian:
            alpha, beta = self._beta_parameters(total.shape)
            with numpy.errstate(all="ignore"):
                if weights:
                    norm = total / total_w2
                    aa = passed * norm + alpha
                    bb = (total - passed) * norm + beta
                else:
                    aa = passed + alpha
                    bb = total - passed + beta
            if self._test_bit(kPosteriorMode):
                eff = beta_mode(aa, bb)
            else:
                eff = beta_mean(aa, bb)
            if weights:
                with numpy.errstate(all="ignore"):
                    eff = numpy.where(total_w2 <= 0, passed / total, eff)
        else:
            with numpy.errstate(all="ignore"):
                eff = numpy.where(total != 0, passed / total, 0.0)

        if not errors:
            return eff

        if bayesian:
            if self._test_bit(kShortestInterval):
                lower, upper = beta_shortest_interval(level, aa, bb)
            else:
                lower = beta_central_interval(level, aa, bb, False)
                upper = beta_central_interval(level, aa, bb, True)
            error_low = eff - lower
            error_high = upper - eff
            if weights:
                error_low = numpy.where(total_w2 <= 0, 0.0, error_low)
                error_high = numpy.where(total_w2 <= 0, 0.0, error_high)

        elif weights:
            # ROOT only supports the normal approximation for weighted
            # frequentist intervals, regardless of fStatisticOption
            passed_w2 = numpy.asarray(self.passed.variances(flow=True), numpy.float64)
            with numpy.errstate(all="ignore"):
                variance = (passed_w2 * (1.0 - 2 * eff) + total_w2 * eff * eff) / (
                    total * total
                )
                sigma = numpy.sqrt(variance)
                delta = _normal_quantile(1.0 - 0.5 * (1.0 - level), sigma)
                error_low = numpy.where(eff - delta < 0, eff, delta)
                error_high = numpy.where(eff + delta > 1, 1.0 - eff, delta)

        else:
            boundary = _frequentist_boundaries.get(
                self.member("fStatisticOption"), clopper_pearson
            )
            error_low = eff - boundary(total, passed, level, False)
            error_high = boundary(total, passed, level, True) - eff

        return eff, error_low, error_high

    @staticmethod
    def _strip_flow(array, flow):
        if flow:
            return array
        return array[tuple(slice(1, -1) for _ in array.shape)]

    def values(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins.

        Efficiency in each bin, as ROOT's ``TEfficiency::GetEfficiency``.

        For frequentist statistic options, this is ``passed / total`` (or 0 if
        ``total`` is 0). For Bayesian options, it is the mean (or mode, if
        :ref:`uproot.behaviors.TEfficiency.TEfficiency.uses_posterior_mode`) of
        the posterior beta distribution.
        """
        return self._strip_flow(self._compute(errors=False), flow)

    def errors(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins.

        Returns a 2-tuple of the lower and upper uncertainties of the efficiency
        in each bin, as ROOT's ``TEfficiency::GetEfficiencyErrorLow`` and
        ``TEfficiency::GetEfficiencyErrorUp``, using this object's statistic
        option and confidence level. Both are non-negative distances from
        :ref:`uproot.behaviors.TEfficiency.TEfficiency.values`.

        For weighted frequentist efficiencies, ROOT only supports the normal
        approximation, which is used regardless of the statistic option.

        Note that ROOT (as of 6.40) does not restore the interval method of a
        ``TEfficiency`` read from a file, so that ROOT computes Clopper-Pearson
        intervals for all frequentist statistic options until
        ``SetStatisticOption`` is called again. Uproot always uses the stored
        statistic option, matching ROOT's results for the object before it was
        written.

        Requires SciPy.
        """
        _, error_low, error_high = self._compute(errors=True)
        return (self._strip_flow(error_low, flow), self._strip_flow(error_high, flow))

    def intervals(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins.

        Returns a 2-tuple of the lower and upper boundaries of the confidence
        interval of the efficiency in each bin.

        Requires SciPy.
        """
        eff, error_low, error_high = self._compute(errors=True)
        return (
            self._strip_flow(eff - error_low, flow),
            self._strip_flow(eff + error_high, flow),
        )
