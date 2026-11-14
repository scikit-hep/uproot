# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behavior of ``TProfile``.
"""

from __future__ import annotations

import numpy

import uproot
from uproot.behaviors.TH1 import boost_axis_metadata, boost_metadata

_kERRORMEAN = 0
_kERRORSPREAD = 1
_kERRORSPREADI = 2
_kERRORSPREADG = 3


# closely follows the ROOT function, using the same names (with 'root_' prepended)
# https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L141-L163
def _effective_counts_1d(fBinEntries, fBinSumw2, fNcells):
    root_sumOfWeights = fBinEntries
    root_sumOfWeights = numpy.asarray(root_sumOfWeights, dtype=numpy.float64)

    root_sumOfWeightSquare = fBinSumw2
    root_sumOfWeightSquare = numpy.asarray(root_sumOfWeightSquare, dtype=numpy.float64)

    if len(root_sumOfWeightSquare) == 0 or len(root_sumOfWeightSquare) != fNcells:
        return root_sumOfWeights

    positive = root_sumOfWeightSquare > 0

    out = numpy.zeros(len(root_sumOfWeights), dtype=numpy.float64)
    out[positive] = root_sumOfWeights[positive] ** 2 / root_sumOfWeightSquare[positive]
    return out


# duplicates the first part of '_values_errors_1d'
def _values_1d(fBinEntries, root_cont):
    root_sum = fBinEntries
    root_sum = numpy.asarray(root_sum, dtype=numpy.float64)
    nonzero = root_sum != 0

    root_contsum = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_contsum[nonzero] = root_cont[nonzero] / root_sum[nonzero]

    return root_contsum


def _error_mode_str(error_mode):
    if error_mode is None or error_mode == _kERRORMEAN or error_mode == "":
        return ""
    elif error_mode == _kERRORSPREAD or error_mode == "s":
        return "S"
    elif error_mode == _kERRORSPREADI or error_mode == "i":
        return "I"
    elif error_mode == _kERRORSPREADG or error_mode == "g":
        return "G"
    else:
        return "_"


# closely follows the ROOT function, using the same names (with 'root_' prepended)
# https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L660-L721
def _values_errors_1d(error_mode, fBinEntries, root_cont, fSumw2, fNcells, fBinSumw2):
    if error_mode is None or error_mode == _kERRORMEAN or error_mode == "":
        error_mode = _kERRORMEAN
    elif error_mode == _kERRORSPREAD or error_mode == "s":
        error_mode = _kERRORSPREAD
    elif error_mode == _kERRORSPREADI or error_mode == "i":
        error_mode = _kERRORSPREADI
    elif error_mode == _kERRORSPREADG or error_mode == "g":
        error_mode = _kERRORSPREADG
    else:
        raise ValueError(
            "error_mode must be None/0/'' for error-on-mean,\n"
            "                        1/'s' for spread (variance),\n"
            "                        2/'i' for integer spread (using sqrt(12)),\n"
            "                     or 3/'g' for Gaussian spread\n"
            "                    not "
            + repr(error_mode)
            + "see https://root.cern.ch/doc/master/classTProfile.html"
        )

    root_sum = fBinEntries
    root_sum = numpy.asarray(root_sum, dtype=numpy.float64)
    nonzero = root_sum != 0

    root_contsum = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_contsum[nonzero] = root_cont[nonzero] / root_sum[nonzero]

    if error_mode == _kERRORSPREADG:
        out = numpy.zeros(len(root_cont), dtype=numpy.float64)
        out[nonzero] = 1.0 / numpy.sqrt(root_sum[nonzero])
        return root_contsum, out

    root_err2 = fSumw2
    if root_err2 is None or len(root_err2) != fNcells:
        root_err2 = numpy.zeros(len(root_cont), dtype=numpy.float64)
    else:
        root_err2 = numpy.asarray(root_err2, dtype=numpy.float64)

    root_neff = _effective_counts_1d(fBinEntries, fBinSumw2, fNcells)

    root_eprisum_sq_dev = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_eprisum_sq_dev[nonzero] = abs(
        root_err2[nonzero] / root_sum[nonzero] - root_contsum[nonzero] ** 2
    )
    root_eprim = numpy.sqrt(root_eprisum_sq_dev)

    if error_mode == _kERRORSPREADI:
        numer = numpy.ones(len(root_cont), dtype=numpy.float64)
        denom = numpy.full(len(root_cont), numpy.sqrt(12 * root_neff))

        eprim_nonzero = root_eprim != 0
        numer[eprim_nonzero] = root_eprim[eprim_nonzero]
        denom[eprim_nonzero] = numpy.sqrt(root_neff[eprim_nonzero])

        out = numpy.zeros(len(root_cont), dtype=numpy.float64)
        out[nonzero] = numer[nonzero] / denom[nonzero]
        return root_contsum, out

    if error_mode == _kERRORSPREAD:
        root_eprim[~nonzero] = 0.0
        return root_contsum, root_eprim

    out = numpy.zeros(len(root_cont), dtype=numpy.float64)
    out[nonzero] = root_eprim[nonzero] / numpy.sqrt(root_neff[nonzero])
    return root_contsum, out


class Profile(uproot.behaviors.TH1.Histogram):
    """
    Abstract class for profile plots.
    """

    @property
    def kind(self):
        return "MEAN"

    def counts(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        The effective number of entries, which is a step in the calculation of
        :ref:`uproot.behaviors.TProfile.Profile.values`. The calculation
        of profile errors exactly follows ROOT's "effective entries", but in a
        vectorized NumPy form.
        """
        raise NotImplementedError(repr(self))

    def values(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Mean value of each bin as a 1, 2, or 3 dimensional ``numpy.ndarray`` of
        ``numpy.float64``.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        raise NotImplementedError(repr(self))

    def errors(self, flow=False, error_mode=""):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.
            error_mode (str): Choose a method for calculating the errors (see below).

        Errors (uncertainties) in the :ref:`uproot.behaviors.TH1.Histogram.values`
        as a 1, 2, or 3 dimensional ``numpy.ndarray`` of ``numpy.float64``.

        The calculation of profile errors exactly follows ROOT's function, but
        in a vectorized NumPy form. The ``error_mode`` may be

        * ``""`` for standard error on the mean
        * ``"s"`` for spread
        * ``"i"`` for integer data
        * ``"g"`` for Gaussian

        following `ROOT's profile documentation <https://root.cern.ch/doc/master/classTProfile.html>`__.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        _values, errors = self._values_errors(flow, error_mode)
        return errors

    def variances(self, flow=False, error_mode=""):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.
            error_mode (str): Choose a method for calculating the errors (see below).

        Variances (uncertainties squared) in the
        :ref:`uproot.behaviors.TH1.Histogram.values` as a 1, 2, or 3
        dimensional ``numpy.ndarray`` of ``numpy.float64``.

        The calculation of profile variances exactly follows ROOT's function, but
        in a vectorized NumPy form. The ``error_mode`` may be

        * ``""`` for standard error on the mean (squared)
        * ``"s"`` for spread (squared)
        * ``"i"`` for integer data (squared)
        * ``"g"`` for Gaussian (squared)

        following `ROOT's profile documentation <https://root.cern.ch/doc/master/classTProfile.html>`__.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        _values, errors = self._values_errors(flow, error_mode)
        return numpy.square(errors)

    def _values_variances(self, flow):
        values, errors = self._values_errors(flow, "")
        return values, numpy.square(errors)


class TProfile(Profile):
    """
    Behaviors for one-dimensional profiles: ROOT's ``TProfile``.
    """

    no_inherit = (uproot.behaviors.TH1.TH1,)

    @property
    def axes(self):
        return (self.member("fXaxis"),)

    def axis(self, axis=0):  # default axis for one-dimensional is intentional
        if axis == 0 or axis == -1 or axis == "x":
            return self.member("fXaxis")
        else:
            raise ValueError("axis must be 0 (-1) or 'x' for a TProfile")

    @property
    def weighted(self):
        fBinSumw2 = self.member("fBinSumw2", none_if_missing=True)
        return fBinSumw2 is not None and len(fBinSumw2) == self.member("fNcells")

    def counts(self, flow=False):
        out = _effective_counts_1d(
            self.member("fBinEntries"),
            self.member("fBinSumw2"),
            self.member("fNcells"),
        )
        if flow:
            return out
        else:
            return out[1:-1]

    def values(self, flow=False):
        if hasattr(self, "_values"):
            values = self._values
        else:
            (root_cont,) = self.base(uproot.models.TArray.Model_TArray)
            root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
            values = _values_1d(
                self.member("fBinEntries"),
                root_cont,
            )
            self._values = values

        if flow:
            return values
        else:
            return values[1:-1]

    def _values_errors(self, flow, error_mode):
        attr = "_errors" + _error_mode_str(error_mode)
        if hasattr(self, attr):
            values = self._values
            errors = getattr(self, attr)
        else:
            (root_cont,) = self.base(uproot.models.TArray.Model_TArray)
            root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
            fSumw2 = self.member("fSumw2", none_if_missing=True)
            if fSumw2 is not None:
                fSumw2 = numpy.asarray(fSumw2)
            values, errors = _values_errors_1d(
                error_mode,
                numpy.asarray(self.member("fBinEntries")),
                root_cont,
                fSumw2,
                self.member("fNcells"),
                numpy.asarray(self.member("fBinSumw2")),
            )
            self._values = values
            setattr(self, attr, errors)

        if flow:
            return values, errors
        else:
            return values[1:-1], errors[1:-1]

    def to_boost(self, metadata=boost_metadata, axis_metadata=boost_axis_metadata):
        boost_histogram = uproot.extras.boost_histogram()

        effective_counts = self.counts(flow=True)
        sum_of_bin_weights = numpy.asarray(
            self.member("fBinEntries"), dtype=numpy.float64
        )
        raw_values = numpy.asarray(self._bases[0]._bases[-1], dtype=numpy.float64)
        fSumw2_member = self.member("fSumw2", none_if_missing=True)
        fNcells = self.member("fNcells")

        # Compute mean = sum(y) / count (ROOT TProfile stores sum(y) in the TArray)
        nonzero = sum_of_bin_weights != 0
        mean_values = numpy.zeros(len(raw_values), dtype=numpy.float64)
        mean_values[nonzero] = raw_values[nonzero] / sum_of_bin_weights[nonzero]

        # Compute sum_sq_dev = sum(y^2) - count * mean^2 directly from fSumw2.
        # fErrorMode is intentionally ignored here: it controls how ROOT displays
        # bin errors but does not change the underlying data. boost-hostogram's
        # storage has a fixed meaning for _sum_of_weighted_deltas_squared,
        # so we can't change it based on fErrorMode.
        if fSumw2_member is not None:
            fSumw2 = numpy.asarray(fSumw2_member, dtype=numpy.float64)
        else:
            fSumw2 = numpy.array([], dtype=numpy.float64)
        if len(fSumw2) == fNcells:
            sum_sq_dev = numpy.maximum(
                fSumw2 - sum_of_bin_weights * mean_values**2, 0.0
            )
        else:
            sum_sq_dev = numpy.zeros(len(raw_values), dtype=numpy.float64)

        if self.weighted:
            storage = boost_histogram.storage.WeightedMean()
        else:
            storage = boost_histogram.storage.Mean()

        xaxis = uproot.behaviors.TH1._boost_axis(self.member("fXaxis"), axis_metadata)
        out = boost_histogram.Histogram(xaxis, storage=storage)
        for k, v in metadata.items():
            setattr(out, k, self.member(v))

        if self.member("fXaxis").member("fLabels") is not None:
            # ROOT's categorical axes (with fLabels) always have an underflow bin (bin 0).
            # boost-histogram's Category axes do not have underflow.
            # We slice off the first ROOT bin (underflow) to align with boost-histogram.
            effective_counts = effective_counts[1:]
            mean_values = mean_values[1:]
            sum_sq_dev = sum_sq_dev[1:]
            sum_of_bin_weights = sum_of_bin_weights[1:]

        # TODO: This should only be needed for weighted storage, but there seems to be some bug in Uproot's serialization of fBinSumw2
        # that causes weighted TProfiles to appear unweighted when read back in.
        out.metadata = {
            "fEntries": self.member("fEntries"),
        }
        view = out.view(flow=True)

        # https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L668-L671
        if self.weighted:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                sum_of_bin_weights_squared = (sum_of_bin_weights**2) / effective_counts

            # TODO: Drop this when boost-histogram has a way to set using the constructor.
            # New version should look something like this:
            # view[...] = np.stack(sum_of_bin_weights, sum_of_bin_weights_squared, mean_values, sum_sq_dev)
            # Current / classic version:
            view["sum_of_weights"] = sum_of_bin_weights
            view["sum_of_weights_squared"] = sum_of_bin_weights_squared
            view["value"] = mean_values
            view["_sum_of_weighted_deltas_squared"] = sum_sq_dev
        else:
            view["count"] = sum_of_bin_weights
            view["value"] = mean_values
            view["_sum_of_deltas_squared"] = sum_sq_dev

        return out
