# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behavior of ``TProfile``.
"""

from __future__ import annotations

import hashlib
import math

import numpy

import uproot
from uproot.behaviors.TH1 import boost_axis_metadata, boost_metadata

_kERRORMEAN = 0
_kERRORSPREAD = 1
_kERRORSPREADI = 2
_kERRORSPREADG = 3


def _bins_fingerprint(hist):
    """
    A hash of what ROOT's global statistics (fEntries, fTsumw, fTsumw2,
    fTsumwx, fTsumwx2) depend on in a boost-histogram profile: each bin's sum of
    weights (and of squared weights) and where the bin is. ``to_boost`` records
    it with those statistics, and the writer compares it to check that the
    statistics still go with the bins.
    """
    view = hist.view(flow=True)
    fields = (
        ("sum_of_weights", "sum_of_weights_squared")
        if "sum_of_weights" in view.dtype.names
        else ("count",)
    )
    digest = hashlib.sha256()
    for field in fields:
        digest.update(numpy.ascontiguousarray(view[field], numpy.float64).tobytes())
    for axis in hist.axes:
        digest.update(numpy.ascontiguousarray(axis.edges, numpy.float64).tobytes())
        if axis.traits.discrete:
            digest.update(repr([axis.value(i) for i in range(len(axis))]).encode())
    return digest.hexdigest()


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


def _error_mode_int(error_mode):
    if error_mode is None or error_mode == _kERRORMEAN or error_mode == "":
        return _kERRORMEAN
    elif error_mode == _kERRORSPREAD or error_mode == "s":
        return _kERRORSPREAD
    elif error_mode == _kERRORSPREADI or error_mode == "i":
        return _kERRORSPREADI
    elif error_mode == _kERRORSPREADG or error_mode == "g":
        return _kERRORSPREADG
    else:
        raise ValueError(
            "error_mode must be None/0/'' for error-on-mean,\n"
            "                        1/'s' for spread (variance),\n"
            "                        2/'i' for integer spread (using sqrt(12)),\n"
            "                     or 3/'g' for Gaussian spread\n"
            "                    not "
            + repr(error_mode)
            + "\nsee https://root.cern.ch/doc/master/classTProfile.html"
        )


# closely follows the ROOT function, using the same names (with 'root_' prepended)
# https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L660-L721
def _values_errors_1d(error_mode, fBinEntries, root_cont, fSumw2, fNcells, fBinSumw2):
    error_mode = _error_mode_int(error_mode)

    root_sum = fBinEntries
    root_sum = numpy.asarray(root_sum, dtype=numpy.float64)
    nonzero = root_sum != 0

    root_contsum = numpy.zeros(len(root_cont), dtype=numpy.float64)
    root_contsum[nonzero] = root_cont[nonzero] / root_sum[nonzero]

    if error_mode == _kERRORSPREADG:
        out = numpy.zeros(len(root_cont), dtype=numpy.float64)
        # NaN for a negative sum of weights, as in ROOT
        with numpy.errstate(invalid="ignore"):
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

    @property
    def weighted(self):
        """
        True if the profile has weights (``fBinSumw2``); False otherwise.
        """
        return len(self._fBinSumw2()) == self.member("fNcells")

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

    def errors(self, flow=False, error_mode=None):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.
            error_mode (None or str): Choose a method for calculating the errors
                (see below). If None, use the profile's own error option
                (``fErrorMode``), as ROOT's ``GetBinError`` does.

        Errors (uncertainties) in the :ref:`uproot.behaviors.TH1.Histogram.values`
        as a 1, 2, or 3 dimensional ``numpy.ndarray`` of ``numpy.float64``.

        The calculation of profile errors exactly follows ROOT's function, but
        in a vectorized NumPy form. The ``error_mode`` may be

        * ``""`` for standard error on the mean
        * ``"s"`` for spread
        * ``"i"`` for integer data
        * ``"g"`` for Gaussian

        following `ROOT's profile documentation <https://root.cern.ch/doc/master/classTProfile.html>`__.

        Note that these follow ROOT's conventions, which differ from those of
        boost-histogram's ``variances()`` (for example, after ``to_hist``):
        ROOT does not apply Bessel's correction to the spread, gives zero
        error (rather than NaN) for bins with a single entry, and, with weights,
        divides by the effective number of entries rather than the sum of
        weights. boost-histogram also always gives the error on the mean,
        whatever the profile's error option.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        _values, errors = self._values_errors(flow, self._error_mode(error_mode))
        return errors

    def variances(self, flow=False, error_mode=None):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.
            error_mode (None or str): Choose a method for calculating the errors
                (see below). If None, use the profile's own error option
                (``fErrorMode``), as ROOT's ``GetBinError`` does.

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

        See :ref:`uproot.behaviors.TProfile.Profile.errors` for how these
        conventions differ from boost-histogram's.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        _values, errors = self._values_errors(flow, self._error_mode(error_mode))
        return numpy.square(errors)

    def to_numpy(self, flow=False, dd=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins; otherwise,
                only normal (finite-width) bins are included.
            dd (bool): If True, the return type follows
                `numpy.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`__;
                otherwise, it follows `numpy.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__
                and `numpy.histogram2d <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`__.

        Converts the profile into a form like the ones produced by the NumPy
        histogram functions, with the mean of each bin
        (:ref:`uproot.behaviors.TProfile.Profile.values`) as the bin contents.
        """
        values = self.values(flow=flow)
        edges = tuple(axis.edges(flow=flow) for axis in self.axes)
        if dd:
            return values, edges
        else:
            return (values, *edges)

    def _error_mode(self, error_mode):
        if error_mode is None:
            return self.member("fErrorMode", none_if_missing=True)
        else:
            return error_mode

    def _values_variances(self, flow):
        return self.values(flow=flow), self.variances(flow=flow)

    def _fBinSumw2(self):
        # TProfile class versions before 6 (ROOT < 5.26) have no fBinSumw2;
        # ROOT treats them like an empty fBinSumw2 (unweighted)
        fBinSumw2 = self.member("fBinSumw2", none_if_missing=True)
        if fBinSumw2 is None:
            return numpy.array([], dtype=numpy.float64)
        else:
            return numpy.asarray(fBinSumw2)


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

    def counts(self, flow=False):
        out = _effective_counts_1d(
            self.member("fBinEntries"),
            self._fBinSumw2(),
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
                self._fBinSumw2(),
            )
            self._values = values
            setattr(self, attr, errors)

        if flow:
            return values, errors
        else:
            return values[1:-1], errors[1:-1]

    def to_boost(self, metadata=boost_metadata, axis_metadata=boost_axis_metadata):
        """
        Args:
            metadata (dict of str \u2192 str): Metadata to collect (keys) and
                their C++ class member names (values).
            axis_metadata (dict of str \u2192 str): Metadata to collect from
                each axis.

        Converts the profile into a ``boost-histogram`` object with ``Mean``
        storage, or ``WeightedMean`` storage if the profile is weighted.

        The per-bin sums are carried over exactly, but boost-histogram computes
        ``variances()`` with its own conventions, which differ from
        :ref:`uproot.behaviors.TProfile.Profile.variances`: it applies Bessel's
        correction, gives NaN for bins with a single entry, divides by the sum
        of weights rather than the effective number of entries (for weighted
        profiles), doesn't take the absolute value of a negative sum of squared
        deviations (possible with negative weights), and ignores the profile's
        error option (which is kept in ``metadata["fErrorMode"]``).

        ROOT statistics that boost-histogram doesn't track are kept in
        ``metadata`` for writing the profile back to ROOT: ``fEntries``,
        ``fErrorMode``, and (if they go with the bins) ``fTsumw``, ``fTsumw2``,
        ``fTsumwx`` and ``fTsumwx2``. They go with a ``"bins_fingerprint"`` of
        the bins, and if the bins change (filled, added, reordered, ...), they
        are recomputed from the bins when writing. Statistics in a histogram's
        metadata without a ``"bins_fingerprint"`` (set by hand, or by older
        versions of Uproot) are taken as they are, even if they are stale.
        """
        boost_histogram = uproot.extras.boost_histogram()

        sum_of_bin_weights = numpy.asarray(
            self.member("fBinEntries"), dtype=numpy.float64
        )
        # ROOT's effective entries are sum(w)**2 / fBinSumw2, so fBinSumw2 is the
        # sum of squared weights (0, not 0/0, in empty bins)
        # https://github.com/root-project/root/blob/ffc7c588ac91aca30e75d356ea971129ee6a836a/hist/hist/src/TProfileHelper.h#L141-L163
        sum_of_bin_weights_squared = numpy.asarray(
            self._fBinSumw2(), dtype=numpy.float64
        )
        raw_values = numpy.asarray(self._bases[0]._bases[-1], dtype=numpy.float64)
        fSumw2_member = self.member("fSumw2", none_if_missing=True)
        fNcells = self.member("fNcells")

        # Compute mean = sum(y) / count (ROOT TProfile stores sum(y) in the TArray)
        nonzero = sum_of_bin_weights != 0
        mean_values = numpy.zeros(len(raw_values), dtype=numpy.float64)
        mean_values[nonzero] = raw_values[nonzero] / sum_of_bin_weights[nonzero]
        if numpy.any(~nonzero & (raw_values != 0)):
            uproot._util.warn_outside_uproot(
                "some bins have a zero sum of weights but a nonzero sum of "
                "weighted values (possible with negative weights), which "
                "boost-histogram's mean storage can't represent, so those sums "
                "are dropped",
            )

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
            # This can be legitimately negative with negative weights (ROOT takes
            # its absolute value when computing errors), so only clamp round-off.
            sum_sq_dev = fSumw2 - sum_of_bin_weights * mean_values**2
            sum_sq_dev[(sum_sq_dev < 0) & (-sum_sq_dev <= 1e-10 * abs(fSumw2))] = 0.0
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

        if self.member("fXaxis").traits.discrete:
            # ROOT's categorical axes (with fLabels) always have an underflow bin (bin 0).
            # boost-histogram's Category axes do not have underflow.
            # We slice off the first ROOT bin (underflow) to align with boost-histogram.
            uproot.behaviors.TH1._warn_category_underflow(
                self.member("fXaxis"),
                sum_of_bin_weights[0] != 0 or raw_values[0] != 0,
            )
            mean_values = mean_values[1:]
            sum_sq_dev = sum_sq_dev[1:]
            sum_of_bin_weights = sum_of_bin_weights[1:]
            sum_of_bin_weights_squared = sum_of_bin_weights_squared[1:]

        if numpy.any(sum_sq_dev < 0):
            # kept as they are, so that writing back to ROOT gives the same sums
            uproot._util.warn_outside_uproot(
                "some bins have a negative sum of squared deviations (possible "
                "with negative weights), so boost-histogram's variances() are "
                "negative or NaN in those bins; ROOT's errors (and this "
                "profile's errors()) use its absolute value",
            )

        # boost-histogram does not track the number of fills (ROOT's fEntries),
        # the error option, or global statistics such as the sums of filled x
        # values (it only knows bin centers) and of squared weights (which ROOT
        # can have without fBinSumw2, after merging with a weighted profile), so
        # keep them in the metadata for writing back to ROOT. These are only
        # valid for these bins, so the metadata also has a fingerprint of the
        # bins (see _bins_fingerprint), which the writer compares to check that
        # they haven't changed.
        fXaxis = self.member("fXaxis")
        if fXaxis.traits.discrete:
            # the underflow was dropped above
            sum_of_noflow_weights = sum_of_bin_weights[:-1]
        else:
            sum_of_noflow_weights = sum_of_bin_weights[1:-1]
        out.metadata = {
            "fEntries": self.member("fEntries"),
            "fErrorMode": self.member("fErrorMode", none_if_missing=True),
        }
        # ROOT's statistics are out of date if they don't go with the bins (e.g.
        # after SetBinContent or an uneven Rebin) or include the flow bins (with
        # SetStatOverflows); otherwise, its fTsumw differs from the bins' only
        # by the round-off of summing the fills in a different order.
        fTsumw = self.member("fTsumw")
        sumw = math.fsum(sum_of_noflow_weights)
        if abs(fTsumw - sumw) <= 1e-9 * math.fsum(numpy.abs(sum_of_noflow_weights)):
            out.metadata["fTsumw"] = fTsumw
            out.metadata["fTsumw2"] = self.member("fTsumw2")
            # The x sums are in the axis's coordinates, which a labeled axis keeps
            # only if it spans [0, nbins], since boost-histogram's categories are
            # written back as bins 0, 1, ...
            if not fXaxis.traits.discrete or (
                fXaxis.member("fXmin") == 0
                and fXaxis.member("fXmax") == fXaxis.member("fNbins")
            ):
                out.metadata["fTsumwx"] = self.member("fTsumwx")
                out.metadata["fTsumwx2"] = self.member("fTsumwx2")
        view = out.view(flow=True)

        if self.weighted:
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

        out.metadata["bins_fingerprint"] = _bins_fingerprint(out)
        return out
