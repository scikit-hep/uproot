# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behavior of ``TProfile2D``.
"""

from __future__ import annotations

import numpy

import uproot
import uproot.behaviors.TH2
import uproot.behaviors.TProfile
from uproot.behaviors.TH1 import boost_axis_metadata, boost_metadata


class TProfile2D(uproot.behaviors.TProfile.Profile):
    """
    Behaviors for two-dimensional profiles: ROOT's ``TProfile2D``.
    """

    no_inherit = (uproot.behaviors.TH2.TH2,)

    @property
    def axes(self):
        return (self.member("fXaxis"), self.member("fYaxis"))

    def axis(self, axis):
        if axis == 0 or axis == -2 or axis == "x":
            return self.member("fXaxis")
        elif axis == 1 or axis == -1 or axis == "y":
            return self.member("fYaxis")
        else:
            raise ValueError("axis must be 0 (-2), 1 (-1) or 'x', 'y' for a TProfile2D")

    @property
    def weighted(self):
        fBinSumw2 = self.member("fBinSumw2", none_if_missing=True)
        return fBinSumw2 is not None and len(fBinSumw2) == self.member("fNcells")

    def counts(self, flow=False):
        xaxis_fNbins = self.member("fXaxis").member("fNbins")
        yaxis_fNbins = self.member("fYaxis").member("fNbins")
        out = uproot.behaviors.TProfile._effective_counts_1d(
            numpy.asarray(self.member("fBinEntries")).reshape(-1),
            numpy.asarray(self.member("fBinSumw2")).reshape(-1),
            self.member("fNcells"),
        )
        out = numpy.transpose(out.reshape(yaxis_fNbins + 2, xaxis_fNbins + 2))
        if flow:
            return out
        else:
            return out[1:-1, 1:-1]

    def values(self, flow=False):
        if hasattr(self, "_values"):
            values = self._values
        else:
            (root_cont,) = self.base(uproot.models.TArray.Model_TArray)
            root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
            values = uproot.behaviors.TProfile._values_1d(
                numpy.asarray(self.member("fBinEntries")).reshape(-1),
                root_cont.reshape(-1),
            )
            xaxis_fNbins = self.member("fXaxis").member("fNbins")
            yaxis_fNbins = self.member("fYaxis").member("fNbins")
            values = numpy.transpose(values.reshape(yaxis_fNbins + 2, xaxis_fNbins + 2))
            self._values = values

        if flow:
            return values
        else:
            return values[1:-1, 1:-1]

    def _values_errors(self, flow, error_mode):
        attr = "_errors" + uproot.behaviors.TProfile._error_mode_str(error_mode)
        if hasattr(self, attr):
            values = self._values
            errors = getattr(self, attr)
        else:
            (root_cont,) = self.base(uproot.models.TArray.Model_TArray)
            root_cont = numpy.asarray(root_cont, dtype=numpy.float64)
            fSumw2 = self.member("fSumw2", none_if_missing=True)
            if fSumw2 is not None:
                fSumw2 = numpy.asarray(fSumw2).reshape(-1)
            values, errors = uproot.behaviors.TProfile._values_errors_1d(
                error_mode,
                numpy.asarray(self.member("fBinEntries")).reshape(-1),
                root_cont.reshape(-1),
                fSumw2,
                self.member("fNcells"),
                numpy.asarray(self.member("fBinSumw2")).reshape(-1),
            )
            xaxis_fNbins = self.member("fXaxis").member("fNbins")
            yaxis_fNbins = self.member("fYaxis").member("fNbins")
            values = numpy.transpose(values.reshape(yaxis_fNbins + 2, xaxis_fNbins + 2))
            errors = numpy.transpose(errors.reshape(yaxis_fNbins + 2, xaxis_fNbins + 2))
            self._values = values
            setattr(self, attr, errors)

        if flow:
            return values, errors
        else:
            return values[1:-1, 1:-1], errors[1:-1, 1:-1]

    def to_boost(self, metadata=boost_metadata, axis_metadata=boost_axis_metadata):
        boost_histogram = uproot.extras.boost_histogram()

        xaxis_fNbins = self.member("fXaxis").member("fNbins")
        yaxis_fNbins = self.member("fYaxis").member("fNbins")
        fNcells = self.member("fNcells")

        def _reshape_2d(values):
            return numpy.transpose(
                numpy.asarray(values, dtype=numpy.float64).reshape(
                    yaxis_fNbins + 2, xaxis_fNbins + 2
                )
            )

        effective_counts = _reshape_2d(
            numpy.asarray(self.counts(flow=True)).reshape(-1)
        )
        sum_of_bin_weights = _reshape_2d(self.member("fBinEntries"))
        raw_values = _reshape_2d(self._bases[0]._bases[-1])
        fSumw2_member = self.member("fSumw2", none_if_missing=True)

        # Compute mean = sum(y) / count (ROOT TProfile stores sum(y) in the TArray)
        nonzero = sum_of_bin_weights != 0
        mean_values = numpy.zeros(raw_values.shape, dtype=numpy.float64)
        mean_values[nonzero] = raw_values[nonzero] / sum_of_bin_weights[nonzero]

        # Compute sum_sq_dev = sum(y^2) - count * mean^2 directly from fSumw2.
        # fErrorMode is intentionally ignored here: it controls how ROOT displays
        # bin errors but does not change the underlying data. boost-hostogram's
        # storage has a fixed meaning for _sum_of_weighted_deltas_squared,
        # so we can't change it based on fErrorMode.
        if fSumw2_member is not None:
            fSumw2 = numpy.asarray(fSumw2_member, dtype=numpy.float64).reshape(-1)
        else:
            fSumw2 = numpy.array([], dtype=numpy.float64)
        if len(fSumw2) == fNcells:
            fSumw2 = _reshape_2d(fSumw2)
            sum_sq_dev = numpy.maximum(
                fSumw2 - sum_of_bin_weights * mean_values**2, 0.0
            )
        else:
            sum_sq_dev = numpy.zeros(raw_values.shape, dtype=numpy.float64)

        if self.weighted:
            storage = boost_histogram.storage.WeightedMean()
        else:
            storage = boost_histogram.storage.Mean()

        xaxis = uproot.behaviors.TH1._boost_axis(self.member("fXaxis"), axis_metadata)
        yaxis = uproot.behaviors.TH1._boost_axis(self.member("fYaxis"), axis_metadata)
        out = boost_histogram.Histogram(xaxis, yaxis, storage=storage)
        for k, v in metadata.items():
            setattr(out, k, self.member(v))

        # ROOT's categorical axes (with fLabels) always have an underflow bin (bin 0).
        # boost-histogram's Category axes do not have underflow.
        # We slice off the first ROOT bin along each categorical axis.
        if self.member("fXaxis").member("fLabels") is not None:
            effective_counts = effective_counts[1:, :]
            mean_values = mean_values[1:, :]
            sum_sq_dev = sum_sq_dev[1:, :]
            sum_of_bin_weights = sum_of_bin_weights[1:, :]
        if self.member("fYaxis").member("fLabels") is not None:
            effective_counts = effective_counts[:, 1:]
            mean_values = mean_values[:, 1:]
            sum_sq_dev = sum_sq_dev[:, 1:]
            sum_of_bin_weights = sum_of_bin_weights[:, 1:]

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
