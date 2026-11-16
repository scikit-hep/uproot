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
        raise NotImplementedError("FIXME @henryiii: this one kinda doesn't exist")
