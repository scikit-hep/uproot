# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines the behaviors of ``TH3`` and its subclasses (not including
``TProfile3D``).
"""

from __future__ import annotations

import numpy

import uproot


class TH3(uproot.behaviors.TH1.Histogram):
    """
    Behaviors for three-dimensional histograms: descendants of ROOT's
    ``TH3``, not including ``TProfile3D``.
    """

    no_inherit = (uproot.behaviors.TH1.TH1,)

    @property
    def axes(self):
        return (self.member("fXaxis"), self.member("fYaxis"), self.member("fZaxis"))

    def axis(self, axis):
        if axis == 0 or axis == -3 or axis == "x":
            return self.member("fXaxis")
        elif axis == 1 or axis == -2 or axis == "y":
            return self.member("fYaxis")
        elif axis == 2 or axis == -1 or axis == "z":
            return self.member("fZaxis")
        else:
            raise ValueError(
                "axis must be 0 (-3), 1 (-2), 2 (-1) or 'x', 'y', 'z' for a TH3"
            )

    @property
    def kind(self):
        return "COUNT"

    def values(self, flow=False):
        if hasattr(self, "_values"):
            values = self._values
        else:
            (values,) = self.base(uproot.models.TArray.Model_TArray)
            values = numpy.asarray(values, dtype=values.dtype.newbyteorder("="))

            xaxis_fNbins = self.member("fXaxis").member("fNbins")
            yaxis_fNbins = self.member("fYaxis").member("fNbins")
            zaxis_fNbins = self.member("fZaxis").member("fNbins")
            values = numpy.transpose(
                values.reshape(zaxis_fNbins + 2, yaxis_fNbins + 2, xaxis_fNbins + 2)
            )
            self._values = values

        if flow:
            return values
        else:
            return values[1:-1, 1:-1, 1:-1]

    def _values_variances(self, flow):
        values = self.values(flow=True)

        if hasattr(self, "_variances"):
            variances = self._variances
        else:
            variances = numpy.zeros(values.shape, dtype=numpy.float64)
            sumw2 = self.member("fSumw2", none_if_missing=True)
            if sumw2 is not None and len(sumw2) == self.member("fNcells"):
                sumw2 = numpy.asarray(sumw2, dtype=sumw2.dtype.newbyteorder("="))
                sumw2 = numpy.transpose(numpy.reshape(sumw2, values.shape[::-1]))
                positive = sumw2 > 0
                variances[positive] = sumw2[positive]
            else:
                # like ROOT's TH1::GetBinError, the error is sqrt(|content|)
                variances[...] = numpy.abs(values.astype(numpy.float64))
            self._variances = variances

        if flow:
            return values, variances
        else:
            return values[1:-1, 1:-1, 1:-1], variances[1:-1, 1:-1, 1:-1]

    def to_numpy(self, flow=False, dd=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins; otherwise,
                only normal (finite-width) bins are included.
            dd (bool): If True, the return type follows
                `numpy.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`__;
                otherwise, it follows `numpy.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__
                and `numpy.histogram2d <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`__.

        Converts the histogram into a form like the ones produced by the NumPy
        histogram functions.
        """
        values = self.values(flow=flow)
        xedges = self.axis(0).edges(flow=flow)
        yedges = self.axis(1).edges(flow=flow)
        zedges = self.axis(2).edges(flow=flow)
        if dd:
            return values, (xedges, yedges, zedges)
        else:
            return values, xedges, yedges, zedges
