# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``RooHist``.
"""

from __future__ import annotations

import numpy

import uproot
import uproot.behaviors.TGraphAsymmErrors

# ['@fUniqueID', '@fBits', 'fName', 'fTitle', 'fLineColor', 'fLineStyle',
#  'fLineWidth', 'fFillColor', 'fFillStyle', 'fMarkerColor', 'fMarkerStyle',
#  'fMarkerSize', 'fNpoints', 'fX', 'fY', 'fFunctions', 'fHistogram',
#  'fMinimum', 'fMaximum', 'fEXlow', 'fEXhigh', 'fEYlow', 'fEYhigh']


class RooHist(uproot.behaviors.TGraphAsymmErrors.TGraphAsymmErrors):
    """
    Behavior for ``RooHist``.

    This consists of ``to_boost`` (and ``to_hist``) and ``to_numpy``
    providing access to the histogram.
    """

    @property
    def name(self):
        """
        The name of the histogram.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        The title of the histogram.
        """
        return self.member("fTitle")

    def to_numpy(self, dd=False):
        """
        Args:
            dd (bool): If True, the return type follows
                `numpy.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`__;
                otherwise, it follows `numpy.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__
                and `numpy.histogram2d <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`__.

        Converts the ``RooHist`` into a form like the ones produced by the NumPy
        histogram functions.
        """
        bin_centers, values = self.values()
        bin_edges_low = bin_centers - self.errors(which="low", axis="x")
        bin_edges_high = bin_centers + self.errors(which="high", axis="x")
        if not numpy.all(numpy.isclose(bin_edges_low[1:], bin_edges_high[:-1])):
            raise ValueError("bin_edges_low[1:] != bin_edges_high[:-1]")
        bin_edges = numpy.append(bin_edges_low, [bin_edges_high[-1]])
        if dd:
            return values, (bin_edges,)
        else:
            return values, bin_edges

    def to_boost(self):
        """
        Converts ``RooHist`` into a ``boost-histogram`` object.
        """
        boost_histogram = uproot.extras.boost_histogram()
        bin_centers, values = self.values()
        bin_edges_low = bin_centers - self.errors(which="low", axis="x")
        bin_edges_high = bin_centers + self.errors(which="high", axis="x")
        # Boost histogram only supports symmetric errors
        if not numpy.all(numpy.isclose(bin_edges_low[1:], bin_edges_high[:-1])):
            raise ValueError("bin_edges_low[1:] != bin_edges_high[:-1]")
        errors = self.errors(which="mean", axis="y")
        variances = numpy.square(errors)

        # Now make the Boost histogram
        bin_edges = numpy.append(bin_edges_low, [bin_edges_high[-1]])
        axis = boost_histogram.axis.Variable(
            bin_edges,
            underflow=False,
            overflow=False,
        )
        axis.name = self.name
        axis.title = self.title
        hist = boost_histogram.Histogram(axis, storage=boost_histogram.storage.Weight())
        hist.name = self.name
        hist.title = self.title
        view = hist.view()
        view.value = values
        view.variance = variances
        return hist

    def to_hist(self):
        """
        Converts ``RooHist`` into a ``hist`` object.
        """
        return uproot.extras.hist().Hist(self.to_boost())

    def _to_boost_histogram_(self):
        return self.to_boost()
