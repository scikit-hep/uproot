# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``RooCurve``.
"""

from __future__ import annotations

import numpy

import uproot
import uproot.behaviors.TGraph

# '@fUniqueID', '@fBits', 'fName', 'fTitle', 'fLineColor', 'fLineStyle', 'fLineWidth',
# 'fFillColor', 'fFillStyle', 'fMarkerColor', 'fMarkerStyle', 'fMarkerSize', 'fNpoints',
# 'fX', 'fY', 'fFunctions', 'fHistogram', 'fMinimum', 'fMaximum', '_yAxisLabel',
# '_ymin', '_ymax', '_normValue'


def _parse_errs(xvalues, errs):
    xvals, yvals = errs.values()
    # Index of one-past right edge
    right_ind = numpy.argmax(xvals) + 2
    up_x = xvals[:right_ind]
    up_y = yvals[:right_ind]
    down_x = numpy.flip(xvals[right_ind:])
    down_y = numpy.flip(yvals[right_ind:])
    if (not numpy.all(numpy.diff(up_x) >= 0)) or (
        not numpy.all(numpy.diff(down_x) >= 0)
    ):
        raise ValueError("RooCurve x values are not increasing")
    up = numpy.interp(xvalues, up_x, up_y)
    down = numpy.interp(xvalues, down_x, down_y)
    return (up, down)


def _centers(edges):
    return (edges[1:] + edges[:-1]) / 2


class RooCurve(uproot.behaviors.TGraph.TGraph):
    """Behaviors for RooCurve.

    Beyond the behavior of a ``TGraph`` this also provides functionality to
    interpolate the graph at provided points, or extract a stored histogram
    (given bin edges).
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

    @property
    def curve_type(self):
        """
        Determines whether curve represents values or errors by checking if it is open or closed.

        Returns "VALUES" or "ERRORS".
        """
        xvals = self.values(axis="x")
        if numpy.isclose(xvals[0], xvals[-1]):
            return "ERRORS"
        else:
            return "VALUES"

    def interpolate(self, xvalues):
        """
        Args:
            xvalues (array_like): xvalues to interpolate at.

        Returns y values when RooCurve is interpolated at the given x values.
        """
        if self.curve_type != "VALUES":
            raise ValueError(
                "interpolate can only be called on a value (open) curve. "
                "Try interpolate_errors."
            )
        xvals, yvals = self.values()
        return numpy.interp(xvalues, xvals, yvals)

    def interpolate_asymm_errors(self, xvalues):
        """
        Args:
            xvalues (array_like): xvalues to interpolate at.

        Returns:
            up (array_like): Upper boundary of uncertainty band.
            down (array_like): Lower boundary of uncertainty band.

        Returns asymmetric y errors when RooCurve is interpolated at the given x values.
        """
        if self.curve_type != "ERRORS":
            raise ValueError(
                "interpolate_errors can only be called on an error (closed) curve. "
                "Try interpolate."
            )
        up, down = _parse_errs(xvalues, self)
        return (up, down)

    def interpolate_errors(self, xvalues):
        """
        Args:
            xvalues (array_like): xvalues to interpolate at.

        Returns y errors when RooCurve is interpolated at the given x values.
        """
        if self.curve_type != "ERRORS":
            raise ValueError(
                "interpolate_errors can only be called on an error (closed) curve. "
                "Try interpolate."
            )
        up, down = _parse_errs(xvalues, self)
        return numpy.abs((up - down) / 2)

    def to_boost(self, bin_edges, error_curve=None):
        """
        Args:
            bin_edges (array_like): Bin edges for histogram.
            error_curve (RooCurve): RooCurve visualizing errors.

        Returns ``boost-histogram`` object by interpolating ``RooCurve``.
        """
        if self.curve_type != "VALUES":
            raise ValueError(
                "to_boost should be called on the value curve. The error curve is passed using the"
                "error_curve parameter."
            )
        boost_histogram = uproot.extras.boost_histogram()
        # Lists and tuples would be concatenated, not added, by _centers
        bin_edges = numpy.asarray(bin_edges, dtype=numpy.float64)
        axis = boost_histogram.axis.Variable(bin_edges, underflow=False, overflow=False)
        axis.name = self.name
        axis.title = self.title
        centers = _centers(bin_edges)
        values = self.interpolate(centers)
        if error_curve is not None:
            errs = error_curve.interpolate_errors(centers)
            variances = numpy.square(errs)
            hist = boost_histogram.Histogram(
                axis, storage=boost_histogram.storage.Weight()
            )
            hist.name = self.name
            hist.title = self.title
            view = hist.view()
            view.value = values
            view.variance = variances
            return hist
        else:
            hist = boost_histogram.Histogram(
                axis, storage=boost_histogram.storage.Double()
            )
            hist.name = self.name
            hist.title = self.title
            view = hist.view()
            view[...] = values
            return hist

    def to_hist(self, bin_edges, error_curve=None):
        """
        Args:
            bin_edges (array_like): Bin edges for histogram.
            error_curve (RooCurve): RooCurve visualizing errors.

        Returns ``hist`` object by interpolating ``RooCurve``.
        """
        return uproot.extras.hist().Hist(self.to_boost(bin_edges, error_curve))
