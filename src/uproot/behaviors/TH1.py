# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines the behaviors of ``TH1`` and its subclasses (not including ``TH2``,
``TH3``, or ``TProfile``).
"""

from __future__ import annotations

import numpy

import uproot

boost_metadata = {"name": "fName", "label": "fTitle"}
boost_axis_metadata = {"name": "fName", "label": "fTitle"}


def _boost_axis(axis, metadata):
    boost_histogram = uproot.extras.boost_histogram()

    fNbins = axis.member("fNbins")
    fXbins = axis.member("fXbins", none_if_missing=True)

    fLabels = axis.member("fLabels", none_if_missing=True)
    if fLabels is not None:
        try:
            labels = [int(x) for x in fLabels]
            category_cls = boost_histogram.axis.IntCategory
        except ValueError:
            labels = [str(x) for x in fLabels]
            category_cls = boost_histogram.axis.StrCategory
        out = category_cls(labels)

    elif fXbins is None or len(fXbins) != fNbins + 1:
        out = boost_histogram.axis.Regular(
            fNbins,
            axis.member("fXmin"),
            axis.member("fXmax"),
            underflow=True,
            overflow=True,
        )

    else:
        out = boost_histogram.axis.Variable(fXbins, underflow=True, overflow=True)

    for k, v in metadata.items():
        setattr(out, k, axis.member(v))
    return out


class Histogram:
    """
    Abstract class for histograms.
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

    def __eq__(self, other):
        """
        Two histograms are equal if their axes are equal, their values are equal,
        and their variances are equal.
        """
        if type(self) is not type(other):
            return False
        if self.axes != other.axes:
            return False
        self_values, self_variances = self._values_variances(True)
        other_values, other_variances = other._values_variances(True)
        values_equal = numpy.array_equal(self_values, other_values)
        variances_equal = numpy.array_equal(self_variances, other_variances)
        return values_equal and variances_equal

    @property
    def axes(self):
        """
        A tuple of all :doc:`uproot.behaviors.TAxis.TAxis` objects.
        """
        raise NotImplementedError(repr(self))

    def axis(self, axis):
        """
        Returns a specified :doc:`uproot.behaviors.TAxis.TAxis` object.

        The ``axis`` can be specified as

        * a non-negative integer: ``0`` is the first axis, ``1`` is the second,
          and ``2`` is the third.
        * a negative integer: ``-1`` is the last axis, ``-2`` is the
          second-to-last, and ``-3`` is the third-to-last.
        * a string: ``"x"`` is the first axis, ``"y"`` is the second, and ``"z"``
          is the third

        (assuming that the histogram dimension supports a given ``axis``).
        """
        raise NotImplementedError(repr(self))

    @property
    def weighted(self):
        """
        True if the histogram has weights (``fSumw2``); False otherwise.
        """
        sumw2 = self.member("fSumw2")
        return (
            sumw2 is not None
            and len(sumw2) > 0
            and len(sumw2) == self.member("fNcells")
        )

    @property
    def kind(self):
        """
        The meaning of this object: ``"COUNT"`` for true histograms (TH*) and
        ``"MEAN"`` for profile plots (TProfile*).
        """
        raise NotImplementedError(repr(self))

    def values(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Bin contents as a 1, 2, or 3 dimensional ``numpy.ndarray``. The
        ``numpy.dtype`` of this array depends on the histogram type.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        raise NotImplementedError(repr(self))

    def errors(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Errors (uncertainties) in the :ref:`uproot.behaviors.TH1.Histogram.values`
        as a 1, 2, or 3 dimensional ``numpy.ndarray`` of ``numpy.float64``.

        If ``fSumw2`` (weights) are available, they will be used in the
        calculation of the errors. If not, errors are assumed to be the square
        root of the values.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        _values, variances = self._values_variances(flow)
        return numpy.sqrt(variances)

    def variances(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Variances (uncertainties squared) in the
        :ref:`uproot.behaviors.TH1.Histogram.values` as a 1, 2, or 3
        dimensional ``numpy.ndarray`` of ``numpy.float64``.

        If ``fSumw2`` (weights) are available, they will be used in the
        calculation of the variances. If not, variances are assumed to be equal
        to the values.

        Setting ``flow=True`` increases the length of each dimension by two.
        """
        _values, variances = self._values_variances(flow)
        return variances

    def counts(self, flow=False):
        """
        Args:
            flow (bool): If True, include underflow and overflow bins before and
                after the normal (finite-width) bins.

        Returns the (possibly weighted) number of entries in each bin. For
        histograms, this is equal to :ref:`uproot.behaviors.TH1.Histogram.values`.
        """
        return self.values(flow=flow)

    def to_boost(self, metadata=None, axis_metadata=None):
        """
        Args:
            metadata (dict of str \u2192 str): Metadata to collect (keys) and
                their C++ class member names (values).
            axis_metadata (dict of str \u2192 str): Metadata to collect from
                each axis.

        Converts the histogram into a ``boost-histogram`` object.
        """
        assert len(self.axes) <= 3, "Only 1D, 2D, and 3D histograms are supported"

        if axis_metadata is None:
            axis_metadata = boost_axis_metadata
        if metadata is None:
            metadata = boost_metadata

        boost_histogram = uproot.extras.boost_histogram()

        sumw2 = None
        if self.weighted:  # ensures self.member("fSumw2") exists
            values, sumw2 = self._values_variances(flow=True)
            storage = boost_histogram.storage.Weight()
        else:
            values = self.values(flow=True)
            if issubclass(values.dtype.type, numpy.integer):
                storage = boost_histogram.storage.Int64()
            else:
                storage = boost_histogram.storage.Double()

        axes = [
            _boost_axis(self.member(name), axis_metadata)
            for name in ["fXaxis", "fYaxis", "fZaxis"][0 : len(self.axes)]
        ]
        out = boost_histogram.Histogram(*axes, storage=storage)
        for k, v in metadata.items():
            setattr(out, k, self.member(v))

        assert len(values.shape) == len(
            axes
        ), "Number of dimensions must match number of axes"
        for i, axis in enumerate(axes):
            if not isinstance(
                axis,
                (boost_histogram.axis.IntCategory, boost_histogram.axis.StrCategory),
            ):
                continue
            slicer = (slice(None),) * i + (slice(1, None),)
            values = values[slicer]
            if sumw2 is not None:
                sumw2 = sumw2[slicer]

        view = out.view(flow=True)
        if sumw2 is not None:
            assert (
                sumw2.shape == values.shape
            ), "weights (fSumw2) and values should have same shape"
            view.value = values
            view.variance = sumw2
        else:
            view[...] = values

        return out

    def to_hist(self, metadata=None, axis_metadata=None):
        """
        Args:
            metadata (dict of str \u2192 str): Metadata to collect (keys) and
                their C++ class member names (values).
            axis_metadata (dict of str \u2192 str): Metadata to collect from
                each axis.

        Converts the histogram into a ``hist`` object.
        """
        if axis_metadata is None:
            axis_metadata = boost_axis_metadata
        if metadata is None:
            metadata = boost_metadata

        return uproot.extras.hist().Hist(
            self.to_boost(metadata=metadata, axis_metadata=axis_metadata)
        )

    # Support direct conversion to histograms, such as bh.Histogram(self) or hist.Hist(self)
    def _to_boost_histogram_(self):
        return self.to_boost()


class TH1(Histogram):
    """
    Behaviors for one-dimensional histograms: descendants of ROOT's
    ``TH1``, not including ``TProfile``, ``TH2``, ``TH3``, or their descendants.
    """

    @property
    def axes(self):
        return (self.member("fXaxis"),)

    def axis(self, axis=0):  # default axis for one-dimensional is intentional
        if axis == 0 or axis == -1 or axis == "x":
            return self.member("fXaxis")
        else:
            raise ValueError("axis must be 0 (-1) or 'x' for a TH1")

    @property
    def kind(self):
        return "COUNT"

    def values(self, flow=False):
        if hasattr(self, "_values"):
            values = self._values
        else:
            (values,) = self.base(uproot.models.TArray.Model_TArray)
            values = numpy.asarray(values, dtype=values.dtype.newbyteorder("="))
            self._values = values

        if flow:
            return values
        else:
            return values[1:-1]

    def _values_variances(self, flow):
        values = self.values(flow=True)

        if hasattr(self, "_variances"):
            variances = self._variances
        else:
            variances = numpy.zeros(values.shape, dtype=numpy.float64)
            sumw2 = self.member("fSumw2", none_if_missing=True)
            if sumw2 is not None and len(sumw2) == self.member("fNcells"):
                sumw2 = numpy.asarray(sumw2, dtype=sumw2.dtype.newbyteorder("="))
                sumw2 = numpy.reshape(sumw2, values.shape)
                positive = sumw2 > 0
                variances[positive] = sumw2[positive]
            else:
                # like ROOT's TH1::GetBinError, the error is sqrt(|content|)
                variances[...] = numpy.abs(values.astype(numpy.float64))
            self._variances = variances

        if flow:
            return values, variances
        else:
            return values[1:-1], variances[1:-1]

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
        if dd:
            return values, (xedges,)
        else:
            return values, xedges
