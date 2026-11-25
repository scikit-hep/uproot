# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``TAxis``, an axis of a histogram or profile plot.
"""

from __future__ import annotations

import operator

import numpy


class AxisTraits:
    """
    Describes read-only properties of a histogram axis.

    For example, ``axis.traits.discrete`` is True if the histogram has
    labels; False otherwise.
    """

    def __init__(self, axis):
        self._axis = axis

    def __repr__(self):
        return f"AxisTraits({self._axis!r})"

    @property
    def circular(self):
        """
        True if the axis "wraps around" (always False for ROOT histograms).
        """
        return False

    @property
    def discrete(self):
        """
        True if bins are discrete: if they have string-valued labels.
        """
        return self._axis._bin_labels() is not None


class TAxis:
    """
    Describes a histogram axis.
    """

    def _bin_labels(self):
        """
        The string-valued labels ordered by bin, or None if not every bin is
        labeled.

        ROOT keeps the labels in the order they were set, with each label's bin
        number in its ``fUniqueID``.
        """
        if not hasattr(self, "_cached_bin_labels"):
            self._cached_bin_labels = self._compute_bin_labels()
        return self._cached_bin_labels

    def _compute_bin_labels(self):
        fNbins = self.member("fNbins")
        fLabels = self.member("fLabels", none_if_missing=True)
        if fLabels is None:
            return None

        # like ROOT's TAxis::GetBinLabel, the first label with a bin's number is
        # that bin's, and labels without a valid bin number are ignored
        out = [None] * fNbins
        any_bin_numbers = False
        for label in fLabels:
            bin_number = label.member("@fUniqueID", none_if_missing=True)
            if bin_number is not None and 1 <= bin_number <= fNbins:
                any_bin_numbers = True
                if out[bin_number - 1] is None:
                    out[bin_number - 1] = str(label)

        if all(x is not None for x in out):
            return out
        elif not any_bin_numbers and len(fLabels) == fNbins:
            # no bin numbers at all; assume the labels were set in bin order
            return [str(x) for x in fLabels]
        else:
            return None

    def __len__(self):
        """
        The number of bins in the axis.
        """
        return self.member("fNbins")

    def __getitem__(self, where):
        """
        Returns the label at ``where`` if it exists or the interval at ``where``.

        The indexing assumes that ``flow=False``.
        """
        fNbins = self.member("fNbins")
        labels = self._bin_labels()

        if labels is not None:
            return labels[where]

        try:
            index = operator.index(where)
        except TypeError:
            # arrays and slices: index into the same intervals that __iter__
            # yields, keeping the shape of the index
            out = self.intervals()[where]
            return out[..., 0], out[..., 1]

        # integers: constant time, without building every interval
        if index < 0:
            index += fNbins
        if not 0 <= index < fNbins:
            raise IndexError(f"bin index {where} out of range for {fNbins} bins")

        fXbins = self.member("fXbins", none_if_missing=True)
        if fXbins is not None and len(fXbins) == fNbins + 1:
            return float(fXbins[index]), float(fXbins[index + 1])

        # the same edges as the numpy.linspace in intervals()
        fXmin, fXmax = self.member("fXmin"), self.member("fXmax")
        step = (fXmax - fXmin) / fNbins
        low = index * step + fXmin
        high = fXmax if index + 1 == fNbins else (index + 1) * step + fXmin
        return low, high

    def __iter__(self):
        """
        Iterate over the output of ``__getitem__``.
        """
        labels = self._bin_labels()

        if labels is not None:
            yield from labels
        else:
            yield from self.intervals()

    def __contains__(self, x):
        """
        Returns True if ``x`` is one of the labels or intervals for this axis;
        False otherwise.
        """
        return any(
            x == y if isinstance(y, str) else numpy.array_equal(x, y) for y in self
        )

    def __eq__(self, other):
        """
        Two axes are equal if they have the same type and ``list(self) == list(other)``.
        """
        if type(self) is not type(other):
            return False

        self_fNbins = self.member("fNbins")
        other_fNbins = other.member("fNbins")
        if self_fNbins != other_fNbins:
            return False

        self_labels = self._bin_labels()
        other_labels = other._bin_labels()

        if self_labels is not None and other_labels is not None:
            return self_labels == other_labels
        elif self_labels is None and other_labels is None:
            return numpy.array_equal(self.edges(), other.edges())
        else:
            return False

    @property
    def traits(self):
        """
        Describes read-only properties of a histogram axis.

        For example, ``axis.traits.discrete`` is True if the histogram has
        labels; False otherwise.
        """
        return AxisTraits(self)

    @property
    def low(self):
        """
        The low edge of the first normal (finite-width) bin.

        For ROOT histograms, numerical edges exist even if the axis also has
        string-valued labels.
        """
        return self.member("fXmin")

    @property
    def high(self):
        """
        The high edge of the last normal (finite-width) bin.

        For ROOT histograms, numerical edges exist even if the axis also has
        string-valued labels.
        """
        return self.member("fXmax")

    @property
    def width(self):
        """
        The average bin width (or only bin width if the binning is uniform).
        """
        fNbins = self.member("fNbins")
        fXbins = self.member("fXbins", none_if_missing=True)

        if fXbins is None or len(fXbins) != fNbins + 1:
            return (self.member("fXmax") - self.member("fXmin")) / fNbins
        else:
            return self.widths().mean()

    def labels(self, flow=False):
        """
        Args:
            flow (bool): If True, include ``"underflow"`` and ``"overflow"``
                before and after the normal (finite-width) bin labels (if they
                exist).

        If string-valued labels exist, this returns them as a Python list of
        Python strings. Otherwise, this returns None.

        Setting ``flow=True`` increases the length of the output by two.
        """
        out = self._bin_labels()

        if out is not None:
            if flow:
                return ["underflow", *out, "overflow"]
            else:
                return list(out)  # a copy, so that callers can't modify the cache
        else:
            return None

    def edges(self, flow=False):
        """
        Args:
            flow (bool): If True, include ``-inf`` and ``inf`` before and
                after the normal (finite-width) bin edges.

        Returns numerical edges between bins as a one-dimensional ``numpy.ndarray``
        of ``numpy.float64``.

        Even with ``flow=False``, the number of edges is *one greater than* the
        number of normal (finite-width) bins because they represent "fenceposts"
        between the bins, including one below and one above the full range.

        Setting ``flow=True`` increases the length of the output by two.

        For ROOT histograms, numerical edges exist even if the axis also has
        string-valued labels.
        """
        fNbins = self.member("fNbins")
        fXbins = self.member("fXbins", none_if_missing=True)

        if fXbins is None or len(fXbins) != fNbins + 1:
            fXbins = numpy.linspace(
                self.member("fXmin"), self.member("fXmax"), fNbins + 1
            )

        if flow:
            out = numpy.empty(fNbins + 3, dtype=numpy.float64)
            out[0] = -numpy.inf
            out[-1] = numpy.inf
            out[1:-1] = fXbins
        else:
            out = numpy.asarray(fXbins, dtype=fXbins.dtype.newbyteorder("="))

        return out

    def intervals(self, flow=False):
        """
        Args:
            flow (bool): If True, include ``[-inf, min]`` and ``[max, inf]``
                before and after the normal (finite-width) intervals.

        Returns low, high pairs for each bin interval as a two-dimensional
        ``numpy.ndarray`` of ``numpy.float64``.

        With ``flow=False``, the number of intervals is equal to the number of
        normal (finite-width) bins.

        Setting ``flow=True`` increases the length of the output by two.

        For ROOT histograms, numerical intervals exist even if the axis also has
        string-valued labels.
        """
        fNbins = self.member("fNbins")
        fXbins = self.member("fXbins", none_if_missing=True)

        if fXbins is None or len(fXbins) != fNbins + 1:
            fXbins = numpy.linspace(
                self.member("fXmin"), self.member("fXmax"), fNbins + 1
            )

        if flow:
            out = numpy.empty((fNbins + 2, 2), dtype=numpy.float64)
            out[0, 0] = -numpy.inf
            out[-1, 1] = numpy.inf
            out[1:, 0] = fXbins
            out[:-1, 1] = fXbins
        else:
            out = numpy.empty((fNbins, 2), dtype=numpy.float64)
            out[:, 0] = fXbins[:-1]
            out[:, 1] = fXbins[1:]

        return out

    def centers(self, flow=False):
        """
        Args:
            flow (bool): If True, include ``-inf`` and ``inf`` before and after
                the normal (finite) bin centers.

        Returns bin center positions as a one-dimensional ``numpy.ndarray`` of
        ``numpy.float64``.

        With ``flow=False``, the number of bin centers is equal to the number of
        normal (finite-width) bins.

        Setting ``flow=True`` increases the length of the output by two.

        For ROOT histograms, numerical bin centers exist even if the axis also has
        string-valued labels.
        """
        edges = self.edges(flow=flow)
        return (edges[1:] + edges[:-1]) / 2.0

    def widths(self, flow=False):
        """
        Args:
            flow (bool): If True, include ``-inf`` and ``inf`` before and after
                the normal (finite) bin widths.

        Returns bin widths as a one-dimensional ``numpy.ndarray`` of
        ``numpy.float64``.

        With ``flow=False``, the number of bin widths is equal to the number of
        normal (finite-width) bins.

        Setting ``flow=True`` increases the length of the output by two.

        For ROOT histograms, numerical bin widths exist even if the axis also has
        string-valued labels.
        """
        fNbins = self.member("fNbins")
        fXbins = self.member("fXbins", none_if_missing=True)

        if not flow and (fXbins is None or len(fXbins) != fNbins + 1):
            width = (self.member("fXmax") - self.member("fXmin")) / fNbins
            return numpy.broadcast_to(width, (fNbins,))
        else:
            edges = self.edges(flow=flow)
            return edges[1:] - edges[:-1]
