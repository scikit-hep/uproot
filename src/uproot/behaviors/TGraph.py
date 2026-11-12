# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``TGraph``.
"""

from __future__ import annotations

import numpy


class TGraph:
    """
    Behaviors for TGraph: return values as a NumPy array
    """

    # This is really just the `values` portion of TGraphAsymmErrors

    def _normalize_array(self, key):
        values = self.member(key)
        return numpy.asarray(values, dtype=values.dtype.newbyteorder("="))

    def values(self, axis="both"):
        """
        Args:
            axis (int or str): If ``0``, ``-2``, or ``"x"``, get the *x* axis.
                If ``1``, ``-1``, or ``"y"``, get the *y* axis. If ``"both"``,
                get ``"x"`` and ``"y"`` axes as a 2-tuple.

        Returns the values of all points in the scatter plot, either as a
        1-dimensional NumPy array (if ``axis`` selects only one) or as a 2-tuple
        (if ``axis="both"``).
        """
        if axis in [0, -2, "x"]:
            return self._normalize_array("fX")
        elif axis in [1, -1, "y"]:
            return self._normalize_array("fY")
        elif axis == "both":
            return (self.values("x"), self.values("y"))
        else:
            raise ValueError(
                "axis must be 0 (-2, 'x'), 1 (-1, 'y'), or 'both' for a TGraph"
            )
