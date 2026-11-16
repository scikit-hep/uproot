# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behavior of ``TH2Poly``.
"""

from __future__ import annotations

import uproot


class TH2Poly(uproot.behaviors.TH1.Histogram):
    """
    Behaviors for two-dimensional polygon histograms: ROOT's ``TH2Poly``.
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
            raise ValueError("axis must be 0 (-2), 1 (-1) or 'x', 'y' for a TH2")
