# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the behaviors of ``TBranchElement``, which is entirely inherited from
functions in :doc:`uproot.behaviors.TBranch.TBranch`.
"""

from __future__ import annotations

import uproot


class TBranchElement(uproot.behaviors.TBranch.TBranch):
    """
    Behaviors for a ``TBranchElement``, which mostly consist of array-reading
    methods.

    Since a :doc:`uproot.behaviors.TBranchElement.TBranchElement` is a
    :doc:`uproot.behaviors.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_branchelement["subbranch"]
        my_branchelement["subbranch"]["subsubbranch"]
        my_branchelement["subbranch/subsubbranch"]
    """
