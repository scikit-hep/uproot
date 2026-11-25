# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines languages for expressions passed to
:ref:`uproot.behaviors.TBranch.HasBranches.arrays` (and similar).

The default is :doc:`uproot.language.python.PythonLanguage`.

All languages must be subclasses of :doc:`uproot.language.Language`.
"""

from __future__ import annotations


class Language:
    """
    Abstract class for all languages, which are used to compute the expressions
    that are passed to :ref:`uproot.behaviors.TBranch.HasBranches.arrays` (and
    similar).

    The default is :doc:`uproot.language.python.PythonLanguage`.
    """
