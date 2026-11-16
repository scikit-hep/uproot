# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` and
temporary array for grouped data; usually applied to a ``TBranch`` that does
not contain data but has subbranches that do.
"""

from __future__ import annotations

import awkward

import uproot


class AsGrouped(uproot.interpretation.Interpretation):
    """
    Args:
        branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` that
            represents the group.
        subbranches (dict of str \u2192 :doc:`uproot.behaviors.TBranch.TBranch`): Names
            and interpretations of the ``TBranches`` that actually contain data.
        typename (None or str): If None, construct a plausible C++ typename.
            Otherwise, take the suggestion as given.

    Interpretation for a group of arrays, usually because they are all
    subbranches of the same :doc:`uproot.behaviors.TBranch.TBranch`.

    Each :doc:`uproot.interpretation.library.Library` presents a group
    differently: :doc:`uproot.interpretation.library.NumPy` puts arrays
    in a dict, :doc:`uproot.interpretation.library.Awkward` makes an
    array of records, :doc:`uproot.interpretation.library.Pandas` makes
    a ``pandas.DataFrame``, etc.
    """

    def __init__(self, branch, subbranches, typename=None):
        self._branch = branch
        self._subbranches = subbranches
        self._typename = typename

    @property
    def branch(self):
        """
        The ``TBranch`` that represents the group.
        """
        return self._branch

    @property
    def subbranches(self):
        """
        The ``TBranches`` that contain the actual data.
        """
        return self._subbranches

    def __repr__(self):
        return f"AsGroup({self._branch}, {self._subbranches})"

    def __eq__(self, other):
        return (
            isinstance(other, AsGrouped)
            and self._branch == other._branch
            and self._subbranches == other._subbranches
        )

    @property
    def cache_key(self):
        return "{}({},[{}])".format(
            type(self).__name__,
            self._branch.name,
            ",".join(
                f"{x!r}:{y.cache_key}"
                for x, y in self._subbranches.items()
                if y is not None
            ),
        )

    @property
    def typename(self):
        if self._typename is not None:
            return self._typename
        else:
            return "(group of {})".format(
                ", ".join(
                    f"{x}:{y.typename}"
                    for x, y in self._subbranches.items()
                    if y is not None
                )
            )

    def awkward_form(
        self,
        file,
        context=None,
        index_format="i64",
        header=False,
        tobject_header=False,
        breadcrumbs=(),
    ):
        context = self._make_context(
            context, index_format, header, tobject_header, breadcrumbs
        )
        names = []
        fields = []
        for x, y in self._subbranches.items():
            if y is not None:
                names.append(x)
                fields.append(y.awkward_form(file, context))
        return awkward.forms.RecordForm(fields, names)

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        options,
    ):
        raise ValueError(
            """grouping branches like {} should not be read directly; instead read the subbranches:

    {}

in file {}
in object {}""".format(
                repr(self._branch.name),
                ", ".join(repr(x) for x in self._subbranches),
                self._branch.file.file_path,
                self._branch.object_path,
            )
        )

    def final_array(
        self,
        basket_arrays,
        entry_start,
        entry_stop,
        entry_offsets,
        library,
        branch,
        options,
    ):
        raise ValueError(
            """grouping branches like {} should not be read directly; instead read the subbranches:

    {}

in file {}
in object {}""".format(
                repr(self._branch.name),
                ", ".join(repr(x) for x in self._subbranches),
                self._branch.file.file_path,
                self._branch.object_path,
            )
        )
