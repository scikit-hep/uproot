# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` as
a base class for all user-defined custom interpretations, and provides
a :doc:`uproot.interpretation.Interpretation` for extracting binary data
from ``TBasket`` objects.
"""

from __future__ import annotations

import awkward
import numpy

import uproot
import uproot.behaviors.TBranch
import uproot.extras
import uproot.interpretation


class CustomInterpretation(uproot.interpretation.Interpretation):
    def __init__(
        self,
        branch: uproot.behaviors.TBranch.TBranch,
        context: dict,
        simplify: bool,
    ):
        """
        Args:
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` to
                interpret as an array.
            context (dict): Auxiliary data used in deserialization.
            simplify (bool): If True, call
                :ref:`uproot.interpretation.objects.AsObjects.simplify` on any
                :doc:`uproot.interpretation.objects.AsObjects` to try to get a
                more efficient interpretation.

        Accept arguments from `uproot.interpretation.identify.interpretation_of`.
        """
        self._branch = branch
        self._context = context
        self._simplify = simplify

    @classmethod
    def match_branch(
        cls,
        branch: uproot.behaviors.TBranch.TBranch,
        context: dict,
        simplify: bool,
    ) -> bool:
        """
        Args:
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` to
                interpret as an array.
            context (dict): Auxiliary data used in deserialization.
            simplify (bool): If True, call
                :ref:`uproot.interpretation.objects.AsObjects.simplify` on any
                :doc:`uproot.interpretation.objects.AsObjects` to try to get a
                more efficient interpretation.

        Accept arguments from `uproot.interpretation.identify.interpretation_of`,
        determine whether this interpretation can be applied to the given branch.
        """
        raise NotImplementedError

    @property
    def typename(self) -> str:
        """
        The name of the type of the interpretation.
        """
        return self._branch.streamer.typename

    @property
    def cache_key(self) -> str:
        """
        The cache key of the interpretation.
        """
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other

    def __repr__(self) -> str:
        """
        The string representation of the interpretation.
        """
        return self.__class__.__name__

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
        """
        Concatenate the arrays from the baskets and return the final array.
        """
        basket_entry_starts = numpy.array(entry_offsets[:-1])
        basket_entry_stops = numpy.array(entry_offsets[1:])

        basket_start_idx = numpy.where(basket_entry_starts <= entry_start)[0].max()
        basket_end_idx = numpy.where(basket_entry_stops >= entry_stop)[0].min()

        arr_to_concat = [
            basket_arrays[i] for i in range(basket_start_idx, basket_end_idx + 1)
        ]

        relative_entry_start = entry_start - basket_entry_starts[basket_start_idx]
        relative_entry_stop = entry_stop - basket_entry_starts[basket_start_idx]

        if isinstance(arr_to_concat[0], numpy.ndarray):
            tot_array = numpy.concatenate(arr_to_concat)
            return tot_array[relative_entry_start:relative_entry_stop]

        if isinstance(arr_to_concat[0], awkward.Array):
            tot_array = awkward.concatenate(arr_to_concat)
            return tot_array[relative_entry_start:relative_entry_stop]

        pandas = uproot.extras.pandas()
        if isinstance(arr_to_concat[0], pandas.Series):
            tot_array = pandas.concat(arr_to_concat, ignore_index=True)
            return tot_array.iloc[relative_entry_start:relative_entry_stop]

        raise TypeError(
            f"Unsupported array type: {type(arr_to_concat)}. "
            "Expected an Awkward Array, NumPy array, or Pandas DataFrame."
        )


class AsBinary(uproot.interpretation.Interpretation):
    """
    Return binary data of the ``TBasket``. Pass an instance of this class
    to :ref:`uproot.behaviors.TBranch.TBranch.array` like this:

    .. code-block:: python
        binary_data = branch.array(interpretation=AsBinary())

    """

    @property
    def cache_key(self) -> str:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        interp_options,
    ):
        if byte_offsets is not None:
            counts = byte_offsets[1:] - byte_offsets[:-1]
        else:
            counts = None

        if library.name == "ak":
            if counts is not None:
                return awkward.unflatten(data, counts)
            else:
                fSize = branch.streamer.member("fSize")
                return awkward.from_numpy(data.reshape(-1, fSize))

        elif library.name == "np":
            if counts is not None:
                assert (
                    numpy.unique(counts).size <= 1
                ), "The byte offsets must be uniform for NumPy arrays."

                bytes_per_event = counts[0] if counts.size != 0 else 0
                return data.reshape(-1, bytes_per_event)
            else:
                fSize = branch.streamer.member("fSize")
                return data.reshape(-1, fSize).view(">u1")
        else:
            raise ValueError(
                f"Unsupported library: {library.name}, can only use 'ak' or 'np'."
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

        basket_entry_starts = numpy.array(entry_offsets[:-1])
        basket_entry_stops = numpy.array(entry_offsets[1:])

        basket_start_idx = numpy.where(basket_entry_starts <= entry_start)[0].max()
        basket_end_idx = numpy.where(basket_entry_stops >= entry_stop)[0].min()

        arr_to_concat = [
            basket_arrays[i] for i in range(basket_start_idx, basket_end_idx + 1)
        ]

        relative_entry_start = entry_start - basket_entry_starts[basket_start_idx]
        relative_entry_stop = entry_stop - basket_entry_starts[basket_start_idx]

        if library.name == "ak":
            return awkward.concatenate(arr_to_concat)[
                relative_entry_start:relative_entry_stop
            ]

        elif library.name == "np":
            return numpy.concatenate(arr_to_concat)[
                relative_entry_start:relative_entry_stop
            ]

        else:
            raise ValueError(
                f"Unsupported library: {library.name}, can only use 'ak' or 'np'."
            )
