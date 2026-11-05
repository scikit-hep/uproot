# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` and
temporary array for jagged (variable-length list) data.

The :doc:`uproot.interpretation.jagged.JaggedArray` class only holds data while
an array is being built from ``TBaskets``. Its final form is determined by
:doc:`uproot.interpretation.library`.
"""

from __future__ import annotations

import awkward
import numpy

import uproot


def fast_divide(array, divisor):
    """
    Vectorized division function for a scalar divisor that is often a power of
    2. Usually bit-shifts, rather than dividing.
    """
    if divisor == 1:
        return array
    elif divisor == 2:
        return numpy.right_shift(array, 1)
    elif divisor == 4:
        return numpy.right_shift(array, 2)
    elif divisor == 8:
        return numpy.right_shift(array, 3)
    else:
        return numpy.floor_divide(array, divisor)


class AsJagged(uproot.interpretation.Interpretation):
    """
    Args:
        content (:doc:`uproot.interpretation.numerical.AsDtype` or :doc:`uproot.interpretation.objects.AsStridedObjects`): Interpretation
            for data in the nested, variable-length lists.
        header_bytes (int): Number of bytes to skip at the beginning of each
            entry.
        typename (None or str): If None, construct a plausible C++ typename.
            Otherwise, take the suggestion as given.
        original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): If
            this interpretation is derived from
            :ref:`uproot.interpretation.objects.AsObjects.simplify`, this is a
            reminder of the original
            :ref:`uproot.interpretation.objects.AsObjects.model`.

    Interpretation for any array that can be described as variable-length lists
    of :doc:`uproot.interpretation.numerical.AsDtype`.
    """

    def __init__(self, content, header_bytes=0, typename=None, original=None):
        if not isinstance(content, uproot.interpretation.numerical.Numerical):
            raise TypeError("AsJagged content can only be Numerical")
        self._content = content
        self._header_bytes = header_bytes
        self._typename = typename
        self._original = original

    @property
    def content(self):
        """
        The :doc:`uproot.interpretation.numerical.AsDtype` or
        :doc:`uproot.interpretation.objects.AsStridedObjects` that interprets
        data in the nested, variable-length lists.
        """
        return self._content

    @property
    def header_bytes(self):
        """
        The number of bytes to skip at the beginning of each entry.
        """
        return self._header_bytes

    @property
    def original(self):
        """
        If not None, this was the original
        :ref:`uproot.interpretation.objects.AsObjects.model` from an
        :doc:`uproot.interpretation.objects.AsObjects` that was simplified
        into this :doc:`uproot.interpretation.jagged.AsJagged`.
        """
        return self._original

    def __repr__(self):
        if self._header_bytes == 0:
            return f"AsJagged({self._content!r})"
        else:
            return f"AsJagged({self._content!r}, header_bytes={self._header_bytes})"

    def __eq__(self, other):
        return (
            isinstance(other, AsJagged)
            and self._content == other._content
            and self._header_bytes == other._header_bytes
        )

    @property
    def numpy_dtype(self):
        return numpy.dtype(object)

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
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(self._content, file, context),
        )

    @property
    def cache_key(self):
        return f"{type(self).__name__}({self._content.cache_key},{self._header_bytes})"

    @property
    def typename(self):
        if self._typename is None:
            content = self._content.typename
            try:
                i = content.index("[")
                return content[:i] + "[]" + content[i:]
            except ValueError:
                return content + "[]"
        else:
            return self._typename

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
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )

        if byte_offsets is None:
            counts = basket.counts
            if counts is None:
                raise uproot.deserialization.DeserializationError(
                    "missing offsets (and missing count branch) for jagged array",
                    None,
                    None,
                    context,
                    branch.file.file_path,
                )
            else:
                itemsize = self._content.from_dtype.itemsize
                counts = numpy.multiply(counts, itemsize)
                byte_offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
                byte_offsets[0] = 0
                numpy.cumsum(counts, out=byte_offsets[1:])

        if self._header_bytes == 0:
            offsets = fast_divide(byte_offsets, self._content.itemsize)
            content = self._content.basket_array(
                data, None, basket, branch, context, cursor_offset, library, options
            )
            output = JaggedArray(offsets, content)

        else:
            byte_starts = byte_offsets[:-1] + self._header_bytes
            byte_stops = byte_offsets[1:]

            # mask out the headers
            header_offsets = numpy.arange(self._header_bytes)
            header_idxs = (byte_offsets[:-1] + header_offsets[:, numpy.newaxis]).ravel()
            mask = numpy.full(len(data), True, dtype=numpy.bool_)
            mask[header_idxs] = False
            data = data[mask]

            content = self._content.basket_array(
                data, None, basket, branch, context, cursor_offset, library, options
            )

            byte_counts = byte_stops - byte_starts
            counts = fast_divide(byte_counts, self._content.itemsize)

            offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
            offsets[0] = 0
            numpy.cumsum(counts, out=offsets[1:])
            output = JaggedArray(offsets, content)

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )

        return output

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
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            options=options,
        )

        basket_offsets = {}
        basket_content = {}
        for k, v in basket_arrays.items():
            basket_offsets[k] = v.offsets
            basket_content[k] = v.content

        if entry_start >= entry_stop:
            offsets = library.zeros((1,), numpy.int64)
            content = numpy.empty(0, self.content.to_dtype)
            output = JaggedArray(offsets, content)

        else:
            length = 0
            start = entry_offsets[0]
            for _, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    length += entry_stop - entry_start
                elif start <= entry_start < stop:
                    length += stop - entry_start
                elif start <= entry_stop <= stop:
                    length += entry_stop - start
                elif entry_start < stop and start <= entry_stop:
                    length += stop - start
                start = stop

            offsets = numpy.empty((length + 1,), numpy.int64)

            before = 0
            start = entry_offsets[0]
            contents = []
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    local_start = entry_start - start
                    local_stop = entry_stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[:] = (
                        before - off[local_start] + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[: stop - entry_start + 1] = (
                        before - off[local_start] + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif start < entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[start - entry_start :] = (
                        before - off[local_start] + off[local_start : local_stop + 1]
                    )
                    before += off[local_stop] - off[local_start]
                    contents.append(cnt[off[local_start] : off[local_stop]])

                elif entry_start < stop and start < entry_stop:
                    off, cnt = basket_offsets[basket_num], basket_content[basket_num]
                    offsets[start - entry_start : stop - entry_start + 1] = (
                        before - off[0] + off
                    )
                    before += off[-1] - off[0]
                    contents.append(cnt[off[0] : off[-1]])

                start = stop

            content = numpy.empty((before,), self.content.to_dtype)
            before = 0
            for cnt in contents:
                content[before : before + len(cnt)] = cnt
                before += len(cnt)

            content = self._content._wrap_almost_finalized(content)

            output = JaggedArray(offsets, content)

            self.hook_before_library_finalize(
                basket_arrays=basket_arrays,
                entry_start=entry_start,
                entry_stop=entry_stop,
                entry_offsets=entry_offsets,
                library=library,
                branch=branch,
                output=output,
            )

        output = library.finalize(
            output, branch, self, entry_start, entry_stop, options
        )

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
            options=options,
        )

        return output


class JaggedArray:
    """
    Args:
        offsets (array of ``numpy.int32``): Starting and stopping entries for
            each variable-length list. The length of the ``offsets`` is one
            greater than the number of lists.
        content (array): Contiguous array for data in all nested lists of the
            jagged array.

    Temporary array filled by
    :ref:`uproot.interpretation.jagged.AsJagged.basket_array`, which will be
    turned into a NumPy, Awkward, or other array, depending on the specified
    :doc:`uproot.interpretation.library.Library`.
    """

    def __init__(self, offsets, content):
        self._offsets = offsets
        self._content = content

    def __repr__(self):
        return f"JaggedArray({self._offsets}, {self._content})"

    @property
    def offsets(self):
        """
        Starting and stopping entries for each variable-length list. The length
        of the ``offsets`` is one greater than the number of lists.
        """
        return self._offsets

    @property
    def content(self):
        """
        Contiguous array for data in all nested lists of the jagged array.
        """
        return self._content

    def __getitem__(self, where):
        return self._content[self._offsets[where] : self._offsets[where + 1]]

    def __len__(self):
        return len(self._offsets) - 1

    def __iter__(self):
        start = self._offsets[0]
        content = self._content
        for stop in self._offsets[1:]:
            yield content[start:stop]
            start = stop

    def parents_localindex(self, entry_start, entry_stop):
        """
        Args:
            entry_start (int): First entry to include.
            entry_stop (int): FIrst entry to exclude (one greater than the last
                entry to include)

        Returns the "parents" and "localindex" of this jagged array, using
        Awkward 0 terminology.

        The "parents" is an array of integers with the same length as
        :ref:`uproot.interpretation.jagged.JaggedArray.content` that indicates
        which list each item belongs to.

        The "localindex" is an array of integers with the same length that
        indicates which subentry each item is, within its nested list.
        """
        counts = self._offsets[1:] - self._offsets[:-1]

        assert entry_stop - entry_start == len(self._offsets) - 1
        indexes = numpy.arange(len(self._offsets) - 1, dtype=numpy.int64)

        parents = numpy.repeat(indexes, counts)

        localindex = numpy.arange(
            self._offsets[0], self._offsets[-1], dtype=numpy.int64
        )
        localindex -= self._offsets[parents]

        return parents + entry_start, localindex
