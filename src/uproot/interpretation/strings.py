# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` and
temporary array for string data.

Note that :doc:`uproot.interpretation.strings.AsStrings` is an interpretation for
top-level strings, but :doc:`uproot.containers.AsString` can be nested within any
other :doc:`uproot.containers.AsContainer`.

The :doc:`uproot.interpretation.strings.StringArray` class only holds data while
an array is being built from ``TBaskets``. Its final form is determined by the
:doc:`uproot.interpretation.library.Library`.
"""

from __future__ import annotations

import struct
import threading

import awkward
import numpy

import uproot

_string_4byte_size = struct.Struct(">I")


class AsStrings(uproot.interpretation.Interpretation):
    """
    Args:
        header_bytes (int): Number of bytes to skip at the beginning of each
            entry.
        length_bytes ("1-5" or "4"): Method used to determine the length of
            a string: "1-5" means one byte if the length is less than 256,
            otherwise the true length is in the next four bytes; "4" means
            always four bytes.
        typename (None or str): If None, construct a plausible C++ typename.
            Otherwise, take the suggestion as given.
        original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): If
            this interpretation is derived from
            :ref:`uproot.interpretation.objects.AsObjects.simplify`, this is a
            reminder of the original
            :ref:`uproot.interpretation.objects.AsObjects.model`.

    An :doc:`uproot.interpretation.Interpretation` for an array of strings.

    This cannot be nested within other
    :doc:`uproot.interpretation.Interpretation` objects; it can only represent
    a ``TBranch`` that only contains strings (not strings within ``std::vector``,
    for instance).

    Note that the :doc:`uproot.containers.AsString` class is for strings nested
    within other objects.

    (:ref:`uproot.interpretation.objects.AsObjects.simplify` converts an
    :doc:`uproot.interpretation.objects.AsObjects` of
    :doc:`uproot.containers.AsString` into a
    :doc:`uproot.interpretation.strings.AsStrings`.)
    """

    _forth_codes = {
        "1-5": """
            input stream

            output out-main uint8
            output out-offsets int64

            0 out-offsets <- stack

            begin
                stream !B-> stack dup 255 = if drop stream !I-> stack then dup out-offsets +<- stack stream #!B-> out-main
            again
        """,
        "4": """
            input stream

            output out-main uint8
            output out-offsets int64

            0 out-offsets <- stack

            begin
                stream !I-> stack dup out-offsets +<- stack stream #B-> out-main
            again
        """,
    }

    def __init__(
        self, header_bytes=0, length_bytes="1-5", typename=None, original=None
    ):
        self._header_bytes = header_bytes

        if length_bytes in ("1-5", "4"):
            self._length_bytes = length_bytes
        else:
            raise ValueError("length_bytes must be '1-5' or '4'")

        self._typename = typename
        self._original = original
        self._forth_code = None
        self._threadlocal = threading.local()

    @property
    def header_bytes(self):
        """
        The number of bytes to skip at the beginning of each entry.
        """
        return self._header_bytes

    @property
    def length_bytes(self):
        """
        Method used to determine the length of a string: "1-5" means one byte
        if the length is less than 256, otherwise the true length is in the
        next four bytes; "4" means always four bytes.
        """
        return self._length_bytes

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
        args = []
        if self._header_bytes != 0:
            args.append(f"header_bytes={self._header_bytes}")
        if self._length_bytes != "1-5":
            args.append(f"length_bytes={self._length_bytes!r}")
        return "AsStrings({})".format(", ".join(args))

    def __eq__(self, other):
        return (
            isinstance(other, AsStrings)
            and self._header_bytes == other._header_bytes
            and self._length_bytes == other._length_bytes
        )

    @property
    def typename(self):
        if self._typename is None:
            return "char*"
        else:
            return self._typename

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
            awkward.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        )

    @property
    def cache_key(self):
        return f"{type(self).__name__}({self._header_bytes},{self._length_bytes!r})"

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
        if (
            isinstance(library, uproot.interpretation.library.Awkward)
            and byte_offsets is None
        ):
            import awkward.forth

            if self._length_bytes == "1-5" or self._length_bytes == "4":
                if not hasattr(self._threadlocal, "forth_vm"):
                    self._threadlocal.forth_vm = awkward.forth.ForthMachine64(
                        self._forth_codes[self._length_bytes]
                    )

                self._threadlocal.forth_vm.begin({"stream": numpy.array(data)})
                self._threadlocal.forth_vm.resume(raise_read_beyond=False)
                offsets = self._threadlocal.forth_vm.output("out-offsets")
                data = self._threadlocal.forth_vm.output("out-main")
                self._threadlocal.forth_vm.reset()

                return awkward.Array(
                    awkward.contents.ListOffsetArray(
                        awkward.index.Index64(offsets),
                        awkward.contents.NumpyArray(
                            data, parameters={"__array__": "char"}
                        ),
                        parameters={"__array__": "string"},
                    )
                )

            else:
                raise AssertionError(repr(self._length_bytes))

        else:
            if byte_offsets is None:
                counts = numpy.empty(len(data), dtype=numpy.int32)
                outdata = numpy.empty(len(data), dtype=data.dtype)

                pos = 0
                entry_num = 0
                len_outdata = 0

                if self._length_bytes == "1-5":
                    while True:
                        if pos >= len(data):
                            break
                        size = data[pos]
                        pos += 1
                        if size == 255:
                            (size,) = _string_4byte_size.unpack(data[pos : pos + 4])
                            pos += 4
                        counts[entry_num] = size
                        entry_num += 1
                        outdata[len_outdata : len_outdata + size] = data[
                            pos : pos + size
                        ]
                        len_outdata += size
                        pos += size

                elif self._length_bytes == "4":
                    while True:
                        if pos >= len(data):
                            break
                        (size,) = _string_4byte_size.unpack(data[pos : pos + 4])
                        pos += 4
                        counts[entry_num] = size
                        entry_num += 1
                        outdata[len_outdata : len_outdata + size] = data[
                            pos : pos + size
                        ]
                        len_outdata += size
                        pos += size

                else:
                    raise AssertionError(repr(self._length_bytes))

                counts = counts[:entry_num]
                data = outdata[:len_outdata]

            else:
                byte_starts = byte_offsets[:-1] + self._header_bytes
                byte_stops = byte_offsets[1:]

                if self._length_bytes == "1-5":
                    length_header_size = numpy.ones(len(byte_starts), dtype=numpy.int32)
                    length_header_size[data[byte_starts] == 255] += 4
                elif self._length_bytes == "4":
                    length_header_size = numpy.full(
                        len(byte_starts), 4, dtype=numpy.int32
                    )
                else:
                    raise AssertionError(repr(self._length_bytes))

                byte_starts += length_header_size

                mask = numpy.zeros(len(data), dtype=numpy.int8)
                mask[byte_starts[byte_starts < len(data)]] = 1
                numpy.add.at(mask, byte_stops[byte_stops < len(data)], -1)
                numpy.cumsum(mask, out=mask)
                data = data[mask.view(numpy.bool_)]

                counts = byte_stops - byte_starts

            offsets = numpy.empty(len(counts) + 1, dtype=numpy.int32)
            offsets[0] = 0
            numpy.cumsum(counts, out=offsets[1:])

            data = data.tobytes()

        output = StringArray(offsets, data)

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

        if any(not isinstance(x, StringArray) for x in basket_arrays.values()):
            # basket_array only returns awkward arrays for library="ak" and
            # baskets without entry offsets; since the presence of entry offsets
            # is a property of each TBasket, the other baskets of the same
            # branch may still be StringArrays, which are converted here.
            assert isinstance(library, uproot.interpretation.library.Awkward)
            awkward_arrays = {
                k: (_string_array_to_awkward(v) if isinstance(v, StringArray) else v)
                for k, v in basket_arrays.items()
            }
            trimmed = uproot._util.trim_final(
                awkward_arrays, entry_start, entry_stop, entry_offsets, library, branch
            )
            output = awkward.concatenate(trimmed, mergebool=False, highlevel=False)

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

        else:
            basket_offsets = {}
            basket_content = {}
            for k, v in basket_arrays.items():
                basket_offsets[k] = v.offsets
                basket_content[k] = v.content

            if entry_start >= entry_stop:
                output = StringArray(library.zeros((1,), numpy.int64), b"")

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
                        off, cnt = (
                            basket_offsets[basket_num],
                            basket_content[basket_num],
                        )
                        offsets[:] = (
                            before
                            - off[local_start]
                            + off[local_start : local_stop + 1]
                        )
                        before += off[local_stop] - off[local_start]
                        contents.append(cnt[off[local_start] : off[local_stop]])

                    elif start <= entry_start < stop:
                        local_start = entry_start - start
                        local_stop = stop - start
                        off, cnt = (
                            basket_offsets[basket_num],
                            basket_content[basket_num],
                        )
                        offsets[: stop - entry_start + 1] = (
                            before
                            - off[local_start]
                            + off[local_start : local_stop + 1]
                        )
                        before += off[local_stop] - off[local_start]
                        contents.append(cnt[off[local_start] : off[local_stop]])

                    elif start < entry_stop <= stop:
                        local_start = 0
                        local_stop = entry_stop - start
                        off, cnt = (
                            basket_offsets[basket_num],
                            basket_content[basket_num],
                        )
                        offsets[start - entry_start :] = (
                            before
                            - off[local_start]
                            + off[local_start : local_stop + 1]
                        )
                        before += off[local_stop] - off[local_start]
                        contents.append(cnt[off[local_start] : off[local_stop]])

                    elif entry_start < stop and start < entry_stop:
                        off, cnt = (
                            basket_offsets[basket_num],
                            basket_content[basket_num],
                        )
                        offsets[start - entry_start : stop - entry_start + 1] = (
                            before - off[0] + off
                        )
                        before += off[-1] - off[0]
                        contents.append(cnt[off[0] : off[-1]])

                    start = stop

                output = StringArray(offsets, b"".join(contents))

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


def _string_array_to_awkward(string_array):
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(numpy.asarray(string_array.offsets, numpy.int64)),
            awkward.contents.NumpyArray(
                numpy.frombuffer(string_array.content, numpy.uint8),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        )
    )


class StringArray:
    """
    Args:
        offsets (array of ``numpy.int32``): Starting and stopping indexes for
            each string. The length of the ``offsets`` is one greater than the
            number of strings.
        content (array): Contiguous array of character data for all strings of
            the array.

    Temporary array filled by
    :ref:`uproot.interpretation.strings.AsStrings.basket_array`, which will be
    turned into a NumPy, Awkward, or other array, depending on the specified
    :doc:`uproot.interpretation.library.Library`.
    """

    def __init__(self, offsets, content):
        self._offsets = offsets
        self._content = content

    def __repr__(self):
        if len(self._content) > 100:
            left, right = self._content[:45], self._content[-45:]
            content = repr(left) + " ... " + repr(right)
        else:
            content = repr(self._content)
        return f"StringArray({self._offsets}, {content})"

    @property
    def offsets(self):
        """
        Starting and stopping indexes for each string. The length of the
        ``offsets`` is one greater than the number of strings.
        """
        return self._offsets

    @property
    def content(self):
        """
        Contiguous array of character data for all strings of the array.
        """
        return self._content

    def __getitem__(self, where):
        data = self._content[self._offsets[where] : self._offsets[where + 1]]
        return uproot._util.ensure_str(data)

    def __len__(self):
        return len(self._offsets) - 1

    def __iter__(self):
        start = self._offsets[0]
        content = self._content
        for stop in self._offsets[1:]:
            yield uproot._util.ensure_str(content[start:stop])
            start = stop
