# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versionless model for ``TBasket``, including much of the
functionality of basket-reading.

Includes both "embedded" ``TBaskets`` (as a member of TBranch) and "free"
``TBaskets`` (top-level objects, located by ``TKeys``).
"""

from __future__ import annotations

import struct

import numpy

import uproot

_tbasket_format1 = struct.Struct(">ihiIhh")
_tbasket_format2 = struct.Struct(">Hiiii")
_tbasket_offsets_dtype = numpy.dtype(">i4")


class Model_TBasket(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TBasket``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def __repr__(self):
        basket_num = self._basket_num if self._basket_num is not None else "(unknown)"
        return f"<TBasket {basket_num} of {self._parent.name!r} at 0x{id(self):012x}>"

    @property
    def raw_data(self):
        """
        The raw but uncompressed data in the ``TBasket``, which combines data
        content with entry offsets, if the latter exists.

        If there are no entry offsets, this is identical to
        :ref:`uproot.models.TBasket.Model_TBasket.data`.
        """
        return self._raw_data

    @property
    def data(self):
        """
        The uncompressed data content in the ``TBasket``, not including any
        entry offsets, if they exist.

        If there are no entry offsets, this is identical to
        :ref:`uproot.models.TBasket.Model_TBasket.raw_data`.
        """
        return self._data

    @property
    def byte_offsets(self):
        """
        The index where each entry starts and stops in the
        :ref:`uproot.models.TBasket.Model_TBasket.data`, not including header.

        The first offset is ``0`` and the number of offsets is one greater than
        the number of entries, such that the last offset is the length of
        :ref:`uproot.models.TBasket.Model_TBasket.data`.
        """
        return self._byte_offsets

    def array(self, interpretation=None, library="ak", ak_add_doc=False):
        """
        The ``TBasket`` data and entry offsets as an array, given an
        :doc:`uproot.interpretation.Interpretation` (or the ``TBranch`` parent's
        :ref:`uproot.behaviors.TBranch.TBranch.interpretation`) and a
        ``library``.
        """
        if interpretation is None:
            interpretation = self._parent.interpretation
        library = uproot.interpretation.library._regularize_library(library)

        interp_options = {"ak_add_doc": ak_add_doc}

        basket_array = interpretation.basket_array(
            self.data,
            self.byte_offsets,
            self,
            self._parent,
            self._parent.context,
            self._members["fKeylen"],
            library,
            interp_options,
        )

        return interpretation.final_array(
            {0: basket_array},
            0,
            self.num_entries,
            [0, self.num_entries],
            library,
            self._parent,
            interp_options,
        )

    @property
    def counts(self):
        """
        The number of items in each entry as a NumPy array, derived from the
        parent ``TBranch``'s
        :ref:`uproot.behaviors.TBranch.TBranch.count_branch`. If there is
        no such branch (e.g. the data are ``std::vector``), then this method
        returns None.
        """
        count_branch = self._parent.count_branch
        if count_branch is not None:
            entry_offsets = count_branch.entry_offsets
            entry_start = entry_offsets[self._basket_num]
            entry_stop = entry_offsets[self._basket_num + 1]
            return count_branch.array(
                entry_start=entry_start, entry_stop=entry_stop, library="np"
            )
        else:
            return None

    @property
    def basket_num(self):
        """
        The index of this ``TBasket`` within its ``TBranch``.
        """
        return self._basket_num

    @property
    def entry_start_stop(self):
        """
        The starting and stopping entry numbers for this ``TBasket``.
        """
        return self._parent.basket_entry_start_stop(self._basket_num)

    @property
    def key_version(self):
        """
        The instance version of the ``TKey`` for this ``TBasket`` (which is
        deserialized along with the ``TBasket``, unlike normal objects).
        """
        return self._key_version

    @property
    def num_entries(self):
        """
        The number of entries in this ``TBasket``.
        """
        return self._members["fNevBuf"]

    @property
    def is_embedded(self):
        """
        If this ``TBasket`` is embedded within its ``TBranch`` (i.e. must be
        deserialized as part of the ``TBranch``), then ``is_embedded`` is True.

        If this ``TBasket`` is a free-standing object, then ``is_embedded`` is
        False.
        """
        return self._members["fNbytes"] <= self._members["fKeylen"]

    @property
    def uncompressed_bytes(self):
        """
        The number of bytes for the uncompressed data, including the TKey header.

        If the ``TBasket`` is uncompressed, this is equal to
        :ref:`uproot.models.TBasket.Model_TBasket.compressed_bytes`.
        """
        return self._members["fKeylen"] + self._members["fObjlen"]

    @property
    def compressed_bytes(self):
        """
        The number of bytes for the compressed data, including the TKey header
        (which is always uncompressed).

        If the ``TBasket`` is uncompressed, this is equal to
        :ref:`uproot.models.TBasket.Model_TBasket.uncompressed_bytes`.
        """
        return self._members["fNbytes"]

    @property
    def block_compression_info(self):
        """
        For compressed ``TBaskets``, a tuple of 3-tuples containing

        ``(name of algorithm, num compressed bytes, num uncompressed bytes)``

        to describe the actual compression algorithms and sizes encountered in
        each block of data.

        The name of the algorithm can be ``"ZLIB"``, ``"LZMA"``, ``"LZ4"``, or
        ``"ZSTD"``.

        For uncompressed ``TBaskets``, this is None.
        """
        return self._block_compression_info

    @property
    def border(self):
        """
        The byte position of the boundary between data content and entry offsets.

        Equal to ``self.member("fLast") - self.member("fKeylen")``.
        """
        return self._members["fLast"] - self._members["fKeylen"]

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        assert isinstance(self._parent, uproot.behaviors.TBranch.TBranch)
        self._basket_num = context.get("basket_num")

        (
            self._members["fNbytes"],
            self._key_version,
            self._members["fObjlen"],
            self._members["fDatime"],
            self._members["fKeylen"],
            self._members["fCycle"],
        ) = cursor.fields(chunk, _tbasket_format1, context)

        # skip the class name, name, and title
        cursor.move_to(
            self._cursor.index + self._members["fKeylen"] - _tbasket_format2.size - 1
        )

        (
            self._members["fVersion"],
            self._members["fBufferSize"],
            self._members["fNevBufSize"],
            self._members["fNevBuf"],
            self._members["fLast"],
        ) = cursor.fields(chunk, _tbasket_format2, context)

        cursor.skip(1)

        self._block_compression_info = None

        if not context.get("read_basket", True):
            self._byte_offsets = None
            self._raw_data = None
            self._data = None
            return

        if self.is_embedded:
            # https://github.com/root-project/root/blob/0e6282a641b65bdf5ad832882e547ca990e8f1a5/tree/tree/inc/TBasket.h#L62-L65
            maybe_entry_size = self._members["fNevBufSize"]
            num_entries = self._members["fNevBuf"]
            key_length = self._members["fKeylen"]

            # Embedded TBaskets are always uncompressed; be sure to copy any memmap arrays
            chunk = chunk.detach_memmap()

            if maybe_entry_size * num_entries + key_length != self._members["fLast"]:
                raw_byte_offsets = cursor.bytes(
                    chunk, 8 + self.num_entries * 4, context
                ).view(_tbasket_offsets_dtype)
                cursor.skip(-4)

                # subtracting fKeylen makes a new buffer and converts to native endian
                self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
                # so modifying it in place doesn't have non-local consequences
                self._byte_offsets[-1] = self.border

            else:
                self._byte_offsets = None

            # second key has no new information
            cursor.skip(self._members["fKeylen"])

            self._raw_data = None
            if self.border == 0:
                self._data = numpy.empty(0, dtype=numpy.uint8)
            else:
                self._data = cursor.bytes(chunk, self.border, context)

        else:
            compressed_bytes = self._members["fNbytes"] - self._members["fKeylen"]
            uncompressed_bytes = self._members["fObjlen"]

            if compressed_bytes != uncompressed_bytes:
                self._block_compression_info = []
                uncompressed = uproot.compression.decompress(
                    chunk,
                    cursor,
                    {},
                    compressed_bytes,
                    uncompressed_bytes,
                    self._block_compression_info,
                )
                self._block_compression_info = tuple(self._block_compression_info)
                self._raw_data = uncompressed.get(
                    0,
                    uncompressed_bytes,
                    uproot.source.cursor.Cursor(0),
                    context,
                )
            else:
                # Uncompressed; be sure to copy any memmap arrays
                chunk = chunk.detach_memmap()

                self._raw_data = cursor.bytes(chunk, uncompressed_bytes, context)

            if self.border != uncompressed_bytes:
                self._data = self._raw_data[: self.border]
                raw_byte_offsets = self._raw_data[self.border :].view(
                    _tbasket_offsets_dtype
                )

                # subtracting fKeylen makes a new buffer and converts to native endian
                self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
                # so modifying it in place doesn't have non-local consequences
                self._byte_offsets[-1] = self.border

            else:
                self._data = self._raw_data
                self._byte_offsets = None


uproot.classes["TBasket"] = Model_TBasket
