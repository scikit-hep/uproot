# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines the :doc:`uproot.source.cursor.Cursor`, which maintains
a thread-local pointer into a :doc:`uproot.source.chunk.Chunk` and performs
the lowest level of interpretation (numbers, strings, raw arrays, etc.).
"""

from __future__ import annotations

import datetime
import struct
import sys

import numpy

import uproot

_printable_characters = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"
    "NOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)
_raw_double32 = struct.Struct(">f")
_raw_float16 = struct.Struct(">BH")

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#basic-types
_rntuple_string_length = struct.Struct("<I")
_rntuple_datetime = struct.Struct("<Q")


class Cursor:
    """
    Args:
        index (int): Global seek position in the ROOT file or local position
            in an uncompressed :doc:`uproot.source.chunk.Chunk`.
        origin (int): Zero-point for numerical keys in ``refs``.
        refs (None or dict): References to data already read in
            :doc:`uproot.deserialization.read_object_any`.

    Represents a seek point in a ROOT file, which may be held for later
    reference or advanced while interpreting data from a
    :doc:`uproot.source.chunk.Chunk`.

    A cursor also holds references to previously read data that might be
    requested by :doc:`uproot.deserialization.read_object_any`.
    """

    def __init__(self, index: int, origin: int = 0, refs: dict | None = None):
        self._index = index
        self._origin = origin
        self._refs = refs

    def __repr__(self):
        o = "" if self._origin == 0 else f", origin={self._origin}"

        if self._refs is None or len(self._refs) == 0:
            r = ""
        elif self._refs is None or len(self._refs) < 3:
            r = ", {} refs: {}".format(
                len(self._refs), ", ".join(str(x) for x in self._refs)
            )
        else:
            r = ", {} refs: {}...".format(
                len(self._refs), ", ".join(str(x) for x in list(self._refs)[:3])
            )

        return f"Cursor({self._index}{o}{r})"

    @property
    def index(self) -> int:
        """
        Global seek position in the ROOT file or local position in an
        uncompressed :doc:`uproot.source.chunk.Chunk`.
        """
        return self._index

    @property
    def origin(self) -> int:
        """
        Zero-point for numerical keys in
        :ref:`uproot.source.cursor.Cursor.refs`.
        """
        return self._origin

    @property
    def refs(self) -> dict:
        """
        References to data already read in
        :doc:`uproot.deserialization.read_object_any`.
        """
        if self._refs is None:
            self._refs = {}
        return self._refs

    def displacement(self, other: Cursor = None) -> int:
        """
        The number of bytes between this :doc:`uproot.source.cursor.Cursor`
        and its :ref:`uproot.source.cursor.Cursor.origin` (if None)
        or the ``other`` :doc:`uproot.source.cursor.Cursor` (if provided).

        If the displacement is positive, ``self`` is later in the file than the
        ``origin`` or ``other``; if negative, it is earlier.
        """
        if other is None:
            return self._index - self._origin
        else:
            return self._index - other._index

    def copy(self, link_refs: bool = True) -> Cursor:
        """
        Returns a copy of this :doc:`uproot.source.cursor.Cursor`. If
        ``link_refs`` is True, any :ref:`uproot.source.cursor.Cursor.refs`
        will be *referenced*, rather than *copied*.
        """
        if link_refs or self._refs is None:
            return Cursor(self._index, origin=self._origin, refs=self._refs)
        else:
            return Cursor(self._index, origin=self._origin, refs=dict(self._refs))

    def move_to(self, index: int):
        """
        Move the :ref:`uproot.source.cursor.Cursor.index` to a specified seek
        position.
        """
        self._index = index

    def skip(self, num_bytes: int):
        """
        Move the :ref:`uproot.source.cursor.Cursor.index` forward
        ``num_bytes``.
        """
        self._index += num_bytes

    def skip_after(self, obj):
        """
        Move the :ref:`uproot.source.cursor.Cursor.index` just after an object
        that has a starting ``obj.cursor`` and an expected ``obj.num_bytes``.
        """
        start_cursor = getattr(obj, "cursor", None)
        num_bytes = getattr(obj, "num_bytes", None)
        if (
            start_cursor is None
            or not isinstance(start_cursor, Cursor)
            or num_bytes is None
        ):
            raise TypeError(
                "Cursor.skip_after can only be used on an object with a "
                f"`cursor` and `num_bytes`, not {type(obj)}"
            )
        self._index = start_cursor.index + num_bytes

    def skip_over(self, chunk: uproot.source.chunk.Chunk, context: dict):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.

        Move the :ref:`uproot.source.cursor.Cursor.index` to a seek position
        beyond the serialized data for an object that can be interpreted with
        :doc:`uproot.deserialization.numbytes_version`.

        Returns True if successful (cursor has moved), False otherwise (cursor
        has NOT moved).
        """
        num_bytes, _version, _is_memberwise = uproot.deserialization.numbytes_version(
            chunk, self, context, move=False
        )
        if num_bytes is None:
            return False
        else:
            self._index += num_bytes
            return True

    def fields(
        self,
        chunk: uproot.source.chunk.Chunk,
        format: struct.Struct,
        context: dict,
        move: bool = True,
    ):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            format (``struct.Struct``): Specification to interpret the bytes of
                data.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` with a
        specified format. Returns a tuple of data whose types and length are
        determined by the ``format``.
        """
        start = self._index
        stop = start + format.size
        if move:
            self._index = stop
        return format.unpack(chunk.get(start, stop, self, context))

    def field(
        self,
        chunk: uproot.source.chunk.Chunk,
        format: struct.Struct,
        context: dict,
        move: bool = True,
    ):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            format (``struct.Struct``): Specification to interpret the bytes of
                data.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` with a
        format that only specifies one field, returning a single item instead of
        a tuple.
        """
        start = self._index
        stop = start + format.size
        if move:
            self._index = stop
        return format.unpack(chunk.get(start, stop, self, context))[0]

    def double32(
        self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True
    ) -> float:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as
        ROOT's ``Double32_t`` type, returning the Python ``float``.
        """
        # https://github.com/root-project/root/blob/e87a6311278f859ca749b491af4e9a2caed39161/io/io/src/TBufferFile.cxx#L448-L464
        start = self._index
        stop = start + _raw_double32.size
        if move:
            self._index = stop
        return _raw_double32.unpack(chunk.get(start, stop, self, context))[0]

    def float16(
        self,
        chunk: uproot.source.chunk.Chunk,
        num_bits: int,
        context: dict,
        move: bool = True,
    ) -> float:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            num_bits (int): Number of bits in the mantissa.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as
        ROOT's ``Float16_t`` type, returning the Python ``float``.
        """
        # https://github.com/root-project/root/blob/e87a6311278f859ca749b491af4e9a2caed39161/io/io/src/TBufferFile.cxx#L432-L442
        # https://github.com/root-project/root/blob/e87a6311278f859ca749b491af4e9a2caed39161/io/io/src/TBufferFile.cxx#L482-L499

        start = self._index
        stop = start + _raw_float16.size
        if move:
            self._index = stop

        exponent, mantissa = _raw_float16.unpack(chunk.get(start, stop, self, context))
        out = numpy.array([exponent], numpy.int32)
        out <<= 23
        out |= (mantissa & ((1 << (num_bits + 1)) - 1)) << (23 - num_bits)
        out = out.view(numpy.float32)

        if (1 << (num_bits + 1) & mantissa) != 0:
            out = -out

        return out.item()

    def byte(self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a raw
        byte.
        """
        out = chunk.get(self._index, self._index + 1, self, context)
        if move:
            self._index += 1
        return out

    def bytes(self, chunk, length, context, move=True):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            length (int): Number of bytes to retrieve.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as raw
        bytes with a given ``length``.
        """
        start = self._index
        stop = start + length
        if move:
            self._index = stop
        return chunk.get(start, stop, self, context)

    def array(
        self,
        chunk: uproot.source.chunk.Chunk,
        length: int,
        dtype: numpy.dtype,
        context: dict,
        move: bool = True,
    ) -> numpy.ndarray:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            length (int): Number of bytes to retrieve.
            dtype (``numpy.dtype``): Data type for the array.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a
        one-dimensional array with a given ``length`` and ``dtype``.
        """
        start = self._index
        stop = start + length * dtype.itemsize
        if move:
            self._index = stop
        return numpy.frombuffer(chunk.get(start, stop, self, context), dtype=dtype)

    _u1 = numpy.dtype("u1")
    _i4 = numpy.dtype(">i4")

    def bytestring(
        self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True
    ) -> bytes:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a
        bytestring.

        The first byte is taken to be the length of the subsequent string unless
        it is equal to 255, in which case, the next 4 bytes are taken to be an
        ``numpy.int32`` length.
        """
        start = self._index
        stop = start + 1
        length = chunk.get(start, stop, self, context)[0]
        if length == 255:
            start = stop
            stop = start + 4
            length_data = chunk.get(start, stop, self, context)
            length = numpy.frombuffer(length_data, dtype=self._u1).view(self._i4)[0]
        start = stop
        stop = start + int(length)
        if move:
            self._index = stop
        return chunk.get(start, stop, self, context).tobytes()

    def string(
        self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True
    ) -> str:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a
        UTF-8 encoded string.

        The first byte is taken to be the length of the subsequent string unless
        it is equal to 255, in which case, the next 4 bytes are taken to be an
        ``numpy.int32`` length.
        """
        return self.bytestring(chunk, context, move=move).decode(
            errors="surrogateescape"
        )

    def bytestring_with_length(
        self,
        chunk: uproot.source.chunk.Chunk,
        context: dict,
        length: int,
        move: bool = True,
    ) -> bytes:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            length (int): Number of bytes in the bytestring.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a
        bytestring.
        """
        start = self._index
        stop = start + length
        if move:
            self._index = stop
        data = chunk.get(start, stop, self, context)
        return data.tobytes()

    def string_with_length(
        self,
        chunk: uproot.source.chunk.Chunk,
        context: dict,
        length: int,
        move: bool = True,
    ) -> str:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            length (int): Number of bytes in the string.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a
        UTF-8 encoded string.
        """
        return self.bytestring_with_length(chunk, context, length, move=move).decode(
            errors="surrogateescape"
        )

    def classname(
        self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True
    ) -> str:
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            context (dict): Auxiliary data used in deserialization.
            move (bool): If True, move the
                :ref:`uproot.source.cursor.Cursor.index` past the fields;
                otherwise, leave it where it is.

        Interpret data at this :ref:`uproot.source.cursor.Cursor.index` as a
        null-terminated, UTF-8 encoded string.
        """
        remainder = chunk.remainder(self._index, self, context)
        terminator = remainder.tobytes().find(0)
        if terminator < 0:
            raise OSError(
                f"""C-style string has no terminator (null byte) in Chunk {chunk.start}:{chunk.stop}
of file path {chunk.source.file_path}"""
            )
        local_stop = terminator + 1

        if move:
            self._index += local_stop

        return remainder[: local_stop - 1].tobytes().decode(errors="surrogateescape")

    def rntuple_string(
        self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True
    ) -> str:
        if move:
            length = self.field(chunk, _rntuple_string_length, context)
            return self.string_with_length(chunk, context, length)
        else:
            index = self._index
            out = self.rntuple_string(chunk, context, move=True)
            self._index = index
            return out

    def rntuple_datetime(
        self, chunk: uproot.source.chunk.Chunk, context: dict, move: bool = True
    ) -> datetime.datetime:
        raw = self.field(chunk, _rntuple_datetime, context, move=move)
        return datetime.datetime.fromtimestamp(raw)

    def debug(
        self,
        chunk: uproot.source.chunk.Chunk,
        context: dict = {},  # noqa: B006 (it's not actually mutated in the function)
        limit_bytes: int | None = None,
        dtype: numpy.dtype = None,
        offset: int = 0,
        stream=sys.stdout,
    ):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Data to examine.
            context (dict): Auxiliary data used in deserialization.
            limit_bytes (None or int): Number of bytes to limit the output to.
                A line of debugging output (without any ``offset``) is 20 bytes,
                so multiples of 20 show full lines. If None, everything is
                shown to the end of the :doc:`uproot.source.chunk.Chunk`,
                which might be large.
            dtype (None, ``numpy.dtype``, or its constructor argument): If None,
                present only the bytes as decimal values (0-255). Otherwise,
                also interpret them as an array of a given NumPy type.
            offset (int): Number of bytes to skip before interpreting a ``dtype``;
                can be helpful if the numerical values are out of phase with
                the first byte shown. Not to be confused with ``skip_bytes``,
                which determines which bytes are shown at all. Any ``offset``
                values that are equivalent modulo ``dtype.itemsize`` show
                equivalent interpretations.
            stream (object with a ``write(str)`` method): Stream to write the
                debugging output to.

        Peek at data by printing it to the ``stream`` (usually stdout). The data
        are always presented as decimal bytes and ASCII characters, but may
        also be interpreted as numbers.

        Example output with ``dtype=">f4"`` and ``offset=3``.

        .. code-block::

            --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
            123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
              {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                                    1.1             2.2             3.3             4.4
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
                --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
                        5.5             6.6             7.7             8.8             9.9
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                202   0   0  67  74   0   0  67 151 128   0 123 123
                --- --- ---   C   J --- ---   C --- --- ---   {   {
                      101.0           202.0           303.0
        """
        data = chunk.remainder(self._index, self, context)
        if limit_bytes is not None:
            data = data[:limit_bytes]

        if dtype is not None:
            if not isinstance(dtype, numpy.dtype):
                dtype = numpy.dtype(dtype)

            interpreted = [None] * len(data)
            i = offset
            interpreted_length = (
                (len(data) - offset) // dtype.itemsize
            ) * dtype.itemsize
            for x in data[offset : offset + interpreted_length].view(dtype):
                i += dtype.itemsize
                interpreted[i - 1] = x

            formatter = f"{{0:>{dtype.itemsize * 4 - 1}.{dtype.itemsize * 4 - 1}s}}"

        for line_start in range(0, int(numpy.ceil(len(data) / 20.0)) * 20, 20):
            line_data = data[line_start : line_start + 20]

            prefix = ""
            if dtype is not None:
                nones = 0
                for x in interpreted[line_start:]:
                    if x is None:
                        nones += 1
                    else:
                        break
                fill = max(0, dtype.itemsize - 1 - nones)
                line_interpreted = [None] * fill + interpreted[
                    line_start : line_start + 20
                ]
                prefix = "    " * fill
                interpreted_prefix = "    " * (fill + nones + 1 - dtype.itemsize)

            stream.write(prefix + ("--+-" * 20) + "\n")
            stream.write(prefix + " ".join(f"{x:3d}" for x in line_data) + "\n")
            stream.write(
                prefix
                + " ".join(
                    f"{x:>3c}" if chr(x) in _printable_characters else "---"
                    for x in line_data
                )
                + "\n"
            )

            if dtype is not None:
                stream.write(
                    interpreted_prefix
                    + " ".join(
                        formatter.format(str(x))
                        for x in line_interpreted
                        if x is not None
                    )
                    + "\n"
                )
