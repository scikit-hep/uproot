# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` for
several numerical types:

* :doc:`uproot.interpretation.numerical.AsDtype`: numbers, which can simply be
  described as a ``numpy.dtype``.
* :doc:`uproot.interpretation.numerical.AsDtypeInPlace`: a predefined array
  into which data may be overwritten.
* :doc:`uproot.interpretation.numerical.AsDouble32`: ROOT's ``Double32_t``
  packed data type.
* :doc:`uproot.interpretation.numerical.AsFloat16`: ROOT's ``Float16_t``
  packed data type.
* :doc:`uproot.interpretation.numerical.AsSTLBits`: an ``std::bitset<N>``
  for some ``N``.
"""

from __future__ import annotations

import awkward
import numpy

import uproot


def _dtype_shape(dtype):
    shape = ()
    while dtype.subdtype is not None:
        dtype, s = dtype.subdtype
        shape = shape + s
    return dtype, shape


class Numerical(uproot.interpretation.Interpretation):
    """
    Abstract superclass of numerical interpretations, including

    * :doc:`uproot.interpretation.numerical.AsDtype`
    * :doc:`uproot.interpretation.numerical.AsSTLBits`
    * :doc:`uproot.interpretation.numerical.TruncatedNumerical`
    """

    def _wrap_almost_finalized(self, array):
        return array

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
        )

        if entry_start >= entry_stop:
            output = self._prepare_output(library, length=0)

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

            output = self._prepare_output(library, length)

            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    local_start = entry_start - start
                    local_stop = entry_stop - start
                    basket_array = basket_arrays[basket_num]
                    output[:] = basket_array[local_start:local_stop]

                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    basket_array = basket_arrays[basket_num]
                    output[: stop - entry_start] = basket_array[local_start:local_stop]

                elif start < entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    basket_array = basket_arrays[basket_num]
                    output[start - entry_start :] = basket_array[local_start:local_stop]

                elif entry_start < stop and start < entry_stop:
                    basket_array = basket_arrays[basket_num]
                    output[start - entry_start : stop - entry_start] = basket_array

                start = stop

        native_dtype = output.dtype.newbyteorder("=")
        if output.dtype != native_dtype:
            output = output.astype(native_dtype)
        self.hook_before_library_finalize(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        output = self._wrap_almost_finalized(output)

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
        )

        return output

    def _prepare_output(self, library, length):
        """
        Prepare the output array in which the data is stored.

        In this default implementation, just create an empty array from the library but specializations might re-use an existing array (ex: :doc:`uproot.interpretation.numerical.AsDtypeInPlace`:)
        """
        output = library.empty((length,), self.to_dtype)
        return output


_numpy_byteorder_to_cache_key = {
    "!": "B",
    ">": "B",
    "<": "L",
    "|": "L",
    "=": "B" if numpy.dtype(">f8").isnative else "L",
}

_dtype_kind_itemsize_to_typename = {
    ("b", 1): "bool",
    ("i", 1): "int8_t",
    ("u", 1): "uint8_t",
    ("i", 2): "int16_t",
    ("u", 2): "uint16_t",
    ("i", 4): "int32_t",
    ("u", 4): "uint32_t",
    ("i", 8): "int64_t",
    ("u", 8): "uint64_t",
    ("f", 4): "float",
    ("f", 8): "double",
}


class AsDtype(Numerical):
    """
    Args:
        from_dtype (``numpy.dtype`` or its constructor argument): Data type to
            *assume* of the raw but uncompressed bytes in the ``TBasket``.
            Usually big-endian; may include named fields and a shape.
        to_dtype (None, ``numpy.dtype``, or its constructor argument): Data
            type to *convert* the data into. Usually native-endian; may include
            named fields and a shape. If None, ``to_dtype`` will be set to the
            native-endian equivalent of ``from_dtype``.

    Interpretation for any array that can be fully described as a
    ``numpy.dtype``.
    """

    def __init__(self, from_dtype, to_dtype=None):
        self._from_dtype = numpy.dtype(from_dtype)
        if to_dtype is None:
            self._to_dtype = self._from_dtype.newbyteorder("=")
        else:
            self._to_dtype = numpy.dtype(to_dtype)

    def __repr__(self):
        if self._to_dtype == self._from_dtype.newbyteorder("="):
            return f"AsDtype({str(self._from_dtype)!r})"
        else:
            return f"AsDtype({str(self._from_dtype)!r}, {str(self._to_dtype)!r})"

    def __eq__(self, other):
        return (
            type(other) is AsDtype
            and self._from_dtype == other._from_dtype
            and self._to_dtype == other._to_dtype
        )

    @property
    def from_dtype(self):
        """
        Data type to expect of the raw but uncompressed bytes in the
        ``TBasket`` data. Usually big-endian; may include named fields and a
        shape.

        Named fields (``dtype.names``) can be used to construct a NumPy
        `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`__.

        A shape (``dtype.shape``) can be used to construct a fixed-size array
        for each entry. (Not applicable to variable-length lists! See
        :doc:`uproot.interpretation.jagged.AsJagged`.) The finalized array's
        ``array.shape[1:] == dtype.shape``.
        """
        return self._from_dtype

    @property
    def to_dtype(self):
        """
        Data type to convert the data into. Usually the native-endian
        equivalent of :ref:`uproot.interpretation.numerical.AsDtype.from_dtype`;
        may include named fields and a shape.

        Named fields (``dtype.names``) can be used to construct a NumPy
        `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`__.

        A shape (``dtype.shape``) can be used to construct a fixed-size array
        for each entry. (Not applicable to variable-length lists! See
        :doc:`uproot.interpretation.jagged.AsJagged`.) The finalized array's
        ``array.shape[1:] == dtype.shape``.
        """
        return self._to_dtype

    @property
    def itemsize(self):
        """
        Number of bytes per item of
        :ref:`uproot.interpretation.numerical.AsDtype.from_dtype`.

        This number of bytes includes the fields and shape, like
        ``dtype.itemsize`` in NumPy.
        """
        return self._from_dtype.itemsize

    @property
    def inner_shape(self):
        _, s = _dtype_shape(self._from_dtype)
        return s

    @property
    def numpy_dtype(self):
        return self._to_dtype

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
        d, s = _dtype_shape(self._to_dtype)
        out = uproot._util.awkward_form(d, file, context)
        for size in s[::-1]:
            out = awkward.forms.RegularForm(out, size)
        return out

    @property
    def cache_key(self):
        def form(dtype, name):
            d, s = _dtype_shape(dtype)
            return "{}{}{}({}{})".format(
                _numpy_byteorder_to_cache_key[d.byteorder],
                d.kind,
                d.itemsize,
                ",".join(repr(x) for x in s),
                name,
            )

        if self.from_dtype.names is None:
            from_dtype = form(self.from_dtype, "")
        else:
            from_dtype = (
                "["
                + ",".join(
                    form(self.from_dtype[n], "," + repr(n))
                    for n in self.from_dtype.names
                )
                + "]"
            )

        if self.to_dtype.names is None:
            to_dtype = form(self.to_dtype, "")
        else:
            to_dtype = (
                "["
                + ",".join(
                    form(self.to_dtype[n], "," + repr(n)) for n in self.to_dtype.names
                )
                + "]"
            )

        return f"{type(self).__name__}({from_dtype},{to_dtype})"

    @property
    def typename(self):
        def form(dtype):
            d, s = _dtype_shape(dtype)
            return _dtype_kind_itemsize_to_typename[d.kind, d.itemsize] + "".join(
                "[" + str(dim) + "]" for dim in s
            )

        if self.from_dtype.names is None:
            return form(self.from_dtype)
        else:
            return (
                "struct {"
                + " ".join(
                    f"{form(self.from_dtype[n])} {n};" for n in self.from_dtype.names
                )
                + "}"
            )

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

        dtype, shape = _dtype_shape(self._from_dtype)
        try:
            output = data.view(dtype).reshape((-1, *shape))
        except ValueError as err:
            raise ValueError(
                f"""basket {basket.basket_num} in tree/branch {branch.object_path} has the wrong number of bytes ({len(data)}) """
                f"""for interpretation {self}
in file {branch.file.file_path}"""
            ) from err

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

    def reshape(self, shape):
        d, _s = _dtype_shape(self._from_dtype)
        self._from_dtype = numpy.dtype((d, shape))
        d, _s = _dtype_shape(self._to_dtype)
        self._to_dtype = numpy.dtype((d, shape))

    def inplace(self, array):
        """
        Returns a AsDtypeInPlace version of self in order to fill the given array in place.

        Example usage :
        ```
        var = np.zeros(N, dtype=np.float32)
        b = uproot.openn('afile.root')['treename']['varname']
        b.array(library='np', interpretation=b.interpretation.inplace(var) )
        ```
        """
        return AsDtypeInPlace(array, self._from_dtype)


class AsDtypeInPlace(AsDtype):
    """
    Like :doc:`uproot.interpretation.numerical.AsDtype`, but a given array is
    filled in-place, rather than creating a new output array.
    """

    def __init__(self, array, from_dtype):
        self._to_fill = array
        self._from_dtype = from_dtype
        self._to_dtype = numpy.dtype(array.dtype)

    def _prepare_output(self, library, length):
        """
        Specialized version of _prepare_output : re-use our target array kept in self._to_fill.
        """
        if library.name != "np":
            raise TypeError(
                f"AsDtypeInPlace can only be used with library 'np', not '{library.name}'"
            )

        output = self._to_fill.view(self.to_dtype)

        if length > len(output):
            raise ValueError(
                f"Requesting to fill an array of size {len(output)} (type {self._to_dtype}) with input of size {length} (type {self._from_dtype})"
            )
        return output[:length]


class AsSTLBits(Numerical):
    """
    Interpretation for ``std::bitset``.
    """

    def __init__(self):
        raise NotImplementedError

    @property
    def itemsize(self):
        return self._num_bytes + 4


class TruncatedNumerical(Numerical):
    """
    Abstract superclass for interpretations that truncate the range and
    granularity of the real number line to pack data into fewer bits.

    Subclasses are

    * :doc:`uproot.interpretation.numerical.AsDouble32`
    * :doc:`uproot.interpretation.numerical.AsFloat16`
    """

    @property
    def low(self):
        """
        Lower bound on the range of real numbers this type can express.
        """
        return self._low

    @property
    def high(self):
        """
        Upper bound on the range of real numbers this type can express.
        """
        return self._high

    @property
    def num_bits(self):
        """
        Number of bytes into which to pack these data.
        """
        return self._num_bits

    @property
    def from_dtype(self):
        """
        The ``numpy.dtype`` of the raw but uncompressed data.

        May be a
        `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`__
        of ``"exponent"`` and ``"mantissa"`` or an integer.
        """
        if self.is_truncated:
            return numpy.dtype(({"exponent": (">u1", 0), "mantissa": (">u2", 1)}, ()))
        else:
            return numpy.dtype(">u4")

    @property
    def itemsize(self):
        """
        Number of bytes in
        :ref:`uproot.interpretation.numerical.TruncatedNumerical.from_dtype`.
        """
        return self.from_dtype.itemsize

    @property
    def to_dims(self):
        """
        The ``dtype.shape`` of the ``to_dtype``.
        """
        return self._to_dims

    @property
    def is_truncated(self):
        """
        If True (:ref:`uproot.interpretation.numerical.TruncatedNumerical.low`
        and :ref:`uproot.interpretation.numerical.TruncatedNumerical.high` are
        both ``0``), the data are truly truncated.
        """
        return self._low == self._high == 0.0

    def __repr__(self):
        args = [repr(self._low), repr(self._high), repr(self._num_bits)]
        if self._to_dims != ():
            args.append(f"to_dims={self._to_dims!r}")
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._low == other._low
            and self._high == other._high
            and self._num_bits == other._num_bits
            and self._to_dims == other._to_dims
        )

    @property
    def numpy_dtype(self):
        return self.to_dtype

    @property
    def cache_key(self):
        return f"{type(self).__name__}({self._low},{self._high},{self._num_bits},{self._to_dims})"

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

        try:
            raw = data.view(self.from_dtype)
        except ValueError as err:
            raise ValueError(
                f"""basket {basket.basket_num} in tree/branch {branch.object_path} has the wrong number of bytes ({len(data)}) """
                f"""for interpretation {self} (expecting raw array of {self.from_dtype!r})
in file {branch.file.file_path}"""
            ) from err

        if self.is_truncated:
            exponent = raw["exponent"].astype(numpy.int32)
            mantissa = raw["mantissa"].astype(numpy.int32)

            exponent <<= 23
            exponent |= (mantissa & ((1 << (self.num_bits + 1)) - 1)) << (
                23 - self.num_bits
            )
            sign = ((1 << (self.num_bits + 1)) & mantissa != 0) * -2 + 1

            output = exponent.view(numpy.float32) * sign

            d, s = _dtype_shape(self.to_dtype)
            output = output.astype(d).reshape((-1, *s))

        else:
            d, s = _dtype_shape(self.to_dtype)
            output = raw.astype(d).reshape((-1, *s))
            numpy.multiply(
                output,
                float(self._high - self._low) / (1 << self._num_bits),
                out=output,
            )
            numpy.add(output, self.low, out=output)

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            raw=raw,
            output=output,
            options=options,
        )

        return output


class AsDouble32(TruncatedNumerical):
    """
    Args:
        low (float): Lower bound on the range of expressible values.
        high (float): Upper bound on the range of expressible values.
        num_bits (int): Number of bits in the representation.
        to_dims (tuple of ints): Shape of
            :ref:`uproot.interpretation.numerical.AsDouble32.to_dtype`.

    Interpretation for ROOT's ``Double32_t`` type.
    """

    def __init__(self, low, high, num_bits, to_dims=()):
        self._low = low
        self._high = high
        self._num_bits = num_bits
        self._to_dims = to_dims

        if not uproot._util.isint(num_bits) or not 2 <= num_bits <= 32:
            raise TypeError("num_bits must be an integer between 2 and 32 (inclusive)")
        if high <= low and not self.is_truncated:
            raise ValueError(f"high ({high}) must be strictly greater than low ({low})")

    @property
    def to_dtype(self):
        """
        The ``numpy.dtype`` of the output array.

        A shape (``dtype.shape``) can be used to construct a fixed-size array
        for each entry. (Not applicable to variable-length lists! See
        :doc:`uproot.interpretation.jagged.AsJagged`.) The finalized array's
        ``array.shape[1:] == dtype.shape``.
        """
        return numpy.dtype((numpy.float64, self.to_dims))

    @property
    def typename(self):
        return "Double32_t" + "".join("[" + str(dim) + "]" for dim in self._to_dims)

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
        out = awkward.forms.NumpyForm("float64")
        for size in self._to_dims[::-1]:
            out = awkward.forms.RegularForm(out, size)
        return out


class AsFloat16(TruncatedNumerical):
    """
    Args:
        low (float): Lower bound on the range of expressible values.
        high (float): Upper bound on the range of expressible values.
        num_bits (int): Number of bits in the representation.
        to_dims (tuple of ints): Shape of
            :ref:`uproot.interpretation.numerical.AsFloat16.to_dtype`.

    Interpretation for ROOT's ``Float16_t`` type.
    """

    def __init__(self, low, high, num_bits, to_dims=()):
        self._low = low
        self._high = high
        self._num_bits = num_bits
        self._to_dims = to_dims

        if not uproot._util.isint(num_bits) or not 2 <= num_bits <= 32:
            raise TypeError("num_bits must be an integer between 2 and 32 (inclusive)")
        if high <= low and not self.is_truncated:
            raise ValueError(f"high ({high}) must be strictly greater than low ({low})")

    @property
    def to_dtype(self):
        """
        The ``numpy.dtype`` of the output array.

        A shape (``dtype.shape``) can be used to construct a fixed-size array
        for each entry. (Not applicable to variable-length lists! See
        :doc:`uproot.interpretation.jagged.AsJagged`.) The finalized array's
        ``array.shape[1:] == dtype.shape``.
        """
        return numpy.dtype((numpy.float32, self.to_dims))

    @property
    def typename(self):
        return "Float16_t" + "".join("[" + str(dim) + "]" for dim in self._to_dims)

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
        out = awkward.forms.NumpyForm("float32")
        for size in self._to_dims[::-1]:
            out = awkward.forms.RegularForm(out, size)
        return out
