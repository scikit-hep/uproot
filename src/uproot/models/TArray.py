# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines versionless models for ``TArray`` and its subclasses.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import awkward
import numpy

import uproot

_tarray_format1 = struct.Struct(">i")


class Model_TArray(uproot.model.Model, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TArray`` and its subclasses.

    These also satisfy Python's abstract ``Sequence`` protocol.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:  # pragma: no cover  (always False: no header read)
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._members["fN"] = cursor.field(chunk, _tarray_format1, context)
        self._data = cursor.array(chunk, self._members["fN"], self.dtype, context)

    def __array__(self, *args, **kwargs):
        if len(args) == len(kwargs) == 0:
            return self._data
        else:
            return numpy.array(self._data, *args, **kwargs)

    @property
    def nbytes(self):
        return self._data.nbytes

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return "<{}{} {} at 0x{:012x}>".format(
            self.classname,
            version,
            numpy.array2string(
                self._data,
                max_line_width=numpy.inf,
                separator=", ",
                formatter={"float": lambda x: f"{x:g}"},
                threshold=6,
            ),
            id(self),
        )

    def tojson(self):
        return self._data.tolist()

    @classmethod
    def awkward_form(cls, file, context):
        return awkward.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls.dtype, file, context),
        )

    writable = True

    def _to_writable_postprocess(self, original):
        self._data = original._data

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)

        out.append(_tarray_format1.pack(self._members["fN"]))
        out.append(self._data.tobytes())

        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 1
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TArrayC(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayC`` (``dtype(">i1")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i1")


class Model_TArrayS(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayS`` (``dtype(">i2")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i2")


class Model_TArrayI(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayI`` (``dtype(">i4")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i4")


class Model_TArrayL(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayL`` (``dtype(">i8")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i8")


class Model_TArrayL64(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayL64`` (``dtype(">i8")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i8")


class Model_TArrayF(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayF`` (``dtype(">f4")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">f4")


class Model_TArrayD(Model_TArray):
    """
    A versionless :doc:`uproot.model.Model` for ``TArrayD`` (``dtype(">f8")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">f8")


uproot.classes["TArray"] = Model_TArray
uproot.classes["TArrayC"] = Model_TArrayC
uproot.classes["TArrayS"] = Model_TArrayS
uproot.classes["TArrayI"] = Model_TArrayI
uproot.classes["TArrayL"] = Model_TArrayL
uproot.classes["TArrayL64"] = Model_TArrayL64
uproot.classes["TArrayF"] = Model_TArrayF
uproot.classes["TArrayD"] = Model_TArrayD
