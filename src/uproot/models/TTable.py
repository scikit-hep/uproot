# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines versioned models for ``TTable``.
"""

from __future__ import annotations

import struct
from collections import namedtuple
from enum import Enum

import numpy

import uproot

_tdataset_format1 = struct.Struct(">I")


class Model_TDataSet(uproot.model.Model):
    """
    A :doc:`uproot.model.Model` for ``TDataSet``.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

        self._bases.append(
            uproot.models.TNamed.Model_TNamed.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._members["fParent"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )
        self._members["fList"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    base_names_versions = [
        ("TNamed", 1),
    ]
    member_names = ["fParent", "fList"]
    class_flags = {}
    class_code = None


class EColumnType(Enum):
    """
    An :doc:`Enum` of possible TTable column types.
    """

    kNAN = 0
    kFloat = 1
    kInt = 2
    kLong = 3
    kShort = 4
    kDouble = 5
    kUInt = 6
    kULong = 7
    kUShort = 8
    kUChar = 9
    kChar = 10
    kPtr = 11
    kBool = 12


_dtype = {
    EColumnType.kFloat: numpy.dtype(">f4"),
    EColumnType.kInt: numpy.dtype(">i4"),
    EColumnType.kLong: numpy.dtype(">i8"),
    EColumnType.kShort: numpy.dtype(">i2"),
    EColumnType.kDouble: numpy.dtype(">d"),
    EColumnType.kUInt: numpy.dtype(">u4"),
    EColumnType.kULong: numpy.dtype(">u8"),
    EColumnType.kUShort: numpy.dtype(">u2"),
    EColumnType.kChar: numpy.dtype("u1"),
    EColumnType.kBool: numpy.dtype("?"),
}

tableDescriptor_st = namedtuple(
    "tableDescriptor_st",
    [
        "fColumnName",
        "fIndexArray0",
        "fIndexArray1",
        "fIndexArray2",
        "fOffset",
        "fSize",
        "fTypeSize",
        "fDimensions",
        "fType",
    ],
)


_ttabledescriptor4_format1 = struct.Struct(">iqq")


class Model_TTableDescriptor_v4(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTableDescriptor`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

        self._bases.append(
            Model_TDataSet.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        (
            self._members["fN"],
            self._members["fMaxIndex"],
            self._members["fSize"],
        ) = cursor.fields(chunk, _ttabledescriptor4_format1, context)

        self._columns = []
        for _ in range(self._members["fMaxIndex"]):
            column = tableDescriptor_st(
                *cursor.fields(chunk, struct.Struct(">32s3iiiiii"), context)
            )
            column = column._replace(
                fColumnName=column.fColumnName.rstrip(b"\x00").decode("utf-8"),
                fType=EColumnType(column.fType),
            )
            self._columns.append(column)

    base_names_versions = []
    member_names = ["fN", "fMaxIndex", "fSize"]
    class_flags = {"has_read_object_any": True}
    class_code = None


class Model_TTableDescriptor(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TTableDescriptor``.
    """

    known_versions = {4: Model_TTableDescriptor_v4}


_ttable4_format1 = struct.Struct(">iqq")


class Model_TTable_v4(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TTable`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

        ioDescriptor = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

        self._bases.append(
            Model_TDataSet.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        (
            self._members["fN"],
            self._members["fMaxIndex"],
            self._members["fSize"],
        ) = cursor.fields(chunk, _ttable4_format1, context)

        assert sum(col.fSize for col in ioDescriptor._columns) == self._members["fSize"]

        buf = cursor.bytes(
            chunk, self._members["fSize"] * self._members["fMaxIndex"], context
        )

        def getFormat(col):
            dtype = _dtype[col.fType]
            if col.fDimensions:
                dtype = (dtype, (col.fSize // _dtype[col.fType].itemsize,))
            return dtype

        dtype = numpy.dtype(
            {
                "names": [col.fColumnName for col in ioDescriptor._columns],
                "formats": [getFormat(col) for col in ioDescriptor._columns],
                "offsets": [col.fOffset for col in ioDescriptor._columns],
                "itemsize": self._members["fSize"],
            }
        )
        self._data = numpy.frombuffer(buf, dtype=dtype)

    @property
    def data(self):
        view = self._data.view()
        view.flags.writeable = False
        return view

    base_names_versions = []
    member_names = ["fN", "fMaxIndex", "fSize"]
    class_flags = {"has_read_object_any": True}
    class_code = None


class Model_TTable(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TTable``.
    """

    known_versions = {4: Model_TTable_v4}


uproot.classes["TDataSet"] = Model_TDataSet
uproot.classes["TTableDescriptor"] = Model_TTableDescriptor
uproot.classes["TTable"] = Model_TTable
