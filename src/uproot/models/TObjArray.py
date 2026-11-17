# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a versionless model for ``TObjArray``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import uproot

_tobjarray_format1 = struct.Struct(">ii")

_rawstreamer_TObjArray_v3 = (
    None,
    b"@\x00\x02\x04\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xee\x00\t@\x00\x00\x17\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\tTObjArray\x00\xa9\x9eeR\x00\x00\x00\x03@\x00\x01\xc5\xff\xff\xff\xffTObjArray\x00@\x00\x01\xb3\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x86\xff\xff\xff\xffTStreamerBase\x00@\x00\x00p\x00\x03@\x00\x00f\x00\x04@\x00\x007\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0eTSeqCollection\x1bSequenceable collection ABC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfcl;\xc6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x00@\x00\x00\x80\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00e\x00\x02@\x00\x00_\x00\x04@\x00\x001\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfLowerBound\x18Lower bound of the array\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x8c\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00q\x00\x02@\x00\x00k\x00\x04@\x00\x00=\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05fLast*Last element in array containing an object\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int\x00",
    "TObjArray",
    3,
)


class Model_TObjArray(uproot.model.Model, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TObjArray``.

    This also satisfies Python's abstract ``Sequence`` protocol.
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
            uproot.models.TObject.Model_TObject.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1, context
        )

        self._data = []
        for _ in range(self._members["fSize"]):
            item = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self._parent
            )
            self._data.append(item)

    writable = True

    def _to_writable_postprocess(self, original):
        # this model is versionless, so keep the version it was read with
        self._instance_version = original._instance_version
        self._data = [None if x is None else x.to_writable() for x in original._data]

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)
        out.append(uproot.serialization.string(self._members["fName"]))
        out.append(
            _tobjarray_format1.pack(
                self._members["fSize"], self._members["fLowerBound"]
            )
        )
        for item in self._data:
            uproot.serialization._serialize_object_any(out, item, None)
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = self._instance_version
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return f"<{self.classname}{version} of {len(self)} items at 0x{id(self):012x}>"

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def tojson(self):
        return {
            "_typename": "TObjArray",
            "name": "TObjArray",
            "arr": [x.tojson() for x in self._data],
        }


uproot.classes["TObjArray"] = Model_TObjArray


class Model_TObjArrayOfTBaskets(Model_TObjArray):
    """
    A specialized :doc:`uproot.model.Model` for a ``TObjArray`` of ``TBaskets``.
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
            uproot.models.TObject.Model_TObject.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"], self._members["fLowerBound"] = cursor.fields(
            chunk, _tobjarray_format1, context
        )

        self._data = []
        for _ in range(self._members["fSize"]):
            item = uproot.deserialization.read_object_any(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                as_class=uproot.models.TBasket.Model_TBasket,
            )
            self._data.append(item)
