# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines versionless models of ``TRef`` and ``TRefArray``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import awkward
import numpy

import uproot

_tref_format1 = struct.Struct(">xxIxxxxxx")

# Note: https://github.com/root-project/root/blob/v6-26-08/io/doc/TFile/tref.md
#       https://github.com/root-project/root/blob/v6-26-08/io/doc/TFile/trefarray.md


class Model_TRef(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TRef``.

    This model does not deserialize all fields, only the reference number.
    """

    @property
    def ref(self):
        """
        The reference number as an integer.
        """
        return self._members["ref"]

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:  # pragma: no cover  (always False: no header read)
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._members["ref"] = cursor.field(chunk, _tref_format1, context)

    def __repr__(self):
        return f"<TRef {self.ref}>"

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = [(None, None)]
        members.append(("@pidf", numpy.dtype(">u2")))
        members.append(("ref", numpy.dtype(">u4")))
        members.append(("@other1", numpy.dtype(">u2")))
        members.append(("@other2", numpy.dtype(">u4")))

        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["tobject_header"]:
            contents["@pidf"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
            contents["ref"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@other1"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
            contents["@other2"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TRef"},
        )


_trefarray_format1 = struct.Struct(">i")
_trefarray_dtype = numpy.dtype(">i4")


class Model_TRefArray(uproot.model.Model, Sequence):
    """
    A versionless :doc:`uproot.model.Model` for ``TRefArray``.

    This also satisfies Python's abstract ``Sequence`` protocol.
    """

    @property
    def refs(self):
        """
        The reference number as a ``numpy.ndarray`` of ``dtype(">i4")``.
        """
        return self._data

    @property
    def nbytes(self):
        """
        The number of bytes in :ref:`uproot.models.TRef.Model_TRefArray.refs`.
        """
        return self._data.nbytes

    @property
    def name(self):
        """
        The name of this TRefArray.
        """
        return self._members["fName"]

    def read_members(self, chunk, cursor, context, file):
        forth_obj = uproot._awkwardforth.get_forth_obj(context)
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        if forth_obj is not None:
            key_number = uproot._awkwardforth.get_first_key_number(context)
            keys = [key_number + i for i in range(6)]
            forth_stash = uproot._awkwardforth.Node(
                f"node{keys[0]} TRef :prebuilt",
                form_details={
                    "class": "RecordArray",
                    "contents": {
                        "fName": {
                            "class": "ListOffsetArray",
                            "offsets": "i64",
                            "content": {
                                "class": "NumpyArray",
                                "primitive": "uint8",
                                "inner_shape": [],
                                "parameters": {"__array__": "char"},
                                "form_key": f"node{keys[2]}",
                            },
                            "parameters": {"__array__": "string"},
                            "form_key": f"node{keys[1]}",
                        },
                        "fSize": {
                            "class": "NumpyArray",
                            "primitive": "int32",
                            "inner_shape": [],
                            "parameters": {},
                            "form_key": f"node{keys[3]}",
                        },
                        "refs": {
                            "class": "ListOffsetArray",
                            "offsets": "i64",
                            "content": {
                                "class": "NumpyArray",
                                "primitive": "int32",
                                "inner_shape": [],
                                "parameters": {},
                                "form_key": f"node{keys[5]}",
                            },
                            "parameters": {},
                            "form_key": f"node{keys[4]}",
                        },
                    },
                    "parameters": {"__record__": "TRefArray"},
                    "form_key": f"node{keys[0]}",
                },
            )
            forth_stash.pre_code.append("10 stream skip\n")
            forth_stash.pre_code.append(
                f"stream !B-> stack dup 255 = if drop stream !I-> stack then dup node{keys[1]}-offsets +<- stack stream #!B-> node{keys[2]}-data\n"
            )
            forth_stash.pre_code.append(
                f"stream !I-> stack dup node{keys[3]}-data <- stack\n"
            )
            forth_stash.pre_code.append("6 stream skip\n")
            forth_stash.pre_code.append(
                f"dup node{keys[4]}-offsets +<- stack stream #!I-> node{keys[5]}-data\n"
            )
            forth_stash.header_code.append(
                f"output node{keys[1]}-offsets int64\noutput node{keys[2]}-data uint8\noutput node{keys[3]}-data int32\noutput node{keys[4]}-offsets int64\noutput node{keys[5]}-data int32\n"
            )
            forth_stash.init_code.append(
                f"0 node{keys[1]}-offsets <- stack\n0 node{keys[4]}-offsets <- stack\n"
            )
            forth_obj.add_node(forth_stash)
            forth_obj.set_active_node(forth_stash)
        cursor.skip(10)
        self._members["fName"] = cursor.string(chunk, context)
        self._members["fSize"] = cursor.field(chunk, _trefarray_format1, context)
        cursor.skip(6)
        self._data = cursor.array(
            chunk, self._members["fSize"], _trefarray_dtype, context
        )

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

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        contents["fName"] = uproot.containers.AsString(
            False, typename="TString"
        ).awkward_form(file, context)
        contents["fSize"] = uproot._util.awkward_form(numpy.dtype("i4"), file, context)
        contents["refs"] = awkward.forms.ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(numpy.dtype("i4"), file, context),
        )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TRefArray"},
        )


uproot.classes["TRef"] = Model_TRef
uproot.classes["TRefArray"] = Model_TRefArray
