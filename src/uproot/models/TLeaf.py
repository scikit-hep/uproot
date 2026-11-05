# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines versioned models for ``TLeaf`` and its subclasses.
"""

from __future__ import annotations

import struct

import uproot

_tleaf2_format0 = struct.Struct(">iii??")

_rawstreamer_TLeaf_v2 = (
    None,
    b"@\x00\x05\x04\xff\xff\xff\xffTStreamerInfo\x00@\x00\x04\xee\x00\t@\x00\x00\x13\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x05TLeaf\x00m\x1e\x81R\x00\x00\x00\x02@\x00\x04\xc9\xff\xff\xff\xffTObjArray\x00@\x00\x04\xb7\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00\x94\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00y\x00\x02@\x00\x00s\x00\x04@\x00\x00E\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x04fLen3Number of fixed length elements in the leaf's data.\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x87\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00l\x00\x02@\x00\x00f\x00\x04@\x00\x008\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fLenType\"Number of bytes for this data type\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x89\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00n\x00\x02@\x00\x00h\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fOffset%Offset in ClonesArray object (if one)\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x01\x14\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00\xf9\x00\x02@\x00\x00\xf3\x00\x04@\x00\x00\xc4\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fIsRange\xae(=kTRUE if leaf has a range, kFALSE otherwise).  This is equivalent to being a 'leafcount'.  For a TLeafElement the range information is actually store in the TBranchElement.\x00\x00\x00\x12\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04bool@\x00\x00\x8f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00t\x00\x02@\x00\x00n\x00\x04@\x00\x00?\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x0bfIsUnsigned&(=kTRUE if unsigned, kFALSE otherwise)\x00\x00\x00\x12\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04bool@\x00\x00\xb2\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00\x93\x00\x02@\x00\x00\x8d\x00\x04@\x00\x00\\\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfLeafCountDPointer to Leaf count if variable length (we do not own the counter)\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TLeaf*\x00",
    "TLeaf",
    2,
)


class Model_TLeaf_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeaf`` version 2.
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
            file.class_named("TNamed", 1).read(
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
            self._members["fLen"],
            self._members["fLenType"],
            self._members["fOffset"],
            self._members["fIsRange"],
            self._members["fIsUnsigned"],
        ) = cursor.fields(chunk, _tleaf2_format0, context)
        self._members["fLeafCount"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    base_names_versions = [("TNamed", 1)]
    member_names = [
        "fLen",
        "fLenType",
        "fOffset",
        "fIsRange",
        "fIsUnsigned",
        "fLeafCount",
    ]
    class_flags = {"has_read_object_any": True}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
    )


class Model_TLeaf(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeaf``.
    """

    known_versions = {2: Model_TLeaf_v2}


_tleafO1_format1 = struct.Struct(">??")


class Model_TLeafO_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafO`` version 1
    (``numpy.bool_``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafO1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x14\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xfe\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafO\x00\x02\xaeH\xd3\x00\x00\x00\x01@\x00\x01\xd8\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc6\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8e\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x12\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04bool@\x00\x00\x8e\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x12\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04bool\x00",
            "TLeafO",
            1,
        ),
    )


class Model_TLeafO(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafO`` (``numpy.bool_``).
    """

    known_versions = {1: Model_TLeafO_v1}


_tleafb1_format1 = struct.Struct(">bb")


class Model_TLeafB_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafB`` version 1
    (``numpy.int8``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafb1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x14\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xfe\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafB\x00\x0f\x1eK^\x00\x00\x00\x01@\x00\x01\xd8\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc6\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8e\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04char@\x00\x00\x8e\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00s\x00\x02@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04char\x00",
            "TLeafB",
            1,
        ),
    )


class Model_TLeafB(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafB`` (``numpy.int8``).
    """

    known_versions = {1: Model_TLeafB_v1}


_tleafs1_format1 = struct.Struct(">hh")


class Model_TLeafS_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafS`` version 1
    (``numpy.int16``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafs1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x16\xff\xff\xff\xffTStreamerInfo\x00@\x00\x02\x00\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafS\x00\x15\x0c\xee\xcf\x00\x00\x00\x01@\x00\x01\xda\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc8\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00t\x00\x02@\x00\x00n\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short@\x00\x00\x8f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00t\x00\x02@\x00\x00n\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05short\x00",
            "TLeafS",
            1,
        ),
    )


class Model_TLeafS(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafS`` (``numpy.int16``).
    """

    known_versions = {1: Model_TLeafS_v1}


_tleafi1_format1 = struct.Struct(">ii")


class Model_TLeafI_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafI`` version 1
    (``numpy.int32``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafi1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x12\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xfc\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafI\x00~j\xae\x19\x00\x00\x00\x01@\x00\x01\xd6\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc4\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8d\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00r\x00\x02@\x00\x00l\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x8d\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00r\x00\x02@\x00\x00l\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int\x00",
            "TLeafI",
            1,
        ),
    )


class Model_TLeafI(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafI`` (``numpy.int32``).
    """

    known_versions = {1: Model_TLeafI_v1}


_tleafg1_format0 = struct.Struct(">qq")


class Model_TLeafG_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafG`` version 1
    (``numpy.int64``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafg1_format0, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None
    class_rawstreamers = None


class Model_TLeafG(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafG`` (``numpy.int64``).
    """

    known_versions = {1: Model_TLeafG_v1}


_tleafl1_format0 = struct.Struct(">qq")


class Model_TLeafL_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafL`` version 1
    (``numpy.int64``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafl1_format0, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x1c\xff\xff\xff\xffTStreamerInfo\x00@\x00\x02\x06\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafL\x00\xde2\x08b\x00\x00\x00\x01@\x00\x01\xe0\xff\xff\xff\xffTObjArray\x00@\x00\x01\xce\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x92\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00w\x00\x02@\x00\x00q\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t@\x00\x00\x92\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00w\x00\x02@\x00\x00q\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08Long64_t\x00",
            "TLeafL",
            1,
        ),
    )


class Model_TLeafL(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafL`` (``numpy.int64``).
    """

    known_versions = {1: Model_TLeafL_v1}


_tleaff1_format1 = struct.Struct(">ff")


class Model_TLeafF_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafF`` version 1
    (``numpy.float32``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleaff1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x16\xff\xff\xff\xffTStreamerInfo\x00@\x00\x02\x00\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafF\x00:\xdd\x9dr\x00\x00\x00\x01@\x00\x01\xda\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc8\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00t\x00\x02@\x00\x00n\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float@\x00\x00\x8f\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00t\x00\x02@\x00\x00n\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05float\x00",
            "TLeafF",
            1,
        ),
    )


class Model_TLeafF(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafF`` (``numpy.float32``).
    """

    known_versions = {1: Model_TLeafF_v1}


_tleafd1_format1 = struct.Struct(">dd")


class Model_TLeafD_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafD`` version 1
    (``numpy.float64``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafd1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x18\xff\xff\xff\xffTStreamerInfo\x00@\x00\x02\x02\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafD\x00\x11\x8e\x87v\x00\x00\x00\x01@\x00\x01\xdc\xff\xff\xff\xffTObjArray\x00@\x00\x01\xca\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x90\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00u\x00\x02@\x00\x00o\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x90\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00u\x00\x02@\x00\x00o\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double\x00",
            "TLeafD",
            1,
        ),
    )


class Model_TLeafD(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafD`` (``numpy.float64``).
    """

    known_versions = {1: Model_TLeafD_v1}


_tleafc1_format1 = struct.Struct(">ii")


class Model_TLeafC_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafC`` version 1
    (variable-length strings).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, _tleafc1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}
    class_code = None

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TLeaf_v2,
        (
            None,
            b"@\x00\x02\x12\xff\xff\xff\xffTStreamerInfo\x00@\x00\x01\xfc\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TLeafC\x00\xfb\xe3\xb2\xf3\x00\x00\x00\x01@\x00\x01\xd6\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc4\x00\x03\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00\x89\xff\xff\xff\xffTStreamerBase\x00@\x00\x00s\x00\x03@\x00\x00i\x00\x04@\x00\x00:\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x05TLeaf'Leaf: description of a Branch data type\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00m\x1e\x81R\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x8d\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00r\x00\x02@\x00\x00l\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum(Minimum value if leaf range is specified\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x8d\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00r\x00\x02@\x00\x00l\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum(Maximum value if leaf range is specified\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int\x00",
            "TLeafC",
            1,
        ),
    )


class Model_TLeafC(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafC`` (variable-length
    strings).
    """

    known_versions = {1: Model_TLeafC_v1}


class Model_TLeafF16_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafF16`` version 1
    (ROOT's ``Float16_t``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"] = cursor.float16(chunk, 12, context)
        self._members["fMaximum"] = cursor.float16(chunk, 12, context)

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}


class Model_TLeafF16(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafF16`` (ROOT's
    ``Float16_t``).
    """

    known_versions = {1: Model_TLeafF16_v1}


class Model_TLeafD32_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafD32`` version 1
    (ROOT's ``Double32_t``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fMinimum"] = cursor.double32(chunk, context)
        self._members["fMaximum"] = cursor.double32(chunk, context)

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fMinimum", "fMaximum"]
    class_flags = {}


class Model_TLeafD32(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafD32`` (ROOT's
    ``Double32_t``).
    """

    known_versions = {1: Model_TLeafD32_v1}


_tleafelement1_format1 = struct.Struct(">ii")


class Model_TLeafElement_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TLeafElement`` version 1
    (arbitrary objects, associated with ``TBranchElement``).
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
            file.class_named("TLeaf", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fID"], self._members["fType"] = cursor.fields(
            chunk, _tleafelement1_format1, context
        )

    base_names_versions = [("TLeaf", 2)]
    member_names = ["fID", "fType"]
    class_flags = {}
    class_code = None


class Model_TLeafElement(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TLeafElement``
    (arbitrary objects, associated with ``TBranchElement``).
    """

    known_versions = {1: Model_TLeafElement_v1}


uproot.classes["TLeaf"] = Model_TLeaf
uproot.classes["TLeafB"] = Model_TLeafB
uproot.classes["TLeafS"] = Model_TLeafS
uproot.classes["TLeafI"] = Model_TLeafI
uproot.classes["TLeafG"] = Model_TLeafG
uproot.classes["TLeafL"] = Model_TLeafL
uproot.classes["TLeafF"] = Model_TLeafF
uproot.classes["TLeafD"] = Model_TLeafD
uproot.classes["TLeafC"] = Model_TLeafC
uproot.classes["TLeafO"] = Model_TLeafO
uproot.classes["TLeafF16"] = Model_TLeafF16
uproot.classes["TLeafD32"] = Model_TLeafD32
uproot.classes["TLeafElement"] = Model_TLeafElement
