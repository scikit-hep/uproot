# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines versioned models for ``TAttLine``, ``TAttFill``, and ``TAttMarker``.
"""

from __future__ import annotations

import struct

import awkward
import numpy

import uproot

_tattline1_format1 = struct.Struct(">hhh")


class Model_TAttLine_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttLine`` version 1.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline1_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fLineColor", numpy.dtype(">i2")))
        members.append(("fLineStyle", numpy.dtype(">i2")))
        members.append(("fLineWidth", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fLineColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fLineStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fLineWidth"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAttLine"},
        )

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    class_code = None


class Model_TAttLine_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttLine`` version 2.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline1_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fLineColor", numpy.dtype(">i2")))
        members.append(("fLineStyle", numpy.dtype(">i2")))
        members.append(("fLineWidth", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fLineColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fLineStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fLineWidth"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAttLine"},
        )

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)
        out.append(
            _tattline1_format1.pack(
                self._members["fLineColor"],
                self._members["fLineStyle"],
                self._members["fLineWidth"],
            )
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 2
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    class_code = None


class Model_TAttLine(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttLine``.
    """

    known_versions = {1: Model_TAttLine_v1, 2: Model_TAttLine_v2}


_tattfill1_format1 = struct.Struct(">hh")
_tattfill2_format1 = struct.Struct(">hh")


class Model_TAttFill_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttFill`` version 1.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill1_format1, context
        )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fFillColor", numpy.dtype(">i2")))
        members.append(("fFillStyle", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fFillColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fFillStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAttFill"},
        )

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    class_code = None


class Model_TAttFill_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttFill`` version 2.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill2_format1, context
        )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fFillColor", numpy.dtype(">i2")))
        members.append(("fFillStyle", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fFillColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fFillStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAttFill"},
        )

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)
        out.append(
            _tattfill2_format1.pack(
                self._members["fFillColor"],
                self._members["fFillStyle"],
            )
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 2
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    class_code = None


class Model_TAttFill(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttFill``.
    """

    known_versions = {1: Model_TAttFill_v1, 2: Model_TAttFill_v2}


_tattmarker2_format1 = struct.Struct(">hhf")


class Model_TAttMarker_v2(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttMarker`` version 2.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        (
            self._members["fMarkerColor"],
            self._members["fMarkerStyle"],
            self._members["fMarkerSize"],
        ) = cursor.fields(chunk, _tattmarker2_format1, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fMarkerColor", numpy.dtype(">i2")))
        members.append(("fMarkerStyle", numpy.dtype(">i2")))
        members.append(("fMarkerSize", numpy.dtype(">f4")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fMarkerColor"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fMarkerStyle"] = uproot._util.awkward_form(
            numpy.dtype("i2"), file, context
        )
        contents["fMarkerSize"] = uproot._util.awkward_form(
            numpy.dtype("f4"), file, context
        )
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAttMarker"},
        )

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)
        out.append(
            _tattmarker2_format1.pack(
                self._members["fMarkerColor"],
                self._members["fMarkerStyle"],
                self._members["fMarkerSize"],
            )
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 2
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))

    base_names_versions = []
    member_names = ["fMarkerColor", "fMarkerStyle", "fMarkerSize"]
    class_flags = {}
    class_code = None


class Model_TAttMarker(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttMarker``.
    """

    known_versions = {2: Model_TAttMarker_v2}


class Model_TAttAxis_v4(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAttAxis`` version 4.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )
        (
            self._members["fNdivisions"],
            self._members["fAxisColor"],
            self._members["fLabelColor"],
            self._members["fLabelFont"],
            self._members["fLabelOffset"],
            self._members["fLabelSize"],
            self._members["fTickLength"],
            self._members["fTitleOffset"],
            self._members["fTitleSize"],
            self._members["fTitleColor"],
            self._members["fTitleFont"],
        ) = cursor.fields(chunk, self._format0, context)

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._members["fNdivisions"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 1:
            self._members["fAxisColor"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 2:
            self._members["fLabelColor"] = cursor.field(
                chunk, self._format_memberwise2, context
            )
        if member_index == 3:
            self._members["fLabelFont"] = cursor.field(
                chunk, self._format_memberwise3, context
            )
        if member_index == 4:
            self._members["fLabelOffset"] = cursor.field(
                chunk, self._format_memberwise4, context
            )
        if member_index == 5:
            self._members["fLabelSize"] = cursor.field(
                chunk, self._format_memberwise5, context
            )
        if member_index == 6:
            self._members["fTickLength"] = cursor.field(
                chunk, self._format_memberwise6, context
            )
        if member_index == 7:
            self._members["fTitleOffset"] = cursor.field(
                chunk, self._format_memberwise7, context
            )
        if member_index == 8:
            self._members["fTitleSize"] = cursor.field(
                chunk, self._format_memberwise8, context
            )
        if member_index == 9:
            self._members["fTitleColor"] = cursor.field(
                chunk, self._format_memberwise9, context
            )
        if member_index == 10:
            self._members["fTitleFont"] = cursor.field(
                chunk, self._format_memberwise10, context
            )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        members.append(("fNdivisions", numpy.dtype(">i4")))
        members.append(("fAxisColor", numpy.dtype(">i2")))
        members.append(("fLabelColor", numpy.dtype(">i2")))
        members.append(("fLabelFont", numpy.dtype(">i2")))
        members.append(("fLabelOffset", numpy.dtype(">f4")))
        members.append(("fLabelSize", numpy.dtype(">f4")))
        members.append(("fTickLength", numpy.dtype(">f4")))
        members.append(("fTitleOffset", numpy.dtype(">f4")))
        members.append(("fTitleSize", numpy.dtype(">f4")))
        members.append(("fTitleColor", numpy.dtype(">i2")))
        members.append(("fTitleFont", numpy.dtype(">i2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"), file, context
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        contents["fNdivisions"] = uproot._util.awkward_form(
            numpy.dtype(">i4"), file, context
        )
        contents["fAxisColor"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        contents["fLabelColor"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        contents["fLabelFont"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        contents["fLabelOffset"] = uproot._util.awkward_form(
            numpy.dtype(">f4"), file, context
        )
        contents["fLabelSize"] = uproot._util.awkward_form(
            numpy.dtype(">f4"), file, context
        )
        contents["fTickLength"] = uproot._util.awkward_form(
            numpy.dtype(">f4"), file, context
        )
        contents["fTitleOffset"] = uproot._util.awkward_form(
            numpy.dtype(">f4"), file, context
        )
        contents["fTitleSize"] = uproot._util.awkward_form(
            numpy.dtype(">f4"), file, context
        )
        contents["fTitleColor"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        contents["fTitleFont"] = uproot._util.awkward_form(
            numpy.dtype(">i2"), file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAttAxis"},
        )

    _format0 = struct.Struct(">ihhhfffffhh")
    _format_memberwise0 = struct.Struct(">i")
    _format_memberwise1 = struct.Struct(">h")
    _format_memberwise2 = struct.Struct(">h")
    _format_memberwise3 = struct.Struct(">h")
    _format_memberwise4 = struct.Struct(">f")
    _format_memberwise5 = struct.Struct(">f")
    _format_memberwise6 = struct.Struct(">f")
    _format_memberwise7 = struct.Struct(">f")
    _format_memberwise8 = struct.Struct(">f")
    _format_memberwise9 = struct.Struct(">h")
    _format_memberwise10 = struct.Struct(">h")
    base_names_versions = []
    member_names = [
        "fNdivisions",
        "fAxisColor",
        "fLabelColor",
        "fLabelFont",
        "fLabelOffset",
        "fLabelSize",
        "fTickLength",
        "fTitleOffset",
        "fTitleSize",
        "fTitleColor",
        "fTitleFont",
    ]
    class_flags = {}

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)
        out.append(
            self._format0.pack(
                self._members["fNdivisions"],
                self._members["fAxisColor"],
                self._members["fLabelColor"],
                self._members["fLabelFont"],
                self._members["fLabelOffset"],
                self._members["fLabelSize"],
                self._members["fTickLength"],
                self._members["fTitleOffset"],
                self._members["fTitleSize"],
                self._members["fTitleColor"],
                self._members["fTitleFont"],
            )
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TAttAxis(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAttAxis``.
    """

    known_versions = {4: Model_TAttAxis_v4}


class Model_TAtt3D_v1(uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TAtt3D`` version 1.
    """

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"""memberwise serialization of {type(self).__name__}
in file {self.file.file_path}"""
            )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        pass

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided(
                "classes that can contain members of the same type cannot be strided because the depth of instances is unbounded"
            )
        breadcrumbs = (*breadcrumbs, cls)
        members = [(None, None)]
        if header:
            members.append(("@num_bytes", numpy.dtype(">u4")))
            members.append(("@instance_version", numpy.dtype(">u2")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import RecordForm

        if cls in context["breadcrumbs"]:
            raise uproot.interpretation.objects.CannotBeAwkward(
                "classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded"
            )
        context = context.copy()
        context["breadcrumbs"] = context["breadcrumbs"] + (cls,)
        contents = {}
        if context["header"]:
            contents["@num_bytes"] = uproot._util.awkward_form(
                numpy.dtype("u4"),
                file,
                context,
            )
            contents["@instance_version"] = uproot._util.awkward_form(
                numpy.dtype("u2"), file, context
            )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TAtt3D"},
        )

    base_names_versions = []
    member_names = []
    class_flags = {}

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, None, tobject_flags)
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 1
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TAtt3D(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TAtt3D``.
    """

    known_versions = {1: Model_TAtt3D_v1}


uproot.classes["TAttLine"] = Model_TAttLine
uproot.classes["TAttFill"] = Model_TAttFill
uproot.classes["TAttMarker"] = Model_TAttMarker
uproot.classes["TAttAxis"] = Model_TAttAxis
uproot.classes["TAtt3D"] = Model_TAtt3D
