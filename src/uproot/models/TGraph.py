# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines versioned models for ``TLeaf`` and its subclasses.
"""

from __future__ import annotations

import struct

import numpy

import uproot
import uproot._util
import uproot.behaviors.TGraph
import uproot.behaviors.TGraphAsymmErrors
import uproot.behaviors.TGraphErrors
import uproot.deserialization
import uproot.model
import uproot.models.TH
import uproot.serialization

_rawstreamer_TGraph_v4 = (
    None,
    b'@\x00\x06s\xff\xff\xff\xffTStreamerInfo\x00@\x00\x06]\x00\t@\x00\x00\x14\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x06TGraph\x00\x05\xf7\xf4e\x00\x00\x00\x04@\x00\x067\xff\xff\xff\xffTObjArray\x00@\x00\x06%\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00@\x00\x00\x8d\xff\xff\xff\xffTStreamerBase\x00@\x00\x00w\x00\x03@\x00\x00m\x00\x04@\x00\x00>\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TNamed*The basis for a named object (name, title)\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xdf\xb7J<\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x01@\x00\x00t\xff\xff\xff\xffTStreamerBase\x00@\x00\x00^\x00\x03@\x00\x00T\x00\x04@\x00\x00%\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttLine\x0fLine attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x94\x07EI\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00y\xff\xff\xff\xffTStreamerBase\x00@\x00\x00c\x00\x03@\x00\x00Y\x00\x04@\x00\x00*\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08TAttFill\x14Fill area attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9*\x92\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00x\xff\xff\xff\xffTStreamerBase\x00@\x00\x00b\x00\x03@\x00\x00X\x00\x04@\x00\x00)\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nTAttMarker\x11Marker attributes\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00)\x1d\x8b\xec\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x02@\x00\x00\x81\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00f\x00\x02@\x00\x00`\x00\x04@\x00\x002\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fNpoints\x1cNumber of points <= fMaxSize\x00\x00\x00\x06\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03int@\x00\x00\x96\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00x\x00\x02@\x00\x00^\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x02fX\x1c[fNpoints] array of X points\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph@\x00\x00\x96\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00x\x00\x02@\x00\x00^\x00\x04@\x00\x00,\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x02fY\x1c[fNpoints] array of Y points\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph@\x00\x00\x9a\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00{\x00\x02@\x00\x00u\x00\x04@\x00\x00D\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfFunctions,Pointer to list of functions (fits and user)\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06TList*@\x00\x00\x97\xff\xff\xff\xffTStreamerObjectPointer\x00@\x00\x00x\x00\x02@\x00\x00r\x00\x04@\x00\x00B\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\nfHistogram*Pointer to histogram used for drawing axis\x00\x00\x00@\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05TH1F*@\x00\x00\x8a\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00o\x00\x02@\x00\x00i\x00\x04@\x00\x008\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMinimum"Minimum value for plotting along y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double@\x00\x00\x8a\xff\xff\xff\xffTStreamerBasicType\x00@\x00\x00o\x00\x02@\x00\x00i\x00\x04@\x00\x008\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x08fMaximum"Maximum value for plotting along y\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06double\x00',
    "TGraph",
    4,
)


class Model_TGraph_v4(uproot.behaviors.TGraph.TGraph, uproot.model.VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` for ``TGraph`` version 4.
    """

    behaviors = (uproot.behaviors.TGraph.TGraph,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"memberwise serialization of {type(self).__name__}\nin file {self.file.file_path}"
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
        self._bases.append(
            file.class_named("TAttLine", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttFill", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases.append(
            file.class_named("TAttMarker", 2).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fNpoints"] = cursor.field(chunk, self._format0, context)
        tmp = self._dtype0
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fX"] = cursor.array(chunk, self.member("fNpoints"), tmp, context)
        tmp = self._dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fY"] = cursor.array(chunk, self.member("fNpoints"), tmp, context)
        self._members["fFunctions"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self
        )
        self._members["fHistogram"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self
        )
        self._members["fMinimum"], self._members["fMaximum"] = cursor.fields(
            chunk, self._format1, context
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
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
        if member_index == 1:
            self._bases.append(
                file.class_named("TAttLine", 2).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 2:
            self._bases.append(
                file.class_named("TAttFill", 2).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 3:
            self._bases.append(
                file.class_named("TAttMarker", 2).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 4:
            self._members["fNpoints"] = cursor.field(
                chunk, self._format_memberwise0, context
            )
        if member_index == 5:
            tmp = self._dtype0
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fX"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
            )
        if member_index == 6:
            tmp = self._dtype1
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fY"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
            )
        if member_index == 7:
            self._members["fFunctions"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self
            )
        if member_index == 8:
            self._members["fHistogram"] = uproot.deserialization.read_object_any(
                chunk, cursor, context, file, self._file, self
            )
        if member_index == 9:
            self._members["fMinimum"] = cursor.field(
                chunk, self._format_memberwise1, context
            )
        if member_index == 10:
            self._members["fMaximum"] = cursor.field(
                chunk, self._format_memberwise2, context
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
        members.extend(
            file.class_named("TNamed", 1)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttLine", 2)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttFill", 2)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.extend(
            file.class_named("TAttMarker", 2)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        members.append(("fNpoints", numpy.dtype(">u4")))
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fX of class TGraph"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fY of class TGraph"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerObjectPointer of type TList* in member fFunctions of class TGraph"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerObjectPointer of type TH1F* in member fHistogram of class TGraph"
        )
        members.append(("fMinimum", numpy.dtype(">f8")))
        members.append(("fMaximum", numpy.dtype(">f8")))
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import ListOffsetForm, RecordForm

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
        tmp_awkward_form = file.class_named("TNamed", 1).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttLine", 2).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttFill", 2).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        tmp_awkward_form = file.class_named("TAttMarker", 2).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fNpoints"] = uproot._util.awkward_form(
            numpy.dtype(">u4"), file, context
        )
        contents["fX"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype0, file, context),
        )
        contents["fY"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype1, file, context),
        )
        contents["fMinimum"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        contents["fMaximum"] = uproot._util.awkward_form(
            numpy.dtype(">f8"), file, context
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TGraph"},
        )

    _format0 = struct.Struct(">I")
    _format1 = struct.Struct(">dd")
    _format_memberwise0 = struct.Struct(">I")
    _format_memberwise1 = struct.Struct(">d")
    _format_memberwise2 = struct.Struct(">d")
    _dtype0 = numpy.dtype(">f8")
    _dtype1 = numpy.dtype(">f8")
    base_names_versions = [
        ("TNamed", 1),
        ("TAttLine", 2),
        ("TAttFill", 2),
        ("TAttMarker", 2),
    ]
    member_names = [
        "fNpoints",
        "fX",
        "fY",
        "fFunctions",
        "fHistogram",
        "fMinimum",
        "fMaximum",
    ]
    class_flags = {"has_read_object_any": True}

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_THashList_v0,
        uproot.models.TH._rawstreamer_TAttAxis_v4,
        uproot.models.TH._rawstreamer_TAxis_v10,
        uproot.models.TH._rawstreamer_TH1_v8,
        uproot.models.TH._rawstreamer_TH1F_v3,
        uproot.models.TH._rawstreamer_TCollection_v3,
        uproot.models.TH._rawstreamer_TSeqCollection_v0,
        uproot.models.TH._rawstreamer_TList_v5,
        uproot.models.TH._rawstreamer_TAttMarker_v2,
        uproot.models.TH._rawstreamer_TAttFill_v2,
        uproot.models.TH._rawstreamer_TAttLine_v2,
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TGraph_v4,
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)
        out.extend(
            [
                struct.pack(">i", self._members["fNpoints"]),
                b"\x01",
                self._members["fX"].astype(">f8").tobytes(),
                b"\x01",
                self._members["fY"].astype(">f8").tobytes(),
                b"@\x00\x00\x1f\xff\xff\xff\xffTList\x00@\x00\x00\x11\x00\x05\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                struct.pack(">d", self._members["fMinimum"]),
                struct.pack(">d", self._members["fMaximum"]),
            ]
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 4
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TGraph(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TGraph``.
    """

    known_versions = {4: Model_TGraph_v4}


class Model_TGraphErrors_v3(
    uproot.behaviors.TGraphErrors.TGraphErrors, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TGraphErrors`` version 3.
    """

    behaviors = (uproot.behaviors.TGraphErrors.TGraphErrors,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"memberwise serialization of {type(self).__name__}\nin file {self.file.file_path}"
            )
        self._bases.append(
            file.class_named("TGraph", 4).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        tmp = self._dtype0
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fEX"] = cursor.array(
            chunk, self.member("fNpoints"), tmp, context
        )
        tmp = self._dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fEY"] = cursor.array(
            chunk, self.member("fNpoints"), tmp, context
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TGraph", 4).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            tmp = self._dtype0
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fEX"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
            )
        if member_index == 2:
            tmp = self._dtype1
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fEY"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
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
        members.extend(
            file.class_named("TGraph", 4)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fEX of class TGraphErrors"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fEY of class TGraphErrors"
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import ListOffsetForm, RecordForm

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
        tmp_awkward_form = file.class_named("TGraph", 4).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fEX"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype0, file, context),
        )
        contents["fEY"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype1, file, context),
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TGraphErrors"},
        )

    _dtype0 = numpy.dtype(">f8")
    _dtype1 = numpy.dtype(">f8")
    base_names_versions = [("TGraph", 4)]
    member_names = ["fEX", "fEY"]
    class_flags = {}

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_THashList_v0,
        uproot.models.TH._rawstreamer_TAttAxis_v4,
        uproot.models.TH._rawstreamer_TAxis_v10,
        uproot.models.TH._rawstreamer_TH1_v8,
        uproot.models.TH._rawstreamer_TH1F_v3,
        uproot.models.TH._rawstreamer_TCollection_v3,
        uproot.models.TH._rawstreamer_TSeqCollection_v0,
        uproot.models.TH._rawstreamer_TList_v5,
        uproot.models.TH._rawstreamer_TAttMarker_v2,
        uproot.models.TH._rawstreamer_TAttFill_v2,
        uproot.models.TH._rawstreamer_TAttLine_v2,
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TGraph_v4,
        (
            None,
            b"@\x00\x02\x1a\xff\xff\xff\xffTStreamerInfo\x00@\x00\x02\x04\x00\t@\x00\x00\x1a\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x0cTGraphErrors\x00*|\xe3\x0f\x00\x00\x00\x03@\x00\x01\xd8\xff\xff\xff\xffTObjArray\x00@\x00\x01\xc6\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00@\x00\x00w\xff\xff\xff\xffTStreamerBase\x00@\x00\x00a\x00\x03@\x00\x00W\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TGraph\x14Graph graphics class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\xf7\xf4e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x04@\x00\x00\x97\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00y\x00\x02@\x00\x00_\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03fEX\x1c[fNpoints] array of X errors\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph@\x00\x00\x97\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00y\x00\x02@\x00\x00_\x00\x04@\x00\x00-\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x03fEY\x1c[fNpoints] array of Y errors\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph\x00",
            "TGraphErrors",
            3,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)
        out.extend(
            [
                b"\x01",
                self._members["fEX"].astype(">f8").tobytes(),
                b"\x01",
                self._members["fEY"].astype(">f8").tobytes(),
            ]
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TGraphErrors(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TGraphErrors``.
    """

    known_versions = {3: Model_TGraphErrors_v3}


class Model_TGraphAsymmErrors_v3(
    uproot.behaviors.TGraphAsymmErrors.TGraphAsymmErrors, uproot.model.VersionedModel
):
    """
    A :doc:`uproot.model.VersionedModel` for ``TGraphAsymmErrors`` version 3.
    """

    behaviors = (uproot.behaviors.TGraphAsymmErrors.TGraphAsymmErrors,)

    def read_members(self, chunk, cursor, context, file):
        if uproot._awkwardforth.get_forth_obj(context) is not None:
            raise uproot.interpretation.objects.CannotBeForth()
        if self.is_memberwise:
            raise NotImplementedError(
                f"memberwise serialization of {type(self).__name__}\nin file {self.file.file_path}"
            )
        self._bases.append(
            file.class_named("TGraph", 4).read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        tmp = self._dtype0
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fEXlow"] = cursor.array(
            chunk, self.member("fNpoints"), tmp, context
        )
        tmp = self._dtype1
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fEXhigh"] = cursor.array(
            chunk, self.member("fNpoints"), tmp, context
        )
        tmp = self._dtype2
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fEYlow"] = cursor.array(
            chunk, self.member("fNpoints"), tmp, context
        )
        tmp = self._dtype3
        if context.get("speedbump", True):
            cursor.skip(1)
        self._members["fEYhigh"] = cursor.array(
            chunk, self.member("fNpoints"), tmp, context
        )

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._bases.append(
                file.class_named("TGraph", 4).read(
                    chunk,
                    cursor,
                    context,
                    file,
                    self._file,
                    self._parent,
                    concrete=self.concrete,
                )
            )
        if member_index == 1:
            tmp = self._dtype0
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fEXlow"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
            )
        if member_index == 2:
            tmp = self._dtype1
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fEXhigh"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
            )
        if member_index == 3:
            tmp = self._dtype2
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fEYlow"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
            )
        if member_index == 4:
            tmp = self._dtype3
            if context.get("speedbump", True):
                cursor.skip(1)
            self._members["fEYhigh"] = cursor.array(
                chunk, self.member("fNpoints"), tmp, context
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
        members.extend(
            file.class_named("TGraph", 4)
            .strided_interpretation(file, header, tobject_header, breadcrumbs)
            .members
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fEXlow of class TGraphAsymmErrors"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fEXhigh of class TGraphAsymmErrors"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fEYlow of class TGraphAsymmErrors"
        )
        raise uproot.interpretation.objects.CannotBeStrided(
            "class members defined by Model_TStreamerBasicPointer of type double* in member fEYhigh of class TGraphAsymmErrors"
        )
        return uproot.interpretation.objects.AsStridedObjects(
            cls, members, original=original
        )

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import ListOffsetForm, RecordForm

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
        tmp_awkward_form = file.class_named("TGraph", 4).awkward_form(file, context)
        contents.update(
            zip(tmp_awkward_form.fields, tmp_awkward_form.contents, strict=True)
        )
        contents["fEXlow"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype0, file, context),
        )
        contents["fEXhigh"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype1, file, context),
        )
        contents["fEYlow"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype2, file, context),
        )
        contents["fEYhigh"] = ListOffsetForm(
            context["index_format"],
            uproot._util.awkward_form(cls._dtype3, file, context),
        )
        return RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters={"__record__": "TGraphAsymmErrors"},
        )

    _dtype0 = numpy.dtype(">f8")
    _dtype1 = numpy.dtype(">f8")
    _dtype2 = numpy.dtype(">f8")
    _dtype3 = numpy.dtype(">f8")
    base_names_versions = [("TGraph", 4)]
    member_names = ["fEXlow", "fEXhigh", "fEYlow", "fEYhigh"]
    class_flags = {}

    class_rawstreamers = (
        uproot.models.TH._rawstreamer_THashList_v0,
        uproot.models.TH._rawstreamer_TAttAxis_v4,
        uproot.models.TH._rawstreamer_TAxis_v10,
        uproot.models.TH._rawstreamer_TH1_v8,
        uproot.models.TH._rawstreamer_TH1F_v3,
        uproot.models.TH._rawstreamer_TCollection_v3,
        uproot.models.TH._rawstreamer_TSeqCollection_v0,
        uproot.models.TH._rawstreamer_TList_v5,
        uproot.models.TH._rawstreamer_TAttMarker_v2,
        uproot.models.TH._rawstreamer_TAttFill_v2,
        uproot.models.TH._rawstreamer_TAttLine_v2,
        uproot.models.TH._rawstreamer_TString_v2,
        uproot.models.TH._rawstreamer_TObject_v1,
        uproot.models.TH._rawstreamer_TNamed_v1,
        _rawstreamer_TGraph_v4,
        (
            None,
            b"@\x00\x03u\xff\xff\xff\xffTStreamerInfo\x00@\x00\x03_\x00\t@\x00\x00\x1f\x00\x01\x00\x01\x00\x00\x00\x00\x03\x01\x00\x00\x11TGraphAsymmErrors\x00\xccF\xaf;\x00\x00\x00\x03@\x00\x03.\xff\xff\xff\xffTObjArray\x00@\x00\x03\x1c\x00\x03\x00\x01\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00@\x00\x00w\xff\xff\xff\xffTStreamerBase\x00@\x00\x00a\x00\x03@\x00\x00W\x00\x04@\x00\x00(\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06TGraph\x14Graph graphics class\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\xf7\xf4e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04BASE\x00\x00\x00\x04@\x00\x00\x9e\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x80\x00\x02@\x00\x00f\x00\x04@\x00\x004\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fEXlow [fNpoints] array of X low errors\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph@\x00\x00\xa0\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x82\x00\x02@\x00\x00h\x00\x04@\x00\x006\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fEXhigh![fNpoints] array of X high errors\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph@\x00\x00\x9e\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x80\x00\x02@\x00\x00f\x00\x04@\x00\x004\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x06fEYlow [fNpoints] array of Y low errors\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph@\x00\x00\xa0\xff\xff\xff\xffTStreamerBasicPointer\x00@\x00\x00\x82\x00\x02@\x00\x00h\x00\x04@\x00\x006\x00\x01\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x07fEYhigh![fNpoints] array of Y high errors\x00\x00\x000\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07double*\x00\x00\x00\x04\x08fNpoints\x06TGraph\x00",
            "TGraphAsymmErrors",
            3,
        ),
    )
    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        for x in self._bases:
            x._serialize(out, True, name, tobject_flags)
        out.extend(
            [
                b"\x01",
                self._members["fEXlow"].astype(">f8").tobytes(),
                b"\x01",
                self._members["fEXhigh"].astype(">f8").tobytes(),
                b"\x01",
                self._members["fEYlow"].astype(">f8").tobytes(),
                b"\x01",
                self._members["fEYhigh"].astype(">f8").tobytes(),
            ]
        )
        if header:
            num_bytes = sum(len(x) for x in out[where:])
            version = 3
            out.insert(where, uproot.serialization.numbytes_version(num_bytes, version))


class Model_TGraphAsymmErrors(uproot.model.DispatchByVersion):
    """
    A :doc:`uproot.model.DispatchByVersion` for ``TGraphAsymmErrors``.
    """

    known_versions = {3: Model_TGraphAsymmErrors_v3}


uproot.classes["TGraph"] = Model_TGraph
uproot.classes["TGraphErrors"] = Model_TGraphErrors
uproot.classes["TGraphAsymmErrors"] = Model_TGraphAsymmErrors
