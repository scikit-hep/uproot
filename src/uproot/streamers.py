# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines models for ``TStreamerInfo`` and its elements, as well as
routines for generating Python code for new classes from streamer data.
"""

from __future__ import annotations

import re
import struct
import sys

import numpy

import uproot
import uproot._awkwardforth as af

_canonical_typename_patterns = [
    (re.compile(r"\bChar_t\b"), "char"),
    (re.compile(r"\bUChar_t\b"), "unsigned char"),
    (re.compile(r"\bShort_t\b"), "short"),
    (re.compile(r"\bUShort_t\b"), "unsigned short"),
    (re.compile(r"\bInt_t\b"), "int"),
    (re.compile(r"\bUInt_t\b"), "unsigned int"),
    (re.compile(r"\bSeek_t\b"), "int"),  # file pointer
    (re.compile(r"\bLong_t\b"), "long"),
    (re.compile(r"\bULong_t\b"), "unsigned long"),
    (re.compile(r"\bFloat_t\b"), "float"),
    (
        re.compile(r"\bFloat16_t\b"),
        "Float16_t",
    ),  # 32-bit, written as 16, trunc mantissa
    (re.compile(r"\bDouble_t\b"), "double"),
    (re.compile(r"\bDouble32_t\b"), "Double32_t"),  # 64-bit, written as 32
    (re.compile(r"\bLongDouble_t\b"), "long double"),
    (re.compile(r"\bText_t\b"), "char"),
    (re.compile(r"\bBool_t\b"), "bool"),
    (re.compile(r"\bByte_t\b"), "unsigned char"),
    (re.compile(r"\bVersion_t\b"), "short"),  # class version id
    (re.compile(r"\bOption_t\b"), "const char"),  # option string
    (re.compile(r"\bSsiz_t\b"), "int"),  # string size
    (re.compile(r"\bReal_t\b"), "float"),  # TVector/TMatrix element
    (re.compile(r"\bLong64_t\b"), "long long"),  # portable int64
    (re.compile(r"\bULong64_t\b"), "unsigned long long"),  # portable uint64
    (re.compile(r"\bAxis_t\b"), "double"),  # axis values type
    (re.compile(r"\bStat_t\b"), "double"),  # statistics type
    (re.compile(r"\bFont_t\b"), "short"),  # font number
    (re.compile(r"\bStyle_t\b"), "short"),  # style number
    (re.compile(r"\bMarker_t\b"), "short"),  # marker number
    (re.compile(r"\bWidth_t\b"), "short"),  # line width
    (re.compile(r"\bColor_t\b"), "short"),  # color number
    (re.compile(r"\bSCoord_t\b"), "short"),  # screen coordinates
    (re.compile(r"\bCoord_t\b"), "double"),  # pad world coordinates
    (re.compile(r"\bAngle_t\b"), "float"),  # graphics angle
    (re.compile(r"\bSize_t\b"), "float"),  # attribute size
]


def _canonical_typename(name):
    for pattern, replacement in _canonical_typename_patterns:
        name = pattern.sub(replacement, name)
    return name


def _sanitize_string(s):
    return (
        s.encode("unicode_escape")
        .decode("ascii")
        .replace('"', '\\"')
        .replace("{", "{{")
        .replace("}", "}}")
    )


_ftype_to_dtype_map = {
    uproot.const.kBool: "numpy.dtype(numpy.bool_)",
    uproot.const.kChar: "numpy.dtype('i1')",
    uproot.const.kUChar: "numpy.dtype('u1')",
    uproot.const.kCharStar: "numpy.dtype('u1')",
    uproot.const.kShort: "numpy.dtype('>i2')",
    uproot.const.kUShort: "numpy.dtype('>u2')",
    uproot.const.kInt: "numpy.dtype('>i4')",
    uproot.const.kBits: "numpy.dtype('>u4')",
    uproot.const.kUInt: "numpy.dtype('>u4')",
    uproot.const.kCounter: "numpy.dtype('>u4')",
    uproot.const.kLong: "numpy.dtype('>i8')",
    uproot.const.kULong: "numpy.dtype('>u8')",
    uproot.const.kLong64: "numpy.dtype('>i8')",
    uproot.const.kULong64: "numpy.dtype('>u8')",
    uproot.const.kFloat: "numpy.dtype('>f4')",
    uproot.const.kFloat16: "numpy.dtype('>f4')",
    uproot.const.kDouble: "numpy.dtype('>f8')",
    uproot.const.kDouble32: "numpy.dtype('>f8')",
}


def _ftype_to_dtype(fType):
    return _ftype_to_dtype_map.get(fType)


_ftype_to_struct_map = {
    uproot.const.kBool: "?",
    uproot.const.kChar: "b",
    uproot.const.kUChar: "B",
    uproot.const.kCharStar: "B",
    uproot.const.kShort: "h",
    uproot.const.kUShort: "H",
    uproot.const.kInt: "i",
    uproot.const.kBits: "I",
    uproot.const.kUInt: "I",
    uproot.const.kCounter: "I",
    uproot.const.kLong: "q",
    uproot.const.kULong: "Q",
    uproot.const.kLong64: "q",
    uproot.const.kULong64: "Q",
    uproot.const.kFloat: "f",
    uproot.const.kFloat16: "f",
    uproot.const.kDouble: "d",
    uproot.const.kDouble32: "d",
}


def _ftype_to_struct(fType):
    try:
        return _ftype_to_struct_map[fType]
    except KeyError:
        raise NotImplementedError(fType) from None


def _copy_bytes(chunk, start, stop, cursor, context):
    return chunk.get(start, stop, cursor, context).tobytes()


_tstreamerinfo_format1 = struct.Struct(">Ii")


class Model_TStreamerInfo(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerInfo``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def __repr__(self):
        return f"<TStreamerInfo for {self.name} version {self.class_version} at 0x{id(self):012x}>"

    def show(self, stream=sys.stdout):
        """
        Args:
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display a ``TStreamerInfo``.

        For example,

        .. code-block::

            TLorentzVector (v4): TObject (v1)
                fP: TVector3 (TStreamerObject)
                fE: double (TStreamerBasicType)
        """
        bases = []
        for element in self.elements:
            if isinstance(element, Model_TStreamerBase):
                bases.append(f"{element.name} (v{element.base_version})")
        bases = "" if len(bases) == 0 else ": " + ", ".join(bases)
        stream.write(f"{self.name} (v{self.class_version}){bases}\n")
        for element in self.elements:
            element.show(stream=stream)

    @property
    def name(self):
        """
        The name (``fName``) of this ``TStreamerInfo``, passed through
        :doc:`uproot.model.classname_regularize`
        """
        return uproot.model.classname_regularize(self.member("fName"))

    @property
    def typename(self):
        """
        The typename/classname (``fName``) of this ``TStreamerInfo``.
        """
        return self.member("fName")

    @property
    def elements(self):
        """
        This ``TStreamerInfo``'s list of ``TStreamerElements``
        (:doc:`uproot.streamers.Model_TStreamerElement`).
        """
        return self._members["fElements"]

    @property
    def class_version(self):
        """
        The class version (``fClassVersion``) of this ``TStreamerInfo``.
        """
        return self._members["fClassVersion"]

    def class_code(self):
        """
        Returns Python code as a string that, when evaluated, would be a suitable
        :doc:`uproot.model.VersionedModel` for this class and version.
        """
        count_names = []
        for element in self.elements:  # (self is a TStreamerInfo)
            if element.has_member("fCountName"):
                count_names.append(element.member("fCountName"))
        read_members = [
            "    def read_members(self, chunk, cursor, context, file):",
            "        if self.is_memberwise:",
            "            raise NotImplementedError(",
            '                f"memberwise serialization of {type(self).__name__}\\nin file {self.file.file_path}"',
            "            )",
            "        forth_obj = af.get_forth_obj(context)",
            "        if forth_obj is not None:",
            "            key_number = af.get_first_key_number(context)",
            "            forth_stash = af.Node(f'read-members {key_number}')",
            "            key_number += 1",
            "            forth_obj.add_node(forth_stash)",
            "            forth_obj.set_active_node(forth_stash)",
        ]
        read_member_n = [
            "    def read_member_n(self, chunk, cursor, context, file, member_index):"
        ]
        strided_interpretation = [
            "    @classmethod",
            "    def strided_interpretation(cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None):",
            "        if cls in breadcrumbs:",
            "            raise CannotBeStrided('classes that can contain members of the same type cannot be strided because the depth of instances is unbounded')",
            "        breadcrumbs = breadcrumbs + (cls,)",
            "        members = [(None, None)]",
            "        if header:",
            "            members.append(('@num_bytes', numpy.dtype('>u4')))",
            "            members.append(('@instance_version', numpy.dtype('>u2')))",
        ]
        awkward_form = [
            "    @classmethod",
            "    def awkward_form(cls, file, context):",
            "        from awkward.forms import NumpyForm, ListOffsetForm, RegularForm, RecordForm",
            "        if cls in context['breadcrumbs']:",
            "            raise CannotBeAwkward('classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded')",
            "        context = context.copy()",
            "        context['breadcrumbs'] = context['breadcrumbs'] + (cls,)",
            "        contents = {}",
            "        if context['header']:",
            "            contents['@num_bytes'] = uproot._util.awkward_form(numpy.dtype('u4'), file, context)",
            "            contents['@instance_version'] = uproot._util.awkward_form(numpy.dtype('u2'), file, context)",
        ]
        fields = []
        formats = []
        dtypes = []
        formats_memberwise = []
        containers = []
        base_names_versions = []
        member_names = []
        class_flags = {}

        for i in range(len(self._members["fElements"])):
            self._members["fElements"][i].class_code(
                self,
                i,
                self._members["fElements"],
                read_members,
                read_member_n,
                strided_interpretation,
                awkward_form,
                fields,
                formats,
                dtypes,
                formats_memberwise,
                containers,
                base_names_versions,
                member_names,
                class_flags,
                count_names,
            )

        read_members.extend(
            [
                "        if forth_obj is not None:",
                f"           forth_stash.form_details = {{'class': 'RecordArray', 'parameters': {{'__record__': {self.name!r}}}}}",
            ]
        )

        if len(read_member_n) == 1:
            read_member_n.append("        pass")

        read_members.append("")
        read_member_n.append("")

        strided_interpretation.append(
            "        return uproot.interpretation.objects.AsStridedObjects(cls, members, original=original)"
        )
        strided_interpretation.append("")

        awkward_form.append(
            f"        return RecordForm(list(contents.values()), list(contents.keys()), parameters={{'__record__': {self.name!r} }})"
        )
        awkward_form.append("")

        class_data = []

        for i, format in enumerate(formats):
            joined_format = "".join(format)
            class_data.append(f"    _format{i} = struct.Struct('>{joined_format}')")

        for i, format in enumerate(formats_memberwise):
            class_data.append(
                f"    _format_memberwise{i} = struct.Struct('>{''.join(format)}')"
            )
        for i, dt in enumerate(dtypes):
            class_data.append(f"    _dtype{i} = {dt}")

        for i, stl in enumerate(containers):
            class_data.append(f"    _stl_container{i} = {stl}")

        class_data.append(
            f"    base_names_versions=[{', '.join(f'({name!r}, {version})' for name, version in base_names_versions)}]"
        )

        joined_member_names = ", ".join(repr(x) for x in member_names)
        class_data.append(f"    member_names = [{joined_member_names}]")

        class_data.append(
            f"    class_flags = {{{', '.join(repr(k) + ': ' + repr(v) for k, v in class_flags.items())}}}"
        )

        # std::pair cannot be strided
        if self.name.startswith("pair<"):
            strided_interpretation = strided_interpretation[:2]
            strided_interpretation.append("        raise CannotBeStrided('std::pair')")

        classname = uproot.model.classname_encode(self.name, self.class_version)
        return "\n".join(
            [
                f"class {classname}(uproot.model.VersionedModel):",
                *read_members,
                *read_member_n,
                *strided_interpretation,
                *awkward_form,
                *class_data,
            ]
        )

    def new_class(self, file):
        """
        Args:
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes as needed from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.

        Returns a new subclass of :doc:`uproot.model.VersionedModel` for this
        class and version.
        """
        class_code = self.class_code()
        classname = uproot.model.classname_encode(self.name, self.class_version)
        classes = uproot.model.maybe_custom_classes(classname, file.custom_classes)
        return uproot.deserialization.compile_class(
            file, classes, class_code, classname
        )

    @property
    def file_uuid(self):
        """
        The unique identifier (:doc:`uproot.reading.ReadOnlyFile`) of the file
        from which this ``TStreamerInfo`` was extracted.
        """
        return self._file.uuid

    def walk_members(self, streamers):
        """
        Args:
            streamers (list of :doc:`uproot.streamers.Model_TStreamerInfo`): The
                complete set of ``TStreamerInfos``, probably including this one.

        Generator that yields all ``TStreamerElements``
        (:doc:`uproot.streamers.Model_TStreamerElement`) for this class and its
        superclasses.

        The ``TStreamerBase`` elements (:doc:`uproot.streamers.Model_TStreamerBase`)
        are not yielded, but they are extracted from ``streamers`` to include
        their elements.
        """
        for element in self._members["fElements"]:
            if isinstance(element, Model_TStreamerBase):
                streamers_with_name = streamers[element.name]
                base_version = element.base_version
                if base_version == "max":
                    base = streamers_with_name[max(streamers_with_name)]
                else:
                    base = streamers_with_name[base_version]
                yield from base.walk_members(streamers)
            else:
                yield element

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

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
        self._bases[0]._members["fName"] = _canonical_typename(
            self._bases[0]._members["fName"]
        )

        self._members["fCheckSum"], self._members["fClassVersion"] = cursor.fields(
            chunk, _tstreamerinfo_format1, context
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

        self._members["fElements"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        out.append(self._serialization)
        uproot.serialization._serialize_object_any(
            out, self._members["fElements"], None
        )
        if header:
            out.insert(
                where,
                uproot.serialization.numbytes_version(
                    sum(len(x) for x in out[where:]), self._instance_version
                ),
            )

    def _dependencies(self, streamers, out):
        out.append((self.name, self.class_version))
        for element in self.elements:
            element._dependencies(streamers, out)


_tstreamerelement_format1 = struct.Struct(">iiii")
_tstreamerelement_format2 = struct.Struct(">i")
_tstreamerelement_format3 = struct.Struct(">ddd")
_tstreamerelement_dtype1 = numpy.dtype(">i4")


class Model_TStreamerElement(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerElement``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def show(self, stream=sys.stdout):
        """
        Args:
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display a ``TStreamerElement``.
        """
        stream.write(
            f"    {self.name}: {self.typename} ({uproot.model.classname_decode(type(self).__name__)[0]})\n"
        )

    @property
    def name(self):
        """
        The name (``fName``) of this ``TStreamerElement``.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        The title (``fTitle``) of this ``TStreamerElement``.
        """
        return self.member("fTitle")

    @property
    def typename(self):
        """
        The typename (``fTypeName``) of this ``TStreamerElement``.
        """
        return self.member("fTypeName")

    @property
    def array_length(self):
        """
        The array length (``fArrayLength``) of this ``TStreamerElement``.
        """
        return self.member("fArrayLength")

    @property
    def file_uuid(self):
        """
        The unique identifier (:doc:`uproot.reading.ReadOnlyFile`) of the file
        from which this ``TStreamerElement`` was extracted.
        """
        return self._file.uuid

    @property
    def fType(self):
        """
        The type code (``fType``) of this ``TStreamerElement``.
        """
        return self.member("fType")

    def read_members(self, chunk, cursor, context, file):
        # https://github.com/root-project/root/blob/master/core/meta/src/TStreamerElement.cxx#L505
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
        (
            self._members["fType"],
            self._members["fSize"],
            self._members["fArrayLength"],
            self._members["fArrayDim"],
        ) = cursor.fields(chunk, _tstreamerelement_format1, context)

        if self._instance_version == 1:
            n = cursor.field(chunk, _tstreamerelement_format2, context)
            self._members["fMaxIndex"] = cursor.array(
                chunk, n, _tstreamerelement_dtype1, context
            )
        else:
            self._members["fMaxIndex"] = cursor.array(
                chunk, 5, _tstreamerelement_dtype1, context
            )

        self._members["fTypeName"] = _canonical_typename(cursor.string(chunk, context))

        if self._members["fType"] == 11 and self._members["fTypeName"] in {
            "Bool_t",
            "bool",
        }:
            self._members["fType"] = 18

        if self._instance_version == 3:
            # only version 3 writes the range, which later versions take from the title
            (
                self._members["fXmin"],
                self._members["fXmax"],
                self._members["fFactor"],
            ) = cursor.fields(chunk, _tstreamerelement_format3, context)

        if self._instance_version <= 2:
            # FIXME
            # self._fSize = self._fArrayLength * gROOT->GetType(GetTypeName())->Size()
            pass

        if self._instance_version > 3:
            # FIXME
            # if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor)
            pass

    writable = True

    def _serialize(self, out, header, name, tobject_flags):
        where = len(out)
        out.append(self._serialization)
        if header:
            out.insert(
                where,
                uproot.serialization.numbytes_version(
                    sum(len(x) for x in out[where:]), self._instance_version
                ),
            )

    def _dependencies(self, streamers, out):
        pass


class Model_TStreamerArtificial(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerArtificial``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")

        message = f"not implemented: class members defined by {type(self).__name__} of type {self.typename} in member {self.name} of class {streamerinfo.name}"

        # untested as of PR #629
        read_members.extend(
            [
                f"        raise DeserializationError({message!r}, chunk, cursor, context, file.file_path)",
            ]
        )

        read_member_n.append(
            f"            raise DeserializationError({message!r}, chunk, cursor, context, file.file_path)"
        )

        strided_interpretation.append(f"        raise CannotBeStrided({message!r})")

        awkward_form.append(f"        raise CannotBeAwkward({message!r})")


_tstreamerbase_format1 = struct.Struct(">i")


class Model_TStreamerBase(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerBase``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def name(self):
        """
        The name (``fName``) of this ``TStreamerBase``, passed through
        :doc:`uproot.model.classname_regularize`.
        """
        return uproot.model.classname_regularize(self.member("fName"))

    @property
    def base_version(self):
        """
        The base version (``fBaseVersion``) of this ``TStreamerBase``.
        """
        if self._members["fBaseVersion"] == -1:
            return "max"
        else:
            return self._members["fBaseVersion"]

    def show(self, stream=sys.stdout):
        pass

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")

        # AwkwardForth testing B: test_0637's 01,02,08,09,11,12,13,15,16,29,38,45,46,49,50
        read_members.extend(
            [
                "        if forth_obj is not None:",
                "            key = key_number ; key_number += 1",
                "            nested_forth_stash = af.Node(f'base-class {key}')",
                "            forth_obj.add_node(nested_forth_stash)",
                "            forth_obj.push_active_node(nested_forth_stash)",
                f"        self._bases.append(c({self.name!r}, {self.base_version!r}).read(chunk, cursor, af.add_to_path(forth_obj, context, af.SpecialPathItem(f'base-class {{len(self._bases)}}')), file, self._file, self._parent, concrete=self.concrete))",
                "        if forth_obj is not None:",
                "            forth_obj.pop_active_node()",
            ]
        )

        read_member_n.append(
            f"            self._bases.append(c({self.name!r}, {self.base_version!r}).read(chunk, cursor, context, file, self._file, self._parent, concrete=self.concrete))"
        )

        strided_interpretation.append(
            f"        members.extend(file.class_named({self.name!r}, {self.base_version!r}).strided_interpretation(file, header, tobject_header, breadcrumbs).members)"
        )
        awkward_form.extend(
            [
                f"        tmp_awkward_form = file.class_named({self.name!r}, {self.base_version!r}).awkward_form(file, context)",
                "        contents.update(zip(tmp_awkward_form.fields, tmp_awkward_form.contents))",
            ]
        )

        base_names_versions.append((self.name, self.base_version))

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        if self._instance_version >= 2:
            self._members["fBaseVersion"] = cursor.field(
                chunk, _tstreamerbase_format1, context
            )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.name)
        if streamer_versions is not None:
            base_version = self.base_version
            streamer = None
            if len(streamer_versions) == 0:
                pass
            elif base_version == "max" or base_version not in streamer_versions:
                streamer = streamer_versions[max(streamer_versions)]
            else:
                streamer = streamer_versions[base_version]
            if (
                streamer is not None
                and (streamer.name, streamer.class_version) not in out
            ):
                streamer._dependencies(streamers, out)


_tstreamerbasicpointer_format1 = struct.Struct(">i")


class Model_TStreamerBasicPointer(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerBasicPointer``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def count_name(self):
        """
        The count name (``fCountName``) of this ``TStreamerBasicPointer``.
        """
        return self._members["fCountName"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")

        read_members.append(f"        tmp = self._dtype{len(dtypes)}")
        read_member_n.append(f"            tmp = self._dtype{len(dtypes)}")

        if streamerinfo.name == "TBranch" and self.name == "fBasketSeek":
            # untested as of PR #629
            read_members.extend(
                [
                    "        if forth_obj is not None:",
                    "            raise CannotBeForth()",
                    "        if context.get('speedbump', True):",
                    "            if cursor.bytes(chunk, 1, context)[0] == 2:",
                    "                tmp = numpy.dtype('>i8')",
                ]
            )
            read_member_n.extend(
                [
                    "            if context.get('speedbump', True):",
                    "                if cursor.bytes(chunk, 1, context)[0] == 2:",
                    "                    tmp = numpy.dtype('>i8')",
                ]
            )

        else:
            count_name = _sanitize_string(self.count_name)
            # AwkwardForth testing C: test_0637's 29,44,56
            read_members.extend(
                [
                    "        if context.get('speedbump', True):",
                    "            cursor.skip(1)",
                    "        if forth_obj is not None:",
                    "            key = key_number ; key_number += 1",
                    "            key2 = key_number ; key_number += 1",
                    f'            nested_forth_stash = af.Node(f"node{{key}}", field_name={self.name!r}, form_details={{"class": "ListOffsetArray", "offsets": "i64", "content": {{ "class": "NumpyArray", "primitive": f"{{af.struct_to_dtype_name[af.dtype_to_struct[self._dtype{len(dtypes)}]]}}", "inner_shape": [], "parameters": {{}}, "form_key": f"node{{key}}"}}, "form_key": f"node{{key2}}"}})',
                    "            if context.get('speedbump', True):",
                    "                nested_forth_stash.pre_code.append('1 stream skip \\n')",
                    f'            nested_forth_stash.header_code.append(f"output node{{key}}-data {{af.struct_to_dtype_name[af.dtype_to_struct[self._dtype{len(dtypes)}]]}}\\n")',
                    '            nested_forth_stash.header_code.append(f"output node{key2}-offsets int64\\n")',
                    '            nested_forth_stash.init_code.append(f"0 node{key2}-offsets <- stack\\n")',
                    f'            nested_forth_stash.pre_code.append(f" var_{count_name} @ dup node{{key2}}-offsets +<- stack \\n stream #!{{af.dtype_to_struct[self._dtype{len(dtypes)}]}}-> node{{key}}-data\\n")',
                    "            forth_obj.add_node(nested_forth_stash)",
                ]
            )

            read_member_n.extend(
                [
                    "            if context.get('speedbump', True):",
                    "                cursor.skip(1)",
                ]
            )

        read_members.append(
            f"        self._members[{self.name!r}] = cursor.array(chunk, self.member({self.count_name!r}), tmp, context);\n"
        )
        read_member_n.append(
            f"            self._members[{self.name!r}] = cursor.array(chunk, self.member({self.count_name!r}), tmp, context);\n"
        )

        strided_message = f"class members defined by {type(self).__name__} of type {self.typename} in member {self.name} of class {streamerinfo.name}"
        strided_interpretation.append(
            f"        raise CannotBeStrided({strided_message!r})"
        )

        awkward_form.append(
            f"        contents[{self.name!r}] = ListOffsetForm(context['index_format'], uproot._util.awkward_form(cls._dtype{len(dtypes)}, file, context))"
        )

        member_names.append(self.name)
        dtypes.append(_ftype_to_dtype(self.fType - uproot.const.kOffsetP))

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerbasicpointer_format1, context
        )
        self._members["fCountName"] = cursor.string(chunk, context)
        self._members["fCountClass"] = cursor.string(chunk, context)

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerBasicType(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerBasicType``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")
        if self.typename == "Double32_t":
            # untested as of PR #629
            read_members.extend(
                [
                    "        if forth_obj is not None:",
                    "            raise CannotBeForth()",
                    f"        self._members[{self.name!r}] = cursor.double32(chunk, context)",
                ]
            )
            read_member_n.append(
                f"            self._members[{self.name!r}] = cursor.double32(chunk, context)"
            )

        elif self.typename == "Float16_t":
            # untested as of PR #629
            read_members.extend(
                [
                    "        if forth_obj is not None:",
                    "            raise CannotBeForth()",
                    f"        self._members[{self.name!r}] = cursor.float16(chunk, 12, context)",
                ]
            )
            read_member_n.append(
                f"            self._members[{self.name!r}] = cursor.float16(chunk, 12, context)"
            )

        elif self.array_length == 0:
            if (
                i == 0
                or not isinstance(elements[i - 1], Model_TStreamerBasicType)
                or elements[i - 1].array_length != 0
                or elements[i - 1].typename in ("Double32_t", "Float16_t")
            ):
                fields.append([])
                formats.append([])

            fields[-1].append(self.name)
            formats[-1].append(_ftype_to_struct(self.fType))
            formats_memberwise.append(_ftype_to_struct(self.fType))
            if (
                i + 1 == len(elements)
                or not isinstance(elements[i + 1], Model_TStreamerBasicType)
                or elements[i + 1].array_length != 0
            ):
                if len(fields[-1]) == 1:
                    # AwkwardForth testing D: test_0637's 01,02,29,38,44,56
                    read_members.extend(
                        [
                            "        if forth_obj is not None:",
                            "            key = key_number ; key_number += 1",
                            '            form_key = f"node{key}-data"',
                            f'            nested_forth_stash = af.Node(f"node{{key}}", field_name={fields[-1][0]!r}, form_details={{ "class": "NumpyArray", "primitive": "{af.struct_to_dtype_name[formats[-1][0]]}", "inner_shape": [], "parameters": {{}}, "form_key": f"node{{key}}"}})',
                            f'            nested_forth_stash.header_code.append(f"output node{{key}}-data {af.struct_to_dtype_name[formats[-1][0]]}\\n")',
                            "            forth_obj.add_node(nested_forth_stash)",
                        ]
                    )
                    if fields[-1][0] in count_names:
                        read_members.extend(
                            [
                                f'            nested_forth_stash.init_code.append(f"variable var_{_sanitize_string(fields[-1][0])}\\n")',
                                f'            nested_forth_stash.pre_code.append(f"stream !{formats[-1][0]}-> stack dup var_{_sanitize_string(fields[-1][0])} ! node{{key}}-data <- stack\\n")',
                            ]
                        )
                    else:
                        read_members.append(
                            f'            nested_forth_stash.pre_code.append(f"stream !{formats[-1][0]}-> node{{key}}-data\\n")'
                        )
                    read_members.append(
                        f"        self._members[{fields[-1][0]!r}] = cursor.field(chunk, self._format{len(formats) - 1}, context)",
                    )

                else:
                    # AwkwardForth testing E: test_0637's 01,02,05,08,09,11,12,13,15,16,29,35,39,45,46,47,49,50,56
                    read_members.append("        if forth_obj is not None:")
                    for j in range(len(formats[-1])):
                        read_members.extend(
                            [
                                "           key = key_number ; key_number += 1",
                                '           form_key = f"node{key}-data"',
                                f'           nested_forth_stash = af.Node(f"node{{key}}", field_name={fields[-1][j]!r}, form_details={{ "class": "NumpyArray", "primitive": "{af.struct_to_dtype_name[formats[-1][j]]}", "inner_shape": [], "parameters": {{}}, "form_key": f"node{{key}}"}})',
                                f'           nested_forth_stash.header_code.append(f"output {{form_key}} {af.struct_to_dtype_name[formats[-1][j]]}\\n")',
                                f'           nested_forth_stash.pre_code.append(f"stream !{formats[-1][j]}-> {{form_key}}\\n")',
                                "           forth_obj.add_node(nested_forth_stash)",
                            ]
                        )

                    assign_members = ", ".join(
                        f"self._members[{x!r}]" for x in fields[-1]
                    )

                    read_members.append(
                        f"\n        {assign_members} = cursor.fields(chunk, self._format{len(formats) - 1}, context)"
                    )

            read_member_n.append(
                f"            self._members[{self.name!r}] = cursor.field(chunk, self._format_memberwise{len(formats_memberwise) - 1}, context)"
            )

        else:
            # AwkwardForth testing F: test_0637's 44,56
            read_members.extend(
                [
                    "        if forth_obj is not None:",
                    "            key = key_number ; key_number += 1",
                    "            key2 = key_number ; key_number += 1",
                    f'            nested_forth_stash = af.Node(f"node{{key}}", field_name={self.name!r}, form_details={{"class": "RegularArray", "size": {self.array_length}, "content": {{ "class": "NumpyArray", "primitive": f"{{af.struct_to_dtype_name[af.dtype_to_struct[self._dtype{len(dtypes)}]]}}", "inner_shape": [], "parameters": {{}}, "form_key": f"node{{key}}"}}, "form_key": f"node{{key2}}"}})',
                    f'            nested_forth_stash.header_code.append(f"output node{{key}}-data {{af.struct_to_dtype_name[af.dtype_to_struct[self._dtype{len(dtypes)}]]}}\\n")',
                    '            nested_forth_stash.header_code.append(f"output node{key2}-offsets int64\\n")',
                    '            nested_forth_stash.init_code.append(f"0 node{key2}-offsets <- stack\\n")',
                    f'            nested_forth_stash.pre_code.append(f"{self.array_length} dup node{{key2}}-offsets +<- stack \\n stream #!{{af.dtype_to_struct[self._dtype{len(dtypes)}]}}-> node{{key}}-data\\n")',
                    "            forth_obj.add_node(nested_forth_stash)",
                    f"        self._members[{self.name!r}] = cursor.array(chunk, {self.array_length}, self._dtype{len(dtypes)}, context)",
                ]
            )

            read_member_n.extend(
                [
                    f"            self._members[{self.name!r}] = cursor.array(chunk, {self.array_length}, self._dtype{len(dtypes)}, context)",
                ]
            )

            dtypes.append(_ftype_to_dtype(self.fType))

        if self.array_length == 0 and self.typename not in ("Double32_t", "Float16_t"):
            strided_interpretation.append(
                f"        members.append(({self.name!r}, {_ftype_to_dtype(self.fType)}))"
            )
        else:
            strided_message = f"class members defined by {type(self).__name__} of type {self.typename} in member {self.name} of class {streamerinfo.name}"
            strided_interpretation.append(
                f"        raise CannotBeStrided({strided_message!r})"
            )

        if self.array_length == 0:
            if self.typename == "Double32_t":
                awkward_form.append(
                    f"        contents[{self.name!r}] = NumpyForm('float64')"
                )

            elif self.typename == "Float16_t":
                awkward_form.append(
                    f"        contents[{self.name!r}] = NumpyForm('float32')"
                )

            else:
                awkward_form.append(
                    f"        contents[{self.name!r}] = uproot._util.awkward_form({_ftype_to_dtype(self.fType)}, file, context)"
                )

        else:
            awkward_form.append(
                f"        contents[{self.name!r}] = RegularForm(uproot._util.awkward_form({_ftype_to_dtype(self.fType)}, file, context), {self.array_length})"
            )

        member_names.append(self.name)

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        if (
            uproot.const.kOffsetL
            < self._bases[0]._members["fType"]
            < uproot.const.kOffsetP
        ):
            self._bases[0]._members["fType"] -= uproot.const.kOffsetL

        basic = True

        if self._bases[0]._members["fType"] in (
            uproot.const.kBool,
            uproot.const.kUChar,
            uproot.const.kChar,
        ):
            self._bases[0]._members["fSize"] = 1

        elif self._bases[0]._members["fType"] in (
            uproot.const.kUShort,
            uproot.const.kShort,
        ):
            self._bases[0]._members["fSize"] = 2

        elif self._bases[0]._members["fType"] in (
            uproot.const.kBits,
            uproot.const.kUInt,
            uproot.const.kInt,
            uproot.const.kCounter,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot.const.kULong,
            uproot.const.kLong,
        ):
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.int64).itemsize

        elif self._bases[0]._members["fType"] in (
            uproot.const.kULong64,
            uproot.const.kLong64,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] in (
            uproot.const.kFloat,
            uproot.const.kFloat16,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot.const.kDouble,
            uproot.const.kDouble32,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] == uproot.const.kCharStar:
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.intp).itemsize

        else:
            basic = False

        if basic and self._bases[0]._members["fArrayLength"] > 0:
            self._bases[0]._members["fSize"] *= self._bases[0]._members["fArrayLength"]

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


_tstreamerloop_format1 = struct.Struct(">i")


class Model_TStreamerLoop(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerLoop``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def count_name(self):
        """
        The count name (``fCountName``) of this ``TStreamerLoop``.
        """
        return self._members["fCountName"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        # untested as of PR #629
        read_members.extend(
            [
                "        if forth_obj is not None:",
                "            raise CannotBeForth()",
                "        cursor.skip(6)",
                f"        self._members[{self.name!r}] = numpy.empty(self.member({self.count_name!r}), dtype='O')",
                f"        for tmp in range(self.member({self.count_name!r})):",
                f"            self._members[{self.name!r}][tmp] = c({self.typename.rstrip('*')!r}).read(chunk, cursor, af.add_to_path(forth_obj, context, {self.name!r}), file, self._file, self.concrete)",
                "            if forth_obj is not None:",
                "                if len(forth_obj.active_node.children) != 0:",
                f"                    forth_obj.active_node.children[-1].field_name = {self.name!r}",
            ]
        )

        strided_message = f"class members defined by {type(self).__name__} of type {self.typename} in member {self.name} of class {streamerinfo.name}"
        strided_interpretation.append(
            f"        raise CannotBeStrided({strided_message!r})"
        )

        awkward_form.extend(
            [
                f"        tmp = file.class_named({self.typename.rstrip('*')!r}, 'max').awkward_form(file, context)",
                f"        contents[{self.name!r}] = ListOffsetForm(context['index_format'], tmp)",
            ]
        )

        member_names.append(self.name)

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerloop_format1, context
        )
        self._members["fCountName"] = cursor.string(chunk, context)
        self._members["fCountClass"] = cursor.string(chunk, context)

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)


_tstreamerstl_format1 = struct.Struct(">ii")


class Model_TStreamerSTL(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerSTL``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def stl_type(self):
        """
        The STL type code (``fSTLtype``) of this ``TStreamerSTL``.
        """
        return self._members["fSTLtype"]

    @property
    def fCtype(self):
        """
        The type code (``fCtype``) of this ``TStreamerSTL``.
        """
        return self._members["fCtype"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")

        stl_container = uproot.interpretation.identify.parse_typename(
            self.typename,
            quote=True,
            outer_header=True,
            inner_header=False,
            string_header=True,
        )

        # AwkwardForth testing G: test_0637's 35,38,39,44,45,47,50,56
        read_members.extend(
            [
                f"        self._members[{self.name!r}] = self._stl_container{len(containers)}.read(chunk, cursor, af.add_to_path(forth_obj, context, {self.name!r}), file, self._file, self.concrete)",
                "        if forth_obj is not None:",
                "            if len(forth_obj.active_node.children) != 0:",
                f"                forth_obj.active_node.children[-1].field_name = {self.name!r}",
            ]
        )

        read_member_n.append(
            f"            self._members[{self.name!r}] = self._stl_container{len(containers)}.read(chunk, cursor, context, file, self._file, self.concrete)"
        )

        strided_interpretation.append(
            f"        members.append(({self.name!r}, cls._stl_container{len(containers)}.strided_interpretation(file, header, tobject_header, breadcrumbs)))"
        )

        awkward_form.append(
            f"        contents[{self.name!r}] = cls._stl_container{len(containers)}.awkward_form(file, context)"
        )

        containers.append(stl_container)
        member_names.append(self.name)

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index
        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fSTLtype"], self._members["fCtype"] = cursor.fields(
            chunk, _tstreamerstl_format1, context
        )

        if self._members["fSTLtype"] in (
            uproot.const.kSTLmultimap,
            uproot.const.kSTLset,
        ):
            if self._bases[0]._members["fTypeName"].startswith(
                "std::set"
            ) or self._bases[0]._members["fTypeName"].startswith("set"):
                self._members["fSTLtype"] = uproot.const.kSTLset

            elif self._bases[0]._members["fTypeName"].startswith(
                "std::multimap"
            ) or self._bases[0]._members["fTypeName"].startswith("multimap"):
                self._members["fSTLtype"] = uproot.const.kSTLmultimap

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerSTLstring(Model_TStreamerSTL):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerSTLString``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerSTL.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class TStreamerPointerTypes:
    """
    A class to share code between
    :doc:`uproot.streamers.Model_TStreamerObjectAnyPointer` and
    :doc:`uproot.streamers.Model_TStreamerObjectPointer`.
    """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")

        if self.fType == uproot.const.kObjectp or self.fType == uproot.const.kAnyp:
            # AwkwardForth testing H: test_0637's (none! untested!)
            read_members.extend(
                [
                    "        if forth_obj is not None:",
                    "            raise CannotBeForth()",
                    f"        self._members[{self.name!r}] = c({self.typename.rstrip('*')!r}).read(chunk, cursor, af.add_to_path(forth_obj, context, {self.name!r}), file, self._file, self.concrete)",
                    "        if forth_obj is not None:",
                    "            if len(forth_obj.active_node.children) != 0:",
                    f"                forth_obj.active_node.children[-1].field_name = {self.name!r}",
                ]
            )

            read_member_n.append(
                f"            self._members[{self.name!r}] = c({self.typename.rstrip('*')!r}).read(chunk, cursor, context, file, self._file, self.concrete)"
            )

            strided_interpretation.append(
                f"        members.append(({self.name!r}, file.class_named({self.typename.rstrip('*')!r}, 'max').strided_interpretation(file, header, tobject_header, breadcrumbs)))"
            )

            awkward_form.append(
                f"        contents[{self.name!r}] = file.class_named({self.typename.rstrip('*')!r}, 'max').awkward_form(file, context)"
            )

        elif self.fType == uproot.const.kObjectP or self.fType == uproot.const.kAnyP:
            # AwkwardForth testing I: test_0637's (none! untested!)
            read_members.extend(
                [
                    "        if forth_obj is not None:",
                    "            raise CannotBeForth()",
                    f"        self._members[{self.name!r}] = read_object_any(chunk, cursor, af.add_to_path(forth_obj, context, {self.name!r}), file, self._file, self)",
                    "        if forth_obj is not None:",
                    "            if len(forth_obj.active_node.children) != 0:",
                    f"                forth_obj.active_node.children[-1].field_name = {self.name!r}",
                ]
            )
            read_member_n.append(
                f"            self._members[{self.name!r}] = read_object_any(chunk, cursor, context, file, self._file, self)"
            )
            strided_message = f"class members defined by {type(self).__name__} of type {self.typename} in member {self.name} of class {streamerinfo.name}"
            strided_interpretation.append(
                f"        raise CannotBeStrided({strided_message!r})"
            )
            class_flags["has_read_object_any"] = True

        else:
            message = f"not implemented: class members defined by {type(self).__name__} with fType {self.fType}"
            # untested as of PR #629
            read_members.append(
                f"        raise DeserializationError({message!r}, chunk, cursor, context, file.file_path)"
            )
            read_member_n.append(
                f"            raise DeserializationError({message!r}, chunk, cursor, context, file.file_path)"
            )

        member_names.append(self.name)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)


class Model_TStreamerObjectAnyPointer(TStreamerPointerTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObjectAnyPointer``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerObjectPointer(TStreamerPointerTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObjectPointer``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class TStreamerObjectTypes:
    """
    A class to share code between
    :doc:`uproot.streamers.Model_TStreamerObject`,
    :doc:`uproot.streamers.Model_TStreamerObjectAny`, and
    :doc:`uproot.streamers.Model_TStreamerString`.
    """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        containers,
        base_names_versions,
        member_names,
        class_flags,
        count_names,
    ):
        read_member_n.append(f"        if member_index == {i}:")

        # AwkwardForth testing J: test_0637's 01,02,29,45,46,49,50,56
        read_members.extend(
            [
                f"        self._members[{self.name!r}] = c({self.typename.rstrip('*')!r}).read(chunk, cursor, af.add_to_path(forth_obj, context, {self.name!r}), file, self._file, self.concrete)",
                "        if forth_obj is not None:",
                "            if len(forth_obj.active_node.children) != 0:",
                f"                forth_obj.active_node.children[-1].field_name = {self.name!r}",
            ]
        )

        read_member_n.append(
            f"            self._members[{self.name!r}] = c({self.typename.rstrip('*')!r}).read(chunk, cursor, context, file, self._file, self.concrete)"
        )

        strided_interpretation.append(
            f"        members.append(({self.name!r}, file.class_named({self.typename.rstrip('*')!r}, 'max').strided_interpretation(file, header, tobject_header, breadcrumbs)))"
        )
        awkward_form.append(
            f"        contents[{self.name!r}] = file.class_named({self.typename.rstrip('*')!r}, 'max').awkward_form(file, context)"
        )

        member_names.append(self.name)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)


class Model_TStreamerObject(TStreamerObjectTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObject``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerObjectAny(TStreamerObjectTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObjectAny``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerString(TStreamerObjectTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerString``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


uproot.classes["TStreamerInfo"] = Model_TStreamerInfo
uproot.classes["TStreamerElement"] = Model_TStreamerElement
uproot.classes["TStreamerArtificial"] = Model_TStreamerArtificial
uproot.classes["TStreamerBase"] = Model_TStreamerBase
uproot.classes["TStreamerBasicPointer"] = Model_TStreamerBasicPointer
uproot.classes["TStreamerBasicType"] = Model_TStreamerBasicType
uproot.classes["TStreamerLoop"] = Model_TStreamerLoop
uproot.classes["TStreamerObject"] = Model_TStreamerObject
uproot.classes["TStreamerObjectAny"] = Model_TStreamerObjectAny
uproot.classes["TStreamerObjectAnyPointer"] = Model_TStreamerObjectAnyPointer
uproot.classes["TStreamerObjectPointer"] = Model_TStreamerObjectPointer
uproot.classes["TStreamerSTL"] = Model_TStreamerSTL
uproot.classes["TStreamerSTLstring"] = Model_TStreamerSTLstring
uproot.classes["TStreamerString"] = Model_TStreamerString
