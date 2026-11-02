# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module for writing RNTuple in the "cascading" file writer.

The implementation in this module does not use the RNTuple infrastructure in
:doc:`uproot.models.RNTuple`.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""

from __future__ import annotations

import datetime
import struct

import awkward
import numpy
import xxhash

import uproot
import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization
from uproot.models.RNTuple import (
    _rntuple_anchor_checksum_format,
    _rntuple_anchor_format,
    _rntuple_checksum_format,
    _rntuple_cluster_group_format,
    _rntuple_cluster_summary_format,
    _rntuple_column_compression_settings_format,
    _rntuple_column_element_offset_format,
    _rntuple_column_record_format,
    _rntuple_env_header_format,
    _rntuple_envlink_size_format,
    _rntuple_feature_flag_format,
    _rntuple_field_description_format,
    _rntuple_first_element_index_format,
    _rntuple_frame_num_items_format,
    _rntuple_frame_size_format,
    _rntuple_locator_offset_format,
    _rntuple_locator_size_format,
    _rntuple_page_num_elements_format,
    _rntuple_repetition_format,
)
from uproot.writing._cascade import CascadeLeaf, CascadeNode, Key, String

_rntuple_string_length_format = struct.Struct("<I")

_ak_primitive_to_typename_dict = {
    "bool": "bool",
    "char": "char",
    "int8": "std::int8_t",
    "uint8": "std::uint8_t",
    "int16": "std::int16_t",
    "uint16": "std::uint16_t",
    "int32": "std::int32_t",
    "uint32": "std::uint32_t",
    "int64": "std::int64_t",
    "uint64": "std::uint64_t",
    "float32": "float",
    "float64": "double",
    "i32": "std::int32_t",  # index type
    "i64": "std::int64_t",  # index type
}
_ak_primitive_to_num_dict = {
    "bool": 0x00,
    "char": 0x02,
    "int8": 0x03,
    "uint8": 0x04,
    "int16": 0x05,
    "uint16": 0x06,
    "int32": 0x07,
    "uint32": 0x08,
    "int64": 0x09,
    "uint64": 0x0A,
    "float32": 0x0C,
    "float64": 0x0D,
    "i32": 0x0E,
    "i64": 0x0F,
}


def _cpp_typename(akform, subcall=False):
    # All numpy arrays are flattened beforehand
    if isinstance(akform, awkward.forms.NumpyForm) and akform.inner_shape == ():
        ak_primitive = akform.primitive
        typename = _ak_primitive_to_typename_dict[ak_primitive]
    elif isinstance(akform, awkward.forms.ListOffsetForm):
        content_typename = _cpp_typename(akform.content, subcall=True)
        typename = f"std::vector<{content_typename}>"
        # Check if it contains strings and fix the type
        if akform.parameters.get("__array__", "") == "string":
            typename = "std::string"
    elif isinstance(akform, awkward.forms.RecordForm):
        if akform.is_tuple:
            field_typenames = [_cpp_typename(t, subcall=True) for t in akform.contents]
            typename = f"std::tuple<{','.join(field_typenames)}>"
        else:
            typename = "UntypedRecord"
    elif isinstance(akform, awkward.forms.RegularForm):
        content_typename = _cpp_typename(akform.content, subcall=True)
        typename = f"std::array<{content_typename},{akform.size}>"
    elif isinstance(akform, awkward.forms.IndexedOptionForm):
        content_typename = _cpp_typename(akform.content, subcall=True)
        typename = f"std::optional<{content_typename}>"
    elif isinstance(akform, awkward.forms.UnionForm):
        field_typenames = [_cpp_typename(t, subcall=True) for t in akform.contents]
        typename = f"std::variant<{','.join(field_typenames)}>"
    elif isinstance(akform, awkward.forms.UnmaskedForm):
        return _cpp_typename(akform.content, subcall=True)
    else:
        msg = "This should not have happened. Please report this issue to the Uproot developers."
        raise AssertionError(msg)
    if not subcall and "UntypedRecord" in typename:
        typename = ""  # empty types for anything that contains UntypedRecord
    return typename


def _form_contains_union(akform):
    """
    Recursively checks whether `akform` contains a UnionForm (variant field)
    anywhere in its structure.

    Used to detect and reject RNTuple variant fields early and clearly when
    reloading a preexisting RNTuple, rather than letting the reload path
    crash deep inside `_cpp_typename` with a generic "please report this"
    assertion. Reconstructing a variant field's on-disk form reintroduces a
    reserved "null" tag as an explicit ``EmptyForm``-backed union content --
    something a freshly constructed union never has, and that neither
    `_cpp_typename` nor `NTuple_Header._build_field_col_records` know how to
    handle.
    """
    if isinstance(akform, awkward.forms.UnionForm):
        return True
    if isinstance(akform, awkward.forms.RecordForm):
        return any(_form_contains_union(c) for c in akform.contents)
    content = getattr(akform, "content", None)
    if content is not None:
        return _form_contains_union(content)
    return False


class RBlob_Key(Key):
    def __init__(
        self,
        location,
        uncompressed_bytes,
        compressed_bytes,
        created_on=None,
        big=None,
    ):
        super().__init__(
            location,
            uncompressed_bytes,
            compressed_bytes,
            String(None, "RBlob"),
            String(None, ""),
            String(None, ""),
            0,
            0,
            location,
            created_on=created_on,
            big=big,
        )


def _serialize_string(content):
    content_bytes = str.encode(content, encoding="utf-8")
    return _rntuple_string_length_format.pack(len(content_bytes)) + content_bytes


def _compression_code(compression):
    # The file's compression is None when it is written uncompressed; encode that
    # as ZLIB level 0, like TTree's fCompress, since a bare 0 is not an algorithm.
    if compression is None:
        return uproot.compression.ZLIB(0).code
    return compression.code


def _record_frame_wrap(payload, includeself=True):
    aloc = len(payload)
    if includeself:
        aloc += _rntuple_frame_size_format.size
    raw_bytes = _rntuple_frame_size_format.pack(aloc) + payload
    return raw_bytes


def _serialize_rntuple_list_frame(items, wrap=True, rawinput=False, extra_payload=None):
    n_items = len(items)
    if wrap and rawinput:
        payload_bytes = b"".join([_record_frame_wrap(x) for x in items])
    elif rawinput:
        payload_bytes = b"".join(items)
    elif wrap:
        payload_bytes = b"".join([_record_frame_wrap(x.serialize()) for x in items])
    else:
        payload_bytes = b"".join([x.serialize() for x in items])
    if extra_payload is not None:
        payload_bytes += extra_payload
    size = (
        _rntuple_frame_size_format.size
        + _rntuple_frame_num_items_format.size
        + len(payload_bytes)
    )
    raw_bytes = _rntuple_frame_size_format.pack(-size)  # negative size means list
    raw_bytes += _rntuple_frame_num_items_format.pack(n_items)
    raw_bytes += payload_bytes
    return raw_bytes


def _serialize_envelope_header(type, length):
    assert type in uproot.const.RNTupleEnvelopeType
    assert 0 <= length < 1 << 48
    data = length
    data <<= 16
    data |= type
    return _rntuple_env_header_format.pack(data)


# https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md#field-description
class NTuple_Field_Description:
    def __init__(
        self,
        parent_field_id,
        struct_role,
        field_name,
        type_name,
        field_version=0,
        type_version=0,
        flags=uproot.const.RNTupleFieldFlags.NOFLAG,
        type_alias="",
        field_description="",
        repetition=None,
    ):
        self.field_version = field_version
        self.type_version = type_version
        self.parent_field_id = parent_field_id
        self.struct_role = struct_role
        self.flags = flags
        self.field_name = field_name
        self.type_name = type_name
        self.type_alias = type_alias
        self.field_description = field_description
        self.repetition = repetition

    def __repr__(self):
        return f"{type(self).__name__}({self.field_version!r}, {self.type_version!r}, {self.parent_field_id!r}, {self.struct_role!r}, {self.flags!r}, {self.field_name!r}, {self.type_name!r}, {self.type_alias!r}, {self.field_description!r})"

    def serialize(self):
        header_bytes = _rntuple_field_description_format.pack(
            self.field_version,
            self.type_version,
            self.parent_field_id,
            self.struct_role,
            self.flags,
        )
        string_bytes = b"".join(
            [
                _serialize_string(x)
                for x in (
                    self.field_name,
                    self.type_name,
                    self.type_alias,
                    self.field_description,
                )
            ]
        )
        additional_bytes = b""
        if self.flags & uproot.const.RNTupleFieldFlags.REPETITIVE:
            additional_bytes += _rntuple_repetition_format.pack(self.repetition)
        return b"".join([header_bytes, string_bytes, additional_bytes])


# https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md#column-description
class NTuple_Column_Description:
    def __init__(
        self, type_num, bits_on_disk, field_id, flags, repr_index, first_element_index=0
    ):
        self.type_num = type_num
        self.bits_on_disk = bits_on_disk
        self.field_id = field_id
        self.flags = flags
        self.repr_index = repr_index
        self.first_element_index = first_element_index

    def serialize(self):
        import uproot.const

        flags = self.flags
        if self.first_element_index > 0:
            flags = flags | uproot.const.RNTupleColumnFlags.DEFERRED
        header_bytes = _rntuple_column_record_format.pack(
            self.type_num,
            self.bits_on_disk,
            self.field_id,
            int(flags),
            self.repr_index,
        )
        if self.first_element_index > 0:
            header_bytes += _rntuple_first_element_index_format.pack(
                int(self.first_element_index)
            )
        return header_bytes


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#header-envelope
class NTuple_Header(CascadeLeaf):
    def __init__(self, location, name, ntuple_description, akform):
        self._name = name
        self._ntuple_description = ntuple_description
        self._akform = _to_packed_form(akform)

        self._serialize = None
        self._checksum = None
        self._field_records = []
        self._column_records = []
        self._column_keys = []
        self._ak_node_count = 0
        aloc = len(self.serialize())
        super().__init__(location, aloc)

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._akform]),
        )

    def _build_field_col_records(
        self, akform, field_name=None, parent_fid=None, add_field=True, description=""
    ):
        field_id = len(self._field_records)
        if parent_fid is None:
            parent_fid = field_id
        if field_name is None:
            field_name = f"_{field_id}"
        self._ak_node_count += 1
        if "__doc__" in akform.parameters:
            description = akform.parameters["__doc__"]
        if isinstance(akform, awkward.forms.NumpyForm):
            type_name = _cpp_typename(akform)
            field = NTuple_Field_Description(
                parent_fid,
                uproot.const.RNTupleFieldRole.LEAF,
                field_name,
                type_name,
                field_description=description,
            )
            if add_field:
                self._field_records.append(field)
            else:
                field_id = parent_fid
            ak_primitive = akform.parameters.get("__array__", akform.primitive)
            type_num = _ak_primitive_to_num_dict[ak_primitive]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)
            self._column_records.append(col)
            self._column_keys.append(f"node{self._ak_node_count}-data")
        elif isinstance(akform, awkward.forms.ListOffsetForm):
            type_name = _cpp_typename(akform)
            field_role = uproot.const.RNTupleFieldRole.COLLECTION
            if akform.parameters.get("__array__", "") == "string":
                type_name = "std::string"
                field_role = uproot.const.RNTupleFieldRole.LEAF
            field = NTuple_Field_Description(
                parent_fid,
                field_role,
                field_name,
                type_name,
                field_description=description,
            )
            self._field_records.append(field)
            ak_offset = akform.offsets
            type_num = _ak_primitive_to_num_dict[ak_offset]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)
            self._column_records.append(col)
            self._column_keys.append(f"node{self._ak_node_count}-offsets")
            # content data
            self._build_field_col_records(
                akform.content,
                parent_fid=field_id,
                add_field=field_role == uproot.const.RNTupleFieldRole.COLLECTION,
                field_name="_0",
                description=description,
            )
        elif isinstance(akform, awkward.forms.RecordForm):
            type_name = _cpp_typename(akform)
            field = NTuple_Field_Description(
                parent_fid,
                uproot.const.RNTupleFieldRole.RECORD,
                field_name,
                type_name,
                field_description=description,
            )
            self._field_records.append(field)
            for i, subakform in enumerate(akform.contents):
                subfield_name = f"_{i}" if akform.is_tuple else akform.fields[i]
                self._build_field_col_records(
                    subakform,
                    field_name=subfield_name,
                    parent_fid=field_id,
                )
        elif isinstance(akform, awkward.forms.RegularForm):
            type_name = _cpp_typename(akform)
            field_role = uproot.const.RNTupleFieldRole.LEAF
            field = NTuple_Field_Description(
                parent_fid,
                field_role,
                field_name,
                type_name,
                flags=uproot.const.RNTupleFieldFlags.REPETITIVE,
                repetition=akform.size,
                field_description=description,
            )
            self._field_records.append(field)
            self._build_field_col_records(
                akform.content,
                parent_fid=field_id,
                field_name="_0",
                description=description,
            )
        elif isinstance(akform, awkward.forms.IndexedOptionForm):
            type_name = _cpp_typename(akform)
            field = NTuple_Field_Description(
                parent_fid,
                uproot.const.RNTupleFieldRole.COLLECTION,
                field_name,
                type_name,
                field_description=description,
            )
            self._field_records.append(field)
            ak_index = akform.index
            type_num = _ak_primitive_to_num_dict[ak_index]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)
            self._column_records.append(col)
            self._column_keys.append(f"node{self._ak_node_count}-index")
            # content data
            self._build_field_col_records(
                akform.content,
                parent_fid=field_id,
                field_name="_0",
                description=description,
            )
        elif isinstance(akform, awkward.forms.UnionForm):
            type_name = _cpp_typename(akform)
            field = NTuple_Field_Description(
                parent_fid,
                uproot.const.RNTupleFieldRole.VARIANT,
                field_name,
                type_name,
                field_description=description,
            )
            self._field_records.append(field)
            type_num = uproot.const.rntuple_col_type_to_num_dict["switch"]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]
            col = NTuple_Column_Description(type_num, type_size, field_id, 0, 0)
            self._column_records.append(col)
            self._column_keys.append(f"node{self._ak_node_count}-switch")
            for i, subakform in enumerate(akform.contents):
                subfield_name = f"_{i}"
                self._build_field_col_records(
                    subakform,
                    field_name=subfield_name,
                    parent_fid=field_id,
                )
        elif isinstance(akform, awkward.forms.UnmaskedForm):
            # Do nothing
            self._build_field_col_records(
                akform.content,
                parent_fid=parent_fid,
                field_name=field_name,
                description=description,
            )
        else:
            msg = "This should not have happened. Please report this issue to the Uproot developers."
            raise AssertionError(msg)

    def generate_field_col_records(self):
        akform = self._akform
        for i, topakform in enumerate(akform.contents):
            field_name = f"_{i}" if akform.is_tuple else akform.fields[i]
            self._build_field_col_records(
                topakform,
                field_name=field_name,
            )

    def serialize(self):
        if self._serialize:
            return self._serialize

        feature_flag = _rntuple_feature_flag_format.pack(0)
        name = _serialize_string(self._name)
        description = _serialize_string(self._ntuple_description)
        writer = _serialize_string(f"Uproot {uproot.__version__}")

        out = []
        out.extend([feature_flag, name, description, writer])

        self.generate_field_col_records()
        alias_records = []
        extra_type_info = []

        out.append(_serialize_rntuple_list_frame(self._field_records))
        out.append(_serialize_rntuple_list_frame(self._column_records))
        out.append(_serialize_rntuple_list_frame(alias_records))
        out.append(_serialize_rntuple_list_frame(extra_type_info))
        payload = b"".join(out)

        env_header = _serialize_envelope_header(
            uproot.const.RNTupleEnvelopeType.HEADER,
            len(payload)
            + _rntuple_env_header_format.size
            + _rntuple_checksum_format.size,
        )
        header_and_payload = b"".join([env_header, payload])
        self._checksum = xxhash.xxh3_64_intdigest(header_and_payload)
        checksum_bytes = _rntuple_checksum_format.pack(self._checksum)

        final_bytes = b"".join([header_and_payload, checksum_bytes])
        self._serialize = final_bytes
        return self._serialize


# https://github.com/root-project/root/blob/8cd9eed6f3a32e55ef1f0f1df8e5462e753c735d/tree/ntuple/v7/doc/BinaryFormatSpecification.md#footer-envelope
class NTuple_Footer(CascadeLeaf):
    def __init__(self, location, header_checksum):
        self._header_checksum = header_checksum

        self.extension_field_record_frames = []
        self.extension_column_record_frames = []
        self.extension_alias_record_frames = []
        self.extension_extra_type_info = []

        self.cluster_group_record_frames = [
            #    NTuple_ClusterGroupRecord(0, NTuple_EnvLink(48, None))
        ]
        self._checksum = None

        super().__init__(location, None)

    def __repr__(self):
        return f"{type(self).__name__}(extension_field_record_frames={self.extension_field_record_frames}, extension_column_record_frames={self.extension_column_record_frames}, cluster_group_record_frames={self.cluster_group_record_frames})"

    def serialize(self):
        out = []
        out.extend(
            [
                _rntuple_feature_flag_format.pack(0),
                _rntuple_checksum_format.pack(self._header_checksum),
            ]
        )
        schema_extension_payload = b"".join(
            [
                _serialize_rntuple_list_frame(self.extension_field_record_frames),
                _serialize_rntuple_list_frame(self.extension_column_record_frames),
                _serialize_rntuple_list_frame(self.extension_alias_record_frames),
                _serialize_rntuple_list_frame(self.extension_extra_type_info),
            ]
        )
        out.append(_record_frame_wrap(schema_extension_payload))

        out.append(
            _serialize_rntuple_list_frame(
                [x.frame for x in self.cluster_group_record_frames],
                wrap=False,
                rawinput=True,
            )
        )
        payload = b"".join(out)

        env_header = _serialize_envelope_header(
            uproot.const.RNTupleEnvelopeType.FOOTER,
            len(payload)
            + _rntuple_env_header_format.size
            + _rntuple_checksum_format.size,
        )
        header_and_payload = b"".join([env_header, payload])
        self._checksum = xxhash.xxh3_64_intdigest(header_and_payload)
        checksum_bytes = _rntuple_checksum_format.pack(self._checksum)

        final_bytes = b"".join([header_and_payload, checksum_bytes])
        return final_bytes


class NTuple_Locator:
    def __init__(self, num_bytes, offset):
        assert num_bytes < (1 << 32)
        self.num_bytes = num_bytes
        self.offset = offset

    def serialize(self):
        outbytes = _rntuple_locator_size_format.pack(
            self.num_bytes
        ) + _rntuple_locator_offset_format.pack(self.offset)
        return outbytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_bytes}, {self.offset})"


class NTuple_EnvLink:
    def __init__(self, uncomp_size, locator):
        self.uncomp_size = uncomp_size
        self.locator = locator

    def serialize(self):
        out = [
            _rntuple_envlink_size_format.pack(self.uncomp_size),
            self.locator.serialize(),
        ]
        return b"".join(out)

    def __repr__(self):
        return f"{type(self).__name__}({self.uncomp_size}, {self.locator})"


class NTuple_ClusterGroupRecord:
    """
    Every extension rewrites the whole footer, which holds one of these per
    extension so far, so the record's bytes (already wrapped as a list-frame
    item) are computed once, here. The record is read-only so that the bytes
    cannot go stale; ``page_list_envlink`` is captured as it is at construction.
    """

    def __init__(self, min_entry, entry_span, num_clusters, page_list_envlink):
        self._min_entry = min_entry
        self._entry_span = entry_span
        self._num_clusters = num_clusters
        self._page_list_envlink = page_list_envlink
        self._frame = _record_frame_wrap(
            _rntuple_cluster_group_format.pack(min_entry, entry_span, num_clusters)
            + page_list_envlink.serialize()
        )

    @property
    def min_entry(self):
        return self._min_entry

    @property
    def entry_span(self):
        return self._entry_span

    @property
    def num_clusters(self):
        return self._num_clusters

    @property
    def page_list_envlink(self):
        return self._page_list_envlink

    @property
    def frame(self):
        return self._frame

    def serialize(self):
        return self._frame[_rntuple_frame_size_format.size :]

    def __repr__(self):
        return f"{type(self).__name__}({self.num_clusters}, {self.page_list_envlink})"


class NTuple_ClusterSummary:
    def __init__(self, num_first_entry, num_entries, flags=0):
        self.num_first_entry = num_first_entry
        self.num_entries = num_entries
        self.flags = flags

    def serialize(self):
        # Highest 8 bits are flags reserved for future use
        assert 0 <= self.num_first_entry < 2**56
        assert 0 <= self.flags < 2**8
        num_entries = (self.flags << 56) | self.num_entries
        payload_bytes = _rntuple_cluster_summary_format.pack(
            self.num_first_entry, num_entries
        )
        return payload_bytes

    def __repr__(self):
        return f"{type(self).__name__}({self.num_first_entry}, {self.num_entries}, {self.flags})"


class NTuple_PageDescription:
    def __init__(self, num_entries, locator):
        assert num_entries < (1 << 32)
        self.num_entries = num_entries
        self.locator = locator

    def serialize(self):
        out = [
            _rntuple_page_num_elements_format.pack(self.num_entries),
            self.locator.serialize(),
        ]
        return b"".join(out)

    def __repr__(self):
        return f"{type(self).__name__}({self.num_entries}, {self.locator})"


class NTuple_ColumnPageListDescription:
    def __init__(self, pagelist, element_offset, compression_settings):
        self.pagelist = pagelist
        self.element_offset = element_offset
        self.compression_settings = compression_settings

    def serialize(self):
        out = _serialize_rntuple_list_frame(
            self.pagelist,
            wrap=False,
            extra_payload=b"".join(
                [
                    _rntuple_column_element_offset_format.pack(self.element_offset),
                    _rntuple_column_compression_settings_format.pack(
                        self.compression_settings
                    ),
                ]
            ),
        )
        return out

    def __repr__(self):
        return f"{type(self).__name__}({self.pagelist}, {self.element_offset}, {self.compression_settings})"


class NTuple_PageListEnvelope:
    def __init__(self, header_checksum, cluster_summaries, cluster_list):
        self.header_checksum = header_checksum
        self.cluster_summaries = cluster_summaries
        self.cluster_list = cluster_list
        self._checksum = None
        assert len(cluster_summaries) == len(cluster_list)

    def serialize(self):
        nested_pagelist_rawbytes = _serialize_rntuple_list_frame(
            [
                _serialize_rntuple_list_frame(
                    cluster,
                    wrap=False,
                )
                for cluster in self.cluster_list
            ],
            rawinput=True,
            wrap=False,
        )
        out = [
            _rntuple_checksum_format.pack(self.header_checksum),
            _serialize_rntuple_list_frame(self.cluster_summaries),
            nested_pagelist_rawbytes,
        ]
        payload = b"".join(out)

        env_header = _serialize_envelope_header(
            uproot.const.RNTupleEnvelopeType.PAGELIST,
            len(payload)
            + _rntuple_env_header_format.size
            + _rntuple_checksum_format.size,
        )
        header_and_payload = b"".join([env_header, payload])
        self._checksum = xxhash.xxh3_64_intdigest(header_and_payload)
        checksum_bytes = _rntuple_checksum_format.pack(self._checksum)

        final_bytes = b"".join([header_and_payload, checksum_bytes])
        return final_bytes


class NTuple_Anchor(CascadeLeaf):
    def __init__(
        self,
        location,
        version_epoch,
        version_major,
        version_minor,
        version_patch,
        seek_header,
        nbytes_header,
        len_header,
        seek_footer,
        nbytes_footer,
        len_footer,
        max_key_size,
    ):
        aloc = (
            _rntuple_anchor_format.size
            + _rntuple_checksum_format.size
            + 6  # 6 bytes from header
        )
        super().__init__(location, aloc)
        self.version_epoch = version_epoch
        self.version_major = version_major
        self.version_minor = version_minor
        self.version_patch = version_patch
        self.seek_header = seek_header
        self.nbytes_header = nbytes_header
        self.len_header = len_header
        self.seek_footer = seek_footer
        self.nbytes_footer = nbytes_footer
        self.len_footer = len_footer
        self.max_key_size = max_key_size
        self.checksum = None

    @property
    def _fields(self):
        return [
            self.version_epoch,
            self.version_major,
            self.version_minor,
            self.version_patch,
            self.seek_header,
            self.nbytes_header,
            self.len_header,
            self.seek_footer,
            self.nbytes_footer,
            self.len_footer,
            self.max_key_size,
        ]

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self._location,
            ", ".join([repr(x) for x in self._fields]),
        )

    def serialize(self):
        out = _rntuple_anchor_format.pack(*self._fields)
        self.checksum = xxhash.xxh3_64_intdigest(out)
        checksum_bytes = _rntuple_anchor_checksum_format.pack(self.checksum)
        header = uproot.serialization.numbytes_version(
            _rntuple_anchor_format.size, 2
        )  # TODO: check
        out = b"".join([header, out, checksum_bytes])
        return out


class NTuple(CascadeNode):
    """
    Writes a RNTuple, including all fields, columns, and (upon ``extend``) pages.

    The ``write_anew`` method writes the whole ntuple, possibly for the first time, possibly
    because it has been moved (exceeded its initial allocation of TBasket pointers).

    The ``write_updates`` method rewrites the parts that change when new TBaskets are
    added.

    The ``extend`` method adds a page cluster to every column.

    See `ROOT RNTuple specification <https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md>`__.
    """

    def __init__(
        self,
        directory,
        ak_form,
        freesegments,
        header,
        footer,
        cluster_metadata,
        anchor,
    ):
        super().__init__(footer, anchor, freesegments)
        self._directory = directory
        self._header = header
        self._footer = footer
        self._cluster_metadata = cluster_metadata
        self._anchor = anchor
        self._freesegments = freesegments

        self._key = None
        self._header_key = None
        self._num_entries = 0

        self._column_counts = numpy.zeros(len(self._header._column_keys), dtype=int)

    def __repr__(self):
        return f"{type(self).__name__}({self._directory}, {self._header}, {self._footer}, {self._cluster_metadata}, {self._anchor}, {self._freesegments})"

    @property
    def directory(self):
        return self._directory

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._key.name

    @property
    def title(self):
        return self._key.title

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def location(self):
        return self._key.location

    @property
    def num_entries(self):
        return self._num_entries

    def extend(self, file, sink, data):
        """
        1. Write pages
        2. Write page list for new cluster group
        3. Relocate footer
        4. Update anchor's foot metadata values in-place
        5. Release old footer
        """

        data = _regularize_input_type_to_awkward(data)
        if not isinstance(data, awkward.Array):
            raise TypeError(
                "data must be an Awkward array, a dict, or a Pandas DataFrame"
            )

        data = _to_packed(data.layout)

        # A tuple RNTuple reads back with fields named "0", "1", ..., so a dict or record
        # with those names can extend it too
        akform = self._header._akform
        if (
            akform.is_tuple
            and isinstance(data, awkward.contents.RecordArray)
            and not data.is_tuple
            and sorted(data.fields) == sorted(akform.fields)
        ):
            data = awkward.contents.RecordArray(
                [data.content(field) for field in akform.fields],
                None,
                length=data.length,
                parameters=data.parameters,
            )

        if data.form != self._header._akform:
            msg = f"Data is not compatible with this RNTuple. Expected {self._header._akform}, got {data.form}"
            raise ValueError(msg)

        # awkward Form equality does not consider field order for a RecordArray,
        # only the set of fields and their types, so the check above can pass
        # even when data's fields are physically ordered differently than
        # self._header._akform's -- at any level of nesting, not just the top.
        # The column buffers below are looked up by positional node-index
        # (self._header._column_keys, built by walking self._header._akform in
        # its field order) -- if data isn't reordered to match, a column's
        # bytes silently get written under a different column's key,
        # corrupting every field from the first mismatch onward.
        data = _reorder_fields_to_match_form(data, self._header._akform)

        # 1. Write pages
        # We write a single page for each column for now

        cluster_page_data = []  # list of list of (locator, len, offset)
        data_buffers = awkward.to_buffers(data)[2]

        # We need to modify make a few modifications since not everything directly translates to RNTuples
        for key in list(data_buffers.keys()):
            barekey = key.split("-")[0]
            if "offset" in key:
                # RNTuples don't store the first offset
                data_buffers[key] = data_buffers[key][1:]
            elif "index" in key and barekey + "-tags" in data_buffers:
                # We group indices and tags into a single array
                dtype = numpy.dtype([("index", "int64"), ("tag", "int32")])
                indices = data_buffers[barekey + "-index"]
                tags = data_buffers[barekey + "-tags"]
                switches = numpy.zeros(len(indices), dtype=dtype)
                switches["index"] = indices
                switches["tag"] = tags + 1
                data_buffers[barekey + "-switch"] = switches
            elif "index" in key:
                # At this point, non-negative indices are guaranteed to be sorted
                # so we can easily convert to a ListOffsetArray
                index = data_buffers[key]
                deltas = numpy.array(index >= 0, dtype=index.dtype)
                data_buffers[key] = numpy.cumsum(deltas, dtype=deltas.dtype)

        # TODO: need better logic to specify per-column/field compression
        compression = self._directory.freesegments.fileheader.compression
        compression_code = _compression_code(compression)

        for idx, key in enumerate(self._header._column_keys):
            col_data = data_buffers[key]
            col_len = len(col_data.reshape(-1))
            raw_data = col_data.reshape(-1).view("uint8")
            if col_data.dtype == numpy.dtype("bool"):
                raw_data = numpy.packbits(raw_data, bitorder="little")
            uncompressed_bytes = len(raw_data)
            raw_data = uproot.compression.compress(raw_data, compression)
            # TODO: need to add some logic for page splitting
            pages = []
            if col_len > 0:
                page_key = self.add_rblob(sink, raw_data, uncompressed_bytes)
                page_locator = NTuple_Locator(
                    len(raw_data), page_key.location + page_key.allocation
                )
                pages.append(NTuple_PageDescription(col_len, page_locator))
            cluster_page_data.append(
                NTuple_ColumnPageListDescription(
                    pages, self._column_counts[idx], compression_code
                )
            )
            self._column_counts[idx] += col_len
        page_data = [
            cluster_page_data
        ]  # list of list of list of (locator, len, offset)

        # 2. Write page list envelope for new cluster group

        # only a single cluster for now
        cluster_summaries = [NTuple_ClusterSummary(self._num_entries, len(data))]
        self._num_entries += len(data)

        pagelistenv = NTuple_PageListEnvelope(
            self._header._checksum,
            cluster_summaries,
            page_data,
        )
        pagelistenv_rawdata = pagelistenv.serialize()
        pagelistenv_key = self.add_rblob(
            sink, pagelistenv_rawdata, len(pagelistenv_rawdata)
        )
        pagelistenv_locator = NTuple_Locator(
            len(pagelistenv_rawdata),
            pagelistenv_key.location + pagelistenv_key.allocation,
        )
        pagelistenv_envlink = NTuple_EnvLink(
            len(pagelistenv_rawdata), pagelistenv_locator
        )

        cluster_group = NTuple_ClusterGroupRecord(
            self._num_entries - len(data), len(data), 1, pagelistenv_envlink
        )

        self._footer.cluster_group_record_frames.append(cluster_group)

        # 3. Relocate footer
        # The old footer stays allocated until the anchor no longer points to
        # it, so that nothing written before then can land on top of it.

        old_footer_key = self._footer_key
        footer_raw_data = self._footer.serialize()
        self._footer_key = self.add_rblob(sink, footer_raw_data, len(footer_raw_data))
        self.sync(sink)

        # 4. Update anchor's foot metadata values in-place

        self._anchor.seek_footer = (
            self._footer_key.location + self._footer_key.allocation
        )
        self._anchor.nbytes_footer = len(footer_raw_data)
        self._anchor.len_footer = self._anchor.nbytes_footer

        anchor_raw_data = self._anchor.serialize()
        sink.write(self._anchor.location, anchor_raw_data)

        # 5. Release old footer

        self._freesegments.release(
            old_footer_key.location,
            old_footer_key.location
            + old_footer_key.num_bytes
            + old_footer_key.compressed_bytes,
        )
        self.sync(sink)

    def sync(self, sink):
        """
        Pushes everything written so far to storage, along with the FreeSegments
        record and the file header, which ``add_rblob`` may have made stale by
        allocating over the FreeSegments record's old location.

        ``add_rblob`` does none of this, because an extension adds one blob per
        column plus the page list and the footer, and doing it per blob is
        expensive for remote sinks. Instead, callers sync once before
        overwriting the anchor in place (so that it never points to data that
        has not reached storage) and once when done.
        """
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

    def add_rblob(
        self,
        sink,
        raw_data,
        uncompressed_bytes,
    ):
        # The strings included with the RBlob ("RBlob", "", "")
        strings_size = 8

        # Always use big files
        requested_bytes = (
            uproot.reading._key_format_big.size + strings_size + len(raw_data)
        )
        location = self._freesegments.allocate(requested_bytes, dry_run=False)

        key = RBlob_Key(
            location,
            uncompressed_bytes,
            len(raw_data),
            created_on=datetime.datetime.now(),
            big=True,
        )

        key.write(sink)
        sink.write(location + key.num_bytes, raw_data)
        # no set_file_length or flush here: see sync
        return key

    def write(self, sink):
        #### Header ##############################
        header_raw_data = self._header.serialize()
        self._header_key = self.add_rblob(sink, header_raw_data, len(header_raw_data))
        self._anchor.seek_header = (
            self._header_key.location + self._header_key.allocation
        )
        self._anchor.nbytes_header = len(header_raw_data)
        self._anchor.len_header = len(header_raw_data)
        #### Header end ##############################

        #### Footer ##############################
        footer_raw_data = self._footer.serialize()
        self._footer_key = self.add_rblob(sink, footer_raw_data, len(footer_raw_data))
        self._anchor.seek_footer = (
            self._footer_key.location + self._footer_key.allocation
        )
        self._anchor.nbytes_footer = len(footer_raw_data)
        self._anchor.len_footer = len(footer_raw_data)
        #### Footer end ##############################

        #### Anchor ##############################
        # add_object syncs after writing the anchor and the directory, which
        # point to the header and footer, so those must be synced first
        self.sync(sink)
        anchor_raw_data = self._anchor.serialize()
        self._key = self._directory.add_object(
            sink,
            "ROOT::RNTuple",
            self._header._name,
            self._header._name,
            anchor_raw_data,
            len(anchor_raw_data),
            replaces=self._key,
            big=False,
        )
        self._anchor.location = self._key.location + self._key.allocation
        #### Anchor end ##############################

        self.sync(sink)


def _to_packed_form(form):
    match form.__class__:
        case awkward.forms.BitMaskedForm | awkward.forms.ByteMaskedForm:
            return awkward.forms.IndexedOptionForm(
                "i64", _to_packed_form(form.content), parameters=form.parameters
            )
        case awkward.forms.EmptyForm:
            return form
        case awkward.forms.IndexedForm:
            return _to_packed_form(form.content)
        case awkward.forms.IndexedOptionForm:
            return awkward.forms.IndexedOptionForm(
                form.index, _to_packed_form(form.content), parameters=form.parameters
            )
        case awkward.forms.ListForm | awkward.forms.ListOffsetForm:
            return awkward.forms.ListOffsetForm(
                "i64", _to_packed_form(form.content), parameters=form.parameters
            )
        case awkward.forms.NumpyForm:
            shape = form.inner_shape
            out = awkward.forms.NumpyForm(form.primitive, parameters=form.parameters)
            for i in range(len(shape) - 1, -1, -1):
                out = awkward.forms.RegularForm(out, shape[i])
            return out
        case awkward.forms.RecordForm:
            return awkward.forms.RecordForm(
                [_to_packed_form(x) for x in form.contents],
                form._fields,
                parameters=form.parameters,
            )
        case awkward.forms.RegularForm:
            return awkward.forms.RegularForm(
                _to_packed_form(form.content), form._size, parameters=form.parameters
            )
        case awkward.forms.UnionForm:
            return awkward.forms.UnionForm(
                "i8",
                form.index,
                [_to_packed_form(x) for x in form.contents],
                parameters=form.parameters,
            )
        case awkward.forms.UnmaskedForm:
            return awkward.forms.UnmaskedForm(
                _to_packed_form(form._content), parameters=form.parameters
            )
        case _:
            msg = f"Form type {type(form)} cannot be written. If you believe this should be supported, please let the Uproot developers know."
            raise NotImplementedError(msg)


def _to_packed(layout):
    """
    This is similar to `to_packed` in Awkward, but a bit more consistent.
    """
    match layout.__class__:
        case awkward.contents.BitMaskedArray:
            next = layout.to_IndexedOptionArray64()
            content = _to_packed(next._content[: layout.length])
            return awkward.contents.IndexedOptionArray(
                next._index, content, parameters=next.parameters
            )
        case awkward.contents.ByteMaskedArray:
            next = layout.to_IndexedOptionArray64()
            content = _to_packed(next._content[: layout._mask.length])
            return awkward.contents.IndexedOptionArray(
                next._index, content, parameters=next.parameters
            )
        case awkward.contents.EmptyArray:
            return layout
        case awkward.contents.IndexedArray:
            projected = layout.project()
            return _to_packed(projected)
        case awkward.contents.IndexedOptionArray:
            nplike = layout._backend.nplike
            original_index = layout._index.data
            is_none = original_index < 0
            num_none = nplike.index_as_shape_item(nplike.count_nonzero(is_none))
            new_index = nplike.empty(layout._index.length, dtype=layout._index.dtype)
            new_index[is_none] = -1
            new_index[~is_none] = nplike.arange(
                nplike.shape_item_as_index(new_index.size - num_none),
                dtype=layout._index.dtype,
            )
            projected = layout.project()
            return awkward.contents.IndexedOptionArray(
                awkward.index.Index(new_index, nplike=layout._backend.nplike),
                _to_packed(projected),
                parameters=layout._parameters,
            )
        case awkward.contents.ListArray:
            next = layout.to_ListOffsetArray64(True)
            return _to_packed(next)
        case awkward.contents.ListOffsetArray:
            next = layout.to_ListOffsetArray64(True)
            next_content = next._content[: next._offsets[-1]]
            return awkward.contents.ListOffsetArray(
                next._offsets,
                _to_packed(next_content),
                parameters=next._parameters,
            )
        case awkward.contents.NumpyArray:
            return layout.to_contiguous().to_RegularArray()
        case awkward.contents.RecordArray:
            return awkward.contents.RecordArray(
                [_to_packed(x[: layout.length]) for x in layout._contents],
                layout._fields,
                layout.length,
                parameters=layout._parameters,
                backend=layout._backend,
            )
        case awkward.contents.RegularArray:
            nplike = layout._backend.nplike
            length = layout.length * layout._size
            content = layout._content[: nplike.shape_item_as_index(length)]

            return awkward.contents.RegularArray(
                _to_packed(content),
                layout._size,
                layout.length,
                parameters=layout._parameters,
            )
        case awkward.contents.UnionArray:
            nplike = layout._backend.nplike
            tags = layout._tags.data
            original_index = index = layout._index.data[
                : nplike.shape_item_as_index(tags.shape[0])
            ]

            contents = list(layout._contents)

            for tag in range(len(layout._contents)):
                is_tag = tags == tag
                num_tag = nplike.index_as_shape_item(nplike.count_nonzero(is_tag))

                if (
                    contents[tag].length is not awkward._nplikes.shape.unknown_length
                    and num_tag is not awkward._nplikes.shape.unknown_length
                    and contents[tag].length > num_tag
                ):
                    if original_index is index:
                        index = index.copy()
                    new_index_values = layout._backend.nplike.arange(
                        num_tag, dtype=index.dtype
                    )
                    index[is_tag] = new_index_values
                    contents[tag] = layout.project(tag)

                contents[tag] = _to_packed(contents[tag])

            return awkward.contents.UnionArray(
                awkward.index.Index8(tags, nplike=layout._backend.nplike),
                awkward.index.Index(index, nplike=layout._backend.nplike),
                contents,
                parameters=layout._parameters,
            )
        case awkward.contents.UnmaskedArray:
            return awkward.contents.UnmaskedArray(
                _to_packed(layout._content),
                parameters=layout._parameters,
            )
        case _:
            msg = f"Array type {type(layout)} cannot be written. If you believe this should be supported, please let the Uproot developers know."
            raise NotImplementedError(msg)


def _reorder_fields_to_match_form(layout, form):
    """
    Recursively reorders record fields in `layout` to match `form`'s field
    order, at every level of nesting (not just the top level). `layout` and
    `form` are assumed to already be form-compatible (same set of fields and
    types at every level, just possibly differently ordered within a record)
    -- see the caller, which only reaches this after that comparison passes.

    This mirrors the node types NTuple_Header._build_field_col_records knows
    how to walk: NumpyForm (no children), ListOffsetForm/RegularForm/
    IndexedOptionForm/UnmaskedForm (one child, via `.content`), RecordForm
    (named children, via `.fields`/`.contents`), and UnionForm (positional
    children, via `.contents`).
    """
    if isinstance(form, awkward.forms.RecordForm):
        target_fields = list(form.fields)
        if list(layout.fields) != target_fields:
            layout = layout[target_fields]
        new_contents = [
            _reorder_fields_to_match_form(layout.content(f), form.content(f))
            for f in target_fields
        ]
        return layout.copy(contents=new_contents)
    elif isinstance(form, awkward.forms.UnionForm):
        new_contents = [
            _reorder_fields_to_match_form(layout.content(i), form.content(i))
            for i in range(len(form.contents))
        ]
        return layout.copy(contents=new_contents)
    elif hasattr(form, "content"):
        return layout.copy(
            content=_reorder_fields_to_match_form(layout.content, form.content)
        )
    return layout


def _regularize_input_type_to_awkward(obj):
    import awkward

    if uproot._util.from_module(obj, "pandas"):
        import pandas

        if isinstance(
            obj, pandas.DataFrame
        ) and uproot._util.pandas_has_attr_is_numeric(pandas)(obj.index):
            obj = uproot.writing._cascadetree.dataframe_to_dict(obj)
            # Try to retype dtype=object columns
            for k in obj.keys():
                if obj[k].dtype == object:
                    obj[k] = awkward.from_iter(obj[k])
            obj = awkward.Array(obj)

    elif isinstance(obj, numpy.ndarray) and obj.dtype.fields is not None:
        obj = awkward.Array(obj)

    elif isinstance(obj, dict):
        # Sort dictionary keys to avoid issues
        obj = {k: obj[k] for k in sorted(obj.keys())}
        obj = awkward.Array(obj)

    elif isinstance(obj, awkward.contents.Content):
        obj = awkward.Array(obj)

    return obj
