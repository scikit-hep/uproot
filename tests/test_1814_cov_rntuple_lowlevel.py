# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pickle
import struct
import types

import numpy as np
import pytest
import xxhash

import uproot
import uproot.const
import uproot.models.RNTuple as models
from uproot.interpretation.objects import CannotBeForth

ak = pytest.importorskip("awkward")


def _write(path, data):
    with uproot.recreate(path, compression=None) as file:
        file.mkrntuple("ntuple", data)
    return path


def _list_frame_items(buffer, position):
    """
    Returns the positions of the payloads of a list frame of record frames, and the
    position that follows the list frame.
    """
    list_size, num_items = struct.unpack_from("<qI", buffer, position)
    assert list_size < 0
    items = []
    item = position + 12
    for _ in range(num_items):
        items.append(item + 8)
        item += struct.unpack_from("<q", buffer, item)[0]
    return items, position - list_size


def _record_positions(header):
    """
    Returns the positions of the field records and of the column records in a header
    envelope.
    """
    position = 16  # envelope type and size, feature flags
    for _ in range(3):  # name, description, writer
        position += 4 + struct.unpack_from("<I", header, position)[0]
    fields, position = _list_frame_items(header, position)
    columns, _ = _list_frame_items(header, position)
    return fields, columns


def _set_parent(header, fields, field_id, parent_id):
    struct.pack_into("<I", header, fields[field_id] + 8, parent_id)


def _set_role(header, fields, field_id, role):
    struct.pack_into("<H", header, fields[field_id] + 12, role)


def _set_column_type(header, columns, column_id, column_type):
    struct.pack_into("<H", header, columns[column_id], column_type)


def _set_column_field(header, columns, column_id, field_id):
    struct.pack_into("<I", header, columns[column_id] + 4, field_id)


def _rewrite_header(path, patch):
    """
    Lets ``patch(header, fields, columns)`` change the header of an RNTuple in place and
    updates the checksums that depend on it, so that only the patch is wrong in the file.
    """
    with uproot.open(path) as file:
        members = file["ntuple"]._members
    assert members["fNBytesHeader"] == members["fLenHeader"]
    assert members["fNBytesFooter"] == members["fLenFooter"]

    raw = bytearray(path.read_bytes())
    start, stop = members["fSeekHeader"], members["fSeekHeader"] + members["fLenHeader"]
    header = bytearray(raw[start:stop])
    patch(header, *_record_positions(header))
    checksum = xxhash.xxh3_64_intdigest(bytes(header[:-8]))
    header[-8:] = struct.pack("<Q", checksum)
    raw[start:stop] = header

    start, stop = members["fSeekFooter"], members["fSeekFooter"] + members["fLenFooter"]
    footer = bytearray(raw[start:stop])
    struct.pack_into("<Q", footer, 16, checksum)
    footer[-8:] = struct.pack("<Q", xxhash.xxh3_64_intdigest(bytes(footer[:-8])))
    raw[start:stop] = footer

    path.write_bytes(raw)


def _union(*contents, tags, index):
    return ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.array(tags, dtype=np.int8)),
            ak.index.Index64(np.array(index, dtype=np.int64)),
            [ak.Array(content).layout for content in contents],
        )
    )


def _chunk(data):
    return uproot.source.chunk.Chunk.wrap(None, np.frombuffer(data, dtype=np.uint8))


def _read(reader, data):
    return reader.read(_chunk(data), uproot.source.cursor.Cursor(0), {})


def test_rewriting_the_header_changes_only_what_is_patched(tmp_path):
    path = _write(tmp_path / "file.root", {"v": ak.Array([[1], [2, 3]])})
    _rewrite_header(path, lambda header, fields, columns: None)
    with uproot.open(path) as file:
        assert file["ntuple"].arrays().v.to_list() == [[1], [2, 3]]


def test_unsplit_field(tmp_path):
    path = _write(tmp_path / "file.root", {"rec": ak.Array([{"a": 1}]), "x": [1]})

    def patch(header, fields, columns):
        _set_role(header, fields, 0, uproot.const.RNTupleFieldRole.STREAMER)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert (
            ntuple["rec"].record.struct_role == uproot.const.RNTupleFieldRole.STREAMER
        )
        assert ntuple.arrays(filter_name="x").x.to_list() == [1]
        with pytest.raises(NotImplementedError, match="Unsplit fields are not"):
            ntuple.arrays(filter_name="rec")


def test_unknown_field_role(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2])})

    def patch(header, fields, columns):
        _set_role(header, fields, 0, 7)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        with pytest.raises(NotImplementedError, match="Unknown field role"):
            file["ntuple"].arrays()


def test_split_column_of_an_unsupported_bit_width(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.arange(4, dtype=np.int32)})

    def patch(header, fields, columns):
        split_int32 = uproot.const.rntuple_col_type_to_num_dict["splitint32"]
        _set_column_type(header, columns, 0, split_int32)
        struct.pack_into("<H", header, columns[0] + 2, 24)  # the column's bit width

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        assert file["ntuple"].column_records[0].nbits == 24
        with pytest.raises(ValueError, match="split column 0 has 24 bits"):
            file["ntuple"].arrays()


def test_string_column_without_chars(tmp_path):
    path = _write(tmp_path / "file.root", {"s": ak.Array(["a", "bb"])})

    def patch(header, fields, columns):
        _set_column_type(
            header, columns, 1, uproot.const.rntuple_col_type_to_num_dict["uint8"]
        )

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        with pytest.raises(RuntimeError, match="Missing special case: 0"):
            file["ntuple"].arrays()


def test_collection_without_columns(tmp_path):
    path = _write(
        tmp_path / "file.root", {"v": ak.Array([[1], [2, 3]]), "y": np.array([1, 2])}
    )

    def patch(header, fields, columns):
        _set_column_field(header, columns, 0, 2)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        with pytest.raises(RuntimeError, match="field_id: 0 is missing"):
            file["ntuple"].arrays(filter_name="v")


def test_col_form_of_a_field_without_columns(tmp_path):
    path = _write(tmp_path / "file.root", {"rec": ak.Array([{"a": 1}])})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        with pytest.raises(RuntimeError, match="field_id: 0 is missing"):
            ntuple.col_form(ntuple["rec"].field_id)


def test_collection_of_many_fields(tmp_path):
    path = _write(tmp_path / "file.root", {"v": ak.Array([[{"x": 1, "y": 2}]])})

    def patch(header, fields, columns):
        for child in (2, 3):
            _set_parent(header, fields, child, 0)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert ntuple.keys() == ["v"]
        # subfields of a collection are anonymous, so they cannot be selected
        form = ntuple.to_akform()[0].content("v")
        assert isinstance(form, ak.forms.RecordForm)
        assert form.contents == []


def test_collection_of_many_fields_in_a_variant(tmp_path):
    variant = _union([1], [[{"x": 1, "y": 2}]], tags=[0, 1], index=[0, 0])
    path = _write(tmp_path / "file.root", {"u": variant})

    def patch(header, fields, columns):
        for child in (4, 5):
            _set_parent(header, fields, child, 2)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        role = ntuple.field_records[2].struct_role
        assert role == uproot.const.RNTupleFieldRole.COLLECTION
        form = ntuple.field_form(2, [])
        assert isinstance(form, ak.forms.RecordForm)
        assert form.fields == ["_0", "x", "y"]


def test_collection_without_fields(tmp_path):
    path = _write(tmp_path / "file.root", {"v": ak.Array([[1], [2, 3]])})

    def patch(header, fields, columns):
        _set_parent(header, fields, 1, 1)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert ntuple.field_form(0, []).contents == []


def test_variant_without_alternatives(tmp_path):
    path = _write(
        tmp_path / "file.root", {"u": _union([1], [1.5], tags=[0, 1], index=[0, 0])}
    )

    def patch(header, fields, columns):
        for child in (1, 2):
            _set_parent(header, fields, child, child)

    _rewrite_header(path, patch)
    with uproot.open(path) as file:
        form = file["ntuple"].field_form(0, [])
        assert isinstance(form, ak.forms.UnionForm)
        # only the one that stands for an invalid variant
        assert [type(content) for content in form.contents] == [
            ak.forms.IndexedOptionForm
        ]
        assert isinstance(form.contents[0].content, ak.forms.EmptyForm)


def test_header_chunk_is_not_prepared_twice(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2])})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        ntuple._prepare_header_chunk()
        ntuple._prepare_footer_chunk()
        header_chunk, footer_chunk = ntuple._header_chunk, ntuple._footer_chunk
        assert ntuple.header.ntuple_name == "ntuple"
        assert ntuple.footer.cluster_group_records[0].num_clusters == 1
        assert ntuple._header_chunk is header_chunk
        assert ntuple._footer_chunk is footer_chunk


def test_memberwise_anchor(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2])})
    with uproot.open(path) as file:
        members = file["ntuple"]._members
    anchor = models._rntuple_anchor_format.pack(*members.values())
    raw = bytearray(path.read_bytes())
    assert raw.count(anchor) == 1
    version = raw.index(anchor) - 2
    assert struct.unpack_from(">H", raw, version)[0] < 0x4000
    raw[version] |= 0x40
    path.write_bytes(raw)

    with uproot.open(path) as file:
        with pytest.raises(NotImplementedError, match="memberwise serialization"):
            file["ntuple"]


def test_big_endian_systems_are_rejected(tmp_path, monkeypatch):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2])})
    monkeypatch.setattr(models, "sys", types.SimpleNamespace(byteorder="big"))
    with uproot.open(path) as file:
        with pytest.raises(NotImplementedError, match="little-endian"):
            file["ntuple"]


def test_rntuple_cannot_be_forth(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2])})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        context = {"forth": types.SimpleNamespace(gen=object())}
        with pytest.raises(CannotBeForth):
            ntuple.read_members(None, None, context, ntuple.file)


@pytest.mark.parametrize(
    "field",
    [
        {"v": ak.Array([[1], [2, 3]])},
        {"u": _union([1], [1.5], tags=[0, 1], index=[0, 0])},
    ],
)
def test_base_col_form_of_columns_without_a_form(tmp_path, field):
    # index and switch columns only give their buffers a name
    path = _write(tmp_path / "file.root", field)
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert ntuple.base_col_form(ntuple.column_records[0]) == "column-0"


def test_cluster_range_skips_negative_clusters(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2, 3])})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        length, starts, _ = ntuple._expected_array_length_starts_dtype(0, -1, 1)
        assert (length, starts) == (3, [0])
        assert ntuple.read_cluster_range(0, -1, 1).tolist() == [1, 2, 3]


def test_access_log_must_be_extendable(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2, 3])})
    with uproot.open(path) as file:
        arrays = file["ntuple"].arrays(virtual=True, access_log=object())
        with pytest.raises(ValueError, match="needs to implement '__iadd__'"):
            arrays.x.to_list()


def test_read_cluster_pages_without_destination_from_cache(tmp_path):
    path = _write(tmp_path / "file.root", {"x": np.array([1, 2, 3])})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        metadata = ntuple.get_field_metadata(0)
        cache = {}
        out = ntuple.read_cluster_pages(0, 0, metadata, array_cache=cache)
        assert out.tolist() == [1, 2, 3]
        assert list(cache) == [f"{ntuple.cache_key}:0:0"]

        cached = ntuple.read_cluster_pages(0, 0, metadata, array_cache=cache)
        assert cached is cache[f"{ntuple.cache_key}:0:0"]


def test_large_locator():
    data = struct.pack("<iQQ", -((1 << 24) | 16), 2**33, 2**34)
    locator = _read(models.LocatorReader(), data)
    assert locator.type == uproot.const.RNTupleLocatorType.LARGE
    assert locator.num_bytes == 2**33
    assert locator.offset == 2**34


def test_unknown_locator_type():
    with pytest.raises(NotImplementedError, match="Unknown locator type"):
        _read(models.LocatorReader(), struct.pack("<iQ", -(2 << 24), 0))


def test_metadata_repr():
    metadata = models.MetaData("Something", a=1, b="two")
    assert metadata.name == "Something"
    assert repr(metadata) == "MetaData('Something', a=1, b='two')"


@pytest.mark.parametrize("name", ["b", "_b"])
def test_metadata_missing_attributes(name):
    metadata = models.MetaData("Something", a=1)
    with pytest.raises(AttributeError, match=f"^{name}$"):
        getattr(metadata, name)


def test_metadata_pickle():
    metadata = pickle.loads(pickle.dumps(models.MetaData("Something", a=1)))
    assert metadata.name == "Something"
    assert metadata.a == 1


def test_extra_type_info():
    name = b"a type"
    data = struct.pack("<III", 0, 7, len(name)) + name
    info = _read(models.ExtraTypeInfoReader(), data)
    assert info.content_id == uproot.const.RNTupleExtraTypeIdentifier.ROOT
    assert info.type_ver == 7
    assert info.type_name == "a type"


def test_sharded_cluster():
    flags = int(uproot.const.RNTupleClusterFlags.SHARDED)
    data = struct.pack("<QQ", 0, (flags << 56) | 5)
    with pytest.raises(NotImplementedError, match="Sharded clusters"):
        _read(models.ClusterSummaryReader(), data)


def test_linked_attribute_set():
    name = b"attributes"
    data = (
        struct.pack("<HHI", 1, 2, 300)
        + struct.pack("<iQ", 100, 200)
        + struct.pack("<I", len(name))
        + name
    )
    linked = _read(models.LinkedAttributeSetRecordReader(), data)
    assert linked.schema_version_major == 1
    assert linked.schema_version_minor == 2
    assert linked.anchor_uncompressed_size == 300
    assert linked.locator.num_bytes == 100
    assert linked.locator.offset == 200
    assert linked.attribute_set_name == "attributes"


def _with_a_wider_representation(path, column_type, nbits):
    """
    Turns the column of ``x`` into ``column_type`` and makes the column of ``y`` a
    second (wider) representation of ``x``, as when a field changes representation
    from one cluster to another.
    """

    def patch(header, fields, columns):
        _set_column_type(
            header,
            columns,
            0,
            uproot.const.rntuple_col_type_to_num_dict[column_type],
        )
        struct.pack_into("<H", header, columns[0] + 2, nbits)
        _set_column_field(header, columns, 1, 0)

    _rewrite_header(path, patch)


def test_zigzag_column_with_a_wider_representation(tmp_path):
    values = [-32768, 32767, -1, 0, 16384, -16385, 1, -2]
    # the zigzag encoding of the values, (v << 1) ^ (v >> 15), by hand
    zigzag = np.array([65535, 65534, 1, 0, 32768, 32769, 2, 3], dtype="<u2")
    # a split column stores the low bytes of all values, then the high bytes
    split = zigzag.view(np.uint8).reshape(-1, 2).T.ravel().view("<i2")
    path = _write(
        tmp_path / "file.root",
        {"x": split, "y": np.zeros(len(values), dtype=np.int32)},
    )
    _with_a_wider_representation(path, "splitint16", 16)

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        field_metadata = ntuple.get_field_metadata(0)
        assert field_metadata.zigzag
        assert field_metadata.dtype == np.dtype("int16")
        assert field_metadata.dtype_result == np.dtype("int32")
        out = ntuple.read_cluster_range(0, 0, 1)
    assert out.dtype == np.dtype("int32")
    assert out.tolist() == values


def test_truncated_float_column_with_a_wider_representation(tmp_path):
    values = [1.5, -2.25, 0.0, 1024.0]
    # the 16 high bits of each float32, packed one after the other
    packed = np.array([0x3FC0, 0xC010, 0x0000, 0x4480], dtype="<u2").tobytes()
    # the page is read as far as the truncated floats need, the rest is ignored
    x = np.frombuffer(packed + bytes(len(packed)), dtype=np.float32)
    path = _write(
        tmp_path / "file.root",
        {"x": x, "y": np.zeros(len(values), dtype=np.float64)},
    )
    _with_a_wider_representation(path, "real32trunc", 16)

    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        field_metadata = ntuple.get_field_metadata(0)
        assert field_metadata.dtype_str == "real32trunc"
        assert field_metadata.dtype_result == np.dtype("float64")
        out = ntuple.read_cluster_range(0, 0, 1)
    assert out.dtype == np.dtype("float64")
    assert out.tolist() == values
