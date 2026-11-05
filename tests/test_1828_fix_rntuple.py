# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import json
import pickle
import struct
import sys

import awkward as ak
import numpy as np
import pytest
import xxhash

import uproot
import uproot.const
import uproot.models.RNTuple as models
from uproot.const import kStartBigFile

ak = pytest.importorskip("awkward")


@pytest.fixture
def ntuple_path(tmp_path):
    """A small RNTuple with nested records, lists, and documented fields."""
    path = tmp_path / "nested.root"
    rec = ak.Array(
        [
            {"a": 1, "inner": {"deep": 1.5}},
            {"a": 2, "inner": {"deep": 2.5}},
            {"a": 3, "inner": {"deep": 3.5}},
        ]
    )
    vec = ak.Array([[{"p": 1}], [], [{"p": 2}, {"p": 3}]])
    data = ak.with_field(ak.Array({"x": [10, 20, 30]}), rec, "rec")
    data = ak.with_field(data, vec, "vec")
    data = ak.with_parameter(data, "__doc__", "Some entries")
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", data, description="The description")
    return path


@pytest.fixture
def ntuple(ntuple_path):
    with uproot.open(ntuple_path) as file:
        yield file["ntuple"]


def test_pickle_with_loaded_metadata(ntuple):
    assert ntuple.header.ntuple_name == "ntuple"
    unpickled = pickle.loads(pickle.dumps(ntuple))
    assert unpickled.header.ntuple_name == "ntuple"
    assert unpickled.arrays().x.to_list() == [10, 20, 30]


def test_field_index_of_an_anonymous_field(ntuple):
    anonymous = ntuple["vec"].ntuple.all_fields[ntuple["vec"].field_id + 1]
    assert anonymous.is_anonymous
    with pytest.raises(ValueError, match="anonymous"):
        anonymous.index


def test_iterate_adds_docs(tmp_path):
    path = tmp_path / "documented.root"
    form = ak.forms.RecordForm(
        [ak.forms.NumpyForm("int32", parameters={"__doc__": "the x"})], ["x"]
    )
    with uproot.recreate(path) as file:
        ntuple = file.mkrntuple("ntuple", form)
        ntuple.extend({"x": np.array([1, 2], dtype=np.int32)})
    with uproot.open(path) as file:
        chunks = list(file["ntuple"].iterate(ak_add_doc=True))
    assert [chunk.x.layout.parameter("__doc__") for chunk in chunks] == ["the x"]


@pytest.mark.parametrize(
    "record",
    [
        {"member": 1, ":_0": {"member": 2}, ":_1": {"member": 3}},
        # only in base classes, whose (empty) leaf fields are equal Mappings
        {":_0": {"member": 2}, ":_1": {"member": 3}},
    ],
)
def test_duplicates_that_cannot_be_told_apart(tmp_path, record):
    # uproot writes records without a type name, which is what prefixes are made of
    path = tmp_path / "file.root"
    data = ak.Array([record] * 2)
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"c": data})
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        for _ in range(2):
            with pytest.raises(RuntimeError, match="names are still not unique"):
                ntuple.all_fields


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


def _chunk(data):
    return uproot.source.chunk.Chunk.wrap(None, np.frombuffer(data, dtype=np.uint8))


def _read(reader, data):
    return reader.read(_chunk(data), uproot.source.cursor.Cursor(0), {})


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


@pytest.mark.parametrize("name", ["b", "_b"])
def test_metadata_missing_attributes(name):
    metadata = models.MetaData("Something", a=1)
    with pytest.raises(AttributeError, match=f"^{name}$"):
        getattr(metadata, name)


def test_metadata_pickle():
    metadata = pickle.loads(pickle.dumps(models.MetaData("Something", a=1)))
    assert metadata.name == "Something"
    assert metadata.a == 1


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


def _optional_array_of_optionals():
    """``[[1, None], [3, 4], None]``, of type ``option[2 * ?int64]``."""
    inner = ak.mask(
        ak.Array([1, 2, 3, 4, 5, 6]), [True, False, True, True, False, False]
    )
    regular = ak.contents.RegularArray(inner.layout, 2)
    return ak.mask(ak.Array(regular), [True, True, False])


@pytest.fixture
def jagged_record_path(tmp_path):
    """A record that NumPy can't represent directly, since it has a jagged field."""
    path = tmp_path / "record.root"
    jagged = ak.Array([[1], [1, 2], []])
    record = ak.zip({"j": jagged, "o": _optional_array_of_optionals()}, depth_limit=1)
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"rec": record})
    return path


def test_record_with_optional_arrays_to_numpy(jagged_record_path):
    with uproot.open(jagged_record_path) as file:
        out = file["ntuple"]["rec"].array(library="np")

    assert out.dtype.names == ("j", "o")
    assert out.dtype["j"] == np.dtype(object)
    assert out.dtype["o"] == np.dtype(object)
    assert [row.tolist() for row in out["j"]] == [[1], [1, 2], []]
    # partly missing, complete, and entirely missing
    assert out["o"][0].tolist() == [1, None]
    assert out["o"][1].tolist() == [3, 4]
    assert out["o"][2] is None


def _roundtrip(path, data):
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", data)
    with uproot.open(path) as file:
        return file["ntuple"].arrays()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            ak.contents.BitMaskedArray(
                ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
                ak.contents.NumpyArray(np.array([1, 2, 3])),
                valid_when=True,
                length=3,
                lsb_order=True,
            ),
            [1, None, 3],
        ),
        (
            ak.contents.ByteMaskedArray(
                ak.index.Index8(np.array([1, 0, 1, 0], dtype=np.int8)),
                ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5])),
                valid_when=True,
            ),
            [1, None, 3, None],
        ),
        (
            ak.mask(ak.Array(["a", "bb", "ccc"]), [False, True, True]),
            [None, "bb", "ccc"],
        ),
        (
            ak.contents.UnmaskedArray(ak.contents.NumpyArray(np.array([1, 2, 3]))),
            [1, 2, 3],
        ),
        # a union that only uses the beginning of its contents
        (ak.Array([1, "a", 2, "b"])[:2], [1, "a"]),
        (ak.Array([1, "a", 2, "b"])[1:3], ["a", 2]),
    ],
)
def test_layouts_are_packed(tmp_path, data, expected):
    out = _roundtrip(tmp_path / "file.root", {"a": ak.Array(data)})
    assert out.a.to_list() == expected


@pytest.mark.parametrize(
    "data",
    [
        {"a": ak.Array([[], []])},
        {"a": ak.Array(ak.contents.EmptyArray())},
        ak.forms.RecordForm([ak.forms.EmptyForm()], ["a"]),
    ],
)
def test_fields_of_unknown_type_cannot_be_written(tmp_path, data):
    with uproot.recreate(tmp_path / "file.root") as file:
        with pytest.raises(TypeError, match="'unknown' type"):
            file.mkrntuple("ntuple", data)


def test_cascading_reprs(tmp_path):
    with uproot.recreate(tmp_path / "file.root") as file:
        cascading = file.mkrntuple(
            "ntuple", {"x": np.array([1, 2], np.int64)}
        )._cascading

        field = cascading._header._field_records[0]
        assert repr(field) == (
            "NTuple_Field_Description(0, 0, 0, <RNTupleFieldRole.LEAF: 0>, "
            "<RNTupleFieldFlags.NOFLAG: 0>, 'x', 'std::int64_t', '', '')"
        )

        assert repr(cascading._header).startswith(
            f"NTuple_Header({cascading._header._location}, RecordForm("
        )
        assert repr(cascading._footer).startswith(
            "NTuple_Footer(extension_field_record_frames=[]"
        )
        assert repr(cascading._anchor).startswith("NTuple_Anchor(")
        assert repr(cascading).startswith("NTuple(")
        assert repr(cascading._footer) in repr(cascading)


def make_big(directory, num_bytes=kStartBigFile):
    # skip over two gigabytes of file instead of writing them, leaving a hole
    directory.file._cascading.freesegments.allocate(num_bytes)


skip_without_sparse_files = pytest.mark.skipif(
    sys.platform == "win32", reason="this needs a sparse file to be fast"
)


@skip_without_sparse_files
@pytest.mark.parametrize("data", [None, {"x": np.array([1.5, 2.5])}])
# past 2 GiB, the anchor's key can't hold its location in a 4-byte fSeekKey
@pytest.mark.parametrize("num_bytes", [kStartBigFile, 2**31 + 1000])
def test_rntuple_past_the_big_file_boundary(tmp_path, data, num_bytes):
    path = tmp_path / "big.root"
    with uproot.recreate(path, compression=None) as file:
        file["small"] = "before the boundary"
        make_big(file, num_bytes)
        # the anchor's key is big, and so must be its copy in the directory's
        # list of keys, which readers take fKeylen from
        if data is None:
            file.mkrntuple("ntuple", {"x": "float64"})
        else:
            file.mkrntuple("ntuple", data)

    with uproot.open(path) as file:
        assert file["small"] == "before the boundary"
        key = file.key("ntuple")
        assert key.fSeekKey >= num_bytes
        assert key.fVersion > 1000  # a big key
        ntuple = file["ntuple"]
        assert ntuple.num_entries == (0 if data is None else 2)
        assert ntuple.arrays()["x"].tolist() == ([] if data is None else [1.5, 2.5])


_WRITERS = """
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace test_1828 {

struct Base {
   int member;
   float other;
};
struct Child : Base {
   int member;
};
struct Left {
   int shared;
};
struct Right {
   int shared;
};
struct Both : Left, Right {
   int member;
};

void write_projections(const char *path)
{
   auto model = ROOT::RNTupleModel::Create();
   auto pt = model->MakeField<float>("pt");
   auto values = model->MakeField<std::vector<int>>("values");
   model
      ->AddProjectedField(ROOT::RFieldBase::Create("pt_alias", "float").Unwrap(),
                          [](const std::string &) { return std::string("pt"); })
      .ThrowOnError();
   model
      ->AddProjectedField(
         ROOT::RFieldBase::Create("num_values", "ROOT::RNTupleCardinality<std::uint64_t>").Unwrap(),
         [](const std::string &) { return std::string("values"); })
      .ThrowOnError();
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", path);
   for (int i = 0; i < 4; i++) {
      *pt = 1.5f * i;
      *values = std::vector<int>(i, i);
      writer->Fill();
   }
}

void write_shadowed_members(const char *path)
{
   auto model = ROOT::RNTupleModel::Create();
   auto child = model->MakeField<Child>("child");
   auto both = model->MakeField<Both>("both");
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", path);
   for (int i = 0; i < 3; i++) {
      child->member = i;
      child->Base::member = 10 * i;
      child->other = 0.5f * i;
      both->member = 100 * i;
      both->Left::shared = 200 * i;
      both->Right::shared = 300 * i;
      writer->Fill();
   }
}

void write_streamer_field(const char *path)
{
   auto model = ROOT::RNTupleModel::Create();
   model->MakeField<int>("x");
   model->AddField(std::make_unique<ROOT::RStreamerField>("streamed", "std::vector<int>"));
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", path);
   auto entry = writer->CreateEntry();
   auto x = entry->GetPtr<int>("x");
   auto streamed = entry->GetPtr<std::vector<int>>("streamed");
   for (int i = 0; i < 2; i++) {
      *x = i;
      streamed->push_back(i);
      writer->Fill(*entry);
   }
}

void write_without_fields(const char *path)
{
   auto writer = ROOT::RNTupleWriter::Recreate(ROOT::RNTupleModel::Create(), "ntuple", path);
}

std::string show(const char *path, int entry)
{
   std::ostringstream stream;
   ROOT::RNTupleReader::Open("ntuple", path)->Show(entry, stream);
   return stream.str();
}

// Show doesn't support variants
std::string show_variant(const char *path, int entry)
{
   auto reader = ROOT::RNTupleReader::Open("ntuple", path);
   auto view = reader->GetView<std::variant<std::int64_t, std::string>>("a");
   auto value = view(entry);
   if (value.index() == 0)
      return std::to_string(std::get<0>(value));
   return std::get<1>(value);
}

} // namespace test_1828
"""


_ATTRIBUTE_WRITER = """
#include <ROOT/RNTupleAttrWriting.hxx>

namespace test_1828 {

void write_attributes(const char *path)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(path, "RECREATE"));
   auto model = ROOT::RNTupleModel::Create();
   auto x = model->MakeField<int>("x");
   auto writer = ROOT::RNTupleWriter::Append(std::move(model), "ntuple", *file);
   auto attribute_model = ROOT::RNTupleModel::Create();
   auto note = attribute_model->MakeField<std::string>("note");
   auto attributes = writer->CreateAttributeSet(std::move(attribute_model), "my_attributes");
   auto range = attributes->BeginRange();
   *note = "a note";
   for (int i = 0; i < 4; i++) {
      *x = i;
      writer->Fill();
   }
   attributes->CommitRange(std::move(range));
   writer.reset();
}

} // namespace test_1828
"""


@pytest.fixture(scope="module")
def root():
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 63800:
        pytest.skip("needs the RNTuple interfaces of a recent ROOT")
    # ROOT warnings become Python warnings, which are errors in this test suite. The
    # classes above have no dictionaries, which ROOT warns about.
    ignore_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kError
    if not hasattr(ROOT, "test_1828"):
        assert ROOT.gInterpreter.Declare(_WRITERS)
        if ROOT.gROOT.GetVersionInt() >= 64000:
            assert ROOT.gInterpreter.Declare(_ATTRIBUTE_WRITER)
    yield ROOT.test_1828
    ROOT.gErrorIgnoreLevel = ignore_level


def _show(root, path, entry):
    return json.loads(str(root.show(str(path), entry)))


def test_members_that_shadow_inherited_members(root, tmp_path):
    path = tmp_path / "file.root"
    root.write_shadowed_members(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        assert sorted(ntuple["child"].keys()) == [
            "member",
            "other",
            "test_1828::Base::member",
        ]
        # members of different bases are told apart by their own prefix, and the one
        # that is not repeated in a base stays as it is
        assert sorted(ntuple["both"].keys()) == [
            "member",
            "test_1828::Left::shared",
            "test_1828::Right::shared",
        ]
        out = ntuple.arrays()
        assert out.child.member.to_list() == [0, 1, 2]
        assert out.child["test_1828::Base::member"].to_list() == [0, 10, 20]
        assert out.both["test_1828::Left::shared"].to_list() == [0, 200, 400]
        assert out.both["test_1828::Right::shared"].to_list() == [0, 300, 600]


def test_linked_attribute_set_written_by_root(root, tmp_path):
    if not hasattr(root, "write_attributes"):
        pytest.skip("needs RNTuple attributes")
    path = tmp_path / "file.root"
    root.write_attributes(str(path))
    with uproot.open(path) as file:
        ntuple = file["ntuple"]
        linked = ntuple.footer.linked_attribute_sets
        assert [record.attribute_set_name for record in linked] == ["my_attributes"]
        assert ntuple.arrays().x.to_list() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            ak.contents.BitMaskedArray(
                ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
                ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
                valid_when=True,
                length=3,
                lsb_order=True,
            ),
            [1, None, 3],
        ),
        (
            ak.contents.UnmaskedArray(
                ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32))
            ),
            [1, 2],
        ),
    ],
)
def test_root_reads_packed_layouts(root, tmp_path, data, expected):
    path = tmp_path / "file.root"
    with uproot.recreate(path) as file:
        file.mkrntuple("ntuple", {"a": ak.Array(data)})
    assert [_show(root, path, i)["a"] for i in range(len(expected))] == expected
