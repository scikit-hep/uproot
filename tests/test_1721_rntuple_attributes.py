# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import io
import os
import pickle
import struct

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

# ranges shared by the lookup tests: two overlapping, one empty, one disjoint
RANGES = [(0, 10), (5, 0), (5, 7), (12, 8)]
TAGS = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}, {"tag": "d"}]


def _attribute_set_record(key, name, schema_version=(1, 0)):
    """
    Builds the footer record that ROOT writes for a linked attribute set.

    ROOT points the locator at the payload of the anchor key of the attribute
    ``RNTuple``, which is what a ``ReadOnlyKey`` exposes through its data cursor and
    its compressed and uncompressed sizes.
    """
    record = uproot.models.RNTuple.MetaData("LinkedAttributeSetRecord")
    record.schema_version_major, record.schema_version_minor = schema_version
    record.anchor_uncompressed_size = key.data_uncompressed_bytes
    record.locator = uproot.models.RNTuple.MetaData("Locator")
    record.locator.offset = key.data_cursor.index
    record.locator.num_bytes = key.data_compressed_bytes
    record.attribute_set_name = name
    return record


def _attribute_data(ranges, user_data):
    # an attribute set with no records still needs a schema, so build one row and drop it
    starts = [start for start, _ in ranges] or [0]
    lengths = [length for _, length in ranges] or [0]
    data = ak.Array(
        {
            "_rangeStart": numpy.array(starts, dtype=numpy.uint64),
            "_rangeLen": numpy.array(lengths, dtype=numpy.uint64),
            "_userData": ak.Array(user_data[: len(starts)]),
        }
    )
    return data[: len(ranges)]


def _ntuple_with_attributes(path, sets, schema_versions=None):
    """
    Writes an ``RNTuple`` named ``Events`` together with one ``RNTuple`` per named
    attribute set in ``sets``, and links them the way ROOT does.

    No test file with attributes is available yet and ROOT only started writing them
    in 6.40, so the links are made here instead of coming from the file itself.
    """
    events = ak.Array({"x": numpy.arange(20, dtype=numpy.int64)})
    schema_versions = {} if schema_versions is None else schema_versions

    with uproot.recreate(path) as file:
        file.mkrntuple("Events", events.layout.form).extend(events)
        for name, (ranges, user_data) in sets.items():
            data = _attribute_data(ranges, user_data)
            writable = file.mkrntuple(name, data.layout.form)
            if len(data) > 0:
                writable.extend(data)

    file = uproot.open(path)
    ntuple = file["Events"]
    ntuple.footer.linked_attribute_sets = [
        _attribute_set_record(file.key(name), name, schema_versions.get(name, (1, 0)))
        for name in sets
    ]
    return ntuple


def _one_set(path, ranges=RANGES, user_data=TAGS, schema_version=(1, 0)):
    ntuple = _ntuple_with_attributes(
        path,
        {"Calibration": (ranges, user_data)},
        {"Calibration": schema_version},
    )
    return ntuple, ntuple.attributes["Calibration"]


def test_no_attributes(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as file:
        file.mkrntuple(
            "ntuple", ak.forms.RecordForm([ak.forms.NumpyForm("int32")], ["one"])
        )

    assert uproot.open(filepath)["ntuple"].attributes == {}

    # v1.0.0.0 predates attributes entirely, v1.0.1.0 has an empty list of them
    for filename in (
        "ntpl001_staff_rntuple_v1-0-0-0.root",
        "ntpl001_staff_rntuple_v1-0-1-0.root",
    ):
        with uproot.open(skhep_testdata.data_path(filename)) as file:
            assert file["Staff"].attributes == {}


def test_attribute_set_record():
    payload = b"".join(
        [
            struct.pack("<HHI", 1, 0, 78),  # schema version and anchor size
            struct.pack("<iQ", 42, 1024),  # locator
            struct.pack("<I", 11),  # name
            b"Calibration",
        ]
    )
    chunk = uproot.source.chunk.Chunk.wrap(None, payload)

    record = uproot.models.RNTuple.LinkedAttributeSetRecordReader().read(
        chunk, uproot.source.cursor.Cursor(0), {}
    )

    assert record.schema_version_major == 1
    assert record.schema_version_minor == 0
    assert record.anchor_uncompressed_size == 78
    assert record.locator.num_bytes == 42
    assert record.locator.offset == 1024
    assert record.attribute_set_name == "Calibration"


def test_attribute_ntuple(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"), [(0, 20)], TAGS[:1])

    assert attributes.name == "Calibration"
    assert attributes.schema_version == (1, 0)

    # the escape hatch exposes the internal layout of the attribute set
    linked = attributes.ntuple
    assert linked.name == "Calibration"
    assert linked.keys(recursive=False) == ["_rangeStart", "_rangeLen", "_userData"]


def test_parent_and_object_path(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    ntuple, attributes = _one_set(filepath, [(0, 20)], TAGS[:1])
    offset = uproot.open(filepath).key("Calibration").data_cursor.index

    # the linked RNTuple is not in any TDirectory, so it must not claim the path of
    # the RNTuple it is linked to
    assert attributes.ntuple.parent is attributes
    assert attributes.object_path == f"(seek pos {offset})/Calibration"
    assert attributes.ntuple.object_path == attributes.object_path
    assert ntuple.object_path == "/Events;1"


def test_attribute_ntuple_compressed_anchor():
    # the anchor written by ROOT is compressed, unlike the one Uproot writes
    with uproot.open(
        skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-1-0.root")
    ) as file:
        staff = file["Staff"]
        key = file.key("Staff")
        assert key.data_compressed_bytes < key.data_uncompressed_bytes

        record = _attribute_set_record(key, "Whatever")
        attributes = uproot.models.RNTuple.RNTupleAttributeSet(record, staff)
        linked = staff.read_attribute_ntuple(record, attributes)
        assert linked.name == "Staff"
        assert linked.num_entries == staff.num_entries


def test_multiple_attribute_sets(tmp_path):
    ntuple = _ntuple_with_attributes(
        os.path.join(tmp_path, "test.root"),
        {
            "Calibration": ([(0, 20)], [{"tag": "v1"}]),
            "Alignment": ([(0, 10), (10, 10)], [{"pass": 1}, {"pass": 2}]),
        },
    )

    # the dict follows the order of the records in the footer
    assert list(ntuple.attributes) == ["Calibration", "Alignment"]
    assert ntuple.attributes["Calibration"].keys() == ["tag"]
    assert ntuple.attributes["Alignment"].keys() == ["pass"]
    assert ntuple.attributes["Alignment"].arrays()["pass"].tolist() == [1, 2]
    assert ntuple.attributes["Alignment"].ranges.tolist() == [(0, 10), (10, 10)]


def test_user_fields(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"),
        [(0, 10), (10, 10)],
        [{"tag": "v1", "index": 1}, {"tag": "v2", "index": 2}],
    )

    assert attributes.keys() == ["tag", "index"]
    assert attributes.typenames() == {"tag": "std::string", "index": "std::int64_t"}
    assert attributes.arrays().tolist() == [
        {"tag": "v1", "index": 1},
        {"tag": "v2", "index": 2},
    ]

    # the meta fields of the attribute schema stay out of the attribute interface
    assert "_rangeStart" not in attributes
    assert attributes.get("_rangeStart") is None
    assert len(attributes) == 2
    assert [field.name for field in attributes] == ["tag", "index"]
    assert [name for name, _ in attributes.items()] == ["tag", "index"]
    assert [field.name for field in attributes.values()] == ["tag", "index"]
    assert attributes.get("tag") is attributes["tag"]
    assert attributes["tag"].array().tolist() == ["v1", "v2"]


def test_filters(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"),
        [(0, 20)],
        [{"tag": "v1", "index": 1}],
    )

    assert attributes.keys(filter_name="tag") == ["tag"]
    assert attributes.typenames(filter_name="index") == {"index": "std::int64_t"}
    assert attributes.arrays(filter_name="tag").tolist() == [{"tag": "v1"}]
    assert attributes.arrays(filter_typename="std::int64_t").tolist() == [{"index": 1}]

    arrays = attributes.arrays(library="np")
    assert list(arrays) == ["tag", "index"]
    assert arrays["tag"].tolist() == ["v1"]
    assert arrays["index"].tolist() == [1]


def test_pandas(tmp_path):
    pytest.importorskip("pandas")
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"),
        [(0, 10), (10, 10)],
        [{"tag": "v1", "index": 1}, {"tag": "v2", "index": 2}],
    )

    df = attributes.arrays(library="pd")
    assert list(df.columns) == ["tag", "index"]
    assert df["tag"].tolist() == ["v1", "v2"]
    assert df["index"].tolist() == [1, 2]
    assert "v2" in repr(df)
    assert attributes.arrays(library="pd", entry_start=1).index.tolist() == [1]


def test_entry_start_stop(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    # these count attribute entries, not entries of the linked RNTuple
    assert attributes.num_entries == len(RANGES)
    assert attributes.arrays(entry_start=1, entry_stop=3).tag.tolist() == ["b", "c"]
    assert attributes.arrays(entry_start=-1).tag.tolist() == ["d"]


def test_iterate(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert [arrays.tag.tolist() for arrays in attributes.iterate(step_size=3)] == [
        ["a", "b", "c"],
        ["d"],
    ]


def test_show(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"),
        [(0, 20)],
        [{"tag": "v1", "index": 1}],
    )

    stream = io.StringIO()
    attributes.show(stream=stream)
    assert stream.getvalue() == (
        "Calibration (ROOT::RNTuple)\n"
        "├─ tag (std::string)\n"
        "└─ index (std::int64_t)\n"
    )


def test_ranges(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert attributes.ranges.dtype.names == ("start", "length")
    assert attributes.ranges.tolist() == RANGES


def test_no_records(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"), [], TAGS[:1])

    assert attributes.keys() == ["tag"]
    assert attributes.ranges.tolist() == []
    assert attributes.arrays().tolist() == []
    assert attributes.get_attributes().tolist() == []
    assert attributes.get_attributes(0).tolist() == []


def test_get_attributes(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    # without an entry, every attribute entry is returned, including empty ranges
    assert attributes.get_attributes().tolist() == [0, 1, 2, 3]
    assert attributes.get_attributes(0).tolist() == [0]
    assert attributes.get_attributes(9).tolist() == [0, 2]
    assert attributes.get_attributes(11).tolist() == [2]
    assert attributes.get_attributes(19).tolist() == [3]
    # the empty range starting at 5 never matches, and neither does an unused entry
    assert attributes.get_attributes(5).tolist() == [0, 2]
    assert attributes.get_attributes(100).tolist() == []

    assert attributes.arrays()[attributes.get_attributes(11)].tag.tolist() == ["c"]


def test_get_attributes_containing_range(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert attributes.get_attributes_containing_range(0, 1).tolist() == [0]
    assert attributes.get_attributes_containing_range(5, 10).tolist() == [0, 2]
    assert attributes.get_attributes_containing_range(10, 12).tolist() == [2]
    # overlapping these entries is not enough, a range has to cover all of them
    assert attributes.get_attributes_containing_range(5, 13).tolist() == []
    assert attributes.get_attributes_containing_range(0, 20).tolist() == []
    assert attributes.get_attributes_containing_range(5, 5).tolist() == []
    assert attributes.get_attributes_containing_range(8, 3).tolist() == []


def test_get_attributes_in_range(tmp_path):
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"))

    assert attributes.get_attributes_in_range(0, 20).tolist() == [0, 2, 3]
    assert attributes.get_attributes_in_range(0, 10).tolist() == [0]
    assert attributes.get_attributes_in_range(5, 20).tolist() == [2, 3]
    # overlapping these entries is not enough, a range has to lie within them
    assert attributes.get_attributes_in_range(5, 10).tolist() == []
    # the empty range at 5 lies within these entries, but empty ranges never match
    assert attributes.get_attributes_in_range(5, 12).tolist() == [2]
    assert attributes.get_attributes_in_range(5, 5).tolist() == []
    assert attributes.get_attributes_in_range(20, 0).tolist() == []


def test_sorted_by_start(tmp_path):
    # ROOT writes a range when it is committed, so nested ranges are out of order
    ranges = [(10, 5), (0, 20), (10, 0), (0, 5)]
    _, attributes = _one_set(os.path.join(tmp_path, "test.root"), ranges)

    assert attributes.ranges.tolist() == ranges
    assert attributes.arrays().tag.tolist() == ["a", "b", "c", "d"]
    # ties keep the order in which the ranges were written
    assert attributes.get_attributes().tolist() == [1, 3, 0, 2]
    assert attributes.get_attributes(12).tolist() == [1, 0]
    assert attributes.get_attributes_containing_range(0, 3).tolist() == [1, 3]
    assert attributes.get_attributes_in_range(0, 20).tolist() == [1, 3, 0]
    assert attributes.arrays()[attributes.get_attributes(12)].tag.tolist() == [
        "b",
        "a",
    ]


def test_cache_key(tmp_path):
    ntuple, attributes = _one_set(
        os.path.join(tmp_path, "test.root"), [(0, 20)], TAGS[:1]
    )
    assert attributes.ntuple.cache_key == attributes.cache_key
    assert attributes.cache_key != ntuple.cache_key

    # the two RNTuples share the file's array cache, so colliding cache keys would
    # make them read each other's pages
    assert ntuple.arrays().x.tolist() == list(range(20))
    assert attributes.ranges.tolist() == [(0, 20)]
    assert attributes.arrays().tag.tolist() == ["a"]
    assert ntuple.arrays().x.tolist() == list(range(20))


def test_cache_key_ignores_name(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    events = ak.Array({"x": numpy.arange(20, dtype=numpy.int64)})
    with uproot.recreate(filepath) as file:
        file.mkrntuple("Events", events.layout.form).extend(events)
        for name, tag in [("first", "a"), ("second", "b")]:
            data = _attribute_data([(0, 20)], [{"tag": tag}])
            file.mkrntuple(name, data.layout.form).extend(data)

    file = uproot.open(filepath)
    ntuple = file["Events"]
    # attribute set names are free text and can look like parts of a cache key
    names = ["Events;1", "Events;1:"]
    ntuple.footer.linked_attribute_sets = [
        _attribute_set_record(file.key("first"), names[0]),
        _attribute_set_record(file.key("second"), names[1]),
    ]
    first, second = (ntuple.attributes[name] for name in names)

    assert len({ntuple.cache_key, first.cache_key, second.cache_key}) == 3
    assert first.arrays().tag.tolist() == ["a"]
    assert second.arrays().tag.tolist() == ["b"]
    assert ntuple.arrays().x.tolist() == list(range(20))


def test_unsupported_schema_version(tmp_path):
    ntuple = _ntuple_with_attributes(
        os.path.join(tmp_path, "test.root"),
        {"FromTheFuture": ([(0, 20)], TAGS[:1]), "Calibration": ([(0, 20)], TAGS[:1])},
        {"FromTheFuture": (2, 0)},
    )

    # listing the attribute sets works; only using the unsupported one raises, and
    # the other attribute sets of the same RNTuple stay readable
    assert list(ntuple.attributes) == ["FromTheFuture", "Calibration"]
    assert ntuple.attributes["FromTheFuture"].schema_version == (2, 0)
    with pytest.raises(NotImplementedError, match="schema version"):
        ntuple.attributes["FromTheFuture"].keys()
    assert ntuple.attributes["Calibration"].keys() == ["tag"]


def test_unknown_minor_schema_version(tmp_path):
    _, attributes = _one_set(
        os.path.join(tmp_path, "test.root"), [(0, 20)], TAGS[:1], schema_version=(1, 7)
    )

    assert attributes.schema_version == (1, 7)
    assert attributes.keys() == ["tag"]


def test_invalid_attribute_set(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    data = ak.Array({"x": numpy.arange(5, dtype=numpy.int64)})
    with uproot.recreate(filepath) as file:
        file.mkrntuple("Events", data.layout.form).extend(data)

    file = uproot.open(filepath)
    ntuple = file["Events"]
    ntuple.footer.linked_attribute_sets = [
        _attribute_set_record(file.key("Events"), "NotAnAttributeSet")
    ]

    with pytest.raises(ValueError, match="_rangeStart"):
        ntuple.attributes["NotAnAttributeSet"].keys()


def test_attributes_are_lazy(tmp_path):
    ntuple = _ntuple_with_attributes(
        os.path.join(tmp_path, "test.root"), {"Calibration": ([(0, 20)], TAGS[:1])}
    )
    attributes = ntuple.attributes["Calibration"]

    # name and schema version come from the footer record, with nothing else read
    assert attributes.name == "Calibration"
    assert attributes.schema_version == (1, 0)
    assert attributes._ntuple is None

    assert attributes.keys() == ["tag"]
    assert attributes._ntuple is not None


def test_arrays_are_unaffected(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    ntuple, _ = _one_set(filepath, [(0, 20)], TAGS[:1])

    # attribute sets are not fields, so they stay out of the field interface
    assert ntuple.keys() == ["x"]
    assert ntuple.arrays().fields == ["x"]
    assert ntuple.arrays().x.tolist() == list(range(20))


def test_unpickle_without_attributes(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")
    data = ak.Array({"x": numpy.arange(5, dtype=numpy.int64)})
    with uproot.recreate(filepath) as file:
        file.mkrntuple("Events", data.layout.form).extend(data)

    ntuple = uproot.open(filepath)["Events"]
    # RNTuples pickled by older versions of Uproot have no _attributes
    del ntuple.__dict__["_attributes"]
    unpickled = pickle.loads(pickle.dumps(ntuple))

    assert unpickled.attributes == {}
    assert unpickled.cache_key == ntuple.cache_key
    assert unpickled.arrays().x.tolist() == list(range(5))


def test_ROOT(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    if ROOT.gROOT.GetVersionInt() < 64000:
        pytest.skip("ROOT version does not support RNTuple attributes")

    filepath = os.path.join(tmp_path, "test.root")

    assert ROOT.gInterpreter.Declare("""
        #include <ROOT/RNTupleAttrReading.hxx>
        #include <ROOT/RNTupleAttrWriting.hxx>
        #include <ROOT/RNTupleModel.hxx>
        #include <ROOT/RNTupleWriter.hxx>
        #include <TFile.h>
        #include <cstdint>
        #include <memory>
        #include <string>
        #include <utility>
        #include <vector>

        void uproot_write_rntuple_attributes(const char *path)
        {
            auto file = std::unique_ptr<TFile>(TFile::Open(path, "RECREATE"));

            auto model = ROOT::RNTupleModel::Create();
            auto pX = model->MakeField<std::int64_t>("x");
            auto writer = ROOT::RNTupleWriter::Append(std::move(model), "Events", *file);

            auto attrModel = ROOT::RNTupleModel::Create();
            auto pTag = attrModel->MakeField<std::string>("tag");
            auto attrs = writer->CreateAttributeSet(std::move(attrModel), "Calibration");

            std::int64_t next = 0;
            auto fill = [&](std::int64_t stop) {
                for (; next < stop; ++next) { *pX = next; writer->Fill(); }
            };

            // ranges are written when committed, so these end up out of start order
            auto outer = attrs->BeginRange();
            fill(5);
            auto inner = attrs->BeginRange();
            fill(8);
            *pTag = "inner";
            attrs->CommitRange(std::move(inner));
            auto empty = attrs->BeginRange();
            *pTag = "empty";
            attrs->CommitRange(std::move(empty));
            fill(10);
            *pTag = "outer";
            attrs->CommitRange(std::move(outer));

            writer.reset();
            file->Close();
        }

        std::vector<std::uint64_t> uproot_collect(ROOT::Experimental::RNTupleAttrEntryIterable indices)
        {
            std::vector<std::uint64_t> out;
            for (auto index : indices)
                out.push_back(index);
            return out;
        }
        """)
    ROOT.uproot_write_rntuple_attributes(filepath)

    reader = ROOT.RNTupleReader.Open("Events", filepath)
    expected = reader.OpenAttributeSet("Calibration")

    ntuple = uproot.open(filepath)["Events"]
    attributes = ntuple.attributes["Calibration"]

    assert attributes.schema_version == (1, 0)
    assert attributes.keys() == ["tag"]
    assert attributes.num_entries == expected.GetNEntries() == 3
    assert attributes.ranges.tolist() == [(5, 3), (8, 0), (0, 10)]
    assert attributes.arrays().tag.tolist() == ["inner", "empty", "outer"]
    assert ntuple.arrays().x.tolist() == list(range(10))

    assert attributes.get_attributes().tolist() == [2, 0, 1]
    assert attributes.get_attributes(6).tolist() == [2, 0]
    assert attributes.get_attributes_containing_range(4, 7).tolist() == [2]
    assert attributes.get_attributes_in_range(4, 10).tolist() == [0]

    def from_ROOT(indices):
        return list(ROOT.uproot_collect(indices))

    assert attributes.get_attributes().tolist() == from_ROOT(expected.GetAttributes())
    for entry in range(11):
        assert attributes.get_attributes(entry).tolist() == from_ROOT(
            expected.GetAttributes(entry)
        )
    # ROOT logs a warning for each empty or reversed range, so only check a few
    ranges = [(start, stop) for start in range(11) for stop in range(start + 1, 12)]
    for start, stop in [*ranges, (8, 8), (8, 3)]:
        assert attributes.get_attributes_containing_range(
            start, stop
        ).tolist() == from_ROOT(expected.GetAttributesContainingRange(start, stop))
        assert attributes.get_attributes_in_range(start, stop).tolist() == from_ROOT(
            expected.GetAttributesInRange(start, stop)
        )
