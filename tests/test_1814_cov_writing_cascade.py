# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import datetime
import struct
import sys
import types
import uuid

import awkward as ak
import numpy as np
import pytest

import uproot
import uproot.writing._cascade as cascade
from uproot.const import kStartBigFile

UUID1 = uuid.UUID(int=1)
UUID2 = uuid.UUID(int=2)


def string(text):
    return cascade.String(None, text)


def make_key(location=None, seek_location=None, parent_location=100, title="title"):
    return cascade.Key(
        location,
        10,
        5,
        string("TClass"),
        string("name"),
        string(title),
        1,
        parent_location,
        seek_location,
    )


@pytest.mark.parametrize("length", [0, 254, 255, 1000])
def test_string_round_trip(length):
    text = "x" * length
    original = cascade.String(10, text)
    serialized = original.serialize()
    assert len(serialized) == length + (1 if length < 255 else 5)

    deserialized, position = cascade.String.deserialize(serialized + b"junk", 10)
    assert deserialized.string == text
    assert position == 10 + len(serialized)


@pytest.mark.parametrize("big", [False, True])
def test_key_round_trip(big):
    location = kStartBigFile + 10 if big else 1000
    original = make_key(location=location, seek_location=location)
    serialized = original.serialize()
    assert original.big is big
    assert len(serialized) == original.num_bytes == (34 if big else 26) + 18

    deserialized = cascade.Key.deserialize(serialized, location, "")
    assert deserialized.location == location
    assert deserialized.big is big
    assert deserialized.parent_location == 100
    assert deserialized.title.string == "title"
    assert deserialized.uncompressed_bytes == 10
    assert deserialized.compressed_bytes == 5


def test_key_read_from_a_file_can_become_big():
    read = cascade.Key.deserialize(
        make_key(location=1000, seek_location=1000).serialize(), 1000, ""
    )
    assert read.big is False

    # the same thing as when FreeSegments moves up the file
    read.location = kStartBigFile + 10
    assert read.big is True
    assert read.num_bytes == 34 + 18


def test_forced_big_keys_stay_big():
    key = cascade.Key(
        1000,
        10,
        5,
        string("TTree"),
        string("name"),
        string("title"),
        1,
        100,
        1000,
        big=True,
    )
    key.location = 2000
    assert key.big is True


def test_unknown_locations():
    sink = None  # never reached

    with pytest.raises(RuntimeError, match="can't write object because location"):
        cascade.String(None, "text").write(sink)

    with pytest.raises(RuntimeError, match="can't serialize key because location"):
        make_key().serialize()

    with pytest.raises(RuntimeError, match="can't write object because location"):
        cascade.DirectoryData(None, 100, []).write(sink)


def leaves():
    header = cascade.FileHeader(1000, 500, 50, 0, 100, None, 200, 300, UUID1)
    dirheader = cascade.DirectoryHeader(200, 100, 58, 300, 100, 0, UUID1)
    now = dirheader.modified_on
    freedata = cascade.FreeSegmentsData(900, ((10, 20),), 1000)
    allocated = cascade.FreeSegmentsData(900, ((10, 20),), 1000)
    allocated._allocation = 22
    return [
        (make_key(1000, 1000), "location", 1000, 2000),
        (make_key(1000, 1000), "parent_location", 100, 200),
        (make_key(1000, 1000), "seek_location", 1000, 2000),
        (freedata, "slices", ((10, 20),), ((10, 30),)),
        (allocated, "allocation", 22, 32),
        (cascade.DirectoryData(300, 100, []), "allocation", 100, 200),
        (dirheader, "modified_on", now, now + datetime.timedelta(days=1)),
        (header, "compression", None, uproot.ZLIB(1)),
        (header, "uuid", UUID1, UUID2),
    ]


@pytest.mark.parametrize(
    "leaf, attribute, same, different",
    leaves(),
    ids=lambda x: type(x).__name__ if isinstance(x, cascade.CascadeLeaf) else "",
)
def test_only_changes_make_a_leaf_dirty(leaf, attribute, same, different):
    leaf._file_dirty = False

    setattr(leaf, attribute, same)
    assert not leaf._file_dirty
    assert getattr(leaf, attribute) == same

    setattr(leaf, attribute, different)
    assert leaf._file_dirty
    assert getattr(leaf, attribute) == different


def test_slice_bookkeeping():
    another_slice = cascade.FreeSegments._another_slice

    # merged with both neighbors, filling the gap between them
    assert another_slice(((10, 20), (30, 40), (50, 60)), 20, 30) == (
        (10, 40),
        (50, 60),
    )
    # grows one of them
    assert another_slice(((10, 20),), 20, 25) == ((10, 25),)
    assert another_slice(((10, 20),), 5, 10) == ((5, 20),)
    # or adds a new one, in order
    assert another_slice(((30, 40),), 10, 20) == ((10, 20), (30, 40))

    for start, stop in [(15, 25), (5, 15), (10, 20)]:
        with pytest.raises(RuntimeError, match="already marked as free"):
            another_slice(((10, 20),), start, stop)


def test_free_segments_past_the_boundary_are_big():
    small = (100, 200)
    big = (kStartBigFile + 100, kStartBigFile + 200)

    for slices, end, num_bytes in [
        ((small,), 1000, 20),
        ((small, big), 1000, 38),
        ((small,), kStartBigFile + 1000, 28),
        ((small, big), kStartBigFile + 1000, 46),
    ]:
        data = cascade.FreeSegmentsData(1000, slices, end)
        assert data.num_bytes == num_bytes
        serialized = data.serialize()
        assert len(serialized) == num_bytes
        read = cascade.FreeSegmentsData.deserialize(
            serialized, 1000, num_bytes, len(slices), ""
        )
        assert read.slices == slices
        assert read.end == end

    # the last segment is everything else, up to a power of two of the boundary
    data = cascade.FreeSegmentsData(1000, (), 3 * kStartBigFile)
    version, first, last = struct.unpack(">HQQ", data.serialize())
    assert (version, first, last) == (1001, 3 * kStartBigFile, 4 * kStartBigFile)


@pytest.mark.parametrize("big", [False, True])
def test_directory_header_round_trip(big):
    location = kStartBigFile + 100 if big else 100
    header = cascade.DirectoryHeader(
        location, location, 58, location + 200, 80, 0, UUID1
    )
    assert header.big is big
    assert header.uuid == UUID1
    assert header.num_bytes == (
        uproot.reading._directory_format_big.size
        if big
        else uproot.reading._directory_format_small.size
    )

    read = cascade.DirectoryHeader.deserialize(header.serialize(), location, "")
    assert read.begin_location == location
    assert read.data_location == location + 200
    assert read.data_num_bytes == 80
    assert read.begin_num_bytes == 58
    assert read.uuid == UUID1
    assert read.modified_on == header.modified_on.replace(microsecond=0)


@pytest.mark.parametrize("big", [False, True])
def test_file_header_round_trip(big):
    end = kStartBigFile + 1000 if big else 1000
    header = cascade.FileHeader(end, 500, 50, 2, 58, uproot.ZLIB(4), 200, 300, UUID1)
    assert header.big is big
    assert header.num_bytes == len(header.serialize())
    assert header.num_bytes == (
        uproot.reading._file_header_fields_big.size
        if big
        else uproot.reading._file_header_fields_small.size
    )

    read = cascade.FileHeader.deserialize(
        header.serialize().ljust(uproot.reading._file_header_fields_big.size, b"\0"),
        0,
    )
    assert read.end == end
    assert read.version == cascade.FileHeader.class_version
    assert read.free_location == 500
    assert read.free_num_slices == 2
    assert read.compression == uproot.ZLIB(4)
    assert read.info_num_bytes == 300
    assert read.uuid == UUID1


def streamer_info(name, version):
    out = uproot.streamers.Model_TStreamerInfo.empty()
    out._members["fName"] = name
    out._members["fClassVersion"] = version
    return out


def test_read_for_update_finds_streamers_by_version():
    streamers = [streamer_info("A", version) for version in (2, 3, 1)] + [
        streamer_info("B", 7),
        # the list of streamers can hold other things, such as schema evolution
        # rules, which aren't streamers even if they have a matching name
        types.SimpleNamespace(name="A", class_version=9),
    ]

    reader = cascade._ReadForUpdate("path", UUID1, tlist_of_streamers=streamers)
    assert [x.class_version for x in reader.streamers_named("A")] == [2, 3, 1]
    assert reader.streamers_named("C") == []
    assert reader.streamer_named("A").class_version == 3
    assert reader.streamer_named("A", "max").class_version == 3
    assert reader.streamer_named("A", "min").class_version == 1
    assert reader.streamer_named("A", 2).class_version == 2
    assert reader.streamer_named("A", 4) is None
    assert reader.streamer_named("C") is None
    assert reader.custom_classes is None

    reader = cascade._ReadForUpdate("path", UUID1)
    assert reader.streamers_named("A") == []
    assert reader.streamer_named("A") is None


def test_directory_data_keys():
    plain = make_key(1000, 1000)
    subdirectory = cascade.Key(
        2000,
        10,
        5,
        string("TDirectory"),
        string("sub"),
        string(""),
        1,
        100,
        2000,
    )
    data = cascade.DirectoryData(300, None, [plain])
    assert data.allocation == data.num_bytes == 4 + plain.num_bytes

    data._file_dirty = False
    data.add_key(subdirectory)
    assert data._file_dirty
    assert data._dirty_keys == []

    newer = cascade.Key(
        3000,
        10,
        5,
        string("TOther"),
        string("name"),
        string(""),
        2,
        100,
        3000,
    )
    data.add_key(newer, incremental=True)
    assert data._dirty_keys == [newer]

    assert data.key_names == ["name", "sub", "name"]
    assert data.dir_names == ["sub"]
    assert data.classname_of("name") == "TOther"
    assert data.classname_of("sub") == "TDirectory"
    assert data.classname_of("missing") is None
    assert data.get_key("name", 1) is plain
    assert data.get_key("name") is newer
    assert data.get_key("name", 5) is None
    assert data.get_key("missing", 1) is None


def test_raw_list_of_strings_can_be_moved():
    original = cascade.RawTListOfStrings(100, b"bytes")
    moved = original.copy_to(200)
    assert moved.location == 200
    assert moved.serialize() == b"bytes"
    assert repr(moved) == "RawTListOfStrings(200, b'bytes')"


def test_cascade_structure_of_a_file(tmp_path):
    path = tmp_path / "structure.root"
    with uproot.recreate(path, uuid_function=lambda: UUID1) as file:
        file["object"] = "text"
        subdirectory = file.mkdir("sub")
        casc = file.file._cascading
        root = casc.rootdirectory
        freesegments = casc.freesegments
        streamers = casc.streamers

        assert casc.tlist_of_streamers is None
        assert root.name.string == "structure.root"
        assert root.title.string == ""
        assert root.location == root.key.location == casc.fileheader.begin == 100
        assert root.datakey.location == root.header.data_location
        assert root.header.uuid == UUID1
        assert root.data.key_names == ["object", "sub"]
        assert root.data.dir_names == ["sub"]

        assert freesegments.fileheader is casc.fileheader
        assert freesegments.key.location == casc.fileheader.free_location
        assert freesegments.data.end == casc.fileheader.end

        assert streamers.key.name.string == "StreamerInfo"
        assert streamers.header.num_entries == len(streamers._rawstreamers) > 0
        assert streamers.header.data_bytes == streamers.num_bytes
        assert streamers.freesegments is freesegments

        sub = subdirectory._cascading
        assert sub.datakey is sub._datakey
        assert sub.parent is root
        assert (
            sub.location == sub.key.location == root.data.get_key("sub").seek_location
        )
        assert isinstance(sub, cascade.SubDirectory)

        file_representation = repr(casc)
        assert file_representation.startswith("CascadingFile(FileHeader(")
        for name in [
            "TListOfStreamers(",
            "TListHeader(",
            "RawStreamerInfo(",
            "RootDirectory(Key(",
            "FreeSegments(Key(",
            "FreeSegmentsData(",
            "DirectoryData(",
            "DirectoryHeader(",
            "String(",
        ]:
            assert name in file_representation
        assert repr(sub).startswith("SubDirectory(Key(")
        assert repr(root.name) == f"String({root.name.location}, 'structure.root')"


def test_streamers_can_wait_for_a_flush(tmp_path):
    path = tmp_path / "out.root"
    raw = cascade.RawStreamerInfo(None, b"unused", "TNotWritten", 1)
    with uproot.recreate(path) as file:
        sink = file.file.sink
        streamers = file.file._cascading.streamers
        before = sink.read(streamers.key.location, streamers.allocation)

        streamers.update_streamers(sink, [raw], flush=False)
        assert len(streamers._rawstreamers) == 1
        assert sink.read(streamers.key.location, streamers.allocation) == before

        streamers.write(sink)
        assert sink.read(streamers.key.location, streamers.allocation) != before


def test_directories_can_wait_for_a_flush(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file["object"] = "text"
        sink = file.file.sink
        root = file._cascading
        num_keys = uproot.reading._directory_format_num_keys

        subdirectory = root.add_directory(sink, "sub", 256, UUID1, flush=False)
        assert root.data.num_keys == 2
        assert num_keys.unpack(sink.read(root.data.location, 4)) == (1,)

        subdirectory.write(sink)
        root.write(sink)
        assert num_keys.unpack(sink.read(root.data.location, 4)) == (2,)

    with uproot.open(path) as file:
        assert file.keys() == ["object;1", "sub;1"]
        assert file["sub"].keys() == []


def test_streamer_of_another_type(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        streamer = types.SimpleNamespace(name="Bogus", class_version=1)
        with pytest.raises(
            TypeError,
            match=r"^streamers must be TStreamerInfo or RawStreamerInfo, not SimpleNamespace$",
        ):
            file.file.update_streamers([streamer])


def test_streamer_list_with_something_else(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file["object"] = "text"
        other = uproot.serialization.serialize_object_any(
            uproot.writing.identify.to_TObjString("not a streamer")
        )
        file.file.update_streamers(
            [cascade.RawStreamerInfo(None, other + b"\x00", "other", 1)]
        )

    with uproot.update(path) as file:
        names = [x.name for x in file.file._cascading.streamers._rawstreamers]
        assert "other" not in names
        assert "TObjString" in names
        file["more"] = "text"

    with uproot.open(path) as file:
        assert file["object"] == "text"
        assert file["more"] == "text"


def corrupt(path, position, data):
    with open(path, "r+b") as file:
        file.seek(position)
        file.write(data)


@pytest.mark.parametrize(
    "where, data, message",
    [
        (lambda casc: 0, b"tree", r"not a ROOT file: first four bytes are b'tree'"),
        (
            lambda casc: casc.freesegments.key.location + 4,
            b"\x00\x09",
            "can't read TKey version 9",
        ),
        (
            lambda casc: casc.freesegments.data.location,
            b"\x00\x09",
            "can't read TFree version 9",
        ),
        (
            lambda casc: casc.rootdirectory.header.location,
            b"\x00\x09",
            "can't read TDirectory version 9",
        ),
    ],
    ids=["magic", "key", "free", "directory"],
)
def test_update_rejects_what_it_cannot_write(tmp_path, where, data, message):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file["object"] = "text"
        position = where(file.file._cascading)

    corrupt(path, position, data)

    with pytest.raises(ValueError, match=message):
        uproot.update(path)


def test_update_rejects_a_short_file(tmp_path):
    path = tmp_path / "out.root"
    path.write_bytes(b"root")
    with pytest.raises(OSError, match="could not read"):
        uproot.update(path)


def test_file_name_length():
    fsspec = pytest.importorskip("fsspec")

    name = "x" * 256 + ".root"
    with pytest.raises(ValueError, match="must be less than 256 bytes"):
        uproot.recreate("memory://" + name)
    fsspec.filesystem("memory").rm("/" + name)


def test_merging_free_segments(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        for name in "abcde":
            file[name] = name * 100

        freesegments = file.file._cascading.freesegments
        spaces = {
            name: (
                file._cascading.data.get_key(name).seek_location,
                file._cascading.data.get_key(name).seek_location
                + file._cascading.data.get_key(name).num_bytes
                + file._cascading.data.get_key(name).compressed_bytes,
            )
            for name in "bcd"
        }
        original = freesegments.data.slices

        del file["b"]
        del file["d"]
        assert spaces["b"] in freesegments.data.slices
        assert spaces["d"] in freesegments.data.slices

        del file["c"]
        merged = (spaces["b"][0], spaces["d"][1])
        assert merged in freesegments.data.slices
        assert len(freesegments.data.slices) == len(original) + 1

    with uproot.open(path) as file:
        assert file.keys(cycle=False) == ["a", "e"]
        assert file["e"] == "e" * 100

    root_file = ROOT.TFile.Open(str(path))
    assert not root_file.IsZombie()
    assert [x.GetName() for x in root_file.GetListOfKeys()] == ["a", "e"]
    assert str(root_file.Get("a")) == "a" * 100
    root_file.Close()


def make_big(directory, num_bytes=kStartBigFile):
    # skip over two gigabytes of file instead of writing them, leaving a hole
    directory.file._cascading.freesegments.allocate(num_bytes)


skip_without_sparse_files = pytest.mark.skipif(
    sys.platform == "win32", reason="this needs a sparse file to be fast"
)


@skip_without_sparse_files
@pytest.mark.parametrize("data", [None, {"x": np.array([1.5, 2.5])}])
def test_rntuple_past_the_big_file_boundary(tmp_path, data):
    path = tmp_path / "big.root"
    with uproot.recreate(path, compression=None) as file:
        file["small"] = "before the boundary"
        make_big(file)
        # the anchor's key is written small, and so must be its copy in the
        # directory's list of keys, which readers take fKeylen from
        if data is None:
            file.mkrntuple("ntuple", {"x": "float64"})
        else:
            file.mkrntuple("ntuple", data)

    with uproot.open(path) as file:
        assert file["small"] == "before the boundary"
        ntuple = file["ntuple"]
        assert ntuple.num_entries == (0 if data is None else 2)
        assert ntuple.arrays()["x"].tolist() == ([] if data is None else [1.5, 2.5])


@skip_without_sparse_files
def test_objects_past_the_big_file_boundary(tmp_path):
    path = tmp_path / "big.root"
    entries = np.array([1.0, 2.0, 3.0])
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    with uproot.recreate(path, compression=None) as file:
        file["before"] = "before the boundary"
        file.mkdir("low")
        make_big(file)

        file["histogram"] = entries, edges
        file["replaced"] = "after the boundary"
        file["deleted"] = "after the boundary, and then deleted"
        file["low/inner"] = "in a directory that begins before the boundary"
        high = file.mkdir("high")
        high["inner"] = "in a directory after the boundary"

        tree = file.mktree("tree", {"x": "float64", "y": "var * int32"})
        tree.extend({"x": np.array([1.5, 2.5]), "y": ak.Array([[1], [2, 3]])})

        for i in range(30):
            file[f"padding{i}"] = "moves the directory's keys past the boundary"

        del file["deleted"]
        assert file.file.is_64bit
        assert file.file.fUnits == 8
        assert file.file._cascading.fileheader.end > kStartBigFile
        assert any(
            stop > kStartBigFile
            for _, stop in file.file._cascading.freesegments.data.slices
        )

    # most of the file is a hole
    assert path.stat().st_size > kStartBigFile

    with uproot.open(path) as file:
        assert file.file.fUnits == 8
        assert file["before"] == "before the boundary"
        assert file["histogram"].values().tolist() == entries.tolist()
        assert file["low/inner"] == "in a directory that begins before the boundary"
        assert file["high/inner"] == "in a directory after the boundary"
        assert file["tree/y"].array().tolist() == [[1], [2, 3]]
        assert "deleted" not in file

    with uproot.update(path) as file:
        assert file.file.is_64bit
        file["new"] = "added in update mode"
        del file["replaced"]
        file["high"]["new"] = "in the directory after the boundary"
        file["tree"].extend(
            {"x": np.array([3.5]), "y": ak.Array([[4, 5]]), "ny": np.array([2])}
        )

    with uproot.open(path) as file:
        assert file["new"] == "added in update mode"
        assert "replaced" not in file
        assert file["high/new"] == "in the directory after the boundary"
        assert file["tree/x"].array().tolist() == [1.5, 2.5, 3.5]
        assert file["tree/y"].array().tolist() == [[1], [2, 3], [4, 5]]

    # the checks above don't need ROOT, so they run without it
    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    assert not root_file.IsZombie()
    assert not root_file.TestBit(ROOT.TFile.kRecovered)
    assert root_file.GetEND() == path.stat().st_size
    assert root_file.GetSeekFree() > kStartBigFile
    assert str(root_file.Get("before")) == "before the boundary"
    assert str(root_file.Get("new")) == "added in update mode"
    assert str(root_file.Get("high/new")) == "in the directory after the boundary"
    assert root_file.Get("replaced") is None or not root_file.Get("replaced")
    hist = root_file.Get("histogram")
    assert [hist.GetBinContent(i + 1) for i in range(3)] == entries.tolist()
    tree = root_file.Get("tree")
    found = []
    for _ in tree:
        found.append((tree.x, tree.ny, [tree.y[i] for i in range(tree.ny)]))
    assert found == [(1.5, 1, [1]), (2.5, 2, [2, 3]), (3.5, 2, [4, 5])]
    root_file.Close()


@skip_without_sparse_files
def test_streamers_moving_past_the_boundary_get_a_big_key(tmp_path):
    path = tmp_path / "big.root"
    with uproot.recreate(path, compression=None) as file:
        file["small"] = "small"
        make_big(file)
        # a tree needs many streamers, which don't fit in their allocation
        tree = file.mktree("tree", {"x": "float64"})
        tree.extend({"x": np.array([1.5])})

        streamers = file.file._cascading.streamers
        assert streamers.key.location > kStartBigFile
        assert streamers.key.big
        assert streamers.num_bytes <= streamers.allocation
        end_of_streamers = (
            streamers.key.location + streamers.key.num_bytes + streamers.allocation
        )
        for key in (
            file._cascading.data.get_key("tree"),
            file.file._cascading.freesegments.key,
        ):
            location = key.location if key.seek_location is None else key.seek_location
            assert not streamers.key.location <= location < end_of_streamers

    with uproot.update(path) as file:
        file["tree"].extend({"x": np.array([2.5])})

    with uproot.open(path) as file:
        assert file["tree/x"].array().tolist() == [1.5, 2.5]
        assert file.file.streamer_named("TTree").class_version == 20


@skip_without_sparse_files
def test_keys_read_from_a_small_file_become_big(tmp_path):
    path = tmp_path / "big.root"
    with uproot.recreate(path) as file:
        file["small"] = "small"
        file.mkdir("sub")

    # past 2 GiB, where a key can't hold a pointer in its 4-byte fields
    beyond_32_bits = 2**31 + 1000
    with uproot.update(path) as file:
        make_big(file, beyond_32_bits)
        file["large"] = "large"
        for i in range(40):
            file[f"padding{i}"] = "moves the directory's keys past the boundary"
        file["sub"]["inner"] = "inner"
        file.mkdir("sub2")["inner"] = "inner"

    with uproot.open(path) as file:
        assert file["large"] == "large"
        assert file["padding39"] == "moves the directory's keys past the boundary"
        assert file["sub/inner"] == "inner"
        assert file["sub2/inner"] == "inner"

    # the checks above don't need ROOT, so they run without it
    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    assert not root_file.TestBit(ROOT.TFile.kRecovered)
    assert str(root_file.Get("large")) == "large"
    assert str(root_file.Get("sub/inner")) == "inner"
    root_file.Close()
