# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import datetime
import io
import struct

import numpy
import pytest
import skhep_testdata

import uproot
import uproot.reading
from uproot.reading import ReadOnlyDirectory, ReadOnlyFile, ReadOnlyKey

ZMUMU = "uproot-Zmumu.root"
NESTED = "uproot-nesteddirs.root"
HEADER = struct.Struct(">4siiiiiiiBiiiH16s")
KEY = struct.Struct(">ihiIhhii")
DIRECTORY = struct.Struct(">hIIiiiii")


def path_of(filename):
    return skhep_testdata.data_path(filename)


def header_of(filename):
    with open(path_of(filename), "rb") as file:
        return HEADER.unpack(file.read(HEADER.size))


def test_opendefaults_only_exists_to_be_unpickled():
    with pytest.raises(NotImplementedError):
        uproot.reading._OpenDefaults()


def test_file_header_properties():
    (
        _,
        fVersion,
        fBEGIN,
        fEND,
        fSeekFree,
        fNbytesFree,
        nfree,
        fNbytesName,
        fUnits,
        fCompress,
        fSeekInfo,
        fNbytesInfo,
        _,
        fUUID,
    ) = header_of(ZMUMU)

    with uproot.open(path_of(ZMUMU)) as directory:
        file = directory.file
        assert file.fVersion == fVersion
        assert file.fBEGIN == fBEGIN
        assert file.fEND == fEND
        assert file.fSeekFree == fSeekFree
        assert file.fNbytesFree == fNbytesFree
        assert file.nfree == nfree
        assert file.fNbytesName == fNbytesName
        assert file.fUnits == fUnits
        assert file.fCompress == fCompress
        assert file.fSeekInfo == fSeekInfo
        assert file.fNbytesInfo == fNbytesInfo
        assert file.fUUID == fUUID
        assert file.begin_chunk.start == 0


@pytest.mark.parametrize(
    "fVersion, version_tuple, version",
    [
        (61206, (6, 12, 6), "6.12/06"),
        (1061206, (6, 12, 6), "6.12/06"),  # with the marker of a 64-bit file
        (53434, (5, 34, 34), "5.34/34"),
    ],
)
def test_root_version(fVersion, version_tuple, version):
    with uproot.open(path_of(ZMUMU)) as directory:
        file = directory.file
        file._fVersion = fVersion
        assert file.root_version_tuple == version_tuple
        assert file.root_version == version


def test_begin_chunk_size_too_small():
    with pytest.raises(
        ValueError, match="begin_chunk_size=10 is not enough to read the TFile header"
    ):
        uproot.open(path_of(ZMUMU), begin_chunk_size=10)


def test_not_a_root_file(tmp_path):
    path = tmp_path / "not-root.root"
    path.write_bytes(b"text" * 200)
    with pytest.raises(
        ValueError, match="not a ROOT file: first four bytes are b'text'"
    ):
        uproot.open(path)


class ExecutorWithoutShutdown:
    def submit(self, task, /, *args, **kwargs):
        return uproot.source.futures.TrivialFuture(task(*args, **kwargs))


def test_executors_need_not_have_shutdown():
    file = ReadOnlyFile(
        path_of(ZMUMU),
        decompression_executor=ExecutorWithoutShutdown(),
        interpretation_executor=ExecutorWithoutShutdown(),
    )
    assert file.root_directory["events/Run"].array(library="np")[0] == 148031
    file.close()
    assert file.closed


def test_readonlyfile_is_a_context_manager():
    with ReadOnlyFile(path_of(ZMUMU)) as file:
        assert file.closed is False
    assert file.closed is True


def test_closed_file_refuses_to_read():
    file = ReadOnlyFile(path_of(ZMUMU))
    file.close()
    with pytest.raises(OSError, match="uproot-Zmumu.root' is closed"):
        file.chunk(0, 4)


@pytest.mark.parametrize(
    "attribute, value, message",
    [
        (
            "object_cache",
            "big",
            "object_cache must be None, a MutableMapping, or an int",
        ),
        ("array_cache", 1.5, "array_cache must be None, a MutableMapping, or a memory"),
        ("custom_classes", 3, "custom_classes must be None or a MutableMapping"),
        ("decompression_executor", 3, "decompression_executor must have a 'submit'"),
        ("interpretation_executor", 3, "interpretation_executor must have a 'submit'"),
    ],
)
def test_setters_reject_wrong_types(attribute, value, message):
    with ReadOnlyFile(path_of(ZMUMU)) as file:
        with pytest.raises(TypeError, match=message):
            setattr(file, attribute, value)


def test_object_cache_setter():
    with ReadOnlyFile(path_of(ZMUMU)) as file:
        file.object_cache = 3
        assert isinstance(file.object_cache, uproot.cache.LRUCache)
        assert file.object_cache.limit == 3
        file.array_cache = "1 kB"
        assert isinstance(file.array_cache, uproot.cache.LRUArrayCache)
        assert file.array_cache.limit == 1000
        file.array_cache = 2000
        assert file.array_cache.limit == 2000


def zero_seek_info(tmp_path):
    source = path_of(ZMUMU)
    with open(source, "rb") as file:
        data = bytearray(file.read())
    data[HEADER.size - 2 - 16 - 8 : HEADER.size - 2 - 16 - 4] = b"\x00\x00\x00\x00"
    out = tmp_path / "no-streamers.root"
    out.write_bytes(data)
    return out


def test_file_without_streamers(tmp_path):
    with ReadOnlyFile(zero_seek_info(tmp_path)) as file:
        assert file.fSeekInfo == 0
        assert file.streamers == {}
        assert file.streamer_rules == []


def test_streamer_rules_are_read_with_the_streamers():
    with ReadOnlyFile(path_of(ZMUMU)) as file:
        assert file._streamers is None
        assert any("fDefaultEntryOffsetLen" in x for x in file.streamer_rules)
        assert "TTree" in file._streamers


def counted(version, payload):
    """An object's serialized form: its number of bytes, version, and members."""
    return struct.pack(">IH", 0x40000000 | (len(payload) + 2), version) + payload


def file_with_streamer_payload(tmp_path, payload):
    """A file in which the streamers are ``payload``, in a new key at the end."""
    path = tmp_path / "streamers.root"
    with uproot.recreate(path, compression=None) as file:
        file["string"] = "text"

    data = bytearray(path.read_bytes())
    fields = list(HEADER.unpack(data[: HEADER.size]))
    assert fields[1] < 1000000  # a small file, whose header has 32-bit seek positions

    seek = len(data)
    # fClassName, fName, fTitle, as ROOT writes them for the streamers
    name = b"\x05TList\x0cStreamerInfo\x12Doubly linked list"
    keylen = KEY.size + len(name)
    data += KEY.pack(
        keylen + len(payload), 4, len(payload), 0, keylen, 1, seek, fields[2]
    )
    data += name + payload

    fields[3] = len(data)  # fEND
    fields[10] = seek  # fSeekInfo
    data[: HEADER.size] = HEADER.pack(*fields)
    path.write_bytes(data)
    return path


def test_unexpected_object_among_the_streamers(tmp_path):
    tobject = struct.pack(">hII", 1, 0, 0x03000000)
    tobjstring = counted(1, tobject + b"\x04text")
    item = struct.pack(">I", 0xFFFFFFFF) + b"TObjString\x00" + tobjstring
    item = struct.pack(">I", 0x40000000 | len(item)) + item
    tlist = counted(5, tobject + b"\x00" + struct.pack(">i", 1) + item + b"\x00")

    with ReadOnlyFile(file_with_streamer_payload(tmp_path, tlist)) as file:
        with pytest.raises(
            ValueError,
            match="unexpected type in TList of streamers and streamer rules: "
            "<class 'uproot.models.TObjString.Model_TObjString'>\nin file ",
        ):
            file.streamers  # noqa: B018 (the property raises)


@pytest.fixture(scope="module")
def versions_file(tmp_path_factory):
    """A file with TObject in its streamers, which everything depends on."""
    path = tmp_path_factory.mktemp("reading") / "versions.root"
    with uproot.recreate(path) as file:
        file["hist"] = numpy.histogram([1, 2, 3])
    return path


def test_streamer_named_by_version(versions_file):
    with ReadOnlyFile(versions_file) as file:
        versions = sorted(file.streamers["TH1"])
        assert file.streamer_named("TH1", "min").class_version == versions[0]
        assert file.streamer_named("TH1", "max").class_version == versions[-1]
        assert file.streamer_named("TH1", versions[0]).class_version == versions[0]
        assert file.streamer_named("TH1", 12345) is None
        assert file.streamer_named("NotThere") is None


class Streamer:
    """A stand-in for a TStreamerInfo with given dependencies."""

    def __init__(self, name, class_version, dependencies=()):
        self.name = name
        self.class_version = class_version
        self.dependencies = dependencies

    def _dependencies(self, streamers, out):
        out.append((self.name, self.class_version))
        out.extend(self.dependencies)


def test_streamer_dependencies_of_all_versions(versions_file):
    with ReadOnlyFile(versions_file) as file:
        file._streamers = {
            "A": {
                2: Streamer("A", 2, [("B", 2), ("C", 1)]),
                1: Streamer("A", 1, [("B", 1), ("C", 1)]),
            },
        }
        # newest first, versions of one class together, nothing repeated
        assert file.streamer_dependencies("A", version="all") == [
            ("C", 1),
            ("B", 2),
            ("B", 1),
            ("A", 2),
            ("A", 1),
        ]


def test_streamer_dependencies_of_a_missing_class(versions_file):
    with ReadOnlyFile(versions_file) as file:
        assert file.streamer_dependencies("NotThere") == []
        assert file.streamer_dependencies("NotThere", version="all") == []


def test_show_streamers_of_all_versions(versions_file):
    stream = io.StringIO()
    with ReadOnlyFile(versions_file) as file:
        file.show_streamers("TH1D", version="all", stream=stream)
    shown = stream.getvalue()
    assert "TH1D (v" in shown
    assert "TH1 (v" in shown
    assert "TNamed (v" in shown


def test_show_every_streamer(versions_file):
    stream = io.StringIO()
    with ReadOnlyFile(versions_file) as file:
        file.show_streamers(stream=stream)
        expected = [
            f"{name} (v{version})"
            for name, versions in file.streamers.items()
            for version in versions
        ]
    shown = stream.getvalue()
    # each one starts a paragraph, in the order of file.streamers
    headers = [x.split("\n")[0].split(":")[0] for x in shown.split("\n\n")]
    assert headers == expected
    assert not shown.startswith("\n")


def test_remove_class_definition_that_is_not_there():
    with ReadOnlyFile(path_of(ZMUMU)) as file:
        file.remove_class_definition("NotThere")
        assert file.custom_classes == dict(uproot.classes)

        file.remove_class_definition("TH1D")
        assert "TH1D" not in file.custom_classes
        assert "TH1D" in uproot.classes


def test_class_named_with_named_versions(versions_file):
    # this file has no streamer for TH2D, so it's up to the classes we know about
    cls = uproot.classes["TH2D"]
    known = sorted(cls.known_versions)
    with ReadOnlyFile(path_of(ZMUMU)) as file:
        assert file.class_named("TH2D", "min") is cls.class_of_version(known[0])
        assert file.class_named("TH2D", "max") is cls.class_of_version(known[-1])


def zero_seek_keys(tmp_path):
    with open(path_of(ZMUMU), "rb") as file:
        data = bytearray(file.read())

    fNbytesName = HEADER.unpack(data[: HEADER.size])[7]
    fBEGIN = HEADER.unpack(data[: HEADER.size])[2]
    offset = fBEGIN + fNbytesName + DIRECTORY.size - 4
    data[offset : offset + 4] = b"\x00\x00\x00\x00"

    out = tmp_path / "no-keys.root"
    out.write_bytes(data)
    return out


def test_directory_without_keys(tmp_path):
    with uproot.open(zero_seek_keys(tmp_path)) as directory:
        assert directory.fSeekKeys == 0
        assert directory.keys() == []
        assert len(directory) == 0


def test_keys_of_root_directory_in_the_begin_chunk():
    with uproot.open(path_of(ZMUMU), begin_chunk_size="10 MB") as directory:
        assert directory.keys() == ["events;1"]


def test_directory_properties():
    with uproot.open(path_of(ZMUMU)) as directory:
        fBEGIN, fNbytesName = directory.file.fBEGIN, directory.file.fNbytesName
        with open(directory.file_path, "rb") as raw:
            raw.seek(fBEGIN + fNbytesName)
            fields = DIRECTORY.unpack(raw.read(DIRECTORY.size))
        (
            fVersion,
            fDatimeC,
            fDatimeM,
            fNbytesKeys,
            fNbytesNamed,
            fSeekDir,
            fSeekParent,
            fSeekKeys,
        ) = fields

        assert repr(directory) == f"<ReadOnlyDirectory '/' at 0x{id(directory):012x}>"
        file = directory.file
        assert repr(file) == f"<ReadOnlyFile {file.file_path!r} at 0x{id(file):012x}>"
        assert directory.file_path == path_of(ZMUMU)
        assert directory.closed is False
        assert directory.cursor.index == fBEGIN + fNbytesName
        assert directory.parent is directory.file
        assert directory.fVersion == fVersion
        assert directory.fDatimeC == fDatimeC
        assert directory.fDatimeM == fDatimeM
        assert directory.fNbytesKeys == fNbytesKeys
        assert directory.fNbytesName == fNbytesNamed
        assert directory.fSeekDir == fSeekDir
        assert directory.fSeekParent == fSeekParent
        assert directory.fSeekKeys == fSeekKeys
        assert directory.created_on == uproot._util.code_to_datetime(fDatimeC)
        assert directory.modified_on == uproot._util.code_to_datetime(fDatimeM)
        assert isinstance(directory.created_on, datetime.datetime)
    assert directory.closed is True


@pytest.fixture(scope="module")
def nested_file(tmp_path_factory):
    """Directories inside directories, one object with two cycles, and a TTree."""
    path = tmp_path_factory.mktemp("reading") / "nested.root"
    with uproot.recreate(path) as file:
        file["top"] = "top string"
        file["a/hist"] = numpy.histogram([1, 2, 3])
        file["a/b/hist"] = numpy.histogram([1, 2, 3])
        file["a/b/twice"] = numpy.histogram([1, 2, 3])
        file["a/b/twice"] = numpy.histogram([4, 5, 6])
        tree = file.mktree("tree", {"x": "i8"}, title="the title")
        tree.extend({"x": numpy.array([1, 2, 3])})
        file.mkrntuple("ntuple", {"x": numpy.array([1, 2, 3])})
    return path


def test_mapping_interface(nested_file):
    with uproot.open(nested_file) as directory:
        expected = directory.keys()
        assert list(directory) == expected
        assert list(directory._ipython_key_completions_()) == expected
        assert directory.items(filter_classname="TH1*", recursive=False) == []
        items = directory.items(filter_classname="TH1D", cycle=False)
        assert [name for name, _ in items] == ["a/hist", "a/b/hist", "a/b/twice"]
        assert all(type(x).__name__.startswith("Model_TH1D") for _, x in items)


def test_recursive_filter_name_rejects_subdirectory_keys(nested_file):
    with uproot.open(nested_file) as directory:
        assert directory.keys(filter_name="a/b/hist", cycle=False) == ["a/b/hist"]


def test_recursive_names_without_cycle_appear_once(nested_file):
    with uproot.open(nested_file) as directory:
        assert directory.keys(filter_name="*twice", cycle=False) == ["a/b/twice"]
        assert directory.keys(filter_name="*twice", cycle=True) == [
            "a/b/twice;1",
            "a/b/twice;2",
        ]


def test_recursive_names_of_repeated_directory_appear_once(tmp_path):
    path = tmp_path / "repeated.root"
    with uproot.recreate(path) as file:
        file["a/string"] = "in a"
        file["b/string"] = "in b"

    # rename directory b to a in the list of keys: two directories have the same name
    with uproot.open(path) as directory:
        key = directory.key("b")
        position = key.cursor.index + KEY.size + len(key.fClassName) + 2
    data = bytearray(path.read_bytes())
    assert data[position - 1 : position + 1] == b"\x01b"
    data[position] = ord("a")
    path.write_bytes(data)

    with uproot.open(path) as directory:
        assert directory.keys(cycle=False) == ["a", "a/string"]


@pytest.mark.parametrize(
    "where, message",
    [
        ("top/x/y", "'x' is not a TDirectory"),
        ("a/hist/x/y", "'x' is not a TDirectory"),
    ],
)
def test_descent_into_something_else(nested_file, where, message):
    with uproot.open(nested_file) as directory:
        with pytest.raises(uproot.KeyInFileError, match=message) as excinfo:
            directory.key(where)
        assert excinfo.value.file_path == str(nested_file)


def test_title_classname_class_and_streamer_of(nested_file):
    with uproot.open(nested_file) as directory:
        assert directory.title_of("tree") == "the title"
        assert directory.title_of("a/hist") == directory["a/hist"].member("fTitle")

        assert directory.classname_of("tree") == "TTree"
        assert directory.classname_of("tree", encoded=True) == "Model_TTree"
        assert directory.classname_of("tree", encoded=True, version=20) == (
            "Model_TTree_v20"
        )
        assert directory.classname_of("a/hist") == "TH1D"

        assert directory.class_of("tree") is uproot.classes["TTree"]
        assert directory.class_of("a/hist") is uproot.classes["TH1D"]

        assert directory.streamer_of("tree").name == "TTree"
        assert directory.streamer_of("a/hist").name == "TH1D"


def test_title_classname_class_and_streamer_of_not_recursive(nested_file):
    with uproot.open(nested_file) as directory:
        subdirectory = directory["a"]
        assert subdirectory.title_of("hist", recursive=False) == (
            subdirectory["hist"].title
        )
        assert subdirectory.classname_of("hist", recursive=False) == "TH1D"
        assert subdirectory.class_of("hist", recursive=False) is uproot.classes["TH1D"]
        assert subdirectory.streamer_of("hist", recursive=False).name == "TH1D"


def test_missing_key_with_a_cycle(nested_file):
    with uproot.open(nested_file) as directory:
        with pytest.raises(
            uproot.KeyInFileError, match="not found: 'tree' with cycle 9"
        ):
            directory["tree;9"]


@pytest.mark.parametrize(
    "where, message",
    [
        ("a/hist:x", "'hist' is not a TDirectory, TTree, TBranch, RNTuple, or RField"),
        ("a/hist/x", "a/hist is not a TDirectory, TTree, TBranch, RNTuple, or RField"),
    ],
)
def test_getitem_descends_into_something_else(nested_file, where, message):
    with uproot.open(nested_file) as directory:
        with pytest.raises(uproot.KeyInFileError, match=message) as excinfo:
            directory[where]
        assert excinfo.value.key == where


@pytest.mark.parametrize("where", ["tree:x", "ntuple:x"])
def test_getitem_descends_with_colon(nested_file, where):
    with uproot.open(nested_file) as directory:
        assert directory[where].array().tolist() == [1, 2, 3]


def test_key_properties():
    with uproot.open(path_of(ZMUMU)) as directory:
        key = directory.key("events")
        with open(directory.file_path, "rb") as raw:
            raw.seek(key.cursor.index)
            fields = KEY.unpack(raw.read(KEY.size))
        (
            fNbytes,
            fVersion,
            fObjlen,
            fDatime,
            fKeylen,
            fCycle,
            fSeekKey,
            fSeekPdir,
        ) = fields

        assert key.fNbytes == fNbytes
        assert key.fVersion == fVersion
        assert key.fObjlen == fObjlen
        assert key.fDatime == fDatime
        assert key.fKeylen == fKeylen
        assert key.fCycle == fCycle
        assert key.fSeekKey == fSeekKey
        assert key.fSeekPdir == fSeekPdir

        assert key.file is directory.file
        assert key.title() == key.fTitle
        assert key.classname() == "TTree"
        assert key.classname(encoded=True) == "Model_TTree"
        assert key.classname(encoded=True, version=19) == "Model_TTree_v19"
        assert key.created_on == uproot._util.code_to_datetime(fDatime)
        assert repr(key) == (
            f"<ReadOnlyKey events;{fCycle}: TTree (seek pos {key.data_cursor.index}) "
            f"at 0x{id(key):012x}>"
        )


def test_key_without_strings():
    with uproot.open(path_of(ZMUMU)) as directory:
        named = directory.key("events")
        chunk = directory.file.chunk(named.cursor.index, named.cursor.index + KEY.size)
        key = ReadOnlyKey(chunk, named.cursor.copy(), {}, directory.file, directory)
        assert key.fName is None
        assert key.fClassName is None
        assert repr(key) == (
            f"<ReadOnlyKey (seek pos {key.data_cursor.index}) at 0x{id(key):012x}>"
        )


def test_object_cached_from_a_closed_file_is_not_used():
    cache = {}
    with uproot.open(path_of(ZMUMU), object_cache=cache) as directory:
        first = directory["events"]
        assert len(cache) == 1

    with uproot.open(path_of(ZMUMU), object_cache=cache) as directory:
        second = directory["events"]
        assert second is not first
        assert second.file.closed is False
        assert second["Run"].array(library="np")[0] == 148031
        assert list(cache.values()) == [second]


def histogram_with_corrupted_tlist(tmp_path):
    """
    A file in which the fFunctions of a TH1D claims the wrong number of bytes, which
    is a DeserializationError in the TList, nested in classes that aren't bootstrapped.
    """
    path = tmp_path / "corrupted.root"
    with uproot.recreate(path, compression=None) as file:
        file["hist"] = numpy.histogram([1, 2, 3])

    with uproot.open(path) as directory:
        start = directory.key("hist").data_cursor.index
        offset = start + directory["hist"].member("fFunctions").cursor.index

    data = bytearray(path.read_bytes())
    assert data[offset] == 0x40  # the mark of a num_bytes
    data[offset + 3] += 1
    path.write_bytes(data)
    return path


def test_deserialization_error_tries_the_files_own_streamers(tmp_path):
    with uproot.open(histogram_with_corrupted_tlist(tmp_path)) as directory:
        file = directory.file
        predefined = uproot.classes["TH1D"]
        assert file.custom_classes is None

        with pytest.raises(
            uproot.deserialization.DeserializationError, match="while reading"
        ):
            directory["hist"]

        # the classes that failed were replaced with ones made from the streamers,
        # but the ones that the streamers themselves need are still there
        assert file.custom_classes["TH1D"] is not predefined
        assert file.custom_classes["TList"] is uproot.classes["TList"]


def test_deserialization_error_with_the_files_own_streamers_has_no_retry(tmp_path):
    with uproot.open(histogram_with_corrupted_tlist(tmp_path)) as directory:
        with pytest.raises(
            uproot.deserialization.DeserializationError, match="while reading"
        ):
            directory["hist"]

        custom_classes = dict(directory.file.custom_classes)

        with pytest.raises(
            uproot.deserialization.DeserializationError, match="while reading"
        ):
            directory["hist"]
        # nothing was removed this time: there's nothing better to try
        assert directory.file.custom_classes == custom_classes
