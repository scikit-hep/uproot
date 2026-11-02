# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import io
import pickle
import queue

import numpy
import pytest
import skhep_testdata

import uproot
from uproot.source.file import FileResource, MemmapSource, MultithreadedFileSource
from uproot.source.object import ObjectResource, ObjectSource

DATA = bytes(range(256))


@pytest.fixture
def path(tmp_path):
    out = tmp_path / "data.bin"
    out.write_bytes(DATA)
    return str(out)


@pytest.fixture
def memmap_fails(monkeypatch):
    """Filesystems and operating systems without memory-mapped files get here."""

    def memmap(*args, **kwargs):
        raise OSError("memory-mapped files are not supported")

    monkeypatch.setattr(numpy, "memmap", memmap)


def test_file_resource(path):
    with FileResource(path) as resource:
        assert resource.closed is False
        assert resource.file.name == path
        assert resource.get(1, 4) == DATA[1:4]
    assert resource.closed is True


def test_memmap_repr(path, monkeypatch):
    with MemmapSource(path, num_fallback_workers=1, use_threads=True) as source:
        assert repr(source) == (
            f"<MemmapSource '...{path[-10:]}' at 0x{id(source):012x}>"
        )
        assert isinstance(source.file, numpy.memmap)
        monkeypatch.setattr(source, "_file_path", "short")
        assert repr(source) == f"<MemmapSource 'short' at 0x{id(source):012x}>"


def test_memmap_closed(path):
    source = MemmapSource(path, num_fallback_workers=1, use_threads=True)
    with source:
        pass
    assert source.closed is True
    with pytest.raises(OSError, match="memmap is closed for file"):
        source.chunk(0, 4)
    with pytest.raises(OSError, match="memmap is closed for file"):
        source.chunks([(0, 4)], queue.Queue())


def test_memmap_without_context_manager_support(path):
    class OldMmap:
        """``mmap.mmap`` objects before they had ``__enter__`` and ``__exit__``"""

        def __init__(self, mmap):
            self._mmap = mmap

        def __getattr__(self, name):
            if name in ("__enter__", "__exit__"):
                raise AttributeError(name)
            return getattr(self._mmap, name)

    class Memmap:
        def __init__(self, mmap):
            self._mmap = OldMmap(mmap)

    source = MemmapSource(path, num_fallback_workers=1, use_threads=True)
    real = source._file
    source._file = Memmap(real._mmap)
    try:
        assert source.__enter__() is source
        assert source.closed is False
        source.__exit__(None, None, None)
        assert source.closed is True
    finally:
        source._file = real


def test_memmap_fallback(path, memmap_fails):
    with MemmapSource(path, num_fallback_workers=2, use_threads=True) as source:
        assert source.file is None
        assert isinstance(source.fallback, MultithreadedFileSource)
        assert source.closed is False
        assert source.num_bytes == len(DATA)
        assert repr(source).endswith(f" with fallback at 0x{id(source):012x}>")

        assert source.chunk(1, 4).raw_data.tobytes() == DATA[1:4]

        notifications = queue.Queue()
        chunks = source.chunks([(0, 2), (8, 16)], notifications)
        assert [x.raw_data.tobytes() for x in chunks] == [DATA[0:2], DATA[8:16]]

    assert source.closed is True


def test_memmap_fallback_reads_a_root_file(memmap_fails):
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.open(path, handler=MemmapSource) as file:
        assert isinstance(file.file.source.fallback, MultithreadedFileSource)
        assert file["events"].num_entries == 2304
        assert file["events/Run"].array(library="np", entry_stop=2).tolist() == [
            148031,
            148031,
        ]


def test_memmap_fallback_is_recreated_when_unpickled(path, memmap_fails):
    source = MemmapSource(path, num_fallback_workers=1, use_threads=True)
    with source:
        with pickle.loads(pickle.dumps(source)) as clone:
            assert clone.fallback is not source.fallback
            assert clone.chunk(1, 4).raw_data.tobytes() == DATA[1:4]


@pytest.mark.parametrize("use_threads", [True, False])
def test_multithreaded_file_source_pickle(path, use_threads):
    with MultithreadedFileSource(
        path, num_workers=1, use_threads=use_threads
    ) as source:
        with pickle.loads(pickle.dumps(source)) as clone:
            assert clone.file_path == path
            assert clone.num_bytes == len(DATA)
            assert clone.chunk(1, 4).raw_data.tobytes() == DATA[1:4]


def test_object_resource():
    obj = io.BytesIO(DATA)
    with ObjectResource(obj) as resource:
        assert resource.obj is obj
        assert resource.closed is False
        assert resource.get(1, 4) == DATA[1:4]
    assert resource.closed is True


def test_object_resource_of_minimal_object():
    class ReadSeek:
        def __init__(self):
            self.position = 0

        def seek(self, position):
            self.position = position

        def read(self, num_bytes):
            return DATA[self.position : self.position + num_bytes]

    with ObjectResource(ReadSeek()) as resource:
        assert resource.closed is False  # it has no such attribute to ask
        assert resource.get(1, 4) == DATA[1:4]


@pytest.mark.parametrize("use_threads", [True, False])
def test_object_source(use_threads):
    obj = io.BytesIO(DATA)
    with ObjectSource(obj, use_threads=use_threads) as source:
        assert source.closed is False
        assert source.file_path == repr(obj)
        assert source.chunk(1, 4).raw_data.tobytes() == DATA[1:4]
    assert source.closed is True


def test_open_object_without_threads():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with open(path, "rb") as obj:
        with uproot.open(obj, use_threads=False) as file:
            assert file["events"].num_entries == 2304
            assert file["events/Run"].array(library="np")[:2].tolist() == [
                148031,
                148031,
            ]
