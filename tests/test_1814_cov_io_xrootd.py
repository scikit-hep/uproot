# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
The first half tests the XRootD sources on a real, local server. The second
half uses a fake client to get the failures that a healthy server doesn't give.
"""

import pickle
import queue
import shutil
import types

import pytest
import skhep_testdata

import uproot
import uproot.source.xrootd
from uproot.source.xrootd import (
    MultithreadedXRootDSource,
    XRootDResource,
    XRootDSource,
    get_server_config,
)

ROOT_FILE = "uproot-Zmumu.root"
TIMEOUT = 300  # a busy machine shouldn't make a request expire


@pytest.fixture(scope="module")
def remote(xrootd_server):
    url, directory = xrootd_server
    shutil.copy(skhep_testdata.data_path(ROOT_FILE), directory)
    return f"{url}/{ROOT_FILE}"


@pytest.fixture(scope="module")
def expected():
    with open(skhep_testdata.data_path(ROOT_FILE), "rb") as file:
        return file.read()


def read(source, ranges):
    notifications = queue.Queue()
    chunks = source.chunks(ranges, notifications)
    out = [x.raw_data.tobytes() for x in chunks]
    # a chunk's data is ready before its notification is sent, so wait for them
    notified = [notifications.get(timeout=TIMEOUT) for _ in ranges]
    assert sorted(map(id, notified)) == sorted(map(id, chunks))
    assert notifications.empty()
    return out


class RecordVectorReads:
    def __init__(self, file):
        self._file = file
        self.sizes = []

    def vector_read(self, chunks, **kwargs):
        self.sizes.append(len(chunks))
        return self._file.vector_read(chunks=chunks, **kwargs)

    def __getattr__(self, name):
        return getattr(self._file, name)


@pytest.mark.parametrize("use_threads", [True, False])
def test_source_properties(remote, expected, use_threads):
    with XRootDSource(remote, timeout=TIMEOUT, use_threads=use_threads) as source:
        assert repr(source).startswith("<XRootDSource '...")
        assert source.timeout == TIMEOUT
        assert source.resource.timeout == TIMEOUT
        assert source.file is source.resource.file
        assert source.closed is False

        assert source.num_bytes == len(expected)
        assert source.num_bytes == len(expected)  # the second time from memory

        assert source.chunk(10, 20).raw_data.tobytes() == expected[10:20]

    assert source.closed is True


def test_no_ranges_send_no_vector_read(remote):
    with XRootDSource(remote, timeout=TIMEOUT, use_threads=False) as source:
        recorder = RecordVectorReads(source._resource._file)
        source._resource._file = recorder
        assert source.chunks([], queue.Queue()) == []
        assert source.num_requests == 0
        assert recorder.sizes == []
        source._resource._file = recorder._file


def test_short_url_repr(remote):
    with XRootDSource(remote, timeout=TIMEOUT, use_threads=False) as source:
        source._file_path = "short"
        assert repr(source) == f"<XRootDSource 'short' at 0x{id(source):012x}>"


def test_vector_read_requests_are_limited(remote, expected):
    ranges = [(0, 10), (100, 110), (200, 210), (300, 310), (400, 410)]
    with XRootDSource(
        remote, timeout=TIMEOUT, use_threads=False, max_num_elements=2
    ) as source:
        assert source._max_num_elements == 2
        recorder = RecordVectorReads(source._resource._file)
        source._resource._file = recorder
        assert read(source, ranges) == [expected[a:b] for a, b in ranges]
        assert recorder.sizes == [2, 2, 1]
        source._resource._file = recorder._file


@pytest.mark.parametrize("multiple", [1, 3])
def test_ranges_larger_than_the_element_size_are_split(remote, expected, multiple):
    size = 1000
    extra = 0 if multiple == 3 else 123  # with and without a remainder
    range_ = (50, 50 + multiple * size + extra)
    with XRootDSource(remote, timeout=TIMEOUT, use_threads=True) as source:
        source._max_element_size = size
        assert read(source, [range_]) == [expected[slice(*range_)]]


def test_pickled_source_still_reads(remote, expected):
    with XRootDSource(remote, timeout=TIMEOUT, use_threads=False) as source:
        with pickle.loads(pickle.dumps(source)) as clone:
            assert clone.chunk(0, 8).raw_data.tobytes() == expected[:8]


@pytest.mark.parametrize("use_threads", [True, False])
def test_multithreaded_source(remote, expected, use_threads):
    with MultithreadedXRootDSource(
        remote, timeout=TIMEOUT, num_workers=1, use_threads=use_threads
    ) as source:
        assert source.timeout == TIMEOUT
        assert source.num_bytes == len(expected)
        assert source.num_bytes == len(expected)

        with pickle.loads(pickle.dumps(source)) as clone:
            assert clone.chunk(0, 8).raw_data.tobytes() == expected[:8]


def test_missing_file(xrootd_server):
    url, _ = xrootd_server
    with pytest.raises(FileNotFoundError, match="does-not-exist.root"):
        XRootDResource(f"{url}/does-not-exist.root", TIMEOUT)


class Status:
    def __init__(self, error=False, code=0, message="message from the server"):
        self.error = error
        self.code = code
        self.message = message


OK = Status()
FAILED = Status(True, 300, "generic failure")
TIMED_OUT = Status(True, 206, "operation timed out")
NOT_FOUND = Status(True, 400, "no such file")


class FakeFile:
    """The parts of ``XRootD.client.File`` that the source uses."""

    properties = {"LastURL": "root://server//file", "DataServer": "root://server"}
    statuses = {}

    def __init__(self):
        self.closed = 0

    def open(self, path, timeout):
        return self.statuses.get("open", OK), None

    def close(self, timeout):
        self.closed += 1

    def get_property(self, name):
        return self.properties[name]

    def stat(self, timeout):
        return self.statuses.get("stat", OK), types.SimpleNamespace(size=100)

    def read(self, start, size, timeout):
        return self.statuses.get("read", OK), b"x" * size

    def vector_read(self, chunks, timeout, callback):
        return self.statuses.get("vector_read", OK)


class FakeURL:
    def __init__(self, url):
        self.protocol, _, rest = url.partition("://")
        self.hostid = rest.partition("/")[0]


@pytest.fixture
def client(monkeypatch):
    """A fake ``XRootD.client``."""

    class FileSystem:
        result = (OK, b"1024\n2097136\n")

        def __init__(self, url):
            self.url = url

        def query(self, code, arg):
            return FileSystem.result

    out = types.SimpleNamespace(
        File=FakeFile,
        URL=FakeURL,
        FileSystem=FileSystem,
        flags=types.SimpleNamespace(
            QueryCode=types.SimpleNamespace(CONFIG="config"),
        ),
    )
    monkeypatch.setattr(uproot.extras, "XRootD_client", lambda: out)
    return out


def with_properties(**properties):
    file = FakeFile()
    file.properties = properties
    return file


def test_server_config_is_asked_of_the_data_server(client):
    client.FileSystem.result = (OK, b"178956968\n1000\n")
    assert get_server_config(FakeFile()) == (178956968, 1000)


def test_server_config_of_local_file_is_the_default(client):
    file = with_properties(LastURL="file://localhost//data/file.root")
    assert get_server_config(file) == (1024, 2097136)


def test_server_config_without_a_data_server(client):
    file = with_properties(LastURL="root://server//file", DataServer="")
    with pytest.raises(NotImplementedError):
        get_server_config(file)


@pytest.mark.parametrize(
    "status, error", [(TIMED_OUT, TimeoutError), (FAILED, OSError)]
)
def test_server_config_query_fails(client, status, error):
    client.FileSystem.result = (status, None)
    with pytest.raises(error, match=status.message):
        get_server_config(FakeFile())


@pytest.mark.parametrize(
    "status, error, message",
    [
        (NOT_FOUND, FileNotFoundError, "no such file"),
        (TIMED_OUT, TimeoutError, "XRootD error: operation timed out in file f.root"),
        (FAILED, OSError, "XRootD error: generic failure in file f.root"),
    ],
)
def test_open_fails(client, monkeypatch, status, error, message):
    monkeypatch.setattr(FakeFile, "statuses", {"open": status})
    with pytest.raises(error, match=message):
        XRootDResource("f.root", 5)


def test_timeout_is_converted_for_the_client(client):
    assert XRootDResource("f.root", None)._xrd_timeout() == 0
    assert XRootDResource("f.root", 5.5)._xrd_timeout() == 5


def test_stat_fails(client, monkeypatch):
    resource = XRootDResource("f.root", 5)
    monkeypatch.setattr(FakeFile, "statuses", {"stat": FAILED})
    with pytest.raises(OSError, match="XRootD error: generic failure in file f.root"):
        resource.num_bytes  # noqa: B018 (the property raises)
    assert resource.file.closed == 1


def test_read_fails(client, monkeypatch):
    resource = XRootDResource("f.root", 5)
    monkeypatch.setattr(FakeFile, "statuses", {"read": TIMED_OUT})
    with pytest.raises(TimeoutError, match="operation timed out in file f.root"):
        resource.get(0, 10)


def test_callback_gives_error_to_all_futures(client):
    resource = XRootDResource("f.root", 5)
    results = {}
    futures = {
        (0, 10): XRootDResource.partfuture(results, 0, 10),
        (10, 20): XRootDResource.partfuture(results, 10, 20),
    }

    resource.callbacker(futures, results)(FAILED, None, None)

    for future in futures.values():
        with pytest.raises(OSError, match="generic failure in file f.root"):
            future.result(timeout=0)


def test_callback_fills_futures_and_results(client):
    resource = XRootDResource("f.root", 5)
    results = {}
    futures = {(4, 8): XRootDResource.partfuture(results, 4, 8)}
    response = types.SimpleNamespace(
        chunks=[types.SimpleNamespace(offset=4, length=4, buffer=b"abcd")]
    )

    resource.callbacker(futures, results)(OK, response, None)

    assert futures[4, 8].result(timeout=0) == b"abcd"


def test_vector_read_fails_when_submitted(client, monkeypatch):
    with XRootDSource("f.root", timeout=5, use_threads=False) as source:
        monkeypatch.setattr(FakeFile, "statuses", {"vector_read": FAILED})
        with pytest.raises(OSError, match="generic failure in file f.root"):
            source.chunks([(0, 10)], queue.Queue())


def test_trivial_resource_is_a_context_manager():
    with uproot.source.xrootd.trivial_resource() as resource:
        assert resource is None
