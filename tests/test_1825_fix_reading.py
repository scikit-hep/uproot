# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import contextlib
import gc
import http.client
import http.server
import queue
import re
import shutil
import struct
import threading
import zlib

import numpy as np
import numpy
import pytest
import skhep_testdata

import uproot
import uproot.compression
import uproot.reading
import uproot.source.xrootd
from uproot.compression import ZLIB
from uproot.reading import ReadOnlyFile
from uproot.source.http import _ResponseBuffer, HTTPSource, MultithreadedHTTPSource
from uproot.source.xrootd import XRootDSource

ak = pytest.importorskip("awkward")


@pytest.fixture
def two_files(tmp_path):
    """``a.root`` has ``x`` (0 to 9) and ``y`` (10 times ``x``); ``b.root`` has only ``x``
    (10 to 14)."""
    a, b = str(tmp_path / "a.root"), str(tmp_path / "b.root")
    with uproot.recreate(a) as file:
        x = np.arange(10, dtype=np.int64)
        file.mktree("t", {"x": "int64", "y": "int64"}).extend({"x": x, "y": x * 10})
    with uproot.recreate(b) as file:
        x = np.arange(10, 15, dtype=np.int64)
        file.mktree("t", {"x": "int64"}).extend({"x": x})
    return a, b


@pytest.mark.parametrize("function", ["arrays", "iterate", "concatenate"])
@pytest.mark.parametrize("cut", [5, ["x > 1"], b"x > 1"])
def test_cut_of_the_wrong_type_is_an_error(two_files, function, cut):
    a, _ = two_files
    with (
        uproot.open(a) as file,
        pytest.raises(TypeError, match="cut must be None or a string"),
    ):
        if function == "arrays":
            file["t"].arrays(["x"], cut=cut)
        elif function == "iterate":
            list(file["t"].iterate(["x"], cut=cut))
        else:
            uproot.concatenate({a: "t"}, ["x"], cut=cut)


DATA = b"uproot compression test " * 200


def source():
    out = uproot.source.chunk.Source()
    out._file_path = "test.root"
    return out


def decompress(block, uncompressed_bytes):
    chunk = uproot.source.chunk.Chunk.wrap(
        source(), numpy.frombuffer(block, dtype=numpy.uint8)
    )
    cursor = uproot.source.cursor.Cursor(0)
    out = uproot.compression.decompress(
        chunk, cursor, {}, len(block), uncompressed_bytes
    )
    return out.raw_data.tobytes()


@pytest.mark.parametrize("library, module", [("isal", "isal"), ("deflate", "deflate")])
def test_zlib_alternative_libraries(library, module, monkeypatch):
    pytest.importorskip(module)
    # reading uses a _DecompressZLIB, and ZLIB objects look at ZLIB, where a test
    # that sets uproot.ZLIB.library directly leaves its value
    monkeypatch.setattr(uproot.compression._DecompressZLIB, "library", library)
    monkeypatch.setattr(ZLIB, "library", library)

    # what the library compresses, the standard library decompresses
    assert zlib.decompress(ZLIB(6).compress(DATA)) == DATA

    # and the other way around
    assert ZLIB(6).decompress(zlib.compress(DATA), len(DATA)) == DATA

    with pytest.raises(ValueError, match=f'ZLIB.library="{library}"'):
        ZLIB(0).compress(DATA)


@pytest.mark.parametrize("library, module", [("isal", "isal"), ("deflate", "deflate")])
def test_zlib_alternative_libraries_read_root_file(library, module, monkeypatch):
    pytest.importorskip(module)
    path = skhep_testdata.data_path("uproot-Zmumu-zlib.root")
    expected = uproot.open(path)["events/px1"].array(library="np")
    assert len(expected) == 2304
    assert expected[:3].tolist() == pytest.approx(
        [-41.1952876442, 35.1180497674, 35.1180497674]
    )

    # reading uses a _DecompressZLIB, and ZLIB objects look at ZLIB, where a test
    # that sets uproot.ZLIB.library directly leaves its value
    monkeypatch.setattr(uproot.compression._DecompressZLIB, "library", library)
    monkeypatch.setattr(ZLIB, "library", library)
    actual = uproot.open(path)["events/px1"].array(library="np")
    assert numpy.array_equal(actual, expected)


def test_multithreaded_source_repr_without_threads():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with uproot.source.file.MultithreadedFileSource(
        path, num_workers=2, use_threads=False
    ) as source:
        assert source.num_workers == 0
        assert repr(source).startswith(
            "<MultithreadedFileSource '...Zmumu.root' (0 workers) at 0x"
        )


def test_steps_within_an_already_read_basket_request_nothing():
    # px1 is a single basket, so after the first of the 24 steps, the rest come from
    # the basket cache and no more requests are made to the source
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as file:
        tree = file["events"]
        assert tree["px1"].num_baskets == 1
        source = file.file.source
        before = source.num_requests
        steps = list(tree.iterate(["px1"], step_size=100, library="np"))
        assert len(steps) == 24
        assert source.num_requests - before == 1
        assert np.array_equal(
            np.concatenate([x["px1"] for x in steps]), tree["px1"].array(library="np")
        )


HTTP_DATA = bytes(range(256)) * 16


BOUNDARY = "THE_BOUNDARY"


@contextlib.contextmanager
def serve(get=None, head=None):
    """
    Starts a server whose GET and HEAD requests are answered by the given
    functions, which default to single-range and content-length answers.

    Yields an object with the ``url`` of a file and the ``requests`` it received.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            server.requests.append(("GET", self.path, dict(self.headers.items())))
            (get or respond_single_range)(self)

        def do_HEAD(self):
            server.requests.append(("HEAD", self.path, dict(self.headers.items())))
            (head or respond_content_length)(self)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("localhost", 0), Handler)
    server.daemon_threads = True
    server.requests = []
    server.url = f"http://localhost:{server.server_address[1]}/file.root"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture(autouse=True)
def nothing_is_left_open():
    # a response left open is a ResourceWarning (hence an error) when the garbage
    # collector gets to it; make that happen here, not in somebody else's test
    yield
    gc.collect()


def requested_ranges(handler):
    return [
        (int(start), int(last) + 1)
        for start, last in re.findall(r"(\d+)-(\d+)", handler.headers["Range"])
    ]


def respond_content_length(handler):
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(HTTP_DATA)))
    handler.end_headers()


def respond_single_range(handler):
    (start, stop), *others = requested_ranges(handler)
    assert others == [], "this server doesn't do multipart"
    body = HTTP_DATA[start:stop]
    handler.send_response(206)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(HTTP_DATA)}")
    handler.end_headers()
    handler.wfile.write(body)


def part(start, stop, size=str(len(HTTP_DATA)), header_before=True, content=None):
    """One part of a multipart/byteranges response."""
    lines = [f"--{BOUNDARY}"]
    if header_before:
        lines.append("Content-Type: application/octet-stream")
    lines.append(f"Content-Range: bytes {start}-{stop - 1}/{size}")
    if not header_before:
        lines.append("Content-Type: application/octet-stream")
    body = HTTP_DATA[start:stop] if content is None else content
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body + b"\r\n"


def respond_multipart(handler, parts=None, content_length=True, status=206):
    if parts is None:
        parts = [part(*x) for x in requested_ranges(handler)]
    body = b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()
    handler.send_response(status)
    handler.send_header("Content-Type", f"multipart/byteranges; boundary={BOUNDARY}")
    if content_length:
        handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def respond_status(status, **headers):
    def respond(handler):
        handler.send_response(status)
        for key, value in headers.items():
            handler.send_header(key, value)
        handler.end_headers()

    return respond


@pytest.fixture(params=[True, False], ids=["threads", "no-threads"])
def use_threads(request):
    return request.param


def read_chunks(source, ranges):
    notifications = queue.Queue()
    chunks = source.chunks(ranges, notifications)
    return chunks, notifications


def test_no_ranges_send_no_request():
    with serve() as server:
        with HTTPSource(server.url, use_threads=False) as source:
            assert source.chunks([], queue.Queue()) == []
            assert source.num_requests == 0
    assert server.requests == []


@pytest.mark.parametrize(
    "get, error, message",
    [
        (respond_status(404), FileNotFoundError, "HTTP.S. returned 404"),
        (
            respond_status(302),
            http.client.HTTPException,
            "status 302 .redirect. without a 'location'\nfor URL ",
        ),
        (
            respond_status(200, **{"Content-Length": "0"}),
            http.client.HTTPException,
            "status 200, rather than 206 .range requests.",
        ),
        (
            respond_status(503, **{"Content-Length": "0"}),
            http.client.HTTPException,
            "status 503, rather than 206 .range requests.",
        ),
    ],
)
def test_chunk_errors(get, error, message, use_threads):
    with serve(get=get) as server:
        source = MultithreadedHTTPSource(
            server.url, num_workers=1, use_threads=use_threads
        )
        with source:
            with pytest.raises(error, match=message):
                source.chunk(0, 10).raw_data  # noqa: B018 (the property raises)


def test_step_within_baskets_already_read_makes_no_request():
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    with open(path, "rb") as file:
        data = file.read()

    def head(handler):
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()

    def get(handler):
        ranges = requested_ranges(handler)
        if len(ranges) == 1:
            start, stop = ranges[0]
            handler.send_response(206)
            handler.send_header("Content-Length", str(stop - start))
            handler.end_headers()
            handler.wfile.write(data[start:stop])
        else:
            respond_multipart(
                handler,
                [
                    part(start, stop, size=str(len(data)), content=data[start:stop])
                    for start, stop in ranges
                ],
            )

    with uproot.open(path) as file:
        expected = file["events/px1"].array(library="np")

    # each TBranch is one TBasket, read in the first step and kept for the others
    with serve(get=get, head=head) as server:
        with uproot.open(f"{server.url}:events", handler=HTTPSource) as tree:
            steps = list(tree.iterate(["px1", "E2"], step_size=1000, library="np"))
            assert tree.file.source.fallback is None

    assert [len(x["px1"]) for x in steps] == [1000, 1000, 304]
    assert numpy.array_equal(numpy.concatenate([x["px1"] for x in steps]), expected)
    assert sum(", " in x[2].get("Range", "") for x in server.requests) == 1


def test_multipart_server_sends_a_range_nobody_asked_for():
    def get(handler):
        respond_multipart(handler, [part(50, 60)])

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, notifications = read_chunks(source, [(0, 10), (20, 30)])
            for chunk in chunks:
                with pytest.raises(
                    http.client.HTTPException,
                    match=r"unrecognized byte range in headers of HTTP multipart: "
                    r"'50-59'\n\n    expecting: 0-9, 20-29\n\nfor URL ",
                ):
                    chunk.wait()

            # whoever waits on the notifications hears about the failure as well
            assert notifications.qsize() == 2


def test_multipart_part_is_truncated():
    def get(handler):
        body = part(0, 10)[: -len(b"\r\n") - 5]  # the last 5 bytes of data are lost
        handler.send_response(206)
        handler.send_header("Content-Type", "multipart/byteranges; boundary=x")
        handler.end_headers()
        handler.wfile.write(body)

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
            with pytest.raises(
                http.client.HTTPException,
                match="wrong chunk length 5 .expected 10. for byte range '0-9'",
            ):
                chunks[0].wait()


def test_multipart_redirect_without_location(use_threads):
    with serve(get=respond_status(302)) as server:
        with HTTPSource(server.url, use_threads=use_threads) as source:
            chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
            for chunk in chunks:
                with pytest.raises(
                    http.client.HTTPException,
                    match="status 302 .redirect. without a 'location'",
                ):
                    chunk.wait()


class Stream:
    """A response body with nothing but a ``read`` method."""

    def __init__(self, data):
        self.data = data

    def read(self, length):
        out, self.data = self.data[:length], self.data[length:]
        return out


@pytest.mark.timeout(60)  # a bug in readline can make it spin forever
def test_response_buffer_read_and_readline():
    lines = b"first\nsecond\n"
    num_x = 2000
    assert len(lines) < _ResponseBuffer.CHUNK < len(lines) + num_x
    buffer = _ResponseBuffer(Stream(lines + b"x" * num_x))
    assert buffer.readline() == b"first\n"
    assert buffer.readline() == b"second\n"

    # the rest of the first block read from the stream is already read
    x_already_read = _ResponseBuffer.CHUNK - len(lines)
    assert buffer.read(3) == b"xxx"
    assert len(buffer.already_read) == x_already_read - 3
    assert buffer.read(len(buffer.already_read)) == b"x" * (x_already_read - 3)
    assert buffer.already_read == b""
    assert buffer.read(10) == b"x" * 10

    assert buffer.read(10) == b"x" * 10
    # ends at the end of the stream
    assert buffer.readline() == b"x" * (num_x - x_already_read - 20)


@pytest.mark.timeout(60)
def test_response_buffer_readline_at_the_end_of_the_stream():
    buffer = _ResponseBuffer(Stream(b"no newline"))
    assert buffer.readline() == b"no newline"
    assert buffer.readline() == b""


ROOT_FILE = "uproot-Zmumu.root"


TIMEOUT = 300  # a busy machine shouldn't make a request expire


@pytest.fixture(scope="module")
def remote(xrootd_server):
    url, directory = xrootd_server
    shutil.copy(skhep_testdata.data_path(ROOT_FILE), directory)
    return f"{url}/{ROOT_FILE}"


class RecordVectorReads:
    def __init__(self, file):
        self._file = file
        self.sizes = []

    def vector_read(self, chunks, **kwargs):
        self.sizes.append(len(chunks))
        return self._file.vector_read(chunks=chunks, **kwargs)

    def __getattr__(self, name):
        return getattr(self._file, name)


def test_no_ranges_send_no_vector_read(remote):
    with XRootDSource(remote, timeout=TIMEOUT, use_threads=False) as source:
        recorder = RecordVectorReads(source._resource._file)
        source._resource._file = recorder
        assert source.chunks([], queue.Queue()) == []
        assert source.num_requests == 0
        assert recorder.sizes == []
        source._resource._file = recorder._file


ZMUMU = "uproot-Zmumu.root"


HEADER = struct.Struct(">4siiiiiiiBiiiH16s")


def path_of(filename):
    return skhep_testdata.data_path(filename)


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


@pytest.fixture(scope="module")
def versions_file(tmp_path_factory):
    """A file with TObject in its streamers, which everything depends on."""
    path = tmp_path_factory.mktemp("reading") / "versions.root"
    with uproot.recreate(path) as file:
        file["hist"] = numpy.histogram([1, 2, 3])
    return path


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


@pytest.mark.parametrize("where", ["tree:x", "ntuple:x"])
def test_getitem_descends_with_colon(nested_file, where):
    with uproot.open(nested_file) as directory:
        assert directory[where].array().tolist() == [1, 2, 3]
