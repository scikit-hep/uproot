# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of the HTTP sources against a local server that answers exactly as each
test tells it to, including all of the ways in which a server can misbehave.
"""

import base64
import contextlib
import gc
import http.client
import http.server
import pickle
import queue
import re
import threading
import urllib.parse

import numpy
import pytest
import skhep_testdata

import uproot
from uproot.source.futures import ResourceFuture
from uproot.source.http import (
    _ResponseBuffer,
    basic_auth_headers,
    full_path,
    get_num_bytes,
    HTTPResource,
    HTTPSource,
    make_connection,
    MultithreadedHTTPSource,
)

DATA = bytes(range(256)) * 16
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
    handler.send_header("Content-Length", str(len(DATA)))
    handler.end_headers()


def respond_single_range(handler):
    (start, stop), *others = requested_ranges(handler)
    assert others == [], "this server doesn't do multipart"
    body = DATA[start:stop]
    handler.send_response(206)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(DATA)}")
    handler.end_headers()
    handler.wfile.write(body)


def respond_with_first_range(handler):
    """A server that handles a list of ranges by ignoring all but the first."""
    (start, stop), *_ = requested_ranges(handler)
    body = DATA[start:stop]
    handler.send_response(206)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(DATA)}")
    handler.end_headers()
    handler.wfile.write(body)


def part(start, stop, size=str(len(DATA)), header_before=True, content=None):
    """One part of a multipart/byteranges response."""
    lines = [f"--{BOUNDARY}"]
    if header_before:
        lines.append("Content-Type: application/octet-stream")
    lines.append(f"Content-Range: bytes {start}-{stop - 1}/{size}")
    if not header_before:
        lines.append("Content-Type: application/octet-stream")
    body = DATA[start:stop] if content is None else content
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


def test_make_connection_schemes():
    parse = urllib.parse.urlparse

    secure = make_connection(parse("https://example.com:8443/x"), 5)
    assert isinstance(secure, http.client.HTTPSConnection)
    assert (secure.host, secure.port, secure.timeout) == ("example.com", 8443, 5)

    plain = make_connection(parse("http://example.com/x"), None)
    assert type(plain) is http.client.HTTPConnection
    assert plain.timeout == 30

    with pytest.raises(ValueError, match="unrecognized URL scheme .*: ftp"):
        make_connection(parse("ftp://example.com/x"), None)


def test_full_path():
    parse = urllib.parse.urlparse
    assert full_path(parse("http://x/file.root")) == "/file.root"
    assert full_path(parse("http://x/file.root?a=1&b=2")) == "/file.root?a=1&b=2"


def test_basic_auth_headers():
    parse = urllib.parse.urlparse
    assert basic_auth_headers(parse("http://x/y")) == {}
    assert basic_auth_headers(parse("http://user@x/y")) == {}
    token = base64.b64encode(b"user:secret").decode()
    assert basic_auth_headers(parse("http://user:secret@x/y")) == {
        "Authorization": f"Basic {token}"
    }


def test_credentials_and_query_reach_the_server():
    with serve() as server:
        url = server.url.replace("http://", "http://user:secret@") + "?token=abc"
        with HTTPSource(url, use_threads=False) as source:
            assert source.num_bytes == len(DATA)
            assert source.chunk(0, 4).raw_data.tobytes() == DATA[:4]

    token = base64.b64encode(b"user:secret").decode()
    assert [x[:2] for x in server.requests] == [
        ("HEAD", "/file.root?token=abc"),
        ("GET", "/file.root?token=abc"),
    ]
    assert all(x[2]["Authorization"] == f"Basic {token}" for x in server.requests)


@pytest.mark.parametrize("cls", [HTTPSource, MultithreadedHTTPSource])
def test_num_bytes_is_asked_only_once(cls):
    options = {} if cls is HTTPSource else {"num_workers": 1}
    with serve() as server:
        with cls(server.url, use_threads=False, **options) as source:
            assert source.num_bytes == len(DATA)
            assert source.num_bytes == len(DATA)
    assert [x[0] for x in server.requests] == ["HEAD"]


def test_num_bytes_follows_redirect():
    with serve() as server:

        def head(handler):
            if handler.path == "/file.root":
                respond_status(301, location=server.url + ".moved")(handler)
            else:
                respond_content_length(handler)

        with serve(head=head) as redirecting:
            with HTTPSource(redirecting.url, use_threads=False) as source:
                assert source.num_bytes == len(DATA)

    assert [x[:2] for x in redirecting.requests] == [("HEAD", "/file.root")]
    assert [x[:2] for x in server.requests] == [("HEAD", "/file.root.moved")]


@pytest.mark.parametrize(
    "head, error, message",
    [
        (
            respond_status(302),
            http.client.HTTPException,
            "status 302 .redirect. without a 'location'\nfor URL ",
        ),
        (respond_status(404), FileNotFoundError, "HTTP.S. returned 404"),
        (
            respond_status(500),
            http.client.HTTPException,
            "HTTP response was 500, rather than 200, in attempt to get file size",
        ),
        (
            respond_status(200),
            http.client.HTTPException,
            "response headers did not include content-length",
        ),
        (
            respond_status(200, **{"Content-Length": "0"}),
            http.client.HTTPException,
            "response headers did not include content-length",
        ),
    ],
)
def test_num_bytes_errors(head, error, message):
    with serve(head=head) as server:
        parsed = urllib.parse.urlparse(server.url)
        with pytest.raises(error, match=message):
            get_num_bytes(server.url, parsed, 30)


def test_resource_properties():
    resource = HTTPResource("http://localhost/file.root", 5)
    assert resource.timeout == 5
    assert resource.parsed_url.path == "/file.root"
    assert resource.auth_headers == {}
    assert resource.__enter__() is resource
    assert resource.__exit__(None, None, None) is None


def test_chunk_follows_redirect(use_threads):
    with serve() as target:

        def get(handler):
            respond_status(307, Location=target.url)(handler)

        with serve(get=get) as server:
            source = MultithreadedHTTPSource(
                server.url, num_workers=1, use_threads=use_threads
            )
            with source:
                assert source.chunk(4, 8).raw_data.tobytes() == DATA[4:8]


def test_chunk_with_truncated_body(use_threads):
    def get(handler):
        handler.send_response(206)
        handler.send_header("Content-Length", "10")
        handler.end_headers()
        handler.wfile.write(DATA[:5])

    with serve(get=get) as server:
        source = MultithreadedHTTPSource(
            server.url, num_workers=1, use_threads=use_threads
        )
        with source:
            with pytest.raises(
                http.client.IncompleteRead, match="5 bytes read, 5 more expected"
            ):
                source.chunk(0, 10).raw_data  # noqa: B018 (the property raises)


@pytest.mark.parametrize("timeout", [None, 7])
def test_chunk_in_wasm_uses_requests(timeout, monkeypatch):
    requests = pytest.importorskip("requests")
    timeouts = []
    original = requests.get

    def get(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return original(*args, **kwargs)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(uproot._util, "wasm", True)

    with serve() as server:
        source = HTTPSource(server.url, use_threads=False, timeout=timeout)
        with source:
            assert source.chunk(4, 8).raw_data.tobytes() == DATA[4:8]

    assert timeouts == [30 if timeout is None else timeout]


def test_chunks_in_wasm_never_try_multipart(monkeypatch):
    pytest.importorskip("requests")
    monkeypatch.setattr(uproot._util, "wasm", True)

    with serve() as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, notifications = read_chunks(source, [(0, 4), (8, 16)])
            assert chunks[0].raw_data.tobytes() == DATA[:4]
            assert chunks[1].raw_data.tobytes() == DATA[8:16]
            assert source.fallback is not None

    # each range in its own request
    assert sorted(x[2]["Range"] for x in server.requests) == [
        "bytes=0-3",
        "bytes=8-15",
    ]
    assert notifications.qsize() == 2


@pytest.mark.parametrize("content_length", [True, False])
def test_multipart(content_length, use_threads):
    def get(handler):
        respond_multipart(handler, content_length=content_length)

    ranges = [(0, 10), (100, 150), (200, 201)]
    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=use_threads) as source:
            chunks, notifications = read_chunks(source, ranges)
            for chunk, (start, stop) in zip(chunks, ranges, strict=True):
                assert chunk.raw_data.tobytes() == DATA[start:stop]
            assert source.fallback is None
            assert source.num_requests == 1
            assert source.num_requested_chunks == 3
            assert source.num_requested_bytes == 61
            filled = [notifications.get(timeout=60) for _ in chunks]
            assert {id(x) for x in filled} == {id(x) for x in chunks}

    assert len(server.requests) == 1
    assert server.requests[0][2]["Range"] == "bytes=0-9, 100-149, 200-200"


@pytest.mark.parametrize(
    "http_max_header_bytes, num_requests",
    [(21786, 1), (70, 2)],
    ids=[
        "one-request",
        "split-in-two",
    ],  # https://github.com/scikit-hep/uproot5/pull/1146
)
def test_multipart_reads_a_root_file(http_max_header_bytes, num_requests):
    path = skhep_testdata.data_path("uproot-Zmumu.root")
    branches = ["px1", "py1", "pz1", "E1", "px2", "py2", "pz2", "E2"]
    with open(path, "rb") as file:
        data = file.read()

    def head(handler):
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()

    def get(handler):
        ranges = requested_ranges(handler)
        parts = [
            part(start, stop, size=str(len(data)), content=data[start:stop])
            for start, stop in ranges
        ]
        if len(ranges) == 1:
            start, stop = ranges[0]
            body = data[start:stop]
            handler.send_response(206)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        else:
            respond_multipart(handler, parts)

    with serve(get=get, head=head) as server:
        with uproot.open(
            server.url,
            handler=HTTPSource,
            use_threads=False,
            http_max_header_bytes=http_max_header_bytes,
        ) as file:
            assert file.file.options["http_max_header_bytes"] == http_max_header_bytes
            remote = file["events"].arrays(branches, library="np")
            assert file.file.source.fallback is None

    headers = [x[2]["Range"] for x in server.requests]
    assert sum(", " in x for x in headers) == num_requests
    assert all(len(x) <= http_max_header_bytes for x in headers)
    with uproot.open(path) as file:
        local = file["events"].arrays(branches, library="np")

    for branch in branches:
        assert numpy.array_equal(remote[branch], local[branch])


def test_multipart_content_range_is_not_the_last_part_header():
    def get(handler):
        parts = [part(*x, header_before=False) for x in requested_ranges(handler)]
        respond_multipart(handler, parts)

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
            assert chunks[1].raw_data.tobytes() == DATA[20:30]
            assert source.fallback is None


def test_multipart_content_range_with_unknown_size():
    def get(handler):
        parts = [part(*x, size="*") for x in requested_ranges(handler)]
        respond_multipart(handler, parts)

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
            assert chunks[0].raw_data.tobytes() == DATA[0:10]
            assert chunks[1].raw_data.tobytes() == DATA[20:30]
            assert source.fallback is None


def test_multipart_server_merges_ranges():
    # parts may cover more than one of the requested ranges, in any order
    def get(handler):
        respond_multipart(handler, [part(0, 20), part(30, 40)])

    ranges = [(10, 20), (0, 10), (30, 40)]
    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, notifications = read_chunks(source, ranges)
            assert [x.raw_data.tobytes() for x in chunks] == [
                DATA[10:20],
                DATA[0:10],
                DATA[30:40],
            ]
            assert source.fallback is None
            assert notifications.qsize() == 3


def test_multipart_redirect():
    with serve(get=respond_multipart) as target:

        def get(handler):
            respond_status(302, Location=target.url)(handler)

        with serve(get=get) as server:
            with HTTPSource(server.url, use_threads=False) as source:
                chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
                assert chunks[1].raw_data.tobytes() == DATA[20:30]

    assert target.requests[0][2]["Range"] == "bytes=0-9, 20-29"


def test_multipart_server_hangs_up(use_threads):
    def get(handler):
        handler.close_connection = True

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=use_threads) as source:
            chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
            for chunk in chunks:
                with pytest.raises(
                    http.client.RemoteDisconnected,
                    match="Remote end closed connection without response",
                ):
                    chunk.wait()


@pytest.mark.parametrize(
    "get",
    [respond_with_first_range, respond_status(200, **{"Content-Length": "0"})],
    ids=["ignores-other-ranges", "status-200"],
)
def test_no_multipart_falls_back(get, use_threads):
    answers = []

    def respond(handler):
        answers.append(handler.headers["Range"])
        if ", " in handler.headers["Range"]:
            get(handler)
        else:
            respond_single_range(handler)

    ranges = [(0, 10), (20, 30)]
    with serve(get=respond) as server:
        options = {"use_threads": use_threads, "num_fallback_workers": 1}
        with HTTPSource(server.url, **options) as source:
            chunks, notifications = read_chunks(source, ranges)
            assert [x.raw_data.tobytes() for x in chunks] == [DATA[:10], DATA[20:30]]
            assert {id(notifications.get(timeout=60)) for _ in chunks} == {
                id(x) for x in chunks
            }
            assert repr(source).endswith(f" with fallback at 0x{id(source):012x}>")
            assert isinstance(source.fallback, MultithreadedHTTPSource)

            # now we know better than to try
            before = len(answers)
            more, _ = read_chunks(source, [(40, 50), (60, 70)])
            assert [x.raw_data.tobytes() for x in more] == [DATA[40:50], DATA[60:70]]
            assert all(", " not in x for x in answers[before:])


def test_multipart_without_content_range_falls_back():
    # 206, not the size of any range, but nothing recognizable in the body
    def get(handler):
        if ", " in handler.headers["Range"]:
            handler.send_response(206)
            handler.end_headers()
            handler.wfile.write(b"this is not\r\nmultipart\r\n")
        else:
            respond_single_range(handler)

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, _ = read_chunks(source, [(0, 10), (20, 30)])
            assert [x.raw_data.tobytes() for x in chunks] == [DATA[:10], DATA[20:30]]
            assert source.fallback is not None


def test_multipart_that_ends_early_falls_back_for_everything():
    def get(handler):
        if ", " in handler.headers["Range"]:
            respond_multipart(handler, [part(0, 10)])
        else:
            respond_single_range(handler)

    with serve(get=get) as server:
        with HTTPSource(server.url, use_threads=False) as source:
            chunks, notifications = read_chunks(source, [(0, 10), (20, 30)])
            assert [x.raw_data.tobytes() for x in chunks] == [DATA[:10], DATA[20:30]]
            assert source.fallback is not None

            # the chunk that had been filled is not announced twice
            assert notifications.qsize() == 2


def test_requests_too_long_for_one_header_become_batches():
    ranges = [(0, 10), (20, 30), (40, 50), (60, 70)]
    with serve(get=respond_multipart) as server:
        options = {"use_threads": False, "http_max_header_bytes": 20}
        with HTTPSource(server.url, **options) as source:
            chunks, _ = read_chunks(source, ranges)
            assert [x.raw_data.tobytes() for x in chunks] == [
                DATA[start:stop] for start, stop in ranges
            ]

    assert [x[2]["Range"] for x in server.requests] == [
        "bytes=0-9, 20-29",
        "bytes=40-49, 60-69",
    ]


def test_every_batch_shares_the_same_fallback():
    def get(handler):
        if ", " in handler.headers["Range"]:
            respond_with_first_range(handler)
        else:
            respond_single_range(handler)

    ranges = [(0, 10), (20, 30), (40, 50), (60, 70)]
    with serve(get=get) as server:
        options = {"use_threads": False, "http_max_header_bytes": 20}
        with HTTPSource(server.url, **options) as source:
            created = []
            original = MultithreadedHTTPSource.__init__

            def init(self, *args, **kwargs):
                created.append(self)
                original(self, *args, **kwargs)

            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(MultithreadedHTTPSource, "__init__", init)
                chunks, _ = read_chunks(source, ranges)
                assert [x.raw_data.tobytes() for x in chunks] == [
                    DATA[start:stop] for start, stop in ranges
                ]

            assert created == [source.fallback]


@pytest.mark.parametrize(
    "path, shown",
    [
        ("http://x", "'http://x'"),
        ("http://localhost/file.root", "'.../file.root'"),
    ],
)
def test_source_repr(path, shown):
    with HTTPSource(path, use_threads=False) as source:
        assert source.executor is source._executor
        assert repr(source) == f"<HTTPSource {shown} at 0x{id(source):012x}>"


@pytest.mark.parametrize(
    "cls, options",
    [
        (HTTPSource, {"use_threads": True}),
        (HTTPSource, {"use_threads": False}),
        (MultithreadedHTTPSource, {"num_workers": 2, "use_threads": True}),
        (MultithreadedHTTPSource, {"num_workers": 1, "use_threads": False}),
    ],
)
def test_pickled_source_still_reads(cls, options):
    with serve() as server:
        with cls(server.url, **options) as source:
            with pickle.loads(pickle.dumps(source)) as clone:
                assert clone.file_path == source.file_path
                assert clone.chunk(4, 8).raw_data.tobytes() == DATA[4:8]
                assert clone.closed is False
            assert clone.closed is True


class Stream:
    """A response body with nothing but a ``read`` method."""

    def __init__(self, data):
        self.data = data

    def read(self, length):
        out, self.data = self.data[:length], self.data[length:]
        return out


def test_response_buffer_read_beyond_what_is_already_read():
    buffer = _ResponseBuffer(Stream(b"a\n" + b"x" * 2000))
    assert buffer.readline() == b"a\n"
    assert buffer.read(1030) == b"x" * 1030


class BareResponse:
    """What's left of a response in environments that have no ``readline`` on it."""

    def __init__(self, data):
        self._stream = Stream(data)
        self.read = self._stream.read


def test_handle_multipart_without_readline_on_the_response():
    resource = HTTPResource("http://localhost/file.root", None)
    ranges = [(0, 10), (20, 30)]
    results = {}
    futures = {}
    for start, stop in ranges:
        futures[start, stop] = HTTPResource.partfuture(results, start, stop)
    original = dict(futures)

    body = part(0, 10) + part(20, 30) + f"--{BOUNDARY}--\r\n".encode()
    resource.handle_multipart(None, futures, results, BareResponse(body), ranges)

    assert futures == {}
    assert original[0, 10].result(timeout=0) == DATA[:10]
    assert original[20, 30].result(timeout=0) == DATA[20:30]
    assert isinstance(original[0, 10], ResourceFuture)
