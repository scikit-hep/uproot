# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pathlib
import queue
import socket
from io import StringIO

import numpy
import pytest
import skhep_testdata
import threading
import uproot

XROOTD_MAX_ELEMENT_SIZE = 2097136
XROOTD_CONTENT = bytes(range(256)) * (XROOTD_MAX_ELEMENT_SIZE // 256 + 2)
XROOTD_SOURCES = [
    (uproot.source.xrootd.MultithreadedXRootDSource, {}),
    (uproot.source.xrootd.XRootDSource, {"max_num_elements": None}),
]


def tobytes(x):
    if hasattr(x, "tobytes"):
        return x.tobytes()
    else:
        return x.tostring()


@pytest.fixture(scope="module")
def issue121_content():
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as file:
        return file.read()


@pytest.fixture
def closed_port_url():
    # Nothing listens on the port: connecting is refused (or times out on Windows).
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
    return f"http://localhost:{port}/does-not-exist"


@pytest.fixture(scope="module")
def xrootd_file(xrootd_server):
    url, directory = xrootd_server
    (pathlib.Path(directory) / "data.raw").write_bytes(XROOTD_CONTENT)
    return f"{url}/data.raw"


@pytest.fixture
def use_threads(request):
    if request.param:
        yield request.param
        return
    else:
        n_threads = threading.active_count()
        yield request.param
        assert threading.active_count() == n_threads


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
def test_file(use_threads, num_workers, tmp_path):
    filename = tmp_path / "tmp.raw"
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with uproot.source.file.MultithreadedFileSource(
        filename, num_workers=num_workers, use_threads=use_threads
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications,
        )
        assert [tobytes(chunk.raw_data) for chunk in chunks] == [
            b"******",
            b"    ",
            b"...",
            b"+++++++",
            b"!!!!!",
            b"@@@@@",
        ]

    assert source.num_bytes == 30


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
def test_file_fail(use_threads, num_workers, tmp_path):
    filename = tmp_path / "tmp.raw"

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        uproot.source.file.MultithreadedFileSource(
            tmp_path / f"{filename.name}-does-not-exist",
            num_workers=num_workers,
            use_threads=use_threads,
        )


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_memmap(use_threads, tmp_path):
    filename = tmp_path / "tmp.raw"

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with uproot.source.file.MemmapSource(
        filename, num_fallback_workers=1, use_threads=use_threads
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)], notifications
        )
        assert [tobytes(chunk.raw_data) for chunk in chunks] == [
            b"******",
            b"    ",
            b"...",
            b"+++++++",
            b"!!!!!",
            b"@@@@@",
        ]

        assert source.num_bytes == 30


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_memmap_fail(use_threads, tmp_path):
    filename = tmp_path / "tmp.raw"

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        with uproot.source.file.MemmapSource(
            tmp_path / f"{filename.name}-does-not-exist",
            num_fallback_workers=1,
            use_threads=use_threads,
        ):
            ...


def test_colons_and_ports():
    assert uproot._util.file_object_path_split("https://example.com:443") == (
        "https://example.com:443",
        None,
    )
    assert uproot._util.file_object_path_split("https://example.com:443/file.root") == (
        "https://example.com:443/file.root",
        None,
    )
    assert uproot._util.file_object_path_split(
        "https://example.com:443/file.root:object"
    ) == ("https://example.com:443/file.root", "object")


@pytest.mark.parametrize("use_threads", [True, False])
def test_http_size(http_server, issue121_content, use_threads):
    # The local server's URL has an explicit port.
    url = f"{http_server}/uproot-issue121.root"
    with uproot.source.http.HTTPSource(
        url,
        timeout=10,
        num_fallback_workers=1,
        use_threads=use_threads,
    ) as source:
        assert source.num_bytes == len(issue121_content)

    with uproot.source.http.MultithreadedHTTPSource(
        url,
        num_workers=1,
        timeout=10,
        use_threads=use_threads,
    ) as source:
        assert source.num_bytes == len(issue121_content)


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_http_fail(closed_port_url, use_threads):
    with uproot.source.http.HTTPSource(
        closed_port_url,
        timeout=0.1,
        num_fallback_workers=1,
        use_threads=use_threads,
    ) as source:
        with pytest.raises(OSError):
            notifications = queue.Queue()
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
            chunks[0].raw_data


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
def test_no_multipart(http_server, issue121_content, use_threads, num_workers):
    with uproot.source.http.MultithreadedHTTPSource(
        f"{http_server}/uproot-issue121.root",
        num_workers=num_workers,
        timeout=10,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert one == issue121_content[0:100]
        assert two == issue121_content[50:55]
        assert three == issue121_content[200:400]
        assert one[:4] == b"root"


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
def test_no_multipart_fail(closed_port_url, use_threads, num_workers):
    with uproot.source.http.MultithreadedHTTPSource(
        closed_port_url,
        num_workers=num_workers,
        timeout=0.1,
        use_threads=use_threads,
    ) as source:
        with pytest.raises(OSError):
            notifications = queue.Queue()
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
            chunks[0].raw_data


@pytest.mark.parametrize("use_threads, num_workers", [(True, 1), (True, 2), (False, 0)])
def test_fallback(http_server, issue121_content, use_threads, num_workers):
    # The local server doesn't do multipart GETs.
    url = f"{http_server}/uproot-issue121.root"
    with uproot.source.http.HTTPSource(
        url,
        timeout=10,
        num_fallback_workers=num_workers,
        use_threads=use_threads,
    ) as source:
        assert source.fallback is None
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert one == issue121_content[0:100]
        assert two == issue121_content[50:55]
        assert three == issue121_content[200:400]
        assert one[:4] == b"root"
        assert isinstance(source.fallback, uproot.source.http.MultithreadedHTTPSource)


@pytest.mark.xrootd
@pytest.mark.parametrize("cls, options", XROOTD_SOURCES)
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd(xrootd_file, cls, options, use_threads):
    with cls(
        xrootd_file, timeout=300, num_workers=1, use_threads=use_threads, **options
    ) as source:
        assert source.num_bytes == len(XROOTD_CONTENT)

        ranges = [(0, 100), (50, 55), (200, 400)]
        chunks = source.chunks(ranges, queue.Queue())
        assert [tobytes(chunk.raw_data) for chunk in chunks] == [
            XROOTD_CONTENT[start:stop] for start, stop in ranges
        ]

        # not representable by a single element of a vector read
        (chunk,) = source.chunks([(0, XROOTD_MAX_ELEMENT_SIZE + 1)], queue.Queue())
        assert tobytes(chunk.raw_data) == XROOTD_CONTENT[: XROOTD_MAX_ELEMENT_SIZE + 1]

        chunk = source.chunk(numpy.int64(0), numpy.int64(100))
        assert tobytes(chunk.raw_data) == XROOTD_CONTENT[:100]


@pytest.mark.xrootd
@pytest.mark.parametrize("cls, options", XROOTD_SOURCES)
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_fail(xrootd_server, cls, options, use_threads):
    url, _ = xrootd_server
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        cls(
            f"{url}/does-not-exist",
            timeout=300,
            num_workers=1,
            use_threads=use_threads,
            **options,
        )


def test_cursor_debug():
    data = numpy.concatenate(
        [
            numpy.array([123, 123, 123], "u1"),
            numpy.array(
                [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 101, 202, 303], ">f4"
            ).view("u1"),
            numpy.array([123, 123], "u1"),
        ]
    )
    future = uproot.source.futures.TrivialFuture(data)

    chunk = uproot.source.chunk.Chunk(None, 0, len(data), future)
    cursor = uproot.source.cursor.Cursor(0)

    output = StringIO()
    cursor.debug(chunk, offset=3, dtype=">f4", stream=output)
    assert (
        output.getvalue()
        == """--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
  {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                        1.1             2.2             3.3             4.4
    --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
    176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
    --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
            5.5             6.6             7.7             8.8             9.9
    --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
    202   0   0  67  74   0   0  67 151 128   0 123 123
    --- --- ---   C   J --- ---   C --- --- ---   {   {
          101.0           202.0           303.0
"""
    )

    output = StringIO()
    cursor.debug(chunk, stream=output)
    assert (
        output.getvalue()
        == """--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
  {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
--- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
202   0   0  67  74   0   0  67 151 128   0 123 123
--- --- ---   C   J --- ---   C --- --- ---   {   {
"""
    )
