# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import shutil
import socket

import pytest
import skhep_testdata

import uproot


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
    shutil.copy(skhep_testdata.data_path("uproot-issue121.root"), directory)
    return f"{url}/uproot-issue121.root"


def test_file(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    expected = [
        b"******",
        b"    ",
        b"...",
        b"+++++++",
        b"!!!!!",
        b"@@@@@",
    ]

    for num_workers in [1, 2]:
        with uproot.source.file.MultithreadedFileSource(
            filename, num_workers=num_workers, use_threads=True
        ) as source:
            for i, (start, stop) in enumerate(
                [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
            ):
                chunk = source.chunk(start, stop)
                assert tobytes(chunk.raw_data) == expected[i]

        with pytest.raises(FileNotFoundError, match="does-not-exist"):
            uproot.source.file.MultithreadedFileSource(
                filename + "-does-not-exist",
                num_workers=num_workers,
                use_threads=True,
            )


def test_memmap(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    expected = [
        b"******",
        b"    ",
        b"...",
        b"+++++++",
        b"!!!!!",
        b"@@@@@",
    ]

    with uproot.source.file.MemmapSource(
        filename, num_fallback_workers=1, use_threads=True
    ) as source:
        for i, (start, stop) in enumerate(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
        ):
            chunk = source.chunk(start, stop)
            assert tobytes(chunk.raw_data) == expected[i]

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        uproot.source.file.MemmapSource(
            filename + "-does-not-exist",
            num_fallback_workers=1,
            use_threads=True,
        )


@pytest.mark.parametrize(
    "num_workers",
    [1, 2],
)
def test_http(http_server, issue121_content, num_workers):
    url = f"{http_server}/uproot-issue121.root"
    with uproot.source.http.MultithreadedHTTPSource(
        url, num_workers=num_workers, timeout=300, use_threads=True
    ) as source:
        for start, stop in [(0, 100), (50, 55), (200, 400)]:
            chunk = source.chunk(start, stop)
            assert tobytes(chunk.raw_data) == issue121_content[start:stop]


@pytest.mark.parametrize(
    "num_workers",
    [1, 2],
)
def test_http_fail(closed_port_url, num_workers):
    with uproot.source.http.MultithreadedHTTPSource(
        closed_port_url,
        num_workers=num_workers,
        timeout=0.1,
        use_threads=True,
    ) as source:
        with pytest.raises(OSError):
            source.chunk(0, 100).raw_data


def test_http_multipart(http_server, issue121_content):
    url = f"{http_server}/uproot-issue121.root"
    with uproot.source.http.HTTPSource(
        url, timeout=300, num_fallback_workers=1, use_threads=True
    ) as source:
        for start, stop in [(0, 100), (50, 55), (200, 400)]:
            chunk = source.chunk(start, stop)
            assert tobytes(chunk.raw_data) == issue121_content[start:stop]


def test_http_multipart_fail(closed_port_url):
    with uproot.source.http.HTTPSource(
        closed_port_url,
        timeout=0.1,
        num_fallback_workers=1,
        use_threads=True,
    ) as source:
        with pytest.raises(OSError):
            tobytes(source.chunk(0, 100).raw_data)


@pytest.mark.xrootd
@pytest.mark.parametrize("num_workers", [1, 5])
def test_xrootd(xrootd_file, issue121_content, num_workers):
    with uproot.source.xrootd.MultithreadedXRootDSource(
        xrootd_file,
        num_workers=num_workers,
        timeout=300,
        use_threads=True,
    ) as source:
        for start, stop in [(0, 100), (50, 55), (200, 400)]:
            chunk = source.chunk(start, stop)
            assert tobytes(chunk.raw_data) == issue121_content[start:stop]


@pytest.mark.xrootd
def test_xrootd_vectorread(xrootd_file, issue121_content):
    with uproot.source.xrootd.XRootDSource(
        xrootd_file,
        timeout=300,
        max_num_elements=None,
        num_workers=1,
        use_threads=True,
    ) as source:
        for start, stop in [(0, 100), (50, 55), (200, 400)]:
            chunk = source.chunk(start, stop)
            assert tobytes(chunk.raw_data) == issue121_content[start:stop]
