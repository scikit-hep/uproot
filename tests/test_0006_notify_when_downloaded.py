# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import queue
import shutil

import pytest
import skhep_testdata

import uproot

CONTENT = b"******    ...+++++++!!!!!@@@@@"
RANGES = [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
REMOTE_RANGES = [(0, 100), (50, 55), (200, 400)]


def check_notifications(source, ranges, content):
    """
    Every chunk is put on the queue exactly once, with the bytes it should have.
    """
    notifications = queue.Queue()
    chunks = source.chunks(ranges, notifications=notifications)
    assert [(chunk.start, chunk.stop) for chunk in chunks] == ranges

    expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
    while len(expected) > 0:
        chunk = notifications.get(timeout=60)
        assert expected.pop((chunk.start, chunk.stop)) is chunk
        assert chunk.raw_data.tobytes() == content[chunk.start : chunk.stop]
    assert notifications.empty()


@pytest.fixture
def raw_file(tmp_path):
    filename = tmp_path / "tmp.raw"
    filename.write_bytes(CONTENT)
    return filename


@pytest.fixture(scope="module")
def issue121_content():
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as file:
        return file.read()


@pytest.fixture(scope="module")
def xrootd_file(xrootd_server):
    url, directory = xrootd_server
    shutil.copy(skhep_testdata.data_path("uproot-issue121.root"), directory)
    return f"{url}/uproot-issue121.root"


@pytest.mark.parametrize("num_workers", [1, 5])
def test_file(raw_file, num_workers):
    with uproot.source.file.MultithreadedFileSource(
        raw_file, num_workers=num_workers, use_threads=True
    ) as source:
        check_notifications(source, RANGES, CONTENT)


def test_memmap(raw_file):
    with uproot.source.file.MemmapSource(
        raw_file, num_fallback_workers=1, use_threads=True
    ) as source:
        check_notifications(source, RANGES, CONTENT)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_http(http_server, issue121_content, num_workers):
    with uproot.source.http.MultithreadedHTTPSource(
        f"{http_server}/uproot-issue121.root",
        timeout=300,
        num_workers=num_workers,
        use_threads=True,
    ) as source:
        check_notifications(source, REMOTE_RANGES, issue121_content)


@pytest.mark.parametrize("num_fallback_workers", [1, 5])
def test_http_fallback(http_server, issue121_content, num_fallback_workers):
    with uproot.source.http.HTTPSource(
        f"{http_server}/uproot-issue121.root",
        timeout=300,
        num_fallback_workers=num_fallback_workers,
        use_threads=True,
    ) as source:
        check_notifications(source, REMOTE_RANGES, issue121_content)
        assert source.fallback is not None


@pytest.mark.xrootd
@pytest.mark.parametrize("num_workers", [1, 5])
def test_xrootd(xrootd_file, issue121_content, num_workers):
    with uproot.source.xrootd.MultithreadedXRootDSource(
        xrootd_file, num_workers=num_workers, timeout=300, use_threads=True
    ) as source:
        check_notifications(source, REMOTE_RANGES, issue121_content)


@pytest.mark.xrootd
def test_xrootd_vectorread(xrootd_file, issue121_content):
    with uproot.source.xrootd.XRootDSource(
        xrootd_file,
        timeout=300,
        max_num_elements=None,
        num_workers=1,
        use_threads=True,
    ) as source:
        check_notifications(source, REMOTE_RANGES, issue121_content)
