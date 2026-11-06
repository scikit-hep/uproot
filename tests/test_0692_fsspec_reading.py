# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

from typing import BinaryIO
import http.server
import numpy as np
import skhep_testdata
import queue
import threading
import fsspec
import requests
import os
import shutil
import sys

import pytest

import uproot
import uproot.source.fsspec
import uproot.source.file
import uproot.source.xrootd

is_windows = sys.platform.startswith("win")

MUON_PX_0 = [-52.899456, 37.73778]


def check_met_pt(data):
    """The content of Events/MET_pt in uproot-issue121.root, in full."""
    assert len(data) == 40
    assert data[:3].tolist() == pytest.approx(
        [25.948482513427734, 40.51326370239258, 38.71651077270508]
    )
    assert data.sum() == pytest.approx(1253.3066)


@pytest.fixture
def xrootd_file(xrootd_server):
    remote_path, local_path = xrootd_server
    filename = "uproot-issue121.root"
    shutil.copy(skhep_testdata.data_path(filename), local_path)
    return os.path.join(remote_path, filename)


@pytest.mark.parametrize(
    "urlpath, source_class",
    [
        ("file.root", uproot.source.fsspec.FSSpecSource),
        ("s3://path/file.root", uproot.source.fsspec.FSSpecSource),
        (r"C:\path\file.root", uproot.source.fsspec.FSSpecSource),
        (r"file://C:\path\file.root", uproot.source.fsspec.FSSpecSource),
        ("root://file.root", uproot.source.fsspec.FSSpecSource),
        (BinaryIO(), uproot.source.object.ObjectSource),
    ],
)
def test_default_source(urlpath, source_class):
    assert uproot._util.file_path_to_source_class(
        urlpath, options=uproot.reading.open.defaults
    ) == (source_class, urlpath)


@pytest.mark.parametrize(
    "to_open, handler, message",
    [
        (
            "file.root",
            "invalid_handler",
            "'handler' is not a class object inheriting from Source: 'invalid_handler'",
        ),
        (
            BinaryIO(),
            uproot.source.fsspec.FSSpecSource,
            "'handler' is not ObjectSource for a file-like object",
        ),
    ],
)
def test_invalid_handler(to_open, handler, message):
    with pytest.raises(TypeError, match=message):
        uproot.open(to_open, handler=handler)


def test_open_fsspec_http(http_server):
    pytest.importorskip("aiohttp")

    url = f"{http_server}/uproot-issue121.root"
    with uproot.open(
        url,
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


@pytest.mark.network
def test_open_fsspec_github():
    try:
        with uproot.open(
            "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root"
        ) as f:
            data = f["Events/MET_pt"].array(library="np")
            check_met_pt(data)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            pytest.skip("GitHub API limit has been reached")
        else:
            raise e


def test_open_fsspec_local():
    local_path = skhep_testdata.data_path("uproot-issue121.root")

    with uproot.open(
        local_path,
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_open_fsspec_s3(handler, s3_server):
    bucket_url, storage_options = s3_server

    with uproot.open(
        f"{bucket_url}/uproot-HZZ-1.root:events",
        handler=handler,
        **storage_options,
    ) as f:
        data = f["Muon_Px"].array(library="np")
        assert len(data) == 2421
        assert data[0].tolist() == pytest.approx(MUON_PX_0)


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_open_fsspec_ssh(handler):
    pytest.importorskip("paramiko")
    import paramiko
    import getpass

    user = getpass.getuser()
    host = "localhost"
    port = 22

    # only test this if we can connect to the host (this will work in GitHub Actions)
    try:
        with paramiko.SSHClient() as client:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user)
    except (
        paramiko.ssh_exception.SSHException,
        paramiko.ssh_exception.NoValidConnectionsError,
    ) as e:
        pytest.skip(f"ssh connection to host failed: {e}")

    # cache the file
    local_path = skhep_testdata.data_path("uproot-issue121.root")

    uri = f"ssh://{user}@{host}:{port}{local_path}"
    with uproot.open(uri, handler=handler) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


@pytest.mark.xrootd
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        uproot.source.xrootd.XRootDSource,
        None,
    ],
)
def test_open_fsspec_xrootd(handler, xrootd_file):
    with uproot.open(xrootd_file, handler=handler) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.file.MemmapSource,
        uproot.source.file.MultithreadedFileSource,
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
@pytest.mark.skipif(
    is_windows, reason="Windows does not support colons (':') in filenames"
)
def test_issue_1054_filename_colons(handler, tmp_path):
    local_path = skhep_testdata.data_path("uproot-issue121.root")
    # copy, rather than rename, so the file stays in the shared skhep_testdata cache
    local_path_new = str(tmp_path / "file:with:colons.root")
    shutil.copyfile(local_path, local_path_new)
    with uproot.open(local_path_new, handler=handler) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)

    with uproot.open(local_path_new + ":Events", handler=handler) as tree:
        data = tree["MET_pt"].array(library="np")
        check_met_pt(data)

    with uproot.open(local_path_new + ":Events/MET_pt", handler=handler) as branch:
        data = branch.array(library="np")
        check_met_pt(data)


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.file.MemmapSource,
        uproot.source.file.MultithreadedFileSource,
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_issue_1054_object_path_split(handler):
    root_filename = "uproot-issue121.root"
    local_path = str(skhep_testdata.data_path(root_filename))
    with uproot.open(local_path, handler=handler) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)

    with uproot.open(local_path + ":Events", handler=handler) as tree:
        data = tree["MET_pt"].array(library="np")
        check_met_pt(data)

    with uproot.open(local_path + ":Events/MET_pt", handler=handler) as branch:
        data = branch.array(library="np")
        check_met_pt(data)


def test_fsspec_chunks(http_server):
    pytest.importorskip("aiohttp")

    url = f"{http_server}/uproot-issue121.root"

    notifications = queue.Queue()
    with uproot.source.fsspec.FSSpecSource(url) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))

        chunk_data_sum = {sum(map(int, chunk.raw_data)) for chunk in chunks}
        assert chunk_data_sum == {3967, 413, 10985}, "Chunk data does not match"


def test_fsspec_memory():
    # read the file into memory
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as f:
        contents = f.read()

    # create a memory filesystem
    fs = fsspec.filesystem(protocol="memory")
    fs.store.clear()
    file_path = "skhep_testdata/uproot-issue121.root"
    fs.touch(file_path)
    # write contents into memory filesystem
    with fs.open(file_path, "wb") as f:
        f.write(contents)

    # read from memory filesystem
    with uproot.open(f"memory://{file_path}") as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


def test_fsspec_tar(tmp_path):
    import tarfile
    import io

    filename = "uproot-issue121.root"
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as f:
        contents = f.read()

    filename_tar = os.path.join(tmp_path, filename + ".tar")
    with tarfile.open(filename_tar, mode="w") as tar:
        file_info = tarfile.TarInfo(name=filename)
        file_info.size = len(contents)
        tar.addfile(file_info, fileobj=io.BytesIO(contents))

    # open with fsspec
    with uproot.open(f"tar://{filename}::file://{filename_tar}") as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


def test_fsspec_zip(tmp_path):
    import zipfile

    filename = "uproot-issue121.root"
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as f:
        contents = f.read()

    filename_zip = os.path.join(tmp_path, filename + ".zip")
    with zipfile.ZipFile(filename_zip, mode="w") as zip_file:
        zip_file.writestr(filename, data=contents)

    # open with fsspec
    with uproot.open(
        f"zip://{filename}:Events/MET_pt::file://{filename_zip}"
    ) as branch:
        data = branch.array(library="np")
        check_met_pt(data)


@pytest.mark.xrootd
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        uproot.source.xrootd.XRootDSource,
        None,
    ],
)
def test_open_fsspec_xrootd_iterate_files(handler, xrootd_file, xrootd_server):
    remote_path, local_path = xrootd_server
    shutil.copy(
        skhep_testdata.data_path("uproot-issue121.root"), local_path + "/b.root"
    )

    iterator = uproot.iterate(
        files=[
            {xrootd_file: "Events"},
            {os.path.join(remote_path, "b.root"): "Events"},
        ],
        expressions=["MET_pt"],
        # MET_pt is in one TBasket: with smaller steps, the later steps of a file
        # would have nothing to request, which XRootD fails on until #1825
        step_size=40,
        handler=handler,
    )

    chunks = list(iterator)
    assert [len(data) for data in chunks] == [40, 40]
    check_met_pt(chunks[0]["MET_pt"].to_numpy())
    check_met_pt(chunks[1]["MET_pt"].to_numpy())


@pytest.mark.xrootd
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        uproot.source.xrootd.XRootDSource,
        None,
    ],
)
def test_open_fsspec_xrootd_iterate_tree(handler, xrootd_file):
    with uproot.open({xrootd_file: "Events"}, handler=handler) as f:
        # one step, for the reason in test_open_fsspec_xrootd_iterate_files
        chunks = list(f.iterate(["MET_pt"], step_size=40))

    assert [len(data) for data in chunks] == [40]
    check_met_pt(chunks[0]["MET_pt"].to_numpy())


# https://github.com/scikit-hep/uproot5/issues/1035
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.file.MemmapSource,
        uproot.source.file.MultithreadedFileSource,
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_issue_1035(handler):
    with uproot.open(skhep_testdata.data_path("uproot-issue-798.root")) as f:
        expected = f["CollectionTree"][
            "MuonSpectrometerTrackParticlesAuxDyn.truthParticleLink"
        ].array()
        expected = expected.tolist()

    with uproot.open(
        skhep_testdata.data_path("uproot-issue-798.root"),
        handler=handler,
        use_threads=True,
        num_workers=10,
    ) as f:
        for _ in range(25):  # intermittent failure
            tree = f["CollectionTree"]
            branch = tree["MuonSpectrometerTrackParticlesAuxDyn.truthParticleLink"]
            data = branch.array()
            assert len(data) == 40
            assert data.tolist() == expected


@pytest.mark.xrootd
@pytest.mark.parametrize(
    "filename",
    [
        lambda pattern: {pattern: "Events"},
        lambda pattern: f"{pattern}:Events",
    ],
    ids=["dict", "str"],
)
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_fsspec_globbing_xrootd(handler, filename, xrootd_server):
    remote_path, local_path = xrootd_server
    for i in range(2):
        shutil.copyfile(
            skhep_testdata.data_path("uproot-issue121.root"),
            os.path.join(local_path, f"globbed-{i}.root"),
        )

    iterator = uproot.iterate(
        filename(os.path.join(remote_path, "globbed-*.root")),
        ["MET_pt"],
        handler=handler,
    )

    arrays = [array for array in iterator]
    assert len(arrays) == 2
    for array in arrays:
        check_met_pt(array["MET_pt"].to_numpy())


@pytest.mark.xrootd
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_fsspec_globbing_xrootd_no_files(handler, xrootd_server):
    remote_path, _ = xrootd_server
    iterator = uproot.iterate(
        {os.path.join(remote_path, "*/ThisFileShouldNotExist.root"): "Events"},
        ["MET_pt"],
        handler=handler,
    )
    with pytest.raises(FileNotFoundError, match="ThisFileShouldNotExist.root"):
        arrays = [array for array in iterator]


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_fsspec_globbing_s3(handler, s3_server):
    bucket_url, storage_options = s3_server

    iterator = uproot.iterate(
        {f"{bucket_url}/uproot-HZZ-*.root": "events"},
        ["Muon_Px"],
        handler=handler,
        **storage_options,
    )

    # the bucket is populated by the s3_server fixture with exactly two matches
    arrays = [array for array in iterator]
    assert len(arrays) == 2
    for array in arrays:
        assert len(array) == 2421
        assert array["Muon_Px"][0].tolist() == pytest.approx(MUON_PX_0)


@pytest.mark.parametrize(
    "protocol_prefix",
    [
        "",
        "simplecache::",
    ],
)
def test_fsspec_cache_xrootd(protocol_prefix, xrootd_server, tmp_path):
    pytest.importorskip("XRootD")
    pytest.importorskip("fsspec_xrootd")

    remote_path, local_path = xrootd_server
    filename = "uproot-issue121.root"
    with open(skhep_testdata.data_path(filename), "rb") as f_read:
        with open(os.path.join(local_path, filename), "wb") as f_write:
            f_write.write(f_read.read())
    remote_file_path = os.path.join(remote_path, filename)  # starts with "root://"

    cache_path = str(tmp_path / "cache")
    with uproot.open(
        protocol_prefix + remote_file_path,
        simplecache={"cache_storage": cache_path},
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


@pytest.mark.parametrize(
    "protocol_prefix",  # http scheme is already included in the server fixture
    [
        "",
        "simplecache::",
    ],
)
def test_fsspec_cache_http(http_server, protocol_prefix):
    pytest.importorskip("aiohttp")

    url = f"{protocol_prefix}{http_server}/uproot-issue121.root"
    with uproot.open(
        url,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)


def test_fsspec_cache_http_directory(http_server, tmp_path):
    pytest.importorskip("aiohttp")

    cache_directory = str(tmp_path / "cache")
    url = f"simplecache::{http_server}/uproot-issue121.root"
    with uproot.open(
        url,
        simplecache={"cache_storage": cache_directory},
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        check_met_pt(data)

    assert len(os.listdir(cache_directory)) == 1


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_fsspec_globbing_http(handler):
    pytest.importorskip("aiohttp")

    class NoListing(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        do_HEAD = do_GET

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("localhost", 0), NoListing)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Globbing does not work with http filesystems: a pattern is looked up as a
    # file name, and a server without that file makes it a FileNotFoundError
    try:
        iterator = uproot.iterate(
            {
                f"http://localhost:{server.server_address[1]}/uproot-issue*.root": "Events"
            },
            ["MET_pt"],
            handler=handler,
        )
        with pytest.raises(FileNotFoundError, match="uproot-issue"):
            arrays = [array for array in iterator]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
