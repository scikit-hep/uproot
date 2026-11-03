# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import shutil
import socket
import subprocess
import sys
import tempfile
import pytest
import threading
import contextlib
import skhep_testdata
from functools import partial
import os
import time

# The base http server does not support range requests. Watch https://github.com/python/cpython/issues/86809 for updates
from http.server import HTTPServer
from RangeHTTPServer import RangeRequestHandler

import uproot

_skhep_data_path = skhep_testdata.data_path


def _data_path(filename, raise_missing=True, cache_dir=None):
    """
    Like ``skhep_testdata.data_path``, but safe to call from several processes.

    skhep_testdata writes a downloaded file straight into the shared cache, so
    under pytest-xdist another worker can find the file half-written and read
    it. Download into a private directory instead, and move the finished file
    into the cache.
    """
    if cache_dir is None:
        cache_path = skhep_testdata.data.cache_path()
        target = cache_path / filename
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=cache_path) as tmp:
                path = _skhep_data_path(filename, raise_missing, tmp)
                if path.startswith(tmp) and os.path.isfile(path):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # not os.replace: on Windows, replacing a file that
                        # another worker is reading makes its next open fail
                        os.rename(path, target)
                    except OSError:
                        if not target.exists():
                            raise
    return _skhep_data_path(filename, raise_missing, cache_dir)


# conftest.py is imported before the test modules, so this also covers
# "from skhep_testdata import data_path"
skhep_testdata.data_path = _data_path


try:
    import graphed
except ImportError:
    graphed = None


@pytest.fixture(autouse=True)
def fresh_graphed_default():
    """Each test starts without graphed's default Session and leaves the previous one in place."""
    if not hasattr(graphed, "set_session"):
        yield
        return
    previous = graphed.set_session(None)
    yield
    graphed.set_session(previous)


@pytest.fixture(scope="function", autouse=False)
def reset_classes():
    # reset_classes reloads the model modules, which replaces their classes.
    # Put the originals back afterward: other tests under this process compare
    # against them or looked them up when they were collected.
    modules = {
        module: dict(vars(module))
        for name, module in list(sys.modules.items())
        if name == "uproot.streamers" or name.startswith("uproot.models.")
    }
    classes, unknown_classes = uproot.classes, uproot.unknown_classes
    try:
        uproot.model.reset_classes()
        yield
    finally:
        for module, contents in modules.items():
            vars(module).update(contents)
        uproot.classes, uproot.unknown_classes = classes, unknown_classes


@contextlib.contextmanager
def serve_http():
    # serve files from the skhep_testdata cache directory.
    # This directory is initially empty and files are downloaded on demand
    class Handler(RangeRequestHandler):
        def _cache_file(self, path: str):
            path = path.lstrip("/")
            if path in skhep_testdata.known_files:
                return skhep_testdata.data_path(path)
            else:
                raise FileNotFoundError(
                    f"File '{path}' not available in skhep_testdata"
                )

        def do_HEAD(self):
            self._cache_file(self.path)
            return super().do_HEAD()

        def do_GET(self):
            self._cache_file(self.path)
            return super().do_GET()

    try:
        # Older skhep_testdata (in Python 3.9 environments)
        cache_path = skhep_testdata.local_files._cache_path()
    except AttributeError:
        # Newer skhep_testdata
        cache_path = skhep_testdata.data.cache_path()

    server = HTTPServer(
        server_address=("localhost", 0),
        RequestHandlerClass=partial(
            Handler,
            directory=cache_path,
        ),
    )
    server.server_activate()

    def serve_forever(httpd=server):
        with httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=serve_forever, daemon=True)

    try:
        thread.start()
        address, port = server.server_address
        yield f"http://{address}:{port}"
    finally:
        # stop the server
        server.shutdown()
        thread.join()


@pytest.fixture(scope="module")
def http_server():
    with serve_http() as server_url:
        yield server_url


@pytest.fixture(scope="module")
def tests_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def s3_server():
    """
    A local (in-process) S3 server, so that the S3 tests don't depend on the
    network or on somebody else paying for the data transfer.

    Yields ``(bucket_url, storage_options)``, where the bucket already contains
    two copies of ``uproot-HZZ.root``, named ``uproot-HZZ-1.root`` and
    ``uproot-HZZ-2.root``.
    """
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")
    import s3fs

    if not hasattr(moto_server.ThreadedMotoServer, "get_host_and_port"):
        # On free-threaded Windows, moto[server] -> docker -> pywin32 has no
        # wheels, so the resolver falls back to a years-old moto
        pytest.skip("moto is too old to report which port it is listening on")

    bucket = "uproot-test"
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()

    try:
        _, port = server.get_host_and_port()
        storage_options = {
            # moto does not check these, but botocore insists on having them
            "key": "testing",
            "secret": "testing",
            "client_kwargs": {"endpoint_url": f"http://127.0.0.1:{port}"},
        }

        with pytest.MonkeyPatch.context() as monkeypatch:
            # botocore raises NoRegionError if it can't find a region anywhere
            monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

            # don't reuse (or poison) filesystem instances from other tests
            s3fs.S3FileSystem.clear_instance_cache()
            fs = s3fs.S3FileSystem(**storage_options)
            fs.mkdir(bucket)
            local_path = skhep_testdata.data_path("uproot-HZZ.root")
            for name in ("uproot-HZZ-1.root", "uproot-HZZ-2.root"):
                fs.put(local_path, f"{bucket}/{name}")

            yield f"s3://{bucket}", storage_options

            s3fs.S3FileSystem.clear_instance_cache()
    finally:
        server.stop()


def _wait_for_port(proc, port, timeout=60):
    """
    Waits until the server process ``proc`` accepts connections on ``port``,
    failing the test if the process dies or doesn't get there in ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        returncode = proc.poll()
        if returncode is not None:
            pytest.fail(
                f"xrootd server exited with code {returncode} before accepting "
                f"connections on port {port}"
            )
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            if time.monotonic() > deadline:
                pytest.fail(
                    f"xrootd server did not accept connections on port {port} "
                    f"within {timeout} seconds"
                )
            time.sleep(0.05)


@pytest.fixture(scope="module")
def xrootd_server(tmpdir_factory):
    pytest.importorskip("XRootD")
    pytest.importorskip("fsspec_xrootd")

    server_dir = tmpdir_factory.mktemp("server")
    temp_path = os.path.join(server_dir, "Folder")
    os.mkdir(temp_path)
    xrootd = shutil.which("xrootd")
    if xrootd is None:
        pytest.skip("xrootd server executable is not available on PATH")
    # not the default port, so that several pytest-xdist workers can each run one
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
    # nor the default admin path (/tmp/.xrootd), where concurrent servers collide
    # binding a unix socket; short, because the pytest tmpdir is too long for one
    admin_dir = tempfile.mkdtemp(prefix="xrd", dir="/tmp")
    config = os.path.join(admin_dir, "xrootd.cfg")
    with open(config, "w") as f:
        f.write(f"all.adminpath {admin_dir}\nall.pidpath {admin_dir}\n")
    proc = subprocess.Popen([xrootd, "-p", str(port), "-c", config, server_dir])
    try:
        _wait_for_port(proc, port)
        yield f"root://localhost:{port}/" + str(temp_path), temp_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(server_dir)
        shutil.rmtree(admin_dir, ignore_errors=True)
