# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: ``uproot.recreate`` must truncate.

Before the move to fsspec, ``uproot.recreate`` opened the path with mode ``"w"``,
which truncated it. Afterwards it went through ``FileSink``, which only truncates
when the file does *not* already exist, so recreating over a larger file left
every byte beyond the new ``fEND`` in place.
"""

from __future__ import annotations

import io
import os

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFile

import uproot


@pytest.mark.parametrize("kind", ["str", "pathlib", "file-like"])
def test_recreate_truncates(tmp_path, kind):
    path = tmp_path / "file.root"
    path.write_bytes(b"\x00" * 100_000)

    if kind == "file-like":
        with open(path, "r+b") as file, uproot.recreate(file) as f:
            f["h"] = "hello"
    else:
        with uproot.recreate(str(path) if kind == "str" else path) as f:
            f["h"] = "hello"

    with uproot.open(path) as f:
        assert f.keys() == ["h;1"]
        assert path.stat().st_size == f.file.fEND


class FileLikeWithoutTruncate:
    def __init__(self):
        self.buffer = io.BytesIO()

    def __getattr__(self, name):
        if name in ("read", "write", "seek", "tell", "flush"):
            return getattr(self.buffer, name)
        raise AttributeError(name)


class RawWithUnsupportedTruncate(io.RawIOBase):
    # inherits io.IOBase.truncate, which raises io.UnsupportedOperation
    def __init__(self):
        self.buffer = io.BytesIO()

    def readable(self):
        return True

    def writable(self):
        return True

    def seekable(self):
        return True

    def readinto(self, b):
        return self.buffer.readinto(b)

    def write(self, b):
        return self.buffer.write(b)

    def seek(self, *args):
        return self.buffer.seek(*args)

    def tell(self):
        return self.buffer.tell()

    def close(self):
        pass


@pytest.mark.parametrize("cls", [FileLikeWithoutTruncate, RawWithUnsupportedTruncate])
def test_recreate_file_like_without_truncate(cls):
    file = cls()
    with uproot.recreate(file) as f:
        f["h"] = "hello"
    with uproot.open(io.BytesIO(file.buffer.getvalue())) as f:
        assert f.keys() == ["h;1"]


def test_recreate_keeps_file_if_it_cannot_be_opened(monkeypatch):
    # like S3 or XRootD, which can write whole files but not open them "r+b"
    fs = fsspec.filesystem("memory")
    original_open = type(fs)._open

    def _open(self, path, mode="rb", **kwargs):
        if mode == "r+b":
            raise NotImplementedError("File mode not supported")
        return original_open(self, path, mode, **kwargs)

    monkeypatch.setattr(type(fs), "_open", _open)
    fs.pipe("/test_1697/file.root", b"contents")
    try:
        with pytest.raises(NotImplementedError):
            uproot.recreate("memory://test_1697/file.root")
        assert fs.cat("/test_1697/file.root") == b"contents"
    finally:
        fs.rm("/test_1697", recursive=True)


def test_create_and_update_do_not_truncate(tmp_path):
    path = tmp_path / "subdir" / "file.root"
    with uproot.create(path) as f:
        f["h"] = "hello"
    contents = path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        uproot.create(path)
    # simplecache creates its local copy exclusively, and it is not cached yet
    with pytest.raises(FileExistsError):
        uproot.create("simplecache::" + path.as_uri())
    assert path.read_bytes() == contents

    # a file-like object has no path to check, so create writes into it as is
    other = tmp_path / "other.root"
    other.write_bytes(b"\x00" * 100_000)
    with open(other, "r+b") as file, uproot.create(file) as f:
        f["h"] = "hello"
    assert other.stat().st_size == 100_000

    with uproot.update(path) as f:
        f["h2"] = "world"
    with uproot.open(path) as f:
        assert f.keys() == ["h;1", "h2;1"]


@pytest.mark.skipif(
    uproot._util.parse_version(fsspec.__version__)
    < uproot._util.parse_version("2026.2.0"),
    reason="before fsspec 2026.2.0, simplecache keeps files opened 'r+b' in its cache",
)
def test_create_ignores_stale_cached_copy(tmp_path):
    # simplecache keeps its local copy of a file after the file itself is deleted
    path = tmp_path / "file.root"
    uri = "simplecache::" + path.as_uri()
    with uproot.create(uri) as f:
        f["h"] = "first" * 10_000
    path.unlink()

    with uproot.create(uri) as f:
        f["h"] = "second"
    with uproot.open(path) as f:
        assert f["h"] == "second"
        # the stale copy is longer, and none of it may be left behind
        assert path.stat().st_size == f.file.fEND

    # the destination exists again, so this is a conflict
    contents = path.read_bytes()
    with pytest.raises(FileExistsError):
        uproot.create(uri)
    assert path.read_bytes() == contents


def test_create_does_not_overwrite_concurrently_created_file(tmp_path, monkeypatch):
    # another writer creates the file after create checks that it does not exist,
    # and the check keeps missing it, as if it disappeared and reappeared
    path = tmp_path / "file.root"
    fs_class = type(fsspec.filesystem("file"))

    def exists(self, p, **kwargs):
        if not path.exists():
            path.write_bytes(b"competing contents")
        return False

    monkeypatch.setattr(fs_class, "exists", exists)
    with pytest.raises(FileExistsError):
        uproot.create(path)
    monkeypatch.undo()
    assert path.read_bytes() == b"competing contents"


@pytest.mark.parametrize("function", [uproot.create, uproot.recreate, uproot.update])
def test_invalid_option_leaves_file_intact(tmp_path, function):
    path = tmp_path / "file.root"
    with uproot.recreate(path) as f:
        f["h"] = "hello"
    contents = path.read_bytes()

    with pytest.raises(TypeError, match="compresion"):
        function(path, compresion=uproot.ZLIB(1))
    assert path.read_bytes() == contents


def test_recreate_truncates_xrootd(xrootd_server):
    # XRootD files can be opened "r+b", but their truncate is io.IOBase's
    remote_path, local_path = xrootd_server
    uri = remote_path + "/test_1697.root"
    path = os.path.join(local_path, "test_1697.root")
    with uproot.recreate(uri) as f:
        f["h"] = "first" * 10_000

    with uproot.recreate(uri) as f:
        f["h"] = "second"
    with uproot.open(path) as f:
        assert f["h"] == "second"
        assert os.path.getsize(path) == f.file.fEND


def test_recreate_truncates_by_path_if_file_objects_cannot(monkeypatch):
    # like XRootD, whose file objects have io.IOBase's truncate
    def truncate(self, size=None):
        raise io.UnsupportedOperation("truncate")

    monkeypatch.setattr(MemoryFile, "truncate", truncate)
    fs = fsspec.filesystem("memory")
    fs.pipe("/test_1697/file.root", b"\x00" * 100_000)
    try:
        with uproot.recreate("memory://test_1697/file.root") as f:
            f["h"] = "hello"
        with uproot.open(io.BytesIO(fs.cat("/test_1697/file.root"))) as f:
            assert f.keys() == ["h;1"]
            assert fs.size("/test_1697/file.root") == f.file.fEND
    finally:
        fs.rm("/test_1697", recursive=True)


def test_create_without_exclusive_mode(monkeypatch):
    # filesystems that do not support mode "xb" still check for the file first
    fs = fsspec.filesystem("memory")
    original_open = type(fs)._open

    def _open(self, path, mode="rb", **kwargs):
        if mode == "xb":
            raise ValueError("unsupported mode")
        return original_open(self, path, mode, **kwargs)

    monkeypatch.setattr(type(fs), "_open", _open)
    try:
        with uproot.create("memory://test_1697/file.root") as f:
            f["h"] = "hello"
        contents = fs.cat("/test_1697/file.root")
        with pytest.raises(FileExistsError):
            uproot.create("memory://test_1697/file.root")
        assert fs.cat("/test_1697/file.root") == contents
    finally:
        fs.rm("/test_1697", recursive=True)


def test_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        uproot.sink.file.FileSink(str(tmp_path / "file.root"), mode="w")
    assert not (tmp_path / "file.root").exists()
