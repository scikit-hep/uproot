# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
fsspec's cat_ranges returns exceptions in its result list by default. A read
must raise an OSError for that range, not a TypeError from slicing it.
"""

from __future__ import annotations

import asyncio
import errno
import queue

import fsspec
import pytest
import skhep_testdata
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.memory import MemoryFileSystem

import uproot
from uproot import CoalescedReadError
from uproot.source.coalesce import CoalesceConfig


class PayloadError(Exception):
    """Not an OSError, like aiohttp.ClientPayloadError."""


class ResetError(PayloadError, ConnectionResetError):
    """Like aiohttp.ClientConnectionResetError."""


class ThirdPartyOSError(PayloadError, OSError):
    """Like aiohttp.ClientOSError."""


DROPPED_CONNECTION = PayloadError("Response payload is not completed")


class FailingMemoryFileSystem(MemoryFileSystem):
    protocol = "failingmemory"
    error = None
    error_start = 0
    chained_error = False

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path.removeprefix(f"{cls.protocol}://"))

    def cat_file(self, path, *args, **kwargs):
        if self.chained_error:
            raise OSError("chained error") from ValueError("cause")
        return super().cat_file(path, *args, **kwargs)

    def cat_ranges(self, paths, starts, ends, **kwargs):
        result = super().cat_ranges(paths, starts, ends, **kwargs)
        if self.error is None:
            return result
        return [
            self.error if start >= self.error_start else item
            for start, item in zip(starts, result)
        ]


class FailingAsyncFileSystem(AsyncFileSystem):
    """Like HTTPFileSystem, inherits the _cat_ranges that returns exceptions."""

    protocol = "failingasync"
    data = b""
    error = None

    async def _info(self, path, **kwargs):
        return {"name": path, "size": len(self.data), "type": "file"}

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        if self.error is not None:
            raise self.error
        return self.data[start:end]


@pytest.fixture
def zmumu_data():
    with open(skhep_testdata.data_path("uproot-Zmumu.root"), "rb") as f:
        return f.read()


@pytest.fixture
def failing_file(zmumu_data):
    fsspec.register_implementation(
        FailingMemoryFileSystem.protocol, FailingMemoryFileSystem, clobber=True
    )
    path = "/test_1450/Zmumu.root"
    fs = FailingMemoryFileSystem()
    fs.pipe_file(path, zmumu_data)
    yield f"{FailingMemoryFileSystem.protocol}://{path}"
    FailingMemoryFileSystem.error = None
    FailingMemoryFileSystem.error_start = 0
    FailingMemoryFileSystem.chained_error = False
    fs.rm(path)


@pytest.fixture
def failing_async_file(zmumu_data):
    fsspec.register_implementation(
        FailingAsyncFileSystem.protocol, FailingAsyncFileSystem, clobber=True
    )
    FailingAsyncFileSystem.data = zmumu_data
    yield f"{FailingAsyncFileSystem.protocol}:///test_1450/Zmumu.root"
    FailingAsyncFileSystem.data = b""
    FailingAsyncFileSystem.error = None


def test_dropped_connection_raises_oserror(failing_file):
    with uproot.open(failing_file, handler=uproot.source.fsspec.FSSpecSource) as f:
        tree = f["events"]
        FailingMemoryFileSystem.error = DROPPED_CONNECTION
        with pytest.raises(CoalescedReadError, match="Zmumu.root") as excinfo:
            tree.arrays(["px1", "py1"], library="np")

    assert isinstance(excinfo.value, OSError)
    assert excinfo.value.__cause__ is DROPPED_CONNECTION


def test_dropped_connection_on_async_filesystem(failing_async_file):
    with uproot.open(
        failing_async_file, handler=uproot.source.fsspec.FSSpecSource
    ) as f:
        tree = f["events"]
        FailingAsyncFileSystem.error = DROPPED_CONNECTION
        with pytest.raises(CoalescedReadError, match="Zmumu.root") as excinfo:
            tree.arrays(["px1", "py1"], library="np")

    assert excinfo.value.__cause__ is DROPPED_CONNECTION


def test_only_failed_ranges_raise(failing_file):
    config = CoalesceConfig(max_range_gap=0)
    with uproot.source.fsspec.FSSpecSource(
        failing_file, coalesce_config=config
    ) as source:
        FailingMemoryFileSystem.error = DROPPED_CONNECTION
        FailingMemoryFileSystem.error_start = 1000
        good, bad = source.chunks([(0, 4), (1000, 1004)], queue.Queue())

        good.wait()
        assert good.raw_data.tobytes() == b"root"
        with pytest.raises(CoalescedReadError, match="bytes 1000:1004"):
            bad.wait()


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, TimeoutError])
def test_builtin_oserror_keeps_its_type(failing_file, error):
    with uproot.open(failing_file, handler=uproot.source.fsspec.FSSpecSource) as f:
        tree = f["events"]
        FailingMemoryFileSystem.error = original = error("original message")
        with pytest.raises(error, match="Zmumu.root") as excinfo:
            tree.arrays(["px1", "py1"], library="np")

    assert type(excinfo.value) is error
    assert excinfo.value.__cause__ is original


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (ResetError("reset"), ConnectionResetError),
        (ThirdPartyOSError(errno.ECONNRESET, "reset"), ConnectionResetError),
        (ThirdPartyOSError(errno.EIO, "failed"), OSError),
        (ThirdPartyOSError("failed"), OSError),
        (OSError("failed"), OSError),
        (asyncio.CancelledError("cancelled"), CoalescedReadError),
    ],
)
def test_error_type_of_other_exceptions(failing_file, original, expected):
    with uproot.open(failing_file, handler=uproot.source.fsspec.FSSpecSource) as f:
        tree = f["events"]
        FailingMemoryFileSystem.error = original
        with pytest.raises(expected, match="Zmumu.root") as excinfo:
            tree.arrays(["px1", "py1"], library="np")

    assert type(excinfo.value) is expected
    assert excinfo.value.__cause__ is original


def test_errno_and_filename_are_kept(failing_file):
    with uproot.open(failing_file, handler=uproot.source.fsspec.FSSpecSource) as f:
        tree = f["events"]
        FailingMemoryFileSystem.error = FileNotFoundError(
            errno.ENOENT, "No such file", "/some/file.root"
        )
        with pytest.raises(FileNotFoundError, match="Zmumu.root") as excinfo:
            tree.arrays(["px1", "py1"], library="np")

    assert excinfo.value.errno == errno.ENOENT
    assert excinfo.value.filename == "/some/file.root"


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize("allowed", [True, (OSError,), (PayloadError,)])
def test_dropped_connection_in_dask_report(failing_file, allowed, open_files):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=open_files,
        allow_read_errors_with_report=allowed,
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.error = DROPPED_CONNECTION
    _, creport = dask.compute(collection, report, scheduler="synchronous")

    assert creport[0].exception == "CoalescedReadError"
    assert "Zmumu.root" in creport[0].message
    assert "Response payload is not completed" in creport[0].message


@pytest.mark.parametrize("open_files", [True, False])
def test_builtin_oserror_in_dask_report(failing_file, open_files):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=open_files,
        allow_read_errors_with_report=(FileNotFoundError,),
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.error = FileNotFoundError("original message")
    _, creport = dask.compute(collection, report, scheduler="synchronous")

    assert creport[0].exception == "FileNotFoundError"


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize(
    "original", [ResetError("reset"), ThirdPartyOSError(errno.ECONNRESET, "reset")]
)
def test_builtin_error_matches_on_its_original_type(failing_file, original, open_files):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=open_files,
        allow_read_errors_with_report=(PayloadError,),
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.error = original
    _, creport = dask.compute(collection, report, scheduler="synchronous")

    assert creport[0].exception == "ConnectionResetError"
    assert repr(original) in creport[0].message


@pytest.mark.parametrize("open_files", [True, False])
def test_dropped_connection_not_allowed_in_dask_report(failing_file, open_files):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=open_files,
        allow_read_errors_with_report=(ValueError,),
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.error = DROPPED_CONNECTION
    with pytest.raises(OSError, match="Zmumu.root"):
        dask.compute(collection, report, scheduler="synchronous")


def test_only_coalesced_errors_match_on_their_cause(failing_file):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=False,
        allow_read_errors_with_report=(ValueError,),
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.chained_error = True
    with pytest.raises(OSError, match="chained error"):
        dask.compute(collection, report, scheduler="synchronous")
