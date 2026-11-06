# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for PR #1654: source/sink error paths, leaks, and cleanups."""

from __future__ import annotations

import os
import time

import pytest

import uproot
import uproot.sink.file
import uproot.source.chunk
import uproot.source.cursor
import uproot.source.futures
from uproot.source.coalesce import CoalesceConfig, RangeRequest, _coalesce


def test_future_result_timeout_raises():
    # A task that never completes: result(timeout=...) must raise TimeoutError
    future = uproot.source.futures.Future(lambda: None, ())
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        future.result(timeout=0.05)
    assert time.monotonic() - start < 5.0


def test_future_result_reraises_exception():
    def boom():
        raise ValueError("kaboom")

    future = uproot.source.futures.Future(boom, ())
    future._run()
    with pytest.raises(ValueError, match="kaboom"):
        future.result()


def test_threadpool_shutdown_joins_workers():
    executor = uproot.source.futures.ThreadPoolExecutor(max_workers=3)
    workers = list(executor.workers)
    executor.shutdown()
    for worker in workers:
        assert not worker.is_alive()


def test_filesink_close_is_permanent(tmp_path):
    path = os.path.join(tmp_path, "sink.bin")
    sink = uproot.sink.file.FileSink(path)
    assert sink.closed is False  # not opened yet, but not closed either
    sink.write(0, b"hello")
    sink.close()
    assert sink.closed is True
    # After close, operations must raise instead of silently reopening
    with pytest.raises(OSError, match="closed"):
        sink.write(0, b"world")
    with pytest.raises(OSError, match="closed"):
        sink.tell()
    with pytest.raises(OSError, match="closed"):
        sink.read(0, 1)
