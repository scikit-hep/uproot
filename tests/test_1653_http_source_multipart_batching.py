# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Regression tests for https://github.com/scikit-hep/uproot5/pull/1653
(part of https://github.com/scikit-hep/uproot5/issues/1646).

Covers:
- ``HTTPSource.chunks`` range-batching: every batch's ``Range`` header must
  list exactly the ranges in that batch, and the batches together must cover
  every requested range exactly once (the old code used a ``ranges[j:j+i]``
  slice that built a superset on the 2nd+ overflow and dropped the final batch
  when the overflow fell on the last range).
- ``HTTPSource.__exit__`` must mark the source closed so a later ``chunk()``
  raises ``OSError`` instead of hanging on a dead worker pool.
- ``HTTPResource.next_header`` must consume all part headers up to the blank
  line, so a ``Content-Range`` that is not the last header does not shift the
  part payload.
"""

import io
import queue

import pytest

import uproot
import uproot.source.http


class _FakeFuture:
    def _set_notify(self, notify):
        pass

    def _run(self, resource):
        pass


class _RecordingResource:
    """Stand-in for ``HTTPSource.ResourceClass`` that records multifuture calls."""

    def __init__(self):
        self.calls = []

    def partfuture(self, results, start, stop):
        return _FakeFuture()

    def multifuture(self, source, range_header, ranges, futures, results):
        self.calls.append(
            {
                "header": range_header["Range"],
                "batch": list(ranges),
                "future_keys": sorted(futures.keys()),
                "result_keys": sorted(results.keys()),
            }
        )
        return ("multifuture", tuple(ranges))


class _FakeExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, future):
        self.submitted.append(future)
        return future


def _make_source(http_max_header_bytes):
    source = uproot.source.http.HTTPSource.__new__(uproot.source.http.HTTPSource)
    source._fallback = None
    source._num_requests = 0
    source._num_requested_chunks = 0
    source._num_requested_bytes = 0
    source._http_max_header_bytes = http_max_header_bytes
    source._executor = _FakeExecutor()
    recorder = _RecordingResource()
    source.ResourceClass = recorder
    return source, recorder


@pytest.mark.parametrize(
    "ranges, http_max_header_bytes, batch_sizes",
    [
        ([(0, 5)], 1000, [1]),  # single range, single batch
        ([(0, 5), (10, 15), (20, 25)], 1000, [3]),  # several ranges, single batch
        ([(i * 10, i * 10 + 5) for i in range(8)], 20, [2, 2, 2, 2]),  # small batches
        ([(0, 5), (10, 15), (20, 25), (30, 35)], 18, [2, 1, 1]),  # mixed batch sizes
        ([(0, 5), (1000000, 1000005)], 12, [1, 1]),  # overflow on the last range
    ],
)
def test_chunks_batching_covers_every_range_exactly_once(
    ranges, http_max_header_bytes, batch_sizes
):
    source, recorder = _make_source(http_max_header_bytes)
    notifications = queue.Queue()

    chunks = source.chunks(ranges, notifications)

    # one chunk per requested range, one submitted future per batch
    assert len(chunks) == len(ranges)
    assert len(source._executor.submitted) == len(recorder.calls)
    assert [len(call["batch"]) for call in recorder.calls] == batch_sizes
    covered = []
    for call in recorder.calls:
        batch = call["batch"]
        batch_keys = sorted(tuple(r) for r in batch)

        # the header lists exactly the batch's ranges, in order
        expected_header = "bytes=" + ", ".join(f"{a}-{b - 1}" for a, b in batch)
        assert call["header"] == expected_header

        # futures/results were built for exactly the batch's ranges
        assert call["future_keys"] == batch_keys
        assert call["result_keys"] == batch_keys

        covered.extend(tuple(r) for r in batch)

    # every requested range is covered exactly once, in order
    assert covered == [tuple(r) for r in ranges]


@pytest.mark.parametrize("use_threads", [True, False])
def test_closed_source_raises_instead_of_hanging(http_server, use_threads):
    url = f"{http_server}/uproot-issue121.root"
    source = uproot.source.http.HTTPSource(
        url,
        timeout=10,
        num_fallback_workers=1,
        use_threads=use_threads,
    )
    assert not source.closed
    source.__exit__(None, None, None)

    # __exit__ must actually mark the executor closed (it previously called
    # shutdown(), which left closed == False)
    assert source.closed

    # submitting onto the closed executor must raise rather than enqueue onto a
    # dead pool (which would hang waiting on a notification that never arrives)
    future = uproot.source.http.HTTPResource.partfuture({}, 0, 100)
    with pytest.raises(OSError, match="resource is closed for file"):
        source._executor.submit(future)


def _multipart_body(boundary, parts):
    """Build a multipart/byteranges body from (trailing_headers, start, data) parts.

    ``trailing_headers`` are emitted *after* the ``Content-Range`` header, which
    is the case that misaligned the payload before the fix.
    """
    out = b""
    for trailing_headers, start, data in parts:
        out += boundary + b"\r\n"
        out += b"Content-Range: bytes %d-%d/100\r\n" % (start, start + len(data) - 1)
        for header in trailing_headers:
            out += header + b"\r\n"
        out += b"\r\n"
        out += data + b"\r\n"
    out += boundary + b"--\r\n"
    return out


def test_next_header_handles_header_after_content_range():
    boundary = b"--BOUNDARY"
    body = _multipart_body(
        boundary,
        [
            # an extra header AFTER Content-Range used to shift the payload
            ([b"Content-Type: application/octet-stream"], 0, b"AAAAA"),
            ([b"X-Extra: junk"], 10, b"BBBBB"),
        ],
    )
    buffer = uproot.source.http._ResponseBuffer(io.BytesIO(body))
    resource = uproot.source.http.HTTPResource.__new__(uproot.source.http.HTTPResource)

    range_string, size = resource.next_header(buffer)
    assert range_string == b"0-4"
    assert size == 100
    assert buffer.read(5) == b"AAAAA"

    range_string, size = resource.next_header(buffer)
    assert range_string == b"10-14"
    assert size == 100
    assert buffer.read(5) == b"BBBBB"
