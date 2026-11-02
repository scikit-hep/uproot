# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Read coalescing algorithms

Inspired in part by https://github.com/cms-sw/cmssw/blob/master/IOPool/TFileAdaptor/src/ReadRepacker.h
"""

from __future__ import annotations

import queue
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import uproot.exceptions
import uproot.source.chunk


@dataclass
class CoalesceConfig:
    max_range_gap: int = 32 * 1024
    max_request_ranges: int = 1024
    max_request_bytes: int = 10 * 1024 * 1024
    min_first_request_bytes: int = 32 * 1024


DEFAULT_CONFIG = CoalesceConfig()


def _read_error(item: BaseException, description: str) -> OSError:
    message = f"failed to read {description}: {item!r}"
    # the nearest builtin OSError type lets callers catch the error by that type,
    # and OSError itself becomes a more specific type if its errno has one
    error_type = next(
        (
            cls
            for cls in type(item).__mro__
            if cls.__module__ == "builtins" and issubclass(cls, OSError)
        ),
        uproot.exceptions.CoalescedReadError,
    )
    if isinstance(item, OSError) and item.errno is not None:
        error = error_type(item.errno, message, item.filename)
    else:
        error = error_type(message)
    # lets allow_read_errors_with_report match on the original error
    error.coalesced_read = True
    return error


class SliceFuture:
    def __init__(self, parent: Future, s: slice | int, description: str = ""):
        self._parent = parent
        self._s = s
        self._description = description

    def add_done_callback(self, callback, *, context=None):
        self._parent.add_done_callback(callback)

    def result(self, timeout=None):
        if uproot._util.wasm:
            # Pyodide futures don't support timeout
            result = self._parent.result()
        else:
            result = self._parent.result(timeout=timeout)
        item = result[self._s]
        # fsspec's cat_ranges returns exceptions in place of data by default
        if isinstance(item, BaseException):
            raise _read_error(item, self._description) from item
        return item


@dataclass
class RangeRequest:
    start: int
    stop: int
    future: Future | None


@dataclass
class Cluster:
    ranges: list[RangeRequest]
    _stop: int | None = None

    @property
    def start(self):
        # since these are built from sorted ranges, this is the min start
        return self.ranges[0].start

    @property
    def stop(self):
        # Note: in order for this to work one has to use `append` for adding new ranges
        return self._stop

    def append(self, range):
        if self._stop is None or range.stop > self._stop:
            self._stop = range.stop
        self.ranges.append(range)

    def __len__(self):
        return self.stop - self.start

    def set_future(self, future: Future):
        for range in self.ranges:
            local_start = range.start - self.start
            local_stop = range.stop - self.start
            range.future = SliceFuture(future, slice(local_start, local_stop))


@dataclass
class CoalescedRequest:
    clusters: list[Cluster]

    def ranges(self):
        return [(cluster.start, cluster.stop) for cluster in self.clusters]

    def set_future(self, future: Future, file_path: str):
        for i, cluster in enumerate(self.clusters):
            description = f"{file_path!r} bytes {cluster.start}:{cluster.stop}"
            cluster.set_future(SliceFuture(future, i, description))


def _merge_adjacent(ranges: list[RangeRequest], config: CoalesceConfig):
    sorted_ranges = sorted(ranges, key=lambda r: r.start)
    cluster = Cluster([])
    for current_range in sorted_ranges:
        if cluster.ranges and current_range.start - cluster.stop > config.max_range_gap:
            yield cluster
            cluster = Cluster([])
        cluster.append(current_range)
    if cluster.ranges:
        yield cluster


def _coalesce(ranges: list[RangeRequest], config: CoalesceConfig):
    clusters: list[Cluster] = []
    request_bytes: int = 0
    first_request = True
    for cluster in _merge_adjacent(ranges, config):
        if clusters and (
            len(clusters) >= config.max_request_ranges
            or request_bytes + len(cluster) >= config.max_request_bytes
            or (first_request and request_bytes >= config.min_first_request_bytes)
        ):
            yield CoalescedRequest(clusters)
            clusters = []
            request_bytes = 0
            first_request = False
        clusters.append(cluster)
        request_bytes += len(cluster)
    if clusters:
        yield CoalescedRequest(clusters)


def coalesce_requests(
    ranges: list[tuple[int, int]],
    submit_fn: Callable[[list[tuple[int, int]]], Future],
    source: uproot.source.chunk.Source,
    notifications: queue.Queue,
    config: CoalesceConfig | None = None,
):
    if config is None:
        config = DEFAULT_CONFIG
    all_requests = [RangeRequest(start, stop, None) for start, stop in ranges]
    for merged_request in _coalesce(all_requests, config):
        future = submit_fn(merged_request.ranges())
        merged_request.set_future(future, source.file_path)

    def chunkify(req: RangeRequest):
        chunk = uproot.source.chunk.Chunk(source, req.start, req.stop, req.future)
        if uproot._util.wasm:
            # Callbacks don't work in pyodide yet, so we call the notifier directly
            uproot.source.chunk.notifier(chunk, notifications)()
        else:
            req.future.add_done_callback(
                uproot.source.chunk.notifier(chunk, notifications)
            )
        return chunk

    return list(map(chunkify, all_requests))
