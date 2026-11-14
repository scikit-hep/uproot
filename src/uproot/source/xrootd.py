# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a physical layer for remote files, accessed via the XRootD protocol.

Defines a :doc:`uproot.source.xrootd.XRootDResource` (``XRootD.File``) and two
sources: :doc:`uproot.source.xrootd.MultithreadedXRootDSource` and
:doc:`uproot.source.xrootd.XRootDSource`. The latter requires the server to
support vector-read requests; if not, it automatically falls back to
:doc:`uproot.source.xrootd.MultithreadedXRootDSource`.
"""

from __future__ import annotations

import contextlib
import queue

import uproot
import uproot.source.chunk
import uproot.source.futures


def get_server_config(file):
    """
    Args:
        file (``XRootD.client.File``): An XRootD file object.

    Query a XRootD server for its configuration.

    Returns a 2-tuple of integers:

    * ``readv_iov_max``: The maximum number of elements that can be
      requested in a single vector read.
    * ``readv_ior_max``: The maximum number of bytes that can be requested
      per *element* in a vector read.
    """
    # Set from some sensible defaults in case this fails
    readv_iov_max = 1024
    readv_ior_max = 2097136

    XRootD_client = uproot.extras.XRootD_client()

    # Check if the file is stored locally on this machine
    last_url = file.get_property("LastURL")
    last_url = XRootD_client.URL(last_url)
    if last_url.protocol == "file" and last_url.hostid == "localhost":
        # The URL will redirect to a local file where XRootD will split the
        # vector reads into multiple sequential reads so just use the default
        return readv_iov_max, readv_ior_max

    # Find where the data is actually stored
    data_server = file.get_property("DataServer")
    if data_server == "":
        raise NotImplementedError

    data_server = XRootD_client.URL(data_server)
    data_server = f"{data_server.protocol}://{data_server.hostid}/"

    # Use a single query call to avoid doubling the latency
    fs = XRootD_client.FileSystem(data_server)
    status, result = fs.query(
        XRootD_client.flags.QueryCode.CONFIG, "readv_iov_max readv_ior_max"
    )
    if status.error:
        if status.code in (206,):
            raise TimeoutError(status.message)
        else:
            raise OSError(status.message)

    # Result is something like b'178956968\n2097136\n'
    readv_iov_max, readv_ior_max = map(int, result.split(b"\n", 1))

    return readv_iov_max, readv_ior_max


@contextlib.contextmanager
def trivial_resource():
    yield


class XRootDResource(uproot.source.chunk.Resource):
    """
    Args:
        file_path (str): A URL of the file to open.
        timeout (None or float): An optional timeout in seconds.

    A :doc:`uproot.source.chunk.Resource` for XRootD connections.
    """

    def __init__(self, file_path: str, timeout: float | None):
        self._file_path = file_path
        self._timeout = timeout
        self._open()

    def _open(self):
        XRootD_client = uproot.extras.XRootD_client()

        self._file = XRootD_client.File()

        status, _dummy = self._file.open(self._file_path, timeout=self._xrd_timeout())
        if status.error:
            self._xrd_error(status)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_file")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    def _xrd_timeout(self) -> int:
        if self._timeout is None:
            return 0
        else:
            return int(self._timeout)

    def _xrd_error(self, status):
        self._file.close(timeout=self._xrd_timeout())

        # https://github.com/xrootd/xrootd/blob/8e91462e76ab969720b40fc324714b84e0b4bd42/src/XrdCl/XrdClStatus.hh#L47-L103
        # https://github.com/xrootd/xrootd/blob/250eced4d3787c2ac5be2c8c922134153bbf7f08/src/XrdCl/XrdClStatus.cc#L34-L74
        if status.code in (101, 304, 400):
            raise uproot._util._file_not_found(self._file_path, status.message)
        elif status.code in (206,):
            raise TimeoutError(
                f"XRootD error: {status.message} in file {self._file_path}"
            )
        else:
            raise OSError(f"XRootD error: {status.message} in file {self._file_path}")

    @property
    def timeout(self) -> float | None:
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def file(self):
        """
        The ``XRootD.client.File`` object.
        """
        return self._file

    @property
    def num_bytes(self) -> int:
        status, info = self._file.stat(self._xrd_timeout())
        if status.error:
            self._xrd_error(status)
        return info.size

    @property
    def closed(self) -> bool:
        return not self._file.is_open()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.close(timeout=self._xrd_timeout())

    def get(self, start: int, stop: int) -> bytes:
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        status, data = self._file.read(
            int(start), int(stop - start), timeout=self._xrd_timeout()
        )
        if status.error:
            self._xrd_error(status)
        return data

    @staticmethod
    def future(source: uproot.source.chunk.Source, start: int, stop: int):
        """
        Args:
            source (:doc:`uproot.source.xrootd.MultithreadedXRootDSource`): The
                data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :doc:`uproot.source.futures.ResourceFuture` that calls
        :ref:`uproot.source.xrootd.XRootDResource.get` with ``start`` and
        ``stop``.
        """

        def task(resource):
            return resource.get(start, stop)

        return uproot.source.futures.ResourceFuture(task)

    @staticmethod
    def partfuture(results, start: int, stop: int):
        """
        Returns a :doc:`uproot.source.futures.ResourceFuture` to simply select
        the ``(start, stop)`` item from the ``results`` dict.

        In :ref:`uproot.source.xrootd.XRootDSource.chunks`, each chunk has a
        :ref:`uproot.source.xrootd.XRootDResource.partfuture` that are collectively
        filled by callbacks from :ref:`uproot.source.xrootd.XRootDResource.callbacker`.
        """

        def task(resource):
            return results[start, stop]

        return uproot.source.futures.ResourceFuture(task)

    @staticmethod
    def mergefuture(partfutures):
        """
        Returns a :doc:`uproot.source.futures.ResourceFuture` that merges the
        chunks previously submitted via
        :ref:`uproot.source.xrootd.XRootDResource.partfuture` which had to be split
        """

        def task(resource):
            if len(partfutures) == 1:
                return partfutures[0].result()

            chunk_buffers = []
            for future in partfutures:
                chunk_buffers.append(future.result())
            return b"".join(chunk_buffers)

        return uproot.source.futures.ResourceFuture(task)

    def callbacker(self, futures, results):
        """
        Returns an XRootD callback function to fill the ``futures`` and
        ``results``.
        """

        def callback(status, response, hosts):
            if status.error:
                try:
                    self._xrd_error(status)
                except Exception as err:
                    for future in futures.values():
                        future._set_excinfo(err)
            else:
                for chunk in response.chunks:
                    start, stop = chunk.offset, chunk.offset + chunk.length
                    results[start, stop] = chunk.buffer
                    futures[start, stop]._run(None)

        return callback


class XRootDSource(uproot.source.chunk.Source):
    """
    Args:
        file_path (str): A URL of the file to open.
        options: Must include ``"timeout"``, ``"max_num_elements"``, ``"use_threads"``, and ``"num_workers"``

    A :doc:`uproot.source.chunk.Source` that uses XRootD's vector-read
    to get many chunks in one request.
    """

    ResourceClass = XRootDResource

    def __init__(self, file_path: str, **options):
        options = dict(uproot.reading.open.defaults, **options)

        self._timeout = options.get("timeout")
        self._desired_max_num_elements = options["max_num_elements"]
        self._use_threads = options["use_threads"]
        self._num_workers = options["num_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._num_bytes = None
        self._open()

    def _open(self):
        self._resource = XRootDResource(self._file_path, self._timeout)

        # this ThreadPool does not need a resource, it's only used to submit
        # futures that wait for chunks that have been split to merge them.
        if self._use_threads:
            self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
                [trivial_resource() for _ in range(self._num_workers)]
            )
        else:
            self._executor = uproot.source.futures.ResourceTrivialExecutor(
                trivial_resource()
            )

        self._max_num_elements, self._max_element_size = get_server_config(
            self._resource.file
        )
        if self._desired_max_num_elements is not None:
            self._max_num_elements = min(
                self._max_num_elements, self._desired_max_num_elements
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return f"<{type(self).__name__} {path} at 0x{id(self):012x}>"

    def chunk(self, start: int, stop: int) -> uproot.source.chunk.Chunk:
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        data = self._resource.get(start, stop)
        future = uproot.source.futures.TrivialFuture(data)
        return uproot.source.chunk.Chunk(self, start, stop, future)

    def chunks(
        self, ranges: list[tuple[int, int]], notifications: queue.Queue
    ) -> list[uproot.source.chunk.Chunk]:
        if len(ranges) == 0:
            # nothing to request: don't send an empty vector read
            return []
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        # ranges for xrootd vector reads
        all_request_ranges = [[]]

        # dictionary telling us which xrootd request ranges correspond to the
        # actually requested ranges (given as (start, stop))
        # this is to track which requests were split into smaller ranges and have to be merged
        sub_ranges = {}

        def add_request_range(
            start: int, length: int, sub_ranges_list: list[tuple[int, int]]
        ):
            if len(all_request_ranges[-1]) >= self._max_num_elements:
                all_request_ranges.append([])
            all_request_ranges[-1].append((start, length))
            sub_ranges_list.append((start, start + length))

        # figure out the vector read ranges
        for start, stop in ranges:
            length = stop - start
            sub_ranges[start, stop] = []

            # if range larger than maximum, split into smaller ranges
            if length > self._max_element_size:
                nsplit = length // self._max_element_size
                rem = length % self._max_element_size
                for i in range(nsplit):
                    add_request_range(
                        start + i * self._max_element_size,
                        self._max_element_size,
                        sub_ranges[start, stop],
                    )
                if rem > 0:
                    add_request_range(
                        start + nsplit * self._max_element_size,
                        rem,
                        sub_ranges[start, stop],
                    )
            else:
                add_request_range(start, length, sub_ranges[start, stop])

        # submit the xrootd vector reads
        global_futures = {}
        for request_ranges in all_request_ranges:
            futures = {}
            results = {}
            for start, size in request_ranges:
                stop = start + size
                partfuture = self.ResourceClass.partfuture(results, start, stop)
                futures[start, stop] = partfuture
                global_futures[start, stop] = partfuture

            callback = self._resource.callbacker(futures, results)

            status = self._resource.file.vector_read(
                chunks=request_ranges,
                timeout=self._resource._xrd_timeout(),
                callback=callback,
            )
            if status.error:
                self._resource._xrd_error(status)

        # create chunks (possibly merging xrootd chunks)
        chunks = []
        for start, stop in ranges:
            partfutures = []
            for sub_start, sub_stop in sub_ranges[start, stop]:
                partfutures.append(global_futures[sub_start, sub_stop])
            future = self.ResourceClass.mergefuture(partfutures)
            chunk = uproot.source.chunk.Chunk(self, start, stop, future)
            future._set_notify(uproot.source.chunk.notifier(chunk, notifications))
            self._executor.submit(future)
            chunks.append(chunk)

        return chunks

    @property
    def resource(self):
        """
        The :doc:`uproot.source.xrootd.XRootDResource` object.
        """
        return self._resource

    @property
    def timeout(self) -> float | None:
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def file(self):
        """
        The ``XRootD.client.File`` object.
        """
        return self._resource.file

    @property
    def closed(self) -> bool:
        return self._resource.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Use shutdown() rather than __exit__() because the executor's resource
        # is trivial_resource(), a contextmanager generator that is never
        # __enter__'d. Calling __exit__() on an un-entered generator advances it
        # to the yield instead of past it, causing "generator didn't stop".
        self._executor.shutdown()
        self._resource.__exit__(exception_type, exception_value, traceback)

    @property
    def num_bytes(self) -> int:
        if self._num_bytes is None:
            self._num_bytes = self._resource.num_bytes
        return self._num_bytes


class MultithreadedXRootDSource(uproot.source.chunk.MultithreadedSource):
    """
    Args:
        file_path (str): A URL of the file to open.
        options: Must include ``"num_workers"``, ``"use_threads"``, and ``"timeout"``.

    A :doc:`uproot.source.chunk.MultithreadedSource` that manages many
    :doc:`uproot.source.xrootd.XRootDResource` objects.
    """

    ResourceClass = XRootDResource

    def __init__(self, file_path: str, **options):
        options = dict(uproot.reading.open.defaults, **options)

        self._num_workers = options["num_workers"]
        self._timeout = options.get("timeout")
        self._use_threads = options["use_threads"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._num_bytes = None
        self._open()

    def _open(self):
        if self._use_threads:
            self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
                [
                    XRootDResource(self._file_path, self._timeout)
                    for _ in range(self._num_workers)
                ]
            )
        else:
            self._executor = uproot.source.futures.ResourceTrivialExecutor(
                XRootDResource(self._file_path, self._timeout)
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    @property
    def timeout(self) -> float | None:
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def num_bytes(self) -> int:
        if self._num_bytes is None:
            if hasattr(self._executor, "workers"):
                self._num_bytes = self._executor.workers[0].resource.num_bytes
            else:
                self._num_bytes = self._executor._resource.num_bytes

        return self._num_bytes
