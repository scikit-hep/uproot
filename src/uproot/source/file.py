# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a physical layer for local files.

Defines a :doc:`uproot.source.file.FileResource` (wrapped Python file handle)
and two sources: :doc:`uproot.source.file.MultithreadedFileSource` and
:doc:`uproot.source.file.MemmapSource`, which provide thread-safe local
file readers using many file handles or a memory-mapped file, respectively.

If the filesystem or operating system does not support memory-mapped files, the
:doc:`uproot.source.file.MultithreadedFileSource` is an automatic fallback.
"""

from __future__ import annotations

import os.path
import queue

import numpy

import uproot
import uproot.source.chunk
import uproot.source.futures


class FileResource(uproot.source.chunk.Resource):
    """
    Args:
        file_path (str): The filesystem path of the file to open.

    A :doc:`uproot.source.chunk.Resource` for a simple file handle.
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        try:
            self._file = open(self._file_path, "rb")
        except FileNotFoundError as err:
            raise uproot._util._file_not_found(file_path) from err

    @property
    def file(self):
        """
        The Python file handle.
        """
        return self._file

    @property
    def closed(self) -> bool:
        return self._file.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.__exit__(exception_type, exception_value, traceback)

    def get(self, start: int, stop: int) -> bytes:
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        self._file.seek(start)
        return self._file.read(stop - start)

    @staticmethod
    def future(source: uproot.source.chunk.Source, start: int, stop: int):
        """
        Args:
            source (:doc:`uproot.source.file.MultithreadedFileSource`): The
                data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :doc:`uproot.source.futures.ResourceFuture` that calls
        :ref:`uproot.source.file.FileResource.get` with ``start`` and ``stop``.
        """

        def task(resource):
            return resource.get(start, stop)

        return uproot.source.futures.ResourceFuture(task)


class MemmapSource(uproot.source.chunk.Source):
    """
    Args:
        file_path (str): The filesystem path of the file to open.
        options: Must include ``"num_fallback_workers"``.

    A :doc:`uproot.source.chunk.Source` that manages one memory-mapped file.
    """

    _dtype = uproot.source.chunk.Chunk._dtype

    def __init__(self, file_path: str, **options):
        self._num_fallback_workers = options["num_fallback_workers"]
        self._fallback_opts = options
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._open()

    def _open(self):
        try:
            self._file = numpy.memmap(self._file_path, dtype=self._dtype, mode="r")
            self._fallback = None
        except OSError:
            self._file = None
            opts = dict(self._fallback_opts)
            opts["num_workers"] = self._num_fallback_workers
            self._fallback = uproot.source.file.MultithreadedFileSource(
                self._file_path, **opts
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_file")
        # _open() recreates the fallback; keeping it here would leak its
        # worker threads (revived by __setstate__) when _open() overwrites it.
        state.pop("_fallback", None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        fallback = ""
        if self._fallback is not None:
            fallback = " with fallback"
        return f"<{type(self).__name__} {path}{fallback} at 0x{id(self):012x}>"

    def chunk(self, start: int, stop: int) -> uproot.source.chunk.Chunk:
        if self._fallback is None:
            if self.closed:
                raise OSError(f"memmap is closed for file {self._file_path}")

            self._num_requests += 1
            self._num_requested_chunks += 1
            self._num_requested_bytes += stop - start

            data = self._file[start:stop]
            future = uproot.source.futures.TrivialFuture(data)
            return uproot.source.chunk.Chunk(self, start, stop, future, is_memmap=True)

        else:
            return self._fallback.chunk(start, stop)

    def chunks(
        self, ranges: list[tuple[int, int]], notifications: queue.Queue
    ) -> list[uproot.source.chunk.Chunk]:
        if self._fallback is None:
            if self.closed:
                raise OSError(f"memmap is closed for file {self._file_path}")

            self._num_requests += 1
            self._num_requested_chunks += len(ranges)
            self._num_requested_bytes += sum(stop - start for start, stop in ranges)

            chunks = []
            for start, stop in ranges:
                data = self._file[start:stop]
                future = uproot.source.futures.TrivialFuture(data)
                chunk = uproot.source.chunk.Chunk(
                    self, start, stop, future, is_memmap=True
                )
                notifications.put(chunk)
                chunks.append(chunk)
            return chunks

        else:
            return self._fallback.chunks(ranges, notifications)

    @property
    def file(self):
        """
        The ``numpy.memmap`` array/file.
        """
        return self._file

    @property
    def fallback(self):
        """
        If None, the :ref:`uproot.source.file.MemmapSource.file` opened
        successfully and no fallback is needed.

        Otherwise, this is a :doc:`uproot.source.file.MultithreadedFileSource`
        to which all requests are forwarded.
        """
        return self._fallback

    @property
    def closed(self) -> bool:
        if self._fallback is None:
            return self._file._mmap.closed
        else:
            return self._fallback.closed

    def __enter__(self):
        if self._fallback is None:
            if hasattr(self._file._mmap, "__enter__"):
                self._file._mmap.__enter__()
        else:
            self._fallback.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self._fallback is None:
            if hasattr(self._file._mmap, "__exit__"):
                self._file._mmap.__exit__(exception_type, exception_value, traceback)
            else:
                self._file._mmap.close()
        else:
            self._fallback.__exit__(exception_type, exception_value, traceback)

    @property
    def num_bytes(self) -> int:
        if self._fallback is None:
            return self._file._mmap.size()
        else:
            return self._fallback.num_bytes


class MultithreadedFileSource(uproot.source.chunk.MultithreadedSource):
    """
    Args:
        file_path (str): The filesystem path of the file to open.
        options: Must include ``"num_workers"`` and ``"use_threads"``.

    A :doc:`uproot.source.chunk.MultithreadedSource` that manages many
    :doc:`uproot.source.file.FileResource` objects.
    """

    ResourceClass = FileResource

    def __init__(self, file_path: str, **options):
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0
        self._use_threads = options["use_threads"]
        self._num_workers = options["num_workers"]

        self._file_path = file_path
        self._open()

    def _open(self):
        if self._use_threads:
            self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
                [FileResource(self._file_path) for x in range(self._num_workers)]
            )
        else:
            self._executor = uproot.source.futures.ResourceTrivialExecutor(
                FileResource(self._file_path)
            )
        self._num_bytes = os.path.getsize(self._file_path)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()
