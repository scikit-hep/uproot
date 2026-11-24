# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import uproot


class CuFileSource:
    """
    Class for physically reading and writing data from a file using kvikio bindings
    to CuFile API.

    Args:
        filepath (str): Path fo file.
        method (str): Method to open file with. e.g. "w", "r"

    Provides a consistent interface to kvikio cufile API. Stores metadata of
    read requests.
    """

    def __init__(self, file_path, method):
        kvikio = uproot.extras.kvikio()
        self._file_path = file_path
        self._handle = kvikio.CuFile(file_path, method)

        self._futures = []
        self._requested_chunk_sizes = []
        self._num_requested_bytes = 0
        self._num_requested_chunks = 0

    def close(self):
        self._handle.close()

    def pread(
        self,
        buffer,
        size: int,
        file_offset: int = 0,
        task_size: int | None = None,
    ):
        self._num_requested_chunks += 1
        self._num_requested_bytes += size
        self._requested_chunk_sizes.append(size)

        future = self._handle.pread(
            buffer, size=size, file_offset=file_offset, task_size=task_size
        )
        self._futures.append(future)
        return future

    def get_all(self):
        """
        Wait for all futures in self._futures to finish.
        """
        for future in self.futures:
            future.get()

    @property
    def futures(self) -> list:
        """
        List of kvikio.IOFutures corresponding to all pread() calls
        """
        return self._futures

    @property
    def file_path(self) -> str:
        """
        A path to the file (or URL).
        """
        return self._file_path

    @property
    def num_requested_chunks(self) -> int:
        """
        The number of requests that have been made (performance counter).
        """
        return self._num_requested_chunks

    @property
    def num_requested_bytes(self) -> int:
        """
        The number of bytes that have been requested (performance counter).
        """
        return self._num_requested_bytes

    @property
    def requested_chunk_sizes(self) -> list:
        """
        The size of requests that have been made in number of bytes (performance counter).
        """
        return self._requested_chunk_sizes
