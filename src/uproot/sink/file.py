# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a physical layer for writing local files.

Defines a :doc:`uproot.sink.file.FileSink`, which can wrap a Python file-like object
that has ``read``, ``write``, ``seek``, ``tell``, and ``flush`` methods and has a
context manager (Python's ``with`` statement) to ensure that files are properly closed
(although files are flushed after every object-write).
"""

from __future__ import annotations

import io
import numbers
import os
from typing import IO

import fsspec
from fsspec.implementations.cached import WholeFileCacheFileSystem

import uproot._util


class FileSink:
    """
    Args:
        urlpath_or_file_like (str, Path, or file-like object): If a string or Path, a
            filesystem URL that specifies the file to open by fsspec. If a file-like object, it
            must have ``read``, ``write``, ``seek``, ``tell``, and ``flush`` methods.
        mode ("create", "recreate", or "update"): Like ROOT's ``TFile`` options.
            ``"create"`` raises ``FileExistsError`` if the path already exists
            (a file-like object is neither checked nor truncated), ``"recreate"``
            truncates an existing file (or a file-like object that has a
            ``truncate`` method), and ``"update"`` keeps its contents. In all three,
            a missing path is created along with its parent directories.

    An object that can write (and read) files on a local or remote filesystem.
    It can be initialized from a file-like object (already opened) or a filesystem URL.
    If initialized from a filesystem URL, fsspec is used to open the file.
    In this case the file is opened in the first read or write operation, unless
    it has to be truncated first.
    """

    def __init__(
        self,
        urlpath_or_file_like: str | IO,
        mode: str = "update",
        **storage_options,
    ):
        if mode not in ("create", "recreate", "update"):
            raise ValueError(
                f"mode must be 'create', 'recreate', or 'update', not {mode!r}"
            )

        self._open_file = None
        self._file = None
        self._closed = False

        if uproot._util.is_file_like(
            urlpath_or_file_like, readable=False, writable=False, seekable=False
        ):
            self._file = urlpath_or_file_like

            if not uproot._util.is_file_like(
                self._file, readable=True, writable=True, seekable=True
            ):
                raise TypeError(
                    """writable file can only be created from a file path or an object that supports reading and writing"""
                )

            truncate = mode == "recreate"
        else:
            fs, path = fsspec.core.url_to_fs(urlpath_or_file_like, **storage_options)
            truncate = False
            if mode == "create":
                if not self._create_file(fs, path):
                    raise FileExistsError(
                        "path exists and refusing to overwrite (use 'uproot.recreate' "
                        f"to overwrite)\n\nfor path {urlpath_or_file_like}"
                    )
                # the file is empty, but filecache and simplecache may open a
                # stale local copy of a since-deleted file instead
                truncate = isinstance(fs, WholeFileCacheFileSystem)
            elif not fs.exists(path):
                self._truncate_file(fs, path)
            elif mode == "recreate":
                truncate = True

            self._open_file = fsspec.core.OpenFile(fs, path, mode="r+b")

        if truncate:
            # through the opened file, rather than fs.touch, so that a filesystem
            # that cannot open it for writing fails before the file is lost
            self._ensure()
            if not self._truncate_opened_file() and not self.from_object:
                # the filesystem can open the file for writing, but its file
                # objects cannot truncate (XRootD's), so truncate it by path
                self._file.close()
                self._file = None
                fs.touch(path, truncate=True)

    def _truncate_opened_file(self) -> bool:
        """
        Truncates the opened file to zero bytes, returning False if its file
        object does not support it.
        """
        # truncate is not required of file-like objects, and io.IOBase's
        # default implementation raises io.UnsupportedOperation
        truncate = getattr(self._file, "truncate", None)
        if not callable(truncate):
            return False
        try:
            truncate(0)
        except io.UnsupportedOperation:
            return False
        return True

    @staticmethod
    def _make_parent_directories(fs, path: str) -> None:
        parent_directory = fs.sep.join(path.split(fs.sep)[:-1])
        fs.mkdirs(parent_directory, exist_ok=True)

    @classmethod
    def _create_file(cls, fs, path: str) -> bool:
        """
        Args:
            fs (fsspec.AbstractFileSystem): The filesystem of the file.
            path (str): The file's path within ``fs``.

        Creates an empty file, returning False if it already exists.
        Creates parent directories if necessary.
        """
        # filecache and simplecache would create only their local copy
        # exclusively, failing on a stale copy of a since-deleted file, and then
        # upload it over the destination, so create it in the filesystem they wrap
        target = fs
        while isinstance(target, WholeFileCacheFileSystem):
            target = target.fs

        if not target.exists(path):
            cls._make_parent_directories(target, path)
            try:
                # exclusive creation, so that a file that appears after the check
                # is not overwritten (atomic on local filesystems)
                with target.open(path, "xb"):
                    pass
                return True
            except FileExistsError:
                pass
            except (ValueError, NotImplementedError):
                # this filesystem does not support mode "xb"
                target.touch(path, truncate=True)
                return True
        return False

    @classmethod
    def _truncate_file(cls, fs, path: str) -> None:
        """
        Args:
            fs (fsspec.AbstractFileSystem): The filesystem of the file.
            path (str): The file's path within ``fs``.

        Truncates the file to zero bytes. Creates parent directories if necessary.
        """
        cls._make_parent_directories(fs, path)
        fs.touch(path, truncate=True)

    @property
    def from_object(self) -> bool:
        """
        True if constructed with a file-like object; False otherwise.
        """
        return self._open_file is None

    @property
    def file_path(self) -> str | None:
        """
        A path to the file, which is None if constructed with a file-like object.
        """
        return self._open_file.path if self._open_file else None

    def _ensure(self):
        """
        Opens the file if it is not already open.
        Sets the file's position to the beginning.
        """
        if self._closed:
            raise OSError(f"file sink is closed{self.in_path}")

        if not self._file:
            self._file = self._open_file.open()

        self._file.seek(0)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_file")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._file = None

    def tell(self) -> int:
        """
        Calls the file or file-like object's ``tell`` method.
        """
        self._ensure()
        return self._file.tell()

    def flush(self) -> None:
        """
        Calls the file or file-like object's ``flush`` method.
        """
        self._ensure()
        return self._file.flush()

    @property
    def closed(self) -> bool:
        """
        True if the file has been closed; False otherwise.
        """
        return self._closed

    def close(self):
        """
        Closes the file (calls ``close`` if it has such a method) and marks the
        sink as closed so that closure is permanent: subsequent reads or writes
        raise instead of silently reopening the file.
        """
        if self._file is not None and hasattr(self._file, "close"):
            self._file.close()
        self._file = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    @property
    def in_path(self) -> str:
        return f"\n\nin path: {self.file_path}" if self.file_path is not None else ""

    def write(self, location: int, serialization) -> int:
        """
        Args:
            location (int): Position in the file to write.
            serialization (bytes or NumPy array): Data to write to the file.

        Writes data to the file at a specific location, calling the file-like
        object's ``seek`` and ``write`` methods.
        """
        self._ensure()
        self._file.seek(location)
        return self._file.write(serialization)

    def set_file_length(self, length: int):
        """
        Sets the file's length to ``length``, truncating with zeros if necessary.

        Calls ``seek``, ``tell``, and possibly ``write``.
        """
        self._ensure()
        self._file.seek(0, os.SEEK_END)
        missing = length - self._file.tell()
        if missing > 0:
            self._file.write(b"\x00" * missing)

    def read(self, location: int, num_bytes: int, insist: bool = True) -> bytes:
        """
        Args:
            location (int): Position in the file to read.
            num_bytes (int): Number of bytes to read from the file.
            insist (bool): If True, raise an OSError if ``num_bytes`` cannot be read
                from the file. Otherwise, this function may return data with fewer
                than ``num_bytes``.

        Returns a bytes object of data from the file by calling the file-like
        object's ``seek`` and ``read`` methods. The ``insist`` parameter can be
        used to ensure that the output has the requested length.
        """
        self._ensure()
        self._file.seek(location)
        out = self._file.read(num_bytes)
        if insist is True:
            if len(out) != num_bytes:
                raise OSError(
                    f"could not read {num_bytes} bytes from the file at position {location}{self.in_path}"
                )
        elif isinstance(insist, numbers.Integral) and len(out) < insist:
            raise OSError(
                f"could not read {insist} bytes from the file at position {location}{self.in_path}"
            )
        return out
