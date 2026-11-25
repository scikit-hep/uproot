# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines a physical layer for file-like objects.

Defines a :doc:`uproot.source.object.ObjectResource` (wrapped Python file-like
object) and one source :doc:`uproot.source.object.ObjectSource` which always
has exactly one worker (we can't assume that the object is thread-safe).
"""

from __future__ import annotations

import uproot
import uproot.source.chunk
import uproot.source.futures


class ObjectResource(uproot.source.chunk.Resource):
    """
    Args:
        obj: The file-like object to use.

    A :doc:`uproot.source.chunk.Resource` for a file-like object.

    This object must have the following methods:

    - ``read(num_bytes)`` where ``num_bytes`` is an integer number of bytes to
      read.
    - ``seek(position)`` where ``position`` is an integer position to seek to.

    Both of these methods change the internal state of the object, its current
    seek position (because ``read`` moves that position forward ``num_bytes``).
    Hence, it is in principle not thread-safe.
    """

    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    @property
    def closed(self) -> bool:
        return getattr(self._obj, "closed", False)

    def __enter__(self):
        if hasattr(self._obj, "__enter__"):
            self._obj.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(self._obj, "__exit__"):
            self._obj.__exit__(exception_type, exception_value, traceback)

    def get(self, start: int, stop: int):
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        self._obj.seek(start)
        return self._obj.read(stop - start)

    @staticmethod
    def future(source: uproot.source.chunk.Source, start: int, stop: int):
        """
        Args:
            source (:doc:`uproot.source.object.ObjectSource`): The data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :doc:`uproot.source.futures.ResourceFuture` that calls
        :ref:`uproot.source.object.ObjectResource.get` with ``start`` and
        ``stop``.
        """

        def task(resource):
            return resource.get(start, stop)

        return uproot.source.futures.ResourceFuture(task)


class ObjectSource(uproot.source.chunk.MultithreadedSource):
    """
    Args:
        obj: The file-like object to use.

    A :doc:`uproot.source.chunk.Source` for a file-like object. (Although this
    is a :doc:`uproot.source.chunk.MultithreadedSource`, it never has more or
    less than one thread.)

    This object must have the following methods:

    - ``read(num_bytes)`` where ``num_bytes`` is an integer number of bytes to
      read.
    - ``seek(position)`` where ``position`` is an integer position to seek to.

    Both of these methods change the internal state of the object, its current
    seek position (because ``read`` moves that position forward ``num_bytes``).
    Hence, it is in principle not thread-safe.
    """

    ResourceClass = ObjectResource

    def __init__(self, obj, **options):
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = repr(obj)

        if options["use_threads"]:
            self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
                [ObjectResource(obj)]
            )
        else:
            self._executor = uproot.source.futures.ResourceTrivialExecutor(
                ObjectResource(obj)
            )

        self._num_bytes = None
