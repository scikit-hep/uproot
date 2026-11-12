# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines simple, thread-safe cache classes satisfying the
``MutableMapping`` protocol.

The :doc:`uproot.cache.LRUCache` implements a least-recently used eviction
policy that limits the number of items in the cache (used as an
``object_cache``).

The :doc:`uproot.cache.LRUArrayCache` implements the same policy, limiting the
total number of bytes, as reported by ``nbytes``.
"""

from __future__ import annotations

import threading
from collections.abc import MutableMapping

import uproot


class LRUCache(MutableMapping):
    """
    Args:
        limit (None or int): Number of objects to allow in the cache before
            evicting the least-recently used. If None, this cache never evicts.

    LRUCache is a ``MutableMapping`` that evicts the least-recently used
    objects when the ``current`` number of objects exceeds the ``limit``.

    Get and set (or explicitly remove) items with ``MutableMapping`` syntax:
    square bracket subscripting.

    LRUCache is thread-safe for all options: getting, setting, deleting,
    iterating, listing keys, values, and items.

    This cache is insensitive to the size of the objects it stores, and hence
    is a better ``object_cache`` than an ``array_cache``  (cf. :doc:`uproot.cache.LRUArrayCache`).
    """

    @classmethod
    def sizeof(cls, obj):
        """
        The "size of" an object in this cache is always exactly 1.
        """
        return 1

    def __init__(self, limit):
        self._limit = limit
        self._current = 0
        self._data = {}
        self._lock = threading.Lock()

    def __getstate__(self):
        return {
            "_limit": self._limit,
        }

    def __setstate__(self, state):
        self._limit = state["_limit"]
        self._current = 0
        self._data = {}
        self._lock = threading.Lock()

    def __repr__(self):
        if self._limit is None:
            limit = "(no limit)"
        else:
            limit = f"({self._current}/{self._limit} full)"
        return f"<LRUCache {limit} at 0x{id(self):012x}>"

    @property
    def limit(self):
        """
        Number of objects to allow in the cache before evicting the
        least-recently used. If None, this cache never evicts.
        """
        return self._limit

    @property
    def current(self):
        """
        Current number of items in the cache.
        """
        return self._current

    def keys(self):
        """
        Returns a copy of the keys currently in the cache, in least-recently
        used order.

        The list ascends from least-recently used to most-recently used: index
        ``0`` is the least-recently used and index ``-1`` is the most-recently
        used.

        (Calling this method does not change the order.)
        """
        with self._lock:
            return list(self._data)

    def values(self):
        """
        Returns a copy of the values currently in the cache, in least-recently
        used order.

        The list ascends from least-recently used to most-recently used: index
        ``0`` is the least-recently used and index ``-1`` is the most-recently
        used.

        (Calling this method does not change the order.)
        """
        with self._lock:
            return list(self._data.values())

    def items(self):
        """
        Returns a copy of the items currently in the cache, in least-recently
        used order.

        The list ascends from least-recently used to most-recently used: index
        ``0`` is the least-recently used and index ``-1`` is the most-recently
        used.

        (Calling this method does not change the order.)
        """
        with self._lock:
            return list(self._data.items())

    def __getitem__(self, where):
        with self._lock:
            out = self._data[where]
            # Move to end (most-recently used)
            self._data[where] = self._data.pop(where)
            return out

    def __setitem__(self, where, what):
        with self._lock:
            if where in self._data:
                # Remove old entry so re-insert puts it at the end
                self._current -= self.sizeof(self._data.pop(where))
            self._data[where] = what
            self._current += self.sizeof(what)

            if self._limit is not None:
                while self._current > self._limit and self._data:
                    # Evict least-recently used (first item)
                    key, val = next(iter(self._data.items()))
                    self._current -= self.sizeof(val)
                    del self._data[key]

    def __delitem__(self, where):
        with self._lock:
            self._current -= self.sizeof(self._data[where])
            del self._data[where]

    def __iter__(self):
        with self._lock:
            keys = list(self._data)
        yield from keys

    def __len__(self):
        with self._lock:
            return len(self._data)


class LRUArrayCache(LRUCache):
    """
    Args:
        limit_bytes (None, int, or str): Amount of data to allow in the cache
            before evicting the least-recently used. An integer is interpreted
            as a number of bytes and a string must be a number followed by a
            unit, such as "100 MB". If None, this cache never evicts.

    LRUArrayCache is a ``MutableMapping`` that evicts the least-recently used
    objects when the ``current`` number of bytes exceeds the ``limit``. The
    size of an object is taken from its ``nbytes`` attribute. If it does not
    have an ``nbytes``, it is presumed to have ``default_nbytes``.

    Get and set (or explicitly remove) items with ``MutableMapping`` syntax:
    square bracket subscripting.

    LRUArrayCache is thread-safe for all options: getting, setting, deleting,
    iterating, listing keys, values, and items.

    This cache is sensitive to the size of the objects it stores, but only if
    those objects have meaningful ``nbytes``. It is therefore a better
    ``array_cache`` than an ``object_cache``.
    """

    default_nbytes = 1024

    @classmethod
    def sizeof(cls, what):
        """
        The "size of" an object in this cache is its

        - ``nbytes`` attribute/property if it has one (NumPy and Awkward Arrays,
          Pandas Series),
        - ``memory_usage().sum()`` if it exists (Pandas DataFrames),
        - ``default_nbytes`` otherwise.
        """
        if hasattr(what, "nbytes"):
            return what.nbytes
        if hasattr(what, "memory_usage") and callable(what.memory_usage):
            tmp = what.memory_usage()
            if hasattr(tmp, "sum") and callable(tmp.sum):
                return tmp.sum()
        return cls.default_nbytes

    def __init__(self, limit_bytes):
        limit = None if limit_bytes is None else uproot._util.memory_size(limit_bytes)
        super().__init__(limit)

    def __repr__(self):
        if self._limit is None:
            limit = "(no limit)"
        else:
            limit = f"({self._current}/{self._limit} bytes full)"
        return f"<LRUArrayCache {limit} at 0x{id(self):012x}>"

    @property
    def limit(self):
        """
        Number of bytes to allow in the cache before evicting the
        least-recently used. If None, this cache never evicts.
        """
        return self._limit

    @property
    def current(self):
        """
        Current number of bytes in the cache.
        """
        return self._current
