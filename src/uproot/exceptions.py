# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines Uproot-specific exceptions, such as
:doc:`uproot.exceptions.KeyInFileError`.
"""

from __future__ import annotations

import uproot


class NameConflictWarning(Warning):
    pass


class CoalescedReadError(OSError):
    """
    Exception raised when a range of a coalesced request fails and the original
    error has no built-in ``OSError`` type to keep, such as
    ``aiohttp.ClientPayloadError``. The original error is the ``__cause__`` of
    this exception and is included in its message.
    """


class KeyInFileError(KeyError):
    """
    Exception raised by attempts to find ROOT objects in ``TDirectories``
    or ``TBranches`` in :doc:`uproot.behaviors.TBranch.HasBranches`, which
    both have a Python ``Mapping`` interface (square bracket syntax to extract
    items).

    This exception descends from Python's ``KeyError``, so it can be used in
    the normal way by interfaces that expect a missing item in a ``Mapping``
    to raise ``KeyError``, but it provides more information, depending on
    availability:

    * ``because``: an explanatory message
    * ``cycle``: the ROOT cycle number requested, if any
    * ``keys``: a list or partial list of keys that *are* in the object, in case
      of misspelling
    * ``file_path``: a path (or URL) to the file
    * ``object_path``: a path to the object within the ROOT file.
    """

    def __init__(
        self, key, because="", cycle=None, keys=None, file_path=None, object_path=None
    ):
        super().__init__(key)
        self.key = key
        self.because = because
        self.cycle = cycle
        self.keys = keys
        self.file_path = file_path
        self.object_path = object_path

    def __str__(self):
        because = "" if self.because == "" else " because " + self.because

        with_keys = ""
        if self.keys is not None:
            to_show = None
            keys = self.keys
            cut = 1
            while len(keys) > 1000 and cut < len(self.key):
                keys = [x for x in keys if x[:cut] == self.key[:cut]]
                cut += 1
            sorted_keys = sorted(
                keys, key=lambda x: uproot._util.damerau_levenshtein(self.key, x)
            )
            for key in sorted_keys:
                if to_show is None:
                    to_show = repr(key)
                else:
                    to_show += ", " + repr(key)
                if len(to_show) > 200:
                    to_show += "..."
                    break
            if to_show is None:
                to_show = "(none!)"
            with_keys = f"\n\n    Available keys: {to_show}\n"

        in_file = ""
        if self.file_path is not None:
            in_file = f"\nin file {self.file_path}"

        in_object = ""
        if self.object_path is not None:
            in_object = f"\nin object {self.object_path}"

        if self.cycle == "any":
            return f"not found: {self.key!r} (with any cycle number){because}{with_keys}{in_file}{in_object}"
        elif self.cycle is None:
            return f"not found: {self.key!r}{because}{with_keys}{in_file}{in_object}"
        else:
            return f"not found: {self.key!r} with cycle {self.cycle}{because}{with_keys}{in_file}{in_object}"
