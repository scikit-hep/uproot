# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines high-level functions and objects for file-writing.

The :doc:`uproot.writing.writable.create`, :doc:`uproot.writing.writable.recreate`, and :doc:`uproot.writing.writable.update`
functions open files for writing, overwriting, or updating, in a way that is similar
to :doc:`uproot.reading.open`.

The :doc:`uproot.writing.writable.WritableFile`, :doc:`uproot.writing.writable.WritableDirectory`,
:doc:`uproot.writing.writable.WritableTree`, and :doc:`uproot.writing.writable.WritableBranch`
classes are writable versions of :doc:`uproot.reading.ReadOnlyFile`, :doc:`uproot.reading.ReadOnlyDirectory`,
:doc:`uproot.behaviors.TTree.TTree`, and :doc:`uproot.behaviors.TBranch.TBranch`.

There is no feature parity between writable and readable versions of each of these
types. Writing and reading are considered separate projects with different capabilities.
"""

from __future__ import annotations

import datetime
import difflib
import itertools
import queue
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import IO

import awkward
import numpy

import uproot._util
import uproot.compression
import uproot.deserialization
import uproot.exceptions
import uproot.model
import uproot.models.TObjString
import uproot.sink.file
import uproot.writing._cascade
import uproot.writing._cascadentuple
import uproot.writing._cascadetree
import uproot.writing.identify
from uproot._util import no_filter, no_rename


def create(file_path: str | Path | IO, **options):
    """
    Args:
        file_path (str, ``pathlib.Path`` or file-like object): The filesystem path of the
            file to open or an open file.
        options: See below.

    Opens a local file for writing. Like ROOT's ``"CREATE"`` option, this function
    raises an error (``FileExistsError``) if a file already exists at ``file_path``.

    Returns a :doc:`uproot.writing.writable.WritableDirectory`.

    ..
      Future Dev Note: These shared writing options are manually copied across
      multiple docstrings to make them available in `help` and online.

    Options (type; default):

    * initial_directory_bytes (int; 256): The number of bytes to allocate for new directories so that
      TKeys can be added to them without needing immediate rewriting of the block.
    * initial_streamers_bytes (int; 1024): The number of bytes to allocate for a new list of streamers
      so that streamers can be added to it without needing immediate rewriting
    * uuid_function (callable; ``uuid.uuid1``): Function to create the file's UUID and/or any directory's UUID.
    * compression (:doc:`uproot.compression.Compression` or None; ``uproot.ZLIB(1)``): Compression algorithm
      and level for new objects added to the file. Can be updated after creating
      the :doc:`uproot.writing.writable.WritableFile`.

    See :doc:`uproot.writing.writable.WritableFile` for details on these options.

    Additional options are passed as ``storage_options`` to the fsspec filesystem
    """
    return _create_or_recreate(file_path, "create", options)


def recreate(file_path: str | Path | IO, **options):
    """
    Args:
        file_path (str, ``pathlib.Path`` or file-like object): The filesystem path of the
            file to open or an open file.
        options: See below.

    Opens a local file for writing. Like ROOT's ``"RECREATE"`` option, this function
    overwrites any file that already exists at ``file_path``.

    Returns a :doc:`uproot.writing.writable.WritableDirectory`.

    ..
      Future Dev Note: These shared writing options are manually copied across
      multiple docstrings to make them available in `help` and online.

    Options (type; default):

    * initial_directory_bytes (int; 256): The number of bytes to allocate for new directories so that
      TKeys can be added to them without needing immediate rewriting of the block.
    * initial_streamers_bytes (int; 1024): The number of bytes to allocate for a new list of streamers
      so that streamers can be added to it without needing immediate rewriting
    * uuid_function (callable; ``uuid.uuid1``): Function to create the file's UUID and/or any directory's UUID.
    * compression (:doc:`uproot.compression.Compression` or None; ``uproot.ZLIB(1)``): Compression algorithm
      and level for new objects added to the file. Can be updated after creating
      the :doc:`uproot.writing.writable.WritableFile`.

    See :doc:`uproot.writing.writable.WritableFile` for details on these options.

    Additional options are passed as ``storage_options`` to the fsspec filesystem
    """
    return _create_or_recreate(file_path, "recreate", options)


def _create_or_recreate(file_path, mode, options):
    file_path = uproot._util.regularize_path(file_path)
    compression = options.pop("compression", create.defaults["compression"])
    initial_directory_bytes = options.pop(
        "initial_directory_bytes", create.defaults["initial_directory_bytes"]
    )
    initial_streamers_bytes = options.pop(
        "initial_streamers_bytes", create.defaults["initial_streamers_bytes"]
    )
    uuid_function = options.pop("uuid_function", create.defaults["uuid_function"])
    # before opening the sink, which may truncate an existing file
    storage_options = _storage_options(options, "uproot.create or uproot.recreate")

    sink = uproot.sink.file.FileSink(file_path, mode=mode, **storage_options)
    cascading = uproot.writing._cascade.create_empty(
        sink,
        compression,
        initial_directory_bytes,
        initial_streamers_bytes,
        uuid_function,
    )
    return WritableFile(
        sink, cascading, initial_directory_bytes, uuid_function
    ).root_directory


def update(file_path: str | Path | IO, **options):
    """
    Args:
        file_path (str, ``pathlib.Path`` or file-like object): The filesystem path of the
            file to open or an open file.
        options: See below.

    Opens a local file for writing. Like ROOT's ``"UPDATE"`` option, this function
    expects a file to already exist at ``file_path`` and opens it so that new data
    can be added to it or individual objects may be deleted from it.

    Returns a :doc:`uproot.writing.writable.WritableDirectory`.

    ..
      Future Dev Note: These shared writing options are manually copied across
      multiple docstrings to make them available in `help` and online.

    Options (type; default):

    * initial_directory_bytes (int; 256): The number of bytes to allocate for new directories so that
      TKeys can be added to them without needing immediate rewriting of the block.
    * uuid_function (callable; ``uuid.uuid1``): Function to create the file's UUID and/or any directory's UUID.

    See :doc:`uproot.writing.writable.WritableFile` for details on these options.

    Additional options are passed as ``storage_options`` to the fsspec filesystem
    """

    file_path = uproot._util.regularize_path(file_path)
    initial_directory_bytes = options.pop(
        "initial_directory_bytes", create.defaults["initial_directory_bytes"]
    )
    uuid_function = options.pop("uuid_function", create.defaults["uuid_function"])
    storage_options = _storage_options(options, "uproot.update")

    sink = uproot.sink.file.FileSink(file_path, mode="update", **storage_options)
    cascading = uproot.writing._cascade.update_existing(
        sink,
        initial_directory_bytes,
        uuid_function,
    )
    return WritableFile(
        sink, cascading, initial_directory_bytes, uuid_function
    ).root_directory


def _storage_options(options, function_name):
    """
    The options left after removing the ones uproot knows are passed as
    ``storage_options`` to the fsspec filesystem.

    A filesystem may silently ignore an option it does not know, so an option
    that is (or closely resembles) an uproot writing option is rejected here
    rather than passed on, e.g. a misspelled ``compresion``.
    """
    mistaken = []
    for name in options:
        # a cutoff above the default 0.6, which would reject real fsspec options
        # such as session or compresslevel for resembling compression
        matches = difflib.get_close_matches(name, create.defaults, n=1, cutoff=0.8)
        if matches and matches[0] != name:
            mistaken.append(f"{name!r} (did you mean {matches[0]!r}?)")
        elif matches:
            mistaken.append(repr(name))
    if mistaken:
        raise TypeError(
            f"unrecognized options for {function_name}: " + ", ".join(mistaken)
        )
    return options


create.defaults = {
    "compression": uproot.compression.ZLIB(1),
    "initial_directory_bytes": 256,
    "initial_streamers_bytes": 1024,  # 256,
    "uuid_function": uuid.uuid1,
}
recreate.defaults = create.defaults
update.defaults = create.defaults


class WritableFile(uproot.reading.CommonFileMethods):
    """
    Args:
        sink (:doc:`uproot.sink.file.FileSink`): The physical layer for file-writing.
        cascading (:doc:`uproot.writing._cascade.CascadingFile`): The low-level file
            object.
        initial_directory_bytes (int): Number of bytes to allocate for new directories,
            so that TKeys can be added to them without immediately needing to rewrite
            the block.
        uuid_function (zero-argument callable returning a ``uuid.UUID``): Function to
            create the file's UUID and/or any directory's UUID.

    Handle to a writable ROOT file, usually created by :doc:`uproot.writing.writable.create`,
    :doc:`uproot.writing.writable.recreate`, or :doc:`uproot.writing.writable.update` and
    accessed through a :doc:`uproot.writing.writable.WritableDirectory`.
    """

    def __init__(self, sink, cascading, initial_directory_bytes, uuid_function):
        self._sink = sink
        self._cascading = cascading
        self._initial_directory_bytes = initial_directory_bytes
        self._uuid_function = uuid_function

        self._file_path = sink.file_path
        self._fVersion = self._cascading.fileheader.version
        self._fBEGIN = self._cascading.fileheader.begin
        self._fNbytesName = self._cascading.fileheader.begin_num_bytes
        self._fUUID = self._cascading.fileheader.uuid.bytes

        self._trees = {}
        self._ntuples = {}

    def __repr__(self):
        return f"<WritableFile {self.file_path!r} at 0x{id(self):012x}>"

    @property
    def sink(self) -> uproot.sink.file.FileSink:
        """
        Returns a :doc:`uproot.sink.file.FileSink`, the physical layer for writing
        (and sometimes reading) data.
        """
        return self._sink

    @property
    def initial_directory_bytes(self) -> int:
        """
        Number of bytes to allocate for new directories, so that TKeys can be added
        to them without immediately needing to rewrite the block.
        """
        return self._initial_directory_bytes

    @initial_directory_bytes.setter
    def initial_directory_bytes(self, value):
        self._initial_directory_bytes = value

    @property
    def uuid_function(self):
        """
        The function used to create the file's UUID and/or any directory's UUID.
        """
        return self._uuid_function

    @uuid_function.setter
    def uuid_function(self, value):
        self._uuid_function = value

    @property
    def options(self):
        """
        The options passed to :doc:`uproot.writing.writable.create`,
        :doc:`uproot.writing.writable.recreate`, or :doc:`uproot.writing.writable.update`
        when opening this file.
        """
        return {
            "initial_directory_bytes": self._initial_directory_bytes,
            "uuid_function": self._uuid_function,
        }

    @property
    def is_64bit(self) -> bool:
        """
        True if the file has 8-byte pointers in its header; False if the pointers are 4-byte.
        """
        return self._cascading.fileheader.big

    @property
    def compression(self) -> uproot.compression.Compression | None:
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new objects added to the file.

        This property can be changed, which allows you to write different objects
        with different compression settings.

        See also :ref:`uproot.writing.writable.WritableFile.fCompress`.
        """
        return self._cascading.fileheader.compression

    @compression.setter
    def compression(self, value):
        if value is None or isinstance(value, uproot.compression.Compression):
            self._cascading.fileheader.compression = value
            self._cascading.fileheader.write(self._sink)
            self._sink.flush()
        else:
            raise TypeError(
                "compression must be None or a uproot.compression.Compression object, like uproot.ZLIB(4) or uproot.ZSTD(0)"
            )

    @property
    def fSeekFree(self):
        """
        The seek point (int) to the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._cascading.fileheader.free_location

    @property
    def fNbytesFree(self) -> int:
        """
        The number of bytes in the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._cascading.fileheader.free_num_bytes

    @property
    def nfree(self) -> int:
        """
        The number of objects in the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._cascading.fileheader.free_num_slices + 1

    @property
    def fUnits(self) -> int:
        """
        Number of bytes in the serialization of file seek points, which can either
        be 4 or 8.
        """
        return 8 if self._cascading.fileheader.big else 4

    @property
    def fCompress(self):
        """
        Compression algorithm and level (as an integer code) for new objects added
        to the file.

        This property can be changed, which allows you to write different objects
        with different compression settings.

        See also :ref:`uproot.writing.writable.WritableFile.compression`.
        """
        if self._cascading.fileheader.compression is None:
            return uproot.compression.ZLIB(0).code
        else:
            return self._cascading.fileheader.compression.code

    @property
    def fSeekInfo(self):
        """
        The seek point (int) to the ``TStreamerInfo`` data, where
        TStreamerInfo records are located.
        """
        return self._cascading.fileheader.info_location

    @property
    def fNbytesInfo(self) -> int:
        """
        The number of bytes in the ``TStreamerInfo`` data, where
        TStreamerInfo records are located.
        """
        return self._cascading.fileheader.info_num_bytes

    @property
    def uuid(self):
        """
        The unique identifier (UUID) of the ROOT file expressed as a Python
        ``uuid.UUID`` object.
        """
        return self._cascading.fileheader.uuid

    @property
    def root_directory(self):
        """
        The root (first) directory in the file as a :doc:`uproot.writing.writable.WritableDirectory`.
        """
        return WritableDirectory((), self, self._cascading.rootdirectory)

    def update_streamers(self, streamers):
        """
        Overwrite the TStreamerInfo in this file with a new list of :doc:`uproot.streamers.Model_TStreamerInfo`
        or :doc:`uproot.writable._cascade.RawStreamerInfo`.
        """
        self._cascading.streamers.update_streamers(self.sink, streamers)

    @property
    def file_path(self) -> str | None:
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file_path

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._sink.close()

    @property
    def closed(self) -> bool:
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._sink.closed

    def __enter__(self):
        self._sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._sink.__exit__(exception_type, exception_value, traceback)

    def _new_tree(self, tree):
        self._trees[tree._cascading.key.seek_location] = tree

    def _new_ntuple(self, ntuple):
        self._ntuples[ntuple._cascading.key.seek_location] = ntuple

    def _has_tree(self, loc):
        return loc in self._trees

    def _get_tree(self, loc):
        return self._trees[loc]

    def _has_ntuple(self, loc):
        return loc in self._ntuples

    def _get_ntuple(self, loc):
        return self._ntuples[loc]

    def _move_tree(self, oldloc, newloc):
        tree = self._trees[oldloc]
        del self._trees[oldloc]
        self._trees[newloc] = tree


class WritableDirectory(MutableMapping):
    """
    Args:
        path (tuple of str): Path of directory names to this subdirectory; ``()`` for
            the root (first) directory.
        file (:doc:`uproot.writing.writable.WritableFile`): Handle to the file in
            which this directory can be found.
        cascading (:doc:`uproot.writing._cascade.CascadingDirectory`): The low-level
            directory object.

    Represents a writable ``TDirectory`` from a ROOT file.

    Be careful not to confuse :doc:`uproot.writing.writable.WritableFile` and
    :doc:`uproot.writing.writable.WritableDirectory`: files are for modifying global
    information such as the TStreamerInfo and FreeSegments, whereas directories
    are for data in local hierarchies.

    A :doc:`uproot.writing.writable.WritableDirectory` is a Python ``MutableMapping``,
    which uses square bracket syntax to read, write, and delete objects:

    .. code-block:: python

        my_directory["histogram"]
        my_directory["histogram"] = np.histogram(...)
        del my_directory["histogram"]

    Objects in ROOT files also have "cycle numbers," which allow multiple versions
    of an object to exist with the same name. A cycle number may be specified after
    a semicolon for *reading* and *deleting* only:

    .. code-block:: python

        my_directory["histogram;2"]
        del my_directory["histogram;2"]

    When *writing*, cycle numbers are generated to avoid overwriting previous objects:

    .. code-block:: python

        my_directory["histogram"] = np.histogram(...)   # creates a new histogram
        my_directory["histogram"] = np.histogram(...)   # creates another histogram

    Note that this is unlike a Python ``MutableMapping``, which would overwrite the
    object in the second assignment. However, it is the way ROOT I/O works; use ``del``
    to remove unwanted versions of objects.

    Any types of objects that can be read from a :doc:`uproot.reading.ReadOnlyDirectory`
    can be read from a :doc:`uproot.writing.writable.WritableDirectory` *except TTrees*. A
    TTree can only be read from a :doc:`uproot.reading.ReadOnlyDirectory` if it was
    created in this open file handle, and then it returns a :doc:`uproot.writing.writable.WritableTree`
    instead of the :doc:`uproot.behaviors.TTree.TTree` that you would get from a
    :doc:`uproot.reading.ReadOnlyDirectory`. Readable TTrees and writable TTrees are
    distinct, with separate sets of features.

    Note that subdirectories can be created by assigning to path names that include
    slashes:

    .. code-block:: python

        my_directory["subdir1/subdir2/new_object"] = new_object

    Subdirectories created this way will never be empty; to make an empty directory,
    use :ref:`uproot.writing.writable.WritableDirectory.mkdir`.

    Similarly, non-empty RNTuples can be created by assignment starting in Uproot
    v5.7.0 (see :doc:`uproot.writing.writable.WritableNTuple` for recognized
    RNTuple-like data), but empty RNTuples require the
    :ref:`uproot.writing.writable.WritableDirectory.mkrntuple` method.
    Writing a TTree requires the :ref:`uproot.writing.writable.WritableDirectory.mktree` method.
    """

    def __init__(self, path, file, cascading):
        self._path = path
        self._file = file
        self._cascading = cascading
        self._subdirs = {}

    def __repr__(self):
        return "<WritableDirectory {} at 0x{:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    @property
    def path(self):
        """
        Path of directory names to this subdirectory as a tuple of strings; e.g. ``()``
        for the root (first) directory.
        """
        return self._path

    @property
    def object_path(self):
        """
        Path of directory names to this subdirectory as a single string, delimited
        by slashes.
        """
        return "/".join(("", *self._path, "")).replace("//", "/")

    @property
    def file_path(self):
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file.file_path

    @property
    def file(self):
        """
        Handle to the :doc:`uproot.writing.writable.WritableDirectory` in which
        this directory can be found.
        """
        return self._file

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._file.close()

    @property
    def closed(self) -> bool:
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._file.closed

    def __enter__(self):
        self._file.sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.sink.__exit__(exception_type, exception_value, traceback)

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new objects added to the file.

        This property can be changed, which allows you to write different objects
        with different compression settings.
        """
        return self._file.compression

    @compression.setter
    def compression(self, value):
        self._file.compression = value

    def __len__(self):
        return self._cascading.data.num_keys + sum(
            len(self._subdir(x)) for x in self._cascading.data.dir_keys
        )

    def __contains__(self, where):
        if self._cascading.data.haskey(where):
            return True
        return any(where in self._subdir(x) for x in self._cascading.data.dir_keys)

    def __iter__(self):
        return self.iterkeys()

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()

    def keys(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names of objects directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in those names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names of the objects in this ``TDirectory`` as a list of
        strings.

        Note that this does not read any data from the file.
        """
        return list(
            self.iterkeys(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def values(
        self,
        *,
        recursive=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return objects directly accessible in this
                ``TDirectory``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns objects in this ``TDirectory`` as a list of
        :doc:`uproot.model.Model`.

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        return list(
            self.itervalues(
                recursive=recursive,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def items(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return (name, object) pairs directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns (name, object) pairs for objects in this ``TDirectory`` as a
        list of 2-tuples of (str, :doc:`uproot.model.Model`).

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        return list(
            self.iteritems(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def classnames(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names and classnames of objects
                directly accessible in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names and C++ (decoded) classnames of the objects in this
        ``TDirectory`` as a dict of str \u2192 str.

        Note that this does not read any data from the file.
        """
        return dict(
            self.iterclassnames(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def iterkeys(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names of objects directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in those names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names of the objects in this ``TDirectory`` as an iterator
        over strings.

        Note that this does not read any data from the file.
        """
        filter_name = uproot._util.regularize_filter(filter_name)
        filter_classname = uproot._util.regularize_filter(filter_classname)
        for keyname, cyclenum, classname in self._cascading.data.key_triples:
            if (filter_name is no_filter or filter_name(keyname)) and (
                filter_classname is no_filter or filter_classname(classname)
            ):
                if cycle:
                    yield f"{keyname};{cyclenum}"
                else:
                    yield keyname

            if recursive and classname in ("TDirectory", "TDirectoryFile"):
                for k1 in self._get(keyname, cyclenum).iterkeys(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=filter_name,
                    filter_classname=filter_classname,
                ):
                    k2 = f"{keyname}/{k1}"
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2

    def itervalues(
        self,
        *,
        recursive=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return objects directly accessible in this
                ``TDirectory``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns objects in this ``TDirectory`` as an iterator over
        :doc:`uproot.model.Model`.

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        for keyname in self.iterkeys(
            recursive=recursive,
            cycle=True,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            yield self[keyname]

    def iteritems(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return (name, object) pairs directly accessible
                in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns (name, object) pairs for objects in this ``TDirectory`` as an
        iterator over 2-tuples of (str, :doc:`uproot.model.Model`).

        Note that this reads all objects that are selected by ``filter_name``
        and ``filter_classname``.
        """
        for keyname in self.iterkeys(
            recursive=recursive,
            cycle=True,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            if not cycle:
                at = keyname.index(";")
                keyname = keyname[:at]  # noqa: PLW2901 (overwriting keyname)
            yield keyname, self[keyname]

    def iterclassnames(
        self,
        *,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        """
        Args:
            recursive (bool): If True, descend into any nested subdirectories.
                If False, only return the names and classnames of objects
                directly accessible in this ``TDirectory``.
            cycle (bool): If True, include the cycle numbers in the names.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.

        Returns the names and C++ (decoded) classnames of the objects in this
        ``TDirectory`` as an iterator of 2-tuples of (str, str).

        Note that this does not read any data from the file.
        """
        filter_name = uproot._util.regularize_filter(filter_name)
        filter_classname = uproot._util.regularize_filter(filter_classname)
        for keyname, cyclenum, classname in self._cascading.data.key_triples:
            if (filter_name is no_filter or filter_name(keyname)) and (
                filter_classname is no_filter or filter_classname(classname)
            ):
                if cycle:
                    yield f"{keyname};{cyclenum}", classname
                else:
                    yield keyname, classname

            if recursive and classname in ("TDirectory", "TDirectoryFile"):
                for k1, c1 in self._get(keyname, cyclenum).iterclassnames(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=filter_name,
                    filter_classname=filter_classname,
                ):
                    k2 = f"{keyname}/{k1}"
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, c1

    def _get_del_search(self, where, isget):
        if "/" in where or ":" in where:
            items = where.split("/")
            step = last = self

            for i, item in enumerate(items):
                if item != "":
                    if isinstance(step, WritableDirectory):
                        if ":" in item and not step._cascading.data.haskey(item):
                            raise uproot.KeyInFileError(
                                where,
                                because="TTrees in writable files can't be indexed by TBranch name",
                                file_path=self.file_path,
                            )
                        else:
                            last = step
                            step = step[item]

                    elif isinstance(step, WritableTree):
                        rest = items[i:]
                        if len(rest) != 0:
                            raise uproot.KeyInFileError(
                                where,
                                because="TTrees in writable files can't be indexed by TBranch name",
                                file_path=self.file_path,
                            )
                        return step

                    else:
                        raise uproot.KeyInFileError(
                            where,
                            because="/".join(items[:i]) + " is not a TDirectory",
                            keys=last._cascading.data.key_names,
                            file_path=self.file_path,
                        )
            return step

        else:
            if ";" in where:
                at = where.rindex(";")
                item, cycle = where[:at], where[at + 1 :]
                try:
                    cycle = int(cycle)
                except ValueError:
                    item, cycle = where, None
            else:
                item, cycle = where, None

            if isget:
                return self._get(item, cycle)
            else:
                return self._del(item, cycle)

    def __getitem__(self, where):
        if self._file.sink.closed:
            raise ValueError("cannot get data from a closed file")
        return self._get_del_search(where, True)

    def __setitem__(self, where, what):
        if self._file.sink.closed:
            raise ValueError("cannot write data to a closed file")
        self.update({where: what})

    def __delitem__(self, where):
        if self._file.sink.closed:
            raise ValueError("cannot delete data from a closed file")
        return self._get_del_search(where, False)

    def _get(self, name, cycle):
        key = self._cascading.data.get_key(name, cycle)
        if key is None:
            raise uproot.exceptions.KeyInFileError(
                name,
                cycle="any" if cycle is None else cycle,
                keys=self._cascading.data.key_names,
                file_path=self.file_path,
                object_path=self.object_path,
            )

        if key.classname.string in ("TDirectory", "TDirectoryFile"):
            return self._subdir(key)

        elif key.classname.string == "TTree":
            if self._file._has_tree(key.seek_location):
                return self._file._get_tree(key.seek_location)
            else:
                # load existing TTree and reconstruct cascade
                return self._load_existing_ttree(key)
        elif key.classname.string == "ROOT::RNTuple":
            if self._file._has_ntuple(key.seek_location):
                return self._file._get_ntuple(key.seek_location)
            else:
                return self._load_existing_ntuple(key)

        else:

            def get_chunk(start, stop):
                raw_bytes = self._file.sink.read(start, stop - start)
                return uproot.source.chunk.Chunk.wrap(
                    readforupdate, raw_bytes, start=start
                )

            readforupdate = uproot.writing._cascade._ReadForUpdate(
                self._file.file_path,
                self._file.uuid,
                get_chunk,
                self._file._cascading.tlist_of_streamers,
            )

            raw_bytes = self._file.sink.read(
                key.seek_location,
                key.num_bytes + key.compressed_bytes,
            )

            chunk = uproot.source.chunk.Chunk.wrap(readforupdate, raw_bytes)
            cursor = uproot.source.cursor.Cursor(0, origin=key.num_bytes)

            readonlykey = uproot.reading.ReadOnlyKey(
                chunk, cursor, {}, readforupdate, self, read_strings=True
            )

            return readonlykey.get()

    def _load_existing_ttree(self, key):
        """
        Loads an existing TTree from disk and reconstructs a writable
        :doc:`uproot.writing.writable.WritableTree` object with a proper
        cascade object, enabling extend via existing machinery.
        """
        import struct as _struct

        import uproot.writing._cascadetree as ct

        if self.file_path is None:
            raise TypeError(
                "uproot.update() on a file-like object does not support accessing "
                "existing TTrees; use uproot.update() with a file path instead."
            )

        name = key.name.string

        _dtype_to_struct = {
            "f4": "f",
            "f8": "d",
            "i4": "i",
            "i8": "q",
            "i2": "h",
            "i1": "b",
            "u4": "I",
            "u8": "Q",
            "u2": "H",
            "u1": "B",
        }

        # read using sink.read + Chunk.wrap + _ReadForUpdate (same as _get)
        # avoids loading entire file into memory
        self._file.sink.flush()

        def _get_chunk(start, stop):
            raw_bytes = self._file.sink.read(start, stop - start)
            return uproot.source.chunk.Chunk.wrap(
                _readforupdate, raw_bytes, start=start
            )

        _readforupdate = uproot.writing._cascade._ReadForUpdate(
            self._file.file_path,
            self._file.uuid,
            _get_chunk,
            self._file._cascading.tlist_of_streamers,
        )
        _readforupdate.options = dict(uproot.reading.open.defaults)
        _readforupdate.options["minimal_ttree_metadata"] = False

        _raw_bytes = self._file.sink.read(
            key.seek_location,
            key.num_bytes + key.compressed_bytes,
        )
        _chunk = uproot.source.chunk.Chunk.wrap(
            _readforupdate, _raw_bytes, start=key.seek_location
        )
        _cursor = uproot.source.cursor.Cursor(key.seek_location, origin=key.num_bytes)
        _readonlykey = uproot.reading.ReadOnlyKey(
            _chunk, _cursor, {}, _readforupdate, self, read_strings=True
        )
        tree = _readonlykey.get()
        branches = list(tree.branches)

        fEntries = tree.member("fEntries")

        # metadata_start (TTree-level) and, per branch, metadata_start and
        # basket_metadata_start are filled in below by casc._build_out(), which
        # walks this cascade's layout the same way write_anew does -- rather than
        # recovering them by searching the decompressed blob for the *current
        # value* of a nearby field, which breaks whenever that value is 0 (e.g. a
        # freshly mktree'd tree has fEntries == fTotBytes == fZipBytes == 0, and a
        # search for zero bytes matches arbitrary unrelated data).

        branch_data = []
        branch_lookup = {}
        _counter_names = {}
        for b in branches:
            if b.classname != "TBranch":
                # TBranchElement, TBranchObject, etc. use an on-disk layout (split
                # objects, embedded streamer info, header bytes in jagged baskets,
                # ...) that this cascade — which only knows how to write plain
                # TBranch/TLeaf — cannot reproduce. Leave them out of branch_data
                # rather than crash trying to read TBranch-only fields (e.g.
                # fMaximum, which TLeafElement doesn't have) or silently
                # mis-write their baskets.
                continue

            try:
                interpretation = b.interpretation
                if isinstance(interpretation, uproot.interpretation.jagged.AsJagged):
                    # numpy_dtype of the AsJagged interpretation itself is
                    # dtype('O'); the basket actually holds the *content*
                    # dtype (e.g. float32 for "var * float32")
                    dtype = interpretation.content.to_dtype.newbyteorder(">")
                elif isinstance(
                    interpretation, uproot.interpretation.strings.AsStrings
                ):
                    # matches the dtype mktree/extend use for a "string" branch
                    dtype = numpy.dtype(str).newbyteorder(">")
                else:
                    dtype = interpretation.numpy_dtype.newbyteorder(">")
            except AttributeError:
                # TBranchElement or other complex branch — skip
                continue

            if dtype.kind == "O":
                raise NotImplementedError(
                    f"branch {b.name!r} has interpretation {interpretation!r}, "
                    "which is not yet supported by uproot.update(); only "
                    "numeric and jagged-numeric branches can currently be "
                    "extended on an existing TTree opened with uproot.update()"
                )

            # a fixed-size array branch (e.g. "float[3]") has a numpy_dtype with
            # a subdtype/shape, like _branch_np splits for a freshly created branch
            if dtype.subdtype is not None:
                dtype, shape = dtype.subdtype
            else:
                shape = ()

            sc = _dtype_to_struct.get(dtype.kind + str(dtype.itemsize), "f")
            # detect counter branches (e.g. njets for jagged jets array)
            _leaves = b.member("fLeaves")
            _is_counter = bool(_leaves) and bool(_leaves[0].member("fIsRange"))
            _leafcount = _leaves[0].member("fLeafCount") if _leaves else None
            _counter_names[b.name] = (
                _leafcount.member("fName") if _leafcount is not None else None
            )
            bd = {
                "fName": b.name,
                "branch_type": dtype,
                "kind": "counter" if _is_counter else "normal",
                "counter": None,
                "dtype": dtype,
                "shape": shape,
                "fTitle": b.member("fTitle"),
                "compression": b.compression,
                "fBasketSize": b.member("fBasketSize"),
                "fEntryOffsetLen": b.member("fEntryOffsetLen"),
                "fOffset": b.member("fOffset"),
                "fSplitLevel": b.member("fSplitLevel"),
                "fFirstEntry": b.member("fFirstEntry"),
                "fTotBytes": b.member("fTotBytes"),
                "fZipBytes": b.member("fZipBytes"),
                "fBasketBytes": b.member("fBasketBytes").copy(),
                "fBasketEntry": b.member("fBasketEntry").copy(),
                "fBasketSeek": b.member("fBasketSeek").copy(),
                "arrays_write_start": b.member("fWriteBasket"),
                "arrays_write_stop": b.member("fWriteBasket"),
                # filled in below by casc._build_out()
                "metadata_start": None,
                "basket_metadata_start": None,
                "tleaf_reference_number": 0,
                "tleaf_maximum_value": (
                    int(b.member("fLeaves")[0].member("fMaximum"))
                    if b.member("fLeaves")
                    else 0
                ),
                "tleaf_special_struct": _struct.Struct(">" + sc + sc),
            }
            if dtype == ">U0":
                # the length of the longest string written so far; extend()
                # accumulates this as max(existing, new), so it must be
                # seeded from what's already on disk or a later extend with
                # only shorter strings would shrink fLen and corrupt reads
                bd["fLen"] = b.member("fLeaves")[0].member("fLen")
            branch_data.append(bd)
            # index into branch_data, not into the raw branches list: branches
            # skipped above (non-TBranch) mean the two can diverge
            branch_lookup[b.name] = len(branch_data) - 1

        # fix counter references for jagged branches
        #
        # fEntryOffsetLen > 0 alone isn't a reliable signal that a branch is
        # jagged-with-a-counter: a string branch also gets fEntryOffsetLen > 0
        # once it has data (its basket-internal offset table), but strings are
        # never counted by a separate branch. Without the dtype check, a string
        # branch could be wrongly paired with an unrelated same-named "n"+name
        # branch (e.g. string branch "id" and an unrelated int branch "nid"),
        # making extend()'s counter/no-counter branch treat it as jagged
        # numeric instead of a string.
        for bd in branch_data:
            if (
                bd.get("fEntryOffsetLen", 0) > 0
                and bd["counter"] is None
                and bd.get("dtype") != ">U0"
            ):
                counter_nm = _counter_names.get(bd["fName"]) or "n" + bd["fName"]
                counter_bd = next(
                    (x for x in branch_data if x["fName"] == counter_nm), None
                )
                if counter_bd is not None:
                    bd["counter"] = counter_bd

        # fWriteBasket/fMaxBaskets are only tracked once per tree (not per
        # branch) by this cascade, but nothing guarantees every branch in a
        # preexisting tree agrees on either -- ROOT commonly flushes a basket
        # once a branch's accumulated data exceeds fBasketSize, so branches
        # with different per-entry sizes accumulate baskets at different
        # rates even when filled together from the start. Detect that here
        # so extend()/add_branches() can refuse rather than silently use one
        # branch's basket bookkeeping for all of them.
        _supported_branches = [b for b in branches if b.classname == "TBranch"]
        _write_basket_values = {b.member("fWriteBasket") for b in _supported_branches}
        _max_basket_values = {b.member("fMaxBaskets") for b in _supported_branches}
        has_divergent_baskets = (
            len(_write_basket_values) > 1 or len(_max_basket_values) > 1
        )
        fWriteBasket = (
            _supported_branches[0].member("fWriteBasket") if _supported_branches else 0
        )
        metadata = {
            k: tree.member(k)
            for k in [
                "fTotBytes",
                "fZipBytes",
                "fSavedBytes",
                "fFlushedBytes",
                "fWeight",
                "fTimerInterval",
                "fScanField",
                "fUpdate",
                "fDefaultEntryOffsetLen",
                "fNClusterRange",
                "fMaxEntries",
                "fMaxEntryLoop",
                "fMaxVirtualSize",
                "fAutoSave",
                "fAutoFlush",
                "fEstimate",
            ]
        }

        dir_key = self._cascading.data.get_key(name, key.cycle)
        freesegments = self._file._cascading.freesegments

        casc = ct.Tree.__new__(ct.Tree)
        casc._directory = self._cascading
        casc._name = name
        casc._title = tree.title
        casc._freesegments = freesegments
        casc._branch_data = branch_data
        casc._branch_lookup = branch_lookup
        casc._has_unsupported_branches = len(branch_data) != len(branches)
        casc._has_divergent_baskets = has_divergent_baskets
        # metadata_start/basket_metadata_start were just derived structurally,
        # assuming this tree is laid out the way Uproot's own writer lays one
        # out. That's only actually true once Uproot itself has written this
        # tree at least once; a preexisting tree may be ROOT-written, with a
        # different byte layout write_updates() would patch at the wrong
        # offsets. extend() checks this and does one full write_anew() before
        # its first incremental patch to establish Uproot's own layout for
        # real. add_branches() already always calls write_anew() regardless,
        # so it clears this flag itself once it does.
        casc._needs_relocation_before_extend = True
        casc._basket_capacity = (
            _supported_branches[0].member("fMaxBaskets") if _supported_branches else 10
        )
        casc._resize_factor = 10.0
        casc._counter_name = lambda counted: "n" + counted
        casc._field_name = None
        casc._num_baskets = fWriteBasket
        casc._num_entries = fEntries
        casc._metadata = metadata
        casc._key = dir_key

        # recovers casc._metadata_start and each branch's metadata_start /
        # basket_metadata_start / tleaf_reference_number by walking the same
        # layout write_anew would emit for this tree, rather than searching the
        # blob for byte patterns; the returned chunks are discarded, nothing is
        # written to the file
        casc._build_out()

        path = (*self._path, name)
        writable_tree = WritableTree(path, self._file, casc)
        self._file._trees[key.seek_location] = writable_tree
        return writable_tree

    def _read_ntuple_envelope(self, key):
        """
        Deserializes the RNTuple envelope object at `key`, using
        :doc:`uproot.writing._cascade._ReadForUpdate` to avoid loading the
        entire file into memory, and using the key's absolute file position
        (rather than 0) as the chunk/cursor origin, since RNTuple's own
        internal envelope offsets (e.g. anchor_location) are threaded through
        in absolute terms.

        Shared by :ref:`_load_existing_ntuple` (the first read of an RNTuple
        reopened via ``uproot.update()``) and by
        :ref:`uproot.writing.writable.WritableNTuple.add_fields`'s reload of
        the footer/page-list state it just wrote — both need exactly this,
        so this factors out what used to be two copies of the same ~25 lines.

        Args:
            key: The ROOT key object pointing to the RNTuple in the file.

        Returns:
            The deserialized ``Model_RNTuple`` (read-side) object.
        """
        self._file.sink.flush()

        def _get_chunk(start, stop):
            raw_bytes = self._file.sink.read(start, stop - start)
            return uproot.source.chunk.Chunk.wrap(readforupdate, raw_bytes, start=start)

        readforupdate = uproot.writing._cascade._ReadForUpdate(
            self._file.file_path,
            self._file.uuid,
            _get_chunk,
            self._file._cascading.tlist_of_streamers,
        )
        readforupdate.options = dict(uproot.reading.open.defaults)

        raw_bytes = self._file.sink.read(
            key.seek_location,
            key.num_bytes + key.compressed_bytes,
        )
        chunk = uproot.source.chunk.Chunk.wrap(
            readforupdate, raw_bytes, start=key.seek_location
        )
        cursor = uproot.source.cursor.Cursor(key.seek_location, origin=key.num_bytes)
        readonlykey = uproot.reading.ReadOnlyKey(
            chunk, cursor, {}, readforupdate, self, read_strings=True
        )
        existing = readonlykey.get()
        _ = existing.keys()
        return existing

    def _load_existing_ntuple(self, key):
        """
        Loads an existing RNTuple from disk, registering a new writable
        :doc:`uproot.writing.writable.WritableNTuple` handle to it.

        Args:
            key: The ROOT key object pointing to the RNTuple in the file.

        Returns:
            :doc:`uproot.writing.writable.WritableNTuple`
        """
        cascading, column_encoding_error = self._load_ntuple_cascading(key)
        path = (*self._path, key.name.string)
        writable_ntuple = WritableNTuple(path, self._file, cascading)
        writable_ntuple._column_encoding_error = column_encoding_error
        self._file._new_ntuple(writable_ntuple)
        return writable_ntuple

    def _load_ntuple_cascading(self, key):
        """
        Loads an existing RNTuple from disk and reconstructs a writable
        :doc:`uproot.writing.writable.WritableNTuple` object from it.

        This is called when accessing a preexisting RNTuple via
        ``f["name"]`` in update mode. Reads the existing metadata
        (anchor, header, footer, page lists) and sets up the in-memory
        state needed for subsequent ``extend`` or ``add_fields`` calls.

        Args:
            key: The ROOT key object pointing to the RNTuple in the file.

        Returns:
            The :doc:`uproot.writing._cascadentuple.NTuple` and the error
            message from checking its column encodings, or None.
        """

        import uproot.writing._cascade as casc
        import uproot.writing._cascadentuple as cnt

        _rblob_key_size = uproot.reading._key_format_big.size + 8

        name = key.name.string
        if self.file_path is None:
            raise TypeError(
                "uproot.update() on a file-like object does not support accessing "
                "existing RNTuples; use uproot.update() with a file path instead."
            )

        existing = self._read_ntuple_envelope(key)
        full_akform, _ = existing.to_akform()
        if cnt._form_contains_union(full_akform):
            raise NotImplementedError(
                f"RNTuple {name!r} has a variant (union-typed) field, which "
                "uproot.update() does not currently support accessing at all "
                "(neither reading nor writing); use uproot.open() instead to "
                "read it"
            )
        am = existing._ntuple.all_members
        anchor_location = key.seek_location + key.num_bytes
        num_entries = existing.num_entries
        existing_footer = existing._footer
        existing_page_list_envelopes = existing.page_list_envelopes
        existing_field_records = existing._ntuple.field_records

        header = cnt.NTuple_Header(
            None, existing.name, existing._header.ntuple_description, full_akform
        )
        header._checksum = existing._header.checksum

        # check column records match existing — store mismatch for later error
        # (e.g. ROOT writes split-encoded columns, uproot writes unsplit)
        existing_col_records = existing.column_records
        new_col_records = header._column_records
        _column_encoding_error = None
        if len(existing_col_records) != len(new_col_records):
            _column_encoding_error = (
                f"cannot extend: existing RNTuple has {len(existing_col_records)} columns "
                f"but reconstructed header has {len(new_col_records)}; "
                f"schema mismatch — this RNTuple may use column encodings uproot cannot write"
            )
        else:
            for i, (existing_cr, new_cr) in enumerate(
                zip(existing_col_records, new_col_records, strict=False)
            ):
                if (
                    existing_cr.type != new_cr.type_num
                    or existing_cr.nbits != new_cr.bits_on_disk
                    or existing_cr.field_id != new_cr.field_id
                ):
                    _column_encoding_error = (
                        f"cannot extend: column {i} type mismatch — "
                        f"existing column has type={existing_cr.type}, nbits={existing_cr.nbits} "
                        f"but uproot would write type={new_cr.type_num}, nbits={new_cr.bits_on_disk}. "
                        f"This RNTuple uses column encodings (e.g. split encoding) that uproot cannot write."
                    )
                    break

        footer = cnt.NTuple_Footer(None, header._checksum)

        for cg in existing_footer.cluster_group_records:
            locator = cnt.NTuple_Locator(
                cg.page_list_link.locator.num_bytes, cg.page_list_link.locator.offset
            )
            envlink = cnt.NTuple_EnvLink(cg.page_list_link.env_uncomp_size, locator)
            footer.cluster_group_record_frames.append(
                cnt.NTuple_ClusterGroupRecord(
                    cg.min_entry_num, cg.entry_span, cg.num_clusters, envlink
                )
            )
        # copy extension field and column records from existing footer
        for fr in existing_footer.extension_links.field_records:
            new_field = cnt.NTuple_Field_Description(
                fr.parent_field_id,
                fr.struct_role,
                fr.field_name,
                fr.type_name,
                field_description=fr.field_desc,
            )
            footer.extension_field_record_frames.append(new_field)
        for cr in existing_footer.extension_links.column_records:
            new_col = cnt.NTuple_Column_Description(
                cr.type,
                cr.nbits,
                cr.field_id,
                int(cr.flags),
                cr.repr_idx,
                first_element_index=cr.first_element_index,
            )
            footer.extension_column_record_frames.append(new_col)
        anchor = cnt.NTuple_Anchor(
            anchor_location,
            am["fVersionEpoch"],
            am["fVersionMajor"],
            am["fVersionMinor"],
            am["fVersionPatch"],
            am["fSeekHeader"],
            am["fNBytesHeader"],
            am["fLenHeader"],
            am["fSeekFooter"],
            am["fNBytesFooter"],
            am["fLenFooter"],
            am["fMaxKeySize"],
        )
        ntuple_cascading = cnt.NTuple(
            self._cascading,
            full_akform,
            self._cascading._freesegments,
            header,
            footer,
            [],
            anchor,
        )
        ntuple_cascading._key = key
        ntuple_cascading._header_key = casc.Key(
            am["fSeekHeader"] - _rblob_key_size,
            am["fLenHeader"],
            am["fNBytesHeader"],
            casc.String(None, "RBlob"),
            casc.String(None, ""),
            casc.String(None, ""),
            1,
            100,
            am["fSeekHeader"],
            big=True,
        )
        ntuple_cascading._footer_key = casc.Key(
            am["fSeekFooter"] - _rblob_key_size,
            am["fLenFooter"],
            am["fNBytesFooter"],
            casc.String(None, "RBlob"),
            casc.String(None, ""),
            casc.String(None, ""),
            1,
            100,
            am["fSeekFooter"],
            big=True,
        )
        ntuple_cascading._num_entries = num_entries
        num_columns = len(existing._header.column_records) + len(
            existing._footer.extension_links.column_records
        )
        # recover per-column element counts from existing page lists
        # for jagged fields, data columns advance by elements not entries
        column_counts = []
        for cg_idx, _cg in enumerate(existing_footer.cluster_group_records):
            ple = existing_page_list_envelopes[cg_idx]
            if not column_counts:
                column_counts = [0] * num_columns
            # a cluster group can hold more than one cluster (e.g. a ROOT-written
            # file); summing only pagelinklist[0] would silently undercount every
            # column's element offset for any cluster after the first, corrupting
            # the next page written for that column (uproot currently only ever
            # writes one cluster per group, so this hasn't been reachable yet)
            for cluster_col_pages in ple.pagelinklist:
                for col_idx, col_pages in enumerate(cluster_col_pages):
                    column_counts[col_idx] += sum(
                        p.num_elements for p in col_pages.pages
                    )
        if not column_counts:
            column_counts = [num_entries] * num_columns
        ntuple_cascading._column_counts = numpy.array(column_counts, dtype=int)
        ntuple_cascading._existing_footer = existing_footer
        ntuple_cascading._existing_page_list_envelopes = existing_page_list_envelopes
        ntuple_cascading._existing_field_records = existing_field_records

        return ntuple_cascading, _column_encoding_error

    def _del(self, name, cycle):
        key = self._cascading.data.get_key(name, cycle)
        if key is None:
            raise uproot.exceptions.KeyInFileError(
                name,
                cycle="any" if cycle is None else cycle,
                keys=self._cascading.data.key_names,
                file_path=self.file_path,
                object_path=self.object_path,
            )
        for handles in (self._file._trees, self._file._ntuples):
            handle = handles.pop(key.seek_location, None)
            if handle is not None:
                handle._deleted = True
        start = key.seek_location
        stop = start + key.num_bytes + key.compressed_bytes
        self._cascading.freesegments.release(start, stop)

        self._cascading._data.remove_key(key)
        self._cascading.header.modified_on = datetime.datetime.now()

        self._cascading.write(self._file.sink)
        self._file.sink.set_file_length(self._cascading.freesegments.fileheader.end)
        self._file.sink.flush()

    def _subdir(self, key):
        name = key.name.string

        if name in self._subdirs:
            sub = self._subdirs[name]
            for tree in self._file._trees.values():
                if (
                    tree._cascading.directory.key.location
                    == sub._cascading.key.location
                    and tree._cascading.directory is not sub._cascading
                ):
                    self._subdirs[name] = WritableDirectory(
                        (*self._path, name), self._file, tree._cascading.directory
                    )
                    break

        if name not in self._subdirs:
            raw_bytes = self._file.sink.read(
                key.seek_location,
                key.num_bytes + uproot.reading._directory_format_big.size + 18,
            )
            directory_key = uproot.writing._cascade.Key.deserialize(
                raw_bytes, key.seek_location, self._file.sink.in_path
            )
            position = key.seek_location + directory_key.num_bytes

            directory_header = uproot.writing._cascade.DirectoryHeader.deserialize(
                raw_bytes[position - key.seek_location :],
                position,
                self._file.sink.in_path,
            )
            assert directory_header.begin_location == key.seek_location

            # # FIXME: why was this here?
            # assert (
            #     directory_header.parent_location
            #     == self._file._cascading.fileheader.begin
            # )

            if directory_header.data_num_bytes == 0:
                directory_datakey = uproot.writing._cascade.Key(
                    None,
                    None,
                    None,
                    uproot.writing._cascade.String(None, "TDirectory"),
                    uproot.writing._cascade.String(None, name),
                    uproot.writing._cascade.String(None, name),
                    directory_key.cycle,
                    directory_header.parent_location,
                    None,
                )

                requested_num_bytes = (
                    directory_datakey.num_bytes + self._file._initial_directory_bytes
                )
                directory_datakey.location = self._cascading.freesegments.allocate(
                    requested_num_bytes
                )
                might_be_slightly_more = (
                    requested_num_bytes - directory_datakey.num_bytes
                )
                directory_data = uproot.writing._cascade.DirectoryData(
                    directory_datakey.location + directory_datakey.num_bytes,
                    might_be_slightly_more,
                    [],
                )

                directory_datakey.uncompressed_bytes = directory_data.allocation
                directory_datakey.compressed_bytes = (
                    directory_datakey.uncompressed_bytes
                )

                subdirectory = uproot.writing._cascade.SubDirectory(
                    directory_key,
                    directory_header,
                    directory_datakey,
                    directory_data,
                    self._cascading,
                    self._cascading.freesegments,
                )

                directory_header.data_location = directory_datakey.location
                directory_header.data_num_bytes = (
                    directory_datakey.num_bytes + directory_data.allocation
                )

                subdirectory.write(self._file.sink)

                self._file.sink.set_file_length(
                    self._cascading.freesegments.fileheader.end
                )
                self._file.sink.flush()

                self._subdirs[name] = WritableDirectory(
                    (*self._path, name), self._file, subdirectory
                )

            else:
                raw_bytes = self._file.sink.read(
                    directory_header.data_location, directory_header.data_num_bytes
                )

                directory_datakey = uproot.writing._cascade.Key.deserialize(
                    raw_bytes, directory_header.data_location, self._file.sink.in_path
                )
                directory_data = uproot.writing._cascade.DirectoryData.deserialize(
                    raw_bytes[directory_datakey.num_bytes :],
                    directory_header.data_location + directory_datakey.num_bytes,
                    self._file.sink.in_path,
                )

                subdirectory = uproot.writing._cascade.SubDirectory(
                    directory_key,
                    directory_header,
                    directory_datakey,
                    directory_data,
                    self._cascading,
                    self._cascading.freesegments,
                )

                self._subdirs[name] = WritableDirectory(
                    (*self._path, name), self._file, subdirectory
                )

        return self._subdirs[name]

    def mkdir(self, name, *, initial_directory_bytes=None):
        """
        Args:
            name (str): Name of the new subdirectory.
            initial_directory_bytes (None or int): Number of bytes to allocate
                for the new directory, so that TKeys can be added to it without
                immediately needing to rewrite the block. If None, the
                :doc:`uproot.writing.writable.WritableFile`'s value is used.

        Creates an empty subdirectory in this directory.

        Note that subdirectories can be created by assigning to path names that
        include slashes:

        .. code-block:: python

            my_directory["subdir1/subdir2/new_object"] = new_object

        but subdirectories created this way will never be empty. Use this method
        to make an empty directory or to control directory parameters.
        """
        if self._file.sink.closed:
            raise ValueError("cannot create a TDirectory in a closed file")

        stripped = name.strip("/")
        try:
            at = stripped.index("/")
        except ValueError:
            head, tail = stripped, None
        else:
            head, tail = stripped[:at], stripped[at + 1 :]

        key = self._cascading.data.get_key(head)
        if key is None:
            if initial_directory_bytes is None:
                initial_directory_bytes = self._file.initial_directory_bytes
            directory = WritableDirectory(
                (*self._path, head),
                self._file,
                self._cascading.add_directory(
                    self._file.sink,
                    head,
                    initial_directory_bytes,
                    self._file.uuid_function(),
                ),
            )
            self._subdirs[head] = directory

        elif key.classname.string not in ("TDirectory", "TDirectoryFile"):
            raise TypeError(
                f"""cannot make a directory named {name!r} because a {key.classname.string} already has that name
in file {self.file_path} in directory {self.path}"""
            )

        else:
            directory = self._subdir(key)

        if tail is None:
            return directory

        else:
            return directory.mkdir(tail)

    def mktree(
        self,
        name,
        branch_types_or_data,
        title="",
        *,
        counter_name=lambda counted: "n" + counted,
        field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner,
        initial_basket_capacity=10,
        resize_factor=10.0,
    ):
        """
        Args:
            name (str): Name of the new TTree.
            branch_types_or_data (dict of str \u2192 NumPy dtype/Awkward type,
                or dict of str \u2192 data to be written in the TBranch): Name
                and type specification for the TBranches. If the values are not valid
                type specifications, they are assumed to be the actual data to be written.
            title (str): Title for the new TTree.
            counter_name (callable of str \u2192 str): Function to generate counter-TBranch
                names for Awkward Arrays of variable-length lists. Arrays whose
                counters get the same name share one counter TBranch. A counter
                replaces a scalar integer TBranch of the same name with an
                ``int32`` one; a name collision with any other TBranch raises
                a ValueError.
            field_name (callable of str \u2192 str): Function to generate TBranch
                names for columns of an Awkward record array or a Pandas DataFrame.
            initial_basket_capacity (int): Number of TBaskets that can be written to the
                TTree without rewriting the TTree metadata to make room.
            resize_factor (float): When the TTree metadata needs to be rewritten,
                this specifies how many more TBasket slots to allocate as a multiplicative
                factor.

        Creates an empty TTree in this directory.

        Note that starting in v5.7.0, Uproot uses RNTuples as the default format for writing
        data when using the dict-like assignment syntax. Writing a TTree requires using this
        method.
        """
        if self._file.sink.closed:
            raise ValueError("cannot create a TTree in a closed file")

        # If data is provided, create an empty TTree and then extend it
        branch_types_or_data = _regularize_input_type_to_dict(branch_types_or_data)
        if not _is_type_specification(branch_types_or_data):
            metadata, data = _unpack_metadata_and_arrays(branch_types_or_data)
            tree = self.mktree(
                name,
                metadata,
                title=title,
                counter_name=counter_name,
                field_name=field_name,
                initial_basket_capacity=initial_basket_capacity,
                resize_factor=resize_factor,
            )
            tree.extend(data)
            return tree

        branch_types = branch_types_or_data

        try:
            at = name.rindex("/")
        except ValueError:
            treename = name
            directory = self
        else:
            dirpath, treename = name[:at], name[at + 1 :]
            directory = self.mkdir(dirpath)

        path = (*directory._path, treename)

        tree = WritableTree(
            path,
            directory._file,
            directory._cascading.add_tree(
                directory._file.sink,
                treename,
                title,
                branch_types,
                counter_name,
                field_name,
                initial_basket_capacity,
                resize_factor,
            ),
        )
        directory._file._new_tree(tree)

        seen = set()
        streamers = []
        for model in (
            uproot.models.TLeaf.Model_TLeafB_v1,
            uproot.models.TLeaf.Model_TLeafS_v1,
            uproot.models.TLeaf.Model_TLeafI_v1,
            uproot.models.TLeaf.Model_TLeafL_v1,
            uproot.models.TLeaf.Model_TLeafF_v1,
            uproot.models.TLeaf.Model_TLeafD_v1,
            uproot.models.TLeaf.Model_TLeafC_v1,
            uproot.models.TLeaf.Model_TLeafO_v1,
            uproot.models.TBranch.Model_TBranch_v13,
            uproot.models.TTree.Model_TTree_v20,
        ):
            for rawstreamer in model.class_rawstreamers:
                classname_version = rawstreamer[-2], rawstreamer[-1]
                if classname_version not in seen:
                    seen.add(classname_version)
                    streamers.append(
                        uproot.writing._cascade.RawStreamerInfo(*rawstreamer)
                    )

        directory._file._cascading.streamers.update_streamers(
            directory._file.sink, streamers
        )

        return tree

    def mkrntuple(
        self,
        name,
        type_spec_or_data,
        description="",
    ):
        """
        Args:
            name (str): Name of the new RNTuple.
            type_spec_or_data (dict of str \u2192 NumPy dtype/Awkward type,
                Awkward RecordForm, or data in the form of a RecordArray, Pandas dataframe, or dict): Name
                and type specification for the fields. If a RecordForm is provided,
                the RNTuple will be empty. If a RecordArray is provided, the RNTuple
                will be initialized with the input data.
            description (str): Description for the new RNTuple.

        Creates an empty RNTuple in this directory.

        Note that starting in v5.7.0, non-empty RNTuples can be created by
        assigning RNTuple-like data to a directory:

        .. code-block:: python

            my_directory["ntuple"] = {"field1": np.array(...), "field2": ak.Array(...)}

        but RNTuples created this way will never be empty. Use this method
        to make an empty RNTuple or to control its parameters.
        """
        if self._file.sink.closed:
            raise ValueError("cannot create a RNTuple in a closed file")

        if _is_type_specification(type_spec_or_data):
            ak_form = _type_specification_to_awkward_form(type_spec_or_data)
            return self.mkrntuple(name, ak_form, description)

        type_spec_or_data = (
            uproot.writing._cascadentuple._regularize_input_type_to_awkward(
                type_spec_or_data
            )
        )
        if isinstance(type_spec_or_data, awkward.Array):
            form = type_spec_or_data.layout.form
            packed_form = uproot.writing._cascadentuple._to_packed_form(form)
            if not isinstance(packed_form, awkward.forms.RecordForm):
                raise TypeError(
                    f"Input Awkward array must be a RecordArray or reducible to such. Got array with form {form!r}."
                )
            ntuple = self.mkrntuple(name, packed_form, description)
            ntuple.extend(type_spec_or_data)
            return ntuple
        if isinstance(type_spec_or_data, awkward.forms.Form):
            packed_form = uproot.writing._cascadentuple._to_packed_form(
                type_spec_or_data
            )
            if not isinstance(packed_form, awkward.forms.RecordForm):
                raise TypeError(
                    f"Input Awkward form must be a RecordForm or reducible to such. Got {type_spec_or_data!r}."
                )
            type_spec_or_data = packed_form
        else:
            raise TypeError(
                "Input must be a type specification (in the form of an Awkward RecordForm, or a dict of str \u2192 NumPy dtype/Awkward type) "
                "or data (in the form of a high-level Awkward record array, Pandas dataframe, or dict). "
                f"Got {type(type_spec_or_data).__name__}."
            )

        # The rest assumes that type_spec_or_data is a RecordForm

        if description == "" and "__doc__" in type_spec_or_data.parameters:
            description = type_spec_or_data.parameters["__doc__"]

        try:
            at = name.rindex("/")
        except ValueError:
            treename = name
            directory = self
        else:
            dirpath, treename = name[:at], name[at + 1 :]
            directory = self.mkdir(dirpath)

        path = (*directory._path, treename)

        ntuple = WritableNTuple(
            path,
            directory._file,
            directory._cascading.add_rntuple(
                directory._file.sink,
                treename,
                description,
                type_spec_or_data,
            ),
        )
        directory._file._new_ntuple(ntuple)
        return ntuple

    def copy_from(
        self,
        source,
        *,
        filter_name=no_filter,
        filter_classname=no_filter,
        rename=no_rename,
        require_matches=True,
    ):
        """
        Args:
            source (:doc:`uproot.writing.writable.WritableDirectory` or :doc:`uproot.reading.ReadOnlyDirectory`): Directory from which to copy.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by name.
            filter_classname (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select keys by C++ (decoded) classname.
            rename (None, regex string in ``"/from/to/"`` syntax, dict of str \u2192 str, function of str \u2192 str, or iterable of the above): A
                function to convert old names into new names.
            require_matches (bool): If True and the filters do not match any data, raise
                a ``ValueError``.

        Bulk-copy method to copy data from one ROOT file to another without interpretation
        or even decompression/recompression.

        This method will likely have performance advantages over copying objects one
        at a time, in part because it avoids interpretation and decompression/recompression,
        and also because it collects TStreamerInfo from all of the data types and
        rewrites the output file's TStreamerInfo exactly once.
        """
        if isinstance(source, WritableDirectory):
            raise NotImplementedError(
                "copying from a WritableDirectory is not yet supported; open the "
                "'source' as a ReadOnlyDirectory (with uproot.open)"
            )
        elif not isinstance(source, uproot.reading.ReadOnlyDirectory):
            raise TypeError("'source' must be a TDirectory")

        old_names = source.keys(
            filter_name=filter_name, filter_classname=filter_classname, cycle=False
        )
        if len(old_names) == 0:
            if require_matches:
                raise ValueError(
                    f"""no objects found with names matching {filter_name!r}
in file {source.file_path} in directory {source.path}"""
                )
            else:
                return

        keys = [source.key(x) for x in old_names]

        for key in keys:
            if key.fClassName == "TTree" or key.fClassName.split("::")[-1] == "RNTuple":
                raise NotImplementedError(
                    f"copy_from cannot copy {key.fClassName} objects yet"
                )

        rename = uproot._util.regularize_rename(rename)
        new_names = [rename(x) for x in old_names]

        notifications = queue.Queue()
        ranges = {}
        for new_name, old_key in zip(new_names, keys, strict=True):
            if old_key.fClassName not in ("TDirectory", "TDirectoryFile"):
                start = old_key.data_cursor.index
                stop = start + old_key.data_compressed_bytes
                ranges[start, stop] = new_name, old_key

        source.file.source.chunks(list(ranges), notifications=notifications)

        classversion_pairs = set()
        for classname in {x.fClassName for x in keys}:
            for streamer in source.file.streamers_named(classname):
                batch = []
                streamer._dependencies(source.file.streamers, batch)
                classversion_pairs.update(batch)

        streamers = [source.file.streamer_named(c, v) for c, v in classversion_pairs]

        self._file._cascading.streamers.update_streamers(self._file.sink, streamers)

        new_dirs = {}
        for new_name, old_key in zip(new_names, keys, strict=True):
            classname = old_key.fClassName
            path = new_name.strip("/").split("/")
            if classname not in ("TDirectory", "TDirectoryFile"):
                path = path[:-1]
            path = "/".join(path)
            if path not in new_dirs:
                new_dirs[path] = 4
            new_dirs[path] += (
                uproot.reading._key_format_big.size
                + 5
                + len(old_key.fClassName)
                + 5
                + len(old_key.fName)
                + 5
                + len(old_key.fTitle)
            )

        for name, allocation in new_dirs.items():
            self.mkdir(
                name,
                initial_directory_bytes=max(
                    self._file.initial_directory_bytes, allocation
                ),
            )

        for _ in range(len(ranges)):
            chunk = notifications.get()
            assert isinstance(chunk, uproot.source.chunk.Chunk)

            raw_data = chunk.raw_data.tobytes()

            new_name, old_key = ranges[chunk.start, chunk.stop]
            path = new_name.strip("/").split("/")
            directory = self
            for item in path[:-1]:
                directory = directory[item]

            directory._cascading.add_object(
                self._file.sink,
                old_key.fClassName,
                path[-1],
                old_key.fTitle,
                raw_data,
                old_key.data_uncompressed_bytes,
            )

    def update(self, pairs=None, **more_pairs):
        """
        Args:
            pairs (dict or pairs of str \u2192 writable data): Names and data to write.
            more_pairs (dict or pairs of str \u2192 writable data): More names and data to write.

        Bulk-update function, like assignment, but it collects TStreamerInfo for a single
        update.
        """
        streamers = []
        if pairs is not None:
            if hasattr(pairs, "keys"):
                all_pairs = itertools.chain(
                    ((k, pairs[k]) for k in pairs.keys()), more_pairs.items()
                )
            else:
                all_pairs = itertools.chain(pairs, more_pairs.items())
        else:
            all_pairs = more_pairs.items()

        for k, v in all_pairs:
            fullpath = k.strip("/").split("/")
            path, name = fullpath[:-1], fullpath[-1]

            if len(path) != 0:
                self.mkdir(
                    "/".join(path),
                    initial_directory_bytes=self._file.initial_directory_bytes,
                )

            directory = self
            for item in path:
                directory = directory[item]

            uproot.writing.identify.add_to_directory(v, name, directory, streamers)
        self._file._cascading.streamers.update_streamers(self._file.sink, streamers)


def _is_jagged(data):
    if isinstance(data, awkward.Array):
        return data.ndim > 1 and not data.layout.purelist_isregular
    if isinstance(data, (list, tuple)):
        try:
            return numpy.asarray(data).dtype.kind == "O"
        except ValueError:
            return True
    return False


class WritableTree:
    """
    Args:
        path (tuple of str): Path of directory names to this TTree.
        file (:doc:`uproot.writing.writable.WritableFile`): Handle to the file in
            which this TTree can be found.
        cascading (:doc:`uproot.writing._cascadetree.Tree`): The low-level
            directory object.

    Represents a writable ``TTree`` from a ROOT file.

    This object can be created using the :ref:`uproot.writing.writable.WritableDirectory.mktree` method. For instance:

    .. code-block:: python

        my_directory.mktree("tree1", {"branch1": np.array(...), "branch2": ak.Array(...)})
        my_directory.mktree("tree2", numpy_structured_array)
        my_directory.mktree("tree3", awkward_record_array)
        my_directory.mktree("tree4", pandas_dataframe)

    Recognized data types:

    * dict of NumPy arrays (flat, multidimensional, and/or structured), Awkward Arrays containing one level of variable-length lists and/or one level of records, or a Pandas DataFrame with a numeric index
    * a single NumPy structured array (one level deep)
    * a single Awkward Array containing one level of variable-length lists and/or one level of records
    * a single Pandas DataFrame with a numeric index

    The arrays may have different types, but their lengths must be identical, at
    least in the first dimension (i.e. number of entries).

    If the Awkward Array contains variable-length lists (i.e. it is "jagged"), a
    counter TBranch will be created along with the data TBranch. ROOT needs the
    counter TBranch to quantify the size of the variable-size arrays. Combining
    Awkward Arrays with the same number of nested items using
    `ak.zip <https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html>`__ prevents
    a proliferation of counter TBranches:

    .. code-block:: python

        my_directory.mktree("tree5", ak.zip({"branch1": array1, "branch2": array2, "branch3": array3}))

    would produce only one counter TBranch.

    The :doc:`uproot.writing.writable.WritableDirectory.mktree` method allows you to separate
    the process of creating the TTree metadata from filling the first TBasket:

    .. code-block:: python

        my_directory.mktree("tree6", {"branch1": numpy_dtype, "branch2": awkward_type})

    The :doc:`uproot.writing.writable.WritableDirectory.mktree` method can also control the
    title of the TTree and the rules used to name counter TBranches and nested field TBranches.

    The ``numpy_dtype`` is any data that NumPy recognizes as a ``np.dtype``, and the
    ``awkward_type`` is an `ak.types.Type <https://awkward-array.readthedocs.io/en/latest/ak.types.Type.html>`__ from
    `ak.type <https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html>`__ or
    a string in that form, such as ``"var * float64"`` for variable-length doubles.

    TBaskets can be added to each TBranch using the :ref:`uproot.writing.writable.WritableTree.extend`
    method:

    .. code-block:: python

        my_directory["tree6"].extend({"branch1": another_numpy_array,
                                      "branch2": another_awkward_array})

    Be sure to make these extensions as large as is feasible within memory constraints,
    because a ROOT file full of small TBaskets is bloated (larger than it needs to be)
    and slow to read (especially for Uproot, but also for ROOT).

    For instance, if you want to write a million events and have enough memory
    available to do that 100 thousand events at a time (total of 10 TBaskets),
    then do so. Filling the TTree a hundred events at a time (total of 10000 TBaskets)
    would be considerably slower for writing and reading, and the file would be much
    larger than it could otherwise be, even with compression.
    """

    def __init__(self, path, file, cascading):
        self._path = path
        self._file = file
        self._cascading = cascading
        self._deleted = False

    def __repr__(self):
        return "<WritableTree {} at 0x{:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    def _check_not_deleted(self):
        if self._deleted:
            raise ValueError("this TTree was deleted")

    @property
    def path(self):
        """
        Path of directory names to this TTree as a tuple of strings.
        """
        return self._path

    @property
    def object_path(self) -> str:
        """
        Path of directory names to this TTree as a single string, delimited by
        slashes.
        """
        return "/".join(("", *self._path, "")).replace("//", "/")

    @property
    def file_path(self) -> str | None:
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file.file_path

    @property
    def file(self):
        """
        Handle to the :doc:`uproot.writing.writable.WritableDirectory` in which
        this directory can be found.
        """
        return self._file

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._file.close()

    @property
    def closed(self) -> bool:
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._file.closed

    def __enter__(self):
        self._file.sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.sink.__exit__(exception_type, exception_value, traceback)

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new TBaskets added to the TTree.

        This property can be changed and doesn't have to be the same as the compression
        of the file, which allows you to write different objects with different
        compression settings.

        The following are equivalent:

        .. code-block:: python

            my_directory["tree"]["branch1"].compression = uproot.ZLIB(1)
            my_directory["tree"]["branch2"].compression = uproot.LZMA(9)

        and

        .. code-block:: python

            my_directory["tree"].compression = {"branch1": uproot.ZLIB(1),
                                                "branch2": uproot.LZMA(9)}
        """
        out = {}
        last = None
        for datum in self._cascading._branch_data:
            if datum["kind"] != "record":
                last = out[datum["fName"]] = datum["compression"]
        if all(x == last for x in out.values()):
            return last
        else:
            return out

    @compression.setter
    def compression(self, value):
        if value is None or isinstance(value, uproot.compression.Compression):
            for datum in self._cascading._branch_data:
                if datum["kind"] != "record":
                    datum["compression"] = value

        elif (
            isinstance(value, Mapping)
            and all(
                isinstance(k, str)
                and (v is None or isinstance(v, uproot.compression.Compression))
                for k, v in value.items()
            )
            and all(
                datum["fName"] in value
                for datum in self._cascading._branch_data
                if datum["kind"] != "record"
            )
            and len(value)
            == len(
                [
                    datum
                    for datum in self._cascading._branch_data
                    if datum["kind"] != "record"
                ]
            )
        ):
            for datum in self._cascading._branch_data:
                if datum["kind"] != "record":
                    datum["compression"] = value[datum["fName"]]

        else:
            raise TypeError(
                "compression must be None, a uproot.compression.Compression object, like uproot.ZLIB(4) or uproot.ZSTD(0), or a mapping of branch names to such objects"
            )

    def __getitem__(self, where):
        for datum in self._cascading._branch_data:
            if datum["kind"] != "record" and datum["fName"] == where:
                return WritableBranch(self, datum)
        else:
            raise uproot.KeyInFileError(
                where,
                because="no such branch in writable tree",
                file_path=self.file_path,
            )

    @property
    def num_entries(self) -> int:
        """
        The number of entries accumulated so far.
        """
        return self._cascading.num_entries

    @property
    def num_baskets(self) -> int:
        """
        The number of TBaskets accumulated so far.
        """
        return self._cascading.num_baskets

    def add_branches(self, branches):
        """
        Args:
            branches (dict of str -> array): Names and data of new branches.

        Adds new branches to this TTree in-place. Only the new branch data and
        an updated TTree header are written; existing data is never touched.

        .. code-block:: python

            with uproot.update("file.root") as f:
                f["tree"].add_branches({"new_branch": np.ones(100, dtype=np.float32)})

        .. note::

            Only trees whose branches are all plain ``TBranch`` (not
            ``TBranchElement``, e.g. split objects) are supported. Rewriting
            the branch listing for a tree with ``TBranchElement`` branches
            would silently drop them, so this raises ``NotImplementedError``
            for such trees instead. This mirrors the same restriction on
            :ref:`uproot.writing.writable.WritableTree.extend`.

            Jagged (variable-length) branches are not yet supported and raise
            ``NotImplementedError``.
        """
        if self._file.sink.closed:
            raise ValueError("cannot modify a TTree in a closed file")
        self._check_not_deleted()

        # validate all branches have same length as existing tree; self._cascading
        # is already fully flushed and current on disk by the end of every
        # extend()/add_branches() call (the same trust extend() itself places in
        # it), so there's no need to re-read and re-parse the whole tree from
        # disk here
        if self._cascading is None:
            raise RuntimeError(
                "_cascading is None — this should not happen; please report this bug"
            )
        casc = self._cascading
        num_entries = casc._num_entries

        for branch_name, branch_data in branches.items():
            if _is_jagged(branch_data):
                raise NotImplementedError(
                    f"branch {branch_name!r} is a jagged array; add_branches does not "
                    "yet support jagged branches (which require a counter branch)"
                )
            arr = numpy.asarray(branch_data)
            if len(arr) != num_entries:
                raise ValueError(
                    f"branch {branch_name!r} has {len(arr)} entries but TTree has "
                    f"{num_entries} entries; all new branches must match the tree length"
                )
            if branch_name in casc._branch_lookup:
                raise ValueError(f"branch {branch_name!r} already exists in this TTree")

        # _load_existing_ttree leaves TBranchElement (and other non-TBranch)
        # branches out of casc._branch_data entirely, so rewriting the branch
        # listing from casc._branch_data alone (as write_anew does) would
        # silently drop them from the file
        if casc._has_unsupported_branches:
            raise NotImplementedError(
                "add_branches for files with TBranchElement (or other non-TBranch) "
                "branches is not yet supported via the cascade approach"
            )
        if casc._num_baskets > 1:
            # add_branches always back-fills a new branch with exactly one
            # basket spanning every existing entry (see below), regardless of
            # how many baskets the tree's other branches already have. On a
            # tree with only one basket that coincidentally matches, but on
            # any tree with more, it would silently create the very
            # divergent-basket-count state extend() already refuses to run on
            # -- turning every future extend() on this tree into a permanent
            # NotImplementedError, with no warning at add_branches() time.
            raise NotImplementedError(
                "add_branches for a TTree that already has more than one basket "
                "is not yet supported via the cascade approach: the new branch "
                "would only ever get one basket, permanently diverging from the "
                "other branches' basket counts and blocking any future extend() "
                "on this tree"
            )

        # add new branch dicts to cascade
        compression = casc._freesegments.fileheader.compression
        for branch_name, branch_data in branches.items():
            arr = numpy.asarray(branch_data)
            if arr.dtype.kind == "O":
                raise TypeError(
                    f"branch {branch_name!r} has object dtype — only simple numeric "
                    f"types are supported for add_branches"
                )
            dtype = arr.dtype.newbyteorder(">")
            new_bd = casc._branch_np(branch_name, arr.dtype, dtype)
            new_bd["compression"] = compression
            casc._branch_data.append(new_bd)
            casc._branch_lookup[branch_name] = len(casc._branch_data) - 1

        # rewrite TTree metadata blob with new branches included -- this relocates
        # the tree (frees its old space, allocates new space for the larger
        # blob), so the WritableFile._trees cache needs to move with it, the same
        # way extend()'s own basket-capacity-expansion relocation does via
        # file._move_tree(); otherwise a stale entry at the old location lingers
        # in that cache indefinitely
        oldloc = casc._key.seek_location
        casc.write_anew(self._file.sink)
        self._file._move_tree(oldloc, casc._key.seek_location)
        # this write_anew() just established Uproot's own canonical layout for
        # real, so a subsequent extend() on this same cascade doesn't need to
        # redundantly do it again
        casc._needs_relocation_before_extend = False

        # write one basket per new branch, matching the existing branches (the
        # guard above ensures they have at most one); an empty tree has no
        # baskets yet, and the next extend() writes basket 0 for every branch
        if casc._num_baskets == 1:
            for branch_name, branch_data in branches.items():
                arr = numpy.asarray(branch_data).astype(
                    casc._branch_data[casc._branch_lookup[branch_name]]["dtype"]
                )
                totbytes, zipbytes, location = casc.write_np_basket(
                    self._file.sink, branch_name, compression, arr
                )
                datum = casc._branch_data[casc._branch_lookup[branch_name]]
                datum["fTotBytes"] += totbytes
                datum["fZipBytes"] += zipbytes
                datum["fBasketBytes"][0] = zipbytes
                datum["fBasketSeek"][0] = location
                datum["fBasketEntry"][1] = num_entries
                datum["arrays_write_start"] = 0
                datum["arrays_write_stop"] = 1
                casc._metadata["fTotBytes"] += totbytes
                casc._metadata["fZipBytes"] += zipbytes

        casc.write_updates(self._file.sink)
        self._file.sink.flush()

    def extend(self, data, *, accept_new_fields=False):
        """
        Args:
            data (dict of str \u2192 arrays): More array data to add to the TTree.
            accept_new_fields (bool): If True, new fields in data are automatically added
                with zeros back-filled for existing entries before extending.

        This method adds data to an existing TTree, whether it was created through
        assignment or :doc:`uproot.writing.writable.WritableDirectory.mktree`.

        The arrays must be a dict, but the values of the dict can be any of the
        array/DataFrame types described in :doc:`uproot.writing.writable.WritableTree`.
        However, these types must be compatible with the established TBranch
        types, the dict must contain a key for every TBranch, and the arrays must have
        the same lengths (in their first dimension).

        For example,

        .. code-block:: python

            my_directory.mktree("tree6", {"branch1": numpy_dtype, "branch2": awkward_type})

            my_directory["tree6"].extend({"branch1": another_numpy_array,
                                          "branch2": another_awkward_array})

        .. warning::

            **As a word of warning,** be sure that each call to :ref:`uproot.writing.writable.WritableTree.extend` includes at least 100 kB per branch/array. (NumPy and Awkward Arrays have an `nbytes <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html>`__ property; you want at least ``100000`` per array.) If you ask Uproot to write very small TBaskets, it will spend more time working on TBasket overhead than actually writing data. The absolute worst case is one-entry-per-:ref:`uproot.writing.writable.WritableTree.extend`. See `#428 (comment) <https://github.com/scikit-hep/uproot5/pull/428#issuecomment-908703486>`__.

        .. note::

            On a tree reopened with :doc:`uproot.writing.writable.WritableDirectory.update`,
            only trees whose branches are all plain ``TBranch`` (not
            ``TBranchElement``, e.g. split objects) are supported, and only
            trees whose branches all agree on basket count/capacity (true by
            construction for a tree Uproot itself wrote and has not touched
            with :ref:`uproot.writing.writable.WritableTree.add_branches`
            since). Either case raises ``NotImplementedError`` rather than
            desynchronizing or corrupting the tree's branches.
        """
        self._check_not_deleted()
        if self._cascading is None:
            raise RuntimeError(
                "_cascading is None — this should not happen; please report this bug"
            )
        if self._cascading._has_unsupported_branches:
            # _load_existing_ttree leaves TBranchElement (and other non-TBranch)
            # branches out of _branch_data entirely, so extend() would only add
            # entries to the branches it knows about, desynchronizing entry
            # counts across the tree's branches
            raise NotImplementedError(
                "extend for files with TBranchElement (or other non-TBranch) "
                "branches is not yet supported via the cascade approach"
            )
        if self._cascading._has_divergent_baskets:
            # this cascade tracks one fWriteBasket/fMaxBaskets pair per tree, not
            # per branch, but nothing guarantees every branch in a preexisting
            # tree agrees on either (e.g. a ROOT-written tree whose branches
            # have different per-entry sizes, and so flush baskets at different
            # rates); extend() would write every branch's new basket at an
            # offset computed from one branch's basket count, corrupting the
            # others (or crash outright if that count exceeds another
            # branch's actual basket capacity)
            raise NotImplementedError(
                "extend for a TTree whose branches do not all have the same "
                "number of baskets / basket capacity is not yet supported via "
                "the cascade approach"
            )
        # validate branches
        # get user-facing branch names (exclude auto-generated counter and record parent branches)
        # get record parent names to exclude their sub-fields
        _record_names = {
            bd.get("name", "")
            for bd in self._cascading._branch_data
            if bd.get("kind") == "record"
        }
        _user_branch_names = [
            bd["fName"]
            for bd in self._cascading._branch_data
            if bd.get("kind") not in ("counter", "record")
            and "fName" in bd
            and not any(
                bd["fName"].startswith(rn + "_") or bd["fName"].startswith(rn + ".")
                for rn in _record_names
                if rn
            )
        ]
        # include counter branches that user explicitly provides in data
        _counter_branch_names = [
            bd["fName"]
            for bd in self._cascading._branch_data
            if bd.get("kind") == "counter" and "fName" in bd
        ]
        # check if data looks like a flat dict of branch arrays (branch name ->
        # array), as opposed to e.g. a single top-level record array. Every
        # awkward Array has a `.fields` attribute regardless of its type --
        # it's just empty for a non-record (flat or jagged) array -- so
        # `hasattr(v, "fields")` alone can't tell a record apart from a plain
        # jagged/flat awkward Array value; checking that `.fields` is
        # non-empty is what actually means "this value is itself a record".
        # Getting this wrong skipped the new-field detection and
        # accept_new_fields auto-add logic below entirely whenever any value
        # in data happened to be an awkward Array, jagged or not -- new
        # awkward-typed fields fell straight through to the low-level
        # extend(), which doesn't know about accept_new_fields and raised a
        # confusing "does not correspond to any branch" naming its
        # auto-generated counter branch instead.
        _data_is_flat_dict = isinstance(data, dict) and all(
            not (hasattr(v, "fields") and v.fields) for v in data.values()
        )
        if isinstance(data, dict) and _data_is_flat_dict:
            # opportunistically flatten a nested-record-shaped value under a
            # key that isn't a recognized branch/record-parent/counter name,
            # mirroring mktree's default field_name convention ("outer_inner").
            # A tree reopened via uproot.update() has no persisted
            # record-parent metadata to recognize: _load_existing_ttree only
            # ever reconstructs "counter"/"normal" branches, because the
            # grouping itself was never written to disk -- only the
            # already-flattened leaf branches were -- so there is nothing on
            # disk to reconstruct it *from* (and field_name is user-
            # customizable besides, so branch names alone can't reveal it
            # reliably). Rather than guess at the file's original structure,
            # expand a dict-valued key only when doing so lines up exactly
            # with real existing branches -- this makes the common
            # (default-separator) case round-trip the same way extend()
            # behaves in the same creation session, without ever silently
            # misinterpreting an unrelated key.
            _known_top_level = (
                set(_user_branch_names) | _record_names | set(_counter_branch_names)
            )
            for _k in list(data.keys()):
                if _k in _known_top_level:
                    continue
                _v = data[_k]
                if isinstance(_v, Mapping):
                    _flattened = {f"{_k}_{_sub}": _subv for _sub, _subv in _v.items()}
                    if _flattened and set(_flattened) <= set(_user_branch_names):
                        data = {**data, **_flattened}
                        del data[_k]

            existing_names = _user_branch_names
            # also get record parent names that the user passes as dicts
            _record_parent_names = {
                bd.get("name")
                for bd in self._cascading._branch_data
                if bd.get("kind") == "record" and bd.get("name")
            }
            new_fields = {
                k: v
                for k, v in data.items()
                if k not in existing_names
                and k not in _record_parent_names
                and k not in _counter_branch_names
            }
            # skip counter branches not provided by user (auto-generated)
            missing = [
                b
                for b in existing_names
                if b not in data and b not in _counter_branch_names
            ]
            # add counter branches to existing_names if user provides them
            existing_names = _user_branch_names + [
                c for c in _counter_branch_names if c in data
            ]
            if missing:
                raise ValueError(
                    f"'extend' must fill every branch with the same number of entries; missing: {missing}"
                )
            if new_fields:
                if not accept_new_fields:
                    raise ValueError(
                        "'extend' was given data that do not correspond to any branch: "
                        + repr(next(iter(new_fields)))
                    )
                # add_branches() (used below to back-fill zeros for new fields)
                # only creates simple scalar TBranch, with no counter-branch
                # support -- it cannot create a new jagged branch. Left
                # unchecked, numpy.asarray(v) below raises a confusing,
                # awkward-internal "cannot convert to RegularArray" error for
                # any jagged new field, instead of saying plainly that this
                # isn't supported.
                _jagged_new_fields = [k for k, v in new_fields.items() if _is_jagged(v)]
                if _jagged_new_fields:
                    raise NotImplementedError(
                        f"accept_new_fields does not yet support jagged (variable-length) "
                        f"new fields: {_jagged_new_fields}. Call add_branches() with a "
                        f"scalar zero-filled backfill for this field first (which itself "
                        f"does not yet support jagged branches either), or restructure the "
                        f"data to not introduce a jagged field via extend()."
                    )
                zeros = {
                    k: numpy.zeros(
                        self._cascading._num_entries, dtype=numpy.asarray(v).dtype
                    )
                    for k, v in new_fields.items()
                }
                self.add_branches(zeros)
        self._cascading.extend(self._file, self._file.sink, data)

    def show(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        name_width=20,
        typename_width=24,
        interpretation_width=30,
        stream=sys.stdout,
    ):
        """
        Opens the TTree for reading and calls :doc:`uproot.behaviors.TBranch.HasBranches.show`
        on it (follow link for documentation of this method).
        """
        uproot.open(self._file.sink._file)[self.object_path].show(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
            name_width=name_width,
            typename_width=typename_width,
            interpretation_width=interpretation_width,
            stream=stream,
        )


class WritableBranch:
    """
    Represents a TBranch from a :doc:`uproot.writing.writable.WritableTree`.

    This object exists only to be able to assign compression settings differently
    on each TBranch:

    .. code-block:: python

        my_directory["tree"]["branch1"].compression = uproot.ZLIB(1)
        my_directory["tree"]["branch2"].compression = uproot.LZMA(9)

    Note that compression settings on all TBranches can be set through
    :doc:`uproot.writing.writable.WritableTree.compression`:

    .. code-block:: python

        my_directory["tree"].compression = {"branch1": uproot.ZLIB(1),
                                            "branch2": uproot.LZMA(9)}
    """

    def __init__(self, tree, datum):
        self._tree = tree
        self._datum = datum

    def __repr__(self):
        return "<WritableBranch {} in {} at 0x{:012x}>".format(
            repr(self._datum["fName"]), repr("/" + "/".join(self._tree.path)), id(self)
        )

    @property
    def type(self):
        """
        The type used to initialize this TBranch.
        """
        return self._datum["branch_type"]

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        for new TBaskets added to the TBranch.

        This property can be changed and doesn't have to be the same as the compression
        of the file or the rest of the TTree, which allows you to write different objects
        with different compression settings.

        The following are equivalent:

        .. code-block:: python

            my_directory["tree"]["branch1"].compression = uproot.ZLIB(1)
            my_directory["tree"]["branch2"].compression = uproot.LZMA(9)

        and

        .. code-block:: python

            my_directory["tree"].compression = {"branch1": uproot.ZLIB(1),
                                                "branch2": uproot.LZMA(9)}
        """
        return self._datum["compression"]

    @compression.setter
    def compression(self, value):
        if value is None or isinstance(value, uproot.compression.Compression):
            self._datum["compression"] = value
        else:
            raise TypeError(
                "compression must be None or a uproot.compression.Compression object, like uproot.ZLIB(4) or uproot.ZSTD(0)"
            )


class WritableNTuple:
    """
    Args:
        path (tuple of str): Path of directory names to this RNTuple.
        file (:doc:`uproot.writing.writable.WritableFile`): Handle to the file in
            which this RNTuple can be found.
        cascading (:doc:`uproot.writing._cascadentuple.NTuple`): The low-level
            directory object.

    Represents a writable ``RNTuple`` from a ROOT file.

    Assigning data to a directory creates an RNTuple object by default starting in Uproot v5.7.0.
    This creates the RNTuple object with all of its metadata and fills it with
    the contents of the arrays in one step. To separate the process of creating the
    RNTuple metadata from filling the first cluster, use the
    :doc:`uproot.writing.writable.WritableDirectory.mkrntuple` method:

    .. code-block:: python

        my_directory.mkrntuple("tuple6", {"branch1": numpy_dtype, "branch2": awkward_type})

    The ``numpy_dtype`` is any data that NumPy recognizes as a ``np.dtype``, and the
    ``awkward_type`` is an `ak.types.Type <https://awkward-array.readthedocs.io/en/latest/ak.types.Type.html>`__ from
    `ak.type <https://awkward-array.readthedocs.io/en/latest/_auto/ak.type.html>`__ or
    a string in that form, such as ``"var * float64"`` for variable-length doubles.

    RNTuple can be extended using :ref:`uproot.writing.writable.WritableNTuple.extend`
    method:

    .. code-block:: python

        my_directory["tuple6"].extend({"branch1": another_numpy_array,
                                      "branch2": another_awkward_array})

    Be sure to make these extensions as large as is feasible within memory constraints,
    because a ROOT file full of small clusters is bloated (larger than it needs to be)
    and slow to read (especially for Uproot, but also for ROOT).

    For instance, if you want to write a million events and have enough memory
    available to do that 100 thousand events at a time (total of 10 clusters),
    then do so. Filling the RNTuple a hundred events at a time (total of 10000 clusters)
    would be considerably slower for writing and reading, and the file would be much
    larger than it could otherwise be, even with compression.
    """

    def __init__(self, path, file, cascading):
        self._path = path
        self._file = file
        self._cascading = cascading
        self._column_encoding_error = None
        self._deleted = False

    def __repr__(self):
        return "<WritableNTuple {} at 0x{:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    def _check_not_deleted(self):
        if self._deleted:
            raise ValueError("this RNTuple was deleted")

    def _existing_key(self):
        """
        Resolves this RNTuple's :doc:`uproot.writing._cascade.Key` by walking
        down to its containing directory (not always the file's root
        directory — this RNTuple may live in a subdirectory) and looking up
        its name and cycle there.
        """
        directory = self._file.root_directory
        for part in self._path[:-1]:
            directory = directory[part]
        cycle = self._cascading.key.cycle
        key = directory._cascading.data.get_key(self._path[-1], cycle)
        if key is None:
            raise uproot.KeyInFileError(
                self._path[-1],
                cycle=cycle,
                keys=directory._cascading.data.key_names,
                file_path=self.file_path,
                object_path=self.object_path,
            )
        return directory, key

    def _reload_cascading(self):
        """
        Replaces this RNTuple's cascading with one read back from disk, keeping
        this object as the file's one handle to it.
        """
        directory, key = self._existing_key()
        self._cascading, self._column_encoding_error = directory._load_ntuple_cascading(
            key
        )
        self._file._new_ntuple(self)

    @property
    def path(self):
        """
        Path of directory names to this RNTuple as a tuple of strings.
        """
        return self._path

    @property
    def object_path(self):
        """
        Path of directory names to this RNTuple as a single string, delimited by
        slashes.
        """
        return "/".join(("", *self._path, "")).replace("//", "/")

    @property
    def file_path(self) -> str | None:
        """
        Filesystem path of the open file, or None if using a file-like object.
        """
        return self._file.file_path

    @property
    def file(self):
        """
        Handle to the :doc:`uproot.writing.writable.WritableDirectory` in which
        this directory can be found.
        """
        return self._file

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, objects cannot be read from or written to the file.
        """
        self._file.close()

    @property
    def closed(self) -> bool:
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :ref:`uproot.writing.writable.WritableFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, objects cannot be read from or written to the file.
        """
        return self._file.closed

    def __enter__(self):
        self._file.sink.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.sink.__exit__(exception_type, exception_value, traceback)

    @property
    def compression(self):
        """
        Compression algorithm and level (:doc:`uproot.compression.Compression` or None)
        used when writing pages to this RNTuple.

        RNTuple compression is file-wide and is taken from the file header; individual
        per-column compression is not yet supported.
        """
        return self._cascading._freesegments.fileheader.compression

    @compression.setter
    def compression(self, value):
        raise NotImplementedError(
            "per-RNTuple compression is not yet supported; set compression on the file instead"
        )

    @property
    def num_entries(self) -> int:
        """
        The number of entries accumulated so far.
        """
        return self._cascading.num_entries

    def extend(self, data, accept_new_fields=False):
        """
        Args:
            data (dict of str \u2192 arrays): More array data to add to the RNTuple.
            accept_new_fields (bool): If False (default), raises ValueError if
                data contains fields not already in the RNTuple, forcing the user
                to call :ref:`uproot.writing.writable.WritableNTuple.add_fields`
                first. If True, new fields are automatically added back-filled
                with zeros for existing entries, then extended with the provided
                values for new entries.

        This method adds data to an existing RNTuple, whether it was created through
        assignment or :doc:`uproot.writing.writable.WritableDirectory.mkrntuple`.

        The arrays must be a dict, but the values of the dict can be any of the
        array/DataFrame types described in :doc:`uproot.writing.writable.WritableTree`.
        However, these types must be compatible with the established TBranch
        types, the dict must contain a key for every TBranch, and the arrays must have
        the same lengths (in their first dimension).

        For example,

        .. code-block:: python

            with uproot.update("file.root") as f:
                f["mytuple"].extend({
                    "x": np.array([4, 5, 6]),
                    "y": np.array([40, 50, 60]),
                })

            # automatically add new field and extend
            with uproot.update("file.root") as f:
                f["mytuple"].extend({"x": np.array([7, 8]), "z": np.array([70, 80])},
                                    accept_new_fields=True)

        .. warning::

            **As a word of warning,** be sure that each call to :ref:`uproot.writing.writable.WritableNTuple.extend` includes at least 100 kB per branch/array. (NumPy and Awkward Arrays have an `nbytes <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html>`__ property; you want at least ``100000`` per array.) If you ask Uproot to write very small TBaskets, it will spend more time working on TBasket overhead than actually writing data. The absolute worst case is one-entry-per-:ref:`uproot.writing.writable.WritableTree.extend`. See `#428 (comment) <https://github.com/scikit-hep/uproot5/pull/428#issuecomment-908703486>`__.

        .. note::

            Extending an RNTuple opened with :doc:`uproot.writing.writable.WritableDirectory.update`
            only works if every column uses an encoding Uproot itself writes.
            RNTuples written by ROOT commonly use column encodings (e.g. split
            encoding) that Uproot does not yet write, and are rejected with a
            ``ValueError`` rather than risk writing an inconsistent file. In
            practice, ``extend``/``add_fields`` after reopening a file only
            work reliably on RNTuples that Uproot itself wrote.
        """
        self._check_not_deleted()
        if self._column_encoding_error is not None:
            raise ValueError(self._column_encoding_error)
        # if _column_counts has more columns than _column_keys, reload cascading
        # this happens after add_fields adds extension columns
        if len(self._cascading._column_counts) > len(
            self._cascading._header._column_keys
        ):
            self._reload_cascading()

        if isinstance(data, dict):
            existing_keys = set(self._cascading._header._akform.fields)
            if hasattr(self._cascading, "_existing_field_records"):
                existing_keys.update(
                    fr.field_name for fr in self._cascading._existing_field_records
                )
            new_field_names = {k for k in data.keys() if k not in existing_keys}
            if new_field_names and not accept_new_fields:
                raise ValueError(
                    f"Data contains fields not in this RNTuple: {sorted(new_field_names)}. "
                    f"Call add_fields() first, or pass accept_new_fields=True to add them automatically."
                )
            elif new_field_names and accept_new_fields:
                self.add_fields(
                    {
                        k: numpy.asarray(
                            awkward.flatten(awkward.Array(data[k]), axis=None)
                        ).dtype
                        for k in new_field_names
                    }
                )
                self._reload_cascading()

        self._cascading.extend(self._file, self._file.sink, data)

    def add_fields(self, new_fields):
        """
        Args:
            new_fields (dict of str -> numpy dtype): New field names and types.
                Only scalar numeric types are accepted (e.g. ``np.int32``,
                ``np.float64``). Variable-length or nested types are not supported.

        Adds new fields to this RNTuple, back-filled with zeros for existing entries.
        Only top-level fields or subfields of existing untyped record fields are supported.

        Note: when using ``accept_new_fields=True`` in :meth:`extend`, new fields
        must be flat (scalar) types. Jagged or nested new fields will fail the
        form compatibility check in ``extend``.

        Raises:
            TypeError: if a field type is not a simple scalar numeric type.
            ValueError: if a field already exists, if the subfield parent is not
                found or is a typed struct (C++ typename), if the file was
                opened without a file path (file-like objects are not supported),
                or if the RNTuple uses column encodings Uproot does not write
                (see the note below).

        For example,

        .. code-block:: python

            with uproot.update("file.root") as f:
                f["mytuple"].add_fields({"z": np.int32, "w": np.float32})

        .. note::

            Like :ref:`uproot.writing.writable.WritableNTuple.extend`, this only
            works on an RNTuple opened with :doc:`uproot.writing.writable.WritableDirectory.update`
            if every column uses an encoding Uproot itself writes. RNTuples
            written by ROOT commonly use column encodings (e.g. split encoding)
            that Uproot does not yet write, and are rejected with a
            ``ValueError``. In practice, ``add_fields``/``extend`` after
            reopening a file only work reliably on RNTuples that Uproot itself
            wrote.
        """
        self._check_not_deleted()
        if self._column_encoding_error is not None:
            raise ValueError(self._column_encoding_error)

        import uproot.writing._cascadentuple as cnt

        if not hasattr(self._cascading, "_existing_footer"):
            # this RNTuple was created in this same session (e.g. via mkrntuple or
            # directory assignment) rather than loaded from an existing file by
            # _load_existing_ntuple, so it never got the _existing_* attributes
            # add_fields needs. Whatever has been written so far is already on
            # disk (mkrntuple/assignment/extend all flush immediately), so it can
            # be loaded the same way a preexisting RNTuple opened via
            # uproot.update() would be.
            self._reload_cascading()

        compression_code = cnt._compression_code(
            self._cascading._freesegments.fileheader.compression
        )
        num_entries = self._cascading._num_entries
        header = self._cascading._header
        footer = self._cascading._footer
        existing_footer = self._cascading._existing_footer
        existing_page_list_envelopes = self._cascading._existing_page_list_envelopes
        existing_field_records = self._cascading._existing_field_records

        next_field_id = len(existing_field_records)

        # a plain (non-dotted) new field always becomes a top-level field (see
        # the "else" branch below), so only collisions against other top-level
        # fields are real duplicates. Checking every field's bare name,
        # including nested ones, rejected legitimate top-level fields whenever
        # their name happened to match some unrelated nested field's name.
        # a field is top-level iff it's its own parent (self-referential); see
        # NTuple_Header._build_field_col_records, which sets parent_fid = the
        # field's own id when it's called without an explicit parent
        existing_root_field_names = {
            fr.field_name
            for i, fr in enumerate(existing_field_records)
            if fr.parent_field_id == i
        }
        for field_name in new_fields:
            if "." not in field_name and field_name in existing_root_field_names:
                raise ValueError(f"Field {field_name!r} already exists in this RNTuple")

        for field_name, field_dtype_raw in new_fields.items():
            ak_form = _type_specification_to_awkward_form(field_dtype_raw)
            if not isinstance(ak_form, awkward.forms.NumpyForm):
                raise TypeError(
                    f"add_fields only supports simple numeric types, got {field_dtype_raw!r}"
                )
            ak_primitive = ak_form.primitive
            type_name = cnt._ak_primitive_to_typename_dict[ak_primitive]
            type_num = cnt._ak_primitive_to_num_dict[ak_primitive]
            type_size = uproot.const.rntuple_col_num_to_size_dict[type_num]

            if "." in field_name:
                parts = field_name.split(".")
                actual_field_name = parts[-1]
                if len(parts) == 2:
                    # single-level ("parent.field"): match the immediate parent
                    # by bare name, at any nesting depth (backward compat) --
                    # but only when that name is unambiguous. Matching the
                    # first same-named field regardless of ambiguity used to
                    # silently misattach a field whenever two records shared a
                    # bare name at different nesting levels (e.g. a nested
                    # "b.a" and an unrelated top-level "a"), with no error.
                    matches = [
                        (i, fr)
                        for i, fr in enumerate(existing_field_records)
                        if fr.field_name == parts[0]
                    ]
                    if len(matches) == 0:
                        raise ValueError(
                            f"Field {parts[0]!r} not found in this RNTuple"
                        )
                    if len(matches) > 1:
                        raise ValueError(
                            f"Field {parts[0]!r} is ambiguous: {len(matches)} fields "
                            f"in this RNTuple have that name at different nesting "
                            f"levels. Use a fully-qualified dotted path (e.g. "
                            f"'grandparent.{parts[0]}.{parts[1]}') to disambiguate."
                        )
                    parent_field_id, fr = matches[0]
                else:
                    # multi-level: walk the full dotted path using the parent-id
                    # chain, one segment at a time
                    current_parent_id = None  # None means root
                    for part_idx, part in enumerate(parts[:-1]):
                        found = None
                        for i, fr in enumerate(existing_field_records):
                            # a field is top-level iff it's its own parent
                            # (self-referential); "parent_field_id == 0" alone
                            # is not a valid root signal -- it also matches any
                            # non-root field whose real parent happens to be at
                            # index 0, e.g. the second field of the first
                            # top-level record
                            is_root_field = fr.parent_field_id == i
                            if fr.field_name == part:
                                if current_parent_id is None and is_root_field:
                                    found = (i, fr)
                                    break
                                elif (
                                    current_parent_id is not None
                                    and fr.parent_field_id == current_parent_id
                                ):
                                    found = (i, fr)
                                    break
                        if found is None:
                            raise ValueError(
                                f"Field {'.'.join(parts[:part_idx+1])!r} not found in this RNTuple"
                            )
                        current_parent_id = found[0]
                        parent_field_id = found[0]
                        fr = found[1]
                if fr.type_name != "":
                    raise ValueError(
                        f"Field {'.'.join(parts[:-1])!r} has type {fr.type_name!r} and cannot be extended. "
                        f"Only untyped records (empty type_name) can have subfields added."
                    )
                if fr.struct_role != uproot.const.RNTupleFieldRole.RECORD:
                    raise ValueError(
                        f"Field {'.'.join(parts[:-1])!r} is not a record and cannot have subfields added."
                    )
                # the top-level duplicate check above only looks at root fields
                # and never matches a dotted key against a bare name, so a
                # subfield with the same name as an existing sibling under the
                # same parent went undetected -- silently writing a second
                # field record with the same (parent_field_id, field_name).
                # That duplicate isn't just cosmetic: it corrupts the RNTuple's
                # field/ancestor bookkeeping badly enough that even a plain
                # read of the file afterwards raises IndexError, and since the
                # footer already reached disk by the time that surfaces, the
                # file is left permanently unreadable.
                if any(
                    other.field_name == actual_field_name
                    and other.parent_field_id == parent_field_id
                    for other in existing_field_records
                ):
                    raise ValueError(
                        f"Field {field_name!r} already exists in this RNTuple"
                    )
            else:
                actual_field_name = field_name
                parent_field_id = next_field_id

            new_field = cnt.NTuple_Field_Description(
                parent_field_id,
                uproot.const.RNTupleFieldRole.LEAF,
                actual_field_name,
                type_name,
            )
            footer.extension_field_record_frames.append(new_field)
            # use deferred column — first_element_index marks where new data starts
            new_col = cnt.NTuple_Column_Description(
                type_num,
                type_size,
                next_field_id,
                0,
                0,
                first_element_index=num_entries,
            )
            footer.extension_column_record_frames.append(new_col)
            next_field_id += 1

        footer.cluster_group_record_frames = []
        for cg_idx, cg in enumerate(existing_footer.cluster_group_records):
            ple = existing_page_list_envelopes[cg_idx]
            all_cluster_page_data = []
            for _cluster_idx, cluster_col_pages in enumerate(ple.pagelinklist):
                new_cluster_page_data = []
                for col_pages in cluster_col_pages:
                    existing_pages = [
                        cnt.NTuple_PageDescription(
                            p.num_elements,
                            cnt.NTuple_Locator(p.locator.num_bytes, p.locator.offset),
                        )
                        for p in col_pages.pages
                    ]
                    new_cluster_page_data.append(
                        cnt.NTuple_ColumnPageListDescription(
                            existing_pages, col_pages.element_offset, compression_code
                        )
                    )
                # deferred columns: no pages needed for existing cluster groups
                all_cluster_page_data.append(new_cluster_page_data)
            cluster_summaries = [
                cnt.NTuple_ClusterSummary(s.num_first_entry, s.num_entries)
                for s in ple.cluster_summaries
            ]
            pagelistenv = cnt.NTuple_PageListEnvelope(
                header._checksum, cluster_summaries, all_cluster_page_data
            )
            pagelistenv_raw = pagelistenv.serialize()
            pagelistenv_key = self._cascading.add_rblob(
                self._file.sink, pagelistenv_raw, len(pagelistenv_raw)
            )
            pagelistenv_locator = cnt.NTuple_Locator(
                len(pagelistenv_raw),
                pagelistenv_key.location + pagelistenv_key.allocation,
            )
            pagelistenv_envlink = cnt.NTuple_EnvLink(
                len(pagelistenv_raw), pagelistenv_locator
            )
            footer.cluster_group_record_frames.append(
                cnt.NTuple_ClusterGroupRecord(
                    cg.min_entry_num,
                    cg.entry_span,
                    cg.num_clusters,
                    pagelistenv_envlink,
                )
            )

        footer_raw = footer.serialize()
        old_footer_key = self._cascading._footer_key
        new_footer_key = self._cascading.add_rblob(
            self._file.sink, footer_raw, len(footer_raw)
        )
        self._cascading._footer_key = new_footer_key
        self._cascading.sync(self._file.sink)
        self._cascading._anchor.seek_footer = (
            new_footer_key.location + new_footer_key.allocation
        )
        self._cascading._anchor.nbytes_footer = len(footer_raw)
        self._cascading._anchor.len_footer = len(footer_raw)
        anchor_raw = self._cascading._anchor.serialize()
        self._file.sink.write(self._cascading._anchor._location, anchor_raw)
        self._cascading._freesegments.release(
            old_footer_key.location,
            old_footer_key.location
            + old_footer_key.num_bytes
            + old_footer_key.compressed_bytes,
        )
        self._cascading.sync(self._file.sink)

        # _existing_field_records and _column_counts are updated directly from what
        # we just wrote, without going back to disk for them specifically.
        #
        # footer.extension_field_record_frames is cumulative across every add_fields()
        # call in this session (it's rewritten to disk in full each time, so it must
        # hold every extension field ever added, not just the ones from this call).
        # header._field_records is fixed at load time (the schema's fields before any
        # add_fields call in this session). Recombining those two — rather than the
        # already-updated `existing_field_records` from the start of this call, which
        # itself already contains earlier extension fields — avoids double-counting
        # every field added by a previous call.
        self._cascading._existing_field_records = list(header._field_records) + list(
            footer.extension_field_record_frames
        )
        self._cascading._column_counts = numpy.append(
            self._cascading._column_counts, numpy.zeros(len(new_fields), dtype=int)
        )
        # _existing_footer, _existing_page_list_envelopes, and _header._akform,
        # by contrast, DO require a full read back from disk here: they need to
        # be read-side model objects (Model_RNTuple_Footer, page-list envelopes),
        # not the write-side NTuple_Footer/etc. objects already held in this
        # method's local variables, and there's no cheaper way to produce those
        # without re-implementing a write-to-read-model adapter. This makes every
        # add_fields() call pay for re-parsing the whole footer and page-list
        # state, scaling with how much has already been written -- a real but
        # unoptimized cost, not a correctness issue.
        directory, _ntuple_key = self._existing_key()
        existing_reload = directory._read_ntuple_envelope(_ntuple_key)
        self._cascading._existing_footer = existing_reload._footer
        self._cascading._existing_page_list_envelopes = (
            existing_reload.page_list_envelopes
        )
        full_akform_reload, _ = existing_reload.to_akform()
        self._cascading._header._akform = full_akform_reload


def _is_type_specification(obj):
    to_check = [obj]
    while len(to_check) > 0:
        obj = to_check.pop()
        if isinstance(obj, Mapping):
            if all(isinstance(k, str) for k in obj.keys()):
                to_check.extend(obj.values())
                continue
            else:
                return False
        if not isinstance(
            obj,
            (
                numpy.dtype,
                awkward.types.Type,
                awkward.types.ArrayType,
                type,
                str,
                tuple,
            ),
        ):
            return False
        # for tuples and strings we need to make sure they actually specify a type and are not just data
        if isinstance(obj, tuple):
            try:
                numpy.dtype(obj)
            except (TypeError, ValueError):
                return False
            else:
                continue
        if isinstance(obj, str):
            try:
                numpy.dtype(obj)
            except (TypeError, ValueError):
                pass
            else:
                continue
            try:
                awkward.types.from_datashape(obj, highlevel=False)
            except Exception:
                pass
            else:
                continue
            return False
    return True


def _type_specification_to_awkward_form(obj):
    if isinstance(obj, awkward.forms.Form):
        return obj
    if isinstance(obj, (awkward.types.Type, awkward.types.ArrayType)):
        return awkward.forms.from_type(obj)
    if isinstance(obj, type):
        obj = numpy.dtype(obj)
        if obj == numpy.dtype("O"):
            raise TypeError(f"Cannot construct a NumPy dtype from {obj!r}.")
    if isinstance(obj, tuple):
        try:
            obj = numpy.dtype(obj)
        except (TypeError, ValueError):
            raise TypeError(
                f"Cannot construct a NumPy dtype from the tuple {obj!r}."
            ) from None
    if isinstance(obj, str):
        # First we try to interpret the string as a NumPy dtype
        # so we can try to convert it to a string Awkward understands
        try:
            dt = numpy.dtype(obj)
        except (TypeError, ValueError):
            pass
        else:
            obj = dt
    if isinstance(obj, numpy.dtype):
        obj = obj.newbyteorder("<")
        if obj.subdtype is None:
            field_shape = ()
        else:
            obj, field_shape = obj.subdtype
        dims = ""
        if len(field_shape) > 0:
            dims = dims + "".join(str(x) + " * " for x in field_shape)
        obj = f"{dims}{obj}"
    if isinstance(obj, str):
        try:
            return awkward.forms.from_type(
                awkward.types.from_datashape(obj, highlevel=False)
            )
        except Exception:
            raise TypeError(
                f"Cannot construct an Awkward Form from type specification {obj!r}"
            ) from None
    if isinstance(obj, Mapping):
        return awkward.forms.RecordForm(
            [_type_specification_to_awkward_form(v) for v in obj.values()],
            list(obj.keys()),
        )
    raise TypeError(
        f"Cannot construct an Awkward Form from {type(obj).__name__}. "
        f"Supported types: Form, Type, ArrayType, dtype, Mapping, str, tuple."
    )


def _regularize_input_type_to_dict(obj):
    if uproot._util.from_module(obj, "pandas"):
        import pandas

        if isinstance(
            obj, pandas.DataFrame
        ) and uproot._util.pandas_has_attr_is_numeric(pandas)(obj.index):
            obj = uproot.writing._cascadetree.dataframe_to_dict(obj)

    if uproot._util.from_module(obj, "awkward"):
        import awkward

        if isinstance(obj, awkward.Array):
            obj = {"": obj}

    if isinstance(obj, numpy.ndarray) and obj.dtype.fields is not None:
        obj = uproot.writing._cascadetree.recarray_to_dict(obj)

    return obj


def _unpack_metadata_and_arrays(obj):
    data = {}
    metadata = {}

    for branch_name, branch_array in obj.items():
        if uproot._util.from_module(branch_array, "pandas"):
            import pandas

            if isinstance(branch_array, pandas.DataFrame):
                branch_array = uproot.writing._cascadetree.dataframe_to_dict(  # noqa: PLW2901 (overwriting branch_array)
                    branch_array
                )

        if (
            isinstance(branch_array, numpy.ndarray)
            and branch_array.dtype.fields is not None
        ):
            branch_array = uproot.writing._cascadetree.recarray_to_dict(  # noqa: PLW2901 (overwriting branch_array)
                branch_array
            )

        if isinstance(branch_array, Mapping) and all(
            isinstance(x, str) for x in branch_array
        ):
            datum = {}
            metadatum = {}
            for kk, vv in branch_array.items():
                try:
                    vv = uproot._util.ensure_numpy(vv)  # noqa: PLW2901 (overwriting vv)
                except TypeError:
                    raise TypeError(
                        f"unrecognizable array type {type(branch_array)} associated with {branch_name!r}"
                    ) from None
                datum[kk] = vv
                branch_dtype = vv.dtype
                branch_shape = vv.shape[1:]
                if branch_shape != ():
                    branch_dtype = numpy.dtype((branch_dtype, branch_shape))
                metadatum[kk] = branch_dtype

            data[branch_name] = datum
            metadata[branch_name] = metadatum

        else:
            if uproot._util.from_module(branch_array, "awkward"):
                data[branch_name] = branch_array
                metadata[branch_name] = branch_array.type

            else:
                try:
                    branch_array = uproot._util.ensure_numpy(  # noqa: PLW2901 (overwriting branch_array)
                        branch_array
                    )
                except TypeError:
                    try:
                        branch_array = awkward.from_iter(  # noqa: PLW2901 (overwriting branch_array)
                            branch_array
                        )
                    except Exception:
                        raise TypeError(
                            f"unrecognizable array type {type(branch_array)} associated with {branch_name!r}"
                        ) from None
                    else:
                        data[branch_name] = branch_array
                        metadata[branch_name] = awkward.type(branch_array)

                else:
                    data[branch_name] = branch_array
                    branch_dtype = branch_array.dtype
                    branch_shape = branch_array.shape[1:]
                    if branch_shape != ():
                        branch_dtype = numpy.dtype((branch_dtype, branch_shape))
                    metadata[branch_name] = branch_dtype
    return metadata, data
