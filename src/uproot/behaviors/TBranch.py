# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines behaviors for :doc:`uproot.behaviors.TBranch.TBranch` and
:doc:`uproot.behaviors.TBranch.HasBranches` (both ``TBranch`` and
``TTree``).

Most of the functionality of TTree-reading is implemented here.

See :doc:`uproot.models.TBranch` for deserialization of the ``TBranch``
objects themselves.
"""

from __future__ import annotations

import queue
import re
import sys
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from keyword import iskeyword

import awkward
import numpy

import uproot
import uproot.interpretation.grouped
import uproot.language.python
import uproot.source.chunk
from uproot._util import get_ttree_form, no_filter, unset

np_uint8 = numpy.dtype("u1")


class _NoClose:
    def __init__(self, hasbranches):
        self.hasbranches = hasbranches

    def __enter__(self):
        return self.hasbranches

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __getattr__(self, attr):
        return getattr(self.hasbranches, attr)

    def __getitem__(self, where):
        return self.hasbranches[where]


def iterate(
    files,
    expressions=None,
    cut=None,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    filter_field=no_filter,
    aliases=None,
    language=uproot.language.python.python_language,
    step_size="100 MB",
    decompression_executor=None,
    interpretation_executor=None,
    library="ak",
    ak_add_doc=False,
    how=None,
    report=False,
    custom_classes=None,
    allow_missing=False,
    **options,
):
    """
    Args:
        files: See below.
        expressions (None, str, or list of str): Names of ``TBranches`` or
            aliases to convert to arrays or mathematical expressions of them.
            Uses the ``language`` to evaluate. If None, all ``TBranches``
            selected by the filters are included.
        cut (None or str): If not None, this expression filters all of the
            ``expressions``.
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
            filter to select ``TBranches`` using the full
            :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
            returns False or None, the ``TBranch`` is excluded; if the function
            returns True, it is included with its standard
            :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
            :doc:`uproot.interpretation.Interpretation`, this interpretation
            overrules the standard one. Only used for TTrees.
        filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
            filter to select ``RFields`` using the full
            :doc:`uproot.models.RNTuple.RField` object. If the function
            returns False or None, the ``RField`` is excluded; if the function
            returns True, it is included. Only used for RNTuples.
        aliases (None or dict of str \u2192 str): Mathematical expressions that
            can be used in ``expressions`` or other aliases (without cycles).
            Uses the ``language`` engine to evaluate. If None, only the
            :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
        language (:doc:`uproot.language.Language`): Language used to interpret
            the ``expressions`` and ``aliases``.
        step_size (int or str): If an integer, the maximum number of entries to
            include in each iteration step; if a string, the maximum memory size
            to include. The string must be a number followed by a memory unit,
            such as "100 MB".
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, a
            :doc:`uproot.source.futures.TrivialExecutor` is created.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, a :doc:`uproot.source.futures.TrivialExecutor`
            is created.
        library (str or :doc:`uproot.interpretation.library.Library`): The library
            that is used to represent arrays. Options are ``"np"`` for NumPy,
            ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
        ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
            to the Awkward ``__doc__`` parameter of the array.
            if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
            Awkward ``key`` parameter of the array.
        how (None, str, or container type): Library-dependent instructions
            for grouping. The only recognized container types are ``tuple``,
            ``list``, and ``dict``. Note that the container *type itself*
            must be passed as ``how``, not an instance of that type (i.e.
            ``how=tuple``, not ``how=()``).
            For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip`` to
            interleave data from compatible branches.
        report (bool): If True, this generator yields
            (arrays, :doc:`uproot.behaviors.TBranch.Report`) pairs; if False,
            it only yields arrays. The report has data about the ``TFile``,
            ``TTree``, and global and local entry ranges.
        custom_classes (None or dict): If a dict, override the classes from
            the :doc:`uproot.reading.ReadOnlyFile` or ``uproot.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        options: See below.

    Iterates through contiguous chunks of entries from a set of files.

    For example:

    .. code-block:: python

        >>> for array in uproot.iterate("files*.root:tree", ["x", "y"], step_size=100):
        ...     # each of the following have 100 entries
        ...     array["x"], array["y"]

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/ttree"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/ttree"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}``
    * already-open TTree objects.
    * iterables of the above.

    ..
      Future Dev Note: These shared reading options are manually copied across multiple files
      so they appear in `help` output of Python interpreter and the online documentation
      for the functions where they are used.

    Options (type; default):

    * handler (:doc:`uproot.source.chunk.Source` class; None): Class implementing reading from the data source.
      If None, deduced from input file type.
    * timeout (float for HTTP, int for XRootD; default defined by source implementation): The time in seconds
      to wait before giving up on the connection. Ignored for non-internet sources like local file paths.
    * max_num_elements (None or int; None): The maximum number of byte ranges requested in a single XRootD
      vector read. This does not limit the number of TTree or RNTuple entries read; pass ``entry_stop`` to
      ``arrays``, ``uproot.iterate``, or ``uproot.concatenate`` instead.
    * num_workers (int; 1): Number of tasks to spawn for reading, only used by some source types
    * use_threads (bool; False on the emscripten platform (i.e. in a web browser), else True):
      Use multi-threading when spawning workers.
    * num_fallback_workers (int; 10): Number of tasks to spawn for reading in fallback mode
      (for example, multi-threading requests instead of a multipart GET for an http source)
    * begin_chunk_size (memory_size; 403, the smallest a ROOT file can be): Size of first chunk that we attempt
      to read in bytes.
    * minimal_ttree_metadata (bool; True): Skip rarely used metadata and defer reading of embedded TBaskets
    * http_max_header_bytes (int; 21784): Maximum size of HTTP packet in bytes when the source is http

    See also :ref:`uproot.behaviors.TBranch.HasBranches.iterate` to iterate
    within a single file.

    Other file entry points:

    * :doc:`uproot.reading.open`: opens one file to read any of its objects.
    * :doc:`uproot.behaviors.TBranch.iterate` (this function): iterates through
      chunks of contiguous entries in ``TTrees``.
    * :doc:`uproot.behaviors.TBranch.concatenate`: returns a single concatenated
      array from ``TTrees``.
    * :doc:`uproot._dask.dask`: returns an unevaluated Dask array from ``TTrees``.
    """
    uproot._util._check_unknown_options(options)
    uproot._util.check_step_size(step_size)
    files = uproot._util.regularize_files(files, steps_allowed=False, **options)
    decompression_executor, interpretation_executor = _regularize_executors(
        decompression_executor, interpretation_executor, None
    )
    library = uproot.interpretation.library._regularize_library(library)

    global_offset = 0
    for file_path, object_path in files:
        hasbranches_context = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, options
        )

        if hasbranches_context is not None:
            with hasbranches_context as hasbranches:
                try:
                    for item in hasbranches.iterate(
                        expressions=expressions,
                        cut=cut,
                        filter_name=filter_name,
                        filter_typename=filter_typename,
                        filter_branch=(
                            filter_branch
                            if isinstance(hasbranches, HasBranches)
                            or filter_branch is not no_filter
                            else unset
                        ),
                        **(
                            {}
                            if isinstance(hasbranches, HasBranches)
                            else {"filter_field": filter_field}
                        ),
                        aliases=aliases,
                        language=language,
                        step_size=step_size,
                        decompression_executor=decompression_executor,
                        interpretation_executor=interpretation_executor,
                        library=library,
                        ak_add_doc=ak_add_doc,
                        how=how,
                        report=report,
                    ):
                        if report:
                            arrays, rep = item
                            arrays = library.global_index(arrays, global_offset)
                            rep = rep.to_global(global_offset)
                            popper = [arrays]
                            del arrays
                            del item
                            yield popper.pop(), rep

                        else:
                            popper = [library.global_index(item, global_offset)]
                            del item
                            yield popper.pop()

                except uproot.exceptions.KeyInFileError:
                    if allow_missing:
                        continue
                    else:
                        raise

                global_offset += hasbranches.num_entries


def concatenate(
    files,
    expressions=None,
    cut=None,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    filter_field=no_filter,
    aliases=None,
    language=uproot.language.python.python_language,
    entry_start=None,
    entry_stop=None,
    decompression_executor=None,
    interpretation_executor=None,
    library="ak",
    ak_add_doc=False,
    how=None,
    custom_classes=None,
    allow_missing=False,
    **options,
):
    """
    Args:
        files: See below.
        expressions (None, str, or list of str): Names of ``TBranches`` or
            aliases to convert to arrays or mathematical expressions of them.
            Uses the ``language`` to evaluate. If None, all ``TBranches``
            selected by the filters are included.
        cut (None or str): If not None, this expression filters all of the
            ``expressions``.
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
            filter to select ``TBranches`` using the full
            :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
            returns False or None, the ``TBranch`` is excluded; if the function
            returns True, it is included with its standard
            :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
            :doc:`uproot.interpretation.Interpretation`, this interpretation
            overrules the standard one. Only used for TTrees.
        filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
            filter to select ``RFields`` using the full
            :doc:`uproot.models.RNTuple.RField` object. If the function
            returns False or None, the ``RField`` is excluded; if the function
            returns True, it is included. Only used for RNTuples.
        aliases (None or dict of str \u2192 str): Mathematical expressions that
            can be used in ``expressions`` or other aliases (without cycles).
            Uses the ``language`` engine to evaluate. If None, only the
            :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
        language (:doc:`uproot.language.Language`): Language used to interpret
            the ``expressions`` and ``aliases``.
        entry_start (None or int): The first entry to include. If None, start
            at zero. If negative, count from the end, like a Python slice.
        entry_stop (None or int): The first entry to exclude (i.e. one greater
            than the last entry to include). If None, stop at
            :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
            count from the end, like a Python slice.
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, a
            :doc:`uproot.source.futures.TrivialExecutor` is created.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, a :doc:`uproot.source.futures.TrivialExecutor`
            is created.
        library (str or :doc:`uproot.interpretation.library.Library`): The library
            that is used to represent arrays. Options are ``"np"`` for NumPy,
            ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
        ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
            to the Awkward ``__doc__`` parameter of the array.
            if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
            Awkward ``key`` parameter of the array.
        how (None, str, or container type): Library-dependent instructions
            for grouping. The only recognized container types are ``tuple``,
            ``list``, and ``dict``. Note that the container *type itself*
            must be passed as ``how``, not an instance of that type (i.e.
            ``how=tuple``, not ``how=()``).
            For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip`` to
            interleave data from compatible branches.
        custom_classes (None or dict): If a dict, override the classes from
            the :doc:`uproot.reading.ReadOnlyFile` or ``uproot.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        options: See below.

    Returns an array with data from a set of files concatenated into one.

    For example:

    .. code-block:: python

        >>> array = uproot.concatenate("files*.root:tree", ["x", "y"])

    Depending on the number of files, the number of selected ``TBranches``, and
    the size of your computer's memory, this function might not have enough
    memory to run.

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/ttree"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/ttree"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}``
    * already-open TTree objects.
    * iterables of the above.

    ..
      Future Dev Note: These shared reading options are manually copied across multiple files
      so they appear in `help` output of Python interpreter and the online documentation
      for the functions where they are used.

    Options (type; default):

    * handler (:doc:`uproot.source.chunk.Source` class; None): Class implementing reading from the data source.
      If None, deduced from input file type.
    * timeout (float for HTTP, int for XRootD; default defined by source implementation): The time in seconds
      to wait before giving up on the connection. Ignored for non-internet sources like local file paths.
    * max_num_elements (None or int; None): The maximum number of byte ranges requested in a single XRootD
      vector read. This does not limit the number of TTree or RNTuple entries read; pass ``entry_stop`` to
      ``arrays``, ``uproot.iterate``, or ``uproot.concatenate`` instead.
    * num_workers (int; 1): Number of tasks to spawn for reading, only used by some source types
    * use_threads (bool; False on the emscripten platform (i.e. in a web browser), else True):
      Use multi-threading when spawning workers.
    * num_fallback_workers (int; 10): Number of tasks to spawn for reading in fallback mode
      (for example, multi-threading requests instead of a multipart GET for an http source)
    * begin_chunk_size (memory_size; 403, the smallest a ROOT file can be): Size of first chunk that we attempt
      to read in bytes.
    * minimal_ttree_metadata (bool; True): Skip rarely used metadata and defer reading of embedded TBaskets
    * http_max_header_bytes (int; 21784): Maximum size of HTTP packet in bytes when the source is http

    Other file entry points:

    * :doc:`uproot.reading.open`: opens one file to read any of its objects.
    * :doc:`uproot.behaviors.TBranch.iterate`: iterates through chunks of
      contiguous entries in ``TTrees``.
    * :doc:`uproot.behaviors.TBranch.concatenate` (this function): returns a
      single concatenated array from ``TTrees``.
    * :doc:`uproot._dask.dask`: returns an unevaluated Dask array from ``TTrees``.
    """
    uproot._util._check_unknown_options(options)
    files = uproot._util.regularize_files(files, steps_allowed=False, **options)
    decompression_executor, interpretation_executor = _regularize_executors(
        decompression_executor, interpretation_executor, None
    )
    library = uproot.interpretation.library._regularize_library(library)

    all_arrays = []
    global_start = 0
    global_stop = 0

    all_hasbranches = []
    for file_path, object_path in files:
        _hasbranches = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, options
        )
        if _hasbranches is not None:
            all_hasbranches.append(_hasbranches)

    total_num_entries = sum(hasbranches.num_entries for hasbranches in all_hasbranches)
    entry_start, entry_stop = _regularize_entries_start_stop(
        total_num_entries, entry_start, entry_stop
    )
    for hasbranches_context in all_hasbranches:
        with hasbranches_context as hasbranches:
            nentries = hasbranches.num_entries
            global_stop += nentries

            if (
                global_start <= entry_start < global_stop
                or global_start < entry_stop <= global_stop
            ):
                # overlap, read only the overlapping entries
                local_entry_start = max(
                    0, entry_start - global_start
                )  # need to clip to 0
                local_entry_stop = entry_stop - global_start  # overflows are fine
            elif entry_start >= global_stop or entry_stop <= global_start:  # no overlap
                # outside of this file's range -> skip
                global_start = global_stop
                continue
            else:
                # read all entries
                local_entry_start = 0
                local_entry_stop = nentries

            try:
                arrays = hasbranches.arrays(
                    expressions=expressions,
                    cut=cut,
                    filter_name=filter_name,
                    filter_typename=filter_typename,
                    filter_branch=(
                        filter_branch
                        if isinstance(hasbranches, HasBranches)
                        or filter_branch is not no_filter
                        else unset
                    ),
                    **(
                        {}
                        if isinstance(hasbranches, HasBranches)
                        else {"filter_field": filter_field}
                    ),
                    aliases=aliases,
                    language=language,
                    entry_start=local_entry_start,
                    entry_stop=local_entry_stop,
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                    array_cache=None,
                    library=library,
                    ak_add_doc=ak_add_doc,
                    how=how,
                )
                arrays = library.global_index(arrays, global_start)
            except uproot.exceptions.KeyInFileError:
                if allow_missing:
                    global_start = global_stop
                    continue
                else:
                    raise

            all_arrays.append(arrays)
            global_start = global_stop

    return library.concatenate(all_arrays)


class Report:
    """
    Args:
        source (:doc:`uproot.behaviors.TBranch.HasBranches`): The
            object (:doc:`uproot.behaviors.TBranch.TBranch` or
            :doc:`uproot.behaviors.TTree.TTree`) that this batch of data
            came from.
        tree_entry_start (int): First entry in the batch, counting zero at
            the start of the current ``TTree`` (current file).
        tree_entry_stop (int): First entry *after* the batch (last entry plus
            one), counting zero at the start of the ``TTree`` (current file).
        global_offset (int): Number of entries between the start of iteration
            and the start of this ``TTree``. The
            :ref:`uproot.behaviors.TBranch.Report.global_entry_start` and
            :ref:`uproot.behaviors.TBranch.Report.global_entry_stop` are
            equal to :ref:`uproot.behaviors.TBranch.Report.tree_entry_start`
            and :ref:`uproot.behaviors.TBranch.Report.tree_entry_stop` plus
            ``global_offset``.

    Information about the current iteration of
    :ref:`uproot.behaviors.TBranch.HasBranches.iterate` (the method) or
    :doc:`uproot.behaviors.TBranch.iterate` (the function).

    Since the :ref:`uproot.behaviors.TBranch.HasBranches.iterate` method
    only iterates over data from one ``TTree``, its ``global_offset`` is always
    zero; :ref:`uproot.behaviors.TBranch.Report.global_entry_start` and
    :ref:`uproot.behaviors.TBranch.Report.global_entry_stop` are equal to
    :ref:`uproot.behaviors.TBranch.Report.tree_entry_start` and
    :ref:`uproot.behaviors.TBranch.Report.tree_entry_stop`, respectively.

    """

    def __init__(self, source, tree_entry_start, tree_entry_stop, global_offset=0):
        self._source = source
        self._tree_entry_start = tree_entry_start
        self._tree_entry_stop = tree_entry_stop
        self._global_offset = global_offset

    def __repr__(self):
        return "<Report start={} stop={} source={}>".format(
            self.global_entry_start,
            self.global_entry_stop,
            repr(self._source.file.file_path + ":" + self._source.object_path),
        )

    @property
    def source(self):
        """
        The object (:doc:`uproot.behaviors.TBranch.TBranch` or
        :doc:`uproot.behaviors.TTree.TTree`) that this batch of data
        came from.
        """
        return self._source

    @property
    def tree(self):
        """
        The :doc:`uproot.behaviors.TTree.TTree` that this batch of data
        came from.
        """
        return self._source.tree

    @property
    def file(self):
        """
        The :doc:`uproot.reading.ReadOnlyFile` that this batch of data
        came from.
        """
        return self._source.file

    @property
    def file_path(self):
        """
        The path/name of the :doc:`uproot.reading.ReadOnlyFile` that
        this batch of data came from.
        """
        return self._source.file.file_path

    @property
    def tree_entry_start(self):
        """
        First entry in the batch, counting zero at the start of the current
        ``TTree`` (current file).
        """
        return self._tree_entry_start

    @property
    def tree_entry_stop(self):
        """
        First entry *after* the batch (last entry plus one), counting zero at
        the start of the ``TTree`` (current file).
        """
        return self._tree_entry_stop

    @property
    def global_entry_start(self):
        """
        First entry in the batch, counting zero at the start of iteration
        (potentially over many files).
        """
        return self._tree_entry_start + self._global_offset

    @property
    def global_entry_stop(self):
        """
        First entry *after* the batch (last entry plust one), counting zero at
        the start of iteration (potentially over many files).
        """
        return self._tree_entry_stop + self._global_offset

    @property
    def start(self):
        """
        A synonym for
        :ref:`uproot.behaviors.TBranch.Report.global_entry_start`.
        """
        return self._tree_entry_start + self._global_offset

    @property
    def stop(self):
        """
        A synonym for
        :ref:`uproot.behaviors.TBranch.Report.global_entry_stop`.
        """
        return self._tree_entry_stop + self._global_offset

    @property
    def global_offset(self):
        """
        Number of entries between the start of iteration and the start of this
        ``TTree``. The
        :ref:`uproot.behaviors.TBranch.Report.global_entry_start` and
        :ref:`uproot.behaviors.TBranch.Report.global_entry_stop` are
        equal to :ref:`uproot.behaviors.TBranch.Report.tree_entry_start`
        and :ref:`uproot.behaviors.TBranch.Report.tree_entry_stop` plus
        ``global_offset``.
        """
        return self._global_offset

    def to_global(self, global_offset):
        """
        Copies the data in this :doc:`uproot.behaviors.TBranch.Report` to
        another with a new
        :ref:`uproot.behaviors.TBranch.Report.global_offset`.
        """
        return Report(
            self._source, self._tree_entry_start, self._tree_entry_stop, global_offset
        )


def _ak_add_doc(array, hasbranches, ak_add_doc, top_level):
    if type(array).__module__ == "awkward.highlevel":
        array.layout.parameters.update(
            uproot._util.ak_add_doc_parameters(hasbranches, ak_add_doc, top_level)
        )
    return array


class HasBranches(Mapping):
    """
    Abstract class of behaviors for anything that "has branches," namely
    :doc:`uproot.behaviors.TTree.TTree` and
    :doc:`uproot.behaviors.TBranch.TBranch`, which mostly consist of array-reading
    methods.

    A :doc:`uproot.behaviors.TBranch.HasBranches` is a Python ``Mapping``, which
    uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_tree["branch"]
        my_tree["branch"]["subbranch"]
        my_tree["branch/subbranch"]
        my_tree["branch/subbranch/subsubbranch"]
    """

    @property
    def branches(self):
        """
        The list of :doc:`uproot.behaviors.TBranch.TBranch` directly under
        this :doc:`uproot.behaviors.TTree.TTree` or
        :doc:`uproot.behaviors.TBranch.TBranch` (i.e. not recursive).
        """
        return self.member("fBranches")

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
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, recursively descend into the branches'
                branches.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the display name.
            name_width (int): Number of characters to reserve for the ``TBranch``
                names.
            typename_width (int): Number of characters to reserve for the C++
                typenames.
            interpretation_width (int): Number of characters to reserve for the
                :doc:`uproot.interpretation.Interpretation` displays.
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display the ``TBranches``.

        For example,

        .. code-block::

            name                 | typename             | interpretation
            ---------------------+----------------------+-----------------------------------
            event_number         | int32_t              | AsDtype('>i4')
            trigger_isomu24      | bool                 | AsDtype('bool')
            eventweight          | float                | AsDtype('>f4')
            MET                  | TVector2             | AsStridedObjects(Model_TVector2_v3
            jetp4                | std::vector<TLorentz | AsJagged(AsStridedObjects(Model_TL
            jetbtag              | std::vector<float>   | AsJagged(AsDtype('>f4'), header_by
            jetid                | std::vector<bool>    | AsJagged(AsDtype('bool'), header_b
        """
        if name_width < 3:
            raise ValueError("'name_width' must be at least 3")
        if typename_width < 3:
            raise ValueError("'typename_width' must be at least 3")
        if interpretation_width < 3:
            raise ValueError("'interpretation_width' must be at least 3")

        formatter = f"{{0:{name_width}.{name_width}}} | {{1:{typename_width}.{typename_width}}} | {{2:{interpretation_width}.{interpretation_width}}}"

        stream.write(formatter.format("name", "typename", "interpretation"))
        stream.write(
            "\n"
            + "-" * name_width
            + "-+-"
            + "-" * typename_width
            + "-+-"
            + "-" * interpretation_width
            + "\n"
        )

        if isinstance(self, TBranch):
            stream.write(
                formatter.format(self.name, self.typename, repr(self.interpretation))
                + "\n"
            )

        for name, branch in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
        ):
            typename = branch.typename
            interp = repr(branch.interpretation)

            if len(name) > name_width:
                name = (  # noqa: PLW2901 (overwriting name)
                    name[: name_width - 3] + "..."
                )
            if len(typename) > typename_width:
                typename = typename[: typename_width - 3] + "..."
            if len(interp) > interpretation_width:
                interp = interp[: interpretation_width - 3] + "..."

            stream.write(formatter.format(name, typename, interp).rstrip(" ") + "\n")

    def arrays(
        self,
        expressions=None,
        cut=None,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot.language.python.python_language,
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        library="ak",
        ak_add_doc=False,
        how=None,
        virtual=False,
        access_log=None,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
                returns False or None, the ``TBranch`` is excluded; if the function
                returns True, it is included with its standard
                :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
                :doc:`uproot.interpretation.Interpretation`, this interpretation
                overrules the standard one.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
                is used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
                Awkward ``key`` parameter of the array.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
                For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip``
                to interleave data from compatible branches.
            virtual (bool): If True, return virtual arrays that compute their
                data on demand; if False, return fully realized arrays.
            access_log (None or object with a ``__iadd__`` method): If an access_log is
                provided, e.g. a list, all materializations of the arrays are
                tracked inside this reference. Only applies if ``virtual=True``.

        Returns a group of arrays from the ``TTree``.

        .. note::

            When any filter (``filter_name``, ``filter_typename``, or
            ``filter_branch``) selects an *AsGrouped* branch (a pure-grouping
            container with no data buffers of its own), the result depends on
            which of its sub-branches are also selected:

            * **Only the parent selected** — all sub-branches are returned grouped
              as a single ``RecordArray``.
            * **Only sub-branches selected (no parent)** — the selected sub-branches
              are returned as separate flat fields.
            * **Parent and all sub-branches selected** — the parent is dropped and
              each sub-branch is returned as a separate flat field.
            * **Parent and some sub-branches selected** — the complete grouped record
              is returned for all sub-branches; the individually-selected
              sub-branches are absorbed into the group and are not also returned
              as flat fields.

        For example:

        .. code-block:: python

            >>> my_tree["x"].array()
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
            >>> my_tree["y"].array()
            <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

        Or read lazily with awkward's virtual arrays:

        .. code-block:: python

            >>> my_tree.arrays(virtual=True, )
            <Array [{run: ??, ...}, ..., {run: ??, ...}] type='40 * {run: uint32, lumin...'>
            >>> access_log = []
            >>> array = my_tree.arrays(virtual=True, access_log=access_log)
            >>> ak.materialize(array.run)
            >>> print(access_log)
            [Accessed(branch='run', buffer_key="('<root>', 'run')-data")]

        See also :ref:`uproot.behaviors.TBranch.TBranch.array` to read a single
        ``TBranch`` as an array.

        See also :ref:`uproot.behaviors.TBranch.HasBranches.iterate` to iterate over
        the array in contiguous ranges of entries.
        """
        if virtual:
            # some kwargs can't be used with virtual arrays
            err = "'{}' cannot be used with 'virtual=True'".format
            if how is not None:
                raise ValueError(err("how"))
            if library != "ak":
                raise ValueError(err("library"))
            if expressions is not None:
                raise ValueError(err("expressions"))
            if cut is not None:
                raise ValueError(err("cut"))
            if aliases is not None:
                raise ValueError(err("aliases"))

            return self._virtual_arrays(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                full_paths=False,
                entry_start=entry_start,
                entry_stop=entry_stop,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                array_cache=array_cache,
                ak_add_doc=ak_add_doc,
                access_log=access_log,
            )
        else:
            # some kwargs can't be used with eager arrays
            err = "'{}' cannot be used with 'virtual=False'".format
            if access_log is not None:
                raise ValueError(err("access_log"))

            return self._eager_arrays(
                expressions=expressions,
                cut=cut,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                aliases=aliases,
                language=language,
                entry_start=entry_start,
                entry_stop=entry_stop,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                array_cache=array_cache,
                library=library,
                ak_add_doc=ak_add_doc,
                how=how,
            )

    def _virtual_arrays(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        ak_add_doc=False,
        access_log=None,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
                returns False or None, the ``TBranch`` is excluded; if the function
                returns True, it is included with its standard
                :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
                :doc:`uproot.interpretation.Interpretation`, this interpretation
                overrules the standard one.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return the names of branches directly accessible
                under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
                is used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            ak_add_doc (bool | dict ): If True, add the TBranch ``title``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value}, add the TBranch ``value`` to the
                Awkward ``key`` parameter of the array.
            access_log (None or object with a ``__iadd__`` method): If an access_log is
                provided, e.g. a list, all materializations of the virtual arrays are
                tracked inside this reference.


        Returns a group of virtual arrays from the ``TTree``. This method can only return Awkward Arrays.

        For example:

        .. code-block:: python

            >>> my_tree._virtual_arrays()
            <Array [{run: ??, ...}, ..., {run: ??, ...}] type='40 * {run: uint32, lumin...'>
            >>> access_log = []
            >>> array = my_tree._virtual_arrays(access_log=access_log)
            >>> ak.materialize(array.Jet_pt)
            >>> print(access_log)
            [Accessed(branch='Jet_pt', buffer_key='<root>.Jet_pt-offsets'), Accessed(branch='Jet_pt', buffer_key='<root>.Jet_pt.content-data')]


        See also :ref:`uproot.behaviors.TBranch.HasBranches.arrays` to iterate over
        the array in contiguous ranges of entries.
        """
        from uproot._dask import FormMappingWithVirtualArrays

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor, self._file
        )
        array_cache = _regularize_array_cache(array_cache, self._file)

        keys = self.keys(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
            ignore_duplicates=ignore_duplicates,
        )

        # Normalise AsGrouped branches in keys according to which of their children
        # also appear in keys (same case logic as _regularize_expressions):
        # - Case 1 (Parent matched, no leaves matched):    include parent grouped
        # - Case 2 (Parent not matched, some leaves matched):   skip parent, keep matched leaves individual
        # - Case 3 (Parent matched, all leaves matched):   skip parent, keep leaves individual
        # - Case 4 (Parent matched, some leaves matched):  include parent grouped, suppress matched leaves
        keys_set = set(keys)
        parent_keys_to_remove = set()
        child_keys_to_suppress = set()
        for k in keys:
            if not isinstance(
                self[k].interpretation,
                uproot.interpretation.grouped.AsGrouped,
            ):
                continue
            if full_paths:
                all_child_keys = [
                    f"{k}/{ck}" for ck in self[k].keys(recursive=True, full_paths=True)
                ]
            else:
                all_child_keys = self[k].keys(recursive=True, full_paths=False)
            matched_children = [c for c in all_child_keys if c in keys_set]
            if len(all_child_keys) > 0 and len(matched_children) == len(all_child_keys):
                parent_keys_to_remove.add(k)
            elif matched_children:
                child_keys_to_suppress.update(matched_children)
        keys = [
            k
            for k in keys
            if k not in parent_keys_to_remove and k not in child_keys_to_suppress
        ]

        # we're dealing with a single branch here:
        if isinstance(self, TBranch) and len(keys) == 0:
            filter_branch = uproot._util.regularize_filter(filter_branch)
            return self.parent._virtual_arrays(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=lambda branch: branch is self and filter_branch(branch),
                recursive=recursive,
                full_paths=full_paths,
                ignore_duplicates=ignore_duplicates,
                entry_start=entry_start,
                entry_stop=entry_stop,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                array_cache=array_cache,
                ak_add_doc=ak_add_doc,
                access_log=access_log,
            )

        base_form = get_ttree_form(
            self,
            keys,
            ak_add_doc,
        )

        expected_form, form_mapping_info = FormMappingWithVirtualArrays()(base_form)

        # The buffer replacements of FormMappingInfoWithVirtualArrays are VirtualArrays
        container = form_mapping_info.buffer_replacements(
            self,
            keys,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            {"ak_add_doc": ak_add_doc, "access_log": access_log},
        )
        return awkward.from_buffers(
            expected_form,
            entry_stop - entry_start,
            container,
            behavior=form_mapping_info.behavior,
            buffer_key=form_mapping_info.buffer_key,
        )

    def _eager_arrays(
        self,
        expressions=None,
        cut=None,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot.language.python.python_language,
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        library="ak",
        ak_add_doc=False,
        how=None,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
                returns False or None, the ``TBranch`` is excluded; if the function
                returns True, it is included with its standard
                :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
                :doc:`uproot.interpretation.Interpretation`, this interpretation
                overrules the standard one.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
                is used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
                Awkward ``key`` parameter of the array.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
                For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip``
                to interleave data from compatible branches.

        Returns a group of arrays from the ``TTree``.

        For example:

        .. code-block:: python

            >>> my_tree["x"].array()
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
            >>> my_tree["y"].array()
            <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

        See also :ref:`uproot.behaviors.TBranch.TBranch.array` to read a single
        ``TBranch`` as an array.

        See also :ref:`uproot.behaviors.TBranch.HasBranches.iterate` to iterate over
        the array in contiguous ranges of entries.
        """
        keys = _keys_deep(self)
        if isinstance(self, TBranch) and expressions is None and len(keys) == 0:
            filter_branch = uproot._util.regularize_filter(filter_branch)
            return self.parent._eager_arrays(
                expressions=expressions,
                cut=cut,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=lambda branch: branch is self and filter_branch(branch),
                aliases=aliases,
                language=language,
                entry_start=entry_start,
                entry_stop=entry_stop,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                array_cache=array_cache,
                library=library,
                ak_add_doc=ak_add_doc,
                how=how,
            )

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.tree.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor, self._file
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot.interpretation.library._regularize_library(library)

        def get_from_cache(branchname, interpretation):
            if array_cache is not None:
                cache_key = f"{self.cache_key}:{branchname}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
                return array_cache.get(cache_key)
            else:
                return None

        aliases = _regularize_aliases(self, aliases)
        arrays, expression_context, branchid_interpretation = _regularize_expressions(
            self,
            expressions,
            cut,
            filter_name,
            filter_typename,
            filter_branch,
            keys,
            aliases,
            language,
            get_from_cache,
        )

        ranges_or_baskets = []
        checked = set()
        for _, context in expression_context:
            for branch in context["branches"]:
                if (
                    branch.cache_key not in checked
                    # branches already satisfied from the array cache are present
                    # in arrays; don't re-read and re-decompress their baskets
                    and branch.cache_key not in arrays
                    and not isinstance(
                        branchid_interpretation[branch.cache_key],
                        uproot.interpretation.grouped.AsGrouped,
                    )
                ):
                    checked.add(branch.cache_key)
                    for (
                        basket_num,
                        range_or_basket,
                    ) in branch.entries_to_ranges_or_baskets(entry_start, entry_stop):
                        ranges_or_baskets.append((branch, basket_num, range_or_basket))

        interp_options = {"ak_add_doc": ak_add_doc}
        _ranges_or_baskets_to_arrays(
            self,
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            library,
            arrays,
            False,
            interp_options,
        )

        # no longer needed; save memory
        del ranges_or_baskets

        _fix_asgrouped(
            arrays,
            expression_context,
            branchid_interpretation,
            library,
            how,
            ak_add_doc,
        )

        if array_cache is not None:
            checked = set()
            for expression, context in expression_context:
                for branch in context["branches"]:
                    if branch.cache_key not in checked:
                        checked.add(branch.cache_key)
                        interpretation = branchid_interpretation[branch.cache_key]
                        cache_key = f"{self.cache_key}:{expression}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
                        array_cache[cache_key] = arrays[branch.cache_key]

        output = language.compute_expressions(
            self,
            arrays,
            expression_context,
            keys,
            aliases,
            self.file.file_path,
            self.object_path,
        )

        # no longer needed; save memory
        del arrays

        expression_context = [
            (e, c) for e, c in expression_context if c["is_primary"] and not c["is_cut"]
        ]

        return _ak_add_doc(
            library.group(output, expression_context, how), self, ak_add_doc, True
        )

    def iterate(
        self,
        expressions=None,
        cut=None,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot.language.python.python_language,
        entry_start=None,
        entry_stop=None,
        step_size="100 MB",
        decompression_executor=None,
        interpretation_executor=None,
        library="ak",
        ak_add_doc=False,
        how=None,
        report=False,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. If the function
                returns False or None, the ``TBranch`` is excluded; if the function
                returns True, it is included with its standard
                :ref:`uproot.behaviors.TBranch.TBranch.interpretation`; if an
                :doc:`uproot.interpretation.Interpretation`, this interpretation
                overrules the standard one.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            step_size (int or str): If an integer, the maximum number of entries to
                include in each iteration step; if a string, the maximum memory size
                to include. The string must be a number followed by a memory unit,
                such as "100 MB".
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
                is used.
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
                Awkward ``key`` parameter of the array.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
                For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip``
                to interleave data from compatible branches.
            report (bool): If True, this generator yields
                (arrays, :doc:`uproot.behaviors.TBranch.Report`) pairs; if False,
                it only yields arrays. The report has data about the ``TFile``,
                ``TTree``, and global and local entry ranges.

        Iterates through contiguous chunks of entries from the ``TTree``.

        For example:

        .. code-block:: python

            >>> for array in tree.iterate(["x", "y"], step_size=100):
            ...     # each of the following have 100 entries
            ...     array["x"], array["y"]

        See also :ref:`uproot.behaviors.TBranch.HasBranches.arrays` to read
        everything in a single step, without iteration.

        See also :doc:`uproot.behaviors.TBranch.iterate` to iterate over many
        files.
        """
        uproot._util.check_step_size(step_size)
        keys = _keys_deep(self)
        if isinstance(self, TBranch) and expressions is None and len(keys) == 0:
            filter_branch = uproot._util.regularize_filter(filter_branch)
            yield from self.parent.iterate(
                expressions=expressions,
                cut=cut,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=lambda branch: branch is self and filter_branch(branch),
                aliases=aliases,
                language=language,
                entry_start=entry_start,
                entry_stop=entry_stop,
                step_size=step_size,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                library=library,
                ak_add_doc=ak_add_doc,
                how=how,
                report=report,
            )

        else:
            entry_start, entry_stop = _regularize_entries_start_stop(
                self.tree.num_entries, entry_start, entry_stop
            )
            decompression_executor, interpretation_executor = _regularize_executors(
                decompression_executor, interpretation_executor, self._file
            )
            library = uproot.interpretation.library._regularize_library(library)

            aliases = _regularize_aliases(self, aliases)
            (
                arrays,
                expression_context,
                branchid_interpretation,
            ) = _regularize_expressions(
                self,
                expressions,
                cut,
                filter_name,
                filter_typename,
                filter_branch,
                keys,
                aliases,
                language,
                (lambda branchname, interpretation: None),
            )

            if len(branchid_interpretation) == 0:
                return

            entry_step = _regularize_step_size(
                self, step_size, entry_start, entry_stop, branchid_interpretation
            )

            previous_baskets = {}
            for sub_entry_start in range(entry_start, entry_stop, entry_step):
                sub_entry_stop = min(sub_entry_start + entry_step, entry_stop)
                if sub_entry_stop - sub_entry_start == 0:
                    continue

                ranges_or_baskets = []
                checked = set()
                for _, context in expression_context:
                    for branch in context["branches"]:
                        if branch.cache_key not in checked and not isinstance(
                            branchid_interpretation[branch.cache_key],
                            uproot.interpretation.grouped.AsGrouped,
                        ):
                            checked.add(branch.cache_key)
                            for (
                                basket_num,
                                range_or_basket,
                            ) in branch.entries_to_ranges_or_baskets(
                                sub_entry_start, sub_entry_stop
                            ):
                                previous_basket = previous_baskets.get(
                                    (branch.cache_key, basket_num)
                                )
                                if previous_basket is None:
                                    ranges_or_baskets.append(
                                        (branch, basket_num, range_or_basket)
                                    )
                                else:
                                    ranges_or_baskets.append(
                                        (branch, basket_num, previous_basket)
                                    )

                arrays = {}
                interp_options = {"ak_add_doc": ak_add_doc}
                _ranges_or_baskets_to_arrays(
                    self,
                    ranges_or_baskets,
                    branchid_interpretation,
                    sub_entry_start,
                    sub_entry_stop,
                    decompression_executor,
                    interpretation_executor,
                    library,
                    arrays,
                    True,
                    interp_options,
                )

                _fix_asgrouped(
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    library,
                    how,
                    ak_add_doc,
                )

                output = language.compute_expressions(
                    self,
                    arrays,
                    expression_context,
                    keys,
                    aliases,
                    self.file.file_path,
                    self.object_path,
                )

                # no longer needed; save memory
                del arrays

                minimized_expression_context = [
                    (e, c)
                    for e, c in expression_context
                    if c["is_primary"] and not c["is_cut"]
                ]

                out = _ak_add_doc(
                    library.group(output, minimized_expression_context, how),
                    self,
                    ak_add_doc,
                    True,
                )

                # no longer needed; save memory
                del output

                next_baskets = {}
                for branch, basket_num, basket in ranges_or_baskets:
                    _basket_entry_start, basket_entry_stop = basket.entry_start_stop
                    if basket_entry_stop > sub_entry_stop:
                        next_baskets[branch.cache_key, basket_num] = basket

                previous_baskets = next_baskets

                # no longer needed; save memory
                popper = [out]
                del out

                if report:
                    yield popper.pop(), Report(self, sub_entry_start, sub_entry_stop)
                else:
                    yield popper.pop()

    def keys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return the names of branches directly accessible
                under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.

        Returns the names of the subbranches as a list of strings.
        """
        return list(
            self.iterkeys(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
                full_paths=full_paths,
                ignore_duplicates=ignore_duplicates,
            )
        )

    def values(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return branches that are directly accessible
                under this object.

        Returns the subbranches as a list of
        :doc:`uproot.behaviors.TBranch.TBranch`.

        (Note: with ``recursive=False``, this is the same as
        :ref:`uproot.behaviors.TBranch.HasBranches.branches`.)
        """
        return list(
            self.itervalues(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
            )
        )

    def items(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, branch) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, branch) pairs of the subbranches as a list of 2-tuples
        of (str, :doc:`uproot.behaviors.TBranch.TBranch`).
        """
        return list(
            self.iteritems(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
                full_paths=full_paths,
            )
        )

    def typenames(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, typename) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, typename) pairs of the subbranches as a dict of
        str \u2192 str.
        """
        return dict(
            self.itertypenames(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
                full_paths=full_paths,
            )
        )

    def iterkeys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return the names of branches directly accessible
                under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.


        Returns the names of the subbranches as an iterator over strings.
        """
        for k, _ in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
            ignore_duplicates=ignore_duplicates,
        ):
            yield k

    def itervalues(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return branches that are directly accessible
                under this object.

        Returns the subbranches as an iterator over
        :doc:`uproot.behaviors.TBranch.TBranch`.

        (Note: with ``recursive=False``, this is the same as
        :ref:`uproot.behaviors.TBranch.HasBranches.branches`.)
        """
        for _, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=False,
        ):
            yield v

    def iteritems(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, branch) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.


        Returns (name, branch) pairs of the subbranches as an iterator over
        2-tuples of (str, :doc:`uproot.behaviors.TBranch.TBranch`).
        """
        filter_name = uproot._util.regularize_filter(filter_name)
        filter_typename = uproot._util.regularize_filter(filter_typename)
        if filter_branch is None:
            filter_branch = no_filter
        elif callable(filter_branch):
            pass
        else:
            raise TypeError(
                f"filter_branch must be None or a function: TBranch -> bool, not {filter_branch!r}"
            )

        keys_set = set()

        for branch in self.branches:
            if (
                (
                    filter_name is no_filter
                    or _filter_name_deep(filter_name, self, branch)
                )
                and (filter_typename is no_filter or filter_typename(branch.typename))
                and (filter_branch is no_filter or filter_branch(branch))
            ):
                if ignore_duplicates and branch.name in keys_set:
                    pass
                else:
                    keys_set.add(branch.name)
                    yield branch.name, branch

            if recursive:
                for k1, v in branch.iteritems(
                    recursive=recursive,
                    filter_name=no_filter,
                    filter_typename=filter_typename,
                    filter_branch=filter_branch,
                    full_paths=full_paths,
                ):
                    k2 = f"{branch.name}/{k1}" if full_paths else k1
                    if filter_name is no_filter or _filter_name_deep(
                        filter_name, self, v
                    ):
                        if ignore_duplicates and k2 in keys_set:
                            pass
                        else:
                            keys_set.add(k2)
                            yield k2, v

    def itertypenames(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, typename) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, typename) pairs of the subbranches as an iterator over
        2-tuples of (str, str).
        """
        for k, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
        ):
            yield k, v.typename

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()

    def num_entries_for(
        self,
        memory_size,
        expressions=None,
        cut=None,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot.language.python.python_language,
        entry_start=None,
        entry_stop=None,
    ):
        """
        Args:
            memory_size (int or str): An integer is interpreted as a number of
                bytes and a string must be a number followed by a unit, such as
                "100 MB".
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :ref:`uproot.behaviors.TBranch.TBranch.aliases` are available.
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.

        Returns an *approximate* step size as a number of entries to read
        a given ``memory_size`` in each step.

        This method does not actually read the ``TBranch`` data or compute any
        expressions to arrive at its estimate. It only uses metadata from the
        already-loaded ``TTree``; it only needs ``language`` to parse the
        expressions, not to evaluate them.

        In addition, the estimate is based on uncompressed ``TBasket`` sizes
        (approximately the amount of data that the final arrays would use in
        memory, without considering ``cuts``), not compressed ``TBasket`` sizes
        (the amount of data that would have to be read). Only the ``TBasket``
        ``TKeys`` are read to obtain these sizes, not the ``TBasket`` data.

        This is the algorithm that
        :ref:`uproot.behaviors.TBranch.HasBranches.iterate` uses to convert a
        ``step_size`` expressed in memory units into a number of entries.
        """
        target_num_bytes = uproot._util.memory_size(memory_size)

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.tree.num_entries, entry_start, entry_stop
        )

        keys = _keys_deep(self)
        aliases = _regularize_aliases(self, aliases)
        _arrays, _expression_context, branchid_interpretation = _regularize_expressions(
            self,
            expressions,
            cut,
            filter_name,
            filter_typename,
            filter_branch,
            keys,
            aliases,
            language,
            (lambda branchname, interpretation: None),
        )

        return _hasbranches_num_entries_for(
            self, target_num_bytes, entry_start, entry_stop, branchid_interpretation
        )

    def common_entry_offsets(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :doc:`uproot.behaviors.TBranch.TBranch` \u2192 bool, :doc:`uproot.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :doc:`uproot.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only consider branches directly accessible under this
                object. (Only applies when ``branches=None``.)

        Returns entry offsets in which ``TBasket`` boundaries align in the
        specified set of branches, plus this ``TBranch``'s own if it is one.
        A branch without ``TBaskets`` (a split parent, any branch of an empty
        ``TTree``) constrains nothing; when no branch has any, the output is
        ``[0]``.

        If this :doc:`uproot.behaviors.TBranch.TBranch` has no subbranches,
        the output is identical to
        :ref:`uproot.behaviors.TBranch.TBranch.entry_offsets`.
        """
        branches = self.itervalues(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
        )
        # reading a TBranch decompresses its own TBaskets too
        if isinstance(self, TBranch):
            branches = (self, *branches)
        common_offsets = None
        for branch in branches:
            if branch.num_baskets == 0:
                continue
            own = set(branch.entry_offsets)
            common_offsets = own if common_offsets is None else common_offsets & own
        return [0] if common_offsets is None else sorted(common_offsets)

    def __getitem__(self, where):
        original_where = where

        if uproot._util.isint(where):
            return self.branches[where]
        elif isinstance(where, str):
            where = uproot._util.ensure_str(where)
        else:
            raise TypeError(f"where must be an integer or a string, not {where!r}")

        if where.startswith("/"):
            recursive = False
            where = where[1:]
        else:
            recursive = True

        got = self._lookup.get(where)
        if got is not None:
            return got

        if "/" in where:
            this = self
            try:
                for piece in where.split("/"):
                    if piece != "":
                        this = this[piece]
            except uproot.KeyInFileError:
                raise uproot.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                ) from None
            return this

        elif recursive:
            got = _get_recursive(self, where)
            if got is not None:
                return got
            else:
                raise uproot.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

        else:
            raise uproot.KeyInFileError(
                original_where,
                keys=self.keys(recursive=recursive),
                file_path=self._file.file_path,
                object_path=self.object_path,
            )

    def __iter__(self):
        yield from self.branches

    def __len__(self):
        return len(self.branches)

    @property
    def source(self) -> uproot.source.chunk.Source | None:
        """Returns the associated source of data for this container, if it exists

        Returns: uproot.source.chunk.Source or None
        """
        if isinstance(self, uproot.model.Model) and isinstance(
            self._file, uproot.reading.ReadOnlyFile
        ):
            return self._file.source
        return None


_branch_clean_name = re.compile(r"(.*\.)*([^\.\[\]]*)(\[.*\])*")
_branch_clean_parent_name = re.compile(r"(.*\.)*([^\.\[\]]*)\.([^\.\[\]]*)(\[.*\])*")


class TBranch(HasBranches):
    """
    Behaviors for a ``TBranch``, which mostly consist of array-reading methods.

    Since a :doc:`uproot.behaviors.TBranch.TBranch` is a
    :doc:`uproot.behaviors.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_branch["subbranch"]
        my_branch["subbranch"]["subsubbranch"]
        my_branch["subbranch/subsubbranch"]
    """

    # Set on first use, so instances unpickled without it still work
    _uncompressed_bytes_cache = None

    def __getstate__(self):
        # The sizes can be read again, but would be sent with every Dask task
        class_data, instance_data = super().__getstate__()
        instance_data.pop("_uncompressed_bytes_cache", None)
        return class_data, instance_data

    def __repr__(self):
        if len(self) == 0:
            return f"<{self.classname} {self.name!r} at 0x{id(self):012x}>"
        else:
            return f"<{self.classname} {self.name!r} ({len(self)} subbranches) at 0x{id(self):012x}>"

    def array(
        self,
        interpretation=None,
        entry_start=None,
        entry_stop=None,
        *,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        library="ak",
        ak_add_doc=False,
    ):
        """
        Args:
            interpretation (None or :doc:`uproot.interpretation.Interpretation`): An
                interpretation of the ``TBranch`` data as an array. If None, the
                standard :ref:`uproot.behaviors.TBranch.TBranch.interpretation`
                is used, which is derived from
                :doc:`uproot.interpretation.identify.interpretation_of`.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the file's :ref:`uproot.reading.ReadOnlyFile.interpretation_executor`
                is used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the TBranch ``value`` to the
                Awkward ``key`` parameter of the array.

        Returns the ``TBranch`` data as an array.

        For example:

        .. code-block:: python

            >>> array = tree
            >>> array = tree.arrays(["x", "y"])    # only reads branches "x" and "y"
            >>> array
            <Array [{x: -41.2, y: 17.4}, ... {x: 32.5, y: 1.2}], type='2304 * {"x": float64,...'>
            >>> array["x"]
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
            >>> array["y"]
            <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

        See also :ref:`uproot.behaviors.TBranch.HasBranches.arrays` to read
        multiple ``TBranches`` into a group of arrays or an array-group.
        """
        if interpretation is None:
            interpretation = self.interpretation
        else:
            interpretation = _regularize_interpretation(interpretation)

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor, self._file
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot.interpretation.library._regularize_library(library)

        def get_from_cache(branchname, interpretation):
            if array_cache is not None:
                cache_key = f"{self.cache_key}:{branchname}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
                return array_cache.get(cache_key)
            else:
                return None

        arrays = {}
        expression_context = []
        branchid_interpretation = {}
        _regularize_branchname(
            self,
            self.name,
            self,
            interpretation,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            True,
            False,
        )

        ranges_or_baskets = []
        checked = set()
        for _, context in expression_context:
            for branch in context["branches"]:
                if branch.cache_key not in checked and not isinstance(
                    branchid_interpretation[branch.cache_key],
                    uproot.interpretation.grouped.AsGrouped,
                ):
                    checked.add(branch.cache_key)
                    for (
                        basket_num,
                        range_or_basket,
                    ) in branch.entries_to_ranges_or_baskets(entry_start, entry_stop):
                        ranges_or_baskets.append((branch, basket_num, range_or_basket))

        interp_options = {"ak_add_doc": ak_add_doc}
        _ranges_or_baskets_to_arrays(
            self,
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            library,
            arrays,
            False,
            interp_options,
        )

        _fix_asgrouped(
            arrays,
            expression_context,
            branchid_interpretation,
            library,
            None,
            ak_add_doc,
        )

        if array_cache is not None:
            cache_key = f"{self.cache_key}:{self.name}:{interpretation.cache_key}:{entry_start}-{entry_stop}:{library.name}"
            array_cache[cache_key] = arrays[self.cache_key]

        return arrays[self.cache_key]

    def __array__(self, *args, **kwargs):
        out = self.array(library="np")
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)

    @property
    def name(self):
        """
        Name of the ``TBranch``.

        Note that ``TBranch`` names are not guaranteed to be unique; it is
        sometimes necessary to address a branch by its
        :ref:`uproot.behaviors.TBranch.TBranch.index`.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        Title of the ``TBranch``.
        """
        return self.member("fTitle")

    @property
    def object_path(self):
        """
        Object path of the ``TBranch``.
        """
        sep = ":" if isinstance(self._parent, uproot.behaviors.TTree.TTree) else "/"
        return f"{self.parent.object_path}{sep}{self.name}"

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``TBranch`` in its path, to use as
        part of object and array cache keys.
        """
        if self._cache_key is None:
            sep = ":" if isinstance(self._parent, uproot.behaviors.TTree.TTree) else "/"
            self._cache_key = f"{self.parent.cache_key}{sep}{self.name}({self.index})"
        return self._cache_key

    @property
    def index(self):
        """
        Integer position of this ``TBranch`` in its parent's list of branches.

        Useful for cases in which the
        :ref:`uproot.behaviors.TBranch.TBranch.name` is not unique: the
        non-recursive index is always unique.
        """
        if not hasattr(self, "_index"):
            # cache index of all branches of the parent to avoid repeating this loop for other branches
            for i, branch in enumerate(self.parent.branches):
                branch._index = i
        return self._index

    @property
    def interpretation(self):
        """
        The standard :doc:`uproot.interpretation.Interpretation` of this
        ``TBranch`` as an array, derived from
        :doc:`uproot.interpretation.identify.interpretation_of`.

        If no interpretation could be found, the value of this property
        would be a :doc:`uproot.interpretation.identify.UnknownInterpretation`,
        which is a Python ``Exception``. Since the exception is *returned*,
        rather than *raised*, a branch lacking an interpretation is not a fatal
        error.

        However, any attempt to use this exception object as an
        :doc:`uproot.interpretation.Interpretation` causes it to raise itself:
        attempting to read a branch lacking an interpretation is a fatal error.
        """
        if self._interpretation is None:
            try:
                self._interpretation = uproot.interpretation.identify.interpretation_of(
                    self, {}
                )
            except uproot.interpretation.identify.UnknownInterpretation as err:
                self._interpretation = err
        return self._interpretation

    @property
    def typename(self):
        """
        The C++ typename of the ``TBranch``, derived from its
        :ref:`uproot.behaviors.TBranch.TBranch.interpretation`. If the
        interpretation is
        :doc:`uproot.interpretation.identify.UnknownInterpretation`, the
        typename is ``"unknown"``.
        """
        if self.interpretation is None:
            return "unknown"
        else:
            return self.interpretation.typename

    @property
    def num_entries(self):
        """
        The number of entries in the ``TBranch``, as reported by ``fEntries``.

        In principle, this could disagree with the
        :ref:`uproot.behaviors.TTree.TTree.num_entries`, which is from the
        ``TTree``'s ``fEntries``.

        The ``TBranch`` also has a ``fEntryNumber``, which ought to be equal to
        the ``TBranch`` and ``TTree``'s ``fEntries``, and the last value of
        :ref:`uproot.behaviors.TBranch.TBranch.entry_offsets` ought to be
        equal to the number of entries as well.
        """
        return int(self.member("fEntries"))  # or fEntryNumber?

    @property
    def entry_offsets(self):
        """
        The starting and stopping entry numbers for all the ``TBaskets`` in the
        ``TBranch`` as a list of increasing, non-negative integers.

        The number of ``entry_offsets`` in this list of integers is one more
        than the number of ``TBaskets``. The first is ``0`` and the last is
        the number of entries
        (:ref:`uproot.behaviors.TBranch.TBranch.num_entries`).
        """
        if self._num_normal_baskets == 0:
            out = [0]
        else:
            out = self.member("fBasketEntry")[: self._num_normal_baskets + 1].tolist()
        num_entries_normal = out[-1]

        for basket in self.embedded_baskets:
            out.append(out[-1] + basket.num_entries)

        if (
            out[-1] != self.num_entries
            and self.interpretation is not None
            and not isinstance(
                self.interpretation, uproot.interpretation.grouped.AsGrouped
            )
        ):
            raise ValueError(
                f"""entries in normal baskets ({num_entries_normal}) plus embedded baskets ({sum(basket.num_entries for basket in self.embedded_baskets)}) """
                f"""don't add up to expected number of entries ({self.num_entries})
in file {self._file.file_path}"""
            )
        else:
            return out

    def basket_entry_start_stop(self, basket_num):
        """
        The starting and stopping entry number for ``TBasket`` number ``basket_num``.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            fBasketEntry = self.member("fBasketEntry")
            return fBasketEntry[basket_num], fBasketEntry[basket_num + 1]

        elif 0 <= basket_num < self.num_baskets:
            baskets_before = self._num_normal_baskets
            if self._num_normal_baskets == 0:
                entries_before = 0
            else:
                entries_before = self.member("fBasketEntry")[self._num_normal_baskets]

            for basket in self.embedded_baskets:
                if basket_num == baskets_before:
                    return entries_before, entries_before + basket.num_entries
                baskets_before += 1
                entries_before += basket.num_entries
            else:
                raise AssertionError

        else:
            raise IndexError(
                f"""branch {self.name!r} has {self.num_baskets} baskets; cannot get starting entry """
                f"""for basket {basket_num}
in file {self._file.file_path}"""
            )

    @property
    def tree(self):
        """
        The ``TTree`` to which this ``TBranch`` belongs. The branch might be
        deeply nested; this property ascends all the way to the top.
        """
        out = self
        while not isinstance(out, uproot.behaviors.TTree.TTree):
            out = out.parent
        return out

    @property
    def top_level(self):
        """
        True if this branch's immediate :ref:`uproot.model.Model.parent`
        is the ``TTree``; False otherwise.
        """
        return isinstance(self.parent, uproot.behaviors.TTree.TTree)

    @property
    def streamer(self):
        """
        The ``TStreamerInfo`` or ``TStreamerElement`` for this ``TBranch``,
        which may be None.

        If the :ref:`uproot.reading.ReadOnlyFile.streamers` have not yet been
        read, this method *might* cause them to be read. (Only
        ``TBranchElements`` can have streamers.)
        """
        if self._streamer is None:
            clean_name = _branch_clean_name.match(self.name).group(2)
            clean_parentname = _branch_clean_parent_name.match(self.name)
            fParentName = self.member("fParentName", none_if_missing=True)
            fClassName = self.member("fClassName", none_if_missing=True)

            fParentName = fParentName.replace(" ", "") if fParentName else None
            fClassName = fClassName.replace(" ", "") if fClassName else None

            if fParentName is not None and fParentName != "":
                matches = self._file.streamers.get(fParentName)

                if matches is not None:
                    streamerinfo = matches[max(matches)]

                    for element in streamerinfo.walk_members(self._file.streamers):
                        if element.name == clean_name and (
                            fClassName is None
                            or fClassName == ""
                            or element.parent is None
                            or element.parent.name == ""
                            or element.parent.name == fClassName
                        ):
                            self._streamer = element
                            break

                    if self._streamer is None and clean_parentname is not None:
                        clean_parentname = clean_parentname.group(2)
                        for element in streamerinfo.walk_members(self._file.streamers):
                            if element.name == clean_parentname:
                                substreamerinfo = self._file.streamer_named(
                                    element.typename
                                )
                                for subelement in substreamerinfo.walk_members(
                                    self._file.streamers
                                ):
                                    if subelement.name == clean_name:
                                        self._streamer = subelement
                                        break
                                break

                    if self.parent.member(
                        "fClassName"
                    ) == "TClonesArray" or self.parent.member(
                        "fClonesName", none_if_missing=True
                    ) == self.member(
                        "fParentName"
                    ):  # Use `self.member("fParentName")` since `fClonesName` could contain spaces between brackets.
                        self._streamer_isTClonesArray = True

            elif fClassName is not None and fClassName != "":
                if fClassName == "TClonesArray":
                    self._streamer_isTClonesArray = True
                    fClonesName = self.member("fClonesName", none_if_missing=True)
                    fClonesName = fClonesName.replace(" ", "") if fClonesName else None
                    matches = self._file.streamers.get(fClonesName)
                else:
                    matches = self._file.streamers.get(fClassName)

                if matches is not None:
                    self._streamer = matches[max(matches)]

        return self._streamer

    @property
    def context(self):
        """
        Auxiliary data used in deserialization. This is a *copy* of the
        ``context`` dict at the time that the ``TBranch`` was deserialized
        with ``"in_TBranch": True`` added.
        """
        return self._context

    @property
    def aliases(self):
        """
        The :ref:`uproot.behaviors.TTree.TTree.aliases`, which are used as the
        ``aliases`` argument to
        :ref:`uproot.behaviors.TBranch.HasBranches.arrays`,
        :ref:`uproot.behaviors.TBranch.HasBranches.iterate`,
        :doc:`uproot.behaviors.TBranch.iterate`, and
        :doc:`uproot.behaviors.TBranch.concatenate` if one is not given.

        The return type is always a dict of str \u2192 str, even if there
        are no aliases (an empty dict).
        """
        return self.tree.aliases

    @property
    def count_branch(self):
        """
        The ``TBranch`` object in which this branch's "counts" reside or None
        if this branch has no "counts".
        """
        out = self.count_leaf
        while isinstance(out, uproot.model.Model) and out.is_instance("TLeaf"):
            out = out.parent
        if isinstance(out, uproot.model.Model) and out.is_instance("TBranch"):
            return out
        else:
            return None

    @property
    def count_leaf(self):
        """
        The ``TLeaf`` object of this branch's "counts" or None if this branch
        has no "counts".
        """
        leaves = self.member("fLeaves")
        if len(leaves) != 1:
            return None
        return leaves[0].member("fLeafCount")

    @property
    def compression(self):
        """
        A :doc:`uproot.compression.Compression` object describing the
        compression setting for the ``TBranch``.

        Note that different ``TBranches`` in a ``TTree`` can be compressed
        differently from each other, and they can be compressed differently
        from the file's global compression setting.

        It is also *in principle possible* for the blocks in each ``TBasket``
        to be compresssed differently: see
        :ref:`uproot.models.TBasket.Model_TBasket.block_compression_info` if
        you're paranoid.
        """
        return uproot.compression.Compression.from_code(self.member("fCompress"))

    @property
    def compressed_bytes(self):
        """
        The number of compressed bytes in all ``TBaskets`` of this ``TBranch``,
        including the TKey headers (which are always uncompressed).

        This information is specified in the ``TBranch`` metadata (``fZipBytes``)
        and can be determined without reading any additional data.
        """
        return self.member("fZipBytes")

    @property
    def uncompressed_bytes(self):
        """
        The number of uncompressed bytes in all ``TBaskets`` of this ``TBranch``,
        including the TKey headers.

        This information is specified in the ``TBranch`` metadata (``fTotBytes``)
        and can be determined without reading any additional data.
        """
        return self.member("fTotBytes")

    @property
    def compression_ratio(self):
        """
        The number of uncompressed bytes divided by the number of compressed
        bytes for this ``TBranch``.

        See :ref:`uproot.behaviors.TBranch.TBranch.compressed_bytes` and
        :ref:`uproot.behaviors.TBranch.TBranch.uncompressed_bytes`.
        """
        return float(self.uncompressed_bytes) / float(self.compressed_bytes)

    @property
    def num_baskets(self):
        """
        The number of ``TBaskets`` in this ``TBranch``, including both normal
        (free) ``TBaskets`` and
        :ref:`uproot.behaviors.TBranch.TBranch.embedded_baskets`.
        """
        return self._num_normal_baskets + len(self.embedded_baskets)

    def basket(self, basket_num):
        """
        The :doc:`uproot.models.TBasket.Model_TBasket` at index ``basket_num``.

        It may be a normal (free) ``TBasket`` or one of the
        :ref:`uproot.behaviors.TBranch.TBranch.embedded_baskets`.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            chunk, cursor = self.basket_chunk_cursor(basket_num)
            return uproot.models.TBasket.Model_TBasket.read(
                chunk, cursor, {"basket_num": basket_num}, self._file, self._file, self
            )
        elif 0 <= basket_num < self.num_baskets:
            return self.embedded_baskets[basket_num - self._num_normal_baskets]
        else:
            raise IndexError(
                f"""branch {self.name!r} has {self.num_baskets} baskets; cannot get basket {basket_num}
in file {self._file.file_path}"""
            )

    def basket_chunk_cursor(self, basket_num):
        """
        Returns a :doc:`uproot.source.chunk.Chunk` and
        :doc:`uproot.source.cursor.Cursor` as a 2-tuple for a given
        ``basket_num``.

        If the file source is :doc:`uproot.source.file.MemmapSource`
        and the file gets closed, accessing the Chunk would cause
        a segfault. If that's a possibility, be sure to call
        :doc:`uproot.source.chunk.Chunk.detach_memmap` to ensure
        that any memmap-derived data gets copied for safety.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            start = self.member("fBasketSeek")[basket_num]
            stop = start + self.basket_compressed_bytes(basket_num)
            # xrootd doesn't like NumPy ints
            start, stop = int(start), int(stop)
            cursor = uproot.source.cursor.Cursor(start)
            chunk = self._file.source.chunk(start, stop)
            return chunk, cursor
        elif 0 <= basket_num < self.num_baskets:
            raise IndexError(
                f"""branch {self.name!r} has {self._num_normal_baskets} normal baskets; cannot get chunk and """
                f"""cursor for basket {basket_num} because only normal baskets have cursors
in file {self._file.file_path}"""
            )
        else:
            raise IndexError(
                f"""branch {self.name!r} has {self.num_baskets} baskets; cannot get cursor and chunk """
                f"""for basket {basket_num}
in file {self._file.file_path}"""
            )

    def basket_compressed_bytes(self, basket_num):
        """
        The number of compressed bytes for the ``TBasket`` at ``basket_num``,
        including the TKey header.

        The number of compressed bytes is specified in the ``TBranch`` metadata
        and can be determined without reading any additional data. The
        uncompressed bytes requires reading the ``TBasket``'s ``TKey`` at least.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            return int(self.member("fBasketBytes")[basket_num])
        elif 0 <= basket_num < self.num_baskets:
            return self.embedded_baskets[
                basket_num - self._num_normal_baskets
            ].compressed_bytes
        else:
            raise IndexError(
                f"""branch {self.name!r} has {self.num_baskets} baskets; cannot get basket chunk {basket_num}
in file {self._file.file_path}"""
            )

    def basket_uncompressed_bytes(self, basket_num):
        """
        The number of uncompressed bytes for the ``TBasket`` at ``basket_num``,
        including the TKey header.

        The number of uncompressed bytes cannot be determined without reading a
        ``TKey``, which are small, but may be slow for remote connections because
        of the latency of round-trip requests. The result is cached.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            _check_file_open(self._file)
            cache = self._get_uncompressed_bytes_cache()
            if cache[basket_num] < 0:
                cache[basket_num] = _key_uncompressed_bytes(self.basket_key(basket_num))
            return int(cache[basket_num])
        elif 0 <= basket_num < self.num_baskets:
            return self.basket(basket_num).uncompressed_bytes
        else:
            raise IndexError(
                f"""branch {self.name!r} has {self.num_baskets} baskets; cannot get basket {basket_num}
in file {self._file.file_path}"""
            )

    def basket_key(self, basket_num):
        """
        The ``TKey`` (:doc:`uproot.reading.ReadOnlyKey`) for the ``TBasket``
        at ``basket_num``.

        Only applies to normal (free) ``TBaskets``, not
        :ref:`uproot.behaviors.TBranch.TBranch.embedded_baskets`.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            start, stop = self._basket_key_range(basket_num)
            # Chunk will not be retained; we don't have to detach_memmap()
            return self._basket_key_from_chunk(self._file.source.chunk(start, stop))

        elif 0 <= basket_num < self.num_baskets:
            raise ValueError(
                f"branch {self.name!r} basket {basket_num} is an embedded basket, which has no TKey"
            )

        else:
            raise IndexError(
                f"""branch {self.name!r} has {self.num_baskets} baskets; cannot get basket chunk {basket_num}
in file {self._file.file_path}"""
            )

    def _get_uncompressed_bytes_cache(self):
        # Uncompressed bytes of each normal basket, -1 if not read yet
        if self._uncompressed_bytes_cache is None:
            self._uncompressed_bytes_cache = numpy.full(
                self._num_normal_baskets, -1, dtype=numpy.int64
            )
        return self._uncompressed_bytes_cache

    def _basket_key_range(self, basket_num):
        # xrootd doesn't like NumPy ints
        start = int(self.member("fBasketSeek")[basket_num])
        return start, start + uproot.reading._key_format_big.size

    def _basket_key_from_chunk(self, chunk):
        return uproot.reading.ReadOnlyKey(
            chunk,
            uproot.source.cursor.Cursor(chunk.start),
            {},
            self._file,
            self,
            read_strings=False,
        )

    @property
    def embedded_baskets(self):
        """
        The ``TBaskets`` that are embedded within the ``TBranch`` metadata,
        usually because the ROOT process that was writing the file closed
        unexpectedly.
        """
        if self._embedded_baskets is None:
            cursor = self._cursor_baskets.copy()
            baskets = uproot.models.TObjArray.Model_TObjArrayOfTBaskets.read(
                self.tree.chunk, cursor, {}, self._file, self._file, self
            )
            with self._embedded_baskets_lock:
                self._embedded_baskets = []
                for basket in baskets:
                    if basket is not None:
                        basket._basket_num = self._num_normal_baskets + len(
                            self._embedded_baskets
                        )
                        self._embedded_baskets.append(basket)

        return self._embedded_baskets

    def entries_to_ranges_or_baskets(self, entry_start, entry_stop):
        """
        Returns a list of (start, stop) integer pairs for free (normal)
        ``TBaskets`` and :doc:`uproot.models.TBasket.Model_TBasket` objects
        for embedded ``TBaskets``.

        The intention is for this list to be updated in place, replacing
        (start, stop) integer pairs with
        :doc:`uproot.models.TBasket.Model_TBasket` objects as they get
        read and interpreted.
        """
        entry_offsets = self.entry_offsets
        out = []
        start = entry_offsets[0]
        for basket_num, stop in enumerate(entry_offsets[1:]):
            if entry_start < stop and (
                start < entry_stop or entry_start == entry_stop == start
            ):
                if 0 <= basket_num < self._num_normal_baskets:
                    byte_start = self.member("fBasketSeek")[basket_num]
                    byte_stop = byte_start + self.basket_compressed_bytes(basket_num)
                    out.append((basket_num, (byte_start, byte_stop)))
                elif 0 <= basket_num < self.num_baskets:
                    out.append((basket_num, self.basket(basket_num)))
                else:
                    raise AssertionError((self.name, basket_num))
            start = stop
        return out

    def postprocess(self, chunk, cursor, context, file):
        fWriteBasket = self.member("fWriteBasket")

        self._lookup = {}
        for branch in self.member("fBranches"):
            name = branch.member("fName")
            if name not in self._lookup:
                self._lookup[name] = branch

        self._interpretation = None
        self._typename = None
        self._streamer = None
        self._streamer_isTClonesArray = False
        self._cache_key = None
        self._context = dict(context)
        self._context["breadcrumbs"] = ()
        self._context["in_TBranch"] = True

        self._num_normal_baskets = 0
        for i, x in enumerate(self.member("fBasketSeek")):
            if x == 0 or i == fWriteBasket:
                break
            self._num_normal_baskets += 1

        # the number of entries in basket i == fBasketEntry[i + 1] - fBasketEntry[i]
        # but it's possible for len(fBasketEntry) == i (ROOT can read such files)
        # so in that rare case, look at the header of the last basket
        # and use that to extend the fBasketEntry array by one
        fBasketEntry = self.member("fBasketEntry")
        if len(fBasketEntry) == self._num_normal_baskets:
            start = self.member("fBasketSeek")[len(fBasketEntry) - 1]
            stop = start + self.basket_compressed_bytes(len(fBasketEntry) - 1)
            cursor = uproot.source.cursor.Cursor(start)
            chunk = self._file.source.chunk(start, stop)
            basket_header = uproot.models.TBasket.Model_TBasket.read(
                chunk, cursor, {"read_basket": False}, self._file, self._file, self
            )
            fBasketEntry = self._members["fBasketEntry"] = numpy.append(
                fBasketEntry, fBasketEntry[-1] + basket_header.member("fNevBuf")
            )

        if self.member("fEntries") == fBasketEntry[self._num_normal_baskets]:
            self._embedded_baskets = []
            self._embedded_baskets_lock = None

        elif self.has_member("fBaskets"):
            self._embedded_baskets = []
            for basket in self.member("fBaskets"):
                if basket is not None:
                    basket._basket_num = self._num_normal_baskets + len(
                        self._embedded_baskets
                    )
                    self._embedded_baskets.append(basket)
            self._embedded_baskets_lock = None

        else:
            self._embedded_baskets = None
            self._embedded_baskets_lock = threading.Lock()

        if "fIOFeatures" in self._parent.members:
            self._tree_iofeatures = self._parent.member("fIOFeatures").member("fIOBits")

        return self

    def _awkward_check(self, interpretation):
        try:
            interpretation.awkward_form(self.file)
        except uproot.interpretation.objects.CannotBeAwkward as err:
            raise ValueError(
                f"""cannot produce Awkward Arrays for interpretation {interpretation!r} because

    {err.because}

instead, try library="np" rather than library="ak" or globally set uproot.default_library

in file {self.file.file_path}
in object {self.object_path}"""
            ) from err

    def debug(
        self,
        entry,
        skip_bytes=None,
        limit_bytes=None,
        dtype=None,
        offset=0,
        stream=sys.stdout,
    ):
        """
        Args:
            entry (int): Entry number to inspect. Note: this debugging routine
                is not applicable to data without entry offsets (nor would it
                be needed).
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the ``entry``. May be negative, to examine the
                byte stream before the ``entry``.
            limit_bytes (None or int): Number of bytes to limit the output to.
                A line of debugging output (without any ``offset``) is 20 bytes,
                so multiples of 20 show full lines. If None, everything is
                shown to the end of the ``entry``, which might be large.
            dtype (None, ``numpy.dtype``, or its constructor argument): If None,
                present only the bytes as decimal values (0-255). Otherwise,
                also interpret them as an array of a given NumPy type.
            offset (int): Number of bytes to skip before interpreting a ``dtype``;
                can be helpful if the numerical values are out of phase with
                the first byte shown. Not to be confused with ``skip_bytes``,
                which determines which bytes are shown at all. Any ``offset``
                values that are equivalent modulo ``dtype.itemsize`` show
                equivalent interpretations.
            stream (object with a ``write(str)`` method): Stream to write the
                debugging output to.

        Presents the data for one entry as raw bytes.

        Example output with ``dtype=">f4"`` and ``offset=3``.

        .. code-block::

            --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
            123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
              {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                                    1.1             2.2             3.3             4.4
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
                --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
                        5.5             6.6             7.7             8.8             9.9
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                202   0   0  67  74   0   0  67 151 128   0 123 123
                --- --- ---   C   J --- ---   C --- --- ---   {   {
                      101.0           202.0           303.0
        """
        data = self.debug_array(entry)
        chunk = uproot.source.chunk.Chunk.wrap(self._file.source, data)
        if skip_bytes is None:
            cursor = uproot.source.cursor.Cursor(0)
        else:
            cursor = uproot.source.cursor.Cursor(skip_bytes)
        cursor.debug(
            chunk, limit_bytes=limit_bytes, dtype=dtype, offset=offset, stream=stream
        )

    def debug_array(self, entry, skip_bytes=0, dtype=np_uint8):
        """
        Args:
            entry (int): Entry number to inspect. Note: this debugging routine
                is not applicable to data without entry offsets (nor would it
                be needed).
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the ``entry``. May be negative, to examine the
                byte stream before the ``entry``.
            dtype (``numpy.dtype`` or its constructor argument): Data type in
                which to interpret the data. (The size of the array returned is
                truncated to this ``dtype.itemsize``.)

        Like :ref:`uproot.behaviors.TBranch.TBranch.debug`, but returns a
        NumPy array for further inspection.
        """
        dtype = numpy.dtype(dtype)
        interpretation = uproot.interpretation.jagged.AsJagged(
            uproot.interpretation.numerical.AsDtype("u1")
        )
        out = self.array(
            interpretation, entry_start=entry, entry_stop=entry + 1, library="np"
        )[0][skip_bytes:]
        return out[: (len(out) // dtype.itemsize) * dtype.itemsize].view(dtype)


def _filter_name_deep(filter_name, hasbranches, branch):
    shallow = name = branch.name
    if filter_name(name):
        return True
    while branch is not hasbranches:
        branch = branch.parent
        if branch is not hasbranches:
            name = branch.name + "/" + name
    if name != shallow and filter_name(name):
        return True
    return filter_name("/" + name)


def _keys_deep(hasbranches):
    out = set()
    for branch in hasbranches.itervalues(recursive=True):
        name = branch.name
        out.add(name)
        while branch is not hasbranches:
            branch = branch.parent  # noqa: PLW2901 (overwriting branch)
            if branch is not hasbranches:
                name = branch.name + "/" + name
        out.add(name)
        out.add("/" + name)
    return out


def _get_recursive(hasbranches, where):
    got = hasbranches._lookup.get(where)
    if got is not None:
        return got
    for branch in hasbranches.branches:
        got = _get_recursive(branch, where)
        if got is not None:
            return got
    else:
        return None


def _regularize_entries_start_stop(num_entries, entry_start, entry_stop):
    if entry_start is None:
        entry_start = 0
    elif entry_start < 0:
        entry_start += num_entries
    entry_start = min(num_entries, max(0, entry_start))

    if entry_stop is None:
        entry_stop = num_entries
    elif entry_stop < 0:
        entry_stop += num_entries
    entry_stop = min(num_entries, max(0, entry_stop))

    if entry_stop < entry_start:
        entry_stop = entry_start

    return int(entry_start), int(entry_stop)


def _regularize_executors(decompression_executor, interpretation_executor, file):
    if file is None:
        if decompression_executor is None:
            decompression_executor = uproot.source.futures.TrivialExecutor()
        if interpretation_executor is None:
            interpretation_executor = uproot.source.futures.TrivialExecutor()
    else:
        if decompression_executor is None:
            decompression_executor = file.decompression_executor
        if interpretation_executor is None:
            interpretation_executor = file.interpretation_executor
    return decompression_executor, interpretation_executor


def _regularize_array_cache(array_cache, file):
    if isinstance(array_cache, MutableMapping):
        return array_cache
    elif isinstance(array_cache, str) and array_cache == "inherit":
        return file._array_cache
    elif array_cache is None:
        return None
    elif uproot._util.isint(array_cache) or isinstance(array_cache, str):
        return uproot.cache.LRUArrayCache(array_cache)
    else:
        raise TypeError("array_cache must be None, a MutableMapping, or a memory size")


def _regularize_aliases(hasbranches, aliases):
    if aliases is None:
        return hasbranches.aliases
    else:
        new_aliases = dict(hasbranches.aliases)
        new_aliases.update(aliases)
        return new_aliases


def _regularize_interpretation(interpretation):
    if isinstance(interpretation, uproot.interpretation.Interpretation):
        return interpretation
    else:
        dtype = numpy.dtype(interpretation)
        dtype = dtype.newbyteorder(">")
        return uproot.interpretation.numerical.AsDtype(dtype)


def _regularize_branchname(
    hasbranches,
    branchname,
    branch,
    interpretation,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    is_primary,
    is_cut,
):
    got = get_from_cache(branchname, interpretation)
    if got is not None:
        arrays[branch.cache_key] = got

    is_jagged = isinstance(interpretation, uproot.interpretation.jagged.AsJagged)

    if isinstance(interpretation, uproot.interpretation.grouped.AsGrouped):
        branches = []
        for subname, subinterp in interpretation.subbranches.items():
            _regularize_branchname(
                hasbranches,
                subname,
                branch[subname],
                subinterp,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                False,
                is_cut,
            )
            branches.extend(expression_context[-1][1]["branches"])

        branches.append(branch)
        arrays[branch.cache_key] = None

    else:
        branches = [branch]

    if branch.cache_key in branchid_interpretation:
        if (
            branchid_interpretation[branch.cache_key].cache_key
            != interpretation.cache_key
        ):
            raise ValueError(
                "a branch cannot be loaded with multiple interpretations: "
                f"{branchid_interpretation[branch.cache_key]!r} and {interpretation!r}"
            )
    else:
        branchid_interpretation[branch.cache_key] = interpretation

    c = {
        "is_primary": is_primary,
        "is_cut": is_cut,
        "is_jagged": is_jagged,
        "is_branch": True,
        "branches": branches,
    }
    expression_context.append((branchname, c))


def _regularize_expression(
    hasbranches,
    expression,
    keys,
    aliases,
    language,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    symbol_path,
    is_cut,
    rename,
):
    is_primary = symbol_path == ()

    branch = hasbranches.get(expression)
    if branch is not None:
        _regularize_branchname(
            hasbranches,
            expression,
            branch,
            branch.interpretation,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            is_primary,
            is_cut,
        )

    else:
        # the value of `expression` is either what we want to compute or a lookup value for it
        to_compute = aliases.get(expression, expression)

        is_jagged = False
        expression_branches = []
        for symbol in language.free_symbols(
            to_compute,
            keys,
            aliases,
            hasbranches.file.file_path,
            hasbranches.object_path,
        ):
            if symbol in symbol_path:
                raise ValueError(
                    """symbol {} is recursively defined with aliases:

    {}

in file {} at {}""".format(
                        repr(symbol),
                        "\n    ".join(f"{k}: {v}" for k, v in aliases.items()),
                        hasbranches.file.file_path,
                        hasbranches.object_path,
                    )
                )

            _regularize_expression(
                hasbranches,
                symbol,
                keys,
                aliases,
                language,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                (*symbol_path, symbol),
                False,
                None,
            )
            if expression_context[-1][1]["is_jagged"]:
                is_jagged = True
            expression_branches.extend(expression_context[-1][1]["branches"])

        c = {
            "is_primary": is_primary,
            "is_cut": is_cut,
            "is_jagged": is_jagged,
            "is_branch": False,
            "branches": expression_branches,
        }
        if rename is not None:
            c["rename"] = rename
        expression_context.append((expression, c))


def _collect_leaf_cache_keys(branch, result=None):
    """Recursively collect TBranch cache keys of all non-AsGrouped descendants.

    Used to classify how an AsGrouped branch should be handled when building
    expression_context: whether ALL its leaves were matched (case 3 → skip) or
    only some / none (case 1/4 → include grouped).  UnknownInterpretation leaves
    are silently skipped because they cannot be read anyway.
    """
    if result is None:
        result = set()
    for subname in branch.interpretation.subbranches:
        subbranch = branch[subname]
        if isinstance(
            subbranch.interpretation, uproot.interpretation.grouped.AsGrouped
        ):
            _collect_leaf_cache_keys(subbranch, result)
        elif not isinstance(
            subbranch.interpretation,
            uproot.interpretation.identify.UnknownInterpretation,
        ):
            result.add(subbranch.cache_key)
    return result


def _iter_branch_ancestors(branch):
    """Yield each TBranch ancestor of *branch*, from parent up to the tree root.

    Used by the nested-AsGrouped de-duplication step in ``_regularize_expressions``
    to detect whether a branch's ancestor is already being added grouped.
    """
    p = getattr(branch, "parent", None)
    while isinstance(p, TBranch):
        yield p
        p = getattr(p, "parent", None)


def _regularize_expressions(
    hasbranches,
    expressions,
    cut,
    filter_name,
    filter_typename,
    filter_branch,
    keys,
    aliases,
    language,
    get_from_cache,
):
    arrays = {}
    expression_context = []
    branchid_interpretation = {}

    if expressions is None:
        # Collect all matched branches without adding to expression_context yet;
        # we need to see all matches before deciding how to handle each AsGrouped.
        matched_regular = []  # (branchname, branch) for non-AsGrouped, non-Unknown
        asgrouped_branches = []  # (branchname, branch)
        for branchname, branch in hasbranches.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=True,
            full_paths=False,
        ):
            if isinstance(
                branch.interpretation,
                uproot.interpretation.identify.UnknownInterpretation,
            ):
                pass
            elif isinstance(
                branch.interpretation,
                uproot.interpretation.grouped.AsGrouped,
            ):
                asgrouped_branches.append((branchname, branch))
            else:
                matched_regular.append((branchname, branch))

        # Classify each AsGrouped branch using its RECURSIVE leaf cache keys so
        # that nested AsGrouped structures are handled correctly:
        # - Case 1 (Parent matched, no leaves matched):    include parent grouped
        # - Case 2 (Parent not matched, some leaves matched):   skip parent, keep matched leaves individual
        # - Case 3 (Parent matched, all leaves matched):   skip parent, keep leaves individual
        # - Case 4 (Parent matched, some leaves matched):  include parent grouped, suppress matched leaves
        all_regular_cache_keys = {b.cache_key for _, b in matched_regular}
        children_to_suppress = set()  # leaf cache keys not to add individually
        asgrouped_to_add = []  # (branchname, branch) AsGrouped parents to add grouped

        for branchname, branch in asgrouped_branches:
            # Skip AsGrouped branches whose direct sub-interpretations include
            # UnknownInterpretation: AsGrouped.cache_key would raise on them.
            if any(
                isinstance(
                    interp,
                    uproot.interpretation.identify.UnknownInterpretation,
                )
                for interp in branch.interpretation.subbranches.values()
            ):
                continue
            leaf_cache_keys = _collect_leaf_cache_keys(branch)
            matched_leaves = leaf_cache_keys & all_regular_cache_keys
            if len(leaf_cache_keys) > 0 and len(matched_leaves) == len(leaf_cache_keys):
                # Case 3: every leaf was matched individually → skip parent
                pass
            else:
                # Case 1 (no leaves) or Case 4 (some leaves):
                # include the parent grouped and suppress matched leaves to prevent
                # them from also being added individually (would create duplicates).
                children_to_suppress |= matched_leaves
                asgrouped_to_add.append((branchname, branch))

        # If a nested AsGrouped is also in asgrouped_to_add, its ancestor will
        # handle it via _regularize_branchname.  Remove descendants to avoid
        # processing them twice.  We use object identity (not names from
        # full_paths=False iteration, which carry no hierarchy info) to detect
        # the ancestor relationship.
        asgrouped_to_add_ids = {id(b) for _, b in asgrouped_to_add}
        asgrouped_to_add = [
            (branchname, branch)
            for branchname, branch in asgrouped_to_add
            if not any(
                id(p) in asgrouped_to_add_ids for p in _iter_branch_ancestors(branch)
            )
        ]

        # Add regular (non-AsGrouped) branches, skipping leaves subsumed by a parent.
        for branchname, branch in matched_regular:
            if branch.cache_key in children_to_suppress:
                continue
            branchname_expression = (
                branchname
                if branchname.isidentifier() and not iskeyword(branchname)
                else language.getter_of(branchname)
            )
            _regularize_expression(
                hasbranches,
                branchname_expression,
                keys,
                aliases,
                language,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                (),
                False,
                branchname,
            )

        # Add grouped AsGrouped parents (cases 1 and 4).
        for branchname, branch in asgrouped_to_add:
            _regularize_branchname(
                hasbranches,
                branchname,
                branch,
                branch.interpretation,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                True,
                False,
            )

    elif isinstance(expressions, str):
        _regularize_expression(
            hasbranches,
            expressions,
            keys,
            aliases,
            language,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            False,
            None,
        )

    elif isinstance(expressions, Iterable):
        if isinstance(expressions, dict):
            items = expressions.items()
        else:
            items = []
            for expression in expressions:
                if isinstance(expression, str):
                    items.append((expression, None))
                elif isinstance(expression, tuple) and len(expression) == 2:
                    items.append(expression)
                else:
                    raise TypeError(
                        "iterable of expressions must be strings or "
                        "name, Interpretation pairs (length-2 tuples), not "
                        + repr(expression)
                    )

        for expression, interp in items:
            if interp is None:
                _regularize_expression(
                    hasbranches,
                    expression,
                    keys,
                    aliases,
                    language,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    (),
                    False,
                    None,
                )
            else:
                branch = hasbranches[expression]
                interp = (  # noqa: PLW2901 (overwriting interp)
                    _regularize_interpretation(interp)
                )
                _regularize_branchname(
                    hasbranches,
                    expression,
                    branch,
                    interp,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    True,
                    False,
                )

    else:
        raise TypeError(
            "expressions must be None (for all branches), a string (single "
            "branch or expression), a list of strings (multiple), or a dict "
            "or list of name, Interpretation pairs (branch names and their "
            f"new Interpretation), not {expressions!r}"
        )

    if cut is None:
        pass
    elif isinstance(cut, str):
        _regularize_expression(
            hasbranches,
            cut,
            keys,
            aliases,
            language,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            True,
            None,
        )

    return arrays, expression_context, branchid_interpretation


_basket_arrays_lock = threading.Lock()


def _ranges_or_baskets_to_arrays(
    hasbranches,
    ranges_or_baskets,
    branchid_interpretation,
    entry_start,
    entry_stop,
    decompression_executor,
    interpretation_executor,
    library,
    arrays,
    update_ranges_or_baskets,
    interp_options,
):
    notifications = queue.Queue()

    branchid_arrays = {}
    branchid_num_baskets = {}
    ranges = []
    range_args = {}
    range_original_index = {}
    original_index = 0
    branchid_to_branch = {}

    for cache_key in branchid_interpretation:
        branchid_num_baskets[cache_key] = 0

    for branch, basket_num, range_or_basket in ranges_or_baskets:
        branchid_num_baskets[branch.cache_key] += 1

        if branch.cache_key not in branchid_arrays:
            branchid_arrays[branch.cache_key] = {}

        if isinstance(range_or_basket, tuple) and len(range_or_basket) == 2:
            range_or_basket = (  # noqa: PLW2901 (overwriting range_or_basket)
                int(range_or_basket[0]),
                int(range_or_basket[1]),
            )
            ranges.append(range_or_basket)
            range_args[range_or_basket] = (branch, basket_num)
            range_original_index[range_or_basket] = original_index
        else:
            notifications.put(range_or_basket)

        original_index += 1  # noqa: SIM113 (don't use `enumerate` for `original_index`)

        branchid_to_branch[branch.cache_key] = branch

    for cache_key, interpretation in branchid_interpretation.items():
        if branchid_num_baskets[cache_key] == 0 and cache_key not in arrays:
            arrays[cache_key] = interpretation.final_array(
                {}, 0, 0, [0], library, None, interp_options
            )

        # check for CannotBeAwkward errors on the main thread before reading any data
        if (
            isinstance(library, uproot.interpretation.library.Awkward)
            and isinstance(interpretation, uproot.interpretation.objects.AsObjects)
            and cache_key in branchid_to_branch
        ):
            branchid_to_branch[cache_key]._awkward_check(interpretation)

    def replace(ranges_or_baskets, original_index, basket):
        branch, basket_num, _range_or_basket = ranges_or_baskets[original_index]
        ranges_or_baskets[original_index] = branch, basket_num, basket

    def chunk_to_basket(chunk, branch, basket_num):
        try:
            cursor = uproot.source.cursor.Cursor(chunk.start)
            basket = uproot.models.TBasket.Model_TBasket.read(
                chunk,
                cursor,
                {"basket_num": basket_num},
                hasbranches._file,
                hasbranches._file,
                branch,
            )
            original_index = range_original_index[(chunk.start, chunk.stop)]
            if update_ranges_or_baskets:
                replace(ranges_or_baskets, original_index, basket)
        except Exception as err:
            notifications.put(err)
        else:
            notifications.put(basket)

    forth_context = {x: threading.local() for x in branchid_interpretation}

    def basket_to_array(basket):
        try:
            assert basket.basket_num is not None
            branch = basket.parent
            interpretation = branchid_interpretation[branch.cache_key]
            basket_arrays = branchid_arrays[branch.cache_key]

            context = dict(branch.context)
            context["forth"] = forth_context[branch.cache_key]

            basket_array = interpretation.basket_array(
                basket.data,
                basket.byte_offsets,
                basket,
                branch,
                context,
                basket.member("fKeylen"),
                library,
                interp_options,
            )
            if basket.num_entries != len(basket_array):
                raise ValueError(
                    f"""basket {basket.basket_num} in tree/branch {branch.object_path} has the wrong number of entries """
                    f"""(expected {basket.num_entries}, obtained {len(basket_array)}) when interpreted as {interpretation}
    in file {branch.file.file_path}"""
                )

            basket_num = basket.basket_num
            basket = None

            with _basket_arrays_lock:
                basket_arrays[basket_num] = basket_array
                len_basket_arrays = len(basket_arrays)

            if len_basket_arrays == branchid_num_baskets[branch.cache_key]:
                arrays[branch.cache_key] = interpretation.final_array(
                    basket_arrays,
                    entry_start,
                    entry_stop,
                    branch.entry_offsets,
                    library,
                    branch,
                    interp_options,
                )
                with _basket_arrays_lock:
                    # no longer needed, save memory
                    basket_arrays.clear()

        except Exception as err:
            notifications.put(err)
        else:
            notifications.put(None)

    if len(arrays) == len(branchid_interpretation):
        # all arrays are already in the cache
        return

    # Request all chunks and then poll notifications queue until we have all the arrays we expect
    hasbranches._file.source.chunks(ranges, notifications=notifications)

    while len(arrays) < len(branchid_interpretation):
        obj = notifications.get()

        if isinstance(obj, uproot.source.chunk.Chunk):
            args = range_args[(obj.start, obj.stop)]
            decompression_executor.submit(chunk_to_basket, obj, *args)

        elif isinstance(obj, uproot.models.TBasket.Model_TBasket):
            interpretation_executor.submit(basket_to_array, obj)

        elif obj is None:
            pass

        else:
            raise obj

        obj = None  # release before blocking


def _fix_asgrouped(
    arrays, expression_context, branchid_interpretation, library, how, ak_add_doc
):
    index_start = 0
    for index_stop, (_, context) in enumerate(expression_context):
        if context["is_branch"]:
            branch = context["branches"][-1]
            interpretation = branchid_interpretation[branch.cache_key]
            if isinstance(interpretation, uproot.interpretation.grouped.AsGrouped):
                if arrays[branch.cache_key] is not None:
                    # Already assembled in a prior iteration (e.g., this branch
                    # was recursively processed as a nested sub-branch of a parent
                    # AsGrouped branch that also appears in expression_context).
                    continue

                limited_context = dict(expression_context[index_start:index_stop])

                subarrays = {}
                subcontext = []
                for subname in interpretation.subbranches:
                    subbranch = branch[subname]
                    subarrays[subname] = arrays[subbranch.cache_key]
                    subcontext.append((subname, limited_context[subname]))

                arrays[branch.cache_key] = _ak_add_doc(
                    library.group(subarrays, subcontext, how), branch, ak_add_doc, False
                )

                index_start = index_stop


def _key_uncompressed_bytes(key):
    # Add fKeylen to match the header-inclusive value reported by
    # Model_TBasket.uncompressed_bytes.
    return key.data_uncompressed_bytes + key.fKeylen


def _check_file_open(file):
    # A closed file must fail whether or not the sizes happen to be cached
    if file.source.closed:
        raise OSError(f"file {file.file_path!r} is closed")


# The gaps between TKeys are basket data, which must not be fetched.
_basket_keys_coalesce_config = uproot.source.coalesce.CoalesceConfig(max_range_gap=0)


def _basket_keys_uncompressed_bytes(file, branch_baskets):
    """
    Total of ``TBranch.basket_uncompressed_bytes`` for normal baskets, with all
    uncached ``TKeys`` read in one request instead of one round trip per basket.
    """
    if not branch_baskets:
        return 0

    _check_file_open(file)
    missing = [
        (branch, num)
        for branch, num in branch_baskets
        if branch._get_uncompressed_bytes_cache()[num] < 0
    ]
    if missing:
        _read_basket_keys(file, missing)
    return sum(
        int(branch._get_uncompressed_bytes_cache()[num])
        for branch, num in branch_baskets
    )


def _read_basket_keys(file, branch_baskets):
    source = file.source
    ranges = [branch._basket_key_range(num) for branch, num in branch_baskets]
    # Chunks will not be retained; we don't have to detach_memmap()
    if isinstance(source, uproot.source.fsspec.FSSpecSource):
        chunks = source.chunks(
            ranges,
            notifications=queue.Queue(),
            coalesce_config=_basket_keys_coalesce_config,
        )
    else:
        chunks = source.chunks(ranges, notifications=queue.Queue())

    for (branch, num), chunk in zip(branch_baskets, chunks, strict=True):
        branch._get_uncompressed_bytes_cache()[num] = _key_uncompressed_bytes(
            branch._basket_key_from_chunk(chunk)
        )


def _hasbranches_num_entries_for(
    hasbranches, target_num_bytes, entry_start, entry_stop, branchid_interpretation
):
    total_bytes = 0.0
    normal_baskets = []
    for branch in hasbranches.itervalues(recursive=True):
        if branch.cache_key in branchid_interpretation:
            entry_offsets = branch.entry_offsets
            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if entry_start < stop and (
                    start < entry_stop or entry_start == entry_stop == start
                ):
                    if basket_num < branch._num_normal_baskets:
                        normal_baskets.append((branch, basket_num))
                    else:
                        total_bytes += branch.basket_uncompressed_bytes(basket_num)
                start = stop

    total_bytes += _basket_keys_uncompressed_bytes(hasbranches._file, normal_baskets)

    total_entries = entry_stop - entry_start
    if total_bytes == 0:
        num_entries = 0
    else:
        num_entries = round(target_num_bytes * total_entries / total_bytes)
    if num_entries <= 0:
        return 1
    else:
        return num_entries


def _regularize_step_size(
    hasbranches, step_size, entry_start, entry_stop, branchid_interpretation
):
    uproot._util.check_step_size(step_size)
    if uproot._util.isint(step_size):
        return step_size
    target_num_bytes = uproot._util.memory_size(
        step_size,
        "number of entries or memory size string with units "
        f"(such as '100 MB') required, not {step_size!r}",
    )
    return _hasbranches_num_entries_for(
        hasbranches, target_num_bytes, entry_start, entry_stop, branchid_interpretation
    )
