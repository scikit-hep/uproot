from __future__ import annotations

import functools
import glob
import math
import socket
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, TypeVar

import awkward
import numpy

import uproot
from uproot._util import no_filter, unset
from uproot.behaviors.RNTuple import (
    _OPTION_FORMS,
    HasFields,
    _filter_field_by_key,
    _form_to_numpy_dtype,
    _with_subfield_keys,
)
from uproot.behaviors.RNTuple import (
    _regularize_step_size as _RNTuple_regularize_step_size,
)
from uproot.behaviors.TBranch import HasBranches, TBranch, _regularize_step_size
from uproot.source.chunk import SourcePerformanceCounters

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport
    from awkward.forms import Form
    from awkward.highlevel import Array as AwkArray


def dask(
    files,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size=unset,
    steps_per_file=unset,
    library="ak",
    ak_add_doc=False,
    custom_classes=None,
    allow_missing=False,
    open_files=True,
    form_mapping=None,
    allow_read_errors_with_report=False,
    known_base_form=None,
    decompression_executor=None,
    interpretation_executor=None,
    **options,
):
    """
    Args:
        files: See below.
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
        recursive (bool): If True, include all subbranches of branches as
            separate fields; otherwise, only search one level deep.
        full_paths (bool): If True, include the full path to each subbranch
            with slashes (``/``); otherwise, use the descendant's name as
            the field name.
        step_size (int or str): If an integer, the maximum number of entries to
            include in each chunk/partition; if a string, the maximum memory_size to include
            in each chunk/partition. The string must be a number followed by a memory unit,
            such as "100 MB". Mutually incompatible with steps_per_file: only set
            step_size or steps_per_file, not both. Cannot be used with
            ``open_files=False``.
        steps_per_file (int, default 1):
            Subdivide files into the specified number of chunks/partitions. Mutually incompatible
            with step_size: only set step_size or steps_per_file, not both.
            If both ``step_size`` and ``steps_per_file`` are unset,
            ``steps_per_file``'s default value of 1 (whole file per chunk/partition) is used,
            regardless of ``open_files``.
        library (str or :doc:`uproot.interpretation.library.Library`): The library
            that is used to represent arrays. If ``library='np'`` it returns a dict
            of dask arrays and if ``library='ak'`` it returns a single dask-awkward
            array. ``library='pd'`` has not been implemented yet and will raise a
            ``NotImplementedError``.
        ak_add_doc (bool | dict ): If True and ``library="ak"``, add the TBranch ``title``
            or RField ``description`` to the Awkward ``__doc__`` parameter of the array.
            if dict = {key:value} and ``library="ak"``, add the TBranch or RField ``value`` to
            the Awkward ``key`` parameter of the array. A ``value`` of ``"title"`` (the TBranch
            attribute) reads the RField ``description``.
        custom_classes (None or dict): If a dict, override the classes from
            the :doc:`uproot.reading.ReadOnlyFile` or ``uproot.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        open_files (bool): If True (default), the function will open the files to read file
            metadata, i.e. only the main data read is delayed till the compute call on
            the dask collections. If False, the opening of the files and reading the
            metadata is also delayed till the compute call. In this case, branch-names
            are inferred by opening only the first file.
        form_mapping (Callable[awkward.forms.Form] -> awkward.forms.Form | None): If not none
            and library="ak" then apply this remapping function to the awkward form of the input
            data. The form keys of the desired form should be available data in the input form.
        allow_read_errors_with_report (bool or tuple of exceptions): If True, catch OSError
            exceptions and return an empty array for these nodes in the task graph. If a tuple,
            catch any of those exceptions and return empty arrays for those nodes. In either of
            those cases, The return of this function becomes a two element tuple, where the
            first return is the dask-awkward collection of interest and the second return is a
            report dask-awkward collection. A failed read of a coalesced range keeps the
            nearest built-in OSError type of the original error, or else is a
            :doc:`uproot.exceptions.CoalescedReadError`. A tuple also matches against that
            original error, and the report includes it in the message.
        known_base_form (awkward.forms.Form | None): If not none use this form instead of opening
            one file to determine the dataset's form. Only available with open_files=False.
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, a
            :doc:`uproot.source.futures.TrivialExecutor` is created.
            This option is primarily useful for very large, CPU-intensive TBaskets
            (e.g. LZMA compression) and can lead to nested parallelism, leaving it as None is recommended
            for most cases. These options are particularly useful in very high thread-count scenarios,
            such as when using Python 3.13 free-threading.
            Executors attached to a file are ``shutdown`` when the file is closed.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, a :doc:`uproot.source.futures.TrivialExecutor`
            is created.
            It is primarily useful when interpretation is CPU-intensive (e.g. for certain Awkward
            ''AsObjects'') and can lead to nested parallelism, leaving it as None is recommended
            for most cases. These options are particularly useful in very high thread-count scenarios,
            such as when using Python 3.13 free-threading.
            Executors attached to a file are ``shutdown`` when the file is closed.
        options: See below.

    Returns dask equivalents of the backends supported by uproot. If ``library='np'``,
    the function returns a Python dict of dask arrays. If ``library='ak'``, the
    function returns a single dask-awkward array.

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

        >>> uproot.dask(root_file)
        dask.awkward<from-uproot, npartitions=1>
        >>> uproot.dask(root_file,library='np')
        {'Type': dask.array<Type-from-uproot, shape=(2304,), dtype=object, chunksize=(2304,), chunktype=numpy.ndarray>, ...}

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
      Example: ``{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}``
    * dict: keys are filesystem paths, values are dicts containing objects-within-ROOT and
      steps (chunks/partitions) as a list of starts and stops or steps as a list of offsets
      Example:

          {{"/data_v1/tree1.root": {"object_path": "ttree_v1", "steps": [[0, 10000], [15000, 20000], ...]},
            "/data_v1/tree2.root": {"object_path": "ttree_v1", "steps": [0, 10000, 20000, ...]}}}

      (This ``files`` pattern is incompatible with ``step_size`` and ``steps_per_file``.)
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
    * :doc:`uproot.behaviors.TBranch.concatenate`: returns a single
      concatenated array from ``TTrees``.
    * :doc:`uproot._dask.dask` (this function): returns an unevaluated Dask
      array from ``TTrees``.
    """

    uproot._util._check_unknown_options(options)
    files = uproot._util.regularize_files(files, steps_allowed=True, **options)

    is_3arg = [len(x) == 3 for x in files]
    if any(is_3arg):
        if not all(is_3arg):
            raise TypeError(
                "partition sizes for some but not all 'files' have been assigned"
            )
        if step_size is not unset:
            raise TypeError(
                "partition sizes for 'files' is incompatible with 'step_size'"
            )
        if steps_per_file is not unset:
            raise TypeError(
                "partition sizes for 'files' is incompatible with 'steps_per_file'"
            )

    library = uproot.interpretation.library._regularize_library(library)

    if step_size is not unset and steps_per_file is not unset:
        raise TypeError(
            f"only 'step_size' or 'steps_per_file' should be set, not both; got step_size={step_size!r} and steps_per_file={steps_per_file!r}"
        )
    elif step_size is not unset:
        if not open_files:
            # the not open_files case FAILS if only step_size is supplied
            raise TypeError(
                f"'step_size' should not be set when 'open_files' is False; got {step_size!r}"
            )
        else:
            # the open_files case uses step_size (only)
            uproot._util.check_step_size(step_size)
    elif steps_per_file is not unset:
        # the not open_files case uses steps_per_file (only)
        # the open_files case converts steps_per_file into step_size
        pass
    else:
        steps_per_file = 1

    if known_base_form is not None and open_files:
        raise TypeError("known_base_form must be None if open_files is True")

    if library.name == "pd":
        raise NotImplementedError()

    if library.name != "ak" and form_mapping is not None:
        raise NotImplementedError()

    real_options = options.copy()
    if "num_workers" not in real_options:
        real_options["num_workers"] = 1
    if "num_fallback_workers" not in real_options:
        real_options["num_fallback_workers"] = 1

    filter_branch = uproot._util.regularize_filter(filter_branch)

    interp_options = {"ak_add_doc": ak_add_doc}

    if library.name == "np":
        if open_files:
            return _get_dask_array(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                step_size,
                custom_classes,
                allow_missing,
                real_options,
                interp_options,
                steps_per_file,
                decompression_executor,
                interpretation_executor,
            )
        else:
            return _get_dask_array_delay_open(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                custom_classes,
                allow_missing,
                real_options,
                interp_options,
                steps_per_file,
                decompression_executor,
                interpretation_executor,
            )
    elif library.name == "ak":
        if open_files:
            return _get_dak_array(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                step_size,
                custom_classes,
                allow_missing,
                real_options,
                interp_options,
                form_mapping,
                steps_per_file,
                allow_read_errors_with_report,
                decompression_executor,
                interpretation_executor,
            )
        else:
            return _get_dak_array_delay_open(
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                custom_classes,
                allow_missing,
                real_options,
                interp_options,
                form_mapping,
                steps_per_file,
                allow_read_errors_with_report,
                known_base_form,
                decompression_executor,
                interpretation_executor,
            )
    else:
        raise NotImplementedError()


class _PackedArgCallable:
    """Wrap a callable such that packed arguments can be unrolled.
    Inspired by dask.dataframe.io.io._PackedArgCallable.
    """

    def __init__(
        self,
        func: Callable,
        args=None,
        kwargs=None,
        packed: bool = False,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.packed = packed

    def __call__(self, packed_arg):
        if not self.packed:
            packed_arg = (packed_arg,)
        return self.func(
            *packed_arg,
            *(self.args or []),
            **(self.kwargs or {}),
        )


class _LazyInputsDict(Mapping):
    """Dictionary with lazy key value pairs
    Parameters
    ----------
    inputs : list[Any]
        The list of dicionary values.
    """

    def __init__(self, inputs, **kwargs) -> None:
        self.inputs = inputs
        self.kwargs = kwargs

    def __len__(self):
        return len(self.inputs)

    def __iter__(self):
        return (self[k] for k in self.keys())

    def __getitem__(self, i):
        return self.inputs[i[0]]

    def __contains__(self, k):
        if isinstance(k, tuple) and isinstance(k[0], int):
            return k[0] >= 0 and k[0] < len(self)
        return False

    def keys(self):
        return ((i,) for i in range(len(self.inputs)))


def _dask_array_from_map(
    func,
    *iterables,
    chunks,
    dtype,
    inner_shape=(),
    args=None,
    label=None,
    token=None,
    **kwargs,
):
    dask = uproot.extras.dask()
    _dask_uses_tasks = hasattr(dask, "_task_spec")

    da = uproot.extras.dask_array()
    if not callable(func):
        raise ValueError("`func` argument must be `callable`")
    lengths = set()
    iters = list(iterables)
    for i, iterable in enumerate(iters):
        if not isinstance(iterable, Iterable):
            raise ValueError(
                f"All elements of `iterables` must be Iterable, got {type(iterable)}"
            )
        try:
            lengths.add(len(iterable))  # type: ignore
        except (AttributeError, TypeError):
            iters[i] = list(iterable)
            lengths.add(len(iters[i]))  # type: ignore
    if len(lengths) == 0:
        raise ValueError("`from_map` requires at least one Iterable input")
    elif len(lengths) > 1:
        raise ValueError("All `iterables` must have the same length")
    if lengths == {0}:
        raise ValueError("All `iterables` must have a non-zero length")

    # Check for `produces_tasks` and `creation_info`
    produces_tasks = kwargs.pop("produces_tasks", False)
    # creation_info = kwargs.pop("creation_info", None)

    if produces_tasks or len(iters) == 1:
        if len(iters) > 1:
            # Tasks are not detected correctly when they are "packed"
            # within an outer list/tuple
            raise ValueError(
                "Multiple iterables not supported when produces_tasks=True"
            )
        inputs = list(iters[0])
        packed = False
    else:
        # Structure inputs such that the tuple of arguments pair each 0th,
        # 1st, 2nd, ... elements together; for example:
        # from_map(f, [1, 2, 3], [4, 5, 6]) --> [f(1, 4), f(2, 5), f(3, 6)]
        inputs = list(zip(*iters, strict=True))
        packed = True

    # Define collection name
    label = label or dask.utils.funcname(func)
    token = token or dask.base.tokenize(func, iters, **kwargs)
    name = f"{label}-{token}"

    # Define io_func
    if packed or args or kwargs:
        io_func = _PackedArgCallable(
            func,
            args=args,
            kwargs=kwargs,
            packed=packed,
        )
    else:
        io_func = func

    io_arg_map = dask.blockwise.BlockwiseDepDict(
        mapping=_LazyInputsDict(inputs),  # type: ignore
        produces_tasks=produces_tasks,
    )

    chunks = (*chunks, *((size,) for size in inner_shape))
    assert all(len(axis_chunks) == 1 for axis_chunks in chunks[1:])
    blockwise_kwargs = {
        "output": name,
        "output_indices": tuple(range(len(chunks))),
        "indices": [(io_arg_map, (0,))],
        "new_axes": {axis: chunks[axis] for axis in range(1, len(chunks))},
        "numblocks": {},
        "annotations": None,
    }

    if _dask_uses_tasks:
        blockwise_kwargs["task"] = dask._task_spec.Task(
            name, io_func, dask._task_spec.TaskRef(dask.blockwise.blockwise_token(0))
        )
    else:
        blockwise_kwargs["dsk"] = {name: (io_func, dask.blockwise.blockwise_token(0))}

    dsk = dask.blockwise.Blockwise(**blockwise_kwargs)

    hlg = dask.highlevelgraph.HighLevelGraph.from_collections(name, dsk)
    return da.core.Array(hlg, name, chunks, dtype=dtype)


def _coerce_field(col, target_dtype):
    if target_dtype.base == numpy.dtype(object):
        if target_dtype.shape == () and col.ndim > 1:
            col_out = numpy.empty(len(col), dtype=object)
            if numpy.ma.isMaskedArray(col):
                mask = numpy.ma.getmaskarray(col)
                for i in range(len(col)):
                    m = mask[i]
                    if m.all():
                        col_out[i] = None
                    elif m.any():
                        row = col.data[i].astype(object)
                        row[m] = None
                        col_out[i] = row
                    else:
                        col_out[i] = col.data[i]
            else:
                for i in range(len(col)):
                    col_out[i] = col[i]
            return col_out
        elif numpy.ma.isMaskedArray(col):
            col_obj = col.data.astype(object)
            col_obj[numpy.ma.getmaskarray(col)] = None
            return col_obj
        elif col.dtype != object:
            return col.astype(object)
        else:
            return col
    else:
        if target_dtype.shape and col.shape[1:] == target_dtype.shape:
            if col.dtype != target_dtype.base:
                return col.astype(target_dtype.base)
            return col
        if col.dtype != target_dtype:
            return col.astype(target_dtype)
        return col


def _coerce_numpy_chunk(arr, expected_dtype):
    if expected_dtype is None:
        return arr
    if expected_dtype.names is not None:
        if arr.dtype != expected_dtype:
            out = numpy.empty(len(arr), dtype=expected_dtype)
            for name in expected_dtype.names:
                out[name] = _coerce_field(arr[name], expected_dtype[name])
            return out
        return arr
    return _coerce_field(arr, expected_dtype)


class _UprootReadNumpy:
    def __init__(
        self,
        ttrees,
        key,
        interp_options,
        expected_dtype=None,
        decompression_executor=None,
        interpretation_executor=None,
    ) -> None:
        self.ttrees = ttrees
        self.key = key
        self.interp_options = interp_options
        self.expected_dtype = expected_dtype
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        arr = self.ttrees[i][self.key].array(
            entry_start=start,
            entry_stop=stop,
            library="np",
            ak_add_doc=self.interp_options["ak_add_doc"],
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
        )
        return _coerce_numpy_chunk(arr, self.expected_dtype)


def _branch_or_field_numpy_dtype(branch):
    if isinstance(branch, HasFields):
        form, path = branch.to_akform()
        sizes = []
        if path is not None:
            for p in path:
                # Step through list/option forms to reach the record inside.
                while isinstance(
                    form,
                    (
                        awkward.forms.ListOffsetForm,
                        awkward.forms.ListForm,
                        awkward.forms.RegularForm,
                        *_OPTION_FORMS,
                    ),
                ):
                    # Any list boundary means the result is a jagged array.
                    if isinstance(
                        form,
                        (
                            awkward.forms.ListOffsetForm,
                            awkward.forms.ListForm,
                        ),
                    ):
                        return numpy.dtype(object)
                    if isinstance(form, awkward.forms.RegularForm):
                        sizes.append(form.size)
                    form = form.content
                if hasattr(form, "fields") and p in form.fields:
                    form = form.contents[form.fields.index(p)]
                else:
                    raise KeyError(f"field {p!r} not found in form {form!r}")
        return _form_to_numpy_dtype(form, initial_sizes=sizes)
    else:
        return branch.interpretation.numpy_dtype


def _check_numpy_dtype(branch, expected_dtype):
    actual_dtype = _branch_or_field_numpy_dtype(branch)
    if actual_dtype != expected_dtype:
        if isinstance(branch, HasFields):
            item_name = "field"
            path = branch.path
            file_path = branch.ntuple.parent._file.file_path
        else:
            item_name = "branch"
            path = branch.object_path
            file_path = branch.file.file_path
        raise ValueError(
            f"inconsistent NumPy dtype for {item_name} {path!r} in file "
            f"{file_path!r}: expected {expected_dtype!r}, "
            f"found {actual_dtype!r}"
        )


class _UprootOpenAndReadNumpy:
    def __init__(
        self,
        custom_classes,
        allow_missing,
        real_options,
        key,
        interp_options,
        expected_dtype,
        decompression_executor=None,
        interpretation_executor=None,
    ):
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.key = key
        self.interp_options = interp_options
        self.expected_dtype = expected_dtype
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def __call__(self, file_path_object_path_istep_nsteps_ischunk):
        (
            file_path,
            object_path,
            istep_or_start,
            nsteps_or_stop,
            ischunk,
        ) = file_path_object_path_istep_nsteps_ischunk
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        _check_numpy_dtype(ttree[self.key], self.expected_dtype)
        num_entries = ttree.num_entries
        start, stop = istep_or_start, nsteps_or_stop
        if not ischunk:
            events_per_steps = math.ceil(num_entries / nsteps_or_stop)
            start, stop = (
                (istep_or_start * events_per_steps),
                min((istep_or_start + 1) * events_per_steps, num_entries),
            )
        elif (not 0 <= start <= num_entries) or (not 0 <= stop <= num_entries):
            raise ValueError(
                f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {num_entries} entries"""
            )

        arr = ttree[self.key].array(
            library="np",
            entry_start=start,
            entry_stop=stop,
            ak_add_doc=self.interp_options["ak_add_doc"],
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
        )
        return _coerce_numpy_chunk(arr, self.expected_dtype)


def _regularize_tree_step_size(ttree, step_size, entry_start, entry_stop, common_keys):
    if isinstance(ttree, HasFields):
        akform, _ = ttree.to_akform(
            filter_field=_filter_field_by_key(ttree, common_keys)
        )
        return _RNTuple_regularize_step_size(
            ttree, akform, step_size, entry_start, entry_stop
        )
    else:
        branchid_interpretation = {}
        for key in common_keys:
            branch = ttree[key]
            branchid_interpretation[branch.cache_key] = branch.interpretation
        return _regularize_step_size(
            ttree, step_size, entry_start, entry_stop, branchid_interpretation
        )


def _get_dask_array(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    step_size,
    custom_classes,
    allow_missing,
    real_options,
    interp_options,
    steps_per_file,
    decompression_executor,
    interpretation_executor,
):
    ttrees, common_keys, _, explicit_chunks = _resolve_trees_and_keys(
        files,
        filter_name,
        filter_typename,
        filter_branch,
        recursive,
        full_paths,
        custom_classes,
        allow_missing,
        real_options,
    )

    # this is the earliest time we can deal with an unset step_size
    if step_size is unset:
        assert steps_per_file is not unset  # either assigned or assumed to be 1
        total_files = len(ttrees)
        total_entries = sum(ttree.num_entries for ttree in ttrees)
        step_size = max(1, math.ceil(total_entries / (total_files * steps_per_file)))

    dask_dict = {}

    step_sum = 0
    for ttree in ttrees:
        entry_start = 0
        entry_stop = ttree.num_entries
        ttree_step = _regularize_tree_step_size(
            ttree, step_size, entry_start, entry_stop, common_keys
        )
        step_sum += int(ttree_step)

    entry_step = round(step_sum / len(ttrees))
    assert entry_step >= 1

    for key in common_keys:
        expected_dtype = _branch_or_field_numpy_dtype(ttrees[0][key])
        for ttree in ttrees[1:]:
            _check_numpy_dtype(ttree[key], expected_dtype)
        dt, inner_shape = uproot.interpretation.numerical._dtype_shape(expected_dtype)

        chunks = []
        chunk_args = []
        for i, ttree in enumerate(ttrees):
            entry_start = 0
            entry_stop = ttree.num_entries

            if explicit_chunks is None:
                for start in range(entry_start, entry_stop, entry_step):
                    stop = min(start + entry_step, entry_stop)
                    length = stop - start
                    if length > 0:
                        chunks.append(length)
                        chunk_args.append((i, start, stop))
            else:
                for start, stop in explicit_chunks[i]:
                    if (not 0 <= start <= entry_stop) or (not 0 <= stop <= entry_stop):
                        raise ValueError(
                            f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {entry_stop} entries"""
                        )
                    length = stop - start
                    if length > 0:
                        chunks.append(length)
                        chunk_args.append((i, start, stop))

        if len(chunk_args) == 0:
            chunks.append(0)
            chunk_args.append((0, 0, 0))

        dask_dict[key] = _dask_array_from_map(
            _UprootReadNumpy(
                ttrees,
                key,
                interp_options,
                expected_dtype,
                decompression_executor,
                interpretation_executor,
            ),
            chunk_args,
            chunks=(tuple(chunks),),
            dtype=dt,
            inner_shape=inner_shape,
            label=f"{key}-from-uproot",
        )

    return dask_dict


def _get_dask_array_delay_open(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    custom_classes,
    allow_missing,
    real_options,
    interp_options,
    steps_per_file,
    decompression_executor,
    interpretation_executor,
):
    ffile_path, fobject_path = files[0][0:2]
    obj = uproot._util.regularize_object_path(
        ffile_path, fobject_path, custom_classes, allow_missing, real_options
    )
    common_keys = obj.keys(
        recursive=recursive,
        filter_name=filter_name,
        filter_typename=filter_typename,
        **{
            (
                "filter_field" if isinstance(obj, HasFields) else "filter_branch"
            ): filter_branch
        },
        # an RNTuple form keeps the nesting, so its fields are selected by full path
        full_paths=True if isinstance(obj, HasFields) else full_paths,
        ignore_duplicates=True,
    )
    if isinstance(obj, HasFields):
        common_keys = _with_subfield_keys(obj, common_keys)

    dask_dict = {}

    for key in common_keys:
        expected_dtype = _branch_or_field_numpy_dtype(obj[key])
        dt, inner_shape = uproot.interpretation.numerical._dtype_shape(expected_dtype)

        partitions = []
        partition_args = []
        for ifile_iobject_maybeichunks in files:
            ifile_path, iobject_path = ifile_iobject_maybeichunks[0:2]

            chunks = None
            if len(ifile_iobject_maybeichunks) == 3:
                chunks = ifile_iobject_maybeichunks[2]

            if chunks is not None:
                partitions.extend([stop - start for start, stop in chunks])
                for start, stop in chunks:
                    partition_args.append(
                        (
                            ifile_path,
                            iobject_path,
                            start,
                            stop,
                            True,
                        )
                    )
            else:
                partitions.extend([numpy.nan] * steps_per_file)
                for istep in range(steps_per_file):
                    partition_args.append(
                        (
                            ifile_path,
                            iobject_path,
                            istep,
                            steps_per_file,
                            False,
                        )
                    )

        dask_dict[key] = _dask_array_from_map(
            _UprootOpenAndReadNumpy(
                custom_classes,
                allow_missing,
                real_options,
                key,
                interp_options,
                expected_dtype,
                decompression_executor,
                interpretation_executor,
            ),
            partition_args,
            chunks=(tuple(partitions),),
            dtype=dt,
            inner_shape=inner_shape,
            label=f"{key}-from-uproot",
        )
    return dask_dict


class ImplementsFormMappingInfo(Protocol):
    @property
    def behavior(self) -> dict | None: ...

    buffer_key: str | Callable

    def parse_buffer_key(self, buffer_key: str) -> tuple[str, str]: ...

    def keys_for_buffer_keys(self, buffer_keys: frozenset[str]) -> frozenset[str]: ...

    def load_buffers(
        self,
        tree: HasBranches,
        keys: frozenset[str],
        start: int,
        stop: int,
        decompression_executor: Executor,
        interpretation_executor: Executor,
        options: Any,
    ) -> Mapping[str, AwkArray]: ...


class ImplementsFormMapping(Protocol):
    def __call__(self, form: Form) -> tuple[Form, ImplementsFormMappingInfo]: ...


class TrivialFormMappingInfo(ImplementsFormMappingInfo):
    def __init__(self, form, subfields=False):
        assert isinstance(form, awkward.forms.RecordForm)

        self._form = form
        # whether buffers are keyed by their dotted subfield paths (an RNTuple)
        self.subfields = subfields
        self._form_key_to_key = self.build_form_key_to_key(form, subfields)

    @property
    def behavior(self) -> None:
        return None

    @staticmethod
    def build_form_key_to_key(
        form: Form, subfields: bool = False
    ) -> dict[str, str | None]:
        """The key each form key's buffers are read with: its top-level field, or with
        ``subfields`` (an RNTuple, which reads each subfield on its own) its own node's dotted
        path, except that a union is read whole.
        """
        form_key_to_path: dict[str, str | None] = {}

        def impl(form, path, split):
            form_key_to_path[form.form_key] = path
            if form.is_record and split:
                for field in form.fields:
                    subpath = field if path is None else f"{path}.{field}"
                    impl(form.content(field), subpath, subfields)
            else:
                for content in _contents(form):
                    impl(content, path, split and not form.is_union)

        impl(form, None, True)
        return form_key_to_path

    buffer_key: Final[str] = "{form_key}-{attribute}"

    def parse_buffer_key(self, buffer_key: str) -> tuple[str, str]:
        form_key, attribute = buffer_key.rsplit("-", maxsplit=1)
        return form_key, attribute

    def keys_for_buffer_keys(self, buffer_keys: frozenset[str]) -> frozenset[str]:
        keys: set[str] = set()
        for buffer_key in buffer_keys:
            # Identify form key
            form_key, _attribute = buffer_key.rsplit("-", maxsplit=1)
            # Identify key from form_key
            keys.add(self._form_key_to_key[form_key])
        # reading a subfield reads the collections above it, so they need not be read whole
        if self.subfields:
            keys -= _ancestors(keys)
        return frozenset(keys)

    def load_buffers(
        self,
        tree: HasBranches,
        keys: frozenset[str],
        start: int,
        stop: int,
        decompression_executor,
        interpretation_executor,
        options: Any,
    ) -> Mapping[str, AwkArray]:
        # An RNTuple reads every leaf beneath a requested key, selected by key because a name
        # like "x" would also select the fields named "x" in other records.
        if isinstance(tree, HasFields):
            # by field-name tuple, so braces and glob characters in a name are literal; a
            # top-level field is selected whole, as a dot in its name (which ROOT forbids,
            # but uproot can write) is only supported there
            form = _tuples_as_records(self._form)
            names = ((k,) if form.has_field(k) else k.split(".") for k in keys)
            form = form.select_columns([tuple(map(glob.escape, n)) for n in names])
            projection_subforms = {f: form.content(f) for f in form.fields}
            leaf_keys = [
                p
                for field, subform in projection_subforms.items()
                for p in _rntuple_leaf_paths(subform, field)
            ]
            selection = {"filter_field": _filter_field_by_key(tree, leaf_keys)}
        else:
            fields = list(keys)
            if self.subfields:  # a TTree reads the whole branches the keys are under
                above = _ancestors(keys)
                fields = [f for f in self._form.fields if f in keys or f in above]
            projection_subforms = {f: self._form.content(f) for f in fields}
            selection = {"expressions": fields}
        # Read the arrays as a top-level awkward RecordArray. Omitting how= (the
        # default) ensures that AsGrouped branches are returned as proper awkward
        # RecordArrays rather than Python tuples of sub-arrays (which how=tuple
        # would produce), allowing awkward.to_buffers() to work correctly below.
        arrays = tree.arrays(
            **selection,
            entry_start=start,
            entry_stop=stop,
            ak_add_doc=options["ak_add_doc"],
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )

        # The subform generated by awkward.to_buffers() has different form keys
        # from those used to perform buffer projection. However, the subform
        # structure should be identical to the projection optimisation
        # subform, as they're derived from `branch.interpretation.awkward_form`
        # Therefore, we can correlate the subform keys using `expected_from_buffers`
        container = {}
        for key, projection_subform in projection_subforms.items():
            # First, convert the sub-array into buffers
            ttree_subform, _length, ttree_container = awkward.to_buffers(arrays[key])

            # Correlate each TTree form key with the projection form key
            for (src, src_dtype), (dst, dst_dtype) in zip(
                ttree_subform.expected_from_buffers().items(),
                projection_subform.expected_from_buffers(self.buffer_key).items(),
                strict=True,
            ):
                if src_dtype != dst_dtype:
                    raise TypeError(f"Data type mismatch: {src_dtype} != {dst_dtype}")
                container[dst] = ttree_container[src]

        return container


def _contents(form):
    if form.is_record or form.is_union:
        return form.contents
    if form.is_list or form.is_option or form.is_indexed:
        return [form.content]
    return []


def _tuples_as_records(form):
    """``form`` with its tuples as records with the same field names, since Awkward renumbers a
    subset of a tuple's fields and they would no longer match the RNTuple's fields."""
    if form.is_record:
        return form.copy(
            contents=[_tuples_as_records(c) for c in form.contents], fields=form.fields
        )
    if form.is_union:
        return form.copy(contents=[_tuples_as_records(c) for c in form.contents])
    if form.is_list or form.is_option or form.is_indexed:
        return form.copy(content=_tuples_as_records(form.content))
    return form


def _ancestors(keys):
    """The dotted paths strictly above ``keys``."""
    return {key[:i] for key in keys for i, c in enumerate(key) if c == "."}


def _subfield_paths(form, path=None):
    """The dotted paths of the record fields beneath ``form``'s top-level fields."""
    if form.is_record:
        for field in form.fields:
            subpath = field if path is None else f"{path}.{field}"
            if path is not None:
                yield subpath
            yield from _subfield_paths(form.content(field), subpath)
    elif form.is_list or form.is_option or form.is_indexed:
        yield from _subfield_paths(form.content, path)


def _covered(key, keys):
    """Whether reading ``keys`` reads ``key``: an RNTuple field is read with its subfields."""
    while key not in keys:
        key, dot, _ = key.rpartition(".")
        if not dot:
            return False
    return True


def _rntuple_leaf_paths(form, prefix):
    """The dotted names ``RNTuple.keys()`` gives the leaves under ``prefix`` in ``form``; a
    collection or option level adds no segment."""
    if form.is_record:
        return [
            p
            for field in form.fields
            for p in _rntuple_leaf_paths(form.content(field), f"{prefix}.{field}")
        ]
    if form.is_union:
        return [p for c in form.contents for p in _rntuple_leaf_paths(c, prefix)]
    if form.is_list or form.is_option or form.is_indexed:
        return _rntuple_leaf_paths(form.content, prefix)
    return [prefix]


class FormMappingInfoWithVirtualArrays(TrivialFormMappingInfo):
    def buffer_replacements(
        self,
        tree: HasBranches,
        keys: frozenset[str],
        start: int,
        stop: int,
        decompression_executor,
        interpretation_executor,
        options: Any,
    ) -> Mapping[str, AwkArray]:

        def generator(tree, buffer_key):
            form_key, _ = self.parse_buffer_key(buffer_key)
            key = self._form_key_to_key[form_key]

            def _generator():
                array = self.load_buffers(
                    tree,
                    frozenset({key}),
                    start,
                    stop,
                    decompression_executor,
                    interpretation_executor,
                    options,
                )[buffer_key]

                # add to access_log
                access_log = options.get("access_log")
                if access_log is not None:
                    if not hasattr(access_log, "__iadd__"):
                        raise ValueError(
                            f"{access_log=} needs to implement '__iadd__'."
                        )
                    else:
                        access_log += [Accessed(branch=key, buffer_key=buffer_key)]

                return array

            return _generator

        container = {}
        for buffer_key in self._form.expected_from_buffers():
            container[buffer_key] = generator(tree, buffer_key)

        return container


class TrivialFormMapping(ImplementsFormMapping):
    def __call__(self, form: Form) -> tuple[Form, TrivialFormMappingInfo]:
        new_form = awkward.forms.form_with_unique_keys(form, ("<root>",))
        return new_form, TrivialFormMappingInfo(new_form)


class FormMappingWithVirtualArrays(ImplementsFormMapping):
    def __call__(self, form: Form) -> tuple[Form, FormMappingInfoWithVirtualArrays]:
        new_form = awkward.forms.form_with_unique_keys(form, ("<root>",))
        return new_form, FormMappingInfoWithVirtualArrays(new_form)


T = TypeVar("T")


def _read_tree(
    tree,
    keys,
    start,
    stop,
    *,
    expected_form,
    form_mapping_info,
    interp_options,
    decompression_executor,
    interpretation_executor,
):
    """Read entries ``[start, stop)`` of ``keys`` from ``tree`` through a form mapping: the
    mapping loads the buffers of ``keys``; every other buffer of ``expected_form`` is filled by
    the mapping's ``buffer_replacements`` when it has them, else by a placeholder."""
    assert start <= stop

    from awkward._nplikes.numpy import Numpy

    nplike = Numpy.instance()

    # The remap implementation should correctly populate the generated
    # buffer mapping in __call__, such that the high-level form can be
    # used in `from_buffers`
    mapping = form_mapping_info.load_buffers(
        tree,
        keys,
        start,
        stop,
        decompression_executor,
        interpretation_executor,
        interp_options,
    )

    # Populate container with placeholders if keys aren't required
    # Otherwise, read from disk
    keys = frozenset(keys)
    # a dotted RNTuple key names a subfield, read with everything beneath it and the
    # structure above it; a dotted TTree branch name is just a name
    if isinstance(tree, HasFields) or getattr(form_mapping_info, "subfields", False):
        above = _ancestors(keys)
        read = lambda key: key in above or _covered(key, keys)  # noqa: E731
    else:
        read = keys.__contains__
    container = {}
    for buffer_key, dtype in expected_form.expected_from_buffers(
        buffer_key=form_mapping_info.buffer_key
    ).items():
        # Which key(s) does this buffer require. This code permits the caller
        # to require multiple keys to compute a single buffer.
        keys_for_buffer = form_mapping_info.keys_for_buffer_keys(
            frozenset({buffer_key})
        )
        # If reading this buffer loads a permitted key, read from the tree
        # We might not have _all_ keys if e.g. buffer A requires one
        # but not two of the keys required for buffer B
        if all(read(k) for k in keys_for_buffer):
            container[buffer_key] = mapping[buffer_key]
        # if the form mapping info provides a replacements, use it
        elif hasattr(form_mapping_info, "buffer_replacements"):
            container[buffer_key] = form_mapping_info.buffer_replacements(
                tree,
                keys_for_buffer,
                start,
                stop,
                decompression_executor,
                interpretation_executor,
                interp_options,
            )[buffer_key]
        # Otherwise, introduce a placeholder (default replacement)
        else:
            container[buffer_key] = awkward.typetracer.PlaceholderArray(
                nplike=nplike,
                shape=(awkward.typetracer.unknown_length,),
                dtype=dtype,
            )

    out = awkward.from_buffers(
        expected_form,
        stop - start,
        container,
        behavior=form_mapping_info.behavior,
        buffer_key=form_mapping_info.buffer_key,
    )
    return out


class UprootReadMixin:
    base_form: Form
    expected_form: Form
    form_mapping_info: ImplementsFormMappingInfo
    common_keys: frozenset[str]
    interp_options: dict[str, Any]
    allow_read_errors_with_report: bool | tuple[type[BaseException], ...]

    @property
    def allowed_exceptions(self):
        if isinstance(self.allow_read_errors_with_report, tuple):
            return self.allow_read_errors_with_report
        return (OSError,)

    @property
    def caught_exceptions(self):
        # coalesced read errors are OSErrors that can match on their original error
        return (*self.allowed_exceptions, OSError)

    def is_allowed_exception(self, err: BaseException) -> bool:
        if isinstance(err, self.allowed_exceptions):
            return True
        return getattr(err, "coalesced_read", False) and isinstance(
            err.__cause__, self.allowed_exceptions
        )

    @property
    def return_report(self) -> bool:
        return bool(self.allow_read_errors_with_report)

    def read_tree(
        self, tree: HasBranches, start: int, stop: int
    ) -> tuple[AwkArray, SourcePerformanceCounters]:
        out = _read_tree(
            tree,
            self.common_keys,
            start,
            stop,
            expected_form=self.expected_form,
            form_mapping_info=self.form_mapping_info,
            interp_options=self.interp_options,
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
        )
        assert tree.source  # we must be reading something here
        return out, tree.source.performance_counters

    def mock(self) -> AwkArray:
        return awkward.typetracer.typetracer_from_form(
            self.expected_form,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
        )

    def mock_empty(self, backend) -> AwkArray:
        return awkward.to_backend(
            self.expected_form.length_zero_array(highlevel=False),
            backend=backend,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
        )

    def prepare_for_projection(self) -> tuple[AwkArray, TypeTracerReport, dict]:
        dask_awkward = uproot.extras.dask_awkward()

        # A form mapping will (may) remap the base form into a new form
        # The remapped form can be queried for structural information

        # Build typetracer and associated report object
        meta, report = awkward.typetracer.typetracer_with_report(
            self.expected_form,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
            buffer_key=self.form_mapping_info.buffer_key,
        )

        return (
            meta,
            report,
            {
                "trace": dask_awkward.lib.utils.trace_form_structure(
                    self.expected_form,
                    buffer_key=self.form_mapping_info.buffer_key,
                ),
                "form_info": self.form_mapping_info,
            },
        )

    def project(self: T, *, report: TypeTracerReport, state: dict) -> T:
        keys = self.necessary_columns(report=report, state=state)
        return self.project_keys(keys)

    def project_manually(self: T, columns: frozenset[str]) -> T:
        return self.project_keys(columns)

    def necessary_columns(
        self, *, report: TypeTracerReport, state: dict
    ) -> frozenset[str]:
        ## Read from stash
        # Form hierarchy information
        form_key_to_parent_form_key: dict = state["trace"][
            "form_key_to_parent_form_key"
        ]
        # Buffer hierarchy information
        form_key_to_buffer_keys: dict = state["trace"]["form_key_to_buffer_keys"]
        # Restructured form information
        form_info = state["form_info"]

        # Require the data of metadata buffers above shape-only requests
        dask_awkward = uproot.extras.dask_awkward()
        data_buffers = {
            *report.data_touched,
            *dask_awkward.lib.utils.buffer_keys_required_to_compute_shapes(
                form_info.parse_buffer_key,
                report.shape_touched,
                form_key_to_parent_form_key,
                form_key_to_buffer_keys,
            ),
        }

        # Determine which TTree keys need to be read; an RNTuple's are read by subfield too
        common_keys = frozenset(self.common_keys)
        subfields = frozenset(_subfield_paths(self.base_form))
        return frozenset(
            key
            for key in form_info.keys_for_buffer_keys(data_buffers)
            if key in common_keys or (key in subfields and _covered(key, common_keys))
        )

    def project_keys(self: T, keys: frozenset[str]) -> T:
        raise NotImplementedError


def _report_failure(exception, call_time, *args, **kwargs):
    return awkward.Array(
        [
            {
                "call_time": call_time,
                "duration": None,
                "performance_counters": None,
                "args": [repr(a) for a in args],
                "kwargs": [[k, repr(v)] for k, v in kwargs.items()],
                "exception": type(exception).__name__,
                "message": str(exception),
                "fqdn": socket.getfqdn(),
                "hostname": socket.gethostname(),
            }
        ]
    )


def _report_success(duration, *args, **kwargs):
    counters = kwargs.pop("counters")
    return awkward.Array(
        [
            {
                "call_time": None,
                "duration": duration,
                "performance_counters": counters.asdict(),
                "args": [repr(a) for a in args],
                "kwargs": [[k, repr(v)] for k, v in kwargs.items()],
                "exception": None,
                "message": None,
                "fqdn": None,
                "hostname": None,
            }
        ]
    )


def with_duration(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = f(*args, **kwargs)
        stop = time.monotonic()
        return result, (stop - start)

    return wrapper


class _UprootRead(UprootReadMixin):
    def __init__(
        self,
        ttrees,
        common_keys: frozenset[str],
        interp_options: dict[str, Any],
        base_form: Form,
        expected_form: Form,
        form_mapping_info: ImplementsFormMappingInfo,
        allow_read_errors_with_report: bool | tuple[type[BaseException], ...],
        decompression_executor,
        interpretation_executor,
    ) -> None:
        self.ttrees = ttrees
        self.common_keys = frozenset(common_keys)
        self.interp_options = interp_options
        self.base_form = base_form
        self.expected_form = expected_form
        self.form_mapping_info = form_mapping_info
        self.allow_read_errors_with_report = allow_read_errors_with_report
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def project_keys(self: T, keys: frozenset[str]) -> T:
        return _UprootRead(
            self.ttrees,
            keys,
            self.interp_options,
            self.base_form,
            self.expected_form,
            self.form_mapping_info,
            self.allow_read_errors_with_report,
            self.decompression_executor,
            self.interpretation_executor,
        )

    def __call__(self, i_start_stop):
        i, start, stop = i_start_stop
        if self.return_report:
            call_time = time.time_ns()
            try:
                (result, counters), duration = with_duration(self._call_impl)(
                    i, start, stop
                )
                return (
                    result,
                    _report_success(
                        duration,
                        self.ttrees[i],
                        start,
                        stop,
                        counters=counters,
                    ),
                )
            except self.caught_exceptions as err:
                if not self.is_allowed_exception(err):
                    raise
                return (
                    self.mock_empty(backend="cpu"),
                    _report_failure(
                        err,
                        call_time,
                        self.ttrees[i],
                        start,
                        stop,
                    ),
                )

        result, _ = self._call_impl(i, start, stop)
        return result

    def _call_impl(self, i, start, stop):
        return self.read_tree(
            self.ttrees[i],
            start,
            stop,
        )


class _UprootOpenAndRead(UprootReadMixin):
    def __init__(
        self,
        custom_classes,
        allow_missing,
        real_options,
        common_keys,
        interp_options,
        base_form: Form,
        expected_form: Form,
        form_mapping_info: ImplementsFormMappingInfo,
        allow_read_errors_with_report: bool | tuple[type[BaseException], ...],
        decompression_executor,
        interpretation_executor,
    ) -> None:
        self.custom_classes = custom_classes
        self.allow_missing = allow_missing
        self.real_options = real_options
        self.common_keys = frozenset(common_keys)
        self.interp_options = interp_options
        self.base_form = base_form
        self.expected_form = expected_form
        self.form_mapping_info = form_mapping_info
        self.allow_read_errors_with_report = allow_read_errors_with_report
        self.decompression_executor = decompression_executor
        self.interpretation_executor = interpretation_executor

    def _call_impl(
        self, file_path, object_path, i_step_or_start, n_steps_or_stop, is_chunk
    ):
        ttree = uproot._util.regularize_object_path(
            file_path,
            object_path,
            self.custom_classes,
            self.allow_missing,
            self.real_options,
        )
        num_entries = ttree.num_entries
        if is_chunk:
            start, stop = i_step_or_start, n_steps_or_stop
            if (not 0 <= start <= num_entries) or (not 0 <= stop <= num_entries):
                raise ValueError(
                    f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {num_entries} entries"""
                )
        else:
            events_per_step = math.ceil(num_entries / n_steps_or_stop)
            start, stop = (
                min((i_step_or_start * events_per_step), num_entries),
                min((i_step_or_start + 1) * events_per_step, num_entries),
            )

        assert start <= stop

        return self.read_tree(
            ttree,
            start,
            stop,
        )

    def __call__(self, blockwise_args):
        (
            file_path,
            object_path,
            i_step_or_start,
            n_steps_or_stop,
            is_chunk,
        ) = blockwise_args

        if self.return_report:
            call_time = time.time_ns()
            try:
                (result, counters), duration = with_duration(self._call_impl)(
                    file_path, object_path, i_step_or_start, n_steps_or_stop, is_chunk
                )
                return (
                    result,
                    _report_success(
                        duration,
                        file_path,
                        object_path,
                        i_step_or_start,
                        n_steps_or_stop,
                        is_chunk,
                        counters=counters,
                    ),
                )
            except self.caught_exceptions as err:
                if not self.is_allowed_exception(err):
                    raise
                return (
                    self.mock_empty(backend="cpu"),
                    _report_failure(
                        err,
                        call_time,
                        file_path,
                        object_path,
                        i_step_or_start,
                        n_steps_or_stop,
                        is_chunk,
                    ),
                )

        result, _ = self._call_impl(
            file_path, object_path, i_step_or_start, n_steps_or_stop, is_chunk
        )
        return result

    def project_keys(self: T, keys: frozenset[str]) -> T:
        return _UprootOpenAndRead(
            self.custom_classes,
            self.allow_missing,
            self.real_options,
            keys,
            self.interp_options,
            self.base_form,
            self.expected_form,
            self.form_mapping_info,
            self.allow_read_errors_with_report,
            self.decompression_executor,
            self.interpretation_executor,
        )


def _leaf_to_tree(obj):
    """A leaf ``TBranch`` given as the object path stands for its parent ``TTree`` with that one
    branch selected: ``(tree, leaf)``, with ``leaf`` ``None`` for any other object."""
    if isinstance(obj, TBranch) and len(obj.keys(recursive=True)) == 0:
        return obj.parent, obj
    return obj, None


def _resolve_trees_and_keys(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    custom_classes,
    allow_missing,
    real_options,
):
    """Open every entry of ``files`` and intersect their keys (:func:`_trees_and_keys`)."""
    found = []
    for file_object_maybechunks in files:
        file_path, object_path = file_object_maybechunks[0:2]
        obj = uproot._util.regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )
        if obj is not None:
            found.append((obj, *file_object_maybechunks[2:]))
    return _trees_and_keys(
        found, files, filter_name, filter_typename, filter_branch, recursive, full_paths
    )


def _trees_and_keys(
    found,
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    first_of=None,
):
    """Intersect the keys of ``found``, the ``(object, *steps)`` of the entries of ``files``
    (named in errors) whose object exists; a leaf ``TBranch`` stands for its parent ``TTree`` with
    that one branch selected (``is_self``). Returns ``(ttrees, common_keys, is_self,
    explicit_chunks)``; ``explicit_chunks`` is ``None`` unless every entry carries its own
    ``steps``. ``first_of=N`` says ``found`` is only the first of ``N`` files, standing for all.
    """
    ttrees = []
    explicit_chunks = []
    common_keys = None
    is_self = []
    rntuple_keys = []

    for given, *steps in found:
        obj, original = _leaf_to_tree(given)
        if original is not None:
            is_self.append(True)

            def real_filter_branch(branch):
                return branch is original and filter_branch(branch)  # noqa: B023

        else:
            is_self.append(False)
            real_filter_branch = filter_branch

        ttrees.append(obj)
        if steps:
            explicit_chunks.append(steps[0])
        else:
            explicit_chunks = None  # they all have it or none of them have it

        new_keys = obj.keys(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            **{
                (
                    "filter_field" if isinstance(obj, HasFields) else "filter_branch"
                ): real_filter_branch
            },
            # an RNTuple form keeps the nesting, so its fields are selected by full path
            full_paths=True if isinstance(obj, HasFields) else full_paths,
            ignore_duplicates=True,
        )
        if isinstance(obj, HasFields):
            # a selected field is read with all of its subfields, so intersect those
            new_keys = _with_subfield_keys(obj, new_keys)
            rntuple_keys.append(set(new_keys))

        if common_keys is None:
            common_keys = new_keys
        else:
            new_keys = set(new_keys)
            common_keys = [key for key in common_keys if key in new_keys]

    if len(rntuple_keys) > 1:
        common_keys = _drop_partly_common_fields(common_keys, rntuple_keys)

    if len(ttrees) == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {}".format(
                "\n    ".join(
                    f"{{{f.file_path if isinstance(f, HasBranches) else f!r}: {f.object_path if isinstance(f, HasBranches) else o!r}}}"
                    for f, o, *_ in files
                )
            )
        )

    if first_of is not None and len(common_keys) == 0:
        tree = ttrees[0]
        raise ValueError(
            f"no TBranches of {{{tree.file.file_path!r}: {tree.object_path!r}}} pass "
            f"filter_name/filter_typename/filter_branch. Only this first file is opened, and its "
            f"TBranches stand for all {first_of} files: check the filters, or whether this file "
            "is representative of the rest (open_files=True checks every file)"
        )
    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    f"{{{f.file_path if isinstance(f, HasBranches) else f!r}: {f.object_path if isinstance(f, HasBranches) else o!r}}}"
                    for f, o, *_ in files
                )
            )
        )

    return ttrees, common_keys, is_self, explicit_chunks


def _drop_partly_common_fields(common_keys, keys_per_file):
    """Drop the RNTuple fields that have subfields missing from some of the files, keeping
    their common subfields. Otherwise, reading a field with all of its subfields would
    request subfields that some files don't have."""
    common = set(common_keys)
    partly_common = set()
    for keys in keys_per_file:
        for key in keys - common:
            parts = key.split(".")
            partly_common.update(".".join(parts[:i]) for i in range(1, len(parts)))
    return [key for key in common_keys if key not in partly_common]


def _normalize_grouped_keys(ttree, common_keys, full_paths):
    """Normalise AsGrouped branches in common_keys according to which of their children also
    appear in common_keys:

    - Case 1 (Parent matched, no leaves matched):    include parent grouped
    - Case 2 (Parent not matched, some leaves matched):   skip parent, keep matched leaves individual
    - Case 3 (Parent matched, all leaves matched):   skip parent, keep leaves individual
    - Case 4 (Parent matched, some leaves matched):  include parent grouped, suppress matched leaves
    """
    if isinstance(ttree, HasFields):
        return common_keys
    common_keys_set = set(common_keys)
    parent_keys_to_remove = set()
    child_keys_to_suppress = set()
    for k in common_keys:
        if not isinstance(
            ttree[k].interpretation,
            uproot.interpretation.grouped.AsGrouped,
        ):
            continue
        if full_paths:
            all_child_keys = [
                f"{k}/{ck}" for ck in ttree[k].keys(recursive=True, full_paths=True)
            ]
        else:
            all_child_keys = ttree[k].keys(recursive=True, full_paths=False)
        matched_children = [c for c in all_child_keys if c in common_keys_set]
        if len(all_child_keys) > 0 and len(matched_children) == len(all_child_keys):
            # Case 3: drop parent, keep individual children
            parent_keys_to_remove.add(k)
        elif matched_children:
            # Case 4: keep parent, suppress the individually-matched children
            child_keys_to_suppress.update(matched_children)
    return [
        k
        for k in common_keys
        if k not in parent_keys_to_remove and k not in child_keys_to_suppress
    ]


def _base_form_of(ttree, common_keys, ak_add_doc, form_mapping):
    """The dataset's base form from one open tree, with ``ak_add_doc`` applied; a ``TTree`` form
    also carries the branch typenames a mapping may read."""
    if isinstance(ttree, HasFields):
        # the form an eager RNTuple read builds, keeping the nesting of the selected fields
        return ttree.to_akform(
            filter_field=_filter_field_by_key(ttree, common_keys),
            ak_add_doc=ak_add_doc,
        )[0]
    base_form = uproot._util.get_ttree_form(ttree, common_keys, ak_add_doc)
    if form_mapping is not None:
        base_form.parameters["typenames"] = ttree.typenames()
    return base_form


def _mapping_of(form_mapping, base_form, subfields=False):
    """``(expected_form, form_mapping_info)``: the trivial mapping when none is given, which
    reads an RNTuple (``subfields``) by subfield."""
    if form_mapping is None:
        expected_form = awkward.forms.form_with_unique_keys(base_form, ("<root>",))
        return expected_form, TrivialFormMappingInfo(expected_form, subfields)
    return form_mapping(base_form)


def _get_dak_array(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    step_size,
    custom_classes,
    allow_missing,
    real_options,
    interp_options,
    form_mapping,
    steps_per_file,
    allow_read_errors_with_report,
    decompression_executor,
    interpretation_executor,
):
    dask_awkward = uproot.extras.dask_awkward()

    ttrees, common_keys, _, explicit_chunks = _resolve_trees_and_keys(
        files,
        filter_name,
        filter_typename,
        filter_branch,
        recursive,
        full_paths,
        custom_classes,
        allow_missing,
        real_options,
    )

    # this is the earliest time we can deal with an unset step_size
    if step_size is unset:
        assert steps_per_file is not unset  # either assigned or assumed to be 1
        total_files = len(ttrees)
        total_entries = sum(ttree.num_entries for ttree in ttrees)
        step_size = max(1, math.ceil(total_entries / (total_files * steps_per_file)))

    common_keys = _normalize_grouped_keys(ttrees[0], common_keys, full_paths)

    step_sum = 0
    for ttree in ttrees:
        entry_start = 0
        entry_stop = ttree.num_entries
        ttree_step = _regularize_tree_step_size(
            ttree, step_size, entry_start, entry_stop, common_keys
        )
        step_sum += int(ttree_step)

    entry_step = round(step_sum / len(ttrees))

    divisions = [0]
    partition_args = []
    for i, ttree in enumerate(ttrees):
        entry_start = 0
        entry_stop = ttree.num_entries

        if explicit_chunks is None:
            for start in range(entry_start, entry_stop, entry_step):
                stop = min(start + entry_step, entry_stop)
                length = stop - start
                if length > 0:
                    divisions.append(divisions[-1] + length)
                    partition_args.append((i, start, stop))
        else:
            for start, stop in explicit_chunks[i]:
                if (not 0 <= start <= entry_stop) or (not 0 <= stop <= entry_stop):
                    raise ValueError(
                        f"""explicit entry start ({start}) or stop ({stop}) from uproot.dask 'files' argument is out of bounds for file

    {ttree.file.file_path}

TTree in path

    {ttree.object_path}

which has {entry_stop} entries"""
                    )
                length = stop - start
                if length > 0:
                    divisions.append(divisions[-1] + length)
                    partition_args.append((i, start, stop))

    base_form = _base_form_of(
        ttrees[0], common_keys, interp_options.get("ak_add_doc"), form_mapping
    )

    if len(partition_args) == 0:
        divisions.append(0)
        partition_args.append((0, 0, 0))

    expected_form, form_mapping_info = _mapping_of(
        form_mapping, base_form, all(isinstance(t, HasFields) for t in ttrees)
    )

    fn = _UprootRead(
        ttrees,
        list(
            base_form.fields
        ),  # the keys a read fetches: for an RNTuple, its top-level fields
        interp_options,
        base_form=base_form,
        expected_form=expected_form,
        form_mapping_info=form_mapping_info,
        allow_read_errors_with_report=allow_read_errors_with_report,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    )

    out = dask_awkward.from_map(
        fn,
        partition_args,
        divisions=tuple(divisions),
        label="from-uproot",
    )
    if allow_read_errors_with_report:
        out[0].clear_divisions()
    return out


def _get_dak_array_delay_open(
    files,
    filter_name,
    filter_typename,
    filter_branch,
    recursive,
    full_paths,
    custom_classes,
    allow_missing,
    real_options,
    interp_options,
    form_mapping,
    steps_per_file,
    allow_read_errors_with_report,
    known_base_form,
    decompression_executor,
    interpretation_executor,
):
    dask_awkward = uproot.extras.dask_awkward()

    ffile_path, fobject_path = files[0][0:2]

    if known_base_form is not None:
        common_keys = list(dict.fromkeys(known_base_form.fields))
        base_form = known_base_form
        is_rntuple = False  # unknown without opening a file: read whole fields
    else:
        obj = uproot._util.regularize_object_path(
            ffile_path, fobject_path, custom_classes, allow_missing, real_options
        )
        common_keys = obj.keys(
            recursive=recursive,
            filter_name=filter_name,
            filter_typename=filter_typename,
            **{
                (
                    "filter_field" if isinstance(obj, HasFields) else "filter_branch"
                ): filter_branch
            },
            # an RNTuple form keeps the nesting, so its fields are selected by full path
            full_paths=True if isinstance(obj, HasFields) else full_paths,
            ignore_duplicates=True,
        )
        common_keys = _normalize_grouped_keys(obj, common_keys, full_paths)
        base_form = _base_form_of(
            obj, common_keys, interp_options.get("ak_add_doc"), form_mapping
        )
        is_rntuple = isinstance(obj, HasFields)

    divisions = [0]
    partition_args = []
    for ifile_iobject_maybeichunks in files:
        chunks = None
        ifile_path, iobject_path = ifile_iobject_maybeichunks[0:2]
        if len(ifile_iobject_maybeichunks) == 3:
            chunks = ifile_iobject_maybeichunks[2]

        if chunks is not None:
            for start, stop in chunks:
                divisions.append(divisions[-1] + (stop - start))
                partition_args.append(
                    (
                        ifile_path,
                        iobject_path,
                        start,
                        stop,
                        True,
                    )
                )
        else:
            divisions = None  # they all have it or none of them have it
            for istep in range(steps_per_file):
                partition_args.append(
                    (
                        ifile_path,
                        iobject_path,
                        istep,
                        steps_per_file,
                        False,
                    )
                )

    expected_form, form_mapping_info = _mapping_of(form_mapping, base_form, is_rntuple)

    fn = _UprootOpenAndRead(
        custom_classes,
        allow_missing,
        real_options,
        list(
            base_form.fields
        ),  # the keys a read fetches: for an RNTuple, its top-level fields
        interp_options,
        base_form=base_form,
        expected_form=expected_form,
        form_mapping_info=form_mapping_info,
        allow_read_errors_with_report=allow_read_errors_with_report,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    )

    out = dask_awkward.from_map(
        fn,
        partition_args,
        divisions=None if divisions is None else tuple(divisions),
        label="from-uproot",
    )
    if allow_read_errors_with_report:
        out[0].clear_divisions()
    return out


class Accessed(NamedTuple):
    branch: str
    buffer_key: str
