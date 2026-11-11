# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""
This module defines :doc:`uproot._graphed.graphed`, which reads ``TTrees`` into a deferred
`graphed <https://github.com/graphed-org/graphed>`__ array — the analogue of
:doc:`uproot._dask.dask` for the ``graphed`` task-graph system.

**Experimental.** ``uproot.graphed``, :doc:`uproot._graphed.graphed_head` and
:doc:`uproot.writing._graphed_write.graphed_write` track a pre-1.0 ``graphed`` and may change
without a deprecation cycle.

``uproot.graphed(files)`` returns a deferred ``graphed`` ``Array`` recorded on the
``graphed.awkward`` backend; construction reads only metadata (the ``TTree`` form). A recorded
analysis runs the way a task graph runs: per partition through a ``graphed-executors`` executor
and tree-reduced (``graphed.aggregate_plan``, ``graphed.awkward.to_parquet``,
``uproot.graphed_write``). The array's source is a ``graphed`` partitioned source; every read goes
through a form mapping exactly as in :doc:`uproot._dask.dask` — ``TrivialFormMapping`` unless the
caller gives one — so a chunk holds every field of the mapped form, with the ``TBranches`` the
recorded graph does not touch standing in as placeholders. The source DECLARES that read list in
``TBranch`` names (``graphed``'s buffer projection, walked back through the mapping);
:doc:`uproot._graphed.necessary_columns` reports it without running anything.

``graphed`` is imported lazily, so importing ``uproot`` does not require it.
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import hashlib
import inspect
import itertools

import uproot
import uproot._util
import uproot.interpretation.library
import uproot.reading
from uproot._dask import (
    _base_form_of,
    _leaf_to_tree,
    _mapping_of,
    _normalize_grouped_keys,
    _read_tree,
    _trees_and_keys,
)
from uproot._util import no_filter, unset
from uproot.behaviors.RNTuple import HasFields
from uproot.behaviors.TBranch import HasBranches, TBranch

#: An already-open TTree (``HasBranches``) or RNTuple (``HasFields``), given as ``files``.
_open_tree_types = (HasBranches, HasFields)

_DASK = "which belongs to uproot.dask; "
_ENTRY_RANGE = (
    "but it takes whole TTrees; give a range as 'steps' in the files of uproot.graphed, "
    "without align_baskets: "
    "{path: {'object_path': tree, 'steps': [[start, stop]]}}"
)
#: Keywords of other ``uproot`` readers that the ``graphed`` entry points have no counterpart for,
#: and where that concern lives instead; ``**options`` would otherwise swallow them silently.
_REFUSED = {
    "step_size": _DASK
    + "partition with uproot.graphed_partitions(step_size=...) and pass the "
    "result to graphed.aggregate_plan(partitions=...)",
    "steps_per_file": _DASK + "pass steps_per_file= where the plan is built: "
    "graphed.aggregate_plan, graphed.awkward.to_parquet or uproot.graphed_write",
    "allow_read_errors_with_report": _DASK
    + "a failed partition is dead-lettered and retried "
    "by graphed.checkpoint.run_resumable",
    "entry_start": _ENTRY_RANGE,
    "entry_stop": _ENTRY_RANGE,
}
#: ``uproot.open``'s own keywords, which ``_open_directory`` hands to ``ReadOnlyFile``.
_FILE_KEYWORDS = frozenset(
    name
    for name, parameter in inspect.signature(uproot.reading.open).parameters.items()
    if parameter.kind is parameter.KEYWORD_ONLY
)


class _GraphedTTreeSource:
    """A lazy ``graphed`` source over resolved ``(file_path, object_path)`` pairs, read through a
    form mapping. ``graphed.write.PartitionedSource``: ``partitions`` are blind (planning opens
    no file) unless the caller gave explicit ``steps``; ``read_partition`` opens each file once per
    worker; ``projected_columns`` declares the ``TBranches`` a set of outputs needs."""

    def __init__(
        self,
        file_tree,
        common_keys,
        *,
        name,
        expected_form,
        form_mapping_info,
        form_mapping=None,
        interp_options,
        custom_classes,
        allow_missing,
        options,
        decompression_executor,
        interpretation_executor,
        explicit_chunks=None,
        align_baskets=False,
    ):
        # [(file_path, object_path or None, _spellings of the tree there)]
        self._file_tree = list(file_tree)
        self._common_keys = list(common_keys)
        self._name = name  # the graphed source name, which the projection answers under
        self._expected_form = expected_form
        self._form_mapping_info = form_mapping_info
        self._form_mapping = form_mapping
        self._interp_options = interp_options
        self._custom_classes = custom_classes
        self._allow_missing = bool(allow_missing)
        self._options = options
        self._decompression_executor = decompression_executor
        self._interpretation_executor = interpretation_executor
        self._explicit_chunks = explicit_chunks  # per file: [(start, stop), ...]
        self._align_baskets = bool(align_baskets)
        self._offsets_memo = {}  # (uri, tree, keys) -> aligned offsets, per process

    def __getstate__(self):
        return {**self.__dict__, "_offsets_memo": {}}

    def source_identity(self):
        """What this source reads beyond its form, compared with ``==``: ``graphed``'s
        ``Session.source`` reuses a name's binding for an equal identity and refuses any other.
        """
        mapping = self._form_mapping
        own = getattr(mapping, "source_identity", None)
        return (
            tuple((file_path, paths) for file_path, _, paths in self._file_tree),
            self._explicit_chunks,
            self._allow_missing,
            self._align_baskets,
            self._custom_classes,
            own() if callable(own) else _Same(mapping),
            _Same(self._decompression_executor),
            _Same(self._interpretation_executor),
        )

    def _digest(self):
        """A short hash of the parts of ``source_identity`` a hash can keep, so reads it cannot
        tell apart (another cycle, ``custom_classes`` or executors, or a ``form_mapping`` with no
        ``source_identity``) get equal digests. ``_Spellings`` stay out because their ``==`` is
        overlap; a file opened to read it has one path, whichever spelling opened it.
        """
        files = [
            (file_path, uproot._util.canonical_object_path(path))
            for file_path, path, _ in self._file_tree
        ]
        read = (
            files,
            self._explicit_chunks,
            self._allow_missing,
            self._align_baskets,
            self._expected_form.to_json(),
        )
        return hashlib.sha256(repr(read).encode()).hexdigest()[:8]

    # ---- opening -------------------------------------------------------------------------
    def _open_directory(self, file_path):
        return _open_directory(file_path, self._custom_classes, self._options)

    def _opened_tree(self, file_path, object_path):
        """:func:`_opened_object` with the options this source was given."""
        return _opened_object(
            file_path,
            object_path,
            self._custom_classes,
            self._allow_missing,
            self._options,
        )

    def open_tree(self, partition, resources):
        """This partition's ``TTree``, from a file opened once per worker with the options
        ``uproot.graphed`` was given; ``None`` when ``allow_missing`` and the file lacks it.
        """
        return uproot._util.object_in_directory(
            resources.open_once(partition.uri, self._open_directory),
            partition.tree or None,
            self._allow_missing,
        )

    def num_entries(self, partition):
        """Entries in the whole ``TTree`` ``partition`` reads, so ``graphed`` can state a blind
        chunk's range when it reports a failure; opens the file, so call it only then.

        ``None`` with ``align_baskets``: a blind chunk's range is then not a fraction of the
        count, and which ``TBaskets`` bound it depends on the ``TBranches`` it read.
        """
        if self._align_baskets:
            return None
        with self._opened_tree(partition.uri, partition.tree) as tree:
            return 0 if tree is None else tree.num_entries

    def _trees(self):
        """Every tree, opened here and now (the whole-dataset loader and ``graphed_head``). A
        tree's file closes when the generator resumes or is closed, so read it before either.
        """
        for file_path, object_path, _ in self._file_tree:
            with self._opened_tree(file_path, object_path) as tree:
                if tree is not None:
                    yield tree

    # ---- reading -------------------------------------------------------------------------
    def read_range(self, tree, keys, start, stop):
        """Entries ``[start, stop)`` of ``keys`` through the mapping (unread buffers are
        placeholders), as :doc:`uproot._dask.dask` reads a partition."""
        return _read_tree(
            tree,
            keys,
            start,
            stop,
            expected_form=self._expected_form,
            form_mapping_info=self._form_mapping_info,
            interp_options=self._interp_options,
            decompression_executor=self._decompression_executor,
            interpretation_executor=self._interpretation_executor,
        )

    def _empty(self):
        import awkward

        return awkward.Array(
            self._expected_form.length_zero_array(highlevel=False),
            behavior=self._form_mapping_info.behavior,
        )

    def __call__(self):
        import awkward

        with contextlib.closing(self._trees()) as trees:
            parts = [
                self.read_range(tree, self._common_keys, 0, tree.num_entries)
                for tree in trees
            ]
        if not parts:
            return self._empty()
        return parts[0] if len(parts) == 1 else awkward.concatenate(parts)

    # ---- graphed.write.PartitionedSource ----------------------------------------------
    def partitions(self, steps_per_file=1):
        """One partition per (file x step) — BLIND, so planning opens no file — or, when the
        ``files`` carried explicit ``steps``, exactly those entry ranges."""
        graphed = uproot.extras.graphed()

        if self._explicit_chunks is not None:
            if steps_per_file != 1:
                raise TypeError(
                    "explicit 'steps' in 'files' are incompatible with steps_per_file"
                )
            return tuple(
                graphed.core.Partition(
                    file_path, object_path or "", int(start), int(stop)
                )
                for (file_path, object_path, _), chunks in zip(
                    self._file_tree, self._explicit_chunks, strict=True
                )
                for start, stop in chunks
            )
        return tuple(
            graphed.core.Partition.blind(
                file_path, object_path or "", step, steps_per_file
            )
            for file_path, object_path, _ in self._file_tree
            for step in range(steps_per_file)
        )

    def read_partition(self, partition, columns, resources):
        """One partition's chunk; ``columns`` is the declared read list (``None`` only from a
        hand-built plan: every selected branch)."""
        tree = self.open_tree(partition, resources)
        if tree is None:
            return self._empty()
        keys = list(columns) if columns is not None else list(self._common_keys)
        start, stop = self.entry_range(partition, tree, keys)
        if stop <= start:  # read_range would still decompress the boundary TBasket
            return self._empty()
        return self.read_range(tree, keys, start, stop)

    def entry_range(self, partition, tree, keys):
        """The ``(start, stop)`` entries ``partition`` reads from ``tree``. With
        ``align_baskets``, a blind step's boundaries move to the nearest ``TBasket`` boundary
        shared by the ``TBranches`` that reading ``keys`` decompresses; a key absent from
        ``tree`` reads nothing, so it constrains nothing."""
        n_entries = tree.num_entries
        start, stop = _partition_range(partition, n_entries)
        if not (self._align_baskets and partition.is_blind):
            return start, stop
        if isinstance(tree, HasFields):
            raise NotImplementedError(
                f"align_baskets=True follows TBasket boundaries; the RNTuple in "
                f"{partition.uri!r} has none (open_files=True refuses it in uproot.graphed)"
            )
        memo_key = (partition.uri, partition.tree, tuple(keys))
        if memo_key not in self._offsets_memo:
            self._offsets_memo[memo_key] = _basket_offsets(
                [tree[key].common_entry_offsets() for key in keys if key in tree],
                n_entries,
            )
        offsets = self._offsets_memo[memo_key]
        if offsets is None:
            return start, stop
        return _snap(start, offsets), _snap(stop, offsets)

    def projected_columns(self, outputs, *, on_fail="pass"):
        """The ``TBranches`` this source declares for ``outputs`` (asked driver-side through
        ``graphed.write.declared_columns``).

        ``graphed``'s buffer projection answers in the mapped form's dotted paths, so each is
        walked back to the buffer keys it needs — a list's structure is a buffer of its own — and
        those go to the mapping's ``keys_for_buffer_keys``. ``on_fail="pass"`` keeps an opaque node
        (a user callable ``graphed`` cannot see through) from raising: the node is treated as
        needing everything its inputs carry, so the read list stays a superset of what is needed.
        """
        graphed = uproot.extras.graphed()

        # one replay of the mapped form for all outputs, not one per output
        return self._columns_of(
            graphed.awkward.projection.project_buffers_many(
                list(outputs), on_fail=on_fail
            ).read_buffers
        )

    def _columns_of(self, read_buffers):
        """This source's ``TBranches`` in a projection's ``read_buffers``."""
        graphed = uproot.extras.graphed()

        info = self._form_mapping_info
        buffer_keys = set()
        for path, need in read_buffers.get(self._name, {}).items():
            buffer_keys.update(
                _mapped_buffer_keys(
                    self._expected_form,
                    path,
                    need is graphed.projection.BufferNeed.DATA,
                    info.buffer_key,
                )
            )
        return tuple(sorted(info.keys_for_buffer_keys(frozenset(buffer_keys))))


class _Same:
    """Equal only to a wrapper of the same object, for parts whose ``==`` is not "reads alike"."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, _Same) and other.obj is self.obj

    __hash__ = None


def _mapped_buffer_keys(form, path, data, buffer_key):
    """The mapped form's buffer keys for one projected ``path``: every structure crossed on the
    way down to it, plus the endpoint's own — recursively when its ``data`` is read, and the list
    offsets alone when only its shape is."""
    keys = set()
    fields = path.split(".") if path != "<root>" else []
    while fields:
        if form.is_record:
            form = form.content(fields.pop(0))
        else:  # a list/option wrapper between records: its structure shapes everything below
            keys.update(
                form.expected_from_buffers(buffer_key=buffer_key, recursive=False)
            )
            form = form.content
    keys.update(form.expected_from_buffers(buffer_key=buffer_key, recursive=data))
    return keys


def _name_from_object_path(file_tree):
    """The ``TTree``'s name when no file was opened to ask it (``known_base_form=``): the object
    path the caller gave, without its ``TDirectory`` prefix or ``;cycle`` suffix — the same name
    the opened ``TTree`` reports, so a source keeps its identity either way."""
    return (file_tree[0][1] or "").rpartition("/")[2].partition(";")[0] or "events"


class _Spellings:
    """Object paths that open one tree in a file; two reads of the file are one source when
    their paths overlap (see :func:`_spellings`)."""

    __slots__ = ("paths",)

    def __init__(self, paths):
        self.paths = frozenset(paths)

    def __eq__(self, other):
        return isinstance(other, _Spellings) and not self.paths.isdisjoint(other.paths)

    __hash__ = None


def _spellings(given, given_path, found):
    """The object paths known, without I/O, to open the tree an entry of ``files`` reads: the
    path given, and for a tree in hand (``found``) its key's ``dir/name;cycle`` plus ``dir/name``
    when that opens the same key, so a spelling only an opened file could equate stays apart.
    """
    paths = set()
    if not isinstance(given, _open_tree_types):
        paths.add((given_path or "").lstrip("/") or None)
    if found is not None:
        tree, inner = _leaf_to_tree(found)[0], []
        while isinstance(tree, TBranch):
            inner.append(tree.name)
            tree = tree.parent
        key = tree.parent
        name = key.name(False)
        path = "/".join((*key.parent.path, name))
        branch = ":" + "/".join(reversed(inner)) if inner else ""
        paths.add(f"{path};{key.fCycle}{branch}")
        try:
            if key.parent.key(name) is key:  # the directory's keys are held: no I/O
                paths.add(path + branch)
        except uproot.KeyInFileError:  # a name like "a;3" reads as a cycle of "a"
            pass
    return _Spellings(paths)


def _file_and_path(file_path, object_path):
    """An already-open ``TTree``/``RNTuple`` given as a file is re-opened by path in the workers."""
    if isinstance(file_path, _open_tree_types):
        return file_path.file.file_path, file_path.object_path
    return file_path, object_path


def _refuse_options(function, options):
    """Refuse any ``_REFUSED`` keyword in the ``options`` of the public ``function``."""
    for option, instead in _REFUSED.items():
        if option in options:
            raise TypeError(f"uproot.{function}() got {option}=, {instead}")


def _check_options(options, files=None):
    """Refuse or warn about ``options`` for a public entry point: before any I/O when ``files``
    is ``None``, and once regularized ``files`` say whether every file is local."""
    if files is None:
        all_local = None
    else:
        paths = (_file_and_path(file_path, None)[0] for file_path, *_ in files)
        all_local = all(
            not isinstance(path, str)  # a given file object has no URL
            or uproot._util.is_local_path(path)
            for path in paths
        )
    uproot._util._check_unknown_options(
        {k: v for k, v in options.items() if k not in _FILE_KEYWORDS},
        all_local=all_local,
        stacklevel=4,
    )


def _open_directory(file_path, custom_classes, options):
    return uproot.reading.ReadOnlyFile(
        file_path,
        # the object cache keeps the TTree across a worker's partitions; arrays are not held
        **{"array_cache": None, **options},
        custom_classes=custom_classes,
    ).root_directory


@contextlib.contextmanager
def _opened_object(file_path, object_path, custom_classes, allow_missing, options):
    """The object at ``object_path`` (None when ``allow_missing`` and absent); a file opened here
    closes with this context whatever it found, and a given ``TTree``/``RNTuple``'s file stays open.
    """
    if isinstance(file_path, _open_tree_types):
        yield file_path
        return
    with contextlib.closing(
        _open_directory(file_path, custom_classes, options)
    ) as directory:
        yield uproot._util.object_in_directory(
            directory, object_path or None, allow_missing
        )


def _selection(leaf):
    return "the whole TTree" if leaf is None else f"the TBranch {leaf!r}"


def _resolve_entries(
    files, open_files, custom_classes, allow_missing, options, directories
):
    """Every entry of ``files`` as ``(file_path, tree_path, steps, paths, found)``, where
    ``paths`` are the entry's :func:`_spellings` and ``found`` is the
    opened object (a leaf ``TBranch`` included) or ``None`` for a file left unopened; the files
    opened go in ``directories``, ``{canonical_path(file_path): directory}``, for the caller to close even when
    this raises. An entry whose file lacks the object is dropped under ``allow_missing``.

    A file is opened when it holds the first entry that has the object, with ``open_files``, when
    it is listed more than once (so the same ``TTree`` spelled two ways resolves to one
    ``tree_path``), when its object path has a ``:``, which names a ``TBranch`` or is a literal
    name depending on the file's keys, or when its object path is not the first entry's and has a
    ``/``, which could end in a ``TTree`` or in a ``TBranch``; any other path names a ``TTree``,
    or the first entry's leaf ``TBranch`` when it is the first entry's path. Every entry must select the first entry's
    leaf ``TBranch``, or none when the first names a ``TTree``."""
    seen_inodes = {}
    entries = []
    for given, given_path, *steps in files:
        file_path, _ = _file_and_path(given, given_path)
        canon = uproot._util.canonical_path(file_path, seen_inodes=seen_inodes)
        entries.append((given, given_path, steps, canon))

    listed = collections.Counter(canon for *_, canon in entries)
    first_path = leaf = unset
    resolved = []
    for given, given_path, steps, canon in entries:
        file_path, object_path = _file_and_path(given, given_path)
        if isinstance(given, _open_tree_types):
            found = given
        elif (
            first_path is unset
            or open_files
            or listed[canon] > 1
            or ":" in (object_path or "")
            or (object_path != first_path and "/" in (object_path or "").strip("/"))
        ):
            if canon not in directories:
                directories[canon] = _open_directory(file_path, custom_classes, options)
            found = uproot._util.object_in_directory(
                directories[canon], object_path or None, allow_missing
            )
            if found is None:
                continue
        else:
            found = None
        if found is not None:
            tree, branch = _leaf_to_tree(found)
            branch = None if branch is None else branch.name
            file_path, tree_path = tree.file.file_path, tree.object_path
        elif object_path == first_path and leaf is not None:
            tree_path, branch = object_path.rpartition("/")[0], leaf
        else:
            tree_path, branch = object_path, None
        if first_path is unset:
            first_path, leaf = object_path, branch
        elif branch != leaf:
            raise ValueError(
                f"{{{file_path!r}: {given_path!r}}} selects {_selection(branch)}, but the first "
                f"file selects {_selection(leaf)}; select the same in every file"
            )
        paths = _spellings(given, given_path, found)
        resolved.append((file_path, tree_path, steps, paths, found))
    return resolved


def _read_sources(array):
    """The ``uproot.graphed`` sources ``array`` reads, as ``{node_id: source}``; its ``Session``
    may hold others."""
    session = array.session
    read = uproot.extras.graphed().by_label.cone(session, array.node_id)
    return {
        nid: s
        for nid, s in session.sources().items()
        if nid in read and isinstance(s, _GraphedTTreeSource)
    }


def _uproot_source(array, what):
    """The one ``uproot.graphed`` source ``array`` reads, as ``(node_id, source)``."""
    sources = list(_read_sources(array).items())
    if len(sources) != 1:
        raise TypeError(
            f"{what} supports exactly one uproot.graphed source per array; "
            f"this array is backed by {len(sources)}"
        )
    return sources[0]


def _refuse_backend(have, wanted, *, behavior, where="session="):
    """Refuse the Session ``where`` names, on ``have``, for a read recorded on ``wanted``: ``have``
    must be a ``type(wanted)`` and, when ``behavior`` names the read's behavior, carry an equal one
    (``None`` counts as empty)."""
    if not isinstance(have, type(wanted)):
        have_text = type(have).__name__
    elif behavior and (have._behavior or {}) != (wanted._behavior or {}):
        have_text = f"{type(have).__name__} with another behavior"
    else:
        return
    wanted_text = type(wanted).__name__ + (f" with {behavior}" if behavior else "")
    fix = "pass a Session on that backend"
    if where != "session=":
        fix += (
            " with session=, call graphed.new_session(<that backend>), or read inside "
            "`with graphed.Session(<that backend>):`"
        )
    raise TypeError(
        f"uproot.graphed: {where} records on {have_text}, and this read on {wanted_text}; {fix}"
    )


def _taken(session, name):
    return any(session.source_name(nid) == name for nid in session.source_ids())


def _refuse_taken(session, name):
    """Refuse recording under a source name ``session`` already has."""
    if _taken(session, name):
        raise ValueError(
            f"uproot.graphed: this Session already has a source named {name!r}; "
            "pass name= to name this read"
        )


def _default_names(session, tree_name, source):
    """The names a read with no ``name=`` tries, the digest last so a first read pays nothing.
    Every taken name has a source, so ``k`` sources leave one of ``k + 1`` names free; ``k`` is
    read live because another thread may add sources between tries.
    """
    yield tree_name
    hashed = f"{tree_name}-{source._digest()}"
    yield hashed
    n = 1
    while n < len(session.source_ids()):
        yield f"{hashed}#{n}"
        n += 1


def _record_under_default_name(session, tree_name, form, source):
    """Record ``source`` under the first of its default names that ``session.source`` accepts,
    which is its own binding on a re-read, so only a first read's name depends on read order.
    """
    for candidate in _default_names(session, tree_name, source):
        source._name = candidate
        try:
            return session.source(candidate, form=form, data=source)
        except ValueError:
            if not _taken(session, candidate):
                raise
    raise AssertionError(f"no free source name for {tree_name!r}")


def graphed(
    files,
    *,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    library="ak",
    ak_add_doc=False,
    custom_classes=None,
    allow_missing=False,
    behavior=None,
    form_mapping=None,
    known_base_form=None,
    backend=None,
    session=None,
    name=None,
    decompression_executor=None,
    interpretation_executor=None,
    align_baskets=False,
    open_files=False,
    **options,
):
    """
    Args:
        files: The ``TTree``(s) to read, in any form accepted by :doc:`uproot._dask.dask` (a
            path with ``"file.root:tree"``, a list, a dict, an open ``TTree``, a dict entry with
            explicit ``"steps"``; a single-``TTree`` file needs no object path).
        filter_name, filter_typename, filter_branch, recursive, full_paths: ``TBranch`` selection,
            as in :doc:`uproot._dask.dask`.
        library (str): Only ``"ak"`` is supported (a single deferred ``graphed`` array). ``"np"`` and
            ``"pd"`` raise ``NotImplementedError``.
        ak_add_doc (bool | dict): As in :doc:`uproot._dask.dask`, for a TTree or an RNTuple: a
            ``value`` of ``"title"`` (the TBranch attribute) reads the RField ``description``.
        custom_classes, allow_missing: As in :doc:`uproot._dask.dask`.
        behavior (dict or None): An awkward behavior dict (e.g. ``vector``'s) registered on the
            recording backend, for the default backend and the unmapped form: with
            ``gak.with_name``, behavior PROPERTIES (``.pt``, ``.mass``) work through plain attribute
            access — typetracer forms at record time, projectable down to exactly the branches a
            property reads.
        form_mapping (ImplementsFormMapping or None): The mapping :doc:`uproot._dask.dask` takes —
            it restructures the flat ``TTree`` form into the one the user sees (coffea's
            ``NanoEventsFactory`` is the archetype) and fills that form's buffers from the file.
            The recorded array then has the MAPPED form and the mapping info's ``behavior`` is
            registered on the backend. Without one, ``TrivialFormMapping`` reads the file's form.
        known_base_form (awkward.forms.Form | None): If not none use this form instead of opening
            one file to determine the dataset's form, as in :doc:`uproot._dask.dask`. No file is
            opened at all; the ``TTree``'s name then comes from the object path in ``files``.
        backend (graphed Backend or None): The ``graphed`` backend **instance** the session
            records on, in place of the default ``graphed.awkward.AwkwardBackend`` — the route for
            a caller's own ``Array`` subclass. Its behavior (and, with ``form_mapping``, the
            mapping info's) is then the caller's business, so ``behavior=`` is refused beside it.
        session (graphed Session or None): Record into this existing ``Session`` (and on its
            backend), so the result can be combined with arrays already recorded there, e.g.
            ``graphed.join(events, uproot.graphed(friends, session=events.session,
            name="friends"), on=[...])``. Exclusive with ``backend=`` and ``behavior=``; its
            backend must be an ``AwkwardBackend`` and, with ``form_mapping=``, carry the mapping's
            behavior. Without it, the read records into ``graphed.current_session()``: the
            innermost open ``with session:`` scope, else the default Session, which this read
            creates on its own backend when there is none; so ``with events.session:
            uproot.graphed(friends, name="friends")`` joins too. That Session must be an instance
            of the read's backend class and, when ``behavior=``, ``backend=`` or ``form_mapping=``
            names a behavior, carry an equal one; any other is refused at the read. With a
            ``graphed`` that has no ``current_session``, each such read gets a new ``Session``.
        name (str or None): The ``graphed`` source's name, its slot in the recorded graph; default
            the ``TTree``'s name, or when another read already holds that, the ``TTree``'s name
            with a hash of what this read reads, numbered (``#1``, ...) when the hash cannot tell
            it from another read. Repeating a read, ``name=`` included, returns that source again;
            any other read under a name given here is refused.
            With a ``graphed`` that has no ``current_session``, every read under a
            taken name is refused, before any file is opened when the name is known from
            ``name=`` or the ``files``' object paths.
        decompression_executor, interpretation_executor: As in :doc:`uproot._dask.dask`; every
            partition read uses them.
        align_baskets (bool): If True, each blind partition's entry range moves, when it is
            read, to the nearest ``TBasket`` boundaries shared by the ``TBranches`` it reads, so
            no ``TBasket`` is decompressed by two partitions. The partition count is unchanged;
            a partition can come out empty. Incompatible with explicit ``"steps"``; ``TTrees``
            only.
        open_files (bool): If False (default), only the first file that has the ``TTree`` is
            opened, for the form and the ``TBranch`` names; the files after it are listed by path
            and opened only when read, so a missing ``TBranch`` in one of them fails the read and,
            with ``allow_missing``, a missing ``TTree`` reads as empty. A file listed more than
            once is opened to resolve each entry's ``TTree``, so the same ``TTree`` listed twice
            in two spellings is one input (``uproot.graphed_write`` refuses the duplicate), and
            with ``allow_missing`` its missing ``TTree`` is dropped rather than read as empty.
            A file whose object path has a ``/`` and differs from the first file's is opened too:
            the path could end in a ``TTree`` or in a ``TBranch``. An unopened
            file is validated only when it is read: without ``allow_missing``, a missing ``TTree``
            (or, with ``align_baskets``, an RNTuple) fails that partition's read, after other
            partitions may have written their parts.
            If True, every file is opened up front, as :doc:`uproot._dask.dask` does by default:
            the ``TBranches`` are those every file has, a file without the ``TTree`` is
            dropped (or refused without ``allow_missing``), and an RNTuple is refused with
            ``align_baskets``.
        options: Passed through to file opening. A key :doc:`uproot.reading.open` does not take
            is an fsspec storage option, refused when every file is local.

    Returns a deferred ``graphed`` ``Array`` for the selected ``TTree``(s). Construction reads only
    metadata; running the recorded analysis (``graphed.aggregate_plan``,
    ``graphed.awkward.to_parquet``, :doc:`uproot.writing._graphed_write.graphed_write`) triggers
    the read, partition by partition, fetching only the ``TBranches`` the recorded graph touches.

    Partitioning is decided where a plan is built (``steps_per_file=`` there), not here, so
    ``step_size`` and ``steps_per_file`` are refused by name, as are ``entry_start`` and
    ``entry_stop``, whose route is ``'steps'`` in ``files``; ``align_baskets=``
    decides that the boundaries follow ``TBaskets``. For one task per ``TBasket``-sparse file,
    pass ``partitions=uproot.graphed_partitions(..., align_baskets=True)`` where the
    plan is built.

    **Experimental**: tracks a pre-1.0 ``graphed``; the API may change without a deprecation cycle.

    This is the ``graphed`` analogue of :doc:`uproot._dask.dask`.
    """
    _refuse_options("graphed", options)
    _check_options(options)
    library = uproot.interpretation.library._regularize_library(library)
    if library.name != "ak":
        raise NotImplementedError(
            f"uproot.graphed currently supports only library='ak', not {library.name!r}"
        )
    if behavior is not None and (
        backend is not None or session is not None or form_mapping is not None
    ):
        raise TypeError(
            "uproot.graphed: behavior= is for the default backend and the unmapped form; with "
            "backend= or session= the behavior belongs to that backend, with form_mapping= to its "
            "mapping info"
        )
    if backend is not None and session is not None:
        raise TypeError(
            "uproot.graphed: session= records on that Session's backend; pass backend= or "
            "session=, not both"
        )

    import awkward

    graphed = uproot.extras.graphed()
    # feature detection: graphed 0.0.9 has no current Session and no one-binding-per-name rule
    current_session = getattr(graphed, "current_session", None)

    if session is not None:
        if not isinstance(session, graphed.Session):
            raise TypeError(
                f"uproot.graphed: session= takes a graphed.Session, not {type(session).__name__}"
            )
        _refuse_backend(
            session.backend, graphed.awkward.AwkwardBackend(), behavior=None
        )

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)
    real_options.setdefault("num_fallback_workers", 1)
    filter_branch = uproot._util.regularize_filter(filter_branch)
    interp_options = {"ak_add_doc": ak_add_doc}

    files = uproot._util.regularize_files(files, steps_allowed=True, **options)
    _check_options(options, files)
    is_3arg = [len(x) == 3 for x in files]
    if any(is_3arg) and not all(is_3arg):
        raise TypeError(
            "partition sizes for some but not all 'files' have been assigned"
        )
    if align_baskets and any(is_3arg):
        raise TypeError(
            "uproot.graphed: align_baskets=True moves blind partitions; it cannot be used "
            "with explicit 'steps' in 'files'"
        )
    # before any I/O when the name is already known
    if session is not None and current_session is None:
        known = name
        if known is None:
            # a path with "/" or ":" may end in a TBranch, and its read is named after the TTree
            names = {
                o
                and ":" not in o
                and "/" not in o.strip("/")
                and _name_from_object_path([(f, o)])
                for f, o in (_file_and_path(f, o) for f, o, *_ in files)
            }
            known = names.pop() if len(names) == 1 else None
        if known:
            _refuse_taken(session, known)

    if known_base_form is not None and open_files:
        raise TypeError(
            "uproot.graphed: known_base_form= opens no file; drop open_files=True"
        )
    if known_base_form is not None:  # the form is given: nothing is opened
        common_keys = list(dict.fromkeys(known_base_form.fields))
        base_form = known_base_form
        file_tree = [
            (
                *_file_and_path(f, o),
                _spellings(f, o, f if isinstance(f, _open_tree_types) else None),
            )
            for f, o, *_ in files
        ]
        explicit_chunks = [f[2] for f in files] if all(is_3arg) else None
        tree_name = _name_from_object_path(file_tree)
        is_rntuple = False  # unknown without opening a file: read whole fields
    else:
        directories = {}
        try:
            resolved = _resolve_entries(
                files,
                open_files,
                custom_classes,
                allow_missing,
                real_options,
                directories,
            )
            found = [(found,) for *_, found in resolved if found is not None]
            ttrees, common_keys, _, _ = _trees_and_keys(
                found if open_files else found[:1],
                files,
                filter_name,
                filter_typename,
                filter_branch,
                recursive,
                full_paths,
                first_of=None if open_files else len(files),
            )
            if align_baskets and any(
                isinstance(t, HasFields) for t in [*ttrees, *(f for (f,) in found)]
            ):
                raise NotImplementedError(
                    "align_baskets=True follows TBasket boundaries; an RNTuple has none"
                )
            common_keys = _normalize_grouped_keys(ttrees[0], common_keys, full_paths)
            base_form = _base_form_of(ttrees[0], common_keys, ak_add_doc, form_mapping)
            is_rntuple = all(isinstance(t, HasFields) for t in ttrees)
        finally:
            for directory in directories.values():
                directory.file.close()
        file_tree = [
            (file_path, tree_path, paths)
            for file_path, tree_path, _, paths, _ in resolved
        ]
        explicit_chunks = (
            [steps[0] for _, _, steps, *_ in resolved] if all(is_3arg) else None
        )
        tree_name = getattr(ttrees[0], "name", None) or _name_from_object_path(
            file_tree
        )
    expected_form, info = _mapping_of(form_mapping, base_form, is_rntuple)
    typetracer = awkward.typetracer.typetracer_from_form(
        expected_form, highlevel=True, behavior=info.behavior
    )
    mapped = "its form_mapping behavior" if form_mapping is not None else None
    if session is None:
        if backend is not None:
            requested, checked = backend, "the backend='s behavior"
        else:
            requested = graphed.awkward.AwkwardBackend(
                behavior=behavior if form_mapping is None else info.behavior
            )
            checked = mapped or (
                "the behavior= given" if behavior is not None else None
            )
        if current_session is None:
            session = graphed.Session(requested)
        else:
            session = current_session(requested)
            _refuse_backend(
                session.backend,
                requested,
                behavior=checked,
                where="the current Session",
            )
    elif mapped:
        _refuse_backend(
            session.backend,
            graphed.awkward.AwkwardBackend(behavior=info.behavior),
            behavior=mapped,
        )
    if current_session is None:
        _refuse_taken(session, tree_name if name is None else name)
    source = _GraphedTTreeSource(
        file_tree,
        list(base_form.fields),
        name=name,
        expected_form=expected_form,
        form_mapping_info=info,
        form_mapping=form_mapping,
        interp_options=interp_options,
        custom_classes=custom_classes,
        allow_missing=allow_missing,
        options=real_options,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
        explicit_chunks=explicit_chunks,
        align_baskets=align_baskets,
    )
    form = graphed.awkward.AwkwardForm(typetracer)
    if name is None:
        return _record_under_default_name(session, tree_name, form, source)
    return session.source(name, form=form, data=source)


def necessary_columns(array, *, on_fail="raise"):
    """The ``TBranches`` each ``uproot.graphed`` source that ``array`` reads needs for it —
    ``graphed``'s buffer projection walked back through the form mapping (metadata-only). Returns
    ``{source_name: frozenset(branch_names)}``."""
    graphed = uproot.extras.graphed()

    read_buffers = graphed.awkward.projection.project_buffers_many(
        [array], on_fail=on_fail
    ).read_buffers
    return {
        array.session.source_name(nid): frozenset(source._columns_of(read_buffers))
        for nid, source in _read_sources(array).items()
    }


def graphed_partitions(
    files,
    *,
    step_size=unset,
    steps_per_file=unset,
    open_files=True,
    align_baskets=False,
    custom_classes=None,
    allow_missing=False,
    **options,
):
    """Partition a uproot dataset into ``graphed.core.Partition`` chunks, for
    ``graphed.aggregate_plan(..., partitions=...)`` or a hand-built ``graphed.core.Plan``.

    Mirrors :doc:`uproot._dask.dask`'s chunking knobs:

    - ``steps_per_file`` (default 1): split each ``TTree`` into that many contiguous chunks.
    - ``step_size`` (int entries, or a memory string like ``"100 MB"``): cap each chunk's size.
      Mutually exclusive with ``steps_per_file``, and **incompatible with** ``open_files=False``.
    - ``open_files`` (default ``True``): open every file to read its entry count and emit exact
      entry ranges. ``open_files=False`` emits **blind** partitions (``Partition.blind``): files
      are not opened here and the entry range is resolved against the file's own count when the
      partition is read (:doc:`uproot._graphed.read_graphed_partition`).
    - ``align_baskets`` (default ``False``): ``True``, or a ``filter_name``-style selection
      (glob, regex, callable, or an iterable of them) of the ``TBranches`` whose ``TBaskets``
      the boundaries follow: pass the branches the analysis reads. Every chunk boundary moves
      to the nearest entry offset where all of those ``TBranches`` start a new ``TBasket`` (a
      tie goes to the lower one), and the chunks this leaves empty are dropped, so no
      ``TBasket`` is decompressed by two chunks. A chunk can then be larger than
      ``step_size``. With ``True`` on a ``TTree`` whose branches' ``TBaskets`` do not line up,
      the shared offsets can shrink to the file's ends and the file yields one chunk, where a
      read with ``uproot.graphed(align_baskets=True)`` follows only the branches each
      partition reads. Needs ``open_files=True``; ``TTrees`` only.

    ``entry_start``, ``entry_stop`` and ``allow_read_errors_with_report`` are refused by name;
    other ``options`` are passed through to file opening.
    """
    _refuse_options("graphed_partitions", options)
    _check_options(options)
    graphed = uproot.extras.graphed()

    have_step_size = not isinstance(step_size, uproot._util._Unset)
    have_steps_per_file = not isinstance(steps_per_file, uproot._util._Unset)
    if align_baskets and not open_files:
        raise TypeError(
            "align_baskets=True needs the entry offsets of opened files; "
            "it cannot be used with open_files=False"
        )
    if have_step_size and not open_files:
        raise TypeError(
            "step_size cannot be used with open_files=False; use steps_per_file"
        )
    if have_step_size and have_steps_per_file:
        raise TypeError(
            "step_size and steps_per_file are mutually exclusive; set only one"
        )
    if have_step_size:
        uproot._util.check_step_size(step_size)
    n_steps = int(steps_per_file) if have_steps_per_file else 1

    resolved = uproot._util.regularize_files(files, steps_allowed=False, **options)
    _check_options(options, resolved)

    real_options = options.copy()
    real_options.setdefault("num_workers", 1)

    partitions = []
    for file_path, object_path, *_ in resolved:
        if not open_files:
            partitions.extend(
                graphed.core.Partition.blind(
                    file_path, object_path or "", step, n_steps
                )
                for step in range(n_steps)
            )
            continue
        with _opened_object(
            file_path, object_path, custom_classes, allow_missing, real_options
        ) as obj:
            if obj is None:
                continue
            if align_baskets and isinstance(obj, HasFields):
                raise NotImplementedError(
                    "align_baskets=True follows TBasket boundaries; an RNTuple has none"
                )
            n_entries = obj.num_entries
            if have_step_size:
                per = (
                    step_size
                    if uproot._util.isint(step_size)
                    else obj.num_entries_for(step_size)
                )
                per = max(1, int(per))
                bounds = [*range(0, n_entries, per), n_entries]
            else:
                bounds = [(i * n_entries) // n_steps for i in range(n_steps + 1)]
            if align_baskets:
                shared = obj.common_entry_offsets(
                    filter_name=no_filter if align_baskets is True else align_baskets,
                    recursive=True,
                )
                # a non-empty tree's fold holds n_entries
                if n_entries and shared == [0]:
                    raise ValueError(
                        f"align_baskets={align_baskets!r} selects no TBranch with TBaskets in "
                        f"{obj.file.file_path}:{obj.object_path}"
                    )
                offsets = _basket_offsets([shared], n_entries)
                bounds = [_snap(bound, offsets) for bound in bounds]
            partitions.extend(
                graphed.core.Partition(
                    obj.file.file_path, obj.object_path, int(start), int(stop)
                )
                for start, stop in itertools.pairwise(bounds)
                if stop > start
            )
    return partitions


def _basket_offsets(offset_lists, n_entries):
    """The sorted intersection of ``offset_lists`` (``common_entry_offsets`` results) with
    ``0`` and ``n_entries`` added; ``None`` when there is nothing to intersect."""
    if not offset_lists:
        return None
    return sorted(set.intersection(*map(set, offset_lists)) | {0, n_entries})


def _snap(boundary, offsets):
    """The offset nearest ``boundary``, the lower on a tie; ``offsets`` is sorted and spans it."""
    i = bisect.bisect_left(offsets, boundary)
    if offsets[i] == boundary:
        return boundary
    lower, upper = offsets[i - 1], offsets[i]
    return upper if upper - boundary < boundary - lower else lower


def read_graphed_partition(
    partition, columns, *, tree=None, library="ak", **open_options
):
    """Read a ``graphed.core.Partition``'s chunk of ``columns`` from its ROOT file, resolving a
    **blind** partition against the file's actual entry count. A partition with no object path
    (``open_files=False`` over a bare file name) reads the file's single ``TTree`` or ``RNTuple``.
    Pass an already-open ``tree``/``RNTuple`` to reuse a per-worker ``open_once`` handle.
    """
    if tree is None:
        with uproot.open(partition.uri, **open_options) as directory:
            tree = uproot._util.object_in_directory(
                directory, partition.tree or None, False
            )
            return read_graphed_partition(
                partition, columns, tree=tree, library=library
            )
    start, stop = _partition_range(partition, tree.num_entries)
    return tree.arrays(
        list(columns), entry_start=start, entry_stop=stop, library=library
    )


def _partition_range(partition, n_entries):
    """A partition's concrete ``(entry_start, entry_stop)`` against a tree's entry count."""
    partition = partition.resolve(n_entries)
    return partition.entry_start, partition.entry_stop


def graphed_head(array, n=5):
    """EAGER peek at the first ``n`` rows of a recorded analysis's OUTPUT, reading only the first
    file's leading entries (and only the branches the graph needs) and evaluating the compiled IR
    over them — never the whole dataset. The prefix read grows until ``n`` output rows exist or the
    first file is exhausted, so a selection or an axis-0 slice peeks at its own first rows.

    The ``graphed`` analogue of a dask collection's ``head``.
    """
    graphed = uproot.extras.graphed()

    session = array.session
    nid, source = _uproot_source(array, "graphed_head")
    columns = list(source.projected_columns((array,)))
    compiled = graphed.compile_ir(session, array)
    externals = graphed.aggregate.external_evaluators(session, compiled)
    source_name = session.source_name(nid)

    with contextlib.closing(source._trees()) as trees:
        tree = next(trees, None)
        rows = 0 if tree is None else tree.num_entries
        stop = min(int(n), rows)
        while True:
            chunk = (
                source._empty()
                if tree is None
                else source.read_range(tree, columns, 0, stop)
            )
            (out,) = graphed.evaluate_ir(
                compiled, session.backend, {source_name: chunk}, externals=externals
            )
            try:
                n_out = len(out)
            except TypeError:  # a scalar peek (a reduction over the prefix)
                return out
            if n_out >= n or stop >= rows:
                return out[:n]
            stop = min(rows, max(2 * stop, int(n)))
