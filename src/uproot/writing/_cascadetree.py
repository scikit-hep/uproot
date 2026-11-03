# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module for writing TTrees in the "cascading" file writer. TTrees
are more like TDirectories than they are like histograms in that they can create
objects, TBaskets, which have to be allocated through the FreeSegments.

The implementation in this module does not use the TTree infrastructure in
:doc:`uproot.models.TTree`, :doc:`uproot.models.TBranch`, and :doc:`uproot.models.TBasket`,
since the models intended for reading have to adapt to different class versions, but
a writer can always write the same class version, and because writing involves allocating
and sometimes freeing data.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""

from __future__ import annotations

import datetime
import math
import struct
import warnings
from collections.abc import Mapping

import awkward
import numpy

import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization

_dtype_to_char = {
    numpy.dtype("bool"): "O",
    numpy.dtype(">i1"): "B",
    numpy.dtype(">u1"): "b",
    numpy.dtype(">i2"): "S",
    numpy.dtype(">u2"): "s",
    numpy.dtype(">i4"): "I",
    numpy.dtype(">u4"): "i",
    numpy.dtype(">i8"): "L",
    numpy.dtype(">u8"): "l",
    numpy.dtype(">f4"): "F",
    numpy.dtype(">f8"): "D",
    numpy.dtype(">U"): "C",
}


class Tree:
    """
    Writes a TTree, including all TBranches, TLeaves, and (upon ``extend``) TBaskets.

    Rather than treating TBranches as a separate object, this *writable* TTree writes
    the whole metadata block in one function, so that interrelationships are easier
    to preserve.

    Writes the following class instance versions:

    - TTree: version 20
    - TBranch: version 13
    - TLeaf: version 2
    - TLeaf*: version 1
    - TBasket: version 3

    The ``write_anew`` method writes the whole tree, possibly for the first time, possibly
    because it has been moved (exceeded its initial allocation of TBasket pointers).

    The ``write_updates`` method rewrites the parts that change when new TBaskets are
    added.

    The ``extend`` method adds a TBasket to every TBranch.

    The ``write_np_basket`` and ``write_jagged_basket`` methods write one TBasket in one
    TBranch, either a rectilinear one from NumPy or a simple jagged array from Awkward Array.

    See `ROOT TTree specification <https://github.com/root-project/root/blob/master/io/doc/TFile/ttree.md>`__.
    """

    def __init__(
        self,
        directory,
        name,
        title,
        branch_types,
        freesegments,
        counter_name,
        field_name,
        initial_basket_capacity,
        resize_factor,
    ):
        self._directory = directory
        self._name = name
        self._title = title
        self._freesegments = freesegments
        self._counter_name = counter_name
        self._field_name = field_name
        self._basket_capacity = initial_basket_capacity
        self._resize_factor = resize_factor
        self._has_unsupported_branches = False
        self._has_divergent_baskets = False
        self._rewrite_blockers = ()
        self._needs_relocation_before_extend = False

        if isinstance(branch_types, Mapping):
            branch_types_items = branch_types.items()
        else:
            branch_types_items = branch_types

        if len(branch_types) == 0:
            raise ValueError("TTree must have at least one branch")

        self._branch_data = []
        self._branch_lookup = {}
        record_fields = set()
        for branch_name, branch_type in branch_types_items:
            branch_dict = None
            branch_dtype = None
            branch_datashape = None

            if isinstance(branch_type, Mapping) and all(
                isinstance(x, str) for x in branch_type
            ):
                branch_dict = branch_type

            else:
                try:
                    if uproot._util.from_module(branch_type, "awkward"):
                        raise TypeError
                    if isinstance(branch_type, str) and branch_type.strip() == "bytes":
                        raise TypeError
                    branch_dtype = numpy.dtype(branch_type)

                except (TypeError, ValueError, SyntaxError) as err:
                    if isinstance(
                        branch_type,
                        (awkward.types.Type, awkward.types.ArrayType),
                    ):
                        branch_datashape = branch_type
                    else:
                        try:
                            branch_datashape = awkward.types.from_datashape(
                                branch_type, highlevel=False
                            )
                        except Exception:
                            raise TypeError(
                                f"not a NumPy dtype or an Awkward datashape: {branch_type!r}"
                            ) from err
                    if isinstance(branch_datashape, awkward.types.ArrayType):
                        branch_datashape = branch_datashape.content

                    branch_dtype = self._branch_ak_to_np(branch_datashape)

            if branch_dict is not None:
                self._check_not_counter(branch_name)
                if branch_name not in self._branch_lookup:
                    self._branch_lookup[branch_name] = len(self._branch_data)
                    self._branch_data.append(
                        {
                            "kind": "record",
                            "name": branch_name,
                            "keys": list(branch_dict),
                        }
                    )

                    for key, content in branch_dict.items():
                        subname = self._field_name(branch_name, key)
                        try:
                            dtype = numpy.dtype(content)
                        except Exception as err:
                            raise TypeError(
                                f"values of a dict must be NumPy types\n\n    key {key!r} has type {content!r}"
                            ) from err
                        self._check_not_counter(subname)
                        record_fields.add(subname)
                        if subname not in self._branch_lookup:
                            self._branch_lookup[subname] = len(self._branch_data)
                            self._branch_data.append(
                                self._branch_np(subname, content, dtype)
                            )

            elif branch_dtype is not None:
                if branch_dtype.subdtype is not None or branch_dtype.kind not in "iu":
                    self._check_not_counter(branch_name)
                if branch_name not in self._branch_lookup:
                    self._branch_lookup[branch_name] = len(self._branch_data)
                    self._branch_data.append(
                        self._branch_np(branch_name, branch_type, branch_dtype)
                    )

            else:
                parameters = branch_datashape.parameters

                if parameters.get("__array__") == "string":
                    self._check_not_counter(branch_name)
                    if branch_name not in self._branch_lookup:
                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            self._branch_np(branch_name, branch_type, numpy.dtype(str))
                        )

                elif parameters.get("__array__") == "bytes":
                    raise NotImplementedError("array of bytes")

                # 'awkward' is not in namespace
                elif type(branch_datashape).__name__ == "ListType":
                    content = branch_datashape.content

                    counter_name = self._counter_name(branch_name)
                    counter_dtype = numpy.dtype(numpy.int32)
                    counter = self._branch_np(
                        counter_name, counter_dtype, counter_dtype, kind="counter"
                    )
                    if counter_name in self._branch_lookup:
                        index = self._branch_lookup[counter_name]
                        existing = self._branch_data[index]
                        if existing["kind"] == "counter":
                            # jagged branches that share a counter share its datum
                            counter = existing
                        elif (
                            existing["kind"] == "normal"
                            and existing["counter"] is None
                            and existing["shape"] == ()
                            and existing["dtype"].kind in "iu"
                            and counter_name not in record_fields
                        ):
                            # counters replace scalar integer branches; replace the
                            # datum in place, because deleting it would shift every
                            # later datum down by one and invalidate the indices
                            # that self._branch_lookup already holds for them
                            self._branch_data[index] = counter
                        else:
                            raise ValueError(
                                f"counter {counter_name!r} for jagged branch {branch_name!r} "
                                f"collides with a branch of the same name that is not a "
                                f"scalar integer; rename one of them or choose a "
                                f"different counter_name"
                            )
                    else:
                        self._branch_lookup[counter_name] = len(self._branch_data)
                        self._branch_data.append(counter)

                    self._check_not_counter(branch_name)

                    if type(content).__name__ == "RecordType":
                        contents = content.contents
                        keys = content.fields
                        if keys is None:
                            keys = [str(x) for x in range(len(contents))]

                        if branch_name not in self._branch_lookup:
                            self._branch_lookup[branch_name] = len(self._branch_data)
                            self._branch_data.append(
                                {"kind": "record", "name": branch_name, "keys": keys}
                            )

                            for key, cont in zip(keys, contents, strict=True):
                                subname = self._field_name(branch_name, key)
                                dtype = self._branch_ak_to_np(cont)
                                if dtype is None:
                                    raise TypeError(
                                        f"fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {key!r} has type {cont!s}"
                                    )
                                self._check_not_counter(subname)
                                if subname not in self._branch_lookup:
                                    self._branch_lookup[subname] = len(
                                        self._branch_data
                                    )
                                    self._branch_data.append(
                                        self._branch_np(
                                            subname, cont, dtype, counter=counter
                                        )
                                    )

                    else:
                        dt = self._branch_ak_to_np(content)
                        if dt is None:
                            raise TypeError(
                                f"cannot write Awkward Array type to ROOT file:\n\n    {branch_datashape!s}"
                            )
                        if branch_name not in self._branch_lookup:
                            self._branch_lookup[branch_name] = len(self._branch_data)
                            self._branch_data.append(
                                self._branch_np(branch_name, dt, dt, counter=counter)
                            )

                elif type(branch_datashape).__name__ == "RecordType":
                    contents = branch_datashape.contents
                    keys = branch_datashape.fields
                    if keys is None:
                        keys = [str(x) for x in range(len(contents))]

                    self._check_not_counter(branch_name)
                    if branch_name not in self._branch_lookup:
                        self._branch_lookup[branch_name] = len(self._branch_data)
                        self._branch_data.append(
                            {"kind": "record", "name": branch_name, "keys": keys}
                        )

                        for key, content in zip(keys, contents, strict=True):
                            subname = self._field_name(branch_name, key)
                            dtype = self._branch_ak_to_np(content)
                            if dtype is None:
                                raise TypeError(
                                    f"fields of a record must be NumPy types, though the record itself may be in a jagged array\n\n    field {key!r} has type {content!s}"
                                )
                            self._check_not_counter(subname)
                            record_fields.add(subname)
                            if subname not in self._branch_lookup:
                                self._branch_lookup[subname] = len(self._branch_data)
                                self._branch_data.append(
                                    self._branch_np(subname, content, dtype)
                                )

                else:
                    raise TypeError(
                        f"cannot write Awkward Array type to ROOT file:\n\n    {branch_datashape!s}"
                    )

        self._num_entries = 0
        self._num_baskets = 0

        self._metadata_start = None
        self._metadata = {
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fSavedBytes": 0,
            "fFlushedBytes": 0,
            "fWeight": 1.0,
            "fTimerInterval": 0,
            "fScanField": 25,
            "fUpdate": 0,
            "fDefaultEntryOffsetLen": 1000,
            "fNClusterRange": 0,
            "fMaxEntries": 1000000000000,
            "fMaxEntryLoop": 1000000000000,
            "fMaxVirtualSize": 0,
            "fAutoSave": -300000000,
            "fAutoFlush": -30000000,
            "fEstimate": 1000000,
        }
        self._key = None

    def _check_not_counter(self, name):
        index = self._branch_lookup.get(name)
        if index is not None and self._branch_data[index]["kind"] == "counter":
            raise ValueError(
                f"branch {name!r} collides with a generated counter of the same "
                f"name; rename one of them or choose a different counter_name or "
                f"field_name"
            )

    def _branch_ak_to_np(self, branch_datashape):
        if type(branch_datashape).__name__ == "UnknownType":
            return numpy.dtype("float64")
        elif type(branch_datashape).__name__ == "NumpyType":
            return numpy.dtype(branch_datashape.primitive)
        elif type(branch_datashape).__name__ == "RegularType":
            content = self._branch_ak_to_np(branch_datashape.content)
            if content is None:
                return None
            elif content.subdtype is None:
                dtype, shape = content, ()
            else:
                dtype, shape = content.subdtype
            return numpy.dtype((dtype, (branch_datashape.size, *shape)))
        else:
            return None

    def _branch_np(
        self, branch_name, branch_type, branch_dtype, counter=None, kind="normal"
    ):
        branch_dtype = branch_dtype.newbyteorder(">")

        if branch_dtype.subdtype is None:
            branch_shape = ()
        else:
            branch_dtype, branch_shape = branch_dtype.subdtype

        letter = _dtype_to_char.get(branch_dtype)
        if letter is None:
            raise TypeError(f"cannot write NumPy dtype {branch_dtype} in TTree")

        dims = ""
        if counter is not None:
            # ROOT finds the counter by its leaf's name, which needn't be its
            # branch's name in a ROOT-written tree
            dims = "[" + counter.get("leaf_name", counter["fName"]) + "]"
        if len(branch_shape) > 0:
            dims = dims + "".join("[" + str(x) + "]" for x in branch_shape)

        title = f"{branch_name}{dims}/{letter}"

        return {
            "fName": branch_name,
            "branch_type": branch_type,
            "kind": kind,
            "counter": counter,
            "dtype": branch_dtype,
            "shape": branch_shape,
            "fTitle": title,
            "compression": self._directory.freesegments.fileheader.compression,
            "fBasketSize": 32000,
            "fEntryOffsetLen": 0 if counter is None else 1000,
            "fOffset": 0,
            "fSplitLevel": 0,
            "fFirstEntry": 0,
            "fTotBytes": 0,
            "fZipBytes": 0,
            "fBasketBytes": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
            ),
            "fBasketEntry": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
            ),
            "fBasketSeek": numpy.zeros(
                self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
            ),
            "arrays_write_start": 0,
            "arrays_write_stop": 0,
            "metadata_start": None,
            "basket_metadata_start": None,
            "tleaf_reference_number": None,
            "tleaf_maximum_value": 0,
            "tleaf_special_struct": None,
        }

    def __repr__(self):
        return "{}({}, {}, {}, {}, {}, {}, {})".format(
            type(self).__name__,
            self._directory,
            self._name,
            self._title,
            [(datum["fName"], datum["branch_type"]) for datum in self._branch_data],
            self._freesegments,
            self._basket_capacity,
            self._resize_factor,
        )

    @property
    def directory(self):
        return self._directory

    @property
    def key(self):
        return self._key

    @property
    def name(self):
        return self._key.name

    @property
    def title(self):
        return self._key.title

    @property
    def freesegments(self):
        return self._freesegments

    @property
    def counter_name(self):
        return self._counter_name

    @property
    def field_name(self):
        return self._field_name

    @property
    def basket_capacity(self):
        return self._basket_capacity

    @property
    def resize_factor(self):
        return self._resize_factor

    @property
    def location(self):
        return self._key.location

    @property
    def num_entries(self):
        return self._num_entries

    @property
    def num_baskets(self):
        return self._num_baskets

    def _relocate(self, file, sink):
        """
        Rewrites this tree's whole metadata blob from scratch via write_anew(),
        relocating it (freeing its old space, allocating new space), and moves
        the WritableFile._trees cache entry to match. Shared by the two
        situations that need a full rewrite rather than an incremental
        write_updates() patch: basket-capacity growth, and (see
        _needs_relocation_before_extend) establishing Uproot's own canonical
        layout for a tree write_updates() hasn't verified it can trust yet.
        """
        # seek_location is the tree's actual physical on-disk position (it's
        # what every sink.read(key.seek_location, ...) call in
        # _load_existing_ttree uses), and what file._move_tree()/
        # WritableFile._trees are keyed by everywhere else in the codebase.
        # location coincides with it for any tree Uproot itself has written
        # (all our own tests, hence never catching this), but can diverge for
        # a preexisting (e.g. ROOT-written) tree's Key object -- silently
        # freeing/keying the wrong byte range and cache entry.
        oldloc = start = self._key.seek_location
        stop = start + self._key.num_bytes + self._key.compressed_bytes

        self.write_anew(sink)

        newloc = self._key.seek_location
        file._move_tree(oldloc, newloc)

        self._freesegments.release(start, stop)
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

    def _prepare_extend(self, data, extra=()):
        """
        Checks ``data`` against this tree's branches and converts it to what
        ``extend`` writes, without changing anything, so that a failure leaves
        the tree as it was. ``extra`` are the datums of branches about to be
        added, which ``data`` is checked as if they already were. Returns
        ``(tofill, num_entries)``.
        """
        branch_data = self._branch_data + list(extra)
        branch_lookup = dict(self._branch_lookup)
        for i, datum in enumerate(extra):
            branch_lookup[datum["fName"]] = len(self._branch_data) + i

        provided = None

        if uproot._util.from_module(data, "pandas"):
            import pandas

            if isinstance(
                data, pandas.DataFrame
            ) and uproot._util.pandas_has_attr_is_numeric(pandas)(data.index):
                provided = dataframe_to_dict(data)

        if isinstance(data, awkward.Array):
            if data.ndim > 1 and not data.layout.purelist_isregular:
                provided = {
                    self._counter_name(""): numpy.asarray(
                        awkward.num(data, axis=1), dtype=">u4"
                    )
                }
            else:
                provided = {}
            for k, v in zip(awkward.fields(data), awkward.unzip(data), strict=True):
                provided[k] = v

        if isinstance(data, numpy.ndarray) and data.dtype.fields is not None:
            provided = recarray_to_dict(data)

        if provided is None:
            if not isinstance(data, Mapping) or not all(
                isinstance(x, str) for x in data
            ):
                raise TypeError(
                    "'extend' requires a mapping from branch name (str) to arrays"
                )

            provided = {}
            for k, v in data.items():
                if not uproot._util.from_module(
                    v, "pandas"
                ) and not uproot._util.from_module(v, "awkward"):
                    datum = branch_data[branch_lookup[k]] if k in branch_lookup else {}
                    if datum.get("counter") is not None and not hasattr(v, "dtype"):
                        # lists for a jagged branch are jagged, even if they all
                        # happen to have the same length or there are none; the
                        # type comes from the branch, not from the values
                        v = _jagged_from_lists(  # noqa: PLW2901 (overwriting v)
                            v, datum["dtype"], datum["shape"]
                        )

                    elif not hasattr(v, "dtype") and not isinstance(v, Mapping):
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter(
                                    "error",
                                    category=getattr(
                                        numpy, "exceptions", numpy
                                    ).VisibleDeprecationWarning,
                                )
                                array = numpy.array(v)
                            if array.dtype == numpy.dtype("O"):
                                raise Exception
                            v = array  # noqa: PLW2901 (overwriting v)
                        except (
                            getattr(
                                numpy, "exceptions", numpy
                            ).VisibleDeprecationWarning,
                            Exception,
                        ):
                            v = awkward.from_iter(v)  # noqa: PLW2901 (overwriting v)

                    if getattr(v, "dtype", None) == numpy.dtype("O"):
                        v = awkward.from_iter(v)  # noqa: PLW2901 (overwriting v)

                if (
                    isinstance(v, awkward.Array)
                    and v.ndim > 1
                    and not v.layout.purelist_isregular
                ):
                    if (
                        k in branch_lookup
                        and branch_data[branch_lookup[k]].get("counter") is not None
                    ):
                        kk = branch_data[branch_lookup[k]]["counter"]["fName"]
                    else:
                        kk = self._counter_name(k)
                    vv = numpy.asarray(awkward.num(v, axis=1), dtype=">u4")
                    if kk in provided:
                        check_counter_agrees(kk, vv, provided[kk], counted=k)
                    provided[kk] = vv

                if k in provided:
                    check_counter_agrees(k, provided[k], v)
                provided[k] = v

        actual_branches = {}
        for datum in branch_data:
            if datum["kind"] == "record":
                if datum["name"] in provided:
                    recordarray = provided.pop(datum["name"])

                    if uproot._util.from_module(recordarray, "pandas"):
                        import pandas

                        if isinstance(recordarray, pandas.DataFrame):
                            tmp = {"index": recordarray.index.values}
                            for column in recordarray.columns:
                                tmp[column] = recordarray[column]
                            recordarray = tmp

                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = recordarray[
                            key
                        ]

                elif datum["name"] == "":
                    for key in datum["keys"]:
                        provided[self._field_name(datum["name"], key)] = provided.pop(
                            key
                        )

                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {}".format(
                            repr(datum["name"])
                        )
                    )

            else:
                if datum["fName"] in provided:
                    actual_branches[datum["fName"]] = provided.pop(datum["fName"])
                else:
                    raise ValueError(
                        "'extend' must be given an array for every branch; missing {}".format(
                            repr(datum["fName"])
                        )
                    )

        if len(provided) != 0:
            raise ValueError(
                "'extend' was given data that do not correspond to any branch: {}".format(
                    ", ".join(repr(x) for x in provided)
                )
            )

        tofill = []
        num_entries = None
        for branch_name, branch_array in actual_branches.items():
            if num_entries is None:
                num_entries = len(branch_array)
            elif num_entries != len(branch_array):
                raise ValueError(
                    f"'extend' must fill every branch with the same number of entries; {branch_name!r} has {len(branch_array)} entries"
                )

            datum = branch_data[branch_lookup[branch_name]]
            if datum["kind"] == "record":  # pragma: no cover  (these are leaf branches)
                continue

            if datum["counter"] is None:
                if datum["dtype"] == ">U0":
                    layout = awkward.to_layout(branch_array)
                    if isinstance(layout, awkward.contents.IndexedArray):
                        layout = layout.project()
                    if isinstance(
                        layout,
                        (
                            awkward.contents.ListArray,
                            awkward.contents.RegularArray,
                            awkward.contents.ListOffsetArray,
                        ),
                    ):
                        # start_at_zero=True normalizes offsets[0] to 0 and
                        # trims the content, which is required for a sliced
                        # array (e.g. ak.Array([...])[2:]) to produce a basket
                        # that reads back correctly.
                        layout = layout.to_ListOffsetArray64(True)

                    lengths = numpy.asarray(awkward.num(layout))
                    which_big = lengths >= 255

                    lengths_extension_offsets = numpy.empty(
                        len(layout) + 1, numpy.int64
                    )
                    lengths_extension_offsets[0] = 0
                    numpy.cumsum(which_big * 4, out=lengths_extension_offsets[1:])

                    lengths_extension = awkward.contents.ListOffsetArray(
                        awkward.index.Index64(lengths_extension_offsets),
                        awkward.contents.NumpyArray(
                            lengths[which_big].astype(">u4").view("u1")
                        ),
                    )

                    lengths[which_big] = 255

                    leafc_data_awkward = awkward.concatenate(
                        [
                            lengths.reshape(-1, 1).astype("u1"),
                            lengths_extension,
                            awkward.without_parameters(layout),
                        ],
                        axis=1,
                    )

                    big_endian = numpy.asarray(awkward.flatten(leafc_data_awkward))
                    big_endian_offsets = (
                        lengths_extension_offsets
                        + numpy.asarray(layout.offsets)
                        + numpy.arange(len(layout.offsets))
                    ).astype(">i4", copy=True)
                    tofill.append(
                        (
                            branch_name,
                            datum["compression"],
                            big_endian,
                            big_endian_offsets,
                        )
                    )
                else:
                    big_endian = uproot._util.ensure_numpy(branch_array).astype(
                        datum["dtype"], copy=False
                    )
                    if big_endian.shape != (len(branch_array),) + datum["shape"]:
                        raise ValueError(
                            "'extend' must fill branches with a consistent shape: has {}, trying to fill with {}".format(
                                datum["shape"],
                                big_endian.shape[1:],
                            )
                        )
                    tofill.append((branch_name, datum["compression"], big_endian, None))

            else:
                big_endian, big_endian_offsets = _normalize_jagged_array(
                    branch_name,
                    branch_array,
                    datum["dtype"],
                    datum["shape"],
                    operation="extend",
                )
                tofill.append(
                    (branch_name, datum["compression"], big_endian, big_endian_offsets)
                )

        return tofill, num_entries

    def extend(self, file, sink, data, prepared=None):
        """
        Writes one more basket per branch with ``data``. ``prepared`` is what
        :ref:`_prepare_extend` returns for ``data``, if the caller already has it.
        """
        # validate everything before touching the file
        if prepared is None:
            prepared = self._prepare_extend(data)
        tofill, num_entries = prepared

        relocated = False
        if self._needs_relocation_before_extend:
            # metadata_start/basket_metadata_start (computed structurally by
            # _build_out()) are only correct for a tree actually laid out the
            # way Uproot's own write_anew() lays one out. A tree loaded from an existing file
            # may have been written by ROOT instead, with a byte-for-byte
            # different (but semantically equivalent) layout Uproot's writer
            # doesn't replicate -- write_updates() patching at Uproot's
            # assumed offsets would then silently corrupt unrelated bytes.
            # write_anew() here establishes Uproot's own canonical layout for
            # real (the same thing add_branches() already always does, which
            # is why it already works on ROOT-written files), so every
            # write_updates() patch from here on is guaranteed to target the
            # right bytes. Costs one full blob rewrite, but only once, on the
            # first extend() after loading -- not on every call.
            self._relocate(file, sink)
            self._needs_relocation_before_extend = False
            relocated = True

        # expand capacity if this would REACH (not EXCEED) the existing capacity
        # that's because completely a full fBasketEntry has nowhere to put the
        # number of entries in the last basket (it's a fencepost principle thing),
        # forcing ROOT and Uproot to look it up from the basket header.

        if self._num_baskets >= self._basket_capacity - 1:
            self._basket_capacity = max(
                self._basket_capacity + 1,
                math.ceil(self._basket_capacity * self._resize_factor),
            )

            for datum in self._branch_data:
                if datum["kind"] == "record":
                    continue

                fBasketBytes = datum["fBasketBytes"]
                fBasketEntry = datum["fBasketEntry"]
                fBasketSeek = datum["fBasketSeek"]
                datum["fBasketBytes"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype1
                )
                datum["fBasketEntry"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype2
                )
                datum["fBasketSeek"] = numpy.zeros(
                    self._basket_capacity, uproot.models.TBranch._tbranch13_dtype3
                )
                datum["fBasketBytes"][: len(fBasketBytes)] = fBasketBytes
                datum["fBasketEntry"][: len(fBasketEntry)] = fBasketEntry
                datum["fBasketSeek"][: len(fBasketSeek)] = fBasketSeek
                datum["fBasketEntry"][len(fBasketEntry)] = self._num_entries

            self._relocate(file, sink)
            relocated = True

        if num_entries == 0:
            # write_anew() leaves what write_updates() patches, such as the
            # counters' fMaximum, at placeholder values
            if relocated:
                self.write_updates(sink)
            return

        # actually write baskets into the file
        uncompressed_bytes = 0
        compressed_bytes = 0
        for branch_name, compression, big_endian, big_endian_offsets in tofill:
            datum = self._branch_data[self._branch_lookup[branch_name]]

            if datum["dtype"] == ">U0":
                totbytes, zipbytes, location, fLen = self.write_string_basket(
                    sink, branch_name, compression, big_endian, big_endian_offsets
                )
                # fLen is the size of the biggest string; accumulate the max
                # across extends and keep it per-branch (not shared tree-wide).
                datum["fLen"] = max(datum.get("fLen", 0), fLen)
                datum["fEntryOffsetLen"] = 4 * (len(big_endian_offsets) - 1)

            elif big_endian_offsets is None:
                totbytes, zipbytes, location = self.write_np_basket(
                    sink, branch_name, compression, big_endian
                )
                if datum["kind"] == "counter" and big_endian.size > 0:
                    datum["tleaf_maximum_value"] = max(
                        int(big_endian.max()), datum["tleaf_maximum_value"]
                    )
            else:
                totbytes, zipbytes, location = self.write_jagged_basket(
                    sink, branch_name, compression, big_endian, big_endian_offsets
                )
                datum["fEntryOffsetLen"] = 4 * (len(big_endian_offsets) - 1)
            uncompressed_bytes += totbytes
            compressed_bytes += zipbytes

            datum["fTotBytes"] += totbytes
            datum["fZipBytes"] += zipbytes

            datum["fBasketBytes"][self._num_baskets] = zipbytes

            if self._num_baskets + 1 < self._basket_capacity:
                fBasketEntry = datum["fBasketEntry"]
                i = self._num_baskets
                fBasketEntry[i + 1] = num_entries + fBasketEntry[i]

            datum["fBasketSeek"][self._num_baskets] = location

            datum["arrays_write_stop"] = self._num_baskets + 1

        # update TTree metadata in file
        self._num_entries += num_entries
        self._num_baskets += 1
        self._metadata["fTotBytes"] += uncompressed_bytes
        self._metadata["fZipBytes"] += compressed_bytes

        self.write_updates(sink)

    def _build_out(self):
        """
        Serializes this TTree's metadata blob (TTree + TBranches + TLeaves,
        everything but the TBaskets themselves) into a list of byte chunks,
        exactly as ``write_anew`` writes it.

        As a side effect, this sets ``self._metadata_start`` and, for every
        branch, ``datum["metadata_start"]``, ``datum["basket_metadata_start"]``,
        and ``datum["tleaf_reference_number"]`` — the byte offsets (relative to
        this blob) that ``write_updates`` writes at directly, without going
        through ``write_anew`` again. This is the single source of truth for
        that layout: computing it any other way (e.g. searching a decompressed
        blob for the current value of a field) risks disagreeing with it,
        especially when that field's current value is 0 and matches unrelated
        bytes. ``uproot.writing.writable.WritableDirectory._load_existing_ttree``
        calls this once, on a freshly reconstructed cascade, purely to recover
        these offsets — the returned ``out`` is discarded, nothing is written.
        """
        key_num_bytes = uproot.reading._key_format_big.size + 6
        name_asbytes = self._name.encode(errors="surrogateescape")
        title_asbytes = self._title.encode(errors="surrogateescape")
        key_num_bytes += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        key_num_bytes += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        out = [None]
        ttree_header_index = 0

        tobject = uproot.models.TObject.Model_TObject.empty()
        tnamed = uproot.models.TNamed.Model_TNamed.empty()
        tnamed._bases.append(tobject)
        tnamed._members["fTitle"] = self._title
        tnamed._serialize(out, True, self._name, uproot.const.kMustCleanup)

        # TAttLine v2, fLineColor: 602 fLineStyle: 1 fLineWidth: 1
        # TAttFill v2, fFillColor: 0, fFillStyle: 1001
        # TAttMarker v2, fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1.0
        out.append(
            b"@\x00\x00\x08\x00\x02\x02Z\x00\x01\x00\x01"
            b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9"
            b"@\x00\x00\n\x00\x02\x00\x01\x00\x01?\x80\x00\x00"
        )

        metadata_out_index = len(out)
        out.append(
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            )
        )

        # speedbump (0), fClusterRangeEnd (empty array),
        # speedbump (0), fClusterSize (empty array)
        # fIOFeatures (TIOFeatures)
        out.append(b"\x00\x00@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

        tleaf_reference_numbers = []

        tobjarray_of_branches_index = len(out)
        out.append(None)

        num_branches = sum(
            0 if datum["kind"] == "record" else 1 for datum in self._branch_data
        )

        # TObjArray header with fName: ""
        out.append(b"\x00\x01\x00\x00\x00\x00\x03\x00@\x00\x00")
        out.append(
            uproot.models.TObjArray._tobjarray_format1.pack(
                num_branches,  # TObjArray fSize
                0,  # TObjArray fLowerBound
            )
        )

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            any_tbranch_index = len(out)
            out.append(None)
            out.append(b"TBranch\x00")

            tbranch_index = len(out)
            out.append(None)

            tbranch_tobject = uproot.models.TObject.Model_TObject.empty()
            tbranch_tnamed = uproot.models.TNamed.Model_TNamed.empty()
            tbranch_tnamed._bases.append(tbranch_tobject)
            tbranch_tnamed._members["fTitle"] = datum["fTitle"]
            tbranch_tnamed._serialize(
                out, True, datum["fName"], numpy.uint32(0x00400000)
            )

            # TAttFill v2, fFillColor: 0, fFillStyle: 1001
            out.append(b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9")

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # Lie about the compression level so that ROOT checks and does the right thing.
            # https://github.com/root-project/root/blob/87a998d48803bc207288d90038e60ff148827664/tree/tree/src/TBasket.cxx#L560-L578
            # Without this, when small buffers are left uncompressed, ROOT complains about them not being compressed.
            # (I don't know where the "no, really, this is uncompressed" bit is.)
            fCompress = 0

            out.append(
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    datum["arrays_write_stop"],  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                )
            )

            # fIOFeatures (TIOFeatures)
            out.append(b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

            out.append(
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    len(datum["fBasketBytes"]),  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                )
            )

            # empty TObjArray of TBranches
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            subtobjarray_of_leaves_index = len(out)
            out.append(None)

            # TObjArray header with fName: "", fSize: 1, fLowerBound: 0
            out.append(
                b"\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00"
            )

            absolute_location = key_num_bytes + sum(
                len(x) for x in out if x is not None
            )
            absolute_location += 8 + 6 * (sum(1 if x is None else 0 for x in out) - 1)
            datum["tleaf_reference_number"] = absolute_location + 2
            tleaf_reference_numbers.append(datum["tleaf_reference_number"])

            subany_tleaf_index = len(out)
            out.append(None)

            letter = _dtype_to_char[datum["dtype"]]
            letter_upper = letter.upper()
            out.append(("TLeaf" + letter_upper).encode() + b"\x00")
            if letter_upper == "O":
                special_struct = uproot.models.TLeaf._tleafO1_format1
            elif letter_upper == "B":
                special_struct = uproot.models.TLeaf._tleafb1_format1
            elif letter_upper == "S":
                special_struct = uproot.models.TLeaf._tleafs1_format1
            elif letter_upper == "I":
                special_struct = uproot.models.TLeaf._tleafi1_format1
            elif letter_upper == "G":  # pragma: no cover  (not in _dtype_to_char)
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif letter_upper == "L":
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif letter_upper == "F":
                special_struct = uproot.models.TLeaf._tleaff1_format1
            elif letter_upper == "D":
                special_struct = uproot.models.TLeaf._tleafd1_format1
            elif letter_upper == "C":  # pragma: no branch  (all of _dtype_to_char)
                special_struct = uproot.models.TLeaf._tleafc1_format1

            fLenType = datum["dtype"].itemsize
            fIsUnsigned = letter != letter_upper

            if datum["shape"] == ():
                dims = ""
            else:
                dims = "".join("[" + str(x) + "]" for x in datum["shape"])

            if datum["counter"] is not None:
                counter = datum["counter"]
                dims = "[" + counter.get("leaf_name", counter["fName"]) + "]" + dims

            # single TLeaf
            leaf_name = datum.get("leaf_name", datum["fName"])
            leaf_title = datum.get("leaf_title", leaf_name + dims)
            leaf_name = leaf_name.encode(errors="surrogateescape")
            leaf_title = leaf_title.encode(errors="surrogateescape")
            leaf_name_length = (1 if len(leaf_name) < 255 else 5) + len(leaf_name)
            leaf_title_length = (1 if len(leaf_title) < 255 else 5) + len(leaf_title)

            leaf_header = numpy.array(
                [
                    64,
                    0,
                    0,
                    76,
                    0,
                    1,
                    64,
                    0,
                    0,
                    54,
                    0,
                    2,
                    64,
                    0,
                    0,
                    30,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                ],
                numpy.uint8,
            )
            tmp = leaf_header[0:4].view(">u4")
            tmp[:] = (
                numpy.uint32(
                    42 + leaf_name_length + leaf_title_length + special_struct.size
                )
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[6:10].view(">u4")
            tmp[:] = (
                numpy.uint32(36 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[12:16].view(">u4")
            tmp[:] = (
                numpy.uint32(12 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )

            out.append(leaf_header.tobytes())
            if len(leaf_name) < 255:
                out.append(
                    struct.pack(f">B{len(leaf_name)}s", len(leaf_name), leaf_name)
                )
            else:
                out.append(
                    struct.pack(f">BI{len(leaf_name)}s", 255, len(leaf_name), leaf_name)
                )
            if len(leaf_title) < 255:
                out.append(
                    struct.pack(f">B{len(leaf_title)}s", len(leaf_title), leaf_title)
                )
            else:
                out.append(
                    struct.pack(
                        f">BI{len(leaf_title)}s", 255, len(leaf_title), leaf_title
                    )
                )

            fLen = 1
            for item in datum["shape"]:
                fLen *= item

            # generic TLeaf members
            out.append(
                uproot.models.TLeaf._tleaf2_format0.pack(
                    fLen,
                    fLenType,
                    0,  # fOffset
                    datum["kind"] == "counter",  # fIsRange
                    fIsUnsigned,
                )
            )

            if datum["counter"] is None:
                # null fLeafCount
                out.append(b"\x00\x00\x00\x00")
            else:
                # reference to fLeafCount
                out.append(
                    uproot.deserialization._read_object_any_format1.pack(
                        datum["counter"]["tleaf_reference_number"]
                    )
                )

            # specialized TLeaf* members (fMinimum, fMaximum)
            out.append(special_struct.pack(0, 0))
            datum["tleaf_special_struct"] = special_struct

            out[subany_tleaf_index] = (
                uproot.serialization._serialize_object_any_format1.pack(
                    numpy.uint32(sum(len(x) for x in out[subany_tleaf_index + 1 :]) + 4)
                    | uproot.const.kByteCountMask,
                    uproot.const.kNewClassTag,
                )
            )

            out[subtobjarray_of_leaves_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[subtobjarray_of_leaves_index + 1 :]),
                3,  # TObjArray
            )

            # empty TObjArray of fBaskets (embedded)
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["basket_metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # speedbump and fBasketBytes
            out.append(b"\x01")
            out.append(datum["fBasketBytes"].tobytes())

            # speedbump and fBasketEntry
            out.append(b"\x01")
            out.append(datum["fBasketEntry"].tobytes())

            # speedbump and fBasketSeek
            out.append(b"\x01")
            out.append(datum["fBasketSeek"].tobytes())

            # empty fFileName
            out.append(b"\x00")

            out[tbranch_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[tbranch_index + 1 :]),
                13,  # TBranch
            )

            out[any_tbranch_index] = (
                uproot.serialization._serialize_object_any_format1.pack(
                    numpy.uint32(sum(len(x) for x in out[any_tbranch_index + 1 :]) + 4)
                    | uproot.const.kByteCountMask,
                    uproot.const.kNewClassTag,
                )
            )

        out[tobjarray_of_branches_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[tobjarray_of_branches_index + 1 :]),
            3,  # TObjArray
        )

        # TObjArray of TLeaf references
        tleaf_reference_bytes = numpy.array(tleaf_reference_numbers, ">u4").tobytes()
        out.append(
            struct.pack(
                ">I13sI4s",
                (21 + len(tleaf_reference_bytes)) | uproot.const.kByteCountMask,
                b"\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00",
                len(tleaf_reference_numbers),
                b"\x00\x00\x00\x00",
            )
        )

        out.append(tleaf_reference_bytes)

        # null fAliases (b"\x00\x00\x00\x00")
        # empty fIndexValues array (4-byte length is zero)
        # empty fIndex array (4-byte length is zero)
        # null fTreeIndex (b"\x00\x00\x00\x00")
        # null fFriends (b"\x00\x00\x00\x00")
        # null fUserInfo (b"\x00\x00\x00\x00")
        # null fBranchRef (b"\x00\x00\x00\x00")
        out.append(b"\x00" * 28)

        out[ttree_header_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[ttree_header_index + 1 :]),
            20,  # TTree
        )

        self._metadata_start = sum(len(x) for x in out[:metadata_out_index])

        return out

    def write_anew(self, sink):
        out = self._build_out()
        raw_data = b"".join(out)
        self._key = self._directory.add_object(
            sink,
            "TTree",
            self._name,
            self._title,
            raw_data,
            len(raw_data),
            replaces=self._key,
            big=True,
        )

    def write_updates(self, sink):
        base = self._key.seek_location + self._key.num_bytes

        sink.write(
            base + self._metadata_start,
            uproot.models.TTree._ttree20_format1.pack(
                self._num_entries,
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            ),
        )

        for datum in self._branch_data:
            if datum["kind"] == "record":
                continue

            position = base + datum["metadata_start"]

            # Lie about the compression level so that ROOT checks and does the right thing.
            # https://github.com/root-project/root/blob/87a998d48803bc207288d90038e60ff148827664/tree/tree/src/TBasket.cxx#L560-L578
            # Without this, when small buffers are left uncompressed, ROOT complains about them not being compressed.
            # (I don't know where the "no, really, this is uncompressed" bit is.)
            fCompress = 0

            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format1.pack(
                    fCompress,
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    datum["arrays_write_stop"],  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                ),
            )

            position += uproot.models.TBranch._tbranch13_format1.size + 11
            sink.write(
                position,
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    len(datum["fBasketBytes"]),  # fMaxBaskets
                    datum["fSplitLevel"],
                    self._num_entries,  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                ),
            )

            start, stop = datum["arrays_write_start"], datum["arrays_write_stop"]

            fBasketBytes_part = datum["fBasketBytes"][start:stop].tobytes()
            fBasketEntry_part = datum["fBasketEntry"][start : stop + 1].tobytes()
            fBasketSeek_part = datum["fBasketSeek"][start:stop].tobytes()

            position = base + datum["basket_metadata_start"] + 1
            position += datum["fBasketBytes"][:start].nbytes
            sink.write(position, fBasketBytes_part)
            position += len(fBasketBytes_part)
            position += datum["fBasketBytes"][stop:].nbytes

            position += 1
            position += datum["fBasketEntry"][:start].nbytes
            sink.write(position, fBasketEntry_part)
            position += len(fBasketEntry_part)
            position += datum["fBasketEntry"][stop + 1 :].nbytes

            position += 1
            position += datum["fBasketSeek"][:start].nbytes
            sink.write(position, fBasketSeek_part)
            position += len(fBasketSeek_part)
            position += datum["fBasketSeek"][stop:].nbytes

            datum["arrays_write_start"] = datum["arrays_write_stop"]

            if datum["dtype"] == ">U0":
                position = (
                    base
                    + datum["basket_metadata_start"]
                    - 25  # empty TObjArray of fBaskets (embedded)
                    - 8  # specialized TLeaf* members (fMinimum, fMaximum)
                    - 4  # null fLeafCount
                    - 14  # generic TLeaf members
                )
                sink.write(
                    position,
                    uproot.models.TLeaf._tleaf2_format0.pack(
                        datum.get("fLen", 0),
                        datum["dtype"].itemsize,
                        0,
                        datum["kind"] == "counter",
                        _dtype_to_char[datum["dtype"]]
                        != _dtype_to_char[datum["dtype"]].upper(),
                    ),
                )

            if datum["kind"] == "counter":
                position = (
                    base
                    + datum["basket_metadata_start"]
                    - 25  # empty TObjArray of fBaskets (embedded)
                    - datum["tleaf_special_struct"].size
                )
                sink.write(
                    position,
                    datum["tleaf_special_struct"].pack(
                        0,
                        datum["tleaf_maximum_value"],
                    ),
                )

        sink.flush()

    def write_np_basket(self, sink, branch_name, compression, array):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item

        uncompressed_data = array.tobytes()
        compressed_data = uproot.compression.compress(uncompressed_data, compression)

        fObjlen = len(uncompressed_data)
        fNbytes = fKeylen + len(compressed_data)

        if max(fObjlen, fNbytes) > uproot.const.kMaxTBasketBytes:
            raise ValueError(
                f"Numpy array data of branch {branch_name} has an uncompressed size of {fObjlen} bytes "
                f"and a compressed size of {fNbytes} bytes, which is too large to fit in a TBasket. "
                "Neither of these sizes can exceed 2 GiB."
            )

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                itemsize,  # fNevBufSize
                len(array),  # fNevBuf
                fKeylen + len(uncompressed_data),  # fLast
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(compressed_data)

        sink.write(location, b"".join(out))
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fKeylen + fObjlen, fNbytes, location

    def write_jagged_basket(self, sink, branch_name, compression, array, offsets):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)

        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        # offsets became a *copy* of the Awkward Array's offsets
        # when it was converted to big-endian (astype with copy=True)
        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:
            itemsize *= item
        offsets *= itemsize
        offsets += fKeylen

        raw_array = array.tobytes()
        raw_offsets = offsets.tobytes()
        uncompressed_data = (
            raw_array + _tbasket_offsets_length.pack(len(offsets)) + raw_offsets
        )
        compressed_data = uproot.compression.compress(uncompressed_data, compression)

        fLast = offsets[-1]
        offsets[-1] = 0

        fObjlen = len(uncompressed_data)
        fNbytes = fKeylen + len(compressed_data)

        if max(fObjlen, fNbytes) > uproot.const.kMaxTBasketBytes:
            raise ValueError(
                f"Jagged array data of branch {branch_name} has an uncompressed size of {fObjlen} bytes "
                f"and a compressed size of {fNbytes} bytes, which is too large to fit in a TBasket. "
                "Neither of these sizes can exceed 2 GiB."
            )

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                len(offsets) + 1,  # fNevBufSize
                len(offsets) - 1,  # fNevBuf
                fLast,
            )
        )
        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(compressed_data)

        sink.write(location, b"".join(out))
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fKeylen + fObjlen, fNbytes, location

    def write_string_basket(self, sink, branch_name, compression, array, offsets):
        fClassName = uproot.serialization.string("TBasket")
        fName = uproot.serialization.string(branch_name)
        fTitle = uproot.serialization.string(self._name)
        fKeylen = (
            uproot.reading._key_format_big.size
            + len(fClassName)
            + len(fName)
            + len(fTitle)
            + uproot.models.TBasket._tbasket_format2.size
            + 1
        )

        itemsize = array.dtype.itemsize
        for item in array.shape[1:]:  # pragma: no cover  (strings are flat bytes)
            itemsize *= item

        offsets *= itemsize
        offsets += fKeylen

        raw_array = array.tobytes()
        raw_offsets = offsets.tobytes()
        uncompressed_data = (
            raw_array
            + _tbasket_offsets_length.pack(len(offsets))
            + raw_offsets[:-4]
            + b"\x00\x00\x00\x00"
        )
        compressed_data = uproot.compression.compress(uncompressed_data, compression)

        # get size of biggest string in this basket
        fLen = 0 if len(offsets) <= 1 else int(numpy.diff(offsets).max())

        fLast = offsets[-1]
        offsets[-1] = 0

        fObjlen = len(uncompressed_data)
        fNbytes = fKeylen + len(compressed_data)

        if max(fObjlen, fNbytes) > uproot.const.kMaxTBasketBytes:
            raise ValueError(
                f"String data of branch {branch_name} has an uncompressed size of {fObjlen} bytes "
                f"and a compressed size of {fNbytes} bytes, which is too large to fit in a TBasket. "
                "Neither of these sizes can exceed 2 GiB."
            )

        parent_location = self._directory.key.location  # FIXME: is this correct?

        location = self._freesegments.allocate(fNbytes, dry_run=False)

        out = []
        out.append(
            uproot.reading._key_format_big.pack(
                fNbytes,
                1004,  # fVersion
                fObjlen,
                uproot._util.datetime_to_code(datetime.datetime.now()),  # fDatime
                fKeylen,
                0,  # fCycle
                location,  # fSeekKey
                parent_location,  # fSeekPdir
            )
        )
        out.append(fClassName)
        out.append(fName)
        out.append(fTitle)
        out.append(
            uproot.models.TBasket._tbasket_format2.pack(
                3,  # fVersion
                32000,  # fBufferSize
                1000,  # fNevBufSize
                len(offsets) - 1,  # fNevBuf
                fLast,
            )
        )

        out.append(b"\x00")  # part of the Key (included in fKeylen, at least)

        out.append(compressed_data)

        sink.write(location, b"".join(out))
        self._freesegments.write(sink)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()

        return fKeylen + fObjlen, fNbytes, location, fLen


_tbasket_offsets_length = struct.Struct(">I")


def dataframe_to_dict(df):
    """
    Converts a Pandas DataFrame into a dict of NumPy arrays for writing.
    """
    out = {"index": df.index.values}
    for column_name in df.columns:
        out[str(column_name)] = df[column_name].values
    return out


def recarray_to_dict(array):
    """
    Converts a NumPy structured array into a dict of non-structured arrays for writing.
    """
    out = {}
    for field_name in array.dtype.fields:
        field = array[field_name]
        if field.dtype.fields is not None:
            for subfield_name, subfield in recarray_to_dict(field).items():
                out[field_name + "." + subfield_name] = subfield
        else:
            out[field_name] = field
    return out


def _jagged_from_lists(lists, dtype, shape):
    """
    Makes an Awkward Array of variable-length lists of ``dtype`` items with
    ``shape`` from Python lists of lists for a jagged branch.
    """
    offsets = numpy.zeros(len(lists) + 1, dtype=numpy.int64)
    numpy.cumsum([len(x) for x in lists], out=offsets[1:])
    items = [item for x in lists for item in x]
    dtype = dtype.newbyteorder("=")
    if len(items) == 0:
        content = numpy.empty((0, *shape), dtype=dtype)
    else:
        try:
            content = numpy.asarray(items, dtype=dtype)
        except ValueError:
            content = None
        if content is None or content.shape[1:] != shape:
            raise ValueError(
                f"'extend' must fill branches with a consistent shape: has {shape}, "
                "trying to fill with lists of items of a different one"
            )
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(offsets),
            awkward.contents.NumpyArray(content),
        )
    )


def check_counter_agrees(counter_name, generated, explicit, counted=None):
    """
    Raises a ValueError if a counter branch's values that were given explicitly
    disagree with the ones generated from the jagged arrays it counts (such as
    ``counted``, if it's known).
    """
    if not numpy.array_equal(generated, numpy.asarray(explicit)):
        if counted is None:
            raise ValueError(
                f"branch {counter_name!r} provided both as an explicit array and "
                "generated as a counter, and they disagree"
            )
        raise ValueError(
            f"branch {counter_name!r} provided both as an explicit array and "
            f"generated as a counter, and they disagree: it's the name of the "
            f"counter of jagged branch {counted!r}, which counter_name can change"
        )


def _normalize_jagged_array(
    branch_name, branch_array, target_dtype, target_shape, operation="extend"
):
    layout = branch_array.layout
    while not isinstance(layout, awkward.contents.ListOffsetArray):
        if isinstance(layout, awkward.contents.IndexedArray):
            layout = layout.project()

        elif isinstance(layout, awkward.contents.ListArray):
            layout = layout.to_ListOffsetArray64(False)

        else:
            raise TypeError(
                f"jagged branch {branch_name!r} can't be filled with this "
                f"Awkward Array:\n\n{layout!r}"
            )

    content = layout.content
    offsets = numpy.asarray(layout.offsets)

    if offsets[0] != 0:
        content = content[offsets[0] :]
        offsets = offsets - offsets[0]
    if len(content) > offsets[-1]:
        content = content[: offsets[-1]]

    shape = [len(content)]
    while not isinstance(content, awkward.contents.NumpyArray):
        if isinstance(content, awkward.contents.IndexedArray):
            content = content.project()

        elif isinstance(content, awkward.contents.EmptyArray):
            content = content.to_NumpyArray(dtype=numpy.float64)

        elif isinstance(content, awkward.contents.RegularArray):
            shape.append(content.size)
            content = content.content

        else:
            raise TypeError(
                f"jagged branch {branch_name!r} can't be filled with lists "
                f"of this Awkward Array:\n\n{content!r}"
            )

    big_endian = numpy.asarray(content.data, dtype=target_dtype)
    shape = tuple(shape) + big_endian.shape[1:]

    if shape[1:] != target_shape:
        raise ValueError(
            f"'{operation}' must fill branches with a consistent shape: has {target_shape}, trying to fill with {shape[1:]}"
        )
    # keep the fixed inner dimensions (if any) so that write_jagged_basket scales
    # the offsets by them; a RegularArray's content can be longer than it needs
    # to be, so drop the excess
    big_endian = big_endian.reshape(-1)[: math.prod(shape)].reshape(shape)
    big_endian_offsets = offsets.astype(">i4", copy=True)
    return big_endian, big_endian_offsets
