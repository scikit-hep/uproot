# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines behaviors for :doc:`uproot.behaviors.RNTuple.RNTuple` and
:doc:`uproot.behaviors.RNTuple.HasFields` (both ``RField`` and
``RNTuple``).

Most of the functionality of RNTuple-reading is implemented here.

See :doc:`uproot.models.RNTuple` for deserialization of the ``RNTuple``
objects themselves.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Mapping
from functools import partial

import awkward as ak
import numpy

import uproot
import uproot.interpretation.grouped
import uproot.language.python
import uproot.source.chunk
from uproot._util import no_filter, unset
from uproot.behaviors.TBranch import (
    _regularize_array_cache,
    concatenate,  # noqa: F401
    iterate,  # noqa: F401
)


class HasFields(Mapping):
    """
    Abstract class of behaviors for anything that "has fields," namely
    :doc:`uproot.models.RNTuple.RNTuple` and
    :doc:`uproot.models.RNTuple.RField`, which mostly consist of array-reading
    methods.

    A :doc:`uproot.behaviors.RNTuple.HasFields` is a Python ``Mapping``, which
    uses square bracket syntax to extract subfields:

    .. code-block:: python

        my_rntuple["field"]
        my_rntuple["field"]["subfield"]
        my_rntuple["field.subfield"]
        my_rntuple["field.subfield.subsubfield"]
        my_rntuple["field/subfield/subsubfield"]
        my_rntuple["field\\subfield\\subsubfield"]
    """

    @property
    def ntuple(self):
        """
        The :doc:`uproot.models.RNTuple.RNTuple` that this
        :doc:`uproot.models.RNTuple.HasFields` is part of.
        """
        return self._ntuple

    @property
    def title(self):
        """
        The ``description``, under the name of the TBranch attribute it corresponds to, so that
        ``ak_add_doc={"__doc__": "title"}`` documents a TTree or an RNTuple.
        """
        return self.description

    @property
    def num_entries(self):
        """
        The number of entries in the ``RNTuple``.
        """
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            if self._num_entries is None:
                self._num_entries = sum(x.num_entries for x in self.cluster_summaries)
            return self._num_entries
        return self.ntuple.num_entries

    @property
    def fields(self):
        """
        The list of :doc:`uproot.models.RNTuple.RField` directly under
        this :doc:`uproot.models.RNTuple.RNTuple` or
        :doc:`uproot.models.RNTuple.RField` (i.e. not recursive).
        """
        if self._fields is None:
            rntuple = self.ntuple
            if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
                fields = [
                    rntuple.all_fields[i]
                    for i, f in enumerate(rntuple.field_records)
                    if f.parent_field_id == i
                ]
            else:
                fields = []
                for i, f in enumerate(rntuple.field_records):
                    if f.parent_field_id != self._fid or f.parent_field_id == i:
                        continue
                    if rntuple.all_fields[i].is_anonymous:
                        # for anonymous fields, we use their children instead
                        fields.extend(rntuple.all_fields[i].fields)
                    else:
                        fields.append(rntuple.all_fields[i])
            self._fields = fields
        return self._fields

    @property
    def path(self):
        """
        The full path of the field in the :doc:`uproot.models.RNTuple.RNTuple`. When it is
        the ``RNTuple`` itself, this is ``"."``.

        Note that this is not the full path within the ROOT file.
        """
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            return "."
        # For some anonymous fields, the path is not available
        if self.is_anonymous or self.in_variant:
            return None
        if self._path is None:
            path = self.name
            parent = self.parent
            field = self
            while not isinstance(parent, uproot.behaviors.RNTuple.RNTuple):
                if not parent.is_anonymous:
                    path = f"{parent.name}.{path}"
                field = parent
                parent = field.parent
            self._path = path
        return self._path

    def to_akform(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        ak_add_doc=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            ak_add_doc (bool | dict ): If True, add the RField ``description``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value}, add the RField ``value`` to the
                Awkward ``key`` parameter of the array.
                A ``value`` of ``"title"`` (the TBranch attribute) reads the RField ``description``.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns a 2-tuple where the first entry is the Awkward Form with the structure of the data in the ``RNTuple`` or ``RField``,
        and the second entry is the relative path of the requested RField. The second entry is needed in cases where the requested RField
        is a subfield of a collection, which requires constructing the form with information about the parent field.
        """

        keys = self.keys(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
        )
        # A selected field comes with all of its subfields
        keys = _with_subfield_keys(self, keys)
        rntuple = self.ntuple

        top_names = []
        record_list = []
        if self is rntuple:
            field_path = None
            key_prefixes = uproot.models.RNTuple._key_prefixes(keys)
            for field in self.fields:
                # the field needs to be in the keys or be a parent of a field in the keys
                if field.name in key_prefixes:
                    top_names.append(field.name)
                    record_list.append(
                        rntuple._field_form(
                            field.field_id, key_prefixes, ak_add_doc=ak_add_doc
                        )
                    )
        else:
            # If it is a subfield of a collection or a std::array (a LEAF with a nonzero
            # repetition), we need to include the outermost one in the keys so that the
            # dimensions they add are not lost
            path_keys = self.path.split(".")
            top_collection = None
            tmp_field = self.ntuple
            field_path = [self.name]
            for i, key in enumerate(path_keys):
                tmp_field = tmp_field[key]
                if (
                    tmp_field.record.struct_role
                    == uproot.const.RNTupleFieldRole.COLLECTION
                    or tmp_field.record.repetition > 0
                ):
                    top_collection = tmp_field
                    field_path = path_keys[i:]
                    break
            # Always use the full path for keys
            # Also include the field itself
            keys = [self.path] + [f"{self.path}.{k}" for k in keys]
            if top_collection is None:
                top_names.append(self.name)
                record_list.append(
                    rntuple.field_form(self.field_id, keys, ak_add_doc=ak_add_doc)
                )
            else:
                keys += [top_collection.path]
                top_names.append(top_collection.name)
                record_list.append(
                    rntuple.field_form(
                        top_collection.field_id, keys, ak_add_doc=ak_add_doc
                    )
                )

        parameters = (
            uproot._util.ak_add_doc_parameters(self, ak_add_doc, top_level=True) or None
        )

        # A tuple is only kept when all of its fields are selected, since Awkward would
        # renumber a subset of them and the form's fields would no longer match the RFields
        if self is rntuple and rntuple.is_tuple and len(top_names) == len(self.fields):
            top_names = None
        form = ak.forms.RecordForm(
            record_list, top_names, form_key="toplevel", parameters=parameters
        )
        return (form, field_path)

    def arrays(
        self,
        expressions=None,  # TODO: Not implemented yet
        cut=None,  # TODO: Not implemented yet
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        aliases=None,  # TODO: Not implemented yet
        language=uproot.language.python.python_language,  # TODO: Not implemented yet
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,  # TODO: Not implemented yet
        array_cache="inherit",
        library="ak",
        backend="cpu",
        interpreter="cpu",
        ak_add_doc=False,
        how=None,
        virtual=False,
        access_log=None,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        interpretation_executor=None,
        filter_branch=unset,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``RFields`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``RFields``
                selected by the filters are included. (Not implemented yet.)
            cut (None or str): If not None, this expression filters all of the
                ``expressions``. (Not implemented yet.)
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. If the function
                returns False or None, the ``RField`` is excluded; if the function
                returns True, it is included.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases.
                Uses the ``language`` engine to evaluate. (Not implemented yet.)
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``. (Not implemented yet.)
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.RNTuple.RNTuple.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``RPages``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used. (Not implemented yet.)
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            backend (str): The backend Awkward Array will use.
            interpreter (str): If "cpu" will use cpu to interpret raw data. If "gpu" and
                ``backend="cuda"`` will use KvikIO bindings to CuFile and nvCOMP to
                interpret raw data on gpu if available.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``description``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
                Awkward ``key`` parameter of the array.
                A ``value`` of ``"title"`` (the TBranch attribute) reads the RField ``description``.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
                For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip``
                to interleave data from compatible branches.
            virtual (bool): If True, return virtual Awkward arrays, meaning that the data will not be
                loaded into memory until it is accessed.
            access_log (None or object with a ``__iadd__`` method): If an access_log is
                provided, e.g. a list, all materializations of the arrays are
                tracked inside this reference. Only applies if ``virtual=True``.
            interpretation_executor (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns a group of arrays from the ``RNTuple``.

        A selected ``RField`` is read with all of its subfields, even if the
        filters also select some of them. If only a subfield is selected, its
        parents are included with nothing but the path down to it.

        For example:

        .. code-block:: python

            >>> my_ntuple.arrays()
            <Array [{my_vector: [1, 2]}, {...}] type='2 * {my_vector: var * int64}'>

        See also :ref:`uproot.behaviors.RNTuple.HasFields.array` to read a single
        ``RField`` as an array.

        See also :ref:`uproot.behaviors.RNTuple.HasFields.iterate` to iterate over
        the array in contiguous ranges of entries.
        """

        # This temporarily provides basic functionality while expressions are properly implemented
        if expressions is not None:
            if filter_name == no_filter:
                filter_name = expressions
            else:
                raise ValueError(
                    "Expressions are not supported yet. They are currently equivalent to filter_name."
                )

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
        else:
            # some kwargs can't be used with eager arrays
            err = "'{}' cannot be used with 'virtual=False'".format
            if access_log is not None:
                raise ValueError(err("access_log"))

        entry_start, entry_stop = (
            uproot.behaviors.TBranch._regularize_entries_start_stop(
                self.num_entries, entry_start, entry_stop
            )
        )
        library = uproot.interpretation.library._regularize_library(library)

        clusters = self.ntuple.cluster_summaries
        start_cluster_idx, stop_cluster_idx, cluster_offset = _cluster_range(
            clusters, entry_start, entry_stop
        )
        cluster_num_entries = numpy.sum(
            [c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]],
            dtype=int,
        )

        array_cache = _regularize_array_cache(array_cache, self.ntuple._file)

        form, field_path = self.to_akform(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
            ak_add_doc=ak_add_doc,
        )

        # only read columns mentioned in the awkward form
        target_cols = []
        container_dict = {}
        _recursive_find(form, target_cols)

        # With GPU interpretation data can be decompressed and deserialized in
        # parallel. Read requested columns all at once (an empty range reads no
        # clusters, so there is nothing to read)
        if (
            interpreter == "gpu"
            and backend == "cuda"
            and start_cluster_idx < stop_cluster_idx
        ):
            clusters_datas = self.ntuple.gpu_read_cluster_range(
                target_cols, start_cluster_idx, stop_cluster_idx
            )
            clusters_datas._decompress()
            content_dict = self.ntuple.gpu_deserialize_decompressed_content(
                clusters_datas,
                start_cluster_idx,
                stop_cluster_idx,
                cluster_offset=cluster_offset,
            )

        for key in target_cols:
            if "column" in key:
                key_nr = int(key.split("-")[1])
                # Find how many elements should be padded at the beginning
                n_padding = self.ntuple.column_records[key_nr].first_element_index
                n_padding = max(n_padding - cluster_offset, 0)
                dtype = None
                if interpreter == "cpu":
                    content_generator = partial(
                        self.ntuple.read_cluster_range,
                        key_nr,
                        start_cluster_idx,
                        stop_cluster_idx,
                        missing_element_padding=n_padding,
                        array_cache=array_cache,
                        access_log=access_log,
                    )
                    if virtual:
                        total_length, _, dtype = (
                            self.ntuple._expected_array_length_starts_dtype(
                                key_nr,
                                start_cluster_idx,
                                stop_cluster_idx,
                                missing_element_padding=n_padding,
                            )
                        )
                        if "cardinality" in key:
                            total_length -= 1
                        content = (total_length, content_generator)
                    else:
                        content = content_generator()
                elif interpreter == "gpu" and backend == "cuda":
                    if start_cluster_idx < stop_cluster_idx:
                        content = content_dict[key_nr]
                    else:
                        # no clusters are read, so the column is only its padding
                        content = uproot.extras.cupy().asarray(
                            self.ntuple.read_cluster_range(
                                key_nr,
                                start_cluster_idx,
                                stop_cluster_idx,
                                missing_element_padding=n_padding,
                            )
                        )
                elif interpreter == "gpu":
                    raise NotImplementedError(
                        f"Backend {backend} GDS support not implemented."
                    )
                else:
                    raise NotImplementedError(f"Backend {backend} not implemented.")
                dtype_byte = self.ntuple.column_records[key_nr].type
                _fill_container_dict(container_dict, content, key, dtype_byte, dtype)

        arrays = ak.from_buffers(
            form,
            cluster_num_entries,
            container_dict,
            backend="cuda" if interpreter == "gpu" and backend == "cuda" else "cpu",
        )[entry_start - cluster_offset : entry_stop - cluster_offset]

        arrays = ak.to_backend(arrays, backend=backend)
        # no longer needed; save memory
        del container_dict

        # If we constructed some parent fields, we need to get back to the requested field
        if field_path is not None:
            for field in field_path[:-1]:
                arrays = arrays[field]

        expression_context = [(f, None) for f in arrays.fields]

        # TODO: The conversion would be ideally be fully handled by Awkward.
        if library.name in ("np", "pd"):
            numpy_data = {}
            for f in arrays.fields:
                try:
                    numpy_data[f] = _rntuple_to_numpy(arrays[f])
                except Exception:
                    msg = f"Field {f} cannot be converted to NumPy/Pandas"
                    raise ValueError(msg) from None
            if library.name == "pd":
                pd = uproot.extras.pandas()
                pandas_index = uproot.interpretation.library._pandas_basic_index(
                    pd, entry_start, entry_start + len(arrays)
                )
                pandas_dict = {}
                for f, arr in numpy_data.items():
                    if getattr(arr, "ndim", 1) > 1:
                        pandas_dict[f] = list(arr)
                    else:
                        pandas_dict[f] = arr
                try:
                    pandas_data = pd.DataFrame(pandas_dict, index=pandas_index)
                except Exception:
                    msg = "RNTuple fields cannot be converted to NumPy/Pandas"
                    raise ValueError(msg) from None
                arrays = pandas_data
            else:
                arrays = numpy_data

        if how is not None:
            arrays = library.group(arrays, expression_context, how)

        return arrays

    def __array__(self, *args, **kwargs):
        if isinstance(self, uproot.behaviors.RNTuple.RNTuple):
            out = self.arrays(library="np")
        else:
            out = self.array(library="np")
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)

    def iterate(
        self,
        expressions=None,  # TODO: Not implemented yet
        cut=None,  # TODO: Not implemented yet
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        aliases=None,  # TODO: Not implemented yet
        language=uproot.language.python.python_language,  # TODO: Not implemented yet
        entry_start=None,
        entry_stop=None,
        step_size="100 MB",
        decompression_executor=None,  # TODO: Not implemented yet
        library="ak",
        backend="cpu",
        interpreter="cpu",
        ak_add_doc=False,
        how=None,
        report=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        interpretation_executor=None,
        filter_branch=unset,
    ):
        """
        Args:
            expressions (None, str, or list of str): Names of ``RFields`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``RFields``
                selected by the filters are included. (Not implemented yet.)
            cut (None or str): If not None, this expression filters all of the
                ``expressions``. (Not implemented yet.)
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``EFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. If the function
                returns False or None, the ``RField`` is excluded; if the function
                returns True, it is included.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases.
                Uses the ``language`` engine to evaluate. (Not implemented yet.)
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``. (Not implemented yet.)
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.RNTuple.HasFields.num_entries`. If negative,
                count from the end, like a Python slice.
            step_size (int or str): If an integer, the maximum number of entries to
                include in each iteration step; if a string, the maximum memory size
                to include. The string must be a number followed by a memory unit,
                such as "100 MB".
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``RPages``; if None, the
                file's :ref:`uproot.reading.ReadOnlyFile.decompression_executor`
                is used. (Not implemented yet.)
            library (str or :doc:`uproot.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, and ``"pd"`` for Pandas.
            ak_add_doc (bool | dict ): If True and ``library="ak"``, add the RField ``description``
                to the Awkward ``__doc__`` parameter of the array.
                if dict = {key:value} and ``library="ak"``, add the RField ``value`` to the
                Awkward ``key`` parameter of the array.
                A ``value`` of ``"title"`` (the TBranch attribute) reads the RField ``description``.
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
                ``RNTuple``, and global and local entry ranges.
            interpretation_executor (None): This argument is not used and is only included for now
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Iterates through contiguous chunks of entries from the ``RNTuple``.

        For example:

        .. code-block:: python

            >>> for array in rntuple.iterate(filter_name=["x", "y"], step_size=100):
            ...     # each of the following have 100 entries
            ...     array["x"], array["y"]

        See also :ref:`uproot.behaviors.RNTuple.HasFields.arrays` to read
        everything in a single step, without iteration.

        See also :doc:`uproot.behaviors.TBranch.iterate` to iterate over many files.
        """
        # This temporarily provides basic functionality while expressions are properly implemented
        if expressions is not None:
            if filter_name == no_filter:
                filter_name = expressions
            else:
                raise ValueError(
                    "Expressions are not supported yet. They are currently equivalent to filter_name."
                )

        entry_start, entry_stop = (
            uproot.behaviors.TBranch._regularize_entries_start_stop(
                self.ntuple.num_entries, entry_start, entry_stop
            )
        )

        akform, _ = self.to_akform(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
            ak_add_doc=ak_add_doc,
        )

        step_size = _regularize_step_size(
            self.ntuple, akform, step_size, entry_start, entry_stop
        )
        # TODO: This can be done more efficiently
        for start in range(entry_start, entry_stop, step_size):
            arrays = self.arrays(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                entry_start=start,
                entry_stop=min(start + step_size, entry_stop),
                library=library,
                backend=backend,
                interpreter=interpreter,
                ak_add_doc=ak_add_doc,
                how=how,
                filter_branch=filter_branch,
            )
            if report:
                sub_entry_stop = min(start + step_size, entry_stop)
                yield (
                    arrays,
                    uproot.behaviors.TBranch.Report(self, start, sub_entry_stop),
                )
            else:
                yield arrays

    def keys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the names of the subfields as a list of strings.
        """
        return list(
            self.iterkeys(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                full_paths=full_paths,
                ignore_duplicates=ignore_duplicates,
                filter_branch=filter_branch,
            )
        )

    def values(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the subfields as a list of
        :doc:`uproot.behaviors.RField.RField`.

        (Note: with ``recursive=False``, this is the same as
        :ref:`uproot.behaviors.TBranch.HasFields.fields`.)
        """
        return list(
            self.itervalues(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                filter_branch=filter_branch,
            )
        )

    def items(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns (name, field) pairs of the subfields as a list of 2-tuples
        of (str, :doc:`uproot.behaviors.RField.RField`).
        """
        return list(
            self.iteritems(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                full_paths=full_paths,
                filter_branch=filter_branch,
            )
        )

    def typenames(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns (name, typename) pairs of the subfields as a dict of
        str \u2192 str.
        """
        return dict(
            self.itertypenames(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_field=filter_field,
                recursive=recursive,
                full_paths=full_paths,
                filter_branch=filter_branch,
            )
        )

    def iterkeys(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.


        Returns the names of the subfields as an iterator over strings.
        """
        for k, _ in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            recursive=recursive,
            full_paths=full_paths,
            ignore_duplicates=ignore_duplicates,
            filter_branch=filter_branch,
        ):
            yield k

    def itervalues(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns the subfields as an iterator over
        :doc:`uproot.behaviors.RField.RField`.

        (Note: with ``recursive=False``, this is the same as
        :ref:`uproot.behaviors.RField.HasFields.fields`.)
        """
        for _, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            recursive=recursive,
            full_paths=False,
            filter_branch=filter_branch,
        ):
            yield v

    def iteritems(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        ignore_duplicates=False,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            ignore_duplicates (bool): If True, return a set of the keys; otherwise, return the full list of keys.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.


        Returns (name, field) pairs of the subfields as an iterator over
        2-tuples of (str, :doc:`uproot.behaviors.RField.RField`).
        """
        if filter_branch is not unset:
            warnings.warn(
                "the filter_branch kwarg should not be used for RNTuples",
                DeprecationWarning,
                stacklevel=1,
            )
            filter_field = filter_branch

        filter_name = uproot._util.regularize_filter(filter_name)
        filter_typename = uproot._util.regularize_filter(filter_typename)
        if filter_field is None:
            filter_field = no_filter
        elif callable(filter_field):
            pass
        else:
            raise TypeError(
                f"filter_field must be None or a function: RField -> bool, not {filter_field!r}"
            )

        keys_set = set()

        for field in self.fields:
            if (
                (
                    filter_name is no_filter
                    or _filter_name_deep(filter_name, self, field)
                )
                and (filter_typename is no_filter or filter_typename(field.typename))
                and (filter_field is no_filter or filter_field(field))
            ):
                if (
                    field.is_anonymous
                    or field.in_variant
                    or (ignore_duplicates and field.name in keys_set)
                ):
                    pass
                else:
                    keys_set.add(field.name)
                    yield field.name, field

            if recursive:
                for k1, v in field.iteritems(
                    recursive=recursive,
                    filter_name=no_filter,
                    filter_typename=filter_typename,
                    filter_field=filter_field,
                    full_paths=full_paths,
                ):
                    k2 = (
                        f"{field.name}.{k1}"
                        if full_paths and not (field.is_anonymous or field.in_variant)
                        else k1
                    )
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
        filter_field=no_filter,
        recursive=True,
        full_paths=True,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields``s by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subfields.
                If False, only return the names of the top fields.
            full_paths (bool): If True, include the full path to each subfield
                with periods (``.``); otherwise, use the descendant's name as
                the output name.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns (name, typename) pairs of the subfields as an iterator over
        2-tuples of (str, str).
        """
        for k, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            recursive=recursive,
            full_paths=full_paths,
            filter_branch=filter_branch,
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
        expressions=None,  # TODO: Not implemented yet
        cut=None,  # TODO: Not implemented yet
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        aliases=None,  # TODO: Not implemented yet
        language=uproot.language.python.python_language,  # TODO: Not implemented yet
        entry_start=None,
        entry_stop=None,
        # For compatibility reasons we also accepts kwargs meant for TTrees
        filter_branch=unset,
    ):
        """
        Args:
            memory_size (int or str): An integer is interpreted as a number of
                bytes and a string must be a number followed by a unit, such as
                "100 MB".
            expressions (None, str, or list of str): Names of ``RFields`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``RFields``
                selected by the filters are included. (Not implemented yet.)
            cut (None or str): If not None, this expression filters all of the
                ``expressions``. (Not implemented yet.)
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RField.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RField.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases.
                Uses the ``language`` engine to evaluate.
            language (:doc:`uproot.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :ref:`uproot.behaviors.RNTuple.HasFields.num_entries`. If negative,
                count from the end, like a Python slice.
            filter_branch (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool): An alias for ``filter_field`` included
                for compatibility with software that was used for :doc:`uproot.behaviors.TBranch.TBranch`. This argument should not be used
                and will be removed in a future version.

        Returns an *approximate* step size as a number of entries to read
        a given ``memory_size`` in each step.

        This method does not actually read the ``RField`` data or compute any
        expressions to arrive at its estimate. It only uses metadata from the
        already-loaded ``RNTuple``; it only needs ``language`` to parse the
        expressions, not to evaluate them.

        This is the algorithm that
        :ref:`uproot.behaviors.RNTuple.HasFields.iterate` uses to convert a
        ``step_size`` expressed in memory units into a number of entries.
        """
        # This temporarily provides basic functionality while expressions are properly implemented
        if expressions is not None:
            if filter_name == no_filter:
                filter_name = expressions
            else:
                raise ValueError(
                    "Expressions are not supported yet. They are currently equivalent to filter_name."
                )

        target_num_bytes = uproot._util.memory_size(memory_size)

        entry_start, entry_stop = (
            uproot.behaviors.TBranch._regularize_entries_start_stop(
                self.ntuple.num_entries, entry_start, entry_stop
            )
        )

        akform, _ = self.to_akform(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_field=filter_field,
            filter_branch=filter_branch,
        )

        if len(akform.contents) == 0:
            return

        return _num_entries_for(
            self.ntuple, akform, target_num_bytes, entry_start, entry_stop
        )

    def __getitem__(self, where):
        original_where = where

        if uproot._util.isint(where):
            return self.fields[where]
        elif isinstance(where, str):
            where = uproot._util.ensure_str(where)
            where = where.replace("/", ".").replace("\\", ".")
        else:
            raise TypeError(f"where must be an integer or a string, not {where!r}")

        if where.startswith("."):
            recursive = False
            where = where[1:]
        else:
            recursive = True

        if self._lookup is None:
            self._lookup = {f.name: f for f in self.fields}
        got = self._lookup.get(where)
        if got is not None:
            return got

        if "." in where:
            this = self
            try:
                for piece in where.split("."):
                    if piece != "":
                        this = this[piece]
            except uproot.KeyInFileError:
                raise uproot.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self.ntuple.parent._file.file_path,
                    object_path=self.path,
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
                    file_path=self.ntuple.parent._file.file_path,
                    object_path=self.path,
                )

        else:
            raise uproot.KeyInFileError(
                original_where,
                keys=self.keys(recursive=recursive),
                file_path=self.ntuple.parent._file.file_path,
                object_path=self.path,
            )

    def __iter__(self):
        yield from self.fields

    def __len__(self):
        return len(self.fields)

    def show(
        self,
        *,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_field=no_filter,
        recursive=True,
        max_width=80,
        stream=sys.stdout,
        **kwargs,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``RFields`` by type.
            filter_field (None or function of :doc:`uproot.models.RNTuple.RField` \u2192 bool, or None): A
                filter to select ``RFields`` using the full
                :doc:`uproot.models.RNTuple.RField` object. The ``RField`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, recursively descend into subfields.
            max_width (int): Maximum number of characters to display in a line.
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display the ``RFields``.

        For example,

        .. code-block::

            >>> my_ntuple.show()
            my_ntuple (ROOT::RNTuple)
            Description: The description of the ntuple
            ├─ my_int (std::int64_t)
            │  Description: The description of the field
            ├─ jagged_list (std::vector<std::int64_t>)
            ├─ nested_list (std::vector<std::vector<std::int64_t>>)
            ├─ struct (MyStruct)
            │  ├─ x (std::int64_t)
            │  └─ y (std::int64_t)
            └─ other_struct (OtherStruct)
                ├─ a (SubStruct)
                │  Description: The description of the subfield
                │  ├─ x (std::int64_t)
                │  └─ y (std::int64_t)
                └─ b (std::int64_t)
        """
        elbow = "└─ "
        pipe = "│  "
        tee = "├─ "
        blank = "   "

        def recursive_show(field, header="", first=True, last=True, recursive=True):
            outstr = f"""{header}{"" if first else (elbow if last else tee)}{field.name} ({"ROOT::RNTuple" if isinstance(field, uproot.behaviors.RNTuple.RNTuple) else field.typename})"""
            stream.write(outstr[:max_width] + "\n")
            if field.description != "":
                outstr = f"""{header}{"" if first else (blank if last else pipe)}Description: {field.description}"""
                stream.write(outstr[:max_width] + "\n")
            if len(field) > 0 and (recursive or first):
                subfields = list(
                    field.itervalues(
                        filter_name=filter_name,
                        filter_typename=filter_typename,
                        filter_field=filter_field,
                        recursive=False,
                    )
                )
                for i, subfield in enumerate(subfields):
                    recursive_show(
                        subfield,
                        header=f"{header}{'' if first else (blank if last else pipe)}",
                        first=False,
                        last=i == len(subfields) - 1,
                        recursive=recursive,
                    )

        recursive_show(self, recursive=recursive)

    @property
    def source(self) -> uproot.source.chunk.Source | None:
        """Returns the associated source of data for this container, if it exists

        Returns: uproot.source.chunk.Source or None
        """
        if isinstance(self.ntuple._file, uproot.reading.ReadOnlyFile):
            return self.ntuple._file.source
        return None


class RNTuple(HasFields):
    """
    Behaviors for an ``RNTuple``, which mostly consist of array-reading methods.

    Since a :doc:`uproot.behaviors.RNTuple.RNTuple` is a
    :doc:`uproot.behaviors.RNTuple.HasFields`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_rntuple["field"]
        my_rntuple["field"]["subfield"]
        my_rntuple["field.subfield"]
        my_rntuple["field.subfield.subsubfield"]
        my_rntuple["field/subfield/subsubfield"]
        my_rntuple["field\\subfield\\subsubfield"]
    """

    def __repr__(self):
        if len(self) == 0:
            return f"<{self.classname} {self.name!r} at 0x{id(self):012x}>"
        else:
            return f"<{self.classname} {self.name!r} ({len(self)} fields) at 0x{id(self):012x}>"

    @property
    def name(self):
        """
        Name of the ``RNTuple``.
        """
        return self.header.ntuple_name

    @property
    def description(self):
        """
        Description of the ``RNTuple``.
        """
        return self.header.ntuple_description

    @property
    def object_path(self):
        """
        Object path of the ``RNTuple``.
        """
        return self.parent.object_path

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``RNTuple`` in its path, to use as
        part of object and array cache keys.
        """
        return f"{self.parent.cache_key}{self.name};{self.parent.fCycle}"


def _filter_name_deep(filter_name, hasfields, field):
    shallow = name = field.name
    if filter_name(name):
        return True
    while field is not hasfields:
        field = field.parent
        # the spelling ``keys()`` shows: a collection's anonymous item field adds no segment
        if field is not hasfields and not (field.is_anonymous or field.in_variant):
            name = field.name + "." + name
    if name != shallow and filter_name(name):
        return True
    return filter_name("." + name)


def _with_subfield_keys(hasfields, keys):
    """
    Returns ``keys`` extended with the keys of all subfields of the fields in ``keys``,
    so that selecting a field selects its whole subtree.
    """
    selected = set(keys)
    if len(selected) == 0:
        return keys
    out = list(keys)
    for key in hasfields.keys():
        if key in selected:
            continue
        pos = key.find(".")
        while pos != -1:
            if key[:pos] in selected:
                out.append(key)
                break
            pos = key.find(".", pos + 1)
    return out


def _filter_field_by_key(hasfields, keys):
    """
    Returns a ``filter_field`` that selects exactly the fields that ``hasfields.keys()``
    lists as ``keys``. Unlike passing ``keys`` as a ``filter_name``, a key doesn't select
    the fields that only share their name with it, such as ``rec.x`` for the key ``x``.
    """
    keys = set(keys)
    field_ids = {field.field_id for key, field in hasfields.iteritems() if key in keys}
    return lambda field: field.field_id in field_ids


def _get_recursive(hasfields, where):
    if hasfields._lookup is None:
        hasfields._lookup = {f.name: f for f in hasfields.fields}
    got = hasfields._lookup.get(where)
    if got is not None:
        return got
    for field in hasfields.fields:
        got = _get_recursive(field, where)
        if got is not None:
            return got
    else:
        return None


def _cluster_range(clusters, entry_start, entry_stop):
    """
    Returns ``(start_cluster_idx, stop_cluster_idx, cluster_offset)``: the range of
    clusters that hold the entries from ``entry_start`` to ``entry_stop`` (exclusive),
    and the first entry of the first of those clusters, where the buffers read from
    that range start. An empty entry range gives an empty cluster range.
    """
    cluster_starts = numpy.array([c.num_first_entry for c in clusters])
    start_cluster_idx = int(
        numpy.searchsorted(cluster_starts, entry_start, side="right") - 1
    )
    if entry_stop <= entry_start:
        stop_cluster_idx = start_cluster_idx
    else:
        # a cluster that starts at entry_stop has no entries in the range
        stop_cluster_idx = int(
            numpy.searchsorted(cluster_starts, entry_stop, side="left")
        )
    cluster_offset = (
        int(cluster_starts[start_cluster_idx]) if start_cluster_idx >= 0 else 0
    )
    return start_cluster_idx, stop_cluster_idx, cluster_offset


def _num_entries_for(ntuple, akform, target_num_bytes, entry_start, entry_stop):
    # a field or a set of fields has the cluster metadata on its RNTuple
    ntuple = ntuple.ntuple
    clusters = ntuple.cluster_summaries
    start_cluster_idx, stop_cluster_idx, _ = _cluster_range(
        clusters, entry_start, entry_stop
    )

    target_cols = []
    _recursive_find(akform, target_cols)

    total_bytes = 0
    for key in target_cols:
        if "column" in key:
            key_nr = int(key.split("-")[1])
            n_entries, _, dt = ntuple._expected_array_length_starts_dtype(
                key_nr, start_cluster_idx, stop_cluster_idx
            )
            total_bytes += n_entries * dt.itemsize

    # the bytes are those of whole clusters, so they are divided over the entries of
    # those clusters rather than over the entries in the range
    total_entries = sum(
        c.num_entries for c in clusters[start_cluster_idx:stop_cluster_idx]
    )
    if total_bytes == 0:
        num_entries = 0
    else:
        num_entries = round(target_num_bytes * total_entries / total_bytes)
    if num_entries <= 0:
        return 1
    return num_entries


def _regularize_step_size(ntuple, akform, step_size, entry_start, entry_stop):
    uproot._util.check_step_size(step_size)
    if uproot._util.isint(step_size):
        return step_size
    target_num_bytes = uproot._util.memory_size(
        step_size,
        "number of entries or memory size string with units "
        f"(such as '100 MB') required, not {step_size!r}",
    )
    return _num_entries_for(ntuple, akform, target_num_bytes, entry_start, entry_stop)


def _recursive_find(form, res):

    if hasattr(form, "form_key") and form.form_key not in res:
        res.append(form.form_key)
    if hasattr(form, "contents"):
        for c in form.contents:
            _recursive_find(c, res)
    if hasattr(form, "content") and issubclass(type(form.content), ak.forms.Form):
        _recursive_find(form.content, res)


def _cupy_insert(arr, obj, value):
    # obj is assumed to be sorted
    # both arr and obj are assumed to be flat arrays
    cupy = uproot.extras.cupy()
    out_size = arr.size + obj.size
    out = cupy.empty(out_size, dtype=arr.dtype)
    src_i = 0
    dst_i = 0
    for idx in obj.get():
        n = idx - src_i
        if n > 0:
            out[dst_i : dst_i + n] = arr[src_i : src_i + n]
            dst_i += n
            src_i += n
        out[dst_i] = value
        dst_i += 1
    out[dst_i:] = arr[src_i:]
    return out


def _fill_container_dict(container_dict, content, key, dtype_byte, dtype):
    Numpy = ak._nplikes.numpy.Numpy

    if isinstance(content, tuple):
        # Virtual arrays not yet implemented for GPU
        array_library_string = "numpy"
        virtual = True
        length = int(content[0])
        raw_generator = content[1]
    else:
        virtual = False
        array_library_string = uproot._util.get_array_library(content)
        length = len(content)

        def raw_generator():
            return content

    if virtual:
        from packaging.version import Version

        if Version(ak.__version__) < Version("2.8.11"):
            raise ImportError("Virtual arrays require Awkward version 2.8.11 or later")
        VirtualNDArray = ak._nplikes.virtual.VirtualNDArray

    library = numpy if array_library_string == "numpy" else uproot.extras.cupy()

    if "cardinality" in key:

        def generator():
            materialized = raw_generator()
            materialized = library.diff(materialized)
            return materialized

        if virtual:
            virtual_array = VirtualNDArray(
                Numpy.instance(), shape=(length,), dtype=dtype, generator=generator
            )
            container_dict[f"{key}-data"] = virtual_array
        else:
            container_dict[f"{key}-data"] = generator()

    elif "optional" in key:

        def generator():
            # We need to convert from a ListOffsetArray to an IndexedOptionArray
            materialized = raw_generator()
            diff = library.diff(materialized)
            missing = library.nonzero(diff == 0)[0]
            missing -= library.arange(len(missing), dtype=missing.dtype)
            dtype = "int64" if materialized.dtype == library.int64 else "int32"
            indices = library.arange(len(materialized) - len(missing), dtype=dtype)
            if array_library_string == "numpy":
                indices = numpy.insert(indices, missing, -1)
            else:
                indices = _cupy_insert(indices, missing, -1)
            return indices[:-1]  # We need to delete the last index

        if virtual:
            virtual_array = VirtualNDArray(
                Numpy.instance(), shape=(length - 1,), dtype=dtype, generator=generator
            )
            container_dict[f"{key}-index"] = virtual_array
        else:
            container_dict[f"{key}-index"] = generator()

    elif dtype_byte == uproot.const.rntuple_col_type_to_num_dict["switch"]:

        def tag_generator():
            content = raw_generator()
            return content["tag"].astype(numpy.int8)

        def index_generator():
            content = raw_generator()
            tags = content["tag"].astype(numpy.int8)
            kindex = content["index"]
            # Find invalid variants and adjust buffers accordingly
            invalid = numpy.flatnonzero(tags == 0)
            kindex[invalid] = 0  # Might not be necessary, but safer
            return kindex

        def nones_index_generator():
            return library.array([-1], dtype=numpy.int64)

        if virtual:
            tag_virtual_array = VirtualNDArray(
                Numpy.instance(),
                shape=(length,),
                dtype=numpy.int8,
                generator=tag_generator,
            )
            container_dict[f"{key}-tags"] = tag_virtual_array
            index_virtual_array = VirtualNDArray(
                Numpy.instance(),
                shape=(length,),
                dtype=numpy.int64,
                generator=index_generator,
            )
            container_dict[f"{key}-index"] = index_virtual_array
            nones_index_virtual_array = VirtualNDArray(
                Numpy.instance(),
                shape=(1,),
                dtype=numpy.int64,
                generator=nones_index_generator,
            )
            container_dict["nones-index"] = nones_index_virtual_array
        else:
            container_dict[f"{key}-tags"] = tag_generator()
            container_dict[f"{key}-index"] = index_generator()
            container_dict["nones-index"] = nones_index_generator()
    else:
        if virtual:
            virtual_array = VirtualNDArray(
                Numpy.instance(), shape=(length,), dtype=dtype, generator=raw_generator
            )
            container_dict[f"{key}-data"] = virtual_array
            container_dict[f"{key}-offsets"] = virtual_array
        else:
            container_dict[f"{key}-data"] = content
            container_dict[f"{key}-offsets"] = content


_OPTION_FORMS = (
    ak.forms.IndexedOptionForm,
    ak.forms.ByteMaskedForm,
    ak.forms.BitMaskedForm,
    ak.forms.UnmaskedForm,
)


def _is_string_form(form):
    curr = form
    while isinstance(curr, _OPTION_FORMS):
        curr = curr.content
    if isinstance(curr, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
        if curr.parameters.get("__array__") in ("string", "bytestring"):
            return True
        if (
            isinstance(curr.content, ak.forms.NumpyForm)
            and curr.content.parameters.get("__array__") == "char"
        ):
            return True
    return False


def _form_to_numpy_dtype(form, initial_sizes=()):
    sizes = list(initial_sizes)
    curr = form
    while isinstance(curr, ak.forms.RegularForm):
        sizes.append(curr.size)
        curr = curr.content

    if isinstance(curr, ak.forms.NumpyForm):
        inner_dt = numpy.dtype(curr.primitive)
        shape = tuple(sizes) + curr.inner_shape
        if shape:
            return numpy.dtype((inner_dt, shape))
        return inner_dt

    if isinstance(curr, ak.forms.RecordForm):
        field_dtypes = []
        for field in curr.fields:
            content_form = curr.content(field)
            field_dtypes.append((field, _form_to_numpy_dtype(content_form)))
        record_dtype = numpy.dtype(field_dtypes)
        if sizes:
            return numpy.dtype((record_dtype, tuple(sizes)))
        return record_dtype

    if sizes:
        return numpy.dtype((numpy.dtype(object), tuple(sizes)))
    return numpy.dtype(object)


def _rntuple_to_numpy(data):
    if isinstance(data, ak.contents.Content):
        data = ak.Array(data)
    elif not isinstance(data, ak.Array):
        return numpy.asarray(data)

    form = data.layout.form

    if _is_string_form(form):
        lst = ak.to_list(data)
        out = numpy.empty(len(lst), dtype=object)
        out[:] = lst
        return out

    if isinstance(form, _OPTION_FORMS):
        curr = form
        while isinstance(curr, (ak.forms.RegularForm, *_OPTION_FORMS)):
            curr = curr.content
        if isinstance(curr, ak.forms.NumpyForm):
            return ak.to_numpy(data)
        else:
            if len(data) == 0:
                return numpy.empty(0, dtype=object)
            mask = ak.is_none(data).to_numpy()
            projected = ak.Array(data.layout.project())
            converted_proj = _rntuple_to_numpy(projected)
            out = numpy.full(len(data), None, dtype=object)
            for idx, val in zip(numpy.flatnonzero(~mask), converted_proj, strict=True):
                out[idx] = val
            return out

    if isinstance(form, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
        if len(data) == 0:
            return numpy.empty(0, dtype=object)
        lo = data.layout.to_ListOffsetArray64(False)
        offsets = numpy.asarray(lo.offsets)
        start = offsets[0]
        stop = offsets[-1]
        inner = _rntuple_to_numpy(ak.Array(lo.content[start:stop]))
        splits = numpy.split(inner, offsets[1:-1] - start)
        out = numpy.empty(len(splits), dtype=object)
        out[:] = splits
        return out

    if isinstance(form, ak.forms.RegularForm):
        curr = form
        while isinstance(curr, (ak.forms.RegularForm, *_OPTION_FORMS)):
            curr = curr.content
        if isinstance(curr, ak.forms.NumpyForm):
            return ak.to_numpy(data)
        else:
            if len(data) == 0:
                dt = _form_to_numpy_dtype(form)
                base_dt, inner_shape = uproot.interpretation.numerical._dtype_shape(dt)
                return numpy.empty((0, *inner_shape), dtype=base_dt)
            flat = ak.flatten(data, axis=1)
            flat_out = _rntuple_to_numpy(flat)
            return flat_out.reshape((len(data), form.size, *flat_out.shape[1:]))

    if isinstance(form, ak.forms.RecordForm):
        try:
            return ak.to_numpy(data)
        except Exception:
            pass

        record_dtype = _form_to_numpy_dtype(form)
        out = numpy.empty(len(data), dtype=record_dtype)
        for field in form.fields:
            c = _rntuple_to_numpy(data[field])
            field_dt = record_dtype[field]
            if field_dt.base == numpy.dtype(object):
                if field_dt.shape == () and c.ndim > 1:
                    c_out = numpy.empty(len(c), dtype=object)
                    if numpy.ma.isMaskedArray(c):
                        mask = numpy.ma.getmaskarray(c)
                        for i in range(len(c)):
                            m = mask[i]
                            if m.all():
                                c_out[i] = None
                            elif m.any():
                                row = c.data[i].astype(object)
                                row[m] = None
                                c_out[i] = row
                            else:
                                c_out[i] = c.data[i]
                    else:
                        for i in range(len(c)):
                            c_out[i] = c[i]
                    out[field] = c_out
                elif numpy.ma.isMaskedArray(c):
                    c_obj = c.data.astype(object)
                    c_obj[numpy.ma.getmaskarray(c)] = None
                    out[field] = c_obj
                elif c.dtype != object:
                    out[field] = c.astype(object)
                else:
                    out[field] = c
            else:
                out[field] = c
        return out

    return ak.to_numpy(data)
