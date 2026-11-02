# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module represents external libraries that define "array-like" types so that users can
choose an output format.

The :doc:`uproot.interpretation.library.NumPy` library always works (NumPy is
Uproot's only strict dependency) and outputs NumPy arrays for single arrays
and dict/tuple/list as groups. Objects and jagged arrays are not efficiently
represented, but it provides a zero-dependency least common denominator.

The :doc:`uproot.interpretation.library.Awkward` library is the default and
depends on Awkward Array (``awkward``). It is usually the best option, as it
was designed for Uproot.

The :doc:`uproot.interpretation.library.Pandas` library outputs
``pandas.Series`` for single arrays and ``pandas.DataFrame`` as groups. Objects
are not efficiently represented, but some jagged arrays are encoded as
``pandas.MultiIndex``.

Lazy arrays (:doc:`uproot.behaviors.TBranch.lazy`) can only use the
:doc:`uproot.interpretation.library.Awkward` library.
"""

from __future__ import annotations

import json

import awkward
import numpy

import uproot


def _rename(name, context):
    if context is None or "rename" not in context:
        return name
    else:
        return context["rename"]


class Library:
    """
    Abstract superclass of array-library handlers, for libraries such as NumPy,
    Awkward Array, and Pandas.

    A library is used in the finalization and grouping stages of producing an
    array, converting it from internal representations like
    :doc:`uproot.interpretation.jagged.JaggedArray`,
    :doc:`uproot.interpretation.strings.StringArray`, and
    :doc:`uproot.interpretation.objects.ObjectArray` into the library's
    equivalents. It can also be required for concatenation and other late-stage
    operations on the output arrays.

    Libraries are usually selected by a string name. These names are held in a
    private registry in the :doc:`uproot.interpretation.library` module.
    """

    @property
    def imported(self):
        """
        Attempts to import the library and returns the imported module.
        """
        raise AssertionError

    def empty(self, shape, dtype):
        """
        Args:
            shape (tuple of int): NumPy array ``shape``. (The first item must
                be zero.)
            dtype (``numpy.dtype`` or its constructor argument): NumPy array
                ``dtype``.

        Returns an empty NumPy-like array.
        """
        return numpy.empty(shape, dtype)

    def zeros(self, shape, dtype):
        """
        Args:
            shape (tuple of int): NumPy array ``shape``. (The first item must
                be zero.)
            dtype (``numpy.dtype`` or its constructor argument): NumPy array
                ``dtype``.

        Returns a NumPy-like array of zeros.
        """
        return numpy.zeros(shape, dtype)

    def finalize(
        self, array, branch, interpretation, entry_start, entry_stop, interp_options
    ):
        """
        Args:
            array (array): Internal, temporary, trimmed array. If this is a
                NumPy array, it may be identical to the output array.
            branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch``
                that is represented by this array.
            interpretation (:doc:`uproot.interpretation.Interpretation`): The
                interpretation that produced the ``array``.
            entry_start (int): First entry that is included in the output.
            entry_stop (int): FIrst entry that is excluded (one greater than
                the last entry that is included) in the output.
            interp_options (dict): Flags and other options passed through the
                interpretation process.

        Create a library-appropriate output array for this temporary ``array``.

        This array would represent one ``TBranch`` (i.e. not a "group").
        """
        raise AssertionError

    def group(self, arrays, expression_context, how):
        """
        Args:
            arrays (dict of str \u2192 array): Mapping from names to finalized
                array objets to combine into a group.
            expression_context (list of (str, dict) tuples): Expression strings
                and a dict of metadata about each.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
                For ``library="ak"``, passing ``how="zip"`` applies ``ak.zip``
                to interleave data from compatible branches.

        Combine the finalized ``arrays`` into a library-appropriate group type.
        """
        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict or how is None:
            return {_rename(name, c): arrays[name] for name, c in expression_context}
        else:
            raise TypeError(
                f"for library {self.name}, how must be tuple, list, dict, or None (for "
                "dict)"
            )

    def global_index(self, array, global_offset):
        """
        Args:
            array (array): The library-appropriate array whose global index
                needs adjustment.
            global_offset (int): A number to add to the global index of
                ``array`` to correct it.

        Apply *in-place* corrections to the global index of ``array`` by adding
        ``global_offset``.

        Even though the operation is performed *in-place*, this method returns
        the ``array``.
        """
        return array

    def concatenate(self, all_arrays):
        """
        Args:
            all_arrays (list of arrays): A list of library-appropriate arrays
                that need to be concatenated.

        Returns a concatenated version of ``all_arrays``.
        """
        raise AssertionError

    def __repr__(self):
        return repr(self.name)

    def __eq__(self, other):
        if not isinstance(other, Library):
            return NotImplemented

        return type(_libraries[self.name]) is type(_libraries[other.name])


class NumPy(Library):
    """
    A :doc:`uproot.interpretation.library.Library` that presents ``TBranch``
    data as NumPy arrays. The standard name for this library is ``"np"``.

    The single-``TBranch`` form for this library is a ``numpy.ndarray``. If
    the data are non-numerical, they will be converted into Python objects and
    stored in an array with ``dtype="O"``. This is inefficient, but it is the
    minimal-dependency option for Python.

    The "group" behavior for this library is:

    * ``how=dict`` or ``how=None``: a dict of str \u2192 array, mapping the
      names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since NumPy arrays are not indexed, ``global_index`` has no effect.
    """

    name = "np"

    @property
    def imported(self):
        import numpy

        return numpy

    def finalize(self, array, branch, interpretation, entry_start, entry_stop, options):
        if isinstance(array, uproot.interpretation.jagged.JaggedArray) and isinstance(
            array.content,
            uproot.interpretation.objects.StridedObjectArray,
        ):
            out = numpy.zeros(len(array), dtype=object)
            for i, x in enumerate(array):
                out[i] = numpy.zeros(x.shape, dtype=object)
                for j, y in x.ndenumerate():
                    out[i][j] = y
            return out

        elif isinstance(
            array,
            uproot.interpretation.objects.StridedObjectArray,
        ):
            out = numpy.zeros(array.shape, dtype=object)
            for i, x in array.ndenumerate():
                out[i] = x
            return out

        elif isinstance(
            array,
            (
                uproot.interpretation.jagged.JaggedArray,
                uproot.interpretation.strings.StringArray,
                uproot.interpretation.objects.ObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=object)
            for i, x in enumerate(array):
                out[i] = x
            return out

        else:
            return array

    def concatenate(self, all_arrays):
        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            raise AssertionError(repr(all_arrays[0]))

        to_concatenate = {k: [] for k in keys}
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = {k: numpy.concatenate(to_concatenate[k]) for k in keys}

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


def _strided_to_awkward(awkward, path, interpretation, data):
    contents = []
    names = []
    data = data.reshape(-1)
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = name
            if len(path) != 0:
                p = path + "/" + name
            if isinstance(member, uproot.interpretation.objects.AsStridedObjects):
                contents.append(_strided_to_awkward(awkward, p, member, data))
            else:
                contents.append(
                    awkward.from_numpy(
                        numpy.array(data[p]), regulararray=True, highlevel=False
                    )
                )
            names.append(name)
    parameters = {
        "__record__": uproot.model.classname_decode(interpretation.model.__name__)[0]
    }
    length = len(data) if len(contents) == 0 else None
    out = awkward.contents.RecordArray(contents, names, length, parameters=parameters)
    for dim in reversed(interpretation.inner_shape):
        out = awkward.contents.RegularArray(out, dim)
    return out


def _object_to_awkward_json(form, obj):
    if form["class"] == "NumpyArray":
        return obj

    elif form["class"] == "RecordArray":
        out = {}
        for name, subform in zip(form["fields"], form["contents"], strict=True):
            if not name.startswith("@"):
                if hasattr(obj, "has_member") and obj.has_member(name):
                    out[name] = _object_to_awkward_json(subform, obj.member(name))
                else:
                    out[name] = _object_to_awkward_json(subform, getattr(obj, name))
        return out

    elif form["class"][:15] == "ListOffsetArray":
        if form.get("parameters", {}).get("__array__") == "string":
            return obj

        elif form["content"].get("parameters", {}).get("__array__") == "sorted_map":
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            return [
                (
                    _object_to_awkward_json(key_form, x),
                    _object_to_awkward_json(value_form, y),
                )
                for x, y in obj.items()
            ]

        else:
            subform = form["content"]
            return [_object_to_awkward_json(subform, x) for x in obj]

    elif form["class"] == "RegularArray":
        subform = form["content"]
        return [_object_to_awkward_json(subform, x) for x in obj]

    else:
        raise AssertionError(form["class"])


def _awkward_offsets(awkward, form, array):
    if isinstance(array, awkward.contents.EmptyArray):
        if form["offsets"] == "i32":
            return awkward.index.Index32(numpy.zeros(1, dtype=numpy.int32))
        elif form["offsets"] == "u32":
            return awkward.index.IndexU32(numpy.zeros(1, dtype=numpy.uint32))
        elif form["offsets"] == "i64":
            return awkward.index.Index64(numpy.zeros(1, dtype=numpy.int64))
        else:
            raise AssertionError(form["offsets"])
    else:
        if form["offsets"] == "i32":
            return awkward.index.Index32(
                numpy.asarray(array.offsets, dtype=numpy.int32)
            )
        elif form["offsets"] == "u32":
            return awkward.index.IndexU32(
                numpy.asarray(array.offsets, dtype=numpy.uint32)
            )
        elif form["offsets"] == "i64":
            return awkward.index.Index64(
                numpy.asarray(array.offsets, dtype=numpy.int64)
            )
        else:
            raise AssertionError(form["offsets"])


def _awkward_json_to_array(awkward, form, array):
    if form["class"] == "NumpyArray":
        form = awkward.forms.from_json(json.dumps(form))
        dtype = awkward.types.numpytype.primitive_to_dtype(form.primitive)
        if isinstance(array, awkward.contents.EmptyArray):
            return awkward.contents.NumpyArray(
                numpy.empty(0, dtype=dtype),
                parameters=form.parameters,
            )
        else:
            return awkward.contents.NumpyArray(
                numpy.asarray(array.data, dtype=dtype),
                parameters=form.parameters,
            )

    elif form["class"] == "RecordArray":
        contents = []
        names = []
        for name, subform in zip(form["fields"], form["contents"], strict=True):
            if not name.startswith("@"):
                if isinstance(array, awkward.contents.EmptyArray):
                    contents.append(_awkward_json_to_array(awkward, subform, array))
                else:
                    contents.append(
                        _awkward_json_to_array(awkward, subform, array[name])
                    )
                names.append(name)
        length = len(array) if len(contents) == 0 else None
        return awkward.contents.RecordArray(
            contents, names, length, parameters=form.get("parameters", {})
        )

    elif form["class"][:15] == "ListOffsetArray":
        if form.get("parameters", {}).get("__array__") == "string":
            if isinstance(array, awkward.contents.EmptyArray):
                content = awkward.contents.NumpyArray(
                    numpy.empty(0, dtype=numpy.uint8),
                    parameters=form["content"].get("parameters", {}),
                )
                return awkward.contents.ListOffsetArray(
                    awkward.index.Index64(numpy.array([0], dtype=numpy.uint8)),
                    content,
                    parameters=form.get("parameters", {}),
                )
            else:
                content = _awkward_json_to_array(
                    awkward, form["content"], array.content
                )
                return type(array)(
                    array.offsets, content, parameters=form.get("parameters", {})
                )

        elif form["content"].get("parameters", {}).get("__array__") == "sorted_map":
            offsets = _awkward_offsets(awkward, form, array)
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            if isinstance(array, awkward.contents.EmptyArray):
                keys = _awkward_json_to_array(awkward, key_form, array)
                values = _awkward_json_to_array(awkward, value_form, array)
                content = awkward.contents.RecordArray(
                    (keys, values),
                    None,
                    0,
                    parameters=form["content"].get("parameters", {}),
                )
            else:
                keys = _awkward_json_to_array(awkward, key_form, array.content["0"])
                values = _awkward_json_to_array(awkward, value_form, array.content["1"])
                length = len(array.content) if len(keys) == 0 else None
                content = awkward.contents.RecordArray(
                    (keys, values),
                    None,
                    length,
                    parameters=form["content"].get("parameters", {}),
                )
            cls = uproot._util._content_cls_from_name(awkward, form["class"])
            return cls(offsets, content, parameters=form.get("parameters", {}))

        else:
            offsets = _awkward_offsets(awkward, form, array)
            if isinstance(array, awkward.contents.EmptyArray):
                content = _awkward_json_to_array(awkward, form["content"], array)
            else:
                content = _awkward_json_to_array(
                    awkward, form["content"], array.content
                )
            cls = uproot._util._content_cls_from_name(awkward, form["class"])
            return cls(offsets, content, parameters=form.get("parameters", {}))

    elif form["class"] == "RegularArray":
        if isinstance(array, awkward.contents.EmptyArray):
            content = _awkward_json_to_array(awkward, form["content"], array)
        else:
            content = _awkward_json_to_array(awkward, form["content"], array.content)
        return awkward.contents.RegularArray(
            content, form["size"], parameters=form.get("parameters", {})
        )

    else:
        raise AssertionError(form["class"])


def _awkward_add_doc(awkward, array, branch, ak_add_doc):
    for key, value in uproot._util.ak_add_doc_parameters(branch, ak_add_doc).items():
        array = awkward.with_parameter(array, key, value)
    return array


def _object_to_awkward_array(awkward, form, array):
    unlabeled = awkward.from_iter(
        (_object_to_awkward_json(form, x) for x in array),
        highlevel=False,
    )
    return awkward.Array(_awkward_json_to_array(awkward, form, unlabeled))


class Awkward(Library):
    """
    A :doc:`uproot.interpretation.library.Library` that presents ``TBranch``
    data as Awkward Arrays. The standard name for this library is ``"ak"``.

    This is the default for all functions that require a
    :doc:`uproot.interpretation.library.Library`, though Uproot does not
    explicitly depend on Awkward Array. If you are confronted with a message
    that Awkward Array is not installed, either install ``awkward`` or
    select another library (likely :doc:`uproot.interpretation.library.NumPy`).

    Both the single-``TBranch`` and "group" forms for this library are
    ``ak.Array``, though groups are always arrays of records. Awkward Array
    was originally developed for Uproot, so the data structures are usually
    optimial for Uproot data.

    The "group" behavior for this library is:

    * ``how=None``: an array of Awkward records.
    * ``how=dict``: a dict of str \u2192 array, mapping the names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since Awkward arrays are not indexed, ``global_index`` has no effect.
    """

    name = "ak"

    @property
    def imported(self):
        import awkward

        return awkward

    def finalize(self, array, branch, interpretation, entry_start, entry_stop, options):
        awkward = self.imported

        ak_add_doc = options.get("ak_add_doc", False)

        if isinstance(array, awkward.contents.Content):
            return _awkward_add_doc(awkward, awkward.Array(array), branch, ak_add_doc)

        elif isinstance(array, uproot.interpretation.objects.StridedObjectArray):
            return _awkward_add_doc(
                awkward,
                awkward.Array(
                    _strided_to_awkward(awkward, "", array.interpretation, array.array)
                ),
                branch,
                ak_add_doc,
            )

        elif isinstance(array, uproot.interpretation.jagged.JaggedArray) and isinstance(
            array.content, uproot.interpretation.objects.StridedObjectArray
        ):
            content = _strided_to_awkward(
                awkward, "", array.content.interpretation, array.content.array
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward.index.Index32(array.offsets)
                layout = awkward.contents.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward.index.Index64(array.offsets)
                layout = awkward.contents.ListOffsetArray(offsets, content)
            return _awkward_add_doc(awkward, awkward.Array(layout), branch, ak_add_doc)

        elif isinstance(array, uproot.interpretation.jagged.JaggedArray):
            content = awkward.from_numpy(
                array.content, regulararray=True, highlevel=False
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward.index.Index32(array.offsets)
                layout = awkward.contents.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward.index.Index64(array.offsets)
                layout = awkward.contents.ListOffsetArray(offsets, content)
            return _awkward_add_doc(awkward, awkward.Array(layout), branch, ak_add_doc)

        elif isinstance(array, uproot.interpretation.strings.StringArray):
            content = awkward.contents.NumpyArray(
                numpy.frombuffer(array.content, dtype=numpy.dtype(numpy.uint8)),
                parameters={"__array__": "char"},
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward.index.Index32(array.offsets)
                layout = awkward.contents.ListOffsetArray32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.uint32):
                offsets = awkward.index.IndexU32(array.offsets)
                layout = awkward.contents.ListOffsetArrayU32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.int64):
                offsets = awkward.index.Index64(array.offsets)
                layout = awkward.contents.ListOffsetArray(
                    offsets, content, parameters={"__array__": "string"}
                )
            else:
                raise AssertionError(repr(array.offsets.dtype))
            return _awkward_add_doc(awkward, awkward.Array(layout), branch, ak_add_doc)

        elif isinstance(interpretation, uproot.interpretation.objects.AsObjects):
            form = json.loads(
                interpretation.awkward_form(interpretation.branch.file).to_json()
            )
            return _awkward_add_doc(
                awkward,
                _object_to_awkward_array(awkward, form, array),
                branch,
                ak_add_doc,
            )

        elif array.dtype.names is not None:
            length, shape = array.shape[0], array.shape[1:]
            array = array.reshape(-1)
            contents = []
            for name in array.dtype.names:
                contents.append(
                    awkward.from_numpy(
                        numpy.array(array[name]), regulararray=True, highlevel=False
                    )
                )
            if len(contents) != 0:
                length = None
            out = awkward.contents.RecordArray(contents, array.dtype.names, length)
            for size in shape[::-1]:
                out = awkward.contents.RegularArray(out, size)
            return _awkward_add_doc(awkward, awkward.Array(out), branch, ak_add_doc)

        else:
            return _awkward_add_doc(
                awkward,
                awkward.from_numpy(array, regulararray=True),
                branch,
                ak_add_doc,
            )

    def group(self, arrays, expression_context, how):
        awkward = self.imported

        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict:
            return {_rename(name, c): arrays[name] for name, c in expression_context}
        elif how is None:
            if len(expression_context) == 0:
                return awkward.Array(
                    awkward.contents.RecordArray([], fields=[], length=0)
                )
            else:
                return awkward.Array(
                    {_rename(name, c): arrays[name] for name, c in expression_context}
                )
        elif how == "zip":
            nonjagged = []
            offsets = []
            jaggeds = []
            renamed_arrays = {}
            for name, context in expression_context:
                array = renamed_arrays[_rename(name, context)] = arrays[name]
                if context["is_jagged"]:
                    if (
                        isinstance(array.layout, awkward.contents.ListArray)
                        or array.layout.offsets[0] != 0
                    ):
                        array_layout = array.layout.to_ListOffsetArray64(True)
                    else:
                        array_layout = array.layout
                    if len(offsets) == 0:
                        offsets.append(array_layout.offsets)
                        jaggeds.append([_rename(name, context)])
                    else:
                        for o, j in zip(offsets, jaggeds, strict=True):
                            if numpy.array_equal(array_layout.offsets, o):
                                j.append(_rename(name, context))
                                break
                        else:
                            offsets.append(array_layout.offsets)
                            jaggeds.append([_rename(name, context)])
                else:
                    nonjagged.append(_rename(name, context))

            out = None
            if len(nonjagged) != 0:
                if len(nonjagged) == 0:
                    out = awkward.Array(
                        awkward.contents.RecordArray([], fields=[], length=0)
                    )
                else:
                    out = awkward.Array(
                        {name: renamed_arrays[name] for name in nonjagged},
                    )
            for number, jagged in enumerate(jaggeds):
                cut = len(jagged[0])
                for name in jagged:
                    cut = min(cut, len(name))
                    while cut > 0 and (
                        name[:cut] != jagged[0][:cut]
                        or name[cut - 1] not in ("_", ".", "/")
                    ):
                        cut -= 1
                    if cut == 0:
                        break
                if (
                    out is not None
                    and cut != 0
                    and jagged[0][:cut].strip("_./") in awkward.fields(out)
                ):
                    cut = 0
                if cut == 0:
                    common = f"jagged{number}"
                    if len(jagged) == 0:
                        subarray = awkward.Array(
                            awkward.contents.RecordArray([], fields=[], length=0)
                        )
                    else:
                        subarray = awkward.zip(
                            {name: renamed_arrays[name] for name in jagged}
                        )
                else:
                    common = jagged[0][:cut].strip("_./")
                    if len(jagged) == 0:
                        subarray = awkward.Array(
                            awkward.contents.RecordArray([], fields=[], length=0)
                        )
                    else:
                        subarray = awkward.zip(
                            {
                                name[cut:].strip("_./"): renamed_arrays[name]
                                for name in jagged
                            }
                        )
                if out is None:
                    out = awkward.Array({common: subarray})
                else:
                    out = awkward.with_field(out, subarray, common)

            return out
        else:
            raise TypeError(
                f'for library {self.name}, how must be tuple, list, dict, "zip" for '
                "a record array with jagged arrays zipped, if possible, or "
                "None, for an unzipped record array"
            )

    def concatenate(self, all_arrays):
        awkward = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            return awkward.concatenate(all_arrays)

        to_concatenate = {k: [] for k in keys}
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = {k: awkward.concatenate(to_concatenate[k]) for k in keys}

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


def _is_pandas_rangeindex(pandas, index):
    if hasattr(pandas, "RangeIndex") and isinstance(index, pandas.RangeIndex):
        return True
    elif hasattr(index, "is_integer") and index.is_integer():
        return True
    elif uproot._util.parse_version(pandas.__version__) < uproot._util.parse_version(
        "1.4.0"
    ) and isinstance(index, pandas.Int64Index):
        return True
    else:
        return False


def _strided_to_pandas(path, interpretation, data, arrays, columns):
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = (*path, name)
            if isinstance(member, uproot.interpretation.objects.AsStridedObjects):
                _strided_to_pandas(p, member, data, arrays, columns)
            else:
                arrays.append(data["/".join(p)])
                columns.append(p)


def _pandas_basic_index(pandas, entry_start, entry_stop):
    if hasattr(pandas, "RangeIndex"):
        return pandas.RangeIndex(entry_start, entry_stop)
    else:
        return pandas.Int64Index(range(entry_start, entry_stop))


def _pandas_only_series(pandas, original_arrays, expression_context):
    arrays = {}
    names = []
    for name, context in expression_context:
        arrays[_rename(name, context)] = original_arrays[name]
        names.append(_rename(name, context))
    return arrays, names


def _process_array_for_pandas(
    array,
    finalize,
    interpretation,
    branch=None,
    entry_start=None,
    entry_stop=None,
    options=None,
    form=None,
):
    if (
        isinstance(array, numpy.ndarray)
        and array.dtype.names is None
        and len(array.shape) == 1
        and array.dtype != numpy.dtype(object)
    ):
        if finalize:
            return array
        else:
            return awkward.Array(array)
    else:
        try:
            interpretation.awkward_form(None)
        except uproot.interpretation.objects.CannotBeAwkward:
            pass
        else:
            if finalize:
                array = _libraries[Awkward.name].finalize(
                    array, branch, interpretation, entry_start, entry_stop, options
                )
                if isinstance(
                    array.type.content, awkward.types.NumpyType
                ) and array.layout.minmax_depth == (1, 1):
                    array = array.to_numpy()
                else:
                    array = uproot.extras.awkward_pandas().AwkwardExtensionArray(array)
            else:
                array = _object_to_awkward_array(awkward, form, array)
        return array


class Pandas(Library):
    """
    A :doc:`uproot.interpretation.library.Library` that presents ``TBranch``
    data as Pandas Series and DataFrames. The standard name for this library is
    ``"pd"``.

    The single-``TBranch`` (with a single ``TLeaf``) form for this library is
    ``pandas.Series``, and the "group" form is ``pandas.DataFrame``.

    The "group" behavior for this library is:

    * ``how=None`` or a string: passed to ``pandas.merge`` as its ``how``
      parameter, which would be relevant if jagged arrays with different
      multiplicity are requested.
    * ``how=dict``: a dict of str \u2192 array, mapping the names to
      ``pandas.Series``.
    * ``how=tuple``: a tuple of ``pandas.Series``, in the order requested.
      (Names are assigned to the ``pandas.Series``.)
    * ``how=list``: a list of ``pandas.Series``, in the order requested.
      (Names are assigned to the ``pandas.Series``.)

    Pandas Series and DataFrames are indexed, so ``global_index`` adjusts them.
    """

    name = "pd"

    @property
    def imported(self):
        return uproot.extras.pandas()

    def finalize(self, array, branch, interpretation, entry_start, entry_stop, options):
        pandas = self.imported
        index = _pandas_basic_index(pandas, entry_start, entry_stop)

        array = _process_array_for_pandas(
            array, True, interpretation, branch, entry_start, entry_stop, options
        )

        return pandas.Series(array, index=index)

    def group(self, arrays, expression_context, how):
        pandas = self.imported

        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)

        elif how is list:
            return [arrays[name] for name, _ in expression_context]

        elif how is dict:
            return {_rename(name, c): arrays[name] for name, c in expression_context}

        elif isinstance(how, str) or how is None:
            arrays, names = _pandas_only_series(pandas, arrays, expression_context)
            if len(arrays) == 0:
                return pandas.DataFrame()
            else:
                arrays = {
                    k: (
                        v
                        if isinstance(v, (pandas.Series, pandas.DataFrame))
                        else pandas.Series(v, name=k)
                    )
                    for k, v in arrays.items()
                }
                out = pandas.concat(arrays, axis=1, ignore_index=True)
                out.columns = names
                return out

        else:
            raise TypeError(
                f"for library {self.name}, how must be tuple, list, dict, str (for "
                "pandas.merge's 'how' parameter, or None (for one or more"
                "DataFrames without merging)"
            )

    def global_index(self, arrays, global_offset):
        if isinstance(arrays, tuple):
            return tuple(self.global_index(x, global_offset) for x in arrays)
        elif isinstance(arrays, list):
            return [self.global_index(x, global_offset) for x in arrays]
        elif isinstance(arrays, dict):
            return {
                name: self.global_index(x, global_offset) for name, x in arrays.items()
            }

        # build a new index rather than shifting the old one in place: its
        # values may be read-only, or shared with other Series or the caller
        # (a RangeIndex stays a RangeIndex)
        arrays.index = arrays.index + global_offset
        return arrays

    def concatenate(self, all_arrays):
        pandas = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            return pandas.concat(all_arrays)

        to_concatenate = {k: [] for k in keys}
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = {k: pandas.concat(to_concatenate[k]) for k in keys}

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


_libraries = {
    NumPy.name: NumPy(),
    Awkward.name: Awkward(),
    Pandas.name: Pandas(),
}

_libraries["numpy"] = _libraries[NumPy.name]
_libraries["Numpy"] = _libraries[NumPy.name]
_libraries["NumPy"] = _libraries[NumPy.name]
_libraries["NUMPY"] = _libraries[NumPy.name]

_libraries["awkward1"] = _libraries[Awkward.name]
_libraries["Awkward1"] = _libraries[Awkward.name]
_libraries["AWKWARD1"] = _libraries[Awkward.name]
_libraries["awkward"] = _libraries[Awkward.name]
_libraries["Awkward"] = _libraries[Awkward.name]
_libraries["AWKWARD"] = _libraries[Awkward.name]

_libraries["pandas"] = _libraries[Pandas.name]
_libraries["Pandas"] = _libraries[Pandas.name]
_libraries["PANDAS"] = _libraries[Pandas.name]


def _regularize_library(library):
    if isinstance(library, Library):
        if library.name in _libraries:
            return _libraries[library.name]
        else:
            raise ValueError(
                f"library {type(library).__name__} ({library.name!r}) cannot be used in this function"
            )

    elif isinstance(library, type) and issubclass(library, Library):
        if library().name in _libraries:
            return _libraries[library().name]
        else:
            raise ValueError(
                f"library {library.__name__} ({library().name!r}) cannot be used in this function"
            )

    else:
        try:
            return _libraries[library]
        except KeyError as err:
            raise ValueError(
                f"""library {library!r} not recognized (for this function); """
                """try "np" (NumPy), "ak" (Awkward Array), or "pd" (Pandas) """
                """instead"""
            ) from err
