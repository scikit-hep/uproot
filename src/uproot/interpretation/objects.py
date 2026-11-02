# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines an :doc:`uproot.interpretation.Interpretation` and
temporary array for object data (Python objects or Awkward Array data structures).

The :doc:`uproot.interpretation.objects.AsObjects` describes fully generic
objects using a :doc:`uproot.model.Model` (or a :doc:`uproot.containers.AsContainer`).
These objects require a non-vectorized loop to deserialize.

The :doc:`uproot.interpretation.objects.AsStridedObjects` describes fixed-width
objects that can be described as a ``numpy.dtype``. These objects can be
interpreted as a single, vectorized cast, and are therefore much faster to
deserialize.

The :doc:`uproot.interpretation.objects.ObjectArray` and
:doc:`uproot.interpretation.objects.StridedObjectArray` classes only hold data
while an array is being built from ``TBaskets``. Its final form is determined
by the :doc:`uproot.interpretation.library.Library`.
"""

from __future__ import annotations

import contextlib
import json
import threading

import awkward
import numpy

import uproot
import uproot._awkwardforth
from uproot.interpretation.known_forth import known_forth_of


class AsObjects(uproot.interpretation.Interpretation):
    """
    Args:
        model (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`): The
            full Uproot deserialization model for the data.
        branch (None or :doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch``
            from which the data are drawn.

    Integerpretation for arrays of any kind of data that might reside in a
    ROOT ``TTree``. This interpretation prescribes the full (slow)
    deserialization process.

    :ref:`uproot.interpretation.objects.AsObjects.simplify` attempts to
    replace this interpretation with a faster-to-read equivalent, but not all
    data types can be simplified.

    :doc:`uproot.interpretation.known_forth` defines forth code and forms for
    special cases that will be picked up here as well
    """

    def __init__(self, model, branch=None):
        self._model = model
        self._branch = branch
        self._forth = True
        known_forth = known_forth_of(self._model)
        if known_forth is not None:
            self._complete_forth_code = known_forth.forth_code
            self._form = known_forth.awkward_form
        else:
            self._complete_forth_code = None
            self._form = None
        self._forth_lock = threading.Lock()

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`).
        """
        return self._model

    @property
    def branch(self):
        """
        The ``TBranch`` from which the data are drawn. May be None.
        """
        return self._branch

    def __repr__(self):
        if isinstance(self._model, type):
            model = self._model.__name__
        else:
            model = repr(self._model)
        return f"AsObjects({model})"

    def __eq__(self, other):
        return isinstance(other, AsObjects) and self._model == other._model

    def __getstate__(self):
        return {
            k: (
                uproot.model._LockPlaceholder()
                if isinstance(v, uproot.model._LockPlaceholder.lock_type)
                else v
            )
            for k, v in self.__dict__.items()
        }

    def __setstate__(self, state):
        instance_data = {
            k: threading.Lock() if isinstance(v, uproot.model._LockPlaceholder) else v
            for k, v in state.items()
        }
        self.__dict__.update(instance_data)

    @property
    def numpy_dtype(self):
        return numpy.dtype(object)

    @property
    def cache_key(self):
        content_key = uproot.containers._content_cache_key(self._model)
        return f"{type(self).__name__}({content_key})"

    @property
    def typename(self):
        if isinstance(self._model, uproot.containers.AsContainer):
            return self._model.typename
        else:
            return uproot.model.classname_decode(self._model.__name__)[0]

    def awkward_form(
        self,
        file,
        context=None,
        index_format="i64",
        header=False,
        tobject_header=False,
        breadcrumbs=(),
    ):
        if self._form is not None:
            return awkward.forms.from_dict(self._form)

        context = self._make_context(
            context, index_format, header, tobject_header, breadcrumbs
        )
        return self._model.awkward_form(self._branch.file, context)

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        options,
    ):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )
        assert basket.byte_offsets is not None

        if self._forth and (
            isinstance(
                library,
                (
                    uproot.interpretation.library.Awkward,
                    uproot.interpretation.library.Pandas,
                ),
            )
        ):
            output = self.basket_array_forth(
                data,
                byte_offsets,
                basket,
                branch,
                context,
                cursor_offset,
                library,
                options,
            )

        else:
            output = ObjectArray(
                self._model, branch, context, byte_offsets, data, cursor_offset
            ).to_numpy()

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )

        return output

    def basket_array_forth(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        options,
    ):
        import awkward.forth

        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )
        assert basket.byte_offsets is not None

        if getattr(context["forth"], "vm", None) is None:
            # context["forth"] is a threading.local()
            context["forth"].vm = None
            context["forth"].gen = uproot._awkwardforth.ForthGenerator(self)
        else:
            context["forth"].gen = None

        if self._complete_forth_code is None:
            # all threads have to wait until complete_forth_code is ready
            with self._forth_lock:
                # another thread might have concluded that this can't be Forth
                if not self._forth:
                    context["forth"].gen = None
                    context["forth"].vm = None
                    return self.basket_array(
                        data,
                        byte_offsets,
                        basket,
                        branch,
                        context,
                        cursor_offset,
                        library,
                        options,
                    )

                # another thread might have already found the complete Forth code
                elif self._complete_forth_code is None:
                    context = dict(context)
                    context["path"] = ()

                    # this thread tries to do it!
                    try:
                        output = self._discover_forth(
                            data, byte_offsets, branch, context, cursor_offset
                        )
                    except CannotBeForth:
                        self._forth = False
                        context["forth"].gen = None
                        context["forth"].vm = None
                        return self.basket_array(
                            data,
                            byte_offsets,
                            basket,
                            branch,
                            context,
                            cursor_offset,
                            library,
                            options,
                        )

                    # Forth discovery was unsuccessful; return Python-derived
                    # output and maybe another basket will succeed
                    if output is not None:
                        self.hook_after_basket_array(
                            data=data,
                            byte_offsets=byte_offsets,
                            basket=basket,
                            branch=branch,
                            context=context,
                            output=output,
                            cursor_offset=cursor_offset,
                            library=library,
                            options=options,
                        )
                        return output

        # if we didn't return already, either this thread generated the
        # complete_forth_code or another did
        assert self._complete_forth_code is not None

        if context["forth"].vm is None:
            try:
                context["forth"].vm = awkward.forth.ForthMachine64(
                    self._complete_forth_code
                )
            except Exception as err:
                raise type(err)(
                    str(err)
                    + "\n\nForth code generated for this data type:\n\n"
                    + self._complete_forth_code
                ) from err

        # get data using Forth
        temp_data = data[byte_offsets[0] : byte_offsets[-1]]
        context["forth"].vm.begin(
            {
                "stream": numpy.array(temp_data),
                "byteoffsets": numpy.array(byte_offsets[:-1]),
                "bytestops": numpy.array(byte_offsets[1:]),
            }
        )
        context["forth"].vm.stack_push(len(byte_offsets) - 1)
        context["forth"].vm.resume()
        container = context["forth"].vm.outputs
        output = awkward.from_buffers(self._form, len(byte_offsets) - 1, container)

        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )

        return output

    def _assemble_forth(self, forth_obj, node, only_base=False):
        if only_base != node.name.startswith("base-class "):
            return

        forth_obj.final_header.extend(node.header_code)
        forth_obj.final_init.extend(node.init_code)

        if node.name.startswith("read-members "):
            for child in node.children:
                self._assemble_forth(forth_obj, child, only_base=True)
        forth_obj.final_code.extend(node.pre_code)
        for child in node.children:
            self._assemble_forth(forth_obj, child)

        forth_obj.final_code.extend(node.post_code)

    def _any_NULL(self, form):
        # Recursion through form.
        # If NULL is found anywhere, return False as Forth is not fully discovered.
        if form == "NULL":
            return True
        else:
            content = form.get("content")
            if content is not None:
                return self._any_NULL(content)
            contents = form.get("contents")
            if isinstance(contents, list):
                for content in contents:
                    if self._any_NULL(content):
                        return True
            elif isinstance(contents, dict):
                for content in contents.values():
                    if self._any_NULL(content):
                        return True
            return False

    def _debug_forth(self, forth_obj):
        self._assemble_forth(forth_obj, forth_obj.model.children[0])
        expected_form = self._model.awkward_form(
            self._branch.file,
            {
                "index_format": "i64",
                "header": False,
                "tobject_header": False,
                "breadcrumbs": (),
            },
        )

        raise Exception(f"""EXPECTED FORM:
{expected_form}

DISCOVERED FORM:
{json.dumps(forth_obj.model.derive_form(), indent=4)}

FORTH CODE:
input stream
    input byteoffsets
    input bytestops
    {"".join(forth_obj.final_header)}
    {"".join(forth_obj.final_init)}
    0 do
    byteoffsets I-> stack
    stream seek
    {"".join(forth_obj.final_code)}
    loop
    """)

    def _discover_forth(self, data, byte_offsets, branch, context, cursor_offset):
        output = numpy.empty(len(byte_offsets) - 1, dtype=numpy.dtype(object))

        for i in range(len(byte_offsets) - 1):
            chunk = uproot.source.chunk.Chunk.wrap(
                branch.file.source, data[byte_offsets[i] : byte_offsets[i + 1]]
            )
            cursor = uproot.source.cursor.Cursor(
                0, origin=-(byte_offsets[i] + cursor_offset)
            )

            context["forth"].gen.reset_active_node()
            output[i] = self._model.read(
                chunk, cursor, context, branch.file, branch.file.detached, branch
            )

            derived_form = context["forth"].gen.model.derive_form()
            if not self._any_NULL(derived_form):
                context["forth"].prereaddone = True
                self._assemble_forth(
                    context["forth"].gen, context["forth"].gen.model.children[0]
                )
                self._complete_forth_code = f"""input stream
    input byteoffsets
    input bytestops
    {"".join(context["forth"].gen.final_header)}
    {"".join(context["forth"].gen.final_init)}
    0 do
    byteoffsets I-> stack
    stream seek
    {"".join(context["forth"].gen.final_code)}
    loop
    """
                self._form = derived_form

                return None  # we should re-read all the data with Forth

        # Python output when Forth-Generation not successful
        return output

    def final_array(
        self,
        basket_arrays,
        entry_start,
        entry_stop,
        entry_offsets,
        library,
        branch,
        options,
    ):
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            options=options,
        )
        trimmed = []
        start = entry_offsets[0]
        has_any_awkward_types = any(
            uproot._util.from_module(x, "awkward") for x in basket_arrays.values()
        )
        for basket_num, stop in enumerate(entry_offsets[1:]):
            to_append = None
            if start <= entry_start and entry_stop <= stop:
                local_start = entry_start - start
                local_stop = entry_stop - start
                to_append = basket_arrays[basket_num][local_start:local_stop]

            elif start <= entry_start < stop:
                local_start = entry_start - start
                local_stop = stop - start
                to_append = basket_arrays[basket_num][local_start:local_stop]

            elif start < entry_stop <= stop:
                local_start = 0
                local_stop = entry_stop - start
                to_append = basket_arrays[basket_num][local_start:local_stop]

            elif entry_start < stop and start < entry_stop:
                to_append = basket_arrays[basket_num]

            if to_append is not None and has_any_awkward_types:
                if isinstance(library, uproot.interpretation.library.NumPy):
                    trimmed.append(to_append)

                elif isinstance(library, uproot.interpretation.library.Awkward):
                    if isinstance(to_append, numpy.ndarray):
                        trimmed.append(
                            uproot.interpretation.library._object_to_awkward_array(
                                awkward, self._form, to_append
                            )
                        )
                    else:
                        trimmed.append(to_append)

                elif isinstance(  # pragma: no branch  (no other libraries)
                    library, uproot.interpretation.library.Pandas
                ):
                    if isinstance(to_append, numpy.ndarray):
                        trimmed.append(
                            uproot.interpretation.library._process_array_for_pandas(
                                to_append,
                                False,
                                self,
                                form=self._form,
                            )
                        )
                    else:
                        trimmed.append(to_append)

            elif to_append is not None:
                trimmed.append(to_append)

            start = stop

        # If *some* of the baskets are Awkward and *some* are not,
        # convert the ones that are not, individually.
        if has_any_awkward_types and isinstance(
            library,
            (
                uproot.interpretation.library.Awkward,
                uproot.interpretation.library.Pandas,
            ),
        ):
            form = json.loads(self.awkward_form(branch.file).to_json())
            for k, v in basket_arrays.items():
                if not uproot._util.from_module(v, "awkward"):
                    basket_arrays[k] = (
                        uproot.interpretation.library._object_to_awkward_array(
                            awkward, form, v
                        )
                    )

        if len(basket_arrays) == 0:
            output = numpy.array([], dtype=self.numpy_dtype)

        elif all(
            uproot._util.from_module(x, "awkward") for x in basket_arrays.values()
        ) and isinstance(
            library,
            (
                uproot.interpretation.library.Awkward,
                uproot.interpretation.library.Pandas,
            ),
        ):
            output = awkward.concatenate(trimmed, mergebool=False, highlevel=False)
        else:
            output = numpy.concatenate(trimmed)

        self.hook_before_library_finalize(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        output = library.finalize(
            output, branch, self, entry_start, entry_stop, options
        )

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
            options=options,
        )

        return output

    def simplify(self):
        """
        Attempts to replace this :doc:`uproot.interpretation.objects.AsObjects`
        with an interpretation that can be executed more quickly.

        If there isn't a simpler interpretation, then this method returns
        ``self``.
        """
        if self._branch is not None:
            with contextlib.suppress(CannotBeStrided):
                return self._model.strided_interpretation(
                    self._branch.file,
                    header=False,
                    tobject_header=True,
                    breadcrumbs=(),
                    original=self._model,
                )

        if isinstance(self._model, uproot.containers.AsString):
            header_bytes = 0
            if self._model.header:
                header_bytes = 6
            return uproot.interpretation.strings.AsStrings(
                header_bytes,
                self._model.length_bytes,
                self._model.typename,
                original=self._model,
            )

        if isinstance(
            self._model,
            (
                uproot.containers.AsArray,
                uproot.containers.AsRVec,
                uproot.containers.AsVector,
            ),
        ):
            header_bytes = 0
            if (
                isinstance(self._model, uproot.containers.AsArray)
                and self._model.speedbump
            ):
                header_bytes += 1
            if (
                isinstance(
                    self._model, (uproot.containers.AsRVec, uproot.containers.AsVector)
                )
                and self._model.header
            ):
                header_bytes += 10

            if isinstance(self._model.values, numpy.dtype):
                content = uproot.interpretation.numerical.AsDtype(self._model.values)
                return uproot.interpretation.jagged.AsJagged(
                    content, header_bytes, self._model.typename, original=self._model
                )

            if self._branch is not None:
                with contextlib.suppress(CannotBeStrided):
                    content = self._model.values.strided_interpretation(
                        self._branch.file,
                        header=False,
                        tobject_header=True,
                        breadcrumbs=(),
                        original=self._model.values,
                    )
                    if (
                        isinstance(self._model, uproot.containers.AsArray)
                        and self._model.inner_shape
                    ):
                        if not isinstance(
                            content, uproot.interpretation.numerical.AsDtype
                        ):
                            raise RuntimeError("I shouldn't be here")
                        content.reshape(self._model.inner_shape)
                    return uproot.interpretation.jagged.AsJagged(
                        content,
                        header_bytes,
                        self._model.typename,
                        original=self._model,
                    )

        return self


def _unravel_members(members):
    out = []
    for name, member in members:
        if isinstance(member, AsStridedObjects):
            for n, m in _unravel_members(member.members):
                out.append((name + "/" + n, m))
        else:
            out.append((name, member))
    return out


def _strided_awkward_form(awkward, classname, members, file, context):
    contents = {}
    for name, member in members:
        if not context["header"] and name in ("@num_bytes", "@instance_version"):
            pass
        elif not context["tobject_header"] and name in (
            None,
            "@num_bytes",
            "@instance_version",
            "@fUniqueID",
            "@fBits",
            "@pidf",
        ):
            pass
        else:
            if isinstance(member, AsStridedObjects):
                cname = uproot.model.classname_decode(member._model.__name__)[0]
                contents[name] = _strided_awkward_form(
                    awkward, cname, member._members, file, context
                )
            else:
                contents[name] = uproot._util.awkward_form(member, file, context)

    return awkward.forms.RecordForm(
        list(contents.values()),
        list(contents.keys()),
        parameters={"__record__": classname},
    )


class AsStridedObjects(uproot.interpretation.numerical.AsDtype):
    """
    Args:
        model (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`): The
            full Uproot deserialization model for the data.
        members (list of (str, :doc:`uproot.interpretation.Interpretation`) tuples): The
            name and fixed-width interpretation for each member of the objects.
        original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.AsContainer`): If
            this interpretation is derived from
            :ref:`uproot.interpretation.objects.AsObjects.simplify`, this is a
            reminder of the original
            :ref:`uproot.interpretation.objects.AsObjects.model`.

    Interpretation for an array (possibly
    :doc:`uproot.interpretation.jagged.AsJagged`) of fixed-size objects. Since
    the objects have a fixed number of fields with a fixed number of bytes each,
    the whole array (or :ref:`uproot.interpretation.jagged.AsJagged.content`)
    can be interpreted in one vectorized array-cast. Therefore, this
    interpretation is faster than :doc:`uproot.interpretation.objects.AsObjects`
    *when it is possible*.

    Unlike :doc:`uproot.interpretation.numerical.AsDtype` with a
    `structured array <https://numpy.org/doc/stable/user/basics.rec.html>`__,
    the objects in the final array have the methods required by its ``model``.
    If the ``library`` is :doc:`uproot.interpretation.library.NumPy`, these
    are instantiated as Python objects (slow); if
    :doc:`uproot.interpretation.library.Awkward`, they are behaviors passed to
    the Awkward Array's local
    `behavior <https://awkward-array.readthedocs.io/en/latest/ak.behavior.html>`__.
    """

    def __init__(self, model, members, original=None):
        # Skip the leading run of ``(None, None)`` header markers.
        first_value_loc = 0
        for first_value_loc in range(len(members)):
            if members[first_value_loc] != (None, None):
                break

        # Drop any ``(None, None)`` header markers that appear after the value
        # members have begun (e.g. from multiple base classes), keeping the
        # actual members.  ``all_headers_prepended`` is True when the kept
        # members end with a value (headers prepended), and False when a
        # trailing header marker was removed.
        all_headers_prepended = False
        kept_members = []
        for member, value in members[first_value_loc:]:
            if member is None and value is None:
                all_headers_prepended = False
            else:
                all_headers_prepended = True
                kept_members.append((member, value))

        self._model = model
        self._members = kept_members
        self._original = original
        self._all_headers_prepended = all_headers_prepended
        super().__init__(_unravel_members(self._members))

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`).
        """
        return self._model

    @property
    def members(self):
        """
        The name (str) and fixed-width
        :doc:`uproot.interpretation.Interpretation` for each member of the
        objects as a list of 2-tuple pairs.
        """
        return self._members

    @property
    def original(self):
        """
        If not None, this was the original
        :ref:`uproot.interpretation.objects.AsObjects.model` from an
        :doc:`uproot.interpretation.objects.AsObjects` that was simplified
        into this :doc:`uproot.interpretation.objects.AsStridedObjects`.
        """
        return self._original

    def __repr__(self):
        if self.inner_shape:
            return f"AsStridedObjects({self._model.__name__}, {self.inner_shape})"
        else:
            return f"AsStridedObjects({self._model.__name__})"

    def __eq__(self, other):
        return isinstance(other, AsStridedObjects) and self._model == other._model

    @property
    def numpy_dtype(self):
        return numpy.dtype(object)

    @property
    def all_headers_prepended(self):
        return self._all_headers_prepended

    def awkward_form(
        self,
        file,
        context=None,
        index_format="i64",
        header=False,
        tobject_header=False,
        breadcrumbs=(),
    ):
        context = self._make_context(
            context, index_format, header, tobject_header, breadcrumbs
        )
        cname = uproot.model.classname_decode(self._model.__name__)[0]
        form = _strided_awkward_form(awkward, cname, self._members, file, context)
        for dim in reversed(self.inner_shape):
            form = awkward.forms.RegularForm(form, dim)
        return form

    @property
    def cache_key(self):
        return f"{type(self).__name__}({self._model.__name__})"

    @property
    def typename(self):
        return uproot.model.classname_decode(self._model.__name__)[0]

    def _wrap_almost_finalized(self, array):
        return StridedObjectArray(self, array)

    def final_array(
        self,
        basket_arrays,
        entry_start,
        entry_stop,
        entry_offsets,
        library,
        branch,
        options,
    ):
        self.hook_before_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
        )

        # The per-basket ``basket_array`` may prepend an ``@headers`` field to
        # the dtype (see ``basket_array`` below). Derive the output dtype from
        # the actual basket arrays rather than from a mutated ``self._to_dtype``
        # so that the interpretation object can be safely shared across threads.
        to_dtype = self._to_dtype
        for basket_array in basket_arrays.values():
            if basket_array.dtype.names is not None and "@headers" in (
                basket_array.dtype.names
            ):
                to_dtype = basket_array.dtype.newbyteorder("=")
                break

        if entry_start >= entry_stop:
            output = library.empty((0,), to_dtype)

        else:
            length = 0
            start = entry_offsets[0]
            for _, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    length += entry_stop - entry_start
                elif start <= entry_start < stop:
                    length += stop - entry_start
                elif start <= entry_stop <= stop:
                    length += entry_stop - start
                elif entry_start < stop and start <= entry_stop:
                    length += stop - start
                start = stop

            output = library.empty((length,), to_dtype)

            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if start <= entry_start and entry_stop <= stop:
                    local_start = entry_start - start
                    local_stop = entry_stop - start
                    basket_array = basket_arrays[basket_num]
                    output[:] = basket_array[local_start:local_stop]

                elif start <= entry_start < stop:
                    local_start = entry_start - start
                    local_stop = stop - start
                    basket_array = basket_arrays[basket_num]
                    output[: stop - entry_start] = basket_array[local_start:local_stop]

                elif start <= entry_stop <= stop:
                    local_start = 0
                    local_stop = entry_stop - start
                    basket_array = basket_arrays[basket_num]
                    output[start - entry_start :] = basket_array[local_start:local_stop]

                elif entry_start < stop and start <= entry_stop:
                    basket_array = basket_arrays[basket_num]
                    output[start - entry_start : stop - entry_start] = basket_array

                start = stop

        output = output.view(output.dtype.newbyteorder("="))
        self.hook_before_library_finalize(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        output = self._wrap_almost_finalized(output)

        output = library.finalize(
            output, branch, self, entry_start, entry_stop, options
        )

        self.hook_after_final_array(
            basket_arrays=basket_arrays,
            entry_start=entry_start,
            entry_stop=entry_stop,
            entry_offsets=entry_offsets,
            library=library,
            branch=branch,
            output=output,
        )

        return output

    def basket_array(
        self,
        data,
        byte_offsets,
        basket,
        branch,
        context,
        cursor_offset,
        library,
        options,
    ):
        self.hook_before_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )

        dtype, shape = uproot.interpretation.numerical._dtype_shape(self._from_dtype)

        if (
            byte_offsets is not None
            and dtype.itemsize != byte_offsets[1] - byte_offsets[0]
            and self.all_headers_prepended
        ):
            dtype = numpy.dtype(
                [
                    (
                        "@headers",
                        "u1",
                        byte_offsets[1] - byte_offsets[0] - dtype.itemsize,
                    )
                ]
                + [
                    (x, str(y[0]))
                    for x, y in sorted(dtype.fields.items(), key=lambda k: k[1][1])
                ]
            )
        try:
            output = data.view(dtype).reshape((-1, *shape))

        except ValueError as err:
            raise ValueError(
                f"""basket {basket.basket_num} in tree/branch {branch.object_path} has the wrong number of bytes ({len(data)}) """
                f"""for interpretation {self}
in file {branch.file.file_path}"""
            ) from err
        self.hook_after_basket_array(
            data=data,
            byte_offsets=byte_offsets,
            basket=basket,
            branch=branch,
            context=context,
            output=output,
            cursor_offset=cursor_offset,
            library=library,
            options=options,
        )
        return output


class CannotBeStrided(Exception):
    """
    Exception used to stop recursion over
    :ref:`uproot.model.Model.strided_interpretation` and
    :ref:`uproot.containers.AsContainer.strided_interpretation` as soon as a
    non-conforming type is found.
    """


class CannotBeForth(Exception):
    """
    Exception used to stop recursion over
    :ref:`uproot.model.Model.read` as soon as a
    non-conforming type is found.
    """


class CannotBeAwkward(Exception):
    """
    Exception used to stop recursion over
    :ref:`uproot.interpretation.Interpretation.awkward_form`,
    :ref:`uproot.model.Model.awkward_form` and
    :ref:`uproot.containers.AsContainer.awkward_form` as soon as a
    non-conforming type is found.
    """

    def __init__(self, because):
        self._because = because

    @property
    def because(self):
        return self._because


class ObjectArray:
    """
    Args:
        model (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`): The
            full Uproot deserialization model for the data.
        branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` from
            which the data are drawn.
        context (dict): Auxiliary data used in deserialization.
        byte_offsets (array of ``numpy.int32``): Index where each entry of the
            ``byte_content`` starts and stops.
        byte_content (array of ``numpy.uint8``): Raw but uncompressed data,
            directly from
            :ref:`uproot.interpretation.Interpretation.basket_array`.
        cursor_offset (int): Correction to the integer keys used in
            :ref:`uproot.source.cursor.Cursor.refs` for objects deserialized
            by reference (:doc:`uproot.deserialization.read_object_any`).

    Temporary array filled by
    :ref:`uproot.interpretation.objects.AsObjects.basket_array`, which will be
    turned into a NumPy, Awkward, or other array, depending on the specified
    :doc:`uproot.interpretation.library.Library`.
    """

    def __init__(
        self, model, branch, context, byte_offsets, byte_content, cursor_offset
    ):
        self._model = model
        self._branch = branch
        self._context = context
        self._byte_offsets = byte_offsets
        self._byte_content = byte_content
        self._cursor_offset = cursor_offset
        self._detached_file = self._branch.file.detached

    def __repr__(self):
        return f"ObjectArray({self._model}, {self._branch}, {self._context}, {self._byte_offsets}, {self._byte_content}, {self._cursor_offset})"

    @property
    def model(self):
        """
        The full Uproot deserialization model for the data
        (:doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`).
        """
        return self._model

    @property
    def branch(self):
        """
        The ``TBranch`` from which the data are drawn.
        """
        return self._branch

    @property
    def context(self):
        """
        Auxiliary data used in deserialization (dict).
        """
        return self._context

    @property
    def byte_offsets(self):
        """
        Index where each entry of the ``byte_content`` starts and stops.
        """
        return self._byte_offsets

    @property
    def byte_content(self):
        """
        Raw but uncompressed data, directly from
        :ref:`uproot.interpretation.Interpretation.basket_array`.
        """
        return self._byte_content

    @property
    def cursor_offset(self):
        """
        Correction to the integer keys used in
        :ref:`uproot.source.cursor.Cursor.refs` for objects deserialized by
        reference (:doc:`uproot.deserialization.read_object_any`).
        """
        return self._cursor_offset

    def to_numpy(self):
        """
        Convert this ObjectArray into a NumPy ``dtype="O"`` (object) array.
        """
        output = numpy.empty(len(self), dtype=numpy.dtype(object))
        for i in range(len(self)):
            output[i] = self[i]
        return output

    def __len__(self):
        return len(self._byte_offsets) - 1

    def __getitem__(self, where):
        if uproot._util.isint(where):
            byte_start = self._byte_offsets[where]
            byte_stop = self._byte_offsets[where + 1]
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot.source.chunk.Chunk.wrap(self._branch.file.source, data)
            cursor = uproot.source.cursor.Cursor(
                0, origin=-(byte_start + self._cursor_offset)
            )
            return self._model.read(
                chunk,
                cursor,
                self._context,
                self._branch.file,
                self._detached_file,
                self._branch,
            )

        elif isinstance(where, slice):
            start, stop, step = where.indices(len(self))
            if step != 1:
                raise NotImplementedError(repr(where))
            return ObjectArray(
                self._model,
                self._branch,
                self._context,
                self._byte_offsets[start : max(start, stop) + 1],
                self._byte_content,
                self._cursor_offset,
            )

        else:
            raise NotImplementedError(repr(where))

    def __iter__(self):
        source = self._branch.file.source
        context = self._context
        file = self._branch.file
        selffile = self._detached_file
        branch = self._branch
        byte_start = self._byte_offsets[0]
        for byte_stop in self._byte_offsets[1:]:
            data = self._byte_content[byte_start:byte_stop]
            chunk = uproot.source.chunk.Chunk.wrap(source, data)
            cursor = uproot.source.cursor.Cursor(
                0, origin=-(byte_start + self._cursor_offset)
            )
            yield self._model.read(chunk, cursor, context, file, selffile, branch)
            byte_start = byte_stop


def _strided_object(path, interpretation, data):
    out = interpretation._model.empty()
    for name, member in interpretation._members:
        p = name
        if len(path) != 0:
            p = path + "/" + name
        if isinstance(member, AsStridedObjects):
            out._members[name] = _strided_object(p, member, data)
        else:
            out._members[name] = data[p]
    return out


class StridedObjectArray:
    """
    Args:
        interpretation (:doc:`uproot.interpretation.objects.AsStridedObjects`): The
            interpretation that produced this array.
        array (array): Underlying array object, which may be NumPy or another
            temporary array.

    Temporary array filled by
    :ref:`uproot.interpretation.objects.AsStridedObjects.basket_array`, which
    will be turned into a NumPy, Awkward, or other array, depending on the
    specified :doc:`uproot.interpretation.library.Library`.
    """

    def __init__(self, interpretation, array):
        self._interpretation = interpretation
        self._array = array
        if not isinstance(self._array, numpy.ndarray):
            raise NotImplementedError(
                "non-numpy temporary arrays require changes in library interpretations"
            )

    @property
    def interpretation(self):
        """
        The interpretation that produced this array.
        """
        return self._interpretation

    @property
    def array(self):
        """
        Underlying array object, which may be NumPy or another temporary array.
        """
        return self._array

    @property
    def shape(self):
        assert self._array.shape == (len(self), *self._interpretation.inner_shape)
        return self._array.shape

    def __repr__(self):
        return f"StridedObjectArray({self._interpretation}, {self._array})"

    def __len__(self):
        return len(self._array)

    def __getitem__(self, where):
        out = self._array[where]
        if isinstance(out, numpy.void):
            return _strided_object("", self._interpretation, out)
        else:
            return StridedObjectArray(self._interpretation, out)

    def __iter__(self):
        for x in self._array:
            if isinstance(x, numpy.void):
                yield _strided_object("", self._interpretation, x)
            else:
                yield StridedObjectArray(self._interpretation, x)

    def ndenumerate(self):
        for i, x in numpy.ndenumerate(self._array):
            yield i, _strided_object("", self._interpretation, x)
