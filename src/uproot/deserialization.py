# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines low-level routines for deserialization, including
:doc:`uproot.deserialization.compile_class`, which creates class objects from
``TStreamerInfo``-derived code, and
:doc:`uproot.deserialization.read_object_any`, which manages references to
previously read objects.
"""

from __future__ import annotations

import struct
import sys

import numpy

import uproot

scope = {
    "struct": struct,
    "numpy": numpy,
    "uproot": uproot,
}

np_uint8 = numpy.dtype("u1")

# Plain-int copies of these constants, used in the hot deserialization path to
# avoid creating numpy scalars and doing numpy bitwise ops on every object read.
_kByteCountMask = int(uproot.const.kByteCountMask)
_kStreamedMemberWise = int(uproot.const.kStreamedMemberWise)
_kClassMask = int(uproot.const.kClassMask)
_kNewClassTag = int(uproot.const.kNewClassTag)


def _actually_compile(class_code, new_scope):
    exec(compile(class_code, "<dynamic>", "exec"), new_scope)


def _yield_all_behaviors(cls, c):
    behavior_cls = uproot.behavior_of(uproot.model.classname_decode(cls.__name__)[0])
    if behavior_cls is not None:
        yield behavior_cls
    if hasattr(cls, "base_names_versions"):
        for base_name, base_version in cls.base_names_versions:
            yield from _yield_all_behaviors(c(base_name, base_version), c)


def compile_class(file, classes, class_code, class_name):
    """
    Args:
        file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
            :doc:`uproot.model.Model` classes as needed from its
            :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
            for error messages.
        classes (dict): MutableMapping in which to add the finished class.
        class_code (str): Python code string defining the new class.
        class_name (str): Python (encoded) name of the new class. See
            :doc:`uproot.model.classname_decode` and
            :doc:`uproot.model.classname_encode`.

    Compile a new class from Python code and insert it in the dict of classes.
    """
    new_scope = scope.copy()
    for cls in classes.values():
        new_scope[cls.__name__] = cls

    def c(name, version=None):
        name = uproot.model.classname_regularize(name)
        cls = new_scope.get(uproot.model.classname_encode(name, version))
        if cls is None:
            cls = new_scope.get(uproot.model.classname_encode(name))
        if cls is None:
            cls = file.class_named(name, version)
        return cls

    new_scope["c"] = c
    new_scope["af"] = uproot._awkwardforth
    new_scope["DeserializationError"] = uproot.deserialization.DeserializationError
    new_scope["CannotBeStrided"] = uproot.interpretation.objects.CannotBeStrided
    new_scope["CannotBeForth"] = uproot.interpretation.objects.CannotBeForth
    new_scope["CannotBeAwkward"] = uproot.interpretation.objects.CannotBeAwkward

    try:
        _actually_compile(class_code, new_scope)
    except SyntaxError as err:
        raise SyntaxError(class_code + "\n\n" + str(err)) from err

    out = new_scope[class_name]
    out.class_code = class_code
    out.__module__ = "uproot.dynamic"
    setattr(uproot.dynamic, out.__name__, out)

    behaviors = tuple(_yield_all_behaviors(out, c))
    exclude = tuple(
        bad for cls in behaviors if hasattr(cls, "no_inherit") for bad in cls.no_inherit
    )
    behaviors = tuple(cls for cls in behaviors if cls not in exclude)
    out.behaviors = behaviors

    if len(behaviors) != 0:
        out = uproot._util.new_class(out.__name__, (*behaviors, out), {})

    return out


_numbytes_version_1 = struct.Struct(">IH")
_numbytes_version_2 = struct.Struct(">H")


def numbytes_version(chunk, cursor, context, move=True):
    """
    Args:
        chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
            from the file :doc:`uproot.source.chunk.Source`.
        cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
            that ``chunk``.
        context (dict): Auxiliary data used in deserialization.
        move (bool): If True, move the ``cursor`` to a position just past the
            bytes that were read. If False, don't move the ``cursor`` at all.

    Deserialize a number of bytes and version number, which is usually 6 bytes
    but may be 2 bytes if the number of bytes isn't given.

    Returns a 3-tuple of

    * number of bytes (int or None if unknown)
    * instance version (int)
    * is memberwise (bool): True if the memberwise bit is set in the version
      number; False otherwise.
    """
    num_bytes, version = cursor.fields(chunk, _numbytes_version_1, context, move=False)

    if num_bytes & _kByteCountMask:
        # Note this extra 4 bytes: the num_bytes field doesn't count itself,
        # but we count the Model.start_cursor position from the point just
        # before these two fields (since num_bytes might not exist, it's a more
        # stable point than after num_bytes).
        #                                                           |
        #                                                           V
        num_bytes = (num_bytes & ~_kByteCountMask) + 4
        if move:
            cursor.skip(_numbytes_version_1.size)

    else:
        num_bytes = None
        version = cursor.field(chunk, _numbytes_version_2, context, move=move)

    is_memberwise = bool(version & _kStreamedMemberWise)
    if is_memberwise:
        version = version & ~_kStreamedMemberWise

    return num_bytes, version, is_memberwise


def numbytes_check(
    chunk, start_cursor, stop_cursor, num_bytes, classname, context, file_path
):
    """
    Args:
        chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
            from the file :doc:`uproot.source.chunk.Source`.
        start_cursor (:doc:`uproot.source.cursor.Cursor`): Initial position in
            that ``chunk``.
        stop_cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
            that ``chunk``.
        num_bytes (int or None): If an integer, the number of bytes to compare
            with the difference between ``start_cursor`` and ``stop_cursor``;
            if None, this function does nothing.
        classname (str): C++ (decoded) name of the class for error messages.
        context (dict): Auxiliary data used in deserialization.
        file_path (str): Name of the file for error messages.

    Verifies that the number of bytes matches the change in position of the
    cursor (if ``num_bytes`` is not None).

    Raises a :doc:`uproot.deserialization.DeserializationError` on failure.
    """
    if num_bytes is not None:
        observed = stop_cursor.displacement(start_cursor)
        if observed != num_bytes:
            raise DeserializationError(
                f"""expected {num_bytes} bytes but cursor moved by {observed} bytes (through {classname})""",
                chunk,
                stop_cursor,
                context,
                file_path,
            )


_read_object_any_format1 = struct.Struct(">I")


def read_object_any(chunk, cursor, context, file, selffile, parent, as_class=None):
    """
    Args:
        chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
            from the file :doc:`uproot.source.chunk.Source`.
        cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
            that ``chunk``.
        context (dict): Auxiliary data used in deserialization.
        file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
            :doc:`uproot.model.Model` classes as needed from its
            :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
            for error messages.
        selffile (:doc:`uproot.reading.CommonFileMethods`): A possibly
            :doc:`uproot.reading.DetachedFile` associated with the ``parent``.
        parent (None or calling object): The previous ``read`` in the
            recursive descent.
        as_class (None or :doc:`uproot.model.Model`): If None, use the class
            indicated in the byte stream; otherwise, use this class.

    Generic read function, which may deliver an instance of any class and may
    reference previously read objects.

    This function is the reason why :doc:`uproot.source.cursor.Cursor` has a
    :ref:`uproot.source.cursor.Cursor.refs`; that dictionary holds previously
    read objects that might need to be accessed later.

    The :doc:`uproot.source.cursor.Cursor` has an
    :ref:`uproot.source.cursor.Cursor.origin` to account for the fact that
    seek positions for keys in the reference dict are relative to the start of
    the :doc:`uproot.source.chunk.Chunk`, rather than the start of the file
    (as it would have to be for decompressed chunks).
    """
    # TBufferFile::ReadObjectAny()
    # https://github.com/root-project/root/blob/c4aa801d24d0b1eeb6c1623fd18160ef2397ee54/io/io/src/TBufferFile.cxx#L2684
    # https://github.com/root-project/root/blob/c4aa801d24d0b1eeb6c1623fd18160ef2397ee54/io/io/src/TBufferFile.cxx#L2404

    beg = cursor.displacement()
    bcnt = cursor.field(chunk, _read_object_any_format1, context)

    if (bcnt & _kByteCountMask) == 0 or (bcnt == _kNewClassTag):
        vers = 0
        start = 0
        tag = bcnt
        bcnt = 0
    else:
        vers = 1
        start = cursor.displacement()
        tag = cursor.field(chunk, _read_object_any_format1, context)

    if tag & _kClassMask == 0:
        # reference object

        if tag == 0:
            return None  # return null

        elif tag == 1:
            return parent  # return parent

        elif tag not in cursor.refs:
            # Skip past this unresolved forward reference, matching ROOT's
            # TBufferFile::CheckByteCount, which positions the buffer at
            # startpos + bcnt + sizeof(UInt_t). Here ``beg`` is the position
            # before the byte-count field (ROOT's startpos) and the trailing
            # ``+ 4`` is sizeof(UInt_t); the stripped ``bcnt`` does not get an
            # extra 4 (unlike numbytes_version, which measures from a different
            # reference point).
            if bcnt & _kByteCountMask:
                bcnt = bcnt & ~_kByteCountMask

            # jump past this object
            cursor.move_to(cursor.origin + beg + bcnt + 4)
            return None  # return null

        else:
            return cursor.refs[int(tag)]  # return object

    elif tag == _kNewClassTag:
        # new class and object

        classname = cursor.classname(chunk, context)

        cls = file.class_named(classname)

        if vers > 0:
            cursor.refs[start + uproot.const.kMapOffset] = cls
        else:
            cursor.refs[len(cursor.refs) + 1] = cls

        if as_class is None:
            obj = cls.read(chunk, cursor, context, file, selffile, parent)
        else:
            obj = as_class.read(chunk, cursor, context, file, selffile, parent)

        if vers > 0:
            cursor.refs[beg + uproot.const.kMapOffset] = obj
        else:
            cursor.refs[len(cursor.refs) + 1] = obj

        return obj  # return object

    else:
        # reference class, new object

        ref = tag & ~_kClassMask

        if as_class is None:
            if ref not in cursor.refs:
                if getattr(file, "file_path", None) is None:
                    in_file = ""
                else:
                    in_file = f"\n\nin file {file.file_path}"
                raise DeserializationError(
                    """invalid class-tag reference: {}

    Known references: {}{}""".format(
                        ref, ", ".join(str(x) for x in cursor.refs), in_file
                    ),
                    chunk,
                    cursor,
                    context,
                    file.file_path,
                )

            cls = cursor.refs[ref]  # reference class
            obj = cls.read(chunk, cursor, context, file, selffile, parent)

        else:
            obj = as_class.read(chunk, cursor, context, file, selffile, parent)

        if vers > 0:
            cursor.refs[beg + uproot.const.kMapOffset] = obj
        else:
            cursor.refs[len(cursor.refs) + 1] = obj

        return obj  # return object


scope["read_object_any"] = read_object_any


class DeserializationError(Exception):
    """
    Error raised when a ROOT file cannot be deserialized.

    If the first attempt in :ref:`uproot.reading.ReadOnlyKey.get` failed with
    predefined :doc:`uproot.model.Model` classes, this exception is caught
    and retried with ``TStreamerInfo``-derived classes, so
    :doc:`uproot.deserialization.DeserializationError` sometimes appears in an
    exception chain two levels deep. (Some ROOT files do have classes that don't
    match the standard ``TStreamerInfo``; they may have been produced from
    private builds of ROOT between official releases.)

    If a :doc:`uproot.deserialization.DeserializationError` is caught, the byte
    stream at the position where it failed can be inspected with

    * :ref:`uproot.deserialization.DeserializationError.debug`
    * :ref:`uproot.deserialization.DeserializationError.debug_array`
    * :ref:`uproot.deserialization.DeserializationError.partial_object`
    """

    def __init__(self, message, chunk, cursor, context, file_path):
        self.message = message
        self.chunk = chunk
        self.cursor = cursor
        self.context = context
        self.file_path = file_path

    def __str__(self):
        lines = []
        indent = "    "
        last = None
        for obj in self.context.get("breadcrumbs", ()):
            lines.append(
                "{}{} version {} as {}.{} ({} bytes)".format(
                    indent,
                    obj.classname,
                    obj.instance_version,
                    type(obj).__module__,
                    type(obj).__name__,
                    "?" if obj.num_bytes is None else obj.num_bytes,
                )
            )
            indent = indent + "    "
            for v in getattr(obj, "_bases", []):
                lines.append(f"{indent}(base): {v!r}")
            for k, v in getattr(obj, "_members", {}).items():
                lines.append(f"{indent}{k}: {v!r}")
            last = obj

        if last is not None:
            base_names_versions = getattr(last, "base_names_versions", None)
            bases = getattr(last, "_bases", None)
            if base_names_versions is not None and bases is not None:
                base_names = [n for n, v in base_names_versions]
                for c in bases:
                    classname = getattr(c, "classname", None)
                    if classname is not None:
                        if classname in base_names:
                            base_names[base_names.index(classname)] = (
                                "(" + classname + ")"
                            )
                        else:
                            base_names.append(classname + "?")
                if len(base_names) != 0:
                    lines.append(
                        "Base classes for {}: {}".format(
                            last.classname, ", ".join(base_names)
                        )
                    )

            member_names = getattr(last, "member_names", None)
            members = getattr(last, "_members", None)
            if member_names is not None and members is not None:
                member_names = list(member_names)
                for n in members:
                    if n in member_names:
                        member_names[member_names.index(n)] = "(" + n + ")"
                    else:
                        member_names.append(n + "?")
                if len(member_names) != 0:
                    lines.append(
                        "Members for {}: {}".format(
                            last.classname, ", ".join(member_names)
                        )
                    )

        in_parent = ""
        if "TBranch" in self.context:
            in_parent = "\nin TBranch {}".format(self.context["TBranch"].object_path)
        elif "TKey" in self.context:
            in_parent = "\nin object {}".format(self.context["TKey"].object_path)

        if len(lines) == 0:
            return f"""{self.message}
in file {self.file_path}{in_parent}"""
        else:
            return """while reading

{}

{}
in file {}{}""".format("\n".join(lines), self.message, self.file_path, in_parent)

    def debug(
        self, skip_bytes=0, limit_bytes=None, dtype=None, offset=0, stream=sys.stdout
    ):
        """
        Args:
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the :doc:`uproot.source.chunk.Chunk`. May be
                negative, to examine the byte stream leading up to the attempted
                deserialization.
            limit_bytes (None or int): Number of bytes to limit the output to.
                A line of debugging output (without any ``offset``) is 20 bytes,
                so multiples of 20 show full lines. If None, everything is
                shown to the end of the :doc:`uproot.source.chunk.Chunk`,
                which might be large.
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

        Presents the byte stream at the point where deserialization failed.

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
        cursor = self.cursor.copy()
        cursor.skip(skip_bytes)
        cursor.debug(
            self.chunk,
            context=self.context,
            limit_bytes=limit_bytes,
            dtype=dtype,
            offset=offset,
            stream=stream,
        )

    def debug_array(self, skip_bytes=0, dtype=np_uint8):
        """
        Args:
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the :doc:`uproot.source.chunk.Chunk`. May be
                negative, to examine the byte stream leading up to the attempted
                deserialization.
            dtype (``numpy.dtype`` or its constructor argument): Data type in
                which to interpret the data. (The size of the array returned is
                truncated to this ``dtype.itemsize``.)

        Like :ref:`uproot.deserialization.DeserializationError.debug`, but
        returns a NumPy array for further inspection.
        """
        dtype = numpy.dtype(dtype)
        cursor = self.cursor.copy()
        cursor.skip(skip_bytes)
        out = self.chunk.remainder(cursor.index, cursor, self.context)
        return out[: (len(out) // dtype.itemsize) * dtype.itemsize].view(dtype)

    @property
    def partial_object(self):
        """
        The object, partially filled, which may contain some clues as to what
        went wrong.
        """
        if "breadcrumbs" in self.context:
            return self.context["breadcrumbs"][-1]
        else:
            return None
