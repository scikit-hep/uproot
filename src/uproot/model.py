# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines utilities for modeling C++ objects as Python objects and the
:doc:`uproot.model.Model` class, which is the superclass of all objects that
are read from ROOT files.

The :doc:`uproot.model.VersionedModel` class is the superclass of all models
whose deserialization routines are specialized by ROOT class version.

A :doc:`uproot.model.DispatchByVersion` subclass selects a versioned model
after reading its version bytes.

The :doc:`uproot.model.UnknownClass` and
:doc:`uproot.model.UnknownClassVersion` are placeholders for data that could
not be modeled, either because the class has no streamer or no streamer for its
version.
"""

from __future__ import annotations

import functools
import re
import sys
import threading
import weakref

import numpy

import uproot

bootstrap_classnames = [
    "TStreamerInfo",
    "TStreamerElement",
    "TStreamerArtificial",
    "TStreamerBase",
    "TStreamerBasicPointer",
    "TStreamerBasicType",
    "TStreamerLoop",
    "TStreamerObject",
    "TStreamerObjectAny",
    "TStreamerObjectAnyPointer",
    "TStreamerObjectPointer",
    "TStreamerSTL",
    "TStreamerSTLstring",
    "TStreamerString",
    "TList",
    "TObjArray",
    "TClonesArray",
    "TObjString",
]
never_from_streamers = [
    "TString",
    "TList",
]

np_uint8 = numpy.dtype("u1")


def bootstrap_classes():
    """
    Returns the basic classes that are needed to load other classes (streamers,
    TList, TObjArray, TObjString).
    """
    import uproot.models.TList
    import uproot.models.TObjArray
    import uproot.models.TObjString
    import uproot.streamers

    custom_classes = {}
    for classname in bootstrap_classnames:
        custom_classes[classname] = uproot.classes[classname]

    return custom_classes


def reset_classes():
    """
    Removes all classes from ``uproot.classes`` and ``uproot.unknown_classes``
    and refills ``uproot.classes`` with original versions of these classes.
    """
    from importlib import reload

    uproot.classes = {}
    uproot.unknown_classes = {}

    reload(uproot.streamers)
    reload(uproot.models.TObject)
    reload(uproot.models.TString)
    reload(uproot.models.TArray)
    reload(uproot.models.TNamed)
    reload(uproot.models.TList)
    reload(uproot.models.THashList)
    reload(uproot.models.TObjArray)
    reload(uproot.models.TClonesArray)
    reload(uproot.models.TObjString)
    reload(uproot.models.TAtt)
    reload(uproot.models.TDatime)
    reload(uproot.models.TRef)

    reload(uproot.models.TTable)
    reload(uproot.models.TTree)
    reload(uproot.models.TBranch)
    reload(uproot.models.TLeaf)
    reload(uproot.models.TBasket)
    reload(uproot.models.RNTuple)
    reload(uproot.models.TH)
    reload(uproot.models.TGraph)
    reload(uproot.models.TEfficiency)
    reload(uproot.models.TMatrixT)
    reload(uproot.models.TTime)


_root_alias_to_c_primitive = {
    "Bool_t": "bool",
    "Char_t": "char",
    "UChar_t": "unsigned char",
    "Short_t": "short",
    "UShort_t": "unsigned short",
    "Int_t": "int",
    "UInt_t": "unsigned int",
    "Long_t": "long",
    "ULong_t": "unsigned long",
    "Long64_t": "long long",
    "ULong64_t": "unsigned long long",
    "Size_t": "size_t",
    "Float_t": "float",
    "Double_t": "double",
    "LongDouble_t": "long double",
}

_classname_regularize = re.compile(r"\s*(<|>|,|::)\s*")
_classname_regularize_type = re.compile(
    r"[<,](" + "|".join([re.escape(p) for p in _root_alias_to_c_primitive]) + r")[>,]"
)

_classname_encode_pattern = re.compile(rb"[^a-zA-Z0-9]+")
_classname_decode_antiversion = re.compile(rb".*_([0-9a-f][0-9a-f])+_v([0-9]+)$")
_classname_decode_version = re.compile(rb".*_v([0-9]+)$")
_classname_decode_pattern = re.compile(rb"_(([0-9a-f][0-9a-f])+)_")


def _classname_decode_convert(hex_characters):
    g = hex_characters.group(1)
    return bytes(int(g[i : i + 2], 16) for i in range(0, len(g), 2))


def _classname_encode_convert(bad_characters):
    g = bad_characters.group(0)
    return b"_" + b"".join(f"{x:02x}".encode() for x in g) + b"_"


def classname_regularize(classname):
    """
    Removes spaces around ``<``, ``>``, and ``::`` characters in a classname
    so that they can be matched by string name.

    If ``classname`` is None, this function returns None. Otherwise, it must be
    a string and it returns a string.
    """
    if classname is not None:
        classname = re.sub(_classname_regularize, r"\1", classname)

        m = _classname_regularize_type.search(classname)

        while m is not None:
            start, stop = m.span(1)
            token = classname[start:stop]
            replacement = _root_alias_to_c_primitive[token]
            classname = classname[:start] + replacement + classname[stop:]

            m = _classname_regularize_type.search(classname)

    return classname


@functools.lru_cache(maxsize=1024)
def classname_decode(encoded_classname):
    """
    Converts a Python (encoded) classname, such as ``Model_Some_3a3a_Thing``
    into a C++ (decoded) classname, such as ``Some::Thing``.

    C++ classnames can include namespace delimiters (``::``) and template
    arguments (``<`` and ``>``), which have to be translated into
    ``[A-Za-z_][A-Za-z0-9_]*`` for Python. Non-conforming characters and also
    underscores are translated to their hexadecimal equivalents and surrounded
    by underscores. Additionally, Python models of C++ classes are prepended
    with ``Model_`` (or ``Unknown_`` if a streamer isn't found).
    """
    if encoded_classname.startswith("Unknown_"):
        raw = encoded_classname[8:].encode()
    elif encoded_classname.startswith("Model_"):
        raw = encoded_classname[6:].encode()
    else:
        raise ValueError(f"not an encoded classname: {encoded_classname}")

    if _classname_decode_antiversion.match(raw) is not None:
        version = None
    else:
        m = _classname_decode_version.match(raw)
        if m is None:
            version = None
        else:
            version = int(m.group(1))
            raw = raw[: -len(m.group(1)) - 2]

    out = _classname_decode_pattern.sub(_classname_decode_convert, raw)
    return out.decode(), version


def classname_encode(classname, version=None, unknown=False):
    """
    Converts a C++ (decoded) classname, such as ``Some::Thing`` into a Python
    classname (encoded), such as ``Model_Some_3a3a_Thing``.

    If ``version`` is a number such as ``2``, the Python name is suffixed by
    version, such as ``Model_Some_3a3a_Thing_v2``.

    If ``unknown`` is True, the ``Model_`` prefix becomes ``Unknown_``.

    C++ classnames can include namespace delimiters (``::``) and template
    arguments (``<`` and ``>``), which have to be translated into
    ``[A-Za-z_][A-Za-z0-9_]*`` for Python. Non-conforming characters and also
    underscores are translated to their hexadecimal equivalents and surrounded
    by underscores. Additionally, Python models of C++ classes are prepended
    with ``Model_`` (or ``Unknown_`` if a streamer isn't found).
    """
    prefix = "Unknown_" if unknown else "Model_"
    if classname.startswith(prefix):
        raise ValueError(f"classname is already encoded: {classname}")

    v = "" if version is None else "_v" + str(version)

    raw = classname.encode()
    out = _classname_encode_pattern.sub(_classname_encode_convert, raw)
    return prefix + out.decode() + v


def classname_version(encoded_classname):
    """
    Extracts a version number from a Python (encoded) classname, if it has one.

    For example, ``Model_Some_3a3a_Thing_v2`` returns ``2``.

    A name without a version number, such as ``Model_Some_3a3a_Thing``, returns
    None.
    """
    raw = encoded_classname.encode()
    if _classname_decode_antiversion.match(raw) is not None:
        return None
    else:
        m = _classname_decode_version.match(raw)
        if m is None:
            return None
        else:
            return int(m.group(1))


def class_named(classname, version=None, custom_classes=None):
    """
    Returns a class with a given C++ (decoded) classname.

    If ``version`` is None, no attempt is made to find a specific version.

    * If the class is a :doc:`uproot.model.DispatchByVersion`, then this is
      object returned.
    * If the class is a versionless model, then this is the object returned.

    If ``version`` is an integer, an attempt is made to find the specific
    version.

    * If the class is a :doc:`uproot.model.DispatchByVersion`, then it is
      queried for a versioned model.
    * If the class is a versionless model, then this is the object returned.

    If ``custom_classes`` are provided, then these are searched (exclusively)
    for the class. If ``custom_classes`` is None, then ``uproot.classes`` is
    used.

    No classes are created if a class is not found (an error is raised).
    """
    if custom_classes is None:
        classes = uproot.classes
        where = "uproot.classes"
    else:
        classes = custom_classes
        where = "the 'custom_classes' dict"

    cls = classes.get(classname)
    if cls is None:
        raise ValueError(f"no class named {classname} in {where}")

    if version is not None and issubclass(cls, DispatchByVersion):
        versioned_cls = cls.class_of_version(version)
        if versioned_cls is not None:
            return versioned_cls
        else:
            raise ValueError(
                f"no class named {classname} with version {version} in {where}"
            )

    else:
        return cls


def has_class_named(classname, version=None, custom_classes=None):
    """
    Returns True if :doc:`uproot.model.class_named` would find a class,
    False if it would raise an exception.
    """
    cls = maybe_custom_classes(classname, custom_classes).get(classname)
    if cls is None:
        return False

    if version is not None and issubclass(cls, DispatchByVersion):
        return cls.has_version(version)
    else:
        return True


def maybe_custom_classes(classname, custom_classes):
    """
    Passes through ``custom_classes`` if it is not None; returns
    ``uproot.classes`` otherwise.

    Some ``classnames`` are never custom (see ``uproot.model.never_from_streamers``).
    """
    if custom_classes is None or classname in never_from_streamers:
        return uproot.classes
    else:
        return custom_classes


def unknown_class(classname, version=None):
    """
    Returns the placeholder class for a ``classname`` whose streamer could not
    be found, creating it (and adding it to ``uproot.unknown_classes``) if it's
    not already there.

    If ``version`` is None, the class is a :doc:`uproot.model.UnknownClass`;
    otherwise, it is a :doc:`uproot.model.UnknownClassVersion` for that version.
    """
    # key on the encoded name, which carries the version and so keeps distinct
    # versions (and the versionless UnknownClass) apart
    encoded_classname = classname_encode(classname, version, unknown=True)
    cls = uproot.unknown_classes.get(encoded_classname)
    if cls is None:
        base = UnknownClass if version is None else UnknownClassVersion
        cls = uproot._util.new_class(encoded_classname, (base,), {})
        uproot.unknown_classes[encoded_classname] = cls
    return cls


class Model:
    """
    Abstract class for all objects extracted from ROOT files (except for
    :doc:`uproot.reading.ReadOnlyFile`, :doc:`uproot.reading.ReadOnlyDirectory`,
    and :doc:`uproot.reading.ReadOnlyKey`).

    A model is instantiated from a file using the :ref:`uproot.model.Model.read`
    classmethod or synthetically using the :ref:`uproot.model.Model.empty`
    classmethod, not through a normal constructor.

    Models point back to the file from which they were created, though only a
    few classes (named in ``uproot.reading.must_be_attached``) have an open,
    readable file attached; the rest have a :doc:`uproot.reading.DetachedFile`
    with information about the file, while not holding the file open.

    Uproot recognizes *some* of ROOT's thousands of classes, by way of methods
    and properties defined in :doc:`uproot.behaviors`. Examples include

    * :doc:`uproot.behaviors.TTree.TTree`
    * :doc:`uproot.behaviors.TH1.TH1`

    These classes are the most convenient to work with and have specialized
    documentation.

    Classes that don't have any predefined behaviors are still usable through
    their member data.

    * :ref:`uproot.model.Model.members`: a dict of C++ member names and values
      directly in this class.
    * :ref:`uproot.model.Model.all_members`: a dict of C++ member names and
      values in this class or any superclasses.
    * :ref:`uproot.model.Model.member`: method that takes a C++ member name
      and returns its value (from this or any superclass).
    * :ref:`uproot.model.Model.has_member`: method that takes a C++ member
      name and returns True if it exists (in this or any superclass), False
      otherwise.

    Accessing a data structure through its C++ members may be a prelude to
    adding custom behaviors for it. Before we know what conveniences to add, we
    need to know how they'll be used: this information comes from the user
    community.

    Pythonic models don't follow the same class inheritance tree as their C++
    counterparts: most of them are direct subclasses of
    :doc:`uproot.model.Model`, :doc:`uproot.model.DispatchByVersion`, or
    :doc:`uproot.model.VersionedModel`. To separate an object's members
    from its superclass members, a model instance is created for each and
    the superclass parts are included in a list called
    :ref:`uproot.model.Model.bases`.
    """

    base_names_versions = []
    member_names = []
    class_flags = {}
    class_code = None
    class_streamer = None
    class_rawstreamers = ()
    writable = False
    _deeply_writable = False
    behaviors = ()

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = f" (version {self.class_version})"
        return f"<{self.classname}{version} at 0x{id(self):012x}>"

    def __enter__(self):
        if isinstance(self._file, uproot.reading.ReadOnlyFile):
            self._file.source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if isinstance(self._file, uproot.reading.ReadOnlyFile):
            self._file.source.__exit__(exception_type, exception_value, traceback)

    @property
    def classname(self):
        """
        The C++ (decoded) classname of the modeled class.

        See :doc:`uproot.model.classname_decode`,
        :doc:`uproot.model.classname_encode`, and
        :doc:`uproot.model.classname_version`.
        """
        return classname_decode(self.encoded_classname)[0]

    @property
    def encoded_classname(self):
        """
        The Python (encoded) classname of the modeled class. May or may not
        include version.

        See :doc:`uproot.model.classname_decode`,
        :doc:`uproot.model.classname_encode`, and
        :doc:`uproot.model.classname_version`.
        """
        return type(self).__name__

    @property
    def class_version(self):
        """
        The version number of the modeled class (int) if any; None otherwise.

        See :doc:`uproot.model.classname_decode`,
        :doc:`uproot.model.classname_encode`, and
        :doc:`uproot.model.classname_version`.
        """
        return classname_decode(self.encoded_classname)[1]

    @property
    def cursor(self):
        """
        A cursor pointing to the start of this instance in the byte stream
        (before :ref:`uproot.model.Model.read_numbytes_version`).
        """
        return self._cursor

    @property
    def file(self):
        """
        A :doc:`uproot.reading.ReadOnlyFile`, which may be open and readable,
        or a :doc:`uproot.reading.DetachedFile`, which only contains
        information about the original file (not an open file handle).
        """
        return self._file

    def close(self):
        """
        Closes the file from which this object is derived, if such a file is
        still attached (i.e. not :doc:`uproot.reading.DetachedFile`).
        """
        if isinstance(self._file, uproot.reading.ReadOnlyFile):
            self._file.close()

    @property
    def closed(self):
        """
        True if the associated file is known to be closed; False if it is known
        to be open. If the associated file is detached
        (:doc:`uproot.reading.DetachedFile`), then the value is None.
        """
        if isinstance(self._file, uproot.reading.ReadOnlyFile):
            return self._file.closed
        else:
            return None

    @property
    def parent(self):
        """
        The object that was deserialized before this one in recursive descent,
        usually the containing object (or the container's container).
        """
        return self._parent

    @property
    def concrete(self):
        """
        The Python instance corresponding to the concrete (instantiated) class
        in C++, which is ``self`` if this is the concrete class or another
        object if this is actually a holder of superclass members for that other
        object (i.e. if this object is in the other's
        :ref:`uproot.model.Model.bases`).
        """
        if self._concrete is None:
            return self
        return self._concrete

    @property
    def members(self):
        """
        A dict of C++ member data directly associated with this class (i.e. not
        its superclasses). For all members, see
        :ref:`uproot.model.Model.all_members`.
        """
        return self._members

    @property
    def all_members(self):
        """
        A dict of C++ member data for this class and its superclasses. For only
        direct members, see :ref:`uproot.model.Model.members`.
        """
        out = {}
        for base in self._bases:
            out.update(base.all_members)
        out.update(self._members)
        return out

    def has_member(self, name, all=True):
        """
        Returns True if calling :ref:`uproot.model.Model.member` with the same
        arguments would return a value; False if the member is missing.
        """
        if name in self._members:
            return True
        if all:
            for base in reversed(self._bases):
                if base.has_member(name, all=True):
                    return True
        return False

    def member(self, name, all=True, none_if_missing=False):
        """
        Args:
            name (str): The name of the member datum to retrieve.
            all (bool): If True, recursively search all superclasses in
                :ref:`uproot.model.Model.bases`. Otherwise, search the
                direct class only.
            none_if_missing (bool): If a member datum doesn't exist in the
                search path, ``none_if_missing=True`` has this function return
                None, but ``none_if_missing=False`` would have it raise an
                exception. Note that None is a possible value for some member
                data.

        Returns a C++ member datum by name.
        """
        if name in self._members:
            return self._members[name]
        if all:
            for base in reversed(self._bases):
                if base.has_member(name, all=True):
                    return base.member(name, all=True)

        if none_if_missing:
            return None
        else:
            raise uproot.KeyInFileError(
                name,
                because="""{}.{} has only the following members:

    {}
""".format(
                    type(self).__module__,
                    type(self).__name__,
                    ", ".join(repr(x) for x in self.all_members),
                ),
                file_path=getattr(self._file, "file_path", None),
            )

    @property
    def bases(self):
        """
        List of :doc:`uproot.model.Model` objects representing superclass data
        for this object in the order given in C++ (opposite method resolution
        order).

        * If this object has no superclasses, ``bases`` is empty.
        * If it has one superclass, which itself might have superclasses,
          ``bases`` has length 1.
        * Only if this object *multiply inherits* from more than one superclass
          at the same level does ``bases`` have length greater than 1.

        Since multiple inheritance is usually avoided, ``bases`` rarely has
        length greater than 1. A linear chain of superclasses deriving from
        super-superclasses is represented by ``bases`` containing an object
        whose ``bases`` contains objects.
        """
        return self._bases

    def base(self, *cls):
        """
        Extracts instances from :ref:`uproot.model.Model.bases` by Python class
        type.

        The ``cls`` arguments may be Python classes or C++ classname strings to match.
        """
        cpp_names = [classname_regularize(x) for x in cls if isinstance(x, str)]
        py_types = tuple(x for x in cls if not isinstance(x, str))

        out = []
        for x in getattr(self, "_bases", []):
            if isinstance(x, py_types) or any(
                getattr(x, "classname", None) == n for n in cpp_names
            ):
                out.append(x)
            if isinstance(x, Model):
                out.extend(x.base(*cls))
        return out

    def is_instance(self, *cls):
        """
        Returns True if this object matches a given type in the C++ class hierarchy.

        The ``cls`` arguments may be Python classes or C++ classname strings to match.
        """
        cpp_names = [classname_regularize(x) for x in cls if isinstance(x, str)]
        py_types = tuple(x for x in cls if not isinstance(x, str))

        if isinstance(self, py_types) or any(self.classname == n for n in cpp_names):
            return True
        else:
            return len(self.base(*cls)) != 0

    @property
    def num_bytes(self):
        """
        Number of bytes expected in the (uncompressed) serialization of this
        instance.

        This value may be None (unknown before reading) or an integer.

        If the value is an integer and the object exists (no exceptions in
        :ref:`uproot.model.Model.read`), then the expected number of bytes
        agreed with the actual number of bytes, and this numer is reliable.

        If this object is re-serialized, it won't necessarily occupy the same
        number of bytes.
        """
        return self._num_bytes

    @property
    def instance_version(self):
        """
        Version of this instance as read from the byte stream.

        If this model is versioned (:doc:`uproot.model.VersionedModel`), the
        ``instance_version`` ought to be equal to the
        :ref:`uproot.model.Model.class_version`.

        If this model is versionless, the ``instance_version`` contains new
        information about the actual version deserialized.
        """
        return self._instance_version

    @property
    def is_memberwise(self):
        """
        True if the object was serialized in ROOT's memberwise format; False
        otherwise.
        """
        return self._is_memberwise

    @classmethod
    def awkward_form(cls, file, context):
        """
        Args:
            cls (subclass of :doc:`uproot.model.Model`): This class.
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            context (dict): Context for the Form-generation; defaults are
                the remaining arguments below.
            index_format (str): Format to use for indexes of the
                ``awkward.forms.Form``; may be ``"i32"``, ``"u32"``, or
                ``"i64"``.
            header (bool): If True, include header fields of each C++ class.
            tobject_header (bool): If True, include header fields of each ``TObject``
                base class.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be Awkward Arrays because the
                depth of instances is unknown.

        The ``awkward.forms.Form`` to use to put objects of type type in an
        Awkward Array.
        """
        raise uproot.interpretation.objects.CannotBeAwkward(
            classname_decode(cls.__name__)[0]
        )

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        """
        Args:
            cls (subclass of :doc:`uproot.model.Model`): This class.
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            header (bool): If True, assume the outermost object has a header.
            tobject_header (bool): If True, assume that ``TObjects`` have headers.
            original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): The
                original, non-strided model or container.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be strided because the
                depth of instances is unknown.

        Returns a list of (str, ``numpy.dtype``) pairs to build a
        :doc:`uproot.interpretation.objects.AsStridedObjects` interpretation.
        """
        raise uproot.interpretation.objects.CannotBeStrided(
            classname_decode(cls.__name__)[0]
        )

    def tojson(self):
        """
        Serializes this object in its ROOT JSON form (as Python lists and dicts,
        which can be passed to ``json.dump`` or ``json.dumps``).
        """
        out = {}
        for base in self._bases:
            tmp = base.tojson()
            if isinstance(tmp, dict):
                out.update(tmp)
        for k, v in self.members.items():
            if isinstance(v, Model):
                out[k] = v.tojson()
            elif isinstance(v, (numpy.number, numpy.ndarray)):
                out[k] = v.tolist()
            else:
                out[k] = v
        out["_typename"] = self.classname
        return out

    @classmethod
    def empty(cls):
        """
        Creates a model instance (of subclass ``cls``) with no data; all
        required attributes are None or empty.
        """
        self = cls.__new__(cls)
        self._cursor = None
        self._file = None
        self._parent = None
        self._concrete = None
        self._members = {}
        self._bases = []
        self._num_bytes = None
        self._instance_version = None
        self._is_memberwise = False
        return self

    @classmethod
    def read(cls, chunk, cursor, context, file, selffile, parent, concrete=None):
        """
        Args:
            cls (subclass of :doc:`uproot.model.Model`): Class to instantiate.
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot.model.Model` classes
                from its :ref:`uproot.reading.ReadOnlyFile.streamers`.
            selffile (:doc:`uproot.reading.CommonFileMethods`): A possibly
                :doc:`uproot.reading.DetachedFile` associated with this object.
            parent (None or calling object): The previous ``read`` in the
                recursive descent.
            concrete (None or :doc:`uproot.model.Model` instance): If None,
                this model corresponds to the concrete (instantiated) class in
                C++. Otherwise, this model represents a superclass part of the
                object, and ``concrete`` points to the concrete instance.

        Creates a model instance by reading data from a file.
        """
        self = cls.__new__(cls)
        self._cursor = cursor.copy()
        self._file = selffile
        self._parent = parent
        self._concrete = concrete
        self._members = {}
        self._bases = []
        self._num_bytes = None
        self._instance_version = None
        self._is_memberwise = False
        old_breadcrumbs = context.get("breadcrumbs", ())
        context["breadcrumbs"] = (*old_breadcrumbs, self)

        self.hook_before_read(chunk=chunk, cursor=cursor, context=context, file=file)
        forth_obj = uproot._awkwardforth.get_forth_obj(context)
        if forth_obj is not None:
            forth_stash = uproot._awkwardforth.Node(
                name=f"wrong-instance-version {uproot._awkwardforth.get_first_key_number(context)}",
                form_details={"offsets": "i64"},
            )
        if context.get("reading", True):
            temp_index = cursor._index
            self.read_numbytes_version(chunk, cursor, context)
            length = cursor._index - temp_index
            if length != 0 and forth_obj is not None:
                forth_stash.pre_code.append(f"{length} stream skip\n")
            if (
                issubclass(cls, VersionedModel)
                and self._instance_version != classname_version(cls.__name__)
                and self._instance_version is not None
            ):
                correct_cls = file.class_named(self.classname, self._instance_version)
                if classname_version(correct_cls.__name__) != classname_version(
                    cls.__name__
                ):
                    cursor.move_to(self._cursor.index)
                    context["breadcrumbs"] = old_breadcrumbs
                    if forth_obj is not None:
                        forth_obj.add_node(forth_stash)
                        forth_obj.push_active_node(forth_stash)
                    temp_var = correct_cls.read(
                        chunk,
                        cursor,
                        context,
                        file,
                        selffile,
                        parent,
                        concrete=concrete,
                    )
                    if forth_obj is not None:
                        forth_obj.pop_active_node()

                    return temp_var

        if context.get("in_TBranch", False):
            # AwkwardForth testing A: test_0637's 01,02,05,08,09,11,12,13,15,16,29,35,38,39,44,45,46,47,49,50,52,56
            if self._num_bytes is None and self._instance_version != self.class_version:
                self._instance_version = None
                cursor = self._cursor
                if forth_obj is not None:
                    forth_stash._pre_code.pop(-1)

            elif self._instance_version == 0:
                if forth_obj is not None:
                    forth_stash.pre_code.append("4 stream skip\n")
                cursor.skip(4)

        if context.get("reading", True):
            self.hook_before_read_members(
                chunk=chunk, cursor=cursor, context=context, file=file
            )
            if forth_obj is not None:
                key = uproot._awkwardforth.get_first_key_number(context)
                forth_stash.name = f"start-of-model {key}"
                forth_obj.add_node(forth_stash)
                forth_obj.push_active_node(forth_stash)
            self.read_members(chunk, cursor, context, file)
            if forth_obj is not None:
                forth_obj.pop_active_node()

            self.hook_after_read_members(
                chunk=chunk, cursor=cursor, context=context, file=file
            )

        self.check_numbytes(chunk, cursor, context)

        self.hook_before_postprocess(
            chunk=chunk, cursor=cursor, context=context, file=file
        )

        out = self.postprocess(chunk, cursor, context, file)

        context["breadcrumbs"] = old_breadcrumbs

        return out

    def read_numbytes_version(self, chunk, cursor, context):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.

        Reads the number of bytes and instance version from the byte stream,
        which is usually 6 bytes (4 + 2). Bits with special meanings are
        appropriately masked out.

        Some types don't have a 6-byte header or handle it differently; in
        those cases, this method should be overridden.
        """
        import uproot.deserialization

        (
            self._num_bytes,
            self._instance_version,
            self._is_memberwise,
        ) = uproot.deserialization.numbytes_version(chunk, cursor, context)

    def read_members(self, chunk, cursor, context, file):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot.model.Model` classes
                from its :ref:`uproot.reading.ReadOnlyFile.streamers`.

        Reads the member data for this class. The abstract class
        :doc:`uproot.model.Model` has an empty ``read_members`` method; this
        *must* be overridden by subclasses.
        """

    def check_numbytes(self, chunk, cursor, context):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.

        Reads nothing; checks the expected number of bytes against the actual
        movement of the ``cursor`` at the end of the object, possibly raising
        a :doc:`uproot.deserialization.DeserializationError` exception.

        If :ref:`uproot.model.Model.num_bytes` is None, this method does
        nothing.

        It is *possible* that a subclass would override this method, but not
        likely.
        """
        import uproot.deserialization

        uproot.deserialization.numbytes_check(
            chunk,
            self._cursor,
            cursor,
            self._num_bytes,
            self.classname,
            context,
            getattr(self._file, "file_path", None),
        )

    def postprocess(self, chunk, cursor, context, file):
        """
        Args:
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot.model.Model` classes
                from its :ref:`uproot.reading.ReadOnlyFile.streamers`.

        Called for any additional processing after the object has been fully
        read.

        The return value from this method is the object that actually represents
        the ROOT data, which might be a different instance or even a different
        type from this class. The default in :doc:`uproot.model.Model` is to
        return ``self``.

        Note that for versioned models,
        :ref:`uproot.model.VersionedModel.postprocess` is called first, then
        :ref:`uproot.model.DispatchByVersion.postprocess` is called on its
        output, allowing a :doc:`uproot.model.DispatchByVersion` to refine all
        data of its type, regardless of version.
        """
        return self

    def hook_before_read(self, **kwargs):
        """
        Called in :ref:`uproot.model.Model.read`, before any data have been
        read.
        """

    def hook_before_read_members(self, **kwargs):
        """
        Called in :ref:`uproot.model.Model.read`, after
        :ref:`uproot.model.Model.read_numbytes_version` and before
        :ref:`uproot.model.Model.read_members`.
        """

    def hook_after_read_members(self, **kwargs):
        """
        Called in :ref:`uproot.model.Model.read`, after
        :ref:`uproot.model.Model.read_members` and before
        :ref:`uproot.model.Model.check_numbytes`.
        """

    def hook_before_postprocess(self, **kwargs):
        """
        Called in :ref:`uproot.model.Model.read`, after
        :ref:`uproot.model.Model.check_numbytes` and before
        :ref:`uproot.model.Model.postprocess`.
        """

    def _to_writable_postprocess(self, original):
        pass

    def _to_writable(self, concrete):
        cls = None
        if self.writable:
            cls = type(self)
        else:
            unversioned = uproot.classes.get(self.classname)
            if unversioned is not None and issubclass(unversioned, DispatchByVersion):
                for versioned_cls in unversioned.known_versions.values():
                    if versioned_cls.writable:
                        cls = versioned_cls
                        break

            elif unversioned is not None and unversioned.writable:
                cls = unversioned

        if cls is None:
            raise NotImplementedError(
                f"this ROOT type is not writable: {self.classname}"
            )
        else:
            out = cls.__new__(cls)
            out._cursor = self._cursor
            out._file = self._file
            out._parent = self._parent
            out._concrete = concrete
            out._num_bytes = self._num_bytes
            out._instance_version = classname_decode(cls.__name__)[1]
            out._is_memberwise = self._is_memberwise

            if concrete is None:
                concrete = out

            out._bases = []
            for base in self._bases:
                out._bases.append(base._to_writable(concrete))

            out._members = {}
            for key, value in self._members.items():
                if isinstance(value, Model):
                    out._members[key] = value._to_writable(None)
                else:
                    out._members[key] = value

            out._to_writable_postprocess(self)
            return out

    def to_writable(self):
        """
        Args:
            obj (:doc:`uproot.model.Model` instance of the same C++ class): The
                object to convert to this class version.

        Returns a writable version of this object or raises a NotImplementedError
        if no writable version exists.
        """
        if self._deeply_writable:
            return self
        else:
            return self._to_writable(None)

    def _serialize(self, out, header, name, tobject_flags):
        raise NotImplementedError(
            f"can't write {type(self).__name__} instances yet ('serialize' method not implemented)"
        )

    def serialize(self, name=None):
        """
        Serialize a object (from num_bytes and version onward) for writing into
        an output ROOT file.

        If a ``name`` is given, override the object's current name.

        This method has not been implemented on all classes (raises
        NotImplementedError).
        """
        out = []
        self._serialize(out, True, name, numpy.uint32(0x00000000))
        return b"".join(out)

    def to_pyroot(self, name=None):
        """
        Args:
            name (str or None): A name for the new PyROOT object.

        Converts this :doc:`uproot.model.Model` into a PyROOT object *if it is writable*.
        A minority of Uproot models are writable, mostly just histograms. Writability
        is necessary for conversion to PyROOT because it is serialized through a
        ROOT TMessage.
        """
        return uproot.pyroot.to_pyroot(self, name=name)


class VersionedModel(Model):
    """
    A Python class that models a specific version of a ROOT C++ class.

    Classes that inherit directly from :doc:`uproot.model.Model` are versionless,
    classes that inherit from :doc:`uproot.model.VersionedModel` depend on
    version.

    Note that automatically generated :doc:`uproot.model.VersionedModel` classes
    are placed in the ``uproot.dynamic`` namespace. This namespace can generate
    :doc:`uproot.model.DynamicModel` classes on demand in Python 3.7 and above,
    which automatically generated :doc:`uproot.model.VersionedModel` classes
    rely upon to be pickleable. Therefore, ROOT object types without predefined
    :doc:`uproot.model.Model` classes cannot be pickled in Python versions
    before 3.7.
    """

    def __getstate__(self):
        return (
            {
                "base_names_versions": self.base_names_versions,
                "member_names": self.member_names,
                "class_flags": self.class_flags,
                "class_code": self.class_code,
                "class_streamer": self.class_streamer,
                "class_rawstreamers": self.class_rawstreamers,
                "writable": self.writable,
                "behaviors": self.behaviors,
            },
            {
                k: (
                    _LockPlaceholder()
                    if isinstance(v, _LockPlaceholder.lock_type)
                    else v
                )
                for k, v in self.__dict__.items()
            },
        )

    def __setstate__(self, state):
        _class_data, instance_data = state
        instance_data = {
            k: threading.Lock() if isinstance(v, _LockPlaceholder) else v
            for k, v in instance_data.items()
        }
        self.__dict__.update(instance_data)


class _LockPlaceholder:
    lock_type = type(threading.Lock())


class DispatchByVersion:
    """
    A Python class that models all versions of a ROOT C++ class by maintaining
    a dict of :doc:`uproot.model.VersionedModel` classes.

    The :ref:`uproot.model.DispatchByVersion.read` classmethod reads the
    instance version number from the byte stream, backs up the
    :doc:`uproot.source.cursor.Cursor` to the starting position, and invokes
    the appropriate :doc:`uproot.model.VersionedModel`'s ``read`` classmethod.

    If a :doc:`uproot.model.VersionedModel` does not exist for the specified
    version, the ``file``'s ``TStreamerInfo`` is queried to attempt to create
    one, and failing that, an :doc:`uproot.model.UnknownClassVersion` is
    created instead.

    Note that :doc:`uproot.model.DispatchByVersion` is not a subclass of
    :doc:`uproot.model.Model`. Instances of this class are not usable as
    stand-ins for ROOT data.
    """

    @classmethod
    def awkward_form(cls, file, context):
        """
        Args:
            cls (subclass of :doc:`uproot.model.DispatchByVersion`): This class.
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            context (dict): Context for the Form-generation; defaults are
                ``{"index_format": "i64", "header": False, "tobject_header": True, "breadcrumbs": ()}``.
                See below for context argument descriptions.
            index_format (str): Format to use for indexes of the
                ``awkward.forms.Form``; may be ``"i32"``, ``"u32"``, or
                ``"i64"``.
            header (bool): If True, include headers in the Form's ``"uproot"``
                parameters.
            tobject_header (bool): If True, include headers for ``TObject``
                classes in the Form's ``"uproot"`` parameters.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be Awkward Arrays because the
                depth of instances is unknown.

        The ``awkward.forms.Form`` to use to put objects of type type in an
        Awkward Array.
        """
        versioned_cls = file.class_named(classname_decode(cls.__name__)[0], "max")
        return versioned_cls.awkward_form(file, context)

    @classmethod
    def strided_interpretation(
        cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None
    ):
        """
        Args:
            cls (subclass of :doc:`uproot.model.DispatchByVersion`): This class.
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.
            header (bool): If True, assume the outermost object has a header.
            tobject_header (bool): If True, assume that ``TObjects`` have headers.
            original (None, :doc:`uproot.model.Model`, or :doc:`uproot.containers.Container`): The
                original, non-strided model or container.
            breadcrumbs (tuple of class objects): Used to check for recursion.
                Types that contain themselves cannot be strided because the
                depth of instances is unknown.

        Returns a list of (str, ``numpy.dtype``) pairs to build a
        :doc:`uproot.interpretation.objects.AsStridedObjects` interpretation.
        """
        versioned_cls = file.class_named(classname_decode(cls.__name__)[0], "max")
        return versioned_cls.strided_interpretation(
            file, header=header, tobject_header=tobject_header, breadcrumbs=breadcrumbs
        )

    @classmethod
    def class_of_version(cls, version):
        """
        Returns the class corresponding to a specified ``version`` if it exists.

        If not, this classmethod returns None. No attempt is made to create a
        missing class.
        """
        out = cls.known_versions.get(version)
        if out is None and version == 0 and len(cls.known_versions) != 0:
            out = cls.known_versions[max(cls.known_versions)]
        return out

    @classmethod
    def has_version(cls, version):
        """
        Returns True if a class corresponding to a specified ``version``
        currently exists; False otherwise.
        """
        return version in cls.known_versions

    @classmethod
    def new_class(cls, file, version):
        """
        Uses ``file`` to create a new class for a specified ``version``.

        As a side-effect, this new class is added to ``cls.known_versions``
        (for :ref:`uproot.model.DispatchByVersion.class_of_version` and
        :ref:`uproot.model.DispatchByVersion.has_version`).

        If the ``file`` lacks a ``TStreamerInfo`` for the class, this function
        returns a :doc:`uproot.model.UnknownClassVersion` (adding it to
        ``uproot.unknown_classes`` if it's not already there).
        """
        classname, _ = classname_decode(cls.__name__)
        classname = classname_regularize(classname)
        streamer = file.streamer_named(classname, version)

        if streamer is None:
            streamer = file.streamer_named(classname, "max")

        if streamer is None and file.custom_classes is not None:
            cls2 = uproot.classes.get(classname)
            versioned_cls2 = cls2.class_of_version(version)
            if versioned_cls2 is not None:
                return versioned_cls2

        if streamer is not None:
            versioned_cls = streamer.new_class(file)
            versioned_cls.class_streamer = streamer
            cls.known_versions[streamer.class_version] = versioned_cls
            return versioned_cls

        else:
            return unknown_class(classname, version)

    @classmethod
    def read(cls, chunk, cursor, context, file, selffile, parent, concrete=None):
        """
        Args:
            cls (subclass of :doc:`uproot.model.DispatchByVersion`): This class.
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot.model.Model` classes
                from its :ref:`uproot.reading.ReadOnlyFile.streamers`.
            selffile (:doc:`uproot.reading.CommonFileMethods`): A possibly
                :doc:`uproot.reading.DetachedFile` associated with this object.
            parent (None or calling object): The previous ``read`` in the
                recursive descent.
            concrete (None or :doc:`uproot.model.Model` instance): If None,
                this model corresponds to the concrete (instantiated) class in
                C++. Otherwise, this model represents a superclass part of the
                object, and ``concrete`` points to the concrete instance.

        Reads the instance version number from the byte stream, backs up the
        :doc:`uproot.source.cursor.Cursor` to the starting position, and
        invokes the appropriate :doc:`uproot.model.VersionedModel`'s ``read``
        classmethod.

        If a :doc:`uproot.model.VersionedModel` does not exist for the
        specified version, the ``file``'s ``TStreamerInfo`` is queried to
        attempt to create one, and failing that, an
        :doc:`uproot.model.UnknownClassVersion` is created instead.
        """
        import uproot.deserialization

        # Ignores context["reading"], because otherwise, there would be nothing to do.
        start_index = cursor._index
        (
            _num_bytes,
            version,
            _is_memberwise,
        ) = uproot.deserialization.numbytes_version(chunk, cursor, context, move=False)

        versioned_cls = cls.class_of_version(version)

        forth_obj = uproot._awkwardforth.get_forth_obj(context)
        if forth_obj is not None:
            forth_stash = uproot._awkwardforth.Node(
                f"dispatch-by-version {uproot._awkwardforth.get_first_key_number(context)}",
                form_details={"offsets": "i64"},
            )
            bytes_skipped = cursor._index - start_index
            forth_stash.pre_code.append(f"{bytes_skipped} stream skip \n")

            forth_obj.add_node(forth_stash)
            forth_obj.push_active_node(forth_stash)

        # numbytes_version always returns an integer version, so if there is no
        # known versioned class for it, one is generated for that version.
        if versioned_cls is None:
            versioned_cls = cls.new_class(file, version)

        # versioned_cls.read starts with numbytes_version again because move=False (above)
        temp_var = cls.postprocess(
            versioned_cls.read(
                chunk, cursor, context, file, selffile, parent, concrete=concrete
            ),
            chunk,
            cursor,
            context,
            file,
        )

        if forth_obj is not None:
            forth_obj.pop_active_node()

        return temp_var

    @classmethod
    def postprocess(cls, self, chunk, cursor, context, file):
        """
        Args:
            cls (subclass of :doc:`uproot.model.DispatchByVersion`): This class.
            chunk (:doc:`uproot.source.chunk.Chunk`): Buffer of contiguous data
                from the file :doc:`uproot.source.chunk.Source`.
            cursor (:doc:`uproot.source.cursor.Cursor`): Current position in
                that ``chunk``.
            context (dict): Auxiliary data used in deserialization.
            file (:doc:`uproot.reading.ReadOnlyFile`): An open file object,
                capable of generating new :doc:`uproot.model.Model` classes
                from its :ref:`uproot.reading.ReadOnlyFile.streamers`.

        Called for any additional processing after the object has been fully
        read.

        The return value from this method is the object that actually represents
        the ROOT data, which might be a different instance or even a different
        type from this class. The default in :doc:`uproot.model.Model` is to
        return ``self``.

        Note that for versioned models,
        :ref:`uproot.model.VersionedModel.postprocess` is called first, then
        :ref:`uproot.model.DispatchByVersion.postprocess` is called on its
        output, allowing a :doc:`uproot.model.DispatchByVersion` to refine all
        data of its type, regardless of version.
        """
        return self


class UnknownClass(Model):
    """
    Placeholder for a C++ class instance that has no
    :doc:`uproot.model.DispatchByVersion` and no ``TStreamerInfo`` in the
    current :doc:`uproot.reading.ReadOnlyFile` to produce one.
    """

    @property
    def chunk(self):
        """
        The ``chunk`` of data associated with the unknown class, referred to by
        a weak reference (to avoid memory leaks in
        :doc:`uproot.model.UnknownClass` objects). If the original ``chunk``
        has been garbage-collected, this raises ``RuntimeError``.

        Primarily useful in the :ref:`uproot.model.UnknownClass.debug` method.
        """
        chunk = self._chunk()
        if chunk is None:
            raise RuntimeError(
                "the 'chunk' associated with this unknown class has been deleted"
            )
        else:
            return chunk

    @property
    def context(self):
        """
        The auxiliary data used in deserialization.

        Primarily useful in the :ref:`uproot.model.UnknownClass.debug` method.
        """
        return self._context

    def __repr__(self):
        return f"<Unknown {self.classname} at 0x{id(self):012x}>"

    def debug(
        self, skip_bytes=0, limit_bytes=None, dtype=None, offset=0, stream=sys.stdout
    ):
        """
        Args:
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the :doc:`uproot.source.chunk.Chunk`. May be
                negative, to examine the byte stream leading up to the attempted
                instantiation. The default, ``0``, starts where the number
                of bytes and version number would be (just before
                :ref:`uproot.model.Model.read_numbytes_version`).
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

        Presents the byte stream at the point where this instance would have been
        deserialized.

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
        cursor = self._cursor.copy()
        cursor.skip(skip_bytes)
        cursor.debug(
            self.chunk,
            context=self._context,
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
                instantiation. The default, ``0``, starts where the number
                of bytes and version number would be (just before
                :ref:`uproot.model.Model.read_numbytes_version`).
            dtype (``numpy.dtype`` or its constructor argument): Data type in
                which to interpret the data. (The size of the array returned is
                truncated to this ``dtype.itemsize``.)

        Like :ref:`uproot.model.UnknownClass.debug`, but returns a NumPy array
        for further inspection.
        """
        dtype = numpy.dtype(dtype)
        cursor = self._cursor.copy()
        cursor.skip(skip_bytes)
        out = self.chunk.remainder(cursor.index, cursor, self._context)
        return out[: (len(out) // dtype.itemsize) * dtype.itemsize].view(dtype)

    def read_members(self, chunk, cursor, context, file):
        self._chunk = weakref.ref(chunk)
        self._context = context

        if self._num_bytes is not None:
            cursor.skip(self._num_bytes - cursor.displacement(self._cursor))

        else:
            raise ValueError(
                f"""unknown class {self.classname} that cannot be skipped because its """
                f"""number of bytes is unknown
in file {file.file_path}"""
            )


class UnknownClassVersion(VersionedModel):
    """
    Placeholder for a C++ class instance that has no ``TStreamerInfo`` in the
    current :doc:`uproot.reading.ReadOnlyFile` to produce one.
    """

    @property
    def chunk(self):
        """
        The ``chunk`` of data associated with the class of unknown version,
        referred to by a weak reference (to avoid memory leaks in
        :doc:`uproot.model.UnknownClassVersion` objects). If the original
        ``chunk`` has been garbage-collected, this raises ``RuntimeError``.

        Primarily useful in the :ref:`uproot.model.UnknownClassVersion.debug`
        method.
        """
        chunk = self._chunk()
        if chunk is None:
            raise RuntimeError(
                "the 'chunk' associated with this class of unknown version has "
                "been deleted"
            )
        else:
            return chunk

    @property
    def context(self):
        """
        The auxiliary data used in deserialization.

        Primarily useful in the :ref:`uproot.model.UnknownClass.debug` method.
        """
        return self._context

    def debug(
        self, skip_bytes=0, limit_bytes=None, dtype=None, offset=0, stream=sys.stdout
    ):
        """
        Args:
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the :doc:`uproot.source.chunk.Chunk`. May be
                negative, to examine the byte stream leading up to the attempted
                instantiation. The default, ``0``, starts where the number
                of bytes and version number would be (just before
                :ref:`uproot.model.Model.read_numbytes_version`).
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

        Presents the byte stream at the point where this instance would have been
        deserialized.

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
        cursor = self._cursor.copy()
        cursor.skip(skip_bytes)
        cursor.debug(
            self.chunk,
            context=self._context,
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
                instantiation. The default, ``0``, starts where the number
                of bytes and version number would be (just before
                :ref:`uproot.model.Model.read_numbytes_version`).
            dtype (``numpy.dtype`` or its constructor argument): Data type in
                which to interpret the data. (The size of the array returned is
                truncated to this ``dtype.itemsize``.)

        Like :ref:`uproot.model.UnknownClassVersion.debug`, but returns a
        NumPy array for further inspection.
        """
        dtype = numpy.dtype(dtype)
        cursor = self._cursor.copy()
        cursor.skip(skip_bytes)
        out = self.chunk.remainder(cursor.index, cursor, self._context)
        return out[: (len(out) // dtype.itemsize) * dtype.itemsize].view(dtype)

    def read_members(self, chunk, cursor, context, file):
        self._chunk = weakref.ref(chunk)
        self._context = context

        if self._num_bytes is not None:
            cursor.skip(self._num_bytes - cursor.displacement(self._cursor))

        else:
            raise ValueError(
                f"""class {self.classname} with unknown version {self._instance_version} cannot be skipped """
                f"""because its number of bytes is unknown
in file {file.file_path}"""
            )

    def __repr__(self):
        return f"<{self.classname} with unknown version {self._instance_version} at 0x{id(self):012x}>"


class DynamicModel(VersionedModel):
    """
    A :doc:`uproot.model.VersionedModel` subclass generated by any attempt to
    extract it from the ``uproot.dynamic`` namespace in Python 3.7 and later.

    This dynamically generated model allows ROOT object types without predefined
    :doc:`uproot.model.Model` classes to be pickled in Python 3.7 and later.
    """

    def __setstate__(self, state):
        cls = type(self)
        class_data, instance_data = state
        for k, v in class_data.items():
            if k not in cls.__dict__:
                setattr(cls, k, v)
        cls.__bases__ = (
            tuple(x for x in class_data["behaviors"] if x not in cls.__bases__)
            + cls.__bases__
        )
        self.__dict__.update(instance_data)
