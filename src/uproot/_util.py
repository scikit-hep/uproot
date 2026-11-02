# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines utilities for internal use. This is not a public interface
and may be changed without notice.
"""

from __future__ import annotations

import datetime
import glob
import itertools
import numbers
import os
import re
import sys
import urllib.parse
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import IO

import awkward
import fsspec
import numpy
import packaging.version

import uproot.source.chunk
import uproot.source.fsspec
import uproot.source.object

wasm = sys.platform in ("emscripten", "wasi")


def isint(x) -> bool:
    """
    Returns True if and only if ``x`` is an integer (including NumPy, not
    including bool).
    """
    return isinstance(x, (int, numbers.Integral, numpy.integer)) and not isinstance(
        x, (bool, numpy.bool_)
    )


def isnum(x) -> bool:
    """
    Returns True if and only if ``x`` is a number (including NumPy, not
    including bool).
    """
    return isinstance(x, (int, float, numbers.Real, numpy.number)) and not isinstance(
        x, (bool, numpy.bool_)
    )


def ensure_str(x) -> str:
    """
    Ensures that ``x`` is a string (decoding with 'surrogateescape' if necessary).
    """
    if isinstance(x, bytes):
        return x.decode(errors="surrogateescape")
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f"expected a string, not {type(x)}")


def ensure_numpy(array, types=(numpy.bool_, numpy.integer, numpy.floating)):
    """
    Returns an ``np.ndarray`` if ``array`` can be converted to an array of the
    desired type and raises TypeError if it cannot.
    """

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error", getattr(numpy, "exceptions", numpy).VisibleDeprecationWarning
        )
        if isinstance(array, awkward.contents.Content):
            out = awkward.to_numpy(array)
        else:
            try:
                out = numpy.asarray(array)
            except (
                ValueError,
                getattr(numpy, "exceptions", numpy).VisibleDeprecationWarning,
            ) as err:
                raise TypeError("cannot be converted to a NumPy array") from err
        if not issubclass(out.dtype.type, types):
            raise TypeError(f"cannot be converted to a NumPy array of type {types}")
        return out


def is_file_like(
    obj, readable: bool = False, writable: bool = False, seekable: bool = False
) -> bool:
    return (
        all(
            callable(getattr(obj, attr, None))
            for attr in ("read", "write", "seek", "tell", "flush")
        )
        and (not readable or not hasattr(obj, "readable") or obj.readable())
        and (not writable or not hasattr(obj, "writable") or obj.writable())
        and (not seekable or not hasattr(obj, "seekable") or obj.seekable())
    )


def parse_version(version: str):
    """
    Converts a semver string into a Version object that can be compared with
    ``<``, ``>=``, etc.

    Currently implemented using ``packaging.Version``
    (exposing that library in the return type).
    """
    return packaging.version.parse(version)


def from_module(obj, module_name: str) -> bool:
    """
    Returns True if ``obj`` is an instance of a class from a module
    given by name.

    This is like ``isinstance`` (in that it searches the whole ``mro``),
    except that the module providing the type to check against doesn't
    have to be imported and doesn't get imported (as a side effect) by
    this function.
    """
    try:
        mro = type(obj).mro()
    except TypeError:
        return False

    for t in mro:
        if t.__module__ == module_name or t.__module__.startswith(module_name + "."):
            return True
    return False


def _regularize_filter_regex_flags(flags):
    flagsbyte = 0
    for flag in flags:
        if flag == "i":
            flagsbyte += re.IGNORECASE
        elif flag == "L":
            flagsbyte += re.LOCALE
        elif flag == "m":
            flagsbyte += re.MULTILINE
        elif flag == "s":
            flagsbyte += re.DOTALL
        elif flag == "u":
            flagsbyte += re.UNICODE
        elif flag == "x":
            flagsbyte += re.VERBOSE
    return flagsbyte


def no_filter(x) -> bool:
    """
    A filter that accepts anything (always returns True).
    """
    return True


_regularize_filter_regex = re.compile("^/(.*)/([iLmsux]*)$")


def regularize_filter(filter):
    """
    Convert None, str, iterable of str, wildcards, and regular expressions into
    the standard form for a filter: a callable returning True or False.
    """
    if filter is None:
        return no_filter
    elif callable(filter):
        return filter
    elif isinstance(filter, str):
        m = _regularize_filter_regex.match(filter)
        if m is not None:
            regex, flags = m.groups()
            matcher = re.compile(regex, _regularize_filter_regex_flags(flags))
            return lambda x: matcher.match(x) is not None
        elif "*" in filter or "?" in filter or "[" in filter:
            return lambda x: glob.fnmatch.fnmatchcase(x, filter)
        else:
            return lambda x: x == filter
    elif isinstance(filter, Iterable) and not isinstance(filter, bytes):
        filters = [regularize_filter(f) for f in filter]
        return lambda x: any(f(x) for f in filters)
    else:
        raise TypeError(
            "filter must be None, callable, a regex string between slashes, or a "
            f"glob pattern, not {filter!r}"
        )


def no_rename(x):
    """
    A renaming function that keeps all names the same (identity function).
    """
    return x


_regularize_filter_regex_rename = re.compile("^s?/(.*)/(.*)/([iLmsux]*)$")


def regularize_rename(rename):
    """
    Convert None, dict, and regular expression mappings into the standard form
    for renaming: a callable that maps strings to strings.
    """
    if rename is None:
        return no_rename

    elif callable(rename):
        return rename

    elif isinstance(rename, str):
        m = _regularize_filter_regex_rename.match(rename)
        if m is not None:
            regex, trans, flags = m.groups()
            matcher = re.compile(regex, _regularize_filter_regex_flags(flags))
            return lambda x: matcher.sub(trans, x)
        else:
            raise TypeError("rename regular expressions must be in '/from/to/' form")

    elif isinstance(rename, dict):
        return lambda x: rename.get(x, x)

    elif isinstance(rename, Iterable) and not isinstance(rename, bytes):
        rules = []
        for x in rename:
            if isinstance(x, str):
                m = _regularize_filter_regex_rename.match(x)
                if m is not None:
                    regex, trans, flags = m.groups()
                    rules.append(
                        (
                            re.compile(regex, _regularize_filter_regex_flags(flags)),
                            trans,
                        )
                    )
                else:
                    raise TypeError(
                        "rename regular expressions must be in '/from/to/' form"
                    )
            else:
                break
        else:

            def applyrules(x):
                for matcher, trans in rules:
                    if matcher.search(x) is not None:
                        return matcher.sub(trans, x)
                else:
                    return x

            return applyrules

    raise TypeError(
        "rename must be None, callable, a '/from/to/' regex, an iterable of "
        f"regex rules, or a dict, not {rename!r}"
    )


_fix_url_path = re.compile(r"^((file|https?|root):/)([^/])", re.IGNORECASE)
_object_in_path_regex = re.compile(r"(.+\.root(\.[0-9]+)?):(.*$)", re.IGNORECASE)
_memory_size_regex = re.compile(
    r"^\s*([+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?)\s*([kmgtpezy]?[i]?b)\s*$",
    re.IGNORECASE,
)


def regularize_path(path):
    """
    Converts pathlib Paths and other os.PathLike objects into plain string paths.
    """
    if isinstance(path, os.PathLike):
        path = _fix_url_path.sub(r"\1/\3", os.fspath(path))

    return path


def is_remote_url(file_path: str) -> bool:
    """
    Returns True if file_path is a URL with a remote scheme (not file or local).
    """
    if "://" not in file_path:
        return False
    parsed = urllib.parse.urlsplit(file_path)
    return parsed.scheme not in ("", "file", "local")


def _normalize_remote_url(file_path: str) -> str:
    # fsspec's own normal form keeps each protocol's semantics: object stores and HTTP servers
    # can hold different objects at "a//b" and "a/b", so no POSIX normalization on top of it
    try:
        fs, path = fsspec.core.url_to_fs(file_path)
        return fs.unstrip_protocol(path)
    except Exception:
        return file_path


def canonical_path(file_path, seen_inodes=None):
    """
    Returns a canonical identity key for a file path or URL for deduplication; it is a key, not
    a path to open. Local paths and file:// URLs are normalized via ``os.path.realpath``; remote
    URLs keep fsspec's normal form for their protocol. With ``seen_inodes``, a dict shared
    across calls, local files that are the same inode (hard links, case-insensitive spellings)
    get the same key.
    """
    if not isinstance(file_path, (str, bytes, os.PathLike)):
        return file_path

    if isinstance(file_path, os.PathLike):
        file_path = os.fspath(file_path)
    if isinstance(file_path, bytes):
        file_path = os.fsdecode(file_path)

    if is_remote_url(file_path):
        return _normalize_remote_url(file_path)

    parsed = urllib.parse.urlsplit(file_path)
    if parsed.scheme in ("file", "local"):
        if parsed.netloc and parsed.netloc != "localhost":
            if (
                sys.platform == "win32"
                and len(parsed.netloc) >= 2
                and parsed.netloc[1] == ":"
            ):
                from urllib.request import url2pathname

                file_path = url2pathname(parsed.netloc + parsed.path)
            else:
                return file_path  # another host's file: its path semantics are not ours
        else:
            from urllib.request import url2pathname

            file_path = url2pathname(parsed.path)

    expanded = os.path.expanduser(file_path)
    resolved = os.path.realpath(expanded)

    if seen_inodes is not None:
        try:
            st = os.stat(resolved)
            key = (st.st_dev, st.st_ino)
            if key in seen_inodes:
                return seen_inodes[key]
            seen_inodes[key] = resolved
        except OSError:
            pass

    return resolved


def canonical_object_path(object_path):
    """
    Returns a canonical identity key for an object path inside a ROOT file for deduplication; it
    is a key, not a path to read. Leading and trailing slashes and the ``;cycle`` suffix are
    dropped, so cycles of one name share a key; ``None`` or an empty path, which names the file's
    only ``TTree``, is ``""``.
    """
    return (object_path or "").strip().strip("/").partition(";")[0].rstrip("/")


def file_object_path_split(urlpath: str) -> tuple[str, str | None]:
    """
    Split a path with a colon into a file path and an object-in-file path.

    Args:
        urlpath: The path to split. Example: ``"https://localhost:8888/file.root:tree"``

    Returns:
        A tuple of the file path and the object-in-file path. If there is no
        object-in-file path, the second element is ``None``.
        Example: ``("https://localhost:8888/file.root", "tree")``
    """

    urlpath: str = regularize_path(urlpath).strip()
    obj = None

    separator = "::"
    parts = urlpath.split(separator)
    for i, part in enumerate(reversed(parts)):
        match = _object_in_path_regex.match(part)
        if match:
            obj = re.sub(r"/+", "/", match.group(3).strip().lstrip("/")).rstrip("/")
            parts[-i - 1] = match.group(1)
            break

    urlpath = separator.join(parts)
    return urlpath, obj


def fsspec_storage_options(options: dict) -> dict:
    """
    Returns the subset of uproot ``options`` that should be forwarded to fsspec:
    everything that is not an uproot option, plus ``timeout`` if it is not None.
    """
    from uproot.reading import open as uproot_open

    return {
        key: value
        for key, value in options.items()
        if (key not in uproot_open.defaults) or (key == "timeout" and value is not None)
    }


def _edit_distance(a: str, b: str) -> int:
    # optimal string alignment distance: insertions, deletions, substitutions,
    # and transpositions of adjacent characters
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            rows[i][j] = min(
                rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[len(a)][len(b)]


def _check_unknown_options(options: dict, file_path=None, stacklevel: int = 3) -> None:
    """
    Raise on unrecognized options for uproot's own sources and warn about likely
    misspellings of uproot options for fsspec, whose storage options are open-ended.
    Handlers that are not uproot Sources are skipped here and rejected later by
    ``file_path_to_source_class``.

    This must be called directly from a public entry point, so that the default
    ``stacklevel`` points at the user's code.
    """
    from uproot.reading import open as uproot_open

    known = set(uproot_open.defaults)
    unknown = [key for key in options if key not in known]
    if len(unknown) == 0:
        return

    source_cls = options.get("handler")
    if source_cls is None:
        if file_path is not None and not isinstance(file_path, str):
            source_cls = uproot.source.object.ObjectSource
        else:
            source_cls = uproot.source.fsspec.FSSpecSource
    if not (
        isinstance(source_cls, type)
        and issubclass(source_cls, uproot.source.chunk.Source)
    ):
        return

    if issubclass(source_cls, uproot.source.fsspec.FSSpecSource):
        for key in unknown:
            matches = [k for k in sorted(known) if _edit_distance(key, k) <= 2]
            if matches:
                warnings.warn(
                    f"option {key!r} is passed to fsspec as a storage option; "
                    f"did you mean the uproot option {matches[0]!r}?",
                    stacklevel=stacklevel,
                )
    else:
        raise TypeError(
            f"unrecognized option(s) for {source_cls.__name__}: "
            f"{', '.join(map(repr, unknown))}; valid options are {sorted(known)}"
        )


def file_path_to_source_class(
    file_path_or_object: str | Path | IO, options: dict
) -> tuple[type[uproot.source.chunk.Source], str | IO]:
    """
    Use a file path to get the :doc:`uproot.source.chunk.Source` class that would read it.

    Returns a tuple of (class, file_path) where the class is a subclass of :doc:`uproot.source.chunk.Source`.
    """

    file_path_or_object: str | IO = regularize_path(file_path_or_object)

    source_cls = options["handler"]
    if source_cls is not None and not (
        isinstance(source_cls, type)
        and issubclass(source_cls, uproot.source.chunk.Source)
    ):
        raise TypeError(
            f"'handler' is not a class object inheriting from Source: {source_cls!r}"
        )

    # Infer the source class from the file path
    if all(
        callable(getattr(file_path_or_object, attr, None)) for attr in ("read", "seek")
    ):
        # need a very soft object check for ubuntu python3.8 pyroot ci tests, cannot use uproot._util.is_file_like
        if (
            source_cls is not None
            and source_cls is not uproot.source.object.ObjectSource
        ):
            raise TypeError(
                f"'handler' is not ObjectSource for a file-like object: {source_cls!r}"
            )
        return uproot.source.object.ObjectSource, file_path_or_object
    elif isinstance(file_path_or_object, str):
        source_cls = (
            uproot.source.fsspec.FSSpecSource if source_cls is None else source_cls
        )
        return source_cls, file_path_or_object
    else:
        raise TypeError(
            f"file_path is not a string or file-like object: {file_path_or_object!r}"
        )


def _file_not_found(files, message=None):
    message = "" if message is None else " (" + message + ")"

    return FileNotFoundError(f"""file not found{message}

    {files!r}

Files may be specified as:
   * str/bytes: relative or absolute filesystem path or URL, without any colons
         other than Windows drive letter or URL schema.
         Examples: "rel/file.root", "C:\\abs\\file.root", "http://where/what.root"
   * str/bytes: same with an object-within-ROOT path, separated by a colon.
         Example: "rel/file.root:tdirectory/ttree"
   * pathlib.Path: always interpreted as a filesystem path or URL only (no
         object-within-ROOT path), regardless of whether there are any colons.
         Examples: Path("rel:/file.root"), Path("/abs/path:stuff.root")

Functions that accept many files (uproot.iterate, etc.) also allow:
   * glob syntax in str/bytes and pathlib.Path.
         Examples: Path("rel/*.root"), "/abs/*.root:tdirectory/ttree"
   * dict: keys are filesystem paths, values are objects-within-ROOT paths.
         Example: {{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}
   * already-open TTree objects.
   * iterables of the above.
""")


def memory_size(data, error_message=None) -> int:
    """
    Regularizes strings like '## kB' and plain integer number of bytes to
    an integer number of bytes.
    """
    if isinstance(data, str):
        m = _memory_size_regex.match(data)
        if m is not None:
            target, unit = float(m.group(1)), m.group(5).upper()
            if unit == "KB":
                target *= 1000
            elif unit == "MB":
                target *= 1000**2
            elif unit == "GB":
                target *= 1000**3
            elif unit == "TB":
                target *= 1000**4
            elif unit == "PB":
                target *= 1000**5
            elif unit == "EB":
                target *= 1000**6
            elif unit == "ZB":
                target *= 1000**7
            elif unit == "YB":
                target *= 1000**8
            elif unit == "KIB":
                target *= 1024
            elif unit == "MIB":
                target *= 1024**2
            elif unit == "GIB":
                target *= 1024**3
            elif unit == "TIB":
                target *= 1024**4
            elif unit == "PIB":
                target *= 1024**5
            elif unit == "EIB":
                target *= 1024**6
            elif unit == "ZIB":
                target *= 1024**7
            elif unit == "YIB":
                target *= 1024**8
            return int(target)

    if isint(data):
        return int(data)

    if error_message is None:
        raise TypeError(
            "number of bytes or memory size string with units "
            f"(such as '100 MB') required, not {data!r}"
        )
    else:
        raise TypeError(error_message)


def check_step_size(step_size) -> None:
    """
    Raises an error unless ``step_size`` is a positive number of entries or a
    positive memory size string with units (such as ``"100 MB"``).
    """
    message = (
        "step_size must be a positive number of entries or a memory size "
        f"string with units (such as '100 MB'), not {step_size!r}"
    )
    if isint(step_size):
        if step_size < 1:
            raise ValueError(message)
    elif memory_size(step_size, message) <= 0:
        raise ValueError(message)


def trim_final(basket_arrays, entry_start, entry_stop, entry_offsets, library, branch):
    """
    Trims the output from a basket_array function and outputs the un-concatenated list of elements.
    """
    trimmed = []
    start = entry_offsets[0]
    for basket_num, stop in enumerate(entry_offsets[1:]):
        if start <= entry_start and entry_stop <= stop:
            local_start = entry_start - start
            local_stop = entry_stop - start
            trimmed.append(basket_arrays[basket_num][local_start:local_stop])

        elif start <= entry_start < stop:
            local_start = entry_start - start
            local_stop = stop - start
            trimmed.append(basket_arrays[basket_num][local_start:local_stop])

        elif start <= entry_stop <= stop:
            local_start = 0
            local_stop = entry_stop - start
            trimmed.append(basket_arrays[basket_num][local_start:local_stop])

        elif entry_start < stop and start <= entry_stop:
            trimmed.append(basket_arrays[basket_num])

        start = stop

    return trimmed


def new_class(name, bases, members):
    """
    Create a new class object with ``type(name, bases, members)`` and put it in
    the ``uproot.dynamic`` library.
    """
    import uproot.dynamic

    out = type(ensure_str(name), bases, members)
    out.__module__ = "uproot.dynamic"
    setattr(uproot.dynamic, out.__name__, out)
    return out


_primitive_awkward_form = {}


def awkward_form(model, file, context):
    """
    Utility function to get an ``ak.forms.Form`` for a :doc:`uproot.model.Model`.
    """

    if isinstance(model, numpy.dtype):
        model = model.newbyteorder("=")

        if model not in _primitive_awkward_form:
            if model == numpy.dtype(numpy.bool_) or model == numpy.dtype(bool):
                _primitive_awkward_form[model] = awkward.forms.from_json('"bool"')
            elif model == numpy.dtype(numpy.int8):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int8"')
            elif model == numpy.dtype(numpy.uint8):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint8"')
            elif model == numpy.dtype(numpy.int16):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int16"')
            elif model == numpy.dtype(numpy.uint16):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint16"')
            elif model == numpy.dtype(numpy.int32):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int32"')
            elif model == numpy.dtype(numpy.uint32):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint32"')
            elif model == numpy.dtype(numpy.int64):
                _primitive_awkward_form[model] = awkward.forms.from_json('"int64"')
            elif model == numpy.dtype(numpy.uint64):
                _primitive_awkward_form[model] = awkward.forms.from_json('"uint64"')
            elif model == numpy.dtype(numpy.float32):
                _primitive_awkward_form[model] = awkward.forms.from_json('"float32"')
            elif model == numpy.dtype(numpy.float64):
                _primitive_awkward_form[model] = awkward.forms.from_json('"float64"')
            elif model.fields is not None:
                fields = []
                contents = []
                for field, (dtype, _) in model.fields.items():
                    fields.append(field)
                    contents.append(awkward_form(dtype, file, context))
                # directly return; don't cache RecordForms in _primitive_awkward_form
                return awkward.forms.RecordForm(contents, fields)
            else:
                raise AssertionError(f"{model!r}: {type(model)}")

        return _primitive_awkward_form[model]

    else:
        return model.awkward_form(file, context)


def recursively_fix_awkward_form_of_iter(awkward, interpretation, form):
    """
    Given an interpretation of a TBranch, fixup any form corresponding to
    a component that cannot be read directly by awkward and needs to go
    through ak.from_iter
    """
    import uproot.interpretation

    if isinstance(interpretation, uproot.interpretation.grouped.AsGrouped):
        fixed = {}
        for key, subinterpretation in interpretation.subbranches.items():
            fixed[key] = recursively_fix_awkward_form_of_iter(
                awkward, subinterpretation, form.content(key)
            )
        return awkward.forms.RecordForm(
            list(fixed.values()),
            list(fixed.keys()),
            parameters=form.parameters,
        )
    elif isinstance(interpretation, uproot.interpretation.objects.AsObjects):
        if uproot.interpretation.objects.awkward_can_optimize(interpretation, form):
            return form
        else:
            return uproot._util.awkward_form_of_iter(awkward, form)
    else:
        return form


def awkward_form_of_iter(awkward, form):
    """
    Fix an ``ak.forms.Form`` object for a given iterable.

    (It might have been read with a different numeric type.)
    """
    if isinstance(form, awkward.forms.BitMaskedForm):
        return awkward.forms.BitMaskedForm(
            form.mask,
            awkward_form_of_iter(awkward, form.content),
            form.valid_when,
            form.lsb_order,
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.ByteMaskedForm):
        return awkward.forms.ByteMaskedForm(
            form.mask,
            awkward_form_of_iter(awkward, form.content),
            form.valid_when,
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.EmptyForm):
        return awkward.forms.EmptyForm(
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.IndexedForm):
        return awkward.forms.IndexedForm(
            form.index,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.IndexedOptionForm):
        return awkward.forms.IndexedOptionForm(
            form.index,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.ListForm):
        return awkward.forms.ListForm(
            form.starts,
            form.stops,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.ListOffsetForm):
        return awkward.forms.ListOffsetForm(
            form.offsets,
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.NumpyForm):
        if form.parameter("__array__") in ("char", "byte"):
            f = form
        else:
            d = awkward.types.numpytype.primitive_to_dtype(form.primitive)
            if issubclass(d.type, numpy.integer):
                d = numpy.dtype(numpy.int64)
            elif issubclass(d.type, numpy.floating):
                d = numpy.dtype(numpy.float64)
            f = awkward.forms.numpyform.from_dtype(d)
        out = awkward.forms.NumpyForm(
            f.primitive,
            form.inner_shape,
            parameters=form.parameters,
        )
        return out
    elif isinstance(form, awkward.forms.RecordForm):
        contents = {
            k: awkward_form_of_iter(awkward, v) for k, v in form.contents.items()
        }
        return awkward.forms.RecordForm(
            list(contents.values()),
            list(contents.keys()),
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.RegularForm):
        return awkward.forms.RegularForm(
            awkward_form_of_iter(awkward, form.content),
            form.size,
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.UnionForm):
        return awkward.forms.UnionForm(
            form.tags,
            form.index,
            [awkward_form_of_iter(awkward, x) for x in form.contents],
            parameters=form.parameters,
        )
    elif isinstance(form, awkward.forms.UnmaskedForm):
        return awkward.forms.UnmaskedForm(
            awkward_form_of_iter(awkward, form.content),
            parameters=form.parameters,
        )
    else:
        raise RuntimeError(f"unrecognized form: {type(form)}")


def ak_add_doc_parameters(obj, ak_add_doc, top_level=False):
    """
    The Awkward parameters that ``ak_add_doc`` gives the form of ``obj``: a TBranch or RField, or
    with ``top_level``, the TTree, RNTuple, TBranch or RField being read as a record.

    ``True`` reads ``title`` into ``__doc__``; a dict reads each ``value`` attribute into its
    ``key``. A top-level record takes only the ``__doc__`` key, and only from an attribute it has.
    Empty values are kept, as a TBranch's empty ``title`` always was.
    """
    if isinstance(ak_add_doc, bool):
        # a branch without a title (#1421) gets an empty __doc__
        return {"__doc__": getattr(obj, "title", "")} if ak_add_doc else {}
    elif not isinstance(ak_add_doc, dict):
        return {}
    elif top_level:
        name = ak_add_doc.get("__doc__")
        return {} if name is None else {"__doc__": getattr(obj, name, "")}
    else:
        return {key: getattr(obj, value) for key, value in ak_add_doc.items()}


def get_ttree_form(
    ttree,
    common_keys,
    ak_add_doc,
):
    contents = []
    for key in common_keys:
        branch = ttree[key]
        content_form = branch.interpretation.awkward_form(ttree.file)
        doc = ak_add_doc_parameters(branch, ak_add_doc)
        if doc:
            content_form = content_form.copy(
                parameters={**content_form.parameters, **doc}
            )
        contents.append(content_form)

    parameters = ak_add_doc_parameters(ttree, ak_add_doc, top_level=True) or None
    return awkward.forms.RecordForm(contents, common_keys, parameters=parameters)


def damerau_levenshtein(a, b):
    """
    Calculates the Damerau-Levenshtein distance of two strings.

    Used by :doc:`uproot.exceptions.KeyInFileError` to return the most likely
    misspellings of a failed attempt to get a key.
    """
    # Modified Damerau-Levenshtein distance. Adds a middling penalty
    # for capitalization.
    # https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    M = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        M[i][0] = i
    for j in range(len(b) + 1):
        M[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:  # Same char
                cost = 0
            elif a[i - 1].lower() == b[j - 1].lower():  # Same if lowered
                cost = 0.5
            else:  # Different char
                cost = 2
            M[i][j] = min(
                M[i - 1][j] + 1,  # Addition
                M[i][j - 1] + 1,  # Removal
                M[i - 1][j - 1] + cost,  # Substitution
            )

            # Transposition
            if (
                i > 1
                and j > 1
                and a[i - 1].lower() == b[j - 2].lower()
                and a[i - 2].lower() == b[j - 1].lower()
            ):
                if a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    # Transpose only
                    M[i][j] = min(M[i][j], M[i - 2][j - 2] + 1)
                else:
                    # Transpose and capitalization
                    M[i][j] = min(M[i][j], M[i - 2][j - 2] + 1.5)

    return M[len(a)][len(b)]


def code_to_datetime(code):
    """
    Converts a ROOT datime code into a Python datetime.
    """
    return datetime.datetime(
        ((code & 0b11111100000000000000000000000000) >> 26) + 1995,
        ((code & 0b00000011110000000000000000000000) >> 22),
        ((code & 0b00000000001111100000000000000000) >> 17),
        ((code & 0b00000000000000011111000000000000) >> 12),
        ((code & 0b00000000000000000000111111000000) >> 6),
        (code & 0b00000000000000000000000000111111),
    )


def datetime_to_code(dt):
    """
    Converts a Python datetime into a ROOT datime code.
    """
    return (
        ((dt.year - 1995) << 26)
        | (dt.month << 22)
        | (dt.day << 17)
        | (dt.hour << 12)
        | (dt.minute << 6)
        | (dt.second)
    )


def objectarray1d(items):
    """
    Converts a sized iterable into a 1D object array

    This avoids ``numpy.array``'s default behavior of turning nested iterables
    into n-d arrays when the shape is rectangular
    """
    out = numpy.empty(len(items), dtype=numpy.dtype(object))
    for i, x in enumerate(items):
        out[i] = x
    return out


_regularize_files_braces = re.compile(r"{([^}]*,)*([^}]*)}")
_regularize_files_isglob = re.compile(r"[\*\?\[\]{}]")


def regularize_steps(steps):
    out = numpy.array(steps)

    if isinstance(steps, dict) or not issubclass(out.dtype.type, numpy.integer):
        raise TypeError(
            "'files' argument's steps must be an iterable of integer offsets or start-stop pairs."
        )

    if len(out.shape) == 1:
        if len(out) == 0 or not numpy.all(out[1:] >= out[:-1]):
            raise ValueError(
                "if 'files' argument's steps are (one-dimensional) offsets, they must be non-empty and monotonically increasing"
            )

    elif len(out.shape) == 2:
        if not (out.shape[1] == 2 and all(out[:, 1] >= out[:, 0])):
            raise ValueError(
                "if 'files' argument's steps are (two-dimensional) start-stop pairs, all stops must be greater than or equal to their corresponding starts"
            )

    else:
        raise TypeError(
            "'files' argument's steps must be an iterable of integer offsets or a list of pairs of integer starts and stops."
        )

    if len(out.shape) == 1:
        out = numpy.stack((out[:-1], out[1:]), axis=1)

    return out.tolist()


def _regularize_files_inner(
    files, parse_colon, counter, HasBranches, steps_allowed, **options
):
    files2 = regularize_path(files)

    maybe_steps = None

    if isinstance(files2, str) and not isinstance(files, str):
        parse_colon = False
        files = files2

    if isinstance(files, str):
        if parse_colon:
            file_path, object_path = file_object_path_split(files)
        else:
            file_path, object_path = files, None

        # This parses the windows drive letter as a scheme!
        if is_remote_url(file_path):
            # user specified a protocol, so we use fsspec to expand the glob and return the full paths
            file_names_full = [
                file.full_name
                for file in fsspec.open_files(
                    file_path,
                    **fsspec_storage_options(options),
                )
            ]
            # https://github.com/fsspec/filesystem_spec/issues/1459
            # Not all protocols return the full_name attribute correctly (if they have url parameters)
            for file_name_full in file_names_full:
                yield file_name_full, object_path, maybe_steps
        else:
            # no protocol, default to local file system
            expanded = os.path.expanduser(file_path)
            if _regularize_files_isglob.search(expanded) is None:
                yield file_path, object_path, maybe_steps

            else:
                matches = list(_regularize_files_braces.finditer(expanded))
                if len(matches) == 0:
                    results = [expanded]
                else:
                    results = []
                    for combination in itertools.product(
                        *[match.group(0)[1:-1].split(",") for match in matches]
                    ):
                        tmp = expanded
                        for c, m in list(zip(combination, matches, strict=True))[::-1]:
                            tmp = tmp[: m.span()[0]] + c + tmp[m.span()[1] :]
                        results.append(tmp)

                seen = set()
                for result in results:
                    for match in glob.glob(result):
                        if match not in seen:
                            yield match, object_path, maybe_steps
                            seen.add(match)

    elif isinstance(files, HasBranches):
        yield files, None, maybe_steps

    elif isinstance(files, dict):
        for key, maybe_object_path in files.items():
            if not isinstance(maybe_object_path, (type(None), str, dict)):
                raise TypeError("object_path may only be a string, dict, or None")
            maybe_steps = None
            if isinstance(maybe_object_path, dict):
                maybe_steps = maybe_object_path.get("steps", None)
                object_path = maybe_object_path.get("object_path", None)
                if maybe_steps is not None:
                    if not steps_allowed:
                        raise TypeError(
                            "unrecognized 'files' pattern for this function ('steps' are only allowed in uproot.dask)"
                        )
                    maybe_steps = regularize_steps(maybe_steps)
            else:
                object_path = maybe_object_path
            for file_path, _, _ in _regularize_files_inner(
                key,
                False,
                counter,
                HasBranches,
                steps_allowed,
                **options,
            ):
                yield file_path, object_path, maybe_steps

    elif isinstance(files, Iterable):
        for file in files:
            counter[0] += 1
            for file_path, object_path, maybe_steps in _regularize_files_inner(
                file, parse_colon, counter, HasBranches, steps_allowed, **options
            ):
                yield file_path, object_path, maybe_steps

    else:
        raise TypeError(
            "'files' must be a file path/URL (string or Path), possibly with "
            "a glob pattern (for local files), a dict of "
            "{path/URL: TTree/TBranch name}, actual TTree/TBranch objects, or "
            f"an iterable of such things, not {files!r}"
        )


def regularize_files(files, steps_allowed, **options):
    """
    Common code for regularizing the possible file inputs accepted by uproot so they can be used by uproot internal functions.
    """
    from uproot.behaviors.TBranch import HasBranches

    out = []
    seen = set()
    counter = [0]
    for file_path, object_path, maybe_steps in _regularize_files_inner(
        files, True, counter, HasBranches, steps_allowed, **options
    ):
        if isinstance(file_path, str):
            key = (counter[0], file_path, object_path)
            if key not in seen:
                out.append((file_path, object_path))
                if maybe_steps is not None:
                    out[-1] = (*out[-1], maybe_steps)

                seen.add(key)
        else:
            out.append((file_path, object_path))
            if maybe_steps is not None:
                out[-1] = (*out[-1], maybe_steps)

    if len(out) == 0:
        raise _file_not_found(files)

    return out


def regularize_object_path(
    file_path, object_path, custom_classes, allow_missing, options
):
    """
    Returns the TTree object from given object and file paths.
    """
    from uproot.behaviors.TBranch import HasBranches, _NoClose
    from uproot.reading import ReadOnlyFile

    if isinstance(file_path, HasBranches):
        return _NoClose(file_path)

    directory = ReadOnlyFile(
        file_path,
        object_cache=None,
        array_cache=None,
        custom_classes=custom_classes,
        **options,
    ).root_directory
    return object_in_directory(directory, object_path, allow_missing)


def object_in_directory(directory, object_path, allow_missing):
    """
    Returns the object at ``object_path`` in an open ``directory`` — the file's single ``TTree``
    when ``object_path`` is None — or None when ``allow_missing`` and it is absent.
    """
    file_path = directory.file.file_path
    if object_path is None:
        trees = directory.keys(filter_classname="TTree", cycle=False)
        if len(trees) == 0:
            if allow_missing:
                return None
            else:
                raise ValueError(f"no TTrees found\nin file {file_path}")
        elif len(trees) == 1:
            return directory[trees[0]]
        else:
            ttree_str = ", ".join(repr(x) for x in trees)
            raise ValueError(
                """TTree object paths must be specified in the 'files' """
                """as {"filenames*.root": "path"} if any files have """
                f"""more than one TTree

    TTrees: {ttree_str}

in file {file_path}"""
            )

    else:
        if allow_missing and object_path not in directory:
            return None
        return directory[object_path]


def _content_cls_from_name(awkward, name):
    if name.endswith(("32", "64")):
        name = name[-2:]
    elif name.endswith("U32"):
        name = name[-3:]
    elif name.endswith(("8_32", "8_64")):
        name = name[-4:]
    elif name.endswith("8_U32"):
        name = name[-5:]
    return getattr(awkward.contents, name)


def pandas_has_attr_is_numeric(pandas):
    try:
        function = pandas.api.types.is_any_real_numeric_dtype
    except AttributeError:

        def function(x):
            return x.is_numeric

    return function


class _Unset:
    def __repr__(self):
        return f"{__name__}.unset"


unset = _Unset()


def get_array_library(arr):
    """
    Determine if an array is a NumPy or CuPy ndarray, without importing CuPy.

    Args:
        arr: The array to check.

    Returns:
        String: 'numpy' if it's a NumPy array, 'cupy' if it's a CuPy array,
        'awkward' if it's an Awkward array, and 'unknown' otherwise.
    """
    module_name = type(arr).__module__

    if module_name.startswith("numpy"):
        return "numpy"
    elif module_name.startswith("cupy"):
        return "cupy"
    elif module_name.startswith("awkward"):
        return "awkward"
    else:
        return "unknown"
