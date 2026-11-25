# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

from __future__ import annotations

import io
import struct
import types

import numpy
import pytest
import skhep_testdata

import uproot

kByteCount = 0x40000000
kClassMask = 0x80000000
kNewClassTag = 0xFFFFFFFF

DeserializationError = uproot.deserialization.DeserializationError


@pytest.fixture(scope="module")
def file():
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        yield f.file


def wrap(file, data):
    return uproot.source.chunk.Chunk.wrap(
        file.source, numpy.frombuffer(data, numpy.uint8)
    )


def tobjstring(text):
    payload = struct.pack(">HII", 1, 0, 0x03000000)
    payload += bytes([len(text)]) + text.encode()
    return struct.pack(">IH", kByteCount | (2 + len(payload)), 1) + payload


def read_any(chunk, cursor, file):
    return uproot.deserialization.read_object_any(
        chunk, cursor, {}, file, file.detached, None
    )


def test_compile_class_syntax_error(file):
    code = "class Broken(:"
    with pytest.raises(SyntaxError, match="class Broken\\(:\n\n") as err:
        uproot.deserialization.compile_class(file, {}, code, "Broken")
    assert isinstance(err.value.__cause__, SyntaxError)


def test_read_object_any_without_byte_counts(file):
    # New class with no byte count, then a reference to that class with no byte count.
    data = (
        struct.pack(">I", kNewClassTag)
        + b"TObjString\x00"
        + tobjstring("one")
        + struct.pack(">I", kClassMask | 1)
        + tobjstring("two")
    )
    chunk = wrap(file, data)
    cursor = uproot.source.cursor.Cursor(0)

    assert read_any(chunk, cursor, file) == "one"
    assert cursor.refs[1] is uproot.classes["TObjString"]
    assert cursor.refs[2] == "one"

    assert read_any(chunk, cursor, file) == "two"
    assert cursor.refs[3] == "two"
    assert cursor.index == len(data)


def test_read_object_any_with_byte_counts(file):
    obj = tobjstring("one")
    tag = struct.pack(">I", kNewClassTag) + b"TObjString\x00"
    data = struct.pack(">I", kByteCount | (len(tag) + len(obj))) + tag + obj
    obj2 = tobjstring("two")
    # the class is registered at the position of its tag, plus kMapOffset
    reference = struct.pack(">I", kClassMask | (4 + uproot.const.kMapOffset))
    data += struct.pack(">I", kByteCount | (len(reference) + len(obj2)))
    data += reference + obj2

    chunk = wrap(file, data)
    cursor = uproot.source.cursor.Cursor(0)
    assert read_any(chunk, cursor, file) == "one"
    assert read_any(chunk, cursor, file) == "two"
    assert cursor.index == len(data)


def test_read_object_any_unresolved_forward_reference(file):
    # A reference to something we don't have: skip past it, using its byte count.
    data = struct.pack(">II", kByteCount | 6, 99) + b"..tail"
    cursor = uproot.source.cursor.Cursor(0)
    assert read_any(wrap(file, data), cursor, file) is None
    assert cursor.index == 10


def test_read_object_any_as_class_with_reference(file):
    # The class tag doesn't matter if we already know the class.
    data = struct.pack(">I", kClassMask | 1) + tobjstring("one")
    cursor = uproot.source.cursor.Cursor(0)
    out = uproot.deserialization.read_object_any(
        wrap(file, data),
        cursor,
        {},
        file,
        file.detached,
        None,
        as_class=uproot.classes["TObjString"],
    )
    assert out == "one"
    assert cursor.refs == {1: "one"}


def test_read_object_any_invalid_class_reference(file):
    data = struct.pack(">I", kClassMask | 9)
    cursor = uproot.source.cursor.Cursor(0)
    cursor.refs[5] = "something"
    with pytest.raises(DeserializationError) as err:
        read_any(wrap(file, data), cursor, file)
    assert err.value.message == (
        "invalid class-tag reference: 9\n\n    Known references: 5"
        f"\n\nin file {file.file_path}"
    )


def test_read_object_any_invalid_class_reference_without_path(file):
    data = struct.pack(">I", kClassMask | 9)
    nameless = types.SimpleNamespace(file_path=None)
    with pytest.raises(DeserializationError) as err:
        uproot.deserialization.read_object_any(
            wrap(file, data), uproot.source.cursor.Cursor(0), {}, nameless, None, None
        )
    assert (
        err.value.message == "invalid class-tag reference: 9\n\n    Known references: "
    )


@pytest.fixture
def real_error():
    filename = skhep_testdata.data_path("uproot-issue-586.root")
    with uproot.open(filename) as f:
        with pytest.raises(DeserializationError) as err:
            f["fail"]
    return err.value


def test_real_error_message(real_error):
    message = str(real_error)
    assert message.startswith(
        "while reading\n\n    ExampleFail version 1 as "
        "uproot.dynamic.Model_ExampleFail_v1 (192 bytes)\n"
    )
    assert "        (base): <TObject 0 50331648 at 0x" in message
    assert "        twovectors: <STLVector [0.0, 1.0, 2.0," in message
    assert "\nBase classes for ExampleFail: (TObject)\n" in message
    assert "\nMembers for ExampleFail: (twovectors)\n" in message
    assert "\n\nexpected 192 bytes but cursor moved by 100 bytes (through" in message
    assert message.endswith("uproot-issue-586.root\nin object /fail;1")


def test_real_error_inspection(real_error):
    assert real_error.partial_object.classname == "ExampleFail"
    assert real_error.partial_object.member("twovectors")[1] == 1.0

    assert real_error.cursor.index == 100
    remainder = real_error.debug_array()
    assert len(remainder) == 192 - 100
    assert remainder.tolist() == real_error.chunk.raw_data[100:].tolist()
    doubles = real_error.debug_array(skip_bytes=-8, dtype=">f8")
    assert doubles[0] == 9.0
    assert len(doubles) == (192 - 100 + 8) // 8

    stream = io.StringIO()
    real_error.debug(skip_bytes=-8, limit_bytes=8, dtype=">f8", stream=stream)
    lines = stream.getvalue().splitlines()
    assert lines[1].split() == ["64", "34", "0", "0", "0", "0", "0", "0"]
    assert lines[3].split() == ["9.0"]


def test_error_in_a_class_with_unexpected_contents(file):
    """TNamed doesn't declare its base or members, so they're marked with "?"."""
    payload = struct.pack(">HII", 1, 0, 0x03000000) + b"\x01a\x01b"
    data = struct.pack(">IH", kByteCount | 100, 1) + payload
    context = {}
    with pytest.raises(DeserializationError) as err:
        uproot.models.TNamed.Model_TNamed.read(
            wrap(file, data),
            uproot.source.cursor.Cursor(0),
            context,
            file,
            file.detached,
            None,
        )

    message = str(err.value)
    assert "Base classes for TNamed: TObject?\n" in message
    assert "Members for TNamed: fName?, fTitle?\n" in message


def error_with(context):
    return DeserializationError("the message", None, None, context, "the-file.root")


@pytest.mark.parametrize(
    "context, expected",
    [
        ({}, "the message\nin file the-file.root"),
        ({"breadcrumbs": ()}, "the message\nin file the-file.root"),
        (
            {"TBranch": types.SimpleNamespace(object_path="/tree/branch")},
            "the message\nin file the-file.root\nin TBranch /tree/branch",
        ),
        (
            {"TKey": types.SimpleNamespace(object_path="/key;1")},
            "the message\nin file the-file.root\nin object /key;1",
        ),
    ],
)
def test_error_without_breadcrumbs(context, expected):
    assert str(error_with(context)) == expected


def test_error_partial_object_without_breadcrumbs():
    assert error_with({}).partial_object is None


def test_error_with_empty_model():
    """No bases and no members to report."""
    obj = uproot.models.TTree.Model_ROOT_3a3a_TIOFeatures.empty()
    assert str(error_with({"breadcrumbs": (obj,)})) == (
        "while reading\n\n"
        "    ROOT::TIOFeatures version None as "
        "uproot.models.TTree.Model_ROOT_3a3a_TIOFeatures (? bytes)\n\n"
        "the message\nin file the-file.root"
    )


def test_error_with_a_base_that_is_not_a_model():
    obj = uproot.models.TNamed.Model_TNamed.empty()
    obj._bases.append("not a model")
    message = str(error_with({"breadcrumbs": (obj,)}))
    assert "(base): 'not a model'" in message
    assert "Base classes for" not in message


def test_error_with_something_that_is_not_a_model():
    """Only the parts that it has are reported."""
    obj = types.SimpleNamespace(classname="Thing", instance_version=3, num_bytes=10)
    assert str(error_with({"breadcrumbs": (obj,)})) == (
        "while reading\n\n"
        "    Thing version 3 as types.SimpleNamespace (10 bytes)\n\n"
        "the message\nin file the-file.root"
    )
