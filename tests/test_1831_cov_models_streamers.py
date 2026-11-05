# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
TStreamerInfo deserialization and class generation, for the kinds of streamers that
the sample files don't have: hand-made TStreamerElements, deserialized by uproot and
compiled into classes that read hand-made data.
"""

from __future__ import annotations

import struct
import threading

import numpy
import pytest
import skhep_testdata

import uproot
from uproot.interpretation.objects import (
    CannotBeAwkward,
    CannotBeForth,
    CannotBeStrided,
)

kByteCount = 0x40000000


@pytest.fixture
def file(reset_classes):
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        yield f.file


def string(text):
    return bytes([len(text)]) + text.encode()


def header(version, payload):
    return struct.pack(">IH", kByteCount | (2 + len(payload)), version) + payload


def wrap(file, data):
    return uproot.source.chunk.Chunk.wrap(
        file.source, numpy.frombuffer(data, numpy.uint8)
    )


def read(cls, data, file, context=None):
    cursor = uproot.source.cursor.Cursor(0)
    obj = cls.read(
        wrap(file, data),
        cursor,
        {} if context is None else context,
        file,
        file.detached,
        None,
    )
    assert cursor.index == len(data)
    return obj


def element_data(
    name, ftype, typename, size=0, array_length=0, version=4, title="", max_index=None
):
    tobject = struct.pack(">HII", 1, 0, 0x03000000)
    payload = header(1, tobject + string(name) + string(title))
    payload += struct.pack(">iiii", ftype, size, array_length, int(array_length != 0))
    if version == 1:
        max_index = [] if max_index is None else max_index
        payload += struct.pack(">i", len(max_index))
        payload += struct.pack(f">{len(max_index)}i", *max_index)
    else:
        payload += struct.pack(">5i", array_length, 0, 0, 0, 0)
    payload += string(typename)
    return header(version, payload)


def count_data(count_name):
    return struct.pack(">i", 1) + string(count_name) + string("TheClass")


def basic(file, name, ftype, typename, **kwargs):
    data = header(2, element_data(name, ftype, typename, **kwargs))
    return read(uproot.streamers.Model_TStreamerBasicType, data, file)


def pointer(file, name, ftype, typename, count_name):
    data = element_data(name, ftype, typename) + count_data(count_name)
    return read(uproot.streamers.Model_TStreamerBasicPointer, header(2, data), file)


def loop(file, name, typename, count_name):
    data = element_data(name, uproot.const.kStreamLoop, typename)
    return read(
        uproot.streamers.Model_TStreamerLoop,
        header(2, data + count_data(count_name)),
        file,
    )


def stl(file, name, typename, stl_type=1, ctype=3):
    data = element_data(name, 500, typename) + struct.pack(">ii", stl_type, ctype)
    return read(uproot.streamers.Model_TStreamerSTL, header(3, data), file)


def object_pointer(file, name, ftype, typename):
    data = header(2, element_data(name, ftype, typename))
    return read(uproot.streamers.Model_TStreamerObjectPointer, data, file)


def streamer_info(name, version, elements):
    named = uproot.models.TNamed.Model_TNamed.empty()
    named._members.update({"fName": name, "fTitle": ""})
    out = uproot.streamers.Model_TStreamerInfo.empty()
    out._bases.append(named)
    out._members.update(
        {"fCheckSum": 0, "fClassVersion": version, "fElements": elements}
    )
    return out


def form_context(**kwargs):
    return {
        "index_format": "i64",
        "header": False,
        "tobject_header": True,
        "breadcrumbs": (),
        **kwargs,
    }


def forth_context():
    forth = threading.local()
    forth.gen = uproot._awkwardforth.ForthGenerator(None)
    return {"forth": forth, "path": ("branch",)}


def test_info_properties():
    with uproot.open(skhep_testdata.data_path("uproot-small-flat-tree.root")) as f:
        info = f.file.streamer_named("TNamed", 1)
        assert repr(info).startswith("<TStreamerInfo for TNamed version 1 at 0x")
        assert info.typename == "TNamed"

        element = info.elements[-1]
        assert element.name == "fTitle"
        assert element.file_uuid == f.file.uuid

        out = []
        info._serialize(out, False, None, numpy.uint32(0))
        assert b"".join(out) == info.serialize()[6:]

        out = []
        element._serialize(out, False, None, numpy.uint32(0))
        assert b"".join(out) == element.serialize()[6:]


def test_element_version_1(file):
    data = element_data("x", 3, "int", size=4, version=1, max_index=[7, 8])
    obj = read(uproot.streamers.Model_TStreamerElement, data, file)
    assert obj.member("fMaxIndex").tolist() == [7, 8]
    assert obj.typename == "int"
    assert obj.fType == 3


@pytest.mark.parametrize("version", [2, 3])
def test_element_old_versions(file, version):
    data = element_data("x", 3, "int", size=4, version=version)
    obj = read(uproot.streamers.Model_TStreamerElement, data, file)
    assert obj.member("fMaxIndex").tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "written, typename, expected",
    [
        (11, "Bool_t", uproot.const.kBool),
        (11, "bool", uproot.const.kBool),
        (11, "UChar_t", uproot.const.kUChar),
    ],
)
def test_element_bool_written_as_uchar(file, written, typename, expected):
    obj = read(
        uproot.streamers.Model_TStreamerElement,
        element_data("x", written, typename),
        file,
    )
    assert obj.fType == expected


def test_artificial_element(file):
    data = element_data("art", 65, "TArtificial")
    artificial = read(uproot.streamers.Model_TStreamerArtificial, header(1, data), file)
    assert artificial._serialization == data
    assert artificial.typename == "TArtificial"

    info = streamer_info("Cov1831Artificial", 1, [artificial])
    cls = info.new_class(file)
    message = (
        "not implemented: class members defined by Model_TStreamerArtificial of type "
        "TArtificial in member art of class Cov1831Artificial"
    )

    with pytest.raises(uproot.deserialization.DeserializationError, match=message):
        read(cls, header(1, b""), file)
    with pytest.raises(uproot.deserialization.DeserializationError, match=message):
        cls.empty().read_member_n(wrap(file, b"\x00"), None, {}, file, 0)
    with pytest.raises(CannotBeStrided, match=message):
        cls.strided_interpretation(file)
    with pytest.raises(CannotBeAwkward, match=message):
        cls.awkward_form(file, form_context())


def test_base_element_version_1(file):
    data = element_data("TObject", 66, "BASE")
    obj = read(uproot.streamers.Model_TStreamerBase, header(1, data), file)
    assert not obj.has_member("fBaseVersion", all=False)
    assert obj._serialization == data


def test_base_dependencies_without_versions(file):
    data = element_data("Cov1831Base", 0, "BASE") + struct.pack(">i", -1)
    base = read(uproot.streamers.Model_TStreamerBase, header(3, data), file)
    assert base.base_version == "max"

    out = []
    base._dependencies({"Cov1831Base": {}}, out)
    assert out == []
    base._dependencies({}, out)
    assert out == []


def test_float_types_in_a_generated_class(file):
    a = basic(file, "a", uproot.const.kDouble32, "Double32_t")
    b = basic(file, "b", uproot.const.kFloat16, "Float16_t")
    c = basic(file, "c", uproot.const.kInt, "int")
    cls = streamer_info("Cov1831Floats", 1, [a, b, c]).new_class(file)

    data = (
        struct.pack(">f", 2.5) + struct.pack(">BH", 127, 0x800) + struct.pack(">i", 7)
    )

    obj = read(cls, header(1, data), file)
    assert obj.member("a") == 2.5
    assert obj.member("b") == 1.5
    assert obj.member("c") == 7
    assert isinstance(obj.member("b"), float)

    obj = cls.empty()
    chunk = wrap(file, data)
    cursor = uproot.source.cursor.Cursor(0)
    for i in range(3):
        obj.read_member_n(chunk, cursor, {}, file, i)
    assert obj.members == {"a": 2.5, "b": 1.5, "c": 7}

    with pytest.raises(CannotBeStrided, match="Double32_t in member a of class"):
        cls.strided_interpretation(file)

    form = cls.awkward_form(file, form_context())
    assert form.fields == ["a", "b", "c"]
    assert [x.primitive for x in form.contents] == ["float64", "float32", "int32"]

    with pytest.raises(CannotBeForth):
        read(cls, header(1, data), file, forth_context())


def test_float16_alone_cannot_be_forth(file):
    b = basic(file, "b", uproot.const.kFloat16, "Float16_t")
    cls = streamer_info("Cov1831Float16", 1, [b]).new_class(file)
    data = struct.pack(">BH", 127, 0x800)
    with pytest.raises(CannotBeForth):
        read(cls, header(1, data), file, forth_context())


def test_unknown_basic_type(file):
    element = basic(file, "x", 123, "NotAType")
    with pytest.raises(NotImplementedError, match="123"):
        streamer_info("Cov1831Unknown", 1, [element]).class_code()


def test_basic_type_sizes(file):
    # sizes of types that aren't basic are left as they are
    other = basic(file, "x", 123, "NotAType", size=4, array_length=3)
    assert other.member("fSize") == 4

    doubles = basic(file, "x", uproot.const.kDouble, "double", array_length=3)
    assert doubles.member("fSize") == 24


@pytest.mark.parametrize("flag, dtype", [(2, ">i8"), (1, ">i4")])
@pytest.mark.parametrize("speedbump", [True, False])
def test_generated_tbranch_fbasketseek(file, flag, dtype, speedbump):
    # TBranch's fBasketSeek says how wide the seek positions are in the speedbump byte.
    # (This class is called TBranch, and the name is what gives it this feature.)
    element = pointer(
        file,
        "fBasketSeek",
        uproot.const.kOffsetP + uproot.const.kInt,
        "int*",
        "fMaxBaskets",
    )
    cls = streamer_info("TBranch", 99, [element]).new_class(file)

    seeks = numpy.array([1, 2], dtype)
    data = bytes([flag]) + seeks.tobytes()
    if not speedbump:
        # without a speedbump, the array starts right here
        data = numpy.array([1, 2], ">i4").tobytes()
        dtype = ">i4"

    context = {"speedbump": speedbump}
    for method, args in [("read_members", ()), ("read_member_n", (0,))]:
        obj = cls.empty()
        obj._members["fMaxBaskets"] = 2
        getattr(obj, method)(
            wrap(file, data), uproot.source.cursor.Cursor(0), context, file, *args
        )
        assert obj.member("fBasketSeek").tolist() == [1, 2]
        assert obj.member("fBasketSeek").dtype == numpy.dtype(dtype)


def test_generated_tbranch_fbasketseek_cannot_be_forth(file):
    element = pointer(
        file,
        "fBasketSeek",
        uproot.const.kOffsetP + uproot.const.kLong64,
        "Long64_t*",
        "fMaxBaskets",
    )
    cls = streamer_info("TBranch", 99, [element]).new_class(file)

    obj = cls.empty()
    obj._members["fMaxBaskets"] = 0
    with pytest.raises(CannotBeForth):
        obj.read_members(
            wrap(file, b"\x01"), uproot.source.cursor.Cursor(0), forth_context(), file
        )


def test_loop_dependencies(file):
    element = loop(file, "items", "Cov1831Item*", "n")
    assert element.typename == "Cov1831Item*"

    class Streamer:
        name = "Cov1831Item"

        def __init__(self, class_version):
            self.class_version = class_version

        def _dependencies(self, streamers, out):
            out.append((self.name, self.class_version))

    streamers = {"Cov1831Item": {1: Streamer(1), 2: Streamer(2)}}
    out = [("Cov1831Item", 1)]
    element._dependencies(streamers, out)
    assert out == [("Cov1831Item", 1), ("Cov1831Item", 2)]

    out = []
    element._dependencies({}, out)
    assert out == []


@pytest.mark.parametrize(
    "typename, written, expected",
    [
        ("set<int>", uproot.const.kSTLmultimap, uproot.const.kSTLset),
        ("std::set<int>", uproot.const.kSTLmultimap, uproot.const.kSTLset),
        ("multimap<int,int>", uproot.const.kSTLset, uproot.const.kSTLmultimap),
        ("std::multimap<int,int>", uproot.const.kSTLset, uproot.const.kSTLmultimap),
        ("vector<int>", uproot.const.kSTLset, uproot.const.kSTLset),
    ],
)
def test_stl_type_from_typename(file, typename, written, expected):
    element = stl(file, "x", typename, stl_type=written, ctype=3)
    assert element.stl_type == expected
    assert element.fCtype == 3


def test_pointer_of_unexpected_type(file):
    element = object_pointer(file, "x", 123, "TObject*")
    cls = streamer_info("Cov1831Pointer", 1, [element]).new_class(file)
    message = (
        "not implemented: class members defined by Model_TStreamerObjectPointer "
        "with fType 123"
    )
    with pytest.raises(uproot.deserialization.DeserializationError, match=message):
        read(cls, header(1, b""), file)
    with pytest.raises(uproot.deserialization.DeserializationError, match=message):
        cls.empty().read_member_n(wrap(file, b"\x00"), None, {}, file, 0)


def test_pointer_dependencies_without_streamers(file):
    element = object_pointer(file, "x", uproot.const.kObjectP, "TObject*")
    out = []
    element._dependencies({}, out)
    assert out == []
