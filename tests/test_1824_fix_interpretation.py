# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pickle
import struct
import types

import awkward
import numpy
import pytest
import skhep_testdata

import uproot
import uproot._util
import uproot.interpretation.custom
import uproot.interpretation.grouped
from uproot.containers import (
    AsArray,
    AsBitSet,
    AsDynamic,
    AsFIXME,
    AsList,
    AsMap,
    AsPointer,
    AsRVec,
    AsSet,
    AsString,
    AsVector,
    ROOTRVec,
    STLList,
    STLMap,
    STLSet,
    STLVector,
)
from uproot.interpretation import library
from uproot.interpretation.custom import AsBinary, CustomInterpretation
from uproot.interpretation.identify import (
    _float16_or_double32,
    interpretation_of,
    parse_typename,
    UnknownInterpretation,
)
from uproot.interpretation.jagged import AsJagged, JaggedArray
from uproot.interpretation.numerical import AsDtype, AsFloat16
from uproot.interpretation.objects import (
    AsObjects,
    AsStridedObjects,
    ObjectArray,
    StridedObjectArray,
)
from uproot.interpretation.strings import AsStrings, StringArray
from uproot.models.TObject import Model_TObject

np = library._libraries["np"]


ak = library._libraries["ak"]


@pytest.fixture
def raw_branch(tmp_path):
    """
    Makes a real TBranch without entry offsets, for which all entries have the
    same size: every argument is a list of entries for a separate TBasket.
    """
    opened = []

    def make(*baskets):
        path = tmp_path / f"raw{len(opened)}.root"
        size = len(baskets[0][0])
        with uproot.recreate(path) as file:
            tree = file.mktree("tree", {"x": numpy.dtype(("u1", (size,)))})
            for entries in baskets:
                data = numpy.frombuffer(b"".join(entries), dtype=numpy.uint8)
                tree.extend({"x": data.reshape(-1, size)})
        opened.append(uproot.open(path))
        return opened[-1]["tree"]["x"]

    yield make

    for file in opened:
        file.close()


@pytest.fixture
def tree(tmp_path):
    path = tmp_path / "written.root"
    with uproot.recreate(path) as file:
        file.mktree("tree", {"x": "int32", "y": "var * uint8"})
        file["tree"].extend({"x": [1, 2], "y": awkward.Array([[1, 2], [3, 4]])})
        file["tree"].extend({"x": [3], "y": awkward.Array([[5, 6]])})

    with uproot.open(path) as file:
        yield file["tree"]


def test_truncated_read_wrong_number_of_bytes(raw_branch):
    branch = raw_branch([b"\x00" * 4])

    with pytest.raises(ValueError) as excinfo:
        branch.array(interpretation=AsFloat16(0, 0, 12), library="np")

    assert str(excinfo.value) == (
        f"basket 0 in tree/branch {branch.object_path} has the wrong number of bytes "
        "(4) for interpretation AsFloat16(0, 0, 12) (expecting raw array of "
        f"{AsFloat16(0, 0, 12).from_dtype!r})\nin file {branch.file.file_path}"
    )


def test_jagged_array_parents_localindex():
    array = JaggedArray(numpy.array([0, 2, 2, 5]), numpy.array([1, 2, 3, 4, 5]))

    parents, localindex = array.parents_localindex(10, 13)

    assert parents.tolist() == [10, 10, 12, 12, 12]
    assert localindex.tolist() == [0, 1, 0, 1, 2]


def string1to5(text):
    if len(text) < 255:
        return bytes([len(text)]) + text
    else:
        return b"\xff" + struct.pack(">I", len(text)) + text


def string4(text):
    return struct.pack(">I", len(text)) + text


@pytest.mark.parametrize("name", ["np", "ak"])
@pytest.mark.parametrize(
    "length_bytes, serialize, first, second, third",
    [
        ("1-5", string1to5, b"abc", b"def", b"ghi"),
        ("1-5", string1to5, b"a" * 300, b"b" * 300, b"c" * 300),
        ("4", string4, b"abc", b"def", b"ghi"),
    ],
    ids=["1-5", "1-5-long", "4"],
)
def test_strings_without_offsets(
    raw_branch, name, length_bytes, serialize, first, second, third
):
    # the second TBasket uses the same AwkwardForth machine as the first
    branch = raw_branch([serialize(first), serialize(second)], [serialize(third)])

    result = branch.array(
        interpretation=AsStrings(length_bytes=length_bytes), library=name
    )

    assert result.tolist() == [first.decode(), second.decode(), third.decode()]


def test_final_array_of_mixed_string_baskets():
    # Whether basket_array returns an awkward array or a StringArray depends
    # on whether each TBasket has entry offsets; build one of each from the
    # same real TBasket, which has none.
    with uproot.open(skhep_testdata.data_path("uproot-issue413.root")) as file:
        branch = file["mytree"]["Str"]
        basket = branch.basket(0)
        interpretation = branch.interpretation
        assert basket.byte_offsets is None

        def basket_array(library):
            return interpretation.basket_array(
                basket.data,
                basket.byte_offsets,
                basket,
                branch,
                branch.context,
                basket.member("fKeylen"),
                library,
                {},
            )

        as_awkward = basket_array(ak)
        as_string_array = basket_array(np)
        assert isinstance(as_awkward, awkward.Array)
        assert isinstance(as_string_array, StringArray)

        strings = [f"evt-{i}" for i in range(5)]
        for baskets in [
            {0: as_awkward, 1: as_string_array},
            {0: as_string_array, 1: as_awkward},
        ]:
            result = interpretation.final_array(
                baskets, 3, 7, [0, 5, 10], ak, branch, {}
            )
            assert isinstance(result, awkward.Array)
            assert str(result.type) == "4 * string"
            assert result.tolist() == strings[3:] + strings[:2]


class Custom(CustomInterpretation):
    pass


def test_custom_interpretation_properties():
    with uproot.open(skhep_testdata.data_path("uproot-Event.root")) as file:
        interpretation = Custom(file["T/event/fTemperature"], {}, True)

        assert repr(interpretation) == "Custom"
        assert interpretation.typename == "Double32_t"
        assert interpretation.cache_key == id(interpretation)
        assert interpretation == interpretation
        assert interpretation != Custom(file["T/event/fTemperature"], {}, True)


def test_as_binary_eq():
    interpretation = AsBinary()
    assert interpretation == interpretation
    assert interpretation != AsBinary()


i4 = numpy.dtype(">i4")


f8 = numpy.dtype(">f8")


@pytest.mark.parametrize(
    "container, typename, cache_key, text",
    [
        (AsDynamic(), "void*", "AsDynamic(None)", "AsDynamic()"),
        (
            AsDynamic(Model_TObject),
            "TObject",
            "AsDynamic(Model_TObject)",
            "AsDynamic(model=Model_TObject)",
        ),
        (
            AsDynamic(AsVector(False, i4)),
            "std::vector<int32_t>",
            "AsDynamic(AsVector(False,Bi4))",
            "AsDynamic(model=AsVector(False, dtype('>i4')))",
        ),
        (
            AsFIXME("not done"),
            "unknown",
            "AsFIXME('not done')",
            "AsFIXME('not done')",
        ),
        (AsString(False), "std::string", "AsString(False,'1-5')", "AsString(False)"),
        (
            AsString(True, length_bytes="4", typename="char*"),
            "char*",
            "AsString(True,'4')",
            "AsString(True, length_bytes='4')",
        ),
        (AsPointer(), "void*", "AsPointer(None)", "AsPointer()"),
        (
            AsPointer(Model_TObject),
            "TObject*",
            "AsPointer(Model_TObject)",
            "AsPointer(Model_TObject)",
        ),
        (
            AsPointer(AsVector(True, f8)),
            "std::vector<double>*",
            "AsPointer(AsVector(True,Bf8))",
            "AsPointer(AsVector(True, dtype('>f8')))",
        ),
        (
            AsArray(True, False, Model_TObject),
            "TObject[]",
            "AsArray(True,False,Model_TObject,())",
            "AsArray(True, False, Model_TObject, ())",
        ),
        (
            AsArray(False, True, i4),
            "int32_t[]",
            "AsArray(False,True,Bi4,())",
            "AsArray(False, True, dtype('>i4'), ())",
        ),
        (
            AsArray(False, False, AsString(False), (2, 3)),
            "std::string[][2][3]",
            "AsArray(False,False,AsString(False,'1-5'),(2, 3))",
            "AsArray(False, False, AsString(False), (2, 3))",
        ),
        (
            AsRVec(True, i4),
            "RVec<int32_t>",
            "AsRVec(True,Bi4)",
            "AsRVec(True, dtype('>i4'))",
        ),
        (
            AsVector(False, Model_TObject),
            "std::vector<TObject>",
            "AsVector(False,Model_TObject)",
            "AsVector(False, Model_TObject)",
        ),
        (
            AsList(True, AsString(False)),
            "std::list<std::string>",
            "AsList(True,AsString(False,'1-5'))",
            "AsList(True, AsString(False))",
        ),
        (
            AsSet(False, f8),
            "std::set<double>",
            "AsSet(False,Bf8)",
            "AsSet(False, dtype('>f8'))",
        ),
        (
            AsBitSet(True, 12),
            "std::bitset<12>",
            "AsBitSet(True,Lb1)",
            "AsBitSet(True, dtype('bool'))",
        ),
        (
            AsBitSet(False, numpy.dtype("?")),
            "std::bitset<bool>",
            "AsBitSet(False,Lb1)",
            "AsBitSet(False, dtype('bool'))",
        ),
        (
            AsMap(True, i4, AsString(False)),
            "std::map<int32_t, std::string>",
            "AsMap(True,Bi4,AsString(False,'1-5'))",
            "AsMap(True, dtype('>i4'), AsString(False))",
        ),
        (
            AsMap(False, AsString(False), Model_TObject),
            "std::map<std::string, TObject>",
            "AsMap(False,AsString(False,'1-5'),Model_TObject)",
            "AsMap(False, AsString(False), Model_TObject)",
        ),
        (
            AsMap(False, Model_TObject, f8),
            "std::map<TObject, double>",
            "AsMap(False,Model_TObject,Bf8)",
            "AsMap(False, Model_TObject, dtype('>f8'))",
        ),
    ],
)
def test_typename_cache_key_repr(container, typename, cache_key, text):
    assert container.typename == typename
    assert container.cache_key == cache_key
    assert repr(container) == text


@pytest.mark.parametrize(
    "left, right, equal",
    [
        (AsFIXME("a"), AsFIXME("a"), True),
        (AsFIXME("a"), AsFIXME("b"), False),
        (AsFIXME("a"), AsString(False), False),
        (AsDynamic(), AsDynamic(), True),
        (AsDynamic(Model_TObject), AsDynamic(Model_TObject), True),
        (AsDynamic(Model_TObject), AsDynamic(), False),
        (AsDynamic(), AsPointer(), False),
        (AsString(False), AsString(False), True),
        (AsString(False), AsString(True), False),
        (AsString(False), AsString(False, length_bytes="4"), False),
        (AsPointer(Model_TObject), AsPointer(Model_TObject), True),
        (AsPointer(Model_TObject), AsPointer(), False),
        (AsPointer(), AsString(False), False),
        (AsArray(True, False, i4), AsArray(True, False, i4), True),
        (AsArray(True, False, i4), AsArray(False, False, i4), False),
        (AsArray(True, False, i4), AsArray(True, True, i4), False),
        (AsArray(True, False, i4), AsArray(True, False, f8), False),
        (
            AsArray(True, False, AsString(False), (2,)),
            AsArray(True, False, AsString(False), (3,)),
            False,
        ),
        (AsArray(True, False, i4), AsVector(True, i4), False),
        (
            AsArray(True, False, numpy.dtype("O")),
            AsArray(True, False, Model_TObject),
            False,
        ),
        (
            AsArray(True, False, Model_TObject),
            AsArray(True, False, numpy.dtype("O")),
            False,
        ),
        (
            AsArray(True, False, Model_TObject),
            AsArray(True, False, Model_TObject),
            True,
        ),
        (
            AsArray(True, False, AsString(False)),
            AsArray(True, False, AsString(True)),
            False,
        ),
        (AsVector(True, i4), AsVector(True, i4), True),
        (AsVector(True, i4), AsVector(False, i4), False),
        (AsVector(True, i4), AsVector(True, f8), False),
        (AsVector(True, i4), AsRVec(True, i4), False),
        (AsVector(True, i4), AsVector(True, AsString(False)), False),
        (AsVector(True, AsString(False)), AsVector(True, i4), False),
        (AsVector(True, AsString(False)), AsVector(True, AsString(False)), True),
        (AsVector(True, AsString(False)), AsVector(True, AsString(True)), False),
        (AsVector(True, Model_TObject), AsVector(True, Model_TObject), True),
        (AsList(False, f8), AsList(False, f8), True),
        (AsSet(False, f8), AsSet(False, f8), True),
        (AsBitSet(False, 5), AsBitSet(False, 5), True),
        (AsMap(True, i4, f8), AsMap(True, i4, f8), True),
        (AsMap(True, i4, f8), AsVector(True, i4), False),
        (AsMap(True, i4, f8), AsMap(False, i4, f8), False),
        (AsMap(True, i4, f8), AsMap(True, f8, f8), False),
        (AsMap(True, i4, f8), AsMap(True, i4, i4), False),
        (AsMap(True, i4, f8), AsMap(True, AsString(False), f8), False),
        (AsMap(True, i4, f8), AsMap(True, i4, AsString(False)), False),
        (
            AsMap(True, AsString(False), AsString(True)),
            AsMap(True, AsString(False), AsString(True)),
            True,
        ),
        (
            AsMap(True, AsString(False), AsString(True)),
            AsMap(True, AsString(True), AsString(True)),
            False,
        ),
    ],
)
def test_eq_ne(left, right, equal):
    assert (left == right) is equal
    assert (left != right) is (not equal)


@pytest.mark.parametrize(
    "left, right",
    [
        (AsFIXME("a"), AsFIXME("a")),
        (AsString(False), AsString(False)),
        (AsPointer(Model_TObject), AsPointer(Model_TObject)),
        (AsDynamic(Model_TObject), AsDynamic(Model_TObject)),
        (AsArray(False, True, i4, ()), AsArray(False, True, i4, ())),
        (AsVector(False, i4), AsVector(False, i4)),
        (
            AsVector(False, AsArray(False, True, i4)),
            AsVector(False, AsArray(False, True, i4)),
        ),
        (AsMap(True, i4, f8), AsMap(True, i4, f8)),
    ],
)
def test_equal_containers_hash_equal(left, right):
    assert left == right
    assert hash(left) == hash(right)
    assert len({left, right}) == 1


def test_fixme_read():
    file = types.SimpleNamespace(file_path="some/file.root")
    with pytest.raises(uproot.deserialization.DeserializationError) as excinfo:
        AsFIXME("this type").read(None, None, {}, file, None, None)
    assert excinfo.value.message == "this type; please file a bug report!"
    assert str(excinfo.value) == (
        "this type; please file a bug report!\nin file some/file.root"
    )


def test_sequences_of_strings_quote_their_items():
    assert str(STLVector(["a", "b"])) == "['a', 'b']"
    assert str(STLList(["a", "b"])) == "['a', 'b']"
    assert str(ROOTRVec(["a", "b"])) == "['a', 'b']"
    assert str(STLVector(numpy.array(["a"], dtype=object))) == "['a']"


def test_set_of_strings():
    container = STLSet(["b", "a"])

    assert str(container) == "{'a', 'b'}"
    assert "a" in container
    assert "c" not in container


def test_map_of_strings():
    container = STLMap(["b", "a"], numpy.array(["x", "y"], dtype=object))

    assert str(container) == "{'a': 'y', 'b': 'x'}"
    assert container["a"] == "y"
    assert "b" in container


scalars = {
    "Bool_t": "?",
    "bool": "?",
    "Char_t": "i1",
    "char": "i1",
    "UChar_t": "u1",
    "unsigned char": "u1",
    "signed char": "i1",
    "Short_t": ">i2",
    "short": ">i2",
    "signed short": ">i2",
    "UShort_t": ">u2",
    "unsigned short": ">u2",
    "Int_t": ">i4",
    "int": ">i4",
    "signed int": ">i4",
    "UInt_t": ">u4",
    "unsigned int": ">u4",
    "long long": ">i8",
    "unsigned long long": ">u8",
    "Long_t": ">i8",
    "Long64_t": ">i8",
    "long": ">i8",
    "signed long": ">i8",
    "ULong_t": ">u8",
    "ULong64_t": ">u8",
    "unsigned long": ">u8",
    "Float_t": ">f4",
    "float": ">f4",
    "Double_t": ">f8",
    "double": ">f8",
    "const int": ">i4",
    "const Double_t": ">f8",
}


pointers = {
    "Bool_t*": "?",
    "bool*": "?",
    "UChar_t*": "u1",
    "unsigned char*": "u1",
    "Short_t*": ">i2",
    "short*": ">i2",
    "UShort_t*": ">u2",
    "unsigned short*": ">u2",
    "Int_t*": ">i4",
    "int *": ">i4",
    "UInt_t*": ">u4",
    "unsigned int*": ">u4",
    "long long*": ">i8",
    "unsigned long long*": ">u8",
    "Long_t*": ">i8",
    "Long64_t*": ">i8",
    "long*": ">i8",
    "ULong_t*": ">u8",
    "ULong64_t*": ">u8",
    "unsigned long*": ">u8",
    "Float_t*": ">f4",
    "float  *": ">f4",
    "Double_t*": ">f8",
    "double*": ">f8",
}


f4 = numpy.dtype(">f4")


# what comes out with the default outer_header=True, inner_header=False,
# string_header=False
containers = {
    "string": AsString(False),
    "std::string": AsString(False),
    "std :: string": AsString(False),
    "TString": AsString(False),
    "char*": AsString(False, length_bytes="4"),
    "Float16_t": AsFIXME("Float16_t in another context"),
    "Double32_t": AsFIXME("Double32_t in another context"),
    "Float16_t*": AsArray(False, True, AsFIXME("Float16_t in array")),
    "Double32_t*": AsArray(
        False,
        True,
        AsFIXME(
            "Double32_t in array (note: Event.root fClosestDistance has an example)"
        ),
    ),
    "bitset<8>": AsBitSet(True, 8),
    "std::bitset<8>": AsBitSet(True, 8),
    "vector<int>": AsVector(True, i4),
    "std::vector<int>": AsVector(True, i4),
    "std::vector<int, std::allocator<int> >": AsVector(True, i4),
    "vector<vector<float> >": AsVector(True, AsVector(False, f4)),
    "vector<string>": AsVector(True, AsString(False)),
    "vector<TObject>": AsVector(True, Model_TObject),
    "vector<TObject*>": AsVector(True, AsPointer(Model_TObject)),
    "const vector<int>": AsVector(True, i4),
    "list<float>": AsList(True, f4),
    "std::list<float>": AsList(True, f4),
    "std::list<float, std::allocator<float> >": AsList(True, f4),
    "RVec<int>": AsRVec(True, i4),
    "VecOps::RVec<int>": AsRVec(True, i4),
    "ROOT::VecOps::RVec<int>": AsRVec(True, i4),
    "ROOT::VecOps::RVec<float, std::allocator<float> >": AsRVec(True, f4),
    "set<int>": AsSet(True, i4),
    "std::set<int>": AsSet(True, i4),
    "std::set<int, std::less<int>, std::allocator<int> >": AsSet(True, i4),
    "map<int,double>": AsMap(True, i4, f8),
    "std::map<int, double>": AsMap(True, i4, f8),
    "std::map<int, double, std::less<int>, std::allocator<int> >": AsMap(True, i4, f8),
    "map<string,vector<int> >": AsMap(True, AsString(True), AsVector(True, i4)),
    "TObject": Model_TObject,
    "TObject*": AsPointer(Model_TObject),
    "TObject * *": AsPointer(AsPointer(Model_TObject)),
}


def quoted_namespace():
    return {"numpy": numpy, "uproot": uproot, "c": uproot.classes.__getitem__}


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("typename, expected", pointers.items())
def test_pointers(typename, expected, header):
    # the second AsArray argument is the speedbump, which depends on the header
    out = parse_typename(typename, outer_header=header)
    assert out == AsArray(False, header, numpy.dtype(expected))


@pytest.mark.parametrize("typename, expected", containers.items())
def test_containers(typename, expected):
    assert parse_typename(typename) == expected


@pytest.mark.parametrize(
    "typename", [*scalars, *pointers, *containers], ids=lambda x: x.replace(" ", "_")
)
def test_quoted_evaluates_to_the_same(typename):
    code = parse_typename(typename, quote=True)
    assert isinstance(code, str)
    assert eval(code, quoted_namespace()) == parse_typename(typename)


@pytest.mark.parametrize(
    "typename, position",
    [
        ("", 0),
        (",", 1),
        ("int,", 3),
        ("vector", 6),
        ("vector int", 8),
        ("vector<int", 10),
        ("vector<int,int", 14),
        ("vector<,int>", 8),
        ("vector<int,int,int>", 15),
        ("vector<int>>", 11),
        ("list", 4),
        ("list<int", 8),
        ("RVec<", 5),
        ("set<int", 7),
        ("set<int,int,int,int>", 16),
        ("map<int", 7),
        ("map<int>", 8),
        ("map<int,", 8),
        ("map<int,int", 11),
        ("map<int,int,int,int,int>", 20),
        ("bitset", 6),
        ("bitset<", 7),
        ("bitset<x>", 8),
        ("bitset<8", 8),
        ("bitset<8,int>", 9),
        ("Template<int", 12),
        ("Template<int,", 13),
    ],
)
def test_malformed(typename, position):
    with pytest.raises(ValueError) as excinfo:
        parse_typename(typename)
    assert str(excinfo.value) == (
        f"invalid C++ type name syntax at char {position}\n\n"
        f"    {typename}\n" + "-" * (4 + position) + "^"
    )


def bad_branch():
    return types.SimpleNamespace(
        streamer=None,
        file=types.SimpleNamespace(file_path="some/file.root"),
        object_path="/tree;1:branch",
    )


def double32_leaf(title):
    return types.SimpleNamespace(classname="TLeafD32", member=lambda name: title)


@pytest.mark.parametrize(
    "title",
    [
        "",
        "no brackets at all",
        "[not valid python",
        "[1,,2]",
        "[10]",
        "[fN]",
        "[0, 'a']",
        "[0, 1, fN]",
        "[0, 1, 2.5]",
        "[0, 1, 2, 3]",
        "[0, f(1)]",
    ],
)
def test_float16_or_double32_without_ranges(title):
    leaf = double32_leaf(title)

    assert _float16_or_double32(bad_branch(), {}, leaf, False, ()) == AsDtype(
        numpy.dtype((">f4", ())), numpy.dtype(("f8", ()))
    )
    assert _float16_or_double32(bad_branch(), {}, leaf, True, ()) == AsFloat16(
        0.0, 0.0, 12, ()
    )


@pytest.mark.parametrize(
    "dtype",
    [">c16", ">c8", ">f2", "S8", "U4", ">M8[s]"],
)
def test_dtype_without_an_awkward_form(dtype):
    with pytest.raises(TypeError, match=r"^no Awkward Form for NumPy dtype dtype\("):
        uproot._util.awkward_form(numpy.dtype(dtype), None, {})


root_macro = r"""
#include "TFile.h"
#include "TTree.h"

class Test1824CovInterpTruncated {
public:
    Double32_t truncated_double; // [0, 0, 12]
    Float16_t truncated_float;   // [0, 0, 10]
    Double32_t ranged;           // [0, 10, 8]
    Test1824CovInterpTruncated() : truncated_double(1.5), truncated_float(-2.5), ranged(5) { }
    ClassDef(Test1824CovInterpTruncated, 1)
};

void write_test_1824_cov_interp_identify(const char* path) {
    TFile file(path, "RECREATE");
    TTree tree("tree", "tree");

    ULong_t unsigned_long = 18446744073709551615UL;
    Long_t signed_long = -5;
    Int_t n = 2;
    Float_t floats[8] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};
    Int_t ints[24];
    for (int i = 0;  i < 24;  i++) {
        ints[i] = i;
    }
    char chars[32] = "hello";

    tree.Branch("unsigned_long", &unsigned_long, "unsigned_long/g");
    tree.Branch("signed_long", &signed_long, "signed_long/G");
    tree.Branch("n", &n, "n/I");
    tree.Branch("both_jagged", floats, "a[n]/F:b[n]/F");
    tree.Branch("half_jagged", floats, "a[n]/F:b/F");
    tree.Branch("same_shape", ints, "a[2][3]/I:b[2][3]/I");
    tree.Branch("common_shape", ints, "a[2][3]/I:b[2][4]/I");
    tree.Branch("no_common_shape", ints, "a[2]/I:b[3]/I");
    tree.Branch("strings_and_ints", chars, "s/C:i/I");

    Double32_t leaf_double = 1.5;
    Float16_t leaf_float = -2.5;
    tree.Branch("leaf_double", &leaf_double, "leaf_double/d");
    tree.Branch("leaf_float", &leaf_float, "leaf_float/f");

    auto* truncated = new Test1824CovInterpTruncated();
    tree.Branch("truncated", &truncated);

    tree.Fill();
    tree.Write();
    file.Close();
}
"""


@pytest.fixture(scope="module")
def root_tree(tmp_path_factory):
    ROOT = pytest.importorskip("ROOT")
    path = tmp_path_factory.mktemp("root") / "written_by_root.root"

    assert ROOT.gInterpreter.Declare(root_macro)
    ROOT.write_test_1824_cov_interp_identify(str(path))

    with uproot.open(path) as file:
        yield file["tree"]


@pytest.mark.parametrize(
    "name, reason",
    [
        ("both_jagged", "leaf-list with non-null fLeafCount"),
        ("half_jagged", "leaf-list with different jaggedness among the leaves"),
        ("strings_and_ints", "more than one TLeaf (2) in a non-numerical TBranch"),
    ],
)
def test_leaf_lists_without_interpretations(root_tree, name, reason):
    branch = root_tree[name]

    with pytest.raises(UnknownInterpretation) as excinfo:
        interpretation_of(branch, {})
    assert excinfo.value.reason == reason
    assert excinfo.value.file_path == branch.file.file_path
    assert excinfo.value.object_path == branch.object_path


@pytest.fixture
def branch_object():
    path = skhep_testdata.data_path("uproot-issue-tbranch-of-th2.root")
    with uproot.open(path) as file:
        yield file["g4SimHits/tree/histogram"]


def test_tbranchobject_without_a_type(branch_object):
    del branch_object._members["fClassName"]

    interpretation = interpretation_of(branch_object, {})

    assert isinstance(interpretation, AsObjects)
    assert interpretation.model == AsDynamic()
    assert interpretation.typename == "void*"


@pytest.fixture
def language_tree(tmp_path):
    path = tmp_path / "language.root"
    with uproot.recreate(path) as file:
        file.mktree("tree", {"x": "float64", "y": "float64"})
        file["tree"].extend({"x": [1.0, 4.0, 9.0], "y": [-1.0, 0.25, 2.5]})

    with uproot.open(path) as file:
        yield file["tree"]


def test_getter_with_something_other_than_a_constant_string(language_tree):
    with pytest.raises(TypeError) as excinfo:
        language_tree.arrays(["get(x)"], library="np")
    assert str(excinfo.value).startswith(
        "expected a constant string as the only argument of 'get'; found "
    )
    assert "Name" in str(excinfo.value)

    with pytest.raises(
        TypeError, match="expected a constant string as the only argument of 'get'"
    ):
        language_tree.arrays(["get('x', 'y')"], library="np")


pd = library._libraries["pd"]


@pytest.fixture
def library_tree(tmp_path):
    path = tmp_path / "library.root"
    with uproot.recreate(path) as file:
        file.mktree("tree", {"x": "int32", "y": "float64", "z": "var * int32"})
        file["tree"].extend(
            {"x": [1, 2], "y": [1.5, 2.5], "z": awkward.Array([[1], []])}
        )
        file["tree"].extend({"x": [3], "y": [3.5], "z": awkward.Array([[2, 3]])})

    with uproot.open(path) as file:
        yield file["tree"]


@pytest.mark.parametrize(
    "name, how, message",
    [
        (
            "np",
            "zip",
            r"^for library np, how must be tuple, list, dict, or None \(for dict\)$",
        ),
        (
            "ak",
            "wrong",
            r'^for library ak, how must be tuple, list, dict, "zip" for a record array '
            r"with jagged arrays zipped, if possible, or None, for an unzipped record "
            r"array$",
        ),
        (
            "pd",
            5,
            r"^for library pd, how must be tuple, list, dict, or None or a str \(for a "
            r"DataFrame of all arrays, concatenated along columns; the value of the str "
            r"is ignored\)$",
        ),
    ],
)
def test_group_invalid_how(name, how, message):
    if name == "pd":
        pytest.importorskip("pandas")

    with pytest.raises(TypeError, match=message):
        library._libraries[name].group({}, [], how)


def test_zip_of_no_arrays_is_an_empty_record_array(library_tree):
    for how in ["zip", None]:
        arrays = library_tree.arrays([], how=how)
        assert isinstance(arrays, awkward.Array)
        assert str(arrays.type) == "0 * {}"


def strided_interpretation():
    return AsStridedObjects(
        Model_TObject, [("a", numpy.dtype(">i4")), ("b", numpy.dtype(">f8"))]
    )


def strided_array():
    data = numpy.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=[("a", "i4"), ("b", "f8")])
    return StridedObjectArray(strided_interpretation(), data)


@pytest.mark.parametrize("offsets_dtype", ["i4", "i8"])
def test_awkward_finalize_jagged_strided(offsets_dtype):
    jagged = JaggedArray(numpy.array([0, 1, 3], dtype=offsets_dtype), strided_array())

    out = ak.finalize(jagged, None, jagged, 0, 2, {})

    assert out.layout.offsets.dtype == numpy.dtype(offsets_dtype)
    assert out.a.tolist() == [[1], [2, 3]]
    assert out.b.tolist() == [[1.5], [2.5, 3.5]]
    assert out.layout.content.parameter("__record__") == "TObject"


@pytest.mark.parametrize("offsets_dtype", ["i4", "i8"])
def test_awkward_finalize_jagged(offsets_dtype):
    jagged = JaggedArray(
        numpy.array([0, 1, 3], dtype=offsets_dtype), numpy.array([1.5, 2.5, 3.5])
    )

    out = ak.finalize(jagged, None, None, 0, 2, {})

    assert out.layout.offsets.dtype == numpy.dtype(offsets_dtype)
    assert out.tolist() == [[1.5], [2.5, 3.5]]


@pytest.mark.parametrize("offsets_dtype", ["i4", "u4", "i8"])
def test_awkward_finalize_strings(offsets_dtype):
    strings = StringArray(numpy.array([0, 1, 3], dtype=offsets_dtype), b"abc")

    out = ak.finalize(strings, None, None, 0, 2, {})

    assert out.layout.offsets.dtype == numpy.dtype(offsets_dtype)
    assert out.tolist() == ["a", "bc"]


strings = {
    "class": "ListOffsetArray",
    "offsets": "i64",
    "content": {
        "class": "NumpyArray",
        "primitive": "uint8",
        "parameters": {"__array__": "char"},
    },
    "parameters": {"__array__": "string"},
}


kByteCountMask = 0x40000000


kNewClassTag = 0xFFFFFFFF


def ints(*values):
    return struct.pack(f">{len(values)}i", *values)


def size(n):
    return struct.pack(">I", n)


def header(payload, version=9):
    """The 6-byte (num_bytes, version) header that precedes objects."""
    return struct.pack(">IH", kByteCountMask | (len(payload) + 2), version) + payload


def memberwise(payload, version=9):
    """A header with the memberwise bit set; 6 more bytes are not interpreted."""
    return header(b"\x00" * 6 + payload, version | 0x4000)


def string(text):
    if len(text) < 255:
        return bytes([len(text)]) + text
    else:
        return b"\xff" + size(len(text)) + text


def as_uint8_jagged(entries):
    return awkward.unflatten(
        numpy.frombuffer(b"".join(entries), dtype=numpy.uint8),
        [len(x) for x in entries],
    )


@pytest.fixture
def objects_raw_branch(tmp_path):
    """
    Makes a real TBranch with the given bytes in each entry: every argument is a
    list of entries that goes into a separate TBasket.
    """
    opened = []

    def make(*baskets):
        path = tmp_path / f"raw{len(opened)}.root"
        with uproot.recreate(path) as file:
            tree = file.mktree("tree", {"x": "var * uint8"})
            for entries in baskets:
                tree.extend({"x": as_uint8_jagged(entries)})
        opened.append(uproot.open(path))
        return opened[-1]["tree"]["x"]

    yield make

    for file in opened:
        file.close()


def as_python(item):
    """Deserialized Python objects as what ak.Array.tolist() would show."""
    if isinstance(item, STLMap):
        return [(as_python(k), as_python(v)) for k, v in item.items()]
    elif isinstance(item, (str, bytes)):
        return item
    elif hasattr(item, "__iter__"):
        return [as_python(x) for x in item]
    elif isinstance(item, numpy.generic):
        return item.item()
    else:
        return item


def read_both_ways(branch, model):
    """
    Reads with the Python reader and with the AwkwardForth reader, checking that
    the second is really AwkwardForth and that the results are the same.
    """
    python = branch.array(
        interpretation=AsObjects(model, branch), library="np", array_cache=None
    )

    interpretation = AsObjects(model, branch)
    forth = branch.array(interpretation=interpretation, library="ak", array_cache=None)
    assert interpretation._complete_forth_code is not None
    assert interpretation._forth is True

    assert as_python(python) == forth.tolist()
    return python, forth


@pytest.mark.parametrize(
    "model, entries, expected",
    [
        (
            AsVector(False, i4),
            [size(0), size(2) + ints(1, 2), size(1) + ints(-3)],
            [[], [1, 2], [-3]],
        ),
        (
            AsVector(True, i4),
            [header(size(2) + ints(1, 2)), header(size(0)), header(size(1) + ints(9))],
            [[1, 2], [], [9]],
        ),
        (
            AsRVec(True, f8),
            [header(size(1) + struct.pack(">d", 1.5)), header(size(0))],
            [[1.5], []],
        ),
        (
            AsList(False, i4),
            [size(2) + ints(7, 8), size(0)],
            [[7, 8], []],
        ),
        (
            AsSet(False, i4),
            [size(2) + ints(1, 2), size(1) + ints(5)],
            [[1, 2], [5]],
        ),
        (
            AsBitSet(False, 3),
            [size(3) + b"\x01\x00\x01", size(3) + b"\x00\x00\x01"],
            [[True, False, True], [False, False, True]],
        ),
        (
            AsVector(False, AsString(False)),
            [size(2) + string(b"ab") + string(b"c"), size(0)],
            [["ab", "c"], []],
        ),
        (
            AsVector(False, AsString(False)),
            [size(1) + string(b"x" * 300), size(1) + string(b"y")],
            [["x" * 300], ["y"]],
        ),
        (
            AsVector(False, AsString(True)),
            [size(2) + header(string(b"ab")) + header(string(b"")), size(0)],
            [["ab", ""], []],
        ),
        (
            AsVector(False, AsString(False, length_bytes="4")),
            [size(1) + size(3) + b"abc", size(2) + size(0) + size(1) + b"d"],
            [["abc"], ["", "d"]],
        ),
        (
            AsVector(False, AsVector(False, i4)),
            [size(2) + size(1) + ints(1) + size(0), size(1) + size(2) + ints(2, 3)],
            [[[1], []], [[2, 3]]],
        ),
        (
            AsVector(True, AsVector(True, i4)),
            [
                header(size(1) + header(size(1) + ints(1))),
                header(size(2) + header(size(0)) + header(size(1) + ints(2))),
            ],
            [[[1]], [[], [2]]],
        ),
        (
            AsMap(True, i4, f8),
            [
                memberwise(
                    size(2) + ints(1, 2) + struct.pack(">dd", 1.5, 2.5),
                ),
                memberwise(size(1) + ints(3) + struct.pack(">d", 3.5)),
            ],
            [[(1, 1.5), (2, 2.5)], [(3, 3.5)]],
        ),
        (
            AsMap(True, AsString(False), i4),
            [
                memberwise(size(2) + string(b"a") + string(b"bc") + ints(1, 2)),
                memberwise(size(1) + string(b"d") + ints(3)),
            ],
            [[("a", 1), ("bc", 2)], [("d", 3)]],
        ),
        (
            AsMap(True, i4, AsString(False)),
            [
                memberwise(size(2) + ints(1, 2) + string(b"a") + string(b"bc")),
                memberwise(size(1) + ints(3) + string(b"d")),
            ],
            [[(1, "a"), (2, "bc")], [(3, "d")]],
        ),
        (
            AsMap(True, AsString(True), AsVector(True, i4)),
            [
                memberwise(
                    size(2)
                    + b"\x00" * 6
                    + string(b"a")
                    + string(b"b")
                    + b"\x00" * 6
                    + size(1)
                    + ints(1)
                    + size(2)
                    + ints(2, 3)
                ),
            ],
            [[("a", [1]), ("b", [2, 3])]],
        ),
        (
            AsArray(True, False, AsString(False)),
            [header(string(b"a") + string(b"bc")), header(string(b"d"))],
            [["a", "bc"], ["d"]],
        ),
        (
            AsArray(False, False, AsString(False)),
            [string(b"a") + string(b"bc"), string(b"d")],
            [["a", "bc"], ["d"]],
        ),
        (
            AsArray(False, True, AsString(False)),
            [b"\x00" + string(b"a") + string(b"bc"), b"\x00" + string(b"d")],
            [["a", "bc"], ["d"]],
        ),
        (
            AsArray(False, False, AsString(False), (2,)),
            [
                string(b"a") + string(b"b") + string(b"c") + string(b"d"),
                string(b"e") + string(b"f"),
            ],
            [[["a", "b"], ["c", "d"]], [["e", "f"]]],
        ),
        (
            AsArray(True, False, AsString(False), (2,)),
            [
                header(string(b"a") + string(b"b") + string(b"c") + string(b"d")),
                header(string(b"e") + string(b"f")),
            ],
            [[["a", "b"], ["c", "d"]], [["e", "f"]]],
        ),
        (
            # not at the top level, an array with a header ends where its header
            # says, not at the end of the entry
            AsVector(False, AsArray(True, False, AsString(False))),
            [
                size(3)
                + header(string(b"a") + string(b"bc"))
                + header(b"")
                + header(string(b"d")),
                size(0),
            ],
            [[["a", "bc"], [], ["d"]], []],
        ),
    ],
    ids=lambda x: x.cache_key if hasattr(x, "cache_key") else "",
)
def test_python_and_forth_agree(objects_raw_branch, model, entries, expected):
    python, forth = read_both_ways(objects_raw_branch(entries), model)
    assert forth.tolist() == expected
    assert isinstance(python, numpy.ndarray)
    assert python.dtype == numpy.dtype(object)


@pytest.mark.parametrize(
    "model, entries, expected",
    [
        # Forth is not attempted for non-memberwise maps
        (
            AsMap(False, i4, i4),
            [size(2) + ints(2, 20, 1, 10), size(0), size(1) + ints(5, 50)],
            [{1: 10, 2: 20}, {}, {5: 50}],
        ),
        (
            AsMap(True, i4, i4),
            [header(size(2) + ints(2, 20, 1, 10)), header(size(0))],
            [{1: 10, 2: 20}, {}],
        ),
        # strings of non-memberwise maps have no headers, whatever the AsString says
        (
            AsMap(True, AsString(True), AsString(True)),
            [
                header(
                    size(2) + string(b"b") + string(b"B") + string(b"a") + string(b"A")
                )
            ],
            [{"a": "A", "b": "B"}],
        ),
        (
            AsMap(False, i4, AsString(False)),
            [size(2) + ints(2) + string(b"b") + ints(1) + string(b"a")],
            [{1: "a", 2: "b"}],
        ),
        # NumPy dtypes without AwkwardForth symbols
        (
            AsVector(False, numpy.dtype("<i4")),
            [size(2) + struct.pack("<ii", 1, 2), size(0)],
            [[1, 2], []],
        ),
        (
            AsMap(True, numpy.dtype("<i4"), i4),
            [memberwise(size(1) + struct.pack("<i", 1) + ints(2))],
            [{1: 2}],
        ),
        # arrays of fixed-width numbers should have been simplified
        (
            AsArray(False, False, i4),
            [ints(1, 2), ints(3)],
            [[1, 2], [3]],
        ),
        (
            AsArray(False, True, i4),
            [b"\x00" + ints(1, 2), b"\x00" + ints(3)],
            [[1, 2], [3]],
        ),
        (
            AsArray(True, False, i4),
            [header(ints(1, 2)), header(ints(3))],
            [[1, 2], [3]],
        ),
    ],
    ids=lambda x: x.cache_key if hasattr(x, "cache_key") else "",
)
def test_models_that_cannot_be_forth(objects_raw_branch, model, entries, expected):
    branch = objects_raw_branch(entries)

    python = branch.array(
        interpretation=AsObjects(model, branch), library="np", array_cache=None
    )
    assert [x.tolist() for x in python] == expected

    interpretation = AsObjects(model, branch)
    forth = branch.array(interpretation=interpretation, library="ak", array_cache=None)
    assert interpretation._forth is False
    assert interpretation._complete_forth_code is None
    if isinstance(model, AsMap):
        assert [dict(x) for x in forth.tolist()] == expected
    else:
        assert forth.tolist() == expected


def test_map_read_empty_has_no_keys(objects_raw_branch):
    # objectwise, each pair is a key followed by its value
    branch = objects_raw_branch([size(0), size(2) + ints(5, 50, 6, 60)])
    result = branch.array(
        interpretation=AsObjects(AsMap(False, i4, i4), branch), library="np"
    )
    assert isinstance(result[0], STLMap)
    assert len(result[0]) == 0
    assert result[0].tolist() == {}
    assert result[1].tolist() == {5: 50, 6: 60}
    # an empty map has the dtypes of the others
    assert result[0].keys().dtype == result[1].keys().dtype == numpy.dtype(numpy.int32)
    assert (
        result[0].values().dtype == result[1].values().dtype == numpy.dtype(numpy.int32)
    )


def test_array_with_header_does_not_read_past_itself(objects_raw_branch):
    # two arrays in the same entry: each one has to stop where its header says
    entry = header(ints(1, 2)) + header(ints(3))
    model = AsVector(False, AsArray(True, False, i4))
    branch = objects_raw_branch([size(2) + entry])

    result = branch.array(
        interpretation=AsObjects(model, branch), library="np", array_cache=None
    )
    assert [x.tolist() for x in result[0]] == [[1, 2], [3]]


@pytest.mark.parametrize("values", [i4, AsString(False)])
def test_array_header_without_a_byte_count(objects_raw_branch, values):
    # only a version: the array's length is unknown
    branch = objects_raw_branch([struct.pack(">H", 9) + ints(1, 2)])
    with pytest.raises(uproot.deserialization.DeserializationError) as excinfo:
        branch.array(
            interpretation=AsObjects(AsArray(True, False, values), branch),
            library="np",
        )
    assert excinfo.value.message.endswith(
        "has a header without a byte count, so its length is unknown"
    )


def vector_of_vectors(*lists):
    out = size(len(lists))
    for values in lists:
        out += size(len(values)) + ints(*values)
    return out


nested_model = AsVector(False, AsVector(False, i4))


def mixed_baskets(objects_raw_branch):
    # In the first basket, the type of the innermost vector is unknown, so it
    # takes Python; the second basket reveals it, so that it takes Forth.
    return objects_raw_branch(
        [vector_of_vectors(), vector_of_vectors()],
        [vector_of_vectors([1, 2]), vector_of_vectors([], [3])],
    )


def test_baskets_of_python_and_forth_in_pandas(objects_raw_branch):
    pytest.importorskip("pandas")
    pytest.importorskip("awkward_pandas")

    branch = mixed_baskets(objects_raw_branch)
    interpretation = AsObjects(nested_model, branch)

    result = branch.array(interpretation=interpretation, library="pd")

    assert interpretation._complete_forth_code is not None
    assert [x.tolist() for x in result] == [[], [], [[1, 2]], [[], [3]]]


class ForthIsGivenUpOnElsewhere(AsObjects):
    # same as another thread giving up on Forth just before this one gets the lock
    def hook_before_basket_array(self, *args, **kwargs):
        self.hook_calls = getattr(self, "hook_calls", 0) + 1
        if self.hook_calls == 2:
            self._forth = False


def test_forth_is_abandoned_by_another_thread(objects_raw_branch):
    branch = objects_raw_branch([size(2) + ints(1, 2)])
    interpretation = ForthIsGivenUpOnElsewhere(AsVector(False, i4), branch)

    result = branch.array(interpretation=interpretation, library="ak")

    assert interpretation._forth is False
    assert interpretation._complete_forth_code is None
    assert result.tolist() == [[1, 2]]


def object_array(branch, *entries, cursor_offset=0):
    return ObjectArray(
        AsVector(False, i4),
        branch,
        dict(branch.context),
        numpy.cumsum([0] + [len(x) for x in entries]),
        numpy.frombuffer(b"".join(entries), dtype=numpy.uint8),
        cursor_offset,
    )


def test_object_array(objects_raw_branch):
    branch = objects_raw_branch([size(0)])
    entries = [size(1) + ints(1), size(0), size(2) + ints(2, 3)]
    array = object_array(branch, *entries, cursor_offset=7)

    assert array.model == AsVector(False, i4)
    assert array.branch is branch
    assert array.context["in_TBranch"] is True
    assert array.byte_offsets.tolist() == [0, 8, 12, 24]
    assert array.byte_content.tobytes() == b"".join(entries)
    assert array.cursor_offset == 7
    assert repr(array).startswith("ObjectArray(AsVector(False, dtype('>i4')), <TBranch")

    assert len(array) == 3
    assert array[0].tolist() == [1]
    assert array[2].tolist() == [2, 3]
    assert array[-1].tolist() == [2, 3]
    assert array[-3].tolist() == [1]
    for where in 3, -4:
        with pytest.raises(IndexError):
            array[where]
    assert [x.tolist() for x in array] == [[1], [], [2, 3]]
    assert [x.tolist() for x in array.to_numpy()] == [[1], [], [2, 3]]


@pytest.mark.parametrize(
    "where, expected",
    [
        (slice(None), [[1], [], [2, 3]]),
        (slice(1, None), [[], [2, 3]]),
        (slice(0, 2), [[1], []]),
        (slice(1, 2), [[]]),
        (slice(-1, None), [[2, 3]]),
        (slice(None, -1), [[1], []]),
        (slice(2, 1), []),
        (slice(5, 7), []),
    ],
)
def test_object_array_slices(objects_raw_branch, where, expected):
    branch = objects_raw_branch([size(0)])
    array = object_array(
        branch, size(1) + ints(1), size(0), size(2) + ints(2, 3), cursor_offset=7
    )
    sliced = array[where]

    assert isinstance(sliced, ObjectArray)
    assert len(sliced) == len(expected)
    assert [x.tolist() for x in sliced] == expected


@pytest.mark.parametrize("where", [slice(None, None, 2), "name", [0, 1]])
def test_object_array_unsupported_selections(objects_raw_branch, where):
    branch = objects_raw_branch([size(0)])
    array = object_array(branch, size(0), size(0))

    with pytest.raises(NotImplementedError) as excinfo:
        array[where]
    assert str(excinfo.value) == repr(where)


def test_object_array_iteration_finds_references(objects_raw_branch):
    # Each entry is a vector of two pointers to the same TObjString: the second
    # one is a reference to the position of the first in the whole buffer.
    branch = objects_raw_branch([size(0)])
    kMapOffset = 2

    def entry(position, text):
        tobject = struct.pack(">HII", 1, 0, 0x03000000)
        instance = header(tobject + string(text), version=1)
        new_object = struct.pack(">II", 0, kNewClassTag) + b"TObjString\x00" + instance
        new_object = (
            struct.pack(">I", kByteCountMask | (len(new_object) - 4)) + new_object[4:]
        )
        reference = struct.pack(">I", position + 4 + kMapOffset)
        return size(2) + new_object + reference

    first = entry(0, b"first")
    second = entry(len(first), b"second")

    array = ObjectArray(
        AsVector(False, AsPointer()),
        branch,
        dict(branch.context),
        numpy.array([0, len(first), len(first) + len(second)]),
        numpy.frombuffer(first + second, dtype=numpy.uint8),
        0,
    )

    by_index = [array[i] for i in range(2)]
    by_iteration = list(array)
    assert [[str(x) for x in vector] for vector in by_index] == [
        ["first", "first"],
        ["second", "second"],
    ]
    assert [[str(x) for x in vector] for vector in by_iteration] == [
        ["first", "first"],
        ["second", "second"],
    ]


def test_as_objects_eq_of_arrays():
    assert AsObjects(AsArray(True, False, i4)) == AsObjects(AsArray(True, False, i4))
    assert AsObjects(AsArray(True, False, i4)) != AsObjects(AsArray(False, False, i4))


def test_simplify_without_branch():
    simple = AsObjects(Model_TObject).simplify()
    assert isinstance(simple, AsObjects)

    # no file means no streamers to find out if the items are fixed-width
    array = AsArray(False, False, Model_TObject)
    simple = AsObjects(array).simplify()
    assert isinstance(simple, AsObjects)
    assert simple.model == array

    jagged = AsObjects(AsArray(False, True, i4)).simplify()
    assert isinstance(jagged, AsJagged)
    assert jagged.header_bytes == 1
    assert jagged.original == AsArray(False, True, i4)

    jagged = AsObjects(AsVector(True, i4)).simplify()
    assert jagged.header_bytes == 10

    strings = AsObjects(AsString(True)).simplify()
    assert isinstance(strings, AsStrings)
    assert strings.header_bytes == 6


def test_strided_object_array():
    data = numpy.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=[("a", "i4"), ("b", "f8")])
    array = StridedObjectArray(strided_interpretation(), data)

    assert array.array is data
    assert array.shape == (3,)
    assert len(array) == 3
    assert repr(array).startswith("StridedObjectArray(AsStridedObjects(Model_TObject)")

    item = array[1]
    assert isinstance(item, Model_TObject)
    assert item.member("a") == 2
    assert item.member("b") == 2.5

    part = array[1:]
    assert isinstance(part, StridedObjectArray)
    assert [x.member("a") for x in part] == [2, 3]

    assert [x.member("b") for x in array] == [1.5, 2.5, 3.5]
    assert [(i, x.member("a")) for i, x in array.ndenumerate()] == [
        ((0,), 1),
        ((1,), 2),
        ((2,), 3),
    ]


def test_strided_object_array_of_two_dimensions():
    data = numpy.array(
        [[(1, 1.5), (2, 2.5)], [(3, 3.5), (4, 4.5)]],
        dtype=[("a", "i4"), ("b", "f8")],
    )
    interpretation = strided_interpretation()
    interpretation.reshape((2,))
    array = StridedObjectArray(interpretation, data)

    assert array.shape == (2, 2)
    rows = list(array)
    assert all(isinstance(row, StridedObjectArray) for row in rows)
    assert [x.member("a") for x in rows[1]] == [3, 4]


NUMBERS_ONLY = awkward.forms.NumpyForm("int32")


@pytest.mark.parametrize(
    "form, expected",
    [
        (awkward.forms.NumpyForm("int8"), awkward.forms.NumpyForm("int64")),
        (awkward.forms.NumpyForm("uint32"), awkward.forms.NumpyForm("int64")),
        (awkward.forms.NumpyForm("float32"), awkward.forms.NumpyForm("float64")),
        (awkward.forms.NumpyForm("bool"), awkward.forms.NumpyForm("bool")),
        (
            awkward.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
            awkward.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
        ),
        (
            awkward.forms.NumpyForm("int32", inner_shape=(3,)),
            awkward.forms.NumpyForm("int64", inner_shape=(3,)),
        ),
        (awkward.forms.EmptyForm(), awkward.forms.EmptyForm()),
        (
            awkward.forms.BitMaskedForm("u8", NUMBERS_ONLY, True, False),
            awkward.forms.BitMaskedForm(
                "u8", awkward.forms.NumpyForm("int64"), True, False
            ),
        ),
        (
            awkward.forms.ByteMaskedForm("i8", NUMBERS_ONLY, True),
            awkward.forms.ByteMaskedForm("i8", awkward.forms.NumpyForm("int64"), True),
        ),
        (
            awkward.forms.IndexedForm("i64", NUMBERS_ONLY),
            awkward.forms.IndexedForm("i64", awkward.forms.NumpyForm("int64")),
        ),
        (
            awkward.forms.IndexedOptionForm("i64", NUMBERS_ONLY),
            awkward.forms.IndexedOptionForm("i64", awkward.forms.NumpyForm("int64")),
        ),
        (
            awkward.forms.ListForm("i64", "i64", NUMBERS_ONLY),
            awkward.forms.ListForm("i64", "i64", awkward.forms.NumpyForm("int64")),
        ),
        (
            awkward.forms.ListOffsetForm("i32", NUMBERS_ONLY),
            awkward.forms.ListOffsetForm("i32", awkward.forms.NumpyForm("int64")),
        ),
        (
            awkward.forms.RecordForm([NUMBERS_ONLY], ["x"]),
            awkward.forms.RecordForm([awkward.forms.NumpyForm("int64")], ["x"]),
        ),
        (
            awkward.forms.RecordForm([NUMBERS_ONLY], None),
            awkward.forms.RecordForm([awkward.forms.NumpyForm("int64")], None),
        ),
        (
            awkward.forms.RegularForm(NUMBERS_ONLY, 3),
            awkward.forms.RegularForm(awkward.forms.NumpyForm("int64"), 3),
        ),
        (
            awkward.forms.UnionForm(
                "i8",
                "i64",
                [NUMBERS_ONLY, awkward.forms.ListOffsetForm("i64", NUMBERS_ONLY)],
            ),
            awkward.forms.UnionForm(
                "i8",
                "i64",
                [
                    awkward.forms.NumpyForm("int64"),
                    awkward.forms.ListOffsetForm(
                        "i64", awkward.forms.NumpyForm("int64")
                    ),
                ],
            ),
        ),
        (
            awkward.forms.UnmaskedForm(NUMBERS_ONLY),
            awkward.forms.UnmaskedForm(awkward.forms.NumpyForm("int64")),
        ),
    ],
)
def test_awkward_form_of_iter(form, expected):
    assert uproot._util.awkward_form_of_iter(awkward, form) == expected


def test_recursively_fix_form_of_grouped_objects():
    interpretation = uproot.interpretation.grouped.AsGrouped(
        None,
        {
            "numerical": uproot.interpretation.numerical.AsDtype(">i4"),
            "objects": uproot.interpretation.objects.AsObjects(
                uproot.containers.AsVector(True, numpy.dtype(">i2"))
            ),
        },
    )
    form = awkward.forms.RecordForm(
        [
            awkward.forms.NumpyForm("int32"),
            awkward.forms.ListOffsetForm("i64", awkward.forms.NumpyForm("int16")),
        ],
        ["numerical", "objects"],
        parameters={"__doc__": "documented"},
    )

    fixed = uproot._util.recursively_fix_awkward_form_of_iter(
        awkward, interpretation, form
    )

    # only the members read with ak.from_iter get its wider numeric types
    assert fixed == awkward.forms.RecordForm(
        [
            awkward.forms.NumpyForm("int32"),
            awkward.forms.ListOffsetForm("i64", awkward.forms.NumpyForm("int64")),
        ],
        ["numerical", "objects"],
        parameters={"__doc__": "documented"},
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ListOffsetArray", "ListOffsetArray"),
        ("ListOffsetArray64", "ListOffsetArray"),
        ("ListOffsetArray32", "ListOffsetArray"),
        ("ListOffsetArrayU32", "ListOffsetArray"),
        ("ListArray64", "ListArray"),
        ("UnionArray8_32", "UnionArray"),
        ("UnionArray8_64", "UnionArray"),
        ("UnionArray8_U32", "UnionArray"),
    ],
)
def test_content_cls_from_name(name, expected):
    assert uproot._util._content_cls_from_name(awkward, name) is getattr(
        awkward.contents, expected
    )


def test_reset_classes_restores_every_class(reset_classes):
    assert set(uproot.model.bootstrap_classes()) == set(
        uproot.model.bootstrap_classnames
    )
    for name in "TClonesArray", "TTime", "TLeafG", "TH1F", "ROOT::RNTuple":
        assert name in uproot.classes


def test_unpickled_in_a_process_without_the_dynamic_class(monkeypatch):
    # A streamer-generated class doesn't exist in a new process until it's unpickled.
    with uproot.open(skhep_testdata.data_path("uproot-issue-350.root")) as f:
        obj = f["data"]
        expected = obj.all_members

    name = type(obj).__name__
    assert type(obj).__module__ == "uproot.dynamic"
    data = pickle.dumps(obj)
    monkeypatch.delitem(vars(uproot.dynamic), name)

    restored = pickle.loads(data)

    assert type(restored) is not type(obj)
    assert type(restored).__name__ == name
    assert isinstance(restored, uproot.model.DynamicModel)
    assert type(restored).member_names == obj.member_names
    assert "_nominalBinWidth" in type(restored).member_names
    assert type(restored).base_names_versions == obj.base_names_versions
    assert restored.classname == obj.classname == "RooHist"
    assert restored.all_members.keys() == expected.keys()
    assert restored.member("fName") == expected["fName"]

    # the second time, the class already has its data
    again = pickle.loads(data)
    assert type(again) is type(restored)
    assert again.member("fName") == expected["fName"]


def sample_path(root_version):
    return skhep_testdata.data_path(f"uproot-sample-{root_version}-uncompressed.root")


def test_tleafg_from_root(tmp_path):
    ROOT = pytest.importorskip("ROOT")

    filename = str(tmp_path / "tleafg.root")
    file = ROOT.TFile(filename, "recreate")
    tree = ROOT.TTree("tree", "tree")
    x = numpy.zeros(1, dtype=numpy.int64)
    tree.Branch("x", x, "x/G")
    for i in range(3):
        x[0] = i * 2**40
        tree.Fill()
    file.Write()
    file.Close()

    with uproot.open(filename) as f:
        branch = f["tree"]["x"]
        assert isinstance(
            branch.member("fLeaves")[0], uproot.models.TLeaf.Model_TLeafG_v1
        )
        assert branch.array(library="np").tolist() == [0, 2**40, 2**41]


def test_tleafg(tmp_path):
    # ROOT's "/G" leaves (Long_t) are like "/L" (Long64_t) on disk.
    with open(sample_path("6.20.04"), "rb") as file:
        data = file.read()

    filename = tmp_path / "tleafg.root"
    filename.write_bytes(data.replace(b"TLeafL", b"TLeafG"))

    with uproot.open(filename) as f:
        branch = f["sample"]["i8"]
        leaf = branch.member("fLeaves")[0]
        assert isinstance(leaf, uproot.models.TLeaf.Model_TLeafG_v1)
        assert (leaf.member("fMinimum"), leaf.member("fMaximum")) == (0, 0)
        assert branch.array(library="np").tolist() == list(range(-15, 15))


def test_tref_branch():
    with uproot.open(skhep_testdata.data_path("uproot-Event.root")) as f:
        tree = f["T"]
        refs = tree["fLastTrack"].array(entry_stop=2, library="np")
        ref_arrays = tree["fHighPt"].array(entry_stop=2, library="np")

    assert [x.ref for x in refs] == [600, 604]
    assert repr(refs[0]) == "<TRef 600>"

    array = ref_arrays[1]
    assert isinstance(array, uproot.models.TRef.Model_TRefArray)
    assert array.name == ""
    assert len(array) == 366
    assert array.nbytes == 366 * 4
    assert array.refs.dtype == numpy.dtype(">i4")
    assert array[0] == array.refs[0] == 1
    assert list(array) == array.refs.tolist()
    assert repr(array).startswith(
        "<TRefArray [  1,   2,   3, ..., 601, 602, 604] at 0x"
    )


def test_tbasket_of_a_compressed_jagged_branch():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as f:
        branch = f["events"]["Jet_Px"]
        basket = branch.basket(0)
        count_branch = branch.count_branch

        assert repr(basket).startswith("<TBasket 0 of 'Jet_Px' at 0x")
        assert basket.key_version == 1004
        assert basket.compressed_bytes < basket.uncompressed_bytes
        assert basket.compressed_bytes == basket.member("fNbytes")
        assert [x[0] for x in basket.block_compression_info] == ["ZLIB"]
        assert sum(x[2] for x in basket.block_compression_info) == basket.member(
            "fObjlen"
        )
        # the raw data ends with the entry offsets
        assert len(basket.raw_data) > len(basket.data) == basket.border
        assert basket.raw_data[: basket.border].tobytes() == basket.data.tobytes()
        assert (
            basket.counts.tolist()
            == count_branch.array(entry_stop=basket.num_entries, library="np").tolist()
        )

        # with no explicit interpretation, it comes from the TBranch
        array = basket.array(library="np")
        assert len(array) == basket.num_entries
        expected = branch.array(entry_stop=basket.num_entries, library="np")
        assert array[0].tolist() == expected[0].tolist()
        assert array[1].tolist() == expected[1].tolist()
