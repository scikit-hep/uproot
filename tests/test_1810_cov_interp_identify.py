# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of uproot.interpretation.identify: the C++ typename parser, driven by
tables of type names, and the heuristics of interpretation_of.
"""

from __future__ import annotations

import types

import numpy
import pytest
import skhep_testdata

import awkward
import uproot
import uproot.interpretation.custom
import uproot.interpretation.grouped
from uproot import const
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
)
from uproot.interpretation.identify import (
    UnknownInterpretation,
    _float16_double32_walk_ast,
    _float16_or_double32,
    _from_leaves,
    _ftype_to_dtype,
    interpretation_of,
    parse_typename,
    register_interpretation,
    unregister_interpretation,
)
from uproot.interpretation.numerical import AsDouble32, AsDtype, AsFloat16
from uproot.interpretation.objects import AsObjects
from uproot.models.TObject import Model_TObject

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

i4 = numpy.dtype(">i4")
f4 = numpy.dtype(">f4")
f8 = numpy.dtype(">f8")

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


@pytest.mark.parametrize("typename, expected", scalars.items())
def test_scalars(typename, expected):
    assert parse_typename(typename) == numpy.dtype(expected)


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("typename, expected", pointers.items())
def test_pointers(typename, expected, header):
    # the second AsArray argument is the speedbump, which depends on the header
    out = parse_typename(typename, outer_header=header)
    assert out == AsArray(False, header, numpy.dtype(expected))


@pytest.mark.parametrize("typename, expected", containers.items())
def test_containers(typename, expected):
    assert parse_typename(typename) == expected


def test_typename_suggestions():
    assert parse_typename("TString").typename == "TString"
    assert parse_typename("char*").typename == "char*"


@pytest.mark.parametrize(
    "typename", [*scalars, *pointers, *containers], ids=lambda x: x.replace(" ", "_")
)
def test_quoted_evaluates_to_the_same(typename):
    code = parse_typename(typename, quote=True)
    assert isinstance(code, str)
    assert eval(code, quoted_namespace()) == parse_typename(typename)


def test_quoted_template_class():
    code = parse_typename("Outer<Inner<int>, float>", quote=True)
    assert code == "c('Outer<Inner<int>,float>')"


def test_template_class_name_is_regularized(monkeypatch):
    class Model_Fake(uproot.model.Model):
        pass

    monkeypatch.setitem(uproot.classes, "Outer<int,float>", Model_Fake)

    assert parse_typename("Outer<int, float>") is Model_Fake
    assert parse_typename("Outer< int , float >") is Model_Fake


def test_headers():
    typename = "vector<vector<string> >"

    assert parse_typename(typename, outer_header=False) == AsVector(
        False, AsVector(False, AsString(False))
    )
    assert parse_typename(typename, inner_header=True) == AsVector(
        True, AsVector(True, AsString(True))
    )


def test_string_header():
    assert parse_typename("string", string_header=True) == AsString(True)
    assert parse_typename("string", string_header=True, outer_header=False) == AsString(
        False
    )
    assert parse_typename("std::string", string_header=False) == AsString(False)


def test_outermost_string_is_quoted_too():
    assert parse_typename("string", quote=True) == "uproot.containers.AsString(False)"


def test_class_named_in_file():
    class FakeFile:
        file_path = "some/file.root"

        def class_named(self, classname):
            assert classname == "TObject"
            return Model_TObject

    assert parse_typename("TObject", file=FakeFile()) is Model_TObject
    assert parse_typename("TObject**", file=FakeFile()) == AsPointer(
        AsPointer(Model_TObject)
    )


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


def test_malformed_with_file():
    file = types.SimpleNamespace(file_path="some/file.root")
    with pytest.raises(ValueError) as excinfo:
        parse_typename("vector<", file=file)
    assert str(excinfo.value) == (
        "invalid C++ type name syntax at char 7\n\n"
        "    vector<\n"
        "-----------^\n"
        "in file some/file.root"
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("kBool", "?"),
        ("kChar", "i1"),
        ("kUChar", "u1"),
        ("kShort", ">i2"),
        ("kUShort", ">u2"),
        ("kInt", ">i4"),
        ("kBits", ">u4"),
        ("kUInt", ">u4"),
        ("kCounter", ">u4"),
        ("kLong", ">i8"),
        ("kULong", ">u8"),
        ("kLong64", ">i8"),
        ("kULong64", ">u8"),
        ("kFloat", ">f4"),
        ("kDouble", ">f8"),
    ],
)
def test_ftype_to_dtype(name, expected):
    fType = getattr(const, name)
    assert _ftype_to_dtype(fType) == numpy.dtype(expected)
    assert _ftype_to_dtype(const.kOffsetL + fType) == numpy.dtype(expected)


@pytest.mark.parametrize("fType", [None, const.kObject, const.kFloat16, const.kOffsetP])
def test_ftype_not_numerical(fType):
    with pytest.raises(uproot.interpretation.identify.NotNumerical):
        _ftype_to_dtype(fType)


def bad_branch():
    return types.SimpleNamespace(
        streamer=None,
        file=types.SimpleNamespace(file_path="some/file.root"),
        object_path="/tree;1:branch",
    )


def double32_leaf(title):
    return types.SimpleNamespace(classname="TLeafD32", member=lambda name: title)


@pytest.mark.parametrize(
    "title, low, high, num_bits_double32, num_bits_float16",
    [
        ("[0, 1]", 0.0, 1.0, 32, 32),
        ("[-1, 1]", -1.0, 1.0, 32, 32),
        ("[0, 1, 16]", 0.0, 1.0, 16, 16),
        ("x[0, 1, 16]/d", 0.0, 1.0, 16, 16),
        ("[0, 1, 2]", 0.0, 1.0, 2, 2),
        ("[0, 1, 32]", 0.0, 1.0, 32, 32),
        ("[0, 1, 1]", 0.0, 1.0, 32, 32),
        ("[0, 1, 33]", 0.0, 1.0, 32, 32),
        ("[0, pi]", 0.0, 3.141592653589793, 32, 32),
        ("[-PI, Pi, 8]", -3.141592653589793, 3.141592653589793, 8, 8),
        ("[0, twopi]", 0.0, 6.283185307179586, 32, 32),
        ("[0, 2*pi]", 0.0, 6.283185307179586, 32, 32),
        ("[-pi/2, pi/2, 10]", -1.5707963267948966, 1.5707963267948966, 10, 10),
        ("[1 + 1, 4 - 1]", 2.0, 3.0, 32, 32),
    ],
)
def test_float16_or_double32_with_ranges(
    title, low, high, num_bits_double32, num_bits_float16
):
    leaf = double32_leaf(title)
    assert _float16_or_double32(bad_branch(), {}, leaf, False, (2,)) == AsDouble32(
        low, high, num_bits_double32, (2,)
    )
    assert _float16_or_double32(bad_branch(), {}, leaf, True, (2,)) == AsFloat16(
        low, high, num_bits_float16, (2,)
    )


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


def test_float16_or_double32_titles_from_streamers():
    branch = bad_branch()
    leaf = types.SimpleNamespace(classname="TLeafElement")

    branch.streamer = types.SimpleNamespace(title="[0, 1, 8]")
    assert _float16_or_double32(branch, {}, leaf, False, ()) == AsDouble32(0, 1, 8)

    branch.streamer = types.SimpleNamespace(title=None)
    assert _float16_or_double32(branch, {}, leaf, True, ()) == AsFloat16(0, 0, 12)

    branch.streamer = None
    assert _float16_or_double32(branch, {}, leaf, True, ()) == AsFloat16(0, 0, 12)


def test_float16_double32_walk_ast_only_takes_ast():
    with pytest.raises(UnknownInterpretation) as excinfo:
        _float16_double32_walk_ast(None, bad_branch(), "[source]")
    assert excinfo.value.reason == "cannot compute streamer title '[source]'"


def test_unknown_interpretation():
    unknown = UnknownInterpretation("reason", "file.root", "/tree;1:branch")

    assert repr(unknown) == "<UnknownInterpretation 'reason'>"
    assert str(unknown) == "reason\nin file file.root\nin object /tree;1:branch"
    assert unknown.typename == "unknown"


@pytest.mark.parametrize(
    "name",
    [
        "cache_key",
        "numpy_dtype",
        "basket_array",
        "final_array",
        "hook_before_basket_array",
        "hook_after_basket_array",
        "hook_before_final_array",
        "hook_before_library_finalize",
        "hook_after_final_array",
        "itemsize",
        "from_dtype",
        "to_dtype",
        "content",
        "header_bytes",
        "size_1to5_bytes",
    ],
)
def test_unknown_interpretation_raises_itself(name):
    unknown = UnknownInterpretation("reason", "file.root", "/tree;1:branch")

    with pytest.raises(UnknownInterpretation) as excinfo:
        getattr(unknown, name)
    assert excinfo.value is unknown


def test_unknown_interpretation_awkward_form_raises_itself():
    unknown = UnknownInterpretation("reason", "file.root", "/tree;1:branch")

    with pytest.raises(UnknownInterpretation) as excinfo:
        unknown.awkward_form(None)
    assert excinfo.value is unknown


@pytest.fixture
def written_tree(tmp_path):
    with uproot.recreate(tmp_path / "written.root") as file:
        file.mktree("tree", {"x": "int32", "y": "var * float64"})
        file["tree"].extend({"x": [1], "y": awkward.Array([[1.5]])})

    with uproot.open(tmp_path / "written.root") as file:
        yield file["tree"]


def test_swap_bytes(written_tree):
    native = interpretation_of(written_tree["x"], {})
    assert native.to_dtype == numpy.dtype("i4")

    interpretation = interpretation_of(written_tree["x"], {"swap_bytes": False})
    assert interpretation == AsDtype(">i4", ">i4")


def test_no_leaves(written_tree):
    branch = written_tree["x"]
    branch._members["fLeaves"] = []

    with pytest.raises(UnknownInterpretation) as excinfo:
        interpretation_of(branch, {})
    assert str(excinfo.value) == (
        f"leaf-list with zero leaves\nin file {branch.file.file_path}\n"
        f"in object {branch.object_path}"
    )


def test_registering_interpretations(written_tree):
    class Always(uproot.interpretation.custom.CustomInterpretation):
        @classmethod
        def match_branch(cls, branch, context, simplify):
            return True

    class AlwaysToo(Always):
        pass

    register_interpretation(Always)
    try:
        assert isinstance(interpretation_of(written_tree["x"], {}), Always)

        with pytest.warns(UserWarning, match="Overwriting existing custom"):
            register_interpretation(Always)

        register_interpretation(AlwaysToo)
        with pytest.raises(
            ValueError,
            match=r"^Multiple custom interpretations matched, uproot cannot determine "
            r"which one to use: \[",
        ):
            interpretation_of(written_tree["x"], {})
    finally:
        unregister_interpretation(Always)
        unregister_interpretation(AlwaysToo)

    assert interpretation_of(written_tree["x"], {}) == AsDtype(">i4")


def test_signed_long_inside_a_container():
    assert parse_typename("std::vector<signed long>").values == numpy.dtype(">i8")


class Incomplete(uproot.interpretation.Interpretation):
    pass


@pytest.mark.parametrize(
    "name, call",
    [
        ("cache_key", lambda x: x.cache_key),
        ("typename", lambda x: x.typename),
        ("numpy_dtype", lambda x: x.numpy_dtype),
        ("awkward_form", lambda x: x.awkward_form(None)),
        ("basket_array", lambda x: x.basket_array(*[None] * 8)),
        ("final_array", lambda x: x.final_array(*[None] * 7)),
    ],
)
def test_interpretation_without_a_method(name, call):
    with pytest.raises(NotImplementedError, match=rf"^Incomplete has no {name}$"):
        call(Incomplete())


def test_custom_interpretation_without_basket_array(written_tree):
    class OnlyMatches(uproot.interpretation.custom.CustomInterpretation):
        @classmethod
        def match_branch(cls, branch, context, simplify):
            return True

    register_interpretation(OnlyMatches)
    try:
        branch = written_tree["x"]
        with pytest.raises(
            NotImplementedError, match=r"^OnlyMatches has no basket_array$"
        ):
            branch.array(interpretation=interpretation_of(branch, {}))
    finally:
        unregister_interpretation(OnlyMatches)


def test_grouped_interpretation_has_no_numpy_dtype():
    path = skhep_testdata.data_path("uproot-issue-1043.root")
    with uproot.open(path) as file:
        interpretation = file["FooBar"]["Foo"].interpretation
        assert isinstance(interpretation, uproot.interpretation.grouped.AsGrouped)
        with pytest.raises(
            NotImplementedError, match=r"^AsGrouped has no numpy_dtype$"
        ):
            interpretation.numpy_dtype


@pytest.mark.parametrize(
    "name, call",
    [
        ("typename", lambda x: x.typename),
        ("numpy_dtype", lambda x: x.numpy_dtype),
        ("awkward_form", lambda x: x.awkward_form(None)),
    ],
)
def test_as_binary_is_only_for_reading_bytes(name, call):
    with pytest.raises(NotImplementedError, match=rf"^AsBinary has no {name}$"):
        call(uproot.interpretation.custom.AsBinary())


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

class Test1810CovInterpTruncated {
public:
    Double32_t truncated_double; // [0, 0, 12]
    Float16_t truncated_float;   // [0, 0, 10]
    Double32_t ranged;           // [0, 10, 8]
    Test1810CovInterpTruncated() : truncated_double(1.5), truncated_float(-2.5), ranged(5) { }
    ClassDef(Test1810CovInterpTruncated, 1)
};

void write_test_1810_cov_interp_identify(const char* path) {
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

    auto* truncated = new Test1810CovInterpTruncated();
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
    ROOT.write_test_1810_cov_interp_identify(str(path))

    with uproot.open(path) as file:
        yield file["tree"]


def test_tleafg(root_tree):
    assert root_tree["unsigned_long"].interpretation == AsDtype(">u8")
    assert root_tree["signed_long"].interpretation == AsDtype(">i8")

    assert root_tree["unsigned_long"].array(library="np").tolist() == [
        18446744073709551615
    ]
    assert root_tree["signed_long"].array(library="np").tolist() == [-5]


@pytest.mark.parametrize(
    "name, dims",
    [("same_shape", (2, 3)), ("common_shape", (2,)), ("no_common_shape", ())],
)
def test_leaf_lists_with_shapes(root_tree, name, dims):
    assert _from_leaves(root_tree[name], {}) == (dims, False)


def test_leaf_lists_with_swap_bytes():
    with uproot.open(skhep_testdata.data_path("uproot-leaflist.root")) as file:
        branch = file["tree/leaflist"]

        native = interpretation_of(branch, {})
        assert native.to_dtype == native.from_dtype.newbyteorder("=")

        swapped = interpretation_of(branch, {"swap_bytes": False})
        assert swapped.to_dtype == swapped.from_dtype
        assert swapped.from_dtype.names == ("x", "y", "z")


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


def test_tbranchobject(branch_object):
    assert branch_object.classname == "TBranchObject"

    interpretation = interpretation_of(branch_object, {})

    assert isinstance(interpretation, AsObjects)
    assert isinstance(interpretation.model, AsDynamic)
    assert interpretation.typename == "TH2F"


def test_tbranchobject_with_a_streamer_and_no_class_name(branch_object):
    del branch_object._members["fClassName"]
    branch_object._streamer = types.SimpleNamespace(typename="TH2F")

    interpretation = interpretation_of(branch_object, {})

    assert isinstance(interpretation, AsObjects)
    assert isinstance(interpretation.model, AsDynamic)
    assert interpretation.typename == "TH2F"


def test_tbranchobject_without_a_type(branch_object):
    del branch_object._members["fClassName"]

    interpretation = interpretation_of(branch_object, {})

    assert isinstance(interpretation, AsObjects)
    assert interpretation.model == AsDynamic()
    assert interpretation.typename == "void*"


def test_object_stored_inline_is_not_a_pointer():
    with uproot.open(skhep_testdata.data_path("uproot-Event.root")) as file:
        branch = file["T/event/fH"]

        assert branch.streamer.typename == "TH1F*"
        assert branch.streamer.member("fType") == const.kObjectp

        interpretation = interpretation_of(branch, {})

        assert isinstance(interpretation, AsObjects)
        assert interpretation.typename == "TH1F"


def test_pointers_in_a_tclonesarray():
    with uproot.open(skhep_testdata.data_path("uproot-issue390.root")) as file:
        branch = file["E/Evt/AAObject/usr_data/usr_data.holder"]

        assert branch._streamer_isTClonesArray is False
        interpretation = interpretation_of(branch, {})

        assert branch._streamer_isTClonesArray is True
        assert isinstance(interpretation.model, AsArray)
        assert interpretation.model.header is True
        assert isinstance(interpretation.model.values, AsPointer)


@pytest.mark.parametrize(
    "name, interpretation, expected",
    [
        ("leaf_double", AsDtype(">f4", "f8"), 1.5),
        ("leaf_float", AsFloat16(0, 0, 12), -2.5),
        ("truncated/truncated_double", AsDouble32(0.0, 0.0, 12), 1.5),
        ("truncated/truncated_float", AsFloat16(0.0, 0.0, 10), -2.5),
        ("truncated/ranged", AsDouble32(0.0, 10.0, 8), 5.0),
    ],
)
def test_truncated_numbers_from_root(root_tree, name, interpretation, expected):
    branch = root_tree[name]

    assert branch.interpretation == interpretation
    assert branch.array(library="np").tolist() == [expected]
    assert branch.array(library="ak").tolist() == [expected]
