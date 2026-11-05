# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of uproot.interpretation.identify: the C++ typename parser, driven by
tables of type names, and the heuristics of interpretation_of.
"""

from __future__ import annotations

import types

import awkward
import numpy
import pytest
import skhep_testdata

import uproot
import uproot.interpretation.custom
import uproot.interpretation.grouped
from uproot import const
from uproot.containers import AsArray, AsDynamic, AsPointer, AsString, AsVector
from uproot.interpretation.identify import (
    _float16_double32_walk_ast,
    _float16_or_double32,
    _from_leaves,
    _ftype_to_dtype,
    interpretation_of,
    parse_typename,
    register_interpretation,
    UnknownInterpretation,
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


i4 = numpy.dtype(">i4")


f8 = numpy.dtype(">f8")


@pytest.mark.parametrize("typename, expected", scalars.items())
def test_scalars(typename, expected):
    assert parse_typename(typename) == numpy.dtype(expected)


def test_typename_suggestions():
    assert parse_typename("TString").typename == "TString"
    assert parse_typename("char*").typename == "char*"


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


root_macro = r"""
#include "TFile.h"
#include "TTree.h"

class Test1829CovInterpTruncated {
public:
    Double32_t truncated_double; // [0, 0, 12]
    Float16_t truncated_float;   // [0, 0, 10]
    Double32_t ranged;           // [0, 10, 8]
    Test1829CovInterpTruncated() : truncated_double(1.5), truncated_float(-2.5), ranged(5) { }
    ClassDef(Test1829CovInterpTruncated, 1)
};

void write_test_1829_cov_interp_identify(const char* path) {
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

    auto* truncated = new Test1829CovInterpTruncated();
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
    ROOT.write_test_1829_cov_interp_identify(str(path))

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
