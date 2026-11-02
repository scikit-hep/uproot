# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of the interpretations that are not objects: uproot.interpretation.numerical,
jagged, strings, grouped, and custom, on real TBranches with crafted bytes.
"""

from __future__ import annotations

import struct

import numpy
import pytest
import skhep_testdata

import awkward
import uproot
from uproot.interpretation import library
from uproot.interpretation.custom import AsBinary, CustomInterpretation
from uproot.interpretation.grouped import AsGrouped
from uproot.interpretation.jagged import AsJagged, JaggedArray, fast_divide
from uproot.interpretation.numerical import (
    AsDouble32,
    AsDtype,
    AsFloat16,
    AsSTLBits,
)
from uproot.interpretation.strings import AsStrings, StringArray

np = library._libraries["np"]
ak = library._libraries["ak"]
pd = library._libraries["pd"]


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


def test_as_dtype_repr():
    assert repr(AsDtype(">i4")) == "AsDtype('>i4')"
    assert repr(AsDtype(">i4", ">f8")) == "AsDtype('>i4', '>f8')"


def test_as_dtype_in_place(tree):
    branch = tree["x"]
    output = numpy.zeros(5, dtype="i4")

    result = branch.array(
        interpretation=branch.interpretation.inplace(output), library="np"
    )

    assert result.tolist() == [1, 2, 3]
    assert output.tolist() == [1, 2, 3, 0, 0]


def test_as_dtype_in_place_needs_numpy(tree):
    branch = tree["x"]
    interpretation = branch.interpretation.inplace(numpy.zeros(5, dtype="i4"))

    with pytest.raises(
        TypeError,
        match=r"^AsDtypeInPlace can only be used with library 'np', not 'ak'$",
    ):
        branch.array(interpretation=interpretation, library="ak")


def test_as_dtype_in_place_needs_enough_room(tree):
    branch = tree["x"]
    interpretation = branch.interpretation.inplace(numpy.zeros(2, dtype="i4"))

    with pytest.raises(
        ValueError,
        match=r"^Requesting to fill an array of size 2 \(type int32\) with input of "
        r"size 3 \(type >i4\)$",
    ):
        branch.array(interpretation=interpretation, library="np")


def test_as_stl_bits():
    with pytest.raises(NotImplementedError):
        AsSTLBits()

    instance = AsSTLBits.__new__(AsSTLBits)
    instance._num_bytes = 3
    assert instance.itemsize == 7


@pytest.mark.parametrize("cls", [AsDouble32, AsFloat16])
def test_truncated_numerical_properties(cls):
    interpretation = cls(-1.5, 2.5, 16, (2, 3))

    assert interpretation.low == -1.5
    assert interpretation.high == 2.5
    assert interpretation.num_bits == 16
    assert interpretation.to_dims == (2, 3)
    assert interpretation.is_truncated is False
    assert interpretation.from_dtype == numpy.dtype(">u4")
    assert interpretation.itemsize == 4
    assert repr(interpretation) == f"{cls.__name__}(-1.5, 2.5, 16, to_dims=(2, 3))"
    assert interpretation.cache_key == f"{cls.__name__}(-1.5,2.5,16,(2, 3))"
    assert interpretation.numpy_dtype == interpretation.to_dtype

    assert interpretation == cls(-1.5, 2.5, 16, (2, 3))
    assert interpretation != cls(-1.0, 2.5, 16, (2, 3))
    assert interpretation != cls(-1.5, 2.0, 16, (2, 3))
    assert interpretation != cls(-1.5, 2.5, 15, (2, 3))
    assert interpretation != cls(-1.5, 2.5, 16, (2,))
    assert interpretation != AsDtype(">f4")


def test_truncated_numerical_is_truncated():
    interpretation = AsFloat16(0, 0, 12)

    assert interpretation.is_truncated is True
    assert repr(interpretation) == "AsFloat16(0, 0, 12)"
    assert interpretation.from_dtype.names == ("exponent", "mantissa")
    assert interpretation.itemsize == 3


@pytest.mark.parametrize(
    "cls, primitive, typename",
    [(AsDouble32, "float64", "Double32_t"), (AsFloat16, "float32", "Float16_t")],
)
def test_truncated_numerical_types(cls, primitive, typename):
    assert cls(0, 1, 8).to_dtype == numpy.dtype(primitive)
    assert cls(0, 1, 8).typename == typename
    assert str(cls(0, 1, 8).awkward_form(None).type) == primitive

    interpretation = cls(0, 1, 8, (2, 3))
    assert interpretation.to_dtype == numpy.dtype((primitive, (2, 3)))
    assert interpretation.typename == f"{typename}[2][3]"
    assert str(interpretation.awkward_form(None).type) == f"2 * 3 * {primitive}"


@pytest.mark.parametrize("cls", [AsDouble32, AsFloat16])
@pytest.mark.parametrize("num_bits", [1, 33, 16.5, "16"])
def test_truncated_numerical_num_bits(cls, num_bits):
    with pytest.raises(
        TypeError, match=r"^num_bits must be an integer between 2 and 32 \(inclusive\)$"
    ):
        cls(0, 1, num_bits)


@pytest.mark.parametrize("cls", [AsDouble32, AsFloat16])
def test_truncated_numerical_range(cls):
    with pytest.raises(
        ValueError, match=r"^high \(1\) must be strictly greater than low \(2\)$"
    ):
        cls(2, 1, 12)


def packed_float(exponent, mantissa):
    return struct.pack(">BH", exponent, mantissa)


# 1.5 is 0 01111111 100... in IEEE 754, of which 12 bits of mantissa are kept;
# the next-higher bit of the packed mantissa is the sign
plus_1p5 = packed_float(127, 0b0_1000_0000_0000)
minus_1p5 = packed_float(127, 0b10_1000_0000_0000)
plus_0p75 = packed_float(126, 0b0_1000_0000_0000)


@pytest.mark.parametrize(
    "cls, dtype", [(AsFloat16, numpy.float32), (AsDouble32, numpy.float64)]
)
@pytest.mark.parametrize("name", ["np", "ak"])
def test_truncated_read(raw_branch, cls, dtype, name):
    branch = raw_branch([plus_1p5, minus_1p5], [plus_0p75])

    result = branch.array(interpretation=cls(0, 0, 12), library=name)

    assert result.tolist() == [1.5, -1.5, 0.75]
    if name == "np":
        assert result.dtype == numpy.dtype(dtype)


def test_truncated_read_with_shape(raw_branch):
    branch = raw_branch([plus_1p5 + minus_1p5, plus_0p75 + plus_1p5])

    result = branch.array(interpretation=AsFloat16(0, 0, 12, (2,)), library="np")

    assert result.tolist() == [[1.5, -1.5], [0.75, 1.5]]


@pytest.mark.parametrize(
    "cls, dtype", [(AsFloat16, numpy.float32), (AsDouble32, numpy.float64)]
)
def test_ranged_read(raw_branch, cls, dtype):
    # with 8 bits, the range is divided into 256 parts
    branch = raw_branch([struct.pack(">I", 0), struct.pack(">I", 128)])

    result = branch.array(interpretation=cls(-1, 3, 8), library="np")

    assert result.tolist() == [-1.0, 1.0]
    assert result.dtype == numpy.dtype(dtype)


def test_truncated_read_wrong_number_of_bytes(raw_branch):
    branch = raw_branch([b"\x00" * 4])

    with pytest.raises(ValueError) as excinfo:
        branch.array(interpretation=AsFloat16(0, 0, 12), library="np")

    assert str(excinfo.value) == (
        f"basket 0 in tree/branch {branch.object_path} has the wrong number of bytes "
        "(4) for interpretation AsFloat16(0, 0, 12) (expecting raw array of "
        f"{AsFloat16(0, 0, 12).from_dtype!r})\nin file {branch.file.file_path}"
    )


def test_as_jagged_needs_numerical():
    with pytest.raises(TypeError, match=r"^AsJagged content can only be Numerical$"):
        AsJagged(AsStrings())


def test_as_jagged_properties():
    original = uproot.containers.AsVector(True, ">i4")
    interpretation = AsJagged(AsDtype(">i4"), 10, original=original)

    assert interpretation.header_bytes == 10
    assert interpretation.original is original
    assert interpretation.numpy_dtype == numpy.dtype(object)


@pytest.mark.parametrize(
    "content, expected",
    [
        (AsDtype(">i4"), "int32_t[]"),
        (AsDtype((">i4", (2, 3))), "int32_t[][2][3]"),
    ],
)
def test_as_jagged_typename(content, expected):
    assert AsJagged(content).typename == expected
    assert AsJagged(content, typename="my_type").typename == "my_type"


def test_as_jagged_without_offsets(tree):
    branch = tree["x"]

    with pytest.raises(uproot.deserialization.DeserializationError) as excinfo:
        branch.array(interpretation=AsJagged(AsDtype(">i4")), library="np")

    assert excinfo.value.message == (
        "missing offsets (and missing count branch) for jagged array"
    )
    assert excinfo.value.file_path == branch.file.file_path


@pytest.mark.parametrize(
    "divisor, expected",
    [(1, [8, 16]), (2, [4, 8]), (4, [2, 4]), (8, [1, 2]), (3, [2, 5])],
)
def test_fast_divide(divisor, expected):
    assert fast_divide(numpy.array([8, 16]), divisor).tolist() == expected


def test_jagged_array():
    array = JaggedArray(numpy.array([0, 2, 2, 5]), numpy.array([1, 2, 3, 4, 5]))

    assert repr(array) == "JaggedArray([0 2 2 5], [1 2 3 4 5])"
    assert array[0].tolist() == [1, 2]
    assert array[1].tolist() == []
    assert array[2].tolist() == [3, 4, 5]
    assert [x.tolist() for x in array] == [[1, 2], [], [3, 4, 5]]


def test_jagged_array_parents_localindex():
    array = JaggedArray(numpy.array([0, 2, 2, 5]), numpy.array([1, 2, 3, 4, 5]))

    parents, localindex = array.parents_localindex(10, 13)

    assert parents.tolist() == [10, 10, 12, 12, 12]
    assert localindex.tolist() == [0, 1, 0, 1, 2]


def test_as_strings_length_bytes():
    with pytest.raises(ValueError, match=r"^length_bytes must be '1-5' or '4'$"):
        AsStrings(length_bytes="3")


def test_as_strings_properties():
    original = uproot.containers.AsString(False)
    interpretation = AsStrings(6, "4", original=original)

    assert interpretation.header_bytes == 6
    assert interpretation.length_bytes == "4"
    assert interpretation.original is original


@pytest.mark.parametrize(
    "interpretation, expected",
    [
        (AsStrings(), "AsStrings()"),
        (AsStrings(header_bytes=6), "AsStrings(header_bytes=6)"),
        (AsStrings(length_bytes="4"), "AsStrings(length_bytes='4')"),
        (
            AsStrings(header_bytes=6, length_bytes="4"),
            "AsStrings(header_bytes=6, length_bytes='4')",
        ),
    ],
)
def test_as_strings_repr(interpretation, expected):
    assert repr(interpretation) == expected


def test_as_strings_eq():
    assert AsStrings(6) == AsStrings(6)
    assert AsStrings(6) != AsStrings(0)
    assert AsStrings() != AsStrings(length_bytes="4")
    assert AsStrings() != "AsStrings()"


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


@pytest.mark.parametrize("name", ["np", "ak"])
def test_strings_without_offsets_in_a_file_from_root(name):
    path = skhep_testdata.data_path("uproot-issue413.root")
    with uproot.open(path) as file:
        branch = file["mytree/Str"]

        assert branch.member("fEntryOffsetLen") == 0
        assert branch.array(library=name).tolist() == [
            "evt-0",
            "evt-1",
            "evt-2",
            "evt-3",
            "evt-4",
        ]


def test_truncated_and_ranged_in_a_file_from_root():
    path = skhep_testdata.data_path("uproot-double32-float16.root")
    with uproot.open(path) as file:
        tree = file["tree"]

        expected = tree["double32_32"].array(library="np")
        assert tree["double32_32"].interpretation == AsDouble32(-2.71, 10.0, 32)
        assert tree["float16_16"].interpretation == AsFloat16(-2.71, 10.0, 16)
        assert tree["float16_16"].array(library="np") == pytest.approx(
            expected, abs=(10.0 + 2.71) / 2**16
        )

        assert tree["array_10"].interpretation.to_dims == (3,)
        assert tree["array_10"].array(library="np").shape == (len(expected), 3)


def test_final_array_of_not_only_awkward_baskets(tree):
    branch = tree["x"]

    result = AsStrings().final_array(
        {0: awkward.Array(["ab", "c"]), 1: numpy.array(["d"])},
        1,
        3,
        [0, 2, 3],
        ak,
        branch,
        {},
    )

    assert result.tolist() == ["c", "d"]


def test_string_array():
    array = StringArray(numpy.array([0, 2, 2, 3]), b"abc")

    assert repr(array) == "StringArray([0 2 2 3], b'abc')"
    assert array[0] == "ab"
    assert array[1] == ""
    assert array[2] == "c"
    assert list(array) == ["ab", "", "c"]
    assert len(array) == 3


def test_string_array_repr_of_long_content():
    content = b"a" * 50 + b"b" * 60
    array = StringArray(numpy.array([0, 110]), content)

    assert repr(array) == (
        "StringArray([  0 110], " + repr(b"a" * 45) + " ... " + repr(b"b" * 45) + ")"
    )


def test_as_grouped(tree):
    branch = tree["x"]
    subbranches = {"a": AsDtype(">i4"), "missing": None, "b": AsDtype(">f8")}
    interpretation = AsGrouped(branch, subbranches)

    assert interpretation.branch is branch
    assert interpretation.subbranches is subbranches
    assert repr(interpretation) == f"AsGroup({branch}, {subbranches})"
    assert interpretation.typename == "(group of a:int32_t, b:double)"
    assert AsGrouped(branch, subbranches, typename="Thing").typename == "Thing"

    assert interpretation.cache_key == (
        "AsGrouped(x,['a':AsDtype(Bi4(),Li4()),'b':AsDtype(Bf8(),Lf8())])"
    )

    form = interpretation.awkward_form(None)
    assert form.fields == ["a", "b"]
    assert str(form.type) == "{a: int32, b: float64}"


def test_as_grouped_eq(tree):
    subbranches = {"a": AsDtype(">i4")}

    assert AsGrouped(tree["x"], subbranches) == AsGrouped(tree["x"], dict(subbranches))
    assert AsGrouped(tree["x"], subbranches) != AsGrouped(tree["x"], {})
    assert AsGrouped(tree["x"], subbranches) != subbranches


@pytest.mark.parametrize("method", ["basket_array", "final_array"])
def test_as_grouped_cannot_be_read(tree, method):
    branch = tree["x"]
    interpretation = AsGrouped(branch, {"a": None, "b": None})

    with pytest.raises(ValueError) as excinfo:
        getattr(interpretation, method)(
            *([None] * 7 if method == "final_array" else [None] * 8)
        )

    assert str(excinfo.value) == (
        "grouping branches like 'x' should not be read directly; instead read the "
        f"subbranches:\n\n    'a', 'b'\n\nin file {branch.file.file_path}\n"
        f"in object {branch.object_path}"
    )


def test_custom_interpretation_needs_match_branch():
    with pytest.raises(NotImplementedError):
        CustomInterpretation.match_branch(None, {}, True)


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


def test_custom_final_array_of_unknown_types():
    pytest.importorskip("pandas")
    interpretation = Custom(None, {}, True)

    with pytest.raises(
        TypeError,
        match=r"^Unsupported array type: <class 'list'>\. Expected an Awkward Array, "
        r"NumPy array, or Pandas DataFrame\.$",
    ):
        interpretation.final_array({0: [1, 2]}, 0, 2, [0, 2], np, None, {})


def test_as_binary_eq():
    interpretation = AsBinary()
    assert interpretation == interpretation
    assert interpretation != AsBinary()


@pytest.mark.parametrize("name", ["np", "ak"])
def test_as_binary_of_jagged(tree, name):
    result = tree["y"].array(interpretation=AsBinary(), library=name)

    assert result.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_as_binary_of_jagged_awkward_is_jagged(tmp_path):
    with uproot.recreate(tmp_path / "jagged.root") as file:
        file.mktree("tree", {"y": "var * uint8"})
        file["tree"].extend({"y": awkward.Array([[1, 2, 3], [], [4]])})

    with uproot.open(tmp_path / "jagged.root") as file:
        result = file["tree/y"].array(interpretation=AsBinary(), library="ak")

    assert result.tolist() == [[1, 2, 3], [], [4]]


def test_as_binary_needs_numpy_or_awkward(tree):
    pytest.importorskip("pandas")
    interpretation = AsBinary()

    with pytest.raises(
        ValueError, match=r"^Unsupported library: pd, can only use 'ak' or 'np'\.$"
    ):
        interpretation.basket_array(
            numpy.array([], dtype="u1"), None, None, tree["x"], {}, 0, pd, {}
        )

    with pytest.raises(
        ValueError, match=r"^Unsupported library: pd, can only use 'ak' or 'np'\.$"
    ):
        interpretation.final_array({0: None}, 0, 1, [0, 1], pd, tree["x"], {})
