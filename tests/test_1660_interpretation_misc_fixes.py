# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import ast
import warnings

import numpy
import pytest

import uproot
import uproot.exceptions
import uproot.interpretation.custom
import uproot.interpretation.identify as identify
import uproot.interpretation.library as library
import uproot.interpretation.objects as objects
from uproot.containers import AsBitSet, STLBitSet
from uproot.language.python import (
    _ast_as_branch_expression,
    _walk_ast_yield_symbols,
)


def test_parse_typename_long_long_pointer():
    # "long long*" and "unsigned long long*" used to be dead branches
    # (shadowed by the scalar "long"/"unsigned long" branches) and constructed
    # AsArray with the wrong number of arguments.
    interp = identify.parse_typename("long long*")
    assert isinstance(interp, uproot.containers.AsArray)
    assert interp.values == numpy.dtype(">i8")

    interp = identify.parse_typename("unsigned long long*")
    assert isinstance(interp, uproot.containers.AsArray)
    assert interp.values == numpy.dtype(">u8")

    # the scalar forms still resolve to plain dtypes
    assert identify.parse_typename("long long") == numpy.dtype(">i8")
    assert identify.parse_typename("unsigned long long") == numpy.dtype(">u8")
    assert identify.parse_typename("long") == numpy.dtype(">i8")
    assert identify.parse_typename("unsigned long") == numpy.dtype(">u8")


def test_pandas_object_branch_returns_data_not_nan():
    # When awkward_form raises CannotBeAwkward, _process_array_for_pandas used
    # to fall off the end and return None, producing an all-NaN Series.
    class CannotBeAwkwardInterp:
        def awkward_form(self, file):
            raise objects.CannotBeAwkward("test")

    array = numpy.empty(3, dtype=object)
    array[0] = "a"
    array[1] = "b"
    array[2] = "c"

    out = library._process_array_for_pandas(array, True, CannotBeAwkwardInterp())
    assert list(out) == ["a", "b", "c"]
    pandas = pytest.importorskip("pandas")
    series = pandas.Series(out, index=range(3))
    assert series.tolist() == ["a", "b", "c"]


def test_asbinary_single_entry_basket():
    # A single-entry basket yields a single count; the old uniformity check
    # (numpy.unique(counts[1:] - counts[:-1]).size == 1) gave size 0 and
    # raised AssertionError on valid data.
    interp = uproot.interpretation.custom.AsBinary()

    class FakeNumpyLibrary:
        name = "np"

    data = numpy.arange(4, dtype=numpy.uint8)
    byte_offsets = numpy.array([0, 4], dtype=numpy.int64)
    out = interp.basket_array(
        data, byte_offsets, None, None, {}, 0, FakeNumpyLibrary(), {}
    )
    assert out.tolist() == [[0, 1, 2, 3]]
    # uniform multi-entry still works
    data = numpy.arange(6, dtype=numpy.uint8)
    byte_offsets = numpy.array([0, 3, 6], dtype=numpy.int64)
    out = interp.basket_array(
        data, byte_offsets, None, None, {}, 0, FakeNumpyLibrary(), {}
    )
    assert out.tolist() == [[0, 1, 2], [3, 4, 5]]
    # an arithmetic progression of non-uniform counts must be rejected
    data = numpy.arange(6, dtype=numpy.uint8)
    byte_offsets = numpy.array([0, 1, 3, 6], dtype=numpy.int64)
    with pytest.raises(AssertionError):
        interp.basket_array(
            data, byte_offsets, None, None, {}, 0, FakeNumpyLibrary(), {}
        )


def test_stlbitset_repr_str_len():
    bits = STLBitSet(numpy.array([True, False, True]))
    assert len(bits) == 3
    assert str(bits) == "[True, False, True]"
    assert repr(bits).startswith("<STLBitSet [True, False, True] at 0x")


def test_asbitset_typename_includes_bit_count():
    interp = identify.parse_typename("std::bitset<8>")
    assert isinstance(interp, AsBitSet)
    assert interp.num_bits == 8
    assert interp.typename == "std::bitset<8>"


def test_expression_branch_named_like_a_function():
    # A branch named "sqrt" used in call position must resolve to the function,
    # not to get('sqrt'), and should warn about the conflict.
    keys = {"sqrt", "x"}
    aliases = {}
    functions = {"sqrt": None, "abs": None}
    getter = "get"

    node = ast.parse("sqrt(x)").body[0].value

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _ast_as_branch_expression(node, keys, aliases, functions, getter)

    assert any(
        issubclass(w.category, uproot.exceptions.NameConflictWarning) for w in caught
    )

    # the call resolves to function['sqrt'](get('x'))
    assert isinstance(out, ast.Call)
    assert isinstance(out.func, ast.Subscript)
    assert out.func.value.id == "function"
    assert out.func.slice.value == "sqrt"

    # only "x" is requested as a branch symbol, not "sqrt"
    symbols = list(_walk_ast_yield_symbols(node, keys, aliases, functions, getter))
    assert symbols == ["x"]

    # in value position, a branch named "sqrt" still resolves to the branch
    node2 = ast.parse("sqrt + x").body[0].value
    out2 = _ast_as_branch_expression(node2, keys, aliases, functions, getter)
    assert isinstance(out2.left, ast.Call)
    assert out2.left.func.id == getter
    assert out2.left.args[0].value == "sqrt"


def test_strided_objects_shared_interpretation_no_state_leak():
    # AsStridedObjects.basket_array used to mutate self._to_dtype; reading the
    # same branch repeatedly (sharing the interpretation) must be stable.
    skhep_testdata = pytest.importorskip("skhep_testdata")
    f = uproot.open(skhep_testdata.data_path("uproot-issue-513.root"))
    branch = f["TrkAnaNeg/trkana"]["demcent"]["_mom"]
    interp = branch.interpretation
    to_dtype_before = interp.to_dtype

    first = branch.array(library="np")
    assert interp.to_dtype == to_dtype_before

    second = branch.array(library="np")
    assert len(first) == len(second) == 101
    assert interp.to_dtype == to_dtype_before
    for a, b in zip(first, second, strict=True):
        assert a.member("fCoordinates").all_members == pytest.approx(
            b.member("fCoordinates").all_members
        )
