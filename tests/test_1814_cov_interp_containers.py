# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Unit tests of uproot.containers: the AsContainer descriptions (their typename,
cache_key, repr, equality, hash and awkward_form) and the Python containers
(STLVector, STLMap, ...) that the deserialization produces.
"""

from __future__ import annotations

import types

import numpy
import pytest

import awkward
import uproot
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
    STLBitSet,
    STLList,
    STLMap,
    STLSet,
    STLVector,
)
from uproot.interpretation.objects import CannotBeAwkward, CannotBeStrided
from uproot.models.TObject import Model_TObject

i4 = numpy.dtype(">i4")
f8 = numpy.dtype(">f8")

# context handed to awkward_form by Interpretation.awkward_form
form_context = {
    "index_format": "i64",
    "header": False,
    "tobject_header": False,
    "breadcrumbs": (),
}

no_file = None


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


def test_header_setter():
    container = AsVector(False, i4)
    container.header = True
    assert container.header is True

    with pytest.raises(TypeError, match=r"^AsVector\.header must be True or False$"):
        container.header = 1


def test_string_length_bytes():
    assert AsString(False, length_bytes="4").length_bytes == "4"
    with pytest.raises(ValueError, match=r"^length_bytes must be '1-5' or '4'$"):
        AsString(False, length_bytes="2")


def test_properties():
    assert AsDynamic(Model_TObject).model is Model_TObject
    assert AsPointer(Model_TObject).pointee is Model_TObject

    array = AsArray(True, False, AsString(False), (2,))
    assert array.speedbump is False
    assert array.values == AsString(False)
    assert array.inner_shape == (2,)

    assert AsBitSet(True, 8).num_bits == 8
    assert AsBitSet(True, numpy.dtype("?")).num_bits is None
    assert AsBitSet(True, 8).keys == numpy.dtype("?")
    assert AsSet(True, i4).keys == i4
    assert AsRVec(True, i4).values == i4
    assert AsList(True, i4).values == i4

    map_ = AsMap(True, i4, Model_TObject)
    assert map_.keys == i4
    assert map_.values is Model_TObject
    assert AsMap(True, Model_TObject, i4).keys is Model_TObject


def test_array_rejects_dtype_with_inner_shape():
    with pytest.raises(
        RuntimeError, match=r"^Logic error: the dtype should have contained the shape$"
    ):
        AsArray(False, False, i4, (2,))


@pytest.mark.parametrize(
    "container, expected",
    [
        (
            AsString(False),
            '{"class": "ListOffsetArray", "offsets": "i64", "content": '
            '{"class": "NumpyArray", "primitive": "uint8", "parameters": '
            '{"__array__": "char"}}, "parameters": {"__array__": "string"}}',
        ),
        (
            AsVector(False, i4),
            '{"class": "ListOffsetArray", "offsets": "i64", "content": "int32"}',
        ),
        (
            AsSet(False, i4),
            '{"class": "ListOffsetArray", "offsets": "i64", "content": "int32", '
            '"parameters": {"__array__": "set"}}',
        ),
        (
            AsBitSet(False, 8),
            '{"class": "ListOffsetArray", "offsets": "i64", "content": "bool", '
            '"parameters": {"__array__": "bitset"}}',
        ),
        (
            AsArray(False, False, AsVector(False, i4), (2, 3)),
            '{"class": "ListOffsetArray", "offsets": "i64", "content": '
            '{"class": "RegularArray", "size": 2, "content": '
            '{"class": "RegularArray", "size": 3, "content": '
            '{"class": "ListOffsetArray", "offsets": "i64", "content": "int32"}}}}',
        ),
    ],
)
def test_awkward_form(container, expected):
    expected_form = awkward.forms.from_json(expected)
    assert container.awkward_form(no_file, dict(form_context)) == expected_form


def test_awkward_form_of_map():
    form = AsMap(False, i4, f8).awkward_form(no_file, dict(form_context))
    assert form.parameters == {}
    assert form.content.parameters == {"__array__": "sorted_map"}
    assert form.content.is_tuple
    assert [x.primitive for x in form.content.contents] == ["int32", "float64"]


def test_awkward_form_of_dynamic_with_model():
    form = AsDynamic(AsVector(False, i4)).awkward_form(no_file, dict(form_context))
    assert form == AsVector(False, i4).awkward_form(no_file, dict(form_context))


@pytest.mark.parametrize(
    "container, because",
    [
        (AsDynamic(), "dynamic type"),
        (AsFIXME("oh no"), "oh no"),
        (AsPointer(Model_TObject), "arbitrary pointer"),
    ],
)
def test_cannot_be_awkward(container, because):
    with pytest.raises(CannotBeAwkward) as excinfo:
        container.awkward_form(no_file, dict(form_context))
    assert excinfo.value.because == because


def test_cannot_be_strided():
    with pytest.raises(CannotBeStrided, match=r"^std::vector<int32_t>$"):
        AsVector(False, i4).strided_interpretation(no_file)


def test_fixme_read():
    file = types.SimpleNamespace(file_path="some/file.root")
    with pytest.raises(uproot.deserialization.DeserializationError) as excinfo:
        AsFIXME("this type").read(None, None, {}, file, None, None)
    assert excinfo.value.message == "this type; please file a bug report!"
    assert str(excinfo.value) == (
        "this type; please file a bug report!\nin file some/file.root"
    )


@pytest.mark.parametrize(
    "length, limit, expected",
    [
        (0, 85, "[]"),
        (1, 85, "[0]"),
        (2, 85, "[0, 1]"),
        (3, 85, "[0, 1, 2]"),
        (10, 85, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"),
        # just enough room to not need an ellipsis, which is reserved until the end
        (4, 14, "[0, 1, 2, 3]"),
        (4, 13, "[0, ..., 3]"),
        # alternates between the left and the right side
        (10, 20, "[0, 1, 2, ..., 8, 9]"),
        (10, 17, "[0, 1, ..., 8, 9]"),
        # the left side is the one that fits
        (10, 12, "[0, ..., 9]"),
        (10, 9, "[0, ...]"),
        # not even one item fits
        (10, 2, "[0, ...]"),
    ],
)
def test_str_with_ellipsis(length, limit, expected):
    assert (
        uproot.containers._str_with_ellipsis(str, length, "[", "]", limit) == expected
    )


def test_str_with_ellipsis_skips_left_if_only_right_breaks():
    # the first item (left) fits, and the second (right) is too long for the limit
    def tostring(i):
        return "x" if i == 0 else "y" * 50

    text = uproot.containers._str_with_ellipsis(tostring, 5, "[", "]", 12)
    assert text == "[x, ...]"


def _generator(values):
    yield from values


@pytest.mark.parametrize("cls", [ROOTRVec, STLList, STLVector])
@pytest.mark.parametrize(
    "make",
    [
        lambda: [1, 2, 3],
        lambda: (1, 2, 3),
        lambda: {1, 2, 3},
        lambda: frozenset([1, 2, 3]),
        lambda: _generator([1, 2, 3]),
        lambda: numpy.array([1, 2, 3]),
    ],
    ids=["list", "tuple", "set", "frozenset", "generator", "ndarray"],
)
def test_sequences_constructors(cls, make):
    container = cls(make())
    assert isinstance(container._values, numpy.ndarray)
    assert sorted(container.tolist()) == [1, 2, 3]
    assert len(container) == 3


@pytest.mark.parametrize("cls", [ROOTRVec, STLList, STLVector])
def test_sequences_interface(cls):
    container = cls([1, 5, 3])

    assert str(container) == "[1, 5, 3]"
    assert repr(container).startswith(f"<{cls.__name__} [1, 5, 3] at 0x")
    assert repr(container).endswith(">")

    assert container[0] == 1
    assert container[-1] == 3
    assert container[1:].tolist() == [5, 3]
    assert 5 in container
    assert 4 not in container
    assert list(iter(container)) == [1, 5, 3]

    backward = reversed(container)
    assert isinstance(backward, cls)
    assert backward.tolist() == [3, 5, 1]


@pytest.mark.parametrize("cls", [ROOTRVec, STLList, STLVector])
def test_sequences_eq_is_elementwise(cls):
    container = cls([1, 5, 3])

    assert (container == cls([1, 0, 3])).tolist() == [True, False, True]
    assert (container == [1, 5, 0]).tolist() == [True, True, False]
    assert (container == "abc") is not True
    assert (container == 123) is False
    assert (container == {"a": 1}) is False


def test_sequences_of_strings_quote_their_items():
    assert str(STLVector(["a", "b"])) == "['a', 'b']"
    assert str(STLList(["a", "b"])) == "['a', 'b']"
    assert str(ROOTRVec(["a", "b"])) == "['a', 'b']"
    assert str(STLVector(numpy.array(["a"], dtype=object))) == "['a']"


@pytest.mark.parametrize("cls", [ROOTRVec, STLVector])
def test_sequences_array(cls):
    assert numpy.asarray(cls([1, 2])).tolist() == [1, 2]
    assert numpy.asarray(cls([1, 2]), dtype="f8").dtype == numpy.dtype("f8")


@pytest.mark.parametrize("cls", [ROOTRVec, STLList, STLVector])
def test_sequences_tolist_is_recursive(cls):
    inner = numpy.empty(3, dtype=object)
    inner[0] = STLVector([1, 2])
    inner[1] = numpy.array([3.5])
    inner[2] = "plain"
    assert cls(inner).tolist() == [[1, 2], [3.5], "plain"]


def test_sequences_stay_as_they_are_if_already_array_like():
    values = numpy.array([1, 2])
    assert STLVector(values)._values is values
    assert STLList(values)._values is values
    assert ROOTRVec(values)._values is values


def test_bitset():
    bitset = STLBitSet([1, 0, 1])

    assert str(bitset) == "[1, 0, 1]"
    assert repr(bitset).startswith("<STLBitSet [1, 0, 1] at 0x")
    assert len(bitset) == 3
    assert bitset[0] == 1
    assert bitset[1:].tolist() == [0, 1]
    assert list(bitset) == [1, 0, 1]
    assert bitset.tolist() == [1, 0, 1]

    assert (bitset == STLBitSet([1, 1, 1])).tolist() == [True, False, True]
    assert (bitset == [0, 0, 1]).tolist() == [False, True, True]
    assert (bitset == 5) is False


@pytest.mark.parametrize(
    "make",
    [
        lambda: [3, 1, 2],
        lambda: _generator([3, 1, 2]),
        lambda: {3, 1, 2},
        lambda: numpy.array([3, 1, 2]),
    ],
    ids=["list", "generator", "set", "ndarray"],
)
def test_set_constructors_sort(make):
    container = STLSet(make())
    assert container._keys.tolist() == [1, 2, 3]
    assert list(container) == [1, 2, 3]


def test_set_interface():
    container = STLSet([3, 1, 2])

    assert str(container) == "{1, 2, 3}"
    assert repr(container).startswith("<STLSet {1, 2, 3} at 0x")
    assert len(container) == 3
    assert numpy.asarray(container).tolist() == [1, 2, 3]
    assert container.tolist() == {1, 2, 3}


def test_set_of_strings():
    container = STLSet(["b", "a"])

    assert str(container) == "{'a', 'b'}"
    assert "a" in container
    assert "c" not in container


@pytest.mark.parametrize(
    "where, expected",
    [(1, True), (3, True), (0, False), (2, False), (4, False), (2.0, False)],
)
def test_set_contains(where, expected):
    assert (where in STLSet([1, 3])) is expected


def test_set_contains_array_is_false():
    assert ([1, 3] in STLSet([1, 3])) is False


@pytest.mark.parametrize(
    "other, expected",
    [
        (STLSet([1, 2]), True),
        (STLSet([1, 3]), False),
        (STLSet([1, 2, 3]), False),
        ({1, 2}, True),
        ({2, 3}, False),
        ({1, 2, 3}, False),
        (frozenset([2, 1]), True),
        ([1, 2], False),
        (5, False),
    ],
)
def test_set_eq(other, expected):
    assert bool(STLSet([2, 1]) == other) is expected
    assert bool(STLSet([2, 1]) != other) is (not expected)


def test_map_constructor_mismatched_lengths():
    with pytest.raises(
        ValueError, match=r"^number of keys must be equal to the number of values$"
    ):
        STLMap([1, 2], [1])


@pytest.mark.parametrize(
    "keys",
    [
        lambda: [2, 1],
        lambda: (2, 1),
        lambda: {2: None, 1: None}.keys(),
        lambda: _generator([2, 1]),
        lambda: {2, 1},
        lambda: numpy.array([2, 1]),
    ],
    ids=["list", "tuple", "keysview", "generator", "set", "ndarray"],
)
@pytest.mark.parametrize(
    "values",
    [
        lambda: [20, 10],
        lambda: {"a": 20, "b": 10}.values(),
        lambda: _generator([20, 10]),
        lambda: numpy.array([20, 10]),
    ],
    ids=["list", "valuesview", "generator", "ndarray"],
)
def test_map_constructors_sort_by_key(keys, values):
    made_keys = keys()
    made_values = values()
    container = STLMap(made_keys, made_values)
    if isinstance(made_keys, set):
        # no defined order for the set; only its sortedness is meaningful
        assert container.keys().tolist() == [1, 2]
        assert sorted(container.values().tolist()) == [10, 20]
    else:
        assert container.tolist() == {1: 10, 2: 20}


def test_map_from_mapping():
    container = STLMap.from_mapping({2: 20.5, 1: 10.5})
    assert container.tolist() == {1: 10.5, 2: 20.5}


def test_map_interface():
    container = STLMap([2, 1], [20, 10])

    assert str(container) == "{1: 10, 2: 20}"
    assert repr(container).startswith("<STLMap {1: 10, 2: 20} at 0x")
    assert len(container) == 2
    assert list(iter(container)) == [1, 2]
    assert container.keys().tolist() == [1, 2]
    assert container.values().tolist() == [10, 20]
    assert container.items().tolist() == [[1, 10], [2, 20]]


def test_map_of_strings():
    container = STLMap(["b", "a"], numpy.array(["x", "y"], dtype=object))

    assert str(container) == "{'a': 'y', 'b': 'x'}"
    assert container["a"] == "y"
    assert "b" in container


def test_map_getitem_scalar():
    container = STLMap([2, 1], [20, 10])

    assert container[1] == 10
    assert container[2] == 20
    with pytest.raises(KeyError):
        container[0]
    with pytest.raises(KeyError):
        container[3]


def test_map_getitem_array():
    container = STLMap([2, 1], [20, 10])

    selected = container[numpy.array([1, 3, 0, 2])]
    assert isinstance(selected, numpy.ma.MaskedArray)
    assert selected.tolist() == [10, None, None, 20]


def test_map_getitem_array_of_empty_map():
    container = STLMap(numpy.array([], dtype="i8"), numpy.array([], dtype="f8"))

    selected = container[numpy.array([1, 2])]
    assert selected.tolist() == [None, None]


def test_map_get_scalar():
    container = STLMap([2, 1], [20, 10])

    assert container.get(1) == 10
    assert container.get(2, 99) == 20
    assert container.get(0) is None
    assert container.get(3, 99) == 99


def test_map_get_array():
    container = STLMap([2, 1], [20, 10])

    assert container.get(numpy.array([1, 3, 0, 2]), -1).tolist() == [10, -1, -1, 20]


def test_map_get_array_of_empty_map():
    container = STLMap(numpy.array([], dtype="i8"), numpy.array([], dtype="f8"))

    assert container.get(numpy.array([1, 2]), -1).tolist() == [-1, -1]


@pytest.mark.parametrize(
    "where, expected", [(1, True), (2, True), (0, False), (3, False), (1.5, False)]
)
def test_map_contains(where, expected):
    assert (where in STLMap([2, 1], [20, 10])) is expected


def test_map_contains_array_is_false():
    assert ([1, 2] in STLMap([2, 1], [20, 10])) is False


@pytest.mark.parametrize(
    "other, expected",
    [
        (STLMap([1, 2], [10, 20]), True),
        (STLMap([1, 2], [10, 21]), False),
        (STLMap([1, 3], [10, 20]), False),
        (STLMap([1], [10]), False),
        ({1: 10, 2: 20}, True),
        ({1: 10, 2: 21}, False),
        ({1: 10}, False),
        ([(1, 10), (2, 20)], False),
        (5, False),
    ],
)
def test_map_eq(other, expected):
    assert bool(STLMap([2, 1], [20, 10]) == other) is expected
    assert bool(STLMap([2, 1], [20, 10]) != other) is (not expected)


def test_map_tolist_is_recursive():
    values = numpy.empty(2, dtype=object)
    values[0] = STLVector([1, 2])
    values[1] = "plain"
    assert STLMap([1, 0], values).tolist() == {0: "plain", 1: [1, 2]}


def test_set_tolist_of_plain_values():
    assert STLSet(numpy.array([1.5, 2.5])).tolist() == {1.5, 2.5}
