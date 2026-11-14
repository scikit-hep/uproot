# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Tests of uproot.interpretation.library: finalizing, grouping and concatenating
arrays for the NumPy, Awkward Array, and Pandas libraries.
"""

from __future__ import annotations

import types

import awkward
import numpy
import pytest

import uproot
from uproot.interpretation import library
from uproot.interpretation.numerical import AsDtype
from uproot.interpretation.objects import AsStridedObjects
from uproot.models.TObject import Model_TObject

np = library._libraries["np"]
ak = library._libraries["ak"]
pd = library._libraries["pd"]


@pytest.fixture
def tree(tmp_path):
    path = tmp_path / "library.root"
    with uproot.recreate(path) as file:
        file.mktree("tree", {"x": "int32", "y": "float64", "z": "var * int32"})
        file["tree"].extend(
            {"x": [1, 2], "y": [1.5, 2.5], "z": awkward.Array([[1], []])}
        )
        file["tree"].extend({"x": [3], "y": [3.5], "z": awkward.Array([[2, 3]])})

    with uproot.open(path) as file:
        yield file["tree"]


def test_names_and_equality():
    assert repr(np) == "'np'"
    assert repr(ak) == "'ak'"
    assert repr(pd) == "'pd'"

    assert np == library.NumPy()
    assert ak == library.Awkward()
    assert pd == library.Pandas()
    assert np != ak
    assert np != "np"
    # comparing with anything else doesn't raise
    assert not (np == object())
    assert not (np == "np")


def test_imported():
    assert np.imported is numpy
    assert ak.imported is awkward
    pandas = pytest.importorskip("pandas")
    assert pd.imported is pandas


def test_empty_and_zeros():
    assert np.empty((0,), "i4").tolist() == []
    assert ak.zeros((2,), "f8").tolist() == [0.0, 0.0]


@pytest.mark.parametrize("how", [tuple, list, dict, None])
def test_arrays_how_numpy(tree, how):
    arrays = tree.arrays(["x", "z"], how=how, library="np")
    x, z = (arrays["x"], arrays["z"]) if how is dict or how is None else tuple(arrays)

    assert type(arrays) is (dict if how is None else how)
    assert isinstance(x, numpy.ndarray)
    assert x.tolist() == [1, 2, 3]
    assert z.dtype == numpy.dtype(object)
    assert [item.tolist() for item in z] == [[1], [], [2, 3]]


@pytest.mark.parametrize("how", [tuple, list, dict])
def test_arrays_how_awkward(tree, how):
    arrays = tree.arrays(["x", "z"], how=how, library="ak")
    x, z = (arrays["x"], arrays["z"]) if how is dict else tuple(arrays)

    assert type(arrays) is how
    assert isinstance(x, awkward.Array)
    assert x.tolist() == [1, 2, 3]
    assert z.tolist() == [[1], [], [2, 3]]


def test_arrays_how_none_awkward(tree):
    arrays = tree.arrays(["x", "z"], library="ak")

    assert isinstance(arrays, awkward.Array)
    assert arrays.fields == ["x", "z"]
    assert arrays.tolist() == [
        {"x": 1, "z": [1]},
        {"x": 2, "z": []},
        {"x": 3, "z": [2, 3]},
    ]


def test_arrays_of_nothing_awkward(tree):
    arrays = tree.arrays([], library="ak")
    assert arrays.fields == []
    assert len(arrays) == 0


@pytest.mark.parametrize("how", [tuple, list, dict])
def test_arrays_how_pandas(tree, how):
    pandas = pytest.importorskip("pandas")

    arrays = tree.arrays(["x", "y"], how=how, library="pd")
    x, y = (arrays["x"], arrays["y"]) if how is dict else tuple(arrays)

    assert type(arrays) is how
    assert isinstance(x, pandas.Series)
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [1.5, 2.5, 3.5]


def test_arrays_dataframe(tree):
    pandas = pytest.importorskip("pandas")

    frame = tree.arrays(["x", "y"], library="pd")
    assert isinstance(frame, pandas.DataFrame)
    assert frame.columns.tolist() == ["x", "y"]
    assert frame["y"].tolist() == [1.5, 2.5, 3.5]

    assert tree.arrays([], library="pd").empty


@pytest.mark.parametrize("how", [None, "outer", "inner", "left", "anything"])
def test_pandas_group_with_str_how(how):
    pandas = pytest.importorskip("pandas")

    arrays = {
        "x": pandas.Series([1, 2, 3], index=[0, 1, 2]),
        "y": pandas.Series([1.5, 2.5], index=[1, 2]),
    }
    out = pd.group(arrays, [("x", None), ("y", None)], how)

    # Whatever the str, the arrays are concatenated along columns with an outer
    # join on their indexes (pandas.concat), not merged.
    assert isinstance(out, pandas.DataFrame)
    assert out.columns.tolist() == ["x", "y"]
    assert out.index.tolist() == [0, 1, 2]
    assert out["x"].tolist() == [1, 2, 3]
    assert out["y"].isna().tolist() == [True, False, False]
    assert out["y"].tolist()[1:] == [1.5, 2.5]


def test_group_renames():
    arrays = {"x": numpy.array([1])}
    context = [("x", {"rename": "renamed"})]

    assert list(np.group(arrays, context, dict)) == ["renamed"]
    assert list(np.group(arrays, context, None)) == ["renamed"]
    assert list(ak.group(arrays, context, dict)) == ["renamed"]
    assert ak.group(arrays, context, None).fields == ["renamed"]


def zipping(**arrays):
    # a name that starts with "flat" is not jagged
    context = [(name, {"is_jagged": not name.startswith("flat")}) for name in arrays]
    return ak.group(
        {name: awkward.Array(x) for name, x in arrays.items()}, context, "zip"
    )


def test_zip_nonjagged():
    out = zipping(flat_x=[1, 2], flat_y=[3, 4])
    assert out.tolist() == [{"flat_x": 1, "flat_y": 3}, {"flat_x": 2, "flat_y": 4}]


def test_zip_with_a_common_prefix():
    out = zipping(Muon_pt=[[1], [2, 3]], Muon_eta=[[4], [5, 6]])
    assert out.fields == ["Muon"]
    assert out.Muon.fields == ["pt", "eta"]
    assert out.Muon.pt.tolist() == [[1], [2, 3]]
    assert awkward.num(out.Muon).tolist() == [1, 2]


def test_zip_without_a_common_prefix():
    out = zipping(a=[[1], [2, 3]], b=[[4], [5, 6]])
    assert out.fields == ["jagged0"]
    assert out.jagged0.fields == ["a", "b"]
    assert out.jagged0.b.tolist() == [[4], [5, 6]]


def test_zip_with_only_a_partly_common_prefix():
    out = zipping(Muon_pt=[[1], [2, 3]], Other_eta=[[4], [5, 6]])
    assert out.fields == ["jagged0"]


def test_zip_different_counts():
    out = zipping(
        flat=[7, 8],
        Muon_pt=[[1], [2, 3]],
        Jet_pt=[[1, 2], []],
        Jet_eta=[[3, 4], []],
        Muon_eta=[[4], [5, 6]],
    )
    assert out.fields == ["flat", "Muon", "Jet"]
    assert out.flat.tolist() == [7, 8]
    assert out.Muon.eta.tolist() == [[4], [5, 6]]
    assert out.Jet.eta.tolist() == [[3, 4], []]


def test_zip_prefix_is_already_a_field():
    context = [
        ("Muon", {"is_jagged": False}),
        ("Muon_pt", {"is_jagged": True}),
        ("Muon_eta", {"is_jagged": True}),
    ]
    arrays = {
        "Muon": awkward.Array([7, 8]),
        "Muon_pt": awkward.Array([[1], [2, 3]]),
        "Muon_eta": awkward.Array([[4], [5, 6]]),
    }
    out = ak.group(arrays, context, "zip")

    assert out.fields == ["Muon", "jagged0"]
    assert out.jagged0.fields == ["Muon_pt", "Muon_eta"]


def test_zip_with_rename():
    context = [
        ("a", {"is_jagged": True, "rename": "Muon_pt"}),
        ("b", {"is_jagged": True, "rename": "Muon_eta"}),
    ]
    arrays = {"a": awkward.Array([[1], []]), "b": awkward.Array([[2], []])}
    out = ak.group(arrays, context, "zip")

    assert out.fields == ["Muon"]
    assert out.Muon.fields == ["pt", "eta"]


def test_zip_layouts_that_are_not_zero_based():
    content = awkward.contents.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6]))
    list_array = awkward.contents.ListArray(
        awkward.index.Index64(numpy.array([0, 1])),
        awkward.index.Index64(numpy.array([1, 3])),
        content,
    )
    sliced = awkward.Array([[0], [1], [2, 3]])[1:]
    assert sliced.layout.offsets[0] != 0

    out = zipping(Muon_pt=awkward.Array(list_array), Muon_eta=sliced)

    assert out.fields == ["Muon"]
    assert out.Muon.pt.tolist() == [[1], [2, 3]]
    assert out.Muon.eta.tolist() == [[1], [2, 3]]


def test_zip_end_to_end(tree):
    arrays = tree.arrays(["x", "z"], how="zip")
    assert arrays.x.tolist() == [1, 2, 3]
    assert arrays.jagged0.z.tolist() == [[1], [], [2, 3]]


def test_global_index_does_nothing_for_unindexed_libraries():
    array = numpy.array([1, 2])
    assert np.global_index(array, 10) is array
    assert ak.global_index(array, 10) is array


def test_global_index_pandas():
    pandas = pytest.importorskip("pandas")

    def series():
        return pandas.Series([1, 2])

    assert pd.global_index(series(), 10).index.tolist() == [10, 11]

    assert isinstance(pd.global_index(series(), 1).index, pandas.RangeIndex)

    out = pd.global_index((series(), series()), 1)
    assert isinstance(out, tuple)
    assert out[1].index.tolist() == [1, 2]

    out = pd.global_index([series()], 1)
    assert isinstance(out, list)
    assert out[0].index.tolist() == [1, 2]

    out = pd.global_index({"x": series()}, 1)
    assert isinstance(out, dict)
    assert out["x"].index.tolist() == [1, 2]


@pytest.mark.parametrize(
    "name, make",
    [
        ("np", numpy.array),
        ("ak", awkward.Array),
        ("pd", None),
    ],
)
@pytest.mark.parametrize("how", [tuple, list, dict])
def test_concatenate_groups(name, make, how):
    if name == "pd":
        make = pytest.importorskip("pandas").Series
    lib = library._libraries[name]

    def group(*arrays):
        if how is dict:
            return {str(i): make(x) for i, x in enumerate(arrays)}
        else:
            return how(make(x) for x in arrays)

    out = lib.concatenate([group([1], [2.5]), group([3, 4], [5.5, 6.5])])

    assert type(out) is how
    first, second = out.values() if how is dict else out
    assert list(first) == [1, 3, 4]
    assert list(second) == [2.5, 5.5, 6.5]


@pytest.mark.parametrize("name", ["np", "ak", "pd"])
def test_concatenate_nothing(name):
    if name == "pd":
        pytest.importorskip("pandas")

    nothing = []
    assert library._libraries[name].concatenate(nothing) is nothing


def test_concatenate_single_arrays():
    assert ak.concatenate([awkward.Array([1]), awkward.Array([2, 3])]).tolist() == [
        1,
        2,
        3,
    ]

    pandas = pytest.importorskip("pandas")
    out = pd.concatenate([pandas.Series([1]), pandas.Series([2, 3])])
    assert isinstance(out, pandas.Series)
    assert out.tolist() == [1, 2, 3]


def test_awkward_finalize_structured():
    array = numpy.array(
        [[(1, 1.5), (2, 2.5)], [(3, 3.5), (4, 4.5)], [(5, 5.5), (6, 6.5)]],
        dtype=[("a", "i4"), ("b", "f8")],
    )
    out = ak.finalize(array, None, None, 0, 3, {})

    assert str(out.type) == "3 * 2 * {a: int32, b: float64}"
    assert out.a.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_awkward_finalize_structured_without_fields():
    array = numpy.zeros(3, dtype=[])
    out = ak.finalize(array, None, None, 0, 3, {})

    assert len(out) == 3
    assert out.fields == []


@pytest.mark.parametrize(
    "form, obj, expected",
    [
        ({"class": "NumpyArray"}, 5, 5),
        (
            {"class": "ListOffsetArray", "parameters": {"__array__": "string"}},
            "hello",
            "hello",
        ),
        (
            {"class": "ListOffsetArray64", "content": {"class": "NumpyArray"}},
            [1, 2],
            [1, 2],
        ),
        (
            {"class": "RegularArray", "content": {"class": "NumpyArray"}},
            [1, 2],
            [1, 2],
        ),
        (
            {
                "class": "ListOffsetArray",
                "content": {
                    "parameters": {"__array__": "sorted_map"},
                    "contents": [{"class": "NumpyArray"}, {"class": "NumpyArray"}],
                },
            },
            {1: 10, 2: 20},
            [(1, 10), (2, 20)],
        ),
    ],
)
def test_object_to_awkward_json(form, obj, expected):
    assert library._object_to_awkward_json(form, obj) == expected


def test_object_to_awkward_json_records():
    form = {
        "class": "RecordArray",
        "fields": ["@header", "x", "y"],
        "contents": [
            {"class": "NumpyArray"},
            {"class": "NumpyArray"},
            {"class": "NumpyArray"},
        ],
    }

    class WithMembers:
        y = 2

        def has_member(self, name):
            return name == "x"

        def member(self, name):
            return 1

    obj = types.SimpleNamespace(x=1, y=2)
    assert library._object_to_awkward_json(form, obj) == {"x": 1, "y": 2}
    assert library._object_to_awkward_json(form, WithMembers()) == {"x": 1, "y": 2}


numbers = {"class": "NumpyArray", "primitive": "int32", "form_key": "node1"}
record = {
    "class": "RecordArray",
    "fields": ["@skip", "x"],
    "contents": [numbers, numbers],
    "parameters": {"__record__": "Thing"},
}
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


def list_of(content, offsets="i64"):
    return {"class": "ListOffsetArray", "offsets": offsets, "content": content}


def map_of(key, value, offsets="i64"):
    return {
        "class": "ListOffsetArray",
        "offsets": offsets,
        "content": {
            "class": "RecordArray",
            "fields": None,
            "contents": [key, value],
            "parameters": {"__array__": "sorted_map"},
        },
    }


regular = {"class": "RegularArray", "size": 2, "content": numbers}


@pytest.mark.parametrize(
    "form, objects, expected, type_",
    [
        (numbers, [1, 2], [1, 2], "2 * int32"),
        (numbers, [], [], "0 * int32"),
        (strings, ["a", "bc"], ["a", "bc"], "2 * string"),
        (strings, [], [], "0 * string"),
        (list_of(numbers, "i32"), [[1], [2, 3]], [[1], [2, 3]], "2 * var * int32"),
        (list_of(numbers, "u32"), [[1], [2, 3]], [[1], [2, 3]], "2 * var * int32"),
        (list_of(numbers), [[1], [2, 3]], [[1], [2, 3]], "2 * var * int32"),
        (list_of(numbers, "i32"), [], [], "0 * var * int32"),
        (list_of(numbers, "u32"), [], [], "0 * var * int32"),
        (list_of(numbers), [], [], "0 * var * int32"),
        (
            map_of(numbers, strings),
            [{1: "a", 2: "bc"}, {}],
            [[(1, "a"), (2, "bc")], []],
            '2 * var * tuple[[int32, string], parameters={"__array__": "sorted_map"}]',
        ),
        (
            map_of(numbers, numbers),
            [],
            [],
            '0 * var * tuple[[int32, int32], parameters={"__array__": "sorted_map"}]',
        ),
        (regular, [[1, 2], [3, 4]], [[1, 2], [3, 4]], "2 * 2 * int32"),
        (regular, [], [], "0 * 2 * int32"),
    ],
    ids=str,
)
def test_object_to_awkward_array(form, objects, expected, type_):
    out = library._object_to_awkward_array(awkward, form, objects)

    assert isinstance(out, awkward.Array)
    assert str(out.type) == type_
    assert out.tolist() == expected


def test_object_to_awkward_array_of_records():
    objects = [types.SimpleNamespace(x=1), types.SimpleNamespace(x=2)]
    out = library._object_to_awkward_array(awkward, record, objects)

    assert out.fields == ["x"]
    assert out.tolist() == [{"x": 1}, {"x": 2}]
    assert out.layout.parameter("__record__") == "Thing"


def test_object_to_awkward_array_of_empty_records():
    form = {"class": "RecordArray", "fields": ["@skip"], "contents": [numbers]}
    objects = [types.SimpleNamespace(), types.SimpleNamespace()]
    out = library._object_to_awkward_array(awkward, form, objects)

    assert len(out) == 2
    assert out.fields == []


class OldIndex:
    pass


def test_is_pandas_rangeindex_with_pandas():
    pandas = pytest.importorskip("pandas")

    assert library._is_pandas_rangeindex(pandas, pandas.RangeIndex(0, 3))


@pytest.mark.parametrize(
    "module, index, expected",
    [
        (
            types.SimpleNamespace(__version__="3.0.0"),
            types.SimpleNamespace(is_integer=lambda: True),
            True,
        ),
        (
            types.SimpleNamespace(__version__="3.0.0"),
            types.SimpleNamespace(is_integer=lambda: False),
            False,
        ),
        (types.SimpleNamespace(__version__="3.0.0"), object(), False),
        (
            types.SimpleNamespace(__version__="1.3.0", Int64Index=OldIndex),
            OldIndex(),
            True,
        ),
        (
            types.SimpleNamespace(__version__="1.3.0", Int64Index=OldIndex),
            object(),
            False,
        ),
    ],
)
def test_is_pandas_rangeindex_in_all_versions(module, index, expected):
    assert library._is_pandas_rangeindex(module, index) is expected


def test_strided_to_pandas():
    inner = AsStridedObjects(Model_TObject, [("b", numpy.dtype(">f8"))])
    outer = AsStridedObjects(
        Model_TObject,
        [("@skipped", numpy.dtype(">i4")), ("a", numpy.dtype(">i4")), ("in", inner)],
    )
    data = {"a": [1, 2], "in/b": [1.5, 2.5]}
    arrays, columns = [], []

    library._strided_to_pandas((), outer, data, arrays, columns)

    assert columns == [("a",), ("in", "b")]
    assert arrays == [[1, 2], [1.5, 2.5]]


def test_pandas_basic_index_old_pandas():
    module = types.SimpleNamespace(Int64Index=list)
    assert library._pandas_basic_index(module, 2, 5) == [2, 3, 4]


def test_pandas_basic_index():
    pandas = pytest.importorskip("pandas")

    index = library._pandas_basic_index(pandas, 2, 5)
    assert isinstance(index, pandas.RangeIndex)
    assert index.tolist() == [2, 3, 4]


def test_process_array_for_pandas_not_finalized():
    out = library._process_array_for_pandas(numpy.array([1, 2]), False, AsDtype(">i4"))
    assert isinstance(out, awkward.Array)
    assert out.tolist() == [1, 2]


def test_process_array_for_pandas_flat_numbers():
    pytest.importorskip("pandas")

    out = library._process_array_for_pandas(
        awkward.contents.NumpyArray(numpy.array([1, 2])),
        True,
        AsDtype(">i4"),
        options={},
    )
    assert isinstance(out, numpy.ndarray)
    assert out.tolist() == [1, 2]


def test_jagged_in_pandas(tree):
    pytest.importorskip("pandas")
    pytest.importorskip("awkward_pandas")

    series = tree["z"].array(library="pd")
    assert [x.tolist() for x in series] == [[1], [], [2, 3]]


class UnregisteredLibrary(library.Library):
    name = "unregistered"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("np", np),
        ("numpy", np),
        ("NumPy", np),
        ("ak", ak),
        ("awkward", ak),
        ("Awkward", ak),
        ("pd", pd),
        ("pandas", pd),
        (np, np),
        (ak, ak),
        (library.NumPy, np),
        (library.Awkward, ak),
        (library.Pandas, pd),
    ],
    ids=str,
)
def test_regularize_library(value, expected):
    assert library._regularize_library(value) is expected


def test_regularize_library_errors():
    with pytest.raises(
        ValueError,
        match=r"^library UnregisteredLibrary \('unregistered'\) cannot be used in this function$",
    ):
        library._regularize_library(UnregisteredLibrary())

    with pytest.raises(
        ValueError,
        match=r"^library UnregisteredLibrary \('unregistered'\) cannot be used in this function$",
    ):
        library._regularize_library(UnregisteredLibrary)

    with pytest.raises(
        ValueError,
        match=r"""^library 'wrong' not recognized \(for this function\); try "np" \(NumPy\), """
        r""""ak" \(Awkward Array\), or "pd" \(Pandas\) instead$""",
    ):
        library._regularize_library("wrong")
