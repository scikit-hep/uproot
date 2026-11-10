# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest

import uproot

import awkward as ak
import numpy as np

TUPLE_DTYPES = [
    (("int32", (2,)), np.int32, (5, 2)),
    (("int32", (2, 4, 1)), np.int32, (5, 2, 4, 1)),
    # int32 that can also be viewed as 4 int8: the values are still one int32 each
    (("int32", ("int8", 4)), np.int32, (5,)),
    ("2i4", np.int32, (5, 2)),
    ("(2,3,4)f4", np.float32, (5, 2, 3, 4)),
]


def tuple_dtype_data(dtype, primitive, shape):
    data = np.zeros(5, dtype=dtype)
    data[...] = np.arange(np.prod(shape), dtype=primitive).reshape(shape)
    return data


@pytest.mark.parametrize("dtype, primitive, shape", TUPLE_DTYPES)
def test_mktree_tuple_dtype(tmp_path, dtype, primitive, shape):
    filepath = os.path.join(tmp_path, "test.root")
    data = tuple_dtype_data(dtype, primitive, shape)

    with uproot.recreate(filepath) as file:
        tree = file.mktree("tree", {"x": dtype})
        tree.extend({"x": data})

    with uproot.open(filepath) as file:
        assert file["tree"].num_entries == 5
        array = file["tree/x"].array(library="np")
        assert array.shape == data.shape
        assert array.tolist() == data.tolist()


@pytest.mark.parametrize("dtype, primitive, shape", TUPLE_DTYPES)
def test_mkrntuple_tuple_dtype(tmp_path, dtype, primitive, shape):
    filepath = os.path.join(tmp_path, "test.root")
    data = tuple_dtype_data(dtype, primitive, shape)

    with uproot.recreate(filepath) as file:
        ntuple = file.mkrntuple("ntuple", {"x": dtype})
        ntuple.extend({"x": data})

    with uproot.open(filepath) as file:
        assert file["ntuple"].num_entries == 5
        assert file["ntuple"].arrays()["x"].tolist() == data.tolist()


def test_unsupported_numpy_dtypes(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        # this should work, but it is interpreted as data, not a dtype
        file.mktree("tree", {"x": ("int32", "not a type")})
        file.mkrntuple("ntuple", {"x": ("int32", "not a type")})

        # structured dtypes are not supported
        complex_dtype = np.dtype([("real", "float32"), ("imag", "float32")])
        with pytest.raises(TypeError, match="cannot write NumPy dtype .* in TTree"):
            file.mktree("tree1", {"x": complex_dtype})
        with pytest.raises(TypeError, match="Cannot construct an Awkward Form from"):
            file.mkrntuple("ntuple1", {"x": complex_dtype})

        # structured dtypes specified as strings are also not supported
        dt_str = "i4, (2,3)f8, f4"
        with pytest.raises(TypeError, match="cannot write NumPy dtype .* in TTree"):
            file.mktree("tree2", {"x": dt_str})
        with pytest.raises(TypeError, match="Cannot construct an Awkward Form from"):
            file.mkrntuple("ntuple2", {"x": dt_str})

        # using tuples to specify structured dtypes is not supported
        dt_tuple = ("int32", [("real", "float32"), ("imag", "float32")])
        with pytest.raises(TypeError, match="cannot write Awkward Array type"):
            file.mktree("tree3", {"x": dt_tuple})
        # this should work, but it is interpreted as data, not a dtype
        file.mkrntuple("ntuple3", {"x": dt_tuple})

        # using some random type should not work
        dt_unknown = frozenset
        with pytest.raises(TypeError, match="cannot write NumPy dtype object"):
            file.mktree("tree4", {"x": dt_unknown})
        with pytest.raises(TypeError, match="Cannot construct a NumPy dtype"):
            file.mkrntuple("ntuple4", {"x": dt_unknown})

        with pytest.raises(AttributeError):
            file.mktree("tree5", {1: "unknown type"})
        with pytest.raises(ValueError, match="JSON syntax error"):
            file.mkrntuple("ntuple5", {1: "unknown type"})

        with pytest.raises(TypeError):
            # this one ends up creating a tree, but fails when writing the data
            file.mktree("tree6", {"x": "unknown type"})
        with pytest.raises(ValueError, match="JSON syntax error"):
            file.mkrntuple("ntuple6", {"x": "unknown type"})

    # make sure that things were written as data instead of being interpreted as dtypes
    with uproot.open(filepath) as file:
        assert file.keys(cycle=False) == ["tree", "ntuple", "ntuple3", "tree6"]
        assert file["tree"].arrays().x[0] == "int32"
        assert file["tree"].arrays().x[1] == "not a type"
        assert file["ntuple"].arrays().x[0] == "int32"
        assert file["ntuple"].arrays().x[1] == "not a type"
        assert file["ntuple3"].arrays().x[0] == "int32"
        assert ak.array_equal(
            file["ntuple3"].arrays().x[1], [("real", "float32"), ("imag", "float32")]
        )


def test_type_specification_to_awkward_form():
    form = ak.forms.EmptyForm()
    assert uproot.writing.writable._type_specification_to_awkward_form(form) is form

    assert uproot.writing.writable._type_specification_to_awkward_form(
        ak.types.NumpyType("int32")
    ) == ak.forms.NumpyForm("int32")

    with pytest.raises(TypeError, match="Cannot construct a NumPy dtype"):
        uproot.writing.writable._type_specification_to_awkward_form(frozenset)

    with pytest.raises(TypeError, match=r"from the tuple \(1, 2, 3\)"):
        uproot.writing.writable._type_specification_to_awkward_form((1, 2, 3))

    with pytest.raises(TypeError, match="Cannot construct an Awkward Form from int"):
        uproot.writing.writable._type_specification_to_awkward_form(0)
