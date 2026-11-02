# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import shutil
import types

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot
from uproot.interpretation.custom import CustomInterpretation
from uproot.writing.writable import WritableTree


def root_entries(path, name, *branches):
    """Every entry of the TTree, as seen by ROOT, as a list of lists."""
    ROOT = pytest.importorskip("ROOT")

    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get(name)
    out = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        entry = []
        for branch in branches:
            value = getattr(tree, branch)
            if hasattr(value, "__len__") and not isinstance(value, str):
                length = int(getattr(tree, "n" + branch))
                value = [value[j] for j in range(length)]
            entry.append(value)
        out.append(entry)
    root_file.Close()
    return out


def low_level_tree(directory, branch_types):
    """
    A tree made by the cascade directly, which (unlike mktree) doesn't filter
    the branch types first and accepts them as (name, type) pairs.
    """
    tree = WritableTree(
        ("t",),
        directory.file,
        directory._cascading.add_tree(
            directory.file.sink,
            "t",
            "",
            branch_types,
            lambda counted: "n" + counted,
            lambda outer, inner: outer + "_" + inner,
            10,
            10.0,
        ),
    )
    directory.file._new_tree(tree)
    return tree


def test_branch_types_can_be_any_mapping(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", types.MappingProxyType({"ab": "float64"}))
        tree.extend({"ab": np.array([1.5, 2.5])})

    assert root_entries(path, "t", "ab") == [[1.5], [2.5]]


def test_no_branches(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(ValueError, match="must have at least one branch"):
            file.mktree("t", {})


@pytest.mark.parametrize(
    "branch_types, expected",
    [
        ([("a", "float64"), ("a", "int32")], ["a"]),
        ([("r", {"x": "float64"}), ("r", {"y": "float64"})], ["r_x"]),
        ([("r_x", "int32"), ("r", {"x": "float64", "y": "float64"})], ["r_x", "r_y"]),
        ([("s", "string"), ("s", "string")], ["s"]),
        ([("j", "var * int32"), ("j", "var * float64")], ["nj", "j"]),
        (
            [("r", "var * {x: int32}"), ("r", "var * {y: int32}")],
            ["nr", "r_x"],
        ),
        (
            [("r_x", "float64"), ("r", "var * {x: int32, y: int32}")],
            ["r_x", "nr", "r_y"],
        ),
        ([("r", "{x: int32}"), ("r", "{y: int32}")], ["r_x"]),
        (
            [("r_x", "float64"), ("r", "{x: int32, y: int32}")],
            ["r_x", "r_y"],
        ),
    ],
)
def test_first_branch_of_a_name_wins(tmp_path, branch_types, expected):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        low_level_tree(file, branch_types)

    with uproot.open(path) as file:
        assert file["t"].keys() == expected


@pytest.mark.parametrize(
    "branch_type, error, message",
    [
        ("not a type", TypeError, "not a NumPy dtype or an Awkward datashape"),
        ({"x": "var * int32"}, TypeError, "values of a dict must be NumPy types"),
        (
            ak.types.ListType(
                ak.types.NumpyType("uint8"), parameters={"__array__": "bytes"}
            ),
            NotImplementedError,
            "array of bytes",
        ),
        ("var * var * int32", TypeError, "cannot write Awkward Array type"),
        ("3 * var * int32", TypeError, "cannot write Awkward Array type"),
        (
            "var * {x: var * int32}",
            TypeError,
            "fields of a record must be NumPy types",
        ),
        ("{x: var * int32}", TypeError, "fields of a record must be NumPy types"),
    ],
)
def test_invalid_branch_types(tmp_path, branch_type, error, message):
    with uproot.recreate(tmp_path / "out.root") as file:
        with pytest.raises(error, match=message):
            low_level_tree(file, {"x": branch_type})


def test_bytes_are_jagged_integers(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"x": "bytes"})
        tree.extend({"x": ak.Array([[1, 2], [3]])})

    with uproot.open(path) as file:
        assert file["t/x"].typename == "uint8_t[]"
        assert file["t/x"].array().tolist() == [[1, 2], [3]]


@pytest.mark.parametrize(
    "branch_type, data, expected",
    [
        (
            "(int32, float64)",
            ak.Array([(1, 1.5), (2, 2.5)]),
            [[1, 1.5], [2, 2.5]],
        ),
        (
            "var * (int32, float64)",
            ak.Array([[(1, 1.5), (2, 2.5)], []]),
            [[[1, 2], [1.5, 2.5]], [[], []]],
        ),
    ],
    ids=["flat", "jagged"],
)
def test_tuples_become_numbered_branches(tmp_path, branch_type, data, expected):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"r": branch_type}, counter_name=lambda x: "n" + x)
        tree.extend({"r": data})

    with uproot.open(path) as file:
        assert file["t"]["r_0"].array().tolist() == [x[0] for x in expected]
        assert file["t"]["r_1"].array().tolist() == [x[1] for x in expected]


def test_capacity_grows_with_records(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree(
            "t",
            {"rec": {"x": "int32", "y": "float64"}},
            initial_basket_capacity=2,
            resize_factor=2.0,
        )
        for i in range(5):
            tree.extend({"rec": {"x": np.array([i]), "y": np.array([i + 0.5])}})
        assert tree.num_baskets == 5
        assert tree._cascading.basket_capacity == 8

    expected = [[i, i + 0.5] for i in range(5)]
    assert root_entries(path, "t", "rec_x", "rec_y") == expected
    with uproot.open(path) as file:
        assert file["t"]["rec_x"].num_baskets == 5
        assert file["t"].arrays(library="np")["rec_y"].tolist() == [
            x[1] for x in expected
        ]


def test_extend_from_no_basket_capacity(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"x": "i4"}, initial_basket_capacity=0)
        tree.extend({"x": np.array([1, 2], dtype="i4")})
        tree.extend({"x": np.array([3], dtype="i4")})

    with uproot.open(path) as file:
        assert file["t/x"].array(library="np").tolist() == [1, 2, 3]
        assert file["t/x"].member("fBasketEntry").tolist()[:3] == [0, 2, 3]


def test_update_leaves_out_an_uninterpretable_branch(tmp_path):
    path = tmp_path / "out.root"
    shutil.copy(skhep_testdata.data_path("uproot-issue-400.root"), path)

    with uproot.update(path) as file:
        tree = file["orange"]
        with pytest.raises(NotImplementedError, match=r"extend"):
            tree.extend({"orange": np.array([1], dtype="i4")})


class ObjectDtype(CustomInterpretation):
    numpy_dtype = np.dtype("O")

    @classmethod
    def match_branch(cls, branch, context, simplify):
        return branch.name == "x"


class NoneDtype(ObjectDtype):
    numpy_dtype = None


class MissingDtype(CustomInterpretation):
    @classmethod
    def match_branch(cls, branch, context, simplify):
        return branch.name == "x"


@pytest.fixture
def plain_tree(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        file.mktree("t", {"x": "i4"}).extend({"x": np.array([1, 2], dtype="i4")})
    return path


def test_update_rejects_a_custom_object_interpretation(plain_tree):
    uproot.register_interpretation(ObjectDtype)
    try:
        with uproot.update(plain_tree) as file:
            with pytest.raises(
                NotImplementedError, match=r"^branch 'x' has interpretation ObjectDtype"
            ):
                file["t"]
    finally:
        uproot.unregister_interpretation(ObjectDtype)


@pytest.mark.parametrize("custom", [NoneDtype, MissingDtype])
def test_update_leaves_out_a_custom_interpretation_without_dtype(plain_tree, custom):
    uproot.register_interpretation(custom)
    try:
        with uproot.update(plain_tree) as file:
            with pytest.raises(NotImplementedError, match=r"extend"):
                file["t"].extend({"x": np.array([3], dtype="i4")})
    finally:
        uproot.unregister_interpretation(custom)

    with uproot.open(plain_tree) as file:
        assert file["t/x"].array(library="np").tolist() == [1, 2]


def test_records_as_lists_of_dicts(tmp_path):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"rec": {"x": "int32", "y": "float64"}})
        tree.extend({"rec": [{"x": 1, "y": 1.5}, {"x": 2, "y": 2.5}]})

    assert root_entries(path, "t", "rec_x", "rec_y") == [[1, 1.5], [2, 2.5]]


def test_records_as_pandas_series(tmp_path):
    pandas = pytest.importorskip("pandas")

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"rec": {"x": "int32", "y": "float64"}})
        tree.extend({"rec": pandas.Series({"x": [1, 2], "y": [1.5, 2.5]})})

    assert root_entries(path, "t", "rec_x", "rec_y") == [[1, 1.5], [2, 2.5]]


def test_extend_needs_a_mapping(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"x": "float64"})
        with pytest.raises(TypeError, match="requires a mapping from branch name"):
            tree.extend([1.0, 2.0])
        with pytest.raises(TypeError, match="requires a mapping from branch name"):
            tree.extend(types.MappingProxyType({1: np.array([1.0, 2.0])}))


def test_extend_needs_a_mapping_not_pandas_without_numeric_index(tmp_path):
    pandas = pytest.importorskip("pandas")

    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"x": "float64"})
        with pytest.raises(TypeError, match="requires a mapping from branch name"):
            tree.extend(pandas.Series([1.0, 2.0]))
        with pytest.raises(TypeError, match="requires a mapping from branch name"):
            tree.extend(pandas.DataFrame({"x": [1.0, 2.0]}, index=["a", "b"]))


@pytest.mark.parametrize(
    "data, message",
    [
        (
            {"x": np.array([1.0])},
            "must be given an array for every branch; missing 'r'",
        ),
        (
            {"r": {"a": np.array([1.0])}, "y": np.array([1.0])},
            "must be given an array for every branch; missing 'x'",
        ),
        (
            {"r": {"a": np.array([1.0])}, "x": np.array([1.0]), "z": np.array([1.0])},
            "data that do not correspond to any branch: 'z'",
        ),
    ],
    ids=["missing-record", "missing-branch", "extra-branch"],
)
def test_extend_needs_exactly_the_branches(tmp_path, data, message):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"r": {"a": "float64"}, "x": "float64"})
        with pytest.raises(ValueError, match=message):
            tree._cascading.extend(file.file, file.file.sink, data)


def test_categorical_strings(tmp_path):
    categories = ak.Array(["one", "three"]).layout
    index = ak.index.Index64([0, 1, 0])

    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"s": "string"})
        tree.extend({"s": ak.Array(ak.contents.IndexedArray(index, categories))})

    with uproot.open(path) as file:
        assert file["t/s"].array().tolist() == ["one", "three", "one"]

    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("t")
    found = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        found.append(tree.GetLeaf("s").GetValueString())
    assert found == ["one", "three", "one"]
    root_file.Close()


def test_numbers_are_not_strings(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"s": "string"})
        with pytest.raises(ValueError, match="exceeds the depth"):
            tree.extend({"s": np.array([1, 2, 3])})


JAGGED = ak.Array([[1, 2], [], [3]])


@pytest.mark.parametrize(
    "layout, expected",
    [
        (
            ak.contents.IndexedArray(ak.index.Index64([2, 0]), JAGGED.layout),
            [[3], [1, 2]],
        ),
        (
            ak.contents.ListArray(
                ak.index.Index64([3, 0]),
                ak.index.Index64([5, 2]),
                ak.contents.NumpyArray(np.array([1, 2, 0, 3, 4], np.int32)),
            ),
            [[3, 4], [1, 2]],
        ),
        (
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 2, 3]),
                ak.contents.IndexedArray(
                    ak.index.Index64([2, 1, 0]),
                    ak.contents.NumpyArray(np.array([1, 2, 3], np.int32)),
                ),
            ),
            [[3, 2], [1]],
        ),
    ],
    ids=["indexed-lists", "list-array", "indexed-content"],
)
def test_jagged_layouts(tmp_path, layout, expected):
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {"x": "var * int32"})
        tree.extend({"x": ak.Array(layout)})

    with uproot.open(path) as file:
        assert file["t/x"].array().tolist() == expected
    assert root_entries(path, "t", "x") == [[x] for x in expected]


@pytest.mark.parametrize(
    "array, counts, message",
    [
        (
            ak.Array([1, 2, 3]),
            [1, 1, 1],
            "jagged branch 'x' can't be filled with this",
        ),
        (
            ak.to_regular(ak.Array([[1], [2], [3]])),
            [1, 1, 1],
            "jagged branch 'x' can't be filled with this",
        ),
        (
            ak.Array([[1, None], [], [3]]),
            [2, 0, 1],
            "jagged branch 'x' can't be filled with lists of this",
        ),
        (
            ak.Array([["one"], [], ["three"]]),
            [1, 0, 1],
            "jagged branch 'x' can't be filled with lists of this",
        ),
    ],
    ids=["flat", "regular", "option", "strings"],
)
def test_jagged_branch_rejects_other_arrays(tmp_path, array, counts, message):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"x": "var * int32"})
        with pytest.raises(TypeError, match=message):
            tree.extend({"x": array, "nx": np.array(counts)})


def test_jagged_inner_shape_must_match(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree("t", {"x": "var * 2 * int32"})
        wrong = ak.to_regular(ak.Array([[[1, 2, 3]], [[4, 5, 6], [7, 8, 9]]]), axis=2)
        with pytest.raises(
            ValueError,
            match=r"consistent shape: has \(2,\), trying to fill with \(3,\)",
        ):
            tree.extend({"x": wrong})


@pytest.mark.parametrize("length", [254, 255, 300])
def test_long_branch_names(tmp_path, length):
    name = "x" * length
    path = tmp_path / "out.root"
    with uproot.recreate(path) as file:
        tree = file.mktree("t", {name: "var * int32"}, counter_name=lambda counted: "n")
        tree.extend({name: ak.Array([[1], [2, 3]])})

    with uproot.open(path) as file:
        assert file["t"].keys() == ["n", name]
        assert file["t"][name].array().tolist() == [[1], [2, 3]]

    ROOT = pytest.importorskip("ROOT")
    root_file = ROOT.TFile.Open(str(path))
    tree = root_file.Get("t")
    leaf = tree.GetBranch(name).GetLeaf(name)
    assert leaf.GetName() == name
    assert leaf.GetTitle() == name + "[n]"
    assert tree.Draw(f"Sum$({name})", "", "goff") == 2
    assert [tree.GetV1()[i] for i in range(2)] == [1, 5]
    root_file.Close()


@pytest.mark.parametrize(
    "branch_type, data, message",
    [
        ("float64", np.arange(10.0), "Numpy array data of branch x has"),
        ("var * float64", ak.Array([[1.0, 2.0]] * 5), "Jagged array data of branch x"),
        ("string", ak.Array(["some string"] * 5), "String data of branch x"),
    ],
    ids=["numpy", "jagged", "string"],
)
def test_baskets_have_a_maximum_size(tmp_path, monkeypatch, branch_type, data, message):
    # two gigabytes would be needed to get to the real limit
    monkeypatch.setattr(uproot.const, "kMaxTBasketBytes", 100)
    with uproot.recreate(tmp_path / "out.root", compression=None) as file:
        tree = file.mktree("t", {"x": branch_type})
        message += ".* which is too large to fit in a TBasket"
        with pytest.raises(ValueError, match=message) as err:
            tree.extend({"x": data})
        assert "Neither of these sizes can exceed 2 GiB" in str(err.value)


def test_cascade_tree_attributes(tmp_path):
    with uproot.recreate(tmp_path / "out.root") as file:
        tree = file.mktree(
            "t",
            {"x": "float64"},
            title="the title",
            initial_basket_capacity=3,
            resize_factor=2.5,
        )
        cascading = tree._cascading
        assert cascading.name.string == "t"
        assert cascading.title.string == "the title"
        assert cascading.freesegments is file.file._cascading.freesegments
        assert cascading.directory is file._cascading
        assert cascading.counter_name("x") == "nx"
        assert cascading.field_name("outer", "inner") == "outer_inner"
        assert cascading.basket_capacity == 3
        assert cascading.resize_factor == 2.5
        assert cascading.location == cascading.key.location
        assert cascading.num_baskets == 0

        representation = repr(cascading)
        assert representation.startswith("Tree(RootDirectory(Key(")
        assert "t, the title, [('x', 'float64')]" in representation
        assert representation.endswith(", 3, 2.5)")
