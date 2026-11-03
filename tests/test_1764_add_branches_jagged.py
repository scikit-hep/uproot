# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import io
import re
import shutil

import numpy as np
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

J = [[1.0], [], [2.0, 3.0]]


def _tree(path, branches, data):
    with uproot.recreate(path) as f:
        f.mktree("tree", branches)
        f["tree"].extend(data)


def _flat_tree(path, data):
    _tree(path, {"a": np.float64}, {"a": data})


def _jagged_tree(path, data):
    _tree(path, {"x": "var * float64"}, {"x": data})


def _values(tree, name):
    if tree[name].interpretation.cache_key.startswith("AsJagged"):
        return tree[name].array().tolist()
    return tree[name].array(library="np").tolist()


def _indexed_layouts():
    lists = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3, 3])),
        ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0])),
    )
    outer = ak.Array(
        ak.contents.IndexedArray(ak.index.Index64(np.array([2, 0, 1])), lists)
    )
    inner = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 3, 3])),
            ak.contents.IndexedArray(
                ak.index.Index64(np.array([2, 1, 0])),
                ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0])),
            ),
        )
    )
    return {"p": outer, "q": inner}


@pytest.mark.parametrize(
    "branches, counter_name, expected, titles",
    [
        pytest.param(
            {"j": ak.Array(J)}, None, {"nj": [1, 0, 2], "j": J}, {}, id="new-counter"
        ),
        pytest.param({"j": J}, None, {"nj": [1, 0, 2], "j": J}, {}, id="lists"),
        pytest.param(
            {"j": ak.Array(J)},
            lambda b: f"num_{b}",
            {"num_j": [1, 0, 2], "j": J},
            {},
            id="custom-counter-name",
        ),
        pytest.param(
            {
                "nj": np.array([1, 0, 2], dtype=np.int32),
                "j": ak.Array([[10.0], [], [20.0, 30.0]]),
            },
            None,
            {"nj": [1, 0, 2], "j": [[10.0], [], [20.0, 30.0]]},
            {},
            id="explicit-counter",
        ),
        pytest.param(
            {"j": ak.Array(J), "nj": np.array([1.0, 0.0, 2.0])},
            None,
            {"nj": [1, 0, 2], "j": J},
            {"nj": "nj/I"},
            id="explicit-float-counter-is-int32",
        ),
        pytest.param(
            {"y": np.array([4.0, 5.0, 6.0]), "j": ak.Array([[1.5], [], [2.5, 3.5]])},
            None,
            {"nj": [1, 0, 2], "y": [4.0, 5.0, 6.0], "j": [[1.5], [], [2.5, 3.5]]},
            {},
            id="flat-and-jagged",
        ),
        pytest.param(
            {"j1": ak.Array(J), "j2": ak.Array([[10.0], [], [20.0, 30.0]])},
            lambda _: "n_particles",
            {"n_particles": [1, 0, 2], "j1": J, "j2": [[10.0], [], [20.0, 30.0]]},
            {},
            id="shared-counter",
        ),
        pytest.param(
            {
                "p": ak.Array(J),
                "q": ak.Array([[4], [], [5, 6]]),
                "n": np.array([1, 0, 2]),
            },
            lambda _: "n",
            {"n": [1, 0, 2], "p": J, "q": [[4], [], [5, 6]]},
            {},
            id="shared-explicit-counter",
        ),
        pytest.param(
            {
                "c1": np.array([1, 2, 0], dtype=np.int32),
                "j1": ak.Array([[1.0], [2.0, 3.0], []]),
                "c2": np.array([2, 1, 0], dtype=np.int32),
                "j2": ak.Array([[10.0, 20.0], [30.0], []]),
            },
            lambda b: "c1" if b == "j1" else "c2",
            {
                "c1": [1, 2, 0],
                "c2": [2, 1, 0],
                "j1": [[1.0], [2.0, 3.0], []],
                "j2": [[10.0, 20.0], [30.0], []],
            },
            {},
            id="explicit-counters-in-order",
        ),
        pytest.param(
            {"j": ak.Array([[1.0, 2.0], [], [3.0], [4.0, 5.0], [6.0]])[::2]},
            None,
            {"nj": [2, 1, 1], "j": [[1.0, 2.0], [3.0], [6.0]]},
            {},
            id="sliced",
        ),
        pytest.param(
            _indexed_layouts(),
            None,
            {
                "np": [0, 1, 2],
                "nq": [1, 2, 0],
                "p": [[], [1.0], [2.0, 3.0]],
                "q": [[3.0], [2.0, 1.0], []],
            },
            {},
            id="indexed-layouts",
        ),
    ],
)
def test_add_branches(tmp_path, branches, counter_name, expected, titles):
    path = str(tmp_path / "add.root")
    _tree(path, {"x": "int32"}, {"x": np.array([1, 2, 3], dtype=np.int32)})

    with uproot.update(path) as f:
        f["tree"].add_branches(branches, counter_name=counter_name)

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x", *expected]
        assert _values(tree, "x") == [1, 2, 3]
        for name, values in expected.items():
            assert _values(tree, name) == values
        for name, title in titles.items():
            assert tree[name].title == title
            assert tree[name].array(library="np").dtype == np.dtype(title[-1].lower())


@pytest.mark.parametrize(
    "branches, counter_name, error, match",
    [
        pytest.param(
            {"j": ak.Array(J), "nj": np.array([9, 9, 9])},
            None,
            ValueError,
            "disagree: it's the name of the counter of jagged branch 'j'",
            id="explicit-counter-disagrees",
        ),
        pytest.param(
            {"j": ak.Array([[1.0], [2.0]])},
            None,
            ValueError,
            "all new branches must match the tree length",
            id="wrong-length",
        ),
        pytest.param(
            {"j": ak.Array(J)},
            "count_{}",
            TypeError,
            "counter_name must be a callable or None",
            id="string-counter-name",
        ),
        pytest.param(
            {"j": ak.Array(J)},
            12345,
            TypeError,
            "counter_name must be a callable or None",
            id="int-counter-name",
        ),
        pytest.param(
            {"y": ak.Array(J)},
            None,
            ValueError,
            "collides with a branch of the same name that is not a counter",
            id="counter-is-a-normal-branch",
        ),
        pytest.param(
            {"y": ak.Array(J)},
            lambda _: "x",
            ValueError,
            "collides",
            id="counter-is-a-jagged-branch",
        ),
        pytest.param(
            {"j": ak.Array(J), "nj": ak.Array([[1.0], [], []])},
            None,
            ValueError,
            "another new jagged branch",
            id="counter-is-a-new-jagged-branch",
        ),
        pytest.param(
            {"j1": ak.Array(J), "j2": ak.Array([[10.0, 20.0], [], [30.0]])},
            lambda _: "n_particles",
            ValueError,
            "disagrees with another jagged branch",
            id="shared-counter-disagrees",
        ),
        pytest.param(
            {"y": ak.Array([[1, 2, 3], [4], []])},
            lambda _: "nx",
            ValueError,
            "disagrees with the existing counter branch's lengths",
            id="existing-counter-disagrees",
        ),
        pytest.param(
            {"x": ak.Array(J)},
            None,
            ValueError,
            "already exists",
            id="existing-branch",
        ),
        pytest.param(
            {"b": np.array(["a", object(), 123], dtype=object)},
            None,
            TypeError,
            "object dtype",
            id="object-dtype",
        ),
        pytest.param(
            {"j": ak.Array([[[1.0]], [], [[2.0], [3.0]]])},
            None,
            TypeError,
            "cannot write Awkward Array type",
            id="nested-jagged",
        ),
        pytest.param(
            {"j": ak.Array([["a"], [], ["b", "c"]])},
            None,
            TypeError,
            "cannot write Awkward Array type",
            id="jagged-strings",
        ),
        pytest.param(
            {"j": ak.Array([[1.0], None, []])},
            None,
            TypeError,
            "cannot write Awkward Array type",
            id="option-type",
        ),
        pytest.param(
            {"j": ak.Array(J), "s": np.array(["x", "y", "z"])},
            None,
            TypeError,
            "cannot write NumPy dtype",
            id="good-and-bad-branch",
        ),
        pytest.param(
            {"b": np.empty((3, 0))},
            None,
            ValueError,
            "empty fixed-size dimension",
            id="empty-dimension",
        ),
        pytest.param(
            {"b": np.zeros((3, 2, 0))},
            None,
            ValueError,
            "empty fixed-size dimension",
            id="empty-inner-dimension",
        ),
        pytest.param(
            {"b": [[[], []], [[], []], [[], []]]},
            None,
            ValueError,
            "empty fixed-size dimension",
            id="nested-empty-lists",
        ),
    ],
)
def test_add_branches_raises(tmp_path, branches, counter_name, error, match):
    path = str(tmp_path / "rejected.root")
    _tree(
        path,
        {"x": "var * float64", "ny": np.float32},
        {"x": ak.Array(J), "ny": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
    )

    with uproot.update(path) as f:
        tree = f["tree"]
        with pytest.raises(error, match=match):
            tree.add_branches(branches, counter_name=counter_name)
        # nothing was added, and the tree still works
        tree.extend({"x": ak.Array([[4.0]]), "ny": np.array([4.0], dtype=np.float32)})

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["nx", "x", "ny"]
        assert _values(f["tree"], "x") == [*J, [4.0]]


@pytest.mark.parametrize(
    "field, expected, title",
    [
        pytest.param(
            ak.Array([[1.0], [2.0, 3.0]]),
            [[], [], [1.0], [2.0, 3.0]],
            "j[nj]/D",
            id="jagged",
        ),
        pytest.param(
            [[1.0], [2.0, 3.0]], [[], [], [1.0], [2.0, 3.0]], "j[nj]/D", id="lists"
        ),
        pytest.param(ak.Array([[1, 2], []]), [[], [], [1, 2], []], "j[nj]/L", id="int"),
        pytest.param([[], []], [[], [], [], []], "j[nj]/D", id="all-empty-lists"),
        pytest.param(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
            "j[2]/D",
            id="fixed-size",
        ),
    ],
)
def test_extend_accept_new_fields(tmp_path, field, expected, title):
    path = str(tmp_path / "new_field.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        f["tree"].extend(
            {"a": np.array([2.0, 3.0]), "j": field}, accept_new_fields=True
        )

    with uproot.open(path) as f:
        assert f["tree"]["j"].title == title
        assert _values(f["tree"], "j") == expected
        if "[nj]" in title:
            assert _values(f["tree"], "nj") == [len(x) for x in expected]


def test_extend_accept_new_fields_shares_existing_counter(tmp_path):
    path = str(tmp_path / "shared_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "var * float64"}, counter_name=lambda counted: "n")
        f["tree"].extend({"x": ak.Array([[1.0], [2.0, 3.0]])})
        f["tree"].extend(
            {
                "x": ak.Array([[4.0], []]),
                "k": ak.Array([[5], []]),
                "m": ak.Array([[6.0], []]),
            },
            accept_new_fields=True,
        )

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["n", "x", "k", "m"]
        assert _values(f["tree"], "n") == [1, 2, 1, 0]
        assert _values(f["tree"], "k") == [[0], [0, 0], [5], []]
        assert _values(f["tree"], "m") == [[0.0], [0.0, 0.0], [6.0], []]


@pytest.mark.parametrize(
    "new_fields, error, match",
    [
        pytest.param(
            {"j": ak.Array([[1.0], [], [2.0]])},
            ValueError,
            "same number of entries",
            id="new-field-length",
        ),
        pytest.param(
            {"j": ak.Array([[1.0], [2.0]]), "nj": np.array([5, 5])},
            ValueError,
            "they disagree",
            id="explicit-counter-disagrees",
        ),
        pytest.param(
            {"a": np.ones((2, 3)), "b": np.arange(2)},
            ValueError,
            "consistent shape",
            id="existing-field-shape-new-flat",
        ),
        pytest.param(
            {"a": np.ones((2, 3)), "k": ak.Array([[1.0], [2.0, 3.0]])},
            ValueError,
            "consistent shape",
            id="existing-field-shape-new-jagged",
        ),
        pytest.param(
            {"j": ak.Array([[[1.0]], []])},
            TypeError,
            "cannot write Awkward Array type",
            id="nested-jagged",
        ),
        pytest.param(
            {"b": ak.Array([[1.0], []])},
            ValueError,
            "collides with a branch .* add_branches with a counter_name",
            id="counter-is-a-normal-branch",
        ),
    ],
)
def test_rejected_accept_new_fields_leaves_tree_alone(
    tmp_path, new_fields, error, match
):
    path = str(tmp_path / "rejected.root")
    _tree(
        path,
        {"a": np.float64, "nb": np.int32},
        {"a": np.arange(2.0), "nb": np.arange(2, dtype=np.int32)},
    )
    data = {
        "a": np.array([2.0, 3.0]),
        "nb": np.array([1, 1], dtype=np.int32),
        **new_fields,
    }

    with uproot.update(path) as f:
        with pytest.raises(error, match=match):
            f["tree"].extend(data, accept_new_fields=True)
        assert list(f["tree"]._cascading._branch_lookup) == ["a", "nb"]
        f["tree"].extend({"a": np.array([2.0]), "nb": np.array([1], dtype=np.int32)})

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["a", "nb"]
        assert _values(f["tree"], "a") == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "branch_type, batches, expected",
    [
        pytest.param(
            "var * float64", [[[1.0], []], [[7.0]]], [[1.0], [], [7.0]], id="one-list"
        ),
        pytest.param(
            "var * float64",
            [[[1.0], []], [[1.0, 2.0], [3.0, 4.0]]],
            [[1.0], [], [1.0, 2.0], [3.0, 4.0]],
            id="equal-lengths",
        ),
        pytest.param(
            "var * float64",
            [[[1.0]], [[], [], []]],
            [[1.0], [], [], []],
            id="all-empty",
        ),
        pytest.param(
            "var * float64",
            [[[1.0], [2.0, 3.0]], []],
            [[1.0], [2.0, 3.0]],
            id="no-lists",
        ),
        pytest.param(
            "var * 2 * float64",
            [[[[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]]],
            [[[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]],
            id="fixed-size-items",
        ),
        pytest.param(
            "var * 3 * float64",
            [[[], []], [[[1.0, 2.0, 3.0]], []]],
            [[], [], [[1.0, 2.0, 3.0]], []],
            id="fixed-size-items-all-empty",
        ),
    ],
)
def test_extend_jagged_branch_with_lists(tmp_path, branch_type, batches, expected):
    """The first batch goes to a new tree, the others to the tree reopened."""
    path = str(tmp_path / "lists.root")
    _tree(path, {"x": branch_type}, {"x": batches[0]})

    with uproot.update(path) as f:
        for batch in batches[1:]:
            f["tree"].extend({"x": batch})
        assert f["tree"].num_entries == len(expected)

    with uproot.open(path) as f:
        assert _values(f["tree"], "x") == expected


@pytest.mark.parametrize("rows", [[[[1.0, 2.0], [3.0]]], [[1.0, 2.0]]])
def test_extend_jagged_branch_of_fixed_size_lists_checks_items(tmp_path, rows):
    path = str(tmp_path / "wrong_items.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"v": "var * 2 * float64"})
        with pytest.raises(ValueError, match="consistent shape"):
            f["tree"].extend({"v": rows})


def test_empty_tree_add_jagged_branches_then_extend(tmp_path):
    path = str(tmp_path / "empty.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "var * float64"})

    with uproot.update(path) as f:
        assert f["tree"].num_baskets == 0
        f["tree"].add_branches({"y": ak.Array([[1]])[:0]}, counter_name=lambda _: "nx")
        f["tree"].add_branches({"j": ak.Array([[1.0]])[:0]})

    with uproot.update(path) as f:
        casc = f["tree"]._cascading
        assert casc._branch_data[casc._branch_lookup["j"]]["counter"] is not None
        f["tree"].extend(
            {
                "x": ak.Array([[1.0, 2.0]]),
                "y": ak.Array([[3, 4]]),
                "j": ak.Array([[10.0, 20.0, 30.0]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["nx", "x", "y", "nj", "j"]
        assert tree["j"].title == "j[nj]/D"
        assert tree["y"].title == "y[nx]/L"
        assert _values(tree, "y") == [[3, 4]]
        assert _values(tree, "nj") == [3]
        assert _values(tree, "j") == [[10.0, 20.0, 30.0]]


def test_counter_maximum_is_kept_and_set(tmp_path):
    """ROOT sizes its read buffer for a jagged branch from fMaximum of its counter."""
    path = str(tmp_path / "maximum.root")
    _jagged_tree(path, ak.Array([[1.0, 2.0, 3.0], []]))

    def maximum(f, name):
        return f["tree"][name].member("fLeaves")[0].member("fMaximum")

    # an extend with no entries right after the tree is rewritten
    with uproot.update(path) as f:
        f["tree"].extend({"x": ak.Array([[1.0]])[:0]})
    with uproot.open(path) as f:
        assert maximum(f, "nx") == 3
        assert _values(f["tree"], "x") == [[1.0, 2.0, 3.0], []]

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": ak.Array([[1.0], [2.0, 3.0]])})
    with uproot.update(path) as f:
        f["tree"].extend({"x": ak.Array([[2.0, 3.0]]), "j": ak.Array([[4.0]])})

    with uproot.open(path) as f:
        assert maximum(f, "nx") == 3
        assert maximum(f, "nj") == 2
        assert _values(f["tree"], "x") == [[1.0, 2.0, 3.0], [], [2.0, 3.0]]
        assert _values(f["tree"], "j") == [[1.0], [2.0, 3.0], [4.0]]


def test_add_branches_releases_and_reuses_file_space(tmp_path):
    path = str(tmp_path / "space.root")
    _flat_tree(path, np.arange(3.0))
    data = ak.Array(J)

    with uproot.update(path) as f:
        key = f["tree"]._cascading._key
        start = key.seek_location
        stop = start + key.num_bytes + key.compressed_bytes
        f["tree"].add_branches({"j0": data})

    with uproot.update(path) as f:
        slices = f._cascading.freesegments._data.slices
        assert any(free[0] <= start and stop <= free[1] for free in slices)

    for i in range(1, 6):
        with uproot.update(path) as f:
            f["tree"].add_branches({f"j{i}": data})

    with uproot.open(path) as f:
        for i in range(6):
            assert _values(f["tree"], f"j{i}") == J


@pytest.mark.parametrize("fails", ["write_jagged_basket", "write_updates"])
def test_add_branches_failure(tmp_path, monkeypatch, fails):
    """
    A failure before the TTree metadata is rewritten leaves the tree as it was;
    one after leaves the new branches, as the metadata on disk has them.
    """
    path = str(tmp_path / "failure.root")
    _flat_tree(path, np.arange(2.0))
    data = ak.Array([[1.0], [2.0, 3.0]])

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with uproot.update(path) as f:
        tree = f["tree"]
        casc = tree._cascading
        fTotBytes = casc._metadata["fTotBytes"]
        with monkeypatch.context() as context:
            context.setattr(uproot.writing._cascadetree.Tree, fails, fail)
            with pytest.raises(OSError, match="disk full"):
                tree.add_branches({"j": data})

        if fails == "write_jagged_basket":
            assert list(casc._branch_lookup) == ["a"]
            assert casc._metadata["fTotBytes"] == fTotBytes
            tree.extend({"a": np.array([2.0])})
            expected = {"a": [0.0, 1.0, 2.0]}
        else:
            assert list(casc._branch_lookup) == ["a", "nj", "j"]
            tree.extend({"a": np.array([2.0]), "j": ak.Array([[4.0]])})
            expected = {"a": [0.0, 1.0, 2.0], "nj": [1, 2, 1], "j": [*data, [4.0]]}

    with uproot.open(path) as f:
        assert f["tree"].keys() == list(expected)
        for name, values in expected.items():
            assert _values(f["tree"], name) == ak.to_list(values)


def test_add_jagged_branch_existing_counter_keeps_file_object_open():
    data = ak.Array(J)
    sink = io.BytesIO()

    with uproot.recreate(sink) as f:
        f.mktree("tree", {"x": "var * float64"})
        f["tree"].extend({"x": data})
        with pytest.raises(ValueError, match="disagrees with the existing"):
            f["tree"].add_branches(
                {"bad": ak.Array([[1.0, 2.0], [], [3.0]])},
                counter_name=lambda _: "nx",
            )
        f["tree"].add_branches(
            {"y": data * 2, "z": data * 3}, counter_name=lambda _: "nx"
        )
        assert not sink.closed
        contents = sink.getvalue()

    with uproot.open(io.BytesIO(contents)) as f:
        assert f["tree"].keys() == ["nx", "x", "y", "z"]
        assert _values(f["tree"], "z") == [[3.0], [], [6.0, 9.0]]


def test_add_jagged_branch_compressed_existing_counter(tmp_path):
    path = str(tmp_path / "compressed.root")
    data = ak.Array([[1.0, 2.0]] * 5000)
    with uproot.recreate(path, compression=uproot.ZLIB(4)) as f:
        f.mktree("tree", {"x": "var * float64"})
        f["tree"].extend({"x": data})

    with uproot.update(path) as f:
        f["tree"].add_branches({"y": data * 2}, counter_name=lambda _: "nx")
        with pytest.raises(ValueError, match="disagrees with the existing"):
            f["tree"].add_branches(
                {"z": ak.Array([[1.0]] * 5000)}, counter_name=lambda _: "nx"
            )

    with uproot.open(path) as f:
        assert ak.array_equal(f["tree"]["y"].array(), data * 2)


def test_existing_counters_are_read_with_one_read_of_the_tree(tmp_path, monkeypatch):
    path = str(tmp_path / "two_counters.root")
    _tree(
        path,
        {"x": "var * float64", "y": "var * float64"},
        {"x": ak.Array([[1.0], []]), "y": ak.Array([[], [2.0, 3.0]])},
    )

    reads = []
    original = uproot.writing.writable.WritableFile._read_key

    def counting(self, key, parent=None):
        reads.append(key)
        return original(self, key, parent)

    with uproot.update(path) as f:
        tree = f["tree"]
        monkeypatch.setattr(uproot.writing.writable.WritableFile, "_read_key", counting)
        tree.add_branches(
            {"p": ak.Array([[4], []]), "q": ak.Array([[], [5, 6]])},
            counter_name={"p": "nx", "q": "ny"}.get,
        )
        assert len(reads) == 1

    with uproot.open(path) as f:
        assert _values(f["tree"], "p") == [[4], []]
        assert _values(f["tree"], "q") == [[], [5, 6]]


def test_reading_a_tree_doesnt_lay_it_out(tmp_path, monkeypatch):
    path = str(tmp_path / "lazy.root")
    _flat_tree(path, np.arange(2.0))

    calls = []
    original = uproot.writing._cascadetree.Tree._build_out

    def counting(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(uproot.writing._cascadetree.Tree, "_build_out", counting)
    with uproot.update(path) as f:
        assert f["tree"].num_entries == 2
        assert calls == []
        f["tree"].add_branches({"b": np.arange(2)})
        assert len(calls) == 1

    with uproot.open(path) as f:
        assert _values(f["tree"], "b") == [0, 1]


def test_add_branches_custom_counter_name_does_not_leak_to_extend(tmp_path):
    path = str(tmp_path / "counter_leak.root")
    _tree(path, {"x": "int32"}, {"x": np.array([1, 2, 3], dtype=np.int32)})

    with uproot.update(path) as f:
        tree = f["tree"]
        tree.add_branches({"x_jag": ak.Array(J)})
        tree.add_branches(
            {"y_jag": ak.Array([[10.0], [], [20.0, 30.0]])},
            counter_name=lambda n: f"{n}_cnt",
        )
        tree.extend(
            {
                "x": np.array([4], dtype=np.int32),
                "x_jag": ak.Array([[4.0]]),
                "y_jag": ak.Array([[40.0]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x", "nx_jag", "x_jag", "y_jag_cnt", "y_jag"]
        assert _values(tree, "x") == [1, 2, 3, 4]
        assert _values(tree, "x_jag") == [*J, [4.0]]
        assert _values(tree, "y_jag") == [[10.0], [], [20.0, 30.0], [40.0]]


@pytest.mark.parametrize(
    "data",
    [
        np.arange(6.0).reshape(3, 2),
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        ak.to_regular(ak.Array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])),
    ],
)
def test_add_fixed_size_branch(tmp_path, data):
    path = str(tmp_path / "fixed_size.root")
    _flat_tree(path, np.arange(3.0))

    with uproot.update(path) as f:
        f["tree"].add_branches({"m": data})
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.array([3.0]), "m": np.array([[6.0, 7.0]])})

    with uproot.open(path) as f:
        assert f["tree"]["m"].title == "m[2]/D"
        assert _values(f["tree"], "m") == [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ]


def _jagged_of_regular(offsets, content, size):
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array(offsets, dtype=np.int64)),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.array(content, dtype=np.float64)), size
            ),
        )
    )


def test_add_jagged_branch_of_fixed_size_lists(tmp_path):
    path = str(tmp_path / "jagged_of_regular.root")
    _flat_tree(path, np.arange(3.0))
    data = _jagged_of_regular([0, 1, 1, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)
    expected = [[[1.0, 2.0]], [], [[3.0, 4.0], [5.0, 6.0]]]

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": data})

    with uproot.open(path) as f:
        assert f["tree"]["j"].title == "j[nj][2]/D"
        assert _values(f["tree"], "j") == expected

    more = _jagged_of_regular([0, 2], [7.0, 8.0, 9.0, 10.0, 99.0, 99.0], 2)
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.array([3.0]), "j": more})

    with uproot.open(path) as f:
        assert _values(f["tree"], "j") == [*expected, [[7.0, 8.0], [9.0, 10.0]]]


@pytest.mark.parametrize("data", [[[], [], []], ([], [], []), ak.Array([[], [], []])])
def test_add_all_empty_lists_is_jagged(tmp_path, data):
    path = str(tmp_path / "all_empty.root")
    _flat_tree(path, np.arange(3.0))

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": data})
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.array([3.0]), "j": ak.Array([[1.0, 2.0]])})

    with uproot.open(path) as f:
        assert _values(f["tree"], "nj") == [0, 0, 0, 2]
        assert _values(f["tree"], "j") == [[], [], [], [1.0, 2.0]]


def test_add_branches_keeps_string_length(tmp_path):
    path = str(tmp_path / "strings.root")
    _tree(path, {"s": "string"}, {"s": np.array(["a", "abcdef", "ab"])})

    with uproot.open(path) as f:
        fLen = f["tree"]["s"].member("fLeaves")[0].member("fLen")
    assert fLen > 1

    with uproot.update(path) as f:
        f["tree"].add_branches({"a": np.arange(3.0)})

    with uproot.open(path) as f:
        assert f["tree"]["s"].member("fLeaves")[0].member("fLen") == fLen
        assert _values(f["tree"], "s") == ["a", "abcdef", "ab"]


def test_read_other_objects_while_updating(tmp_path):
    path = str(tmp_path / "histogram.root")
    with uproot.recreate(path) as f:
        f["h"] = np.histogram([1.0, 2.0, 2.0], bins=2, range=(0.0, 4.0))

    with uproot.update(path) as f:
        assert f["h"].values().tolist() == [1.0, 2.0]


def test_add_branches_uniform_multiple_baskets_raises(tmp_path):
    path = str(tmp_path / "uniform.root")
    _flat_tree(path, np.arange(3.0))
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.arange(3.0)})

    with uproot.update(path) as f:
        with pytest.raises(
            NotImplementedError, match="^add_branches for a TTree .* more than one"
        ):
            f["tree"].add_branches({"b": np.arange(6.0)})
        with pytest.raises(
            NotImplementedError, match="^extend with accept_new_fields=True for a"
        ):
            f["tree"].extend(
                {"a": np.array([3.0]), "b": np.array([1])}, accept_new_fields=True
            )


def test_add_branches_to_root_written_tree_in_two_sessions(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = str(tmp_path / "hzz.root")
    shutil.copyfile(orig_path, path)
    jagged = ak.Array(
        [[float(i), float(i + 1)] if i % 2 == 0 else [] for i in range(2421)]
    )
    second = ak.Array([[float(i)] if i % 2 == 0 else [] for i in range(2421)])

    with uproot.update(path) as f:
        f["events"].add_branches(
            {"new_jagged": jagged, "first": np.zeros(2421, dtype=np.float32)}
        )
    with uproot.update(path) as f:
        f["events"].add_branches({"second_jagged": second})

    with uproot.open(path) as f, uproot.open(orig_path) as orig:
        events = f["events"]
        assert ak.array_equal(events["new_jagged"].array(), jagged)
        assert _values(events, "nnew_jagged") == ak.num(jagged).tolist()
        assert _values(events, "first") == [0.0] * 2421
        assert ak.array_equal(events["second_jagged"].array(), second)
        assert _values(events, "nsecond_jagged") == ak.num(second).tolist()
        assert ak.array_equal(
            events["Muon_Px"].array(), orig["events"]["Muon_Px"].array()
        )


def test_add_branches_divergent_tree_is_independent_of_branch_order(tmp_path):
    """Its first branch has 5 baskets, unlike HZZ, whose first ones have 1."""
    path = str(tmp_path / "sample.root")
    shutil.copy(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"), path
    )

    with uproot.open(path) as f:
        before = f["sample"].arrays()
        num_entries = f["sample"].num_entries

    with uproot.update(path) as f:
        assert f["sample"]._cascading._has_divergent_baskets
        f["sample"].add_branches({"new": np.arange(num_entries, dtype=np.float64)})

    with uproot.open(path) as f:
        assert ak.array_equal(f["sample"].arrays()[before.fields], before)
        np.testing.assert_array_equal(
            f["sample"]["new"].array(library="np"), np.arange(num_entries)
        )


def test_add_branches_preserves_older_and_newer_cycles(tmp_path):
    """ROOT lists the newest cycle first, unlike the cycles Uproot makes."""
    path = str(tmp_path / "cycles.root")
    shutil.copy(skhep_testdata.data_path("uproot-issue-123a.root"), path)

    with uproot.open(path) as f:
        before = f["##Params;2"].arrays(library="np")
        older = f["##Params;1"].arrays(library="np")
        num_entries = f["##Params;2"].num_entries

    with uproot.update(path) as f:
        f["##Params"].add_branches({"new": np.arange(num_entries, dtype=np.float64)})
        f["##Params"].add_branches({"new2": np.arange(num_entries, dtype=np.float64)})

    with uproot.open(path) as f:
        assert set(f["##Params;2"].keys()) == {*before, "new", "new2"}
        assert list(f["##Params;1"].keys()) == list(older)
        after = f["##Params;2"].arrays(library="np")
        for name, value in before.items():
            np.testing.assert_array_equal(after[name], value)
        for name, value in f["##Params;1"].arrays(library="np").items():
            np.testing.assert_array_equal(older[name], value)


@pytest.mark.parametrize(
    "filename, treename, reason",
    [
        ("uproot-metadata-performance.root", "Events", "cluster ranges"),
        ("uproot-double32-float16.root", "tree", "packed branches"),
        ("string-example.root", "Refs", "different number of entries"),
        ("uproot-issue21.root", "nllscan", "embedded baskets"),
        # also has schema evolution rules in its list of streamers
        ("uproot-Event.root", "T", "TBranchElement"),
    ],
)
def test_tree_that_cannot_be_rewritten_is_refused(tmp_path, filename, treename, reason):
    """Rewriting these trees' metadata would make them partly or wholly unreadable."""
    path = str(tmp_path / filename)
    shutil.copy(skhep_testdata.data_path(filename), path)

    def snapshot(tree):
        return tree.keys(), [
            re.sub("0x[0-9a-f]+", "", repr(branch.interpretation))
            for branch in tree.values()
        ]

    with uproot.open(path) as f:
        before = snapshot(f[treename])
        if filename == "uproot-issue21.root":
            arrays = f[treename].arrays(library="np")

    with uproot.update(path) as f:
        tree = f[treename]
        with pytest.raises(NotImplementedError, match=reason):
            tree.add_branches({"new": np.zeros(tree.num_entries)})
        with pytest.raises(NotImplementedError, match=reason):
            tree.extend({})

    with uproot.open(path) as f:
        assert snapshot(f[treename]) == before
        if filename == "uproot-issue21.root":
            for name, value in f[treename].arrays(library="np").items():
                np.testing.assert_array_equal(value, arrays[name])


def _root_tree_with_counter(path, ROOT, counter, leaf, after=False):
    rf = ROOT.TFile(path, "RECREATE")
    rt = ROOT.TTree("tree", "")
    n = np.zeros(1, dtype=np.int32)
    x = np.zeros(10, dtype=np.float64)
    counter_branch = rt.Branch(counter, n, f"{leaf}/I")
    rt.Branch("x", x, f"x[{leaf}]/D")
    for row in ([1.0, 2.0], [], [3.0]):
        n[0] = len(row)
        x[: len(row)] = row
        rt.Fill()
    if after:
        branches = rt.GetListOfBranches()
        branches.Remove(counter_branch)
        branches.Compress()
        branches.Add(counter_branch)
    rt.Write()
    rf.Close()


def test_root_tree_whose_counter_has_another_leaf_name(tmp_path):
    """Branch("nx", &n, "cx/I") is valid ROOT: a leaf isn't named after its branch."""
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "leaf_names.root")
    _root_tree_with_counter(path, ROOT, "nx", "cx")

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"y": ak.Array([[4.0, 5.0], [], [6.0]]), "z": np.arange(3.0)},
            counter_name=lambda counted: "nx",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree["nx"].member("fLeaves")[0].member("fName") == "cx"
        assert tree["x"].member("fLeaves")[0].member("fLeafCount") is not None
        assert _values(tree, "x") == [[1.0, 2.0], [], [3.0]]
        assert tree["y"].title == "y[cx]/D"
        assert tree["y"].member("fLeaves")[0].member("fTitle") == "y[cx]"

    rf = ROOT.TFile(path)
    rt = rf.Get("tree")
    assert rt.GetLeaf("y").GetLeafCount().GetName() == "cx"
    assert rt.Draw("y", "", "goff") == 3
    assert [rt.GetV1()[i] for i in range(3)] == [4.0, 5.0, 6.0]
    rf.Close()


def test_add_branches_on_older_cycle_keeps_newer_cycle(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "older_cycle.root")

    rf = ROOT.TFile(path, "RECREATE")
    rt = ROOT.TTree("tree", "")
    x = np.zeros(1, dtype=np.float64)
    rt.Branch("x", x, "x/D")
    rt.Fill()
    rt.Write()
    rt.Write()
    rf.Close()

    with uproot.update(path) as f:
        newer = f._cascading.data.get_key("tree", 2).seek_location
        f["tree;1"].add_branches({"new": np.zeros(1)})
        assert f._cascading.data.get_key("tree", 2).seek_location == newer

    with uproot.open(path) as f:
        assert f["tree;2"].keys() == ["x"]
        assert f["tree;1"].keys() == ["x", "new"]


def _root_tree_with_metadata(path, ROOT, setup):
    rf = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("tree", "")
    friend = ROOT.TTree("friend", "")
    x = np.zeros(1, dtype=np.int32)
    tree.Branch("x", x, "x/I")
    friend.Branch("x", x, "x/I")
    for i in range(3):
        x[0] = i
        tree.Fill()
        friend.Fill()
    # the tree deletes its user info
    info = ROOT.TNamed("info", "value")
    ROOT.SetOwnership(info, False)
    exec(setup, {"tree": tree, "friend": friend, "info": info})
    tree.Write()
    friend.Write()
    rf.Close()


def _root_tree_without_baskets(path, ROOT):
    rf = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("tree", "")
    tree.SetEntries(3)
    tree.Write()
    rf.Close()


@pytest.mark.parametrize(
    "make, reason",
    [
        pytest.param(
            lambda p, R: _root_tree_with_metadata(p, R, "tree.SetAlias('t', 'x * 2')"),
            "aliases",
            id="aliases",
        ),
        pytest.param(
            lambda p, R: _root_tree_with_metadata(p, R, "tree.BuildIndex('x')"),
            "index",
            id="index",
        ),
        pytest.param(
            lambda p, R: _root_tree_with_metadata(p, R, "tree.AddFriend(friend)"),
            "friends",
            id="friends",
        ),
        pytest.param(
            lambda p, R: _root_tree_with_metadata(p, R, "tree.GetUserInfo().Add(info)"),
            "user info",
            id="user-info",
        ),
        pytest.param(
            lambda p, R: _root_tree_with_metadata(p, R, "tree.BranchRef()"),
            "TRefs",
            id="trefs",
        ),
        pytest.param(
            lambda p, R: _root_tree_with_counter(p, R, "n", "n", after=True),
            "counters after the branches",
            id="counter-after-its-branch",
        ),
        pytest.param(
            _root_tree_without_baskets, "entries but no baskets", id="no-baskets"
        ),
    ],
)
def test_root_tree_that_cannot_be_rewritten_is_refused(tmp_path, make, reason):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "refused.root")
    make(path, ROOT)

    with uproot.open(path) as f:
        keys = f["tree"].keys()
        before = {name: _values(f["tree"], name) for name in keys}

    with uproot.update(path) as f:
        with pytest.raises(NotImplementedError, match=reason):
            f["tree"].add_branches({"y": np.arange(3.0)})
        if keys:
            with pytest.raises(NotImplementedError, match=reason):
                f["tree"].extend({})

    with uproot.open(path) as f:
        assert f["tree"].keys() == keys
        assert {name: _values(f["tree"], name) for name in keys} == before
