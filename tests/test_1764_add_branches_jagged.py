# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import io
import os
import shutil

import numpy as np
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def test_add_jagged_branch_to_root_written_tree(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = os.path.join(tmp_path, "hzz_jagged.root")
    shutil.copyfile(orig_path, path)

    data = ak.Array(
        [[float(i), float(i + 1)] if i % 2 == 0 else [] for i in range(2421)]
    )

    with uproot.update(path) as f:
        events = f["events"]
        events.add_branches({"new_jagged": data})

    with uproot.open(path) as f:
        events = f["events"]
        assert "new_jagged" in events
        assert "nnew_jagged" in events
        assert ak.all(events["new_jagged"].array() == data)
        assert np.array_equal(
            events["nnew_jagged"].array(library="np"),
            ak.num(data, axis=1).to_numpy(),
        )
        orig_muon_px = uproot.open(orig_path)["events"]["Muon_Px"].array()
        assert ak.all(events["Muon_Px"].array() == orig_muon_px)


def test_add_jagged_branch_uproot_written_tree(tmp_path):
    path = os.path.join(tmp_path, "test.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([10, 20, 30], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": ak.Array([[1.0], [], [2.0, 3.0]])})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([10, 20, 30]))
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))


def test_add_jagged_branch_with_custom_counter_name(tmp_path):
    path = os.path.join(tmp_path, "test_custom_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name=lambda b: f"num_{b}",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert "num_j" in tree
        assert np.array_equal(tree["num_j"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))


def test_add_jagged_branch_with_explicit_matching_counter(tmp_path):
    path = os.path.join(tmp_path, "test_explicit.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "nj": np.array([1, 0, 2], dtype=np.int32),
                "j": ak.Array([[10.0], [], [20.0, 30.0]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[10.0], [], [20.0, 30.0]]))


def test_add_jagged_branch_with_explicit_disagreeing_counter_raises(tmp_path):
    path = os.path.join(tmp_path, "test_disagree.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(ValueError, match="disagree"):
        f["tree"].add_branches(
            {
                "nj": np.array([9, 9, 9], dtype=np.int32),
                "j": ak.Array([[10.0], [], [20.0, 30.0]]),
            }
        )


def test_add_jagged_branch_wrong_length_raises(tmp_path):
    path = os.path.join(tmp_path, "test_wrong_len.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(ValueError, match="all new branches must match the tree length"),
    ):
        f["tree"].add_branches({"j": ak.Array([[1.0], [2.0]])})


def test_add_multiple_jagged_and_flat_branches(tmp_path):
    path = os.path.join(tmp_path, "test_mult.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "y": np.array([4.0, 5.0, 6.0], dtype=np.float64),
                "j": ak.Array([[1.5], [], [2.5, 3.5]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3]))
        assert np.array_equal(tree["y"].array(library="np"), np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.5], [], [2.5, 3.5]]))


def test_empty_tree_add_jagged_branch_then_extend(tmp_path):
    path = os.path.join(tmp_path, "test_empty_jagged.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"a": np.int64})
    with uproot.update(path) as f:
        f["t"].add_branches({"j": ak.Array([[1.0]])[:0]})
        f["t"].extend(
            {
                "a": np.array([1, 2], dtype=np.int64),
                "j": ak.Array([[10.0, 20.0], [30.0]]),
            }
        )
    with uproot.open(path) as f:
        tree = f["t"]
        assert np.array_equal(tree["a"].array(library="np"), np.array([1, 2]))
        assert np.array_equal(tree["nj"].array(library="np"), np.array([2, 1]))
        assert ak.all(tree["j"].array() == ak.Array([[10.0, 20.0], [30.0]]))


def test_add_branches_sequential_sessions(tmp_path):
    orig_path = skhep_testdata.data_path("uproot-HZZ.root")
    path = os.path.join(tmp_path, "hzz_seq.root")
    shutil.copyfile(orig_path, path)

    with uproot.update(path) as f:
        f["events"].add_branches({"first": np.zeros(2421, dtype=np.float32)})

    with uproot.update(path) as f:
        f["events"].add_branches(
            {
                "second_jagged": ak.Array(
                    [[float(i)] if i % 2 == 0 else [] for i in range(2421)]
                )
            }
        )

    with uproot.open(path) as f:
        events = f["events"]
        assert "first" in events
        assert "second_jagged" in events
        assert "nsecond_jagged" in events
        assert np.array_equal(
            events["first"].array(library="np"), np.zeros(2421, dtype=np.float32)
        )
        assert np.array_equal(
            events["nsecond_jagged"].array(library="np"),
            [1 if i % 2 == 0 else 0 for i in range(2421)],
        )
        assert ak.all(
            events["Muon_Px"].array()
            == uproot.open(orig_path)["events"]["Muon_Px"].array()
        )


def test_add_jagged_branch_with_string_counter_name_raises(tmp_path):
    path = os.path.join(tmp_path, "test_str_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="counter_name must be a callable or None"),
    ):
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name="count_{}",
        )

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="counter_name must be a callable or None"),
    ):
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name="len_",
        )


def test_add_jagged_branch_invalid_counter_name_raises(tmp_path):
    path = os.path.join(tmp_path, "test_bad_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(TypeError, match="counter_name"):
        f["tree"].add_branches(
            {"j": ak.Array([[1.0], [], [2.0, 3.0]])},
            counter_name=12345,
        )


def test_add_jagged_branch_counter_collides_with_normal_branch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_collision.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32", "nj": "float32"})
        f["tree"].extend(
            {
                "x": np.array([1, 2, 3], dtype="int32"),
                "nj": np.array([1.1, 2.2, 3.3], dtype="float32"),
            }
        )

    with uproot.update(path) as f, pytest.raises(ValueError, match="collides"):
        f["tree"].add_branches({"j": ak.Array([[1.0], [], [2.0, 3.0]])})


def test_add_multiple_jagged_branches_sharing_counter(tmp_path):
    path = os.path.join(tmp_path, "test_shared_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "j1": ak.Array([[1.0], [], [2.0, 3.0]]),
                "j2": ak.Array([[10.0], [], [20.0, 30.0]]),
            },
            counter_name=lambda _: "n_particles",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert "n_particles" in tree
        assert np.array_equal(
            tree["n_particles"].array(library="np"), np.array([1, 0, 2])
        )
        assert ak.all(tree["j1"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))
        assert ak.all(tree["j2"].array() == ak.Array([[10.0], [], [20.0, 30.0]]))


def test_add_multiple_jagged_branches_sharing_counter_disagree_raises(tmp_path):
    path = os.path.join(tmp_path, "test_shared_disagree.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(ValueError, match="disagrees"):
        f["tree"].add_branches(
            {
                "j1": ak.Array([[1.0], [], [2.0, 3.0]]),
                "j2": ak.Array([[10.0, 20.0], [], [30.0]]),
            },
            counter_name=lambda _: "n_particles",
        )


def test_add_jagged_branch_sliced_indexed_array(tmp_path):
    path = os.path.join(tmp_path, "test_sliced_indexed.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2], dtype="int32")})

    big_arr = ak.Array([[1.0, 2.0], [], [3.0], [4.0, 5.0]])[::2]
    with uproot.update(path) as f:
        f["tree"].add_branches({"j": big_arr})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert ak.all(tree["j"].array() == big_arr)
        assert np.array_equal(tree["nj"].array(library="np"), np.array([2, 1]))


def test_add_branch_flat_object_dtype_raises(tmp_path):
    path = os.path.join(tmp_path, "test_obj_dtype.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f, pytest.raises(TypeError, match="object dtype"):
        f["tree"].add_branches({"bad": np.array(["a", object(), 123], dtype=object)})


def test_add_nested_jagged_branch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_nested.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="cannot write Awkward Array type to ROOT file"),
    ):
        f["tree"].add_branches({"j": ak.Array([[[1.0]], [], [[2.0], [3.0]]])})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3]))


def test_add_jagged_string_branch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_jagged_str.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with (
        uproot.update(path) as f,
        pytest.raises(TypeError, match="cannot write Awkward Array type to ROOT file"),
    ):
        f["tree"].add_branches({"j": ak.Array([["a"], [], ["b", "c"]])})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3]))


def test_add_branches_failure_does_not_corrupt_casc(tmp_path):
    path = os.path.join(tmp_path, "test_rollback.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        tree = f["tree"]
        with pytest.raises(TypeError, match="cannot write NumPy dtype"):
            tree.add_branches(
                {
                    "j": ak.Array([[1.0], [], [2.0]]),
                    "s": np.array(["x", "y", "z"]),
                }
            )
        # casc must be cleanly rolled back: phantom branches must not linger
        tree.extend({"x": np.array([4, 5], dtype="int32")})

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3, 4, 5]))


def test_add_jagged_branch_shares_existing_counter_mismatch_raises(tmp_path):
    path = os.path.join(tmp_path, "test_existing_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches({"x_jag": ak.Array([[1], [], [2, 3]])})

    # sharing existing counter nx_jag with mismatching lengths must raise ValueError
    with (
        uproot.update(path) as f,
        pytest.raises(
            ValueError, match="disagrees with the existing counter branch's lengths"
        ),
    ):
        f["tree"].add_branches(
            {"y_jag": ak.Array([[1, 2, 3], [4], []])},
            counter_name=lambda n: "nx_jag",
        )

    # sharing existing counter nx_jag with matching lengths must succeed
    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"y_jag": ak.Array([[10.0], [], [20.0, 30.0]])},
            counter_name=lambda n: "nx_jag",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert "y_jag" in tree
        assert ak.all(tree["y_jag"].array() == ak.Array([[10.0], [], [20.0, 30.0]]))


def test_add_branches_custom_counter_name_does_not_leak_to_extend(tmp_path):
    path = os.path.join(tmp_path, "test_counter_leak.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        tree = f["tree"]
        tree.add_branches({"x_jag": ak.Array([[1.0], [], [2.0, 3.0]])})
        tree.add_branches(
            {"y_jag": ak.Array([[10.0], [], [20.0, 30.0]])},
            counter_name=lambda n: f"{n}_cnt",
        )
        # extending existing jagged branch must still work with its own counter
        tree.extend(
            {
                "x": np.array([4], dtype="int32"),
                "x_jag": ak.Array([[4.0]]),
                "y_jag": ak.Array([[40.0]]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert np.array_equal(tree["x"].array(library="np"), np.array([1, 2, 3, 4]))
        assert ak.all(tree["x_jag"].array() == ak.Array([[1.0], [], [2.0, 3.0], [4.0]]))
        assert ak.all(
            tree["y_jag"].array() == ak.Array([[10.0], [], [20.0, 30.0], [40.0]])
        )


def test_add_jagged_branch_explicit_float_counter_converted_to_int32(tmp_path):
    path = os.path.join(tmp_path, "test_float_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2, 3], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "j": ak.Array([[1.0], [], [2.0, 3.0]]),
                "nj": np.array([1.0, 0.0, 2.0]),
            }
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree["nj"].interpretation.to_dtype == np.dtype("int32")
        assert tree["nj"].array(library="np").dtype == np.dtype("int32")
        assert tree["nj"].title == "nj/I"
        assert np.array_equal(tree["nj"].array(library="np"), np.array([1, 0, 2]))
        assert ak.all(tree["j"].array() == ak.Array([[1.0], [], [2.0, 3.0]]))


def test_add_multiple_explicit_counters_deterministic_order(tmp_path):
    path = os.path.join(tmp_path, "test_det_order.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "int32"})
        f["tree"].extend({"x": np.array([1, 2], dtype="int32")})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "c1": np.array([1, 2], dtype=np.int32),
                "j1": ak.Array([[1.0], [2.0, 3.0]]),
                "c2": np.array([2, 1], dtype=np.int32),
                "j2": ak.Array([[10.0, 20.0], [30.0]]),
            },
            counter_name=lambda b: "c1" if b == "j1" else "c2",
        )

    with uproot.open(path) as f:
        tree = f["tree"]
        assert tree.keys() == ["x", "c1", "c2", "j1", "j2"]


def _flat_tree(path, data):
    with uproot.recreate(path) as f:
        f.mktree("tree", {"a": np.float64})
        f["tree"].extend({"a": data})


def _jagged_tree(path, data):
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "var * float64"})
        f["tree"].extend({"x": data})


def test_add_branches_embedded_baskets_raises(tmp_path):
    """Rewriting a tree's metadata would lose the data in its embedded baskets."""
    path = str(tmp_path / "embedded.root")
    shutil.copy(skhep_testdata.data_path("uproot-issue21.root"), path)

    with uproot.open(path) as f:
        before = f["nllscan"].arrays(library="np")
        num_entries = f["nllscan"].num_entries

    with uproot.update(path) as f:
        tree = f["nllscan"]
        with pytest.raises(NotImplementedError, match="embedded baskets"):
            tree.add_branches({"new": np.zeros(num_entries)})
        with pytest.raises(NotImplementedError, match="embedded baskets"):
            tree.extend({name: value[:1] for name, value in before.items()})

    with uproot.open(path) as f:
        assert f["nllscan"].keys() == list(before)
        after = f["nllscan"].arrays(library="np")
        for name, value in before.items():
            np.testing.assert_array_equal(after[name], value)


def test_add_jagged_branch_existing_counter_keeps_file_object_open():
    data = ak.Array([[1.0], [], [2.0, 3.0]])
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
        assert f["tree"]["z"].array().tolist() == [[3.0], [], [6.0, 9.0]]


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


def test_add_branches_basket_failure_leaves_file_readable(tmp_path, monkeypatch):
    path = str(tmp_path / "basket_failure.root")
    _flat_tree(path, np.arange(3.0))
    data = ak.Array([[1.0], [], [2.0, 3.0]])

    def fail(*args, **kwargs):
        raise ValueError("too large to fit in a TBasket")

    with uproot.update(path) as f:
        tree = f["tree"]
        casc = tree._cascading
        fTotBytes = casc._metadata["fTotBytes"]
        with monkeypatch.context() as context:
            context.setattr(casc, "write_jagged_basket", fail)
            with pytest.raises(ValueError, match="too large"):
                tree.add_branches({"j": data})
        assert [datum["fName"] for datum in casc._branch_data] == ["a"]
        assert casc._metadata["fTotBytes"] == fTotBytes

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["a"]
        assert f["tree"]["a"].array(library="np").tolist() == [0.0, 1.0, 2.0]

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": data})

    with uproot.open(path) as f:
        assert f["tree"]["j"].array().tolist() == data.tolist()


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
        assert f["tree"]["m"].array(library="np").tolist() == [
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
        assert f["tree"]["j"].array().tolist() == expected

    more = _jagged_of_regular([0, 2], [7.0, 8.0, 9.0, 10.0, 99.0, 99.0], 2)
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.array([3.0]), "j": more})

    with uproot.open(path) as f:
        assert f["tree"]["j"].array().tolist() == [
            *expected,
            [[7.0, 8.0], [9.0, 10.0]],
        ]


def test_add_jagged_branch_empty_basket_keeps_counter(tmp_path):
    path = str(tmp_path / "zero_entries.root")
    _flat_tree(path, np.array([], dtype=np.float64))

    with uproot.update(path) as f:
        assert f["tree"].num_baskets == 0
        f["tree"].add_branches({"j": ak.Array([[1.0]])[:0]})

    with uproot.update(path) as f:
        casc = f["tree"]._cascading
        assert casc._branch_data[casc._branch_lookup["j"]]["counter"] is not None
        f["tree"].extend({"a": np.array([1.0]), "j": ak.Array([[1.0, 2.0]])})

    with uproot.open(path) as f:
        assert f["tree"]["j"].title == "j[nj]/D"
        assert f["tree"]["j"].array().tolist() == [[1.0, 2.0]]


def test_extend_after_empty_jagged_extend_keeps_counter(tmp_path):
    path = str(tmp_path / "empty_extend.root")
    _jagged_tree(path, ak.Array([[1.0]]))
    with uproot.update(path) as f:
        f["tree"].extend({"x": ak.Array([[1.0]])[:0]})

    with uproot.update(path) as f:
        f["tree"].extend({"x": ak.Array([[2.0, 3.0]])})

    with uproot.open(path) as f:
        assert f["tree"]["x"].array().tolist() == [[1.0], [2.0, 3.0]]


@pytest.mark.parametrize("field", [ak.Array([[1.0], [2.0, 3.0]]), [[1.0], [2.0, 3.0]]])
def test_extend_accept_new_fields_jagged(tmp_path, field):
    path = str(tmp_path / "new_jagged.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        f["tree"].extend(
            {"a": np.array([2.0, 3.0]), "j": field}, accept_new_fields=True
        )

    with uproot.open(path) as f:
        assert f["tree"]["nj"].array(library="np").tolist() == [0, 0, 1, 2]
        assert f["tree"]["j"].array().tolist() == [[], [], [1.0], [2.0, 3.0]]


def test_extend_accept_new_fields_jagged_keeps_dtype(tmp_path):
    path = str(tmp_path / "new_jagged_dtype.root")
    _flat_tree(path, np.arange(1.0))

    with uproot.update(path) as f:
        f["tree"].extend(
            {"a": np.array([1.0]), "j": ak.Array([[1, 2]])}, accept_new_fields=True
        )

    with uproot.open(path) as f:
        assert f["tree"]["j"].title == "j[nj]/L"
        assert f["tree"]["j"].array().tolist() == [[], [1, 2]]


def test_extend_accept_new_fields_nested_jagged_raises(tmp_path):
    path = str(tmp_path / "new_nested.root")
    _flat_tree(path, np.arange(1.0))

    with uproot.update(path) as f:
        with pytest.raises(TypeError, match="cannot write Awkward Array type"):
            f["tree"].extend(
                {"a": np.array([1.0]), "j": ak.Array([[[1.0]]])},
                accept_new_fields=True,
            )

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["a"]


def test_extend_accept_new_fields_fixed_size(tmp_path):
    path = str(tmp_path / "new_fixed_size.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        f["tree"].extend(
            {"a": np.array([2.0]), "m": np.array([[1.0, 2.0]])},
            accept_new_fields=True,
        )

    with uproot.open(path) as f:
        assert f["tree"]["m"].array(library="np").tolist() == [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 2.0],
        ]


def test_add_branches_releases_old_tree_metadata(tmp_path):
    path = str(tmp_path / "leak.root")
    _flat_tree(path, np.arange(3.0))

    with uproot.update(path) as f:
        key = f["tree"]._cascading._key
        start = key.seek_location
        stop = start + key.num_bytes + key.compressed_bytes
        f["tree"].add_branches({"j": ak.Array([[1.0], [], [2.0, 3.0]])})

    with uproot.update(path) as f:
        slices = f._cascading.freesegments._data.slices
        assert any(free[0] <= start and stop <= free[1] for free in slices)

    with uproot.open(path) as f:
        assert f["tree"]["j"].array().tolist() == [[1.0], [], [2.0, 3.0]]


def test_add_jagged_branch_sets_counter_maximum(tmp_path):
    """ROOT sizes its read buffer for a jagged branch from fMaximum of its counter."""
    path = str(tmp_path / "maximum.root")
    _flat_tree(path, np.arange(3.0))

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": ak.Array([[1.0], [], [2.0, 3.0]])})

    with uproot.open(path) as f:
        assert f["tree"]["nj"].member("fLeaves")[0].member("fMaximum") == 2


def test_add_branches_keeps_string_length(tmp_path):
    path = str(tmp_path / "strings.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"s": "string"})
        f["tree"].extend({"s": np.array(["a", "abcdef", "ab"])})

    with uproot.open(path) as f:
        fLen = f["tree"]["s"].member("fLeaves")[0].member("fLen")
    assert fLen > 1

    with uproot.update(path) as f:
        f["tree"].add_branches({"a": np.arange(3.0)})

    with uproot.open(path) as f:
        assert f["tree"]["s"].member("fLeaves")[0].member("fLen") == fLen
        assert f["tree"]["s"].array(library="np").tolist() == ["a", "abcdef", "ab"]


def test_add_branches_repeatedly_reuses_file_space(tmp_path):
    path = str(tmp_path / "repeated.root")
    _flat_tree(path, np.arange(3.0))
    data = ak.Array([[1.0], [], [2.0, 3.0]])

    for i in range(6):
        with uproot.update(path) as f:
            f["tree"].add_branches({f"j{i}": data})

    with uproot.open(path) as f:
        for i in range(6):
            assert f["tree"][f"j{i}"].array().tolist() == data.tolist()


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


def test_add_branches_uniform_multiple_baskets_raises(tmp_path):
    path = str(tmp_path / "uniform.root")
    _flat_tree(path, np.arange(3.0))
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.arange(3.0)})

    with uproot.update(path) as f:
        with pytest.raises(NotImplementedError, match="more than one basket"):
            f["tree"].add_branches({"b": np.arange(6.0)})


@pytest.mark.parametrize(
    "filename, treename, reason",
    [
        ("uproot-metadata-performance.root", "Events", "cluster ranges"),
        ("uproot-double32-float16.root", "tree", "packed branches"),
        ("string-example.root", "Refs", "different number of entries"),
        ("uproot-issue21.root", "nllscan", "embedded baskets"),
    ],
)
def test_tree_that_cannot_be_rewritten_is_refused(tmp_path, filename, treename, reason):
    """Rewriting these trees' metadata would make them partly or wholly unreadable."""
    path = str(tmp_path / filename)
    shutil.copy(skhep_testdata.data_path(filename), path)

    with uproot.open(path) as f:
        keys = f[treename].keys()
        interpretations = [
            repr(branch.interpretation) for branch in f[treename].values()
        ]

    with uproot.update(path) as f:
        tree = f[treename]
        with pytest.raises(NotImplementedError, match=reason):
            tree.add_branches({"new": np.zeros(tree.num_entries)})
        with pytest.raises(NotImplementedError, match=reason):
            tree.extend({})

    with uproot.open(path) as f:
        assert f[treename].keys() == keys
        assert [
            repr(branch.interpretation) for branch in f[treename].values()
        ] == interpretations


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


@pytest.mark.parametrize("data", [[[], [], []], ([], [], []), ak.Array([[], [], []])])
def test_add_all_empty_lists_is_jagged(tmp_path, data):
    path = str(tmp_path / "all_empty.root")
    _flat_tree(path, np.arange(3.0))

    with uproot.update(path) as f:
        f["tree"].add_branches({"j": data})
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.array([3.0]), "j": ak.Array([[1.0, 2.0]])})

    with uproot.open(path) as f:
        assert f["tree"]["nj"].array(library="np").tolist() == [0, 0, 0, 2]
        assert f["tree"]["j"].array().tolist() == [[], [], [], [1.0, 2.0]]


def test_add_empty_fixed_size_dimension_raises(tmp_path):
    path = str(tmp_path / "empty_dimension.root")
    _flat_tree(path, np.arange(3.0))

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="empty fixed-size dimension"):
            f["tree"].add_branches({"m": np.empty((3, 0))})

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["a"]


def test_extend_accept_new_fields_all_empty_lists(tmp_path):
    path = str(tmp_path / "new_all_empty.root")
    _flat_tree(path, np.arange(1.0))

    with uproot.update(path) as f:
        f["tree"].extend(
            {"a": np.array([1.0, 2.0]), "j": [[], []]}, accept_new_fields=True
        )

    with uproot.open(path) as f:
        assert f["tree"]["j"].array().tolist() == [[], [], []]


@pytest.mark.parametrize(
    "data, message",
    [
        (
            {
                "a": np.array([1.0, 2.0]),
                "j": ak.Array([[1.0], [], [2.0]]),
            },
            "same number of entries",
        ),
        (
            {
                "a": np.array([1.0, 2.0]),
                "j": ak.Array([[1.0], [2.0]]),
                "nj": np.array([5, 5]),
            },
            "they disagree",
        ),
    ],
)
def test_rejected_accept_new_fields_leaves_tree_alone(tmp_path, data, message):
    path = str(tmp_path / "rejected.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match=message):
            f["tree"].extend(data, accept_new_fields=True)
        f["tree"].extend({"a": np.array([2.0])})

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["a"]
        assert f["tree"]["a"].array(library="np").tolist() == [0.0, 1.0, 2.0]


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


def test_add_branches_preserves_leaf_names(tmp_path):
    """Branch("nMuon", &n, "n/I") is valid ROOT: a leaf isn't named after its branch."""
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "leaf_names.root")

    rf = ROOT.TFile(path, "RECREATE")
    rt = ROOT.TTree("tree", "")
    n = np.zeros(1, dtype=np.int32)
    x = np.zeros(10, dtype=np.float64)
    rt.Branch("nx", n, "cx/I")
    rt.Branch("x", x, "x[cx]/D")
    for row in ([1.0, 2.0], [], [3.0]):
        n[0] = len(row)
        x[: len(row)] = row
        rt.Fill()
    rt.Write()
    rf.Close()

    with uproot.update(path) as f:
        f["tree"].add_branches({"y": np.arange(3.0)})

    with uproot.open(path) as f:
        assert f["tree"]["nx"].member("fLeaves")[0].member("fName") == "cx"
        assert f["tree"]["x"].member("fLeaves")[0].member("fLeafCount") is not None
        assert f["tree"]["x"].array().tolist() == [[1.0, 2.0], [], [3.0]]


@pytest.mark.parametrize(
    "new_field",
    [np.arange(2), ak.Array([[1.0], [2.0, 3.0]])],
)
def test_rejected_accept_new_fields_checks_existing_fields_first(tmp_path, new_field):
    path = str(tmp_path / "rejected_shape.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"a": np.float32})
        f["tree"].extend({"a": np.arange(3, dtype=np.float32)})

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="consistent shape"):
            f["tree"].extend(
                {"a": np.ones((2, 3), dtype=np.float32), "b": new_field},
                accept_new_fields=True,
            )
        assert list(f["tree"]._cascading._branch_lookup) == ["a"]

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["a"]
        assert f["tree"]["a"].array(library="np").tolist() == [0.0, 1.0, 2.0]


def test_add_branches_failure_while_rewriting_keeps_new_branches(tmp_path, monkeypatch):
    path = str(tmp_path / "rewrite_failure.root")
    _flat_tree(path, np.arange(2.0))

    original = uproot.writing._cascadetree.Tree.write_updates
    with uproot.update(path) as f:
        tree = f["tree"]
        with monkeypatch.context() as m:

            def fail(self, sink):
                raise OSError("disk full")

            m.setattr(uproot.writing._cascadetree.Tree, "write_updates", fail)
            with pytest.raises(OSError, match="disk full"):
                tree.add_branches({"j": ak.Array([[1.0], [2.0, 3.0]])})

        # the TTree metadata on disk lists the new branches by now, and so does
        # the tree here, so that the next write agrees with it
        assert list(tree._cascading._branch_lookup) == ["a", "nj", "j"]
        assert uproot.writing._cascadetree.Tree.write_updates is original
        tree.extend({"a": np.array([2.0]), "j": ak.Array([[4.0]]), "nj": np.array([1])})

    with uproot.open(path) as f:
        assert f["tree"]["a"].array(library="np").tolist() == [0.0, 1.0, 2.0]
        assert f["tree"]["j"].array().tolist() == [[1.0], [2.0, 3.0], [4.0]]


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
        assert f["tree"]["b"].array(library="np").tolist() == [0, 1]


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
        assert f["tree"]["n"].array(library="np").tolist() == [1, 2, 1, 0]
        assert f["tree"]["k"].array().tolist() == [[0], [0, 0], [5], []]
        assert f["tree"]["m"].array().tolist() == [[0.0], [0.0, 0.0], [6.0], []]


def test_add_jagged_branch_with_existing_counter_to_empty_tree(tmp_path):
    path = str(tmp_path / "empty_existing_counter.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "var * float64"})

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"y": ak.Array([[1]])[:0]}, counter_name=lambda counted: "nx"
        )
        f["tree"].extend({"x": ak.Array([[1.0, 2.0]]), "y": ak.Array([[3, 4]])})

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["nx", "x", "y"]
        assert f["tree"]["y"].array().tolist() == [[3, 4]]


def test_add_jagged_branches_sharing_an_explicit_counter(tmp_path):
    path = str(tmp_path / "explicit_shared_counter.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {
                "p": ak.Array([[1.0], [2.0, 3.0]]),
                "q": ak.Array([[4], [5, 6]]),
                "n": np.array([1, 2]),
            },
            counter_name=lambda counted: "n",
        )

    with uproot.open(path) as f:
        assert f["tree"]["p"].array().tolist() == [[1.0], [2.0, 3.0]]
        assert f["tree"]["q"].array().tolist() == [[4], [5, 6]]
        assert f["tree"]["n"].array(library="np").tolist() == [1, 2]


@pytest.mark.parametrize(
    "data", [[[[], []], [[], []]], np.zeros((2, 2, 0)), np.zeros((2, 0))]
)
def test_add_nested_empty_lists_raises_like_numpy(tmp_path, data):
    path = str(tmp_path / "nested_empty.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="empty fixed-size dimension"):
            f["tree"].add_branches({"b": data})


def test_add_jagged_branch_names_the_counters_leaf(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "counter_leaf_name.root")

    rf = ROOT.TFile(path, "RECREATE")
    rt = ROOT.TTree("tree", "")
    n = np.zeros(1, dtype=np.int32)
    x = np.zeros(10, dtype=np.float64)
    rt.Branch("nx", n, "cx/I")
    rt.Branch("x", x, "x[cx]/D")
    for row in ([1.0, 2.0], [], [3.0]):
        n[0] = len(row)
        x[: len(row)] = row
        rt.Fill()
    rt.Write()
    rf.Close()

    with uproot.update(path) as f:
        f["tree"].add_branches(
            {"y": ak.Array([[4.0, 5.0], [], [6.0]])},
            counter_name=lambda counted: "nx",
        )

    with uproot.open(path) as f:
        assert f["tree"]["y"].title == "y[cx]/D"
        assert f["tree"]["y"].member("fLeaves")[0].member("fTitle") == "y[cx]"

    rf = ROOT.TFile(path)
    rt = rf.Get("tree")
    assert rt.GetLeaf("y").GetLeafCount().GetName() == "cx"
    assert rt.Draw("y", "", "goff") == 3
    assert [rt.GetV1()[i] for i in range(3)] == [4.0, 5.0, 6.0]
    rf.Close()


@pytest.mark.parametrize(
    "setup, reason",
    [
        ("tree.SetAlias('twice', 'x * 2')", "aliases"),
        ("tree.BuildIndex('x')", "index"),
        ("tree.AddFriend(friend)", "friends"),
        ("tree.GetUserInfo().Add(info)", "user info"),
        ("tree.BranchRef()", "TRefs"),
    ],
)
def test_root_tree_metadata_that_cannot_be_rewritten_is_refused(
    tmp_path, setup, reason
):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "metadata.root")

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

    with uproot.update(path) as f:
        with pytest.raises(NotImplementedError, match=reason):
            f["tree"].add_branches({"y": np.arange(3)})
        with pytest.raises(NotImplementedError, match=reason):
            f["tree"].extend({"x": np.arange(3, dtype=np.int32)})

    with uproot.open(path) as f:
        assert f["tree"]["x"].array(library="np").tolist() == [0, 1, 2]


def test_update_skips_schema_rules_in_the_list_of_streamers(tmp_path):
    path = str(tmp_path / "event.root")
    shutil.copyfile(skhep_testdata.data_path("uproot-Event.root"), path)

    with uproot.update(path) as f:
        tree = f["T"]
        assert tree.num_entries == 1000
        with pytest.raises(NotImplementedError, match="TBranchElement"):
            tree.add_branches({"y": np.zeros(1000)})


def test_add_existing_branch_raises(tmp_path):
    path = str(tmp_path / "existing.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="already exists"):
            f["tree"].add_branches({"a": np.arange(2.0)})


def test_add_jagged_branch_of_unsupported_type_raises(tmp_path):
    path = str(tmp_path / "option.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        with pytest.raises(TypeError, match="cannot write Awkward Array type"):
            f["tree"].add_branches({"j": ak.Array([[1.0], None])})


def test_extend_accept_new_fields_counter_collides_with_branch_raises(tmp_path):
    path = str(tmp_path / "collides.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"a": np.float64, "nb": np.int32})
        f["tree"].extend({"a": np.arange(2.0), "nb": np.arange(2, dtype=np.int32)})

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="collides"):
            f["tree"].extend(
                {
                    "a": np.array([2.0]),
                    "nb": np.array([1], dtype=np.int32),
                    "b": ak.Array([[1.0]]),
                },
                accept_new_fields=True,
            )


def test_add_jagged_branches_with_indexed_layouts(tmp_path):
    path = str(tmp_path / "indexed.root")
    _flat_tree(path, np.arange(2.0))

    lists = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3])),
        ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0])),
    )
    outer = ak.Array(
        ak.contents.IndexedArray(ak.index.Index64(np.array([1, 0])), lists)
    )
    inner = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 3])),
            ak.contents.IndexedArray(
                ak.index.Index64(np.array([2, 1, 0])),
                ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0])),
            ),
        )
    )
    with uproot.update(path) as f:
        f["tree"].add_branches({"p": outer, "q": inner})

    with uproot.open(path) as f:
        assert f["tree"]["p"].array().tolist() == [[2.0, 3.0], [1.0]]
        assert f["tree"]["q"].array().tolist() == [[3.0], [2.0, 1.0]]


def test_read_other_objects_while_updating(tmp_path):
    path = str(tmp_path / "histogram.root")
    with uproot.recreate(path) as f:
        f["h"] = np.histogram([1.0, 2.0, 2.0], bins=2, range=(0.0, 4.0))

    with uproot.update(path) as f:
        assert f["h"].values().tolist() == [1.0, 2.0]


def test_add_branches_to_tree_with_entries_but_no_baskets_raises(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "no_baskets.root")

    rf = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("tree", "")
    tree.SetEntries(3)
    tree.Write()
    rf.Close()

    with uproot.update(path) as f:
        with pytest.raises(NotImplementedError, match="entries but no baskets"):
            f["tree"].add_branches({"y": np.arange(3)})


def test_extend_existing_jagged_branch_with_all_empty_lists(tmp_path):
    path = str(tmp_path / "all_empty_existing.root")
    _jagged_tree(path, ak.Array([[1.0]]))

    with uproot.update(path) as f:
        f["tree"].extend({"x": [[], [], []]})

    with uproot.open(path) as f:
        assert f["tree"]["x"].array().tolist() == [[1.0], [], [], []]


def test_add_branch_named_like_a_counter_explains_the_clash(tmp_path):
    path = str(tmp_path / "clash.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="counter of jagged branch 'j'"):
            f["tree"].add_branches(
                {"j": ak.Array([[1.0], [2.0, 3.0]]), "nj": np.array([5.0, 6.0])}
            )


def test_existing_counters_are_read_with_one_read_of_the_tree(tmp_path, monkeypatch):
    path = str(tmp_path / "two_counters.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "var * float64", "y": "var * float64"})
        f["tree"].extend({"x": ak.Array([[1.0], []]), "y": ak.Array([[], [2.0, 3.0]])})

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
        assert f["tree"]["p"].array().tolist() == [[4], []]
        assert f["tree"]["q"].array().tolist() == [[], [5, 6]]


def test_extend_accept_new_fields_errors_name_extend(tmp_path):
    path = str(tmp_path / "two_baskets.root")
    _flat_tree(path, np.arange(2.0))
    with uproot.update(path) as f:
        f["tree"].extend({"a": np.array([2.0])})

    with uproot.update(path) as f:
        with pytest.raises(
            NotImplementedError, match="^extend with accept_new_fields=True for a"
        ):
            f["tree"].extend(
                {"a": np.array([3.0]), "b": np.array([1])}, accept_new_fields=True
            )


def test_counter_after_the_branch_it_counts_is_refused(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    path = str(tmp_path / "counter_after.root")

    rf = ROOT.TFile(path, "RECREATE")
    rt = ROOT.TTree("tree", "")
    n = np.zeros(1, dtype=np.int32)
    x = np.zeros(10, dtype=np.float64)
    counter = rt.Branch("n", n, "n/I")
    rt.Branch("x", x, "x[n]/D")
    for row in ([1.0, 2.0], [], [3.0]):
        n[0] = len(row)
        x[: len(row)] = row
        rt.Fill()
    branches = rt.GetListOfBranches()
    branches.Remove(counter)
    branches.Compress()
    branches.Add(counter)
    rt.Write()
    rf.Close()

    with uproot.update(path) as f:
        with pytest.raises(NotImplementedError, match="counters after the branches"):
            f["tree"].add_branches({"y": np.arange(3.0)})

    with uproot.open(path) as f:
        assert f["tree"].keys() == ["x", "n"]
        assert f["tree"]["x"].array().tolist() == [[1.0, 2.0], [], [3.0]]


@pytest.mark.parametrize("rows", [[[7.0]], [[1.0, 2.0], [3.0, 4.0]]])
def test_extend_jagged_branch_with_lists_of_equal_lengths(tmp_path, rows):
    path = str(tmp_path / "equal_lengths.root")
    _jagged_tree(path, ak.Array([[1.0], []]))

    with uproot.update(path) as f:
        f["tree"].extend({"x": rows})

    with uproot.open(path) as f:
        assert f["tree"]["x"].array().tolist() == [[1.0], [], *rows]


def test_extend_jagged_branch_of_fixed_size_lists_with_lists(tmp_path):
    path = str(tmp_path / "fixed_size_lists.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "var * 2 * float64"})
        f["tree"].extend({"x": [[[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]]})

    with uproot.open(path) as f:
        assert f["tree"]["x"].array().tolist() == [
            [[1.0, 2.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]


def test_counter_named_like_another_new_jagged_branch_raises(tmp_path):
    path = str(tmp_path / "jagged_clash.root")
    _flat_tree(path, np.arange(2.0))

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="another new jagged branch"):
            f["tree"].add_branches(
                {"x": ak.Array([[1], [2, 3]]), "nx": ak.Array([[1.0], []])}
            )


def test_extend_accept_new_fields_counter_collision_suggests_add_branches(tmp_path):
    path = str(tmp_path / "hint.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"a": np.float64, "nb": np.int32})
        f["tree"].extend({"a": np.arange(2.0), "nb": np.arange(2, dtype=np.int32)})

    with uproot.update(path) as f:
        with pytest.raises(ValueError, match="add_branches with a counter_name"):
            f["tree"].extend(
                {
                    "a": np.array([2.0]),
                    "nb": np.array([1], dtype=np.int32),
                    "b": ak.Array([[1.0]]),
                },
                accept_new_fields=True,
            )


def test_empty_extend_of_reopened_tree_keeps_counter_maximum(tmp_path):
    path = str(tmp_path / "empty_extend_maximum.root")
    _jagged_tree(path, ak.Array([[1.0, 2.0, 3.0], []]))

    with uproot.update(path) as f:
        f["tree"].extend({"x": ak.Array([[1.0]])[:0]})

    with uproot.open(path) as f:
        assert f["tree"]["nx"].member("fLeaves")[0].member("fMaximum") == 3
        assert f["tree"]["x"].array().tolist() == [[1.0, 2.0, 3.0], []]


def test_extend_jagged_branch_of_fixed_size_lists_with_empty_lists(tmp_path):
    path = str(tmp_path / "fixed_size_empty.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"v": "var * 3 * float64"})
        f["tree"].extend({"v": [[], []]})
        f["tree"].extend({"v": [[[1.0, 2.0, 3.0]], []]})

    with uproot.open(path) as f:
        assert f["tree"]["v"].array().tolist() == [[], [], [[1.0, 2.0, 3.0]], []]


def test_extend_jagged_branch_with_no_lists_is_a_no_op(tmp_path):
    path = str(tmp_path / "no_lists.root")
    _flat_tree(path, np.arange(2.0))
    with uproot.update(path) as f:
        f["tree"].add_branches({"j": [[1.0], [2.0, 3.0]]})
        f["tree"].extend({"a": [], "j": []})
        assert f["tree"].num_entries == 2

    with uproot.open(path) as f:
        assert f["tree"]["j"].array().tolist() == [[1.0], [2.0, 3.0]]


@pytest.mark.parametrize("rows", [[[[1.0, 2.0], [3.0]]], [[1.0, 2.0]]])
def test_extend_jagged_branch_of_fixed_size_lists_checks_items(tmp_path, rows):
    path = str(tmp_path / "wrong_items.root")
    with uproot.recreate(path) as f:
        f.mktree("tree", {"v": "var * 2 * float64"})
        with pytest.raises(ValueError, match="consistent shape"):
            f["tree"].extend({"v": rows})
