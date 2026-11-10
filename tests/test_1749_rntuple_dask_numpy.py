# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest

import uproot

da = pytest.importorskip("dask.array")


@pytest.fixture
def sample_rntuples(tmp_path):
    f1 = os.path.join(tmp_path, "sample1.root")
    f2 = os.path.join(tmp_path, "sample2.root")
    x1 = np.arange(10.0)
    y1 = np.arange(10, dtype=np.int32) * 3
    x2 = np.arange(10.0, 25.0)
    y2 = np.arange(10, 25, dtype=np.int32) * 3

    with uproot.recreate(f1) as file:
        file["ntuple"] = {"x": x1, "y": y1}
    with uproot.recreate(f2) as file:
        file["ntuple"] = {"x": x2, "y": y2}

    return f1, f2, x1, y1, x2, y2


def test_rntuple_dask_numpy_reads(sample_rntuples):
    f1, f2, x1, y1, x2, y2 = sample_rntuples

    # Single-file read
    dask_arrays = uproot.dask(f"{f1}:ntuple", library="np")
    assert set(dask_arrays.keys()) == {"x", "y"}
    computed_x = dask_arrays["x"].compute()
    computed_y = dask_arrays["y"].compute()
    assert np.array_equal(computed_x, x1)
    assert np.array_equal(computed_y, y1)
    assert computed_x.dtype == x1.dtype
    assert computed_y.dtype == y1.dtype

    # Filter name
    filtered = uproot.dask(f"{f1}:ntuple", filter_name="x", library="np")
    assert list(filtered.keys()) == ["x"]
    assert np.array_equal(filtered["x"].compute(), x1)

    # Step size
    stepped = uproot.dask(f"{f1}:ntuple", step_size=3, library="np")
    assert stepped["x"].chunks == ((3, 3, 3, 1),)
    assert np.array_equal(stepped["x"].compute(), x1)

    # Multi-file read
    multi = uproot.dask([f"{f1}:ntuple", f"{f2}:ntuple"], library="np")
    assert set(multi.keys()) == {"x", "y"}
    assert np.array_equal(multi["x"].compute(), np.concatenate([x1, x2]))
    assert np.array_equal(multi["y"].compute(), np.concatenate([y1, y2]))


def test_rntuple_dask_numpy_delayed_open(sample_rntuples):
    f1, f2, x1, _, x2, _ = sample_rntuples

    dask_arrays = uproot.dask(
        [f"{f1}:ntuple", f"{f2}:ntuple"],
        open_files=False,
        library="np",
    )
    assert dask_arrays["x"].chunks == ((np.nan, np.nan),)
    assert np.array_equal(dask_arrays["x"].compute(), np.concatenate([x1, x2]))


def test_rntuple_dask_numpy_jagged_and_strings(tmp_path):
    ak = pytest.importorskip("awkward")
    filepath = os.path.join(tmp_path, "rntuple_complex.root")
    data_jagged = [[1.0, 2.0], [3.0], []]
    strings = ["a", "bc", "def"]

    with uproot.recreate(filepath) as file:
        file.mkrntuple("ntuple", ak.Array({"jagged": data_jagged, "strings": strings}))

    dask_arrays = uproot.dask(f"{filepath}:ntuple", library="np")
    assert dask_arrays["jagged"].dtype == np.dtype(object)
    assert dask_arrays["jagged"].shape == (3,)
    assert dask_arrays["strings"].dtype == np.dtype(object)
    assert dask_arrays["strings"].shape == (3,)

    computed_j = dask_arrays["jagged"].compute()
    assert computed_j.dtype == np.dtype(object)
    assert len(computed_j) == 3
    assert np.array_equal(computed_j[0], np.array([1.0, 2.0]))
    assert np.array_equal(computed_j[1], np.array([3.0]))
    assert len(computed_j[2]) == 0

    computed_s = dask_arrays["strings"].compute()
    assert computed_s.dtype == np.dtype(object)
    assert list(computed_s) == strings


def test_rntuple_dask_numpy_inconsistent_dtype(tmp_path):
    filepath1 = os.path.join(tmp_path, "rntuple_inc1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_inc2.root")

    with uproot.recreate(filepath1) as file:
        file["ntuple"] = {"x": np.arange(5, dtype=np.float64)}
    with uproot.recreate(filepath2) as file:
        file["ntuple"] = {"x": np.arange(5, dtype=np.int32)}

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for field"):
        uproot.dask([f"{filepath1}:ntuple", f"{filepath2}:ntuple"], library="np")


def test_rntuple_dask_numpy_fixed_array_dimension_mismatch(tmp_path):
    ak = pytest.importorskip("awkward")
    filepath1 = os.path.join(tmp_path, "rntuple_fix1.root")
    filepath2 = os.path.join(tmp_path, "rntuple_fix2.root")

    with uproot.recreate(filepath1) as file:
        file.mkrntuple("ntuple", ak.Array({"fixed": np.ones((5, 3), dtype=np.int32)}))
    with uproot.recreate(filepath2) as file:
        file.mkrntuple("ntuple", ak.Array({"fixed": np.ones((5, 4), dtype=np.int32)}))

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for field"):
        uproot.dask([f"{filepath1}:ntuple", f"{filepath2}:ntuple"], library="np")


def test_ttree_dask_numpy_fixed_array_dimension_mismatch(tmp_path):
    filepath1 = os.path.join(tmp_path, "tree_fix1.root")
    filepath2 = os.path.join(tmp_path, "tree_fix2.root")

    with uproot.recreate(filepath1) as file:
        file.mktree("tree", {"arr": "3*int32"})
        file["tree"].extend({"arr": np.ones((5, 3), dtype=np.int32)})
    with uproot.recreate(filepath2) as file:
        file.mktree("tree", {"arr": "4*int32"})
        file["tree"].extend({"arr": np.ones((5, 4), dtype=np.int32)})

    with pytest.raises(ValueError, match="inconsistent NumPy dtype for branch"):
        uproot.dask([f"{filepath1}:tree", f"{filepath2}:tree"], library="np")


def test_rntuple_dask_numpy_jagged_regular_chunk(tmp_path):
    """Jagged field where all lists in a chunk have the same length (regular)."""
    ak = pytest.importorskip("awkward")

    # All-regular file: every entry has exactly 2 elements.
    regular_path = os.path.join(tmp_path, "jagged_regular.root")
    with uproot.recreate(regular_path) as f:
        f.mkrntuple(
            "ntuple",
            ak.Array({"j": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}),
        )

    # Single file, single chunk — must still be 1-D object array, not 2-D.
    a = uproot.dask(f"{regular_path}:ntuple", library="np")["j"]
    assert a.dtype == np.dtype(object)
    result = a.compute()
    assert result.dtype == np.dtype(object)
    assert result.shape == (3,)
    assert [x.tolist() for x in result] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    # Mixed file (irregular + regular) — step_size=1 forces single-entry chunks.
    mixed_path = os.path.join(tmp_path, "jagged_mixed.root")
    with uproot.recreate(mixed_path) as f:
        f.mkrntuple("ntuple", ak.Array({"j": [[1.0, 2.0], [3.0], []]}))

    result2 = uproot.dask(f"{mixed_path}:ntuple", library="np", step_size=1)[
        "j"
    ].compute()
    assert result2.dtype == np.dtype(object)
    assert result2.shape == (3,)
    assert [x.tolist() for x in result2] == [[1.0, 2.0], [3.0], []]

    # Multi-file: one regular, one irregular.
    multi = uproot.dask(
        [f"{regular_path}:ntuple", f"{mixed_path}:ntuple"], library="np"
    )["j"].compute()
    assert multi.dtype == np.dtype(object)
    assert multi.shape == (6,)
    assert [x.tolist() for x in multi] == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [1.0, 2.0],
        [3.0],
        [],
    ]


def test_rntuple_dask_numpy_jagged_regular_chunk_open_files_false(tmp_path):
    """Same as above but with open_files=False (delayed-open path)."""
    ak = pytest.importorskip("awkward")

    regular_path = os.path.join(tmp_path, "jagged_regular.root")
    with uproot.recreate(regular_path) as f:
        f.mkrntuple(
            "ntuple",
            ak.Array({"j": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}),
        )
    mixed_path = os.path.join(tmp_path, "jagged_mixed.root")
    with uproot.recreate(mixed_path) as f:
        f.mkrntuple("ntuple", ak.Array({"j": [[1.0, 2.0], [3.0], []]}))

    multi = uproot.dask(
        [f"{regular_path}:ntuple", f"{mixed_path}:ntuple"],
        open_files=False,
        library="np",
    )["j"].compute()
    assert multi.dtype == np.dtype(object)
    assert multi.shape == (6,)
    assert [x.tolist() for x in multi] == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [1.0, 2.0],
        [3.0],
        [],
    ]


def test_rntuple_dask_numpy_collection_of_structs(tmp_path):
    """RNTuple with a collection of structs (e.g. jets: [{pt, eta}])."""
    ak = pytest.importorskip("awkward")

    filepath = os.path.join(tmp_path, "collection.root")
    with uproot.recreate(filepath) as f:
        f.mkrntuple(
            "ntuple",
            ak.Array(
                {
                    "x": [1.0, 2.0, 3.0],
                    "jets": [
                        [{"pt": 1.0, "eta": 2.0}],
                        [],
                        [{"pt": 3.0, "eta": 4.0}],
                    ],
                }
            ),
        )

    # Default (all fields) — should not crash with AttributeError.
    dask_arrays = uproot.dask(f"{filepath}:ntuple", library="np")
    assert "x" in dask_arrays
    result_x = dask_arrays["x"].compute()
    assert np.array_equal(result_x, [1.0, 2.0, 3.0])

    # filter_name for a subfield of the collection.
    filtered = uproot.dask(f"{filepath}:ntuple", library="np", filter_name="jets.pt")
    assert "jets.pt" in filtered
    result_pt = filtered["jets.pt"].compute()
    assert result_pt.dtype == np.dtype(object)
    assert [x.tolist() for x in result_pt] == [[1.0], [], [3.0]]

    # open_files=False on collection of structs.
    dask_delayed = uproot.dask(f"{filepath}:ntuple", library="np", open_files=False)
    assert "x" in dask_delayed
    assert np.array_equal(dask_delayed["x"].compute(), [1.0, 2.0, 3.0])
    assert "jets.pt" in dask_delayed
    result_pt_delayed = dask_delayed["jets.pt"].compute()
    assert result_pt_delayed.dtype == np.dtype(object)
    assert [x.tolist() for x in result_pt_delayed] == [[1.0], [], [3.0]]


def test_rntuple_dask_numpy_regular_array_of_structs(tmp_path):
    """Subfields of std::array of structs (e.g. std::array<struct{a, b}, N>)."""
    ak = pytest.importorskip("awkward")

    path3 = os.path.join(tmp_path, "array3.root")
    path4 = os.path.join(tmp_path, "array4.root")

    for path, size in [(path3, 3), (path4, 4)]:
        s = ak.to_regular(ak.Array([[{"a": 1, "b": 2.0}] * size] * 5), axis=1)
        with uproot.recreate(path) as f:
            f.mkrntuple("ntuple", ak.Array({"s": s}))

    # Single-file read: declared shape and computed shape must match (5, 3)
    a = uproot.dask(f"{path3}:ntuple", library="np")["s.a"]
    assert a.shape == (5, 3)
    assert a.compute().tolist() == [[1, 1, 1]] * 5
    assert np.array_equal(a[:, 0].compute(), np.ones(5, dtype=np.int64))

    # open_files=False delayed-open read
    a_delayed = uproot.dask(f"{path3}:ntuple", library="np", open_files=False)["s.a"]
    assert np.isnan(a_delayed.shape[0])
    assert a_delayed.shape[1] == 3
    assert a_delayed.compute().tolist() == [[1, 1, 1]] * 5
    assert np.array_equal(a_delayed[:, 0].compute(), np.ones(5, dtype=np.int64))

    # Cross-file read with size mismatch (size 3 vs 4) raises ValueError
    with pytest.raises(ValueError, match="inconsistent NumPy dtype for field"):
        uproot.dask([f"{path3}:ntuple", f"{path4}:ntuple"], library="np")["s.a"]
